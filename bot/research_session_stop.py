"""Έρευνα: stop μόνο όσο είναι ανοιχτή η αγορά, αντί για stop όλο το 24ωρο;

Η Capital.com εκτελεί stop και στόχους και πριν το άνοιγμα / μετά το κλείσιμο, όπου οι τιμές κάνουν πιο απότομες
κινήσεις (JPM 13/10/2026: stop πριν το άνοιγμα, μετά η μετοχή ανέβηκε). Συγκρίνουμε, για κάθε σήμα της στρατηγικής
(χωρίς όριο θέσεων, ίδιοι κανόνες εισόδου/20 ημερών/σπασμένης τάσης με το bot), τι θα γινόταν αν:
  1. stop και στόχος ισχύουν όλο το 24ωρο (όπως τώρα στην Capital.com),
  2. το stop ισχύει μόνο στο κανονικό ωράριο: αν στο άνοιγμα η τιμή είναι κάτω από αυτό, κλείνει αμέσως,
  3. όπως το 2, και ο στόχος μόνο στο κανονικό ωράριο,
  4–5. όπως το 2, με «stop ασφαλείας» τη νύχτα σε μεγαλύτερη απόσταση (2× ή 1,5× την κανονική).

Δύο πηγές τιμών:
  A. Ωριαία κεριά του Yahoo με τις ώρες εκτός ωραρίου (≈ 2 χρόνια). Εκτός ωραρίου το Yahoo έχει πολλές λανθασμένες
     ακραίες τιμές (καθυστερημένες/μπλοκ συναλλαγές), οπότε εκεί κρατάμε μόνο άνοιγμα και κλείσιμο κάθε κεριού.
     Τρία τεστ: μετοχές του bot στο 1ο και στο 2ο μισό, και 60 μετοχές που δεν έχει δει.
  B. Οι πραγματικές τιμές της Capital.com (κεριά 30 λεπτών, bid/ask) των τελευταίων ~2 μηνών: αυτές ενεργοποιούν τα
     stop. Λίγα σήματα, αλλά ακριβώς ό,τι θα γινόταν. Χρειάζεται CAPITAL_API_KEY/IDENTIFIER/PASSWORD (μόνο ανάγνωση).

Εκτέλεση:  python -m bot.research_session_stop
Γράφει το reports/research_session_stop.md και .json
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import CONFIG, Costs, num
from .data import intraday, load_all
from .research_params import OTHER
from .strategy import Params, market_ok, signals

ROOT = Path(__file__).resolve().parent.parent
FMT = "%Y-%m-%dT%H:%M:%S"

# (όνομα, stop εκτός ωραρίου, στόχος εκτός ωραρίου, stop ασφαλείας εκτός ωραρίου σε πολλαπλάσια της απόστασης)
VARIANTS = [
    ("Τώρα: stop και στόχος όλο το 24ωρο", True, True, None),
    ("Stop μόνο στο ωράριο (στόχος 24ωρο)", False, True, None),
    ("Stop και στόχος μόνο στο ωράριο", False, False, None),
    ("Stop στο ωράριο + stop ασφαλείας τη νύχτα στο 2×", False, True, 2.0),
    ("Stop στο ωράριο + stop ασφαλείας τη νύχτα στο 1,5×", False, True, 1.5),
]


def session(ts: pd.Timestamp) -> str:
    m = ts.hour * 60 + ts.minute
    return "reg" if 570 <= m < 960 else "ext"


def trades_for(sym: str, daily: pd.DataFrame, bars: pd.DataFrame, p: Params, mkt: pd.Series | None,
               costs: Costs, ext_body: bool = False) -> list[dict]:
    """Όλα τα σήματα μιας μετοχής μέσα στο διάστημα των κεριών, με κάθε παραλλαγή εξόδου.

    bars: Open/High/Low/Close (τιμές πώλησης για την Capital.com) και προαιρετικά EntryOpen (τιμή αγοράς)."""
    sg = signals(daily, p)
    days = daily.index
    h_day = bars.index.normalize()
    reg = np.array([session(t) == "reg" for t in bars.index])
    if not reg.any():
        return []
    first_day = h_day[reg].min()
    entry_open = bars["EntryOpen"] if "EntryOpen" in bars else bars["Open"]
    out = []
    for i in range(len(days) - 1):
        d, e = days[i], days[i + 1]                      # σήμα στο κλείσιμο της d, είσοδος στο άνοιγμα της e
        if d < first_day or not bool(sg.at[d, "long_sig"]):
            continue
        if mkt is not None and not bool(mkt.get(d, False)):
            continue
        rd = np.where((h_day == d) & reg)[0]
        re_ = np.where((h_day == e) & reg)[0]
        if not len(rd) or not len(re_):
            continue
        # τα ημερήσια του Yahoo είναι διορθωμένα για μερίσματα, τα κεριά όχι: επίπεδα στην κλίμακα των κεριών
        f = bars["Close"].iloc[rd[-1]] / daily.at[d, "Close"]
        stop, tp = sg.at[d, "stop"] * f, sg.at[d, "tp"] * f
        k0 = re_[0]
        o = float(entry_open.iloc[k0])
        if not (stop < o < tp):
            continue
        dist = o - stop
        pending_day, bars_n = None, 0                    # έξοδος χρόνου/τάσης, όπως το backtest
        for j in range(i + 1, len(days) - 1):
            dd = days[j]
            if dd != e:
                bars_n += 1
            if bars_n >= p.max_bars or (p.trend_exit and bool(sg.at[dd, "trend_broken"])):
                pending_day = days[j + 1]
                break
        res = {name: simulate(bars, reg, h_day, k0, o, stop, tp, dist, se, te, sf, pending_day, costs, ext_body)
               for name, se, te, sf in VARIANTS}
        out.append({"symbol": sym, "entry_date": e, "entry": o, "stop": stop, "tp": tp, "res": res})
    return out


def simulate(bars, reg, h_day, k0, entry, stop, tp, dist, stop_ext, tp_ext, safety, pending_day, costs,
             ext_body: bool = False):
    """Περπατά τα κεριά από την είσοδο ως την έξοδο. ext_body: εκτός ωραρίου μόνο άνοιγμα/κλείσιμο του κεριού."""
    O, H, L, C = (bars[c].values for c in ("Open", "High", "Low", "Close"))
    idx = bars.index
    safe = entry - safety * dist if safety else None
    exit_px, why, when, at = None, "Τέλος", None, None
    for k in range(k0, len(O)):
        is_reg, day = reg[k], h_day[k]
        hi, lo = H[k], L[k]
        if ext_body and not is_reg:
            hi, lo = max(O[k], C[k]), min(O[k], C[k])
        if k == k0:                      # κερί εισόδου (αγορά στο άνοιγμά του): μόνο ό,τι γίνει μέσα σε αυτό
            if lo <= stop:
                exit_px, why, when, at = stop, "Stop", day, idx[k]
                break
            if hi >= tp:
                exit_px, why, when, at = tp, "Στόχος", day, idx[k]
                break
            continue
        opening = is_reg and (not reg[k - 1] or h_day[k - 1] != day)   # πρώτο κερί του ωραρίου της μέρας
        if pending_day is not None and opening and day >= pending_day:
            exit_px, why, when, at = O[k], "Χρόνος/Τάση", day, idx[k]
            break
        s_lvl = stop if (is_reg or stop_ext) else safe
        t_lvl = tp if (is_reg or tp_ext) else None
        if s_lvl is not None:
            name = "Stop" if s_lvl == stop else "Stop ασφαλείας"
            if O[k] <= s_lvl:            # άνοιξε ήδη κάτω (π.χ. στις 9:31 το bot κλείνει στην αγορά)
                exit_px, why, when, at = O[k], name, day, idx[k]
                break
            if lo <= s_lvl:
                exit_px, why, when, at = s_lvl, name, day, idx[k]
                break
        if t_lvl is not None:
            if O[k] >= t_lvl:
                exit_px, why, when, at = O[k], "Στόχος", day, idx[k]
                break
            if hi >= t_lvl:
                exit_px, why, when, at = t_lvl, "Στόχος", day, idx[k]
                break
    if exit_px is None:
        exit_px, when = C[-1], h_day[-1]
    pnl = (exit_px - entry) - (entry + exit_px) * costs.per_side
    return {"r": pnl / dist, "pct": pnl / entry, "reason": why, "exit_date": when, "at": at}


def summarize(trades: list[dict], name: str, include_open: bool = False) -> dict:
    done = [t["res"][name] for t in trades if include_open or t["res"][name]["reason"] != "Τέλος"]
    if not done:
        return {"trades": 0}
    r = np.array([x["r"] for x in done])
    return {"trades": len(done), "avg_r": float(r.mean()), "win": float((r > 0).mean()),
            "total_r": float(r.sum()), "worst_r": float(r.min()), "below_-1.5R": int((r < -1.5).sum()),
            "stop_share": float(np.mean([x["reason"].startswith("Stop") for x in done])),
            "tp_share": float(np.mean([x["reason"] == "Στόχος" for x in done])),
            "open": int(sum(x["reason"] == "Τέλος" for x in done))}


def table(out: list[str], title: str, groups: dict, include_open: bool = False) -> dict:
    summary = {tn: {name: summarize(tt, name, include_open) for name, *_ in VARIANTS} for tn, tt in groups.items()}
    for tn, tt in groups.items():
        out += [f"### {title}{tn} ({len(tt)} σήματα)", "",
                "| Παραλλαγή | Συναλλαγές | Επιτυχία | Μέσο R | Σύνολο R | Stop / Στόχος | Χειρότερη | Κάτω από −1,5 R |",
                "|---|---|---|---|---|---|---|---|"]
        for name, *_ in VARIANTS:
            st = summary[tn][name]
            if not st.get("trades"):
                out.append(f"| {name} | 0 | – | – | – | – | – | – |")
                continue
            out.append(f"| {name} | {st['trades']} | {st['win'] * 100:.0f}% | {num(st['avg_r'], 3)} | "
                       f"{num(st['total_r'], 1)} | {st['stop_share'] * 100:.0f}% / {st['tp_share'] * 100:.0f}% | "
                       f"{num(st['worst_r'])} | {st['below_-1.5R']} |")
        out.append("")
    return summary


def compare(out: list[str], summary: dict) -> list[str]:
    base = VARIANTS[0][0]
    tests = list(summary)

    def verdict(name: str) -> str:
        rows = [summary[tn] for tn in tests if summary[tn][name].get("trades") and summary[tn][base].get("trades")]
        d = [r[name]["avg_r"] - r[base]["avg_r"] for r in rows]
        tail = all(r[name]["below_-1.5R"] <= r[base]["below_-1.5R"] + max(2, 0.01 * r[name]["trades"]) for r in rows)
        if len(d) == len(tests) and all(x >= 0.02 for x in d):
            return "✅ καλύτερο σε όλα" + ("" if tail else " (αλλά περισσότερες μεγάλες ζημιές)")
        if len(d) == len(tests) and all(x <= -0.02 for x in d):
            return "❌ χειρότερο σε όλα"
        return "➖ μικτό"

    out += ["| Παραλλαγή | " + " | ".join(tests) + " | Ετυμηγορία |", "|---|" + "---|" * (len(tests) + 1)]
    good = []
    for name, *_ in VARIANTS[1:]:
        cells = [f"{summary[tn][name].get('avg_r', 0) - summary[tn][base].get('avg_r', 0):+.3f}".replace(".", ",")
                 for tn in tests]
        v = verdict(name)
        good += [name] if v.startswith("✅") else []
        out.append(f"| {name} | " + " | ".join(cells) + f" | {v} |")
    out.append("")
    return good


def examples(out: list[str], trades: list[dict], n: int = 12) -> None:
    base, alt = VARIANTS[0][0], VARIANTS[1][0]
    diff = [t for t in trades if t["res"][base]["reason"] != t["res"][alt]["reason"]
            or abs(t["res"][base]["r"] - t["res"][alt]["r"]) > 0.05]
    if not diff:
        out += ["Κανένα σήμα δεν άλλαξε αποτέλεσμα.", ""]
        return
    out += ["| Μετοχή | Είσοδος | Τώρα (24ωρο) | Πότε | Stop μόνο στο ωράριο |", "|---|---|---|---|---|"]
    for t in sorted(diff, key=lambda t: t["entry_date"])[-n:]:
        a, b = t["res"][base], t["res"][alt]
        at = f"{a['at']:%d/%m %H:%M}" + (" (εκτός ωραρίου)" if a["at"] is not None and session(a["at"]) == "ext" else "") \
            if a["at"] is not None else "–"
        out.append(f"| {t['symbol']} | {t['entry_date']:%d/%m/%Y} | {a['reason']} {num(a['r'])} R | {at} | "
                   f"{b['reason']} {num(b['r'])} R ({b['exit_date']:%d/%m}) |")
    out.append("")


# ───────────── Τιμές της Capital.com ─────────────
def capital_bars(cap, epic: str, days: int = 63) -> pd.DataFrame:
    """Κεριά 30 λεπτών: Open/High/Low/Close = τιμή πώλησης (bid), EntryOpen = τιμή αγοράς (ask) στο άνοιγμα."""
    from .capital import CapitalError

    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    rows = []
    for k in range(days, 0, -14):
        a, b = now - timedelta(days=k), min(now, now - timedelta(days=k - 14))
        try:
            rows += cap.prices(epic, "MINUTE_30", a.strftime(FMT), b.strftime(FMT), 1000)
        except CapitalError as e:
            print(f"{epic}: {e}")
    recs = []
    for p in rows:
        o, h, l, c = (p.get(k) or {} for k in ("openPrice", "highPrice", "lowPrice", "closePrice"))
        recs.append({"t": p.get("snapshotTimeUTC") or p.get("snapshotTime"), "Open": o.get("bid"), "High": h.get("bid"),
                     "Low": l.get("bid"), "Close": c.get("bid"), "EntryOpen": o.get("ask")})
    df = pd.DataFrame(recs).dropna() if recs else pd.DataFrame()
    if df.empty:
        return df
    df["t"] = pd.to_datetime(df["t"], utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
    return df.drop_duplicates("t").set_index("t").sort_index().astype(float)


def capital_part(daily: dict, p: Params, mkt) -> tuple[list[dict], dict] | None:
    if not all(os.environ.get(k, "").strip() for k in ("CAPITAL_API_KEY", "CAPITAL_IDENTIFIER", "CAPITAL_PASSWORD")):
        return None
    from .capital import Capital

    cap = Capital(os.environ["CAPITAL_API_KEY"], os.environ["CAPITAL_IDENTIFIER"], os.environ["CAPITAL_PASSWORD"],
                  "demo")   # μόνο τιμές
    trades, ext = [], {}
    no_cost = Costs(per_side=0.0, financing_annual=0.0)   # η διαφορά bid/ask είναι ήδη μέσα στις τιμές
    for sym, epic in CONFIG["universe"].items():
        if sym not in daily:
            continue
        try:
            b = capital_bars(cap, epic)
        except Exception as e:
            print(f"Capital.com {sym}: {e}")
            continue
        if b.empty:
            continue
        is_ext = np.array([session(t) == "ext" for t in b.index])
        ext[sym] = {"from": f"{b.index[is_ext].min():%d/%m}" if is_ext.any() else None, "ext_share": float(is_ext.mean())}
        trades += trades_for(sym, daily[sym], b, p, mkt, no_cost)
    return trades, ext


def main() -> None:
    p = Params.from_dict(CONFIG["strategy"])
    costs = Costs(**CONFIG["costs"])
    market_sym = CONFIG["market_symbol"]
    uni_a = list(CONFIG["universe"])
    uni_b = [s for s in OTHER if s not in uni_a]
    daily = {}
    for s in sorted(set(uni_a + uni_b + [market_sym])):
        try:
            daily.update(load_all([s]))
        except Exception as e:
            print(f"χωρίς ημερήσια {s}: {e}")
    mkt = market_ok(daily[market_sym], p) if CONFIG["risk"].get("use_market_filter", True) else None

    out = ["# Έρευνα: stop μόνο στο κανονικό ωράριο ή όλο το 24ωρο", "",
           "Κάθε σήμα της στρατηγικής σαν να αγοράστηκε (χωρίς όριο θέσεων), με τους κανόνες του bot. "
           "R = αποτέλεσμα σε πολλαπλάσια του αρχικού ρίσκου (−1 = έπιασε το stop). «Χειρότερη» και «κάτω από −1,5 R» "
           "δείχνουν πόσο χειρότερα από το stop βγαίνει μια θέση όταν η τιμή πέσει απότομα ενώ το stop δεν ισχύει.", ""]
    summary = {}

    # ── B. Πραγματικές τιμές της Capital.com ──
    cp = capital_part(daily, p, mkt)
    if cp:
        ct, ext = cp
        with_ext = sorted(s for s, v in ext.items() if v["ext_share"] > 0.05)
        out += ["## 1. Πραγματικές τιμές της Capital.com (κεριά 30 λεπτών, τιμή πώλησης)", "",
                f"Μετοχές με τιμές εκτός ωραρίου στην Capital.com: {len(with_ext)}/{len(ext)}"
                + (f" ({', '.join(with_ext)})" if with_ext else "") + ". "
                "Είσοδος στην τιμή αγοράς (ask), έξοδος στην τιμή πώλησης (bid), χωρίς άλλο κόστος. "
                "Μετρούν και οι θέσεις που είναι ακόμα ανοιχτές, με την τελευταία τιμή.", ""]
        summary["Capital.com"] = table(out, "", {"Capital.com": ct}, include_open=True)["Capital.com"]
        out += ["Διαφορά μέσου R από το σημερινό:", ""]
        compare(out, {"Capital.com": summary["Capital.com"]})
        out += ["Σήματα όπου άλλαξε το αποτέλεσμα:", ""]
        examples(out, ct, 20)
    else:
        out += ["## 1. Πραγματικές τιμές της Capital.com", "", "Δεν έγινε (λείπουν τα κλειδιά).", ""]

    # ── A. Yahoo, ~2 χρόνια ──
    trades, missing = {"A": [], "B": []}, []
    for u, syms in (("A", uni_a), ("B", uni_b)):
        for s in syms:
            if s not in daily:
                continue
            try:
                h = intraday(s, "1h", "730d", prepost=True)
            except Exception as e:
                missing.append(s)
                print(f"χωρίς ωριαία {s}: {e}")
                continue
            trades[u] += trades_for(s, daily[s], h, p, mkt, costs, ext_body=True)
    all_a = sorted(trades["A"], key=lambda t: t["entry_date"])
    if all_a:
        mid = all_a[len(all_a) // 2]["entry_date"]
        tests = {f"A ως {mid:%m/%Y}": [t for t in all_a if t["entry_date"] < mid],
                 f"A από {mid:%m/%Y}": [t for t in all_a if t["entry_date"] >= mid],
                 "B (60 άγνωστες)": trades["B"]}
        first, last = all_a[0]["entry_date"], all_a[-1]["entry_date"]
        out += ["## 2. Ωριαία κεριά του Yahoo (≈ 2 χρόνια)", "",
                f"Είσοδοι {first:%d/%m/%Y}–{last:%d/%m/%Y}, κόστος {costs.per_side:.2%} ανά πράξη. Εκτός ωραρίου μόνο "
                "άνοιγμα και κλείσιμο κάθε ώρας: οι ακραίες τιμές του Yahoo εκεί είναι συχνά λανθασμένες συναλλαγές "
                "(με αυτές το 24ωρο stop έβγαινε να πιάνεται στο ~65% των συναλλαγών, ακόμα και την ίδια μέρα της "
                "αγοράς χωρίς να έχει πέσει η τιμή). Το Yahoo δεν είναι οι τιμές της Capital.com, οπότε αυτό το μέρος "
                "δείχνει την τάση, όχι το ακριβές νούμερο." + (f" Χωρίς ωριαία: {', '.join(missing)}." if missing else ""), ""]
        ys = table(out, "", tests)
        summary.update({f"Yahoo {k}": v for k, v in ys.items()})
        out += ["Διαφορά μέσου R από το σημερινό (καλύτερο = +0,02 R ή περισσότερο και στα 3):", ""]
        good = compare(out, ys)
        out += ["Πρόσφατα σήματα όπου άλλαξε το αποτέλεσμα (μετοχές του bot):", ""]
        examples(out, all_a)
        summary["_yahoo_better_everywhere"] = good

    out += ["Σημείωση: εκτός ωραρίου το spread της Capital.com είναι μεγαλύτερο από το κανονικό.", ""]
    path = ROOT / "reports"
    path.mkdir(exist_ok=True)
    (path / "research_session_stop.md").write_text("\n".join(out), encoding="utf-8")
    (path / "research_session_stop.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str),
                                                     encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
