"""Έρευνα: stop μόνο όσο είναι ανοιχτή η αγορά, αντί για stop όλο το 24ωρο;

Η Capital.com εκτελεί stop και στόχους και πριν το άνοιγμα / μετά το κλείσιμο, όπου οι τιμές κάνουν πιο απότομες
κινήσεις (JPM 13/10/2026: stop πριν το άνοιγμα, μετά η μετοχή ανέβηκε). Εδώ συγκρίνουμε, με ωριαία κεριά του Yahoo
που έχουν και τις ώρες εκτός ωραρίου (≈ 2 χρόνια), τι θα γινόταν με κάθε σήμα της στρατηγικής αν:
  1. stop και στόχος ισχύουν όλο το 24ωρο (όπως τώρα στην Capital.com),
  2. το stop ισχύει μόνο στο κανονικό ωράριο: αν στο άνοιγμα η τιμή είναι κάτω από αυτό, κλείνει αμέσως,
  3. όπως το 2, και ο στόχος μόνο στο κανονικό ωράριο,
  4–5. όπως το 2, με «stop ασφαλείας» τη νύχτα σε μεγαλύτερη απόσταση (2× ή 1,5× την κανονική).
Σήματα, είσοδος, 20 μέρες και σπασμένη τάση: ακριβώς όπως το backtest του bot (ημερήσια κεριά, φίλτρο αγοράς).
Κρίνεται σε τρία τεστ: μετοχές του bot στον 1ο χρόνο, στον 2ο χρόνο, και 60 μετοχές που δεν έχει δει.

Εκτέλεση:  python -m bot.research_session_stop
Γράφει το reports/research_session_stop.md και .json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import CONFIG, Costs, num
from .data import intraday, load_all
from .research_params import OTHER
from .strategy import Params, market_ok, signals

ROOT = Path(__file__).resolve().parent.parent

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


def trades_for(sym: str, daily: pd.DataFrame, hourly: pd.DataFrame, p: Params, mkt: pd.Series | None,
               costs: Costs) -> list[dict]:
    """Όλα τα σήματα μιας μετοχής μέσα στο διάστημα των ωριαίων κεριών, με κάθε παραλλαγή εξόδου."""
    sg = signals(daily, p)
    days = daily.index
    h_day = hourly.index.normalize()
    reg = np.array([session(t) == "reg" for t in hourly.index])
    first_day = h_day[reg].min() if reg.any() else None
    out = []
    if first_day is None:
        return out
    for i in range(len(days) - 1):
        d, e = days[i], days[i + 1]                      # σήμα στο κλείσιμο της d, είσοδος στο άνοιγμα της e
        if d < first_day or not bool(sg.at[d, "long_sig"]):
            continue
        if mkt is not None and not bool(mkt.get(d, False)):
            continue
        # Το Yahoo διορθώνει τα ημερήσια για μερίσματα αλλά όχι τα ωριαία: μετατροπή των επιπέδων στην κλίμακα των ωριαίων
        rd = np.where((h_day == d) & reg)[0]
        re_ = np.where((h_day == e) & reg)[0]
        if not len(rd) or not len(re_):
            continue
        f = hourly["Close"].iloc[rd[-1]] / daily.at[d, "Close"]
        stop, tp = sg.at[d, "stop"] * f, sg.at[d, "tp"] * f
        k0 = re_[0]
        o = hourly["Open"].iloc[k0]
        if not (stop < o < tp):
            continue
        dist = o - stop
        # μέρες εξόδου λόγω χρόνου/τάσης (όπως το backtest: κλείσιμο στο επόμενο άνοιγμα)
        pending_day = None
        bars = 0
        for j in range(i + 1, len(days) - 1):
            dd = days[j]
            if dd != e:
                bars += 1
            if bars >= p.max_bars or (p.trend_exit and bool(sg.at[dd, "trend_broken"])):
                pending_day = days[j + 1]
                break
        res = {}
        for name, stop_ext, tp_ext, safety in VARIANTS:
            res[name] = simulate(hourly, reg, h_day, k0, o, stop, tp, dist, stop_ext, tp_ext, safety, pending_day, costs)
        out.append({"symbol": sym, "entry_date": e, "entry": o, "stop": stop, "tp": tp, "res": res})
    return out


def simulate(hourly, reg, h_day, k0, entry, stop, tp, dist, stop_ext, tp_ext, safety, pending_day, costs):
    """Περπατά τα ωριαία κεριά από την είσοδο ως την έξοδο. Επιστρέφει R, %, λόγο και ημερομηνία εξόδου."""
    O, H, L = hourly["Open"].values, hourly["High"].values, hourly["Low"].values
    safe = entry - safety * dist if safety else None
    exit_px, why, when = None, "Τέλος", None
    for k in range(k0, len(O)):
        is_reg, day = reg[k], h_day[k]
        if k == k0:                      # κερί εισόδου (αγορά στο άνοιγμά του): μόνο ό,τι γίνει μέσα σε αυτό
            if L[k] <= stop:
                exit_px, why, when = stop, "Stop", day
                break
            if H[k] >= tp:
                exit_px, why, when = tp, "Στόχος", day
                break
            continue
        opening = is_reg and (not reg[k - 1] or h_day[k - 1] != day)   # πρώτο κερί του ωραρίου της μέρας
        if pending_day is not None and opening and day >= pending_day:
            exit_px, why, when = O[k], "Χρόνος/Τάση", day
            break
        s_lvl = stop if (is_reg or stop_ext) else safe
        t_lvl = tp if (is_reg or tp_ext) else None
        if s_lvl is not None:
            name = "Stop" if s_lvl == stop else "Stop ασφαλείας"
            if O[k] <= s_lvl:            # άνοιξε ήδη κάτω (π.χ. στις 9:31 το bot κλείνει στην αγορά)
                exit_px, why, when = O[k], name, day
                break
            if L[k] <= s_lvl:
                exit_px, why, when = s_lvl, name, day
                break
        if t_lvl is not None:
            if O[k] >= t_lvl:
                exit_px, why, when = O[k], "Στόχος", day
                break
            if H[k] >= t_lvl:
                exit_px, why, when = t_lvl, "Στόχος", day
                break
    if exit_px is None:
        exit_px, when = hourly["Close"].values[-1], h_day[-1]
    pnl = (exit_px - entry) - (entry + exit_px) * costs.per_side
    return {"r": pnl / dist, "pct": pnl / entry, "reason": why, "exit_date": when}


def summarize(trades: list[dict], name: str) -> dict:
    done = [t["res"][name] for t in trades if t["res"][name]["reason"] != "Τέλος"]
    if not done:
        return {"trades": 0}
    r = np.array([x["r"] for x in done])
    return {"trades": len(done), "avg_r": float(r.mean()), "win": float((r > 0).mean()),
            "total_r": float(r.sum()), "worst_r": float(r.min()), "below_-1.5R": int((r < -1.5).sum()),
            "stop_share": float(np.mean([x["reason"].startswith("Stop") for x in done])),
            "tp_share": float(np.mean([x["reason"] == "Στόχος" for x in done]))}


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
            trades[u] += trades_for(s, daily[s], h, p, mkt, costs)
    all_a = sorted(trades["A"], key=lambda t: t["entry_date"])
    if not all_a:
        raise SystemExit("Καμία συναλλαγή: λείπουν δεδομένα.")
    mid = all_a[len(all_a) // 2]["entry_date"]
    tests = {f"A ως {mid:%m/%Y}": [t for t in all_a if t["entry_date"] < mid],
             f"A από {mid:%m/%Y}": [t for t in all_a if t["entry_date"] >= mid],
             "B (60 άγνωστες)": trades["B"]}
    summary = {tn: {name: summarize(tt, name) for name, *_ in VARIANTS} for tn, tt in tests.items()}

    first = min(t["entry_date"] for t in all_a)
    last = max(t["entry_date"] for t in all_a)
    out = ["# Έρευνα: stop μόνο στο κανονικό ωράριο ή όλο το 24ωρο", "",
           f"Ωριαία κεριά του Yahoo με τις ώρες πριν το άνοιγμα (από 4:00) και μετά το κλείσιμο (ως 20:00 Νέα Υόρκη). "
           f"Είσοδοι {first:%d/%m/%Y}–{last:%d/%m/%Y}. Κάθε σήμα σαν να αγοράστηκε (χωρίς όριο θέσεων), "
           f"ίδιοι κανόνες με το bot, κόστος {costs.per_side:.2%} ανά πράξη." +
           (f" Χωρίς ωριαία δεδομένα: {', '.join(missing)}." if missing else ""), "",
           "R = αποτέλεσμα σε πολλαπλάσια του αρχικού ρίσκου (−1 = έπιασε το stop). "
           "«Χειρότερη» και «κάτω από −1,5 R» δείχνουν πόσο χειρότερα από το stop μπορεί να βγει μια θέση "
           "όταν η τιμή πέσει απότομα ενώ το stop δεν ισχύει.", ""]
    for tn, tt in tests.items():
        out += [f"## {tn} ({len(tt)} σήματα)", "",
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

    base = VARIANTS[0][0]

    def verdict(name: str) -> str:
        d = [summary[tn][name]["avg_r"] - summary[tn][base]["avg_r"] for tn in tests if summary[tn][name].get("trades")]
        tail = all(summary[tn][name]["below_-1.5R"] <= summary[tn][base]["below_-1.5R"] + max(2, 0.01 * summary[tn][name]["trades"])
                   for tn in tests if summary[tn][name].get("trades"))
        if len(d) == len(tests) and all(x >= 0.02 for x in d):
            return "✅ καλύτερο και στα 3" + ("" if tail else " (αλλά περισσότερες μεγάλες ζημιές)")
        if len(d) == len(tests) and all(x <= -0.02 for x in d):
            return "❌ χειρότερο και στα 3"
        return "➖ μικτό"

    out += ["## Σύγκριση με το σημερινό (μέσο R)", "",
            "| Παραλλαγή | " + " | ".join(tests) + " | Ετυμηγορία |", "|---|" + "---|" * (len(tests) + 1)]
    for name, *_ in VARIANTS[1:]:
        cells = [f"{summary[tn][name].get('avg_r', 0) - summary[tn][base].get('avg_r', 0):+.3f}".replace(".", ",")
                 for tn in tests]
        out.append(f"| {name} | " + " | ".join(cells) + f" | {verdict(name)} |")

    # Παραδείγματα: πρόσφατες συναλλαγές όπου το stop εκτός ωραρίου άλλαξε το αποτέλεσμα
    diff = [t for t in all_a if t["res"][base]["reason"] != t["res"][VARIANTS[1][0]]["reason"]
            or abs(t["res"][base]["r"] - t["res"][VARIANTS[1][0]]["r"]) > 0.05]
    if diff:
        out += ["", "## Πρόσφατα παραδείγματα όπου άλλαξε το αποτέλεσμα (μετοχές του bot)", "",
                "| Μετοχή | Είσοδος | Τώρα (24ωρο) | Stop μόνο στο ωράριο |", "|---|---|---|---|"]
        for t in diff[-12:]:
            a, b = t["res"][base], t["res"][VARIANTS[1][0]]
            out.append(f"| {t['symbol']} | {t['entry_date']:%d/%m/%Y} | {a['reason']} {num(a['r'])} R ({a['exit_date']:%d/%m}) | "
                       f"{b['reason']} {num(b['r'])} R ({b['exit_date']:%d/%m}) |")
    out += ["", "Σημείωση: εκτός ωραρίου το spread της Capital.com είναι μεγαλύτερο απ' ό,τι εδώ, οπότε στην πράξη "
                "το stop όλο το 24ωρο είναι λίγο χειρότερο απ' όσο δείχνει ο πίνακας.", ""]

    path = ROOT / "reports"
    path.mkdir(exist_ok=True)
    (path / "research_session_stop.md").write_text("\n".join(out), encoding="utf-8")
    (path / "research_session_stop.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=float),
                                                     encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
