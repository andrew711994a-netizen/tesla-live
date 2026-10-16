"""Έρευνα: πιάνει η Capital.com τον στόχο και το stop όπως τα μετρά το backtest;

Στις 15/10/2026 το υψηλό του Yahoo άγγιξε τον στόχο της MRK (152,28) και το εικονικό ημερολόγιο την έκλεισε
εκεί, αλλά στην Capital.com ο στόχος δεν εκτελέστηκε. Η Capital.com κλείνει μια αγορά όταν η τιμή πώλησης (bid)
φτάσει το επίπεδο, και το bid είναι λίγο κάτω από τις τιμές του Yahoo. Επιπλέον εκτελεί stop και στόχους και
εκτός ωραρίου (πριν το άνοιγμα / μετά το κλείσιμο), ενώ τα ημερήσια κεριά του Yahoo έχουν μόνο το κανονικό ωράριο.

1. Μέτρηση (μόνο ανάγνωση τιμών, καμία εντολή): κεριά 30 λεπτών της Capital.com με bid/ask για τις μετοχές του
   bot, σε σύγκριση με τα ημερήσια υψηλά/χαμηλά του Yahoo.
2. Backtest στα τρία τεστ του bot/research_params.py με το παλιό «άγγιγμα» και με τη μετρημένη απόκλιση, και
   δοκιμή αν βοηθά να καταχωρούνται stop και στόχος λίγο πιο χαμηλά στην Capital.com.

Εκτέλεση:  python -m bot.research_fills   (με CAPITAL_API_KEY/IDENTIFIER/PASSWORD για το μέρος 1)
Γράφει το reports/research_fills.md και .json
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import CONFIG, Costs, Risk, every_signal, num, pct, run, stats, trade_stats
from .data import daily, load_all
from .research_params import OTHER, TESTS
from .strategy import Params

ROOT = Path(__file__).resolve().parent.parent
NY = "America/New_York"
FMT = "%Y-%m-%dT%H:%M:%S"


# ───────────── 1. Μέτρηση στην Capital.com ─────────────
def candles(cap, epic: str, days: int = 42) -> pd.DataFrame:
    """Κεριά 30 λεπτών (bid/ask) των τελευταίων ημερών, σε ώρα Νέας Υόρκης, σε κομμάτια των 14 ημερών."""
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
        h, l, c = (p.get(k) or {} for k in ("highPrice", "lowPrice", "closePrice"))
        recs.append({"t": p.get("snapshotTimeUTC") or p.get("snapshotTime"), "hb": h.get("bid"), "lb": l.get("bid"),
                     "cb": c.get("bid"), "ca": c.get("ask")})
    df = pd.DataFrame(recs).dropna() if recs else pd.DataFrame()
    if df.empty:
        return df
    df["t"] = pd.to_datetime(df["t"], utc=True).dt.tz_convert(NY)
    return df.drop_duplicates("t").set_index("t").sort_index().astype(float)


def per_day(c: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    """Ανά μέρα: bid υψηλό/χαμηλό της Capital.com (κανονικό ωράριο και όλη η μέρα) σε σχέση με το Yahoo."""
    m = c.index.hour * 60 + c.index.minute
    c = c.assign(day=c.index.tz_localize(None).normalize(), reg=(m >= 570) & (m < 960))
    out = []
    for d, g in c.groupby("day"):
        if d not in y.index or not g["reg"].any():
            continue
        r = g[g["reg"]]
        H, L = float(y.at[d, "High"]), float(y.at[d, "Low"])
        mid = (r["ca"] + r["cb"]) / 2
        out.append({"day": d, "H": H, "L": L, "hb_reg": r["hb"].max(), "lb_reg": r["lb"].min(),
                    "hb_all": g["hb"].max(), "lb_all": g["lb"].min(),
                    "spread": float(((r["ca"] - r["cb"]) / mid).median())})
    t = pd.DataFrame(out)
    if t.empty:
        return t
    for k in ("reg", "all"):
        t[f"hi_{k}"] = t[f"hb_{k}"] / t["H"] - 1   # αρνητικό: το bid της Capital.com δεν φτάνει το υψηλό του Yahoo
        t[f"lo_{k}"] = t[f"lb_{k}"] / t["L"] - 1   # αρνητικό: το bid πέφτει πιο χαμηλά από το χαμηλό του Yahoo
    return t


def measure() -> dict | None:
    if not all(os.environ.get(k, "").strip() for k in ("CAPITAL_API_KEY", "CAPITAL_IDENTIFIER", "CAPITAL_PASSWORD")):
        return None
    from .capital import Capital

    cap = Capital(os.environ["CAPITAL_API_KEY"], os.environ["CAPITAL_IDENTIFIER"], os.environ["CAPITAL_PASSWORD"],
                  "demo")   # μόνο τιμές: ίδιες στο demo και στο live
    start = (datetime.now(timezone.utc) - timedelta(days=80)).strftime("%Y-%m-%d")
    rows, per_sym = [], {}
    for sym, epic in CONFIG["universe"].items():
        try:
            c, y = candles(cap, epic), daily(sym, start=start)
        except Exception as e:
            print(f"{sym}: {e}")
            continue
        t = per_day(c, y) if not c.empty else pd.DataFrame()
        if t.empty:
            continue
        t["symbol"] = sym
        rows.append(t)
        per_sym[sym] = {"days": len(t), "spread": float(t["spread"].median()),
                        "hi_reg": float(t["hi_reg"].median()), "lo_reg": float(t["lo_reg"].median()),
                        "hi_all": float(t["hi_all"].median()), "lo_all": float(t["lo_all"].median())}
    if not rows:
        return None
    t = pd.concat(rows, ignore_index=True)

    def q(col):
        v = t[col].astype(float)
        return {"median": float(v.median()), "p10": float(v.quantile(0.1)), "p25": float(v.quantile(0.25)),
                "p75": float(v.quantile(0.75)), "p90": float(v.quantile(0.9))}

    res = {"symbols": len(per_sym), "days": int(len(t)), "from": f"{t['day'].min():%d/%m/%Y}",
           "to": f"{t['day'].max():%d/%m/%Y}", "per_symbol": per_sym,
           **{k: q(k) for k in ("spread", "hi_reg", "lo_reg", "hi_all", "lo_all")},
           # στόχος που ήταν μέσα στο υψηλό του Yahoo αλλά όχι στο bid της Capital.com (όλη η μέρα)
           "tp_miss_share": float((t["hb_all"] < t["H"]).mean()),
           # μέρες που το bid της Capital.com (με το εκτός ωραρίου) έπεσε πιο χαμηλά από το χαμηλό του Yahoo
           "lower_low_share": float((t["lb_all"] < t["L"]).mean())}
    mrk = t[(t["symbol"] == "MRK") & (t["day"] == pd.Timestamp("2026-10-15"))]
    if not mrk.empty:
        r = mrk.iloc[0]
        res["mrk_1510"] = {"yahoo_high": float(r["H"]), "bid_high_regular": float(r["hb_reg"]),
                           "bid_high_all": float(r["hb_all"]), "target": 152.28}
    return res


def shifts(m: dict, which: str = "median") -> tuple[float, float]:
    """Από τη μέτρηση στις ρυθμίσεις του backtest: πόσο πιο ψηλά στο Yahoo πρέπει να φτάσει η τιμή για να
    πιαστεί ο στόχος, και πόσο πιο ψηλά στο Yahoo πιάνεται ήδη το stop (με το εκτός ωραρίου)."""
    hi, lo = m["hi_all"][which], m["lo_all"][which]
    return 1 / (1 + hi) - 1, 1 / (1 + lo) - 1


# ───────────── 2. Backtest στα τρία τεστ ─────────────
def main() -> None:
    m = measure()
    path = ROOT / "reports"
    path.mkdir(exist_ok=True)
    if m is None:   # χωρίς σύνδεση: η τελευταία μέτρηση, αν υπάρχει
        old = path / "research_fills.json"
        m = json.loads(old.read_text(encoding="utf-8")).get("measure") if old.exists() else None

    base_p = Params.from_dict(CONFIG["strategy"])
    rk = CONFIG["risk"]
    risk = Risk(rk["risk_per_trade"], rk["max_positions"], rk["max_total_risk"], rk["max_notional"], True)
    costs = Costs(**CONFIG["costs"])
    market_sym = CONFIG["market_symbol"]
    uni_a = list(CONFIG["universe"])
    uni_b = [s for s in OTHER if s not in uni_a]
    raw = {}
    for s in sorted(set(uni_a + uni_b + [market_sym])):
        try:
            raw.update(load_all([s]))
        except Exception as e:
            print(f"χωρίς δεδομένα {s}: {e}")
    market = raw[market_sym]
    universes = {"A": {s: raw[s] for s in uni_a if s in raw},
                 "B": {s: raw[s] for s in uni_b if s in raw and raw[s].index[0] <= pd.Timestamp("2015-01-01")}}

    sp = costs.per_side

    def fp(x: float, d: int = 2) -> str:
        return f"{x:.{d}%}".replace(".", ",")

    variants = [("Παλιό backtest: άγγιγμα στο Yahoo", {}),
                (f"Bid Capital.com, απόκλιση {fp(sp)}", {"tp_shift": sp, "stop_shift": sp}),
                ("Bid Capital.com, απόκλιση 0,15%", {"tp_shift": 0.0015, "stop_shift": 0.0015}),
                ("Bid Capital.com, απόκλιση 0,30%", {"tp_shift": 0.003, "stop_shift": 0.003})]
    if m:
        ts, ss = shifts(m)
        ts75, ss75 = shifts({"hi_all": m["hi_all"], "lo_all": m["lo_all"]}, "p25")   # χειρότερο τέταρτο
        variants += [(f"Μετρημένο (στόχος +{fp(ts)}, stop +{fp(ss)})", {"tp_shift": ts, "stop_shift": ss}),
                     (f"Μετρημένο, χειρότερο τέταρτο (στόχος +{fp(ts75)}, stop +{fp(ss75)})",
                      {"tp_shift": ts75, "stop_shift": ss75})]
        for adj in (0.001, 0.002, 0.004):
            variants.append((f"Μετρημένο + stop/στόχος {fp(adj, 1)} πιο χαμηλά στην Capital.com",
                             {"tp_shift": ts, "stop_shift": ss, "broker_adj": adj}))

    def evaluate(kw: dict) -> dict:
        out = {}
        for label, u, start, end in TESTS:
            res = run(universes[u], base_p, risk, costs, start, end, market, **kw)
            st = stats(res)
            tr = res["trades"]
            st["tp_share"] = float(np.mean([t["reason"] == "Στόχος" for t in tr])) if tr else 0.0
            st["stop_share"] = float(np.mean([t["reason"] == "Stop" for t in tr])) if tr else 0.0
            # κάθε σήμα χωριστά: δεν εξαρτάται από το ποιες θέσεις χώρεσαν, άρα δείχνει καθαρότερα τη διαφορά
            ev = trade_stats(run(universes[u], base_p, every_signal(risk), costs, start, end, market, **kw)["trades"])
            st["every_avg_r"], st["every_trades"] = ev.get("avg_r", 0.0), ev.get("trades", 0)
            out[label] = st
        return out

    summary = {name: evaluate(kw) for name, kw in variants}
    for name, r in summary.items():
        print(name, {k: round(v["sharpe"], 2) for k, v in r.items()})

    out = ["# Έρευνα: στόχος και stop στην Capital.com σε σχέση με το backtest", ""]
    if m:
        out += [f"## 1. Μέτρηση στην Capital.com ({m['symbols']} μετοχές, {m['days']} μετοχο-μέρες, "
                f"{m['from']}–{m['to']})", "",
                "Κεριά 30 λεπτών της Capital.com (τιμή πώλησης, bid) σε σύγκριση με τα ημερήσια υψηλά/χαμηλά του "
                "Yahoo που χρησιμοποιεί το backtest. Αρνητικό = πιο χαμηλά από το Yahoo.", "",
                "| | διάμεσος | 10% | 25% | 75% | 90% |", "|---|---|---|---|---|---|"]
        names = {"spread": "Spread (κανονικό ωράριο)", "hi_reg": "Υψηλό bid, κανονικό ωράριο",
                 "hi_all": "Υψηλό bid, με το εκτός ωραρίου", "lo_reg": "Χαμηλό bid, κανονικό ωράριο",
                 "lo_all": "Χαμηλό bid, με το εκτός ωραρίου"}
        for k, n in names.items():
            v = m[k]
            out.append(f"| {n} | " + " | ".join(f"{v[x] * 100:+.3f}%".replace(".", ",")
                                                 for x in ("median", "p10", "p25", "p75", "p90")) + " |")
        out += ["", f"- Μέρες που το υψηλό bid της Capital.com ήταν κάτω από το υψηλό του Yahoo: "
                    f"**{m['tp_miss_share'] * 100:.0f}%**· τότε ένας στόχος ακριβώς στο υψηλό του Yahoo δεν θα εκτελούνταν.",
                f"- Μέρες που το χαμηλό bid (με το εκτός ωραρίου) ήταν κάτω από το χαμηλό του Yahoo: "
                f"**{m['lower_low_share'] * 100:.0f}%**· τότε ένα stop πιάνεται πιο εύκολα απ' όσο δείχνει το Yahoo."]
        if "mrk_1510" in m:
            x = m["mrk_1510"]
            out.append(f"- MRK 15/10: στόχος {x['target']:.2f} · υψηλό Yahoo {x['yahoo_high']:.2f} · υψηλό bid "
                       f"Capital.com {x['bid_high_regular']:.2f} (κανονικό ωράριο), {x['bid_high_all']:.2f} (όλη η μέρα).")
        wide = sorted(m["per_symbol"].items(), key=lambda kv: -kv[1]["spread"])[:8]
        out += ["", "Μεγαλύτερα spread: " + ", ".join(f"{s} {v['spread'] * 100:.2f}%".replace(".", ",")
                                                    for s, v in wide), ""]
    else:
        out += ["Χωρίς μέτρηση στην Capital.com (λείπουν τα κλειδιά): μόνο οι σταθερές αποκλίσεις.", ""]

    out += ["## 2. Backtest στα τρία τεστ", "",
            "Ίδια στρατηγική και ίδια όρια ρίσκου με το bot. A = οι μετοχές του bot, B = "
            f"{len(universes['B'])} μετοχές που το bot δεν έχει δει ποτέ.", "",
            "Χαρτοφυλάκιο (όπως το bot, έως 8 θέσεις): απόδοση ανά έτος / μέγιστη πτώση / Sharpe. Μικρές αλλαγές "
            "εδώ αλλάζουν ποιες θέσεις χωράνε, οπότε διαφορές ±1% τον χρόνο είναι συχνά τυχαίες. Γι' αυτό δείχνουμε και "
            "το μέσο R **κάθε σήματος χωριστά** (χωρίς όριο θέσεων), που δείχνει καθαρότερα την επίδραση.", "",
            "| Παραλλαγή | " + " | ".join(f"{t}: ανά έτος / πτώση / Sharpe" for t, *_ in TESTS) +
            " | Κάθε σήμα, μέσο R: " + " / ".join(t for t, *_ in TESTS) + " | A 2022–: στόχος / stop |",
            "|---|" + "---|" * (len(TESTS) + 2)]
    for name, r in summary.items():
        a = r["A 2022–σήμερα"]
        out.append(f"| {name} | " + " | ".join(f"{pct(r[t]['cagr'])} / {pct(r[t]['max_dd'])} / {num(r[t]['sharpe'])}"
                                               for t, *_ in TESTS) +
                   " | " + " / ".join(num(r[t]["every_avg_r"], 3) for t, *_ in TESTS) +
                   f" | {a['tp_share'] * 100:.0f}% / {a['stop_share'] * 100:.0f}% |")

    base = summary[variants[0][0]]
    if m:
        real = summary[variants[4][0]]
        d = {t: real[t]["cagr"] - base[t]["cagr"] for t, *_ in TESTS}
        dr = {t: real[t]["every_avg_r"] - base[t]["every_avg_r"] for t, *_ in TESTS}
        out += ["", "## Συμπέρασμα", "",
                "- Με τη μετρημένη συμπεριφορά της Capital.com, σε σχέση με το παλιό backtest, το μέσο R ανά σήμα "
                "αλλάζει κατά " + ", ".join(f"{t}: {v:+.3f}".replace(".", ",") for t, v in dr.items()) +
                " και η απόδοση του χαρτοφυλακίου ανά έτος κατά " + ", ".join(f"{t}: {pct(v)}" for t, v in d.items()) + "."]
        adj = [(n, summary[n]) for n, kw in variants if kw.get("broker_adj")]
        better = [n for n, r in adj if all(r[t]["every_avg_r"] - real[t]["every_avg_r"] >= 0.01
                                           and r[t]["sharpe"] - real[t]["sharpe"] >= 0.05 for t, *_ in TESTS)]
        out.append("- Πιο χαμηλά stop/στόχος στην Capital.com: " +
                   (f"καλύτερο και στα 3 τεστ: {', '.join(better)} (χρειάζεται την έγκρισή σου)." if better else
                    "δεν βελτιώνει σταθερά και στα 3 τεστ, άρα δεν αξίζει να αλλάξουν οι εντολές του bot."))
    out.append("")

    (path / "research_fills.md").write_text("\n".join(out), encoding="utf-8")
    (path / "research_fills.json").write_text(json.dumps({"measure": m, "backtest": summary}, ensure_ascii=False,
                                                         indent=1, default=float), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
