"""Μπορεί η τωρινή στρατηγική να βγάζει περισσότερα χωρίς περισσότερο ρίσκο;

Ίδιος αυστηρός έλεγχος με το research.py: 2016–2021 για «εκπαίδευση», 2022–σήμερα ως «άγνωστα» χρόνια.
Καμία παραλλαγή δεν αλλάζει τους κανόνες εισόδου/εξόδου· αλλάζουν μόνο τα όρια χαρτοφυλακίου και το τι
κάνουν τα αδιάθετα χρήματα.
Εκτέλεση:  python -m bot.research_plus   →   reports/research_plus.md
"""

from __future__ import annotations

import json
import math

import pandas as pd

from .backtest import CONFIG, PERIODS, ROOT, Costs, Risk, buy_hold, num, pct, run, stats
from .research import run_index_timing, run_rotation
from .strategy import Params


def _stats_curve(eq: pd.Series, initial: float, trades: list | None = None) -> dict:
    st = stats({"equity": eq, "trades": trades or [], "initial": initial, "exposure": 1.0})
    if not trades:
        st.update({"trades": 0, "win": float("nan"), "pf": float("nan")})
    return st


def idle_overlay(res: dict, sleeve: pd.Series, costs: Costs) -> pd.Series:
    """Τα αδιάθετα χρήματα της στρατηγικής πάνε σε ένα «κουβά» (π.χ. Nasdaq-100 με φίλτρο) μέχρι να χρειαστούν.

    sleeve: η καμπύλη αξίας του κουβά (ήδη με το δικό της φίλτρο και κόστος).
    """
    eq, cash = res["equity"], res["cash"].clip(lower=0.0)
    idle = (cash / eq).clip(0.0, 1.0)                                  # μερίδιο αδιάθετων στο κλείσιμο
    r_sleeve = sleeve.reindex(eq.index).ffill().pct_change().fillna(0.0)
    w = idle.shift(1).fillna(0.0)                                       # ισχύει από την επόμενη μέρα
    turnover = idle.diff().abs().shift(1).fillna(0.0)                   # αγορές/πωλήσεις για να βρεθούν μετρητά
    net = eq.pct_change().fillna(0.0) + w * r_sleeve - turnover * costs.per_side
    return res["initial"] * (1 + net).cumprod()


def blend(eq_a: pd.Series, eq_b: pd.Series, w_a: float, initial: float) -> pd.Series:
    """Δύο «κουβάδες» στον ίδιο λογαριασμό, με επανεξισορρόπηση στο τέλος κάθε μήνα."""
    ra = eq_a.pct_change().fillna(0.0)
    rb = eq_b.reindex(eq_a.index).ffill().pct_change().fillna(0.0)
    va, vb = initial * w_a, initial * (1 - w_a)
    months = eq_a.index.to_period("M")
    out = []
    for i in range(len(eq_a)):
        va *= 1 + ra.iloc[i]
        vb *= 1 + rb.iloc[i]
        out.append(va + vb)
        if i + 1 < len(eq_a) and months[i + 1] != months[i]:
            va, vb = out[-1] * w_a, out[-1] * (1 - w_a)
    return pd.Series(out, index=eq_a.index)


HEAD = ("| Παραλλαγή | Συνολικό | Ανά έτος | Μέγ. πτώση | Sharpe | Ανά έτος / πτώση | Συναλλαγές | Profit factor |\n"
        "|---|---|---|---|---|---|---|---|")


def line(name: str, st: dict) -> str:
    ratio = st["cagr"] / abs(st["max_dd"]) if st["max_dd"] < 0 else float("inf")
    if not st["trades"] or (isinstance(st["pf"], float) and math.isnan(st["pf"])):
        n, pf = "–", "–"
    else:
        n, pf = st["trades"], num(st["pf"])
    return (f"| {name} | {pct(st['total'])} | {pct(st['cagr'])} | {pct(st['max_dd'])} | {num(st['sharpe'])} | "
            f"{num(ratio)} | {n} | {pf} |")


def main() -> None:
    from .data import load_all

    uni = list(CONFIG["universe"])
    market_sym = CONFIG["market_symbol"]
    data = load_all(sorted(set(uni + [market_sym])))
    market = data[market_sym]
    stocks = {s: data[s] for s in uni if s in data}
    costs = Costs(**CONFIG["costs"])
    rk = CONFIG["risk"]
    p = Params.from_dict(CONFIG["strategy"])
    initial = 10_000.0
    money = f"{initial:,.0f}".replace(",", ".")

    def risk(n: int, r: float) -> Risk:
        return Risk(r, n, n * r, rk["max_notional"], True)

    variants = {
        "A. Τωρινό bot: 5 θέσεις × 1% ρίσκο": risk(rk["max_positions"], rk["risk_per_trade"]),
        "B. 8 θέσεις × 1% ρίσκο": risk(8, 0.01),
        "C. 10 θέσεις × 1% ρίσκο": risk(10, 0.01),
        "D. 5 θέσεις × 1,5% ρίσκο": risk(5, 0.015),
    }
    out = ["# Έρευνα: περισσότερα κέρδη από την ίδια στρατηγική;", "",
           f"{len(stocks)} μετοχές, {money} $ αρχικό κεφάλαιο, Capital.com 1:1 (χωρίς χρέωση νύχτας), "
           f"spread {costs.per_side:.2%} ανά πράξη. Ίδιοι κανόνες εισόδου/εξόδου με το bot· αλλάζουν μόνο τα όρια "
           "και τι κάνουν τα αδιάθετα χρήματα.", "",
           "«Ανά έτος / πτώση»: απόδοση ανά μονάδα χειρότερης πτώσης (όσο μεγαλύτερο, τόσο καλύτερο).", "",
           "> Προσοχή: οι μετοχές είναι οι σημερινές μεγάλες εταιρείες, οπότε τα νούμερα είναι ελαφρώς φουσκωμένα. "
           "Οι συγκρίσεις μεταξύ παραλλαγών μένουν δίκαιες, γιατί όλες έχουν το ίδιο πρόβλημα.", ""]
    summary: dict[str, dict] = {}
    for period, (start, end) in PERIODS.items():
        rows: list[tuple[str, dict]] = []
        results = {}
        for name, rcfg in variants.items():
            res = run(stocks, p, rcfg, costs, start, end, market, initial)
            results[name] = res
            st = stats(res)
            idle = (res["cash"].clip(lower=0) / res["equity"]).mean()
            rows.append((f"{name} (αδιάθετα κατά μέσο όρο {idle:.0%})", st))
        base = results["A. Τωρινό bot: 5 θέσεις × 1% ρίσκο"]
        ndx = run_index_timing(market, costs, start, end)["equity"]
        rot = run_rotation(stocks, market, costs, start, end)["equity"]
        rows.append(("E. Τωρινό bot + αδιάθετα στον Nasdaq-100 με φίλτρο 200 ημερών",
                     _stats_curve(idle_overlay(base, ndx, costs), initial)))
        rows.append(("F. Τωρινό bot + αδιάθετα στις 5 πιο δυνατές μετοχές (μηνιαία)",
                     _stats_curve(idle_overlay(base, rot, costs), initial)))
        rows.append(("G. 50% τωρινό bot + 50% Nasdaq-100 με φίλτρο",
                     _stats_curve(blend(base["equity"], ndx, 0.5, initial), initial)))
        rows.append(("Σύγκριση: Nasdaq-100 με φίλτρο 200 ημερών", _stats_curve(ndx, initial)))
        rows.append(("Σύγκριση: αγορά και κράτηση Nasdaq-100", buy_hold(market, start, end)))
        out += [f"### {period}", "", HEAD] + [line(n, st) for n, st in rows] + [""]
        summary[period] = {n: st for n, st in rows}
    path = ROOT / "reports"
    path.mkdir(exist_ok=True)
    (path / "research_plus.md").write_text("\n".join(out), encoding="utf-8")
    (path / "research_plus.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=float),
                                             encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
