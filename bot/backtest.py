"""Backtest χαρτοφυλακίου με όρια ρίσκου, κόστη και έλεγχο σε «άγνωστα» χρόνια.

Εκτέλεση:  python -m bot.backtest
Γράφει τα αποτελέσματα στο reports/backtest.md
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .strategy import Params, market_ok, signals

ROOT = Path(__file__).resolve().parent.parent
CONFIG = json.loads((Path(__file__).parent / "config.json").read_text(encoding="utf-8"))

PERIODS = {
    "Όλη η περίοδος": ("2016-01-01", None),
    "2016–2021 (εκπαίδευση)": ("2016-01-01", "2021-12-31"),
    "2022–σήμερα (άγνωστα χρόνια)": ("2022-01-01", None),
}


@dataclass
class Risk:
    risk_per_trade: float = 0.01
    max_positions: int = 5
    max_total_risk: float = 0.05
    max_notional: float = 1.0
    use_market_filter: bool = True


@dataclass
class Costs:
    per_side: float = 0.001
    financing_annual: float = 0.08


def run(data: dict[str, pd.DataFrame], p: Params, risk: Risk, costs: Costs,
        start: str, end: str | None = None, market: pd.DataFrame | None = None,
        initial: float = 10_000.0) -> dict:
    sigs = {s: signals(df, p) for s, df in data.items()}
    mkt = market_ok(market, p) if (risk.use_market_filter and market is not None) else None
    end_ts = pd.Timestamp(end) if end else None
    dates = sorted(set().union(*[df.index for df in data.values()]))
    dates = [d for d in dates if d >= pd.Timestamp(start) and (end_ts is None or d <= end_ts)]

    cash, equity = initial, initial
    positions: dict[str, dict] = {}
    pending: list[tuple] = []
    trades: list[dict] = []
    curve: list[tuple] = []
    last_close: dict[str, float] = {}
    prev_d = None
    days_invested = 0

    def close_pos(s: str, px: float, d, reason: str):
        nonlocal cash
        pos = positions.pop(s)
        exit_cost = pos["qty"] * px * costs.per_side
        cash += pos["qty"] * px - exit_cost
        pnl = pos["qty"] * (px - pos["entry"]) - pos["entry_cost"] - exit_cost - pos["fin"]
        trades.append({"symbol": s, "entry_date": pos["entry_date"], "exit_date": d,
                       "entry": pos["entry"], "exit": px, "qty": pos["qty"], "pnl": pnl,
                       "r": pnl / (pos["qty"] * pos["dist"]), "reason": reason})

    def check_intraday(s: str, row, d) -> bool:
        pos = positions[s]
        o, h, l = row["Open"], row["High"], row["Low"]
        if pos["entry_date"] != d and o <= pos["stop"]:
            close_pos(s, o, d, "Stop"); return True
        if l <= pos["stop"]:
            close_pos(s, pos["stop"], d, "Stop"); return True
        if pos["entry_date"] != d and o >= pos["tp"]:
            close_pos(s, o, d, "Στόχος"); return True
        if h >= pos["tp"]:
            close_pos(s, pos["tp"], d, "Στόχος"); return True
        return False

    for d in dates:
        # 1) Έξοδοι στο άνοιγμα και μέσα στη μέρα
        for s in list(positions):
            df = data[s]
            if d not in df.index:
                continue
            row = df.loc[d]
            if positions[s]["exit_pending"]:
                close_pos(s, row["Open"], d, positions[s]["exit_pending"])
            else:
                check_intraday(s, row, d)

        # 2) Νέες θέσεις στο άνοιγμα (από σήματα της χθεσινής μέρας)
        for s, stop, tp, dist in pending:
            if s in positions or len(positions) >= risk.max_positions or d not in data[s].index:
                continue
            row = data[s].loc[d]
            o = row["Open"]
            if not (stop < o < tp) or dist <= 0:
                continue
            eq = equity
            qty = eq * risk.risk_per_trade / dist
            notional_used = sum(ps["qty"] * last_close.get(x, ps["entry"]) for x, ps in positions.items())
            qty = min(qty, max(0.0, risk.max_notional * eq - notional_used) / o)
            risk_used = sum(ps["qty"] * ps["dist"] for ps in positions.values())
            qty = min(qty, max(0.0, risk.max_total_risk * eq - risk_used) / dist)
            if qty * o < 0.02 * eq:
                continue
            entry_cost = qty * o * costs.per_side
            cash -= qty * o + entry_cost
            positions[s] = {"qty": qty, "entry": o, "stop": stop, "tp": tp, "dist": dist,
                            "entry_date": d, "entry_cost": entry_cost, "fin": 0.0,
                            "bars": 0, "exit_pending": ""}
            check_intraday(s, row, d)
        pending = []

        # 3) Κλείσιμο ημέρας: αποτίμηση, κόστος χρηματοδότησης, έξοδοι για αύριο
        nights = (d - prev_d).days if prev_d is not None else 1
        for s, pos in positions.items():
            if d in data[s].index:
                last_close[s] = data[s].at[d, "Close"]
            px = last_close.get(s, pos["entry"])
            fin = pos["qty"] * px * costs.financing_annual * nights / 365
            cash -= fin
            pos["fin"] += fin
            if d not in sigs[s].index:
                continue
            if pos["entry_date"] != d:
                pos["bars"] += 1
            if pos["bars"] >= p.max_bars:
                pos["exit_pending"] = "Χρόνος"
            elif p.trend_exit and bool(sigs[s].at[d, "trend_broken"]):
                pos["exit_pending"] = "Τάση"
        for s in data:
            if d in data[s].index:
                last_close[s] = data[s].at[d, "Close"]
        equity = cash + sum(ps["qty"] * last_close.get(s, ps["entry"]) for s, ps in positions.items())
        curve.append((d, equity))
        if positions:
            days_invested += 1

        # 4) Νέα σήματα στο κλείσιμο
        allowed = True if mkt is None else bool(mkt.get(d, False))
        if allowed:
            cands = []
            for s, sg in sigs.items():
                if s in positions or d not in sg.index or not bool(sg.at[d, "long_sig"]):
                    continue
                cands.append((sg.at[d, "mom63"] if not math.isnan(sg.at[d, "mom63"]) else -9, s,
                              sg.at[d, "stop"], sg.at[d, "tp"], p.sl_atr * sg.at[d, "atr"]))
            cands.sort(reverse=True)
            pending = [(s, stop, tp, dist) for _, s, stop, tp, dist in cands]
        prev_d = d

    # Κλείσιμο ό,τι μένει ανοιχτό στην τελευταία τιμή
    if dates:
        for s in list(positions):
            close_pos(s, last_close.get(s, positions[s]["entry"]), dates[-1], "Τέλος")
    eq_series = pd.Series([e for _, e in curve], index=[d for d, _ in curve], dtype=float)
    return {"trades": trades, "equity": eq_series, "initial": initial,
            "exposure": days_invested / max(len(dates), 1)}


def stats(res: dict) -> dict:
    eq, tr = res["equity"], res["trades"]
    if eq.empty:
        return {}
    total = eq.iloc[-1] / res["initial"] - 1
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    cagr = (eq.iloc[-1] / res["initial"]) ** (1 / years) - 1
    dd = (eq / eq.cummax() - 1).min()
    pnl = np.array([t["pnl"] for t in tr]) if tr else np.array([])
    wins, losses = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
    daily = eq.pct_change().dropna()
    sharpe = daily.mean() / daily.std() * math.sqrt(252) if daily.std() > 0 else 0.0
    return {
        "total": total, "cagr": cagr, "max_dd": dd, "trades": len(tr),
        "win": float((pnl > 0).mean()) if len(pnl) else 0.0,
        "pf": float(wins / losses) if losses > 0 else float("inf"),
        "avg_r": float(np.mean([t["r"] for t in tr])) if tr else 0.0,
        "sharpe": sharpe, "exposure": res["exposure"],
    }


def buy_hold(df: pd.DataFrame, start: str, end: str | None) -> dict:
    c = df["Close"]
    c = c[c.index >= pd.Timestamp(start)]
    if end:
        c = c[c.index <= pd.Timestamp(end)]
    eq = c / c.iloc[0] * 10_000
    return stats({"equity": eq, "trades": [], "initial": 10_000.0, "exposure": 1.0})


def pct(x: float) -> str:
    return f"{x * 100:+.1f}%".replace(".", ",")


def num(x: float, d: int = 2) -> str:
    return "∞" if x == float("inf") else f"{x:.{d}f}".replace(".", ",")


def row(name: str, st: dict) -> str:
    if not st:
        return f"| {name} | – | – | – | – | – | – | – |"
    return (f"| {name} | {pct(st['total'])} | {pct(st['cagr'])} | {pct(st['max_dd'])} | "
            f"{st['trades']} | {st['win'] * 100:.0f}% | {num(st['pf'])} | {num(st['sharpe'])} |")


HEAD = ("| | Συνολικό | Ανά έτος | Μέγ. πτώση | Συναλλαγές | Επιτυχία | Profit factor | Sharpe |\n"
        "|---|---|---|---|---|---|---|---|")


def main() -> None:
    from .data import load_all

    p = Params.from_dict(CONFIG["strategy"])
    symbols = list(CONFIG["universe"])
    market_sym = CONFIG["market_symbol"]
    data = load_all(sorted(set(symbols + [market_sym])))
    market = data[market_sym]
    tradable = {s: data[s] for s in symbols}

    rk = CONFIG["risk"]
    real_costs = Costs(**CONFIG["costs"])
    variants = {
        "Χαρτοφυλάκιο χωρίς φίλτρο αγοράς": Risk(rk["risk_per_trade"], rk["max_positions"], rk["max_total_risk"],
                                                rk["max_notional"], False),
        "Χαρτοφυλάκιο με φίλτρο αγοράς": Risk(rk["risk_per_trade"], rk["max_positions"], rk["max_total_risk"],
                                              rk["max_notional"], True),
    }

    out = ["# Αποτελέσματα backtest", "",
           f"Δεδομένα έως {max(df.index[-1] for df in data.values()):%d/%m/%Y}. "
           f"Αρχικό κεφάλαιο 10.000 $ σε κάθε έλεγχο.", "",
           "## 1. Χαρτοφυλάκιο (όλες οι μετοχές μαζί, ρεαλιστικά κόστη)", "",
           f"Ρίσκο {rk['risk_per_trade']:.0%} ανά συναλλαγή, έως {rk['max_positions']} θέσεις, "
           f"συνολικό ρίσκο έως {rk['max_total_risk']:.0%}, χωρίς μόχλευση. Κόστος {real_costs.per_side:.1%} "
           f"ανά πράξη και χρηματοδότηση CFD {real_costs.financing_annual:.0%} τον χρόνο.", ""]
    summary = {}
    for period, (start, end) in PERIODS.items():
        out += [f"### {period}", "", HEAD]
        for name, rcfg in variants.items():
            st = stats(run(tradable, p, rcfg, real_costs, start, end, market))
            summary[f"{period} | {name}"] = st
            out.append(row(name, st))
        bh = buy_hold(market, start, end)
        summary[f"{period} | Αγορά και κράτηση Nasdaq-100"] = bh
        out += [row("Σύγκριση: αγορά και κράτηση Nasdaq-100", bh), ""]

    # Έλεγχος ότι η Python βγάζει τα ίδια με το TradingView (κάθε μετοχή μόνη, κόστη όπως στο Pine)
    tv_risk = Risk(rk["risk_per_trade"], 1, 1.0, 1.0, False)
    tv_costs = Costs(per_side=0.0005, financing_annual=0.0)
    out += ["## 2. Κάθε μετοχή μόνη της (ρυθμίσεις όπως στο TradingView, για σύγκριση)", "", HEAD]
    per_symbol = {}
    for s in symbols:
        st = stats(run({s: data[s]}, p, tv_risk, tv_costs, "2016-01-01", None))
        per_symbol[s] = st
        out.append(row(s, st))
    out.append("")

    path = ROOT / "reports"
    path.mkdir(exist_ok=True)
    (path / "backtest.md").write_text("\n".join(out), encoding="utf-8")
    (path / "backtest.json").write_text(json.dumps({"portfolio": summary, "per_symbol": per_symbol},
                                                   ensure_ascii=False, indent=1, default=float),
                                        encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
