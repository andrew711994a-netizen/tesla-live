"""Σύγκριση οικογενειών στρατηγικών με αυστηρό έλεγχο (2022–σήμερα = «άγνωστα» χρόνια).

Όλες οι ρυθμίσεις είναι οι τυπικές της βιβλιογραφίας, χωρίς βελτιστοποίηση πάνω στα δεδομένα.
Εκτέλεση:  python -m bot.research   →   reports/research.md
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .backtest import CONFIG, PERIODS, ROOT, Costs, Risk, buy_hold, num, pct, run, stats
from .strategy import Params, _rma


# ───────────── Βραχυπρόθεσμη επιστροφή μετά από πτώση (RSI 2) ─────────────
def mr_signals(df: pd.DataFrame) -> pd.DataFrame:
    c = df["Close"]
    d = c.diff()
    up, dn = _rma(d.clip(lower=0), 2), _rma(-d.clip(upper=0), 2)
    rsi2 = (100 - 100 / (1 + up / dn.replace(0, np.nan))).where(dn != 0, 100.0)
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - c.shift()).abs(), (df["Low"] - c.shift()).abs()],
                   axis=1).max(axis=1)
    atr = _rma(tr, 14)
    out = pd.DataFrame(index=df.index)
    out["entry"] = (c > c.rolling(200).mean()) & (rsi2 < 10)
    out["exit"] = c > c.rolling(5).mean()
    out["stop_dist"] = 3 * atr
    out["rank"] = rsi2
    return out


def run_signal_engine(data: dict[str, pd.DataFrame], sigs: dict[str, pd.DataFrame], costs: Costs,
                      start: str, end: str | None, market_ok: pd.Series | None = None, max_pos: int = 5,
                      max_hold: int = 10, initial: float = 10_000.0) -> dict:
    """Ίσα βάρη (1/max_pos του κεφαλαίου ανά θέση), χωρίς μόχλευση, είσοδος/έξοδος στο επόμενο άνοιγμα."""
    end_ts = pd.Timestamp(end) if end else None
    dates = sorted(set().union(*[df.index for df in data.values()]))
    dates = [d for d in dates if d >= pd.Timestamp(start) and (end_ts is None or d <= end_ts)]
    cash, equity = initial, initial
    positions: dict[str, dict] = {}
    pending: list[tuple] = []
    trades: list[dict] = []
    curve: list[tuple] = []
    last_close: dict[str, float] = {}
    invested_days = 0

    def close_pos(s, px, d, reason):
        nonlocal cash
        pos = positions.pop(s)
        exit_cost = pos["qty"] * px * costs.per_side
        cash += pos["qty"] * px - exit_cost
        pnl = pos["qty"] * (px - pos["entry"]) - pos["entry_cost"] - exit_cost
        trades.append({"symbol": s, "pnl": pnl, "r": pnl / (pos["qty"] * pos["dist"]), "reason": reason,
                       "entry_date": pos["entry_date"], "exit_date": d})

    for d in dates:
        for s in list(positions):
            df = data[s]
            if d not in df.index:
                continue
            row = df.loc[d]
            pos = positions[s]
            if pos["exit_pending"]:
                close_pos(s, row["Open"], d, pos["exit_pending"])
            elif row["Open"] <= pos["stop"]:
                close_pos(s, row["Open"], d, "Stop")
            elif row["Low"] <= pos["stop"]:
                close_pos(s, pos["stop"], d, "Stop")
        for s, dist in pending:
            if s in positions or len(positions) >= max_pos or d not in data[s].index:
                continue
            row = data[s].loc[d]
            alloc = min(equity / max_pos, cash)
            if alloc < 0.02 * equity or dist <= 0:
                continue
            o = row["Open"]
            qty = alloc / (o * (1 + costs.per_side))
            entry_cost = qty * o * costs.per_side
            cash -= qty * o + entry_cost
            positions[s] = {"qty": qty, "entry": o, "stop": o - dist, "dist": dist, "bars": 0,
                            "exit_pending": "", "entry_cost": entry_cost, "entry_date": d}
            if row["Low"] <= o - dist:
                close_pos(s, o - dist, d, "Stop")
        pending = []
        for s, pos in positions.items():
            if d in data[s].index:
                last_close[s] = data[s].at[d, "Close"]
            if d not in sigs[s].index:
                continue
            if pos["entry_date"] != d:
                pos["bars"] += 1
            if bool(sigs[s].at[d, "exit"]):
                pos["exit_pending"] = "Έξοδος"
            elif pos["bars"] >= max_hold:
                pos["exit_pending"] = "Χρόνος"
        for s in data:
            if d in data[s].index:
                last_close[s] = data[s].at[d, "Close"]
        equity = cash + sum(ps["qty"] * last_close.get(s, ps["entry"]) for s, ps in positions.items())
        curve.append((d, equity))
        invested_days += bool(positions)
        if market_ok is None or bool(market_ok.get(d, False)):
            cands = []
            for s, sg in sigs.items():
                if s in positions or d not in sg.index or not bool(sg.at[d, "entry"]):
                    continue
                dist = sg.at[d, "stop_dist"]
                if pd.notna(dist):
                    cands.append((sg.at[d, "rank"], s, float(dist)))
            cands.sort()
            pending = [(s, dist) for _, s, dist in cands]
    if dates:
        for s in list(positions):
            close_pos(s, last_close.get(s, positions[s]["entry"]), dates[-1], "Τέλος")
    eq = pd.Series([e for _, e in curve], index=[d for d, _ in curve], dtype=float)
    return {"trades": trades, "equity": eq, "initial": initial, "exposure": invested_days / max(len(dates), 1)}


# ───────────── Περιστροφή ορμής και φίλτρο τάσης στον δείκτη (μηνιαία / ημερήσια βάρη) ─────────────
def _equity_from_weights(closes: pd.DataFrame, weights: pd.DataFrame, costs: Costs, start: str, end: str | None,
                         initial: float = 10_000.0, financing: float = 0.0) -> dict:
    rets = closes.pct_change().fillna(0.0)
    w = weights.shift(1).fillna(0.0)                               # τα βάρη ισχύουν από την επόμενη μέρα
    gross = (w * rets).sum(axis=1)
    turnover = weights.diff().abs().sum(axis=1).shift(1).fillna(0.0)
    net = gross - turnover * costs.per_side - w.sum(axis=1) * financing / 252
    net = net[net.index >= pd.Timestamp(start)]
    if end:
        net = net[net.index <= pd.Timestamp(end)]
    eq = initial * (1 + net).cumprod()
    entries = int(((weights > 0) & (weights.shift(1).fillna(0) == 0)).sum().sum())
    monthly = (1 + net).resample("ME").prod() - 1
    return {"trades": [], "equity": eq, "initial": initial, "exposure": float((w.sum(axis=1) > 0).mean()),
            "entries": entries, "pos_months": float((monthly > 0).mean()) if len(monthly) else 0.0}


def run_rotation(data: dict[str, pd.DataFrame], market: pd.DataFrame, costs: Costs, start: str, end: str | None,
                 top_n: int = 5, lookback: int = 126) -> dict:
    closes = pd.DataFrame({s: df["Close"] for s, df in data.items()}).sort_index().ffill(limit=5)
    mom = closes / closes.shift(lookback) - 1
    above = closes > closes.rolling(200).mean()
    mc = market["Close"].reindex(closes.index).ffill()
    mkt_ok = mc > mc.rolling(200).mean()
    month_end = closes.index.to_series().groupby(closes.index.to_period("M")).max()
    weights = pd.DataFrame(np.nan, index=closes.index, columns=closes.columns)
    for d in month_end:
        row = pd.Series(0.0, index=closes.columns)
        if bool(mkt_ok.get(d, False)):
            picks = mom.loc[d][above.loc[d]].dropna().nlargest(top_n).index
            row[picks] = 1.0 / top_n
        weights.loc[d] = row
    weights = weights.ffill().fillna(0.0)
    return _equity_from_weights(closes, weights, costs, start, end)


def run_index_timing(market: pd.DataFrame, costs: Costs, start: str, end: str | None,
                     financing: float = 0.0) -> dict:
    c = market["Close"]
    closes = c.to_frame("IDX")
    weights = (c > c.rolling(200).mean()).astype(float).to_frame("IDX")
    return _equity_from_weights(closes, weights, costs, start, end, financing=financing)


# ───────────── Αναφορά ─────────────
HEAD = ("| Στρατηγική | Συνολικό | Ανά έτος | Μέγ. πτώση | Συναλλαγές | Επιτυχία | Profit factor | Sharpe |\n"
        "|---|---|---|---|---|---|---|---|")


def line(name: str, st: dict, res: dict | None = None) -> str:
    if not st:
        return f"| {name} | – | – | – | – | – | – | – |"
    if res is not None and "entries" in res:     # στρατηγικές βαρών: θετικοί μήνες αντί για επιτυχία
        n, win, pf = res["entries"], f"{res['pos_months'] * 100:.0f}% μήνες", "–"
    else:
        n, win, pf = st["trades"], f"{st['win'] * 100:.0f}%", num(st["pf"])
    return (f"| {name} | {pct(st['total'])} | {pct(st['cagr'])} | {pct(st['max_dd'])} | {n} | {win} | {pf} | "
            f"{num(st['sharpe'])} |")


def main() -> None:
    from .data import load_all

    uni = CONFIG["research_universe"]
    sectors = CONFIG["sector_universe"]
    market_sym = CONFIG["market_symbol"]
    data = load_all(sorted(set(uni + sectors + [market_sym])))
    market = data[market_sym]
    stocks = {s: data[s] for s in uni if s in data}
    secs = {s: data[s] for s in sectors if s in data}
    costs = Costs(**CONFIG["costs"])
    rk = CONFIG["risk"]
    p = Params.from_dict(CONFIG["strategy"])
    mkt_ok = market["Close"] > market["Close"].rolling(200).mean()
    mr_sigs = {s: mr_signals(df) for s, df in stocks.items()}

    out = ["# Έρευνα στρατηγικών", "",
           f"{len(stocks)} μετοχές (μαζί με εταιρείες που πήγαν άσχημα), 10.000 $ αρχικό κεφάλαιο, Capital.com 1:1, "
           f"spread {costs.per_side:.2%} ανά πράξη, χωρίς μόχλευση. Τυπικές ρυθμίσεις, χωρίς βελτιστοποίηση.", "",
           "> Προσοχή: οι μετοχές είναι οι σημερινές μεγάλες εταιρείες, οπότε τα αποτελέσματα σε μετοχές είναι "
           "ελαφρώς φουσκωμένα. Η γραμμή με τα ETF κλάδων δεν έχει αυτό το πρόβλημα.", ""]
    summary = {}
    for period, (start, end) in PERIODS.items():
        rows = []
        res_a = run(stocks, p, Risk(rk["risk_per_trade"], rk["max_positions"], rk["max_total_risk"],
                                    rk["max_notional"], True), costs, start, end, market)
        rows.append(("1. Τάση: υποχώρηση/σπάσιμο (το τωρινό bot)", stats(res_a), None))
        res_b = run_signal_engine(stocks, mr_sigs, costs, start, end, market_ok=mkt_ok)
        rows.append(("2. Επιστροφή μετά από πτώση (RSI 2), με φίλτρο αγοράς", stats(res_b), None))
        res_b2 = run_signal_engine(stocks, mr_sigs, costs, start, end, market_ok=None)
        rows.append(("2β. Επιστροφή μετά από πτώση, χωρίς φίλτρο", stats(res_b2), None))
        res_c = run_rotation(stocks, market, costs, start, end)
        rows.append(("3. Περιστροφή ορμής: top 5 μετοχές, μηνιαία", stats(res_c), res_c))
        res_c2 = run_rotation(secs, market, costs, start, end, top_n=3)
        rows.append(("3β. Περιστροφή ορμής: top 3 ETF κλάδων", stats(res_c2), res_c2))
        res_d = run_index_timing(market, costs, start, end)
        rows.append(("4. Nasdaq-100 με φίλτρο 200 ημερών (ETF)", stats(res_d), res_d))
        res_d2 = run_index_timing(market, costs, start, end, financing=0.08)
        rows.append(("4β. Ίδιο με CFD δείκτη (χρέωση νύχτας 8%)", stats(res_d2), res_d2))
        bh = buy_hold(market, start, end)
        rows.append(("Σύγκριση: αγορά και κράτηση Nasdaq-100", bh, None))
        out += [f"### {period}", "", HEAD] + [line(n, st, r) for n, st, r in rows] + [""]
        summary[period] = {n: st for n, st, _ in rows}
    path = ROOT / "reports"
    path.mkdir(exist_ok=True)
    (path / "research.md").write_text("\n".join(out), encoding="utf-8")
    (path / "research.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=float),
                                        encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
