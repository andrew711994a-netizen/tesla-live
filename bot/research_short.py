"""Έρευνα: να κάνει το bot και short (πώληση για κέρδος όταν μια μετοχή πέφτει);

Οι κανόνες του bot ανάποδα, για μετοχές σε πτωτική τάση (EMA 50 κάτω από EMA 150 και κλείσιμο κάτω από τον αργό):
  - αποτυχημένη άνοδος: το RSI 14 ξαναπέφτει κάτω από το 60,
  - σπάσιμο του χαμηλού 20 ημερών με αυξημένο όγκο (×1,2),
  - stop 2 ATR πάνω, στόχος 3 ATR κάτω, έως 20 μέρες, έξοδος αν το κλείσιμο ξαναβγεί πάνω από τον αργό EMA.
Νέα short μόνο όταν η αγορά (Nasdaq-100) είναι πτωτική (κάτω από τον EMA 150), εκτός αν λέει αλλιώς η παραλλαγή.
Κόστη: ίδιο spread, χρέωση νύχτας για τα short (στην Capital.com η μηδενική χρέωση 1:1 δεν ισχύει για short), και
τα μερίσματα τα πληρώνει όποιος έχει short (τα διορθωμένα κλεισίματα του Yahoo το περιέχουν ήδη).
Long και short μοιράζονται το ίδιο κεφάλαιο και τα ίδια όρια (8 θέσεις, 0,75% ρίσκο, συνολικό ρίσκο 6%).

Κρίνεται όπως στο bot/research_params.py: καλύτερο Sharpe και στα τρία τεστ χωρίς μεγαλύτερη πτώση, και μετά
γειτονικές τιμές. Δεν αλλάζει τίποτα στο bot.

Εκτέλεση:  python -m bot.research_short
Γράφει το reports/research_short.md και .json
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import CONFIG, Costs, Risk, fills, num, pct, stats
from .data import load_all
from .research_params import OTHER, TESTS
from .strategy import Params, _ema, market_ok, signals

ROOT = Path(__file__).resolve().parent.parent
BASE_SHORT = {"mode": "both", "rsi": 60, "fin": 0.03, "regime": 150, "risk_mult": 1.0, "long": True}


def short_signals(df: pd.DataFrame, p: Params, sg: pd.DataFrame, sh: dict) -> pd.DataFrame:
    c, l, v = df["Close"], df["Low"], df["Volume"].fillna(0)
    down = (c < sg["ema_s"]) & (sg["ema_f"] < sg["ema_s"])
    cross_dn = (sg["rsi"] < sh["rsi"]) & (sg["rsi"].shift() >= sh["rsi"])
    vol_avg = v.rolling(20).mean()
    vol_ok = ~(vol_avg > 0) | (v > p.vol_mult * vol_avg)
    ll = l.rolling(p.brk_len).min().shift(1)
    rally = down & cross_dn
    brk = down & (c < ll) & vol_ok
    sig = (rally if sh["mode"] in ("both", "rally") else False) | (brk if sh["mode"] in ("both", "break") else False)
    return pd.DataFrame({"short_sig": (pd.Series(sig, index=df.index).fillna(False) & sg["atr"].notna()),
                         "stop": c + p.sl_atr * sg["atr"], "tp": c - p.tp_atr * sg["atr"],
                         "trend_up": c > sg["ema_s"], "mom63": sg["mom63"]}, index=df.index)


def run_ls(data: dict[str, pd.DataFrame], p: Params, risk: Risk, costs: Costs, start: str, end: str | None,
           market: pd.DataFrame, fl: dict, sh: dict | None = None, initial: float = 10_000.0) -> dict:
    """Όπως το backtest.run (με τα fills της Capital.com), και με short όταν δίνεται sh. sh=None: ίδιο με το bot."""
    sigs = {s: signals(df, p) for s, df in data.items()}
    ssig = {s: short_signals(data[s], p, sigs[s], sh) for s in data} if sh else {}
    do_long = True if sh is None else sh.get("long", True)
    mkt = market_ok(market, p) if (risk.use_market_filter and market is not None) else None
    bear = None
    if sh and sh.get("regime"):
        mc = market["Close"]
        bear = (mc < _ema(mc, sh["regime"])).fillna(False)
    end_ts = pd.Timestamp(end) if end else None
    dates = sorted(set().union(*[df.index for df in data.values()]))
    dates = [d for d in dates if d >= pd.Timestamp(start) and (end_ts is None or d <= end_ts)]
    tp_shift, stop_shift = fl["tp_shift"], fl["stop_shift"]
    bid_fill = bool(tp_shift or stop_shift)

    cash, equity = initial, initial
    positions: dict[str, dict] = {}
    pending: list[tuple] = []
    trades: list[dict] = []
    curve: list[tuple] = []
    last_close: dict[str, float] = {}
    prev_d = None
    days_invested = 0

    def value(s: str, ps: dict) -> float:
        lc = last_close.get(s, ps["entry"])
        return ps["qty"] * lc if ps["dir"] > 0 else ps["qty"] * (2 * ps["entry"] - lc)

    def close_pos(s: str, px: float, d, reason: str, at_level: bool = False):
        nonlocal cash
        pos = positions.pop(s)
        k = pos["dir"]
        exit_cost = 0.0 if at_level else pos["qty"] * px * costs.per_side
        cash += (pos["qty"] * px if k > 0 else pos["qty"] * (2 * pos["entry"] - px)) - exit_cost
        pnl = k * pos["qty"] * (px - pos["entry"]) - pos["entry_cost"] - exit_cost - pos["fin"]
        trades.append({"symbol": s, "dir": k, "entry_date": pos["entry_date"], "exit_date": d, "entry": pos["entry"],
                       "exit": px, "qty": pos["qty"], "pnl": pnl, "r": pnl / (pos["qty"] * pos["dist"]),
                       "reason": reason})

    def check_intraday(s: str, row, d) -> bool:
        pos = positions[s]
        o, h, l = row["Open"], row["High"], row["Low"]
        first = pos["entry_date"] == d
        if pos["dir"] > 0:
            if not first and o <= pos["stop"]:
                close_pos(s, o, d, "Stop"); return True
            if l <= pos["stop"]:
                close_pos(s, pos["stop_lvl"], d, "Stop", bid_fill); return True
            if not first and o >= pos["tp"]:
                close_pos(s, o, d, "Στόχος"); return True
            if h >= pos["tp"]:
                close_pos(s, pos["tp_lvl"], d, "Στόχος", bid_fill); return True
            return False
        if not first and o >= pos["stop"]:
            close_pos(s, o, d, "Stop"); return True
        if h >= pos["stop"]:
            close_pos(s, pos["stop_lvl"], d, "Stop", bid_fill); return True
        if not first and o <= pos["tp"]:
            close_pos(s, o, d, "Στόχος"); return True
        if l <= pos["tp"]:
            close_pos(s, pos["tp_lvl"], d, "Στόχος", bid_fill); return True
        return False

    for d in dates:
        for s in list(positions):
            df = data[s]
            if d not in df.index:
                continue
            row = df.loc[d]
            if positions[s]["exit_pending"]:
                close_pos(s, row["Open"], d, positions[s]["exit_pending"])
            else:
                check_intraday(s, row, d)

        for s, k, stop, tp, dist in pending:
            if s in positions or len(positions) >= risk.max_positions or d not in data[s].index:
                continue
            row = data[s].loc[d]
            o = row["Open"]
            if k > 0:
                if not (stop < o < tp) or dist <= 0:
                    continue
                dist = o - stop
            else:
                if not (tp < o < stop) or dist <= 0:
                    continue
                dist = stop - o
            eq = equity
            mult = 1.0 if k > 0 else sh["risk_mult"]
            qty = eq * risk.risk_per_trade * mult / dist
            notional_used = sum(ps["qty"] * last_close.get(x, ps["entry"]) for x, ps in positions.items())
            qty = min(qty, max(0.0, risk.max_notional * eq - notional_used) / o)
            risk_used = sum(ps["qty"] * ps["dist"] for ps in positions.values())
            qty = min(qty, max(0.0, risk.max_total_risk * eq - risk_used) / dist)
            if qty * o < 0.02 * eq:
                continue
            entry_cost = qty * o * costs.per_side
            cash -= qty * o + entry_cost     # 1:1: όλη η αξία δεσμεύεται, και στα short
            if k > 0:
                st, tt = stop * (1 + stop_shift), tp * (1 + tp_shift)
            else:                             # το short κλείνει στην τιμή αγοράς (ask), λίγο πάνω από το Yahoo
                st, tt = stop * (1 - stop_shift), tp * (1 - tp_shift)
            positions[s] = {"dir": k, "qty": qty, "entry": o, "stop": st, "tp": tt, "stop_lvl": stop, "tp_lvl": tp,
                            "dist": dist, "entry_date": d, "entry_cost": entry_cost, "fin": 0.0, "bars": 0,
                            "exit_pending": ""}
            check_intraday(s, row, d)
        pending = []

        nights = (d - prev_d).days if prev_d is not None else 1
        for s, pos in positions.items():
            if d in data[s].index:
                last_close[s] = data[s].at[d, "Close"]
            px = last_close.get(s, pos["entry"])
            rate = costs.financing_annual if pos["dir"] > 0 else sh["fin"]
            fin = pos["qty"] * px * rate * nights / 365
            cash -= fin
            pos["fin"] += fin
            if d not in sigs[s].index:
                continue
            if pos["entry_date"] != d:
                pos["bars"] += 1
            broken = (bool(sigs[s].at[d, "trend_broken"]) if pos["dir"] > 0 else bool(ssig[s].at[d, "trend_up"]))
            if pos["bars"] >= p.max_bars:
                pos["exit_pending"] = "Χρόνος"
            elif p.trend_exit and broken:
                pos["exit_pending"] = "Τάση"
        for s in data:
            if d in data[s].index:
                last_close[s] = data[s].at[d, "Close"]
        equity = cash + sum(value(s, ps) for s, ps in positions.items())
        curve.append((d, equity, cash))
        if positions:
            days_invested += 1

        allowed = True if mkt is None else bool(mkt.get(d, False))
        if allowed and do_long:
            cands = []
            for s, sg in sigs.items():
                if s in positions or d not in sg.index or not bool(sg.at[d, "long_sig"]):
                    continue
                cands.append((sg.at[d, "mom63"] if not math.isnan(sg.at[d, "mom63"]) else -9, s,
                              sg.at[d, "stop"], sg.at[d, "tp"], p.sl_atr * sg.at[d, "atr"]))
            cands.sort(reverse=True)
            pending = [(s, 1, stop, tp, dist) for _, s, stop, tp, dist in cands]
        if sh and (bear is None or bool(bear.get(d, False))):
            cs = []
            for s, ss in ssig.items():
                if s in positions or d not in ss.index or not bool(ss.at[d, "short_sig"]):
                    continue
                m = ss.at[d, "mom63"]
                cs.append((m if not math.isnan(m) else 9, s, ss.at[d, "stop"], ss.at[d, "tp"],
                           p.sl_atr * sigs[s].at[d, "atr"]))
            cs.sort()                          # οι πιο αδύναμες πρώτα
            pending += [(s, -1, stop, tp, dist) for _, s, stop, tp, dist in cs]
        prev_d = d

    if dates:
        for s in list(positions):
            close_pos(s, last_close.get(s, positions[s]["entry"]), dates[-1], "Τέλος")
    idx = [d for d, _, _ in curve]
    return {"trades": trades, "equity": pd.Series([e for _, e, _ in curve], index=idx, dtype=float),
            "cash": pd.Series([c for _, _, c in curve], index=idx, dtype=float), "initial": initial,
            "exposure": days_invested / max(len(dates), 1)}


def main() -> None:
    p = Params.from_dict(CONFIG["strategy"])
    rk = CONFIG["risk"]
    risk = Risk(rk["risk_per_trade"], rk["max_positions"], rk["max_total_risk"], rk["max_notional"], True)
    costs = Costs(**CONFIG["costs"])
    fl = fills()
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
    U = {"A": {s: raw[s] for s in uni_a if s in raw},
         "B": {s: raw[s] for s in uni_b if s in raw and raw[s].index[0] <= pd.Timestamp("2015-01-01")}}

    # Έλεγχος ακρίβειας: χωρίς short = ακριβώς το backtest του bot
    from .backtest import run
    ref = run(U["A"], p, risk, costs, "2022-01-01", None, market, **fl)
    mine = run_ls(U["A"], p, risk, costs, "2022-01-01", None, market, fl, None)
    same = len(ref["trades"]) == len(mine["trades"]) and abs(ref["equity"].iloc[-1] - mine["equity"].iloc[-1]) < 1e-6

    def evaluate(sh: dict | None) -> dict:
        out = {}
        for label, u, start, end in TESTS:
            res = run_ls(U[u], p, risk, costs, start, end, market, fl, sh)
            st = stats(res)
            sr = [t["r"] for t in res["trades"] if t["dir"] < 0]
            st["short_trades"] = len(sr)
            st["short_avg_r"] = float(np.mean(sr)) if sr else 0.0
            st["short_win"] = float(np.mean([r > 0 for r in sr])) if sr else 0.0
            out[label] = st
        return out

    def v(**kw) -> dict:
        return {**BASE_SHORT, **kw}

    base = evaluate(None)
    variants = [
        ("Long + short όταν η αγορά είναι πτωτική (χρέωση νύχτας 3%/έτος)", v(),
         [("χρέωση 0%", v(fin=0.0)), ("χρέωση 6%", v(fin=0.06)), ("αγορά κάτω από EMA 100", v(regime=100)),
          ("αγορά κάτω από EMA 200", v(regime=200)), ("μισό ρίσκο στα short", v(risk_mult=0.5)),
          ("RSI 55", v(rsi=55)), ("RSI 65", v(rsi=65))]),
        ("Long + short, μόνο σπάσιμο χαμηλού", v(mode="break"),
         [("χρέωση 6%", v(mode="break", fin=0.06)), ("αγορά κάτω από EMA 200", v(mode="break", regime=200))]),
        ("Long + short, μόνο αποτυχημένη άνοδος", v(mode="rally"),
         [("χρέωση 6%", v(mode="rally", fin=0.06)), ("RSI 55", v(mode="rally", rsi=55)),
          ("RSI 65", v(mode="rally", rsi=65))]),
        ("Long + short πάντα (και σε ανοδική αγορά)", v(regime=None), []),
        ("Μόνο short, όταν η αγορά είναι πτωτική (για σύγκριση)", v(long=False), []),
    ]

    def verdict(res: dict) -> str:
        d_sh = [res[t]["sharpe"] - base[t]["sharpe"] for t, *_ in TESTS]
        d_dd = [res[t]["max_dd"] - base[t]["max_dd"] for t, *_ in TESTS]
        if all(x >= 0.05 for x in d_sh) and all(x >= -0.03 for x in d_dd):
            return "✅ καλύτερο και στα 3"
        if all(x <= -0.05 for x in d_sh):
            return "❌ χειρότερο και στα 3"
        return "➖ μικτό"

    summary = {"Σημερινό bot (μόνο long)": base}
    rows, passed = [], []
    for name, sh, nbs in variants:
        res = evaluate(sh)
        summary[name] = res
        vv = verdict(res)
        nb_rows, robust = [], None
        if vv.startswith("✅") and nbs:
            robust = True
            for desc, nsh in nbs:
                nr = evaluate(nsh)
                ok = all(nr[t]["sharpe"] >= base[t]["sharpe"] for t, *_ in TESTS)
                robust &= ok
                summary[f"{name} | γείτονας {desc}"] = nr
                nb_rows.append((desc, nr, ok))
            if robust:
                passed.append(name)
        rows.append((name, res, vv, nb_rows, robust))
        print(name, {k: round(x["sharpe"], 2) for k, x in res.items()}, vv)

    head = " | ".join(f"{t}: ανά έτος / πτώση / Sharpe" for t, *_ in TESTS)
    out = ["# Έρευνα: short (πώληση για κέρδος όταν μια μετοχή πέφτει)", "",
           f"Δεδομένα έως {market.index[-1]:%d/%m/%Y}. A = οι {len(U['A'])} μετοχές του bot· B = {len(U['B'])} μετοχές που "
           "το bot δεν έχει δει ποτέ. Ίδια όρια ρίσκου και κόστη με το bot, στόχος/stop όπως στην Capital.com.", "",
           f"Έλεγχος ακρίβειας: χωρίς short η προσομοίωση δίνει {'ακριβώς τα ίδια' if same else 'ΔΙΑΦΟΡΕΤΙΚΑ'} "
           "αποτελέσματα με το backtest του bot.", "",
           f"| Παραλλαγή | {head} | Short: συναλλαγές / επιτυχία / μέσο R (όλα τα τεστ) | Ετυμηγορία |",
           "|---|" + "---|" * (len(TESTS) + 2),
           "| **Σημερινό bot (μόνο long)** | " + " | ".join(
               f"{pct(base[t]['cagr'])} / {pct(base[t]['max_dd'])} / {num(base[t]['sharpe'])}" for t, *_ in TESTS) +
           " | – | – |"]
    for name, res, vv, nb_rows, robust in rows:
        n = sum(res[t]["short_trades"] for t, *_ in TESTS)
        avg_r = (sum(res[t]["short_avg_r"] * res[t]["short_trades"] for t, *_ in TESTS) / n) if n else 0.0
        win = (sum(res[t]["short_win"] * res[t]["short_trades"] for t, *_ in TESTS) / n) if n else 0.0
        v2 = vv + (" · κρατά με γειτονικές τιμές" if robust else " · ΔΕΝ κρατά με γειτονικές τιμές" if robust is False else "")
        out.append(f"| {name} | " + " | ".join(
            f"{pct(res[t]['cagr'])} / {pct(res[t]['max_dd'])} / {num(res[t]['sharpe'])}" for t, *_ in TESTS) +
            f" | {n} / {win * 100:.0f}% / {num(avg_r)} | {v2} |")
    for name, res, vv, nb_rows, robust in rows:
        if nb_rows:
            out += ["", f"Γειτονικές τιμές για «{name}» (Sharpe, πρέπει ≥ σημερινού και στα 3):", "",
                    "| Γειτονική τιμή | " + " | ".join(t for t, *_ in TESTS) + " | Κρατά; |",
                    "|---|" + "---|" * (len(TESTS) + 1)]
            for desc, nr, ok in nb_rows:
                out.append(f"| {desc} | " + " | ".join(num(nr[t]["sharpe"]) for t, *_ in TESTS) +
                           f" | {'ναι' if ok else 'όχι'} |")
    out += ["", "## Συμπέρασμα", "",
            ("Πέρασαν όλους τους ελέγχους: " + "; ".join(passed) + ". Χρειάζονται την έγκρισή σου και δοκιμή στο demo "
             "πριν μπουν στο bot." if passed else "Καμία παραλλαγή short δεν πέρασε όλους τους ελέγχους."), ""]

    path = ROOT / "reports"
    path.mkdir(exist_ok=True)
    (path / "research_short.md").write_text("\n".join(out), encoding="utf-8")
    (path / "research_short.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=float),
                                              encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
