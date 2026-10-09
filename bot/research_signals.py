"""Έλεγχος συνέπειας σημάτων: βγάζει το bot τα ίδια σήματα με το backtest;

Το bot κατεβάζει μόνο 500 μέρες ιστορικού, ενώ το backtest όλο το ιστορικό από το 2014. Οι εκθετικοί μέσοι
(EMA 150) εξαρτώνται λίγο από το πού ξεκινούν τα δεδομένα, άρα σε οριακές μέρες τα σήματα μπορεί να διαφέρουν.
Εδώ μετράμε πόσο συχνά συμβαίνει αυτό τις τελευταίες μέρες, και δείχνουμε αναλυτικά την AVGO της 06–08/10.

Εκτέλεση:  python -m bot.research_signals
Γράφει το reports/research_signals.md
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .data import load_all
from .strategy import Params, signals

ROOT = Path(__file__).resolve().parent.parent
CONFIG = json.loads((Path(__file__).parent / "config.json").read_text(encoding="utf-8"))
WINDOW_DAYS = 500     # όσο κατεβάζει σήμερα το bot/live.py
LAST_DAYS = 60        # πόσες πρόσφατες μέρες ελέγχουμε
COLS = ["close", "ema_f", "ema_s", "rsi", "atr", "long_sig", "kind", "stop", "tp", "trend_broken"]


def at_day(df: pd.DataFrame, day: pd.Timestamp, p: Params, window: int | None) -> pd.Series:
    """Το σήμα της μέρας day, όπως θα το έβλεπε το bot το επόμενο πρωί (window μέρες πίσω από εκείνο το πρωί)."""
    hist = df[df.index <= day]
    if window is not None:
        morning = day + pd.tseries.offsets.BDay(1)
        hist = hist[hist.index >= morning - pd.Timedelta(days=window)]
    return signals(hist, p).iloc[-1]


def main() -> None:
    p = Params.from_dict(CONFIG["strategy"])
    symbols = list(CONFIG["universe"])
    data = load_all(sorted(set(symbols + [CONFIG["market_symbol"]])))
    end = max(df.index[-1] for df in data.values())
    days = data[CONFIG["market_symbol"]].index[-LAST_DAYS:]

    rows, diffs = [], []
    for s in symbols:
        df = data[s]
        for d in days:
            if d not in df.index:
                continue
            full, short = at_day(df, d, p, None), at_day(df, d, p, WINDOW_DAYS)
            ema_gap = abs(short["ema_s"] / full["ema_s"] - 1) if full["ema_s"] else 0.0
            rows.append({"symbol": s, "day": d, "ema_gap": ema_gap,
                         "sig_full": bool(full["long_sig"]), "sig_short": bool(short["long_sig"]),
                         "trend_full": bool(full["trend_broken"]), "trend_short": bool(short["trend_broken"])})
            if (bool(full["long_sig"]) != bool(short["long_sig"])
                    or bool(full["trend_broken"]) != bool(short["trend_broken"])):
                diffs.append(f"| {s} | {d:%d/%m} | {'ναι' if full['long_sig'] else 'όχι'} {full['kind']} | "
                             f"{'ναι' if short['long_sig'] else 'όχι'} {short['kind']} | "
                             f"{'ναι' if full['trend_broken'] else 'όχι'} | {'ναι' if short['trend_broken'] else 'όχι'} | "
                             f"{ema_gap * 100:.2f}% |")
    t = pd.DataFrame(rows)
    n_sig = int(t["sig_full"].sum())
    out = ["# Έλεγχος συνέπειας σημάτων (bot σε σχέση με το backtest)", "",
           f"Δεδομένα έως {end:%d/%m/%Y}. Τελευταίες {LAST_DAYS} μέρες, {len(symbols)} μετοχές, {len(t)} μετοχο-μέρες.",
           f"Το bot κατεβάζει {WINDOW_DAYS} ημερολογιακές μέρες ιστορικού· το backtest όλο το ιστορικό από το 2014.", "",
           f"- Σήματα αγοράς με όλο το ιστορικό: **{n_sig}** · με {WINDOW_DAYS} μέρες: **{int(t['sig_short'].sum())}**",
           f"- Διαφορετικό σήμα αγοράς: **{int((t['sig_full'] != t['sig_short']).sum())}** μετοχο-μέρες",
           f"- Διαφορετική «σπασμένη τάση»: **{int((t['trend_full'] != t['trend_short']).sum())}** μετοχο-μέρες",
           f"- Απόκλιση του EMA {p.ema_slow}: μέση {t['ema_gap'].mean() * 100:.2f}% · μέγιστη {t['ema_gap'].max() * 100:.2f}%", ""]
    if diffs:
        out += ["## Μέρες με διαφορά", "",
                "| Μετοχή | Μέρα | Σήμα (όλο το ιστορικό) | Σήμα (500 μέρες) | Τάση έσπασε (όλο) | Τάση έσπασε (500) | Απόκλιση EMA |",
                "|---|---|---|---|---|---|---|", *diffs, ""]

    # AVGO: τι είδε το bot στις 07/10 και στις 09/10, και τι το backtest
    if "AVGO" in data:
        df = data["AVGO"]
        out += ["## AVGO 06–08/10 με διαφορετικά παράθυρα δεδομένων", "",
                "| Κλείσιμο | Παράθυρο | " + " | ".join(COLS) + " | όγκος | μέσος όγκος 20 | υψηλό 20 ημ. |",
                "|---|---|" + "---|" * (len(COLS) + 3)]
        for day in ("2026-10-06", "2026-10-07", "2026-10-08"):
            d = pd.Timestamp(day)
            if d not in df.index:
                continue
            for name, start in (("όλο από 2014", None),
                                ("500 μέρες πριν τις 07/10", pd.Timestamp("2026-10-07") - pd.Timedelta(days=WINDOW_DAYS)),
                                ("500 μέρες πριν τις 09/10", pd.Timestamp("2026-10-09") - pd.Timedelta(days=WINDOW_DAYS))):
                hist = df[df.index <= d]
                if start is not None:
                    hist = hist[hist.index >= start]
                sg = signals(hist, p).iloc[-1]
                vol, vavg = hist["Volume"].iloc[-1], hist["Volume"].rolling(20).mean().iloc[-1]
                hh = hist["High"].rolling(p.brk_len).max().shift(1).iloc[-1]
                vals = [f"{sg[c]:.2f}" if isinstance(sg[c], float) else str(sg[c]) for c in COLS]
                out.append(f"| {d:%d/%m} | {name} | " + " | ".join(vals) + f" | {vol:,.0f} | {vavg:,.0f} | {hh:.2f} |")
        nxt = df[df.index > pd.Timestamp("2026-10-06")].head(3)
        out += ["", "Άνοιγμα των επόμενων ημερών: " +
                ", ".join(f"{d:%d/%m} {r['Open']:.2f}" for d, r in nxt.iterrows()), ""]

    path = ROOT / "reports"
    path.mkdir(exist_ok=True)
    (path / "research_signals.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
