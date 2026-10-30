"""Επίπεδα stop/στόχου για μια χειροκίνητη θέση (π.χ. ασήμι): μόνο για ενημέρωση, το bot δεν το χρησιμοποιεί.

Από τα ημερήσια κεριά του Yahoo: τελευταία τιμή, μέση ημερήσια κίνηση (ATR 14), στηρίξεις/αντιστάσεις, και πόσο
συχνά στο παρελθόν μια αγορά έφτασε πρώτα στον στόχο ή στο stop για διάφορους συνδυασμούς (σε πολλαπλάσια του ATR).
Αν σε μία μέρα αγγίχτηκαν και τα δύο, μετράει ως stop (συντηρητικά).

Εκτέλεση:  python -m bot.levels   (σύμβολα από LEVELS_SYMBOLS, προεπιλογή SI=F = ασήμι)
Γράφει το reports/levels.md
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .data import daily
from .strategy import _ema

ROOT = Path(__file__).resolve().parent.parent
NAMES = {"SI=F": "Ασήμι (συμβόλαιο Comex)", "GC=F": "Χρυσός (συμβόλαιο Comex)"}
COMBOS = [(1.5, 3.0), (2.0, 3.0), (2.0, 4.0), (2.5, 4.0), (3.0, 4.5), (3.0, 6.0)]
HOLD = 40   # έως 40 συνεδριάσεις (~2 μήνες)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev = df["Close"].shift()
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - prev).abs(), (df["Low"] - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def outcomes(df: pd.DataFrame, a: pd.Series, k_stop: float, k_tp: float, entries: pd.Index) -> dict:
    """Αγορά στο κλείσιμο κάθε μέρας εισόδου· τι ήρθε πρώτο μέσα σε HOLD συνεδριάσεις."""
    hi, lo, cl = df["High"].to_numpy(), df["Low"].to_numpy(), df["Close"].to_numpy()
    pos = {d: i for i, d in enumerate(df.index)}
    tp_n = st_n = r_sum = n = 0
    for d in entries:
        i = pos[d]
        if i + HOLD >= len(df):
            continue
        px, dist = cl[i], k_stop * a.iloc[i]
        stop, tp = px - dist, px + k_tp * a.iloc[i]
        r = None
        for j in range(i + 1, i + HOLD + 1):
            if lo[j] <= stop:
                r = -1.0
                st_n += 1
                break
            if hi[j] >= tp:
                r = k_tp / k_stop
                tp_n += 1
                break
        if r is None:
            r = (cl[i + HOLD] - px) / dist
        r_sum += r
        n += 1
    return {"n": n, "tp": tp_n / n if n else 0, "stop": st_n / n if n else 0, "avg_r": r_sum / n if n else 0}


def main() -> None:
    symbols = [s.strip() for s in os.environ.get("LEVELS_SYMBOLS", "SI=F").split(",") if s.strip()]
    out = ["# Επίπεδα για χειροκίνητη θέση", "",
           "Μόνο για ενημέρωση· το bot δεν το χρησιμοποιεί. Οι τιμές είναι του Yahoo (συμβόλαιο Comex): η τιμή της "
           "Capital.com διαφέρει λίγο, γι' αυτό τα επίπεδα δίνονται και σε % από την τιμή αγοράς.", ""]
    for sym in symbols:
        df = daily(sym, start="2014-01-01")
        a = atr(df)
        last, a_now = float(df["Close"].iloc[-1]), float(a.iloc[-1])
        e50, e150 = float(_ema(df["Close"], 50).iloc[-1]), float(_ema(df["Close"], 150).iloc[-1])

        def chg(n: int) -> float:
            return last / float(df["Close"].iloc[-1 - n]) - 1

        out += [f"## {NAMES.get(sym, sym)} · {df.index[-1]:%d/%m/%Y}", "",
                f"- Τελευταίο κλείσιμο: **{last:.2f} $** · μεταβολή 1 μέρα {chg(1):+.1%}, 5 μέρες {chg(5):+.1%}, "
                f"20 μέρες {chg(20):+.1%}",
                f"- Μέση ημερήσια κίνηση (ATR 14): **{a_now:.2f} $ ({a_now / last:.1%})**",
                f"- Χαμηλό 20 ημερών {df['Low'].iloc[-20:].min():.2f} · χαμηλό 60 ημερών {df['Low'].iloc[-60:].min():.2f}"
                f" · υψηλό 60 ημερών {df['High'].iloc[-60:].max():.2f} · υψηλό έτους {df['High'].iloc[-252:].max():.2f}",
                f"- Μέσος 50 ημερών {e50:.2f} ({last / e50 - 1:+.1%}) · μέσος 150 ημερών {e150:.2f} ({last / e150 - 1:+.1%})",
                "", "**Ιστορικά (2014–σήμερα), αγορά στο κλείσιμο, έως 2 μήνες:**", "",
                "| Stop | Στόχος | Stop σε $ από σήμερα | Στόχος σε $ από σήμερα | Όλες οι μέρες: στόχος πρώτα / stop πρώτα / μέσο R "
                "| Μετά από πτώση ≥10% από το υψηλό 60 ημερών: στόχος / stop / μέσο R |",
                "|---|---|---|---|---|---|"]
        hist = df.index[150:]
        dip = hist[(df["Close"] / df["High"].rolling(60).max()).loc[hist] <= 0.90]
        for ks, kt in COMBOS:
            al, dp = outcomes(df, a, ks, kt, hist), outcomes(df, a, ks, kt, dip)
            out.append(f"| {ks:g}×ATR ({-ks * a_now / last:.1%}) | {kt:g}×ATR ({kt * a_now / last:+.1%}) | "
                       f"{last - ks * a_now:.2f} | {last + kt * a_now:.2f} | "
                       f"{al['tp']:.0%} / {al['stop']:.0%} / {al['avg_r']:+.2f} | "
                       f"{dp['tp']:.0%} / {dp['stop']:.0%} / {dp['avg_r']:+.2f} ({dp['n']} μέρες) |")
        out.append("")
    comma = re.compile(r"(\d)\.(\d)")   # ελληνική υποδιαστολή
    text = "\n".join(comma.sub(r"\1,\2", line) if line.startswith(("-", "|")) else line for line in out)
    path = ROOT / "reports"
    path.mkdir(exist_ok=True)
    (path / "levels.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
