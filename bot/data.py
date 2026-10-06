"""Ημερήσια δεδομένα από το Yahoo Finance."""

from __future__ import annotations

import time

import pandas as pd

COLS = ["Open", "High", "Low", "Close", "Volume"]


def daily(symbol: str, start: str = "2014-01-01", tries: int = 3) -> pd.DataFrame:
    import yfinance as yf

    last_err = None
    for attempt in range(tries):
        try:
            df = yf.Ticker(symbol).history(start=start, interval="1d", auto_adjust=True)
            if df.empty:
                raise ValueError(f"κενά δεδομένα για {symbol}")
            df = df[COLS].dropna(subset=["Close"])
            df.index = df.index.tz_localize(None).normalize()
            return df
        except Exception as e:  # προσωρινό όριο αιτημάτων κ.λπ.
            last_err = e
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Αποτυχία λήψης {symbol}: {last_err}")


def load_all(symbols: list[str], start: str = "2014-01-01") -> dict[str, pd.DataFrame]:
    return {s: daily(s, start) for s in symbols}
