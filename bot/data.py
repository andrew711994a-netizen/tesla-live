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


def intraday(symbol: str, interval: str = "1h", period: str = "730d", tries: int = 3) -> pd.DataFrame:
    """Ενδοσυνεδριακά κεριά (κανονικό ωράριο), με ώρα Νέας Υόρκης χωρίς ζώνη."""
    import yfinance as yf

    last_err = None
    for attempt in range(tries):
        try:
            df = yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=True)
            if df.empty:
                raise ValueError(f"κενά δεδομένα για {symbol}")
            df = df[COLS].dropna(subset=["Close"])
            df.index = df.index.tz_convert("America/New_York").tz_localize(None)
            return df
        except Exception as e:
            last_err = e
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Αποτυχία λήψης {symbol} ({interval}): {last_err}")


def earnings_times(symbol: str, limit: int = 100, tries: int = 3) -> list[pd.Timestamp]:
    """Ημερομηνίες και ώρες ανακοίνωσης αποτελεσμάτων (παλιές και επόμενες), ώρα Νέας Υόρκης."""
    import yfinance as yf

    last_err = None
    for attempt in range(tries):
        try:
            tk = yf.Ticker(symbol)
            df = tk.get_earnings_dates(limit=limit)
            if df is not None and not df.empty:
                return sorted(pd.Timestamp(t) for t in df.index)
            # Εφεδρικά: μόνο η επόμενη ημερομηνία, χωρίς ώρα (μεσάνυχτα = άγνωστη ώρα)
            nxt = (tk.calendar or {}).get("Earnings Date") or []
            if nxt:
                return sorted(pd.Timestamp(d).tz_localize("America/New_York") for d in nxt)
            raise ValueError(f"χωρίς ημερομηνίες αποτελεσμάτων για {symbol}")
        except Exception as e:
            last_err = e
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"Αποτυχία ημερομηνιών αποτελεσμάτων {symbol}: {last_err}")
