"""Λήψη δεδομένων: Yahoo Finance (μέσω yfinance), RSS ειδήσεων, StockTwits.

Κάθε συνάρτηση κρατά τα αποτελέσματα στη μνήμη για λίγο (ttl), ώστε η
εφαρμογή να μη στέλνει περιττά αιτήματα. Σε αποτυχία πετάει εξαίρεση και
η εφαρμογή δείχνει τι δεν ήταν διαθέσιμο.
"""

from __future__ import annotations

import datetime as dt
import html
import re
from urllib.parse import quote_plus

import pandas as pd
import requests
import streamlit as st
import yfinance as yf

OHLCV = ["Open", "High", "Low", "Close", "Volume"]
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"}

META_KEYS = ["currency", "exchangeName", "fullExchangeName", "longName", "shortName",
             "regularMarketPrice", "chartPreviousClose", "previousClose",
             "regularMarketDayHigh", "regularMarketDayLow", "regularMarketVolume",
             "regularMarketTime", "currentTradingPeriod", "exchangeTimezoneName",
             "fiftyTwoWeekHigh", "fiftyTwoWeekLow"]


@st.cache_data(ttl=8, show_spinner=False)
def intraday(ticker: str) -> tuple[pd.DataFrame, dict]:
    """Κεριά 1 λεπτού της τελευταίας συνεδρίασης (μαζί με pre/after-market)."""
    t = yf.Ticker(ticker)
    df = t.history(period="1d", interval="1m", prepost=True)
    md = t.history_metadata or {}
    # Μόνο τα κλειδιά που χρειαζόμαστε: το dict(md) θα ζητούσε κι άλλα δεδομένα από το δίκτυο.
    meta = {k: md.get(k) for k in META_KEYS}
    fallback = False
    if df.empty:
        df = t.history(period="5d", interval="1m", prepost=True)
        fallback = True
        if not df.empty:
            last_day = df.index[-1].date()
            df = df[df.index.date == last_day]
    if fallback:
        meta["chartPreviousClose"] = None  # αφορά άλλη περίοδο, δεν ισχύει
    return df[OHLCV].copy(), meta


@st.cache_data(ttl=300, show_spinner=False)
def daily(ticker: str, period: str = "10y") -> pd.DataFrame:
    df = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=True)
    if df.empty:
        return df
    df = df[OHLCV].dropna(subset=["Close"])
    df.index = df.index.tz_localize(None).normalize()
    return df


INFO_KEYS = ["longName", "shortName", "sector", "industry", "country", "website", "exchange",
             "fullExchangeName", "currency", "marketCap", "trailingPE", "forwardPE", "beta",
             "dividendYield", "shortPercentOfFloat", "shortRatio", "heldPercentInstitutions",
             "heldPercentInsiders", "recommendationKey", "recommendationMean",
             "numberOfAnalystOpinions", "targetMeanPrice", "targetHighPrice", "targetLowPrice",
             "fiftyTwoWeekHigh", "fiftyTwoWeekLow", "averageVolume", "longBusinessSummary",
             "fullTimeEmployees", "quoteType"]


@st.cache_data(ttl=3600, show_spinner=False)
def info(ticker: str) -> dict:
    raw = yf.Ticker(ticker).info or {}
    return {k: raw.get(k) for k in INFO_KEYS}


@st.cache_data(ttl=3600, show_spinner=False)
def analysts(ticker: str) -> dict:
    t = yf.Ticker(ticker)
    out: dict = {}
    for key, getter in (("recs", lambda: t.recommendations),
                        ("targets", lambda: t.analyst_price_targets),
                        ("changes", lambda: t.upgrades_downgrades),
                        ("calendar", lambda: t.calendar)):
        try:
            out[key] = getter()
        except Exception:
            out[key] = None
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def insiders(ticker: str) -> pd.DataFrame | None:
    return yf.Ticker(ticker).insider_transactions


@st.cache_data(ttl=3600, show_spinner=False)
def sec_filings(ticker: str) -> list[dict]:
    f = yf.Ticker(ticker).sec_filings
    if isinstance(f, dict):
        f = f.get("filings", [])
    return [{"date": x.get("date"), "type": x.get("type"), "title": x.get("title"),
             "url": x.get("edgarUrl")} for x in (f or [])][:15]


@st.cache_data(ttl=900, show_spinner=False)
def options(ticker: str, max_expiries: int = 3) -> list[dict]:
    t = yf.Ticker(ticker)
    exps = list(t.options or [])[:max_expiries]
    today = dt.date.today()
    cols = ["strike", "lastPrice", "bid", "ask", "volume", "openInterest", "impliedVolatility"]
    chains = []
    for e in exps:
        ch = t.option_chain(e)
        if ch.calls is None or ch.puts is None:
            continue
        exp = dt.date.fromisoformat(e)
        chains.append({"expiry": exp, "days": (exp - today).days,
                       "calls": ch.calls[[c for c in cols if c in ch.calls]].copy(),
                       "puts": ch.puts[[c for c in cols if c in ch.puts]].copy()})
    return chains


@st.cache_data(ttl=3600, show_spinner=False)
def search(query: str) -> list[dict]:
    res = yf.Search(query, max_results=8, news_count=0).quotes or []
    out = []
    for q in res:
        if q.get("quoteType") not in ("EQUITY", "ETF"):
            continue
        out.append({"symbol": q.get("symbol"),
                    "name": q.get("longname") or q.get("shortname") or q.get("symbol"),
                    "exchange": q.get("exchDisp") or q.get("exchange") or ""})
    return out


# ---------------------------------------------------------------------------
# Ειδήσεις
# ---------------------------------------------------------------------------

_TAG = re.compile(r"<[^>]+>")


def _clean(text: str | None) -> str:
    return html.unescape(_TAG.sub(" ", text or "")).strip()


def _norm_key(title: str) -> str:
    return re.sub(r"[^a-z0-9]", "", title.lower())[:70]


def _yahoo_items(ticker: str) -> list[dict]:
    items = []
    for raw in yf.Ticker(ticker).get_news(count=20) or []:
        c = raw.get("content", raw) if isinstance(raw, dict) else {}
        title = c.get("title")
        if not title:
            continue
        prov = c.get("provider")
        publisher = prov.get("displayName") if isinstance(prov, dict) else raw.get("publisher")
        url = ((c.get("canonicalUrl") or {}).get("url") or (c.get("clickThroughUrl") or {}).get("url")
               or raw.get("link"))
        when = c.get("pubDate") or c.get("displayTime")
        if when:
            ts = pd.to_datetime(when, utc=True, errors="coerce")
        elif raw.get("providerPublishTime"):
            ts = pd.to_datetime(raw["providerPublishTime"], unit="s", utc=True)
        else:
            ts = pd.NaT
        items.append({"title": _clean(title), "summary": _clean(c.get("summary") or c.get("description")),
                      "publisher": publisher or "Yahoo Finance", "url": url, "time": ts,
                      "source": "Yahoo Finance"})
    return items


def _rss_items(url: str, source: str, strip_publisher: bool = False) -> list[dict]:
    import feedparser

    resp = requests.get(url, headers=UA, timeout=10)
    resp.raise_for_status()
    feed = feedparser.parse(resp.content)
    items = []
    for e in feed.entries:
        title = _clean(e.get("title"))
        publisher = (e.get("source") or {}).get("title") if isinstance(e.get("source"), dict) else None
        if strip_publisher and " - " in title:
            title, tail = title.rsplit(" - ", 1)
            publisher = publisher or tail
        ts = pd.NaT
        if e.get("published_parsed"):
            ts = pd.Timestamp(dt.datetime(*e.published_parsed[:6]), tz="UTC")
        summary = "" if strip_publisher else _clean(e.get("summary"))
        items.append({"title": title, "summary": summary, "publisher": publisher or source,
                      "url": e.get("link"), "time": ts, "source": source})
    return items


@st.cache_data(ttl=300, show_spinner=False)
def news(ticker: str, company: str | None) -> tuple[list[dict], list[str]]:
    """Ειδήσεις από 3 πηγές, χωρίς διπλότυπα. Επιστρέφει (ειδήσεις, πηγές που απέτυχαν)."""
    name = (company or "").split(",")[0]
    for suffix in (" Inc.", " Inc", " Corporation", " Corp.", " Corp", " Ltd.", " plc", " N.V."):
        name = name.replace(suffix, "")
    query = f'"{name.strip()}" OR {ticker} stock' if name.strip() else f"{ticker} stock"
    sources = [
        ("Yahoo Finance", lambda: _yahoo_items(ticker)),
        ("Yahoo RSS", lambda: _rss_items(
            f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={quote_plus(ticker)}&region=US&lang=en-US",
            "Yahoo Finance")),
        ("Google News", lambda: _rss_items(
            f"https://news.google.com/rss/search?q={quote_plus(query + ' when:7d')}&hl=en-US&gl=US&ceid=US:en",
            "Google News", strip_publisher=True)),
    ]
    seen, merged, failed = set(), [], []
    for label, fetch in sources:
        try:
            for it in fetch():
                key = _norm_key(it["title"])
                if key and key not in seen:
                    seen.add(key)
                    merged.append(it)
        except Exception:
            failed.append(label)
    oldest = pd.Timestamp("1970-01-01", tz="UTC")
    merged.sort(key=lambda x: x["time"] if pd.notna(x["time"]) else oldest, reverse=True)
    return merged[:60], failed


@st.cache_data(ttl=300, show_spinner=False)
def stocktwits(ticker: str) -> list[dict]:
    r = requests.get(f"https://api.stocktwits.com/api/2/streams/symbol/{quote_plus(ticker)}.json",
                     headers=UA, timeout=8)
    r.raise_for_status()
    out = []
    for m in r.json().get("messages", []):
        sent = ((m.get("entities") or {}).get("sentiment") or {}) or {}
        out.append({"body": m.get("body", ""), "time": pd.to_datetime(m.get("created_at"), utc=True),
                    "user": (m.get("user") or {}).get("username", ""), "sentiment": sent.get("basic")})
    return out
