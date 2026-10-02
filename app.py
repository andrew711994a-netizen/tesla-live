"""Παρακολούθηση μετοχής: live τιμή, τεχνική ανάλυση, πιθανότητες, ειδήσεις.

Εκκίνηση:  streamlit run app.py
"""

from __future__ import annotations

import datetime as dt
import html
import time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import analysis as an
import data

ATHENS = ZoneInfo("Europe/Athens")
UP, DOWN, NEUTRAL = "#1f9d55", "#d64545", "#8a8f98"
ACCENT, ORANGE, PURPLE = "#3b7ddd", "#e8912d", "#9a6bd6"
FULL_REFRESH_SECS = 300

st.set_page_config(page_title="Παρακολούθηση μετοχής", page_icon="📈", layout="wide")

st.markdown(
    f"""
<style>
.block-container {{ padding-top: 2.2rem; }}
.card {{ border: 1px solid rgba(128,128,128,.28); border-radius: 10px; padding: 14px 16px; height: 100%; }}
.card .label {{ font-size: .8rem; opacity: .72; letter-spacing: .02em; }}
.card .big {{ font-size: 1.55rem; font-weight: 700; font-variant-numeric: tabular-nums; margin-top: 2px; }}
.card .sub {{ font-size: .82rem; opacity: .8; margin-top: 4px; }}
.split {{ display: flex; justify-content: space-between; font-weight: 600; font-variant-numeric: tabular-nums; margin-top: 6px; }}
.bar {{ display: flex; height: 10px; border-radius: 5px; overflow: hidden; background: rgba(128,128,128,.2); margin-top: 6px; }}
.bar .u {{ background: {UP}; }} .bar .d {{ background: {DOWN}; }}
.up {{ color: {UP}; }} .down {{ color: {DOWN}; }}
.chip {{ display: inline-block; padding: 1px 8px; margin: 2px 4px 2px 0; border-radius: 999px; font-size: .74rem; font-weight: 600; white-space: nowrap; }}
.chip.pos {{ background: rgba(31,157,85,.15); color: {UP}; }}
.chip.neg {{ background: rgba(214,69,69,.15); color: {DOWN}; }}
.chip.neu {{ background: rgba(128,128,128,.18); }}
.chip.topic {{ background: rgba(59,125,221,.14); color: {ACCENT}; }}
.chip.rumor {{ background: rgba(232,145,45,.18); color: {ORANGE}; }}
.news-item {{ padding: 10px 0; border-bottom: 1px solid rgba(128,128,128,.18); }}
.news-item a {{ font-weight: 600; text-decoration: none; }}
.news-meta {{ font-size: .78rem; opacity: .7; margin-top: 2px; }}
.state {{ display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: .8rem; font-weight: 600; }}
.state.open {{ background: rgba(31,157,85,.15); color: {UP}; }}
.state.ext {{ background: rgba(232,145,45,.18); color: {ORANGE}; }}
.state.closed {{ background: rgba(128,128,128,.18); }}
.sig {{ border: 1px solid; border-radius: 12px; padding: 14px 16px; margin: 4px 0 10px; }}
.sig.buy {{ background: rgba(31,157,85,.10); border-color: rgba(31,157,85,.45); }}
.sig.sell {{ background: rgba(214,69,69,.10); border-color: rgba(214,69,69,.45); }}
.sig.hold {{ background: rgba(128,128,128,.10); border-color: rgba(128,128,128,.35); }}
.sig-head {{ display: flex; flex-wrap: wrap; align-items: baseline; gap: 4px 14px; }}
.sig-kicker {{ font-size: .78rem; letter-spacing: .04em; opacity: .75; width: 100%; }}
.sig-label {{ font-size: 1.6rem; font-weight: 800; }}
.sig.buy .sig-label {{ color: {UP}; }} .sig.sell .sig-label {{ color: {DOWN}; }}
.sig-str {{ font-size: .9rem; font-weight: 600; opacity: .85; }}
.sig-score {{ margin-left: auto; font-size: .82rem; opacity: .75; font-variant-numeric: tabular-nums; }}
.sig-levels {{ display: flex; flex-wrap: wrap; gap: 8px 32px; margin-top: 10px; }}
.sig-levels .k {{ font-size: .78rem; opacity: .75; }}
.sig-levels .v {{ font-size: 1.2rem; font-weight: 700; font-variant-numeric: tabular-nums; }}
.sig-text {{ margin-top: 8px; }}
.sig-note {{ font-size: .82rem; opacity: .82; margin-top: 8px; }}
.sig-warn {{ font-size: .85rem; margin-top: 8px; color: {ORANGE}; font-weight: 600; }}
.live {{ display: flex; flex-wrap: wrap; gap: 14px 40px; align-items: flex-start; margin: 4px 0 8px; }}
.live .label {{ font-size: .8rem; opacity: .72; margin-bottom: 2px; }}
.live .price {{ font-size: 2.3rem; font-weight: 700; line-height: 1.1; font-variant-numeric: tabular-nums; }}
.live .chg {{ font-weight: 600; font-variant-numeric: tabular-nums; }}
.live .val {{ font-size: 1.2rem; font-weight: 600; font-variant-numeric: tabular-nums; }}
.live .sub {{ font-size: .8rem; opacity: .78; margin-top: 3px; }}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Βοηθητικά μορφοποίησης (ελληνική μορφή αριθμών: 1.234,56)
# ---------------------------------------------------------------------------


def num(x, d=2) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    return f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def pct(x, d=1, sign=True) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    s = f"{x * 100:{'+' if sign else ''}.{d}f}%"
    return s.replace(".", ",")


CURRENCY = {"USD": "$", "EUR": "€", "GBP": "£", "GBp": "p", "JPY": "¥"}


def money(x, cur="USD", d=2) -> str:
    sym = CURRENCY.get(cur or "USD", (cur or "") + " ")
    return f"{num(x, d)} {sym}".strip()


def big(x, cur=None) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    for div, word in ((1e12, "τρισ."), (1e9, "δισ."), (1e6, "εκ."), (1e3, "χιλ.")):
        if abs(x) >= div:
            s = f"{num(x / div, 2)} {word}"
            break
    else:
        s = num(x, 0)
    return f"{s} {CURRENCY.get(cur, '')}".strip() if cur else s


def ago(ts) -> str:
    if ts is None or pd.isna(ts):
        return ""
    secs = (pd.Timestamp.now(tz="UTC") - pd.Timestamp(ts)).total_seconds()
    if secs < 3600:
        m = max(1, int(secs // 60))
        return f"πριν από {m} λεπτ{'ό' if m == 1 else 'ά'}"
    if secs < 86400:
        h = int(secs // 3600)
        return f"πριν από {h} ώρ{'α' if h == 1 else 'ες'}"
    d = int(secs // 86400)
    return f"πριν από {d} ημέρ{'α' if d == 1 else 'ες'}"


def esc(s) -> str:
    """Για HTML: διαφυγή χαρακτήρων και του $ (αλλιώς το Streamlit το διαβάζει ως LaTeX)."""
    return html.escape(str(s or "")).replace("$", "&#36;")


def md(s) -> str:
    """Για απλό markdown: το $ γίνεται \\$ ώστε να μη θεωρηθεί μαθηματικός τύπος."""
    return str(s).replace("$", "\\$")


def card(label, big_text, sub="", color=None) -> str:
    style = f' style="color:{color}"' if color else ""
    return (f'<div class="card"><div class="label">{esc(label)}</div>'
            f'<div class="big"{style}>{big_text}</div><div class="sub">{sub}</div></div>')


def prob_card(title, subtitle, p_up) -> str:
    u = round(p_up * 100)
    return (f'<div class="card"><div class="label">{esc(title)} · {esc(subtitle)}</div>'
            f'<div class="split"><span class="up">▲ Άνοδος {u}%</span>'
            f'<span class="down">Πτώση {100 - u}% ▼</span></div>'
            f'<div class="bar"><div class="u" style="width:{u}%"></div>'
            f'<div class="d" style="width:{100 - u}%"></div></div></div>')


def verdict_text(v: float) -> str:
    return "▲ Ανοδικό" if v > 0.1 else "▼ Πτωτικό" if v < -0.1 else "● Ουδέτερο"


def score_color(s: float) -> str:
    return UP if s >= 15 else DOWN if s <= -15 else NEUTRAL


failed_sources: list[str] = []


def safe(label, fn, *args, default=None):
    try:
        return fn(*args)
    except Exception:
        failed_sources.append(label)
        return default


# ---------------------------------------------------------------------------
# Πλευρική στήλη
# ---------------------------------------------------------------------------

if "ticker" not in st.session_state:
    st.session_state.ticker = st.query_params.get("t", "TSLA")


def set_ticker(sym: str):
    st.session_state.ticker = sym


with st.sidebar:
    st.header("Ρυθμίσεις")
    st.text_input("Σύμβολο μετοχής", key="ticker",
                  help="Το σύμβολο στο χρηματιστήριο, π.χ. AAPL (Apple), TSLA (Tesla), "
                       "NVDA (Nvidia), MSFT (Microsoft), AMZN (Amazon).")
    with st.expander("Δεν ξέρεις το σύμβολο; Βρες το με το όνομα"):
        q = st.text_input("Όνομα εταιρείας", key="search_q", placeholder="π.χ. Coca-Cola")
        if q.strip():
            results = safe("Αναζήτηση", data.search, q.strip(), default=[])
            if not results:
                st.caption("Δεν βρέθηκε εταιρεία με αυτό το όνομα.")
            for r in results:
                st.button(f"{r['symbol']} · {r['name']} ({r['exchange']})", key=f"pick_{r['symbol']}",
                          on_click=set_ticker, args=(r["symbol"],), width="stretch")

    refresh = st.select_slider("Ανανέωση τιμής κάθε", options=[10, 15, 30, 60, 120], value=30,
                               format_func=lambda s: f"{s} δευτ.", key="refresh")
    st.subheader("Ειδοποιήσεις τιμής")
    st.number_input("Ειδοποίησέ με αν ανέβει πάνω από", min_value=0.0, value=0.0, step=1.0,
                    key="alert_up", help="0 = απενεργοποιημένη")
    st.number_input("Ειδοποίησέ με αν πέσει κάτω από", min_value=0.0, value=0.0, step=1.0,
                    key="alert_down", help="0 = απενεργοποιημένη")
    st.divider()
    st.caption("Δεδομένα: Yahoo Finance, Google News, StockTwits. Οι αμερικανικές αγορές είναι "
               "ανοιχτές 16:30–23:00 ώρα Ελλάδας (pre-market από 11:00, after-hours έως 03:00).")
    st.caption("Δεν αποτελεί επενδυτική συμβουλή. Οι πιθανότητες είναι στατιστικές εκτιμήσεις "
               "από το παρελθόν και η αγορά συχνά κινείται απρόβλεπτα.")

ticker = (st.session_state.ticker or "").strip().upper()
if not ticker:
    st.info("Γράψε ένα σύμβολο μετοχής στην αριστερή στήλη (π.χ. AAPL).")
    st.stop()
st.query_params["t"] = ticker

# ---------------------------------------------------------------------------
# Βασικά δεδομένα
# ---------------------------------------------------------------------------

@st.cache_resource
def last_good() -> dict:
    """Τα τελευταία δεδομένα που ήρθαν σωστά, για όταν το Yahoo δεν απαντά για λίγο."""
    return {}


try:
    with st.spinner(f"Φορτώνω το ιστορικό της {ticker}…"):
        daily_df = data.daily(ticker)
    if not daily_df.empty:
        last_good()[("daily", ticker)] = daily_df
except Exception as e:  # συνήθως όριο αιτημάτων ή πρόβλημα σύνδεσης
    daily_df = last_good().get(("daily", ticker))
    if daily_df is None:
        st.error(f"Δεν ήταν δυνατή η λήψη δεδομένων από το Yahoo Finance ({type(e).__name__}). "
                 "Συνήθως είναι προσωρινό όριο αιτημάτων: η σελίδα θα ξαναδοκιμάσει μόνη της σε ένα λεπτό.")
        time.sleep(60)
        st.rerun()

if daily_df.empty or len(daily_df) < 30:
    st.error(f"Δεν βρέθηκαν δεδομένα για το σύμβολο «{ticker}». Έλεγξε ότι είναι σωστό "
             "ή βρες το με την αναζήτηση ονόματος στα αριστερά.")
    st.stop()

info = safe("Στοιχεία εταιρείας", data.info, ticker, default={}) or {}
currency = info.get("currency") or "USD"
company = info.get("longName") or info.get("shortName") or ticker


@st.cache_data(ttl=FULL_REFRESH_SECS, show_spinner=False)
def run_analysis(df: pd.DataFrame):
    ind = an.add_indicators(df)
    return (ind, an.technical_signals(ind), an.combined_probability(ind),
            an.volatility_ranges(ind["Close"]))


with st.spinner("Υπολογίζω δείκτες και πιθανότητες…"):
    ind, signals, probs, vranges = run_analysis(daily_df)
tech_score = an.score_signals(signals)


@st.cache_data(ttl=3600, show_spinner=False)
def run_backtest(_ind: pd.DataFrame, key: tuple) -> dict:
    return an.backtest_signals(_ind)


try:
    with st.spinner("Ελέγχω πόσο έπεφτε μέσα το σήμα στο παρελθόν…"):
        backtest = run_backtest(ind, (ticker, str(ind.index[-1].date()), len(ind)))
except Exception:
    backtest = {}

# Αναλυτές και ημερομηνία αποτελεσμάτων (χρειάζονται και στο σήμα)
an_data = safe("Αναλυτές", data.analysts, ticker, default={}) or {}
targets = an_data.get("targets") or {}
calendar = an_data.get("calendar") or {}
earnings_date = None
ed = calendar.get("Earnings Date") if isinstance(calendar, dict) else None
if ed:
    first = ed[0] if isinstance(ed, (list, tuple)) else ed
    try:
        earnings_date = pd.Timestamp(first).date()
    except Exception:
        earnings_date = None

# ---------------------------------------------------------------------------
# Κεφαλίδα και live τιμή
# ---------------------------------------------------------------------------

meta_bits = [ticker, info.get("fullExchangeName") or info.get("exchange"), info.get("sector"),
             info.get("industry")]
st.title(company)
st.caption(" · ".join(b for b in meta_bits if b))


def market_state(meta: dict) -> tuple[str, str]:
    now = pd.Timestamp.now(tz="UTC")
    ctp = meta.get("currentTradingPeriod") or {}
    for key, label in (("regular", "Αγορά ανοιχτή"), ("pre", "Pre-market"), ("post", "After-hours")):
        p = ctp.get(key) or {}
        s, e = p.get("start"), p.get("end")
        if s is None or e is None:
            continue
        s = pd.Timestamp(s, unit="s", tz="UTC") if isinstance(s, (int, float)) else pd.Timestamp(s)
        e = pd.Timestamp(e, unit="s", tz="UTC") if isinstance(e, (int, float)) else pd.Timestamp(e)
        if s <= now <= e:
            return key, label
    return "closed", "Αγορά κλειστή"


def regular_bounds(meta: dict, intra: pd.DataFrame):
    p = (meta.get("currentTradingPeriod") or {}).get("regular") or {}
    s, e = p.get("start"), p.get("end")
    try:
        s = pd.Timestamp(s, unit="s", tz="UTC") if isinstance(s, (int, float)) else pd.Timestamp(s)
        e = pd.Timestamp(e, unit="s", tz="UTC") if isinstance(e, (int, float)) else pd.Timestamp(e)
        if not intra.empty and s.date() == intra.index[-1].tz_convert("UTC").date():
            return s, e
    except Exception:
        pass
    # Εναλλακτικά: 9:30–16:00 ώρα Νέας Υόρκης
    if intra.empty:
        return None, None
    day = intra.index[-1].tz_convert("America/New_York").date()
    ny = ZoneInfo("America/New_York")
    return (pd.Timestamp(dt.datetime.combine(day, dt.time(9, 30)), tz=ny),
            pd.Timestamp(dt.datetime.combine(day, dt.time(16, 0)), tz=ny))


def live_snapshot():
    """Επιστρέφει όλα όσα χρειάζεται η live ενότητα, ή None αν δεν απάντησε το Yahoo."""
    stale_since = None
    try:
        intra, meta = data.intraday(ticker)
        if intra.empty:
            raise ValueError("empty")
        last_good()[("intra", ticker)] = (intra, meta, time.time())
    except Exception as e:
        saved = last_good().get(("intra", ticker))
        if saved is None:
            return {"error": type(e).__name__}
        intra, meta, stale_since = saved
    reg_start, reg_end = regular_bounds(meta, intra)
    if reg_start is not None:
        regular = intra[(intra.index >= reg_start) & (intra.index < reg_end)]
    else:
        regular = intra
    last_bar_price = float(intra["Close"].iloc[-1])
    price = meta.get("regularMarketPrice") or (float(regular["Close"].iloc[-1]) if len(regular) else last_bar_price)
    session_day = (regular.index[0] if len(regular) else intra.index[-1]).tz_convert(
        "America/New_York").date()
    prev_close = meta.get("chartPreviousClose")
    if not prev_close:
        earlier = daily_df[daily_df.index.date < session_day]
        prev_close = float(earlier["Close"].iloc[-1]) if len(earlier) else None
    state, state_label = market_state(meta)
    ext_price = None
    if state in ("pre", "post") or (len(regular) and intra.index[-1] > regular.index[-1]):
        if abs(last_bar_price - price) > 1e-9:
            ext_price = last_bar_price
    return {"intra": intra, "regular": regular, "meta": meta, "price": float(price),
            "prev_close": prev_close, "state": state, "state_label": state_label,
            "ext_price": ext_price, "updated": intra.index[-1], "stale_since": stale_since}


def check_alerts(price: float):
    for key, cond, word in (("alert_up", lambda a: price >= a, "ανέβηκε πάνω από"),
                            ("alert_down", lambda a: price <= a, "έπεσε κάτω από")):
        level = st.session_state.get(key) or 0
        fired_key = f"{key}_fired"
        if level and cond(level):
            if st.session_state.get(fired_key) != level:
                st.toast(md(f"{ticker} {word} {money(level, currency)} (τώρα {money(price, currency)})"), icon="🔔")
                st.session_state[fired_key] = level
            (st.success if key == "alert_up" else st.error)(
                md(f"Ειδοποίηση: η {ticker} {word} {money(level, currency)}."))
        elif st.session_state.get(fired_key) == level:
            st.session_state[fired_key] = None


def signal_card(sig: dict, price: float) -> str:
    a = sig["action"]
    cls = {"BUY": "buy", "SELL": "sell", "HOLD": "hold"}[a]
    arrow = {"BUY": "▲", "SELL": "▼", "HOLD": "●"}[a]
    head = (f'<div class="sig-head"><div class="sig-kicker">ΣΗΜΑ ΤΩΡΑ · ΓΙΑ ΤΙΣ ΕΠΟΜΕΝΕΣ ~2 ΕΒΔΟΜΑΔΕΣ</div>'
            f'<span class="sig-label">{arrow} {esc(sig["label"])}</span>'
            + (f'<span class="sig-str">σήμα {esc(sig["strength"])}</span>' if sig["strength"] else "")
            + f'<span class="sig-score">βαθμός {sig["composite"]:+.0f} από −100 έως +100</span></div>')
    if a == "HOLD":
        up, down = sig["up_trigger"], sig["down_trigger"]
        body = (f'<div class="sig-text">Δεν υπάρχει καθαρή εικόνα αυτή τη στιγμή. Παρακολούθησε:</div>'
                f'<div class="sig-levels">'
                f'<div><div class="k">Γίνεται ανοδικό αν περάσει πάνω από</div>'
                f'<div class="v up">{esc(money(up, currency))} <small>({pct(up / price - 1)})</small></div></div>'
                f'<div><div class="k">Γίνεται πτωτικό αν πέσει κάτω από</div>'
                f'<div class="v down">{esc(money(down, currency))} <small>({pct(down / price - 1)})</small></div></div>'
                f'</div>')
    else:
        buy = a == "BUY"
        tgt_cls, stop_cls = ("up", "down") if buy else ("down", "up")
        body = (f'<div class="sig-levels">'
                f'<div><div class="k">{"Στόχος: μέχρι" if buy else "Στόχος: πτώση μέχρι"}</div>'
                f'<div class="v {tgt_cls}">{esc(money(sig["target"], currency))} '
                f'<small>({pct(sig["target"] / price - 1)})</small></div></div>'
                f'<div><div class="k">Το σήμα ακυρώνεται αν {"πέσει κάτω από" if buy else "ανέβει πάνω από"}</div>'
                f'<div class="v {stop_cls}">{esc(money(sig["stop"], currency))} '
                f'<small>({pct(sig["stop"] / price - 1)})</small></div></div>'
                f'<div><div class="k">Πιθανό κέρδος / ρίσκο</div>'
                f'<div class="v">{pct(sig["reward"], 1, sign=False)} / {pct(sig["risk"], 1, sign=False)}</div></div>'
                f'</div>')
    warn = ""
    if a != "HOLD" and sig["reward"] < sig["risk"]:
        warn += ('<div class="sig-warn">Το πιθανό κέρδος είναι μικρότερο από το ρίσκο: η τιμή έχει ήδη '
                 'πλησιάσει τον στόχο, οπότε μια κίνηση τώρα είναι λιγότερο ελκυστική.</div>')
    if sig["against_trend"]:
        warn += ('<div class="sig-warn">Προσοχή: το σήμα πάει αντίθετα στη μεγάλη τάση της μετοχής, '
                 'άρα είναι πιο ριψοκίνδυνο.</div>')
    if earnings_date and 0 <= (earnings_date - dt.date.today()).days <= 14:
        warn += (f'<div class="sig-warn">Αποτελέσματα στις {earnings_date:%d/%m}: τότε η μετοχή μπορεί να '
                 f'περάσει στόχο ή stop με ένα άλμα, και τα σήματα είναι λιγότερο αξιόπιστα.</div>')
    bt = backtest.get(a) if a != "HOLD" else None
    hist = ""
    if bt:
        hist = (f'<div class="sig-note"><b>Πόσο έπεφτε μέσα:</b> τα τελευταία {backtest["years"]} χρόνια η '
                f'{esc(ticker)} είχε {bt["n"]} μέρες με τεχνικό σήμα {"αγοράς" if a == "BUY" else "πώλησης"}. '
                f'Ο στόχος ήρθε πρώτος στο {bt["win"]:.0%}, το stop στο {bt["loss"]:.0%} '
                f'(και κανένα από τα δύο στο {bt["open"]:.0%}). Μετά από {backtest["horizon"]} συνεδριάσεις η '
                f'μετοχή είχε πάει προς τη σωστή κατεύθυνση στο {bt["hit"]:.0%} των περιπτώσεων.</div>')
    why = '<div class="sig-note"><b>Γιατί:</b> ' + " · ".join(esc(r) for r in sig["reasons"]) + "</div>"
    disclaimer = ('<div class="sig-note">Αυτόματο σήμα από δείκτες και στατιστική. '
                  'Δεν είναι επενδυτική συμβουλή.</div>')
    return f'<div class="sig {cls}">{head}{body}{warn}{why}{hist}{disclaimer}</div>'


@st.fragment(run_every=refresh)
def live_header():
    snap = live_snapshot()
    if "error" in snap:
        st.warning("Η live τιμή δεν είναι διαθέσιμη αυτή τη στιγμή (το Yahoo Finance δεν απάντησε). "
                   "Θα ξαναδοκιμάσω αυτόματα.")
        return
    price, prev = snap["price"], snap["prev_close"]
    meta, regular, intra = snap["meta"], snap["regular"], snap["intra"]
    chg = price - prev if prev else None
    chg_pct = chg / prev if prev else None
    cls = {"regular": "open", "pre": "ext", "post": "ext"}.get(snap["state"], "closed")

    arrow = "▲" if (chg or 0) >= 0 else "▼"
    chg_cls = "up" if (chg or 0) >= 0 else "down"
    chg_html = (f'<div class="chg {chg_cls}">{arrow} {num(abs(chg))} ({pct(chg_pct, 2)})</div>'
                if chg is not None else "")
    ext_html = ""
    if snap["ext_price"] is not None:
        ext_chg = snap["ext_price"] / price - 1
        lbl = "Pre-market" if snap["state"] == "pre" or (
            len(regular) and intra.index[-1] < regular.index[0]) else "After-hours"
        ext_html = (f'<div class="sub">{lbl}: <b>{esc(money(snap["ext_price"], currency))}</b> '
                    f'<span class="{"up" if ext_chg >= 0 else "down"}">({pct(ext_chg, 2)})</span></div>')
    hi = meta.get("regularMarketDayHigh") or (regular["High"].max() if len(regular) else None)
    lo = meta.get("regularMarketDayLow") or (regular["Low"].min() if len(regular) else None)
    vol = meta.get("regularMarketVolume") or (regular["Volume"].sum() if len(regular) else None)
    avg_vol = info.get("averageVolume")
    vol_sub = f"{num(vol / avg_vol, 1)}× του μέσου όρου" if vol and avg_vol else ""
    isig = an.intraday_signals(regular if len(regular) >= 15 else intra)
    intraday_score = an.score_signals(isig) if isig else None
    if isig:
        s = intraday_score
        day_html = (f'<div class="val" style="color:{score_color(s)}">{an.score_label(s)}</div>'
                    f'<div class="sub">{s:+.0f} / 100 · VWAP, EMA, RSI</div>')
    else:
        day_html = '<div class="val">–</div>'
    upd = pd.Timestamp(snap["updated"]).tz_convert(ATHENS)
    st.markdown(
        f'<div class="live">'
        f'<div class="stat"><div class="label">Τιμή</div><div class="price">{esc(money(price, currency))}</div>'
        f'{chg_html}{ext_html}</div>'
        f'<div class="stat"><div class="label">Εύρος ημέρας</div><div class="val">{num(lo)} – {num(hi)}</div>'
        f'<div class="sub">Χθεσινό κλείσιμο {num(prev)}</div></div>'
        f'<div class="stat"><div class="label">Όγκος</div><div class="val">{big(vol)}</div>'
        f'<div class="sub">{vol_sub}</div></div>'
        f'<div class="stat"><div class="label">Εικόνα ημέρας</div>{day_html}</div>'
        f'<div class="stat"><div class="label">Κατάσταση</div><span class="state {cls}">{esc(snap["state_label"])}</span>'
        f'<div class="sub">Τελευταία συναλλαγή {upd:%H:%M} · έλεγχος {dt.datetime.now(ATHENS):%H:%M:%S}<br>'
        f'ώρα Ελλάδας · ανανέωση κάθε {refresh} δευτ.</div></div>'
        f'</div>', unsafe_allow_html=True)
    current = snap["ext_price"] or price
    p5 = probs[5]["p_up"] if 5 in probs else None
    sig = an.trade_signal(ind, current, tech_score, p5, intraday_score)
    st.markdown(signal_card(sig, current), unsafe_allow_html=True)
    if snap["stale_since"]:
        mins = max(1, int((time.time() - snap["stale_since"]) // 60))
        st.caption(f"Το Yahoo Finance δεν απάντησε στην τελευταία προσπάθεια. Δείχνω την τιμή από πριν από "
                   f"{mins} λεπτ{'ό' if mins == 1 else 'ά'} και ξαναδοκιμάζω αυτόματα.")
    check_alerts(snap["ext_price"] or price)
    st.session_state["_live_price"] = price


live_header()


@st.fragment(run_every=FULL_REFRESH_SECS)
def periodic_full_refresh():
    """Κάθε 5 λεπτά ξαναϋπολογίζει όλη την ανάλυση (δείκτες, πιθανότητες, ειδήσεις)."""
    last = st.session_state.get("_full_run_at", 0)
    if time.time() - last >= FULL_REFRESH_SECS - 5:
        st.rerun()


st.session_state["_full_run_at"] = time.time()
periodic_full_refresh()

# ---------------------------------------------------------------------------
# Δευτερεύοντα δεδομένα (ειδήσεις, αναλυτές)
# ---------------------------------------------------------------------------

news_items, news_failed = safe("Ειδήσεις", data.news, ticker, company, default=([], ["Ειδήσεις"]))
failed_sources.extend(news_failed)
news = an.analyze_news(news_items)
# ---------------------------------------------------------------------------
# Σύνοψη
# ---------------------------------------------------------------------------

st.subheader("Με μια ματιά")
if probs:
    cols = st.columns(len(probs))
    for col, (h, p) in zip(cols, probs.items()):
        col.markdown(prob_card(an.HORIZONS[h], an.HORIZON_DAYS[h], p["p_up"]), unsafe_allow_html=True)
else:
    st.info("Το ιστορικό της μετοχής είναι πολύ μικρό για υπολογισμό πιθανοτήτων.")

st.write("")
k1, k2, k3 = st.columns(3)
k1.markdown(card("Τεχνική εικόνα (ημερήσια)", an.score_label(tech_score),
                 f"Βαθμός {tech_score:+.0f} από −100 έως +100 · {len(signals)} δείκτες",
                 score_color(tech_score)), unsafe_allow_html=True)
if news["items"]:
    lbl = an.sentiment_label(news["avg"])
    k2.markdown(card("Κλίμα ειδήσεων", lbl,
                     f"{news['pos']} θετικές · {news['neg']} αρνητικές · {news['neu']} ουδέτερες",
                     UP if lbl == "Θετική" else DOWN if lbl == "Αρνητική" else NEUTRAL), unsafe_allow_html=True)
else:
    k2.markdown(card("Κλίμα ειδήσεων", "–", "Δεν βρέθηκαν ειδήσεις"), unsafe_allow_html=True)

REC_TEXT = {"strong_buy": "Ισχυρή αγορά", "buy": "Αγορά", "hold": "Διακράτηση",
            "underperform": "Υποαπόδοση", "sell": "Πώληση", "strong_sell": "Ισχυρή πώληση"}
rec = REC_TEXT.get(info.get("recommendationKey") or "", None)
tgt = targets.get("mean") or info.get("targetMeanPrice")
last_close = float(ind["Close"].iloc[-1])
live_price = st.session_state.get("_live_price") or last_close
if rec or tgt:
    sub = []
    if tgt:
        sub.append(f"Μέσος στόχος {money(tgt, currency)} ({pct(tgt / live_price - 1)})")
    if info.get("numberOfAnalystOpinions"):
        sub.append(f"{info['numberOfAnalystOpinions']} αναλυτές")
    k3.markdown(card("Αναλυτές", rec or "–", " · ".join(sub),
                     UP if rec in ("Αγορά", "Ισχυρή αγορά") else DOWN if rec and "πώληση" in rec.lower() else None),
                unsafe_allow_html=True)
else:
    k3.markdown(card("Αναλυτές", "–", "Δεν υπάρχουν στοιχεία"), unsafe_allow_html=True)

if earnings_date:
    days_to = (earnings_date - dt.date.today()).days
    if 0 <= days_to <= 14:
        st.warning(f"Αποτελέσματα τριμήνου σε {days_to} ημέρες ({earnings_date:%d/%m}). Γύρω από τα "
                   "αποτελέσματα η μετοχή κινείται πολύ περισσότερο από το συνηθισμένο και οι "
                   "τεχνικοί δείκτες χάνουν μέρος της αξίας τους.")

# ---------------------------------------------------------------------------
# Καρτέλες
# ---------------------------------------------------------------------------

tab_live, tab_prob, tab_tech, tab_news, tab_co = st.tabs(
    ["Live γράφημα", "Πιθανότητες", "Τεχνική ανάλυση", "Ειδήσεις & φήμες", "Αναλυτές & εταιρεία"])


def to_athens_naive(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return idx.tz_convert(ATHENS).tz_localize(None)


with tab_live:
    bar_size = st.radio("Κερί", ["1 λεπτό", "5 λεπτά", "15 λεπτά"], horizontal=True, index=1, key="bar_size")

    @st.fragment(run_every=refresh)
    def live_chart():
        snap = live_snapshot()
        if "error" in snap:
            st.info("Το γράφημα θα εμφανιστεί μόλις απαντήσει το Yahoo Finance.")
            return
        intra = snap["intra"]
        rule = {"1 λεπτό": None, "5 λεπτά": "5min", "15 λεπτά": "15min"}[st.session_state.bar_size]
        bars = intra if rule is None else intra.resample(rule).agg(
            {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna(subset=["Close"])
        reg = snap["regular"]
        vw = an.vwap(reg) if len(reg) else None
        x = to_athens_naive(bars.index)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.78, 0.22], vertical_spacing=0.03)
        fig.add_trace(go.Candlestick(x=x, open=bars["Open"], high=bars["High"], low=bars["Low"], close=bars["Close"],
                                     name="Τιμή", increasing_line_color=UP, decreasing_line_color=DOWN), 1, 1)
        if vw is not None and vw.notna().any():
            fig.add_trace(go.Scatter(x=to_athens_naive(vw.index), y=vw, name="VWAP",
                                     line=dict(color=ORANGE, width=1.6)), 1, 1)
        if len(bars) >= 21:
            fig.add_trace(go.Scatter(x=x, y=an.ema(bars["Close"], 9), name="EMA 9",
                                     line=dict(color=ACCENT, width=1.1)), 1, 1)
            fig.add_trace(go.Scatter(x=x, y=an.ema(bars["Close"], 21), name="EMA 21",
                                     line=dict(color=PURPLE, width=1.1)), 1, 1)
        if snap["prev_close"]:
            fig.add_hline(y=snap["prev_close"], line=dict(color=NEUTRAL, dash="dot", width=1),
                          annotation_text="Χθεσινό κλείσιμο", annotation_position="top left", row=1, col=1)
        colors = np.where(bars["Close"] >= bars["Open"], UP, DOWN)
        fig.add_trace(go.Bar(x=x, y=bars["Volume"], marker_color=colors, name="Όγκος", opacity=0.6,
                             showlegend=False), 2, 1)
        if len(reg):
            r0, r1 = to_athens_naive(reg.index[[0, -1]])
            if x[0] < r0:
                fig.add_vrect(x0=x[0], x1=r0, fillcolor=NEUTRAL, opacity=0.08, line_width=0)
            if x[-1] > r1:
                fig.add_vrect(x0=r1, x1=x[-1], fillcolor=NEUTRAL, opacity=0.08, line_width=0)
        fig.update_layout(height=520, margin=dict(l=8, r=8, t=30, b=8), xaxis_rangeslider_visible=False,
                          legend=dict(orientation="h", y=1.06, x=0), hovermode="x unified", uirevision=ticker)
        fig.update_yaxes(title_text=currency, row=1, col=1)
        st.plotly_chart(fig, width="stretch", key="live_fig")
        st.caption("Ώρα Ελλάδας. Οι γκρι ζώνες είναι pre-market και after-hours. VWAP = μέση τιμή "
                   "σταθμισμένη με τον όγκο της ημέρας: πάνω της συνήθως κυριαρχούν οι αγοραστές.")
        isig = an.intraday_signals(reg if len(reg) >= 15 else intra)
        if isig:
            st.dataframe(pd.DataFrame([{"Δείκτης": s.name, "Τιμή": s.value, "Σήμα": verdict_text(s.verdict),
                                        "Τι σημαίνει": s.note} for s in isig]),
                         hide_index=True, width="stretch")

    live_chart()

# --- Πιθανότητες -----------------------------------------------------------
with tab_prob:
    st.markdown(
        "Η πιθανότητα βγαίνει από τρεις ανεξάρτητους τρόπους, που ο καθένας «τραβιέται» προς τον "
        "ιστορικό μέσο όρο της μετοχής όσο λιγότερο αξιόπιστος είναι:\n"
        "1. **Βάση**: πόσο συχνά ανέβηκε η μετοχή στο παρελθόν σε τέτοιο διάστημα.\n"
        "2. **Παρόμοιες καταστάσεις**: τι έκανε η μετοχή τις μέρες που οι δείκτες της ήταν όπως σήμερα.\n"
        "3. **Μοντέλο**: λογιστική παλινδρόμηση σε 15 δείκτες, ελεγμένη στο τελευταίο 25% του ιστορικού "
        "που δεν είχε δει. Αν δεν τα πήγε καλύτερα από μια σταθερή πρόβλεψη, δεν μετράει."
    )
    if not probs:
        st.info("Χρειάζεται μεγαλύτερο ιστορικό για υπολογισμό πιθανοτήτων.")
    for h, p in probs.items():
        with st.container(border=True):
            st.markdown(f"**{an.HORIZONS[h]}** ({an.HORIZON_DAYS[h]}) → "
                        f"<span class='up'>άνοδος {p['p_up']:.0%}</span> · "
                        f"<span class='down'>πτώση {p['p_down']:.0%}</span>", unsafe_allow_html=True)
            n_hist = f"{p['n_hist']:,}".replace(",", ".")
            lines = [f"- **Βάση:** {p['base']:.0%} · τόσες από τις {n_hist} περιόδους του ιστορικού "
                     "έκλεισαν πιο ψηλά"]
            a = p["analog"]
            if a:
                lines.append(f"- **Παρόμοιες καταστάσεις:** {a['p']:.0%} → {a['p_adj']:.0%} μετά τη διόρθωση · "
                             f"{a['state']} · {a['n']} φορές στο παρελθόν · διάμεση μεταβολή {pct(a['median'])}")
            m = p["model"]
            if m:
                verdict = (" · δεν ξεπέρασε την τύχη, άρα δεν μετράει" if m["weight"] == 0 else
                           f" · βάρος {m['weight']:.0%}")
                lines.append(f"- **Μοντέλο δεικτών:** {m['p']:.0%} → {m['p_adj']:.0%} · ευστοχία σε δεδομένα "
                             f"που δεν είχε δει {m['acc']:.0%} (σταθερή πρόβλεψη {m['acc_const']:.0%}){verdict}")
            st.markdown("\n".join(lines))
            if m and m["weight"] > 0:
                contrib = sorted(m["contrib"].items(), key=lambda kv: kv[1])
                downs = [an.FEATURE_TEXT[k] for k, v in contrib[:3] if v < 0]
                ups = [an.FEATURE_TEXT[k] for k, v in contrib[::-1][:3] if v > 0]
                st.caption(f"Σπρώχνουν προς τα πάνω: {', '.join(ups) or '–'} · "
                           f"προς τα κάτω: {', '.join(downs) or '–'}")

    st.subheader("Πόσο μπορεί να κινηθεί")
    if vranges:
        rows = []
        for h, v in vranges.items():
            rows.append({"Διάστημα": f"{an.HORIZONS[h]} ({an.HORIZON_DAYS[h]})",
                         "Εύρος με πιθανότητα 90%": f"{money(v['low'], currency)} – {money(v['high'], currency)}",
                         "Σε %": f"{pct(v['low_pct'])} έως {pct(v['high_pct'])}",
                         "Μεγάλη άνοδος": f"{v['p_up_big']:.0%} (πάνω από {pct(v['thr'], 0)})",
                         "Μεγάλη πτώση": f"{v['p_down_big']:.0%} (κάτω από {pct(-v['thr'], 0)})"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("Από 20.000 προσομοιώσεις με τις πραγματικές ημερήσιες κινήσεις του τελευταίου έτους, "
                   "χωρίς υπόθεση για την κατεύθυνση. Δείχνει πόσο «νευρική» είναι η μετοχή.")

    chains = safe("Δικαιώματα (options)", data.options, ticker, default=[])
    ov = an.options_view(chains, live_price) if chains else None
    if ov:
        st.subheader("Τι προεξοφλεί η αγορά δικαιωμάτων (options)")
        o1, o2, o3 = st.columns(3)
        if ov.get("move_pct"):
            o1.markdown(card(f"Αναμενόμενη κίνηση έως {ov['expiry']:%d/%m}",
                             f"±{pct(ov['move_pct'], 1, sign=False)}",
                             f"Τιμή straddle στο strike {num(ov['strike'])} · {ov['days']} ημέρες"),
                        unsafe_allow_html=True)
        if ov.get("iv"):
            o2.markdown(card("Τεκμαρτή μεταβλητότητα (IV)", pct(ov["iv"], 0, sign=False),
                             f"Εύρος 1σ: {money(ov['range_low'], currency)} – {money(ov['range_high'], currency)}"),
                        unsafe_allow_html=True)
        pcv = ov.get("pc_volume")
        if pcv is not None:
            mood = ("Περισσότερη ζήτηση για προστασία από πτώση" if pcv > 1 else
                    "Περισσότερα στοιχήματα για άνοδο" if pcv < 0.7 else "Ισορροπία")
            o3.markdown(card("Λόγος Puts / Calls (όγκος)", num(pcv, 2),
                             f"{mood} · ανοιχτές θέσεις {num(ov.get('pc_oi'), 2)}",
                             DOWN if pcv > 1 else UP if pcv < 0.7 else None), unsafe_allow_html=True)
        st.caption("Η αγορά δικαιωμάτων δεν προβλέπει κατεύθυνση, αλλά το μέγεθος της κίνησης που "
                   "περιμένουν οι επαγγελματίες. Εκτός ωραρίου συναλλαγών οι τιμές μπορεί να είναι παλιές.")

    st.info("Πώς να τα διαβάζεις: στις αμερικανικές μετοχές οι ιστορικές πιθανότητες ανόδου είναι "
            "συνήθως 51–56%. Μια εκτίμηση 60% είναι ήδη ισχυρή ένδειξη. Κανένα μοντέλο δεν προβλέπει "
            "ειδήσεις, αποτελέσματα ή απρόοπτα γεγονότα.")

# --- Τεχνική ανάλυση --------------------------------------------------------
with tab_tech:
    g1, g2 = st.columns([1, 2])
    with g1:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=tech_score, number={"valueformat": "+.0f"},
            title={"text": an.score_label(tech_score)},
            gauge={"axis": {"range": [-100, 100], "tickvals": [-100, -40, -15, 15, 40, 100]},
                   "bar": {"color": score_color(tech_score), "thickness": 0.3},
                   "steps": [{"range": [-100, -40], "color": "rgba(214,69,69,.30)"},
                             {"range": [-40, -15], "color": "rgba(214,69,69,.14)"},
                             {"range": [-15, 15], "color": "rgba(128,128,128,.14)"},
                             {"range": [15, 40], "color": "rgba(31,157,85,.14)"},
                             {"range": [40, 100], "color": "rgba(31,157,85,.30)"}]}))
        gauge.update_layout(height=260, margin=dict(l=20, r=20, t=50, b=10))
        st.plotly_chart(gauge, width="stretch")
        n_up = sum(s.verdict > 0.1 for s in signals)
        n_dn = sum(s.verdict < -0.1 for s in signals)
        st.caption(f"{n_up} ανοδικά, {n_dn} πτωτικά, {len(signals) - n_up - n_dn} ουδέτερα σήματα. "
                   "Η βαθμολογία δείχνει τι λένε οι δείκτες, όχι πιθανότητα. Η στατιστική τους αξία "
                   "ελέγχεται στην καρτέλα «Πιθανότητες».")
    with g2:
        sup, res = an.swing_levels(ind)
        piv = an.pivot_points(ind)
        st.markdown("**Στηρίξεις και αντιστάσεις**")
        lc1, lc2 = st.columns(2)
        with lc1:
            st.markdown("Αντιστάσεις (πάνω από την τιμή)")
            for lv, touches in res:
                st.markdown(f"<span class='down'>{esc(money(lv, currency))}</span> · {pct(lv / last_close - 1)} · "
                            f"{touches} αγγίγματα", unsafe_allow_html=True)
            if not res:
                st.caption("Η τιμή είναι πάνω από όλα τα πρόσφατα υψηλά.")
        with lc2:
            st.markdown("Στηρίξεις (κάτω από την τιμή)")
            for lv, touches in sup:
                st.markdown(f"<span class='up'>{esc(money(lv, currency))}</span> · {pct(lv / last_close - 1)} · "
                            f"{touches} αγγίγματα", unsafe_allow_html=True)
            if not sup:
                st.caption("Η τιμή είναι κάτω από όλα τα πρόσφατα χαμηλά.")
        if piv:
            st.caption("Pivot points ημέρας: " + " · ".join(f"{k} {num(v)}" for k, v in piv.items()))

    sig_df = pd.DataFrame([{"Ομάδα": s.group, "Δείκτης": s.name, "Τιμή": s.value,
                            "Σήμα": verdict_text(s.verdict), "Τι σημαίνει": s.note} for s in signals])
    if not sig_df.empty:
        def color_sig(v):
            return (f"color: {UP}; font-weight: 600" if v.startswith("▲") else
                    f"color: {DOWN}; font-weight: 600" if v.startswith("▼") else "")
        st.dataframe(sig_df.style.map(color_sig, subset=["Σήμα"]), hide_index=True,
                     width="stretch", height=35 * (len(sig_df) + 1) + 3)

    p1, p2 = st.columns([1, 2])
    period = p1.radio("Περίοδος", ["3 μήνες", "6 μήνες", "1 έτος", "2 έτη", "5 έτη"], index=2,
                      horizontal=True, key="period")
    overlays = p2.multiselect("Στο γράφημα", ["Κινητοί μέσοι", "Bollinger", "Ichimoku", "Parabolic SAR",
                                              "Στηρίξεις/αντιστάσεις"],
                              default=["Κινητοί μέσοι", "Bollinger"], key="overlays")
    n_days = {"3 μήνες": 63, "6 μήνες": 126, "1 έτος": 252, "2 έτη": 504, "5 έτη": 1260}[period]
    d = ind.tail(n_days)
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.025,
                        row_heights=[0.55, 0.13, 0.16, 0.16])
    fig.add_trace(go.Candlestick(x=d.index, open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"],
                                 name="Τιμή", increasing_line_color=UP, decreasing_line_color=DOWN), 1, 1)
    if "Bollinger" in overlays:
        fig.add_trace(go.Scatter(x=d.index, y=d["BB_up"], line=dict(color=NEUTRAL, width=0.8),
                                 name="Bollinger", legendgroup="bb"), 1, 1)
        fig.add_trace(go.Scatter(x=d.index, y=d["BB_low"], line=dict(color=NEUTRAL, width=0.8), fill="tonexty",
                                 fillcolor="rgba(138,143,152,.10)", showlegend=False, legendgroup="bb"), 1, 1)
    if "Ichimoku" in overlays:
        fig.add_trace(go.Scatter(x=d.index, y=d["SPAN_A"], line=dict(color=UP, width=0.6), name="Σύννεφο A",
                                 legendgroup="ichi"), 1, 1)
        fig.add_trace(go.Scatter(x=d.index, y=d["SPAN_B"], line=dict(color=DOWN, width=0.6), fill="tonexty",
                                 fillcolor="rgba(59,125,221,.10)", name="Σύννεφο B", legendgroup="ichi"), 1, 1)
    if "Κινητοί μέσοι" in overlays:
        for col, color in (("SMA20", ACCENT), ("SMA50", ORANGE), ("SMA200", PURPLE)):
            fig.add_trace(go.Scatter(x=d.index, y=d[col], name=col.replace("SMA", "SMA "),
                                     line=dict(color=color, width=1.3)), 1, 1)
    if "Parabolic SAR" in overlays:
        fig.add_trace(go.Scatter(x=d.index, y=d["PSAR"], mode="markers", name="SAR",
                                 marker=dict(size=3, color=np.where(d["PSAR_BULL"], UP, DOWN))), 1, 1)
    if "Στηρίξεις/αντιστάσεις" in overlays:
        sup, res = an.swing_levels(ind)
        for lv, _ in sup:
            fig.add_hline(y=lv, line=dict(color=UP, dash="dash", width=1), row=1, col=1)
        for lv, _ in res:
            fig.add_hline(y=lv, line=dict(color=DOWN, dash="dash", width=1), row=1, col=1)
    vcolors = np.where(d["Close"] >= d["Open"], UP, DOWN)
    fig.add_trace(go.Bar(x=d.index, y=d["Volume"], marker_color=vcolors, opacity=0.6, name="Όγκος",
                         showlegend=False), 2, 1)
    fig.add_trace(go.Scatter(x=d.index, y=d["RSI"], name="RSI", line=dict(color=PURPLE, width=1.2),
                             showlegend=False), 3, 1)
    for lvl in (30, 70):
        fig.add_hline(y=lvl, line=dict(color=NEUTRAL, dash="dot", width=1), row=3, col=1)
    fig.add_trace(go.Bar(x=d.index, y=d["MACD_hist"], marker_color=np.where(d["MACD_hist"] >= 0, UP, DOWN),
                         opacity=0.5, name="MACD ιστόγραμμα", showlegend=False), 4, 1)
    fig.add_trace(go.Scatter(x=d.index, y=d["MACD"], line=dict(color=ACCENT, width=1.1), name="MACD",
                             showlegend=False), 4, 1)
    fig.add_trace(go.Scatter(x=d.index, y=d["MACD_signal"], line=dict(color=ORANGE, width=1.1), name="Σήμα",
                             showlegend=False), 4, 1)
    fig.update_yaxes(title_text="Όγκος", row=2, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=3, col=1)
    fig.update_yaxes(title_text="MACD", row=4, col=1)
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    fig.update_layout(height=760, margin=dict(l=8, r=8, t=30, b=8), xaxis_rangeslider_visible=False,
                      legend=dict(orientation="h", y=1.04, x=0), hovermode="x unified")
    st.plotly_chart(fig, width="stretch")

# --- Ειδήσεις & φήμες ------------------------------------------------------
with tab_news:
    if news["topics"]:
        chips = "".join(f"<span class='chip {'rumor' if t.startswith('Φήμη') else 'topic'}'>{esc(t)} · {n}</span>"
                        for t, n in news["topics"].items())
        st.markdown(f"**Θέματα που κυκλοφορούν:** {chips}", unsafe_allow_html=True)

    twits = safe("StockTwits", data.stocktwits, ticker, default=[])
    if twits:
        tagged = [t for t in twits if t["sentiment"]]
        bulls = sum(t["sentiment"] == "Bullish" for t in tagged)
        with st.container(border=True):
            if tagged:
                share = bulls / len(tagged)
                st.markdown(f"**StockTwits (κοινωνικά δίκτυα επενδυτών):** {share:.0%} αισιόδοξοι "
                            f"από {len(tagged)} μηνύματα με δηλωμένη άποψη, στα τελευταία {len(twits)}.")
            else:
                st.markdown(f"**StockTwits:** {len(twits)} πρόσφατα μηνύματα, χωρίς δηλωμένη άποψη.")
            with st.expander("Τελευταία μηνύματα"):
                for t in twits[:12]:
                    tag = {"Bullish": "<span class='chip pos'>Αισιόδοξο</span>",
                           "Bearish": "<span class='chip neg'>Απαισιόδοξο</span>"}.get(t["sentiment"], "")
                    st.markdown(f"{tag} {esc(t['body'][:280])}<div class='news-meta'>@{esc(t['user'])} · "
                                f"{ago(t['time'])}</div>", unsafe_allow_html=True)
            st.caption("Απόψεις ιδιωτών επενδυτών, όχι επιβεβαιωμένες πληροφορίες.")

    f1, f2 = st.columns(2)
    topic_opts = ["Όλα"] + list(news["topics"].keys())
    topic_f = f1.selectbox("Θέμα", topic_opts, key="topic_filter")
    sent_f = f2.selectbox("Κλίμα", ["Όλα", "Θετική", "Αρνητική", "Ουδέτερη"], key="sent_filter")
    shown = [it for it in news["items"]
             if (topic_f == "Όλα" or topic_f in it["topics"]) and (sent_f == "Όλα" or it["label"] == sent_f)]
    if not shown:
        st.info("Δεν βρέθηκαν ειδήσεις με αυτά τα φίλτρα." if news["items"] else
                "Δεν ήταν δυνατή η λήψη ειδήσεων αυτή τη στιγμή.")
    for it in shown[:40]:
        cls = {"Θετική": "pos", "Αρνητική": "neg"}.get(it["label"], "neu")
        chips = f"<span class='chip {cls}'>{it['label']}</span>" + "".join(
            f"<span class='chip {'rumor' if t.startswith('Φήμη') else 'topic'}'>{esc(t)}</span>" for t in it["topics"])
        title = esc(it["title"])
        link = f"<a href='{esc(it['url'])}' target='_blank'>{title}</a>" if it.get("url") else title
        summary = f"<div style='font-size:.85rem;opacity:.85;margin-top:3px'>{esc(it['summary'][:260])}</div>" \
            if it.get("summary") else ""
        st.markdown(f"<div class='news-item'>{link}{summary}<div class='news-meta'>{esc(it['publisher'])} · "
                    f"{ago(it['time'])}</div><div>{chips}</div></div>", unsafe_allow_html=True)
    st.caption("Το κλίμα κάθε είδησης βγαίνει αυτόματα από τη διατύπωση του τίτλου (λεξικό VADER με "
               "χρηματοοικονομικούς όρους) και μπορεί να κάνει λάθος. «Φήμη / ανεπιβεβαίωτο» σημαίνει "
               "διατυπώσεις όπως «σύμφωνα με πηγές», «εξετάζει», «σε συζητήσεις».")

    filings = safe("Ανακοινώσεις SEC", data.sec_filings, ticker, default=[])
    if filings:
        with st.expander("Επίσημες ανακοινώσεις στην SEC (8-K, 10-Q, Form 4 κ.ά.)"):
            for f in filings:
                title = esc(f.get("title") or f.get("type"))
                link = f"<a href='{esc(f['url'])}' target='_blank'>{title}</a>" if f.get("url") else title
                st.markdown(f"**{esc(f.get('type'))}** · {f.get('date')} · {link}", unsafe_allow_html=True)

# --- Αναλυτές & εταιρεία ---------------------------------------------------
with tab_co:
    a1, a2 = st.columns([1, 1.4])
    with a1:
        st.markdown("**Συστάσεις αναλυτών**")
        recs = an_data.get("recs")
        if isinstance(recs, pd.DataFrame) and not recs.empty:
            r0 = recs.iloc[0]
            labels = [("strongBuy", "Ισχυρή αγορά", "#147a42"), ("buy", "Αγορά", UP), ("hold", "Διακράτηση", NEUTRAL),
                      ("sell", "Πώληση", DOWN), ("strongSell", "Ισχυρή πώληση", "#a32f2f")]
            rf = go.Figure()
            for key, lbl, color in labels:
                rf.add_trace(go.Bar(y=["Τώρα"], x=[int(r0.get(key, 0) or 0)], name=lbl, orientation="h",
                                    marker_color=color, text=[int(r0.get(key, 0) or 0)], textposition="inside"))
            rf.update_layout(barmode="stack", height=150, margin=dict(l=8, r=8, t=8, b=8),
                             legend=dict(orientation="h", y=-0.3), xaxis=dict(visible=False),
                             yaxis=dict(visible=False))
            st.plotly_chart(rf, width="stretch")
        else:
            st.caption("Δεν υπάρχουν συστάσεις.")
        if targets:
            lo_t, mean_t, hi_t = targets.get("low"), targets.get("mean"), targets.get("high")
            if mean_t:
                st.markdown(md(f"Στόχος τιμής: χαμηλός **{money(lo_t, currency)}**, μέσος "
                               f"**{money(mean_t, currency)}** ({pct(mean_t / live_price - 1)}), "
                               f"υψηλός **{money(hi_t, currency)}**"))
    with a2:
        st.markdown("**Αναβαθμίσεις και υποβαθμίσεις (τελευταίοι 3 μήνες)**")
        ch = an_data.get("changes")
        if isinstance(ch, pd.DataFrame) and not ch.empty:
            ch = ch.copy()
            ch.index = pd.to_datetime(ch.index)
            if ch.index.tz is not None:
                ch.index = ch.index.tz_localize(None)
            recent = ch[ch.index >= pd.Timestamp.now() - pd.Timedelta(days=90)].sort_index(ascending=False)
            act = {"up": "▲ Αναβάθμιση", "down": "▼ Υποβάθμιση", "main": "Διατήρηση", "init": "Έναρξη κάλυψης",
                   "reit": "Επανάληψη"}
            if recent.empty:
                st.caption("Καμία αλλαγή τους τελευταίους 3 μήνες.")
            else:
                def col(name):
                    return (recent[name] if name in recent else pd.Series("", index=recent.index)).fillna("").astype(str)

                tbl = pd.DataFrame({
                    "Ημερομηνία": recent.index.strftime("%d/%m"),
                    "Εταιρεία": col("Firm").to_numpy(),
                    "Ενέργεια": col("Action").map(lambda a: act.get(a, a)).to_numpy(),
                    "Σύσταση": (col("FromGrade") + " → " + col("ToGrade")).str.strip(" →").to_numpy(),
                })
                if "currentPriceTarget" in recent:
                    tbl["Στόχος"] = [num(v) if v and not pd.isna(v) else "–" for v in recent["currentPriceTarget"]]
                st.dataframe(tbl.head(20), hide_index=True, width="stretch")
        else:
            st.caption("Δεν υπάρχουν στοιχεία.")

    st.markdown("**Βασικά στοιχεία**")
    stats = [
        ("Κεφαλαιοποίηση", big(info.get("marketCap"), currency)),
        ("P/E (12μηνο)", num(info.get("trailingPE"), 1)),
        ("P/E (εκτίμηση)", num(info.get("forwardPE"), 1)),
        ("Beta", num(info.get("beta"), 2)),
        ("Short (% free float)", pct(info.get("shortPercentOfFloat"), 1, sign=False)
         if info.get("shortPercentOfFloat") else "–"),
        ("Ημέρες κάλυψης short", num(info.get("shortRatio"), 1)),
        ("Θεσμικοί επενδυτές", pct(info.get("heldPercentInstitutions"), 0, sign=False)
         if info.get("heldPercentInstitutions") else "–"),
        ("Επόμενα αποτελέσματα", f"{earnings_date:%d/%m/%Y}" if earnings_date else "–"),
    ]
    for start in range(0, len(stats), 4):
        for col_, (k, v) in zip(st.columns(4), stats[start:start + 4]):
            col_.markdown(card(k, v), unsafe_allow_html=True)
        st.write("")
    short_pf = info.get("shortPercentOfFloat")
    if short_pf and short_pf > 0.15:
        st.warning(f"Υψηλό short interest ({pct(short_pf, 0, sign=False)} των μετοχών): πολλοί ποντάρουν "
                   "σε πτώση. Μπορεί όμως να προκαλέσει και απότομη άνοδο (short squeeze).")

    ins = safe("Συναλλαγές insiders", data.insiders, ticker)
    if isinstance(ins, pd.DataFrame) and not ins.empty and "Text" in ins:
        ins = ins.copy()
        if "Start Date" in ins:
            ins["Start Date"] = pd.to_datetime(ins["Start Date"], errors="coerce")
            ins = ins[ins["Start Date"] >= pd.Timestamp.now() - pd.Timedelta(days=180)]
        text = ins["Text"].fillna("").str.lower()
        sales, buys = ins[text.str.contains("sale")], ins[text.str.contains("purchase")]
        def total(df_):
            return f" ({big(df_['Value'].sum(), currency)})" if "Value" in df_ and len(df_) else ""

        st.markdown(md(f"**Συναλλαγές στελεχών (insiders), τελευταίοι 6 μήνες:** "
                       f"{len(buys)} αγορές{total(buys)} · {len(sales)} πωλήσεις{total(sales)}"))
        st.caption("Οι πωλήσεις στελεχών είναι συχνές (φόροι, διαφοροποίηση). Οι αγορές με δικά τους "
                   "χρήματα θεωρούνται πιο σημαντικό σήμα.")
        with st.expander("Λεπτομέρειες"):
            show = [c for c in ("Start Date", "Insider", "Position", "Text", "Shares", "Value") if c in ins]
            st.dataframe(ins[show].head(30), hide_index=True, width="stretch")

    if info.get("longBusinessSummary"):
        with st.expander("Τι κάνει η εταιρεία"):
            st.write(info["longBusinessSummary"])

if failed_sources:
    st.caption("Δεν ήταν διαθέσιμα αυτή τη φορά: " + ", ".join(sorted(set(failed_sources))) +
               ". Θα ξαναδοκιμάσω στην επόμενη ανανέωση.")
