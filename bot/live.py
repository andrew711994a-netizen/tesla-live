"""Καθημερινή εκτέλεση: ελέγχει σήματα και ανοίγει/κλείνει θέσεις στην Capital.com.

Τρέχει κάθε εργάσιμη μέρα λίγο μετά το άνοιγμα της Νέας Υόρκης, με τα σήματα
του χθεσινού κλεισίματος (όπως το backtest και το TradingView).

Μεταβλητές περιβάλλοντος:
  CAPITAL_API_KEY, CAPITAL_IDENTIFIER, CAPITAL_PASSWORD   στοιχεία API (GitHub Secrets)
  CAPITAL_ENV      demo (προεπιλογή) ή live
  ALLOW_LIVE       πρέπει να είναι I_UNDERSTAND_THE_RISK για live λογαριασμό
  DRY_RUN          true (προεπιλογή): μόνο γράφει τι θα έκανε, χωρίς εντολές
                   (στον live χρειάζεται επιπλέον "live": {"orders": true} στο config.json, αλλιώς πάντα δοκιμή)
  BOT_ENABLED      false: δεν κάνει τίποτα (κουμπί διακοπής)
  NTFY_TOPIC       θέμα στο ntfy.sh για ειδοποιήσεις στο κινητό
  FORCE            true: αγνοεί τον έλεγχο ώρας (για χειροκίνητη δοκιμή)
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import math
import os
import struct
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from pandas.tseries.holiday import (AbstractHolidayCalendar, GoodFriday, Holiday, USLaborDay,
                                    USMartinLutherKingJr, USMemorialDay, USPresidentsDay,
                                    USThanksgivingDay, nearest_workday, sunday_to_monday)

from .capital import Capital, CapitalError
from .data import daily, earnings_times
from .notify import notify
from .strategy import Params, event_sessions, market_ok, signals

NY = ZoneInfo("America/New_York")
CONFIG = json.loads((Path(__file__).parent / "config.json").read_text(encoding="utf-8"))
JOURNAL = Path(__file__).resolve().parent.parent / "journal"


class NYSEHolidays(AbstractHolidayCalendar):
    """Οι αργίες του χρηματιστηρίου της Νέας Υόρκης (χωρίς τις έκτακτες, π.χ. εθνικό πένθος)."""
    rules = [
        Holiday("Πρωτοχρονιά", month=1, day=1, observance=sunday_to_monday),
        USMartinLutherKingJr, USPresidentsDay, GoodFriday, USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01", observance=nearest_workday),
        Holiday("Ημέρα Ανεξαρτησίας", month=7, day=4, observance=nearest_workday),
        USLaborDay, USThanksgivingDay,
        Holiday("Χριστούγεννα", month=12, day=25, observance=nearest_workday),
    ]


def market_holiday(day: pd.Timestamp) -> bool:
    return len(NYSEHolidays().holidays(day, day)) > 0


# ── Ημερολόγιο (χωρίς ποσά: μόνο σύμβολα, τιμές και ποσοστά) ──
def load_json(f: Path) -> dict:
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


def load_state() -> dict:
    data = load_json(JOURNAL / "positions.json")
    if "positions" not in data:          # παλιά μορφή: σκέτο λεξικό θέσεων
        data = {"positions": data, "peak_equity": None}
    return data


def save_state(data: dict) -> None:
    JOURNAL.mkdir(parents=True, exist_ok=True)
    (JOURNAL / "positions.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def append_csv(name: str, header: str, row: list) -> None:
    JOURNAL.mkdir(parents=True, exist_ok=True)
    f = JOURNAL / name
    new = not f.exists()
    with f.open("a", encoding="utf-8") as fh:
        if new:
            fh.write(header + "\n")
        fh.write(",".join(str(x).replace(",", ";").replace("\r", " ").replace("\n", " ") for x in row) + "\n")


def log_trade(entry: dict, exit_date, exit_px: float, reason: str, env: str) -> None:
    res = exit_px / entry["entry"] - 1
    r = (exit_px - entry["entry"]) / max(entry["entry"] - entry["stop"], 1e-9)
    append_csv("trades.csv", "open_date,close_date,symbol,entry,exit,result_pct,r_multiple,reason,env",
               [entry["date"], exit_date, entry["symbol"], f"{entry['entry']:.2f}", f"{exit_px:.2f}",
                f"{res * 100:+.2f}", f"{r:+.2f}", reason, env])


# Το αποθετήριο είναι δημόσιο: ποσά του πραγματικού λογαριασμού (το υψηλότερο σημείο για το φρένο και η αξία στην
# έναρξη) αποθηκεύονται κρυπτογραφημένα με το κλειδί API, που υπάρχει μόνο στα GitHub Secrets.
def _mac(key: bytes, msg: bytes) -> bytes:
    return hmac.new(key, msg, hashlib.sha256).digest()


def seal(x: float) -> str:
    key, nonce = os.environ.get("CAPITAL_API_KEY", "").encode(), os.urandom(8)
    body = bytes(a ^ b for a, b in zip(struct.pack(">d", x), _mac(key, b"amount" + nonce)))
    return base64.urlsafe_b64encode(nonce + body + _mac(key, nonce + body)[:6]).decode()


def unseal(token: str | None) -> float | None:
    """None αν λείπει ή αν κρυπτογραφήθηκε με άλλο κλειδί (π.χ. μετά από νέο κλειδί API)."""
    key = os.environ.get("CAPITAL_API_KEY", "").encode()
    try:
        raw = base64.urlsafe_b64decode((token or "").encode())
    except ValueError:
        return None
    nonce, body, tag = raw[:8], raw[8:16], raw[16:]
    if len(raw) != 22 or not hmac.compare_digest(tag, _mac(key, nonce + body)[:6]):
        return None
    return struct.unpack(">d", bytes(a ^ b for a, b in zip(body, _mac(key, b"amount" + nonce))))[0]


VIRTUAL = "virtual.json"


def update_virtual(frames: dict, p: Params, rk: dict, state: dict, sigs: dict, market_is_ok: bool,
                   today: pd.Timestamp, earn: dict | None = None) -> str:
    """Εικονικό ημερολόγιο: κάθε σήμα από την πρώτη μέρα του bot σαν να αγοράστηκε, χωρίς όριο θέσεων ή κεφαλαίου.

    Καμία εντολή, μόνο καταγραφή, για να κριθεί η στρατηγική με πολύ περισσότερες συναλλαγές. Ξαναϋπολογίζεται
    κάθε μέρα από τα ημερήσια κεριά με τους κανόνες του backtest (είσοδος στο άνοιγμα, stop/στόχος, 20 μέρες,
    σπασμένη τάση, κόστη), οπότε δεν χάνει μέρες ούτε μετρά κάτι δύο φορές.
    """
    from .backtest import Costs, Risk, every_signal, fills, run as replay, trade_stats

    market = frames.get(CONFIG["market_symbol"])
    days_log = JOURNAL / "daily.csv"
    if market is None or not days_log.exists():
        return "Εικονικό ημερολόγιο: δεν ενημερώθηκε σήμερα (λείπουν δεδομένα αγοράς)."
    first = pd.Timestamp(pd.read_csv(days_log)["date"].min())   # πρώτη μέρα του bot
    before = market.index[market.index < first]
    start = before[-1] if len(before) else first               # τα σήματα εκείνου του κλεισίματος μπαίνουν την πρώτη μέρα
    data = {s: frames[s] for s in CONFIG["universe"] if s in frames}
    risk = every_signal(Risk(rk["risk_per_trade"], use_market_filter=rk["use_market_filter"]))
    ec = CONFIG.get("earnings", {})
    events = {k: v for k, v in (earn or {}).items() if v is not None}
    trades = replay(data, p, risk, Costs(**CONFIG["costs"]), str(start.date()), market=market, events=events,
                    block=int(ec.get("block_sessions", 0)), exit_before=bool(ec.get("exit_before", False)),
                    **fills())["trades"]

    real = {(v["symbol"], v["date"]) for v in state.values()}  # ό,τι αγόρασε πραγματικά το bot
    if (JOURNAL / "trades.csv").exists():
        t = pd.read_csv(JOURNAL / "trades.csv", dtype=str)
        real |= set(zip(t["symbol"], t["open_date"]))

    def is_real(t: dict) -> bool:
        return (t["symbol"], f"{t['entry_date']:%Y-%m-%d}") in real

    def net(t: dict) -> float:   # % μετά τα κόστη
        return t["pnl"] / (t["qty"] * t["entry"]) * 100

    done = sorted((t for t in trades if t["reason"] != "Τέλος"), key=lambda t: (t["exit_date"], t["symbol"]))
    still = sorted((t for t in trades if t["reason"] == "Τέλος"), key=lambda t: (t["entry_date"], t["symbol"]))
    JOURNAL.mkdir(parents=True, exist_ok=True)
    lines = ["open_date,close_date,symbol,entry,exit,result_pct,r_multiple,reason,real"]
    lines += [f"{t['entry_date']:%Y-%m-%d},{t['exit_date']:%Y-%m-%d},{t['symbol']},{t['entry']:.2f},{t['exit']:.2f},"
              f"{net(t):+.2f},{t['r']:+.2f},{t['reason']},{'ναι' if is_real(t) else 'όχι'}" for t in done]
    (JOURNAL / "virtual_trades.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    holding = {t["symbol"] for t in still}
    allowed = market_is_ok or not rk["use_market_filter"]
    new = sorted(s for s in data if allowed and s in sigs and bool(sigs[s]["long_sig"]) and s not in holding)
    book = {"updated": str(today.date()), "since": str(first.date()),
            "missing": sorted(set(CONFIG["universe"]) - set(data)),
            "closed": trade_stats(done),
            "open": [{"symbol": t["symbol"], "date": f"{t['entry_date']:%Y-%m-%d}", "entry": round(t["entry"], 2),
                      "last_close": round(t["exit"], 2), "result_pct": round(net(t), 2), "r": round(t["r"], 2),
                      "real": is_real(t)} for t in still],
            "signals_today": new}
    (JOURNAL / VIRTUAL).write_text(json.dumps(book, ensure_ascii=False, indent=1), encoding="utf-8")
    st = book["closed"]
    msg = f"Εικονικό ημερολόγιο (όλα τα σήματα από {first:%d/%m}): {len(done)} κλεισμένες"
    if done:
        msg += f" (επιτυχία {st['win']:.0%}, μέσο R {st['avg_r']:+.2f})"
    msg += f" · {len(still)} ανοιχτές · νέα σήματα σήμερα: {', '.join(new) or 'κανένα'}"
    if book["missing"]:
        msg += f" · ⚠️ χωρίς δεδομένα: {', '.join(book['missing'])}"
    return msg


CLOSE_REASONS = {"SL": "Stop", "TP": "Στόχος", "USER": "Χειροκίνητα"}


def broker_close(cap, epic: str, opened: str, deal_id: str | None = None) -> tuple[float, str, str] | None:
    """Η πραγματική τιμή, αιτία και μέρα κλεισίματος μιας θέσης από το ιστορικό της Capital.com.

    Το ιστορικό δέχεται διαστήματα έως μίας μέρας, άρα ψάχνουμε προς τα πίσω, ανά 23 ώρες, ως τη μέρα αγοράς.
    Αν ξέρουμε τον αριθμό της θέσης (deal_id), κρατάμε μόνο το δικό της κλείσιμο.
    """
    end = pd.Timestamp.now(tz="UTC").tz_localize(None)
    opened_ts = pd.Timestamp(opened)
    stop_at = max(opened_ts, end - pd.Timedelta(days=40))
    while end > stop_at:
        start = max(end - pd.Timedelta(hours=23), opened_ts)
        acts = cap.activity(start.strftime("%Y-%m-%dT%H:%M:%S"), end.strftime("%Y-%m-%dT%H:%M:%S"))
        closes = [a for a in acts if a.get("epic") == epic and a.get("type") == "POSITION"
                  and (a.get("details") or {}).get("direction") == "SELL" and (a.get("details") or {}).get("level")
                  and pd.Timestamp(a.get("dateUTC") or a.get("date") or end) >= opened_ts
                  and (deal_id is None or a.get("dealId") == deal_id)]
        if closes:
            last = max(closes, key=lambda a: a.get("dateUTC", ""))
            src = last.get("source", "")
            when = pd.Timestamp(last.get("dateUTC") or last.get("date")).tz_localize("UTC").tz_convert(NY)
            return float(last["details"]["level"]), CLOSE_REASONS.get(src, src or "Άγνωστο"), str(when.date())
        end = start
    return None


def trading_sessions(past: pd.DatetimeIndex, today: pd.Timestamp, days: int = 150) -> pd.DatetimeIndex:
    """Οι συνεδριάσεις: οι περασμένες από τα δεδομένα και οι επόμενες εργάσιμες χωρίς τις αργίες της Νέας Υόρκης."""
    future = pd.bdate_range(today, today + pd.Timedelta(days=days))
    hol = set(NYSEHolidays().holidays(today, today + pd.Timedelta(days=days)))
    return past[past < today].union(pd.DatetimeIndex([d for d in future if d not in hol]))


def earnings_calendar(symbols, sessions: pd.DatetimeIndex) -> dict[str, list[pd.Timestamp] | None]:
    """Οι συνεδριάσεις με αποτελέσματα ανά μετοχή· None αν δεν βρέθηκαν ημερομηνίες."""
    out: dict[str, list[pd.Timestamp] | None] = {}
    for sym in symbols:
        try:
            out[sym] = event_sessions(earnings_times(sym, limit=25, tries=2), sessions)
        except Exception:
            out[sym] = None
    return out


def sessions_to_results(events: list[pd.Timestamp] | None, sessions: pd.DatetimeIndex,
                        today: pd.Timestamp) -> float | None:
    """Σε πόσες συνεδριάσεις από σήμερα έρχονται τα επόμενα αποτελέσματα (1 = αύριο).

    inf: δεν υπάρχει επόμενη ημερομηνία αλλά τα τελευταία αποτελέσματα βγήκαν πρόσφατα (≤ 75 μέρες), άρα τα
    επόμενα αργούν. None: άγνωστο (για σιγουριά, καμία αγορά).
    """
    if events is None:
        return None
    i = int(sessions.searchsorted(today))
    later = [e for e in events if e > today]
    if later:
        return int(sessions.searchsorted(later[0])) - i
    past = [e for e in events if e <= today]
    return float("inf") if past and (today - past[-1]).days <= 75 else None


def env_flag(name: str, default: bool) -> bool:
    v = os.environ.get(name, "").strip().lower()
    return default if v == "" else v in ("1", "true", "yes", "on")


def dry_flag() -> bool:
    """Λειτουργία δοκιμής: σβήνει ΜΟΝΟ με ρητό false/0/no/off· οποιοδήποτε άλλο (ή λάθος) γράψιμο = δοκιμή.

    Στον πραγματικό λογαριασμό χρειάζεται επιπλέον "live": {"orders": true} στο config.json.
    """
    live = os.environ.get("CAPITAL_ENV", "").strip().lower() == "live"
    if live and CONFIG.get("live", {}).get("orders") is not True:
        return True
    return os.environ.get("DRY_RUN", "").strip().lower() not in ("0", "false", "no", "off")


def fx_rate(src: str, dst: str) -> float:
    if src == dst:
        return 1.0
    df = daily(f"{src}{dst}=X", start=(pd.Timestamp.today() - pd.Timedelta(days=10)).strftime("%Y-%m-%d"))
    return float(df["Close"].iloc[-1])


def round_size(qty: float, step: float, min_size: float) -> float:
    if step <= 0:
        step = min_size or 1.0
    size = math.floor(qty / step) * step
    decimals = max(0, -int(math.floor(math.log10(step)))) if step < 1 else 0
    size = round(size, decimals)
    return size if size >= min_size else 0.0


LOG: list[str] = []
SKIP = -1  # εκτός ωραρίου ή έτρεξε ήδη σήμερα: το σημείωμα της κανονικής εκτέλεσης μένει ως έχει
MIN_POSITION = 0.02  # όπως το backtest: καμία θέση κάτω από το 2% του λογαριασμού
TRADE_AT = (9, 31)   # αγορές/κλεισίματα 1 λεπτό μετά το άνοιγμα της Νέας Υόρκης (≈ τιμή ανοίγματος, όπως το backtest)
MAX_WAIT_MIN = 35    # το GitHub ξεκινά νωρίτερα και το bot περιμένει, γιατί οι προγραμματισμένες εκτελέσεις αργούν
DRY_MARK = "dry_run.json"
HISTORY_START = "2014-01-01"  # όπως το backtest: με όλο το ιστορικό οι μέσοι όροι (EMA) βγαίνουν ακριβώς ίδιοι


def write_run_log(status: str) -> None:
    """Σύντομο σημείωμα της τελευταίας εκτέλεσης (χωρίς ποσά ή κλειδιά), για τον καθημερινό έλεγχο.

    Οι δοκιμές γράφουν σε χωριστό αρχείο, για να μη σβήνουν το σημείωμα της πραγματικής εκτέλεσης.
    """
    JOURNAL.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(NY).strftime("%Y-%m-%d %H:%M")
    dry = dry_flag()
    name, kind = ("last_dry_run.md", "δοκιμή") if dry else ("last_run.md", "εκτέλεση")
    (JOURNAL / name).write_text(f"# Τελευταία {kind} {stamp} (Νέα Υόρκη) · {status}\n\n" +
                                "\n".join(f"- {line}" for line in LOG) + "\n", encoding="utf-8")


def main() -> int:
    try:
        code = run()
    except Exception as e:  # απρόβλεπτο σφάλμα: καταγραφή και ειδοποίηση
        import traceback
        LOG.append(f"ΣΦΑΛΜΑ: {type(e).__name__}: {e}")
        LOG.extend(line.strip() for line in traceback.format_exc().strip().splitlines()[-3:-1])
        print("\n".join(LOG))
        notify("Bot: απρόβλεπτο σφάλμα", "\n".join(LOG[-5:]))
        code = 1
    if code == SKIP:
        return 0
    write_run_log("OK" if code == 0 else "ΣΦΑΛΜΑ")
    return code


def run() -> int:
    log = LOG

    def say(msg: str):
        print(msg)
        log.append(msg)

    if not env_flag("BOT_ENABLED", True):
        say("Το bot είναι απενεργοποιημένο (BOT_ENABLED=false).")
        return 0

    env = os.environ.get("CAPITAL_ENV", "demo").strip().lower() or "demo"
    if env == "live":   # χωριστό ημερολόγιο για τον πραγματικό λογαριασμό
        global JOURNAL
        if JOURNAL.name != "live":
            JOURNAL = JOURNAL / "live"
    if env == "live" and os.environ.get("ALLOW_LIVE") != "I_UNDERSTAND_THE_RISK":
        say("Άρνηση: ο live λογαριασμός χρειάζεται ALLOW_LIVE=I_UNDERSTAND_THE_RISK.")
        return 1
    names = ["CAPITAL_API_KEY", "CAPITAL_IDENTIFIER", "CAPITAL_PASSWORD"]
    if env == "live" and not any(os.environ.get(n, "").strip() for n in names):
        print("Δεν έχουν οριστεί ακόμα τα στοιχεία του πραγματικού λογαριασμού (CAPITAL_LIVE_*). Τίποτα να κάνω.")
        return SKIP
    dry = dry_flag()
    force = env_flag("FORCE", False)
    now = datetime.now(NY)
    trade_at = now.replace(hour=TRADE_AT[0], minute=TRADE_AT[1], second=0, microsecond=0)
    if not force:
        if now.weekday() >= 5 or (trade_at - now).total_seconds() > MAX_WAIT_MIN * 60 or now.hour >= 11:
            print(f"Εκτός παραθύρου εκτέλεσης ({now:%a %H:%M} Νέα Υόρκη). Τίποτα να κάνω.")
            return SKIP
        if market_holiday(pd.Timestamp(now.date())):
            print("Αργία στο χρηματιστήριο της Νέας Υόρκης σήμερα. Τίποτα να κάνω.")
            return SKIP
        mark = load_json(JOURNAL / DRY_MARK) if dry else load_state()
        if mark.get("last_run") == str(now.date()):
            # Προγραμματισμένες εκτελέσεις κάθε 15 λεπτά: συναλλαγές μία φορά τη μέρα.
            # Αν η πρώτη απέτυχε, η επόμενη λειτουργεί ως επανάληψη.
            print("Το bot έτρεξε ήδη σήμερα. Τίποτα να κάνω.")
            return SKIP

    p = Params.from_dict(CONFIG["strategy"])
    rk = CONFIG["risk"]
    universe: dict[str, str] = CONFIG["universe"]
    mode = f"{env.upper()}{' · ΔΟΚΙΜΗ (χωρίς εντολές)' if dry else ''}"
    say(f"Έναρξη {now:%d/%m %H:%M} Νέα Υόρκη · {mode}")

    # 1) Δεδομένα και σήματα από το τελευταίο ολοκληρωμένο κλείσιμο
    today = pd.Timestamp(now.date())
    start = HISTORY_START
    sigs: dict[str, pd.Series] = {}
    bars: dict[str, pd.DatetimeIndex] = {}
    frames: dict[str, pd.DataFrame] = {}
    warnings: list[str] = []                               # προβλήματα δεδομένων: πάνε και στην ειδοποίηση
    market_is_ok = False
    for sym in set(universe) | {CONFIG["market_symbol"]}:
        try:
            df = daily(sym, start=start)
            df = df[df.index < today]                      # χωρίς το μισό κερί της σημερινής μέρας
            sg = signals(df, p)
            sigs[sym] = sg.iloc[-1]
            bars[sym] = df.index                           # για μέτρηση ημερών σε θέση
            frames[sym] = df
            if sym == CONFIG["market_symbol"]:
                market_is_ok = bool(market_ok(df, p).iloc[-1])
        except Exception as e:
            say(f"⚠️ {sym}: δεν ήρθαν δεδομένα ({e})")
    n_all = len(set(universe) | {CONFIG["market_symbol"]})
    if len(frames) < n_all:
        warnings.append(f"⚠️ Δεδομένα Yahoo μόνο για {len(frames)}/{n_all} σύμβολα.")
    # Μόνο φρέσκα δεδομένα: χωρίς το τελευταίο κερί της αγοράς, τα σήματα μιας μετοχής είναι παλιά
    if frames:
        last_day = max(df.index[-1] for df in frames.values())
        stale = sorted(s for s, df in frames.items() if df.index[-1] < last_day)
        for sym in stale:
            say(f"⚠️ {sym}: παλιά δεδομένα (τελευταίο κερί {frames[sym].index[-1]:%d/%m}), χωρίς σήμα σήμερα.")
            sigs.pop(sym, None)
        if stale:
            warnings.append(f"⚠️ Παλιά δεδομένα (αγνοήθηκαν σήμερα): {', '.join(stale)}.")
    if CONFIG["market_symbol"] not in sigs:
        market_is_ok = False
        say("Χωρίς δεδομένα αγοράς, δεν ανοίγω νέες θέσεις σήμερα.")
        warnings.append("⚠️ Χωρίς δεδομένα για τον Nasdaq-100: καμία νέα θέση σήμερα.")

    # Ημερολόγιο αποτελεσμάτων: καμία αγορά λίγο πριν τα αποτελέσματα, κλείσιμο την προηγούμενη συνεδρίαση
    earn_cfg = CONFIG.get("earnings", {})
    block = int(earn_cfg.get("block_sessions", 0))
    exit_before = bool(earn_cfg.get("exit_before", False))
    sessions = trading_sessions(frames[CONFIG["market_symbol"]].index if CONFIG["market_symbol"] in frames
                                else pd.DatetimeIndex([]), today)
    # Μόνο οι μετοχές που χρειάζονται σήμερα (σήματα αγοράς, και οι ανοιχτές αν κλείνουμε πριν τα αποτελέσματα)·
    # οι υπόλοιπες κατεβαίνουν μετά τις συναλλαγές, για το εικονικό ημερολόγιο
    need = [s for s in universe if s in sigs and bool(sigs[s]["long_sig"])]
    if exit_before:
        need = list(universe)
    earn = earnings_calendar(need, sessions) if (block or exit_before) else {}
    no_dates = sorted(s for s, ev in earn.items() if ev is None)
    if no_dates:
        say(f"⚠️ Χωρίς ημερομηνίες αποτελεσμάτων για: {', '.join(no_dates)} (καμία αγορά τους σήμερα).")
        warnings.append(f"⚠️ Χωρίς ημερομηνίες αποτελεσμάτων (καμία αγορά τους σήμερα): {', '.join(no_dates)}.")

    # Αναμονή ως 1 λεπτό μετά το άνοιγμα: αγορές και κλεισίματα στην τιμή ανοίγματος, όπως στο backtest
    wait = (trade_at - datetime.now(NY)).total_seconds()
    if not force and wait > 0:
        say(f"Αναμονή {wait / 60:.0f}′ ως τις {trade_at:%H:%M} Νέα Υόρκη (άνοιγμα της αγοράς).")
        time.sleep(wait)

    # 2) Σύνδεση με την Capital.com
    missing = [n.replace("CAPITAL_", "CAPITAL_LIVE_") if env == "live" else n
               for n in names if not os.environ.get(n, "").strip()]
    if len(missing) == len(names):
        say("Δεν έχουν οριστεί τα στοιχεία σύνδεσης (GitHub Secrets). Δες το bot/README.md. Τίποτα να κάνω.")
        return 0
    if missing:
        say(f"🛑 Λείπει ή είναι κενό το secret: {', '.join(missing)}. "
            "GitHub → Settings → Secrets and variables → Actions: έλεγξε ότι το όνομα είναι γραμμένο ακριβώς έτσι.")
        notify("Bot: λείπει στοιχείο σύνδεσης", log[-1])
        return 1
    try:
        cap = Capital(os.environ["CAPITAL_API_KEY"], os.environ["CAPITAL_IDENTIFIER"],
                      os.environ["CAPITAL_PASSWORD"], env)
        acc = cap.account()
        for _ in range(3):   # η Capital.com δείχνει καμιά φορά για λίγα λεπτά τον demo στο 0: ξαναρωτάμε
            if acc["equity"] > 0:
                break
            time.sleep(20)
            acc = cap.account()
        all_positions = cap.positions()
        positions = [ps for ps in all_positions if ps["epic"] in universe.values()]
        others = [ps for ps in all_positions if ps["epic"] not in universe.values()]
    except KeyError as e:
        # Δεν έχει στηθεί ακόμα: τερματισμός χωρίς σφάλμα, για να μη στέλνει το GitHub email αποτυχίας κάθε μέρα
        say(f"Δεν έχει οριστεί το secret {e}. Δες το bot/README.md. Τίποτα να κάνω.")
        return 0
    except CapitalError as e:
        say(f"Σφάλμα Capital.com: {e}")
        notify("Bot: σφάλμα σύνδεσης", "\n".join(log))
        return 1
    equity = acc["equity"]
    say(f"Λογαριασμός {acc['currency']} · ανοιχτές θέσεις bot: {len(positions)}")
    if env == "demo":   # εικονικά χρήματα: τα ποσά βοηθούν στον έλεγχο
        say(f"Demo: υπόλοιπο {acc['balance']:.2f} · χωρίς ανοιχτά {acc['deposit']:.2f} · "
            f"ανοιχτά κ/ζ {acc['pnl']:+.2f} · διαθέσιμα {acc['available']:.2f} · "
            f"λογαριασμοί {acc['n_accounts']} · τρέχων: {'ναι' if acc['is_current'] else 'όχι'}")
    else:               # πραγματικά χρήματα σε δημόσιο αποθετήριο: κανένα ποσό
        say(f"Λογαριασμοί στη σύνδεση: {acc['n_accounts']} · τρέχων: {'ναι' if acc['is_current'] else 'όχι'}")
    for ps in positions:   # κάθε θέση του bot πρέπει να έχει stop-loss και στόχο καταχωρημένα στην Capital.com
        px = float(ps["bid"] or ps["level"])
        chg = (px / ps["level"] - 1) * 100 if ps["level"] else 0.0
        say(f"Θέση {ps['epic']} @ {ps['level']:.2f} · τώρα {px:.2f} ({chg:+.1f}%) · "
            f"stop {ps['stop'] if ps['stop'] is not None else '—'} · στόχος {ps['tp'] if ps['tp'] is not None else '—'}")
        if ps["stop"] is None or ps["tp"] is None:
            missing = " και ".join(n for n, v in (("stop-loss", ps["stop"]), ("στόχο", ps["tp"])) if v is None)
            warnings.append(f"⚠️ Η θέση {ps['epic']} δεν έχει {missing} στην Capital.com! Βάλ' το από την εφαρμογή.")
    if others:   # θέσεις που δεν είναι του bot (π.χ. χειροκίνητες): δεσμεύουν διαθέσιμα και επηρεάζουν την αξία
        if env == "demo":
            say("Άλλες θέσεις στον λογαριασμό (όχι του bot): " + "; ".join(
                f"{ps['epic']} {ps['direction']} {ps['size']} @ {ps['level']} · κ/ζ {ps['upl']:+.2f}" for ps in others))
        else:
            say(f"Άλλες θέσεις στον λογαριασμό (όχι του bot): {len(others)}")
    if equity <= 0 and env == "live" and dry:   # πριν την κατάθεση: ελέγχουμε τουλάχιστον τη μόχλευση
        for asset, want in CONFIG.get("leverage", {}).items():
            try:
                cur = cap.leverage(asset)
                say(f"Μόχλευση {asset}: 1:{cur}" + (" ✓" if cur == want else f" ⚠️ πρέπει να γίνει 1:{want}"))
            except CapitalError as e:
                say(f"⚠️ Δεν μπόρεσα να διαβάσω τη μόχλευση {asset}: {e}")
        say("Ο πραγματικός λογαριασμός δεν έχει ακόμα χρήματα. Τίποτα να κάνω.")
        if not force:   # μία φορά τη μέρα, όπως κάθε δοκιμή
            (JOURNAL / DRY_MARK).write_text(json.dumps({"last_run": str(now.date())}), encoding="utf-8")
        return 0
    if equity <= 0:
        say("🛑 Η αξία του λογαριασμού φαίνεται μηδενική ή αρνητική. Δεν κάνω τίποτα, έλεγξε τον λογαριασμό.")
        notify("Bot: πρόβλημα στον λογαριασμό", "\n".join(log[-3:]))
        return 1
    if env_flag("CHECK_EPICS", False):
        # Διαγνωστικό (μόνο σε χειροκίνητη δοκιμή): υπάρχουν όλες οι μετοχές στην Capital.com;
        bad = []
        for sym, epic in universe.items():
            try:
                m = cap.market(epic)
                if m["min_size"] <= 0:
                    bad.append(f"{sym} (ελάχιστο μέγεθος {m['min_size']})")
            except CapitalError as e:
                bad.append(f"{sym} ({str(e)[:60]})")
        say(f"Έλεγχος συμβόλων: {len(universe) - len(bad)}/{len(universe)} εντάξει" +
            (f" · προβλήματα: {', '.join(bad)}" if bad else ""))
    epic_to_sym = {v: k for k, v in universe.items()}
    actions: list[str] = []
    record = not dry                                    # το ημερολόγιο γράφει μόνο πραγματικές κινήσεις (demo ή live)
    meta = load_state() if record else {"positions": {}, "peak_equity": None}
    state = meta["positions"]

    def persist() -> None:
        if record:
            save_state(meta)

    live_epics = {ps["epic"] for ps in positions}
    if record and state and not set(state) <= live_epics:
        # Η Capital.com δεν έδειξε κάποιες θέσεις του ημερολογίου: ξαναρωτάμε μία φορά (προσωρινό κενό απάντησης)
        time.sleep(5)
        try:
            all_positions = cap.positions()
            positions = [ps for ps in all_positions if ps["epic"] in universe.values()]
            live_epics = {ps["epic"] for ps in positions}
        except Exception as e:
            say(f"⚠️ Δεύτερη ανάγνωση θέσεων απέτυχε ({type(e).__name__}).")
    unverified: list[str] = []          # λείπουν από την Capital.com χωρίς απόδειξη κλεισίματος
    closed_today: set[str] = set()      # έκλεισαν στην Capital.com πριν το σημερινό άνοιγμα (π.χ. stop)
    for epic, entry in list(state.items()):
        if epic in live_epics:
            continue
        # Έκλεισε στην Capital.com από stop ή στόχο: η πραγματική τιμή από το ιστορικό της,
        # αλλιώς εκτίμηση από τα ημερήσια κεριά (που δεν βλέπουν τις προσυνεδριακές συναλλαγές)
        df = frames.get(entry["symbol"])
        px, why = entry["entry"], "Άγνωστο"
        close_day = today.date()
        try:
            found = broker_close(cap, epic, entry["date"], entry.get("deal_id"))
        except Exception as e:
            found = None
            say(f"⚠️ Δεν διάβασα το ιστορικό της Capital.com για τη {entry['symbol']} ({type(e).__name__}): εκτίμηση από τα κεριά.")
        if found:
            px, why, close_day = found
            if close_day == str(today.date()):
                closed_today.add(entry["symbol"])
        elif df is not None:
            after = df[df.index >= pd.Timestamp(entry["date"])]
            hit_stop = after[after["Low"] <= entry["stop"]]
            hit_tp = after[after["High"] >= entry["tp"]]
            first_stop = hit_stop.index[0] if len(hit_stop) else None
            first_tp = hit_tp.index[0] if len(hit_tp) else None
            if first_stop is not None and (first_tp is None or first_stop <= first_tp):
                px, why = entry["stop"], "Stop"
            elif first_tp is not None:
                px, why = entry["tp"], "Στόχος"
        if not found and why not in ("Stop", "Στόχος"):
            # Καμία απόδειξη ότι έκλεισε: δεν τη σβήνουμε από το ημερολόγιο και δεν ανοίγουμε νέες θέσεις σήμερα
            unverified.append(entry["symbol"])
            say(f"⚠️ Η {entry['symbol']} δεν φαίνεται στην Capital.com, αλλά δεν βρέθηκε κλείσιμο. Τη κρατώ στο ημερολόγιο.")
            continue
        log_trade(entry, close_day, px, why, env)
        say(f"Έκλεισε στην Capital.com: {entry['symbol']} · {why} · {(px / entry['entry'] - 1) * 100:+.1f}%")
        state.pop(epic)
        persist()

    if unverified:
        warnings.append(f"⚠️ Θέσεις του ημερολογίου που δεν φαίνονται στην Capital.com: {', '.join(unverified)}. "
                        "Καμία νέα αγορά σήμερα· έλεγξε την εφαρμογή.")

    def adopt(broker_positions: list[dict]) -> None:
        """Θέση του bot στην Capital.com χωρίς εγγραφή (π.χ. διακοπή δικτύου μετά την εντολή): μπαίνει στο ημερολόγιο."""
        for ps in broker_positions:
            if record and ps["epic"] not in state:
                created = pd.Timestamp(ps["created"]) if ps["created"] else today
                state[ps["epic"]] = {"symbol": epic_to_sym[ps["epic"]], "date": str(created.date()),
                                     "entry": float(ps["level"]), "stop": float(ps["stop"] or ps["level"]),
                                     "tp": float(ps["tp"] or ps["level"]), "kind": "", "deal_id": ps["deal_id"]}
                persist()
                say(f"⚠️ Η θέση {ps['epic']} υπάρχει στην Capital.com χωρίς εγγραφή στο ημερολόγιο: την πρόσθεσα.")

    adopt(positions)

    # 3) Έξοδοι λόγω χρόνου ή σπασμένης τάσης (stop και στόχος είναι ήδη στην Capital.com)
    uncertain = False
    for ps in positions:
        sym = epic_to_sym[ps["epic"]]
        sg = sigs.get(sym)
        if sg is None:
            continue
        created = today
        if ps["created"]:
            c = pd.Timestamp(ps["created"])
            created = (c.tz_convert(None) if c.tzinfo is not None else c).normalize()
        held = int((bars[sym] > created).sum())
        reason = "Χρόνος" if held >= p.max_bars else ("Τάση" if p.trend_exit and bool(sg["trend_broken"]) else "")
        if not reason and exit_before and sessions_to_results(earn.get(sym), sessions, today) == 1:
            reason = "Αποτελέσματα αύριο"
        if not reason:
            continue
        size_txt = f" ({ps['size']})" if env == "demo" else ""   # στον live κανένα μέγεθος (δημόσιο αποθετήριο)
        msg = f"ΚΛΕΙΣΙΜΟ {sym}{size_txt} · λόγος: {reason}, {held} μέρες σε θέση"
        if not dry:
            try:
                conf = cap.close(ps["deal_id"])
                exit_px = float(conf.get("level") or ps.get("bid") or sg["close"])
                if ps["epic"] in state:
                    log_trade(state.pop(ps["epic"]), today.date(), exit_px, reason, env)
                    persist()
            except Exception as e:   # και σφάλμα δικτύου: δεν ξέρουμε αν έκλεισε, άρα όχι νέες αγορές σήμερα
                msg += f" · ΑΠΕΤΥΧΕ: {type(e).__name__}: {e}"
                uncertain = True
        actions.append(msg)
        say(msg)
    closed = {a.split()[1] for a in actions if a.startswith("ΚΛΕΙΣΙΜΟ") and "ΑΠΕΤΥΧΕ" not in a}
    still_open = [ps for ps in positions if epic_to_sym[ps["epic"]] not in closed]
    if closed and not dry:   # τα χρήματα που ελευθερώθηκαν από τα σημερινά κλεισίματα μετράνε για τις αγορές
        try:
            fresh = cap.account()
            if fresh["equity"] > 0:
                acc, equity = fresh, fresh["equity"]
        except Exception as e:
            say(f"⚠️ Δεν ξαναδιάβασα τον λογαριασμό μετά τα κλεισίματα ({type(e).__name__}).")
    closed |= closed_today   # όπως το backtest: όχι ξανά αγορά τη μέρα που έκλεισε (και από stop πριν το άνοιγμα)

    # 4) Όρια προστασίας
    allow_new = not unverified and not uncertain
    if uncertain:
        say("⚠️ Ένα κλείσιμο δεν επιβεβαιώθηκε: καμία νέα αγορά σήμερα.")
        warnings.append("⚠️ Ένα κλείσιμο δεν επιβεβαιώθηκε (σφάλμα δικτύου). Έλεγξε τις θέσεις στην εφαρμογή.")
    # Το bot ΔΕΝ αλλάζει ρυθμίσεις λογαριασμού: μόνο ελέγχει ότι η μόχλευση είναι 1:1
    # (χωρίς 1:1 η Capital.com χρεώνει χρέωση νύχτας και η στρατηγική δεν βγαίνει).
    for asset, want in CONFIG.get("leverage", {}).items():
        try:
            cur = cap.leverage(asset)
        except CapitalError as e:
            cur = None
            say(f"⚠️ Δεν μπόρεσα να διαβάσω τη μόχλευση {asset}: {e}")
        if cur != want:
            allow_new = False
            msg = (f"🛑 Η μόχλευση για {asset} είναι 1:{cur}, όχι 1:{want}. Δεν ανοίγω νέες θέσεις. "
                   f"Βάλ' την 1:{want} στις ρυθμίσεις της Capital.com.")
            say(msg)
            actions.append(msg)
    # Φρένο: πτώση από το υψηλότερο σημείο του λογαριασμού αφότου ξεκίνησε το bot
    if env == "live":
        old_peak, start_eq = unseal(meta.get("peak_sealed")), unseal(meta.get("start_sealed"))
        if meta.get("peak_sealed") and old_peak is None:
            say("⚠️ Νέο κλειδί API: το υψηλότερο σημείο και η έναρξη του λογαριασμού μετρούν ξανά από σήμερα.")
            start_eq = None
        start_eq = start_eq or equity
    else:
        old_peak = meta.get("peak_equity")
    peak = max(float(old_peak or equity), equity)
    if env == "live":
        meta.pop("peak_equity", None)
        meta["peak_sealed"], meta["start_sealed"] = seal(peak), seal(start_eq)
        if record:   # μόνο ποσοστά (οι καταθέσεις και οι αναλήψεις μετράνε κι αυτές)
            say(f"Πραγματικός λογαριασμός: {(equity / start_eq - 1) * 100:+.1f}% από την έναρξη του bot · "
                f"{(equity / peak - 1) * 100:+.1f}% από το υψηλότερο σημείο")
    else:
        meta["peak_equity"] = peak
    if equity < peak * (1 - rk["max_drawdown_stop"]):
        allow_new = False
        msg = (f"🛑 Ο λογαριασμός έπεσε {(1 - equity / peak) * 100:.0f}% από το υψηλότερο σημείο του "
               f"(όριο {rk['max_drawdown_stop']:.0%}). Δεν ανοίγω νέες θέσεις μέχρι να το ελέγξεις.")
        say(msg)
        actions.append(msg)
    if rk.get("use_market_filter") and not market_is_ok:
        allow_new = False
        say(f"Φίλτρο αγοράς: ο Nasdaq-100 είναι κάτω από τον μέσο {p.ema_slow} ημερών. Χωρίς νέες αγορές σήμερα.")

    # 5) Νέες θέσεις (όλα τα ποσά στο νόμισμα του λογαριασμού)
    rates: dict[str, float] = {}

    def to_acc(cur: str) -> float:
        if cur not in rates:
            rates[cur] = fx_rate(cur, acc["currency"])
        return rates[cur]

    risk_used = notional_used = 0.0
    if allow_new:
        try:
            for ps in still_open:
                r = to_acc(ps.get("currency") or cap.market(ps["epic"])["currency"])
                # χωρίς stop: όλη η αξία της θέσης θεωρείται ρίσκο
                risk_used += (abs(ps["level"] - float(ps["stop"])) if ps["stop"] is not None else ps["level"]) * ps["size"] * r
                notional_used += float(ps["bid"] or ps["level"]) * ps["size"] * r
        except Exception as e:
            allow_new = False
            say(f"⚠️ Δεν υπολογίστηκε το ρίσκο των ανοιχτών θέσεων ({e}). Χωρίς νέες θέσεις σήμερα.")
    if allow_new:
        cands = []
        for sym, epic in universe.items():
            sg = sigs.get(sym)
            if sg is None or not bool(sg["long_sig"]) or epic in {ps["epic"] for ps in still_open}:
                continue
            if sym in closed:   # όπως το backtest: όχι ξανά αγορά τη μέρα που βγήκε (διπλό spread χωρίς λόγο)
                say(f"Παράλειψη {sym}: έκλεισε σήμερα, όχι ξανά αγορά την ίδια μέρα.")
                continue
            if block:
                n = sessions_to_results(earn.get(sym), sessions, today)
                if n is None:
                    say(f"Παράλειψη {sym}: άγνωστη ημερομηνία αποτελεσμάτων, για σιγουριά όχι αγορά.")
                    continue
                if n <= block:
                    when = sessions[int(sessions.searchsorted(today)) + int(n)]
                    say(f"Παράλειψη {sym}: βγάζει αποτελέσματα σε {int(n)} συνεδρ. ({when:%d/%m}), όχι αγορά πριν από αυτά.")
                    continue
            mom = sg["mom63"] if not pd.isna(sg["mom63"]) else -9
            cands.append((mom, sym, epic, sg))
        cands.sort(key=lambda x: x[0], reverse=True)
        n_open = len(still_open)
        available = acc["available"] * 0.98            # λίγο περιθώριο για spread και συναλλάγματα
        if dry:   # στη δοκιμή τα κλεισίματα δεν γίνονται: προσθέτουμε την αξία που θα ελευθέρωναν
            for ps in positions:
                if epic_to_sym[ps["epic"]] in closed:
                    cur = ps.get("currency") or cap.market(ps["epic"])["currency"]
                    available += float(ps["bid"] or ps["level"]) * ps["size"] * to_acc(cur) * 0.98
        for _, sym, epic, sg in cands:
            if n_open >= rk["max_positions"]:
                say(f"Παράλειψη {sym}: έχουμε ήδη {n_open} θέσεις (όριο {rk['max_positions']}).")
                continue
            try:
                m = cap.market(epic)
                rate = to_acc(m["currency"])
            except Exception as e:
                say(f"Παράλειψη {sym}: {e}")
                continue
            if m["status"] != "TRADEABLE":
                say(f"Παράλειψη {sym}: η αγορά δεν είναι ανοιχτή ({m['status']}).")
                continue
            price = float(m["offer"] or sg["close"])
            stop, tp = float(sg["stop"]), float(sg["tp"])
            if not (stop < price < tp):
                say(f"Παράλειψη {sym}: η τιμή {price:.2f} είναι ήδη εκτός stop/στόχου ({stop:.2f}–{tp:.2f}).")
                continue
            dist = price - stop
            min_dist = price * m["min_stop"] / 100 if m["min_stop_unit"] == "PERCENTAGE" else m["min_stop"]
            if dist < min_dist or tp - price < min_dist:
                say(f"Παράλειψη {sym}: stop/στόχος πιο κοντά από το ελάχιστο της Capital.com.")
                continue
            # Μέγεθος σε μονάδες του προϊόντος, με όλα τα ποσά στο νόμισμα του λογαριασμού
            qty_risk = equity * rk["risk_per_trade"] / (dist * rate)
            qty_room = min(max(0.0, rk["max_total_risk"] * equity - risk_used) / (dist * rate),
                           max(0.0, rk["max_notional"] * equity - notional_used) / (price * rate),
                           max(0.0, available) / (price * rate))
            size = round_size(min(qty_risk, qty_room), m["size_step"], m["min_size"])
            if size * price * rate < MIN_POSITION * equity:
                why = ("δεν περισσεύει ρίσκο ή κεφάλαιο" if qty_room < qty_risk
                       else f"πολύ μικρή θέση (κάτω από {MIN_POSITION:.0%} του λογαριασμού)")
                say(f"Παράλειψη {sym}: {why}.")
                continue
            amount = f"{size} μονάδες" if env == "demo" else f"{size * price * rate / equity:.0%} του λογαριασμού"
            msg = (f"ΑΓΟΡΑ {sym} ({epic}) · {amount} στα ~{price:.2f} · stop {stop:.2f} "
                   f"({(stop / price - 1) * 100:+.1f}%) · στόχος {tp:.2f} ({(tp / price - 1) * 100:+.1f}%) · {sg['kind']}")
            ok = True
            if not dry:
                try:
                    conf = cap.open(epic, size, stop, tp)
                    deal = next((d.get("dealId") for d in conf.get("affectedDeals") or [] if d.get("dealId")),
                                conf.get("dealId"))
                    state[epic] = {"symbol": sym, "date": str(today.date()), "entry": float(conf.get("level") or price),
                                   "stop": stop, "tp": tp, "kind": sg["kind"], "deal_id": deal}
                    persist()
                except CapitalError as e:      # η Capital.com απέρριψε την εντολή: δεν δεσμεύει χρήματα
                    msg += f" · ΑΠΕΤΥΧΕ: {e}"
                    ok = False
                except Exception as e:
                    # Σφάλμα δικτύου: ίσως άνοιξε. Αν άνοιξε, μπαίνει αμέσως στο ημερολόγιο· μετά η εκτέλεση
                    # σταματά ως αποτυχημένη, ώστε η επόμενη εκκίνηση να ξαναδοκιμάσει (βλέπει τις θέσεις, χωρίς διπλές)
                    say(f"{msg} · ΑΠΕΤΥΧΕ: {type(e).__name__}: {e}")
                    try:
                        time.sleep(3)
                        adopt([ps for ps in cap.positions() if ps["epic"] in universe.values()])
                    except Exception as e2:
                        say(f"⚠️ Δεν ξαναδιάβασα τις θέσεις ({type(e2).__name__}).")
                    raise RuntimeError(f"άγνωστη έκβαση εντολής για {sym}: {type(e).__name__}") from e
            actions.append(msg)
            say(msg)
            if ok:
                n_open += 1
                risk_used += dist * size * rate
                notional_used += price * size * rate
                available -= price * size * rate

    if not actions:
        say("Καμία κίνηση σήμερα.")
    if dry and not force:   # και οι προγραμματισμένες δοκιμές τρέχουν μία φορά τη μέρα
        JOURNAL.mkdir(parents=True, exist_ok=True)
        (JOURNAL / DRY_MARK).write_text(json.dumps({"last_run": str(today.date())}), encoding="utf-8")
    if record:
        meta["last_run"] = str(today.date())
        save_state(meta)
        open_now = ";".join(f"{epic_to_sym[ps['epic']]}:{((float(ps['bid'] or ps['level']) / ps['level']) - 1) * 100:+.1f}%"
                            for ps in still_open if ps["level"])
        append_csv("daily.csv", "date,env,equity_demo,open_positions,market_filter_ok,actions",
                   [today.date(), env, f"{equity:.2f}" if env == "demo" else "", open_now or "-",
                    market_is_ok, " | ".join(actions) or "-"])
        try:   # μόνο καταγραφή: ένα σφάλμα εδώ δεν επηρεάζει τις πραγματικές συναλλαγές
            if block or exit_before:
                earn |= earnings_calendar([s for s in universe if s not in earn], sessions)
            say(update_virtual(frames, p, rk, state, sigs, market_is_ok, today, earn))
        except Exception as e:
            say(f"⚠️ Εικονικό ημερολόγιο: {type(e).__name__}: {e}")
    title = (f"Bot {env}{' (δοκιμή)' if dry else ''}: " + (f"{len(actions)} κινήσεις" if actions else "καμία κίνηση")
             + (" ⚠️" if warnings else ""))
    notify(title, "\n".join(warnings + actions) or "Δεν υπήρξαν σήματα σήμερα.")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        Path(summary).write_text("\n\n".join(log), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
