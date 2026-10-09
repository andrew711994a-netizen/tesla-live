"""Καθημερινή εκτέλεση: ελέγχει σήματα και ανοίγει/κλείνει θέσεις στην Capital.com.

Τρέχει κάθε εργάσιμη μέρα λίγο μετά το άνοιγμα της Νέας Υόρκης, με τα σήματα
του χθεσινού κλεισίματος (όπως το backtest και το TradingView).

Μεταβλητές περιβάλλοντος:
  CAPITAL_API_KEY, CAPITAL_IDENTIFIER, CAPITAL_PASSWORD   στοιχεία API (GitHub Secrets)
  CAPITAL_ENV      demo (προεπιλογή) ή live
  ALLOW_LIVE       πρέπει να είναι I_UNDERSTAND_THE_RISK για live λογαριασμό
  DRY_RUN          true (προεπιλογή): μόνο γράφει τι θα έκανε, χωρίς εντολές
  BOT_ENABLED      false: δεν κάνει τίποτα (κουμπί διακοπής)
  NTFY_TOPIC       θέμα στο ntfy.sh για ειδοποιήσεις στο κινητό
  FORCE            true: αγνοεί τον έλεγχο ώρας (για χειροκίνητη δοκιμή)
"""

from __future__ import annotations

import json
import math
import os
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
from .data import daily
from .notify import notify
from .strategy import Params, market_ok, signals

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
    JOURNAL.mkdir(exist_ok=True)
    (JOURNAL / "positions.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def append_csv(name: str, header: str, row: list) -> None:
    JOURNAL.mkdir(exist_ok=True)
    f = JOURNAL / name
    new = not f.exists()
    with f.open("a", encoding="utf-8") as fh:
        if new:
            fh.write(header + "\n")
        fh.write(",".join(str(x).replace(",", ";") for x in row) + "\n")


def log_trade(entry: dict, exit_date, exit_px: float, reason: str, env: str) -> None:
    res = exit_px / entry["entry"] - 1
    r = (exit_px - entry["entry"]) / max(entry["entry"] - entry["stop"], 1e-9)
    append_csv("trades.csv", "open_date,close_date,symbol,entry,exit,result_pct,r_multiple,reason,env",
               [entry["date"], exit_date, entry["symbol"], f"{entry['entry']:.2f}", f"{exit_px:.2f}",
                f"{res * 100:+.2f}", f"{r:+.2f}", reason, env])


VIRTUAL = "virtual.json"


def update_virtual(frames: dict, p: Params, rk: dict, state: dict, sigs: dict, market_is_ok: bool,
                   today: pd.Timestamp) -> str:
    """Εικονικό ημερολόγιο: κάθε σήμα από την πρώτη μέρα του bot σαν να αγοράστηκε, χωρίς όριο θέσεων ή κεφαλαίου.

    Καμία εντολή, μόνο καταγραφή, για να κριθεί η στρατηγική με πολύ περισσότερες συναλλαγές. Ξαναϋπολογίζεται
    κάθε μέρα από τα ημερήσια κεριά με τους κανόνες του backtest (είσοδος στο άνοιγμα, stop/στόχος, 20 μέρες,
    σπασμένη τάση, κόστη), οπότε δεν χάνει μέρες ούτε μετρά κάτι δύο φορές.
    """
    from .backtest import Costs, Risk, every_signal, run as replay, trade_stats

    market = frames.get(CONFIG["market_symbol"])
    days_log = JOURNAL / "daily.csv"
    if market is None or not days_log.exists():
        return "Εικονικό ημερολόγιο: δεν ενημερώθηκε σήμερα (λείπουν δεδομένα αγοράς)."
    first = pd.Timestamp(pd.read_csv(days_log)["date"].min())   # πρώτη μέρα του bot
    before = market.index[market.index < first]
    start = before[-1] if len(before) else first               # τα σήματα εκείνου του κλεισίματος μπαίνουν την πρώτη μέρα
    data = {s: frames[s] for s in CONFIG["universe"] if s in frames}
    risk = every_signal(Risk(rk["risk_per_trade"], use_market_filter=rk["use_market_filter"]))
    trades = replay(data, p, risk, Costs(**CONFIG["costs"]), str(start.date()), market=market)["trades"]

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
    JOURNAL.mkdir(exist_ok=True)
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


def env_flag(name: str, default: bool) -> bool:
    v = os.environ.get(name, "").strip().lower()
    return default if v == "" else v in ("1", "true", "yes", "on")


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
    JOURNAL.mkdir(exist_ok=True)
    stamp = datetime.now(NY).strftime("%Y-%m-%d %H:%M")
    dry = env_flag("DRY_RUN", True)
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
    if env == "live" and os.environ.get("ALLOW_LIVE") != "I_UNDERSTAND_THE_RISK":
        say("Άρνηση: ο live λογαριασμός χρειάζεται ALLOW_LIVE=I_UNDERSTAND_THE_RISK.")
        return 1
    dry = env_flag("DRY_RUN", True)
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

    # Αναμονή ως 1 λεπτό μετά το άνοιγμα: αγορές και κλεισίματα στην τιμή ανοίγματος, όπως στο backtest
    wait = (trade_at - datetime.now(NY)).total_seconds()
    if not force and wait > 0:
        say(f"Αναμονή {wait / 60:.0f}′ ως τις {trade_at:%H:%M} Νέα Υόρκη (άνοιγμα της αγοράς).")
        time.sleep(wait)

    # 2) Σύνδεση με την Capital.com
    names = ["CAPITAL_API_KEY", "CAPITAL_IDENTIFIER", "CAPITAL_PASSWORD"]
    missing = [n for n in names if not os.environ.get(n, "").strip()]
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
    for ps in positions:   # κάθε θέση του bot πρέπει να έχει stop-loss και στόχο καταχωρημένα στην Capital.com
        px = float(ps["bid"] or ps["level"])
        say(f"Θέση {ps['epic']} @ {ps['level']:.2f} · τώρα {px:.2f} ({(px / ps['level'] - 1) * 100:+.1f}%) · "
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
    for epic, entry in list(state.items()):
        if epic in live_epics:
            continue
        # Έκλεισε στην Capital.com από stop ή στόχο: εκτίμηση από τα ημερήσια κεριά
        df = frames.get(entry["symbol"])
        px, why = entry["entry"], "Άγνωστο"
        if df is not None:
            after = df[df.index >= pd.Timestamp(entry["date"])]
            hit_stop = after[after["Low"] <= entry["stop"]]
            hit_tp = after[after["High"] >= entry["tp"]]
            first_stop = hit_stop.index[0] if len(hit_stop) else None
            first_tp = hit_tp.index[0] if len(hit_tp) else None
            if first_stop is not None and (first_tp is None or first_stop <= first_tp):
                px, why = entry["stop"], "Stop"
            elif first_tp is not None:
                px, why = entry["tp"], "Στόχος"
            elif len(after):
                px, why = float(after["Close"].iloc[-1]), "Άγνωστο (εκτίμηση)"
        log_trade(entry, today.date(), px, why, env)
        say(f"Έκλεισε στην Capital.com: {entry['symbol']} · {why} · {(px / entry['entry'] - 1) * 100:+.1f}%")
        state.pop(epic)
        persist()

    # 3) Έξοδοι λόγω χρόνου ή σπασμένης τάσης (stop και στόχος είναι ήδη στην Capital.com)
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
        if not reason:
            continue
        msg = f"ΚΛΕΙΣΙΜΟ {sym} ({ps['size']}) · λόγος: {reason}, {held} μέρες σε θέση"
        if not dry:
            try:
                conf = cap.close(ps["deal_id"])
                exit_px = float(conf.get("level") or ps.get("bid") or sg["close"])
                if ps["epic"] in state:
                    log_trade(state.pop(ps["epic"]), today.date(), exit_px, reason, env)
                    persist()
            except CapitalError as e:
                msg += f" · ΑΠΕΤΥΧΕ: {e}"
        actions.append(msg)
        say(msg)
    closed = {a.split()[1] for a in actions if a.startswith("ΚΛΕΙΣΙΜΟ") and "ΑΠΕΤΥΧΕ" not in a}
    still_open = [ps for ps in positions if epic_to_sym[ps["epic"]] not in closed]

    # 4) Όρια προστασίας
    allow_new = True
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
    peak = max(float(meta.get("peak_equity") or equity), equity)
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
                risk_used += abs(ps["level"] - float(ps["stop"] or ps["level"])) * ps["size"] * r
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
            mom = sg["mom63"] if not pd.isna(sg["mom63"]) else -9
            cands.append((mom, sym, epic, sg))
        cands.sort(key=lambda x: x[0], reverse=True)
        n_open = len(still_open)
        available = acc["available"] * 0.98            # λίγο περιθώριο για spread και συναλλάγματα
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
            msg = (f"ΑΓΟΡΑ {sym} ({epic}) · {size} μονάδες στα ~{price:.2f} · stop {stop:.2f} "
                   f"({(stop / price - 1) * 100:+.1f}%) · στόχος {tp:.2f} ({(tp / price - 1) * 100:+.1f}%) · {sg['kind']}")
            if not dry:
                try:
                    conf = cap.open(epic, size, stop, tp)
                    state[epic] = {"symbol": sym, "date": str(today.date()), "entry": float(conf.get("level") or price),
                                   "stop": stop, "tp": tp, "kind": sg["kind"]}
                    persist()
                except CapitalError as e:
                    msg += f" · ΑΠΕΤΥΧΕ: {e}"
            actions.append(msg)
            say(msg)
            n_open += 1
            risk_used += dist * size * rate
            notional_used += price * size * rate
            available -= price * size * rate

    if not actions:
        say("Καμία κίνηση σήμερα.")
    if dry and not force:   # και οι προγραμματισμένες δοκιμές τρέχουν μία φορά τη μέρα
        JOURNAL.mkdir(exist_ok=True)
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
            say(update_virtual(frames, p, rk, state, sigs, market_is_ok, today))
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
