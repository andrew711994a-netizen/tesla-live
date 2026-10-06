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
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from .capital import Capital, CapitalError
from .data import daily
from .notify import notify
from .strategy import Params, market_ok, signals

NY = ZoneInfo("America/New_York")
CONFIG = json.loads((Path(__file__).parent / "config.json").read_text(encoding="utf-8"))


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


def main() -> int:
    log: list[str] = []

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
    now = datetime.now(NY)
    if not env_flag("FORCE", False):
        if now.weekday() >= 5 or not ((now.hour, now.minute) >= (9, 35) and now.hour < 11):
            say(f"Εκτός παραθύρου εκτέλεσης ({now:%a %H:%M} Νέα Υόρκη). Τίποτα να κάνω.")
            return 0

    p = Params.from_dict(CONFIG["strategy"])
    rk = CONFIG["risk"]
    universe: dict[str, str] = CONFIG["universe"]
    mode = f"{env.upper()}{' · ΔΟΚΙΜΗ (χωρίς εντολές)' if dry else ''}"
    say(f"Έναρξη {now:%d/%m %H:%M} Νέα Υόρκη · {mode}")

    # 1) Δεδομένα και σήματα από το τελευταίο ολοκληρωμένο κλείσιμο
    today = pd.Timestamp(now.date())
    start = (today - pd.Timedelta(days=500)).strftime("%Y-%m-%d")
    sigs: dict[str, pd.Series] = {}
    bars: dict[str, pd.DatetimeIndex] = {}
    market_is_ok = False
    for sym in set(universe) | {CONFIG["market_symbol"]}:
        try:
            df = daily(sym, start=start)
            df = df[df.index < today]                      # χωρίς το μισό κερί της σημερινής μέρας
            sg = signals(df, p)
            sigs[sym] = sg.iloc[-1]
            bars[sym] = df.index                           # για μέτρηση ημερών σε θέση
            if sym == CONFIG["market_symbol"]:
                market_is_ok = bool(market_ok(df, p).iloc[-1])
        except Exception as e:
            say(f"⚠️ {sym}: δεν ήρθαν δεδομένα ({e})")
    if CONFIG["market_symbol"] not in sigs:
        say("Χωρίς δεδομένα αγοράς, δεν ανοίγω νέες θέσεις σήμερα.")

    # 2) Σύνδεση με την Capital.com
    try:
        cap = Capital(os.environ["CAPITAL_API_KEY"], os.environ["CAPITAL_IDENTIFIER"],
                      os.environ["CAPITAL_PASSWORD"], env)
        acc = cap.account()
        positions = [ps for ps in cap.positions() if ps["epic"] in universe.values()]
    except KeyError as e:
        say(f"Λείπει το secret {e}. Δες το bot/README.md.")
        notify("Bot: λείπουν στοιχεία σύνδεσης", "\n".join(log))
        return 1
    except CapitalError as e:
        say(f"Σφάλμα Capital.com: {e}")
        notify("Bot: σφάλμα σύνδεσης", "\n".join(log))
        return 1
    equity = acc["equity"]
    say(f"Λογαριασμός {acc['currency']} · ανοιχτές θέσεις bot: {len(positions)}")
    epic_to_sym = {v: k for k, v in universe.items()}
    actions: list[str] = []

    # 3) Έξοδοι λόγω χρόνου ή σπασμένης τάσης (stop και στόχος είναι ήδη στην Capital.com)
    for ps in positions:
        sym = epic_to_sym[ps["epic"]]
        sg = sigs.get(sym)
        if sg is None:
            continue
        created = pd.Timestamp(ps["created"]).tz_localize(None).normalize() if ps["created"] else today
        held = int((bars[sym] > created).sum())
        reason = "Χρόνος" if held >= p.max_bars else ("Τάση" if p.trend_exit and bool(sg["trend_broken"]) else "")
        if not reason:
            continue
        msg = f"ΚΛΕΙΣΙΜΟ {sym} ({ps['size']}) · λόγος: {reason}, {held} μέρες σε θέση"
        if not dry:
            try:
                cap.close(ps["deal_id"])
            except CapitalError as e:
                msg += f" · ΑΠΕΤΥΧΕ: {e}"
        actions.append(msg)
        say(msg)
    closed = {a.split()[1] for a in actions if a.startswith("ΚΛΕΙΣΙΜΟ") and "ΑΠΕΤΥΧΕ" not in a}
    still_open = [ps for ps in positions if epic_to_sym[ps["epic"]] not in closed]

    # 4) Όρια προστασίας
    allow_new = True
    if acc["deposit"] > 0 and equity < acc["deposit"] * (1 - rk["max_drawdown_stop"]):
        allow_new = False
        say(f"🛑 Ο λογαριασμός έπεσε πάνω από {rk['max_drawdown_stop']:.0%} από τις καταθέσεις. "
            "Δεν ανοίγω νέες θέσεις μέχρι να το ελέγξεις.")
    if rk.get("use_market_filter") and not market_is_ok:
        allow_new = False
        say("Φίλτρο αγοράς: ο Nasdaq-100 είναι κάτω από τον μέσο 200 ημερών. Χωρίς νέες αγορές σήμερα.")

    # 5) Νέες θέσεις
    if allow_new:
        cands = []
        for sym, epic in universe.items():
            sg = sigs.get(sym)
            if sg is None or not bool(sg["long_sig"]) or epic in {ps["epic"] for ps in still_open}:
                continue
            mom = sg["mom63"] if not pd.isna(sg["mom63"]) else -9
            cands.append((mom, sym, epic, sg))
        cands.sort(key=lambda x: x[0], reverse=True)
        risk_used = sum(abs(ps["level"] - float(ps["stop"] or ps["level"])) * ps["size"] for ps in still_open)
        notional_used = sum(ps["level"] * ps["size"] for ps in still_open)
        n_open = len(still_open)
        for _, sym, epic, sg in cands:
            if n_open >= rk["max_positions"]:
                say(f"Παράλειψη {sym}: έχουμε ήδη {n_open} θέσεις (όριο {rk['max_positions']}).")
                continue
            try:
                m = cap.market(epic)
                rate = fx_rate(m["currency"], acc["currency"])
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
            qty = equity * rk["risk_per_trade"] / (dist * rate)
            qty = min(qty, max(0.0, rk["max_total_risk"] * equity - risk_used) / (dist * rate))
            qty = min(qty, max(0.0, rk["max_notional"] * equity - notional_used) / (price * rate))
            size = round_size(qty, m["size_step"], m["min_size"])
            if size <= 0:
                say(f"Παράλειψη {sym}: πολύ μικρό μέγεθος για το ελάχιστο της Capital.com ({m['min_size']}).")
                continue
            msg = (f"ΑΓΟΡΑ {sym} ({epic}) · {size} μονάδες στα ~{price:.2f} · stop {stop:.2f} "
                   f"({(stop / price - 1) * 100:+.1f}%) · στόχος {tp:.2f} ({(tp / price - 1) * 100:+.1f}%) · {sg['kind']}")
            if not dry:
                try:
                    cap.open(epic, size, stop, tp)
                except CapitalError as e:
                    msg += f" · ΑΠΕΤΥΧΕ: {e}"
            actions.append(msg)
            say(msg)
            n_open += 1
            risk_used += dist * size
            notional_used += price * size

    if not actions:
        say("Καμία κίνηση σήμερα.")
    title = f"Bot {env}{' (δοκιμή)' if dry else ''}: " + (f"{len(actions)} κινήσεις" if actions else "καμία κίνηση")
    notify(title, "\n".join(actions) if actions else "Δεν υπήρξαν σήματα σήμερα.")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        Path(summary).write_text("\n\n".join(log), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
