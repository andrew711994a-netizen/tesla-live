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
JOURNAL = Path(__file__).resolve().parent.parent / "journal"


# ── Ημερολόγιο (χωρίς ποσά: μόνο σύμβολα, τιμές και ποσοστά) ──
def load_state() -> dict:
    f = JOURNAL / "positions.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


def save_state(state: dict) -> None:
    JOURNAL.mkdir(exist_ok=True)
    (JOURNAL / "positions.json").write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")


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


def write_run_log(status: str) -> None:
    """Σύντομο σημείωμα της τελευταίας εκτέλεσης (χωρίς ποσά ή κλειδιά), για τον καθημερινό έλεγχο."""
    JOURNAL.mkdir(exist_ok=True)
    stamp = datetime.now(NY).strftime("%Y-%m-%d %H:%M")
    (JOURNAL / "last_run.md").write_text(f"# Τελευταία εκτέλεση {stamp} (Νέα Υόρκη) · {status}\n\n" +
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
    frames: dict[str, pd.DataFrame] = {}
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
    if CONFIG["market_symbol"] not in sigs:
        say("Χωρίς δεδομένα αγοράς, δεν ανοίγω νέες θέσεις σήμερα.")

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
        positions = [ps for ps in cap.positions() if ps["epic"] in universe.values()]
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
    epic_to_sym = {v: k for k, v in universe.items()}
    actions: list[str] = []
    record = not dry                                    # το ημερολόγιο γράφει μόνο πραγματικές κινήσεις (demo ή live)
    state = load_state() if record else {}
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
                conf = cap.close(ps["deal_id"])
                exit_px = float(conf.get("level") or ps.get("bid") or sg["close"])
                if ps["epic"] in state:
                    log_trade(state.pop(ps["epic"]), today.date(), exit_px, reason, env)
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
                    conf = cap.open(epic, size, stop, tp)
                    state[epic] = {"symbol": sym, "date": str(today.date()), "entry": float(conf.get("level") or price),
                                   "stop": stop, "tp": tp, "kind": sg["kind"]}
                except CapitalError as e:
                    msg += f" · ΑΠΕΤΥΧΕ: {e}"
            actions.append(msg)
            say(msg)
            n_open += 1
            risk_used += dist * size
            notional_used += price * size

    if not actions:
        say("Καμία κίνηση σήμερα.")
    if record:
        save_state(state)
        open_now = ";".join(f"{epic_to_sym[ps['epic']]}:{((float(ps['bid'] or ps['level']) / ps['level']) - 1) * 100:+.1f}%"
                            for ps in still_open if ps["level"])
        append_csv("daily.csv", "date,env,equity_demo,open_positions,market_filter_ok,actions",
                   [today.date(), env, f"{equity:.2f}" if env == "demo" else "", open_now or "-",
                    market_is_ok, " | ".join(actions) or "-"])
    title = f"Bot {env}{' (δοκιμή)' if dry else ''}: " + (f"{len(actions)} κινήσεις" if actions else "καμία κίνηση")
    notify(title, "\n".join(actions) if actions else "Δεν υπήρξαν σήματα σήμερα.")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        Path(summary).write_text("\n\n".join(log), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
