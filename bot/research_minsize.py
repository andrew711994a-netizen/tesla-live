"""Έλεγχος: με πόσα χρήματα μπορεί το bot να αγοράζει κανονικά όλες τις μετοχές του;

Η Capital.com έχει ελάχιστο μέγεθος εντολής και βήμα μεγέθους για κάθε μετοχή. Με μικρό λογαριασμό, το μέγεθος
που βγάζει το ρίσκο 0,75% ανά συναλλαγή μπορεί να είναι κάτω από το ελάχιστο (η μετοχή δεν αγοράζεται), ή να
στρογγυλεύεται τόσο προς τα κάτω που η θέση βγαίνει πολύ μικρότερη από το κανονικό. Μόνο ανάγνωση: καμία εντολή.

Εκτέλεση:  python -m bot.research_minsize   (με CAPITAL_API_KEY/IDENTIFIER/PASSWORD)
Γράφει το reports/research_minsize.md
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from .backtest import CONFIG
from .data import daily
from .live import MIN_POSITION, round_size
from .strategy import Params, signals

ROOT = Path(__file__).resolve().parent.parent
ACCOUNTS = [500, 1000, 2000, 3000, 5000, 10000]


def main() -> None:
    from .capital import Capital

    cap = Capital(os.environ["CAPITAL_API_KEY"], os.environ["CAPITAL_IDENTIFIER"], os.environ["CAPITAL_PASSWORD"],
                  "demo")   # οι κανόνες μεγέθους είναι ίδιοι στο demo και στο live (να επιβεβαιωθεί στο live)
    p = Params.from_dict(CONFIG["strategy"])
    rk = CONFIG["risk"]
    start = (pd.Timestamp.today() - pd.Timedelta(days=400)).strftime("%Y-%m-%d")
    fx: dict[str, float] = {}
    rows = []
    for sym, epic in CONFIG["universe"].items():
        try:
            m = cap.market(epic)
            df = daily(sym, start=start)
        except Exception as e:
            print(f"{sym}: {e}")
            continue
        cur = m["currency"]
        if cur not in fx:
            fx[cur] = 1.0 if cur == "EUR" else float(daily(f"{cur}EUR=X", start=start)["Close"].iloc[-1])
        sg = signals(df, p).iloc[-1]
        price = float(m["offer"] or df["Close"].iloc[-1])
        dist = p.sl_atr * float(sg["atr"])           # τυπική απόσταση stop σήμερα (2 × ATR)
        rows.append({"symbol": sym, "price": price, "cur": cur, "rate": fx[cur], "min_size": m["min_size"],
                     "step": m["size_step"], "dist": dist, "dist_pct": dist / price,
                     "min_eur": m["min_size"] * price * fx[cur]})

    def check(r: dict, equity: float) -> tuple[str, float, float]:
        want = equity * rk["risk_per_trade"] / (r["dist"] * r["rate"])          # μονάδες για ρίσκο 0,75%
        want = min(want, rk["max_notional"] * equity / (r["price"] * r["rate"]))
        size = round_size(want, r["step"], r["min_size"])
        value = size * r["price"] * r["rate"]
        if size <= 0:
            return "όχι", 0.0, want
        if value < MIN_POSITION * equity:
            return "όχι", value, want
        return ("μικρότερη" if size < 0.75 * want else "κανονικά"), value, want

    out = ["# Ελάχιστο ποσό λογαριασμού για το bot", "",
           f"Κανόνες μεγέθους της Capital.com για τις {len(rows)} μετοχές του bot (ανάγνωση {pd.Timestamp.today():%d/%m/%Y}), "
           f"ρίσκο {rk['risk_per_trade']:.2%} ανά συναλλαγή, stop στα 2 × ATR όπως σήμερα. «Κανονικά» = η θέση βγαίνει "
           "τουλάχιστον στο 75% του κανονικού μεγέθους· «μικρότερη» = η στρογγυλοποίηση την κόβει πολύ· «όχι» = κάτω από "
           "το ελάχιστο της Capital.com ή κάτω από το 2% του λογαριασμού, άρα το bot την παραλείπει.", "",
           "| Λογαριασμός | Κανονικά | Μικρότερη | Όχι | Ποιες δεν αγοράζονται |", "|---|---|---|---|---|"]
    summary = {}
    for eq in ACCOUNTS:
        res = {r["symbol"]: check(r, eq) for r in rows}
        no = [s for s, v in res.items() if v[0] == "όχι"]
        small = [s for s, v in res.items() if v[0] == "μικρότερη"]
        summary[eq] = {"ok": len(rows) - len(no) - len(small), "smaller": small, "no": no}
        out.append(f"| {eq:,} € | {summary[eq]['ok']} | {len(small)}" + (f" ({', '.join(small)})" if small else "") +
                   f" | {len(no)} | {', '.join(no) if no else '—'} |")
    out += ["", "## Ανά μετοχή", "",
            "| Μετοχή | Τιμή | Ελάχιστο / βήμα (μονάδες) | Ελάχιστη θέση σε € | Τυπική θέση με 3.000 € | Με 3.000 € |",
            "|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: -r["min_eur"]):
        st, val, want = check(r, 3000)
        typical = want * r["price"] * r["rate"]
        out.append(f"| {r['symbol']} | {r['price']:.2f} {r['cur']} | {r['min_size']:g} / {r['step']:g} | "
                   f"{r['min_eur']:.0f} € | {typical:.0f} € | {st} |")
    out.append("")
    path = ROOT / "reports"
    path.mkdir(exist_ok=True)
    (path / "research_minsize.md").write_text("\n".join(out), encoding="utf-8")
    (path / "research_minsize.json").write_text(json.dumps({"rows": rows, "summary": summary}, ensure_ascii=False,
                                                           indent=1, default=float), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
