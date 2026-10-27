"""Θερμόμετρο κινδύνου της αγοράς: μόνο για ενημέρωση, το bot δεν το χρησιμοποιεί.

Κάθε βράδυ, μετά το κλείσιμο, διαβάζει από το Yahoo μερικούς δείκτες που συνήθως χειροτερεύουν όταν η αγορά μπαίνει
σε κρίση, και δίνει χρώμα σε καθέναν (🟢 ήρεμα, 🟡 προσοχή, 🔴 ένταση):
  - VIX (ο «δείκτης φόβου»: πόσο μεγάλες κινήσεις περιμένει η αγορά στο S&P 500),
  - VIX/VIX3M (πάνω από 1 = ο φόβος για τον επόμενο μήνα ξεπερνά τον φόβο για το τρίμηνο, σημάδι έντασης),
  - HYG (ομόλογα υψηλού ρίσκου: όταν πέφτουν, οι επενδυτές φοβούνται χρεοκοπίες),
  - 10ετές επιτόκιο ΗΠΑ (απότομη άνοδος πιέζει τις μετοχές),
  - δολάριο (απότομη άνοδος = φυγή προς ασφάλεια),
  - Nasdaq-100 σε σχέση με το φίλτρο του bot (EMA 150) και πόσο απέχει από το υψηλό του έτους.
Στην έρευνα (reports/research_ideas.md) κανένας από αυτούς δεν βελτίωσε το bot ως κανόνας· εδώ απλώς ενημερώνουν.

Εκτέλεση:  python -m bot.macro
Γράφει το reports/macro.md και .json (με το χθεσινό επίπεδο, για να φαίνεται η αλλαγή)
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from .backtest import CONFIG
from .data import daily
from .strategy import Params, _ema

ROOT = Path(__file__).resolve().parent.parent
GREEN, YELLOW, RED = "🟢", "🟡", "🔴"


def band(x: float, yellow: float, red: float, higher_is_worse: bool = True) -> str:
    if not higher_is_worse:
        x, yellow, red = -x, -yellow, -red
    return RED if x > red else YELLOW if x > yellow else GREEN


def series(sym: str, start: str) -> pd.Series | None:
    try:
        return daily(sym, start=start)["Close"].dropna()
    except Exception as e:
        print(f"{sym}: {e}")
        return None


def main() -> None:
    p = Params.from_dict(CONFIG["strategy"])
    start = (pd.Timestamp.today() - pd.Timedelta(days=500)).strftime("%Y-%m-%d")
    rows: list[tuple[str, str, str, str]] = []      # (χρώμα, δείκτης, τιμή, εξήγηση)
    data: dict = {}

    vix = series("^VIX", start)
    if vix is not None:
        v = float(vix.iloc[-1])
        data["vix"] = v
        rows.append((band(v, 20, 30), "VIX (φόβος)", f"{v:.1f}",
                     "κάτω από 20 ήρεμα · 20–30 νευρικότητα · πάνω από 30 ένταση"))
    v3 = series("^VIX3M", start)
    if vix is not None and v3 is not None:
        r = float(vix.iloc[-1] / v3.reindex(vix.index).ffill().iloc[-1])
        data["vix_ratio"] = r
        rows.append((band(r, 0.95, 1.0), "VIX / VIX3M", f"{r:.2f}", "πάνω από 1 = ο φόβος για τον επόμενο μήνα είναι μεγαλύτερος"))
    hyg = series("HYG", start)
    if hyg is not None:
        gap = float(hyg.iloc[-1] / hyg.rolling(50).mean().iloc[-1] - 1)
        m1 = float(hyg.iloc[-1] / hyg.iloc[-22] - 1)
        data["hyg_gap"], data["hyg_1m"] = gap, m1
        color = RED if m1 < -0.03 else YELLOW if gap < 0 else GREEN
        rows.append((color, "Ομόλογα υψηλού ρίσκου (HYG)", f"{gap * 100:+.1f}% από τον μέσο 50 ημ. · μήνας {m1 * 100:+.1f}%",
                     "κάτω από τον μέσο = πιστωτική πίεση"))
    tnx = series("^TNX", start)
    if tnx is not None:
        lvl, ch = float(tnx.iloc[-1]), float(tnx.iloc[-1] - tnx.iloc[-22])
        data["us10y"], data["us10y_1m"] = lvl, ch
        rows.append((band(ch, 0.25, 0.40), "Επιτόκιο 10ετούς ΗΠΑ", f"{lvl:.2f}% · μήνας {ch:+.2f}",
                     "άνοδος πάνω από 0,40 σε έναν μήνα πιέζει τις μετοχές"))
    dxy = series("DX-Y.NYB", start)
    if dxy is not None:
        ch = float(dxy.iloc[-1] / dxy.iloc[-22] - 1)
        data["dollar_1m"] = ch
        rows.append((band(ch, 0.02, 0.04), "Δολάριο (DXY)", f"{float(dxy.iloc[-1]):.1f} · μήνας {ch * 100:+.1f}%",
                     "απότομη άνοδος = φυγή προς ασφάλεια"))
    ndx = series(CONFIG["market_symbol"], (pd.Timestamp.today() - pd.Timedelta(days=900)).strftime("%Y-%m-%d"))
    if ndx is not None:
        ema = _ema(ndx, p.ema_slow)
        dist = float(ndx.iloc[-1] / ema.iloc[-1] - 1)
        dd = float(ndx.iloc[-1] / ndx.iloc[-252:].max() - 1)
        data["ndx_vs_filter"], data["ndx_drawdown"] = dist, dd
        rows.append((RED if dist < 0 else YELLOW if dist < 0.03 else GREEN, f"Nasdaq-100 και φίλτρο του bot (EMA {p.ema_slow})",
                     f"{dist * 100:+.1f}%", "κάτω από 0 = το bot σταματά τις νέες αγορές· 0 έως +3% = κοντά στο φίλτρο"))
        rows.append((band(-dd, 0.10, 0.20), "Nasdaq-100 από το υψηλό του έτους", f"{dd * 100:+.1f}%",
                     "−10% διόρθωση · −20% πτωτική αγορά"))

    reds = sum(r[0] == RED for r in rows)
    yellows = sum(r[0] == YELLOW for r in rows)
    below_filter = data.get("ndx_vs_filter", 1) < 0
    level = RED if (reds >= 2 or below_filter) else YELLOW if (reds >= 1 or yellows >= 2) else GREEN
    words = {GREEN: "ήρεμα", YELLOW: "προσοχή", RED: "ένταση"}

    path = ROOT / "reports"
    path.mkdir(exist_ok=True)
    old = path / "macro.json"
    prev = json.loads(old.read_text(encoding="utf-8")).get("level") if old.exists() else None
    asof = max((s.index[-1] for s in (vix, hyg, tnx, ndx) if s is not None), default=pd.Timestamp.today())
    out = [f"# Θερμόμετρο κινδύνου ({asof:%d/%m/%Y})", "",
           f"**Συνολικά: {level} {words[level]}**" + (f" (χθες: {prev} {words.get(prev, '')})" if prev and prev != level else ""),
           "", "Μόνο για ενημέρωση: το bot δεν αλλάζει τίποτα από αυτά. Η δική του προστασία είναι το φίλτρο αγοράς, "
           "τα stop και το φρένο του −15%.", "",
           "| | Δείκτης | Τώρα | Τι σημαίνει |", "|---|---|---|---|"]
    comma = re.compile(r"(\d)\.(\d)")      # ελληνική υποδιαστολή
    out += ["| " + " | ".join((c, n, comma.sub(r"\1,\2", val), why)) + " |" for c, n, val, why in rows]
    out.append("")
    (path / "macro.md").write_text("\n".join(out), encoding="utf-8")
    old.write_text(json.dumps({"asof": f"{asof:%Y-%m-%d}", "level": level, "previous": prev, "changed": prev != level,
                               "values": data, "rows": [list(r) for r in rows]}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
