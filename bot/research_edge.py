"""Έρευνα: καλύτερη επιλογή σημάτων και πιο μακρινοί στόχοι, με πιο ευαίσθητο τεστ.

Δύο ερωτήσεις, και οι δύο με τους ίδιους κανόνες εισόδου του bot:
  1. ΣΕΙΡΑ ΕΠΙΛΟΓΗΣ. Όταν τα σήματα είναι περισσότερα από τα διαθέσιμα χρήματα, το bot παίρνει πρώτα όσα έχουν τη
     μεγαλύτερη άνοδο 3 μηνών. Υπάρχει καλύτερο κριτήριο;
  2. ΣΤΟΧΟΣ. Ο στόχος 3×ATR κόβει κάθε κέρδος στο +1,5R. Αποδίδει περισσότερο ένας πιο μακρινός στόχος;

Οι προηγούμενες έρευνες σύγκριναν το Sharpe ολόκληρων χαρτοφυλακίων σε τρία διαστήματα 5–11 ετών. Εκεί η τύχη
μετράει όσο και οι κανόνες, οπότε σχεδόν όλα έβγαιναν «μικτά». Εδώ η σύγκριση γίνεται ζευγαρωτά:
  • Χαρτοφυλάκιο: η διαφορά της ημερήσιας απόδοσης (παραλλαγή − σημερινό bot) τις ίδιες μέρες. Οι δύο εκδοχές
    έχουν τις περισσότερες συναλλαγές κοινές, άρα η διαφορά τους έχει πολύ λιγότερο θόρυβο από τα δύο Sharpe.
    Το p βγαίνει με block bootstrap (κομμάτια 21 συνεδριάσεων).
  • Με ίδιο ρίσκο: μια παραλλαγή που απλώς μένει περισσότερο καιρό επενδυμένη βγάζει περισσότερα σε ανοδική αγορά,
    αλλά με ανάλογα μεγαλύτερες διακυμάνσεις· αυτό δεν είναι βελτίωση. Γι' αυτό η απόδοση της παραλλαγής
    κλιμακώνεται πρώτα ώστε να έχει τις ίδιες ημερήσιες διακυμάνσεις με το σημερινό bot, και μετά συγκρίνεται.
    (Το βρήκε η δοκιμή με τυχαίες τιμές: εκεί ο πιο μακρινός στόχος «κέρδιζε» μόνο από τον παραπάνω χρόνο στην αγορά.)
  • Ανά σήμα: κάθε σήμα σαν να αγοράστηκε (όπως το εικονικό ημερολόγιο), χιλιάδες συναλλαγές αντί για 3 αριθμούς.

ΚΑΝΟΝΑΣ ΑΠΟΦΑΣΗΣ, γραμμένος πριν τρέξει η έρευνα. Μια παραλλαγή «περνά» μόνο αν ισχύουν ΟΛΑ:
  α) με ίδιο ρίσκο κερδίζει το σημερινό bot και στα τρία τεστ (A 2016–21, A 2022–σήμερα, B = 60 μετοχές που το
     bot δεν είχε δει),
  β) η διαφορά με ίδιο ρίσκο στα A και B μαζί έχει p < 0,05 (μονόπλευρο),
  γ) σε κανένα τεστ η χειρότερη πτώση δεν μεγαλώνει πάνω από 2 μονάδες,
  δ) συμφωνεί και η μέτρηση ανά σήμα, χωριστά στα A και στα B,
  ε) οι γειτονικές τιμές της ρύθμισης κερδίζουν κι αυτές το σημερινό bot (A και B μαζί).
Με 12 παραλλαγές, κάποια μπορεί να περάσει το (β) από τύχη· γι' αυτό σημειώνεται χωριστά ποιες περνούν και το
αυστηρό όριο p < 0,05 / 6 (διόρθωση για τις 6 παραλλαγές κάθε ομάδας).

Δεν αλλάζει τίποτα στο bot: όλα υπάρχουν μόνο εδώ.

Εκτέλεση:  python -m bot.research_edge
Γράφει το reports/research_edge.md
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import CONFIG, Costs, Risk, every_signal, fills, num, pct, run, stats
from .data import load_all
from .research_params import OTHER
from .strategy import Params, signals

ROOT = Path(__file__).resolve().parent.parent
TESTS = [("A 2016–21", "A", "2016-01-01", "2021-12-31"), ("A 2022–σήμερα", "A", "2022-01-01", None),
         ("B 2016–σήμερα", "B", "2016-01-01", None)]
N_BOOT = 2000     # επαναλήψεις bootstrap
BLOCK = 21        # συνεδριάσεις ανά κομμάτι (περίπου ένας μήνας)
SEED = 20261110
P_LEVEL = 0.05
DD_TOL = 0.02     # πόσο χειρότερη πτώση ανεχόμαστε σε κάθε τεστ
RANDOM_SEEDS = 5  # τυχαία σειρά επιλογής, για να φανεί πόσο αξίζει το σημερινό κριτήριο


# ───────────── Κριτήρια κατάταξης (όλα από δεδομένα έως το κλείσιμο της μέρας του σήματος) ─────────────
def f_mom(n: int):
    return lambda df: df["Close"] / df["Close"].shift(n) - 1


def f_mom_skip(n: int, skip: int):
    return lambda df: df["Close"].shift(skip) / df["Close"].shift(n) - 1


def f_high(n: int):
    return lambda df: df["Close"] / df["High"].rolling(n).max()


def f_momvol(n: int):
    return lambda df: (df["Close"] / df["Close"].shift(n) - 1) / df["Close"].pct_change().rolling(n).std()


def f_calm(n: int):
    return lambda df: -df["Close"].pct_change().rolling(n).std()


BASE_RANK = ("mom63", "Άνοδος 3 μηνών (το σημερινό κριτήριο)", f_mom(63))
RANKERS = [
    ("mom21", "Άνοδος 1 μήνα", f_mom(21), [("15 μέρες", f_mom(15)), ("30 μέρες", f_mom(30))]),
    ("mom126", "Άνοδος 6 μηνών", f_mom(126), [("5 μήνες", f_mom(105)), ("7 μήνες", f_mom(150))]),
    ("mom12_1", "Άνοδος 12 μηνών χωρίς τον τελευταίο μήνα", f_mom_skip(252, 21),
     [("9 μήνες", f_mom_skip(189, 21)), ("χωρίς τους 2 τελευταίους", f_mom_skip(252, 42))]),
    ("high52", "Πιο κοντά στο υψηλό 52 εβδομάδων", f_high(252), [("9 μήνες", f_high(189)), ("15 μήνες", f_high(315))]),
    ("momvol", "Άνοδος 3 μηνών σε σχέση με τις διακυμάνσεις", f_momvol(63),
     [("50 μέρες", f_momvol(50)), ("80 μέρες", f_momvol(80))]),
    ("calm", "Οι πιο ήρεμες μετοχές (μικρότερες διακυμάνσεις 3 μηνών)", f_calm(63),
     [("2 μήνες", f_calm(42)), ("4 μήνες", f_calm(90))]),
]
EXITS = [
    ("tp4", "Στόχος 4×ATR", {"tp_atr": 4.0}, [("3,5×ATR", {"tp_atr": 3.5}), ("4,5×ATR", {"tp_atr": 4.5})]),
    ("tp5", "Στόχος 5×ATR", {"tp_atr": 5.0}, [("4,5×ATR", {"tp_atr": 4.5}), ("5,5×ATR", {"tp_atr": 5.5})]),
    ("tp6", "Στόχος 6×ATR", {"tp_atr": 6.0}, [("5,5×ATR", {"tp_atr": 5.5}), ("7×ATR", {"tp_atr": 7.0})]),
    ("tpx", "Χωρίς στόχο (έξοδος με stop, τάση ή 20 μέρες)", {"tp_atr": 99.0}, [("στόχος 8×ATR", {"tp_atr": 8.0})]),
    ("tp5_30", "Στόχος 5×ATR, έως 30 μέρες", {"tp_atr": 5.0, "max_bars": 30},
     [("έως 25 μέρες", {"tp_atr": 5.0, "max_bars": 25}), ("έως 35 μέρες", {"tp_atr": 5.0, "max_bars": 35})]),
    ("tpx_30", "Χωρίς στόχο, έως 30 μέρες", {"tp_atr": 99.0, "max_bars": 30},
     [("έως 25 μέρες", {"tp_atr": 99.0, "max_bars": 25}), ("έως 35 μέρες", {"tp_atr": 99.0, "max_bars": 35})]),
]


# ───────────── Στατιστική ─────────────
def log_ret(eq: pd.Series) -> pd.Series:
    return np.log(eq.astype(float)).diff().dropna()


def block_boot_means(d: np.ndarray, n_boot: int = N_BOOT, block: int = BLOCK, seed: int = SEED) -> np.ndarray:
    """Κυκλικό block bootstrap του μέσου μιας ημερήσιας σειράς (κρατά τη συσχέτιση διαδοχικών ημερών)."""
    d = np.asarray(d, dtype=float)
    n = len(d)
    if n == 0:
        return np.zeros(n_boot)
    rng = np.random.default_rng(seed)
    k = math.ceil(n / block)
    starts = rng.integers(0, n, size=(n_boot, k))
    idx = ((starts[:, :, None] + np.arange(block)[None, None, :]) % n).reshape(n_boot, -1)[:, :n]
    return d[idx].mean(axis=1)


def paired(diff: pd.Series, n_boot: int = N_BOOT) -> dict:
    """Διαφορά ημερήσιας λογαριθμικής απόδοσης → ετήσια διαφορά, εύρος 90% και μονόπλευρο p (διαφορά ≤ 0)."""
    d = diff.dropna().to_numpy()
    if len(d) < 2 * BLOCK:
        return {"diff": 0.0, "lo": 0.0, "hi": 0.0, "p": 1.0, "days": int(len(d))}
    b = block_boot_means(d, n_boot) * 252
    return {"diff": float(d.mean() * 252), "lo": float(np.quantile(b, 0.05)), "hi": float(np.quantile(b, 0.95)),
            "p": float((b <= 0).mean()), "days": int(len(d))}


def month_index(months: pd.Series, n_boot: int = N_BOOT, seed: int = SEED) -> tuple[list, np.ndarray]:
    """Οι μήνες των συναλλαγών και οι επαναληπτικές επιλογές τους (bootstrap ανά μήνα: οι συναλλαγές του ίδιου
    μήνα πάνε μαζί, γιατί κερδίζουν ή χάνουν μαζί με την αγορά)."""
    ms = sorted(set(months))
    rng = np.random.default_rng(seed)
    return ms, rng.integers(0, max(len(ms), 1), size=(n_boot, max(len(ms), 1)))


def month_sum(tab: pd.DataFrame, values: pd.Series, ms: list) -> np.ndarray:
    return values.groupby(tab["month"]).sum().reindex(ms).fillna(0.0).to_numpy(dtype=float)


def _ratio(numer: np.ndarray, denom: np.ndarray) -> np.ndarray:
    return np.divide(numer, denom, out=np.zeros_like(numer, dtype=float), where=denom != 0)


def sides(tab: pd.DataFrame, col: str) -> pd.Series:
    """+1 = στο πάνω μισό των σημάτων της ίδιας μέρας (και ίδιας ομάδας μετοχών), −1 = στο κάτω μισό, 0 = αλλιώς."""
    side = pd.Series(0, index=tab.index, dtype=int)
    for _, g in tab.dropna(subset=[col]).groupby(["uni", "sig_date"]):
        h = len(g) // 2
        if h == 0:
            continue
        order = g.sort_values([col, "symbol"], ascending=[False, True]).index
        side.loc[order[:h]] = 1
        side.loc[order[-h:]] = -1
    return side


def spread(tab: pd.DataFrame, col: str, y: str, ms: list | None = None, idx: np.ndarray | None = None) -> dict:
    """Μέσο αποτέλεσμα του πάνω μισού μείον του κάτω μισού, για σήματα που ανταγωνίζονται την ίδια μέρα."""
    sd = sides(tab, col)
    top, bot = tab[y].where(sd == 1, 0.0), tab[y].where(sd == -1, 0.0)
    nt, nb = int((sd == 1).sum()), int((sd == -1).sum())
    out = {"spread": float(top.sum() / nt - bot.sum() / nb) if nt and nb else 0.0, "pairs": nt}
    if ms is not None and idx is not None:
        st, sb = month_sum(tab, top, ms), month_sum(tab, bot, ms)
        ct, cb = month_sum(tab, (sd == 1).astype(float), ms), month_sum(tab, (sd == -1).astype(float), ms)
        out["boot"] = _ratio(st[idx].sum(1), ct[idx].sum(1)) - _ratio(sb[idx].sum(1), cb[idx].sum(1))
    return out


def efficiency(tab: pd.DataFrame, risk_per_trade: float, ms: list | None = None, idx: np.ndarray | None = None) -> dict:
    """Απόδοση ανά έτος αν τα χρήματα ήταν συνεχώς επενδυμένα: κέρδος (ρίσκο × R) προς κεφάλαιο × μέρες που δέσμευσε.
    Έτσι μια έξοδος που κρατά τις θέσεις περισσότερο κρίνεται δίκαια: πρέπει να βγάζει περισσότερα ανά μέρα."""
    gain, used = tab["r"] * risk_per_trade, tab["days"] * tab["cap"]
    out = {"eff": float(252 * gain.sum() / used.sum()) if used.sum() > 0 else 0.0}
    if ms is not None and idx is not None:
        g, u = month_sum(tab, gain, ms), month_sum(tab, used, ms)
        out["boot"] = 252 * _ratio(g[idx].sum(1), u[idx].sum(1))
    return out


def compare_curves(v: dict, b: dict, n_boot: int = N_BOOT) -> dict:
    """Ζευγαρωτή σύγκριση μιας παραλλαγής (v) με το σημερινό bot (b)· και τα δύο: τεστ → {stats, lr, invested}.
    «raw» = σκέτη διαφορά απόδοσης. Τα υπόλοιπα είναι με ίδιο ρίσκο: η απόδοση της παραλλαγής πολλαπλασιάζεται
    ώστε οι ημερήσιες διακυμάνσεις της να είναι ίσες με του bot."""
    t1, t2, t3 = (t for t, *_ in TESTS)

    def by_uni(c: dict) -> dict:
        return {"A": pd.concat([c[t1]["lr"], c[t2]["lr"]]).sort_index(), "B": c[t3]["lr"]}

    lv, lb = by_uni(v), by_uni(b)
    k = {u: float(lb[u].std() / lv[u].std()) if lv[u].std() > 0 else 1.0 for u in lv}
    raw = {u: (lv[u] - lb[u]).dropna() for u in lv}
    adj = {u: (k[u] * lv[u] - lb[u]).dropna() for u in lv}

    def pool(d: dict) -> pd.Series:
        return pd.concat([d["A"], d["B"]], axis=1, join="inner").mean(axis=1)

    def per_test(d: dict) -> dict:
        parts = {t1: d["A"].reindex(v[t1]["lr"].index), t2: d["A"].reindex(v[t2]["lr"].index), t3: d["B"]}
        return {t: float(x.dropna().mean() * 252) if x.notna().any() else 0.0 for t, x in parts.items()}

    return {"tests": per_test(adj), "raw_tests": per_test(raw), "scale": k,
            "A": paired(adj["A"], n_boot), "B": paired(adj["B"], n_boot),
            "pooled": paired(pool(adj), n_boot), "raw": paired(pool(raw), n_boot),
            "stats": {t: v[t]["stats"] for t, *_ in TESTS},
            "invested": {t: v[t]["invested"] for t, *_ in TESTS},
            "dd_ok": all(v[t]["stats"]["max_dd"] >= b[t]["stats"]["max_dd"] - DD_TOL for t, *_ in TESTS)}


# ───────────── Το εργαστήριο: προσομοιώσεις και πίνακες σημάτων ─────────────
class Lab:
    def __init__(self, universes: dict, market: pd.DataFrame, p: Params, risk: Risk, costs: Costs, fl: dict,
                 n_boot: int = N_BOOT):
        self.U, self.market, self.p, self.risk, self.costs, self.fl, self.n_boot = universes, market, p, risk, costs, fl, n_boot
        self.data = {s: df for u in universes.values() for s, df in u.items()}
        self.order = {s: i for i, s in enumerate(sorted(self.data))}
        self.base_sigs = {s: signals(df, p) for s, df in self.data.items()}
        self._curves: dict = {}

    # κατάταξη: το backtest ταξινομεί τα σήματα με τη στήλη mom63, οπότε την αντικαθιστούμε
    def rank_by(self, fn):
        cache: dict = {}

        def mod(s: str, sg: pd.DataFrame) -> pd.DataFrame:
            if s not in cache:
                cache[s] = fn(self.data[s]).replace([np.inf, -np.inf], np.nan)
            return sg.assign(mom63=cache[s].reindex(sg.index).fillna(-1e9))
        return mod

    def rank_random(self, seed: int):
        def fn(df: pd.DataFrame) -> pd.Series:
            return pd.Series(np.random.default_rng([SEED, seed, len(df), int(df["Close"].iloc[0] * 1000)]).random(len(df)),
                             index=df.index)
        return self.rank_by(fn)

    def curves(self, key: str, pchg: dict | None = None, sig_mod=None) -> dict:
        """Οι τρεις προσομοιώσεις χαρτοφυλακίου μιας παραλλαγής: στατιστικά και ημερήσιες αποδόσεις."""
        if key not in self._curves:
            p = replace(self.p, **(pchg or {}))
            out = {}
            for label, u, start, end in TESTS:
                res = run(self.U[u], p, self.risk, self.costs, start, end, self.market, sig_mod=sig_mod, **self.fl)
                out[label] = {"stats": stats(res), "lr": log_ret(res["equity"]),
                              "invested": float(1 - (res["cash"] / res["equity"]).clip(0, 1).mean())}
            self._curves[key] = out
        return self._curves[key]

    def compare(self, key: str, base_key: str = "base") -> dict:
        return compare_curves(self._curves[key], self._curves[base_key], self.n_boot)

    def signal_table(self, pchg: dict | None = None, feats: dict | None = None) -> pd.DataFrame:
        """Κάθε σήμα σαν να αγοράστηκε (χωρίς όριο θέσεων ή κεφαλαίου), με τα χαρακτηριστικά της μέρας του σήματος."""
        p = replace(self.p, **(pchg or {}))
        rows = []
        for u, data in self.U.items():
            res = run(data, p, every_signal(self.risk), self.costs, "2016-01-01", None, self.market, **self.fl)
            fcols = {name: {s: fn(df).replace([np.inf, -np.inf], np.nan) for s, df in data.items()}
                     for name, fn in (feats or {}).items()}
            for t in res["trades"]:
                s, ix = t["symbol"], data[t["symbol"]].index
                i = int(ix.get_loc(t["entry_date"]))
                j = max(int(ix.searchsorted(t["exit_date"], side="right")) - 1, i)
                if i == 0:
                    continue
                sd = ix[i - 1]
                dist_pct = (t["entry"] - float(self.base_sigs[s].at[sd, "stop"])) / t["entry"]
                if not dist_pct > 0:
                    continue
                row = {"uni": u, "symbol": s, "sig_date": sd, "month": f"{t['entry_date']:%Y-%m}", "r": t["r"],
                       "pct": t["pnl"] / (t["qty"] * t["entry"]), "reason": t["reason"],
                       "kind": self.base_sigs[s].at[sd, "kind"],
                       "days": max((j - i) + (0.5 if t["reason"] in ("Stop", "Στόχος") else 0.0), 0.5),
                       "cap": min(self.risk.max_notional, self.risk.risk_per_trade / dist_pct)}
                for name, by_sym in fcols.items():
                    val = by_sym[s].get(sd, np.nan)
                    row[name] = float(val) if pd.notna(val) else np.nan
                rows.append(row)
        return pd.DataFrame(rows)


# ───────────── Ετυμηγορία ─────────────
def verdict(cmp: dict, sig_ok: bool, family: int) -> tuple[str, bool]:
    """Ο κανόνας απόφασης του docstring, σημεία (α)–(δ). Το (ε), οι γειτονικές τιμές, ελέγχεται μόνο για όσες περνούν."""
    pool = cmp["pooled"]
    wins3 = all(v > 0 for v in cmp["tests"].values())
    if wins3 and pool["p"] < P_LEVEL and cmp["dd_ok"] and sig_ok:
        return ("✅ περνά" + (" (και το αυστηρό όριο)" if pool["p"] < P_LEVEL / family else ""), True)
    if pool["diff"] < 0 and pool["p"] > 1 - P_LEVEL:
        return "❌ χειρότερο", False
    why = []
    if not wins3:
        why.append("με ίδιο ρίσκο χάνει σε " + ", ".join(t for t, v in cmp["tests"].items() if v <= 0))
    if pool["p"] >= P_LEVEL:
        why.append(f"p = {num(pool['p'])}")
    if not cmp["dd_ok"]:
        why.append("μεγαλύτερη πτώση")
    if not sig_ok:
        why.append("δεν συμφωνεί η μέτρηση ανά σήμα")
    return "➖ δεν αποδείχθηκε (" + "· ".join(why) + ")", False


def cell(st: dict) -> str:
    return f"{pct(st['cagr'])} / {pct(st['max_dd'])} / {num(st['sharpe'])}" if st else "–"


def diff_cells(cmp: dict) -> str:
    pool = cmp["pooled"]
    return (f"{pct(cmp['raw']['diff'])} | " + " / ".join(pct(v) for v in cmp["tests"].values()) +
            f" | {pct(pool['diff'])} ({pct(pool['lo'])} έως {pct(pool['hi'])}) | {num(pool['p'])}")


KEYS = ("tests", "raw_tests", "scale", "A", "B", "pooled", "raw", "stats", "invested", "dd_ok")
PORT_HEAD = ("| Παραλλαγή | " + " | ".join(f"{t}: ανά έτος / πτώση / Sharpe" for t, *_ in TESTS) +
             " | Σκέτη διαφορά ανά έτος (A και B μαζί) | Με ίδιο ρίσκο: " + " / ".join(t for t, *_ in TESTS) +
             " | Με ίδιο ρίσκο, A και B μαζί (εύρος 90%) | p | Ετυμηγορία |\n|---|" + "---|" * (len(TESTS) + 5))


def main() -> None:
    p = Params.from_dict(CONFIG["strategy"])
    rk = CONFIG["risk"]
    risk = Risk(rk["risk_per_trade"], rk["max_positions"], rk["max_total_risk"], rk["max_notional"], True)
    costs = Costs(**CONFIG["costs"])
    market_sym = CONFIG["market_symbol"]
    uni_a = list(CONFIG["universe"])
    uni_b = [s for s in OTHER if s not in uni_a]
    raw = {}
    for s in sorted(set(uni_a + uni_b + [market_sym])):
        try:
            raw.update(load_all([s]))
        except Exception as e:
            print(f"χωρίς δεδομένα {s}: {e}")
    market = raw[market_sym]
    U = {"A": {s: raw[s] for s in uni_a if s in raw},
         "B": {s: raw[s] for s in uni_b if s in raw and raw[s].index[0] <= pd.Timestamp("2015-01-01")}}
    lab = Lab(U, market, p, risk, costs, fills(), N_BOOT)
    summary: dict = {"data_to": f"{market.index[-1]:%Y-%m-%d}", "rankers": {}, "exits": {}}

    base = lab.curves("base")
    lab.curves("check", sig_mod=lab.rank_by(BASE_RANK[2]))   # η ίδια κατάταξη μέσα από τον μηχανισμό της έρευνας
    exact = all(abs(v) < 1e-12 for v in lab.compare("check")["tests"].values())

    # ── Τα σήματα, ένα ένα ──
    feats = {BASE_RANK[0]: BASE_RANK[2]} | {k: fn for k, _, fn, _ in RANKERS}
    sig = lab.signal_table(feats=feats)
    ms, idx = month_index(sig["month"], lab.n_boot)
    taken = sum(base[t]["stats"]["trades"] for t, *_ in TESTS)

    def by_uni(tab: pd.DataFrame, u: str | None) -> pd.DataFrame:
        return tab if u is None else tab[tab["uni"] == u]

    out = ["# Έρευνα: καλύτερη επιλογή σημάτων και πιο μακρινοί στόχοι", "",
           f"Δεδομένα έως {market.index[-1]:%d/%m/%Y}. A = οι {len(U['A'])} μετοχές του bot· B = {len(U['B'])} μεγάλες "
           "μετοχές που το bot δεν έχει δει ποτέ. Ίδια όρια ρίσκου, κόστη και εκτέλεση στόχου/stop με το bot.", "",
           "**Τι αλλάζει σε σχέση με τις παλιές έρευνες:** η σύγκριση γίνεται ζευγαρωτά, μέρα προς μέρα, με το "
           "σημερινό bot. Έτσι φαίνονται και μικρές διαφορές που πριν χάνονταν μέσα στην τύχη. Το **p** είναι η "
           "πιθανότητα να βλέπαμε τέτοια διαφορά μόνο από τύχη: κάτω από 0,05 μετράει ως ένδειξη.", "",
           "**«Με ίδιο ρίσκο»:** μια παραλλαγή που απλώς μένει περισσότερο καιρό επενδυμένη βγάζει περισσότερα σε "
           "ανοδική αγορά, αλλά με ανάλογα μεγαλύτερες διακυμάνσεις. Αυτό δεν είναι βελτίωση. Γι' αυτό η απόδοσή της "
           "φέρνεται πρώτα στις ίδιες ημερήσιες διακυμάνσεις με το σημερινό bot και μετά συγκρίνεται.", "",
           "**Μια παραλλαγή περνά μόνο αν:** με ίδιο ρίσκο κερδίζει το σημερινό bot και στα τρία τεστ, έχει p < 0,05 "
           f"στα A και B μαζί, δεν μεγαλώνει τη χειρότερη πτώση πάνω από {DD_TOL * 100:.0f} μονάδες, συμφωνεί η "
           "μέτρηση ανά σήμα (χωριστά στα A και στα B) και οι γειτονικές τιμές κερδίζουν κι αυτές. Ο κανόνας γράφτηκε "
           "πριν τρέξει η έρευνα.", "",
           f"Έλεγχος ακρίβειας: η κατάταξη με «άνοδο 3 μηνών» εδώ δίνει {'ακριβώς το ίδιο' if exact else 'ΔΙΑΦΟΡΕΤΙΚΟ'} "
           "αποτέλεσμα με το backtest του bot.", "",
           "| | " + " | ".join(f"{t}: ανά έτος / πτώση / Sharpe" for t, *_ in TESTS) + " |",
           "|---|" + "---|" * len(TESTS),
           "| **Σημερινό bot** | " + " | ".join(cell(base[t]["stats"]) for t, *_ in TESTS) + " |", "",
           "## 1. Τα σήματα, ένα ένα", "",
           "Κάθε σήμα σαν να αγοράστηκε, χωρίς όριο θέσεων ή κεφαλαίου (όπως το εικονικό ημερολόγιο). "
           "R = φορές το ρίσκο της συναλλαγής.", "",
           "| | Σήματα | Επιτυχία | Μέσο R | Μέσο αποτέλεσμα | Μέρες σε θέση |", "|---|---|---|---|---|---|"]
    facts = {}
    for name, part in [("A, όλα", by_uni(sig, "A")), ("B, όλα", by_uni(sig, "B")),
                       ("A και B, σπάσιμο υψηλού", sig[sig["kind"] == "breakout"]),
                       ("A και B, υποχώρηση", sig[sig["kind"] == "pullback"])]:
        if part.empty:
            continue
        facts[name] = {"n": len(part), "win": float((part["r"] > 0).mean()), "avg_r": float(part["r"].mean()),
                       "avg_pct": float(part["pct"].mean()), "days": float(part["days"].mean())}
        f = facts[name]
        out.append(f"| {name} | {f['n']} | {f['win'] * 100:.0f}% | {num(f['avg_r'])} | {pct(f['avg_pct'])} | {num(f['days'], 1)} |")
    summary["signals"] = facts
    out += ["", f"Το χαρτοφυλάκιο του bot έκανε {taken} συναλλαγές στα τρία τεστ, από {len(sig)} σήματα: τα υπόλοιπα "
            "δεν χώρεσαν. Γι' αυτό μετράει ποια διαλέγει πρώτα.", ""]

    # ── 2. Σειρά επιλογής ──
    base_sp = {u: spread(by_uni(sig, u), BASE_RANK[0], "pct") for u in ("A", "B")}
    base_all = spread(sig, BASE_RANK[0], "pct", ms, idx)
    out += ["## 2. Σειρά επιλογής σημάτων", "",
            "**Ανά σήμα.** Τις μέρες με δύο ή περισσότερα σήματα: μέσο αποτέλεσμα όσων το κριτήριο βάζει στο πάνω "
            "μισό, μείον όσων βάζει στο κάτω μισό. Θετικό = το κριτήριο διαλέγει σωστά. Το p δείχνει αν ξεχωρίζει "
            "καλύτερα από το σημερινό κριτήριο.", "",
            "| Κριτήριο | A: πάνω − κάτω μισό | B: πάνω − κάτω μισό | A και B μαζί | Σε R (A και B) | p έναντι σημερινού |",
            "|---|---|---|---|---|---|",
            f"| {BASE_RANK[1]} | {pct(base_sp['A']['spread'])} | {pct(base_sp['B']['spread'])} | "
            f"{pct(base_all['spread'])} | {num(spread(sig, BASE_RANK[0], 'r')['spread'])} | – |"]
    sig_rank = {}
    for key, label, _, _ in RANKERS:
        sp = {u: spread(by_uni(sig, u), key, "pct") for u in ("A", "B")}
        al = spread(sig, key, "pct", ms, idx)
        p_vs = float(((al["boot"] - base_all["boot"]) <= 0).mean())
        sig_rank[key] = {"A": sp["A"]["spread"], "B": sp["B"]["spread"], "all": al["spread"], "p_vs_base": p_vs,
                         "ok": sp["A"]["spread"] > base_sp["A"]["spread"] and sp["B"]["spread"] > base_sp["B"]["spread"]}
        out.append(f"| {label} | {pct(sp['A']['spread'])} | {pct(sp['B']['spread'])} | {pct(al['spread'])} | "
                   f"{num(spread(sig, key, 'r')['spread'])} | {num(p_vs)} |")
    out += ["", f"Σήματα που μπήκαν στη σύγκριση (πάνω μισό): {base_all['pairs']}.", "",
            "**Χαρτοφυλάκιο.** Το ίδιο bot, μόνο με άλλη σειρά επιλογής.", "", PORT_HEAD]
    passed: list[tuple[str, str, str]] = []   # (ομάδα, κλειδί, όνομα)
    for key, label, fn, _ in RANKERS:
        lab.curves(key, sig_mod=lab.rank_by(fn))
        cmp = lab.compare(key)
        v, ok = verdict(cmp, sig_rank[key]["ok"], len(RANKERS))
        summary["rankers"][key] = {"label": label, "signal": sig_rank[key], "verdict": v,
                                   **{k: cmp[k] for k in KEYS}}
        if ok:
            passed.append(("rankers", key, label))
        out.append(f"| {label} | " + " | ".join(cell(cmp["stats"][t]) for t, *_ in TESTS) + f" | {diff_cells(cmp)} | {v} |")
    rnd = []
    for k in range(RANDOM_SEEDS):
        lab.curves(f"rnd{k}", sig_mod=lab.rank_random(k))
        rnd.append(lab.compare(f"rnd{k}"))
    rnd_tests = {t: float(np.mean([r["tests"][t] for r in rnd])) for t, *_ in TESTS}
    rnd_pool = float(np.mean([r["pooled"]["diff"] for r in rnd]))
    rnd_raw = float(np.mean([r["raw"]["diff"] for r in rnd]))
    summary["random_order"] = {"tests": rnd_tests, "pooled": rnd_pool, "raw": rnd_raw,
                               "runs_raw": [r["raw"]["diff"] for r in rnd]}
    out += [f"| Τυχαία σειρά (έλεγχος, μέσος όρος {RANDOM_SEEDS} δοκιμών) | – | – | – | {pct(rnd_raw)} | "
            + " / ".join(pct(v) for v in rnd_tests.values()) + f" | {pct(rnd_pool)} | – | – |", "",
            f"Η τυχαία σειρά δείχνει πόσο αξίζει το σημερινό κριτήριο: {pct(-rnd_raw)} τον χρόνο πάνω από την τύχη "
            f"(οι {RANDOM_SEEDS} δοκιμές: " + ", ".join(pct(r["raw"]["diff"]) for r in rnd) + ").", ""]

    # ── 3. Στόχος ──
    eff_base = {u: efficiency(by_uni(sig, u), risk.risk_per_trade)["eff"] for u in ("A", "B")}
    out += ["## 3. Πιο μακρινός στόχος", "",
            "**Ανά σήμα.** «Απόδοση επενδυμένου κεφαλαίου»: τι θα έβγαζαν τα χρήματα τον χρόνο αν ήταν συνεχώς σε "
            "θέσεις. Μια έξοδος που κρατά τις θέσεις περισσότερο πρέπει να βγάζει και περισσότερα ανά μέρα, αλλιώς "
            "απλώς δεσμεύει τα χρήματα. Το p συγκρίνει με τον σημερινό στόχο στα A και B μαζί.", "",
            "Η τελευταία στήλη δείχνει πόσο από τον λογαριασμό είναι κατά μέσο όρο σε θέσεις στο χαρτοφυλάκιο.", "",
            "| Έξοδος | Σήματα | Επιτυχία | Μέσο R | Μέρες σε θέση | A: απόδοση επενδυμένου κεφαλαίου | B: απόδοση επενδυμένου κεφαλαίου | p έναντι σημερινού | Επενδυμένο κατά μέσο όρο: "
            + " / ".join(t for t, *_ in TESTS) + " |",
            "|---|---|---|---|---|---|---|---|---|",
            f"| Στόχος 3×ATR, έως 20 μέρες (σήμερα) | {len(sig)} | {(sig['r'] > 0).mean() * 100:.0f}% | "
            f"{num(sig['r'].mean())} | {num(sig['days'].mean(), 1)} | {pct(eff_base['A'])} | {pct(eff_base['B'])} | – | "
            + " / ".join(f"{base[t]['invested'] * 100:.0f}%" for t, *_ in TESTS) + " |"]
    sig_exit, port_rows = {}, []
    for key, label, pchg, _ in EXITS:
        tab = lab.signal_table(pchg)
        ms2, idx2 = month_index(pd.concat([sig["month"], tab["month"]]), lab.n_boot)
        eb, ev = efficiency(sig, risk.risk_per_trade, ms2, idx2), efficiency(tab, risk.risk_per_trade, ms2, idx2)
        eu = {u: efficiency(by_uni(tab, u), risk.risk_per_trade)["eff"] for u in ("A", "B")}
        p_vs = float(((ev["boot"] - eb["boot"]) <= 0).mean())
        sig_exit[key] = {"n": len(tab), "win": float((tab["r"] > 0).mean()), "avg_r": float(tab["r"].mean()),
                         "days": float(tab["days"].mean()), "A": eu["A"], "B": eu["B"], "p_vs_base": p_vs,
                         "ok": eu["A"] > eff_base["A"] and eu["B"] > eff_base["B"]}
        e = sig_exit[key]
        lab.curves(key, pchg)
        cmp = lab.compare(key)
        out.append(f"| {label} | {e['n']} | {e['win'] * 100:.0f}% | {num(e['avg_r'])} | {num(e['days'], 1)} | "
                   f"{pct(e['A'])} | {pct(e['B'])} | {num(p_vs)} | "
                   + " / ".join(f"{x * 100:.0f}%" for x in cmp["invested"].values()) + " |")
        v, ok = verdict(cmp, e["ok"], len(EXITS))
        summary["exits"][key] = {"label": label, "signal": e, "verdict": v, **{k: cmp[k] for k in KEYS}}
        if ok:
            passed.append(("exits", key, label))
        port_rows.append(f"| {label} | " + " | ".join(cell(cmp["stats"][t]) for t, *_ in TESTS) + f" | {diff_cells(cmp)} | {v} |")
    out += ["", "**Χαρτοφυλάκιο.** Το ίδιο bot, μόνο με άλλο στόχο ή διάρκεια.", "", PORT_HEAD, *port_rows, ""]

    # ── 4. Γειτονικές τιμές για όσες πέρασαν ──
    robust: list[tuple[str, str, str]] = []
    if passed:
        out += ["## 4. Γειτονικές τιμές για όσες πέρασαν", "",
                "Μια πραγματική βελτίωση δεν εξαφανίζεται αν αλλάξει λίγο ο αριθμός. Κάθε γειτονική τιμή πρέπει να "
                "κερδίζει το σημερινό bot με ίδιο ρίσκο στα A και B μαζί, χωρίς μεγαλύτερη πτώση.", "",
                "| Παραλλαγή | Γειτονική τιμή | Σκέτη διαφορά ανά έτος (A και B μαζί) | Με ίδιο ρίσκο: "
                + " / ".join(t for t, *_ in TESTS) + " | Με ίδιο ρίσκο, A και B μαζί (εύρος 90%) | p | Κρατά; |",
                "|---|---|---|---|---|---|---|"]
        for fam, key, label in passed:
            nbs = next(x[3] for x in (RANKERS if fam == "rankers" else EXITS) if x[0] == key)
            all_ok = True
            for nb_label, nb in nbs:
                nk = f"{key}|{nb_label}"
                if fam == "rankers":
                    lab.curves(nk, sig_mod=lab.rank_by(nb))
                else:
                    lab.curves(nk, nb)
                cmp = lab.compare(nk)
                ok = cmp["pooled"]["diff"] > 0 and cmp["dd_ok"]
                all_ok &= ok
                summary[fam][key].setdefault("neighbors", {})[nb_label] = {k: cmp[k] for k in ("tests", "pooled", "raw", "dd_ok")}
                out.append(f"| {label} | {nb_label} | {diff_cells(cmp)} | {'ναι' if ok else 'όχι'} |")
            summary[fam][key]["robust"] = all_ok
            if all_ok:
                robust.append((fam, key, label))
        out.append("")

    # ── 5. Συνδυασμός, αν πέρασε κάτι και από τις δύο ομάδες ──
    r_ok, e_ok = [x for x in robust if x[0] == "rankers"], [x for x in robust if x[0] == "exits"]
    if r_ok and e_ok:
        best_r = min(r_ok, key=lambda x: summary["rankers"][x[1]]["pooled"]["p"])
        best_e = min(e_ok, key=lambda x: summary["exits"][x[1]]["pooled"]["p"])
        fn = next(x[2] for x in RANKERS if x[0] == best_r[1])
        pchg = next(x[2] for x in EXITS if x[0] == best_e[1])
        lab.curves("combo", pchg, lab.rank_by(fn))
        cmp = lab.compare("combo")
        summary["combo"] = {"ranker": best_r[2], "exit": best_e[2], **{k: cmp[k] for k in KEYS}}
        out += ["## 5. Οι δύο αλλαγές μαζί", "", PORT_HEAD,
                f"| {best_r[2]} + {best_e[2]} | " + " | ".join(cell(cmp["stats"][t]) for t, *_ in TESTS) +
                f" | {diff_cells(cmp)} | {'κερδίζει το σημερινό' if cmp['pooled']['diff'] > 0 and cmp['dd_ok'] else 'δεν κερδίζει'} |", ""]

    # ── Συμπέρασμα ──
    out += ["## Συμπέρασμα", ""]
    if robust:
        out.append("Πέρασαν όλους τους ελέγχους (τρία τεστ, p, πτώση, μέτρηση ανά σήμα, γειτονικές τιμές): **" +
                   ", ".join(x[2] for x in robust) + "**. Χρειάζονται την έγκρισή σου πριν μπουν στο bot.")
    elif passed:
        out.append("Πέρασαν τα τεστ αλλά όχι τον έλεγχο γειτονικών τιμών: " + ", ".join(x[2] for x in passed) +
                   ". Δεν αρκεί για αλλαγή στο bot.")
    else:
        out.append("**Καμία παραλλαγή δεν πέρασε όλους τους ελέγχους.** Το «δεν αποδείχθηκε» δεν σημαίνει ότι μια "
                   "παραλλαγή είναι χειρότερη· σημαίνει ότι με τα δεδομένα που έχουμε δεν ξεχωρίζει από την τύχη.")
    n_all = len(RANKERS) + len(EXITS)
    out += ["", f"Σημείωση: δοκιμάστηκαν {n_all} παραλλαγές. Αν καμία δεν είχε πραγματική διαφορά, περίπου "
            f"{n_all * P_LEVEL:.1f} θα έβγαζαν p < 0,05 από τύχη· γι' αυτό μετράνε και οι υπόλοιποι έλεγχοι. Ακόμα και "
            "για όσες περνούν, η πραγματική διαφορά είναι πιθανότατα μικρότερη από αυτή που μετρήθηκε, γιατί "
            "διαλέγουμε τις καλύτερες από πολλές.", ""]
    summary["passed"] = [x[2] for x in passed]
    summary["robust"] = [x[2] for x in robust]

    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items() if k != "boot"}
        return float(o) if isinstance(o, (np.floating, np.integer)) else o

    path = ROOT / "reports"
    path.mkdir(exist_ok=True)
    (path / "research_edge.md").write_text("\n".join(out), encoding="utf-8")
    (path / "research_edge.json").write_text(json.dumps(clean(summary), ensure_ascii=False, indent=1, default=float),
                                             encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
