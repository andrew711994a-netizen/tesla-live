"""Πελάτης για το REST API της Capital.com (https://open-api.capital.com)."""

from __future__ import annotations

import time

import requests

URLS = {
    "demo": "https://demo-api-capital.backend-capital.com/api/v1",
    "live": "https://api-capital.backend-capital.com/api/v1",
}


class CapitalError(RuntimeError):
    pass


class Capital:
    def __init__(self, api_key: str, identifier: str, password: str, env: str = "demo"):
        if env not in URLS:
            raise CapitalError(f"Άγνωστο περιβάλλον {env!r}")
        self.base = URLS[env]
        self.env = env
        self.s = requests.Session()
        self.s.headers.update({"X-CAP-API-KEY": api_key, "Content-Type": "application/json"})
        self._login(identifier, password)

    # ── βασικά ──
    def _login(self, identifier: str, password: str) -> None:
        r = self.s.post(f"{self.base}/session",
                        json={"identifier": identifier, "password": password, "encryptedPassword": False},
                        timeout=20)
        if r.status_code != 200:
            raise CapitalError(f"Αποτυχία σύνδεσης ({r.status_code}): {r.text[:200]}")
        self.s.headers.update({"CST": r.headers["CST"], "X-SECURITY-TOKEN": r.headers["X-SECURITY-TOKEN"]})

    def _req(self, method: str, path: str, **kw) -> dict:
        for attempt in range(3):
            r = self.s.request(method, f"{self.base}{path}", timeout=20, **kw)
            if r.status_code == 429:  # όριο αιτημάτων
                time.sleep(1 + attempt)
                continue
            if r.status_code >= 400:
                raise CapitalError(f"{method} {path} → {r.status_code}: {r.text[:300]}")
            time.sleep(0.15)  # μένουμε κάτω από τα 10 αιτήματα/δευτ.
            return r.json() if r.text else {}
        raise CapitalError(f"{method} {path}: πολλά αιτήματα, δοκίμασε αργότερα")

    # ── λογαριασμός ──
    def account(self) -> dict:
        accs = self._req("GET", "/accounts").get("accounts", [])
        pref = next((a for a in accs if a.get("preferred")), accs[0] if accs else None)
        if not pref:
            raise CapitalError("Δεν βρέθηκε λογαριασμός")
        b = pref.get("balance", {})
        return {"currency": pref.get("currency", "USD"),
                "balance": float(b.get("balance", 0)), "pnl": float(b.get("profitLoss", 0)),
                "deposit": float(b.get("deposit", 0)), "available": float(b.get("available", 0)),
                "equity": float(b.get("balance", 0)) + float(b.get("profitLoss", 0))}

    # ── αγορές ──
    def market(self, epic: str) -> dict:
        m = self._req("GET", f"/markets/{epic}")
        inst, rules, snap = m.get("instrument", {}), m.get("dealingRules", {}), m.get("snapshot", {})

        def rule(name, default=0.0):
            v = rules.get(name) or {}
            return float(v.get("value", default) or default), v.get("unit", "")

        min_size, _ = rule("minDealSize", 1.0)
        step, _ = rule("minSizeIncrement", 0.0)
        min_stop, min_stop_unit = rule("minStopOrProfitDistance", 0.0)
        return {"epic": epic, "currency": inst.get("currency", "USD"), "name": inst.get("name", epic),
                "min_size": min_size, "size_step": step or min_size,
                "min_stop": min_stop, "min_stop_unit": min_stop_unit,
                "status": snap.get("marketStatus", ""), "bid": snap.get("bid"), "offer": snap.get("offer")}

    def search(self, term: str) -> list[dict]:
        return self._req("GET", "/markets", params={"searchTerm": term}).get("markets", [])

    # ── θέσεις ──
    def positions(self) -> list[dict]:
        out = []
        for item in self._req("GET", "/positions").get("positions", []):
            pos, mkt = item.get("position", {}), item.get("market", {})
            out.append({"deal_id": pos.get("dealId"), "epic": mkt.get("epic"),
                        "direction": pos.get("direction"), "size": float(pos.get("size", 0)),
                        "level": float(pos.get("level", 0)), "created": pos.get("createdDateUTC") or pos.get("createdDate"),
                        "stop": pos.get("stopLevel"), "tp": pos.get("profitLevel"),
                        "upl": float(pos.get("upl", 0) or 0), "bid": mkt.get("bid")})
        return out

    def open(self, epic: str, size: float, stop: float, tp: float, direction: str = "BUY") -> dict:
        ref = self._req("POST", "/positions", json={
            "epic": epic, "direction": direction, "size": size, "guaranteedStop": False,
            "stopLevel": round(stop, 2), "profitLevel": round(tp, 2)}).get("dealReference")
        return self.confirm(ref)

    def close(self, deal_id: str) -> dict:
        ref = self._req("DELETE", f"/positions/{deal_id}").get("dealReference")
        return self.confirm(ref) if ref else {}

    def confirm(self, ref: str | None) -> dict:
        if not ref:
            raise CapitalError("Δεν επέστρεψε dealReference")
        for _ in range(5):
            c = self._req("GET", f"/confirms/{ref}")
            if c.get("dealStatus") in ("ACCEPTED", "REJECTED"):
                if c["dealStatus"] == "REJECTED":
                    raise CapitalError(f"Απορρίφθηκε: {c.get('reason') or c}")
                return c
            time.sleep(1)
        return c
