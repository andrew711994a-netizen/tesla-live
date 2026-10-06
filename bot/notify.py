"""Ειδοποιήσεις στο κινητό μέσω ntfy.sh (δωρεάν εφαρμογή «ntfy»)."""

from __future__ import annotations

import os

import requests


def notify(title: str, body: str) -> None:
    topic = os.environ.get("NTFY_TOPIC", "").strip()
    if not topic:
        return
    try:
        requests.post("https://ntfy.sh/", json={"topic": topic, "title": title, "message": body,
                                                "tags": ["chart_with_upwards_trend"]}, timeout=15)
    except Exception as e:  # η ειδοποίηση δεν πρέπει ποτέ να σταματήσει το bot
        print(f"Η ειδοποίηση απέτυχε: {e}")
