"""Device-oriented auth risk signals."""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd


def _signal(name: str, fired: bool, score: float, evidence: str) -> dict[str, Any]:
    return {
        "signal_name": name,
        "fired": bool(fired),
        "score": float(max(0.0, min(1.0, score))),
        "evidence": evidence,
    }


def evaluate_device_signals(
    event: dict[str, Any],
    user_history: Optional[pd.DataFrame] = None,
    global_history: Optional[pd.DataFrame] = None,
) -> list[dict[str, Any]]:
    user_history = user_history if user_history is not None else pd.DataFrame()
    global_history = global_history if global_history is not None else pd.DataFrame()

    now = pd.to_datetime(event.get("timestamp"), utc=True, errors="coerce")
    device_hash = str(event.get("device_hash") or "")
    user_agent = (event.get("user_agent") or "").lower()

    user_devices = user_history[user_history.get("device_hash", pd.Series(dtype=object)) == device_hash]
    seen_before = not user_devices.empty

    new_device = not seen_before and bool(device_hash)

    dormant_days = 0.0
    if seen_before and pd.notna(now):
        last_seen = pd.to_datetime(user_devices["timestamp"], utc=True, errors="coerce").max()
        dormant_days = float((now - last_seen).total_seconds() / 86400.0) if pd.notna(last_seen) else 0.0
    device_dormant = dormant_days > 90

    multi_account_count = 0
    if not global_history.empty and device_hash:
        multi_account_count = int(
            global_history[global_history.get("device_hash", pd.Series(dtype=object)) == device_hash]["user_id"]
            .astype(str)
            .nunique()
        )
    multi_account_device = multi_account_count >= 3

    device_velocity_count = 0
    if not user_history.empty and pd.notna(now):
        history = user_history.copy()
        history["timestamp"] = pd.to_datetime(history["timestamp"], utc=True, errors="coerce")
        recent = history[history["timestamp"] >= now - pd.Timedelta(hours=24)]
        device_velocity_count = int(recent.get("device_hash", pd.Series(dtype=object)).astype(str).nunique())
    device_velocity = device_velocity_count >= 4

    session_churn_count = 0
    if not user_history.empty and pd.notna(now):
        history = user_history.copy()
        history["timestamp"] = pd.to_datetime(history["timestamp"], utc=True, errors="coerce")
        recent = history[history["timestamp"] >= now - pd.Timedelta(hours=1)]
        revokes = recent[recent.get("event_type", "").astype(str) == "session_revoked"]
        creates = recent[recent.get("event_type", "").astype(str) == "session_created"]
        session_churn_count = int(len(revokes) + len(creates))
    session_churn = session_churn_count >= 6

    emulator_patterns = ["headless", "emulator", "selenium", "playwright", "python-requests"]
    emulator_heuristic = any(pattern in user_agent for pattern in emulator_patterns)

    return [
        _signal("new_device", new_device, 0.65 if new_device else 0.0, f"device_hash={device_hash} seen_before={seen_before}"),
        _signal(
            "device_dormant",
            device_dormant,
            0.55 if device_dormant else 0.0,
            f"dormant_days={dormant_days:.1f}",
        ),
        _signal(
            "multi_account_device",
            multi_account_device,
            min(1.0, 0.25 * max(0, multi_account_count - 1)),
            f"accounts_on_device={multi_account_count}",
        ),
        _signal(
            "device_velocity",
            device_velocity,
            min(1.0, 0.2 * max(0, device_velocity_count - 1)),
            f"distinct_devices_24h={device_velocity_count}",
        ),
        _signal(
            "session_churn",
            session_churn,
            min(1.0, 0.15 * session_churn_count),
            f"session_events_1h={session_churn_count}",
        ),
        _signal(
            "emulator_heuristic",
            emulator_heuristic,
            0.7 if emulator_heuristic else 0.0,
            f"ua={event.get('user_agent', '')}",
        ),
    ]
