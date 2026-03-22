"""Recovery-flow abuse signals for identity-risk-engine."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from ._common import event_ts, filter_user, haversine_km, in_window, signal, to_frame

RECOVERY_EVENTS = {"recovery_requested", "recovery_success", "recovery_failure"}


def evaluate_recovery_signals(
    event: Mapping[str, Any],
    user_history: pd.DataFrame | list[Mapping[str, Any]] | None = None,
    global_history: pd.DataFrame | list[Mapping[str, Any]] | None = None,
) -> list[dict[str, object]]:
    now = event_ts(event)
    user_df = to_frame(user_history)
    global_df = to_frame(global_history)

    user_id = str(event.get("user_id") or "")
    event_type = str(event.get("event_type") or "")
    metadata = event.get("metadata") or {}
    ip = str(event.get("ip") or "")
    device_hash = str(event.get("device_hash") or "")

    user_events = filter_user(user_df, user_id)
    if "timestamp" in user_events.columns:
        user_events = user_events.sort_values("timestamp")
    else:
        user_events = pd.DataFrame(columns=["timestamp"])
    recent_user_30 = in_window(user_events, now, 30)

    results: list[dict[str, object]] = []

    # recovery_unfamiliar_env
    unfamiliar_device = bool(metadata.get("new_device"))
    unfamiliar_asn = bool(metadata.get("new_asn"))
    recovery_unfamiliar_env = event_type in RECOVERY_EVENTS and unfamiliar_device and unfamiliar_asn
    results.append(
        signal(
            "recovery_unfamiliar_env",
            recovery_unfamiliar_env,
            0.85 if recovery_unfamiliar_env else 0.0,
            "Recovery from new device and new ASN" if recovery_unfamiliar_env else "No unfamiliar recovery environment",
        )
    )

    # recovery_after_lockout
    lockout_failures = 0
    if "event_type" in recent_user_30.columns:
        lockout_failures = int(
            recent_user_30[
                recent_user_30["event_type"].astype(str).isin(["login_failure", "mfa_challenge_failed", "recovery_failure"])
            ].shape[0]
        )
    recovery_after_lockout = event_type == "recovery_requested" and lockout_failures >= 5
    results.append(
        signal(
            "recovery_after_lockout",
            recovery_after_lockout,
            min(lockout_failures / 10.0, 1.0) if recovery_after_lockout else 0.0,
            f"Recovery requested after {lockout_failures} failures" if recovery_after_lockout else "No lockout-then-recovery pattern",
        )
    )

    # recovery_plus_credential_change
    recent_recovery_success = False
    if "event_type" in recent_user_30.columns:
        recent_recovery_success = (
            recent_user_30["event_type"].astype(str).eq("recovery_success").any()
        )
    recovery_plus_change = event_type in {"email_changed", "phone_changed"} and recent_recovery_success
    results.append(
        signal(
            "recovery_plus_credential_change",
            recovery_plus_change,
            0.75 if recovery_plus_change else 0.0,
            "Credential changed shortly after recovery success"
            if recovery_plus_change
            else "No risky post-recovery credential change",
        )
    )

    # recovery_fanout
    fanout_accounts = 0
    if not global_df.empty:
        recent_global = in_window(global_df, now, 30)
        mask = False
        if ip and "ip" in recent_global.columns:
            mask = recent_global["ip"].astype(str) == ip
        if device_hash and "device_hash" in recent_global.columns:
            device_mask = recent_global["device_hash"].astype(str) == device_hash
            mask = device_mask if isinstance(mask, bool) else (mask | device_mask)
        if not isinstance(mask, bool) and "event_type" in recent_global.columns:
            recovery_rows = recent_global[mask & recent_global["event_type"].astype(str).isin(RECOVERY_EVENTS)]
            fanout_accounts = int(recovery_rows["user_id"].astype(str).nunique()) if "user_id" in recovery_rows.columns else 0
    recovery_fanout = fanout_accounts >= 4
    results.append(
        signal(
            "recovery_fanout",
            recovery_fanout,
            min(fanout_accounts / 8.0, 1.0) if recovery_fanout else 0.0,
            f"Recovery requests across {fanout_accounts} accounts" if recovery_fanout else "No recovery fanout",
        )
    )

    # recovery_impossible_travel
    recovery_impossible_travel = False
    evidence = "No immediate impossible-travel context"
    if event_type in RECOVERY_EVENTS and len(user_events) >= 1:
        prev = user_events.iloc[-1]
        prev_ts = prev.get("timestamp")
        prev_lat = _to_float(prev.get("lat_coarse"))
        prev_lon = _to_float(prev.get("lon_coarse"))
        curr_lat = _to_float(event.get("lat_coarse"))
        curr_lon = _to_float(event.get("lon_coarse"))
        if pd.notna(prev_ts) and prev_lat is not None and prev_lon is not None and curr_lat is not None and curr_lon is not None:
            delta_h = max((now - prev_ts).total_seconds() / 3600.0, 1e-6)
            speed = haversine_km(prev_lat, prev_lon, curr_lat, curr_lon) / delta_h
            recovery_impossible_travel = speed > 900
            evidence = f"Recovery follows travel speed {speed:.1f} km/h"
    results.append(
        signal(
            "recovery_impossible_travel",
            recovery_impossible_travel,
            1.0 if recovery_impossible_travel else 0.0,
            evidence,
        )
    )

    return results


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
