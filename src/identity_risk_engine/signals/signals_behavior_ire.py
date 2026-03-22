"""Behavioral auth-flow signals for identity-risk-engine."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from ._common import event_ts, filter_user, in_window, signal, to_bool, to_frame

FAILURE_EVENTS = {"login_failure", "passkey_auth_failure", "mfa_challenge_failed", "recovery_failure"}
SUCCESS_EVENTS = {"login_success", "passkey_auth_success", "mfa_challenge_passed", "recovery_success"}


def evaluate_behavior_signals(
    event: Mapping[str, Any],
    user_history: pd.DataFrame | list[Mapping[str, Any]] | None = None,
    global_history: pd.DataFrame | list[Mapping[str, Any]] | None = None,
) -> list[dict[str, object]]:
    now = event_ts(event)
    user_df = to_frame(user_history)
    global_df = to_frame(global_history)
    user_id = str(event.get("user_id") or "")

    user_events = filter_user(user_df, user_id)
    if "timestamp" in user_events.columns:
        user_events = user_events.sort_values("timestamp")
    else:
        user_events = pd.DataFrame(columns=["timestamp"])
    recent_user_15 = in_window(user_events, now, 15)
    recent_user_60 = in_window(user_events, now, 60)

    event_type = str(event.get("event_type") or "")
    success = to_bool(event.get("success"))
    auth_method = str(event.get("auth_method") or "")
    ip = str(event.get("ip") or "")
    device_hash = str(event.get("device_hash") or "")

    results: list[dict[str, object]] = []

    # failure_burst
    failure_burst_count = 0
    if "event_type" in recent_user_15.columns:
        failure_burst_count = int(recent_user_15[recent_user_15["event_type"].astype(str).isin(FAILURE_EVENTS)].shape[0])
    failure_burst = failure_burst_count >= 5
    results.append(
        signal(
            "failure_burst",
            failure_burst,
            min(failure_burst_count / 10.0, 1.0) if failure_burst else 0.0,
            f"{failure_burst_count} failures in 15 minutes" if failure_burst else "No failure burst",
        )
    )

    # success_after_burst
    success_after_burst = success and event_type in SUCCESS_EVENTS and failure_burst_count >= 3
    results.append(
        signal(
            "success_after_burst",
            success_after_burst,
            0.8 if success_after_burst else 0.0,
            "Success immediately after a burst of failures"
            if success_after_burst
            else "No suspicious success-after-failure pattern",
        )
    )

    # unusual_hour
    unusual_hour = False
    unusual_score = 0.0
    hour = now.hour
    if not user_events.empty and "timestamp" in user_events.columns:
        success_events = user_events
        if "success" in user_events.columns:
            success_events = user_events[user_events["success"].fillna(False).astype(bool)]
        if not success_events.empty:
            hist = success_events["timestamp"].dt.hour.value_counts(normalize=True)
            prob = float(hist.get(hour, 0.0))
            unusual_score = max(0.0, 1.0 - prob * 4.0)
            unusual_hour = prob < 0.05
    results.append(
        signal(
            "unusual_hour",
            unusual_hour,
            unusual_score if unusual_hour else 0.0,
            f"Hour {hour} is atypical for this account" if unusual_hour else "Hour aligns with account pattern",
        )
    )

    # auth_method_switch
    established_method = ""
    if "auth_method" in user_events.columns and not user_events.empty:
        mode = user_events["auth_method"].dropna().astype(str).mode()
        if not mode.empty:
            established_method = str(mode.iloc[0])
    method_switch = bool(auth_method and established_method and auth_method != established_method)
    results.append(
        signal(
            "auth_method_switch",
            method_switch,
            0.5 if method_switch else 0.0,
            f"Established method={established_method}, current={auth_method}" if method_switch else "Auth method consistent",
        )
    )

    # mfa_fatigue
    mfa_fatigue_count = 0
    if "event_type" in recent_user_60.columns:
        mfa_fatigue_count = int(
            recent_user_60[
                recent_user_60["event_type"].astype(str).isin(["mfa_challenge_sent", "mfa_challenge_failed"])
            ].shape[0]
        )
    mfa_fatigue = mfa_fatigue_count >= 6 and event_type in {"mfa_challenge_passed", "mfa_challenge_failed"}
    results.append(
        signal(
            "mfa_fatigue",
            mfa_fatigue,
            min(mfa_fatigue_count / 12.0, 1.0) if mfa_fatigue else 0.0,
            f"{mfa_fatigue_count} MFA prompts/failures in 60 minutes" if mfa_fatigue else "No MFA fatigue pattern",
        )
    )

    # recovery_abuse
    recovery_event = event_type.startswith("recovery_")
    unfamiliar = bool((event.get("metadata") or {}).get("new_device")) or bool((event.get("metadata") or {}).get("new_asn"))
    recent_failures = failure_burst_count >= 3
    recovery_abuse = recovery_event and unfamiliar and recent_failures
    results.append(
        signal(
            "recovery_abuse",
            recovery_abuse,
            0.85 if recovery_abuse else 0.0,
            "Recovery action from unfamiliar environment after failures" if recovery_abuse else "No recovery abuse pattern",
        )
    )

    # login_cadence_anomaly
    cadence_anomaly = False
    cadence_score = 0.0
    if len(user_events) >= 5 and "timestamp" in user_events.columns:
        deltas = user_events["timestamp"].sort_values().diff().dt.total_seconds().dropna() / 3600.0
        if not deltas.empty:
            median = float(deltas.median())
            mad = float((deltas - median).abs().median()) or 1.0
            current_gap = float((now - user_events["timestamp"].max()).total_seconds() / 3600.0)
            z = abs(current_gap - median) / mad
            cadence_anomaly = z >= 4.0
            cadence_score = min(z / 10.0, 1.0)
    results.append(
        signal(
            "login_cadence_anomaly",
            cadence_anomaly,
            cadence_score if cadence_anomaly else 0.0,
            "Current login cadence deviates from user baseline" if cadence_anomaly else "Cadence within expected range",
        )
    )

    # account_fanout
    account_fanout_count = 0
    if not global_df.empty:
        recent_global = in_window(global_df, now, 30)
        if not recent_global.empty:
            mask = False
            if ip and "ip" in recent_global.columns:
                mask = recent_global["ip"].astype(str) == ip
            if device_hash and "device_hash" in recent_global.columns:
                device_mask = recent_global["device_hash"].astype(str) == device_hash
                mask = device_mask if isinstance(mask, bool) else (mask | device_mask)
            if not isinstance(mask, bool):
                account_fanout_count = int(recent_global[mask]["user_id"].astype(str).nunique())
    account_fanout = account_fanout_count >= 6
    results.append(
        signal(
            "account_fanout",
            account_fanout,
            min(account_fanout_count / 12.0, 1.0) if account_fanout else 0.0,
            f"Shared IP/device touched {account_fanout_count} accounts in 30 minutes"
            if account_fanout
            else "No cross-account fanout",
        )
    )

    return results
