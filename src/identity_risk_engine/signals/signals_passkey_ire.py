"""Passkey/authenticator related signals."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from ._common import event_ts, filter_user, in_window, signal, to_frame

PASSKEY_SUCCESS = {"passkey_auth_success"}
PASSKEY_REGISTER = {"passkey_registered", "device_enrolled"}
PASSKEY_REMOVE = {"device_removed"}


def evaluate_passkey_signals(
    event: Mapping[str, Any],
    user_history: pd.DataFrame | list[Mapping[str, Any]] | None = None,
    global_history: pd.DataFrame | list[Mapping[str, Any]] | None = None,
) -> list[dict[str, object]]:
    now = event_ts(event)
    user_df = to_frame(user_history)
    user_events = filter_user(user_df, str(event.get("user_id") or ""))
    if "timestamp" in user_events.columns:
        user_events = user_events.sort_values("timestamp")
    else:
        user_events = pd.DataFrame(columns=["timestamp"])

    event_type = str(event.get("event_type") or "")
    device_hash = str(event.get("device_hash") or "")
    metadata = event.get("metadata") or {}

    results: list[dict[str, object]] = []

    # new_passkey_unfamiliar_device
    familiar_device = False
    if device_hash and not user_events.empty and "device_hash" in user_events.columns:
        familiar_device = (user_events["device_hash"].astype(str) == device_hash).any()
    unfamiliar_passkey = event_type in PASSKEY_SUCCESS and not familiar_device
    results.append(
        signal(
            "new_passkey_unfamiliar_device",
            unfamiliar_passkey,
            0.75 if unfamiliar_passkey else 0.0,
            "Passkey success from previously unseen device"
            if unfamiliar_passkey
            else "Passkey on familiar device or non-passkey event",
        )
    )

    # passkey_registration_burst
    recent_30 = in_window(user_events, now, 30)
    registration_count = 0
    if "event_type" in recent_30.columns:
        registration_count = int(recent_30[recent_30["event_type"].astype(str).isin(PASSKEY_REGISTER)].shape[0])
    registration_burst = registration_count >= 3 and event_type in PASSKEY_REGISTER
    results.append(
        signal(
            "passkey_registration_burst",
            registration_burst,
            min(registration_count / 6.0, 1.0) if registration_burst else 0.0,
            f"{registration_count} passkeys/authenticators added in 30 minutes"
            if registration_burst
            else "No passkey registration burst",
        )
    )

    # passkey_after_password_failure
    recent_failures = 0
    if "event_type" in recent_30.columns:
        recent_failures = int(
            recent_30[recent_30["event_type"].astype(str).isin(["login_failure", "password_reset_failure"])].shape[0]
        )
    passkey_after_password_failure = event_type in PASSKEY_SUCCESS and recent_failures >= 2
    results.append(
        signal(
            "passkey_after_password_failure",
            passkey_after_password_failure,
            0.7 if passkey_after_password_failure else 0.0,
            "Passkey success immediately follows password failures"
            if passkey_after_password_failure
            else "No passkey-after-password-failure pattern",
        )
    )

    # authenticator_churn
    recent_day = in_window(user_events, now, 24 * 60)
    churn_count = 0
    if "event_type" in recent_day.columns:
        churn_count = int(recent_day[recent_day["event_type"].astype(str).isin(PASSKEY_REGISTER | PASSKEY_REMOVE)].shape[0])
    authenticator_churn = churn_count >= 6
    results.append(
        signal(
            "authenticator_churn",
            authenticator_churn,
            min(churn_count / 10.0, 1.0) if authenticator_churn else 0.0,
            f"Authenticator add/remove events in 24h={churn_count}" if authenticator_churn else "No authenticator churn",
        )
    )

    # credential_novelty
    aaguid = str(metadata.get("authenticator_aaguid") or "")
    known_aaguids: set[str] = set()
    if "metadata" in user_events.columns:
        for meta in user_events["metadata"].dropna().tolist():
            if isinstance(meta, dict) and meta.get("authenticator_aaguid"):
                known_aaguids.add(str(meta["authenticator_aaguid"]))
    credential_novelty = bool(aaguid) and aaguid not in known_aaguids
    results.append(
        signal(
            "credential_novelty",
            credential_novelty,
            0.55 if credential_novelty else 0.0,
            f"AAGUID {aaguid} not seen for user" if credential_novelty else "Authenticator metadata is familiar",
        )
    )

    return results
