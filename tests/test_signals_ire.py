from __future__ import annotations

import pandas as pd

from identity_risk_engine.signals import (
    evaluate_behavior_signals,
    evaluate_device_signals,
    evaluate_geo_signals,
    evaluate_passkey_signals,
    evaluate_recovery_signals,
)
from identity_risk_engine.simulator_ire import ATTACK_TYPES_IRE, generate_synthetic_auth_events

REQUIRED_KEYS = {"signal_name", "fired", "score", "evidence"}


def _assert_signal_schema(items: list[dict[str, object]]) -> None:
    assert items
    for item in items:
        assert REQUIRED_KEYS.issubset(item.keys())


def test_device_signals_fire_for_new_device() -> None:
    user_history = pd.DataFrame(
        [
            {
                "user_id": "u1",
                "timestamp": "2026-03-22T00:00:00Z",
                "device_hash": "known_device",
                "event_type": "login_success",
                "user_agent": "Mozilla/5.0",
            }
        ]
    )
    event = {
        "user_id": "u1",
        "timestamp": "2026-03-22T01:00:00Z",
        "device_hash": "new_device",
        "user_agent": "Mozilla/5.0",
    }
    signals = evaluate_device_signals(event=event, user_history=user_history, global_history=user_history)
    _assert_signal_schema(signals)
    fired = {s["signal_name"] for s in signals if s["fired"]}
    assert "new_device" in fired


def test_geo_signals_fire_for_impossible_travel() -> None:
    user_history = pd.DataFrame(
        [
            {
                "user_id": "u1",
                "timestamp": "2026-03-22T00:00:00Z",
                "lat_coarse": 40.7128,
                "lon_coarse": -74.0060,
                "country": "US",
                "ip": "8.8.8.8",
                "metadata": {"ip_asn": "RESIDENTIAL-AS15169"},
                "success": True,
            }
        ]
    )
    event = {
        "user_id": "u1",
        "timestamp": "2026-03-22T01:00:00Z",
        "lat_coarse": 51.5074,
        "lon_coarse": -0.1278,
        "country": "GB",
        "ip": "35.1.2.3",
        "metadata": {"ip_asn": "DATACENTER-AS99999"},
        "success": True,
    }
    signals = evaluate_geo_signals(event=event, user_history=user_history, global_history=user_history)
    _assert_signal_schema(signals)
    fired = {s["signal_name"] for s in signals if s["fired"]}
    assert "impossible_travel" in fired


def test_behavior_signals_fire_for_failure_burst() -> None:
    rows = []
    base = pd.Timestamp("2026-03-22T00:00:00Z")
    for i in range(6):
        rows.append(
            {
                "user_id": "u1",
                "timestamp": (base + pd.Timedelta(minutes=2 * i)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "event_type": "login_failure",
                "success": False,
                "ip": "9.9.9.9",
                "device_hash": "d1",
            }
        )
    history = pd.DataFrame(rows)
    event = {
        "user_id": "u1",
        "timestamp": "2026-03-22T00:14:00Z",
        "event_type": "login_failure",
        "success": False,
        "auth_method": "password",
        "ip": "9.9.9.9",
        "device_hash": "d1",
        "metadata": {},
    }
    signals = evaluate_behavior_signals(event=event, user_history=history, global_history=history)
    _assert_signal_schema(signals)
    fired = {s["signal_name"] for s in signals if s["fired"]}
    assert "failure_burst" in fired


def test_passkey_signals_fire_for_unfamiliar_device() -> None:
    history = pd.DataFrame(
        [
            {
                "user_id": "u1",
                "timestamp": "2026-03-22T00:00:00Z",
                "event_type": "passkey_auth_success",
                "device_hash": "known_pk_device",
                "metadata": {"authenticator_aaguid": "aaguid-1111"},
            }
        ]
    )
    event = {
        "user_id": "u1",
        "timestamp": "2026-03-22T00:10:00Z",
        "event_type": "passkey_auth_success",
        "device_hash": "new_pk_device",
        "metadata": {"authenticator_aaguid": "aaguid-2222"},
    }
    signals = evaluate_passkey_signals(event=event, user_history=history, global_history=history)
    _assert_signal_schema(signals)
    fired = {s["signal_name"] for s in signals if s["fired"]}
    assert "new_passkey_unfamiliar_device" in fired


def test_recovery_signals_fire_for_unfamiliar_env() -> None:
    history = pd.DataFrame(
        [
            {
                "user_id": "u1",
                "timestamp": "2026-03-22T00:00:00Z",
                "event_type": "login_failure",
                "lat_coarse": 37.7749,
                "lon_coarse": -122.4194,
                "ip": "8.8.8.8",
                "device_hash": "d1",
            }
        ]
    )
    event = {
        "user_id": "u1",
        "timestamp": "2026-03-22T00:10:00Z",
        "event_type": "recovery_requested",
        "ip": "35.1.2.3",
        "device_hash": "d2",
        "lat_coarse": 51.5074,
        "lon_coarse": -0.1278,
        "metadata": {"new_device": True, "new_asn": True},
    }
    signals = evaluate_recovery_signals(event=event, user_history=history, global_history=history)
    _assert_signal_schema(signals)
    fired = {s["signal_name"] for s in signals if s["fired"]}
    assert "recovery_unfamiliar_env" in fired


def test_simulator_ire_contains_all_attack_types() -> None:
    df = generate_synthetic_auth_events(num_users=120, num_sessions=4000, attack_ratio=0.25, seed=42)
    present = set(df.loc[df["label"] == 1, "attack_type"].unique())
    assert set(ATTACK_TYPES_IRE).issubset(present)
