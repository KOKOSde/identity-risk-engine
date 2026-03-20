from __future__ import annotations

import pandas as pd

from identity_risk_engine.geo_velocity import compute_geo_velocity_features


def test_same_location_speed_zero() -> None:
    df = pd.DataFrame(
        [
            {"user_id": "u1", "timestamp": "2026-01-01T00:00:00Z", "lat": 37.77, "lon": -122.42},
            {"user_id": "u1", "timestamp": "2026-01-01T01:00:00Z", "lat": 37.77, "lon": -122.42},
        ]
    )
    out = compute_geo_velocity_features(df)
    second = out.iloc[1]
    assert second["distance_km"] == 0
    assert second["speed_kmh"] == 0
    assert second["geo_velocity_score"] == 0
    assert not bool(second["impossible_travel"])


def test_null_coordinates_do_not_crash() -> None:
    df = pd.DataFrame(
        [
            {"user_id": "u1", "timestamp": "2026-01-01T00:00:00Z", "lat": 37.77, "lon": -122.42},
            {"user_id": "u1", "timestamp": "2026-01-01T01:00:00Z", "lat": None, "lon": -122.42},
        ]
    )
    out = compute_geo_velocity_features(df)
    second = out.iloc[1]
    assert second["speed_kmh"] == 0
    assert second["geo_velocity_score"] == 0
    assert not bool(second["impossible_travel"])


def test_impossible_travel_detected() -> None:
    df = pd.DataFrame(
        [
            {"user_id": "u1", "timestamp": "2026-01-01T00:00:00Z", "lat": 40.7128, "lon": -74.0060},  # NYC
            {"user_id": "u1", "timestamp": "2026-01-01T01:00:00Z", "lat": 51.5074, "lon": -0.1278},  # London
        ]
    )
    out = compute_geo_velocity_features(df)
    second = out.iloc[1]
    assert second["speed_kmh"] > 900
    assert bool(second["impossible_travel"])
    assert 0.99 <= second["geo_velocity_score"] <= 1.0
