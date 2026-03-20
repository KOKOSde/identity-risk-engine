from __future__ import annotations

from identity_risk_engine.synthetic_data_generator import (
    ATTACK_TYPES,
    generate_synthetic_login_data,
)


def test_generator_contains_all_attack_types() -> None:
    df = generate_synthetic_login_data(
        num_users=120,
        num_sessions=5000,
        attack_ratio=0.25,
        seed=13,
    )

    present = set(df.loc[df["label"] == 1, "attack_type"].unique())
    assert set(ATTACK_TYPES).issubset(present)


def test_generator_has_required_features() -> None:
    df = generate_synthetic_login_data(num_users=80, num_sessions=2000, attack_ratio=0.2, seed=9)
    expected_cols = {
        "geo_velocity_score",
        "device_novelty_score",
        "behavior_anomaly_score",
        "ip_reputation",
        "session_duration",
        "country_mismatch",
        "failed_attempts",
        "account_age_days",
        "attack_type",
        "label",
    }
    assert expected_cols.issubset(set(df.columns))


def test_new_account_fraud_is_young_account() -> None:
    df = generate_synthetic_login_data(num_users=60, num_sessions=1800, attack_ratio=0.3, seed=99)
    subset = df[df["attack_type"] == "new_account_fraud"]
    assert len(subset) > 0
    assert (subset["account_age_days"] < 1.0).mean() >= 0.8
