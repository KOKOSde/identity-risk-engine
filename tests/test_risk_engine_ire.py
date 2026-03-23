from __future__ import annotations

from identity_risk_engine.policy_engine import PolicyEngine
from identity_risk_engine.risk_engine_ire import score_dataframe
from identity_risk_engine.simulator_ire import generate_synthetic_auth_events


def test_attack_events_get_nonzero_risk_scores() -> None:
    events = generate_synthetic_auth_events(
        num_users=80,
        num_sessions=2200,
        attack_ratio=0.25,
        seed=42,
    )
    scored = score_dataframe(
        events,
        policy_engine=PolicyEngine(),
        history_window=150,
        include_explanations=False,
    )

    attacks = scored[scored["label"].astype(int) == 1]
    assert len(attacks) > 0

    low_risk_ratio = float((attacks["risk_score"] < 0.1).mean())
    assert low_risk_ratio <= 0.10
