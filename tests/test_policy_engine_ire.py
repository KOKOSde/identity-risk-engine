from __future__ import annotations

from identity_risk_engine.policy_engine import PolicyEngine


def test_default_policy_mapping() -> None:
    engine = PolicyEngine()
    assert engine.decide(0.10)["action"] == "allow"
    assert engine.decide(0.95)["action"] == "block"


def test_auth_method_override_applies() -> None:
    engine = PolicyEngine(config="configs/default_policy.yaml")
    decision = engine.decide(0.50, auth_method="passkey")
    assert decision["action"] == "step_up_with_passkey"


def test_tenant_override_applies() -> None:
    engine = PolicyEngine(config="configs/default_policy.yaml")
    decision = engine.decide(0.15, tenant_id="tenant_1", auth_method="password")
    assert decision["action"] == "allow_with_monitoring"


def test_dry_run_override() -> None:
    engine = PolicyEngine(config={"dry_run": False})
    assert engine.decide(0.2)["dry_run"] is False
    assert engine.decide(0.2, dry_run=True)["dry_run"] is True
