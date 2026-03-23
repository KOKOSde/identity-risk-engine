"""Policy engine mapping risk scores to auth actions."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import yaml

DEFAULT_ACTIONS = [
    "allow",
    "allow_with_monitoring",
    "step_up_with_passkey",
    "step_up_with_totp",
    "step_up_with_email_code",
    "require_recovery_review",
    "manual_review",
    "block",
    "revoke_session",
]

DEFAULT_POLICY: dict[str, Any] = {
    "dry_run": False,
    "thresholds": [
        {"max_score": 0.15, "action": "allow"},
        {"max_score": 0.30, "action": "allow_with_monitoring"},
        {"max_score": 0.45, "action": "step_up_with_passkey"},
        {"max_score": 0.60, "action": "step_up_with_totp"},
        {"max_score": 0.72, "action": "step_up_with_email_code"},
        {"max_score": 0.82, "action": "require_recovery_review"},
        {"max_score": 0.90, "action": "manual_review"},
        {"max_score": 0.97, "action": "block"},
        {"max_score": 1.00, "action": "revoke_session"},
    ],
    "auth_method_overrides": {},
    "tenant_overrides": {},
}


@dataclass
class PolicyDecision:
    action: str
    risk_score: float
    reasons: list[str]
    evidence: list[str]
    dry_run: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "risk_score": self.risk_score,
            "reasons": self.reasons,
            "evidence": self.evidence,
            "dry_run": self.dry_run,
        }


def load_policy_config(
    config: Optional[Union[str, Path, Mapping[str, Any]]] = None
) -> dict[str, Any]:
    """Load policy config from YAML path or dict, merged over defaults."""

    loaded: dict[str, Any]
    if config is None:
        loaded = {}
    elif isinstance(config, Mapping):
        loaded = dict(config)
    else:
        path = Path(config)
        with path.open("r", encoding="utf-8") as handle:
            loaded_yaml = yaml.safe_load(handle) or {}
        if not isinstance(loaded_yaml, dict):
            raise ValueError("Policy YAML root must be a mapping")
        loaded = loaded_yaml

    merged = deepcopy(DEFAULT_POLICY)
    merged.update({k: v for k, v in loaded.items() if k not in {"auth_method_overrides", "tenant_overrides"}})
    merged["auth_method_overrides"] = dict(DEFAULT_POLICY.get("auth_method_overrides", {}))
    merged["auth_method_overrides"].update(dict(loaded.get("auth_method_overrides") or {}))
    merged["tenant_overrides"] = dict(DEFAULT_POLICY.get("tenant_overrides", {}))
    for tenant_id, tenant_cfg in dict(loaded.get("tenant_overrides") or {}).items():
        tenant_id = str(tenant_id)
        base = dict(merged["tenant_overrides"].get(tenant_id) or {})
        base.update(dict(tenant_cfg or {}))
        merged["tenant_overrides"][tenant_id] = base
    return merged


class PolicyEngine:
    """Score-to-action policy evaluator with method/tenant overrides."""

    def __init__(self, config: Optional[Union[str, Path, Mapping[str, Any]]] = None) -> None:
        self.config = load_policy_config(config)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> PolicyEngine:
        return cls(config=path)

    def _thresholds_for(self, auth_method: str, tenant_id: str) -> list[dict[str, Any]]:
        thresholds = list(self.config.get("thresholds", []))

        tenant_cfg = (self.config.get("tenant_overrides") or {}).get(tenant_id, {})
        if tenant_cfg.get("thresholds"):
            thresholds = list(tenant_cfg["thresholds"])

        method_cfg = (self.config.get("auth_method_overrides") or {}).get(auth_method, {})
        if method_cfg.get("thresholds"):
            thresholds = list(method_cfg["thresholds"])

        tenant_method_cfg = (tenant_cfg.get("auth_method_overrides") or {}).get(auth_method, {})
        if tenant_method_cfg.get("thresholds"):
            thresholds = list(tenant_method_cfg["thresholds"])

        cleaned: list[dict[str, Any]] = []
        for item in thresholds:
            action = str(item.get("action", "allow"))
            if action not in DEFAULT_ACTIONS:
                action = "manual_review"
            cleaned.append({"max_score": float(item.get("max_score", 1.0)), "action": action})
        cleaned.sort(key=lambda x: x["max_score"])
        return cleaned

    def _effective_dry_run(self, auth_method: str, tenant_id: str, dry_run: Optional[bool]) -> bool:
        if dry_run is not None:
            return bool(dry_run)

        value = bool(self.config.get("dry_run", False))
        tenant_cfg = (self.config.get("tenant_overrides") or {}).get(tenant_id, {})
        if "dry_run" in tenant_cfg:
            value = bool(tenant_cfg.get("dry_run"))

        method_cfg = (self.config.get("auth_method_overrides") or {}).get(auth_method, {})
        if "dry_run" in method_cfg:
            value = bool(method_cfg.get("dry_run"))

        tenant_method_cfg = (tenant_cfg.get("auth_method_overrides") or {}).get(auth_method, {})
        if "dry_run" in tenant_method_cfg:
            value = bool(tenant_method_cfg.get("dry_run"))

        return value

    def decide(
        self,
        risk_score: float,
        *,
        reasons: Optional[list[str]] = None,
        evidence: Optional[list[str]] = None,
        auth_method: str = "password",
        tenant_id: str = "default",
        dry_run: Optional[bool] = None,
    ) -> dict[str, Any]:
        clamped_score = float(max(0.0, min(1.0, risk_score)))
        thresholds = self._thresholds_for(auth_method=auth_method, tenant_id=tenant_id)

        selected_action = "revoke_session"
        for row in thresholds:
            if clamped_score <= float(row["max_score"]):
                selected_action = str(row["action"])
                break

        effective_dry_run = self._effective_dry_run(auth_method=auth_method, tenant_id=tenant_id, dry_run=dry_run)

        decision = PolicyDecision(
            action=selected_action,
            risk_score=clamped_score,
            reasons=reasons or [],
            evidence=evidence or [],
            dry_run=effective_dry_run,
        )
        return decision.as_dict()
