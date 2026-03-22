"""Auth event schema for identity-risk-engine v0.2.0."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AuthEventType(str, Enum):
    login_attempt = "login_attempt"
    login_success = "login_success"
    login_failure = "login_failure"
    logout = "logout"
    passkey_registered = "passkey_registered"
    passkey_auth_success = "passkey_auth_success"
    passkey_auth_failure = "passkey_auth_failure"
    password_reset_requested = "password_reset_requested"
    password_reset_success = "password_reset_success"
    password_reset_failure = "password_reset_failure"
    recovery_requested = "recovery_requested"
    recovery_success = "recovery_success"
    recovery_failure = "recovery_failure"
    mfa_challenge_sent = "mfa_challenge_sent"
    mfa_challenge_passed = "mfa_challenge_passed"
    mfa_challenge_failed = "mfa_challenge_failed"
    device_enrolled = "device_enrolled"
    device_removed = "device_removed"
    session_created = "session_created"
    session_revoked = "session_revoked"
    email_changed = "email_changed"
    phone_changed = "phone_changed"


class AuthEvent(BaseModel):
    """Canonical auth event model consumed by the risk engine."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True, use_enum_values=True)

    event_id: str
    event_type: AuthEventType
    user_id: str
    session_id: str | None = None
    timestamp: str
    ip: str | None = None
    country: str | None = None
    city_coarse: str | None = None
    lat_coarse: float | None = None
    lon_coarse: float | None = None
    user_agent: str | None = None
    device_hash: str | None = None
    device_type: str | None = None
    browser: str | None = None
    os: str | None = None
    auth_method: str | None = None
    success: bool = False
    failure_reason: str | None = None
    challenge_type: str | None = None
    recovery_channel: str | None = None
    email_domain: str | None = None
    tenant_id: str = Field(default="default")
    metadata: dict[str, Any] = Field(default_factory=dict)


def event_to_row(event: AuthEvent | dict[str, Any]) -> dict[str, Any]:
    """Normalize event payload to a plain dict suitable for DataFrames."""

    if isinstance(event, AuthEvent):
        return event.model_dump(mode="json")
    return AuthEvent(**event).model_dump(mode="json")
