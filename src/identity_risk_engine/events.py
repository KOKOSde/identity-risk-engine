"""Auth event schema for identity-risk-engine v0.2.0."""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional, Union

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
    session_id: Optional[str] = None
    timestamp: str
    ip: Optional[str] = None
    country: Optional[str] = None
    city_coarse: Optional[str] = None
    lat_coarse: Optional[float] = None
    lon_coarse: Optional[float] = None
    user_agent: Optional[str] = None
    device_hash: Optional[str] = None
    device_type: Optional[str] = None
    browser: Optional[str] = None
    os: Optional[str] = None
    auth_method: Optional[str] = None
    success: bool = False
    failure_reason: Optional[str] = None
    challenge_type: Optional[str] = None
    recovery_channel: Optional[str] = None
    email_domain: Optional[str] = None
    tenant_id: str = Field(default="default")
    metadata: dict[str, Any] = Field(default_factory=dict)


def event_to_row(event: Union[AuthEvent, dict[str, Any]]) -> dict[str, Any]:
    """Normalize event payload to a plain dict suitable for DataFrames."""

    if isinstance(event, AuthEvent):
        return event.model_dump(mode="json")
    return AuthEvent(**event).model_dump(mode="json")
