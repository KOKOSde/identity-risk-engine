from __future__ import annotations

import pytest
from pydantic import ValidationError

from identity_risk_engine.events import AuthEvent, AuthEventType, event_to_row


def test_auth_event_validates_required_schema() -> None:
    event = AuthEvent(
        event_id="evt_1",
        event_type=AuthEventType.login_attempt,
        user_id="u1",
        timestamp="2026-03-22T00:00:00Z",
        ip="1.2.3.4",
    )
    assert event.user_id == "u1"
    assert event.event_type == AuthEventType.login_attempt.value


def test_event_to_row_serializes_enum_to_string() -> None:
    event = AuthEvent(
        event_id="evt_2",
        event_type=AuthEventType.passkey_auth_success,
        user_id="u2",
        timestamp="2026-03-22T01:00:00Z",
        success=True,
    )
    row = event_to_row(event)
    assert row["event_type"] == "passkey_auth_success"
    assert isinstance(row["metadata"], dict)


def test_invalid_event_type_raises_validation_error() -> None:
    with pytest.raises(ValidationError):
        AuthEvent(
            event_id="evt_bad",
            event_type="not_a_real_event",
            user_id="u1",
            timestamp="2026-03-22T00:00:00Z",
        )
