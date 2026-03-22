from __future__ import annotations

import pandas as pd

from identity_risk_engine.explainer_ire import explain_scored_event


def test_explainer_output_shape_and_content() -> None:
    event = {
        "event_id": "evt_123",
        "event_type": "login_success",
        "user_id": "u1",
        "timestamp": "2026-03-22T02:00:00Z",
    }
    signals = [
        {"signal_name": "new_device", "fired": True, "score": 0.8, "evidence": "device unseen"},
        {"signal_name": "new_country", "fired": True, "score": 0.7, "evidence": "country mismatch"},
        {"signal_name": "failure_burst", "fired": True, "score": 0.9, "evidence": "6 failures"},
        {"signal_name": "ip_velocity", "fired": False, "score": 0.2, "evidence": "low"},
    ]
    user_history = pd.DataFrame(
        [
            {"user_id": "u1", "timestamp": "2026-03-21T10:00:00Z", "risk_score": 0.10},
            {"user_id": "u1", "timestamp": "2026-03-21T11:00:00Z", "risk_score": 0.20},
        ]
    )

    out = explain_scored_event(event=event, signal_results=signals, risk_score=0.88, user_history=user_history)

    assert set(out.keys()) == {
        "top_reasons",
        "feature_values",
        "human_summary",
        "machine_json",
        "baseline_comparison",
    }
    assert len(out["top_reasons"]) <= 3
    assert out["machine_json"]["event_id"] == "evt_123"
    assert isinstance(out["human_summary"], str)
    assert "risk" in out["human_summary"].lower()
