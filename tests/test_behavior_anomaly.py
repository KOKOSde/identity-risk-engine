from __future__ import annotations

import pandas as pd

from identity_risk_engine.behavior_anomaly import BehaviorAnomalyScorer


def _behavior_df(n: int = 8) -> pd.DataFrame:
    base = pd.Timestamp("2026-01-01T08:00:00Z")
    rows = []
    for i in range(n):
        rows.append(
            {
                "user_id": "u1",
                "timestamp": base + pd.Timedelta(hours=24 * i),
                "session_duration": 320 + (i % 3) * 10,
                "actions_count": 4 + (i % 2),
                "action_entropy": 1.1 + (i % 2) * 0.1,
            }
        )
    return pd.DataFrame(rows)


def test_new_user_no_history_returns_uncertainty_score() -> None:
    model = BehaviorAnomalyScorer(min_history=5)
    model.fit(_behavior_df())

    score = model.score_session(
        {
            "user_id": "new_user",
            "timestamp": "2026-02-01T01:00:00Z",
            "session_duration": 200,
            "actions_count": 2,
            "action_entropy": 0.5,
        }
    )
    assert 0.5 <= score <= 0.7


def test_insufficient_baseline_still_scores() -> None:
    model = BehaviorAnomalyScorer(min_history=5)
    model.fit(_behavior_df(n=3))

    score = model.score_session(
        {
            "user_id": "u1",
            "timestamp": "2026-01-05T08:00:00Z",
            "session_duration": 350,
            "actions_count": 4,
            "action_entropy": 1.0,
        }
    )
    assert 0.0 <= score <= 1.0


def test_clear_behavior_outlier_scores_higher_than_normal() -> None:
    model = BehaviorAnomalyScorer(min_history=5)
    model.fit(_behavior_df(n=10))

    normal_score = model.score_session(
        {
            "user_id": "u1",
            "timestamp": "2026-01-15T08:00:00Z",
            "session_duration": 330,
            "actions_count": 4,
            "action_entropy": 1.1,
        }
    )
    outlier_score = model.score_session(
        {
            "user_id": "u1",
            "timestamp": "2026-01-15T03:00:00Z",
            "session_duration": 30,
            "actions_count": 1,
            "action_entropy": 0.05,
            "inter_login_hours": 0.2,
        }
    )

    assert outlier_score > normal_score
