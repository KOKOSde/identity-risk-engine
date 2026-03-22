from __future__ import annotations

import numpy as np

from identity_risk_engine import CompositeRiskScorer
from identity_risk_engine.synthetic_data_generator import (
    generate_synthetic_login_data,
    train_test_time_split,
)


def test_composite_model_reaches_auc_target_on_synthetic_data() -> None:
    df = generate_synthetic_login_data(
        num_users=100,
        num_sessions=2400,
        attack_ratio=0.22,
        seed=21,
    )
    train_df, test_df = train_test_time_split(df, test_ratio=0.25)

    model = CompositeRiskScorer()
    model.fit(train_df, target_col="label")

    metrics = model.evaluate(test_df, target_col="label")
    auc_score = metrics["global"]["auc"]
    assert auc_score > 0.90


def test_predict_proba_shape_and_operating_points() -> None:
    df = generate_synthetic_login_data(
        num_users=60,
        num_sessions=1200,
        attack_ratio=0.2,
        seed=31,
    )
    train_df, test_df = train_test_time_split(df, test_ratio=0.3)

    model = CompositeRiskScorer()
    model.fit(train_df, target_col="label")

    probs = model.predict_proba(test_df)
    assert probs.shape == (len(test_df), 2)
    assert np.all((probs[:, 1] >= 0.0) & (probs[:, 1] <= 1.0))

    block = model.get_operating_point("block_mode")
    friction = model.get_operating_point("friction_mode")
    assert 0.0 <= block.threshold <= 1.0
    assert 0.0 <= friction.threshold <= 1.0
