from __future__ import annotations

import pandas as pd

from identity_risk_engine.device_fingerprint import DeviceNoveltyScorer


def _base_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "user_id": "u1",
                "user_agent": "Chrome/122 desktop",
                "screen_resolution": "1920x1080",
                "timezone": "America/New_York",
                "language": "en-US",
                "ip_asn": "AS15169",
            },
            {
                "user_id": "u1",
                "user_agent": "Chrome/122 desktop",
                "screen_resolution": "1920x1080",
                "timezone": "America/New_York",
                "language": "en-US",
                "ip_asn": "AS15169",
            },
            {
                "user_id": "u1",
                "user_agent": "Safari/17 mobile",
                "screen_resolution": "1179x2556",
                "timezone": "America/New_York",
                "language": "en-US",
                "ip_asn": "AS13335",
            },
            {
                "user_id": "u2",
                "user_agent": "Firefox/123 desktop",
                "screen_resolution": "2560x1440",
                "timezone": "Europe/London",
                "language": "en-GB",
                "ip_asn": "AS8075",
            },
        ]
    )


def test_known_device_gets_low_novelty() -> None:
    df = _base_df()
    model = DeviceNoveltyScorer(eps=0.4, min_samples=1)
    model.fit(df)

    score = model.score_session(df.iloc[0])
    assert 0.0 <= score <= 0.4


def test_new_device_is_more_novel() -> None:
    df = _base_df()
    model = DeviceNoveltyScorer(eps=0.4, min_samples=1)
    model.fit(df)

    known_score = model.score_session(df.iloc[0])
    outlier = {
        "user_id": "u1",
        "user_agent": "python-requests/2.31 bot",
        "screen_resolution": "1024x768",
        "timezone": "Asia/Tokyo",
        "language": "ja-JP",
        "ip_asn": "AS9009",
    }
    outlier_score = model.score_session(outlier)

    assert outlier_score > known_score
    assert outlier_score >= 0.5


def test_new_user_without_history_scores_max_novelty() -> None:
    model = DeviceNoveltyScorer(min_samples=1)
    model.fit(_base_df())
    score = model.score_session(
        {
            "user_id": "unknown_user",
            "user_agent": "Chrome/122",
            "screen_resolution": "1920x1080",
            "timezone": "America/Los_Angeles",
            "language": "en-US",
            "ip_asn": "AS15169",
        }
    )
    assert score == 1.0
