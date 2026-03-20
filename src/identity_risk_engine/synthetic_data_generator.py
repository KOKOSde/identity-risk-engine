"""Synthetic session data generation for identity-risk experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .behavior_anomaly import BehaviorAnomalyScorer
from .device_fingerprint import DeviceNoveltyScorer
from .geo_velocity import compute_geo_velocity_features

ATTACK_TYPES = (
    "account_takeover",
    "credential_stuffing",
    "bot_behavior",
    "impossible_travel",
    "new_account_fraud",
)

SAFE_ASNS = [f"AS{asn}" for asn in (15169, 13335, 14618, 8075, 20940, 16509)]
RISKY_ASNS = [f"AS{asn}" for asn in (9009, 20473, 14061, 398324, 16276, 4134)]

DEVICE_TEMPLATES = {
    "desktop": [
        (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/122.0",
            "1920x1080",
        ),
        (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_3) AppleWebKit/605.1.15 Version/17.2 Safari/605.1.15",
            "2560x1440",
        ),
        (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/121.0",
            "1920x1200",
        ),
    ],
    "mobile": [
        (
            "Mozilla/5.0 (iPhone; CPU iPhone OS 17_3 like Mac OS X) AppleWebKit/605.1.15 Mobile/15E148",
            "1179x2556",
        ),
        (
            "Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 Chrome/121.0 Mobile",
            "1080x2400",
        ),
        (
            "Mozilla/5.0 (Linux; Android 13; SM-S918U) AppleWebKit/537.36 SamsungBrowser/23.0 Mobile",
            "1080x2340",
        ),
    ],
}

BOT_USER_AGENT = "python-requests/2.31 bot-session"

LOCATIONS = [
    {
        "country": "US",
        "city": "San Francisco",
        "lat": 37.7749,
        "lon": -122.4194,
        "timezone": "America/Los_Angeles",
        "language": "en-US",
        "region": "NA",
    },
    {
        "country": "US",
        "city": "New York",
        "lat": 40.7128,
        "lon": -74.0060,
        "timezone": "America/New_York",
        "language": "en-US",
        "region": "NA",
    },
    {
        "country": "GB",
        "city": "London",
        "lat": 51.5074,
        "lon": -0.1278,
        "timezone": "Europe/London",
        "language": "en-GB",
        "region": "EU",
    },
    {
        "country": "DE",
        "city": "Berlin",
        "lat": 52.52,
        "lon": 13.405,
        "timezone": "Europe/Berlin",
        "language": "de-DE",
        "region": "EU",
    },
    {
        "country": "BR",
        "city": "Sao Paulo",
        "lat": -23.5505,
        "lon": -46.6333,
        "timezone": "America/Sao_Paulo",
        "language": "pt-BR",
        "region": "LATAM",
    },
    {
        "country": "IN",
        "city": "Mumbai",
        "lat": 19.076,
        "lon": 72.8777,
        "timezone": "Asia/Kolkata",
        "language": "en-IN",
        "region": "APAC",
    },
    {
        "country": "JP",
        "city": "Tokyo",
        "lat": 35.6762,
        "lon": 139.6503,
        "timezone": "Asia/Tokyo",
        "language": "ja-JP",
        "region": "APAC",
    },
    {
        "country": "AU",
        "city": "Sydney",
        "lat": -33.8688,
        "lon": 151.2093,
        "timezone": "Australia/Sydney",
        "language": "en-AU",
        "region": "APAC",
    },
]


@dataclass
class UserProfile:
    user_id: str
    home_location: dict[str, object]
    preferred_hour: int
    device_type: str
    user_agent: str
    screen_resolution: str
    account_created: pd.Timestamp


def _pick_location(rng: np.random.Generator, *, not_country: str | None = None) -> dict[str, object]:
    if not_country is None:
        return dict(LOCATIONS[int(rng.integers(0, len(LOCATIONS)))])
    options = [loc for loc in LOCATIONS if str(loc["country"]) != not_country]
    return dict(options[int(rng.integers(0, len(options)))])


def _pick_device(rng: np.random.Generator, device_type: str) -> tuple[str, str]:
    options = DEVICE_TEMPLATES[device_type]
    return options[int(rng.integers(0, len(options)))]


def _assign_attacks(df: pd.DataFrame, attack_count: int, rng: np.random.Generator) -> list[tuple[int, str]]:
    if attack_count <= 0 or df.empty:
        return []

    attack_count = min(attack_count, len(df))
    with_pos = df.copy()
    with_pos["event_pos"] = with_pos.groupby("user_id", sort=False).cumcount()

    chosen: list[tuple[int, str]] = []
    used: set[int] = set()

    mandatory = list(ATTACK_TYPES[: min(len(ATTACK_TYPES), attack_count)])
    for attack_type in mandatory:
        if attack_type == "impossible_travel":
            candidates = with_pos.index[with_pos["event_pos"] > 0].to_numpy()
        else:
            candidates = with_pos.index.to_numpy()
        candidates = np.array([idx for idx in candidates if int(idx) not in used], dtype=int)
        if len(candidates) == 0:
            continue
        selected = int(candidates[int(rng.integers(0, len(candidates)))])
        used.add(selected)
        chosen.append((selected, attack_type))

    remaining = attack_count - len(chosen)
    if remaining <= 0:
        return chosen

    available = np.array([idx for idx in with_pos.index.to_numpy() if int(idx) not in used], dtype=int)
    if len(available) == 0:
        return chosen

    sampled_indices = rng.choice(available, size=min(remaining, len(available)), replace=False)
    sampled_types = rng.choice(np.array(ATTACK_TYPES, dtype=object), size=len(sampled_indices), replace=True)
    for idx, attack_type in zip(sampled_indices, sampled_types):
        chosen.append((int(idx), str(attack_type)))

    return chosen


def generate_synthetic_login_data(
    num_users: int = 200,
    num_sessions: int = 12000,
    attack_ratio: float = 0.2,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate realistic synthetic login sessions with multiple attack patterns."""

    if num_users <= 0:
        raise ValueError("num_users must be positive")
    if num_sessions <= 0:
        raise ValueError("num_sessions must be positive")
    if not 0.0 <= attack_ratio <= 1.0:
        raise ValueError("attack_ratio must be in [0, 1]")

    rng = np.random.default_rng(seed)
    start_time = pd.Timestamp("2026-01-01T00:00:00Z")

    user_ids = [f"user_{i:05d}" for i in range(num_users)]
    sessions_per_user = rng.multinomial(num_sessions, np.full(num_users, 1.0 / num_users))

    profiles: dict[str, UserProfile] = {}
    for user_id in user_ids:
        home = _pick_location(rng)
        device_type = "mobile" if rng.random() < 0.45 else "desktop"
        user_agent, screen_resolution = _pick_device(rng, device_type)
        preferred_hour = int(rng.integers(7, 23))
        account_created = start_time - pd.Timedelta(days=float(rng.uniform(90, 1800)))
        profiles[user_id] = UserProfile(
            user_id=user_id,
            home_location=home,
            preferred_hour=preferred_hour,
            device_type=device_type,
            user_agent=user_agent,
            screen_resolution=screen_resolution,
            account_created=account_created,
        )

    rows: list[dict[str, object]] = []
    session_counter = 0
    for user_id, count in zip(user_ids, sessions_per_user):
        if count <= 0:
            continue

        profile = profiles[user_id]
        current = start_time + pd.Timedelta(days=float(rng.uniform(0, 4)))

        for _ in range(int(count)):
            current = current + pd.Timedelta(hours=max(float(rng.normal(18.0, 5.0)), 0.4))
            hour = int(np.clip(round(rng.normal(profile.preferred_hour, 2.0)), 0, 23))
            minute = int(rng.integers(0, 60))
            ts = pd.Timestamp(
                year=current.year,
                month=current.month,
                day=current.day,
                hour=hour,
                minute=minute,
                tz="UTC",
            )

            home = profile.home_location
            rows.append(
                {
                    "session_id": f"sess_{session_counter:08d}",
                    "user_id": user_id,
                    "timestamp": ts,
                    "country": str(home["country"]),
                    "city": str(home["city"]),
                    "region": str(home["region"]),
                    "lat": float(home["lat"]) + float(rng.normal(0.0, 0.06)),
                    "lon": float(home["lon"]) + float(rng.normal(0.0, 0.06)),
                    "timezone": str(home["timezone"]),
                    "language": str(home["language"]),
                    "user_agent": profile.user_agent,
                    "screen_resolution": profile.screen_resolution,
                    "device_type": profile.device_type,
                    "ip_asn": SAFE_ASNS[int(rng.integers(0, len(SAFE_ASNS)))],
                    "ip_reputation": float(rng.beta(2.0, 8.0)),
                    "session_duration": float(np.clip(rng.normal(420.0, 110.0), 50.0, 2400.0)),
                    "actions_count": int(np.clip(rng.poisson(5), 1, 20)),
                    "action_entropy": float(np.clip(rng.normal(1.2, 0.25), 0.2, 2.5)),
                    "failed_attempts": int(min(rng.poisson(0.5), 3)),
                    "high_value_action": 1 if rng.random() < 0.03 else 0,
                    "success": 1,
                    "attack_type": "normal",
                    "home_country": str(home["country"]),
                    "account_created": profile.account_created,
                }
            )
            session_counter += 1

    df = pd.DataFrame(rows).sort_values(["user_id", "timestamp", "session_id"]).reset_index(drop=True)

    attack_count = int(round(num_sessions * attack_ratio))
    assignments = _assign_attacks(df, attack_count=attack_count, rng=rng)

    grouped_indices = {
        user: np.asarray(indices, dtype=int) for user, indices in df.groupby("user_id", sort=False).indices.items()
    }

    for idx, attack_type in assignments:
        user_id = str(df.loc[idx, "user_id"])
        profile = profiles[user_id]

        if attack_type == "account_takeover":
            foreign = _pick_location(rng, not_country=str(profile.home_location["country"]))
            alt_type = "desktop" if profile.device_type == "mobile" else "mobile"
            user_agent, resolution = _pick_device(rng, alt_type)

            df.loc[idx, ["country", "city", "region", "lat", "lon", "timezone", "language"]] = [
                foreign["country"],
                foreign["city"],
                foreign["region"],
                float(foreign["lat"]),
                float(foreign["lon"]),
                foreign["timezone"],
                foreign["language"],
            ]
            df.loc[idx, "user_agent"] = user_agent
            df.loc[idx, "screen_resolution"] = resolution
            df.loc[idx, "device_type"] = alt_type
            df.loc[idx, "ip_asn"] = RISKY_ASNS[int(rng.integers(0, len(RISKY_ASNS)))]
            df.loc[idx, "ip_reputation"] = float(rng.uniform(0.75, 1.0))
            df.loc[idx, "failed_attempts"] = int(rng.integers(3, 9))
            df.loc[idx, "session_duration"] = float(rng.uniform(40.0, 260.0))
            df.loc[idx, "actions_count"] = int(rng.integers(1, 4))
            df.loc[idx, "action_entropy"] = float(rng.uniform(0.1, 0.8))

        elif attack_type == "credential_stuffing":
            df.loc[idx, "ip_asn"] = RISKY_ASNS[int(rng.integers(0, len(RISKY_ASNS)))]
            df.loc[idx, "ip_reputation"] = float(rng.uniform(0.82, 1.0))
            df.loc[idx, "failed_attempts"] = int(rng.integers(8, 28))
            df.loc[idx, "session_duration"] = float(rng.uniform(15.0, 140.0))
            df.loc[idx, "actions_count"] = int(rng.integers(1, 3))
            df.loc[idx, "action_entropy"] = float(rng.uniform(0.05, 0.5))

        elif attack_type == "bot_behavior":
            user_positions = grouped_indices[user_id]
            pos = int(np.where(user_positions == idx)[0][0]) if idx in set(user_positions.tolist()) else -1
            if pos > 0:
                prev_idx = int(user_positions[pos - 1])
                prev_ts = pd.to_datetime(df.loc[prev_idx, "timestamp"], utc=True)
                df.loc[idx, "timestamp"] = (prev_ts + pd.Timedelta(minutes=15)).floor("us")

            df.loc[idx, "user_agent"] = BOT_USER_AGENT
            df.loc[idx, "screen_resolution"] = "1024x768"
            df.loc[idx, "device_type"] = "bot"
            df.loc[idx, "ip_asn"] = RISKY_ASNS[int(rng.integers(0, len(RISKY_ASNS)))]
            df.loc[idx, "ip_reputation"] = float(rng.uniform(0.7, 0.98))
            df.loc[idx, "session_duration"] = float(rng.uniform(8.0, 50.0))
            df.loc[idx, "actions_count"] = 1
            df.loc[idx, "action_entropy"] = float(rng.uniform(0.01, 0.2))
            df.loc[idx, "failed_attempts"] = int(rng.integers(0, 3))

        elif attack_type == "impossible_travel":
            user_positions = grouped_indices[user_id]
            pos = int(np.where(user_positions == idx)[0][0]) if idx in set(user_positions.tolist()) else -1
            if pos > 0:
                prev_idx = int(user_positions[pos - 1])
                prev_country = str(df.loc[prev_idx, "country"])
                far = _pick_location(rng, not_country=prev_country)
                prev_ts = pd.to_datetime(df.loc[prev_idx, "timestamp"], utc=True)
                df.loc[idx, "timestamp"] = (
                    prev_ts + pd.Timedelta(minutes=float(rng.uniform(4.0, 22.0)))
                ).floor("us")
                df.loc[idx, ["country", "city", "region", "lat", "lon", "timezone", "language"]] = [
                    far["country"],
                    far["city"],
                    far["region"],
                    float(far["lat"]),
                    float(far["lon"]),
                    far["timezone"],
                    far["language"],
                ]
            df.loc[idx, "ip_asn"] = RISKY_ASNS[int(rng.integers(0, len(RISKY_ASNS)))]
            df.loc[idx, "ip_reputation"] = float(rng.uniform(0.8, 1.0))
            df.loc[idx, "failed_attempts"] = int(rng.integers(1, 6))

        elif attack_type == "new_account_fraud":
            ts = pd.to_datetime(df.loc[idx, "timestamp"], utc=True)
            df.loc[idx, "account_created"] = (
                ts - pd.Timedelta(hours=float(rng.uniform(1.0, 18.0)))
            ).floor("us")
            df.loc[idx, "high_value_action"] = 1
            df.loc[idx, "ip_reputation"] = float(rng.uniform(0.75, 1.0))
            df.loc[idx, "failed_attempts"] = int(rng.integers(0, 4))
            df.loc[idx, "session_duration"] = float(rng.uniform(130.0, 620.0))
            df.loc[idx, "actions_count"] = int(rng.integers(3, 10))
            df.loc[idx, "action_entropy"] = float(rng.uniform(0.6, 2.1))
            if rng.random() < 0.7:
                foreign = _pick_location(rng, not_country=str(profile.home_location["country"]))
                df.loc[idx, ["country", "city", "region", "lat", "lon", "timezone", "language"]] = [
                    foreign["country"],
                    foreign["city"],
                    foreign["region"],
                    float(foreign["lat"]),
                    float(foreign["lon"]),
                    foreign["timezone"],
                    foreign["language"],
                ]

        df.loc[idx, "attack_type"] = attack_type

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.sort_values(["user_id", "timestamp", "session_id"], kind="mergesort").reset_index(drop=True)

    df["account_created"] = pd.to_datetime(df["account_created"], utc=True, errors="coerce")
    df["account_age_days"] = (
        (df["timestamp"] - df["account_created"]).dt.total_seconds() / 86400.0
    ).clip(lower=0.0)

    df["country_mismatch"] = (df["country"] != df["home_country"]).astype(int)
    df["is_attack"] = (df["attack_type"] != "normal").astype(int)
    df["label"] = df["is_attack"].astype(int)

    geo = compute_geo_velocity_features(df[["user_id", "timestamp", "lat", "lon"]])
    df["distance_km"] = geo["distance_km"].to_numpy()
    df["speed_kmh"] = geo["speed_kmh"].to_numpy()
    df["impossible_travel_flag"] = geo["impossible_travel"].astype(int).to_numpy()
    df["geo_velocity_score"] = geo["geo_velocity_score"].to_numpy()

    baseline = df[df["attack_type"] == "normal"]
    if len(baseline) < 50:
        baseline = df

    device_model = DeviceNoveltyScorer()
    device_model.fit(baseline)
    device_raw = device_model.score_dataframe(df).to_numpy()

    behavior_model = BehaviorAnomalyScorer(min_history=5)
    behavior_model.fit(
        baseline[
            ["user_id", "timestamp", "session_duration", "actions_count", "action_entropy"]
        ]
    )
    behavior_scored = behavior_model.score(
        df[["user_id", "timestamp", "session_duration", "actions_count", "action_entropy"]]
    )
    behavior_raw = behavior_scored["behavior_anomaly_score"].to_numpy()
    df["insufficient_baseline"] = behavior_scored["insufficient_baseline"].to_numpy()

    attack_bias = {
        "normal": 0.0,
        "account_takeover": 0.32,
        "credential_stuffing": 0.24,
        "bot_behavior": 0.26,
        "impossible_travel": 0.30,
        "new_account_fraud": 0.22,
    }
    bias = df["attack_type"].map(attack_bias).fillna(0.0).to_numpy(dtype=float)

    df["device_novelty_score"] = np.clip(device_raw + 0.85 * bias + rng.normal(0.0, 0.03, len(df)), 0.0, 1.0)
    df["behavior_anomaly_score"] = np.clip(
        behavior_raw + 0.75 * bias + rng.normal(0.0, 0.04, len(df)),
        0.0,
        1.0,
    )
    df["geo_velocity_score"] = np.clip(
        df["geo_velocity_score"].to_numpy(dtype=float)
        + np.where(df["attack_type"] == "impossible_travel", 0.35, 0.0)
        + rng.normal(0.0, 0.02, len(df)),
        0.0,
        1.0,
    )

    df["failed_attempts"] = pd.to_numeric(df["failed_attempts"], errors="coerce").fillna(0).astype(int)
    df["failed_attempts"] = df["failed_attempts"].clip(lower=0, upper=50)

    return df.reset_index(drop=True)


def train_test_time_split(df: pd.DataFrame, test_ratio: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simple timestamp-ordered split helper for demos/tests."""

    if not 0.0 < test_ratio < 1.0:
        raise ValueError("test_ratio must be in (0, 1)")

    ordered = df.sort_values("timestamp", kind="mergesort")
    cut = int((1.0 - test_ratio) * len(ordered))
    return ordered.iloc[:cut].copy(), ordered.iloc[cut:].copy()


__all__ = ["ATTACK_TYPES", "generate_synthetic_login_data", "train_test_time_split"]
