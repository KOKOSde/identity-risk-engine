"""Impossible-travel feature extraction using geodesic velocity."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0088
DEFAULT_MAX_AIRCRAFT_SPEED_KMH = 900.0


@dataclass(frozen=True)
class GeoVelocityConfig:
    """Configuration for geo-velocity computation."""

    max_speed_kmh: float = DEFAULT_MAX_AIRCRAFT_SPEED_KMH


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return great-circle distance in kilometers between two coordinates."""

    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = (
        np.sin(dphi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return float(EARTH_RADIUS_KM * c)


def _events_to_df(events: Union[pd.DataFrame, Iterable[Sequence[object]]]) -> pd.DataFrame:
    if isinstance(events, pd.DataFrame):
        df = events.copy()
    else:
        df = pd.DataFrame(events, columns=["user_id", "timestamp", "lat", "lon"])

    required = {"user_id", "timestamp", "lat", "lon"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    return df


def compute_geo_velocity_features(
    events: Union[pd.DataFrame, Iterable[Sequence[object]]],
    config: Optional[GeoVelocityConfig] = None,
) -> pd.DataFrame:
    """Compute per-event geo-velocity features and impossible-travel flags."""

    cfg = config or GeoVelocityConfig()
    df = _events_to_df(events)

    if df.empty:
        return df.assign(
            prev_timestamp=pd.NaT,
            prev_lat=np.nan,
            prev_lon=np.nan,
            distance_km=0.0,
            time_delta_hours=0.0,
            speed_kmh=0.0,
            geo_velocity_score=0.0,
            impossible_travel=False,
        )

    df = df.reset_index(drop=False).rename(columns={"index": "_orig_idx"})
    df = df.sort_values(["user_id", "timestamp", "_orig_idx"], kind="mergesort")

    grp = df.groupby("user_id", sort=False)
    df["prev_timestamp"] = grp["timestamp"].shift(1)
    df["prev_lat"] = grp["lat"].shift(1)
    df["prev_lon"] = grp["lon"].shift(1)

    valid = (
        df["lat"].notna()
        & df["lon"].notna()
        & df["prev_lat"].notna()
        & df["prev_lon"].notna()
        & df["timestamp"].notna()
        & df["prev_timestamp"].notna()
    )

    distance = np.zeros(len(df), dtype=float)
    for i in np.where(valid.to_numpy())[0]:
        distance[i] = haversine_km(
            float(df.iloc[i]["prev_lat"]),
            float(df.iloc[i]["prev_lon"]),
            float(df.iloc[i]["lat"]),
            float(df.iloc[i]["lon"]),
        )
    df["distance_km"] = distance

    delta_hours = (
        (df["timestamp"] - df["prev_timestamp"]).dt.total_seconds().fillna(0.0) / 3600.0
    )
    delta_hours = np.where(delta_hours > 0, delta_hours, 0.0)
    df["time_delta_hours"] = delta_hours

    speed = np.zeros(len(df), dtype=float)
    nonzero_delta = df["time_delta_hours"].to_numpy() > 0
    speed[nonzero_delta] = (
        df.loc[nonzero_delta, "distance_km"].to_numpy() / df.loc[nonzero_delta, "time_delta_hours"].to_numpy()
    )
    speed[~np.isfinite(speed)] = 0.0
    df["speed_kmh"] = speed

    impossible = (
        (df["distance_km"].to_numpy() > 0)
        & (df["speed_kmh"].to_numpy() > float(cfg.max_speed_kmh))
    )
    df["impossible_travel"] = impossible

    df["geo_velocity_score"] = np.clip(
        df["speed_kmh"].to_numpy() / float(cfg.max_speed_kmh),
        0.0,
        1.0,
    )

    out = df.sort_values("_orig_idx", kind="mergesort").drop(columns=["_orig_idx"])
    return out.reset_index(drop=True)


def flag_impossible_travel(
    events: Union[pd.DataFrame, Iterable[Sequence[object]]],
    config: Optional[GeoVelocityConfig] = None,
) -> pd.Series:
    """Return boolean impossible-travel flags in event order."""

    return compute_geo_velocity_features(events, config=config)["impossible_travel"]


class GeoVelocityDetector:
    """Backwards-compatible detector wrapper."""

    def __init__(self, max_speed_kmh: float = DEFAULT_MAX_AIRCRAFT_SPEED_KMH) -> None:
        self.config = GeoVelocityConfig(max_speed_kmh=max_speed_kmh)

    def score_events(self, events: Union[pd.DataFrame, Iterable[Sequence[object]]]) -> pd.DataFrame:
        return compute_geo_velocity_features(events, config=self.config)
