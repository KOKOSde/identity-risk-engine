"""Common helpers for `_ire` signal modules."""

from __future__ import annotations

from collections.abc import Mapping
from math import asin, cos, radians, sin, sqrt
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype


def signal(signal_name: str, fired: bool, score: float, evidence: str) -> dict[str, object]:
    return {
        "signal_name": signal_name,
        "fired": bool(fired),
        "score": float(np.clip(score, 0.0, 1.0)),
        "evidence": str(evidence),
    }


def to_frame(history: pd.DataFrame | list[Mapping[str, Any]] | None) -> pd.DataFrame:
    if history is None:
        return pd.DataFrame()
    if isinstance(history, pd.DataFrame):
        if "timestamp" not in history.columns or is_datetime64_any_dtype(history["timestamp"]):
            return history
        df = history.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        return df
    else:
        df = pd.DataFrame(list(history))

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    return df


def event_ts(event: Mapping[str, Any]) -> pd.Timestamp:
    ts = pd.to_datetime(event.get("timestamp"), utc=True, errors="coerce")
    if pd.isna(ts):
        return pd.Timestamp.now(tz="UTC")
    return ts


def filter_user(df: pd.DataFrame, user_id: str | None) -> pd.DataFrame:
    if user_id is None or df.empty or "user_id" not in df.columns:
        return pd.DataFrame()
    return df[df["user_id"].astype(str) == str(user_id)]


def in_window(df: pd.DataFrame, now: pd.Timestamp, minutes: int) -> pd.DataFrame:
    if df.empty or "timestamp" not in df.columns:
        return pd.DataFrame()
    start = now - pd.Timedelta(minutes=minutes)
    return df[(df["timestamp"] >= start) & (df["timestamp"] <= now)]


def to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"true", "1", "yes", "y"}


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km."""

    r = 6371.0088
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 2 * r * asin(sqrt(a))
