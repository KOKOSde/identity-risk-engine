"""Per-user behavioral baseline and anomaly scoring."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def _safe_std(values: pd.Series) -> float:
    std = float(values.std(ddof=0)) if len(values) else 0.0
    return std if std > 1e-6 else 1.0


def _z_to_unit(z: float, scale: float = 3.0) -> float:
    return float(np.clip(z / scale, 0.0, 1.0))


@dataclass
class UserBehaviorProfile:
    count: int
    hour_hist: np.ndarray
    duration_mean: float
    duration_std: float
    gap_mean: float
    gap_std: float
    actions_mean: float
    actions_std: float
    entropy_mean: float
    entropy_std: float
    iso_model: IsolationForest | None
    last_timestamp: pd.Timestamp | None


class BehaviorAnomalyScorer:
    """Build user behavior baselines and score incoming sessions."""

    def __init__(self, min_history: int = 5, random_state: int = 42) -> None:
        self.min_history = min_history
        self.random_state = random_state
        self.profiles: dict[str, UserBehaviorProfile] = {}

    @staticmethod
    def _prepare(df: pd.DataFrame) -> pd.DataFrame:
        required = {"user_id", "timestamp", "session_duration"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        out = df.copy()
        out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
        out["session_duration"] = pd.to_numeric(out["session_duration"], errors="coerce").fillna(0.0)

        if "actions_count" not in out.columns:
            out["actions_count"] = out.get("action_count", 0)
        out["actions_count"] = pd.to_numeric(out["actions_count"], errors="coerce").fillna(0.0)

        if "action_entropy" not in out.columns:
            out["action_entropy"] = 0.0
        out["action_entropy"] = pd.to_numeric(out["action_entropy"], errors="coerce").fillna(0.0)

        return out

    def fit(self, history_df: pd.DataFrame) -> BehaviorAnomalyScorer:
        df = self._prepare(history_df)
        self.profiles = {}

        if df.empty:
            return self

        for user_id, grp in df.groupby("user_id", sort=False):
            grp = grp.sort_values("timestamp")
            count = len(grp)

            hours = grp["timestamp"].dt.hour.fillna(0).astype(int).to_numpy()
            hour_hist = np.bincount(hours, minlength=24).astype(float)
            hour_hist = hour_hist / max(float(hour_hist.sum()), 1.0)

            duration_mean = float(grp["session_duration"].mean())
            duration_std = _safe_std(grp["session_duration"])

            actions_mean = float(grp["actions_count"].mean())
            actions_std = _safe_std(grp["actions_count"])

            entropy_mean = float(grp["action_entropy"].mean())
            entropy_std = _safe_std(grp["action_entropy"])

            gaps = grp["timestamp"].diff().dt.total_seconds().fillna(0.0) / 3600.0
            gaps = gaps.clip(lower=0.0)
            gap_mean = float(gaps.mean())
            gap_std = _safe_std(gaps)

            iso_model = None
            if count >= self.min_history:
                features = np.column_stack(
                    [
                        hours,
                        np.log1p(grp["session_duration"].to_numpy()),
                        grp["actions_count"].to_numpy(),
                        grp["action_entropy"].to_numpy(),
                        gaps.to_numpy(),
                    ]
                )
                iso_model = IsolationForest(
                    n_estimators=100,
                    contamination="auto",
                    random_state=self.random_state,
                )
                iso_model.fit(features)

            self.profiles[str(user_id)] = UserBehaviorProfile(
                count=count,
                hour_hist=hour_hist,
                duration_mean=duration_mean,
                duration_std=duration_std,
                gap_mean=gap_mean,
                gap_std=gap_std,
                actions_mean=actions_mean,
                actions_std=actions_std,
                entropy_mean=entropy_mean,
                entropy_std=entropy_std,
                iso_model=iso_model,
                last_timestamp=grp["timestamp"].iloc[-1],
            )

        return self

    def _score_row(self, row: pd.Series) -> tuple[float, bool]:
        user_id = str(row["user_id"])
        profile = self.profiles.get(user_id)
        if profile is None:
            return 0.6, True

        timestamp = row["timestamp"]
        hour = int(timestamp.hour) if pd.notna(timestamp) else 0
        hour_prob = float(profile.hour_hist[hour])
        hour_score = float(
            np.clip(1.0 - hour_prob / max(float(profile.hour_hist.max()), 1e-6), 0.0, 1.0)
        )

        duration = float(row["session_duration"])
        actions = float(row.get("actions_count", 0.0))
        entropy = float(row.get("action_entropy", 0.0))

        z_duration = abs(duration - profile.duration_mean) / profile.duration_std
        z_actions = abs(actions - profile.actions_mean) / profile.actions_std
        z_entropy = abs(entropy - profile.entropy_mean) / profile.entropy_std

        if profile.last_timestamp is not None and pd.notna(timestamp):
            gap_h = max((timestamp - profile.last_timestamp).total_seconds() / 3600.0, 0.0)
        else:
            gap_h = profile.gap_mean
        z_gap = abs(gap_h - profile.gap_mean) / profile.gap_std

        duration_score = _z_to_unit(z_duration)
        actions_score = _z_to_unit(z_actions)
        entropy_score = _z_to_unit(z_entropy)
        gap_score = _z_to_unit(z_gap)

        iso_score = 0.0
        if profile.iso_model is not None:
            feat = np.array([[hour, np.log1p(duration), actions, entropy, gap_h]], dtype=float)
            raw = -float(profile.iso_model.decision_function(feat)[0])
            iso_score = float(1.0 / (1.0 + np.exp(-3.0 * raw)))

        insufficient = profile.count < self.min_history
        if insufficient:
            score = (
                0.25 * hour_score
                + 0.30 * duration_score
                + 0.20 * actions_score
                + 0.15 * gap_score
                + 0.10 * entropy_score
            )
        else:
            score = (
                0.30 * iso_score
                + 0.15 * hour_score
                + 0.20 * duration_score
                + 0.15 * actions_score
                + 0.10 * gap_score
                + 0.10 * entropy_score
            )

        return float(np.clip(score, 0.0, 1.0)), bool(insufficient)

    def score(self, sessions_df: pd.DataFrame) -> pd.DataFrame:
        df = self._prepare(sessions_df)
        if df.empty:
            return pd.DataFrame(
                {
                    "behavior_anomaly_score": pd.Series(dtype=float),
                    "insufficient_baseline": pd.Series(dtype=object),
                }
            )

        scores: list[float] = []
        insufficient_flags: list[bool] = []
        for _, row in df.iterrows():
            score, insufficient = self._score_row(row)
            scores.append(score)
            insufficient_flags.append(bool(insufficient))

        result = pd.DataFrame(index=df.index)
        result["behavior_anomaly_score"] = scores
        result["insufficient_baseline"] = pd.Series(
            [bool(v) for v in insufficient_flags],
            index=df.index,
            dtype=object,
        )
        return result

    def score_dataframe(self, sessions_df: pd.DataFrame) -> pd.Series:
        """Return only the numeric anomaly score series."""

        return self.score(sessions_df)["behavior_anomaly_score"]

    def score_session(self, session: dict[str, object] | pd.Series) -> float:
        """Score a single session payload."""

        if isinstance(session, pd.Series):
            frame = pd.DataFrame([session.to_dict()])
        else:
            frame = pd.DataFrame([dict(session)])
        return float(self.score_dataframe(frame).iloc[0])
