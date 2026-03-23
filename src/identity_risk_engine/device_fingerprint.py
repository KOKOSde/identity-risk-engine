"""Device novelty scoring with TF-IDF and per-user clustering."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Union

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer

REQUIRED_DEVICE_COLUMNS = [
    "user_agent",
    "screen_resolution",
    "timezone",
    "language",
    "ip_asn",
]


@dataclass
class UserDeviceProfile:
    """Per-user clustered device profile."""

    centroids: np.ndarray
    sample_count: int


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-12:
        return vec
    return vec / norm


def _device_text(df: pd.DataFrame) -> pd.Series:
    pieces = []
    for col in REQUIRED_DEVICE_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Missing required device column: {col}")
        pieces.append(col + "=" + df[col].fillna("missing").astype(str))

    text = pieces[0]
    for piece in pieces[1:]:
        text = text + " " + piece
    return text


class DeviceNoveltyScorer:
    """TF-IDF + DBSCAN novelty scorer for device fingerprints."""

    def __init__(self, eps: float = 0.55, min_samples: int = 2, max_features: int = 512) -> None:
        self.eps = eps
        self.min_samples = min_samples
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
        self.user_profiles: dict[str, UserDeviceProfile] = {}
        self._is_fitted = False

    def fit(self, df: pd.DataFrame) -> DeviceNoveltyScorer:
        if "user_id" not in df.columns:
            raise ValueError("Missing required column: user_id")

        self.user_profiles = {}
        if df.empty:
            self._is_fitted = True
            return self

        work = df.copy()
        text = _device_text(work)
        matrix = self.vectorizer.fit_transform(text).toarray()

        user_ids = work["user_id"].astype(str).to_numpy()
        for user in pd.unique(user_ids):
            idx = np.where(user_ids == user)[0]
            user_matrix = matrix[idx]
            if len(user_matrix) == 0:
                continue

            if len(user_matrix) == 1:
                centroids = np.array([_normalize(user_matrix[0])], dtype=float)
            else:
                labels = DBSCAN(metric="cosine", eps=self.eps, min_samples=self.min_samples).fit_predict(
                    user_matrix
                )
                centroids_list: list[np.ndarray] = []
                for label in sorted(set(labels)):
                    if label == -1:
                        continue
                    cluster_vectors = user_matrix[labels == label]
                    if len(cluster_vectors) == 0:
                        continue
                    centroids_list.append(_normalize(cluster_vectors.mean(axis=0)))

                if not centroids_list:
                    centroids_list = [_normalize(user_matrix.mean(axis=0))]
                centroids = np.vstack(centroids_list)

            self.user_profiles[user] = UserDeviceProfile(centroids=centroids, sample_count=len(idx))

        self._is_fitted = True
        return self

    def _row_to_df(self, session: Union[Mapping[str, object], pd.Series]) -> pd.DataFrame:
        if isinstance(session, pd.Series):
            return pd.DataFrame([session.to_dict()])
        return pd.DataFrame([dict(session)])

    def score_session(self, session: Union[Mapping[str, object], pd.Series]) -> float:
        return float(self.score_dataframe(self._row_to_df(session)).iloc[0])

    def score_dataframe(self, df: pd.DataFrame) -> pd.Series:
        if "user_id" not in df.columns:
            raise ValueError("Missing required column: user_id")

        if df.empty:
            return pd.Series(dtype=float, name="device_novelty_score")

        if not self._is_fitted or not self.user_profiles:
            return pd.Series(np.ones(len(df), dtype=float), index=df.index, name="device_novelty_score")

        vectors = self.vectorizer.transform(_device_text(df)).toarray()

        scores: list[float] = []
        for i, (_, row) in enumerate(df.iterrows()):
            user_id = str(row["user_id"])
            profile = self.user_profiles.get(user_id)
            if profile is None:
                scores.append(1.0)
                continue

            vec = _normalize(vectors[i])
            if np.linalg.norm(vec) <= 1e-12:
                scores.append(1.0)
                continue

            similarities = np.clip(profile.centroids @ vec, -1.0, 1.0)
            score = float(np.clip(1.0 - float(np.max(similarities)), 0.0, 1.0))
            scores.append(score)

        return pd.Series(scores, index=df.index, name="device_novelty_score")

    # Backward-compatible alias.
    def score(self, df: pd.DataFrame) -> pd.Series:
        return self.score_dataframe(df)


# Backward-compatible alias used by some external code paths.
DeviceFingerprintScorer = DeviceNoveltyScorer
