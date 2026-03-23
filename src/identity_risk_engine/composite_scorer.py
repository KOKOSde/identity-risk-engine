"""Composite ensemble scorer for session-level identity risk."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

try:  # pragma: no cover
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None

try:  # pragma: no cover
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover
    LGBMClassifier = None

REQUIRED_FEATURES = [
    "geo_velocity_score",
    "device_novelty_score",
    "behavior_anomaly_score",
    "ip_reputation",
    "session_duration",
    "country_mismatch",
    "failed_attempts",
    "account_age_days",
]


@dataclass
class OperatingPoint:
    threshold: float
    precision: float
    recall: float


def _prepare_features(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    missing = [col for col in feature_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    out = df[feature_columns].copy()
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
        if out[col].isna().all():
            out[col] = 0.0
        else:
            out[col] = out[col].fillna(float(out[col].median()))
    return out


def _precision_recall(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    return (
        float(precision_score(y_true, y_pred, zero_division=0)),
        float(recall_score(y_true, y_pred, zero_division=0)),
    )


def _search_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    min_recall: Optional[float] = None,
    min_precision: Optional[float] = None,
) -> OperatingPoint:
    thresholds = np.linspace(0.01, 0.99, 199)
    best: Optional[OperatingPoint] = None

    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(int)
        precision, recall = _precision_recall(y_true, y_pred)

        if min_recall is not None and recall >= min_recall:
            candidate = OperatingPoint(float(threshold), precision, recall)
            if best is None or candidate.precision > best.precision:
                best = candidate

        if min_precision is not None and precision >= min_precision:
            candidate = OperatingPoint(float(threshold), precision, recall)
            if best is None or candidate.recall > best.recall:
                best = candidate

    if best is not None:
        return best

    fallback_threshold = 0.5
    fallback_pred = (y_score >= fallback_threshold).astype(int)
    precision, recall = _precision_recall(y_true, fallback_pred)
    return OperatingPoint(threshold=fallback_threshold, precision=precision, recall=recall)


def _precision_at_recall(y_true: np.ndarray, y_score: np.ndarray, target_recall: float) -> float:
    best = 0.0
    for threshold in np.linspace(0.01, 0.99, 199):
        y_pred = (y_score >= threshold).astype(int)
        precision, recall = _precision_recall(y_true, y_pred)
        if recall >= target_recall and precision > best:
            best = precision
    return float(best)


def _recall_at_precision(y_true: np.ndarray, y_score: np.ndarray, target_precision: float) -> float:
    best = 0.0
    for threshold in np.linspace(0.01, 0.99, 199):
        y_pred = (y_score >= threshold).astype(int)
        precision, recall = _precision_recall(y_true, y_pred)
        if precision >= target_precision and recall > best:
            best = recall
    return float(best)


def enrich_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute and append feature columns expected by the composite model."""

    from .behavior_anomaly import BehaviorAnomalyScorer
    from .device_fingerprint import DeviceNoveltyScorer
    from .geo_velocity import compute_geo_velocity_features

    out = df.copy()

    geo = compute_geo_velocity_features(
        out[[col for col in ["user_id", "timestamp", "lat", "lon"] if col in out.columns]]
    )
    out["geo_velocity_score"] = geo["geo_velocity_score"].to_numpy()

    device = DeviceNoveltyScorer()
    device.fit(out)
    out["device_novelty_score"] = device.score_dataframe(out).to_numpy()

    behavior = BehaviorAnomalyScorer(min_history=5)
    behavior.fit(out)
    out["behavior_anomaly_score"] = behavior.score_dataframe(out).to_numpy()

    return out


class CompositeRiskScorer:
    """XGBoost + LightGBM ensemble with calibration and threshold tuning."""

    def __init__(
        self,
        feature_columns: Optional[list[str]] = None,
        random_state: int = 42,
    ) -> None:
        self.feature_columns = feature_columns or REQUIRED_FEATURES.copy()
        self.random_state = random_state

        self.model_xgb = None
        self.model_lgbm = None
        self.calibrator: Optional[IsotonicRegression] = None
        self._fitted = False

        self._validation_scores: Optional[np.ndarray] = None
        self._validation_targets: Optional[np.ndarray] = None

        self.operating_points: dict[str, OperatingPoint] = {}

    def _make_xgb(self):
        if XGBClassifier is None:
            return GradientBoostingClassifier(random_state=self.random_state)
        return XGBClassifier(
            n_estimators=120,
            max_depth=4,
            learning_rate=0.05,
            subsample=1.0,
            colsample_bytree=1.0,
            eval_metric="logloss",
            n_jobs=1,
            tree_method="hist",
            random_state=self.random_state,
        )

    def _make_lgbm(self):
        if LGBMClassifier is None:
            return GradientBoostingClassifier(random_state=self.random_state + 1)
        return LGBMClassifier(
            n_estimators=160,
            learning_rate=0.05,
            num_leaves=31,
            subsample=1.0,
            colsample_bytree=1.0,
            n_jobs=1,
            random_state=self.random_state,
            bagging_seed=self.random_state,
            feature_fraction_seed=self.random_state,
            data_random_seed=self.random_state,
            deterministic=True,
            force_col_wise=True,
            verbosity=-1,
        )

    def fit(self, df: pd.DataFrame, target_col: str = "is_attack") -> CompositeRiskScorer:
        if target_col not in df.columns:
            raise ValueError(f"Missing target column: {target_col}")

        X = _prepare_features(df, self.feature_columns)
        y = pd.to_numeric(df[target_col], errors="coerce").fillna(0).astype(int).to_numpy()

        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=0.25,
            stratify=y,
            random_state=self.random_state,
        )

        stage_xgb = self._make_xgb()
        stage_lgbm = self._make_lgbm()
        stage_xgb.fit(X_train, y_train)
        stage_lgbm.fit(X_train, y_train)

        val_raw = (
            0.5 * stage_xgb.predict_proba(X_val)[:, 1]
            + 0.5 * stage_lgbm.predict_proba(X_val)[:, 1]
        )
        self.calibrator = IsotonicRegression(out_of_bounds="clip")
        self.calibrator.fit(val_raw, y_val)
        val_calibrated = np.asarray(self.calibrator.predict(val_raw), dtype=float)

        self._validation_scores = val_calibrated
        self._validation_targets = y_val
        self.tune_thresholds(
            pd.DataFrame({"_y": y_val, "_p": val_calibrated}),
            target_col="_y",
            proba_col="_p",
            min_recall=0.95,
            min_precision=0.95,
        )

        self.model_xgb = self._make_xgb()
        self.model_lgbm = self._make_lgbm()
        self.model_xgb.fit(X, y)
        self.model_lgbm.fit(X, y)

        self._fitted = True
        return self

    def _predict_positive_proba(self, df: pd.DataFrame) -> np.ndarray:
        if not self._fitted or self.model_xgb is None or self.model_lgbm is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        X = _prepare_features(df, self.feature_columns)
        raw = (
            0.5 * self.model_xgb.predict_proba(X)[:, 1]
            + 0.5 * self.model_lgbm.predict_proba(X)[:, 1]
        )
        if self.calibrator is not None:
            return np.asarray(self.calibrator.predict(raw), dtype=float)
        return np.asarray(raw, dtype=float)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Return class probabilities with shape (n_samples, 2)."""

        pos = self._predict_positive_proba(df)
        neg = 1.0 - pos
        return np.column_stack([neg, pos])

    def predict(self, df: pd.DataFrame, mode: str = "friction_mode") -> np.ndarray:
        if mode not in self.operating_points:
            raise ValueError(f"Unknown mode: {mode}")
        threshold = self.operating_points[mode].threshold
        return (self._predict_positive_proba(df) >= threshold).astype(int)

    def tune_thresholds(
        self,
        df: pd.DataFrame,
        target_col: str = "is_attack",
        proba_col: Optional[str] = None,
        min_recall: float = 0.95,
        min_precision: float = 0.95,
    ) -> dict[str, float]:
        if target_col not in df.columns:
            raise ValueError(f"Missing target column: {target_col}")

        y_true = pd.to_numeric(df[target_col], errors="coerce").fillna(0).astype(int).to_numpy()
        if proba_col is None:
            y_score = self._predict_positive_proba(df)
        else:
            y_score = pd.to_numeric(df[proba_col], errors="coerce").fillna(0.0).to_numpy()

        block = _search_threshold(y_true, y_score, min_recall=min_recall)
        friction = _search_threshold(y_true, y_score, min_precision=min_precision)

        # Ensure expected order for decision strictness.
        block_threshold = min(block.threshold, friction.threshold)
        friction_threshold = max(block.threshold, friction.threshold)

        self.operating_points = {
            "block_mode": OperatingPoint(
                threshold=float(block_threshold),
                precision=block.precision,
                recall=block.recall,
            ),
            "friction_mode": OperatingPoint(
                threshold=float(friction_threshold),
                precision=friction.precision,
                recall=friction.recall,
            ),
        }

        return {
            "block_mode": float(self.operating_points["block_mode"].threshold),
            "friction_mode": float(self.operating_points["friction_mode"].threshold),
        }

    def get_operating_point(self, mode: str) -> OperatingPoint:
        if mode not in self.operating_points:
            raise ValueError(f"Unknown mode: {mode}")
        return self.operating_points[mode]

    def get_operating_points(self) -> dict[str, dict[str, float]]:
        return {
            name: {
                "threshold": float(point.threshold),
                "precision": float(point.precision),
                "recall": float(point.recall),
            }
            for name, point in self.operating_points.items()
        }

    def feature_importance(self) -> pd.DataFrame:
        if not self._fitted or self.model_xgb is None or self.model_lgbm is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        xgb_importance = np.asarray(
            getattr(self.model_xgb, "feature_importances_", np.zeros(len(self.feature_columns))),
            dtype=float,
        )
        lgbm_importance = np.asarray(
            getattr(self.model_lgbm, "feature_importances_", np.zeros(len(self.feature_columns))),
            dtype=float,
        )

        if xgb_importance.sum() > 0:
            xgb_importance = xgb_importance / xgb_importance.sum()
        if lgbm_importance.sum() > 0:
            lgbm_importance = lgbm_importance / lgbm_importance.sum()

        combined = 0.5 * xgb_importance + 0.5 * lgbm_importance
        return (
            pd.DataFrame({"feature": self.feature_columns, "importance": combined})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def evaluate(
        self,
        df: pd.DataFrame,
        target_col: str = "is_attack",
        attack_type_col: str = "attack_type",
    ) -> dict[str, object]:
        if target_col not in df.columns:
            raise ValueError(f"Missing target column: {target_col}")

        y = pd.to_numeric(df[target_col], errors="coerce").fillna(0).astype(int).to_numpy()
        proba = self._predict_positive_proba(df)

        overall = {
            "auc": float(roc_auc_score(y, proba)),
            "precision_at_95_recall": _precision_at_recall(y, proba, 0.95),
            "recall_at_95_precision": _recall_at_precision(y, proba, 0.95),
        }

        if attack_type_col not in df.columns:
            return {"global": overall, "by_attack": pd.DataFrame()}

        rows = []
        for attack_type in sorted(df[attack_type_col].dropna().astype(str).unique()):
            if attack_type == "normal":
                continue
            y_attack = (df[attack_type_col].astype(str) == attack_type).astype(int).to_numpy()
            if y_attack.sum() == 0:
                continue

            if y_attack.sum() == len(y_attack):
                attack_auc = 1.0
            else:
                attack_auc = float(roc_auc_score(y_attack, proba))

            rows.append(
                {
                    "attack_type": attack_type,
                    "auc": attack_auc,
                    "precision_at_95_recall": _precision_at_recall(y_attack, proba, 0.95),
                    "recall_at_95_precision": _recall_at_precision(y_attack, proba, 0.95),
                }
            )

        breakdown = pd.DataFrame(rows).sort_values("attack_type").reset_index(drop=True)
        return {"global": overall, "by_attack": breakdown}
