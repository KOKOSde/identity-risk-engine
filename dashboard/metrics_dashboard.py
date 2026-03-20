"""Plotly dashboard utilities for session-risk model analysis."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)


def _as_numpy_int(values: Iterable[int]) -> np.ndarray:
    return np.asarray(list(values), dtype=int)



def _as_numpy_float(values: Iterable[float]) -> np.ndarray:
    return np.asarray(list(values), dtype=float)



def roc_curve_figure(y_true: Iterable[int], y_score: Iterable[float]) -> go.Figure:
    """Create ROC curve figure."""
    y_true_arr = _as_numpy_int(y_true)
    y_score_arr = _as_numpy_float(y_score)

    fpr, tpr, _ = roc_curve(y_true_arr, y_score_arr)
    roc_auc = auc(fpr, tpr)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"ROC (AUC={roc_auc:.3f})",
            line={"width": 3},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random",
            line={"dash": "dash", "color": "gray"},
        )
    )
    fig.update_layout(
        title="ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template="plotly_white",
    )
    return fig



def pr_curve_figure(y_true: Iterable[int], y_score: Iterable[float]) -> go.Figure:
    """Create precision-recall curve figure."""
    y_true_arr = _as_numpy_int(y_true)
    y_score_arr = _as_numpy_float(y_score)

    precision, recall, _ = precision_recall_curve(y_true_arr, y_score_arr)
    pr_auc = auc(recall, precision)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=recall,
            y=precision,
            mode="lines",
            name=f"PR (AUC={pr_auc:.3f})",
            line={"width": 3},
        )
    )
    fig.update_layout(
        title="Precision-Recall Curve",
        xaxis_title="Recall",
        yaxis_title="Precision",
        template="plotly_white",
    )
    return fig



def feature_importance_figure(
    feature_importances: Mapping[str, float] | pd.DataFrame,
) -> go.Figure:
    """Create feature-importance bar chart from dict or DataFrame."""
    if isinstance(feature_importances, pd.DataFrame):
        if not {"feature", "importance"}.issubset(feature_importances.columns):
            raise ValueError("feature_importances DataFrame must include ['feature', 'importance'] columns")
        imp_df = feature_importances[["feature", "importance"]].copy()
    else:
        imp_df = pd.DataFrame(
            {
                "feature": list(feature_importances.keys()),
                "importance": list(feature_importances.values()),
            }
        )

    if imp_df.empty:
        imp_df = pd.DataFrame({"feature": ["n/a"], "importance": [0.0]})

    imp_df = imp_df.sort_values("importance", ascending=False).reset_index(drop=True)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=imp_df["feature"],
            y=imp_df["importance"],
            marker_color="#1f77b4",
            name="Importance",
        )
    )
    fig.update_layout(
        title="Feature Importance",
        xaxis_title="Feature",
        yaxis_title="Importance",
        template="plotly_white",
    )
    return fig



def confusion_matrix_figure(
    y_true: Iterable[int],
    y_score: Iterable[float],
    threshold: float = 0.5,
) -> go.Figure:
    """Create confusion matrix heatmap at a threshold."""
    y_true_arr = _as_numpy_int(y_true)
    y_score_arr = _as_numpy_float(y_score)
    y_pred_arr = (y_score_arr >= threshold).astype(int)
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])

    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=["Pred: Benign", "Pred: Attack"],
            y=["True: Benign", "True: Attack"],
            text=cm,
            texttemplate="%{text}",
            colorscale="Blues",
            showscale=True,
        )
    )
    fig.update_layout(
        title=f"Confusion Matrix @ threshold={threshold:.2f}",
        template="plotly_white",
    )
    return fig



def score_distribution_figure(y_true: Iterable[int], y_score: Iterable[float]) -> go.Figure:
    """Create class-conditional score distribution histogram."""
    y_true_arr = _as_numpy_int(y_true)
    y_score_arr = _as_numpy_float(y_score)

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=y_score_arr[y_true_arr == 0],
            opacity=0.65,
            name="Benign",
            nbinsx=40,
            marker_color="#2ca02c",
        )
    )
    fig.add_trace(
        go.Histogram(
            x=y_score_arr[y_true_arr == 1],
            opacity=0.65,
            name="Attack",
            nbinsx=40,
            marker_color="#d62728",
        )
    )
    fig.update_layout(
        barmode="overlay",
        title="Risk Score Distribution",
        xaxis_title="Risk score",
        yaxis_title="Count",
        template="plotly_white",
    )
    return fig



def cohort_analysis_figure(
    df: pd.DataFrame,
    score_col: str = "risk_score",
    target_col: str = "is_attack",
) -> go.Figure:
    """Create new-vs-established cohort panel."""
    if score_col not in df.columns:
        raise ValueError(f"Missing score column: {score_col}")
    if target_col not in df.columns:
        raise ValueError(f"Missing target column: {target_col}")
    if "account_age_days" not in df.columns:
        raise ValueError("Missing required column: account_age_days")

    work = df.copy()
    work["cohort"] = np.where(work["account_age_days"] < 7, "new(<7d)", "established")

    grouped = (
        work.groupby("cohort", as_index=False)
        .agg(
            avg_score=(score_col, "mean"),
            attack_rate=(target_col, "mean"),
            sessions=(target_col, "size"),
        )
        .sort_values("cohort")
    )

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(x=grouped["cohort"], y=grouped["avg_score"], name="Avg risk score"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=grouped["cohort"],
            y=grouped["attack_rate"],
            mode="lines+markers",
            name="Attack rate",
            line={"width": 3},
        ),
        secondary_y=True,
    )
    fig.update_layout(title="Cohort Analysis: New vs Established", template="plotly_white")
    fig.update_yaxes(title_text="Average risk score", secondary_y=False)
    fig.update_yaxes(title_text="Attack rate", secondary_y=True)
    return fig



def threshold_metrics_table(
    y_true: Iterable[int],
    y_score: Iterable[float],
    thresholds: Iterable[float] | None = None,
) -> pd.DataFrame:
    """Return precision/recall metrics across thresholds."""
    y_true_arr = _as_numpy_int(y_true)
    y_score_arr = _as_numpy_float(y_score)

    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)

    rows = []
    for t in thresholds:
        pred = (y_score_arr >= float(t)).astype(int)
        precision = float(precision_score(y_true_arr, pred, zero_division=0))
        recall = float(recall_score(y_true_arr, pred, zero_division=0))
        tn, fp, fn, tp = confusion_matrix(y_true_arr, pred, labels=[0, 1]).ravel()
        rows.append(
            {
                "threshold": float(t),
                "precision": precision,
                "recall": recall,
                "tp": int(tp),
                "fp": int(fp),
                "tn": int(tn),
                "fn": int(fn),
            }
        )
    return pd.DataFrame(rows)



def threshold_slider_figure(y_true: Iterable[int], y_score: Iterable[float]) -> go.Figure:
    """Create an animated threshold slider backed by confusion matrix frames."""
    y_true_arr = _as_numpy_int(y_true)
    y_score_arr = _as_numpy_float(y_score)
    metrics_df = threshold_metrics_table(y_true_arr, y_score_arr, np.linspace(0.05, 0.95, 19))

    frames = []
    for _, row in metrics_df.iterrows():
        t = float(row["threshold"])
        pred = (y_score_arr >= t).astype(int)
        cm = confusion_matrix(y_true_arr, pred, labels=[0, 1])
        frames.append(
            go.Frame(
                name=f"{t:.2f}",
                data=[
                    go.Heatmap(
                        z=cm,
                        x=["Pred: Benign", "Pred: Attack"],
                        y=["True: Benign", "True: Attack"],
                        text=cm,
                        texttemplate="%{text}",
                        colorscale="Blues",
                        showscale=True,
                    )
                ],
                layout=go.Layout(
                    title=(
                        "Threshold Sweep (Confusion Matrix)"
                        f"<br><sup>threshold={t:.2f}, precision={row['precision']:.3f}, "
                        f"recall={row['recall']:.3f}</sup>"
                    )
                ),
            )
        )

    initial = frames[0].data[0]
    fig = go.Figure(data=[initial], frames=frames)

    steps = [
        {
            "label": frame.name,
            "method": "animate",
            "args": [[frame.name], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}],
        }
        for frame in frames
    ]

    fig.update_layout(
        title="Threshold Sweep (Confusion Matrix)",
        sliders=[{"active": 0, "steps": steps}],
        template="plotly_white",
    )
    return fig



def build_metrics_dashboard(
    df: pd.DataFrame,
    *,
    score_col: str = "risk_score",
    target_col: str = "is_attack",
    feature_importances: Mapping[str, float] | pd.DataFrame | None = None,
) -> dict[str, go.Figure]:
    """Return named figures for use in notebooks/apps."""
    if score_col not in df.columns:
        raise ValueError(f"Missing score column: {score_col}")
    if target_col not in df.columns:
        raise ValueError(f"Missing target column: {target_col}")

    y_true = df[target_col].astype(int).to_numpy()
    y_score = df[score_col].astype(float).to_numpy()

    if feature_importances is None:
        feature_importances = {}

    return {
        "roc": roc_curve_figure(y_true, y_score),
        "pr": pr_curve_figure(y_true, y_score),
        "feature_importance": feature_importance_figure(feature_importances),
        "confusion_matrix": confusion_matrix_figure(y_true, y_score),
        "score_distribution": score_distribution_figure(y_true, y_score),
        "cohort_analysis": cohort_analysis_figure(df, score_col=score_col, target_col=target_col),
        "threshold_slider": threshold_slider_figure(y_true, y_score),
    }


__all__ = [
    "build_metrics_dashboard",
    "cohort_analysis_figure",
    "confusion_matrix_figure",
    "feature_importance_figure",
    "pr_curve_figure",
    "roc_curve_figure",
    "score_distribution_figure",
    "threshold_metrics_table",
    "threshold_slider_figure",
]
