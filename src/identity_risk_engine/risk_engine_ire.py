"""Shared scoring pipeline for CLI and FastAPI demo."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Optional

import numpy as np
import pandas as pd

from .explainer_ire import build_explanation
from .policy_engine import PolicyEngine
from .signals import (
    evaluate_behavior_signals,
    evaluate_device_signals,
    evaluate_geo_signals,
    evaluate_passkey_signals,
    evaluate_recovery_signals,
)

SIGNAL_FNS = [
    evaluate_device_signals,
    evaluate_geo_signals,
    evaluate_behavior_signals,
    evaluate_passkey_signals,
    evaluate_recovery_signals,
]

FAST_SIGNAL_FNS = [
    evaluate_device_signals,
    evaluate_geo_signals,
]

SignalFunction = Callable[..., list[dict[str, Any]]]


def evaluate_signals_for_event(
    event: dict[str, Any],
    history_df: pd.DataFrame,
    signal_fns: Optional[list[SignalFunction]] = None,
) -> list[dict[str, Any]]:
    user_id = str(event.get("user_id") or "")
    if history_df.empty or "user_id" not in history_df.columns:
        user_history = history_df.iloc[0:0]
    else:
        user_history = history_df[history_df["user_id"].astype(str) == user_id]
    active_signal_fns = signal_fns if signal_fns is not None else SIGNAL_FNS
    outputs: list[dict[str, Any]] = []
    for fn in active_signal_fns:
        outputs.extend(fn(event=event, user_history=user_history, global_history=history_df))
    return outputs


def aggregate_risk_score(signal_results: list[dict[str, Any]]) -> float:
    if not signal_results:
        return 0.0

    fired_scores = [float(s.get("score", 0.0)) for s in signal_results if s.get("fired")]
    if not fired_scores:
        return 0.0

    no_risk_prob = float(np.prod([1.0 - max(0.0, min(1.0, x)) for x in fired_scores]))
    return float(max(0.0, min(1.0, 1.0 - no_risk_prob)))


def score_event(
    *,
    event: dict[str, Any],
    history_df: pd.DataFrame,
    policy_engine: PolicyEngine,
    dry_run: bool = False,
    signal_fns: Optional[list[SignalFunction]] = None,
    include_explanation: bool = True,
) -> dict[str, Any]:
    signal_results = evaluate_signals_for_event(
        event=event,
        history_df=history_df,
        signal_fns=signal_fns,
    )
    risk_score = aggregate_risk_score(signal_results)

    fired = [s for s in signal_results if s.get("fired")]
    reasons = [str(s.get("signal_name")) for s in fired]
    evidence = [str(s.get("evidence")) for s in fired]

    decision = policy_engine.decide(
        risk_score,
        reasons=reasons,
        evidence=evidence,
        auth_method=str(event.get("auth_method") or "password"),
        tenant_id=str(event.get("tenant_id") or "default"),
        dry_run=dry_run,
    )

    explanation = {}
    if include_explanation:
        explanation = build_explanation(
            event=event,
            signal_results=signal_results,
            risk_score=risk_score,
            user_history=(
                history_df.iloc[0:0]
                if history_df.empty or "user_id" not in history_df.columns
                else history_df[
                    history_df["user_id"].astype(str) == str(event.get("user_id") or "")
                ]
            ),
        )

    return {
        "event": event,
        "risk_score": risk_score,
        "decision": decision,
        "signals": signal_results,
        "explanation": explanation,
    }


def score_dataframe(
    df: pd.DataFrame,
    policy_engine: PolicyEngine,
    dry_run: bool = False,
    history_window: int = 200,
    signal_fns: Optional[list[SignalFunction]] = None,
    include_explanations: bool = True,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    ordered = df.copy()
    ordered["timestamp"] = pd.to_datetime(ordered["timestamp"], utc=True, errors="coerce")
    ordered = ordered.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

    for idx, row in ordered.iterrows():
        event = row.to_dict()
        start = max(0, idx - int(history_window))
        history = ordered.iloc[start:idx]
        result = score_event(
            event=event,
            history_df=history,
            policy_engine=policy_engine,
            dry_run=dry_run,
            signal_fns=signal_fns,
            include_explanation=include_explanations,
        )

        scored = event.copy()
        scored["risk_score"] = result["risk_score"]
        scored["action"] = result["decision"]["action"]
        scored["reasons"] = "|".join(result["decision"]["reasons"]) if result["decision"]["reasons"] else ""
        scored["evidence"] = "|".join(result["decision"]["evidence"]) if result["decision"]["evidence"] else ""
        scored["human_summary"] = (
            str(result["explanation"]["human_summary"])
            if include_explanations and result.get("explanation")
            else ""
        )
        rows.append(scored)

    return pd.DataFrame(rows)
