"""Explainability helper for scored auth events."""

from __future__ import annotations

import re
from typing import Any, Optional

import pandas as pd


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _baseline_comparison(
    event: dict[str, Any],
    user_history: Optional[pd.DataFrame],
    risk_score: float,
) -> dict[str, Any]:
    if user_history is None or user_history.empty:
        return {
            "user_avg_risk": None,
            "deviation": None,
            "hour_is_unusual": None,
            "message": "No historical baseline available.",
        }

    baseline: dict[str, Any] = {
        "user_avg_risk": None,
        "deviation": None,
        "hour_is_unusual": None,
        "message": "Baseline available.",
    }

    if "risk_score" in user_history.columns:
        avg = float(pd.to_numeric(user_history["risk_score"], errors="coerce").fillna(0.0).mean())
        baseline["user_avg_risk"] = avg
        baseline["deviation"] = float(risk_score - avg)

    if "timestamp" in user_history.columns and event.get("timestamp") is not None:
        hist_ts = pd.to_datetime(user_history["timestamp"], utc=True, errors="coerce").dropna()
        cur_ts = pd.to_datetime(event.get("timestamp"), utc=True, errors="coerce")
        if not hist_ts.empty and pd.notna(cur_ts):
            hour_dist = hist_ts.dt.hour.value_counts(normalize=True)
            hour_prob = float(hour_dist.get(int(cur_ts.hour), 0.0))
            baseline["hour_is_unusual"] = bool(hour_prob < 0.05)

    deviation = baseline.get("deviation")
    if deviation is not None:
        baseline["message"] = f"Current risk is {deviation:+.2f} vs user baseline."
    else:
        baseline["message"] = "Risk baseline unavailable; behavioral baseline only."

    return baseline


def build_explanation(
    *,
    event: dict[str, Any],
    signal_results: list[dict[str, Any]],
    risk_score: float,
    user_history: Optional[pd.DataFrame] = None,
) -> dict[str, Any]:
    """Create human + machine-friendly explanations for a scored event."""

    ordered = sorted(signal_results, key=lambda x: _safe_float(x.get("score", 0.0)), reverse=True)
    fired_first = [s for s in ordered if bool(s.get("fired"))]
    top = (fired_first if fired_first else ordered)[:3]

    top_reasons = [
        {
            "signal_name": s.get("signal_name"),
            "score": _safe_float(s.get("score", 0.0)),
            "evidence": str(s.get("evidence", "")),
        }
        for s in top
    ]

    feature_values = {str(s.get("signal_name")): _safe_float(s.get("score", 0.0)) for s in ordered}

    signal_map = {str(s.get("signal_name")): s for s in ordered}
    impossible_score = _safe_float(feature_values.get("impossible_travel_composite", 0.0))
    impossible_fired = bool(signal_map.get("impossible_travel", {}).get("fired")) or impossible_score > 0.0

    if impossible_fired:
        speed_txt = "high velocity"
        impossible_ev = str(signal_map.get("impossible_travel", {}).get("evidence", ""))
        speed_match = re.search(r"max_speed_kmh=([0-9.]+)", impossible_ev)
        if speed_match:
            speed_txt = f"{float(speed_match.group(1)):.0f} km/h"

        detail_parts: list[str] = []
        mismatch_sig = signal_map.get("device_location_mismatch", {})
        if mismatch_sig.get("fired"):
            detail_parts.append(str(mismatch_sig.get("evidence") or "device-location mismatch"))
        user_country_sig = signal_map.get("new_country_for_user", {})
        if user_country_sig.get("fired"):
            detail_parts.append(str(user_country_sig.get("evidence") or "new country for user"))
        session_break_sig = signal_map.get("geo_session_break", {})
        if session_break_sig.get("fired"):
            detail_parts.append(str(session_break_sig.get("evidence") or "geo-session continuity break"))
        if not detail_parts and impossible_ev:
            detail_parts.append(impossible_ev)

        human_summary = f"Impossible travel detected: {speed_txt}, " + ", ".join(detail_parts) + "."
    elif top:
        reason_text = "; ".join([f"{s.get('signal_name')} ({s.get('score', 0):.2f})" for s in top])
        human_summary = f"Risk score {risk_score:.2f}: {reason_text}."
    else:
        human_summary = f"Risk score {risk_score:.2f}: no high-risk signals fired."

    baseline_comparison = _baseline_comparison(event=event, user_history=user_history, risk_score=float(risk_score))

    machine_json = {
        "event_id": event.get("event_id"),
        "event_type": event.get("event_type"),
        "risk_score": float(risk_score),
        "signals": ordered,
        "top_reasons": top_reasons,
        "feature_values": feature_values,
        "baseline_comparison": baseline_comparison,
    }

    return {
        "top_reasons": top_reasons,
        "feature_values": feature_values,
        "human_summary": human_summary,
        "machine_json": machine_json,
        "baseline_comparison": baseline_comparison,
    }


def explain_scored_event(
    event: dict[str, Any],
    signal_results: list[dict[str, Any]],
    risk_score: float,
    user_history: Optional[pd.DataFrame] = None,
) -> dict[str, Any]:
    """Alias for external callers expecting explain_scored_event()."""

    return build_explanation(
        event=event,
        signal_results=signal_results,
        risk_score=risk_score,
        user_history=user_history,
    )
