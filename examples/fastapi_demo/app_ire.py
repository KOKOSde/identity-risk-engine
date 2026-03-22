"""FastAPI demo app for identity-risk-engine v0.2.0."""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


def _ensure_src_on_path() -> None:
    root = Path(__file__).resolve().parents[2]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


_ensure_src_on_path()

from identity_risk_engine.events import AuthEvent, event_to_row  # noqa: E402
from identity_risk_engine.policy_engine import PolicyEngine  # noqa: E402
from identity_risk_engine.risk_engine_ire import score_dataframe, score_event  # noqa: E402
from identity_risk_engine.simulator_ire import generate_synthetic_auth_events  # noqa: E402

APP_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POLICY_PATH = APP_ROOT / "configs" / "default_policy.yaml"

POLICY_ENGINE = PolicyEngine(config=DEFAULT_POLICY_PATH if DEFAULT_POLICY_PATH.exists() else None)
EVENT_HISTORY = pd.DataFrame()
SCORED_HISTORY = pd.DataFrame()
HISTORY_LOCK = threading.Lock()

app = FastAPI(title="identity-risk-engine FastAPI Demo", version="0.2.0")


class EventIngestRequest(BaseModel):
    event: AuthEvent
    dry_run: bool = False


class SimulateRequest(BaseModel):
    num_users: int = Field(default=10, ge=1)
    num_sessions: int = Field(default=100, ge=1)
    attack_ratio: float = Field(default=0.2, ge=0.0, le=1.0)
    seed: int = Field(default=42)
    passkey_adoption_rate: float = Field(default=0.35, ge=0.0, le=1.0)
    recovery_flow_rate: float = Field(default=0.08, ge=0.0, le=1.0)
    dry_run: bool = False
    sample_size: int = Field(default=20, ge=1, le=200)


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _json_safe_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe_value(v) for v in value]
    if pd.isna(value) if not isinstance(value, (dict, list, str, bytes)) else False:
        return None
    return value


def _coerce_metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value is None:
        return {}
    text = str(value).strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


@app.get("/health")
def health() -> dict[str, Any]:
    with HISTORY_LOCK:
        event_count = int(len(EVENT_HISTORY))
        scored_count = int(len(SCORED_HISTORY))
    return {
        "status": "ok",
        "service": "identity-risk-engine",
        "version": "0.2.0",
        "events_seen": event_count,
        "events_scored": scored_count,
    }


@app.post("/events")
def ingest_event(payload: EventIngestRequest) -> dict[str, Any]:
    global EVENT_HISTORY, SCORED_HISTORY

    event_row = event_to_row(payload.event)
    if "event_type" not in event_row:
        raise HTTPException(status_code=400, detail="event_type is required")

    with HISTORY_LOCK:
        history = EVENT_HISTORY.copy()
        result = score_event(event=event_row, history_df=history, policy_engine=POLICY_ENGINE, dry_run=payload.dry_run)

        EVENT_HISTORY = pd.concat([EVENT_HISTORY, pd.DataFrame([event_row])], ignore_index=True)
        scored_row = event_row.copy()
        scored_row["risk_score"] = result["risk_score"]
        scored_row["action"] = result["decision"]["action"]
        scored_row["reasons"] = result["decision"].get("reasons", [])
        scored_row["evidence"] = result["decision"].get("evidence", [])
        scored_row["human_summary"] = result["explanation"]["human_summary"]
        SCORED_HISTORY = pd.concat([SCORED_HISTORY, pd.DataFrame([scored_row])], ignore_index=True)

    return {
        "event_id": event_row.get("event_id"),
        "risk_score": float(result["risk_score"]),
        "action": result["decision"]["action"],
        "decision": _json_safe_value(result["decision"]),
        "explanation": _json_safe_value(result["explanation"]),
        "signals": _json_safe_value(result["signals"]),
    }


@app.post("/simulate")
def simulate_events(payload: SimulateRequest) -> dict[str, Any]:
    global EVENT_HISTORY, SCORED_HISTORY

    df = generate_synthetic_auth_events(
        num_users=payload.num_users,
        num_sessions=payload.num_sessions,
        attack_ratio=payload.attack_ratio,
        seed=payload.seed,
        passkey_adoption_rate=payload.passkey_adoption_rate,
        recovery_flow_rate=payload.recovery_flow_rate,
    )
    if "metadata" in df.columns:
        df["metadata"] = df["metadata"].apply(_coerce_metadata)

    scored = score_dataframe(df, policy_engine=POLICY_ENGINE, dry_run=payload.dry_run)

    with HISTORY_LOCK:
        EVENT_HISTORY = pd.concat([EVENT_HISTORY, df], ignore_index=True)
        SCORED_HISTORY = pd.concat([SCORED_HISTORY, scored], ignore_index=True)

    sample = scored.head(payload.sample_size).copy()
    sample = sample.where(pd.notnull(sample), None)
    sample_records = [{k: _json_safe_value(v) for k, v in row.items()} for row in sample.to_dict(orient="records")]

    action_counts = (
        scored["action"].fillna("unknown").astype(str).value_counts().to_dict() if "action" in scored.columns else {}
    )
    attack_counts = (
        scored["attack_type"].fillna("normal").astype(str).value_counts().to_dict()
        if "attack_type" in scored.columns
        else {}
    )

    return {
        "generated_events": int(len(df)),
        "scored_events": int(len(scored)),
        "mean_risk_score": float(scored["risk_score"].mean()) if "risk_score" in scored.columns and not scored.empty else 0.0,
        "action_counts": action_counts,
        "attack_counts": attack_counts,
        "sample": sample_records,
    }


@app.get("/dashboard-data")
def dashboard_data() -> dict[str, Any]:
    with HISTORY_LOCK:
        scored = SCORED_HISTORY.copy()

    if scored.empty:
        return {
            "total_events": 0,
            "risk_distribution": [],
            "action_counts": {},
            "attack_counts": {},
            "recent_high_risk": [],
        }

    scored["risk_score"] = pd.to_numeric(scored.get("risk_score"), errors="coerce").fillna(0.0)
    bins = [-0.0001, 0.2, 0.4, 0.6, 0.8, 1.0]
    labels = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
    dist = pd.cut(scored["risk_score"], bins=bins, labels=labels).value_counts().reindex(labels, fill_value=0)
    risk_distribution = [{"bucket": str(bucket), "count": int(count)} for bucket, count in dist.items()]

    action_counts = scored["action"].fillna("unknown").astype(str).value_counts().to_dict()
    attack_counts: dict[str, int] = {}
    if "attack_type" in scored.columns:
        attack_counts = scored["attack_type"].fillna("normal").astype(str).value_counts().to_dict()

    recent = scored.sort_values("risk_score", ascending=False).head(20)
    recent_cols = [c for c in ["event_id", "event_type", "user_id", "risk_score", "action", "human_summary"] if c in recent.columns]
    recent = recent[recent_cols].where(pd.notnull(recent), None)
    recent_high_risk = [{k: _json_safe_value(v) for k, v in row.items()} for row in recent.to_dict(orient="records")]

    return {
        "total_events": int(len(scored)),
        "risk_distribution": risk_distribution,
        "action_counts": action_counts,
        "attack_counts": attack_counts,
        "recent_high_risk": recent_high_risk,
    }
