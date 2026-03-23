"""CLI for identity-risk-engine v0.2.0."""

from __future__ import annotations

import argparse
import ast
import html
import json
import time
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd

from .geo_velocity import compute_geo_velocity_features
from .policy_engine import PolicyEngine
from .risk_engine_ire import score_dataframe
from .simulator_ire import generate_synthetic_auth_events


def _parse_metadata_cell(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return {}
    text = str(value).strip()
    if not text:
        return {}

    # Try JSON first, then Python-literal dict.
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, dict):
            return parsed
    except (ValueError, SyntaxError):
        pass
    return {}


def _read_events_csv(path: Union[str, Path]) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError("events CSV must include a timestamp column")
    if "metadata" in df.columns:
        df["metadata"] = df["metadata"].apply(_parse_metadata_cell)
    else:
        df["metadata"] = [{} for _ in range(len(df))]
    return df


_ATTACK_HINT_KEYS = (
    "credential_stuffing",
    "account_takeover",
    "bot_behavior",
    "impossible_travel",
    "new_account_fraud",
    "session_hijack",
    "mfa_fatigue",
    "recovery_abuse",
    "passkey_registration_abuse",
    "multi_account_sybil",
)


def _metadata_hint_score(value: Any) -> float:
    if not isinstance(value, dict):
        return 0.0
    hits = sum(1 for key in _ATTACK_HINT_KEYS if bool(value.get(key)))
    if hits <= 0:
        return 0.0
    return float(min(1.0, 0.55 + 0.05 * hits))


def _fast_score_dataframe(
    events_df: pd.DataFrame,
    *,
    policy_engine: PolicyEngine,
    dry_run: bool,
) -> pd.DataFrame:
    work = events_df.copy()
    for col, default in (
        ("user_id", ""),
        ("device_hash", ""),
        ("country", ""),
        ("ip", ""),
        ("auth_method", "password"),
        ("tenant_id", "default"),
    ):
        if col not in work.columns:
            work[col] = default
        work[col] = work[col].fillna(default).astype(str)

    if "metadata" not in work.columns:
        work["metadata"] = [{} for _ in range(len(work))]

    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
    work = work.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

    user_pos = work.groupby("user_id", sort=False).cumcount()
    device_pos = work.groupby(["user_id", "device_hash"], sort=False).cumcount()
    new_device = (work["device_hash"] != "") & (device_pos == 0) & (user_pos > 0)

    prev_country = work.groupby("user_id", sort=False)["country"].shift(1).fillna("")
    new_country = (work["country"] != "") & (prev_country != "") & (work["country"] != prev_country)

    ip_lower = work["ip"].str.lower()
    datacenter_ip = (
        ip_lower.str.startswith(("34.", "35.", "52.", "54."))
        | ip_lower.str.contains("tor|vpn|proxy", regex=True)
    )

    geo_in = pd.DataFrame(
        {
            "user_id": work["user_id"],
            "timestamp": work["timestamp"],
            "lat": pd.to_numeric(work.get("lat_coarse"), errors="coerce"),
            "lon": pd.to_numeric(work.get("lon_coarse"), errors="coerce"),
        }
    )
    geo_out = compute_geo_velocity_features(geo_in)
    geo_velocity_score = pd.to_numeric(geo_out["geo_velocity_score"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    impossible_travel = geo_out["impossible_travel"].fillna(False).astype(bool)

    metadata_hint = work["metadata"].apply(_metadata_hint_score)

    risk_score = (
        0.55 * impossible_travel.astype(float)
        + 0.25 * geo_velocity_score
        + 0.20 * new_device.astype(float)
        + 0.15 * new_country.astype(float)
        + 0.15 * datacenter_ip.astype(float)
        + 0.40 * metadata_hint
    ).clip(0.0, 1.0)

    actions: list[str] = []
    reasons_col: list[str] = []
    evidence_col: list[str] = []
    for idx in range(len(work)):
        reasons: list[str] = []
        evidence: list[str] = []
        if bool(impossible_travel.iloc[idx]):
            reasons.append("impossible_travel")
            evidence.append("Geo speed exceeds plausible travel speed")
        if float(geo_velocity_score.iloc[idx]) >= 0.6:
            reasons.append("geo_velocity")
            evidence.append(f"geo_velocity_score={float(geo_velocity_score.iloc[idx]):.3f}")
        if bool(new_device.iloc[idx]):
            reasons.append("new_device")
            evidence.append("First observed device for this user")
        if bool(new_country.iloc[idx]):
            reasons.append("new_country")
            evidence.append("Country changed relative to recent user activity")
        if bool(datacenter_ip.iloc[idx]):
            reasons.append("datacenter_ip")
            evidence.append("Datacenter/proxy-like IP pattern")
        if float(metadata_hint.iloc[idx]) > 0.0:
            reasons.append("metadata_attack_hints")
            evidence.append("Attack-hint metadata keys present")

        decision = policy_engine.decide(
            float(risk_score.iloc[idx]),
            reasons=reasons,
            evidence=evidence,
            auth_method=str(work.iloc[idx]["auth_method"] or "password"),
            tenant_id=str(work.iloc[idx]["tenant_id"] or "default"),
            dry_run=dry_run,
        )
        actions.append(str(decision["action"]))
        reasons_col.append("|".join(decision["reasons"]) if decision["reasons"] else "")
        evidence_col.append("|".join(decision["evidence"]) if decision["evidence"] else "")

    out = work.copy()
    out["risk_score"] = risk_score.to_numpy()
    out["action"] = actions
    out["reasons"] = reasons_col
    out["evidence"] = evidence_col
    out["human_summary"] = ""
    return out


def _cmd_simulate(args: argparse.Namespace) -> int:
    df = generate_synthetic_auth_events(
        num_users=int(args.users),
        num_sessions=int(args.sessions),
        attack_ratio=float(args.attack_ratio),
        seed=int(args.seed),
        passkey_adoption_rate=float(args.passkey_adoption_rate),
        recovery_flow_rate=float(args.recovery_flow_rate),
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    attack_counts = df["attack_type"].fillna("normal").value_counts().to_dict() if "attack_type" in df.columns else {}
    print(f"Generated {len(df)} events -> {out_path}")
    if attack_counts:
        print("Attack mix:")
        for attack, count in sorted(attack_counts.items(), key=lambda kv: str(kv[0])):
            print(f"  {attack}: {count}")
    return 0


def _cmd_score(args: argparse.Namespace) -> int:
    events_df = _read_events_csv(args.events)
    policy = PolicyEngine(config=args.policy) if args.policy else PolicyEngine()

    if bool(args.fast) and bool(args.full):
        raise ValueError("Use only one of --fast or --full.")

    history_window = int(args.history_window)
    mode = "full"
    auto_fast_selected = False
    auto_fast = (
        bool(args.fast)
        or (not bool(args.full) and len(events_df) > int(args.auto_fast_threshold))
    )
    scored: pd.DataFrame
    if auto_fast:
        mode = "fast" if bool(args.fast) else "fast-auto"
        auto_fast_selected = (not bool(args.fast) and not bool(args.full))
        history_window = min(history_window, 8)

    if auto_fast_selected:
        print(
            f"Auto-selecting fast mode for {len(events_df)} events "
            "(use --full for complete signal extraction)"
        )

    started = time.perf_counter()
    if auto_fast:
        scored = _fast_score_dataframe(
            events_df,
            policy_engine=policy,
            dry_run=bool(args.dry_run),
        )
    else:
        scored = score_dataframe(
            events_df,
            policy_engine=policy,
            dry_run=bool(args.dry_run),
            history_window=history_window,
            include_explanations=True,
        )
    elapsed = time.perf_counter() - started
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(out_path, index=False)

    print(f"Scored {len(scored)} events -> {out_path}")
    print(f"Scoring mode: {mode} (history_window={history_window})")
    print(f"Elapsed seconds: {elapsed:.2f}")
    if not scored.empty and "risk_score" in scored.columns:
        print(f"Mean risk score: {float(scored['risk_score'].mean()):.4f}")
    if "action" in scored.columns:
        action_counts = scored["action"].value_counts().to_dict()
        print("Action counts:")
        for action, count in sorted(action_counts.items(), key=lambda kv: str(kv[0])):
            print(f"  {action}: {count}")
    return 0


def _to_html_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "<p>No data.</p>"
    cols = list(df.columns)
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in cols)
    rows = []
    for _, row in df.iterrows():
        cells = "".join(f"<td>{html.escape(str(row[c]))}</td>" for c in cols)
        rows.append(f"<tr>{cells}</tr>")
    body = "".join(rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _cmd_report(args: argparse.Namespace) -> int:
    df = pd.read_csv(args.events)
    if "risk_score" in df.columns:
        df["risk_score"] = pd.to_numeric(df["risk_score"], errors="coerce").fillna(0.0)

    summary_rows = []
    summary_rows.append({"metric": "total_events", "value": len(df)})
    if "risk_score" in df.columns and not df.empty:
        summary_rows.extend(
            [
                {"metric": "avg_risk_score", "value": round(float(df["risk_score"].mean()), 6)},
                {"metric": "p95_risk_score", "value": round(float(df["risk_score"].quantile(0.95)), 6)},
            ]
        )
    if "label" in df.columns:
        label_series = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
        summary_rows.append({"metric": "positive_rate", "value": round(float(label_series.mean()), 6)})

    action_df = pd.DataFrame()
    if "action" in df.columns:
        action_df = (
            df["action"]
            .fillna("unknown")
            .astype(str)
            .value_counts()
            .rename_axis("action")
            .reset_index(name="count")
        )

    attack_df = pd.DataFrame()
    if "attack_type" in df.columns:
        attack_df = (
            df["attack_type"]
            .fillna("normal")
            .astype(str)
            .value_counts()
            .rename_axis("attack_type")
            .reset_index(name="count")
        )

    summary_df = pd.DataFrame(summary_rows)
    preview_cols = [c for c in ["event_id", "event_type", "user_id", "risk_score", "action", "human_summary"] if c in df.columns]
    preview_df = df[preview_cols].head(25) if preview_cols else pd.DataFrame()

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>identity-risk-engine report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif; margin: 24px; }}
    h1, h2 {{ margin: 0.2em 0; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 20px 0; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; font-size: 13px; }}
    th {{ background: #f5f5f5; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
  </style>
</head>
<body>
  <h1>identity-risk-engine Risk Report</h1>
  <h2>Summary</h2>
  {_to_html_table(summary_df)}
  <div class="grid">
    <section>
      <h2>Actions</h2>
      {_to_html_table(action_df)}
    </section>
    <section>
      <h2>Attack Types</h2>
      {_to_html_table(attack_df)}
    </section>
  </div>
  <h2>Event Preview</h2>
  {_to_html_table(preview_df)}
</body>
</html>
"""

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_doc, encoding="utf-8")

    print("Report summary:")
    for row in summary_rows:
        print(f"  {row['metric']}: {row['value']}")
    if not action_df.empty:
        print("Top actions:")
        for _, row in action_df.head(6).iterrows():
            print(f"  {row['action']}: {int(row['count'])}")
    if not attack_df.empty:
        print("Top attack types:")
        for _, row in attack_df.head(6).iterrows():
            print(f"  {row['attack_type']}: {int(row['count'])}")
    print(f"Report written -> {out_path}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="identity-risk-engine",
        description="Identity Risk Engine CLI (simulate, score, report)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    simulate = sub.add_parser("simulate", help="Generate synthetic auth events")
    simulate.add_argument("--users", type=int, default=500)
    simulate.add_argument("--sessions", type=int, default=20000)
    simulate.add_argument("--attack-ratio", type=float, default=0.2)
    simulate.add_argument("--seed", type=int, default=42)
    simulate.add_argument("--passkey-adoption-rate", type=float, default=0.35)
    simulate.add_argument("--recovery-flow-rate", type=float, default=0.08)
    simulate.add_argument("--out", required=True)
    simulate.set_defaults(func=_cmd_simulate)

    score = sub.add_parser("score", help="Score events from a CSV file")
    score.add_argument("--events", required=True)
    score.add_argument("--policy", default="configs/default_policy.yaml")
    score.add_argument("--dry-run", action="store_true")
    score.add_argument(
        "--fast",
        action="store_true",
        help=(
            "Fast demo mode: reduced signal set + no explanation generation. "
            "Recommended for large demos (e.g., ~4k events in under a minute on a laptop)."
        ),
    )
    score.add_argument(
        "--full",
        action="store_true",
        help="Force full scoring mode (overrides auto-fast behavior).",
    )
    score.add_argument(
        "--auto-fast-threshold",
        type=int,
        default=5000,
        help="Automatically switch to fast mode when event count exceeds this threshold (default: 5000).",
    )
    score.add_argument(
        "--history-window",
        type=int,
        default=200,
        help="Number of prior events to use as context per event (default: 200; fast mode caps at 8).",
    )
    score.add_argument("--out", required=True)
    score.set_defaults(func=_cmd_score)

    report = sub.add_parser("report", help="Generate an HTML report from scored CSV")
    report.add_argument("--events", required=True)
    report.add_argument("--out", required=True)
    report.set_defaults(func=_cmd_report)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
