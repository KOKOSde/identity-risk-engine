"""CLI for identity-risk-engine v0.2.0."""

from __future__ import annotations

import argparse
import ast
import html
import json
from pathlib import Path
from typing import Any

import pandas as pd

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


def _read_events_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError("events CSV must include a timestamp column")
    if "metadata" in df.columns:
        df["metadata"] = df["metadata"].apply(_parse_metadata_cell)
    else:
        df["metadata"] = [{} for _ in range(len(df))]
    return df


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

    scored = score_dataframe(events_df, policy_engine=policy, dry_run=bool(args.dry_run))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(out_path, index=False)

    print(f"Scored {len(scored)} events -> {out_path}")
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
    score.add_argument("--out", required=True)
    score.set_defaults(func=_cmd_score)

    report = sub.add_parser("report", help="Generate an HTML report from scored CSV")
    report.add_argument("--events", required=True)
    report.add_argument("--out", required=True)
    report.set_defaults(func=_cmd_report)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
