#!/usr/bin/env python3
"""Generate reproducible benchmark metrics/table for README sync."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from identity_risk_engine.composite_scorer import CompositeRiskScorer  # noqa: E402
from identity_risk_engine.simulator_ire import generate_synthetic_auth_events  # noqa: E402


def _precision_at_recall(y_true: np.ndarray, scores: np.ndarray, target_recall: float = 0.95) -> float:
    best = 0.0
    for t in np.linspace(0.01, 0.99, 199):
        pred = (scores >= t).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        if recall >= target_recall and precision > best:
            best = precision
    return best


def _recall_at_precision(y_true: np.ndarray, scores: np.ndarray, target_precision: float = 0.95) -> float:
    best = 0.0
    for t in np.linspace(0.01, 0.99, 199):
        pred = (scores >= t).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        if precision >= target_precision and recall > best:
            best = recall
    return best


def _cohort_rows(
    events: pd.DataFrame,
    scores: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    y_global = events["label"].to_numpy(dtype=int)
    rows.append(
        {
            "cohort": "Global",
            "auc": float(roc_auc_score(y_global, scores)),
            "precision_at_95_recall": float(_precision_at_recall(y_global, scores, 0.95)),
            "recall_at_95_precision": float(_recall_at_precision(y_global, scores, 0.95)),
        }
    )

    attacks = sorted(events.loc[events["label"] == 1, "attack_type"].fillna("normal").astype(str).unique())
    for attack in attacks:
        mask = (events["attack_type"].astype(str) == attack).to_numpy()
        y_bin = mask.astype(int)
        if y_bin.sum() == 0:
            continue
        if y_bin.sum() == len(y_bin):
            auc = 1.0
        else:
            auc = float(roc_auc_score(y_bin, scores))

        rows.append(
            {
                "cohort": attack,
                "auc": auc,
                "precision_at_95_recall": float(_precision_at_recall(y_bin, scores, 0.95)),
                "recall_at_95_precision": float(_recall_at_precision(y_bin, scores, 0.95)),
            }
        )
    return rows


def _to_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Cohort | AUC | Precision@0.95Recall | Recall@0.95Precision |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {cohort} | {auc:.6f} | {p:.6f} | {r:.6f} |".format(
                cohort=row["cohort"],
                auc=float(row["auc"]),
                p=float(row["precision_at_95_recall"]),
                r=float(row["recall_at_95_precision"]),
            )
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate benchmark table for README.")
    parser.add_argument("--num-users", type=int, default=100)
    parser.add_argument("--num-sessions", type=int, default=4000)
    parser.add_argument("--attack-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-md", default="demo_outputs/benchmark_table_ire.md")
    parser.add_argument("--out-json", default="demo_outputs/benchmark_table_ire.json")
    args = parser.parse_args()

    events = generate_synthetic_auth_events(
        num_users=args.num_users,
        num_sessions=args.num_sessions,
        attack_ratio=args.attack_ratio,
        seed=args.seed,
    )
    model = CompositeRiskScorer(random_state=42)
    model.fit(events, target_col="label")
    scores = model.predict_proba(events)[:, 1]
    global_auc = float(roc_auc_score(events["label"].to_numpy(), scores))
    rows = _cohort_rows(events, scores)
    md = _to_markdown(rows)

    out_md = REPO_ROOT / args.out_md
    out_json = REPO_ROOT / args.out_json
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "config": {
            "num_users": args.num_users,
            "num_sessions": args.num_sessions,
            "attack_ratio": args.attack_ratio,
            "seed": args.seed,
            "model_random_state": 42,
        },
        "global_auc": global_auc,
        "rows": rows,
    }
    out_md.write_text(md + "\n", encoding="utf-8")
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Global AUROC: {global_auc:.6f}")
    print(f"Markdown table: {out_md}")
    print(f"JSON metrics: {out_json}")
    print()
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
