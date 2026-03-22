from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_cli_simulate_score_report_smoke(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    synthetic_path = tmp_path / "synthetic.csv"
    scored_path = tmp_path / "scored.csv"
    report_path = tmp_path / "report.html"

    cmd_sim = [
        sys.executable,
        "-m",
        "identity_risk_engine.cli_ire",
        "simulate",
        "--users",
        "10",
        "--sessions",
        "60",
        "--attack-ratio",
        "0.2",
        "--out",
        str(synthetic_path),
    ]
    res_sim = subprocess.run(cmd_sim, cwd=repo_root, capture_output=True, text=True, check=False)
    assert res_sim.returncode == 0, res_sim.stderr
    assert synthetic_path.exists()

    cmd_score = [
        sys.executable,
        "-m",
        "identity_risk_engine.cli_ire",
        "score",
        "--events",
        str(synthetic_path),
        "--policy",
        "configs/default_policy.yaml",
        "--out",
        str(scored_path),
    ]
    res_score = subprocess.run(cmd_score, cwd=repo_root, capture_output=True, text=True, check=False)
    assert res_score.returncode == 0, res_score.stderr
    assert scored_path.exists()

    scored_df = pd.read_csv(scored_path)
    assert "risk_score" in scored_df.columns
    assert "action" in scored_df.columns

    cmd_report = [
        sys.executable,
        "-m",
        "identity_risk_engine.cli_ire",
        "report",
        "--events",
        str(scored_path),
        "--out",
        str(report_path),
    ]
    res_report = subprocess.run(cmd_report, cwd=repo_root, capture_output=True, text=True, check=False)
    assert res_report.returncode == 0, res_report.stderr
    assert report_path.exists()
    assert "identity-risk-engine Risk Report" in report_path.read_text(encoding="utf-8")
