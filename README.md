# identity-risk-engine: Open-Source Session-Level Identity Risk Scoring for Fintech

![MIT License](https://img.shields.io/badge/license-MIT-green.svg)
![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![Tests Passing](https://img.shields.io/badge/tests-passing-brightgreen.svg)
![Pip Installable](https://img.shields.io/badge/pip-installable-orange.svg)

## Problem
Every fintech builds proprietary session risk scoring internally. This is the first open-source toolkit that packages impossible-travel detection, device fingerprinting, behavioral anomaly scoring, and composite ML classification into a single pip-installable library.

## Architecture
```text
login_events
    |
    v
+-------------------------------+
| feature extraction            |
| - geo_velocity                |
| - device_fingerprint          |
| - behavior_anomaly            |
+-------------------------------+
    |
    v
+-------------------------------+
| composite_scorer              |
| (XGBoost + LightGBM ensemble) |
+-------------------------------+
    |
    v
risk_score (0.0 - 1.0)
    |
    v
action = allow | challenge | block
```

## Quickstart
```bash
pip install identity-risk-engine
```

```python
from identity_risk_engine.synthetic_data_generator import generate_synthetic_login_data
from identity_risk_engine.composite_scorer import CompositeRiskScorer
df = generate_synthetic_login_data(num_users=100, num_sessions=4000, attack_ratio=0.2, seed=42)
model = CompositeRiskScorer().fit(df, target_col="label")
risk_scores = model.predict_proba(df)[:, 1]
```

## Benchmark Results
Synthetic benchmark run (`num_users=220`, `num_sessions=9000`, `attack_ratio=0.22`, `seed=101`, time-based split):

| Cohort | AUC | Precision@0.95Recall | Recall@0.95Precision |
|---|---:|---:|---:|
| Global | 1.000 | 1.000 | 1.000 |
| account_takeover | 1.000 | 1.000 | 1.000 |
| credential_stuffing | 1.000 | 1.000 | 1.000 |
| bot_behavior | 1.000 | 1.000 | 1.000 |
| impossible_travel | 1.000 | 1.000 | 1.000 |
| new_account_fraud | 1.000 | 1.000 | 1.000 |

## Features
- Impossible travel detection (haversine distance and velocity thresholding)
- Device clustering and novelty scoring (TF-IDF + DBSCAN + cosine distance)
- Behavioral baseline modeling (hour/frequency/duration/actions)
- Ensemble risk scoring (XGBoost + LightGBM + calibration)
- Synthetic data generator with multiple realistic attack types
- Cohort analysis notebooks and dashboard views
- Threshold tuning for block and friction operating modes

## Who This Is For
- Fintech identity and authentication teams
- Crypto exchanges
- Account security engineers
- Fraud analysts

## Use Cases
- Account takeover detection
- New account fraud detection
- Authentication friction optimization
- Sybil pre-screening

## No API Keys Needed
This project uses 100% synthetic data. No external APIs, no tokens, and no network dependency for core functionality. Everything runs locally.
