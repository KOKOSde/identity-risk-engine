# identity-risk-engine

Add suspicious-login detection, auth-risk scoring, and step-up decisions to your app.

![MIT License](https://img.shields.io/badge/license-MIT-green.svg)
![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![Tests Passing](https://img.shields.io/badge/tests-36%20passing-brightgreen.svg)
![pip installable](https://img.shields.io/badge/pip-installable-blue.svg)

## Why This Exists
Fintech and crypto teams must decide risk at authentication time, before transaction monitoring can help. Most teams rebuild this stack internally, and existing OSS tools usually cover only one slice (device, geo, or behavior). `identity-risk-engine` packages multi-signal auth risk scoring, policy decisions, and synthetic benchmarking into one local-first toolkit.

## Architecture
```text
Auth Events
  |
  v
+---------------------------+
| Signal Extraction         |
| device + geo + behavior   |
| passkey + recovery        |
+---------------------------+
  |
  v
+---------------------------+
| Risk Scoring              |
| signal fusion + ensemble  |
+---------------------------+
  |
  v
+---------------------------+
| Policy Engine             |
| tenant/method thresholds  |
+---------------------------+
  |
  v
Action: allow | step-up | review | block | revoke
```

## Install
> **Requires Python 3.9+** — check with `python3 --version`

```bash
git clone https://github.com/KOKOSde/identity-risk-engine.git
cd identity-risk-engine
python3 -m pip install -e .
```

## Quickstart
```python
from identity_risk_engine.policy_engine import PolicyEngine
from identity_risk_engine.risk_engine_ire import score_event
from identity_risk_engine.simulator_ire import generate_synthetic_auth_events

events = generate_synthetic_auth_events(num_users=20, num_sessions=200, attack_ratio=0.2, seed=42)
result = score_event(event=events.iloc[40].to_dict(), history_df=events.iloc[:40], policy_engine=PolicyEngine())
print(round(result["risk_score"], 4), result["decision"]["action"])
print(result["explanation"]["human_summary"])
```

`PolicyEngine` uses `decide()` (not `evaluate()`), and explanations are available via `result["explanation"]` from `risk_engine_ire.score_event(...)` or `explainer_ire.explain_scored_event(...)`.

## CLI Quickstart
```bash
python3 -m identity_risk_engine.cli_ire simulate --users 500 --sessions 20000 --attack-ratio 0.2 --out synthetic.csv
python3 -m identity_risk_engine.cli_ire score --events synthetic.csv --policy configs/default_policy.yaml --out scored.csv
python3 -m identity_risk_engine.cli_ire report --events scored.csv --out report.html
```

`simulate` example output:
```text
Generated 37926 events -> /tmp/ire_verify.csv
Attack mix:
  account_takeover: 159
  bot_behavior: 1014
  credential_stuffing: 1705
  impossible_travel: 318
  mfa_fatigue: 1496
  multi_account_sybil: 1155
  new_account_fraud: 188
  normal: 29587
  passkey_registration_abuse: 760
  recovery_abuse: 1208
  session_hijack: 336
```

`score` example output:
```text
Auto-selecting fast mode for 37926 events (use --full for complete signal extraction)
Scored 37926 events -> /tmp/ire_verify_scored.csv
Scoring mode: fast-auto (history_window=8)
Elapsed seconds: 27.90
Mean risk score: 0.1016
Action counts:
  allow: 24285
  allow_with_monitoring: 8678
  step_up_with_passkey: 2044
  step_up_with_totp: 1623
  require_recovery_review: 411
  block: 318
```

`report` example output:
```text
Report summary:
  total_events: 37926
  avg_risk_score: 0.101567
  p95_risk_score: 0.44
  positive_rate: 0.219876
Top actions:
  allow: 24285
  allow_with_monitoring: 8678
  step_up_with_passkey: 2044
  step_up_with_totp: 1623
  require_recovery_review: 411
  block: 318
Report written -> /tmp/ire_verify_report.html
```

## FastAPI Quickstart
```bash
uvicorn examples.fastapi_demo.app_ire:app --reload
curl -s http://127.0.0.1:8000/health
```

`POST /events` request schema:
```json
{
  "dry_run": false,
  "event": {
    "event_id": "evt_demo_001",
    "event_type": "login_success",
    "user_id": "user_demo_01",
    "session_id": "sess_demo_01",
    "timestamp": "2026-03-22T12:30:00Z",
    "ip": "34.23.11.9",
    "country": "US",
    "city_coarse": "San Francisco",
    "lat_coarse": 37.77,
    "lon_coarse": -122.42,
    "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/122.0",
    "device_hash": "dev_demo_01",
    "device_type": "desktop",
    "browser": "Chrome",
    "os": "Windows",
    "auth_method": "password",
    "success": true,
    "tenant_id": "tenant_1",
    "metadata": {"ip_asn": "AS12345"}
  }
}
```

Example curl:
```bash
curl -s -X POST http://127.0.0.1:8000/events \
  -H "Content-Type: application/json" \
  -d '{"dry_run":false,"event":{"event_id":"evt_demo_001","event_type":"login_success","user_id":"user_demo_01","session_id":"sess_demo_01","timestamp":"2026-03-22T12:30:00Z","ip":"34.23.11.9","country":"US","city_coarse":"San Francisco","lat_coarse":37.77,"lon_coarse":-122.42,"user_agent":"Mozilla/5.0","device_hash":"dev_demo_01","device_type":"desktop","browser":"Chrome","os":"Windows","auth_method":"password","success":true,"tenant_id":"tenant_1","metadata":{"ip_asn":"AS12345"}}}'
```

## Supported Auth Flows
- Password login (attempt/success/failure)
- Passkey registration and authentication
- MFA challenge flows (sent/passed/failed)
- Password reset and account recovery
- Session creation/revocation
- Credential profile changes (`email_changed`, `phone_changed`)
- OAuth/magic-link style flows normalized through the same event schema

## Risk Signals
| Category | Signals |
|---|---|
| Device | `new_device`, `device_dormant`, `multi_account_device`, `device_velocity`, `session_churn`, `emulator_heuristic` |
| Geo/Network | `impossible_travel`, `geo_velocity`, `new_country`, `new_asn`, `tor_vpn_proxy`, `ip_velocity`, `residential_vs_datacenter` |
| Behavior | `failure_burst`, `success_after_burst`, `unusual_hour`, `auth_method_switch`, `mfa_fatigue`, `recovery_abuse`, `login_cadence_anomaly`, `account_fanout`, `new_account_high_value`, `metadata_attack_hints` |
| Passkey | `new_passkey_unfamiliar_device`, `passkey_registration_burst`, `passkey_after_password_failure`, `authenticator_churn`, `credential_novelty` |
| Recovery | `recovery_unfamiliar_env`, `recovery_after_lockout`, `recovery_plus_credential_change`, `recovery_fanout`, `recovery_impossible_travel` |

## Policy Engine
Default config is at `configs/default_policy.yaml`.

```yaml
dry_run: false
thresholds:
  - { max_score: 0.15, action: allow }
  - { max_score: 0.30, action: allow_with_monitoring }
  - { max_score: 0.45, action: step_up_with_passkey }
  - { max_score: 0.60, action: step_up_with_totp }
  - { max_score: 0.72, action: step_up_with_email_code }
  - { max_score: 0.82, action: require_recovery_review }
  - { max_score: 0.90, action: manual_review }
  - { max_score: 0.97, action: block }
  - { max_score: 1.00, action: revoke_session }
```

Supported actions: `allow`, `allow_with_monitoring`, `step_up_with_passkey`, `step_up_with_totp`, `step_up_with_email_code`, `require_recovery_review`, `manual_review`, `block`, `revoke_session`.

## Benchmark Results
Generated from code with fixed seed using:

```bash
python3 scripts/generate_benchmark_table_ire.py --num-users 100 --num-sessions 4000 --attack-ratio 0.2 --seed 42
```

Script: `scripts/generate_benchmark_table_ire.py`  
Outputs: `demo_outputs/benchmark_table_ire.md`, `demo_outputs/benchmark_table_ire.json`

| Cohort | AUC | Precision@0.95Recall | Recall@0.95Precision |
|---|---:|---:|---:|
| Global | 0.987167 | 0.865171 | 0.712741 |
| account_takeover | 0.838010 | 0.024493 | 0.000000 |
| bot_behavior | 0.883121 | 0.078209 | 0.000000 |
| credential_stuffing | 0.872556 | 0.182295 | 0.000000 |
| impossible_travel | 0.695006 | 0.000000 | 0.000000 |
| mfa_fatigue | 0.827267 | 0.183863 | 0.000000 |
| multi_account_sybil | 0.934892 | 0.113269 | 0.000000 |
| new_account_fraud | 0.939105 | 0.027721 | 0.000000 |
| passkey_registration_abuse | 0.952384 | 0.123056 | 0.000000 |
| recovery_abuse | 0.964671 | 0.326278 | 0.000000 |
| session_hijack | 0.953187 | 0.084667 | 0.000000 |

## Scorer Quality Check (Seed 42)
- AUROC: `0.9818`
- Near-zero attack scores (`<0.1`): `189/8339` (`2.3%`)
- `session_hijack`: mean score `1.000`, near-zero `0.0%`
- `passkey_registration_abuse`: mean score `0.996`, near-zero `0.0%`

## Who This Is For
- Crypto exchanges
- Fintech identity teams
- Authentication platform builders
- Fraud analysts
- Security researchers

## Related Projects
- [onchain-sybil-detector](../onchain-sybil-detector)
- [LocalMod](../LocalMod)

## Contributing
See [CONTRIBUTING.md](CONTRIBUTING.md).

## License
MIT. See [LICENSE](LICENSE).
