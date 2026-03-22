# Changelog (IRE)

## [0.2.0] - 2026-03-22
### Added
- Expanded auth event model (20+ event types, passkey/recovery/MFA support)
- 25+ risk signals across 5 categories (device, geo, behavior, passkey, recovery)
- Policy engine with YAML config, per-tenant/per-method policies, dry-run mode
- Explainability output (human-readable summaries + machine-readable JSON)
- 5 new attack scenarios in simulator (session hijack, MFA fatigue, recovery abuse, passkey registration abuse, multi-account Sybil)
- FastAPI demo app with `/events`, `/simulate`, `/dashboard-data` endpoints
- CLI: `simulate`, `score`, `report` commands
- Full test suite (32 tests passing)

## [0.1.0] - 2026-03-01
### Added
- Initial release: session-level identity risk scoring
- Impossible travel detection, device fingerprinting, behavioral anomaly scoring
- XGBoost + LightGBM composite ensemble
- Synthetic data generator with 5 attack types
- Plotly dashboard and cohort analysis notebooks
