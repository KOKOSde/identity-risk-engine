# FastAPI Demo (`app_ire.py`)

Minimal API for ingesting auth events, scoring risk, simulating traffic, and returning dashboard-ready metrics.

## Run

```bash
uvicorn app_ire:app --reload
```

From repo root:

```bash
uvicorn examples.fastapi_demo.app_ire:app --reload
```

## Endpoints

- `GET /health`
- `POST /events`
- `POST /simulate`
- `GET /dashboard-data`

## Curl Examples

Health:

```bash
curl -s http://127.0.0.1:8000/health
```

Score one event:

```bash
curl -s -X POST http://127.0.0.1:8000/events \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

`POST /events` expects a top-level object with:
- `dry_run` (bool, optional)
- `event` (required auth event object matching `identity_risk_engine.events.AuthEvent`)

Simulate and score:

```bash
curl -s -X POST http://127.0.0.1:8000/simulate \
  -H "Content-Type: application/json" \
  -d '{"num_users": 10, "num_sessions": 100, "attack_ratio": 0.2}'
```

Dashboard data:

```bash
curl -s http://127.0.0.1:8000/dashboard-data
```

## Sample Response Snippets

`GET /health`

```json
{
  "status": "ok",
  "service": "identity-risk-engine",
  "version": "0.2.0",
  "events_seen": 120,
  "events_scored": 120
}
```

`POST /events`

```json
{
  "event_id": "evt_demo_001",
  "risk_score": 0.73,
  "action": "require_recovery_review",
  "decision": {
    "action": "require_recovery_review",
    "risk_score": 0.73,
    "reasons": ["new_country", "new_device"],
    "evidence": ["country=US", "device_hash=dev_demo_01 seen_before=False"],
    "dry_run": false
  }
}
```

`GET /dashboard-data`

```json
{
  "total_events": 100,
  "risk_distribution": [
    {"bucket": "0.0-0.2", "count": 37},
    {"bucket": "0.2-0.4", "count": 22},
    {"bucket": "0.4-0.6", "count": 18},
    {"bucket": "0.6-0.8", "count": 16},
    {"bucket": "0.8-1.0", "count": 7}
  ],
  "action_counts": {
    "allow": 34,
    "allow_with_monitoring": 21,
    "step_up_with_totp": 18,
    "manual_review": 8
  }
}
```
