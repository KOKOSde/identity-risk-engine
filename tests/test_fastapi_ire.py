from __future__ import annotations

from fastapi.testclient import TestClient

from examples.fastapi_demo import app_ire


def setup_function() -> None:
    app_ire.EVENT_HISTORY = app_ire.pd.DataFrame()
    app_ire.SCORED_HISTORY = app_ire.pd.DataFrame()


def test_health_endpoint() -> None:
    client = TestClient(app_ire.app)
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "identity-risk-engine"


def test_simulate_and_dashboard_endpoints() -> None:
    client = TestClient(app_ire.app)
    sim = client.post("/simulate", json={"num_users": 5, "num_sessions": 60, "attack_ratio": 0.2})
    assert sim.status_code == 200
    sim_body = sim.json()
    assert sim_body["generated_events"] > 0
    assert "mean_risk_score" in sim_body

    dash = client.get("/dashboard-data")
    assert dash.status_code == 200
    dash_body = dash.json()
    assert dash_body["total_events"] >= sim_body["generated_events"]
    assert isinstance(dash_body["risk_distribution"], list)


def test_events_endpoint_returns_scored_decision() -> None:
    client = TestClient(app_ire.app)
    payload = {
        "event": {
            "event_id": "evt_api_1",
            "event_type": "login_attempt",
            "user_id": "api_user_1",
            "session_id": "sess_api_1",
            "timestamp": "2026-03-22T12:00:00Z",
            "ip": "8.8.8.8",
            "country": "US",
            "city_coarse": "San Francisco",
            "lat_coarse": 37.7749,
            "lon_coarse": -122.4194,
            "user_agent": "Mozilla/5.0",
            "device_hash": "dev_api_1",
            "device_type": "desktop",
            "browser": "Chrome",
            "os": "macOS",
            "auth_method": "password",
            "success": False,
            "failure_reason": "invalid_credentials",
            "challenge_type": None,
            "recovery_channel": None,
            "email_domain": "example.com",
            "tenant_id": "default",
            "metadata": {"ip_asn": "RESIDENTIAL-AS15169"},
        },
        "dry_run": True,
    }

    resp = client.post("/events", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["event_id"] == "evt_api_1"
    assert 0.0 <= body["risk_score"] <= 1.0
    assert isinstance(body["action"], str)
    assert "decision" in body
    assert "signals" in body
