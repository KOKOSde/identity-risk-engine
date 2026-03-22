"""Geo and network oriented auth risk signals."""

from __future__ import annotations

from typing import Any

import pandas as pd

from identity_risk_engine.geo_velocity import (
    DEFAULT_MAX_AIRCRAFT_SPEED_KMH,
    haversine_km,
)


def _signal(name: str, fired: bool, score: float, evidence: str) -> dict[str, Any]:
    return {
        "signal_name": name,
        "fired": bool(fired),
        "score": float(max(0.0, min(1.0, score))),
        "evidence": evidence,
    }


def _ip_asn(event: dict[str, Any]) -> str:
    meta = event.get("metadata") or {}
    return str(event.get("ip_asn") or meta.get("ip_asn") or "")


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def evaluate_geo_signals(
    event: dict[str, Any],
    user_history: pd.DataFrame | None = None,
    global_history: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    user_history = user_history if user_history is not None else pd.DataFrame()
    global_history = global_history if global_history is not None else pd.DataFrame()

    country = str(event.get("country") or "")
    ip = str(event.get("ip") or "")
    asn = _ip_asn(event)

    impossible = False
    velocity_score = 0.0
    speed_kmh = 0.0
    now = pd.to_datetime(event.get("timestamp"), utc=True, errors="coerce")
    cur_lat = _to_float(event.get("lat_coarse"))
    cur_lon = _to_float(event.get("lon_coarse"))
    if not user_history.empty and pd.notna(now) and cur_lat is not None and cur_lon is not None:
        hist = user_history
        if "timestamp" in hist.columns:
            hist = hist.copy()
            hist["timestamp"] = pd.to_datetime(hist["timestamp"], utc=True, errors="coerce")
            hist = hist[hist["timestamp"].notna() & (hist["timestamp"] <= now)]
        if "lat_coarse" in hist.columns and "lon_coarse" in hist.columns:
            hist = hist[hist["lat_coarse"].notna() & hist["lon_coarse"].notna()]
        if not hist.empty:
            prev = hist.sort_values("timestamp", kind="mergesort").iloc[-1]
            prev_lat = _to_float(prev.get("lat_coarse"))
            prev_lon = _to_float(prev.get("lon_coarse"))
            prev_ts = pd.to_datetime(prev.get("timestamp"), utc=True, errors="coerce")
            if prev_lat is not None and prev_lon is not None and pd.notna(prev_ts):
                delta_h = max((now - prev_ts).total_seconds() / 3600.0, 1e-6)
                distance_km = haversine_km(prev_lat, prev_lon, cur_lat, cur_lon)
                speed_kmh = float(distance_km / delta_h) if delta_h > 0 else 0.0
                impossible = bool(
                    distance_km > 0 and speed_kmh > DEFAULT_MAX_AIRCRAFT_SPEED_KMH
                )
                velocity_score = float(
                    max(0.0, min(1.0, speed_kmh / DEFAULT_MAX_AIRCRAFT_SPEED_KMH))
                )

    new_country = True
    if not user_history.empty and "country" in user_history.columns:
        new_country = country not in set(user_history["country"].fillna("").astype(str))

    new_asn = True
    if not user_history.empty and "metadata" in user_history.columns:
        seen = set()
        for item in user_history["metadata"].dropna():
            if isinstance(item, dict) and item.get("ip_asn"):
                seen.add(str(item["ip_asn"]))
        new_asn = asn not in seen if asn else False

    lowered_ip = ip.lower()
    lowered_asn = asn.lower()
    tor_vpn_proxy = any(x in lowered_ip for x in ["tor", "vpn", "proxy"]) or any(
        x in lowered_asn for x in ["hosting", "datacenter", "cloud", "vpn"]
    )

    ip_velocity_count = 0
    failure_from_ip = 0
    if not global_history.empty and "ip" in global_history.columns:
        same_ip = global_history[global_history["ip"].fillna("").astype(str) == ip]
        ip_velocity_count = int(same_ip.get("user_id", pd.Series(dtype=object)).astype(str).nunique())
        failure_from_ip = int((~same_ip.get("success", pd.Series(dtype=bool)).fillna(False)).sum())
    ip_velocity = ip_velocity_count >= 8 or failure_from_ip >= 15

    residential_vs_datacenter = "datacenter" if tor_vpn_proxy or ip.startswith("35.") or ip.startswith("34.") else "residential"
    is_datacenter = residential_vs_datacenter == "datacenter"

    return [
        _signal("impossible_travel", impossible, 1.0 if impossible else 0.0, f"speed_kmh={speed_kmh:.1f}"),
        _signal("geo_velocity", velocity_score > 0.6, velocity_score, f"geo_velocity_score={velocity_score:.3f}"),
        _signal("new_country", new_country, 0.55 if new_country else 0.0, f"country={country}"),
        _signal("new_asn", new_asn, 0.5 if new_asn else 0.0, f"asn={asn}"),
        _signal("tor_vpn_proxy", tor_vpn_proxy, 0.75 if tor_vpn_proxy else 0.0, f"ip={ip} asn={asn}"),
        _signal(
            "ip_velocity",
            ip_velocity,
            min(1.0, 0.1 * ip_velocity_count + 0.03 * failure_from_ip),
            f"accounts_on_ip={ip_velocity_count}, failures_on_ip={failure_from_ip}",
        ),
        _signal(
            "residential_vs_datacenter",
            is_datacenter,
            0.5 if is_datacenter else 0.0,
            f"classification={residential_vs_datacenter}",
        ),
    ]
