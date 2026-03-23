"""Expanded auth-event simulator for identity-risk-engine v0.2.0."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

from .events import AuthEventType

ATTACK_TYPES_IRE = (
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


FAILURE_EVENT_TYPES = {
    AuthEventType.login_failure.value,
    AuthEventType.passkey_auth_failure.value,
    AuthEventType.mfa_challenge_failed.value,
    AuthEventType.recovery_failure.value,
    AuthEventType.password_reset_failure.value,
}


LOCATIONS: list[tuple[str, str, float, float]] = [
    ("US", "San Francisco", 37.7749, -122.4194),
    ("US", "New York", 40.7128, -74.0060),
    ("GB", "London", 51.5074, -0.1278),
    ("DE", "Berlin", 52.5200, 13.4050),
    ("IN", "Mumbai", 19.0760, 72.8777),
    ("SG", "Singapore", 1.3521, 103.8198),
    ("JP", "Tokyo", 35.6762, 139.6503),
    ("AU", "Sydney", -33.8688, 151.2093),
]


PASSWORD_UAS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/122.0.0.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_3) Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) Firefox/123.0",
]

PASSKEY_UAS = [
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_3) Mobile Safari/605.1.15",
    "Mozilla/5.0 (Linux; Android 14; Pixel 8) Chrome/121.0 Mobile",
]

BOT_UA = "python-requests/2.31 (+bot)"


@dataclass
class UserProfile:
    user_id: str
    home_country: str
    home_city: str
    home_lat: float
    home_lon: float
    home_device_hash: str
    device_type: str
    email_domain: str
    tenant_id: str
    preferred_auth_method: str
    account_age_days: float


def _weighted_attack_types(
    attack_count: int,
    rng: np.random.Generator,
    attack_mix: Optional[dict[str, float]],
) -> list[str]:
    if attack_count <= 0:
        return []

    types = list(ATTACK_TYPES_IRE)
    if attack_mix:
        weights = np.array([float(attack_mix.get(t, 0.0)) for t in types], dtype=float)
        if weights.sum() <= 0:
            weights = np.ones(len(types), dtype=float)
    else:
        weights = np.ones(len(types), dtype=float)

    weights = weights / weights.sum()
    sampled = rng.choice(np.array(types, dtype=object), size=attack_count, replace=True, p=weights).tolist()

    if attack_count >= len(types):
        seen = set(sampled)
        for i, attack in enumerate(types):
            if attack not in seen:
                sampled[i] = attack
                seen.add(attack)

    return [str(x) for x in sampled]


def _pick_far_location(rng: np.random.Generator, current_country: str) -> tuple[str, str, float, float]:
    far = [loc for loc in LOCATIONS if loc[0] != current_country]
    if not far:
        far = LOCATIONS
    return far[int(rng.integers(0, len(far)))]


def _random_ip(rng: np.random.Generator, datacenter: bool = False) -> str:
    if datacenter:
        first = int(rng.choice(np.array([34, 35, 52, 54], dtype=int)).item())
    else:
        first = int(rng.choice(np.array([8, 24, 45, 67, 73, 99, 108, 142, 172, 184, 203], dtype=int)).item())
    return f"{first}.{int(rng.integers(1, 255))}.{int(rng.integers(1, 255))}.{int(rng.integers(1, 255))}"


def _browser_os_from_ua(ua: str) -> tuple[str, str]:
    browser = "Chrome" if "Chrome" in ua else ("Safari" if "Safari" in ua else "Firefox")
    if "iPhone" in ua:
        os = "iOS"
    elif "Android" in ua:
        os = "Android"
    elif "Mac OS X" in ua:
        os = "macOS"
    elif "Windows" in ua:
        os = "Windows"
    else:
        os = "Linux"
    return browser, os


def _mk_event(
    event_id: int,
    *,
    event_type: str,
    user_id: str,
    session_id: Optional[str],
    timestamp: Union[pd.Timestamp, str],
    ip: str,
    country: str,
    city_coarse: str,
    lat_coarse: float,
    lon_coarse: float,
    user_agent: str,
    device_hash: str,
    device_type: str,
    browser: str,
    os: str,
    auth_method: str,
    success: bool,
    failure_reason: Optional[str] = None,
    challenge_type: Optional[str] = None,
    recovery_channel: Optional[str] = None,
    email_domain: Optional[str] = None,
    tenant_id: str = "default",
    metadata: Optional[dict[str, Any]] = None,
    attack_type: str = "normal",
    label: int = 0,
) -> dict[str, Any]:
    ts = pd.to_datetime(timestamp, utc=True, errors="coerce")
    ts_str = ts.strftime("%Y-%m-%dT%H:%M:%SZ") if pd.notna(ts) else ""

    return {
        "event_id": f"evt_{event_id:09d}",
        "event_type": event_type,
        "user_id": user_id,
        "session_id": session_id,
        "timestamp": ts_str,
        "ip": ip,
        "country": country,
        "city_coarse": city_coarse,
        "lat_coarse": float(lat_coarse),
        "lon_coarse": float(lon_coarse),
        "user_agent": user_agent,
        "device_hash": device_hash,
        "device_type": device_type,
        "browser": browser,
        "os": os,
        "auth_method": auth_method,
        "success": bool(success),
        "failure_reason": failure_reason,
        "challenge_type": challenge_type,
        "recovery_channel": recovery_channel,
        "email_domain": email_domain,
        "tenant_id": tenant_id,
        "metadata": dict(metadata or {}),
        "attack_type": attack_type,
        "label": int(label),
    }


def _event_from_base(
    event_id: int,
    base: dict[str, Any],
    *,
    event_type: str,
    timestamp: Union[pd.Timestamp, str],
    attack_type: str,
    label: int = 1,
    **updates: Any,
) -> dict[str, Any]:
    merged = {
        "event_type": event_type,
        "user_id": updates.pop("user_id", base["user_id"]),
        "session_id": updates.pop("session_id", base.get("session_id")),
        "timestamp": timestamp,
        "ip": updates.pop("ip", base.get("ip")),
        "country": updates.pop("country", base.get("country")),
        "city_coarse": updates.pop("city_coarse", base.get("city_coarse")),
        "lat_coarse": updates.pop("lat_coarse", base.get("lat_coarse")),
        "lon_coarse": updates.pop("lon_coarse", base.get("lon_coarse")),
        "user_agent": updates.pop("user_agent", base.get("user_agent")),
        "device_hash": updates.pop("device_hash", base.get("device_hash")),
        "device_type": updates.pop("device_type", base.get("device_type")),
        "browser": updates.pop("browser", base.get("browser")),
        "os": updates.pop("os", base.get("os")),
        "auth_method": updates.pop("auth_method", base.get("auth_method")),
        "success": updates.pop("success", base.get("success", False)),
        "failure_reason": updates.pop("failure_reason", None),
        "challenge_type": updates.pop("challenge_type", base.get("challenge_type")),
        "recovery_channel": updates.pop("recovery_channel", base.get("recovery_channel")),
        "email_domain": updates.pop("email_domain", base.get("email_domain")),
        "tenant_id": updates.pop("tenant_id", base.get("tenant_id", "default")),
        "metadata": updates.pop("metadata", dict(base.get("metadata") or {})),
        "attack_type": attack_type,
        "label": label,
    }
    return _mk_event(event_id, **merged)


def generate_synthetic_auth_events(
    num_users: int = 500,
    num_sessions: int = 20000,
    attack_ratio: float = 0.2,
    seed: int = 42,
    passkey_adoption_rate: float = 0.35,
    recovery_flow_rate: float = 0.08,
    attack_mix: Optional[dict[str, float]] = None,
) -> pd.DataFrame:
    """Generate synthetic auth events with modern flows and attack patterns."""

    if num_users <= 0:
        raise ValueError("num_users must be > 0")
    if num_sessions <= 0:
        raise ValueError("num_sessions must be > 0")
    if not 0.0 <= attack_ratio <= 1.0:
        raise ValueError("attack_ratio must be in [0, 1]")
    if not 0.0 <= passkey_adoption_rate <= 1.0:
        raise ValueError("passkey_adoption_rate must be in [0, 1]")
    if not 0.0 <= recovery_flow_rate <= 1.0:
        raise ValueError("recovery_flow_rate must be in [0, 1]")

    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2026-01-01T00:00:00Z")

    profiles: list[UserProfile] = []
    profile_by_user: dict[str, UserProfile] = {}
    user_last_ts: dict[str, pd.Timestamp] = {}

    for i in range(num_users):
        user_id = f"user_{i:06d}"
        country, city, lat, lon = LOCATIONS[int(rng.integers(0, len(LOCATIONS)))]
        device_type = "mobile" if rng.random() < 0.45 else "desktop"
        preferred_auth_method = "passkey" if rng.random() < passkey_adoption_rate else "password"
        account_age_days = float(np.clip(rng.gamma(shape=2.0, scale=120.0), 0.1, 3650.0))
        email_domain = "example.com" if rng.random() < 0.65 else "proton.me"
        tenant_id = f"tenant_{int(rng.integers(1, 6))}"

        profile = UserProfile(
            user_id=user_id,
            home_country=country,
            home_city=city,
            home_lat=float(lat),
            home_lon=float(lon),
            home_device_hash=f"dev_{i:06d}_{int(rng.integers(1000, 9999))}",
            device_type=device_type,
            email_domain=email_domain,
            tenant_id=tenant_id,
            preferred_auth_method=preferred_auth_method,
            account_age_days=account_age_days,
        )
        profiles.append(profile)
        profile_by_user[user_id] = profile
        user_last_ts[user_id] = start + pd.Timedelta(hours=float(rng.uniform(0, 18)))

    rows: list[dict[str, Any]] = []
    event_id = 0
    session_anchors: list[dict[str, Any]] = []

    for _ in range(num_sessions):
        profile = profiles[int(rng.integers(0, len(profiles)))]
        user_id = profile.user_id

        t0 = user_last_ts[user_id] + pd.Timedelta(hours=max(float(rng.normal(18.0, 8.0)), 0.2))
        user_last_ts[user_id] = t0

        session_id = f"sess_{int(rng.integers(10_000_000, 99_999_999))}"
        auth_method = (
            profile.preferred_auth_method
            if rng.random() < 0.88
            else ("passkey" if profile.preferred_auth_method == "password" else "password")
        )

        is_datacenter = bool(rng.random() < 0.06)
        ip = _random_ip(rng, datacenter=is_datacenter)
        ip_asn = (
            f"DATACENTER-AS{int(rng.integers(10000, 90000))}"
            if is_datacenter
            else f"RESIDENTIAL-AS{int(rng.integers(1000, 9999))}"
        )

        if rng.random() < 0.96:
            country = profile.home_country
            city = profile.home_city
            lat = profile.home_lat + float(rng.normal(0.0, 0.08))
            lon = profile.home_lon + float(rng.normal(0.0, 0.08))
        else:
            country, city, lat, lon = _pick_far_location(rng, profile.home_country)

        if rng.random() < 0.88:
            device_hash = profile.home_device_hash
            device_type = profile.device_type
        else:
            device_hash = f"dev_alt_{user_id}_{int(rng.integers(1000, 9999))}"
            device_type = "mobile" if rng.random() < 0.5 else "desktop"

        user_agent = (
            PASSKEY_UAS[int(rng.integers(0, len(PASSKEY_UAS)))]
            if auth_method == "passkey"
            else PASSWORD_UAS[int(rng.integers(0, len(PASSWORD_UAS)))]
        )
        browser, os_name = _browser_os_from_ua(user_agent)

        metadata_base = {
            "ip_asn": ip_asn,
            "home_country": profile.home_country,
            "account_age_days": profile.account_age_days,
            "risk_seed": int(rng.integers(0, 1_000_000)),
        }

        rows.append(
            _mk_event(
                event_id,
                event_type=AuthEventType.login_attempt.value,
                user_id=user_id,
                session_id=session_id,
                timestamp=t0,
                ip=ip,
                country=country,
                city_coarse=city,
                lat_coarse=lat,
                lon_coarse=lon,
                user_agent=user_agent,
                device_hash=device_hash,
                device_type=device_type,
                browser=browser,
                os=os_name,
                auth_method=auth_method,
                success=False,
                email_domain=profile.email_domain,
                tenant_id=profile.tenant_id,
                metadata=metadata_base,
            )
        )
        event_id += 1

        success_prob = 0.96 if auth_method == "passkey" else 0.90
        auth_success = bool(rng.random() < success_prob)
        auth_event_type = (
            AuthEventType.passkey_auth_success.value
            if auth_method == "passkey" and auth_success
            else AuthEventType.passkey_auth_failure.value
            if auth_method == "passkey"
            else AuthEventType.login_success.value
            if auth_success
            else AuthEventType.login_failure.value
        )

        rows.append(
            _mk_event(
                event_id,
                event_type=auth_event_type,
                user_id=user_id,
                session_id=session_id,
                timestamp=t0 + pd.Timedelta(seconds=30),
                ip=ip,
                country=country,
                city_coarse=city,
                lat_coarse=lat,
                lon_coarse=lon,
                user_agent=user_agent,
                device_hash=device_hash,
                device_type=device_type,
                browser=browser,
                os=os_name,
                auth_method=auth_method,
                success=auth_success,
                failure_reason=None if auth_success else "invalid_credentials",
                email_domain=profile.email_domain,
                tenant_id=profile.tenant_id,
                metadata=metadata_base,
            )
        )
        anchor_row_idx = len(rows) - 1
        event_id += 1

        if auth_method == "password" and rng.random() < 0.10:
            rows.append(
                _mk_event(
                    event_id,
                    event_type=AuthEventType.mfa_challenge_sent.value,
                    user_id=user_id,
                    session_id=session_id,
                    timestamp=t0 + pd.Timedelta(minutes=1),
                    ip=ip,
                    country=country,
                    city_coarse=city,
                    lat_coarse=lat,
                    lon_coarse=lon,
                    user_agent=user_agent,
                    device_hash=device_hash,
                    device_type=device_type,
                    browser=browser,
                    os=os_name,
                    auth_method=auth_method,
                    success=False,
                    challenge_type="totp",
                    email_domain=profile.email_domain,
                    tenant_id=profile.tenant_id,
                    metadata=metadata_base,
                )
            )
            event_id += 1

            mfa_passed = bool(rng.random() < 0.85)
            rows.append(
                _mk_event(
                    event_id,
                    event_type=(
                        AuthEventType.mfa_challenge_passed.value
                        if mfa_passed
                        else AuthEventType.mfa_challenge_failed.value
                    ),
                    user_id=user_id,
                    session_id=session_id,
                    timestamp=t0 + pd.Timedelta(minutes=2),
                    ip=ip,
                    country=country,
                    city_coarse=city,
                    lat_coarse=lat,
                    lon_coarse=lon,
                    user_agent=user_agent,
                    device_hash=device_hash,
                    device_type=device_type,
                    browser=browser,
                    os=os_name,
                    auth_method=auth_method,
                    success=mfa_passed,
                    challenge_type="totp",
                    failure_reason=None if mfa_passed else "mfa_failed",
                    email_domain=profile.email_domain,
                    tenant_id=profile.tenant_id,
                    metadata=metadata_base,
                )
            )
            event_id += 1

        if auth_success:
            rows.append(
                _mk_event(
                    event_id,
                    event_type=AuthEventType.session_created.value,
                    user_id=user_id,
                    session_id=session_id,
                    timestamp=t0 + pd.Timedelta(minutes=2),
                    ip=ip,
                    country=country,
                    city_coarse=city,
                    lat_coarse=lat,
                    lon_coarse=lon,
                    user_agent=user_agent,
                    device_hash=device_hash,
                    device_type=device_type,
                    browser=browser,
                    os=os_name,
                    auth_method=auth_method,
                    success=True,
                    email_domain=profile.email_domain,
                    tenant_id=profile.tenant_id,
                    metadata=metadata_base,
                )
            )
            event_id += 1

            if rng.random() < 0.75:
                rows.append(
                    _mk_event(
                        event_id,
                        event_type=AuthEventType.logout.value,
                        user_id=user_id,
                        session_id=session_id,
                        timestamp=t0 + pd.Timedelta(minutes=float(rng.uniform(8, 180))),
                        ip=ip,
                        country=country,
                        city_coarse=city,
                        lat_coarse=lat,
                        lon_coarse=lon,
                        user_agent=user_agent,
                        device_hash=device_hash,
                        device_type=device_type,
                        browser=browser,
                        os=os_name,
                        auth_method=auth_method,
                        success=True,
                        email_domain=profile.email_domain,
                        tenant_id=profile.tenant_id,
                        metadata=metadata_base,
                    )
                )
                event_id += 1

        if rng.random() < recovery_flow_rate:
            rows.append(
                _mk_event(
                    event_id,
                    event_type=AuthEventType.recovery_requested.value,
                    user_id=user_id,
                    session_id=session_id,
                    timestamp=t0 + pd.Timedelta(minutes=4),
                    ip=ip,
                    country=country,
                    city_coarse=city,
                    lat_coarse=lat,
                    lon_coarse=lon,
                    user_agent=user_agent,
                    device_hash=device_hash,
                    device_type=device_type,
                    browser=browser,
                    os=os_name,
                    auth_method=auth_method,
                    success=False,
                    recovery_channel="email",
                    email_domain=profile.email_domain,
                    tenant_id=profile.tenant_id,
                    metadata=metadata_base,
                )
            )
            event_id += 1

            recovery_success = bool(rng.random() < 0.80)
            rows.append(
                _mk_event(
                    event_id,
                    event_type=(
                        AuthEventType.recovery_success.value
                        if recovery_success
                        else AuthEventType.recovery_failure.value
                    ),
                    user_id=user_id,
                    session_id=session_id,
                    timestamp=t0 + pd.Timedelta(minutes=6),
                    ip=ip,
                    country=country,
                    city_coarse=city,
                    lat_coarse=lat,
                    lon_coarse=lon,
                    user_agent=user_agent,
                    device_hash=device_hash,
                    device_type=device_type,
                    browser=browser,
                    os=os_name,
                    auth_method=auth_method,
                    success=recovery_success,
                    failure_reason=None if recovery_success else "recovery_token_invalid",
                    recovery_channel="email",
                    email_domain=profile.email_domain,
                    tenant_id=profile.tenant_id,
                    metadata=metadata_base,
                )
            )
            event_id += 1

        if auth_method == "passkey" and rng.random() < 0.03:
            rows.append(
                _mk_event(
                    event_id,
                    event_type=AuthEventType.passkey_registered.value,
                    user_id=user_id,
                    session_id=session_id,
                    timestamp=t0 + pd.Timedelta(minutes=3),
                    ip=ip,
                    country=country,
                    city_coarse=city,
                    lat_coarse=lat,
                    lon_coarse=lon,
                    user_agent=user_agent,
                    device_hash=device_hash,
                    device_type=device_type,
                    browser=browser,
                    os=os_name,
                    auth_method="passkey",
                    success=True,
                    email_domain=profile.email_domain,
                    tenant_id=profile.tenant_id,
                    metadata={**metadata_base, "authenticator_aaguid": f"aaguid-{int(rng.integers(1000,9999))}"},
                )
            )
            event_id += 1

        session_anchors.append({"row_idx": anchor_row_idx, "session_id": session_id})

    attack_count = int(round(num_sessions * attack_ratio))
    attack_types = _weighted_attack_types(attack_count, rng, attack_mix)
    sybil_ip = f"34.{int(rng.integers(1, 255))}.{int(rng.integers(1, 255))}.{int(rng.integers(1, 255))}"
    sybil_device = f"sybil_{int(rng.integers(1000, 9999))}"

    if attack_types and session_anchors:
        chosen = rng.choice(np.arange(len(session_anchors)), size=min(len(attack_types), len(session_anchors)), replace=False)
        for anchor_idx, attack in zip(chosen.tolist(), attack_types):
            anchor = session_anchors[int(anchor_idx)]
            base = rows[int(anchor["row_idx"])]
            base_ts = pd.to_datetime(base["timestamp"], utc=True, errors="coerce")
            if pd.isna(base_ts):
                base_ts = start

            base["label"] = 1
            base["attack_type"] = attack
            base_meta = dict(base.get("metadata") or {})

            if attack == "credential_stuffing":
                base["event_type"] = AuthEventType.login_success.value
                base["success"] = True
                base["failure_reason"] = None
                base_meta["failed_attempts"] = int(rng.integers(6, 16))
                base_meta["credential_stuffing"] = True
                base["metadata"] = base_meta

                for i in range(int(base_meta["failed_attempts"])):
                    rows.append(
                        _event_from_base(
                            event_id,
                            base,
                            event_type=AuthEventType.login_failure.value,
                            timestamp=base_ts - pd.Timedelta(minutes=12) + pd.Timedelta(seconds=60 * i),
                            attack_type=attack,
                            success=False,
                            failure_reason="invalid_credentials",
                            metadata={**base_meta, "credential_stuffing": True},
                        )
                    )
                    event_id += 1

            elif attack == "account_takeover":
                country, city, lat, lon = _pick_far_location(rng, str(base.get("country") or ""))
                base["event_type"] = AuthEventType.login_success.value
                base["success"] = True
                base["country"] = country
                base["city_coarse"] = city
                base["lat_coarse"] = float(lat)
                base["lon_coarse"] = float(lon)
                base["device_hash"] = f"atk_dev_{int(rng.integers(100000, 999999))}"
                base["ip"] = _random_ip(rng, datacenter=True)
                base_meta.update({"new_device": True, "new_asn": True, "account_takeover": True, "ip_asn": f"DATACENTER-AS{int(rng.integers(10000, 90000))}"})
                base["metadata"] = base_meta

            elif attack == "bot_behavior":
                base["user_agent"] = BOT_UA
                base["device_type"] = "bot"
                base_meta.update({"bot_behavior": True, "cadence_seconds": 30})
                base["metadata"] = base_meta

                for i in range(5):
                    rows.append(
                        _event_from_base(
                            event_id,
                            base,
                            event_type=AuthEventType.login_attempt.value,
                            timestamp=base_ts + pd.Timedelta(seconds=30 * (i + 1)),
                            attack_type=attack,
                            success=False,
                            ip=_random_ip(rng, datacenter=True),
                            user_agent=BOT_UA,
                            device_type="bot",
                            metadata={**base_meta, "bot_behavior": True, "cadence_seconds": 30},
                        )
                    )
                    event_id += 1

            elif attack == "impossible_travel":
                orig_country = str(base.get("country") or "US")
                prev_country, prev_city, prev_lat, prev_lon = _pick_far_location(rng, orig_country)
                rows.append(
                    _event_from_base(
                        event_id,
                        base,
                        event_type=AuthEventType.login_success.value,
                        timestamp=base_ts - pd.Timedelta(minutes=9),
                        attack_type=attack,
                        success=True,
                        country=prev_country,
                        city_coarse=prev_city,
                        lat_coarse=prev_lat,
                        lon_coarse=prev_lon,
                        metadata={**base_meta, "impossible_travel": True},
                    )
                )
                event_id += 1
                base_meta["impossible_travel"] = True
                base["metadata"] = base_meta

            elif attack == "new_account_fraud":
                base["event_type"] = AuthEventType.login_success.value
                base["success"] = True
                base_meta.update(
                    {
                        "account_age_days": float(rng.uniform(0.01, 0.95)),
                        "new_account_fraud": True,
                        "high_value_action": "instant_withdrawal",
                    }
                )
                base["metadata"] = base_meta

            elif attack == "session_hijack":
                base["event_type"] = AuthEventType.session_created.value
                base["success"] = True
                base["device_hash"] = f"hijack_{int(rng.integers(100000, 999999))}"
                base["ip"] = _random_ip(rng, datacenter=True)
                base_meta.update({"session_hijack": True, "stolen_cookie_replay": True, "new_device": True, "new_asn": True})
                base["metadata"] = base_meta

                rows.append(
                    _event_from_base(
                        event_id,
                        base,
                        event_type=AuthEventType.session_revoked.value,
                        timestamp=base_ts + pd.Timedelta(minutes=5),
                        attack_type=attack,
                        success=True,
                        metadata={**base_meta, "session_hijack": True},
                    )
                )
                event_id += 1

            elif attack == "mfa_fatigue":
                base["event_type"] = AuthEventType.mfa_challenge_passed.value
                base["success"] = True
                base["challenge_type"] = "push"
                base_meta.update({"mfa_fatigue": True, "mfa_challenges": int(rng.integers(6, 15))})
                base["metadata"] = base_meta

                for i in range(int(base_meta["mfa_challenges"])):
                    rows.append(
                        _event_from_base(
                            event_id,
                            base,
                            event_type=AuthEventType.mfa_challenge_sent.value,
                            timestamp=base_ts - pd.Timedelta(minutes=20) + pd.Timedelta(seconds=90 * i),
                            attack_type=attack,
                            success=False,
                            challenge_type="push",
                            metadata={**base_meta, "mfa_fatigue": True},
                        )
                    )
                    event_id += 1

            elif attack == "recovery_abuse":
                base["event_type"] = AuthEventType.recovery_success.value
                base["success"] = True
                base["recovery_channel"] = "email"
                base_meta.update({"recovery_abuse": True, "new_device": True, "new_asn": True})
                base["metadata"] = base_meta

                for i in range(6):
                    rows.append(
                        _event_from_base(
                            event_id,
                            base,
                            event_type=AuthEventType.login_failure.value,
                            timestamp=base_ts - pd.Timedelta(minutes=15) + pd.Timedelta(seconds=90 * i),
                            attack_type=attack,
                            success=False,
                            failure_reason="invalid_credentials",
                            metadata={**base_meta, "recovery_abuse": True},
                        )
                    )
                    event_id += 1
                rows.append(
                    _event_from_base(
                        event_id,
                        base,
                        event_type=AuthEventType.recovery_requested.value,
                        timestamp=base_ts - pd.Timedelta(minutes=1),
                        attack_type=attack,
                        success=False,
                        recovery_channel="email",
                        metadata={**base_meta, "recovery_abuse": True},
                    )
                )
                event_id += 1

            elif attack == "passkey_registration_abuse":
                base["event_type"] = AuthEventType.passkey_registered.value
                base["auth_method"] = "passkey"
                base["success"] = True
                base_meta.update(
                    {
                        "passkey_registration_abuse": True,
                        "registrations_30m": int(rng.integers(3, 8)),
                        "authenticator_aaguid": f"aaguid-{int(rng.integers(10000, 99999))}",
                    }
                )
                base["metadata"] = base_meta

                burst = int(base_meta["registrations_30m"])
                for i in range(max(0, burst - 1)):
                    rows.append(
                        _event_from_base(
                            event_id,
                            base,
                            event_type=(
                                AuthEventType.passkey_registered.value if i % 2 == 0 else AuthEventType.device_enrolled.value
                            ),
                            timestamp=base_ts + pd.Timedelta(minutes=2 * (i + 1)),
                            attack_type=attack,
                            success=True,
                            device_hash=f"pk_abuse_{int(rng.integers(100000, 999999))}",
                            auth_method="passkey",
                            metadata={
                                **base_meta,
                                "passkey_registration_abuse": True,
                                "authenticator_aaguid": f"aaguid-{int(rng.integers(10000, 99999))}",
                            },
                        )
                    )
                    event_id += 1

            elif attack == "multi_account_sybil":
                base["ip"] = sybil_ip
                base["device_hash"] = sybil_device
                base_meta.update({"multi_account_sybil": True, "sybil_cluster": "cluster_1", "new_device": True})
                base["metadata"] = base_meta

                fanout_n = min(6, len(profiles))
                fanout_users = rng.choice(
                    np.array([p.user_id for p in profiles], dtype=object),
                    size=fanout_n,
                    replace=False,
                )
                for j, fanout_user in enumerate(fanout_users.tolist()):
                    fan_profile = profile_by_user[str(fanout_user)]
                    fan_method = fan_profile.preferred_auth_method
                    ua = PASSKEY_UAS[0] if fan_method == "passkey" else PASSWORD_UAS[0]
                    br, os_name = _browser_os_from_ua(ua)
                    rows.append(
                        _mk_event(
                            event_id,
                            event_type=AuthEventType.login_attempt.value,
                            user_id=str(fanout_user),
                            session_id=f"sybil_sess_{int(rng.integers(1000, 9999))}_{j}",
                            timestamp=base_ts + pd.Timedelta(seconds=20 * j),
                            ip=sybil_ip,
                            country=fan_profile.home_country,
                            city_coarse=fan_profile.home_city,
                            lat_coarse=fan_profile.home_lat,
                            lon_coarse=fan_profile.home_lon,
                            user_agent=ua,
                            device_hash=sybil_device,
                            device_type="mobile",
                            browser=br,
                            os=os_name,
                            auth_method=fan_method,
                            success=False,
                            email_domain=fan_profile.email_domain,
                            tenant_id=fan_profile.tenant_id,
                            metadata={
                                "ip_asn": "DATACENTER-AS77777",
                                "home_country": fan_profile.home_country,
                                "account_age_days": fan_profile.account_age_days,
                                "multi_account_sybil": True,
                                "sybil_cluster": "cluster_1",
                            },
                            attack_type=attack,
                            label=1,
                        )
                    )
                    event_id += 1

            rows[int(anchor["row_idx"])] = base

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.sort_values(["timestamp", "event_id"], kind="mergesort").reset_index(drop=True)

    metadata_series = df["metadata"].apply(lambda x: x if isinstance(x, dict) else {})
    df["ip_asn"] = metadata_series.apply(lambda m: m.get("ip_asn"))
    df["account_age_days"] = pd.to_numeric(metadata_series.apply(lambda m: m.get("account_age_days")), errors="coerce")

    failure_mask = df["event_type"].isin(FAILURE_EVENT_TYPES)
    failures_per_session = failure_mask.groupby(df["session_id"]).transform("sum")
    df["failed_attempts"] = failures_per_session.fillna(0).astype(int)

    session_duration_sec = (
        df.groupby("session_id")["timestamp"].transform("max") - df.groupby("session_id")["timestamp"].transform("min")
    ).dt.total_seconds()
    df["session_duration"] = session_duration_sec.fillna(0.0).clip(lower=0.0)

    df["country_mismatch"] = (
        df["country"].fillna("").astype(str)
        != metadata_series.apply(lambda m: str(m.get("home_country") or ""))
    ).astype(int)

    df["is_failure"] = failure_mask.astype(int)
    df["is_passkey"] = (
        (df["auth_method"].fillna("") == "passkey")
        | df["event_type"].astype(str).str.startswith("passkey_")
    ).astype(int)
    df["is_recovery"] = df["event_type"].astype(str).str.startswith("recovery_").astype(int)
    df["hour"] = df["timestamp"].dt.hour.fillna(0).astype(int)

    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    return df


def train_test_time_split_ire(df: pd.DataFrame, test_ratio: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Time-ordered split for simulator_ire outputs."""

    if not 0.0 < test_ratio < 1.0:
        raise ValueError("test_ratio must be in (0, 1)")

    ordered = df.copy()
    ordered["_ts"] = pd.to_datetime(ordered["timestamp"], utc=True, errors="coerce")
    ordered = ordered.sort_values("_ts", kind="mergesort")
    cut = int((1.0 - test_ratio) * len(ordered))

    train = ordered.iloc[:cut].drop(columns=["_ts"]).copy()
    test = ordered.iloc[cut:].drop(columns=["_ts"]).copy()
    return train, test
