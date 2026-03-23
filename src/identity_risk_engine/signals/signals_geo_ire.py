"""Geo and network oriented auth risk signals."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

from identity_risk_engine.geo_velocity import haversine_km

IMPOSSIBLE_TRAVEL_SPEED_KMH = 1000.0
PAIRWISE_LOOKBACK = 5


@dataclass
class GeoCompositeResult:
    pairwise_geo_velocity_score: float
    pairwise_geo_velocity_kmh: float
    pairwise_impossible: bool
    pairwise_evidence: str
    new_country_for_user_score: float
    new_country_for_user_evidence: str
    first_country_for_user: bool
    rare_country_global_score: float
    rare_country_global_evidence: str
    device_location_mismatch_score: float
    device_location_mismatch_evidence: str
    geo_session_break_score: float
    geo_session_break_evidence: str
    impossible_travel_composite_score: float
    impossible_travel_composite_evidence: str


def _signal(name: str, fired: bool, score: float, evidence: str) -> dict[str, Any]:
    return {
        "signal_name": name,
        "fired": bool(fired),
        "score": float(max(0.0, min(1.0, score))),
        "evidence": evidence,
    }


def _ip_asn(event: dict[str, Any]) -> str:
    meta = event.get("metadata") or {}
    if not isinstance(meta, dict):
        meta = {}
    return str(event.get("ip_asn") or meta.get("ip_asn") or "")


def _to_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_history(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    return df if df is not None else pd.DataFrame()


def _recent_user_rows(
    user_history: pd.DataFrame,
    now: pd.Timestamp,
    *,
    limit: int = PAIRWISE_LOOKBACK,
) -> pd.DataFrame:
    if user_history.empty:
        return pd.DataFrame()

    hist = user_history.copy()
    hist["timestamp"] = pd.to_datetime(hist.get("timestamp"), utc=True, errors="coerce")
    hist = hist[hist["timestamp"].notna()]
    if pd.notna(now):
        hist = hist[hist["timestamp"] <= now]
    if hist.empty:
        return hist
    return hist.sort_values("timestamp", kind="mergesort").tail(limit)


def _pairwise_velocity_signal(
    event: dict[str, Any],
    user_history: pd.DataFrame,
) -> tuple[float, float, bool, str]:
    now = pd.to_datetime(event.get("timestamp"), utc=True, errors="coerce")
    cur_lat = _to_float(event.get("lat_coarse"))
    cur_lon = _to_float(event.get("lon_coarse"))
    if pd.isna(now) or cur_lat is None or cur_lon is None:
        return 0.0, 0.0, False, "Insufficient coordinates/timestamp for velocity"

    recent = _recent_user_rows(user_history, now, limit=PAIRWISE_LOOKBACK)
    if recent.empty:
        return 0.0, 0.0, False, "No recent login history for pairwise velocity"

    max_speed = 0.0
    max_distance = 0.0
    max_prev_country = ""
    for _, prev in recent.iterrows():
        prev_ts = pd.to_datetime(prev.get("timestamp"), utc=True, errors="coerce")
        prev_lat = _to_float(prev.get("lat_coarse"))
        prev_lon = _to_float(prev.get("lon_coarse"))
        if pd.isna(prev_ts) or prev_lat is None or prev_lon is None:
            continue
        delta_h = (now - prev_ts).total_seconds() / 3600.0
        if delta_h <= 0:
            continue
        distance_km = haversine_km(prev_lat, prev_lon, cur_lat, cur_lon)
        speed_kmh = distance_km / delta_h if delta_h > 0 else 0.0
        if speed_kmh > max_speed:
            max_speed = float(speed_kmh)
            max_distance = float(distance_km)
            max_prev_country = str(prev.get("country") or "")

    score = float(min(1.0, max_speed / IMPOSSIBLE_TRAVEL_SPEED_KMH))
    fired = bool(max_distance > 0 and max_speed > IMPOSSIBLE_TRAVEL_SPEED_KMH)
    evidence = (
        f"max_speed_kmh={max_speed:.1f} over distance_km={max_distance:.1f}"
        + (f" from {max_prev_country}" if max_prev_country else "")
    )
    return score, max_speed, fired, evidence


def _country_rarity_for_user(
    country: str,
    user_history: pd.DataFrame,
) -> tuple[float, str, bool]:
    if not country:
        return 0.0, "Country missing", False
    if user_history.empty or "country" not in user_history.columns:
        return 0.8, f"First login from {country} for this user", True

    hist_countries = user_history["country"].fillna("").astype(str)
    hist_countries = hist_countries[hist_countries != ""]
    if hist_countries.empty:
        return 0.8, f"First login from {country} for this user", True

    counts = hist_countries.value_counts()
    total = int(counts.sum())
    seen = int(counts.get(country, 0))
    if seen == 0:
        return 0.8, f"First login from {country} for this user", True

    freq = seen / total if total > 0 else 1.0
    if freq < 0.05:
        return 0.4, f"Country {country} is rare for this user ({freq:.2%} history share)", False
    return 0.0, f"Country {country} is common for this user ({freq:.2%} history share)", False


def _country_rarity_global(
    country: str,
    global_history: pd.DataFrame,
) -> tuple[float, str]:
    if not country:
        return 0.0, "Country missing"
    if global_history.empty or "country" not in global_history.columns:
        return 0.6, f"Country {country} has no prior global history"

    global_countries = global_history["country"].fillna("").astype(str)
    global_countries = global_countries[global_countries != ""]
    if global_countries.empty:
        return 0.6, f"Country {country} has no prior global history"

    counts = global_countries.value_counts()
    total = int(counts.sum())
    seen = int(counts.get(country, 0))
    freq = seen / total if total > 0 else 1.0
    if freq < 0.01:
        return 0.6, f"Country {country} appears in {freq:.2%} of global logins"
    if freq < 0.05:
        return 0.3, f"Country {country} appears in {freq:.2%} of global logins"
    return 0.0, f"Country {country} is common globally ({freq:.2%})"


def _device_location_mismatch(
    event: dict[str, Any],
    user_history: pd.DataFrame,
    *,
    pairwise_impossible: bool,
) -> tuple[float, str]:
    if user_history.empty:
        return 0.0, "No prior login for device-location comparison"
    if "timestamp" not in user_history.columns:
        return 0.0, "No timestamped history for device-location comparison"

    hist = user_history.copy()
    hist["timestamp"] = pd.to_datetime(hist["timestamp"], utc=True, errors="coerce")
    hist = hist[hist["timestamp"].notna()].sort_values("timestamp", kind="mergesort")
    if hist.empty:
        return 0.0, "No prior login for device-location comparison"

    prev = hist.iloc[-1]
    cur_device = str(event.get("device_hash") or "")
    prev_device = str(prev.get("device_hash") or "")
    cur_country = str(event.get("country") or "")
    prev_country = str(prev.get("country") or "")

    cur_lat = _to_float(event.get("lat_coarse"))
    cur_lon = _to_float(event.get("lon_coarse"))
    prev_lat = _to_float(prev.get("lat_coarse"))
    prev_lon = _to_float(prev.get("lon_coarse"))
    distance_km = 0.0
    if cur_lat is not None and cur_lon is not None and prev_lat is not None and prev_lon is not None:
        distance_km = haversine_km(prev_lat, prev_lon, cur_lat, cur_lon)

    different_device = bool(cur_device and prev_device and cur_device != prev_device)
    different_country = bool(cur_country and prev_country and cur_country != prev_country)
    geo_shift = bool(different_country or distance_km > 200.0)

    if different_device and pairwise_impossible:
        return 0.95, "Different device with impossible travel speed"
    if different_device and different_country:
        return 0.9, "Different device AND different country from last login"
    if not different_device and geo_shift:
        return 0.2, "Same device with geographic shift (likely VPN)"
    return 0.0, "Same device and country pattern as prior login"


def _geo_session_break(
    country: str,
    user_history: pd.DataFrame,
) -> tuple[float, str]:
    if not country or user_history.empty or "country" not in user_history.columns:
        return 0.0, "No geo-session continuity baseline"

    hist = user_history.copy()
    if "timestamp" in hist.columns:
        hist["timestamp"] = pd.to_datetime(hist["timestamp"], utc=True, errors="coerce")
        hist = hist.sort_values("timestamp", kind="mergesort")
    countries = [str(c) for c in hist["country"].fillna("").astype(str).tolist() if c]
    if not countries:
        return 0.0, "No geo-session continuity baseline"

    last_country = countries[-1]
    streak = 0
    for c in reversed(countries):
        if c == last_country:
            streak += 1
        else:
            break

    if streak >= 3 and country != last_country:
        score = float(min(1.0, streak * 0.15))
        return score, f"Broke {streak}-login streak from {last_country}"
    return 0.0, f"Country continuity preserved (streak={streak} from {last_country})"


def _compute_geo_composite_for_event(
    event: dict[str, Any],
    user_history: pd.DataFrame,
    global_history: pd.DataFrame,
) -> GeoCompositeResult:
    country = str(event.get("country") or "")

    pair_score, pair_speed, pair_fired, pair_evidence = _pairwise_velocity_signal(event, user_history)
    user_country_score, user_country_evidence, first_country = _country_rarity_for_user(country, user_history)
    global_country_score, global_country_evidence = _country_rarity_global(country, global_history)
    mismatch_score, mismatch_evidence = _device_location_mismatch(
        event,
        user_history,
        pairwise_impossible=pair_fired,
    )
    session_break_score, session_break_evidence = _geo_session_break(country, user_history)

    composite = (
        (0.30 * pair_score)
        + (0.20 * user_country_score)
        + (0.10 * global_country_score)
        + (0.25 * mismatch_score)
        + (0.15 * session_break_score)
    )
    composite = float(max(0.0, min(1.0, composite)))
    composite_evidence = (
        "pairwise_geo_velocity={:.3f}; new_country_for_user={:.3f}; "
        "rare_country_global={:.3f}; device_location_mismatch={:.3f}; geo_session_break={:.3f}".format(
            pair_score,
            user_country_score,
            global_country_score,
            mismatch_score,
            session_break_score,
        )
    )

    return GeoCompositeResult(
        pairwise_geo_velocity_score=pair_score,
        pairwise_geo_velocity_kmh=pair_speed,
        pairwise_impossible=pair_fired,
        pairwise_evidence=pair_evidence,
        new_country_for_user_score=user_country_score,
        new_country_for_user_evidence=user_country_evidence,
        first_country_for_user=first_country,
        rare_country_global_score=global_country_score,
        rare_country_global_evidence=global_country_evidence,
        device_location_mismatch_score=mismatch_score,
        device_location_mismatch_evidence=mismatch_evidence,
        geo_session_break_score=session_break_score,
        geo_session_break_evidence=session_break_evidence,
        impossible_travel_composite_score=composite,
        impossible_travel_composite_evidence=composite_evidence,
    )


def compute_geo_composite_features(events_df: pd.DataFrame) -> pd.DataFrame:
    """Compute impossible-travel composite scores for a dataframe in timestamp order."""

    if events_df.empty:
        return pd.DataFrame(
            {
                "impossible_travel_composite_score": [],
                "geo_velocity_score": [],
                "impossible_travel_speed_kmh": [],
                "new_country_for_user_score": [],
                "rare_country_global_score": [],
                "device_location_mismatch_score": [],
                "geo_session_break_score": [],
            }
        )

    work = events_df.copy()
    work = work.reset_index(drop=False).rename(columns={"index": "_orig_idx"})
    work["timestamp"] = pd.to_datetime(
        work.get("timestamp", pd.Series([pd.NaT] * len(work), index=work.index)),
        utc=True,
        errors="coerce",
    )
    work["country"] = (
        work.get("country", pd.Series([""] * len(work), index=work.index))
        .fillna("")
        .astype(str)
    )
    work["device_hash"] = (
        work.get("device_hash", pd.Series([""] * len(work), index=work.index))
        .fillna("")
        .astype(str)
    )
    work["user_id"] = (
        work.get("user_id", pd.Series([""] * len(work), index=work.index))
        .fillna("")
        .astype(str)
    )
    work["lat_coarse"] = pd.to_numeric(
        work.get("lat_coarse", pd.Series([np.nan] * len(work), index=work.index)),
        errors="coerce",
    )
    work["lon_coarse"] = pd.to_numeric(
        work.get("lon_coarse", pd.Series([np.nan] * len(work), index=work.index)),
        errors="coerce",
    )
    work = work.sort_values(["timestamp", "_orig_idx"], kind="mergesort")

    n = len(work)
    comp = np.zeros(n, dtype=float)
    geo = np.zeros(n, dtype=float)
    speed = np.zeros(n, dtype=float)
    user_country = np.zeros(n, dtype=float)
    global_country = np.zeros(n, dtype=float)
    mismatch = np.zeros(n, dtype=float)
    session_break = np.zeros(n, dtype=float)

    user_recent: dict[str, deque[dict[str, Any]]] = defaultdict(lambda: deque(maxlen=PAIRWISE_LOOKBACK))
    user_country_counts: dict[str, dict[str, int]] = defaultdict(dict)
    user_country_totals: dict[str, int] = defaultdict(int)
    user_last_country: dict[str, str] = defaultdict(str)
    user_country_streak: dict[str, int] = defaultdict(int)
    global_country_counts: dict[str, int] = {}
    global_total = 0

    for row_idx, row in enumerate(work.itertuples(index=False)):
        user_id = str(getattr(row, "user_id", "") or "")
        country = str(getattr(row, "country", "") or "")
        device_hash = str(getattr(row, "device_hash", "") or "")
        now = pd.to_datetime(getattr(row, "timestamp", pd.NaT), utc=True, errors="coerce")
        cur_lat = _to_float(getattr(row, "lat_coarse", None))
        cur_lon = _to_float(getattr(row, "lon_coarse", None))

        # Signal 1: pairwise velocity against last 3-5 logins (using 5 here).
        max_speed = 0.0
        for prev in user_recent[user_id]:
            prev_ts = prev.get("timestamp")
            prev_lat = prev.get("lat_coarse")
            prev_lon = prev.get("lon_coarse")
            if pd.isna(now) or pd.isna(prev_ts) or cur_lat is None or cur_lon is None:
                continue
            if prev_lat is None or prev_lon is None:
                continue
            delta_h = (now - prev_ts).total_seconds() / 3600.0
            if delta_h <= 0:
                continue
            distance_km = haversine_km(prev_lat, prev_lon, cur_lat, cur_lon)
            speed_kmh = distance_km / delta_h if delta_h > 0 else 0.0
            if speed_kmh > max_speed:
                max_speed = float(speed_kmh)
        geo_score = float(min(1.0, max_speed / IMPOSSIBLE_TRAVEL_SPEED_KMH))
        impossible = bool(max_speed > IMPOSSIBLE_TRAVEL_SPEED_KMH)

        # Signal 2: country rarity for user.
        seen_user = int(user_country_counts[user_id].get(country, 0))
        total_user = int(user_country_totals[user_id])
        if country and seen_user == 0:
            user_country_score = 0.8
        elif country and total_user > 0 and (seen_user / total_user) < 0.05:
            user_country_score = 0.4
        else:
            user_country_score = 0.0

        # Signal 3: country rarity in global population.
        seen_global = int(global_country_counts.get(country, 0))
        if country and global_total > 0:
            global_freq = seen_global / global_total
            if global_freq < 0.01:
                global_country_score = 0.6
            elif global_freq < 0.05:
                global_country_score = 0.3
            else:
                global_country_score = 0.0
        elif country:
            global_country_score = 0.6
        else:
            global_country_score = 0.0

        # Signal 4: device + location mismatch.
        if user_recent[user_id]:
            prev = user_recent[user_id][-1]
            prev_country = str(prev.get("country") or "")
            prev_device = str(prev.get("device_hash") or "")
            different_device = bool(device_hash and prev_device and device_hash != prev_device)
            different_country = bool(country and prev_country and country != prev_country)
            geo_shift = different_country
            if cur_lat is not None and cur_lon is not None and prev.get("lat_coarse") is not None and prev.get("lon_coarse") is not None:
                geo_shift = geo_shift or (haversine_km(prev["lat_coarse"], prev["lon_coarse"], cur_lat, cur_lon) > 200.0)

            if different_device and impossible:
                mismatch_score = 0.95
            elif different_device and different_country:
                mismatch_score = 0.9
            elif (not different_device) and geo_shift:
                mismatch_score = 0.2
            else:
                mismatch_score = 0.0
        else:
            mismatch_score = 0.0

        # Signal 5: geo session continuity break.
        prev_country = user_last_country[user_id]
        prev_streak = int(user_country_streak[user_id])
        if country and prev_country and prev_streak >= 3 and country != prev_country:
            session_break_score = float(min(1.0, prev_streak * 0.15))
        else:
            session_break_score = 0.0

        comp_score = (
            (0.30 * geo_score)
            + (0.20 * user_country_score)
            + (0.10 * global_country_score)
            + (0.25 * mismatch_score)
            + (0.15 * session_break_score)
        )
        comp_score = float(max(0.0, min(1.0, comp_score)))

        comp[row_idx] = comp_score
        geo[row_idx] = geo_score
        speed[row_idx] = max_speed
        user_country[row_idx] = user_country_score
        global_country[row_idx] = global_country_score
        mismatch[row_idx] = mismatch_score
        session_break[row_idx] = session_break_score

        # Update trackers after computing current-event scores.
        if country:
            user_country_counts[user_id][country] = seen_user + 1
            user_country_totals[user_id] = total_user + 1
            global_country_counts[country] = seen_global + 1
            global_total += 1

            if prev_country == country:
                user_country_streak[user_id] = prev_streak + 1
            else:
                user_country_streak[user_id] = 1
            user_last_country[user_id] = country

        user_recent[user_id].append(
            {
                "timestamp": now,
                "lat_coarse": cur_lat,
                "lon_coarse": cur_lon,
                "country": country,
                "device_hash": device_hash,
            }
        )

    out = pd.DataFrame(
        {
            "_orig_idx": work["_orig_idx"].to_numpy(),
            "impossible_travel_composite_score": comp,
            "geo_velocity_score": geo,
            "impossible_travel_speed_kmh": speed,
            "new_country_for_user_score": user_country,
            "rare_country_global_score": global_country,
            "device_location_mismatch_score": mismatch,
            "geo_session_break_score": session_break,
        }
    )
    out = out.sort_values("_orig_idx", kind="mergesort").drop(columns=["_orig_idx"]).reset_index(drop=True)
    return out


def evaluate_geo_signals(
    event: dict[str, Any],
    user_history: Optional[pd.DataFrame] = None,
    global_history: Optional[pd.DataFrame] = None,
) -> list[dict[str, Any]]:
    user_history = _safe_history(user_history)
    global_history = _safe_history(global_history)

    country = str(event.get("country") or "")
    ip = str(event.get("ip") or "")
    asn = _ip_asn(event)

    comp = _compute_geo_composite_for_event(event, user_history, global_history)
    new_country = comp.first_country_for_user

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
        ip_velocity_count = int(
            same_ip.get("user_id", pd.Series(dtype=object)).astype(str).nunique()
        )
        failure_from_ip = int(
            (~same_ip.get("success", pd.Series(dtype=bool)).fillna(False)).sum()
        )
    ip_velocity = ip_velocity_count >= 8 or failure_from_ip >= 15

    residential_vs_datacenter = (
        "datacenter"
        if tor_vpn_proxy or ip.startswith("35.") or ip.startswith("34.")
        else "residential"
    )
    is_datacenter = residential_vs_datacenter == "datacenter"

    return [
        _signal(
            "impossible_travel",
            comp.pairwise_impossible,
            comp.pairwise_geo_velocity_score,
            comp.pairwise_evidence,
        ),
        _signal(
            "geo_velocity",
            comp.pairwise_geo_velocity_score >= 0.6,
            comp.pairwise_geo_velocity_score,
            comp.pairwise_evidence,
        ),
        _signal(
            "new_country",
            new_country,
            0.55 if new_country else 0.0,
            f"country={country}",
        ),
        _signal(
            "new_country_for_user",
            comp.new_country_for_user_score > 0.0,
            comp.new_country_for_user_score,
            comp.new_country_for_user_evidence,
        ),
        _signal(
            "rare_country_global",
            comp.rare_country_global_score > 0.0,
            comp.rare_country_global_score,
            comp.rare_country_global_evidence,
        ),
        _signal(
            "device_location_mismatch",
            comp.device_location_mismatch_score > 0.0,
            comp.device_location_mismatch_score,
            comp.device_location_mismatch_evidence,
        ),
        _signal(
            "geo_session_break",
            comp.geo_session_break_score > 0.0,
            comp.geo_session_break_score,
            comp.geo_session_break_evidence,
        ),
        _signal(
            "impossible_travel_composite",
            comp.impossible_travel_composite_score >= 0.45,
            comp.impossible_travel_composite_score,
            comp.impossible_travel_composite_evidence,
        ),
        _signal("new_asn", new_asn, 0.5 if new_asn else 0.0, f"asn={asn}"),
        _signal(
            "tor_vpn_proxy",
            tor_vpn_proxy,
            0.75 if tor_vpn_proxy else 0.0,
            f"ip={ip} asn={asn}",
        ),
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
