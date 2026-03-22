"""identity-risk-engine public API."""

from .behavior_anomaly import BehaviorAnomalyScorer, UserBehaviorProfile
from .composite_scorer import CompositeRiskScorer, OperatingPoint
from .device_fingerprint import DeviceNoveltyScorer, UserDeviceProfile
from .events import AuthEvent, AuthEventType, event_to_row
from .explainer_ire import explain_scored_event
from .geo_velocity import (
    DEFAULT_MAX_AIRCRAFT_SPEED_KMH,
    GeoVelocityConfig,
    compute_geo_velocity_features,
    flag_impossible_travel,
    haversine_km,
)
from .policy_engine import PolicyEngine
from .risk_engine_ire import score_dataframe, score_event
from .simulator_ire import (
    ATTACK_TYPES_IRE,
    generate_synthetic_auth_events,
    train_test_time_split_ire,
)
from .synthetic_data_generator import (
    ATTACK_TYPES,
    generate_synthetic_login_data,
    train_test_time_split,
)

__all__ = [
    "ATTACK_TYPES",
    "ATTACK_TYPES_IRE",
    "AuthEvent",
    "AuthEventType",
    "BehaviorAnomalyScorer",
    "CompositeRiskScorer",
    "DEFAULT_MAX_AIRCRAFT_SPEED_KMH",
    "DeviceNoveltyScorer",
    "PolicyEngine",
    "GeoVelocityConfig",
    "OperatingPoint",
    "UserBehaviorProfile",
    "UserDeviceProfile",
    "compute_geo_velocity_features",
    "event_to_row",
    "explain_scored_event",
    "flag_impossible_travel",
    "generate_synthetic_auth_events",
    "generate_synthetic_login_data",
    "haversine_km",
    "score_dataframe",
    "score_event",
    "train_test_time_split",
    "train_test_time_split_ire",
]
