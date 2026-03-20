"""identity-risk-engine public API."""

from .behavior_anomaly import BehaviorAnomalyScorer, UserBehaviorProfile
from .composite_scorer import CompositeRiskScorer, OperatingPoint
from .device_fingerprint import DeviceNoveltyScorer, UserDeviceProfile
from .geo_velocity import (
    DEFAULT_MAX_AIRCRAFT_SPEED_KMH,
    GeoVelocityConfig,
    compute_geo_velocity_features,
    flag_impossible_travel,
    haversine_km,
)
from .synthetic_data_generator import (
    ATTACK_TYPES,
    generate_synthetic_login_data,
    train_test_time_split,
)

__all__ = [
    "ATTACK_TYPES",
    "BehaviorAnomalyScorer",
    "CompositeRiskScorer",
    "DEFAULT_MAX_AIRCRAFT_SPEED_KMH",
    "DeviceNoveltyScorer",
    "GeoVelocityConfig",
    "OperatingPoint",
    "UserBehaviorProfile",
    "UserDeviceProfile",
    "compute_geo_velocity_features",
    "flag_impossible_travel",
    "generate_synthetic_login_data",
    "haversine_km",
    "train_test_time_split",
]
