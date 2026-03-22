"""Signal extraction package for identity-risk-engine v0.2.0."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from .signals_behavior_ire import evaluate_behavior_signals
from .signals_device_ire import evaluate_device_signals
from .signals_geo_ire import evaluate_geo_signals
from .signals_passkey_ire import evaluate_passkey_signals
from .signals_recovery_ire import evaluate_recovery_signals


def evaluate_all_signals(
    event: Mapping[str, Any],
    user_history: pd.DataFrame | list[Mapping[str, Any]] | None = None,
    global_history: pd.DataFrame | list[Mapping[str, Any]] | None = None,
) -> list[dict[str, object]]:
    """Run all signal families and return a flattened list of signal outputs."""

    return [
        *evaluate_device_signals(event, user_history=user_history, global_history=global_history),
        *evaluate_geo_signals(event, user_history=user_history, global_history=global_history),
        *evaluate_behavior_signals(event, user_history=user_history, global_history=global_history),
        *evaluate_passkey_signals(event, user_history=user_history, global_history=global_history),
        *evaluate_recovery_signals(event, user_history=user_history, global_history=global_history),
    ]


__all__ = [
    "evaluate_all_signals",
    "evaluate_behavior_signals",
    "evaluate_device_signals",
    "evaluate_geo_signals",
    "evaluate_passkey_signals",
    "evaluate_recovery_signals",
]
