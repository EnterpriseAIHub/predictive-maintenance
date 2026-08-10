"""Shared feature engineering — the ONE function training and both
inference paths (real-time, batch) all call, which is what makes
train/serve skew impossible rather than just unlikely.

Pure function: takes plain data in, returns a plain dict out. No DB,
no HTTP, no ORM objects — that's what makes it testable in isolation
and reusable identically from every caller.
"""

from datetime import datetime

import pandas as pd

# Fixed set of sensor channels this model expects. A reading whose
# sensor_type isn't in this tuple is simply not used — new channels
# require an intentional model retrain, not a silent feature change.
EXPECTED_SENSOR_TYPES = ("temperature", "vibration", "pressure")

# Canonical, ordered feature list. Both training and inference build
# their feature rows in exactly this order — this constant is the
# actual train/serve contract, not just documentation of it.
FEATURE_COLUMNS = [
    f"{sensor}_{stat}"
    for sensor in EXPECTED_SENSOR_TYPES
    for stat in ("rolling_mean", "rolling_std", "rate_of_change")
] + ["equipment_age_days", "criticality_tier"]


def build_feature_vector(
    readings: pd.DataFrame,
    equipment_install_date: datetime,
    equipment_criticality_tier: int,
    as_of: datetime,
) -> dict[str, float]:
    """Builds one feature row for one equipment asset.

    `readings` must have columns [sensor_type, timestamp, value] and
    is assumed to already be filtered to the desired lookback window
    (the caller — real-time or batch — decides that window; this
    function doesn't re-filter).
    """
    features: dict[str, float] = {}

    for sensor_type in EXPECTED_SENSOR_TYPES:
        subset = readings[readings["sensor_type"] == sensor_type].sort_values("timestamp")
        features[f"{sensor_type}_rolling_mean"] = _rolling_mean(subset)
        features[f"{sensor_type}_rolling_std"] = _rolling_std(subset)
        features[f"{sensor_type}_rate_of_change"] = _rate_of_change(subset)

    features["equipment_age_days"] = float((as_of - equipment_install_date).days)
    features["criticality_tier"] = float(equipment_criticality_tier)

    return features


def _rolling_mean(subset: pd.DataFrame) -> float:
    if subset.empty:
        return float("nan")
    return float(subset["value"].mean())


def _rolling_std(subset: pd.DataFrame) -> float:
    if len(subset) < 2:
        return 0.0
    return float(subset["value"].std(ddof=0))


def _rate_of_change(subset: pd.DataFrame) -> float:
    if len(subset) < 2:
        return 0.0
    elapsed_hours = (
        subset["timestamp"].iloc[-1] - subset["timestamp"].iloc[0]
    ).total_seconds() / 3600
    if elapsed_hours <= 0:
        return 0.0
    return float((subset["value"].iloc[-1] - subset["value"].iloc[0]) / elapsed_hours)
