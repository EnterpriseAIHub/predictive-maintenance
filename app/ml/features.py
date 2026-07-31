from datetime import datetime
import pandas as pd

# Sensor types expected by the ML model
EXPECTED_SENSOR_TYPES = (
    "temperature",
    "vibration",
    "pressure",
)

# Final order of features passed to the ML model
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
    """
    Converts raw sensor readings into ML features for one equipment.
    """

    # Store all calculated features here
    features: dict[str, float] = {}

    # Calculate features for each sensor type
    for sensor_type in EXPECTED_SENSOR_TYPES:

        # Get readings of only the current sensor and sort by time
        subset = readings[
            readings["sensor_type"] == sensor_type
        ].sort_values("timestamp")

        # Calculate mean, standard deviation and rate of change
        features[f"{sensor_type}_rolling_mean"] = _rolling_mean(subset)
        features[f"{sensor_type}_rolling_std"] = _rolling_std(subset)
        features[f"{sensor_type}_rate_of_change"] = _rate_of_change(subset)

    # Calculate equipment age in days
    features["equipment_age_days"] = float(
        (as_of - equipment_install_date).days
    )

    # Store equipment criticality
    features["criticality_tier"] = float(equipment_criticality_tier)

    return features


# Calculate average sensor value
def _rolling_mean(subset: pd.DataFrame) -> float:
    if subset.empty:
        return float("nan")      # No readings available

    return float(subset["value"].mean())


# Calculate variation in sensor values
def _rolling_std(subset: pd.DataFrame) -> float:
    if len(subset) < 2:
        return 0.0               # Need at least 2 readings

    return float(subset["value"].std(ddof=0))


# Calculate how fast sensor value is changing per hour
def _rate_of_change(subset: pd.DataFrame) -> float:
    if len(subset) < 2:
        return 0.0

    # Time difference between first and last reading
    elapsed_hours = (
        subset["timestamp"].iloc[-1]
        - subset["timestamp"].iloc[0]
    ).total_seconds() / 3600

    # Prevent division by zero
    if elapsed_hours <= 0:
        return 0.0

    # Change in value divided by time
    return float(
        (subset["value"].iloc[-1] - subset["value"].iloc[0])
        / elapsed_hours
    )