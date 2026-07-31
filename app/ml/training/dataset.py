from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

# Import sensor names and feature engineering function
from app.ml.features import EXPECTED_SENSOR_TYPES, build_feature_vector


# Default sensor values for a healthy machine
# (average value, noise)
_BASELINE = {
    "temperature": (70.0, 2.0),
    "vibration": (0.5, 0.05),
    "pressure": (100.0, 3.0),
}

# How much each sensor increases every hour for a degrading machine
_DEGRADED_DRIFT_PER_HOUR = {
    "temperature": 0.05,
    "vibration": 0.0012,
    "pressure": 0.06,
}

# Total sensor readings generated for one machine
_READINGS_PER_ASSET = 48


def _simulate_readings(
    as_of: datetime,
    degrading: bool,
    drift_factor: float,
    rng: np.random.Generator,
) -> pd.DataFrame:

    """Generate fake sensor history for ONE machine."""

    # Create timestamps for the last 7 days
    timestamps = [
        as_of - timedelta(hours=h)
        for h in np.linspace(168, 0, _READINGS_PER_ASSET)
    ]

    rows = []

    # Generate readings for every sensor
    for sensor_type in EXPECTED_SENSOR_TYPES:

        baseline, noise = _BASELINE[sensor_type]

        # Only degrading machines have increasing sensor values
        drift_per_hour = (
            _DEGRADED_DRIFT_PER_HOUR[sensor_type] * drift_factor
            if degrading
            else 0.0
        )

        # Create sensor value for every timestamp
        for ts in timestamps:

            hours_elapsed = (
                as_of - ts
            ).total_seconds() / 3600

            # Drift becomes larger near the current time
            drift = (
                drift_per_hour * (168 - hours_elapsed)
                if degrading
                else 0.0
            )

            # Final sensor value
            # = healthy value
            # + degradation
            # + random sensor noise
            value = baseline + drift + rng.normal(0, noise)

            rows.append({
                "sensor_type": sensor_type,
                "timestamp": ts,
                "value": value,
            })

    return pd.DataFrame(rows)


def generate_synthetic_training_data(

    # Number of fake machines
    n_assets: int = 400,

    # Percentage of failed machines
    positive_rate: float = 0.15,

    # Makes random generation reproducible
    random_seed: int = 42,

    # Percentage of labels to flip intentionally
    label_noise_rate: float = 0.08,

) -> pd.DataFrame:

    """Generate fake dataset for ML model training."""

    # Random number generator
    rng = np.random.default_rng(random_seed)

    # Assume today's date
    as_of = datetime(2026, 1, 1, tzinfo=UTC)

    # Assume every machine is 2 years old
    install_date = as_of - timedelta(days=730)

    # Number of failure machines
    n_positive = int(round(n_assets * positive_rate))

    # Create labels
    # 1 = failure
    # 0 = healthy
    true_labels = np.array(
        [1] * n_positive +
        [0] * (n_assets - n_positive)
    )

    # Shuffle labels randomly
    rng.shuffle(true_labels)

    rows = []

    # Create one machine at a time
    for i, true_label in enumerate(true_labels):

        # Failed machines get stronger drift
        if true_label:
            drift_factor = float(rng.uniform(0.4, 1.6))

        # Healthy machines mostly have no drift
        # but 20% get a tiny drift
        else:
            drift_factor = (
                float(rng.uniform(0, 0.3))
                if rng.random() < 0.2
                else 0.0
            )

        # Generate fake sensor readings
        readings = _simulate_readings(
            as_of,
            degrading=bool(true_label) or drift_factor > 0,
            drift_factor=drift_factor,
            rng=rng,
        )

        # Random criticality level
        criticality_tier = int(rng.integers(1, 4))

        # Convert sensor history into ML features
        features = build_feature_vector(
            readings,
            install_date,
            criticality_tier,
            as_of,
        )

        # Add realistic mistakes by flipping some labels
        observed_label = (
            1 - true_label
            if rng.random() < label_noise_rate
            else true_label
        )

        # Add target column
        features["label"] = int(observed_label)

        # Give each machine a unique ID
        features["equipment_id"] = f"synthetic-eq-{i}"

        # Save this machine
        rows.append(features)

    # Return complete dataset
    return pd.DataFrame(rows)