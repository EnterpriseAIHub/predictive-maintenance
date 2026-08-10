"""Training dataset construction.

There is no real historical sensor/failure history yet (that arrives
when a real dataset — e.g. a relabeled public degradation dataset —
is wired in). This module generates a synthetic but structurally
realistic stand-in: each simulated asset gets a sensor-reading history
and a failure label, run through the SAME `build_feature_vector` the
serving paths will use later, so the training pipeline below is
already exercising the real train/serve contract, not a shortcut
around it. Swapping this generator for a real data loader later
should not require any change to train.py — both must only produce a
DataFrame with FEATURE_COLUMNS + "label" + "equipment_id".
"""

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from app.ml.features import EXPECTED_SENSOR_TYPES, build_feature_vector

# Baseline sensor value and noise level for a HEALTHY asset, per
# sensor type. A DEGRADING asset drifts upward from this baseline
# over the simulated window, which is what gives the rolling-mean/
# rate-of-change features real signal to learn from. Drift magnitudes
# are kept small relative to noise deliberately — real degradation
# signal is subtle, and a synthetic dataset with clean separation
# between classes would make the pipeline's metrics meaningless (a
# perfect-looking model here would just mean the data was too easy,
# not that the pipeline works).
_BASELINE = {"temperature": (70.0, 2.0), "vibration": (0.5, 0.05), "pressure": (100.0, 3.0)}
_DEGRADED_DRIFT_PER_HOUR = {"temperature": 0.05, "vibration": 0.0012, "pressure": 0.06}

_READINGS_PER_ASSET = 48  # e.g. one reading every ~3.5 hours over the 7-day lookback


def _simulate_readings(
    as_of: datetime, degrading: bool, drift_factor: float, rng: np.random.Generator
) -> pd.DataFrame:
    """One asset's sensor history for the lookback window.

    `drift_factor` scales how strongly this specific asset follows
    its class's expected trend — degrading assets vary in how
    obviously they're degrading, and healthy assets occasionally
    drift mildly too, which is what keeps the two classes from being
    trivially separable.
    """
    timestamps = [
        as_of - timedelta(hours=h)
        for h in np.linspace(168, 0, _READINGS_PER_ASSET)  # 7-day window, oldest first
    ]
    rows = []
    for sensor_type in EXPECTED_SENSOR_TYPES:
        baseline, noise = _BASELINE[sensor_type]
        drift_per_hour = _DEGRADED_DRIFT_PER_HOUR[sensor_type] * drift_factor if degrading else 0.0
        for ts in timestamps:
            hours_elapsed = (as_of - ts).total_seconds() / 3600
            # drift grows as we approach `as_of` — degradation is most
            # visible right before the labeled failure point
            drift = drift_per_hour * (168 - hours_elapsed) if degrading else 0.0
            value = baseline + drift + rng.normal(0, noise)
            rows.append({"sensor_type": sensor_type, "timestamp": ts, "value": value})
    return pd.DataFrame(rows)


def generate_synthetic_training_data(
    n_assets: int = 400,
    positive_rate: float = 0.15,
    random_seed: int = 42,
    label_noise_rate: float = 0.08,
) -> pd.DataFrame:
    """Returns a DataFrame with FEATURE_COLUMNS + 'label' + 'equipment_id',
    one row per simulated asset. `label=1` means the asset was
    simulated as degrading toward failure within the prediction
    window; `label=0` is a healthy asset. `positive_rate` intentionally
    mirrors real predictive-maintenance class imbalance.

    `label_noise_rate` independently flips a fraction of labels after
    the sensor data is generated — real failures aren't perfectly
    predicted by the sensor pattern that precedes them (a different
    failure mode, a false alarm), and a synthetic dataset with zero
    label noise would let the model achieve unrealistic ROC-AUC/PR-AUC
    of 1.0, which would say nothing useful about the pipeline.
    """
    rng = np.random.default_rng(random_seed)
    as_of = datetime(2026, 1, 1, tzinfo=UTC)
    install_date = as_of - timedelta(days=730)  # fixed 2-year asset age for all synthetic units

    n_positive = int(round(n_assets * positive_rate))
    true_labels = np.array([1] * n_positive + [0] * (n_assets - n_positive))
    rng.shuffle(true_labels)

    rows = []
    for i, true_label in enumerate(true_labels):
        # Degrading assets vary from mildly to strongly degrading;
        # "healthy" assets get a small chance of mild drift too (e.g.
        # a sensor with unrelated noise) — this overlap is what makes
        # the classification problem realistically hard rather than
        # trivial.
        if true_label:
            drift_factor = float(rng.uniform(0.4, 1.6))
        else:
            drift_factor = float(rng.uniform(0, 0.3)) if rng.random() < 0.2 else 0.0

        readings = _simulate_readings(
            as_of,
            degrading=bool(true_label) or drift_factor > 0,
            drift_factor=drift_factor,
            rng=rng,
        )
        criticality_tier = int(rng.integers(1, 4))  # 1-3
        features = build_feature_vector(readings, install_date, criticality_tier, as_of)

        # The sensor data above is generated from the TRUE degradation
        # state; the stored label is independently noised, so the
        # features and the label can legitimately disagree — exactly
        # like a real dataset where the sensor signal is informative
        # but not perfectly predictive.
        observed_label = 1 - true_label if rng.random() < label_noise_rate else true_label

        features["label"] = int(observed_label)
        features["equipment_id"] = f"synthetic-eq-{i}"
        rows.append(features)

    return pd.DataFrame(rows)
