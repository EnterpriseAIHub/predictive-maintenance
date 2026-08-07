from datetime import UTC, datetime, timedelta

import pandas as pd

from app.ml.features import FEATURE_COLUMNS, build_feature_vector

AS_OF = datetime(2026, 1, 8, tzinfo=UTC)
INSTALL_DATE = datetime(2024, 1, 1, tzinfo=UTC)


def _readings(rows: list[tuple[str, datetime, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["sensor_type", "timestamp", "value"])


def test_output_has_every_expected_column():
    features = build_feature_vector(_readings([]), INSTALL_DATE, 2, AS_OF)
    assert set(features.keys()) == set(FEATURE_COLUMNS)


def test_missing_sensor_type_produces_nan_mean_and_zero_trend():
    features = build_feature_vector(_readings([]), INSTALL_DATE, 2, AS_OF)
    assert pd.isna(features["temperature_rolling_mean"])
    assert features["temperature_rolling_std"] == 0.0
    assert features["temperature_rate_of_change"] == 0.0


def test_single_reading_has_no_trend_but_has_mean():
    rows = [("temperature", AS_OF - timedelta(hours=1), 70.0)]
    features = build_feature_vector(_readings(rows), INSTALL_DATE, 2, AS_OF)
    assert features["temperature_rolling_mean"] == 70.0
    assert features["temperature_rolling_std"] == 0.0
    assert features["temperature_rate_of_change"] == 0.0


def test_rate_of_change_is_per_hour_slope_between_first_and_last():
    rows = [
        ("temperature", AS_OF - timedelta(hours=10), 70.0),
        ("temperature", AS_OF - timedelta(hours=5), 75.0),
        ("temperature", AS_OF, 80.0),  # first->last: 10 degrees over 10 hours
    ]
    features = build_feature_vector(_readings(rows), INSTALL_DATE, 2, AS_OF)
    assert features["temperature_rolling_mean"] == 75.0
    assert features["temperature_rate_of_change"] == 1.0


def test_equipment_age_and_criticality_pass_through():
    features = build_feature_vector(_readings([]), INSTALL_DATE, 3, AS_OF)
    assert features["equipment_age_days"] == (AS_OF - INSTALL_DATE).days
    assert features["criticality_tier"] == 3
