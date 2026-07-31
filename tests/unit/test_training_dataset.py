from app.ml.features import FEATURE_COLUMNS
from app.ml.training.dataset import generate_synthetic_training_data


def test_output_shape_and_columns():
    df = generate_synthetic_training_data(n_assets=50, random_seed=1)
    assert len(df) == 50
    assert set(FEATURE_COLUMNS) <= set(df.columns)
    assert {"label", "equipment_id"} <= set(df.columns)


def test_labels_are_binary_and_approximately_match_requested_rate():
    df = generate_synthetic_training_data(n_assets=200, positive_rate=0.1, random_seed=1)
    assert set(df["label"].unique()) <= {0, 1}
    # Label noise means this won't be exact — just in the right ballpark.
    assert 10 <= df["label"].sum() <= 35


def test_equipment_ids_are_unique():
    df = generate_synthetic_training_data(n_assets=30, random_seed=1)
    assert df["equipment_id"].nunique() == 30


def test_same_seed_is_reproducible():
    df1 = generate_synthetic_training_data(n_assets=20, random_seed=7)
    df2 = generate_synthetic_training_data(n_assets=20, random_seed=7)
    assert df1.equals(df2)


def test_degrading_assets_trend_higher_than_healthy_on_average():
    df = generate_synthetic_training_data(n_assets=200, positive_rate=0.5, random_seed=1)
    degrading = df[df["label"] == 1]["temperature_rolling_mean"]
    healthy = df[df["label"] == 0]["temperature_rolling_mean"]
    assert degrading.mean() > healthy.mean()
