import lightgbm as lgb

from app.ml.explain import explain_prediction
from app.ml.features import FEATURE_COLUMNS
from app.ml.training.dataset import generate_synthetic_training_data


# Train a small model that all tests can use
def _train_small_model():
    df = generate_synthetic_training_data(n_assets=100, random_seed=3)
    model = lgb.LGBMClassifier(objective="binary", n_estimators=50, random_state=3, verbosity=-1)
    model.fit(df[FEATURE_COLUMNS], df["label"])
    return model.booster_, df


# Check that explain_prediction returns the requested number of
# valid features and that they are ranked correctly.
def test_explanation_has_requested_top_n_and_valid_features():
    booster, df = _train_small_model()
    row = df.iloc[0][FEATURE_COLUMNS].to_dict()

    attributions = explain_prediction(booster, row, top_n=3)

    assert len(attributions) == 3
    assert all(a.feature in FEATURE_COLUMNS for a in attributions)

    # SHAP values should be sorted by importance
    magnitudes = [abs(a.shap_value) for a in attributions]
    assert magnitudes == sorted(magnitudes, reverse=True)


# Verify that SHAP explanations correctly reconstruct
# the model's original prediction.
def test_shap_values_satisfy_additivity_against_the_real_model():
    """Checks SHAP's additivity property on a real trained model."""
    import shap

    booster, df = _train_small_model()
    row_df = df.iloc[[0]][FEATURE_COLUMNS]

    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(row_df)
    values = shap_values[1] if isinstance(shap_values, list) else shap_values

    raw_prediction = booster.predict(row_df, raw_score=True)[0]
    reconstructed = explainer.expected_value + values[0].sum()

    assert abs(reconstructed - raw_prediction) < 1e-4


# Check that an extreme temperature value is identified
# as one of the most important features.
def test_feature_with_largest_magnitude_value_tends_to_dominate_a_clear_outlier():
    """Sanity check for feature importance."""
    booster, df = _train_small_model()
    row = df.iloc[0][FEATURE_COLUMNS].to_dict()

    # Create an obvious temperature anomaly
    row["temperature_rolling_mean"] = 500.0
    row["temperature_rate_of_change"] = 50.0

    attributions = explain_prediction(booster, row, top_n=3)

    top_features = {a.feature for a in attributions}
    assert "temperature_rolling_mean" in top_features or "temperature_rate_of_change" in top_features