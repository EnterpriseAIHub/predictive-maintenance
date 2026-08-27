"""Per-prediction explainability.

Uses SHAP's TreeExplainer — fast and exact for gradient-boosted trees
"""

from typing import NamedTuple

import lightgbm as lgb
import pandas as pd
import shap

from app.ml.features import FEATURE_COLUMNS


class FeatureAttribution(NamedTuple):
    feature: str
    shap_value: float
    feature_value: float


def explain_prediction(
    booster: lgb.Booster, feature_row: dict[str, float], top_n: int = 3
) -> list[FeatureAttribution]:
    """Returns the `top_n` features contributing most to this single
    prediction, ranked by absolute SHAP value (largest push toward
    failure or away from it, either direction).
    """
    row_df = pd.DataFrame([feature_row])[FEATURE_COLUMNS]

    explainer = shap.TreeExplainer(booster)
    raw_shap_values = explainer.shap_values(row_df)

    # SHAP's return shape has varied across versions/model types
    # (ndarray vs. a list of per-class ndarrays for some classifiers).
    # Normalize to a single 1D array of per-feature values for this
    # one row rather than assuming one specific shape.
    values = raw_shap_values[1] if isinstance(raw_shap_values, list) else raw_shap_values
    row_values = values[0]

    attributions = [
        FeatureAttribution(
            feature=name, shap_value=float(value), feature_value=float(feature_row[name])
        )
        for name, value in zip(FEATURE_COLUMNS, row_values, strict=True)
    ]
    attributions.sort(key=lambda a: abs(a.shap_value), reverse=True)
    return attributions[:top_n]
