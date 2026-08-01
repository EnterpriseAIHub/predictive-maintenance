"""Per-prediction explainability.

Uses SHAP's TreeExplainer to explain a prediction made by a LightGBM
model. Given one feature row, it returns the features that contributed
the most to the prediction.

This helps identify which sensor readings influenced the model the
most. It explains the model's decision, not the actual physical cause
of the failure.

This module only explains predictions. Loading or caching the model
is handled separately in the inference module (Phase 5).
"""

from typing import NamedTuple

import lightgbm as lgb
import pandas as pd
import shap

from app.ml.features import FEATURE_COLUMNS


# Stores the explanation for one feature.
class FeatureAttribution(NamedTuple):
    feature: str         
    shap_value: float    
    feature_value: float 


def explain_prediction(
    booster: lgb.Booster, feature_row: dict[str, float], top_n: int = 3
) -> list[FeatureAttribution]:
    """Returns the top features that influenced a single prediction."""

    row_df = pd.DataFrame([feature_row])[FEATURE_COLUMNS]

    # Create a SHAP explainer for the trained model.
    explainer = shap.TreeExplainer(booster)

    # Calculate SHAP values for this prediction.
    raw_shap_values = explainer.shap_values(row_df)

    # Different SHAP versions return different formats.
    # Convert everything into one 1D array of SHAP values.
    values = raw_shap_values[1] if isinstance(raw_shap_values, list) else raw_shap_values
    row_values = values[0]

    # Create an attribution object for every feature.
    attributions = [
        FeatureAttribution(
            feature=name,
            shap_value=float(value),
            feature_value=float(feature_row[name]),
        )
        for name, value in zip(FEATURE_COLUMNS, row_values, strict=True)
    ]

    # Sort features by highest contribution.
    attributions.sort(key=lambda a: abs(a.shap_value), reverse=True)

    # Return only the top N features.
    return attributions[:top_n]