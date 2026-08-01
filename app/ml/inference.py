"""The inference wrapper — the single ML-layer entry point the
service layer (Phase 6) calls. Combines three things that were
deliberately kept separate until now:

- loading a registered model (and validating it matches the current
  feature schema)
- producing a calibrated probability
- producing the SHAP-based explanation for that same prediction

Loading is cached at module level (`get_model()`), so the booster is
read from disk once per process, not once per request — the model
itself is loaded once at startup and reused, per NFR2 (inference
latency).
"""

import json
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import pandas as pd

from app.config import settings
from app.ml.explain import FeatureAttribution, explain_prediction
from app.ml.features import FEATURE_COLUMNS


class ModelNotFoundError(RuntimeError):
    """No trained, registered model is available to load."""


@dataclass(frozen=True)
class PredictionResult:
    probability: float
    model_version: str
    attributions: list[FeatureAttribution]


class PredictiveMaintenanceModel:
    """A loaded booster paired with the exact feature-column order it
    was trained on. Only ever constructed via `load_model()` — never
    directly — so every instance is guaranteed internally consistent.
    """

    def __init__(self, booster: lgb.Booster, feature_columns: list[str], version: str):
        if feature_columns != FEATURE_COLUMNS:
            raise ValueError(
                f"Model {version} was trained on a different feature schema than the "
                "current app.ml.features.FEATURE_COLUMNS — refusing to serve predictions "
                "that would silently misalign columns."
            )
        self.booster = booster
        self.feature_columns = feature_columns
        self.version = version

    def predict_proba(self, feature_row: dict[str, float]) -> float:
        row_df = pd.DataFrame([feature_row])[self.feature_columns]
        return float(self.booster.predict(row_df)[0])

    def predict_with_explanation(self, feature_row: dict[str, float], top_n: int = 3) -> PredictionResult:
        probability = self.predict_proba(feature_row)
        attributions = explain_prediction(self.booster, feature_row, top_n=top_n)
        return PredictionResult(
            probability=probability, model_version=self.version, attributions=attributions
        )


def load_model(
    version: str | None = None, registry_dir: Path = settings.model_registry_dir
) -> PredictiveMaintenanceModel:
    """Loads a specific registry version, or the manifest's 'latest'
    if none is given. Raises ModelNotFoundError with a clear message
    rather than letting a FileNotFoundError/KeyError leak upward —
    the service layer (Phase 6) will need to catch this specific type
    to return a meaningful error rather than a 500.
    """
    manifest_path = registry_dir / "manifest.json"
    if not manifest_path.exists():
        raise ModelNotFoundError(
            f"No model registry found at {registry_dir} — run the training pipeline first."
        )
    manifest = json.loads(manifest_path.read_text())

    resolved_version = version or manifest.get("latest")
    if not resolved_version or resolved_version not in manifest.get("versions", {}):
        raise ModelNotFoundError(f"Model version '{resolved_version}' not found in {manifest_path}.")

    version_dir = registry_dir / resolved_version
    model_path = version_dir / "model.txt"
    columns_path = version_dir / "feature_columns.json"
    if not model_path.exists() or not columns_path.exists():
        raise ModelNotFoundError(f"Model artifact files missing under {version_dir}.")

    booster = lgb.Booster(model_file=str(model_path))
    feature_columns = json.loads(columns_path.read_text())
    return PredictiveMaintenanceModel(booster, feature_columns, resolved_version)


# --- Load-once cache ---------------------------------------------------
# The API startup hook (Phase 8) calls get_model() once so every
# request reuses the same loaded booster instead of hitting disk again.

_cached_model: PredictiveMaintenanceModel | None = None


def get_model(force_reload: bool = False) -> PredictiveMaintenanceModel:
    global _cached_model
    if _cached_model is None or force_reload:
        _cached_model = load_model()
    return _cached_model


def reset_model_cache() -> None:
    """Test-only hook — clears the cached model so tests don't leak
    state into each other."""
    global _cached_model
    _cached_model = None