import pytest

from app.ml import inference as inference_module
from app.ml.features import FEATURE_COLUMNS
from app.ml.inference import (
    ModelNotFoundError,
    PredictiveMaintenanceModel,
    load_model,
)
from app.ml.training import train as train_module
from app.ml.training.dataset import generate_synthetic_training_data


@pytest.fixture
def trained_registry(tmp_path, monkeypatch):
    """Trains a real (small) model into a temp registry dir, exactly
    like Phase 3's own test does — this phase's tests load that real
    artifact rather than mocking the file system.
    """
    monkeypatch.setattr(train_module, "REGISTRY_DIR", tmp_path)
    train_module.train_and_save()
    return tmp_path


def test_load_model_loads_the_manifest_latest_version(trained_registry):
    model = load_model(registry_dir=trained_registry)
    assert isinstance(model, PredictiveMaintenanceModel)
    assert model.feature_columns == FEATURE_COLUMNS


def test_predict_with_explanation_returns_a_sane_result(trained_registry):
    model = load_model(registry_dir=trained_registry)
    df = generate_synthetic_training_data(n_assets=1, random_seed=5)
    row = df.iloc[0][FEATURE_COLUMNS].to_dict()

    result = model.predict_with_explanation(row, top_n=3)

    assert 0.0 <= result.probability <= 1.0
    assert result.model_version == model.version
    assert len(result.attributions) == 3


def test_load_model_raises_when_registry_is_empty(tmp_path):
    with pytest.raises(ModelNotFoundError):
        load_model(registry_dir=tmp_path)


def test_load_model_raises_for_unknown_version(trained_registry):
    with pytest.raises(ModelNotFoundError):
        load_model(version="v-does-not-exist", registry_dir=trained_registry)


def test_model_construction_rejects_a_mismatched_feature_schema():
    with pytest.raises(ValueError):
        PredictiveMaintenanceModel(booster=None, feature_columns=["wrong_column"], version="v1")


def test_get_model_caches_across_calls(monkeypatch):
    call_count = {"n": 0}

    def fake_load_model(*args, **kwargs):
        call_count["n"] += 1
        return object()

    monkeypatch.setattr(inference_module, "load_model", fake_load_model)
    inference_module.reset_model_cache()

    first = inference_module.get_model()
    second = inference_module.get_model()
    assert first is second
    assert call_count["n"] == 1

    inference_module.get_model(force_reload=True)
    assert call_count["n"] == 2

    inference_module.reset_model_cache()
