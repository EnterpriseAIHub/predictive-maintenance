import json

from app.ml.training import train as train_module


def test_train_and_save_produces_a_usable_registry_entry(tmp_path, monkeypatch):
    # Redirect the registry to a temp dir so this test doesn't write
    # into the real model/registry/.
    monkeypatch.setattr(train_module, "REGISTRY_DIR", tmp_path)

    version = train_module.train_and_save()
    version_dir = tmp_path / version

    assert (version_dir / "model.txt").exists()
    assert (version_dir / "feature_columns.json").exists()
    assert (version_dir / "metrics.json").exists()
    assert (version_dir / "model_card.md").exists()
    assert (version_dir / "calibration_curve.png").exists()

    metrics = json.loads((version_dir / "metrics.json").read_text())
    # A random classifier's PR-AUC equals the positive rate; the
    # trained model should clearly beat that on this (easy, synthetic)
    # dataset — this is a real regression check, not a rubber stamp.
    assert metrics["pr_auc"] > metrics["positive_rate_test"]
    assert metrics["roc_auc"] > 0.65

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["latest"] == version
    assert version in manifest["versions"]
