"""Offline training pipeline.

Run directly: `python -m app.ml.training.train`
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")  # no display available in this environment; write plots straight to file
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from app.config import settings
from app.ml.features import FEATURE_COLUMNS
from app.ml.training.dataset import generate_synthetic_training_data

REGISTRY_DIR = settings.model_registry_dir

_MODEL_PARAMS = {
    "objective": "binary",
    "is_unbalance": True,
    "num_leaves": 15,
    "learning_rate": 0.05,
    "n_estimators": 200,
    "random_state": 42,
    "verbosity": -1,
}


def _split_by_equipment(df):
    """Group split so no equipment unit appears in both train and
    test — evaluating on the same asset's other timestamps would
    overstate generalization (see the learning guide, §4)."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(splitter.split(df, groups=df["equipment_id"]))
    return df.iloc[train_idx], df.iloc[test_idx]


def _evaluate(model: lgb.LGBMClassifier, X_test, y_test) -> dict:
    probabilities = model.predict_proba(X_test)[:, 1]
    return {
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "pr_auc": float(average_precision_score(y_test, probabilities)),
        "positive_rate_test": float(y_test.mean()),
    }, probabilities


def _save_calibration_plot(y_test, probabilities, version_dir: Path) -> None:
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_test, probabilities, n_bins=10, strategy="quantile"
    )
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", label="perfectly calibrated")
    ax.plot(mean_predicted_value, fraction_of_positives, marker="o", label="model")
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed failure frequency")
    ax.set_title("Calibration curve")
    ax.legend()
    fig.savefig(version_dir / "calibration_curve.png", bbox_inches="tight")
    plt.close(fig)


def _write_model_card(
    version: str, version_dir: Path, metrics: dict, n_train: int, n_test: int
) -> None:
    card = f"""# Model Card — predictive-maintenance {version}

**Trained:** {datetime.now(UTC).isoformat()}
**Algorithm:** LightGBM binary classifier (gradient-boosted trees)
**Training data:** synthetic (see app/ml/training/dataset.py) — placeholder
for a real historical dataset; swapping the data source does not change
this pipeline's structure.

## Data split
Group split by `equipment_id` — no asset appears in both train and test.
- Train assets: {n_train}
- Test assets: {n_test}

## Metrics (test set)
- ROC-AUC: {metrics['roc_auc']:.3f}
- PR-AUC: {metrics['pr_auc']:.3f}
- Positive rate (test): {metrics['positive_rate_test']:.3f}

PR-AUC, not accuracy, is the metric that matters here — failures are
rare, so accuracy alone would look deceptively good on a model that
never flags anything (see the learning guide, §9).

## Feature columns (exact training order)
{chr(10).join(f"- {c}" for c in FEATURE_COLUMNS)}

## Known limitations
- Trained on synthetic data — metrics above describe pipeline
  correctness, not real-world model quality.
- No SHAP explainability yet (Phase 4).
- Not yet calibration-corrected if the calibration curve shows drift —
  see calibration_curve.png in this folder.
"""
    (version_dir / "model_card.md").write_text(card)


def _update_manifest(version: str, metrics: dict) -> None:
    manifest_path = REGISTRY_DIR / "manifest.json"
    manifest = {"latest": version, "versions": {}}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    manifest["latest"] = version
    manifest["versions"][version] = {
        "trained_at": datetime.now(UTC).isoformat(),
        "metrics": metrics,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))


def _publish_model_card_to_repo_root(version_dir: Path, repo_root: Path) -> None:
    """Mirrors the latest model card to MODEL_CARD.md at the repo root.
    model/registry/<version>/ is correct and complete but requires
    knowing (or looking up) a dynamic, timestamped version string to
    find — this gives it one stable, discoverable path, refreshed on
    every training run. Purely a documentation convenience (Phase 13);
    the versioned copy under model/registry/ remains the source of
    truth.
    """
    (repo_root / "MODEL_CARD.md").write_text((version_dir / "model_card.md").read_text())


def train_and_save() -> str:
    """Runs the full pipeline once and returns the new model version."""
    df = generate_synthetic_training_data()
    train_df, test_df = _split_by_equipment(df)

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df["label"]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df["label"]

    model = lgb.LGBMClassifier(**_MODEL_PARAMS)
    model.fit(X_train, y_train)

    metrics, probabilities = _evaluate(model, X_test, y_test)

    version = f"v{datetime.now(UTC):%Y%m%d%H%M%S}"
    version_dir = REGISTRY_DIR / version
    version_dir.mkdir(parents=True, exist_ok=True)

    model.booster_.save_model(str(version_dir / "model.txt"))
    (version_dir / "feature_columns.json").write_text(json.dumps(FEATURE_COLUMNS, indent=2))
    (version_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    _save_calibration_plot(y_test, probabilities, version_dir)
    _write_model_card(version, version_dir, metrics, len(train_df), len(test_df))
    _update_manifest(version, metrics)
    _publish_model_card_to_repo_root(version_dir, repo_root=REGISTRY_DIR.parent.parent)

    return version


if __name__ == "__main__":
    saved_version = train_and_save()
    print(f"Saved model {saved_version} to {REGISTRY_DIR / saved_version}")
