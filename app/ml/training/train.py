"""Offline training pipeline.

Run directly:
    python -m app.ml.training.train

Trains the ML model, evaluates it, and saves everything
(model, metrics, feature list, plots, model card)
inside model/registry/.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import lightgbm as lgb
import matplotlib

# Use a non-GUI backend because training may run on servers.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit


from app.config import settings
from app.ml.features import FEATURE_COLUMNS
from app.ml.training.dataset import generate_synthetic_training_data

# Folder where every trained model version will be stored.
REGISTRY_DIR = settings.model_registry_dir

# Parameters used while training the LightGBM model.
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
    """
    Helper function:
    Splits the dataset into train and test sets while making sure
    the same equipment never appears in both.
    """
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.25,
        random_state=42
    )

    train_idx, test_idx = next(
        splitter.split(df, groups=df["equipment_id"])
    )

    return df.iloc[train_idx], df.iloc[test_idx]


def _evaluate(model: lgb.LGBMClassifier, X_test, y_test) -> dict:
    """
    Helper function:
    Evaluates the trained model and returns its metrics along
    with prediction probabilities.
    """

    # Probability that each equipment will fail.
    probabilities = model.predict_proba(X_test)[:, 1]

    return {
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "pr_auc": float(average_precision_score(y_test, probabilities)),
        "positive_rate_test": float(y_test.mean()),
    }, probabilities


def _save_calibration_plot(
    y_test,
    probabilities,
    version_dir: Path,
) -> None:
    """
    Helper function:
    Creates and saves a calibration curve to check whether the
    predicted probabilities match actual outcomes.
    """

    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_test,
        probabilities,
        n_bins=10,
        strategy="quantile",
    )

    fig, ax = plt.subplots(figsize=(5, 5))

    # Ideal prediction line.
    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="perfectly calibrated",
    )

    # Model prediction line.
    ax.plot(
        mean_predicted_value,
        fraction_of_positives,
        marker="o",
        label="model",
    )

    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed failure frequency")
    ax.set_title("Calibration curve")
    ax.legend()

    fig.savefig(
        version_dir / "calibration_curve.png",
        bbox_inches="tight",
    )

    plt.close(fig)


def _write_model_card(
    version: str,
    version_dir: Path,
    metrics: dict,
    n_train: int,
    n_test: int,
) -> None:
    """
    Helper function:
    Generates a Markdown report describing the trained model.
    """

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
never flags anything.

## Feature columns (exact training order)
{chr(10).join(f"- {c}" for c in FEATURE_COLUMNS)}

## Known limitations
- Trained on synthetic data.
- No SHAP explainability yet.
- Calibration may require improvement.
"""

    (version_dir / "model_card.md").write_text(card)


def _update_manifest(
    version: str,
    metrics: dict,
) -> None:
    """
    Helper function:
    Updates manifest.json so the project knows which model is
    the latest and stores basic information about every version.
    """

    manifest_path = REGISTRY_DIR / "manifest.json"

    # Create a new manifest if it doesn't exist.
    manifest = {
        "latest": version,
        "versions": {},
    }

    # Otherwise load the existing one.
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())

    manifest["latest"] = version

    manifest["versions"][version] = {
        "trained_at": datetime.now(UTC).isoformat(),
        "metrics": metrics,
    }

    manifest_path.write_text(
        json.dumps(manifest, indent=2)
    )


def train_and_save() -> str:
    """
    Main function:
    Runs the complete training pipeline from start to finish.
    """

    # Generate synthetic training dataset.
    df = generate_synthetic_training_data()

    # Split into training and testing sets.
    train_df, test_df = _split_by_equipment(df)

    # Separate features and labels.
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["label"]

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["label"]

    # Create and train the LightGBM model.
    model = lgb.LGBMClassifier(**_MODEL_PARAMS)
    model.fit(X_train, y_train)

    # Evaluate model performance.
    metrics, probabilities = _evaluate(
        model,
        X_test,
        y_test,
    )

    # Create a unique version name using the current timestamp.
    version = f"v{datetime.now(UTC):%Y%m%d%H%M%S}"

    version_dir = REGISTRY_DIR / version
    version_dir.mkdir(parents=True, exist_ok=True)

    # Save everything needed to reuse this model later.
    model.booster_.save_model(str(version_dir / "model.txt"))
    (version_dir / "feature_columns.json").write_text(
        json.dumps(FEATURE_COLUMNS, indent=2)
    )
    (version_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2)
    )

    _save_calibration_plot(
        y_test,
        probabilities,
        version_dir,
    )

    _write_model_card(
        version,
        version_dir,
        metrics,
        len(train_df),
        len(test_df),
    )

    _update_manifest(version, metrics)

    return version


if __name__ == "__main__":
    # Run the complete pipeline if this file is executed directly.
    saved_version = train_and_save()

    print(
        f"Saved model {saved_version} to {REGISTRY_DIR / saved_version}"
    )