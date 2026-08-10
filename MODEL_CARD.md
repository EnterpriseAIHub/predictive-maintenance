# Model Card — predictive-maintenance v20260808122535

**Trained:** 2026-08-08T12:25:35.637970+00:00
**Algorithm:** LightGBM binary classifier (gradient-boosted trees)
**Training data:** synthetic (see app/ml/training/dataset.py) — placeholder
for a real historical dataset; swapping the data source does not change
this pipeline's structure.

## Data split
Group split by `equipment_id` — no asset appears in both train and test.
- Train assets: 300
- Test assets: 100

## Metrics (test set)
- ROC-AUC: 0.719
- PR-AUC: 0.608
- Positive rate (test): 0.180

PR-AUC, not accuracy, is the metric that matters here — failures are
rare, so accuracy alone would look deceptively good on a model that
never flags anything (see the learning guide, §9).

## Feature columns (exact training order)
- temperature_rolling_mean
- temperature_rolling_std
- temperature_rate_of_change
- vibration_rolling_mean
- vibration_rolling_std
- vibration_rate_of_change
- pressure_rolling_mean
- pressure_rolling_std
- pressure_rate_of_change
- equipment_age_days
- criticality_tier

## Known limitations
- Trained on synthetic data — metrics above describe pipeline
  correctness, not real-world model quality.
- No SHAP explainability yet (Phase 4).
- Not yet calibration-corrected if the calibration curve shows drift —
  see calibration_curve.png in this folder.
