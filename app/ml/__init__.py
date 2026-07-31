# Feature engineering, the model wrapper, and the SHAP explainer live
# here. This layer has no knowledge of HTTP, databases, or work
# orders — it is a pure function from features to (prediction,
# explanation), which is what makes it independently unit-testable and
# reusable identically from both the real-time and batch inference
# paths. Added in the phase that implements the model.
# Feature engineering (features.py) and the offline training pipeline
# (training/) are implemented. SHAP explainability (Phase 4) and the
# inference wrapper (Phase 5) come next.
 