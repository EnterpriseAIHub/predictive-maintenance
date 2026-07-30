# Feature engineering, the model wrapper, and the SHAP explainer live
# here. This layer has no knowledge of HTTP, databases, or work
# orders — it is a pure function from features to (prediction,
# explanation), which is what makes it independently unit-testable and
# reusable identically from both the real-time and batch inference
# paths. Added in the phase that implements the model.
