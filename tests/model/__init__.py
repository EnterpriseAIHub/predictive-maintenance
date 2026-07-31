# Model-quality tests: run the real training pipeline end-to-end and
# assert on outcomes (metrics thresholds, artifact existence) rather
# than mocking pieces of it. Slower than unit tests by nature —
# training a real model — kept in their own category per the EDD's
# four test types (unit, model, integration, contract).
