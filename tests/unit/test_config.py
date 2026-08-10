"""Sanity checks on the settings themselves — not testing pydantic-
settings' own behavior, but catching a real, easy-to-make
configuration mistake: if urgent_priority_threshold were ever set
below failure_risk_threshold, risk_policy's logic would silently
produce nonsensical recommendations (URGENT below the action
threshold). This is cheap insurance against exactly that.
"""

from app.config import Settings, settings


def test_urgent_threshold_is_at_or_above_the_action_threshold():
    assert settings.urgent_priority_threshold >= settings.failure_risk_threshold


def test_feature_lookback_hours_is_positive():
    assert settings.feature_lookback_hours > 0


def test_model_registry_dir_is_configured():
    assert settings.model_registry_dir is not None


def test_settings_can_be_constructed_with_overrides():
    # Confirms the Settings class itself accepts explicit overrides
    # (e.g. how a test or a different environment would configure it)
    # without the protected_namespaces fix from Phase 5 regressing.
    custom = Settings(failure_risk_threshold=0.5, urgent_priority_threshold=0.8)
    assert custom.failure_risk_threshold == 0.5
    assert custom.urgent_priority_threshold == 0.8
