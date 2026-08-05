from app.config import settings
from app.schemas.work_order import WorkOrderPriority
from app.services.risk_policy import evaluate_risk


def test_below_action_threshold_recommends_nothing():
    assert evaluate_risk(settings.failure_risk_threshold - 0.01) is None


def test_at_action_threshold_recommends_elevated():
    assert evaluate_risk(settings.failure_risk_threshold) == WorkOrderPriority.ELEVATED


def test_between_thresholds_recommends_elevated():
    midpoint = (settings.failure_risk_threshold + settings.urgent_priority_threshold) / 2
    assert evaluate_risk(midpoint) == WorkOrderPriority.ELEVATED


def test_at_urgent_threshold_recommends_urgent():
    assert evaluate_risk(settings.urgent_priority_threshold) == WorkOrderPriority.URGENT


def test_well_above_urgent_threshold_recommends_urgent():
    assert evaluate_risk(0.999) == WorkOrderPriority.URGENT