from app.api.routes import health as health_module
from app.ml.inference import ModelNotFoundError, PredictiveMaintenanceModel


def test_readiness_is_503_when_no_model_is_registered(client, monkeypatch):
    def _raise():
        raise ModelNotFoundError("no model registered")

    monkeypatch.setattr(health_module, "get_model", _raise)

    response = client.get("/health/ready")
    assert response.status_code == 503


def test_readiness_is_200_when_model_and_db_are_available(client, monkeypatch):
    # A minimal stand-in — readiness only needs get_model() to succeed,
    # it doesn't touch the model's methods.
    fake_model = object.__new__(PredictiveMaintenanceModel)
    monkeypatch.setattr(health_module, "get_model", lambda: fake_model)

    response = client.get("/health/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ready"}
