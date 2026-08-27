"""Runtime configuration, sourced from environment variables.

Values are read once at import time and reused everywhere via the
`settings` singleton below. 
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", protected_namespaces=())

    # Identity / environment
    app_name: str = "predictive-maintenance"
    environment: str = "local"  # local | ci | production
    log_level: str = "INFO"

    # Data stores
    database_url: str = (
        "postgresql+psycopg2://postgres:postgres@localhost:5432/predictive_maintenance"
    )
    redis_url: str = "redis://localhost:6379/0"

    # Business configuration (used starting in later phases; declared
    # now so the schema of "what this service is configured by" is
    # visible from day one)
    failure_risk_threshold: float = 0.7
    urgent_priority_threshold: float = 0.9
    prediction_window_days: int = 7

    # Feature engineering — the lookback window callers pass to
    # sensor_reading_repository.get_recent() before handing readings
    # to app.ml.features.build_feature_vector().
    feature_lookback_hours: int = 168  # 7 days

    # Model registry — the offline training pipeline (app/ml/training/)
    # writes here; the inference wrapper (app/ml/inference.py) reads
    # from here. One shared setting so neither side can drift out of
    # sync with a hardcoded path of its own.
    model_registry_dir: Path = REPO_ROOT / "model" / "registry"


@lru_cache
def get_settings() -> Settings:
    """Cached settings accessor — avoids re-parsing the environment on
    every call while still being test-friendly (lru_cache can be
    cleared between tests if a suite needs different env values)."""
    return Settings()


settings = get_settings()
