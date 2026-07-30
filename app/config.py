"""Runtime configuration, sourced from environment variables.

Values are read once at import time and reused everywhere via the
`settings` singleton below. Nothing in this repo should call
`os.environ` directly outside this module.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Identity / environment
    app_name: str = "predictive-maintenance"
    environment: str = "local"  # local | ci | production
    log_level: str = "INFO"

    # Data stores
    database_url: str = "postgresql+psycopg2://postgres:postgres@localhost:5432/predictive_maintenance"
    redis_url: str = "redis://localhost:6379/0"

    # Business configuration (used starting in later phases; declared
    # now so the schema of "what this service is configured by" is
    # visible from day one)
    failure_risk_threshold: float = 0.7
    prediction_window_days: int = 7


@lru_cache
def get_settings() -> Settings:
    """Cached settings accessor — avoids re-parsing the environment on
    every call while still being test-friendly (lru_cache can be
    cleared between tests if a suite needs different env values)."""
    return Settings()


settings = get_settings()
