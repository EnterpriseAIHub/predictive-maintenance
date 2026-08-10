"""FastAPI application entrypoint.

Route handlers themselves stay in app/api/routes/ — this module only
assembles the app: logging, routers, error handlers, and the startup
hook that loads the registered model into memory once at process
start (per NFR2 — the model is loaded once, not reloaded per request).
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.error_handlers import register_error_handlers
from app.api.routes.agent import router as agent_router
from app.api.routes.health import router as health_router
from app.api.routes.prediction import router as prediction_router
from app.api.routes.work_orders import router as work_orders_router
from app.config import settings
from app.logging_config import configure_logging, get_logger
from app.ml.inference import ModelNotFoundError, get_model

configure_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    logger.info("service_starting", environment=settings.environment)
    try:
        model = get_model()
        logger.info("model_loaded", version=model.version)
    except ModelNotFoundError:
        # Don't crash the whole process if no model has been trained
        # yet — /health/ready will correctly report 503 until one is
        # registered, which is the intended way to surface this state.
        logger.warning("model_not_available_at_startup")
    yield
    logger.info("service_stopping")


app = FastAPI(title=settings.app_name, lifespan=lifespan)

register_error_handlers(app)

app.include_router(health_router)
app.include_router(prediction_router)
app.include_router(work_orders_router)
app.include_router(agent_router)
