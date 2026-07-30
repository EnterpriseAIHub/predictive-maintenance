"""FastAPI application entrypoint.

Route handlers themselves stay in app/api/routes/ — this module only
assembles the app: logging, routers, and (in later phases) startup
hooks like loading the registered model into memory once at process
start (per NFR2 — the model is loaded once, not reloaded per request).
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes.health import router as health_router
from app.config import settings
from app.logging_config import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    logger.info("service_starting", environment=settings.environment)
    yield
    logger.info("service_stopping")


app = FastAPI(title=settings.app_name, lifespan=lifespan)

app.include_router(health_router)
