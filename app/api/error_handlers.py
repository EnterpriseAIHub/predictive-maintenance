"""Maps service-layer and ML-layer exceptions to HTTP responses.

Route handlers call service functions and let these exceptions
propagate — they don't catch them individually. This keeps the mapping
from "business exception" to "HTTP status code" in exactly one place,
so a new route never has to remember to handle
EquipmentNotFoundError correctly; it's handled globally.

Every handler here also logs — this is the centralized point where
"a request failed, and here's why" becomes visible in the log stream,
not just in the HTTP response.
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.logging_config import get_logger
from app.ml.inference import ModelNotFoundError
from app.services.errors import (
    EquipmentNotFoundError,
    InvalidApprovalError,
    WorkOrderNotFoundError,
)

logger = get_logger(__name__)


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(EquipmentNotFoundError)
    async def _equipment_not_found(_: Request, exc: EquipmentNotFoundError) -> JSONResponse:
        logger.warning("request_failed_equipment_not_found", detail=str(exc))
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(WorkOrderNotFoundError)
    async def _work_order_not_found(_: Request, exc: WorkOrderNotFoundError) -> JSONResponse:
        logger.warning("request_failed_work_order_not_found", detail=str(exc))
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(InvalidApprovalError)
    async def _invalid_approval(_: Request, exc: InvalidApprovalError) -> JSONResponse:
        # 409 Conflict: the request is well-formed, but the work order's
        # current state doesn't allow this action (already approved, or
        # no pending urgent recommendation).
        logger.warning("request_failed_invalid_approval", detail=str(exc))
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(ModelNotFoundError)
    async def _model_not_found(_: Request, exc: ModelNotFoundError) -> JSONResponse:
        # 503, not 500: this is a known, recoverable operational state
        # (no model has been trained/registered yet), not a bug. Logged
        # at error level (not warning) because — unlike a 404 for a
        # single bad request — this means EVERY prediction request is
        # currently failing, which is an operationally significant state.
        logger.error("request_failed_model_not_available", detail=str(exc))
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.exception_handler(Exception)
    async def _unhandled_exception(_: Request, exc: Exception) -> JSONResponse:
        # Catch-all for anything NOT one of the known business
        # exceptions above — a DB connection drop, a bug, anything
        # unanticipated. Two things matter here: (1) the full exception
        # is logged server-side with a traceback so it's diagnosable,
        # and (2) the client gets a generic message — never the raw
        # exception text, which could leak internal details (a query,
        # a file path, a stack frame) to an API caller.
        logger.error("unhandled_exception", error=str(exc), exc_info=True)
        return JSONResponse(status_code=500, content={"detail": "An internal error occurred."})
