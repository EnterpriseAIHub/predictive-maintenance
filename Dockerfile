# Single-stage build: install deps → copy app + trained model → run
# under uvicorn. The nightly batch job reuses this same image with a
# different container command (`python -m app.batch.nightly_job`)
# rather than a second Dockerfile — same code, different entrypoint,
# per the architecture doc's Docker strategy.

#FROM python:3.12-slim AS base
FROM python:3.12-slim AS base

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv/app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

COPY requirements.txt requirements-ml.txt ./
RUN pip install -r requirements.txt -r requirements-ml.txt

COPY app ./app
COPY alembic ./alembic
COPY alembic.ini .

# Bakes the currently-trained model into the image (Phase 12 decision:
# simplest option for a standalone portfolio deployment — see
# DEPLOYMENT.md for the trade-off against an external model registry).
# Retraining means rebuilding this image, not a live hot-swap.
COPY model ./model

EXPOSE 8000

# Shell form (not exec form) so $PORT expands — Render assigns a
# dynamic port via this env var and health-checks against it; a
# hardcoded port would silently fail that health check in production
# while still working fine locally via docker-compose (where PORT is
# unset and the fallback of 8000 applies).
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
