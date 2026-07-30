# Multi-stage build: install deps → copy app → run under uvicorn.
# The nightly batch job (added in a later phase) reuses this same
# image with a different container command
# (`python -m app.batch.nightly_score_job`) rather than a second
# Dockerfile — same code, different entrypoint, per the architecture
# doc's Docker strategy.

FROM python:3.12-slim AS base

WORKDIR /srv/app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app
COPY alembic ./alembic
COPY alembic.ini .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
