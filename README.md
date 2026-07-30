# predictive-maintenance

Predicts near-term equipment failure risk from sensor data and traces
elevated risk back to its likely root cause. Part of the Enterprise AI
Platform (a GitHub-org multi-repo system) — this repo is fully
standalone: it can be cloned and run with no other platform repo
present.

Full design rationale lives in the project's technical design and
learning guide and the platform architecture document (not included
in this repo — see the platform-level docs).

## Status

**Data & Data Layer Foundation complete.** ORM models (`Equipment`,
`SensorReading`, `WorkOrder`, `RiskScore`), the first Alembic
migration, and repository functions for each are in place, with
integration tests running against a real Postgres instance. No
business logic, ML, or event publishing yet — those land in later
milestones.

## Database migrations

```bash
alembic upgrade head    # apply all migrations
alembic downgrade base  # roll back to empty schema
```

## Running locally

```bash
cp .env.example .env
docker-compose up --build
```

Then check:

```bash
curl http://localhost:8000/health        # liveness
curl http://localhost:8000/health/ready  # liveness + DB reachable
```

## Running tests

```bash
pip install -r requirements-dev.txt
pytest
```

## Project layout

```
app/
├── main.py            # FastAPI app assembly
├── config.py           # environment-sourced settings
├── logging_config.py   # structured JSON logging
├── api/                 # HTTP layer only — no business logic
├── schemas/             # Pydantic contracts (temporary local home
│                          for entities that will move to
│                          platform-data-contracts / platform-agent-sdk)
├── services/             # business logic (added in a later phase)
├── ml/                   # feature engineering, model, explainer (later)
├── data/                 # SQLAlchemy models and session management
├── events/               # Redis Streams publisher (later)
└── batch/                # nightly scoring job entrypoint (later)
alembic/                 # database migrations
tests/
model/registry/           # trained model artifacts (later)
```
