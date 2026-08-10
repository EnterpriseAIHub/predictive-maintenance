# Predictive Maintenance

Predicts near-term equipment failure risk from sensor data, explains *why*
using SHAP, and routes high-risk predictions through a human-approval gate
before anything gets marked urgent. Part of a larger Enterprise AI Platform
(a GitHub-org multi-repo system) — this repo is fully standalone: it can be
cloned and run with no other platform repo present.

**Status: feature-complete and deployment-ready.** Real-time API, nightly
batch scoring, SHAP explainability, event publishing, structured
observability, 91 passing tests at 97% coverage, and CI — all built and
verified end to end. See [`PROJECT_HANDBOOK.md`](./PROJECT_HANDBOOK.md) for
the full technical deep dive (architecture, every design decision, every
bug found and fixed, interview prep) and
[`DEPLOYMENT.md`](./DEPLOYMENT.md) for deploying it.

## What makes this worth a closer look

- **Train/serve skew is structurally impossible, not just avoided.** One
  pure function (`build_feature_vector`) is the only feature-engineering
  code path — training, the real-time API, and the nightly batch job all
  call it, so there's no second implementation to drift out of sync.
- **SHAP correctness is actually tested**, not just wired up: a property
  test verifies `expected_value + Σ(shap_values) == the model's real
  prediction` against a genuinely trained model.
- **A real human-approval gate**, not a checkbox: the system can *never*
  persist `priority=URGENT` as a direct result of a prediction — only one
  function, called only by a human action, can do that.
- **Every bug in this repo's history is documented with its real root
  cause** in the handbook (Chapter 24) — including two found only by
  actually running the built Docker image rather than trusting a 91-test,
  97%-coverage suite to be sufficient proof of deployability.

## Architecture

```
API layer        →  Service layer          →  ML layer         →  Data layer
(FastAPI routes)    (business rules,           (feature eng,       (Postgres via
                     the approval gate,         SHAP, model         SQLAlchemy)
                     owns transactions)         inference)
                            ↓
                     Events layer (Redis Streams, best-effort)
```

Full layering rationale, every design decision, and the platform-wide
context in `PROJECT_HANDBOOK.md`, Chapters 3 and 20.

## Quick start

```bash
cp .env.example .env
docker compose up --build

# in another terminal, once the api service is healthy:
docker compose exec api alembic upgrade head
docker compose exec api python -m app.scripts.seed_demo_data

curl http://localhost:8000/health/ready
```

See [`DEMO.md`](./DEMO.md) for a guided walkthrough hitting every endpoint
against real demo data (a healthy asset, a degrading one, and one with zero
sensor history — deliberately included, not avoided).

## API

| Method & path | What it does |
|---|---|
| `GET /health` | Liveness — process is up |
| `GET /health/ready` | Readiness — DB reachable AND model loaded |
| `POST /predict` | Real-time prediction for one asset |
| `GET /equipment/{id}/risk` | Latest persisted risk score (no new prediction) |
| `POST /work-orders/{id}/approve` | The only path that can escalate a work order to URGENT |
| `POST /agent` | Platform-wide Agent Contract endpoint |

## Running tests

```bash
pip install -r requirements.txt -r requirements-ml.txt -r requirements-dev.txt
alembic upgrade head
pytest
```

91 tests across four categories (`unit`, `integration`, `model`,
`contract`) — see Chapter 15 of the handbook for what each covers and why.
CI (`.github/workflows/ci.yml`) runs the same checks against real Postgres
and Redis containers on every push.

## Training a model

```bash
python -m app.ml.training.train
```

Produces a new versioned entry under `model/registry/`, including a model
card. The current model's card is also mirrored to
[`MODEL_CARD.md`](./MODEL_CARD.md) at the repo root for a stable link —
`model/registry/manifest.json` tracks which version is `"latest"`.

## Project layout

```
app/
├── main.py              # FastAPI app assembly (lifespan preloads the model)
├── config.py             # environment-sourced settings
├── logging_config.py     # structured JSON logging
├── api/                   # HTTP layer — routes, error handling, no business logic
├── schemas/               # Pydantic contracts (Equipment, WorkOrder, Agent Contract)
├── services/               # business logic — risk policy, work orders, orchestration
├── ml/                     # feature engineering, training, SHAP, inference — no DB, no HTTP
├── data/                   # SQLAlchemy models and repositories
├── events/                 # Redis Streams publisher (best-effort)
├── batch/                  # nightly scoring job (reuses the real-time path)
└── scripts/                 # seed_demo_data.py
alembic/                   # database migrations
tests/                      # unit / integration / model / contract
model/registry/              # trained model artifacts, baked into the Docker image
render.yaml, DEPLOYMENT.md   # deployment configuration
PROJECT_HANDBOOK.md          # full technical + interview/resume reference
```

## Further reading

- [`PROJECT_HANDBOOK.md`](./PROJECT_HANDBOOK.md) — architecture, every
  phase's history, every bug found and fixed, 30+ interview questions,
  resume talking points.
- [`DEPLOYMENT.md`](./DEPLOYMENT.md) — deploying to Render, and the
  documented trade-offs (model baked into the image, no live retraining
  hot-swap).
- [`DEMO.md`](./DEMO.md) — a guided walkthrough against real demo data.
