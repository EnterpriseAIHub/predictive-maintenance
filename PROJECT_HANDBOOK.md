# Predictive Maintenance — Project Handbook

**Status as of this update:** Phases 0–12 complete (project foundation,
data layer, feature engineering, training pipeline, explainability,
inference wrapper, service layer, event publishing, real-time API, batch
scoring job, observability hardening, full test suite & CI, deployment
packaging). Phase 13 (documentation & portfolio polish) is documented
under **Future Scope** — not yet implemented, and unlike every other
phase, doesn't map to any FR/NFR — see Chapter 6.

This document is the single source of truth for the project: technical reference,
developer handbook, and interview/resume preparation guide in one file. It is
updated — never recreated — after every implementation phase.

---

## Table of Contents

1. Project Overview
2. End-to-End System Workflow
3. Complete Architecture
4. Folder-by-Folder Explanation
5. Technology Stack
6. Phase-by-Phase Implementation History
7. Database Design
8. Feature Engineering
9. Machine Learning Pipeline
10. Explainability (SHAP)
11. Inference Pipeline
12. Service Layer
13. Repository Layer
14. API Layer
15. Testing
16. Docker
17. Configuration
18. Logging
19. Design Patterns
20. Major Architectural Decisions
21. Scalability
22. Security
23. Production Improvements
24. Common Bugs and Debugging Notes
25. Interview Questions
26. HR Interview Questions
27. Resume Talking Points
28. Project Explanation Scripts
29. Important Things to Remember
30. Final Revision Notes

---

## 1. Project Overview

### The business problem

Manufacturing plants run equipment on one of three strategies: **reactive**
(fix it when it breaks), **preventive** (fix it on a fixed calendar regardless
of actual condition), or **predictive** (fix it when the data says it's about
to fail). Reactive causes unplanned downtime, rushed diagnosis, and emergency
parts shipping. Preventive wastes labor and parts maintaining healthy
equipment early, and still misses failures that happen inside the calendar
gap. Predictive is strictly more information-efficient than either — but it
only works if raw sensor data can be turned into a probability a human
actually trusts.

### What this project does

Given a piece of equipment's recent sensor history, the system:

1. Computes a set of engineered features describing recent behavior.
2. Predicts the probability that the asset fails within a defined window
   (currently 7 days, configurable).
3. Explains *why* — which sensors are driving that prediction (SHAP).
4. Decides whether the risk is high enough to act on, and if so, opens a
   work order — with a built-in human-approval gate before anything gets
   marked urgent.

### Business value

- Reduced unplanned downtime (the headline metric).
- Reduced emergency labor/parts cost (no more rush shipping under pressure).
- Better labor planning (a 7-day warning changes how a planner schedules
  work, versus a same-day reaction).
- Faster root-cause diagnosis — the SHAP attribution narrows "something is
  wrong" down to "these two sensors, on this specific asset," even though a
  human still makes the final call.

### Expected users

- **Maintenance technician** — needs to know which asset and why, in
  language they can act on.
- **Maintenance planner/supervisor** — needs prioritized, explainable risk
  across all equipment to plan labor and parts ahead of time.
- **Platform orchestrator** *(future — see platform-level docs)* — a
  separate system that composes this repo's output with other repos' in
  a daily cross-department briefing.

### Overall workflow (today)

```
sensor readings (DB) --> feature engineering --> trained model
                                                       |
                                                       v
                                          probability + SHAP explanation
                                                       |
                                                       v
                                    risk policy (threshold check)
                                                       |
                                                       v
                              work order created (approval-gated if urgent)
```

This whole chain is exercised end-to-end today through
`app.services.prediction_service.run_prediction_for_equipment()` — but it is
only reachable by calling that function directly (e.g. from a test or a
Python shell). As of Phase 8, it's also reachable via `POST /predict` and
the Agent Contract endpoint — see Chapter 14.

---

## 2. End-to-End System Workflow

### Current state (Phases 0–12): two entry points, one shared pipeline

```
Real-time: POST /predict, POST /agent          Batch: nightly_job.run_nightly_scoring()
    |                                                       |
    v                                                       v
              prediction_service.run_prediction_for_equipment(db, equipment_id, as_of, source)
    |
    |--> equipment_repository.get_by_id()         [does the asset exist? logs a warning if not]
    |--> sensor_reading_repository.get_recent()    [pull the lookback window]
    |--> build_feature_vector()                    [raw readings -> features; NaN is valid — see Ch.8]
    |--> get_model()                               [load-once cached model]
    |--> model.predict_with_explanation()          [probability + SHAP]
    |--> risk_score_repository.create()            [persist EVERY prediction, tagged source=REAL_TIME|BATCH]
    |--> create_work_order_for_prediction()         [risk_policy + approval gate]
    |--> db.commit()                                [atomic: score + work order together]
    |--> logger.info("prediction_scored", ...)      [structured log — Ch.18]
    |--> publish_equipment_failure_risk()          [async event; if Redis down, logged & dropped, write durable]
    v
PredictionOutcome(probability, model_version, attributions, work_order)
```

The batch job (`app/batch/nightly_job.py`, Phase 9) is not a second
implementation — it's a loop over every equipment asset calling this exact
same function with `source=RiskScoreSource.BATCH`. There is no separate
"batch scoring logic" anywhere in the codebase to keep in sync with the
real-time path.

### Stage-by-stage explanation

**1. Equipment lookup.** Fails fast (`EquipmentNotFoundError`) if the asset
doesn't exist — nothing downstream should run against an asset that isn't
real. Logs `prediction_requested_for_unknown_equipment` before raising
(Phase 10), so this is visible in the log stream, not just the HTTP 404.

**2. Sensor reading retrieval.** Pulls every reading in the configured
lookback window (`settings.feature_lookback_hours`, default 168h/7 days) for
that one asset, oldest-first. Zero readings is a valid state (see Chapter 24,
Bug 5 and Bug 9) — a brand-new asset, or one queried outside its data's
range, doesn't crash; it produces `NaN` features honestly representing "no
data."

**3. Feature engineering.** `build_feature_vector()` is a **pure function** —
no DB, no ORM objects in or out. It takes a plain DataFrame of readings plus
the asset's install date and criticality tier, and returns a flat dict of
features in a fixed, named order (`FEATURE_COLUMNS`). This exact function is
called by the real-time path, the nightly batch job, AND the training
pipeline — that's what makes train/serve skew structurally impossible
rather than just "unlikely."

**4. Model inference.** `get_model()` returns a cached, already-loaded
booster (loaded once per process, not once per call). `predict_with_explanation()`
returns both a calibrated probability and the SHAP attribution for that
exact prediction, in one call.

**5. Persistence of the prediction.** Every prediction is written to
`risk_score`, regardless of whether it crosses any action threshold, tagged
with `source` (`REAL_TIME` or `BATCH`) so the two paths' history stays
distinguishable — this history is what a future model-drift check and the
orchestrator's daily briefing would both read from.

**6. Risk policy + work order.** A pure function (`risk_policy.evaluate_risk`)
decides whether the probability warrants a work order at all, and at what
recommended priority. If it does, `work_order_service.create_work_order_for_prediction`
creates it — holding any `URGENT` recommendation at `ELEVATED` until a human
explicitly approves it (see Chapter 12 and Chapter 20 for why). Logs
`work_order_created` with `held_pending_approval` flagged (Phase 10) — an
audit trail for FR4, not just a database row.

**7. Commit.** The service layer — not the repositories — owns the
transaction boundary. The risk score and any resulting work order commit
together, atomically, in one call.

**8. Structured logging (Phase 10).** `prediction_scored` is logged after
every prediction — equipment, probability, model version, whether a work
order was created. This is the "why did the system do this" record; see
Chapter 18.

**9. Event publishing (Phase 7).** Best-effort — a Redis outage is logged
(`event_publish_failed`, Phase 10) but never fails the request. The work
order and risk score are already durably committed by this point.

### Future Scope

- **Phase 13 (documentation & polish):** the only remaining phase — and
  unlike every phase above, it adds no runtime capability. See Chapter 6
  for why it exists anyway.

---

## 3. Complete Architecture

### Layered architecture

```
┌─────────────────────────────────────────────────────┐
│  API layer          (app/api/)         [Phase 0: health only]
├─────────────────────────────────────────────────────┤
│  Service layer       (app/services/)   [Phase 6]
│    - risk_policy.py       (pure business rule)
│    - work_order_service.py (work orders + approval gate)
│    - prediction_service.py (orchestration, owns commit)
├─────────────────────────────────────────────────────┤
│  ML layer            (app/ml/)         [Phases 2-5]
│    - features.py          (pure feature engineering)
│    - training/             (offline pipeline)
│    - explain.py           (SHAP)
│    - inference.py         (load-once model wrapper)
├─────────────────────────────────────────────────────┤
│  Data layer           (app/data/)      [Phase 1]
│    - models/               (SQLAlchemy ORM)
│    - repositories/          (CRUD, no business rules)
├─────────────────────────────────────────────────────┤
│  Events layer          (app/events/)   [Phase 7 — Redis Streams publisher]
│  Batch layer           (app/batch/)    [Phase 9 — nightly scoring job]
└─────────────────────────────────────────────────────┘
```

### Why layered, and why THIS split

Each layer has exactly one reason to change. The API layer changes when the
external HTTP contract changes. The service layer changes when a business
rule changes (e.g., the urgent-priority threshold). The ML layer changes
when the modeling approach changes. The data layer changes when the schema
changes. Mixing these — e.g., putting threshold logic inside an API route
handler — makes every change riskier, because touching one concern risks
breaking an unrelated one.

A concrete test used throughout this project: *"if I swapped the model
algorithm entirely, how many files would I need to touch?"* The answer
should be confined to `app/ml/` — and it is, because the service layer only
knows about `get_model()` and `predict_with_explanation()`, never about
LightGBM specifically.

### Dependency flow (who is allowed to import whom)

```
api        --> services --> ml, data
services   --> ml, data
ml         --> (nothing above it; features.py has ZERO knowledge of DB/HTTP)
data       --> (nothing above it)
```

The ML layer is the strictest boundary in the codebase: `app/ml/features.py`
and `app/ml/explain.py` take and return plain Python/pandas objects only —
no SQLAlchemy session, no ORM row, ever crosses into that layer. The one
place that boundary is deliberately crossed is
`prediction_service._readings_to_dataframe()`, which exists specifically to
convert ORM rows into the plain DataFrame shape the ML layer expects — kept
in the service layer, not the ML layer, so the ML layer stays reusable and
independently testable.

### Advantages of this architecture

- Each layer is independently unit-testable (the ML layer needs no
  database at all; the pure `risk_policy` needs neither a database nor a
  trained model).
- A new consumer (the future API, the future batch job) doesn't duplicate
  logic — it just calls `prediction_service.run_prediction_for_equipment()`.
- Swapping an implementation detail (a different gradient boosting library,
  a different database) is contained to one layer.

### Trade-offs

- More files and more indirection than a single-script approach — a fair
  cost for a project meant to demonstrate production-style organization,
  but genuinely more ceremony than a one-off script would need.
- Cross-layer changes (like Phase 6's `WorkOrder` schema change, which
  touched the ORM model, the Pydantic schema, and a new migration) require
  coordinated updates across layers rather than one file edit.

### Industry comparison

This mirrors a fairly standard "clean architecture" / "hexagonal
architecture" style layering used in many production Python services:
domain logic (service layer) has no framework dependency, framework-specific
code (FastAPI routes) is a thin adapter at the edge, and infrastructure
(database, ML model loading) is behind an interface the domain layer calls
without knowing the implementation. It's the same underlying idea whether
the outer framework is FastAPI, Django, or Flask.

---

## 4. Folder-by-Folder Explanation

```
predictive-maintenance/
├── .github/workflows/
│   └── ci.yml                    Lint + migrate + full suite, real Postgres/Redis containers
├── app/
│   ├── main.py                FastAPI app assembly (lifespan preloads the model, routers, error handlers)
│   ├── config.py               Settings singleton (env-var sourced)
│   ├── logging_config.py       Structured JSON logging setup
│   ├── api/
│   │   ├── deps.py              Shared FastAPI dependencies (get_db)
│   │   ├── error_handlers.py    Central exception -> HTTP status mapping + catch-all 500 handler
│   │   └── routes/
│   │       ├── health.py         /health, /health/ready (also checks the model is loaded)
│   │       ├── prediction.py     /predict, /equipment/{id}/risk
│   │       ├── work_orders.py    /work-orders/{id}/approve — the only HTTP path to priority=URGENT
│   │       └── agent.py          /agent — the Agent Contract endpoint
│   ├── schemas/                 Pydantic contracts (API/cross-repo shape)
│   │   ├── equipment.py
│   │   ├── work_order.py
│   │   └── agent_contract.py    AgentRequest/AgentResponse (platform-wide)
│   ├── services/                 Business logic — no HTTP, no ORM sessions passed around casually
│   │   ├── errors.py
│   │   ├── risk_policy.py        Pure: probability -> recommended priority
│   │   ├── work_order_service.py  Work order creation + approval gate + logging
│   │   └── prediction_service.py  Orchestration; owns the transaction; logging; event publish
│   ├── ml/                       No DB, no HTTP — pure ML logic
│   │   ├── features.py            Shared feature engineering
│   │   ├── explain.py             SHAP attribution
│   │   ├── inference.py           Load-once model wrapper
│   │   └── training/
│   │       ├── dataset.py          Synthetic training data generator
│   │       └── train.py            Training pipeline entrypoint
│   ├── data/
│   │   ├── base.py                 Engine, session factory, declarative Base
│   │   ├── models/                  SQLAlchemy ORM classes
│   │   │   ├── equipment.py
│   │   │   ├── sensor_reading.py
│   │   │   ├── work_order.py
│   │   │   └── risk_score.py
│   │   └── repositories/             CRUD functions only, no business rules
│   ├── events/
│   │   └── publisher.py            Redis Streams publisher — best-effort, versioned event schemas
│   ├── batch/
│   │   └── nightly_job.py          Nightly scoring — reuses prediction_service, no separate logic
│   └── scripts/
│       └── seed_demo_data.py       Idempotent demo data for a fresh deployment
├── alembic/                          Database migrations
├── tests/
│   ├── unit/                          No DB, no external services
│   ├── integration/                    Real Postgres, savepoint-isolated
│   ├── model/                          Trains/loads real models — slower, real correctness checks
│   └── contract/                       Agent Contract shapes + schema/ORM parity (Phase 11)
├── model/registry/                     Versioned trained artifacts (baked into the Docker image)
├── Dockerfile, docker-compose.yml
├── render.yaml, DEPLOYMENT.md          Render Blueprint + deploy instructions (Phase 12)
├── requirements.txt / requirements-dev.txt / requirements-ml.txt
├── alembic.ini, pyproject.toml, .env.example
└── README.md
```

Each file's responsibility, in one line, is described directly inside that
file's module docstring — this handbook explains the *reasoning*; the code
explains the *contract*.

---

## 5. Technology Stack

| Technology | What | Why chosen here | Alternatives considered |
|---|---|---|---|
| **Python** | Language | ML ecosystem (pandas, LightGBM, SHAP) and a production web framework (FastAPI) both live natively in one language | None seriously — splitting ML/API languages adds a network boundary for no benefit at this scale |
| **FastAPI** | Web framework | Native Pydantic integration — request/response models double as validation and OpenAPI docs | Flask (no native validation), Django (too heavy for this size) |
| **PostgreSQL** | Relational DB | Structured, relational entities (Equipment has many SensorReadings); handles this data volume's time-series shape fine without a dedicated TSDB | MongoDB (data is relational, not document-shaped), a dedicated TSDB (real infra you don't need at this scale — see Chapter 21) |
| **SQLAlchemy** | ORM | Typed Python objects instead of hand-written SQL everywhere; integrates with Alembic | Raw SQL (more boilerplate), other ORMs (SQLAlchemy is the ecosystem standard) |
| **Alembic** | Migrations | Versioned, reproducible schema history instead of manual DDL | Manually running SQL (no history, not reproducible) |
| **Pydantic / pydantic-settings** | Validation, config | One schema definition gives both runtime validation and static types; settings sourced from env vars | Manual dict validation, dataclasses without validation |
| **LightGBM** | Model | Gradient-boosted trees suit small-to-medium noisy tabular data with class imbalance handled natively; fast, exact SHAP support | Logistic regression (too linear), Random Forest (boosting typically wins on tabular data), LSTM/RNN (needs more data, no fast exact SHAP), Isolation Forest (wrong problem framing — anomaly score isn't a calibrated probability) — full reasoning in Chapter 9 |
| **SHAP** | Explainability | `TreeExplainer` is fast and exact for tree models specifically, not an approximation | LIME (slower, approximate), built-in feature importance (global only, not per-prediction) |
| **pandas / NumPy** | Data manipulation | Standard tabular data interchange format every downstream library expects | Polars (faster on huge data, less mature ecosystem fit here) |
| **scikit-learn** | Metrics, splitting | `GroupShuffleSplit`, `roc_auc_score`, `average_precision_score`, `calibration_curve` — battle-tested implementations of easy-to-get-wrong metrics | Hand-implementing metrics (reinventing tested code) |
| **matplotlib** | Evaluation plots | SHAP's own plotting is built on it; used for the calibration curve in the model card | Plotly (unnecessary interactivity for a static model card) |
| **structlog** | Logging | Structured JSON logs — required for reconstructing "why did the system do this" after the fact | Plain `print`/unstructured logging (fails that requirement outright) |
| **Docker / docker-compose** | Containerization | This repo must run standalone (`docker-compose up`, zero other repos present) — own Postgres + Redis | Running directly on host (doesn't demonstrate deployability, "works on my machine" risk) |
| **pytest / pytest-cov** | Testing | Ecosystem standard, strong fixture support (used heavily for the savepoint-isolated DB fixture) | `unittest` (more boilerplate, weaker fixtures) |
| **ruff** | Linting | Fast, single tool covering what used to need multiple (flake8 + isort-style import sorting) | flake8 + isort separately (more config, more tools) |
| **GitHub Actions** | CI | Runs lint + migrations + full suite against real Postgres/Redis service containers on every push/PR (Phase 11) | A hosted CI SaaS (unnecessary — GitHub Actions is already free and sufficient at this scale) |
| **Render Blueprint** | Deployment | One `render.yaml` describes the whole topology (web service, Postgres, Redis, the nightly job as a Cron Job) — see Chapter 16 (Phase 12) | Manually configuring each service in a dashboard (more error-prone, not reproducible) |

All planned dependencies are now in use — no Future Scope items remain in
the technology stack itself (compare to earlier revisions of this
handbook, when Redis and GitHub Actions were still pending).

---

## 6. Phase-by-Phase Implementation History

### Phase 0 — Project Foundation

**Objective:** repository skeleton before any business logic.

**Files added:** the full `app/` package skeleton (empty-but-commented
`services/`, `ml/`, `events/`, `batch/`), `config.py`, `logging_config.py`,
`app/data/base.py`, `app/schemas/` (temporary local home for `Equipment`,
`WorkOrder`, the Agent Contract — see Chapter 20), `app/api/` with only
`/health` and `/health/ready`, Alembic wiring, `Dockerfile`,
`docker-compose.override.yml`, `requirements.txt`/`requirements-dev.txt`,
`pyproject.toml`, one liveness test.

**Why this shape:** every later phase needed *somewhere correct* to put its
code on day one — the empty packages aren't filler, they're the layering
decision made explicit before there was any logic to organize.

### Phase 1 — Database & Data Layer Foundation

**Objective:** a clean, production-style data layer for every future phase
to build on.

**Files added:** four ORM models (`Equipment`, `SensorReading`, `WorkOrder`,
`RiskScore`), the first Alembic migration, four repository modules.

**Important implementation detail — a design correction made mid-phase:**
repository `create()` functions were changed from `commit()` to `flush()`.
The transaction boundary belongs to whoever orchestrates a multi-step write
(later: the service layer), not to an individual repository call — this
decision paid off directly in Phase 6, where `prediction_service` commits a
risk score and a work order together, atomically.

**A real bug found and fixed:** Postgres ENUM types are separate objects
from the columns that use them. The autogenerated `downgrade()` dropped
tables but not the enum types, so re-running `upgrade` after a `downgrade`
failed with `type already exists`. Fixed by adding explicit
`sa.Enum(...).drop(op.get_bind(), checkfirst=True)` calls to `downgrade()`.
Verified with a full upgrade → downgrade → upgrade cycle on a clean
database, not just visual inspection. (See Chapter 24 for the full story.)

### Phase 2 — Feature Engineering

**Objective:** one feature-computation function shared by every future
consumer (training, real-time inference, batch inference).

**Files added:** `app/ml/features.py` (`build_feature_vector`,
`FEATURE_COLUMNS`, `EXPECTED_SENSOR_TYPES`), `requirements-ml.txt` (pandas,
numpy), unit tests.

**Files modified:** `app/config.py` (added `feature_lookback_hours`).

**Why a pure function:** the ML layer has zero knowledge of HTTP, the
database, or ORM objects — that's what makes it independently unit-testable
and guarantees the same code path handles training and both inference
modes. This single decision is what makes train/serve skew structurally
impossible rather than merely unlikely (a classic, hard-to-detect
production ML bug where training and serving compute "the same" feature
subtly differently).

### Phase 3 — Model Training Pipeline

**Objective:** a reproducible offline pipeline producing a versioned model
artifact.

**Files added:** `app/ml/training/dataset.py` (synthetic data generator —
explicitly a stand-in for a real historical dataset), `app/ml/training/train.py`
(group split by `equipment_id`, LightGBM training, evaluation, artifact +
model card + manifest), model and unit tests.

**A real quality issue found and fixed:** the first version of the
synthetic dataset was too cleanly separable — the model scored a suspicious
ROC-AUC/PR-AUC of 1.0. Fixed by adding per-asset drift-strength variance,
mild drift on some "healthy" assets, and ~8% independent label noise. Final
realistic metrics: ROC-AUC ≈0.72–0.85, PR-AUC clearly above the positive-rate
baseline but nowhere near 1.0 — a portfolio reviewer would (rightly)
distrust a perfect score, so this was worth fixing even though nothing was
functionally "broken."

### Phase 4 — Explainability

**Objective:** per-prediction SHAP attribution — the root-cause tracer.

**Files added:** `app/ml/explain.py` (`explain_prediction`,
`FeatureAttribution`), model tests including a rigorous **additivity check**
(`expected_value + Σshap_values == raw model prediction`, verified against
a genuinely trained model, not a mock).

**Files modified:** `requirements-ml.txt` (added `shap`).

### Phase 5 — Inference Wrapper

**Objective:** one ML-layer entry point combining load-once model loading,
prediction, and explanation.

**Files added:** `app/ml/inference.py` (`PredictiveMaintenanceModel`,
`load_model`, cached `get_model`/`reset_model_cache`, `ModelNotFoundError`).

**Files modified (necessary refactor):** `app/config.py` (added
`model_registry_dir` as one shared setting) and `app/ml/training/train.py`
(now reads that same setting instead of computing its own path). Reasoning:
importing `REGISTRY_DIR` from `train.py` into `inference.py` would have
pulled matplotlib/scikit-learn training-only imports into the serving path
as a side effect — a real separation-of-concerns violation, fixed before it
shipped.

### Phase 6 — Service Layer: Prediction & Work Order Logic

**Objective:** the business rules — risk thresholding, work order creation,
the urgent-priority human-approval gate (FR4).

**Files added:** `app/services/risk_policy.py` (pure), `app/services/work_order_service.py`
(work order creation + `approve_urgent_priority`), `app/services/prediction_service.py`
(orchestration, owns the commit), `app/services/errors.py`, unit + integration
tests.

**Files modified:** `app/data/models/work_order.py` and `app/schemas/work_order.py`
(added `recommended_priority`, `priority_approved_at`, `priority_approved_by` —
this was explicitly flagged back in Phase 1 as deferred to "the services
milestone"), `app/data/repositories/work_order_repository.py` (added
`get_by_id`), `app/config.py` (added `urgent_priority_threshold`), a new
Alembic migration.

**A real bug found and fixed:** once the service layer started legitimately
calling `db.commit()`, the existing test fixture's "rollback in a `finally`
block" isolation strategy silently stopped working — committed data leaked
across tests, causing duplicate-key failures. Fixed by rewriting the
integration test fixture to bind each test's session to a SAVEPOINT nested
inside an outer transaction; `session.commit()` in the code under test now
only releases the savepoint, and the outer transaction is unconditionally
rolled back when the test ends. Verified by running the full suite twice
back-to-back. (Full story in Chapter 24.)

### Phase 7 — Event Publishing

**Objective:** publish async domain events so other (future) platform repos
can react without this repo knowing about them.

**Files added:** `app/events/publisher.py` (event dataclasses:
`EquipmentFailureRiskEvent`, `WorkOrderApprovedEvent`; publish functions;
error handling), unit + unit-integration tests.

**Files modified:** `app/services/prediction_service.py` and
`app/services/work_order_service.py` (wire event publishing into both flows,
after commit, best-effort — if Redis is down, event is dropped but the
write is always durable), `app/events/__init__.py` (updated comment).

**Design decision — events are best-effort, not durable:** a Redis outage
must never block a prediction or an approval. The event is published *after*
the database commit succeeds, so it's always a bonus notification, never a
transaction participant. A production system would add a dead-letter queue
(Phase 10) if event delivery durability became critical, but this
architecture choice keeps Phase 7 simple while making that future addition
straightforward.

### Phase 8 — Real-Time API

**Objective:** expose the service layer over HTTP — `/predict`,
`/equipment/{id}/risk`, work order approval, and the Agent Contract
endpoint.

**Files added:** `app/api/routes/prediction.py`, `app/api/routes/work_orders.py`,
`app/api/routes/agent.py`, `app/api/error_handlers.py` (central
exception → HTTP status mapping), API integration tests (prediction,
work orders, agent, health).

**Files modified:** `app/main.py` (registers routers + error handlers;
lifespan now loads the model once at startup, per NFR2), `app/api/routes/health.py`
(readiness now also checks the model is loaded, 503 if not — this was
explicitly flagged back in Phase 0's comment as deferred to "once
inference exists"), `tests/integration/conftest.py` (added a `client`
fixture bound to the same savepoint-isolated session as `db`, so API
tests can seed data directly and see it through HTTP while staying
isolated).

**Error mapping (`app/api/error_handlers.py`):**

| Exception | HTTP status |
|---|---|
| `EquipmentNotFoundError` | 404 |
| `WorkOrderNotFoundError` | 404 |
| `InvalidApprovalError` | 409 |
| `ModelNotFoundError` | 503 |

**A real bug found and fixed:** `_readings_to_dataframe` (in
`prediction_service.py`) built a `pd.DataFrame` from a list comprehension
without specifying columns — for equipment with **zero** sensor readings in
the lookback window (e.g. a newly onboarded asset, or any asset queried far
outside its seeded data's time range), this produced a column-less empty
DataFrame that crashed `build_feature_vector` with `KeyError: 'sensor_type'`
the moment it tried to filter by column name. This had been latent since
Phase 6 — every existing test happened to seed readings inside the query
window. Caught by the Agent Contract test once it queried with a real
"now" timestamp far outside a fixed seeded reading's date. Fixed by
specifying `columns=["sensor_type", "timestamp", "value"]` explicitly in
the DataFrame constructor, and added a dedicated regression test
(`test_predict_works_for_equipment_with_zero_sensor_readings`).

**A repeat of Bug 3:** `PredictResponse` and `RiskScoreResponse` both have a
`model_version` field, which triggered the same pydantic protected-namespace
warning as `model_registry_dir` did in Phase 5. Same fix:
`model_config = ConfigDict(protected_namespaces=())`.

### Phase 9 — Batch Scoring Job

**Objective:** score every equipment asset nightly, reusing the real-time
path's exact logic rather than a second implementation.

**Files added:** `app/batch/nightly_job.py` (`run_nightly_scoring`),
integration tests.

**Files modified:** `app/services/prediction_service.py` — added a
`source: RiskScoreSource` parameter to `run_prediction_for_equipment`
(default `REAL_TIME`), so the batch job can tag its writes `BATCH` while
calling the identical function the API uses. This is a public function
signature change (used by both the API and the batch job), which is why
Phase 9 triggered a full-suite test run rather than a phase-scoped one,
per the testing policy established that phase.

**A real lesson (not a shipped bug) found while testing:** the batch job's
per-asset error isolation calls `db.rollback()` when one asset's scoring
fails, so one bad asset doesn't abort the whole run. In the first version of
the test, seeded equipment was only `flush()`ed, not committed — so
`db.rollback()` (which rolls back to the session's last savepoint) wiped out
*both* seeded assets, not just the failed one, and the test failed with
`ObjectDeletedError`. Fixed by committing the seed data in the test, which
matches production reality: equipment reference data is always
already-committed before a batch run starts, never pending mid-transaction.
See Chapter 24, Bug 7.

### Phase 10 — Observability & Failure-Mode Hardening

**Objective:** structured logging across the service layer (previously only
app startup/shutdown was logged), and a defined, tested response for every
known failure mode.

**Files modified:** `app/services/prediction_service.py` and
`app/services/work_order_service.py` — both now log their key decisions
(`prediction_scored`, `work_order_created` with `held_pending_approval`
flagged, `work_order_urgent_priority_approved`) and, critically, the
Redis-publish failure path changed from a silent `except Exception: pass`
to `logger.warning("event_publish_failed", ...)` — a dropped downstream
notification is now findable in the log stream instead of invisible.
`app/api/error_handlers.py` — every existing handler now logs, and a new
**catch-all `Exception` handler** was added: any unexpected error (a DB
connection drop, a bug) now logs the full traceback server-side but returns
a generic `{"detail": "An internal error occurred."}` to the client, never
leaking internals.

**Files added:** `tests/integration/test_failure_modes.py` — tests that
actually exercise failure paths (model unavailable → 503, Redis down during
prediction → still 200 with work order created, Redis down during approval
→ still succeeds, a genuine unhandled exception → 500 with no leaked
detail), not just the happy path.

### Phase 11 — Full Test Suite & CI

**Objective:** close remaining test coverage gaps, add the fourth EDD test
category (contract tests — declared from the start, never actually built
until now), and automate verification via CI.

**Files added:** `tests/contract/` — a new test category. `test_agent_contract.py`
validates the Agent Contract's shapes directly (required fields, confidence
bounds, JSON-serializability) independent of any one route's usage of them.
`test_schema_orm_parity.py` verifies the Pydantic schemas' claim to "match
the ORM model exactly" automatically, rather than trusting the docstring —
including a round-trip test converting a real ORM instance to its schema.
`tests/unit/test_config.py` — catches a real misconfiguration class (e.g.
`urgent_priority_threshold` set below `failure_risk_threshold`, which would
silently break `risk_policy`'s logic). `.github/workflows/ci.yml` — lint,
migrate, and full test suite against real Postgres and Redis service
containers on every push/PR.

**Files modified:** `app/schemas/agent_contract.py` — `AgentResponse.confidence`
is now constrained to `Field(ge=0.0, le=1.0)`; an unconstrained float let a
bug silently produce a nonsensical confidence like 5.0. `tests/integration/conftest.py` —
the `client` fixture now uses `TestClient` as a context manager so FastAPI's
`lifespan` actually runs in tests (previously silently skipped, meaning the
model-preload startup logic had zero test coverage). `pyproject.toml` —
`ruff` excludes `alembic/versions/` (auto-generated, not hand-authored) and
`pytest` now enforces `--cov-fail-under=90` as a real gate, not just a
report.

**Real bug found and fixed:** while wrapping a long line during lint
cleanup, a `str_replace` accidentally deleted the `SensorReading.timestamp`
column entirely. Caught immediately by reviewing the file diff before
moving on, restored, and verified with a full test run before continuing —
worth recording as a concrete example of the "review before trusting an
edit" discipline this project tries to model throughout.

**Result:** coverage rose from 94% → 97% (real gaps closed — startup logic,
`app/schemas/equipment.py` which was previously 0%, the generic-exception
branches in `publisher.py` — not padding).

### Phase 12 — Deployment

**Objective:** package the already-complete, already-hardened system so it
can actually run somewhere reachable — packaging and configuration only, no
new runtime capability and no architecture changes.

**Files added:** `app/scripts/seed_demo_data.py` — four idempotent demo
assets (healthy, degrading, borderline, and one with **zero** sensor
history, deliberately chosen to demonstrate the Bug 5/Bug 9 fix rather than
avoid the case). `render.yaml` — a Render Blueprint describing the whole
service topology (web service, Postgres, Redis, the nightly batch job as a
Cron Job) in one file. `DEPLOYMENT.md` — the deploy steps, local-run
instructions, and the model-baking trade-off stated plainly.

**Files modified:** `docker-compose.override.yml` renamed to
`docker-compose.yml` — Compose treats the `.override.yml` suffix as an
overlay on a base file, which never existed here, so `docker compose up`
failed without an explicit `-f` flag; this was correctly diagnosed in an
earlier conversation but the actual rename was deferred until this phase.
`Dockerfile` — bakes `model/registry/` into the image (the deployment
strategy decided on: simplest option for a standalone repo at this scope;
the trade-off — retraining requires a rebuild, not a live hot-swap — is
documented in `DEPLOYMENT.md` rather than solved with an external model
registry that nothing in this project yet needs). `CMD` now binds to
`${PORT:-8000}` instead of a hardcoded port — Render assigns a dynamic port
via `$PORT` and health-checks against it; a hardcoded port would have
silently failed that health check in production while working fine
locally. `app/api/routes/prediction.py` — the NaN-to-null fix (Bug 9 below).

**Two real bugs found and fixed — both caught only because the system was
actually run, not just reviewed:**

1. **The Dockerfile could not serve a single prediction.** It only
   installed `requirements.txt`, never `requirements-ml.txt` — so
   `lightgbm`/`shap`/`pandas` were entirely missing from the image. This had
   been true since Phase 0 and was invisible to every test (tests run in
   this dev environment, which has these packages installed directly, not
   inside the container). Confirmed and fixed by literally reproducing the
   Dockerfile's exact build steps in a clean environment.
2. **NaN crashed the exact case Bug 5 was built to fix (Bug 9, Chapter 24).**
   An asset with zero sensor readings produces legitimate `NaN` feature/SHAP
   values — but Starlette's `JSONResponse` rejects raw `NaN` outright, so
   `/predict` against that exact asset type returned a 500. 91 passing tests
   had not caught this; it surfaced only when the reproduced-Dockerfile
   server was actually hit with a real request.

**A verification honesty note, worth recording:** Docker itself is not
installed in the sandbox this project was built in. Every "container"
verification in this project (Phases 8–12) was a manual reproduction of the
Dockerfile's `COPY`/install steps in a clean directory, run and hit with
real HTTP requests — not a literal `docker build`. This proves the *logic*
is correct; it does not rule out a Docker-runtime-specific issue (e.g. how
a real container engine handles the shell-form `CMD`'s `$PORT` expansion)
that only an actual `docker build` would catch on first real deploy.

### Future Scope

- **Phase 13 — Documentation & Portfolio Polish.** The only remaining
  phase, and a different kind from every one above: it maps to no FR/NFR in
  the EDD — every other phase added or hardened runtime capability, Phase
  13 adds nothing the system *does*. It exists because a portfolio project
  is also read, not just run: a polished `README.md` (the current one is
  still Phase 0's scaffolding-era version, actively describing the repo as
  "Phase 1 — foundation," which is now inconsistent with the real state,
  not just minimal), a surfaced model card, and a short guided-demo script
  distinct from this handbook's Chapter 28 talking points. Skipping it
  costs nothing functional — the system is complete, tested, and deployable
  without it — but costs real legibility to a recruiter skimming the repo
  cold.

---

## 7. Database Design

### Tables

```
equipment                    sensor_reading                work_order
------------------           ------------------             ------------------
id (PK)                      id (PK, autoincr)               id (PK)
plant_id (idx)                equipment_id (FK, idx)          equipment_id (FK, idx)
type                          timestamp (idx)                 opened_by
install_date                  sensor_type                     priority (enum)
criticality_tier              value                           recommended_priority (enum, nullable)
                                                                status (enum)
                                                                created_at
                                                                priority_approved_at (nullable)
                                                                priority_approved_by (nullable)

risk_score
------------------
id (PK, autoincr)
equipment_id (FK, idx)
probability
model_version
source (enum: real_time | batch)
created_at (idx)
```

### Relationships

`Equipment` is the hub — `SensorReading`, `WorkOrder`, and `RiskScore` all
have a `ForeignKey("equipment.id")` and a corresponding `relationship()`
back to `Equipment`. One equipment row, many readings/work orders/risk
scores.

### Ownership rule (bounded contexts, platform-wide)

`Equipment` is a platform-owned reference entity — this repo only reads it,
never writes it (the real owner will be a shared reference service, not yet
built). `SensorReading`, `WorkOrder`, and `RiskScore` are owned by this repo.

### Constraints

- All foreign keys enforced at the database level (`ForeignKey("equipment.id")`).
- `priority` and `status` on `WorkOrder`, and `source` on `RiskScore`, are
  Postgres native ENUM types — not free-text strings — so an invalid value
  is rejected at the database level, not just by application code.
- `equipment_id` is indexed on every child table (it's the most common
  filter column). `timestamp` is indexed on `sensor_reading` (range queries
  for the lookback window). `created_at` is indexed on `risk_score`
  (ordering for "get latest").

### Migrations

Two migrations exist so far:
1. `aa0f4a9c09c9` — creates all four tables.
2. `81f1de0dd7af` — adds the three approval-gate columns to `work_order`.

Both have been verified with a full upgrade → downgrade → upgrade cycle
against a real Postgres instance, not just autogenerated and trusted.

---

## 8. Feature Engineering

### The features, and the intuition behind each

For each of the three sensor types (`temperature`, `vibration`, `pressure`),
three features are computed over the lookback window:

| Feature | Business intuition | Mathematical definition |
|---|---|---|
| `{sensor}_rolling_mean` | "Is this sensor's typical reading elevated?" | Mean of all readings for that sensor in the window |
| `{sensor}_rolling_std` | "Is this sensor behaving erratically, not just high?" | Population std dev (`ddof=0`) of readings in the window; `0.0` if fewer than 2 readings |
| `{sensor}_rate_of_change` | "Is this sensor actively trending, not just currently elevated?" | `(last_value - first_value) / elapsed_hours` between the first and last reading in the window |

Plus two static features: `equipment_age_days` (days since install — older
equipment may behave differently) and `criticality_tier` (a business-defined
1–3 rating, not derived from sensor data at all).

### Why rolling-window features instead of raw readings

A single raw reading ("72°C at 3:14pm") means nothing without context — is
that high? Rising? Normal for this asset type? The rolling-window
transformation is what turns a raw signal into something with predictive
meaning. This matters more to model quality than which specific gradient
boosting library is used downstream.

### Why these three statistics specifically

Mean captures *level* (is it elevated), std captures *volatility* (is it
erratic), rate-of-change captures *trend* (is it getting worse). Together
they cover the three basic shapes degradation can show up as in a sensor
signal, without requiring a sequence model to learn them from scratch.

### Missing data handling

If a sensor type has zero readings in the window, `rolling_mean` is `NaN`
(honestly "unknown," not a fabricated zero), while `rolling_std` and
`rate_of_change` are `0.0` (no evidence of volatility or trend, which is a
defensible default — LightGBM also handles `NaN` natively at split time, so
this doesn't break training or inference).

### Files

`app/ml/features.py` — `build_feature_vector()`, `FEATURE_COLUMNS`,
`EXPECTED_SENSOR_TYPES`.

---

## 9. Machine Learning Pipeline

### Algorithm: gradient-boosted decision trees (LightGBM)

| Alternative | Why rejected |
|---|---|
| Logistic regression | Sensor-degradation-to-failure relationships are nonlinear with interaction effects; logistic regression can't capture that without manual feature-interaction engineering. Good as a baseline, not as the final model. |
| Random forest | Also tree-based and nonlinear, but boosting typically wins on tabular data at comparable cost, with better native class-imbalance handling. |
| LSTM / RNN | Needs far more data than this scale provides; SHAP's fast exact `TreeExplainer` doesn't apply to RNNs (would need a slower approximate method); harder to defend "why this" in an interview when a simpler, equally accurate model exists. |
| Isolation Forest (anomaly detection) | Answers a different question ("is this unusual") rather than "probability of failure in N days." Anomaly scores aren't calibrated probabilities, which this project explicitly needs. |

### Preprocessing

None beyond the feature engineering in Chapter 8 — LightGBM handles
mixed-scale numeric features and missing values (`NaN`) natively, so no
scaling/imputation step is needed.

### Training

`GroupShuffleSplit` by `equipment_id` (25% test) — critical detail: no
equipment unit appears in both train and test. Splitting by row instead
would let the model "cheat" by learning patterns from the same asset's
other timestamps, overstating real generalization.

`LGBMClassifier` with `is_unbalance=True` (native class-imbalance handling,
no manual resampling), shallow-ish trees (`num_leaves=15`), modest learning
rate (`0.05`), fixed `random_state=42` for reproducibility.

### Evaluation

- **PR-AUC**, not accuracy — failures are rare; a model that never flags
  anything would score misleadingly well on accuracy or even ROC-AUC.
  PR-AUC focuses specifically on performance on the rare positive class.
- **ROC-AUC** as a secondary, complementary metric.
- **Calibration curve** — a separate concern from ranking quality: does a
  predicted 0.8 correspond to roughly 80% real-world failure frequency?
  This matters because the threshold-based work-order automation is only
  trustworthy if the probability means what it says.

### Registry and versioning

Every training run writes a new timestamped version
(`vYYYYMMDDHHMMSS`) under `model/registry/`, containing:
`model.txt` (the LightGBM booster, native text format), `feature_columns.json`
(exact training column order — validated at load time, see Chapter 11),
`metrics.json`, `model_card.md`, `calibration_curve.png`. A `manifest.json`
tracks every version's metadata and which one is `"latest"`.

### Reproducibility

Fixed random seeds throughout (dataset generation, train/test split, model
training) mean the same code produces the same model — a real property,
verified by a unit test (`test_same_seed_is_reproducible`).

### Files

`app/ml/training/dataset.py`, `app/ml/training/train.py`.

---

## 10. Explainability (SHAP)

### Intuition

SHAP (SHapley Additive exPlanations) answers: *"how much did each feature
push this specific prediction away from the average prediction?"* It's
grounded in Shapley values from cooperative game theory — treat each
feature as a "player" contributing to the "payout" (the prediction), and
fairly distribute credit among them.

### Expected value

`explainer.expected_value` is the model's average output over its training
data — the baseline a specific prediction is explained *relative to*. A
prediction of 0.9 when the baseline is 0.15 means something is pushing this
specific asset's risk far above typical.

### Local vs. global explanation

- **Local** (what this project uses): explains one specific prediction —
  "why is THIS asset's risk 0.9." This is what `explain_prediction()`
  produces.
- **Global**: aggregates local explanations across many predictions to show
  overall feature importance across the whole model — not yet built as a
  first-class feature here, though the model card's structure could support
  adding it later.

### Why `TreeExplainer` specifically

Unlike model-agnostic methods (e.g. LIME, which approximates by perturbing
inputs and fitting a local surrogate model), `TreeExplainer` computes SHAP
values **exactly** for tree-based models by exploiting the tree structure
directly — and it's fast. This is a big part of why LightGBM was chosen
over, say, an RNN: the explainability method matches the model family
exactly.

### The additivity property — and why it's tested, not assumed

The defining correctness property of SHAP: `expected_value + Σ(all feature
shap_values) == the model's actual raw prediction` for that row. This
project's model tests verify this holds **against a genuinely trained
model**, not a mock — `tests/model/test_explain.py::test_shap_values_satisfy_additivity_against_the_real_model`.
This is the strongest possible correctness check for an explainability
integration: if it passes, the SHAP wiring is provably correct for that
model, not just "looks reasonable."

### What SHAP does NOT prove

A high SHAP value tells you the *model* weighted that feature heavily — not
that the feature *physically caused* the outcome. It's a strong diagnostic
hint for a technician to start their investigation, not a certified root
cause. This distinction is stated explicitly in the module docstring and
worth stating explicitly in an interview too.

### Files

`app/ml/explain.py` — `explain_prediction()`, `FeatureAttribution`.

---

## 11. Inference Pipeline

### From a feature dict to a full result

```
feature_row (dict)
    |
    v
PredictiveMaintenanceModel.predict_with_explanation(feature_row, top_n=3)
    |
    |--> predict_proba(feature_row)         --> booster.predict(row_df)[0]
    |--> explain_prediction(booster, row)   --> ranked FeatureAttribution list
    v
PredictionResult(probability, model_version, attributions)
```

### Load-once caching

`get_model()` is a module-level cached singleton — the booster is read from
disk once per process (intended to be triggered by the future API's startup
hook, Phase 8) and reused for every subsequent call, rather than reloading
from disk on every prediction. `force_reload=True` and `reset_model_cache()`
exist specifically to make this testable without process restarts.

### Schema validation at load time

`PredictiveMaintenanceModel.__init__` raises `ValueError` if the loaded
model's `feature_columns` don't exactly match the current
`app.ml.features.FEATURE_COLUMNS` — refusing to silently serve predictions
with misaligned columns rather than producing a wrong-but-plausible-looking
number.

### Error handling

`ModelNotFoundError` (a specific exception type, not a bare
`FileNotFoundError`) is raised when no registry/manifest/artifact exists —
intended to be caught by the future API layer and turned into a meaningful
HTTP error rather than a generic 500.

### Files

`app/ml/inference.py`.

---

## 12. Service Layer

### Responsibilities

The service layer is where every **business rule** lives — nothing here is
an HTTP concern (that's the API layer, Phase 8) and nothing here is a raw
CRUD concern (that's the repository layer, Chapter 13).

### `risk_policy.py` — pure business rule

```python
def evaluate_risk(probability: float) -> WorkOrderPriority | None
```

No DB, no model, no I/O. Two configurable thresholds
(`failure_risk_threshold`, `urgent_priority_threshold`) decide whether a
probability warrants no action, an `ELEVATED` recommendation, or an
`URGENT` recommendation. Being pure makes this the easiest-to-test file in
the entire codebase — five unit tests, zero fixtures, zero database.

### `work_order_service.py` — the approval gate (FR4)

The single most important business rule in this project:
**the system never persists `priority=URGENT` directly as a result of a
prediction.** When `risk_policy` recommends `URGENT`,
`create_work_order_for_prediction` persists the work order at
`priority=ELEVATED` with `recommended_priority=URGENT` — a human must call
`approve_urgent_priority()` to actually escalate it. This is a designed
governance checkpoint, not a limitation — see Chapter 20 for the full
reasoning.

### `prediction_service.py` — orchestration and the transaction boundary

Ties the data layer and ML layer together into the one operation Phase 8's
API and Phase 9's batch job will both call. Deliberately **owns the
`db.commit()`** — every repository call underneath it only flushes (a Phase
1 design decision that paid off directly here: the risk score and any
resulting work order commit together, atomically, as one unit of work).

### Interaction with other layers

```
prediction_service
    --> equipment_repository, sensor_reading_repository, risk_score_repository  (data layer)
    --> build_feature_vector, get_model                                         (ML layer)
    --> work_order_service --> work_order_repository, risk_policy                (service + data layer)
```

### Files

`app/services/risk_policy.py`, `app/services/work_order_service.py`,
`app/services/prediction_service.py`, `app/services/errors.py`.

---

## 13. Repository Layer

### What it is

A thin function-per-operation wrapper around SQLAlchemy queries — `get_by_id`,
`get_recent`, `create`, `get_open_for_equipment`, `get_latest`, etc. — one
module per entity (`equipment_repository.py`, `sensor_reading_repository.py`,
`work_order_repository.py`, `risk_score_repository.py`).

### Why this pattern (Repository Pattern)

It isolates every place raw SQLAlchemy `Session` operations happen. No code
outside `app/data/repositories/` constructs a `select()` statement or calls
`db.query()` directly. This means: (1) the service layer can be tested by
mocking or faking a repository call instead of standing up complex query
logic inline, and (2) if the ORM were ever swapped for something else, the
blast radius is contained to this one folder.

### The `flush()`-not-`commit()` decision

Every `create()` function calls `db.flush()` (assigns IDs, runs constraints)
but never `db.commit()`. The transaction boundary is a decision for the
*caller* — a repository has no way to know whether its write is the only
operation in this unit of work, or one of several that need to succeed or
fail together (like Phase 6's risk-score-plus-work-order write).

### CRUD coverage today

| Entity | Read | Write |
|---|---|---|
| Equipment | `get_by_id`, `list_all` | — (read-only; platform-owned) |
| SensorReading | `get_recent` | `create` |
| WorkOrder | `get_by_id`, `get_open_for_equipment` | `create` |
| RiskScore | `get_latest` | `create` |

### Benefits realized in practice

The Phase 1 `flush()`-not-`commit()` decision, made before there was any
business logic to justify it, turned out to be exactly right once Phase 6
needed an atomic multi-table write — no repository code needed to change.

### Files

`app/data/repositories/*.py`.

---

## 14. API Layer

### Endpoints (implemented, Phase 8)

```python
GET  /health                          # liveness — process is up, touches nothing
GET  /health/ready                    # readiness — DB reachable AND model loaded
POST /predict                         # runs a real-time prediction for one asset
GET  /equipment/{equipment_id}/risk   # latest persisted risk score (no new prediction)
POST /work-orders/{id}/approve        # the ONLY HTTP path that can set priority=URGENT
POST /agent                           # Agent Contract — platform-wide uniform interface
```

`app/api/deps.py` provides `get_db()`, a FastAPI dependency yielding a
SQLAlchemy session per request. `app/main.py` assembles the FastAPI app
using a `lifespan` context manager (not the deprecated `@app.on_event`
hook); on startup it now also calls `get_model()` once, so the first
real request never pays the model-load cost — if no model is registered
yet, startup logs a warning rather than crashing, and `/health/ready`
correctly reports 503 until one exists.

### Why it was built last

Everything this layer exposes already had a fully tested, working
implementation underneath it in the service layer before this phase
started. Building the API layer last means it's a thin adapter over
something already proven correct, rather than HTTP plumbing wrapped
around business logic that didn't exist yet.

### Request/response validation

Every route has typed Pydantic request/response models
(`PredictRequest`/`PredictResponse`, `ApproveRequest`/`WorkOrderApprovalResponse`,
etc.) — malformed input is rejected by FastAPI before the route body
even runs, and the response shape is guaranteed rather than
best-effort.

### Exception handling — one central mapping, not scattered try/except

`app/api/error_handlers.py` registers global FastAPI exception handlers
for the service/ML-layer's own exception types:

| Exception | HTTP status | Meaning |
|---|---|---|
| `EquipmentNotFoundError` | 404 | asset doesn't exist |
| `WorkOrderNotFoundError` | 404 | work order doesn't exist |
| `InvalidApprovalError` | 409 | work order's state doesn't allow this approval action |
| `ModelNotFoundError` | 503 | no trained model registered yet (a known operational state, not a bug — hence 503, not 500) |
| Any other `Exception` | 500 | **catch-all, added Phase 10.** Full traceback logged server-side; client gets a generic `{"detail": "An internal error occurred."}` — never the raw exception text, which could leak internal details (a query, a file path) to an API caller. |

Route handlers never catch these individually — they let them propagate
and the global handler maps them. Adding a new route never risks
forgetting to handle `EquipmentNotFoundError` correctly, because there's
exactly one place that decision is made.

### NaN-safety in responses (Phase 12)

`app/ml/features.py`'s `rolling_mean` is legitimately `NaN` when a sensor
channel has zero readings (see Chapter 8) — a real, valid state, not an
error. But Starlette's `JSONResponse` rejects raw `NaN` outright
(`allow_nan=False`), which would 500 any prediction against a
sparse-history asset. `FeatureAttributionResponse` fields are typed
`float | None`, and `prediction.py`'s `_json_safe_float()` converts
`NaN → None` at the response boundary — representing "no data" as JSON
`null`, the honest translation, rather than letting an internal numeric
convention leak into the wire format. Found and fixed during Phase 12's
Docker verification — see Chapter 24, Bug 9.

### The Agent Contract endpoint

`POST /agent` is the platform-wide interface a future
`platform-orchestrator` repo would call — same `AgentRequest`/`AgentResponse`
shape regardless of whether the repo behind it is this trained ML model
or a RAG pipeline in a different repo. For this repo specifically, there's
no natural-language query parsing — `context.equipment_id` is required
(`context.as_of` is optionally accepted too, mainly useful for
deterministic testing). The SHAP attributions are reported as `provenance`
strings, mirroring how a RAG-based repo would cite source documents in the
same field. `AgentResponse.confidence` is constrained to `[0.0, 1.0]`
(Phase 11) — a real Agent Contract compliance fix, since an unconstrained
float previously let a bug silently produce a nonsensical confidence value.

### Dependency injection for testability

`get_db` is overridden in `tests/integration/conftest.py`'s `client`
fixture to use the same savepoint-isolated session as the `db` fixture —
this is what makes it possible to seed data with `db.add(...)` directly
and then verify it through a real HTTP call, all inside one rolled-back
test transaction.

---

## 15. Testing

### Four test categories, four different jobs (Phase 11 completed the set)

```
tests/unit/         no DB, no trained model — pure logic only
tests/integration/  real Postgres, savepoint-isolated
tests/model/         trains/loads a REAL model — slowest, but proves real correctness
tests/contract/      verifies the shapes this repo exposes externally stay valid and in sync
```

The EDD named all four test categories from the start; `tests/contract/`
sat undeclared as a gap until Phase 11 actually built it.

### `tests/unit/` — pure logic

`test_features.py`, `test_training_dataset.py`, `test_risk_policy.py`,
`test_config.py` (Phase 11 — catches a real misconfiguration class, e.g.
`urgent_priority_threshold` set below `failure_risk_threshold`),
`test_events.py`/`test_events_publish.py` (Phase 7/11 — event dataclass
shape and Redis-failure wrapping). No fixtures beyond plain Python objects.
Fast, runs anywhere.

### `tests/integration/` — real database

Uses the `db` fixture (`tests/integration/conftest.py`), which runs against
a real local Postgres instance rather than mocking the ORM — repository and
service-layer query behavior is exactly the kind of thing that *looks*
correct and silently isn't.

**Test isolation via savepoints (the Chapter 24 bug fix, explained in
full):** each test's session is bound to a SAVEPOINT nested inside an outer
connection-level transaction (`join_transaction_mode="create_savepoint"`).
When code under test calls `session.commit()` (as the service layer
legitimately does), that only releases the savepoint — the outer
transaction is still rolled back unconditionally when the test ends, so no
test's data ever survives into the next test, regardless of how many times
the code under test commits. The `client` fixture (Phase 8, fixed in Phase
11 to actually trigger `lifespan`) shares this same session via a
`get_db` dependency override, so API tests can seed with `db.add(...)`
directly and verify through real HTTP calls.

### `tests/model/` — real model training and loading

`test_train.py` trains a real model into a temp registry and asserts
metric thresholds and artifact existence — a genuine regression check, not
a rubber stamp (`pr_auc > positive_rate_test`, i.e., meaningfully better
than random). `test_explain.py` includes the SHAP additivity check.
`test_inference.py` trains into a temp registry, then loads and predicts
against that real artifact.

### `tests/contract/` — external shape guarantees (Phase 11)

`test_agent_contract.py` validates the Agent Contract's Pydantic shapes
directly — required fields, the `confidence` probability bound, JSON
serializability — independent of any one route's usage. `test_schema_orm_parity.py`
enforces the claim (previously just a docstring comment) that
`app/schemas/equipment.py` and `app/schemas/work_order.py` match their ORM
models exactly: it compares the Pydantic field set against the actual
database column set programmatically, and round-trips a real ORM instance
through `model_validate(..., from_attributes=True)` — the same conversion
a route would perform — rather than trusting the two definitions stay in
sync by convention.

### No mocking of the ML pipeline, deliberately

Where a fake IS used (`_FakeModel` in `tests/integration/test_prediction_service.py`
and the API tests), it's a deliberate, narrow choice: those tests verify
*service-layer orchestration* (does the right branch execute, is the risk
score persisted, is the work order held back correctly) — not ML
correctness, which is already covered exhaustively under `tests/model/`.
Testing the same thing twice at two different layers would be redundant,
not more rigorous.

### Coverage strategy

`pytest-cov` runs on every test invocation (configured in `pyproject.toml`)
with `--cov=app --cov-report=term-missing --cov-fail-under=90` — the
90% floor (Phase 11) is an enforced gate, not just a report; a coverage
regression fails the test run, not just the summary line.

### CI (Phase 11)

`.github/workflows/ci.yml` runs on every push/PR: install deps → `ruff
check` → `alembic upgrade head` → `pytest`, against real Postgres and
Redis service containers (not mocks — consistent with how this repo's own
tests are written). Written from GitHub Actions' documented syntax but not
yet verified against a live run in this repo (this environment has no
network access to trigger one) — see the same honesty note in Chapter 6's
Phase 12 entry about `render.yaml`.

### Current test count: 91 passing, 97% coverage (as of Phase 12)

---

## 16. Docker

### Dockerfile

Single-stage build: `python:3.12-slim` base, install `requirements.txt`
**and `requirements-ml.txt`**, copy `app/` + `alembic/` + `model/registry/`,
run under `uvicorn`. The batch job reuses this same image with a different
container command (`python -m app.batch.nightly_job`) rather than a second
Dockerfile — same code, different entrypoint.

**A real bug, fixed in Phase 12:** the Dockerfile originally only installed
`requirements.txt` — `requirements-ml.txt` (lightgbm, shap, pandas, etc.)
was missing entirely, so the container as it existed from Phase 0 through
Phase 11 could not have served a single prediction. This had gone
undetected because every test in this project runs directly on the host
environment (which has all dependencies installed), never inside the
actual container — no test exercises the Dockerfile itself. See Chapter 24,
Bug 8.

**The model is baked into the image** (`COPY model ./model`), not
volume-mounted or fetched from external storage — the simplest option for
this project's scope, matching the platform architecture doc's "hero-path
deployment" approach. Trade-off, stated plainly: retraining requires a
rebuild and redeploy, not a live hot-swap. A larger production system would
fetch a versioned artifact from external storage (S3, a model registry
service) at startup instead — real, meaningful complexity with no
corresponding capability in this project today (there's no live retraining
trigger to hot-swap for), so it's documented as a named future improvement
rather than built. Full reasoning in `DEPLOYMENT.md`.

**`CMD` binds to `${PORT:-8000}`** (shell form, so the variable expands),
not a hardcoded port — Render assigns a dynamic port via `$PORT` and
health-checks against it; a hardcoded port would silently fail that check
in production while working fine locally, where `$PORT` is unset and the
`8000` fallback applies.

### docker-compose.yml

Gives this repo its own Postgres and Redis — **not shared** with any other
platform repo's compose setup — so `docker-compose up` here needs zero
other repository present. This directly matches the "every repo is
independently deployable and independently demoable" principle from the
platform-level architecture doc.

**Renamed from `docker-compose.override.yml` in Phase 12.** Compose treats
the `.override.yml` suffix as an overlay merged onto a base
`docker-compose.yml` — which never existed in this repo, so plain
`docker compose up` failed with "no configuration file provided" unless an
explicit `-f docker-compose.override.yml` flag was used. This was correctly
diagnosed in an earlier conversation (the person running this project asked
a sharp question about exactly this inconsistency) but the actual rename
was deliberately deferred until Phase 12, since renaming a file mid-phase
for something not yet blocking wasn't in scope at the time it was raised.

### Networking

Services communicate over the default Docker Compose bridge network via
service names (`db`, `cache`) — the API container connects to
`postgresql://...@db:5432/...`, not `localhost`, since `localhost` inside a
container refers to the container itself.

### Volumes

`pm_db_data` — a named volume persisting Postgres data across container
restarts, so `docker-compose down && docker-compose up` doesn't lose local
data (only `docker-compose down -v` would).

### A verification honesty note

Docker itself is not installed in the sandbox this project was built in.
Every "container" verification across Phases 8–12 was a manual reproduction
of the Dockerfile's exact `COPY`/install steps in a clean directory, run
and hit with real HTTP requests — not a literal `docker build`. This is
what actually caught both Docker-related bugs in Phase 12 (the missing ML
deps, and the NaN serialization issue) — but it does not rule out a
Docker-runtime-specific issue that only a real `docker build` would catch
on first live deploy. Recorded here rather than implied, per the same
verification discipline this project tries to apply throughout.

### Files

`Dockerfile`, `docker-compose.yml`, `.dockerignore`, `render.yaml`,
`DEPLOYMENT.md`.

---

## 17. Configuration

### Pattern: one settings singleton, sourced from environment variables

```python
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", protected_namespaces=())
    ...
settings = get_settings()   # module-level singleton, cached via lru_cache
```

Nothing in the codebase calls `os.environ` directly — every configurable
value is a typed field on `Settings`, read once, reused everywhere via
`from app.config import settings`.

### Current settings

| Setting | Default | Used by |
|---|---|---|
| `app_name`, `environment`, `log_level` | — | logging, app metadata |
| `database_url` | local Postgres | `app/data/base.py` |
| `redis_url` | local Redis | `app/events/publisher.py` (Phase 7) |
| `failure_risk_threshold` | 0.7 | `risk_policy.evaluate_risk` |
| `urgent_priority_threshold` | 0.9 | `risk_policy.evaluate_risk` |
| `prediction_window_days` | 7 | *(defines the label horizon; not yet consumed by code — synthetic dataset currently hardcodes a matching assumption)* |
| `feature_lookback_hours` | 168 (7 days) | callers of `sensor_reading_repository.get_recent` |
| `model_registry_dir` | `<repo>/model/registry` | training (`train.py`) and inference (`inference.py`) — one shared setting, not two independent path constants |

### `.env` handling

`.env.example` documents every variable with safe local defaults;
`.env` itself is gitignored. `pydantic-settings`' `env_file=".env"` means a
developer only needs to `cp .env.example .env` to get a working local
config.

### A real bug fixed here

`model_registry_dir` initially triggered a pydantic warning because
`model_` is a namespace pydantic reserves for its own internals.
Fixed with `protected_namespaces=()` in `SettingsConfigDict` rather than
renaming the field to something less descriptive — see Chapter 24.

---

## 18. Logging

### Structured (JSON) logging via `structlog`

```python
structlog.configure(
    processors=[..., structlog.processors.TimeStamper(fmt="iso"), ..., structlog.processors.JSONRenderer()],
    ...
)
```

Every log line is a JSON object with a timestamp, level, and structured
fields — not a free-text string. This is what makes "why did the system do
this" reconstructable after the fact by querying/filtering logs, rather
than grepping through prose.

### Current usage (Phase 10 completed the coverage)

Every business-significant decision now logs, not just app lifecycle:

| Event | Where | What it captures |
|---|---|---|
| `service_starting` / `service_stopping` | `app/main.py` | process lifecycle (Phase 0) |
| `model_loaded` / `model_not_available_at_startup` | `app/main.py` | whether the preloaded model succeeded |
| `prediction_requested_for_unknown_equipment` | `prediction_service.py` | a 404 before it's raised — visible in the log stream, not just the HTTP response |
| `prediction_scored` | `prediction_service.py` | equipment, probability, model version, source (real-time/batch), whether a work order was created |
| `event_publish_failed` | `prediction_service.py`, `work_order_service.py` | a dropped Redis event — previously a silent `except Exception: pass`, now findable |
| `work_order_created` | `work_order_service.py` | priority, recommended priority, and `held_pending_approval` — an audit trail for FR4, not just a database row |
| `work_order_urgent_priority_approved` | `work_order_service.py` | who approved an urgent escalation, and when |
| `batch_scoring_failed_for_equipment` | `nightly_job.py` | one asset's failure during a batch run, without aborting the rest |
| `batch_scoring_complete` | `nightly_job.py` | run summary: total scored, work orders created |
| `request_failed_*` (per exception type) | `error_handlers.py` | every mapped 4xx/5xx, logged centrally |
| `unhandled_exception` | `error_handlers.py` | full traceback for anything unexpected — the client never sees this detail, only the log stream does |

The earlier version of this handbook noted logging covered "only app
startup/shutdown" and explicitly deferred the rest to "the observability
phase" — that phase (10) is now done, and the table above is the result.

### Files

`app/logging_config.py`, plus logging calls throughout
`app/services/*.py`, `app/api/error_handlers.py`, and `app/batch/nightly_job.py`.

---

## 19. Design Patterns

Only patterns that actually exist in this codebase today:

| Pattern | Where | Why it's actually this pattern, not just "organized code" |
|---|---|---|
| **Repository Pattern** | `app/data/repositories/` | All persistence access goes through named functions returning ORM objects; no query logic exists outside this layer |
| **Service Layer Pattern** | `app/services/` | Business rules live independently of both the HTTP layer and the persistence layer; `prediction_service` orchestrates other services/repositories rather than containing raw SQL or route logic |
| **Dependency Injection** | `app/api/deps.py::get_db`, FastAPI's `Depends()` | The API layer never constructs a `Session` itself — it declares a dependency and FastAPI provides it, which is also what makes it trivially swappable in tests |
| **Singleton** | `app/config.py::settings`, `app/ml/inference.py::get_model()` | Both are process-wide, lazily-initialized-once, reused-everywhere instances — settings via `lru_cache`, the model via a manual cached-global pattern (chosen over `lru_cache` here specifically because `force_reload`/`reset_model_cache` need explicit cache-busting, which `lru_cache` doesn't expose cleanly) |
| **Strategy (implicit)** | `risk_policy.evaluate_risk` | The risk-to-priority mapping is isolated as a swappable pure function — a future change to the business rule (e.g., a third priority tier) touches one function, not scattered `if` statements across the service layer |

### Patterns deliberately NOT used (yet)

- **Factory** — no object creation is complex enough yet to warrant one;
  `WorkOrder(...)` is constructed directly in `work_order_service.py`.
- **Builder** — no multi-step object construction exists yet.
- **Unit of Work (as a named, explicit class)** — the concept is applied
  (repositories flush, the service layer commits) but there's no formal
  `UnitOfWork` class wrapping a `Session`; at this codebase's size, the
  service function itself plays that role.

---

## 20. Major Architectural Decisions

### Decision: feature engineering as a pure, shared function (Phase 2)

**Why:** train/serve skew — a classic production ML bug where training and
serving compute "the same" feature subtly differently — is prevented
architecturally (one function, imported everywhere), not just by
discipline ("we'll be careful to keep them in sync").

### Decision: `flush()` in repositories, `commit()` in services (Phase 1,
validated in Phase 6)

**Why:** a repository has no way to know if its write is the only operation
in its transaction. Deciding this in Phase 1 — before there was any
multi-step write to justify it — meant Phase 6's atomic risk-score-plus-
work-order write required zero changes to the repository layer.

### Decision: the urgent-priority human-approval gate (Phase 6, FR4)

**Why:** this is the project's concrete answer to a live, current industry
conversation — agentic AI systems taking autonomous action without human
oversight. The system is explicitly designed so `priority=URGENT` is never
a *direct* consequence of a model output; it's always a *human's* action on
top of the model's *recommendation*. Enforced at the service layer
specifically (`work_order_service.approve_urgent_priority` is the *only*
code path that can write `priority=URGENT`), so it can't be silently
bypassed by a future shortcut elsewhere in the codebase.

### Decision: model schema validation at load time (Phase 5)

**Why:** `PredictiveMaintenanceModel.__init__` refuses to construct if the
loaded model's feature columns don't match the current code's
`FEATURE_COLUMNS`. The alternative — silently predicting with misaligned
columns — would produce a wrong-but-plausible-looking number, which is
worse than an explicit crash.

### Decision: local Pydantic schemas as a temporary stand-in for shared
platform packages (Phase 0)

**Why:** `Equipment`, `WorkOrder`, and the Agent Contract shapes are meant
to eventually live in shared platform-level packages
(`platform-data-contracts`, `platform-agent-sdk`) that don't exist yet.
Rather than block this repo on packages that aren't built, matching local
definitions live here now — documented explicitly as temporary, with the
stated intent that swapping them for real imports later shouldn't require
changing anything else in this repo.

### Decision: synthetic training data with deliberate noise (Phase 3)

**Why:** with no real historical dataset available yet, a synthetic
generator was needed to exercise the training pipeline end-to-end. Making
it *realistically noisy* rather than cleanly separable was a deliberate
choice — a model that never sees a hard example doesn't prove the pipeline
works, it just proves the data was too easy.

---

## 21. Scalability

### Current bottlenecks, honestly assessed

- **Single Postgres instance, no read replicas** — fine at this project's
  scale; would need addressing before real industrial-scale sensor
  ingestion.
- **`SensorReading` as one growing table** — currently unpartitioned. At
  real scale (millions of readings per minute across a real plant), this is
  the first thing to change — either a dedicated time-series database
  (TimescaleDB/InfluxDB) or table partitioning by time range.
- **Model loaded into a single process's memory** — fine for one API
  instance; horizontally scaling the API (multiple instances behind a load
  balancer) means each instance loads its own copy, which is normal and
  fine for a model this size.
- **No caching layer for repeated `/equipment/{id}/risk` reads** (not built
  yet, but worth naming) — Redis (already planned for Phase 7's event bus)
  could double as a read cache for hot lookups later.

### What would change first at real scale

1. Move `SensorReading` ingestion off synchronous request/response entirely
   (a queue-fed ingestion path) rather than assuming readings arrive
   one-by-one via the same service that also serves predictions.
2. Partition or migrate `sensor_reading` to a time-series-optimized store.
3. Separate the batch scoring job (Phase 9) onto its own worker process/
   container so a long-running nightly job never competes with request
   latency on the same process.

### What does NOT need to change

The layered architecture itself — the ML layer, service layer, and
repository layer boundaries hold regardless of scale; only the
*implementations* behind those boundaries (which database, how the model is
served) would need to change.

---

## 22. Security

### Implemented today

- Database credentials sourced from environment variables (`.env`,
  gitignored) — never hardcoded.
- Postgres ENUM types at the database level prevent invalid
  `priority`/`status`/`source` values from being written even if
  application-level validation were somehow bypassed.
- Pydantic validation on every schema — malformed input is rejected before
  reaching business logic (once the API layer exists to receive external
  input — today's only external-facing endpoints are the health checks,
  which accept no input).

### Future Scope

- **Authentication/authorization** on the future API endpoints — not yet
  designed. The Agent Contract endpoint in particular will need to decide
  how it authenticates the future orchestrator repo's calls.
- **Secrets management** beyond `.env` for a real deployment (Render's
  environment variable configuration, not a committed file) — planned for
  Phase 12.
- **Rate limiting** on `/predict` once it exists, to prevent abuse of a
  compute-costing endpoint.
- **Input sanitization for `approved_by`** (currently a free-text string on
  `approve_urgent_priority`) — should eventually be tied to an authenticated
  identity rather than an arbitrary caller-supplied string.

---

## 23. Production Improvements

Realistic next steps beyond this portfolio's current scope, in rough
priority order:

1. Replace the synthetic dataset with a real historical dataset (the
   pipeline's structure — `FEATURE_COLUMNS` in, labeled DataFrame out — was
   designed specifically so this swap requires no change to `train.py`).
2. Comprehensive structured logging across the service layer (today only
   app startup/shutdown is logged — see Chapter 18).
3. A model-drift monitoring job comparing recent `risk_score` distributions
   against training-time distributions.
4. Proper secrets management for a real deployment.
5. Horizontal scaling story for the API once real request volume exists.

---

## 24. Common Bugs and Debugging Notes

This section is the honest paper trail — every real bug found during
development, not just the final clean code.

### Bug 1 — Postgres ENUM types survive `DROP TABLE` (Phase 1)

**Symptom:** `alembic downgrade` then `alembic upgrade` failed with
`type "risk_score_source" already exists`.

**Root cause:** in Postgres, an `ENUM` used by a column is a separate named
database object from the table itself. Dropping the table doesn't drop the
type.

**Fix:** added explicit `sa.Enum(name='...').drop(op.get_bind(), checkfirst=True)`
calls at the end of the migration's `downgrade()`.

**Lesson:** never trust an autogenerated Alembic migration's `downgrade()`
without actually running the full upgrade → downgrade → upgrade cycle
against a real database.

### Bug 2 — synthetic training data was too cleanly separable (Phase 3)

**Symptom:** first training run scored a suspicious ROC-AUC/PR-AUC of
exactly 1.0.

**Root cause:** the synthetic degradation drift was large relative to
sensor noise, and every "degrading" asset degraded identically —
essentially zero class overlap.

**Fix:** added per-asset drift-strength variance, mild unrelated drift on
some "healthy" assets, and ~8% independent label noise.

**Lesson:** a perfect score on a synthetic dataset is a red flag about the
*data*, not evidence the *model* is good — worth fixing even when nothing
is functionally broken.

### Bug 3 — pydantic `protected_namespaces` warning on `model_registry_dir` (Phase 5)

**Symptom:** a `UserWarning` about a field name conflicting with
pydantic's protected `model_` namespace.

**Root cause:** pydantic reserves the `model_` prefix for its own internal
methods (`model_dump()`, etc.); a field genuinely named `model_registry_dir`
triggers a namespace-collision warning even though there's no real
conflict.

**Fix:** `protected_namespaces=()` in `SettingsConfigDict` — chosen over
renaming the field, since `model_registry_dir` is the most accurate,
descriptive name available.

### Bug 4 — test isolation silently broke once services started committing
(Phase 6)

**Symptom:** integration tests started failing with
`duplicate key value violates unique constraint "equipment_pkey"` — but
only after `work_order_service`/`prediction_service` were added.

**Root cause:** the original test fixture rolled back the session in a
`finally` block — but `session.rollback()` after a `session.commit()` has
already happened does nothing; the committed data was already permanently
written and visible to the next test.

**Fix:** rewrote the fixture to bind each test's session to a SAVEPOINT
inside an outer connection-level transaction
(`join_transaction_mode="create_savepoint"`), so `commit()` calls from code
under test only release the savepoint — the outer transaction (and
everything inside it) is unconditionally rolled back when the test ends.

**Lesson:** a test-isolation strategy that works today can silently stop
working the moment the code under test changes how it manages transactions
— this is exactly the kind of bug that wouldn't show up until the specific
test order/interaction that exposes it, which is why running the full
suite twice back-to-back was worth doing as a verification step, not just
running it once and moving on.

### Bug 5 — empty sensor-reading window crashed feature building (Phase 8)

**Symptom:** `KeyError: 'sensor_type'` inside `build_feature_vector`, only
when a test queried an asset with no readings inside the lookback window.

**Root cause:** `_readings_to_dataframe` built a `pd.DataFrame` from a
Python list comprehension without specifying `columns=`. When the input
list was empty (zero readings), `pd.DataFrame([])` produced a frame with
**zero columns**, not the three expected empty columns — so the very next
line, filtering by `readings["sensor_type"]`, raised `KeyError` instead of
returning an empty (and valid) subset.

**Fix:** pass `columns=["sensor_type", "timestamp", "value"]` explicitly to
the `pd.DataFrame(...)` constructor, guaranteeing those columns exist even
with zero rows.

**Lesson:** this bug had been latent since Phase 6 — every prior test
happened to seed readings inside its query window by construction. It
surfaced only once a test (the Agent Contract test) queried with a
timestamp that put a real asset's readings outside the window, which is
exactly the situation a genuinely new/idle piece of equipment would hit in
production. A regression test
(`test_predict_works_for_equipment_with_zero_sensor_readings`) was added
specifically to keep this case covered going forward.

### Bug 6 — the `model_` namespace warning came back (Phase 8)

**Symptom:** the same pydantic `UserWarning` from Bug 3, reappearing for
two new Pydantic models.

**Root cause:** `PredictResponse` and `RiskScoreResponse` both needed a
`model_version` field (the same name that caused Bug 3 in `Settings`), and
each new Pydantic `BaseModel` subclass needs its own `protected_namespaces`
override — it isn't inherited automatically from unrelated classes.

**Fix:** same pattern as Bug 3 — `model_config = ConfigDict(protected_namespaces=())`
on both response models.

**Lesson:** a fix made once in one class doesn't prevent the same warning
elsewhere; worth grep-ing for `model_` prefixed field names across new
Pydantic models when this class of warning reappears, rather than
special-casing each occurrence in isolation.

### Bug 7 — batch job's error isolation rolled back more than the failed asset (Phase 9)

**Symptom:** `sqlalchemy.orm.exc.ObjectDeletedError` on the *second*
equipment asset in a batch-job test, even though only the *first* asset's
model call was made to fail.

**Root cause:** `nightly_job.run_nightly_scoring` calls `db.rollback()`
when one asset's scoring raises, so a bad asset doesn't abort the whole
run. But `session.rollback()` on a savepoint-bound session rolls back to
the *last savepoint* — and the test's seed data had only been `flush()`ed,
not committed, so it was still inside that same savepoint. The rollback
undid both seeded assets, not just the failed one's partial work.

**Fix:** commit the seed data in the test before calling the batch job —
which is also the *accurate* representation of reality: equipment
reference data is always already-committed before a real batch run starts,
never pending mid-transaction the way a test fixture's `flush()`-only
seeding can be.

**Lesson:** `db.rollback()` as a per-item error-isolation strategy is only
as safe as knowing exactly what's "in" the current savepoint when it fires
— it rolls back everything since the last commit, not just the current
loop iteration's work. Real production usage was never at risk here (real
equipment data is committed by other processes before a batch job even
starts) — but the test that caught this is exactly why it's documented
rather than dismissed as "just a test artifact."

### Bug 8 — the Dockerfile could not serve a single prediction (Phase 12)

**Symptom:** not caught by any test — found only by manually reproducing
the Dockerfile's build steps and starting the server that way.

**Root cause:** the Dockerfile, unchanged since Phase 0, only ran
`pip install -r requirements.txt`. `requirements-ml.txt` (lightgbm, shap,
pandas, numpy, scikit-learn, matplotlib) was never installed in the image.
Every test in this project runs directly against the host dev environment,
which has both requirements files installed — so 91 passing tests gave no
signal that the *container* was missing half its dependencies.

**Fix:** `COPY requirements.txt requirements-ml.txt ./` and install both.

**Lesson:** a test suite proves the *code* works; it doesn't prove the
*packaging* is correct unless something actually exercises the packaging.
This is the core reason Phase 12 insisted on running a reproduction of the
real container rather than reviewing the Dockerfile and trusting it looked
right.

### Bug 9 — NaN crashed the exact case Bug 5 was built to fix (Phase 12)

**Symptom:** `POST /predict` against an asset with zero sensor readings —
the precise scenario Bug 5 (Phase 8) fixed the *crash* for — returned a
500 anyway, discovered only when the reproduced-container server was hit
with a real request for a demo asset built specifically to exercise this
case.

**Root cause:** zero readings makes `rolling_mean` legitimately `NaN` (see
Chapter 8) — Bug 5 made this not crash `build_feature_vector`, but nothing
downstream handled a `NaN` reaching the HTTP response. Starlette's
`JSONResponse` rejects raw `NaN` outright (`allow_nan=False`), so the
first `NaN` in `FeatureAttributionResponse` crashed the whole response
with an unhandled `ValueError` — caught by Phase 10's catch-all handler
(which did its job correctly, returning a generic 500), but a 500
nonetheless for a case that should return 200.

**Fix:** `FeatureAttributionResponse` fields typed `float | None`; a
`_json_safe_float()` helper converts `NaN → None` at the response boundary
— representing "no data for this sensor" as JSON `null`, the honest
translation, rather than letting an internal numeric convention (`NaN`
meaning "missing") leak into the wire contract.

**Lesson:** fixing a crash at one layer (Bug 5, the feature-engineering
layer) doesn't guarantee the *value* that stopped crashing is actually
valid everywhere downstream. `NaN` is a legitimate internal representation
but an illegal JSON value — the translation between those two facts has to
happen explicitly somewhere, and "somewhere" turned out to be one layer
further than Bug 5's fix reached. Found only because the system was
actually run end-to-end with a real request against real demo data
representing this exact case — not because it was reviewed.

---

## 25. Interview Questions

*(Growing collection — target 100+ by project completion. Currently ~45,
covering Phases 0–6.)*

### Beginner

1. **What problem does predictive maintenance solve, and why not just do
   preventive maintenance on a schedule?**
   Preventive maintenance guesses at a safe interval regardless of actual
   condition — wasting labor/parts on healthy equipment and still missing
   failures inside the interval. Predictive maintenance uses actual
   condition data, which is strictly more information-efficient.

2. **Why is this a classification problem, not a regression problem?**
   The system predicts a calibrated probability of failure within a fixed
   window, which is what a planner needs to act on — a raw "13.4 days"
   estimate (regression/remaining-useful-life) implies more precision than
   the data likely supports at this stage.

3. **What is a rolling window feature, and why use one here?**
   A statistic (mean, std, rate of change) computed over a recent time
   window rather than a single point-in-time value — necessary because a
   single raw sensor reading has no context (is it rising? normal for this
   asset?) without it.

4. **Why does `rolling_std` return 0.0 instead of NaN when there's only
   one reading?**
   With one data point there's no evidence of variability, but there IS a
   value — `0.0` is a defensible "no observed volatility," distinct from
   `rolling_mean`'s `NaN`, which honestly represents "no data at all" for a
   missing sensor type entirely.

5. **What's the difference between `flush()` and `commit()` in
   SQLAlchemy?**
   `flush()` sends pending changes to the database (assigns IDs, runs
   constraints) within the current transaction, without ending it.
   `commit()` ends the transaction, making changes permanent (or, in this
   project's test fixture, ends a SAVEPOINT rather than the outer
   transaction).

### Intermediate

6. **Why does the repository layer only flush, never commit?**
   The repository doesn't know whether its write is the only operation in
   its unit of work. The caller (service layer) decides the transaction
   boundary — validated concretely in Phase 6, where a risk score and a
   work order need to commit together atomically.

7. **Why PR-AUC instead of accuracy for this model?**
   Failures are rare (class imbalance). A model that never predicts
   failure could still score high accuracy while being useless. PR-AUC
   specifically measures performance on the rare positive class.

8. **What does "calibration" mean, and why does it matter here
   specifically?**
   A calibrated 0.8 probability should correspond to roughly 80% real-world
   failure frequency at that score level. It matters because the
   threshold-based work-order automation is only trustworthy if the
   probability genuinely means what it claims — a well-*ranked* but
   poorly-*calibrated* model could still trigger the wrong threshold
   behavior.

9. **Why split train/test by `equipment_id` instead of randomly by row?**
   Splitting by row would let the model see other timestamps from the same
   asset during training and be tested on that asset again — overstating
   real-world generalization to genuinely unseen equipment.

10. **What is train/serve skew, and how is it prevented here — not just
    "avoided by being careful"?**
    Train/serve skew is when training and serving compute "the same"
    feature subtly differently. It's prevented architecturally: one pure
    function (`build_feature_vector`) is imported by training, real-time
    inference, and (eventually) batch inference — there's no second
    implementation that could drift out of sync.

11. **Why SHAP's `TreeExplainer` instead of LIME?**
    `TreeExplainer` computes exact SHAP values for tree models by
    exploiting the tree structure directly; LIME approximates by
    perturbing inputs and fitting a local surrogate — slower and
    approximate where an exact method is available.

12. **What is the SHAP additivity property, and why test it explicitly?**
    `expected_value + sum(all shap_values) == the model's raw prediction`
    for a given row. Testing it against a genuinely trained model (not a
    mock) is the strongest possible proof that the explainability
    integration is wired correctly — if it fails, something about the
    SHAP/model pairing is broken.

13. **Why does the model wrapper validate `feature_columns` at load time?**
    To fail loudly if a model trained on a different feature schema is
    loaded — silently predicting with misaligned columns would produce a
    wrong-but-plausible number, which is worse than an explicit crash.

14. **Explain the difference between the Repository Pattern and the
    Service Layer Pattern as used in this project.**
    Repositories are thin CRUD wrappers with zero business logic —
    `get_by_id`, `create`. The service layer contains the actual business
    rules (thresholds, the approval gate) and orchestrates one or more
    repository calls into a meaningful operation.

15. **Why is `risk_policy.py` a pure function instead of a method on some
    class?**
    No DB, no model, no I/O — a pure function is trivially unit-testable
    (five tests, zero fixtures) and makes the business rule itself
    reusable and inspectable independent of any orchestration code.

### Advanced

16. **Walk through exactly what happens, end to end, when
    `run_prediction_for_equipment` is called with a probability of 0.95.**
    *(Candidate should trace: equipment lookup → sensor reading fetch →
    feature build → model.predict_with_explanation → risk score persisted
    → risk_policy recommends URGENT → work_order_service holds it at
    ELEVATED with recommended_priority=URGENT → commit → PredictionOutcome
    returned.)*

17. **Why is the urgent-priority approval gate enforced at the service
    layer specifically, rather than, say, the API layer?**
    Because the API layer doesn't exist yet as the only caller — the batch
    job (Phase 9) will also call into this same path. Enforcing the gate in
    `work_order_service` means it's structurally impossible to bypass
    regardless of which future caller triggers a prediction, not just
    inconvenient to bypass from one specific entry point.

18. **This project uses gradient-boosted trees instead of a neural
    network. Defend that choice under pushback — "why not use deep
    learning, it's more powerful"?**
    Power isn't free here: this data volume doesn't provide the scale deep
    learning needs to outperform tree models; SHAP's exact, fast
    `TreeExplainer` doesn't apply to RNNs, which would force a slower,
    approximate explainability method against a project where
    explainability is a first-class requirement; and defending "why the
    simpler tool" under interview scrutiny is itself a signal of judgment,
    not a weakness.

19. **What's a concrete scenario where SHAP's local explanation could be
    misleading if taken at face value?**
    A feature that's highly correlated with the true cause (but isn't
    itself causal) could show a high SHAP value — e.g., if criticality
    tier happens to correlate with asset age in the training data, the
    model might lean on whichever one it saw more variance in, and SHAP
    would faithfully report that reliance — which is still "what the model
    did," not "what's physically true."

20. **How would you extend this system to detect model drift?**
    Compare the distribution of `risk_score.probability` over a recent
    window against the training-time distribution (already tracked per
    prediction, since every prediction — not just actionable ones — is
    persisted). A significant shift would suggest the deployed model's
    assumptions about "normal" no longer match current sensor behavior.

21. **Why use a SAVEPOINT-based test fixture instead of just truncating
    tables between tests?**
    Truncation works but is slower (real DDL/DML per test) and doesn't
    naturally handle nested/rollback semantics when the code under test
    itself manages transactions. A savepoint-nested-in-a-transaction
    approach isolates each test with a single rollback, regardless of how
    many times the code under test commits internally.

22. **The `WorkOrder` schema changed in Phase 6 to add
    `recommended_priority`. Why wasn't this added back in Phase 1 when the
    table was first created?**
    Phase 1's scope was explicitly the data-layer foundation — the
    approval-gate *business rule* didn't exist yet to justify the schema.
    Adding speculative columns for a rule that isn't implemented yet risks
    guessing wrong about the exact shape needed; the column was added
    exactly when the business logic that needed it was built, with a
    clean, tested migration.

23. **Why does `prediction_service` convert ORM rows to a plain DataFrame
    before calling into the ML layer, instead of just passing the ORM
    objects?**
    To preserve the ML layer's strict boundary — `app/ml/` has zero
    knowledge of SQLAlchemy or the database. Keeping that conversion in
    the service layer (not the ML layer) is what keeps the ML code
    reusable and testable without any database at all.

24. **What would break if two processes called `get_model()` at the exact
    same time on first load?**
    *(Good follow-up/probing question — current implementation is not
    thread-safe/process-safe for the very first load: two near-simultaneous
    calls could both see `_cached_model is None` and both load the model
    redundantly. Harmless (both loads succeed, one instance simply gets
    discarded) but wasteful. Worth naming as a known limitation rather than
    claiming false rigor.)*

25. **How does this project's layering compare to a typical Django MVC
    app?**
    Similar underlying idea to "fat models, thin views," but organized
    around explicit layers instead of an MVC convention — the service
    layer here is closer to Django's sometimes-recommended "service
    objects" pattern than to a model method, kept deliberately framework-
    agnostic so it isn't tied to FastAPI-specific concepts at all.

### Phases 9–12: batch processing, observability, testing maturity, deployment

26. **Why does the nightly batch job call the same function as the API
    instead of its own scoring logic?**
    So there is no second implementation of the prediction pipeline to
    keep in sync — `run_prediction_for_equipment` takes a `source`
    parameter (`REAL_TIME` or `BATCH`) and is otherwise identical either
    way. A bug fix or feature change to scoring logic automatically
    applies to both paths.

27. **The batch job isolates per-asset failures with `db.rollback()`. What
    could go wrong with that approach, and how was it actually caught?**
    `session.rollback()` rolls back to the last *savepoint*, not just the
    current loop iteration — if other uncommitted work shares that
    savepoint, it gets undone too. This was caught by a test where seed
    data was only `flush()`ed, not committed, so a rollback for one failed
    asset wiped out a second, unrelated asset's data too. Fixed by
    ensuring reference data is committed before the batch run starts,
    matching real production usage.

28. **Why log `event_publish_failed` instead of leaving the Redis-publish
    failure silent?**
    A silently swallowed exception (`except Exception: pass`) means a
    downstream platform repo could stop receiving events with zero
    visibility into why. Logging it doesn't change the behavior (the
    prediction still succeeds either way) — it changes whether the failure
    is *findable* later.

29. **Why does the API have a catch-all exception handler in addition to
    the specific ones for known business exceptions?**
    The specific handlers (404, 409, 503) cover *known, expected* failure
    states. A catch-all for everything else ensures two things regardless
    of what goes wrong: the full error is logged server-side for
    diagnosis, and the client never sees raw exception text — which could
    leak internal details (a query, a file path, a stack frame).

30. **What is a contract test, and why did this project add that category
    in Phase 11 instead of earlier?**
    A contract test verifies the *shape* this repo exposes externally
    (the Agent Contract, or a Pydantic schema's claim to match its ORM
    model) stays valid — independent of whether any specific route uses it
    correctly. It was added once there was an actual external contract
    worth verifying (the Agent Contract, built in Phase 8) — building the
    test category before the contract existed would have had nothing to
    test.

31. **`AgentResponse.confidence` was originally an unconstrained float.
    Why does that matter, and what changed?**
    An unconstrained float allows a bug to silently produce a nonsensical
    confidence value (e.g. 5.0), which a downstream consumer treating it
    as a probability would misinterpret without any error being raised.
    Constraining it to `Field(ge=0.0, le=1.0)` makes an invalid value fail
    fast at construction time instead of propagating.

32. **Why does the Dockerfile bake the trained model into the image
    instead of mounting it as a volume or fetching it from external
    storage?**
    Baking is the simplest option that's actually deployable as-is on a
    platform like Render (no persistent disk needed). A volume mount would
    require a separate persistent-disk feature and doesn't solve "how does
    the model get there" in production any more than baking does. External
    storage (S3, a model registry service) would enable live model
    hot-swapping without a redeploy — real capability, but nothing in this
    project currently has a live retraining trigger that would use it, so
    the added complexity wouldn't pay for itself yet.

33. **A 91-test suite with 97% coverage didn't catch that the Docker image
    was missing half its dependencies. Why not, and what does that imply
    about test coverage as a metric?**
    Every test in this project runs directly against the host development
    environment, which already has both `requirements.txt` and
    `requirements-ml.txt` installed — no test ever exercises what's
    actually *inside the built image*. High coverage proves the *code
    paths* are exercised; it says nothing about whether the *packaging*
    that ships those code paths is correct. That gap is exactly why
    Phase 12 insisted on literally reproducing the container's build steps
    and hitting a real running instance, rather than treating a passing
    test suite as sufficient proof of deployability.

34. **Walk through the NaN bug (Bug 9) end to end — what broke, and why
    didn't tests catch it?**
    An asset with zero sensor readings produces `NaN` in
    `rolling_mean` — a legitimate internal value (see Chapter 8). That
    `NaN` flowed untouched into the API response, where Starlette's
    `JSONResponse` rejects raw `NaN` outright, crashing the request with a
    500. No test caught it because the existing zero-readings regression
    test (from Bug 5) asserted the *response succeeded*, but used a fake
    model returning ordinary float attributions — it never exercised a
    *real* `NaN` reaching the actual response serialization step. Only
    hitting a live server with the real zero-history demo asset surfaced
    it.

*(This section will continue to grow with any future work on this
project — currently ~34 questions across four phases' worth of real,
project-specific decisions and bugs, not generic ML trivia.)*

---

## 26. HR Interview Questions

### "Tell me about a challenging project you worked on."

Frame around a real decision, not just "I built an ML app": *"I built an
enterprise-style predictive maintenance system, and the part I'm most proud
of isn't the model — it's a design decision I made around human oversight.
The system never lets a model prediction directly mark a work order as
urgent; it always requires a human to approve that escalation. I built that
because it mirrors a real, current conversation in the AI industry about
agent autonomy and trust, and I wanted my project to take a real position
on it rather than just wire an API together."*

### "How do you handle disagreement about technical decisions?"

Point to a real self-correction in this project: *"During development, my
first version of a synthetic training dataset scored a perfect 1.0 on every
metric. Instead of treating that as a win, I recognized it as a red flag —
a perfect score usually means the data is unrealistically easy, not that
the model is good — and I went back and added realistic noise until the
metrics looked like something a reviewer would actually trust."*

### "How do you approach testing?"

*"I split tests by what they're actually verifying — pure business logic
gets fast unit tests with no dependencies, database interactions get real
integration tests against a real Postgres instance rather than mocks, and
ML correctness gets its own category that trains real models and checks
mathematical properties like SHAP's additivity, not just 'does it run
without crashing.'"*

### "Describe a time you found and fixed a bug."

Use Bug 4 from Chapter 24 (the test isolation regression) — it's a strong
story because it shows the bug was caused by a *good* change (the service
layer correctly starting to own its transaction boundary) surfacing a
*latent* issue elsewhere, and the fix required understanding SQLAlchemy
transaction semantics at a deeper level than "just add try/except."

---

## 27. Resume Talking Points

### Resume bullet (one line)

> Built a production-style predictive maintenance service (Python,
> FastAPI, PostgreSQL, LightGBM, SHAP) with a layered architecture,
> group-aware model validation, and a human-approval gate for
> high-risk automated actions.

### Resume bullet (expanded, 2–3 lines)

> Designed and implemented an enterprise-style predictive maintenance
> system predicting equipment failure risk from sensor time-series data,
> with SHAP-based root-cause explainability and a governed human-approval
> workflow for high-severity automated actions. Enforced strict train/serve
> consistency via a shared feature-engineering module and validated model
> correctness with property-based tests (SHAP additivity) against real
> trained models rather than mocks.

### LinkedIn summary (short)

> I'm building an enterprise-style AI platform for manufacturing, starting
> with a predictive maintenance service that predicts equipment failure
> risk, explains its reasoning with SHAP, and routes high-risk
> recommendations through a human-approval step before anything urgent gets
> actioned automatically.

### Recruiter explanation (30 seconds, spoken)

*"I built a system that predicts when factory equipment is about to fail,
using sensor data and a gradient-boosted model, and explains which sensors
are driving that prediction. The part recruiters usually find interesting
is that I designed it so the AI can't unilaterally mark something urgent —
a human always has to approve that step, which mirrors real conversations
happening right now about AI agent autonomy."*

---

## 28. Project Explanation Scripts

### 30 seconds

*"I built a predictive maintenance system for manufacturing equipment — it
takes sensor data, predicts failure risk with a gradient-boosted model,
explains which sensors are driving that risk using SHAP, and creates work
orders automatically — but with a human-approval gate before anything gets
marked urgent. It's built with a clean layered architecture: API, service,
ML, and data layers, each independently testable."*

### 2 minutes

*(30-second version, plus:)* *"The core engineering challenge I focused on
was avoiding train/serve skew — a common bug where the feature computation
used during training subtly differs from what's used at prediction time. I
solved that architecturally: there's exactly one feature-engineering
function, imported by training and both inference paths, so there's no
second implementation to drift out of sync. I also put real effort into
explainability — not just calling a SHAP library, but writing a test that
verifies SHAP's mathematical additivity property against an actually
trained model, which is the strongest correctness check available for that
kind of integration. And the human-approval gate for urgent work orders is
enforced at the service layer specifically, so there's no code path that
can bypass it."*

### 5 minutes

*(2-minute version, plus:)* walk through the layered architecture diagram
(Chapter 3), explain the repository-vs-service-layer split and why
repositories only flush while services commit (Chapter 12–13), and tell
the Bug 4 story from Chapter 24 as a concrete example of catching a
regression through disciplined testing rather than luck.

### 10 minutes

*(5-minute version, plus:)* go deeper on the ML design decisions (Chapter
9's algorithm-selection table), the model registry/versioning approach
(Chapter 9), and close with the roadmap — what Phases 7–13 will add (event
publishing, the real-time API, the batch job, and eventually how this one
repo fits into a larger multi-repo enterprise AI platform).

---

## 29. Important Things to Remember

- `FEATURE_COLUMNS` is the single most load-bearing constant in this
  codebase — training, inference, and the model's own schema validation
  all depend on it staying in sync.
- Repositories `flush()`, services `commit()` — never the other way around.
- The urgent-priority approval gate has exactly one legitimate code path:
  `work_order_service.approve_urgent_priority`.
- Every prediction is persisted to `risk_score`, tagged `source` (real-time
  vs. batch) — not just the ones that trigger a work order.
- The test suite has four categories for a reason — don't add a
  DB-dependent test to `tests/unit/`, don't add an ML-training test to
  `tests/integration/`, and external-shape checks belong in `tests/contract/`,
  not duplicated into `tests/integration/`.
- Always run the full test suite **twice in a row** after touching test
  fixtures or transaction-handling code — Bug 4 and Bug 7 both involved
  transaction/savepoint behavior that only broke under specific conditions.
- Any Alembic migration touching an Enum column needs its `downgrade()`
  checked by hand — autogenerate does not reliably handle Postgres enum
  type drops.
- `NaN` is a legitimate internal value (missing sensor data) but illegal
  JSON — anywhere a feature value could reach an HTTP response, it needs
  the same `_json_safe_float()` treatment Bug 9 added, not just wherever it
  was first noticed.
- A passing test suite proves the code works; it does NOT prove the
  Docker image is correct, since every test in this project runs against
  the host environment, not the actual built container (Bug 8). Trust a
  reproduction of the real build/run steps, not a code review of the
  Dockerfile.
- The model is baked into the Docker image, not fetched externally —
  retraining means rebuild + redeploy, not a live hot-swap. This is a
  documented, deliberate scope decision (Chapter 16), not an oversight.

---

## 30. Final Revision Notes (20–30 minute pre-interview read)

**The one-sentence pitch:** a predictive maintenance service with SHAP
explainability and a human-approval gate for high-risk automated actions,
built with a layered architecture designed around one core principle —
train/serve consistency enforced structurally, not by discipline — and
deployed with the same "verify by actually running it" discipline applied
throughout every phase, not just the code.

**The five things to have crisp answers for:**
1. Why gradient-boosted trees over deep learning (Chapter 9's table).
2. What train/serve skew is and how it's architecturally prevented
   (Chapter 8, `FEATURE_COLUMNS` — and how the batch job, Chapter 6 Phase
   9, reuses the exact same path rather than a second implementation).
3. What the SHAP additivity property is and why it's tested against a real
   model (Chapter 10).
4. What FR4's approval gate is and why it's enforced at the service layer
   (Chapter 20).
5. One real bug you found and fixed, with the actual root cause — not "I
   fixed some bugs." Bug 8 and Bug 9 (Chapter 24) are strong choices here
   specifically because they were only caught by actually running the
   system, not by code review or a passing test suite — a good story about
   verification discipline, not just debugging.

**If asked "what would you do differently":** two honest, specific answers
now available instead of one — (1) comprehensive service-layer logging was
deferred to Phase 10 rather than built alongside the business logic, which
would have made earlier bugs slightly faster to diagnose; (2) no test in
this project exercises the actual built Docker image, which is exactly why
Bug 8 (missing ML dependencies) went undetected for eleven phases — a
container-level smoke test in CI would have caught it months earlier than
manual Phase 12 verification did.

**If asked "what's next":** Phase 13 (documentation & portfolio polish —
the only remaining phase, and notably the only one that maps to no FR/NFR;
every phase before it added or hardened runtime capability, this one is
purely about making the finished work legible to a reader), and eventually
this repo becoming one of several in a larger GitHub-org-based enterprise
AI platform, alongside the program-level brief already written for the
next four prioritized projects (Maintenance Copilot, Quality Intelligence,
Supplier Risk & RFQ, Finance Exception Intelligence).
