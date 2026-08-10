# Deployment

This repo is deployment-ready: `render.yaml` describes the full service
topology (API, Postgres, Redis, nightly batch job as a Cron Job), and
`Dockerfile` produces a self-contained image with the trained model already
baked in. Actually going live requires a few manual steps in your own Render
account — there's no way to fully automate "connect a GitHub repo to a cloud
account" from inside the repo itself.

## Deploying to Render

1. Push this repo to GitHub (if not already there).
2. In the Render dashboard: **New → Blueprint**, point it at this repo.
   Render reads `render.yaml` and provisions the web service, the Postgres
   database, the Redis instance, and the nightly Cron Job together.
3. No secrets need to be set manually — `DATABASE_URL` and `REDIS_URL` are
   populated automatically from the provisioned Postgres/Redis instances via
   `fromDatabase` / `fromService` in `render.yaml`.
4. Wait for the first deploy. `preDeployCommand: alembic upgrade head` runs
   the migrations before traffic is routed to the new version.
5. Once live, seed demo data:
   ```bash
   # from Render's shell for the web service, or run locally against
   # the deployed DATABASE_URL
   python -m app.scripts.seed_demo_data
   ```
6. Verify: `GET https://<your-service>.onrender.com/health/ready` should
   return `{"status": "ready"}`. If it returns 503, the model failed to
   load — see "Known limitations" below.

`render.yaml` was written from Render's documented Blueprint syntax but has
not been verified against a live Render deployment (this repo was built in
an environment with no access to Render's platform). Treat field names as
probably-right, not guaranteed-right, on the first real deploy — Render's
own error messages on a failed Blueprint parse are usually specific enough
to fix quickly.

## Running locally instead

No cloud account needed for this:

```bash
docker compose up --build
# in another terminal, once the api service is healthy:
docker compose exec api alembic upgrade head
docker compose exec api python -m app.scripts.seed_demo_data
curl http://localhost:8000/health/ready
```

## How the trained model gets into the container

The Dockerfile bakes whatever is currently in `model/registry/` into the
image at build time (`COPY model ./model`). This was a deliberate choice for
this project's scope: simplest possible option, and it matches the platform
architecture doc's "hero-path deployment" approach for a standalone repo.

**Trade-off, stated plainly:** retraining the model means rebuilding and
redeploying the image — there's no live hot-swap. A real production system
at larger scale would instead fetch a versioned model artifact from external
storage (S3, a proper model registry service) at container startup, so a new
model version could ship without a full redeploy. That's real, meaningful
complexity with no corresponding capability in this project today (there's
no live retraining trigger to hot-swap for) — documented here as a named
future improvement rather than built now.

## Known limitations

- **Model updates require a rebuild.** See above.
- **Render's free-tier Postgres/Redis** may have connection limits or
  idle-sleep behavior that adds cold-start latency to a demo link — a UX
  characteristic of the free tier, not a bug in this repo.
- **No authentication on any endpoint.** Acceptable for a portfolio demo;
  flagged as a real gap for anything beyond that (see the handbook's
  Security chapter).
- **The nightly Cron Job's `dockerCommand` overrides the web service's
  container command** to run the batch job instead of the API — same
  image, different entrypoint, exactly as designed back in Phase 0's
  Docker strategy.
