# Nightly batch scoring lives here (nightly_job.py). It composes the
# same service-layer functions the real-time API uses — there is no
# separate batch scoring implementation, only a different caller and
# a different RiskScoreSource tag.
