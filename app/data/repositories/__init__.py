# Data-access functions only — no business rules. Each function takes
# a Session and returns ORM objects (or None). Callers (the service
# layer, added in a later milestone) decide what those results mean.
