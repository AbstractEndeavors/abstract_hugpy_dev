from .imports import *
# jobs.py — compat shim. The Job/JobStore that used to live here (download
# jobs only) is now the shared, all-transport store in comms.jobs (F5): one
# schema for chat requests AND downloads, canonical lifecycle
# pending→processing→streaming→(done|cancelled|failed), with the old
# queued/running/completed names normalized on write. Download callers keep
# importing from here unchanged; the singleton is the same store every other
# transport enqueues into.
from abstract_hugpy_dev.comms.jobs import (
    CANONICAL_STATUSES,
    LEGACY_FOR_CANONICAL,
    TERMINAL_STATUSES,
    Job,
    JobError,
    JobStore,
    job_store,
    normalize_status,
)
