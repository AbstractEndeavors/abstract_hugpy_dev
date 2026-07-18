"""chaos — the chaos-and-learn exerciser (p1, EPOCH CLOSER).

Randomly exercises the live assortment (models x cards x alloc modes x ctx%)
through the REAL public paths and records ONE predicted-vs-measured observation
per trial, so the t28 learner can learn placement templates from measured
reality. See ``schema.py`` (the contract) and ``SCHEMA.md``.

Entry point:  python -m abstract_hugpy_dev.chaos.runner  (or bin/hugpy-chaos)

This module is SELF-CONTAINED: it never edits worker-agent internals, flex.py,
or need-pricing (the learner's / worker's turf). It only drives HTTP and reads
/models/<key>/meta for central's own cheap prediction."""

from .schema import SCHEMA_VERSION  # noqa: F401

__all__ = ["SCHEMA_VERSION"]
