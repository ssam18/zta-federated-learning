"""
Observability for the agentic ZTA-FL pipeline.

Two channels:

* **Structured logger** — standard ``logging`` configured per
  :class:`~src.agentic.config.ObservabilityConfig`.  Used for human-readable
  operational events (decisions, state transitions, errors).

* **Metrics sink** — JSONL writer for one record per (round, metric) tuple.
  This is the contract that downstream dashboards or audit pipelines consume
  in production.  In a real deployment, the sink would back onto Prometheus
  or OpenTelemetry; the local file is a stand-in that makes the pipeline
  fully self-contained for the public release.

The sink is intentionally append-only and event-shaped — every record is a
flat dict with an ``event`` field so a downstream consumer can route by
event type.  This is the same shape SektorCERT-class incident-response
playbooks expect from operational telemetry.
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional


def configure_logging(level: str = "INFO",
                      fmt: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                      ) -> None:
    """Idempotent logging setup; safe to call multiple times."""
    root = logging.getLogger()
    if root.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(fmt))
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))


class MetricsSink:
    """Append-only JSONL metrics writer with explicit close()."""

    def __init__(self, path: str, run_id: Optional[str] = None) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path
        # Truncate at start of run; downstream reader is event-stream aware
        open(path, "w").close()
        self.run_id = run_id or f"run-{int(time.time())}"
        self._logger = logging.getLogger(__name__)

    def emit(self, event: str, **fields: Any) -> None:
        rec = {
            "ts":     time.time(),
            "run_id": self.run_id,
            "event":  event,
        }
        rec.update(fields)
        with open(self.path, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")

    @contextmanager
    def round(self, round_number: int) -> Iterator["MetricsSink"]:
        t0 = time.time()
        self.emit("round_start", round=round_number)
        try:
            yield self
        finally:
            self.emit("round_end", round=round_number,
                      duration_s=round(time.time() - t0, 4))


class NullMetricsSink(MetricsSink):
    """No-op sink for tests and lightweight runs (skips file I/O)."""

    def __init__(self) -> None:  # pragma: no cover  -- trivial
        self.path   = "/dev/null"
        self.run_id = "null"

    def emit(self, event: str, **fields: Any) -> None:  # pragma: no cover
        pass
