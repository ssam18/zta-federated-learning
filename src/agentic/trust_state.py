"""
Per-client trust state machine and ledger.

The trust ledger keeps a small persistent record for every client across
rounds.  This is what makes the agentic layer *stateful* rather than
threshold-on-current-round-only: the policy can ask "has this client been
producing anomalous updates for three consecutive rounds?" rather than
seeing each round in isolation.

State transitions
-----------------

::

    NEW ──► PROBATION ──► TRUSTED
              ▲ │            │
              │ ▼            ▼
              QUARANTINE ──► BLOCKED

* ``NEW`` — first observation.  Update is accepted with reduced weight
  while the policy gathers signal.
* ``PROBATION`` — recently joined or recently exited quarantine.  Receives
  full weight only after ``promotion_window`` consecutive clean rounds.
* ``TRUSTED`` — passing all checks.  Full weight in aggregation.
* ``QUARANTINE`` — currently failing one or more checks.  Update is observed
  but not aggregated.  May return to ``PROBATION`` after ``quarantine_cool_off``
  clean rounds.
* ``BLOCKED`` — repeatedly quarantined.  Permanently rejected for the
  remainder of the run.

The state and the per-round signal traces are written to
``results/agentic_audit.jsonl`` so an external auditor can replay every
decision.
"""

from __future__ import annotations

import enum
import json
import os
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional


class TrustState(str, enum.Enum):
    NEW         = "NEW"
    PROBATION   = "PROBATION"
    TRUSTED     = "TRUSTED"
    QUARANTINE  = "QUARANTINE"
    BLOCKED     = "BLOCKED"


@dataclass
class TrustHistory:
    """Per-client running record."""

    client_id: str
    state: TrustState = TrustState.NEW
    rounds_in_state: int = 0
    quarantine_count: int = 0           # how many times quarantined ever
    consecutive_clean: int = 0          # rounds in a row with no flags
    consecutive_flagged: int = 0        # rounds in a row with at least one flag
    recent_signals: Deque[Dict[str, float]] = field(
        default_factory=lambda: deque(maxlen=10)
    )

    def summary(self) -> Dict[str, float]:
        """Return aggregate statistics over the recent-signals window."""
        if not self.recent_signals:
            return {
                "n_seen": 0,
                "mean_norm_z": 0.0,
                "max_norm_z": 0.0,
                "mean_cosine_global": 1.0,
                "min_cosine_global": 1.0,
                "mean_shap": 0.0,
                "max_shap": 0.0,
            }
        zs    = [s["update_norm_zscore"] for s in self.recent_signals]
        cgs   = [s["cosine_to_global"]   for s in self.recent_signals]
        shaps = [s["shap_stability"]     for s in self.recent_signals]
        return {
            "n_seen":             float(len(self.recent_signals)),
            "mean_norm_z":        float(sum(zs) / len(zs)),
            "max_norm_z":         float(max(zs)),
            "mean_cosine_global": float(sum(cgs) / len(cgs)),
            "min_cosine_global": float(min(cgs)),
            "mean_shap":          float(sum(shaps) / len(shaps)),
            "max_shap":           float(max(shaps)),
        }


class TrustLedger:
    """Container for per-client :class:`TrustHistory` records and audit log."""

    def __init__(
        self,
        promotion_window: int = 2,
        quarantine_cool_off: int = 2,
        block_after: int = 3,
        audit_path: Optional[str] = None,
    ) -> None:
        self.promotion_window     = promotion_window
        self.quarantine_cool_off  = quarantine_cool_off
        self.block_after          = block_after
        self._records: Dict[str, TrustHistory] = {}
        self._audit_path = audit_path
        if audit_path is not None:
            os.makedirs(os.path.dirname(audit_path) or ".", exist_ok=True)
            # Truncate at start of run
            open(audit_path, "w").close()

    # ------------------------------------------------------------------
    # Lookup / mutation
    # ------------------------------------------------------------------

    def get(self, client_id: str) -> TrustHistory:
        if client_id not in self._records:
            self._records[client_id] = TrustHistory(client_id=client_id)
        return self._records[client_id]

    def enrich(self, signals) -> None:
        """Attach the per-client running summary to a ClientSignals object."""
        signals.history_summary = self.get(signals.client_id).summary()

    def record(self, signals, decision) -> None:
        """Update internal state from (signals, decision) and append to audit log."""
        h = self.get(signals.client_id)

        # Append signal trace (numbers only — JSON serialisable)
        h.recent_signals.append({
            "round":              signals.round,
            "update_norm_zscore": signals.update_norm_zscore,
            "cosine_to_global":   signals.cosine_to_global,
            "shap_stability":     signals.shap_stability,
        })

        # State transitions driven by the decision
        flagged = decision.action.value in ("quarantine", "block", "discount")
        if flagged:
            h.consecutive_flagged += 1
            h.consecutive_clean = 0
        else:
            h.consecutive_clean += 1
            h.consecutive_flagged = 0

        new_state = self._next_state(h, decision)
        if new_state != h.state:
            h.rounds_in_state = 0
            h.state = new_state
            if new_state == TrustState.QUARANTINE:
                h.quarantine_count += 1
        else:
            h.rounds_in_state += 1

        # Audit
        if self._audit_path is not None:
            with open(self._audit_path, "a") as f:
                f.write(json.dumps({
                    "round":     signals.round,
                    "client_id": signals.client_id,
                    "signals":   signals.as_dict(),
                    "decision":  decision.as_dict(),
                    "state":     h.state.value,
                }) + "\n")

    def _next_state(self, h: TrustHistory, decision) -> TrustState:
        action = decision.action.value

        if action == "block":
            return TrustState.BLOCKED

        if h.state == TrustState.BLOCKED:
            return TrustState.BLOCKED  # absorbing

        if action == "quarantine":
            if h.quarantine_count + 1 >= self.block_after:
                return TrustState.BLOCKED
            return TrustState.QUARANTINE

        if h.state == TrustState.QUARANTINE:
            if h.consecutive_clean >= self.quarantine_cool_off:
                return TrustState.PROBATION
            return TrustState.QUARANTINE

        if h.state in (TrustState.NEW, TrustState.PROBATION):
            if h.consecutive_clean >= self.promotion_window:
                return TrustState.TRUSTED
            return TrustState.PROBATION

        return TrustState.TRUSTED

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def state_distribution(self) -> Dict[str, int]:
        """Return a count of clients per trust state (for run summaries)."""
        counts = {s.value: 0 for s in TrustState}
        for h in self._records.values():
            counts[h.state.value] += 1
        return counts

    def all_records(self) -> List[TrustHistory]:
        return list(self._records.values())
