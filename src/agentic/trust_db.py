"""
TrustDB — paper-faithful trust-score policy from ZTA-FL Section V.A.

Implements the trust-score state machine described in Singh, Roy & So
(2025), arXiv:2512.23809, Section V.A "TrustDB Policy".  Every rule and
constant in this module corresponds to a specific clause in the paper:

* Initialization
    "New agents start at τ_i = 0.7 after successful first attestation."

* Positive updates
    "After each successful round with s_i > μ_s (above-average SHAP
     stability), τ_i ← min(1, τ_i + 0.02)."

* Penalties
    "Failed attestation or SHAP filtering triggers τ_i ← τ_i × 0.5.
     Agents with τ_i < τ_min = 0.6 enter quarantine."

* Quarantine and remediation
    "Quarantined agents must pass 5 consecutive attestations with valid
     PCRs before rejoining (τ_i reset to 0.65)."

* PCR drift handling
    "Legitimate firmware updates are pre-registered with signed manifests;
     the fog node updates PCR_ref upon verifying the manufacturer
     signature, avoiding false rejections."

The policy is intentionally rule-based and deterministic so that the
behaviour reproduced by ``scripts/run_agentic_experiment.py`` is the
behaviour analysed in the paper.  Extensions (learned policies, LLM
reasoning) are in :mod:`src.agentic.policies` and are clearly marked as
beyond-paper additions.
"""

from __future__ import annotations

import enum
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Paper-exact constants (Section V.A)
# ---------------------------------------------------------------------------

TAU_INIT          = 0.7    # initialisation after first attestation
TAU_REJOIN        = 0.65   # value assigned when a quarantined agent rejoins
TAU_MIN           = 0.6    # quarantine threshold
TAU_REWARD_DELTA  = 0.02   # increment on positive SHAP-above-mean round
TAU_PENALTY_RATIO = 0.5    # multiplicative penalty on failed round
REJOIN_ATTESTS    = 5      # consecutive clean attestations required to rehab
DELTA_T_MAX_S     = 60.0   # max attestation token age (Section V.A)


class AgentStatus(str, enum.Enum):
    UNKNOWN     = "UNKNOWN"     # before first attestation
    ACTIVE      = "ACTIVE"      # τ_i ≥ τ_min
    QUARANTINED = "QUARANTINED" # τ_i < τ_min, awaiting rehabilitation


@dataclass
class TrustRecord:
    """Per-agent persistent trust state, mirroring the paper's TrustDB row."""
    agent_id:         str
    tau:              float = 0.0           # trust score in [0, 1]
    status:           AgentStatus = AgentStatus.UNKNOWN
    consecutive_clean_attests: int = 0
    rounds_seen:      int = 0
    times_quarantined: int = 0

    # Audit history (most-recent at the back)
    last_event:       str = ""
    last_event_time:  float = 0.0


# ---------------------------------------------------------------------------
# TrustDB
# ---------------------------------------------------------------------------

class TrustDB:
    """
    The TrustDB instance the fog node holds.  All state transitions go
    through this class so the policy is single-sourced and auditable.
    """

    def __init__(
        self,
        tau_init:         float = TAU_INIT,
        tau_rejoin:       float = TAU_REJOIN,
        tau_min:          float = TAU_MIN,
        tau_reward:       float = TAU_REWARD_DELTA,
        tau_penalty:      float = TAU_PENALTY_RATIO,
        rejoin_attests:   int   = REJOIN_ATTESTS,
        audit_path:       Optional[str] = None,
    ) -> None:
        self.tau_init       = tau_init
        self.tau_rejoin     = tau_rejoin
        self.tau_min        = tau_min
        self.tau_reward     = tau_reward
        self.tau_penalty    = tau_penalty
        self.rejoin_attests = rejoin_attests
        self._records: Dict[str, TrustRecord] = {}
        self._audit_path = audit_path
        if audit_path:
            os.makedirs(os.path.dirname(audit_path) or ".", exist_ok=True)
            open(audit_path, "w").close()

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def __contains__(self, agent_id: str) -> bool:
        return agent_id in self._records

    def get(self, agent_id: str) -> TrustRecord:
        if agent_id not in self._records:
            self._records[agent_id] = TrustRecord(agent_id=agent_id)
        return self._records[agent_id]

    def all(self) -> List[TrustRecord]:
        return list(self._records.values())

    def status_counts(self) -> Dict[str, int]:
        counts = {s.value: 0 for s in AgentStatus}
        for r in self._records.values():
            counts[r.status.value] += 1
        return counts

    # ------------------------------------------------------------------
    # Paper-spec state transitions
    # ------------------------------------------------------------------

    def first_attestation(self, agent_id: str) -> TrustRecord:
        """Section V.A initialisation: new agent admitted after first attest."""
        r = self.get(agent_id)
        r.tau                       = self.tau_init
        r.status                    = AgentStatus.ACTIVE
        r.consecutive_clean_attests = 1
        self._log(r, "first_attestation",
                  f"τ_i ← {self.tau_init} (paper Section V.A)")
        return r

    def positive_round(self, agent_id: str,
                       shap_above_mean: bool = True) -> TrustRecord:
        """Section V.A positive update rule: ``τ_i ← min(1, τ_i + 0.02)``."""
        r = self.get(agent_id)
        if r.status == AgentStatus.UNKNOWN:
            return self.first_attestation(agent_id)
        if shap_above_mean:
            r.tau = min(1.0, r.tau + self.tau_reward)
        r.consecutive_clean_attests += 1
        r.rounds_seen += 1
        # Quarantined agent rehabilitates after REJOIN_ATTESTS clean rounds
        if (r.status == AgentStatus.QUARANTINED
                and r.consecutive_clean_attests >= self.rejoin_attests):
            r.tau    = self.tau_rejoin
            r.status = AgentStatus.ACTIVE
            self._log(r, "rehabilitated",
                      f"passed {self.rejoin_attests} clean attestations; "
                      f"τ_i ← {self.tau_rejoin}")
        else:
            self._log(r, "positive_round",
                      f"τ_i={r.tau:.3f}  consec_clean={r.consecutive_clean_attests}")
        return r

    def penalty(self, agent_id: str, reason: str) -> TrustRecord:
        """Section V.A penalty rule: ``τ_i ← τ_i × 0.5``; quarantine if < τ_min."""
        r = self.get(agent_id)
        if r.status == AgentStatus.UNKNOWN:
            r.status = AgentStatus.ACTIVE
            r.tau    = self.tau_init
        r.tau = r.tau * self.tau_penalty
        r.consecutive_clean_attests = 0
        r.rounds_seen += 1
        if r.tau < self.tau_min and r.status != AgentStatus.QUARANTINED:
            r.status = AgentStatus.QUARANTINED
            r.times_quarantined += 1
            self._log(r, "quarantined",
                      f"{reason}; τ_i={r.tau:.3f} < τ_min={self.tau_min}")
        else:
            self._log(r, "penalty", f"{reason}; τ_i={r.tau:.3f}")
        return r

    def is_admitted(self, agent_id: str) -> bool:
        """A client is admitted to aggregation iff status == ACTIVE."""
        if agent_id not in self._records:
            return False
        return self._records[agent_id].status == AgentStatus.ACTIVE

    # ------------------------------------------------------------------
    # Audit
    # ------------------------------------------------------------------

    def _log(self, r: TrustRecord, event: str, detail: str) -> None:
        r.last_event      = event
        r.last_event_time = time.time()
        msg = (f"[trust_db] {r.agent_id} {event}: {detail}  "
               f"status={r.status.value}")
        logger.info(msg)
        if self._audit_path:
            with open(self._audit_path, "a") as f:
                f.write(json.dumps({
                    "ts":      r.last_event_time,
                    "agent":   r.agent_id,
                    "event":   event,
                    "detail":  detail,
                    "tau":     round(r.tau, 4),
                    "status":  r.status.value,
                    "consec_clean":      r.consecutive_clean_attests,
                    "rounds_seen":       r.rounds_seen,
                    "times_quarantined": r.times_quarantined,
                }) + "\n")
