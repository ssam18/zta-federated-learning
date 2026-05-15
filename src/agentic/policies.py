"""
Decision policies for the agentic layer.

Three pluggable implementations are provided:

* :class:`ThresholdPolicy` — hand-crafted multi-signal rules.  Fast, fully
  auditable, no training data required.  Used as the production default.

* :class:`LearnedPolicy` — small MLP that consumes the signal vector and
  outputs an action distribution.  Trained offline on labelled
  (signals, action) pairs harvested from threshold-policy runs and red-team
  exercises.  Falls back to the threshold policy when not yet trained.

* :class:`LLMPolicy` — pluggable interface for an LLM-backed reasoning
  policy.  The default implementation calls a user-supplied callable; the
  wiring is in place but no specific provider is bundled (so the package
  has no external API dependency).

All three return :class:`Decision` objects so the rest of the agentic
pipeline is policy-agnostic.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn as nn

from src.agentic.signals import ClientSignals
from src.agentic.trust_state import TrustHistory, TrustState


class Action(str, enum.Enum):
    ACCEPT     = "accept"
    DISCOUNT   = "discount"
    QUARANTINE = "quarantine"
    BLOCK      = "block"


@dataclass
class Decision:
    """Output of a policy evaluation for one (client, round)."""

    action: Action
    weight: float                 # 0.0 to 1.0; aggregator multiplies the update
    confidence: float             # policy's self-reported certainty in [0, 1]
    reason: str                   # human-readable explanation for audit log

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["action"] = self.action.value
        return d


class Policy:
    """Abstract base class.  Subclasses implement :meth:`decide`."""

    name: str = "base"

    def decide(self, signals: ClientSignals, history: TrustHistory) -> Decision:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 1. ThresholdPolicy — hand-crafted rules
# ---------------------------------------------------------------------------

class ThresholdPolicy(Policy):
    """
    Rule-based policy that combines hard checks (attestation) with soft
    multi-signal anomaly scoring.

    Decision logic, in order:

    1. **Hard rejects**: invalid attestation, replay-aged token, BLOCKED state.
    2. **Trajectory-driven escalation**: ``QUARANTINE`` when accumulated
       evidence over the trailing window exceeds the soft thresholds.
    3. **Single-round anomaly**: ``DISCOUNT`` when this round's signals are
       borderline.
    4. **Otherwise**: ``ACCEPT`` with weight derived from the trust state
       (``NEW``/``PROBATION`` get fractional weight; ``TRUSTED`` gets 1.0).
    """

    name = "threshold"

    def __init__(
        self,
        norm_z_quarantine:    float = 4.0,
        norm_z_discount:      float = 2.0,
        cosine_quarantine:    float = -0.1,
        cosine_discount:      float = 0.3,
        shap_quarantine:      float = 8.0,
        shap_discount:        float = 4.0,
        attestation_age_max:  float = 60.0,
        probation_weight:     float = 0.5,
        new_weight:           float = 0.3,
    ) -> None:
        self.norm_z_quarantine   = norm_z_quarantine
        self.norm_z_discount     = norm_z_discount
        self.cosine_quarantine   = cosine_quarantine
        self.cosine_discount     = cosine_discount
        self.shap_quarantine     = shap_quarantine
        self.shap_discount       = shap_discount
        self.attestation_age_max = attestation_age_max
        self.probation_weight    = probation_weight
        self.new_weight          = new_weight

    def decide(self, signals: ClientSignals, history: TrustHistory) -> Decision:
        # 1. Hard rejects
        if history.state == TrustState.BLOCKED:
            return Decision(Action.BLOCK, 0.0, 1.0,
                            "client previously blocked")

        if not signals.attestation_valid:
            return Decision(Action.QUARANTINE, 0.0, 1.0,
                            "attestation token invalid")

        if signals.attestation_age_s > self.attestation_age_max:
            return Decision(Action.QUARANTINE, 0.0, 0.95,
                            f"attestation token age "
                            f"{signals.attestation_age_s:.1f}s exceeds limit")

        # 2. Trajectory-driven escalation (uses history summary)
        h = signals.history_summary
        flags = []
        if h.get("max_norm_z", 0.0) > self.norm_z_quarantine:
            flags.append(f"max(norm_z)={h['max_norm_z']:.2f}")
        if h.get("min_cosine_global", 1.0) < self.cosine_quarantine:
            flags.append(f"min(cos_global)={h['min_cosine_global']:.2f}")
        if h.get("max_shap", 0.0) > self.shap_quarantine:
            flags.append(f"max(shap)={h['max_shap']:.2f}")
        if len(flags) >= 2 and history.consecutive_flagged >= 1:
            return Decision(
                Action.QUARANTINE, 0.0, 0.85,
                "trajectory anomaly: " + ", ".join(flags),
            )

        # 3. Single-round anomaly → discount
        soft_flags = []
        if abs(signals.update_norm_zscore) > self.norm_z_discount:
            soft_flags.append(
                f"|norm_z|={signals.update_norm_zscore:.2f}>"
                f"{self.norm_z_discount:.1f}"
            )
        if signals.cosine_to_global < self.cosine_discount:
            soft_flags.append(
                f"cos_global={signals.cosine_to_global:.2f}<"
                f"{self.cosine_discount:.2f}"
            )
        if signals.shap_stability > self.shap_discount:
            soft_flags.append(
                f"shap={signals.shap_stability:.2f}>{self.shap_discount:.1f}"
            )
        if soft_flags:
            # Discount weight scales with severity (1 flag → 0.6, 2 → 0.4, 3 → 0.2)
            w = max(0.2, 1.0 - 0.2 - 0.2 * len(soft_flags))
            return Decision(
                Action.DISCOUNT, w, 0.7,
                "soft anomaly: " + ", ".join(soft_flags),
            )

        # 4. Accept with state-dependent weight
        if history.state == TrustState.NEW:
            return Decision(Action.ACCEPT, self.new_weight, 0.6,
                            "new client; reduced weight while gathering signal")
        if history.state == TrustState.PROBATION:
            return Decision(Action.ACCEPT, self.probation_weight, 0.7,
                            "probation; partial weight pending promotion")
        if history.state == TrustState.QUARANTINE:
            # Cooling off — accept update for observation, no aggregation weight
            return Decision(Action.ACCEPT, 0.0, 0.6,
                            "quarantine cool-off; observation only")
        return Decision(Action.ACCEPT, 1.0, 0.9,
                        "trusted client; all signals nominal")


# ---------------------------------------------------------------------------
# 2. LearnedPolicy — small MLP, falls back to threshold when untrained
# ---------------------------------------------------------------------------

class _PolicyHead(nn.Module):
    """Tiny MLP that maps the signal vector to action logits."""

    def __init__(self, n_in: int = 12, n_actions: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _signals_to_vector(signals: ClientSignals, history: TrustHistory) -> torch.Tensor:
    """Pack a :class:`ClientSignals` + :class:`TrustHistory` into a fixed-shape tensor."""
    h = signals.history_summary
    return torch.tensor([
        float(signals.attestation_valid),
        signals.attestation_age_s,
        signals.update_norm,
        signals.update_norm_zscore,
        signals.cosine_to_global,
        signals.cosine_to_peer_median,
        signals.shap_stability,
        signals.loss_decrease,
        signals.participation_rate,
        h.get("mean_norm_z", 0.0),
        h.get("min_cosine_global", 1.0),
        h.get("max_shap", 0.0),
    ], dtype=torch.float32)


class LearnedPolicy(Policy):
    """
    Small MLP policy.  Until the head has been fit with :meth:`fit` it
    delegates every call to a :class:`ThresholdPolicy`, so the runtime
    behaviour is well-defined out of the box.

    The intent is that an operator collects (signals, decision) traces from
    threshold-policy runs (and from red-team exercises with known Byzantine
    clients) and trains the head on those.  The policy then generalises
    beyond hand-tuned thresholds.
    """

    name = "learned"

    def __init__(self, fallback: Optional[Policy] = None,
                 device: str = "cpu") -> None:
        self.head = _PolicyHead()
        self.head.eval()
        self.device = device
        self.fallback = fallback or ThresholdPolicy()
        self._trained = False

    def fit(self, X: torch.Tensor, y: torch.Tensor,
            n_epochs: int = 50, lr: float = 1e-3) -> float:
        """
        Supervised training on labelled (signal vector, action) pairs.

        Parameters
        ----------
        X : torch.Tensor, shape ``(N, 12)``
        y : torch.Tensor, shape ``(N,)`` — action indices in {0,1,2,3}
            (ACCEPT, DISCOUNT, QUARANTINE, BLOCK)
        """
        opt = torch.optim.Adam(self.head.parameters(), lr=lr)
        crit = nn.CrossEntropyLoss()
        self.head.train()
        loss_val = 0.0
        for _ in range(n_epochs):
            opt.zero_grad()
            logits = self.head(X)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            loss_val = float(loss.item())
        self.head.eval()
        self._trained = True
        return loss_val

    def decide(self, signals: ClientSignals, history: TrustHistory) -> Decision:
        if not self._trained:
            d = self.fallback.decide(signals, history)
            d.reason = f"[learned-untrained, fallback] {d.reason}"
            return d

        x = _signals_to_vector(signals, history).to(self.device).unsqueeze(0)
        with torch.no_grad():
            logits = self.head(x).squeeze(0)
            probs  = torch.softmax(logits, dim=-1)
        idx = int(probs.argmax().item())
        action = list(Action)[idx]
        confidence = float(probs[idx].item())

        # Map action → weight (mirrors the threshold policy contract)
        weight = {Action.ACCEPT: 1.0, Action.DISCOUNT: 0.4,
                  Action.QUARANTINE: 0.0, Action.BLOCK: 0.0}[action]

        return Decision(
            action, weight, confidence,
            f"learned policy: probs={probs.tolist()}",
        )


# ---------------------------------------------------------------------------
# 3. LLMPolicy — interface stub for LLM-backed reasoning
# ---------------------------------------------------------------------------

class LLMPolicy(Policy):
    """
    Pluggable LLM-backed policy.  The :class:`Policy` itself contains no
    LLM client; instead the user supplies a ``decide_fn`` callable that
    takes a signal-summary string and returns a JSON dict
    ``{"action": ..., "weight": ..., "confidence": ..., "reason": ...}``.

    This keeps the package free of any vendor SDK while allowing
    integrations against any LLM (Anthropic, OpenAI, vLLM, etc.).  Until
    the callable is wired in, the policy delegates to a fallback so the
    pipeline works out of the box.
    """

    name = "llm"

    def __init__(
        self,
        decide_fn: Optional[Callable[[str], Dict[str, Any]]] = None,
        fallback: Optional[Policy] = None,
    ) -> None:
        self.decide_fn = decide_fn
        self.fallback = fallback or ThresholdPolicy()

    @staticmethod
    def _format_prompt(signals: ClientSignals, history: TrustHistory) -> str:
        h = signals.history_summary
        return (
            f"Client {signals.client_id} round {signals.round}.\n"
            f"State: {history.state.value}, "
            f"quarantine_count={history.quarantine_count}, "
            f"consecutive_flagged={history.consecutive_flagged}.\n"
            f"This round: norm_z={signals.update_norm_zscore:.2f}, "
            f"cos_global={signals.cosine_to_global:.2f}, "
            f"shap={signals.shap_stability:.2f}, "
            f"attestation_valid={signals.attestation_valid}.\n"
            f"History: max_norm_z={h.get('max_norm_z', 0):.2f}, "
            f"min_cos_global={h.get('min_cosine_global', 1):.2f}, "
            f"max_shap={h.get('max_shap', 0):.2f}.\n"
            "Decide: accept | discount | quarantine | block. "
            "Return JSON {action, weight, confidence, reason}."
        )

    def decide(self, signals: ClientSignals, history: TrustHistory) -> Decision:
        if self.decide_fn is None:
            d = self.fallback.decide(signals, history)
            d.reason = f"[llm-unwired, fallback] {d.reason}"
            return d

        prompt = self._format_prompt(signals, history)
        try:
            resp = self.decide_fn(prompt)
            return Decision(
                action=Action(resp["action"]),
                weight=float(resp["weight"]),
                confidence=float(resp.get("confidence", 0.5)),
                reason=f"[llm] {resp.get('reason', '')}",
            )
        except Exception as e:
            d = self.fallback.decide(signals, history)
            d.reason = f"[llm-error: {e}, fallback] {d.reason}"
            return d
