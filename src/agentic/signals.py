"""
Decision signals consumed by the agentic policies.

A :class:`ClientSignals` record is computed once per (client, round) and
captures every observable that the policy is allowed to look at when
deciding what to do with that client's update.  The signals are deliberately
*passive* observations of the FL state; they contain no side effects and no
policy logic — those live in :mod:`src.agentic.policies`.

The signal extractor is the bridge between the federated-learning core
(which knows about model deltas, attestation tokens, SHAP scores) and the
agentic layer (which only sees the numbers).  Keeping it isolated makes it
easy to swap the underlying detection primitives without touching the
policy code.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn


@dataclass
class ClientSignals:
    """
    All observable evidence about one client at one round.

    Attributes
    ----------
    client_id : str
        Stable identifier across rounds.
    round : int
        Current round number, 1-based.
    attestation_valid : bool
        TPM verification result.
    attestation_age_s : float
        Seconds between the attestation timestamp and the fog node clock.
        Large values flag stale/replayed tokens.
    update_norm : float
        L2 norm of the client's parameter delta (this round).
    update_norm_zscore : float
        Z-score of ``update_norm`` against the per-round peer median.
        Large positive values flag norm-inflation attacks (gradient
        manipulation).
    cosine_to_global : float
        Cosine similarity of the delta to the global update direction
        (FedAvg of all clients).  Low or negative values flag direction
        attacks.
    cosine_to_peer_median : float
        Cosine similarity of the delta to the per-round peer median direction.
    shap_stability : float
        Integrated-gradient SHAP distance to the previous global model.
        Higher = explanations diverged more.  Computed by the FL core.
    loss_decrease : float
        Local training loss reduction during this client's round.  Negative
        values flag clients that did not actually train (or trained on a
        manipulated objective).
    participation_rate : float
        Fraction of rounds this client has participated in over the trailing
        window.  Used to penalise sporadic Sybil-style participation.
    history_summary : dict
        Per-client running summary (mean/min/max of past signals) supplied by
        the trust ledger.  Lets the policy reason over a trajectory rather
        than only the current round.
    """

    client_id: str
    round: int

    # Attestation
    attestation_valid: bool = True
    attestation_age_s: float = 0.0

    # Update geometry
    update_norm: float = 0.0
    update_norm_zscore: float = 0.0
    cosine_to_global: float = 1.0
    cosine_to_peer_median: float = 1.0

    # Explanation stability
    shap_stability: float = 0.0

    # Behavioural / training
    loss_decrease: float = 0.0
    participation_rate: float = 1.0

    # Aggregate history (filled in by TrustLedger.enrich)
    history_summary: Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        """Return a plain dict serialisation (for JSON logging / audit)."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Signal extraction
# ---------------------------------------------------------------------------

def _flatten_params(model: nn.Module, device=None) -> torch.Tensor:
    """Return a single flat tensor view of every model parameter."""
    if device is None:
        device = next(model.parameters()).device
    return torch.cat([p.data.view(-1).float().to(device) for p in model.parameters()])


def _delta(local_model: nn.Module, global_model: nn.Module) -> torch.Tensor:
    """Delta evaluated on the local model's device (handles cross-device pairs)."""
    device = next(local_model.parameters()).device
    return _flatten_params(local_model, device=device) \
         - _flatten_params(global_model, device=device)


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    na = a.norm(p=2).clamp(min=1e-12)
    nb = b.norm(p=2).clamp(min=1e-12)
    return float(((a @ b) / (na * nb)).item())


def build_signals(
    *,
    client_id: str,
    round_number: int,
    local_model: nn.Module,
    global_model: nn.Module,
    peer_models: List[nn.Module],
    attestation_valid: bool = True,
    attestation_age_s: float = 0.0,
    shap_stability: float = 0.0,
    loss_decrease: float = 0.0,
    participation_rate: float = 1.0,
) -> ClientSignals:
    """
    Extract a :class:`ClientSignals` record from FL state.

    All cosine and z-score statistics are computed against the peer
    population (every model that participated in the same round).  When
    ``peer_models`` is empty the geometric signals fall back to neutral
    defaults (cosine=1, z=0) so the policy never sees ``NaN``.
    """
    delta = _delta(local_model, global_model)
    update_norm = float(delta.norm(p=2).item())

    if peer_models:
        peer_deltas = [_delta(m, global_model) for m in peer_models]
        peer_norms = torch.tensor([d.norm(p=2).item() for d in peer_deltas])
        med = float(peer_norms.median().item())
        mad = float((peer_norms - med).abs().median().item())
        # Robust z-score using MAD (~1.4826 * MAD ≈ σ for Gaussian)
        z = (update_norm - med) / max(1e-12, 1.4826 * mad)
        cosine_global = _cosine(delta, torch.stack(peer_deltas, dim=0).mean(dim=0))
        cosine_peer_median = _cosine(
            delta,
            torch.stack(peer_deltas, dim=0).median(dim=0).values,
        )
    else:
        z = 0.0
        cosine_global = 1.0
        cosine_peer_median = 1.0

    return ClientSignals(
        client_id=client_id,
        round=round_number,
        attestation_valid=attestation_valid,
        attestation_age_s=attestation_age_s,
        update_norm=update_norm,
        update_norm_zscore=float(z),
        cosine_to_global=cosine_global,
        cosine_to_peer_median=cosine_peer_median,
        shap_stability=float(shap_stability),
        loss_decrease=float(loss_decrease),
        participation_rate=float(participation_rate),
    )
