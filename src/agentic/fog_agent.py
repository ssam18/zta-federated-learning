"""
Fog Agent — paper-faithful implementation of ZTA-FL Section IV (Fog Layer)
and Section V.B (SHAP-Weighted Robust Aggregation).

Verbatim from the paper:

    Fog Layer: This layer will collect attestation tokens from all K Edge
    Agents and verify each token based upon its signature, freshness and
    PCR value.  After verifying each token, the Fog Layer will perform
    SHAP-Weighted Robust Aggregation of the verified attestation tokens
    using SHAP Stability Scores to identify and remove Byzantine Updates
    prior to forwarding the aggregated data to the Cloud Layer.

And Section V.B:

    The fog node calculates the SHAP stability scores
        s_i = 1 − ||φ_i − φ_ref||_2 / (||φ_ref||_2 + ε)
    for all agents in every federated learning round, where φ_i is a
    vector of feature importances. An agent with a score less than
    μ_s − 2σ_s will be identified as a potential byzantine actor,
    therefore eliminating its data from the aggregation.  In contrast,
    valid updates are weighted using
        w_i ∝ s_i · acc_i · √|D_i|
    which considers multiple factors: SHAP stability, validation
    accuracy, and dataset size.  Finally, we include a sanity check to
    revert back to the last round's global model if the aggregated
    accuracy drops to 80 percent or lower than it was in the previous
    round.

This module implements every clause above, with the constants pulled
from :class:`~src.agentic.config.AgenticConfig` so the deployment
parameters live in one file.
"""

from __future__ import annotations

import copy
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from src.federation.aggregation       import federated_averaging
from src.security.attestation         import AttestationAuthority
from src.utils.metrics                import accuracy as _acc, compute_shap_stability
from src.agentic.config               import AgenticConfig
from src.agentic.edge_agent           import AttestationToken
from src.agentic.observability        import MetricsSink, NullMetricsSink
from src.agentic.trust_db             import TrustDB, AgentStatus


logger = logging.getLogger(__name__)


@dataclass
class RoundSummary:
    """Bundle returned by :meth:`FogAgent.run_round`."""
    round:               int
    aggregated_state:    Dict[str, torch.Tensor]
    rolled_back:         bool
    n_admitted:          int
    n_filtered_attest:   int
    n_filtered_shap:     int
    weights:             Dict[str, float]   = field(default_factory=dict)
    stability_scores:    Dict[str, float]   = field(default_factory=dict)
    aggregated_accuracy: float              = 0.0
    prev_accuracy:       float              = 0.0


# ---------------------------------------------------------------------------
# Fog Agent
# ---------------------------------------------------------------------------

class FogAgent:
    """
    Paper-faithful fog aggregator.

    For every round the fog agent runs the four-step pipeline below.  Every
    step references the paper section it implements so a reviewer can
    verify the mapping at a glance:

      1. **Attestation verification** (Section V.A)
         For each submitted update, verify token via the
         :class:`~src.security.attestation.AttestationAuthority`.  Failed
         verification → ``TrustDB.penalty(...)`` → not admitted to
         aggregation.

      2. **SHAP stability score** (Section V.B)
         For each surviving update, compute
         :math:`s_i = 1 - \\|\\phi_i - \\phi_{\\text{ref}}\\|_2 /
         (\\|\\phi_{\\text{ref}}\\|_2 + \\varepsilon)`
         using GradientSHAP-equivalent integrated gradients.

      3. **Statistical filter** (Section V.B)
         Drop agents with :math:`s_i < \\mu_s - 2\\sigma_s`.  Penalise
         those agents in the TrustDB.

      4. **Weighted aggregation + rollback** (Section V.B)
         :math:`w_i \\propto s_i \\cdot \\text{acc}_i \\cdot \\sqrt{|D_i|}`,
         then standard FedAvg.  If the aggregated model's validation
         accuracy drops below ``0.8 ×`` the previous-round accuracy,
         revert to the previous global model (paper's "sanity check").
    """

    def __init__(
        self,
        global_model:        nn.Module,
        attestation_authority: AttestationAuthority,
        config:              Optional[AgenticConfig] = None,
        trust_db:            Optional[TrustDB] = None,
        metrics:             Optional[MetricsSink] = None,
        device:              str = "cpu",
    ) -> None:
        self.config           = config or AgenticConfig()
        self.global_model     = global_model.to(device)
        self.previous_state   = {k: v.detach().clone()
                                 for k, v in global_model.state_dict().items()}
        self.previous_accuracy = 0.0
        self.attestation_auth = attestation_authority
        self.trust_db         = trust_db or TrustDB(
            tau_init       = self.config.trust_db.tau_init,
            tau_rejoin     = self.config.trust_db.tau_rejoin,
            tau_min        = self.config.trust_db.tau_min,
            tau_reward     = self.config.trust_db.tau_reward,
            tau_penalty    = self.config.trust_db.tau_penalty,
            rejoin_attests = self.config.trust_db.rejoin_attests,
            audit_path     = self.config.observability.audit_path,
        )
        self.metrics = metrics or NullMetricsSink()
        self.device  = device

    # ------------------------------------------------------------------
    # Step 1: Attestation verification (Section V.A)
    # ------------------------------------------------------------------

    def _verify_attestations(
        self,
        tokens:    List[AttestationToken],
        agent_ids: List[str],
        current_time: float,
    ) -> List[bool]:
        """Return a boolean mask of admitted clients."""
        admitted: List[bool] = []
        for tok, cid in zip(tokens, agent_ids):
            # Token freshness check (paper: ∆t_max = 60s)
            age = current_time - tok.timestamp
            if age > self.config.attestation.delta_t_max_s:
                self.trust_db.penalty(cid, f"token age {age:.2f}s > Δt_max")
                admitted.append(False)
                continue
            # Cryptographic verification — pass the full canonical
            # AttestationToken to the authority (signature, PCR digest,
            # nonce, timestamp are all on the dataclass).
            ok, reason = self.attestation_auth.verify(
                tok, current_time=current_time,
            )
            if not ok:
                self.trust_db.penalty(cid, f"attestation failed: {reason}")
                admitted.append(False)
                continue
            # First sighting → admit per Section V.A
            if cid not in self.trust_db:
                self.trust_db.first_attestation(cid)
            admitted.append(True)
        return admitted

    # ------------------------------------------------------------------
    # Step 2: SHAP stability score (Section V.B)
    # ------------------------------------------------------------------

    def _stability_scores(
        self,
        local_models:    List[nn.Module],
        ref_model:       nn.Module,
        X_val:           torch.Tensor,
        y_val:           torch.Tensor,
        n_classes:       int,
    ) -> List[float]:
        """
        Compute s_i = 1 − ||φ_i − φ_ref||₂ / (||φ_ref||₂ + ε) for each model.

        The integrated-gradient approximation in
        :func:`compute_shap_stability` returns ``||φ_i − φ_ref||₂`` already;
        this function normalises it into the paper's ``s_i`` form.
        """
        if not local_models:
            return []

        eps = self.config.shap.epsilon

        # Compute reference attribution norm (||φ_ref||₂) via a self-comparison
        # of ref_model vs ref_model is degenerate (==0); the paper interprets
        # ||φ_ref||₂ as the magnitude of the global model's attributions.  We
        # estimate it as the median of the per-client distance vector before
        # normalisation, which keeps the score in [0, 1] in practice.
        raw_dists: List[float] = []
        for m in local_models:
            try:
                d = compute_shap_stability(
                    m, ref_model, X_val, y_val,
                    n_explain=min(self.config.shap.n_background, X_val.shape[0]),
                    n_classes=n_classes,
                    device=self.device,
                )
            except Exception as exc:
                logger.warning(f"SHAP stability failed: {exc}")
                d = 0.0
            raw_dists.append(float(d))

        if not raw_dists:
            return []
        ref_norm = float(torch.tensor(raw_dists).median().item())
        if ref_norm < eps:
            ref_norm = max(raw_dists) + eps

        # Stability score: 1 − (distance / (ref_norm + ε)); clamp to [0, 1]
        scores = [max(0.0, min(1.0, 1.0 - d / (ref_norm + eps)))
                  for d in raw_dists]
        return scores

    # ------------------------------------------------------------------
    # Step 3: Statistical filter (Section V.B)
    # ------------------------------------------------------------------

    def _filter_byzantine(
        self,
        scores:    List[float],
        agent_ids: List[str],
    ) -> Tuple[List[bool], float, float]:
        """
        Identify s_i < μ_s − k·σ_s as Byzantine.  Return per-agent admit
        mask, plus the population (μ_s, σ_s).
        """
        if not scores:
            return [], 0.0, 0.0
        t = torch.tensor(scores)
        mu    = float(t.mean().item())
        sigma = float(t.std(correction=0).item()) if len(scores) > 1 else 0.0
        k     = self.config.shap.sigma_threshold
        threshold = mu - k * sigma
        admit_mask: List[bool] = []
        for cid, s in zip(agent_ids, scores):
            if s < threshold:
                self.trust_db.penalty(cid,
                                      f"SHAP filter: s={s:.3f} < μ-{k}σ={threshold:.3f}")
                admit_mask.append(False)
            else:
                self.trust_db.positive_round(cid, shap_above_mean=(s > mu))
                admit_mask.append(True)
        return admit_mask, mu, sigma

    # ------------------------------------------------------------------
    # Step 4: Weighted aggregation + rollback (Section V.B)
    # ------------------------------------------------------------------

    def _weighted_aggregate(
        self,
        local_models: List[nn.Module],
        scores:       List[float],
        accuracies:   List[float],
        sizes:        List[int],
    ) -> Tuple[nn.Module, List[float]]:
        """w_i ∝ s_i · acc_i · √|D_i|  →  normalise and FedAvg."""
        raw = [max(0.0, scores[i]) * max(0.0, accuracies[i]) * math.sqrt(max(1, sizes[i]))
               for i in range(len(local_models))]
        total = sum(raw)
        if total < 1e-12:
            weights = [1.0 / len(local_models)] * len(local_models)
        else:
            weights = [w / total for w in raw]
        agg = federated_averaging(local_models, weights=weights)
        return agg, weights

    def _eval_accuracy(self, model: nn.Module,
                       X: torch.Tensor, y: torch.Tensor) -> float:
        """Quick validation accuracy (used for rollback decision)."""
        model.train()  # cuDNN LSTM constraint
        model.to(self.device)
        with torch.no_grad():
            preds = []
            for i in range(0, X.size(0), 256):
                chunk = X[i:i+256].to(self.device)
                if chunk.size(0) < 2:
                    continue
                preds.append(model(chunk).argmax(dim=-1).cpu())
            if not preds:
                return 0.0
            p = torch.cat(preds)
        return float(_acc(y[:p.size(0)], p))

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run_round(
        self,
        round_number: int,
        agent_ids:    List[str],
        local_models: List[nn.Module],
        tokens:       List[AttestationToken],
        local_sizes:  List[int],
        local_accs:   List[float],
        X_val:        torch.Tensor,
        y_val:        torch.Tensor,
        n_classes:    int,
        current_time: Optional[float] = None,
    ) -> RoundSummary:
        """
        Execute the four-step paper pipeline for one global round.

        Returns a :class:`RoundSummary` with the aggregated state and full
        provenance (per-agent stability score, weight, attestation result,
        and rollback flag).
        """
        import time as _time
        t_now = current_time if current_time is not None else _time.time()
        n = len(agent_ids)
        if n == 0:
            return RoundSummary(round=round_number,
                                aggregated_state=self.previous_state,
                                rolled_back=False, n_admitted=0,
                                n_filtered_attest=0, n_filtered_shap=0)

        with self.metrics.round(round_number):

            # === Step 1: attestation verification ============================
            attest_admit = self._verify_attestations(tokens, agent_ids, t_now)
            n_filtered_attest = sum(1 for a in attest_admit if not a)
            self.metrics.emit("attestation",
                              round=round_number,
                              n_total=n,
                              n_filtered=n_filtered_attest)

            survivors_a = [(cid, m, sz, ac) for cid, m, sz, ac, ok
                           in zip(agent_ids, local_models, local_sizes,
                                  local_accs, attest_admit) if ok]
            if not survivors_a:
                logger.warning("[fog] no clients survived attestation; "
                               "keeping previous global state")
                return RoundSummary(
                    round=round_number,
                    aggregated_state=self.previous_state,
                    rolled_back=True, n_admitted=0,
                    n_filtered_attest=n_filtered_attest, n_filtered_shap=0,
                    prev_accuracy=self.previous_accuracy,
                )

            ids_a, models_a, sizes_a, accs_a = (list(z) for z in zip(*survivors_a))

            # === Step 2: SHAP stability scores ===============================
            scores = self._stability_scores(models_a, self.global_model,
                                            X_val, y_val, n_classes)
            self.metrics.emit("shap_scores",
                              round=round_number,
                              scores=dict(zip(ids_a, scores)))

            # === Step 3: statistical filter μ_s − 2σ_s =======================
            shap_admit, mu_s, sigma_s = self._filter_byzantine(scores, ids_a)
            n_filtered_shap = sum(1 for a in shap_admit if not a)
            self.metrics.emit("shap_filter",
                              round=round_number,
                              mu=mu_s, sigma=sigma_s,
                              k=self.config.shap.sigma_threshold,
                              n_filtered=n_filtered_shap)

            survivors_s = [(cid, m, sz, ac, s)
                           for cid, m, sz, ac, s, ok
                           in zip(ids_a, models_a, sizes_a, accs_a,
                                  scores, shap_admit) if ok]
            if not survivors_s:
                logger.warning("[fog] no clients survived SHAP filter")
                return RoundSummary(
                    round=round_number,
                    aggregated_state=self.previous_state,
                    rolled_back=True, n_admitted=0,
                    n_filtered_attest=n_filtered_attest,
                    n_filtered_shap=n_filtered_shap,
                    prev_accuracy=self.previous_accuracy,
                )

            ids_s, models_s, sizes_s, accs_s, scores_s = (
                list(z) for z in zip(*survivors_s)
            )

            # === Step 4: weighted aggregation + rollback =====================
            agg_model, weights = self._weighted_aggregate(
                models_s, scores_s, accs_s, sizes_s
            )
            new_acc = self._eval_accuracy(agg_model, X_val.cpu(), y_val.cpu())
            rollback_threshold = (self.previous_accuracy
                                  * self.config.shap.rollback_ratio)
            rolled_back = (self.previous_accuracy > 0
                           and new_acc < rollback_threshold)
            if rolled_back:
                logger.warning(
                    f"[fog] rollback triggered: new_acc={new_acc:.3f} < "
                    f"{self.config.shap.rollback_ratio} × prev_acc="
                    f"{self.previous_accuracy:.3f} = {rollback_threshold:.3f}"
                )
                self.metrics.emit("rollback",
                                  round=round_number,
                                  prev_acc=self.previous_accuracy,
                                  new_acc=new_acc,
                                  threshold=rollback_threshold)
                aggregated_state = self.previous_state
            else:
                aggregated_state = {k: v.detach().clone()
                                    for k, v in agg_model.state_dict().items()}
                self.previous_state    = aggregated_state
                self.previous_accuracy = new_acc
                self.global_model.load_state_dict(aggregated_state)

            self.metrics.emit("aggregation",
                              round=round_number,
                              n_aggregated=len(survivors_s),
                              weights=dict(zip(ids_s, weights)),
                              accuracy=new_acc,
                              rolled_back=rolled_back)

        return RoundSummary(
            round              = round_number,
            aggregated_state   = aggregated_state,
            rolled_back        = rolled_back,
            n_admitted         = len(survivors_s),
            n_filtered_attest  = n_filtered_attest,
            n_filtered_shap    = n_filtered_shap,
            weights            = dict(zip(ids_s, weights)),
            stability_scores   = dict(zip(ids_s, scores_s)),
            aggregated_accuracy = new_acc,
            prev_accuracy      = self.previous_accuracy,
        )
