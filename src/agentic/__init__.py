"""
Agentic decision layer for ZTA-FL.

This package implements the agent architecture described in
Singh, Roy & So (2025), *Zero-Trust Agentic Federated Learning for Secure
IIoT Defense Systems* (arXiv:2512.23809), specifically:

* **Section IV** — three-tier hierarchical structure with the 5-module Edge
  Agent (perception, local IDS, adversarial training, TPM attestation,
  secure communication).
* **Section V.A** — TrustDB policy (τ_i ∈ [0,1] state machine with the
  paper's exact +0.02 / ×0.5 / 5-attestation-cool-off rules).
* **Section V.B** — Fog-side SHAP-Weighted Robust Aggregation
  (attestation verify → SHAP filter at μ_s − 2σ_s → weighted FedAvg
  → rollback if accuracy collapses to ≤ 0.8 × prior round).

The public API mirrors the paper's structure 1-1:

    from src.agentic import (
        EdgeAgent,            # Section IV: 5-module IIoT device
        FogAgent,             # Section IV/V: aggregator + decisions
        TrustDB, AgentStatus, # Section V.A: trust policy state
        AgenticConfig,        # all paper constants in one file
    )

Two further interfaces are provided as **extensions beyond the paper**.
They are not part of the published claims; they are pluggable hooks for
research follow-up:

    from src.agentic.policies import (
        LearnedPolicy,        # tiny MLP fit on historical trace data
        LLMPolicy,            # callable interface for LLM-backed reasoning
    )

These extensions are not used by the default :class:`FogAgent`; they exist
to let downstream researchers swap in alternative decision rules without
modifying the rest of the pipeline.  See the package docstring of
:mod:`src.agentic.policies` for the contract.
"""

# --- Paper-faithful core ----------------------------------------------------

from src.agentic.config        import (
    AgenticConfig,
    TrustDBConfig,
    AttestationConfig,
    ShapAggregationConfig,
    AdvTrainingConfig,
    FederationConfig,
    ObservabilityConfig,
)
from src.agentic.trust_db      import TrustDB, TrustRecord, AgentStatus
from src.agentic.edge_agent    import (
    EdgeAgent,
    PerceptionModule,
    AttestationModule,
    AttestationToken,
    SecureChannel,
)
from src.agentic.fog_agent     import FogAgent, RoundSummary
from src.agentic.observability import (
    configure_logging,
    MetricsSink,
    NullMetricsSink,
)

# --- Optional research extensions (not part of paper claims) ----------------
# Imported lazily-friendly: still surfaced at the package root for
# discoverability, but documentation marks them as extensions.
from src.agentic.signals       import ClientSignals, build_signals
from src.agentic.policies      import (
    Action,
    Decision,
    Policy,
    ThresholdPolicy,
    LearnedPolicy,
    LLMPolicy,
)

__all__ = [
    # Paper-faithful
    "AgenticConfig",
    "TrustDBConfig",
    "AttestationConfig",
    "ShapAggregationConfig",
    "AdvTrainingConfig",
    "FederationConfig",
    "ObservabilityConfig",
    "TrustDB", "TrustRecord", "AgentStatus",
    "EdgeAgent", "PerceptionModule", "AttestationModule",
    "AttestationToken", "SecureChannel",
    "FogAgent", "RoundSummary",
    "configure_logging", "MetricsSink", "NullMetricsSink",
    # Research extensions
    "ClientSignals", "build_signals",
    "Action", "Decision",
    "Policy", "ThresholdPolicy", "LearnedPolicy", "LLMPolicy",
]
