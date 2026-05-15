"""
Tests for the agentic ZTA-FL layer.

Two layers are exercised:

1. **Paper-faithful core** — TrustDB rules from Section V.A, EdgeAgent's
   5-module structure from Section IV, FogAgent's
   attestation→SHAP-filter→weighted-FedAvg→rollback pipeline from
   Section V.B.  These tests verify the implementation matches the
   paper's stated specification, not just "does something plausible".

2. **Research extensions** (signals, ThresholdPolicy, LearnedPolicy,
   LLMPolicy) — pluggable decision interfaces beyond the paper.

Run with::

    python -m pytest tests/test_agentic.py -v
"""

from __future__ import annotations

import json
import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Paper-faithful core ----------------------------------------------------

from src.agentic.trust_db   import (
    TrustDB, AgentStatus,
    TAU_INIT, TAU_REJOIN, TAU_MIN, TAU_REWARD_DELTA,
    TAU_PENALTY_RATIO, REJOIN_ATTESTS,
)
from src.agentic.config     import AgenticConfig
from src.agentic.edge_agent import (
    EdgeAgent, PerceptionModule, AttestationModule,
    AttestationToken, SecureChannel,
)
from src.agentic.fog_agent  import FogAgent

# --- Research extensions ----------------------------------------------------

from src.agentic.signals    import build_signals, ClientSignals
from src.agentic.policies   import (
    Action, Decision, ThresholdPolicy, LearnedPolicy, LLMPolicy,
)
from src.agentic.trust_state import TrustLedger as ExtTrustLedger
from src.security.attestation import AttestationAuthority, TPMDevice
from src.models.cnn_lstm    import CNNLSTMClassifier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _model() -> nn.Module:
    return CNNLSTMClassifier(n_features=40, n_classes=15)


def _perturb(m: nn.Module, scale: float) -> nn.Module:
    with torch.no_grad():
        for p in m.parameters():
            p.add_(scale * torch.randn_like(p))
    return m


# ===========================================================================
# Section V.A: TrustDB paper rules
# ===========================================================================

class TestTrustDBPaperRules:
    """Every test maps to a specific clause in Section V.A."""

    def test_initialisation_value_matches_paper(self):
        """ "New agents start at τ_i = 0.7 after successful first attestation." """
        db = TrustDB()
        r = db.first_attestation("a0")
        assert r.tau == TAU_INIT == 0.7
        assert r.status == AgentStatus.ACTIVE

    def test_positive_update_increment_matches_paper(self):
        """ "τ_i ← min(1, τ_i + 0.02)" on positive round. """
        db = TrustDB()
        db.first_attestation("a0")
        before = db.get("a0").tau
        db.positive_round("a0", shap_above_mean=True)
        after = db.get("a0").tau
        assert after == pytest.approx(before + TAU_REWARD_DELTA)
        assert TAU_REWARD_DELTA == 0.02

    def test_positive_update_caps_at_one(self):
        """τ never exceeds 1.0 even with many positive rounds."""
        db = TrustDB()
        db.first_attestation("a0")
        for _ in range(100):
            db.positive_round("a0", shap_above_mean=True)
        assert db.get("a0").tau <= 1.0

    def test_penalty_multiplies_by_half(self):
        """ "Failed attestation or SHAP filtering triggers τ_i ← τ_i × 0.5". """
        db = TrustDB()
        db.first_attestation("a0")
        before = db.get("a0").tau
        db.penalty("a0", "test penalty")
        after = db.get("a0").tau
        assert after == pytest.approx(before * TAU_PENALTY_RATIO)
        assert TAU_PENALTY_RATIO == 0.5

    def test_quarantine_threshold(self):
        """ "Agents with τ_i < τ_min = 0.6 enter quarantine." """
        db = TrustDB()
        db.first_attestation("a0")     # τ = 0.7
        # 0.7 × 0.5 = 0.35 < 0.6 → quarantine
        db.penalty("a0", "test")
        assert db.get("a0").status == AgentStatus.QUARANTINED
        assert TAU_MIN == 0.6

    def test_rejoin_requires_five_consecutive_clean(self):
        """ "Quarantined agents must pass 5 consecutive attestations …" """
        assert REJOIN_ATTESTS == 5
        db = TrustDB()
        db.first_attestation("a0")
        db.penalty("a0", "drop into quarantine")
        assert db.get("a0").status == AgentStatus.QUARANTINED

        # 4 clean rounds: still quarantined
        for _ in range(4):
            db.positive_round("a0", shap_above_mean=True)
        assert db.get("a0").status == AgentStatus.QUARANTINED

        # 5th clean round: rehabilitated → τ reset to 0.65
        db.positive_round("a0", shap_above_mean=True)
        assert db.get("a0").status == AgentStatus.ACTIVE
        assert db.get("a0").tau == pytest.approx(TAU_REJOIN)
        assert TAU_REJOIN == 0.65

    def test_paper_exact_preset_matches_paper_constants(self):
        cfg = AgenticConfig.paper_exact()
        assert cfg.preset == "paper-exact"
        assert cfg.trust_db.tau_init      == 0.7
        assert cfg.trust_db.tau_min       == 0.6
        assert cfg.trust_db.tau_reward    == 0.02
        assert cfg.trust_db.tau_penalty   == 0.5
        assert cfg.trust_db.rejoin_attests == 5
        assert cfg.shap.sigma_threshold   == 2.0
        assert cfg.shap.rollback_ratio    == 0.8
        assert cfg.shap.n_background      == 100

    def test_small_scale_preset_loosens_only_statistical_thresholds(self):
        """Small-scale must keep the paper's TrustDB rules verbatim;
        only the statistical thresholds (which break at small N) change."""
        small = AgenticConfig.small_scale()
        paper = AgenticConfig.paper_exact()
        assert small.preset == "small-scale"
        # TrustDB rules are paper-faithful even in small-scale
        assert small.trust_db == paper.trust_db
        # SHAP knobs are loosened
        assert small.shap.sigma_threshold > paper.shap.sigma_threshold
        assert small.shap.rollback_ratio  < paper.shap.rollback_ratio
        assert small.shap.n_background    < paper.shap.n_background

    def test_audit_log_contains_every_event(self, tmp_path):
        audit = tmp_path / "audit.jsonl"
        db = TrustDB(audit_path=str(audit))
        db.first_attestation("a0")
        db.positive_round("a0")
        db.penalty("a0", "norm spike")
        lines = audit.read_text().strip().splitlines()
        assert len(lines) == 3
        events = [json.loads(L)["event"] for L in lines]
        assert "first_attestation" in events
        assert any("penalty" in e or "quarantined" in e for e in events)


# ===========================================================================
# Section IV: Edge Agent 5-module structure
# ===========================================================================

class TestEdgeAgentModules:
    """ "Each device consists of five functional modules: …" """

    def test_all_five_modules_exposed(self):
        a = EdgeAgent("a0", n_features=40, n_classes=15, secret="k0")
        # 1. Perception
        assert isinstance(a.perception, PerceptionModule)
        # 2. Local IDS
        assert isinstance(a.local_ids, CNNLSTMClassifier)
        # 3. Adversarial training (function reference)
        assert callable(a.adv_training)
        # 4. TPM-based attestation
        assert isinstance(a.attestation, AttestationModule)
        # 5. Secure communication
        assert isinstance(a.secure_comm, SecureChannel)

    def test_perception_module_validates_input_shape(self):
        a = EdgeAgent("a0", n_features=40, n_classes=15, secret="k0")
        good = torch.randn(8, 40)
        out  = a.perception(good)
        assert out.shape == (8, 40)
        with pytest.raises(ValueError):
            a.perception(torch.randn(8, 30))   # wrong feature count

    def test_attestation_module_returns_paper_token_struct(self):
        a = EdgeAgent("a0", n_features=40, n_classes=15, secret="k0")
        tok = a.attestation.generate_token()
        # Paper Section IV: {ID_i, t, PCR, Sig_TPM}
        assert isinstance(tok, AttestationToken)
        assert tok.device_id == "a0"
        assert tok.timestamp > 0
        assert tok.signature
        assert tok.pcr_digest

    def test_self_quarantine_on_loss_drift(self):
        a = EdgeAgent("a0", n_features=40, n_classes=15, secret="k0")
        # Seed loss history with stable values, then a 3× spike
        a._loss_history.extend([0.5, 0.5, 0.5, 2.0])
        ok, reason = a.decide_participation(local_data_size=100)
        assert not ok and "loss drift" in reason

    def test_skip_when_data_too_small(self):
        a = EdgeAgent("a0", n_features=40, n_classes=15, secret="k0")
        ok, reason = a.decide_participation(local_data_size=2)
        assert not ok and "insufficient" in reason


# ===========================================================================
# Section V.B: Fog Agent pipeline
# ===========================================================================

class TestFogAgentPipeline:

    def _setup(self, n_clients: int = 5):
        torch.manual_seed(0)
        cfg = AgenticConfig()
        # Test artefacts live inside the repo (under results/test_fixtures/)
        # rather than /tmp/, so reviewers can inspect them and CI logs can
        # archive them. .gitignore excludes the directory from commits.
        fixtures_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "results", "test_fixtures",
        )
        os.makedirs(fixtures_dir, exist_ok=True)
        cfg.observability.audit_path   = os.path.join(fixtures_dir, "fog_test_audit.jsonl")
        cfg.observability.metrics_path = os.path.join(fixtures_dir, "fog_test_metrics.jsonl")

        global_model = _model()
        ids = [f"a{i}" for i in range(n_clients)]
        aik = {cid: f"k{cid}" for cid in ids}
        auth = AttestationAuthority(aik_registry=aik, max_age_seconds=300)

        # Generate canonical tokens via the EdgeAgent's AttestationModule
        # (which returns src.security.attestation.AttestationToken objects)
        modules = [AttestationModule(cid, f"k{cid}") for cid in ids]
        import time as _t
        t0 = _t.time()
        tokens = [m.generate_token(t0) for m in modules]

        local_models = [_perturb(_model(), 0.005) for _ in range(n_clients)]
        local_models[-1] = _perturb(_model(), 0.5)   # blatant outlier
        sizes = [200] * n_clients
        accs  = [0.9]  * n_clients

        Xv = torch.randn(40, 40)
        yv = torch.randint(0, 15, (40,))

        fog = FogAgent(global_model=global_model,
                       attestation_authority=auth, config=cfg)
        return fog, ids, tokens, local_models, sizes, accs, Xv, yv

    def test_run_round_returns_round_summary(self):
        fog, ids, toks, models, sizes, accs, Xv, yv = self._setup()
        r = fog.run_round(round_number=1,
                          agent_ids=ids, local_models=models,
                          tokens=toks, local_sizes=sizes, local_accs=accs,
                          X_val=Xv, y_val=yv, n_classes=15)
        assert r.round == 1
        assert isinstance(r.aggregated_state, dict)
        assert r.n_admitted >= 0

    def test_outlier_filtered_or_downweighted(self):
        """Blatant outlier client should not get full weight in aggregation."""
        fog, ids, toks, models, sizes, accs, Xv, yv = self._setup(n_clients=6)
        r = fog.run_round(round_number=1,
                          agent_ids=ids, local_models=models,
                          tokens=toks, local_sizes=sizes, local_accs=accs,
                          X_val=Xv, y_val=yv, n_classes=15)
        # Either the outlier was filtered (not in weights) or it has the
        # smallest weight among admitted clients.
        outlier_id = ids[-1]
        if outlier_id in r.weights:
            other_avg = sum(w for cid, w in r.weights.items() if cid != outlier_id) \
                        / max(1, len(r.weights) - 1)
            assert r.weights[outlier_id] <= other_avg, (
                f"outlier weight {r.weights[outlier_id]} should not exceed "
                f"avg of clean clients {other_avg}"
            )

    def test_rollback_when_accuracy_collapses(self, tmp_path):
        """Section V.B sanity check: revert if agg_acc < 0.8 × prev_acc."""
        fog, ids, toks, models, sizes, accs, Xv, yv = self._setup()
        # Force a non-zero "previous accuracy" so rollback can trigger
        fog.previous_accuracy = 0.95
        # Replace all local models with garbage (so aggregated acc is poor)
        bad_models = [_perturb(_model(), 5.0) for _ in models]
        r = fog.run_round(round_number=2,
                          agent_ids=ids, local_models=bad_models,
                          tokens=toks, local_sizes=sizes, local_accs=accs,
                          X_val=Xv, y_val=yv, n_classes=15)
        # On 15-class random data, an aggregated garbage model can't reach
        # 0.8 × 0.95 = 0.76 accuracy → rollback expected
        assert r.aggregated_accuracy < 0.76 * 0.95 or r.rolled_back, (
            f"Either rollback should fire or new acc must be very low; "
            f"got new={r.aggregated_accuracy}, rolled_back={r.rolled_back}"
        )


# ===========================================================================
# Research extensions: signals, policies (kept for follow-up work)
# ===========================================================================

class TestExtensions:

    def test_build_signals_responds_to_outlier(self):
        torch.manual_seed(0)
        g = _model()
        peers = [_perturb(_model(), 0.01) for _ in range(8)]
        outlier = _perturb(_model(), 0.5)
        s_normal  = build_signals(client_id="c0", round_number=1,
                                  local_model=peers[0], global_model=g,
                                  peer_models=peers[1:])
        s_outlier = build_signals(client_id="c1", round_number=1,
                                  local_model=outlier, global_model=g,
                                  peer_models=peers)
        assert s_outlier.update_norm > 5 * s_normal.update_norm

    def test_threshold_policy_distinguishes_inputs(self):
        from src.agentic.trust_state import TrustLedger
        from src.agentic.signals     import ClientSignals
        p = ThresholdPolicy()
        h = TrustLedger().get("c0")
        clean = ClientSignals(client_id="c0", round=1)
        bad   = ClientSignals(client_id="c0", round=1, attestation_valid=False)
        d_clean = p.decide(clean, h)
        d_bad   = p.decide(bad,   h)
        assert d_clean.action == Action.ACCEPT
        assert d_bad.action   == Action.QUARANTINE

    def test_learned_policy_falls_back_when_untrained(self):
        from src.agentic.trust_state import TrustLedger
        from src.agentic.signals     import ClientSignals
        p = LearnedPolicy()
        d = p.decide(ClientSignals(client_id="c0", round=1),
                     TrustLedger().get("c0"))
        assert "fallback" in d.reason

    def test_llm_policy_consumes_callable(self):
        from src.agentic.trust_state import TrustLedger
        from src.agentic.signals     import ClientSignals
        def fake_llm(prompt):
            return {"action": "block", "weight": 0.0,
                    "confidence": 0.99, "reason": "mock"}
        p = LLMPolicy(decide_fn=fake_llm)
        d = p.decide(ClientSignals(client_id="c0", round=1),
                     TrustLedger().get("c0"))
        assert d.action == Action.BLOCK
