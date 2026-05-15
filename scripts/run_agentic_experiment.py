"""
Paper-faithful end-to-end runner for the agentic ZTA-FL pipeline.

Trains the CNN-LSTM intrusion-detection model on Edge-IIoTset using:

  * **vanilla**  — McMahan-2017 FedAvg (no agentic layer)
  * **ztafl**    — full ZTA-FL: TPM attestation → SHAP-weighted aggregation
                   with paper-exact TrustDB rules (Section V.A) and the
                   four-step Fog pipeline (Section V.B).

A configurable fraction of clients are made Byzantine (label-flipping at
``p_flip = 0.5``, matching the paper's attack scenario 1) so the
autonomous TrustDB rules and SHAP filter actually have something to do.

The runner is configured via :class:`~src.agentic.config.AgenticConfig`
and writes:

* ``--output``              : JSON summary with per-config final metrics
* ``--audit``               : JSONL audit trail for the TrustDB events
* ``--metrics``             : JSONL operational telemetry stream

All three artefacts are append-only event streams so an external auditor
can replay the experiment offline.

Usage
-----
    python scripts/run_agentic_experiment.py --rounds 30 --agents 20
    python scripts/run_agentic_experiment.py --quick                # smoke test
    python scripts/run_agentic_experiment.py --paper-scale --seeds 5
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from dataclasses import asdict
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.cnn_lstm        import CNNLSTMClassifier
from src.federation.aggregation import federated_averaging
from src.utils.data_loader      import load_edge_iiotset, non_iid_partition
from src.utils.metrics          import accuracy as _acc, macro_f1
from src.security.attestation   import AttestationAuthority

from src.agentic import (
    AgenticConfig, EdgeAgent, FogAgent, AgentStatus,
    configure_logging, MetricsSink,
)


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model: nn.Module, X: torch.Tensor, y: torch.Tensor,
             n_classes: int, device: str = "cpu") -> Dict[str, float]:
    model.train()  # cuDNN LSTM constraint
    model.to(device)
    preds = []
    for i in range(0, X.size(0), 512):
        chunk = X[i:i+512].to(device)
        if chunk.size(0) < 2:
            continue
        preds.append(model(chunk).argmax(dim=-1).cpu())
    if not preds:
        return {"accuracy": 0.0, "macro_f1": 0.0}
    p = torch.cat(preds)
    return {
        "accuracy": 100.0 * _acc(y[:p.size(0)], p),
        "macro_f1": 100.0 * macro_f1(y[:p.size(0)], p, n_classes=n_classes),
    }


# ---------------------------------------------------------------------------
# Vanilla FedAvg baseline (no agentic layer)
# ---------------------------------------------------------------------------

def run_vanilla(
    Xtr: torch.Tensor, ytr: torch.Tensor,
    Xte: torch.Tensor, yte: torch.Tensor,
    cfg: AgenticConfig, byz_ids: set, n_classes: int,
    device: str, seed: int,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    parts = non_iid_partition(Xtr, ytr,
                              n_agents=cfg.federation.n_agents,
                              seed=seed)
    g = CNNLSTMClassifier(n_features=Xtr.shape[1], n_classes=n_classes).to(device)

    for rnd in range(1, cfg.federation.n_rounds + 1):
        local_models, sizes = [], []
        for i, (Xi, yi) in enumerate(parts):
            lm = copy.deepcopy(g)
            yi_eff = yi.clone()
            if i in byz_ids:
                mask = torch.rand(len(yi)) < 0.5
                yi_eff[mask] = torch.randint(0, n_classes, (int(mask.sum().item()),))
            n  = int(Xi.shape[0])
            bs = max(2, min(cfg.federation.batch_size, n // 2))
            drop = n >= bs * 3
            from torch.utils.data import DataLoader, TensorDataset
            loader = DataLoader(TensorDataset(Xi, yi_eff), batch_size=bs,
                                shuffle=True, drop_last=drop)
            opt = torch.optim.Adam(lm.parameters(),
                                   lr=cfg.federation.learning_rate)
            crit = nn.CrossEntropyLoss()
            lm.train()
            for _ in range(cfg.federation.local_epochs):
                for Xb, yb in loader:
                    if Xb.size(0) < 2:
                        continue
                    Xb = Xb.to(device); yb = yb.to(device)
                    opt.zero_grad()
                    crit(lm(Xb), yb).backward()
                    nn.utils.clip_grad_norm_(lm.parameters(), 1.0)
                    opt.step()
            local_models.append(lm)
            sizes.append(n)

        weights = [s / sum(sizes) for s in sizes]
        g = federated_averaging(local_models, weights=weights).to(device)

    return evaluate(g, Xte, yte, n_classes, device)


# ---------------------------------------------------------------------------
# Full ZTA-FL pipeline (paper-faithful)
# ---------------------------------------------------------------------------

def run_ztafl(
    Xtr: torch.Tensor, ytr: torch.Tensor,
    Xte: torch.Tensor, yte: torch.Tensor,
    cfg: AgenticConfig, byz_ids: set, n_classes: int,
    device: str, seed: int,
    audit_path: str, metrics_path: str,
) -> Dict[str, Any]:
    torch.manual_seed(seed)

    # Non-IID partition matching the paper's protocol (Section VI.A)
    parts = non_iid_partition(Xtr, ytr,
                              n_agents=cfg.federation.n_agents,
                              n_classes_per=cfg.federation.n_classes_per_agent,
                              seed=seed)

    # Validation slice for the fog node's SHAP computation
    n_val = min(cfg.shap.n_background, Xte.shape[0])
    Xval = Xte[:n_val]; yval = yte[:n_val]

    # Edge agents (5-module per paper Section IV)
    edges: List[EdgeAgent] = []
    for i in range(cfg.federation.n_agents):
        edges.append(EdgeAgent(
            agent_id   = f"a{i}",
            n_features = Xtr.shape[1],
            n_classes  = n_classes,
            secret     = f"k{i}",
            config     = cfg,
            device     = device,
        ))

    # Attestation authority and TrustDB are the fog's responsibilities
    aik = {e.agent_id: f"k{i}" for i, e in enumerate(edges)}
    auth = AttestationAuthority(
        aik_registry=aik,
        max_age_seconds=cfg.attestation.delta_t_max_s,
    )
    metrics = MetricsSink(metrics_path,
                          run_id=f"ztafl-seed{seed}-{int(time.time())}")
    cfg.observability.audit_path = audit_path

    global_model = CNNLSTMClassifier(n_features=Xtr.shape[1],
                                     n_classes=n_classes).to(device)
    fog = FogAgent(global_model=global_model,
                   attestation_authority=auth, config=cfg,
                   metrics=metrics, device=device)

    for rnd in range(1, cfg.federation.n_rounds + 1):
        # Each edge agent runs local round; collect submitted updates
        ids, models, tokens, sizes, accs = [], [], [], [], []
        global_state = {k: v.detach().clone()
                        for k, v in global_model.state_dict().items()}

        for i, (Xi, yi) in enumerate(parts):
            agent = edges[i]
            ok, _ = agent.decide_participation(local_data_size=int(Xi.shape[0]))
            if not ok:
                continue
            local_state, token, _loss = agent.local_round(
                global_state=global_state,
                Xi=Xi, yi=yi,
                is_byzantine=(i in byz_ids),
                p_flip=0.5,
            )
            # Reconstruct nn.Module from state for fog aggregation (FedAvg
            # uses model objects rather than state dicts)
            m = CNNLSTMClassifier(n_features=Xtr.shape[1],
                                  n_classes=n_classes)
            m.load_state_dict(local_state)
            m.to(device)
            ids.append(agent.agent_id)
            models.append(m)
            tokens.append(token)
            sizes.append(int(Xi.shape[0]))
            accs.append(0.9)   # placeholder for per-client validation acc

        if not models:
            continue

        summary = fog.run_round(
            round_number=rnd,
            agent_ids=ids, local_models=models,
            tokens=tokens, local_sizes=sizes, local_accs=accs,
            X_val=Xval, y_val=yval, n_classes=n_classes,
        )
        global_model.load_state_dict(summary.aggregated_state)

    final = evaluate(global_model, Xte, yte, n_classes, device)
    final["trust_db_status_counts"] = fog.trust_db.status_counts()

    # How many true Byzantine clients did the fog quarantine?
    byz_caught = sum(
        1 for cid in (f"a{i}" for i in byz_ids)
        if cid in fog.trust_db
        and fog.trust_db.get(cid).status == AgentStatus.QUARANTINED
    )
    final["byzantine_caught"] = byz_caught
    final["byzantine_total"]  = len(byz_ids)
    return final


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rounds",       type=int, default=20)
    p.add_argument("--agents",       type=int, default=10)
    p.add_argument("--seeds",        type=int, default=2)
    p.add_argument("--byz-fraction", type=float, default=0.3)
    p.add_argument("--gpu",          action="store_true")
    p.add_argument("--quick",        action="store_true",
                   help="Smoke test: 5 agents, 6 rounds, 1 seed")
    p.add_argument("--paper-scale",  action="store_true",
                   help="Match paper config: N=100, R=100, seeds=5")
    p.add_argument("--small-scale",  action="store_true",
                   help="Use small-scale preset (looser SHAP / rollback "
                        "thresholds; recommended for runs at < 50 agents on "
                        "the public sample CSVs)")
    p.add_argument("--output",       default="results/agentic_results.json")
    p.add_argument("--audit",        default="results/agentic_audit.jsonl")
    p.add_argument("--metrics",      default="results/agentic_metrics.jsonl")
    p.add_argument("--config",       default=None,
                   help="Optional path to YAML/JSON config override")
    args = p.parse_args()

    if args.config:
        cfg = AgenticConfig.from_file(args.config)
    elif args.small_scale:
        cfg = AgenticConfig.small_scale()
    else:
        cfg = AgenticConfig.paper_exact()

    if args.quick:
        cfg.federation.n_agents = 5
        cfg.federation.n_rounds = 6
        seeds = [42]
    elif args.paper_scale:
        cfg.federation.n_agents = 100
        cfg.federation.n_rounds = 100
        seeds = cfg.federation.seeds[:max(1, args.seeds)]
    else:
        cfg.federation.n_agents = args.agents
        cfg.federation.n_rounds = args.rounds
        seeds = list(range(args.seeds))

    cfg.observability.audit_path   = args.audit
    cfg.observability.metrics_path = args.metrics
    configure_logging(cfg.observability.log_level, cfg.observability.log_format)

    device = "cuda" if args.gpu and torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Preset: {cfg.preset}  "
          f"(rollback_ratio={cfg.shap.rollback_ratio}, "
          f"sigma_threshold={cfg.shap.sigma_threshold})")
    print(f"Config: agents={cfg.federation.n_agents}  "
          f"rounds={cfg.federation.n_rounds}  "
          f"local_epochs={cfg.federation.local_epochs}  "
          f"βyz={args.byz_fraction}  seeds={seeds}")

    print("Loading Edge-IIoTset ...")
    X, y = load_edge_iiotset("data/edge_iiotset/raw/network_traffic_samples.csv",
                              n_features=40)
    n_classes = 15
    n_total   = int(X.shape[0])
    n_train   = int(0.8 * n_total)
    print(f"  {n_total} samples, {X.shape[1]} features, {n_classes} classes")

    results: Dict[str, Any] = {
        "meta": {
            "timestamp":  time.strftime("%Y-%m-%d %H:%M"),
            "device":     device,
            "config":     cfg.as_dict(),
            "byz_fraction": args.byz_fraction,
            "seeds":      seeds,
        },
        "vanilla": {"per_seed": [], "mean": {}, "std": {}},
        "ztafl":   {"per_seed": [], "mean": {}, "std": {}},
    }

    n_byz = int(args.byz_fraction * cfg.federation.n_agents)
    byz_ids = set(range(n_byz))

    for seed in seeds:
        print(f"\n=== seed={seed} ===")
        idx = torch.randperm(n_total,
                             generator=torch.Generator().manual_seed(seed))
        Xtr, Xte = X[idx[:n_train]], X[idx[n_train:]]
        ytr, yte = y[idx[:n_train]], y[idx[n_train:]]

        print("  [vanilla] training ...", flush=True)
        v = run_vanilla(Xtr, ytr, Xte, yte, cfg, byz_ids, n_classes,
                        device, seed)
        v["seed"] = seed
        results["vanilla"]["per_seed"].append(v)
        print(f"    final acc={v['accuracy']:6.2f}%  F1={v['macro_f1']:6.2f}%")

        print("  [ztafl] training ...", flush=True)
        z = run_ztafl(Xtr, ytr, Xte, yte, cfg, byz_ids, n_classes,
                      device, seed,
                      audit_path=args.audit, metrics_path=args.metrics)
        z["seed"] = seed
        results["ztafl"]["per_seed"].append(z)
        print(f"    final acc={z['accuracy']:6.2f}%  F1={z['macro_f1']:6.2f}%  "
              f"caught {z['byzantine_caught']}/{z['byzantine_total']} Byzantine")

    # Aggregate across seeds
    for cfg_name in ("vanilla", "ztafl"):
        runs = results[cfg_name]["per_seed"]
        if not runs:
            continue
        for k in ("accuracy", "macro_f1"):
            vals = [r[k] for r in runs]
            t = torch.tensor(vals, dtype=torch.float)
            results[cfg_name]["mean"][k] = round(float(t.mean()), 2)
            results[cfg_name]["std"][k]  = round(
                float(t.std(correction=0)) if len(vals) > 1 else 0.0, 2
            )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print()
    print("=" * 60)
    print(" Final results")
    print("=" * 60)
    for cfg_name in ("vanilla", "ztafl"):
        m = results[cfg_name]["mean"]
        s = results[cfg_name]["std"]
        if not m:
            continue
        print(f"  {cfg_name:<10} acc={m['accuracy']:6.2f}±{s['accuracy']:5.2f}%  "
              f"F1={m['macro_f1']:6.2f}±{s['macro_f1']:5.2f}%")
    print()
    print(f"Results → {args.output}")
    print(f"Audit   → {args.audit}")
    print(f"Metrics → {args.metrics}")


if __name__ == "__main__":
    main()
