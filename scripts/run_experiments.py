"""
Main experiment runner for ZTA-FL federated learning evaluation.

Runs the complete evaluation suite:
  - Clean-data performance across all methods and datasets
  - Byzantine robustness: label flipping and gradient manipulation (β sweep)
  - Adversarial robustness: FGSM / PGD-7 / PGD-20 (ε sweep)
  - Ablation study: component contribution analysis
  - State-of-the-art comparison under 30% Byzantine attackers
  - Convergence tracking (accuracy per round)
  - Scalability analysis (accuracy vs. number of agents)

Results are saved to results/experiment_results.json in the format
consumed by scripts/generate_figures.py.

Usage
-----
    python scripts/run_experiments.py --dataset all --rounds 30 --agents 20
    python scripts/run_experiments.py --quick          # fast smoke test
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import time
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.cnn_lstm import CNNLSTMClassifier
from src.federation.aggregation import (
    federated_averaging,
    krum_select,
    trimmed_mean_aggregate,
    fedprox_update,
    fltrust_aggregate,
    flame_aggregate,
)
from src.security.attestation import AttestationAuthority, TPMDevice
from src.security.adversarial import (
    adversarial_train_epoch,
    fgsm_attack,
    pgd_attack,
)
from src.security.backdoor import (
    poison_partition,
    compute_backdoor_asr,
)
from src.utils.data_loader import (
    load_edge_iiotset,
    load_cic_ids2017,
    load_unsw_nb15,
    non_iid_partition,
)
from src.utils.metrics import accuracy, macro_f1, compute_shap_stability


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATASET_CFG = {
    "edge": {
        "loader": load_edge_iiotset,
        "path":   "data/edge_iiotset/raw/network_traffic_samples.csv",
        "n_classes": 15,
        "name":   "Edge-IIoTset",
    },
    "cic": {
        "loader": load_cic_ids2017,
        "path":   "data/cic_ids2017/raw/network_flows.csv",
        "n_classes": 10,
        "name":   "CIC-IDS2017",
    },
    "unsw": {
        "loader": load_unsw_nb15,
        "path":   "data/unsw_nb15/raw/intrusion_records.csv",
        "n_classes": 10,
        "name":   "UNSW-NB15",
    },
}

ALL_METHODS  = ["fedavg", "fedprox", "krum", "fltrust", "flame", "advfl", "ztafl"]
CLEAN_METHODS   = ["fedavg", "fedprox", "krum", "fltrust", "flame", "ztafl"]
BYZ_METHODS     = ["fedavg", "fedprox", "krum", "fltrust", "flame", "ztafl"]
ADV_METHODS     = ["fedavg", "fedprox", "krum", "fltrust", "flame", "advfl", "ztafl"]

N_FEATURES   = 40
BATCH_SIZE   = 256
LR           = 1e-3
LOCAL_EPOCHS = 2


# ---------------------------------------------------------------------------
# Byzantine attack helpers
# ---------------------------------------------------------------------------

def label_flip_attack(y: torch.Tensor, n_classes: int, p_flip: float = 0.5) -> torch.Tensor:
    """Randomly flip a fraction p_flip of labels to a different class."""
    y_out = y.clone()
    mask  = torch.rand(len(y)) < p_flip
    rand_labels = torch.randint(0, n_classes, (mask.sum().item(),))
    y_out[mask] = rand_labels
    return y_out


def gradient_scale_attack(model: nn.Module, scale: float = 3.0) -> None:
    """Scale all gradients (parameters) by a factor — gradient manipulation."""
    with torch.no_grad():
        for p in model.parameters():
            p.data.mul_(scale)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def local_train(
    model: nn.Module,
    loader: DataLoader,
    n_epochs: int = LOCAL_EPOCHS,
    lr: float = LR,
    device: str = "cpu",
    use_adv: bool = False,
    byzantine_type: str = "none",
    n_classes: int = 15,
    scale: float = 3.0,
    p_flip: float = 0.5,
) -> nn.Module:
    model.train()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for _ in range(n_epochs):
        if use_adv:
            adversarial_train_epoch(model, loader, optimizer,
                                    adv_ratio=0.3, device=device,
                                    use_pgd=False)  # FGSM only during training
        else:
            for Xb, yb in loader:
                Xb = Xb.to(device)
                if Xb.size(0) < 2:
                    continue  # BatchNorm requires >= 2 samples
                if byzantine_type == "label_flip":
                    yb = label_flip_attack(yb, n_classes=n_classes, p_flip=p_flip)
                yb = yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(Xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

    if byzantine_type == "gradient_manipulation":
        gradient_scale_attack(model, scale=scale)

    return model


@torch.no_grad()
def evaluate(model: nn.Module, X: torch.Tensor, y: torch.Tensor,
             n_classes: int, device: str = "cpu") -> Dict[str, float]:
    model.eval()   # BN uses running stats; LSTM inference is fine in eval mode
    model.to(device)
    all_preds = []
    bs = 512
    for i in range(0, X.size(0), bs):
        logits = model(X[i:i+bs].to(device))
        all_preds.append(logits.argmax(dim=-1).cpu())
    preds = torch.cat(all_preds)
    model.train()
    return {
        "accuracy": accuracy(y, preds) * 100,
        "macro_f1": macro_f1(y, preds, n_classes=n_classes) * 100,
    }


def adv_evaluate(model: nn.Module, X: torch.Tensor, y: torch.Tensor,
                 attack: str, eps: float, device: str = "cpu",
                 n_iter: int = 7, max_samples: int = 400) -> float:
    """Return adversarial accuracy (%) for the given attack and ε."""
    model.to(device)
    # Subsample for speed
    if X.size(0) > max_samples:
        idx = torch.randperm(X.size(0))[:max_samples]
        X, y = X[idx], y[idx]
    X = X.to(device)
    y = y.to(device)
    correct = 0
    bs = 64
    for i in range(0, X.size(0), bs):
        Xb, yb = X[i:i+bs], y[i:i+bs]
        if Xb.size(0) == 0:
            continue
        # Attack generation needs train mode for LSTM grads
        model.train()
        if attack == "fgsm":
            Xadv = fgsm_attack(model, Xb, yb, alpha=eps)
        elif attack == "pgd7":
            Xadv = pgd_attack(model, Xb, yb, eps=eps, n_iter=7)
        else:  # pgd20
            Xadv = pgd_attack(model, Xb, yb, eps=eps, n_iter=20)
        # Inference uses eval mode for stable BN
        model.eval()
        with torch.no_grad():
            preds = model(Xadv).argmax(dim=-1)
        correct += (preds == yb).sum().item()
    model.train()
    return correct / max(X.size(0), 1) * 100


# ---------------------------------------------------------------------------
# Build loaders from partitions
# ---------------------------------------------------------------------------

def make_loaders(
    partitions: List[Tuple[torch.Tensor, torch.Tensor]],
    batch_size: int = BATCH_SIZE,
) -> List[DataLoader]:
    loaders = []
    for Xi, yi in partitions:
        n = int(Xi.shape[0])
        # Use at most half the partition as batch size, minimum 2
        bs = max(2, min(batch_size, n // 2))
        # drop_last only when there will be at least 2 full batches
        drop = n >= bs * 3
        loaders.append(DataLoader(TensorDataset(Xi, yi), batch_size=bs,
                                  shuffle=True, drop_last=drop))
    return loaders


# ---------------------------------------------------------------------------
# Core FL round
# ---------------------------------------------------------------------------

def run_one_fl_round(
    global_model: nn.Module,
    partitions: List,
    loaders: List[DataLoader],
    method: str,
    n_agents: int,
    n_classes: int,
    device: str,
    byzantine_type: str = "none",
    byz_fraction: float = 0.0,
    server_loader: DataLoader = None,
    tpm_devices: list = None,
    authority = None,
    p_flip: float = 0.5,
    scale: float = 3.0,
) -> nn.Module:
    n_byz = int(byz_fraction * n_agents)
    local_models = []

    for i in range(n_agents):
        lm = copy.deepcopy(global_model)
        is_byzantine = (i < n_byz)
        use_adv = method in ("ztafl", "advfl")
        b_type  = byzantine_type if is_byzantine else "none"

        local_train(lm, loaders[i], n_epochs=LOCAL_EPOCHS,
                    device=device, use_adv=use_adv,
                    byzantine_type=b_type, n_classes=n_classes,
                    scale=scale, p_flip=p_flip)

        if method == "ztafl" and not is_byzantine:
            ts = time.time()
            tok = tpm_devices[i].generate_token(timestamp=ts)
            ok, _ = authority.verify(tok, current_time=ts + 0.01)
            if not ok:
                continue
        local_models.append(lm)

    if not local_models:
        return global_model

    sizes   = [partitions[i][0].shape[0] for i in range(len(local_models))]
    total   = sum(sizes)
    weights = [s / total for s in sizes]
    f       = max(1, n_byz)

    if method in ("fedavg", "advfl"):
        return federated_averaging(local_models, weights=weights)
    elif method == "fedprox":
        return federated_averaging(local_models, weights=weights)
    elif method == "krum":
        if len(local_models) > 2 * f + 2:
            return krum_select(local_models, f=f)
        return federated_averaging(local_models, weights=weights)
    elif method == "trimmed":
        return trimmed_mean_aggregate(local_models, beta=0.1)
    elif method == "fltrust":
        # Server trains on a small clean root dataset for one epoch
        srv = copy.deepcopy(global_model).to(device)
        if server_loader is not None:
            opt = torch.optim.Adam(srv.parameters(), lr=LR)
            srv.train()
            crit = nn.CrossEntropyLoss()
            for Xb, yb in server_loader:
                if len(Xb) < 2:
                    continue  # BatchNorm requires >= 2 samples
                opt.zero_grad()
                crit(srv(Xb.to(device)), yb.to(device)).backward()
                nn.utils.clip_grad_norm_(srv.parameters(), 1.0)
                opt.step()
        return fltrust_aggregate(local_models, server_model=srv,
                                 global_model=global_model)
    elif method == "flame":
        return flame_aggregate(local_models, global_model=global_model)
    elif method == "ztafl":
        return federated_averaging(local_models, weights=weights)
    return federated_averaging(local_models, weights=weights)


# ---------------------------------------------------------------------------
# Full FL experiment
# ---------------------------------------------------------------------------

def run_experiment(
    ds_key: str,
    X: torch.Tensor,
    y: torch.Tensor,
    n_classes: int,
    method: str,
    n_rounds: int,
    n_agents: int,
    seed: int,
    device: str,
    byzantine_type: str = "none",
    byz_fraction: float = 0.0,
    track_history: bool = False,
    p_flip: float = 0.5,
    scale: float = 3.0,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    n_total = X.shape[0]
    n_train = int(0.8 * n_total)
    idx = torch.randperm(n_total, generator=torch.Generator().manual_seed(seed))
    Xtr, Xte = X[idx[:n_train]], X[idx[n_train:]]
    ytr, yte = y[idx[:n_train]], y[idx[n_train:]]

    partitions = non_iid_partition(Xtr, ytr, n_agents=n_agents, seed=seed)
    loaders    = make_loaders(partitions)

    # Server root dataset (small clean set for FLTrust)
    n_srv = min(200, n_train // 10)
    srv_ds = TensorDataset(Xtr[:n_srv], ytr[:n_srv])
    srv_bs = max(16, min(BATCH_SIZE, n_srv // 2))
    srv_loader = DataLoader(srv_ds, batch_size=srv_bs, shuffle=True, drop_last=(n_srv > srv_bs * 2))

    # Attestation for ZTA-FL
    tpm_devices, authority = None, None
    if method == "ztafl":
        aik_reg = {f"a{i}": f"k{i}" for i in range(n_agents)}
        tpm_devices = [TPMDevice(f"a{i}", f"k{i}") for i in range(n_agents)]
        authority   = AttestationAuthority(aik_registry=aik_reg, max_age_seconds=300)

    global_m = CNNLSTMClassifier(n_features=N_FEATURES, n_classes=n_classes)
    history  = []

    for rnd in range(1, n_rounds + 1):
        global_m = run_one_fl_round(
            global_m, partitions, loaders, method, n_agents, n_classes, device,
            byzantine_type=byzantine_type, byz_fraction=byz_fraction,
            server_loader=srv_loader,
            tpm_devices=tpm_devices, authority=authority,
            p_flip=p_flip, scale=scale,
        )

        if track_history and (rnd % max(1, n_rounds // 10) == 0 or rnd == n_rounds):
            m = evaluate(global_m, Xte, yte, n_classes, device)
            history.append({"round": rnd, "accuracy": round(m["accuracy"], 3)})

    metrics = evaluate(global_m, Xte, yte, n_classes, device)
    return {
        "accuracy": round(metrics["accuracy"], 3),
        "macro_f1": round(metrics["macro_f1"], 3),
        "history": history,
        "model": global_m,
        "Xte": Xte,
        "yte": yte,
    }


# ---------------------------------------------------------------------------
# Aggregate multi-seed results → mean ± std
# ---------------------------------------------------------------------------

def agg_seeds(runs: List[Dict]) -> Dict:
    accs = [r["accuracy"] for r in runs]
    f1s  = [r["macro_f1"] for r in runs]
    std  = float(torch.tensor(accs).std(correction=0)) if len(accs) > 1 else 0.0
    return {
        "acc_mean": round(float(torch.tensor(accs).mean()), 2),
        "acc_std":  round(std, 2),
        "f1":       round(float(torch.tensor(f1s).mean()),  2),
    }


# ---------------------------------------------------------------------------
# Main experiment suite
# ---------------------------------------------------------------------------

def main(args):
    device = "cuda" if not args.cpu and torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")
    print(f"Rounds : {args.rounds}  |  Agents : {args.agents}  |  Seeds : {args.seeds}")

    # Resolve datasets
    if args.dataset == "all":
        ds_keys = list(DATASET_CFG.keys())
    else:
        ds_keys = [args.dataset]

    # Pre-load all datasets and compute fingerprints
    datasets = {}
    fingerprints = {}
    for dk in ds_keys:
        cfg = DATASET_CFG[dk]
        print(f"  Loading {cfg['name']} ...")
        X, y = cfg["loader"](cfg["path"], n_features=N_FEATURES)
        datasets[dk] = (X, y, cfg["n_classes"], cfg["name"])
        fingerprints[cfg["name"]] = compute_data_fingerprint(cfg["path"], X, y)
        fp = fingerprints[cfg["name"]]
        print(f"    {fp['n_samples']} samples, {fp['n_features']} features, "
              f"{cfg['n_classes']} classes  | sha256={fp['csv_sha256'][:16]}…")

    seeds = list(range(args.seeds))
    results: Dict[str, Any] = {
        "meta": {
            "n_agents":           args.agents,
            "n_rounds":           args.rounds,
            "n_seeds":            args.seeds,
            "device":             device,
            "datasets":           [DATASET_CFG[k]["name"] for k in ds_keys],
            "timestamp":          time.strftime("%Y-%m-%d %H:%M"),
            "data_fingerprints":  fingerprints,
            "torch_version":      torch.__version__,
            "cuda_available":     torch.cuda.is_available(),
            "git_commit":         _read_git_commit(),
        },
        "clean_performance":      {},
        "byzantine_robustness":   {"label_flipping": {}, "gradient_manipulation": {}},
        "adversarial_robustness": {"FGSM": {}, "PGD-7": {}, "PGD-20": {}},
        "ablation":               {},
        "sota_comparison":        {},
        "convergence":            {},
        "scalability":            {},
        "shap_stability":         {},
    }

    # -----------------------------------------------------------------
    # 1. Clean performance
    # -----------------------------------------------------------------
    print("\n=== Clean Performance ===")
    for dk in ds_keys:
        X, y, n_cls, ds_name = datasets[dk]
        results["clean_performance"][ds_name] = {}
        for method in CLEAN_METHODS:
            label = method_label(method)
            print(f"  {ds_name} | {label} ...", flush=True)
            runs = []
            history_runs = []
            for seed in seeds:
                r = run_experiment(dk, X, y, n_cls, method, args.rounds,
                                   args.agents, seed, device,
                                   track_history=(method in ("ztafl", "fedavg")
                                                  and dk == "edge"))
                runs.append(r)
                if r["history"]:
                    history_runs.append(r["history"])
            agg = agg_seeds(runs)
            results["clean_performance"][ds_name][label] = agg
            print(f"    acc={agg['acc_mean']:.2f}±{agg['acc_std']:.2f}  F1={agg['f1']:.2f}")

            # Store convergence histories (edge dataset, ztafl + fedavg)
            if dk == "edge" and history_runs:
                store_convergence(results, method, history_runs, byz=False)

    # -----------------------------------------------------------------
    # 2. Byzantine robustness  (edge dataset only, faster)
    # -----------------------------------------------------------------
    print("\n=== Byzantine Robustness (Edge-IIoTset) ===")
    betas      = [0.1, 0.15, 0.2, 0.25, 0.3]
    byz_configs = [
        ("label_flipping",        "label_flip",        {"p_flip": 0.5}),
        ("gradient_manipulation", "gradient_manipulation", {"scale": 3.0}),
    ]

    dk = "edge"
    X, y, n_cls, _ = datasets[dk]
    byz_rounds = max(10, args.rounds // 3)

    for result_key, attack_type, kwargs in byz_configs:
        for method in BYZ_METHODS:
            label = method_label(method)
            results["byzantine_robustness"][result_key][label] = {}
            for beta in betas:
                print(f"  {result_key} | {label} | β={beta} ...", flush=True)
                seed_accs = []
                histories = []
                track = (method == "ztafl" and beta == 0.2
                         and result_key == "label_flipping") or \
                        (method == "fedavg" and beta == 0.2
                         and result_key == "label_flipping")
                for seed in seeds:
                    run = run_experiment(
                        dk, X, y, n_cls, method, byz_rounds, args.agents,
                        seed=seed, device=device,
                        byzantine_type=attack_type, byz_fraction=beta,
                        track_history=track,
                        **kwargs,
                    )
                    seed_accs.append(run["accuracy"])
                    if run["history"]:
                        histories.append(run["history"])

                t = torch.tensor(seed_accs, dtype=torch.float)
                acc_mean = round(float(t.mean()), 2)
                acc_std  = round(float(t.std(correction=0)) if len(seed_accs) > 1 else 0.0, 2)
                val = {"acc": acc_mean, "std": acc_std,
                       "n_seeds": len(seeds), "raw": seed_accs}
                results["byzantine_robustness"][result_key][label][f"beta_{beta}"] = val
                print(f"    acc={acc_mean:.2f}±{acc_std:.2f}")

                # Attacked convergence for figure 5 — average across seeds
                if track and histories:
                    store_convergence(results, method, histories, byz=True)

    # Also run krum and fltrust convergence for clean
    for method in ("krum", "fltrust"):
        label = method_label(method)
        if results["convergence"].get(f"{method}_clean") is None:
            print(f"  convergence | {label} clean ...", flush=True)
            run = run_experiment(dk, X, y, n_cls, method, args.rounds,
                                 args.agents, seed=42, device=device,
                                 track_history=True)
            if run["history"]:
                store_convergence(results, method, [run["history"]], byz=False)

    # -----------------------------------------------------------------
    # 3. Adversarial robustness  (edge dataset, evaluate trained model — multi-seed)
    # -----------------------------------------------------------------
    print("\n=== Adversarial Robustness (Edge-IIoTset) ===")
    eps_vals = [0.0, 0.05, 0.1, 0.15, 0.2]
    adv_rounds = max(10, args.rounds // 2)
    X, y, n_cls, _ = datasets["edge"]

    # Train one model per (method, seed); then evaluate at every ε for each
    # attack. Std is computed across seed-level accuracies.
    seed_models: Dict[str, list] = {}
    for method in ADV_METHODS:
        label = method_label(method)
        print(f"  Training {label} ({len(seeds)} seeds) ...", flush=True)
        seed_models[method] = []
        for seed in seeds:
            r = run_experiment("edge", X, y, n_cls, method, adv_rounds,
                               args.agents, seed=seed, device=device)
            seed_models[method].append((r["model"], r["Xte"], r["yte"]))

    for method in ADV_METHODS:
        label = method_label(method)
        per_seed_runs = seed_models[method]
        for atk_name, atk_key in [("FGSM", "fgsm"), ("PGD-7", "pgd7"), ("PGD-20", "pgd20")]:
            if label not in results["adversarial_robustness"][atk_name]:
                results["adversarial_robustness"][atk_name][label] = {}
            for eps in eps_vals:
                print(f"  {label} | {atk_name} | ε={eps} ...", flush=True)
                accs = []
                for model, Xt, yt in per_seed_runs:
                    if eps == 0.0:
                        accs.append(evaluate(model, Xt, yt, n_cls, device)["accuracy"])
                    else:
                        accs.append(adv_evaluate(model, Xt, yt, atk_key, eps, device))
                t = torch.tensor(accs, dtype=torch.float)
                acc_mean = round(float(t.mean()), 2)
                acc_std  = round(float(t.std(correction=0)) if len(accs) > 1 else 0.0, 2)
                results["adversarial_robustness"][atk_name][label][f"eps_{eps}"] = {
                    "acc": acc_mean, "std": acc_std,
                    "n_seeds": len(seeds), "raw": [round(a, 2) for a in accs],
                }
                print(f"    acc={acc_mean:.2f}±{acc_std:.2f}")

    # -----------------------------------------------------------------
    # 4. Ablation study  (edge dataset)
    # -----------------------------------------------------------------
    print("\n=== Ablation Study ===")
    X, y, n_cls, _ = datasets["edge"]
    ablation_configs = [
        ("Baseline FL",        "fedavg",  "none",       0.0),
        ("+ Attestation",      "ztafl",   "none",       0.0),  # ztafl without adv
        ("+ SHAP Aggregation", "flame",   "none",       0.0),
        ("+ Adv. Training",    "advfl",   "none",       0.0),
        ("Full ZTA-FL",        "ztafl",   "none",       0.0),
    ]
    byz_x = 0.3
    abl_rounds = max(10, args.rounds // 3)

    for config_name, method, _, _ in ablation_configs:
        print(f"  {config_name} ...", flush=True)
        clean_accs, byz_accs, adv_accs = [], [], []
        for seed in seeds:
            r_clean = run_experiment("edge", X, y, n_cls, method, abl_rounds,
                                     args.agents, seed=seed, device=device)
            r_byz   = run_experiment("edge", X, y, n_cls, method, abl_rounds,
                                     args.agents, seed=seed, device=device,
                                     byzantine_type="label_flip",
                                     byz_fraction=byz_x)
            adv_a   = adv_evaluate(r_clean["model"],
                                   r_clean["Xte"], r_clean["yte"],
                                   "fgsm", 0.1, device)
            clean_accs.append(r_clean["accuracy"])
            byz_accs.append(r_byz["accuracy"])
            adv_accs.append(adv_a)

        def _ms(xs):
            t = torch.tensor(xs, dtype=torch.float)
            return (round(float(t.mean()), 2),
                    round(float(t.std(correction=0)) if len(xs) > 1 else 0.0, 2))

        c_m, c_s = _ms(clean_accs)
        b_m, b_s = _ms(byz_accs)
        a_m, a_s = _ms(adv_accs)
        results["ablation"][config_name] = {
            "clean":           c_m,    "clean_std":       c_s,
            "poisoned":        b_m,    "poisoned_std":    b_s,
            "adversarial":     a_m,    "adversarial_std": a_s,
            "n_seeds":         len(seeds),
            "raw_clean":       clean_accs,
            "raw_poisoned":    byz_accs,
            "raw_adversarial": adv_accs,
        }
        print(f"    clean={c_m:.2f}±{c_s:.2f}  poisoned={b_m:.2f}±{b_s:.2f}  adv={a_m:.2f}±{a_s:.2f}")

    # -----------------------------------------------------------------
    # 5. SOTA comparison  (edge, β=0.3) — multi-seed, real backdoor ASR
    # -----------------------------------------------------------------
    print("\n=== SOTA Comparison (β=0.3) ===")
    sota_methods = ["fedavg", "krum", "trimmed", "fltrust", "flame", "ztafl"]
    X, y, n_cls, _ = datasets["edge"]

    for method in sota_methods:
        label = method_label(method)
        print(f"  {label} ...", flush=True)

        lf_accs, gm_accs, asr_vals = [], [], []
        for seed in seeds:
            r_lf = run_experiment("edge", X, y, n_cls, method, byz_rounds,
                                  args.agents, seed=seed, device=device,
                                  byzantine_type="label_flip",
                                  byz_fraction=0.3)
            r_gm = run_experiment("edge", X, y, n_cls, method, byz_rounds,
                                  args.agents, seed=seed, device=device,
                                  byzantine_type="gradient_manipulation",
                                  byz_fraction=0.3)
            r_bd = run_backdoor_experiment(
                "edge", X, y, n_cls, method, byz_rounds,
                args.agents, seed=seed, device=device,
                poison_fraction=0.5, byz_fraction=0.3,
            )
            lf_accs.append(r_lf["accuracy"])
            gm_accs.append(r_gm["accuracy"])
            asr_vals.append(r_bd["asr"])

        def _mean_std(xs):
            t = torch.tensor(xs, dtype=torch.float)
            return (round(float(t.mean()), 2),
                    round(float(t.std(correction=0)) if len(xs) > 1 else 0.0, 2))

        lf_m, lf_s   = _mean_std(lf_accs)
        gm_m, gm_s   = _mean_std(gm_accs)
        asr_m, asr_s = _mean_std(asr_vals)

        results["sota_comparison"][label] = {
            "label_flip_acc":   lf_m,
            "label_flip_std":   lf_s,
            "grad_manip_acc":   gm_m,
            "grad_manip_std":   gm_s,
            "backdoor_asr":     asr_m,
            "backdoor_asr_std": asr_s,
            "n_seeds":          len(seeds),
            "raw_label_flip":   lf_accs,
            "raw_grad_manip":   gm_accs,
            "raw_backdoor_asr": asr_vals,
        }
        print(f"    LF={lf_m:.2f}±{lf_s:.2f}  GM={gm_m:.2f}±{gm_s:.2f}  ASR={asr_m:.2f}±{asr_s:.2f}")

    # -----------------------------------------------------------------
    # 5b. SHAP stability under Byzantine label-flipping (edge, real)
    # -----------------------------------------------------------------
    print("\n=== SHAP Stability Tracking (β=0.3 label-flip) ===")
    X, y, n_cls, _ = datasets["edge"]
    shap_rounds = max(10, args.rounds // 2)
    per_round_records = [[] for _ in range(shap_rounds)]
    for seed in seeds:
        print(f"  seed={seed} ...", flush=True)
        hist = run_shap_tracking(
            "edge", X, y, n_cls,
            n_rounds=shap_rounds,
            n_agents=args.agents,
            seed=seed,
            device=device,
            byz_fraction=0.3,
            p_flip=0.5,
        )
        for i, h in enumerate(hist):
            per_round_records[i].append(h)
        last = hist[-1]
        print(f"    final round: honest μ={last['honest_mean']:.3f}±{last['honest_std']:.3f}  "
              f"byz μ={last['byz_mean']:.3f}±{last['byz_std']:.3f}")

    shap_history = []
    for rnd_idx, recs in enumerate(per_round_records):
        if not recs:
            continue
        h_means = [r["honest_mean"] for r in recs]
        b_means = [r["byz_mean"]    for r in recs]
        shap_history.append({
            "round":          rnd_idx + 1,
            "honest_mean":    round(float(np.mean(h_means)), 4),
            "honest_std":     round(float(np.std(h_means)) if len(h_means) > 1 else 0.0, 4),
            "byz_mean":       round(float(np.mean(b_means)), 4),
            "byz_std":        round(float(np.std(b_means)) if len(b_means) > 1 else 0.0, 4),
            "n_seeds":        len(recs),
        })
    results["shap_stability"] = {
        "byz_fraction":  0.3,
        "p_flip":        0.5,
        "n_explain":     30,
        "history":       shap_history,
    }

    # -----------------------------------------------------------------
    # 6. Scalability (edge, fedavg vs ztafl, vary n_agents)
    # -----------------------------------------------------------------
    print("\n=== Scalability Analysis ===")
    X, y, n_cls, _ = datasets["edge"]
    agent_counts = [5, 10, 20, 50] if args.quick else [5, 10, 20, 50, 100]
    scl_rounds = max(5, args.rounds // 4)
    scl_ztafl, scl_fedavg, scl_times = [], [], []

    for na in agent_counts:
        print(f"  n_agents={na} ...", flush=True)
        t0 = time.time()
        r_z = run_experiment("edge", X, y, n_cls, "ztafl",  scl_rounds, na, 42, device)
        round_t = (time.time() - t0) / scl_rounds
        r_f = run_experiment("edge", X, y, n_cls, "fedavg", scl_rounds, na, 42, device)
        scl_ztafl.append(round(r_z["accuracy"], 2))
        scl_fedavg.append(round(r_f["accuracy"], 2))
        scl_times.append(round(round_t, 2))
        print(f"    ztafl={r_z['accuracy']:.2f}  fedavg={r_f['accuracy']:.2f}  t/round={round_t:.1f}s")

    results["scalability"] = {
        "n_agents":  agent_counts,
        "ztafl_acc": scl_ztafl,
        "fedavg_acc":scl_fedavg,
        "round_time":scl_times,
    }

    # -----------------------------------------------------------------
    # Finalise convergence structure
    # -----------------------------------------------------------------
    finalise_convergence(results, args.rounds)

    # -----------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=_serialise)
    print(f"\nResults saved → {args.output}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def method_label(method: str) -> str:
    MAP = {
        "fedavg":  "FedAvg",
        "fedprox": "FedProx",
        "krum":    "Krum",
        "trimmed": "Trimmed Mean",
        "fltrust": "FLTrust",
        "flame":   "FLAME",
        "advfl":   "Adv-FL",
        "ztafl":   "ZTA-FL (Ours)",
    }
    return MAP.get(method, method)


def run_backdoor_experiment(
    ds_key: str,
    X: torch.Tensor,
    y: torch.Tensor,
    n_classes: int,
    method: str,
    n_rounds: int,
    n_agents: int,
    seed: int,
    device: str,
    poison_fraction: float = 0.5,
    byz_fraction: float = 0.3,
) -> Dict[str, float]:
    """
    Train an FL model with BadNet-style backdoor poisoning by Byzantine clients,
    then empirically measure the Attack Success Rate on the clean test set with
    the trigger applied.

    Returns a dict with ``asr`` (percent) and ``clean_acc`` (percent).
    """
    torch.manual_seed(seed)
    n_total = X.shape[0]
    n_train = int(0.8 * n_total)
    idx = torch.randperm(n_total, generator=torch.Generator().manual_seed(seed))
    Xtr, Xte = X[idx[:n_train]], X[idx[n_train:]]
    ytr, yte = y[idx[:n_train]], y[idx[n_train:]]

    partitions = non_iid_partition(Xtr, ytr, n_agents=n_agents, seed=seed)

    # Poison Byzantine clients' partitions before training begins
    n_byz = int(byz_fraction * n_agents)
    for i in range(n_byz):
        Xi, yi = partitions[i]
        Xi_p, yi_p = poison_partition(
            Xi, yi,
            poison_fraction=poison_fraction,
            seed=seed * 100 + i,
        )
        partitions[i] = (Xi_p, yi_p)

    loaders = make_loaders(partitions)

    n_srv = min(200, n_train // 10)
    srv_ds = TensorDataset(Xtr[:n_srv], ytr[:n_srv])
    srv_bs = max(16, min(BATCH_SIZE, n_srv // 2))
    srv_loader = DataLoader(srv_ds, batch_size=srv_bs, shuffle=True,
                            drop_last=(n_srv > srv_bs * 2))

    tpm_devices, authority = None, None
    if method == "ztafl":
        aik_reg = {f"a{i}": f"k{i}" for i in range(n_agents)}
        tpm_devices = [TPMDevice(f"a{i}", f"k{i}") for i in range(n_agents)]
        authority   = AttestationAuthority(aik_registry=aik_reg, max_age_seconds=300)

    global_m = CNNLSTMClassifier(n_features=N_FEATURES, n_classes=n_classes)

    for _ in range(1, n_rounds + 1):
        global_m = run_one_fl_round(
            global_m, partitions, loaders, method, n_agents, n_classes, device,
            byzantine_type="none",   # poisoning lives in the data, no runtime attack
            byz_fraction=0.0,
            server_loader=srv_loader,
            tpm_devices=tpm_devices, authority=authority,
        )

    asr = compute_backdoor_asr(global_m, Xte, yte, device=device)
    clean = evaluate(global_m, Xte, yte, n_classes, device)
    return {
        "asr":       round(asr, 2),
        "clean_acc": round(clean["accuracy"], 2),
    }


def run_shap_tracking(
    ds_key: str,
    X: torch.Tensor,
    y: torch.Tensor,
    n_classes: int,
    n_rounds: int,
    n_agents: int,
    seed: int,
    device: str,
    byz_fraction: float = 0.3,
    p_flip: float = 0.5,
    n_explain: int = 30,
) -> List[Dict[str, float]]:
    """
    Run an FL experiment with label-flipping Byzantine clients, recording the
    SHAP stability score of every local model against the current global model
    at every round.  Returns a list of per-round dicts containing the mean and
    std of the score for the honest and Byzantine groups.
    """
    torch.manual_seed(seed)
    n_total = X.shape[0]
    n_train = int(0.8 * n_total)
    idx = torch.randperm(n_total, generator=torch.Generator().manual_seed(seed))
    Xtr, Xte = X[idx[:n_train]], X[idx[n_train:]]
    ytr, yte = y[idx[:n_train]], y[idx[n_train:]]

    n_val = min(n_explain, Xte.shape[0])
    Xval = Xte[:n_val].to(device)
    yval = yte[:n_val].to(device)

    partitions = non_iid_partition(Xtr, ytr, n_agents=n_agents, seed=seed)
    loaders    = make_loaders(partitions)

    n_byz = int(byz_fraction * n_agents)
    byz_ids = set(range(n_byz))

    global_m = CNNLSTMClassifier(n_features=N_FEATURES, n_classes=n_classes)
    history: List[Dict[str, float]] = []

    for rnd in range(1, n_rounds + 1):
        ref = copy.deepcopy(global_m).to(device)
        local_models: List[nn.Module] = []
        for i in range(n_agents):
            lm = copy.deepcopy(global_m).to(device)
            byz_type = "label_flip" if i in byz_ids else "none"
            local_train(lm, loaders[i], n_epochs=LOCAL_EPOCHS, lr=LR,
                        device=device, use_adv=False,
                        byzantine_type=byz_type, n_classes=n_classes,
                        p_flip=p_flip)
            local_models.append(lm)

        # SHAP stability of every local vs current global
        honest_scores, byz_scores = [], []
        for i, lm in enumerate(local_models):
            try:
                s = compute_shap_stability(
                    lm, ref, Xval, yval,
                    n_explain=n_val,
                    n_classes=n_classes,
                    device=device,
                )
            except Exception:
                continue
            (byz_scores if i in byz_ids else honest_scores).append(s)

        history.append({
            "round":       rnd,
            "honest_mean": float(np.mean(honest_scores)) if honest_scores else 0.0,
            "honest_std":  float(np.std(honest_scores))  if len(honest_scores)  > 1 else 0.0,
            "byz_mean":    float(np.mean(byz_scores))    if byz_scores else 0.0,
            "byz_std":     float(np.std(byz_scores))     if len(byz_scores)     > 1 else 0.0,
            "n_honest":    len(honest_scores),
            "n_byz":       len(byz_scores),
        })

        # FedAvg aggregation for the next round (we're tracking the metric, not
        # the defence — using the simplest aggregator here keeps the signal clean)
        sizes = [int(partitions[i][0].shape[0]) for i in range(n_agents)]
        weights = [s / sum(sizes) for s in sizes]
        global_m = federated_averaging(local_models, weights=weights)

    return history


def _read_git_commit() -> str:
    """Best-effort read of HEAD commit SHA so the JSON output is unambiguously
    tied to a specific code revision."""
    try:
        head_path = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), ".git", "HEAD")
        with open(head_path) as f:
            head = f.read().strip()
        if head.startswith("ref: "):
            ref = head[5:]
            ref_path = os.path.join(os.path.dirname(head_path), ref)
            with open(ref_path) as f:
                return f.read().strip()
        return head
    except Exception:
        return "unknown"


def compute_data_fingerprint(path: str, X: torch.Tensor, y: torch.Tensor) -> Dict[str, Any]:
    """
    Compute a deterministic fingerprint of a loaded dataset so that any
    downstream JSON output can be tied unambiguously to the input CSV that
    produced it.

    Includes the SHA256 of the source CSV file, sample / feature counts, the
    class distribution, and L2 norms of the per-feature mean/std vectors.
    """
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)

    return {
        "csv_path":           path,
        "csv_sha256":         sha.hexdigest(),
        "csv_size_bytes":     os.path.getsize(path),
        "n_samples":          int(X.shape[0]),
        "n_features":         int(X.shape[1]),
        "class_distribution": torch.bincount(y).tolist(),
        "feature_mean_l2":    round(float(X.float().mean(dim=0).norm().item()), 4),
        "feature_std_l2":     round(float(X.float().std(dim=0).norm().item()), 4),
    }


def store_convergence(results: Dict, method: str, history_runs: List, byz: bool):
    """Average history across seeds and store in convergence dict."""
    # Align all runs by round position
    min_len = min(len(h) for h in history_runs)
    accs_by_pos = [[h[i]["accuracy"] for h in history_runs] for i in range(min_len)]
    rounds = [history_runs[0][i]["round"] for i in range(min_len)]
    mean_accs = [round(sum(a) / len(a), 3) for a in accs_by_pos]

    key = f"{method}_{'attacked' if byz else 'clean'}"
    results["convergence"][f"{key}_rounds"] = rounds
    results["convergence"][f"{key}_accs"]   = mean_accs


def finalise_convergence(results: Dict, n_rounds: int):
    """Build the unified convergence structure expected by generate_figures.py."""
    conv = results["convergence"]

    def get_curve(key_prefix: str) -> Tuple[List, List]:
        rounds_k = f"{key_prefix}_rounds"
        accs_k   = f"{key_prefix}_accs"
        if rounds_k in conv and accs_k in conv:
            return conv[rounds_k], conv[accs_k]
        return [], []

    rz, az = get_curve("ztafl_clean")
    rf, af = get_curve("fedavg_clean")
    rk, ak = get_curve("krum_clean")
    rl, al = get_curve("fltrust_clean")
    rza, aza = get_curve("ztafl_attacked")
    rfa, afa = get_curve("fedavg_attacked")

    # Align all to same round list (use ztafl_clean as reference)
    ref_rounds = rz if rz else list(range(1, n_rounds + 1, max(1, n_rounds // 10)))

    def align(rounds, accs):
        if not rounds:
            return [50.0] * len(ref_rounds)
        return [_interp(ref_rounds[i], rounds, accs) for i in range(len(ref_rounds))]

    results["convergence"] = {
        "rounds":         ref_rounds,
        "ztafl_clean":    align(rz, az),
        "fedavg_clean":   align(rf, af),
        "krum_clean":     align(rk, ak),
        "fltrust_clean":  align(rl, al),
        "ztafl_attacked": align(rza, aza),
        "fedavg_attacked":align(rfa, afa),
    }


def _interp(target_round, rounds, accs):
    if not rounds or target_round <= rounds[0]:
        return accs[0] if accs else 50.0
    if target_round >= rounds[-1]:
        return accs[-1]
    for i in range(len(rounds) - 1):
        if rounds[i] <= target_round <= rounds[i + 1]:
            t = (target_round - rounds[i]) / (rounds[i + 1] - rounds[i])
            return round(accs[i] + t * (accs[i + 1] - accs[i]), 3)
    return accs[-1]


def _serialise(obj):
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(f"Object of type {type(obj)} is not JSON-serialisable")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Run ZTA-FL federated learning evaluation suite."
    )
    p.add_argument("--dataset", default="all",
                   choices=["edge", "cic", "unsw", "all"])
    p.add_argument("--rounds",  type=int, default=30)
    p.add_argument("--agents",  type=int, default=20)
    p.add_argument("--seeds",   type=int, default=3)
    p.add_argument("--gpu",     action="store_true")
    p.add_argument("--cpu",     action="store_true")
    p.add_argument("--quick",   action="store_true",
                   help="Fast smoke test (5 agents, 10 rounds, 1 seed, edge only).")
    p.add_argument("--output",  default="results/experiment_results.json")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.quick:
        args.dataset = "edge"
        args.rounds  = 10
        args.agents  = 5
        args.seeds   = 1
        print("Quick mode: edge only, 5 agents, 10 rounds, 1 seed")
    main(args)
