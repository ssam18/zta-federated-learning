"""
Baseline comparison experiment: ZTA-FL vs. standard federated learning methods.

Evaluates five methods on each IIoT intrusion detection dataset:
  1. FedAvg       — McMahan et al. (2017) standard federated averaging
  2. FedProx      — Li et al. (2020) proximal-term regularisation
  3. Krum          — Blanchard et al. (2017) Byzantine-robust selection
  4. Trimmed Mean  — Yin et al. (2018) coordinate-wise trimmed aggregation
  5. Adv-FL        — Adversarial training without zero-trust attestation
  6. ZTA-FL        — Full zero-trust framework (this work)

Results are written to results/baseline_comparison.json.

Usage
-----
    python experiments/baseline_comparison.py
    python experiments/baseline_comparison.py --dataset edge --rounds 30 --agents 10
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from typing import Dict, Any, List

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
)
from src.security.attestation import AttestationAuthority, TPMDevice
from src.security.adversarial import adversarial_train_epoch
from src.utils.data_loader import (
    load_edge_iiotset,
    load_cic_ids2017,
    load_unsw_nb15,
    non_iid_partition,
)
from src.utils.metrics import accuracy, macro_f1


DATASET_CONFIG = {
    "edge": {
        "loader": load_edge_iiotset,
        "path": "data/edge_iiotset/raw/network_traffic_samples.csv",
        "n_classes": 15,
        "name": "Edge-IIoTset",
    },
    "cic": {
        "loader": load_cic_ids2017,
        "path": "data/cic_ids2017/raw/network_flows.csv",
        "n_classes": 10,
        "name": "CIC-IDS2017",
    },
    "unsw": {
        "loader": load_unsw_nb15,
        "path": "data/unsw_nb15/raw/intrusion_records.csv",
        "n_classes": 10,
        "name": "UNSW-NB15",
    },
}

N_FEATURES = 40
BATCH_SIZE = 64
LOCAL_EPOCHS = 3
LR = 1e-3
METHODS = ["fedavg", "fedprox", "krum", "trimmed", "advfl", "ztafl"]


def local_train(model, loader, use_adv=False, device="cpu"):
    model.train()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    loss_val = 0.0
    for _ in range(LOCAL_EPOCHS):
        if use_adv:
            loss_val = adversarial_train_epoch(model, loader, optimizer, device=device)
        else:
            total, nb = 0.0, 0
            for X_b, y_b in loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                if len(X_b) < 2:
                    continue
                optimizer.zero_grad()
                loss = criterion(model(X_b), y_b)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total += loss.item()
                nb += 1
            loss_val = total / max(nb, 1)
    return loss_val


def evaluate(model, X, y, n_classes, device="cpu"):
    model.train()
    model.to(device)
    with torch.no_grad():
        preds = model(X.to(device)).argmax(dim=-1).cpu()
    return {
        "accuracy": accuracy(y, preds),
        "macro_f1": macro_f1(y, preds, n_classes=n_classes),
    }


def run_method(method, X_train, y_train, X_test, y_test, n_classes,
               n_rounds, n_agents, seed, device):
    torch.manual_seed(seed)
    partitions = non_iid_partition(X_train, y_train, n_agents=n_agents, seed=seed)
    loaders = [
        DataLoader(TensorDataset(Xi, yi), batch_size=BATCH_SIZE, shuffle=True)
        for Xi, yi in partitions
    ]
    global_model = CNNLSTMClassifier(n_features=N_FEATURES, n_classes=n_classes)

    if method == "ztafl":
        aik_reg = {f"agent-{i}": f"secret-{seed}-{i}" for i in range(n_agents)}
        tpm_devs = [TPMDevice(f"agent-{i}", f"secret-{seed}-{i}") for i in range(n_agents)]
        auth = AttestationAuthority(aik_registry=aik_reg, max_age_seconds=300.0)

    history = []
    for rnd in range(1, n_rounds + 1):
        local_models = []
        for i in range(n_agents):
            local_m = copy.deepcopy(global_model)
            local_train(local_m, loaders[i], use_adv=(method in ("ztafl", "advfl")), device=device)
            if method == "ztafl":
                ts = time.time()
                tok = tpm_devs[i].generate_token(timestamp=ts)
                ok, _ = auth.verify(tok, current_time=ts + 0.1)
                if not ok:
                    continue
            local_models.append(local_m)

        if not local_models:
            continue

        sizes = [partitions[i][0].shape[0] for i in range(len(local_models))]
        weights = [s / sum(sizes) for s in sizes]

        if method in ("fedavg", "advfl", "ztafl"):
            global_model = federated_averaging(local_models, weights=weights)
        elif method == "krum":
            f = max(1, len(local_models) // 4)
            if len(local_models) > 2 * f + 2:
                global_model = krum_select(local_models, f=f)
            else:
                global_model = federated_averaging(local_models, weights=weights)
        elif method == "trimmed":
            global_model = trimmed_mean_aggregate(local_models, beta=0.1)
        elif method == "fedprox":
            for i, lm in enumerate(local_models):
                ds = TensorDataset(partitions[i][0], partitions[i][1])
                ld = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
                opt = torch.optim.Adam(lm.parameters(), lr=LR)
                fedprox_update(lm, global_model, ld, opt, mu=0.01, device=device)
            global_model = federated_averaging(local_models, weights=weights)

        if rnd % max(1, n_rounds // 5) == 0 or rnd == n_rounds:
            m = evaluate(global_model, X_test, y_test, n_classes, device=device)
            history.append({"round": rnd, **m})

    final = evaluate(global_model, X_test, y_test, n_classes, device=device)
    return {"method": method, "final_metrics": final, "history": history}


def parse_args():
    parser = argparse.ArgumentParser(description="Baseline comparison experiment.")
    parser.add_argument("--dataset", default="all", choices=["edge", "cic", "unsw", "all"])
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--agents", type=int, default=10)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="results/baseline_comparison.json")
    return parser.parse_args()


def main():
    args = parse_args()
    datasets = list(DATASET_CONFIG.keys()) if args.dataset == "all" else [args.dataset]
    all_results = []

    for ds_key in datasets:
        cfg = DATASET_CONFIG[ds_key]
        print(f"\nLoading dataset: {cfg['name']} ...")
        X, y = cfg["loader"](cfg["path"], n_features=N_FEATURES)
        n_classes = cfg["n_classes"]
        n_total = X.shape[0]
        n_train = int(0.8 * n_total)

        for seed in range(args.seeds):
            idx = torch.randperm(n_total, generator=torch.Generator().manual_seed(seed))
            X_train, X_test = X[idx[:n_train]], X[idx[n_train:]]
            y_train, y_test = y[idx[:n_train]], y[idx[n_train:]]

            for method in METHODS:
                print(f"  Method={method}, seed={seed} ...")
                result = run_method(
                    method, X_train, y_train, X_test, y_test,
                    n_classes=n_classes, n_rounds=args.rounds,
                    n_agents=args.agents, seed=seed, device=args.device,
                )
                result.update({
                    "dataset": cfg["name"],
                    "seed": seed,
                    "n_rounds": args.rounds,
                    "n_agents": args.agents,
                })
                all_results.append(result)
                print(
                    f"    acc={result['final_metrics']['accuracy']:.4f}, "
                    f"f1={result['final_metrics']['macro_f1']:.4f}"
                )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
