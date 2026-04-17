"""
Byzantine robustness evaluation experiment.

Evaluates ZTA-FL and baseline methods under varying fractions of Byzantine
agents β ∈ {0.1, 0.2, 0.3}.  Two attack types are tested:
  1. Label Flipping  — Byzantine agents randomly reassign training labels
  2. Gradient Manipulation — Byzantine agents scale gradients by a large factor

Results are written to results/byzantine_robustness.json.

Usage
-----
    python experiments/byzantine_robustness.py
    python experiments/byzantine_robustness.py --dataset edge --rounds 30 --agents 20
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
)
from src.security.attestation import AttestationAuthority, TPMDevice
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

BYZANTINE_FRACTIONS = [0.1, 0.2, 0.3]
ATTACK_TYPES = ["label_flip", "gradient_manipulation"]
METHODS = ["fedavg", "krum", "trimmed", "ztafl"]

N_FEATURES = 40
BATCH_SIZE = 64
LOCAL_EPOCHS = 3
LR = 1e-3


def label_flip_loader(loader: DataLoader, n_classes: int) -> DataLoader:
    """Return a new loader with randomly flipped labels (Byzantine attack)."""
    X_all, y_all = [], []
    for X_b, y_b in loader:
        X_all.append(X_b)
        y_all.append(y_b)
    X_all = torch.cat(X_all, dim=0)
    y_all = torch.cat(y_all, dim=0)
    # Flip to a random different class
    y_flipped = torch.randint(0, n_classes, y_all.shape)
    return DataLoader(TensorDataset(X_all, y_flipped), batch_size=BATCH_SIZE, shuffle=True)


def local_train_byzantine(
    model: nn.Module,
    loader: DataLoader,
    attack: str,
    n_classes: int,
    scale: float = 10.0,
    device: str = "cpu",
) -> None:
    """Train a Byzantine agent using the specified attack strategy."""
    model.train()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    if attack == "label_flip":
        loader = label_flip_loader(loader, n_classes)

    for _ in range(LOCAL_EPOCHS):
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            if len(X_b) < 2:
                continue
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            if attack == "gradient_manipulation":
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.mul_(scale)
            nn.utils.clip_grad_norm_(model.parameters(), 100.0)
            optimizer.step()


def local_train_honest(model, loader, device="cpu"):
    model.train()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    for _ in range(LOCAL_EPOCHS):
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            if len(X_b) < 2:
                continue
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()


def evaluate(model, X, y, n_classes, device="cpu"):
    model.train()
    model.to(device)
    with torch.no_grad():
        preds = model(X.to(device)).argmax(dim=-1).cpu()
    return {
        "accuracy": accuracy(y, preds),
        "macro_f1": macro_f1(y, preds, n_classes=n_classes),
    }


def run_byzantine_experiment(
    dataset_key, method, attack, beta, n_rounds, n_agents, seed, device
):
    cfg = DATASET_CONFIG[dataset_key]
    torch.manual_seed(seed)

    X, y = cfg["loader"](cfg["path"], n_features=N_FEATURES)
    n_classes = cfg["n_classes"]
    n_total = X.shape[0]
    n_train = int(0.8 * n_total)
    idx = torch.randperm(n_total, generator=torch.Generator().manual_seed(seed))
    X_train, X_test = X[idx[:n_train]], X[idx[n_train:]]
    y_train, y_test = y[idx[:n_train]], y[idx[n_train:]]

    partitions = non_iid_partition(X_train, y_train, n_agents=n_agents, seed=seed)
    loaders = [
        DataLoader(TensorDataset(Xi, yi), batch_size=BATCH_SIZE, shuffle=True)
        for Xi, yi in partitions
    ]

    n_byzantine = max(1, int(beta * n_agents))
    byzantine_ids = set(range(n_agents - n_byzantine, n_agents))

    if method == "ztafl":
        aik_reg = {f"agent-{i}": f"secret-{seed}-{i}" for i in range(n_agents)}
        tpm_devs = [TPMDevice(f"agent-{i}", f"secret-{seed}-{i}") for i in range(n_agents)]
        auth = AttestationAuthority(aik_registry=aik_reg, max_age_seconds=300.0)

    global_model = CNNLSTMClassifier(n_features=N_FEATURES, n_classes=n_classes)
    history = []

    for rnd in range(1, n_rounds + 1):
        local_models = []
        for i in range(n_agents):
            local_m = copy.deepcopy(global_model)
            if i in byzantine_ids:
                local_train_byzantine(local_m, loaders[i], attack, n_classes, device=device)
            else:
                local_train_honest(local_m, loaders[i], device=device)

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

        if method in ("fedavg", "ztafl"):
            global_model = federated_averaging(local_models, weights=weights)
        elif method == "krum":
            f = max(1, int(beta * len(local_models)))
            if len(local_models) > 2 * f + 2:
                global_model = krum_select(local_models, f=f)
            else:
                global_model = federated_averaging(local_models, weights=weights)
        elif method == "trimmed":
            global_model = trimmed_mean_aggregate(local_models, beta=beta)

        if rnd % max(1, n_rounds // 5) == 0 or rnd == n_rounds:
            m = evaluate(global_model, X_test, y_test, n_classes, device=device)
            history.append({"round": rnd, **m})

    final = evaluate(global_model, X_test, y_test, n_classes, device=device)
    return {
        "dataset": cfg["name"],
        "method": method,
        "attack": attack,
        "byzantine_fraction": beta,
        "n_byzantine": n_byzantine,
        "seed": seed,
        "final_metrics": final,
        "history": history,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Byzantine robustness evaluation.")
    parser.add_argument("--dataset", default="edge", choices=["edge", "cic", "unsw", "all"])
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--agents", type=int, default=20)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="results/byzantine_robustness.json")
    return parser.parse_args()


def main():
    args = parse_args()
    datasets = list(DATASET_CONFIG.keys()) if args.dataset == "all" else [args.dataset]
    all_results = []

    for ds_key in datasets:
        for method in METHODS:
            for attack in ATTACK_TYPES:
                for beta in BYZANTINE_FRACTIONS:
                    for seed in range(args.seeds):
                        print(
                            f"Dataset={DATASET_CONFIG[ds_key]['name']} | "
                            f"method={method} | attack={attack} | "
                            f"beta={beta} | seed={seed}"
                        )
                        result = run_byzantine_experiment(
                            ds_key, method, attack, beta,
                            n_rounds=args.rounds, n_agents=args.agents,
                            seed=seed, device=args.device,
                        )
                        all_results.append(result)
                        print(
                            f"  acc={result['final_metrics']['accuracy']:.4f}, "
                            f"f1={result['final_metrics']['macro_f1']:.4f}"
                        )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
