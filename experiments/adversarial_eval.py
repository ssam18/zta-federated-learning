"""
Adversarial robustness evaluation experiment.

Measures clean and adversarial accuracy of trained federated models at
multiple perturbation budgets ε ∈ {0.01, 0.05, 0.1, 0.2, 0.3} under both
FGSM and PGD attacks.

Methods evaluated:
  - Standard FedAvg (no adversarial training)
  - Adversarial-FL  (FedAvg with per-round adversarial training)
  - ZTA-FL          (full framework with attestation + adversarial training)

Results are written to results/adversarial_eval.json.

Usage
-----
    python experiments/adversarial_eval.py
    python experiments/adversarial_eval.py --dataset edge --rounds 20 --agents 10
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
from src.federation.aggregation import federated_averaging
from src.security.attestation import AttestationAuthority, TPMDevice
from src.security.adversarial import adversarial_train_epoch, evaluate_robustness
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

EPS_VALUES = [0.01, 0.05, 0.1, 0.2, 0.3]
ATTACKS = ["fgsm", "pgd"]
METHODS = ["fedavg", "advfl", "ztafl"]

N_FEATURES = 40
BATCH_SIZE = 64
LOCAL_EPOCHS = 3
LR = 1e-3


def local_train(model, loader, use_adv=False, device="cpu"):
    model.train()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    for _ in range(LOCAL_EPOCHS):
        if use_adv:
            adversarial_train_epoch(model, loader, optimizer, device=device)
        else:
            for X_b, y_b in loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                if len(X_b) < 2:
                    continue
                optimizer.zero_grad()
                loss = criterion(model(X_b), y_b)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()


def train_federated_model(
    method, X_train, y_train, n_classes, n_rounds, n_agents, seed, device
):
    """Train a federated model and return the final global model."""
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

    for rnd in range(1, n_rounds + 1):
        local_models = []
        for i in range(n_agents):
            local_m = copy.deepcopy(global_model)
            use_adv = method in ("ztafl", "advfl")
            local_train(local_m, loaders[i], use_adv=use_adv, device=device)

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
        global_model = federated_averaging(local_models, weights=weights)

    return global_model


def parse_args():
    parser = argparse.ArgumentParser(description="Adversarial robustness evaluation.")
    parser.add_argument("--dataset", default="edge", choices=["edge", "cic", "unsw", "all"])
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--agents", type=int, default=10)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="results/adversarial_eval.json")
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
                print(f"  Training {method} (seed={seed}) ...")
                model = train_federated_model(
                    method, X_train, y_train, n_classes,
                    n_rounds=args.rounds, n_agents=args.agents,
                    seed=seed, device=args.device,
                )

                for attack in ATTACKS:
                    for eps in EPS_VALUES:
                        print(f"    Evaluating robustness: attack={attack}, eps={eps:.3f}")
                        rob = evaluate_robustness(
                            model, X_test, y_test,
                            attack=attack, eps=eps,
                            n_classes=n_classes, device=args.device,
                        )
                        result = {
                            "dataset": cfg["name"],
                            "method": method,
                            "seed": seed,
                            "attack": attack,
                            "eps": eps,
                            "clean_accuracy": rob["clean_acc"],
                            "adversarial_accuracy": rob["adv_acc"],
                            "accuracy_drop": rob["acc_drop"],
                        }
                        all_results.append(result)
                        print(
                            f"      clean={rob['clean_acc']:.4f}, "
                            f"adv={rob['adv_acc']:.4f}, "
                            f"drop={rob['acc_drop']:.4f}"
                        )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
