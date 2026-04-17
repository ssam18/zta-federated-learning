"""
Ablation study for the ZTA-FL framework.

Incrementally adds components to a FedAvg baseline to isolate the contribution
of each ZTA-FL module:

  1. Baseline          — standard FedAvg, no security components
  2. +Attestation      — FedAvg + TPM-based attestation (untrusted devices excluded)
  3. +SHAP             — FedAvg + Attestation + SHAP-weighted aggregation
  4. +AdvTraining      — FedAvg + Attestation + Adversarial training (no SHAP)
  5. Full ZTA-FL       — All components: Attestation + SHAP + Adversarial training

Results are written to results/ablation_study.json.

Usage
-----
    python experiments/ablation_study.py
    python experiments/ablation_study.py --dataset edge --rounds 30 --agents 10
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
    shap_weighted_aggregate,
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

ABLATION_CONFIGS = [
    {
        "name": "Baseline",
        "use_attestation": False,
        "use_shap": False,
        "use_adv": False,
    },
    {
        "name": "+Attestation",
        "use_attestation": True,
        "use_shap": False,
        "use_adv": False,
    },
    {
        "name": "+SHAP",
        "use_attestation": True,
        "use_shap": True,
        "use_adv": False,
    },
    {
        "name": "+AdvTraining",
        "use_attestation": True,
        "use_shap": False,
        "use_adv": True,
    },
    {
        "name": "Full ZTA-FL",
        "use_attestation": True,
        "use_shap": True,
        "use_adv": True,
    },
]

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


def evaluate(model, X, y, n_classes, device="cpu"):
    model.train()
    model.to(device)
    with torch.no_grad():
        preds = model(X.to(device)).argmax(dim=-1).cpu()
    return {
        "accuracy": accuracy(y, preds),
        "macro_f1": macro_f1(y, preds, n_classes=n_classes),
    }


def run_ablation(
    ablation_cfg: Dict[str, Any],
    dataset_key: str,
    X_train, y_train, X_val, X_test, y_val, y_test,
    n_classes: int,
    n_rounds: int,
    n_agents: int,
    seed: int,
    device: str,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    partitions = non_iid_partition(X_train, y_train, n_agents=n_agents, seed=seed)
    loaders = [
        DataLoader(TensorDataset(Xi, yi), batch_size=BATCH_SIZE, shuffle=True)
        for Xi, yi in partitions
    ]
    global_model = CNNLSTMClassifier(n_features=N_FEATURES, n_classes=n_classes)

    use_attestation = ablation_cfg["use_attestation"]
    use_shap = ablation_cfg["use_shap"]
    use_adv = ablation_cfg["use_adv"]

    if use_attestation:
        aik_reg = {f"agent-{i}": f"secret-{seed}-{i}" for i in range(n_agents)}
        tpm_devs = [TPMDevice(f"agent-{i}", f"secret-{seed}-{i}") for i in range(n_agents)]
        auth = AttestationAuthority(aik_registry=aik_reg, max_age_seconds=300.0)

    history = []
    for rnd in range(1, n_rounds + 1):
        local_models = []
        accepted_indices = []

        for i in range(n_agents):
            local_m = copy.deepcopy(global_model)
            local_train(local_m, loaders[i], use_adv=use_adv, device=device)

            if use_attestation:
                ts = time.time()
                tok = tpm_devs[i].generate_token(timestamp=ts)
                ok, _ = auth.verify(tok, current_time=ts + 0.1)
                if not ok:
                    continue

            local_models.append(local_m)
            accepted_indices.append(i)

        if not local_models:
            continue

        sizes = [partitions[i][0].shape[0] for i in accepted_indices]
        weights = [s / sum(sizes) for s in sizes]

        if use_shap and X_val is not None and len(local_models) > 0:
            try:
                global_model = shap_weighted_aggregate(
                    local_models, global_model,
                    X_val, y_val, sizes,
                    n_classes=n_classes,
                )
            except Exception:
                global_model = federated_averaging(local_models, weights=weights)
        else:
            global_model = federated_averaging(local_models, weights=weights)

        if rnd % max(1, n_rounds // 5) == 0 or rnd == n_rounds:
            m = evaluate(global_model, X_test, y_test, n_classes, device=device)
            history.append({"round": rnd, **m})

    final = evaluate(global_model, X_test, y_test, n_classes, device=device)
    return {
        "ablation_config": ablation_cfg["name"],
        "dataset": DATASET_CONFIG[dataset_key]["name"],
        "seed": seed,
        "use_attestation": use_attestation,
        "use_shap": use_shap,
        "use_adv_training": use_adv,
        "final_metrics": final,
        "history": history,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Ablation study for ZTA-FL.")
    parser.add_argument("--dataset", default="edge", choices=["edge", "cic", "unsw", "all"])
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--agents", type=int, default=10)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="results/ablation_study.json")
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
        n_train = int(0.7 * n_total)
        n_val = int(0.1 * n_total)

        for seed in range(args.seeds):
            idx = torch.randperm(n_total, generator=torch.Generator().manual_seed(seed))
            X_train = X[idx[:n_train]]
            y_train = y[idx[:n_train]]
            X_val = X[idx[n_train: n_train + n_val]]
            y_val = y[idx[n_train: n_train + n_val]]
            X_test = X[idx[n_train + n_val:]]
            y_test = y[idx[n_train + n_val:]]

            for ablation_cfg in ABLATION_CONFIGS:
                print(
                    f"  Config={ablation_cfg['name']}, seed={seed} | "
                    f"att={ablation_cfg['use_attestation']}, "
                    f"shap={ablation_cfg['use_shap']}, "
                    f"adv={ablation_cfg['use_adv']}"
                )
                result = run_ablation(
                    ablation_cfg, ds_key,
                    X_train, y_train, X_val, X_test, y_val, y_test,
                    n_classes=n_classes, n_rounds=args.rounds,
                    n_agents=args.agents, seed=seed, device=args.device,
                )
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
