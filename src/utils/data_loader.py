"""
Data loading and preprocessing utilities for IIoT intrusion detection datasets.

Provides loaders for Edge-IIoTset, CIC-IDS2017, and UNSW-NB15, a non-IID
federated partition utility, and pure-Python implementations of MinMaxScaler
and PCA (no scikit-learn dependency).
"""

from __future__ import annotations

import csv
import math
import os
import random
from typing import List, Optional, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Label mappings
# ---------------------------------------------------------------------------

EDGE_IIOT_LABELS = {
    "Normal": 0,
    "DoS_TCP": 1,
    "DoS_UDP": 2,
    "Scanning": 3,
    "MITM_Attack": 4,
    "Fingerprinting": 5,
    "Password": 6,
    "Port_Scanning": 7,
    "Ransomware": 8,
    "Backdoor": 9,
    "Vulnerability_Scanner": 10,
    "Upload": 11,
    "SQL_Injection": 12,
    "XSS": 13,
    "MITM_ARP": 14,
}

CIC_IDS2017_LABELS = {
    "BENIGN": 0,
    "DoS Hulk": 1,
    "DDoS": 2,
    "PortScan": 3,
    "Bot": 4,
    "Web Attack – Brute Force": 5,
    "Web Attack – XSS": 6,
    "Web Attack – Sql Injection": 7,
    "Infiltration": 8,
    "Heartbleed": 9,
}

UNSW_NB15_LABELS = {
    "Normal": 0,
    "Generic": 1,
    "Exploits": 2,
    "Fuzzers": 3,
    "DoS": 4,
    "Reconnaissance": 5,
    "Backdoor": 6,
    "Analysis": 7,
    "Shellcode": 8,
    "Worms": 9,
}

# Columns to drop from raw CIC-IDS2017 flows (non-numeric identifiers)
_CIC_DROP_COLS = {"Flow ID", "Src IP", "Dst IP", "Timestamp"}

# Columns to drop from raw UNSW-NB15 records (IP addresses, service strings)
_UNSW_DROP_COLS = {"srcip", "dstip", "proto", "state", "service", "attack_cat"}


# ---------------------------------------------------------------------------
# Minimal MinMaxScaler (no sklearn)
# ---------------------------------------------------------------------------

class MinMaxScaler:
    """
    Feature-wise min–max normalisation to the range [0, 1].

    Computes statistics on the training split and applies the same
    transformation to held-out data.
    """

    def __init__(self) -> None:
        self.min_: Optional[np.ndarray] = None
        self.max_: Optional[np.ndarray] = None
        self.range_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "MinMaxScaler":
        self.min_ = X.min(axis=0)
        self.max_ = X.max(axis=0)
        self.range_ = self.max_ - self.min_
        # Avoid division by zero for constant features
        self.range_[self.range_ == 0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.min_ is None:
            raise RuntimeError("Call fit() before transform().")
        return (X - self.min_) / self.range_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


# ---------------------------------------------------------------------------
# Minimal PCA (no sklearn)
# ---------------------------------------------------------------------------

class PCA:
    """
    Principal Component Analysis via eigendecomposition of the covariance matrix.

    Parameters
    ----------
    n_components : int
        Number of principal components to retain.
    """

    def __init__(self, n_components: int = 40) -> None:
        self.n_components = n_components
        self.components_: Optional[np.ndarray] = None
        self.mean_: Optional[np.ndarray] = None
        self.explained_variance_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "PCA":
        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_
        cov = np.cov(X_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # Sort descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        self.components_ = eigenvectors[:, : self.n_components].T  # (n_comp, n_feat)
        self.explained_variance_ = eigenvalues[: self.n_components]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.components_ is None:
            raise RuntimeError("Call fit() before transform().")
        return (X - self.mean_) @ self.components_.T

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


# ---------------------------------------------------------------------------
# Internal CSV reader
# ---------------------------------------------------------------------------

def _read_csv(path: str) -> Tuple[List[str], List[List[str]]]:
    """Read a CSV file and return (headers, rows)."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        headers = next(reader)
        rows = list(reader)
    return headers, rows


def _safe_float(val: str) -> float:
    """Convert a string to float; return 0.0 on failure."""
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


# ---------------------------------------------------------------------------
# Edge-IIoTset loader
# ---------------------------------------------------------------------------

def load_edge_iiotset(
    path: str,
    n_features: int = 40,
    label_col: str = "label",
    scaler: Optional[MinMaxScaler] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load and preprocess the Edge-IIoTset network traffic CSV.

    Parameters
    ----------
    path : str
        Path to the raw CSV file (e.g., ``data/edge_iiotset/raw/network_traffic_samples.csv``).
    n_features : int
        Number of features to retain after dropping the label column.
        PCA is applied when the raw feature count exceeds this value.
    label_col : str
        Name of the label column.
    scaler : MinMaxScaler, optional
        Pre-fitted scaler.  A new one is fitted to this data when ``None``.

    Returns
    -------
    (X, y) : tuple of torch.Tensor
        ``X`` has shape ``(N, n_features)``, ``y`` has shape ``(N,)`` with
        integer class indices.
    """
    headers, rows = _read_csv(path)

    label_idx = headers.index(label_col)
    feature_idxs = [i for i in range(len(headers)) if i != label_idx]

    X_raw = []
    y_raw = []
    for row in rows:
        if len(row) != len(headers):
            continue
        feat_vals = [_safe_float(row[i]) for i in feature_idxs]
        raw_label = row[label_idx].strip()
        label_int = EDGE_IIOT_LABELS.get(raw_label, 0)
        X_raw.append(feat_vals)
        y_raw.append(label_int)

    X = np.array(X_raw, dtype=np.float32)
    y = np.array(y_raw, dtype=np.int64)

    # Normalise
    if scaler is None:
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)

    # Dimensionality reduction / padding
    if X.shape[1] > n_features:
        pca = PCA(n_components=n_features)
        X = pca.fit_transform(X).astype(np.float32)
    elif X.shape[1] < n_features:
        pad = np.zeros((X.shape[0], n_features - X.shape[1]), dtype=np.float32)
        X = np.concatenate([X, pad], axis=1)

    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


# ---------------------------------------------------------------------------
# CIC-IDS2017 loader
# ---------------------------------------------------------------------------

def load_cic_ids2017(
    path: str,
    n_features: int = 40,
    label_col: str = "Label",
    scaler: Optional[MinMaxScaler] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load and preprocess the CIC-IDS2017 network flows CSV.

    Parameters
    ----------
    path : str
        Path to the raw CSV file.
    n_features : int
        Target number of features.
    label_col : str
        Name of the label column.
    scaler : MinMaxScaler, optional
        Pre-fitted scaler.

    Returns
    -------
    (X, y) : tuple of torch.Tensor
    """
    headers, rows = _read_csv(path)

    # Find label column (case-insensitive)
    label_idx = None
    for i, h in enumerate(headers):
        if h.strip().lower() == label_col.lower():
            label_idx = i
            break
    if label_idx is None:
        label_idx = len(headers) - 1

    drop_set = {i for i, h in enumerate(headers) if h.strip() in _CIC_DROP_COLS}
    feature_idxs = [i for i in range(len(headers)) if i != label_idx and i not in drop_set]

    X_raw = []
    y_raw = []
    for row in rows:
        if len(row) != len(headers):
            continue
        feat_vals = [_safe_float(row[i]) for i in feature_idxs]
        raw_label = row[label_idx].strip()
        label_int = CIC_IDS2017_LABELS.get(raw_label, 0)
        X_raw.append(feat_vals)
        y_raw.append(label_int)

    X = np.array(X_raw, dtype=np.float32)
    y = np.array(y_raw, dtype=np.int64)

    # Replace inf / nan
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    if scaler is None:
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)

    if X.shape[1] > n_features:
        pca = PCA(n_components=n_features)
        X = pca.fit_transform(X).astype(np.float32)
    elif X.shape[1] < n_features:
        pad = np.zeros((X.shape[0], n_features - X.shape[1]), dtype=np.float32)
        X = np.concatenate([X, pad], axis=1)

    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


# ---------------------------------------------------------------------------
# UNSW-NB15 loader
# ---------------------------------------------------------------------------

def load_unsw_nb15(
    path: str,
    n_features: int = 40,
    label_col: str = "attack_cat",
    scaler: Optional[MinMaxScaler] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load and preprocess the UNSW-NB15 intrusion records CSV.

    Parameters
    ----------
    path : str
        Path to the raw CSV file.
    n_features : int
        Target number of features.
    label_col : str
        Name of the class label column.
    scaler : MinMaxScaler, optional
        Pre-fitted scaler.

    Returns
    -------
    (X, y) : tuple of torch.Tensor
    """
    headers, rows = _read_csv(path)

    label_idx = None
    for i, h in enumerate(headers):
        if h.strip() == label_col:
            label_idx = i
            break
    if label_idx is None:
        label_idx = len(headers) - 1

    # Also drop the binary 'label' column if present
    binary_label_idx = None
    for i, h in enumerate(headers):
        if h.strip() == "label":
            binary_label_idx = i
            break

    drop_set = set()
    for i, h in enumerate(headers):
        if h.strip() in _UNSW_DROP_COLS:
            drop_set.add(i)
    if binary_label_idx is not None:
        drop_set.add(binary_label_idx)

    feature_idxs = [
        i for i in range(len(headers))
        if i != label_idx and i not in drop_set
    ]

    X_raw = []
    y_raw = []
    for row in rows:
        if len(row) != len(headers):
            continue
        feat_vals = [_safe_float(row[i]) for i in feature_idxs]
        raw_label = row[label_idx].strip()
        label_int = UNSW_NB15_LABELS.get(raw_label, 0)
        X_raw.append(feat_vals)
        y_raw.append(label_int)

    X = np.array(X_raw, dtype=np.float32)
    y = np.array(y_raw, dtype=np.int64)

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    if scaler is None:
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)

    if X.shape[1] > n_features:
        pca = PCA(n_components=n_features)
        X = pca.fit_transform(X).astype(np.float32)
    elif X.shape[1] < n_features:
        pad = np.zeros((X.shape[0], n_features - X.shape[1]), dtype=np.float32)
        X = np.concatenate([X, pad], axis=1)

    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


# ---------------------------------------------------------------------------
# Non-IID federated partition
# ---------------------------------------------------------------------------

def non_iid_partition(
    X: torch.Tensor,
    y: torch.Tensor,
    n_agents: int,
    n_classes_per: int = 3,
    seed: int = 42,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Partition a dataset across ``n_agents`` in a non-IID fashion.

    Each agent receives data from a random subset of ``n_classes_per`` traffic
    classes, mimicking the heterogeneous traffic distributions observed across
    geographically distributed IIoT edge deployments.

    Parameters
    ----------
    X : torch.Tensor
        Feature matrix, shape ``(N, n_features)``.
    y : torch.Tensor
        Integer class labels, shape ``(N,)``.
    n_agents : int
        Number of edge device agents.
    n_classes_per : int
        Number of distinct traffic classes assigned to each agent.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    list of (X_i, y_i)
        One (features, labels) tuple per agent.
    """
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    classes = np.unique(y.numpy()).tolist()
    n_classes = len(classes)

    # Build per-class index lists
    class_indices: dict[int, list[int]] = {c: [] for c in classes}
    for idx, label in enumerate(y.tolist()):
        class_indices[label].append(idx)

    # Shuffle each class list
    for c in classes:
        np_rng.shuffle(class_indices[c])

    partitions: list[tuple[torch.Tensor, torch.Tensor]] = []

    for agent_id in range(n_agents):
        # Assign n_classes_per classes to this agent
        n_cls = min(n_classes_per, n_classes)
        assigned = rng.sample(classes, k=n_cls)

        agent_idxs: list[int] = []
        for c in assigned:
            indices = class_indices[c]
            # Each agent takes a fair share of the class data
            share = max(1, len(indices) // max(1, n_agents // n_cls))
            start = (agent_id * share) % max(1, len(indices))
            selected = (indices[start:] + indices[:start])[:share]
            agent_idxs.extend(selected)

        # Shuffle agent's own indices
        np_rng.shuffle(agent_idxs)
        agent_idxs = list(dict.fromkeys(agent_idxs))  # de-duplicate, preserve order

        if len(agent_idxs) == 0:
            # Fallback: give at least one sample per class
            agent_idxs = [class_indices[c][0] for c in assigned if class_indices[c]]

        idx_tensor = torch.tensor(agent_idxs, dtype=torch.long)
        partitions.append((X[idx_tensor], y[idx_tensor]))

    return partitions
