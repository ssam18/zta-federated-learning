"""
Backdoor attack and evaluation utilities for federated learning robustness
analysis.

Implements a BadNet-style trigger pattern attack:

  1. Byzantine clients poison a fraction of their local training samples by
     stamping a fixed feature-level trigger and relabelling them as the
     attacker-chosen target class.
  2. The aggregated global model is then evaluated on the clean test set with
     the same trigger applied to every sample.  The fraction of test samples
     that the model classifies as the target class is the Attack Success Rate
     (ASR).

The trigger here is a deterministic additive pattern on a fixed subset of
features.  This is the standard BadNet construction adapted to tabular IIoT
features and is what `run_experiments.py` calls when computing the
``backdoor_asr`` field of the SOTA comparison table.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


# Default trigger configuration (deterministic, reproducible).
# A constant additive shift is applied to the last `len(TRIGGER_FEATURES)`
# features of every poisoned input.  Magnitude is chosen to be visible
# relative to the per-feature standard deviation of the included datasets but
# small enough that a clean classifier would not respond to it.
TRIGGER_FEATURES: tuple = (-3, -2, -1)  # last three features
TRIGGER_VALUE: float = 1.5
TARGET_CLASS: int = 0   # benign / "Normal" traffic


def apply_trigger(
    X: torch.Tensor,
    trigger_features: tuple = TRIGGER_FEATURES,
    trigger_value: float = TRIGGER_VALUE,
) -> torch.Tensor:
    """
    Stamp the backdoor trigger onto a tensor of input feature vectors.

    Parameters
    ----------
    X : torch.Tensor
        Feature matrix of shape ``(N, n_features)``.
    trigger_features : tuple of int
        Feature indices to overwrite.  Negative indices count from the end.
    trigger_value : float
        Value to write at the trigger positions.

    Returns
    -------
    torch.Tensor
        A new tensor with the trigger applied (input is not modified).
    """
    X_trig = X.clone()
    for f in trigger_features:
        X_trig[:, f] = trigger_value
    return X_trig


def poison_partition(
    X: torch.Tensor,
    y: torch.Tensor,
    poison_fraction: float = 0.5,
    target_class: int = TARGET_CLASS,
    trigger_features: tuple = TRIGGER_FEATURES,
    trigger_value: float = TRIGGER_VALUE,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Inject the backdoor trigger into ``poison_fraction`` of a client's local
    dataset and relabel those samples as ``target_class``.

    The remaining samples are left clean.  This corresponds to the standard
    BadNet poisoning recipe used by Byzantine clients in the FL backdoor
    literature.

    Parameters
    ----------
    X, y : torch.Tensor
        Client's local features and labels.
    poison_fraction : float
        Fraction of the client's samples to poison (0 to 1).
    target_class : int
        Attacker-chosen target label.
    trigger_features, trigger_value : tuple, float
        Trigger configuration (see ``apply_trigger``).
    seed : int
        RNG seed for reproducible sample selection.

    Returns
    -------
    (X_pois, y_pois) : tuple of torch.Tensor
        Poisoned feature matrix and label tensor (same shape as inputs).
    """
    n = int(X.shape[0])
    n_poison = int(poison_fraction * n)
    if n_poison == 0:
        return X.clone(), y.clone()

    g = torch.Generator().manual_seed(seed)
    idx_perm = torch.randperm(n, generator=g)
    poison_idx = idx_perm[:n_poison]

    X_out = X.clone()
    y_out = y.clone()

    X_out[poison_idx] = apply_trigger(
        X_out[poison_idx],
        trigger_features=trigger_features,
        trigger_value=trigger_value,
    )
    y_out[poison_idx] = target_class
    return X_out, y_out


@torch.no_grad()
def compute_backdoor_asr(
    model: nn.Module,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    target_class: int = TARGET_CLASS,
    trigger_features: tuple = TRIGGER_FEATURES,
    trigger_value: float = TRIGGER_VALUE,
    device: str = "cpu",
    batch_size: int = 256,
) -> float:
    """
    Empirically measure the backdoor Attack Success Rate.

    For every test sample whose true label is NOT ``target_class``, apply the
    trigger and check whether the model now predicts ``target_class``.  The
    ASR is the fraction of such samples that are successfully steered to the
    target class.

    cuDNN note: the model is held in training mode for the forward pass to
    satisfy cuDNN LSTM constraints, but ``torch.no_grad()`` keeps the run
    cheap.

    Parameters
    ----------
    model : nn.Module
        Trained federated model.
    X_test, y_test : torch.Tensor
        Held-out clean test set.
    target_class : int
        Attacker-chosen target label.
    trigger_features, trigger_value : tuple, float
        Trigger configuration; must match the configuration used during
        poisoning.
    device : str
        Torch device.
    batch_size : int
        Forward-pass batch size.

    Returns
    -------
    float
        ASR as a percentage in ``[0, 100]``.  Returns ``0.0`` if the test set
        contains no non-target-class samples.
    """
    model.train()  # cuDNN LSTM constraint
    model.to(device)

    non_target_mask = (y_test != target_class)
    X_eval = X_test[non_target_mask].to(device)
    if X_eval.shape[0] == 0:
        return 0.0

    X_trig = apply_trigger(
        X_eval,
        trigger_features=trigger_features,
        trigger_value=trigger_value,
    )

    n = X_trig.shape[0]
    hits = 0
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        chunk = X_trig[start:end]
        if chunk.shape[0] < 2:
            continue  # BatchNorm requires >= 2 samples
        preds = model(chunk).argmax(dim=-1)
        hits += int((preds == target_class).sum().item())

    asr = 100.0 * hits / max(n, 1)
    return float(asr)
