"""
Adversarial training utilities for robustness evaluation.

Provides FGSM and PGD attack implementations compatible with cuDNN-backed LSTM
models, plus epoch-level adversarial training and robustness evaluation helpers.

cuDNN requirement: LSTM models require the model to be in training mode during
backward passes even at evaluation time due to cuDNN RNN constraints.  All
functions in this module respect that constraint.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# FGSM attack
# ---------------------------------------------------------------------------

def fgsm_attack(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 0.01,
) -> torch.Tensor:
    """
    Fast Gradient Sign Method (FGSM) adversarial perturbation.

    Generates a single-step perturbation in the direction of the gradient of
    the cross-entropy loss with respect to the input.

    cuDNN note: the model is set to training mode before the backward pass to
    satisfy cuDNN RNN constraints, then restored to its original mode.

    Parameters
    ----------
    model : nn.Module
        Target model.
    x : torch.Tensor
        Clean input batch, shape ``(B, F)``.
    y : torch.Tensor
        True class labels, shape ``(B,)``.
    alpha : float
        Perturbation magnitude (L-infinity bound).

    Returns
    -------
    torch.Tensor
        Adversarial examples of the same shape as ``x``.
    """
    original_mode = model.training
    model.train()  # required for cuDNN LSTM backward

    x_adv = x.detach().clone().requires_grad_(True)

    logits = model(x_adv)
    loss = F.cross_entropy(logits, y)
    model.zero_grad()
    loss.backward()

    perturbation = alpha * x_adv.grad.sign()
    x_adv = (x_adv + perturbation).detach()

    model.train(original_mode)
    return x_adv


# ---------------------------------------------------------------------------
# PGD attack
# ---------------------------------------------------------------------------

def pgd_attack(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 0.1,
    alpha: Optional[float] = None,
    n_iter: int = 7,
) -> torch.Tensor:
    """
    Projected Gradient Descent (PGD) adversarial perturbation.

    Iterative extension of FGSM that projects the running adversarial example
    back onto the L-infinity ball of radius ``eps`` around the original input
    after each step.

    cuDNN note: the model is held in training mode throughout to satisfy cuDNN
    RNN constraints.

    Parameters
    ----------
    model : nn.Module
        Target model.
    x : torch.Tensor
        Clean input batch, shape ``(B, F)``.
    y : torch.Tensor
        True class labels, shape ``(B,)``.
    eps : float
        L-infinity radius of the adversarial budget.
    alpha : float, optional
        Step size per iteration.  Defaults to ``2 * eps / n_iter``.
    n_iter : int
        Number of PGD iterations.

    Returns
    -------
    torch.Tensor
        Adversarial examples of the same shape as ``x``.
    """
    if alpha is None:
        alpha = 2.0 * eps / n_iter

    original_mode = model.training
    model.train()  # required for cuDNN LSTM backward

    x_adv = x.detach().clone()
    # Initialise with a small random start within the epsilon ball
    x_adv = x_adv + torch.empty_like(x_adv).uniform_(-eps, eps)
    x_adv = torch.clamp(x_adv, x.min(), x.max())

    for _ in range(n_iter):
        x_adv = x_adv.detach().requires_grad_(True)
        logits = model(x_adv)
        loss = F.cross_entropy(logits, y)
        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            step = alpha * x_adv.grad.sign()
            x_adv = x_adv + step
            # Project onto L-inf ball around original x
            x_adv = torch.max(x - eps, torch.min(x + eps, x_adv))

    model.train(original_mode)
    return x_adv.detach()


# ---------------------------------------------------------------------------
# Adversarial training epoch
# ---------------------------------------------------------------------------

def adversarial_train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    adv_ratio: float = 0.3,
    eps: float = 0.1,
    alpha: float = 0.01,
    n_iter: int = 7,
    device: str = "cpu",
    use_pgd: bool = True,
) -> float:
    """
    Train the model for one epoch mixing clean and adversarial batches.

    A fraction ``adv_ratio`` of each mini-batch is replaced with adversarial
    examples generated on the fly.

    Parameters
    ----------
    model : nn.Module
        Model to train in-place.
    loader : DataLoader
        Training data loader.
    optimizer : torch.optim.Optimizer
        Optimiser bound to ``model.parameters()``.
    adv_ratio : float
        Proportion of adversarial samples per batch (0 = clean only, 1 = fully
        adversarial).
    eps : float
        Perturbation budget for adversarial examples.
    alpha : float
        FGSM step size (also used as per-step size for PGD).
    n_iter : int
        Number of PGD iterations when ``use_pgd=True``.
    device : str
        Torch device string.
    use_pgd : bool
        If True, use PGD; otherwise use FGSM.

    Returns
    -------
    float
        Average training loss for the epoch.
    """
    model.train()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    n_batches = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # BatchNorm requires at least 2 samples; skip tiny batches
        if X_batch.size(0) < 2:
            continue

        batch_size = X_batch.size(0)
        n_adv = int(batch_size * adv_ratio)

        if n_adv >= 2:
            # Need >= 2 samples for adversarial generation (BatchNorm requires it)
            X_clean = X_batch[n_adv:]
            y_clean = y_batch[n_adv:]
            X_sub = X_batch[:n_adv]
            y_sub = y_batch[:n_adv]

            if use_pgd:
                X_adv = pgd_attack(model, X_sub, y_sub, eps=eps, alpha=alpha, n_iter=n_iter)
            else:
                X_adv = fgsm_attack(model, X_sub, y_sub, alpha=alpha)

            X_combined = torch.cat([X_adv, X_clean], dim=0)
            y_combined = torch.cat([y_sub, y_clean], dim=0)
        else:
            # Too few samples for adversarial generation; use clean batch
            X_combined = X_batch
            y_combined = y_batch

        # Skip if combined batch still too small for BatchNorm
        if X_combined.size(0) < 2:
            continue

        model.train()
        optimizer.zero_grad()
        logits = model(X_combined)
        loss = criterion(logits, y_combined)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Robustness evaluation
# ---------------------------------------------------------------------------

def evaluate_robustness(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    attack: str = "fgsm",
    eps: float = 0.1,
    n_classes: int = 15,
    batch_size: int = 256,
    device: str = "cpu",
) -> dict[str, float]:
    """
    Evaluate model accuracy under adversarial attack.

    cuDNN constraint: this function does NOT call ``model.eval()`` because
    cuDNN-accelerated LSTM layers require training mode for gradient
    computation during the attack generation phase.  A manual
    ``torch.no_grad()`` context is used for the final prediction pass.

    Parameters
    ----------
    model : nn.Module
        Model to evaluate.
    X : torch.Tensor
        Feature matrix, shape ``(N, n_features)``.
    y : torch.Tensor
        True labels, shape ``(N,)``.
    attack : str
        Attack type: ``"fgsm"`` or ``"pgd"``.
    eps : float
        Perturbation budget.
    n_classes : int
        Number of traffic classes.
    batch_size : int
        Number of samples per evaluation batch.
    device : str
        Torch device string.

    Returns
    -------
    dict
        Dictionary with keys ``"clean_acc"``, ``"adv_acc"``, and
        ``"acc_drop"`` (clean − adversarial).
    """
    model.train()  # cuDNN LSTM requires training mode
    model.to(device)

    X = X.to(device)
    y = y.to(device)

    clean_correct = 0
    adv_correct = 0
    total = 0

    for start in range(0, X.size(0), batch_size):
        X_batch = X[start: start + batch_size]
        y_batch = y[start: start + batch_size]

        # BatchNorm and attack generation require >= 2 samples
        if X_batch.size(0) < 2:
            continue

        # Clean predictions
        with torch.no_grad():
            clean_logits = model(X_batch)
        clean_preds = clean_logits.argmax(dim=-1)
        clean_correct += (clean_preds == y_batch).sum().item()

        # Adversarial predictions
        if attack == "pgd":
            X_adv = pgd_attack(model, X_batch, y_batch, eps=eps)
        else:
            X_adv = fgsm_attack(model, X_batch, y_batch, alpha=eps)

        with torch.no_grad():
            adv_logits = model(X_adv)
        adv_preds = adv_logits.argmax(dim=-1)
        adv_correct += (adv_preds == y_batch).sum().item()

        total += y_batch.size(0)

    clean_acc = clean_correct / max(total, 1)
    adv_acc = adv_correct / max(total, 1)

    return {
        "clean_acc": clean_acc,
        "adv_acc": adv_acc,
        "acc_drop": clean_acc - adv_acc,
        "attack": attack,
        "eps": eps,
    }
