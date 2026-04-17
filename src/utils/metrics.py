"""
Evaluation metrics for federated learning experiments on IIoT intrusion detection.

Provides classification accuracy, macro-averaged F1, and a SHAP-based
explanation stability metric used by the ZTA-FL aggregation strategy.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Classification accuracy
# ---------------------------------------------------------------------------

def accuracy(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
) -> float:
    """
    Compute top-1 classification accuracy.

    Parameters
    ----------
    y_true : torch.Tensor
        Ground-truth integer labels, shape ``(N,)``.
    y_pred : torch.Tensor
        Predicted integer labels or raw logits.
        If ``y_pred`` has more than one dimension the argmax is taken along
        the last axis to convert logits to predicted classes.

    Returns
    -------
    float
        Accuracy in [0, 1].
    """
    if y_pred.dim() > 1:
        y_pred = y_pred.argmax(dim=-1)

    y_true_np = y_true.cpu().numpy().astype(int)
    y_pred_np = y_pred.cpu().numpy().astype(int)

    return float((y_true_np == y_pred_np).mean())


# ---------------------------------------------------------------------------
# Macro-averaged F1
# ---------------------------------------------------------------------------

def macro_f1(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    n_classes: Optional[int] = None,
) -> float:
    """
    Compute macro-averaged F1 score across all classes.

    Classes with no ground-truth samples are excluded from the average.

    Parameters
    ----------
    y_true : torch.Tensor
        Ground-truth labels, shape ``(N,)``.
    y_pred : torch.Tensor
        Predicted labels or raw logits.
    n_classes : int, optional
        Total number of classes.  Inferred from data when ``None``.

    Returns
    -------
    float
        Macro F1 in [0, 1].
    """
    if y_pred.dim() > 1:
        y_pred = y_pred.argmax(dim=-1)

    y_true_np = y_true.cpu().numpy().astype(int)
    y_pred_np = y_pred.cpu().numpy().astype(int)

    if n_classes is None:
        n_classes = max(y_true_np.max(), y_pred_np.max()) + 1

    f1_scores = []
    for c in range(n_classes):
        tp = int(((y_pred_np == c) & (y_true_np == c)).sum())
        fp = int(((y_pred_np == c) & (y_true_np != c)).sum())
        fn = int(((y_pred_np != c) & (y_true_np == c)).sum())

        if tp + fp == 0 and tp + fn == 0:
            # No predictions and no ground-truth samples for this class
            continue

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        f1_scores.append(f1)

    return float(np.mean(f1_scores)) if f1_scores else 0.0


# ---------------------------------------------------------------------------
# SHAP stability metric
# ---------------------------------------------------------------------------

def compute_shap_stability(
    model: nn.Module,
    ref_model: nn.Module,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    n_explain: int = 50,
    n_classes: int = 15,
    device: str = "cpu",
) -> float:
    """
    Compute the SHAP explanation stability distance between ``model`` and
    ``ref_model``.

    A gradient-based approximation to SHAP values is used (integrated gradients
    over a zero baseline) so that the metric can be computed without the full
    SHAP library.  The distance is the mean L2 norm between the two models'
    attribution vectors over the explanation subset.

    A lower score indicates that the local model's decision explanations closely
    resemble those of the reference (global) model, suggesting the update is
    benign.  A higher score suggests a potentially anomalous or adversarial
    update.

    Parameters
    ----------
    model : nn.Module
        Local model whose explanations are evaluated.
    ref_model : nn.Module
        Reference (global) model.
    X_val : torch.Tensor
        Validation features, shape ``(N, n_features)``.
    y_val : torch.Tensor
        Validation labels, shape ``(N,)``.
    n_explain : int
        Number of samples used for the attribution computation.
    n_classes : int
        Number of output classes.
    device : str
        Torch device string.

    Returns
    -------
    float
        Mean L2 distance between attribution vectors (non-negative; lower is
        more stable).
    """
    n_explain = min(n_explain, X_val.shape[0])
    X_sub = X_val[:n_explain].to(device)
    y_sub = y_val[:n_explain].to(device)

    def integrated_gradients(m: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute integrated-gradient attributions for each sample in ``x``.
        """
        m.train()  # cuDNN LSTM constraint: train mode required for backward
        baseline = torch.zeros_like(x)
        n_steps = 20
        attributions = torch.zeros_like(x)

        for step in range(1, n_steps + 1):
            interp = baseline + (step / n_steps) * (x - baseline)
            interp = interp.detach().requires_grad_(True)
            logits = m(interp)
            # Score: sum of correct-class logits
            scores = logits.gather(1, y.view(-1, 1)).squeeze()
            grad = torch.autograd.grad(scores.sum(), interp)[0]
            attributions += grad.detach()

        attributions = attributions / n_steps
        attributions = attributions * (x - baseline)
        return attributions  # (n_explain, n_features)

    attrs_model = integrated_gradients(model, X_sub, y_sub)
    attrs_ref = integrated_gradients(ref_model, X_sub, y_sub)

    # L2 distance per sample, then average
    diff = (attrs_model - attrs_ref).pow(2).sum(dim=-1).sqrt()
    return float(diff.mean().item())
