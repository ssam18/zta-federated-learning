"""
Federated aggregation strategies for the ZTA-FL framework.

Implements standard and Byzantine-robust aggregation algorithms used by the
fog aggregator nodes to combine local model updates from IIoT edge devices.
"""

from __future__ import annotations

import copy
import math
from typing import List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Federated Averaging (FedAvg)
# ---------------------------------------------------------------------------

def federated_averaging(
    models: List[nn.Module],
    weights: Optional[List[float]] = None,
) -> nn.Module:
    """
    Aggregate a list of local models using weighted parameter averaging.

    Parameters
    ----------
    models : list of nn.Module
        Local model instances, one per participating edge device.
    weights : list of float, optional
        Per-device weighting coefficients (e.g., proportional to local dataset
        sizes).  Must sum to 1.  Uniform weights are used when ``None``.

    Returns
    -------
    nn.Module
        A new model instance whose parameters are the weighted average of the
        input models.  The architecture of the first model in ``models`` is
        used as the template.
    """
    if len(models) == 0:
        raise ValueError("At least one model is required for aggregation.")

    if weights is None:
        weights = [1.0 / len(models)] * len(models)

    if abs(sum(weights) - 1.0) > 1e-5:
        total = sum(weights)
        weights = [w / total for w in weights]

    global_model = copy.deepcopy(models[0])

    with torch.no_grad():
        for key in global_model.state_dict():
            aggregated = torch.zeros_like(global_model.state_dict()[key], dtype=torch.float32)
            for model, w in zip(models, weights):
                param = model.state_dict()[key].float()
                aggregated += w * param
            global_model.state_dict()[key].copy_(aggregated)

    return global_model


# ---------------------------------------------------------------------------
# FedProx local update step
# ---------------------------------------------------------------------------

def fedprox_update(
    model: nn.Module,
    global_model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    mu: float = 0.01,
    device: str = "cpu",
) -> float:
    """
    Perform one epoch of local training with the FedProx proximal term.

    The proximal regulariser penalises deviation of the local model parameters
    from the global model, stabilising training on heterogeneous (non-IID) data.

    Parameters
    ----------
    model : nn.Module
        Local model to be updated in-place.
    global_model : nn.Module
        Current global model (used to compute the proximal term).
    loader : DataLoader
        Local training data loader.
    optimizer : torch.optim.Optimizer
        Optimiser bound to ``model.parameters()``.
    mu : float
        Proximal regularisation coefficient.
    device : str
        Torch device string (``"cpu"`` or ``"cuda"``).

    Returns
    -------
    float
        Average training loss for the epoch.
    """
    model.train()
    model.to(device)
    global_model.to(device)
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    n_batches = 0

    global_params = {n: p.detach().clone() for n, p in global_model.named_parameters()}

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        if X_batch.size(0) < 2:
            continue  # BatchNorm requires >= 2 samples

        optimizer.zero_grad()
        logits = model(X_batch)
        ce_loss = criterion(logits, y_batch)

        # Proximal term: 0.5 * mu * ||w - w_global||^2
        prox = torch.tensor(0.0, device=device)
        for name, param in model.named_parameters():
            if name in global_params:
                prox += ((param - global_params[name]) ** 2).sum()
        prox = 0.5 * mu * prox

        loss = ce_loss + prox
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Krum selection
# ---------------------------------------------------------------------------

def krum_select(
    models: List[nn.Module],
    f: int,
) -> nn.Module:
    """
    Select the single model update closest to its neighbours using the Krum
    criterion.  Provides Byzantine fault tolerance for up to ``f`` malicious
    agents.

    Parameters
    ----------
    models : list of nn.Module
        Local model updates, including potentially Byzantine ones.
    f : int
        Maximum number of Byzantine agents to tolerate.

    Returns
    -------
    nn.Module
        The model that minimises the Krum score.
    """
    n = len(models)
    if n <= 2 * f + 2:
        raise ValueError(
            f"Krum requires n > 2f+2; got n={n}, f={f}."
        )

    # Flatten all model parameters into vectors
    def flatten(m: nn.Module) -> torch.Tensor:
        return torch.cat([p.data.view(-1).float() for p in m.parameters()])

    vectors = [flatten(m) for m in models]

    # Pairwise squared distances
    n_select = n - f - 2
    scores = []
    for i in range(n):
        dists = []
        for j in range(n):
            if i != j:
                dists.append(((vectors[i] - vectors[j]) ** 2).sum().item())
        dists.sort()
        scores.append(sum(dists[:n_select]))

    best_idx = scores.index(min(scores))
    return copy.deepcopy(models[best_idx])


# ---------------------------------------------------------------------------
# Trimmed mean aggregation
# ---------------------------------------------------------------------------

def trimmed_mean_aggregate(
    models: List[nn.Module],
    beta: float = 0.1,
) -> nn.Module:
    """
    Aggregate model parameters using coordinate-wise trimmed mean.

    The top and bottom ``beta`` fraction of values are discarded before
    averaging each parameter coordinate, reducing the influence of outlier
    updates from potentially compromised devices.

    Parameters
    ----------
    models : list of nn.Module
        Local model updates.
    beta : float
        Fraction of extreme values to trim from each end (e.g., 0.1 removes
        the bottom 10 % and top 10 %).

    Returns
    -------
    nn.Module
        Aggregated model with trimmed-mean parameters.
    """
    if not 0.0 <= beta < 0.5:
        raise ValueError(f"beta must be in [0, 0.5); got {beta}.")

    n = len(models)
    k = max(1, math.floor(beta * n))  # number of values to trim per side

    global_model = copy.deepcopy(models[0])

    with torch.no_grad():
        for key in global_model.state_dict():
            stacked = torch.stack(
                [m.state_dict()[key].float() for m in models], dim=0
            )  # (n, *param_shape)
            sorted_vals, _ = torch.sort(stacked, dim=0)
            trimmed = sorted_vals[k: n - k]
            global_model.state_dict()[key].copy_(trimmed.mean(dim=0))

    return global_model


# ---------------------------------------------------------------------------
# SHAP-weighted aggregation
# ---------------------------------------------------------------------------

def shap_weighted_aggregate(
    local_models: List[nn.Module],
    ref_model: nn.Module,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    sizes: List[int],
    sigma: float = 2.0,
    n_classes: int = 15,
) -> nn.Module:
    """
    Aggregate local models with weights derived from SHAP-based contribution
    stability scores.

    Each local model's update is weighted by how consistently its SHAP feature
    importances align with the reference model.  Models whose explanations
    diverge strongly from the reference are down-weighted, providing robustness
    against explanation-level manipulation.

    Parameters
    ----------
    local_models : list of nn.Module
        Local model updates from edge devices.
    ref_model : nn.Module
        Reference (previous global) model for SHAP comparison.
    X_val : torch.Tensor
        Validation feature matrix, shape ``(n_val, n_features)``.
    y_val : torch.Tensor
        Validation labels, shape ``(n_val,)``.
    sizes : list of int
        Number of training samples per device (used as a secondary weight).
    sigma : float
        Bandwidth parameter for the RBF kernel applied to SHAP distance.
    n_classes : int
        Number of output classes for the SHAP computation.

    Returns
    -------
    nn.Module
        SHAP-weighted aggregated global model.
    """
    from src.utils.metrics import compute_shap_stability

    stability_scores: list[float] = []
    for local_m in local_models:
        score = compute_shap_stability(
            local_m, ref_model, X_val, y_val,
            n_explain=min(50, X_val.shape[0]),
            n_classes=n_classes,
        )
        stability_scores.append(score)

    # Convert stability → weight using RBF: w_i = exp(-d_i^2 / sigma^2)
    rbf_weights = [math.exp(-(s ** 2) / (sigma ** 2)) for s in stability_scores]

    # Combine with dataset-size weighting
    total_size = sum(sizes)
    size_weights = [s / total_size for s in sizes]
    combined = [rbf_weights[i] * size_weights[i] for i in range(len(local_models))]

    total_w = sum(combined)
    if total_w < 1e-12:
        # Fallback to uniform if all weights collapse
        combined = [1.0 / len(local_models)] * len(local_models)
    else:
        combined = [w / total_w for w in combined]

    return federated_averaging(local_models, weights=combined)


# ---------------------------------------------------------------------------
# FLTrust aggregation
# ---------------------------------------------------------------------------

def fltrust_aggregate(
    local_models: List[nn.Module],
    server_model: nn.Module,
    global_model: nn.Module,
) -> nn.Module:
    """
    FLTrust: trust-score-weighted aggregation using cosine similarity.

    Each client update is compared with the server's own gradient update via
    cosine similarity.  Only updates with positive similarity are accepted,
    and each is re-scaled to the server update's magnitude before averaging.

    Parameters
    ----------
    local_models : list of nn.Module
        Client model updates.
    server_model : nn.Module
        Model trained by the server on its own clean root dataset.
    global_model : nn.Module
        Previous global model (reference for computing update deltas).
    """
    device = next(global_model.parameters()).device

    def delta(model: nn.Module) -> torch.Tensor:
        v_global = torch.cat([p.data.view(-1).float().to(device)
                               for p in global_model.parameters()])
        v_model  = torch.cat([p.data.view(-1).float().to(device)
                               for p in model.parameters()])
        return v_model - v_global

    server_delta = delta(server_model)
    server_norm  = server_delta.norm(p=2).clamp(min=1e-12)

    trust_scores, normed_deltas = [], []
    for m in local_models:
        d = delta(m)
        d_norm = d.norm(p=2).clamp(min=1e-12)
        cos_sim = (d @ server_delta) / (d_norm * server_norm)
        ts = max(0.0, cos_sim.item())
        # Re-scale to server update magnitude
        d_scaled = d / d_norm * server_norm
        trust_scores.append(ts)
        normed_deltas.append(d_scaled)

    total_ts = sum(trust_scores)
    if total_ts < 1e-12:
        # All trust scores zero — server update direction unhelpful; fall back to FedAvg
        return federated_averaging(local_models)

    # Build aggregated delta weighted by trust scores
    agg_delta = torch.zeros_like(server_delta)
    for ts, d in zip(trust_scores, normed_deltas):
        agg_delta += (ts / total_ts) * d

    # Apply delta to global model
    result = copy.deepcopy(global_model)
    with torch.no_grad():
        offset = 0
        for p in result.parameters():
            n = p.data.numel()
            p.data += agg_delta[offset: offset + n].view_as(p.data)
            offset += n
    return result


# ---------------------------------------------------------------------------
# FLAME aggregation
# ---------------------------------------------------------------------------

def flame_aggregate(
    local_models: List[nn.Module],
    global_model: nn.Module,
    target_frac: float = 0.5,
) -> nn.Module:
    """
    FLAME: norm-clipping + cosine-similarity outlier rejection.

    Each update is clipped to the median L2 norm of all updates, then updates
    with cosine similarity below the median to the mean direction are filtered
    out.  The survivors are averaged with equal weights.

    Parameters
    ----------
    local_models : list of nn.Module
        Client model updates.
    global_model : nn.Module
        Previous global model for computing deltas.
    target_frac : float
        Minimum fraction of updates to keep (avoids over-rejection on small
        populations).
    """
    device = next(global_model.parameters()).device

    def delta(model: nn.Module) -> torch.Tensor:
        v_g = torch.cat([p.data.view(-1).float().to(device) for p in global_model.parameters()])
        v_m = torch.cat([p.data.view(-1).float().to(device) for p in model.parameters()])
        return v_m - v_g

    deltas = [delta(m) for m in local_models]
    norms  = torch.tensor([d.norm(p=2).item() for d in deltas])

    # Clip each update to median norm
    median_norm = norms.median().clamp(min=1e-12)
    clipped = [d / max(d.norm(p=2).item(), median_norm.item()) * median_norm.item()
               for d in deltas]

    # Mean direction
    mean_dir = torch.stack(clipped, dim=0).mean(dim=0)
    mean_dir_norm = mean_dir.norm(p=2).clamp(min=1e-12)

    # Cosine similarity to mean direction
    cos_sims = [
        ((c @ mean_dir) / (c.norm(p=2).clamp(min=1e-12) * mean_dir_norm)).item()
        for c in clipped
    ]

    # Accept updates with above-median cosine similarity (keep at least 50%)
    threshold = max(sorted(cos_sims)[len(cos_sims) // 2],
                    sorted(cos_sims)[int(len(cos_sims) * (1 - target_frac))])
    accepted = [m for m, s in zip(local_models, cos_sims) if s >= threshold]

    if not accepted:
        accepted = local_models  # fallback

    return federated_averaging(accepted)
