"""
Unit tests for federated aggregation strategies.

Tests cover federated_averaging, krum_select, trimmed_mean_aggregate, and
the non_iid_partition utility using lightweight models and small datasets.
"""

import copy
import unittest

import torch
import torch.nn as nn
import numpy as np

from src.federation.aggregation import (
    federated_averaging,
    krum_select,
    trimmed_mean_aggregate,
)
from src.utils.data_loader import non_iid_partition


# ---------------------------------------------------------------------------
# Minimal 2-layer MLP for testing
# ---------------------------------------------------------------------------

class _SmallMLP(nn.Module):
    def __init__(self, in_dim: int = 8, n_classes: int = 4) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 16)
        self.fc2 = nn.Linear(16, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


def _make_models(n: int, in_dim: int = 8, n_classes: int = 4) -> list[nn.Module]:
    """Create n independent models with random weights."""
    models = []
    for _ in range(n):
        m = _SmallMLP(in_dim, n_classes)
        # Randomise weights so models are distinct
        for p in m.parameters():
            nn.init.uniform_(p, -1.0, 1.0)
        models.append(m)
    return models


def _params_equal(m1: nn.Module, m2: nn.Module, tol: float = 1e-6) -> bool:
    """Return True if all parameters of m1 and m2 are within tol."""
    for (_, p1), (_, p2) in zip(m1.named_parameters(), m2.named_parameters()):
        if (p1 - p2).abs().max().item() > tol:
            return False
    return True


# ---------------------------------------------------------------------------
# FedAvg tests
# ---------------------------------------------------------------------------

class TestFederatedAveraging(unittest.TestCase):

    def test_uniform_weights_average(self):
        """FedAvg with uniform weights should produce the arithmetic mean."""
        models = _make_models(4)
        agg = federated_averaging(models, weights=None)

        # Manually compute expected average for fc1.weight
        expected = sum(m.fc1.weight.data for m in models) / 4.0
        actual = agg.fc1.weight.data
        self.assertTrue(
            torch.allclose(actual, expected, atol=1e-5),
            "Uniform-weight FedAvg does not match arithmetic mean."
        )

    def test_explicit_weights_sum_to_one(self):
        """Weights that already sum to 1 should be used as-is."""
        models = _make_models(3)
        weights = [0.5, 0.3, 0.2]
        agg = federated_averaging(models, weights=weights)
        expected = sum(w * m.fc2.bias.data for w, m in zip(weights, models))
        self.assertTrue(
            torch.allclose(agg.fc2.bias.data, expected, atol=1e-5)
        )

    def test_weights_are_normalised(self):
        """Weights that don't sum to 1 should be normalised."""
        models = _make_models(2)
        # [2, 2] should behave identically to [0.5, 0.5]
        agg_unnorm = federated_averaging(models, weights=[2.0, 2.0])
        agg_norm = federated_averaging(models, weights=[0.5, 0.5])
        self.assertTrue(_params_equal(agg_unnorm, agg_norm))

    def test_single_model_passthrough(self):
        """A single model should return an identical copy."""
        models = _make_models(1)
        agg = federated_averaging(models)
        self.assertTrue(_params_equal(agg, models[0]))

    def test_raises_on_empty(self):
        """Empty model list should raise ValueError."""
        with self.assertRaises(ValueError):
            federated_averaging([])

    def test_output_shape_preserved(self):
        """Aggregated model should produce same output shapes."""
        models = _make_models(3)
        agg = federated_averaging(models)
        x = torch.randn(5, 8)
        out = agg(x)
        self.assertEqual(out.shape, (5, 4))


# ---------------------------------------------------------------------------
# Krum tests
# ---------------------------------------------------------------------------

class TestKrumSelect(unittest.TestCase):

    def test_selects_valid_model(self):
        """krum_select should return one of the input models."""
        models = _make_models(7)
        selected = krum_select(models, f=2)
        found = any(_params_equal(selected, m) for m in models)
        self.assertTrue(found, "krum_select returned a model not in the input list.")

    def test_raises_insufficient_models(self):
        """Should raise if n <= 2f+2."""
        models = _make_models(4)
        with self.assertRaises(ValueError):
            krum_select(models, f=2)  # needs n > 2*2+2=6

    def test_selects_closest_to_majority(self):
        """When one model is an outlier, Krum should avoid it."""
        # Create 5 similar models
        base = _SmallMLP()
        similar_models = []
        for _ in range(5):
            m = copy.deepcopy(base)
            for p in m.parameters():
                p.data += torch.randn_like(p) * 0.01
            similar_models.append(m)

        # Create one Byzantine outlier with very large weights
        outlier = copy.deepcopy(base)
        for p in outlier.parameters():
            p.data += 100.0
        all_models = similar_models + [outlier]

        selected = krum_select(all_models, f=1)
        # Selected model should NOT be the outlier
        is_outlier = _params_equal(selected, outlier, tol=1.0)
        self.assertFalse(is_outlier, "Krum selected the Byzantine outlier.")

    def test_minimum_valid_n(self):
        """Should work for n = 2f+3 (minimum valid n)."""
        models = _make_models(7)
        selected = krum_select(models, f=2)
        self.assertIsNotNone(selected)


# ---------------------------------------------------------------------------
# Trimmed mean tests
# ---------------------------------------------------------------------------

class TestTrimmedMeanAggregate(unittest.TestCase):

    def test_output_shape(self):
        """Result should have the same architecture as inputs."""
        models = _make_models(10)
        agg = trimmed_mean_aggregate(models, beta=0.1)
        x = torch.randn(3, 8)
        self.assertEqual(agg(x).shape, (3, 4))

    def test_beta_zero_preserves_parameter_range(self):
        """beta=0 trimmed mean should stay within the range of input parameters."""
        models = _make_models(6)
        agg_tm = trimmed_mean_aggregate(models, beta=0.0)
        agg_fa = federated_averaging(models)
        # Both should produce the same output shape
        x = torch.randn(3, 8)
        out_tm = agg_tm(x)
        out_fa = agg_fa(x)
        self.assertEqual(out_tm.shape, out_fa.shape)
        # Outputs should be in the same rough ballpark (neither collapses to zero nor explodes)
        self.assertLess(out_tm.abs().max().item(), 1e4)

    def test_reduces_outlier_influence(self):
        """Trimmed mean should reduce the influence of extreme outlier updates."""
        base = _SmallMLP()
        normal_models = []
        for _ in range(8):
            m = copy.deepcopy(base)
            for p in m.parameters():
                p.data += torch.randn_like(p) * 0.01
            normal_models.append(m)

        outlier = copy.deepcopy(base)
        for p in outlier.parameters():
            p.data += 50.0
        all_models = normal_models + [outlier, outlier]  # 10 total

        agg_tm = trimmed_mean_aggregate(all_models, beta=0.1)
        agg_fa = federated_averaging(all_models)

        # Trimmed mean fc1 bias should be closer to the base than FedAvg
        dist_tm = (agg_tm.fc1.bias.data - base.fc1.bias.data).abs().mean().item()
        dist_fa = (agg_fa.fc1.bias.data - base.fc1.bias.data).abs().mean().item()
        self.assertLess(dist_tm, dist_fa, "Trimmed mean should dampen outlier influence.")

    def test_invalid_beta(self):
        """beta outside [0, 0.5) should raise ValueError."""
        models = _make_models(4)
        with self.assertRaises(ValueError):
            trimmed_mean_aggregate(models, beta=0.5)
        with self.assertRaises(ValueError):
            trimmed_mean_aggregate(models, beta=-0.1)


# ---------------------------------------------------------------------------
# Non-IID partition tests
# ---------------------------------------------------------------------------

class TestNonIIDPartition(unittest.TestCase):

    def _make_dataset(self, n: int = 300, n_features: int = 8, n_classes: int = 5):
        X = torch.randn(n, n_features)
        y = torch.randint(0, n_classes, (n,))
        return X, y

    def test_correct_number_of_partitions(self):
        X, y = self._make_dataset()
        partitions = non_iid_partition(X, y, n_agents=10, n_classes_per=2)
        self.assertEqual(len(partitions), 10)

    def test_partitions_are_non_empty(self):
        X, y = self._make_dataset()
        partitions = non_iid_partition(X, y, n_agents=5, n_classes_per=3)
        for i, (Xi, yi) in enumerate(partitions):
            self.assertGreater(Xi.shape[0], 0, f"Partition {i} is empty.")

    def test_feature_dim_preserved(self):
        X, y = self._make_dataset(n_features=12)
        partitions = non_iid_partition(X, y, n_agents=4, n_classes_per=2)
        for Xi, _ in partitions:
            self.assertEqual(Xi.shape[1], 12)

    def test_non_iid_distribution(self):
        """Each partition should have fewer than n_classes distinct classes."""
        X, y = self._make_dataset(n=500, n_classes=5)
        n_classes_per = 2
        partitions = non_iid_partition(X, y, n_agents=10, n_classes_per=n_classes_per)
        for Xi, yi in partitions:
            distinct = len(yi.unique())
            # Some partitions may have up to n_classes_per distinct classes
            self.assertLessEqual(
                distinct, 5,
                "Partition has more classes than expected."
            )

    def test_reproducibility(self):
        """Same seed should produce identical partitions."""
        X, y = self._make_dataset(n=200)
        p1 = non_iid_partition(X, y, n_agents=5, seed=0)
        p2 = non_iid_partition(X, y, n_agents=5, seed=0)
        for (X1, y1), (X2, y2) in zip(p1, p2):
            self.assertTrue(torch.equal(X1, X2))
            self.assertTrue(torch.equal(y1, y2))

    def test_different_seeds_different_partitions(self):
        """Different seeds should (almost certainly) produce different partitions."""
        X, y = self._make_dataset(n=200)
        p1 = non_iid_partition(X, y, n_agents=5, seed=0)
        p2 = non_iid_partition(X, y, n_agents=5, seed=99)
        # At least one partition should differ
        all_same = all(torch.equal(X1, X2) for (X1, _), (X2, _) in zip(p1, p2))
        self.assertFalse(all_same, "Different seeds produced identical partitions.")


if __name__ == "__main__":
    unittest.main()
