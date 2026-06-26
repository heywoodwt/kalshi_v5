"""Online normalizers for observations and rewards.

Provides a simple running mean/std (Welford) implementation suitable for
online normalization of features and rewards. Normalizers are intentionally
lightweight and easy to port to another language (Rust) later.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np


class RunningMeanStd:
    """Compute running mean and variance per-dimension using Welford's algorithm.

    This keeps O(1) memory per feature and updates in O(d) time for d dims.
    """

    def __init__(self, dim: int, epsilon: float = 1e-4) -> None:
        self.dim = dim
        self.mean = np.zeros(dim, dtype=np.float64)
        self.var = np.ones(dim, dtype=np.float64)
        self.count = epsilon

    def update(self, x: np.ndarray) -> None:
        """Update running stats with a new observation x (shape: (dim,) or (n,dim))."""
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        assert x.shape[1] == self.dim, "Input dim mismatch"
        batch_count = x.shape[0]
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)

        # Welford / parallel update
        new_count = self.count + batch_count
        delta = batch_mean - self.mean
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + (delta ** 2) * (self.count * batch_count / new_count)

        self.mean = self.mean + delta * (batch_count / new_count)
        self.var = M2 / new_count
        self.count = new_count

    def normalize(self, x: np.ndarray, clip: float = 10.0) -> np.ndarray:
        """Return normalized array with same shape as x.

        Normalization: (x - mean) / (std + 1e-8), clipped to [-clip, clip].
        """
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
            squeezed = True
        else:
            squeezed = False
        std = np.sqrt(self.var)
        out = (x - self.mean) / (std + 1e-8)
        out = np.clip(out, -clip, clip)
        return out.reshape(-1, self.dim)[0] if squeezed else out


# Global observation normalizer (18-dim state vector expected)
OBS_DIM = 18
OBS_NORMALIZER = RunningMeanStd(OBS_DIM)

# Reward normalizer (scalar running mean/std)
class RewardNormalizer:
    def __init__(self, epsilon: float = 1e-4):
        self.count = epsilon
        self.mean = 0.0
        self.var = 1.0

    def update(self, r: float) -> None:
        # Online update for scalar
        batch_count = 1
        batch_mean = float(r)
        batch_var = 0.0
        new_count = self.count + batch_count
        delta = batch_mean - self.mean
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + (delta ** 2) * (self.count * batch_count / new_count)
        self.mean = self.mean + delta * (batch_count / new_count)
        self.var = M2 / new_count
        self.count = new_count

    def normalize(self, r: float, clip: float = 10.0) -> float:
        std = math.sqrt(self.var)
        out = (r - self.mean) / (std + 1e-8)
        return float(max(-clip, min(clip, out)))

REWARD_NORMALIZER = RewardNormalizer()

