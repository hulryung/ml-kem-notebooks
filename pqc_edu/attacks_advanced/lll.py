"""LLL (Lenstra–Lenstra–Lovász) basis reduction.

Takes a basis (rows are vectors) and returns a reduced basis whose first
vector is short. Pure-Python educational implementation — full Gram-Schmidt
recomputation on every update. Fine up to n ~ 40.
"""
from __future__ import annotations
import numpy as np


def gram_schmidt(B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Gram-Schmidt orthogonalization.

    Args:
        B: (n, m) array of lattice basis vectors (rows).
    Returns:
        (B_star, mu) where B_star[i] is the i-th GSO vector,
        mu[i, j] is the projection coefficient of b_i onto b*_j for j < i.
    """
    B = B.astype(np.float64)
    n = B.shape[0]
    B_star = np.zeros_like(B)
    mu = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        B_star[i] = B[i].copy()
        for j in range(i):
            denom = np.dot(B_star[j], B_star[j])
            if denom == 0:
                mu[i, j] = 0.0
            else:
                mu[i, j] = np.dot(B[i], B_star[j]) / denom
            B_star[i] = B_star[i] - mu[i, j] * B_star[j]
    return B_star, mu


def lll_reduce(B: np.ndarray, delta: float = 0.75) -> np.ndarray:
    """LLL-reduce the basis B (rows are vectors) with parameter delta.

    Standard delta range: 1/4 < delta < 1. Common choice: 3/4 for speed,
    0.99 for stronger reduction. Returns a new int64 array.
    """
    if not (0.25 < delta < 1.0):
        raise ValueError(f"delta must be in (1/4, 1), got {delta}")
    B = B.astype(np.int64).copy()
    n = B.shape[0]

    def recompute():
        return gram_schmidt(B)

    B_star, mu = recompute()
    k = 1
    while k < n:
        # Size reduction: make |mu[k, j]| <= 1/2 for j < k.
        for j in range(k - 1, -1, -1):
            if abs(mu[k, j]) > 0.5:
                r = int(round(mu[k, j]))
                B[k] = B[k] - r * B[j]
                B_star, mu = recompute()

        # Lovász condition: ||b*_k||^2 >= (delta - mu[k, k-1]^2) * ||b*_{k-1}||^2
        lhs = np.dot(B_star[k], B_star[k])
        rhs = (delta - mu[k, k - 1] ** 2) * np.dot(B_star[k - 1], B_star[k - 1])
        if lhs >= rhs:
            k += 1
        else:
            B[[k, k - 1]] = B[[k - 1, k]]
            B_star, mu = recompute()
            k = max(k - 1, 1)
    return B
