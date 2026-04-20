"""Toy attacks on small-parameter LWE.

Two approaches:

1. brute_force_secret: enumerate all candidate secrets in Z_q^n. O(q^n),
   only feasible for n <= ~5.

2. gaussian_elimination_noiseless: treat m samples as a linear system
   and solve exactly. Works only when sigma is artificially set to 0,
   showing why the noise term is what hides the secret.

Neither of these scales. Notebook 03 plots wall-clock vs n and watches
them explode.
"""
from __future__ import annotations
import itertools
import time
from dataclasses import dataclass
from typing import Optional
import numpy as np

from .lwe import ToyPublicKey, ToySecretKey


@dataclass
class AttackResult:
    secret: Optional[np.ndarray]
    seconds: float
    method: str
    gave_up: bool = False


def brute_force_secret(pk: ToyPublicKey, error_tolerance: int, time_budget_s: float = 60.0) -> AttackResult:
    """Enumerate s in Z_q^n and score by how many samples it explains within tolerance."""
    n = pk.samples.shape[1] - 1
    q = pk.q
    A = pk.samples[:, :-1]
    b = pk.samples[:, -1]
    start = time.time()
    best_score = -1
    best_s = None
    for s_tuple in itertools.product(range(q), repeat=n):
        if time.time() - start > time_budget_s:
            return AttackResult(secret=best_s, seconds=time.time() - start, method="brute_force", gave_up=True)
        s = np.array(s_tuple)
        residuals = (b - A @ s) % q
        # Wrap residual into [-q/2, q/2]
        residuals = np.where(residuals > q // 2, residuals - q, residuals)
        score = int(np.sum(np.abs(residuals) <= error_tolerance))
        if score > best_score:
            best_score = score
            best_s = s
            if score == len(b):
                break
    return AttackResult(secret=best_s, seconds=time.time() - start, method="brute_force")


def gaussian_elimination_noiseless(pk: ToyPublicKey) -> AttackResult:
    """Solve A s = b (mod q) exactly. Only works if sigma == 0 when pk was generated.

    Implementation: row-reduce [A | b] over Z_q using modular inverse.
    Requires q prime.
    """
    n = pk.samples.shape[1] - 1
    q = pk.q
    start = time.time()
    M = pk.samples.astype(np.int64) % q
    rows = M.shape[0]

    col = 0
    for row in range(min(rows, n)):
        # Find pivot.
        piv = None
        for r in range(row, rows):
            if M[r, col] != 0:
                piv = r; break
        if piv is None:
            col += 1
            if col >= n:
                break
            continue
        M[[row, piv]] = M[[piv, row]]
        inv = pow(int(M[row, col]), -1, q)
        M[row] = (M[row] * inv) % q
        for r in range(rows):
            if r != row and M[r, col]:
                M[r] = (M[r] - M[r, col] * M[row]) % q
        col += 1

    s = M[:n, -1].copy()
    return AttackResult(secret=s, seconds=time.time() - start, method="gaussian_elimination_noiseless")


def verify_secret(sk_true: ToySecretKey, recovered: Optional[np.ndarray]) -> bool:
    if recovered is None:
        return False
    return np.array_equal(np.asarray(recovered) % sk_true.q, sk_true.s % sk_true.q)
