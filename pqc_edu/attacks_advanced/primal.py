"""Primal attack on LWE: Kannan embedding + BKZ + secret recovery."""
from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from pqc_edu.lwe import ToyPublicKey
from .bkz import bkz_reduce
from .embedding import kannan_embedding


@dataclass
class PrimalResult:
    secret: Optional[np.ndarray]
    recovered_error: Optional[np.ndarray]
    reduction_time: float
    block_size: int
    status: str  # "success" | "reduction_failed" | "short_vector_not_secret"


def _center(x: np.ndarray, q: int) -> np.ndarray:
    """Reduce each entry to (-q/2, q/2]."""
    y = x % q
    return np.where(y > q // 2, y - q, y)


def primal_attack(
    pk: ToyPublicKey,
    block_size: int = 4,
    time_budget_s: float = 120.0,
    M: int = 1,
) -> PrimalResult:
    """Run the primal attack. Tries every reduced basis row to find the
    one that decodes to the secret; returns on first success."""
    A = pk.samples[:, :-1].astype(np.int64)
    b = pk.samples[:, -1].astype(np.int64)
    m, n = A.shape
    q = pk.q

    start = time.time()
    try:
        B = kannan_embedding(pk, M=M)
        B_reduced = bkz_reduce(B, block_size=block_size)
    except Exception:
        return PrimalResult(
            secret=None, recovered_error=None,
            reduction_time=time.time() - start,
            block_size=block_size,
            status="reduction_failed",
        )
    reduction_time = time.time() - start

    # Try each reduced row (and its negation) as a candidate target vector.
    for row in B_reduced:
        for sign in (1, -1):
            v = sign * row
            if v[n + m] != M:
                continue
            e_cand = v[:m]
            s_cand = -v[m : m + n]
            # Verify: A·s + e ≡ b (mod q)
            residual = _center(A @ s_cand + e_cand - b, q)
            if np.all(residual == 0):
                return PrimalResult(
                    secret=(s_cand.astype(np.int64) % q),
                    recovered_error=e_cand.astype(np.int64),
                    reduction_time=reduction_time,
                    block_size=block_size,
                    status="success",
                )

    return PrimalResult(
        secret=None, recovered_error=None,
        reduction_time=reduction_time,
        block_size=block_size,
        status="short_vector_not_secret",
    )
