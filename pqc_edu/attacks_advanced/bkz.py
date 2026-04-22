"""BKZ (Block Korkine-Zolotarev) reduction and SVP enumeration.

Educational implementation. BKZ iteratively:
  1. LLL-reduces the current basis.
  2. For each block of size `block_size`, finds the shortest vector via
     brute-force enumeration, inserts it, and re-LLLs.
Repeats until no block improves (or `max_tours` reached).

SVP enumeration here is the simplest correct variant: iterative deepening
over bounded integer combinations of basis vectors. Fine up to block size
~ 6 in pure Python; for larger blocks use fpylll.
"""
from __future__ import annotations
import itertools
import numpy as np

from .lll import gram_schmidt, lll_reduce


def _integer_combination(B: np.ndarray, coeffs: tuple) -> np.ndarray:
    out = np.zeros(B.shape[1], dtype=np.int64)
    for i, c in enumerate(coeffs):
        if c != 0:
            out = out + c * B[i]
    return out


def svp_enumerate(B: np.ndarray) -> np.ndarray:
    """Return the shortest nonzero lattice vector via bounded enumeration.

    Uses iterative deepening: tries coefficient range [-c, c] for c = 1,
    2, 3, ... until the same shortest vector reappears.
    """
    B = B.astype(np.int64)
    n = B.shape[0]
    best = B[0].copy()
    best_norm_sq = int(np.dot(best, best))
    for row in B[1:]:
        ns = int(np.dot(row, row))
        if 0 < ns < best_norm_sq:
            best_norm_sq = ns
            best = row.copy()

    c = 1
    prev_best_norm_sq = None
    # Cap to prevent runaway; block sizes > 6 aren't supported well anyway.
    while c <= 4:
        for coeffs in itertools.product(range(-c, c + 1), repeat=n):
            if all(x == 0 for x in coeffs):
                continue
            v = _integer_combination(B, coeffs)
            ns = int(np.dot(v, v))
            if 0 < ns < best_norm_sq:
                best_norm_sq = ns
                best = v
        if prev_best_norm_sq is not None and best_norm_sq == prev_best_norm_sq:
            break
        prev_best_norm_sq = best_norm_sq
        c += 1
    return best


def bkz_reduce(
    B: np.ndarray,
    block_size: int,
    delta: float = 0.99,
    max_tours: int = 10,
) -> np.ndarray:
    """BKZ reduction with the given block size. Returns a new int64 basis."""
    if block_size < 2:
        raise ValueError(f"block_size must be >= 2, got {block_size}")
    B = lll_reduce(B.astype(np.int64), delta=delta)
    n = B.shape[0]

    for tour in range(max_tours):
        changed = False
        for k in range(n - 1):
            end = min(k + block_size, n)
            block = B[k:end].copy()
            # Enumerate shortest vector in the sublattice spanned by `block`.
            short = svp_enumerate(block)
            # If it differs from block[0], insert it and re-LLL the extended basis.
            short_norm_sq = int(np.dot(short, short))
            if short_norm_sq < int(np.dot(block[0], block[0])):
                # Build extended basis: [short, B[k], ..., B[end-1]]
                extended = np.vstack([short.reshape(1, -1), B[k:end]]).astype(np.int64)
                reduced_block = lll_reduce(extended, delta=delta)
                # Take first `end-k` non-zero rows.
                nonzero = reduced_block[
                    np.any(reduced_block != 0, axis=1)
                ][: end - k]
                if nonzero.shape[0] == end - k:
                    B[k:end] = nonzero
                    B = lll_reduce(B, delta=delta)
                    changed = True
        if not changed:
            break
    return B
