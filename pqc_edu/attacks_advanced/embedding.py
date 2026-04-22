"""Kannan embedding: convert an LWE instance into a uSVP lattice.

Given public key (A, b) where b = A·s + e mod q, build a lattice whose
shortest nonzero vector encodes (e, -s, M). BKZ-reducing this lattice
recovers (s, e).
"""
from __future__ import annotations
import numpy as np

from pqc_edu.lwe import ToyPublicKey


def kannan_embedding(pk: ToyPublicKey, M: int | None = None) -> np.ndarray:
    """Build the Kannan primal-attack basis.

    Matrix layout (rows = basis vectors, dim = m + n + 1):
      Rows 0..n-1   : (A[:, j], e_j, 0)     — "y" rows, one per secret coord
      Rows n..n+m-1 : (q·e_i, 0_n, 0)       — "q·I" rows, one per sample
      Row n+m       : (b, 0_n, M)           — "target" row

    Args:
        pk: toy LWE public key (has .samples of shape (m, n+1) and .q).
        M:  embedding factor for the last coord. Defaults to 1, which is
            adequate for toy parameters.

    Returns:
        int64 ndarray of shape (m+n+1, m+n+1).
    """
    q = pk.q
    A = pk.samples[:, :-1].astype(np.int64)  # shape (m, n)
    b = pk.samples[:, -1].astype(np.int64)   # shape (m,)
    m, n = A.shape
    if M is None:
        M = 1
    dim = m + n + 1
    B = np.zeros((dim, dim), dtype=np.int64)

    # y rows (one per secret coord j):  [A[:, j], e_j, 0]
    for j in range(n):
        B[j, :m] = A[:, j]
        B[j, m + j] = 1

    # q·I rows (one per sample i):  [q·e_i, 0_n, 0]
    for i in range(m):
        B[n + i, i] = q

    # target row:  [b, 0_n, M]
    B[n + m, :m] = b
    B[n + m, n + m] = M

    return B
