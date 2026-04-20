"""Polynomial ring R_q = Z_q[x] / (x^n + 1).

The anti-cyclic identity x^n = -1 is what makes ML-KEM's arithmetic fast
once combined with the NTT (see pqc_edu.ntt).
"""
from __future__ import annotations
import numpy as np


class Poly:
    """Element of Z_q[x]/(x^n+1). Immutable by convention; operators return new objects."""

    __slots__ = ("coeffs", "q", "n")

    def __init__(self, coeffs, q: int):
        arr = np.asarray(coeffs, dtype=np.int64) % q
        self.coeffs = arr
        self.q = q
        self.n = len(arr)

    # -- arithmetic ----------------------------------------------------
    def __add__(self, other: "Poly") -> "Poly":
        self._check(other)
        return Poly((self.coeffs + other.coeffs) % self.q, self.q)

    def __sub__(self, other: "Poly") -> "Poly":
        self._check(other)
        return Poly((self.coeffs - other.coeffs) % self.q, self.q)

    def __neg__(self) -> "Poly":
        return Poly((-self.coeffs) % self.q, self.q)

    def __mul__(self, other: "Poly") -> "Poly":
        self._check(other)
        # Schoolbook convolution, then fold x^n = -1.
        n, q = self.n, self.q
        full = np.zeros(2 * n, dtype=np.int64)
        for i in range(n):
            if self.coeffs[i]:
                full[i : i + n] += self.coeffs[i] * other.coeffs
        # Fold: coefficients [n .. 2n-1] contribute with sign -1 to [0 .. n-1].
        out = (full[:n] - full[n:]) % q
        return Poly(out, q)

    # -- misc ----------------------------------------------------------
    def __eq__(self, other) -> bool:
        if not isinstance(other, Poly):
            return NotImplemented
        return self.q == other.q and np.array_equal(self.coeffs, other.coeffs)

    def __repr__(self) -> str:
        return f"Poly(n={self.n}, q={self.q}, coeffs={self.coeffs.tolist()[:4]}...)"

    def _check(self, other: "Poly") -> None:
        if self.n != other.n or self.q != other.q:
            raise ValueError("Poly operands must share (n, q)")

    @classmethod
    def zero(cls, n: int, q: int) -> "Poly":
        return cls(np.zeros(n, dtype=np.int64), q)

    @classmethod
    def from_int(cls, value: int, n: int, q: int) -> "Poly":
        p = cls.zero(n, q)
        p.coeffs[0] = value % q
        return p
