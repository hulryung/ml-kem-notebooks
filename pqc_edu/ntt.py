"""Number-Theoretic Transform for R_q with q=3329, n=256.

This is the exact NTT that FIPS 203 (ML-KEM) uses. The primitive 256-th
root of unity mod 3329 is 17. Coefficients are held in bit-reversed
order in the NTT domain (matches FIPS 203 Algorithm 9 / 10).

Educational implementation: O(n log n) via the classic Cooley-Tukey
butterfly.
"""
from __future__ import annotations
import numpy as np
from .polyring import Poly

Q = 3329
N = 256
ROOT = 17  # primitive 256-th root of unity mod 3329


def _bitrev7(i: int) -> int:
    r = 0
    for _ in range(7):
        r = (r << 1) | (i & 1)
        i >>= 1
    return r


# Precompute zetas in bit-reversed order: zeta[i] = ROOT^{bitrev7(i)} mod q
_ZETAS = np.array(
    [pow(ROOT, _bitrev7(i), Q) for i in range(128)], dtype=np.int64
)


def ntt(p: Poly) -> Poly:
    """Forward NTT. Returns a Poly whose coeffs live in the NTT domain."""
    assert p.n == N and p.q == Q
    a = p.coeffs.copy()
    k = 1
    length = 128
    while length >= 2:
        for start in range(0, N, 2 * length):
            zeta = int(_ZETAS[k])
            k += 1
            for j in range(start, start + length):
                t = (zeta * int(a[j + length])) % Q
                a[j + length] = (int(a[j]) - t) % Q
                a[j] = (int(a[j]) + t) % Q
        length //= 2
    return Poly(a, Q)


def intt(p: Poly) -> Poly:
    """Inverse NTT."""
    assert p.n == N and p.q == Q
    a = p.coeffs.copy()
    k = 127
    length = 2
    while length <= 128:
        for start in range(0, N, 2 * length):
            zeta = int(_ZETAS[k])
            k -= 1
            for j in range(start, start + length):
                t = int(a[j])
                a[j] = (t + int(a[j + length])) % Q
                a[j + length] = (zeta * (int(a[j + length]) - t)) % Q
        length *= 2
    # Multiply by 128^{-1} mod q (7 butterfly levels, not 8 — pairs stay packed).
    n_inv = pow(128, -1, Q)
    a = (a * n_inv) % Q
    return Poly(a, Q)


def pointwise_mul(a_ntt: Poly, b_ntt: Poly) -> Poly:
    """Base-case multiplication in the NTT domain (FIPS 203 Algorithm 11).

    For each pair i=0..127, multiply (a0+a1 X) * (b0+b1 X) mod (X^2 - gamma),
    where gamma = ROOT^{2*bitrev7(i) + 1} per FIPS 203.
    """
    assert a_ntt.n == N and b_ntt.n == N and a_ntt.q == Q
    out = np.zeros(N, dtype=np.int64)
    for i in range(128):
        gamma = pow(ROOT, 2 * _bitrev7(i) + 1, Q)
        a0 = int(a_ntt.coeffs[2 * i])
        a1 = int(a_ntt.coeffs[2 * i + 1])
        b0 = int(b_ntt.coeffs[2 * i])
        b1 = int(b_ntt.coeffs[2 * i + 1])
        out[2 * i] = (a0 * b0 + a1 * b1 * gamma) % Q
        out[2 * i + 1] = (a0 * b1 + a1 * b0) % Q
    return Poly(out, Q)


def poly_mul_ntt(a: Poly, b: Poly) -> Poly:
    """Multiply two polynomials via the NTT. Equivalent to `a * b`."""
    return intt(pointwise_mul(ntt(a), ntt(b)))
