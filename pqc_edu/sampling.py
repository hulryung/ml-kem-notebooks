"""Deterministic samplers used by ML-KEM (FIPS 203 §4.2).

- prf(seed, nonce, out_len)       : SHAKE256 PRF
- sample_uniform(rho, i, j)       : rejection sampling over XOF stream (Alg. 7 "SampleNTT")
- cbd(buf, eta, n)                : Centered Binomial Distribution (Alg. 8)
"""
from __future__ import annotations
import hashlib
import numpy as np
from .polyring import Poly

Q = 3329
N = 256


def prf(seed: bytes, nonce: int, out_len: int) -> bytes:
    """SHAKE256 keyed by (seed || nonce). FIPS 203 'PRF'."""
    h = hashlib.shake_256()
    h.update(seed + bytes([nonce]))
    return h.digest(out_len)


def sample_uniform(rho: bytes, i: int, j: int, q: int = Q, n: int = N) -> Poly:
    """FIPS 203 Algorithm 7 SampleNTT.

    Squeezes SHAKE128(rho || j || i) and rejection-samples 12-bit values
    less than q. Over-squeezes generously; falls back to re-squeezing if
    n samples are not produced.
    """
    seed = rho + bytes([j, i])  # FIPS 203 order
    out = np.zeros(n, dtype=np.int64)
    count = 0
    length = 504  # 3 shake128 blocks; plenty for q=3329 rejection
    while count < n:
        buf = hashlib.shake_128(seed).digest(length)
        idx = 0
        while idx + 3 <= len(buf) and count < n:
            b0 = buf[idx]; b1 = buf[idx + 1]; b2 = buf[idx + 2]
            idx += 3
            d1 = b0 | ((b1 & 0x0F) << 8)
            d2 = (b1 >> 4) | (b2 << 4)
            if d1 < q:
                out[count] = d1; count += 1
                if count == n:
                    break
            if d2 < q and count < n:
                out[count] = d2; count += 1
        length += 168  # one more shake128 block
    return Poly(out, q)


def cbd(buf: bytes, eta: int, n: int = N, q: int = Q) -> Poly:
    """FIPS 203 Algorithm 8 SamplePolyCBD.

    Consumes `n * eta / 4` bytes of `buf` and produces a length-n polynomial
    with coefficients in {-eta, ..., eta}, reduced mod q.
    """
    needed = n * eta // 4
    if len(buf) < needed:
        raise ValueError(f"cbd needs {needed} bytes, got {len(buf)}")
    bits = np.unpackbits(np.frombuffer(buf[:needed], dtype=np.uint8), bitorder="little")
    coeffs = np.zeros(n, dtype=np.int64)
    for i in range(n):
        a = int(bits[2 * i * eta : 2 * i * eta + eta].sum())
        b = int(bits[2 * i * eta + eta : 2 * i * eta + 2 * eta].sum())
        coeffs[i] = (a - b) % q
    return Poly(coeffs, q)
