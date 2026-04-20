"""Toy single-bit LWE PKE (Regev 2005, simplified).

Parameters (n, q, sigma) are chosen small enough to be solvable in
notebook 03. This is an educational stepping stone to ML-KEM, not a
secure scheme.

KeyGen:
    s  <- Z_q^n uniform                (secret)
    Generate m samples (a_i, b_i = <a_i, s> + e_i mod q) with e_i small Gaussian.
    Public key = list of (a_i, b_i). Secret key = s.

Encrypt(bit m in {0,1}):
    S <- random subset of the m samples
    a* = sum_{i in S} a_i mod q
    b* = sum_{i in S} b_i + m * floor(q/2) mod q
    return (a*, b*)

Decrypt:
    v = b* - <a*, s> mod q
    If v closer to 0  -> 0, if closer to q/2 -> 1.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass(frozen=True)
class ToyPublicKey:
    q: int
    samples: np.ndarray  # shape (m, n+1); last column is b, first n columns are a


@dataclass(frozen=True)
class ToySecretKey:
    q: int
    s: np.ndarray


@dataclass(frozen=True)
class ToyCiphertext:
    a: np.ndarray
    b: int


def _sample_gaussian(rng, sigma: float, size, q: int) -> np.ndarray:
    return np.round(rng.normal(0, sigma, size)).astype(np.int64) % q


def toy_keygen(n: int, q: int, sigma: float, rng, m: int | None = None) -> Tuple[ToyPublicKey, ToySecretKey]:
    if m is None:
        m = 2 * n + 10   # enough samples to have many subsets during encryption
    s = rng.integers(0, q, n)
    A = rng.integers(0, q, (m, n))
    e = _sample_gaussian(rng, sigma, m, q)
    b = (A @ s + e) % q
    samples = np.concatenate([A, b.reshape(-1, 1)], axis=1)
    return ToyPublicKey(q=q, samples=samples), ToySecretKey(q=q, s=s)


def toy_encrypt(pk: ToyPublicKey, bit: int, rng) -> ToyCiphertext:
    if bit not in (0, 1):
        raise ValueError("bit must be 0 or 1")
    m = pk.samples.shape[0]
    # Pick a random subset by sampling a 0/1 mask.
    mask = rng.integers(0, 2, m)
    subset = pk.samples[mask.astype(bool)]
    a_star = np.sum(subset[:, :-1], axis=0) % pk.q
    b_star = int(np.sum(subset[:, -1]) % pk.q)
    b_star = (b_star + bit * (pk.q // 2)) % pk.q
    return ToyCiphertext(a=a_star, b=b_star)


def toy_decrypt(sk: ToySecretKey, ct: ToyCiphertext) -> int:
    v = (ct.b - int(np.dot(ct.a, sk.s))) % sk.q
    # Nearest of {0, q/2}
    return 0 if min(v, sk.q - v) < abs(v - sk.q // 2) else 1
