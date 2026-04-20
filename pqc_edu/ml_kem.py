"""FIPS 203 ML-KEM. Algorithms cross-referenced in docstrings.

This file implements the K-PKE inner scheme first; the ML-KEM wrapper
(with the Fujisaki-Okamoto transform) is added in the next task.

Design notes:
- All polynomials live in R_q = Z_q[x]/(x^n+1) (see pqc_edu.polyring).
- In "NTT domain" a Poly's coeffs are the NTT-transformed values.
- Byte encodings use the Compress_d / Decompress_d functions from FIPS 203 §4.2.
"""
from __future__ import annotations
import hashlib
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

from .polyring import Poly
from .ntt import ntt, intt, pointwise_mul, Q, N
from .sampling import prf, cbd, sample_uniform
from .params import MLKEMParams


# --- compression / encoding ---------------------------------------------
def compress(p: Poly, d: int) -> Poly:
    """FIPS 203 Compress_d: coeffs in Z_q -> coeffs in Z_{2^d}."""
    q = p.q
    factor = 1 << d
    c = ((p.coeffs * factor + q // 2) // q) % factor
    return Poly(c, factor)


def decompress(p: Poly, d: int, q: int = Q) -> Poly:
    """FIPS 203 Decompress_d: coeffs in Z_{2^d} -> coeffs in Z_q."""
    factor = 1 << d
    out = ((p.coeffs * q + factor // 2) // factor) % q
    return Poly(out, q)


def byte_encode(polys: List[Poly], d: int) -> bytes:
    """Pack d-bit coefficients of each poly into a byte string. FIPS 203 Algorithm 5.

    Little-endian bit packing within each coefficient.
    """
    n = polys[0].n
    total_bits = len(polys) * n * d
    all_bits = np.zeros(total_bits, dtype=np.uint8)
    offset = 0
    for p in polys:
        for i in range(n):
            val = int(p.coeffs[i])
            for b in range(d):
                all_bits[offset + i * d + b] = (val >> b) & 1
        offset += n * d
    return np.packbits(all_bits, bitorder="little").tobytes()


def byte_decode(buf: bytes, d: int, n: int, count: int, q_target: int) -> List[Poly]:
    """Inverse of byte_encode. Returns `count` polynomials of length n, modulus q_target.

    q_target is q (=3329) for d=12 coefficients, or 2^d for compressed values.
    """
    bits = np.unpackbits(np.frombuffer(buf, dtype=np.uint8), bitorder="little")
    polys = []
    offset = 0
    for _ in range(count):
        coeffs = np.zeros(n, dtype=np.int64)
        for i in range(n):
            val = 0
            for j in range(d):
                val |= int(bits[offset + i * d + j]) << j
            coeffs[i] = val
        offset += n * d
        polys.append(Poly(coeffs, q_target))
    return polys


def _bytes_to_bits(b: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(b, dtype=np.uint8), bitorder="little")


def _bits_to_bytes(bits: np.ndarray) -> bytes:
    return np.packbits(bits.astype(np.uint8), bitorder="little").tobytes()


# --- K-PKE --------------------------------------------------------------
@dataclass
class KPKEKeys:
    ek: bytes   # encryption key
    dk: bytes   # decryption key


def k_pke_keygen(params: MLKEMParams, d_seed: bytes) -> KPKEKeys:
    """FIPS 203 Algorithm 13 K-PKE.KeyGen."""
    assert len(d_seed) == 32
    g = hashlib.sha3_512(d_seed).digest()
    rho, sigma = g[:32], g[32:]
    # Matrix A: k x k polys, each uniform in R_q, NTT domain
    A_hat = [[sample_uniform(rho, i, j) for j in range(params.k)] for i in range(params.k)]
    # Secret s and error e, each a length-k vector of CBD samples
    N_ctr = 0
    s = []
    for _ in range(params.k):
        s.append(cbd(prf(sigma, N_ctr, params.n * params.eta1 // 4), params.eta1))
        N_ctr += 1
    e = []
    for _ in range(params.k):
        e.append(cbd(prf(sigma, N_ctr, params.n * params.eta1 // 4), params.eta1))
        N_ctr += 1
    s_hat = [ntt(p) for p in s]
    e_hat = [ntt(p) for p in e]
    # t_hat = A_hat @ s_hat + e_hat  (all in NTT domain)
    t_hat = []
    for i in range(params.k):
        acc = Poly.zero(N, Q)
        for j in range(params.k):
            acc = acc + pointwise_mul(A_hat[i][j], s_hat[j])
        acc = acc + e_hat[i]
        t_hat.append(acc)
    ek = byte_encode(t_hat, 12) + rho
    dk = byte_encode(s_hat, 12)
    return KPKEKeys(ek=ek, dk=dk)


def k_pke_encrypt(params: MLKEMParams, ek: bytes, message: bytes, coins: bytes) -> bytes:
    """FIPS 203 Algorithm 14 K-PKE.Encrypt."""
    assert len(message) == 32
    assert len(coins) == 32
    t_hat_bytes = ek[:-32]
    rho = ek[-32:]
    t_hat = byte_decode(t_hat_bytes, 12, N, params.k, Q)
    # A_hat^T (transposed): A_hat_T[i][j] = A_hat[j][i]
    A_hat_T = [[sample_uniform(rho, j, i) for j in range(params.k)] for i in range(params.k)]
    N_ctr = 0
    r = []
    for _ in range(params.k):
        r.append(cbd(prf(coins, N_ctr, N * params.eta1 // 4), params.eta1))
        N_ctr += 1
    e1 = []
    for _ in range(params.k):
        e1.append(cbd(prf(coins, N_ctr, N * params.eta2 // 4), params.eta2))
        N_ctr += 1
    e2 = cbd(prf(coins, N_ctr, N * params.eta2 // 4), params.eta2)
    r_hat = [ntt(p) for p in r]
    # u = INTT(A_hat^T @ r_hat) + e1
    u = []
    for i in range(params.k):
        acc = Poly.zero(N, Q)
        for j in range(params.k):
            acc = acc + pointwise_mul(A_hat_T[i][j], r_hat[j])
        u.append(intt(acc) + e1[i])
    # v = INTT(t_hat . r_hat) + e2 + Decompress_1(message bits)
    acc = Poly.zero(N, Q)
    for j in range(params.k):
        acc = acc + pointwise_mul(t_hat[j], r_hat[j])
    mu = decompress(Poly(_bytes_to_bits(message), 2), 1)
    v = intt(acc) + e2 + mu
    c1 = byte_encode([compress(p, params.du) for p in u], params.du)
    c2 = byte_encode([compress(v, params.dv)], params.dv)
    return c1 + c2


def k_pke_decrypt(params: MLKEMParams, dk: bytes, ct: bytes) -> bytes:
    """FIPS 203 Algorithm 15 K-PKE.Decrypt."""
    c1_len = 32 * params.du * params.k
    c1, c2 = ct[:c1_len], ct[c1_len:]
    u_compressed = byte_decode(c1, params.du, N, params.k, 1 << params.du)
    v_compressed = byte_decode(c2, params.dv, N, 1, 1 << params.dv)[0]
    u = [decompress(p, params.du) for p in u_compressed]
    v = decompress(v_compressed, params.dv)
    s_hat = byte_decode(dk, 12, N, params.k, Q)
    # m' = Compress_1(v - INTT(s_hat . NTT(u)))
    u_hat = [ntt(p) for p in u]
    acc = Poly.zero(N, Q)
    for j in range(params.k):
        acc = acc + pointwise_mul(s_hat[j], u_hat[j])
    w = v - intt(acc)
    m_poly = compress(w, 1)
    return _bits_to_bytes(m_poly.coeffs.astype(np.uint8))


# ======================================================================
# ML-KEM wrapper (FO-transform) — FIPS 203 §7
# ======================================================================
import os as _os


def _ml_kem_keygen_from_seeds(params: MLKEMParams, d: bytes, z: bytes) -> Tuple[bytes, bytes]:
    assert len(d) == 32 and len(z) == 32
    k_pke = k_pke_keygen(params, d)
    ek = k_pke.ek
    # dk = dk_pke || ek || H(ek) || z
    h_ek = hashlib.sha3_256(ek).digest()
    dk = k_pke.dk + ek + h_ek + z
    return ek, dk


def ml_kem_keygen(params: MLKEMParams) -> Tuple[bytes, bytes]:
    """Algorithm 16 ML-KEM.KeyGen."""
    return _ml_kem_keygen_from_seeds(params, _os.urandom(32), _os.urandom(32))


def _ml_kem_encaps_from_seed(params: MLKEMParams, ek: bytes, m: bytes) -> Tuple[bytes, bytes]:
    assert len(m) == 32
    h_ek = hashlib.sha3_256(ek).digest()
    g = hashlib.sha3_512(m + h_ek).digest()
    K, r = g[:32], g[32:]
    ct = k_pke_encrypt(params, ek, m, r)
    return K, ct


def ml_kem_encaps(params: MLKEMParams, ek: bytes) -> Tuple[bytes, bytes]:
    """Algorithm 17 ML-KEM.Encaps."""
    return _ml_kem_encaps_from_seed(params, ek, _os.urandom(32))


def ml_kem_decaps(params: MLKEMParams, dk: bytes, ct: bytes) -> bytes:
    """Algorithm 18 ML-KEM.Decaps.

    FO-transform rejection branch: if re-encryption disagrees, return a
    pseudo-random shared secret derived from z and the ciphertext rather
    than a bogus key. This makes decryption failure indistinguishable.
    """
    dk_pke_len = 384 * params.k
    ek_len = params.ek_bytes
    dk_pke = dk[:dk_pke_len]
    ek = dk[dk_pke_len : dk_pke_len + ek_len]
    z = dk[dk_pke_len + ek_len + 32 :]
    m_prime = k_pke_decrypt(params, dk_pke, ct)
    h_ek = hashlib.sha3_256(ek).digest()
    g = hashlib.sha3_512(m_prime + h_ek).digest()
    K_prime, r_prime = g[:32], g[32:]
    ct_prime = k_pke_encrypt(params, ek, m_prime, r_prime)
    if ct == ct_prime:
        return K_prime
    # Implicit rejection: return pseudo-random K = SHAKE256(z || ct, 32)
    h = hashlib.shake_256(); h.update(z + ct)
    return h.digest(32)
