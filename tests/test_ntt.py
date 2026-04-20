import numpy as np
import pytest
from pqc_edu.polyring import Poly
from pqc_edu.ntt import ntt, intt, poly_mul_ntt

Q = 3329
N = 256


def test_ntt_roundtrip():
    rng = np.random.default_rng(42)
    for _ in range(10):
        a = Poly(rng.integers(0, Q, N), Q)
        assert intt(ntt(a)) == a


def test_ntt_matches_schoolbook():
    rng = np.random.default_rng(1)
    a = Poly(rng.integers(0, Q, N), Q)
    b = Poly(rng.integers(0, Q, N), Q)
    assert poly_mul_ntt(a, b) == a * b


def test_ntt_multiplication_zero():
    z = Poly.zero(N, Q)
    a = Poly(np.random.default_rng(0).integers(0, Q, N), Q)
    assert poly_mul_ntt(a, z) == z
