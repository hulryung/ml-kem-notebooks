import numpy as np
import pytest
from pqc_edu.polyring import Poly

Q = 3329
N = 256


def test_add_is_mod_q():
    a = Poly(np.full(N, Q - 1), Q)
    b = Poly(np.ones(N, dtype=int), Q)
    c = a + b
    assert np.all(c.coeffs == 0)


def test_mul_anticyclic_rule():
    # x^n in Z_q[x]/(x^n+1) should equal -1 (= q-1)
    x = Poly(np.zeros(N, dtype=int), Q); x.coeffs[1] = 1
    # Compute x^N via iterated multiplication.
    result = Poly(np.zeros(N, dtype=int), Q); result.coeffs[0] = 1
    for _ in range(N):
        result = result * x
    # result should be -1 mod q = q-1 at coefficient 0, others 0
    assert result.coeffs[0] == Q - 1
    assert np.all(result.coeffs[1:] == 0)


def test_distributive():
    rng = np.random.default_rng(0)
    a = Poly(rng.integers(0, Q, N), Q)
    b = Poly(rng.integers(0, Q, N), Q)
    c = Poly(rng.integers(0, Q, N), Q)
    lhs = a * (b + c)
    rhs = (a * b) + (a * c)
    assert np.array_equal(lhs.coeffs, rhs.coeffs)


def test_small_ring_by_hand():
    # In Z_17[x]/(x^4+1): (1+x) * (x^3) = x^3 + x^4 = x^3 - 1 = 16 + 0x + 0x^2 + x^3
    q = 17
    a = Poly(np.array([1, 1, 0, 0]), q)
    b = Poly(np.array([0, 0, 0, 1]), q)
    c = a * b
    assert np.array_equal(c.coeffs, np.array([16, 0, 0, 1]))
