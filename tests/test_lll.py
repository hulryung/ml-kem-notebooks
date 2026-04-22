import numpy as np
import pytest
from pqc_edu.attacks_advanced.lll import gram_schmidt, lll_reduce


def lll_conditions_ok(B, delta=0.75):
    """Check that B satisfies the two LLL conditions."""
    B = np.asarray(B, dtype=np.float64)
    B_star, mu = gram_schmidt(B.astype(np.int64))
    n = B.shape[0]
    for i in range(n):
        for j in range(i):
            if abs(mu[i, j]) > 0.5 + 1e-9:
                return False, f"size-reduced violated: |mu[{i},{j}]|={abs(mu[i,j])}"
    for i in range(1, n):
        lhs = np.dot(B_star[i], B_star[i])
        rhs = (delta - mu[i, i-1] ** 2) * np.dot(B_star[i-1], B_star[i-1])
        if lhs < rhs - 1e-9:
            return False, f"Lovász violated at i={i}: {lhs} < {rhs}"
    return True, "ok"


def test_gram_schmidt_2d_orthogonal():
    B = np.array([[4, 1], [0, 2]], dtype=np.int64)
    B_star, mu = gram_schmidt(B)
    assert np.allclose(B_star[0], [4, 1])
    assert abs(np.dot(B_star[0], B_star[1])) < 1e-10


def test_lll_handles_2d_skewed_basis():
    # Hoffstein textbook example: B = [[19, 2], [12, 15]]
    B = np.array([[19, 2], [12, 15]], dtype=np.int64)
    reduced = lll_reduce(B)
    ok, msg = lll_conditions_ok(reduced)
    assert ok, msg


def test_lll_preserves_determinant():
    rng = np.random.default_rng(1)
    B = rng.integers(-10, 10, (5, 5)).astype(np.int64)
    # Ensure non-singular
    while abs(np.linalg.det(B)) < 1e-6:
        B = rng.integers(-10, 10, (5, 5)).astype(np.int64)
    det_before = abs(np.linalg.det(B))
    reduced = lll_reduce(B)
    det_after = abs(np.linalg.det(reduced))
    assert abs(det_before - det_after) < 1e-6 * det_before


def test_lll_shortens_first_vector():
    rng = np.random.default_rng(2)
    for seed in range(5):
        rng = np.random.default_rng(seed)
        B = rng.integers(-50, 50, (4, 4)).astype(np.int64)
        while abs(np.linalg.det(B)) < 1e-6:
            B = rng.integers(-50, 50, (4, 4)).astype(np.int64)
        reduced = lll_reduce(B)
        assert np.linalg.norm(reduced[0]) <= np.linalg.norm(B[0]) + 1e-9


def test_lll_satisfies_conditions_on_random_bases():
    rng = np.random.default_rng(3)
    for _ in range(10):
        n = rng.integers(3, 7)
        B = rng.integers(-30, 30, (n, n)).astype(np.int64)
        while abs(np.linalg.det(B)) < 1e-6:
            B = rng.integers(-30, 30, (n, n)).astype(np.int64)
        reduced = lll_reduce(B)
        ok, msg = lll_conditions_ok(reduced)
        assert ok, f"n={n}: {msg}"
