import numpy as np
import pytest
from pqc_edu.attacks_advanced.lll import lll_reduce
from pqc_edu.attacks_advanced.bkz import svp_enumerate, bkz_reduce


def test_svp_enumerate_2d_known():
    B = np.array([[10, 0], [3, 1]], dtype=np.int64)
    v = svp_enumerate(B)
    # The shortest nonzero vector of L([10,0],[3,1]) is (3,1) with norm sqrt(10).
    assert int(np.dot(v, v)) == 10


def test_svp_enumerate_3d_identity():
    # L = Z^3; shortest nonzero vector has norm 1.
    B = np.eye(3, dtype=np.int64)
    v = svp_enumerate(B)
    assert int(np.dot(v, v)) == 1


def test_bkz_block2_equals_lll():
    rng = np.random.default_rng(5)
    B = rng.integers(-20, 20, (5, 5)).astype(np.int64)
    while abs(np.linalg.det(B)) < 1e-6:
        B = rng.integers(-20, 20, (5, 5)).astype(np.int64)
    lll_result = lll_reduce(B, delta=0.99)
    bkz_result = bkz_reduce(B, block_size=2, delta=0.99)
    # First-vector lengths must match (both are LLL-reduced with same delta).
    assert np.linalg.norm(lll_result[0]) == pytest.approx(
        np.linalg.norm(bkz_result[0]), rel=1e-9
    )


def test_bkz_larger_block_shortens_first_vector():
    rng = np.random.default_rng(6)
    B = rng.integers(-30, 30, (6, 6)).astype(np.int64)
    while abs(np.linalg.det(B)) < 1e-6:
        B = rng.integers(-30, 30, (6, 6)).astype(np.int64)
    lengths = []
    for bs in (2, 4, 6):
        reduced = bkz_reduce(B, block_size=bs)
        lengths.append(np.linalg.norm(reduced[0]))
    # Monotone non-increasing.
    for i in range(1, len(lengths)):
        assert lengths[i] <= lengths[i - 1] + 1e-9


def test_bkz_preserves_determinant():
    rng = np.random.default_rng(7)
    B = rng.integers(-15, 15, (4, 4)).astype(np.int64)
    while abs(np.linalg.det(B)) < 1e-6:
        B = rng.integers(-15, 15, (4, 4)).astype(np.int64)
    d_before = abs(np.linalg.det(B))
    reduced = bkz_reduce(B, block_size=3)
    d_after = abs(np.linalg.det(reduced))
    assert abs(d_before - d_after) < 1e-6 * d_before
