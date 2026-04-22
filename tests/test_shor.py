import numpy as np
import pytest
from pqc_edu.quantum.shor import factor, ShorResult
from pqc_edu.quantum.classical_utils import continued_fraction_denominator


def test_continued_fraction_basic():
    # 64/256 = 1/4 -> r = 4
    assert continued_fraction_denominator(64, 8, 15) == 4
    # 3/8 -> r = 8
    assert continued_fraction_denominator(3, 3, 100) == 8


@pytest.mark.parametrize("N,expected_factors", [
    (15, {3, 5}),
    (21, {3, 7}),
    (35, {5, 7}),
])
def test_factor_succeeds(N, expected_factors):
    # Use a fixed seed; max_attempts large enough so at least one succeeds.
    rng = np.random.default_rng(0)
    result = factor(N, rng, max_attempts=20)
    assert result.factors is not None, f"failed to factor {N}"
    a, b = result.factors
    assert a * b == N or a * b == N * 1  # allow repeated factor presentation
    assert {a, b} == expected_factors


def test_result_records_attempts():
    rng = np.random.default_rng(0)
    result = factor(15, rng, max_attempts=10)
    assert len(result.attempts) > 0
    # Every attempt has a status tag.
    for att in result.attempts:
        assert att.status in {
            "success", "a_not_coprime", "odd_period",
            "trivial_factor", "period_not_found",
        }


def test_even_N_shortcut():
    rng = np.random.default_rng(0)
    result = factor(30, rng, max_attempts=5)
    # 30 is even, so factor() should short-circuit.
    assert result.factors is not None
    assert 2 in result.factors
