import numpy as np
import pytest
from pqc_edu.lwe import toy_keygen
from pqc_edu.attacks_advanced.primal import primal_attack, PrimalResult


@pytest.mark.parametrize("n,q,sigma,block_size", [
    (8, 31, 0.8, 4),
    (10, 61, 0.8, 4),
    (12, 97, 0.8, 4),
])
def test_primal_attack_succeeds_on_small(n, q, sigma, block_size):
    # Small secret is required for Kannan embedding to find the planted secret.
    rng = np.random.default_rng(0)
    pk, sk = toy_keygen(n=n, q=q, sigma=sigma, rng=rng, m=2 * n, small_secret=True)
    result = primal_attack(pk, block_size=block_size, time_budget_s=60)
    assert result.status == "success", f"n={n}: status={result.status}"
    # Centered comparison: secret should match modulo q, accounting for small-secret range.
    recovered = result.secret % q
    expected = sk.s.astype(np.int64) % q
    assert np.array_equal(recovered, expected), (
        f"n={n}: recovered {recovered.tolist()} != expected {expected.tolist()}"
    )


def test_primal_result_fields():
    rng = np.random.default_rng(0)
    pk, _ = toy_keygen(n=6, q=31, sigma=0.8, rng=rng, m=12, small_secret=True)
    result = primal_attack(pk, block_size=3, time_budget_s=30)
    assert isinstance(result, PrimalResult)
    assert result.block_size == 3
    assert result.reduction_time >= 0
    assert result.status in {"success", "reduction_failed", "short_vector_not_secret"}
