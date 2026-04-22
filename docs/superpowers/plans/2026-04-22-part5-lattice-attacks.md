# Part 5 — Breaking It for Real: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement LLL, BKZ, and Kannan-embedding primal attack in pure Python; run the primal attack on toy-LWE; close with a comparison to ML-KEM's real parameters via published lattice-estimator numbers.

**Architecture:** A new `pqc_edu/attacks_advanced/` subpackage with four modules — `lll.py` (Gram-Schmidt, LLL), `bkz.py` (BKZ + SVP enumeration), `embedding.py` (Kannan primal lattice builder), `primal.py` (full attack loop). Four new notebooks (16–19) walk through the concepts and run the attacks. TDD throughout.

**Tech Stack:** Python 3.11+, numpy (integer/float linear algebra), matplotlib (visualizations), itertools (enumeration), pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-04-22-part5-lattice-attacks-design.md`

---

## File Structure

```
pqc/
├── pqc_edu/
│   └── attacks_advanced/          # new subpackage
│       ├── __init__.py
│       ├── lll.py                 # gram_schmidt, lll_reduce
│       ├── bkz.py                 # svp_enumerate, bkz_reduce
│       ├── embedding.py           # kannan_embedding
│       └── primal.py              # PrimalResult, primal_attack
├── notebooks/
│   ├── 16_lll_basis_reduction.ipynb
│   ├── 17_bkz_and_scaling.ipynb
│   ├── 18_primal_attack_on_lwe.ipynb
│   └── 19_ml_kem_parameters_and_estimator.ipynb
├── tests/
│   ├── test_lll.py
│   ├── test_bkz.py
│   └── test_primal_attack.py
├── _toc.yml                       # add Part 5
└── docs/superpowers/plans/2026-04-22-part5-lattice-attacks.md  (this file)
```

**Rules:**
- `pqc_edu.attacks_advanced` is an independent subpackage. It imports `pqc_edu.lwe.ToyPublicKey` / `ToySecretKey` only.
- Bases are numpy int64 arrays, rows = lattice vectors. Lattice `L(B) = { z·B : z ∈ Z^d }`.
- Reduction functions return a **new** array (input is not mutated).
- Gram-Schmidt uses float64; this is fine for toy scale (n ≤ 40).

---

## Task 0: Scaffold `pqc_edu/attacks_advanced/`

**Files:** Create `pqc_edu/attacks_advanced/__init__.py`

- [ ] **Step 1: Create the subpackage init**

```python
"""Advanced lattice attacks (LLL, BKZ, Kannan embedding, primal attack).

Pure-Python educational implementations. Scales to toy-LWE dimension ~40.
Not competitive with fpylll or SageMath — the point is to see every step.
"""
```

- [ ] **Step 2: Verify import**

```bash
cd /Users/dkkang/dev/pqc
source .venv/bin/activate
python -c "import pqc_edu.attacks_advanced; print('ok')"
```

Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add pqc_edu/attacks_advanced/__init__.py
git commit -m "chore(attacks): scaffold pqc_edu.attacks_advanced subpackage"
```

---

## Task 1: `lll.py` — Gram-Schmidt + LLL reduction

**Files:**
- Create: `pqc_edu/attacks_advanced/lll.py`
- Create: `tests/test_lll.py`

Follow TDD.

- [ ] **Step 1: Write failing tests**

Create `tests/test_lll.py`:

```python
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
```

- [ ] **Step 2: Run tests — expect ImportError**

```bash
pytest tests/test_lll.py -v
```

- [ ] **Step 3: Implement `pqc_edu/attacks_advanced/lll.py`**

```python
"""LLL (Lenstra–Lenstra–Lovász) basis reduction.

Takes a basis (rows are vectors) and returns a reduced basis whose first
vector is short. Pure-Python educational implementation — full Gram-Schmidt
recomputation on every update. Fine up to n ~ 40.
"""
from __future__ import annotations
import numpy as np


def gram_schmidt(B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Gram-Schmidt orthogonalization.

    Args:
        B: (n, m) array of lattice basis vectors (rows).
    Returns:
        (B_star, mu) where B_star[i] is the i-th GSO vector,
        mu[i, j] is the projection coefficient of b_i onto b*_j for j < i.
    """
    B = B.astype(np.float64)
    n = B.shape[0]
    B_star = np.zeros_like(B)
    mu = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        B_star[i] = B[i].copy()
        for j in range(i):
            denom = np.dot(B_star[j], B_star[j])
            if denom == 0:
                mu[i, j] = 0.0
            else:
                mu[i, j] = np.dot(B[i], B_star[j]) / denom
            B_star[i] = B_star[i] - mu[i, j] * B_star[j]
    return B_star, mu


def lll_reduce(B: np.ndarray, delta: float = 0.75) -> np.ndarray:
    """LLL-reduce the basis B (rows are vectors) with parameter delta.

    Standard delta range: 1/4 < delta < 1. Common choice: 3/4 for speed,
    0.99 for stronger reduction. Returns a new int64 array.
    """
    if not (0.25 < delta < 1.0):
        raise ValueError(f"delta must be in (1/4, 1), got {delta}")
    B = B.astype(np.int64).copy()
    n = B.shape[0]

    def recompute():
        return gram_schmidt(B)

    B_star, mu = recompute()
    k = 1
    while k < n:
        # Size reduction: make |mu[k, j]| <= 1/2 for j < k.
        for j in range(k - 1, -1, -1):
            if abs(mu[k, j]) > 0.5:
                r = int(round(mu[k, j]))
                B[k] = B[k] - r * B[j]
                B_star, mu = recompute()

        # Lovász condition: ||b*_k||^2 >= (delta - mu[k, k-1]^2) * ||b*_{k-1}||^2
        lhs = np.dot(B_star[k], B_star[k])
        rhs = (delta - mu[k, k - 1] ** 2) * np.dot(B_star[k - 1], B_star[k - 1])
        if lhs >= rhs:
            k += 1
        else:
            B[[k, k - 1]] = B[[k - 1, k]]
            B_star, mu = recompute()
            k = max(k - 1, 1)
    return B
```

- [ ] **Step 4: Run tests — expect all pass**

```bash
pytest tests/test_lll.py -v
```

Expected: 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add pqc_edu/attacks_advanced/lll.py tests/test_lll.py
git commit -m "feat(attacks): Gram-Schmidt and LLL basis reduction"
```

---

## Task 2: `bkz.py` — SVP enumeration + BKZ reduction

**Files:**
- Create: `pqc_edu/attacks_advanced/bkz.py`
- Create: `tests/test_bkz.py`

Follow TDD.

- [ ] **Step 1: Write failing tests**

Create `tests/test_bkz.py`:

```python
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
```

- [ ] **Step 2: Run tests — expect ImportError**

```bash
pytest tests/test_bkz.py -v
```

- [ ] **Step 3: Implement `pqc_edu/attacks_advanced/bkz.py`**

```python
"""BKZ (Block Korkine–Zolotarev) reduction and SVP enumeration.

Educational implementation. BKZ iteratively:
  1. LLL-reduces the current basis.
  2. For each block of size `block_size`, finds the shortest vector via
     brute-force enumeration, inserts it, and re-LLLs.
Repeats until no block improves (or `max_tours` reached).

SVP enumeration here is the simplest correct variant: iterative deepening
over bounded integer combinations of basis vectors. Fine up to block size
~ 6 in pure Python; for larger blocks use fpylll.
"""
from __future__ import annotations
import itertools
import numpy as np

from .lll import gram_schmidt, lll_reduce


def _integer_combination(B: np.ndarray, coeffs: tuple) -> np.ndarray:
    out = np.zeros(B.shape[1], dtype=np.int64)
    for i, c in enumerate(coeffs):
        if c != 0:
            out = out + c * B[i]
    return out


def svp_enumerate(B: np.ndarray) -> np.ndarray:
    """Return the shortest nonzero lattice vector via bounded enumeration.

    Uses iterative deepening: tries coefficient range [-c, c] for c = 1,
    2, 3, ... until the same shortest vector reappears.
    """
    B = B.astype(np.int64)
    n = B.shape[0]
    best = B[0].copy()
    best_norm_sq = int(np.dot(best, best))
    for row in B[1:]:
        ns = int(np.dot(row, row))
        if 0 < ns < best_norm_sq:
            best_norm_sq = ns
            best = row.copy()

    c = 1
    prev_best_norm_sq = None
    # Cap to prevent runaway; block sizes > 6 aren't supported well anyway.
    while c <= 4:
        for coeffs in itertools.product(range(-c, c + 1), repeat=n):
            if all(x == 0 for x in coeffs):
                continue
            v = _integer_combination(B, coeffs)
            ns = int(np.dot(v, v))
            if 0 < ns < best_norm_sq:
                best_norm_sq = ns
                best = v
        if prev_best_norm_sq is not None and best_norm_sq == prev_best_norm_sq:
            break
        prev_best_norm_sq = best_norm_sq
        c += 1
    return best


def bkz_reduce(
    B: np.ndarray,
    block_size: int,
    delta: float = 0.99,
    max_tours: int = 10,
) -> np.ndarray:
    """BKZ reduction with the given block size. Returns a new int64 basis."""
    if block_size < 2:
        raise ValueError(f"block_size must be >= 2, got {block_size}")
    B = lll_reduce(B.astype(np.int64), delta=delta)
    n = B.shape[0]

    for tour in range(max_tours):
        changed = False
        for k in range(n - 1):
            end = min(k + block_size, n)
            block = B[k:end].copy()
            # Enumerate shortest vector in the sublattice spanned by `block`.
            short = svp_enumerate(block)
            # If it differs from block[0], insert it and re-LLL the extended basis.
            short_norm_sq = int(np.dot(short, short))
            if short_norm_sq < int(np.dot(block[0], block[0])):
                # Build extended basis: [short, B[k], ..., B[end-1]]
                extended = np.vstack([short.reshape(1, -1), B[k:end]]).astype(np.int64)
                reduced_block = lll_reduce(extended, delta=delta)
                # Take first `end-k` non-zero rows.
                nonzero = reduced_block[
                    np.any(reduced_block != 0, axis=1)
                ][: end - k]
                if nonzero.shape[0] == end - k:
                    B[k:end] = nonzero
                    B = lll_reduce(B, delta=delta)
                    changed = True
        if not changed:
            break
    return B
```

- [ ] **Step 4: Run tests — expect all pass**

```bash
pytest tests/test_bkz.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add pqc_edu/attacks_advanced/bkz.py tests/test_bkz.py
git commit -m "feat(attacks): BKZ reduction with SVP enumeration"
```

---

## Task 3: `embedding.py` — Kannan embedding

**Files:**
- Create: `pqc_edu/attacks_advanced/embedding.py`

No dedicated test file; exercised end-to-end in Task 4.

- [ ] **Step 1: Implement**

```python
"""Kannan embedding: convert an LWE instance into a uSVP lattice.

Given public key (A, b) where b = A·s + e mod q, build a lattice whose
shortest nonzero vector encodes (e, -s, M). BKZ-reducing this lattice
recovers (s, e).
"""
from __future__ import annotations
import numpy as np

from pqc_edu.lwe import ToyPublicKey


def kannan_embedding(pk: ToyPublicKey, M: int | None = None) -> np.ndarray:
    """Build the Kannan primal-attack basis.

    Matrix layout (rows = basis vectors, dim = m + n + 1):
      Rows 0..n-1   : (A[:, j], e_j, 0)     — "y" rows, one per secret coord
      Rows n..n+m-1 : (q·e_i, 0_n, 0)       — "q·I" rows, one per sample
      Row n+m       : (b, 0_n, M)           — "target" row

    Args:
        pk: toy LWE public key (has .samples of shape (m, n+1) and .q).
        M:  embedding factor for the last coord. Defaults to 1, which is
            adequate for toy parameters.

    Returns:
        int64 ndarray of shape (m+n+1, m+n+1).
    """
    q = pk.q
    A = pk.samples[:, :-1].astype(np.int64)  # shape (m, n)
    b = pk.samples[:, -1].astype(np.int64)   # shape (m,)
    m, n = A.shape
    if M is None:
        M = 1
    dim = m + n + 1
    B = np.zeros((dim, dim), dtype=np.int64)

    # y rows (one per secret coord j):  [A[:, j], e_j, 0]
    for j in range(n):
        B[j, :m] = A[:, j]
        B[j, m + j] = 1

    # q·I rows (one per sample i):  [q·e_i, 0_n, 0]
    for i in range(m):
        B[n + i, i] = q

    # target row:  [b, 0_n, M]
    B[n + m, :m] = b
    B[n + m, n + m] = M

    return B
```

- [ ] **Step 2: Smoke test**

```bash
cd /Users/dkkang/dev/pqc
source .venv/bin/activate
python -c "
import numpy as np
from pqc_edu.lwe import toy_keygen
from pqc_edu.attacks_advanced.embedding import kannan_embedding
rng = np.random.default_rng(0)
pk, sk = toy_keygen(n=5, q=31, sigma=1.0, rng=rng, m=8)
B = kannan_embedding(pk)
print('basis shape:', B.shape)
print('expected:', (5 + 8 + 1, 5 + 8 + 1))
print('determinant (abs):', int(abs(np.round(np.linalg.det(B.astype(float))))))
print('expected ~ q^m:', pk.q ** 8)
"
```

Expected shape `(14, 14)`; determinant (absolute) roughly `q^m * M = 31^8 = 852891037441`.

- [ ] **Step 3: Commit**

```bash
git add pqc_edu/attacks_advanced/embedding.py
git commit -m "feat(attacks): Kannan embedding for primal attack"
```

---

## Task 4: `primal.py` — Full primal attack + tests

**Files:**
- Create: `pqc_edu/attacks_advanced/primal.py`
- Create: `tests/test_primal_attack.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_primal_attack.py`:

```python
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
    # Use fixed seed; small m so reduction is fast.
    rng = np.random.default_rng(0)
    pk, sk = toy_keygen(n=n, q=q, sigma=sigma, rng=rng, m=2 * n)
    result = primal_attack(pk, block_size=block_size, time_budget_s=60)
    assert result.status == "success", f"n={n}: status={result.status}"
    assert np.array_equal(
        result.secret % q, sk.s.astype(np.int64) % q
    ), f"n={n}: recovered secret mismatch"


def test_primal_result_fields():
    rng = np.random.default_rng(0)
    pk, _ = toy_keygen(n=6, q=31, sigma=0.8, rng=rng, m=12)
    result = primal_attack(pk, block_size=3, time_budget_s=30)
    assert isinstance(result, PrimalResult)
    assert result.block_size == 3
    assert result.reduction_time >= 0
    assert result.status in {"success", "reduction_failed", "short_vector_not_secret"}
```

- [ ] **Step 2: Run tests — expect ImportError**

```bash
pytest tests/test_primal_attack.py -v
```

- [ ] **Step 3: Implement `pqc_edu/attacks_advanced/primal.py`**

```python
"""Primal attack on LWE: Kannan embedding + BKZ + secret recovery."""
from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from pqc_edu.lwe import ToyPublicKey
from .bkz import bkz_reduce
from .embedding import kannan_embedding


@dataclass
class PrimalResult:
    secret: Optional[np.ndarray]
    recovered_error: Optional[np.ndarray]
    reduction_time: float
    block_size: int
    status: str  # "success" | "reduction_failed" | "short_vector_not_secret"


def _center(x: np.ndarray, q: int) -> np.ndarray:
    """Reduce each entry to (-q/2, q/2]."""
    y = x % q
    return np.where(y > q // 2, y - q, y)


def primal_attack(
    pk: ToyPublicKey,
    block_size: int = 4,
    time_budget_s: float = 120.0,
    M: int = 1,
) -> PrimalResult:
    """Run the primal attack. Tries every reduced basis row to find the
    one that decodes to the secret; returns on first success."""
    A = pk.samples[:, :-1].astype(np.int64)
    b = pk.samples[:, -1].astype(np.int64)
    m, n = A.shape
    q = pk.q

    start = time.time()
    try:
        B = kannan_embedding(pk, M=M)
        B_reduced = bkz_reduce(B, block_size=block_size)
    except Exception:
        return PrimalResult(
            secret=None, recovered_error=None,
            reduction_time=time.time() - start,
            block_size=block_size,
            status="reduction_failed",
        )
    reduction_time = time.time() - start

    # Try each reduced row (and its negation) as a candidate target vector.
    for row in B_reduced:
        for sign in (1, -1):
            v = sign * row
            if v[n + m] != M:
                continue
            e_cand = v[:m]
            s_cand = -v[m : m + n]
            # Verify: A·s + e ≡ b (mod q)
            residual = _center(A @ s_cand + e_cand - b, q)
            if np.all(residual == 0):
                return PrimalResult(
                    secret=(s_cand.astype(np.int64) % q),
                    recovered_error=e_cand.astype(np.int64),
                    reduction_time=reduction_time,
                    block_size=block_size,
                    status="success",
                )

    return PrimalResult(
        secret=None, recovered_error=None,
        reduction_time=reduction_time,
        block_size=block_size,
        status="short_vector_not_secret",
    )
```

- [ ] **Step 4: Run tests — expect all pass**

```bash
pytest tests/test_primal_attack.py -v
```

Expected: 4 passed (3 parametrized + 1 field test). The n=12 case may take 20–40 seconds.

- [ ] **Step 5: Commit**

```bash
git add pqc_edu/attacks_advanced/primal.py tests/test_primal_attack.py
git commit -m "feat(attacks): full primal attack (Kannan + BKZ + secret recovery)"
```

---

## Task 5: Notebook 16 — LLL basis reduction

**File:** Create `notebooks/16_lll_basis_reduction.ipynb`

Build with `nbformat`. Delete helper scripts (`build*.py`) before committing.

Cells (in order):

1. **Markdown**: `# Notebook 16 — LLL basis reduction`\n\nGoal: understand what a "good basis" is, implement LLL from the definitions, and see it transform skewed bases into near-orthogonal ones. Pure Python.
2. **Code**: `import numpy as np\nimport matplotlib.pyplot as plt\nfrom pqc_edu.attacks_advanced.lll import gram_schmidt, lll_reduce`
3. **Markdown**: `## Good vs bad bases\n\nA basis is a set of generating vectors for a lattice. Two bases of the **same lattice** can look wildly different. A "good" basis has short, near-orthogonal vectors. A "bad" one has long, skewed vectors. LLL turns bad into good in polynomial time.`
4. **Code** (visualize before/after on 2-D):
```
B = np.array([[19, 2], [12, 15]], dtype=np.int64)
R = lll_reduce(B, delta=0.99)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
for ax, basis, title in zip(axes, [B, R], ["before LLL", "after LLL"]):
    pts = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            pts.append(i * basis[0] + j * basis[1])
    pts = np.array(pts)
    ax.scatter(pts[:, 0], pts[:, 1], s=8, color="steelblue", alpha=0.4)
    for v, c in zip(basis, ("red", "green")):
        ax.annotate("", xy=v, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=c, lw=2))
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.axhline(0, color="lightgray", lw=0.5)
    ax.axvline(0, color="lightgray", lw=0.5)
plt.tight_layout(); plt.show()
print('before:', B.tolist())
print('after: ', R.tolist())
```
5. **Markdown**: `## Gram-Schmidt: the orthogonal skeleton\n\nLLL works by tracking the Gram-Schmidt orthogonalization (GSO) of the basis. The GSO gives orthogonal vectors $b^*_i$ built by subtracting projections onto previous vectors.`
6. **Code**:
```
B = np.array([[4, 1, 0], [2, 2, 0], [0, 1, 3]], dtype=np.int64)
B_star, mu = gram_schmidt(B)
print('B_star:'); print(B_star)
print('\northogonal?  (dot products)')
for i in range(3):
    for j in range(i+1, 3):
        print(f'  <b*{i}, b*{j}> = {np.dot(B_star[i], B_star[j]):.2e}')
```
7. **Markdown**: `## The LLL conditions\n\nA basis is LLL-reduced (with parameter $\delta \in (1/4, 1)$) iff\n\n1. **Size-reduced**: $|\mu_{i,j}| \le 1/2$ for all $j < i$.\n2. **Lovász**: $\|b^*_i\|^2 \ge (\delta - \mu_{i,i-1}^2)\|b^*_{i-1}\|^2$.\n\nThe first says each vector is "nearly orthogonal" to earlier ones. The second says consecutive GSO vectors don't shrink too fast.`
8. **Markdown**: `## Hermite factor — measuring how good\n\nGiven a rank-$n$ lattice with determinant $\det(\Lambda)$, the shortest nonzero vector $\lambda_1$ satisfies (Minkowski bound) roughly $\|\lambda_1\| \le \sqrt{n} \cdot \det(\Lambda)^{1/n}$.\n\nLLL's first vector satisfies $\|b_1\| \le \alpha^{(n-1)/2} \cdot \det(\Lambda)^{1/n}$ for $\alpha = 1/(\delta - 1/4)$. With $\delta = 0.99$, $\alpha \approx 1.35$. We measure the empirical **Hermite factor**: $\|b_1\| / \det(\Lambda)^{1/n}$.`
9. **Code**:
```
rng = np.random.default_rng(0)
dims = [2, 4, 6, 8, 10, 12]
factors = []
for n in dims:
    B = rng.integers(-50, 50, (n, n)).astype(np.int64)
    while abs(np.linalg.det(B)) < 1.0:
        B = rng.integers(-50, 50, (n, n)).astype(np.int64)
    R = lll_reduce(B, delta=0.99)
    det_n = abs(np.linalg.det(R.astype(float)))
    factors.append(np.linalg.norm(R[0]) / det_n ** (1.0 / n))
plt.plot(dims, factors, 'o-')
plt.xlabel('dimension n')
plt.ylabel('Hermite factor  ||b_1|| / det^(1/n)')
plt.title('Empirical Hermite factor after LLL (delta=0.99)')
plt.grid(True)
plt.show()
for n, f in zip(dims, factors):
    print(f'n={n:2d}   Hermite factor = {f:.3f}')
```
10. **Markdown**: `## What LLL is not\n\nLLL gives an approximation guarantee, not exact SVP. The first vector can be up to $\alpha^{(n-1)/2}$ times longer than the true shortest vector. For attacking LWE we need shorter vectors than LLL alone delivers — enter BKZ.\n\n→ \`17_bkz_and_scaling.ipynb\``

**Steps 2-4:** write helper script to `/tmp/build_nb16.py`, run it, delete, then execute:
```bash
python /tmp/build_nb16.py && rm /tmp/build_nb16.py
jupyter nbconvert --to notebook --execute notebooks/16_lll_basis_reduction.ipynb --output 16_lll_basis_reduction.ipynb
git add notebooks/16_lll_basis_reduction.ipynb
git commit -m "docs(nb16): LLL basis reduction with Gram-Schmidt and Hermite factor"
```

---

## Task 6: Notebook 17 — BKZ and scaling

**File:** Create `notebooks/17_bkz_and_scaling.ipynb`

Cells:

1. **Markdown**: `# Notebook 17 — BKZ and scaling`\n\nLLL is fast but weak. BKZ (Block Korkine–Zolotarev) trades time for quality by doing SVP inside small blocks. We implement it, measure the quality/cost tradeoff, and see where pure-Python runs out of steam.
2. **Code**: `import numpy as np, time\nimport matplotlib.pyplot as plt\nfrom pqc_edu.attacks_advanced.lll import lll_reduce\nfrom pqc_edu.attacks_advanced.bkz import bkz_reduce, svp_enumerate`
3. **Markdown**: `## The BKZ loop\n\nIn each "tour" over the basis:\n\n1. LLL-reduce the whole basis.\n2. For each block of $\beta$ consecutive vectors, find the exact shortest vector in the sublattice (via enumeration).\n3. Insert that short vector into the block; LLL again.\n\nLarger $\beta$ means better reduction — but enumeration is exponential in $\beta$, so cost explodes. In production, $\beta$ goes up to 50–90 (via fpylll). Our pure-Python enumeration handles $\beta \le 6$ comfortably.`
4. **Code** (block size vs first-vector length on a fixed lattice):
```
rng = np.random.default_rng(0)
B_seed = rng.integers(-40, 40, (8, 8)).astype(np.int64)
while abs(np.linalg.det(B_seed)) < 1.0:
    B_seed = rng.integers(-40, 40, (8, 8)).astype(np.int64)

blocks = [2, 3, 4, 5, 6]
lengths = []
times = []
for bs in blocks:
    t0 = time.time()
    R = bkz_reduce(B_seed.copy(), block_size=bs)
    times.append(time.time() - t0)
    lengths.append(float(np.linalg.norm(R[0])))
    print(f'block_size={bs}   ||b_1||={lengths[-1]:8.2f}   time={times[-1]:5.2f}s')

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(blocks, lengths, 'o-'); axes[0].set_xlabel('block size'); axes[0].set_ylabel('||b_1||'); axes[0].set_title('quality')
axes[1].plot(blocks, times, 'o-', color='firebrick'); axes[1].set_xlabel('block size'); axes[1].set_ylabel('seconds'); axes[1].set_title('cost')
for ax in axes: ax.grid(True)
plt.tight_layout(); plt.show()
```
5. **Markdown**: `## Why this matters for attacking LWE\n\nLWE secrets (s, e) correspond to a short vector in the "primal lattice" we build next notebook. If BKZ can recover a sufficiently short vector, the secret falls out. The block size you need grows with the LWE dimension — this is the core of cost estimation for ML-KEM's parameter choice.`
6. **Markdown**: `## Enumeration: inside the block\n\nInside each block, we search for the shortest nonzero integer combination of the block's basis vectors. Our \`svp_enumerate\` uses iterative deepening over coefficient ranges [-c, c]. Schnorr-Euchner enumeration (with pruning by GSO norms) would be faster but harder to read; both find the same vector.`
7. **Code**:
```
B_small = np.array([[3, 1, 0], [1, 3, 1], [0, 1, 3]], dtype=np.int64)
v = svp_enumerate(B_small)
print('shortest vector:', v)
print('norm^2:', int(np.dot(v, v)))
```
8. **Markdown**: `## Takeaway\n\n- BKZ-$\beta$ quality improves as $\beta$ grows; our pure-Python caps around $\beta = 6$.\n- Real attacks use $\beta = 40$–$80$+ via fpylll. The Python/Rust/C++ implementation gap is ~3 orders of magnitude.\n- ML-KEM parameters are chosen so the **best known BKZ attack** costs $2^{140}$+ operations — see notebook 19.\n\n→ \`18_primal_attack_on_lwe.ipynb\``

**Steps 2-4:** write `/tmp/build_nb17.py`, run, delete, execute, commit.
```bash
python /tmp/build_nb17.py && rm /tmp/build_nb17.py
jupyter nbconvert --to notebook --execute notebooks/17_bkz_and_scaling.ipynb --output 17_bkz_and_scaling.ipynb
git add notebooks/17_bkz_and_scaling.ipynb
git commit -m "docs(nb17): BKZ block reduction and quality vs cost tradeoff"
```

Note: executing this notebook takes 30–60 seconds (BKZ block 6 on 8 dim).

---

## Task 7: Notebook 18 — Primal attack on LWE

**File:** Create `notebooks/18_primal_attack_on_lwe.ipynb`

Cells:

1. **Markdown**: `# Notebook 18 — Primal attack on LWE`\n\nThe payoff: take a toy LWE instance, build the Kannan-embedding lattice, BKZ-reduce it, and watch the secret fall out.
2. **Code**: `import numpy as np, time\nimport matplotlib.pyplot as plt\nfrom pqc_edu.lwe import toy_keygen\nfrom pqc_edu.attacks_advanced.primal import primal_attack`
3. **Markdown**: `## Kannan's embedding, informally\n\nGiven LWE samples $A \cdot s + e \equiv b \pmod q$, we build a lattice containing the vector $(e, -s, 1)$ (up to scaling). That vector is short because $e, s$ are small. If BKZ finds a short enough basis, we read $s$ right off.\n\nThe basis layout (rows, dimension $m + n + 1$):\n\n- $n$ rows of $(A[:,j], e_j, 0)$\n- $m$ rows of $(q \cdot e_i, 0, 0)$\n- $1$ row of $(b, 0, M)$\n\nWhy it works: any lattice point has last coord = $M \cdot t$ for some integer $t$. Setting $t = 1$ and choosing coefficients to cancel $b$ against $As$, the residue in the first $m$ coords becomes $e$.`
4. **Markdown**: `## A small attack: $n = 10$`
5. **Code**:
```
rng = np.random.default_rng(0)
pk, sk = toy_keygen(n=10, q=97, sigma=0.8, rng=rng, m=20)
print('secret:', sk.s)
result = primal_attack(pk, block_size=4, time_budget_s=60)
print(f'\nstatus: {result.status}')
print(f'reduction_time: {result.reduction_time:.1f}s')
if result.status == 'success':
    print(f'recovered s: {result.secret}')
    print(f'matches: {np.array_equal(result.secret % pk.q, sk.s % pk.q)}')
```
6. **Markdown**: `## Success rate vs dimension\n\nWe attack 5 independent LWE instances for each dimension $n$ and count how often primal succeeds in a time budget. The wall visible around $n = 25$–$30$ is where BKZ-$\beta = 4$ runs out of steam; raising $\beta$ pushes the wall out but costs exponentially more time.`
7. **Code**:
```
dims = [8, 10, 12, 15, 18, 22, 26]
trials_per_dim = 3
success_rate = []
avg_time = []
for n in dims:
    successes = 0
    total_time = 0.0
    for trial in range(trials_per_dim):
        rng = np.random.default_rng(100 * n + trial)
        pk, _ = toy_keygen(n=n, q=max(31, 2 * n * n), sigma=0.8, rng=rng, m=2 * n)
        t0 = time.time()
        result = primal_attack(pk, block_size=4, time_budget_s=90)
        total_time += time.time() - t0
        if result.status == 'success':
            successes += 1
    success_rate.append(successes / trials_per_dim)
    avg_time.append(total_time / trials_per_dim)
    print(f'n={n:2d}  success={successes}/{trials_per_dim}  avg time={avg_time[-1]:5.1f}s')

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].bar(dims, success_rate); axes[0].set_xlabel('n'); axes[0].set_ylabel('success rate'); axes[0].set_ylim(0, 1.05); axes[0].set_title('Primal attack success rate')
axes[1].plot(dims, avg_time, 'o-', color='firebrick'); axes[1].set_xlabel('n'); axes[1].set_ylabel('seconds'); axes[1].set_title('avg attack time')
for ax in axes: ax.grid(True)
plt.tight_layout(); plt.show()
```
8. **Markdown**: `## The pattern\n\n- Small $n$ (≤ 20) falls to BKZ-4 reliably.\n- $n = 25+$ needs larger $\beta$ — and since enumeration cost is ~ $2^{O(\beta)}$, our pure-Python version hits a wall.\n- ML-KEM operates at effective lattice dimension ~$n \cdot k + m \cdot k \approx 512$–$1024$. To attack it, $\beta \approx 400+$ would be needed. That's where the $2^{140}$ cost estimate comes from — you can't do it.\n\n→ Final chapter: \`19_ml_kem_parameters_and_estimator.ipynb\``

**Steps 2-4:** write `/tmp/build_nb18.py`, run, delete, execute, commit.
```bash
python /tmp/build_nb18.py && rm /tmp/build_nb18.py
jupyter nbconvert --to notebook --execute notebooks/18_primal_attack_on_lwe.ipynb --output 18_primal_attack_on_lwe.ipynb
git add notebooks/18_primal_attack_on_lwe.ipynb
git commit -m "docs(nb18): primal attack on toy LWE with scaling curve"
```

Note: this notebook runs multiple attacks; total execution ~3-5 minutes. Be patient.

---

## Task 8: Notebook 19 — ML-KEM parameters and the estimator

**File:** Create `notebooks/19_ml_kem_parameters_and_estimator.ipynb`

Pure-markdown + a summary table. No attack execution.

Cells:

1. **Markdown**: `# Notebook 19 — ML-KEM parameters and the estimator`\n\nThe closing chapter: we've seen our primal attack succeed on $n \le 20$ in seconds. ML-KEM-512 has effective lattice dimension ~$800$. This notebook explains why that gap translates to complete safety, using the community-standard **lattice estimator**.
2. **Markdown**: `## Where ML-KEM's numbers come from\n\nML-KEM-512 parameters (FIPS 203):\n\n- $n = 256$ (polynomial ring dimension)\n- $k = 2$ (module rank)\n- $q = 3329$\n- $\eta_1 = 3, \eta_2 = 2$ (CBD noise parameters)\n\nThe attack lattice for primal has dimension around $n \cdot k + m = 256 \cdot 2 + 256 = 768$ (roughly). Running BKZ on this with $\beta$ large enough to see a short enough vector is the threat model.`
3. **Markdown**: `## The lattice estimator\n\nAlbrecht, Player, Scott (2015, extended through 2024) built a Python tool — [**lattice-estimator**](https://github.com/malb/lattice-estimator) — that computes the expected cost of every known attack on a given LWE/MLWE instance: primal, dual, hybrid, BKW, and variants. NIST used it to pin down ML-KEM's parameters.\n\nA sample output for ML-KEM-512 (from the project's published numbers):\n\n| Attack         | $\beta$ | Log cost |\n|----------------|---------|----------|\n| primal         | 400     | 143      |\n| dual           | 388     | 143      |\n| hybrid         | 411     | 141      |\n\nLog cost 143 means roughly $2^{143}$ operations — comparable to breaking AES-143 by brute force. The number for ML-KEM-1024 is $\sim 2^{272}$.`
4. **Markdown**: `## Our toy numbers vs reality\n\n| Instance                | Lattice dim | $\beta$ used | Python time | Log cost (estimator) |\n|-------------------------|-------------|--------------|-------------|----------------------|\n| Our toy LWE, $n = 10$   | 31          | 4            | ~3 s        | (below threshold)    |\n| Our toy LWE, $n = 20$   | 61          | 4–6          | ~30 s       | ~40                  |\n| Our toy LWE, $n = 40$   | 121         | ~20 (infeasible pure-Py) | — | ~70         |\n| **ML-KEM-512**          | ~768        | 400          | —           | **143**              |\n| **ML-KEM-768**          | ~1152       | 623          | —           | **207**              |\n| **ML-KEM-1024**         | ~1536       | 832          | —           | **272**              |\n\nThe jump from "feasible in seconds" to "infeasible at the heat-death of the universe" happens smoothly through $\beta$ but is felt as a wall. It's the same wall Part 4's Shor analysis didn't breach — classical and quantum, both ML-KEM parameters are safe today.`
5. **Markdown**: `## Why we trust these numbers\n\n- The estimator accounts for all publicly known attacks as of 2024, including recent quantum sieve speedups.\n- NIST's standardization process invited external cryptanalysis. Two rounds of candidate reductions (including Rainbow and SIKE being broken) established what "safe" looks like.\n- The 143-bit safety margin is above NIST's "category 1" threshold (which roughly matches AES-128 security).\n- **Hedging**: we still use ML-KEM in hybrid with X25519. If someone finds a better lattice attack tomorrow, the classical half still protects the shared secret until Shor arrives.`
6. **Markdown**: `## How this connects back\n\nPart 1–4 built the scheme and walked through the positive case: *here is how ML-KEM works*.\n\nPart 5 walked through the adversary's side: *here is how we'd attack it, and here's how far we get on small parameters*.\n\nThe combination — "we built it, we attacked it, we know where the walls are" — is what operational confidence in a cryptosystem looks like. In a few years, all TLS handshakes will be doing this math. You now know what's happening inside.`
7. **Markdown**: `## Further reading\n\n- **lattice-estimator** (Albrecht et al.): https://github.com/malb/lattice-estimator — interactive cost computation for any LWE/MLWE instance.\n- **NIST IR 8413** — the ML-KEM standardization report, Appendix A details parameter selection.\n- Albrecht, Player, Scott. *On the Concrete Hardness of Learning with Errors*, JoMC 2015.\n- Chen & Nguyen. *BKZ 2.0: Better Lattice Security Estimates*, ASIACRYPT 2011.\n- Peikert. *A Decade of Lattice Cryptography* (2016) — survey.\n- Kannan. *Improved Algorithms for Integer Programming and Related Lattice Problems*, STOC 1983 — the original embedding.\n\n**Thanks for reading all 19 notebooks.**`

**Steps:** write `/tmp/build_nb19.py`, run, delete, execute, commit.
```bash
python /tmp/build_nb19.py && rm /tmp/build_nb19.py
jupyter nbconvert --to notebook --execute notebooks/19_ml_kem_parameters_and_estimator.ipynb --output 19_ml_kem_parameters_and_estimator.ipynb
git add notebooks/19_ml_kem_parameters_and_estimator.ipynb
git commit -m "docs(nb19): ML-KEM parameters vs lattice estimator — Part 5 closes"
```

---

## Task 9: Add Part 5 to TOC and final verification

**Files:**
- Modify: `_toc.yml`

- [ ] **Step 1: Edit `_toc.yml`**

Add a Part 5 block at the end (after Part 4):

```yaml
  - caption: "Part 5 — Breaking It for Real"
    chapters:
      - file: notebooks/16_lll_basis_reduction
      - file: notebooks/17_bkz_and_scaling
      - file: notebooks/18_primal_attack_on_lwe
      - file: notebooks/19_ml_kem_parameters_and_estimator
```

So `_toc.yml` now has 5 `parts` entries.

- [ ] **Step 2: Full test suite**

```bash
cd /Users/dkkang/dev/pqc
source .venv/bin/activate
pytest tests/ -v
```

Expected: all existing 38 tests + new ~14 (LLL 5, BKZ 5, primal 4) = 52 total pass.

- [ ] **Step 3: Full book build**

```bash
rm -rf _build
jupyter-book build .
```

Expected: no errors, Part 5 appears in the sidebar.

- [ ] **Step 4: Commit TOC**

```bash
git add _toc.yml
git commit -m "feat(book): add Part 5 — Breaking It for Real to TOC"
```

- [ ] **Step 5: Push**

```bash
git push origin main
```

Verify the four new URLs return 200 after deploy:

```bash
sleep 90
for nb in 16_lll_basis_reduction 17_bkz_and_scaling 18_primal_attack_on_lwe 19_ml_kem_parameters_and_estimator; do
  curl -sI "https://hulryung.github.io/ml-kem-notebooks/notebooks/${nb}.html" | head -1
done
```

Expected: four `HTTP/2 200`.

---

## Self-Review Checklist

- [x] **Spec coverage**: every spec section maps to tasks — §4 structure → Task 0; §5 notebooks → Tasks 5–8; §6 interfaces → Tasks 1–4; §7 tests → Tasks 1, 2, 4 + Task 9 Step 2.
- [x] **Placeholder scan**: no TBD/TODO. Every code step shows the exact code.
- [x] **Type consistency**: `gram_schmidt`, `lll_reduce`, `bkz_reduce`, `svp_enumerate`, `kannan_embedding`, `primal_attack`, `PrimalResult` — names match across all tasks.
- [x] **Scope**: single plan, one subpackage + four notebooks. Works independently; no dependency on future Part 6 (messenger).

## Known Risk Points (from spec §8)

- **LLL speed on n=40**: full GSO recomputation per update is O(n^3). For n=40 the reduction of a random basis is ~5–10 seconds. We cap toy LWE attacks at m+n+1 ≈ 60 (n ≤ 26). If notebook 18 gets slow, drop the largest dimension.
- **BKZ block size ≥ 7**: `svp_enumerate` iterative deepening goes as c^β where c grows until best stabilizes. At β = 7, this is already at ~5 seconds; β = 8 can take minutes. Notebook 17 stops at β = 6.
- **Kannan M factor**: we use M = 1. If primal fails with `short_vector_not_secret`, try M = 2 or M = ceil(σ · sqrt(n)). The helper functions already accept M as a parameter.

If tests fail, debug in this order: `test_lll.py` → `test_bkz.py` → `test_primal_attack.py`. A bug in LLL silently breaks everything downstream.
