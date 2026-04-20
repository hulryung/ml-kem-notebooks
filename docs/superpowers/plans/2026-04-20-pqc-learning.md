# PQC Learning Project Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Jupyter-notebook curriculum that teaches lattice-based post-quantum cryptography by implementing toy LWE, ring/NTT machinery, and ML-KEM (FIPS 203) from scratch in Python.

**Architecture:** Two-layer structure. A reusable package `pqc_edu/` (single-responsibility modules: `lattice`, `lwe`, `attacks`, `polyring`, `ntt`, `sampling`, `ml_kem`, `params`) holds correct, tested logic. Nine linear notebooks (`01`–`09`) introduce each concept, derive it inline on small examples, then import the package for reuse. Pytest covers polyring/NTT/LWE/ML-KEM roundtrips.

**Tech Stack:** Python 3.11+, numpy (linear algebra, polynomial coefficients), jupyter (curriculum), matplotlib (visualizations), hashlib/secrets (SHAKE, randomness), pytest (tests). No external PQC libraries.

**Spec:** `docs/superpowers/specs/2026-04-20-pqc-learning-design.md`

---

## File Structure

```
pqc/
├── README.md
├── pyproject.toml
├── pqc_edu/
│   ├── __init__.py
│   ├── params.py            # MLKEMParams dataclass + 3 presets
│   ├── lattice.py           # 2D/3D lattice helpers + plots
│   ├── lwe.py               # toy LWE keygen/encrypt/decrypt (bit-level)
│   ├── attacks.py           # brute force + Gaussian elimination on toy LWE
│   ├── polyring.py          # Poly class for Z_q[x]/(x^n+1)
│   ├── ntt.py               # NTT/INTT over Z_3329[x]/(x^256+1)
│   ├── sampling.py          # SHAKE-based PRF, CBD, uniform rejection
│   └── ml_kem.py            # FIPS 203: K-PKE + ML-KEM.{KeyGen,Encaps,Decaps}
├── tests/
│   ├── test_polyring.py
│   ├── test_ntt.py
│   ├── test_lwe.py
│   └── test_ml_kem.py
├── notebooks/
│   ├── 01_lattice_intro.ipynb
│   ├── 02_toy_lwe.ipynb
│   ├── 03_attacking_toy_lwe.ipynb
│   ├── 04_polynomial_rings.ipynb
│   ├── 05_ring_lwe.ipynb
│   ├── 06_ml_kem_spec.ipynb
│   ├── 07_ml_kem_tests.ipynb
│   ├── 08_hybrid_kem.ipynb
│   └── 09_wrap_up.ipynb
└── docs/superpowers/
    ├── specs/2026-04-20-pqc-learning-design.md
    └── plans/2026-04-20-pqc-learning.md   (this file)
```

**Notebook/package boundary rule:** A concept is first derived inline in the notebook where it is introduced (so the student sees it built up). Any subsequent notebook that uses it imports from `pqc_edu`. No function body appears in two places.

---

## Task 0: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `README.md`
- Create: `pqc_edu/__init__.py`
- Create: `tests/__init__.py`
- Create: `.gitignore`

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "pqc-edu"
version = "0.1.0"
description = "Educational implementation of lattice-based post-quantum cryptography"
requires-python = ">=3.11"
dependencies = [
    "numpy>=1.26",
    "matplotlib>=3.8",
    "jupyter>=1.0",
    "cryptography>=42",  # only for X25519 in notebook 08
]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["pqc_edu*"]
```

- [ ] **Step 2: Write `README.md`**

```markdown
# PQC Learning Project

Jupyter notebooks that teach lattice-based post-quantum cryptography by implementing ML-KEM (Kyber / FIPS 203) from scratch.

## ⚠️ Educational Only
This implementation is not constant-time, not side-channel resistant, and has not been tested against NIST KATs. **Do not use for real encryption.**

## Setup
```
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Reading Order
1. `notebooks/01_lattice_intro.ipynb` — what is a lattice, why is it hard?
2. `notebooks/02_toy_lwe.ipynb` — the LWE problem on small numbers
3. `notebooks/03_attacking_toy_lwe.ipynb` — break small LWE ourselves
4. `notebooks/04_polynomial_rings.ipynb` — Z_q[x]/(x^n+1) and NTT
5. `notebooks/05_ring_lwe.ipynb` — LWE → Ring-LWE → MLWE
6. `notebooks/06_ml_kem_spec.ipynb` — FIPS 203 implementation
7. `notebooks/07_ml_kem_tests.ipynb` — roundtrip tests, size tables, benchmarks
8. `notebooks/08_hybrid_kem.ipynb` — X25519 + ML-KEM hybrid
9. `notebooks/09_wrap_up.ipynb` — summary and further reading

## Testing
```
pytest tests/ -v
```
```

- [ ] **Step 3: Write `.gitignore`**

```
.venv/
__pycache__/
*.pyc
.ipynb_checkpoints/
.pytest_cache/
*.egg-info/
dist/
build/
```

- [ ] **Step 4: Empty package init files**

Create `pqc_edu/__init__.py` and `tests/__init__.py` both with a single line:

```python
"""Educational PQC implementation — not for production use."""
```

(tests/__init__.py may be empty but keep a comment for discoverability.)

- [ ] **Step 5: Verify install**

```bash
cd /Users/dkkang/dev/pqc
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -c "import pqc_edu; print('ok')"
```

Expected: `ok` printed. `pytest tests/` reports "no tests ran" (0 collected) — that's fine, there are no tests yet.

- [ ] **Step 6: Initialize git and commit**

```bash
git init
git add pyproject.toml README.md .gitignore pqc_edu/ tests/ docs/
git commit -m "chore: project scaffolding for PQC learning project"
```

---

## Task 1: `pqc_edu/polyring.py` — Polynomial ring Z_q[x]/(x^n+1)

**Rationale:** We build the polynomial ring first (before LWE) even though the learner meets it in notebook 04. This lets us write correct, fast LWE attacks and removes sequencing risk. The ring module is self-contained and easy to test.

**Files:**
- Create: `pqc_edu/polyring.py`
- Create: `tests/test_polyring.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_polyring.py`:

```python
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
    # Compute x^N via iterated squaring using naive multiplication.
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
```

- [ ] **Step 2: Run tests — expect failure**

```bash
pytest tests/test_polyring.py -v
```

Expected: `ImportError: cannot import name 'Poly'` (module does not exist yet).

- [ ] **Step 3: Implement `pqc_edu/polyring.py`**

```python
"""Polynomial ring R_q = Z_q[x] / (x^n + 1).

The anti-cyclic identity x^n = -1 is what makes ML-KEM's arithmetic fast
once combined with the NTT (see pqc_edu.ntt).
"""
from __future__ import annotations
import numpy as np


class Poly:
    """Element of Z_q[x]/(x^n+1). Immutable by convention; operators return new objects."""

    __slots__ = ("coeffs", "q", "n")

    def __init__(self, coeffs, q: int):
        arr = np.asarray(coeffs, dtype=np.int64) % q
        self.coeffs = arr
        self.q = q
        self.n = len(arr)

    # -- arithmetic ----------------------------------------------------
    def __add__(self, other: "Poly") -> "Poly":
        self._check(other)
        return Poly((self.coeffs + other.coeffs) % self.q, self.q)

    def __sub__(self, other: "Poly") -> "Poly":
        self._check(other)
        return Poly((self.coeffs - other.coeffs) % self.q, self.q)

    def __neg__(self) -> "Poly":
        return Poly((-self.coeffs) % self.q, self.q)

    def __mul__(self, other: "Poly") -> "Poly":
        self._check(other)
        # Schoolbook convolution, then fold x^n = -1.
        n, q = self.n, self.q
        full = np.zeros(2 * n, dtype=np.int64)
        for i in range(n):
            if self.coeffs[i]:
                full[i : i + n] += self.coeffs[i] * other.coeffs
        # Fold: coefficients [n .. 2n-1] contribute with sign -1 to [0 .. n-1].
        out = (full[:n] - full[n:]) % q
        return Poly(out, q)

    # -- misc ----------------------------------------------------------
    def __eq__(self, other) -> bool:
        if not isinstance(other, Poly):
            return NotImplemented
        return self.q == other.q and np.array_equal(self.coeffs, other.coeffs)

    def __repr__(self) -> str:
        return f"Poly(n={self.n}, q={self.q}, coeffs={self.coeffs.tolist()[:4]}...)"

    def _check(self, other: "Poly") -> None:
        if self.n != other.n or self.q != other.q:
            raise ValueError("Poly operands must share (n, q)")

    @classmethod
    def zero(cls, n: int, q: int) -> "Poly":
        return cls(np.zeros(n, dtype=np.int64), q)

    @classmethod
    def from_int(cls, value: int, n: int, q: int) -> "Poly":
        p = cls.zero(n, q)
        p.coeffs[0] = value % q
        return p
```

- [ ] **Step 4: Run tests — expect pass**

```bash
pytest tests/test_polyring.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add pqc_edu/polyring.py tests/test_polyring.py
git commit -m "feat(polyring): implement Z_q[x]/(x^n+1) ring with tests"
```

---

## Task 2: `pqc_edu/ntt.py` — NTT for Kyber parameters

**Files:**
- Create: `pqc_edu/ntt.py`
- Create: `tests/test_ntt.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_ntt.py`:

```python
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
```

- [ ] **Step 2: Run — expect failure (module missing)**

```bash
pytest tests/test_ntt.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `pqc_edu/ntt.py`**

```python
"""Number-Theoretic Transform for R_q with q=3329, n=256.

This is the exact NTT that FIPS 203 (ML-KEM) uses. The primitive 256-th
root of unity mod 3329 is 17. Coefficients are held in bit-reversed
order in the NTT domain (matches FIPS 203 Algorithm 9 / 10).

Educational implementation: O(n log n) via the classic Cooley–Tukey
butterfly, written with numpy vectors only where it aids clarity.
"""
from __future__ import annotations
import numpy as np
from .polyring import Poly

Q = 3329
N = 256
ROOT = 17  # primitive 256-th root of unity mod 3329


def _bitrev7(i: int) -> int:
    r = 0
    for _ in range(7):
        r = (r << 1) | (i & 1)
        i >>= 1
    return r


# Precompute zetas in bit-reversed order: zeta[i] = ROOT^{bitrev7(i)} mod q
_ZETAS = np.array(
    [pow(ROOT, _bitrev7(i), Q) for i in range(128)], dtype=np.int64
)


def ntt(p: Poly) -> Poly:
    """Forward NTT. Returns a Poly whose coeffs live in the NTT domain."""
    assert p.n == N and p.q == Q
    a = p.coeffs.copy()
    k = 1
    length = 128
    while length >= 2:
        for start in range(0, N, 2 * length):
            zeta = _ZETAS[k]
            k += 1
            for j in range(start, start + length):
                t = (zeta * a[j + length]) % Q
                a[j + length] = (a[j] - t) % Q
                a[j] = (a[j] + t) % Q
        length //= 2
    return Poly(a, Q)


def intt(p: Poly) -> Poly:
    """Inverse NTT."""
    assert p.n == N and p.q == Q
    a = p.coeffs.copy()
    k = 127
    length = 2
    while length <= 128:
        for start in range(0, N, 2 * length):
            zeta = _ZETAS[k]
            k -= 1
            for j in range(start, start + length):
                t = a[j]
                a[j] = (t + a[j + length]) % Q
                a[j + length] = (zeta * (a[j + length] - t)) % Q
        length *= 2
    # Multiply by n^{-1} mod q (n=256, inverse = 3303 mod 3329).
    n_inv = pow(N, -1, Q)
    a = (a * n_inv) % Q
    return Poly(a, Q)


def pointwise_mul(a_ntt: Poly, b_ntt: Poly) -> Poly:
    """Base-case multiplication in the NTT domain.

    FIPS 203 uses degree-1 polynomial multiplications on 128 paired slots.
    For an educational version we collapse this to plain coefficient-wise
    multiplication plus the correction factor — see the notebook for the
    derivation. The end-to-end equivalence with schoolbook is checked by
    test_ntt_matches_schoolbook.
    """
    assert a_ntt.n == N and b_ntt.n == N and a_ntt.q == Q
    out = np.zeros(N, dtype=np.int64)
    for i in range(128):
        zeta = _ZETAS[64 + (i >> 1)] if False else _ZETAS[64 + (i // 2)]
        # We use the standard FIPS 203 pair-wise formula for (a0+a1 X)(b0+b1 X) mod (X^2 - zeta).
        a0 = int(a_ntt.coeffs[2 * i])
        a1 = int(a_ntt.coeffs[2 * i + 1])
        b0 = int(b_ntt.coeffs[2 * i])
        b1 = int(b_ntt.coeffs[2 * i + 1])
        # sign: zeta here is the right zeta for this pair per FIPS 203 Algorithm 11.
        gamma = pow(ROOT, 2 * _bitrev7(i) + 1, Q)
        out[2 * i] = (a0 * b0 + a1 * b1 * gamma) % Q
        out[2 * i + 1] = (a0 * b1 + a1 * b0) % Q
    return Poly(out, Q)


def poly_mul_ntt(a: Poly, b: Poly) -> Poly:
    """Multiply two polynomials via the NTT. Equivalent to `a * b`."""
    return intt(pointwise_mul(ntt(a), ntt(b)))
```

- [ ] **Step 4: Run tests — expect pass**

```bash
pytest tests/test_ntt.py -v
```

Expected: 3 passed. (If the pointwise_mul formula mismatches schoolbook, the test will fail — fix gamma indexing per FIPS 203 Algorithm 11 before proceeding.)

- [ ] **Step 5: Commit**

```bash
git add pqc_edu/ntt.py tests/test_ntt.py
git commit -m "feat(ntt): Cooley-Tukey NTT for Kyber parameters with tests"
```

---

## Task 3: `pqc_edu/lattice.py` — Lattice helpers for visualization

**Files:**
- Create: `pqc_edu/lattice.py`

No dedicated test file: this module is helper/visualization code and is exercised by notebook 01. We still keep all logic in the package so notebooks don't own `def`-ed math.

- [ ] **Step 1: Implement `pqc_edu/lattice.py`**

```python
"""2-D and 3-D lattice helpers used by the first notebook.

A lattice L is the set {B @ z : z in Z^d} for a basis matrix B. This
module provides generators for lattice points and a few plotting
helpers built on matplotlib.
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from itertools import product


def lattice_points(basis: np.ndarray, radius: int):
    """Yield lattice points B @ z for z with |z_i| <= radius."""
    d = basis.shape[0]
    for z in product(range(-radius, radius + 1), repeat=d):
        yield basis @ np.array(z)


def plot_lattice_2d(basis: np.ndarray, radius: int = 5, target: np.ndarray | None = None, ax=None):
    """Scatter plot a 2-D lattice. If `target` is given, also show the closest lattice point."""
    assert basis.shape == (2, 2)
    pts = np.array(list(lattice_points(basis, radius)))
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(pts[:, 0], pts[:, 1], s=8, color="steelblue")
    ax.axhline(0, color="lightgray", lw=0.5)
    ax.axvline(0, color="lightgray", lw=0.5)
    # Draw basis vectors.
    for v, color in zip(basis.T, ("red", "green")):
        ax.annotate("", xy=v, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=color))
    if target is not None:
        dists = np.linalg.norm(pts - target, axis=1)
        closest = pts[np.argmin(dists)]
        ax.scatter(*target, marker="*", color="orange", s=100, label="target")
        ax.scatter(*closest, marker="x", color="red", s=80, label="closest lattice point")
        ax.legend()
    ax.set_aspect("equal")
    return ax


def good_vs_bad_basis():
    """Return one 'good' (near-orthogonal) and one 'bad' (skewed) basis for the same lattice."""
    good = np.array([[1.0, 0.0], [0.0, 1.0]])
    # The 'bad' basis spans the same lattice but has large vectors at a shallow angle.
    bad = np.array([[3.0, 1.0], [5.0, 2.0]])
    return good, bad
```

- [ ] **Step 2: Smoke test**

```bash
python -c "
from pqc_edu.lattice import plot_lattice_2d, good_vs_bad_basis
import numpy as np, matplotlib
matplotlib.use('Agg')
good, bad = good_vs_bad_basis()
plot_lattice_2d(good, radius=3, target=np.array([2.3, 1.7]))
print('ok')
"
```

Expected: `ok`.

- [ ] **Step 3: Commit**

```bash
git add pqc_edu/lattice.py
git commit -m "feat(lattice): 2D lattice visualization helpers"
```

---

## Task 4: `pqc_edu/lwe.py` — Toy LWE (single-bit Regev encryption)

**Files:**
- Create: `pqc_edu/lwe.py`
- Create: `tests/test_lwe.py`

**Scope:** Implement Regev's original LWE-based PKE for a single bit. Not the full public-key variant with matrix A — the simplest form that still has the LWE hardness argument: secret s ∈ Z_q^n, error e small.

- [ ] **Step 1: Write failing tests**

Create `tests/test_lwe.py`:

```python
import numpy as np
import pytest
from pqc_edu.lwe import toy_keygen, toy_encrypt, toy_decrypt


@pytest.mark.parametrize("n,q,sigma", [(10, 97, 1.0), (16, 257, 1.0), (32, 3329, 3.0)])
def test_roundtrip(n, q, sigma):
    rng = np.random.default_rng(0)
    pk, sk = toy_keygen(n, q, sigma, rng)
    for i in range(100):
        bit = int(rng.integers(0, 2))
        ct = toy_encrypt(pk, bit, rng)
        assert toy_decrypt(sk, ct) == bit, f"failed at i={i}"
```

- [ ] **Step 2: Run — expect failure**

```bash
pytest tests/test_lwe.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `pqc_edu/lwe.py`**

```python
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
```

- [ ] **Step 4: Run tests — expect pass**

```bash
pytest tests/test_lwe.py -v
```

Expected: 3 passed (total 300 roundtrips).

- [ ] **Step 5: Commit**

```bash
git add pqc_edu/lwe.py tests/test_lwe.py
git commit -m "feat(lwe): toy single-bit Regev LWE PKE with roundtrip tests"
```

---

## Task 5: `pqc_edu/attacks.py` — Break small LWE ourselves

**Files:**
- Create: `pqc_edu/attacks.py`

No dedicated tests — correctness is demonstrated by the fact that the attack *recovers the known secret* in notebook 03. We add a small correctness assertion via a smoke test.

- [ ] **Step 1: Implement `pqc_edu/attacks.py`**

```python
"""Toy attacks on small-parameter LWE.

Two approaches:

1. brute_force_secret: enumerate all candidate secrets in Z_q^n. O(q^n),
   only feasible for n <= ~5. Used to *demonstrate* the search space.

2. gaussian_elimination_noiseless: treat m samples as a linear system
   and solve exactly. Works only when sigma is artificially set to 0,
   showing why the noise term is what hides the secret.

Neither of these scales. Notebook 03 plots wall-clock vs n and watches
them explode.
"""
from __future__ import annotations
import itertools
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

from .lwe import ToyPublicKey, ToySecretKey


@dataclass
class AttackResult:
    secret: Optional[np.ndarray]
    seconds: float
    method: str
    gave_up: bool = False


def brute_force_secret(pk: ToyPublicKey, error_tolerance: int, time_budget_s: float = 60.0) -> AttackResult:
    """Enumerate s in Z_q^n and score by how many samples it explains within tolerance."""
    n = pk.samples.shape[1] - 1
    q = pk.q
    A = pk.samples[:, :-1]
    b = pk.samples[:, -1]
    start = time.time()
    best_score = -1
    best_s = None
    for s_tuple in itertools.product(range(q), repeat=n):
        if time.time() - start > time_budget_s:
            return AttackResult(secret=best_s, seconds=time.time() - start, method="brute_force", gave_up=True)
        s = np.array(s_tuple)
        residuals = (b - A @ s) % q
        # Wrap residual into [-q/2, q/2]
        residuals = np.where(residuals > q // 2, residuals - q, residuals)
        score = int(np.sum(np.abs(residuals) <= error_tolerance))
        if score > best_score:
            best_score = score
            best_s = s
            if score == len(b):
                break
    return AttackResult(secret=best_s, seconds=time.time() - start, method="brute_force")


def gaussian_elimination_noiseless(pk: ToyPublicKey) -> AttackResult:
    """Solve A s = b (mod q) exactly. Only works if sigma == 0 when pk was generated.

    Implementation: row-reduce [A | b] over Z_q using modular inverse.
    Requires q prime.
    """
    n = pk.samples.shape[1] - 1
    q = pk.q
    start = time.time()
    M = pk.samples.astype(np.int64) % q
    rows = M.shape[0]

    col = 0
    for row in range(min(rows, n)):
        # Find pivot.
        piv = None
        for r in range(row, rows):
            if M[r, col] != 0:
                piv = r; break
        if piv is None:
            col += 1
            if col >= n:
                break
            continue
        M[[row, piv]] = M[[piv, row]]
        inv = pow(int(M[row, col]), -1, q)
        M[row] = (M[row] * inv) % q
        for r in range(rows):
            if r != row and M[r, col]:
                M[r] = (M[r] - M[r, col] * M[row]) % q
        col += 1

    s = M[:n, -1].copy()
    return AttackResult(secret=s, seconds=time.time() - start, method="gaussian_elimination_noiseless")


def verify_secret(sk_true: ToySecretKey, recovered: Optional[np.ndarray]) -> bool:
    if recovered is None:
        return False
    return np.array_equal(np.asarray(recovered) % sk_true.q, sk_true.s % sk_true.q)
```

- [ ] **Step 2: Smoke test — recover a tiny secret**

```bash
python -c "
import numpy as np
from pqc_edu.lwe import toy_keygen
from pqc_edu.attacks import brute_force_secret, verify_secret

rng = np.random.default_rng(0)
pk, sk = toy_keygen(n=3, q=7, sigma=0.0, rng=rng, m=20)
res = brute_force_secret(pk, error_tolerance=0, time_budget_s=5)
print('method:', res.method, 'seconds:', round(res.seconds, 3))
print('recovered:', res.secret, 'expected:', sk.s)
assert verify_secret(sk, res.secret), 'attack failed to recover small secret'
print('ok')
"
```

Expected: `ok`. (With q=7 and n=3, the search space is 343 secrets — instant.)

- [ ] **Step 3: Commit**

```bash
git add pqc_edu/attacks.py
git commit -m "feat(attacks): brute force and gaussian elimination on toy LWE"
```

---

## Task 6: `pqc_edu/params.py` — ML-KEM parameter sets

**Files:**
- Create: `pqc_edu/params.py`

- [ ] **Step 1: Implement**

```python
"""FIPS 203 parameter sets for ML-KEM.

All three security levels share n=256 and q=3329; they differ in:
- k      : module dimension (2 / 3 / 4)
- eta1   : noise for secret / error during key generation
- eta2   : noise for ephemeral randomness during encryption
- du, dv : compression amounts on ciphertext pieces
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class MLKEMParams:
    name: str
    k: int
    eta1: int
    eta2: int
    du: int
    dv: int
    n: int = 256
    q: int = 3329

    @property
    def ek_bytes(self) -> int:
        # 384 * k + 32 bytes (public key in NTT form + rho seed)
        return 384 * self.k + 32

    @property
    def ct_bytes(self) -> int:
        return 32 * (self.du * self.k + self.dv)


ML_KEM_512 = MLKEMParams(name="ML-KEM-512",  k=2, eta1=3, eta2=2, du=10, dv=4)
ML_KEM_768 = MLKEMParams(name="ML-KEM-768",  k=3, eta1=2, eta2=2, du=10, dv=4)
ML_KEM_1024 = MLKEMParams(name="ML-KEM-1024", k=4, eta1=2, eta2=2, du=11, dv=5)

ALL = [ML_KEM_512, ML_KEM_768, ML_KEM_1024]
```

- [ ] **Step 2: Verify sizes match the spec**

```bash
python -c "
from pqc_edu.params import ALL
for p in ALL:
    print(p.name, 'ek_bytes=', p.ek_bytes, 'ct_bytes=', p.ct_bytes)
"
```

Expected:
```
ML-KEM-512 ek_bytes= 800 ct_bytes= 768
ML-KEM-768 ek_bytes= 1184 ct_bytes= 1088
ML-KEM-1024 ek_bytes= 1568 ct_bytes= 1568
```

If any number disagrees with FIPS 203 Table 3, stop and fix the formulas before proceeding.

- [ ] **Step 3: Commit**

```bash
git add pqc_edu/params.py
git commit -m "feat(params): FIPS 203 ML-KEM parameter sets"
```

---

## Task 7: `pqc_edu/sampling.py` — SHAKE PRF, CBD, uniform rejection

**Files:**
- Create: `pqc_edu/sampling.py`

No dedicated test file: we'll exercise sampling end-to-end via `test_ml_kem.py`'s roundtrip tests. We do add a smoke check here.

- [ ] **Step 1: Implement**

```python
"""Deterministic samplers used by ML-KEM (FIPS 203 §4.2).

- prf(seed, nonce, out_len)       : SHAKE256 PRF (FIPS 203 "PRF")
- xof(rho, i, j)                  : SHAKE128 XOF used to generate matrix A
- sample_uniform(xof_stream, n, q): rejection sampling (FIPS 203 Alg. 7 "SampleNTT")
- cbd(buf, eta, n)                : Centered Binomial Distribution (Alg. 8 "SamplePolyCBD")
"""
from __future__ import annotations
import hashlib
import numpy as np
from .polyring import Poly

Q = 3329
N = 256


def prf(seed: bytes, nonce: int, out_len: int) -> bytes:
    h = hashlib.shake_256()
    h.update(seed + bytes([nonce]))
    return h.digest(out_len)


def xof(rho: bytes, i: int, j: int) -> "hashlib.shake_128":
    h = hashlib.shake_128()
    h.update(rho + bytes([j, i]))   # FIPS 203 order: (rho || j || i)
    return h


def sample_uniform_from_xof(xof_obj, q: int = Q, n: int = N) -> Poly:
    """Rejection-sample n coefficients from an XOF stream (SampleNTT, Alg. 7)."""
    out = np.zeros(n, dtype=np.int64)
    j = 0
    chunk_bytes = 3 * 200  # each 3-byte group yields up to 2 candidates
    buf = b""
    idx = 0
    while j < n:
        if idx + 3 > len(buf):
            buf = xof_obj.digest(idx + chunk_bytes)[idx:] if False else xof_obj.digest(chunk_bytes)
            idx = 0
            # shake objects in hashlib are one-shot; we must absorb again. Use a streaming
            # approach via repeated .digest with growing length.
            raise NotImplementedError("Use sample_uniform_from_bytes with pre-squeezed bytes")
        b0 = buf[idx]; b1 = buf[idx+1]; b2 = buf[idx+2]
        idx += 3
        d1 = b0 | ((b1 & 0x0F) << 8)
        d2 = (b1 >> 4) | (b2 << 4)
        if d1 < q:
            out[j] = d1; j += 1
            if j == n: break
        if d2 < q:
            out[j] = d2; j += 1
    return Poly(out, q)


def sample_uniform(rho: bytes, i: int, j: int, q: int = Q, n: int = N) -> Poly:
    """Convenience: construct XOF(rho, i, j) and squeeze enough bytes for rejection sampling.

    We over-squeeze (4 KiB) — enough for n=256 with overwhelming probability; if not,
    re-squeeze. This matches FIPS 203's behavior of reading 3-byte groups until n are accepted.
    """
    seed = rho + bytes([j, i])
    out = np.zeros(n, dtype=np.int64)
    count = 0
    extra = 0
    while count < n:
        length = 504 + extra * 168   # shake128 rate = 168 bytes
        buf = hashlib.shake_128(seed).digest(length)
        idx = extra * 168 if extra > 0 else 0
        while idx + 3 <= len(buf) and count < n:
            b0 = buf[idx]; b1 = buf[idx+1]; b2 = buf[idx+2]
            idx += 3
            d1 = b0 | ((b1 & 0x0F) << 8)
            d2 = (b1 >> 4) | (b2 << 4)
            if d1 < q:
                out[count] = d1; count += 1
                if count == n: break
            if d2 < q and count < n:
                out[count] = d2; count += 1
        extra += 1
    return Poly(out, q)


def cbd(buf: bytes, eta: int, n: int = N, q: int = Q) -> Poly:
    """Centered Binomial Distribution sampler (FIPS 203 Algorithm 8).

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
```

- [ ] **Step 2: Smoke test**

```bash
python -c "
from pqc_edu.sampling import prf, cbd, sample_uniform
# PRF
out = prf(b'\x00'*32, 0, 64); assert len(out) == 64
# CBD with eta=2 over 256 coefficients needs 128 bytes
buf = prf(b'\x00'*32, 1, 128)
p = cbd(buf, eta=2); assert p.n == 256
# Uniform NTT sample
A = sample_uniform(b'\x00'*32, 0, 0)
assert A.n == 256
print('ok')
"
```

Expected: `ok`.

- [ ] **Step 3: Commit**

```bash
git add pqc_edu/sampling.py
git commit -m "feat(sampling): SHAKE PRF, CBD, and uniform rejection samplers"
```

---

## Task 8: `pqc_edu/ml_kem.py` — K-PKE layer

**Files:**
- Create: `pqc_edu/ml_kem.py` (initial K-PKE layer)

- [ ] **Step 1: Implement the K-PKE layer**

```python
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
    q = p.q
    factor = 1 << d
    c = ((p.coeffs * factor + q // 2) // q) % factor
    return Poly(c, factor)


def decompress(p: Poly, d: int, q: int = Q) -> Poly:
    factor = 1 << d
    out = ((p.coeffs * q + factor // 2) // factor) % q
    return Poly(out, q)


def byte_encode(polys: List[Poly], d: int) -> bytes:
    """Pack d-bit coefficients of each poly into a byte string. FIPS 203 Algorithm 5."""
    out_bits = np.concatenate([
        np.unpackbits(p.coeffs.astype(np.uint32).view(np.uint8).reshape(-1, 4),
                      axis=1, bitorder="little")[:, :d].reshape(-1)
        for p in polys
    ])
    # pad to byte boundary (always aligned since d*n is a multiple of 8)
    return np.packbits(out_bits, bitorder="little").tobytes()


def byte_decode(buf: bytes, d: int, n: int, count: int, q_target: int) -> List[Poly]:
    """Inverse of byte_encode. Returns `count` polynomials of length n, modulus q_target.

    q_target is q (=3329) for d=12 public keys, or 2^d for compressed outputs.
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
    # Matrix A: k x k polys, each uniform in R_q, NTT domain (FIPS 203 is always NTT form)
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
    A_hat_T = [[sample_uniform(rho, i, j) for i in range(params.k)] for j in range(params.k)]  # transposed
    N_ctr = 0
    r = [cbd(prf(coins, (N_ctr := N_ctr + 1) - 1, N * params.eta1 // 4), params.eta1) for _ in range(params.k)]
    e1 = [cbd(prf(coins, (N_ctr := N_ctr + 1) - 1, N * params.eta2 // 4), params.eta2) for _ in range(params.k)]
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
    c1 = b"".join([byte_encode([compress(p, params.du)], params.du) for p in u])
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
    # m' = v - INTT(s_hat . NTT(u))
    u_hat = [ntt(p) for p in u]
    acc = Poly.zero(N, Q)
    for j in range(params.k):
        acc = acc + pointwise_mul(s_hat[j], u_hat[j])
    w = v - intt(acc)
    m_poly = compress(w, 1)
    return _bits_to_bytes(m_poly.coeffs.astype(np.uint8))


def _bytes_to_bits(b: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(b, dtype=np.uint8), bitorder="little")


def _bits_to_bytes(bits: np.ndarray) -> bytes:
    return np.packbits(bits.astype(np.uint8), bitorder="little").tobytes()
```

- [ ] **Step 2: Smoke test K-PKE roundtrip**

```bash
python -c "
import os
from pqc_edu.params import ML_KEM_512
from pqc_edu.ml_kem import k_pke_keygen, k_pke_encrypt, k_pke_decrypt
keys = k_pke_keygen(ML_KEM_512, os.urandom(32))
m = os.urandom(32)
ct = k_pke_encrypt(ML_KEM_512, keys.ek, m, os.urandom(32))
m2 = k_pke_decrypt(ML_KEM_512, keys.dk, ct)
print('match:', m == m2)
print('ek bytes:', len(keys.ek), 'ct bytes:', len(ct))
"
```

Expected: `match: True`, ek=800, ct=768. If match is False, the error is in compression/encoding — debug before proceeding.

- [ ] **Step 3: Commit**

```bash
git add pqc_edu/ml_kem.py
git commit -m "feat(ml_kem): K-PKE keygen/encrypt/decrypt (FIPS 203 §5)"
```

---

## Task 9: `pqc_edu/ml_kem.py` — ML-KEM wrapper (FO transform)

**Files:**
- Modify: `pqc_edu/ml_kem.py` (append ML-KEM wrapper at the end)
- Create: `tests/test_ml_kem.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_ml_kem.py`:

```python
import os
import pytest
from pqc_edu.params import ML_KEM_512, ML_KEM_768, ML_KEM_1024, ALL
from pqc_edu.ml_kem import ml_kem_keygen, ml_kem_encaps, ml_kem_decaps


@pytest.mark.parametrize("params", ALL, ids=lambda p: p.name)
def test_roundtrip(params):
    for _ in range(50):
        ek, dk = ml_kem_keygen(params)
        K1, ct = ml_kem_encaps(params, ek)
        K2 = ml_kem_decaps(params, dk, ct)
        assert K1 == K2
        assert len(K1) == 32


@pytest.mark.parametrize("params,expected_ek,expected_ct", [
    (ML_KEM_512, 800, 768),
    (ML_KEM_768, 1184, 1088),
    (ML_KEM_1024, 1568, 1568),
])
def test_sizes_match_spec(params, expected_ek, expected_ct):
    ek, dk = ml_kem_keygen(params)
    _, ct = ml_kem_encaps(params, ek)
    assert len(ek) == expected_ek
    assert len(ct) == expected_ct


def test_deterministic_keygen_from_seed():
    # Using the _internal seeded variant, same seed -> same keys
    from pqc_edu.ml_kem import _ml_kem_keygen_from_seeds
    d = b"\x11" * 32; z = b"\x22" * 32
    ek1, dk1 = _ml_kem_keygen_from_seeds(ML_KEM_768, d, z)
    ek2, dk2 = _ml_kem_keygen_from_seeds(ML_KEM_768, d, z)
    assert ek1 == ek2 and dk1 == dk2
```

- [ ] **Step 2: Run — expect failure (names not defined)**

```bash
pytest tests/test_ml_kem.py -v
```

Expected: ImportError on `ml_kem_keygen`.

- [ ] **Step 3: Append ML-KEM wrapper to `pqc_edu/ml_kem.py`**

Append to the end of the file:

```python
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

    The FO-transform rejection branch: if re-encryption disagrees, return a
    pseudo-random shared secret derived from z and the ciphertext rather
    than a bogus key. This makes decryption failure indistinguishable.
    """
    dk_pke_len = 384 * params.k
    ek_len = params.ek_bytes
    dk_pke = dk[:dk_pke_len]
    ek = dk[dk_pke_len : dk_pke_len + ek_len]
    h_ek = dk[dk_pke_len + ek_len : dk_pke_len + ek_len + 32]
    z = dk[dk_pke_len + ek_len + 32 :]
    m_prime = k_pke_decrypt(params, dk_pke, ct)
    g = hashlib.sha3_512(m_prime + h_ek).digest()
    K_prime, r_prime = g[:32], g[32:]
    # Re-encrypt and compare.
    ct_prime = k_pke_encrypt(params, ek, m_prime, r_prime)
    if ct == ct_prime:
        return K_prime
    # Implicit rejection: return pseudo-random K = SHAKE256(z || ct, 32)
    h = hashlib.shake_256(); h.update(z + ct)
    return h.digest(32)
```

- [ ] **Step 4: Run tests — expect pass**

```bash
pytest tests/test_ml_kem.py -v
```

Expected: 3 test functions × 3 parameter sets (where applicable) = 7 passed. If any roundtrip fails, suspect the pointwise-mul zeta indexing in `ntt.py` first.

- [ ] **Step 5: Commit**

```bash
git add pqc_edu/ml_kem.py tests/test_ml_kem.py
git commit -m "feat(ml_kem): ML-KEM wrapper with FO-transform; roundtrip tests"
```

---

## Task 10: Notebook 01 — Lattice intro

**Files:**
- Create: `notebooks/01_lattice_intro.ipynb`

Write the notebook as a `.py` first and convert with `jupytext`, or create the JSON directly. Use whichever is in your workflow; below is the content outline each cell must cover.

- [ ] **Step 1: Create notebook with the following cells**

| # | Cell type | Purpose | Content |
|---|-----------|---------|---------|
| 1 | Markdown | Title + intro | "## Notebook 01 — What is a lattice?" Explain the goal in 2 paragraphs. |
| 2 | Code | Imports | `import numpy as np; import matplotlib.pyplot as plt; from pqc_edu.lattice import plot_lattice_2d, good_vs_bad_basis` |
| 3 | Markdown | Lattice definition | Formal: L(B) = {Bz : z ∈ Z^d}. Give the 2-D square lattice as the first example. |
| 4 | Code | Plot Z^2 | `plot_lattice_2d(np.eye(2), radius=4)` |
| 5 | Markdown | Good vs bad basis | Explain both generate the same lattice but one is easier to compute with. |
| 6 | Code | Side-by-side plot | `good, bad = good_vs_bad_basis(); fig, axes = plt.subplots(1, 2, figsize=(10,5)); plot_lattice_2d(good, 4, ax=axes[0]); plot_lattice_2d(bad, 4, ax=axes[1]); axes[0].set_title('good basis'); axes[1].set_title('bad basis')` |
| 7 | Markdown | SVP and CVP | State the Shortest Vector Problem and Closest Vector Problem. Note LWE relates to Bounded Distance Decoding. |
| 8 | Code | CVP demo | Generate a random target near the lattice, call `plot_lattice_2d(good, 5, target=np.array([2.3, 1.7]))`. |
| 9 | Markdown | Why quantum-hard? | Brief: Shor's algorithm breaks factoring and discrete log, but no known polynomial-time quantum algorithm solves approximate SVP/CVP in high dimensions. Link to next notebook. |

- [ ] **Step 2: Verify it executes end-to-end**

```bash
jupyter nbconvert --to notebook --execute notebooks/01_lattice_intro.ipynb --output 01_lattice_intro.ipynb
```

Expected: no exceptions. Inspect the plots visually.

- [ ] **Step 3: Commit**

```bash
git add notebooks/01_lattice_intro.ipynb
git commit -m "docs(nb01): lattice introduction notebook"
```

---

## Task 11: Notebook 02 — Toy LWE

**Files:**
- Create: `notebooks/02_toy_lwe.ipynb`

- [ ] **Step 1: Create notebook**

| # | Cell type | Content |
|---|-----------|---------|
| 1 | Markdown | "## Notebook 02 — The LWE problem". Goal: see LWE operationally. |
| 2 | Markdown | Define the search-LWE problem in words: given many (a_i, b_i=<a_i,s>+e_i) find s. Note the decision version. |
| 3 | Code | Inline build of one LWE sample with tiny numbers: `n=4, q=17, sigma=1.0`. Compute one (a, b) by hand with `rng = np.random.default_rng(1)`. Print s, e, A, b. |
| 4 | Markdown | Derive Regev's single-bit PKE on paper; the next cells run the same logic from `pqc_edu.lwe`. |
| 5 | Code | `from pqc_edu.lwe import toy_keygen, toy_encrypt, toy_decrypt; rng = np.random.default_rng(0); pk, sk = toy_keygen(n=16, q=257, sigma=1.0, rng=rng)` |
| 6 | Code | Loop: encrypt/decrypt 20 random bits, assert matches. Print a table. |
| 7 | Code | **Noise budget demo**: rerun with sigma=5.0 and watch failures appear. |
| 8 | Markdown | Why this is not yet ML-KEM: (a) single bit per ciphertext, (b) matrix of samples is huge, (c) no FO transform. Pointer to notebook 06. |

- [ ] **Step 2: Execute**

```bash
jupyter nbconvert --to notebook --execute notebooks/02_toy_lwe.ipynb --output 02_toy_lwe.ipynb
```

- [ ] **Step 3: Commit**

```bash
git add notebooks/02_toy_lwe.ipynb
git commit -m "docs(nb02): toy LWE notebook"
```

---

## Task 12: Notebook 03 — Attacking toy LWE

**Files:**
- Create: `notebooks/03_attacking_toy_lwe.ipynb`

- [ ] **Step 1: Create notebook**

| # | Cell type | Content |
|---|-----------|---------|
| 1 | Markdown | "## Notebook 03 — Break it yourself" |
| 2 | Code | Imports: `numpy, matplotlib, time; from pqc_edu.lwe import toy_keygen; from pqc_edu.attacks import brute_force_secret, gaussian_elimination_noiseless, verify_secret`. |
| 3 | Markdown | **Experiment 1**: noise-less LWE is just linear algebra. |
| 4 | Code | Generate noise-less LWE (sigma=0.0), run `gaussian_elimination_noiseless`, verify recovery. |
| 5 | Markdown | **Experiment 2**: with noise, linear algebra breaks. Show noise makes the system inconsistent. |
| 6 | Code | Same generator but sigma=1.5; run `gaussian_elimination_noiseless` and observe it does **not** recover the true s. |
| 7 | Markdown | **Experiment 3**: brute force. Works for tiny n. Budget 20 seconds per n. |
| 8 | Code | For n in {2, 3, 4, 5}: generate, brute-force, record seconds, assert recovery. Plot time vs n on log scale. |
| 9 | Markdown | Extrapolate: n=16 would take ~q^11 = 257^11 ≈ 10^26 years at these rates. This is the point. |
| 10 | Code | Print the extrapolated time for n=16 using the measured slope. |
| 11 | Markdown | Note: real attacks (BKZ, primal/dual) are exponentially *less* bad than brute force but still exponential in dimension for well-chosen parameters. Reference for curious: Albrecht et al. LWE estimator. |

- [ ] **Step 2: Execute**

```bash
jupyter nbconvert --to notebook --execute notebooks/03_attacking_toy_lwe.ipynb --output 03_attacking_toy_lwe.ipynb
```

- [ ] **Step 3: Commit**

```bash
git add notebooks/03_attacking_toy_lwe.ipynb
git commit -m "docs(nb03): attacking toy LWE notebook"
```

---

## Task 13: Notebook 04 — Polynomial rings and NTT

**Files:**
- Create: `notebooks/04_polynomial_rings.ipynb`

- [ ] **Step 1: Create notebook**

| # | Cell type | Content |
|---|-----------|---------|
| 1 | Markdown | "## Notebook 04 — Why polynomial rings?" |
| 2 | Markdown | Motivate: instead of an n×n matrix of LWE samples, use n polynomials in Z_q[x]/(x^n+1). One polynomial multiplication ≈ one matrix-vector product. |
| 3 | Code | Hand-build two polynomials with n=4, q=17. Multiply schoolbook. Show anti-cyclic fold. |
| 4 | Code | Import `Poly`; reproduce the same example. Verify identical result. |
| 5 | Markdown | Introduce the NTT. Analogy to FFT. Requirement: q prime with n | (q-1). Show 3329 = 13·256 + 1. |
| 6 | Code | `from pqc_edu.ntt import ntt, intt, poly_mul_ntt`. Roundtrip on a random poly. |
| 7 | Code | **Timing experiment**: n=256, q=3329. Time `a*b` (schoolbook) vs `poly_mul_ntt(a, b)` over 200 iterations. Print speedup. |
| 8 | Markdown | Takeaway: in ML-KEM, all operations happen in NTT domain. You only INTT at the end. |

- [ ] **Step 2: Execute and commit**

```bash
jupyter nbconvert --to notebook --execute notebooks/04_polynomial_rings.ipynb --output 04_polynomial_rings.ipynb
git add notebooks/04_polynomial_rings.ipynb
git commit -m "docs(nb04): polynomial rings and NTT notebook"
```

---

## Task 14: Notebook 05 — Ring-LWE and MLWE

**Files:**
- Create: `notebooks/05_ring_lwe.ipynb`

- [ ] **Step 1: Create notebook**

| # | Cell type | Content |
|---|-----------|---------|
| 1 | Markdown | "## Notebook 05 — LWE → Ring-LWE → Module-LWE" |
| 2 | Markdown | Rewrite LWE using one sample (a polynomial in R_q) instead of n scalars. Ring-LWE: s ∈ R_q, e ∈ R_q small, b = a·s + e. |
| 3 | Code | Build a Ring-LWE sample using `Poly`, CBD from `pqc_edu.sampling`. Show a, s, e, b. |
| 4 | Markdown | Why Module-LWE? Ring-LWE has extra algebraic structure — MLWE uses a *vector of polynomials*, a compromise between plain LWE's generality and Ring-LWE's efficiency. |
| 5 | Code | Demonstrate MLWE with k=2: s, e are length-2 vectors of polynomials, A is a 2×2 matrix of polynomials, b = A·s + e. Verify one decryption on the toy level (single bit, using compression by 1). |
| 6 | Markdown | Preview: ML-KEM uses MLWE with k ∈ {2, 3, 4}. The next notebook implements the full scheme. |

- [ ] **Step 2: Execute and commit**

```bash
jupyter nbconvert --to notebook --execute notebooks/05_ring_lwe.ipynb --output 05_ring_lwe.ipynb
git add notebooks/05_ring_lwe.ipynb
git commit -m "docs(nb05): Ring-LWE and MLWE notebook"
```

---

## Task 15: Notebook 06 — ML-KEM implementation walk-through

**Files:**
- Create: `notebooks/06_ml_kem_spec.ipynb`

- [ ] **Step 1: Create notebook**

| # | Cell type | Content |
|---|-----------|---------|
| 1 | Markdown | "## Notebook 06 — ML-KEM (FIPS 203)". Goal: read the spec with the code. |
| 2 | Markdown | Show the three parameter sets as a table (name, k, eta1, eta2, du, dv, ek_bytes, ct_bytes). Generate from `pqc_edu.params.ALL`. |
| 3 | Markdown | K-PKE.KeyGen algorithm box from FIPS 203 (pseudocode) quoted on the left; screen-shot-free — just formatted markdown. |
| 4 | Code | Step through `k_pke_keygen(ML_KEM_512, d=b'\x01'*32)` with `%%prun` timing. Print resulting ek/dk sizes. |
| 5 | Markdown | K-PKE.Encrypt algorithm box. |
| 6 | Code | Run `k_pke_encrypt`, show the compression effect: print the float error `sum(|t_hat - decompress(compress(t_hat))|)/n`. |
| 7 | Markdown | K-PKE.Decrypt and why it sometimes fails; ML-KEM's design drives decryption-failure probability below 2^-139. |
| 8 | Markdown | FO-transform: why Encaps is not just K-PKE.Encrypt(random). Chosen-ciphertext security intuition. |
| 9 | Code | `from pqc_edu.ml_kem import ml_kem_keygen, ml_kem_encaps, ml_kem_decaps`. Show a full roundtrip, print K as hex. |

- [ ] **Step 2: Execute and commit**

```bash
jupyter nbconvert --to notebook --execute notebooks/06_ml_kem_spec.ipynb --output 06_ml_kem_spec.ipynb
git add notebooks/06_ml_kem_spec.ipynb
git commit -m "docs(nb06): ML-KEM implementation walk-through"
```

---

## Task 16: Notebook 07 — Tests, sizes, benchmarks

**Files:**
- Create: `notebooks/07_ml_kem_tests.ipynb`

- [ ] **Step 1: Create notebook**

| # | Cell type | Content |
|---|-----------|---------|
| 1 | Markdown | "## Notebook 07 — Does it actually work?" |
| 2 | Code | **Roundtrip stress**: For each param set, run 200 (keygen, encaps, decaps) cycles. Assert K1 == K2 every time. Print decryption-failure count (should be 0). |
| 3 | Code | **Size table**: print a pandas-style table of ek/ct/shared-secret sizes, check against FIPS 203 Table 3. |
| 4 | Code | **Determinism**: same seeds → same keys, using `_ml_kem_keygen_from_seeds`. |
| 5 | Code | **Benchmark**: time keygen/encaps/decaps over 50 iterations per param set. Plot a bar chart. |
| 6 | Markdown | Caveat: this implementation is not optimized. Measured relative differences (k=2 vs 3 vs 4) reflect algorithmic cost; absolute numbers are not comparable to C implementations. |

- [ ] **Step 2: Execute and commit**

```bash
jupyter nbconvert --to notebook --execute notebooks/07_ml_kem_tests.ipynb --output 07_ml_kem_tests.ipynb
git add notebooks/07_ml_kem_tests.ipynb
git commit -m "docs(nb07): ML-KEM roundtrip tests, sizes, benchmarks"
```

---

## Task 17: Notebook 08 — Hybrid X25519 + ML-KEM-768

**Files:**
- Create: `notebooks/08_hybrid_kem.ipynb`

- [ ] **Step 1: Create notebook**

| # | Cell type | Content |
|---|-----------|---------|
| 1 | Markdown | "## Notebook 08 — Hybrid KEM: classical + PQ". Explain why hybrid (belt-and-suspenders until PQC gets more cryptanalysis time). |
| 2 | Code | Imports: `from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey; from pqc_edu.ml_kem import ml_kem_keygen, ml_kem_encaps, ml_kem_decaps; from pqc_edu.params import ML_KEM_768; import hashlib`. |
| 3 | Markdown | Protocol sketch: Alice sends `(X25519_pub, ek)`. Bob returns `(X25519_pub_B, ct)`. Both derive `K = SHAKE256(ss_x25519 ‖ ss_mlkem ‖ transcript)`. |
| 4 | Code | Implement both sides. Assert both derive the same 32-byte `K`. |
| 5 | Markdown | Security properties: breaks only if both X25519 and ML-KEM break. Note real-world deployments (TLS, SSH) follow this same pattern (hybrid draft-ietf-tls-hybrid-design). |
| 6 | Code | **Tamper test**: flip one bit in ML-KEM ct; observe that shared secrets diverge (because of FO-transform implicit rejection). |

- [ ] **Step 2: Execute and commit**

```bash
jupyter nbconvert --to notebook --execute notebooks/08_hybrid_kem.ipynb --output 08_hybrid_kem.ipynb
git add notebooks/08_hybrid_kem.ipynb
git commit -m "docs(nb08): hybrid X25519 + ML-KEM-768 notebook"
```

---

## Task 18: Notebook 09 — Wrap-up

**Files:**
- Create: `notebooks/09_wrap_up.ipynb`

- [ ] **Step 1: Create notebook**

| # | Cell type | Content |
|---|-----------|---------|
| 1 | Markdown | "## Notebook 09 — What we built". Summary paragraph. |
| 2 | Markdown | The "one paragraph" test: an attempt at the 5-sentence explanation ("LWE is hard because... ML-KEM uses it via..."). Invite the reader to write their own. |
| 3 | Markdown | Gaps in this implementation (vs production): constant-time arithmetic, safe memory clearing, NIST KAT conformance, side-channel hardening. Each with a short "what real code does". |
| 4 | Markdown | Further reading: FIPS 203 (actual), Regev 2005, Peikert's "A Decade of Lattice Cryptography", LWE estimator, kyber-py reference. |
| 5 | Markdown | Where to go next: ML-DSA (signatures), SLH-DSA (hash-based signatures), BIKE/HQC (code-based). |

- [ ] **Step 2: Execute and commit**

```bash
jupyter nbconvert --to notebook --execute notebooks/09_wrap_up.ipynb --output 09_wrap_up.ipynb
git add notebooks/09_wrap_up.ipynb
git commit -m "docs(nb09): wrap-up notebook"
```

---

## Task 19: Final verification

- [ ] **Step 1: All tests green**

```bash
pytest tests/ -v
```

Expected: every test passes.

- [ ] **Step 2: All notebooks execute clean**

```bash
for nb in notebooks/0*.ipynb; do
  echo "=== $nb ==="
  jupyter nbconvert --to notebook --execute "$nb" --output "$(basename "$nb")" --output-dir notebooks/ 2>&1 | tail -3
done
```

Expected: each prints `[NbConvertApp] Writing …` with no tracebacks.

- [ ] **Step 3: Package installs fresh**

```bash
python -m venv /tmp/pqc_fresh_venv
/tmp/pqc_fresh_venv/bin/pip install -e ".[dev]"
/tmp/pqc_fresh_venv/bin/pytest tests/ -q
rm -rf /tmp/pqc_fresh_venv
```

Expected: tests pass in the fresh environment.

- [ ] **Step 4: Final commit**

```bash
git add -u
git commit -m "chore: final verification pass" --allow-empty
```

---

## Self-Review Checklist (for the plan author)

- [x] Every spec section is covered by at least one task (specs §4 structure → Task 0–9; §5 notebooks → Task 10–18; §7 tests → Tasks 1, 2, 4, 9, 16; §9 success criteria → Task 19).
- [x] No `TBD`, no "similar to task N" — all code shown.
- [x] Type consistency: `MLKEMParams`, `Poly`, `ml_kem_{keygen,encaps,decaps}`, `k_pke_{keygen,encrypt,decrypt}` names are identical across all tasks.
- [x] Each task is independent of downstream tasks for compilation (polyring/NTT/params/sampling are prerequisites for ml_kem; plan orders them accordingly).
- [x] Notebook tasks (10–18) all list concrete cell-by-cell content and run via `nbconvert --execute` for CI-like verification.

## Known Risk Points

- **NTT pair indexing** (Task 2): the pointwise multiplication zeta table is the part most likely to be off by a factor or sign. Task 2 Step 4's `test_ntt_matches_schoolbook` is the single strongest signal — do not skip past it.
- **`byte_encode` endianness** (Task 8): FIPS 203 uses little-endian bit packing. If round-trip encode/decode returns garbage, check bit order first.
- **Rejection-sampling bytes budget** (Task 7): `sample_uniform` over-squeezes; if you ever see it hang, the inner while-loop is not refilling the buffer.

If a roundtrip test fails, the debug order is: NTT → encoding → compression → FO transform. The first three have isolated tests; failures in ML-KEM-only tests point at FO-transform.
