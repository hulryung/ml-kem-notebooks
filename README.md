# ML-KEM from Scratch

Jupyter notebooks that teach lattice-based post-quantum cryptography by implementing **ML-KEM (Kyber / FIPS 203)** from scratch in Python.

## 📖 Read online (recommended)

The notebooks are easiest to read as a rendered book with navigation, math, and search:

- **English**: https://hulryung.github.io/ml-kem-notebooks/
- **한글**: https://hulryung.github.io/ml-kem-notebooks/ko/

The hosted site auto-renders plots, supports cross-notebook navigation, and switches language via the floating button in the top-right of every page.

## ⚠️ Educational only

This implementation is not constant-time, not side-channel resistant, and has not been validated against NIST Known-Answer Tests. **Do not use for real encryption.**

## 🧭 Reading order

Read in sequence — each notebook builds on the previous one.

1. `notebooks/01_lattice_intro.ipynb` — what is a lattice, why is it hard?
2. `notebooks/02_toy_lwe.ipynb` — the LWE problem on small numbers
3. `notebooks/03_attacking_toy_lwe.ipynb` — break small LWE ourselves
4. `notebooks/04_polynomial_rings.ipynb` — Z_q[x]/(x^n+1) and NTT
5. `notebooks/05_ring_lwe.ipynb` — LWE → Ring-LWE → MLWE
6. `notebooks/06_ml_kem_spec.ipynb` — FIPS 203 implementation walk-through
7. `notebooks/07_ml_kem_tests.ipynb` — roundtrip tests, size tables, benchmarks
8. `notebooks/08_hybrid_kem.ipynb` — X25519 + ML-KEM hybrid key exchange
9. `notebooks/09_wrap_up.ipynb` — summary and further reading

## 🛠 Run locally

    python -m venv .venv && source .venv/bin/activate
    pip install -e ".[dev]"
    pytest tests/ -v
    jupyter lab notebooks/

## 📚 Build the book locally

    pip install -e ".[book]"
    jupyter-book build .       # English site in _build/html/
    jupyter-book build ko/     # Korean site in ko/_build/html/

## 📝 Korean version

한글 버전은 `ko/notebooks/`에 있으며, 코드 셀은 원본과 동일하고 markdown 셀만 번역돼 있습니다. 온라인 책: https://hulryung.github.io/ml-kem-notebooks/ko/
