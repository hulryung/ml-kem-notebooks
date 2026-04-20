# PQC Learning Project

Jupyter notebooks that teach lattice-based post-quantum cryptography by implementing ML-KEM (Kyber / FIPS 203) from scratch.

## Educational Only (Warning)

This implementation is not constant-time, not side-channel resistant, and has not been tested against NIST KATs. **Do not use for real encryption.**

## Setup

    python -m venv .venv && source .venv/bin/activate
    pip install -e ".[dev]"

## Reading Order

1. `notebooks/01_lattice_intro.ipynb` — what is a lattice, why is it hard?
2. `notebooks/02_toy_lwe.ipynb` — the LWE problem on small numbers
3. `notebooks/03_attacking_toy_lwe.ipynb` — break small LWE ourselves
4. `notebooks/04_polynomial_rings.ipynb` — Z_q[x]/(x^n+1) and NTT
5. `notebooks/05_ring_lwe.ipynb` — LWE to Ring-LWE to MLWE
6. `notebooks/06_ml_kem_spec.ipynb` — FIPS 203 implementation
7. `notebooks/07_ml_kem_tests.ipynb` — roundtrip tests, size tables, benchmarks
8. `notebooks/08_hybrid_kem.ipynb` — X25519 + ML-KEM hybrid
9. `notebooks/09_wrap_up.ipynb` — summary and further reading

## Testing

    pytest tests/ -v
