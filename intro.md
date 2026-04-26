# ML-KEM from Scratch

> 🌐 English · <a href="ko/">한국어</a> · **v1.0** · <a href="https://github.com/hulryung/ml-kem-notebooks/blob/main/CHANGELOG.md">Changelog</a>

A 14-notebook curriculum that teaches lattice-based post-quantum cryptography by implementing **ML-KEM (Kyber / FIPS 203)** from scratch in Python — with optional prerequisite primers for beginners and extension notebooks on signatures and real-world deployment.

```{warning}
This implementation is educational only. It is not constant-time, not side-channel resistant, and has not been validated against NIST Known-Answer Tests. **Do not use for real encryption.**
```

## What you will build

By the end, you will have written — and tested — Python code that:

- Generates ML-KEM-512 / 768 / 1024 key pairs with exactly the byte sizes from FIPS 203 Table 3
- Encapsulates and decapsulates 32-byte shared secrets
- Passes 200 roundtrip tests per parameter set with zero decryption failures
- Combines with X25519 in a hybrid KEM, the pattern TLS 1.3 is deploying today

And you will have *broken* toy-sized LWE yourself, seeing first-hand why the real parameters are out of reach.

## Three parts, one narrative

**Part 1 — Prerequisites (optional).** For readers who want context before the lattice math. Covers the quantum threat model, classical cryptography in 10 minutes, and the exact math vocabulary used later. Skip if you already know what RSA does and are comfortable with modular arithmetic.

**Part 2 — Core: Building ML-KEM.** The main event. Nine notebooks that go from 2-D lattices → toy LWE → breaking toy LWE → polynomial rings and NTT → Ring/Module-LWE → FIPS 203 ML-KEM implementation → tests and benchmarks → hybrid X25519 + ML-KEM → summary.

**Part 3 — Beyond ML-KEM.** Two panoramic notebooks: an overview of lattice and hash-based signatures (ML-DSA, SLH-DSA) and a reference chapter with deployment status, FAQ, and a glossary.

## Reading order

Follow the parts in sequence, or jump to Part 2 if you already know the basics.

## Source

Code, tests, and notebooks: [github.com/hulryung/ml-kem-notebooks](https://github.com/hulryung/ml-kem-notebooks)
