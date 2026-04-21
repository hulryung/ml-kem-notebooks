# ML-KEM from Scratch

A nine-notebook curriculum that teaches lattice-based post-quantum cryptography by implementing **ML-KEM (Kyber / FIPS 203)** from scratch in Python.

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

## Reading order

The notebooks build on each other — read in order. You can skim 04 and 05 if you already know ring-LWE and NTTs.

1. Lattice intro — what is a lattice, why is it hard?
2. Toy LWE — the LWE problem on small numbers
3. Attacking toy LWE — break small LWE ourselves
4. Polynomial rings and NTT — the trick that makes ML-KEM fast
5. Ring-LWE and Module-LWE — lifting LWE into polynomial rings
6. ML-KEM spec walk-through — FIPS 203 implementation
7. Tests and benchmarks — verify the implementation
8. Hybrid X25519 + ML-KEM — real-world deployment pattern
9. Wrap-up — what we built, what's next

## Source

Code, tests, and notebooks: [github.com/hulryung/ml-kem-notebooks](https://github.com/hulryung/ml-kem-notebooks)
