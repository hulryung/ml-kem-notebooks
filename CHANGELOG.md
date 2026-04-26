# Changelog

All notable changes are kept in this file. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow [SemVer](https://semver.org/).

## [1.0.0] — 2026-04-26

First stable release. The 5-part curriculum (prerequisites, core ML-KEM build, beyond ML-KEM, Quantum Reckoning, lattice attacks) is complete in both English and Korean.

### Highlights

- **Part 1**: optional primers — why PQC, classical-crypto recap, math toolbox
- **Part 2**: core path — toy LWE → ring LWE → ML-KEM (FIPS 203 spec + tests) → hybrid KEM
- **Part 3**: signatures (ML-DSA / SLH-DSA), deployment + FAQ glossary
- **Part 4**: Quantum Reckoning — quantum basics, QFT/period finding, Shor breaks RSA, why lattices resist Shor
- **Part 5**: practical attacks — LLL basis reduction, BKZ, primal attack on LWE, parameter estimation
- **Capstone**: links to [pq-messenger](https://pqmsg.hulryung.com/) — a Signal-style PQ messenger built on this `pqc_edu` library
- Full Korean translation served at `/ko/`

### Site / Infra

- Canonical site moved to **`pqc.hulryung.com`** (Vercel-hosted)
- Legacy `hulryung.github.io/ml-kem-notebooks/*` URLs redirect path-aware to the new domain
- CI deploys to Vercel on every push to `main`
- `actions/cache` for `.jupyter_cache` so doc-only pushes skip notebook re-execution
- hreflang + JSON-LD TechArticle structured data on every page

[1.0.0]: https://github.com/hulryung/ml-kem-notebooks/releases/tag/v1.0.0
