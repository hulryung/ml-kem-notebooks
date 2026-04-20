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
