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
