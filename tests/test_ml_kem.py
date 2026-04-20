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
    from pqc_edu.ml_kem import _ml_kem_keygen_from_seeds
    d = b"\x11" * 32; z = b"\x22" * 32
    ek1, dk1 = _ml_kem_keygen_from_seeds(ML_KEM_768, d, z)
    ek2, dk2 = _ml_kem_keygen_from_seeds(ML_KEM_768, d, z)
    assert ek1 == ek2 and dk1 == dk2
