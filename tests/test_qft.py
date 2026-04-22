import numpy as np
import pytest
from pqc_edu.quantum.simulator import QuantumState, H
from pqc_edu.quantum.circuits import qft, inverse_qft, qft_matrix


def test_qft_matrix_one_qubit_is_hadamard():
    F = qft_matrix(1)
    assert np.allclose(F, H)


def test_qft_matrix_is_unitary():
    for n in (1, 2, 3, 4):
        F = qft_matrix(n)
        assert np.allclose(F @ F.conj().T, np.eye(1 << n)), f"n={n} not unitary"


def test_qft_then_inverse_restores_state():
    rng = np.random.default_rng(7)
    for n in (1, 2, 3, 4):
        q = QuantumState(n)
        q.amps = rng.standard_normal(1 << n) + 1j * rng.standard_normal(1 << n)
        q.amps /= np.linalg.norm(q.amps)
        before = q.amps.copy()
        qft(q, list(range(n)))
        inverse_qft(q, list(range(n)))
        assert np.allclose(q.amps, before, atol=1e-10), f"roundtrip failed for n={n}"


def test_qft_on_basis_state_2qubit():
    # For n=2, QFT|j> = (1/2) sum_k omega^{jk} |k> with omega = exp(2_i/4) = i.
    # Take j=1:  QFT|1> = 1/2 (|0> + i|1> - |2> - i|3>)
    q = QuantumState(2)
    q.amps[:] = 0.0
    q.amps[1] = 1.0
    qft(q, [0, 1])
    expected = 0.5 * np.array([1, 1j, -1, -1j], dtype=complex)
    assert np.allclose(q.amps, expected, atol=1e-10)


def test_qft_reveals_period_in_spectrum():
    # Prepare amps with period-4 pattern in an 8-dim register: f(k) = (k mod 4 == 0 ? 1 : 0)
    n = 3
    q = QuantumState(n)
    q.amps[:] = 0.0
    for k in (0, 4):  # period-4 within 0..7
        q.amps[k] = 1.0
    q.amps /= np.linalg.norm(q.amps)
    qft(q, [0, 1, 2])
    # The QFT of a periodic delta comb is a delta comb at multiples of N/period = 8/4 = 2.
    mags = np.abs(q.amps)
    # Expect peaks only at indices 0, 2, 4, 6.
    for k in range(8):
        if k in (0, 2, 4, 6):
            assert mags[k] > 0.3
        else:
            assert mags[k] < 1e-9
