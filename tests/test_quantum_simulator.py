import numpy as np
import pytest
from pqc_edu.quantum.simulator import (
    QuantumState, H, X, Z, CNOT_MATRIX, Rk,
)


def amps_close(a, b, atol=1e-10):
    return np.allclose(a, b, atol=atol)


def test_initial_state_is_zero():
    q = QuantumState(3)
    expected = np.zeros(8, dtype=complex); expected[0] = 1.0
    assert amps_close(q.amps, expected)


def test_hadamard_on_zero():
    q = QuantumState(1)
    q.apply_1q(H, 0)
    expected = np.array([1, 1], dtype=complex) / np.sqrt(2)
    assert amps_close(q.amps, expected)


def test_x_gate_flips_bit():
    q = QuantumState(1)
    q.apply_1q(X, 0)
    expected = np.array([0, 1], dtype=complex)
    assert amps_close(q.amps, expected)


def test_cnot_on_10_gives_11():
    # qubit 0 is LSB. |10> in little-endian means qubit 1 = 1, qubit 0 = 0, so amp[2] = 1.
    q = QuantumState(2)
    q.amps[:] = 0.0
    q.amps[2] = 1.0
    # Apply CNOT with qubit 1 as control, qubit 0 as target.
    q.apply_controlled_1q(X, ctrl=1, target=0)
    # Expected: |11> in little-endian has qubit 1 = 1, qubit 0 = 1, so amp[3] = 1.
    expected = np.zeros(4, dtype=complex); expected[3] = 1.0
    assert amps_close(q.amps, expected)


def test_bell_state():
    # H on qubit 0, then CNOT(0 -> 1), produces (|00> + |11>)/sqrt(2)
    q = QuantumState(2)
    q.apply_1q(H, 0)
    q.apply_controlled_1q(X, ctrl=0, target=1)
    expected = np.zeros(4, dtype=complex)
    expected[0] = 1 / np.sqrt(2)  # |00>
    expected[3] = 1 / np.sqrt(2)  # |11>
    assert amps_close(q.amps, expected)


def test_gates_are_unitary():
    for U, name in [(H, "H"), (X, "X"), (Z, "Z")]:
        assert np.allclose(U @ U.conj().T, np.eye(2)), f"{name} not unitary"
    assert np.allclose(CNOT_MATRIX @ CNOT_MATRIX.conj().T, np.eye(4))
    for k in range(1, 5):
        Uk = Rk(k)
        assert np.allclose(Uk @ Uk.conj().T, np.eye(2)), f"Rk({k}) not unitary"


def test_measure_deterministic_classical_state():
    rng = np.random.default_rng(0)
    q = QuantumState(3)
    q.amps[:] = 0.0
    q.amps[5] = 1.0  # |101> in little-endian: qubit 0=1, qubit 1=0, qubit 2=1 -> integer 5
    assert q.measure([0, 1, 2], rng) == 5


def test_measure_bell_state_correlated():
    rng = np.random.default_rng(0)
    outcomes = []
    for _ in range(200):
        q = QuantumState(2)
        q.apply_1q(H, 0)
        q.apply_controlled_1q(X, ctrl=0, target=1)
        m = q.measure([0, 1], rng)
        outcomes.append(m)
    # Bell state: only 0 (|00>) or 3 (|11>) should appear.
    assert all(m in (0, 3) for m in outcomes)
    # Both should appear with roughly equal frequency given 200 trials.
    count0 = outcomes.count(0)
    assert 70 < count0 < 130, f"unbalanced: {count0}/200"


def test_apply_n_qubit_identity():
    rng = np.random.default_rng(1)
    q = QuantumState(3)
    # Put the state into a random superposition first.
    q.amps = rng.standard_normal(8) + 1j * rng.standard_normal(8)
    q.amps /= np.linalg.norm(q.amps)
    before = q.amps.copy()
    # Applying 8x8 identity should leave it unchanged.
    q.apply_n_qubit(np.eye(8), [0, 1, 2])
    assert amps_close(q.amps, before)


def test_apply_classical_function():
    # Place |x> = |3> on qubits {0,1} (so amp[3] = 1), and |y> = 0 on qubits {2,3}.
    q = QuantumState(4)
    q.amps[:] = 0.0
    q.amps[3] = 1.0  # qubit 0=1, qubit 1=1, qubits 2,3=0
    # Apply f(x) = x^2 mod 7 on y register.
    def f(x):
        return pow(x, 2, 7)
    q.apply_classical_function(f, x_qubits=[0, 1], y_qubits=[2, 3])
    # After: |x=3>|y=f(3)=2>. In little-endian encoding into the 4-bit integer:
    # qubits 0,1 carry x=3 -> bits 1,1. qubits 2,3 carry y=2 -> bits 0,1.
    # Integer index = q0 + 2*q1 + 4*q2 + 8*q3 = 1 + 2 + 0 + 8 = 11.
    expected = np.zeros(16, dtype=complex); expected[11] = 1.0
    assert amps_close(q.amps, expected)
