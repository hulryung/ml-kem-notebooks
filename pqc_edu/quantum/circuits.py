"""Quantum circuit building blocks: QFT, inverse QFT, and reversible
classical arithmetic (modular exponentiation) used by Shor's algorithm.

The QFT here is the standard textbook definition:
    QFT|j> = (1/sqrt(N)) sum_{k=0}^{N-1} exp(2*pi*i*j*k/N) |k>
with little-endian qubit ordering: qubits[0] is the least-significant bit.
"""
from __future__ import annotations
import numpy as np
from .simulator import QuantumState


def qft_matrix(n: int) -> np.ndarray:
    """Return the 2^n x 2^n QFT matrix."""
    N = 1 << n
    j = np.arange(N).reshape(-1, 1)
    k = np.arange(N).reshape(1, -1)
    omega = np.exp(2j * np.pi / N)
    return omega ** (j * k) / np.sqrt(N)


def qft(state: QuantumState, qubits: list[int]) -> None:
    """Apply the QFT to the specified qubits, in place."""
    n = len(qubits)
    state.apply_n_qubit(qft_matrix(n), qubits)


def inverse_qft(state: QuantumState, qubits: list[int]) -> None:
    """Apply the inverse QFT (= QFT^dagger) to the specified qubits."""
    n = len(qubits)
    state.apply_n_qubit(qft_matrix(n).conj().T, qubits)


def modular_exp_circuit(
    state: QuantumState,
    a: int,
    N: int,
    x_qubits: list[int],
    y_qubits: list[int],
) -> None:
    """Apply the reversible transform |x>|y> -> |x>|y XOR (a^x mod N)>.

    Educational shortcut: we implement this as a single state-vector
    update (via apply_classical_function) rather than decomposing into
    quantum reversible arithmetic gates. The result is algorithmically
    identical - the simulation reproduces the superposition
    (1/sqrt(2^n)) sum_x |x>|a^x mod N> that a real quantum modular
    exponentiation circuit would produce - but the simulator spends
    O(2^n) classical work per invocation rather than building a
    polynomial-size circuit. This is flagged in notebook 14.
    """
    def f(x):
        return pow(a, x, N)
    state.apply_classical_function(f, x_qubits, y_qubits)
