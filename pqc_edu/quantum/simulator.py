"""Quantum state-vector simulator.

Represents an n-qubit state as a length-2^n complex amplitude vector.
Qubit indexing is little-endian: qubit 0 contributes bit 0 of the
integer index. Applying a gate updates amplitudes in place.
"""
from __future__ import annotations
import numpy as np


# ---- Standard 2x2 gates ------------------------------------------------
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

# 4x4 CNOT where qubit 0 is control, qubit 1 is target, in the basis
# ordering (q1, q0) -> index = 2*q1 + q0. We expose it for reference;
# most code paths should use apply_controlled_1q(X, ctrl, target).
CNOT_MATRIX = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0],
], dtype=complex)


def Rk(k: int) -> np.ndarray:
    """Phase gate diag(1, exp(2*pi*i / 2^k)). Used inside the QFT."""
    return np.array(
        [[1, 0], [0, np.exp(2j * np.pi / (2 ** k))]],
        dtype=complex,
    )


# ---- The state ---------------------------------------------------------
class QuantumState:
    """n-qubit pure state. Amplitudes stored as a length-2^n complex vector."""

    __slots__ = ("n", "amps")

    def __init__(self, n: int):
        if n < 1 or n > 20:
            raise ValueError(f"n must be in [1, 20], got {n}")
        self.n = n
        self.amps = np.zeros(2 ** n, dtype=complex)
        self.amps[0] = 1.0

    # --- single-qubit gate ----------------------------------------------
    def apply_1q(self, U: np.ndarray, target: int) -> None:
        """Apply 2x2 unitary U to qubit `target`, in place."""
        if U.shape != (2, 2):
            raise ValueError(f"U must be 2x2, got {U.shape}")
        dim = 1 << self.n
        bit = 1 << target
        new_amps = self.amps.copy()
        # For every pair (i0, i1) differing only in the target bit, update amplitudes.
        for i in range(dim):
            if i & bit:
                continue
            i0 = i
            i1 = i | bit
            a0, a1 = self.amps[i0], self.amps[i1]
            new_amps[i0] = U[0, 0] * a0 + U[0, 1] * a1
            new_amps[i1] = U[1, 0] * a0 + U[1, 1] * a1
        self.amps = new_amps

    # --- controlled single-qubit gate -----------------------------------
    def apply_controlled_1q(self, U: np.ndarray, ctrl: int, target: int) -> None:
        """Apply U to `target` qubit when `ctrl` qubit is |1>; identity otherwise."""
        if U.shape != (2, 2):
            raise ValueError(f"U must be 2x2, got {U.shape}")
        if ctrl == target:
            raise ValueError("ctrl and target must differ")
        dim = 1 << self.n
        ctrl_bit = 1 << ctrl
        target_bit = 1 << target
        new_amps = self.amps.copy()
        for i in range(dim):
            if (i & ctrl_bit) == 0:
                continue  # ctrl is 0 -> no change
            if i & target_bit:
                continue  # will handle by pair with target-bit=0
            i0 = i
            i1 = i | target_bit
            a0, a1 = self.amps[i0], self.amps[i1]
            new_amps[i0] = U[0, 0] * a0 + U[0, 1] * a1
            new_amps[i1] = U[1, 0] * a0 + U[1, 1] * a1
        self.amps = new_amps

    # --- generic multi-qubit gate ---------------------------------------
    def apply_n_qubit(self, U: np.ndarray, targets: list[int]) -> None:
        """Apply 2^k x 2^k unitary U to the k qubits in `targets`.

        targets[i] becomes bit i of the internal k-qubit index used to
        address rows/columns of U (little-endian).
        """
        k = len(targets)
        expected = 2 ** k
        if U.shape != (expected, expected):
            raise ValueError(f"U must be {expected}x{expected}, got {U.shape}")
        dim = 1 << self.n
        new_amps = np.zeros(dim, dtype=complex)
        for idx in range(dim):
            amp = self.amps[idx]
            if amp == 0:
                continue
            t_val = 0
            for i, t in enumerate(targets):
                if (idx >> t) & 1:
                    t_val |= (1 << i)
            for out_val in range(expected):
                factor = U[out_val, t_val]
                if factor == 0:
                    continue
                new_idx = idx
                for i, t in enumerate(targets):
                    if (out_val >> i) & 1:
                        new_idx |= (1 << t)
                    else:
                        new_idx &= ~(1 << t)
                new_amps[new_idx] += amp * factor
        self.amps = new_amps

    # --- classical reversible function ----------------------------------
    def apply_classical_function(self, f, x_qubits: list[int], y_qubits: list[int]) -> None:
        """Unitary: |x>|y> -> |x>|y XOR f(x)>.

        f maps an integer (LSB-first across x_qubits) to a non-negative integer
        that fits in y_qubits. This is how we simulate reversible classical
        arithmetic (like modular exponentiation) inside the quantum circuit.
        Not a quantum-gate circuit — a single state-vector update.
        """
        def read_bits(idx, qubits):
            val = 0
            for i, q in enumerate(qubits):
                if (idx >> q) & 1:
                    val |= (1 << i)
            return val

        def write_bits(idx, qubits, val):
            out = idx
            for i, q in enumerate(qubits):
                if (val >> i) & 1:
                    out |= (1 << q)
                else:
                    out &= ~(1 << q)
            return out

        dim = 1 << self.n
        new_amps = np.zeros(dim, dtype=complex)
        for idx in range(dim):
            amp = self.amps[idx]
            if amp == 0:
                continue
            x = read_bits(idx, x_qubits)
            y = read_bits(idx, y_qubits)
            fx = f(x) & ((1 << len(y_qubits)) - 1)  # fit into y_qubits
            new_y = y ^ fx
            new_idx = write_bits(idx, y_qubits, new_y)
            new_amps[new_idx] += amp
        self.amps = new_amps

    # --- measurement ----------------------------------------------------
    def measure(self, qubits: list[int], rng) -> int:
        """Measure the specified qubits in the computational basis.

        Returns the measured integer (LSB first across the `qubits` list)
        and collapses the state accordingly.
        """
        k = len(qubits)
        probs = np.zeros(1 << k)
        dim = 1 << self.n
        for idx in range(dim):
            v = 0
            for i, q in enumerate(qubits):
                if (idx >> q) & 1:
                    v |= (1 << i)
            probs[v] += abs(self.amps[idx]) ** 2
        total = probs.sum()
        if abs(total - 1.0) > 1e-8:
            raise RuntimeError(f"probabilities sum to {total}, expected 1.0")
        probs = probs / total
        outcome = int(rng.choice(1 << k, p=probs))

        # Collapse.
        new_amps = np.zeros_like(self.amps)
        norm = 0.0
        for idx in range(dim):
            v = 0
            for i, q in enumerate(qubits):
                if (idx >> q) & 1:
                    v |= (1 << i)
            if v == outcome:
                new_amps[idx] = self.amps[idx]
                norm += abs(self.amps[idx]) ** 2
        self.amps = new_amps / np.sqrt(norm)
        return outcome
