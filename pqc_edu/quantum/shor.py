"""Shor's algorithm: full loop of quantum order finding + classical
post-processing, with retries on bad random `a` choices.

Parameters chosen for educational clarity:
- counting_qubits = 8 (enough to resolve periods r up to 2^4 = 16,
  which covers multiplicative orders modulo N <= 35).
- target register has ceil(log2(N)) qubits.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .simulator import QuantumState, H, X
from .circuits import qft, inverse_qft, modular_exp_circuit
from .classical_utils import continued_fraction_denominator, gcd


COUNTING_QUBITS = 8


@dataclass
class ShorAttempt:
    a: int
    measured: int
    inferred_period: Optional[int]
    status: str   # "success" | "a_not_coprime" | "odd_period" | "trivial_factor" | "period_not_found"


@dataclass
class ShorResult:
    N: int
    factors: Optional[tuple[int, int]]
    attempts: list[ShorAttempt] = field(default_factory=list)
    total_quantum_runs: int = 0


def _quantum_order_finding(a: int, N: int, rng) -> int:
    """Run one quantum order-finding subroutine. Returns the measured integer
    from the counting register (0..2^COUNTING_QUBITS - 1)."""
    target_qubits_n = max(1, math.ceil(math.log2(N)))
    counting = list(range(COUNTING_QUBITS))
    target = list(range(COUNTING_QUBITS, COUNTING_QUBITS + target_qubits_n))
    total = COUNTING_QUBITS + target_qubits_n

    state = QuantumState(total)
    # Initialize target register to |1>: flip first target qubit with X.
    state.apply_1q(X, target[0])

    # Hadamard on every counting qubit.
    for q in counting:
        state.apply_1q(H, q)

    # Modular exponentiation: |x>|1> -> |x>|a^x mod N>
    modular_exp_circuit(state, a, N, x_qubits=counting, y_qubits=target)

    # Inverse QFT on counting register.
    inverse_qft(state, counting)

    return state.measure(counting, rng)


def factor(N: int, rng, max_attempts: int = 10) -> ShorResult:
    """Shor's algorithm. Returns factors of N (or None if all attempts fail)."""
    if N < 2:
        raise ValueError(f"N must be >= 2, got {N}")
    if N % 2 == 0:
        return ShorResult(N=N, factors=(2, N // 2))

    result = ShorResult(N=N, factors=None)

    for _ in range(max_attempts):
        a = int(rng.integers(2, N))  # choose random a in [2, N-1]

        g = gcd(a, N)
        if g > 1:
            # Got lucky: a shares a factor with N.
            result.attempts.append(
                ShorAttempt(a=a, measured=-1, inferred_period=None, status="a_not_coprime")
            )
            result.factors = (g, N // g)
            return result

        measured = _quantum_order_finding(a, N, rng)
        result.total_quantum_runs += 1

        r = continued_fraction_denominator(measured, COUNTING_QUBITS, N)
        if r is None or r == 0 or pow(a, r, N) != 1:
            result.attempts.append(
                ShorAttempt(a=a, measured=measured, inferred_period=r, status="period_not_found")
            )
            continue

        if r % 2 == 1:
            result.attempts.append(
                ShorAttempt(a=a, measured=measured, inferred_period=r, status="odd_period")
            )
            continue

        # r is even -> try to extract factors.
        x = pow(a, r // 2, N)
        if x == N - 1:  # trivial square root
            result.attempts.append(
                ShorAttempt(a=a, measured=measured, inferred_period=r, status="trivial_factor")
            )
            continue

        p = gcd(x - 1, N)
        q = gcd(x + 1, N)
        if p > 1 and p < N:
            result.attempts.append(
                ShorAttempt(a=a, measured=measured, inferred_period=r, status="success")
            )
            result.factors = (p, N // p)
            return result
        if q > 1 and q < N:
            result.attempts.append(
                ShorAttempt(a=a, measured=measured, inferred_period=r, status="success")
            )
            result.factors = (q, N // q)
            return result

        # Neither p nor q is a non-trivial factor — retry.
        result.attempts.append(
            ShorAttempt(a=a, measured=measured, inferred_period=r, status="trivial_factor")
        )

    return result
