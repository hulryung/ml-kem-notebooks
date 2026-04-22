"""Classical post-processing for Shor's algorithm.

The quantum circuit produces an integer `measured` from a
`counting_qubits`-bit register. This integer encodes a rational
approximation to s/r, where r is the period (multiplicative order
of a modulo N). We recover r via continued-fraction expansion.
"""
from __future__ import annotations
import math
from fractions import Fraction


def gcd(a: int, b: int) -> int:
    """Euclidean gcd. Wraps math.gcd for explicitness; we use it across Shor."""
    return math.gcd(a, b)


def continued_fraction_denominator(
    measured: int,
    counting_qubits: int,
    N: int,
) -> int | None:
    """Recover the period r from a measured phase m / 2^q.

    Given that the quantum circuit's output m satisfies m/2^q ≈ s/r for
    some integer s, the best rational approximation with denominator
    ≤ N gives us r. Returns r, or None if the convergents never land
    on a plausible period (r=1 is rejected).
    """
    if measured == 0:
        return None
    q = counting_qubits
    denom = 1 << q
    frac = Fraction(measured, denom)
    # Continued-fraction expansion: take convergents until denominator > N.
    a = []
    x = frac
    for _ in range(64):  # safety bound
        a_k = int(x)
        a.append(a_k)
        remainder = x - a_k
        if remainder == 0:
            break
        x = 1 / remainder
    # Build convergents p_k/q_k incrementally.
    p_prev, p_curr = 1, a[0]
    q_prev, q_curr = 0, 1
    best_r = None
    if a[0] == 0:
        # Skip the leading zero convergent.
        pass
    for k in range(1, len(a)):
        p_next = a[k] * p_curr + p_prev
        q_next = a[k] * q_curr + q_prev
        if q_next > N:
            break
        p_prev, p_curr = p_curr, p_next
        q_prev, q_curr = q_curr, q_next
        if q_curr > 1:
            best_r = q_curr
    return best_r
