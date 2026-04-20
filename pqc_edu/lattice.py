"""2-D and 3-D lattice helpers used by the first notebook.

A lattice L is the set {B @ z : z in Z^d} for a basis matrix B. This
module provides generators for lattice points and a few plotting
helpers built on matplotlib.
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from itertools import product


def lattice_points(basis: np.ndarray, radius: int):
    """Yield lattice points B @ z for z with |z_i| <= radius."""
    d = basis.shape[0]
    for z in product(range(-radius, radius + 1), repeat=d):
        yield basis @ np.array(z)


def plot_lattice_2d(basis: np.ndarray, radius: int = 5, target: np.ndarray | None = None, ax=None):
    """Scatter plot a 2-D lattice. If `target` is given, also show the closest lattice point."""
    assert basis.shape == (2, 2)
    pts = np.array(list(lattice_points(basis, radius)))
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(pts[:, 0], pts[:, 1], s=8, color="steelblue")
    ax.axhline(0, color="lightgray", lw=0.5)
    ax.axvline(0, color="lightgray", lw=0.5)
    # Draw basis vectors.
    for v, color in zip(basis.T, ("red", "green")):
        ax.annotate("", xy=v, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=color))
    if target is not None:
        dists = np.linalg.norm(pts - target, axis=1)
        closest = pts[np.argmin(dists)]
        ax.scatter(*target, marker="*", color="orange", s=100, label="target")
        ax.scatter(*closest, marker="x", color="red", s=80, label="closest lattice point")
        ax.legend()
    ax.set_aspect("equal")
    return ax


def good_vs_bad_basis():
    """Return one 'good' (near-orthogonal) and one 'bad' (skewed) basis for the same lattice."""
    good = np.array([[1.0, 0.0], [0.0, 1.0]])
    # The 'bad' basis spans the same lattice but has large vectors at a shallow angle.
    bad = np.array([[3.0, 1.0], [5.0, 2.0]])
    return good, bad
