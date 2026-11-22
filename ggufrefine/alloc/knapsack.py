"""Exact multiple-choice knapsack for per-tensor type assignment.

Every item is one decision (a layer's group of tensors that must share a type) with options
(bytes, cost, label); solve() minimises total cost under a byte budget by DP on a byte grid. price() gives the
bytes of n weights at a ggml type from the format's block size, so new types or models need no table edits.
Costs are the caller's (e.g. held-out Hessian-weighted error, optionally scaled by measured KLD sensitivity).
"""
import numpy as np

from .. import paths  # noqa: F401
from gguf import GGMLQuantizationType as QT  # noqa: E402
from gguf.constants import GGML_QUANT_SIZES  # noqa: E402


def bpw(t):
    block, size = GGML_QUANT_SIZES[QT[t] if isinstance(t, str) else t]
    return size * 8 / block


def price(n_weights, t):
    """bytes of n_weights at type t (n_weights a multiple of the block size)"""
    block, size = GGML_QUANT_SIZES[QT[t] if isinstance(t, str) else t]
    assert n_weights % block == 0, (n_weights, t, block)
    return n_weights // block * size


def _cells(sz, grid):
    """grid cells an option occupies (ceiling; exact in integers when both are integers)"""
    if isinstance(sz, (int, np.integer)) and isinstance(grid, (int, np.integer)):
        return -(-int(sz) // int(grid))
    return int(np.ceil(sz / grid - 1e-9))


def solve(items, budget, grid):
    """items: [[(bytes, cost, label), ...], ...]; budget and grid in the same unit as bytes (integers for exact
    byte budgets). Returns (chosen option index per item, total cost). Each option's size is rounded up to the grid
    and the budget down, so the result never exceeds the budget."""
    if grid <= 0:
        raise ValueError("grid must be positive")
    B = int(budget) // int(grid) if isinstance(budget, (int, np.integer)) and isinstance(grid, (int, np.integer)) \
        else int(np.floor(budget / grid))
    best = np.full(B + 1, np.inf)
    best[0] = 0.0
    choice = []
    for opts in items:
        nb = np.full(B + 1, np.inf)
        arg = np.full(B + 1, -1, np.int32)
        for k, (sz, cost, _) in enumerate(opts):
            w = _cells(sz, grid)
            if w > B:
                continue
            cand = np.full(B + 1, np.inf)
            cand[w:] = best[:B + 1 - w] + cost
            m = cand < nb
            nb[m], arg[m] = cand[m], k
        best = nb
        choice.append(arg)
    b = int(np.argmin(best))
    if not np.isfinite(best[b]):
        raise ValueError("infeasible budget")
    pick = []
    for opts, arg in zip(reversed(items), reversed(choice)):
        k = int(arg[b])
        pick.append(k)
        b -= _cells(opts[k][0], grid)
    return list(reversed(pick)), float(best.min())
