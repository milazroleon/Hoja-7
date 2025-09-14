from __future__ import annotations
import numpy as np

from bellman import bellman_update


def iterative_policy_evaluation(
    P: np.ndarray,
    r: np.ndarray,
    gamma: float,
    eps: float = 1e-6,
    max_iters: int = 100000,
) -> np.ndarray:
    """
    Iterative (incremental) policy evaluation via repeated Bellman updates.
    Stop when ||v_new - v||_inf < eps * (1 - gamma) / gamma

    Parameters
    ----------
    P : (S,S) array (row-stochastic)
    r : (S,) array
    gamma : float in (0,1]
    eps : float
        Target error tolerance (on true error via slide bound).
    max_iters : int

    Returns
    -------
    v : (S,) array
    """
    P = np.asarray(P)
    r = np.asarray(r)
    gamma = float(gamma)

    v = np.zeros_like(r)
    for _ in range(max_iters):
        v_new = bellman_update(v, P, r, gamma)
        if np.max(np.abs(v_new - v)) < eps * (1 - gamma) / gamma:
            break
        v = v_new

    return v
