from __future__ import annotations
import numpy as np

ArrayLike = np.ndarray


def bellman_update(v: ArrayLike, P: ArrayLike, r: ArrayLike, gamma: float) -> ArrayLike:
    """
    Apply one Bellman update for a fixed policy.
    Vectorized: v_new = r + gamma * (P @ v)

    Parameters
    ----------
    v : (S,) array
        Current value estimates.
    P : (S,S) array
        Policy-induced transition matrix (row-stochastic).
    r : (S,) array
        Reward-on-entry vector aligned to the state indexing.
    gamma : float in (0,1]
        Discount factor.

    Returns
    -------
    v_new : (S,) array
    """

    v = np.asarray(v)
    P = np.asarray(P)
    r = np.asarray(r)
    gamma = np.asarray(gamma)

    v_new= r + gamma * (P @ v)

    return v_new


def exact_policy_evaluation(P: ArrayLike, r: ArrayLike, gamma: float) -> ArrayLike:
    """
    Solve (I - gamma P) v = r for v.

    Parameters
    ----------
    P : (S,S) array
    r : (S,) array
    gamma : float in (0,1]

    Returns
    -------
    v : (S,) array
    """
    P = np.asarray(P)
    r = np.asarray(r)
    I = np.eye(P.shape[0])
    v = np.linalg.solve(I - gamma * P, r)
    return v
