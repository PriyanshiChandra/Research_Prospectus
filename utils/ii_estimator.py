"""
utils/ii_estimator.py
---------------------
Canonical implementation of the Information Imbalance (II) estimator.
Imported by all experiment scripts — do not duplicate this logic elsewhere.

Definition (rank-based, non-parametric):
    II(X -> Y) measures how much knowing the rank structure of X
    reduces uncertainty about Y, relative to knowing Y's own rank structure.

    II^n = 1 - (2 / n^2) * sum_{i,j} |rank_Y(i, knn_X(j)) - rank_Y(i, j)|
    (exact form depends on your kernel — adjust _ii_kernel if needed)

References:
    - Your dissertation notation (update this as the proof is finalised)
    - Lin and Han (2022) framework for Hájek projection
"""

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist


def _rank_distances(source: np.ndarray, k: int) -> np.ndarray:
    """
    For each point in source, find the rank of every other point
    by distance from that point.

    Parameters
    ----------
    source : (n, d) array
    k      : number of nearest neighbours to use

    Returns
    -------
    knn_indices : (n, k) array of nearest-neighbour indices (excluding self)
    """
    tree = cKDTree(source)
    # k+1 because cKDTree includes the point itself
    distances, indices = tree.query(source, k=k + 1)
    return indices[:, 1:]  # drop self


def compute_ii(
    X: np.ndarray,
    Y: np.ndarray,
    k: int = 1,
) -> float:
    """
    Estimate II(X -> Y): the Information Imbalance from X to Y.

    Parameters
    ----------
    X : (n, d_X) array — source variable
    Y : (n, d_Y) array — target variable
    k : int — number of nearest neighbours (default 1)

    Returns
    -------
    ii : float in [0, 1]
         0 means X predicts Y perfectly (in rank sense)
         1 means X carries no information about Y beyond chance
         Values near 0.5 indicate moderate dependence

    Notes
    -----
    The estimator computes, for each point i, the rank of point i's
    k-NN in X-space when re-ranked in Y-space. Small ranks mean
    X-neighbours are also Y-neighbours — i.e. X predicts Y.
    """
    X = np.atleast_2d(X) if X.ndim == 1 else X
    Y = np.atleast_2d(Y) if Y.ndim == 1 else Y

    n = X.shape[0]
    assert Y.shape[0] == n, "X and Y must have the same number of rows."

    # Step 1: find k-NN of each point in X-space
    knn_in_X = _rank_distances(X, k=k)   # (n, k)

    # Step 2: build KD-tree in Y-space to rank those neighbours
    tree_Y = cKDTree(Y)

    ii_sum = 0.0
    for i in range(n):
        # For point i, find its k-NN in X (the "predicted" neighbours)
        nn_idx = knn_in_X[i, 0]  # using k=1: single nearest neighbour

        # Rank of nn_idx when points are sorted by distance from i in Y-space
        # i.e. what rank does the X-neighbour get in Y?
        distances_from_i_in_Y, _ = tree_Y.query(Y[i], k=n)
        # distances_from_i_in_Y is sorted; find where nn_idx lands
        # re-query: get rank of Y[nn_idx] from Y[i]
        dist_to_nn_in_Y = np.linalg.norm(Y[i] - Y[nn_idx])
        # rank = number of points closer to i in Y than nn_idx is
        rank = np.sum(np.linalg.norm(Y - Y[i], axis=1) < dist_to_nn_in_Y)
        ii_sum += rank

    # Normalise to [0, 1]: divide by n*(n-1)/2 (maximum possible sum)
    ii = (2.0 * ii_sum) / (n * (n - 1))
    return float(ii)


def compute_ii_vectorized(
    X: np.ndarray,
    Y: np.ndarray,
    k: int = 1,
) -> float:
    """
    Vectorized II estimator — use this in batch/MC experiments instead of compute_ii.

    Implements Setup.md Step 1-3 exactly:
        II_hat = (2 / n^2) * sum_i  R^Y_i(N_X(i))
        R^Y_i  = #{j != i : ||y_j - y_i|| <= ||y_{N_X(i)} - y_i||}

    Avoids the per-point inner loop in compute_ii and removes the dead
    tree_Y.query call, reducing per-replication cost from O(n^2 log n)
    to O(n^2 * d_Y).  Peak memory: one (n, n) float64 distance matrix
    (~200 MB at n=5000).

    Parameters
    ----------
    X : (n, d_X)
    Y : (n, d_Y)
    k : nearest-neighbour order (default 1)

    Returns
    -------
    ii : float in [0, 1]
    """
    X = np.atleast_2d(X) if X.ndim == 1 else X
    Y = np.atleast_2d(Y) if Y.ndim == 1 else Y
    n = X.shape[0]
    assert Y.shape[0] == n, "X and Y must have the same number of rows."

    # Step 1 — 1-NN in X-space (k+1 to skip self at index 0)
    tree_X = cKDTree(X)
    _, indices = tree_X.query(X, k=k + 1)
    nn_idx = indices[:, 1]                          # (n,)

    # Step 2 — distance from each point to its X-NN in Y-space
    d_nn = np.linalg.norm(Y[nn_idx] - Y, axis=1)   # (n,)

    # Step 3 — full pairwise Y-distances; count ranks
    D_Y = cdist(Y, Y, metric='euclidean')           # (n, n)
    np.fill_diagonal(D_Y, np.inf)                   # exclude self

    ranks = np.sum(D_Y <= d_nn[:, None], axis=1)    # (n,)
    return 2.0 / n ** 2 * float(np.sum(ranks))


def compute_ii_both_directions(
    X: np.ndarray,
    Y: np.ndarray,
    k: int = 1,
) -> tuple:
    """
    Compute II(X->Y) and II(Y->X) in one call.

    Returns
    -------
    (ii_xy, ii_yx) : tuple of floats
    """
    return compute_ii(X, Y, k=k), compute_ii(Y, X, k=k)
