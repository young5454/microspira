"""Platform-agnostic spatial neighbour graphs."""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree


def build_knn_graph(coords: np.ndarray, n_neighbors: int = 6) -> sparse.csr_matrix:
    """Build a symmetric binary k-nearest-neighbour graph from coordinates."""
    xy = np.asarray(coords, dtype=float)
    if xy.ndim != 2 or xy.shape[1] < 2:
        raise ValueError("coords must have shape (n_units, >=2).")
    n = xy.shape[0]
    if n < 3:
        raise ValueError("At least three spatial units are required to build a graph.")
    if not 1 <= n_neighbors < n:
        raise ValueError("n_neighbors must be between 1 and n_units - 1.")

    tree = cKDTree(xy[:, :2])
    _, neighbours = tree.query(xy[:, :2], k=n_neighbors + 1)
    row = np.repeat(np.arange(n), n_neighbors)
    col = neighbours[:, 1:].reshape(-1)
    data = np.ones(len(row), dtype=float)
    W = sparse.coo_matrix((data, (row, col)), shape=(n, n)).tocsr()
    W = W.maximum(W.T)
    W.setdiag(0.0)
    W.eliminate_zeros()
    if W.nnz == 0:
        raise ValueError("Spatial graph has no edges.")
    return W


def induced_subgraph(W: sparse.csr_matrix, mask: np.ndarray) -> sparse.csr_matrix:
    """Return the graph induced by a boolean node mask."""
    m = np.asarray(mask, dtype=bool)
    return W[m][:, m].tocsr()

