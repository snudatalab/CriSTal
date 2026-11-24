from __future__ import annotations
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


def directed_distances(adj: np.ndarray) -> np.ndarray:
    """Directed shortest-path cost matrix D[i, n] = cost from i to n.

    Edge cost is 1 / adjacency weight (stronger edge -> shorter).
    """
    cost = np.zeros_like(adj, dtype=np.float64)
    nz = adj > 0
    cost[nz] = 1.0 / np.maximum(adj[nz], 1e-6)
    return dijkstra(csr_matrix(cost), directed=True, indices=None)


def distances_from_cost(C: np.ndarray) -> np.ndarray:
    """Directed shortest-path distances from raw edge costs C (inf = no edge),
    so D sums physical distance along the path."""
    Cf = np.where(np.isfinite(C), C, 0.0)
    return dijkstra(csr_matrix(Cf), directed=True, indices=None)


def build_paths(adj: np.ndarray, J: int, K: int, detour_tol: float = 0.15,
                D: 'np.ndarray | None' = None):
    """Build up to J incoming propagation paths toward every query node.

    D : optional precomputed directed distance matrix. When None, it is derived
        from the adjacency (1 / weight); pass a physical-distance matrix to build
        corridors on real distances instead.

    Returns three (N, J, K) arrays:
      paths : node index of the k-th node on path j of query n (-1 = padding)
      dbar  : normalized directed distance to n, clipped to [0, 1]
      valid : 1 for real entries, 0 for padding
    """
    N = adj.shape[0]
    if D is None:
        D = directed_distances(adj)
    finite = np.isfinite(D) & (D > 0)
    dmax = np.percentile(D[finite], 95) if finite.any() else 1.0

    paths = -np.ones((N, J, K), dtype=np.int64)
    dbar = np.ones((N, J, K), dtype=np.float32)
    valid = np.zeros((N, J, K), dtype=np.float32)
    in_nb = [np.nonzero(adj[:, n] > 0)[0] for n in range(N)]

    for n in range(N):
        seeds = in_nb[n]
        if len(seeds) == 0:
            continue
        order = np.argsort(-adj[seeds, n])
        seeds = [s for s in seeds[order] if s != n][:J]
        for j, seed in enumerate(seeds):
            used = {n, seed}
            u = seed
            for k in range(K):
                paths[n, j, k] = u
                d = D[u, n]
                dbar[n, j, k] = min(d / dmax, 1.0) if np.isfinite(d) else 1.0
                valid[n, j, k] = 1.0
                cands = [v for v in in_nb[u] if v not in used]
                if not cands:
                    break
                best, best_score = None, None
                for v in cands:
                    dvn, dvu, dun = D[v, n], D[v, u], D[u, n]
                    if not (np.isfinite(dvn) and np.isfinite(dvu) and np.isfinite(dun)):
                        continue
                    detour = abs(dvn - (dvu + dun)) / max(dvn, 1e-9)
                    if detour <= detour_tol and (best_score is None or detour < best_score):
                        best, best_score = v, detour
                if best is None:
                    break
                used.add(best)
                u = best
    return paths, dbar, valid
