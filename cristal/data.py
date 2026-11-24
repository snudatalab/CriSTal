from __future__ import annotations
import csv
import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset

STEPS_PER_DAY = 288  # 5-minute datasets


def _load_pickle(path):
    with open(path, 'rb') as f:
        try:
            return pickle.load(f)
        except UnicodeDecodeError:
            f.seek(0)
            return pickle.load(f, encoding='latin1')


def load_dataset(name: str, data_dir: str):
    """Returns (data (L, N) float32, adj (N, N) float32 weighted directed).

    PEMS04/07/08 : PEMSxx.npz (key 'data', channel 0) + PEMSxx.csv (from,to,cost)
    METR-LA / PEMS-BAY : speed_data.npy (or the h5) + adj_mx*.pkl (DCRNN release)
    KnowAir : data.npy (PM2.5, or the raw (L,N,C) with PM2.5 in channel 17) + adj.npy
    LamaH   : qobs.npy (streamflow) + adj.npy
    """
    name = name.lower().replace('_', '-')
    if name == 'knowair':
        d = np.load(os.path.join(data_dir, 'data.npy'))
        data = (d[..., 17] if d.ndim == 3 else d).astype(np.float32)   # PM2.5
        adj = np.load(os.path.join(data_dir, 'adj.npy')).astype(np.float32)
        np.fill_diagonal(adj, 0.0)
        return data, adj
    if name == 'lamah':
        data = np.load(os.path.join(data_dir, 'qobs.npy')).astype(np.float32)
        adj = np.load(os.path.join(data_dir, 'adj.npy')).astype(np.float32)
        np.fill_diagonal(adj, 0.0)
        return data, adj
    if name in ('metr-la', 'pems-bay'):
        npy = os.path.join(data_dir, 'speed_data.npy')
        if os.path.exists(npy):
            data = np.load(npy).astype(np.float32)
        else:
            import pandas as pd
            data = pd.read_hdf(os.path.join(data_dir, f'{name}.h5')).values.astype(np.float32)
        pkl = 'adj_mx.pkl' if name == 'metr-la' else 'adj_mx_bay.pkl'
        obj = _load_pickle(os.path.join(data_dir, pkl))
        adj = np.asarray(obj[-1] if isinstance(obj, (list, tuple)) else obj, dtype=np.float32)
        np.fill_diagonal(adj, 0.0)
        return data, adj
    if name in ('pems04', 'pems07', 'pems08'):
        num = name[-2:]
        data = np.load(os.path.join(data_dir, f'PEMS{num}.npz'))['data'][..., 0].astype(np.float32)
        N = data.shape[1]
        adj = np.zeros((N, N), dtype=np.float32)
        with open(os.path.join(data_dir, f'PEMS{num}.csv')) as f:
            reader = csv.reader(f)
            header = next(reader)
            if header and header[0].strip().isdigit():
                f.seek(0)
                reader = csv.reader(f)
            rows = [(int(r[0]), int(r[1]), float(r[2])) for r in reader]
        sigma = np.std([c for _, _, c in rows]) + 1e-6
        for i, j, c in rows:
            adj[i, j] = max(adj[i, j], float(np.exp(-(c / sigma) ** 2)))
        return data, adj
    raise ValueError(f'unknown dataset {name}')


def _haversine(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = (np.sin((p2 - p1) / 2) ** 2
         + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2)
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def load_edge_cost(name: str, data_dir: str, adj: np.ndarray):
    """Physical directed edge costs C (N, N), inf where there is no edge.

    Used when a dataset's adjacency encodes connectivity rather than distance
    (KnowAir / LamaH): the propagation corridors are then built on real physical
    distances. Returns None if the required coordinate file is missing.
    """
    name = name.lower().replace('_', '-')
    N = adj.shape[0]
    if name == 'knowair':
        path = os.path.join(data_dir, 'city.txt')      # index name lon lat
        if not os.path.exists(path):
            return None
        lon = np.zeros(N); lat = np.zeros(N)
        with open(path) as f:
            for line in f:
                p = line.split()
                if len(p) >= 4 and int(p[0]) < N:
                    lon[int(p[0])], lat[int(p[0])] = float(p[-2]), float(p[-1])
        C = np.full((N, N), np.inf)
        ii, jj = np.nonzero(adj > 0)
        C[ii, jj] = _haversine(lat[ii], lon[ii], lat[jj], lon[jj])
        return C
    if name == 'lamah':
        p = os.path.join(data_dir, 'coords.npy')        # (N, 2) projected metres
        if not os.path.exists(p):
            return None
        xy = np.load(p).astype(np.float64)
        C = np.full((N, N), np.inf)
        ii, jj = np.nonzero(adj > 0)
        C[ii, jj] = np.hypot(xy[ii, 0] - xy[jj, 0], xy[ii, 1] - xy[jj, 1])
        return C
    return None


class Scaler:
    def __init__(self, mean, std):
        self.mean, self.std = float(mean), float(std)

    def transform(self, x):
        return (x - self.mean) / self.std

    def inverse(self, x):
        return x * self.std + self.mean


def compute_sigma(train_data: np.ndarray, T: int = STEPS_PER_DAY,
                  min_count: int = 4, eps: float = 1e-3) -> np.ndarray:
    L, N = train_data.shape
    idx = np.arange(L)
    tod = idx % T
    dow = (idx // T) % 7
    key = dow * T + tod
    sig = np.zeros((7 * T, N), dtype=np.float32)
    sig_tod = np.zeros((T, N), dtype=np.float32)
    for t in range(T):
        rows = train_data[tod == t]
        sig_tod[t] = rows.std(axis=0) if len(rows) > 1 else 1.0
    for k in range(7 * T):
        rows = train_data[key == k]
        sig[k] = rows.std(axis=0) if len(rows) >= min_count else sig_tod[k % T]
    return np.maximum(sig, eps)


class STWindowDataset(Dataset):
    def __init__(self, data_norm, data_raw, sigma, starts, P, Q, W,
                 T=STEPS_PER_DAY, period_days=0, null_val=None):
        self.xn = data_norm
        self.yr = data_raw
        self.sigma = sigma
        self.starts = starts
        self.P, self.Q, self.W, self.T = P, Q, W, T
        self.period_days = period_days
        self.null_val = null_val

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        s = self.starts[i]
        P, Q, W, T = self.P, self.Q, self.W, self.T
        x = self.xn[s:s + P].T
        y = self.yr[s + P:s + P + Q].T
        last = np.arange(s + P - W, s + P)
        key = ((last // T) % 7) * T + (last % T)
        sw = self.sigma[key].T
        out = (torch.from_numpy(np.ascontiguousarray(x)),
               torch.from_numpy(np.ascontiguousarray(y)),
               torch.from_numpy(np.ascontiguousarray(sw)),
               int(s))
        if self.period_days > 0:
            D = self.period_days
            i0 = max(1, -(-Q // T))                 # ceil(Q / T): keep indices in the past
            idx = (s + P + np.arange(Q)[None, :] - np.arange(i0, i0 + D)[:, None] * T)
            idx = np.clip(idx, 0, self.xn.shape[0] - 1)
            xp = self.xn[idx].transpose(2, 0, 1)    # (N, D, Q)
            if self.null_val is not None:
                ok = (self.yr[idx] != self.null_val).transpose(2, 0, 1).astype(xp.dtype)
                xp = np.concatenate([xp * ok, ok], axis=1)   # (N, 2D, Q)
            out = out + (torch.from_numpy(np.ascontiguousarray(xp)),)
        return out


def make_splits(name, data, P, Q):
    L = data.shape[0]
    name = name.lower().replace('_', '-')
    r = (0.7, 0.1, 0.2) if name in ('metr-la', 'pems-bay') else (0.6, 0.2, 0.2)
    n_train = int(L * r[0])
    n_val = int(L * r[1])
    tr = np.arange(0, n_train - P - Q + 1)
    va = np.arange(n_train - P, n_train + n_val - P - Q + 1)
    te = np.arange(n_train + n_val - P, L - P - Q + 1)
    return n_train, tr, va[va >= 0], te[te >= 0]
