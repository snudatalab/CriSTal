"""CP lookup-table initialization for CriSTal.
"""
from __future__ import annotations
import numpy as np
import torch


def reconstruct(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
    return torch.einsum('bnr,btr,bdr->bntd', A, B, C)


def _init_factors(X, R):
    b, N, T, D = X.shape
    dev, dt = X.device, X.dtype
    node = X.mean(dim=(2, 3))
    tod = X.mean(dim=(1, 3))
    day = X.mean(dim=(1, 2))
    s = X.abs().mean(dim=(1, 2, 3)).clamp_min(1e-6)
    A = 0.01 * torch.randn(b, N, R, device=dev, dtype=dt)
    Bf = 0.01 * torch.randn(b, T, R, device=dev, dtype=dt)
    C = 0.01 * torch.randn(b, D, R, device=dev, dtype=dt)
    A[..., 0] = node / s[:, None]
    Bf[..., 0] = tod / s[:, None]
    C[..., 0] = day / s[:, None] * s[:, None]
    return A, Bf, C


def _solve(F, G, ridge=1e-6):
    R = G.shape[-1]
    eye = torch.eye(R, device=G.device, dtype=G.dtype)
    G = G + ridge * eye
    return torch.linalg.solve(G, F.transpose(1, 2)).transpose(1, 2)


def _solve_C_smooth(F, G, lam, ridge=1e-6):
    b, D, R = F.shape
    dev, dt = F.device, F.dtype
    if D == 1 or lam <= 0:
        return _solve(F, G, ridge)
    lap = torch.zeros(D, D, device=dev, dtype=dt)
    for d in range(D):
        if d > 0:
            lap[d, d] += 1.0
            lap[d, d - 1] -= 1.0
        if d < D - 1:
            lap[d, d] += 1.0
            lap[d, d + 1] -= 1.0
    evals, Qm = torch.linalg.eigh(lap)
    Fp = torch.einsum('de,ber->bdr', Qm.T, F)
    eye = torch.eye(R, device=dev, dtype=dt)
    Cp = torch.empty_like(Fp)
    for d in range(D):
        Gd = G + (lam * evals[d] + ridge) * eye
        Cp[:, d] = torch.linalg.solve(Gd, Fp[:, d].unsqueeze(-1)).squeeze(-1)
    return torch.einsum('de,ber->bdr', Qm, Cp)


@torch.no_grad()
def fit_cp(X: torch.Tensor, R: int, sweeps: int = 5, lam_smooth: float = 1e-3,
           factors=None):
    A, Bf, C = _init_factors(X, R) if factors is None else [f.clone() for f in factors]
    for _ in range(sweeps):
        G = torch.bmm(Bf.transpose(1, 2), Bf) * torch.bmm(C.transpose(1, 2), C)
        F = torch.einsum('bntd,btr,bdr->bnr', X, Bf, C)
        A = _solve(F, G)
        G = torch.bmm(A.transpose(1, 2), A) * torch.bmm(C.transpose(1, 2), C)
        F = torch.einsum('bntd,bnr,bdr->btr', X, A, C)
        Bf = _solve(F, G)
        G = torch.bmm(A.transpose(1, 2), A) * torch.bmm(Bf.transpose(1, 2), Bf)
        F = torch.einsum('bntd,bnr,btr->bdr', X, A, Bf)
        C = _solve_C_smooth(F, G, lam_smooth)
    return A, Bf, C


def _init_from_climatology(model, data_norm, n_train, T, R, sweeps=30, device='cpu'):
    L, N = data_norm.shape
    fine, coarse = model.fine_period, model.coarse_period
    idx = np.arange(min(n_train, L))
    tod, dow = idx % fine, (idx // fine) % coarse
    M = np.zeros((N, fine, coarse), dtype=np.float64)
    cnt = np.zeros((fine, coarse), dtype=np.int64)
    np.add.at(cnt, (tod, dow), 1)
    for t, d, i in zip(tod, dow, idx):
        M[:, t, d] += data_norm[i]
    M /= np.maximum(cnt, 1)[None]
    tod_mean = data_norm[idx].T @ np.eye(fine)[tod] / np.maximum(np.bincount(tod, minlength=fine), 1)
    empty = cnt == 0
    if empty.any():
        ti, di = np.nonzero(empty)
        M[:, ti, di] = tod_mean[:, ti]
    X = torch.from_numpy(M).float().unsqueeze(0).to(device)
    A, B, C = fit_cp(X, R, sweeps, 0.0)
    with torch.no_grad():
        model.fA.weight.copy_(A[0])
        model.fB.weight.copy_(B[0])
        model.fC.weight.copy_(C[0])


def init_cp_lookup(model, data_norm, n_train, T, R,
                   method='xavier', sweeps=30, device='cpu'):
    """Seed the CP lookup tables (fA / fB / fC) of `model`.

    method='xavier' applies Xavier-uniform random initialization;
    method='climatology' fits a rank-R CP to the training climatology.
    """
    if method == 'xavier':
        for e in (model.fA, model.fB, model.fC):
            torch.nn.init.xavier_uniform_(e.weight)
    else:
        _init_from_climatology(model, data_norm, n_train, T, R, sweeps, device)
