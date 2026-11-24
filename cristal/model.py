"""CriSTal: conditional, propagation-aware spatio-temporal forecasting.

Step 1  Time-folded self diagnosis   -> self forecast, gate g, residual query q
Step 2  Query-adaptive spatio-folded evidence tensor 
Step 3  Propagation-kernel encoding + query attention -> spatial correction S
Step 4  Gated fusion  y = y_self + g * S   (hard-routed at inference)
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(d_in, d_hidden, d_out, layers=2):
    mods, d = [], d_in
    for _ in range(layers - 1):
        mods += [nn.Linear(d, d_hidden), nn.ReLU()]
        d = d_hidden
    mods += [nn.Linear(d, d_out)]
    return nn.Sequential(*mods)


def build_templates(Kphi: int, Pphi: int):
    """Binary propagation templates over the local (upstream-offset, lag) window.

    sync : nearby nodes, recent lags        fast/slow : diagonal distance-lag delay
    sust : a wider temporal band (persistent upstream influence)
    """
    a = torch.arange(Kphi).view(Kphi, 1).float()
    b = torch.arange(Pphi).view(1, Pphi).float()
    sync = ((a <= max(1, Kphi // 2)) & (b <= max(1, Pphi // 2))).float()
    fast = ((b - 0.5 * a >= 0) & (b - 0.5 * a <= 1.0)).float()
    slow = ((b - 1.5 * a >= 0) & (b - 1.5 * a <= 1.0)).float()
    b1, b2 = 1, (Pphi - 2 if Pphi > 3 else Pphi - 1)
    sust = ((b >= b1) & (b <= b2)).float().expand(Kphi, Pphi).clone()
    return {'sync': sync, 'fast': fast, 'slow': slow, 'sust': sust}


class PropagationEncoder(nn.Module):
    """Masked 2D kernel bank over the (K, L) path-lag plane, then a weighted pool.

    Each kernel is masked to a propagation template, so it may only look at its
    prescribed distance-lag shape. Corridor pooling uses the validity mask times
    a learned distance decay, so nearer upstream nodes can weigh more.
    """

    def __init__(self, H, Kphi=3, Pphi=5, F_per_type=8, in_ch=2):
        super().__init__()
        tmpl = build_templates(Kphi, Pphi)
        om = torch.stack(list(tmpl.values())).repeat_interleave(F_per_type, dim=0)
        self.register_buffer('omega', om.unsqueeze(1))          
        F_tot = F_per_type * len(tmpl)
        self.weight = nn.Parameter(torch.empty(F_tot, in_ch, Kphi, Pphi))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.bias = nn.Parameter(torch.zeros(F_tot))
        self.pad = (Pphi // 2, Pphi - 1 - Pphi // 2, Kphi // 2, Kphi - 1 - Kphi // 2)
        self.mix = nn.Conv2d(F_tot, H, kernel_size=1)
        self.mlp_path = mlp(H, H, H)

    def forward(self, U, pw):
        z = F.conv2d(F.pad(U, self.pad), self.weight * self.omega, self.bias)
        z = F.relu(self.mix(z))                                 
        z = (z * pw).sum(dim=(2, 3)) / pw.expand_as(z).sum(dim=(2, 3)).clamp_min(1e-6)
        return self.mlp_path(z)                                 


class CrisTal(nn.Module):
    def __init__(self, N, T, D, Q, paths, dbar, valid,
                 J=4, K=12, L=12, H=64, W=6, R=32,
                 Kphi=3, Pphi=5, F_per_type=8,
                 tau_node=0.1, tau_time=0.5, eps=1e-3,
                 steps_per_day=288, period_days=7, xp_ch=1, fe=32):
        super().__init__()
        self.N, self.T, self.D, self.Q = N, T, D, Q
        self.P = T * D
        self.J, self.K, self.L, self.H, self.W, self.R = J, K, L, H, W, R
        self.tau_node, self.tau_time, self.eps = tau_node, tau_time, eps
        self.steps_per_day = steps_per_day
        self.fine_period = steps_per_day          
        self.coarse_period = 7                     
        self.period_days = period_days

        # ---- Step 1: time-folded self diagnosis ----
        self.fA = nn.Embedding(N, R)
        self.fB = nn.Embedding(self.fine_period, R)
        self.fC = nn.Embedding(self.coarse_period, R)
        for e in (self.fA, self.fB, self.fC):
            nn.init.normal_(e.weight, std=0.1)
        self.proj_tr = nn.Linear(self.P, H)
        self.proj_res = nn.Linear(self.P, H)
        self.he_node = nn.Embedding(N, fe)
        self.he_tod = nn.Embedding(self.steps_per_day, fe)
        self.he_dow = nn.Embedding(7, fe)
        for e in (self.he_node, self.he_tod, self.he_dow):
            nn.init.xavier_uniform_(e.weight)
        d_in = 2 * H + 3 * fe
        if period_days > 0:
            self.p_enc = mlp(xp_ch * period_days * Q, H, H, layers=2)
            d_in += H
        self.head = mlp(d_in, H, Q, layers=3)

        self.mlp_g = mlp(W, H, 1)
        self.mlp_q = mlp(W, H, H)

        # ---- Steps 2-3: query-adaptive spatial evidence + propagation ----
        self.f_eta = mlp(H, H, J)                  
        self.f_ell = mlp(H + 1, H, 1)              
        self.prop = PropagationEncoder(H, Kphi, Pphi, F_per_type, in_ch=2)
        self.dist_alpha = nn.Parameter(torch.tensor(-5.0))   
        self.out = nn.Linear(H, Q)                 

        self.register_buffer('paths', torch.as_tensor(paths, dtype=torch.long))
        self.register_buffer('dbar', torch.as_tensor(dbar, dtype=torch.float32))
        self.register_buffer('valid', torch.as_tensor(valid, dtype=torch.float32))

    def calendar_trend(self, starts):
        """Global CP trend indexed by wall-clock calendar; independent of the
        window's observed values (causal at every lag). starts: (b,) long."""
        dev = starts.device
        pos = starts.view(-1, 1) + torch.arange(self.P, device=dev).view(1, -1)
        tod = pos % self.fine_period
        dow = (pos // self.fine_period) % self.coarse_period
        BC = self.fB(tod) * self.fC(dow)                       
        return torch.einsum('nr,bpr->bnp', self.fA.weight, BC)  

    def self_head(self, x_tr, x_res, starts, xp):
        b, N, _ = x_tr.shape
        dev = x_tr.device
        f = [self.proj_tr(x_tr), self.proj_res(x_res)]
        pos = starts.view(-1, 1) + self.P                       
        node = torch.arange(N, device=dev)
        tod = (pos % self.steps_per_day).squeeze(1)
        dow = ((pos // self.steps_per_day) % 7).squeeze(1)
        f += [self.he_node(node).unsqueeze(0).expand(b, N, -1),
              self.he_tod(tod).unsqueeze(1).expand(b, N, -1),
              self.he_dow(dow).unsqueeze(1).expand(b, N, -1)]
        if self.period_days > 0 and xp is not None:
            f.append(self.p_enc(xp.reshape(xp.shape[0], xp.shape[1], -1)))
        return self.head(torch.cat(f, dim=-1))                 

    def self_branch(self, x, sigma_w, starts, xp):
        x_tr = self.calendar_trend(starts)                     
        x_res = x - x_tr
        y_self = self.self_head(x_tr, x_res, starts, xp)
        r = (x[..., -self.W:] - x_tr[..., -self.W:]) / (sigma_w + self.eps)
        g = torch.sigmoid(self.mlp_g(r)).squeeze(-1)           
        q = self.mlp_q(r)                                      
        return y_self, g, q, x_tr

    def spatial_correction(self, x, q, b_idx=None, n_idx=None):
        """Spatial correction S for selected (batch, node) pairs.
        x: (b, N, P); q: (M, H) residual queries. Returns S: (M, Q)."""
        b, N, P = x.shape
        L, J, K, H = self.L, self.J, self.K, self.H
        xl = x[..., -L:].flip(-1)                              
        if b_idx is None:
            b_idx = torch.arange(b, device=x.device).repeat_interleave(N)
            n_idx = torch.arange(N, device=x.device).repeat(b)
        pn = self.paths[n_idx]                                 
        dn = self.dbar[n_idx]
        vn = self.valid[n_idx]
        ev = xl[b_idx.view(-1, 1, 1), pn.clamp_min(0)]         

        eta = torch.sigmoid(self.f_eta(q))                     
        m_node = torch.sigmoid((eta.unsqueeze(-1) - dn) / self.tau_node) * vn

        q_exp = q.unsqueeze(1).unsqueeze(2).expand(-1, J, K, H)
        lam = torch.sigmoid(self.f_ell(torch.cat([q_exp, dn.unsqueeze(-1)], -1))).squeeze(-1)
        ell = 1.0 + (L - 1) * lam
        delta = torch.arange(1, L + 1, device=x.device, dtype=x.dtype)
        m_time = torch.sigmoid((ell.unsqueeze(-1) - delta) / self.tau_time)

        m = m_node.unsqueeze(-1) * m_time                      
        U = torch.stack([ev * m, m], dim=2).view(-1, 2, K, L)  
        pw = (vn.reshape(-1, 1, K, 1)
              * torch.exp(-F.softplus(self.dist_alpha) * dn.reshape(-1, 1, K, 1)))
        h = self.prop(U, pw).view(-1, J, H)                    

        path_ok = (vn.sum(-1) > 0)
        logits = torch.einsum('mjh,mh->mj', h, q) / math.sqrt(H)
        logits = logits.masked_fill(~path_ok, float('-inf'))
        beta = torch.nan_to_num(torch.softmax(logits, dim=-1), nan=0.0)
        u = torch.einsum('mj,mjh->mh', beta, h)                
        return self.out(u)                                     

    def forward(self, x, sigma_w, starts, xp=None, theta=None):
        """theta=None: soft gating over all nodes (training).
        theta=float: hard routing -- spatial branch runs only where g >= theta.
        Returns (y, y_self, g, activation_rate, x_tr)."""
        b, N, _ = x.shape
        y_self, g, q, x_tr = self.self_branch(x, sigma_w, starts, xp)
        if theta is None:
            S = self.spatial_correction(x, q.reshape(b * N, -1)).view(b, N, self.Q)
            y = y_self + g.unsqueeze(-1) * S
            act = torch.ones((), device=x.device)
        else:
            y = y_self.clone()
            sel = g >= theta
            act = sel.float().mean()
            if sel.any():
                b_idx, n_idx = sel.nonzero(as_tuple=True)
                S = self.spatial_correction(x, q[b_idx, n_idx], b_idx, n_idx)
                y[b_idx, n_idx] = y_self[b_idx, n_idx] + g[b_idx, n_idx].unsqueeze(-1) * S
        return y, y_self, g, act, x_tr
