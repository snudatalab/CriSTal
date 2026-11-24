"""Train and evaluate CriSTal.
"""
from __future__ import annotations
import argparse
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import (STEPS_PER_DAY, Scaler, STWindowDataset, compute_sigma,
                   load_dataset, load_edge_cost, make_splits)
from .graph import build_paths, distances_from_cost
from .initialize import init_cp_lookup
from .metrics import horizon_report
from .model import CrisTal


@torch.no_grad()
def evaluate(model, loader, scaler, device, theta=None, null_val=None,
             horizons=(3, 6, 12)):
    model.eval()
    preds, ys, act_sum, n = [], [], 0.0, 0
    for batch in loader:
        x, y, sw, starts = batch[0], batch[1], batch[2], batch[3]
        xp = batch[4].to(device) if len(batch) > 4 else None
        st = torch.as_tensor([int(s) for s in starts], device=device)
        x, sw = x.to(device), sw.to(device)
        yhat, _, _, act, _ = model(x, sw, st, xp=xp, theta=theta)
        preds.append(torch.as_tensor(scaler.inverse(yhat.cpu().numpy())))
        ys.append(y)
        act_sum += float(act) * x.shape[0]
        n += x.shape[0]
    rep = horizon_report(torch.cat(preds), torch.cat(ys), horizons, null_val)
    rep['spatial_activation_rate'] = act_sum / max(n, 1)
    return rep


# Per-dataset defaults. Short-term traffic uses a within-day fold (fold_D=1,
# 288 steps/day); long-term air-quality / streamflow fold by day and carry their own sampling rate. 
DATASET_CFG = {
    'pems04':   dict(P=12,  Q=12, fold_D=1, steps_per_day=288, W=6,  dist='kernel'),
    'pems07':   dict(P=12,  Q=12, fold_D=1, steps_per_day=288, W=6,  dist='kernel'),
    'pems08':   dict(P=12,  Q=12, fold_D=1, steps_per_day=288, W=6,  dist='kernel'),
    'metr-la':  dict(P=12,  Q=12, fold_D=1, steps_per_day=288, W=6,  dist='kernel'),
    'pems-bay': dict(P=12,  Q=12, fold_D=1, steps_per_day=288, W=6,  dist='kernel'),
    'knowair':  dict(P=48,  Q=24, fold_D=6, steps_per_day=8,   W=8,  dist='raw'),
    'lamah':    dict(P=168, Q=24, fold_D=7, steps_per_day=24,  W=24, dist='raw'),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=list(DATASET_CFG))
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--out_dir', default='runs')
    # forecasting setup (None -> per-dataset default in DATASET_CFG)
    ap.add_argument('--P', type=int, default=None)
    ap.add_argument('--Q', type=int, default=None)
    ap.add_argument('--fold_D', type=int, default=None)          
    ap.add_argument('--steps_per_day', type=int, default=None)   
    ap.add_argument('--dist_source', default=None, choices=['kernel', 'raw'])
    ap.add_argument('--period_days', type=int, default=7)
    # model
    ap.add_argument('--R', type=int, default=32)
    ap.add_argument('--H', type=int, default=64)
    ap.add_argument('--W', type=int, default=None)
    ap.add_argument('--J', type=int, default=4)             
    ap.add_argument('--K', type=int, default=12)            
    ap.add_argument('--L', type=int, default=12)            
    ap.add_argument('--Kphi', type=int, default=3)          
    ap.add_argument('--Pphi', type=int, default=5)          
    ap.add_argument('--F_per_type', type=int, default=8)    
    ap.add_argument('--tau_node', type=float, default=0.1)
    ap.add_argument('--tau_time', type=float, default=0.5)
    ap.add_argument('--lam_cp', type=float, default=0.0)
    # optimization
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--wd', type=float, default=1e-5)
    ap.add_argument('--bs', type=int, default=16)
    ap.add_argument('--epochs', type=int, default=150)
    ap.add_argument('--patience', type=int, default=30)
    ap.add_argument('--clip', type=float, default=3.0)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--cp_init', default='xavier', choices=['xavier', 'climatology'],
                    help="CP lookup init: Xavier (random), or climatology ALS fit")
    ap.add_argument('--eval_only', action='store_true')
    ap.add_argument('--ckpt', default=None, help='checkpoint to load for --eval_only')
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # fill unset forecasting/fold options from the per-dataset defaults
    cfg = DATASET_CFG[args.dataset]
    P = args.P if args.P is not None else cfg['P']
    Q = args.Q if args.Q is not None else cfg['Q']
    fold_D = args.fold_D if args.fold_D is not None else cfg['fold_D']
    T = args.steps_per_day if args.steps_per_day is not None else cfg['steps_per_day']
    W = args.W if args.W is not None else cfg['W']
    dist_source = args.dist_source if args.dist_source is not None else cfg['dist']
    if P % fold_D != 0:
        raise SystemExit(f'--fold_D {fold_D} must divide P={P}')
    fold_T = P // fold_D
    # METR-LA / PEMS-BAY mask sensor-dropout (0) entries in both loss and metrics.
    null_val = 0.0 if args.dataset in ('metr-la', 'pems-bay') else None

    # ---- data ----
    data, adj = load_dataset(args.dataset, args.data_dir)
    N = data.shape[1]
    n_train, tr_s, va_s, te_s = make_splits(args.dataset, data, P, Q)
    scaler = Scaler(data[:n_train].mean(), data[:n_train].std())
    data_norm = scaler.transform(data).astype(np.float32)
    sigma = compute_sigma(data_norm[:n_train], T)

    def make_loader(starts, shuffle):
        ds = STWindowDataset(data_norm, data, sigma, starts, P, Q, W, T,
                             period_days=args.period_days, null_val=null_val)
        return DataLoader(ds, batch_size=args.bs, shuffle=shuffle,
                          num_workers=args.num_workers, pin_memory=True)

    tr_loader = make_loader(tr_s, True)
    va_loader = make_loader(va_s, False)
    te_loader = make_loader(te_s, False)

    # ---- graph paths ----
    print('building incoming paths...')
    D_dir = None
    if dist_source == 'raw':
        C = load_edge_cost(args.dataset, args.data_dir, adj)
        if C is None:
            raise SystemExit(f'--dist_source raw needs a coordinate file for {args.dataset}')
        D_dir = distances_from_cost(C)
    paths, dbar, valid = build_paths(adj, args.J, args.K, D=D_dir)

    # ---- model ----
    xp_ch = 2 if null_val is not None else 1
    model = CrisTal(N, fold_T, fold_D, Q, paths, dbar, valid,
                    J=args.J, K=args.K, L=args.L, H=args.H, W=W, R=args.R,
                    Kphi=args.Kphi, Pphi=args.Pphi, F_per_type=args.F_per_type,
                    tau_node=args.tau_node, tau_time=args.tau_time,
                    steps_per_day=T, period_days=args.period_days,
                    xp_ch=xp_ch).to(device)
    init_cp_lookup(model, data_norm, n_train, T, args.R, method=args.cp_init, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'dataset={args.dataset} N={N} P={P} Q={Q} params={n_params:,} device={device}')

    os.makedirs(args.out_dir, exist_ok=True)
    best_path = os.path.join(args.out_dir, f'cristal_{args.dataset}.pt')

    def masked_l1(a, y_n, y_raw):
        if null_val is None:
            return (a - y_n).abs().mean()
        m = (y_raw != null_val).to(device, dtype=y_n.dtype)
        return ((a - y_n).abs() * m).sum() / m.sum().clamp(min=1.0)

    # ---- training ----
    best_val, bad = float('inf'), 0
    if not args.eval_only:
        for ep in range(1, args.epochs + 1):
            model.train()
            t0, loss_sum, nb = time.time(), 0.0, 0
            for batch in tr_loader:
                x, y, sw, starts = batch[0], batch[1], batch[2], batch[3]
                xp = batch[4].to(device) if len(batch) > 4 else None
                st = torch.as_tensor([int(s) for s in starts], device=device)
                x, sw = x.to(device), sw.to(device)
                y_n = torch.from_numpy(scaler.transform(y.numpy())).to(device)
                yhat, _, _, _, x_tr = model(x, sw, st, xp=xp, theta=None)
                loss = masked_l1(yhat, y_n, y)
                if args.lam_cp > 0:
                    loss = loss + args.lam_cp * (x - x_tr).pow(2).mean()
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                opt.step()
                loss_sum += float(loss.detach())
                nb += 1
            v = evaluate(model, va_loader, scaler, device, theta=None, null_val=null_val)['MAE@avg']
            print(f'ep {ep:03d} loss {loss_sum / max(nb, 1):.4f} valMAE {v:.4f} '
                  f'time {time.time() - t0:.1f}s')
            if v < best_val - 1e-4:
                best_val, bad = v, 0
                torch.save(model.state_dict(), best_path)
            else:
                bad += 1
                if bad >= args.patience:
                    print('early stop')
                    break
        model.load_state_dict(torch.load(best_path, map_location=device))
    elif args.ckpt:
        model.load_state_dict(torch.load(args.ckpt, map_location=device))

    best_theta, best_theta_mae = None, float('inf')
    for th in [None] + [round(t, 2) for t in np.arange(0.1, 0.95, 0.1)]:
        rep = evaluate(model, va_loader, scaler, device, theta=th, null_val=null_val)
        if th is not None and rep['MAE@avg'] < best_theta_mae:
            best_theta_mae, best_theta = rep['MAE@avg'], th

    # ---- test ----
    test_rep = evaluate(model, te_loader, scaler, device, theta=best_theta, null_val=null_val)
    result = {'dataset': args.dataset, 'params': n_params,
              'theta_selected': best_theta, 'test': test_rep}
    with open(os.path.join(args.out_dir, f'result_{args.dataset}.json'), 'w') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(test_rep, indent=2))
    print(f'saved: {best_path}')


if __name__ == '__main__':
    main()
