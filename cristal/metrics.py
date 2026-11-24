import torch


def _mask(y, null_val):
    if null_val is None:
        return torch.ones_like(y)
    m = (y != null_val).float()
    return m / m.mean().clamp_min(1e-8)


def mae(pred, y, null_val=None):
    m = _mask(y, null_val)
    return ((pred - y).abs() * m).mean().item()


def rmse(pred, y, null_val=None):
    m = _mask(y, null_val)
    return (((pred - y) ** 2 * m).mean()).sqrt().item()


def mape(pred, y, null_val=None):
    valid = (y.abs() > 1e-3).float()
    if null_val is not None:
        valid = valid * (y != null_val).float()
    valid = valid / valid.mean().clamp_min(1e-8)
    return (((pred - y).abs() / y.abs().clamp_min(1e-3)) * valid).mean().item() * 100.0


def horizon_report(pred, y, horizons=(3, 6, 12), null_val=None):
    out = {}
    for h in horizons:
        if h <= pred.shape[-1]:
            out[f'MAE@{h}'] = mae(pred[..., h - 1], y[..., h - 1], null_val)
            out[f'RMSE@{h}'] = rmse(pred[..., h - 1], y[..., h - 1], null_val)
            out[f'MAPE@{h}'] = mape(pred[..., h - 1], y[..., h - 1], null_val)
    out['MAE@avg'] = mae(pred, y, null_val)
    out['RMSE@avg'] = rmse(pred, y, null_val)
    out['MAPE@avg'] = mape(pred, y, null_val)
    return out
