"""Losses and diagnostics for the wavelet-Kalman detector."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _align_mask(mask, target):
    if mask is None:
        return None
    while mask.dim() < target.dim():
        mask = mask.unsqueeze(-1)
    return mask


def masked_mse(pred, target, mask=None):
    err = (pred - target) ** 2
    mask = _align_mask(mask, err)
    if mask is None:
        return err.mean()
    return (err * mask).sum() / mask.sum().clamp_min(1.0)


def masked_freq_loss(pred, target, mask=None):
    diff = pred - target
    loss = diff.abs()
    if diff.size(1) > 2:
        curv = diff[:, 2:] - 2 * diff[:, 1:-1] + diff[:, :-2]
        loss = loss + F.pad(curv.abs(), (0, 0, 1, 1))
    mask = _align_mask(mask, loss)
    if mask is None:
        return loss.mean()
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def kalman_student_t_nll(
    innovation,
    innov_cov,
    mask=None,
    eps_floor: float = 1e-4,
    reduction: str = "mean",
    return_maha: bool = False,
    df: float = 4.0,
):
    dim = innovation.size(-1)
    if innov_cov.shape == innovation.shape:
        var = innov_cov.clamp_min(eps_floor)
        maha = (innovation.square() / var).sum(-1)
        logdet = torch.log(var).sum(-1)
    else:
        identity = torch.eye(
            dim,
            device=innovation.device,
            dtype=innovation.dtype,
        )
        cov = innov_cov + identity * eps_floor
        chol = torch.linalg.cholesky(cov)
        solve = torch.cholesky_solve(innovation.unsqueeze(-1), chol).squeeze(-1)
        maha = (innovation * solve).sum(-1)
        logdet = 2 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(-1)
    nll = 0.5 * logdet + 0.5 * (df + dim) * torch.log1p(maha / df)

    if mask is not None:
        m = mask.mean(-1) if mask.dim() == innovation.dim() else mask
        nll = nll * m
        denom = m.sum().clamp_min(1.0)
    else:
        denom = torch.tensor(nll.numel(), device=nll.device, dtype=nll.dtype)

    value = nll.sum() / denom if reduction == "mean" else nll
    maha_per_dim = (maha / max(dim, 1)).mean()
    return (value, maha_per_dim) if return_maha else value


def kalman_score(
    innovation,
    innov_cov,
    score_type="mahalanobis",
    eps_floor=1e-4,
    state=None,
    covariance=None,
):
    if score_type == "innovation_abs":
        return innovation.abs().mean(-1)
    dim = innovation.size(-1)
    if innov_cov.shape == innovation.shape:
        var = innov_cov.clamp_min(eps_floor)
        maha = (innovation.square() / var).sum(-1)
        logdet = torch.log(var).sum(-1)
    else:
        identity = torch.eye(
            dim,
            device=innovation.device,
            dtype=innovation.dtype,
        )
        cov = innov_cov + identity * eps_floor
        solve = torch.linalg.solve(cov, innovation.unsqueeze(-1)).squeeze(-1)
        maha = (innovation * solve).sum(-1)
        logdet = torch.logdet(cov)
    if score_type == "loglik":
        return 0.5 * (maha + logdet + dim * math.log(2 * math.pi))
    return maha


@torch.no_grad()
def diagnose_kalman_output(innovation, innov_cov, mask=None, eps_floor: float = 1e-4):
    score = kalman_score(innovation, innov_cov, eps_floor=eps_floor)
    if innov_cov.shape == innovation.shape:
        diag = innov_cov
    else:
        diag = innov_cov.diagonal(dim1=-2, dim2=-1)
    dim = innovation.size(-1)

    if mask is not None:
        m = mask.mean(-1)
        denom = m.sum().clamp_min(1.0)
        score_mean = (score * m).sum() / denom
        innov_abs = (innovation.abs().mean(-1) * m).sum() / denom
    else:
        score_mean = score.mean()
        innov_abs = innovation.abs().mean()

    if innov_cov.shape == innovation.shape:
        logdet_full = torch.log(diag.clamp_min(eps_floor)).sum(-1)
    else:
        I = torch.eye(dim, device=innovation.device, dtype=innovation.dtype) * eps_floor
        cov_floored = innov_cov + I
        logdet_full = torch.logdet(cov_floored)

    return {
        "log_det_mean": logdet_full.mean(),
        "maha_mean": score_mean,
        "S_diag_mean": diag.mean(),
        "S_diag_min": diag.min(),
        "innov_abs_mean": innov_abs,
    }
