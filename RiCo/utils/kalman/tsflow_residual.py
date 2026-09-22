"""Point-wise Kalman-conditional flow matching with density scoring.

This module intentionally follows the RNN+CNF pattern discussed in the
conversation, but replaces the RNN condition with the Kalman prior at each
time/channel:

    cond_{t,c} = [Kalman obs_pred_{t,c}, Kalman obs_cov_diag_{t,c}, band_id]

The flow is trained point-wise from data to Gaussian noise:

    x0 = residual point
    x1 = standard Gaussian
    x_s = (1 - s) * x0 + s * x1
    target_v = x1 - x0

At inference, `cfm_score` computes a point-wise negative log-likelihood by
integrating the learned CNF from the observed data point to the Gaussian base
and applying the continuous change-of-variables formula. This returns a score
with the same shape as the old velocity score: (B, T) when channels are reduced
or (B, T, C) when `reduce_channels=False`.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class KalmanFlow(nn.Module):
    def __init__(
        self,
        dim: int,
        eps_floor: float = 1e-4,
        hidden: int = 64,
        n_steps: int = 4,
        sigmin: float = 1e-3,
        sigmax: float = 1.0,
        use_path_noise: bool = False,
        num_residual_blocks: int = 4,
        step_emb: int = 128,
        l_max: Optional[int] = None,
        mode: str = "nplr",
        measure: str = "legs",
        dropout: float = 0.0,
        cond_feat_dim: int = 0,
    ):
        super().__init__()
        self.dim = dim
        self.eps_floor = eps_floor
        self.n_steps = n_steps
        self.sigmin = sigmin
        self.sigmax = sigmax
        self.use_path_noise = use_path_noise
        self.cond_feat_dim = cond_feat_dim

        cond_dim = 2 + cond_feat_dim  # Kalman mean, Kalman var, optional band/context features.
        in_dim = 1 + 1 + cond_dim     # point value, flow time, condition.
        layers = []
        for idx in range(max(1, num_residual_blocks)):
            layers.append(nn.Linear(in_dim if idx == 0 else hidden, hidden))
            layers.append(nn.SiLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

    def _diag_var(self, cov: Optional[torch.Tensor], ref: torch.Tensor) -> torch.Tensor:
        if cov is None:
            return torch.ones_like(ref).clamp_min(self.eps_floor)
        if cov.shape == ref.shape:
            return cov.to(device=ref.device, dtype=ref.dtype).clamp_min(self.eps_floor)
        if cov.ndim >= 2 and cov.shape[-1] == cov.shape[-2]:
            var = cov.diagonal(dim1=-2, dim2=-1)
        else:
            var = cov
        return var.to(device=ref.device, dtype=ref.dtype).clamp_min(self.eps_floor)

    def _expand_time(self, t: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(t):
            t = torch.tensor(t, device=ref.device, dtype=ref.dtype)
        t = t.to(device=ref.device, dtype=ref.dtype)
        if t.ndim == 0:
            return t.view(1, 1, 1).expand_as(ref)
        if t.ndim == 1:
            return t.view(-1, 1, 1).expand_as(ref)
        if t.ndim == 2:
            return t.unsqueeze(-1).expand_as(ref)
        if t.shape[-1] == 1:
            return t.expand_as(ref)
        return t

    def _point_condition(
        self,
        ref: torch.Tensor,
        mean: torch.Tensor,
        cov: Optional[torch.Tensor] = None,
        cond_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        mean = mean.to(device=ref.device, dtype=ref.dtype)
        var = self._diag_var(cov, ref)
        parts = [mean.unsqueeze(-1), var.unsqueeze(-1)]

        if self.cond_feat_dim > 0:
            if cond_feat is None:
                raise ValueError(
                    f"cond_feat_dim={self.cond_feat_dim} but no cond_feat was passed."
                )
            cond_feat = cond_feat.to(device=ref.device, dtype=ref.dtype)
            if cond_feat.shape[-1] != self.cond_feat_dim:
                raise ValueError(
                    f"Expected cond_feat last dim {self.cond_feat_dim}, got {cond_feat.shape[-1]}."
                )
            cond_feat = cond_feat.unsqueeze(-2).expand(*ref.shape, self.cond_feat_dim)
            parts.append(cond_feat)

        return torch.cat(parts, dim=-1)

    def _flat_inputs(
        self,
        x: torch.Tensor,
        s: torch.Tensor,
        mean: torch.Tensor,
        cov: Optional[torch.Tensor] = None,
        cond_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        s = self._expand_time(s, x)
        cond = self._point_condition(x, mean, cov, cond_feat=cond_feat)
        return torch.cat([x.unsqueeze(-1), s.unsqueeze(-1), cond], dim=-1)

    def velocity(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        mean: torch.Tensor,
        cov: Optional[torch.Tensor] = None,
        cond_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        inp = self._flat_inputs(z_t, t, mean, cov, cond_feat=cond_feat)
        return self.net(inp).squeeze(-1)

    def forward_path(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor):
        t = self._expand_time(t, x0)
        if self.use_path_noise:
            eps = torch.randn_like(x0)
            sig_t = (1.0 - t) * self.sigmax + t * self.sigmin
            x_t = (1.0 - t) * x0 + t * x1 + sig_t * eps
            target_v = x1 - x0 + (self.sigmin - self.sigmax) * eps
        else:
            x_t = (1.0 - t) * x0 + t * x1
            target_v = x1 - x0
        return x_t, target_v

    def cfm_loss(
        self,
        residual: torch.Tensor,
        state: torch.Tensor,
        innov_cov: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        x0: Optional[torch.Tensor] = None,
        cond_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        data = residual
        noise = (
            torch.randn_like(data)
            if x0 is None
            else x0.to(device=data.device, dtype=data.dtype)
        )
        s = torch.rand(data.shape[0], 1, 1, device=data.device, dtype=data.dtype)

        x_s, target_v = self.forward_path(data, noise, s)
        pred_v = self.velocity(x_s, s, state, innov_cov, cond_feat=cond_feat)
        err = (pred_v - target_v) ** 2

        if mask is not None:
            while mask.dim() < err.dim():
                mask = mask.unsqueeze(-1)
            err = err * mask
            return err.sum() / (mask.sum() * data.shape[-1] + 1e-8)
        return err.mean()

    def _standard_normal_logp(self, z: torch.Tensor) -> torch.Tensor:
        return -0.5 * (z ** 2 + math.log(2.0 * math.pi))

    def _divergence(
        self,
        y: torch.Tensor,
        s: torch.Tensor,
        mean: torch.Tensor,
        cov: Optional[torch.Tensor] = None,
        cond_feat: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        y_req = y.detach().requires_grad_(True)
        v = self.velocity(y_req, s, mean, cov, cond_feat=cond_feat)
        div = torch.autograd.grad(
            v.sum(),
            y_req,
            create_graph=False,
            retain_graph=False,
            only_inputs=True,
        )[0]
        return v.detach(), div.detach()

    @torch.no_grad()
    def reconstruct(
        self,
        residual: Optional[torch.Tensor] = None,
        state: Optional[torch.Tensor] = None,
        innov_cov: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
        n_steps: Optional[int] = None,
        deterministic: bool = True,
        cond_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if state is None:
            raise ValueError("Kalman prior mean `state` must be provided.")
        z = (
            torch.randn_like(state)
            if noise is None
            else noise.to(device=state.device, dtype=state.dtype)
        )
        steps = self.n_steps if n_steps is None else n_steps
        if steps <= 0:
            return z

        # The trained direction is data -> noise, so sampling reconstructs in reverse.
        dt = 1.0 / steps
        y = z
        for i in range(steps):
            s = torch.full_like(y, 1.0 - i * dt)
            y = y - dt * self.velocity(y, s, state, innov_cov, cond_feat=cond_feat)
        return y

    def density_score(
        self,
        residual: torch.Tensor,
        state: torch.Tensor,
        innov_cov: Optional[torch.Tensor] = None,
        cond_feat: Optional[torch.Tensor] = None,
        n_steps: Optional[int] = None,
        reduce_channels: bool = True,
    ) -> torch.Tensor:
        """Return point-wise NLL by integrating observed data to Gaussian base."""
        steps = self.n_steps if n_steps is None else n_steps
        if steps <= 0:
            nll = -self._standard_normal_logp(residual)
            return nll.mean(dim=-1) if reduce_channels else nll

        with torch.enable_grad():
            y = residual.detach()
            log_det = torch.zeros_like(y)
            dt = 1.0 / steps
            for i in range(steps):
                s0 = i * dt
                s = torch.full_like(y, s0)
                v, div = self._divergence(y, s, state, innov_cov, cond_feat=cond_feat)
                y = y + dt * v
                log_det = log_det + dt * div

        logp_base = self._standard_normal_logp(y)
        logp_data = logp_base + log_det
        nll = -logp_data
        return nll.mean(dim=-1) if reduce_channels else nll

    def cfm_score(
        self,
        residual: torch.Tensor,
        state: torch.Tensor,
        innov_cov: Optional[torch.Tensor] = None,
        cond_feat: Optional[torch.Tensor] = None,
        n_samples: int = 3,
        reduce_channels: bool = True,
    ) -> torch.Tensor:
        return self.density_score(
            residual,
            state,
            innov_cov,
            cond_feat=cond_feat,
            n_steps=self.n_steps,
            reduce_channels=reduce_channels,
        )

    @torch.no_grad()
    def sample(
        self,
        state: torch.Tensor,
        innov_cov: Optional[torch.Tensor] = None,
        shape: Optional[tuple[int, ...]] = None,
        n_steps: Optional[int] = None,
        cond_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if shape is not None and tuple(shape) != tuple(state.shape):
            state = state.expand(shape)
        return self.reconstruct(None, state, innov_cov, n_steps=n_steps, cond_feat=cond_feat)

    def forward(
        self,
        residual: Optional[torch.Tensor] = None,
        state: Optional[torch.Tensor] = None,
        innov_cov: Optional[torch.Tensor] = None,
        cond_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.reconstruct(
            residual=residual, state=state, innov_cov=innov_cov, cond_feat=cond_feat
        )
