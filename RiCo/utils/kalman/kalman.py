"""Diagonal Kalman filter for residual dynamics."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class KalmanResidualFilter(nn.Module):
    """Estimate predictive residual means and variances with a diagonal KF."""

    def __init__(
        self,
        state_dim: int,
        init: str = "identity",
        context_dim: int = 0,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.context_dim = context_dim

        if init == "identity":
            self.F_diag = nn.Parameter(torch.ones(state_dim))
        else:
            self.F_diag = nn.Parameter(torch.randn(state_dim))

        self.H_diag = nn.Parameter(torch.ones(state_dim))
        self.q_diag = nn.Parameter(torch.zeros(state_dim))
        self.r_diag = nn.Parameter(torch.zeros(state_dim))
        self.cov_jitter = 1e-4

        self.context_to_qr: Optional[nn.Linear] = None
        if context_dim > 0:
            self.context_to_qr = nn.Linear(context_dim, 2 * state_dim)
            nn.init.zeros_(self.context_to_qr.weight)
            nn.init.zeros_(self.context_to_qr.bias)

    def compute_Q_R(
        self,
        batch_size: int,
        context_t: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return positive diagonal process and observation variances."""
        process_var = (F.softplus(self.q_diag) + 1e-3).unsqueeze(0)
        observation_var = (F.softplus(self.r_diag) + 1e-3).unsqueeze(0)
        process_var = process_var.expand(batch_size, -1)
        observation_var = observation_var.expand(batch_size, -1)

        if context_t is not None and self.context_to_qr is not None:
            delta = self.context_to_qr(context_t).clamp(min=-4.0, max=4.0)
            process_delta, observation_delta = delta.chunk(2, dim=-1)
            process_var = process_var * torch.exp(process_delta)
            observation_var = observation_var * torch.exp(observation_delta)

        return process_var, observation_var

    def one_step(
        self,
        state: torch.Tensor,
        observation: torch.Tensor,
        covariance: torch.Tensor,
        context_t: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Advance the filter by one observation using the Joseph update."""
        process_var, observation_var = self.compute_Q_R(
            state.shape[0],
            context_t=context_t,
        )

        predicted_state = self.F_diag * state
        predicted_covariance = self.F_diag.square().unsqueeze(0) * covariance
        predicted_covariance = predicted_covariance + process_var

        predicted_observation = self.H_diag * predicted_state
        innovation_covariance = (
            self.H_diag.square().unsqueeze(0) * predicted_covariance
            + observation_var
            + self.cov_jitter
        )
        kalman_gain = (
            predicted_covariance
            * self.H_diag.unsqueeze(0)
            / innovation_covariance.clamp_min(self.cov_jitter)
        )
        innovation = observation - predicted_observation

        updated_state = predicted_state + kalman_gain * innovation
        correction = 1.0 - kalman_gain * self.H_diag.unsqueeze(0)
        updated_covariance = (
            correction.square() * predicted_covariance
            + kalman_gain.square() * observation_var
        ).clamp_min(self.cov_jitter)

        return (
            updated_state,
            updated_covariance,
            innovation,
            innovation_covariance,
            predicted_observation,
        )

    def forward(
        self,
        residual: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Filter a residual window and expose predictive prior statistics.

        Args:
            residual: Residual tensor with shape [batch, time, channels].
            context: Optional conditioning tensor. RiCo's main configuration
                does not use direct context conditioning.

        Returns:
            Predictive means, predictive variances, innovations, and innovation
            variances for all time steps.
        """
        batch_size, steps, channels = residual.shape
        if context is not None:
            if self.context_to_qr is None:
                raise ValueError(
                    "context was provided to a filter with context_dim=0"
                )
            expected_shape = (batch_size, steps, self.context_dim)
            if tuple(context.shape) != expected_shape:
                raise ValueError(
                    f"expected context shape {expected_shape}, "
                    f"got {tuple(context.shape)}"
                )

        base_process_var = (F.softplus(self.q_diag) + 1e-3).view(
            1,
            1,
            channels,
        )
        base_observation_var = (F.softplus(self.r_diag) + 1e-3).view(
            1,
            1,
            channels,
        )

        if context is None:
            process_var = base_process_var.expand(batch_size, steps, channels)
            observation_var = base_observation_var.expand(
                batch_size,
                steps,
                channels,
            )
        else:
            delta = self.context_to_qr(context.reshape(batch_size * steps, -1))
            delta = delta.reshape(batch_size, steps, 2 * channels)
            process_delta, observation_delta = delta.clamp(-4.0, 4.0).chunk(
                2,
                dim=-1,
            )
            process_var = base_process_var * torch.exp(process_delta)
            observation_var = base_observation_var * torch.exp(
                observation_delta
            )

        state = torch.zeros(
            batch_size,
            channels,
            device=residual.device,
            dtype=residual.dtype,
        )
        covariance = torch.ones_like(state)

        innovations = []
        innovation_covariances = []
        predicted_observations = []
        transition_squared = self.F_diag.square().unsqueeze(0)
        observation_squared = self.H_diag.square().unsqueeze(0)
        observation_row = self.H_diag.unsqueeze(0)

        for step in range(steps):
            predicted_state = self.F_diag * state
            predicted_covariance = (
                transition_squared * covariance + process_var[:, step, :]
            )

            predicted_observation = self.H_diag * predicted_state
            innovation_covariance = (
                observation_squared * predicted_covariance
                + observation_var[:, step, :]
                + self.cov_jitter
            )
            kalman_gain = (
                predicted_covariance
                * observation_row
                / innovation_covariance.clamp_min(self.cov_jitter)
            )
            innovation = residual[:, step, :] - predicted_observation

            state = predicted_state + kalman_gain * innovation
            correction = 1.0 - kalman_gain * observation_row
            covariance = (
                correction.square() * predicted_covariance
                + kalman_gain.square() * observation_var[:, step, :]
            ).clamp_min(self.cov_jitter)

            innovations.append(innovation)
            innovation_covariances.append(innovation_covariance)
            predicted_observations.append(predicted_observation)

        innovation = torch.stack(innovations, dim=1)
        innovation_covariance = torch.stack(innovation_covariances, dim=1)
        predicted_observation = torch.stack(predicted_observations, dim=1)

        return {
            "obs_pred": predicted_observation,
            "obs_cov": innovation_covariance,
            "innovation": innovation,
            "innov_cov": innovation_covariance,
        }
