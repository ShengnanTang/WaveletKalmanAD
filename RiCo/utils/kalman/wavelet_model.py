"""Wavelet reconstruction model used by the two-stage detector."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..ADWT_1D import DWT


class WaveletModel(nn.Module):
    """DWT-based reconstruction module.

    The external tensor layout is ``[batch, time, channels]``. Internally, the
    DWT module consumes ``[batch, channels, time]``.
    """

    def __init__(
        self,
        dwt: DWT,
        seq_len: int,
        c_in: int,
        instance_norm: bool = False,
    ) -> None:
        super().__init__()
        self.dwt = dwt
        self.seq_len = seq_len
        self.c_in = c_in
        self.instance_norm = instance_norm

    @staticmethod
    def _instance_norm(
        x_btc: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = x_btc.mean(dim=1, keepdim=True).detach()
        variance = torch.var(
            x_btc - mean,
            dim=1,
            keepdim=True,
            unbiased=False,
        )
        std = torch.sqrt(variance + 1e-5)
        return (x_btc - mean) / std, mean, std

    @staticmethod
    def _denorm(
        x_norm: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> torch.Tensor:
        return x_norm * std + mean

    @staticmethod
    def _match_time_length(x_btc: torch.Tensor, target_len: int) -> torch.Tensor:
        if x_btc.size(1) == target_len:
            return x_btc
        if x_btc.size(1) > target_len:
            return x_btc[:, :target_len, :]
        return F.pad(x_btc, (0, 0, 0, target_len - x_btc.size(1)))

    def decompose(self, x_btc: torch.Tensor):
        x_bct = x_btc.transpose(1, 2).contiguous()
        return self.dwt(x_bct, is_dec=1)

    def reconstruct(self, yl: torch.Tensor, yh) -> torch.Tensor:
        x_bct = self.dwt((yl, yh), is_dec=0)
        return x_bct.transpose(1, 2).contiguous()

    def forward(self, x_btc: torch.Tensor) -> torch.Tensor:
        if self.instance_norm:
            x_norm, mean, std = self._instance_norm(x_btc)
        else:
            x_norm, mean, std = x_btc, None, None

        yl, yh = self.decompose(x_norm)
        rec_norm = self._match_time_length(
            self.reconstruct(yl, yh),
            x_btc.size(1),
        )

        if self.instance_norm:
            return self._denorm(rec_norm, mean, std)
        return rec_norm

    def forward_with_yl(self, x_btc: torch.Tensor):
        if self.instance_norm:
            x_norm, mean, std = self._instance_norm(x_btc)
        else:
            x_norm, mean, std = x_btc, None, None

        yl, yh = self.decompose(x_norm)
        rec_norm = self._match_time_length(
            self.reconstruct(yl, yh),
            x_btc.size(1),
        )
        rec = self._denorm(rec_norm, mean, std) if self.instance_norm else rec_norm
        return rec, yl
    def low_freq_reconstruct(self, x_btc: torch.Tensor) -> torch.Tensor:
        """Reconstruct the LF component with all detail bands set to zero."""
        if self.instance_norm:
            x_norm, mean, std = self._instance_norm(x_btc)
        else:
            x_norm, mean, std = x_btc, None, None

        yl, yh = self.decompose(x_norm)
        if isinstance(yh, (list, tuple)):
            yh_zero = [torch.zeros_like(h) for h in yh]
        else:
            yh_zero = torch.zeros_like(yh)

        rec_norm = self._match_time_length(
            self.reconstruct(yl, yh_zero),
            x_btc.size(1),
        )
        if self.instance_norm:
            return self._denorm(rec_norm, mean, std)
        return rec_norm
