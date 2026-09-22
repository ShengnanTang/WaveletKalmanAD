"""Low-frequency context reconstruction model."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class LowContextReconstructor(nn.Module):
    """Reconstruct masked low-frequency blocks and expose per-time context."""

    def __init__(
        self,
        channels: int,
        context_dim: int = 16,
        hidden: int = 32,
        depth: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.context_dim = context_dim
        self.mask_token = nn.Parameter(torch.zeros(1, 1, channels))

        layers = []
        in_ch = channels + channels
        for idx in range(depth):
            dilation = 2 ** idx
            layers.extend(
                [
                    nn.Conv1d(
                        in_ch if idx == 0 else hidden,
                        hidden,
                        kernel_size=3,
                        padding=dilation,
                        dilation=dilation,
                    ),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
        self.encoder = nn.Sequential(*layers)
        self.to_context = nn.Conv1d(hidden, context_dim, kernel_size=1)
        self.decoder = nn.Conv1d(context_dim, channels, kernel_size=1)

    def _encode(self, low: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        token = self.mask_token.to(device=low.device, dtype=low.dtype)
        low_masked = low * (1.0 - mask) + token * mask
        inp = torch.cat([low_masked, mask], dim=-1).transpose(1, 2).contiguous()
        hidden = self.encoder(inp)
        return self.to_context(hidden).transpose(1, 2).contiguous()

    def make_block_mask(
        self,
        x: torch.Tensor,
        mask_ratio: float = 0.1,
        min_block: int = 4,
        max_block: int = 8,
    ) -> torch.Tensor:
        batch, steps, channels = x.shape
        target = max(1, int(steps * mask_ratio))
        max_block = max(min_block, min(max_block, steps))
        n_blocks = max(1, int((target + min_block - 1) // min_block))

        block = torch.randint(
            min_block,
            max_block + 1,
            (batch, n_blocks),
            device=x.device,
        )
        start_hi = (steps - block + 1).clamp_min(1)
        start = (
            torch.rand(batch, n_blocks, device=x.device) * start_hi
        ).floor().long()
        pos = torch.arange(steps, device=x.device).view(1, 1, steps)
        time_mask = (
            (pos >= start.unsqueeze(-1))
            & (pos < (start + block).unsqueeze(-1))
        ).any(dim=1)
        return (
            time_mask.to(dtype=x.dtype)
            .unsqueeze(-1)
            .expand(batch, steps, channels)
        )

    def forward(
        self,
        low: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if mask is None:
            mask = torch.zeros_like(low)
        context = self._encode(low, mask)
        recon = self.decoder(context.transpose(1, 2))
        recon = recon.transpose(1, 2).contiguous()
        return recon, context, mask

    def forward_with_clean_context(
        self,
        low: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        zero_mask = torch.zeros_like(low)
        low_pair = torch.cat([low, low], dim=0)
        mask_pair = torch.cat([mask, zero_mask], dim=0)
        context_pair = self._encode(low_pair, mask_pair)
        masked_context, clean_context = context_pair.chunk(2, dim=0)
        recon = self.decoder(masked_context.transpose(1, 2))
        recon = recon.transpose(1, 2).contiguous()
        return recon, masked_context, clean_context, mask

    def context(self, low: torch.Tensor) -> torch.Tensor:
        return self._encode(low, mask=torch.zeros_like(low))
