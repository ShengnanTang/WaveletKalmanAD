"""Learnable one-dimensional discrete wavelet transform."""

from __future__ import annotations

import numpy as np
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F


def reflect(values, minimum, maximum):
    """Reflect coordinates into a closed interval."""
    values = np.asarray(values)
    span = maximum - minimum
    folded = np.mod(values - minimum, 2 * span)
    folded = np.where(folded > span, 2 * span - folded, folded)
    return (folded + minimum).astype(values.dtype)


class DWT(nn.Module):
    """Multi-level DWT with optionally trainable low-pass filters."""

    def __init__(
        self,
        kernel_size: int = 4,
        level: int = 1,
        random_init: bool = False,
        wavelet: str = "db2",
        init_wave: bool = True,
        learnable: bool = True,
        device=None,
    ) -> None:
        super().__init__()
        self.level = level
        self.wavelet = wavelet

        wave = pywt.Wavelet(wavelet)
        self.orthogonal = wave.orthogonal
        initializer = nn.init.trunc_normal_ if random_init else lambda x: x

        if learnable:
            self.low_row = nn.ParameterList()
            for _ in range(level):
                if init_wave:
                    weight = torch.tensor(
                        wave.dec_lo[::-1],
                        dtype=torch.float32,
                    )
                else:
                    weight = torch.zeros(kernel_size, dtype=torch.float32)
                self.low_row.append(nn.Parameter(initializer(weight)))
        else:
            self.low_row = []
            for index in range(level):
                if init_wave:
                    weight = torch.tensor(
                        wave.dec_lo[::-1],
                        dtype=torch.float32,
                    )
                else:
                    weight = torch.zeros(kernel_size, dtype=torch.float32)
                weight = initializer(weight).to(device)
                name = f"low_row_{index}"
                self.register_buffer(name, weight)
                self.low_row.append(getattr(self, name))

        filter_length = len(wave.dec_lo)
        high_dot = torch.tensor(
            [(-1) ** index for index in range(filter_length)],
            dtype=torch.float32,
        )
        self.register_buffer("high_dot", high_dot.to(device))

        # Biorthogonal families require distinct fixed analysis and synthesis
        # filters. Orthogonal families derive them from the learned low pass.
        if not self.orthogonal:
            for index in range(level):
                self.register_buffer(
                    f"analysis_high_row_{index}",
                    torch.tensor(
                        wave.dec_hi[::-1],
                        dtype=torch.float32,
                        device=device,
                    ),
                )
                self.register_buffer(
                    f"synthesis_low_row_{index}",
                    torch.tensor(
                        wave.rec_lo,
                        dtype=torch.float32,
                        device=device,
                    ),
                )
                self.register_buffer(
                    f"synthesis_high_row_{index}",
                    torch.tensor(
                        wave.rec_hi,
                        dtype=torch.float32,
                        device=device,
                    ),
                )

    def _analysis_filters(self, level_index: int):
        low = self.low_row[level_index]
        if self.orthogonal:
            high = self.high_dot * torch.flip(low, dims=(0,))
        else:
            high = getattr(self, f"analysis_high_row_{level_index}")
        return low, high

    def _synthesis_filters(self, level_index: int):
        if self.orthogonal:
            low = self.low_row[level_index]
            high = self.high_dot * torch.flip(low, dims=(0,))
        else:
            low = getattr(self, f"synthesis_low_row_{level_index}")
            high = getattr(self, f"synthesis_high_row_{level_index}")
        return low, high

    def decompose(self, values: torch.Tensor):
        """Return the final approximation and per-level detail bands."""
        detail_bands = []
        approximation = values
        for level_index in range(self.level):
            expanded = approximation[:, :, None, :]
            low_filter, high_filter = self._analysis_filters(level_index)
            low_row = low_filter.reshape(1, 1, 1, -1)
            high_row = high_filter.reshape(1, 1, 1, -1)
            coefficients = afb1d(
                expanded,
                low_row,
                high_row,
                mode="zero",
                dim=3,
            )
            approximation = coefficients[:, ::2, 0].contiguous()
            detail_bands.append(coefficients[:, 1::2, 0].contiguous())
        return approximation, detail_bands

    def reconstruct(self, coefficients) -> torch.Tensor:
        """Reconstruct a signal from approximation and detail bands."""
        approximation, detail_bands = coefficients
        for reverse_index, detail in enumerate(reversed(detail_bands)):
            level_index = len(detail_bands) - 1 - reverse_index
            if detail is None:
                detail = torch.zeros_like(approximation)
            if approximation.shape[-1] > detail.shape[-1]:
                approximation = approximation[..., :-1]

            expanded = approximation[:, :, None, :]
            detail = detail[:, :, None, :]
            low_filter, high_filter = self._synthesis_filters(level_index)
            low_row = low_filter.reshape(1, 1, 1, -1)
            high_row = high_filter.reshape(1, 1, 1, -1)
            reconstructed = sfb1d(
                expanded,
                detail,
                low_row,
                high_row,
                mode="zero",
                dim=3,
            )
            approximation = reconstructed[:, :, 0]
        return approximation

    def forward(self, values, is_dec: bool):
        if is_dec:
            return self.decompose(values)
        return self.reconstruct(values)


def afb1d(x, h0, h1, mode: str = "zero", dim: int = -1):
    """Apply a one-dimensional analysis filter bank."""
    channels = x.shape[1]
    axis = dim % 4
    stride = (2, 1) if axis == 2 else (1, 2)
    length = x.shape[axis]
    filter_length = h0.numel()
    half_length = filter_length // 2

    shape = [1, 1, 1, 1]
    shape[axis] = filter_length
    if h0.shape != tuple(shape):
        h0 = h0.reshape(*shape)
    if h1.shape != tuple(shape):
        h1 = h1.reshape(*shape)
    filters = torch.cat([h0, h1] * channels, dim=0)

    if mode in {"per", "periodization"}:
        if x.shape[dim] % 2 == 1:
            if axis == 2:
                x = torch.cat((x, x[:, :, -1:]), dim=2)
            else:
                x = torch.cat((x, x[:, :, :, -1:]), dim=3)
            length += 1

        x = roll(x, -half_length, dim=axis)
        padding = (
            (filter_length - 1, 0)
            if axis == 2
            else (0, filter_length - 1)
        )
        coefficients = F.conv2d(
            x,
            filters,
            padding=padding,
            stride=stride,
            groups=channels,
        )
        output_length = length // 2
        if axis == 2:
            coefficients[:, :, :half_length] += coefficients[
                :, :, output_length : output_length + half_length
            ]
            coefficients = coefficients[:, :, :output_length]
        else:
            coefficients[:, :, :, :half_length] += coefficients[
                :, :, :, output_length : output_length + half_length
            ]
            coefficients = coefficients[:, :, :, :output_length]
        return coefficients

    output_size = pywt.dwt_coeff_len(length, filter_length, mode=mode)
    total_padding = 2 * (output_size - 1) - length + filter_length
    if mode == "zero":
        if total_padding % 2 == 1:
            padding = (0, 0, 0, 1) if axis == 2 else (0, 1, 0, 0)
            x = F.pad(x, padding)
        padding = (
            (total_padding // 2, 0)
            if axis == 2
            else (0, total_padding // 2)
        )
        return F.conv2d(
            x,
            filters,
            padding=padding,
            stride=stride,
            groups=channels,
        )

    if mode in {"symmetric", "reflect", "periodic"}:
        padding = (
            (0, 0, total_padding // 2, (total_padding + 1) // 2)
            if axis == 2
            else (total_padding // 2, (total_padding + 1) // 2, 0, 0)
        )
        x = mypad(x, pad=padding, mode=mode)
        return F.conv2d(x, filters, stride=stride, groups=channels)

    raise ValueError(f"unknown padding mode: {mode}")


def sfb1d(lo, hi, g0, g1, mode: str = "zero", dim: int = -1):
    """Apply a one-dimensional synthesis filter bank."""
    channels = lo.shape[1]
    axis = dim % 4
    filter_length = g0.numel()

    shape = [1, 1, 1, 1]
    shape[axis] = filter_length
    output_length = 2 * lo.shape[axis]
    if g0.shape != tuple(shape):
        g0 = g0.reshape(*shape)
    if g1.shape != tuple(shape):
        g1 = g1.reshape(*shape)

    stride = (2, 1) if axis == 2 else (1, 2)
    g0 = torch.cat([g0] * channels, dim=0)
    g1 = torch.cat([g1] * channels, dim=0)

    if mode in {"per", "periodization"}:
        output = F.conv_transpose2d(
            lo,
            g0,
            stride=stride,
            groups=channels,
        ) + F.conv_transpose2d(
            hi,
            g1,
            stride=stride,
            groups=channels,
        )
        overlap = filter_length - 2
        if axis == 2:
            output[:, :, :overlap] += output[
                :, :, output_length : output_length + overlap
            ]
            output = output[:, :, :output_length]
        else:
            output[:, :, :, :overlap] += output[
                :, :, :, output_length : output_length + overlap
            ]
            output = output[:, :, :, :output_length]
        return roll(output, 1 - filter_length // 2, dim=dim)

    if mode in {"zero", "symmetric", "reflect", "periodic"}:
        padding = (
            (filter_length - 2, 0)
            if axis == 2
            else (0, filter_length - 2)
        )
        return F.conv_transpose2d(
            lo,
            g0,
            stride=stride,
            padding=padding,
            groups=channels,
        ) + F.conv_transpose2d(
            hi,
            g1,
            stride=stride,
            padding=padding,
            groups=channels,
        )

    raise ValueError(f"unknown padding mode: {mode}")


def roll(x, shift: int, dim: int, make_even: bool = False):
    """Roll a tensor while preserving the legacy optional even-length trim."""
    if shift < 0:
        shift = x.shape[dim] + shift

    end = 1 if make_even and x.shape[dim] % 2 == 1 else 0
    if dim == 0:
        return torch.cat((x[-shift:], x[:-shift + end]), dim=0)
    if dim == 1:
        return torch.cat((x[:, -shift:], x[:, :-shift + end]), dim=1)
    if dim in {2, -2}:
        return torch.cat(
            (x[:, :, -shift:], x[:, :, :-shift + end]),
            dim=2,
        )
    if dim in {3, -1}:
        return torch.cat(
            (x[:, :, :, -shift:], x[:, :, :, :-shift + end]),
            dim=3,
        )
    raise ValueError(f"unsupported dimension: {dim}")


def mypad(x, pad, mode: str = "constant", value: float = 0):
    """Pad the final two tensor dimensions using NumPy-compatible modes."""
    if mode == "symmetric":
        if pad[0] == 0 and pad[1] == 0:
            before, after = pad[2], pad[3]
            length = x.shape[-2]
            indices = reflect(
                np.arange(-before, length + after, dtype="int32"),
                -0.5,
                length - 0.5,
            )
            return x[:, :, indices]

        if pad[2] == 0 and pad[3] == 0:
            before, after = pad[0], pad[1]
            length = x.shape[-1]
            indices = reflect(
                np.arange(-before, length + after, dtype="int32"),
                -0.5,
                length - 0.5,
            )
            return x[:, :, :, indices]

        before, after = pad[0], pad[1]
        row_length = x.shape[-1]
        row_indices = reflect(
            np.arange(-before, row_length + after, dtype="int32"),
            -0.5,
            row_length - 0.5,
        )
        before, after = pad[2], pad[3]
        column_length = x.shape[-2]
        column_indices = reflect(
            np.arange(-before, column_length + after, dtype="int32"),
            -0.5,
            column_length - 0.5,
        )
        rows = np.outer(column_indices, np.ones(row_indices.shape[0]))
        columns = np.outer(np.ones(column_indices.shape[0]), row_indices)
        return x[:, :, rows, columns]

    if mode == "periodic":
        if pad[0] == 0 and pad[1] == 0:
            indices = np.arange(x.shape[-2])
            indices = np.pad(indices, (pad[2], pad[3]), mode="wrap")
            return x[:, :, indices]

        if pad[2] == 0 and pad[3] == 0:
            indices = np.arange(x.shape[-1])
            indices = np.pad(indices, (pad[0], pad[1]), mode="wrap")
            return x[:, :, :, indices]

        column_indices = np.arange(x.shape[-2])
        column_indices = np.pad(
            column_indices,
            (pad[2], pad[3]),
            mode="wrap",
        )
        row_indices = np.arange(x.shape[-1])
        row_indices = np.pad(
            row_indices,
            (pad[0], pad[1]),
            mode="wrap",
        )
        rows = np.outer(column_indices, np.ones(row_indices.shape[0]))
        columns = np.outer(np.ones(column_indices.shape[0]), row_indices)
        return x[:, :, rows, columns]

    if mode in {"constant", "reflect", "replicate"}:
        return F.pad(x, pad, mode, value)
    if mode == "zero":
        return F.pad(x, pad)
    raise ValueError(f"unknown padding mode: {mode}")
