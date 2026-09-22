"""Sliding-window datasets for reconstruction-based detectors."""

from __future__ import annotations

import numpy as np
import torch


class ReconstructDataset(torch.utils.data.Dataset):
    """Materialize fixed-length windows from a time series."""

    def __init__(
        self,
        data,
        window_size: int,
        stride: int = 1,
        normalize: bool = False,
        mask=None,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.stride = stride
        self.data = self._normalize_data(data) if normalize else data
        self.mask = mask
        self.univariate = self.data.shape[1] == 1
        self.sample_num = max(
            0,
            (self.data.shape[0] - window_size) // stride + 1,
        )
        self.samples, self.targets = self._generate_samples()
        self.masks = self._generate_masks() if self.mask is not None else None

    @staticmethod
    def _normalize_data(data, epsilon: float = 1e-8):
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        std = np.where(std == 0, epsilon, std)
        return (data - mean) / std

    def _window_slices(self):
        for index in range(self.sample_num):
            start = index * self.stride
            yield slice(start, start + self.window_size)

    def _generate_samples(self):
        data = torch.as_tensor(self.data, dtype=torch.float32)
        if self.univariate:
            data = data.squeeze(-1)
            samples = torch.stack(
                [data[window] for window in self._window_slices()]
            )
            samples = samples.unsqueeze(-1)
        else:
            samples = torch.stack(
                [data[window, :] for window in self._window_slices()]
            )
        return samples, samples

    def _generate_masks(self):
        mask = torch.as_tensor(self.mask, dtype=torch.float32)
        if self.univariate:
            mask = mask.squeeze(-1)
            masks = torch.stack(
                [mask[window] for window in self._window_slices()]
            )
            masks = masks.unsqueeze(-1)
        else:
            masks = torch.stack(
                [mask[window, :] for window in self._window_slices()]
            )
        return masks

    def __len__(self) -> int:
        return self.sample_num

    def __getitem__(self, index: int):
        if self.masks is not None:
            return index, self.samples[index], self.masks[index]
        return self.samples[index], self.targets[index]
