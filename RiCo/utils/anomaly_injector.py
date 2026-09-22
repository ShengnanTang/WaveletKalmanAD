"""Synthetic perturbations for RiCo invariance training.

The injector supports point, contextual, and seasonal perturbations. Returned
masks use one for valid samples and zero for injected or excluded samples.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .debounce_mask import compute_debounce_mask

# Fixed permutation table for one-dimensional Perlin noise.
_P = [
    0x97, 0xA0, 0x89, 0x5B, 0x5A, 0x0F, 0x83, 0x0D, 0xC9, 0x5F, 0x60, 0x35, 0xC2, 0xE9, 0x07, 0xE1,
    0x8C, 0x24, 0x67, 0x1E, 0x45, 0x8E, 0x08, 0x63, 0x25, 0xF0, 0x15, 0x0A, 0x17, 0xBE, 0x06, 0x94,
    0xF7, 0x78, 0xEA, 0x4B, 0x00, 0x1A, 0xC5, 0x3E, 0x5E, 0xFC, 0xDB, 0xCB, 0x75, 0x23, 0x0B, 0x20,
    0x39, 0xB1, 0x21, 0x58, 0xED, 0x95, 0x38, 0x57, 0xAE, 0x14, 0x7D, 0x88, 0xAB, 0xA8, 0x44, 0xAF,
    0x4A, 0xA5, 0x47, 0x86, 0x8B, 0x30, 0x1B, 0xA6, 0x4D, 0x92, 0x9E, 0xE7, 0x53, 0x6F, 0xE5, 0x7A,
    0x3C, 0xD3, 0x85, 0xE6, 0xDC, 0x69, 0x5C, 0x29, 0x37, 0x2E, 0xF5, 0x28, 0xF4, 0x66, 0x8F, 0x36,
    0x41, 0x19, 0x3F, 0xA1, 0x01, 0xD8, 0x50, 0x49, 0xD1, 0x4C, 0x84, 0xBB, 0xD0, 0x59, 0x12, 0xA9,
    0xC8, 0xC4, 0x87, 0x82, 0x74, 0xBC, 0x9F, 0x56, 0xA4, 0x64, 0x6D, 0xC6, 0xAD, 0xBA, 0x03, 0x40,
    0x34, 0xD9, 0xE2, 0xFA, 0x7C, 0x7B, 0x05, 0xCA, 0x26, 0x93, 0x76, 0x7E, 0xFF, 0x52, 0x55, 0xD4,
    0xCF, 0xCE, 0x3B, 0xE3, 0x2F, 0x10, 0x3A, 0x11, 0xB6, 0xBD, 0x1C, 0x2A, 0xDF, 0xB7, 0xAA, 0xD5,
    0x77, 0xF8, 0x98, 0x02, 0x2C, 0x9A, 0xA3, 0x46, 0xDD, 0x99, 0x65, 0x9B, 0xA7, 0x2B, 0xAC, 0x09,
    0x81, 0x16, 0x27, 0xFD, 0x13, 0x62, 0x6C, 0x6E, 0x4F, 0x71, 0xE0, 0xE8, 0xB2, 0xB9, 0x70, 0x68,
    0xDA, 0xF6, 0x61, 0xE4, 0xFB, 0x22, 0xF2, 0xC1, 0xEE, 0xD2, 0x90, 0x0C, 0xBF, 0xB3, 0xA2, 0xF1,
    0x51, 0x33, 0x91, 0xEB, 0xF9, 0x0E, 0xEF, 0x6B, 0x31, 0xC0, 0xD6, 0x1F, 0xB5, 0xC7, 0x6A, 0x9D,
    0xB8, 0x54, 0xCC, 0xB0, 0x73, 0x79, 0x32, 0x2D, 0x7F, 0x04, 0x96, 0xFE, 0x8A, 0xEC, 0xCD, 0x5D,
    0xDE, 0x72, 0x43, 0x1D, 0x18, 0x48, 0xF3, 0x8D, 0x80, 0xC3, 0x4E, 0x42, 0xD7, 0x3D, 0x9C, 0xB4,
]
_P = _P + _P


def _fade(t):
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _lerp(a, b, t):
    return a + t * (b - a)


def _dot_grad_1d(h, xf):
    return xf if (h & 0x1) else -xf


def perlin_1d(x: float) -> float:
    """Return one Perlin-noise sample, approximately in [-0.7, 0.7]."""
    xi0 = int(np.floor(x))
    xf0 = x - xi0
    xf1 = xf0 - 1.0
    xi = xi0 & 0xFF
    u = _fade(xf0)
    h0 = _P[xi]
    h1 = _P[xi + 1]
    return _lerp(_dot_grad_1d(h0, xf0), _dot_grad_1d(h1, xf1), u)


def perlin_1d_series(
    length: int,
    scale: float = 0.15,
    offset: float = 0.0,
) -> np.ndarray:
    """Generate a one-dimensional Perlin-noise sequence."""
    xs = np.arange(length) * scale + offset
    return np.array([perlin_1d(x) for x in xs])


class AnomalyInjector:
    """Inject synthetic anomalies into a single time-series window."""

    DEFAULT_WEIGHTS = {
        "contextual": 3.0,
        "seasonal": 3.0,
        "point": 3.0,
    }

    def __init__(
        self,
        p_inject: float = 0.5,
        anomaly_weights: Optional[dict] = None,
        severity: float = 1.0,
        seed: Optional[int] = 42,
    ) -> None:
        self.p_inject = p_inject
        self.severity = severity
        self.weights = (
            anomaly_weights
            if anomaly_weights is not None
            else dict(self.DEFAULT_WEIGHTS)
        )
        self.rng = np.random.default_rng(seed)
        self.types = list(self.weights.keys())
        w = np.array([self.weights[t] for t in self.types], dtype=np.float64)
        self.probs = w / w.sum()

    def __call__(
        self,
        x: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ):
        """Return a perturbed window and its validity mask.

        Args:
            x: Clean window with shape [time, channels].
            valid_mask: Optional mask that restricts perturbation placement.

        Returns:
            A pair ``(corrupted, mask)`` with the same shape as ``x``.
        """
        is_tensor = torch.is_tensor(x)
        if is_tensor:
            device, dtype = x.device, x.dtype
            window = x.detach().cpu().numpy().astype(np.float64)
        else:
            window = np.asarray(x, dtype=np.float64)

        if window.ndim != 2:
            raise ValueError(f"x must be (T, C), got shape {window.shape}")
        steps, channels = window.shape
        valid_np = self._prepare_valid_mask(valid_mask, steps, channels)

        if self.rng.random() >= self.p_inject or valid_np.sum() == 0:
            corrupt_np = window.copy()
            mask_np = np.ones((steps, channels), dtype=np.float32)
        else:
            corrupt_np = window.copy()
            mask_np = np.ones((steps, channels), dtype=np.float32)

            for channel in range(channels):
                if self.rng.random() >= self.p_inject:
                    continue
                kind = self.rng.choice(self.types, p=self.probs)
                method = getattr(self, f"_inject_{kind}")
                channel_slice = slice(channel, channel + 1)
                corrupted, injected = method(
                    window[:, channel_slice],
                    valid_np[:, channel_slice],
                )
                corrupt_np[:, channel_slice] = corrupted
                mask_np[injected, channel] = 0.0

        if is_tensor:
            corrupt = torch.from_numpy(corrupt_np.astype(np.float32)).to(
                device=device,
                dtype=dtype,
            )
            mask = torch.from_numpy(mask_np).to(device=device, dtype=dtype)
        else:
            corrupt = corrupt_np.astype(np.float32)
            mask = mask_np
        return corrupt, mask

    def _pick_channels(
        self,
        C: int,
        ratio_range=(0.2, 1.0),
    ) -> np.ndarray:
        ratio = self.rng.uniform(*ratio_range)
        n = max(1, int(np.round(C * ratio)))
        return self.rng.choice(C, size=n, replace=False)

    def _channel_std(self, window: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        s = window.std(axis=0)
        s = np.where(s < eps, 1.0, s)
        return s

    def _prepare_valid_mask(self, valid_mask, T: int, C: int) -> np.ndarray:
        if valid_mask is None:
            return np.ones((T, C), dtype=bool)
        if torch.is_tensor(valid_mask):
            valid_np = valid_mask.detach().cpu().numpy()
        else:
            valid_np = np.asarray(valid_mask)
        if valid_np.ndim == 1:
            valid_np = valid_np[:, None]
        if valid_np.shape != (T, C):
            raise ValueError(
                f"valid_mask must have shape {(T, C)} or {(T,)}, "
                f"got {valid_np.shape}"
            )
        return valid_np.astype(bool)

    def _valid_time_mask(
        self,
        valid_mask: Optional[np.ndarray],
        T: int,
    ) -> np.ndarray:
        if valid_mask is None:
            return np.ones(T, dtype=bool)
        if valid_mask.ndim == 2:
            return valid_mask.any(axis=1).astype(bool)
        return valid_mask.astype(bool)

    def _valid_runs(self, valid_time: np.ndarray):
        runs = []
        start = None
        for i, ok in enumerate(valid_time):
            if ok and start is None:
                start = i
            elif not ok and start is not None:
                runs.append((start, i))
                start = None
        if start is not None:
            runs.append((start, len(valid_time)))
        return runs

    def _choose_valid_segment(
        self,
        valid_mask: Optional[np.ndarray],
        T: int,
        min_len: int,
        max_len: int,
    ):
        valid_time = self._valid_time_mask(valid_mask, T)
        runs = self._valid_runs(valid_time)
        candidates = [(s, e) for s, e in runs if e - s >= min_len]
        if not candidates:
            return None
        run_s, run_e = candidates[int(self.rng.integers(0, len(candidates)))]
        max_len = min(max_len, run_e - run_s)
        min_len = min(min_len, max_len)
        L = int(self.rng.integers(min_len, max_len + 1))
        s = int(self.rng.integers(run_s, run_e - L + 1))
        return s, s + L

    def _choose_valid_positions(
        self,
        valid_mask: Optional[np.ndarray],
        T: int,
        n_points: int,
    ) -> np.ndarray:
        valid_time = self._valid_time_mask(valid_mask, T)
        positions = np.flatnonzero(valid_time)
        if len(positions) == 0:
            return np.array([], dtype=np.int64)
        n_points = min(n_points, len(positions))
        return self.rng.choice(positions, size=n_points, replace=False)

    def _inject_point(
        self,
        window: np.ndarray,
        valid_mask: Optional[np.ndarray] = None,
    ):
        """Inject isolated amplitude changes."""
        T, C = window.shape
        out = window.copy()
        mask = np.zeros(T, dtype=bool)
        std = self._channel_std(window)

        n_points = max(
            1,
            int(
                self.rng.integers(
                    max(1, T // 20),
                    max(2, T // 7) + 1,
                )
            ),
        )
        positions = self._choose_valid_positions(valid_mask, T, n_points)
        if len(positions) == 0:
            return out, mask
        channels = self._pick_channels(C, ratio_range=(0.1, 0.5))

        for pos in positions:
            for ch in channels:
                k = self.rng.uniform(3.0, 6.0) * self.severity
                sign = self.rng.choice([-1.0, 1.0])
                out[pos, ch] += sign * k * std[ch]
            mask[pos] = True
        return out, mask

    def _inject_contextual(
        self,
        window: np.ndarray,
        valid_mask: Optional[np.ndarray] = None,
    ):
        """Inject a smooth local deviation generated with Perlin noise."""
        T, C = window.shape
        out = window.copy()
        mask = np.zeros(T, dtype=bool)
        std = self._channel_std(window)

        min_len = max(4, int(T * 0.8))
        max_len = max(5, int(T * 0.9))
        segment = self._choose_valid_segment(valid_mask, T, min_len, max_len)
        if segment is None:
            segment = self._choose_valid_segment(valid_mask, T, 4, T)
        if segment is None:
            return out, mask
        s, e = segment
        L = e - s

        channels = self._pick_channels(C, ratio_range=(0.5, 0.8))
        scale = self.rng.uniform(4.0, 8.0) / L * 10.0
        amp_factor = self.rng.uniform(1.5, 2.0) * self.severity
        fade_window = np.hanning(L)

        for ch in channels:
            offset = self.rng.uniform(0, 1000)
            perlin = perlin_1d_series(L, scale=scale, offset=offset)
            perlin = perlin / 0.7
            out[s:e, ch] += amp_factor * std[ch] * perlin * fade_window

        mask[s:e] = True
        return out, mask

    def _inject_seasonal(
        self,
        window: np.ndarray,
        valid_mask: Optional[np.ndarray] = None,
    ):
        """Inject a local temporal resampling perturbation."""
        T, C = window.shape
        out = window.copy()
        mask = np.zeros(T, dtype=bool)

        if T < 12:
            return out, mask

        segment = self._choose_valid_segment(
            valid_mask,
            T,
            max(8, T // 4),
            max(9, T // 2),
        )
        if segment is None:
            segment = self._choose_valid_segment(valid_mask, T, 8, T)
        if segment is None:
            return out, mask
        s, e = segment
        L = e - s
        segment = window[s:e, :]

        if self.rng.random() < 0.5:
            ratio = self.rng.uniform(0.4, 0.7)
        else:
            ratio = self.rng.uniform(1.5, 2.5)
        new_L = max(2, int(round(L * ratio)))

        x_orig = np.linspace(0, 1, L)
        x_warp = np.linspace(0, 1, new_L)
        warped = np.zeros((new_L, C), dtype=segment.dtype)
        for ch in range(C):
            warped[:, ch] = np.interp(x_warp, x_orig, segment[:, ch])
        x_back = np.linspace(0, 1, L)
        x_src = np.linspace(0, 1, new_L)
        result = np.zeros_like(segment)
        for ch in range(C):
            result[:, ch] = np.interp(x_back, x_src, warped[:, ch])

        out[s:e, :] = result
        mask[s:e] = True
        return out, mask

class DenoisingWrapper(Dataset):
    """Combine a reconstruction dataset with synthetic perturbations.

    Each item is returned as ``(corrupted, clean, validity_mask)``. The base
    dataset may return a single tensor, a two-item tuple, or an indexed
    three-item tuple that already contains a validity mask.
    """

    def __init__(
        self,
        base_dataset,
        injector,
        debounce_kwargs=None,
        use_debounce: bool = True,
    ) -> None:
        self.base_dataset = base_dataset
        self.injector = injector
        self.debounce_kwargs = debounce_kwargs or {}
        self.use_debounce = use_debounce

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx):
        item = self.base_dataset[idx]

        base_mask = None
        if isinstance(item, (tuple, list)):
            if len(item) == 3:
                _item_idx, x_clean, base_mask = item
            elif len(item) == 2:
                x_clean = item[0]
            else:
                x_clean = item[0]
        else:
            x_clean = item

        if base_mask is not None:
            if torch.is_tensor(base_mask):
                debounce_mask = base_mask.float()
            else:
                debounce_mask = torch.as_tensor(
                    base_mask,
                    dtype=torch.float32,
                )
        elif self.use_debounce:
            if torch.is_tensor(x_clean):
                x_np = x_clean.detach().cpu().numpy()
            else:
                x_np = np.asarray(x_clean, dtype=np.float32)
            debounce_np = compute_debounce_mask(
                x_np,
                **self.debounce_kwargs,
            ).astype(np.float32)
            debounce_mask = torch.from_numpy(debounce_np)
        else:
            if torch.is_tensor(x_clean):
                debounce_mask = torch.ones_like(x_clean)
            else:
                debounce_mask = torch.ones(
                    x_clean.shape,
                    dtype=torch.float32,
                )

        x_corrupt, inject_mask = self.injector(x_clean, valid_mask=debounce_mask)

        # Restore excluded regions so synthetic anomalies appear only where
        # the data-quality mask marks samples as valid.
        x_corrupt = x_corrupt * debounce_mask + x_clean * (1.0 - debounce_mask)

        vmask = debounce_mask * inject_mask
        return x_corrupt, x_clean, vmask
