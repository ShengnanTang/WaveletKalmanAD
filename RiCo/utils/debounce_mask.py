"""Per-channel data-quality masks for flat or stalled signal regions.

A value of one marks a valid sample that may participate in anomaly injection
and loss computation. A value of zero marks an excluded sample.
"""

from __future__ import annotations

import numpy as np


def compute_debounce_mask_1d(
    signal: np.ndarray,
    window_size: int = 10,
    std_threshold: float = 0.02,
    debounce_steps: int = 15,
    scale_percentile: float = 75.0,
) -> np.ndarray:
    """Build a validity mask for one signal channel.

    A sample is considered inactive when its centered rolling standard
    deviation is below a scale-relative threshold. Inactive regions are then
    dilated to suppress transition artifacts at their boundaries.

    Args:
        signal: One-dimensional input signal with shape [time].
        window_size: Width of the centered rolling window.
        std_threshold: Fraction of the reference scale used as the inactivity
            threshold.
        debounce_steps: Number of samples by which to expand each inactive
            region on both sides.
        scale_percentile: Percentile of positive rolling deviations used as
            the reference scale.

    Returns:
        A uint8 array with one for valid samples and zero for excluded samples.
    """
    if signal.ndim != 1:
        raise ValueError(f"expected a 1D signal, got shape {signal.shape}")

    length = signal.shape[0]
    if length < window_size:
        return np.ones(length, dtype=np.uint8)

    rolling_std = _rolling_std_centered(signal, window_size)
    rolling_std = np.where(np.isnan(rolling_std), 1.0, rolling_std)

    positive_std = rolling_std[rolling_std > 0]
    if positive_std.size:
        scale = float(np.percentile(positive_std, scale_percentile))
    else:
        scale = 1.0
    scale = max(scale, 1e-8)

    valid_mask = rolling_std > std_threshold * scale
    inactive_mask = ~valid_mask
    if debounce_steps > 0:
        kernel = np.ones(2 * debounce_steps + 1)
        inactive_mask = np.convolve(inactive_mask, kernel, mode="same") > 0

    return (~inactive_mask).astype(np.uint8)


def compute_debounce_mask(
    values: np.ndarray,
    window_size: int = 10,
    std_threshold: float = 0.02,
    debounce_steps: int = 15,
    scale_percentile: float = 75.0,
) -> np.ndarray:
    """Build independent validity masks for all signal channels.

    Args:
        values: Input with shape [time] or [time, channels].
        window_size: Width of the centered rolling window.
        std_threshold: Scale-relative inactivity threshold.
        debounce_steps: Inactive-region dilation radius.
        scale_percentile: Percentile used to estimate typical local variation.

    Returns:
        A mask with the same shape as values and dtype uint8.
    """
    if values.ndim == 1:
        return compute_debounce_mask_1d(
            values,
            window_size,
            std_threshold,
            debounce_steps,
            scale_percentile,
        )
    if values.ndim != 2:
        raise ValueError(f"expected shape [T] or [T, C], got {values.shape}")

    output = np.zeros_like(values, dtype=np.uint8)
    for channel in range(values.shape[1]):
        output[:, channel] = compute_debounce_mask_1d(
            values[:, channel],
            window_size,
            std_threshold,
            debounce_steps,
            scale_percentile,
        )
    return output


def _rolling_std_centered(values: np.ndarray, window_size: int) -> np.ndarray:
    """Compute a centered rolling standard deviation in linear time.

    The calculation uses ddof=1 to match the conventional sample standard
    deviation. Positions without a complete window are returned as NaN.
    """
    length = values.shape[0]
    if window_size <= 1:
        return np.zeros(length)

    cumulative = np.concatenate(([0.0], np.cumsum(values)))
    cumulative_squared = np.concatenate(([0.0], np.cumsum(values * values)))

    output = np.full(length, np.nan, dtype=np.float64)
    left = np.arange(length) - window_size // 2
    right = left + window_size
    valid = (left >= 0) & (right <= length)
    indices = np.flatnonzero(valid)

    sums = cumulative[right[indices]] - cumulative[left[indices]]
    squared_sums = (
        cumulative_squared[right[indices]] - cumulative_squared[left[indices]]
    )
    means = sums / window_size
    variance = np.clip(squared_sums / window_size - means * means, 0.0, None)
    variance *= window_size / (window_size - 1)
    output[indices] = np.sqrt(variance)
    return output
