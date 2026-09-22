"""Automatic evaluation-window selection used by the original runner."""

from __future__ import annotations

import numpy as np
from scipy.signal import argrelextrema
from statsmodels.tsa.stattools import acf


def find_length_rank(data: np.ndarray, rank: int = 1) -> int:
    """Estimate a dominant period from autocorrelation.

    This preserves the selection logic used by the original Wavelet-Kalman
    experiment runner so that VUS metrics remain directly comparable.
    """
    series = np.asarray(data).squeeze()
    if series.ndim > 1:
        return 100
    if rank == 0:
        return 1

    series = series[: min(20_000, len(series))]
    base = 3
    auto_corr = acf(series, nlags=400, fft=True)[base:]
    local_maxima = argrelextrema(auto_corr, np.greater)[0]

    try:
        sorted_maxima = np.argsort(
            [auto_corr[index] for index in local_maxima]
        )[::-1]
        selected = sorted_maxima[0]
        if rank == 2:
            for index in sorted_maxima[1:]:
                if index > sorted_maxima[0]:
                    selected = index
                    break
        elif rank == 3:
            second = selected
            for index in sorted_maxima[1:]:
                if index > sorted_maxima[0]:
                    second = index
                    break
            for index in sorted_maxima[second:]:
                if index > sorted_maxima[second]:
                    selected = index
                    break

        period = int(local_maxima[selected] + base)
        return 125 if period < 3 or period > 300 else period
    except (IndexError, ValueError):
        return 125
