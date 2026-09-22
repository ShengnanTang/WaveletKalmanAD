

"""Dataset wrappers for wavelet-Kalman anomaly detection."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from ..debounce_mask import compute_debounce_mask


class NoAugWrapper(Dataset):
    """Wrap a reconstruction dataset without synthetic anomaly injection."""

    def __init__(self, base_dataset: Dataset, debounce_kwargs: dict, use_debounce: bool) -> None:
        self.base = base_dataset
        self.debounce_kwargs = debounce_kwargs
        self.use_debounce = use_debounce

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        item = self.base[idx]
        base_mask = None
        if isinstance(item, (tuple, list)) and len(item) == 3:
            _, window, base_mask = item
        else:
            window = item[0] if isinstance(item, tuple) else item

        if torch.is_tensor(window):
            x_np = window.detach().cpu().numpy().astype(np.float32)
        else:
            x_np = np.asarray(window, dtype=np.float32)

        if base_mask is not None:
            if torch.is_tensor(base_mask):
                mask_tensor = base_mask.float()
            else:
                mask_tensor = torch.as_tensor(
                    base_mask,
                    dtype=torch.float32,
                )
            x_tensor = torch.from_numpy(x_np)
            return x_tensor, x_tensor, mask_tensor
        if self.use_debounce:
            vmask_np = compute_debounce_mask(x_np, **self.debounce_kwargs)
        else:
            vmask_np = np.ones_like(x_np, dtype=np.uint8)

        x_tensor = torch.from_numpy(x_np)
        mask_tensor = torch.from_numpy(vmask_np.astype(np.float32))
        return x_tensor, x_tensor, mask_tensor
