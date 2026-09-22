"""RiCo: two-stage wavelet/Kalman/conditional-flow anomaly detector.

Stage 1 learns an invariant wavelet representation. Stage 2 freezes the
wavelet and jointly trains masked LF reconstruction, a diagonal Kalman filter,
and a Kalman-conditioned residual flow.

This module follows the paper's main method rather than the ablation variants:
candidate-guided LF reconstruction produces CAS, conditional residual
likelihood produces RAS, and the two scores are fused after channel-wise
median/MAD normalization.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from torch.utils.data import DataLoader

from utils.ADWT_1D import DWT
from utils.anomaly_injector import AnomalyInjector, DenoisingWrapper
from utils.dataset import ReconstructDataset
from utils.debounce_mask import compute_debounce_mask
from utils.kalman.kalman import KalmanResidualFilter
from utils.kalman.low_context import LowContextReconstructor
from utils.kalman.tsflow_residual import KalmanFlow
from utils.kalman.wavelet_kalman_data import NoAugWrapper
from utils.kalman.wavelet_kalman_losses import (
    kalman_student_t_nll,
    masked_freq_loss,
    masked_mse,
)
from utils.kalman.wavelet_model import WaveletModel
from utils.torch_utility import EarlyStoppingTorch, adjust_learning_rate, get_gpu


class RiCo:
    """Paper-aligned RiCo detector."""

    def __init__(
        self,
        win_size: int = 100,
        enc_in: int = 1,
        epochs: int = 10,
        stage1_epochs: Optional[int] = None,
        stage2_epochs: Optional[int] = None,
        batch_size: int = 128,
        lr: float = 1e-3,
        lr_stage1: Optional[float] = None,
        lr_stage2: Optional[float] = None,
        lradj: str = "consistent",
        patience: int = 3,
        validation_size: float = 0.2,
        use_anomaly_aug: bool = True,
        anomaly_p_inject: float = 0.5,
        anomaly_severity: float = 1.0,
        anomaly_weights: Optional[dict] = None,
        anomaly_seed: Optional[int] = 42,
        use_debounce: bool = True,
        debounce_window: int = 10,
        debounce_std_thr: float = 0.0002,
        debounce_steps: int = 5,
        lambda_invariant: float = 1.0,
        lambda_recon: float = 1.0,
        lambda_flow: float = 200.0,
        nll_eps_floor: float = 1e-4,
        kalman_init: str = "identity",
        cuda: bool = True,
        num_workers: int = 8,
        low_context_dim: int = 32,
        lambda_low_context: float = 1.0,
        candidate_topk_ratio: float = 0.5,
        candidate_smooth_window: int = 5,
        candidate_pad: int = 4,
        flow_sample_ratio: float = 0.5,
        wavelet_level: int = 3,
        flow_matching_steps: int = 4,
        wavelet: str = "db2",
    ) -> None:
        if wavelet_level not in (1, 2, 3, 4):
            raise ValueError("wavelet_level must be one of: 1, 2, 3, 4")
        if flow_matching_steps not in (1, 2, 4, 8, 16):
            raise ValueError(
                "flow_matching_steps must be one of: 1, 2, 4, 8, 16"
            )
        if wavelet not in ("db1", "db2", "bior3.1", "sym2", "sym4"):
            raise ValueError(
                "wavelet must be one of: db1, db2, bior3.1, sym2, sym4"
            )
        if not 0.0 < candidate_topk_ratio <= 1.0:
            raise ValueError("candidate_topk_ratio must be in (0, 1]")

        self.win_size = win_size
        self.enc_in = enc_in
        self.wavelet_level = wavelet_level
        self.flow_matching_steps = flow_matching_steps
        self.wavelet = wavelet
        self.stage1_epochs = 10 if stage1_epochs is None else stage1_epochs
        self.stage2_epochs = epochs if stage2_epochs is None else stage2_epochs
        self.batch_size = batch_size
        self.lr_stage1 = 0.005 if lr_stage1 is None else lr_stage1
        self.lr_stage2 = 0.005 if lr_stage2 is None else lr_stage2
        self.lradj = lradj
        self.patience = patience
        self.validation_size = validation_size
        self.num_workers = num_workers
        self.low_context_dim = low_context_dim
        self.lambda_low_context = lambda_low_context
        self.candidate_topk_ratio = candidate_topk_ratio
        self.candidate_smooth_window = candidate_smooth_window
        self.candidate_pad = candidate_pad
        self.flow_sample_ratio = flow_sample_ratio

        self.use_anomaly_aug = use_anomaly_aug
        self.use_debounce = use_debounce
        self.anomaly_p_inject = anomaly_p_inject
        self.anomaly_severity = anomaly_severity
        self.anomaly_weights = anomaly_weights
        self.anomaly_seed = anomaly_seed

        # Synthetic anomaly augmentation provides the corrupted/clean pair
        # required by the Stage-1 invariance objective. Disabling augmentation
        # therefore disables that objective as part of the full ablation, even
        # when this detector is instantiated directly instead of via main.py.
        self.lambda_invariant = lambda_invariant if use_anomaly_aug else 0.0
        self.lambda_recon = lambda_recon
        self.lambda_flow = lambda_flow
        self.nll_eps_floor = nll_eps_floor
        self.kalman_init = kalman_init
        self.device = get_gpu(cuda)
        self._active_channels: Optional[list] = None

        self.dwt = DWT(
            kernel_size=4,
            level=self.wavelet_level,
            wavelet=self.wavelet,
            random_init=False,
            learnable=True,
            device=self.device,
        ).to(self.device)
        self.wavelet_model = WaveletModel(
            self.dwt,
            seq_len=win_size,
            c_in=enc_in,
        ).to(self.device)
        self.low_context = LowContextReconstructor(
            enc_in,
            context_dim=low_context_dim,
            hidden=32,
            depth=3,
            dropout=0.1,
        ).to(self.device)
        self.kalman = self._new_kalman(enc_in)
        self.flow = KalmanFlow(
            dim=enc_in,
            eps_floor=nll_eps_floor,
            n_steps=self.flow_matching_steps,
            hidden=64,
            num_residual_blocks=4,
            cond_feat_dim=0,
        ).to(self.device)

        self.injector: Optional[AnomalyInjector] = None
        if use_anomaly_aug:
            self.injector = AnomalyInjector(
                p_inject=anomaly_p_inject,
                anomaly_weights=anomaly_weights,
                severity=anomaly_severity,
                seed=anomaly_seed,
            )

        self.debounce_kwargs = {
            "window_size": debounce_window,
            "std_threshold": debounce_std_thr,
            "debounce_steps": debounce_steps,
        }
        self._anomaly_score: Optional[np.ndarray] = None
        self._low_score: Optional[np.ndarray] = None
        self._high_score: Optional[np.ndarray] = None
        self._high_residual: Optional[np.ndarray] = None
        self._kalman_prior_mean: Optional[np.ndarray] = None
        self._kalman_prior_var: Optional[np.ndarray] = None
        self.y_hats: Optional[np.ndarray] = None
        self._x_true: Optional[np.ndarray] = None

    def _new_kalman(self, state_dim: int) -> KalmanResidualFilter:
        return KalmanResidualFilter(
            state_dim=state_dim,
            init=self.kalman_init,
            context_dim=0,
        ).to(self.device)

    def parameter_count(self) -> Dict[str, int]:
        """Return effective parameter counts without double-counting modules."""
        modules = {
            "wavelet": self.wavelet_model,
            "low_context": self.low_context,
            "kalman": self.kalman,
            "flow": self.flow,
        }
        counts = {
            name: sum(parameter.numel() for parameter in module.parameters())
            for name, module in modules.items()
        }
        counts["total"] = sum(counts.values())
        counts["active_channels"] = len(self._active_channels or range(self.enc_in))
        return counts

    def fit(self, data: np.ndarray) -> None:
        """Train both stages on the provided time series."""
        data = self._ensure_2d(data)
        n_train = max(
            self.win_size + 1,
            int((1 - self.validation_size) * len(data)),
        )
        n_train = min(n_train, len(data))
        train_data = data[:n_train]
        if len(data) - n_train >= self.win_size:
            valid_data = data[n_train:]
        else:
            valid_data = train_data

        self._configure_active_channels(train_data)
        train_loader, valid_loader = self._build_loaders(train_data, valid_data)

        if self.stage1_epochs > 0:
            self._fit_stage1(train_loader, valid_loader)
        else:
            print("[skip] stage1_epochs == 0")

        if self.stage2_epochs > 0:
            self._fit_stage2(train_loader, valid_loader)
        else:
            print("[skip] stage2_epochs == 0")

    @staticmethod
    def _ensure_2d(data: np.ndarray) -> np.ndarray:
        arr = np.asarray(data, dtype=np.float32)
        if arr.ndim not in (1, 2):
            raise ValueError(f"expected a 1D or 2D time series, got shape {arr.shape}")
        return arr[:, None] if arr.ndim == 1 else arr

    @staticmethod
    def _robust_positive(score: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Equation (26): channel-wise median/MAD scaling and positive part."""
        median = score.median(dim=0, keepdim=True).values
        mad = (score - median).abs().median(dim=0, keepdim=True).values
        return F.relu((score - median) / (mad + eps))

    @torch.no_grad()
    def _score_per_channel(self, data: np.ndarray) -> np.ndarray:
        """Run Algorithm 2 and return the fused channel-wise scores [T, C]."""
        data = self._ensure_2d(data)
        if len(data) < self.win_size:
            raise ValueError(
                f"time series length {len(data)} is shorter than window {self.win_size}"
            )

        base_test = ReconstructDataset(data, window_size=self.win_size)
        test_set = NoAugWrapper(base_test, self.debounce_kwargs, self.use_debounce)
        test_loader = DataLoader(
            test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        self.wavelet_model.eval()
        self.low_context.eval()
        self.kalman.eval()
        self.flow.eval()

        total_len = len(data)
        channels = len(self._active_channels) if self._active_channels else data.shape[1]
        cas_sum = torch.zeros(total_len, channels, device=self.device)
        ras_sum = torch.zeros(total_len, channels, device=self.device)
        score_count = torch.zeros(total_len, 1, device=self.device)
        residual_sum = torch.zeros(total_len, channels, device=self.device)
        prior_mean_sum = torch.zeros(total_len, channels, device=self.device)
        prior_var_sum = torch.zeros(total_len, channels, device=self.device)

        for batch_index, (_, x_clean, _) in enumerate(tqdm.tqdm(test_loader, leave=False)):
            x_clean = x_clean.to(self.device, dtype=torch.float32)
            batch = x_clean.size(0)
            low, residual = self._compute_bands(x_clean)
            prior = self.kalman(residual)
            ras = self.flow.cfm_score(
                residual,
                prior["obs_pred"],
                prior["obs_cov"],
                cond_feat=None,
                reduce_channels=False,
            )
            cas = self._candidate_low_context_score(low, ras, reduce_channels=False)

            starts = (
                torch.arange(batch, device=self.device)
                + batch_index * test_loader.batch_size
            )
            absolute = starts[:, None] + torch.arange(self.win_size, device=self.device)[None, :]
            valid = absolute < total_len
            index = absolute[valid]

            cas_sum.index_add_(0, index, cas[valid])
            ras_sum.index_add_(0, index, ras[valid])
            residual_sum.index_add_(0, index, residual[valid])
            prior_mean_sum.index_add_(0, index, prior["obs_pred"][valid])
            prior_var_sum.index_add_(0, index, prior["obs_cov"][valid])
            score_count.index_add_(
                0,
                index,
                torch.ones(index.numel(), 1, device=self.device),
            )

        count = score_count.clamp_min(1.0)
        raw_cas = cas_sum / count
        raw_ras = ras_sum / count
        cas = self._robust_positive(raw_cas)
        ras = self._robust_positive(raw_ras)
        fused = cas + ras

        self._low_score = cas.mean(dim=-1).cpu().numpy()
        self._high_score = ras.mean(dim=-1).cpu().numpy()
        self._high_residual = (residual_sum / count).cpu().numpy()
        self._kalman_prior_mean = (prior_mean_sum / count).cpu().numpy()
        self._kalman_prior_var = (prior_var_sum / count).cpu().numpy()
        self.y_hats = data[:, self._active_channels[0] if self._active_channels else 0]
        self._x_true = self.y_hats.copy()

        active_scores = fused.cpu().numpy()
        if self._active_channels is not None and len(self._active_channels) != data.shape[1]:
            scores = np.zeros((total_len, data.shape[1]), dtype=active_scores.dtype)
            scores[:, self._active_channels] = active_scores
            return scores
        return active_scores

    def decision_function(self, data: np.ndarray) -> np.ndarray:
        """Return the paper's point-wise score, averaged across variables."""
        channel_scores = self._score_per_channel(data)
        active = self._active_channels or list(range(channel_scores.shape[1]))
        scores = channel_scores[:, active].mean(axis=1)
        self._anomaly_score = scores
        return scores

    def decision_function_perchannel(self, data: np.ndarray) -> np.ndarray:
        """Return the fused CAS+RAS score for every time/channel position."""
        return self._score_per_channel(data)

    def anomaly_score(self) -> Optional[np.ndarray]:
        return self._anomaly_score

    def low_score(self) -> Optional[np.ndarray]:
        return self._low_score

    def high_score(self) -> Optional[np.ndarray]:
        return self._high_score

    def kalman_prior_trace(self) -> Optional[Dict[str, np.ndarray]]:
        """Return cached high residual and Kalman prior statistics from scoring."""
        if self._high_residual is None or self._kalman_prior_mean is None:
            return None
        return {
            "high": self._high_residual,
            "obs_pred": self._kalman_prior_mean,
            "obs_var": self._kalman_prior_var,
            "active_channels": np.asarray(self._active_channels, dtype=int),
        }

    def get_y_hat(self) -> Optional[np.ndarray]:
        """Return the first active channel signal cached during scoring."""
        return self.y_hats

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def _fit_stage1(self, train_loader: DataLoader, valid_loader: DataLoader) -> None:
        print("\n" + "=" * 60)
        print("Stage 1: Lightly fine-tuning learnable wavelet")
        print("=" * 60)
        optimizer = torch.optim.Adam(self.wavelet_model.parameters(), lr=self.lr_stage1)
        early_stopping = EarlyStoppingTorch(None, patience=self.patience)

        for epoch in range(1, self.stage1_epochs + 1):
            self.wavelet_model.train()
            loss_acc = recon_acc = inv_acc = 0.0
            n_batches = 0
            loop = tqdm.tqdm(train_loader, total=len(train_loader), leave=True)
            for x_corrupt, x_clean, vmask in loop:
                x_clean = x_clean.float().to(self.device)
                vmask = vmask.float().to(self.device)
                x_corrupt = x_corrupt.float().to(self.device)

                optimizer.zero_grad()
                changed = (x_corrupt != x_clean).to(dtype=vmask.dtype)
                valid_non_dead = torch.maximum(vmask, changed)
                rec_clean, _ = self.wavelet_model.forward_with_yl(x_clean)
                loss_recon = masked_mse(rec_clean, x_clean, mask=valid_non_dead)

                low_clean = self.wavelet_model.low_freq_reconstruct(x_clean)
                low_corrupt = self.wavelet_model.low_freq_reconstruct(x_corrupt)
                # ``vmask`` excludes injected anomaly positions.  That is
                # appropriate for the other objectives, but invariance must
                # compare the clean/corrupted views at those positions too.
                # Dead/debounce regions remain masked because the wrapper
                # restores them exactly, so x_corrupt == x_clean there.
                loss_inv = masked_freq_loss(
                    low_corrupt,
                    low_clean.detach(),
                    mask=valid_non_dead,
                )

                loss = (
                    self.lambda_recon * loss_recon
                    + self.lambda_invariant * loss_inv
                )
                loss.backward()
                optimizer.step()
                loss_acc += loss.item()
                recon_acc += self.lambda_recon * loss_recon.item()
                inv_acc += self.lambda_invariant * loss_inv.item()
                n_batches += 1
                loop.set_description(
                    f"S1[wavelet] [{epoch}/{self.stage1_epochs}]"
                )
                loop.set_postfix(
                    L=f"{loss_acc / n_batches:.4f}",
                    Lrec=f"{recon_acc / n_batches:.4f}",
                    Linv=f"{inv_acc / n_batches:.6f}",
                )

            valid_recon, valid_inv = self._validate_stage1(valid_loader)
            monitor = (
                self.lambda_recon * valid_recon
                + self.lambda_invariant * valid_inv
            )
            print(
                f"  [S1] epoch {epoch}: valid_recon={valid_recon:.5f} "
                f"valid_inv={valid_inv:.5f} monitor={monitor:.5f}"
            )
            early_stopping(monitor, self.wavelet_model)
            counter = getattr(early_stopping, "counter", 0)
            patience = getattr(early_stopping, "patience", self.patience)
            print(f"  [S1] early_stop counter: {counter}/{patience}")

            if early_stopping.early_stop:
                print("  [S1] Early stopping")
                break
            adjust_learning_rate(
                optimizer,
                epoch + 1,
                self.lradj,
                self.lr_stage1,
            )

        if early_stopping.best_state is not None:
            self.wavelet_model.load_state_dict(early_stopping.best_state)
    @torch.no_grad()
    def _validate_stage1(self, valid_loader: DataLoader) -> Tuple[float, float]:
        self.wavelet_model.eval()
        recon_losses, inv_losses = [], []
        for x_corrupt, x_clean, vmask in valid_loader:
            x_corrupt = x_corrupt.float().to(self.device)
            x_clean = x_clean.float().to(self.device)
            vmask = vmask.float().to(self.device)
            changed = (x_corrupt != x_clean).to(dtype=vmask.dtype)
            valid_non_dead = torch.maximum(vmask, changed)
            rec_clean, _ = self.wavelet_model.forward_with_yl(x_clean)
            recon_losses.append(
                masked_mse(rec_clean, x_clean, valid_non_dead).item()
            )
            low_clean = self.wavelet_model.low_freq_reconstruct(x_clean)
            low_corrupt = self.wavelet_model.low_freq_reconstruct(x_corrupt)
            inv_losses.append(
                masked_freq_loss(
                    low_corrupt,
                    low_clean,
                    mask=valid_non_dead,
                ).item()
            )
        recon_mean = float(np.mean(recon_losses)) if recon_losses else 0.0
        invariant_mean = float(np.mean(inv_losses)) if inv_losses else 0.0
        return recon_mean, invariant_mean

    def _fit_stage2(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader,
    ) -> None:
        print("\n" + "=" * 60)
        print("Stage 2: Training CAB, Kalman RIC, and conditional Flow")
        print("=" * 60)
        self._freeze_wavelet()
        optimizer = torch.optim.Adam(
            list(self.low_context.parameters())
            + list(self.kalman.parameters())
            + list(self.flow.parameters()),
            lr=self.lr_stage2,
        )
        early_stopping = EarlyStoppingTorch(None, patience=self.patience)

        for epoch in range(1, self.stage2_epochs + 1):
            self.low_context.train()
            self.kalman.train()
            self.flow.train()
            n_batches = 0
            loss_history = {"kf": [], "flow": [], "low_ctx": []}
            loop = tqdm.tqdm(train_loader, total=len(train_loader), leave=True)
            log_every = max(1, len(train_loader) // 20)
            running = torch.zeros(4, device=self.device)
            for _, x_clean, vmask in loop:
                x_clean = x_clean.to(
                    self.device,
                    dtype=torch.float32,
                    non_blocking=True,
                )
                vmask = self._align_mask_channels(
                    vmask.to(
                        self.device,
                        dtype=torch.float32,
                        non_blocking=True,
                    )
                )
                low, high = self._compute_bands(x_clean)
                loss_low_ctx, _ = self._low_context_loss(low, vmask)
                out_high = self.kalman(high)
                flow_mean, flow_cov = out_high["obs_pred"], out_high["obs_cov"]

                loss_kf_high, maha_high = kalman_student_t_nll(
                    out_high["innovation"],
                    out_high["innov_cov"],
                    mask=vmask,
                    eps_floor=self.nll_eps_floor,
                    reduction="mean",
                    return_maha=True,
                )
                loss_kf = loss_kf_high
                maha_per_dim = maha_high

                flow_sample_ratio = min(
                    max(float(self.flow_sample_ratio), 0.0),
                    1.0,
                )
                if flow_sample_ratio < 1.0:
                    steps = high.size(1)
                    n_sample = max(1, int(math.ceil(steps * flow_sample_ratio)))
                    time_idx = torch.randperm(steps, device=high.device)[:n_sample]
                    time_idx, _ = torch.sort(time_idx)
                    loss_flow_high = self.flow.cfm_loss(
                        high[:, time_idx, :],
                        flow_mean[:, time_idx, :],
                        flow_cov[:, time_idx, ...],
                        mask=vmask[:, time_idx, :],
                        cond_feat=None,
                    )
                else:
                    loss_flow_high = self.flow.cfm_loss(
                        high,
                        flow_mean,
                        flow_cov,
                        mask=vmask,
                        cond_feat=None,
                    )
                loss_flow = loss_flow_high

                loss = (
                    self.lambda_low_context * loss_low_ctx
                    + loss_kf
                    + self.lambda_flow * loss_flow
                )
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.low_context.parameters())
                    + list(self.kalman.parameters())
                    + list(self.flow.parameters()),
                    max_norm=5.0,
                )
                optimizer.step()

                loss_history["kf"].append(loss_kf.detach())
                loss_history["flow"].append(loss_flow.detach())
                loss_history["low_ctx"].append(loss_low_ctx.detach())

                running += torch.stack(
                    [
                        loss.detach(),
                        loss_flow.detach(),
                        loss_kf.detach(),
                        loss_low_ctx.detach(),
                    ]
                )
                n_batches += 1

                should_log = (
                    n_batches == 1
                    or n_batches % log_every == 0
                    or n_batches == len(train_loader)
                )
                if should_log:
                    averages = (running / n_batches).detach().cpu().tolist()
                    avg_loss, avg_flow, avg_kf, avg_low_ctx = averages
                    loop.set_description(
                        f"S2 LowCtx-HighFlow [{epoch}/{self.stage2_epochs}]"
                    )
                    loop.set_postfix(
                        L=f"{avg_loss:.4f}",
                        Lctx=f"{avg_low_ctx:.4f}",
                        CFM=f"{avg_flow:.4f}",
                        NLL=f"{avg_kf:.4f}",
                        maha=f"{float(maha_per_dim.detach().cpu()):.3f}",
                    )

            self._print_loss_stats(epoch, loss_history)
            valid_loss, diagnostics = self._validate_stage2(valid_loader)
            print(
                f"  [S2] epoch {epoch}: valid_loss={valid_loss:.5f} "
                f"low_ctx={diagnostics['low_ctx']:.5f} "
                f"flow={diagnostics['flow']:.5f} "
                f"kalman_nll={diagnostics['kalman_nll']:.5f} "
                f"maha={diagnostics['maha_mean']:.4f}"
            )
            early_stopping(
                valid_loss,
                nn.ModuleList([self.low_context, self.kalman, self.flow]),
            )

            counter = getattr(early_stopping, "counter", 0)
            patience = getattr(early_stopping, "patience", self.patience)
            if counter > 0 and not early_stopping.early_stop:
                print(f"  [S2] early stopping counter: {counter}/{patience}")

            if early_stopping.early_stop:
                print("  [S2] Early stopping")
                break
            adjust_learning_rate(optimizer, epoch + 1, self.lradj, self.lr_stage2)

        if early_stopping.best_state is not None:
            nn.ModuleList([self.low_context, self.kalman, self.flow]).load_state_dict(
                early_stopping.best_state
            )

    def _print_loss_stats(self, epoch: int, loss_history: Dict[str, list]) -> None:
        """Print raw and weighted Stage-2 loss statistics."""
        print(f"\n  [S2 loss stats] epoch {epoch}")
        print(
            f"  {'name':<10} {'mean':>12} {'std':>12} "
            f"{'min':>12} {'max':>12} {'weighted':>12}"
        )
        print(f"  {'-' * 72}")

        weights = {
            "kf": ("kalman_NLL", 1.0),
            "flow": ("flow_CFM", self.lambda_flow),
            "low_ctx": ("low_CTX", self.lambda_low_context),
        }
        means = {}
        for key, vals in loss_history.items():
            arr = torch.stack(vals).detach()
            mean = arr.mean().item()
            std = arr.std(unbiased=False).item()
            min_val = arr.min().item()
            max_val = arr.max().item()
            means[key] = mean
            name, w = weights[key]
            print(
                f"  {name:<10} {mean:>12.4e} {std:>12.4e} "
                f"{min_val:>12.4e} {max_val:>12.4e} "
                f"{w * mean:>12.4e}"
            )

        contribs = {k: abs(weights[k][1] * means[k]) for k in loss_history}
        total = sum(contribs.values()) + 1e-12
        details = "  ".join(
            f"{weights[key][0]}={contribs[key] / total * 100:5.1f}%"
            for key in loss_history
        )
        print(f"  contribution:  {details}")

    @torch.no_grad()
    def _validate_stage2(
        self,
        valid_loader: DataLoader,
    ) -> Tuple[float, Dict[str, float]]:
        self.low_context.eval()
        self.kalman.eval()
        self.flow.eval()
        totals = []
        ctx_values = []
        ric_values = []
        flow_values = []
        maha_values = []
        for _, x_clean, valid_mask in valid_loader:
            x_clean = x_clean.to(self.device, dtype=torch.float32)
            valid_mask = self._align_mask_channels(
                valid_mask.to(self.device, dtype=torch.float32)
            )
            low, residual = self._compute_bands(x_clean)
            context_loss, _ = self._low_context_loss(low, valid_mask)
            prior = self.kalman(residual)
            ric_loss, maha = kalman_student_t_nll(
                prior["innovation"],
                prior["innov_cov"],
                mask=valid_mask,
                eps_floor=self.nll_eps_floor,
                reduction="mean",
                return_maha=True,
            )
            flow_loss = self.flow.cfm_loss(
                residual,
                prior["obs_pred"],
                prior["obs_cov"],
                mask=valid_mask,
                cond_feat=None,
            )
            total = (
                self.lambda_low_context * context_loss
                + ric_loss
                + self.lambda_flow * flow_loss
            )
            totals.append(total)
            ctx_values.append(context_loss)
            ric_values.append(ric_loss)
            flow_values.append(flow_loss)
            maha_values.append(maha)

        def mean(values):
            return torch.stack(values).mean().item() if values else 0.0

        diagnostics = {
            "low_ctx": mean(ctx_values),
            "kalman_nll": mean(ric_values),
            "flow": mean(flow_values),
            "maha_mean": mean(maha_values),
        }
        return mean(totals), diagnostics

    def _freeze_wavelet(self) -> None:
        for parameter in self.wavelet_model.parameters():
            parameter.requires_grad_(False)
        self.wavelet_model.eval()

    def _compute_bands(
        self,
        x_btc: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            low_freq = self.wavelet_model.low_freq_reconstruct(x_btc)
        high_freq = x_btc - low_freq
        if (
            self._active_channels is not None
            and len(self._active_channels) != x_btc.size(-1)
        ):
            low_freq = low_freq[:, :, self._active_channels]
            high_freq = high_freq[:, :, self._active_channels]
        return low_freq, high_freq

    def _low_context_loss(
        self,
        low: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ctx_mask = self.low_context.make_block_mask(low)
        low_recon, _, clean_context, _ = self.low_context.forward_with_clean_context(
            low, mask=ctx_mask
        )
        while mask.dim() < ctx_mask.dim():
            mask = mask.unsqueeze(-1)
        weight = ctx_mask * mask
        err = (low_recon - low) ** 2
        loss = (err * weight).sum() / (weight.sum() + 1e-8)
        return loss, clean_context

    def _smooth_score(self, score: torch.Tensor) -> torch.Tensor:
        window = int(self.candidate_smooth_window)
        if window <= 1:
            return score
        if window % 2 == 0:
            window += 1
        pad = window // 2
        if score.dim() == 2:
            score_bct = score.unsqueeze(1)
            return F.avg_pool1d(
                score_bct,
                kernel_size=window,
                stride=1,
                padding=pad,
            ).squeeze(1)
        score_bct = score.transpose(1, 2).contiguous()
        return F.avg_pool1d(
            score_bct,
            kernel_size=window,
            stride=1,
            padding=pad,
        ).transpose(1, 2)

    def _candidate_mask_from_score(
        self,
        high_score: torch.Tensor,
        reduce_channels: bool = True,
    ) -> torch.Tensor:
        if reduce_channels and high_score.dim() == 3:
            score = high_score.mean(dim=-1)
        else:
            score = high_score
        score = self._smooth_score(score.detach())
        batch, steps = score.shape[:2]
        ratio = min(max(float(self.candidate_topk_ratio), 0.0), 1.0)
        k = max(1, int(math.ceil(steps * ratio)))
        idx = torch.topk(score, k=k, dim=1).indices
        mask = torch.zeros_like(score)
        mask.scatter_(1, idx, 1.0)
        pad = int(self.candidate_pad)
        if pad > 0:
            kernel = 2 * pad + 1
            if mask.dim() == 2:
                mask = F.max_pool1d(
                    mask.unsqueeze(1),
                    kernel_size=kernel,
                    stride=1,
                    padding=pad,
                ).squeeze(1)
            else:
                mask_bct = mask.transpose(1, 2).contiguous()
                mask = F.max_pool1d(
                    mask_bct,
                    kernel_size=kernel,
                    stride=1,
                    padding=pad,
                ).transpose(1, 2)
        return mask

    def _low_proxy_score(
        self,
        low: torch.Tensor,
        reduce_channels: bool = True,
    ) -> torch.Tensor:
        """Measure LF deviation from a local moving average."""
        score = low.mean(dim=-1) if reduce_channels else low

        window = int(self.candidate_smooth_window)
        if window <= 1:
            window = 5
        if window % 2 == 0:
            window += 1

        pad = window // 2
        if score.dim() == 2:
            smooth = F.avg_pool1d(
                score.unsqueeze(1),
                kernel_size=window,
                stride=1,
                padding=pad,
            ).squeeze(1)
        else:
            score_bct = score.transpose(1, 2).contiguous()
            smooth = F.avg_pool1d(
                score_bct,
                kernel_size=window,
                stride=1,
                padding=pad,
            ).transpose(1, 2)

        return (score - smooth).abs()

    def _low_fft_recon_score(
        self,
        low: torch.Tensor,
        low_recon: torch.Tensor,
        cand_mask: torch.Tensor,
        reduce_channels: bool = True,
    ) -> torch.Tensor:
        batch, steps, channels = low.shape
        n_fft = min(32, steps)
        if n_fft < 2:
            score = (low_recon - low).pow(2)
            if reduce_channels:
                return score.mean(dim=-1) * cand_mask
            return score * cand_mask

        hop_length = max(1, n_fft // 8)
        window = torch.hann_window(n_fft, device=low.device, dtype=low.dtype)

        low_flat = low.transpose(1, 2).reshape(batch * channels, steps)
        recon_flat = low_recon.transpose(1, 2).reshape(batch * channels, steps)
        low_stft = torch.stft(
            low_flat,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window=window,
            center=True,
            normalized=True,
            return_complex=True,
        )
        recon_stft = torch.stft(
            recon_flat,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window=window,
            center=True,
            normalized=True,
            return_complex=True,
        )

        spec_err = (recon_stft - low_stft).abs().pow(2)
        frame_score = spec_err.mean(dim=1).reshape(batch, channels, -1)
        interp_score = F.interpolate(
            frame_score,
            size=steps,
            mode="linear",
            align_corners=False,
        )
        if reduce_channels:
            return interp_score.mean(dim=1) * cand_mask
        point_score = interp_score.transpose(1, 2).contiguous()
        return point_score * cand_mask
    def _candidate_low_context_score(
        self,
        low: torch.Tensor,
        high_score: torch.Tensor,
        reduce_channels: bool = True,
    ) -> torch.Tensor:
        """Compute CAS over the union of LF- and RAS-derived candidates."""
        high_mask = self._candidate_mask_from_score(
            high_score,
            reduce_channels=reduce_channels,
        )
        low_proxy = self._low_proxy_score(low, reduce_channels=reduce_channels)
        low_mask = self._candidate_mask_from_score(
            low_proxy,
            reduce_channels=reduce_channels,
        )

        cand_mask = ((high_mask > 0) | (low_mask > 0)).float()
        if cand_mask.dim() == 2:
            ctx_mask = cand_mask.unsqueeze(-1).expand_as(low)
        else:
            ctx_mask = cand_mask
        low_recon, _, _ = self.low_context(low, mask=ctx_mask)

        return self._low_fft_recon_score(
            low,
            low_recon,
            cand_mask,
            reduce_channels=reduce_channels,
        )

    def _align_mask_channels(self, mask: torch.Tensor) -> torch.Tensor:
        if (
            self._active_channels is not None
            and mask.size(-1) != len(self._active_channels)
        ):
            return mask[:, :, self._active_channels]
        return mask

    def _configure_active_channels(self, train_data: np.ndarray) -> None:
        train_std = train_data.std(axis=0)
        self._active_channels = np.where(train_std > 1e-8)[0].tolist()
        if not self._active_channels:
            self._active_channels = [0]
        n_active = len(self._active_channels)
        n_dropped = train_data.shape[1] - n_active
        print(f"[channel filter] raw channels: {train_data.shape[1]}")
        print(f"[channel filter] active channels: {n_active}")
        print(f"[channel filter] dropped channels: {n_dropped}")
        if n_dropped > 0:
            dropped = sorted(
                set(range(train_data.shape[1])) - set(self._active_channels)
            )
            print(f"[channel filter] dropped channel indices: {dropped}")
        if n_active != self.enc_in:
            print(
                "[channel filter] rebuild modules: "
                f"{self.enc_in} -> {n_active} channels"
            )
            self.low_context = LowContextReconstructor(
                n_active,
                context_dim=self.low_context_dim,
                hidden=32,
                depth=3,
                dropout=0.1,
            ).to(self.device)
            self.kalman = self._new_kalman(n_active)
            self.flow = KalmanFlow(
                dim=n_active,
                eps_floor=self.nll_eps_floor,
                n_steps=self.flow_matching_steps,
                hidden=64,
                num_residual_blocks=4,
                cond_feat_dim=0,
            ).to(self.device)

    def _build_loaders(
        self,
        train_data: np.ndarray,
        valid_data: np.ndarray,
    ) -> Tuple[DataLoader, DataLoader]:
        if self.use_debounce:
            train_mask = compute_debounce_mask(
                train_data,
                **self.debounce_kwargs,
            )
            valid_mask = compute_debounce_mask(
                valid_data,
                **self.debounce_kwargs,
            )
        else:
            train_mask = None
            valid_mask = None

        base_train = ReconstructDataset(
            train_data,
            window_size=self.win_size,
            mask=train_mask,
        )
        base_valid = ReconstructDataset(
            valid_data,
            window_size=self.win_size,
            mask=valid_mask,
        )
        if self.use_anomaly_aug:
            train_set = DenoisingWrapper(
                base_train,
                self.injector,
                debounce_kwargs=self.debounce_kwargs,
                use_debounce=self.use_debounce,
            )
            valid_injector = AnomalyInjector(
                p_inject=self.injector.p_inject,
                anomaly_weights=getattr(self.injector, "anomaly_weights", None),
                severity=getattr(self.injector, "severity", 1.0),
                seed=12345,
            )
            valid_set = DenoisingWrapper(
                base_valid,
                valid_injector,
                debounce_kwargs=self.debounce_kwargs,
                use_debounce=self.use_debounce,
            )
        else:
            train_set = NoAugWrapper(
                base_train,
                self.debounce_kwargs,
                self.use_debounce,
            )
            valid_set = NoAugWrapper(
                base_valid,
                self.debounce_kwargs,
                self.use_debounce,
            )
        loader_kwargs = {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "pin_memory": self.device.type == "cuda",
        }
        if self.num_workers > 0:
            loader_kwargs["persistent_workers"] = True
        return (
            DataLoader(train_set, shuffle=True, **loader_kwargs),
            DataLoader(valid_set, shuffle=False, **loader_kwargs),
        )
