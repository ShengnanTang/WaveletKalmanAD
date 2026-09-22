"""Command-line runner and paper-aligned evaluation for RiCo."""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_curve

from detectors.rico import RiCo
from utils.metrics import get_metrics
from utils.sliding_window import find_length_rank


TRAIN_PATTERN = re.compile(r"(?:^|_)tr_(\d+)(?:_|\.)")
PAPER_METRICS = (
    "AUC-ROC",
    "AUC-PR",
    "VUS-PR",
    "VUS-ROC",
    "BestF1",
    "RangeF1",
)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_series(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    """Load features, labels, and the training-prefix length."""
    frame = pd.read_csv(path)
    if frame.empty or frame.shape[1] < 2:
        raise ValueError(
            f"{path.name}: expected feature columns followed by a label column"
        )

    values = frame.iloc[:, :-1].to_numpy(dtype=np.float32)
    labels = frame.iloc[:, -1].to_numpy(dtype=np.float32)
    match = TRAIN_PATTERN.search(path.name)
    if not match:
        raise ValueError(f"{path.name}: filename must contain '_tr_<train_length>_'")

    train_length = int(match.group(1))
    if not 100 < train_length < len(values):
        raise ValueError(
            f"{path.name}: train length {train_length} must be >100 and <{len(values)}"
        )
    return values, labels, train_length


def normalize_from_training(values: np.ndarray, train_length: int) -> np.ndarray:
    """Apply the original train-prefix standardization to the full series."""
    train = values[:train_length]
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std = np.where(std == 0.0, 1e-3, std)
    return ((values - mean) / std).astype(np.float32)


def category_from_file_name(file_name: str) -> str:
    """Extract the TSB-AD category using the original naming convention."""
    parts = file_name.split("_")
    return parts[1] if len(parts) > 1 else Path(file_name).stem


def best_f1_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    """Return the oracle threshold used by the original output code."""
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    if thresholds.size == 0:
        return float(np.mean(scores) + 3.0 * np.std(scores))
    f1_scores = 2.0 * precision * recall / (precision + recall + 1e-10)
    return float(thresholds[np.argmax(f1_scores[:-1])])


def build_score_frame(
    labels: np.ndarray,
    scores: np.ndarray,
    detector: RiCo,
    train_length: int,
    threshold: float,
) -> pd.DataFrame:
    """Build the complete point-wise score table."""
    low_scores = np.asarray(detector.low_score())
    high_scores = np.asarray(detector.high_score())
    # Equation (26) fuses the robustly normalized branches with equal weight.
    weighted_low = low_scores
    weighted_high = high_scores

    frame = pd.DataFrame(
        {
            "Index": np.arange(len(labels)),
            "Split": np.where(
                np.arange(len(labels)) < train_length,
                "train",
                "test",
            ),
            "True Labels": labels,
            "Anomaly scores": scores,
            "Decision threshold": threshold,
            "Predicted Labels": (scores > threshold).astype(np.int8),
            "Low-frequency score": low_scores,
            "High-frequency NLL score": high_scores,
            "Weighted low contribution": weighted_low,
            "Weighted high contribution": weighted_high,
            "Fusion direction u": (weighted_low + weighted_high) / np.sqrt(2.0),
            "Contribution difference v": (
                weighted_high - weighted_low
            ) / np.sqrt(2.0),
            "Rotated decision threshold": threshold / np.sqrt(2.0),
        }
    )

    trace = detector.kalman_prior_trace()
    if trace is not None:
        frame["High residual ch0"] = trace["high"][:, 0]
        frame["Kalman obs_pred ch0"] = trace["obs_pred"][:, 0]
        frame["Kalman obs_var ch0"] = trace["obs_var"][:, 0]
    return frame


def run_file(
    path: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Train, score, evaluate, and persist one TSB-AD series."""
    values, labels, train_length = load_series(path)
    values = normalize_from_training(values, train_length)
    sliding_window = find_length_rank(values[:, 0].reshape(-1, 1), rank=1)

    detector = RiCo(
        win_size=args.window_size,
        enc_in=values.shape[1],
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cuda=not args.cpu,
        wavelet=args.wavelet,
        wavelet_level=args.wavelet_level,
        flow_matching_steps=args.ode_steps,
        anomaly_seed=args.seed,
    )

    if torch.cuda.is_available() and not args.cpu:
        torch.cuda.synchronize()
    started_at = time.perf_counter()
    detector.fit(values[:train_length])
    scores = detector.decision_function(values)
    if torch.cuda.is_available() and not args.cpu:
        torch.cuda.synchronize()
    runtime_seconds = time.perf_counter() - started_at

    # Preserve the original evaluation entry point and its exact arguments.
    metrics = get_metrics(
        scores,
        labels,
        slidingWindow=sliding_window,
        pred=None,
        version="opt",
        thre=250,
    )
    threshold = best_f1_threshold(labels, scores)
    counts = detector.parameter_count()
    category = category_from_file_name(path.name)

    score_dir = output_dir / "Filewise_scores"
    metric_dir = output_dir / "Filewise_metrics"
    score_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)

    score_path = score_dir / f"{path.stem}_output.csv"
    build_score_frame(labels, scores, detector, train_length, threshold).to_csv(
        score_path,
        index=False,
    )

    result = {
        "file": path.name,
        "Category": category,
        "AUC-ROC": float(metrics["AUC-ROC"]),
        "AUC-PR": float(metrics["AUC-PR"]),
        "VUS-PR": float(metrics["VUS-PR"]),
        "VUS-ROC": float(metrics["VUS-ROC"]),
        "BestF1": float(metrics["Standard-F1"]),
        "RangeF1": float(metrics["R-based-F1"]),
        "PA-F1": float(metrics["PA-F1"]),
        "Event-based-F1": float(metrics["Event-based-F1"]),
        "Affiliation-F": float(metrics["Affiliation-F"]),
        "Decision-Threshold": threshold,
        "Threshold-Selection": "best_pr_f1_on_evaluation_labels",
        "Sliding-Window": sliding_window,
        "Train-Length": train_length,
        "Series-Length": len(values),
        "Channels": int(values.shape[1]),
        "Runtime-Seconds": runtime_seconds,
        "Parameters-Total": counts["total"],
        "Parameters-Wavelet": counts["wavelet"],
        "Parameters-LowContext": counts["low_context"],
        "Parameters-Kalman": counts["kalman"],
        "Parameters-Flow": counts["flow"],
        "Active-Channels": counts["active_channels"],
        "Score-File": str(score_path),
    }
    pd.DataFrame([result]).to_csv(
        metric_dir / f"{path.stem}_metrics.csv",
        index=False,
    )
    return result


def save_aggregate_results(
    results: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    """Write global, category, runtime, and machine-readable summaries."""
    result_frame = pd.DataFrame(results)
    average_row: dict[str, Any] = {
        "file": "AVERAGE",
        "Category": "OVERALL",
    }
    for metric in PAPER_METRICS:
        average_row[metric] = float(result_frame[metric].mean())

    summary_frame = pd.concat(
        [result_frame, pd.DataFrame([average_row])],
        ignore_index=True,
    )
    summary_frame.to_csv(output_dir / "summary_metrics.csv", index=False)
    pd.DataFrame([average_row]).to_csv(
        output_dir / "average_metrics.csv",
        index=False,
    )

    category_frame = (
        result_frame.groupby("Category", as_index=False)
        .agg(
            N=("file", "count"),
            **{metric: (metric, "mean") for metric in PAPER_METRICS},
        )
        .sort_values("Category")
    )
    category_frame.to_csv(output_dir / "categorical_metrics.csv", index=False)

    runtime_frame = result_frame[
        ["file", "Category", "Runtime-Seconds"]
    ].copy()
    runtime_frame.to_csv(output_dir / "runtime_per_file.csv", index=False)
    runtime_by_category = (
        runtime_frame.groupby("Category", as_index=False)
        .agg(
            N=("file", "count"),
            avg_runtime_sec=("Runtime-Seconds", "mean"),
        )
        .sort_values("Category")
    )
    runtime_by_category.to_csv(
        output_dir / "runtime_by_category.csv",
        index=False,
    )

    json_records = summary_frame.where(
        pd.notna(summary_frame),
        None,
    ).to_dict("records")
    (output_dir / "summary_metrics.json").write_text(
        json.dumps(json_records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Run the paper-aligned RiCo detector"
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="TSB-AD CSV file or directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation_results"),
    )
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="maximum epochs per stage",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--wavelet",
        default="db2",
        choices=["db1", "db2", "bior3.1", "sym2", "sym4"],
    )
    parser.add_argument(
        "--wavelet-level",
        type=int,
        default=3,
        choices=[1, 2, 3, 4],
    )
    parser.add_argument(
        "--ode-steps",
        type=int,
        default=4,
        choices=[1, 2, 4, 8, 16],
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    return parser


def main() -> None:
    """Run RiCo over one file or a directory of benchmark files."""
    args = build_parser().parse_args()
    set_seed(args.seed)
    files = (
        [args.data]
        if args.data.is_file()
        else sorted(args.data.glob("*.csv"))
    )
    if not files:
        raise FileNotFoundError(f"no CSV files found under {args.data}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for index, path in enumerate(files, start=1):
        print(f"[{index}/{len(files)}] {path.name}")
        result = run_file(path, args.output_dir, args)
        results.append(result)
        save_aggregate_results(results, args.output_dir)
        print(
            "    "
            f"AUC-ROC={result['AUC-ROC']:.4f}, "
            f"AUC-PR={result['AUC-PR']:.4f}, "
            f"VUS-PR={result['VUS-PR']:.4f}, "
            f"VUS-ROC={result['VUS-ROC']:.4f}, "
            f"BestF1={result['BestF1']:.4f}, "
            f"RangeF1={result['RangeF1']:.4f}"
        )

    print(f"Results saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
