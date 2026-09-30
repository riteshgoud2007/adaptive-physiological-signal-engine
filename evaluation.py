"""
evaluation.py — Algorithm evaluation and comparison for the Adaptive ECG Engine.

Provides:
  * Ground-truth evaluation (TP/FP/FN/Precision/Recall/F1).
  * Algorithm comparison (Adaptive vs Pan-Tompkins vs Hamilton vs Christov).
  * Runtime benchmarking.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import signal as sp_signal

from config import REFRACTORY_S, COMPARISON_ALGORITHMS


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class DetectionMetrics:
    """Standard detection metrics for a single algorithm run."""
    method: str
    tp: int = 0
    fp: int = 0
    fn: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    mean_timing_error_ms: Optional[float] = None
    std_timing_error_ms: Optional[float] = None
    runtime_s: float = 0.0
    n_detected: int = 0
    n_reference: int = 0
    note: str = ""


@dataclass
class ComparisonReport:
    """Comparison of multiple detection algorithms."""
    metrics: list[DetectionMetrics] = field(default_factory=list)
    ground_truth_available: bool = False
    recording_duration_s: float = 0.0


# ---------------------------------------------------------------------------
# Ground-truth evaluation
# ---------------------------------------------------------------------------

def evaluate_against_ground_truth(
    detected: np.ndarray,
    reference: np.ndarray,
    fs: float,
    tolerance_s: float = 0.05,
    method_name: str = "Adaptive Engine",
    runtime_s: float = 0.0,
) -> DetectionMetrics:
    """
    Compare detected R-peaks against reference annotations.

    Parameters
    ----------
    detected :
        Detected R-peak sample indices.
    reference :
        Reference R-peak sample indices from ground truth.
    fs :
        Sampling rate in Hz.
    tolerance_s :
        Matching tolerance in seconds (default 50 ms).
    method_name :
        Label for this algorithm.
    runtime_s :
        Wall-clock time for detection.

    Returns
    -------
    DetectionMetrics
    """
    tol_samples = int(tolerance_s * fs)
    detected_sorted = np.sort(detected).astype(int)
    reference_sorted = np.sort(reference).astype(int)

    tp = 0
    fp = 0
    fn = 0
    timing_errors: list[float] = []

    matched_ref = np.zeros(len(reference_sorted), dtype=bool)
    matched_det = np.zeros(len(detected_sorted), dtype=bool)

    for di, dpeak in enumerate(detected_sorted):
        diffs = np.abs(reference_sorted - dpeak)
        closest_idx = int(np.argmin(diffs))
        if diffs[closest_idx] <= tol_samples:
            if not matched_ref[closest_idx]:
                tp += 1
                matched_ref[closest_idx] = True
                matched_det[di] = True
                timing_errors.append(
                    float(dpeak - reference_sorted[closest_idx]) / fs * 1000  # ms
                )
            else:
                # Already matched — false positive
                fp += 1
        else:
            fp += 1

    fn = int(np.sum(~matched_ref))

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = (2 * precision * recall / max(1e-9, precision + recall))

    mean_te = float(np.mean(timing_errors)) if timing_errors else None
    std_te = float(np.std(timing_errors)) if timing_errors else None

    return DetectionMetrics(
        method=method_name,
        tp=tp, fp=fp, fn=fn,
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        mean_timing_error_ms=round(mean_te, 2) if mean_te is not None else None,
        std_timing_error_ms=round(std_te, 2) if std_te is not None else None,
        runtime_s=runtime_s,
        n_detected=len(detected_sorted),
        n_reference=len(reference_sorted),
    )


# ---------------------------------------------------------------------------
# Algorithm comparison
# ---------------------------------------------------------------------------

def compare_algorithms(
    ecg: np.ndarray,
    fs: float,
    adaptive_peaks: np.ndarray,
    adaptive_runtime_s: float,
    reference_peaks: Optional[np.ndarray] = None,
) -> ComparisonReport:
    """
    Compare the Adaptive Engine against several established algorithms.

    Parameters
    ----------
    ecg :
        Raw ECG signal.
    fs :
        Sampling rate in Hz.
    adaptive_peaks :
        Peaks detected by the Adaptive Engine.
    adaptive_runtime_s :
        Wall-clock time for the adaptive engine.
    reference_peaks :
        Optional ground-truth annotations.

    Returns
    -------
    ComparisonReport
    """
    report = ComparisonReport(
        ground_truth_available=(reference_peaks is not None),
        recording_duration_s=len(ecg) / fs,
    )

    # Preprocessing shared for all methods
    from adaptive_engine import _interpolate_nans, _remove_baseline
    from scipy.signal import butter, sosfiltfilt
    from config import BP_LOW_HZ, BP_HIGH_HZ, BP_ORDER

    interpolated = _interpolate_nans(ecg)
    detrended = _remove_baseline(interpolated, fs)
    sos = butter(
        BP_ORDER,
        [BP_LOW_HZ / (fs / 2), min(0.99, BP_HIGH_HZ / (fs / 2))],
        btype="band", output="sos",
    )
    filtered = sosfiltfilt(sos, detrended)

    # Adaptive Engine
    if reference_peaks is not None:
        adaptive_metrics = evaluate_against_ground_truth(
            adaptive_peaks, reference_peaks, fs,
            method_name="Adaptive Engine",
            runtime_s=adaptive_runtime_s,
        )
    else:
        adaptive_metrics = DetectionMetrics(
            method="Adaptive Engine",
            n_detected=len(adaptive_peaks),
            runtime_s=adaptive_runtime_s,
            note="No ground truth",
        )
    report.metrics.append(adaptive_metrics)

    # Pan-Tompkins
    try:
        t0 = time.perf_counter()
        pt_peaks = _pantompkins(filtered, fs)
        pt_runtime = time.perf_counter() - t0
        if reference_peaks is not None:
            pt_metrics = evaluate_against_ground_truth(
                pt_peaks, reference_peaks, fs,
                method_name="Pan-Tompkins",
                runtime_s=pt_runtime,
            )
        else:
            pt_metrics = DetectionMetrics(
                method="Pan-Tompkins",
                n_detected=len(pt_peaks),
                runtime_s=pt_runtime,
                note="No ground truth",
            )
        report.metrics.append(pt_metrics)
    except Exception as exc:
        report.metrics.append(DetectionMetrics(
            method="Pan-Tompkins",
            note=f"Error: {exc}"
        ))

    # Hamilton
    try:
        t0 = time.perf_counter()
        hamilton_peaks = _hamilton(filtered, fs)
        hamilton_runtime = time.perf_counter() - t0
        if reference_peaks is not None:
            h_metrics = evaluate_against_ground_truth(
                hamilton_peaks, reference_peaks, fs,
                method_name="Hamilton",
                runtime_s=hamilton_runtime,
            )
        else:
            h_metrics = DetectionMetrics(
                method="Hamilton",
                n_detected=len(hamilton_peaks),
                runtime_s=hamilton_runtime,
                note="No ground truth",
            )
        report.metrics.append(h_metrics)
    except Exception as exc:
        report.metrics.append(DetectionMetrics(
            method="Hamilton",
            note=f"Error: {exc}"
        ))

    # Christov (simplified)
    try:
        t0 = time.perf_counter()
        christov_peaks = _christov(filtered, fs)
        christov_runtime = time.perf_counter() - t0
        if reference_peaks is not None:
            c_metrics = evaluate_against_ground_truth(
                christov_peaks, reference_peaks, fs,
                method_name="Christov",
                runtime_s=christov_runtime,
            )
        else:
            c_metrics = DetectionMetrics(
                method="Christov",
                n_detected=len(christov_peaks),
                runtime_s=christov_runtime,
                note="No ground truth",
            )
        report.metrics.append(c_metrics)
    except Exception as exc:
        report.metrics.append(DetectionMetrics(
            method="Christov",
            note=f"Error: {exc}"
        ))

    return report


# ---------------------------------------------------------------------------
# Reference algorithm implementations
# ---------------------------------------------------------------------------

def _pantompkins(filtered: np.ndarray, fs: float) -> np.ndarray:
    """Classic Pan-Tompkins with fixed 25% threshold."""
    from config import MWI_WINDOW_S, REFRACTORY_S
    deriv = np.diff(filtered, prepend=filtered[0])
    squared = deriv ** 2
    w = max(1, int(MWI_WINDOW_S * fs))
    cs = np.cumsum(squared)
    mwi = np.zeros_like(squared)
    mwi[w:] = (cs[w:] - cs[:-w]) / w
    mwi[:w] = cs[:w] / np.arange(1, w + 1)
    thresh = float(np.max(mwi)) * 0.25
    ref_n = max(1, int(REFRACTORY_S * fs))
    peaks, _ = sp_signal.find_peaks(mwi, height=thresh, distance=ref_n)
    return peaks


def _hamilton(filtered: np.ndarray, fs: float) -> np.ndarray:
    """
    Hamilton & Tompkins simplified detector.
    Uses derivative + threshold based on running estimate.
    """
    from config import REFRACTORY_S
    # Absolute derivative
    deriv = np.abs(np.diff(filtered, prepend=filtered[0]))
    # Smooth
    w = max(1, int(0.08 * fs))
    cs = np.cumsum(deriv)
    smoothed = np.zeros_like(deriv)
    smoothed[w:] = (cs[w:] - cs[:-w]) / w
    smoothed[:w] = cs[:w] / np.arange(1, w + 1)
    # Running threshold
    running_avg = float(np.mean(smoothed[:int(2 * fs)]))
    threshold = running_avg * 0.35
    ref_n = max(1, int(REFRACTORY_S * fs))
    peaks, _ = sp_signal.find_peaks(smoothed, height=threshold, distance=ref_n)
    return peaks


def _christov(filtered: np.ndarray, fs: float) -> np.ndarray:
    """
    Simplified Christov detector.
    Uses squared + gradient energy with adaptive threshold.
    """
    from config import REFRACTORY_S
    # Gradient + square
    grad = np.gradient(filtered)
    energy = grad ** 2 + filtered ** 2
    # Smooth energy
    w = max(1, int(0.05 * fs))
    cs = np.cumsum(energy)
    mwi = np.zeros_like(energy)
    mwi[w:] = (cs[w:] - cs[:-w]) / w
    mwi[:w] = cs[:w] / np.arange(1, w + 1)
    # Adaptive: use top 25%
    threshold = float(np.percentile(mwi, 75)) * 0.5
    ref_n = max(1, int(REFRACTORY_S * fs))
    peaks, _ = sp_signal.find_peaks(mwi, height=threshold, distance=ref_n)
    return peaks
