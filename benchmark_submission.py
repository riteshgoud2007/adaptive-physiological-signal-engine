"""
benchmark_submission.py — Benchmark system for the Adaptive Physiological Signal Engine.

Measures actual algorithmic performance.
NEVER fabricates metrics.
If data is unavailable, reports 'UNAVAILABLE' clearly.

Usage:
    python benchmark_submission.py

Outputs a multi-metric evaluation report to stdout.
"""

from __future__ import annotations

import sys
import time
import tracemalloc
from typing import Any, Optional

import numpy as np
from scipy import signal as sp_signal

# Import the competition engine
sys.path.insert(0, ".")
import final_submission as engine


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(
    ecg_signal: Optional[np.ndarray] = None,
    fs: float = 360.0,
    r_peaks_ref: Optional[np.ndarray] = None,
    rhythm_ref: Optional[str] = None,
) -> dict[str, Any]:
    """
    Run the full benchmark evaluation.

    Parameters
    ----------
    ecg_signal : np.ndarray, optional
        Real ECG signal to benchmark. If None, uses a synthetic sine wave
        for RUNTIME benchmarking only (clearly labelled).
    fs : float
        Sampling rate in Hz.
    r_peaks_ref : np.ndarray, optional
        Reference R-peak annotations (ground truth) for TP/FP/FN calculation.
    rhythm_ref : str, optional
        Reference rhythm label (ground truth) for classification accuracy.

    Returns
    -------
    dict with benchmark results.
    """
    report: dict[str, Any] = {}

    # --- Determine signal source ---
    if ecg_signal is None:
        # Sine-wave for runtime profiling ONLY
        n = int(60 * fs)
        t = np.linspace(0, 60, n)
        ecg_signal = np.sin(2 * np.pi * 1.2 * t) + 0.05 * np.random.RandomState(42).randn(n)
        report["signal_source"] = "SYNTHETIC SINE (runtime benchmark only — NOT clinical data)"
        report["clinical_metrics_available"] = False
    else:
        report["signal_source"] = "User-provided real signal"
        report["clinical_metrics_available"] = True

    report["signal_length_s"] = len(ecg_signal) / fs
    report["fs"] = fs

    # =================================================================
    # A. RUNTIME EFFICIENCY
    # =================================================================
    tracemalloc.start()
    t0 = time.perf_counter()
    result = engine.run(ecg_signal, fs, modality="ecg")
    t1 = time.perf_counter()
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    proc_time = t1 - t0
    n_samples = len(ecg_signal)
    samples_per_sec = n_samples / max(proc_time, 1e-9)
    rt_ratio = proc_time / max(report["signal_length_s"], 1e-9)

    report["runtime"] = {
        "processing_time_s": round(proc_time, 4),
        "samples_per_second": round(samples_per_sec, 0),
        "realtime_ratio": round(rt_ratio, 4),
        "peak_memory_kb": round(peak_mem / 1024.0, 1),
        "note": (
            "Faster than real-time (edge-suitable)."
            if rt_ratio < 1.0
            else "Slower than real-time."
        ),
    }

    # Engine result fields
    r_peaks_det = result.get("r_peaks", np.array([], dtype=int))
    hr_bpm = result.get("heart_rate_bpm", 0.0)
    rhythm = result.get("rhythm_label", "UNAVAILABLE")
    quality = result.get("signal_quality", "UNAVAILABLE")
    snr_before = result.get("snr_before_db")
    snr_after = result.get("snr_after_db")
    snr_improvement = result.get("snr_improvement_db")
    emg_severity = result.get("emg_severity", "UNAVAILABLE")
    reliability = result.get("detection_reliability", 0.0)
    morph_score = result.get("morphology_score")

    # =================================================================
    # B. AUTOMATED SOLUTION QUALITY
    # =================================================================
    report["solution_quality"] = {}
    sq = report["solution_quality"]

    # R-peak metrics (only if ground truth available)
    if r_peaks_ref is not None and len(r_peaks_ref) > 0 and report["clinical_metrics_available"]:
        tolerance_s = 0.05  # 50 ms standard tolerance
        metrics = _compute_detection_metrics(r_peaks_det, r_peaks_ref, fs, tolerance_s)
        sq["r_peak_precision"] = round(metrics["precision"], 4)
        sq["r_peak_recall"] = round(metrics["recall"], 4)
        sq["r_peak_f1"] = round(metrics["f1"], 4)
        sq["r_peak_tp"] = metrics["tp"]
        sq["r_peak_fp"] = metrics["fp"]
        sq["r_peak_fn"] = metrics["fn"]
        sq["r_peak_n_detected"] = len(r_peaks_det)
        sq["r_peak_n_reference"] = len(r_peaks_ref)

        # Timing jitter (key competition metric)
        jitter = _compute_timing_jitter(r_peaks_det, r_peaks_ref, fs, tolerance_s)
        sq["timing_error_mean_ms"] = round(jitter["mean_ms"], 2) if jitter["mean_ms"] is not None else None
        sq["timing_error_median_ms"] = round(jitter["median_ms"], 2) if jitter["median_ms"] is not None else None
        sq["timing_error_std_ms"] = round(jitter["std_ms"], 2) if jitter["std_ms"] is not None else None
        sq["timing_jitter_ms"] = round(jitter["jitter_ms"], 2) if jitter["jitter_ms"] is not None else None
        sq["timing_error_max_ms"] = round(jitter["max_ms"], 2) if jitter["max_ms"] is not None else None
    else:
        sq["r_peak_note"] = "UNAVAILABLE — no ground-truth annotations provided."
        sq["timing_jitter_note"] = "UNAVAILABLE — ground truth required for jitter calculation."

    # Rhythm classification (only if ground truth available)
    if rhythm_ref is not None and report["clinical_metrics_available"]:
        sq["rhythm_classified"] = rhythm
        sq["rhythm_reference"] = rhythm_ref
        sq["rhythm_correct"] = _rhythm_match(rhythm, rhythm_ref)
        sq["rhythm_confidence"] = result.get("rhythm_confidence", "UNRELIABLE")
        sq["rhythm_confidence_score"] = result.get("rhythm_confidence_score", 0.0)
    else:
        sq["rhythm_classified"] = rhythm
        sq["rhythm_confidence"] = result.get("rhythm_confidence", "UNRELIABLE")
        sq["rhythm_accuracy_note"] = "UNAVAILABLE — no ground-truth rhythm label provided."

    sq["heart_rate_bpm"] = round(hr_bpm, 1)
    sq["signal_quality"] = quality
    sq["detection_reliability"] = round(reliability, 3)
    sq["emg_severity"] = emg_severity
    sq["possible_vt"] = result.get("possible_vt", False)
    sq["possible_vf"] = result.get("possible_vf", False)

    # Sensitivity / Specificity (only with ground truth)
    if r_peaks_ref is not None and "r_peak_recall" in sq:
        sq["sensitivity"] = sq["r_peak_recall"]
        sq["note_specificity"] = "Specificity requires non-beat (TN) ground truth not provided here."
    else:
        sq["sensitivity_note"] = "UNAVAILABLE — ground truth required."

    # False alarm metrics (internal)
    dur_h = report["signal_length_s"] / 3600.0
    detection_latency = result.get("detection_latency_s")
    report["alert_metrics"] = {
        "detection_latency_s": round(detection_latency, 3) if detection_latency else None,
        "target_latency_s": 3.0,
        "latency_target_met": result.get("latency_target_met"),
        "false_alarm_rate_per_hour": "INTERNAL METRIC — requires annotated false alarms.",
    }

    # =================================================================
    # C. SIGNAL QUALITY / SNR
    # =================================================================
    report["signal_metrics"] = {
        "snr_before_db": round(snr_before, 2) if snr_before is not None else "UNAVAILABLE",
        "snr_after_db": round(snr_after, 2) if snr_after is not None else "UNAVAILABLE",
        "snr_improvement_db": round(snr_improvement, 2) if snr_improvement is not None else "UNAVAILABLE",
        "morphology_score": round(morph_score, 3) if morph_score is not None else "UNAVAILABLE",
        "qrs_amplitude_before": result.get("qrs_amplitude_before"),
        "qrs_amplitude_after": result.get("qrs_amplitude_after"),
        "st_level_before": result.get("st_level_before"),
        "st_level_after": result.get("st_level_after"),
        "powerline_50hz": result.get("powerline_50hz", False),
        "powerline_60hz": result.get("powerline_60hz", False),
    }

    # =================================================================
    # D. OFFICIAL FITNESS SCORE
    # =================================================================
    report["official_fitness"] = {
        "score": "UNAVAILABLE",
        "note": (
            "Official OptiForge evaluator not found in local project files. "
            "Score cannot be computed locally. "
            "Run the official evaluator to obtain the actual fitness score."
        ),
    }

    return report


def print_report(report: dict[str, Any]) -> None:
    """Print benchmark report to stdout."""
    sep = "=" * 70
    print(sep)
    print("  ADAPTIVE PHYSIOLOGICAL SIGNAL ENGINE — BENCHMARK REPORT")
    print(sep)
    print(f"  Signal source   : {report.get('signal_source', 'N/A')}")
    print(f"  Duration        : {report.get('signal_length_s', 0):.1f} s")
    print(f"  Sampling rate   : {report.get('fs', 0):.0f} Hz")
    print()

    print("A. AUTOMATED SOLUTION QUALITY")
    print("-" * 50)
    sq = report.get("solution_quality", {})
    for k, v in sq.items():
        print(f"  {k:<35}: {v}")
    print()

    print("B. RUNTIME EFFICIENCY")
    print("-" * 50)
    rt = report.get("runtime", {})
    for k, v in rt.items():
        print(f"  {k:<35}: {v}")
    print()

    print("C. SIGNAL METRICS / SNR")
    print("-" * 50)
    sm = report.get("signal_metrics", {})
    for k, v in sm.items():
        print(f"  {k:<35}: {v}")
    print()

    print("D. ALERT METRICS")
    print("-" * 50)
    am = report.get("alert_metrics", {})
    for k, v in am.items():
        print(f"  {k:<35}: {v}")
    print()

    print("E. OFFICIAL FITNESS SCORE")
    print("-" * 50)
    of = report.get("official_fitness", {})
    for k, v in of.items():
        print(f"  {k:<35}: {v}")
    print()
    print(sep)


# ---------------------------------------------------------------------------
# Metric calculation utilities
# ---------------------------------------------------------------------------

def _compute_detection_metrics(
    detected: np.ndarray,
    reference: np.ndarray,
    fs: float,
    tolerance_s: float = 0.05,
) -> dict[str, Any]:
    """Compute TP/FP/FN and Precision/Recall/F1."""
    tol_samples = int(tolerance_s * fs)
    tp, fp, fn = 0, 0, 0
    matched_ref: set[int] = set()

    for det in detected:
        distances = np.abs(reference.astype(int) - int(det))
        if len(distances) == 0:
            fp += 1
            continue
        closest_idx = int(np.argmin(distances))
        if distances[closest_idx] <= tol_samples and closest_idx not in matched_ref:
            tp += 1
            matched_ref.add(closest_idx)
        else:
            fp += 1

    fn = len(reference) - len(matched_ref)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
    }


def _compute_timing_jitter(
    detected: np.ndarray,
    reference: np.ndarray,
    fs: float,
    tolerance_s: float = 0.05,
) -> dict[str, Optional[float]]:
    """Compute timing error statistics for matched peaks."""
    tol_samples = int(tolerance_s * fs)
    errors_ms: list[float] = []
    matched_ref: set[int] = set()

    for det in detected:
        distances = np.abs(reference.astype(int) - int(det))
        if len(distances) == 0:
            continue
        closest_idx = int(np.argmin(distances))
        if distances[closest_idx] <= tol_samples and closest_idx not in matched_ref:
            matched_ref.add(closest_idx)
            error_ms = (int(det) - int(reference[closest_idx])) * 1000.0 / fs
            errors_ms.append(error_ms)

    if not errors_ms:
        return {"mean_ms": None, "median_ms": None, "std_ms": None,
                "jitter_ms": None, "max_ms": None}

    arr = np.array(errors_ms)
    return {
        "mean_ms": float(np.mean(np.abs(arr))),
        "median_ms": float(np.median(np.abs(arr))),
        "std_ms": float(np.std(arr)),
        "jitter_ms": float(np.std(arr)),  # Jitter = std of timing error
        "max_ms": float(np.max(np.abs(arr))),
    }


def _rhythm_match(classified: str, reference: str) -> bool:
    """Flexible rhythm label matching."""
    c = classified.lower()
    r = reference.lower()
    # Exact match
    if c == r:
        return True
    # Partial / keyword match
    keywords = ["normal", "tach", "brady", "fibrill", "flutter", "vt", "vf", "irregular"]
    for kw in keywords:
        if kw in c and kw in r:
            return True
    return False


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print()
    print("Running benchmark with synthetic signal (runtime profiling only).")
    print("For clinical metrics, call run_benchmark(ecg_signal, fs, r_peaks_ref, rhythm_ref).")
    print()
    report = run_benchmark()
    print_report(report)
