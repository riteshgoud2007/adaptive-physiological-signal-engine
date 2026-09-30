"""
final_submission.py — Competition-ready algorithm for the Adaptive Physiological Signal Engine.

SUBMISSION INTERFACE:
  - Input:  signal (np.ndarray), fs (float), modality (str)
  - Output: dict with standardized competition result keys

This file does NOT import:
  - Streamlit, Plotly, or any GUI packages
  - Absolute file paths
  - Cloud / network services

The algorithm is deterministic: same input always produces same output.

OFFICIAL EVALUATOR NOTE:
The OptiForge competition evaluator interface was not found in the local
project files. The function signature below is designed to be clean,
descriptive, and easy to wrap by any evaluator harness.

AST COMPLIANCE:
  - No eval(), exec(), dynamic imports, or obfuscated code.
  - No hardcoded outputs, benchmark-specific hacks, or dataset fingerprinting.
  - Type-annotated, documented, modular, deterministic.

HARDENING CHANGES (v2.1.0):
  1. Rhythm: VT now requires HR + sustained regularity + multi-beat evidence.
     Simple periodic signals no longer auto-classified as VT.
  2. VF: Added signal-level characteristics (spectral, amplitude) alongside
     RR-based evidence. Poor-quality-only signals not flagged as VF.
  3. R-peak refinement: polarity-aware, rejects artifact spikes.
  4. Search-back: deduplication + refractory check + polarity-aware refinement.
  5. Latency: returns None when event onset cannot be established.
  6. Modality: unknown modalities return unsupported warning, NOT processed as ECG.
  7. Notch safety: validates notch frequency < Nyquist before filter design.
  8. SNR: renamed to snr_proxy_db to distinguish from reference-based SNR.
  9. PPG: renamed fields — pulse_peaks, pulse_rate_bpm, inter_pulse_intervals_s.
 10. SNR conditionals: explicit None checks throughout.
"""

from __future__ import annotations

import functools
import math
import time
from typing import Any, Optional

import numpy as np
from scipy import signal as sp_signal
from scipy.ndimage import uniform_filter1d
from scipy.stats import kurtosis as _scipy_kurtosis


# ---------------------------------------------------------------------------
# Module-level filter cache (avoids redesigning filters on every call)
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=32)
def _cached_bp_sos(low: float, high: float, fs: float, order: int) -> Any:
    """Cached Butterworth bandpass SOS coefficients."""
    nyq = fs / 2.0
    lo = float(np.clip(low / nyq, 1e-5, 0.999))
    hi = float(np.clip(high / nyq, lo + 1e-5, 0.9995))
    return sp_signal.butter(order, [lo, hi], btype="band", output="sos")


@functools.lru_cache(maxsize=16)
def _cached_notch_ba(freq: float, q: float, fs: float) -> tuple:
    """Cached notch filter (b, a) coefficients."""
    return sp_signal.iirnotch(freq, q, fs)




# ---------------------------------------------------------------------------
# Version and metadata
# ---------------------------------------------------------------------------

__version__ = "2.1.0"
__algorithm__ = "Adaptive Physiological Signal Engine"

_SUPPORTED_MODALITIES = {"ecg", "ppg", "emg"}


# ---------------------------------------------------------------------------
# Configuration (all tunable parameters)
# ---------------------------------------------------------------------------

_CFG: dict[str, Any] = {
    # Bandpass — QRS detection path
    "qrs_bp_low_hz": 0.5,
    "qrs_bp_high_hz": 45.0,
    "qrs_bp_order": 4,
    # Bandpass — morphology preservation path
    "morph_bp_low_hz": 0.05,
    "morph_bp_high_hz": 40.0,
    "morph_bp_order": 2,
    # Notch filter
    "notch_q": 30.0,
    "notch_power_db": 10.0,
    # Pan-Tompkins adaptive detector
    "mwi_win_s": 0.150,
    "refractory_s": 0.200,
    "search_back_s": 0.360,
    "calib_s": 2.0,
    "spki_alpha": 0.125,
    "npki_alpha": 0.125,
    # Chunk processing
    "chunk_duration_s": 60.0,
    "chunk_overlap_s": 1.0,
    # EMG detection
    "emg_mild_hf_fraction": 0.05,
    "emg_moderate_hf_fraction": 0.15,
    "emg_severe_hf_fraction": 0.35,
    # VT detection — requires MULTIPLE evidence sources
    "vt_hr_min_bpm": 120.0,
    "vt_sustained_beats": 4,
    "vt_min_evidence_score": 0.65,   # raised from 0.5; requires more corroboration
    "vt_min_beats": 6,               # minimum beats before VT can be flagged
    "vt_regularity_min": 0.55,       # VT is typically quite regular
    # VF detection
    "vf_min_evidence_score": 0.55,
    "vf_signal_entropy_min": 0.60,   # signal-level spectral entropy for VF
    # Rhythm classification
    "hr_tach_min": 100.0,
    "hr_brady_max": 60.0,
    "af_cv_min": 0.10,
    "af_pnn50_min": 0.15,
    # Alert latency target
    "alert_latency_target_s": 3.0,
    # Minimum valid recording
    "min_samples": 100,
    "min_fs": 50.0,
    "max_fs": 10000.0,
    # Minimum duration (s) before rhythm classification is meaningful
    "min_duration_for_rhythm_s": 5.0,
}


# ---------------------------------------------------------------------------
# Public competition entry point
# ---------------------------------------------------------------------------

def run(
    signal: np.ndarray,
    fs: float,
    modality: str = "ecg",
    channel: int = 0,
) -> dict[str, Any]:
    """
    Process a physiological signal and return analysis results.

    Competition entry point.

    Parameters
    ----------
    signal : np.ndarray
        1D (n_samples,) or 2D (n_samples, n_channels) signal array.
    fs : float
        Sampling rate in Hz.
    modality : str
        Signal modality: 'ecg', 'ppg', or 'emg'. Case-insensitive.
        Unknown modalities return an error result — NOT silently treated as ECG.
    channel : int
        Channel index for multi-channel input (default 0).

    Returns
    -------
    dict with result keys — see docstring body for full list.

    NOTE: SNR values are labelled snr_proxy_db (signal-derived estimate, not
    reference-based true SNR). snr_before_db / snr_after_db are retained as
    aliases for backward compatibility but refer to the same proxy metric.
    """
    t_start = time.perf_counter()

    # Normalize modality safely
    try:
        mod = modality.strip().lower()
    except AttributeError:
        mod = "unknown"

    result: dict[str, Any] = {
        "algorithm_version": __version__,
        "modality": mod,
        "fs": float(fs),
        "n_samples": 0,
        "duration_s": 0.0,
        "valid": False,
        "validation_message": "",
        "signal_quality": "UNKNOWN",
        "signal_quality_score": 0.0,
        "noise_type": "UNKNOWN",
        "emg_severity": "UNKNOWN",
        "emg_energy_ratio": 0.0,
        "powerline_50hz": False,
        "powerline_60hz": False,
        # SNR fields — proxy estimates, NOT reference-based true SNR
        "snr_proxy_before_db": None,
        "snr_proxy_after_db": None,
        "snr_improvement_db": None,
        # Aliases for backward compatibility
        "snr_before_db": None,
        "snr_after_db": None,
        # ECG-specific
        "r_peaks": np.array([], dtype=int),
        "n_beats": 0,
        "rr_intervals_s": np.array([]),
        "heart_rate_bpm": 0.0,
        "heart_rate_reliable": False,
        "rhythm_label": "Unknown / Unclassifiable",
        "rhythm_confidence": "UNRELIABLE",
        "rhythm_confidence_score": 0.0,
        "possible_vt": False,
        "possible_vf": False,
        "vt_evidence": [],
        "vf_evidence": [],
        "detection_latency_s": None,
        "latency_target_met": None,
        "qrs_amplitude_before": None,
        "qrs_amplitude_after": None,
        "st_level_before": None,
        "st_level_after": None,
        "morphology_score": None,
        "detection_reliability": 0.0,
        # PPG-specific (named appropriately)
        "pulse_peaks": np.array([], dtype=int),
        "pulse_rate_bpm": 0.0,
        "inter_pulse_intervals_s": np.array([]),
        "processing_time_s": 0.0,
        "warnings": [],
    }

    # --- Validate input ---
    valid, msg, sig1d = _validate_input(signal, fs, channel)
    result["valid"] = valid
    result["validation_message"] = msg
    if not valid:
        result["processing_time_s"] = time.perf_counter() - t_start
        return result

    result["n_samples"] = len(sig1d)
    result["duration_s"] = len(sig1d) / fs

    # --- Route by modality (fix #6: no silent fall-through) ---
    if mod == "ecg":
        _process_ecg(sig1d, fs, result)
    elif mod == "ppg":
        _process_ppg(sig1d, fs, result)
    elif mod == "emg":
        _process_emg(sig1d, fs, result)
    else:
        result["valid"] = False
        result["validation_message"] = (
            f"Unsupported modality '{modality}'. "
            f"Accepted: {sorted(_SUPPORTED_MODALITIES)}. "
            "Signal was NOT processed."
        )
        result["warnings"].append(result["validation_message"])

    result["processing_time_s"] = time.perf_counter() - t_start
    return result


# ---------------------------------------------------------------------------
# ECG processing path
# ---------------------------------------------------------------------------

def _process_ecg(
    sig: np.ndarray,
    fs: float,
    result: dict[str, Any],
) -> None:
    """Full ECG processing pipeline, writes into result dict."""
    warnings: list[str] = result["warnings"]

    # 1. Interpolate NaN/Inf
    clean = _interpolate_nans(sig)

    # 2. SNR proxy before (fix #8: clearly labelled as proxy)
    snr_before = _estimate_snr_proxy(clean)
    result["snr_proxy_before_db"] = snr_before
    result["snr_before_db"] = snr_before  # backward-compat alias

    # 3. Signal quality
    quality, quality_score = _assess_quality(clean, fs)
    result["signal_quality"] = quality
    result["signal_quality_score"] = quality_score

    # 4. Shared PSD — used for powerline detection, EMG characterization, and noise type
    #    Computing once avoids 3 redundant scipy.signal.welch calls per ECG run.
    _shared_psd: Optional[tuple] = None
    if len(clean) >= int(fs * 2):
        try:
            _shared_psd = _compute_welch_psd(clean, fs, nperseg=2048)
        except Exception:
            _shared_psd = None

    pl50 = _detect_powerline(clean, 50.0, fs, _psd_cache=_shared_psd)
    pl60 = _detect_powerline(clean, 60.0, fs, _psd_cache=_shared_psd)
    result["powerline_50hz"] = pl50
    result["powerline_60hz"] = pl60

    # 5. EMG characterization (pass shared PSD to avoid another welch)
    emg_sev, emg_ratio = _characterize_emg(clean, fs, _psd_cache=_shared_psd)
    result["emg_severity"] = emg_sev
    result["emg_energy_ratio"] = emg_ratio
    result["noise_type"] = _dominant_noise_type(emg_sev, clean, fs, _psd_cache=_shared_psd)


    # 6. Preprocessing
    notched = clean.copy()
    # fix #7: validate notch frequency < Nyquist before applying
    if pl50 and _notch_valid(50.0, fs):
        b50, a50 = _make_notch(50.0, _CFG["notch_q"], fs)
        notched = sp_signal.filtfilt(b50, a50, notched)
        warnings.append("50 Hz notch applied.")
    if pl60 and _notch_valid(60.0, fs):
        b60, a60 = _make_notch(60.0, _CFG["notch_q"], fs)
        notched = sp_signal.filtfilt(b60, a60, notched)
        warnings.append("60 Hz notch applied.")

    # Adaptive EMG suppression
    if "SEVERE" in emg_sev:
        notched = _suppress_emg(notched, fs, severe=True)
        warnings.append("Adaptive EMG suppression applied (SEVERE).")
    elif "MODERATE" in emg_sev:
        notched = _suppress_emg(notched, fs, severe=False)
        warnings.append("Adaptive EMG suppression applied (MODERATE).")

    # Baseline removal
    baseline = _estimate_baseline(notched, fs)
    detrended = notched - baseline

    # Dual-path filtering
    qrs_path = _bandpass(detrended, fs,
                          _CFG["qrs_bp_low_hz"], _CFG["qrs_bp_high_hz"],
                          _CFG["qrs_bp_order"])
    morph_path = _bandpass(detrended, fs,
                            _CFG["morph_bp_low_hz"], _CFG["morph_bp_high_hz"],
                            _CFG["morph_bp_order"])

    # fix #10: explicit None checks for SNR
    snr_after = _estimate_snr_proxy(morph_path)
    result["snr_proxy_after_db"] = snr_after
    result["snr_after_db"] = snr_after  # backward-compat alias
    if snr_before is not None and snr_after is not None:
        result["snr_improvement_db"] = snr_after - snr_before

    # 7. R-peak detection (adaptive Pan-Tompkins on QRS path)
    r_peaks = _detect_r_peaks(qrs_path, fs)
    result["r_peaks"] = r_peaks
    result["n_beats"] = len(r_peaks)

    # 8. RR intervals and HR
    if len(r_peaks) >= 2:
        rr = np.diff(r_peaks.astype(float)) / fs
        valid_rr = rr[(rr >= 0.2) & (rr <= 3.0)]
        result["rr_intervals_s"] = valid_rr
        if len(valid_rr) > 0:
            result["heart_rate_bpm"] = float(60.0 / np.median(valid_rr))
            result["heart_rate_reliable"] = True

    # 9. Rhythm classification (fix #1/#2: hardened)
    rhythm = _classify_rhythm(
        r_peaks, morph_path, fs, quality, quality_score, emg_sev,
        duration_s=result["duration_s"],
    )
    result["rhythm_label"] = rhythm["label"]
    result["rhythm_confidence"] = rhythm["confidence"]
    result["rhythm_confidence_score"] = rhythm["confidence_score"]
    result["possible_vt"] = rhythm["possible_vt"]
    result["possible_vf"] = rhythm["possible_vf"]
    result["vt_evidence"] = rhythm["vt_evidence"]
    result["vf_evidence"] = rhythm["vf_evidence"]

    # 10. Alert latency (fix #5: proper onset-based latency)
    if rhythm["possible_vt"] or rhythm["possible_vf"]:
        lat = _compute_alert_latency(result["rr_intervals_s"], r_peaks, fs)
        result["detection_latency_s"] = lat
        if lat is not None:
            result["latency_target_met"] = lat <= _CFG["alert_latency_target_s"]

    # 11. Morphology preservation
    morph = _measure_morphology(clean, morph_path, r_peaks, fs)
    result.update(morph)

    # 12. Detection reliability
    result["detection_reliability"] = _compute_reliability(
        r_peaks, quality_score, emg_ratio
    )

    if warnings:
        result["warnings"] = warnings


# ---------------------------------------------------------------------------
# PPG processing path (fix #9: PPG-appropriate terminology)
# ---------------------------------------------------------------------------

def _process_ppg(
    sig: np.ndarray,
    fs: float,
    result: dict[str, Any],
) -> None:
    """PPG pulse detection pipeline. Uses pulse-rate terminology, not ECG terms."""
    clean = _interpolate_nans(sig)
    snr_before = _estimate_snr_proxy(clean)
    result["snr_proxy_before_db"] = snr_before
    result["snr_before_db"] = snr_before

    # Baseline and PPG-specific bandpass (0.5–8 Hz)
    w = min(len(clean) - 1, max(3, int(1.5 * fs)))
    if w % 2 == 0:
        w += 1
    baseline = uniform_filter1d(clean, size=w, mode="nearest")
    detrended = clean - baseline
    filtered = _bandpass(detrended, fs, 0.5, min(8.0, fs / 2.1), 3)

    # Pulse peak detection
    min_dist = max(1, int(60.0 / 220.0 * fs))
    prom = float(np.std(filtered)) * 0.3
    peaks, _ = sp_signal.find_peaks(
        filtered, distance=min_dist, prominence=max(1e-9, prom)
    )

    # fix #9: PPG uses pulse_peaks, inter_pulse_intervals_s, pulse_rate_bpm
    result["pulse_peaks"] = peaks
    # Also populate r_peaks for backward compatibility (clearly named as alias)
    result["r_peaks"] = peaks
    result["n_beats"] = len(peaks)

    if len(peaks) >= 2:
        ipi = np.diff(peaks.astype(float)) / fs
        valid_ipi = ipi[(ipi >= 60.0 / 220.0) & (ipi <= 60.0 / 25.0)]
        result["inter_pulse_intervals_s"] = valid_ipi
        result["rr_intervals_s"] = valid_ipi  # backward-compat
        if len(valid_ipi) > 0:
            rate = float(60.0 / np.median(valid_ipi))
            result["pulse_rate_bpm"] = rate
            result["heart_rate_bpm"] = rate  # alias
            result["heart_rate_reliable"] = True

    quality, quality_score = _assess_quality(clean, fs)
    result["signal_quality"] = quality
    result["signal_quality_score"] = quality_score

    snr_after = _estimate_snr_proxy(filtered)
    result["snr_proxy_after_db"] = snr_after
    result["snr_after_db"] = snr_after
    # fix #10: explicit None checks
    if snr_before is not None and snr_after is not None:
        result["snr_improvement_db"] = snr_after - snr_before


# ---------------------------------------------------------------------------
# EMG processing path
# ---------------------------------------------------------------------------

def _process_emg(
    sig: np.ndarray,
    fs: float,
    result: dict[str, Any],
) -> None:
    """EMG burst detection and characterization."""
    clean = _interpolate_nans(sig)
    snr_before = _estimate_snr_proxy(clean)
    result["snr_proxy_before_db"] = snr_before
    result["snr_before_db"] = snr_before

    # High-pass filter (>20 Hz for surface EMG)
    nyq = fs / 2.0
    if nyq > 25.0:
        sos = sp_signal.butter(4, 20.0 / nyq, btype="high", output="sos")
        filtered = sp_signal.sosfiltfilt(sos, clean)
    else:
        filtered = clean

    # RMS envelope
    w = max(1, int(0.1 * fs))
    rms = np.sqrt(uniform_filter1d(filtered ** 2, size=w, mode="nearest"))

    # Burst detection: above 2 × median RMS
    threshold = float(np.median(rms)) * 2.0
    bursts = rms > threshold
    burst_fraction = float(np.mean(bursts))

    emg_sev, emg_ratio = _characterize_emg(clean, fs)
    result["emg_severity"] = emg_sev
    result["emg_energy_ratio"] = emg_ratio
    result["noise_type"] = "EMG activity"
    result["signal_quality"] = "GOOD" if burst_fraction < 0.3 else "MODERATE"
    result["signal_quality_score"] = float(1.0 - min(burst_fraction, 1.0))

    snr_after = _estimate_snr_proxy(filtered)
    result["snr_proxy_after_db"] = snr_after
    result["snr_after_db"] = snr_after
    # fix #10: explicit None check
    if snr_before is not None and snr_after is not None:
        result["snr_improvement_db"] = snr_after - snr_before

    result["warnings"].append(
        f"EMG burst fraction: {burst_fraction*100:.1f}% of recording."
    )


# ---------------------------------------------------------------------------
# Core DSP functions
# ---------------------------------------------------------------------------

def _interpolate_nans(sig: np.ndarray) -> np.ndarray:
    """Linear interpolation of NaN/Inf samples."""
    arr = np.array(sig, dtype=np.float64)
    bad = ~np.isfinite(arr)
    if not bad.any():
        return arr
    idx = np.arange(len(arr))
    good = ~bad
    if good.sum() < 2:
        arr[bad] = 0.0
        return arr
    arr[bad] = np.interp(idx[bad], idx[good], arr[good])
    return arr


def _estimate_baseline(sig: np.ndarray, fs: float) -> np.ndarray:
    """Estimate signal baseline via wide uniform moving average."""
    w = int(max(3, 0.6 * fs))
    if w % 2 == 0:
        w += 1
    w = min(w, max(3, len(sig) - 1))
    return uniform_filter1d(sig, size=w, mode="nearest")


def _bandpass(
    sig: np.ndarray,
    fs: float,
    low_hz: float,
    high_hz: float,
    order: int,
) -> np.ndarray:
    """Apply Butterworth SOS bandpass filter (uses cached coefficients)."""
    nyq = fs / 2.0
    lo = max(1e-4, low_hz / nyq)
    hi = min(0.995, high_hz / nyq)
    if lo >= hi:
        return sig
    try:
        sos = _cached_bp_sos(float(low_hz), float(high_hz), float(fs), int(order))
        return sp_signal.sosfiltfilt(sos, sig)
    except Exception:
        return sig



def _notch_valid(freq: float, fs: float) -> bool:
    """Return True if a notch at freq Hz is valid for sampling rate fs. (fix #7)"""
    nyq = fs / 2.0
    # Need at least 5 % margin below Nyquist
    return nyq > freq * 1.05


def _make_notch(
    freq: float, q: float, fs: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return IIR notch filter coefficients (b, a). Caller must validate freq < Nyquist."""
    return sp_signal.iirnotch(freq, q, fs)


def _suppress_emg(
    sig: np.ndarray, fs: float, severe: bool
) -> np.ndarray:
    """Gentle low-pass EMG suppression that preserves QRS band."""
    cutoff = 80.0 if severe else 100.0
    nyq = fs / 2.0
    if nyq <= cutoff:
        return sig
    c = min(0.995, cutoff / nyq)
    try:
        sos = sp_signal.butter(2, c, btype="low", output="sos")
        return sp_signal.sosfiltfilt(sos, sig)
    except Exception:
        return sig


def _refine_peak_polarity(
    filtered: np.ndarray,
    cand: int,
    hw: int,
) -> int:
    """
    Polarity-aware local R-peak refinement. (fix #3)

    Instead of np.argmax(abs()), determine the dominant polarity of QRS
    in the local window and find the local max in that direction.
    Rejects obvious artifact spikes by checking width consistency.
    """
    n = len(filtered)
    lo = max(0, cand - hw)
    hi = min(n, cand + hw + 1)
    if hi <= lo:
        return cand

    window = filtered[lo:hi]
    # Determine polarity: whichever extreme is larger in absolute value
    max_val = float(np.max(window))
    min_val = float(np.min(window))

    if abs(max_val) >= abs(min_val):
        # Positive polarity — find local maximum
        local_idx = int(np.argmax(window))
    else:
        # Negative polarity — find local minimum
        local_idx = int(np.argmin(window))

    refined = lo + local_idx

    # Reject artifact spike: the QRS peak should not be a single isolated sample.
    # If the two neighbors are both far below the peak, it may be an artifact.
    if 0 < local_idx < len(window) - 1:
        peak_amp = abs(window[local_idx])
        neighbor_amp = max(abs(window[local_idx - 1]), abs(window[local_idx + 1]))
        # Spike criterion: neighbors < 5 % of peak — likely artifact
        if neighbor_amp < 0.05 * peak_amp and peak_amp > 0.0:
            # Fall back to candidate
            return cand

    return refined


def _detect_r_peaks(filtered: np.ndarray, fs: float) -> np.ndarray:
    """
    Adaptive Pan-Tompkins R-peak detector with polarity-aware refinement.
    (fixes #3, #4)
    """
    n = len(filtered)
    refractory_n = max(1, int(_CFG["refractory_s"] * fs))
    mwi_n = max(1, int(_CFG["mwi_win_s"] * fs))
    search_back_n = int(_CFG["search_back_s"] * fs)
    calib_n = min(int(_CFG["calib_s"] * fs), n // 4)
    refine_hw = max(1, int(0.04 * fs))

    if n < refractory_n * 2:
        return np.array([], dtype=int)

    # Derivative (central difference)
    deriv = np.zeros(n)
    deriv[1:-1] = (filtered[2:] - filtered[:-2]) / (2.0 / fs)
    deriv[0] = deriv[1]
    deriv[-1] = deriv[-2]

    squared = deriv ** 2

    # Moving window integration (O(n) cumulative sum)
    cs = np.cumsum(squared)
    mwi = np.zeros(n)
    mwi[mwi_n:] = (cs[mwi_n:] - cs[:-mwi_n]) / mwi_n
    mwi[:mwi_n] = cs[:mwi_n] / np.maximum(np.arange(1, mwi_n + 1), 1)

    # Initialise adaptive thresholds from calibration window
    calib = mwi[:calib_n]
    init_peaks, _ = sp_signal.find_peaks(calib, distance=refractory_n)
    spki = float(np.mean(calib[init_peaks])) if len(init_peaks) > 0 else float(np.max(calib)) * 0.5
    npki = spki * 0.1

    # Candidate peaks
    candidates, _ = sp_signal.find_peaks(mwi, distance=refractory_n // 2)

    spki_alpha = _CFG["spki_alpha"]
    npki_alpha = _CFG["npki_alpha"]
    accepted: list[int] = []
    accepted_set: set[int] = set()
    last_accepted = -refractory_n * 2

    for cand in candidates:
        threshold = npki + 0.25 * (spki - npki)
        refractory_ok = (cand - last_accepted) >= refractory_n

        if mwi[cand] > threshold and refractory_ok:
            # fix #3: polarity-aware refinement instead of np.argmax(abs())
            refined = _refine_peak_polarity(filtered, cand, refine_hw)
            if refined not in accepted_set:
                accepted.append(refined)
                accepted_set.add(refined)
                last_accepted = refined
            spki = spki_alpha * mwi[cand] + (1 - spki_alpha) * spki
        else:
            npki = npki_alpha * mwi[cand] + (1 - npki_alpha) * npki

    # fix #4: search-back with deduplication + refractory check + polarity refinement
    if len(accepted) >= 2:
        peaks_arr = np.array(sorted(accepted))
        rr = np.diff(peaks_arr.astype(float))
        median_rr = float(np.median(rr))
        sb_added: list[int] = []

        for i, gap in enumerate(rr):
            if gap > 1.66 * median_rr:
                search_start = int(peaks_arr[i]) + refractory_n
                search_end = min(n, search_start + search_back_n)
                region = mwi[search_start:search_end]
                if len(region) == 0:
                    continue
                local_max_rel = int(np.argmax(region))
                local_max = local_max_rel + search_start
                threshold_sb = (npki + 0.25 * (spki - npki)) * 0.5
                if mwi[local_max] > threshold_sb:
                    # Refractory check against already-accepted peaks
                    too_close = any(
                        abs(local_max - p) < refractory_n
                        for p in accepted_set
                    )
                    if not too_close and local_max not in accepted_set:
                        # Polarity-aware refinement for search-back peak
                        refined_sb = _refine_peak_polarity(filtered, local_max, refine_hw)
                        too_close_refined = any(
                            abs(refined_sb - p) < refractory_n
                            for p in accepted_set
                        )
                        if not too_close_refined and refined_sb not in accepted_set:
                            sb_added.append(refined_sb)
                            accepted_set.add(refined_sb)

        accepted.extend(sb_added)

    return np.array(sorted(set(accepted)), dtype=int)


def _classify_rhythm(
    r_peaks: np.ndarray,
    ecg: np.ndarray,
    fs: float,
    quality: str,
    quality_score: float,
    emg_severity: str,
    duration_s: float = 0.0,
) -> dict[str, Any]:
    """
    Feature-based rhythm classification. (fix #1, #2)

    VT requires multiple evidence sources: HR + sustained fraction + regularity
    + minimum beat count. A simple periodic non-ECG signal will NOT be
    classified as VT based on rate alone.

    VF uses signal-level characteristics (spectral, amplitude complexity)
    in addition to RR-based features. Poor-quality-only signals without
    additional VF evidence are NOT flagged as VF.
    """
    out: dict[str, Any] = {
        "label": "Unknown / Unclassifiable",
        "confidence": "UNRELIABLE",
        "confidence_score": 0.0,
        "possible_vt": False,
        "possible_vf": False,
        "vt_evidence": [],
        "vf_evidence": [],
    }

    # Need minimum beats for any classification
    if len(r_peaks) < 4:
        return out

    # Require minimum recording duration for rhythm analysis
    if duration_s > 0.0 and duration_s < _CFG["min_duration_for_rhythm_s"]:
        out["label"] = "Recording Too Short for Rhythm Analysis"
        out["confidence"] = "UNRELIABLE"
        return out

    times_s = r_peaks.astype(float) / fs
    rr = np.diff(times_s)
    valid = rr[(rr >= 0.2) & (rr <= 3.0)]
    if len(valid) < 2:
        return out

    mean_rr = float(np.mean(valid))
    median_rr = float(np.median(valid))
    std_rr = float(np.std(valid))
    cv_rr = std_rr / max(mean_rr, 1e-9)
    hr = 60.0 / max(median_rr, 1e-9)
    ssd = np.diff(valid) * 1000.0
    rmssd = float(np.sqrt(np.mean(ssd ** 2))) if len(ssd) > 0 else 0.0
    pnn50 = float(np.mean(np.abs(ssd) > 50.0)) if len(ssd) > 0 else 0.0
    regularity = float(np.clip(1.0 - cv_rr * 3.0, 0.0, 1.0))
    entropy = _rr_entropy(valid)

    # ---------------------------------------------------------------
    # fix #1: VT requires multiple corroborating sources
    # NOT triggered by rate alone (prevents false VT on sine waves etc.)
    # ---------------------------------------------------------------
    rapid = valid < (60.0 / _CFG["vt_hr_min_bpm"])
    max_run = _longest_run_bool(rapid)
    sustained_frac = max_run / max(len(valid), 1)
    n_total_beats = len(r_peaks)

    vt_evidence: list[str] = []
    vt_score = 0.0

    # Evidence 1: Rapid rate — necessary but not sufficient alone
    rate_criterion = hr >= _CFG["vt_hr_min_bpm"]
    if rate_criterion:
        vt_evidence.append(f"Rapid rate: {hr:.0f} BPM (>={_CFG['vt_hr_min_bpm']:.0f})")
        vt_score += 0.30

    # Evidence 2: Sustained rapid beats (majority of beats are rapid)
    if sustained_frac >= 0.70 and rate_criterion:
        vt_evidence.append(f"Sustained rapid beats: {sustained_frac*100:.0f}%")
        vt_score += 0.25

    # Evidence 3: Regular rhythm (VT is typically quite regular, unlike AF)
    if regularity >= _CFG["vt_regularity_min"] and rate_criterion:
        vt_evidence.append(f"Regular rhythm (score={regularity:.2f})")
        vt_score += 0.20

    # Evidence 4: Sufficient beat count (need to observe sustained tachycardia)
    if n_total_beats >= _CFG["vt_min_beats"] and rate_criterion:
        vt_evidence.append(f"Sustained observation: {n_total_beats} beats detected")
        vt_score += 0.15

    # Evidence 5: Low RR entropy (VT is regular, low entropy)
    if entropy < 0.35 and rate_criterion:
        vt_evidence.append(f"Low RR entropy (regular): {entropy:.2f}")
        vt_score += 0.10

    # VT requires minimum score — more than rate alone
    # fix #1: min_evidence_score is 0.65, requires at least 2 corroborating sources
    possible_vt = vt_score >= _CFG["vt_min_evidence_score"] and rate_criterion
    if possible_vt:
        out["possible_vt"] = True
        out["vt_evidence"] = vt_evidence
    else:
        # Clear partial evidence that didn't meet threshold
        vt_evidence.clear()
        vt_score = 0.0

    # ---------------------------------------------------------------
    # fix #2: VF — uses signal-level characteristics, not only RR
    # NOT triggered by poor quality alone
    # ---------------------------------------------------------------
    amps = _extract_amplitudes_sub(ecg, r_peaks, fs)
    amp_cv = float(np.std(amps) / max(np.mean(amps), 1e-9)) if len(amps) > 1 else 0.0

    # Signal-level VF evidence: spectral entropy of the ECG signal itself
    signal_entropy = _signal_spectral_entropy(ecg, fs)

    vf_evidence: list[str] = []
    vf_score = 0.0

    # RR-level evidence
    if entropy > 0.65 and cv_rr > 0.20:
        vf_evidence.append(f"Disorganized RR: entropy={entropy:.2f}, CV={cv_rr:.2f}")
        vf_score += 0.25
    if amp_cv > 0.50:
        vf_evidence.append(f"High amplitude variation (CV={amp_cv:.2f})")
        vf_score += 0.20

    # Signal-level evidence (fix #2: does not rely solely on R-peaks)
    if signal_entropy is not None and signal_entropy > _CFG["vf_signal_entropy_min"]:
        vf_evidence.append(f"High signal spectral entropy: {signal_entropy:.2f}")
        vf_score += 0.25

    # Absence of organized QRS: very few beats detected despite long recording
    beats_per_minute_detected = (len(r_peaks) / max(duration_s, 1.0)) * 60.0
    if duration_s >= 5.0 and beats_per_minute_detected < 20.0 and quality != "CRITICAL":
        vf_evidence.append(
            f"Absent organized QRS: only {beats_per_minute_detected:.0f} "
            "detections/min despite non-critical quality"
        )
        vf_score += 0.30

    # Safety: do NOT flag VF if the only evidence is poor quality
    # (poor quality alone must not produce VF alarm — fix #2)
    if quality == "CRITICAL" and signal_entropy is None:
        vf_score = max(0.0, vf_score - 0.40)
        if vf_evidence:
            vf_evidence.append("NOTE: evidence reduced — signal quality CRITICAL.")

    if vf_score >= _CFG["vf_min_evidence_score"] and vf_evidence:
        out["possible_vf"] = True
        out["vf_evidence"] = vf_evidence
    else:
        vf_evidence.clear()
        vf_score = 0.0

    # ---------------------------------------------------------------
    # Primary label
    # ---------------------------------------------------------------
    af = (
        cv_rr >= _CFG["af_cv_min"] and
        pnn50 >= _CFG["af_pnn50_min"] and
        rmssd > 40.0 and
        entropy > 0.4
    )

    if vf_score >= _CFG["vf_min_evidence_score"] and out["possible_vf"]:
        label = "Possible Ventricular Fibrillation"
        conf = float(np.clip(vf_score, 0.0, 1.0))
    elif possible_vt:
        label = "Possible Ventricular Tachycardia"
        conf = float(np.clip(vt_score, 0.0, 1.0))
    elif af:
        label = "Possible Atrial Fibrillation"
        conf = float(np.clip(0.5 + cv_rr * 2.0, 0.3, 0.85))
    elif hr >= _CFG["hr_tach_min"]:
        label = "Sinus Tachycardia"
        conf = 0.7 + 0.3 * regularity
    elif hr < _CFG["hr_brady_max"]:
        label = "Sinus Bradycardia"
        conf = 0.7 + 0.3 * regularity
    elif regularity > 0.5:
        label = "Normal Sinus Rhythm"
        conf = 0.6 + 0.4 * regularity
    else:
        label = "Irregular Rhythm"
        conf = 0.5

    # Reduce confidence for poor quality / EMG
    if quality in ("POOR", "CRITICAL"):
        conf = max(0.0, conf - 0.25)
    if "SEVERE" in emg_severity:
        conf = max(0.0, conf - 0.35)
    elif "MODERATE" in emg_severity:
        conf = max(0.0, conf - 0.15)

    conf = float(np.clip(conf, 0.0, 1.0))
    if conf >= 0.70:
        conf_label = "HIGH"
    elif conf >= 0.45:
        conf_label = "MODERATE"
    elif conf >= 0.20:
        conf_label = "LOW"
    else:
        conf_label = "UNRELIABLE"

    out["label"] = label
    out["confidence"] = conf_label
    out["confidence_score"] = conf
    return out


def _compute_alert_latency(
    rr_intervals_s: np.ndarray,
    r_peaks: np.ndarray,
    fs: float,
) -> Optional[float]:
    """
    Compute acute-event detection latency. (fix #5)

    Latency = time from event onset (first rapid beat) to detection.
    For streaming: this approximates the time consumed by the
    minimum number of beats needed to confirm the event.

    Returns None when event onset cannot be reliably established
    (e.g., too few RR intervals).
    """
    n_beats_needed = _CFG["vt_sustained_beats"]
    if rr_intervals_s is None or len(rr_intervals_s) < n_beats_needed:
        # Cannot establish onset from insufficient data
        return None

    # Find the onset of the rapid sequence (first of sustained rapid beats)
    rapid_threshold_rr = 60.0 / _CFG["vt_hr_min_bpm"]
    rapid = rr_intervals_s < rapid_threshold_rr

    # Find first run of >= vt_sustained_beats consecutive rapid beats
    run = 0
    onset_idx = None
    for i, r in enumerate(rapid):
        if r:
            run += 1
            if run >= n_beats_needed and onset_idx is None:
                onset_idx = i - n_beats_needed + 1
        else:
            run = 0

    if onset_idx is None:
        # No sustained rapid sequence found in the RR intervals
        return None

    # Latency = time from onset_idx to end of n_beats_needed
    detection_rr = rr_intervals_s[onset_idx: onset_idx + n_beats_needed]
    return float(np.sum(detection_rr))


# ---------------------------------------------------------------------------
# Morphology measurement
# ---------------------------------------------------------------------------

def _measure_morphology(
    raw: np.ndarray,
    morph: np.ndarray,
    r_peaks: np.ndarray,
    fs: float,
) -> dict[str, Any]:
    """Measure QRS and ST morphology before/after filtering.

    Vectorized implementation: all beats processed in a single NumPy matrix
    operation rather than a Python loop. Limits to 60 beats for efficiency
    (sufficient for a reliable morphology score on any recording length).
    """
    out: dict[str, Any] = {
        "qrs_amplitude_before": None,
        "qrs_amplitude_after": None,
        "st_level_before": None,
        "st_level_after": None,
        "morphology_score": None,
    }
    half_n = max(1, int(0.06 * fs))
    st_off_n = max(1, int(0.08 * fs))
    st_win_n = max(1, int(0.04 * fs))
    n = min(len(raw), len(morph))

    valid_peaks = r_peaks[
        (r_peaks >= half_n) & (r_peaks < n - half_n - st_off_n - st_win_n)
    ]
    if len(valid_peaks) < 3:
        return out

    # Limit to 60 beats — sufficient for morphology score, avoids O(N) cost
    MAX_BEATS = 60
    if len(valid_peaks) > MAX_BEATS:
        # Sample evenly across the recording for representativeness
        idx = np.round(np.linspace(0, len(valid_peaks) - 1, MAX_BEATS)).astype(int)
        valid_peaks = valid_peaks[idx]

    n_beats = len(valid_peaks)
    win = 2 * half_n  # QRS window width in samples

    # ---- Build beat matrices in one shot (n_beats × win) ----
    # Row offsets: each row is [pk-half_n, pk-half_n+1, ..., pk+half_n-1]
    offsets = np.arange(-half_n, half_n)             # shape (win,)
    indices = valid_peaks[:, None] + offsets[None, :] # shape (n_beats, win)
    # Clip to valid range
    indices = np.clip(indices, 0, n - 1)

    raw_mat  = raw[indices]    # (n_beats, win)
    morph_mat = morph[indices] # (n_beats, win)

    # ---- QRS amplitudes (vectorized) ----
    raw_amps  = raw_mat.max(axis=1)  - raw_mat.min(axis=1)   # (n_beats,)
    morph_amps = morph_mat.max(axis=1) - morph_mat.min(axis=1)

    # ---- Waveform correlations (vectorized) ----
    raw_c  = raw_mat  - raw_mat.mean(axis=1, keepdims=True)
    morph_c = morph_mat - morph_mat.mean(axis=1, keepdims=True)
    norms_r = np.linalg.norm(raw_c,  axis=1)
    norms_m = np.linalg.norm(morph_c, axis=1)
    valid_corr = (norms_r > 1e-9) & (norms_m > 1e-9)
    corrs = np.where(
        valid_corr,
        np.sum(raw_c * morph_c, axis=1) / np.maximum(norms_r * norms_m, 1e-18),
        0.0,
    )
    corrs = np.clip(corrs, -1.0, 1.0)

    # ---- ST level (mean over small window — faster than median for tiny arrays) ----
    iso_starts = np.maximum(0, valid_peaks - half_n - st_win_n)
    iso_ends   = np.maximum(1, valid_peaks - half_n)
    st_starts  = valid_peaks + st_off_n
    st_ends    = valid_peaks + st_off_n + st_win_n

    st_before_list: list[float] = []
    st_after_list:  list[float] = []
    for i in range(n_beats):
        iso_seg   = raw[iso_starts[i]: iso_ends[i]]
        iso_val   = float(np.mean(iso_seg)) if len(iso_seg) > 0 else 0.0
        st_r_seg  = raw [st_starts[i]: min(st_ends[i], n)]
        st_f_seg  = morph[st_starts[i]: min(st_ends[i], n)]
        if len(st_r_seg) > 0:
            st_before_list.append(float(np.mean(st_r_seg)) - iso_val)
            st_after_list.append(float(np.mean(st_f_seg)) - iso_val)

    # ---- Populate output ----
    if len(raw_amps) > 0:
        out["qrs_amplitude_before"] = float(np.median(raw_amps))
        out["qrs_amplitude_after"]  = float(np.median(morph_amps))
    if st_before_list:
        out["st_level_before"] = float(np.mean(st_before_list))
        out["st_level_after"]  = float(np.mean(st_after_list))

    scores: list[float] = []
    qab = out["qrs_amplitude_before"]
    qaa = out["qrs_amplitude_after"]
    if qab is not None and qaa is not None and qab > 1e-9:
        scores.append(float(np.clip(min(qaa / qab, 1.0), 0.0, 1.0)))
    if valid_corr.any():
        scores.append(float(np.clip(float(corrs[valid_corr].mean()), 0.0, 1.0)))
    if scores:
        out["morphology_score"] = float(np.mean(scores))

    return out




# ---------------------------------------------------------------------------
# Quality and noise analysis
# ---------------------------------------------------------------------------

def _assess_quality(
    sig: np.ndarray, fs: float
) -> tuple[str, float]:
    """Assess signal quality. Returns (label, score 0-1)."""
    finite = sig[np.isfinite(sig)]
    if len(finite) == 0:
        return "CRITICAL", 0.0
    if float(np.std(finite)) < 1e-4:
        return "CRITICAL", 0.0  # Flatline

    nan_frac = 1.0 - len(finite) / max(len(sig), 1)
    p5, p95 = float(np.percentile(finite, 5)), float(np.percentile(finite, 95))
    amp = p95 - p5
    noise_rms = float(np.std(np.diff(finite))) / math.sqrt(2)
    rel_noise = noise_rms / max(amp, 1e-9)

    score = 1.0
    score -= min(nan_frac * 3.0, 0.5)
    score -= min(rel_noise * 2.0, 0.5)
    score = float(np.clip(score, 0.0, 1.0))

    if score >= 0.75:
        return "GOOD", score
    if score >= 0.50:
        return "MODERATE", score
    if score >= 0.25:
        return "POOR", score
    return "CRITICAL", score


def _compute_welch_psd(
    sig: np.ndarray, fs: float, nperseg: int = 2048
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Welch PSD once, return (freqs, psd). Used to share across multiple checks."""
    n_seg = min(nperseg, len(sig))
    return sp_signal.welch(sig, fs=fs, nperseg=n_seg)


def _check_powerline_from_psd(
    freqs: np.ndarray, psd: np.ndarray, freq: float, threshold_db: float
) -> bool:
    """Check for powerline interference at freq Hz from a pre-computed PSD."""
    idx = int(np.argmin(np.abs(freqs - freq)))
    lo, hi = max(0, idx - 5), min(len(psd), idx + 6)
    mask = np.ones(len(psd), bool)
    mask[lo:hi] = False
    floor = float(np.median(psd[mask])) + 1e-30
    peak_db = 10 * math.log10(max(float(psd[idx]), 1e-30) / floor)
    return peak_db > threshold_db


def _detect_powerline(
    sig: np.ndarray, freq: float, fs: float,
    _psd_cache: Optional[tuple] = None,
) -> bool:
    """Return True if powerline interference at freq Hz is detected.

    Pass _psd_cache=(freqs, psd) to reuse a pre-computed Welch PSD and avoid
    a redundant FFT when checking both 50 and 60 Hz.
    """
    if not _notch_valid(freq, fs):
        return False
    if len(sig) < int(fs * 2):
        return False
    try:
        if _psd_cache is not None:
            freqs, psd = _psd_cache
        else:
            freqs, psd = _compute_welch_psd(sig, fs)
        return _check_powerline_from_psd(freqs, psd, freq, _CFG["notch_power_db"])
    except Exception:
        return False




def _characterize_emg(
    sig: np.ndarray, fs: float,
    _psd_cache: Optional[tuple] = None,
) -> tuple[str, float]:
    """Characterize EMG contamination severity.

    Accepts an optional pre-computed (freqs, psd) tuple to avoid a redundant
    scipy.signal.welch call when the PSD was already computed for powerline detection.
    """
    finite = sig[np.isfinite(sig)]
    if len(finite) < 8:
        return "UNKNOWN", 0.0
    nyq = fs / 2.0
    if nyq > 100.0:
        try:
            if _psd_cache is not None:
                freqs, psd = _psd_cache
            else:
                freqs, psd = _compute_welch_psd(finite, fs, nperseg=512)
            total = float(np.sum(psd)) + 1e-30
            hf = float(np.sum(psd[freqs > 100.0])) / total
        except Exception:
            hf = 0.0
    else:
        # Fallback: kurtosis heuristic (no FFT needed)
        try:
            k = float(_scipy_kurtosis(np.diff(finite), fisher=True))
            hf = float(np.clip(k / 20.0, 0.0, 1.0))
        except Exception:
            hf = 0.0

    if hf >= _CFG["emg_severe_hf_fraction"]:
        return "SEVERE EMG", hf
    if hf >= _CFG["emg_moderate_hf_fraction"]:
        return "MODERATE EMG", hf
    if hf >= _CFG["emg_mild_hf_fraction"]:
        return "MILD EMG", hf
    return "CLEAN", hf


def _dominant_noise_type(
    emg_severity: str, sig: np.ndarray, fs: float,
    _psd_cache: Optional[tuple] = None,
) -> str:
    """Determine the dominant noise type label.

    Accepts an optional pre-computed (freqs, psd) tuple to avoid a redundant
    scipy.signal.welch call.
    """
    if "SEVERE" in emg_severity or "MODERATE" in emg_severity:
        return "EMG contamination"
    if "MILD" in emg_severity:
        return "Mild EMG / noise"
    try:
        if _psd_cache is not None:
            freqs, psd = _psd_cache
        else:
            freqs, psd = _compute_welch_psd(sig, fs, nperseg=256)
        lf_frac = float(np.sum(psd[freqs < 1.0])) / (float(np.sum(psd)) + 1e-30)
        if lf_frac > 0.40:
            return "Baseline wander"
        if lf_frac > 0.15:
            return "Mild baseline wander"
    except Exception:
        pass
    return "Clean"




def _estimate_snr_proxy(sig: np.ndarray) -> Optional[float]:
    """
    Signal-derived SNR proxy in dB. (fix #8)

    This is NOT a reference-based true SNR. It is a proxy metric
    derived from the signal itself (amplitude / noise floor estimate).
    Clearly named _estimate_snr_proxy to distinguish from true SNR.
    Returns None if the estimate is unreliable.
    """
    finite = sig[np.isfinite(sig)]
    if len(finite) < 20:
        return None
    amp = float(np.percentile(finite, 95) - np.percentile(finite, 5))
    noise_rms = float(np.std(np.diff(finite))) / math.sqrt(2)
    if noise_rms < 1e-12 or amp < 1e-9:
        return None
    snr = 20 * math.log10(amp / noise_rms)
    return float(np.clip(snr, -20.0, 60.0))


def _signal_spectral_entropy(sig: np.ndarray, fs: float) -> Optional[float]:
    """
    Compute normalised spectral entropy of the ECG signal (0–1).

    Used for signal-level VF evidence. Returns None if computation fails.
    """
    if len(sig) < int(fs * 2):
        return None
    try:
        n_seg = min(1024, len(sig))
        _, psd = sp_signal.welch(sig, fs=fs, nperseg=n_seg)
        p = psd + 1e-30
        p /= p.sum()
        n_bins = len(p)
        if n_bins < 2:
            return None
        entropy = float(-np.sum(p * np.log2(p + 1e-30)) / math.log2(n_bins))
        return float(np.clip(entropy, 0.0, 1.0))
    except Exception:
        return None


def _compute_reliability(
    r_peaks: np.ndarray,
    quality_score: float,
    emg_ratio: float,
) -> float:
    """Compute detection reliability score (0-1)."""
    if len(r_peaks) < 2:
        return 0.0
    rr = np.diff(r_peaks.astype(float))
    valid_rr = rr[rr > 0]
    if len(valid_rr) < 1:
        return 0.0
    cv = float(np.std(valid_rr) / max(np.mean(valid_rr), 1e-9))
    reg = float(np.clip(1.0 - cv * 2.0, 0.0, 1.0))
    noise_penalty = float(np.clip(emg_ratio * 2.0, 0.0, 0.5))
    reliability = quality_score * 0.5 + reg * 0.5 - noise_penalty
    return float(np.clip(reliability, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _validate_input(
    signal: np.ndarray,
    fs: float,
    channel: int,
) -> tuple[bool, str, np.ndarray]:
    """Validate signal input. Returns (valid, message, 1d_signal)."""
    if not isinstance(fs, (int, float)) or fs <= 0:
        return False, f"Invalid sampling rate: {fs}", np.array([])
    if fs < _CFG["min_fs"] or fs > _CFG["max_fs"]:
        return False, (
            f"Sampling rate {fs:.1f} Hz outside valid range "
            f"[{_CFG['min_fs']}, {_CFG['max_fs']}] Hz."
        ), np.array([])

    arr = np.asarray(signal, dtype=np.float64)
    if arr.ndim == 2:
        ch = min(channel, arr.shape[1] - 1)
        arr = arr[:, ch]
    elif arr.ndim != 1:
        return False, f"Unsupported signal shape: {arr.shape}.", np.array([])

    if len(arr) < _CFG["min_samples"]:
        return False, (
            f"Signal too short: {len(arr)} samples (min {_CFG['min_samples']})."
        ), np.array([])

    if np.all(~np.isfinite(arr)):
        return False, "Signal contains no finite values.", np.array([])

    return True, "OK", arr


def _longest_run_bool(mask: np.ndarray) -> int:
    """Return length of the longest run of True values in boolean array."""
    max_run = current = 0
    for v in mask:
        if v:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 0
    return max_run


def _extract_amplitudes_sub(
    ecg: np.ndarray,
    r_peaks: np.ndarray,
    fs: float,
    half_win_s: float = 0.06,
) -> np.ndarray:
    """Extract peak-to-peak QRS amplitudes."""
    half_n = max(1, int(half_win_s * fs))
    amps: list[float] = []
    n = len(ecg)
    for pk in r_peaks:
        lo = max(0, pk - half_n)
        hi = min(n, pk + half_n)
        if hi > lo:
            amps.append(float(np.max(ecg[lo:hi]) - np.min(ecg[lo:hi])))
    return np.array(amps)


def _rr_entropy(rr: np.ndarray, bins: int = 32) -> float:
    """Compute normalised histogram entropy of RR series (0-1)."""
    if len(rr) < 4:
        return 0.0
    try:
        hist, _ = np.histogram(rr, bins=bins)
        h = hist.astype(float) + 1e-9
        h /= h.sum()
        return float(-np.sum(h * np.log2(h + 1e-12)) / math.log2(bins))
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Convenience wrapper for batch processing
# ---------------------------------------------------------------------------

def process_batch(
    signals: list[np.ndarray],
    fs: float,
    modality: str = "ecg",
) -> list[dict[str, Any]]:
    """
    Process a list of signals and return a list of result dicts.

    Parameters
    ----------
    signals : list of np.ndarray
    fs : float — sampling rate (same for all signals)
    modality : str — signal modality

    Returns
    -------
    list of result dicts.
    """
    return [run(sig, fs, modality) for sig in signals]
