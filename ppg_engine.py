"""
ppg_engine.py — Photoplethysmography (PPG) pulse detection and analysis.

DO NOT apply ECG R-peak detection algorithms directly to PPG.
PPG has different morphology: slower, broader peaks, no QRS complex.

Pipeline:
  PPG
   ↓ Validation
   ↓ Baseline correction
   ↓ Artifact detection
   ↓ Pulse peak detection
   ↓ Inter-pulse intervals
   ↓ Pulse rate
   ↓ Signal quality
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import signal as sp_signal
from scipy.ndimage import uniform_filter1d


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class PPGResult:
    """Result of PPG pulse detection and analysis."""
    pulse_peaks: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    inter_pulse_intervals_s: np.ndarray = field(default_factory=lambda: np.array([]))
    pulse_rate_bpm: float = 0.0
    pulse_rate_reliable: bool = False
    signal_quality: str = "UNKNOWN"
    artifacts_detected: bool = False
    n_pulses: int = 0
    duration_s: float = 0.0
    filtered_ppg: np.ndarray = field(default_factory=lambda: np.array([]))
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# PPG engine
# ---------------------------------------------------------------------------

class PPGEngine:
    """
    PPG pulse detection engine.

    PPG pulses are detected by finding peaks in the
    band-pass filtered (0.5–8 Hz) PPG signal.
    This is NOT the same as ECG QRS detection.

    Parameters
    ----------
    fs :
        Sampling rate in Hz.
    """

    # PPG-specific bandpass (much lower frequency than ECG)
    PPG_BP_LOW_HZ: float = 0.5
    PPG_BP_HIGH_HZ: float = 8.0
    PPG_BP_ORDER: int = 3

    # Physiological limits for human pulse rate
    MIN_PULSE_BPM: float = 25.0
    MAX_PULSE_BPM: float = 220.0

    def __init__(self, fs: float) -> None:
        self.fs = float(fs)
        self._min_distance = max(1, int(60.0 / self.MAX_PULSE_BPM * fs))
        self._build_filters()

    def _build_filters(self) -> None:
        """Build PPG-specific SOS filter."""
        nyq = self.fs / 2.0
        lo = max(1e-4, self.PPG_BP_LOW_HZ / nyq)
        hi = min(0.99, self.PPG_BP_HIGH_HZ / nyq)
        if lo >= hi:
            hi = min(0.99, lo * 2.0)
        self._sos = sp_signal.butter(
            self.PPG_BP_ORDER, [lo, hi], btype="band", output="sos"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, ppg: np.ndarray) -> PPGResult:
        """
        Detect pulse peaks and compute pulse rate from PPG signal.

        Parameters
        ----------
        ppg :
            PPG signal (n_samples,).

        Returns
        -------
        PPGResult
        """
        result = PPGResult()
        result.duration_s = len(ppg) / self.fs

        # Validate
        finite = ppg[np.isfinite(ppg)]
        if len(finite) < int(self.fs * 2):
            result.signal_quality = "POOR"
            result.warnings.append("PPG signal too short for reliable analysis.")
            return result

        if float(np.std(finite)) < 1e-6:
            result.signal_quality = "FLATLINE"
            result.warnings.append("PPG signal appears to be a flatline.")
            return result

        # Interpolate NaN
        arr = _ppg_interpolate_nans(ppg)

        # Baseline removal (wide median subtraction)
        w = max(3, int(1.5 * self.fs))
        if w % 2 == 0:
            w += 1
        baseline = uniform_filter1d(arr, size=w, mode="nearest")
        detrended = arr - baseline

        # Bandpass filter
        try:
            filtered = sp_signal.sosfiltfilt(self._sos, detrended)
        except Exception:
            filtered = detrended
        result.filtered_ppg = filtered

        # Artifact detection (high kurtosis on derivative)
        try:
            from scipy.stats import kurtosis
            diff_kurt = float(kurtosis(np.diff(filtered), fisher=True))
            result.artifacts_detected = diff_kurt > 10.0
        except Exception:
            result.artifacts_detected = False

        # Peak detection — PPG peaks are positive (systolic peaks)
        prominence_min = float(np.std(filtered)) * 0.3
        peaks, props = sp_signal.find_peaks(
            filtered,
            distance=self._min_distance,
            prominence=max(1e-9, prominence_min),
        )

        result.pulse_peaks = peaks
        result.n_pulses = len(peaks)

        if len(peaks) < 2:
            result.signal_quality = "POOR"
            result.pulse_rate_reliable = False
            result.warnings.append("Fewer than 2 PPG peaks detected.")
            return result

        # Inter-pulse intervals
        ipi_s = np.diff(peaks.astype(float)) / self.fs
        # Filter physiologically reasonable
        valid_ipi = ipi_s[
            (ipi_s >= 60.0 / self.MAX_PULSE_BPM) &
            (ipi_s <= 60.0 / self.MIN_PULSE_BPM)
        ]
        result.inter_pulse_intervals_s = ipi_s

        if len(valid_ipi) == 0:
            result.pulse_rate_reliable = False
            result.warnings.append("No physiologically valid inter-pulse intervals.")
            result.signal_quality = "POOR"
            return result

        result.pulse_rate_bpm = float(60.0 / np.median(valid_ipi))
        result.pulse_rate_reliable = True
        result.signal_quality = "POOR" if result.artifacts_detected else "GOOD"

        return result

    def compare_with_ecg(
        self,
        ppg_result: PPGResult,
        ecg_hr_bpm: float,
        tolerance_bpm: float = 10.0,
    ) -> dict[str, object]:
        """
        Compare PPG pulse rate with ECG heart rate.

        Parameters
        ----------
        ppg_result :
            PPG analysis result.
        ecg_hr_bpm :
            ECG-derived heart rate in BPM.
        tolerance_bpm :
            Agreement tolerance in BPM.

        Returns
        -------
        dict with keys: agreement (bool), difference_bpm (float), note (str).
        """
        if not ppg_result.pulse_rate_reliable or ecg_hr_bpm <= 0:
            return {
                "agreement": None,
                "difference_bpm": None,
                "note": "Comparison unavailable (unreliable HR or pulse rate).",
            }
        diff = abs(ppg_result.pulse_rate_bpm - ecg_hr_bpm)
        agree = diff <= tolerance_bpm
        return {
            "agreement": agree,
            "difference_bpm": round(diff, 1),
            "ecg_hr_bpm": round(ecg_hr_bpm, 1),
            "ppg_pulse_bpm": round(ppg_result.pulse_rate_bpm, 1),
            "note": (
                f"ECG HR and PPG pulse rate agree within {tolerance_bpm:.0f} BPM."
                if agree else
                f"Disagreement of {diff:.1f} BPM — possible artifact or arrhythmia."
            ),
        }


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _ppg_interpolate_nans(sig: np.ndarray) -> np.ndarray:
    """Linear interpolation of NaN/Inf values in PPG signal."""
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
