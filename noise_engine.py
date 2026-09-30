"""
noise_engine.py — Advanced noise characterization and adaptive cancellation.

Handles:
  * Baseline wander detection and removal.
  * EMG contamination: detection, severity classification, adaptive suppression.
  * Motion artifacts.
  * 50/60 Hz powerline interference (adaptive notch).
  * Changing noise characteristics over time.
  * Signal saturation and clipping regions.
  * Missing sample interpolation.
  * SNR estimation (before / after).
  * Dual-path output: QRS detection path + morphology-preservation path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
from scipy import signal as sp_signal
from scipy.ndimage import uniform_filter1d


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class EMGSeverity(str, Enum):
    """EMG contamination severity levels."""
    CLEAN = "CLEAN"
    MILD = "MILD EMG"
    MODERATE = "MODERATE EMG"
    SEVERE = "SEVERE EMG"


class NoiseType(str, Enum):
    """Dominant noise type in a segment."""
    CLEAN = "Clean"
    BASELINE = "Baseline wander"
    EMG = "EMG contamination"
    MOTION = "Motion artifact"
    POWERLINE = "Powerline interference"
    SATURATION = "Saturation"
    MIXED = "Mixed noise"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class NoiseProfile:
    """Noise characterization for a single segment or the full recording."""
    start_s: float = 0.0
    end_s: float = 0.0
    noise_type: NoiseType = NoiseType.CLEAN
    emg_severity: EMGSeverity = EMGSeverity.CLEAN
    emg_energy_ratio: float = 0.0       # HF energy / total energy
    baseline_drift_score: float = 0.0   # LF power fraction
    motion_score: float = 0.0           # kurtosis-based score
    powerline_50hz: bool = False
    powerline_60hz: bool = False
    is_saturated: bool = False
    has_missing: bool = False
    noise_rms: float = 0.0
    snr_db: float = 0.0
    message: str = ""


@dataclass
class DualPathOutput:
    """Result of dual-path filtering."""
    # QRS detection path (aggressive bandpass, optimized for peak detection)
    qrs_path: np.ndarray = field(default_factory=lambda: np.array([]))
    # Morphology-preservation path (lighter filtering, preserves ST segment)
    morph_path: np.ndarray = field(default_factory=lambda: np.array([]))
    # Baseline estimate
    baseline: np.ndarray = field(default_factory=lambda: np.array([]))
    # Applied filters log
    applied_notch_50: bool = False
    applied_notch_60: bool = False
    applied_emg_suppression: bool = False
    snr_before_db: Optional[float] = None
    snr_after_db: Optional[float] = None
    snr_improvement_db: Optional[float] = None
    noise_profiles: list[NoiseProfile] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main noise engine
# ---------------------------------------------------------------------------

class NoiseEngine:
    """
    Adaptive noise characterization and dual-path ECG filtering.

    Parameters
    ----------
    fs :
        Sampling rate in Hz.
    segment_duration_s :
        Duration of each analysis segment in seconds.
    """

    # Bandpass for QRS detection path (0.5–45 Hz, 4th-order Butterworth)
    QRS_BP_LOW_HZ: float = 0.5
    QRS_BP_HIGH_HZ: float = 45.0
    QRS_BP_ORDER: int = 4

    # Bandpass for morphology path (0.05–40 Hz, 2nd-order — gentler)
    MORPH_BP_LOW_HZ: float = 0.05
    MORPH_BP_HIGH_HZ: float = 40.0
    MORPH_BP_ORDER: int = 2

    # EMG thresholds (fraction of total power in >100 Hz band)
    EMG_MILD_THRESHOLD: float = 0.05
    EMG_MODERATE_THRESHOLD: float = 0.15
    EMG_SEVERE_THRESHOLD: float = 0.35

    # Baseline wander (fraction of power below 1 Hz)
    BASELINE_MODERATE: float = 0.15
    BASELINE_SEVERE: float = 0.40

    def __init__(self, fs: float, segment_duration_s: float = 5.0) -> None:
        self.fs = float(fs)
        self.seg_s = segment_duration_s
        self.seg_n = max(4, int(segment_duration_s * fs))
        self._build_filters()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, raw: np.ndarray) -> DualPathOutput:
        """
        Full noise characterization and dual-path filtering pipeline.

        Parameters
        ----------
        raw :
            Raw ECG signal (n_samples,). May contain NaN/Inf.

        Returns
        -------
        DualPathOutput
        """
        out = DualPathOutput()

        # 1. Interpolate NaN/Inf
        clean = _interpolate_nans(raw)

        # 2. SNR estimate BEFORE filtering
        out.snr_before_db = self._estimate_snr(clean)

        # 3. Segment-level noise profiling
        out.noise_profiles = self._profile_segments(clean)

        # 4. Detect powerline interference on full signal
        pl50 = self._detect_powerline(clean, 50.0)
        pl60 = self._detect_powerline(clean, 60.0)
        out.applied_notch_50 = pl50
        out.applied_notch_60 = pl60

        # 5. Dominant EMG severity across segments
        emg_severities = [p.emg_severity for p in out.noise_profiles]
        dominant_emg = self._dominant_emg(emg_severities)

        # 6. Apply adaptive notch if powerline detected
        notched = clean.copy()
        if pl50:
            b50, a50 = self._notch_50
            notched = sp_signal.filtfilt(b50, a50, notched)
        if pl60:
            b60, a60 = self._notch_60
            notched = sp_signal.filtfilt(b60, a60, notched)

        # 7. EMG suppression (adaptive, only when EMG is moderate/severe)
        suppressed = notched.copy()
        if dominant_emg in (EMGSeverity.MODERATE, EMGSeverity.SEVERE):
            suppressed = self._suppress_emg(notched, dominant_emg)
            out.applied_emg_suppression = True
            out.warnings.append(
                f"Adaptive EMG suppression applied ({dominant_emg.value})."
            )

        # 8. Baseline removal
        baseline = self._estimate_baseline(suppressed)
        detrended = suppressed - baseline
        out.baseline = baseline

        # 9. Dual-path filtering
        out.qrs_path = sp_signal.sosfiltfilt(self._sos_qrs, detrended)
        out.morph_path = sp_signal.sosfiltfilt(self._sos_morph, detrended)

        # 10. SNR estimate AFTER filtering
        out.snr_after_db = self._estimate_snr(out.morph_path)
        if out.snr_before_db is not None and out.snr_after_db is not None:
            out.snr_improvement_db = out.snr_after_db - out.snr_before_db

        return out

    def characterize_emg(
        self, segment: np.ndarray
    ) -> tuple[EMGSeverity, float]:
        """
        Characterize EMG severity in a single segment.

        Returns (severity, emg_energy_ratio).
        """
        finite = segment[np.isfinite(segment)]
        if len(finite) < 8:
            return EMGSeverity.CLEAN, 0.0

        nyq = self.fs / 2.0
        if nyq <= 100.0:
            # Cannot detect EMG above Nyquist; use kurtosis heuristic
            return self._emg_by_kurtosis(finite)

        # High-frequency energy ratio (>100 Hz)
        try:
            n_seg = min(512, len(finite))
            freqs, psd = sp_signal.welch(finite, fs=self.fs, nperseg=n_seg)
            total = float(np.sum(psd)) + 1e-30
            hf_mask = freqs > 100.0
            hf_energy = float(np.sum(psd[hf_mask])) / total
        except Exception:
            return self._emg_by_kurtosis(finite)

        if hf_energy >= self.EMG_SEVERE_THRESHOLD:
            return EMGSeverity.SEVERE, hf_energy
        if hf_energy >= self.EMG_MODERATE_THRESHOLD:
            return EMGSeverity.MODERATE, hf_energy
        if hf_energy >= self.EMG_MILD_THRESHOLD:
            return EMGSeverity.MILD, hf_energy
        return EMGSeverity.CLEAN, hf_energy

    # ------------------------------------------------------------------
    # Private — filter construction
    # ------------------------------------------------------------------

    def _build_filters(self) -> None:
        """Build all SOS / IIR filters once."""
        nyq = self.fs / 2.0

        # QRS detection bandpass
        lo = self.QRS_BP_LOW_HZ / nyq
        hi = min(0.995, self.QRS_BP_HIGH_HZ / nyq)
        self._sos_qrs = sp_signal.butter(
            self.QRS_BP_ORDER, [lo, hi], btype="band", output="sos"
        )

        # Morphology-preservation bandpass
        lo_m = max(1e-4, self.MORPH_BP_LOW_HZ / nyq)
        hi_m = min(0.995, self.MORPH_BP_HIGH_HZ / nyq)
        self._sos_morph = sp_signal.butter(
            self.MORPH_BP_ORDER, [lo_m, hi_m], btype="band", output="sos"
        )

        # Notch filters
        if nyq > 50.0:
            self._notch_50 = sp_signal.iirnotch(50.0, 30.0, self.fs)
        else:
            self._notch_50 = (np.array([1.0]), np.array([1.0]))

        if nyq > 60.0:
            self._notch_60 = sp_signal.iirnotch(60.0, 30.0, self.fs)
        else:
            self._notch_60 = (np.array([1.0]), np.array([1.0]))

    # ------------------------------------------------------------------
    # Private — segment profiling
    # ------------------------------------------------------------------

    def _profile_segments(self, sig: np.ndarray) -> list[NoiseProfile]:
        """Profile noise characteristics per segment."""
        n = len(sig)
        n_segs = max(1, math.ceil(n / self.seg_n))
        profiles: list[NoiseProfile] = []

        for i in range(n_segs):
            start = i * self.seg_n
            end = min(start + self.seg_n, n)
            seg = sig[start:end]
            start_s = start / self.fs
            end_s = end / self.fs
            profiles.append(self._profile_one(seg, start_s, end_s))

        return profiles

    def _profile_one(
        self, seg: np.ndarray, start_s: float, end_s: float
    ) -> NoiseProfile:
        """Profile a single segment."""
        p = NoiseProfile(start_s=start_s, end_s=end_s)
        finite = seg[np.isfinite(seg)]
        p.has_missing = len(finite) < len(seg) * 0.99

        if len(finite) < 4:
            p.noise_type = NoiseType.CLEAN
            p.has_missing = True
            return p

        # Saturation
        plo, phi = float(np.percentile(finite, 1)), float(np.percentile(finite, 99))
        amp = phi - plo
        if amp > 1e-9:
            margin = amp * 0.02
            sat_frac = float(np.sum((finite >= phi - margin) | (finite <= plo + margin))) / len(finite)
            p.is_saturated = sat_frac > 0.02

        # Baseline drift
        p.baseline_drift_score = self._baseline_power_fraction(finite)

        # EMG
        p.emg_severity, p.emg_energy_ratio = self.characterize_emg(finite)

        # Motion (kurtosis on diff)
        try:
            from scipy.stats import kurtosis
            p.motion_score = float(kurtosis(np.diff(finite), fisher=True))
        except Exception:
            p.motion_score = 0.0

        # Powerline
        p.powerline_50hz = self._detect_powerline(finite, 50.0)
        p.powerline_60hz = self._detect_powerline(finite, 60.0)

        # Noise RMS
        p.noise_rms = float(np.std(np.diff(finite))) / math.sqrt(2)

        # SNR
        p.snr_db = self._estimate_snr(finite)

        # Dominant type
        p.noise_type = self._classify_noise_type(p)

        return p

    def _classify_noise_type(self, p: NoiseProfile) -> NoiseType:
        """Determine dominant noise type from profile metrics."""
        issues: list[tuple[float, NoiseType]] = []
        if p.emg_severity == EMGSeverity.SEVERE:
            issues.append((3.0, NoiseType.EMG))
        elif p.emg_severity == EMGSeverity.MODERATE:
            issues.append((2.0, NoiseType.EMG))
        elif p.emg_severity == EMGSeverity.MILD:
            issues.append((1.0, NoiseType.EMG))
        if p.baseline_drift_score > self.BASELINE_SEVERE:
            issues.append((2.5, NoiseType.BASELINE))
        elif p.baseline_drift_score > self.BASELINE_MODERATE:
            issues.append((1.5, NoiseType.BASELINE))
        if p.motion_score > 15.0:
            issues.append((2.0, NoiseType.MOTION))
        if p.powerline_50hz or p.powerline_60hz:
            issues.append((1.0, NoiseType.POWERLINE))
        if p.is_saturated:
            issues.append((2.5, NoiseType.SATURATION))
        if not issues:
            return NoiseType.CLEAN
        if len(issues) > 2:
            return NoiseType.MIXED
        return max(issues, key=lambda x: x[0])[1]

    # ------------------------------------------------------------------
    # Private — noise suppression
    # ------------------------------------------------------------------

    def _suppress_emg(
        self, sig: np.ndarray, severity: EMGSeverity
    ) -> np.ndarray:
        """
        Adaptive EMG suppression via spectral gating (soft threshold on PSD).
        Does NOT apply aggressive low-pass that would destroy QRS.
        """
        if severity == EMGSeverity.SEVERE:
            # Moderate additional low-pass above QRS band
            cutoff = min(0.995, 80.0 / (self.fs / 2))
        else:
            cutoff = min(0.995, 100.0 / (self.fs / 2))

        if cutoff <= 0.01:
            return sig
        try:
            sos = sp_signal.butter(2, cutoff, btype="low", output="sos")
            return sp_signal.sosfiltfilt(sos, sig)
        except Exception:
            return sig

    def _estimate_baseline(self, sig: np.ndarray) -> np.ndarray:
        """
        Robust baseline estimate via wide moving average.
        Window = 600 ms to span at least one full RR cycle's flat regions.
        """
        w = int(max(0.6 * self.fs, 3))
        if w % 2 == 0:
            w += 1
        w = min(w, max(3, len(sig) - 1))
        return uniform_filter1d(sig.astype(np.float64), size=w, mode="nearest")

    # ------------------------------------------------------------------
    # Private — spectral analysis
    # ------------------------------------------------------------------

    def _detect_powerline(self, sig: np.ndarray, freq: float) -> bool:
        """Return True if strong powerline interference at `freq` Hz."""
        nyq = self.fs / 2.0
        if nyq <= freq or len(sig) < int(self.fs * 2):
            return False
        try:
            n_seg = min(2048, len(sig))
            freqs, psd = sp_signal.welch(sig, fs=self.fs, nperseg=n_seg)
            idx = int(np.argmin(np.abs(freqs - freq)))
            lo, hi = max(0, idx - 5), min(len(psd), idx + 6)
            mask = np.ones(len(psd), bool)
            mask[lo:hi] = False
            floor = float(np.median(psd[mask])) + 1e-30
            peak_db = 10 * math.log10(max(float(psd[idx]), 1e-30) / floor)
            return peak_db > 10.0
        except Exception:
            return False

    def _baseline_power_fraction(self, sig: np.ndarray) -> float:
        """Fraction of PSD below 1 Hz (indicates baseline wander)."""
        if len(sig) < 32:
            return 0.0
        try:
            n_seg = min(256, len(sig))
            freqs, psd = sp_signal.welch(sig, fs=self.fs, nperseg=n_seg)
            total = float(np.sum(psd)) + 1e-30
            low = float(np.sum(psd[freqs < 1.0]))
            return float(np.clip(low / total, 0.0, 1.0))
        except Exception:
            return 0.0

    def _estimate_snr(self, sig: np.ndarray) -> Optional[float]:
        """Estimate signal SNR in dB. Returns None if unreliable."""
        finite = sig[np.isfinite(sig)]
        if len(finite) < 20:
            return None
        amp = float(np.percentile(finite, 95) - np.percentile(finite, 5))
        noise_rms = float(np.std(np.diff(finite))) / math.sqrt(2)
        if noise_rms < 1e-12 or amp < 1e-9:
            return None
        snr = 20 * math.log10(amp / noise_rms)
        return float(np.clip(snr, -20.0, 60.0))

    def _emg_by_kurtosis(
        self, sig: np.ndarray
    ) -> tuple[EMGSeverity, float]:
        """Fallback EMG detection using kurtosis when fs < 200 Hz."""
        try:
            from scipy.stats import kurtosis
            k = float(kurtosis(np.diff(sig), fisher=True))
        except Exception:
            return EMGSeverity.CLEAN, 0.0
        # High kurtosis on diff ≈ impulsive EMG bursts
        ratio = float(np.clip(k / 20.0, 0.0, 1.0))
        if k > 20:
            return EMGSeverity.SEVERE, ratio
        if k > 10:
            return EMGSeverity.MODERATE, ratio
        if k > 5:
            return EMGSeverity.MILD, ratio
        return EMGSeverity.CLEAN, ratio

    @staticmethod
    def _dominant_emg(severities: list[EMGSeverity]) -> EMGSeverity:
        """Return the most severe EMG level seen across segments."""
        order = [EMGSeverity.CLEAN, EMGSeverity.MILD,
                 EMGSeverity.MODERATE, EMGSeverity.SEVERE]
        best = EMGSeverity.CLEAN
        for s in severities:
            if order.index(s) > order.index(best):
                best = s
        return best


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _interpolate_nans(sig: np.ndarray) -> np.ndarray:
    """Linear interpolation of NaN/Inf values."""
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
