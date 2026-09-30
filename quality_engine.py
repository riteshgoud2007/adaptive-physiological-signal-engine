"""
quality_engine.py — ECG signal quality analysis and adaptive monitoring.

Provides:
  * Per-recording quality assessment.
  * Per-segment quality monitoring.
  * Change-point / distribution-shift detection.
  * Artifact classification (motion, saturation, flatline, baseline drift).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
from scipy import signal as sp_signal
from scipy.stats import kurtosis, entropy

from config import (
    SEGMENT_DURATION_S,
    SATURATION_PERCENTILE,
    SATURATION_THRESHOLD,
    FLATLINE_STD_THRESHOLD,
    CLIPPING_FRACTION_WARN,
    KL_SHIFT_THRESHOLD,
    CHANGE_CUSUM_THRESHOLD,
    CHANGE_CUSUM_DRIFT,
    NOTCH_50_HZ,
    NOTCH_60_HZ,
    NOTCH_POWER_THRESHOLD_DB,
)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class QualityLevel(str, Enum):
    GOOD = "GOOD"
    MODERATE = "MODERATE"
    POOR = "POOR"
    CRITICAL = "CRITICAL"


class SegmentStatus(str, Enum):
    STABLE = "Stable"
    MOTION_ARTIFACT = "Motion artifact"
    NOISE_INCREASED = "Noise increased"
    NOISE_CHANGED = "Noise profile changed"
    BASELINE_DRIFT = "Baseline drift increasing"
    SATURATION = "Signal saturation"
    MISSING_DATA = "Missing data"
    FLATLINE = "Flatline"
    WEAK_QRS = "Weak QRS"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class SignalQualityReport:
    """Comprehensive quality report for an ECG recording."""

    # Overall
    overall: QualityLevel = QualityLevel.GOOD
    overall_score: float = 1.0           # 0–1

    # Specific metrics
    noise_level: QualityLevel = QualityLevel.GOOD
    noise_rms: float = 0.0
    baseline_drift: QualityLevel = QualityLevel.GOOD
    saturation: QualityLevel = QualityLevel.GOOD
    saturation_fraction: float = 0.0
    missing_samples: QualityLevel = QualityLevel.GOOD
    missing_fraction: float = 0.0
    qrs_visibility: QualityLevel = QualityLevel.GOOD
    powerline_50hz: bool = False
    powerline_60hz: bool = False
    flatline: bool = False
    kurtosis_value: float = 0.0
    snr_db: float = 0.0
    messages: list[str] = field(default_factory=list)


@dataclass
class SegmentQuality:
    """Quality assessment for a single time segment."""

    segment_idx: int
    start_s: float
    end_s: float
    status: SegmentStatus
    noise_rms: float
    baseline_offset: float
    is_saturated: bool
    has_missing: bool
    qrs_strength: float          # relative, 0–1
    reliability: float           # 0–1
    message: str = ""


@dataclass
class AdaptiveMonitorReport:
    """Report from the adaptive signal monitor."""

    segments: list[SegmentQuality] = field(default_factory=list)
    change_points: list[float] = field(default_factory=list)  # times in seconds
    noise_profile_changes: list[float] = field(default_factory=list)
    sensor_change_detected: bool = False
    powerline_50hz: bool = False
    powerline_60hz: bool = False


# ---------------------------------------------------------------------------
# Main quality engine
# ---------------------------------------------------------------------------

class QualityEngine:
    """
    Analyses an ECG signal and produces quality reports.

    Parameters
    ----------
    fs :
        Sampling rate in Hz.
    segment_duration_s :
        Duration of each analysis segment in seconds.
    """

    def __init__(self, fs: float, segment_duration_s: float = SEGMENT_DURATION_S) -> None:
        self.fs = fs
        self.seg_s = segment_duration_s
        self.seg_n = max(1, int(segment_duration_s * fs))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assess_recording(self, ecg: np.ndarray) -> SignalQualityReport:
        """
        Produce a full quality report for the entire recording.

        Parameters
        ----------
        ecg :
            Raw ECG signal (n_samples,).
        """
        rpt = SignalQualityReport()

        clean = ecg[np.isfinite(ecg)]
        if len(clean) == 0:
            rpt.overall = QualityLevel.CRITICAL
            rpt.overall_score = 0.0
            rpt.messages.append("Signal contains no finite values.")
            return rpt

        # Missing samples
        nan_count = int(np.sum(~np.isfinite(ecg)))
        rpt.missing_fraction = nan_count / max(1, len(ecg))
        rpt.missing_samples = self._level_from_fraction(
            rpt.missing_fraction, 0.005, 0.05
        )
        if nan_count > 0:
            rpt.messages.append(
                f"{nan_count} missing/invalid samples ({rpt.missing_fraction*100:.1f}%)."
            )

        # Flatline
        rpt.flatline = bool(np.std(clean) < FLATLINE_STD_THRESHOLD)
        if rpt.flatline:
            rpt.overall = QualityLevel.CRITICAL
            rpt.overall_score = 0.0
            rpt.messages.append("Signal appears to be a flatline.")
            return rpt

        # Saturation / clipping
        rpt.saturation_fraction, rpt.saturation = self._assess_saturation(clean)
        if rpt.saturation in (QualityLevel.MODERATE, QualityLevel.POOR, QualityLevel.CRITICAL):
            rpt.messages.append(
                f"Saturation/clipping detected in {rpt.saturation_fraction*100:.1f}% of samples."
            )

        # Noise estimate (high-frequency RMS)
        rpt.noise_rms, rpt.noise_level = self._estimate_noise(clean)

        # Baseline drift
        rpt.baseline_drift = self._assess_baseline_drift(clean)

        # Kurtosis
        try:
            rpt.kurtosis_value = float(kurtosis(clean, fisher=True))
        except Exception:
            rpt.kurtosis_value = 0.0

        # SNR estimate
        rpt.snr_db = self._estimate_snr(clean)

        # QRS visibility
        rpt.qrs_visibility = self._assess_qrs_visibility(clean)

        # Power-line interference
        rpt.powerline_50hz = self._detect_powerline(clean, NOTCH_50_HZ)
        rpt.powerline_60hz = self._detect_powerline(clean, NOTCH_60_HZ)
        if rpt.powerline_50hz:
            rpt.messages.append("50 Hz power-line interference detected.")
        if rpt.powerline_60hz:
            rpt.messages.append("60 Hz power-line interference detected.")

        # Overall score
        rpt.overall_score, rpt.overall = self._compute_overall(rpt)
        return rpt

    def monitor_segments(
        self, ecg: np.ndarray
    ) -> AdaptiveMonitorReport:
        """
        Slide a window over the recording and classify each segment.

        Returns an AdaptiveMonitorReport with per-segment statuses and
        detected change points.
        """
        report = AdaptiveMonitorReport()
        n = len(ecg)
        n_segs = max(1, math.ceil(n / self.seg_n))

        prev_noise_rms: Optional[float] = None
        prev_hist: Optional[np.ndarray] = None

        noise_series: list[float] = []

        for i in range(n_segs):
            start = i * self.seg_n
            end = min(start + self.seg_n, n)
            seg = ecg[start:end]
            start_s = start / self.fs
            end_s = end / self.fs

            sq = self._assess_segment(
                seg, i, start_s, end_s, prev_noise_rms
            )
            report.segments.append(sq)
            noise_series.append(sq.noise_rms)

            # Noise profile change detection via histogram KL divergence
            if prev_hist is not None:
                curr_hist = self._signal_histogram(seg)
                kl = _kl_divergence(prev_hist, curr_hist)
                if kl > KL_SHIFT_THRESHOLD:
                    report.noise_profile_changes.append(start_s)
                    sq.status = SegmentStatus.NOISE_CHANGED
                    sq.message = f"Noise profile changed (KL={kl:.2f})"

            prev_hist = self._signal_histogram(seg)
            prev_noise_rms = sq.noise_rms

        # CUSUM change detection on noise series
        if len(noise_series) >= 4:
            cp_indices = _cusum_change_points(
                np.array(noise_series),
                threshold=CHANGE_CUSUM_THRESHOLD,
                drift=CHANGE_CUSUM_DRIFT,
            )
            for idx in cp_indices:
                if idx < len(report.segments):
                    report.change_points.append(report.segments[idx].start_s)

        # Power-line detection on full signal
        clean = ecg[np.isfinite(ecg)]
        report.powerline_50hz = self._detect_powerline(clean, NOTCH_50_HZ)
        report.powerline_60hz = self._detect_powerline(clean, NOTCH_60_HZ)

        # Sensor/device change: look for a sudden amplitude shift
        report.sensor_change_detected = self._detect_amplitude_jump(noise_series)

        return report

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _assess_segment(
        self,
        seg: np.ndarray,
        idx: int,
        start_s: float,
        end_s: float,
        prev_noise_rms: Optional[float],
    ) -> SegmentQuality:
        """Classify a single segment."""
        finite = seg[np.isfinite(seg)]
        has_missing = bool(len(finite) < len(seg) * 0.99)
        is_saturated = False
        baseline_offset = 0.0
        noise_rms = 0.0
        qrs_strength = 0.0
        status = SegmentStatus.STABLE
        msg = ""

        if len(finite) < 4:
            status = SegmentStatus.MISSING_DATA
            return SegmentQuality(
                idx, start_s, end_s, status, 0.0, 0.0,
                False, True, 0.0, 0.0, "Segment missing most data."
            )

        # Flatline check
        if np.std(finite) < FLATLINE_STD_THRESHOLD:
            return SegmentQuality(
                idx, start_s, end_s, SegmentStatus.FLATLINE, 0.0, float(np.mean(finite)),
                False, has_missing, 0.0, 0.0, "Flatline segment."
            )

        # Saturation
        p1, p99 = np.percentile(finite, [1, 99])
        amp_range = p99 - p1
        if amp_range > 0:
            sat_frac, sat_level = self._assess_saturation(finite)
            is_saturated = sat_level in (QualityLevel.POOR, QualityLevel.CRITICAL)

        # Baseline offset
        baseline_offset = float(np.median(finite))

        # Noise RMS (difference signal approximation)
        noise_rms = float(np.std(np.diff(finite))) / math.sqrt(2)

        # QRS strength (peak-to-peak vs noise)
        p5, p95 = np.percentile(finite, [5, 95])
        qrs_pp = p95 - p5
        qrs_strength = min(1.0, qrs_pp / (noise_rms * 6 + 1e-9))
        qrs_strength = float(np.clip(qrs_strength, 0.0, 1.0))

        # Motion artifact: kurtosis > threshold on diff signal
        try:
            diff_kurt = float(kurtosis(np.diff(finite), fisher=True))
            if diff_kurt > 10:
                status = SegmentStatus.MOTION_ARTIFACT
                msg = f"High kurtosis diff ({diff_kurt:.1f})"
        except Exception:
            diff_kurt = 0.0

        # Noise increase vs previous segment
        if prev_noise_rms is not None and prev_noise_rms > 1e-9:
            ratio = noise_rms / prev_noise_rms
            if ratio > 3.0 and status == SegmentStatus.STABLE:
                status = SegmentStatus.MOTION_ARTIFACT
                msg = f"Noise increased {ratio:.1f}×"
            elif ratio > 1.5 and status == SegmentStatus.STABLE:
                status = SegmentStatus.NOISE_INCREASED
                msg = f"Noise slightly increased ({ratio:.1f}×)"

        # Baseline drift: low-frequency power vs total
        if status == SegmentStatus.STABLE:
            drift_score = self._baseline_drift_score(finite)
            if drift_score > 0.4:
                status = SegmentStatus.BASELINE_DRIFT
                msg = f"Baseline drift score {drift_score:.2f}"

        if is_saturated and status == SegmentStatus.STABLE:
            status = SegmentStatus.SATURATION
            msg = "Signal clipping/saturation."

        if has_missing and status == SegmentStatus.STABLE:
            status = SegmentStatus.MISSING_DATA
            msg = "Missing samples in segment."

        if qrs_strength < 0.1 and status == SegmentStatus.STABLE:
            status = SegmentStatus.WEAK_QRS
            msg = "Low QRS amplitude."

        reliability = float(np.clip(
            qrs_strength * (0.5 + 0.5 * (1 - sat_frac if amp_range > 0 else 1.0))
            * (0.0 if is_saturated else 1.0)
            * (0.5 if has_missing else 1.0)
            * (0.3 if status == SegmentStatus.FLATLINE else 1.0),
            0.0, 1.0
        ))

        return SegmentQuality(
            segment_idx=idx,
            start_s=start_s,
            end_s=end_s,
            status=status,
            noise_rms=noise_rms,
            baseline_offset=baseline_offset,
            is_saturated=is_saturated,
            has_missing=has_missing,
            qrs_strength=qrs_strength,
            reliability=reliability,
            message=msg,
        )

    # ------------------------------------------------------------------ #
    # Metric helpers
    # ------------------------------------------------------------------ #

    def _assess_saturation(self, sig: np.ndarray) -> tuple[float, QualityLevel]:
        p1 = np.percentile(sig, 1)
        p99 = np.percentile(sig, SATURATION_PERCENTILE)
        amp = p99 - p1
        if amp < 1e-9:
            return 1.0, QualityLevel.CRITICAL
        # Fraction near extremes
        margin = amp * (1 - SATURATION_THRESHOLD)
        clipped = np.sum((sig >= p99 - margin) | (sig <= p1 + margin))
        frac = float(clipped) / len(sig)
        level = self._level_from_fraction(frac, CLIPPING_FRACTION_WARN, 0.05)
        return frac, level

    def _estimate_noise(
        self, sig: np.ndarray
    ) -> tuple[float, QualityLevel]:
        """Estimate high-frequency noise RMS using difference approximation."""
        diff = np.diff(sig)
        rms = float(np.std(diff)) / math.sqrt(2)
        # Normalise by signal amplitude
        amp = float(np.percentile(sig, 95) - np.percentile(sig, 5))
        if amp < 1e-9:
            return rms, QualityLevel.CRITICAL
        rel = rms / amp
        level = self._level_from_fraction(rel, 0.05, 0.20)
        return rms, level

    def _assess_baseline_drift(self, sig: np.ndarray) -> QualityLevel:
        score = self._baseline_drift_score(sig)
        return self._level_from_fraction(score, 0.1, 0.35)

    def _baseline_drift_score(self, sig: np.ndarray) -> float:
        """Fraction of power in very low frequencies (< 1 Hz)."""
        if len(sig) < 32:
            return 0.0
        try:
            # Power below 1 Hz
            freqs, psd = sp_signal.welch(sig, fs=self.fs, nperseg=min(256, len(sig)))
            low_mask = freqs < 1.0
            total_power = float(np.sum(psd))
            if total_power < 1e-20:
                return 0.0
            low_power = float(np.sum(psd[low_mask]))
            return float(np.clip(low_power / total_power, 0.0, 1.0))
        except Exception:
            return 0.0

    def _estimate_snr(self, sig: np.ndarray) -> float:
        """Rough SNR estimate in dB."""
        if len(sig) < 10:
            return 0.0
        amp = float(np.percentile(sig, 95) - np.percentile(sig, 5))
        noise_rms, _ = self._estimate_noise(sig)
        if noise_rms < 1e-12:
            return 60.0
        snr = 20 * math.log10(max(amp, 1e-12) / max(noise_rms, 1e-12))
        return float(np.clip(snr, -20.0, 60.0))

    def _assess_qrs_visibility(self, sig: np.ndarray) -> QualityLevel:
        """Check if QRS complexes are likely visible above noise."""
        if len(sig) < int(self.fs * 0.5):
            return QualityLevel.POOR
        snr = self._estimate_snr(sig)
        if snr > 15:
            return QualityLevel.GOOD
        if snr > 8:
            return QualityLevel.MODERATE
        if snr > 2:
            return QualityLevel.POOR
        return QualityLevel.CRITICAL

    def _detect_powerline(self, sig: np.ndarray, freq: float) -> bool:
        """Return True if strong power-line interference at `freq` Hz is present."""
        if self.fs / 2 <= freq or len(sig) < int(self.fs * 2):
            return False
        try:
            freqs, psd = sp_signal.welch(
                sig, fs=self.fs, nperseg=min(2048, len(sig))
            )
            idx = int(np.argmin(np.abs(freqs - freq)))
            # Compare peak to surrounding noise floor
            lo = max(0, idx - 5)
            hi = min(len(psd), idx + 6)
            mask = np.ones(len(psd), bool)
            mask[lo:hi] = False
            noise_floor = float(np.median(psd[mask])) + 1e-30
            peak_db = 10 * math.log10(float(psd[idx]) / noise_floor)
            return peak_db > NOTCH_POWER_THRESHOLD_DB
        except Exception:
            return False

    def _detect_amplitude_jump(self, noise_series: list[float]) -> bool:
        """Heuristic: detect a sudden step in the noise/amplitude series."""
        if len(noise_series) < 4:
            return False
        arr = np.array(noise_series)
        diffs = np.abs(np.diff(arr))
        median_diff = float(np.median(diffs))
        if median_diff < 1e-9:
            return False
        return bool(np.any(diffs > 5 * median_diff))

    def _signal_histogram(
        self, sig: np.ndarray, bins: int = 32
    ) -> np.ndarray:
        """Normalised histogram of finite signal values."""
        finite = sig[np.isfinite(sig)]
        if len(finite) == 0:
            return np.ones(bins) / bins
        hist, _ = np.histogram(finite, bins=bins)
        hist = hist.astype(float) + 1e-9
        return hist / hist.sum()

    @staticmethod
    def _level_from_fraction(
        frac: float, warn_thresh: float, poor_thresh: float
    ) -> QualityLevel:
        if frac < warn_thresh:
            return QualityLevel.GOOD
        if frac < poor_thresh:
            return QualityLevel.MODERATE
        if frac < poor_thresh * 3:
            return QualityLevel.POOR
        return QualityLevel.CRITICAL

    def _compute_overall(
        self, rpt: SignalQualityReport
    ) -> tuple[float, QualityLevel]:
        """Compute overall quality score from individual metrics."""
        level_score = {
            QualityLevel.GOOD: 1.0,
            QualityLevel.MODERATE: 0.65,
            QualityLevel.POOR: 0.35,
            QualityLevel.CRITICAL: 0.0,
        }
        weights = [
            (rpt.noise_level, 0.25),
            (rpt.baseline_drift, 0.15),
            (rpt.saturation, 0.20),
            (rpt.missing_samples, 0.15),
            (rpt.qrs_visibility, 0.25),
        ]
        score = sum(level_score[lvl] * w for lvl, w in weights)
        if rpt.flatline:
            score = 0.0
        score = float(np.clip(score, 0.0, 1.0))
        if score >= 0.75:
            level = QualityLevel.GOOD
        elif score >= 0.50:
            level = QualityLevel.MODERATE
        elif score >= 0.25:
            level = QualityLevel.POOR
        else:
            level = QualityLevel.CRITICAL
        return score, level


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Symmetric KL divergence between two histograms."""
    p = p + 1e-9
    q = q + 1e-9
    p /= p.sum()
    q /= q.sum()
    kl_pq = float(entropy(p, q))
    kl_qp = float(entropy(q, p))
    return (kl_pq + kl_qp) / 2.0


def _cusum_change_points(
    series: np.ndarray,
    threshold: float = CHANGE_CUSUM_THRESHOLD,
    drift: float = CHANGE_CUSUM_DRIFT,
) -> list[int]:
    """CUSUM change-point detection. Returns indices of change points."""
    mu = float(np.mean(series))
    sigma = float(np.std(series)) + 1e-9
    z = (series - mu) / sigma
    s_pos = np.zeros(len(z))
    s_neg = np.zeros(len(z))
    change_points = []
    for i in range(1, len(z)):
        s_pos[i] = max(0.0, s_pos[i - 1] + z[i] - drift)
        s_neg[i] = max(0.0, s_neg[i - 1] - z[i] - drift)
        if s_pos[i] > threshold or s_neg[i] > threshold:
            change_points.append(i)
            s_pos[i] = 0.0
            s_neg[i] = 0.0
    return change_points
