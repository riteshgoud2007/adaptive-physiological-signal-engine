"""
rhythm_engine.py — Feature extraction and rhythm classification for ECG.

Round 1: Clean rhythm classification.
Round 2: Rhythm classification under severe EMG perturbation.

Rhythm classes:
  NORMAL_SINUS  — Regular rhythm, 60-100 BPM
  SINUS_TACH    — Sinus tachycardia, >100 BPM
  SINUS_BRADY   — Sinus bradycardia, <60 BPM
  AFIB          — Atrial fibrillation: highly irregular RR
  POSSIBLE_VT   — Possible ventricular tachycardia
  POSSIBLE_VF   — Possible ventricular fibrillation
  IRREGULAR     — Irregular rhythm (not AF-specific)
  UNKNOWN       — Cannot classify reliably

Feature-based, lightweight, deterministic.
No neural networks, no external model files.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
from scipy import signal as sp_signal


class RhythmLabel(str, Enum):
    NORMAL_SINUS = "Normal Sinus Rhythm"
    SINUS_TACH = "Sinus Tachycardia"
    SINUS_BRADY = "Sinus Bradycardia"
    AFIB = "Possible Atrial Fibrillation"
    POSSIBLE_VT = "Possible Ventricular Tachycardia"
    POSSIBLE_VF = "Possible Ventricular Fibrillation"
    IRREGULAR = "Irregular Rhythm"
    UNKNOWN = "Unknown / Unclassifiable"


class ConfidenceLevel(str, Enum):
    HIGH = "HIGH"
    MODERATE = "MODERATE"
    LOW = "LOW"
    UNRELIABLE = "UNRELIABLE"


@dataclass
class RhythmFeatures:
    mean_rr_s: float = 0.0
    median_rr_s: float = 0.0
    std_rr_s: float = 0.0
    rmssd_ms: float = 0.0
    cv_rr: float = 0.0
    pnn50: float = 0.0
    heart_rate_bpm: float = 0.0
    min_rr_s: float = 0.0
    max_rr_s: float = 0.0
    n_rr_intervals: int = 0
    regularity_score: float = 0.0
    spectral_entropy: float = 0.0
    longest_pause_s: float = 0.0
    short_rr_fraction: float = 0.0
    sustained_rapid_fraction: float = 0.0
    qrs_amplitude_cv: float = 0.0


@dataclass
class RhythmClassification:
    label: RhythmLabel = RhythmLabel.UNKNOWN
    confidence: ConfidenceLevel = ConfidenceLevel.UNRELIABLE
    confidence_score: float = 0.0
    heart_rate_bpm: float = 0.0
    features: Optional[RhythmFeatures] = None
    possible_vt: bool = False
    possible_vf: bool = False
    possible_vt_evidence: list[str] = field(default_factory=list)
    possible_vf_evidence: list[str] = field(default_factory=list)
    signal_quality_used: str = "UNKNOWN"
    n_beats: int = 0
    reliability_note: str = ""
    warnings: list[str] = field(default_factory=list)


class FeatureExtractor:
    """Extract rhythm features from R-peak indices and ECG."""

    MIN_BEATS: int = 4

    def __init__(self, fs: float) -> None:
        self.fs = float(fs)

    def extract(
        self,
        r_peaks: np.ndarray,
        ecg: Optional[np.ndarray] = None,
    ) -> Optional[RhythmFeatures]:
        """Extract rhythm features from R-peak indices."""
        if len(r_peaks) < self.MIN_BEATS:
            return None

        times_s = r_peaks.astype(float) / self.fs
        rr = np.diff(times_s)
        valid = rr[(rr >= 0.2) & (rr <= 3.0)]
        if len(valid) < 2:
            return None

        f = RhythmFeatures()
        f.n_rr_intervals = len(valid)
        f.mean_rr_s = float(np.mean(valid))
        f.median_rr_s = float(np.median(valid))
        f.std_rr_s = float(np.std(valid))
        f.min_rr_s = float(np.min(valid))
        f.max_rr_s = float(np.max(valid))
        f.heart_rate_bpm = 60.0 / max(f.median_rr_s, 1e-9)
        f.cv_rr = f.std_rr_s / max(f.mean_rr_s, 1e-9)

        ssd = np.diff(valid) * 1000.0
        f.rmssd_ms = float(np.sqrt(np.mean(ssd ** 2))) if len(ssd) > 0 else 0.0
        f.pnn50 = float(np.mean(np.abs(ssd) > 50.0)) if len(ssd) > 0 else 0.0
        f.regularity_score = float(np.clip(1.0 - f.cv_rr * 3.0, 0.0, 1.0))
        f.spectral_entropy = _rr_spectral_entropy(valid)
        f.longest_pause_s = float(np.max(valid))
        f.short_rr_fraction = float(np.mean(valid < 0.3))

        rapid = valid < 0.43
        f.sustained_rapid_fraction = _longest_run(rapid) / max(len(valid), 1)

        if ecg is not None and len(r_peaks) > 0:
            amps = _extract_amplitudes(ecg, r_peaks, self.fs)
            if len(amps) > 1:
                f.qrs_amplitude_cv = float(np.std(amps) / max(np.mean(amps), 1e-9))

        return f


class RhythmClassifier:
    """
    Deterministic feature-based rhythm classifier.

    Parameters
    ----------
    fs :
        Sampling rate in Hz.
    """

    HR_TACH_MIN: float = 100.0
    HR_BRADY_MAX: float = 60.0
    VT_HR_MIN: float = 120.0
    VT_SUSTAINED_FRAC: float = 0.75
    VT_REGULARITY_MIN: float = 0.6
    AF_CV_MIN: float = 0.10
    AF_PNN50_MIN: float = 0.15
    VF_ENTROPY_MIN: float = 0.65
    VF_AMPLITUDE_CV_MAX: float = 0.50
    MIN_BEATS_RELIABLE: int = 6

    def __init__(self, fs: float) -> None:
        self.fs = float(fs)
        self._extractor = FeatureExtractor(fs)

    def classify(
        self,
        r_peaks: np.ndarray,
        ecg: Optional[np.ndarray] = None,
        signal_quality: str = "UNKNOWN",
        emg_severity: str = "CLEAN",
    ) -> RhythmClassification:
        """
        Classify cardiac rhythm from R-peak indices.

        Parameters
        ----------
        r_peaks : np.ndarray
            Detected R-peak sample indices.
        ecg : np.ndarray, optional
            Filtered ECG (morphology path) for amplitude features.
        signal_quality : str
            Quality label ('GOOD', 'MODERATE', 'POOR', 'CRITICAL').
        emg_severity : str
            EMG severity string ('CLEAN', 'MILD EMG', 'MODERATE EMG', 'SEVERE EMG').

        Returns
        -------
        RhythmClassification
        """
        result = RhythmClassification()
        result.n_beats = len(r_peaks)
        result.signal_quality_used = signal_quality

        poor_quality = signal_quality in ("POOR", "CRITICAL")
        severe_emg = "SEVERE" in emg_severity

        features = self._extractor.extract(r_peaks, ecg)
        result.features = features

        if features is None:
            result.label = RhythmLabel.UNKNOWN
            result.confidence = ConfidenceLevel.UNRELIABLE
            result.reliability_note = (
                f"Insufficient beats for classification ({len(r_peaks)} detected)."
            )
            return result

        result.heart_rate_bpm = features.heart_rate_bpm

        vf_evidence, vf_score = self._check_vf(features, ecg, signal_quality)
        if vf_evidence:
            result.possible_vf = True
            result.possible_vf_evidence = vf_evidence

        vt_evidence, vt_score = self._check_vt(features)
        if vt_evidence:
            result.possible_vt = True
            result.possible_vt_evidence = vt_evidence

        label, base_confidence = self._primary_label(features)
        result.label = label

        if severe_emg:
            base_confidence = max(0.0, base_confidence - 0.35)
            result.warnings.append(
                "Severe EMG detected — rhythm classification reliability reduced."
            )
        elif "MODERATE" in emg_severity:
            base_confidence = max(0.0, base_confidence - 0.15)

        if poor_quality:
            base_confidence = max(0.0, base_confidence - 0.25)
            result.warnings.append(
                "Poor signal quality — classification may be inaccurate."
            )

        if features.n_rr_intervals < self.MIN_BEATS_RELIABLE:
            base_confidence = max(0.0, base_confidence - 0.20)
            result.reliability_note = (
                f"Only {features.n_rr_intervals} RR intervals — more beats needed."
            )

        result.confidence_score = float(np.clip(base_confidence, 0.0, 1.0))
        result.confidence = _score_to_level(result.confidence_score)

        if result.possible_vf and vf_score > 0.5:
            result.label = RhythmLabel.POSSIBLE_VF
        elif result.possible_vt and vt_score > 0.5:
            result.label = RhythmLabel.POSSIBLE_VT

        return result

    def _primary_label(
        self, f: RhythmFeatures
    ) -> tuple[RhythmLabel, float]:
        hr = f.heart_rate_bpm
        cv = f.cv_rr
        pnn50 = f.pnn50
        reg = f.regularity_score

        if (
            cv >= self.AF_CV_MIN and
            pnn50 >= self.AF_PNN50_MIN and
            f.rmssd_ms > 40.0 and
            f.spectral_entropy > 0.4
        ):
            confidence = float(np.clip(
                0.5 + 0.5 * min(cv / 0.20, 1.0) + 0.2 * min(pnn50 / 0.30, 1.0)
                - 0.3 * (hr > 150), 0.0, 1.0
            ))
            return RhythmLabel.AFIB, confidence

        if hr >= self.HR_TACH_MIN:
            return RhythmLabel.SINUS_TACH, 0.7 + 0.3 * reg

        if hr < self.HR_BRADY_MAX:
            return RhythmLabel.SINUS_BRADY, 0.7 + 0.3 * reg

        if reg > 0.5:
            return RhythmLabel.NORMAL_SINUS, 0.6 + 0.4 * reg

        if cv > 0.05:
            return RhythmLabel.IRREGULAR, 0.5

        return RhythmLabel.NORMAL_SINUS, 0.65

    def _check_vt(
        self, f: RhythmFeatures
    ) -> tuple[list[str], float]:
        evidence: list[str] = []
        score = 0.0

        if f.heart_rate_bpm >= self.VT_HR_MIN:
            evidence.append(f"Rapid rate: {f.heart_rate_bpm:.0f} BPM")
            score += 0.4

        if f.sustained_rapid_fraction >= self.VT_SUSTAINED_FRAC:
            evidence.append(
                f"Sustained rapid beats: {f.sustained_rapid_fraction*100:.0f}% of beats"
            )
            score += 0.4

        if f.regularity_score >= self.VT_REGULARITY_MIN and score > 0:
            evidence.append(f"Regular rhythm (score={f.regularity_score:.2f})")
            score += 0.2

        return evidence, float(np.clip(score, 0.0, 1.0))

    def _check_vf(
        self,
        f: RhythmFeatures,
        ecg: Optional[np.ndarray],
        signal_quality: str,
    ) -> tuple[list[str], float]:
        evidence: list[str] = []
        score = 0.0

        if f.spectral_entropy > self.VF_ENTROPY_MIN:
            evidence.append(f"High RR spectral entropy: {f.spectral_entropy:.2f}")
            score += 0.3

        if f.cv_rr > 0.25:
            evidence.append(f"Highly irregular RR (CV={f.cv_rr:.2f})")
            score += 0.2

        if f.qrs_amplitude_cv > self.VF_AMPLITUDE_CV_MAX:
            evidence.append(f"High QRS amplitude variation (CV={f.qrs_amplitude_cv:.2f})")
            score += 0.2

        if signal_quality in ("POOR", "CRITICAL") and f.heart_rate_bpm > 150:
            evidence.append("Poor quality at high rate — possible VF-like pattern")
            score += 0.1

        return evidence, float(np.clip(score, 0.0, 1.0))


def _rr_spectral_entropy(rr: np.ndarray, bins: int = 32) -> float:
    """Compute spectral entropy of RR interval series."""
    if len(rr) < 4:
        return 0.0
    try:
        hist, _ = np.histogram(rr, bins=bins)
        hist = hist.astype(float) + 1e-9
        hist /= hist.sum()
        return float(-np.sum(hist * np.log2(hist + 1e-12)) / math.log2(bins))
    except Exception:
        return 0.0


def _longest_run(mask: np.ndarray) -> int:
    """Return length of the longest run of True values."""
    if not mask.any():
        return 0
    max_run = current = 0
    for v in mask:
        if v:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 0
    return max_run


def _extract_amplitudes(
    ecg: np.ndarray,
    r_peaks: np.ndarray,
    fs: float,
    half_win_s: float = 0.06,
) -> np.ndarray:
    """Extract peak-to-peak amplitudes at R-peak positions."""
    half_n = max(1, int(half_win_s * fs))
    amps: list[float] = []
    n = len(ecg)
    for pk in r_peaks:
        lo = max(0, pk - half_n)
        hi = min(n, pk + half_n)
        if hi > lo:
            amps.append(float(np.max(ecg[lo:hi]) - np.min(ecg[lo:hi])))
    return np.array(amps)


def _score_to_level(score: float) -> ConfidenceLevel:
    """Map 0-1 score to ConfidenceLevel."""
    if score >= 0.70:
        return ConfidenceLevel.HIGH
    if score >= 0.45:
        return ConfidenceLevel.MODERATE
    if score >= 0.20:
        return ConfidenceLevel.LOW
    return ConfidenceLevel.UNRELIABLE
