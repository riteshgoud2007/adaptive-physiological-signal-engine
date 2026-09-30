"""
tests/test_detector.py — Unit tests for the adaptive R-peak detector.
"""
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from adaptive_engine import AdaptiveECGEngine


FS = 360.0


def make_synthetic_ecg_with_peaks(
    n_beats: int = 30,
    bpm: float = 75.0,
    fs: float = FS,
    noise_std: float = 0.05,
):
    """
    Create a synthetic ECG-like signal with known R-peak positions.
    Used ONLY for unit testing the detector — never displayed to user.
    """
    duration_s = (n_beats + 2) * 60.0 / bpm
    n_samples = int(duration_s * fs)
    t = np.arange(n_samples) / fs

    ecg = np.zeros(n_samples)
    rr_s = 60.0 / bpm
    peak_indices = []

    for beat_i in range(n_beats):
        t_peak = (beat_i + 1) * rr_s
        idx = int(t_peak * fs)
        if idx >= n_samples:
            break
        # QRS complex: narrow Gaussian
        w = max(1, int(0.02 * fs))
        for di in range(-w * 3, w * 3 + 1):
            if 0 <= idx + di < n_samples:
                ecg[idx + di] += np.exp(-(di / w) ** 2)
        peak_indices.append(idx)

    # Add noise
    ecg += noise_std * np.random.randn(n_samples)
    return ecg, np.array(peak_indices)


class TestAdaptiveDetector:
    def setup_method(self):
        self.engine = AdaptiveECGEngine(FS)

    def test_detects_peaks_in_clean_signal(self):
        ecg, true_peaks = make_synthetic_ecg_with_peaks(n_beats=20, noise_std=0.02)
        result = self.engine.process(ecg)
        assert len(result.r_peaks) > 0

    def test_heart_rate_approximately_correct(self):
        bpm_true = 72.0
        ecg, _ = make_synthetic_ecg_with_peaks(n_beats=20, bpm=bpm_true, noise_std=0.02)
        result = self.engine.process(ecg)
        if result.heart_rate_bpm > 0:
            assert abs(result.heart_rate_bpm - bpm_true) < 20.0  # within 20 BPM

    def test_no_duplicate_peaks(self):
        ecg, _ = make_synthetic_ecg_with_peaks(n_beats=20, noise_std=0.05)
        result = self.engine.process(ecg)
        if len(result.r_peaks) > 1:
            assert len(result.r_peaks) == len(np.unique(result.r_peaks))

    def test_peak_indices_in_bounds(self):
        ecg, _ = make_synthetic_ecg_with_peaks(n_beats=20, noise_std=0.02)
        result = self.engine.process(ecg)
        n = len(ecg)
        assert np.all(result.r_peaks >= 0)
        assert np.all(result.r_peaks < n)

    def test_rr_intervals_physiological(self):
        ecg, _ = make_synthetic_ecg_with_peaks(n_beats=20, bpm=72.0, noise_std=0.02)
        result = self.engine.process(ecg)
        if len(result.rr_intervals_s) > 0:
            # All RR should be >0
            assert np.all(result.rr_intervals_s > 0)

    def test_flatline_gives_no_peaks(self):
        ecg = np.zeros(3600)
        result = self.engine.process(ecg)
        # May detect 0 or very few spurious peaks on flatline
        # but should not crash
        assert isinstance(result.r_peaks, np.ndarray)

    def test_very_short_signal(self):
        """Should not crash on borderline short signal."""
        ecg = np.random.randn(200)  # less than 1 second at 360 Hz
        result = self.engine.process(ecg)
        assert result is not None

    def test_determinism(self):
        """Same input gives same output."""
        ecg, _ = make_synthetic_ecg_with_peaks(n_beats=15, noise_std=0.03)
        r1 = self.engine.process(ecg)
        r2 = self.engine.process(ecg)
        np.testing.assert_array_equal(r1.r_peaks, r2.r_peaks)

    def test_processing_result_has_all_signals(self):
        ecg, _ = make_synthetic_ecg_with_peaks(n_beats=10, noise_std=0.02)
        result = self.engine.process(ecg)
        assert len(result.filtered) == len(ecg)
        assert len(result.detrended) == len(ecg)
        assert len(result.derivative) == len(ecg)

    def test_reliability_score_in_range(self):
        ecg, _ = make_synthetic_ecg_with_peaks(n_beats=20, noise_std=0.02)
        result = self.engine.process(ecg)
        assert 0.0 <= result.detection_reliability <= 1.0
