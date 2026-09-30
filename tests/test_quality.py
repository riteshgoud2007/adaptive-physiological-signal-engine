"""
tests/test_quality.py — Unit tests for quality_engine.py
"""
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from quality_engine import QualityEngine, QualityLevel, SegmentStatus


FS = 360.0


def make_ecg(n=3600, fs=FS):
    """Simple synthetic-like ECG pattern for testing (NOT for demo)."""
    t = np.linspace(0, n / fs, n)
    return 0.5 * np.sin(2 * np.pi * 1.2 * t) + 0.1 * np.random.randn(n)


class TestQualityEngine:
    def setup_method(self):
        self.qe = QualityEngine(FS)

    def test_good_signal(self):
        ecg = make_ecg(3600)
        rpt = self.qe.assess_recording(ecg)
        # A relatively clean synthetic signal should not be CRITICAL
        assert rpt.overall != QualityLevel.CRITICAL
        assert rpt.overall_score > 0.0

    def test_flatline_detected(self):
        ecg = np.zeros(3600)
        rpt = self.qe.assess_recording(ecg)
        assert rpt.flatline is True
        assert rpt.overall == QualityLevel.CRITICAL

    def test_all_nan_detected(self):
        ecg = np.full(3600, np.nan)
        rpt = self.qe.assess_recording(ecg)
        assert rpt.overall == QualityLevel.CRITICAL

    def test_saturation_detected(self):
        ecg = make_ecg(3600)
        # Saturate 10% of samples
        n_sat = 360
        ecg[:n_sat] = 5.0  # clip high
        rpt = self.qe.assess_recording(ecg)
        assert rpt.saturation_fraction > 0.0

    def test_high_noise_detected(self):
        t = np.linspace(0, 10, 3600)
        ecg = 0.01 * np.sin(2 * np.pi * t) + 5 * np.random.randn(3600)  # noisy
        rpt = self.qe.assess_recording(ecg)
        assert rpt.noise_level in (QualityLevel.MODERATE, QualityLevel.POOR, QualityLevel.CRITICAL)

    def test_missing_samples_detected(self):
        ecg = make_ecg(3600)
        ecg[500:600] = np.nan  # 100 NaN samples
        rpt = self.qe.assess_recording(ecg)
        assert rpt.missing_fraction > 0.0

    def test_segment_monitor_stable(self):
        ecg = make_ecg(3600)
        monitor = self.qe.monitor_segments(ecg)
        assert len(monitor.segments) > 0
        # Clean signal should have at least some stable segments
        statuses = [s.status for s in monitor.segments]
        assert SegmentStatus.STABLE in statuses

    def test_quality_score_in_range(self):
        ecg = make_ecg(3600)
        rpt = self.qe.assess_recording(ecg)
        assert 0.0 <= rpt.overall_score <= 1.0

    def test_powerline_50hz_detection(self):
        fs = 500.0
        t = np.linspace(0, 10, int(10 * fs))
        ecg = 0.5 * np.sin(2 * np.pi * 1.2 * t) + 0.3 * np.random.randn(len(t))
        # Add strong 50 Hz component
        ecg += 1.5 * np.sin(2 * np.pi * 50 * t)
        qe = QualityEngine(fs)
        rpt = qe.assess_recording(ecg)
        assert rpt.powerline_50hz is True
