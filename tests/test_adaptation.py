"""
tests/test_adaptation.py — Unit tests for adaptive monitoring and change detection.
"""
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from quality_engine import QualityEngine, SegmentStatus
from adaptive_engine import AdaptiveECGEngine


FS = 360.0


def make_ecg_segment(duration_s: float, noise_std: float = 0.05, fs: float = FS) -> np.ndarray:
    t = np.linspace(0, duration_s, int(duration_s * fs))
    return 0.5 * np.sin(2 * np.pi * 1.2 * t) + noise_std * np.random.randn(len(t))


class TestAdaptiveMonitoring:
    def setup_method(self):
        self.qe = QualityEngine(FS)
        self.engine = AdaptiveECGEngine(FS)

    def test_motion_artifact_detected(self):
        """High noise segment should not be classified as STABLE."""
        clean = make_ecg_segment(10.0, noise_std=0.02)
        noisy = make_ecg_segment(10.0, noise_std=3.0)  # very high noise
        ecg = np.concatenate([clean, noisy, clean])
        monitor = self.qe.monitor_segments(ecg)
        statuses = [s.status for s in monitor.segments]
        non_stable = [s for s in statuses if s != SegmentStatus.STABLE]
        assert len(non_stable) > 0, "Expected at least one non-stable segment"

    def test_noise_change_detected_in_noisy_signal(self):
        """Sudden large noise increase should produce noise_profile_changes or CUSUM change."""
        clean = make_ecg_segment(20.0, noise_std=0.02)
        noisy = make_ecg_segment(20.0, noise_std=5.0)
        ecg = np.concatenate([clean, noisy])
        monitor = self.qe.monitor_segments(ecg)
        # Either noise_profile_changes detected or sensor_change detected
        any_change = (
            len(monitor.noise_profile_changes) > 0
            or len(monitor.change_points) > 0
            or monitor.sensor_change_detected
        )
        assert any_change, "Expected at least one type of change detection"

    def test_segment_count_proportional_to_duration(self):
        """More segments for longer recordings."""
        short = make_ecg_segment(10.0)
        long = make_ecg_segment(30.0)
        m_short = self.qe.monitor_segments(short)
        m_long = self.qe.monitor_segments(long)
        assert len(m_long.segments) > len(m_short.segments)

    def test_flatline_segment_classified(self):
        """Flatline should be detected at segment level."""
        ecg = np.zeros(int(30 * FS))
        monitor = self.qe.monitor_segments(ecg)
        flatline_segs = [s for s in monitor.segments if s.status == SegmentStatus.FLATLINE]
        assert len(flatline_segs) > 0

    def test_missing_data_segment_classified(self):
        """Segment with mostly NaNs should be MISSING_DATA."""
        ecg = make_ecg_segment(30.0)
        # Inject a block of NaNs spanning a full 5-second segment (1800 samples at 360 Hz)
        seg_n = int(5.0 * FS)
        # Replace the second segment entirely with NaN
        ecg[seg_n: seg_n * 2] = np.nan
        monitor = self.qe.monitor_segments(ecg)
        missing_segs = [s for s in monitor.segments if s.status == SegmentStatus.MISSING_DATA]
        # At least one segment should be MISSING_DATA or have has_missing=True
        has_missing_any = any(s.has_missing for s in monitor.segments)
        assert len(missing_segs) > 0 or has_missing_any, \
            "Expected at least one segment with missing data"

    def test_reliability_per_segment_in_range(self):
        ecg = make_ecg_segment(30.0)
        monitor = self.qe.monitor_segments(ecg)
        for seg in monitor.segments:
            assert 0.0 <= seg.reliability <= 1.0

    def test_chunk_boundary_no_duplicate_peaks(self):
        """Processing a long signal in chunks should produce no duplicate peaks."""
        # Create a long signal (> 60 seconds to trigger chunking)
        duration_s = 90.0
        fs = FS
        t = np.linspace(0, duration_s, int(duration_s * fs))
        # Rhythmic signal
        ecg = 0.5 * np.sin(2 * np.pi * 1.2 * t) + 0.05 * np.random.randn(len(t))
        result = self.engine.process(ecg)
        if len(result.r_peaks) > 1:
            assert len(result.r_peaks) == len(np.unique(result.r_peaks)), \
                "Duplicate peaks found at chunk boundaries"

    def test_adaptive_processing_completes_on_mixed_signal(self):
        """Engine should complete without error on mixed-quality signal."""
        clean = make_ecg_segment(10.0, noise_std=0.02)
        noisy = make_ecg_segment(5.0, noise_std=2.0)
        flat = np.zeros(int(5.0 * FS))
        mixed = np.concatenate([clean, noisy, flat, clean])
        result = self.engine.process(mixed)
        assert result is not None
        assert isinstance(result.r_peaks, np.ndarray)


class TestEvaluationMetrics:
    def test_perfect_detection_metrics(self):
        from evaluation import evaluate_against_ground_truth
        peaks = np.array([100, 200, 300, 400, 500])
        result = evaluate_against_ground_truth(peaks, peaks, FS, tolerance_s=0.05)
        assert result.tp == 5
        assert result.fp == 0
        assert result.fn == 0
        assert result.precision == pytest.approx(1.0)
        assert result.recall == pytest.approx(1.0)
        assert result.f1 == pytest.approx(1.0)

    def test_all_false_positives(self):
        from evaluation import evaluate_against_ground_truth
        detected = np.array([50, 150, 250])  # none match reference
        reference = np.array([100, 200, 300, 400, 500])
        result = evaluate_against_ground_truth(detected, reference, FS, tolerance_s=0.01)
        assert result.tp == 0
        assert result.fp == 3
        assert result.fn == 5

    def test_empty_detection(self):
        from evaluation import evaluate_against_ground_truth
        result = evaluate_against_ground_truth(
            np.array([]), np.array([100, 200, 300]), FS
        )
        assert result.tp == 0
        assert result.precision == pytest.approx(0.0, abs=1e-9)
        assert result.recall == pytest.approx(0.0, abs=1e-9)
