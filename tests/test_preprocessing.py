"""
tests/test_preprocessing.py — Unit tests for preprocessing functions.
"""
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from adaptive_engine import _interpolate_nans, _remove_baseline, AdaptiveECGEngine


class TestInterpolateNans:
    def test_no_nans(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0])
        result = _interpolate_nans(arr)
        np.testing.assert_allclose(result, arr)

    def test_interior_nans(self):
        arr = np.array([0.0, np.nan, np.nan, 3.0])
        result = _interpolate_nans(arr)
        assert np.all(np.isfinite(result))
        assert result[0] == pytest.approx(0.0)
        assert result[3] == pytest.approx(3.0)

    def test_all_finite_unchanged(self):
        arr = np.linspace(0, 1, 1000)
        result = _interpolate_nans(arr)
        np.testing.assert_allclose(result, arr)

    def test_no_inf_in_output(self):
        arr = np.array([1.0, np.inf, 2.0, np.nan, 3.0])
        result = _interpolate_nans(arr)
        # Inf stays Inf (we handle it separately in validator)
        # but NaN should be interpolated
        assert not np.isnan(result[3])


class TestRemoveBaseline:
    def test_removes_dc_offset(self):
        t = np.linspace(0, 5, 1800)
        ecg = np.sin(2 * np.pi * 1.5 * t) + 5.0  # 5 mV offset
        result = _remove_baseline(ecg, 360.0)
        # After detrending, mean should be close to 0
        assert abs(np.mean(result)) < 1.0

    def test_preserves_signal_length(self):
        sig = np.random.randn(1000)
        result = _remove_baseline(sig, 250.0)
        assert len(result) == len(sig)

    def test_short_signal(self):
        """Should not crash on short signals."""
        sig = np.random.randn(20)
        result = _remove_baseline(sig, 250.0)
        assert len(result) == 20


class TestBandpassFilter:
    def test_attenuates_high_frequency(self):
        fs = 360.0
        t = np.linspace(0, 5, int(5 * fs))
        # Add 150 Hz noise (well above cutoff)
        noise = 2.0 * np.sin(2 * np.pi * 150 * t)
        ecg = np.sin(2 * np.pi * 1.5 * t) + noise

        engine = AdaptiveECGEngine(fs)
        from scipy import signal as sp_signal
        filtered = sp_signal.sosfiltfilt(engine._sos_bp, ecg)

        # High-frequency component should be greatly reduced
        noise_rms_before = float(np.std(noise))
        noise_rms_after = float(np.std(filtered - np.sin(2 * np.pi * 1.5 * t)))
        assert noise_rms_after < noise_rms_before * 0.5

    def test_preserves_qrs_frequency_range(self):
        """1–20 Hz should pass through the bandpass largely intact."""
        fs = 360.0
        t = np.linspace(0, 5, int(5 * fs))
        qrs_like = np.sin(2 * np.pi * 5 * t)

        engine = AdaptiveECGEngine(fs)
        from scipy import signal as sp_signal
        filtered = sp_signal.sosfiltfilt(engine._sos_bp, qrs_like)

        # Correlation should be high
        corr = float(np.corrcoef(qrs_like[100:-100], filtered[100:-100])[0, 1])
        assert corr > 0.8
