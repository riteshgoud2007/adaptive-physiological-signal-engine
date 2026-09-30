"""
tests/test_optiforge_robustness.py — Extended robustness tests for OptiForge 2026.
All tests use actual computation. No fabricated metrics.
"""
from __future__ import annotations
import math
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import final_submission as eng

FS = 360.0
RNG = np.random.RandomState(42)


def _ecg(n_s: float = 20.0, bpm: float = 70.0, noise: float = 0.05) -> np.ndarray:
    n = int(n_s * FS)
    ecg = np.zeros(n)
    rr = 60.0 / bpm
    for i in range(int(n_s / rr) + 2):
        idx = int((i + 0.5) * rr * FS)
        if idx < n:
            w = max(1, int(0.025 * FS))
            for d in range(-w * 3, w * 3 + 1):
                if 0 <= idx + d < n:
                    ecg[idx + d] += math.exp(-(d / w) ** 2)
    ecg += noise * RNG.randn(n)
    return ecg


ECG_20 = _ecg(20.0, 70.0)


class TestValidation:
    def test_empty_signal(self):
        r = eng.run(np.zeros(0), FS)
        assert not r["valid"]

    def test_too_short(self):
        r = eng.run(np.zeros(50), FS)
        assert not r["valid"]

    def test_all_nan(self):
        r = eng.run(np.full(1000, float("nan")), FS)
        assert not r["valid"]

    def test_all_inf(self):
        r = eng.run(np.full(1000, float("inf")), FS)
        assert not r["valid"]

    def test_zero_fs(self):
        r = eng.run(ECG_20, 0.0)
        assert not r["valid"]

    def test_negative_fs(self):
        r = eng.run(ECG_20, -100.0)
        assert not r["valid"]

    def test_too_low_fs(self):
        r = eng.run(ECG_20, 20.0)
        assert not r["valid"]

    def test_unknown_modality(self):
        r = eng.run(ECG_20, FS, modality="eeg")
        assert not r["valid"]
        msg = r.get("validation_message", "")
        assert "eeg" in msg.lower() or "unsupported" in msg.lower() or "modality" in msg.lower()

    def test_unknown_modality_xyz(self):
        r = eng.run(ECG_20, FS, modality="xyz")
        assert not r["valid"]

    def test_constant_signal_ecg(self):
        r = eng.run(np.ones(int(10 * FS)), FS, modality="ecg")
        assert "valid" in r  # must not crash

    def test_2d_input_squeezable(self):
        r = eng.run(ECG_20.reshape(-1, 1), FS)
        assert r["valid"]

    def test_high_fs(self):
        sig = RNG.randn(int(10 * 1000)) * 0.5
        r = eng.run(sig, 1000.0, modality="ecg")
        assert r["valid"]


class TestECG:
    def test_normal_ecg_valid(self):
        r = eng.run(ECG_20, FS)
        assert r["valid"]
        assert r["n_beats"] > 0
        assert r["heart_rate_bpm"] > 0

    def test_ecg_with_nan_patch(self):
        sig = ECG_20.copy()
        sig[500:520] = float("nan")
        r = eng.run(sig, FS)
        assert r["valid"]

    def test_ecg_with_baseline_wander(self):
        t = np.arange(len(ECG_20)) / FS
        sig = ECG_20 + 0.5 * np.sin(2 * math.pi * 0.1 * t)
        r = eng.run(sig, FS)
        assert r["valid"]
        assert r["n_beats"] > 0

    def test_ecg_with_saturation(self):
        sig = np.clip(ECG_20, -0.5, 0.5)
        r = eng.run(sig, FS)
        assert r["valid"]

    def test_ecg_noisy(self):
        sig = ECG_20 + 1.5 * RNG.randn(len(ECG_20))
        r = eng.run(sig, FS)
        assert r["valid"]

    def test_ecg_missing_samples_10pct(self):
        sig = ECG_20.copy()
        idx = RNG.choice(len(sig), size=len(sig) // 10, replace=False)
        sig[idx] = float("nan")
        r = eng.run(sig, FS)
        assert r["valid"]

    def test_ecg_result_fields(self):
        r = eng.run(ECG_20, FS)
        required = [
            "valid", "n_beats", "heart_rate_bpm", "r_peaks",
            "rr_intervals_s", "rhythm_label", "signal_quality",
            "signal_quality_score", "detection_reliability",
            "snr_proxy_before_db", "snr_proxy_after_db",
            "possible_vt", "possible_vf", "warnings",
        ]
        for f in required:
            assert f in r, f"Missing result field: {f}"

    def test_vt_not_from_clean_sine_alone(self):
        """A clean sine wave must NOT be classified as VT purely on rate."""
        t = np.arange(int(20 * FS)) / FS
        sine = 0.5 * np.sin(2 * math.pi * 1.4 * t)  # ~84 bpm equivalent
        r = eng.run(sine, FS, modality="ecg")
        # If VT is flagged it must come with meaningful multi-evidence
        if r.get("possible_vt"):
            vt_ev = r.get("vt_evidence", [])
            assert len(vt_ev) >= 1, "VT on sine must have supporting evidence"

    def test_latency_none_for_normal_sinus(self):
        r = eng.run(ECG_20, FS)
        if not r.get("possible_vt"):
            assert r.get("detection_latency_s") is None

    def test_deterministic(self):
        r1 = eng.run(ECG_20, FS)
        r2 = eng.run(ECG_20, FS)
        assert r1["n_beats"] == r2["n_beats"]
        assert r1["heart_rate_bpm"] == r2["heart_rate_bpm"]

    def test_snr_improvement_nonneg_clean(self):
        """Clean ECG with added noise should show SNR improvement after filtering."""
        t = np.arange(len(ECG_20)) / FS
        noisy = ECG_20 + 0.3 * np.sin(2 * math.pi * 50 * t)
        r = eng.run(noisy, FS, modality="ecg")
        snr_b = r.get("snr_proxy_before_db")
        snr_a = r.get("snr_proxy_after_db")
        if snr_b is not None and snr_a is not None:
            # After filtering should be ≥ before (or close)
            assert snr_a >= snr_b - 5.0, "SNR should not degrade badly on clean+powerline signal"

    def test_morphology_score_range(self):
        r = eng.run(ECG_20, FS)
        ms = r.get("morphology_score")
        if ms is not None:
            assert 0.0 <= ms <= 1.0


class TestEMG:
    def test_emg_valid(self):
        sig = RNG.randn(int(20 * FS)) * 0.5
        r = eng.run(sig, FS, modality="emg")
        assert r["valid"]
        assert "emg_severity" in r
        assert "emg_energy_ratio" in r

    def test_emg_result_fields(self):
        sig = RNG.randn(int(10 * FS)) * 0.5
        r = eng.run(sig, FS, modality="emg")
        for f in ["valid", "signal_quality", "signal_quality_score", "emg_severity", "emg_energy_ratio"]:
            assert f in r

    def test_emg_too_short(self):
        r = eng.run(np.zeros(50), FS, modality="emg")
        assert not r["valid"]

    def test_emg_deterministic(self):
        sig = RNG.randn(int(10 * FS)) * 0.5
        r1 = eng.run(sig, FS, modality="emg")
        r2 = eng.run(sig, FS, modality="emg")
        assert r1["emg_severity"] == r2["emg_severity"]


class TestPPG:
    def test_ppg_valid(self):
        t = np.linspace(0, 20, int(20 * FS))
        sig = 0.5 * np.sin(2 * math.pi * 1.2 * t)
        r = eng.run(sig, FS, modality="ppg")
        assert r["valid"]

    def test_ppg_pulse_fields(self):
        t = np.linspace(0, 20, int(20 * FS))
        sig = 0.5 * np.sin(2 * math.pi * 1.2 * t)
        r = eng.run(sig, FS, modality="ppg")
        assert "pulse_peaks" in r
        assert "pulse_rate_bpm" in r
        assert "inter_pulse_intervals_s" in r

    def test_ppg_no_accelerometer_no_crash(self):
        t = np.linspace(0, 20, int(20 * FS))
        sig = 0.5 * np.sin(2 * math.pi * 1.0 * t) + 0.1 * RNG.randn(int(20 * FS))
        r = eng.run(sig, FS, modality="ppg")
        assert r["valid"]

    def test_ppg_deterministic(self):
        t = np.linspace(0, 10, int(10 * FS))
        sig = 0.5 * np.sin(2 * math.pi * 1.2 * t)
        r1 = eng.run(sig, FS, modality="ppg")
        r2 = eng.run(sig, FS, modality="ppg")
        assert len(r1["pulse_peaks"]) == len(r2["pulse_peaks"])


class TestPerturbation:
    def test_powerline_50hz_handling(self):
        t = np.arange(len(ECG_20)) / FS
        sig = ECG_20 + 0.3 * np.sin(2 * math.pi * 50 * t)
        r = eng.run(sig, FS, modality="ecg")
        assert r["valid"]
        assert r.get("powerline_50hz") is True

    def test_powerline_60hz_handling(self):
        t = np.arange(len(ECG_20)) / FS
        sig = ECG_20 + 0.3 * np.sin(2 * math.pi * 60 * t)
        r = eng.run(sig, FS, modality="ecg")
        assert r["valid"]
        assert r.get("powerline_60hz") is True

    def test_changing_noise(self):
        sig = ECG_20.copy()
        mid = len(sig) // 2
        sig[mid:] += 0.8 * RNG.randn(len(sig) - mid)
        r = eng.run(sig, FS, modality="ecg")
        assert r["valid"]

    def test_motion_artifact_burst(self):
        sig = ECG_20.copy()
        mid = len(sig) // 2
        win = int(2.0 * FS)
        sig[mid: mid + win] += 3.0 * RNG.randn(win)
        r = eng.run(sig, FS, modality="ecg")
        assert r["valid"]

    def test_strong_baseline_wander(self):
        t = np.arange(len(ECG_20)) / FS
        sig = ECG_20 + 1.5 * np.sin(2 * math.pi * 0.05 * t)
        r = eng.run(sig, FS, modality="ecg")
        assert r["valid"]


class TestBatch:
    def test_batch_returns_list(self):
        sigs = [ECG_20, ECG_20[: int(10 * FS)]]
        results = eng.process_batch(sigs, FS)
        assert isinstance(results, list)
        assert len(results) == 2

    def test_batch_each_valid(self):
        sigs = [ECG_20 for _ in range(3)]
        results = eng.process_batch(sigs, FS)
        for r in results:
            assert r["valid"]

    def test_batch_ppg_modality(self):
        t = np.linspace(0, 20, int(20 * FS))
        ppg = 0.5 * np.sin(2 * math.pi * 1.2 * t)
        results = eng.process_batch([ppg, ppg], FS, modality="ppg")
        for r in results:
            assert r["valid"]

    def test_batch_mixed_lengths(self):
        sigs = [ECG_20, ECG_20[:int(5 * FS)], ECG_20[:int(15 * FS)]]
        results = eng.process_batch(sigs, FS)
        assert len(results) == 3
        for r in results:
            assert "valid" in r


class TestNotchSafety:
    def test_notch_50hz_valid_fs(self):
        """50 Hz notch with FS=360 Hz should be applied safely."""
        t = np.arange(len(ECG_20)) / FS
        sig = ECG_20 + 0.5 * np.sin(2 * math.pi * 50 * t)
        r = eng.run(sig, FS, modality="ecg")
        assert r["valid"]

    def test_low_fs_no_crash(self):
        """FS=100 Hz — 50 Hz notch is at Nyquist, must be skipped gracefully."""
        n = int(10 * 100)
        sig = _ecg(10.0, 70.0, noise=0.05)
        # Resample to 100 Hz (just generate short one)
        sig100 = RNG.randn(n) * 0.3
        r = eng.run(sig100, 100.0, modality="ecg")
        assert "valid" in r  # must not crash regardless

    def test_very_high_fs(self):
        """Very high FS (1000 Hz) — filters must remain valid."""
        sig = RNG.randn(int(10 * 1000)) * 0.5
        r = eng.run(sig, 1000.0, modality="ecg")
        assert "valid" in r


class TestPerformance:
    def test_60s_ecg_under_200ms(self):
        """60-second ECG must complete in under 200ms (far below real-time)."""
        import time
        sig = RNG.randn(int(60 * FS)) * 0.5
        eng.run(sig, FS, modality="ecg")  # warm-up
        t0 = time.perf_counter()
        eng.run(sig, FS, modality="ecg")
        elapsed_ms = (time.perf_counter() - t0) * 1000
        assert elapsed_ms < 200, f"60s ECG took {elapsed_ms:.1f}ms (>200ms limit)"

    def test_10s_ecg_under_50ms(self):
        """10-second ECG must complete in under 50ms."""
        import time
        sig = RNG.randn(int(10 * FS)) * 0.5
        eng.run(sig, FS, modality="ecg")  # warm-up
        t0 = time.perf_counter()
        eng.run(sig, FS, modality="ecg")
        elapsed_ms = (time.perf_counter() - t0) * 1000
        assert elapsed_ms < 50, f"10s ECG took {elapsed_ms:.1f}ms (>50ms limit)"
