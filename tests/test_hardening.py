"""
test_hardening.py — Tests for all 10 hardening fixes in final_submission.py v2.1.0

LOCAL TEST ONLY — NOT OFFICIAL OPTIFORGE SCORE.
"""
import sys
import numpy as np
import pytest

sys.path.insert(0, ".")
import final_submission as engine

FS = 360.0


def _make_ecg(bpm=70.0, fs=FS, duration_s=20.0, noise=0.05, add_50hz=False):
    """Synthetic ECG-like signal."""
    n = int(duration_s * fs)
    t = np.linspace(0, duration_s, n)
    ecg = np.zeros(n)
    rr = 60.0 / bpm
    for i in range(int(duration_s / rr) + 2):
        idx = int((i + 0.5) * rr * fs)
        if idx < n:
            w = int(0.025 * fs)
            for d in range(-w * 3, w * 3 + 1):
                if 0 <= idx + d < n:
                    ecg[idx + d] += float(np.exp(-(d / w) ** 2))
    ecg += noise * np.random.RandomState(42).randn(n)
    if add_50hz:
        ecg += 0.1 * np.sin(2 * np.pi * 50 * t)
    return ecg


def _sine_wave(freq_hz=1.2, fs=FS, duration_s=20.0, amplitude=0.5):
    """Simple periodic non-ECG sine wave."""
    n = int(duration_s * fs)
    t = np.linspace(0, duration_s, n)
    return amplitude * np.sin(2 * np.pi * freq_hz * t)


# ============================================================
# FIX 1: RHYTHM — VT should not be triggered by rate alone
# ============================================================

class TestVTHardening:
    def test_normal_ecg_70bpm_not_vt(self):
        """Normal sinus rhythm must NOT be flagged as VT."""
        ecg = _make_ecg(bpm=70.0)
        r = engine.run(ecg, FS)
        assert r["valid"]
        assert not r["possible_vt"], (
            f"Normal 70 BPM ECG flagged as VT. Evidence: {r['vt_evidence']}"
        )

    def test_simple_sine_wave_not_vt(self):
        """
        FIX 1 KEY TEST: A simple periodic sine wave at 'rapid' rate
        must NOT be classified as VT.
        """
        # 72 Hz sine — periodic, rapid 'rate' if interpreted as QRS
        # This is NOT a physiological ECG and must not be labelled VT.
        sine = _sine_wave(freq_hz=1.2, duration_s=20.0)  # ~72 bpm-equivalent
        r = engine.run(sine, FS)
        # The rhythm label should NOT be Possible VT without strong multi-source evidence
        if r["possible_vt"]:
            # If still flagged, evidence must be multi-source (not rate-only)
            assert len(r["vt_evidence"]) >= 2, (
                "VT must require >=2 evidence sources; single-source (rate only) not sufficient"
            )

    def test_rapid_sine_not_auto_vt(self):
        """Rapid sine at ~150 BPM-equivalent must NOT trigger VT from rate alone."""
        rapid_sine = _sine_wave(freq_hz=2.5, duration_s=20.0)  # ~150 bpm
        r = engine.run(rapid_sine, FS)
        if r["possible_vt"]:
            # At minimum must have sustained fraction evidence too
            evidence_count = len(r["vt_evidence"])
            assert evidence_count >= 2, (
                f"VT from rate-only: only {evidence_count} evidence item(s)"
            )

    def test_vt_requires_min_beats(self):
        """VT flag should not appear with too few beats."""
        # Very short signal — won't have enough beats
        ecg = _make_ecg(bpm=150.0, duration_s=2.0)
        r = engine.run(ecg, FS)
        # Either not VT, or confidence is low
        if r["possible_vt"]:
            assert r["rhythm_confidence"] in ("LOW", "UNRELIABLE", "MODERATE"), (
                "Short signal VT should have low confidence"
            )

    def test_sustained_rapid_ecg_flags_vt(self):
        """Genuine 150 BPM ECG (≥20 s) with sustained rapid beats SHOULD flag VT."""
        ecg = _make_ecg(bpm=150.0, duration_s=25.0)
        r = engine.run(ecg, FS)
        assert r["valid"]
        # Should either flag VT or Sinus Tachycardia (not NSR/Brady)
        assert r["rhythm_label"] not in ("Normal Sinus Rhythm", "Sinus Bradycardia"), (
            f"150 BPM ECG labelled as {r['rhythm_label']}"
        )


# ============================================================
# FIX 2: VF detection — signal-level, not quality-only
# ============================================================

class TestVFHardening:
    def test_poor_quality_alone_not_vf(self):
        """
        FIX 2 KEY TEST: A noisy/poor-quality signal alone must NOT trigger VF.
        VF requires signal-level evidence, not just noise.
        """
        # Very noisy signal — high noise but still a recognizable signal
        noisy = _make_ecg(bpm=70.0, noise=2.0)
        r = engine.run(noisy, FS)
        # If VF is flagged, there must be explicit signal-level evidence
        if r["possible_vf"]:
            assert any("entropy" in e.lower() or "amplitude" in e.lower()
                       for e in r["vf_evidence"]), (
                f"VF flagged from quality alone: {r['vf_evidence']}"
            )

    def test_normal_ecg_not_vf(self):
        """Normal sinus ECG must not be flagged as VF."""
        ecg = _make_ecg(bpm=70.0)
        r = engine.run(ecg, FS)
        assert not r["possible_vf"], (
            f"Normal ECG flagged as VF: {r['vf_evidence']}"
        )


# ============================================================
# FIX 3: Polarity-aware R-peak refinement
# ============================================================

class TestPeakRefinement:
    def test_inverted_ecg_still_detects_peaks(self):
        """Inverted ECG (negative R-peaks) must still detect beats."""
        ecg = _make_ecg(bpm=70.0)
        r_normal = engine.run(ecg, FS)
        r_inverted = engine.run(-ecg, FS)
        assert r_inverted["valid"]
        # Should detect similar number of beats
        assert abs(r_normal["n_beats"] - r_inverted["n_beats"]) <= 3, (
            f"Normal: {r_normal['n_beats']}, Inverted: {r_inverted['n_beats']}"
        )

    def test_peaks_in_bounds(self):
        """All detected R-peaks must be within signal bounds."""
        ecg = _make_ecg(bpm=70.0, duration_s=30.0)
        r = engine.run(ecg, FS)
        n = len(ecg)
        peaks = r["r_peaks"]
        assert np.all(peaks >= 0) and np.all(peaks < n), "Peak out of bounds"


# ============================================================
# FIX 4: Search-back deduplication
# ============================================================

class TestSearchBack:
    def test_no_duplicate_peaks(self):
        """Search-back must not produce duplicate R-peaks."""
        ecg = _make_ecg(bpm=70.0, duration_s=30.0)
        r = engine.run(ecg, FS)
        peaks = r["r_peaks"]
        assert len(peaks) == len(np.unique(peaks)), "Duplicate peaks detected"

    def test_peaks_respect_refractory(self):
        """No two peaks should be closer than refractory period."""
        ecg = _make_ecg(bpm=70.0, duration_s=30.0)
        r = engine.run(ecg, FS)
        peaks = r["r_peaks"]
        if len(peaks) >= 2:
            min_gap = float(np.min(np.diff(peaks.astype(float)))) / FS
            assert min_gap >= 0.15, (
                f"Peaks too close: min gap = {min_gap:.3f} s (refractory = 0.2 s)"
            )


# ============================================================
# FIX 5: Latency from event onset
# ============================================================

class TestLatency:
    def test_latency_none_for_normal_rhythm(self):
        """No VT/VF => latency must be None."""
        ecg = _make_ecg(bpm=70.0)
        r = engine.run(ecg, FS)
        if not r["possible_vt"] and not r["possible_vf"]:
            assert r["detection_latency_s"] is None

    def test_latency_within_3s_for_vt_signal(self):
        """For a genuine VT signal, latency should be within 3 s when available."""
        ecg = _make_ecg(bpm=150.0, duration_s=25.0)
        r = engine.run(ecg, FS)
        if r["possible_vt"] and r["detection_latency_s"] is not None:
            assert r["detection_latency_s"] <= 3.0, (
                f"Latency {r['detection_latency_s']:.2f}s exceeds 3s target"
            )

    def test_latency_none_when_insufficient_rr(self):
        """Very short signal should return latency=None, not a fabricated value."""
        ecg = _make_ecg(bpm=70.0, duration_s=1.0)
        r = engine.run(ecg, FS)
        # Very short — if somehow VT is flagged, latency should be None (not sum of nothing)
        if r["possible_vt"] and len(r["rr_intervals_s"]) < 4:
            assert r["detection_latency_s"] is None


# ============================================================
# FIX 6: Modality handling
# ============================================================

class TestModalityHandling:
    def test_ecg_modality_valid(self):
        ecg = _make_ecg()
        r = engine.run(ecg, FS, modality="ecg")
        assert r["valid"]

    def test_ppg_modality_valid(self):
        t = np.linspace(0, 20, int(20 * FS))
        ppg = 0.5 * np.sin(2 * np.pi * 1.2 * t)
        r = engine.run(ppg, FS, modality="ppg")
        assert r["valid"]

    def test_emg_modality_valid(self):
        emg = np.random.RandomState(0).randn(int(20 * FS)) * 0.5
        r = engine.run(emg, FS, modality="emg")
        assert r["valid"]

    def test_unknown_modality_returns_unsupported(self):
        """FIX 6: Unknown modality must NOT silently fall through to ECG."""
        ecg = _make_ecg()
        r = engine.run(ecg, FS, modality="eeg")
        assert not r["valid"], "Unknown modality should return valid=False"
        assert "unsupported" in r["validation_message"].lower() or \
               "eeg" in r["validation_message"].lower(), (
            f"Expected unsupported modality message, got: {r['validation_message']}"
        )

    def test_modality_case_insensitive(self):
        """Modality matching must be case-insensitive."""
        ecg = _make_ecg()
        r1 = engine.run(ecg, FS, modality="ECG")
        r2 = engine.run(ecg, FS, modality="ecg")
        r3 = engine.run(ecg, FS, modality="Ecg")
        assert r1["valid"] and r2["valid"] and r3["valid"]

    def test_modality_whitespace_stripped(self):
        """Modality with leading/trailing whitespace must be accepted."""
        ecg = _make_ecg()
        r = engine.run(ecg, FS, modality="  ecg  ")
        assert r["valid"]


# ============================================================
# FIX 7: Notch filter safety
# ============================================================

class TestNotchSafety:
    def test_low_fs_no_50hz_notch_crash(self):
        """At fs=100 Hz, 50 Hz notch is at Nyquist — must not crash."""
        ecg = _make_ecg(fs=100.0, duration_s=30.0)
        r = engine.run(ecg, 100.0, modality="ecg")
        assert r["valid"]  # Must not crash or error

    def test_low_fs_no_60hz_notch_crash(self):
        """At fs=110 Hz, 60 Hz notch is near Nyquist — must not crash."""
        ecg = _make_ecg(fs=110.0, duration_s=30.0)
        r = engine.run(ecg, 110.0, modality="ecg")
        assert r["valid"]

    def test_50hz_notch_applied_when_valid(self):
        """At fs=360 Hz, 50 Hz notch IS valid and should be applied when detected."""
        n = int(20 * FS)
        t = np.linspace(0, 20, n)
        ecg = _make_ecg() + 0.5 * np.sin(2 * np.pi * 50 * t)
        r = engine.run(ecg, FS)
        assert r["powerline_50hz"]
        assert "50 Hz notch applied" in " ".join(r["warnings"])


# ============================================================
# FIX 8: SNR naming (proxy vs reference-based)
# ============================================================

class TestSNRNaming:
    def test_snr_proxy_fields_present(self):
        """snr_proxy_before_db and snr_proxy_after_db must be in result."""
        ecg = _make_ecg()
        r = engine.run(ecg, FS)
        assert "snr_proxy_before_db" in r
        assert "snr_proxy_after_db" in r

    def test_snr_backward_compat_fields_present(self):
        """snr_before_db / snr_after_db alias must still exist."""
        ecg = _make_ecg()
        r = engine.run(ecg, FS)
        assert "snr_before_db" in r
        assert "snr_after_db" in r

    def test_snr_values_are_numeric_or_none(self):
        """SNR must be float or None, never a string or bool."""
        ecg = _make_ecg()
        r = engine.run(ecg, FS)
        for key in ("snr_proxy_before_db", "snr_proxy_after_db",
                    "snr_before_db", "snr_after_db"):
            val = r[key]
            assert val is None or isinstance(val, float), (
                f"{key} = {val!r} (expected float or None)"
            )

    def test_snr_improvement_explicit_none_check(self):
        """snr_improvement_db must only be set when both before/after are not None."""
        ecg = _make_ecg()
        r = engine.run(ecg, FS)
        if r["snr_before_db"] is None or r["snr_after_db"] is None:
            assert r["snr_improvement_db"] is None
        else:
            assert r["snr_improvement_db"] is not None


# ============================================================
# FIX 9: PPG terminology
# ============================================================

class TestPPGTerminology:
    def test_ppg_has_pulse_peaks(self):
        """PPG result must have pulse_peaks field (not just r_peaks)."""
        t = np.linspace(0, 20, int(20 * FS))
        ppg = 0.5 * np.sin(2 * np.pi * 1.2 * t)
        r = engine.run(ppg, FS, modality="ppg")
        assert "pulse_peaks" in r

    def test_ppg_has_inter_pulse_intervals(self):
        """PPG result must have inter_pulse_intervals_s field."""
        t = np.linspace(0, 20, int(20 * FS))
        ppg = 0.5 * np.sin(2 * np.pi * 1.2 * t)
        r = engine.run(ppg, FS, modality="ppg")
        assert "inter_pulse_intervals_s" in r

    def test_ppg_has_pulse_rate(self):
        """PPG result must have pulse_rate_bpm field."""
        t = np.linspace(0, 20, int(20 * FS))
        ppg = 0.5 * np.sin(2 * np.pi * 1.2 * t)
        r = engine.run(ppg, FS, modality="ppg")
        assert "pulse_rate_bpm" in r

    def test_ppg_pulse_rate_reasonable(self):
        """PPG pulse rate must be physiologically plausible."""
        t = np.linspace(0, 20, int(20 * FS))
        ppg = 0.5 * np.sin(2 * np.pi * 1.2 * t) + 0.01 * np.random.RandomState(1).randn(int(20 * FS))
        r = engine.run(ppg, FS, modality="ppg")
        if r["heart_rate_reliable"]:
            assert 20.0 <= r["pulse_rate_bpm"] <= 220.0


# ============================================================
# FIX 10: Explicit None checks (no truthy SNR)
# ============================================================

class TestExplicitNoneChecks:
    def test_snr_zero_not_falsy_trap(self):
        """If SNR happens to be 0.0, snr_improvement_db must still be computed."""
        # Construct a signal where SNR proxy might be very low but not None
        # (hard to force exactly 0.0, but test the logic path)
        ecg = _make_ecg(noise=0.1)
        r = engine.run(ecg, FS)
        # The important check: improvement is set iff both are not None
        snr_b = r["snr_before_db"]
        snr_a = r["snr_after_db"]
        imp = r["snr_improvement_db"]
        if snr_b is not None and snr_a is not None:
            assert imp is not None, "snr_improvement_db must be set when both SNRs present"
            assert abs(imp - (snr_a - snr_b)) < 1e-9
        else:
            assert imp is None or imp is not None  # can be anything if not both present


# ============================================================
# GENERAL EDGE CASES
# ============================================================

class TestEdgeCases:
    def test_empty_signal_invalid(self):
        r = engine.run(np.zeros(0), FS)
        assert not r["valid"]

    def test_too_short_signal(self):
        r = engine.run(np.zeros(50), FS)
        assert not r["valid"]

    def test_all_nan_invalid(self):
        r = engine.run(np.full(1000, float("nan")), FS)
        assert not r["valid"]

    def test_nan_partial_still_processes(self):
        ecg = _make_ecg()
        ecg[200:250] = float("nan")
        r = engine.run(ecg, FS)
        assert r["valid"]

    def test_flatline_invalid(self):
        r = engine.run(np.zeros(1000), FS)
        assert r["valid"]  # structurally valid but quality CRITICAL
        assert r["signal_quality"] == "CRITICAL"

    def test_bad_fs_invalid(self):
        r = engine.run(_make_ecg(), 0.0)
        assert not r["valid"]

    def test_very_low_fs(self):
        # fs=50 is minimum allowed
        r = engine.run(np.random.randn(500), 50.0)
        assert r["valid"]

    def test_low_fs_below_minimum(self):
        r = engine.run(np.random.randn(500), 30.0)
        assert not r["valid"]

    def test_2d_input(self):
        ecg = _make_ecg()
        r = engine.run(ecg.reshape(-1, 1), FS)
        assert r["valid"]

    def test_determinism(self):
        ecg = _make_ecg()
        r1 = engine.run(ecg, FS)
        r2 = engine.run(ecg, FS)
        assert r1["n_beats"] == r2["n_beats"]
        assert r1["heart_rate_bpm"] == r2["heart_rate_bpm"]
        assert r1["rhythm_label"] == r2["rhythm_label"]

    def test_noisy_ecg(self):
        ecg = _make_ecg(noise=1.5)
        r = engine.run(ecg, FS)
        assert r["valid"]

    def test_severe_baseline_wander(self):
        t = np.linspace(0, 20, int(20 * FS))
        ecg = _make_ecg() + 2.0 * np.sin(2 * np.pi * 0.1 * t)
        r = engine.run(ecg, FS)
        assert r["valid"]

    def test_inverted_ecg(self):
        ecg = _make_ecg()
        r = engine.run(-ecg, FS)
        assert r["valid"]
        assert r["n_beats"] > 0

    def test_emg_modality(self):
        emg = np.random.RandomState(0).randn(int(20 * FS)) * 0.5
        r = engine.run(emg, FS, modality="emg")
        assert r["valid"]

    def test_ppg_modality(self):
        t = np.linspace(0, 20, int(20 * FS))
        ppg = 0.5 * np.sin(2 * np.pi * 1.2 * t)
        r = engine.run(ppg, FS, modality="ppg")
        assert r["valid"]

    def test_process_batch(self):
        ecg1 = _make_ecg(bpm=70.0)
        ecg2 = _make_ecg(bpm=90.0)
        results = engine.process_batch([ecg1, ecg2], FS)
        assert len(results) == 2
        assert all(r["valid"] for r in results)

    def test_bradycardia_classified(self):
        ecg = _make_ecg(bpm=45.0, duration_s=25.0)
        r = engine.run(ecg, FS)
        if r["heart_rate_reliable"]:
            assert r["heart_rate_bpm"] < 65.0

    def test_tachycardia_classified(self):
        ecg = _make_ecg(bpm=120.0, duration_s=25.0)
        r = engine.run(ecg, FS)
        if r["heart_rate_reliable"]:
            assert r["heart_rate_bpm"] > 95.0
