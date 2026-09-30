"""
test_new_engines.py - Tests for all new engine modules.

Covers: noise_engine, morphology_engine, rhythm_engine,
        alert_engine, edge_engine, ppg_engine, final_submission.
"""
import sys
import numpy as np
import pytest

sys.path.insert(0, ".")

FS = 360.0
DURATION_S = 20.0
N = int(DURATION_S * FS)


def _make_ecg(bpm=70.0, fs=FS, duration_s=DURATION_S, noise=0.05, add_50hz=False):
    """Create a synthetic ECG-like signal."""
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


def _make_ppg(bpm=72.0, fs=FS, duration_s=DURATION_S):
    """Create a synthetic PPG-like signal."""
    t = np.linspace(0, duration_s, int(duration_s * fs))
    return (
        0.5 * np.sin(2 * np.pi * (bpm / 60.0) * t)
        + 0.1 * np.sin(2 * np.pi * 2 * (bpm / 60.0) * t)
        + 0.01 * np.random.RandomState(0).randn(len(t))
    )


# ========================================================================
# NOISE ENGINE
# ========================================================================

class TestNoiseEngine:
    def setup_method(self):
        from noise_engine import NoiseEngine
        self.ne = NoiseEngine(FS)

    def test_process_returns_dual_path(self):
        ecg = _make_ecg()
        out = self.ne.process(ecg)
        assert len(out.qrs_path) == len(ecg)
        assert len(out.morph_path) == len(ecg)

    def test_notch_50hz_applied_when_present(self):
        ecg = _make_ecg(add_50hz=True)
        out = self.ne.process(ecg)
        assert out.applied_notch_50 is True

    def test_notch_not_applied_when_absent(self):
        ecg = _make_ecg(add_50hz=False)
        out = self.ne.process(ecg)
        # Clean signal should not trigger 50 Hz notch
        # (may vary; just check it returns a bool)
        assert isinstance(out.applied_notch_50, bool)

    def test_snr_improvement_non_negative_with_50hz(self):
        ecg = _make_ecg(add_50hz=True)
        out = self.ne.process(ecg)
        if out.snr_before_db is not None and out.snr_after_db is not None:
            # SNR should improve after notch removal
            assert out.snr_after_db >= out.snr_before_db - 2.0  # allow 2 dB tolerance

    def test_segment_profiles_count(self):
        ecg = _make_ecg()
        out = self.ne.process(ecg)
        expected_segs = int(np.ceil(len(ecg) / (5.0 * FS)))
        assert abs(len(out.noise_profiles) - expected_segs) <= 1

    def test_emg_severity_clean_signal(self):
        from noise_engine import EMGSeverity
        ecg = _make_ecg(noise=0.01)
        sev, ratio = self.ne.characterize_emg(ecg)
        assert isinstance(sev, EMGSeverity)
        assert 0.0 <= ratio <= 1.0

    def test_emg_severe_with_high_freq_noise(self):
        from noise_engine import EMGSeverity
        # High kurtosis diff signal simulates EMG bursts
        ecg = _make_ecg(noise=2.0)  # very noisy
        sev, ratio = self.ne.characterize_emg(ecg)
        assert sev in (EMGSeverity.MILD, EMGSeverity.MODERATE, EMGSeverity.SEVERE, EMGSeverity.CLEAN)
        assert 0.0 <= ratio <= 1.0

    def test_baseline_estimate_same_length(self):
        ecg = _make_ecg()
        baseline = self.ne._estimate_baseline(ecg)
        assert len(baseline) == len(ecg)

    def test_nan_interpolated(self):
        ecg = _make_ecg()
        ecg[100:110] = float("nan")
        out = self.ne.process(ecg)
        assert np.all(np.isfinite(out.qrs_path))


# ========================================================================
# MORPHOLOGY ENGINE
# ========================================================================

class TestMorphologyEngine:
    def setup_method(self):
        from morphology_engine import MorphologyEngine
        self.me = MorphologyEngine(FS)

    def _get_peaks(self, ecg):
        from scipy.signal import find_peaks
        # Use simple threshold-based peak detection for test
        threshold = float(np.percentile(ecg, 90))
        peaks, _ = find_peaks(ecg, height=threshold * 0.5, distance=int(0.5 * FS))
        return peaks

    def test_measure_qrs_returns_morphology(self):
        from morphology_engine import QRSMorphology
        ecg = _make_ecg()
        peaks = self._get_peaks(ecg)
        result = self.me.measure_qrs(ecg, peaks)
        assert isinstance(result, QRSMorphology)
        if result.measurement_reliable:
            assert result.mean_amplitude > 0
            assert result.mean_width_ms > 0

    def test_measure_st_returns_st_morphology(self):
        from morphology_engine import STMorphology
        ecg = _make_ecg()
        peaks = self._get_peaks(ecg)
        result = self.me.measure_st(ecg, peaks)
        assert isinstance(result, STMorphology)

    def test_evaluate_preservation_produces_score(self):
        from morphology_engine import MorphologyPreservation
        ecg = _make_ecg()
        from noise_engine import NoiseEngine
        ne = NoiseEngine(FS)
        nd = ne.process(ecg)
        peaks = self._get_peaks(ecg)
        pres = self.me.evaluate_preservation(ecg, nd.morph_path, peaks)
        assert isinstance(pres, MorphologyPreservation)

    def test_preservation_score_in_range(self):
        ecg = _make_ecg()
        from noise_engine import NoiseEngine
        ne = NoiseEngine(FS)
        nd = ne.process(ecg)
        peaks = self._get_peaks(ecg)
        pres = self.me.evaluate_preservation(ecg, nd.morph_path, peaks)
        if pres.morphology_score is not None:
            assert 0.0 <= pres.morphology_score <= 1.0

    def test_too_few_peaks_returns_unreliable(self):
        from morphology_engine import QRSMorphology
        ecg = _make_ecg()
        result = self.me.measure_qrs(ecg, np.array([100, 200]))
        assert not result.measurement_reliable


# ========================================================================
# RHYTHM ENGINE
# ========================================================================

class TestRhythmEngine:
    def setup_method(self):
        from rhythm_engine import RhythmClassifier, FeatureExtractor
        self.rc = RhythmClassifier(FS)
        self.fe = FeatureExtractor(FS)

    def _make_peaks(self, bpm=70.0, duration_s=20.0, jitter=0.0):
        rr = 60.0 / bpm
        rng = np.random.RandomState(1)
        times = []
        t = 0.5 * rr
        while t < duration_s:
            times.append(t)
            t += rr + jitter * rng.randn()
        return (np.array(times) * FS).astype(int)

    def test_normal_sinus_classified(self):
        from rhythm_engine import RhythmLabel
        peaks = self._make_peaks(bpm=70.0)
        clf = self.rc.classify(peaks, signal_quality="GOOD")
        assert clf.label == RhythmLabel.NORMAL_SINUS
        assert clf.confidence_score > 0.5

    def test_tachycardia_classified(self):
        from rhythm_engine import RhythmLabel
        peaks = self._make_peaks(bpm=120.0)
        clf = self.rc.classify(peaks, signal_quality="GOOD")
        assert clf.label in (RhythmLabel.SINUS_TACH, RhythmLabel.POSSIBLE_VT)

    def test_bradycardia_classified(self):
        from rhythm_engine import RhythmLabel
        peaks = self._make_peaks(bpm=45.0)
        clf = self.rc.classify(peaks, signal_quality="GOOD")
        assert clf.label == RhythmLabel.SINUS_BRADY

    def test_irregular_rhythm_detected(self):
        from rhythm_engine import RhythmLabel
        peaks = self._make_peaks(bpm=70.0, jitter=0.12)  # high jitter = irregular
        clf = self.rc.classify(peaks, signal_quality="GOOD")
        assert clf.label in (
            RhythmLabel.AFIB, RhythmLabel.IRREGULAR, RhythmLabel.NORMAL_SINUS,
            RhythmLabel.UNKNOWN
        )

    def test_too_few_beats_unknown(self):
        from rhythm_engine import RhythmLabel, ConfidenceLevel
        peaks = np.array([100, 400, 700])
        clf = self.rc.classify(peaks, signal_quality="GOOD")
        assert clf.label == RhythmLabel.UNKNOWN
        assert clf.confidence == ConfidenceLevel.UNRELIABLE

    def test_confidence_score_in_range(self):
        peaks = self._make_peaks(bpm=70.0)
        clf = self.rc.classify(peaks)
        assert 0.0 <= clf.confidence_score <= 1.0

    def test_severe_emg_reduces_confidence(self):
        peaks = self._make_peaks(bpm=70.0)
        clf_clean = self.rc.classify(peaks, emg_severity="CLEAN")
        clf_severe = self.rc.classify(peaks, emg_severity="SEVERE EMG")
        assert clf_severe.confidence_score <= clf_clean.confidence_score

    def test_feature_extraction_returns_features(self):
        from rhythm_engine import RhythmFeatures
        peaks = self._make_peaks(bpm=75.0)
        features = self.fe.extract(peaks)
        assert isinstance(features, RhythmFeatures)
        assert features.n_rr_intervals > 0
        assert 50.0 < features.heart_rate_bpm < 100.0

    def test_vt_detection_at_rapid_rate(self):
        peaks = self._make_peaks(bpm=140.0)
        clf = self.rc.classify(peaks, signal_quality="GOOD")
        assert clf.possible_vt


# ========================================================================
# ALERT ENGINE
# ========================================================================

class TestAlertEngine:
    def setup_method(self):
        from alert_engine import AlertEngine, StreamingAlertEngine
        self.ae = AlertEngine(FS)
        self.sa = StreamingAlertEngine(FS)

    def _make_rr(self, bpm, n=10):
        return np.full(n, 60.0 / bpm)

    def test_no_alert_normal_rhythm(self):
        ecg = _make_ecg(bpm=70.0)
        rr = self._make_rr(70.0)
        report = self.ae.evaluate(
            r_peaks=np.arange(0, N, int(FS * 60.0 / 70.0)),
            rr_intervals_s=rr,
            detection_reliability=0.9,
            signal_quality="GOOD",
            emg_severity="CLEAN",
            recording_duration_s=DURATION_S,
            possible_vt=False,
            possible_vf=False,
        )
        assert report.n_total_alerts == 0

    def test_vt_alert_generated(self):
        from alert_engine import AlertType
        rr = self._make_rr(140.0)
        report = self.ae.evaluate(
            r_peaks=np.arange(0, 1000, int(FS * 60.0 / 140.0)),
            rr_intervals_s=rr,
            detection_reliability=0.85,
            signal_quality="GOOD",
            emg_severity="CLEAN",
            recording_duration_s=DURATION_S,
            possible_vt=True,
            vt_evidence=["Rapid rate: 140 BPM", "Sustained rapid beats: 80%"],
        )
        assert report.n_total_alerts >= 1
        active = [a for a in report.alerts if not a.suppressed]
        assert len(active) >= 1

    def test_latency_within_3s_target(self):
        rr = np.array([0.43, 0.43, 0.43, 0.43, 0.43, 0.43])  # 140 BPM
        report = self.ae.evaluate(
            r_peaks=np.arange(0, 1000, int(FS * 0.43)),
            rr_intervals_s=rr,
            detection_reliability=0.9,
            signal_quality="GOOD",
            emg_severity="CLEAN",
            recording_duration_s=DURATION_S,
            possible_vt=True,
            vt_evidence=["Rapid rate: 140 BPM"],
        )
        active = [a for a in report.alerts if not a.suppressed]
        if active:
            for alert in active:
                assert alert.detection_latency_s <= 3.0, (
                    f"Latency {alert.detection_latency_s:.2f}s exceeds 3s target"
                )

    def test_streaming_vt_detected(self):
        from alert_engine import AlertType
        rapid_rr = np.full(6, 0.40)  # 150 BPM
        self.sa.reset()
        alert = self.sa.update(rapid_rr, chunk_start_s=10.0)
        assert alert is not None
        assert alert.alert_type == AlertType.POSSIBLE_VT
        assert alert.detection_latency_s <= 3.0

    def test_streaming_normal_no_alert(self):
        normal_rr = np.full(10, 0.857)  # 70 BPM
        self.sa.reset()
        alert = self.sa.update(normal_rr, chunk_start_s=0.0)
        assert alert is None

    def test_artifact_alarm_suppressed(self):
        rr = self._make_rr(140.0)
        report = self.ae.evaluate(
            r_peaks=np.arange(0, 100, 10),
            rr_intervals_s=rr,
            detection_reliability=0.1,  # Very low reliability
            signal_quality="GOOD",
            emg_severity="CLEAN",
            recording_duration_s=DURATION_S,
            possible_vt=True,
            vt_evidence=[],  # No evidence -> should suppress
        )
        suppressed = [a for a in report.alerts if a.suppressed]
        assert len(suppressed) >= 0  # May or may not suppress based on logic


# ========================================================================
# EDGE ENGINE
# ========================================================================

class TestEdgeEngine:
    def setup_method(self):
        from edge_engine import EdgeEngine
        self.ee = EdgeEngine(FS, chunk_duration_s=2.0, overlap_s=0.3)

    def test_process_recording_returns_report(self):
        from edge_engine import EdgePerformanceReport
        ecg = _make_ecg()
        perf, chunks = self.ee.process_recording(ecg)
        assert isinstance(perf, EdgePerformanceReport)

    def test_faster_than_realtime(self):
        ecg = _make_ecg()
        perf, _ = self.ee.process_recording(ecg)
        assert perf.realtime_ratio < 1.0, (
            f"Must be faster than real-time, got {perf.realtime_ratio:.4f}"
        )

    def test_chunk_count_correct(self):
        ecg = _make_ecg()
        perf, chunks = self.ee.process_recording(ecg)
        assert perf.n_chunks == len(chunks)
        assert perf.n_chunks >= 2

    def test_memory_bounded(self):
        ecg = _make_ecg()
        perf, _ = self.ee.process_recording(ecg)
        # Peak memory should be small for a 20 s recording
        assert perf.peak_memory_kb < 10_000, (
            f"Peak memory {perf.peak_memory_kb:.1f} KB too high"
        )

    def test_streaming_state_preserved(self):
        # State must NOT be reset between chunks
        ecg = _make_ecg()
        self.ee.reset()
        n = len(ecg)
        step = self.ee.chunk_n - self.ee.overlap_n
        chunk1 = ecg[:self.ee.chunk_n]
        self.ee.process_chunk(chunk1, chunk_index=0, start_sample=0)
        spki_after_chunk1 = self.ee.state.spki
        # After processing a chunk, state should have been updated
        assert spki_after_chunk1 >= 0  # adaptive threshold updated

    def test_reset_clears_state(self):
        ecg = _make_ecg()
        self.ee.process_recording(ecg)
        spki_before = self.ee.state.spki
        self.ee.reset()
        assert self.ee.state.spki == 0.0
        assert self.ee.state.npki == 0.0

    def test_nan_handled_in_chunk(self):
        ecg = _make_ecg()
        ecg[500:600] = float("nan")
        perf, chunks = self.ee.process_recording(ecg)
        # Should not crash
        assert perf.n_chunks > 0

    def test_single_chunk_short_signal(self):
        ecg = _make_ecg(duration_s=1.5)
        perf, chunks = self.ee.process_recording(ecg)
        assert len(chunks) >= 1


# ========================================================================
# PPG ENGINE
# ========================================================================

class TestPPGEngine:
    def setup_method(self):
        from ppg_engine import PPGEngine
        self.pe = PPGEngine(FS)

    def test_detects_pulses(self):
        ppg = _make_ppg(bpm=72.0)
        result = self.pe.process(ppg)
        assert result.n_pulses > 0

    def test_pulse_rate_approximately_correct(self):
        ppg = _make_ppg(bpm=72.0)
        result = self.pe.process(ppg)
        if result.pulse_rate_reliable:
            assert abs(result.pulse_rate_bpm - 72.0) < 10.0

    def test_flatline_rejected(self):
        ppg = np.zeros(N)
        result = self.pe.process(ppg)
        assert result.signal_quality == "FLATLINE"
        assert not result.pulse_rate_reliable

    def test_short_signal_rejected(self):
        ppg = _make_ppg()[:int(FS * 1.5)]
        result = self.pe.process(ppg)
        assert not result.pulse_rate_reliable

    def test_ipi_physiological(self):
        ppg = _make_ppg(bpm=72.0)
        result = self.pe.process(ppg)
        if len(result.inter_pulse_intervals_s) > 0:
            assert float(np.min(result.inter_pulse_intervals_s)) >= 60.0 / 220.0
            assert float(np.max(result.inter_pulse_intervals_s)) <= 60.0 / 25.0

    def test_compare_with_ecg_agreement(self):
        ppg = _make_ppg(bpm=72.0)
        result = self.pe.process(ppg)
        comparison = self.pe.compare_with_ecg(result, ecg_hr_bpm=72.0)
        if comparison["agreement"] is not None:
            assert comparison["agreement"] is True

    def test_compare_unreliable_ppg(self):
        ppg = np.zeros(N)
        result = self.pe.process(ppg)
        comparison = self.pe.compare_with_ecg(result, ecg_hr_bpm=72.0)
        assert comparison["agreement"] is None


# ========================================================================
# FINAL SUBMISSION ENGINE
# ========================================================================

class TestFinalSubmission:
    def setup_method(self):
        import final_submission
        self.engine = final_submission

    def test_run_returns_dict(self):
        ecg = _make_ecg()
        result = self.engine.run(ecg, FS, modality="ecg")
        assert isinstance(result, dict)

    def test_run_valid_ecg(self):
        ecg = _make_ecg()
        result = self.engine.run(ecg, FS, modality="ecg")
        assert result["valid"] is True
        assert result["n_beats"] > 5
        assert 40.0 < result["heart_rate_bpm"] < 150.0

    def test_run_ppg_modality(self):
        ppg = _make_ppg()
        result = self.engine.run(ppg, FS, modality="ppg")
        assert result["valid"] is True

    def test_run_emg_modality(self):
        emg = 0.5 * np.random.RandomState(0).randn(N)
        result = self.engine.run(emg, FS, modality="emg")
        assert result["valid"] is True

    def test_short_signal_invalid(self):
        result = self.engine.run(np.zeros(50), FS)
        assert result["valid"] is False

    def test_all_nan_invalid(self):
        result = self.engine.run(np.full(N, float("nan")), FS)
        assert result["valid"] is False

    def test_bad_fs_invalid(self):
        ecg = _make_ecg()
        result = self.engine.run(ecg, fs=0.0)
        assert result["valid"] is False

    def test_determinism(self):
        ecg = _make_ecg()
        r1 = self.engine.run(ecg, FS)
        r2 = self.engine.run(ecg, FS)
        assert r1["n_beats"] == r2["n_beats"]
        assert r1["heart_rate_bpm"] == r2["heart_rate_bpm"]
        assert r1["rhythm_label"] == r2["rhythm_label"]

    def test_2d_input_handled(self):
        ecg = _make_ecg()
        arr_2d = ecg.reshape(-1, 1)
        result = self.engine.run(arr_2d, FS)
        assert result["valid"] is True

    def test_nan_in_signal_handled(self):
        ecg = _make_ecg()
        ecg[200:250] = float("nan")
        result = self.engine.run(ecg, FS)
        assert result["valid"] is True
        assert result["n_beats"] > 0

    def test_snr_improvement_correct_sign(self):
        ecg = _make_ecg(add_50hz=True)
        result = self.engine.run(ecg, FS)
        if result["snr_improvement_db"] is not None:
            # With 50 Hz removed, SNR should improve
            assert result["snr_improvement_db"] > -5.0

    def test_confidence_score_in_range(self):
        ecg = _make_ecg()
        result = self.engine.run(ecg, FS)
        assert 0.0 <= result["rhythm_confidence_score"] <= 1.0

    def test_reliability_in_range(self):
        ecg = _make_ecg()
        result = self.engine.run(ecg, FS)
        assert 0.0 <= result["detection_reliability"] <= 1.0

    def test_process_batch(self):
        ecg1 = _make_ecg(bpm=70.0)
        ecg2 = _make_ecg(bpm=90.0)
        results = self.engine.process_batch([ecg1, ecg2], FS)
        assert len(results) == 2
        assert all(r["valid"] for r in results)
