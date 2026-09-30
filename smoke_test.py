"""
smoke_test.py — Full pipeline smoke test for the upgraded engine.
Run with: python smoke_test.py
"""
import sys
import numpy as np
sys.path.insert(0, ".")

import final_submission as engine
from noise_engine import NoiseEngine
from morphology_engine import MorphologyEngine
from rhythm_engine import RhythmClassifier
from alert_engine import AlertEngine, StreamingAlertEngine
from edge_engine import EdgeEngine, benchmark_throughput
from ppg_engine import PPGEngine

FS = 360.0
t = np.linspace(0, 30, int(30 * FS))

# Build a realistic ECG-like signal with 70 BPM rhythm
ecg = np.zeros(len(t))
rr_s = 60.0 / 70.0
for i in range(40):
    idx = int((i + 0.5) * rr_s * FS)
    if idx < len(t):
        w = int(0.025 * FS)
        for d in range(-w * 3, w * 3 + 1):
            if 0 <= idx + d < len(t):
                ecg[idx + d] += float(np.exp(-(d / w) ** 2))

# Add 50 Hz powerline + baseline drift + noise
ecg += 0.1 * np.sin(2 * np.pi * 50 * t)
ecg += 0.3 * np.sin(2 * np.pi * 0.2 * t)
ecg += 0.05 * np.random.RandomState(42).randn(len(t))

# -------------------------------------------------------------------
print("=== 1. FULL PIPELINE (final_submission.run) ===")
result = engine.run(ecg, FS, modality="ecg")
print(f"  Valid:            {result['valid']}")
print(f"  Signal quality:   {result['signal_quality']} ({result['signal_quality_score']:.2f})")
print(f"  EMG severity:     {result['emg_severity']}")
print(f"  Powerline 50Hz:   {result['powerline_50hz']}")
print(f"  R-peaks:          {result['n_beats']}")
print(f"  Heart rate:       {result['heart_rate_bpm']:.1f} BPM")
print(f"  Rhythm:           {result['rhythm_label']}")
print(f"  Confidence:       {result['rhythm_confidence']} ({result['rhythm_confidence_score']:.2f})")
print(f"  Possible VT:      {result['possible_vt']}")
print(f"  Possible VF:      {result['possible_vf']}")
snr_b = result['snr_before_db']
snr_a = result['snr_after_db']
snr_i = result['snr_improvement_db']
print(f"  SNR before:       {round(snr_b, 2) if snr_b is not None else None} dB")
print(f"  SNR after:        {round(snr_a, 2) if snr_a is not None else None} dB")
print(f"  SNR improvement:  {round(snr_i, 2) if snr_i is not None else None} dB")
print(f"  Morphology score: {result['morphology_score']}")
print(f"  Reliability:      {result['detection_reliability']:.2f}")
print(f"  Processing time:  {result['processing_time_s']*1000:.1f} ms")
print()

assert result["valid"], "Pipeline must return valid=True"
assert result["n_beats"] > 10, f"Expected >10 beats, got {result['n_beats']}"
assert 50 < result["heart_rate_bpm"] < 100, f"HR={result['heart_rate_bpm']} out of range"

# -------------------------------------------------------------------
print("=== 2. NOISE ENGINE ===")
ne = NoiseEngine(FS)
nd = ne.process(ecg)
print(f"  Notch 50Hz applied:   {nd.applied_notch_50}")
print(f"  EMG suppression:      {nd.applied_emg_suppression}")
print(f"  N segments:           {len(nd.noise_profiles)}")
print(f"  SNR before:           {round(nd.snr_before_db, 2) if nd.snr_before_db is not None else None} dB")
print(f"  SNR after:            {round(nd.snr_after_db, 2) if nd.snr_after_db is not None else None} dB")
print(f"  QRS path shape:       {nd.qrs_path.shape}")
print(f"  Morph path shape:     {nd.morph_path.shape}")
print()

assert nd.applied_notch_50, "50 Hz notch should be applied (50 Hz interference added)"
assert nd.qrs_path.shape[0] == len(ecg)
assert nd.morph_path.shape[0] == len(ecg)

# -------------------------------------------------------------------
print("=== 3. RHYTHM CLASSIFIER ===")
rc = RhythmClassifier(FS)
rp = result["r_peaks"]
clf = rc.classify(rp, ecg, signal_quality="GOOD", emg_severity="CLEAN")
print(f"  Label:      {clf.label.value}")
print(f"  Confidence: {clf.confidence.value} ({clf.confidence_score:.2f})")
print(f"  Beats:      {clf.n_beats}")
print(f"  HR:         {clf.heart_rate_bpm:.1f} BPM")
print()

assert clf.label is not None
assert 0.0 <= clf.confidence_score <= 1.0

# -------------------------------------------------------------------
print("=== 4. EDGE ENGINE (streaming, 2 s chunks) ===")
ee = EdgeEngine(FS, chunk_duration_s=2.0, overlap_s=0.3)
perf, chunks = ee.process_recording(ecg)
total_peaks = sum(len(c.r_peaks) for c in chunks)
print(f"  Chunks:          {perf.n_chunks}")
print(f"  Processing time: {perf.total_processing_time_s*1000:.1f} ms")
print(f"  Realtime ratio:  {perf.realtime_ratio:.4f}x (< 1.0 = faster than real-time)")
print(f"  Samples/sec:     {perf.samples_per_second:,.0f}")
print(f"  Peak memory:     {perf.peak_memory_kb:.1f} KB")
print(f"  Total R-peaks:   {total_peaks}")
print()

assert perf.realtime_ratio < 1.0, f"Must be faster than real-time, got {perf.realtime_ratio:.4f}"
assert total_peaks > 5, f"Expected R-peaks from chunks, got {total_peaks}"

# -------------------------------------------------------------------
print("=== 5. STREAMING ALERT ENGINE (VT scenario) ===")
# Simulate 6 rapid RR intervals at 150 BPM (0.4 s each)
rapid_rr = np.array([0.4] * 6, dtype=float)
sa = StreamingAlertEngine(FS)
alert = sa.update(rapid_rr, chunk_start_s=10.0)
if alert:
    print(f"  Alert type:    {alert.alert_type.value}")
    print(f"  Latency:       {alert.detection_latency_s:.2f} s")
    print(f"  Within 3s:     {'YES' if alert.detection_latency_s <= 3.0 else 'NO'}")
    print(f"  Message:       {alert.message}")
    assert alert.detection_latency_s <= 3.0, "VT must be detected within 3.0 s"
else:
    print("  No alert generated from rapid RR (unexpected)")
print()

# -------------------------------------------------------------------
print("=== 6. PPG ENGINE ===")
ppg = (
    0.5 * np.sin(2 * np.pi * 1.2 * t)
    + 0.1 * np.sin(2 * np.pi * 2.4 * t)
    + 0.02 * np.random.RandomState(0).randn(len(t))
)
pe = PPGEngine(FS)
pr = pe.process(ppg)
ecg_hr = result["heart_rate_bpm"]
cmp = pe.compare_with_ecg(pr, ecg_hr)
print(f"  Pulse rate:    {pr.pulse_rate_bpm:.1f} BPM")
print(f"  Pulses found:  {pr.n_pulses}")
print(f"  Quality:       {pr.signal_quality}")
print(f"  ECG HR:        {ecg_hr:.1f} BPM")
print(f"  Agreement:     {cmp['agreement']} (diff={cmp['difference_bpm']} BPM)")
print()

assert pr.n_pulses > 0, "PPG must detect at least 1 pulse"

# -------------------------------------------------------------------
print("=== 7. MORPHOLOGY PRESERVATION ===")
me = MorphologyEngine(FS)
pres = me.evaluate_preservation(ecg, nd.morph_path, rp)
print(f"  QRS amp before:    {round(pres.qrs_amplitude_before, 4) if pres.qrs_amplitude_before else None}")
print(f"  QRS amp after:     {round(pres.qrs_amplitude_after, 4) if pres.qrs_amplitude_after else None}")
print(f"  QRS correlation:   {round(pres.qrs_waveform_correlation, 4) if pres.qrs_waveform_correlation else None}")
print(f"  Morphology score:  {round(pres.morphology_score, 4) if pres.morphology_score else None}")
print()

# -------------------------------------------------------------------
print("=== 8. DETERMINISM CHECK ===")
r1 = engine.run(ecg, FS, modality="ecg")
r2 = engine.run(ecg, FS, modality="ecg")
assert r1["n_beats"] == r2["n_beats"], "Must be deterministic"
assert r1["heart_rate_bpm"] == r2["heart_rate_bpm"], "HR must be deterministic"
print(f"  Run 1 beats: {r1['n_beats']}  Run 2 beats: {r2['n_beats']}  MATCH: OK")
print()

# -------------------------------------------------------------------
print("=== 9. MULTI-MODAL: PPG run() ===")
ppg_result = engine.run(ppg, FS, modality="ppg")
print(f"  PPG valid:         {ppg_result['valid']}")
print(f"  PPG pulse rate:    {ppg_result['heart_rate_bpm']:.1f} BPM")
print(f"  PPG signal quality:{ppg_result['signal_quality']}")
print()

# -------------------------------------------------------------------
print("=== 10. EDGE CASE: NaN-heavy signal ===")
nan_ecg = ecg.copy()
nan_ecg[500:600] = float("nan")
nan_result = engine.run(nan_ecg, FS, modality="ecg")
print(f"  Valid with NaNs:  {nan_result['valid']}")
print(f"  Beats with NaNs:  {nan_result['n_beats']}")
print()

# -------------------------------------------------------------------
print("=== 11. EDGE CASE: Very short signal ===")
short_result = engine.run(ecg[:50], FS, modality="ecg")
print(f"  Short signal valid: {short_result['valid']} (expected False)")
assert not short_result["valid"]
print()

print("=" * 60)
print("ALL SMOKE TESTS PASSED")
print("=" * 60)
