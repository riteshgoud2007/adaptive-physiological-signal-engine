# Adaptive Physiological Signal Engine

**Competition Submission — OptiForge 2026**  
Version 2.1.0 | Python 3.10+

> **Research Disclaimer**: This system is for research and demonstration purposes only.  
> It is **not** a certified medical device and does **not** provide clinical diagnoses.

---

## Problem Statement

Physiological signals (ECG, EMG, PPG) recorded in real-world conditions exhibit non-stationary noise, baseline wander, muscle artifact, power-line interference, motion artifacts, and missing samples. Static preprocessing pipelines that assume clean, stationary signals fail on these recordings.

This engine implements an **adaptive signal processing pipeline** that:
1. Detects changing signal conditions segment-by-segment
2. Adapts preprocessing parameters to the detected characteristics
3. Produces explainable outputs describing what changed and why

---

## Competition Entry Point

The official competition interface is **`final_submission.py`** — fully self-contained, no project dependencies.

### `run(signal, fs, modality='ecg', channel=0) -> dict`

| Parameter | Type | Description |
|-----------|------|-------------|
| `signal` | `np.ndarray` | Raw 1D signal (or 2D: samples × channels) |
| `fs` | `float` | Sampling rate Hz (valid: 50–10000) |
| `modality` | `str` | `'ecg'`, `'emg'`, or `'ppg'` (case-insensitive) |
| `channel` | `int` | Channel index for 2D input (default: 0) |

**Selected output keys:**

| Key | Description |
|-----|-------------|
| `valid` | Whether the signal passed validation |
| `signal_quality` | GOOD / MODERATE / POOR / CRITICAL |
| `signal_quality_score` | 0.0–1.0 quality score |
| `r_peaks` | Detected R-peak sample indices (ECG) |
| `pulse_peaks` | Detected pulse peak indices (PPG) |
| `n_beats` | Detected beats/pulses |
| `heart_rate_bpm` | Mean heart/pulse rate |
| `rr_intervals_s` | Inter-beat intervals (s) |
| `rhythm_label` | Rhythm class |
| `rhythm_confidence_score` | 0.0–1.0 |
| `possible_vt` | Possible VT (multi-evidence) |
| `possible_vf` | Possible VF (signal-level) |
| `detection_latency_s` | Seconds from event onset to detection |
| `snr_proxy_before_db` | Signal-derived SNR proxy before filtering (dB) |
| `snr_proxy_after_db` | Signal-derived SNR proxy after filtering (dB) |
| `snr_improvement_db` | SNR proxy improvement |
| `morphology_score` | QRS morphology preservation (0–1) |
| `emg_severity` | EMG contamination level |
| `powerline_50hz` | 50 Hz interference detected |
| `powerline_60hz` | 60 Hz interference detected |
| `detection_reliability` | Peak detection confidence (0–1) |
| `processing_time_s` | Wall-clock processing time |
| `warnings` | Processing warnings list |

> **Ground-truth metrics** (sensitivity, specificity, F1, timing jitter) require annotated benchmark data and the official evaluator — not computed without them.

### `process_batch(signals, fs, modality='ecg') -> list[dict]`

Processes a list of signals. Each result matches the `run()` output format.

---

## Architecture

```
final_submission.py  (self-contained competition engine)
├── run()
│   ├── _validate_input()       Input validation + dtype coercion
│   ├── _process_ecg()          ECG full pipeline
│   │   ├── _interpolate_nans()
│   │   ├── _estimate_baseline()
│   │   ├── _compute_welch_psd()  Shared PSD — used for ALL spectral checks
│   │   ├── _detect_powerline()   Adaptive: notch only when interference found
│   │   ├── _characterize_emg()   Reuses shared PSD
│   │   ├── _dominant_noise_type() Reuses shared PSD
│   │   ├── _bandpass()           Butterworth SOS (cached via lru_cache)
│   │   ├── _detect_r_peaks()     Pan-Tompkins adaptive threshold
│   │   ├── _classify_rhythm()    Multi-evidence VT/VF/NSR/etc.
│   │   ├── _compute_alert_latency()
│   │   └── _measure_morphology() Vectorized QRS correlation
│   ├── _process_ppg()
│   └── _process_emg()
└── process_batch()
```

**Key design**: `final_submission.py` is entirely standalone — imports only `numpy`, `scipy`. Reproducible, portable competition evaluation.

---

## ECG Processing Pipeline

| Step | Detail |
|------|--------|
| 1. Input validation | dtype coercion, length/FS/NaN checks |
| 2. NaN interpolation | linear over missing samples |
| 3. Baseline removal | median filter 200ms window |
| 4. Shared Welch PSD | computed ONCE, reused for steps 5, 6, 7 |
| 5. Adaptive powerline notch | 50/60 Hz applied only when detected |
| 6. EMG characterization | from shared PSD, no extra FFT |
| 7. Noise type | from shared PSD, no extra FFT |
| 8. Adaptive bandpass | Butterworth SOS, cached per (low, high, fs, order) |
| 9. Pan-Tompkins MWI | derivative → squaring → moving-window integration |
| 10. Adaptive threshold | SPKI/NPKI running estimates, updated per beat |
| 11. Search-back | missing beats recovered in long RR gaps (>166% median) |
| 12. Polarity refinement | handles inverted ECG waveforms |
| 13. Rhythm classification | multi-evidence: HR + regularity + entropy + beat count |
| 14. VT detection | requires ≥6 beats, HR, regularity AND entropy evidence |
| 15. VF detection | signal-level: absent QRS, spectral entropy, amplitude chaos |
| 16. Alert latency | from event *onset* (not signal start) |
| 17. Morphology | vectorized QRS correlation before/after (capped at 60 beats) |

---

## Adaptive Preprocessing

| Condition | Adaptation |
|-----------|------------|
| 50 Hz powerline | Apply 50 Hz notch filter |
| 60 Hz powerline | Apply 60 Hz notch filter |
| No powerline | No notch (preserves morphology) |
| Severe EMG | Adaptive highcut suppression |
| Baseline drift | Median-filter subtraction |
| Missing samples (NaN) | Linear interpolation before filtering |
| Inverted ECG | Polarity-aware peak detection |
| Low signal quality | Lower reliability score |

Every adaptive decision is reported in the `warnings` list.

---

## Performance (v2.1.0 — optimized)

Measured on a standard laptop (Python 3.14, scipy 1.x):

| Signal | Before (v2.0) | After (v2.1.0) | Speedup |
|--------|--------------|----------------|---------|
| ECG 5s | 25.2 ms | 5.5 ms | **4.6×** |
| ECG 10s | 27.3 ms | 7.8 ms | **3.5×** |
| ECG 30s | 55.5 ms | 12.3 ms | **4.5×** |
| ECG 60s | 88.4 ms | 20.3 ms | **4.4×** |
| EMG 60s | 17.6 ms | 8.0 ms | **2.2×** |
| PPG 60s | 10.1 ms | 4.6 ms | **2.2×** |

**Key optimizations:**
- Filter coefficients cached via `functools.lru_cache` — designed once per (low, high, fs, order) tuple
- Single Welch PSD shared across powerline detection + EMG characterization + noise type (3 FFTs → 1)
- `_measure_morphology` fully vectorized — NumPy matrix ops replace Python loop (2500+ median calls → 6 vectorized operations)
- Beat sampling capped at 60 for morphology (sufficient for reliable score)

---

## Rhythm Classes

| Class | Conditions |
|-------|------------|
| Normal Sinus Rhythm | 60–100 BPM, regular |
| Bradycardia | < 60 BPM |
| Tachycardia | 100–150 BPM, regular |
| Possible VT | ≥150 BPM, ≥6 beats, regular, low entropy, multi-evidence |
| Possible VF | Absent/disorganized QRS, spectral chaos |
| Irregular Rhythm | High RR variability |
| Undetermined | Insufficient data or low quality |

---

## Running Instructions

### Competition engine
```python
import numpy as np
import final_submission as eng

signal = np.loadtxt('ecg.csv')
result = eng.run(signal, fs=360.0, modality='ecg')
print(result['heart_rate_bpm'])
print(result['rhythm_label'])
print(result['signal_quality'])
```

### Streamlit UI
```bash
pip install -r requirements.txt
streamlit run app.py
```

### Tests
```bash
pip install pytest
python -m pytest tests/ -v
```

### Benchmark
```bash
python benchmark_submission.py
```

---

## Testing

```
tests/
├── test_adaptive_engine.py        Core processing tests
├── test_config.py                 Configuration tests
├── test_ecg_loader.py             ECG loading tests
├── test_evaluation.py             Evaluation framework tests
├── test_quality_engine.py         Signal quality tests
├── test_new_engines.py            Extended engine tests
└── test_hardening.py              Hardening/edge-case tests (155 total)
```

Run all: `python -m pytest tests/ -v`

---

## Security

- **No secrets, API keys, or tokens** anywhere in the project
- **No network access** in the competition engine
- **No absolute paths** — fully portable
- **No cloud dependencies** — runs fully offline
- **No GUI dependencies** in `final_submission.py`
- See `.env.example` for optional Streamlit configuration

---

## Limitations

- Not a medical device — outputs for research only
- Sensitivity/specificity require annotated ground truth
- OptiForge fitness score requires the official evaluator (not available locally)
- VT/VF flags are probabilistic indicators, not clinical diagnoses
- Motion status in PPG dashboard estimated from signal quality (no accelerometer)
- SNR proxy is signal-derived (high-frequency difference method), not reference-based

---

## Ground-Truth Metric Policy

This implementation **never fabricates** clinical metrics.

- SNR improvement: computed from actual raw vs processed signal
- Morphology score: computed from actual QRS correlation before/after
- Detection latency: computed from actual event onset in RR series

**Not reported** without annotated ground truth: sensitivity, specificity, F1, timing jitter, false positive rate, OptiForge fitness score.
