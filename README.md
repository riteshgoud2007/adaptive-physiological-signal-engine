# Adaptive ECG Signal Intelligence Engine

> **Real ECG → Signal Understanding → Noise Cancellation → Adaptive Processing → Explainable Results**

## Overview

A fully functional Python application that analyses **real ECG recordings** from external sources (PhysioNet, MIT-BIH, BIDMC, etc.).

The engine continuously monitors signal quality, detects changes in noise and signal characteristics, adapts its processing strategy, and explains every decision.

### Key Concept

> Existing ECG algorithms can detect heartbeats, but real ECG recordings can change because of noise, motion, baseline drift, saturation, missing samples, and other signal changes. This system continuously monitors these changes, adapts its processing and detection strategy, identifies unreliable regions, and explains its decisions.

---

## Quick Start

### 1. Install Python

Download Python 3.10+ from https://python.org

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Optional (for WFDB/PhysioNet support):
```bash
pip install wfdb
```

### 3. Run the application

**Windows:**
```
Double-click run_app.bat
```

Or from terminal:
```bash
streamlit run app.py
```

### 4. Open browser

Navigate to: http://localhost:8501

---

## Supported Input Formats

| Format | Description |
|--------|-------------|
| **CSV** | Time+ECG columns, ECG-only, or multi-channel |
| **TXT** | Whitespace-separated numeric data |
| **NPY** | NumPy array (1D or 2D) |
| **NPZ** | NumPy archive (keys: ecg, signal, data) |
| **WFDB** | PhysioNet .dat/.hea records |

### CSV Format Examples

**Time + ECG:**
```csv
time,ecg
0.000,0.12
0.004,0.15
```

**ECG only:**
```csv
0.12
0.15
0.13
```

**Multi-channel:**
```csv
time,Lead_I,Lead_II,Lead_III
0.000,0.12,0.20,0.15
```

---

## Recommended ECG Sources

- [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/) — 2-channel, 360 Hz, annotated
- [MIT-BIH Normal Sinus Rhythm](https://physionet.org/content/nsrdb/) — normal ECG recordings
- [BIDMC ECG Database](https://physionet.org/content/bidmc/) — clinical ICU recordings

---

## Application Features

### Core Pipeline
```
Real ECG → Validation → Signal Quality → Noise Characterisation
         → Baseline Correction → Adaptive Noise Cancellation
         → ECG Filtering → Adaptive R-Peak Detection
         → RR Intervals → Heart Rate → Reliability Assessment
```

### Signal Quality Engine
- Flatline detection
- Saturation / clipping detection
- High-frequency noise estimation
- Baseline drift analysis
- SNR estimation
- Kurtosis analysis
- QRS visibility assessment
- 50/60 Hz powerline interference detection

### Adaptive Features
- Noise profile change detection (KL divergence)
- CUSUM change-point detection
- Sensor/device change detection
- Per-segment quality monitoring
- Adaptive notch filtering (only applied when needed)

### R-Peak Detection
- Derivative + squaring + moving-window integration (Pan-Tompkins)
- Adaptive signal/noise threshold tracking
- 200 ms refractory period
- Search-back for missed beats
- Local peak refinement
- Explainable per-peak decisions

### Algorithm Comparison Lab
- Pan-Tompkins (fixed threshold reference)
- Hamilton detector
- Christov detector
- Precision / Recall / F1 when ground-truth annotations available

---

## Project Structure

```
adaptive_ecg_engine/
├── app.py                  # Streamlit UI
├── adaptive_engine.py      # Core processing + R-peak detection
├── ecg_loader.py           # File loading (CSV, TXT, NPY, NPZ, WFDB)
├── quality_engine.py       # Signal quality + adaptive monitoring
├── evaluation.py           # Algorithm comparison + evaluation
├── config.py               # Central configuration
├── requirements.txt
├── run_app.bat
├── README.md
└── tests/
    ├── test_loader.py
    ├── test_preprocessing.py
    ├── test_quality.py
    ├── test_detector.py
    └── test_adaptation.py
```

---

## System Requirements

- **OS:** Windows 10/11, Linux, macOS
- **Python:** 3.10+
- **RAM:** 8 GB+ recommended
- **CPU:** 4+ cores recommended
- **GPU:** Not required
- **Internet:** Not required after installation

---

## Medical Disclaimer

> For research and demonstration purposes only. This prototype is not a medical diagnostic device. Do not use for clinical diagnosis or treatment decisions.

---

## Hackathon Demo Guide

For the strongest demo, prepare three real ECG recordings:

1. **Clean ECG** — show baseline performance
2. **Noisy ECG** — show adaptive noise cancellation
3. **Artifact/drift ECG** — show adaptive monitoring and processing

For each:
1. Upload the recording
2. Observe Signal Quality report
3. Observe Adaptive Signal Monitor timeline
4. Compare Before/After panels
5. Enable Research Mode to show algorithm internals
6. Enable Algorithm Comparison to show benchmarks
