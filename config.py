"""
config.py — Central configuration for the Adaptive Physiological Signal Engine.

All processing parameters, thresholds, and UI constants live here.
No business logic. Modify this file to tune the system.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Sampling / time constants
# ---------------------------------------------------------------------------

DEFAULT_FS: float = 360.0
MIN_FS: float = 50.0
MAX_FS: float = 10_000.0
MIN_DURATION_S: float = 2.0
MIN_SAMPLES: int = 100

# ---------------------------------------------------------------------------
# QRS detection bandpass (Butterworth SOS)
# ---------------------------------------------------------------------------

BP_LOW_HZ: float = 0.5
BP_HIGH_HZ: float = 45.0
BP_ORDER: int = 4

# Morphology-preservation bandpass (gentler filtering)
MORPH_BP_LOW_HZ: float = 0.05
MORPH_BP_HIGH_HZ: float = 40.0
MORPH_BP_ORDER: int = 2

# Notch filter
NOTCH_Q: float = 30.0
NOTCH_50_HZ: float = 50.0
NOTCH_60_HZ: float = 60.0
NOTCH_POWER_THRESHOLD_DB: float = 10.0

# ---------------------------------------------------------------------------
# Baseline / detrending
# ---------------------------------------------------------------------------

BASELINE_WINDOW_S: float = 0.6       # Wide window for robust baseline estimation
BASELINE_POLY_ORDER: int = 3

# ---------------------------------------------------------------------------
# Pan-Tompkins / Adaptive detector
# ---------------------------------------------------------------------------

MWI_WINDOW_S: float = 0.150
REFRACTORY_S: float = 0.200
SEARCH_BACK_S: float = 0.360
INITIAL_CALIB_S: float = 2.0
SPKI_ALPHA: float = 0.125
NPKI_ALPHA: float = 0.125

# ---------------------------------------------------------------------------
# Signal quality / change detection
# ---------------------------------------------------------------------------

SEGMENT_DURATION_S: float = 5.0
SATURATION_PERCENTILE: float = 99.0
SATURATION_THRESHOLD: float = 0.98
FLATLINE_STD_THRESHOLD: float = 1e-4
CLIPPING_FRACTION_WARN: float = 0.01
KL_SHIFT_THRESHOLD: float = 0.15
CHANGE_CUSUM_THRESHOLD: float = 4.0
CHANGE_CUSUM_DRIFT: float = 0.5

# ---------------------------------------------------------------------------
# EMG detection thresholds
# ---------------------------------------------------------------------------

EMG_MILD_HF_FRACTION: float = 0.05        # HF energy fraction (>100 Hz)
EMG_MODERATE_HF_FRACTION: float = 0.15
EMG_SEVERE_HF_FRACTION: float = 0.35
EMG_KURTOSIS_MILD: float = 5.0
EMG_KURTOSIS_MODERATE: float = 10.0
EMG_KURTOSIS_SEVERE: float = 20.0

# ---------------------------------------------------------------------------
# Rhythm classification
# ---------------------------------------------------------------------------

HR_TACH_MIN_BPM: float = 100.0
HR_BRADY_MAX_BPM: float = 60.0
AF_CV_MIN: float = 0.10
AF_PNN50_MIN: float = 0.15

# ---------------------------------------------------------------------------
# VT / VF acute event detection
# ---------------------------------------------------------------------------

VT_HR_MIN_BPM: float = 120.0
VT_SUSTAINED_BEATS: int = 4
VF_RR_CV_MIN: float = 0.30
VF_ENTROPY_MIN: float = 0.65
ALERT_LATENCY_TARGET_S: float = 3.0

# ---------------------------------------------------------------------------
# Chunk / streaming processing
# ---------------------------------------------------------------------------

CHUNK_DURATION_S: float = 60.0
CHUNK_OVERLAP_S: float = 1.0
EDGE_CHUNK_DURATION_S: float = 2.0      # Smaller chunk for edge/streaming mode
EDGE_CHUNK_OVERLAP_S: float = 0.3

# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

MAX_PLOT_SAMPLES: int = 20_000
PLOT_COLORS = {
    "raw":       "#4e79a7",
    "filtered":  "#1a3a6b",
    "morph":     "#2196f3",
    "peaks":     "#e15759",
    "artifact":  "#f28e2b",
    "baseline":  "#59a14f",
    "threshold": "#76b7b2",
    "rejected":  "#bab0ac",
    "ppg":       "#9c27b0",
    "emg":       "#ff5722",
}

# ---------------------------------------------------------------------------
# UI theme
# ---------------------------------------------------------------------------

THEME = {
    "primary":   "#1a3a6b",
    "secondary": "#4e79a7",
    "accent":    "#2196f3",
    "good":      "#43a047",
    "warn":      "#fb8c00",
    "critical":  "#e53935",
    "bg":        "#ffffff",
    "text":      "#1c1c1c",
}

# ---------------------------------------------------------------------------
# Algorithm comparison
# ---------------------------------------------------------------------------

COMPARISON_ALGORITHMS = ["pantompkins", "hamilton", "christov"]

# ---------------------------------------------------------------------------
# Modalities
# ---------------------------------------------------------------------------

SUPPORTED_MODALITIES = ["ecg", "ppg", "emg"]
