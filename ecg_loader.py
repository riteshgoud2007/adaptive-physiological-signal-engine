"""
ecg_loader.py — ECG file loading for the Adaptive ECG Signal Intelligence Engine.

Supports:
  * CSV (time+ecg, ecg-only, multi-channel)
  * TXT (whitespace-separated or single-column)
  * NPY / NPZ (NumPy arrays)
  * WFDB / PhysioNet (.dat + .hea) — optional, degrades gracefully
"""

from __future__ import annotations

import io
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import wfdb  # type: ignore
    WFDB_AVAILABLE = True
except ImportError:
    WFDB_AVAILABLE = False

from config import DEFAULT_FS, MIN_FS, MAX_FS, MIN_SAMPLES, MIN_DURATION_S


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ECGRecord:
    """Holds a loaded ECG recording and its metadata."""

    signal: np.ndarray          # shape (n_samples,) — single chosen channel
    fs: float                   # sampling rate in Hz
    channel_name: str = "ECG"
    all_channels: Optional[np.ndarray] = None   # shape (n_samples, n_ch)
    channel_names: list[str] = field(default_factory=list)
    time_axis: Optional[np.ndarray] = None      # seconds
    source_path: str = ""
    n_samples: int = 0
    duration_s: float = 0.0
    load_time_s: float = 0.0
    annotations: Optional[np.ndarray] = None    # sample indices (WFDB)
    annotation_symbols: Optional[list[str]] = None
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.n_samples = len(self.signal)
        self.duration_s = self.n_samples / self.fs
        if self.time_axis is None:
            self.time_axis = np.arange(self.n_samples) / self.fs


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def load_ecg(
    file_obj,
    filename: str,
    fs_override: Optional[float] = None,
    channel_index: int = 0,
) -> ECGRecord:
    """
    Load an ECG from any supported format.

    Parameters
    ----------
    file_obj :
        File-like object (from Streamlit uploader) or a path string.
    filename :
        Original filename (used to determine format).
    fs_override :
        If provided, use this sampling rate instead of any inferred value.
    channel_index :
        Which channel to select from multi-channel files.

    Returns
    -------
    ECGRecord
    """
    t0 = time.perf_counter()
    ext = Path(filename).suffix.lower()

    if ext in (".csv", ".txt"):
        record = _load_csv_txt(file_obj, filename, fs_override, channel_index)
    elif ext == ".npy":
        record = _load_npy(file_obj, filename, fs_override, channel_index)
    elif ext == ".npz":
        record = _load_npz(file_obj, filename, fs_override, channel_index)
    elif ext in (".dat", ".hea"):
        record = _load_wfdb(file_obj, filename, fs_override, channel_index)
    else:
        # Try CSV as a fallback
        record = _load_csv_txt(file_obj, filename, fs_override, channel_index)

    if fs_override is not None:
        record.fs = float(fs_override)
        record.duration_s = record.n_samples / record.fs
        record.time_axis = np.arange(record.n_samples) / record.fs

    record.load_time_s = time.perf_counter() - t0
    _validate(record)
    return record


# ---------------------------------------------------------------------------
# Format-specific loaders
# ---------------------------------------------------------------------------

def _load_csv_txt(
    file_obj,
    filename: str,
    fs_override: Optional[float],
    channel_index: int,
) -> ECGRecord:
    """Load CSV or TXT ECG file."""
    warnings: list[str] = []

    # Read raw bytes / text
    if hasattr(file_obj, "read"):
        raw = file_obj.read()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
    else:
        with open(file_obj, "r", errors="replace") as f:
            raw = f.read()

    lines = raw.strip().splitlines()
    if not lines:
        raise ValueError("File is empty.")

    # ------------------------------------------------------------------ #
    # Detect delimiter and header
    # ------------------------------------------------------------------ #
    delimiters = [",", "\t", " ", ";"]
    delimiter = ","
    for d in delimiters:
        if d in lines[0]:
            delimiter = d
            break

    # Does the first line look like a header?
    first = lines[0].strip()
    has_header = False
    try:
        float(first.split(delimiter)[0].strip())
    except ValueError:
        has_header = True

    if has_header:
        header_line = lines[0]
        data_lines = lines[1:]
        col_names = [c.strip().strip('"') for c in header_line.split(delimiter)]
    else:
        data_lines = lines
        col_names = []

    if not data_lines:
        raise ValueError("No data rows found in file.")

    # Parse numeric data
    rows = []
    for ln in data_lines:
        stripped = ln.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split(delimiter)
        try:
            row = [float(p.strip()) if p.strip() not in ("", "nan", "NaN", "NA", "na")
                   else float("nan")
                   for p in parts]
            rows.append(row)
        except ValueError:
            continue

    if not rows:
        raise ValueError("Could not parse any numeric data from the file.")

    data = np.array(rows, dtype=np.float64)

    # ------------------------------------------------------------------ #
    # Identify time column and ECG column(s)
    # ------------------------------------------------------------------ #
    time_col: Optional[np.ndarray] = None
    n_cols = data.shape[1] if data.ndim == 2 else 1

    if data.ndim == 1:
        # Single column — ECG only
        ecg_data = data.reshape(-1, 1)
        ecg_col_names = ["ECG"]
    else:
        # Check if first column is monotone increasing (time axis)
        first_col = data[:, 0]
        diffs = np.diff(first_col)
        is_time = np.all(diffs > 0) and (first_col[0] >= 0)

        if is_time and col_names and col_names[0].lower() in ("time", "t", "seconds", "timestamp", "sample"):
            is_time = True
        elif is_time and n_cols >= 2:
            # Heuristic: if values are very small and monotone, treat as time
            is_time = first_col[-1] < 1e6

        if is_time and data.ndim == 2 and data.shape[1] > 1:
            time_col = data[:, 0]
            ecg_data = data[:, 1:]
            ecg_col_names = col_names[1:] if len(col_names) > 1 else [
                f"Ch{i}" for i in range(ecg_data.shape[1])
            ]
        else:
            ecg_data = data
            ecg_col_names = col_names if col_names else [
                f"Ch{i}" for i in range(data.shape[1])
            ]

    # Infer FS from time column
    fs_inferred: Optional[float] = None
    if time_col is not None and len(time_col) > 1:
        dt_values = np.diff(time_col)
        median_dt = float(np.median(dt_values))
        if median_dt > 0:
            fs_inferred = round(1.0 / median_dt, 2)

    fs = float(fs_override) if fs_override is not None else (fs_inferred or DEFAULT_FS)
    if fs_inferred and fs_override is None:
        warnings.append(f"Sampling rate inferred from time column: {fs_inferred:.1f} Hz")
    elif fs_inferred is None and fs_override is None:
        warnings.append(f"Sampling rate not found in file; using default {DEFAULT_FS} Hz")

    # Select channel
    n_channels = ecg_data.shape[1] if ecg_data.ndim == 2 else 1
    ch_idx = min(channel_index, n_channels - 1)
    signal = ecg_data[:, ch_idx].copy() if ecg_data.ndim == 2 else ecg_data.flatten()
    chosen_name = ecg_col_names[ch_idx] if ch_idx < len(ecg_col_names) else f"Ch{ch_idx}"

    return ECGRecord(
        signal=signal,
        fs=fs,
        channel_name=chosen_name,
        all_channels=ecg_data if ecg_data.ndim == 2 else ecg_data.reshape(-1, 1),
        channel_names=ecg_col_names,
        time_axis=time_col,
        source_path=filename,
        warnings=warnings,
    )


def _load_npy(
    file_obj,
    filename: str,
    fs_override: Optional[float],
    channel_index: int,
) -> ECGRecord:
    """Load .npy NumPy array."""
    if hasattr(file_obj, "read"):
        data = np.load(io.BytesIO(file_obj.read()))
    else:
        data = np.load(file_obj)

    data = data.astype(np.float64)
    if data.ndim == 1:
        signal = data
        all_ch = data.reshape(-1, 1)
        ch_names = ["ECG"]
    elif data.ndim == 2:
        ch_idx = min(channel_index, data.shape[1] - 1)
        signal = data[:, ch_idx].copy()
        all_ch = data
        ch_names = [f"Ch{i}" for i in range(data.shape[1])]
    else:
        raise ValueError(f"Unexpected array shape: {data.shape}")

    fs = float(fs_override) if fs_override is not None else DEFAULT_FS
    return ECGRecord(
        signal=signal,
        fs=fs,
        channel_name=ch_names[min(channel_index, len(ch_names) - 1)],
        all_channels=all_ch,
        channel_names=ch_names,
        source_path=filename,
        warnings=[f"No metadata in .npy; using {'override' if fs_override else 'default'} FS={fs:.0f} Hz"],
    )


def _load_npz(
    file_obj,
    filename: str,
    fs_override: Optional[float],
    channel_index: int,
) -> ECGRecord:
    """Load .npz NumPy archive. Expects keys: 'ecg', 'signal', or first array."""
    if hasattr(file_obj, "read"):
        npz = np.load(io.BytesIO(file_obj.read()))
    else:
        npz = np.load(file_obj)

    # Find signal key
    signal_key = None
    for k in ("ecg", "signal", "ecg_signal", "data"):
        if k in npz:
            signal_key = k
            break
    if signal_key is None:
        signal_key = list(npz.keys())[0]

    data = npz[signal_key].astype(np.float64)
    if data.ndim == 1:
        signal = data
    elif data.ndim == 2:
        ch_idx = min(channel_index, data.shape[1] - 1)
        signal = data[:, ch_idx].copy()
    else:
        raise ValueError(f"Unexpected array shape: {data.shape}")

    # Try to read FS from archive
    fs_inferred = None
    for k in ("fs", "sampling_rate", "Fs", "SR"):
        if k in npz:
            fs_inferred = float(npz[k])
            break

    fs = float(fs_override) if fs_override is not None else (fs_inferred or DEFAULT_FS)
    warnings = []
    if fs_inferred is None and fs_override is None:
        warnings.append(f"No FS in .npz; using default {DEFAULT_FS} Hz")

    return ECGRecord(
        signal=signal,
        fs=fs,
        channel_name="ECG",
        source_path=filename,
        warnings=warnings,
    )


def _load_wfdb(
    file_obj,
    filename: str,
    fs_override: Optional[float],
    channel_index: int,
) -> ECGRecord:
    """Load WFDB / PhysioNet record. Requires the `wfdb` package."""
    if not WFDB_AVAILABLE:
        raise ImportError(
            "The `wfdb` package is required for WFDB files. "
            "Install with: pip install wfdb"
        )

    # WFDB needs a file path, not a file object
    if hasattr(file_obj, "name"):
        path = Path(file_obj.name)
    elif isinstance(file_obj, (str, Path)):
        path = Path(file_obj)
    else:
        raise ValueError("WFDB loading requires a file path, not an in-memory object.")

    record_path = str(path.with_suffix(""))
    rec = wfdb.rdrecord(record_path)
    fs = float(rec.fs)
    if fs_override is not None:
        fs = float(fs_override)

    data = rec.p_signal.astype(np.float64)  # (n_samples, n_channels)
    ch_idx = min(channel_index, data.shape[1] - 1)
    signal = data[:, ch_idx].copy()
    ch_names = rec.sig_name or [f"Ch{i}" for i in range(data.shape[1])]

    # Try to load annotations
    annotations = None
    ann_symbols = None
    try:
        ann = wfdb.rdann(record_path, "atr")
        annotations = np.array(ann.sample)
        ann_symbols = ann.symbol
    except Exception:
        pass

    return ECGRecord(
        signal=signal,
        fs=fs,
        channel_name=ch_names[ch_idx],
        all_channels=data,
        channel_names=ch_names,
        source_path=str(path),
        annotations=annotations,
        annotation_symbols=ann_symbols,
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate(record: ECGRecord) -> None:
    """Raise ValueError for unacceptable recordings; append warnings for mild issues."""

    if record.n_samples < MIN_SAMPLES:
        raise ValueError(
            f"Recording too short: {record.n_samples} samples "
            f"(minimum {MIN_SAMPLES})."
        )

    if not (MIN_FS <= record.fs <= MAX_FS):
        raise ValueError(
            f"Invalid sampling rate: {record.fs:.1f} Hz "
            f"(expected {MIN_FS}–{MAX_FS} Hz)."
        )

    if record.duration_s < MIN_DURATION_S:
        raise ValueError(
            f"Recording too short: {record.duration_s:.2f} s "
            f"(minimum {MIN_DURATION_S} s)."
        )

    # Check for all-NaN
    if np.all(np.isnan(record.signal)):
        raise ValueError("Signal contains only NaN values.")

    # Replace Inf
    inf_mask = ~np.isfinite(record.signal)
    if inf_mask.any():
        record.signal[inf_mask] = np.nan
        record.warnings.append(
            f"{inf_mask.sum()} Inf values replaced with NaN."
        )


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def get_channel_names(record: ECGRecord) -> list[str]:
    """Return the list of available channel names."""
    return record.channel_names or ["ECG"]


def select_channel(record: ECGRecord, channel_index: int) -> ECGRecord:
    """Return a new ECGRecord with the specified channel selected."""
    if record.all_channels is None:
        return record
    ch_idx = min(channel_index, record.all_channels.shape[1] - 1)
    new_signal = record.all_channels[:, ch_idx].copy()
    ch_names = record.channel_names
    chosen = ch_names[ch_idx] if ch_idx < len(ch_names) else f"Ch{ch_idx}"
    import dataclasses
    return dataclasses.replace(
        record,
        signal=new_signal,
        channel_name=chosen,
        n_samples=len(new_signal),
    )
