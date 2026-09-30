"""
tests/test_loader.py — Unit tests for ecg_loader.py
"""
import io
import sys
import os
import numpy as np
import pytest

# Allow importing project modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ecg_loader import load_ecg, ECGRecord


def make_csv_bytes(rows: list[str], header: str = "") -> io.BytesIO:
    content = (header + "\n" + "\n".join(rows)) if header else "\n".join(rows)
    return io.BytesIO(content.encode("utf-8"))


class TestCSVLoading:
    def test_ecg_only_single_column(self):
        """ECG-only CSV (single column, no header)."""
        values = [str(v) for v in np.sin(np.linspace(0, 2*np.pi, 500)).tolist()]
        buf = make_csv_bytes(values)
        record = load_ecg(buf, "test.csv", fs_override=250.0)
        assert isinstance(record, ECGRecord)
        assert record.n_samples == 500
        assert record.fs == 250.0
        assert np.all(np.isfinite(record.signal))

    def test_time_ecg_two_columns(self):
        """Time + ECG two-column CSV (3 seconds to pass min duration)."""
        t = np.linspace(0, 3, 1080)  # 3 seconds at 360 Hz
        ecg = np.sin(2 * np.pi * 2 * t)
        lines = ["time,ecg"] + [f"{ti:.4f},{ei:.6f}" for ti, ei in zip(t, ecg)]
        buf = io.BytesIO("\n".join(lines).encode())
        record = load_ecg(buf, "ecg.csv")
        assert record.n_samples == 1080
        assert record.fs == pytest.approx(360.0, rel=0.05)

    def test_multichannel_csv(self):
        """Multi-channel CSV — select channel 1 (3 seconds)."""
        t = np.linspace(0, 3, 750)  # 3 seconds at 250 Hz
        ch0 = np.sin(t)
        ch1 = np.cos(t)
        lines = ["time,Lead_I,Lead_II"] + [
            f"{t[i]:.4f},{ch0[i]:.4f},{ch1[i]:.4f}" for i in range(750)
        ]
        buf = io.BytesIO("\n".join(lines).encode())
        record = load_ecg(buf, "multi.csv", channel_index=1)
        assert record.channel_name == "Lead_II"
        assert np.allclose(record.signal, ch1, atol=1e-3)

    def test_short_recording_rejected(self):
        """Very short recording should raise ValueError."""
        values = ["0.1", "0.2", "0.3"]
        buf = make_csv_bytes(values)
        with pytest.raises(ValueError):
            load_ecg(buf, "tiny.csv", fs_override=250.0)

    def test_empty_file_rejected(self):
        """Empty file should raise ValueError."""
        buf = io.BytesIO(b"")
        with pytest.raises(ValueError):
            load_ecg(buf, "empty.csv", fs_override=250.0)

    def test_invalid_sampling_rate_rejected(self):
        """Sampling rate below minimum should raise ValueError."""
        values = [str(v) for v in np.random.randn(500).tolist()]
        buf = make_csv_bytes(values)
        with pytest.raises(ValueError):
            load_ecg(buf, "test.csv", fs_override=1.0)  # below 50 Hz

    def test_nan_handling(self):
        """NaN values should be handled gracefully (3 seconds at 250 Hz)."""
        values = ["0.1", "nan", "0.2", "0.3"] * 188  # ~752 samples > 3 sec at 250 Hz
        buf = make_csv_bytes(values)
        record = load_ecg(buf, "nan_test.csv", fs_override=250.0)
        # Should succeed and have loaded most samples
        assert record.n_samples > 0

    def test_npy_loading(self):
        """NPY array loading."""
        arr = np.random.randn(1000)
        buf = io.BytesIO()
        np.save(buf, arr)
        buf.seek(0)
        record = load_ecg(buf, "test.npy", fs_override=360.0)
        assert record.n_samples == 1000
        assert record.fs == 360.0
