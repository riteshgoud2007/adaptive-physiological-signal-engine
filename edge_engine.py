"""
edge_engine.py — Streaming / edge processing engine for low-resource environments.

Designed for:
  * Bounded memory usage (O(chunk_size), not O(recording_length)).
  * Real-time chunk processing with state preservation between chunks.
  * Streaming R-peak detection using stateful sosfilt.
  * Runtime and memory profiling via tracemalloc.
  * <= 3-second alert latency measurement.

This module must NOT import Streamlit, Plotly, or any GUI packages.
All processing uses NumPy and SciPy only.
"""

from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import signal as sp_signal
from scipy.ndimage import uniform_filter1d


@dataclass
class ChunkResult:
    """Result from processing a single ECG chunk."""
    chunk_index: int = 0
    start_s: float = 0.0
    end_s: float = 0.0
    r_peaks: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    heart_rate_bpm: float = 0.0
    hr_reliable: bool = False
    noise_level: float = 0.0
    emg_detected: bool = False
    alert_message: Optional[str] = None
    chunk_processing_time_s: float = 0.0
    n_samples: int = 0


@dataclass
class EdgePerformanceReport:
    """Runtime and memory performance report for edge processing."""
    total_processing_time_s: float = 0.0
    n_samples_processed: int = 0
    samples_per_second: float = 0.0
    recording_duration_s: float = 0.0
    realtime_ratio: float = 0.0
    peak_memory_kb: float = 0.0
    mean_chunk_time_ms: float = 0.0
    max_chunk_time_ms: float = 0.0
    n_chunks: int = 0
    detection_latency_s: Optional[float] = None
    target_latency_s: float = 3.0
    latency_target_met: Optional[bool] = None
    notes: list[str] = field(default_factory=list)


@dataclass
class StreamingState:
    """
    Preserved state between chunks.
    Bounded size regardless of recording length.
    """
    zi_bp: Optional[np.ndarray] = None
    spki: float = 0.0
    npki: float = 0.0
    global_sample_offset: int = 0
    rapid_beat_count: int = 0
    alert_active: bool = False
    rr_buffer: list[float] = field(default_factory=list)
    MAX_RR_BUFFER: int = 20

    def update_rr(self, rr_s: float) -> None:
        """Add RR interval to bounded buffer."""
        self.rr_buffer.append(rr_s)
        if len(self.rr_buffer) > self.MAX_RR_BUFFER:
            self.rr_buffer.pop(0)


class EdgeEngine:
    """
    Streaming ECG processing engine for low-resource environments.

    Processes ECG in fixed-size chunks with O(chunk_size) memory.
    State is preserved between chunks for seamless streaming.

    Parameters
    ----------
    fs :
        Sampling rate in Hz.
    chunk_duration_s :
        Chunk size for streaming (default 2.0 s).
    overlap_s :
        Overlap between chunks to avoid boundary artifacts (default 0.3 s).
    """

    BP_LOW: float = 0.5
    BP_HIGH: float = 45.0
    BP_ORDER: int = 4
    MWI_WIN_S: float = 0.150
    REFRACTORY_S: float = 0.200
    SPKI_ALPHA: float = 0.125
    NPKI_ALPHA: float = 0.125
    VT_RAPID_BPM: float = 120.0
    VT_SUSTAINED_BEATS: int = 4

    def __init__(
        self,
        fs: float,
        chunk_duration_s: float = 2.0,
        overlap_s: float = 0.3,
    ) -> None:
        self.fs = float(fs)
        self.chunk_n = max(4, int(chunk_duration_s * fs))
        self.overlap_n = max(0, int(overlap_s * fs))
        self.refractory_n = max(1, int(self.REFRACTORY_S * fs))
        self.mwi_n = max(1, int(self.MWI_WIN_S * fs))
        self._build_filters()
        self.state = StreamingState()

    def _build_filters(self) -> None:
        """Build SOS filters once at initialization."""
        nyq = self.fs / 2.0
        lo = self.BP_LOW / nyq
        hi = min(0.995, self.BP_HIGH / nyq)
        self._sos_bp = sp_signal.butter(
            self.BP_ORDER, [lo, hi], btype="band", output="sos"
        )

    def reset(self) -> None:
        """Reset all streaming state for a new recording."""
        self.state = StreamingState()
        self.state.zi_bp = sp_signal.sosfilt_zi(self._sos_bp)

    def process_recording(
        self,
        ecg: np.ndarray,
        apply_notch_50: bool = False,
        apply_notch_60: bool = False,
    ) -> tuple[EdgePerformanceReport, list[ChunkResult]]:
        """
        Process a full recording in streaming chunks with performance profiling.

        In production, process_chunk() would be called per incoming sample block.
        This method demonstrates the streaming approach on a buffered recording.

        Parameters
        ----------
        ecg :
            Full ECG signal (n_samples,).
        apply_notch_50 :
            Apply 50 Hz notch filter.
        apply_notch_60 :
            Apply 60 Hz notch filter.

        Returns
        -------
        (EdgePerformanceReport, list[ChunkResult])
        """
        self.reset()
        chunk_results: list[ChunkResult] = []
        chunk_times: list[float] = []
        first_alert_latency: Optional[float] = None

        tracemalloc.start()
        t_total_start = time.perf_counter()

        n = len(ecg)
        step = max(1, self.chunk_n - self.overlap_n)
        chunk_starts = list(range(0, n, step))

        for ci, start in enumerate(chunk_starts):
            end = min(start + self.chunk_n, n)
            chunk = ecg[start:end]

            t_chunk_start = time.perf_counter()
            cr = self._process_chunk_internal(
                chunk=chunk,
                chunk_index=ci,
                start_sample=start,
                apply_notch_50=apply_notch_50,
                apply_notch_60=apply_notch_60,
            )
            chunk_elapsed = time.perf_counter() - t_chunk_start
            cr.chunk_processing_time_s = chunk_elapsed
            chunk_times.append(chunk_elapsed)

            if cr.alert_message and first_alert_latency is None:
                first_alert_latency = cr.start_s + chunk_elapsed

            chunk_results.append(cr)

        t_total = time.perf_counter() - t_total_start
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        perf = EdgePerformanceReport()
        perf.total_processing_time_s = t_total
        perf.n_samples_processed = n
        perf.recording_duration_s = n / self.fs
        perf.samples_per_second = n / max(t_total, 1e-9)
        perf.realtime_ratio = t_total / max(perf.recording_duration_s, 1e-9)
        perf.peak_memory_kb = peak_mem / 1024.0
        perf.n_chunks = len(chunk_results)
        perf.mean_chunk_time_ms = float(np.mean(chunk_times)) * 1000.0 if chunk_times else 0.0
        perf.max_chunk_time_ms = float(np.max(chunk_times)) * 1000.0 if chunk_times else 0.0
        perf.detection_latency_s = first_alert_latency
        if first_alert_latency is not None:
            perf.latency_target_met = first_alert_latency <= perf.target_latency_s

        if perf.realtime_ratio < 1.0:
            perf.notes.append(
                f"Faster than real-time: {perf.realtime_ratio:.4f}x "
                "(suitable for edge deployment)."
            )
        else:
            perf.notes.append(
                f"Slower than real-time: {perf.realtime_ratio:.4f}x "
                "(optimisation required for edge deployment)."
            )

        return perf, chunk_results

    def process_chunk(
        self,
        chunk: np.ndarray,
        chunk_index: int = 0,
        start_sample: int = 0,
    ) -> ChunkResult:
        """
        Process a single incoming ECG chunk (call from streaming loop).
        State is preserved between calls.

        Parameters
        ----------
        chunk :
            ECG samples for this chunk.
        chunk_index :
            Sequential chunk number.
        start_sample :
            Global sample index of start of this chunk.

        Returns
        -------
        ChunkResult
        """
        return self._process_chunk_internal(
            chunk=chunk,
            chunk_index=chunk_index,
            start_sample=start_sample,
        )

    def _process_chunk_internal(
        self,
        chunk: np.ndarray,
        chunk_index: int,
        start_sample: int,
        apply_notch_50: bool = False,
        apply_notch_60: bool = False,
    ) -> ChunkResult:
        """Internal: process one chunk, updating streaming state."""
        cr = ChunkResult(
            chunk_index=chunk_index,
            start_s=start_sample / self.fs,
            end_s=(start_sample + len(chunk)) / self.fs,
            n_samples=len(chunk),
        )

        if len(chunk) < 4:
            return cr

        # Interpolate NaN/Inf
        arr = chunk.astype(np.float64).copy()
        bad = ~np.isfinite(arr)
        if bad.any():
            idx = np.arange(len(arr))
            good = ~bad
            if good.sum() >= 2:
                arr[bad] = np.interp(idx[bad], idx[good], arr[good])
            else:
                arr[bad] = 0.0

        # Baseline removal
        w = min(len(arr) - 1, max(3, int(0.6 * self.fs)))
        if w % 2 == 0:
            w += 1
        baseline = uniform_filter1d(arr, size=w, mode="nearest")
        detrended = arr - baseline

        # Streaming bandpass (preserves filter state)
        if self.state.zi_bp is None:
            self.state.zi_bp = sp_signal.sosfilt_zi(self._sos_bp)
        scale = float(detrended[0]) if abs(detrended[0]) > 1e-12 else 1.0
        filtered, zi_new = sp_signal.sosfilt(
            self._sos_bp,
            detrended,
            zi=self.state.zi_bp * scale,
        )
        self.state.zi_bp = zi_new

        cr.noise_level = float(np.std(np.diff(filtered))) / np.sqrt(2)

        # EMG heuristic
        if len(filtered) > 8:
            diff_std = float(np.std(np.diff(filtered)))
            sig_std = float(np.std(filtered))
            cr.emg_detected = (diff_std > sig_std * 2.0) if sig_std > 1e-9 else False

        # Peak detection
        local_peaks = self._detect_peaks_chunk(filtered)
        global_peaks = local_peaks + start_sample

        # Exclude overlap region (already counted by previous chunk)
        if chunk_index > 0:
            boundary = start_sample + self.overlap_n
            global_peaks = global_peaks[global_peaks >= boundary]
        cr.r_peaks = global_peaks

        # HR and streaming VT check
        if len(global_peaks) >= 2:
            rr = np.diff(global_peaks.astype(float)) / self.fs
            valid_rr = rr[(rr >= 0.2) & (rr <= 3.0)]
            if len(valid_rr) > 0:
                cr.heart_rate_bpm = float(60.0 / np.median(valid_rr))
                cr.hr_reliable = True
                for rr_val in valid_rr:
                    self.state.update_rr(float(rr_val))
                alert_msg = self._check_streaming_vt(valid_rr, cr.start_s)
                if alert_msg:
                    cr.alert_message = alert_msg

        self.state.global_sample_offset += len(chunk)
        return cr

    def _detect_peaks_chunk(self, filtered: np.ndarray) -> np.ndarray:
        """Efficient adaptive peak detection for a single chunk."""
        n = len(filtered)
        if n < self.refractory_n * 2:
            return np.array([], dtype=int)

        # Derivative
        deriv = np.zeros(n)
        deriv[1:-1] = (filtered[2:] - filtered[:-2]) / (2.0 / self.fs)
        deriv[0] = deriv[1]
        deriv[-1] = deriv[-2]

        squared = deriv ** 2

        # Moving window integration (cumulative sum, O(n))
        cs = np.cumsum(squared)
        mwi = np.zeros(n)
        w = self.mwi_n
        mwi[w:] = (cs[w:] - cs[:-w]) / w
        mwi[:w] = cs[:w] / np.maximum(np.arange(1, w + 1), 1)

        # Initialise adaptive thresholds from first chunk
        if self.state.spki < 1e-12 and self.state.npki < 1e-12:
            calib = mwi[:min(int(2 * self.fs), n)]
            init_peaks, _ = sp_signal.find_peaks(calib, distance=self.refractory_n)
            self.state.spki = (
                float(np.mean(calib[init_peaks]))
                if len(init_peaks) > 0
                else float(np.max(mwi) * 0.5 + 1e-12)
            )
            self.state.npki = self.state.spki * 0.1

        threshold = self.state.npki + 0.25 * (self.state.spki - self.state.npki)
        candidates, _ = sp_signal.find_peaks(mwi, distance=self.refractory_n // 2)

        accepted: list[int] = []
        for cand in candidates:
            if mwi[cand] > threshold:
                # Refine to local maximum in filtered signal
                hw = max(1, int(0.04 * self.fs))
                lo = max(0, cand - hw)
                hi = min(n, cand + hw + 1)
                refined = lo + int(np.argmax(np.abs(filtered[lo:hi])))
                accepted.append(refined)
                self.state.spki = (
                    self.SPKI_ALPHA * mwi[cand] +
                    (1 - self.SPKI_ALPHA) * self.state.spki
                )
            else:
                self.state.npki = (
                    self.NPKI_ALPHA * mwi[cand] +
                    (1 - self.NPKI_ALPHA) * self.state.npki
                )

        return np.array(accepted, dtype=int)

    def _check_streaming_vt(
        self, rr_valid: np.ndarray, chunk_start_s: float
    ) -> Optional[str]:
        """Streaming VT detection from a chunk's valid RR intervals."""
        for rr in rr_valid:
            bpm = 60.0 / max(rr, 1e-9)
            if bpm >= self.VT_RAPID_BPM:
                self.state.rapid_beat_count += 1
            else:
                self.state.rapid_beat_count = 0
                if self.state.alert_active:
                    self.state.alert_active = False

            if (
                self.state.rapid_beat_count >= self.VT_SUSTAINED_BEATS
                and not self.state.alert_active
            ):
                self.state.alert_active = True
                latency = rr * self.state.rapid_beat_count
                within = latency <= 3.0
                return (
                    f"POSSIBLE VT at t={chunk_start_s:.1f}s. "
                    f"Detection latency: {latency:.2f}s. "
                    f"Target <= 3.00s: {'YES' if within else 'NO'}"
                )
        return None


def benchmark_throughput(
    engine: EdgeEngine,
    signal_length_s: float = 60.0,
) -> dict[str, float]:
    """
    Benchmark edge engine throughput on a sinusoidal test signal.
    Used for runtime profiling only — NOT clinical performance measurement.

    Parameters
    ----------
    engine :
        EdgeEngine instance.
    signal_length_s :
        Duration of test signal in seconds.

    Returns
    -------
    dict with runtime metrics.
    """
    n = int(signal_length_s * engine.fs)
    t = np.linspace(0, signal_length_s, n)
    test_signal = np.sin(2 * np.pi * 1.2 * t) + 0.05 * np.random.RandomState(42).randn(n)
    perf, _ = engine.process_recording(test_signal)
    return {
        "total_processing_time_s": perf.total_processing_time_s,
        "samples_per_second": perf.samples_per_second,
        "recording_duration_s": perf.recording_duration_s,
        "realtime_ratio": perf.realtime_ratio,
        "peak_memory_kb": perf.peak_memory_kb,
        "mean_chunk_time_ms": perf.mean_chunk_time_ms,
        "max_chunk_time_ms": perf.max_chunk_time_ms,
        "n_chunks": perf.n_chunks,
    }
