"""
adaptive_engine.py — Core signal processing and adaptive R-peak detection engine.

Pipeline
--------
  Raw ECG
    → Preprocessing (baseline removal, detrending)
    → Powerline notch (if detected)
    → Bandpass Butterworth SOS filter
    → Pan-Tompkins MWI
    → Adaptive threshold R-peak detection
    → Search-back
    → Peak refinement
    → RR intervals → Heart rate
    → Reliability assessment
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import signal as sp_signal

from config import (
    BP_LOW_HZ, BP_HIGH_HZ, BP_ORDER,
    NOTCH_Q, NOTCH_50_HZ, NOTCH_60_HZ,
    MWI_WINDOW_S, REFRACTORY_S, SEARCH_BACK_S,
    INITIAL_CALIB_S, SPKI_ALPHA, NPKI_ALPHA,
    CHUNK_DURATION_S, CHUNK_OVERLAP_S,
    BASELINE_WINDOW_S,
)
from quality_engine import QualityEngine, QualityLevel, AdaptiveMonitorReport


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class PeakDetail:
    """Detailed information about a single detected R-peak."""
    index: int                  # sample index in filtered signal
    time_s: float               # time in seconds
    amplitude: float            # filtered signal amplitude at peak
    raw_amplitude: float        # raw signal amplitude at peak
    rr_prev_s: Optional[float]  # RR interval to previous peak
    rr_next_s: Optional[float]  # RR interval to next peak
    qrs_strength: float         # normalised 0–1
    noise_level: float          # local noise estimate
    search_back: bool           # found via search-back?
    refractory_pass: bool       # passed refractory check?
    refined: bool               # local peak refinement applied?
    accepted: bool              # accepted or rejected?
    reject_reason: str = ""     # reason if rejected
    segment_noise: float = 0.0  # segment-level noise RMS
    segment_reliability: float = 1.0


@dataclass
class ProcessingResult:
    """Complete output of the adaptive processing pipeline."""
    # Signals
    raw: np.ndarray = field(default_factory=lambda: np.array([]))
    detrended: np.ndarray = field(default_factory=lambda: np.array([]))
    filtered: np.ndarray = field(default_factory=lambda: np.array([]))
    derivative: np.ndarray = field(default_factory=lambda: np.array([]))
    squared: np.ndarray = field(default_factory=lambda: np.array([]))
    mwi: np.ndarray = field(default_factory=lambda: np.array([]))
    threshold_signal: np.ndarray = field(default_factory=lambda: np.array([]))
    threshold_noise: np.ndarray = field(default_factory=lambda: np.array([]))

    # Peaks
    r_peaks: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    rejected_peaks: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    peak_details: list[PeakDetail] = field(default_factory=list)

    # Intervals
    rr_intervals_s: np.ndarray = field(default_factory=lambda: np.array([]))
    heart_rate_bpm: float = 0.0
    hr_series_bpm: np.ndarray = field(default_factory=lambda: np.array([]))
    hr_times_s: np.ndarray = field(default_factory=lambda: np.array([]))

    # Quality
    detection_reliability: float = 0.0
    overall_quality: QualityLevel = QualityLevel.GOOD
    monitor_report: Optional[AdaptiveMonitorReport] = None

    # Processing info
    processing_time_load_s: float = 0.0
    processing_time_preproc_s: float = 0.0
    processing_time_detect_s: float = 0.0
    processing_time_total_s: float = 0.0
    applied_notch_50: bool = False
    applied_notch_60: bool = False
    n_chunks: int = 1
    warnings: list[str] = field(default_factory=list)
    fs: float = 360.0


# ---------------------------------------------------------------------------
# Adaptive processing engine
# ---------------------------------------------------------------------------

class AdaptiveECGEngine:
    """
    Full adaptive ECG processing pipeline.

    Usage::

        engine = AdaptiveECGEngine(fs=360.0)
        result = engine.process(raw_ecg)
    """

    def __init__(self, fs: float) -> None:
        self.fs = float(fs)
        self._refractory_n = int(REFRACTORY_S * self.fs)
        self._mwi_n = max(1, int(MWI_WINDOW_S * self.fs))
        self._search_back_n = int(SEARCH_BACK_S * self.fs)
        self._calib_n = int(INITIAL_CALIB_S * self.fs)
        self._quality_engine = QualityEngine(self.fs)

        # Build filters once
        self._sos_bp = sp_signal.butter(
            BP_ORDER,
            [BP_LOW_HZ / (self.fs / 2), min(0.99, BP_HIGH_HZ / (self.fs / 2))],
            btype="band",
            output="sos",
        )
        self._sos_notch_50 = sp_signal.iirnotch(
            NOTCH_50_HZ, NOTCH_Q, self.fs
        )
        self._sos_notch_60 = sp_signal.iirnotch(
            NOTCH_60_HZ, NOTCH_Q, self.fs
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self,
        raw: np.ndarray,
        monitor_report: Optional[AdaptiveMonitorReport] = None,
    ) -> ProcessingResult:
        """
        Run the full adaptive pipeline on a raw ECG array.

        Parameters
        ----------
        raw :
            Raw ECG signal (n_samples,).
        monitor_report :
            Pre-computed AdaptiveMonitorReport (if None, will be computed).

        Returns
        -------
        ProcessingResult
        """
        t_total = time.perf_counter()
        result = ProcessingResult(fs=self.fs)
        result.raw = raw.copy()

        # ---- Quality monitoring ---------------------------------------- #
        if monitor_report is None:
            monitor_report = self._quality_engine.monitor_segments(raw)
        result.monitor_report = monitor_report
        result.applied_notch_50 = monitor_report.powerline_50hz
        result.applied_notch_60 = monitor_report.powerline_60hz

        # ---- Preprocessing --------------------------------------------- #
        t_pp = time.perf_counter()
        interpolated = _interpolate_nans(raw)
        detrended = _remove_baseline(interpolated, self.fs)
        result.detrended = detrended

        # Notch filter (adaptive: only when powerline interference detected)
        notched = detrended.copy()
        if monitor_report.powerline_50hz:
            b50, a50 = self._sos_notch_50
            notched = sp_signal.filtfilt(b50, a50, notched)
        if monitor_report.powerline_60hz:
            b60, a60 = self._sos_notch_60
            notched = sp_signal.filtfilt(b60, a60, notched)

        # Bandpass
        filtered = sp_signal.sosfiltfilt(self._sos_bp, notched)
        result.filtered = filtered
        result.processing_time_preproc_s = time.perf_counter() - t_pp

        # ---- Chunk detection ------------------------------------------- #
        t_det = time.perf_counter()
        n = len(filtered)
        chunk_n = int(CHUNK_DURATION_S * self.fs)
        overlap_n = int(CHUNK_OVERLAP_S * self.fs)

        if n <= chunk_n + overlap_n:
            # Single chunk
            (
                r_peaks, rejected,
                deriv, squared, mwi,
                thr_sig, thr_noise,
                details,
            ) = self._detect_adaptive(
                filtered, raw, monitor_report, offset=0
            )
            result.n_chunks = 1
        else:
            # Multi-chunk processing
            (
                r_peaks, rejected,
                deriv, squared, mwi,
                thr_sig, thr_noise,
                details,
            ) = self._detect_chunked(
                filtered, raw, monitor_report, chunk_n, overlap_n
            )

        result.derivative = deriv
        result.squared = squared
        result.mwi = mwi
        result.threshold_signal = thr_sig
        result.threshold_noise = thr_noise
        result.peak_details = details
        result.r_peaks = r_peaks
        result.rejected_peaks = rejected
        result.processing_time_detect_s = time.perf_counter() - t_det

        # ---- RR / HR --------------------------------------------------- #
        result = self._compute_hr(result)

        # ---- Reliability ----------------------------------------------- #
        result.detection_reliability = self._compute_reliability(result, monitor_report)

        result.processing_time_total_s = time.perf_counter() - t_total
        return result

    # ------------------------------------------------------------------
    # Internal — preprocessing
    # ------------------------------------------------------------------

    def _detect_chunked(
        self,
        filtered: np.ndarray,
        raw: np.ndarray,
        monitor_report: AdaptiveMonitorReport,
        chunk_n: int,
        overlap_n: int,
    ) -> tuple:
        """Process in overlapping chunks and merge peaks."""
        n = len(filtered)
        all_peaks: list[int] = []
        all_rejected: list[int] = []
        all_details: list[PeakDetail] = []

        # Storage for full-length diagnostic signals
        deriv = np.zeros(n)
        squared = np.zeros(n)
        mwi = np.zeros(n)
        thr_sig = np.zeros(n)
        thr_noise = np.zeros(n)

        chunk_starts = list(range(0, n, chunk_n - overlap_n))
        n_chunks = len(chunk_starts)

        for chunk_idx, start in enumerate(chunk_starts):
            end = min(start + chunk_n, n)
            chunk = filtered[start:end]
            raw_chunk = raw[start:end] if start < len(raw) else raw[start:]

            (
                peaks, rejected,
                d_chunk, sq_chunk, mwi_chunk,
                ts_chunk, tn_chunk,
                details_chunk,
            ) = self._detect_adaptive(
                chunk, raw_chunk, monitor_report, offset=start
            )

            # Fill diagnostic signals
            cn = len(chunk)
            deriv[start:start + cn] = d_chunk[:cn]
            squared[start:start + cn] = sq_chunk[:cn]
            mwi[start:start + cn] = mwi_chunk[:cn]
            thr_sig[start:start + cn] = ts_chunk[:cn]
            thr_noise[start:start + cn] = tn_chunk[:cn]

            # Avoid duplicates in overlap region
            if chunk_idx == 0:
                all_peaks.extend(peaks)
                all_rejected.extend(rejected)
                all_details.extend(details_chunk)
            else:
                boundary = start + overlap_n
                new_peaks = [p for p in peaks if p >= boundary]
                new_rej = [p for p in rejected if p >= boundary]
                new_details = [d for d in details_chunk if d.index >= boundary]
                all_peaks.extend(new_peaks)
                all_rejected.extend(new_rej)
                all_details.extend(new_details)

        peaks_arr = np.array(sorted(set(all_peaks)), dtype=int)
        rej_arr = np.array(sorted(set(all_rejected)), dtype=int)
        return (
            peaks_arr, rej_arr,
            deriv, squared, mwi,
            thr_sig, thr_noise,
            all_details,
        )

    # ------------------------------------------------------------------
    # Internal — Pan-Tompkins adaptive detection
    # ------------------------------------------------------------------

    def _detect_adaptive(
        self,
        filtered: np.ndarray,
        raw: np.ndarray,
        monitor_report: AdaptiveMonitorReport,
        offset: int = 0,
    ) -> tuple:
        """
        Adaptive Pan-Tompkins style R-peak detector.

        Returns
        -------
        (peaks, rejected, deriv, squared, mwi, thr_sig, thr_noise, details)
        All peak indices are in global (offset-corrected) coordinates.
        """
        n = len(filtered)
        if n < self._refractory_n * 2:
            empty = np.array([], dtype=int)
            zeros = np.zeros(n)
            return empty, empty, zeros, zeros, zeros, zeros, zeros, []

        # ---- 1. Derivative --------------------------------------------- #
        deriv = np.zeros(n)
        deriv[1:-1] = (filtered[2:] - filtered[:-2]) / (2.0 / self.fs)
        # Boundary
        deriv[0] = deriv[1]
        deriv[-1] = deriv[-2]

        # ---- 2. Squaring ------------------------------------------------ #
        squared = deriv ** 2

        # ---- 3. Moving-window integration (cumulative sum trick) --------- #
        mwi = self._moving_window_integrate(squared)

        # ---- 4. Adaptive threshold initialisation ----------------------- #
        calib_n = min(self._calib_n, n // 4)
        calib = mwi[:calib_n]
        # Find local peaks in calibration window
        calib_peaks, _ = sp_signal.find_peaks(calib, distance=self._refractory_n)
        if len(calib_peaks) > 0:
            spki = float(np.mean(calib[calib_peaks]))
        else:
            spki = float(np.max(calib)) * 0.5
        npki = spki * 0.1

        thrs = np.zeros(n)   # signal threshold trace
        thrn = np.zeros(n)   # noise threshold trace

        # ---- 5. Candidate detection ------------------------------------- #
        # All local peaks
        min_dist = self._refractory_n
        candidates, props = sp_signal.find_peaks(
            mwi,
            distance=min_dist // 2,   # loose distance for candidates
            prominence=0,
        )

        accepted_peaks: list[int] = []
        rejected_peaks_list: list[int] = []
        details: list[PeakDetail] = []

        last_accepted = -self._refractory_n * 2
        prev_rr = int(self.fs * 0.8)   # guess

        for ci, cand in enumerate(candidates):
            threshold1 = npki + 0.25 * (spki - npki)
            thrs[cand] = threshold1
            thrn[cand] = npki

            is_qrs = mwi[cand] > threshold1

            # Refractory check
            refractory_ok = (cand - last_accepted) >= self._refractory_n

            if is_qrs and refractory_ok:
                # Refine to local maximum in filtered signal
                refined_idx = self._refine_peak(
                    filtered, cand, window=max(1, int(self.fs * 0.04))
                )
                # Update adaptive estimates
                spki = SPKI_ALPHA * mwi[cand] + (1 - SPKI_ALPHA) * spki
                accepted_peaks.append(refined_idx + offset)
                last_accepted = refined_idx
                prev_rr = refined_idx - (accepted_peaks[-2] - offset if len(accepted_peaks) > 1 else refined_idx - int(self.fs * 0.8))

                d = self._build_peak_detail(
                    refined_idx, offset, filtered, raw,
                    mwi, monitor_report, search_back=False,
                    refractory_pass=True, refined=True, accepted=True,
                )
                details.append(d)

            elif not refractory_ok and is_qrs:
                # Refractory violation
                npki = NPKI_ALPHA * mwi[cand] + (1 - NPKI_ALPHA) * npki
                rejected_peaks_list.append(cand + offset)
                d = self._build_peak_detail(
                    cand, offset, filtered, raw,
                    mwi, monitor_report, search_back=False,
                    refractory_pass=False, refined=False, accepted=False,
                    reject_reason="Refractory-period violation",
                )
                details.append(d)

            else:
                # Noise peak
                npki = NPKI_ALPHA * mwi[cand] + (1 - NPKI_ALPHA) * npki
                rejected_peaks_list.append(cand + offset)
                if is_qrs and not refractory_ok:
                    reason = "Refractory-period violation"
                else:
                    reason = "Below adaptive threshold"
                d = self._build_peak_detail(
                    cand, offset, filtered, raw,
                    mwi, monitor_report, search_back=False,
                    refractory_pass=refractory_ok, refined=False, accepted=False,
                    reject_reason=reason,
                )
                if mwi[cand] > threshold1 * 0.5:
                    details.append(d)   # only log near-misses to avoid clutter

        # ---- 6. Search-back --------------------------------------------- #
        accepted_peaks = self._search_back(
            accepted_peaks, mwi, filtered, raw,
            monitor_report, details, offset
        )

        # Fill threshold traces (linear interpolation between peak points)
        _fill_trace(thrs, candidates, n)
        _fill_trace(thrn, candidates, n)

        peaks_arr = np.array(sorted(set(accepted_peaks)), dtype=int)
        rej_arr = np.array(sorted(set(rejected_peaks_list)), dtype=int)
        return (
            peaks_arr, rej_arr,
            deriv, squared, mwi,
            thrs, thrn,
            details,
        )

    def _moving_window_integrate(self, squared: np.ndarray) -> np.ndarray:
        """Efficient moving-window integration via cumulative sum."""
        cs = np.cumsum(squared)
        mwi = np.zeros(len(squared))
        w = self._mwi_n
        mwi[w:] = (cs[w:] - cs[:-w]) / w
        mwi[:w] = cs[:w] / (np.arange(1, w + 1))
        return mwi

    def _refine_peak(
        self, filtered: np.ndarray, idx: int, window: int
    ) -> int:
        """Refine peak index to local maximum in filtered signal."""
        lo = max(0, idx - window)
        hi = min(len(filtered), idx + window + 1)
        local = filtered[lo:hi]
        return lo + int(np.argmax(np.abs(local)))

    def _search_back(
        self,
        peaks: list[int],
        mwi: np.ndarray,
        filtered: np.ndarray,
        raw: np.ndarray,
        monitor_report: AdaptiveMonitorReport,
        details: list[PeakDetail],
        offset: int,
    ) -> list[int]:
        """Search-back: look for missed beats in long RR gaps."""
        if len(peaks) < 2:
            return peaks

        rrs = np.diff(peaks)
        median_rr = float(np.median(rrs))
        new_peaks = list(peaks)

        for i in range(len(rrs)):
            if rrs[i] > 1.66 * median_rr:
                # Long gap: search in MWI with lower threshold
                start = peaks[i] - offset
                end = peaks[i + 1] - offset
                if end <= start or end > len(mwi):
                    continue
                segment = mwi[start:end]
                threshold2 = float(np.max(mwi)) * 0.4
                found, _ = sp_signal.find_peaks(
                    segment, height=threshold2, distance=self._refractory_n
                )
                for f in found:
                    global_idx = start + f
                    if global_idx + offset not in new_peaks:
                        refined = self._refine_peak(
                            filtered, global_idx, window=int(self.fs * 0.04)
                        )
                        global_refined = refined + offset
                        if global_refined not in new_peaks:
                            new_peaks.append(global_refined)
                            d = self._build_peak_detail(
                                refined, offset, filtered, raw,
                                mwi, monitor_report,
                                search_back=True, refractory_pass=True,
                                refined=True, accepted=True,
                            )
                            details.append(d)

        return sorted(new_peaks)

    def _build_peak_detail(
        self,
        local_idx: int,
        offset: int,
        filtered: np.ndarray,
        raw: np.ndarray,
        mwi: np.ndarray,
        monitor_report: AdaptiveMonitorReport,
        search_back: bool,
        refractory_pass: bool,
        refined: bool,
        accepted: bool,
        reject_reason: str = "",
    ) -> PeakDetail:
        """Build a PeakDetail for a candidate peak."""
        global_idx = local_idx + offset
        time_s = global_idx / self.fs

        amp = float(filtered[local_idx]) if 0 <= local_idx < len(filtered) else 0.0
        raw_amp = float(raw[local_idx]) if 0 <= local_idx < len(raw) else 0.0

        # Local noise from MWI neighbourhood
        lo = max(0, local_idx - self._mwi_n * 2)
        hi = min(len(mwi), local_idx + self._mwi_n * 2)
        noise_level = float(np.percentile(mwi[lo:hi], 20)) if hi > lo else 0.0
        qrs_strength = float(np.clip(
            mwi[local_idx] / (noise_level + 1e-9) / 20.0, 0.0, 1.0
        ))

        # Segment reliability from monitor report
        seg_reliability = 1.0
        seg_noise = 0.0
        for seg in monitor_report.segments:
            if seg.start_s <= time_s <= seg.end_s:
                seg_reliability = seg.reliability
                seg_noise = seg.noise_rms
                break

        return PeakDetail(
            index=global_idx,
            time_s=time_s,
            amplitude=amp,
            raw_amplitude=raw_amp,
            rr_prev_s=None,
            rr_next_s=None,
            qrs_strength=float(np.clip(qrs_strength, 0.0, 1.0)),
            noise_level=noise_level,
            search_back=search_back,
            refractory_pass=refractory_pass,
            refined=refined,
            accepted=accepted,
            reject_reason=reject_reason,
            segment_noise=seg_noise,
            segment_reliability=seg_reliability,
        )

    # ------------------------------------------------------------------
    # HR / RR
    # ------------------------------------------------------------------

    def _compute_hr(
        self, result: ProcessingResult
    ) -> ProcessingResult:
        """Compute RR intervals and heart rate statistics."""
        peaks = result.r_peaks
        if len(peaks) < 2:
            result.warnings.append("Fewer than 2 R-peaks detected. HR unavailable.")
            return result

        # Fill RR prev/next in details
        peak_times = peaks / self.fs
        rr = np.diff(peak_times)
        result.rr_intervals_s = rr

        # Annotate details with RR
        peak_set = {d.index: i for i, d in enumerate(result.peak_details) if d.accepted}
        for pk_i, d in enumerate(result.peak_details):
            if not d.accepted:
                continue
            matched = [i for i, p in enumerate(peaks) if p == d.index]
            if not matched:
                continue
            pi = matched[0]
            d.rr_prev_s = float(rr[pi - 1]) if pi > 0 else None
            d.rr_next_s = float(rr[pi]) if pi < len(rr) else None

        # Filter physiologically unreasonable RR (20–300 BPM)
        valid_rr = rr[(rr >= 0.2) & (rr <= 3.0)]
        if len(valid_rr) == 0:
            result.warnings.append("No physiologically valid RR intervals. HR unreliable.")
            return result

        result.heart_rate_bpm = float(60.0 / np.median(valid_rr))

        # Instantaneous HR series
        rr_times = peak_times[:-1] + rr / 2  # midpoint
        result.hr_times_s = rr_times
        result.hr_series_bpm = np.where(
            (rr > 0.2) & (rr < 3.0),
            60.0 / rr,
            np.nan,
        )

        return result

    # ------------------------------------------------------------------
    # Reliability
    # ------------------------------------------------------------------

    def _compute_reliability(
        self,
        result: ProcessingResult,
        monitor_report: AdaptiveMonitorReport,
    ) -> float:
        """
        Compute a detection reliability score 0–1 based on measurable
        signal and detection characteristics.
        """
        score = 1.0

        # Peak consistency: coefficient of variation of RR
        if len(result.rr_intervals_s) > 1:
            valid_rr = result.rr_intervals_s[
                (result.rr_intervals_s >= 0.2) & (result.rr_intervals_s <= 3.0)
            ]
            if len(valid_rr) > 1:
                cv = float(np.std(valid_rr)) / (float(np.mean(valid_rr)) + 1e-9)
                score *= max(0.0, 1.0 - min(cv * 2, 1.0))

        # Segment reliability average
        seg_rels = [seg.reliability for seg in monitor_report.segments]
        if seg_rels:
            avg_seg_rel = float(np.mean(seg_rels))
            score *= (0.4 + 0.6 * avg_seg_rel)

        # Peak count sanity
        if len(result.r_peaks) < 2:
            score = 0.0
        elif result.heart_rate_bpm > 0:
            duration_min = len(result.raw) / self.fs / 60.0
            expected_min = 40 * duration_min
            expected_max = 200 * duration_min
            n_peaks = len(result.r_peaks)
            if not (expected_min <= n_peaks <= expected_max):
                score *= 0.5

        return float(np.clip(score, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Standalone algorithm comparison (Pan-Tompkins reference)
# ---------------------------------------------------------------------------

class PanTompkinsDetector:
    """
    Classic Pan-Tompkins detector with fixed thresholds.
    Used for algorithm comparison in research mode.
    """

    def __init__(self, fs: float) -> None:
        self.fs = fs
        self._engine = AdaptiveECGEngine(fs)

    def detect(self, ecg: np.ndarray) -> np.ndarray:
        """Return R-peak indices (simple fixed-threshold version)."""
        # Use the same preprocessing
        interpolated = _interpolate_nans(ecg)
        detrended = _remove_baseline(interpolated, self.fs)
        sos = sp_signal.butter(
            BP_ORDER,
            [BP_LOW_HZ / (self.fs / 2), min(0.99, BP_HIGH_HZ / (self.fs / 2))],
            btype="band", output="sos",
        )
        filtered = sp_signal.sosfiltfilt(sos, detrended)

        # Derivative + square + integrate
        deriv = np.diff(filtered, prepend=filtered[0])
        squared = deriv ** 2
        w = max(1, int(MWI_WINDOW_S * self.fs))
        cs = np.cumsum(squared)
        mwi = np.zeros_like(squared)
        mwi[w:] = (cs[w:] - cs[:-w]) / w
        mwi[:w] = cs[:w] / np.arange(1, w + 1)

        # Fixed threshold at 25% of max
        threshold = float(np.max(mwi)) * 0.25
        ref_n = int(REFRACTORY_S * self.fs)
        peaks, _ = sp_signal.find_peaks(mwi, height=threshold, distance=ref_n)
        return peaks


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _interpolate_nans(sig: np.ndarray) -> np.ndarray:
    """Linear interpolation of NaN values."""
    arr = sig.copy().astype(np.float64)
    nans = ~np.isfinite(arr)
    if not nans.any():
        return arr
    idx = np.arange(len(arr))
    arr[nans] = np.interp(idx[nans], idx[~nans], arr[~nans])
    return arr


def _remove_baseline(
    sig: np.ndarray, fs: float
) -> np.ndarray:
    """
    Remove baseline drift using a moving median subtraction.
    Preserves QRS morphology.
    """
    window_s = max(0.2, BASELINE_WINDOW_S)
    # Use a window of ~200 ms for local median (but at least 201ms wide)
    # The baseline window should be ~0.6s to capture TP segment
    w = int(max(0.6 * fs, 2 * int(window_s * fs) + 1))
    if w % 2 == 0:
        w += 1
    w = min(w, len(sig) - 1)
    if w < 3:
        return sig - float(np.median(sig))

    # Efficient moving median via cumulative approach isn't available in NumPy;
    # use scipy uniform_filter as an approximation for baseline
    from scipy.ndimage import uniform_filter1d
    baseline = uniform_filter1d(sig, size=w, mode="nearest")
    return sig - baseline


def _fill_trace(
    trace: np.ndarray, indices: np.ndarray, n: int
) -> None:
    """Fill a sparse trace by forward-filling from candidate indices."""
    last_val = trace[0] if len(trace) > 0 else 0.0
    j = 0
    for i in range(n):
        if j < len(indices) and i == indices[j]:
            last_val = trace[i]
            j += 1
        elif i > 0:
            trace[i] = last_val
