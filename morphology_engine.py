"""
morphology_engine.py — QRS and ST morphology analysis and preservation measurement.

Provides:
  * QRS amplitude measurement (before/after filtering).
  * QRS width estimation.
  * ST-level estimation.
  * Morphology preservation score.
  * Beat-to-beat morphology consistency.
  * Waveform correlation between raw and filtered ECG.

NOTE: This module provides engineering metrics only.
It does NOT make clinical diagnoses.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import signal as sp_signal


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class QRSMorphology:
    """QRS morphology measurements at detected R-peaks."""
    mean_amplitude: Optional[float] = None      # Mean peak-to-peak amplitude
    std_amplitude: Optional[float] = None
    mean_width_ms: Optional[float] = None       # Mean QRS width in ms
    std_width_ms: Optional[float] = None
    amplitude_variation_pct: Optional[float] = None  # Beat-to-beat variation %
    n_beats_measured: int = 0
    measurement_reliable: bool = False
    note: str = ""


@dataclass
class STMorphology:
    """ST-segment level measurements."""
    mean_st_level: Optional[float] = None       # Mean ST level (relative)
    std_st_level: Optional[float] = None
    n_beats_measured: int = 0
    measurement_reliable: bool = False
    note: str = ""


@dataclass
class MorphologyPreservation:
    """Preservation quality of morphology after filtering."""
    # QRS amplitude before and after
    qrs_amplitude_before: Optional[float] = None
    qrs_amplitude_after: Optional[float] = None
    qrs_amplitude_ratio: Optional[float] = None    # after/before (ideal = 1.0)
    qrs_amplitude_preserved_pct: Optional[float] = None

    # QRS waveform correlation (before vs after)
    qrs_waveform_correlation: Optional[float] = None

    # ST level before and after
    st_level_before: Optional[float] = None
    st_level_after: Optional[float] = None
    st_difference: Optional[float] = None          # absolute change in ST level

    # Overall morphology score (0–1)
    morphology_score: Optional[float] = None

    # Warnings
    warnings: list[str] = field(default_factory=list)
    note: str = ""


# ---------------------------------------------------------------------------
# Morphology engine
# ---------------------------------------------------------------------------

class MorphologyEngine:
    """
    Analyses QRS and ST morphology before and after filtering.

    Parameters
    ----------
    fs :
        Sampling rate in Hz.
    """

    # Analysis windows (seconds)
    QRS_HALF_WIN_S: float = 0.06      # ±60 ms around R-peak for QRS
    ST_OFFSET_S: float = 0.08         # 80 ms after R-peak for ST
    ST_WIN_S: float = 0.04            # 40 ms ST averaging window
    MIN_BEATS: int = 3                 # Minimum beats for reliable stats

    def __init__(self, fs: float) -> None:
        self.fs = float(fs)
        self._qrs_half_n = max(1, int(self.QRS_HALF_WIN_S * fs))
        self._st_offset_n = max(1, int(self.ST_OFFSET_S * fs))
        self._st_win_n = max(1, int(self.ST_WIN_S * fs))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def measure_qrs(
        self,
        ecg: np.ndarray,
        r_peaks: np.ndarray,
    ) -> QRSMorphology:
        """
        Measure QRS morphology at detected R-peaks.

        Parameters
        ----------
        ecg :
            ECG signal (morphology path, not QRS detection path).
        r_peaks :
            Array of R-peak sample indices.

        Returns
        -------
        QRSMorphology
        """
        m = QRSMorphology()
        valid_peaks = r_peaks[
            (r_peaks >= self._qrs_half_n) &
            (r_peaks < len(ecg) - self._qrs_half_n)
        ]

        if len(valid_peaks) < self.MIN_BEATS:
            m.note = f"Too few valid peaks ({len(valid_peaks)}) for reliable QRS measurement."
            return m

        amplitudes: list[float] = []
        widths_ms: list[float] = []

        for pk in valid_peaks:
            window = ecg[pk - self._qrs_half_n: pk + self._qrs_half_n]
            amp = float(np.max(window) - np.min(window))
            amplitudes.append(amp)

            # Width: region where signal > 50% of peak amplitude
            peak_val = float(ecg[pk])
            half_amp = float(np.min(window)) + amp * 0.5
            above = window > half_amp
            width_n = int(np.sum(above))
            widths_ms.append(1000.0 * width_n / self.fs)

        m.n_beats_measured = len(amplitudes)
        m.mean_amplitude = float(np.mean(amplitudes))
        m.std_amplitude = float(np.std(amplitudes))
        m.mean_width_ms = float(np.mean(widths_ms))
        m.std_width_ms = float(np.std(widths_ms))

        if m.mean_amplitude > 1e-9:
            m.amplitude_variation_pct = 100.0 * m.std_amplitude / m.mean_amplitude
        m.measurement_reliable = (
            m.n_beats_measured >= self.MIN_BEATS and
            m.mean_amplitude > 1e-6
        )
        return m

    def measure_st(
        self,
        ecg: np.ndarray,
        r_peaks: np.ndarray,
    ) -> STMorphology:
        """
        Estimate ST-segment levels at detected R-peaks.

        Parameters
        ----------
        ecg :
            ECG signal (morphology path).
        r_peaks :
            Array of R-peak sample indices.

        Returns
        -------
        STMorphology
        """
        m = STMorphology()
        n = len(ecg)
        valid_peaks = r_peaks[
            (r_peaks + self._st_offset_n + self._st_win_n < n) &
            (r_peaks >= self._qrs_half_n)
        ]

        if len(valid_peaks) < self.MIN_BEATS:
            m.note = "Too few valid peaks for ST measurement."
            return m

        st_levels: list[float] = []
        for pk in valid_peaks:
            st_start = pk + self._st_offset_n
            st_end = st_start + self._st_win_n
            if st_end > n:
                continue
            st_window = ecg[st_start:st_end]
            # ST level relative to the beat's baseline (pre-QRS isoelectric)
            iso_start = max(0, pk - self._qrs_half_n - self._st_win_n)
            iso_end = max(1, pk - self._qrs_half_n)
            iso_level = float(np.median(ecg[iso_start:iso_end]))
            st_level = float(np.median(st_window)) - iso_level
            st_levels.append(st_level)

        if len(st_levels) < self.MIN_BEATS:
            m.note = "Insufficient ST windows computed."
            return m

        m.n_beats_measured = len(st_levels)
        m.mean_st_level = float(np.mean(st_levels))
        m.std_st_level = float(np.std(st_levels))
        m.measurement_reliable = True
        return m

    def evaluate_preservation(
        self,
        raw: np.ndarray,
        morph_filtered: np.ndarray,
        r_peaks: np.ndarray,
    ) -> MorphologyPreservation:
        """
        Compare morphology before and after filtering.

        Parameters
        ----------
        raw :
            Raw (or lightly preprocessed) ECG signal.
        morph_filtered :
            Morphology-path filtered ECG.
        r_peaks :
            Detected R-peak indices.

        Returns
        -------
        MorphologyPreservation
        """
        pres = MorphologyPreservation()

        # Require matched lengths
        n = min(len(raw), len(morph_filtered))
        if n == 0:
            pres.note = "Empty signal."
            return pres

        raw_n = raw[:n]
        filt_n = morph_filtered[:n]

        valid_peaks = r_peaks[
            (r_peaks >= self._qrs_half_n) &
            (r_peaks < n - self._qrs_half_n)
        ]

        if len(valid_peaks) < self.MIN_BEATS:
            pres.note = "Too few peaks for preservation measurement."
            return pres

        # QRS amplitude comparison
        raw_amps, filt_amps = [], []
        corr_vals: list[float] = []

        for pk in valid_peaks:
            raw_win = raw_n[pk - self._qrs_half_n: pk + self._qrs_half_n]
            filt_win = filt_n[pk - self._qrs_half_n: pk + self._qrs_half_n]

            raw_amp = float(np.max(raw_win) - np.min(raw_win))
            filt_amp = float(np.max(filt_win) - np.min(filt_win))
            raw_amps.append(raw_amp)
            filt_amps.append(filt_amp)

            # Waveform correlation per beat
            raw_c = raw_win - np.mean(raw_win)
            filt_c = filt_win - np.mean(filt_win)
            nr = float(np.linalg.norm(raw_c))
            nf = float(np.linalg.norm(filt_c))
            if nr > 1e-9 and nf > 1e-9:
                corr = float(np.dot(raw_c, filt_c) / (nr * nf))
                corr_vals.append(float(np.clip(corr, -1.0, 1.0)))

        if not raw_amps:
            pres.note = "No amplitude data computed."
            return pres

        pres.qrs_amplitude_before = float(np.median(raw_amps))
        pres.qrs_amplitude_after = float(np.median(filt_amps))

        if pres.qrs_amplitude_before > 1e-9:
            pres.qrs_amplitude_ratio = pres.qrs_amplitude_after / pres.qrs_amplitude_before
            pres.qrs_amplitude_preserved_pct = 100.0 * min(
                pres.qrs_amplitude_ratio, 1.0 / max(pres.qrs_amplitude_ratio, 1e-9)
            ) * pres.qrs_amplitude_ratio
            # Simpler: percentage preserved (capped at 100%)
            pres.qrs_amplitude_preserved_pct = float(
                np.clip(100.0 * min(pres.qrs_amplitude_ratio, 1.0), 0.0, 100.0)
            )

        if corr_vals:
            pres.qrs_waveform_correlation = float(np.mean(corr_vals))

        # ST measurement
        st_before = self.measure_st(raw_n, valid_peaks)
        st_after = self.measure_st(filt_n, valid_peaks)
        if st_before.measurement_reliable and st_after.measurement_reliable:
            pres.st_level_before = st_before.mean_st_level
            pres.st_level_after = st_after.mean_st_level
            if pres.st_level_before is not None and pres.st_level_after is not None:
                pres.st_difference = abs(
                    pres.st_level_after - pres.st_level_before
                )
        else:
            pres.warnings.append("ST measurement unreliable — insufficient peaks or signal quality.")

        # Overall morphology score (0-1)
        score_components: list[float] = []
        if pres.qrs_waveform_correlation is not None:
            score_components.append(float(np.clip(pres.qrs_waveform_correlation, 0.0, 1.0)))
        if pres.qrs_amplitude_preserved_pct is not None:
            score_components.append(pres.qrs_amplitude_preserved_pct / 100.0)
        if score_components:
            pres.morphology_score = float(np.mean(score_components))

        return pres
