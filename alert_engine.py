"""
alert_engine.py — False-alarm suppression, acute-event detection, and alert timing.

Provides:
  * False-alarm suppression (artifact-driven alarm reduction).
  * Possible VT / VF detection alerts.
  * Alert timing measurement (target: <= 3 seconds).
  * False-alarm fatigue measurement (internal engineering metric).
  * Streaming alert engine for edge/real-time use.

IMPORTANT:
  - Reports 'POSSIBLE VT/VF' only, NOT definitive medical diagnosis.
  - Prioritizes sensitivity: does NOT silence alerts on poor quality alone.
  - Poor quality + possible acute event => retained alert flagged for review.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class AlertType(str, Enum):
    NONE = "None"
    POSSIBLE_VT = "Possible Ventricular Tachycardia"
    POSSIBLE_VF = "Possible Ventricular Fibrillation"
    ARTIFACT_ALARM = "Artifact-driven alarm (suppressed)"
    QUALITY_LIMITED = "Quality-limited alert"


class AlertPriority(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    SUPPRESSED = "SUPPRESSED"


@dataclass
class Alert:
    """A single alert event."""
    alert_type: AlertType = AlertType.NONE
    priority: AlertPriority = AlertPriority.LOW
    onset_s: float = 0.0
    detection_latency_s: float = 0.0
    evidence: list[str] = field(default_factory=list)
    signal_quality: str = "UNKNOWN"
    suppressed: bool = False
    suppress_reason: str = ""
    message: str = ""
    requires_review: bool = False


@dataclass
class AlertReport:
    """Complete alert analysis report."""
    alerts: list[Alert] = field(default_factory=list)
    n_total_alerts: int = 0
    n_suppressed: int = 0
    n_artifact_alarms: int = 0
    n_quality_limited: int = 0
    detection_latencies_s: list[float] = field(default_factory=list)
    mean_latency_s: Optional[float] = None
    max_latency_s: Optional[float] = None
    target_latency_s: float = 3.0
    pct_within_target: Optional[float] = None
    false_alarm_rate_per_hour: Optional[float] = None
    recording_duration_s: float = 0.0
    note: str = "Internal engineering metric — not an official competition formula."


class AlertEngine:
    """
    Manages alert generation, suppression, and timing measurement.

    Parameters
    ----------
    fs :
        Sampling rate in Hz.
    alert_latency_target_s :
        Target alert latency in seconds (default 3.0).
    """

    VT_HR_MIN: float = 120.0
    VT_SUSTAINED_BEATS: int = 4
    SUPPRESS_RELIABILITY_THRESHOLD: float = 0.30

    def __init__(self, fs: float, alert_latency_target_s: float = 3.0) -> None:
        self.fs = float(fs)
        self.target_latency_s = alert_latency_target_s

    def evaluate(
        self,
        r_peaks: np.ndarray,
        rr_intervals_s: np.ndarray,
        detection_reliability: float,
        signal_quality: str,
        emg_severity: str,
        recording_duration_s: float,
        possible_vt: bool = False,
        possible_vf: bool = False,
        vt_evidence: Optional[list[str]] = None,
        vf_evidence: Optional[list[str]] = None,
    ) -> AlertReport:
        """
        Generate and evaluate alerts for the current recording.

        Parameters
        ----------
        r_peaks : np.ndarray
            Detected R-peak indices.
        rr_intervals_s : np.ndarray
            RR interval array in seconds.
        detection_reliability : float
            Detection reliability score (0-1).
        signal_quality : str
            Signal quality label.
        emg_severity : str
            EMG severity label.
        recording_duration_s : float
            Total recording duration in seconds.
        possible_vt : bool
            VT flag from rhythm classifier.
        possible_vf : bool
            VF flag from rhythm classifier.
        vt_evidence : list[str]
            VT evidence strings.
        vf_evidence : list[str]
            VF evidence strings.

        Returns
        -------
        AlertReport
        """
        report = AlertReport(recording_duration_s=recording_duration_s)
        vt_evidence = vt_evidence or []
        vf_evidence = vf_evidence or []
        alerts: list[Alert] = []

        if possible_vf:
            alert = self._build_alert(
                AlertType.POSSIBLE_VF, AlertPriority.CRITICAL,
                vf_evidence, signal_quality, detection_reliability,
                emg_severity, r_peaks, rr_intervals_s,
            )
            alerts.append(alert)

        if possible_vt and not possible_vf:
            alert = self._build_alert(
                AlertType.POSSIBLE_VT, AlertPriority.HIGH,
                vt_evidence, signal_quality, detection_reliability,
                emg_severity, r_peaks, rr_intervals_s,
            )
            alerts.append(alert)

        report.alerts = alerts
        report.n_total_alerts = len(alerts)
        report.n_suppressed = sum(1 for a in alerts if a.suppressed)
        report.n_artifact_alarms = sum(
            1 for a in alerts if a.alert_type == AlertType.ARTIFACT_ALARM
        )
        report.n_quality_limited = sum(
            1 for a in alerts if a.alert_type == AlertType.QUALITY_LIMITED
        )

        lats = [a.detection_latency_s for a in alerts if not a.suppressed]
        report.detection_latencies_s = lats
        if lats:
            report.mean_latency_s = float(np.mean(lats))
            report.max_latency_s = float(np.max(lats))
            report.pct_within_target = float(
                np.mean([lat <= self.target_latency_s for lat in lats]) * 100.0
            )

        if recording_duration_s > 0:
            dur_h = recording_duration_s / 3600.0
            n_artifact = report.n_artifact_alarms + report.n_suppressed
            report.false_alarm_rate_per_hour = n_artifact / max(dur_h, 1e-9)

        return report

    def _build_alert(
        self,
        alert_type: AlertType,
        priority: AlertPriority,
        evidence: list[str],
        signal_quality: str,
        detection_reliability: float,
        emg_severity: str,
        r_peaks: np.ndarray,
        rr_intervals_s: np.ndarray,
    ) -> Alert:
        """Build a single Alert with suppression logic applied."""
        alert = Alert(
            alert_type=alert_type,
            priority=priority,
            evidence=list(evidence),
            signal_quality=signal_quality,
        )

        # Compute detection latency from sustained rapid beats
        valid_rr = rr_intervals_s[(rr_intervals_s >= 0.2) & (rr_intervals_s <= 3.0)]
        if len(valid_rr) >= self.VT_SUSTAINED_BEATS:
            latency = float(np.sum(valid_rr[:self.VT_SUSTAINED_BEATS]))
        elif len(valid_rr) > 0:
            latency = float(np.sum(valid_rr))
        else:
            latency = 0.0
        alert.detection_latency_s = latency

        poor_quality = signal_quality in ("POOR", "CRITICAL")
        severe_emg = "SEVERE" in emg_severity

        # Key rule: poor quality + possible event => flag for review, NOT suppress
        if poor_quality and severe_emg and detection_reliability < 0.6:
            alert.suppressed = False
            alert.alert_type = AlertType.QUALITY_LIMITED
            alert.priority = AlertPriority.MEDIUM
            alert.requires_review = True
            alert.message = (
                f"POSSIBLE {alert_type.value.upper()} — SIGNAL QUALITY LIMITED. "
                "Review required. Do not dismiss without evaluation."
            )
            alert.suppress_reason = (
                "Signal quality limited (severe EMG). Alert retained for review."
            )
        elif detection_reliability < self.SUPPRESS_RELIABILITY_THRESHOLD and not evidence:
            # No evidence + very low reliability => artifact alarm, suppress
            alert.suppressed = True
            alert.alert_type = AlertType.ARTIFACT_ALARM
            alert.priority = AlertPriority.SUPPRESSED
            alert.suppress_reason = (
                "Low reliability and no clinical evidence — artifact alarm suppressed."
            )
        else:
            alert.message = (
                f"{alert_type.value}: "
                + (" | ".join(evidence) if evidence else "Evidence from RR analysis.")
            )

        return alert


class StreamingAlertEngine:
    """
    Stateful alert engine for streaming/chunked processing.

    Parameters
    ----------
    fs :
        Sampling rate in Hz.
    """

    VT_RAPID_BPM: float = 120.0
    VT_SUSTAINED_BEATS: int = 4

    def __init__(self, fs: float) -> None:
        self.fs = float(fs)
        self._rapid_count: int = 0
        self._alert_active: bool = False

    def reset(self) -> None:
        """Reset state for a new recording."""
        self._rapid_count = 0
        self._alert_active = False

    def update(
        self,
        rr_intervals_s: np.ndarray,
        chunk_start_s: float,
    ) -> Optional[Alert]:
        """
        Update streaming state with new RR intervals from a chunk.

        Parameters
        ----------
        rr_intervals_s :
            RR intervals from the latest processing chunk.
        chunk_start_s :
            Start time of the current chunk in seconds.

        Returns
        -------
        Alert if an event is detected, else None.
        """
        for rr in rr_intervals_s:
            if not (0.2 <= rr <= 3.0):
                continue
            if 60.0 / rr >= self.VT_RAPID_BPM:
                self._rapid_count += 1
            else:
                self._rapid_count = 0
                if self._alert_active:
                    self._alert_active = False

            if self._rapid_count >= self.VT_SUSTAINED_BEATS and not self._alert_active:
                self._alert_active = True
                latency = rr * self._rapid_count
                within_target = latency <= 3.0
                return Alert(
                    alert_type=AlertType.POSSIBLE_VT,
                    priority=AlertPriority.HIGH,
                    onset_s=chunk_start_s,
                    detection_latency_s=latency,
                    evidence=[
                        f"{self._rapid_count} consecutive beats >= {self.VT_RAPID_BPM:.0f} BPM"
                    ],
                    message=(
                        f"Possible VT at t={chunk_start_s:.1f}s. "
                        f"Detection latency: {latency:.2f}s. "
                        f"Target <= 3.00s: {'YES' if within_target else 'NO'}"
                    ),
                )
        return None
