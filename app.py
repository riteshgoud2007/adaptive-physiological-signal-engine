"""
app.py — Adaptive ECG Signal Intelligence Engine — Streamlit Dashboard

Run with: streamlit run app.py
"""

from __future__ import annotations

import sys
import time
import math
from pathlib import Path
from typing import Optional

import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.subplots as sp

# Project imports
try:
    from ecg_loader import load_ecg, get_channel_names, select_channel, ECGRecord
    from quality_engine import QualityEngine, QualityLevel, SegmentStatus, AdaptiveMonitorReport
    from adaptive_engine import AdaptiveECGEngine, ProcessingResult, PeakDetail
    from evaluation import compare_algorithms, ComparisonReport
    import config
except ImportError as e:
    st.error(f"Import error: {e}. Make sure all project files are present.")
    st.stop()

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Adaptive ECG Signal Intelligence Engine",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — professional biomedical theme
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    /* Main background */
    .main { background-color: #ffffff; }
    .stApp { background-color: #ffffff; }
    
    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #f0f4f8; }
    
    /* Header */
    .ecg-header {
        background: linear-gradient(135deg, #1a3a6b 0%, #2196f3 100%);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
    }
    .ecg-header h1 { color: white; font-size: 1.8rem; margin: 0; }
    .ecg-header p { color: rgba(255,255,255,0.85); margin: 0.3rem 0 0 0; font-size: 0.95rem; }
    
    /* Result cards */
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        border-left: 4px solid #2196f3;
    }
    .metric-card .value { font-size: 2rem; font-weight: 700; color: #1a3a6b; }
    .metric-card .label { font-size: 0.8rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; }
    
    /* Status badges */
    .badge-good { background:#e8f5e9; color:#2e7d32; padding:2px 10px; border-radius:12px; font-weight:600; font-size:0.85rem; }
    .badge-moderate { background:#fff3e0; color:#e65100; padding:2px 10px; border-radius:12px; font-weight:600; font-size:0.85rem; }
    .badge-poor { background:#ffebee; color:#c62828; padding:2px 10px; border-radius:12px; font-weight:600; font-size:0.85rem; }
    .badge-critical { background:#b71c1c; color:#fff; padding:2px 10px; border-radius:12px; font-weight:600; font-size:0.85rem; }
    
    /* Section headers */
    .section-header {
        border-bottom: 2px solid #1a3a6b;
        padding-bottom: 0.4rem;
        margin: 1.5rem 0 1rem 0;
        color: #1a3a6b;
        font-weight: 700;
        font-size: 1.1rem;
    }
    
    /* Disclaimer */
    .disclaimer {
        background: #fff8e1;
        border: 1px solid #ffe082;
        border-radius: 6px;
        padding: 0.7rem 1rem;
        font-size: 0.8rem;
        color: #6d4c41;
        margin-top: 1rem;
    }
    
    /* Monitor timeline */
    .timeline-item { padding: 0.3rem 0.5rem; border-radius: 4px; margin: 0.2rem 0; font-size: 0.85rem; }
    .timeline-stable { background:#e8f5e9; }
    .timeline-warn { background:#fff3e0; }
    .timeline-critical { background:#ffebee; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helper — quality badge HTML
# ---------------------------------------------------------------------------

def quality_badge(level: QualityLevel) -> str:
    """Return an HTML badge string for the given quality level."""
    badges = {
        QualityLevel.GOOD: '<span class="badge-good">✓ GOOD</span>',
        QualityLevel.MODERATE: '<span class="badge-moderate">⚠ MODERATE</span>',
        QualityLevel.POOR: '<span class="badge-poor">✗ POOR</span>',
        QualityLevel.CRITICAL: '<span class="badge-critical">🔴 CRITICAL</span>',
    }
    return badges.get(level, str(level))


def reliability_bar(score: float) -> str:
    """Return an HTML reliability progress bar string."""
    pct = int(score * 100)
    filled = int(score * 20)
    bar = '█' * filled + '░' * (20 - filled)
    color = "#43a047" if pct >= 75 else ("#fb8c00" if pct >= 50 else "#e53935")
    return f'<span style="color:{color};font-family:monospace">{bar}</span> <strong>{pct}%</strong>'


# ---------------------------------------------------------------------------
# Plotly helpers
# ---------------------------------------------------------------------------

MAX_PLOT_SAMPLES = config.MAX_PLOT_SAMPLES


def _downsample(x: np.ndarray, y: np.ndarray, max_pts: int = MAX_PLOT_SAMPLES):
    """Min-max downsample preserving peaks for ECG rendering."""
    n = len(y)
    if n <= max_pts:
        return x, y
    step = n // (max_pts // 2)
    if step < 2:
        return x, y
    n_blocks = n // step
    xd, yd = [], []
    for i in range(n_blocks):
        s = i * step
        e = s + step
        block_y = y[s:e]
        block_x = x[s:e]
        imin = int(np.argmin(block_y))
        imax = int(np.argmax(block_y))
        if imin < imax:
            xd += [block_x[imin], block_x[imax]]
            yd += [block_y[imin], block_y[imax]]
        else:
            xd += [block_x[imax], block_x[imin]]
            yd += [block_y[imax], block_y[imin]]
    return np.array(xd), np.array(yd)


def plot_raw_ecg(time_axis: np.ndarray, signal: np.ndarray, title: str = "Original ECG") -> go.Figure:
    """Plot the raw ECG signal."""
    tx, sy = _downsample(time_axis, signal)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=tx, y=sy,
        mode='lines',
        name='Raw ECG',
        line=dict(color=config.PLOT_COLORS["raw"], width=1.2),
        hovertemplate='Time: %{x:.3f}s<br>Amplitude: %{y:.4f}<extra></extra>',
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#1a3a6b")),
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        font=dict(color="#1c1c1c"),
        margin=dict(l=60, r=30, t=50, b=50),
        xaxis=dict(showgrid=True, gridcolor="#f0f0f0", zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="#f0f0f0", zeroline=True, zerolinecolor="#e0e0e0"),
        height=300,
    )
    return fig


def plot_filtered_with_peaks(
    time_axis: np.ndarray,
    filtered: np.ndarray,
    r_peaks: np.ndarray,
    rejected: np.ndarray,
    fs: float,
    title: str = "Filtered ECG + Detected R-Peaks",
) -> go.Figure:
    """Plot filtered ECG with accepted and rejected peaks marked."""
    tx, fy = _downsample(time_axis, filtered)
    fig = go.Figure()

    # Filtered ECG
    fig.add_trace(go.Scatter(
        x=tx, y=fy,
        mode='lines',
        name='Filtered ECG',
        line=dict(color=config.PLOT_COLORS["filtered"], width=1.5),
        hovertemplate='Time: %{x:.3f}s<br>Amplitude: %{y:.4f}<extra></extra>',
    ))

    # R-peaks
    if len(r_peaks) > 0:
        valid = r_peaks[(r_peaks >= 0) & (r_peaks < len(filtered))]
        pk_times = time_axis[valid]
        pk_amps = filtered[valid]
        fig.add_trace(go.Scatter(
            x=pk_times, y=pk_amps,
            mode='markers',
            name='R-Peaks',
            marker=dict(
                color=config.PLOT_COLORS["peaks"],
                size=10, symbol='circle',
                line=dict(color='white', width=1.5),
            ),
            hovertemplate='R-Peak<br>Time: %{x:.3f}s<br>Amp: %{y:.4f}<extra></extra>',
        ))

    # Rejected candidates (dimmed)
    if len(rejected) > 0:
        valid_rej = rejected[(rejected >= 0) & (rejected < len(filtered))]
        if len(valid_rej) > 0:
            rj_times = time_axis[valid_rej]
            rj_amps = filtered[valid_rej]
            fig.add_trace(go.Scatter(
                x=rj_times, y=rj_amps,
                mode='markers',
                name='Rejected',
                marker=dict(
                    color=config.PLOT_COLORS["rejected"],
                    size=6, symbol='x',
                ),
                hovertemplate='Rejected<br>Time: %{x:.3f}s<br>Amp: %{y:.4f}<extra></extra>',
                visible='legendonly',
            ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#1a3a6b")),
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        font=dict(color="#1c1c1c"),
        margin=dict(l=60, r=30, t=50, b=50),
        xaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
        yaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
        height=350,
        legend=dict(orientation='h', y=1.05),
    )
    return fig


def plot_before_after(
    time_axis: np.ndarray,
    raw: np.ndarray,
    filtered: np.ndarray,
    r_peaks: np.ndarray,
    fs: float,
) -> go.Figure:
    """Side-by-side before/after subplot comparison."""
    fig = sp.make_subplots(
        rows=1, cols=2,
        subplot_titles=("BEFORE — Original ECG", "AFTER — Filtered ECG + R-Peaks"),
        shared_yaxes=False,
    )

    tx_r, sy_r = _downsample(time_axis, raw)
    tx_f, sy_f = _downsample(time_axis, filtered)

    fig.add_trace(go.Scatter(
        x=tx_r, y=sy_r, mode='lines',
        name='Raw', line=dict(color=config.PLOT_COLORS["raw"], width=1.0),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=tx_f, y=sy_f, mode='lines',
        name='Filtered', line=dict(color=config.PLOT_COLORS["filtered"], width=1.2),
    ), row=1, col=2)

    if len(r_peaks) > 0:
        valid = r_peaks[(r_peaks >= 0) & (r_peaks < len(filtered))]
        fig.add_trace(go.Scatter(
            x=time_axis[valid], y=filtered[valid],
            mode='markers',
            name='R-Peaks',
            marker=dict(color=config.PLOT_COLORS["peaks"], size=9, symbol='circle'),
        ), row=1, col=2)

    fig.update_layout(
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        font=dict(color="#1c1c1c"),
        height=320,
        margin=dict(l=60, r=30, t=60, b=50),
        showlegend=False,
    )
    fig.update_xaxes(title_text="Time (s)", gridcolor="#f0f0f0")
    fig.update_yaxes(title_text="Amplitude", gridcolor="#f0f0f0")
    return fig


def plot_hr_series(
    hr_times: np.ndarray, hr_bpm: np.ndarray
) -> go.Figure:
    """Plot instantaneous heart rate over time."""
    valid = np.isfinite(hr_bpm)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hr_times[valid], y=hr_bpm[valid],
        mode='lines+markers',
        name='Instantaneous HR',
        line=dict(color='#e53935', width=1.5),
        marker=dict(size=4),
        hovertemplate='Time: %{x:.2f}s<br>HR: %{y:.1f} BPM<extra></extra>',
    ))
    fig.add_hline(y=float(np.nanmedian(hr_bpm)), line_dash='dash',
                  line_color='#1a3a6b', annotation_text='Median HR')
    fig.update_layout(
        title='Instantaneous Heart Rate',
        xaxis_title='Time (s)', yaxis_title='BPM',
        plot_bgcolor='#ffffff', paper_bgcolor='#ffffff',
        height=250, margin=dict(l=60, r=30, t=50, b=50),
        xaxis=dict(gridcolor='#f0f0f0'),
        yaxis=dict(gridcolor='#f0f0f0', range=[20, 220]),
    )
    return fig


def plot_research_mode(result: ProcessingResult, time_axis: np.ndarray) -> go.Figure:
    """Multi-panel research mode visualisation."""
    n = len(result.filtered)
    t = time_axis[:n]

    fig = sp.make_subplots(
        rows=6, cols=1,
        subplot_titles=[
            "1. Raw ECG", "2. Detrended ECG", "3. Filtered ECG",
            "4. Derivative", "5. Squared + MWI", "6. Adaptive Threshold"
        ],
        shared_xaxes=True,
        vertical_spacing=0.04,
    )

    def ds(y):
        """Downsample helper."""
        return _downsample(t, y)

    # 1. Raw
    tx, sy = ds(result.raw[:n])
    fig.add_trace(go.Scatter(x=tx, y=sy, mode='lines',
                             line=dict(color=config.PLOT_COLORS["raw"], width=1),
                             name='Raw', showlegend=False), row=1, col=1)
    # 2. Detrended
    tx, sy = ds(result.detrended[:n])
    fig.add_trace(go.Scatter(x=tx, y=sy, mode='lines',
                             line=dict(color='#59a14f', width=1),
                             name='Detrended', showlegend=False), row=2, col=1)
    # 3. Filtered
    tx, sy = ds(result.filtered[:n])
    fig.add_trace(go.Scatter(x=tx, y=sy, mode='lines',
                             line=dict(color=config.PLOT_COLORS["filtered"], width=1.2),
                             name='Filtered', showlegend=False), row=3, col=1)
    # Add peaks on filtered
    if len(result.r_peaks) > 0:
        valid = result.r_peaks[(result.r_peaks >= 0) & (result.r_peaks < n)]
        fig.add_trace(go.Scatter(
            x=time_axis[valid], y=result.filtered[valid],
            mode='markers', marker=dict(color=config.PLOT_COLORS["peaks"], size=7),
            name='Peaks', showlegend=False,
        ), row=3, col=1)
    # 4. Derivative
    tx, sy = ds(result.derivative[:n])
    fig.add_trace(go.Scatter(x=tx, y=sy, mode='lines',
                             line=dict(color='#76b7b2', width=1),
                             name='Deriv', showlegend=False), row=4, col=1)
    # 5. Squared + MWI
    if len(result.squared) >= n:
        tx2, sq = ds(result.squared[:n])
        fig.add_trace(go.Scatter(x=tx2, y=sq, mode='lines',
                                 line=dict(color='#f28e2b', width=0.8, dash='dot'),
                                 name='Squared', showlegend=False), row=5, col=1)
    if len(result.mwi) >= n:
        tx3, mwi = ds(result.mwi[:n])
        fig.add_trace(go.Scatter(x=tx3, y=mwi, mode='lines',
                                 line=dict(color='#e15759', width=1.2),
                                 name='MWI', showlegend=False), row=5, col=1)
    # 6. Threshold
    if len(result.threshold_signal) >= n:
        tx4, ts = ds(result.threshold_signal[:n])
        fig.add_trace(go.Scatter(x=tx4, y=ts, mode='lines',
                                 line=dict(color='#1a3a6b', width=1.2),
                                 name='Signal thr', showlegend=False), row=6, col=1)
    if len(result.threshold_noise) >= n:
        txn, tn = ds(result.threshold_noise[:n])
        fig.add_trace(go.Scatter(x=txn, y=tn, mode='lines',
                                 line=dict(color='#bab0ac', width=1, dash='dash'),
                                 name='Noise thr', showlegend=False), row=6, col=1)

    fig.update_layout(
        height=1000,
        plot_bgcolor='#ffffff',
        paper_bgcolor='#ffffff',
        margin=dict(l=70, r=20, t=80, b=50),
        font=dict(color='#1c1c1c'),
    )
    fig.update_xaxes(title_text='Time (s)', row=6, col=1, gridcolor='#f0f0f0')
    for row in range(1, 7):
        fig.update_xaxes(gridcolor='#f0f0f0', row=row, col=1)
        fig.update_yaxes(gridcolor='#f0f0f0', row=row, col=1)
    return fig


# ---------------------------------------------------------------------------
# Segment monitor timeline renderer
# ---------------------------------------------------------------------------

STATUS_ICONS = {
    SegmentStatus.STABLE: ("✓", "timeline-stable"),
    SegmentStatus.MOTION_ARTIFACT: ("⚠", "timeline-warn"),
    SegmentStatus.NOISE_INCREASED: ("⚠", "timeline-warn"),
    SegmentStatus.NOISE_CHANGED: ("⚠", "timeline-warn"),
    SegmentStatus.BASELINE_DRIFT: ("⚠", "timeline-warn"),
    SegmentStatus.SATURATION: ("🔴", "timeline-critical"),
    SegmentStatus.MISSING_DATA: ("⚠", "timeline-warn"),
    SegmentStatus.FLATLINE: ("🔴", "timeline-critical"),
    SegmentStatus.WEAK_QRS: ("⚠", "timeline-warn"),
}


def render_monitor_timeline(
    monitor: AdaptiveMonitorReport,
    max_segments: int = 40,
) -> str:
    """Render timeline HTML for the adaptive signal monitor."""
    lines = []
    segs = monitor.segments
    # Merge consecutive identical statuses
    if not segs:
        return "<em>No segment data.</em>"

    merged = []
    i = 0
    while i < len(segs):
        current = segs[i]
        j = i + 1
        while j < len(segs) and segs[j].status == current.status:
            j += 1
        merged.append((current.start_s, segs[j - 1].end_s, current.status))
        i = j

    for start_s, end_s, status in merged[:max_segments]:
        icon, css = STATUS_ICONS.get(status, ("•", "timeline-stable"))
        label = status.value
        t_start = _fmt_time(start_s)
        t_end = _fmt_time(end_s)
        lines.append(
            f'<div class="timeline-item {css}">{icon} '
            f'<strong>{t_start}–{t_end}</strong> &nbsp; {label}</div>'
        )

    if monitor.sensor_change_detected:
        lines.append(
            '<div class="timeline-item timeline-warn">⚠ <strong>Possible sensor/device characteristic change detected</strong></div>'
        )
    return "\n".join(lines)


def _fmt_time(s: float) -> str:
    """Format seconds as MM:SS.s string."""
    m = int(s) // 60
    sec = s - m * 60
    return f"{m:02d}:{sec:04.1f}"


# ---------------------------------------------------------------------------
# Peak explainer
# ---------------------------------------------------------------------------

def render_peak_detail(detail: PeakDetail) -> None:
    """Display explainable detail card for a single peak."""
    status_color = "#43a047" if detail.accepted else "#e53935"
    decision = "ACCEPTED" if detail.accepted else "REJECTED"
    with st.expander(
        f"{'R-Peak' if detail.accepted else 'Candidate'} @ {detail.time_s:.3f} s — {decision}",
        expanded=False,
    ):
        cols = st.columns([1, 1, 1])
        with cols[0]:
            st.markdown(f"**Time:** {detail.time_s:.3f} s")
            st.markdown(f"**Amplitude:** {detail.amplitude:.4f}")
            if detail.rr_prev_s is not None:
                st.markdown(f"**Previous RR:** {detail.rr_prev_s:.3f} s")
            if detail.rr_next_s is not None:
                st.markdown(f"**Next RR:** {detail.rr_next_s:.3f} s")
        with cols[1]:
            qrs_pct = int(detail.qrs_strength * 100)
            st.markdown(f"**QRS Strength:** {qrs_pct}%")
            st.markdown(f"**Noise Level:** {detail.noise_level:.4f}")
            st.markdown(f"**Segment Reliability:** {detail.segment_reliability*100:.0f}%")
        with cols[2]:
            st.markdown(f"**Refractory:** {'PASS' if detail.refractory_pass else 'FAIL'}")
            st.markdown(f"**Search-back:** {'YES' if detail.search_back else 'NO'}")
            st.markdown(f"**Refined:** {'YES' if detail.refined else 'NO'}")
            st.markdown(
                f"**Decision:** <span style='color:{status_color};font-weight:700'>{decision}</span>"
                + (f"<br>Reason: {detail.reject_reason}" if detail.reject_reason else ""),
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Algorithm comparison table
# ---------------------------------------------------------------------------

def render_comparison_table(comp_report: ComparisonReport) -> None:
    """Render algorithm comparison results as a dataframe table."""
    import pandas as pd
    rows = []
    for m in comp_report.metrics:
        row = {"Method": m.method, "Detected": m.n_detected, "Runtime (s)": f"{m.runtime_s:.3f}"}
        if comp_report.ground_truth_available:
            row["TP"] = m.tp
            row["FP"] = m.fp
            row["FN"] = m.fn
            row["Precision"] = f"{m.precision:.3f}" if m.precision > 0 else "N/A"
            row["Recall"] = f"{m.recall:.3f}" if m.recall > 0 else "N/A"
            row["F1"] = f"{m.f1:.3f}" if m.f1 > 0 else "N/A"
        if m.note:
            row["Note"] = m.note
        rows.append(row)
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

def main() -> None:
    """Main Streamlit application entry point."""
    # ---- Header -------------------------------------------------------- #
    st.markdown("""
    <div class="ecg-header">
        <h1>❤️ Adaptive ECG Signal Intelligence Engine</h1>
        <p>Real ECG → Signal Understanding → Noise Cancellation → Adaptive Processing → Explainable Results</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="disclaimer">⚠️ <strong>Disclaimer:</strong> For research and demonstration purposes only. '
        'This prototype is not a medical diagnostic device. Do not use for clinical decisions.</div>',
        unsafe_allow_html=True,
    )

    # ---- Sidebar ------------------------------------------------------- #
    with st.sidebar:
        st.markdown("## ⚙️ Settings")

        uploaded = st.file_uploader(
            "Upload ECG File",
            type=["csv", "txt", "npy", "npz", "dat", "hea"],
            help="CSV, TXT, NPY, NPZ, or WFDB (.dat/.hea) format",
        )

        fs_auto = st.checkbox("Auto-detect sampling rate", value=True)
        if fs_auto:
            fs_input = None
        else:
            fs_input = st.number_input(
                "Sampling Rate (Hz)",
                min_value=float(config.MIN_FS),
                max_value=float(config.MAX_FS),
                value=360.0,
                step=1.0,
            )

        st.markdown("---")
        st.markdown("### Channel Selection")
        channel_index = st.number_input(
            "Channel index (0 = first)",
            min_value=0, max_value=31, value=0, step=1,
        )

        st.markdown("---")
        research_mode = st.checkbox("🔬 Research Mode", value=False)
        algorithm_comparison = st.checkbox("📊 Algorithm Comparison Lab", value=False)
        show_peak_details = st.checkbox("🔍 Explainable Peak Details", value=False)
        show_hr_series = st.checkbox("📈 Instantaneous HR Series", value=True)

        st.markdown("---")
        st.caption("Adaptive ECG Engine v1.0")
        st.caption("CPU-only • Offline • No data sent externally")

    # ---- No file ------------------------------------------------------- #
    if uploaded is None:
        _render_landing()
        return

    # ---- Load ECG ------------------------------------------------------ #
    with st.spinner("Loading ECG file…"):
        try:
            record = load_ecg(
                uploaded,
                uploaded.name,
                fs_override=fs_input,
                channel_index=int(channel_index),
            )
        except Exception as exc:
            st.error(f"⚠ Unable to load ECG file.\n\n{_format_load_error(exc)}")
            return

    if record.warnings:
        for w in record.warnings:
            st.warning(w)

    # ---- File Information --------------------------------------------- #
    st.markdown('<div class="section-header">📄 File Information</div>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: st.metric("File", uploaded.name[:20])
    with c2: st.metric("Samples", f"{record.n_samples:,}")
    with c3: st.metric("Duration", f"{record.duration_s:.1f} s")
    with c4: st.metric("Sampling Rate", f"{record.fs:.0f} Hz")
    with c5:
        n_ch = len(record.channel_names) if record.channel_names else 1
        st.metric("Channels", str(n_ch))

    if record.channel_names and len(record.channel_names) > 1:
        st.info(f"Active channel: **{record.channel_name}** (index {channel_index})")

    # ---- BEFORE: Original ECG ----------------------------------------- #
    st.markdown('<div class="section-header">📈 BEFORE — Original ECG</div>', unsafe_allow_html=True)
    st.plotly_chart(
        plot_raw_ecg(record.time_axis, record.signal, f"Original ECG — {record.channel_name}"),
        use_container_width=True,
    )

    # ---- Signal Quality Analysis --------------------------------------- #
    st.markdown('<div class="section-header">🔬 Signal Quality Analysis</div>', unsafe_allow_html=True)
    qe = QualityEngine(record.fs)

    with st.spinner("Analysing signal quality…"):
        quality_report = qe.assess_recording(record.signal)

    qcol1, qcol2 = st.columns([1, 1])
    with qcol1:
        st.markdown(f"**Overall Quality:** {quality_badge(quality_report.overall)}",
                    unsafe_allow_html=True)
        st.write("")
        rows = [
            ("Noise", quality_report.noise_level),
            ("Baseline Drift", quality_report.baseline_drift),
            ("Saturation", quality_report.saturation),
            ("Missing Samples", quality_report.missing_samples),
            ("QRS Visibility", quality_report.qrs_visibility),
        ]
        for label, level in rows:
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid #f0f0f0'>"
                f"<span>{label}</span>{quality_badge(level)}</div>",
                unsafe_allow_html=True,
            )

    with qcol2:
        st.markdown(f"**SNR Estimate:** {quality_report.snr_db:.1f} dB")
        st.markdown(f"**Kurtosis:** {quality_report.kurtosis_value:.2f}")
        st.markdown(f"**Missing fraction:** {quality_report.missing_fraction*100:.2f}%")
        st.markdown(f"**Saturation fraction:** {quality_report.saturation_fraction*100:.2f}%")
        pl_50 = "✔ Detected" if quality_report.powerline_50hz else "✖ Not detected"
        pl_60 = "✔ Detected" if quality_report.powerline_60hz else "✖ Not detected"
        st.markdown(f"**50 Hz interference:** {pl_50}")
        st.markdown(f"**60 Hz interference:** {pl_60}")
        if quality_report.messages:
            for msg in quality_report.messages:
                st.caption(f"ℹ️ {msg}")

    # ---- Run adaptive pipeline ----------------------------------------- #
    st.markdown('<div class="section-header">⚡ Adaptive Processing Pipeline</div>', unsafe_allow_html=True)

    pipeline_placeholder = st.empty()
    pipeline_placeholder.info("⏳ Running adaptive pipeline…")

    with st.spinner("Processing ECG…"):
        t_pipeline_start = time.perf_counter()
        try:
            monitor = qe.monitor_segments(record.signal)
            engine = AdaptiveECGEngine(record.fs)
            result = engine.process(record.signal, monitor_report=monitor)
            result.processing_time_load_s = record.load_time_s
        except Exception as exc:
            pipeline_placeholder.empty()
            st.error(f"⚠ Processing failed: {exc}")
            st.exception(exc)
            return

    pipeline_placeholder.success(
        f"✓ Pipeline complete in {result.processing_time_total_s:.2f} s"
    )

    # Pipeline timing breakdown
    with st.expander("Pipeline Timing", expanded=False):
        pc1, pc2, pc3, pc4 = st.columns(4)
        with pc1: st.metric("Loading", f"{result.processing_time_load_s:.2f} s")
        with pc2: st.metric("Preprocessing", f"{result.processing_time_preproc_s:.2f} s")
        with pc3: st.metric("Detection", f"{result.processing_time_detect_s:.2f} s")
        with pc4: st.metric("Total", f"{result.processing_time_total_s:.2f} s")
        if result.applied_notch_50:
            st.info("50 Hz notch filter applied (powerline interference detected).")
        if result.applied_notch_60:
            st.info("60 Hz notch filter applied (powerline interference detected).")

    # ---- Adaptive Signal Monitor -------------------------------------- #
    st.markdown('<div class="section-header">📡 Adaptive Signal Monitor</div>', unsafe_allow_html=True)
    monitor_html = render_monitor_timeline(monitor)
    st.markdown(monitor_html, unsafe_allow_html=True)

    if monitor.sensor_change_detected:
        st.warning("⚠ Possible sensor/device characteristic change detected. Signal characteristics shifted significantly.")

    if monitor.change_points:
        cp_str = ", ".join(f"{_fmt_time(t)}" for t in monitor.change_points[:10])
        st.info(f"Change-point(s) detected at: {cp_str}")

    if monitor.noise_profile_changes:
        nc_str = ", ".join(f"{_fmt_time(t)}" for t in monitor.noise_profile_changes[:10])
        st.warning(f"Noise profile change(s) at: {nc_str}")

    with st.expander("Change Detection Flow", expanded=False):
        st.markdown("""
        ```
        CHANGE DETECTED
               ↓
        REASSESS QUALITY
               ↓
        ADAPT PROCESSING
               ↓
        ADAPT DETECTION
               ↓
        VALIDATE PEAKS
        ```
        """)

    # ---- AFTER: Filtered ECG + Peaks ---------------------------------- #
    st.markdown('<div class="section-header">✅ AFTER — Filtered ECG + Detected R-Peaks</div>', unsafe_allow_html=True)
    st.plotly_chart(
        plot_filtered_with_peaks(
            record.time_axis, result.filtered,
            result.r_peaks, result.rejected_peaks, record.fs
        ),
        use_container_width=True,
    )

    # ---- Result Cards ------------------------------------------------- #
    st.markdown('<div class="section-header">📊 Results</div>', unsafe_allow_html=True)
    rc1, rc2, rc3, rc4, rc5 = st.columns(5)

    hr_display = f"{result.heart_rate_bpm:.0f}" if result.heart_rate_bpm > 0 else "N/A"
    rel_pct = int(result.detection_reliability * 100)
    rel_label = "HIGH" if rel_pct >= 75 else ("MODERATE" if rel_pct >= 50 else "LOW")
    qual_map = {
        QualityLevel.GOOD: "GOOD", QualityLevel.MODERATE: "MODERATE",
        QualityLevel.POOR: "POOR", QualityLevel.CRITICAL: "CRITICAL"
    }

    with rc1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="value">{hr_display}</div>
            <div class="label">Heart Rate (BPM)</div>
        </div>""", unsafe_allow_html=True)
    with rc2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="value">{len(result.r_peaks)}</div>
            <div class="label">R-Peaks</div>
        </div>""", unsafe_allow_html=True)
    with rc3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="value">{qual_map[quality_report.overall]}</div>
            <div class="label">Signal Quality</div>
        </div>""", unsafe_allow_html=True)
    with rc4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="value">{rel_label}</div>
            <div class="label">Detection Reliability</div>
        </div>""", unsafe_allow_html=True)
    with rc5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="value">{result.processing_time_total_s:.2f}s</div>
            <div class="label">Processing Time</div>
        </div>""", unsafe_allow_html=True)

    # Reliability detail
    st.write("")
    st.markdown("**Detection Reliability**")
    st.markdown(reliability_bar(result.detection_reliability), unsafe_allow_html=True)

    # RR stats
    if len(result.rr_intervals_s) > 0:
        valid_rr = result.rr_intervals_s[
            (result.rr_intervals_s >= 0.2) & (result.rr_intervals_s <= 3.0)
        ]
        if len(valid_rr) > 0:
            rr_col1, rr_col2, rr_col3, rr_col4 = st.columns(4)
            with rr_col1: st.metric("Median RR", f"{np.median(valid_rr)*1000:.0f} ms")
            with rr_col2: st.metric("Min RR", f"{valid_rr.min()*1000:.0f} ms")
            with rr_col3: st.metric("Max RR", f"{valid_rr.max()*1000:.0f} ms")
            with rr_col4: st.metric("RR Std Dev", f"{valid_rr.std()*1000:.0f} ms")

    if result.heart_rate_bpm <= 0:
        st.warning("⚠ Heart-rate estimate unreliable. Too few valid R-R intervals detected.")

    # ---- Before / After Comparison ------------------------------------ #
    st.markdown('<div class="section-header">🔄 Before vs After Comparison</div>', unsafe_allow_html=True)
    st.plotly_chart(
        plot_before_after(
            record.time_axis, record.signal,
            result.filtered, result.r_peaks, record.fs
        ),
        use_container_width=True,
    )

    # ---- Instantaneous HR series -------------------------------------- #
    if show_hr_series and len(result.hr_times_s) > 1:
        st.markdown('<div class="section-header">📈 Instantaneous Heart Rate</div>', unsafe_allow_html=True)
        st.plotly_chart(plot_hr_series(result.hr_times_s, result.hr_series_bpm),
                        use_container_width=True)

    # ---- Explainable peak details ------------------------------------- #
    if show_peak_details:
        st.markdown('<div class="section-header">🔍 Explainable Peak Details</div>', unsafe_allow_html=True)
        accepted = [d for d in result.peak_details if d.accepted][:100]
        rejected_list = [d for d in result.peak_details if not d.accepted][:50]

        st.markdown(f"Showing {len(accepted)} accepted peaks and {len(rejected_list)} rejected candidates.")
        tab1, tab2 = st.tabs(["Accepted R-Peaks", "Rejected Candidates"])
        with tab1:
            for d in accepted:
                render_peak_detail(d)
        with tab2:
            for d in rejected_list:
                render_peak_detail(d)

    # ---- Research Mode ------------------------------------------------ #
    if research_mode:
        st.markdown('<div class="section-header">🔬 Research Mode — Internal Signals</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(plot_research_mode(result, record.time_axis),
                        use_container_width=True)

    # ---- Algorithm Comparison ---------------------------------------- #
    if algorithm_comparison:
        st.markdown('<div class="section-header">📊 Algorithm Comparison Lab</div>', unsafe_allow_html=True)

        ref_available = record.annotations is not None
        if not ref_available:
            st.info(
                "Ground-truth annotations not available for this recording. "
                "Load a WFDB recording with an .atr annotation file to enable accuracy metrics. "
                "Showing peak counts and runtimes only."
            )

        with st.spinner("Running comparison algorithms…"):
            try:
                comp_report = compare_algorithms(
                    ecg=record.signal,
                    fs=record.fs,
                    adaptive_peaks=result.r_peaks,
                    adaptive_runtime_s=result.processing_time_total_s,
                    reference_peaks=record.annotations,
                )
                render_comparison_table(comp_report)
                if not ref_available:
                    st.markdown(
                        "> Ground-truth accuracy unavailable for this recording. "
                        "Load a WFDB .atr annotation file for Precision / Recall / F1."
                    )
            except Exception as exc:
                st.error(f"Algorithm comparison error: {exc}")

    # ---- Warnings ---------------------------------------------------- #
    if result.warnings:
        with st.expander("Processing Warnings", expanded=False):
            for w in result.warnings:
                st.warning(w)


# ---------------------------------------------------------------------------
# Landing page (no file)
# ---------------------------------------------------------------------------

def _render_landing() -> None:
    """Render the landing/welcome page when no file is uploaded."""
    st.markdown("""## How to use

1. **Upload** a real ECG recording using the sidebar.
2. The engine will automatically:
   - Display the original ECG.
   - Analyse signal quality.
   - Detect and characterise noise.
   - Remove baseline drift and apply adaptive filtering.
   - Detect R-peaks using the adaptive engine.
   - Calculate heart rate.
   - Show before/after comparison.
3. Enable **Research Mode** to see internal algorithm signals.
4. Enable **Algorithm Comparison** to benchmark against established methods.

---

### Supported Formats

| Format | Description |
|--------|-------------|
| CSV | Time+ECG, ECG-only, or multi-channel |
| TXT | Whitespace-separated numeric data |
| NPY | NumPy array |
| NPZ | NumPy archive (keys: ecg, signal, data) |
| WFDB | PhysioNet .dat/.hea (requires `wfdb` package) |

### Suggested ECG Sources

- [PhysioNet MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/)
- [MIT-BIH Normal Sinus Rhythm](https://physionet.org/content/nsrdb/)
- [BIDMC ECG Database](https://physionet.org/content/bidmc/)

---

### Key Features

- **Adaptive noise cancellation** — adjusts to changing signal characteristics
- **Baseline drift correction** — removes slow baseline movement
- **Powerline interference suppression** — 50/60 Hz notch (only when detected)
- **Adaptive R-peak detection** — adaptive thresholds, refractory period, search-back
- **Explainable decisions** — every peak decision explained
- **Research mode** — see every stage of the Pan-Tompkins pipeline
- **Algorithm comparison** — Pan-Tompkins, Hamilton, Christov

---

> ⚠️ **Medical disclaimer:** For research and demonstration purposes only. Not a medical device.
""")


# ---------------------------------------------------------------------------
# Error formatting
# ---------------------------------------------------------------------------

def _format_load_error(exc: Exception) -> str:
    """Format a load error with hints for the user."""
    msg = str(exc)
    hints = [
        "• Invalid CSV or TXT format",
        "• Missing ECG column",
        "• Invalid or missing sampling rate",
        "• Recording too short (minimum 2 seconds)",
        "• Unsupported data format",
        "• Corrupted or non-numeric values",
    ]
    hint_str = "\n".join(hints)
    return f"**Error:** {msg}\n\n**Possible causes:**\n{hint_str}"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
