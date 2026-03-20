from __future__ import annotations

from core.auth import require_login, render_user_menu

require_login()
render_user_menu()

from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from core.phase import analyze_phase

st.set_page_config(page_title="Watermelon System | Phase Analysis", layout="wide")


def apply_page_style() -> None:
    st.markdown(
        """
        <style>
        .main > div {
            padding-top: 0.18rem;
        }

        .stApp {
            background-color: #f3f4f6;
        }

        section[data-testid="stSidebar"] {
            background: #e5e7eb;
            border-right: 1px solid #cbd5e1;
        }

        section.main div[data-testid="stButton"] > button,
        section.main div[data-testid="stDownloadButton"] > button {
            min-height: 52px;
            border-radius: 16px;
            font-weight: 700;
            border: 1px solid #bfd8ff !important;
            background: linear-gradient(180deg, #f8fbff 0%, #eef6ff 100%) !important;
            color: #2563eb !important;
            box-shadow: 0 8px 20px rgba(37, 99, 235, 0.08);
            transition: all 0.18s ease;
        }

        section.main div[data-testid="stButton"] > button:hover,
        section.main div[data-testid="stDownloadButton"] > button:hover {
            border-color: #93c5fd !important;
            background: linear-gradient(180deg, #ffffff 0%, #f3f8ff 100%) !important;
            color: #1d4ed8 !important;
            box-shadow: 0 12px 24px rgba(37, 99, 235, 0.12);
        }

        section.main div[data-testid="stButton"] > button *,
        section.main div[data-testid="stDownloadButton"] > button *,
        section.main div[data-testid="stButton"] > button p,
        section.main div[data-testid="stDownloadButton"] > button p,
        section.main div[data-testid="stButton"] > button span,
        section.main div[data-testid="stDownloadButton"] > button span,
        section.main div[data-testid="stButton"] > button div,
        section.main div[data-testid="stDownloadButton"] > button div {
            color: #2563eb !important;
        }

        .wm-phase-shell {
            padding-top: 0.1rem;
        }

        .wm-phase-title {
            font-size: 0.82rem;
            font-weight: 800;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: #64748b;
            margin-bottom: 0.75rem;
        }

        .wm-phase-summary-card {
            background: linear-gradient(180deg, rgba(255,255,255,0.95), rgba(255,255,255,0.88));
            border: 1px solid #dbe3ee;
            border-radius: 20px;
            padding: 18px 18px 14px 18px;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
            min-height: 170px;
            margin-bottom: 0.9rem;
        }

        .wm-phase-name {
            font-size: 1rem;
            font-weight: 800;
            color: #0f172a;
            margin-bottom: 14px;
            line-height: 1.2;
        }

        .wm-phase-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 10px 14px;
        }

        .wm-phase-k {
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #64748b;
            margin-bottom: 3px;
        }

        .wm-phase-v {
            font-size: 1.15rem;
            font-weight: 800;
            color: #0f172a;
            line-height: 1.05;
        }

        .wm-phase-sub {
            font-size: 0.88rem;
            color: #64748b;
            margin-top: 2px;
        }

        .wm-phase-status {
            display: inline-block;
            margin-top: 14px;
            padding: 8px 12px;
            border-radius: 999px;
            border: 1px solid #dbeafe;
            background: #f8fbff;
            color: #2563eb;
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.04em;
        }

        .wm-phase-export-actions {
            margin-top: 1rem;
            margin-bottom: 0.2rem;
        }

        div[data-testid="stExpander"] {
            border: 1px solid #dbe3ee;
            border-radius: 16px;
            background: rgba(255,255,255,0.65);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def format_number(value, digits=3, fallback="—") -> str:
    if value is None:
        return fallback
    try:
        v = float(value)
        if not np.isfinite(v):
            return fallback
        return f"{v:.{digits}f}"
    except Exception:
        return fallback


def stability_badge_value(phase_stability_deg: float | None) -> str:
    if phase_stability_deg is None or not np.isfinite(phase_stability_deg):
        return "No stability data"
    if phase_stability_deg <= 3.0:
        return "Stable phase"
    if phase_stability_deg <= 8.0:
        return "Moderate variation"
    return "High variation"


def build_summary_card_html(row: pd.Series) -> str:
    return f"""
    <div class="wm-phase-summary-card">
        <div class="wm-phase-name">{row["Signal"]}</div>
        <div class="wm-phase-grid">
            <div>
                <div class="wm-phase-k">1X Amplitude</div>
                <div class="wm-phase-v">{row["1X Amplitude (PP)"]}</div>
            </div>
            <div>
                <div class="wm-phase-k">1X Phase</div>
                <div class="wm-phase-v">{row["1X Phase (deg)"]}</div>
            </div>
            <div>
                <div class="wm-phase-k">Phase Stability</div>
                <div class="wm-phase-v">{row["Phase Stability (deg)"]}</div>
            </div>
            <div>
                <div class="wm-phase-k">Mean RPM</div>
                <div class="wm-phase-v">{row["Mean RPM"]}</div>
            </div>
        </div>
        <div class="wm-phase-status">{row["Status"]}</div>
    </div>
    """


def build_export_safe_table_figure(summary_df: pd.DataFrame) -> go.Figure:
    header_values = list(summary_df.columns)
    cell_values = [summary_df[col].tolist() for col in summary_df.columns]

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=header_values,
                    fill_color="#eaf3ff",
                    line_color="#dbe3ee",
                    align="left",
                    font=dict(color="#0f172a", size=22, family="Arial"),
                    height=42,
                ),
                cells=dict(
                    values=cell_values,
                    fill_color="#ffffff",
                    line_color="#e5e7eb",
                    align="left",
                    font=dict(color="#111827", size=20, family="Arial"),
                    height=38,
                ),
            )
        ]
    )

    n_rows = max(len(summary_df), 1)
    fig.update_layout(
        width=4200,
        height=max(1200, 280 + n_rows * 74),
        margin=dict(l=60, r=60, t=70, b=50),
        paper_bgcolor="#f3f4f6",
        plot_bgcolor="#f3f4f6",
    )
    return fig


def build_export_png_bytes(summary_df: pd.DataFrame):
    try:
        export_fig = build_export_safe_table_figure(summary_df)
        png_bytes = export_fig.to_image(format="png", width=4200, height=export_fig.layout.height, scale=2)
        return png_bytes, None
    except Exception as e:
        return None, str(e)


apply_page_style()

if "wm_phase_export_png_bytes" not in st.session_state:
    st.session_state.wm_phase_export_png_bytes = None
if "wm_phase_export_png_key" not in st.session_state:
    st.session_state.wm_phase_export_png_key = None
if "wm_phase_export_error" not in st.session_state:
    st.session_state.wm_phase_export_error = None

if "signals" not in st.session_state or not st.session_state["signals"]:
    st.warning("No signals loaded.")
    st.stop()

signal_names = list(st.session_state["signals"].keys())

default_selection = signal_names[: min(4, len(signal_names))]

with st.sidebar:
    st.markdown("### Phase Processing")

    selected_signals = st.multiselect(
        "Select Signals",
        options=signal_names,
        default=default_selection,
    )

if not selected_signals:
    st.warning("Select at least one signal.")
    st.stop()

summary_rows = []
diagnostics_rows = []

for signal_name in selected_signals:
    signal = st.session_state["signals"][signal_name]
    result = analyze_phase(signal)

    phase_per_rev = np.asarray(result.get("phase_per_rev_deg", []), dtype=float)
    amp_per_rev = np.asarray(result.get("amplitude_pp_per_rev", []), dtype=float)
    complex_1x = np.asarray(result.get("complex_1x_per_rev", []))
    debug = result.get("debug", {})

    mean_amp = result.get("mean_amplitude_pp")
    mean_phase = result.get("mean_phase_deg")
    phase_stability = result.get("phase_stability_deg")
    mean_rpm = result.get("mean_rpm", 0.0)

    summary_rows.append(
        {
            "Signal": signal_name,
            "1X Amplitude (PP)": f'{format_number(mean_amp, 3)}',
            "1X Phase (deg)": f'{format_number(mean_phase, 2)}',
            "Phase Stability (deg)": f'{format_number(phase_stability, 2)}',
            "Mean RPM": f'{format_number(mean_rpm, 1)}',
            "Status": stability_badge_value(phase_stability),
        }
    )

    diagnostics_rows.append(
        {
            "Signal": signal_name,
            "Mode": result.get("mode", "-"),
            "Samples/Rev": int(result.get("samples_per_rev", 0) or 0),
            "Revolutions": int(result.get("n_revs", 0) or 0),
            "Sync Source": str(debug.get("sync_source", "-")),
            "Header Revs": int(debug.get("header_number_of_revs", 0) or 0),
            "Time-Inferred Revs": int(debug.get("inferred_revs_from_time", 0) or 0),
            "Mean Phase": format_number(mean_phase, 2),
            "Phase Stability": format_number(phase_stability, 2),
            "Mean Amplitude": format_number(mean_amp, 3),
            "C1 Count": int(len(complex_1x)),
            "Phase Convention": str(debug.get("phase_convention", "-")),
        }
    )

summary_df = pd.DataFrame(summary_rows)
diagnostics_df = pd.DataFrame(diagnostics_rows)

export_state_key = "|".join(
    [
        ",".join(selected_signals),
        *summary_df.astype(str).fillna("").values.flatten().tolist(),
    ]
)

if st.session_state.wm_phase_export_png_key != export_state_key:
    st.session_state.wm_phase_export_png_bytes = None
    st.session_state.wm_phase_export_png_key = export_state_key
    st.session_state.wm_phase_export_error = None

st.markdown('<div class="wm-phase-shell">', unsafe_allow_html=True)
st.markdown('<div class="wm-phase-title">Phase Comparison</div>', unsafe_allow_html=True)

for row_start in range(0, len(summary_rows), 3):
    chunk = summary_rows[row_start:row_start + 3]
    cols = st.columns(3, gap="large")
    for i, row in enumerate(chunk):
        with cols[i]:
            st.markdown(build_summary_card_html(pd.Series(row)), unsafe_allow_html=True)

st.dataframe(summary_df, use_container_width=True, hide_index=True)

with st.expander("Technical Diagnostics", expanded=False):
    st.dataframe(diagnostics_df, use_container_width=True, hide_index=True)

st.markdown('<div class="wm-phase-export-actions"></div>', unsafe_allow_html=True)

left_pad, col_export1, col_export2, right_pad = st.columns([2.4, 1.3, 1.3, 2.4])

with col_export1:
    if st.button("Prepare PNG HD", use_container_width=True):
        with st.spinner("Generating HD export..."):
            png_bytes, export_error = build_export_png_bytes(summary_df)
            st.session_state.wm_phase_export_png_bytes = png_bytes
            st.session_state.wm_phase_export_error = export_error

with col_export2:
    if st.session_state.wm_phase_export_png_bytes is not None:
        st.download_button(
            "Download PNG HD",
            data=st.session_state.wm_phase_export_png_bytes,
            file_name="watermelon_phase_analysis_hd.png",
            mime="image/png",
            use_container_width=True,
        )
    else:
        st.button("Download PNG HD", disabled=True, use_container_width=True)

if st.session_state.wm_phase_export_error:
    st.warning(f"PNG export error: {st.session_state.wm_phase_export_error}")

st.markdown("</div>", unsafe_allow_html=True)