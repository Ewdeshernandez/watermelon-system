from __future__ import annotations

from core.auth import require_login, render_user_menu

require_login()
render_user_menu()

from io import BytesIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

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

        .wm-metric-card {
            background: linear-gradient(180deg, rgba(255,255,255,0.94), rgba(255,255,255,0.86));
            border: 1px solid #dbe3ee;
            border-radius: 18px;
            padding: 16px 18px 14px 18px;
            box-shadow: 0 6px 18px rgba(15, 23, 42, 0.05);
            margin-bottom: 0.4rem;
        }

        .wm-metric-label {
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #64748b;
            margin-bottom: 6px;
        }

        .wm-metric-value {
            font-size: 1.6rem;
            font-weight: 800;
            color: #0f172a;
            line-height: 1.0;
        }

        .wm-export-actions {
            margin-top: 0.9rem;
            margin-bottom: 0.25rem;
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


def _build_export_safe_figure(fig: go.Figure) -> go.Figure:
    export_fig = go.Figure()

    for trace in fig.data:
        if isinstance(trace, go.Scattergl):
            trace_json = trace.to_plotly_json()
            export_fig.add_trace(
                go.Scatter(
                    x=np.array(trace_json.get("x")) if trace_json.get("x") is not None else None,
                    y=np.array(trace_json.get("y")) if trace_json.get("y") is not None else None,
                    mode=trace_json.get("mode"),
                    line=trace_json.get("line"),
                    marker=trace_json.get("marker"),
                    fill=trace_json.get("fill"),
                    fillcolor=trace_json.get("fillcolor"),
                    hovertemplate=trace_json.get("hovertemplate"),
                    showlegend=trace_json.get("showlegend"),
                    connectgaps=trace_json.get("connectgaps", False),
                    name=trace_json.get("name"),
                )
            )
        else:
            export_fig.add_trace(trace)

    export_fig.update_layout(fig.layout)
    return export_fig


def _scale_export_figure(export_fig: go.Figure) -> go.Figure:
    fig = go.Figure(export_fig)

    new_data = []
    for trace in fig.data:
        trace_json = trace.to_plotly_json()

        if trace_json.get("type") == "scatter":
            mode = trace_json.get("mode", "")

            if "lines" in mode:
                line = dict(trace_json.get("line", {}) or {})
                line["width"] = max(4.8, float(line.get("width", 1.0)) * 2.8)
                trace_json["line"] = line

            if "markers" in mode:
                marker = dict(trace_json.get("marker", {}) or {})
                marker["size"] = max(14, float(marker.get("size", 6)) * 1.9)
                trace_json["marker"] = marker

        elif trace_json.get("type") == "scatterpolar":
            mode = trace_json.get("mode", "")

            if "lines" in mode:
                line = dict(trace_json.get("line", {}) or {})
                line["width"] = max(4.8, float(line.get("width", 1.0)) * 2.8)
                trace_json["line"] = line

            if "markers" in mode:
                marker = dict(trace_json.get("marker", {}) or {})
                marker["size"] = max(14, float(marker.get("size", 6)) * 1.9)
                trace_json["marker"] = marker

        trace_type = trace_json.get("type")
        if trace_type == "scatterpolar":
            new_data.append(go.Scatterpolar(**trace_json))
        else:
            new_data.append(go.Scatter(**trace_json))

    fig = go.Figure(data=new_data, layout=fig.layout)

    fig.update_layout(
        width=4200,
        height=2200,
        margin=dict(l=120, r=90, t=180, b=120),
        paper_bgcolor="#f3f4f6",
        plot_bgcolor="#f8fafc",
        font=dict(size=30, color="#111827"),
    )

    fig.update_xaxes(title_font=dict(size=40), tickfont=dict(size=26))
    fig.update_yaxes(title_font=dict(size=40), tickfont=dict(size=26))

    for ann in fig.layout.annotations:
        if ann.font is not None:
            ann.font.size = max(22, int((ann.font.size or 12) * 2.05))

    return fig


def build_export_png_bytes(fig: go.Figure):
    try:
        export_fig = _build_export_safe_figure(fig)
        export_fig = _scale_export_figure(export_fig)
        png_bytes = export_fig.to_image(format="png", width=4200, height=2200, scale=2)
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

with st.sidebar:
    st.markdown("### Phase Processing")

    selected_signal = st.selectbox(
        "Select Signal",
        signal_names,
        index=0,
    )

signal = st.session_state["signals"][selected_signal]
result = analyze_phase(signal)

phase_per_rev = np.asarray(result["phase_per_rev_deg"], dtype=float)
amp_per_rev = np.asarray(result["amplitude_pp_per_rev"], dtype=float)
rev_idx = np.arange(1, len(phase_per_rev) + 1)
complex_1x = np.asarray(result["complex_1x_per_rev"])

phase_fig = go.Figure()
phase_fig.add_trace(
    go.Scattergl(
        x=rev_idx,
        y=phase_per_rev,
        mode="lines+markers",
        name="1X Phase",
        line=dict(width=2.2, color="#5b9cf0"),
        marker=dict(size=7, color="#2f80ed"),
        hovertemplate="Rev %{x}<br>Phase %{y:.2f}°<extra></extra>",
        showlegend=False,
    )
)
phase_fig.update_layout(
    title="Phase vs Revolution",
    height=470,
    margin=dict(l=28, r=18, t=58, b=28),
    plot_bgcolor="#f8fafc",
    paper_bgcolor="#f3f4f6",
    font=dict(color="#111827"),
    xaxis=dict(
        title="Revolution",
        showgrid=True,
        gridcolor="rgba(148, 163, 184, 0.18)",
        zeroline=False,
        showline=True,
        linecolor="#9ca3af",
        ticks="outside",
        tickcolor="#6b7280",
        ticklen=4,
    ),
    yaxis=dict(
        title="Phase (deg)",
        showgrid=True,
        gridcolor="rgba(148, 163, 184, 0.18)",
        zeroline=False,
        showline=True,
        linecolor="#9ca3af",
        ticks="outside",
        tickcolor="#6b7280",
        ticklen=4,
    ),
)

polar_fig = go.Figure()
polar_fig.add_trace(
    go.Scatterpolar(
        r=amp_per_rev,
        theta=phase_per_rev,
        mode="lines+markers",
        name="1X Polar",
        line=dict(width=2.2, color="#5b6df0"),
        marker=dict(size=7, color="#2f80ed"),
        hovertemplate="Amp %{r:.3f} mil pp<br>Phase %{theta:.2f}°<extra></extra>",
        showlegend=False,
    )
)
polar_fig.update_layout(
    title="Polar Plot",
    height=470,
    margin=dict(l=20, r=20, t=58, b=20),
    plot_bgcolor="#f3f4f6",
    paper_bgcolor="#f3f4f6",
    font=dict(color="#111827"),
    polar=dict(
        bgcolor="#f8fafc",
        radialaxis=dict(
            showgrid=True,
            gridcolor="rgba(148, 163, 184, 0.18)",
            linecolor="#9ca3af",
            tickcolor="#6b7280",
        ),
        angularaxis=dict(
            direction="clockwise",
            rotation=90,
            gridcolor="rgba(148, 163, 184, 0.18)",
            linecolor="#9ca3af",
            tickcolor="#6b7280",
        ),
    ),
)

col_plot_1, col_plot_2 = st.columns(2, gap="large")

with col_plot_1:
    st.plotly_chart(
        phase_fig,
        use_container_width=True,
        config={"displaylogo": False},
        key="wm_phase_plot_main_view",
    )

with col_plot_2:
    st.plotly_chart(
        polar_fig,
        use_container_width=True,
        config={"displaylogo": False},
        key="wm_phase_plot_polar_view",
    )

metric_cols = st.columns(4, gap="large")

metrics = [
    ("1X Amplitude (PP)", format_number(result.get("mean_amplitude_pp"), 3)),
    ("1X Phase (deg)", format_number(result.get("mean_phase_deg"), 2)),
    ("Phase Stability (deg)", format_number(result.get("phase_stability_deg"), 2)),
    ("Mean RPM", format_number(result.get("mean_rpm", 0.0), 1)),
]

for col, (label, value) in zip(metric_cols, metrics):
    with col:
        st.markdown(
            f"""
            <div class="wm-metric-card">
                <div class="wm-metric-label">{label}</div>
                <div class="wm-metric-value">{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with st.expander("Technical Diagnostics", expanded=False):
    debug = result.get("debug", {})

    df_diag = pd.DataFrame(
        {
            "Revolution": rev_idx,
            "Amplitude_PP": amp_per_rev,
            "Phase_deg": phase_per_rev,
            "C1_Real": np.real(complex_1x),
            "C1_Imag": np.imag(complex_1x),
        }
    )
    st.dataframe(df_diag, use_container_width=True, hide_index=True)

    info_cols = st.columns(6)
    info_cols[0].metric("Mode", result.get("mode", "-"))
    info_cols[1].metric("Samples/Rev", int(result.get("samples_per_rev", 0)))
    info_cols[2].metric("Revolutions", int(result.get("n_revs", 0)))
    info_cols[3].metric("Sync Source", str(debug.get("sync_source", "-")))
    info_cols[4].metric("Header Revs", int(debug.get("header_number_of_revs", 0) or 0))
    info_cols[5].metric("Time-Inferred Revs", int(debug.get("inferred_revs_from_time", 0) or 0))

    candidate_table = debug.get("candidate_table", [])
    if candidate_table:
        st.markdown("**Geometry Candidates**")
        st.dataframe(pd.DataFrame(candidate_table), use_container_width=True, hide_index=True)

    st.caption(f"Phase convention: {debug.get('phase_convention', '-')}")
    st.caption(
        f"Variable hint: "
        f"{debug.get('variable_hint_samples_per_rev', '-')}"
        f"X / {debug.get('variable_hint_revs', '-')}"
    )

export_state_key = "|".join(
    [
        selected_signal,
        format_number(result.get("mean_amplitude_pp"), 6),
        format_number(result.get("mean_phase_deg"), 6),
        format_number(result.get("phase_stability_deg"), 6),
        format_number(result.get("mean_rpm", 0.0), 6),
        str(len(phase_per_rev)),
    ]
)

if st.session_state.wm_phase_export_png_key != export_state_key:
    st.session_state.wm_phase_export_png_bytes = None
    st.session_state.wm_phase_export_png_key = export_state_key
    st.session_state.wm_phase_export_error = None

phase_export_fig = go.Figure(phase_fig)
phase_export_fig.update_layout(title="Phase Analysis")

st.markdown('<div class="wm-export-actions"></div>', unsafe_allow_html=True)

left_pad, col_export1, col_export2, right_pad = st.columns([2.4, 1.3, 1.3, 2.4])

with col_export1:
    if st.button("Prepare PNG HD", use_container_width=True):
        with st.spinner("Generating HD export..."):
            png_bytes, export_error = build_export_png_bytes(phase_export_fig)
            st.session_state.wm_phase_export_png_bytes = png_bytes
            st.session_state.wm_phase_export_error = export_error

with col_export2:
    if st.session_state.wm_phase_export_png_bytes is not None:
        st.download_button(
            "Download PNG HD",
            data=st.session_state.wm_phase_export_png_bytes,
            file_name=f"{selected_signal}_phase_analysis_hd.png",
            mime="image/png",
            use_container_width=True,
        )
    else:
        st.button("Download PNG HD", disabled=True, use_container_width=True)

if st.session_state.wm_phase_export_error:
    st.warning(f"PNG export error: {st.session_state.wm_phase_export_error}")