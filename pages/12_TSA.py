from core.auth import require_login, render_user_menu

require_login()
render_user_menu()

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.tsa import analyze_tsa
from core.ui.page_header import render_page_header
from core.plot_export import export_plot_png  # 👈 NUEVO


st.set_page_config(layout="wide")

render_page_header(
    title="TSA",
    subtitle="Time synchronous averaging for synchronous waveform enhancement"
)

# =========================
# DATA CHECK
# =========================
if "signals" not in st.session_state or not st.session_state["signals"]:
    st.warning("No signals loaded.")
    st.stop()

signal_names = list(st.session_state["signals"].keys())

selected_signal = st.sidebar.selectbox(
    "Select Signal",
    signal_names,
    index=0,
)

signal = st.session_state["signals"][selected_signal]
result = analyze_tsa(signal)

# =========================
# DATA
# =========================
x_rev = np.asarray(result["x_rev"], dtype=float)
tsa_mean = np.asarray(result["tsa_mean"], dtype=float)
residual = np.asarray(result["residual_waveforms"], dtype=float)
time_axis_ms = np.asarray(result["time_axis_ms"], dtype=float)

# 🔥 NUEVO (ENVELOPE)
tsa_min = np.min(x_rev, axis=0)
tsa_max = np.max(x_rev, axis=0)

n_revs = int(result["n_revs"])
samples_per_rev = int(result["samples_per_rev"])

# =========================
# METRICS
# =========================
col1, col2, col3, col4 = st.columns(4)
col1.metric("Revolutions Used", f"{n_revs}")
col2.metric("Samples / Rev", f"{samples_per_rev}")
col3.metric("TSA Peak-to-Peak", f"{result['peak_to_peak_tsa']:.3f}")
col4.metric("Mean RPM", f"{result.get('mean_rpm', 0.0):.1f}")

col5, col6, col7, col8 = st.columns(4)
col5.metric("Sync RMS", f"{result['sync_rms']:.3f}")
col6.metric("Residual RMS", f"{result['residual_rms']:.3f}")
col7.metric("Sync Ratio", f"{result['sync_ratio']:.3f}")
col8.metric("Mean Correlation", f"{result['mean_corr']:.3f}")

# =========================
# FIG 1 → TSA + ENVELOPE 🔥
# =========================
fig_tsa = go.Figure()

# Envelope (max primero)
fig_tsa.add_trace(go.Scatter(
    x=time_axis_ms,
    y=tsa_max,
    line=dict(width=0),
    showlegend=False,
    hoverinfo="skip"
))

# Envelope fill
fig_tsa.add_trace(go.Scatter(
    x=time_axis_ms,
    y=tsa_min,
    fill='tonexty',
    fillcolor='rgba(30,167,255,0.2)',
    line=dict(width=0),
    name="Dispersion Envelope"
))

# TSA mean
fig_tsa.add_trace(go.Scatter(
    x=time_axis_ms,
    y=tsa_mean,
    mode="lines",
    name="TSA Mean",
    line=dict(color="#1ea7ff", width=3),
    hovertemplate="TSA Mean<br>Time %{x:.3f} ms<br>Amp %{y:.3f}<extra></extra>",
))

fig_tsa.update_layout(
    title="TSA Mean + Dispersion Envelope",
    xaxis_title="Time within Revolution (ms)",
    yaxis_title="Amplitude",
    template="plotly_white",
    height=460,
    margin=dict(l=20, r=20, t=60, b=20),
)

st.plotly_chart(fig_tsa, use_container_width=True)

# =========================
# FIG 2 → RESIDUAL
# =========================
fig_residual = go.Figure()

residual_mean = np.mean(residual, axis=0) if residual.ndim == 2 else residual
residual_rms_trace = np.sqrt(np.mean(residual ** 2, axis=0)) if residual.ndim == 2 else np.abs(residual)

fig_residual.add_trace(go.Scatter(
    x=time_axis_ms,
    y=residual_mean,
    mode="lines",
    name="Residual Mean",
    line=dict(width=2),
))

fig_residual.add_trace(go.Scatter(
    x=time_axis_ms,
    y=residual_rms_trace,
    mode="lines",
    name="Residual RMS Envelope",
    line=dict(width=2, dash="dash"),
))

fig_residual.update_layout(
    title="Residual (Non-Synchronous Content)",
    xaxis_title="Time within Revolution (ms)",
    yaxis_title="Amplitude",
    template="plotly_white",
    height=420,
    margin=dict(l=20, r=20, t=60, b=20),
)

st.plotly_chart(fig_residual, use_container_width=True)

# =========================
# 🔥 EXPORT PNG HD (ESTILO WATERMELON)
# =========================
colA, colB = st.columns(2)

if "tsa_png_ready" not in st.session_state:
    st.session_state["tsa_png_ready"] = None

if colA.button("Prepare PNG HD"):
    st.session_state["tsa_png_ready"] = export_plot_png(fig_tsa)

if colB.button("Download PNG HD"):
    if st.session_state["tsa_png_ready"] is not None:
        st.download_button(
            label="Download",
            data=st.session_state["tsa_png_ready"],
            file_name=f"{selected_signal}_TSA.png",
            mime="image/png"
        )
    else:
        st.warning("Prepare image first")

# =========================
# DEBUG
# =========================
with st.expander("Technical Diagnostics", expanded=False):
    debug = result.get("debug", {})

    info_cols = st.columns(6)
    info_cols[0].metric("Mode", result.get("mode", "-"))
    info_cols[1].metric("Sync Source", str(debug.get("sync_source", "-")))
    info_cols[2].metric("Header Revs", int(debug.get("header_number_of_revs", 0) or 0))
    info_cols[3].metric("Time-Inferred Revs", int(debug.get("inferred_revs_from_time", 0) or 0))
    info_cols[4].metric("Variable Hint X", int(debug.get("variable_hint_samples_per_rev", 0) or 0))
    info_cols[5].metric("Variable Hint Revs", int(debug.get("variable_hint_revs", 0) or 0))