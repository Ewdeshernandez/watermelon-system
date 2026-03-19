from core.auth import require_login, render_user_menu

require_login()
render_user_menu()

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.order_tracking import analyze_order_tracking
from core.ui.page_header import render_page_header
from core.plot_export import export_plot_png


st.set_page_config(layout="wide")

render_page_header(
    title="Order Tracking",
    subtitle="Amplitude and phase by order across revolutions"
)

if "signals" not in st.session_state or not st.session_state["signals"]:
    st.warning("No signals loaded.")
    st.stop()

signal_names = list(st.session_state["signals"].keys())

selected_signal = st.sidebar.selectbox(
    "Select Signal",
    signal_names,
    index=0,
)

max_order = st.sidebar.slider(
    "Max Order",
    min_value=3,
    max_value=8,
    value=5,
    step=1,
)

signal = st.session_state["signals"][selected_signal]
result = analyze_order_tracking(signal, max_order=max_order)

orders = result["orders"]
order_results = result["order_results"]
dominant_order = result["dominant_order"]
rev_idx = np.asarray(result["rev_idx"], dtype=int)

dominant = order_results[dominant_order]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Dominant Order", f"{dominant_order}X")
col2.metric("Dominant Amp (PP)", f"{dominant['mean_amp_pp']:.3f}")
col3.metric("Dominant Phase (deg)", f"{dominant['mean_phase_deg']:.2f}")
col4.metric("Mean RPM", f"{result.get('mean_rpm', 0.0):.1f}")

col5, col6, col7, col8 = st.columns(4)
col5.metric("Revolutions Used", f"{result['n_revs']}")
col6.metric("Samples / Rev", f"{result['samples_per_rev']}")
col7.metric("Dominant Max Amp", f"{dominant['max_amp_pp']:.3f}")
col8.metric("Phase Stability", f"{dominant['phase_stability_deg']:.2f}")

# =========================
# FIG 1 → ORDER AMPLITUDE BAR
# =========================
mean_amps = [order_results[o]["mean_amp_pp"] for o in orders]
labels = [f"{o}X" for o in orders]

fig_bar = go.Figure()
fig_bar.add_trace(
    go.Bar(
        x=labels,
        y=mean_amps,
        name="Mean Order Amplitude",
        hovertemplate="Order %{x}<br>Amp %{y:.3f}<extra></extra>",
    )
)

fig_bar.update_layout(
    title="Mean Amplitude by Order",
    xaxis_title="Order",
    yaxis_title="Amplitude (PP)",
    template="plotly_white",
    height=420,
    margin=dict(l=20, r=20, t=60, b=20),
)

st.plotly_chart(fig_bar, use_container_width=True)

# =========================
# FIG 2 → DOMINANT ORDER AMP VS REV
# =========================
fig_amp = go.Figure()
fig_amp.add_trace(
    go.Scatter(
        x=rev_idx,
        y=dominant["amp_pp_per_rev"],
        mode="lines+markers",
        name=f"{dominant_order}X Amplitude",
        line=dict(width=2),
        marker=dict(size=7),
        hovertemplate="Rev %{x}<br>Amp %{y:.3f}<extra></extra>",
    )
)

fig_amp.update_layout(
    title=f"{dominant_order}X Amplitude vs Revolution",
    xaxis_title="Revolution",
    yaxis_title="Amplitude (PP)",
    template="plotly_white",
    height=420,
    margin=dict(l=20, r=20, t=60, b=20),
)

st.plotly_chart(fig_amp, use_container_width=True)

# =========================
# FIG 3 → DOMINANT ORDER PHASE VS REV
# =========================
fig_phase = go.Figure()
fig_phase.add_trace(
    go.Scatter(
        x=rev_idx,
        y=dominant["phase_deg_per_rev"],
        mode="lines+markers",
        name=f"{dominant_order}X Phase",
        line=dict(width=2),
        marker=dict(size=7),
        hovertemplate="Rev %{x}<br>Phase %{y:.2f}°<extra></extra>",
    )
)

fig_phase.update_layout(
    title=f"{dominant_order}X Phase vs Revolution",
    xaxis_title="Revolution",
    yaxis_title="Phase (deg)",
    template="plotly_white",
    height=420,
    margin=dict(l=20, r=20, t=60, b=20),
)

st.plotly_chart(fig_phase, use_container_width=True)

# =========================
# EXPORT PNG HD
# =========================
colA, colB = st.columns(2)

if "order_tracking_png_ready" not in st.session_state:
    st.session_state["order_tracking_png_ready"] = None

if colA.button("Prepare PNG HD"):
    st.session_state["order_tracking_png_ready"] = export_plot_png(fig_amp)

if colB.button("Download PNG HD"):
    if st.session_state["order_tracking_png_ready"] is not None:
        st.download_button(
            label="Download",
            data=st.session_state["order_tracking_png_ready"],
            file_name=f"{selected_signal}_order_tracking.png",
            mime="image/png"
        )
    else:
        st.warning("Prepare image first")

# =========================
# DEBUG
# =========================
with st.expander("Technical Diagnostics", expanded=False):
    debug = result.get("debug", {})

    info_cols = st.columns(5)
    info_cols[0].metric("Mode", result.get("mode", "-"))
    info_cols[1].metric("FS Used", f"{debug.get('fs_used', 0.0):.1f}")
    info_cols[2].metric("RPM Used", f"{debug.get('rpm_used', 0.0):.1f}")
    info_cols[3].metric("Usable Samples", int(debug.get("usable_samples", 0) or 0))
    info_cols[4].metric("Dominant Order", f"{debug.get('dominant_order', 0)}X")

    rows = []
    for order in orders:
        data = order_results[order]
        rows.append(
            {
                "Order": f"{order}X",
                "Mean_Amp_PP": data["mean_amp_pp"],
                "Max_Amp_PP": data["max_amp_pp"],
                "Mean_Phase_deg": data["mean_phase_deg"],
                "Phase_Stability_deg": data["phase_stability_deg"],
            }
        )

    st.markdown("**Order Summary**")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)