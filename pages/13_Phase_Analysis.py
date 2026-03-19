from core.auth import require_login, render_user_menu

require_login()
render_user_menu()

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.phase import analyze_phase
from core.ui.page_header import render_page_header


st.set_page_config(layout="wide")

render_page_header(
    title="Phase Analysis",
    subtitle="1X amplitude and phase from geometry synchronization and robust 1X fitting"
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

signal = st.session_state["signals"][selected_signal]
result = analyze_phase(signal)

phase_per_rev = np.asarray(result["phase_per_rev_deg"], dtype=float)
amp_per_rev = np.asarray(result["amplitude_pp_per_rev"], dtype=float)
rev_idx = np.arange(1, len(phase_per_rev) + 1)

col1, col2, col3, col4 = st.columns(4)
col1.metric("1X Amplitude (PP)", f"{result['mean_amplitude_pp']:.3f}")
col2.metric("1X Phase (deg)", f"{result['mean_phase_deg']:.2f}")
col3.metric("Phase Stability (deg)", f"{result['phase_stability_deg']:.2f}")
col4.metric("Mean RPM", f"{result.get('mean_rpm', 0.0):.1f}")

fig_phase = go.Figure()
fig_phase.add_trace(
    go.Scatter(
        x=rev_idx,
        y=phase_per_rev,
        mode="lines+markers",
        name="1X Phase",
        line=dict(width=2),
        marker=dict(size=7),
        hovertemplate="Rev %{x}<br>Phase %{y:.2f}°<extra></extra>",
    )
)
fig_phase.update_layout(
    title="Phase vs Revolution",
    xaxis_title="Revolution",
    yaxis_title="Phase (deg)",
    template="plotly_white",
    height=460,
    margin=dict(l=20, r=20, t=60, b=20),
)

st.plotly_chart(fig_phase, use_container_width=True)

fig_polar = go.Figure()
fig_polar.add_trace(
    go.Scatterpolar(
        r=amp_per_rev,
        theta=phase_per_rev,
        mode="lines+markers",
        name="1X Polar",
        line=dict(width=2),
        marker=dict(size=7),
        hovertemplate="Amp %{r:.3f} mil pp<br>Phase %{theta:.2f}°<extra></extra>",
    )
)
fig_polar.update_layout(
    title="Polar Plot",
    template="plotly_white",
    height=460,
    margin=dict(l=20, r=20, t=60, b=20),
    polar=dict(
        angularaxis=dict(direction="clockwise", rotation=90),
    ),
)

st.plotly_chart(fig_polar, use_container_width=True)

with st.expander("Technical Diagnostics", expanded=False):
    debug = result.get("debug", {})
    c1 = np.asarray(result["complex_1x_per_rev"])

    df_diag = pd.DataFrame(
        {
            "Revolution": rev_idx,
            "Amplitude_PP": amp_per_rev,
            "Phase_deg": phase_per_rev,
            "C1_Real": np.real(c1),
            "C1_Imag": np.imag(c1),
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

csv_export = pd.DataFrame(
    {
        "Revolution": rev_idx,
        "Amplitude_PP": amp_per_rev,
        "Phase_deg": phase_per_rev,
    }
)
csv_bytes = csv_export.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Export Phase CSV",
    data=csv_bytes,
    file_name=f"{selected_signal}_phase_analysis.csv",
    mime="text/csv",
    use_container_width=False,
)