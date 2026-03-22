from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go
import numpy as np

from core.auth import require_login, render_user_menu

require_login()
render_user_menu()

st.set_page_config(page_title="Reports", layout="wide")

# ============================================================
# VALIDACIÓN
# ============================================================

signals = st.session_state.get("signals", {})

if not signals:
    st.error("No hay señales cargadas. Ve a Load Data primero.")
    st.stop()

# ============================================================
# SIDEBAR CONTROL
# ============================================================

st.sidebar.title("Report Builder")

selected_signal_keys = st.sidebar.multiselect(
    "Selecciona señales",
    options=list(signals.keys())
)

block_type = st.sidebar.selectbox(
    "Tipo de bloque",
    [
        "Waveform",
        "Spectrum",
        "Tabular"
    ]
)

if "report_blocks" not in st.session_state:
    st.session_state["report_blocks"] = []

# ============================================================
# FUNCIONES CORE
# ============================================================

def build_waveform(signal):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=signal.time,
        y=signal.x,
        mode="lines",
        name="Waveform"
    ))

    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=20, b=10),
        template="plotly_white"
    )

    return fig


def build_spectrum(signal):
    y = signal.x
    n = len(y)

    fft_vals = np.fft.fft(y)
    fft_vals = np.abs(fft_vals)[:n // 2]

    freqs = np.fft.fftfreq(n, d=(signal.time[1] - signal.time[0]))[:n // 2]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=freqs,
        y=fft_vals,
        mode="lines",
        name="Spectrum"
    ))

    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=20, b=10),
        template="plotly_white"
    )

    return fig


def build_table(signal):
    import pandas as pd

    df = pd.DataFrame({
        "Time": signal.time,
        "Amplitude": signal.x
    })

    return df


# ============================================================
# AGREGAR BLOQUES
# ============================================================

if st.sidebar.button("Agregar al reporte"):

    for key in selected_signal_keys:

        signal = signals[key]

        st.session_state["report_blocks"].append({
            "type": block_type,
            "signal_key": key
        })

# ============================================================
# RENDER
# ============================================================

st.title("Report Builder")

if not st.session_state["report_blocks"]:
    st.info("Aún no hay bloques en el reporte")
    st.stop()

for i, block in enumerate(st.session_state["report_blocks"]):

    signal = signals[block["signal_key"]]

    st.markdown(f"### {block['signal_key']} — {block['type']}")

    if block["type"] == "Waveform":
        st.plotly_chart(build_waveform(signal), use_container_width=True)

    elif block["type"] == "Spectrum":
        st.plotly_chart(build_spectrum(signal), use_container_width=True)

    elif block["type"] == "Tabular":
        st.dataframe(build_table(signal), use_container_width=True)

    col1, col2 = st.columns([1, 6])

    with col1:
        if st.button("Eliminar", key=f"del_{i}"):
            st.session_state["report_blocks"].pop(i)
            st.rerun()