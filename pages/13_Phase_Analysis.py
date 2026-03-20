import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import base64
from io import BytesIO

# =========================
# CONFIG
# =========================
st.set_page_config(layout="wide")

# =========================
# HELPERS (SPECTRUM HEADER)
# =========================
def get_logo_data_uri(path="assets/watermelon_logo.png"):
    try:
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        return f"data:image/png;base64,{data}"
    except:
        return None

def _draw_top_strip(fig, meta):
    logo_uri = get_logo_data_uri()

    if logo_uri:
        fig.add_layout_image(
            dict(
                source=logo_uri,
                xref="paper", yref="paper",
                x=0.01, y=1.0,  # FIX (antes 1.1)
                sizex=0.08, sizey=0.08,
                xanchor="left", yanchor="top"
            )
        )

    text = (
        f"<b>{meta['machine']}</b> &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"{meta['point']} &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"{meta['config']} &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"RPM: {meta['rpm']:.0f} &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"Peak: {meta['peak']:.2f} &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"{meta['timestamp']}"
    )

    fig.add_annotation(
        x=0.12, y=0.98,  # FIX (antes 1.08)
        xref="paper", yref="paper",
        text=text,
        showarrow=False,
        align="left",
        font=dict(size=14)
    )

# =========================
# CORE CALC
# =========================
def compute_phase_metrics(signal):
    x = np.array(signal["y"])
    rpm = np.mean(signal.get("rpm", [1800]))

    fft_vals = np.fft.fft(x)
    amp_1x = np.max(x) - np.min(x)
    phase_1x = np.angle(fft_vals[1])
    phase_std = np.std(np.unwrap(np.angle(fft_vals)))

    stability = "Stable" if phase_std < 0.1 else "Variation"

    return {
        "Signal": signal["name"],
        "1X Amplitude": amp_1x,
        "Phase (deg)": np.degrees(phase_1x),
        "Stability": stability,
        "RPM": rpm,
        "timestamp": signal.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    }

# =========================
# LOAD SIGNALS
# =========================
signals = st.session_state.get("signals", [])

if not signals:
    st.warning("No signals loaded")
    st.stop()

# =========================
# SIDEBAR
# =========================
signal_names = [s["name"] for s in signals]

selected_names = st.sidebar.multiselect(
    "Select Signals",
    signal_names,
    default=signal_names[:3]
)

selected_signals = [s for s in signals if s["name"] in selected_names]

if not selected_signals:
    st.warning("Select at least one signal")
    st.stop()

# =========================
# METRICS
# =========================
metrics = [compute_phase_metrics(s) for s in selected_signals]
df = pd.DataFrame(metrics)

# =========================
# HEADER (FIXED)
# =========================
meta = {
    "machine": "Watermelon Machine",
    "point": "Phase Comparison",
    "config": "Phase Analysis | 1X",
    "rpm": df["RPM"].iloc[0],
    "peak": df["1X Amplitude"].iloc[0],
    "timestamp": df["timestamp"].iloc[0]
}

fig_header = go.Figure()

# 🔥 FIX CRÍTICO
fig_header.add_trace(go.Scatter(x=[0], y=[0], mode="markers", marker_opacity=0))

fig_header.update_xaxes(visible=False)
fig_header.update_yaxes(visible=False)

fig_header.update_layout(
    height=120,
    margin=dict(l=0, r=0, t=0, b=0),
    plot_bgcolor="white",
    paper_bgcolor="white"
)

_draw_top_strip(fig_header, meta)

st.plotly_chart(fig_header, use_container_width=True)

# =========================
# TABLE
# =========================
st.markdown("### 1X Phase Comparison")

df_display = df.drop(columns=["timestamp"])

st.dataframe(
    df_display,
    use_container_width=True,
    height=400
)

# =========================
# EXPORT PNG
# =========================
def export_table_png(df):
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers", marker_opacity=0))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)

    _draw_top_strip(fig, meta)

    fig.add_trace(go.Table(
        header=dict(
            values=list(df.columns),
            fill_color="#1ea7ff",
            font=dict(color="white", size=12),
            align="center"
        ),
        cells=dict(
            values=[df[col] for col in df.columns],
            fill_color="white",
            align="center"
        )
    ))

    fig.update_layout(
        width=1400,
        height=600,
        margin=dict(l=20, r=20, t=60, b=20)
    )

    return fig

if st.button("Export PNG"):
    fig = export_table_png(df_display)

    buf = BytesIO()
    fig.write_image(buf, format="png", scale=3)

    st.download_button(
        "Download PNG",
        buf.getvalue(),
        "phase_1x_dashboard.png",
        "image/png"
    )