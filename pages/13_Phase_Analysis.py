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
# HELPERS
# =========================
def get_logo_data_uri(path="assets/watermelon_logo.png"):
    try:
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        return f"data:image/png;base64,{data}"
    except:
        return None

def render_header(meta):
    logo_uri = get_logo_data_uri()

    st.markdown(f"""
    <div style="
        background-color:white;
        padding:12px 18px;
        border-radius:10px;
        box-shadow:0px 2px 8px rgba(0,0,0,0.08);
        display:flex;
        align-items:center;
        gap:15px;
    ">
        {f"<img src='{logo_uri}' style='height:40px;'/>" if logo_uri else ""}
        <div style="font-size:14px;">
            <b>{meta['machine']}</b> |
            {meta['point']} |
            {meta['config']} |
            RPM: {meta['rpm']:.0f} |
            Peak: {meta['peak']:.2f} |
            {meta['timestamp']}
        </div>
    </div>
    """, unsafe_allow_html=True)

# =========================
# CORE CALC
# =========================
def compute_phase_metrics(signal):
    try:
        x = np.array(signal.get("y", []), dtype=float)

        if len(x) < 2:
            return None

        rpm = np.mean(signal.get("rpm", [1800]))

        fft_vals = np.fft.fft(x)

        amp_1x = float(np.max(x) - np.min(x))
        phase_1x = float(np.angle(fft_vals[1])) if len(fft_vals) > 1 else 0.0
        phase_std = float(np.std(np.unwrap(np.angle(fft_vals))))

        stability = "Stable" if phase_std < 0.1 else "Variation"

        return {
            "Signal": signal.get("name", "Unknown"),
            "1X Amplitude": amp_1x,
            "Phase (deg)": np.degrees(phase_1x),
            "Stability": stability,
            "RPM": rpm,
            "timestamp": signal.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        }

    except:
        return None

# =========================
# LOAD SIGNALS
# =========================
signals = st.session_state.get("signals", [])

if not signals or len(signals) == 0:
    st.warning("No signals loaded")
    st.stop()

# =========================
# SIDEBAR
# =========================
signal_names = [s.get("name", f"Signal_{i}") for i, s in enumerate(signals)]

selected_names = st.sidebar.multiselect(
    "Select Signals",
    signal_names,
    default=signal_names[:3]
)

selected_signals = [
    s for s in signals if s.get("name", "") in selected_names
]

if not selected_signals:
    st.warning("Select at least one signal")
    st.stop()

# =========================
# METRICS
# =========================
metrics = []

for s in selected_signals:
    m = compute_phase_metrics(s)
    if m:
        metrics.append(m)

if not metrics:
    st.error("No valid signals for processing")
    st.stop()

df = pd.DataFrame(metrics)

# =========================
# HEADER (ULTRA ESTABLE)
# =========================
meta = {
    "machine": "Watermelon Machine",
    "point": "Phase Comparison",
    "config": "Phase Analysis | 1X",
    "rpm": float(df["RPM"].iloc[0]),
    "peak": float(df["1X Amplitude"].iloc[0]),
    "timestamp": df["timestamp"].iloc[0]
}

render_header(meta)

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
# EXPORT PNG (ROBUSTO)
# =========================
def export_table_png(df):
    fig = go.Figure()

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
        margin=dict(l=20, r=20, t=40, b=20)
    )

    return fig

if st.button("Export PNG"):
    try:
        fig = export_table_png(df_display)

        buf = BytesIO()
        fig.write_image(buf, format="png", scale=3)

        st.download_button(
            "Download PNG",
            buf.getvalue(),
            "phase_1x_dashboard.png",
            "image/png"
        )

    except Exception as e:
        st.error("Export failed. Install kaleido: pip install kaleido")