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
    except Exception:
        return None

def render_phase_header(meta):
    logo_uri = get_logo_data_uri()

    logo_html = ""
    if logo_uri:
        logo_html = f"""
        <img src="{logo_uri}" style="height:42px; width:auto; object-fit:contain;" />
        """

    st.markdown(
        f"""
        <div style="
            background:#ffffff;
            border:1px solid #e9eef5;
            border-radius:12px;
            padding:14px 18px;
            box-shadow:0 2px 10px rgba(0,0,0,0.06);
            display:flex;
            align-items:center;
            gap:14px;
            margin-bottom:14px;
        ">
            {logo_html}
            <div style="display:flex; flex-wrap:wrap; gap:10px; align-items:center; font-size:13px; color:#1f2937;">
                <span><b>{meta['machine']}</b></span>
                <span style="color:#9ca3af;">|</span>
                <span>{meta['point']}</span>
                <span style="color:#9ca3af;">|</span>
                <span>{meta['config']}</span>
                <span style="color:#9ca3af;">|</span>
                <span><b>Peak Amp:</b> {meta['peak']:.2f}</span>
                <span style="color:#9ca3af;">|</span>
                <span><b>RPM:</b> {meta['rpm']:.0f}</span>
                <span style="color:#9ca3af;">|</span>
                <span>{meta['timestamp']}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
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

if df.empty:
    st.warning("No valid phase data available")
    st.stop()

# =========================
# HEADER (HTML BLINDADO)
# =========================
meta = {
    "machine": "Watermelon Machine",
    "point": "Phase Comparison",
    "config": "Phase Analysis | 1X",
    "rpm": float(df["RPM"].iloc[0]),
    "peak": float(df["1X Amplitude"].iloc[0]),
    "timestamp": str(df["timestamp"].iloc[0])
}

render_phase_header(meta)

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
            align="center",
            font=dict(color="black", size=11),
            height=30
        )
    ))

    fig.update_layout(
        width=1400,
        height=600,
        margin=dict(l=20, r=20, t=20, b=20)
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
    except Exception:
        st.error("Export failed. Install kaleido with: pip install kaleido")