# pages/13_Phase_Analysis.py

import os
import base64
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st


PRIMARY = "#1ea7ff"


def safe_signals():
    signals = st.session_state.get("signals", [])
    if isinstance(signals, list):
        return [s for s in signals if isinstance(s, dict)]
    return []


def to_array(value):
    if value is None:
        return np.array([], dtype=float)

    if isinstance(value, np.ndarray):
        try:
            return value.astype(float, copy=False).flatten()
        except Exception:
            return np.array([], dtype=float)

    if isinstance(value, (list, tuple, pd.Series)):
        try:
            return np.asarray(value, dtype=float).flatten()
        except Exception:
            return np.array([], dtype=float)

    try:
        return np.array([float(value)], dtype=float)
    except Exception:
        return np.array([], dtype=float)


def safe_float(value, default=np.nan):
    try:
        if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
            arr = to_array(value)
            if arr.size == 0:
                return default
            finite = arr[np.isfinite(arr)]
            if finite.size == 0:
                return default
            return float(np.mean(finite))
        return float(value)
    except Exception:
        return default


def fmt_num(value, decimals=2, suffix=""):
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "—"
        return f"{float(value):,.{decimals}f}{suffix}"
    except Exception:
        return "—"


def fmt_text(value, default="—"):
    if value is None:
        return default
    txt = str(value).strip()
    return txt if txt else default


def get_logo_html():
    candidates = [
        "assets/watermelon_logo.png",
        "assets/watermelon.png",
        "assets/logo.png",
        "logo.png",
    ]

    for path in candidates:
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    data = base64.b64encode(f.read()).decode("utf-8")
                return f'<img src="data:image/png;base64,{data}" style="height:40px;" />'
            except Exception:
                pass

    return f"""
    <div style="
        width:40px;
        height:40px;
        border-radius:10px;
        background:{PRIMARY};
        color:white;
        display:flex;
        align-items:center;
        justify-content:center;
        font-weight:700;
        font-size:14px;
    ">WM</div>
    """


def extract_rpm(signal):
    for key in ["rpm", "RPM", "speed", "shaft_rpm"]:
        if key in signal:
            rpm = safe_float(signal.get(key))
            if not np.isnan(rpm):
                return rpm
    return np.nan


def extract_timestamp(signal):
    for key in ["timestamp", "Timestamp", "datetime", "date"]:
        if key in signal:
            return fmt_text(signal.get(key))
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def compute_phase_metrics(signal):
    name = fmt_text(signal.get("name"), "Unnamed Signal")
    y = to_array(signal.get("y"))
    rpm = extract_rpm(signal)

    if y.size >= 8:
        finite = y[np.isfinite(y)]
    else:
        finite = np.array([], dtype=float)

    if finite.size >= 8:
        centered = finite - np.mean(finite)
        fft_vals = np.fft.rfft(centered)
        if len(fft_vals) > 1:
            phase_deg = float(np.degrees(np.angle(fft_vals[1])) % 360.0)
        else:
            phase_deg = np.nan

        amp_1x = float((np.max(finite) - np.min(finite)) / 2.0)

        rms = float(np.sqrt(np.mean(np.square(finite)))) if finite.size else 0.0
        crest = float(np.max(np.abs(finite)) / rms) if rms > 0 else np.nan

        if np.isnan(crest):
            stability = "Watch"
        elif crest < 3.0:
            stability = "Stable"
        elif crest < 5.0:
            stability = "Watch"
        else:
            stability = "Unstable"
    else:
        phase_deg = np.nan
        amp_1x = np.nan
        stability = "Watch"

    return {
        "Signal": name,
        "1X Amplitude": amp_1x,
        "Phase": phase_deg,
        "Stability": stability,
        "RPM": rpm,
        "timestamp": extract_timestamp(signal),
    }


st.title("Phase Analysis")

signals = safe_signals()

if not signals:
    st.info("No signals loaded")
    st.stop()

signal_names = [fmt_text(s.get("name"), f"Signal {i+1}") for i, s in enumerate(signals)]

selected_names = st.sidebar.multiselect(
    "Select Signals",
    signal_names,
    default=signal_names[:min(3, len(signal_names))]
)

selected_signals = [s for s in signals if fmt_text(s.get("name")) in selected_names]

if not selected_signals:
    st.warning("Select at least one signal")
    st.stop()

metrics = [compute_phase_metrics(s) for s in selected_signals]
df = pd.DataFrame(metrics)

if df.empty:
    st.warning("No valid phase data available")
    st.stop()

first_signal = selected_signals[0]
header_name = fmt_text(first_signal.get("name"), "Unnamed Signal")
header_rpm = fmt_num(df["RPM"].iloc[0], 0)
header_peak = fmt_num(df["1X Amplitude"].iloc[0], 3)
header_ts = fmt_text(df["timestamp"].iloc[0])

st.markdown(
    f"""
    <div style="
        background:white;
        border:1px solid #e9eef5;
        border-radius:12px;
        padding:14px 18px;
        margin-bottom:14px;
    ">
        <div style="display:flex; align-items:center; gap:12px; flex-wrap:wrap;">
            {get_logo_html()}
            <div style="font-size:13px; color:#334155;">
                <b style="color:#0f172a;">1X Phase Analysis</b>
                &nbsp;&nbsp;|&nbsp;&nbsp;
                <b>Signal:</b> {header_name}
                &nbsp;&nbsp;|&nbsp;&nbsp;
                <b>RPM:</b> {header_rpm}
                &nbsp;&nbsp;|&nbsp;&nbsp;
                <b>Peak:</b> {header_peak}
                &nbsp;&nbsp;|&nbsp;&nbsp;
                <b>Timestamp:</b> {header_ts}
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

df_display = df[["Signal", "1X Amplitude", "Phase", "Stability", "RPM"]].copy()
df_display["1X Amplitude"] = df_display["1X Amplitude"].apply(lambda v: fmt_num(v, 3))
df_display["Phase"] = df_display["Phase"].apply(lambda v: fmt_num(v, 1, "°"))
df_display["RPM"] = df_display["RPM"].apply(lambda v: fmt_num(v, 0))

st.dataframe(
    df_display,
    use_container_width=True,
    hide_index=True
)

csv_bytes = df_display.to_csv(index=False).encode("utf-8")

st.download_button(
    "Export CSV",
    data=csv_bytes,
    file_name="phase_analysis.csv",
    mime="text/csv"
)