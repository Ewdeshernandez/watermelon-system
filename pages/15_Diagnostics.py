import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from core.auth import require_login, render_user_menu


# =========================================================
# CONFIG
# =========================================================
st.set_page_config(page_title="Watermelon System | Diagnostic", layout="wide")
require_login()
render_user_menu()


# =========================================================
# HEADER
# =========================================================
st.markdown("""
<div style="
    background: linear-gradient(135deg, #061326 0%, #0d2a4a 42%, #163f78 100%);
    padding: 24px 28px;
    border-radius: 20px;
    margin-bottom: 18px;
">
    <h1 style="color:white;margin:0;">Diagnostic</h1>
    <p style="color:rgba(255,255,255,0.8);margin-top:8px;">
    Real engineering diagnostic layer. Correlates signals, orders and rules to generate actionable conclusions.
    </p>
</div>
""", unsafe_allow_html=True)


# =========================================================
# LOAD SIGNALS
# =========================================================
signals = st.session_state.get("signals", {})

if not signals:
    st.warning("No signals loaded")
    st.stop()


# =========================================================
# CORE FUNCTIONS
# =========================================================
def get_array(sig, key):
    try:
        arr = np.asarray(sig.get(key, []), dtype=float)
        return arr.flatten()
    except:
        return np.array([])


def estimate_fs(t):
    if len(t) < 2:
        return 0
    dt = np.diff(t)
    dt = dt[dt > 0]
    if len(dt) == 0:
        return 0
    return 1 / np.median(dt)


def fft_signal(x, fs):
    if len(x) < 64:
        return [], []

    x = x - np.mean(x)
    n = len(x)
    yf = np.fft.rfft(x * np.hanning(n))
    xf = np.fft.rfftfreq(n, 1/fs)

    amp = (2.0/n) * np.abs(yf)
    return xf, amp


def get_harmonic(freqs, amps, target):
    if target <= 0:
        return 0
    mask = (freqs > target*0.95) & (freqs < target*1.05)
    if np.sum(mask) == 0:
        return 0
    return float(np.max(amps[mask]))


def classify_fault(overall, a1, a2):
    if a1 > a2 * 1.3:
        return "Unbalance"
    if a2 > a1 * 0.8:
        return "Misalignment"
    if overall > 0 and a1 < overall * 0.2:
        return "Broadband / Looseness"
    return "Normal behavior"


def severity(overall):
    if overall > 5:
        return "Danger"
    if overall > 3:
        return "Alert"
    if overall > 2:
        return "Observe"
    return "Normal"


# =========================================================
# PROCESS SIGNALS
# =========================================================
results = []

for key, sig in signals.items():

    t = get_array(sig, "time")
    x = get_array(sig, "x")
    md = sig.get("metadata", {})

    if len(t) < 64:
        continue

    fs = estimate_fs(t)
    freqs, amps = fft_signal(x, fs)

    rpm = float(md.get("RPM", 0))
    if rpm == 0 and len(freqs) > 0:
        rpm = freqs[np.argmax(amps)] * 60

    run = rpm / 60

    a05 = get_harmonic(freqs, amps, run * 0.5)
    a1 = get_harmonic(freqs, amps, run * 1)
    a2 = get_harmonic(freqs, amps, run * 2)

    overall = np.max(x) - np.min(x)

    fault = classify_fault(overall, a1, a2)
    status = severity(overall)

    confidence = min(100, max(30, (a1 + a2) * 20))

    results.append({
        "Machine": md.get("Machine", "Unknown"),
        "Point": md.get("Point", key),
        "RPM": round(rpm, 1),
        "Overall": round(overall, 3),
        "0.5X": round(a05, 3),
        "1X": round(a1, 3),
        "2X": round(a2, 3),
        "Fault": fault,
        "Status": status,
        "Confidence": round(confidence, 1)
    })


df = pd.DataFrame(results)

if df.empty:
    st.warning("No valid signals")
    st.stop()


# =========================================================
# KPI
# =========================================================
col1, col2, col3 = st.columns(3)

col1.metric("Signals", len(df))
col2.metric("Critical", len(df[df["Status"].isin(["Alert", "Danger"])]))
col3.metric("Avg Confidence", f"{df['Confidence'].mean():.1f}%")


# =========================================================
# VERDICT BOARD (🔥 CORE)
# =========================================================
st.markdown("## Diagnostic Verdict")

for _, r in df.iterrows():

    color = {
        "Normal": "#18a957",
        "Observe": "#d7a600",
        "Alert": "#e26f00",
        "Danger": "#d92d20"
    }.get(r["Status"], "#185ea9")

    st.markdown(f"""
    <div style="
        border-left:6px solid {color};
        padding:14px;
        margin-bottom:10px;
        background:white;
        border-radius:10px;
    ">
        <b>{r["Machine"]} - {r["Point"]}</b><br>
        Status: <b>{r["Status"]}</b> | Fault: <b>{r["Fault"]}</b><br>
        RPM: {r["RPM"]} | Overall: {r["Overall"]}<br>
        1X: {r["1X"]} | 2X: {r["2X"]}<br>
        Confidence: {r["Confidence"]}%
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# EVIDENCE MATRIX (COMPACTA)
# =========================================================
st.markdown("## Evidence Matrix")
st.dataframe(df, use_container_width=True)


# =========================================================
# MAP
# =========================================================
fig = go.Figure()

severity_map = {"Normal":0,"Observe":1,"Alert":2,"Danger":3}

fig.add_trace(go.Scatter(
    x=df["RPM"],
    y=df["Overall"],
    mode="markers+text",
    text=df["Point"],
    marker=dict(
        size=df["Confidence"],
        color=df["Status"].map(severity_map),
        colorscale="RdYlGn_r"
    )
))

fig.update_layout(height=400, template="plotly_white")

st.plotly_chart(fig, use_container_width=True)