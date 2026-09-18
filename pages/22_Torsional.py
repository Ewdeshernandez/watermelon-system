"""
pages/22_Torsional.py — Watermelon Torsional (WEB)
==================================================

Web = SOLO análisis (la CONFIGURACIÓN y la captura se hacen en el software de
campo, `native/watermelon_torsional.py`, con la telemetría Binsfeld TorqueTrak
10K leída por una NI 9229/9215). Consume las corridas de par que el campo sube
o que se cargan como archivo; si no hay red / no hay corridas, usa un dataset
simulado para no quedar vacía. Espejo del módulo Modal (pages/18).

Navegación PERSISTENTE (segmented control, no st.tabs): Overview · Spectrum &
orders · Order tracking · Fatigue (rainflow).

Toda la matemática vive en el núcleo compartido `core.torsional.*` (el mismo que
consume la app de campo). UI del analista en inglés (política de idioma web).

Marco: par de eje (Vishay TN-512) · order tracking · fatiga rainflow ASTM E1049.
"""
from __future__ import annotations

import io

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from core.auth import (
    require_login, render_user_menu, get_current_user, is_page_allowed_for_role,
)
from core.ui_theme import page_header

from core.torsional.scaling import (
    ShaftGeometry, GageConfig, BridgeType, TorqueScaling, voltage_to_torque,
)
from core.torsional.sim_source import (
    TorsionalStreamConfig, SimulatedTorsionalSource, make_torsional_channels,
)
from core.torsional.analysis import (
    torque_metrics, torque_spectrum, order_amplitudes,
    keyphasor_to_rpm, order_tracking, fatigue_ranges,
)

st.set_page_config(page_title="Watermelon System | Torsional", page_icon="🍉", layout="wide")

require_login()
render_user_menu()
_user = get_current_user() or {}
_my_role = str(_user.get("role", "")).lower()
if not is_page_allowed_for_role("pages/22_Torsional.py", _my_role):
    st.error("Your role does not have access to this module.")
    st.stop()

NAVY = "#0F1E3D"; GREEN = "#16a34a"; BLUE = "#2563eb"; AMBER = "#f59e0b"; RED = "#dc2626"; SLATE = "#475569"


def _navplot(fig, **kw):
    """st.plotly_chart sin la barra de Plotly — look limpio (igual que pages/18)."""
    return st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, **kw)


def _style(fig, height=340):
    fig.update_layout(
        template="plotly_white", height=height, margin=dict(l=50, r=20, t=30, b=40),
        font=dict(color=NAVY), legend=dict(orientation="h", y=1.12, x=0),
    )
    return fig


def _dot(color, label):
    return f"<span style='color:{color};font-size:15px'>●</span> <b>{label}</b>"


# =====================================================================
# Datasets (demo simulado). Cacheados — el mismo núcleo que la app de campo.
# =====================================================================
@st.cache_data(show_spinner=False)
def _demo_steady(rpm=1800.0, mean=1200.0, a1=70.0, a2=30.0, res_hz=0.0, units="nm"):
    fs = 2560.0
    sc = TorqueScaling.from_geometry(ShaftGeometry(3.0), GageConfig(2.0, 4000), units=units)
    cfg = TorsionalStreamConfig(
        sample_rate_hz=fs, rpm=rpm, channels=make_torsional_channels(units=units),
        block_seconds=0.25, buffer_seconds=6.0, mean_torque=mean,
        orders=((1.0, a1, 0.0), (2.0, a2, 0.0)), torsional_res_hz=res_hz,
        torque_noise_rms_eu=5.0, scaling=sc, torque_units=units)
    src = SimulatedTorsionalSource(cfg); src.start()
    data = np.concatenate([src.read_block() for _ in range(24)], axis=1)  # 6 s
    kph_i = cfg.keyphasor_index(); ti = next(i for i in range(cfg.n_channels) if i != kph_i)
    torque = voltage_to_torque(data[ti], sc)
    return dict(torque=torque, kph=data[kph_i], fs=fs, rpm=rpm, units=units)


@st.cache_data(show_spinner=False)
def _demo_runup(r0=600.0, r1=3600.0, res_hz=30.0, units="nm"):
    fs = 2560.0
    sc = TorqueScaling.from_geometry(ShaftGeometry(3.0), GageConfig(2.0, 4000), units=units)
    ramp = 5.0
    cfg = TorsionalStreamConfig(
        sample_rate_hz=fs, channels=make_torsional_channels(units=units),
        block_seconds=0.25, buffer_seconds=ramp + 1.0, speed_profile="runup",
        rpm_start=r0, rpm_end=r1, ramp_seconds=ramp, mean_torque=500.0,
        orders=((1.0, 100.0, 0.0), (2.0, 45.0, 0.0)), torsional_res_hz=res_hz,
        torque_noise_rms_eu=4.0, scaling=sc, torque_units=units)
    src = SimulatedTorsionalSource(cfg); src.start()
    n = int(ramp / cfg.block_seconds)
    data = np.concatenate([src.read_block() for _ in range(n)], axis=1)
    kph_i = cfg.keyphasor_index(); ti = next(i for i in range(cfg.n_channels) if i != kph_i)
    torque = voltage_to_torque(data[ti], sc)
    return dict(torque=torque, kph=data[kph_i], fs=fs, res_hz=res_hz, units=units)


def _parse_upload(file, sc: TorqueScaling):
    """Lee un .npz de corrida: keys 'fs' y ('torque' en EU  |  'volts' + opcional 'kph')."""
    z = np.load(io.BytesIO(file.read()))
    fs = float(z["fs"]) if "fs" in z else 2560.0
    if "torque" in z:
        torque = np.asarray(z["torque"], float)
    elif "volts" in z:
        torque = voltage_to_torque(np.asarray(z["volts"], float), sc)
    else:
        raise ValueError("El .npz debe tener 'torque' (EU) o 'volts'.")
    kph = np.asarray(z["kph"], float) if "kph" in z else np.zeros_like(torque)
    return dict(torque=torque, kph=kph, fs=fs, rpm=None, units=sc.units)


# =====================================================================
# Encabezado + fuente de datos
# =====================================================================
page_header("Watermelon Torsional",
            "Shaft torque & torsional vibration — TorqueTrak 10K · NI 9229 · analysis only")

with st.expander("⚙  Data source & scaling", expanded=False):
    c1, c2 = st.columns([1, 1])
    with c1:
        source = st.radio("Source", ["Simulated demo", "Upload run (.npz)"], horizontal=True)
        units = st.radio("Units", ["N·m", "ft-lb"], horizontal=True)
        units_key = "nm" if units == "N·m" else "ftlb"
    with c2:
        st.caption("Scaling (used to convert uploaded volts → torque; Appendix B)")
        do = st.number_input("Outer Ø Do (in)", 0.1, 100.0, 3.0, 0.1)
        gf = st.number_input("Gage factor GF", 1.0, 3.0, 2.0, 0.01)
        gxmt = st.selectbox("Transmitter gain GXMT", [500, 1000, 2000, 4000, 8000, 16000], index=3)
    sc = TorqueScaling.from_geometry(ShaftGeometry(do), GageConfig(gf, int(gxmt)), units=units_key)
    st.caption(f"Full-scale torque (10 V): **{sc.full_scale_torque:,.1f} {units}** · "
               f"**{sc.eu_per_volt:,.2f} {units}/V**")

    run = None
    if source == "Upload run (.npz)":
        up = st.file_uploader("Run file (.npz with 'torque' or 'volts', 'kph', 'fs')", type=["npz"])
        if up is not None:
            try:
                run = _parse_upload(up, sc)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not read run: {exc}")

if run is None:
    run = _demo_steady(units=units_key)
    st.info("Showing a **simulated demo** run. Capture in the field app or upload a .npz to analyze real data.", icon="🔬")

torque = run["torque"]; kph = run["kph"]; fs = run["fs"]; u = "N·m" if run["units"] == "nm" else "ft-lb"

# rpm efectiva: del keyphasor si hay pulsos, si no del run
_, _rpm_series = keyphasor_to_rpm(kph, fs)
rpm = float(np.median(_rpm_series)) if _rpm_series.size else (run.get("rpm") or 1800.0)

# =====================================================================
# Navegación persistente
# =====================================================================
_NAV = ["Overview", "Spectrum & orders", "Order tracking", "Fatigue"]
if "tors_nav" not in st.session_state:
    st.session_state["tors_nav"] = _NAV[0]
if hasattr(st, "segmented_control"):
    nav = st.segmented_control("Section", _NAV, key="tors_nav",
                               label_visibility="collapsed") or st.session_state["tors_nav"]
else:
    nav = st.radio("Section", _NAV, horizontal=True, key="tors_nav")


# --------------------------------------------------------------- Overview
if nav == "Overview":
    m = torque_metrics(torque)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(f"Mean torque ({u})", f"{m.mean:,.1f}")
    k2.metric(f"Peak-peak ({u})", f"{m.peak_to_peak:,.1f}")
    ripple_txt = "∞" if m.ripple_pct == float("inf") else f"{m.ripple_pct:.1f}%"
    k3.metric("Ripple", ripple_txt)
    k4.metric("Speed", f"{rpm:,.0f} rpm")

    # Bolita de severidad del rizado (referencia práctica, ajustable por norma)
    sev_color, sev_txt = (GREEN, "Low") if m.ripple_pct < 10 else \
        ((AMBER, "Moderate") if m.ripple_pct < 25 else (RED, "High"))
    st.markdown("Torque ripple: " + _dot(sev_color, sev_txt), unsafe_allow_html=True)

    t = np.arange(torque.size) / fs
    fig = go.Figure(go.Scatter(x=t, y=torque, mode="lines", line=dict(color=BLUE, width=1.4)))
    fig.add_hline(y=m.mean, line=dict(color=SLATE, dash="dash"))
    fig.update_layout(xaxis_title="time (s)", yaxis_title=f"torque ({u})", title="Torque vs time")
    _navplot(_style(fig))

# ---------------------------------------------------- Spectrum & orders
elif nav == "Spectrum & orders":
    freqs, amp = torque_spectrum(torque, fs)
    mask = freqs <= 600.0  # techo de banda del equipo (500 Hz -3dB)
    fig = go.Figure(go.Scatter(x=freqs[mask], y=amp[mask], mode="lines", line=dict(color=NAVY)))
    # marcadores de órdenes
    f1 = rpm / 60.0
    for k in range(1, 6):
        if k * f1 <= 600:
            fig.add_vline(x=k * f1, line=dict(color=AMBER, dash="dot", width=1))
    fig.update_layout(xaxis_title="frequency (Hz)", yaxis_title=f"amplitude ({u})",
                      title=f"Torque spectrum (1× = {f1:.1f} Hz)")
    _navplot(_style(fig))

    oa = order_amplitudes(torque, fs, rpm, orders=(1, 2, 3, 4, 5))
    orders = list(range(1, 6))
    figo = go.Figure(go.Bar(x=[f"{o}×" for o in orders],
                            y=[oa[float(o)][0] for o in orders], marker_color=BLUE))
    figo.update_layout(yaxis_title=f"amplitude ({u})", title="Order amplitudes")
    _navplot(_style(figo, height=260))

    rows = [{"Order": f"{o}×", "Freq (Hz)": round(o * f1, 2),
             f"Amplitude ({u})": round(oa[float(o)][0], 2),
             "Phase (°)": round(oa[float(o)][1], 1)} for o in orders]
    st.dataframe(rows, use_container_width=True, hide_index=True)

# ------------------------------------------------------- Order tracking
elif nav == "Order tracking":
    st.caption("Run-up order tracking — amplitude of each order vs speed. Peaks reveal torsional "
               "resonances (an order crossing a natural frequency). Using a simulated run-up demo.")
    ru = _demo_runup(units=run["units"])
    rt = ru["torque"]; rkph = ru["kph"]; rfs = ru["fs"]
    t_rev, rpm_inst = keyphasor_to_rpm(rkph, rfs)
    tt = np.arange(rt.size) / rfs
    rpm_ps = np.interp(tt, t_rev, rpm_inst, left=rpm_inst[0], right=rpm_inst[-1])
    tracks = order_tracking(rt, rfs, rpm_ps, orders=(1, 2, 3), n_segments=30)
    fig = go.Figure()
    for tr, col in zip(tracks, (BLUE, GREEN, AMBER)):
        fig.add_trace(go.Scatter(x=tr.rpm, y=tr.amplitude, mode="lines",
                                 line=dict(color=col, width=2), name=f"{int(tr.order)}×"))
    if ru["res_hz"] > 0:
        fig.add_vline(x=ru["res_hz"] * 60.0, line=dict(color=RED, dash="dash"),
                      annotation_text=f"torsional natural {ru['res_hz']:.0f} Hz")
    fig.update_layout(xaxis_title="RPM", yaxis_title=f"order amplitude ({u})",
                      title="Order tracking — amplitude vs speed")
    _navplot(_style(fig, height=420))

# -------------------------------------------------------------- Fatigue
elif nav == "Fatigue":
    st.caption("Rainflow cycle counting (ASTM E1049) on the torque history — the input for "
               "fatigue life / Miner damage on the shaft.")
    ranges = fatigue_ranges(torque)
    if not ranges:
        st.info("Not enough reversals to count cycles.")
    else:
        rr = np.array([r for r, _ in ranges]); cc = np.array([c for _, c in ranges])
        fig = go.Figure(go.Bar(x=rr, y=cc, marker_color=NAVY))
        fig.update_layout(xaxis_title=f"torque range ({u})", yaxis_title="cycle count",
                          title="Rainflow range histogram")
        _navplot(_style(fig))
        total = float(cc.sum()); largest = float(rr.max())
        c1, c2 = st.columns(2)
        c1.metric("Total cycles", f"{total:,.1f}")
        c2.metric(f"Largest range ({u})", f"{largest:,.1f}")
