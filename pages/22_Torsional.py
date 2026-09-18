"""
pages/22_Torsional.py — Watermelon Torsional (WEB)
==================================================

Web = SOLO análisis (la CONFIGURACIÓN y la captura se hacen en el software de
campo, `native/watermelon_torsional.py`, con la telemetría Binsfeld TorqueTrak
10K leída por una NI 9229/9215). Consume las corridas de par que el campo sube
o que se cargan como archivo; si no hay red / no hay corridas, usa un dataset
simulado para no quedar vacía. Espejo del módulo Modal (pages/18) — MISMO
sistema visual: template de plots "watermelon", KPI cards, chips de estado,
tablas `wm-modes` y dark mode.

Navegación PERSISTENTE (segmented control, no st.tabs): Overview · Spectrum &
orders · Order tracking · Fatigue (rainflow).

Toda la matemática vive en el núcleo compartido `core.torsional.*`. UI del
analista en inglés (política de idioma web).

Marco: par de eje (Vishay TN-512) · order tracking · fatiga rainflow ASTM E1049.
"""
from __future__ import annotations

import io

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from core.auth import (
    require_login, render_user_menu, get_current_user, is_page_allowed_for_role,
)
from core.ui_theme import page_header

from core.torsional.scaling import (
    ShaftGeometry, GageConfig, TorqueScaling, voltage_to_torque,
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


# --- Tema de gráficos "watermelon" (idéntico al módulo Modal) ---
pio.templates["watermelon"] = go.layout.Template(
    layout=dict(
        font=dict(family="IBM Plex Sans, Segoe UI, Arial", size=13, color="#334155"),
        title=dict(font=dict(family="IBM Plex Sans", size=16, color=NAVY), x=0.01, xanchor="left", y=0.97),
        colorway=[BLUE, GREEN, RED, "#7c3aed", AMBER, "#0891b2", "#db2777", SLATE],
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#fbfcfe",
        hovermode="x unified", hoverlabel=dict(bgcolor="white", bordercolor="#e2e8f0",
                                               font=dict(family="IBM Plex Mono", size=12)),
        xaxis=dict(gridcolor="#eef2f8", zerolinecolor="#dbe4f0", linecolor="#cbd5e1",
                   ticks="outside", tickcolor="#cbd5e1", ticklen=4,
                   title=dict(font=dict(size=12, color="#64748b"))),
        yaxis=dict(gridcolor="#eef2f8", zerolinecolor="#dbe4f0", linecolor="#cbd5e1",
                   ticks="outside", tickcolor="#cbd5e1", ticklen=4,
                   title=dict(font=dict(size=12, color="#64748b"))),
        margin=dict(l=62, r=24, t=48, b=52),
        legend=dict(bgcolor="rgba(255,255,255,.85)", bordercolor="#e2e8f0", borderwidth=1,
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    ))


def _inject_theme():
    st.markdown("""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600&display=swap');
      html, body, [class*="css"] { font-family:'IBM Plex Sans',system-ui,sans-serif; }
      .block-container { padding-top: 0.8rem; max-width: 1200px; }
      /* KPI cards */
      .wm-kpis { display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin:10px 0 4px; }
      .wm-kpi { background:#fff; border:1px solid #e6ecf5; border-radius:11px; padding:10px 14px;
        box-shadow:0 1px 2px rgba(15,30,61,.04),0 4px 12px rgba(15,30,61,.04); }
      .wm-kpi .v { font-family:'IBM Plex Mono',monospace; font-size:21px; font-weight:600; color:#0F1E3D; line-height:1.1; }
      .wm-kpi .l { font-size:11px; color:#64748b; text-transform:uppercase; letter-spacing:.05em; margin-top:3px; }
      .wm-kpi .s { font-size:11px; color:#94a3b8; margin-top:1px; }
      /* Chips de estado */
      .wm-chip { display:inline-block; font-size:11px; font-weight:700; letter-spacing:.03em;
        text-transform:uppercase; padding:5px 12px; border-radius:999px; }
      .wm-go { background:#16a34a; color:#fff; } .wm-rev { background:#f59e0b; color:#0f1e3d; }
      .wm-nogo { background:#dc2626; color:#fff; }
      /* Hero de contexto de la corrida */
      .wm-hero { background:linear-gradient(110deg,#0F1E3D 0%,#12325a 55%,#16a34a 165%);
        border-radius:12px; padding:13px 20px; color:#fff; box-shadow:0 6px 18px rgba(15,30,61,.16);
        display:flex; justify-content:space-between; align-items:center; gap:14px; flex-wrap:wrap; margin:6px 0 2px; }
      .wm-hero h1 { font-size:18px; font-weight:700; margin:0 0 3px; letter-spacing:-.01em; }
      .wm-hero .meta { color:#cbd5e1; font-size:12px; }
      /* Badges de tabla */
      table.wm-modes .pill { padding:3px 11px; border-radius:999px; font-size:11px; font-weight:700; white-space:nowrap; }
      /* Tabla bonita (mismo estilo que el módulo Modal) */
      table.wm-modes { width:100%; border-collapse:separate; border-spacing:0;
        font-family:'IBM Plex Sans',sans-serif; border:1px solid #e6ecf5; border-radius:14px;
        overflow:hidden; box-shadow:0 6px 18px rgba(15,30,61,.06); margin:6px 0 10px; }
      table.wm-modes th { background:#0F1E3D; color:#fff; font-size:11px; font-weight:600;
        letter-spacing:.04em; text-transform:uppercase; text-align:center; padding:11px 10px; }
      table.wm-modes td { padding:10px 10px; text-align:center; border-top:1px solid #eef2f8;
        font-size:14px; color:#0f1e3d; }
      table.wm-modes tr:nth-child(even) td { background:#f7fafd; }
      table.wm-modes tr:hover td { background:#eef6ff; }
      table.wm-modes td.idx { color:#94a3b8; font-family:'IBM Plex Mono',monospace; width:44px; }
      table.wm-modes td.num { font-family:'IBM Plex Mono',monospace; font-weight:600; }
      table.wm-modes .u { color:#94a3b8; font-size:11px; font-weight:400; }
      /* Section caption */
      .wm-sec { font-weight:700; color:#0F1E3D; font-size:13px; margin:14px 0 2px; letter-spacing:.01em; }
      .wm-sec span { font-weight:400; color:#94a3b8; font-size:11px; }
      @media (prefers-color-scheme: dark){
        .wm-kpi{ background:#141b26; border-color:#243040; }
        .wm-kpi .v{ color:#eaf0f7; } .wm-kpi .l{ color:#8ea0bd; } .wm-kpi .s{ color:#6b7d99; }
        .wm-sec{ color:#eaf0f7; }
        table.wm-modes{ background:#0f1622; box-shadow:0 1px 3px rgba(0,0,0,.4); }
        table.wm-modes th{ background:#0b1220; }
        table.wm-modes td{ color:#dbe4f0; border-top-color:#1e2836; }
        table.wm-modes tr:nth-child(even) td{ background:#141d2b; }
        table.wm-modes tr:hover td{ background:#1b2740; }
        table.wm-modes td.idx{ color:#5f7290; } table.wm-modes .u{ color:#6b7d99; }
      }
    </style>
    """, unsafe_allow_html=True)


def _navplot(fig, **kw):
    return st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, **kw)


def _apply(fig, title="", height=360, ylab="", xlab=""):
    fig.update_layout(template="watermelon", height=height, title=title)
    if xlab:
        fig.update_xaxes(title_text=xlab)
    if ylab:
        fig.update_yaxes(title_text=ylab)
    return fig


def _kpis(items):
    """items = [(value_html, label, sub_or_None), …] → grid de KPI cards."""
    cells = "".join(
        f'<div class="wm-kpi"><div class="v">{v}</div><div class="l">{l}</div>'
        + (f'<div class="s">{s}</div>' if s else "") + "</div>"
        for v, l, s in items)
    st.markdown(f'<div class="wm-kpis">{cells}</div>', unsafe_allow_html=True)


def _sec(title, hint=""):
    st.markdown(f'<div class="wm-sec">{title} <span>{hint}</span></div>', unsafe_allow_html=True)


def _pill(text, color, bg):
    return f"<span class='pill' style='color:{color};background:{bg}'>{text}</span>"


# Paleta de pills (igual que el módulo Modal)
PILL_GREEN = ("#16a34a", "#eaf7ef"); PILL_AMBER = ("#b45309", "#fef3e2")
PILL_RED = ("#dc2626", "#fdeaea"); PILL_SLATE = ("#64748b", "#eef2f8")


# Nav con bolitas de color (emojis de círculo, como el Modal)
T_OVR = "🟢  Overview"
T_SPEC = "🟡  Spectrum & orders"
T_ORD = "🔵  Order tracking"
T_FAT = "🔴  Fatigue"


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
    data = np.concatenate([src.read_block() for _ in range(24)], axis=1)
    kph_i = cfg.keyphasor_index(); ti = next(i for i in range(cfg.n_channels) if i != kph_i)
    return dict(torque=voltage_to_torque(data[ti], sc), kph=data[kph_i], fs=fs, rpm=rpm,
                units=units, name="Simulated demo — Motor-Pump shaft", field=False)


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
    return dict(torque=voltage_to_torque(data[ti], sc), kph=data[kph_i], fs=fs, res_hz=res_hz, units=units)


def _parse_upload(file, sc: TorqueScaling):
    z = np.load(io.BytesIO(file.read()))
    fs = float(z["fs"]) if "fs" in z else 2560.0
    if "torque" in z:
        torque = np.asarray(z["torque"], float)
    elif "volts" in z:
        torque = voltage_to_torque(np.asarray(z["volts"], float), sc)
    else:
        raise ValueError("El .npz debe tener 'torque' (EU) o 'volts'.")
    kph = np.asarray(z["kph"], float) if "kph" in z else np.zeros_like(torque)
    return dict(torque=torque, kph=kph, fs=fs, rpm=None, units=sc.units,
                name=getattr(file, "name", "Uploaded run"), field=True)


# =====================================================================
# Encabezado + tema
# =====================================================================
page_header("Watermelon Torsional",
            "Shaft torque & torsional vibration — TorqueTrak 10K · NI 9229 · analysis only")
_inject_theme()

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

torque = run["torque"]; kph = run["kph"]; fs = run["fs"]; u = "N·m" if run["units"] == "nm" else "ft-lb"
_, _rpm_series = keyphasor_to_rpm(kph, fs)
rpm = float(np.median(_rpm_series)) if _rpm_series.size else (run.get("rpm") or 1800.0)

# =====================================================================
# Hero de contexto de la corrida + KPIs persistentes (como el Modal)
# =====================================================================
m = torque_metrics(torque)
if m.ripple_pct < 10:
    _chip, _cls = "TORQUE STABLE", "wm-go"
elif m.ripple_pct < 25:
    _chip, _cls = "MODERATE RIPPLE", "wm-rev"
else:
    _chip, _cls = "HIGH RIPPLE", "wm-nogo"
_src = "☁ Field run" if run.get("field") else "⚪ Simulated dataset"
ripple_txt = "∞" if m.ripple_pct == float("inf") else f"{m.ripple_pct:.1f}%"
st.markdown(f"""
<div class="wm-hero">
  <div>
    <h1>{run.get('name', 'Torsional run')}</h1>
    <div class="meta">TorqueTrak 10K · NI 9229 · {u} · fs {fs:,.0f} Hz · {torque.size/fs:.1f} s</div>
  </div>
  <div style="text-align:right">
    <span class="wm-chip {_cls}">● {_chip}</span>
    <div class="meta" style="margin-top:8px">{_src}</div>
  </div>
</div>
<div class="wm-kpis">
  <div class="wm-kpi"><div class="v">{m.mean:,.1f}<span style="font-size:13px"> {u}</span></div>
    <div class="l">Mean torque</div><div class="s">static</div></div>
  <div class="wm-kpi"><div class="v">{m.peak_to_peak:,.1f}<span style="font-size:13px"> {u}</span></div>
    <div class="l">Peak-peak</div><div class="s">dynamic</div></div>
  <div class="wm-kpi"><div class="v">{ripple_txt}</div>
    <div class="l">Ripple</div><div class="s">pp / mean</div></div>
  <div class="wm-kpi"><div class="v">{rpm:,.0f}<span style="font-size:13px"> rpm</span></div>
    <div class="l">Running speed</div><div class="s">1× = {rpm/60:.1f} Hz</div></div>
</div>
""", unsafe_allow_html=True)

# =====================================================================
# Navegación persistente — bolitas de color (segmented control)
# =====================================================================
_NAV = [T_OVR, T_SPEC, T_ORD, T_FAT]
if "tors_nav" not in st.session_state:
    st.session_state["tors_nav"] = T_OVR
if hasattr(st, "segmented_control"):
    nav = st.segmented_control("Section", _NAV, key="tors_nav",
                               label_visibility="collapsed") or st.session_state["tors_nav"]
else:
    nav = st.radio("Section", _NAV, horizontal=True, key="tors_nav")


# --------------------------------------------------------------- Overview
if nav == T_OVR:
    _sec("Torque waveform", "engineering units vs time")
    t = np.arange(torque.size) / fs
    fig = go.Figure(go.Scatter(x=t, y=torque, mode="lines", line=dict(color=BLUE, width=1.3),
                               name="torque"))
    fig.add_hline(y=m.mean, line=dict(color=SLATE, dash="dash"),
                  annotation_text=f"mean {m.mean:,.0f} {u}", annotation_position="top left")
    _navplot(_apply(fig, height=400, xlab="time (s)", ylab=f"torque ({u})"))
    st.caption(f"RMS **{m.rms:,.1f} {u}** · crest factor **{m.crest_factor:.2f}** · "
               f"dynamic peak **{m.peak_to_peak/2:,.1f} {u}**")

# ---------------------------------------------------- Spectrum & orders
elif nav == T_SPEC:
    freqs, amp = torque_spectrum(torque, fs)
    mask = freqs <= 600.0
    f1 = rpm / 60.0
    _sec("Torque spectrum", f"1× = {f1:.1f} Hz · equipment bandwidth 500 Hz")
    fig = go.Figure(go.Scatter(x=freqs[mask], y=amp[mask], mode="lines",
                               line=dict(color=NAVY, width=1.6), fill="tozeroy",
                               fillcolor="rgba(15,30,61,0.05)", name="amplitude"))
    for k in range(1, 6):
        if k * f1 <= 600:
            fig.add_vline(x=k * f1, line=dict(color=AMBER, dash="dot", width=1))
            fig.add_annotation(x=k * f1, y=1, yref="paper", text=f"{k}×", showarrow=False,
                               font=dict(size=10, color=AMBER), yshift=-2)
    _navplot(_apply(fig, height=340, xlab="frequency (Hz)", ylab=f"amplitude ({u})"))

    oa = order_amplitudes(torque, fs, rpm, orders=(1, 2, 3, 4, 5))
    orders = list(range(1, 6))
    colors = [BLUE, GREEN, AMBER, "#7c3aed", SLATE]
    _sec("Order amplitudes")
    figo = go.Figure(go.Bar(x=[f"{o}×" for o in orders], y=[oa[float(o)][0] for o in orders],
                            marker_color=colors, text=[f"{oa[float(o)][0]:.1f}" for o in orders],
                            textposition="outside", cliponaxis=False))
    figo.update_layout(hovermode=False)
    _navplot(_apply(figo, height=260, ylab=f"amplitude ({u})"))

    amax = max(oa[float(o)][0] for o in orders) or 1.0

    def _level(a):
        if a >= 0.5 * amax:
            return _pill("dominant", *PILL_GREEN)
        if a >= 0.1 * amax:
            return _pill("present", *PILL_AMBER)
        return _pill("trace", *PILL_SLATE)
    rows = "".join(
        f'<tr><td class="idx">{o}×</td><td class="num">{o*f1:.2f}<span class="u"> Hz</span></td>'
        f'<td class="num">{oa[float(o)][0]:.2f}<span class="u"> {u}</span></td>'
        f'<td class="num">{oa[float(o)][1]:+.1f}<span class="u"> °</span></td>'
        f'<td>{_level(oa[float(o)][0])}</td></tr>' for o in orders)
    st.markdown(
        '<table class="wm-modes"><thead><tr><th>Order</th><th>Frequency</th>'
        f'<th>Amplitude</th><th>Phase</th><th>Level</th></tr></thead><tbody>{rows}</tbody></table>',
        unsafe_allow_html=True)

# ------------------------------------------------------- Order tracking
elif nav == T_ORD:
    _sec("Run-up order tracking", "amplitude of each order vs speed — peaks reveal torsional resonances")
    ru = _demo_runup(units=run["units"])
    rt = ru["torque"]; rkph = ru["kph"]; rfs = ru["fs"]
    t_rev, rpm_inst = keyphasor_to_rpm(rkph, rfs)
    tt = np.arange(rt.size) / rfs
    rpm_ps = np.interp(tt, t_rev, rpm_inst, left=rpm_inst[0], right=rpm_inst[-1])
    tracks = order_tracking(rt, rfs, rpm_ps, orders=(1, 2, 3), n_segments=30)
    fig = go.Figure()
    for tr, col in zip(tracks, (BLUE, GREEN, AMBER)):
        fig.add_trace(go.Scatter(x=tr.rpm, y=tr.amplitude, mode="lines+markers",
                                 line=dict(color=col, width=2.2), marker=dict(size=4),
                                 name=f"{int(tr.order)}×"))
    if ru["res_hz"] > 0:
        fig.add_vline(x=ru["res_hz"] * 60.0, line=dict(color=RED, dash="dash"),
                      annotation_text=f"torsional natural {ru['res_hz']:.0f} Hz",
                      annotation_position="top", annotation_font=dict(color=RED, size=11))
    _navplot(_apply(fig, height=440, xlab="RPM", ylab=f"order amplitude ({u})"))
    st.caption("Simulated run-up demo. Each order peaks where k × running speed crosses the "
               "torsional natural frequency (30 Hz here → 1× at 1800 rpm, 2× at 900 rpm).")

# -------------------------------------------------------------- Fatigue
elif nav == T_FAT:
    _sec("Rainflow cycle counting", "ASTM E1049 — input for fatigue life / Miner damage on the shaft")
    ranges = fatigue_ranges(torque)
    if not ranges:
        st.info("Not enough reversals to count cycles.")
    else:
        rr = np.array([r for r, _ in ranges]); cc = np.array([c for _, c in ranges])
        rmax = float(rr.max())
        # Ciclos de rango grande (>50% del máx) — los que dominan el daño de fatiga.
        damaging = float(cc[rr > 0.5 * rmax].sum())
        _kpis([
            (f"{cc.sum():,.0f}", "Total cycles", "counted"),
            (f"{rmax:,.1f}<span style='font-size:13px'> {u}</span>", "Largest range", "worst cycle"),
            (f"{np.average(rr, weights=cc):,.1f}<span style='font-size:13px'> {u}</span>",
             "Mean range", "cycle-weighted"),
            (f"{damaging:,.0f}", "High-range cycles", "> 50% of max"),
        ])
        # Histograma BINEADO (con ruido cada rango es único → sin binear son miles
        # de barras ilegibles). Suma de conteos por bucket de rango de par.
        nb = int(np.clip(len(rr), 8, 24))
        edges = np.linspace(0.0, rmax * 1.0001, nb + 1)
        hist, _ = np.histogram(rr, bins=edges, weights=cc)
        centers = 0.5 * (edges[:-1] + edges[1:])
        fig = go.Figure(go.Bar(x=centers, y=hist, width=(edges[1] - edges[0]) * 0.92,
                               marker_color=NAVY, marker_line=dict(color="#0b1220", width=0.5)))
        fig.update_layout(hovermode="closest", bargap=0.05)
        _navplot(_apply(fig, height=360, xlab=f"torque range ({u})", ylab="cycle count"))
