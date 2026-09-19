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
    rainflow_cycles, shaft_torsional_fatigue, RainflowCycle,
)
from core.modal.campbell import compute_crossings, SpeedBand, separation_margin_pct

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
T_CAMP = "🟤  Campbell"
T_FAT = "🔴  Fatigue"
T_REPORT = "⚪  Report"


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


def _is_runup(kph_arr, fs_v):
    """¿La corrida es un run-up/coast-down (barrido real de rpm)? Solo entonces
    se pueden MEDIR las naturales torsionales — una captura estacionaria no las
    revela. Devuelve (es_runup, rpm_min, rpm_max)."""
    _, rpm_inst = keyphasor_to_rpm(np.asarray(kph_arr, float), fs_v)
    if rpm_inst.size < 5:
        return False, 0.0, 0.0
    lo, hi = float(np.min(rpm_inst)), float(np.max(rpm_inst))
    mid = 0.5 * (lo + hi)
    return (mid > 0 and (hi - lo) / mid > 0.20), lo, hi


@st.cache_data(show_spinner=False)
def _detect_naturals(torque_arr, kph_arr, fs_v):
    """Order tracking + detección de naturales sobre CUALQUIER traza (real o demo).
    Naturales por ESPECTRO DE RESONANCIA SUMADO POR ÓRDENES: cada orden se mapea a
    fn = orden·rpm/60 y se acumula en una grilla común. La natural real la cruzan
    TODAS las órdenes (a distinta rpm) → se refuerza; bumps de una sola orden no se
    alinean → se suprimen. Robusto vs argmax."""
    rt = np.asarray(torque_arr, float); rkph = np.asarray(kph_arr, float)
    t_rev, rpm_inst = keyphasor_to_rpm(rkph, fs_v)
    if rpm_inst.size < 3:
        return dict(tracks=[], naturals=[], rpm_min=0.0, rpm_max=0.0)
    tt = np.arange(rt.size) / fs_v
    rpm_ps = np.interp(tt, t_rev, rpm_inst, left=rpm_inst[0], right=rpm_inst[-1])
    tracks = order_tracking(rt, fs_v, rpm_ps, orders=(1, 2, 3), n_segments=30)
    fgrid = np.linspace(2.0, 200.0, 600)
    acc = np.zeros_like(fgrid)
    for tr in tracks:
        fn_arr = tr.order * tr.rpm / 60.0
        srt = np.argsort(fn_arr)
        acc += np.interp(fgrid, fn_arr[srt], tr.amplitude[srt], left=0.0, right=0.0)
    naturals = []
    if acc.max() > 0:
        thr = 0.35 * acc.max()
        for i in range(1, len(acc) - 1):
            if acc[i] > thr and acc[i] >= acc[i - 1] and acc[i] > acc[i + 1]:
                f = float(fgrid[i])
                if not any(abs(f - x) < 3.0 for x in naturals):
                    naturals.append(f)
    return dict(tracks=[(float(t.order), t.rpm, t.amplitude) for t in tracks],
                naturals=sorted(naturals), rpm_min=float(rpm_ps.min()),
                rpm_max=float(rpm_ps.max()))


def _runup_source(torque_arr, kph_arr, fs_v, is_field, units="nm"):
    """Fuente del análisis de resonancia, con HONESTIDAD de datos:
    - Si la corrida subida es un run-up real → analiza la DATA REAL (measured=True).
    - Si es estacionaria o simulada → usa el demo simulado (measured=False) para
      ILUSTRAR el método; las naturales NO son medidas de esta corrida."""
    sweeping, _lo, _hi = _is_runup(kph_arr, fs_v)
    if is_field and sweeping:
        d = _detect_naturals(torque_arr, kph_arr, fs_v)
        d["measured"] = True; d["res_hz"] = 0.0
        return d
    demo = _demo_runup(units=units)
    d = _detect_naturals(demo["torque"], demo["kph"], demo["fs"])
    d["measured"] = False; d["res_hz"] = demo["res_hz"]
    return d


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


@st.cache_data(ttl=60, show_spinner=False)
def _cloud_run_list():
    """Lista las corridas de campo subidas a la nube (cacheada 60 s)."""
    try:
        from core.torsional import cloud
        return cloud.list_runs()
    except Exception:  # noqa: BLE001
        return []


@st.cache_data(show_spinner=False)
def _load_cloud_run(run_id):
    """Baja metadata + cruda de una corrida de nube y arma el dict de corrida.
    La cruda trae [Torque_V, KPH]; el par se reconstruye con eu_per_volt del campo."""
    from core.torsional import cloud
    meta = cloud.load_run(run_id)
    if not meta:
        return None
    if meta.get("kind") == "monitor":          # corrida de MONITOREO 24h: histograma, no onda
        setup = meta.get("setup", {}) if isinstance(meta.get("setup"), dict) else {}
        name = setup.get("machine") or setup.get("tag") or "Torsional monitor"
        return dict(kind="monitor", payload=meta, units=meta.get("units", "nm"),
                    name=name, field=True, setup=setup)
    dl = cloud.download_raw(meta.get("raw_ref")) if meta.get("raw_ref") else None
    if dl is None:
        return None
    data, fs_dl = dl
    data = np.asarray(data, float)
    if data.ndim == 1:
        data = data[:, None]
    eupv = float(meta.get("eu_per_volt") or 1.0)
    torque = data[:, 0] * eupv
    kph = data[:, 1] if data.shape[1] > 1 else np.zeros_like(torque)
    fs = float(meta.get("fs") or fs_dl or 2560.0)
    setup = meta.get("setup", {}) if isinstance(meta.get("setup"), dict) else {}
    name = setup.get("machine") or setup.get("tag") or "Cloud run"
    return dict(torque=torque, kph=kph, fs=fs, rpm=None, units=meta.get("units", "nm"),
                name=name, field=True, setup=setup)


# =====================================================================
# Encabezado + tema
# =====================================================================
page_header("Watermelon Torsional",
            "Shaft torque & torsional vibration — TorqueTrak 10K · NI 9229 · analysis only")
_inject_theme()

with st.expander("⚙  Data source & scaling", expanded=False):
    c1, c2 = st.columns([1, 1])
    with c1:
        source = st.radio("Source", ["Cloud runs (field)", "Simulated demo", "Upload run (.npz)"],
                          horizontal=True)
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
    if source == "Cloud runs (field)":
        _runs = _cloud_run_list()
        if not _runs:
            st.info("No cloud runs yet (or offline). Field uploads from the .exe appear here automatically.")
        else:
            _opts = {f"{r.get('name','run')}  ·  {r.get('client') or '—'}  ·  "
                     f"{str(r.get('updated_at',''))[:16]}  ·  {r.get('hostname','')}": r["id"]
                     for r in _runs}
            _pick = st.selectbox(f"Field run  ({len(_runs)} available)", list(_opts))
            run = _load_cloud_run(_opts[_pick])
            if run is None:
                st.warning("Could not load the raw data for this run (missing raw or service key).")
    elif source == "Upload run (.npz)":
        up = st.file_uploader("Run file (.npz with 'torque' or 'volts', 'kph', 'fs')", type=["npz"])
        if up is not None:
            try:
                run = _parse_upload(up, sc)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not read run: {exc}")

if run is None:
    run = _demo_steady(units=units_key)

# ============================================================ MONITOR 24h
# Corrida de monitoreo de larga duración: histograma acumulado (no onda). Reporte propio.
if isinstance(run, dict) and run.get("kind") == "monitor":
    pay = run["payload"]
    um = "N·m" if pay.get("units", "nm") == "nm" else "ft-lb"
    ranges_m = [(float(r), float(c)) for r, c in pay.get("ranges", [])]
    trend_m = pay.get("trend", [])
    events_m = pay.get("events", [])
    dur_h = float(pay.get("duration_s", 0.0)) / 3600.0
    meanm = float(pay.get("mean", 0.0)); tmaxm = float(pay.get("tmax") or 0.0)
    shaft = pay.get("shaft", {}) if isinstance(pay.get("shaft"), dict) else {}
    ncyc = sum(c for _, c in ranges_m)
    rmax_m = max((r for r, _ in ranges_m), default=0.0)

    _sec("24-hour torsional monitor", "Accumulated rainflow fatigue (real Miner damage) · trend · overload events")
    _mm1, _mm2, _mm3 = st.columns(3)
    do_m = _mm1.number_input("Shaft Ø outer (in)", 0.1, 100.0, float(shaft.get("do", 3.0)), 0.1, key="mon_do")
    di_m = _mm2.number_input("Shaft Ø inner (in)", 0.0, 99.0, float(shaft.get("di", 0.0)), 0.1, key="mon_di")
    sut_m = _mm3.number_input("Ultimate Sut (ksi)", 10.0, 400.0, float(shaft.get("sut_ksi", 90.0)), 1.0, key="mon_sut")
    cyc_m = [RainflowCycle(range=r, mean=meanm, count=c) for r, c in ranges_m if r > 0]
    life_m = None
    if cyc_m:
        try:
            life_m = shaft_torsional_fatigue(cyc_m, do_m, di_m, sut_m * 1000.0, pay.get("units", "nm"),
                                             window_seconds=max(float(pay.get("duration_s", 1.0)), 1.0))
        except ValueError as exc:
            st.warning(f"Check shaft geometry — {exc}")
    if life_m is not None:
        _bg = {"green": ("#dcfce7", "#166534"), "yellow": ("#fef9c3", "#854d0e"),
               "red": ("#fee2e2", "#991b1b")}[life_m.status]
        _sf = "∞" if life_m.safety_factor == float("inf") else f"{life_m.safety_factor:.2f}"
        _lf = ("Infinite life" if life_m.infinite else
               (f"~{life_m.life_hours:,.0f} h" if life_m.life_hours < 8760 else f"~{life_m.life_hours/8760:,.1f} yr"))
        st.markdown(f"<div style='background:{_bg[0]};color:{_bg[1]};border-radius:12px;padding:14px 18px;"
                    f"font-size:18px;font-weight:800;margin:6px 0'>● {life_m.label_en} "
                    f"<span style='font-weight:600;font-size:14px'>· SF {_sf} · {_lf} · over {dur_h:.1f} h monitored</span></div>",
                    unsafe_allow_html=True)
    _kpis([
        (f"{dur_h:,.1f}<span style='font-size:13px'> h</span>", "Duration", "monitored"),
        (f"{ncyc:,.0f}", "Total cycles", "rainflow"),
        (f"{len(events_m)}", "Overload events", "> 3× mean pp"),
        (f"{tmaxm:,.0f}<span style='font-size:13px'> {um}</span>", "Max torque", "peak seen"),
    ])
    # tendencia (par pp vs tiempo)
    if trend_m:
        tt = [row[0] / 3600.0 for row in trend_m]; pp = [row[2] for row in trend_m]
        fig_t = go.Figure(go.Scatter(x=tt, y=pp, mode="lines", line=dict(color=BLUE, width=1.6), name="torque pp"))
        _navplot(_apply(fig_t, height=300, xlab="time (h)", ylab=f"torque pp ({um})"))
    # histograma acumulado
    if ranges_m:
        rr = np.array([r for r, _ in ranges_m]); cc = np.array([c for _, c in ranges_m])
        nb = int(np.clip(len(rr), 8, 30)); edg = np.linspace(0, rr.max() * 1.0001, nb + 1)
        hh, _ = np.histogram(rr, bins=edg, weights=cc)
        fig_h = go.Figure(go.Bar(x=0.5 * (edg[:-1] + edg[1:]), y=hh, width=(edg[1] - edg[0]) * 0.92, marker_color=NAVY))
        _navplot(_apply(fig_h, height=300, xlab=f"torque range ({um})", ylab="cycle count"))
    # eventos
    if events_m:
        _erows = "".join(f"<tr><td class='num'>{e.get('t',0)/3600.0:.2f}<span class='u'> h</span></td>"
                         f"<td class='num'>{e.get('peak',0):,.0f}<span class='u'> {um}</span></td>"
                         f"<td class='num'>{e.get('rpm',0):,.0f}<span class='u'> rpm</span></td></tr>" for e in events_m[:50])
        st.markdown("<table class='wm-modes'><thead><tr><th>Time</th><th>Peak torque</th><th>Speed</th></tr></thead>"
                    f"<tbody>{_erows}</tbody></table>", unsafe_allow_html=True)
    else:
        st.success("No overload events during the campaign.")

    # --- PDF de monitoreo 24h ---
    _mlang = st.radio("Language", ["Español", "English"], horizontal=True, key="mon_rep_lang")
    _mes = (_mlang == "Español")
    if st.button(("📄 Generar reporte de monitoreo 24h (PDF)" if _mes else "📄 Generate 24h monitor report (PDF)"),
                 type="primary", key="mon_pdf"):
        with st.spinner("…"):
            from core.torsional.report import build_torsional_pdf, plotly_to_png
            f_tr = go.Figure()
            if trend_m:
                f_tr.add_trace(go.Scatter(x=[r[0] / 3600.0 for r in trend_m], y=[r[2] for r in trend_m],
                                          line=dict(color=BLUE, width=1.6)))
                _apply(f_tr, xlab="time (h)", ylab=f"torque pp ({um})")
            f_hi = go.Figure()
            if ranges_m:
                f_hi.add_trace(go.Bar(x=0.5 * (edg[:-1] + edg[1:]), y=hh, width=(edg[1] - edg[0]) * 0.92, marker_color=NAVY))
                _apply(f_hi, xlab=f"torque range ({um})", ylab="cycle count")
            _sfx = ("∞" if (life_m and life_m.safety_factor == float("inf")) else (f"{life_m.safety_factor:.2f}" if life_m else "—"))
            _verd = (life_m.label_es if _mes else life_m.label_en) if life_m else "—"
            find = ([f"Campaña de monitoreo torsional de {dur_h:.1f} h; {ncyc:,.0f} ciclos rainflow acumulados (ASTM E1049).",
                     f"Par medio {meanm:,.0f} {um}; par máximo observado {tmaxm:,.0f} {um}; mayor rango {rmax_m:,.0f} {um}.",
                     f"Vida a fatiga (Goodman/Miner): factor de seguridad {_sfx} — {_verd}.",
                     f"{len(events_m)} evento(s) de sobrecarga registrados durante la campaña."] if _mes else
                    [f"{dur_h:.1f} h torsional monitoring campaign; {ncyc:,.0f} accumulated rainflow cycles (ASTM E1049).",
                     f"Mean torque {meanm:,.0f} {um}; peak torque {tmaxm:,.0f} {um}; largest range {rmax_m:,.0f} {um}.",
                     f"Fatigue life (Goodman/Miner): safety factor {_sfx} — {_verd}.",
                     f"{len(events_m)} overload event(s) logged during the campaign."])
            recs = (["Comparar el daño acumulado contra el diagrama S-N/Goodman del eje en la galga.",
                     "Investigar los eventos de sobrecarga (arranques/trips/transitorios de proceso)." if events_m else
                     "Sin sobrecargas; mantener el plan de monitoreo periódico."] if _mes else
                    ["Compare accumulated damage against the shaft S-N/Goodman diagram at the gage.",
                     "Investigate the overload events (startups/trips/process transients)." if events_m else
                     "No overloads; keep the periodic monitoring plan."])
            _setup = pay.get("setup", {})
            meta = {"report_title": ("MONITOREO TORSIONAL 24 H" if _mes else "24-HOUR TORSIONAL MONITORING"),
                    "asset": _setup.get("machine", run.get("name", "")), "client": _setup.get("client", ""),
                    "location": _setup.get("location", ""), "prepared_by": _setup.get("operator", ""),
                    "prepared_role": "Especialista", "reviewed_by": _setup.get("approved_by", ""), "reviewed_role": "Gerente",
                    "consecutive": f"TOR-MON-{run.get('name','')[:12]}", "report_date": "", "date": "",
                    "format_code": "SIGA-FMT-181", "format_version": "1"}
            ctx = {"name": run.get("name", "monitor"), "units_label": um, "rpm": (trend_m[-1][4] if trend_m else 0),
                   "mean": meanm, "pp": rmax_m, "ripple": "—", "rms": 0.0, "fs": 0.0, "dominant": "—"}
            pdf = build_torsional_pdf(meta=meta, context=ctx, findings=find, recommendations=recs,
                                      waveform_png=plotly_to_png(f_tr), spectrum_png=None,
                                      campbell_png=None, fatigue_png=plotly_to_png(f_hi),
                                      order_rows=[], crossing_rows=[], naturals=[], lang=("es" if _mes else "en"))
            st.session_state["_mon_pdf"] = pdf
        st.success("Reporte listo." if _mes else "Report ready.")
    if st.session_state.get("_mon_pdf"):
        st.download_button(("⬇ Descargar reporte 24h (PDF)" if _mes else "⬇ Download 24h report (PDF)"),
                           st.session_state["_mon_pdf"], file_name="watermelon_torsional_monitor_24h.pdf",
                           mime="application/pdf", type="primary")
    st.stop()

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
_NAV = [T_OVR, T_SPEC, T_ORD, T_CAMP, T_FAT, T_REPORT]
if "tors_nav" not in st.session_state:
    st.session_state["tors_nav"] = T_OVR
if hasattr(st, "segmented_control"):
    nav = st.segmented_control("Section", _NAV, key="tors_nav",
                               label_visibility="collapsed") or st.session_state["tors_nav"]
else:
    nav = st.radio("Section", _NAV, horizontal=True, key="tors_nav")


# --------------------------------------------------------------- Overview
if nav == T_OVR:
    _sec("Torque waveform", "zoomed to ~10 revolutions · ISO 22266 / API 684")
    # Ventana de ~10 vueltas para que el rizado se lea (no un blob denso).
    n_win = int(np.clip(fs * 10.0 * 60.0 / max(rpm, 1.0), 256, torque.size))
    tw = np.arange(n_win) / fs
    yw = torque[:n_win]
    fig = go.Figure()
    # relleno suave entre la media y la curva (banda de par dinámico)
    fig.add_trace(go.Scatter(x=tw, y=np.full(n_win, m.mean), mode="lines",
                             line=dict(width=0), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=tw, y=yw, mode="lines", name="torque",
                             line=dict(color=BLUE, width=1.8), fill="tonexty",
                             fillcolor="rgba(37,99,235,0.10)"))
    # marcadores de vuelta (keyphasor) dentro de la ventana
    below = kph[:n_win] < -1.0
    edges = np.where(below[1:] & ~below[:-1])[0] + 1
    for e in edges:
        fig.add_vline(x=e / fs, line=dict(color="#cbd5e1", width=1))
    fig.add_hline(y=m.mean, line=dict(color=SLATE, dash="dash"),
                  annotation_text=f"mean {m.mean:,.0f} {u}", annotation_position="top left")
    _navplot(_apply(fig, height=400, xlab="time (s)", ylab=f"torque ({u})"))
    st.caption(f"RMS **{m.rms:,.1f} {u}** · crest factor **{m.crest_factor:.2f}** · "
               f"dynamic peak **{m.peak_to_peak/2:,.1f} {u}** · vertical lines = shaft revolutions (keyphasor)")

# ---------------------------------------------------- Spectrum & orders
elif nav == T_SPEC:
    freqs, amp = torque_spectrum(torque, fs)
    mask = freqs <= 600.0
    f1 = rpm / 60.0
    oa = order_amplitudes(torque, fs, rpm, orders=(1, 2, 3, 4, 5))
    orders = list(range(1, 6))
    scale = st.radio("Scale", ["dB", "Linear"], horizontal=True, key="tors_spec_scale",
                     label_visibility="collapsed")
    _sec("Torque order spectrum", f"1× = {f1:.1f} Hz · equipment bandwidth 500 Hz · {scale}")

    if scale == "dB":
        y = 20.0 * np.log10(np.maximum(amp[mask], 1e-9))
        ylab = f"amplitude (dB re 1 {u})"
    else:
        y = amp[mask]; ylab = f"amplitude ({u})"
    fig = go.Figure(go.Scatter(x=freqs[mask], y=y, mode="lines",
                               line=dict(color=NAVY, width=1.6),
                               fill=("tozeroy" if scale == "Linear" else None),
                               fillcolor="rgba(15,30,61,0.06)", name="spectrum"))
    # cursores de orden (líneas de excitación k×)
    for k in range(1, 6):
        if k * f1 <= 600:
            fig.add_vline(x=k * f1, line=dict(color=AMBER, dash="dot", width=1))
            fig.add_annotation(x=k * f1, y=1, yref="paper", text=f"{k}×", showarrow=False,
                               font=dict(size=10, color=AMBER), yshift=-2)
    # marcadores de pico en las 3 órdenes dominantes (etiqueta con amplitud)
    top = sorted(range(1, 6), key=lambda k: oa[float(k)][0], reverse=True)[:3]
    for k in top:
        fk = k * f1; ak = oa[float(k)][0]
        yv = 20.0 * np.log10(max(ak, 1e-9)) if scale == "dB" else ak
        fig.add_trace(go.Scatter(x=[fk], y=[yv], mode="markers+text", text=[f" {k}× {ak:.1f}"],
                                 textposition="top center", showlegend=False,
                                 marker=dict(color=GREEN, size=9, line=dict(color="white", width=1.5)),
                                 textfont=dict(size=10, color=NAVY)))
    fig.update_layout(showlegend=False)
    _navplot(_apply(fig, height=360, xlab="frequency (Hz)", ylab=ylab))
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
    src = _runup_source(torque, kph, fs, bool(run.get("field")), units=run["units"])
    if src["measured"]:
        st.success(f"● Measured from this run-up ({src['rpm_min']:,.0f} → {src['rpm_max']:,.0f} rpm). "
                   f"Detected torsional natural(s): {', '.join(f'{x:.1f} Hz' for x in src['naturals']) or '—'}.")
    else:
        st.warning("● Simulated run-up demo — this run is **not** a run-up, so torsional naturals "
                   "cannot be measured from it. A run-up / coast-down capture is required. "
                   "The curves below illustrate the method only.")
    fig = go.Figure()
    for (o, rpm_arr, amp_arr), col in zip(src["tracks"], (BLUE, GREEN, AMBER)):
        fig.add_trace(go.Scatter(x=rpm_arr, y=amp_arr, mode="lines+markers",
                                 line=dict(color=col, width=2.2), marker=dict(size=4),
                                 name=f"{int(o)}×"))
    for fn in src["naturals"]:
        fig.add_vline(x=fn * 60.0, line=dict(color=RED, dash="dash"),
                      annotation_text=f"natural {fn:.1f} Hz",
                      annotation_position="top", annotation_font=dict(color=RED, size=11))
    _navplot(_apply(fig, height=440, xlab="RPM", ylab=f"order amplitude ({u})"))
    st.caption("Each order peaks where k × running speed crosses a torsional natural frequency. "
               "Naturals are detected only from a real run-up / coast-down, never from a steady capture.")

# -------------------------------------------------------------- Fatigue
elif nav == T_FAT:
    _sec("Shaft fatigue-life diagnostic",
         "ASTM E1049 rainflow → shear stress → Goodman → Palmgren-Miner. Traffic light by safety factor (API 684 / ASME B106.1M)")
    ranges = fatigue_ranges(torque)
    if not ranges:
        st.info("Not enough reversals to count cycles.")
    else:
        rr = np.array([r for r, _ in ranges]); cc = np.array([c for _, c in ranges])
        rmax = float(rr.max())
        # --- Datos del eje (para el diagnóstico de vida) ---
        # (nombre, Sut ksi, Se'/Sut) — mismo catálogo que la app de campo.
        _MATCAT = {
            "Steel 4140 Q&T": (140.0, 0.50), "Steel 4140 annealed": (95.0, 0.50),
            "Steel 1045": (90.0, 0.50), "Ductile iron (80-55-06)": (80.0, 0.45),
            "Gray cast iron (class 40)": (40.0, 0.40), "Stainless (410/17-4)": (95.0, 0.50),
            "Aluminum (6061-T6)": (45.0, 0.45), "Titanium (Ti-6Al-4V)": (130.0, 0.45),
            "Custom": (None, 0.50),
        }
        _shaft = run.get("shaft", {}) if isinstance(run.get("shaft"), dict) else {}
        c1, c2, c3, c4, c5 = st.columns(5)
        mat = c1.selectbox("Material", list(_MATCAT), key="tors_fat_mat")
        _mat_sut, _endr = _MATCAT[mat]
        do_in = c2.number_input("Shaft Ø outer (in)", 0.1, 100.0,
                                float(_shaft.get("do", 3.0)), 0.1, key="tors_fat_do")
        di_in = c3.number_input("Shaft Ø inner (in)", 0.0, 99.0,
                                float(_shaft.get("di", 0.0)), 0.1, key="tors_fat_di")
        if _mat_sut is not None:
            sut_ksi = float(_mat_sut)
            c4.metric("Ultimate Sut", f"{sut_ksi:.0f} ksi")
        else:
            sut_ksi = c4.number_input("Ultimate Sut (ksi)", 10.0, 400.0,
                                      float(_shaft.get("sut", 90.0)), 1.0, key="tors_fat_sut")
        design_sf = c5.selectbox("Design safety factor", [2.0, 1.5, 3.0], key="tors_fat_sf")
        try:
            life = shaft_torsional_fatigue(
                rainflow_cycles(torque), outer_diameter_in=do_in, inner_diameter_in=di_in,
                ultimate_strength_psi=sut_ksi * 1000.0, torque_units=run["units"],
                window_seconds=float(torque.size / fs), design_safety_factor=float(design_sf),
                endurance_ratio=float(_endr))
        except ValueError as exc:      # geometría no diagnosticable
            st.warning(f"Check shaft geometry — {exc}")
            st.stop()
        _bg = {"green": ("#dcfce7", "#166534", GREEN), "yellow": ("#fef9c3", "#854d0e", AMBER),
               "red": ("#fee2e2", "#991b1b", RED)}[life.status]
        _sf_txt = "∞" if life.safety_factor == float("inf") else f"{life.safety_factor:.2f}"
        if life.infinite:
            _life_txt = "Infinite life"
        elif life.life_hours < 8760:
            _life_txt = f"~{life.life_hours:,.0f} h"
        else:
            _life_txt = f"~{life.life_hours/8760:,.1f} yr"
        st.markdown(
            f"<div style='background:{_bg[0]};color:{_bg[1]};border-radius:12px;padding:14px 18px;"
            f"font-size:18px;font-weight:800;margin:6px 0'>● {life.label_en} "
            f"<span style='font-weight:600;font-size:14px'>· SF {_sf_txt} · {_life_txt}</span></div>",
            unsafe_allow_html=True)
        # Ciclos de rango grande (>50% del máx) — los que dominan el daño de fatiga.
        damaging = float(cc[rr > 0.5 * rmax].sum())
        _kpis([
            (f"{_sf_txt}", "Safety factor", "vs endurance limit"),
            (f"{life.tau_alt_max_psi/1000:,.1f}<span style='font-size:13px'> ksi</span>", "Alt. shear τar", "Goodman-corrected"),
            (f"{cc.sum():,.0f}", "Total cycles", "counted"),
            (f"{rmax:,.1f}<span style='font-size:13px'> {u}</span>", "Largest range", "worst cycle"),
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

# -------------------------------------------------------------- Campbell
elif nav == T_CAMP:
    _sec("Campbell / interference diagram",
         "API 684 — excitation orders k×RPM vs torsional natural frequencies")
    ra = _runup_source(torque, kph, fs, bool(run.get("field")), units=run["units"])
    naturals = ra["naturals"] or ([ra["res_hz"]] if not ra["measured"] else [])
    if ra["measured"]:
        st.success(f"● Natural(s) measured from this run-up: "
                   f"{', '.join(f'{x:.1f} Hz' for x in naturals) or '—'}.")
    else:
        st.warning("● Illustrative Campbell — torsional naturals were **not** measured from this "
                   "run (no run-up). Shown for method demonstration; not a diagnosis of this asset.")
    rpm_max = max(ra["rpm_max"] * 1.05, rpm * 1.15)
    orders_c = (1.0, 2.0, 3.0, 4.0, 6.0)          # 6× cubre VFD/engrane
    band = SpeedBand(center_rpm=rpm, tol_rpm=0.10 * rpm, label="Operating ±10%")
    labels = [f"TNF{i+1}" for i in range(len(naturals))]
    crossings = compute_crossings(naturals, 0.0, rpm_max, orders_c, bands=[band],
                                  mode_labels=labels)
    xr = np.linspace(0.0, rpm_max, 80)
    fig = go.Figure()
    fig.add_vrect(x0=band.low, x1=band.high, fillcolor="rgba(245,158,11,0.13)", line_width=0,
                  annotation_text="Operating ±10%", annotation_position="top left",
                  annotation_font=dict(size=10, color="#b45309"))
    for o in orders_c:
        fig.add_trace(go.Scatter(x=xr, y=o * xr / 60.0, mode="lines", name=f"{o:g}×",
                                 line=dict(color="#94a3b8", width=1, dash="dot"),
                                 hovertemplate=f"{o:g}× rpm<extra></extra>"))
        fig.add_annotation(x=rpm_max * 0.98, y=o * rpm_max / 60.0, text=f"{o:g}×",
                           showarrow=False, font=dict(size=10, color="#94a3b8"))
    for i, fn in enumerate(naturals):
        fig.add_trace(go.Scatter(x=[0.0, rpm_max], y=[fn, fn], mode="lines",
                                 name=f"TNF{i+1} · {fn:.1f} Hz", line=dict(color=GREEN, width=2.4)))
    fig.add_vline(x=rpm, line=dict(color=NAVY, width=2, dash="dash"),
                  annotation_text=f"Operating {rpm:.0f} rpm", annotation_position="bottom right")
    _sc = {"coincidence": RED, "near": AMBER, "clear": "#94a3b8"}
    for c in crossings:
        fig.add_trace(go.Scatter(x=[c.crossing_rpm], y=[c.mode_hz], mode="markers", showlegend=False,
                                 marker=dict(color=_sc[c.severity], size=13, symbol="x",
                                             line=dict(width=2, color="#7f1d1d")),
                                 hovertemplate=(f"{c.crossing_rpm:.0f} rpm · {c.order:g}× · "
                                                f"margin {c.sep_margin_pct:.0f}%<extra></extra>")))
    ymax = (max(naturals) * 1.35) if naturals else 100.0
    fig.update_layout(showlegend=True, yaxis_range=[0, ymax])
    _navplot(_apply(fig, height=470, xlab="speed (RPM)", ylab="frequency (Hz)"))

    if crossings:
        def _sevpill(s):
            _p = {"coincidence": PILL_RED, "near": PILL_AMBER, "clear": PILL_SLATE}[s]
            _t = {"coincidence": "coincidence", "near": "near", "clear": "clear"}[s]
            return _pill(_t, *_p)
        rows = "".join(
            f'<tr><td class="idx">{c.mode_label}</td>'
            f'<td class="num">{c.mode_hz:.1f}<span class="u"> Hz</span></td>'
            f'<td class="num">{c.order:g}×</td>'
            f'<td class="num">{c.crossing_rpm:.0f}<span class="u"> rpm</span></td>'
            f'<td class="num">{c.sep_margin_pct:.0f}<span class="u"> %</span></td>'
            f'<td>{_sevpill(c.severity)}</td></tr>' for c in crossings)
        st.markdown('<table class="wm-modes"><thead><tr><th>Natural</th><th>Frequency</th>'
                    '<th>Order</th><th>Crossing</th><th>Sep. margin</th><th>Status</th>'
                    f'</tr></thead><tbody>{rows}</tbody></table>', unsafe_allow_html=True)
    else:
        st.success("No order crossings within the evaluated speed range — clear of resonances.")
    st.caption("API 684 §1.6 — a natural ↔ order coincidence marks a zone of interest, not a "
               "confirmed resonance; correlate with amplitude & phase in operation. Target "
               "separation margin ≥ 10%.")

# ---------------------------------------------------------------- Report
elif nav == T_REPORT:
    _sec("Executive report", "SIGA torsional analysis — auto-generated · API 684 / ISO 22266")
    ra = _runup_source(torque, kph, fs, bool(run.get("field")), units=run["units"])
    measured = ra["measured"]     # ¿las naturales salen de un run-up REAL de esta corrida?
    naturals = ra["naturals"] or ([ra["res_hz"]] if not measured else [])
    band = SpeedBand(center_rpm=rpm, tol_rpm=0.10 * rpm, label="Operating ±10%")
    crossings = compute_crossings(naturals, 0.0, max(ra["rpm_max"] * 1.05, rpm * 1.15),
                                  (1.0, 2.0, 3.0, 4.0, 6.0), bands=[band],
                                  mode_labels=[f"TNF{i+1}" for i in range(len(naturals))])
    if not measured:
        st.warning("⚠ Esta corrida no es un run-up: las frecuencias naturales torsionales NO se "
                   "midieron aquí. El Campbell y las conclusiones de resonancia del reporte son "
                   "ilustrativos, no un diagnóstico de resonancia de este activo. Sube una corrida "
                   "de run-up / coast-down para el veredicto API 684."
                   if st.session_state.get("tors_rep_lang", "Español") == "Español" else
                   "⚠ This run is not a run-up: torsional naturals were NOT measured here. The "
                   "report's Campbell and resonance conclusions are illustrative, not a resonance "
                   "diagnosis of this asset. Upload a run-up / coast-down for the API 684 verdict.")
    coincid = [c for c in crossings if c.severity == "coincidence"]
    worst = min((c.sep_margin_pct for c in crossings), default=float("inf"))
    oa = order_amplitudes(torque, fs, rpm, orders=(1, 2, 3, 4, 5))
    dom = max(range(1, 6), key=lambda k: oa[float(k)][0])
    ranges = fatigue_ranges(torque)
    rmax = max((r for r, _ in ranges), default=0.0)

    if not measured:
        verdict, vcls = ("SIN VEREDICTO DE RESONANCIA — falta un run-up para medir las naturales", "wm-rev")
    elif coincid:
        verdict, vcls = ("ATENCIÓN — coincidencia de orden en la banda de operación", "wm-nogo")
    elif worst < 10.0:
        verdict, vcls = ("REVISAR — margen de separación menor a 10%", "wm-rev")
    else:
        verdict, vcls = ("ACEPTABLE — libre de resonancias torsionales", "wm-go")

    _lang = st.radio("Language", ["Español", "English"], horizontal=True, key="tors_rep_lang")
    _es = (_lang == "Español")
    _rip = "∞" if m.ripple_pct == float("inf") else f"{m.ripple_pct:.1f}%"
    _nats = ", ".join(f"{fn:.1f} Hz" for fn in naturals) if naturals else ("—")

    def _auto_findings(es):
        f = []
        if es:
            f.append(f"Par medio {m.mean:,.0f} {u}, pico-pico dinámico {m.peak_to_peak:,.0f} {u} (rizado {_rip}) a {rpm:,.0f} rpm.")
            f.append(f"Orden de excitación dominante: {dom}× ({oa[float(dom)][0]:.1f} {u}).")
            if measured:
                f.append(f"Frecuencias naturales torsionales MEDIDAS en el run-up ({ra['rpm_min']:,.0f}→{ra['rpm_max']:,.0f} rpm): {_nats}.")
                f.append(f"{len(coincid)} coincidencia(s) de orden dentro de la banda de operación — margen mínimo {worst:.0f}% (objetivo API 684 ≥ 10%)." if coincid
                         else f"Sin coincidencias en la banda de operación; margen de separación mínimo {worst:.0f}%.")
            else:
                f.append("No se capturó run-up en esta corrida — las frecuencias naturales torsionales NO se midieron; se requiere un run-up / coast-down para evaluar resonancias (API 684). El Campbell mostrado es ilustrativo.")
            f.append(f"Fatiga: mayor rango rainflow del par {rmax:,.0f} {u} (ASTM E1049) — entrada para esfuerzo/Goodman del eje.")
        else:
            f.append(f"Mean torque {m.mean:,.0f} {u}, dynamic peak-peak {m.peak_to_peak:,.0f} {u} (ripple {_rip}) at {rpm:,.0f} rpm.")
            f.append(f"Dominant excitation order: {dom}× ({oa[float(dom)][0]:.1f} {u}).")
            if measured:
                f.append(f"Torsional natural frequencies MEASURED in the run-up ({ra['rpm_min']:,.0f}→{ra['rpm_max']:,.0f} rpm): {_nats}.")
                f.append(f"{len(coincid)} order coincidence(s) inside the operating band — worst separation margin {worst:.0f}% (API 684 target ≥ 10%)." if coincid
                         else f"No coincidences in the operating band; worst separation margin {worst:.0f}%.")
            else:
                f.append("No run-up captured in this run — torsional natural frequencies were NOT measured; a run-up / coast-down is required to assess resonances (API 684). The Campbell shown is illustrative.")
            f.append(f"Fatigue: largest rainflow torque range {rmax:,.0f} {u} (ASTM E1049) — input for shaft stress / Goodman.")
        return f

    def _auto_recs(es):
        r = []
        if coincid:
            r.append("Confirmar la coincidencia señalada con una corrida de amplitud/fase en operación; si se confirma, desintonizar (rigidez de acople / inercia) o restringir la banda de velocidad." if es
                     else "Confirm the flagged coincidence with an operating amplitude/phase run; if confirmed, detune (coupling stiffness / inertia) or restrict the speed band.")
        if m.ripple_pct >= 25:
            r.append("Rizado de par elevado — revisar órdenes de engrane / VFD y la condición del acople." if es
                     else "High torque ripple — check gear mesh / VFD orders and coupling condition.")
        r.append("Evaluar la fatiga del eje con el histograma rainflow contra el diagrama S-N / Goodman en la ubicación de la galga." if es
                 else "Evaluate shaft fatigue with the rainflow histogram against the shaft S-N / Goodman diagram at the gage location.")
        r.append("Reverificar la calibración con el shunt a bordo (Ref 1/Ref 2) antes de la próxima campaña." if es
                 else "Re-verify calibration with the on-board shunt (Ref 1/Ref 2) before the next campaign.")
        return r

    # Auto-diagnóstico (banner azul, como el Modal)
    if not measured:
        _nar = (f"Órdenes de excitación 1×–5× a {rpm:,.0f} rpm. Esta corrida NO es un run-up, así que "
                "las frecuencias naturales torsionales no se midieron: no hay veredicto de resonancia. "
                "Sube un run-up / coast-down para evaluar coincidencias (API 684)." if _es else
                f"Excitation orders 1×–5× at {rpm:,.0f} rpm. This run is NOT a run-up, so torsional "
                "natural frequencies were not measured: no resonance verdict. Upload a run-up / "
                "coast-down to assess coincidences (API 684).")
    elif _es:
        _nar = (f"Se MIDIERON las naturales torsionales en el run-up ({ra['rpm_min']:,.0f}→{ra['rpm_max']:,.0f} rpm): {_nats}. "
                + (f"⚠ Coincidencia {coincid[0].order:g}× dentro de la banda de operación (margen {worst:.0f}%, API 684) — riesgo de resonancia torsional; correlacionar con amplitud/fase."
                   if coincid else f"Sin coincidencias dentro de la banda de operación (margen mínimo {worst:.0f}%)."))
    else:
        _nar = (f"Torsional naturals MEASURED in the run-up ({ra['rpm_min']:,.0f}→{ra['rpm_max']:,.0f} rpm): {_nats}. "
                + (f"⚠ {coincid[0].order:g}× coincidence within the operating band (margin {worst:.0f}%, API 684) — torsional resonance risk; correlate with amplitude/phase."
                   if coincid else f"No coincidences within the operating band (worst margin {worst:.0f}%)."))
    st.markdown(f"<div style='background:#eef6ff;border-left:4px solid {BLUE};border-radius:8px;"
                f"padding:12px 16px;margin:6px 0'><b>{'Auto-diagnóstico' if _es else 'Auto-diagnosis'}</b><br>"
                f'<span class="wm-chip {vcls}" style="margin:4px 0 6px 0">● {verdict}</span><br>{_nar}</div>',
                unsafe_allow_html=True)

    # Identificación del reporte (consecutivo automático + firmas)
    from datetime import date as _date
    st.markdown(f"**{'Identificación del reporte' if _es else 'Report identification'}**")
    _pfx = f"TOR-{_date.today().year}-"
    if st.session_state.get("tors_consec_pref") != _pfx:
        try:
            from core.reports_archive import next_consecutive as _nc
            st.session_state["tors_consec_auto"] = _nc(_pfx, _user.get("email", ""), _my_role)
        except Exception:  # noqa: BLE001
            st.session_state["tors_consec_auto"] = f"{_pfx}001"
        st.session_state["tors_consec_pref"] = _pfx
        st.session_state.pop("tors_consec", None)
    _uname = (_user.get("full_name") or _user.get("name") or _user.get("email", "")).split("@")[0]
    ci = st.columns([1.2, 1, 1.4])
    with ci[0]:
        r_consec = st.text_input("Consecutive (auto)", key="tors_consec",
                                 value=st.session_state.get("tors_consec", st.session_state["tors_consec_auto"]),
                                 help="Prefijo automático TOR-AÑO-NNN por histórico. Editable.")
    with ci[1]:
        r_date = st.text_input("Date", key="tors_date", value=st.session_state.get("tors_date", str(_date.today())))
    with ci[2]:
        r_asset = st.text_input("Asset / Tag", key="tors_asset", value=st.session_state.get("tors_asset", run.get("name", "")))
    cj = st.columns(2)
    with cj[0]:
        r_client = st.text_input("Client", key="tors_client", value=st.session_state.get("tors_client", ""))
    with cj[1]:
        r_location = st.text_input("Location", key="tors_loc", value=st.session_state.get("tors_loc", ""))
    ck = st.columns(2)
    with ck[0]:
        r_prep = st.text_input("Prepared by (Realizado por)", key="tors_prep", value=st.session_state.get("tors_prep", _uname))
        r_prep_role = st.text_input("Role", key="tors_prep_role", value=st.session_state.get("tors_prep_role", "Especialista"))
    with ck[1]:
        r_rev = st.text_input("Approved by (Aprobado por)", key="tors_rev", value=st.session_state.get("tors_rev", ""))
        r_rev_role = st.text_input("Role ", key="tors_rev_role", value=st.session_state.get("tors_rev_role", "Gerente"))
    r_city = st.text_input("City", key="tors_city", value=st.session_state.get("tors_city", "Bogotá D.C."))

    # Hallazgos y recomendaciones EDITABLES
    st.markdown(f"**{'Hallazgos y recomendaciones' if _es else 'Findings & recommendations'}** "
                f"*({'uno por línea — editable' if _es else 'one per line — edit freely'})*")
    fr = st.columns(2)
    with fr[0]:
        find_txt = st.text_area("Findings (Hallazgos)", key="tors_find",
                                value=st.session_state.get("tors_find", "\n".join(_auto_findings(_es))), height=175)
    with fr[1]:
        rec_txt = st.text_area("Recommendations (Recomendaciones)", key="tors_rec",
                               value=st.session_state.get("tors_rec", "\n".join(_auto_recs(_es))), height=175)
    st.caption("Incrustado en el PDF: onda de par · espectro de órdenes · Campbell (API 684) · fatiga rainflow · tablas de órdenes y cruces." if _es
               else "Embedded in the PDF: torque waveform · order spectrum · Campbell (API 684) · rainflow fatigue · order & crossing tables.")

    if st.button(("📄 Generar reporte completo (PDF)" if _es else "📄 Generate full report (PDF)"),
                 type="primary", key="tors_pdf_gen"):
        with st.spinner("Renderizando figuras y armando el reporte SIGA…" if _es else "Rendering figures & building the SIGA report…"):
            from core.torsional.report import build_torsional_pdf, plotly_to_png
            t = np.arange(min(torque.size, int(fs * 10 * 60 / max(rpm, 1)))) / fs
            yw = torque[:t.size]
            f_wave = go.Figure()
            f_wave.add_trace(go.Scatter(x=t, y=np.full(t.size, m.mean), line=dict(width=0), hoverinfo="skip", showlegend=False))
            f_wave.add_trace(go.Scatter(x=t, y=yw, line=dict(color=BLUE, width=1.8), fill="tonexty",
                                        fillcolor="rgba(37,99,235,0.10)", name="torque"))
            _apply(f_wave, xlab="time (s)", ylab=f"torque ({u})")
            freqs, amp = torque_spectrum(torque, fs); mk = freqs <= 600.0
            f_spec = go.Figure(go.Scatter(x=freqs[mk], y=20 * np.log10(np.maximum(amp[mk], 1e-9)),
                                          line=dict(color=NAVY, width=1.6), name="spectrum"))
            for k in range(1, 6):
                if k * rpm / 60 <= 600:
                    f_spec.add_vline(x=k * rpm / 60, line=dict(color=AMBER, dash="dot", width=1))
            _apply(f_spec, xlab="frequency (Hz)", ylab=f"amplitude (dB re 1 {u})")
            rpm_max = max(ra["rpm_max"] * 1.05, rpm * 1.15)
            xr = np.linspace(0.0, rpm_max, 80)
            f_camp = go.Figure()
            f_camp.add_vrect(x0=band.low, x1=band.high, fillcolor="rgba(245,158,11,0.13)", line_width=0)
            for o in (1.0, 2.0, 3.0, 4.0, 6.0):
                f_camp.add_trace(go.Scatter(x=xr, y=o * xr / 60, mode="lines", name=f"{o:g}×",
                                            line=dict(color="#94a3b8", width=1, dash="dot")))
            for i, fn in enumerate(naturals):
                f_camp.add_trace(go.Scatter(x=[0, rpm_max], y=[fn, fn], mode="lines",
                                            name=f"TNF{i+1} {fn:.1f} Hz", line=dict(color=GREEN, width=2.4)))
            f_camp.add_vline(x=rpm, line=dict(color=NAVY, width=2, dash="dash"))
            _scc = {"coincidence": RED, "near": AMBER, "clear": "#94a3b8"}
            for c in crossings:
                f_camp.add_trace(go.Scatter(x=[c.crossing_rpm], y=[c.mode_hz], mode="markers", showlegend=False,
                                            marker=dict(color=_scc[c.severity], size=12, symbol="x",
                                                        line=dict(width=2, color="#7f1d1d"))))
            _apply(f_camp, xlab="speed (RPM)", ylab="frequency (Hz)")
            f_fat = go.Figure()
            if ranges:
                rr = np.array([r for r, _ in ranges]); cc = np.array([c for _, c in ranges])
                nb = int(np.clip(len(rr), 8, 24)); edges = np.linspace(0, rr.max() * 1.0001, nb + 1)
                hist, _ = np.histogram(rr, bins=edges, weights=cc)
                f_fat.add_trace(go.Bar(x=0.5 * (edges[:-1] + edges[1:]), y=hist,
                                       width=(edges[1] - edges[0]) * 0.92, marker_color=NAVY))
                _apply(f_fat, xlab=f"torque range ({u})", ylab="cycle count")

            _amax = max(oa[float(k)][0] for k in range(1, 6)) or 1.0
            _lvl = (lambda a: ("dominante" if a >= 0.5 * _amax else "presente" if a >= 0.1 * _amax else "traza")) if _es \
                else (lambda a: ("dominant" if a >= 0.5 * _amax else "present" if a >= 0.1 * _amax else "trace"))
            order_rows = [[f"{o}×", f"{o*rpm/60:.2f} Hz", f"{oa[float(o)][0]:.2f} {u}",
                           f"{oa[float(o)][1]:+.1f}°", _lvl(oa[float(o)][0])] for o in range(1, 6)]
            _stx = ({"coincidence": "Coincidencia", "near": "Cercano", "clear": "Libre"} if _es
                    else {"coincidence": "Coincidence", "near": "Near", "clear": "Clear"})
            crossing_rows = [[c.mode_label, f"{c.mode_hz:.1f} Hz", f"{c.order:g}×", f"{c.crossing_rpm:.0f} rpm",
                              f"{c.sep_margin_pct:.0f} %", _stx[c.severity]] for c in crossings]

            meta = {"report_title": ("ANÁLISIS DE VIBRACIÓN TORSIONAL" if _es else "TORSIONAL VIBRATION ANALYSIS"),
                    "asset": r_asset, "client": r_client, "location": r_location,
                    "prepared_by": r_prep, "prepared_role": r_prep_role, "prepared_city": r_city,
                    "reviewed_by": r_rev, "reviewed_role": r_rev_role,
                    "consecutive": r_consec, "report_date": r_date, "date": r_date,
                    "format_code": "SIGA-FMT-180", "format_version": "1"}
            ctx = {"name": r_asset or run.get("name", "run"), "units_label": u, "rpm": rpm, "mean": m.mean,
                   "pp": m.peak_to_peak, "ripple": _rip, "rms": m.rms, "fs": fs, "dominant": f"{dom}×"}
            findings_list = [x.strip() for x in find_txt.splitlines() if x.strip()]
            recs_list = [x.strip() for x in rec_txt.splitlines() if x.strip()]
            pdf = build_torsional_pdf(
                meta=meta, context=ctx, findings=findings_list, recommendations=recs_list,
                waveform_png=plotly_to_png(f_wave), spectrum_png=plotly_to_png(f_spec),
                campbell_png=plotly_to_png(f_camp), fatigue_png=plotly_to_png(f_fat) if ranges else None,
                order_rows=order_rows, crossing_rows=crossing_rows, naturals=naturals,
                lang=("es" if _es else "en"))
            st.session_state["_tors_pdf"] = pdf
            st.session_state["_tors_pdf_name"] = (r_consec or "watermelon_torsional_report")
        st.success("Reporte SIGA listo." if _es else "SIGA report ready.")

    if st.session_state.get("_tors_pdf"):
        st.download_button(("⬇ Descargar reporte SIGA (PDF)" if _es else "⬇ Download SIGA report (PDF)"),
                           st.session_state["_tors_pdf"],
                           file_name=f"{st.session_state.get('_tors_pdf_name','watermelon_torsional_report')}.pdf",
                           mime="application/pdf", type="primary")
