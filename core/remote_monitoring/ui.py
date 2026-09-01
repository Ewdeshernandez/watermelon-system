"""
core/remote_monitoring/ui.py — Modo "Remote Monitoring" (render Streamlit)
=========================================================================

Página propia del sidebar (pages/02_Remote_Monitoring.py), justo debajo
de Live Monitoring. Independiente de Live Monitoring (que sigue mostrando
los escalares Modbus del Bently). Acá se hace adquisición dinámica en vivo
(NI / simulado) y se dibuja rotordinámica.

Dos tabs:
  ⚙️ Setup   — config amigable de máquina + canales (ui_setup.render_setup).
  📡 Monitor — adquisición en vivo + gráficos, usando los canales guardados
               en Setup. Estado estacionario: waveform/spectrum/órbita/1X/
               tendencia. Transitorio (arranque/parada): bode/cascade.

Patrón de refresco: SÍNCRONO. El AcqAgent vive en st.session_state; en cada
rerun bombeamos unos bloques (agent.pump) y redibujamos. "Live" hace
st.rerun() en loop. Nada de hilos en la UI — los hilos son para el Agent
headless de sitio (core/remote_monitoring/agent.py start/stop).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

import numpy as np
import streamlit as st

from core.modal.acq_backend import ChannelConfig
from core.remote_monitoring.agent import AcqAgent
from core.remote_monitoring.stream_source import (
    StreamConfig,
    SimulatedStreamSource,
    is_keyphasor_channel,
)
from core.remote_monitoring.keyphasor import one_x_vector
from core.remote_monitoring.transient import TransientCapture, TransientConfig
from core.remote_monitoring import states as rm_states


# =====================================================================
# Punto de entrada — llamado por pages/02_Remote_Monitoring.py
# =====================================================================
def render_remote_monitoring() -> None:
    """Renderiza la página completa Remote Monitoring (Setup + Monitor)."""
    try:
        from core.ui_theme import page_header
        page_header(title="Remote Monitoring",
                    subtitle="Live dynamic acquisition · rotordynamics · ISO 20816 / API 670 / API 684")
    except Exception:  # noqa: BLE001
        st.title("Remote Monitoring")

    # Pestañas con bolitas de color (patrón de la casa — Calibración/Reports),
    # scopeadas con .st-key-rm_view para NO afectar los otros radios del Setup.
    st.markdown("""
        <style>
        .st-key-rm_view label > div:first-of-type { display:none !important; }
        .st-key-rm_view label {
            font-size:16px; font-weight:600; padding:6px 20px 9px 20px; color:#64748b;
        }
        .st-key-rm_view label:hover { color:#0F1E3D; }
        .st-key-rm_view label:has(input:checked) {
            color:#0F1E3D; box-shadow:inset 0 -3px 0 #1AAEE5;
        }
        .st-key-rm_view label::before {
            content:"●"; font-size:19px; margin-right:10px; vertical-align:-2px;
        }
        .st-key-rm_view label:nth-of-type(1)::before { color:#1AAEE5; }
        .st-key-rm_view label:nth-of-type(2)::before { color:#16A34A; }
        .st-key-rm_view label:nth-of-type(3)::before { color:#8B5CF6; }

        /* Sub-pestañas (Análisis) con bolitas de color — sin emojis */
        .stTabs [data-baseweb="tab-list"] [data-baseweb="tab"]::before {
            content:"●"; font-size:16px; margin-right:8px; vertical-align:-1px;
        }
        .stTabs [data-baseweb="tab-list"] [data-baseweb="tab"]:nth-child(1)::before { color:#1AAEE5; }
        .stTabs [data-baseweb="tab-list"] [data-baseweb="tab"]:nth-child(2)::before { color:#16A34A; }
        .stTabs [data-baseweb="tab-list"] [data-baseweb="tab"]:nth-child(3)::before { color:#8B5CF6; }
        .stTabs [data-baseweb="tab-list"] [data-baseweb="tab"]:nth-child(4)::before { color:#D89B22; }
        .stTabs [data-baseweb="tab-list"] [data-baseweb="tab"]:nth-child(5)::before { color:#EF4444; }
        .stTabs [data-baseweb="tab-list"] [data-baseweb="tab"]:nth-child(6)::before { color:#06B6D4; }
        .stTabs [data-baseweb="tab-list"] [data-baseweb="tab"]:nth-child(7)::before { color:#EC4899; }
        .stTabs [data-baseweb="tab-list"] [data-baseweb="tab"]:nth-child(8)::before { color:#2563EB; }
        .stTabs [data-baseweb="tab-list"] [data-baseweb="tab"]:nth-child(9)::before { color:#F97316; }
        .stTabs [data-baseweb="tab-list"] [data-baseweb="tab"]:nth-child(10)::before { color:#64748B; }

        /* Controles minimalistas (menos "cuadros feos"): planos, borde tenue */
        [data-baseweb="select"] > div,
        .stNumberInput div[data-baseweb="input"],
        .stTextInput div[data-baseweb="input"] {
            background:#fff !important; border-color:#e2e8f2 !important;
            border-radius:9px !important; min-height:38px !important;
        }
        [data-baseweb="select"] > div:hover,
        .stNumberInput div[data-baseweb="input"]:hover { border-color:#c3cede !important; }
        /* Radio horizontal más aireado, sin cajas */
        .stRadio [role="radiogroup"] { gap:6px 18px !important; }

        /* Controles de rango de la tendencia: chicos y alineados */
        .st-key-rm_trend_ctrls { margin-top:-4px; }
        :is(.st-key-rm_trend_ctrls,.st-key-rm_casc_ctrls,.st-key-rm_wf3_ctrls) [role="radiogroup"] { gap:2px 12px !important; align-items:center; }
        :is(.st-key-rm_trend_ctrls,.st-key-rm_casc_ctrls,.st-key-rm_wf3_ctrls) [role="radiogroup"] label p { font-size:12px !important; }
        :is(.st-key-rm_trend_ctrls,.st-key-rm_casc_ctrls,.st-key-rm_wf3_ctrls) [role="radiogroup"] label { padding:1px 0 !important; }
        .st-key-rm_casc_ctrls, .st-key-rm_wf3_ctrls { margin-top:-4px; }
        /* Multiselección de canales: compacta, en la misma línea */
        .st-key-rm_trend_ctrls [data-testid="stMultiSelect"] { min-width:220px !important; }
        .st-key-rm_trend_ctrls [data-testid="stMultiSelect"] div[data-baseweb="select"] > div {
            min-height:30px !important; border-radius:8px !important; }
        .st-key-rm_trend_ctrls [data-testid="stMultiSelect"] [data-baseweb="tag"] {
            height:20px !important; font-size:11px !important; }
        /* Cajita de cantidad: el borde vive en stNumberInputContainer → lo quito ahí.
           Sin borde externo ni rojo; solo sombreado al pasar/enfocar. */
        .st-key-rm_trend_ctrls [data-testid="stNumberInput"] { width:64px !important; }
        .st-key-rm_trend_ctrls [data-testid="stNumberInput"] button { display:none !important; }
        .st-key-rm_trend_ctrls [data-testid="stNumberInputContainer"] {
            border:none !important; box-shadow:none !important; outline:none !important;
            background:transparent !important; min-height:30px !important; border-radius:8px !important; }
        .st-key-rm_trend_ctrls [data-testid="stNumberInputContainer"]:hover,
        .st-key-rm_trend_ctrls [data-testid="stNumberInputContainer"]:focus-within {
            background:#e9f0fb !important; }
        .st-key-rm_trend_ctrls [data-testid="stNumberInput"] input {
            background:transparent !important; text-align:center;
            font-size:12px !important; padding:2px 6px !important; }

        /* Controles de la órbita: par + filtro + vueltas, chicos y alineados */
        .st-key-rm_orbit_ctrls { margin-top:-4px; }
        .st-key-rm_orbit_ctrls [role="radiogroup"] { gap:2px 12px !important; align-items:center; }
        .st-key-rm_orbit_ctrls [role="radiogroup"] label p { font-size:12px !important; }
        .st-key-rm_orbit_ctrls [role="radiogroup"] label { padding:1px 0 !important; }
        .st-key-rm_orbit_ctrls [data-testid="stSelectbox"] { min-width:150px !important; }
        .st-key-rm_orbit_ctrls [data-testid="stSelectbox"] div[data-baseweb="select"] > div {
            min-height:30px !important; border-radius:8px !important; font-size:12px !important; }
        .st-key-rm_orbit_ctrls [data-testid="stNumberInput"] { width:66px !important; }
        .st-key-rm_orbit_ctrls [data-testid="stNumberInput"] button { display:none !important; }
        .st-key-rm_orbit_ctrls [data-testid="stNumberInputContainer"] {
            border:none !important; box-shadow:none !important; outline:none !important;
            background:transparent !important; min-height:30px !important; border-radius:8px !important; }
        .st-key-rm_orbit_ctrls [data-testid="stNumberInputContainer"]:hover,
        .st-key-rm_orbit_ctrls [data-testid="stNumberInputContainer"]:focus-within {
            background:#e9f0fb !important; }
        .st-key-rm_orbit_ctrls [data-testid="stNumberInput"] input {
            background:transparent !important; text-align:center;
            font-size:12px !important; padding:2px 6px !important; }

        /* Controles del Bode: panel de instrumento — toggle + campos de rpm */
        :is(.st-key-rm_bode_ctrls,.st-key-rm_polar_ctrls) {
            margin:2px 0 8px; padding:8px 14px; background:#f7f9fd;
            border:1px solid #e3e9f2; border-radius:10px; align-items:center !important; }
        :is(.st-key-rm_bode_ctrls,.st-key-rm_polar_ctrls) [data-testid="stNumberInput"] { width:158px !important; }
        :is(.st-key-rm_bode_ctrls,.st-key-rm_polar_ctrls) [data-testid="stNumberInput"] label p {
            font-size:9.5px !important; text-transform:uppercase; letter-spacing:.05em;
            font-weight:700 !important; color:#8a97ab !important; margin-bottom:1px !important; }
        :is(.st-key-rm_bode_ctrls,.st-key-rm_polar_ctrls) [data-testid="stNumberInput"] button { display:none !important; }
        :is(.st-key-rm_bode_ctrls,.st-key-rm_polar_ctrls) [data-testid="stNumberInputContainer"] {
            border:1px solid #d6deea !important; box-shadow:none !important; outline:none !important;
            background:#ffffff !important; min-height:32px !important; border-radius:8px !important;
            transition:border-color .12s, background .12s; }
        :is(.st-key-rm_bode_ctrls,.st-key-rm_polar_ctrls) [data-testid="stNumberInputContainer"]:hover,
        :is(.st-key-rm_bode_ctrls,.st-key-rm_polar_ctrls) [data-testid="stNumberInputContainer"]:focus-within {
            border-color:#2f6fb0 !important; background:#f4f8ff !important; }
        :is(.st-key-rm_bode_ctrls,.st-key-rm_polar_ctrls) [data-testid="stNumberInput"] input {
            background:transparent !important; text-align:center; color:#0F1E3D !important;
            font-family:ui-monospace,monospace !important; font-weight:600 !important;
            font-size:13px !important; padding:3px 8px !important; }
        :is(.st-key-rm_bode_ctrls,.st-key-rm_polar_ctrls) [data-testid="stWidgetLabel"] p { font-size:12px !important; }
        </style>
    """, unsafe_allow_html=True)

    view = st.radio("View", ["Configuration", "Monitor", "Analysis"], horizontal=True,
                    key="rm_view", label_visibility="collapsed")
    # Hairline sutil (sin el st.divider() que gasta mucho espacio vertical).
    st.markdown('<hr style="margin:2px 0 10px;border:none;border-top:1px solid #e6ecf5">',
                unsafe_allow_html=True)
    if view == "Configuration":
        from core.remote_monitoring.ui_setup import render_setup
        render_setup()
    elif view == "Monitor":
        _render_monitoreo()
    else:
        _render_analisis()


# =====================================================================
# Helpers de construcción
# =====================================================================
def _demo_channels(n_bearings: int = 4) -> List[ChannelConfig]:
    from core.remote_monitoring.config import MachineConfig, auto_layout, setup_to_channel_configs, AcqSetup
    m = MachineConfig(n_bearings=n_bearings)
    return setup_to_channel_configs(AcqSetup(machine=m, channels=auto_layout(m)))


def _channels_fingerprint(channels: List[ChannelConfig]) -> str:
    return "|".join(f"{c.name}:{c.bnc_port}:{c.coupling}:{c.sensitivity_mv_per_eu}" for c in channels)


def _build_agent(channels: List[ChannelConfig], source_kind: str, fs: float,
                 sim: dict, instance_id: str) -> AcqAgent:
    cfg = StreamConfig(
        sample_rate_hz=fs, channels=channels,
        block_seconds=0.25, buffer_seconds=8.0,
        rpm=sim.get("rpm", 3600.0), defect=sim.get("defect", "none"),
        speed_profile=sim.get("speed_profile", "constant"),
        rpm_start=sim.get("rpm_start", 0.0), rpm_end=sim.get("rpm_end", 0.0),
        ramp_seconds=sim.get("ramp_seconds", 30.0),
        sim_critical_rpm=sim.get("sim_critical_rpm", 0.0),
        sim_critical_rpm2=sim.get("sim_critical_rpm2", 0.0),
    )
    if source_kind == "Field (plant)":
        from core.remote_monitoring.ni_stream_source import NIStreamSource
        source = NIStreamSource(cfg)
    else:
        source = SimulatedStreamSource(cfg)
    return AcqAgent(source, instance_id=instance_id)


# =====================================================================
# Tab Monitor — adquisición + gráficos
# =====================================================================
def _no_config_gate() -> bool:
    """True si NO hay config activa (muestra ayuda + demo). Comparte Monitoreo/Análisis."""
    if st.session_state.get("rm_channels"):
        return False
    st.info("No active configuration. Go to **Configuration** and save a machine, "
            "or load a demo layout to try it now.")
    if st.button("Load demo layout (4 bearings + keyphasor)"):
        chans = _demo_channels(4)
        st.session_state["rm_channels"] = chans
        st.session_state["rm_machine_rpm"] = 3600.0
        st.session_state["rm_machine_name"] = "Demo"
        st.session_state["rm_machine_rotation"] = "CCW"
        # Valores de ejemplo (proximidad, mil pp / V) para que Gap, Alarma,
        # Danger y las líneas de tendencia se vean sin configurar nada.
        vibs = [c.name for c in chans if not is_keyphasor_channel(c)]
        st.session_state["rm_alarms_by_name"] = {n: (2.5, 4.0) for n in vibs}
        st.session_state["rm_gap_by_name"] = {n: -9.5 for n in vibs}
        st.session_state["rm_type_by_name"] = {n: "proximity" for n in vibs}
        # Ángulos reales de sonda (Y=45°L→315°, X=45°R→45°) + pares X/Y por cojinete
        st.session_state["rm_angle_by_name"] = {
            n: (315.0 if "Y" in n.upper() else 45.0) for n in vibs}
        _brgs = sorted({int("".join(ch for ch in n if ch.isdigit()) or 0) for n in vibs})
        st.session_state["rm_pairs_saved"] = [[f"{b}Y", f"{b}X"] for b in _brgs
                                              if f"{b}Y" in vibs and f"{b}X" in vibs]
        # Params de adquisición (proximidad) → Fmax/Fmin acotan el espectro
        from dataclasses import asdict as _asdict
        from core.remote_monitoring.config import default_acq_for_type
        _pp = _asdict(default_acq_for_type("proximity"))
        st.session_state["rm_acq_saved"] = _pp
        st.session_state["rm_acq_by_type_saved"] = {"proximity": _pp}
        st.rerun()
    return True


def _is_sim(agent) -> bool:
    return type(agent.source).__name__ == "SimulatedStreamSource"


def _acq(agent, n_blocks: int = 4) -> None:
    """Alimenta la adquisición sin bloquear la UI.
    · Simulado → bombea n bloques (genera a demanda).
    · Campo / NI real → arranca (idempotente) el HILO de fondo que lee el stream
      en continuo, así el driver NI nunca se atrasa ni congela la pantalla."""
    if _is_sim(agent):
        agent.pump(n_blocks)
    else:
        agent.start()          # idempotente: no re-arranca si ya corre


def _acq_stop(agent) -> None:
    """Detiene la adquisición (hilo + tarea NI en campo)."""
    try:
        agent.stop()
    except Exception:  # noqa: BLE001
        pass


def _ensure_agent() -> Optional[AcqAgent]:
    """Construye/devuelve el agente desde la config activa + fuente/params
    guardados en session. Compartido por Monitoreo y Análisis."""
    channels = st.session_state.get("rm_channels")
    if not channels:
        return None
    default_rpm = float(st.session_state.get("rm_machine_rpm", 3600.0))
    source_kind = st.session_state.get("rm_source_kind", "Simulado (dev/Mac)")
    fs = int(st.session_state.get("rm_fs", 25600))
    sim = st.session_state.get("rm_sim") or {"rpm": default_rpm, "defect": "none",
                                             "speed_profile": "constant"}
    sig = f"{source_kind}|{fs}|{sim}|{_channels_fingerprint(channels)}"
    if st.session_state.get("rm_agent_sig") != sig:
        old = st.session_state.get("rm_agent")
        if old is not None:
            try:
                old.stop()
            except Exception:  # noqa: BLE001
                pass
        st.session_state["rm_agent"] = _build_agent(
            channels, source_kind, float(fs), sim, st.session_state.get("rm_machine_name", "adhoc"))
        st.session_state["rm_agent_sig"] = sig
        st.session_state.setdefault("rm_running", False)
        st.session_state["rm_trend"] = []
        _bt = st.session_state.get("rm_acq_by_type_saved") or {}
        _fmax = (max((float(v.get("fmax_hz", 1000)) for v in _bt.values()), default=1000.0) if _bt
                 else float((st.session_state.get("rm_acq_saved") or {}).get("fmax_hz", 1000.0)))
        # Ventana de captura ~0.5 s (más muestras/vuelta a mayor fs → order
        # tracking más fino), acotada a potencia razonable.
        _fs = int(st.session_state.get("rm_fs", 25600))
        _cap = int(min(16384, max(4096, round(0.5 * _fs / 1024) * 1024)))
        st.session_state["rm_transient"] = TransientCapture(
            TransientConfig(fmax_hz=_fmax, capture_samples=_cap))
        st.session_state["rm_prev_rpm"] = None
    return st.session_state["rm_agent"]


def _acquire(agent: AcqAgent, pump_n: int):
    """Bombea (si pump_n>0), toma snapshot, estima rpm/estado, alimenta transitorio.
    Devuelve (snap, rpm, state, tc, err, vib_channels)."""
    err = None
    try:
        if pump_n:
            _acq(agent, pump_n)
    except ImportError as e:
        err = str(e)
        st.session_state["rm_running"] = False
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"
        st.session_state["rm_running"] = False
    snap = agent.snapshot()
    vib = [(i, ch) for i, ch in enumerate(agent.channels) if not is_keyphasor_channel(ch)]
    rpm = agent.estimate_rpm(snap) if snap.shape[1] else None
    state = rm_states.classify_state(rpm, st.session_state.get("rm_prev_rpm"))
    st.session_state["rm_prev_rpm"] = rpm
    tc = st.session_state.setdefault("rm_transient", TransientCapture())
    if rpm and snap.shape[1]:
        tc.feed(snap, rpm, agent.sample_rate_hz, vib,
                kph_idx=agent.source.config.keyphasor_index())
    return snap, rpm, state, tc, err, vib


def _render_source_params() -> None:
    """Fuente y parámetros (solo en Monitoreo). Guarda en session."""
    default_rpm = float(st.session_state.get("rm_machine_rpm", 3600.0))
    with st.expander("⚙️ Source and parameters", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            source_kind = st.selectbox("Data source", ["Simulated", "Field (plant)"],
                                       key="rm_src_kind",
                                       help="Campo = on-site acquisition. Simulado = testing.")
        with col2:
            fs = st.select_slider("Sample rate (Hz)", options=[2560, 5120, 10240, 25600, 51200],
                                  value=int(st.session_state.get("rm_fs", 25600)), key="rm_fs_w")
        with col3:
            defect = (st.selectbox("Simulated defect",
                                   ["none", "unbalance", "misalignment", "oil_whirl"], key="rm_defect",
                                   help="oil_whirl: injects fluid-film instability — whirl "
                                        "(~0.45X) that locks in (whip) past 2× the critical.")
                      if source_kind.startswith("Simulated") else "none")
        sim = {"rpm": default_rpm, "defect": defect, "speed_profile": "constant"}
        if source_kind.startswith("Simulated"):
            st.markdown("**Machine mode** (defines the sampling)")
            _MODE_MAP = {"Steady": "constant", "Startup": "runup",
                         "Coastdown": "coastdown", "Startup + Coastdown": "runup_coastdown"}
            p1, p2, p3, p4 = st.columns(4)
            with p1:
                mode_es = st.selectbox("Mode", list(_MODE_MAP.keys()), key="rm_prof")
            prof = _MODE_MAP[mode_es]
            sim["speed_profile"] = prof
            if prof == "constant":
                with p2:
                    sim["rpm"] = st.number_input("RPM", 300, 30000, int(default_rpm), step=60, key="rm_simrpm")
            else:
                with p2:
                    sim["rpm_start"] = st.number_input("Start RPM", 0, 30000, 600, step=60, key="rm_rpmst")
                with p3:
                    sim["rpm_end"] = st.number_input("End RPM", 0, 30000, 6000, step=60, key="rm_rpmend")
                with p4:
                    sim["sim_critical_rpm"] = st.number_input("Critical 1 (rpm)", 0, 30000, 3000, step=60, key="rm_crit")
                cc1, cc2 = st.columns(2)
                with cc1:
                    sim["sim_critical_rpm2"] = st.number_input("Critical 2 (rpm, optional)", 0, 30000, 0,
                                                               step=60, key="rm_crit2",
                                                               help="0 = no second resonance.")
                with cc2:
                    sim["ramp_seconds"] = st.slider("Duration (s)", 5, 300, 90, key="rm_ramp",
                                                    help="Slower runup/coastdown = more points "
                                                         "in Bode/Polar. A real "
                                                         "coastdown is slow → very dense curve.")
            st.caption("**Steady**: constant speed (steady state, continuous sampling). "
                       "**Startup / Coastdown**: speed sweep → **transient** capture "
                       "(finer, by Δrpm) for bode / cascade / waterfall.")
    st.session_state["rm_source_kind"] = source_kind
    st.session_state["rm_fs"] = fs
    st.session_state["rm_sim"] = sim


# =====================================================================
# Vista MONITOREO — config activa + adquisición + tabular list (ADRE)
# =====================================================================
def _render_monitoreo() -> None:
    if _no_config_gate():
        return
    channels = st.session_state["rm_channels"]
    machine_name = st.session_state.get("rm_machine_name", "adhoc")
    n_vib = sum(1 for c in channels if not is_keyphasor_channel(c))
    st.caption(f"Active machine: **{machine_name}** · {len(channels)} channels "
               f"({n_vib} vibration) · source ready to acquire.")

    _render_source_params()
    agent = _ensure_agent()

    b1, b2, b3, b4, b5, b6 = st.columns([1, 1, 1.3, 1.3, 1, 1.6])
    with b1:
        if st.button("▶ Start", use_container_width=True):
            try:
                _acq(agent, 1)                  # campo: arranca el hilo de fondo
                st.session_state["rm_running"] = True
            except Exception as e:  # noqa: BLE001
                st.session_state["rm_running"] = False
                st.error(f"⚠ Could not start acquisition: {type(e).__name__}: {e}")
    with b2:
        if st.button("⏸ Stop", use_container_width=True):
            st.session_state["rm_running"] = False
            _acq_stop(agent)                    # campo: para el hilo + tarea NI
    with b3:
        capture = st.button("📸 Capture", use_container_width=True, type="primary",
                            help="Takes a fresh reading and saves it in a single click.")
    with b4:
        take = st.button("🔄 Take 1 reading", use_container_width=True,
                         help="Refreshes a reading without saving it.")
    with b5:
        save = st.button("💾 Save", use_container_width=True,
                         help="Saves the current window without taking new data.")
    with b6:
        live = st.checkbox("🟢 Live (auto-refresh)", value=st.session_state.get("rm_running", False))
        st.session_state["rm_running"] = live

    # --- Grabador de transitorio: onda cruda completa a disco (no se pierde nada) ---
    rec = st.session_state.get("rm_recorder")
    recording = bool(rec and getattr(rec, "open", False))
    rc1, rc2, rc3 = st.columns([1.4, 1.4, 4])
    with rc1:
        if not recording and st.button("⏺ Record transient", use_container_width=True,
                                       help="Records the full RAW WAVEFORM to disk throughout the "
                                            "whole ramp (startup/coastdown). Nothing is lost; it is "
                                            "then reprocessed to Bode/Cascade at maximum resolution."):
            from core.remote_monitoring.recorder import TransientRecorder, free_bytes
            _free = free_bytes()
            if _free and _free < 200 * 1024 * 1024:   # < 200 MB libres → no arranca
                st.error(f"⚠ Low disk space ({_free/1e6:.0f} MB free). Free up with "
                         f"**🗑 Clear local recordings** before recording.")
            else:
                try:
                    ch_meta = [{"name": c.name, "units": c.units, "coupling": c.coupling,
                                "bnc_port": c.bnc_port,
                                "sensitivity_mv_per_eu": float(c.sensitivity_mv_per_eu or 0.0)}
                               for c in agent.channels]
                    rec = TransientRecorder(agent.instance_id, agent.sample_rate_hz, ch_meta,
                                            machine=machine_name)
                    agent.on_block = rec.append
                    st.session_state["rm_recorder"] = rec
                    st.session_state["rm_running"] = True     # que fluyan bloques
                    _acq(agent, 1)                            # campo: arranca el hilo
                    st.rerun()
                except OSError as e:
                    st.error(f"⚠ Could not start recording (disk): {e}. "
                             f"Free up with **🗑 Clear local recordings**.")
    with rc2:
        if recording and st.button("⏹ Stop recording", use_container_width=True, type="primary"):
            from core.remote_monitoring.recorder import upload_recording
            rec.stop()
            agent.on_block = None
            st.session_state["rm_recorder"] = rec
            up = upload_recording(rec.dir)                 # intenta subir a Supabase
            if up.get("ok"):
                st.success(f"📼 Recording **{rec.rec_id}** ({rec.status.duration_s:.0f} s · "
                           f"{rec.status.size_mb:.1f} MB) saved and **☁ uploaded to Supabase**. "
                           f"Reprocess it in **Análisis → 📼 Reprocess**.")
            else:
                st.warning(f"📼 Recording **{rec.rec_id}** saved locally ({rec.status.size_mb:.1f} MB). "
                           f"**Pending upload** ({up.get('reason', 'no connection')}). It uploads "
                           f"with **☁ Upload pending** when internet is available.")
            st.rerun()
    with rc3:
        if recording:
            agent.on_block = rec.append   # re-asegura el hook por si el agente se reusó
            s = rec.status
            st.markdown(f"<div style='padding:6px 10px'>🔴 <b>RECORDING</b> · "
                        f"{s.duration_s:.0f} s · {s.blocks} blocks · {s.size_mb:.1f} MB · "
                        f"raw to disk</div>", unsafe_allow_html=True)
        else:
            from core.remote_monitoring.recorder import pending_count, sync_pending
            _pend = pending_count(agent.instance_id)
            if _pend and st.button(f"☁ Upload pending ({_pend})", use_container_width=True,
                                   help="Uploads to Supabase the recordings that stayed local (field with no internet)."):
                ok, fail = sync_pending(agent.instance_id)
                (st.success if not fail else st.warning)(f"☁ Uploaded {ok} · failed {fail}.")
                st.rerun()

    # Aviso si la grabación se detuvo por disco lleno + gestión de espacio.
    if rec is not None and getattr(rec, "error", None):
        st.error(f"⚠ Recording stopped: **{rec.error}**. Free up disk below and record again.")
    from core.remote_monitoring.recorder import local_usage, free_bytes, clear_recordings
    _cnt, _used = local_usage(agent.instance_id)
    if _cnt:
        _free = free_bytes()
        d1, d2 = st.columns([2.3, 1.4])
        with d1:
            st.caption(f"💽 Local recordings: **{_cnt}** · {_used/1e6:.0f} MB used · "
                       f"{_free/1e6:.0f} MB free on disk.")
        with d2:
            if st.button(f"🗑 Clear local recordings ({_cnt})", use_container_width=True,
                         help="Deletes the recordings from local disk to free up space. Those already "
                              "uploaded to Supabase stay in the cloud."):
                n, freed = clear_recordings(agent.instance_id)
                st.session_state.pop("rm_recorder", None)
                st.success(f"🗑 Deleted {n} recording(s) · freed {freed/1e6:.0f} MB.")
                st.rerun()

    # Acciones one-shot en el run principal (fuera del fragment).
    if capture:
        try:
            _acq(agent, 8)
            _s = agent.snapshot()
            if _s.shape[1]:
                _save_snapshot(agent, _s, agent.estimate_rpm(_s))
            else:
                st.warning("No data to capture. Press ▶ Start first.")
        except Exception as e:  # noqa: BLE001
            st.session_state["rm_running"] = False
            st.error(f"⚠ Could not capture: {type(e).__name__}: {e}")
    if take:
        try:
            _acq(agent, 8)
        except Exception as e:  # noqa: BLE001
            st.session_state["rm_running"] = False
            st.error(f"⚠ Could not acquire: {type(e).__name__}: {e}")
    if save:
        _s = agent.snapshot()
        if _s.shape[1]:
            _save_snapshot(agent, _s, agent.estimate_rpm(_s))

    # Auto-refresh SIN elementos fantasma: fragment con run_every en vez del
    # loop time.sleep()+st.rerun() (que dejaba residuos al cambiar de vista).
    st.fragment(_monitoreo_display, run_every=(0.5 if live else None))()


def _monitoreo_display() -> None:
    agent = st.session_state.get("rm_agent")
    if agent is None:
        return
    live = st.session_state.get("rm_running", False)
    _rec = st.session_state.get("rm_recorder")
    if _rec is not None and getattr(_rec, "open", False):
        agent.on_block = _rec.append          # el pump de acá persiste cada bloque
    if live:
        try:
            _acq(agent, 4)
        except Exception as e:  # noqa: BLE001
            st.session_state["rm_running"] = False
            st.error(f"⚠ {type(e).__name__}: {e}")
            return
    if _rec is not None and getattr(_rec, "open", False):
        s = _rec.status
        st.markdown(f"<div style='padding:4px 10px;background:#fdecec;border:1px solid #f5c2c2;"
                    f"border-radius:8px;color:#b91c1c;font-size:12px'>🔴 <b>RECORDING transient</b> · "
                    f"{s.duration_s:.0f} s · {s.blocks} blocks · {s.size_mb:.1f} MB · raw waveform to disk"
                    f"</div>", unsafe_allow_html=True)
    snap = agent.snapshot()
    if snap.shape[1] == 0:
        st.info("No data yet. Press **▶ Start** or **Take 1 reading**.")
        return
    rpm = agent.estimate_rpm(snap)
    vib = [(i, ch) for i, ch in enumerate(agent.channels) if not is_keyphasor_channel(ch)]
    state = rm_states.classify_state(rpm, st.session_state.get("rm_prev_rpm"))
    st.session_state["rm_prev_rpm"] = rpm
    tc = st.session_state.setdefault("rm_transient", TransientCapture())
    if rpm:
        tc.feed(snap, rpm, agent.sample_rate_hz, vib,
                kph_idx=agent.source.config.keyphasor_index())
    _render_stat_strip(agent, snap, rpm, state, tc)
    st.markdown("##### Tabular list — current values")
    _render_tabular_list(agent, snap, rpm, vib)


# =====================================================================
# Vista ANÁLISIS — todos los gráficos en orden
# =====================================================================
def _render_analisis() -> None:
    if _no_config_gate():
        return
    agent = _ensure_agent()
    top = st.columns([0.9, 0.7, 4])
    with top[0]:
        take = st.button("🔄 Refresh", use_container_width=True,
                         help="Refreshes a reading (useful with Live off).")
    with top[1]:
        live = st.checkbox("🟢 Live", value=st.session_state.get("rm_running", False),
                           help="Auto-refresh of the plots.")
        st.session_state["rm_running"] = live
    if take:
        try:
            _acq(agent, 8)
        except Exception:  # noqa: BLE001
            pass
    _render_new_recordings_alert(agent)
    _render_reprocess(agent)
    st.fragment(_analisis_display, run_every=(0.5 if live else None))()


def _render_new_recordings_alert(agent: AcqAgent) -> None:
    """Aviso al especialista: hay nueva grabación (en Supabase) para analizar."""
    from core.remote_monitoring.recorder import cloud_recordings
    recs = cloud_recordings(agent.instance_id)
    if not recs:
        return
    seen = st.session_state.setdefault("rm_seen_recs", set())
    fresh = [r for r in recs if r.get("rec_id") not in seen]
    if not fresh:
        return
    latest = fresh[0]
    st.markdown(
        f"<div style='padding:11px 16px;margin:2px 0 10px;background:#fff7ed;border:1px solid #fdba74;"
        f"border-radius:10px;color:#9a3412;font-size:13.5px'>🔔 <b>{len(fresh)} new recording(s) "
        f"to analyze</b> — latest <b>{latest.get('rec_id')}</b> "
        f"({(latest.get('duration_s') or 0):.0f} s, {latest.get('machine','')}). Reprocess it below 👇"
        f"</div>", unsafe_allow_html=True)
    if st.button("Mark as seen", key="rm_seen_btn"):
        seen.update(r.get("rec_id") for r in recs)
        st.rerun()


def _render_reprocess(agent: AcqAgent) -> None:
    """Reprocesa una GRABACIÓN (incluida la del colector headless) a TODOS los
    gráficos. Auto-contenido: usa los canales/sensibilidades de la propia
    grabación, sin depender de la config viva."""
    import types
    from core.remote_monitoring.recorder import (list_all_recordings, load_recording,
                                                 cloud_recordings_all, download_recording)
    recs = list_all_recordings()
    # Suma las grabaciones que están SOLO en la nube (subidas por el equipo de
    # campo Windows) para poder verlas desde la Mac — de CUALQUIER máquina, así
    # aparecen aunque la web esté en otra máquina. Se descargan al reprocesar.
    _local_ids = {(m.get("_instance", ""), m.get("rec_id")) for m in recs}
    for cr in cloud_recordings_all(limit=60):
        if (cr.get("instance_id", ""), cr.get("rec_id")) in _local_ids:
            continue
        recs.append({"rec_id": cr.get("rec_id"), "_instance": cr.get("instance_id", ""),
                     "started": cr.get("started", 0), "duration_s": cr.get("duration_s"),
                     "n_channels": cr.get("n_channels", 0), "fs": cr.get("fs", 1),
                     "machine": cr.get("machine", ""), "_cloud": True})
    if not recs:
        return
    with st.expander(f"📼 Reprocess recording ({len(recs)} available)"):
        def _lbl(m):
            import datetime as _dt
            ts = m.get("started", 0)
            try:
                t = _dt.datetime.fromtimestamp(float(ts)).strftime("%d %b %H:%M")
            except Exception:  # noqa: BLE001
                t = str(ts)[:16]
            dur = m.get("duration_s") or (m.get("samples", 0) / (m.get("fs", 1) or 1))
            tag = "☁ " if m.get("_cloud") else ""
            return (f"{tag}{m.get('_instance','')} · {m['rec_id']} · {t} · {dur or 0:.0f} s · "
                    f"{m.get('n_channels', 0)} channels")
        labels = [_lbl(m) for m in recs]
        sel = st.selectbox("Recording", labels, key="rm_reproc_sel")
        c1, c2 = st.columns([1, 3])
        with c1:
            go_rep = st.button("⚙ Reprocess", use_container_width=True, type="primary")
        with c2:
            st.caption("Rebuilds waveform/spectrum/orbit/tabular (with cursor) + Bode/Cascade/Polar "
                       "of the whole run, from the raw record. Independent of the live feed.")
        if go_rep:
            m = recs[labels.index(sel)]
            if m.get("_cloud") and not m.get("_dir"):
                with st.spinner("Downloading recording from the cloud…"):
                    d = download_recording(m.get("_instance", ""), m["rec_id"])
                if not d:
                    st.error("Could not download from the cloud (check credentials/internet).")
                    return
                m["_dir"] = d
            with st.spinner("Reprocessing full record…"):
                manifest, full = load_recording(m["_dir"])
                # Canales AUTO-CONTENIDOS desde la grabación (nombre, sensib, unidad).
                ch_objs = []
                for cm in manifest.get("channels", []):
                    ch_objs.append(types.SimpleNamespace(
                        name=cm.get("name", "ch"), units=cm.get("units", "g rms"),
                        sensitivity_mv_per_eu=float(cm.get("sensitivity_mv_per_eu") or 100.0),
                        coupling=cm.get("coupling", "IEPE"), bnc_port=int(cm.get("bnc_port", 1) or 1)))
                tmap = st.session_state.setdefault("rm_type_by_name", {})
                for o in ch_objs:
                    if not is_keyphasor_channel(o):
                        tmap[o.name] = ("accelerometer" if str(o.coupling).upper() == "IEPE"
                                        else "proximity")
                vib = [(i, o) for i, o in enumerate(ch_objs) if not is_keyphasor_channel(o)]
                kph = next((i for i, o in enumerate(ch_objs) if is_keyphasor_channel(o)), None)
                fsr = float(manifest.get("fs", 5120))
                tc = TransientCapture(st.session_state.get("rm_transient").config
                                      if st.session_state.get("rm_transient") else None)
                npts = tc.process_full(full, fsr, vib, kph_idx=kph, delta_rpm=8.0)
                st.session_state["rm_transient"] = tc
                st.session_state["rm_replay"] = {
                    "full": full, "fs": fsr, "dur": full.shape[1] / fsr,
                    "win": min(int(2.0 * fsr), full.shape[1]), "rec_id": m["rec_id"],
                    "vib": vib, "kph": kph, "channels": ch_objs}
                st.session_state["rm_running"] = False     # replay, no vivo
            st.success(f"✓ Reprocessed: **{npts} points** from {full.shape[1]/fsr:.0f} s of raw waveform. "
                       f"Below you have **ALL** the plots: move the **🎚 cursor** for waveform/spectrum/"
                       f"orbit/tabular at each instant, and Bode/Cascade/Polar of the whole run.")


def _analisis_display() -> None:
    agent = st.session_state.get("rm_agent")
    if agent is None:
        return
    live = st.session_state.get("rm_running", False)
    if live:
        try:
            _acq(agent, 4)
        except Exception:  # noqa: BLE001
            st.session_state["rm_running"] = False
    snap = agent.snapshot()
    replay = st.session_state.get("rm_replay")
    replay_mode = snap.shape[1] == 0 and replay is not None
    fs = agent.sample_rate_hz
    kph_idx = agent.source.config.keyphasor_index()
    rpm = None
    if replay_mode:
        # REPLAY de la grabación: cursor de tiempo → ventana → TODOS los gráficos
        # (onda, espectro, órbita, tabular) en ese instante; Bode/Cascada usan todo.
        full = replay["full"]
        fs = float(replay["fs"])
        kph_idx = replay.get("kph")            # keyphasor de la GRABACIÓN
        win = int(replay["win"])
        dur = float(replay["dur"])
        maxt = max(0.0, dur - win / fs)
        cc = st.columns([4, 1])
        with cc[0]:
            t = st.slider(f"🎚 Recording cursor · {replay.get('rec_id', '')} (s)",
                          0.0, round(maxt, 2), min(round(maxt, 2), 0.0),
                          step=(round(maxt / 200, 3) or 0.05), key="rm_replay_t")
        with cc[1]:
            if st.button("✕ Exit replay", use_container_width=True):
                st.session_state.pop("rm_replay", None)
                st.rerun()
        off = int(t * fs)
        snap = np.ascontiguousarray(full[:, off:off + win])
        st.info(f"📼 **Recording replay** · cursor {t:.1f} / {dur:.0f} s — move the cursor and see "
                f"**waveform / spectrum / orbit / tabular** at that instant; **Bode/Polar/Cascade/"
                f"Waterfall** use the whole recording.")
        if kph_idx is not None and snap.shape[1]:
            from core.remote_monitoring.keyphasor import detect_keyphasor
            rpm = detect_keyphasor(snap[kph_idx, -min(snap.shape[1], int(2 * fs)):], fs).rpm
    if snap.shape[1] == 0:
        st.info("No data. Go to **Monitor** and press **▶ Start** "
                "(or reprocess a recording above).")
        return
    if not replay_mode:
        rpm = agent.estimate_rpm(snap)
    vib = (replay["vib"] if replay_mode
           else [(i, ch) for i, ch in enumerate(agent.channels) if not is_keyphasor_channel(ch)])
    state = rm_states.classify_state(rpm, st.session_state.get("rm_prev_rpm"))
    st.session_state["rm_prev_rpm"] = rpm
    tc = st.session_state.setdefault("rm_transient", TransientCapture())
    if rpm and not replay_mode:      # en replay NO alimentar (tc ya viene del reproceso)
        tc.feed(snap, rpm, fs, vib, kph_idx=kph_idx)
    names = [ch.name for _, ch in vib]
    _vent = snap.shape[1] / fs
    # (El contexto RPM/estado/ventana vive en el header de cada gráfico,
    #  no en una tira global — para que cada gráfico sea autocontenido.)

    tabs = st.tabs(["Tabular", "Trends", "Waveforms", "Spectrum",
                    "Orbits", "Bode", "Polar", "Shaft Centerline",
                    "Cascade", "Waterfall"])
    with tabs[0]:
        _render_tabular_list(agent, snap, rpm, vib)
    with tabs[1]:
        if names:
            tmap = st.session_state.get("rm_type_by_name") or {}
            # Selección de canales + rango van en UNA fila abajo (dentro de _plot_trend).
            _plot_trend(snap, vib, tmap, rpm=rpm)
    with tabs[2]:
        if names:
            sels = st.multiselect("Channels", names, default=[names[0]], key="rm_wf_ch",
                                  help="Pick one or more channels to see their stacked waveforms.")
            if sels:
                chans = [(snap[vib[names.index(s)][0]], vib[names.index(s)][1]) for s in sels]
                _plot_waveform(chans, fs, rpm, rm_states.state_label(state),
                               rm_states.state_color(state), _vent)
            else:
                st.info("Pick at least one channel.")
    with tabs[3]:
        if names:
            sels = st.multiselect("Channels", names, default=[names[0]], key="rm_sp_ch",
                                  help="One or more channels — stacked spectra.")
            if sels:
                chans = [(snap[vib[names.index(s)][0]], vib[names.index(s)][1]) for s in sels]
                _plot_spectrum(chans, fs, rpm)
            else:
                st.info("Pick at least one channel.")
    with tabs[4]:
        _plot_orbit(snap, vib, fs, rpm)
    with tabs[5]:
        if names:
            sel = st.selectbox("Channel", names, key="rm_bode_ch")
            _plot_bode(tc, sel)
    with tabs[6]:
        if names:
            sel = st.selectbox("Channel", names, key="rm_polar_ch")
            _plot_polar(tc, sel, snap, vib, fs, rpm)
    with tabs[7]:
        _plot_shaft_centerline(snap, vib)
    with tabs[8]:
        if names:
            sel = st.selectbox("Channel", names, key="rm_casc_ch")
            _plot_cascade(tc, sel, rpm)
    with tabs[9]:
        if names:
            sel = st.selectbox("Channel", names, key="rm_wf3_ch")
            _plot_waterfall(tc, sel)


# =====================================================================
# Tabular list (current values — estilo ADRE)
# =====================================================================
def _render_tabular_list(agent: AcqAgent, snap: np.ndarray, rpm: Optional[float], vib) -> None:
    from core.remote_monitoring.ui_setup import NAVY, CYAN, GRAY_LIGHT
    orders = sorted((st.session_state.get("rm_acq_saved") or {}).get("orders") or [1.0, 2.0])
    alarms = st.session_state.get("rm_alarms_by_name") or {}
    gaps = st.session_state.get("rm_gap_by_name") or {}
    tmap = st.session_state.get("rm_type_by_name") or {}
    fs = agent.sample_rate_hz
    f1 = (rpm / 60.0) if rpm else None

    heads = ["Sensor", "Gap", "Overall"]
    if f1:
        for o in orders:
            heads += [f"{o:g}X", f"{o:g}X phase"]
    heads += ["Alarm", "Danger", "Status"]
    th = "".join(f'<th style="padding:9px 12px;text-align:left;font-size:11px;'
                 f'text-transform:uppercase;font-weight:700;color:{CYAN};white-space:nowrap;">{h}</th>'
                 for h in heads)

    body = []
    for k, (i, ch) in enumerate(vib):
        eu = snap[i] * 1000.0 / ch.sensitivity_mv_per_eu
        conv, _norm, k0, krms = _amp_conv(tmap.get(ch.name, "proximity"))
        u = _amp_unit(ch.units, conv)
        overall = float(np.sqrt(np.mean((eu - np.mean(eu)) ** 2))) * krms
        al, dg = alarms.get(ch.name, (0.0, 0.0))
        gap = gaps.get(ch.name, 0.0)
        if dg > 0 and overall >= dg:
            status, scol = "DANGER", "#dc2626"
        elif al > 0 and overall >= al:
            status, scol = "ALERT", "#D89B22"
        else:
            status, scol = "OK", "#16a34a"
        gap_txt = (f'<span style="font-family:monospace">{gap:.2f} V</span>'
                   if gap else '<span style="color:#94a3b8">—</span>')
        cells = [f'<b style="color:{NAVY}">{ch.name}</b>',
                 gap_txt,
                 f'<span style="font-family:monospace">{overall:.4g} {u}</span>']
        if f1:
            freqs_c, mag_c = _spectrum(eu, fs)
            for o in orders:
                a, ph = _order_amp_phase(freqs_c, mag_c, eu, fs, o * f1)
                a *= k0
                cells.append(f'<span style="font-family:monospace">{a:.3g}</span>')
                cells.append(f'<span style="font-family:monospace;color:#64748b">{ph:.0f}°</span>')
        al_txt = (f'<span style="font-family:monospace;color:#D89B22">{al:.3g} {u}</span>'
                  if al > 0 else '<span style="color:#94a3b8">—</span>')
        dg_txt = (f'<span style="font-family:monospace;color:#dc2626">{dg:.3g} {u}</span>'
                  if dg > 0 else '<span style="color:#94a3b8">—</span>')
        cells += [al_txt, dg_txt,
                  f'<span style="color:{scol};font-weight:800">{status}</span>']
        bg = "#ffffff" if k % 2 == 0 else GRAY_LIGHT
        tds = "".join(f'<td style="padding:10px 12px;font-size:13px;border-top:1px solid #e8edf5;'
                      f'white-space:nowrap">{c}</td>' for c in cells)
        body.append(f'<tr style="background:{bg}">{tds}</tr>')

    if not f1:
        st.caption("Without a keyphasor there are no 1X/2X vectors — only Overall. Enable the keyphasor in Configuration.")
    st.markdown(
        f'<div style="border:1px solid #d6deea;border-radius:12px;overflow-x:auto;'
        f'box-shadow:0 6px 18px rgba(15,30,61,.08)">'
        f'<table style="width:100%;border-collapse:collapse;min-width:560px">'
        f'<thead><tr style="background:{NAVY}">{th}</tr></thead>'
        f'<tbody>{"".join(body)}</tbody></table></div>', unsafe_allow_html=True)
    st.caption("Amplitudes by standard: **displacement in pp** (API 670 · ISO 7919), "
               "**velocity/acceleration in RMS** (ISO 20816). Overall and alarm/danger "
               "in the same convention as the sensor.")


# =====================================================================
# Widgets de gráfico
# =====================================================================
def _render_stat_strip(agent: AcqAgent, snap: np.ndarray, rpm: Optional[float],
                       state: str, tc: TransientCapture) -> None:
    """Tira compacta de estado + contadores (secundaria al tabular).
    Fuente pequeña, tooltips explicativos al pasar el mouse."""
    from core.remote_monitoring.ui_setup import NAVY, CYAN, GRAY_LIGHT
    fs = agent.sample_rate_hz
    filled = snap.shape[1]
    vent_s = filled / fs if fs else 0.0
    try:
        block = agent.source.config.block_samples
        block_s = agent.source.config.block_seconds
    except Exception:  # noqa: BLE001
        block, block_s = 0, 0.0
    total_samples = agent.blocks_read * block
    size_mb = len(agent.channels) * filled * 8 / 1e6
    saved = int(st.session_state.get("rm_saved_count", 0))
    color = rm_states.state_color(state)

    def cell(label, value, *, vcolor=NAVY, tip="") -> str:
        t = f' title="{tip}"' if tip else ""
        return (f'<div{t} style="display:flex;flex-direction:column;gap:1px;padding:0 14px;'
                f'border-left:2px solid #eef2f8;cursor:{"help" if tip else "default"}">'
                f'<span style="font-size:9.5px;text-transform:uppercase;letter-spacing:.05em;'
                f'font-weight:700;color:#8a97ab;white-space:nowrap">{label}</span>'
                f'<span style="font-size:15px;font-weight:800;color:{vcolor};'
                f'font-family:ui-monospace,monospace;line-height:1.15">{value}</span></div>')

    live = "corriendo" if fs else ""
    cells = [
        cell("RPM", f"{rpm:.0f}" if rpm else "—",
             tip="Speed estimated from the keyphasor."),
        cell("1X", f"{rpm/60:.1f} Hz" if rpm else "—",
             tip="Rotational frequency (RPM/60)."),
        cell("Status", rm_states.state_label(state), vcolor=color,
             tip="Steady / Startup / Coastdown depending on the RPM change."),
        cell("Window", f"{vent_s:.1f} s",
             tip=f"Rolling buffer: the last {vent_s:.0f} s of waveform are kept. "
                 f"Spectrum, orbit and waveforms are computed over this window."),
        cell("Samples", f"{total_samples:,}",
             tip=f"Total samples acquired since ▶ Start "
                 f"(blocks of {block_s:g} s at {fs:g} Hz)."),
        cell("Vectors", f"{tc.n_samples}",
             tip="Speed points captured for Bode/Cascade. Only grows during "
                 "a transient (startup/coastdown); at steady state it stays at 1."),
        cell("Saved", f"{saved}",
             tip="Waveforms saved to disk with the 💾 Save button."),
        cell("Size", f"{size_mb:.2f} MB",
             tip="Memory used by the current window in RAM (channels × samples × 8 bytes)."),
    ]
    st.markdown(
        f'<div style="display:flex;flex-wrap:wrap;align-items:stretch;gap:8px 0;'
        f'padding:10px 4px;margin:2px 0 6px;background:{GRAY_LIGHT};border-radius:10px;'
        f'border:1px solid #e6ecf5">{"".join(cells)}</div>', unsafe_allow_html=True)


def _acq_for_channel(name: str) -> dict:
    """Params de adquisición efectivos del canal (por tipo → fallback general)."""
    _bt = st.session_state.get("rm_acq_by_type_saved") or {}
    _tmap = st.session_state.get("rm_type_by_name") or {}
    ctype = _tmap.get(name, "proximity")
    return dict(_bt.get(ctype) or st.session_state.get("rm_acq_saved") or {})


def _plot_waveform(chans, fs: float, rpm: Optional[float] = None,
                   state_lbl: str = "", state_col: str = "#9fb3d1", vent_s: float = 0.0) -> None:
    """Una o varias formas de onda apiladas, estilo estación de análisis:
    encabezado autocontenido (tag · canal · RPM · estado · ventana · fecha),
    caja de datos DENTRO de cada onda, eje X en ms desde 0 y puntos de
    keyphasor por vuelta. `chans` = lista de (señal, ChannelConfig)."""
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    machine = st.session_state.get("rm_machine_name", "—")
    # 32 vueltas fijas (punto dulce, como System1 Wf 64X/32) — sin control extra.
    n_rev = 32
    prepared = []
    for sig, ch in chans:
        eu = sig * 1000.0 / ch.sensitivity_mv_per_eu if ch.sensitivity_mv_per_eu else sig
        eu = eu - np.mean(eu)
        if rpm and rpm > 0:
            n_show = min(len(eu), max(1, int(round(n_rev * (60.0 / rpm) * fs))))
        else:
            n_show = min(len(eu), int(0.5 * fs))
        eu = (eu[-n_show:] if n_show else eu)
        eu = eu - np.mean(eu)
        t_ms = np.arange(len(eu)) / fs * 1000.0
        rms = float(np.sqrt(np.mean(eu ** 2))) if len(eu) else 0.0
        pk = float(np.max(np.abs(eu))) if len(eu) else 0.0
        pp = float(np.ptp(eu)) if len(eu) else 0.0
        crest = pk / rms if rms > 0 else 0.0
        prepared.append(dict(ch=ch, eu=eu, t=t_ms, rms=rms, pk=pk, pp=pp, crest=crest))
    if not prepared:
        return
    xmax_ms = max((float(p["t"][-1]) for p in prepared if len(p["t"])), default=1.0)
    sr = (fs * 60.0 / rpm) if rpm else 0.0   # samples/rev aprox

    # Encabezado autocontenido (contexto de máquina, común a todos los canales):
    # tag · RPM · estado · ventana · fecha. La identidad de cada canal va en su
    # propio pill dentro de la onda.
    ts = datetime.now().strftime("%d %b %Y · %H:%M:%S")
    ctx = ""
    if rpm:
        ctx = (f'<span style="color:#c7d6ea">RPM <b style="color:#fff">{rpm:.0f}</b> · '
               f'State <b style="color:{state_col}">{state_lbl}</b> · '
               f'Window <b style="color:#fff">{vent_s:.1f} s</b></span>')
    st.markdown(
        f'<div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;'
        f'gap:4px 18px;padding:7px 12px;background:{_S1_TITLE};border-radius:8px 8px 0 0;color:#fff;'
        f'font-size:12px;font-family:Arial,Helvetica,sans-serif">'
        f'<span><b>{machine}</b></span>'
        f'{ctx}'
        f'<span style="color:#9fb3d1">🕒 {ts}</span></div>', unsafe_allow_html=True)

    wfcol = ["#2f6fb0", "#7c5cd6", "#159a5b", "#d9822b"]   # color por canal (le da vida)
    rows = len(prepared)
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.09)
    for r, p in enumerate(prepared, start=1):
        c = wfcol[(r - 1) % len(wfcol)]
        fig.add_trace(go.Scatter(
            x=p["t"], y=p["eu"], mode="lines", line=dict(width=1.1, color=c),
            hovertemplate=f"%{{x:.1f}} ms<br>%{{y:.4g}} {p['ch'].units}<extra></extra>"),
            row=r, col=1)
        if rpm and rpm > 0 and len(p["t"]):
            period_ms = 60000.0 / rpm
            kt = np.arange(0.0, float(p["t"][-1]), period_ms)
            ky = np.interp(kt, p["t"], p["eu"])
            fig.add_trace(go.Scatter(x=kt, y=ky, mode="markers",
                                     marker=dict(size=6, color=_S1_KPH), showlegend=False,
                                     hovertemplate="Keyphasor<br>%{x:.1f} ms<extra></extra>"),
                          row=r, col=1)
        ymax = _nice_top(p["pk"] * 1.15) if p["pk"] > 0 else 1.0
        fig.update_yaxes(title_text=p["ch"].units, range=[-ymax, ymax],
                         zeroline=True, zerolinecolor="#d5dbe4", showgrid=True, gridcolor=_S1_GRID,
                         showline=True, linecolor=_S1_AXIS, ticks="outside", tickcolor=_S1_AXIS,
                         row=r, col=1)
        # Pill de identidad del canal (arriba-izquierda de cada onda).
        _sfx0 = "" if r == 1 else str(r)
        fig.add_annotation(
            xref=f"x{_sfx0} domain", yref=f"y{_sfx0} domain", x=0.008, y=0.95,
            xanchor="left", yanchor="top", showarrow=False, text=f"<b>{p['ch'].name}</b>",
            font=dict(size=12, color="#fff", family="Arial, Helvetica, sans-serif"),
            bgcolor=c, borderpad=4, opacity=0.96)
        # Caja de datos DENTRO de la onda (abajo-derecha), marco azul, título en
        # negrita y valor en cursiva — como la competencia (System1).
        _sfx = "" if r == 1 else str(r)
        _kv = (lambda k, v: f'<b style="color:{_S1_TITLE}">{k}</b> '
                            f'<i style="color:#2f6fb0">{v}</i>')
        fig.add_annotation(
            xref=f"x{_sfx} domain", yref=f"y{_sfx} domain", x=0.992, y=0.04,
            xanchor="right", yanchor="bottom", align="left", showarrow=False,
            text=("&nbsp;&nbsp;·&nbsp;&nbsp;".join([
                _kv("pp", f'{p["pp"]:.3g} {p["ch"].units}'),
                _kv("rms", f'{p["rms"]:.3g}'),
                _kv("CF", f'{p["crest"]:.2f}')])),
            font=dict(size=11, family="Arial, Helvetica, sans-serif"),
            bgcolor="rgba(244,249,255,0.94)", bordercolor="#2f6fb0",
            borderwidth=1.4, borderpad=7)
    fig.update_xaxes(title_text="ms", range=[0, xmax_ms], showgrid=True, gridcolor=_S1_GRID,
                     showline=True, linecolor=_S1_AXIS, ticks="outside", tickcolor=_S1_AXIS,
                     showspikes=True, spikecolor="#94a3b8", spikemode="across",
                     spikesnap="cursor", row=rows, col=1)
    fig.update_layout(height=max(340, 300 * rows), margin=dict(l=58, r=12, t=8, b=40),
                      plot_bgcolor="#ffffff", paper_bgcolor="#ffffff", font=_S1_FONT,
                      hovermode="closest", showlegend=False)
    st.plotly_chart(fig, use_container_width=True, config=_PLOTLY_CFG)


def _spectrum(x: np.ndarray, fs: float):
    x = x - np.mean(x)
    w = np.hanning(len(x))
    mag = np.abs(np.fft.rfft(x * w)) / (np.sum(w) / 2)
    freqs = np.fft.rfftfreq(len(x), 1.0 / fs)
    return freqs, mag


# Convención de amplitud por tipo de sensor, según norma internacional:
#   proximity (desplazamiento relativo) -> pp   [API 670 · ISO 7919 / ISO 20816-2]
#   velometer (velocidad de carcasa)    -> rms  [ISO 20816 / ISO 10816]
#   accelerometer (aceleración)         -> rms  [ISO 20816]
# pp = 2·(0-pk) = 2·√2·rms  ·  rms = (0-pk)/√2
_SQRT2 = float(np.sqrt(2.0))
_AMP_STD = {
    "proximity":     ("pp",  "API 670 · ISO 7919"),
    "velometer":     ("rms", "ISO 20816"),
    "accelerometer": ("rms", "ISO 20816"),
}


def _amp_conv(ctype):
    """(sufijo, norma, k_desde_0pk, k_desde_rms) para el tipo de sensor.
    k_desde_0pk: multiplica una amplitud espectral 0-pk → convención.
    k_desde_rms: multiplica un RMS del dominio del tiempo → convención."""
    conv, norm = _AMP_STD.get(ctype or "proximity", _AMP_STD["proximity"])
    if conv == "pp":
        return "pp", norm, 2.0, 2.0 * _SQRT2
    return "rms", norm, 1.0 / _SQRT2, 1.0


def _amp_unit(units_native, conv):
    """Unidad de display sin duplicar el sufijo: la unidad nativa suele venir con
    'pp'/'rms' (ej. 'mil pp') → se limpia y se le pega la convención de la norma.
    'mil pp' + pp → 'mil pp' (no 'mil pp pp'); 'mm/s' + rms → 'mm/s rms'."""
    base = (str(units_native or "")
            .replace("pp", "").replace("PP", "")
            .replace("rms", "").replace("RMS", "").strip())
    return f"{base} {conv}".strip()


def _order_amp_phase(freqs, mag, eu, fs, f_target, tol_frac=0.06):
    """Amplitud 0-pk del armónico (leída del PICO real del espectro cerca de
    f_target) + su fase. Robusto cuando el rpm estimado no cae exacto en la
    frecuencia real: evita el colapso de la proyección síncrona a f_target.
    Devuelve (amp_0pk_EU, fase_deg)."""
    if f_target <= 0 or not len(freqs):
        return 0.0, 0.0
    df = (freqs[1] - freqs[0]) if len(freqs) > 1 else 1.0
    tol = max(3 * df, tol_frac * f_target)
    band = np.abs(freqs - f_target) <= tol
    if band.any():
        idx = np.where(band)[0]
        j = int(idx[int(np.argmax(mag[idx]))])
    else:
        j = int(np.argmin(np.abs(freqs - f_target)))
    _, ph = one_x_vector(eu, fs, float(freqs[j]))
    return float(mag[j]), ph


def _plot_spectrum(chans, fs: float, rpm: Optional[float] = None) -> None:
    """Uno o varios espectros apilados, estilo estación de análisis: header
    autocontenido, fondo blanco, traza fina, caja de ARMÓNICOS por canal. Las
    amplitudes 1X..6X se leen del PICO real del espectro (no de proyección
    síncrona) → coinciden con lo que se ve. `chans` = lista de (señal, Channel)."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from core.remote_monitoring.config import hz_to_display, freq_label
    machine = st.session_state.get("rm_machine_name", "—")
    tmap = st.session_state.get("rm_type_by_name") or {}
    f1 = (rpm / 60.0) if rpm else None

    prepared = []
    for sig, ch in chans:
        p = _acq_for_channel(ch.name)
        fmin_hz = float(p.get("fmin_hz", 0) or 0)
        fmax_hz = float(p.get("fmax_hz", 0) or 0)
        freq_unit = p.get("freq_unit", "cpm")
        conv, norm, k0, krms = _amp_conv(tmap.get(ch.name, "proximity"))
        eu = sig * 1000.0 / ch.sensitivity_mv_per_eu if ch.sensitivity_mv_per_eu else sig
        freqs, mag = _spectrum(eu, fs)
        amp = mag * k0                          # amplitud en la convención de la norma
        unit = freq_label(freq_unit)
        fdisp = freqs * (60.0 if unit == "CPM" else 1.0)
        xmin = hz_to_display(fmin_hz, freq_unit) if fmin_hz > 0 else 0.0
        xmax = hz_to_display(fmax_hz, freq_unit) if fmax_hz > 0 else (fdisp[-1] if len(fdisp) else 1.0)
        band = (freqs >= (fmin_hz or 0)) & (freqs <= (fmax_hz if fmax_hz > 0 else (freqs[-1] if len(freqs) else 0)))
        ov = float(np.sqrt(np.sum(mag[band] ** 2) / 2.0)) * krms if band.any() else 0.0
        orders = (freqs / f1) if f1 else np.zeros_like(freqs)
        peak = float(amp[band].max()) if band.any() else (float(amp.max()) if len(amp) else 0.0)
        prepared.append(dict(ch=ch, eu=eu, freqs=freqs, fdisp=fdisp, amp=amp, unit=unit,
                             uconv=conv, u=_amp_unit(ch.units, conv), norm=norm,
                             freq_unit=freq_unit, xmin=xmin, xmax=xmax,
                             fmax_hz=fmax_hz, ov=ov, orders=orders, peak=peak))
    if not prepared:
        return

    def _harm(p, k):
        """Amplitud pp del armónico k leyendo el PICO del espectro cerca de k·f1."""
        tgt = hz_to_display(k * f1, p["freq_unit"])
        fd = p["fdisp"]
        dfd = (fd[1] - fd[0]) if len(fd) > 1 else 1.0
        tol = max(3 * dfd, 0.006 * tgt)
        m = np.abs(fd - tgt) <= tol
        a = float(p["amp"][m].max()) if m.any() else float(p["amp"][int(np.argmin(np.abs(fd - tgt)))])
        return a, tgt

    # Header (tag · Espectro · canales · rpm · fecha).
    ts = datetime.now().strftime("%d %b %Y · %H:%M:%S")
    chlabel = ", ".join(p["ch"].name for p in prepared)
    st.markdown(
        f'<div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;'
        f'gap:4px 18px;padding:7px 12px;background:{_S1_TITLE};border-radius:8px 8px 0 0;color:#fff;'
        f'font-size:12px;font-family:Arial,Helvetica,sans-serif">'
        f'<span><b>{machine}</b> · Spectrum · {chlabel}'
        + (f' · <span style="color:#c7d6ea">{rpm:.0f} rpm</span>' if rpm else '') + '</span>'
        f'<span style="color:#9fb3d1">🕒 {ts}</span></div>', unsafe_allow_html=True)

    rows_n = len(prepared)
    fig = make_subplots(rows=rows_n, cols=1, shared_xaxes=False, vertical_spacing=0.10)
    _kv = (lambda k, v: f'<b style="color:{_S1_TITLE}">{k}</b> <i style="color:#2f6fb0">{v}</i>')
    for r, p in enumerate(prepared, start=1):
        c = _S1_BLUE
        fig.add_trace(go.Scatter(
            x=p["fdisp"], y=p["amp"], mode="lines", line=dict(width=1.0, color=c),
            customdata=p["orders"],
            hovertemplate=(f"%{{x:.0f}} {p['unit']}<br>%{{y:.4g}} {p['u']}"
                           + ("<br>%{customdata:.2f}X" if f1 else "") + "<extra></extra>")),
            row=r, col=1)
        if f1:
            for k, lbl in [(1, "1X"), (2, "2X"), (3, "3X")]:
                fx = hz_to_display(k * f1, p["freq_unit"])
                if p["xmin"] <= fx <= p["xmax"]:
                    fig.add_vline(x=fx, line=dict(color="#e26d6d", width=1, dash="dot"),
                                  annotation_text=lbl, annotation_font=dict(size=9, color="#c0392b"),
                                  row=r, col=1)
        ymax = _nice_top(p["peak"] * 1.15) if p["peak"] > 0 else 1.0
        fig.update_yaxes(title_text=f"{p['ch'].name} ({p['u']})", range=[0, ymax],
                         showgrid=True, gridcolor=_S1_GRID, showline=True, linecolor=_S1_AXIS,
                         ticks="outside", tickcolor=_S1_AXIS, row=r, col=1)
        fig.update_xaxes(range=[p["xmin"], p["xmax"]], showgrid=True, gridcolor=_S1_GRID,
                         showline=True, linecolor=_S1_AXIS, ticks="outside", tickcolor=_S1_AXIS,
                         showspikes=True, spikecolor="#94a3b8", spikemode="across",
                         spikesnap="cursor", row=r, col=1,
                         title_text=(f"Frequency ({p['unit']})" if r == rows_n else None))
        # Caja de ARMÓNICOS por canal (arriba-derecha del subplot).
        _sfx = "" if r == 1 else str(r)
        hrows = [_kv("O/All", f"{p['ov']:.3g} {p['u']}")]
        if f1:
            fmax_eff = p["fmax_hz"] if p["fmax_hz"] > 0 else (p["freqs"][-1] if len(p["freqs"]) else 0.0)
            for k in range(1, 7):
                if k * f1 > fmax_eff:
                    break
                a, tgt = _harm(p, k)
                hrows.append(_kv(f"{k}X", f"{a:.3g} @ {tgt:.0f} {p['unit']}"))
        fig.add_annotation(
            xref=f"x{_sfx} domain", yref=f"y{_sfx} domain", x=0.992, y=0.96,
            xanchor="right", yanchor="top", align="left", showarrow=False, text="<br>".join(hrows),
            font=dict(size=10.5, family="Arial, Helvetica, sans-serif"),
            bgcolor="rgba(244,249,255,0.94)", bordercolor="#2f6fb0", borderwidth=1.4, borderpad=7)
    fig.update_layout(height=max(360, 320 * rows_n), margin=dict(l=58, r=16, t=10, b=42),
                      plot_bgcolor="#ffffff", paper_bgcolor="#ffffff", font=_S1_FONT,
                      hovermode="x", showlegend=False)
    st.plotly_chart(fig, use_container_width=True, config=_PLOTLY_CFG)
    _norms = ", ".join(dict.fromkeys(f"{p['ch'].name}: {p['uconv']} ({p['norm']})" for p in prepared))
    st.caption(f"Amplitude by standard — {_norms}.")


def _orbit_dir_arrow(fig, rotation: str, R: float, cx0: float = 0.0, cy0: float = 0.0) -> None:
    """Arco con flecha indicando el sentido de giro, centrado en (cx0, cy0)."""
    import plotly.graph_objects as go
    r = R * 1.20
    cw = (rotation or "CCW").upper() == "CW"
    ang = np.radians(np.linspace(140, 60, 24) if cw else np.linspace(60, 140, 24))
    ax_, ay_ = cx0 + r * np.cos(ang), cy0 + r * np.sin(ang)
    fig.add_trace(go.Scatter(x=ax_, y=ay_, mode="lines", line=dict(color="#0F1E3D", width=2.0),
                             hoverinfo="skip", showlegend=False))
    fig.add_annotation(x=ax_[-1], y=ay_[-1], ax=ax_[-4], ay=ay_[-4],
                       xref="x", yref="y", axref="x", ayref="y", showarrow=True,
                       arrowhead=2, arrowsize=1.4, arrowwidth=2.0, arrowcolor="#0F1E3D", text="")


def _orbit_probe_axes(fig, m_deg_x: float, m_deg_y: float, lim: float,
                      xname: str, yname: str, cx0: float = 0.0) -> None:
    """Ejes de las sondas en su ÁNGULO REAL de montaje + cajita X/Y en la punta,
    centrados en (cx0, 0). (System1: el eje va donde está físicamente la sonda.)"""
    for m_deg, nm in [(m_deg_x, xname), (m_deg_y, yname)]:
        m = np.radians(m_deg)
        cx, cy = np.cos(m), np.sin(m)
        fig.add_shape(type="line", x0=cx0 - lim * cx, y0=-lim * cy, x1=cx0 + lim * cx, y1=lim * cy,
                      line=dict(color="#dbe2ee", width=1.2), layer="below")
        fig.add_annotation(x=cx0 + 0.9 * lim * cx, y=0.9 * lim * cy, showarrow=False,
                           text=f"<b>{nm}</b>", font=dict(size=10.5, color="#334155"),
                           bgcolor="#ffffff", bordercolor="#c7d0dc", borderwidth=1, borderpad=3)


def _plot_orbit(snap: np.ndarray, vib_channels, fs: float, rpm: Optional[float] = None) -> None:
    """Una o varias órbitas lado a lado, rotadas al ángulo REAL de las sondas
    (estilo System1). Info dentro de cada figura; opción de unir los keyphasor
    con una curva (locus por vuelta) para análisis."""
    import plotly.graph_objects as go
    if len(vib_channels) < 2:
        st.info("The orbit needs an X/Y pair. Associate one in **Configuration → Channel editor → X/Y pair**.")
        return
    name_to = {ch.name: (i, ch) for i, ch in vib_channels}
    saved = st.session_state.get("rm_pairs_saved") or []
    valid = [(a, b) for a, b in saved if a in name_to and b in name_to]
    if not valid:
        names = [ch.name for _, ch in vib_channels]
        valid = [(names[i], names[i + 1]) for i in range(0, len(vib_channels) - 1, 2)]
    if not valid:
        st.info("The orbit needs an X/Y pair. Associate one in **Configuration → Channel editor → X/Y pair**.")
        return

    labels = [f"{a}–{b}" for a, b in valid]
    f1 = (rpm / 60.0) if rpm else None
    angmap = st.session_state.get("rm_angle_by_name") or {}
    tmap = st.session_state.get("rm_type_by_name") or {}
    rotation = st.session_state.get("rm_machine_rotation", "CCW")

    # Controles compactos en UNA fila: pares · filtro · (vueltas) · unir keyphasor.
    with st.container(key="rm_orbit_ctrls", horizontal=True,
                      vertical_alignment="center", gap="medium"):
        st.session_state.setdefault("rm_orbit_pairs", [labels[0]])
        st.session_state["rm_orbit_pairs"] = [s for s in st.session_state["rm_orbit_pairs"]
                                              if s in labels] or [labels[0]]
        sels = st.multiselect("Pairs", labels, key="rm_orbit_pairs",
                              label_visibility="collapsed", placeholder="Orbits…")
        fmode = st.radio("Filter", ["Direct", "1X", "2X"], horizontal=True,
                         key="rm_orbit_filter", label_visibility="collapsed",
                         help="Directa = full waveform. 1X/2X = orbit filtered to the order (ellipse).")
        n_rev = 12
        if fmode == "Direct":
            n_rev = int(st.number_input("Revolutions", 3, 60, 12, step=1, key="rm_orbit_revs",
                                        label_visibility="collapsed"))
        kphline = st.toggle("Link keyphasors", key="rm_orbit_kphline",
                            help="Links the keyphasors revolution to revolution (locus). Visible in "
                                 "transients; at steady state they coincide in one point.")
    sels = sels or [labels[0]]

    def _prep(pair_lbl):
        a, b = valid[labels.index(pair_lbl)]
        is_y = lambda n: "Y" in n.upper()
        if is_y(a) and not is_y(b):
            yname, xname = a, b
        elif is_y(b) and not is_y(a):
            yname, xname = b, a
        else:
            yname, xname = a, b
        yi, chy = name_to[yname]
        xi, chx = name_to[xname]
        y = snap[yi] * 1000.0 / chy.sensitivity_mv_per_eu
        x = snap[xi] * 1000.0 / chx.sensitivity_mv_per_eu
        x, y = x - np.mean(x), y - np.mean(y)
        kx = ky = vec = None
        if fmode == "Direct" or not f1:
            if f1:
                n_show = min(len(x), max(1, int(round(n_rev * (1.0 / f1) * fs))))
                x, y = x[-n_show:], y[-n_show:]
            x, y = x - np.mean(x), y - np.mean(y)
            if f1:
                spr = fs / f1
                kt = np.arange(0.0, float(len(x)), spr)
                idx = np.arange(len(x))
                kx, ky = np.interp(kt, idx, x), np.interp(kt, idx, y)
        else:
            n = 1.0 if fmode == "1X" else 2.0
            f = n * f1
            axv, pxv = one_x_vector(x, fs, f)
            ayv, pyv = one_x_vector(y, fs, f)
            tt = np.linspace(0.0, 1.0 / f1, 400)
            x = axv * np.cos(2 * np.pi * f * tt + np.radians(pxv))
            y = ayv * np.cos(2 * np.pi * f * tt + np.radians(pyv))
            kx, ky = np.array([x[0]]), np.array([y[0]])
            vec = (axv, pxv, ayv, pyv)
        # Ángulos: absoluto (desde TDC, horario) → matemático (desde +X, CCW).
        ax_abs = float(angmap.get(xname, 45.0))    # 45°R por defecto (típico)
        ay_abs = float(angmap.get(yname, 315.0))   # 45°L por defecto
        mX, mY = 90.0 - ax_abs, 90.0 - ay_abs
        mXr, mYr = np.radians(mX), np.radians(mY)
        # Rotación a marco físico (vertical = TDC arriba).
        h = x * np.cos(mXr) + y * np.cos(mYr)
        v = x * np.sin(mXr) + y * np.sin(mYr)
        khh = khv = None
        if kx is not None:
            khh = kx * np.cos(mXr) + ky * np.cos(mYr)
            khv = kx * np.sin(mXr) + ky * np.sin(mYr)
        conv, norm, _k0, _krms = _amp_conv(tmap.get(yname, "proximity"))
        return dict(lbl=pair_lbl, xname=xname, yname=yname, chx=chx, chy=chy,
                    h=h, v=v, khh=khh, khv=khv, mX=mX, mY=mY, vec=vec, conv=conv, norm=norm,
                    u=_amp_unit(chy.units, conv), xpp=float(np.ptp(x)), ypp=float(np.ptp(y)),
                    smax=float(np.max(np.sqrt(h ** 2 + v ** 2))) if len(h) else 0.0)

    def _lead_int(s):
        n = ""
        for c in str(s):
            if c.isdigit():
                n += c
            elif n:
                break
        return int(n) if n else 999

    data = [_prep(s) for s in sels]
    # Orden SIEMPRE de cojinete 1 (lado libre) al último → la forma deflectada
    # arranca en el lado libre y recorre la máquina.
    data.sort(key=lambda d: _lead_int(d["xname"]))
    gR = max((max(float(np.max(np.abs(d["h"]))) if len(d["h"]) else 0.0,
                  float(np.max(np.abs(d["v"]))) if len(d["v"]) else 0.0) for d in data),
             default=1e-9)
    lim = gR * 1.5   # escala COMÚN → órbitas comparables

    # Header autocontenido (tag · Órbita · filtro · rpm · fecha).
    machine = st.session_state.get("rm_machine_name", "—")
    ts = datetime.now().strftime("%d %b %Y · %H:%M:%S")
    st.markdown(
        f'<div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;'
        f'gap:4px 18px;padding:7px 12px;background:{_S1_TITLE};border-radius:8px 8px 0 0;color:#fff;'
        f'font-size:12px;font-family:Arial,Helvetica,sans-serif">'
        f'<span><b>{machine}</b> · Orbit · <span style="color:#c7d6ea">{fmode}</span>'
        + (f' · <span style="color:#c7d6ea">{rpm:.0f} rpm</span>' if rpm else '') + '</span>'
        f'<span style="color:#9fb3d1">🕒 {ts}</span></div>', unsafe_allow_html=True)

    _kv = (lambda k, v: f'<b style="color:{_S1_TITLE}">{k}</b> <i style="color:#2f6fb0">{v}</i>')
    # UN solo lienzo: las órbitas se colocan una seguida de otra (trasladadas en X)
    # dentro del MISMO sistema de coords → la línea que une los keyphasor de bearing
    # 1→2→3 es directa (= forma deflectada del eje en el instante de keyphasor).
    N = len(data)
    Dx = 2.7 * lim                      # separación centro-a-centro
    fig = go.Figure()
    node_x, node_y, node_lbl = [], [], []
    for k, d in enumerate(data):
        cx0 = k * Dx
        fig.add_trace(go.Scatter(
            x=d["h"] + cx0, y=d["v"], mode="lines", line=dict(width=1.4, color=_S1_BLUE),
            name=d["lbl"], hovertemplate=f"{d['lbl']}<br>H %{{customdata:.3g}}<br>"
            f"V %{{y:.3g}} {d['u']}<extra></extra>", customdata=d["h"]))
        _orbit_probe_axes(fig, d["mX"], d["mY"], lim, d["xname"], d["yname"], cx0=cx0)
        if d["khh"] is not None and len(d["khh"]):
            fig.add_trace(go.Scatter(x=d["khh"] + cx0, y=d["khv"], mode="markers", showlegend=False,
                                     marker=dict(size=7, color="#dc2626", line=dict(width=1, color="#fff")),
                                     hovertemplate="Keyphasor<extra></extra>"))
            node_x.append(float(np.mean(d["khh"])) + cx0)
            node_y.append(float(np.mean(d["khv"])))
            node_lbl.append(d["lbl"])
        _orbit_dir_arrow(fig, rotation, lim / 1.5, cx0=cx0)
        # Solo el TAG del par, chico y limpio, debajo de la órbita (sin caja encima).
        fig.add_annotation(x=cx0, y=-1.2 * lim, xref="x", yref="y", showarrow=False,
                           text=f"<b>{d['lbl']}</b>", font=dict(size=11, color=_S1_TITLE))
    # Forma deflectada: une los keyphasor de cada bearing (instante común).
    if kphline and len(node_x) > 1:
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y, mode="lines+markers", line=dict(width=2.4, color="#e0982a", shape="spline"),
            marker=dict(size=9, color="#e0982a", line=dict(width=1.5, color="#fff")),
            hovertemplate="Forma deflectada<extra></extra>", showlegend=False))

    x_lo, x_hi = -1.35 * lim, (N - 1) * Dx + 1.35 * lim
    # Alto proporcional al aspecto (aprox ancho 760) para evitar bandas blancas.
    h_px = int(max(240, min(470, 760.0 * (2.7 * lim) / (x_hi - x_lo))))
    fig.update_layout(height=h_px, margin=dict(l=6, r=6, t=6, b=6),
                      plot_bgcolor="#ffffff", paper_bgcolor="#ffffff", font=_S1_FONT,
                      showlegend=False)
    fig.update_xaxes(range=[x_lo, x_hi], showgrid=False, zeroline=False,
                     showticklabels=False, showline=False, ticks="")
    fig.update_yaxes(range=[-1.35 * lim, 1.35 * lim], zeroline=True, zerolinecolor="#c9d2e0",
                     zerolinewidth=1, showgrid=True, gridcolor=_S1_GRID, showline=True,
                     linecolor=_S1_AXIS, ticks="outside", tickcolor=_S1_AXIS,
                     scaleanchor="x", scaleratio=1)
    st.plotly_chart(fig, use_container_width=True, config=_PLOTLY_CFG)

    # Valores FUERA del gráfico, una tarjeta por órbita ALINEADA bajo cada una
    # (los centros de órbita caen en el centro de cada columna → st.columns encaja).
    # La letra se achica con más órbitas para que todo quepa en su columna.
    _fs = 12.0 if N <= 2 else (11.0 if N == 3 else (10.0 if N <= 5 else 9.0))

    def _card(d):
        if d["vec"] is None:
            inner = (_kv("Xpp", f"{d['xpp']:.3g} {d['u']}") + "<br>" +
                     _kv("Ypp", f"{d['ypp']:.3g} {d['u']}") + "<br>" +
                     _kv("Smax", f"{d['smax']:.3g} {d['chy'].units}"))
        else:
            axv, pxv, ayv, pyv = d["vec"]
            inner = (_kv(f"{fmode} X", f"{axv * 2:.3g} {d['u']} ∠{pxv:.0f}°") + "<br>" +
                     _kv(f"{fmode} Y", f"{ayv * 2:.3g} {d['u']} ∠{pyv:.0f}°") + "<br>" +
                     _kv("Smax", f"{d['smax']:.3g} {d['chy'].units}"))
        return (f'<div style="padding:6px 8px;border:1px solid #d6deea;border-radius:8px;'
                f'background:#f8fafd;font-size:{_fs}px;line-height:1.55;text-align:center;'
                f'word-break:break-word;font-family:Arial,Helvetica,sans-serif">{inner}</div>')
    vcols = st.columns(len(data), gap="small")
    for d, vc in zip(data, vcols):
        with vc:
            st.markdown(_card(d), unsafe_allow_html=True)


def _table_orders(snap: np.ndarray, vib_channels, fs: float, rpm: Optional[float],
                  orders: Optional[list] = None) -> None:
    if not rpm:
        st.warning("Without a keyphasor there are no vectors. Enable the keyphasor in Configuration.")
        return
    orders = sorted(orders or [1.0, 2.0])
    f1 = rpm / 60.0
    rows = []
    for i, ch in vib_channels:
        eu = snap[i] * 1000.0 / ch.sensitivity_mv_per_eu
        row = {"Sensor": ch.name}
        for o in orders:
            amp, phase = one_x_vector(eu, fs, o * f1)
            row[f"{o:g}X ({ch.units})"] = round(amp, 4)
            row[f"{o:g}X °"] = round(phase, 1)
        rows.append(row)
    st.dataframe(rows, use_container_width=True, hide_index=True)
    st.caption(f"Synchronous vectors at {', '.join(f'{o:g}X' for o in orders)} "
               f"(referenced to the keyphasor). 1X = {f1:.1f} Hz.")


_FAMILY_ES = {"proximity": "Desplazamiento", "velometer": "Velocidad", "accelerometer": "Aceleración"}
_TREND_PALETTE = ["#2563eb", "#06b6d4", "#16a34a", "#8b5cf6", "#ef4444",
                  "#f97316", "#ec4899", "#64748b", "#0f766e", "#7c3aed"]

# Sin la barra de íconos de plotly (se veía "de quinta"). Solo gráfico limpio.
_PLOTLY_CFG = {"displayModeBar": False, "staticPlot": False, "scrollZoom": False}

# Paleta "System1 mejorado": fondo blanco, traza azul aciano fina, grilla tenue.
_S1_BLUE = "#4f8fd0"        # traza (cornflower, como System1)
_S1_KPH = "#12467f"         # puntos de keyphasor (azul profundo)
_S1_GRID = "rgba(15,30,61,0.06)"
_S1_AXIS = "#aeb6c2"
_S1_TITLE = "#0F1E3D"
_S1_FONT = dict(color="#334155", size=11, family="Arial, Helvetica, sans-serif")


def _s1_readout(fig, lines, *, x=0.995, y=0.98) -> None:
    """Caja de lectura tipo 'cursor' de System1 (arriba-derecha, monoespaciada)."""
    header = lines[0]
    body = "<br>".join(lines[1:])
    txt = (f"<b>{header}</b>" + ("<br>" + body if body else ""))
    fig.add_annotation(xref="paper", yref="paper", x=x, y=y, xanchor="right", yanchor="top",
                       align="left", showarrow=False, text=txt,
                       font=dict(size=10, color="#1f2937", family="ui-monospace, monospace"),
                       bordercolor="#c7d0dc", borderwidth=1, borderpad=7,
                       bgcolor="rgba(255,255,255,0.92)")


def _nice_top(v: float) -> float:
    """Techo 'redondo' >= v para el eje Y (1,1.2,1.5,2,2.5,3,4,5,6,8,10 × 10ⁿ).
    Ej.: 5.75 → 6, 0.575 → 0.6, 57.5 → 60."""
    import math
    if v <= 0:
        return 1.0
    exp = math.floor(math.log10(v))
    base = 10.0 ** exp
    for m in (1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10):
        if m * base >= v - 1e-12:
            return round(m * base, 10)
    return 10.0 * base


def _half_power_af(rr, aa, ipk):
    """Factor de amplificación (AF/SAF) por ancho de banda de media potencia
    (API 684): AF = Nc / (N2 − N1), con N1,N2 donde amp = pico/√2.
    Devuelve (AF, N1, N2, h) o None."""
    apk = aa[ipk]
    if apk <= 0:
        return None
    h = apk / np.sqrt(2.0)
    N1 = N2 = None
    for i in range(ipk, 0, -1):
        if aa[i] >= h > aa[i - 1]:
            N1 = float(np.interp(h, [aa[i - 1], aa[i]], [rr[i - 1], rr[i]])); break
    for i in range(ipk, len(aa) - 1):
        if aa[i] >= h > aa[i + 1]:
            N2 = float(np.interp(h, [aa[i + 1], aa[i]], [rr[i + 1], rr[i]])); break
    if N1 is None or N2 is None or N2 <= N1:
        return None
    return rr[ipk] / (N2 - N1), N1, N2, h


def _detect_criticals(rr, aa, thr_frac=0.4, merge_frac=0.08, top=4):
    """Índices de picos de resonancia (locales prominentes, fusionando cercanos),
    ordenados por rpm ascendente."""
    if len(aa) < 3:
        return [int(np.argmax(aa))] if len(aa) else []
    mx = float(np.max(aa))
    cand = [i for i in range(1, len(aa) - 1)
            if aa[i] >= aa[i - 1] and aa[i] >= aa[i + 1] and aa[i] > thr_frac * mx]
    span = (rr[-1] - rr[0]) if len(rr) > 1 else 1.0
    out = []
    for i in cand:
        if out and abs(rr[i] - rr[out[-1]]) < merge_frac * span:
            if aa[i] > aa[out[-1]]:
                out[-1] = i
        else:
            out.append(i)
    out = sorted(sorted(out, key=lambda i: -aa[i])[:top], key=lambda i: rr[i])
    return out or [int(np.argmax(aa))]


def _op_margin(rr, crit_idx, af_by_crit, op_rpm):
    """Estado de la velocidad de operación vs la crítica más próxima (API 684/617).
    Devuelve dict(status, msg, color) o None si no aplica."""
    if op_rpm <= 0 or not len(crit_idx):
        return None
    _ORD = ["1st", "2nd", "3rd", "4th", "5th"]
    j = min(crit_idx, key=lambda i: abs(rr[i] - op_rpm))
    ncj, afj = float(rr[j]), af_by_crit.get(j)
    sm = abs(ncj - op_rpm) / op_rpm * 100.0
    below = ncj < op_rpm
    req = 15.0 if below else 20.0
    needs = (afj is None) or (afj >= 2.5)
    oi = crit_idx.index(j)
    ordl = _ORD[oi] if oi < len(_ORD) else f"{oi+1}ª"
    af_txt = f", AF {afj:.1f}" if afj else ""
    if sm < 3.0:
        return dict(status="res", color="#dc2626",
                    msg=(f"⚠ **AT RESONANCE** — the operating speed ({op_rpm:.0f} rpm) coincides "
                         f"with the {ordl} critical ({ncj:.0f} rpm{af_txt}). Amplitude and phase very sensitive; "
                         f"**API 684 does not allow operating on a critical**."))
    if needs and sm < req:
        return dict(status="margin", color="#dc2626",
                    msg=(f"⚠ **Insufficient margin** — operating at **{sm:.0f}%** of the {ordl} critical "
                         f"({ncj:.0f} rpm{af_txt}); API 684 requires **≥{req:.0f}%** "
                         f"({'critical below' if below else 'critical above'}) for "
                         f"lightly damped modes (AF≥2.5). Risk of high amplitudes."))
    return dict(status="ok", color="#16a34a",
                msg=(f"✓ **Separation margin {sm:.0f}%** to the {ordl} critical ({ncj:.0f} rpm) — "
                     f"meets API 684 (≥{req:.0f}%)."))


def _cascade_diagnosis(rpms, freqs, mat, crit_rpms):
    """Auto-diagnóstico de la cascada: velocidades críticas (modos) +
    inestabilidades SUBSÍNCRONAS con nombre propio, discriminando oil whirl (sigue
    el rpm ~0.45X), oil whip (se engancha a una natural = frecuencia fija) y ½X
    (roce/holgura). Devuelve lista de (nivel, título, detalle)."""
    out = []
    for nc in crit_rpms:
        out.append(("info", f"Critical speed ≈ {nc:.0f} rpm",
                    "Resonance of a rotor bending mode (the 1X is amplified). "
                    "Check the separation margin (API 684)."))
    if not len(rpms) or freqs is None or not len(freqs):
        return out
    df = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 1.0
    gmax = float(np.max(mat)) if mat.size else 0.0
    pts = []          # (rpm, freq_sub, order, amp)
    for i, rp in enumerate(rpms):
        f1 = rp / 60.0
        if f1 <= 0:
            continue
        band = (freqs >= 0.15 * f1) & (freqs <= 0.9 * f1)     # zona subsíncrona
        if not band.any():
            continue
        sub = mat[i][band]
        fb = freqs[band]
        j = int(np.argmax(sub))
        af, ff = float(sub[j]), float(fb[j])
        b1 = np.abs(freqs - f1) <= max(df, 0.05 * f1)
        a1 = float(mat[i][b1].max()) if b1.any() else 0.0
        if af > 0.25 * max(a1, 1e-9) and af > 0.06 * max(gmax, 1e-9):
            pts.append((rp, ff, ff / f1, af))
    if len(pts) < max(4, len(rpms) // 8):
        return out
    P = np.array(pts, float)
    ffreq, orders = P[:, 1], P[:, 2]
    mo = float(np.median(orders))
    f_cv = float(np.std(ffreq) / (np.mean(ffreq) + 1e-9))     # ~0 = frecuencia fija (whip)
    o_cv = float(np.std(orders) / (np.mean(orders) + 1e-9))   # ~0 = orden fijo (whirl/½X)
    near_crit = bool(crit_rpms) and any(abs(np.mean(ffreq) * 60.0 - nc) < 0.18 * nc for nc in crit_rpms)
    if f_cv < 0.14 and o_cv > 0.18 and near_crit:
        out.append(("danger", "⚠ OIL WHIP",
                    f"Subsynchronous vibration LOCKED to a natural frequency (~{np.mean(ffreq):.0f} Hz "
                    f"≈ 1st critical): it stays FIXED as speed increases. **Severe and destructive** "
                    f"oil-film instability (API 684) — act (load, clearance, "
                    f"bearing type)."))
    elif 0.35 <= mo <= 0.49 and o_cv < 0.16:
        out.append(("danger", "⚠ OIL WHIRL",
                    f"Subsynchronous vibration at **~{mo:.2f}X** that FOLLOWS speed (frequency = "
                    f"~{mo:.2f}×rpm): oil-film instability of the bearing (API 684). "
                    f"It can degenerate into oil whip past ~2× the 1st critical."))
    elif 0.47 <= mo <= 0.53 and o_cv < 0.14:
        out.append(("warn", "½X subharmonic",
                    f"Component at **~0.5X** that follows speed → typical of **rub** or "
                    f"**mechanical looseness**, NOT oil film. Distinguished from oil whirl "
                    f"(which runs at ~0.42–0.48X)."))
    else:
        out.append(("warn", f"Subsynchronous ~{mo:.2f}X",
                    "There is energy below 1X; watch how it evolves with speed to "
                    "classify it (whirl follows rpm; whip fixes on a natural)."))
    return out


_TREND_BRIGHT = ["#2f6fb0", "#16a34a", "#e11d48", "#7c3aed",
                 "#0891b2", "#ea580c", "#db2777", "#0f766e"]


def _plot_trend(snap: np.ndarray, vib_channels, type_map: dict,
                rpm: Optional[float] = None) -> None:
    """Tendencia overall estilo System1: elegís los CANALES (multiselección),
    pero solo se pueden mezclar los de misma unidad Y misma alarma/danger (no se
    puede mezclar mils con g). Controles en una fila abajo; barra deslizable en
    períodos históricos."""
    import plotly.graph_objects as go
    from datetime import timedelta
    hist = st.session_state.setdefault("rm_trend", [])
    overall, unit_by = {}, {}
    for i, ch in vib_channels:
        eu = snap[i] * 1000.0 / ch.sensitivity_mv_per_eu if ch.sensitivity_mv_per_eu else snap[i]
        overall[ch.name] = float(np.sqrt(np.mean((eu - np.mean(eu)) ** 2)))
        unit_by[ch.name] = ch.units
    hist.append((datetime.now(), overall))
    if len(hist) > 5000:
        del hist[: len(hist) - 5000]

    name_to_ch = {ch.name: ch for _, ch in vib_channels}
    all_names = list(name_to_ch.keys())
    if not all_names:
        st.info("No vibration channels.")
        return
    machine = st.session_state.get("rm_machine_name", "—")
    alarms = st.session_state.get("rm_alarms_by_name") or {}

    def _compat(n):   # clave de compatibilidad: unidad + alarma + danger
        al, dg = alarms.get(n, (0, 0))
        return (unit_by.get(n, ""), round(float(al), 6), round(float(dg), 6))

    # Selección de canales (multiselección). Solo se plotean los compatibles con
    # el primero; los incompatibles se descartan con aviso ("no se pueden mezclar").
    st.session_state.setdefault("rm_tr_chans", [all_names[0]])
    sels_raw = [s for s in st.session_state["rm_tr_chans"] if s in all_names] or [all_names[0]]
    ref = _compat(sels_raw[0])
    plotted = [s for s in sels_raw if _compat(s) == ref]
    excluded = [s for s in sels_raw if _compat(s) != ref]
    if st.session_state["rm_tr_chans"] != plotted:
        st.session_state["rm_tr_chans"] = plotted   # limpia los incompatibles
    plot_chs = [name_to_ch[n] for n in plotted]
    unit = ref[0]
    als = [ref[1]] if ref[1] > 0 else []
    dgs = [ref[2]] if ref[2] > 0 else []

    # Ventana de tiempo. "Now" = período en vivo de 15 min (sin selector).
    _unit = {"Hours": timedelta(hours=1), "Days": timedelta(days=1),
             "Weeks": timedelta(weeks=1), "Months": timedelta(days=30)}
    _dfl = {"Hours": 6, "Days": 7, "Weeks": 4, "Months": 6}
    sel = st.session_state.get("rm_trend_win", "Now")
    if sel == "Now":
        delta = timedelta(minutes=15)
    else:
        qty = int(st.session_state.get(f"rm_trend_qty_{sel}", _dfl.get(sel, 1)) or 1)
        delta = _unit[sel] * qty
    now = datetime.now()
    xs = [h[0] for h in hist]

    # Encabezado (máquina · Tendencia · canales · rpm · fecha).
    ts = datetime.now().strftime("%d %b %Y · %H:%M:%S")
    st.markdown(
        f'<div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;'
        f'gap:4px 18px;padding:7px 12px;background:{_S1_TITLE};border-radius:8px 8px 0 0;color:#fff;'
        f'font-size:12px;font-family:Arial,Helvetica,sans-serif">'
        f'<span><b>{machine}</b> · Trend · {", ".join(plotted)}'
        + (f' · <span style="color:#c7d6ea">{rpm:.0f} rpm</span>' if rpm else '') + '</span>'
        f'<span style="color:#9fb3d1">🕒 {ts}</span></div>', unsafe_allow_html=True)

    fig = go.Figure()
    for k, ch in enumerate(plot_chs):
        fig.add_trace(go.Scatter(
            x=xs, y=[h[1].get(ch.name) for h in hist], mode="lines", name=ch.name,
            line=dict(width=1.6, color=_TREND_BRIGHT[k % len(_TREND_BRIGHT)])))
    for lvl, col, lbl in [(min(als) if als else None, "#f59e0b", "Alarm"),
                          (min(dgs) if dgs else None, "#e11d48", "Danger")]:
        if lvl:
            fig.add_hline(y=lvl, line=dict(color=col, width=1.4),
                          annotation_text=f"{lbl} {lvl:g}", annotation_position="top left",
                          annotation_font=dict(size=9, color=col))
    data_peak = max((h[1].get(n) for h in hist for n in plotted if h[1].get(n) is not None),
                    default=0.0)
    peak = max([data_peak] + als + dgs) if (als or dgs or data_peak) else 0.0
    ymax = _nice_top(peak * 1.15) if peak > 0 else 1.0

    fig.update_layout(height=520, margin=dict(l=10, r=12, t=46, b=30),
                      plot_bgcolor="#ffffff", paper_bgcolor="#ffffff", font=_S1_FONT,
                      yaxis_title=f"Overall ({unit})" if unit else "Overall",
                      legend=dict(orientation="h", y=1.06, yanchor="bottom", x=1, xanchor="right",
                                  font=dict(size=11), bgcolor="rgba(255,255,255,0)"))
    fig.update_xaxes(type="date", range=[now - delta, now], showgrid=True, gridcolor=_S1_GRID,
                     showline=True, linecolor=_S1_AXIS, ticks="outside", tickcolor=_S1_AXIS)
    if sel != "Now":
        fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.07, bgcolor="#f6f8fc",
                                          bordercolor="#d7deea", borderwidth=1))
    fig.update_yaxes(range=[0, ymax], rangemode="tozero", showgrid=True, gridcolor=_S1_GRID,
                     showline=True, linecolor=_S1_AXIS, ticks="outside", tickcolor=_S1_AXIS)
    st.plotly_chart(fig, use_container_width=True, config=_PLOTLY_CFG)

    if excluded:
        st.caption(f"⚠ Cannot mix: **{', '.join(excluded)}** has a different "
                   f"unit or alarm/danger than **{plotted[0]}**. Removed from the selection.")

    # Controles en UNA fila abajo: canales + rango + cantidad (compactos).
    with st.container(key="rm_trend_ctrls", horizontal=True,
                      vertical_alignment="center", gap="medium"):
        st.multiselect("Channels", all_names, key="rm_tr_chans",
                       label_visibility="collapsed", placeholder="Channels…")
        st.radio("Range", ["Now"] + list(_unit.keys()), horizontal=True,
                 key="rm_trend_win", label_visibility="collapsed")
        if sel != "Now":
            _k = f"rm_trend_qty_{sel}"
            st.session_state.setdefault(_k, _dfl.get(sel, 1))
            st.number_input("count", 1, 999, step=1, key=_k,
                            label_visibility="collapsed")


def _plot_bode(tc: TransientCapture, channel: str, order: int = 1) -> None:
    """Bode estilo System1 (mejorado): FASE arriba (lag, eje invertido) + AMPLITUD
    abajo vs rpm. Marca la velocidad crítica (pico), compensación de slow-roll
    opcional, amplitud por norma. Se llena durante arranque/parada."""
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    rpms, amp, phase = tc.bode(channel)
    if len(rpms) < 2:
        st.info("The Bode fills up during a **transient**. In **Source and parameters** "
                "pick a *runup* or *coastdown* profile, press ▶ Live, and you will see the curve "
                "build up as it passes through the critical speed.")
        return

    tmap = st.session_state.get("rm_type_by_name") or {}
    conv, norm, k0, _krms = _amp_conv(tmap.get(channel, "proximity"))
    chs = st.session_state.get("rm_channels") or []
    units = next((c.units for c in chs if c.name == channel), "mil")
    uu = _amp_unit(units, conv)
    machine = st.session_state.get("rm_machine_name", "—")

    rpms = np.asarray(rpms, float)
    lo_rpm, hi_rpm = float(rpms.min()), float(rpms.max())
    _op_nominal = float(st.session_state.get("rm_machine_rpm", 0) or 0)
    with st.container(key="rm_bode_ctrls", horizontal=True, vertical_alignment="center", gap="medium"):
        comp = st.toggle("Compensate slow-roll", key="rm_bode_comp",
                         help="Subtracts the slow-roll 1X vector (mechanical + electrical runout) from "
                              "ALL points → isolates the real dynamic response. Standard "
                              "API 684 / API 670 practice. Pick the slow-roll speed "
                              "(typically < 10% of the 1st critical, with the shaft nearly rigid).")
        sr_rpm = None
        if comp:
            sr_rpm = st.number_input("Slow-roll speed (rpm)", int(lo_rpm), int(hi_rpm),
                                     int(round(lo_rpm)), step=60, key="rm_bode_srrpm",
                                     help="Reference speed of the slow-roll vector.")
        # Velocidad de operación: arranca con la nominal de la config, editable
        # para evaluar el margen de separación a otra velocidad.
        op_rpm = float(st.number_input(
            "Operating speed (rpm)", 0, 60000, int(round(_op_nominal)) if _op_nominal > 0 else 3600,
            step=60, key="rm_bode_oprpm",
            help="Operating speed (nominal). Defaults to the config value; change it to "
                 "evaluate the separation margin at another speed. 0 = hide."))

    # Vector complejo 1X → compensación slow-roll (resta el vector de referencia).
    z = amp * np.exp(1j * np.radians(phase))
    zref = 0.0 + 0.0j
    if comp:
        tgt = float(sr_rpm) if sr_rpm else lo_rpm
        band = np.abs(rpms - tgt) <= max(60.0, 0.05 * tgt)
        zref = complex(np.mean(z[band])) if band.any() else z[int(np.argmin(np.abs(rpms - tgt)))]
        z = z - zref
    amp_disp = np.abs(z) * k0
    ph_disp = np.degrees(np.angle(z)) % 360.0          # 0-360 (lectura del cursor)
    # Fase DESENROLLADA para el trazo → curva continua (sin saltos de 360°),
    # como Bently. rpms viene ordenado ascendente, así que unwrap es válido.
    ph_plot = np.degrees(np.unwrap(np.angle(z)))
    ph_plot = ph_plot - float(np.round(ph_plot[0] / 360.0)) * 360.0   # arranca cerca de 0-360

    i_pk = int(np.argmax(amp_disp))
    ncrit = float(rpms[i_pk])
    a_pk = float(amp_disp[i_pk])
    ph_pk = float(ph_disp[i_pk])

    # Header autocontenido (tag · Bode · canal · orden · rango rpm · fecha).
    ts = datetime.now().strftime("%d %b %Y · %H:%M:%S")
    comp_txt = (f' · <span style="color:#c7d6ea">Comp {np.abs(zref) * k0:.3g} {uu} '
                f'∠{np.degrees(np.angle(zref)) % 360:.0f}°</span>') if comp else ''
    st.markdown(
        f'<div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;'
        f'gap:4px 18px;padding:7px 12px;background:{_S1_TITLE};border-radius:8px 8px 0 0;color:#fff;'
        f'font-size:12px;font-family:Arial,Helvetica,sans-serif">'
        f'<span><b>{machine}</b> · {channel} · Bode · <span style="color:#c7d6ea">{order}X</span>'
        f'{comp_txt}</span>'
        f'<span style="color:#9fb3d1">🕒 {ts}</span></div>', unsafe_allow_html=True)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05)
    # FASE arriba (System1). Eje invertido → el lag crece hacia abajo. Línea
    # continua sin marcadores (como Bently) — spline para suavizar.
    fig.add_trace(go.Scatter(x=rpms, y=ph_plot, mode="lines",
                             line=dict(width=1.8, color=_S1_BLUE, shape="spline", smoothing=0.6),
                             hovertemplate="%{x:.0f} rpm<br>%{y:.0f}°<extra></extra>"), row=1, col=1)
    # AMPLITUD abajo.
    fig.add_trace(go.Scatter(x=rpms, y=amp_disp, mode="lines",
                             line=dict(width=1.8, color=_S1_BLUE, shape="spline", smoothing=0.6),
                             hovertemplate=f"%{{x:.0f}} rpm<br>%{{y:.3g}} {uu}<extra></extra>"),
                  row=2, col=1)
    # Velocidad(es) crítica(s): picos locales prominentes (hasta 3), fusionando
    # los cercanos. Marca cada uno con línea + etiqueta Ncrit.
    rr, aa = np.asarray(rpms, float), np.asarray(amp_disp, float)
    mx = float(aa.max()) if len(aa) else 0.0
    cand = [i for i in range(1, len(aa) - 1)
            if aa[i] >= aa[i - 1] and aa[i] >= aa[i + 1] and aa[i] > 0.4 * mx]
    span = (rr[-1] - rr[0]) if len(rr) > 1 else 1.0
    crit_idx = []
    for i in cand:
        if crit_idx and abs(rr[i] - rr[crit_idx[-1]]) < 0.08 * span:
            if aa[i] > aa[crit_idx[-1]]:
                crit_idx[-1] = i
        else:
            crit_idx.append(i)
    # Quedarse con los más prominentes, pero ETIQUETAR por orden de rpm (1ª, 2ª…).
    crit_idx = sorted(sorted(crit_idx, key=lambda i: -aa[i])[:4], key=lambda i: rr[i]) or [i_pk]
    _ORD = ["1st", "2nd", "3rd", "4th", "5th"]
    af_by_crit = {}
    for k, i in enumerate(crit_idx):
        af = _half_power_af(rr, aa, i)
        af_by_crit[i] = af[0] if af else None
        for r in (1, 2):
            fig.add_vline(x=float(rr[i]), line=dict(color="#e26d6d", width=1, dash="dot"), row=r, col=1)
        lbl = f"{_ORD[k] if k < len(_ORD) else str(k+1)} critical ≈ {rr[i]:.0f} rpm"
        if af:
            lbl += f" · AF {af[0]:.1f}"
            # Banda de media potencia (visual del método API 684).
            fig.add_shape(type="line", x0=af[1], x1=af[2], y0=af[3], y1=af[3], row=2, col=1,
                          line=dict(color="#c0392b", width=1, dash="dot"))
        fig.add_annotation(x=float(rr[i]), y=float(aa[i]), row=2, col=1, text=lbl,
                           showarrow=True, arrowhead=2, arrowsize=0.8, arrowcolor="#c0392b",
                           ax=26, ay=-16, font=dict(size=10, color="#c0392b"),
                           bgcolor="rgba(255,255,255,0.85)")
    # Velocidad de operación + margen de separación (API 684). Roja si la máquina
    # queda en resonancia o con margen insuficiente; verde si cumple.
    op_status, op_msg = None, ""
    if op_rpm > 0 and len(crit_idx):
        j = min(crit_idx, key=lambda i: abs(rr[i] - op_rpm))
        ncj, afj = float(rr[j]), af_by_crit.get(j)
        sm = abs(ncj - op_rpm) / op_rpm * 100.0
        below = ncj < op_rpm
        req = 15.0 if below else 20.0                    # API 617/684: 15% abajo, 20% arriba
        needs_margin = (afj is None) or (afj >= 2.5)     # margen exigido si AF ≥ 2.5
        oi = crit_idx.index(j)
        ordl = _ORD[oi] if oi < len(_ORD) else f"{oi+1}th"
        af_txt = f", AF {afj:.1f}" if afj else ""
        if sm < 3.0:
            op_status = "res"
            op_msg = (f"⚠ **AT RESONANCE** — the operating speed ({op_rpm:.0f} rpm) coincides "
                      f"with the {ordl} critical ({ncj:.0f} rpm{af_txt}). Amplitude and phase very "
                      f"sensitive to any change; **API 684 does not allow operating on a critical**.")
        elif needs_margin and sm < req:
            op_status = "margin"
            op_msg = (f"⚠ **Insufficient margin** — operating at **{sm:.0f}%** of the {ordl} critical "
                      f"({ncj:.0f} rpm{af_txt}); API 684 requires **≥{req:.0f}%** "
                      f"({'critical below' if below else 'critical above'}) for "
                      f"lightly damped modes (AF≥2.5). Risk of high amplitudes near resonance.")
        else:
            op_status = "ok"
            op_msg = (f"✓ **Separation margin {sm:.0f}%** to the {ordl} critical ({ncj:.0f} rpm) — "
                      f"meets API 684 (≥{req:.0f}%).")
    op_col = "#dc2626" if op_status in ("res", "margin") else "#16a34a"
    if op_rpm > 0 and lo_rpm <= op_rpm <= hi_rpm:
        _bad = op_status in ("res", "margin")
        for r in (1, 2):
            fig.add_vline(x=op_rpm, line=dict(color=op_col, width=1.8 if _bad else 1.2,
                                              dash="solid" if _bad else "solid"), row=r, col=1)
        fig.add_annotation(x=op_rpm, y=1.0, yref="y2 domain", row=2, col=1, yanchor="bottom",
                           text=f"{'⚠ ' if _bad else ''}Op {op_rpm:.0f}", showarrow=False,
                           font=dict(size=9.5, color=op_col))

    ymax = _nice_top(a_pk * 1.15) if a_pk > 0 else 1.0
    fig.update_yaxes(title_text="Phase 1X (°)", autorange="reversed", dtick=90, row=1, col=1,
                     showgrid=True, gridcolor=_S1_GRID, showline=True, linecolor=_S1_AXIS,
                     ticks="outside", tickcolor=_S1_AXIS, zeroline=False)
    fig.update_yaxes(title_text=f"1X ({uu})", range=[0, ymax], row=2, col=1,
                     showgrid=True, gridcolor=_S1_GRID, showline=True, linecolor=_S1_AXIS,
                     ticks="outside", tickcolor=_S1_AXIS, zeroline=False)
    fig.update_xaxes(row=1, col=1, showgrid=True, gridcolor=_S1_GRID, showline=True,
                     linecolor=_S1_AXIS, ticks="outside", tickcolor=_S1_AXIS)
    fig.update_xaxes(title_text="RPM", rangemode="tozero", row=2, col=1, showgrid=True,
                     gridcolor=_S1_GRID, showline=True, linecolor=_S1_AXIS, ticks="outside",
                     tickcolor=_S1_AXIS)
    # Caja tipo cursor (arriba-derecha): resonancia principal (API 684).
    _kv = (lambda k, v: f'<b style="color:{_S1_TITLE}">{k}</b> <i style="color:#2f6fb0">{v}</i>')
    af_main = _half_power_af(rr, aa, i_pk)
    box = [_kv("Ncrit", f"{ncrit:.0f} rpm"), _kv("Amp", f"{a_pk:.3g} {uu}"),
           _kv("High spot", f"{ph_pk:.0f}°")]
    if af_main:
        box.append(_kv("AF", f"{af_main[0]:.1f}"))
    # En el subplot de AMPLITUD (arriba-derecha, zona vacía a alta rpm) para NO
    # tapar la curva de fase.
    fig.add_annotation(
        xref="x2 domain", yref="y2 domain", x=0.992, y=0.97, xanchor="right", yanchor="top",
        align="left", showarrow=False, text="<br>".join(box),
        font=dict(size=10.5, family="Arial, Helvetica, sans-serif"),
        bgcolor="rgba(244,249,255,0.94)", bordercolor="#2f6fb0", borderwidth=1.4, borderpad=6)
    fig.update_layout(height=480, margin=dict(l=58, r=16, t=8, b=40),
                      plot_bgcolor="#ffffff", paper_bgcolor="#ffffff", font=_S1_FONT, showlegend=False)
    st.plotly_chart(fig, use_container_width=True, config=_PLOTLY_CFG)

    # Banner de estado de operación (API 684): rojo si resonancia/margen insuf.
    if op_status in ("res", "margin"):
        st.error(op_msg)
    elif op_status == "ok":
        st.success(op_msg)
    st.caption("**High spot** = phase of the peak (where the shaft comes closest to the probe). "
               "Below the 1st critical, high spot ≈ **heavy spot** (unbalance); at the critical they turn "
               "~90°, above ~180° (API 684). **AF** = amplification factor by half power.")


def _plot_cascade(tc: TransientCapture, channel: str, rpm: Optional[float]) -> None:
    """Cascada estilo System1 mejorado: espectros APILADOS vs rpm (hidden-line),
    con líneas de orden diagonales (½X, 1X, 2X, 3X), críticas marcadas y amplitud
    por norma. Se llena durante un transitorio (runup/coastdown)."""
    import plotly.graph_objects as go
    rpms, freqs, mat = tc.cascade(channel)
    if len(rpms) < 2:
        st.info("The cascade fills up during a **transient** (runup/coastdown). "
                "Pick the profile in **Source and parameters** and press ▶ Live.")
        return
    tmap = st.session_state.get("rm_type_by_name") or {}
    conv, norm, k0, _krms = _amp_conv(tmap.get(channel, "proximity"))
    chs = st.session_state.get("rm_channels") or []
    units = next((c.units for c in chs if c.name == channel), "mil")
    uu = _amp_unit(units, conv)
    machine = st.session_state.get("rm_machine_name", "—")
    ts = datetime.now().strftime("%d %b %Y · %H:%M:%S")
    st.markdown(
        f'<div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;'
        f'gap:4px 18px;padding:7px 12px;background:{_S1_TITLE};border-radius:8px 8px 0 0;color:#fff;'
        f'font-size:12px;font-family:Arial,Helvetica,sans-serif">'
        f'<span><b>{machine}</b> · {channel} · Cascade · '
        f'<span style="color:#c7d6ea">{rpms.min():.0f}–{rpms.max():.0f} rpm</span></span>'
        f'<span style="color:#9fb3d1">🕒 {ts}</span></div>', unsafe_allow_html=True)

    # Unidad del eje de frecuencia: Hz (estructurales verticales), CPM, u Órdenes
    # (1X vertical — cómodo para seguir armónicos).
    with st.container(key="rm_casc_ctrls", horizontal=True, vertical_alignment="center", gap="medium"):
        xmode = st.radio("Frequency", ["Hz", "CPM", "Orders"], horizontal=True,
                         key="rm_casc_xunit", label_visibility="collapsed",
                         help="Hz: structural resonances stay vertical. Órdenes: the 1X "
                              "stays vertical (follows rpm).")
    mat = mat * k0                                   # amplitud en convención de norma
    fmax = float(freqs[-1]) if len(freqs) else 1.0
    # Submuestreo a ~60 espectros para el DIBUJO (más = mancha ilegible); el
    # diagnóstico y el Bode usan TODOS los puntos capturados, no estos.
    n = len(rpms)
    idx = np.unique(np.linspace(0, n - 1, min(60, n)).round().astype(int))
    rr = rpms[idx]
    MM = mat[idx]
    span = float(rr[-1] - rr[0]) or 1.0
    peak = float(MM.max()) or 1.0
    scale = (span / max(1, len(rr))) * 1.7 / peak    # alto de cada espectro (en rpm)

    if xmode == "Orders":
        _xf = lambda fr, rp: fr * 60.0 / rp if rp > 0 else fr
        xtitle, xhi = "Order (× rpm)", 6.0
    elif xmode == "CPM":
        _xf = lambda fr, rp: fr * 60.0
        xtitle, xhi = "Frequency (CPM)", fmax * 60.0
    else:
        _xf = lambda fr, rp: fr
        xtitle, xhi = "Frequency (Hz)", fmax

    fig = go.Figure()
    # Espectros apilados con HIDDEN-LINE: de atrás (alta rpm) a adelante (baja),
    # cada uno con relleno blanco que oculta los de atrás (look System1).
    for i in range(len(rr) - 1, -1, -1):
        base = float(rr[i])
        xv = _xf(freqs, base)
        y = base + MM[i] * scale
        fig.add_trace(go.Scatter(
            x=np.concatenate([xv, xv[::-1]]),
            y=np.concatenate([y, np.full(len(xv), base)]),
            fill="toself", fillcolor="rgba(255,255,255,0.97)", line=dict(width=0),
            hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(
            x=xv, y=y, mode="lines", line=dict(color=_S1_BLUE, width=0.9),
            hovertemplate=f"%{{x:.2f}} {xmode} · {base:.0f} rpm<extra></extra>", showlegend=False))
    # Líneas de orden: diagonales en Hz/CPM, VERTICALES en Órdenes.
    for k, lbl, col in [(0.5, "½X", "#94a3b8"), (1.0, "1X", "#e26d6d"),
                        (2.0, "2X", "#e0982a"), (3.0, "3X", "#8b5cf6")]:
        if xmode == "Orders":
            if k > xhi:
                continue
            fig.add_vline(x=k, line=dict(color=col, width=1, dash="dot"))
            fig.add_annotation(x=k, y=rr[-1], text=lbl, showarrow=False, yshift=9,
                               font=dict(size=9.5, color=col))
        else:
            fx = (k * rr / 60.0) * (60.0 if xmode == "CPM" else 1.0)   # k·rpm/60 en Hz/CPM
            m = fx <= xhi
            if m.sum() < 2:
                continue
            fig.add_trace(go.Scatter(x=fx[m], y=rr[m], mode="lines",
                          line=dict(color=col, width=1, dash="dot"), hoverinfo="skip", showlegend=False))
            fig.add_annotation(x=float(fx[m][-1]), y=float(rr[m][-1]), text=lbl, showarrow=False,
                               yshift=9, font=dict(size=9.5, color=col))
    # Críticas (picos 1X del Bode) como línea horizontal tenue.
    rb, ab, _ph = tc.bode(channel)
    crit_rpms = []
    if len(rb) >= 3:
        for ci in _detect_criticals(np.asarray(rb, float), np.asarray(ab, float)):
            yc = float(rb[ci])
            crit_rpms.append(yc)
            fig.add_hline(y=yc, line=dict(color="#e26d6d", width=1, dash="dot"),
                          annotation_text=f"Ncrit {yc:.0f}", annotation_position="right",
                          annotation_font=dict(size=9, color="#c0392b"))

    fig.update_layout(height=620, margin=dict(l=58, r=16, t=8, b=42),
                      plot_bgcolor="#ffffff", paper_bgcolor="#ffffff", font=_S1_FONT, showlegend=False)
    fig.update_xaxes(title_text=xtitle, range=[0, xhi], showgrid=True, gridcolor=_S1_GRID,
                     showline=True, linecolor=_S1_AXIS, ticks="outside", tickcolor=_S1_AXIS)
    fig.update_yaxes(title_text="RPM", range=[rr[0] - span * 0.02, rr[-1] + peak * scale * 1.1],
                     showgrid=True, gridcolor=_S1_GRID, showline=True, linecolor=_S1_AXIS,
                     ticks="outside", tickcolor=_S1_AXIS)
    st.plotly_chart(fig, use_container_width=True, config=_PLOTLY_CFG)
    st.caption(f"Cascade: each trace is a spectrum at its rpm (stacked). The **diagonals** are the "
               f"**orders** (½X, 1X, 2X, 3X): the peak rising along **1X** is unbalance/synchronous "
               f"response; energy at **½X** = instability (whirl/whip), at **2X** = "
               f"misalignment. Amplitude in {uu.split()[-1] if uu else 'pp'} (API 670).")

    # Auto-diagnóstico (modos + inestabilidad de película con nombre propio).
    findings = _cascade_diagnosis(rpms, freqs, mat, crit_rpms)
    if findings:
        st.markdown("**🔎 Auto-diagnosis (API 684)**")
        for lvl, title, detail in findings:
            body = f"**{title}** — {detail}"
            if lvl == "danger":
                st.error(body)
            elif lvl == "warn":
                st.warning(body)
            else:
                st.info(body)


def _plot_polar(tc: TransientCapture, channel: str, snap: np.ndarray, vib,
                fs: float, rpm: Optional[float]) -> None:
    """Polar / Nyquist del vector 1X vs velocidad (locus), estilo System1 mejorado:
    amplitud radial por norma (API 670 pp), fase angular, slow-roll compensable
    (API 684/670), críticas marcadas, velocidad de operación con alerta de
    resonancia (API 684), etiquetas de rpm, sentido de velocidad y caja de cursor."""
    import plotly.graph_objects as go
    rpms, amp, phase = tc.bode(channel)
    tmap = st.session_state.get("rm_type_by_name") or {}
    conv, norm, k0, _krms = _amp_conv(tmap.get(channel, "proximity"))
    chs = st.session_state.get("rm_channels") or []
    units = next((c.units for c in chs if c.name == channel), "mil")
    uu = _amp_unit(units, conv)
    machine = st.session_state.get("rm_machine_name", "—")
    _hdr = (lambda extra: st.markdown(
        f'<div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;'
        f'gap:4px 18px;padding:7px 12px;background:{_S1_TITLE};border-radius:8px 8px 0 0;color:#fff;'
        f'font-size:12px;font-family:Arial,Helvetica,sans-serif">'
        f'<span><b>{machine}</b> · {channel} · Polar{extra}</span>'
        f'<span style="color:#9fb3d1">🕒 {datetime.now().strftime("%d %b %Y · %H:%M:%S")}</span></div>',
        unsafe_allow_html=True))

    # --- Estacionario: solo el punto actual ---
    if len(rpms) < 2:
        _hdr(' · <span style="color:#c7d6ea">current point (run a runup for the locus)</span>')
        fig = go.Figure()
        name_to = {ch.name: (i, ch) for i, ch in vib}
        if channel in name_to and rpm:
            i, ch = name_to[channel]
            eu = snap[i] * 1000.0 / ch.sensitivity_mv_per_eu
            a, ph = one_x_vector(eu, fs, rpm / 60.0)
            fig.add_trace(go.Scatterpolar(r=[a * k0], theta=[ph], mode="markers+text",
                          text=[f"{a * k0:.3g} {uu} ∠{ph:.0f}°"], textposition="top center",
                          marker=dict(size=15, color=_S1_KPH)))
        fig.update_layout(height=460, margin=dict(l=30, r=30, t=10, b=20), showlegend=False,
                          paper_bgcolor="#ffffff", font=_S1_FONT,
                          polar=dict(bgcolor="#ffffff",
                                     angularaxis=dict(rotation=90, direction="clockwise")))
        st.plotly_chart(fig, use_container_width=True, config=_PLOTLY_CFG)
        return

    rpms = np.asarray(rpms, float)
    lo_rpm, hi_rpm = float(rpms.min()), float(rpms.max())
    _op_nominal = float(st.session_state.get("rm_machine_rpm", 0) or 0)
    with st.container(key="rm_polar_ctrls", horizontal=True, vertical_alignment="center", gap="medium"):
        comp = st.toggle("Compensate slow-roll", key="rm_polar_comp",
                         help="Subtracts the slow-roll 1X vector (runout) from the whole locus → real "
                              "dynamic response. API 684 / API 670.")
        sr_rpm = None
        if comp:
            sr_rpm = st.number_input("Slow-roll speed (rpm)", int(lo_rpm), int(hi_rpm),
                                     int(round(lo_rpm)), step=60, key="rm_polar_srrpm")
        op_rpm = float(st.number_input(
            "Operating speed (rpm)", 0, 60000, int(round(_op_nominal)) if _op_nominal > 0 else 3600,
            step=60, key="rm_polar_oprpm",
            help="Operating speed. Marks the point on the locus and evaluates resonance (API 684)."))

    z = np.asarray(amp, float) * np.exp(1j * np.radians(phase))
    zref = 0.0 + 0.0j
    if comp:
        tgt = float(sr_rpm) if sr_rpm else lo_rpm
        band = np.abs(rpms - tgt) <= max(60.0, 0.05 * tgt)
        zref = complex(np.mean(z[band])) if band.any() else z[int(np.argmin(np.abs(rpms - tgt)))]
        z = z - zref
    r = np.abs(z) * k0
    th = np.degrees(np.angle(z)) % 360.0

    # Orientación FÍSICA (como System1): el 0° de fase se ubica en el plano de la
    # sonda (ángulo real de montaje desde TDC), y la fase (lag) corre EN CONTRA
    # del sentido de giro. Así el vector apunta al high spot en la sección real.
    angmap = st.session_state.get("rm_angle_by_name") or {}
    beta = float(angmap.get(channel, 315.0 if "Y" in channel.upper() else 45.0))  # abs desde TDC (horario)
    rot_dir = (st.session_state.get("rm_machine_rotation", "CCW") or "CCW").upper()
    rot0 = (90.0 - beta) % 360.0                          # fase 0 → hacia la sonda
    ang_dir = "counterclockwise" if rot_dir == "CW" else "clockwise"   # lag contra el giro

    comp_txt = (f' · <span style="color:#c7d6ea">Comp {np.abs(zref) * k0:.3g} {uu} '
                f'∠{np.degrees(np.angle(zref)) % 360:.0f}°</span>') if comp else ''
    _hdr(f' · <span style="color:#c7d6ea">1X</span>{comp_txt}')

    # Críticas + AF
    crit_idx = _detect_criticals(rpms, r)
    af_by_crit = {i: (_half_power_af(rpms, r, i) or [None])[0] for i in crit_idx}
    i_pk = int(np.argmax(r))

    fig = go.Figure()
    # Locus 1X (línea aciano + puntos finos), color por rpm en el hover.
    fig.add_trace(go.Scatterpolar(
        r=r, theta=th, mode="lines+markers", line=dict(color=_S1_BLUE, width=1.6),
        marker=dict(size=3.5, color=_S1_KPH), customdata=rpms,
        hovertemplate="%{customdata:.0f} rpm<br>%{r:.3g} " + f"{uu}" +
                      "<br>%{theta:.0f}°<extra></extra>", showlegend=False))
    # Etiquetas de rpm cada N puntos, salteando el amontonamiento cerca del centro.
    rmx = float(r.max()) if len(r) else 1.0
    cand_lbl = [i for i in range(len(rpms)) if r[i] > 0.14 * rmx]
    step = max(1, len(cand_lbl) // 12)
    ii = cand_lbl[::step]
    if ii:
        fig.add_trace(go.Scatterpolar(r=r[ii], theta=th[ii], mode="text",
                      text=[f"{rpms[i]:.0f}" for i in ii], textposition="top center",
                      textfont=dict(size=8, color="#7a8699"), hoverinfo="skip", showlegend=False))
    # Crítica(s): estrella roja + etiqueta con AF.
    for i in crit_idx:
        afv = af_by_crit.get(i)
        fig.add_trace(go.Scatterpolar(
            r=[r[i]], theta=[th[i]], mode="markers+text", marker=dict(size=12, color="#dc2626", symbol="star"),
            text=[f"Ncrit {rpms[i]:.0f}" + (f" · AF {afv:.1f}" if afv else "")],
            textposition="middle right", textfont=dict(size=9, color="#c0392b"),
            hoverinfo="skip", showlegend=False))
    # Velocidad de operación: verde/roja según resonancia (API 684).
    op_stat = _op_margin(rpms, crit_idx, af_by_crit, op_rpm)
    if op_stat and lo_rpm <= op_rpm <= hi_rpm:
        jo = int(np.argmin(np.abs(rpms - op_rpm)))
        fig.add_trace(go.Scatterpolar(
            r=[r[jo]], theta=[th[jo]], mode="markers+text",
            marker=dict(size=12, color=op_stat["color"], symbol="circle",
                        line=dict(width=1.5, color="#fff")),
            text=[f"{'⚠ ' if op_stat['status'] != 'ok' else ''}Op {op_rpm:.0f}"],
            textposition="bottom center", textfont=dict(size=9, color=op_stat["color"]),
            hoverinfo="skip", showlegend=False))

    rpk = float(r.max()) if r.max() > 0 else 1.0
    rmax = _nice_top(rpk * 1.15)
    rlim = rmax * 1.16                       # anillo extra para la flecha de giro
    # Marca de la sonda en 0° (dónde está físicamente montada).
    fig.add_trace(go.Scatterpolar(r=[rmax], theta=[0], mode="markers+text",
                  marker=dict(size=10, color="#0F1E3D", symbol="triangle-up"),
                  text=[f" {channel}"], textposition="middle right",
                  textfont=dict(size=10, color="#0F1E3D"), hoverinfo="skip", showlegend=False))
    # Flecha del SENTIDO DE GIRO de la máquina (arriba, fuera del dato).
    mdir = 1 if rot_dir == "CCW" else -1                 # +1 = CCW en pantalla
    s = 1 if ang_dir == "counterclockwise" else -1
    scr = np.linspace(62, 118, 22) if mdir > 0 else np.linspace(118, 62, 22)   # ángulos de pantalla
    th_arc = (scr - rot0) / s
    fig.add_trace(go.Scatterpolar(r=[rmax * 1.09] * len(scr), theta=th_arc, mode="lines",
                  line=dict(color="#0F1E3D", width=2.6), hoverinfo="skip", showlegend=False,
                  cliponaxis=False))
    _tang = scr[-1] + 90 * mdir
    fig.add_trace(go.Scatterpolar(r=[rmax * 1.09], theta=[th_arc[-1]], mode="markers",
                  marker=dict(size=15, color="#0F1E3D", symbol="triangle-up", angle=90 - _tang),
                  hoverinfo="skip", showlegend=False, cliponaxis=False))
    fig.add_trace(go.Scatterpolar(r=[rmax * 1.15], theta=[(90 - rot0) / s], mode="text",
                  text=[f"rotation {rot_dir}"], textfont=dict(size=10, color="#0F1E3D"),
                  hoverinfo="skip", showlegend=False, cliponaxis=False))
    fig.update_layout(height=560, margin=dict(l=30, r=30, t=10, b=20), showlegend=False,
                      paper_bgcolor="#ffffff", font=_S1_FONT,
                      polar=dict(bgcolor="#ffffff",
                                 radialaxis=dict(range=[0, rlim], gridcolor=_S1_GRID, angle=rot0,
                                                 tickangle=0, nticks=6, tickfont=dict(size=8.5, color="#9aa6b6"),
                                                 title=dict(text="")),
                                 angularaxis=dict(rotation=rot0, direction=ang_dir, dtick=30,
                                                  gridcolor=_S1_GRID, tickfont=dict(size=9),
                                                  ticksuffix="°", linecolor=_S1_AXIS)))
    # Caja de cursor (pico de resonancia) — esquina, en coords de papel.
    _kv = (lambda k, v: f'<b style="color:{_S1_TITLE}">{k}</b> <i style="color:#2f6fb0">{v}</i>')
    af_main = _half_power_af(rpms, r, i_pk)
    box = [_kv("Ncrit", f"{rpms[i_pk]:.0f} rpm"), _kv("Amp", f"{r[i_pk]:.3g} {uu}"),
           _kv("Phase", f"{th[i_pk]:.0f}°")]
    if af_main:
        box.append(_kv("AF", f"{af_main[0]:.1f}"))
    fig.add_annotation(xref="paper", yref="paper", x=0.005, y=0.99, xanchor="left", yanchor="top",
                       align="left", showarrow=False, text="<br>".join(box),
                       font=dict(size=10.5, family="Arial, Helvetica, sans-serif"),
                       bgcolor="rgba(244,249,255,0.94)", bordercolor="#2f6fb0", borderwidth=1.4, borderpad=6)
    # Nota de orientación física.
    fig.add_annotation(xref="paper", yref="paper", x=0.99, y=0.99, xanchor="right", yanchor="top",
                       showarrow=False,
                       text=f"0° = probe {channel} · phase against rotation ({rot_dir})",
                       font=dict(size=9.5, color="#64748b"))
    st.plotly_chart(fig, use_container_width=True, config=_PLOTLY_CFG)

    if op_stat and op_stat["status"] in ("res", "margin"):
        st.error(op_stat["msg"])
    elif op_stat and op_stat["status"] == "ok":
        st.success(op_stat["msg"])
    st.caption(f"1X locus oriented to the **real plane of probe {channel}** (0° = its mounting "
               f"angle from TDC) with **phase growing against the direction of rotation** "
               f"({rot_dir}). The **loop** marks the resonance; the vector "
               f"points to the high spot. Radial amplitude in **{uu.split()[-1] if uu else 'pp'}** "
               f"(API 670); slow-roll and operation per **API 684**.")


def _plot_shaft_centerline(snap: np.ndarray, vib) -> None:
    """Shaft centerline: posición media del eje (gap DC) en el par X/Y."""
    import plotly.graph_objects as go
    name_to = {ch.name: (i, ch) for i, ch in vib}
    saved = st.session_state.get("rm_pairs_saved") or []
    valid = [(a, b) for a, b in saved if a in name_to and b in name_to]
    if not valid:
        st.info("Configure X/Y pairs in **Configuration → Channel editor → X/Y pair** for the shaft centerline.")
        return
    fig = go.Figure()
    th = np.linspace(0, 2 * np.pi, 120)
    for a, b in valid:
        is_y = lambda n: "Y" in n.upper()
        yn, xn = (a, b) if is_y(a) and not is_y(b) else ((b, a) if is_y(b) else (a, b))
        yi, chy = name_to[yn]
        xi, chx = name_to[xn]
        gy = float(np.mean(snap[yi] * 1000.0 / chy.sensitivity_mv_per_eu))
        gx = float(np.mean(snap[xi] * 1000.0 / chx.sensitivity_mv_per_eu))
        fig.add_trace(go.Scatter(x=[gx], y=[gy], mode="markers+text", text=[f"{xn}/{yn}"],
                                 textposition="top center", marker=dict(size=15, color="#2563eb")))
    fig.add_trace(go.Scatter(x=np.cos(th) * 0.01, y=np.sin(th) * 0.01, mode="lines",
                             line=dict(color="#e2e8f0"), hoverinfo="skip"))
    fig.update_layout(height=440, title="Shaft Centerline (shaft position)",
                      xaxis_title="X", yaxis_title="Y", showlegend=False,
                      yaxis=dict(scaleanchor="x", scaleratio=1))
    st.plotly_chart(fig, use_container_width=True, config=_PLOTLY_CFG)
    st.caption("⚠ Requires DC gap. With AC-coupled channels the static position is ~0. "
               "For a real Shaft Centerline the proximitor's DC gap is needed.")


def _hexlerp(c1: str, c2: str, t: float) -> str:
    a, b = int(c1[1:], 16), int(c2[1:], 16)
    ar, ag, ab = (a >> 16) & 255, (a >> 8) & 255, a & 255
    br, bg, bb = (b >> 16) & 255, (b >> 8) & 255, b & 255
    return f"#{round(ar+(br-ar)*t):02x}{round(ag+(bg-ag)*t):02x}{round(ab+(bb-ab)*t):02x}"


def _plot_waterfall(tc: TransientCapture, channel: str) -> None:
    """Waterfall 3D estilo System1 mejorado: espectros apilados en 3D (traza por
    rpm, gradiente por velocidad), líneas de orden en el piso, selector
    Hz/CPM/Órdenes, amplitud por norma. Transitorio (runup/coastdown)."""
    import plotly.graph_objects as go
    rpms, freqs, mat = tc.cascade(channel)
    if len(rpms) < 2:
        st.info("The Waterfall fills up during a **transient** (runup/coastdown). Run one in Monitoreo.")
        return
    tmap = st.session_state.get("rm_type_by_name") or {}
    conv, norm, k0, _krms = _amp_conv(tmap.get(channel, "proximity"))
    chs = st.session_state.get("rm_channels") or []
    units = next((c.units for c in chs if c.name == channel), "mil")
    uu = _amp_unit(units, conv)
    machine = st.session_state.get("rm_machine_name", "—")
    ts = datetime.now().strftime("%d %b %Y · %H:%M:%S")
    st.markdown(
        f'<div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;'
        f'gap:4px 18px;padding:7px 12px;background:{_S1_TITLE};border-radius:8px 8px 0 0;color:#fff;'
        f'font-size:12px;font-family:Arial,Helvetica,sans-serif">'
        f'<span><b>{machine}</b> · {channel} · Waterfall 3D · '
        f'<span style="color:#c7d6ea">{rpms.min():.0f}–{rpms.max():.0f} rpm</span></span>'
        f'<span style="color:#9fb3d1">🕒 {ts}</span></div>', unsafe_allow_html=True)

    with st.container(key="rm_wf3_ctrls", horizontal=True, vertical_alignment="center", gap="medium"):
        xmode = st.radio("Frequency", ["Hz", "CPM", "Orders"], horizontal=True,
                         key="rm_wf3_xunit", label_visibility="collapsed")
    mat = mat * k0
    fmax = float(freqs[-1]) if len(freqs) else 1.0
    n = len(rpms)
    idx = np.unique(np.linspace(0, n - 1, min(80, n)).round().astype(int))
    rr, MM = rpms[idx], mat[idx]
    tmin, tmax = float(rr[0]), float(rr[-1])
    if xmode == "Orders":
        _xf = lambda fr, rp: fr * 60.0 / rp if rp > 0 else fr
        xtitle, xhi = "Order (× rpm)", 6.0
    elif xmode == "CPM":
        _xf = lambda fr, rp: fr * 60.0
        xtitle, xhi = "Frequency (CPM)", fmax * 60.0
    else:
        _xf = lambda fr, rp: fr
        xtitle, xhi = "Frequency (Hz)", fmax

    fig = go.Figure()
    for i in range(len(rr)):
        frac = (rr[i] - tmin) / ((tmax - tmin) or 1.0)
        fig.add_trace(go.Scatter3d(
            x=_xf(freqs, rr[i]), y=np.full(len(freqs), rr[i]), z=MM[i], mode="lines",
            line=dict(color=_hexlerp("#7fb3ea", "#0F1E3D", frac), width=2.5),
            hovertemplate=f"%{{x:.1f}} {xmode} · {rr[i]:.0f} rpm · %{{z:.3g}} {uu}<extra></extra>",
            showlegend=False))
    # Líneas de orden en el piso (z=0).
    for k, lbl, col in [(1.0, "1X", "#e26d6d"), (2.0, "2X", "#e0982a"), (3.0, "3X", "#8b5cf6")]:
        if xmode == "Orders":
            if k > xhi:
                continue
            fig.add_trace(go.Scatter3d(x=[k, k], y=[rr[0], rr[-1]], z=[0, 0], mode="lines",
                          line=dict(color=col, width=3, dash="dot"), hoverinfo="skip", showlegend=False))
        else:
            fx = (k * rr / 60.0) * (60.0 if xmode == "CPM" else 1.0)
            m = fx <= xhi
            if m.sum() < 2:
                continue
            fig.add_trace(go.Scatter3d(x=fx[m], y=rr[m], z=np.zeros(int(m.sum())), mode="lines",
                          line=dict(color=col, width=3, dash="dot"), hoverinfo="skip", showlegend=False))

    fig.update_layout(height=720, margin=dict(l=0, r=0, t=4, b=0), paper_bgcolor="#ffffff",
                      font=_S1_FONT, showlegend=False,
                      scene=dict(
                          xaxis=dict(title=xtitle, range=[0, xhi], backgroundcolor="#ffffff",
                                     gridcolor="#e6ecf5", zerolinecolor="#cdd7e6"),
                          yaxis=dict(title="RPM", backgroundcolor="#ffffff", gridcolor="#e6ecf5",
                                     zerolinecolor="#cdd7e6"),
                          zaxis=dict(title=uu, backgroundcolor="#ffffff", gridcolor="#e6ecf5",
                                     zerolinecolor="#cdd7e6"),
                          aspectmode="manual", aspectratio=dict(x=1.25, y=2.4, z=0.42),
                          # Vista clásica de waterfall: casi de costado y baja →
                          # los picos se alinean en ridges diagonales (como System1).
                          camera=dict(eye=dict(x=1.35, y=-2.05, z=0.4),
                                      up=dict(x=0, y=0, z=1),
                                      center=dict(x=0, y=0, z=-0.12))))
    st.plotly_chart(fig, use_container_width=True, config=_PLOTLY_CFG)
    st.caption(f"Waterfall 3D: each trace = a spectrum at its rpm (depth), colored by speed. "
               f"The floor lines are the **orders**; the ridge rising along **1X** = synchronous, "
               f"the **fixed in frequency** = structural, the **subsynchronous** = instability. "
               f"Amplitude in {uu.split()[-1] if uu else 'pp'} (API 670). Drag to rotate.")


def _save_snapshot(agent: AcqAgent, snap: np.ndarray, rpm: Optional[float]) -> None:
    try:
        from core.remote_monitoring.store import LocalStore
        store = st.session_state.setdefault("rm_store", LocalStore())
        ch_meta = [{"name": ch.name, "bnc_port": ch.bnc_port, "coupling": ch.coupling,
                    "sensitivity_mv_per_eu": float(ch.sensitivity_mv_per_eu or 0.0),
                    "units": ch.units} for ch in agent.channels]
        meta = store.save_snapshot(agent.instance_id, snap, ch_meta, agent.sample_rate_hz,
                                   rpm=rpm, captured_at=datetime.now(timezone.utc).isoformat())
        st.session_state["rm_saved_count"] = int(st.session_state.get("rm_saved_count", 0)) + 1
        st.success(f"💾 Saved offline: {meta.snapshot_id} "
                   f"({store.count(only_pending=True)} pending sync)")
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not save: {type(e).__name__}: {e}")
