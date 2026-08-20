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
                    subtitle="Adquisición dinámica en vivo · rotordinámica · ISO 20816 / API 670 / API 684")
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
        .st-key-rm_trend_ctrls [role="radiogroup"] { gap:2px 12px !important; align-items:center; }
        .st-key-rm_trend_ctrls [role="radiogroup"] label p { font-size:12px !important; }
        .st-key-rm_trend_ctrls [role="radiogroup"] label { padding:1px 0 !important; }
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
        </style>
    """, unsafe_allow_html=True)

    view = st.radio("Vista", ["Configuración", "Monitoreo", "Análisis"], horizontal=True,
                    key="rm_view", label_visibility="collapsed")
    # Hairline sutil (sin el st.divider() que gasta mucho espacio vertical).
    st.markdown('<hr style="margin:2px 0 10px;border:none;border-top:1px solid #e6ecf5">',
                unsafe_allow_html=True)
    if view == "Configuración":
        from core.remote_monitoring.ui_setup import render_setup
        render_setup()
    elif view == "Monitoreo":
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
    )
    if source_kind == "Campo (planta)":
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
    st.info("No hay configuración activa. Andá a **Configuración** y guardá una máquina, "
            "o cargá un layout demo para probar ya.")
    if st.button("Cargar layout demo (4 cojinetes + keyphasor)"):
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


def _ensure_agent() -> Optional[AcqAgent]:
    """Construye/devuelve el agente desde la config activa + fuente/params
    guardados en session. Compartido por Monitoreo y Análisis."""
    channels = st.session_state.get("rm_channels")
    if not channels:
        return None
    default_rpm = float(st.session_state.get("rm_machine_rpm", 3600.0))
    source_kind = st.session_state.get("rm_source_kind", "Simulado (dev/Mac)")
    fs = int(st.session_state.get("rm_fs", 5120))
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
        st.session_state["rm_transient"] = TransientCapture(TransientConfig(fmax_hz=_fmax))
        st.session_state["rm_prev_rpm"] = None
    return st.session_state["rm_agent"]


def _acquire(agent: AcqAgent, pump_n: int):
    """Bombea (si pump_n>0), toma snapshot, estima rpm/estado, alimenta transitorio.
    Devuelve (snap, rpm, state, tc, err, vib_channels)."""
    err = None
    try:
        if pump_n:
            agent.pump(pump_n)
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
        tc.feed(snap, rpm, agent.sample_rate_hz, vib)
    return snap, rpm, state, tc, err, vib


def _render_source_params() -> None:
    """Fuente y parámetros (solo en Monitoreo). Guarda en session."""
    default_rpm = float(st.session_state.get("rm_machine_rpm", 3600.0))
    with st.expander("⚙️ Fuente y parámetros", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            source_kind = st.selectbox("Fuente de datos", ["Simulado", "Campo (planta)"],
                                       key="rm_src_kind",
                                       help="Campo = adquisición en sitio. Simulado = pruebas.")
        with col2:
            fs = st.select_slider("Sample rate (Hz)", options=[2560, 5120, 10240, 25600, 51200],
                                  value=int(st.session_state.get("rm_fs", 5120)), key="rm_fs_w")
        with col3:
            defect = (st.selectbox("Defecto simulado", ["none", "unbalance", "misalignment"], key="rm_defect")
                      if source_kind.startswith("Simulado") else "none")
        sim = {"rpm": default_rpm, "defect": defect, "speed_profile": "constant"}
        if source_kind.startswith("Simulado"):
            st.markdown("**Modo de la máquina** (define el muestreo)")
            _MODE_MAP = {"Estable": "constant", "Arranque": "runup",
                         "Parada": "coastdown", "Arranque + Parada": "runup_coastdown"}
            p1, p2, p3, p4 = st.columns(4)
            with p1:
                mode_es = st.selectbox("Modo", list(_MODE_MAP.keys()), key="rm_prof")
            prof = _MODE_MAP[mode_es]
            sim["speed_profile"] = prof
            if prof == "constant":
                with p2:
                    sim["rpm"] = st.number_input("RPM", 300, 30000, int(default_rpm), step=60, key="rm_simrpm")
            else:
                with p2:
                    sim["rpm_start"] = st.number_input("RPM inicio", 0, 30000, 600, step=60, key="rm_rpmst")
                with p3:
                    sim["rpm_end"] = st.number_input("RPM fin", 0, 30000, 6000, step=60, key="rm_rpmend")
                with p4:
                    sim["sim_critical_rpm"] = st.number_input("Crítica (rpm)", 0, 30000, 3000, step=60, key="rm_crit")
                sim["ramp_seconds"] = st.slider("Duración (s)", 5, 120, 30, key="rm_ramp")
            st.caption("**Estable**: velocidad constante (estado estacionario, muestreo continuo). "
                       "**Arranque / Parada**: barrido de velocidad → captura **transitoria** "
                       "(finer, por Δrpm) para bode / cascade / waterfall.")
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
    st.caption(f"Máquina activa: **{machine_name}** · {len(channels)} canales "
               f"({n_vib} de vibración) · fuente lista para adquirir.")

    _render_source_params()
    agent = _ensure_agent()

    b1, b2, b3, b4, b5, b6 = st.columns([1, 1, 1.3, 1.3, 1, 1.6])
    with b1:
        if st.button("▶ Iniciar", use_container_width=True):
            st.session_state["rm_running"] = True
    with b2:
        if st.button("⏸ Detener", use_container_width=True):
            st.session_state["rm_running"] = False
    with b3:
        capture = st.button("📸 Capturar", use_container_width=True, type="primary",
                            help="Toma una lectura fresca y la guarda en un solo clic.")
    with b4:
        take = st.button("🔄 Tomar 1 lectura", use_container_width=True,
                         help="Refresca una lectura sin guardarla.")
    with b5:
        save = st.button("💾 Guardar", use_container_width=True,
                         help="Guarda la ventana actual sin tomar datos nuevos.")
    with b6:
        live = st.checkbox("🟢 Live (auto-refresh)", value=st.session_state.get("rm_running", False))
        st.session_state["rm_running"] = live

    # Acciones one-shot en el run principal (fuera del fragment).
    if capture:
        try:
            agent.pump(8)
            _s = agent.snapshot()
            if _s.shape[1]:
                _save_snapshot(agent, _s, agent.estimate_rpm(_s))
            else:
                st.warning("Sin datos para capturar. Pulsá ▶ Iniciar primero.")
        except Exception as e:  # noqa: BLE001
            st.session_state["rm_running"] = False
            st.error(f"⚠ No se pudo capturar: {type(e).__name__}: {e}")
    if take:
        try:
            agent.pump(8)
        except Exception as e:  # noqa: BLE001
            st.session_state["rm_running"] = False
            st.error(f"⚠ No se pudo adquirir: {type(e).__name__}: {e}")
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
    if live:
        try:
            agent.pump(4)
        except Exception as e:  # noqa: BLE001
            st.session_state["rm_running"] = False
            st.error(f"⚠ {type(e).__name__}: {e}")
            return
    snap = agent.snapshot()
    if snap.shape[1] == 0:
        st.info("Sin datos aún. Pulsá **▶ Iniciar** o **Tomar 1 lectura**.")
        return
    rpm = agent.estimate_rpm(snap)
    vib = [(i, ch) for i, ch in enumerate(agent.channels) if not is_keyphasor_channel(ch)]
    state = rm_states.classify_state(rpm, st.session_state.get("rm_prev_rpm"))
    st.session_state["rm_prev_rpm"] = rpm
    tc = st.session_state.setdefault("rm_transient", TransientCapture())
    if rpm:
        tc.feed(snap, rpm, agent.sample_rate_hz, vib)
    _render_stat_strip(agent, snap, rpm, state, tc)
    st.markdown("##### Tabular list — valores actuales")
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
        take = st.button("🔄 Actualizar", use_container_width=True,
                         help="Refresca una lectura (útil con Live apagado).")
    with top[1]:
        live = st.checkbox("🟢 Live", value=st.session_state.get("rm_running", False),
                           help="Auto-refresco de los gráficos.")
        st.session_state["rm_running"] = live
    if take:
        try:
            agent.pump(8)
        except Exception:  # noqa: BLE001
            pass
    st.fragment(_analisis_display, run_every=(0.5 if live else None))()


def _analisis_display() -> None:
    agent = st.session_state.get("rm_agent")
    if agent is None:
        return
    live = st.session_state.get("rm_running", False)
    if live:
        try:
            agent.pump(4)
        except Exception:  # noqa: BLE001
            st.session_state["rm_running"] = False
    snap = agent.snapshot()
    if snap.shape[1] == 0:
        st.info("Sin datos. Andá a **Monitoreo** y pulsá **▶ Iniciar**.")
        return
    rpm = agent.estimate_rpm(snap)
    vib = [(i, ch) for i, ch in enumerate(agent.channels) if not is_keyphasor_channel(ch)]
    state = rm_states.classify_state(rpm, st.session_state.get("rm_prev_rpm"))
    st.session_state["rm_prev_rpm"] = rpm
    tc = st.session_state.setdefault("rm_transient", TransientCapture())
    if rpm:
        tc.feed(snap, rpm, agent.sample_rate_hz, vib)
    fs = agent.sample_rate_hz
    names = [ch.name for _, ch in vib]
    _vent = snap.shape[1] / fs
    # (El contexto RPM/estado/ventana vive en el header de cada gráfico,
    #  no en una tira global — para que cada gráfico sea autocontenido.)

    tabs = st.tabs(["Tabular", "Tendencias", "Formas de onda", "Espectro",
                    "Órbitas", "Bode", "Polar", "Shaft Centerline",
                    "Cascada", "Waterfall"])
    with tabs[0]:
        _render_tabular_list(agent, snap, rpm, vib)
    with tabs[1]:
        if names:
            tmap = st.session_state.get("rm_type_by_name") or {}
            # Selección de canales + rango van en UNA fila abajo (dentro de _plot_trend).
            _plot_trend(snap, vib, tmap, rpm=rpm)
    with tabs[2]:
        if names:
            sels = st.multiselect("Canales", names, default=[names[0]], key="rm_wf_ch",
                                  help="Elegí uno o varios canales para ver sus formas de onda apiladas.")
            if sels:
                chans = [(snap[vib[names.index(s)][0]], vib[names.index(s)][1]) for s in sels]
                _plot_waveform(chans, fs, rpm, rm_states.state_label(state),
                               rm_states.state_color(state), _vent)
            else:
                st.info("Elegí al menos un canal.")
    with tabs[3]:
        if names:
            sels = st.multiselect("Canales", names, default=[names[0]], key="rm_sp_ch",
                                  help="Uno o varios canales — espectros apilados.")
            if sels:
                chans = [(snap[vib[names.index(s)][0]], vib[names.index(s)][1]) for s in sels]
                _plot_spectrum(chans, fs, rpm)
            else:
                st.info("Elegí al menos un canal.")
    with tabs[4]:
        _plot_orbit(snap, vib, fs, rpm)
    with tabs[5]:
        if names:
            sel = st.selectbox("Canal", names, key="rm_bode_ch")
            _plot_bode(tc, sel)
    with tabs[6]:
        if names:
            sel = st.selectbox("Canal", names, key="rm_polar_ch")
            _plot_polar(tc, sel, snap, vib, fs, rpm)
    with tabs[7]:
        _plot_shaft_centerline(snap, vib)
    with tabs[8]:
        if names:
            sel = st.selectbox("Canal", names, key="rm_casc_ch")
            _plot_cascade(tc, sel, rpm)
    with tabs[9]:
        if names:
            sel = st.selectbox("Canal", names, key="rm_wf3_ch")
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
            heads += [f"{o:g}X", f"{o:g}X fase"]
    heads += ["Alarma", "Danger", "Estado"]
    th = "".join(f'<th style="padding:9px 12px;text-align:left;font-size:11px;'
                 f'text-transform:uppercase;font-weight:700;color:{CYAN};white-space:nowrap;">{h}</th>'
                 for h in heads)

    body = []
    for k, (i, ch) in enumerate(vib):
        eu = snap[i] * 1000.0 / ch.sensitivity_mv_per_eu
        conv, _norm, k0, krms = _amp_conv(tmap.get(ch.name, "proximity"))
        u = f"{ch.units} {conv}"
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
        st.caption("Sin keyphasor no hay vectores 1X/2X — solo Overall. Activá el keyphasor en Configuración.")
    st.markdown(
        f'<div style="border:1px solid #d6deea;border-radius:12px;overflow-x:auto;'
        f'box-shadow:0 6px 18px rgba(15,30,61,.08)">'
        f'<table style="width:100%;border-collapse:collapse;min-width:560px">'
        f'<thead><tr style="background:{NAVY}">{th}</tr></thead>'
        f'<tbody>{"".join(body)}</tbody></table></div>', unsafe_allow_html=True)
    st.caption("Amplitudes según norma: **desplazamiento en pp** (API 670 · ISO 7919), "
               "**velocidad/aceleración en RMS** (ISO 20816). Overall y alarma/danger "
               "en la misma convención del sensor.")


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
             tip="Velocidad estimada por el keyphasor."),
        cell("1X", f"{rpm/60:.1f} Hz" if rpm else "—",
             tip="Frecuencia de giro (RPM/60)."),
        cell("Estado", rm_states.state_label(state), vcolor=color,
             tip="Estable / Arranque / Parada según el cambio de RPM."),
        cell("Ventana", f"{vent_s:.1f} s",
             tip=f"Buffer rodante: se guardan los últimos {vent_s:.0f} s de forma de onda. "
                 f"Sobre esta ventana se calculan espectro, órbita y formas de onda."),
        cell("Samples", f"{total_samples:,}",
             tip=f"Muestras totales adquiridas desde ▶ Iniciar "
                 f"(bloques de {block_s:g} s a {fs:g} Hz)."),
        cell("Vectores", f"{tc.n_samples}",
             tip="Puntos de velocidad capturados para Bode/Cascada. Solo crece durante "
                 "un transitorio (arranque/parada); en estable se queda en 1."),
        cell("Guardados", f"{saved}",
             tip="Formas de onda guardadas a disco con el botón 💾 Guardar."),
        cell("Tamaño", f"{size_mb:.2f} MB",
             tip="Memoria que ocupa la ventana actual en RAM (canales × muestras × 8 bytes)."),
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
               f'Estado <b style="color:{state_col}">{state_lbl}</b> · '
               f'Ventana <b style="color:#fff">{vent_s:.1f} s</b></span>')
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
                             uconv=conv, norm=norm, freq_unit=freq_unit, xmin=xmin, xmax=xmax,
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
        f'<span><b>{machine}</b> · Espectro · {chlabel}'
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
            hovertemplate=(f"%{{x:.0f}} {p['unit']}<br>%{{y:.4g}} {p['ch'].units} {p['uconv']}"
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
        fig.update_yaxes(title_text=f"{p['ch'].name} ({p['ch'].units} {p['uconv']})", range=[0, ymax],
                         showgrid=True, gridcolor=_S1_GRID, showline=True, linecolor=_S1_AXIS,
                         ticks="outside", tickcolor=_S1_AXIS, row=r, col=1)
        fig.update_xaxes(range=[p["xmin"], p["xmax"]], showgrid=True, gridcolor=_S1_GRID,
                         showline=True, linecolor=_S1_AXIS, ticks="outside", tickcolor=_S1_AXIS,
                         showspikes=True, spikecolor="#94a3b8", spikemode="across",
                         spikesnap="cursor", row=r, col=1,
                         title_text=(f"Frecuencia ({p['unit']})" if r == rows_n else None))
        # Caja de ARMÓNICOS por canal (arriba-derecha del subplot).
        _sfx = "" if r == 1 else str(r)
        hrows = [_kv("O/All", f"{p['ov']:.3g} {p['ch'].units} {p['uconv']}")]
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
    st.caption(f"Amplitud según norma — {_norms}.")


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
        st.info("La órbita necesita un par X/Y. Asocialo en **Configuración → Par X/Y**.")
        return
    name_to = {ch.name: (i, ch) for i, ch in vib_channels}
    saved = st.session_state.get("rm_pairs_saved") or []
    valid = [(a, b) for a, b in saved if a in name_to and b in name_to]
    if not valid:
        names = [ch.name for _, ch in vib_channels]
        valid = [(names[i], names[i + 1]) for i in range(0, len(vib_channels) - 1, 2)]
    if not valid:
        st.info("La órbita necesita un par X/Y. Asocialo en **Configuración → Par X/Y**.")
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
        sels = st.multiselect("Pares", labels, key="rm_orbit_pairs",
                              label_visibility="collapsed", placeholder="Órbitas…")
        fmode = st.radio("Filtro", ["Directa", "1X", "2X"], horizontal=True,
                         key="rm_orbit_filter", label_visibility="collapsed",
                         help="Directa = onda completa. 1X/2X = órbita filtrada al orden (elipse).")
        n_rev = 12
        if fmode == "Directa":
            n_rev = int(st.number_input("Vueltas", 3, 60, 12, step=1, key="rm_orbit_revs",
                                        label_visibility="collapsed"))
        kphline = st.toggle("Unir keyphasor", key="rm_orbit_kphline",
                            help="Une los keyphasor vuelta a vuelta (locus). Visible en "
                                 "transitorios; en régimen estable coinciden en un punto.")
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
        if fmode == "Directa" or not f1:
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
                    u=f"{chy.units} {conv}", xpp=float(np.ptp(x)), ypp=float(np.ptp(y)),
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
        f'<span><b>{machine}</b> · Órbita · <span style="color:#c7d6ea">{fmode}</span>'
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
        st.warning("Sin keyphasor no hay vectores. Activá el keyphasor en Configuración.")
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
    st.caption(f"Vectores síncronos a {', '.join(f'{o:g}X' for o in orders)} "
               f"(referenciados al keyphasor). 1X = {f1:.1f} Hz.")


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
        st.info("No hay canales de vibración.")
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

    # Ventana de tiempo. "Actual" = período en vivo de 15 min (sin selector).
    _unit = {"Horas": timedelta(hours=1), "Días": timedelta(days=1),
             "Semanas": timedelta(weeks=1), "Meses": timedelta(days=30)}
    _dfl = {"Horas": 6, "Días": 7, "Semanas": 4, "Meses": 6}
    sel = st.session_state.get("rm_trend_win", "Actual")
    if sel == "Actual":
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
        f'<span><b>{machine}</b> · Tendencia · {", ".join(plotted)}'
        + (f' · <span style="color:#c7d6ea">{rpm:.0f} rpm</span>' if rpm else '') + '</span>'
        f'<span style="color:#9fb3d1">🕒 {ts}</span></div>', unsafe_allow_html=True)

    fig = go.Figure()
    for k, ch in enumerate(plot_chs):
        fig.add_trace(go.Scatter(
            x=xs, y=[h[1].get(ch.name) for h in hist], mode="lines", name=ch.name,
            line=dict(width=1.6, color=_TREND_BRIGHT[k % len(_TREND_BRIGHT)])))
    for lvl, col, lbl in [(min(als) if als else None, "#f59e0b", "Alarma"),
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
    if sel != "Actual":
        fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.07, bgcolor="#f6f8fc",
                                          bordercolor="#d7deea", borderwidth=1))
    fig.update_yaxes(range=[0, ymax], rangemode="tozero", showgrid=True, gridcolor=_S1_GRID,
                     showline=True, linecolor=_S1_AXIS, ticks="outside", tickcolor=_S1_AXIS)
    st.plotly_chart(fig, use_container_width=True, config=_PLOTLY_CFG)

    if excluded:
        st.caption(f"⚠ No se pueden mezclar: **{', '.join(excluded)}** tiene distinta "
                   f"unidad o alarma/danger que **{plotted[0]}**. Se quitó de la selección.")

    # Controles en UNA fila abajo: canales + rango + cantidad (compactos).
    with st.container(key="rm_trend_ctrls", horizontal=True,
                      vertical_alignment="center", gap="medium"):
        st.multiselect("Canales", all_names, key="rm_tr_chans",
                       label_visibility="collapsed", placeholder="Canales…")
        st.radio("Rango", ["Actual"] + list(_unit.keys()), horizontal=True,
                 key="rm_trend_win", label_visibility="collapsed")
        if sel != "Actual":
            _k = f"rm_trend_qty_{sel}"
            st.session_state.setdefault(_k, _dfl.get(sel, 1))
            st.number_input("cantidad", 1, 999, step=1, key=_k,
                            label_visibility="collapsed")


def _plot_bode(tc: TransientCapture, channel: str) -> None:
    """Bode: 1X amplitud & fase vs rpm. Se llena durante arranque/parada."""
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    rpms, amp, phase = tc.bode(channel)
    if len(rpms) < 2:
        st.info("El Bode se llena durante un **transitorio**. En **Fuente y parámetros** "
                "elegí perfil *runup* o *coastdown*, pulsá ▶ Live, y verás la curva "
                "construirse al pasar por la velocidad crítica.")
        return
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("Amplitud 1X", "Fase 1X (°)"))
    fig.add_trace(go.Scatter(x=rpms, y=amp, mode="lines+markers", line=dict(width=1.5),
                             name="Amp 1X"), row=1, col=1)
    fig.add_trace(go.Scatter(x=rpms, y=phase, mode="lines+markers", line=dict(width=1.5),
                             marker=dict(color="#f59e0b"), name="Fase"), row=2, col=1)
    fig.update_yaxes(title_text="0-pk", row=1, col=1)
    fig.update_yaxes(title_text="grados", row=2, col=1)
    fig.update_xaxes(title_text="RPM", row=2, col=1)
    fig.update_layout(height=460, margin=dict(l=10, r=10, t=40, b=10), showlegend=False,
                      title=f"Bode · {channel} ({len(rpms)} puntos)")
    st.plotly_chart(fig, use_container_width=True, config=_PLOTLY_CFG)


def _plot_cascade(tc: TransientCapture, channel: str, rpm: Optional[float]) -> None:
    """Cascade / spectral map: espectro vs rpm (heatmap). Transitorio."""
    import plotly.graph_objects as go
    rpms, freqs, mat = tc.cascade(channel)
    if len(rpms) < 2:
        st.info("El Cascade se llena durante un **transitorio** (runup/coastdown). "
                "Elegí el perfil en **Fuente y parámetros** y pulsá ▶ Live.")
        return
    fig = go.Figure(go.Heatmap(x=freqs, y=rpms, z=mat, colorscale="Turbo",
                               colorbar=dict(title="Ampl")))
    # línea de orden 1X (diagonal): freq = rpm/60
    order1 = rpms / 60.0
    fig.add_trace(go.Scatter(x=order1, y=rpms, mode="lines",
                             line=dict(color="white", width=1, dash="dot"), name="1X"))
    fig.update_layout(height=460, margin=dict(l=10, r=10, t=40, b=10),
                      xaxis_title="Frecuencia (Hz)", yaxis_title="RPM",
                      title=f"Cascade · {channel} ({len(rpms)} espectros)", showlegend=False)
    st.plotly_chart(fig, use_container_width=True, config=_PLOTLY_CFG)


def _plot_polar(tc: TransientCapture, channel: str, snap: np.ndarray, vib,
                fs: float, rpm: Optional[float]) -> None:
    """Polar 1X: vector amplitud/fase. Con transitorio muestra el locus vs
    velocidad (tipo Nyquist); en estacionario, el punto actual."""
    import plotly.graph_objects as go
    rpms, amp, phase = tc.bode(channel)
    fig = go.Figure()
    if len(rpms) >= 2:
        fig.add_trace(go.Scatterpolar(
            r=amp, theta=phase, mode="lines+markers",
            marker=dict(size=7, color=rpms, colorscale="Turbo", colorbar=dict(title="RPM")),
            line=dict(width=1.5)))
        title = f"Polar 1X · {channel} — locus vs velocidad"
    else:
        name_to = {ch.name: (i, ch) for i, ch in vib}
        if channel in name_to and rpm:
            i, ch = name_to[channel]
            eu = snap[i] * 1000.0 / ch.sensitivity_mv_per_eu
            a, ph = one_x_vector(eu, fs, rpm / 60.0)
            fig.add_trace(go.Scatterpolar(r=[a], theta=[ph], mode="markers+text",
                                          text=[f"{a:.3g}∠{ph:.0f}°"], textposition="top center",
                                          marker=dict(size=16, color="#8B5CF6")))
        title = f"Polar 1X · {channel} — punto actual (corré un runup para el locus)"
    fig.update_layout(height=440, margin=dict(l=30, r=30, t=44, b=20), showlegend=False,
                      polar=dict(angularaxis=dict(rotation=90, direction="clockwise")), title=title)
    st.plotly_chart(fig, use_container_width=True, config=_PLOTLY_CFG)


def _plot_shaft_centerline(snap: np.ndarray, vib) -> None:
    """Shaft centerline: posición media del eje (gap DC) en el par X/Y."""
    import plotly.graph_objects as go
    name_to = {ch.name: (i, ch) for i, ch in vib}
    saved = st.session_state.get("rm_pairs_saved") or []
    valid = [(a, b) for a, b in saved if a in name_to and b in name_to]
    if not valid:
        st.info("Configurá pares X/Y en **Configuración → Par X/Y** para el shaft centerline.")
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
    fig.update_layout(height=440, title="Shaft Centerline (posición del eje)",
                      xaxis_title="X", yaxis_title="Y", showlegend=False,
                      yaxis=dict(scaleanchor="x", scaleratio=1))
    st.plotly_chart(fig, use_container_width=True, config=_PLOTLY_CFG)
    st.caption("⚠ Requiere gap DC. Con canales AC-coupled la posición estática es ~0. "
               "Para Shaft Centerline real se necesita el gap DC del proximitor.")


def _plot_waterfall(tc: TransientCapture, channel: str) -> None:
    """Waterfall 3D: espectro vs velocidad (superficie). Transitorio."""
    import plotly.graph_objects as go
    rpms, freqs, mat = tc.cascade(channel)
    if len(rpms) < 2:
        st.info("El Waterfall se llena en un transitorio (runup/coastdown). Corré uno en Monitoreo.")
        return
    fig = go.Figure(go.Surface(x=freqs, y=rpms, z=mat, colorscale="Turbo", showscale=True))
    fig.update_layout(height=520, title=f"Waterfall 3D · {channel}",
                      scene=dict(xaxis_title="Hz", yaxis_title="RPM", zaxis_title="Ampl"),
                      margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig, use_container_width=True, config=_PLOTLY_CFG)


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
        st.success(f"💾 Guardado offline: {meta.snapshot_id} "
                   f"({store.count(only_pending=True)} pendiente(s) de sync)")
    except Exception as e:  # noqa: BLE001
        st.error(f"No se pudo guardar: {type(e).__name__}: {e}")
