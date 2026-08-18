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
               en Setup (o un demo si aún no configuraste).

Patrón de refresco: SÍNCRONO. El AcqAgent vive en st.session_state; en cada
rerun bombeamos unos bloques (agent.pump) y redibujamos. "Live" hace
st.rerun() en loop. Nada de hilos en la UI — los hilos son para el Agent
headless de sitio (core/remote_monitoring/agent.py start/stop).
"""

from __future__ import annotations

import time
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


# =====================================================================
# Punto de entrada — llamado por pages/02_Remote_Monitoring.py
# =====================================================================
def render_remote_monitoring() -> None:
    """Renderiza la página completa Remote Monitoring (Setup + Monitor)."""
    try:
        from core.ui_theme import page_header
        page_header(title="Remote Monitoring",
                    subtitle="Adquisición dinámica en vivo · rotordinámica NI · ISO 20816 / API 670")
    except Exception:  # noqa: BLE001
        st.title("Remote Monitoring")

    tab_setup, tab_monitor = st.tabs(["⚙️ Setup", "📡 Monitor"])
    with tab_setup:
        from core.remote_monitoring.ui_setup import render_setup
        render_setup()
    with tab_monitor:
        _render_monitor()


# =====================================================================
# Helpers de construcción
# =====================================================================
def _demo_channels() -> List[ChannelConfig]:
    """Layout demo (2 cojinetes X/Y + keyphasor) para arrancar sin config."""
    from core.remote_monitoring.config import MachineConfig, auto_layout, setup_to_channel_configs, AcqSetup
    m = MachineConfig(n_bearings=2)
    return setup_to_channel_configs(AcqSetup(machine=m, channels=auto_layout(m)))


def _channels_fingerprint(channels: List[ChannelConfig]) -> str:
    return "|".join(f"{c.name}:{c.bnc_port}:{c.coupling}:{c.sensitivity_mv_per_eu}" for c in channels)


def _build_agent(channels: List[ChannelConfig], source_kind: str, rpm: float,
                 defect: str, fs: float, instance_id: str) -> AcqAgent:
    cfg = StreamConfig(
        sample_rate_hz=fs, channels=channels,
        block_seconds=0.25, buffer_seconds=8.0,
        rpm=rpm, defect=defect,
    )
    if source_kind == "NI 9178 (campo)":
        from core.remote_monitoring.ni_stream_source import NIStreamSource
        source = NIStreamSource(cfg)
    else:
        source = SimulatedStreamSource(cfg)
    return AcqAgent(source, instance_id=instance_id)


# =====================================================================
# Tab Monitor — adquisición + gráficos
# =====================================================================
def _render_monitor() -> None:
    channels: Optional[List[ChannelConfig]] = st.session_state.get("rm_channels")

    if not channels:
        st.info("No hay configuración activa. Andá al tab **⚙️ Setup** y guardá una "
                "máquina, o cargá un layout demo para probar ya.")
        if st.button("Cargar layout demo (2 cojinetes + keyphasor)"):
            st.session_state["rm_channels"] = _demo_channels()
            st.session_state["rm_machine_rpm"] = 3600.0
            st.session_state["rm_machine_name"] = "Demo"
            st.rerun()
        return

    machine_name = st.session_state.get("rm_machine_name", "adhoc")
    default_rpm = float(st.session_state.get("rm_machine_rpm", 3600.0))

    st.caption(f"Máquina activa: **{machine_name}** · {len(channels)} canales "
               f"({sum(1 for c in channels if not is_keyphasor_channel(c))} de vibración).")

    # ---------------- Controles de adquisición ----------------
    with st.expander("⚙️ Fuente y parámetros", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            source_kind = st.selectbox(
                "Fuente de datos", ["Simulado (dev/Mac)", "NI 9178 (campo)"],
                help="En el PC de sitio elegí NI 9178. En Mac, Simulado.")
        with col2:
            fs = st.select_slider("Sample rate (Hz)",
                                  options=[2560, 5120, 10240, 25600, 51200], value=5120)
        with col3:
            if source_kind.startswith("Simulado"):
                rpm = st.number_input("RPM simulada", 300, 30000, int(default_rpm), step=60)
                defect = st.selectbox("Defecto simulado", ["none", "unbalance", "misalignment"])
            else:
                rpm, defect = default_rpm, "none"

    sig = f"{source_kind}|{fs}|{rpm}|{defect}|{_channels_fingerprint(channels)}"
    if st.session_state.get("rm_agent_sig") != sig:
        old = st.session_state.get("rm_agent")
        if old is not None:
            try:
                old.stop()
            except Exception:  # noqa: BLE001
                pass
        st.session_state["rm_agent"] = _build_agent(
            channels, source_kind, float(rpm), defect, float(fs), machine_name)
        st.session_state["rm_agent_sig"] = sig
        st.session_state["rm_running"] = False
        st.session_state["rm_trend"] = []

    agent: AcqAgent = st.session_state["rm_agent"]

    b1, b2, b3, b4, b5 = st.columns([1, 1, 1, 1, 2])
    with b1:
        if st.button("▶ Iniciar", use_container_width=True):
            st.session_state["rm_running"] = True
    with b2:
        if st.button("⏸ Detener", use_container_width=True):
            st.session_state["rm_running"] = False
    with b3:
        take = st.button("🔄 Tomar 1 lectura", use_container_width=True)
    with b4:
        save = st.button("💾 Guardar", use_container_width=True,
                         help="Persiste la ventana actual al store local.")
    with b5:
        live = st.checkbox("🟢 Live (auto-refresh)",
                           value=st.session_state.get("rm_running", False))
        st.session_state["rm_running"] = live

    # ---------------- Adquisición ----------------
    err = None
    try:
        if live or take:
            agent.pump(4 if live else 8)
    except ImportError as e:
        err = str(e)
        st.session_state["rm_running"] = False
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"
        st.session_state["rm_running"] = False

    if err:
        st.error(f"⚠ No se pudo adquirir: {err}")
        if "nidaqmx" in err.lower():
            st.info("En Mac usá la fuente **Simulado**. La fuente NI solo corre "
                    "en el PC de sitio (Windows con NI-DAQmx + 9178 conectado).")

    snap = agent.snapshot()
    if snap.shape[1] == 0:
        st.info("Sin datos aún. Pulsá **▶ Iniciar** o **Tomar 1 lectura**.")
        return

    fs = agent.sample_rate_hz
    rpm_est = agent.estimate_rpm(snap)
    _render_status(agent, snap, rpm_est)

    if save:
        _save_snapshot(agent, snap, rpm_est)

    vib_channels = [(i, ch) for i, ch in enumerate(agent.channels)
                    if not is_keyphasor_channel(ch)]
    names = [ch.name for _, ch in vib_channels]

    tab_wf, tab_sp, tab_orb, tab_1x, tab_tr = st.tabs(
        ["📈 Waveform", "📊 Spectrum", "🔵 Orbita", "🎯 1X Vectores", "📉 Tendencia"])
    with tab_wf:
        if names:
            sel = st.selectbox("Canal", names, key="rm_wf_ch")
            idx = names.index(sel)
            _plot_waveform(snap[vib_channels[idx][0]], fs, vib_channels[idx][1])
    with tab_sp:
        if names:
            sel = st.selectbox("Canal", names, key="rm_sp_ch")
            idx = names.index(sel)
            _plot_spectrum(snap[vib_channels[idx][0]], fs, vib_channels[idx][1], rpm_est)
    with tab_orb:
        _plot_orbit(snap, vib_channels, fs)
    with tab_1x:
        _table_1x(snap, vib_channels, fs, rpm_est)
    with tab_tr:
        _update_and_plot_trend(snap, vib_channels)

    if st.session_state.get("rm_running"):
        time.sleep(0.4)
        st.rerun()


# =====================================================================
# Widgets de gráfico
# =====================================================================
def _render_status(agent: AcqAgent, snap: np.ndarray, rpm: Optional[float]) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("RPM", f"{rpm:.0f}" if rpm else "—")
    c2.metric("1X (Hz)", f"{rpm/60:.1f}" if rpm else "—")
    c3.metric("Ventana", f"{snap.shape[1]/agent.sample_rate_hz:.1f} s")
    c4.metric("Bloques leídos", agent.blocks_read)


def _plot_waveform(x: np.ndarray, fs: float, ch: ChannelConfig) -> None:
    import plotly.graph_objects as go
    eu = x * 1000.0 / ch.sensitivity_mv_per_eu if ch.sensitivity_mv_per_eu else x
    eu = eu - np.mean(eu)
    t = np.arange(len(eu)) / fs
    fig = go.Figure(go.Scatter(x=t, y=eu, mode="lines", line=dict(width=1)))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10),
                      xaxis_title="s", yaxis_title=ch.units,
                      title=f"Waveform · {ch.name}")
    st.plotly_chart(fig, use_container_width=True)


def _spectrum(x: np.ndarray, fs: float):
    x = x - np.mean(x)
    w = np.hanning(len(x))
    mag = np.abs(np.fft.rfft(x * w)) / (np.sum(w) / 2)
    freqs = np.fft.rfftfreq(len(x), 1.0 / fs)
    return freqs, mag


def _plot_spectrum(x: np.ndarray, fs: float, ch: ChannelConfig,
                   rpm: Optional[float]) -> None:
    import plotly.graph_objects as go
    eu = x * 1000.0 / ch.sensitivity_mv_per_eu if ch.sensitivity_mv_per_eu else x
    freqs, mag = _spectrum(eu, fs)
    fig = go.Figure(go.Scatter(x=freqs, y=mag, mode="lines", line=dict(width=1)))
    if rpm:
        f1 = rpm / 60.0
        for k, lbl in [(1, "1X"), (2, "2X"), (3, "3X")]:
            if k * f1 < freqs[-1]:
                fig.add_vline(x=k * f1, line=dict(color="#ef4444", width=1, dash="dot"),
                              annotation_text=lbl)
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10),
                      xaxis_title="Hz", yaxis_title=ch.units,
                      title=f"Spectrum · {ch.name}")
    st.plotly_chart(fig, use_container_width=True)


def _plot_orbit(snap: np.ndarray, vib_channels, fs: float) -> None:
    import plotly.graph_objects as go
    if len(vib_channels) < 2:
        st.info("La órbita necesita un par X/Y. Configurá al menos 1 par.")
        return
    names = [ch.name for _, ch in vib_channels]
    pairs = [f"{names[i]}–{names[i+1]}" for i in range(0, len(vib_channels) - 1, 2)]
    if not pairs:
        st.info("La órbita necesita un par X/Y consecutivo.")
        return
    sel = st.selectbox("Par de proximidad", pairs, key="rm_orbit_pair")
    pi = pairs.index(sel) * 2
    yi, chy = vib_channels[pi]
    xi, chx = vib_channels[pi + 1]
    y = snap[yi] * 1000.0 / chy.sensitivity_mv_per_eu
    x = snap[xi] * 1000.0 / chx.sensitivity_mv_per_eu
    y = y - np.mean(y)
    x = x - np.mean(x)
    fig = go.Figure(go.Scatter(x=x, y=y, mode="lines", line=dict(width=1)))
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10),
                      xaxis_title=f"{chx.name} ({chx.units})",
                      yaxis_title=f"{chy.name} ({chy.units})",
                      title=f"Órbita · {sel}",
                      yaxis=dict(scaleanchor="x", scaleratio=1))
    st.plotly_chart(fig, use_container_width=True)


def _table_1x(snap: np.ndarray, vib_channels, fs: float,
              rpm: Optional[float]) -> None:
    if not rpm:
        st.warning("Sin keyphasor no hay vectores 1X. Activá el keyphasor en Setup.")
        return
    f1 = rpm / 60.0
    rows = []
    for i, ch in vib_channels:
        eu = snap[i] * 1000.0 / ch.sensitivity_mv_per_eu
        amp, phase = one_x_vector(eu, fs, f1)
        rows.append({"Sensor": ch.name, f"1X ({ch.units} 0-pk)": round(amp, 4),
                     "Fase (°)": round(phase, 1)})
    st.dataframe(rows, use_container_width=True, hide_index=True)


def _update_and_plot_trend(snap: np.ndarray, vib_channels) -> None:
    import plotly.graph_objects as go
    hist = st.session_state.setdefault("rm_trend", [])
    rms = {ch.name: float(np.sqrt(np.mean((snap[i] - np.mean(snap[i])) ** 2)))
           for i, ch in vib_channels}
    hist.append((datetime.now().strftime("%H:%M:%S"), rms))
    if len(hist) > 120:
        del hist[: len(hist) - 120]
    fig = go.Figure()
    for _, ch in vib_channels:
        fig.add_trace(go.Scatter(x=[h[0] for h in hist],
                                 y=[h[1].get(ch.name) for h in hist],
                                 mode="lines+markers", name=ch.name, line=dict(width=1)))
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10),
                      xaxis_title="hora", yaxis_title="RMS (V)",
                      title="Tendencia overall (RMS por canal)")
    st.plotly_chart(fig, use_container_width=True)


def _save_snapshot(agent: AcqAgent, snap: np.ndarray, rpm: Optional[float]) -> None:
    try:
        from core.remote_monitoring.store import LocalStore
        store = st.session_state.setdefault("rm_store", LocalStore())
        ch_meta = [{"name": ch.name, "bnc_port": ch.bnc_port, "coupling": ch.coupling,
                    "sensitivity_mv_per_eu": float(ch.sensitivity_mv_per_eu or 0.0),
                    "units": ch.units} for ch in agent.channels]
        meta = store.save_snapshot(
            agent.instance_id, snap, ch_meta, agent.sample_rate_hz, rpm=rpm,
            captured_at=datetime.now(timezone.utc).isoformat())
        st.success(f"💾 Guardado offline: {meta.snapshot_id} "
                   f"({store.count(only_pending=True)} pendiente(s) de sync)")
    except Exception as e:  # noqa: BLE001
        st.error(f"No se pudo guardar: {type(e).__name__}: {e}")
