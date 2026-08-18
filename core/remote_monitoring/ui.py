"""
core/remote_monitoring/ui.py — Modo "Remote Monitoring" (render Streamlit)
=========================================================================

Se monta DENTRO de pages/02_Live_Monitoring.py como un modo alterno:
lo actual (Active Asset, datos Modbus) queda intacto; este modo hace
adquisición dinámica en vivo (NI / simulado) y dibuja rotordinámica.

Patrón de refresco: SÍNCRONO. El AcqAgent vive en st.session_state; en cada
rerun bombeamos unos bloques (agent.pump) y redibujamos. "Live" hace
st.rerun() en loop. Nada de hilos en la UI — los hilos son para el Agent
headless de sitio (core/remote_monitoring/agent.py start/stop).

En Mac corre con SimulatedStreamSource. En el PC de sitio se elige la
fuente NI (NIStreamSource) — misma UI, misma tubería.
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
)
from core.remote_monitoring.keyphasor import detect_keyphasor, one_x_vector

_MODE_KEY = "wm_live_view_mode"
_MODE_ACTIVE = "Active Asset"
_MODE_REMOTE = "Remote Monitoring"


# =====================================================================
# Punto de entrada — llamado desde el dispatcher de la página
# =====================================================================
def maybe_render_remote_monitoring() -> bool:
    """Dibuja el selector de modo arriba de todo. Si el usuario eligió
    Remote Monitoring, renderiza el módulo y devuelve True (la página
    legacy NO debe correr). Si eligió Active Asset, devuelve False."""
    mode = st.radio(
        "Modo de monitoreo",
        [_MODE_ACTIVE, _MODE_REMOTE],
        horizontal=True,
        key=_MODE_KEY,
        label_visibility="collapsed",
    )
    if mode != _MODE_REMOTE:
        return False
    _render_remote_monitoring()
    return True


# =====================================================================
# Helpers de configuración
# =====================================================================
def _make_channels(n_pairs: int, with_keyphasor: bool) -> List[ChannelConfig]:
    """Genera pares X/Y de proximidad (mil) + keyphasor opcional.

    Nombres: 1Y,1X,2Y,2X,... (convención Watermelon, sin underscore en display).
    """
    chs: List[ChannelConfig] = []
    bnc = 1
    for p in range(1, n_pairs + 1):
        chs.append(ChannelConfig(name=f"{p}Y", coupling="AC",
                                  sensitivity_mv_per_eu=200.0, bnc_port=bnc, units="mil"))
        bnc += 1
        chs.append(ChannelConfig(name=f"{p}X", coupling="AC",
                                  sensitivity_mv_per_eu=200.0, bnc_port=bnc, units="mil"))
        bnc += 1
    if with_keyphasor:
        chs.append(ChannelConfig(name="KPH", coupling="DC",
                                  sensitivity_mv_per_eu=1.0, bnc_port=bnc, units="V"))
    return chs


def _config_signature(source_kind, fs, n_pairs, kph, rpm, defect, instance_id) -> str:
    return f"{source_kind}|{fs}|{n_pairs}|{kph}|{rpm}|{defect}|{instance_id}"


def _build_agent(source_kind: str, fs: float, n_pairs: int, with_kph: bool,
                 rpm: float, defect: str, instance_id: str) -> AcqAgent:
    channels = _make_channels(n_pairs, with_kph)
    cfg = StreamConfig(
        sample_rate_hz=fs, channels=channels,
        block_seconds=0.25, buffer_seconds=8.0,
        keyphasor_name="KPH" if with_kph else None,
        rpm=rpm, defect=defect,
    )
    if source_kind == "NI 9178 (campo)":
        from core.remote_monitoring.ni_stream_source import NIStreamSource
        source = NIStreamSource(cfg)
    else:
        source = SimulatedStreamSource(cfg)
    return AcqAgent(source, instance_id=instance_id)


# =====================================================================
# Render principal
# =====================================================================
def _render_remote_monitoring() -> None:
    try:
        from core.ui_theme import page_header
        page_header(title="Remote Monitoring",
                    subtitle="Adquisición dinámica en vivo · rotordinámica NI · ISO 20816 / API 670")
    except Exception:
        st.title("Remote Monitoring")

    # ---------------- Panel de configuración ----------------
    with st.expander("⚙️ Configuración de adquisición", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            source_kind = st.selectbox(
                "Fuente de datos",
                ["Simulado (dev/Mac)", "NI 9178 (campo)"],
                help="En el PC de sitio elegí NI 9178. En Mac, Simulado.",
            )
            instance_id = st.text_input("Máquina / instancia", value="adhoc",
                                        help="Ad-hoc o el ID de un activo configurado.")
        with c2:
            fs = st.select_slider("Sample rate (Hz)",
                                  options=[2560, 5120, 10240, 25600, 51200], value=5120)
            n_pairs = st.number_input("Pares de proximidad (X/Y)", 1, 8, 2)
        with c3:
            with_kph = st.checkbox("Keyphasor (fase 1X)", value=True)
            if source_kind.startswith("Simulado"):
                rpm = st.number_input("RPM simulada", 300, 30000, 3600, step=60)
                defect = st.selectbox("Defecto simulado",
                                      ["none", "unbalance", "misalignment"])
            else:
                rpm, defect = 3600.0, "none"

    sig = _config_signature(source_kind, fs, n_pairs, with_kph, rpm, defect, instance_id)
    if st.session_state.get("rm_cfg_sig") != sig:
        # Config cambió → reconstruir agente (detener el anterior)
        old = st.session_state.get("rm_agent")
        if old is not None:
            try:
                old.stop()
            except Exception:
                pass
        st.session_state["rm_agent"] = _build_agent(
            source_kind, fs, n_pairs, with_kph, rpm, defect, instance_id)
        st.session_state["rm_cfg_sig"] = sig
        st.session_state["rm_running"] = False
        st.session_state["rm_trend"] = []

    agent: AcqAgent = st.session_state["rm_agent"]

    # ---------------- Controles ----------------
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
            agent.pump(4 if live else 8)  # ~1–2 s por refresh
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

    # ---------------- Guardar ----------------
    if save:
        _save_snapshot(agent, snap, rpm_est)

    # ---------------- Gráficos ----------------
    vib_channels = [(i, ch) for i, ch in enumerate(agent.channels)
                    if not (with_kph and ch.name == "KPH")]
    names = [ch.name for _, ch in vib_channels]

    tab_wf, tab_sp, tab_orb, tab_1x, tab_tr = st.tabs(
        ["📈 Waveform", "📊 Spectrum", "🔵 Orbita", "🎯 1X Vectores", "📉 Tendencia"])

    with tab_wf:
        sel = st.selectbox("Canal", names, key="rm_wf_ch")
        idx = names.index(sel)
        _plot_waveform(snap[vib_channels[idx][0]], fs, vib_channels[idx][1])
    with tab_sp:
        sel = st.selectbox("Canal", names, key="rm_sp_ch")
        idx = names.index(sel)
        _plot_spectrum(snap[vib_channels[idx][0]], fs, vib_channels[idx][1], rpm_est)
    with tab_orb:
        _plot_orbit(snap, vib_channels, fs)
    with tab_1x:
        _table_1x(snap, vib_channels, fs, rpm_est)
    with tab_tr:
        _update_and_plot_trend(snap, vib_channels)

    # ---------------- Auto-refresh loop ----------------
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
    pairs = []
    names = [ch.name for _, ch in vib_channels]
    for i in range(0, len(vib_channels) - 1, 2):
        pairs.append(f"{names[i]}–{names[i+1]}")
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
        st.warning("Sin keyphasor no hay vectores 1X. Activá el keyphasor en la config.")
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
