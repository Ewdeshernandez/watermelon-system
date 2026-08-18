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
                    subtitle="Adquisición dinámica en vivo · rotordinámica NI · ISO 20816 / API 670")
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
        </style>
    """, unsafe_allow_html=True)

    view = st.radio("Vista", ["Configuración", "Monitoreo"], horizontal=True,
                    key="rm_view", label_visibility="collapsed")
    st.divider()
    if view == "Configuración":
        from core.remote_monitoring.ui_setup import render_setup
        render_setup()
    else:
        _render_monitor()


# =====================================================================
# Helpers de construcción
# =====================================================================
def _demo_channels() -> List[ChannelConfig]:
    from core.remote_monitoring.config import MachineConfig, auto_layout, setup_to_channel_configs, AcqSetup
    m = MachineConfig(n_bearings=2)
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

    # ---------------- Fuente y parámetros ----------------
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
            defect = st.selectbox("Defecto simulado", ["none", "unbalance", "misalignment"]) \
                if source_kind.startswith("Simulado") else "none"

        sim = {"rpm": default_rpm, "defect": defect, "speed_profile": "constant"}
        if source_kind.startswith("Simulado"):
            st.markdown("**Perfil de velocidad** (transitorios → bode/cascade)")
            p1, p2, p3, p4 = st.columns(4)
            with p1:
                prof = st.selectbox("Perfil", ["constant", "runup", "coastdown", "runup_coastdown"])
            sim["speed_profile"] = prof
            if prof == "constant":
                with p2:
                    sim["rpm"] = st.number_input("RPM", 300, 30000, int(default_rpm), step=60)
            else:
                with p2:
                    sim["rpm_start"] = st.number_input("RPM inicio", 0, 30000, 600, step=60)
                with p3:
                    sim["rpm_end"] = st.number_input("RPM fin", 0, 30000, 6000, step=60)
                with p4:
                    sim["sim_critical_rpm"] = st.number_input("Crítica (rpm)", 0, 30000, 3000, step=60)
                sim["ramp_seconds"] = st.slider("Duración rampa (s)", 5, 120, 30)

    sig = f"{source_kind}|{fs}|{sim}|{_channels_fingerprint(channels)}"
    if st.session_state.get("rm_agent_sig") != sig:
        old = st.session_state.get("rm_agent")
        if old is not None:
            try:
                old.stop()
            except Exception:  # noqa: BLE001
                pass
        st.session_state["rm_agent"] = _build_agent(channels, source_kind, float(fs), sim, machine_name)
        st.session_state["rm_agent_sig"] = sig
        st.session_state["rm_running"] = False
        st.session_state["rm_trend"] = []
        _acq = st.session_state.get("rm_acq_saved") or {}
        _fmax = float(_acq.get("fmax_hz", 1000.0))
        st.session_state["rm_transient"] = TransientCapture(TransientConfig(fmax_hz=_fmax))
        st.session_state["rm_prev_rpm"] = None

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

    vib_channels = [(i, ch) for i, ch in enumerate(agent.channels)
                    if not is_keyphasor_channel(ch)]
    names = [ch.name for _, ch in vib_channels]

    # Estado + captura transitoria (bode/cascade se llenan en arranque/parada)
    prev = st.session_state.get("rm_prev_rpm")
    state = rm_states.classify_state(rpm_est, prev)
    st.session_state["rm_prev_rpm"] = rpm_est
    tc: TransientCapture = st.session_state.setdefault("rm_transient", TransientCapture())
    if rpm_est:
        tc.feed(snap, rpm_est, fs, vib_channels)

    _render_status(agent, snap, rpm_est, state, tc.n_samples)

    if save:
        _save_snapshot(agent, snap, rpm_est)

    tabs = st.tabs(["📈 Waveform", "📊 Spectrum", "🔵 Orbita", "🎯 Vectores",
                    "📉 Tendencia", "📐 Bode", "🌊 Cascade"])
    with tabs[0]:
        if names:
            sel = st.selectbox("Canal", names, key="rm_wf_ch")
            i = names.index(sel)
            _plot_waveform(snap[vib_channels[i][0]], fs, vib_channels[i][1])
    with tabs[1]:
        if names:
            sel = st.selectbox("Canal", names, key="rm_sp_ch")
            i = names.index(sel)
            _fmax = float((st.session_state.get("rm_acq_saved") or {}).get("fmax_hz", 0) or 0)
            _plot_spectrum(snap[vib_channels[i][0]], fs, vib_channels[i][1], rpm_est,
                           fmax=_fmax or None)
    with tabs[2]:
        _plot_orbit(snap, vib_channels, fs)
    with tabs[3]:
        _orders = (st.session_state.get("rm_acq_saved") or {}).get("orders") or [1.0, 2.0]
        _table_orders(snap, vib_channels, fs, rpm_est, _orders)
    with tabs[4]:
        _update_and_plot_trend(snap, vib_channels)
    with tabs[5]:
        if names:
            sel = st.selectbox("Canal", names, key="rm_bode_ch")
            _plot_bode(tc, sel)
    with tabs[6]:
        if names:
            sel = st.selectbox("Canal", names, key="rm_casc_ch")
            _plot_cascade(tc, sel, rpm_est)

    if st.session_state.get("rm_running"):
        time.sleep(0.4)
        st.rerun()


# =====================================================================
# Widgets de gráfico
# =====================================================================
def _render_status(agent: AcqAgent, snap: np.ndarray, rpm: Optional[float],
                   state: str, n_transient: int) -> None:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("RPM", f"{rpm:.0f}" if rpm else "—")
    c2.metric("1X (Hz)", f"{rpm/60:.1f}" if rpm else "—")
    color = rm_states.state_color(state)
    c3.markdown(f"**Estado**<br><span style='color:{color};font-weight:700;font-size:1.3rem'>"
                f"{rm_states.state_label(state)}</span>", unsafe_allow_html=True)
    c4.metric("Ventana", f"{snap.shape[1]/agent.sample_rate_hz:.1f} s")
    c5.metric("Pts transitorio", n_transient)


def _plot_waveform(x: np.ndarray, fs: float, ch: ChannelConfig) -> None:
    import plotly.graph_objects as go
    eu = x * 1000.0 / ch.sensitivity_mv_per_eu if ch.sensitivity_mv_per_eu else x
    eu = eu - np.mean(eu)
    t = np.arange(len(eu)) / fs
    fig = go.Figure(go.Scatter(x=t, y=eu, mode="lines", line=dict(width=1)))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10),
                      xaxis_title="s", yaxis_title=ch.units, title=f"Waveform · {ch.name}")
    st.plotly_chart(fig, use_container_width=True)


def _spectrum(x: np.ndarray, fs: float):
    x = x - np.mean(x)
    w = np.hanning(len(x))
    mag = np.abs(np.fft.rfft(x * w)) / (np.sum(w) / 2)
    freqs = np.fft.rfftfreq(len(x), 1.0 / fs)
    return freqs, mag


def _plot_spectrum(x: np.ndarray, fs: float, ch: ChannelConfig, rpm: Optional[float],
                   fmax: Optional[float] = None) -> None:
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
                      xaxis_title="Hz", yaxis_title=ch.units, title=f"Spectrum · {ch.name}")
    if fmax and fmax > 0:
        fig.update_xaxes(range=[0, fmax])  # span Fmax de los parámetros de adquisición
    st.plotly_chart(fig, use_container_width=True)


def _plot_orbit(snap: np.ndarray, vib_channels, fs: float) -> None:
    import plotly.graph_objects as go
    if len(vib_channels) < 2:
        st.info("La órbita necesita un par X/Y. Asocialo en **Configuración → Par X/Y**.")
        return
    name_to = {ch.name: (i, ch) for i, ch in vib_channels}
    # Pares EXPLÍCITOS desde la config (Par X/Y). Fallback a consecutivos.
    saved = st.session_state.get("rm_pairs_saved") or []
    valid = [(a, b) for a, b in saved if a in name_to and b in name_to]
    if not valid:
        names = [ch.name for _, ch in vib_channels]
        valid = [(names[i], names[i + 1]) for i in range(0, len(vib_channels) - 1, 2)]
    if not valid:
        st.info("La órbita necesita un par X/Y. Asocialo en **Configuración → Par X/Y**.")
        return

    labels = [f"{a}–{b}" for a, b in valid]
    sel = st.selectbox("Par de órbita", labels, key="rm_orbit_pair")
    a, b = valid[labels.index(sel)]
    # Y = vertical (nombre con 'Y'), X = horizontal
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
    y = y - np.mean(y)
    x = x - np.mean(x)
    fig = go.Figure(go.Scatter(x=x, y=y, mode="lines", line=dict(width=1)))
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10),
                      xaxis_title=f"{chx.name} ({chx.units})", yaxis_title=f"{chy.name} ({chy.units})",
                      title=f"Órbita · {sel}", yaxis=dict(scaleanchor="x", scaleratio=1))
    st.plotly_chart(fig, use_container_width=True)


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


def _update_and_plot_trend(snap: np.ndarray, vib_channels) -> None:
    import plotly.graph_objects as go
    hist = st.session_state.setdefault("rm_trend", [])
    rms = {ch.name: float(np.sqrt(np.mean((snap[i] - np.mean(snap[i])) ** 2))) for i, ch in vib_channels}
    hist.append((datetime.now().strftime("%H:%M:%S"), rms))
    if len(hist) > 120:
        del hist[: len(hist) - 120]
    fig = go.Figure()
    for _, ch in vib_channels:
        fig.add_trace(go.Scatter(x=[h[0] for h in hist], y=[h[1].get(ch.name) for h in hist],
                                 mode="lines+markers", name=ch.name, line=dict(width=1)))
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10),
                      xaxis_title="hora", yaxis_title="RMS (V)", title="Tendencia overall (RMS por canal)")
    st.plotly_chart(fig, use_container_width=True)


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
    st.plotly_chart(fig, use_container_width=True)


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
    st.plotly_chart(fig, use_container_width=True)


def _save_snapshot(agent: AcqAgent, snap: np.ndarray, rpm: Optional[float]) -> None:
    try:
        from core.remote_monitoring.store import LocalStore
        store = st.session_state.setdefault("rm_store", LocalStore())
        ch_meta = [{"name": ch.name, "bnc_port": ch.bnc_port, "coupling": ch.coupling,
                    "sensitivity_mv_per_eu": float(ch.sensitivity_mv_per_eu or 0.0),
                    "units": ch.units} for ch in agent.channels]
        meta = store.save_snapshot(agent.instance_id, snap, ch_meta, agent.sample_rate_hz,
                                   rpm=rpm, captured_at=datetime.now(timezone.utc).isoformat())
        st.success(f"💾 Guardado offline: {meta.snapshot_id} "
                   f"({store.count(only_pending=True)} pendiente(s) de sync)")
    except Exception as e:  # noqa: BLE001
        st.error(f"No se pudo guardar: {type(e).__name__}: {e}")
