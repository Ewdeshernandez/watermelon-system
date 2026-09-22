"""
core.dynamic_raw_view — Análisis Avanzado desde onda CRUDA de System1
=====================================================================

Vista Streamlit que consume el bucket `dynamic_raw` (lo llena el agente de la
VM Parex, s1_agent.py) y reconstruye, por punto/instante:
  • Forma de onda  (X, Y + marcas de keyphasor)
  • Espectro       (FFT → órdenes, cursores 1X/2X/3X)
  • Órbita         (X vs Y + punto de referencia de keyphasor)

Embebible en Live Monitoring como expander. Autocontenida: no depende del
modelo "snapshot"; usa core.dynamic_raw (numpy) + Plotly.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import streamlit as st

from core.dynamic_raw import (
    Capture, CH_X, CH_Y, CH_KPH, download_capture, list_captures,
    load_system1_captures, keyphasor_edges, orders_axis, rpm_from_keyphasor,
    spectrum, top_peaks,
)

_VIEWS = [
    ":material/show_chart: Waveform",
    ":material/equalizer: Spectrum",
    ":material/track_changes: Orbit",
]
_INK = "#0f172a"
_GRID = "rgba(148,163,184,0.18)"
_X_COLOR = "#378ADD"
_Y_COLOR = "#D85A30"
_KPH_COLOR = "#16a34a"


@st.cache_data(ttl=120, show_spinner=False)
def _list_cached(asset: str) -> List[Dict[str, str]]:
    return list_captures(asset)


@st.cache_data(ttl=600, show_spinner=False)
def _download_cached(key: str) -> Optional[Dict]:
    cap = download_capture(key)
    if cap is None:
        return None
    return {"meta": cap.meta,
            "t": cap.t.tolist(),
            "channels": {k: v.tolist() for k, v in cap.channels.items()}}


def _cap_from_cached(d: Dict) -> Capture:
    return Capture(meta=d["meta"],
                   t=np.asarray(d["t"], float),
                   channels={k: np.asarray(v, float)
                             for k, v in d["channels"].items()})


def _resolve_asset(candidates: List[str]) -> tuple[str, List[Dict[str, str]]]:
    seen = []
    for c in candidates:
        c = (c or "").strip()
        if not c or c in seen:
            continue
        seen.append(c)
        caps = _list_cached(c)
        if caps:
            return c, caps
    return (seen[0] if seen else ""), []


@st.cache_data(ttl=120, show_spinner=False)
def _load_s1_cached(asset: str) -> List[Dict]:
    """Captures desde los CSV que System1 exporta (subidos por el uploader
    PowerShell a {asset}/s1/). Serializado para cache."""
    caps = load_system1_captures(asset)
    return [{"meta": c.meta, "t": c.t.tolist(),
             "channels": {k: v.tolist() for k, v in c.channels.items()}}
            for c in caps]


def _chip(resolved: str, point: str, source: str) -> None:
    st.markdown(
        f"<span style='background:{_INK};color:#f1f5f9;border-radius:8px;"
        f"padding:4px 12px;font-weight:700;font-size:12px;letter-spacing:.06em;'>"
        f"{resolved} · {point}</span> "
        f"<span style='color:#64748b;font-size:12px;'>fuente: {source}</span>",
        unsafe_allow_html=True)


def _render_capture(cap: Capture, instance_id: str) -> None:
    rpm = cap.rpm
    if rpm is None and cap.has(CH_KPH):
        rpm = rpm_from_keyphasor(cap.t, cap.get(CH_KPH))
    fs = cap.fs_hz
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("RPM", f"{rpm:,.0f}" if rpm else "—")
    m2.metric("Fs", f"{fs/1000:.1f} kHz" if fs else "—")
    m3.metric("Muestras/rev", cap.samples_per_rev or "—")
    m4.metric("Duración", f"{cap.t[-1]:.3f} s" if cap.t.size else "—")

    try:
        view = st.segmented_control("Vista", _VIEWS, default=_VIEWS[0],
                                    key=f"wm_dr_view_{instance_id}",
                                    label_visibility="collapsed")
    except Exception:  # noqa: BLE001
        view = st.radio("Vista", _VIEWS, horizontal=True,
                        key=f"wm_dr_view_r_{instance_id}",
                        label_visibility="collapsed")
    view = view or _VIEWS[0]
    if "Waveform" in view:
        _plot_waveform(cap)
    elif "Spectrum" in view:
        _plot_spectrum(cap, rpm)
    else:
        _plot_orbit(cap, rpm)


def render_dynamic_raw(instance_id: str, tag: Optional[str] = None,
                       asset: Optional[str] = None) -> None:
    """Dibuja el análisis avanzado dinámico del activo desde onda cruda.
    Prioridad: export en vivo de System1 (uploader PowerShell) → capturas v1
    del agente Python."""
    cand = [c for c in [asset or "", (tag or "").upper(), tag or "",
                        str(instance_id)] if c]

    # 1) System1 export en vivo (PowerShell uploader → {asset}/s1/)
    for c in cand:
        s1 = _load_s1_cached(c.strip())
        if s1:
            caps = [_cap_from_cached(d) for d in s1]
            pts = [c2.point or f"BRG{i}" for i, c2 in enumerate(caps)]
            point = st.selectbox("Punto (cojinete)", pts,
                                 key=f"wm_dr_pt_{instance_id}")
            _chip(c.strip(), point, "System1 (export en vivo)")
            _render_capture(caps[pts.index(point)], instance_id)
            return

    # 2) capturas v1 del agente Python
    resolved, caps = _resolve_asset(cand)
    if not caps:
        st.info(
            "Aún no hay onda cruda dinámica para este activo. En el server corre "
            "el uploader (`upload_csv.ps1`) o el agente (`s1_agent.py --csv`)."
        )
        st.caption(f"Buscado como activo: {resolved or '—'} · bucket `dynamic_raw`")
        return

    by_point: Dict[str, List[Dict[str, str]]] = {}
    for c in caps:
        by_point.setdefault(c["point"], []).append(c)
    points = sorted(by_point.keys())

    c1, c2 = st.columns([2, 3])
    with c1:
        point = st.selectbox("Punto (cojinete)", points,
                             key=f"wm_dr_pt_{instance_id}")
    times = by_point[point]

    def _lbl(o: Dict[str, str]) -> str:
        d, t = o.get("day", ""), o.get("time", "")
        if len(d) == 8 and len(t) == 6:
            return f"{d[:4]}-{d[4:6]}-{d[6:]} {t[:2]}:{t[2:4]}:{t[4:]}"
        return o.get("name", "")

    with c2:
        sel = st.selectbox("Fecha/hora de captura", times, format_func=_lbl,
                           key=f"wm_dr_ts_{instance_id}_{point}")
    _chip(resolved, point, "System1 (raw)")

    cached = _download_cached(sel["key"])
    if not cached:
        st.warning("No se pudo descargar esta captura.")
        return
    _render_capture(_cap_from_cached(cached), instance_id)


# =========================================================
# PLOTS
# =========================================================
def _base_layout(fig, height=360, title=""):
    fig.update_layout(
        height=height, title=title, template="plotly_white",
        margin=dict(l=48, r=16, t=40 if title else 16, b=40),
        showlegend=True, legend=dict(orientation="h", y=1.06, x=0),
        font=dict(size=12),
    )
    fig.update_xaxes(gridcolor=_GRID, zeroline=False)
    fig.update_yaxes(gridcolor=_GRID, zeroline=True, zerolinecolor=_GRID)
    return fig


def _plot_waveform(cap: Capture) -> None:
    import plotly.graph_objects as go
    t = cap.t
    fig = go.Figure()
    for ch, color in ((CH_X, _X_COLOR), (CH_Y, _Y_COLOR)):
        v = cap.get(ch)
        if v is not None:
            fig.add_scatter(x=t, y=v, mode="lines", name=f"{ch} [{cap.unit_of(ch)}]",
                            line=dict(color=color, width=1.3))
    # marcas de keyphasor
    if cap.has(CH_KPH):
        edges = keyphasor_edges(cap.get(CH_KPH))
        for i, e in enumerate(edges[:64]):
            if e < t.size:
                fig.add_vline(x=float(t[e]), line=dict(color=_KPH_COLOR, width=1,
                              dash="dot"), opacity=0.45)
        if edges.size:
            fig.add_scatter(x=[None], y=[None], mode="lines",
                            line=dict(color=_KPH_COLOR, dash="dot"),
                            name="Keyphasor")
    _base_layout(fig, title="Forma de onda")
    fig.update_xaxes(title="Tiempo [s]")
    fig.update_yaxes(title="Amplitud")
    st.plotly_chart(fig, use_container_width=True,
                    key=f"wm_dr_wf_{cap.point}_{cap.captured_at}")


def _plot_spectrum(cap: Capture, rpm: Optional[float]) -> None:
    import plotly.graph_objects as go
    fs = cap.fs_hz
    if not fs:
        st.info("Sin frecuencia de muestreo — no se puede calcular el espectro.")
        return
    fig = go.Figure()
    use_orders = bool(rpm and rpm > 0)
    peaks_all: List[Dict[str, float]] = []
    for ch, color in ((CH_X, _X_COLOR), (CH_Y, _Y_COLOR)):
        v = cap.get(ch)
        if v is None:
            continue
        freqs, amp = spectrum(v, fs)
        if freqs.size == 0:
            continue
        xaxis = orders_axis(freqs, rpm) if use_orders else freqs
        fig.add_scatter(x=xaxis, y=amp, mode="lines", name=f"{ch} [{cap.unit_of(ch)}]",
                        line=dict(color=color, width=1.3))
        peaks_all += [{**p, "ch": ch} for p in top_peaks(freqs, amp, rpm, n=3)]
    # cursores de orden 1X/2X/3X
    if use_orders:
        for k in (1, 2, 3):
            fig.add_vline(x=k, line=dict(color="#94a3b8", width=1, dash="dash"),
                          opacity=0.6)
            fig.add_annotation(x=k, yref="paper", y=1.0, text=f"{k}X",
                               showarrow=False, font=dict(size=11, color="#64748b"))
        fig.update_xaxes(title="Orden (× velocidad de giro)", range=[0, 10])
    else:
        fig.update_xaxes(title="Frecuencia [Hz]")
    _base_layout(fig, title="Espectro (FFT)")
    fig.update_yaxes(title="Amplitud (pico)")
    st.plotly_chart(fig, use_container_width=True,
                    key=f"wm_dr_sp_{cap.point}_{cap.captured_at}")

    # tabla de picos
    if peaks_all:
        peaks_all.sort(key=lambda p: p["amp"], reverse=True)
        rows = []
        for p in peaks_all[:6]:
            rows.append({
                "Canal": p["ch"],
                "Frecuencia [Hz]": round(p["freq_hz"], 1),
                "Orden": round(p.get("order", 0), 2) if use_orders else "—",
                "Amplitud": round(p["amp"], 2),
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)


def _plot_orbit(cap: Capture, rpm: Optional[float]) -> None:
    import plotly.graph_objects as go
    x, y = cap.get(CH_X), cap.get(CH_Y)
    if x is None or y is None:
        st.info("Faltan canales X/Y para la órbita.")
        return
    spr = cap.samples_per_rev
    # órbita filtrada 1X: promedio de vueltas si hay samples_per_rev
    xf, yf = x, y
    if spr and spr > 4 and x.size >= 2 * spr:
        nrev = x.size // spr
        xf = x[:nrev * spr].reshape(nrev, spr).mean(axis=0)
        yf = y[:nrev * spr].reshape(nrev, spr).mean(axis=0)
        xf = np.append(xf, xf[0]); yf = np.append(yf, yf[0])  # cerrar

    fig = go.Figure()
    fig.add_scatter(x=x, y=y, mode="lines", name="Raw",
                    line=dict(color="rgba(148,163,184,0.55)", width=1))
    fig.add_scatter(x=xf, y=yf, mode="lines", name="Filtrada (prom. vueltas)",
                    line=dict(color=_INK, width=2.2))
    # punto de keyphasor (referencia de fase): primera marca
    if cap.has(CH_KPH):
        edges = keyphasor_edges(cap.get(CH_KPH))
        if edges.size and edges[0] < x.size:
            e = int(edges[0])
            fig.add_scatter(x=[x[e]], y=[y[e]], mode="markers",
                            name="Keyphasor (0°)",
                            marker=dict(color=_KPH_COLOR, size=11,
                                        line=dict(color="white", width=1.5)))
    u = cap.unit_of(CH_X) or ""
    _base_layout(fig, height=440, title="Órbita")
    fig.update_xaxes(title=f"X [{u}]", scaleanchor="y", scaleratio=1)
    fig.update_yaxes(title=f"Y [{cap.unit_of(CH_Y) or u}]")
    st.plotly_chart(fig, use_container_width=True,
                    key=f"wm_dr_orb_{cap.point}_{cap.captured_at}")
    if rpm:
        st.caption(f"Órbita a {rpm:,.0f} RPM · promedio de vueltas para filtrar "
                   f"ruido (síncrono 1X). Punto verde = referencia de keyphasor.")


__all__ = ["render_dynamic_raw"]
