"""
core.dynamic_raw_view — Análisis dinámico desde onda CRUDA (con keyphasor)
==========================================================================

Vista Streamlit que consume el bucket `dynamic_raw` (lo llena el agente en
sitio) y reconstruye, por cojinete/instante:
  • Forma de onda  (X, Y + marcas de keyphasor)
  • Espectro       (FFT → órdenes, cursores 1X/2X/3X + picos)
  • Órbita         (X vs Y, filtrada por vueltas + referencia de keyphasor)

Presentación clase mundial (estilo rotodinámica profesional). Autocontenida:
numpy + Plotly, sin depender del modelo "snapshot".
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import streamlit as st

from core.dynamic_raw import (
    Capture, CH_X, CH_Y, CH_KPH, download_capture, list_captures,
    keyphasor_edges, orders_axis, rpm_from_keyphasor, spectrum, top_peaks,
)

# ---- paleta profesional ----
_INK = "#0f172a"
_MUTED = "#64748b"
_GRID = "rgba(148,163,184,0.16)"
_X_COLOR = "#2563eb"      # azul — sensor X (horizontal)
_Y_COLOR = "#ea580c"      # naranja — sensor Y (vertical)
_KPH_COLOR = "#16a34a"    # verde — keyphasor
_ORBIT_RAW = "rgba(148,163,184,0.50)"
_ORBIT_FILT = "#0f172a"
_ACCENT = "#e11d48"

_VIEWS = [
    ":material/show_chart: Onda",
    ":material/equalizer: Espectro",
    ":material/track_changes: Órbita",
]


# =========================================================
# Cache + carga
# =========================================================
@st.cache_data(ttl=90, show_spinner=False)
def _list_cached(asset: str) -> List[Dict[str, str]]:
    return list_captures(asset)


@st.cache_data(ttl=600, show_spinner=False)
def _download_cached(key: str) -> Optional[Dict]:
    cap = download_capture(key)
    if cap is None:
        return None
    return {"meta": cap.meta, "t": cap.t.tolist(),
            "channels": {k: v.tolist() for k, v in cap.channels.items()}}


def _cap_from_cached(d: Dict) -> Capture:
    return Capture(meta=d["meta"], t=np.asarray(d["t"], float),
                   channels={k: np.asarray(v, float)
                             for k, v in d["channels"].items()})


def _resolve_asset(candidates: List[str]) -> Tuple[str, List[Dict[str, str]]]:
    seen: List[str] = []
    for c in candidates:
        c = (c or "").strip()
        if not c or c in seen:
            continue
        seen.append(c)
        caps = _list_cached(c)
        if caps:
            return c, caps
    return (seen[0] if seen else ""), []


# =========================================================
# Etiquetas amigables por cojinete (Turbina 1/2 · Generador 5/6…)
# =========================================================
def _brg_num(point: str) -> Optional[int]:
    m = re.match(r"BRG(\d+)", (point or "").upper())
    return int(m.group(1)) if m else None


def _group_of(n: Optional[int]) -> str:
    if n is None:
        return ""
    return "Turbina" if n <= 4 else "Generador"


def _point_label(point: str) -> str:
    n = _brg_num(point)
    if n is None:
        return point
    return f"{_group_of(n)} · Cojinete {n}"


def _sensor_names(point: str) -> Tuple[str, str]:
    """Etiquetas de los sensores X/Y del cojinete, ej. '1X' / '1Y'."""
    n = _brg_num(point)
    if n is None:
        return "X", "Y"
    return f"{n}X", f"{n}Y"


def _order_points(points: List[str]) -> List[str]:
    return sorted(points, key=lambda p: (_brg_num(p) is None, _brg_num(p) or 999, p))


# =========================================================
# Encabezado
# =========================================================
def _chip(resolved: str, point: str, when: str) -> None:
    n = _brg_num(point)
    color = "#1d4ed8" if (n and n <= 4) else "#b45309"
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:10px;flex-wrap:wrap;"
        f"margin:2px 0 6px'>"
        f"<span style='background:{_INK};color:#f1f5f9;border-radius:8px;"
        f"padding:5px 13px;font-weight:800;font-size:12.5px;letter-spacing:.04em;'>"
        f"{_point_label(point)}</span>"
        f"<span style='background:{color};color:#fff;border-radius:6px;"
        f"padding:3px 9px;font-weight:700;font-size:11px;'>{'/'.join(_sensor_names(point))}</span>"
        f"<span style='color:{_MUTED};font-size:12px;'>{resolved} · captura {when}</span>"
        f"</div>", unsafe_allow_html=True)


def render_dynamic_raw(instance_id: str, tag: Optional[str] = None,
                       asset: Optional[str] = None) -> None:
    """Análisis dinámico del activo desde onda cruda (bucket dynamic_raw)."""
    cand = [c for c in [asset or "", (tag or "").upper(), tag or "",
                        str(instance_id)] if c]
    resolved, caps = _resolve_asset(cand)
    if not caps:
        st.info("Aún no hay onda cruda para este equipo. El agente en sitio la "
                "captura y sube automáticamente cada 2 horas.")
        st.caption(f"Equipo: {resolved or '—'}")
        return

    by_point: Dict[str, List[Dict[str, str]]] = {}
    for c in caps:
        by_point.setdefault(c["point"], []).append(c)
    points = _order_points(list(by_point.keys()))

    c1, c2 = st.columns([2, 3])
    with c1:
        point = st.selectbox("Cojinete", points, format_func=_point_label,
                             key=f"wm_dr_pt_{instance_id}")
    times = by_point[point]

    def _lbl(o: Dict[str, str]) -> str:
        d, t = o.get("day", ""), o.get("time", "")
        if len(d) == 8 and len(t) == 6:
            return f"{d[:4]}-{d[4:6]}-{d[6:]}  {t[:2]}:{t[2:4]}:{t[4:]}"
        return o.get("name", "")

    with c2:
        sel = st.selectbox("Captura", times, format_func=_lbl,
                           key=f"wm_dr_ts_{instance_id}_{point}")
    _chip(resolved, point, _lbl(sel).split("  ")[-1] if "  " in _lbl(sel) else _lbl(sel))

    cached = _download_cached(sel["key"])
    if not cached:
        st.warning("No se pudo descargar esta captura.")
        return
    _render_capture(_cap_from_cached(cached), instance_id, point)


def _render_capture(cap: Capture, instance_id: str, point: str) -> None:
    rpm = cap.rpm
    if rpm is None and cap.has(CH_KPH):
        rpm = rpm_from_keyphasor(cap.t, cap.get(CH_KPH))
    fs = cap.fs_hz
    nx, ny = _sensor_names(point)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Velocidad", f"{rpm:,.0f} RPM" if rpm else "—")
    m2.metric("Muestreo", f"{fs/1000:.1f} kHz" if fs else "—")
    m3.metric("Muestras/vuelta", cap.samples_per_rev or "—")
    m4.metric("Duración", f"{cap.t[-1]*1000:.0f} ms" if cap.t.size else "—")

    try:
        view = st.segmented_control("Vista", _VIEWS, default=_VIEWS[0],
                                    key=f"wm_dr_view_{instance_id}",
                                    label_visibility="collapsed")
    except Exception:  # noqa: BLE001
        view = st.radio("Vista", _VIEWS, horizontal=True,
                        key=f"wm_dr_view_r_{instance_id}",
                        label_visibility="collapsed")
    view = view or _VIEWS[0]
    if "Onda" in view:
        _plot_waveform(cap, nx, ny)
    elif "Espectro" in view:
        _plot_spectrum(cap, rpm, nx, ny)
    else:
        _plot_orbit(cap, rpm, point, nx, ny)


# =========================================================
# Layout base
# =========================================================
def _base_layout(fig, height=380, title=""):
    fig.update_layout(
        height=height, template="plotly_white",
        margin=dict(l=56, r=18, t=52 if title else 18, b=46),
        showlegend=True,
        legend=dict(orientation="h", y=1.10, x=0, bgcolor="rgba(0,0,0,0)",
                    font=dict(size=11)),
        font=dict(size=12, color=_INK),
        plot_bgcolor="white", paper_bgcolor="white",
        title=dict(text=title, x=0, xanchor="left",
                   font=dict(size=15, color=_INK, family="Arial Black")),
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor=_GRID, zeroline=False, showline=True,
                     linecolor="rgba(15,23,42,0.25)", ticks="outside",
                     tickcolor="rgba(15,23,42,0.25)")
    fig.update_yaxes(gridcolor=_GRID, zeroline=True,
                     zerolinecolor="rgba(15,23,42,0.28)", showline=True,
                     linecolor="rgba(15,23,42,0.25)")
    return fig


def _pp(v: np.ndarray) -> float:
    return float(np.nanmax(v) - np.nanmin(v)) if v.size else 0.0


# =========================================================
# ONDA
# =========================================================
def _plot_waveform(cap: Capture, nx: str, ny: str) -> None:
    import plotly.graph_objects as go
    t = cap.t * 1000.0  # ms
    u = cap.unit_of(CH_X) or "µm"
    fig = go.Figure()
    for ch, color, nm in ((CH_X, _X_COLOR, nx), (CH_Y, _Y_COLOR, ny)):
        v = cap.get(ch)
        if v is not None:
            fig.add_scatter(x=t, y=v, mode="lines",
                            name=f"{nm}  ({_pp(v):.1f} {u} pp)",
                            line=dict(color=color, width=1.5))
    # marcas de keyphasor (arranque de vuelta) — tenues + marcador de fase
    if cap.has(CH_KPH):
        edges = keyphasor_edges(cap.get(CH_KPH))
        ymin = min([float(np.nanmin(cap.get(c))) for c in (CH_X, CH_Y)
                    if cap.get(c) is not None] or [0.0])
        for e in edges[:96]:
            if e < t.size:
                fig.add_vline(x=float(t[e]), line=dict(color=_KPH_COLOR, width=1,
                              dash="dot"), opacity=0.30)
        kx = [float(t[e]) for e in edges[:96] if e < t.size]
        if kx:
            fig.add_scatter(x=kx, y=[ymin]*len(kx), mode="markers",
                            name="Keyphasor", marker=dict(color=_KPH_COLOR,
                            size=7, symbol="triangle-up"))
    _base_layout(fig, title="Forma de onda")
    fig.update_xaxes(title="Tiempo [ms]")
    fig.update_yaxes(title=f"Amplitud [{u}]")
    st.plotly_chart(fig, use_container_width=True,
                    key=f"wm_dr_wf_{cap.point}_{cap.captured_at}")


# =========================================================
# ESPECTRO
# =========================================================
def _plot_spectrum(cap: Capture, rpm: Optional[float], nx: str, ny: str) -> None:
    import plotly.graph_objects as go
    fs = cap.fs_hz
    if not fs:
        st.info("Sin frecuencia de muestreo — no se puede calcular el espectro.")
        return
    u = cap.unit_of(CH_X) or "µm"
    use_orders = bool(rpm and rpm > 0)
    fig = go.Figure()
    peaks_all: List[Dict[str, float]] = []
    fills = {CH_X: "rgba(37,99,235,0.10)", CH_Y: "rgba(234,88,12,0.09)"}
    for ch, color, nm in ((CH_X, _X_COLOR, nx), (CH_Y, _Y_COLOR, ny)):
        v = cap.get(ch)
        if v is None:
            continue
        freqs, amp = spectrum(v, fs)
        if freqs.size == 0:
            continue
        xaxis = orders_axis(freqs, rpm) if use_orders else freqs
        fig.add_scatter(x=xaxis, y=amp, mode="lines", name=nm,
                        line=dict(color=color, width=1.5),
                        fill="tozeroy", fillcolor=fills[ch])
        peaks_all += [{**p, "ch": nm} for p in top_peaks(freqs, amp, rpm, n=3)]
    # cursores de orden
    if use_orders:
        for k in (1, 2, 3, 4, 5):
            fig.add_vline(x=k, line=dict(color="rgba(100,116,139,0.55)",
                          width=1, dash="dash"))
            fig.add_annotation(x=k, yref="paper", y=1.02, text=f"{k}X",
                               showarrow=False,
                               font=dict(size=11, color=_MUTED))
        fig.update_xaxes(title="Orden (× velocidad de giro)", range=[0, 10])
    else:
        fig.update_xaxes(title="Frecuencia [Hz]")
    # etiquetar los 2 picos mayores
    for p in sorted(peaks_all, key=lambda z: z["amp"], reverse=True)[:2]:
        xp = p.get("order") if use_orders else p["freq_hz"]
        fig.add_annotation(x=xp, y=p["amp"], text=f"{p['amp']:.1f}",
                           showarrow=True, arrowhead=0, ax=0, ay=-16,
                           font=dict(size=10, color=_INK))
    _base_layout(fig, title="Espectro (FFT)")
    fig.update_yaxes(title=f"Amplitud pico [{u}]")
    st.plotly_chart(fig, use_container_width=True,
                    key=f"wm_dr_sp_{cap.point}_{cap.captured_at}")
    if peaks_all:
        peaks_all.sort(key=lambda p: p["amp"], reverse=True)
        rows = [{"Sensor": p["ch"], "Frecuencia [Hz]": round(p["freq_hz"], 1),
                 "Orden": round(p.get("order", 0), 2) if use_orders else "—",
                 f"Amplitud [{u}]": round(p["amp"], 2)}
                for p in peaks_all[:6]]
        st.dataframe(rows, use_container_width=True, hide_index=True)


# =========================================================
# ÓRBITA
# =========================================================
def _plot_orbit(cap: Capture, rpm: Optional[float], point: str,
                nx: str, ny: str) -> None:
    import plotly.graph_objects as go
    x, y = cap.get(CH_X), cap.get(CH_Y)
    if x is None or y is None:
        st.info("Faltan canales X/Y para la órbita.")
        return
    u = cap.unit_of(CH_X) or "µm"
    spr = cap.samples_per_rev
    xf, yf = x, y
    if spr and spr > 4 and x.size >= 2 * spr:
        nrev = x.size // spr
        xf = x[:nrev * spr].reshape(nrev, spr).mean(axis=0)
        yf = y[:nrev * spr].reshape(nrev, spr).mean(axis=0)
        xf = np.append(xf, xf[0]); yf = np.append(yf, yf[0])

    fig = go.Figure()
    # órbita cruda tenue
    fig.add_scatter(x=x, y=y, mode="lines", name="Cruda",
                    line=dict(color=_ORBIT_RAW, width=1))
    # órbita filtrada (promedio de vueltas) — protagonista
    fig.add_scatter(x=xf, y=yf, mode="lines", name="Filtrada (síncrona)",
                    line=dict(color=_ORBIT_FILT, width=2.6))
    # keyphasor: punto de referencia de fase + sentido de giro
    if cap.has(CH_KPH):
        edges = keyphasor_edges(cap.get(CH_KPH))
        if edges.size and edges[0] < x.size:
            e = int(edges[0])
            fig.add_scatter(x=[x[e]], y=[y[e]], mode="markers+text",
                            name="Keyphasor (0°)", text=["  0°"],
                            textposition="middle right",
                            textfont=dict(color=_KPH_COLOR, size=11),
                            marker=dict(color=_KPH_COLOR, size=13,
                                        line=dict(color="white", width=2)))
            # flecha de sentido de giro (primeras muestras)
            if xf.size > 6:
                fig.add_annotation(x=xf[5], y=yf[5], ax=xf[1], ay=yf[1],
                                   xref="x", yref="y", axref="x", ayref="y",
                                   showarrow=True, arrowhead=3, arrowsize=1.4,
                                   arrowcolor=_ACCENT, arrowwidth=2)
    # cruz en el origen
    fig.add_hline(y=0, line=dict(color="rgba(15,23,42,0.18)", width=1))
    fig.add_vline(x=0, line=dict(color="rgba(15,23,42,0.18)", width=1))

    amp_pp = max(_pp(xf), _pp(yf))
    _base_layout(fig, height=470, title=f"Órbita · {_point_label(point)}")
    fig.update_xaxes(title=f"{nx} [{u}]", scaleanchor="y", scaleratio=1,
                     zeroline=False)
    fig.update_yaxes(title=f"{ny} [{u}]", zeroline=False)
    st.plotly_chart(fig, use_container_width=True,
                    key=f"wm_dr_orb_{cap.point}_{cap.captured_at}")
    cap_rpm = f"{rpm:,.0f} RPM · " if rpm else ""
    st.caption(f"{cap_rpm}amplitud ≈ {amp_pp:.1f} {u} pp · órbita filtrada por "
               f"promedio de vueltas (síncrona 1X). Punto verde = keyphasor (0°); "
               f"flecha roja = sentido de giro.")


__all__ = ["render_dynamic_raw"]
