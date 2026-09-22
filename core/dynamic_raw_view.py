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
    rpm_from_keyphasor,
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

# Pestañas con BOLITAS de color (patrón del módulo Calibración), sin iconitos.
_VIEWS = ["🔵  Onda", "🟢  Espectro", "🟠  Órbita"]


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
# Sin barra de herramientas de Plotly (cámara/zoom/±) — se ve limpio.
_PCFG = {"displayModeBar": False, "displaylogo": False, "scrollZoom": False}


def _base_layout(fig, height=380, title=""):
    fig.update_layout(
        height=height, template="plotly_white",
        margin=dict(l=64, r=18, t=52 if title else 18, b=48),
        showlegend=True,
        legend=dict(orientation="h", y=1.10, x=0, bgcolor="rgba(0,0,0,0)",
                    font=dict(size=11)),
        font=dict(size=12, color=_INK),
        plot_bgcolor="white", paper_bgcolor="white",
        title=dict(text=title, x=0, xanchor="left",
                   font=dict(size=15, color=_INK, family="Arial Black")),
        hovermode="closest", dragmode=False,
    )
    fig.update_xaxes(gridcolor=_GRID, zeroline=False, showline=True,
                     linecolor="rgba(15,23,42,0.35)", ticks="outside",
                     tickcolor="rgba(15,23,42,0.30)")
    fig.update_yaxes(gridcolor=_GRID, showline=True,
                     linecolor="rgba(15,23,42,0.35)", ticks="outside",
                     tickcolor="rgba(15,23,42,0.30)")
    return fig


def _pp(v: np.ndarray) -> float:
    return float(np.nanmax(v) - np.nanmin(v)) if v.size else 0.0


def _disp_type(unit: str) -> str:
    """Tipo de medición por la unidad (viene de la config de System1)."""
    u = (unit or "").lower()
    if any(k in u for k in ("m/s2", "m/s²", " g", "accel")) or u == "g":
        return "accel"
    if any(k in u for k in ("mm/s", "in/s", "ips", "vel")):
        return "vel"
    return "disp"  # µm / mils → desplazamiento


# CPM máx de display por tipo (desplazamiento 60k, velocidad 300k, accel 600k)
_FMAX_CPM = {"disp": 60000, "vel": 300000, "accel": 600000}


def _hires_spectrum(v: np.ndarray, fs: float, zpad: int = 8):
    """FFT de ALTA RESOLUCIÓN: ventana Hann + zero-padding (curva suave, no
    dentada). Devuelve (freqs_hz, amp_pico) calibrada a amplitud real."""
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    n = v.size
    if n < 8 or not fs or fs <= 0:
        return np.zeros(0), np.zeros(0)
    w = np.hanning(n)
    vw = (v - v.mean()) * w
    nfft = int(2 ** np.ceil(np.log2(n * max(1, zpad))))
    sp = np.fft.rfft(vw, n=nfft)
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    amp = np.abs(sp) * 2.0 / (n * 0.5)   # ≈ amplitud pico (corrige ventana Hann)
    if amp.size:
        amp[0] /= 2.0
    return freqs, amp


# =========================================================
# ONDA — un gráfico INDEPENDIENTE por sensor (apilados)
# =========================================================
def _plot_waveform(cap: Capture, nx: str, ny: str) -> None:
    from plotly.subplots import make_subplots
    t = cap.t * 1000.0  # ms
    u = cap.unit_of(CH_X) or "µm"
    xmax = float(t[-1]) if t.size else 1.0
    chans = [(CH_X, _X_COLOR, nx), (CH_Y, _Y_COLOR, ny)]
    chans = [(c, col, nm) for c, col, nm in chans if cap.get(c) is not None]
    if not chans:
        st.info("Sin canales de onda.")
        return
    fig = make_subplots(rows=len(chans), cols=1, shared_xaxes=True,
                        vertical_spacing=0.10,
                        subplot_titles=[f"{nm}   ·   {_pp(cap.get(c)):.1f} {u} pp"
                                        for c, col, nm in chans])
    for i, (ch, color, nm) in enumerate(chans, start=1):
        v = cap.get(ch)
        fig.add_scatter(x=t, y=v, mode="lines", name=nm, row=i, col=1,
                        line=dict(color=color, width=1.6), showlegend=False)
        fig.update_yaxes(title_text=f"[{u}]", row=i, col=1, zeroline=True,
                         zerolinecolor="rgba(15,23,42,0.30)")
        fig.update_xaxes(range=[0, xmax], row=i, col=1)
    fig.update_xaxes(title_text="Tiempo [ms]", row=len(chans), col=1)
    _base_layout(fig, height=200 * len(chans) + 60, title="Forma de onda")
    fig.update_layout(showlegend=False)
    _style_subtitles(fig)
    st.plotly_chart(fig, use_container_width=True, config=_PCFG,
                    key=f"wm_dr_wf_{cap.point}_{cap.captured_at}")
    st.caption(f"Duración capturada: {xmax:.0f} ms · {cap.samples_per_rev or '—'} "
               f"muestras/vuelta. Más tiempo/resolución = más vueltas en la "
               f"captura de campo.")


# =========================================================
# ESPECTRO — un gráfico INDEPENDIENTE por sensor (alta resolución, CPM)
# =========================================================
def _plot_spectrum(cap: Capture, rpm: Optional[float], nx: str, ny: str) -> None:
    from plotly.subplots import make_subplots
    fs = cap.fs_hz
    if not fs:
        st.info("Sin frecuencia de muestreo — no se puede calcular el espectro.")
        return
    unit = cap.unit_of(CH_X) or "µm"
    is_disp = _disp_type(unit) == "disp"
    ysuf = f"{unit} pp" if is_disp else f"{unit} pico"
    fmax_cpm = min(_FMAX_CPM[_disp_type(unit)], fs / 2.0 * 60.0)
    fills = {CH_X: "rgba(37,99,235,0.12)", CH_Y: "rgba(234,88,12,0.10)"}
    chans = [(CH_X, _X_COLOR, nx), (CH_Y, _Y_COLOR, ny)]
    chans = [(c, col, nm) for c, col, nm in chans if cap.get(c) is not None]
    if not chans:
        st.info("Sin canales para el espectro.")
        return
    fig = make_subplots(rows=len(chans), cols=1, shared_xaxes=True,
                        vertical_spacing=0.12,
                        subplot_titles=[nm for c, col, nm in chans])
    for i, (ch, color, nm) in enumerate(chans, start=1):
        freqs, amp = _hires_spectrum(cap.get(ch), fs)
        if freqs.size == 0:
            continue
        cpm = freqs * 60.0
        yv = amp * 2.0 if is_disp else amp     # pp para desplazamiento
        fig.add_scatter(x=cpm, y=yv, mode="lines", name=nm, row=i, col=1,
                        line=dict(color=color, width=1.4),
                        fill="tozeroy", fillcolor=fills[ch], showlegend=False)
        fig.update_xaxes(range=[0, fmax_cpm], tickformat=",d", row=i, col=1)
        fig.update_yaxes(title_text=f"[{ysuf}]", rangemode="tozero", row=i, col=1)
    fig.update_xaxes(title_text="Frecuencia [CPM]", row=len(chans), col=1)
    _base_layout(fig, height=220 * len(chans) + 60, title="Espectro (FFT)")
    fig.update_layout(showlegend=False)
    _style_subtitles(fig)
    st.plotly_chart(fig, use_container_width=True, config=_PCFG,
                    key=f"wm_dr_sp_{cap.point}_{cap.captured_at}")


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
    fig.add_hline(y=0, line=dict(color="rgba(15,23,42,0.14)", width=1))
    fig.add_vline(x=0, line=dict(color="rgba(15,23,42,0.14)", width=1))
    fig.add_scatter(x=x, y=y, mode="lines", name="Cruda",
                    line=dict(color=_ORBIT_RAW, width=1))
    fig.add_scatter(x=xf, y=yf, mode="lines", name="Filtrada (síncrona)",
                    line=dict(color=_ORBIT_FILT, width=2.6))
    amp_pp = max(_pp(xf), _pp(yf))
    _base_layout(fig, height=480, title=f"Órbita · {_point_label(point)}")
    fig.update_xaxes(title=f"{nx} [{u}]", scaleanchor="y", scaleratio=1,
                     zeroline=False)
    fig.update_yaxes(title=f"{ny} [{u}]", zeroline=False)
    st.plotly_chart(fig, use_container_width=True, config=_PCFG,
                    key=f"wm_dr_orb_{cap.point}_{cap.captured_at}")
    cap_rpm = f"{rpm:,.0f} RPM · " if rpm else ""
    st.caption(f"{cap_rpm}amplitud ≈ {amp_pp:.1f} {u} pp · filtrada por promedio "
               f"de vueltas (síncrona 1X).")


def _style_subtitles(fig) -> None:
    """Deja los títulos de subplot alineados a la izquierda y discretos."""
    for ann in fig.layout.annotations:
        ann.update(x=0, xanchor="left", font=dict(size=12, color=_INK,
                   family="Arial"))


__all__ = ["render_dynamic_raw"]
