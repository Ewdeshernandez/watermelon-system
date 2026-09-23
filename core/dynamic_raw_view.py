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
        f"<div style='display:flex;align-items:center;gap:9px;flex-wrap:wrap;"
        f"margin:2px 0 8px'>"
        f"<span style='background:{_INK};color:#f1f5f9;border-radius:8px;"
        f"padding:5px 13px;font-weight:800;font-size:12.5px;letter-spacing:.04em;'>"
        f"{resolved} · {_point_label(point)}</span>"
        f"<span style='background:{color};color:#fff;border-radius:6px;"
        f"padding:3px 9px;font-weight:700;font-size:11px;'>{'/'.join(_sensor_names(point))}</span>"
        f"<span style='color:{_MUTED};font-size:12px;'>· Desplazamiento</span>"
        f"<span style='margin-left:auto;background:#e6f7ec;color:#0f7a3d;"
        f"border:1px solid #a7dcbd;border-radius:8px;padding:4px 12px;"
        f"font-weight:700;font-size:12px;white-space:nowrap;'>"
        f"<span style='color:#16a34a'>●</span>&nbsp; {when}</span>"
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
    d, tt = sel.get("day", ""), sel.get("time", "")
    if len(d) == 8 and len(tt) == 6:
        from datetime import datetime
        when = datetime(int(d[:4]), int(d[4:6]), int(d[6:]),
                        int(tt[:2]), int(tt[2:4]), int(tt[4:])
                        ).strftime("%d/%m/%Y %I:%M:%S %p")
    else:
        when = _lbl(sel)
    _chip(resolved, point, when)

    cached = _download_cached(sel["key"])
    if not cached:
        st.warning("No se pudo descargar esta captura.")
        return
    _render_capture(_cap_from_cached(cached), instance_id, point, resolved, when)


def _render_capture(cap: Capture, instance_id: str, point: str,
                    resolved: str = "", when: str = "") -> None:
    rpm = cap.rpm
    if rpm is None and cap.has(CH_KPH):
        rpm = rpm_from_keyphasor(cap.t, cap.get(CH_KPH))
    nx, ny = _sensor_names(point)
    # Encabezado que viaja DENTRO del gráfico (para que salga en el JPG)
    rpm_txt = f"{rpm:,.0f} RPM" if rpm else "— RPM"
    header = (f"{resolved} · {_point_label(point)} ({nx}/{ny}) · "
              f"Desplazamiento · {rpm_txt} · {when}")

    if rpm:
        st.markdown(
            f"<div style='color:#334155;font-size:13px;margin:0 0 8px'>"
            f"Velocidad de giro&nbsp; <b style='color:#0f172a;font-size:15px'>"
            f"{rpm:,.0f}</b> RPM</div>", unsafe_allow_html=True)

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
        _plot_waveform(cap, nx, ny, header)
    elif "Espectro" in view:
        _plot_spectrum(cap, rpm, nx, ny, header)
    else:
        _plot_orbit(cap, rpm, point, nx, ny, header)


# =========================================================
# Layout base
# =========================================================
# Barra mínima: SOLO el botón de descarga a imagen (JPG). Quita zoom/pan/±.
_PCFG = {
    "displayModeBar": True, "displaylogo": False, "scrollZoom": False,
    "modeBarButtonsToRemove": ["zoom2d", "pan2d", "select2d", "lasso2d",
                               "zoomIn2d", "zoomOut2d", "autoScale2d",
                               "resetScale2d", "toggleSpikelines",
                               "hoverClosestCartesian", "hoverCompareCartesian"],
    "toImageButtonOptions": {"format": "jpeg", "scale": 2,
                             "filename": "watermelon_grafico"},
}
_GRID_MAJ = "rgba(148,163,184,0.30)"
_GRID_MIN = "rgba(148,163,184,0.12)"


def _base_layout(fig, height=380, title=""):
    fig.update_layout(
        height=height, template="plotly_white",
        margin=dict(l=64, r=18, t=66 if title else 18, b=48),
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


def _title_html(title: str, header: str = "") -> str:
    """Título del gráfico + subtítulo con el encabezado (máquina·sensor·rpm·
    fecha). Va DENTRO de la figura para que aparezca en la imagen exportada."""
    if header:
        return (f"{title}<br><span style='font-size:11.5px;color:#475569;'>"
                f"{header}</span>")
    return title


def _hoverstyle(fig) -> None:
    """Cuadro de hover elegante (estilo Cursor A de la competencia)."""
    fig.update_layout(hoverlabel=dict(
        bgcolor="rgba(15,23,42,0.96)", bordercolor="rgba(255,255,255,0.30)",
        font=dict(color="#f8fafc", size=12.5, family="Arial"),
        align="left"))


def _nice_ceil(x: float) -> float:
    """Redondea hacia arriba a un tope 'bonito' (1/2/2.5/5/10 × 10^k).
    Ej.: 44→50, 29.7→30, 18→20."""
    import math
    if x is None or x <= 0:
        return 1.0
    e = math.floor(math.log10(x))
    base = 10 ** e
    for m in (1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10):
        if x <= m * base + 1e-9:
            return m * base
    return 10 * base


def _stats(v: np.ndarray) -> Tuple[float, float, float]:
    """(pp, RMS_AC, Crest Factor) como System1."""
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0, 0.0, 0.0
    ac = v - v.mean()
    rms = float(np.sqrt(np.mean(ac ** 2)))
    pk = float(np.max(np.abs(ac)))
    cf = pk / rms if rms > 1e-9 else 0.0
    return _pp(v), rms, cf


def _grid_xy(fig, row, dx_maj, dx_min, dy_maj, dy_min):
    """Estilo System1: X = solo TICKS sobre el eje (mayor+menor), sin líneas
    verticales; Y = solo líneas horizontales MAYORES (menos líneas)."""
    fig.update_xaxes(row=row, col=1, dtick=dx_maj, showgrid=False,
                     ticks="outside", ticklen=6,
                     tickcolor="rgba(15,23,42,0.45)",
                     minor=dict(dtick=dx_min, showgrid=False, ticks="outside",
                                ticklen=3, tickcolor="rgba(15,23,42,0.28)"))
    fig.update_yaxes(row=row, col=1, dtick=dy_maj, showgrid=True,
                     gridcolor=_GRID_MAJ, minor=dict(showgrid=False))


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
def _plot_waveform(cap: Capture, nx: str, ny: str, header: str = "") -> None:
    from plotly.subplots import make_subplots
    t = cap.t * 1000.0  # ms
    u = cap.unit_of(CH_X) or "µm"
    xmax = float(t[-1]) if t.size else 1.0
    chans = [(CH_X, _X_COLOR, nx), (CH_Y, _Y_COLOR, ny)]
    chans = [(c, col, nm) for c, col, nm in chans if cap.get(c) is not None]
    if not chans:
        st.info("Sin canales de onda.")
        return
    amax = _nice_ceil(max(float(np.nanmax(np.abs(cap.get(c)))) for c, _, _ in chans)
                      * 1.10)            # misma escala simétrica ambos + 10%
    dy_maj = _nice_ceil(amax / 3.0); dy_min = dy_maj / 5.0
    dx_maj = 10.0 if xmax >= 40 else 5.0; dx_min = dx_maj / 5.0
    stats = {c: _stats(cap.get(c)) for c, col, nm in chans}
    titles = [f"{nm}   ·   {stats[c][0]:.1f} {u} pp   ·   RMS {stats[c][1]:.1f} {u}"
              f"   ·   CF {stats[c][2]:.2f}" for c, col, nm in chans]
    fig = make_subplots(rows=len(chans), cols=1, shared_xaxes=True,
                        vertical_spacing=0.11, subplot_titles=titles)
    for i, (ch, color, nm) in enumerate(chans, start=1):
        v = cap.get(ch)
        pp, rms, cf = stats[ch]
        ht = (f"<b>{nm}</b> · %{{x:.2f}} ms<br>"
              f"<b>%{{y:.2f}} {u}</b><br>"
              f"pp {pp:.1f}  ·  RMS {rms:.1f}  ·  CF {cf:.2f}<extra></extra>")
        fig.add_scatter(x=t, y=v, mode="lines", name=nm, row=i, col=1,
                        line=dict(color=color, width=1.6), showlegend=False,
                        hovertemplate=ht)
        fig.update_yaxes(title_text=f"[{u}]", row=i, col=1, range=[-amax, amax],
                         zeroline=True, zerolinecolor="rgba(15,23,42,0.35)")
        fig.update_xaxes(range=[0, xmax], row=i, col=1)
        _grid_xy(fig, i, dx_maj, dx_min, dy_maj, dy_min)
    fig.update_xaxes(title_text="Tiempo [ms]", row=len(chans), col=1)
    _base_layout(fig, height=210 * len(chans) + 68, title=_title_html("Forma de onda", header))
    fig.update_layout(showlegend=False, hovermode="closest")
    _hoverstyle(fig)
    _style_subtitles(fig)
    st.plotly_chart(fig, use_container_width=True, config=_PCFG,
                    key=f"wm_dr_wf_{cap.point}_{cap.captured_at}")
    st.caption(f"Duración capturada: {xmax:.0f} ms · {cap.samples_per_rev or '—'} "
               f"muestras/vuelta. Más tiempo/resolución = más vueltas en la "
               f"captura de campo.")


# =========================================================
# ESPECTRO — un gráfico INDEPENDIENTE por sensor (alta resolución, CPM)
# =========================================================
def _plot_spectrum(cap: Capture, rpm: Optional[float], nx: str, ny: str,
                   header: str = "") -> None:
    from plotly.subplots import make_subplots
    fs = cap.fs_hz
    if not fs:
        st.info("Sin frecuencia de muestreo — no se puede calcular el espectro.")
        return
    unit = cap.unit_of(CH_X) or "µm"
    is_disp = _disp_type(unit) == "disp"
    ysuf = f"{unit} pp" if is_disp else f"{unit} pico"
    # Rango por tipo de medición (desplazamiento 60k / velocidad 300k / accel
    # 600k CPM), tope en Nyquist.
    fmax_cpm = min(_FMAX_CPM[_disp_type(unit)], fs / 2.0 * 60.0)
    dtick_cpm = 10000 if fmax_cpm <= 80000 else 100000
    chans = [(CH_X, _X_COLOR, nx), (CH_Y, _Y_COLOR, ny)]
    chans = [(c, col, nm) for c, col, nm in chans if cap.get(c) is not None]
    if not chans:
        st.info("Sin canales para el espectro.")
        return
    # calcular espectros + pico dominante (para el título/lectura tipo System1)
    data = []
    titles = []
    peak_max = 0.0
    for ch, color, nm in chans:
        freqs, amp = _hires_spectrum(cap.get(ch), fs)
        cpm = freqs * 60.0
        yv = amp * 2.0 if is_disp else amp
        if yv.size > 2:
            k = int(np.argmax(yv[1:]) + 1)
            titles.append(f"{nm}   ·   {yv[k]:.2f} {ysuf} @ {cpm[k]:,.0f} CPM")
            peak_max = max(peak_max, float(yv[k]))
        else:
            titles.append(nm)
        data.append((cpm, yv, color, nm))
    ymax = _nice_ceil(peak_max * 1.10)   # misma escala ambos + 10% de aire
    dy_maj = _nice_ceil(ymax / 4.0); dy_min = dy_maj / 5.0
    fig = make_subplots(rows=len(data), cols=1, shared_xaxes=True,
                        vertical_spacing=0.16, subplot_titles=titles)
    for i, (cpm, yv, color, nm) in enumerate(data, start=1):
        fig.add_scatter(x=cpm, y=yv, mode="lines", name=nm, row=i, col=1,
                        line=dict(color=color, width=1.2), showlegend=False,
                        hovertemplate="%{x:,.0f} CPM<br>%{y:.2f} " + ysuf +
                        "<extra>" + nm + "</extra>")
        # ejes explícitos → X e Y se cruzan en (0,0), misma escala Y ambos
        fig.update_xaxes(range=[0, fmax_cpm], tickformat=",d", row=i, col=1)
        fig.update_yaxes(title_text=f"[{ysuf}]", range=[0, ymax], row=i, col=1)
        _grid_xy(fig, i, dtick_cpm, dtick_cpm / 5.0, dy_maj, dy_min)
    fig.update_xaxes(title_text="Frecuencia [CPM]", row=len(data), col=1)
    _base_layout(fig, height=240 * len(data) + 68,
                 title=_title_html("Espectro (FFT)", header))
    fig.update_layout(showlegend=False, hovermode="closest")
    _hoverstyle(fig)
    _style_subtitles(fig)
    st.plotly_chart(fig, use_container_width=True, config=_PCFG,
                    key=f"wm_dr_sp_{cap.point}_{cap.captured_at}")
    st.caption(f"Rango 0–{fmax_cpm:,.0f} CPM ({_disp_type(unit)}). Lectura del "
               f"pico dominante de cada sensor en su título.")


# =========================================================
# ÓRBITA
# =========================================================
def _plot_orbit(cap: Capture, rpm: Optional[float], point: str,
                nx: str, ny: str, header: str = "") -> None:
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
    _base_layout(fig, height=490,
                 title=_title_html(f"Órbita · {_point_label(point)}", header))
    fig.update_layout(hovermode="closest")
    _hoverstyle(fig)
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
