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
    keyphasor_edges, rpm_from_keyphasor,
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
    """Nº de cojinete de una ÓRBITA (proximidad): BRGn (turbina/gen) o GBn
    (gearbox). Los canales sueltos GB_4XA no tienen (devuelve None)."""
    m = re.match(r"(?:BRG|GB)(\d+)$", (point or "").upper())
    return int(m.group(1)) if m else None


def _gb_single(point: str) -> Optional[dict]:
    """Canal ÚNICO del gearbox: GB_4XA / GB_3YV → {bearing,axis,kind}."""
    m = re.match(r"GB_(\d+)([XY])([DAV])$", (point or "").upper())
    if not m:
        return None
    return {"bearing": m.group(1), "axis": m.group(2), "kind": m.group(3)}


def _is_gb(point: str) -> bool:
    return (point or "").upper().startswith("GB")


def _component_of(point: str) -> str:
    if _is_gb(point):
        return "GearBox"
    n = _brg_num(point)
    if n is None:
        return ""
    return "Turbina" if n <= 4 else "Generador"


# medida (por letra de tipo o por unidad): (es, unidad canónica)
_MEASURE_ES = {"D": ("Desplazamiento", "µm"), "A": ("Aceleración", "g"),
               "V": ("Velocidad", "mm/s")}
_DISP2ES = {"disp": ("Desplazamiento", "µm"), "accel": ("Aceleración", "g"),
            "vel": ("Velocidad", "mm/s")}


def _measure_of(point: str, unit: str = "") -> Tuple[str, str]:
    """(etiqueta ES, unidad) de la medición. Prioriza la unidad real de la
    captura; cae al sufijo del nombre del canal."""
    if unit:
        return _DISP2ES.get(_disp_type(unit), ("Desplazamiento", unit or "µm"))
    g = _gb_single(point)
    if g:
        return _MEASURE_ES.get(g["kind"], ("Desplazamiento", "µm"))
    return ("Desplazamiento", "µm")


def _point_label(point: str) -> str:
    g = _gb_single(point)
    if g:
        meas, _u = _MEASURE_ES.get(g["kind"], ("", ""))
        return f"GearBox · {g['bearing']}{g['axis']}{g['kind']} · {meas}"
    n = _brg_num(point)
    if n is None:
        return point
    return f"{_component_of(point)} · Cojinete {n}"


def _sensor_names(point: str) -> Tuple[str, str]:
    """Etiquetas de los sensores X/Y del cojinete, ej. '1X' / '1Y'."""
    g = _gb_single(point)
    if g:
        return f"{g['bearing']}{g['axis']}{g['kind']}", ""
    n = _brg_num(point)
    if n is None:
        return "X", "Y"
    return f"{n}X", f"{n}Y"


def _order_points(points: List[str]) -> List[str]:
    # órbitas primero (por nº), luego canales sueltos del gearbox por nombre
    return sorted(points, key=lambda p: (_brg_num(p) is None, _brg_num(p) or 999,
                                         _is_gb(p), p))


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
        point = st.selectbox("Punto", points, format_func=_point_label,
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

    cached = _download_cached(sel["key"])
    if not cached:
        st.warning("No se pudo descargar esta captura.")
        return
    try:
        _render_capture(_cap_from_cached(cached), instance_id, point, resolved,
                        when)
    except Exception as exc:  # noqa: BLE001  (nunca romper la pantalla)
        st.warning(f"No se pudo dibujar esta captura ({exc}).")


def _render_capture(cap: Capture, instance_id: str, point: str,
                    resolved: str = "", when: str = "") -> None:
    rpm = cap.rpm
    if rpm is None and cap.has(CH_KPH):
        rpm = rpm_from_keyphasor(cap.t, cap.get(CH_KPH))
    nx, ny = _sensor_names(point)
    # medida real (Desplazamiento µm / Aceleración g / Velocidad mm·s⁻¹) por la
    # unidad de la captura → funciona igual para gearbox mixto
    meas, _munit = _measure_of(point, cap.unit_of(CH_X) or "")
    has_orbit = cap.has(CH_Y)          # canal único (A/V) → sin órbita
    # ocultar el toolbar de Streamlit (botón expandir/fullscreen) — deja la
    # cámara de Plotly intacta (esa vive en el modebar de Plotly, no aquí)
    st.markdown(
        "<style>[data-testid='stElementToolbar'],"
        "[data-testid='StyledFullScreenButton'],"
        "button[title='View fullscreen'],button[title='Fullscreen']"
        "{display:none!important;visibility:hidden!important}</style>",
        unsafe_allow_html=True)
    # Subtítulo/identidad que viaja DENTRO de cada gráfico (sale en el JPG):
    # máquina · punto · medida · rpm · fecha (verde). Se pone en cada onda/espectro.
    rpm_txt = f"{rpm:,.0f} RPM" if rpm else "— RPM"
    sub = (f"<span style='color:#475569'>{resolved} · {_point_label(point)} · "
           f"{meas} · {rpm_txt}</span> &nbsp; "
           f"<span style='color:#16a34a'>● {when}</span>")
    tag = f"GB{_brg_num(point)}" if _is_gb(point) and _brg_num(point) \
        else (point if _is_gb(point) else f"Brg{_brg_num(point) or ''}")
    fbase = f"{resolved}_{tag}"

    views = _VIEWS if has_orbit else _VIEWS[:2]   # sin Órbita si no hay Y
    try:
        view = st.segmented_control("Vista", views, default=views[0],
                                    key=f"wm_dr_view_{instance_id}",
                                    label_visibility="collapsed")
    except Exception:  # noqa: BLE001
        view = st.radio("Vista", views, horizontal=True,
                        key=f"wm_dr_view_r_{instance_id}",
                        label_visibility="collapsed")
    view = view or views[0]
    if "Onda" in view:
        _plot_waveform(cap, nx, ny, sub, fbase)
    elif "Espectro" in view:
        _plot_spectrum(cap, rpm, nx, ny, sub, fbase)
    else:
        # Ángulos de montaje de las sondas (grados desde el TOP). Bently típico:
        # X a 45° derecha, Y a 45° izquierda. Configurable por máquina.
        oa, ob, oc = st.columns([1, 1, 2])
        ang_x = oa.number_input(f"{nx} · ° a la derecha del TOP", value=45,
                                min_value=0, max_value=180, step=5,
                                key=f"wm_dr_ax_{instance_id}")
        ang_y = ob.number_input(f"{ny} · ° a la izquierda del TOP", value=45,
                                min_value=0, max_value=180, step=5,
                                key=f"wm_dr_ay_{instance_id}")
        _plot_orbit(cap, rpm, point, nx, ny, sub, fbase, ang_x, ang_y)


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
        hovermode="closest", clickmode="event+select",
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


def _pin_annotation(fig, sel, color, xunit, xfmt, ysuf, extra=""):
    """Si el usuario clicó un punto (selección de Streamlit), CLAVA ahí un cuadro
    tipo Cursor A. Doble-clic en el gráfico limpia la selección → se quita."""
    pts = []
    try:
        seln = sel.get("selection") if hasattr(sel, "get") else \
            getattr(sel, "selection", None)
        if seln is not None:
            pts = (seln.get("points") if hasattr(seln, "get")
                   else getattr(seln, "points", None)) or []
    except Exception:  # noqa: BLE001
        pts = []
    if not pts:
        return
    p = pts[-1]
    try:
        xx = float(p["x"] if hasattr(p, "__getitem__") else getattr(p, "x"))
        yy = float(p["y"] if hasattr(p, "__getitem__") else getattr(p, "y"))
    except Exception:  # noqa: BLE001
        return
    xtxt = format(xx, xfmt)
    fig.add_annotation(
        x=xx, y=yy, showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.5,
        arrowcolor=color, ax=0, ay=-46,
        text=(f"<b>{yy:.2f} {ysuf}</b>  @  {xtxt} {xunit}"
              + (f"<br>{extra}" if extra else "")),
        align="left", bgcolor="rgba(15,23,42,0.96)", bordercolor="#ffffff",
        borderwidth=1, borderpad=6,
        font=dict(color="#f8fafc", size=11.5, family="Arial"))


def _chart(fig, key, cfg):
    """Renderiza con selección por punto (click fija, doble-clic limpia)."""
    try:
        return st.plotly_chart(fig, use_container_width=True, config=cfg,
                               key=key, on_select="rerun",
                               selection_mode="points")
    except TypeError:  # Streamlit viejo sin on_select
        return st.plotly_chart(fig, use_container_width=True, config=cfg, key=key)


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
def _cfg(fbase: str, kind: str) -> dict:
    """Config Plotly con nombre de archivo del export = máquina_Brg_sensor_tipo."""
    c = dict(_PCFG)
    c["toImageButtonOptions"] = {"format": "jpeg", "scale": 2,
                                 "filename": f"{fbase}_{kind}".replace(" ", "_")}
    return c


def _plot_waveform(cap: Capture, nx: str, ny: str, sub: str = "",
                   fbase: str = "wf") -> None:
    import plotly.graph_objects as go
    t = cap.t * 1000.0  # ms
    u = cap.unit_of(CH_X) or "µm"
    xmax = float(t[-1]) if t.size else 1.0
    chans = [(CH_X, _X_COLOR, nx), (CH_Y, _Y_COLOR, ny)]
    chans = [(c, col, nm) for c, col, nm in chans if cap.get(c) is not None]
    if not chans:
        st.info("Sin canales de onda.")
        return
    # misma escala simétrica para todos los sensores (comparable)
    amax = _nice_ceil(max(float(np.nanmax(np.abs(cap.get(c)))) for c, _, _ in chans)
                      * 1.10)
    dy_maj = _nice_ceil(amax / 3.0); dy_min = dy_maj / 5.0
    dx_maj = 10.0 if xmax >= 40 else 5.0; dx_min = dx_maj / 5.0
    for ch, color, nm in chans:              # UN gráfico independiente por sensor
        v = cap.get(ch)
        pp, rms, cf = _stats(v)
        ht = (f"<b>Cursor · {nm}</b><br>"
              f"<b>%{{y:.2f}} {u}</b>  @  %{{x:.2f}} ms<br>"
              f"────────────<br>"
              f"Crest Factor&nbsp;&nbsp;<b>{cf:.2f}</b><br>"
              f"Overall RMS&nbsp;&nbsp;<b>{rms:.1f} {u}</b><br>"
              f"Pico–pico&nbsp;&nbsp;<b>{pp:.1f} {u}</b><extra></extra>")
        title = _title_html(
            f"Forma de onda — {nm}   ·   {pp:.1f} {u} pp · RMS {rms:.1f} · "
            f"CF {cf:.2f}", sub)
        fig = go.Figure()
        fig.add_scatter(x=t, y=v, mode="lines", name=nm,
                        line=dict(color=color, width=1.6), showlegend=False,
                        hovertemplate=ht)
        fig.update_yaxes(title_text=f"[{u}]", range=[-amax, amax], zeroline=True,
                         zerolinecolor="rgba(15,23,42,0.35)", dtick=dy_maj,
                         gridcolor=_GRID_MAJ, minor=dict(showgrid=False))
        fig.update_xaxes(title_text="Tiempo [ms]", range=[0, xmax],
                         showgrid=False, ticks="outside", ticklen=6, dtick=dx_maj,
                         tickcolor="rgba(15,23,42,0.45)",
                         minor=dict(dtick=dx_min, showgrid=False, ticks="outside",
                                    ticklen=3, tickcolor="rgba(15,23,42,0.28)"))
        _base_layout(fig, height=300, title=title)
        fig.update_layout(showlegend=False, hovermode="closest")
        _hoverstyle(fig)
        key = f"wm_dr_wf_{cap.point}_{nm}_{cap.captured_at}"
        _pin_annotation(fig, st.session_state.get(key), color, "ms", ".2f", u,
                        extra=f"CF {cf:.2f}  ·  RMS {rms:.1f} {u}  ·  "
                        f"pp {pp:.1f} {u}")
        _chart(fig, key, _cfg(fbase, f"{nm}_onda"))


# =========================================================
# ESPECTRO — un gráfico INDEPENDIENTE por sensor (alta resolución, CPM)
# =========================================================
def _plot_spectrum(cap: Capture, rpm: Optional[float], nx: str, ny: str,
                   sub: str = "", fbase: str = "sp") -> None:
    import plotly.graph_objects as go
    fs = cap.fs_hz
    if not fs:
        st.info("Sin frecuencia de muestreo — no se puede calcular el espectro.")
        return
    unit = cap.unit_of(CH_X) or "µm"
    is_disp = _disp_type(unit) == "disp"
    ysuf = f"{unit} pp" if is_disp else f"{unit} pico"
    fmax_cpm = min(_FMAX_CPM[_disp_type(unit)], fs / 2.0 * 60.0)
    dtick_cpm = 10000 if fmax_cpm <= 80000 else 100000
    chans = [(CH_X, _X_COLOR, nx), (CH_Y, _Y_COLOR, ny)]
    chans = [(c, col, nm) for c, col, nm in chans if cap.get(c) is not None]
    if not chans:
        st.info("Sin canales para el espectro.")
        return
    # 1ª pasada: espectros + pico dominante → misma escala Y para ambos
    specs = []
    peak_max = 0.0
    for ch, color, nm in chans:
        freqs, amp = _hires_spectrum(cap.get(ch), fs)
        cpm = freqs * 60.0
        yv = amp * 2.0 if is_disp else amp
        pk_cpm = pk_amp = 0.0
        if yv.size > 2:
            k = int(np.argmax(yv[1:]) + 1)
            pk_cpm, pk_amp = float(cpm[k]), float(yv[k])
            peak_max = max(peak_max, pk_amp)
        specs.append((cpm, yv, color, nm, pk_cpm, pk_amp))
    ymax = _nice_ceil(peak_max * 1.10)
    dy_maj = _nice_ceil(ymax / 4.0)
    for cpm, yv, color, nm, pk_cpm, pk_amp in specs:   # figura por sensor
        title = _title_html(
            f"Espectro — {nm}   ·   {pk_amp:.2f} {ysuf} @ {pk_cpm:,.0f} CPM", sub)
        fig = go.Figure()
        fig.add_scatter(x=cpm, y=yv, mode="lines", name=nm,
                        line=dict(color=color, width=1.2), showlegend=False,
                        hovertemplate="%{x:,.0f} CPM<br><b>%{y:.2f} " + ysuf +
                        "</b><extra>" + nm + "</extra>")
        fig.update_xaxes(title_text="Frecuencia [CPM]", range=[0, fmax_cpm],
                         tickformat=",d", showgrid=False, ticks="outside",
                         ticklen=6, dtick=dtick_cpm, showline=True,
                         linecolor="#334155", linewidth=1.3, mirror=False,
                         zeroline=True, zerolinecolor="#334155", zerolinewidth=1.3,
                         tickcolor="rgba(15,23,42,0.45)",
                         minor=dict(dtick=dtick_cpm / 5.0, showgrid=False,
                                    ticks="outside", ticklen=3,
                                    tickcolor="rgba(15,23,42,0.28)"))
        fig.update_yaxes(title_text=f"[{ysuf}]", range=[0, ymax], dtick=dy_maj,
                         gridcolor=_GRID_MAJ, minor=dict(showgrid=False),
                         showline=True, linecolor="#334155", linewidth=1.3,
                         mirror=False, zeroline=True, zerolinecolor="#334155",
                         zerolinewidth=1.3)
        _base_layout(fig, height=300, title=title)
        fig.update_layout(showlegend=False, hovermode="closest")
        _hoverstyle(fig)
        key = f"wm_dr_sp_{cap.point}_{nm}_{cap.captured_at}"
        _pin_annotation(fig, st.session_state.get(key), color, "CPM", ",.0f", ysuf)
        _chart(fig, key, _cfg(fbase, f"{nm}_espectro"))


# =========================================================
# ÓRBITA
# =========================================================
def _plot_orbit(cap: Capture, rpm: Optional[float], point: str, nx: str, ny: str,
                sub: str = "", fbase: str = "orb", ang_x: int = 45,
                ang_y: int = 45) -> None:
    import plotly.graph_objects as go
    px, py = cap.get(CH_X), cap.get(CH_Y)
    if px is None or py is None:
        st.info("Faltan canales X/Y para la órbita.")
        return
    u = cap.unit_of(CH_X) or "µm"
    # NOTA fase: el export .KPH de cada sonda comparte el MISMO keyphasor, así que
    # px[i] y py[i] son el mismo ángulo de eje → la fase entre sondas YA está
    # preservada (medido: dphi 1X ≈ 88° en cuadratura). NO se rola nada. (El viejo
    # np.roll(90°) aplastaba la órbita a una línea; validado con data real.)
    # --- Reconstrucción en coordenadas de MÁQUINA (H→derecha, V→arriba) ---
    # Cada sonda mide en su ángulo real desde el TOP (X a la derecha, Y a la
    # izquierda). Se resuelve el sistema 2×2 para el H y V verdaderos.
    tx = np.radians(float(ang_x)); ty = np.radians(-float(ang_y))
    a, b = np.sin(tx), np.cos(tx)
    c, d = np.sin(ty), np.cos(ty)
    det = a * d - b * c
    if abs(det) < 1e-6:
        H, V = px, py
    else:
        H = (d * px - b * py) / det
        V = (-c * px + a * py) / det
    spr = cap.samples_per_rev
    # órbita síncrona 1X = promedio de TODAS las vueltas.
    Hc, Vc = H, V          # una vuelta CERRADA (para área/sentido/escala/pp)
    Ho, Vo = H, V          # trazo a DIBUJAR (abierto, con hueco en el keyphasor)
    kph_h = kph_v = None
    if spr and spr > 4 and H.size >= 2 * spr:
        nrev = H.size // spr
        Ha = H[:nrev * spr].reshape(nrev, spr).mean(axis=0)
        Va = V[:nrev * spr].reshape(nrev, spr).mean(axis=0)
        kph_h, kph_v = float(Ha[0]), float(Va[0])      # muestra 0 = keyphasor
        Hc = np.append(Ha, Ha[0]); Vc = np.append(Va, Va[0])
        # trazo ABIERTO: deja un pequeño hueco alrededor del keyphasor → no se
        # pega al punto brillante; marca inicio/fin como Bently.
        gap = max(1, spr // 24)
        Ho, Vo = Ha[gap:], Va[gap:]
    area = float(np.sum(Hc[:-1] * Vc[1:] - Hc[1:] * Vc[:-1])) if Hc.size > 2 else 0
    sentido = "↺ CCW" if area > 0 else "↻ CW"
    # escala: muestra la BANDA de todas las vueltas (percentil 99 para evitar
    # picos) → llena el marco como System1
    rad = np.hypot(H, V)
    amax = _nice_ceil(max(float(np.percentile(rad, 99)),
                          float(np.nanmax(np.abs(Hc))),
                          float(np.nanmax(np.abs(Vc))), 1.0) * 1.12)
    dmaj = _nice_ceil(amax / 3.0)
    amp_pp = max(_pp(Hc), _pp(Vc))

    fig = go.Figure()
    fig.add_hline(y=0, line=dict(color="rgba(15,23,42,0.18)", width=1))
    fig.add_vline(x=0, line=dict(color="rgba(15,23,42,0.18)", width=1))
    # posición física real de cada sonda (línea tenue + etiqueta)
    for ang, nm in ((tx, nx), (ty, ny)):
        hx, vy = np.sin(ang) * amax, np.cos(ang) * amax
        fig.add_scatter(x=[0, hx], y=[0, vy], mode="lines", showlegend=False,
                        hoverinfo="skip",
                        line=dict(color="rgba(100,116,139,0.45)", width=1,
                                  dash="dot"))
        fig.add_annotation(x=hx, y=vy, text=f"<b>{nm}</b>", showarrow=False,
                           font=dict(color=_MUTED, size=11),
                           xshift=int(np.sin(ang) * 12),
                           yshift=int(np.cos(ang) * 12))
    # banda de todas las vueltas (azul fino, estilo System1)
    fig.add_scatter(x=H, y=V, mode="lines", name="Vueltas", hoverinfo="skip",
                    line=dict(color="rgba(37,99,235,0.45)", width=0.8))
    # órbita promedio (síncrona 1X) — protagonista, ABIERTA (hueco en keyphasor)
    fig.add_scatter(x=Ho, y=Vo, mode="lines", name="Órbita 1X",
                    line=dict(color=_ORBIT_FILT, width=2.4),
                    hovertemplate="H %{x:.1f} · V %{y:.1f} " + u + "<extra></extra>")
    # keyphasor: UN punto brillante donde ARRANCA la órbita (marca de fase). El
    # hueco antes de él + hacia dónde sale el trazo indican el sentido de giro.
    if kph_h is not None:
        fig.add_scatter(x=[kph_h], y=[kph_v], mode="markers", name="Keyphasor",
                        hovertemplate="Keyphasor (inicio)<extra></extra>",
                        marker=dict(color=_KPH_COLOR, size=11,
                                    line=dict(width=1.6, color="white")))
    ttl = f"Órbita — {amp_pp:.1f} {u} pp   ·   {sentido}"
    _base_layout(fig, height=620, title=_title_html(ttl, sub))
    # gráfico CUADRADO (como System1): ancho fijo = alto, no estirar
    fig.update_layout(hovermode="closest", showlegend=False, width=620,
                      margin=dict(l=60, r=20, t=66, b=52))
    _hoverstyle(fig)
    # como System1: SIN cuadrícula de fondo — solo cross central (H=0,V=0) +
    # ticks mayor/menor sobre los ejes
    axkw = dict(range=[-amax, amax], zeroline=False, showgrid=False,
                dtick=dmaj, ticks="outside", ticklen=6,
                tickcolor="rgba(15,23,42,0.45)",
                minor=dict(dtick=dmaj / 5.0, showgrid=False, ticks="outside",
                           ticklen=3, tickcolor="rgba(15,23,42,0.28)"))
    fig.update_xaxes(title="Horizontal [%s] →" % u, scaleanchor="y",
                     scaleratio=1, **axkw)
    fig.update_yaxes(title="Vertical [%s] ↑" % u, **axkw)
    oc1, oc2, oc3 = st.columns([1, 3, 1])   # centrar el cuadrado
    with oc2:
        st.plotly_chart(fig, use_container_width=False,
                        config=_cfg(fbase, "orbita"),
                        key=f"wm_dr_orb_{cap.point}_{cap.captured_at}")


def _style_subtitles(fig) -> None:
    """Deja los títulos de subplot alineados a la izquierda y discretos."""
    for ann in fig.layout.annotations:
        ann.update(x=0, xanchor="left", font=dict(size=12, color=_INK,
                   family="Arial"))


__all__ = ["render_dynamic_raw"]
