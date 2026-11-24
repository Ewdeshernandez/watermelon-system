"""
core.rd_figures — Motor de gráficos de ROTODINÁMICA (fuente única).
====================================================================

Forma de onda · Espectro · Órbita a partir de los Channel/Bearing de
``core.analysis.model``. Lo usan:

  · Rotodinámica (pages/02_Analysis.py) — vista del analista.
  · Live Monitoring (core.rd_live_mirror) — COPIA ESPEJO para el cliente:
    la misma captura procesada con el mismo motor, sin re-implementar nada.

Sin streamlit (plotly + numpy) → testeable headless. Los builders devuelven
(figura, filas_de_tabla) para que cada página decida cómo mostrar la tabla.
"""
from __future__ import annotations

import math
import re
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import plotly.graph_objects as go

NAVY = "#12305e"
KPH_GREEN = "#1f9d55"
PAL = ["#12305e", "#e8890c", "#1f9d55", "#8b5cf6", "#0ea5e9", "#dc3545",
       "#84cc16", "#d946ef", "#14b8a6", "#f59e0b"]
NOBAR = {"displayModeBar": False}
_GRID = "rgba(120,140,170,.12)"

# Montaje de sondas por cojinete (SGT-300A/B, Bently 0°=arriba): generador (5/6)
# y gearbox (3/4) ARRIBA 45°R/45°L; turbina (1/2) ABAJO 135°R/135°L.
GEN_BEARINGS = {"5", "6"}
PROBES_UP_45 = {"3", "4", "5", "6"}

SPECTRUM_CPM_MAX = {"disp": 60000, "vel": 300000, "accel": 600000}


# ---------------------------------------------------------------- comunes
def layout(fig: go.Figure, title: str, h: int = 440) -> go.Figure:
    fig.update_layout(template="plotly_white", height=h,
                      title=dict(text=title, font=dict(size=15)),
                      margin=dict(l=62, r=24, t=48, b=48),
                      font=dict(family="IBM Plex Sans"), showlegend=False)
    return fig


def bearing_id(*vals) -> str:
    """Primer número de cojinete hallado en un bearing/label/nombre ('5XD' → '5')."""
    for v in vals:
        m = re.search(r"\d+", str(v or ""))
        if m:
            return m.group(0)
    return ""


def probe_angles(bid) -> Tuple[float, float, str]:
    """(ang_x, ang_y, zona) por nº de cojinete."""
    if str(bid) in PROBES_UP_45:
        return 45.0, 45.0, ("generador" if str(bid) in GEN_BEARINGS else "gearbox")
    return 135.0, 135.0, "turbina"


def bearing_angles(bb) -> Tuple[float, float, str]:
    return probe_angles(bearing_id(getattr(bb.x, "bearing", ""), getattr(bb.y, "bearing", ""),
                                   bb.x.label, bb.y.label, bb.name))


def spectrum_kind(unit: str) -> str:
    u = (unit or "").lower()
    if "mm/s" in u:
        return "vel"
    if u.strip() in ("g", "g pk", "g rms") or "m/s2" in u or "m/s²" in u:
        return "accel"
    return "disp"


def fmt_dt(s) -> str:
    """'7/12/2026 6:50:38 AM' → '12/07/2026 06:50' (24h, día/mes/año)."""
    from datetime import datetime
    s = str(s or "").strip()
    for _f in ("%m/%d/%Y %I:%M:%S %p", "%m/%d/%Y %I:%M %p",
               "%m/%d/%Y %H:%M:%S", "%m/%d/%Y %H:%M"):
        try:
            return datetime.strptime(s, _f).strftime("%d/%m/%Y %H:%M")
        except ValueError:
            continue
    return s


# ---------------------------------------------------------------- onda
def waveform_figure(channels: Sequence, kph: bool = True, title: str = "Forma de onda",
                    h: int = 500, time_label: str = "Tiempo (ms)",
                    amp_label: str = "Amplitud") -> Tuple[go.Figure, List[list], str]:
    """Varios canales en el mismo eje de tiempo + marcas de keyphasor (1/vuelta).
    Devuelve (fig, filas [Sensor, pp, RMS, Pico, Crest, RPM], unidad)."""
    from core.waveform_metrics import compute_waveform_metrics
    fig = go.Figure()
    units, rows, tmax, ymax = set(), [], 0.0, 0.0
    for i, ch in enumerate(channels):
        vv = ch.v
        t_ms = ch.t * 1000.0 if ch.t.size else np.arange(vv.size) * 1.0
        if t_ms.size:
            tmax = max(tmax, float(t_ms[-1]))
        vf = vv[np.isfinite(vv)]
        if vf.size:
            ymax = max(ymax, float(np.max(np.abs(vf))))
        col = PAL[i % len(PAL)]
        fig.add_trace(go.Scatter(
            x=t_ms, y=vv, mode="lines", name=ch.label,
            line=dict(color=col, width=1.4, shape="spline", smoothing=0.4),
            hovertemplate=f"<b>{ch.label}</b><br>%{{x:.2f}} ms<br>"
                          f"%{{y:.2f}} {ch.unit or ''}<extra></extra>"))
        units.add(ch.unit or "")
        if kph and ch.rpm and t_ms.size:
            Tms = 60000.0 / ch.rpm
            if Tms < (t_ms[-1] - t_ms[0]):
                kts = np.arange(float(t_ms[0]), float(t_ms[-1]), Tms)
                kvs = np.interp(kts, t_ms, vv)
                fig.add_trace(go.Scatter(
                    x=kts, y=kvs, mode="markers",
                    marker=dict(color=col, size=7, symbol="circle",
                                line=dict(color="white", width=1)),
                    name=f"KPH {ch.label}", showlegend=False, hoverinfo="skip"))
        mx = compute_waveform_metrics(vv) or {}
        u = ch.unit or ""
        rows.append([ch.label, f"{mx.get('peak_to_peak', 0):.2f} {u}",
                     f"{mx.get('rms', 0):.2f} {u}", f"{mx.get('peak', 0):.2f} {u}",
                     f"{mx.get('crest_factor', 0):.2f}",
                     f"{ch.rpm:,.0f}" if ch.rpm else "—"])
    wu = next(iter(units)) if len(units) == 1 else "mixto"
    fig = layout(fig, title, h=h)
    ym = ymax * 1.08 if ymax else None
    # X arranca en 0 hasta el fin de la captura; Y centrado en 0 y simétrico.
    fig.update_xaxes(title=time_label, showgrid=True, gridcolor=_GRID, showline=True,
                     linecolor="#d9e1ec", zeroline=False,
                     range=[0, tmax] if tmax else None,
                     ticks="outside", ticklen=4, tickcolor="#d9e1ec")
    fig.update_layout(
        yaxis=dict(title=f"{amp_label} ({wu or 'a.u.'})", showgrid=True, gridcolor=_GRID,
                   zeroline=True, zerolinecolor="rgba(120,140,170,.45)", zerolinewidth=1.2,
                   range=[-ym, ym] if ym else None,
                   ticks="outside", ticklen=4, tickcolor="#d9e1ec"),
        showlegend=True,
        legend=dict(orientation="h", yanchor="top", y=-0.14, xanchor="center",
                    x=0.5, font=dict(size=12)),
        margin=dict(l=64, r=28, t=52, b=72), hovermode="x unified",
        plot_bgcolor="white", paper_bgcolor="white")
    return fig, rows, wu


# ---------------------------------------------------------------- espectro
def channel_spectrum(ch):
    """(cpm, amp_display, fr_hz, kind, mode, unit) o None. FFT Hann, cero-a-pico
    corregido; desplazamiento en pp, velocidad/aceleración en RMS."""
    v = ch.v[np.isfinite(ch.v)]
    fs = ch.fs_hz
    if v.size < 8 or not fs:
        return None
    w = v - np.mean(v)
    win = np.hanning(w.size)
    am = np.abs(np.fft.rfft(w * win)) * 2.0 / np.sum(win)
    fr = np.fft.rfftfreq(w.size, d=1.0 / fs)
    kind = spectrum_kind(ch.unit)
    mode = "pp" if kind == "disp" else "rms"
    ampd = am * (2.0 if mode == "pp" else 1.0 / np.sqrt(2.0))
    ud = ch.unit or ("µm pp" if kind == "disp" else "mm/s rms" if kind == "vel" else "g")
    return fr * 60.0, ampd, fr, kind, mode, ud


def spectrum_figure(channels: Sequence, xmax: Optional[float] = -1, title: str = "Espectro (FFT)",
                    h: int = 500, freq_label: str = "Frecuencia (cpm)",
                    amp_label: str = "Amplitud"):
    """Espectros superpuestos en cpm con órdenes 1X-4X del primer keyphasor.
    xmax=-1 → automático por tipo (60k despl / 300k vel / 600k accel); None → completo.
    Devuelve (fig | None, filas [Sensor, RPM, Overall, 1X, 2X, 3X, 4X])."""
    fig = go.Figure()
    units, rows, cpmax_auto, ord_ref = set(), [], 0, None
    for i, ch in enumerate(channels):
        r = channel_spectrum(ch)
        if r is None:
            continue
        cpm, ampd, fr, kind, mode, ud = r
        units.add(ud)
        cpmax_auto = max(cpmax_auto, SPECTRUM_CPM_MAX[kind])
        col = PAL[i % len(PAL)]
        fig.add_trace(go.Scatter(
            x=cpm, y=ampd, mode="lines", name=ch.label,
            line=dict(color=col, width=1.2),      # sin spline: picos nítidos
            hovertemplate=f"<b>{ch.label}</b><br>%{{x:,.0f}} cpm<br>"
                          f"%{{y:.2f}} {ud}<extra></extra>"))
        if ch.rpm:
            if ord_ref is None:
                ord_ref = ch.rpm

            def _aat(fh, _fr=fr, _a=ampd):
                k = int(np.argmin(np.abs(_fr - fh)))
                lo, hi = max(0, k - 2), min(_a.size, k + 3)
                return float(np.max(_a[lo:hi])) if hi > lo else 0.0
            f1 = ch.rpm / 60.0
            v = ch.v[np.isfinite(ch.v)]
            ov = float(np.ptp(v)) if mode == "pp" else float(np.std(v))
            rows.append([ch.label, f"{ch.rpm:,.0f}", f"{ov:.2f} {ch.unit or ''}",
                         f"{_aat(f1):.2f}", f"{_aat(2 * f1):.2f}",
                         f"{_aat(3 * f1):.2f}", f"{_aat(4 * f1):.2f}"])
    if not fig.data:
        return None, []
    _xmax = cpmax_auto if xmax == -1 else xmax
    su = next(iter(units)) if len(units) == 1 else "mixto"
    if ord_ref:
        for k, ol in [(1, "1X"), (2, "2X"), (3, "3X"), (4, "4X")]:
            xo = k * ord_ref
            if _xmax is None or xo <= _xmax:
                fig.add_vline(x=xo, line=dict(color="#8090a6", width=1, dash="dot"),
                              annotation_text=ol, annotation_position="top",
                              annotation_font_size=10)
    fig = layout(fig, title, h=h)
    fig.update_xaxes(title=freq_label, showgrid=True, gridcolor=_GRID, showline=True,
                     linecolor="#d9e1ec", zeroline=False,
                     range=[0, _xmax] if _xmax else None)
    fig.update_layout(
        yaxis=dict(title=f"{amp_label} ({su})", showgrid=True, gridcolor=_GRID,
                   rangemode="tozero"),
        showlegend=True,
        legend=dict(orientation="h", yanchor="top", y=-0.14, xanchor="center",
                    x=0.5, font=dict(size=12)),
        margin=dict(l=64, r=28, t=52, b=72), hovermode="x unified",
        plot_bgcolor="white", paper_bgcolor="white")
    return fig, rows


# ---------------------------------------------------------------- órbita
def orbit_data(b, fmode: str, ang_x: float, ang_y: float) -> dict:
    """Órbita de un cojinete. Cálculo = core.orbit_render.compute_orbit (el mismo
    de los reportes)."""
    from core.orbit_render import compute_orbit
    _time = getattr(b.x, "cap_time", "") or getattr(b.y, "cap_time", "")
    _mach = getattr(b.x, "machine", "") or getattr(b.y, "machine", "")
    _stamp = " · ".join(x for x in (_mach, fmt_dt(_time)) if x)
    d = compute_orbit(
        b.x.v, b.y.v,
        fs_hz=float(b.x.fs_hz or b.y.fs_hz or 0),
        spr=int(b.x.spr or b.y.spr or 0),
        rpm=float(b.x.rpm or 0),
        bearing_label=(b.x.label or b.name or ""),
        x_label=(b.x.label or "X"), y_label=(b.y.label or "Y"),
        unit=(b.x.unit or "µm"), machine=_mach, stamp=_stamp,
        fmode=fmode, ang_x=ang_x, ang_y=ang_y)
    d["time"] = _time
    return d


def _keyphasor_traces(d, add, size_many=6, size_one=12, w_one=1.2):
    km = d["kmarks"]
    if not km:
        return
    kx = [p[0] for p in km]
    ky = [p[1] for p in km]
    if d["order"] == 0 and len(km) > 1:
        add(go.Scatter(x=kx, y=ky, mode="markers",
                       marker=dict(color=KPH_GREEN, size=size_many,
                                   line=dict(color="white", width=0.8)), showlegend=False))
        return
    ks = (d["smax"] if d["smax"] else 1.0) * 0.07
    for th in range(0, 360, 60):
        r = np.radians(th)
        add(go.Scatter(x=[kx[0], kx[0] + ks * np.sin(r)], y=[ky[0], ky[0] + ks * np.cos(r)],
                       mode="lines", line=dict(color=KPH_GREEN, width=w_one),
                       hoverinfo="skip", showlegend=False))
    add(go.Scatter(x=kx[:1], y=ky[:1], mode="markers",
                   marker=dict(color=KPH_GREEN, size=size_one,
                               line=dict(color="white", width=1.5)), showlegend=False))


def orbit_figure(b, d, R, ang_x, ang_y, fmode, h=560, compact=False,
                 title: Optional[str] = None) -> go.Figure:
    """UNA órbita con escala forzada R. Giro real (API 670) = texto CW/CCW."""
    from core.orbit_render import add_probe_glyphs
    fig = go.Figure()
    add_probe_glyphs(fig, R, ang_x, ang_y, b.x.label, b.y.label,
                     with_cone=not compact, with_top=False)
    fig.add_trace(go.Scatter(x=d["xv"], y=d["yv"], mode="lines",
                  line=dict(color=NAVY, width=1.6), name=b.name, showlegend=False))
    _keyphasor_traces(d, fig.add_trace)
    if d["sp"] is not None and not compact:
        fig.add_annotation(x=d["sp"][0], y=d["sp"][1], text=f"KΦ {d['kpos']}", showarrow=False,
                           yshift=-14, font=dict(color=KPH_GREEN, size=10, family="IBM Plex Mono"))
    fig.add_annotation(xref="paper", yref="paper", x=0.985, y=0.985, xanchor="right",
                       yanchor="top", text=f"<b>{d['rot_api']}</b>", showarrow=False,
                       font=dict(color="#64748b", size=13, family="IBM Plex Mono"),
                       bgcolor="rgba(255,255,255,0.65)")
    if d.get("stamp"):
        fig.add_annotation(xref="paper", yref="paper", x=0.015, y=0.015, xanchor="left",
                           yanchor="bottom", text=d["stamp"], showarrow=False,
                           font=dict(color="#94a3b8", size=10, family="IBM Plex Mono"))
    rng = R * 1.5 if R else None
    fig.update_xaxes(title=(b.x.unit or "µm") if not compact else "", scaleanchor="y",
                     scaleratio=1, zeroline=True, zerolinecolor="rgba(120,140,170,.45)",
                     gridcolor=_GRID, range=[-rng, rng] if rng else None)
    fig.update_yaxes(title=(b.y.unit or "µm") if not compact else "", zeroline=True,
                     zerolinecolor="rgba(120,140,170,.45)", gridcolor=_GRID,
                     range=[-rng, rng] if rng else None)
    if title is None:
        title = (f"{b.name} · {fmode}" if compact
                 else f"Órbita · {b.name} · {fmode}  ·  precesión {d['precd']}")
    fig = layout(fig, title, h=h)
    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", showlegend=False)
    return fig


def orbit_add_cell(fig, bb, d, R, ang_x, ang_y, row, col) -> None:
    """UNA órbita en la celda (row, col) de un subplot, con escala R."""
    from core.orbit_render import add_probe_glyphs
    rc = dict(row=row, col=col)
    add_probe_glyphs(fig, R, ang_x, ang_y, bb.x.label, bb.y.label,
                     with_top=False, with_cone=False, row=row, col=col)
    fig.add_trace(go.Scatter(x=d["xv"], y=d["yv"], mode="lines",
                  line=dict(color=NAVY, width=1.6), showlegend=False), **rc)
    _keyphasor_traces(d, lambda tr: fig.add_trace(tr, **rc), size_many=5, size_one=11, w_one=1.1)
    fig.add_annotation(x=R * 1.33, y=R * 1.33, text=f"<b>{d['rot_api']}</b>", showarrow=False,
                       font=dict(color="#64748b", size=12, family="IBM Plex Mono"),
                       bgcolor="rgba(255,255,255,0.65)", **rc)


def orbit_grid_figure(datas: Sequence[Tuple[object, dict]], R: float, fmode: str,
                      ang_for: Callable, title: Optional[str] = None,
                      cell_title: Optional[Callable] = None, ncol: int = 2) -> go.Figure:
    """Varias órbitas, MISMA escala R (comparar tamaños a ojo)."""
    from plotly.subplots import make_subplots
    nrow = math.ceil(len(datas) / ncol)
    if cell_title is None:
        def cell_title(bb, d):
            return (f"{bb.name} · {d['prec']} · giro {d['rot_api']}"
                    + (f"<br><span style='font-size:11px;color:#94a3b8'>{d['stamp']}</span>"
                       if d.get("stamp") else ""))
    figM = make_subplots(rows=nrow, cols=ncol,
                         subplot_titles=[cell_title(bb, d) for bb, d in datas],
                         horizontal_spacing=0.08,
                         vertical_spacing=0.18 if nrow > 1 else 0.12)
    L = R * 1.5
    for i, (bb, d) in enumerate(datas):
        r, c = i // ncol + 1, i % ncol + 1
        orbit_add_cell(figM, bb, d, R, *ang_for(bb), r, c)
        idx = (r - 1) * ncol + c
        yid = "y" if idx == 1 else f"y{idx}"
        figM.update_xaxes(range=[-L, L], scaleanchor=yid, scaleratio=1, row=r, col=c,
                          zeroline=True, zerolinecolor="rgba(120,140,170,.4)",
                          gridcolor="rgba(120,140,170,.1)")
        figM.update_yaxes(range=[-L, L], row=r, col=c, zeroline=True,
                          zerolinecolor="rgba(120,140,170,.4)",
                          gridcolor="rgba(120,140,170,.1)")
    figM.update_layout(
        template="plotly_white", height=360 * nrow + 40, showlegend=False,
        title=dict(text=title if title is not None else f"Órbitas · {fmode}",
                   font=dict(color=NAVY, size=16)),
        font=dict(family="IBM Plex Sans"), plot_bgcolor="white",
        paper_bgcolor="white", margin=dict(l=55, r=35, t=66, b=45))
    return figM


def orbit_row(bb, d) -> list:
    """[Cojinete, RPM, Precesión, Giro, Eje mayor, Eje menor, Elipticidad, Smax]."""
    u = bb.x.unit or "µm"
    return [bb.name, f"{d['rpm']:,.0f}" if d["rpm"] else "—", d["precd"],
            f"{d['rot_api']} ({d['maq']})", f"{d['major']:.1f} {u}",
            f"{d['minor']:.1f} {u}", f"{d['ellip']:.2f}", f"{d['smax']:.1f} {u}"]


__all__ = [
    "PAL", "NAVY", "NOBAR", "layout", "bearing_id", "probe_angles", "bearing_angles",
    "spectrum_kind", "fmt_dt", "waveform_figure", "channel_spectrum", "spectrum_figure",
    "orbit_data", "orbit_figure", "orbit_add_cell", "orbit_grid_figure", "orbit_row",
]
