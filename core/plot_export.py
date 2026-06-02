"""
core.plot_export
================

Helper COMPARTIDO para exportar figuras Plotly a PNG sin reventar la memoria
del worker (causa de los 502/OOM en Render — "Ran out of memory, used over
2GB" — al dar "Enviar a Reporte" / "Prepare PNG HD" con datos densos).

Estrategia (Ciclo 23.155):
  1. Decimar las trazas densas ANTES de pasarlas a kaleido:
       - cartesianas (x/y): envolvente MIN-MAX por bloques → preserva picos
         +/- (clave en vibración: peak, crest factor, transitorios).
       - polares (r/θ) u otras: decimación por stride (preserva la forma).
  2. Reconstruir un trace del MISMO tipo, soltando arrays accesorias
     (customdata/hovertext/text) que kaleido no necesita para el PNG.
  3. Renderizar con scale=1 (raster ~4× más liviano que scale=2).

Lo usan todos los módulos de export (Waveforms, Spectrum, Trends, Orbit,
Polar, Bode, Shaft Centerline, TSA, Order Tracking).
"""
from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go

_MAX_PTS = 4000


def _len(a) -> int:
    try:
        return len(a) if a is not None else 0
    except Exception:
        return 0


def _minmax_decimate(x: np.ndarray, y: np.ndarray, max_pts: int) -> Tuple[list, list]:
    """Envolvente min-max: por bloque conserva el punto de y mínimo y el de y
    máximo (preserva picos)."""
    n = len(y)
    if n <= max_pts:
        return list(x), list(y)
    n_blocks = max(1, max_pts // 2)
    step = int(np.ceil(n / n_blocks))
    xs: List[Any] = []
    ys: List[Any] = []
    for s in range(0, n, step):
        by = y[s:s + step]
        bx = x[s:s + step]
        if by.size == 0:
            continue
        for i in sorted({int(np.argmin(by)), int(np.argmax(by))}):
            xs.append(bx[i])
            ys.append(by[i])
    return xs, ys


def _stride_decimate(a: np.ndarray, b: np.ndarray, max_pts: int) -> Tuple[list, list]:
    """Decimación uniforme por stride (cuando la forma importa más que los
    picos, ej. curvas polares)."""
    n = len(a)
    if n <= max_pts:
        return list(a), list(b)
    step = int(np.ceil(n / max_pts))
    return list(a[::step]), list(b[::step])


def _lighten(tr) -> None:
    """Suelta arrays accesorias que kaleido no necesita para el PNG."""
    for attr in ("customdata", "hovertext", "text", "hovertemplate"):
        try:
            setattr(tr, attr, None)
        except Exception:
            pass


def _decimate_trace(tr, max_pts: int):
    """Versión decimada del trace conservando su TIPO. Si no hay nada que
    decimar, devuelve el trace original."""
    x = getattr(tr, "x", None)
    y = getattr(tr, "y", None)
    r = getattr(tr, "r", None)
    theta = getattr(tr, "theta", None)

    # Cartesiano (Scatter/Scattergl/Bar): min-max sobre y
    if _len(x) > max_pts and _len(x) == _len(y):
        try:
            xs, ys = _minmax_decimate(np.asarray(x), np.asarray(y, dtype=float), max_pts)
            t2 = tr.__class__(tr.to_plotly_json())
            t2.x = xs
            t2.y = ys
            _lighten(t2)
            return t2
        except Exception:
            return tr

    # Polar (Scatterpolar): stride sobre (theta, r)
    if _len(r) > max_pts and _len(r) == _len(theta):
        try:
            ths, rs = _stride_decimate(np.asarray(theta, dtype=float),
                                       np.asarray(r, dtype=float), max_pts)
            t2 = tr.__class__(tr.to_plotly_json())
            t2.theta = ths
            t2.r = rs
            _lighten(t2)
            return t2
        except Exception:
            return tr

    return tr


def downsample_fig_for_export(fig: go.Figure, max_pts: int = _MAX_PTS) -> go.Figure:
    """Copia liviana de la figura con las trazas densas decimadas (preserva
    layout: shapes, annotations, images, ejes)."""
    new = go.Figure(layout=fig.layout)
    for tr in fig.data:
        new.add_trace(_decimate_trace(tr, max_pts))
    return new


def fig_to_png_bytes(
    fig: go.Figure,
    *,
    width: int = 1600,
    height: int = 900,
    scale: int = 1,
    max_pts: int = _MAX_PTS,
) -> Tuple[Optional[bytes], Optional[str]]:
    """Exporta a PNG de forma SEGURA (anti-OOM). Devuelve (png_bytes, error).
    Nunca relanza: si falla, (None, mensaje)."""
    try:
        import plotly.io as pio
        safe_fig = downsample_fig_for_export(fig, max_pts=max_pts)
        png_bytes = pio.to_image(safe_fig, format="png",
                                 width=width, height=height, scale=scale)
        return png_bytes, None
    except Exception as e:
        return None, str(e)


def export_plot_png(fig) -> Optional[bytes]:
    """Compat (TSA / Order Tracking): devuelve solo los bytes del PNG, ahora
    por el camino seguro (downsample + scale=1). Antes era to_image directo a
    1920×1080 scale=2 → riesgo de OOM con datos densos."""
    return fig_to_png_bytes(fig, width=1920, height=1080, scale=1)[0]


__all__ = [
    "downsample_fig_for_export",
    "fig_to_png_bytes",
    "export_plot_png",
]
