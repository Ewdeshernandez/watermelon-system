"""
core.calibration.curve
=======================

Renderizado de las curvas de linealidad:

  - `linearity_curve_svg(...)`  → SVG puro para la web (sin dependencias),
    estilo enterprise. Puntos medidos + recta best-fit + badge PASA/FALLA.
  - `curve_png(...)`            → PNG (matplotlib) para el certificado PDF.
    Para proximidad reproduce el layout de la Figura 4 de API 670 (paneles
    apilados: error ISF, DSL y característica gap→voltaje).

Headless: nada de Streamlit. matplotlib se importa perezosamente (Agg).
"""
from __future__ import annotations

import io
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

_NAVY = "#0F1E3D"
_CYAN = "#1AAEE5"
_GREEN = "#16a34a"
_RED = "#dc2626"
_AMBER = "#D89B22"
_GRID = "#e2e8f0"
_MUTE = "#94a3b8"


# =====================================================================
# SVG para la web
# =====================================================================
def linearity_curve_svg(
    x: Sequence[float], y: Sequence[float], y_fit: Sequence[float],
    *, title: str = "Curva de linealidad", x_label: str = "Gap [mil]",
    y_label: str = "Salida [V]", verdict: str = "", badge_detail: str = "",
    width: int = 640, height: int = 360,
) -> str:
    """Curva XY (puntos + best-fit) en SVG, con badge de veredicto."""
    xa = [float(v) for v in x]
    ya = [float(v) for v in y]
    yf = [float(v) for v in y_fit] if y_fit is not None else []
    if not xa or not ya:
        return "<div style='color:#94a3b8'>Sin datos para graficar.</div>"

    ml, mr, mt, mb = 58, 24, 44, 46
    pw, ph = width - ml - mr, height - mt - mb
    xmin, xmax = min(xa), max(xa)
    ally = ya + yf
    ymin, ymax = min(ally), max(ally)
    # márgenes de datos
    xpad = (xmax - xmin) * 0.04 or 1.0
    ypad = (ymax - ymin) * 0.08 or 1.0
    xmin -= xpad; xmax += xpad; ymin -= ypad; ymax += ypad

    def sx(v: float) -> float:
        return ml + (v - xmin) / (xmax - xmin) * pw

    def sy(v: float) -> float:
        return mt + (1 - (v - ymin) / (ymax - ymin)) * ph

    out: List[str] = []
    # marco
    out.append(f'<rect x="{ml}" y="{mt}" width="{pw}" height="{ph}" '
               f'fill="#ffffff" stroke="{_GRID}" stroke-width="1"/>')
    # rejilla + ticks (5 en x, 5 en y)
    for i in range(6):
        gx = ml + pw * i / 5
        vx = xmin + (xmax - xmin) * i / 5
        out.append(f'<line x1="{gx:.1f}" y1="{mt}" x2="{gx:.1f}" y2="{mt+ph}" '
                   f'stroke="{_GRID}" stroke-width="0.6"/>')
        out.append(f'<text x="{gx:.1f}" y="{mt+ph+16:.1f}" text-anchor="middle" '
                   f'font-size="10" fill="{_MUTE}">{vx:.0f}</text>')
        gy = mt + ph * i / 5
        vy = ymax - (ymax - ymin) * i / 5
        out.append(f'<line x1="{ml}" y1="{gy:.1f}" x2="{ml+pw}" y2="{gy:.1f}" '
                   f'stroke="{_GRID}" stroke-width="0.6"/>')
        out.append(f'<text x="{ml-8:.1f}" y="{gy+3:.1f}" text-anchor="end" '
                   f'font-size="10" fill="{_MUTE}">{vy:.2f}</text>')

    # recta best-fit
    if yf:
        out.append(f'<line x1="{sx(xa[0]):.1f}" y1="{sy(yf[0]):.1f}" '
                   f'x2="{sx(xa[-1]):.1f}" y2="{sy(yf[-1]):.1f}" '
                   f'stroke="{_CYAN}" stroke-width="2" stroke-dasharray="6 4"/>')
    # línea que une puntos medidos
    pts = " ".join(f"{sx(xa[i]):.1f},{sy(ya[i]):.1f}" for i in range(len(xa)))
    out.append(f'<polyline points="{pts}" fill="none" stroke="{_NAVY}" '
               f'stroke-width="1.6" opacity="0.55"/>')
    # puntos
    for i in range(len(xa)):
        out.append(f'<circle cx="{sx(xa[i]):.1f}" cy="{sy(ya[i]):.1f}" r="3.6" '
                   f'fill="{_NAVY}" stroke="white" stroke-width="1"/>')

    # ejes labels
    out.append(f'<text x="{ml+pw/2:.1f}" y="{height-8:.1f}" text-anchor="middle" '
               f'font-size="11" fill="{_NAVY}" font-weight="600">{x_label}</text>')
    out.append(f'<text x="14" y="{mt+ph/2:.1f}" text-anchor="middle" '
               f'font-size="11" fill="{_NAVY}" font-weight="600" '
               f'transform="rotate(-90 14 {mt+ph/2:.1f})">{y_label}</text>')
    # título
    out.append(f'<text x="{ml}" y="20" font-size="13" fill="{_NAVY}" '
               f'font-weight="800">{title}</text>')
    # leyenda best-fit
    out.append(f'<line x1="{ml+pw-150:.0f}" y1="16" x2="{ml+pw-128:.0f}" y2="16" '
               f'stroke="{_CYAN}" stroke-width="2" stroke-dasharray="6 4"/>'
               f'<text x="{ml+pw-124:.0f}" y="19" font-size="10" fill="{_MUTE}">'
               f'best-fit</text>')

    # badge veredicto
    if verdict:
        ok = verdict.upper().startswith("PASA")
        col = _GREEN if ok else _RED
        bg = "#eafaf0" if ok else "#fdecec"
        out.append(f'<rect x="{ml+8}" y="{mt+8}" width="118" height="34" rx="7" '
                   f'fill="{bg}" stroke="{col}" stroke-width="1.4"/>')
        out.append(f'<text x="{ml+18}" y="{mt+24}" font-size="13" fill="{col}" '
                   f'font-weight="800">{"✓" if ok else "✗"} {verdict}</text>')
        if badge_detail:
            out.append(f'<text x="{ml+18}" y="{mt+37}" font-size="9" fill="{col}">'
                       f'{badge_detail}</text>')

    return (f'<div style="max-width:{width}px;margin:6px 0;">'
            f'<svg viewBox="0 0 {width} {height}" width="100%" '
            f'style="max-height:{height}px" xmlns="http://www.w3.org/2000/svg" '
            f'font-family="Inter,system-ui,sans-serif">{"".join(out)}</svg></div>')


# =====================================================================
# PNG (matplotlib) para el PDF
# =====================================================================
def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def proximity_png(analysis: Dict[str, Any], *, title: str = "") -> Optional[bytes]:
    """Certificado gráfico de proximidad estilo API 670 Fig. 4: paneles
    apilados de error ISF (%), DSL (unidad-x) y característica gap→voltaje."""
    try:
        plt = _mpl()
    except Exception:
        return None

    xu = analysis.get("x_unit", "mil")
    x = analysis["x"]; y = analysis["y"]; yfit = analysis["y_fit"]
    xmid = analysis["x_mid"]; isf_err = analysis["isf_err_pct"]
    dsl = analysis["dsl_x"]; isf_tol = analysis["isf_tol_pct"]
    dsl_tol = analysis["dsl_tol_x"]

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(6.6, 6.4), dpi=170,
        gridspec_kw={"height_ratios": [1, 1, 2.1], "hspace": 0.32})

    # --- Panel 1: error ISF ---
    ax1.axhspan(-isf_tol, isf_tol, color="#eafaf0", zorder=0)
    ax1.axhline(isf_tol, color=_GREEN, lw=1, ls="--")
    ax1.axhline(-isf_tol, color=_GREEN, lw=1, ls="--")
    ax1.axhline(0, color=_MUTE, lw=0.8)
    ax1.plot(xmid, isf_err, "o-", color=_NAVY, ms=4, lw=1.4)
    ax1.set_ylabel("ISF err\n(%)", fontsize=8)
    ax1.set_title("Incremental Scale Factor (±%) vs 200 mV/mil", fontsize=8.5,
                  color=_NAVY)
    ax1.set_ylim(-max(isf_tol * 1.8, max((abs(e) for e in isf_err), default=1) * 1.3),
                 max(isf_tol * 1.8, max((abs(e) for e in isf_err), default=1) * 1.3))
    ax1.tick_params(labelsize=7)
    ax1.grid(True, alpha=0.3)

    # --- Panel 2: DSL ---
    ax2.axhspan(-dsl_tol, dsl_tol, color="#eafaf0", zorder=0)
    ax2.axhline(dsl_tol, color=_GREEN, lw=1, ls="--")
    ax2.axhline(-dsl_tol, color=_GREEN, lw=1, ls="--")
    ax2.axhline(0, color=_MUTE, lw=0.8)
    ax2.plot(x, dsl, "s-", color=_AMBER, ms=4, lw=1.4)
    ax2.set_ylabel(f"DSL\n[{xu}]", fontsize=8)
    ax2.set_title(f"Deviation from Straight Line (±{dsl_tol:g} {xu})",
                  fontsize=8.5, color=_NAVY)
    lim = max(dsl_tol * 1.8, max((abs(d) for d in dsl), default=1) * 1.3)
    ax2.set_ylim(-lim, lim)
    ax2.tick_params(labelsize=7)
    ax2.grid(True, alpha=0.3)

    # --- Panel 3: característica gap→voltaje ---
    ax3.plot(x, yfit, "--", color=_CYAN, lw=1.6, label="best-fit")
    ax3.plot(x, y, "o-", color=_NAVY, ms=5, lw=1.6, label="medido")
    ax3.set_xlabel(f"Gap [{xu}]", fontsize=8.5)
    ax3.set_ylabel("Salida [V]", fontsize=8.5)
    ax3.set_title("Característica de transducción gap → voltaje", fontsize=8.5,
                  color=_NAVY)
    ax3.legend(fontsize=7.5, loc="best", frameon=False)
    ax3.tick_params(labelsize=7)
    ax3.grid(True, alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=10, fontweight="bold", color=_NAVY, y=0.99)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def amplitude_png(analysis: Dict[str, Any], *, title: str = "") -> Optional[bytes]:
    """Linealidad de amplitud: best-fit + residual %FS."""
    try:
        plt = _mpl()
    except Exception:
        return None
    x = analysis["x"]; y = analysis["y"]; yfit = analysis["y_fit"]
    dev = analysis["dev_pct_fs"]; tol = analysis["tol_pct"]
    lu = analysis.get("level_unit", ""); ou = analysis.get("output_unit", "")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.6, 5.0), dpi=170,
                                   gridspec_kw={"height_ratios": [2, 1],
                                                "hspace": 0.32})
    ax1.plot(x, yfit, "--", color=_CYAN, lw=1.6, label="best-fit")
    ax1.plot(x, y, "o-", color=_NAVY, ms=5, lw=1.6, label="medido")
    ax1.set_ylabel(f"Salida [{ou}]", fontsize=8.5)
    ax1.set_title("Linealidad de amplitud", fontsize=9, color=_NAVY)
    ax1.legend(fontsize=7.5, frameon=False)
    ax1.tick_params(labelsize=7); ax1.grid(True, alpha=0.3)

    ax2.axhspan(-tol, tol, color="#eafaf0", zorder=0)
    ax2.axhline(tol, color=_GREEN, lw=1, ls="--")
    ax2.axhline(-tol, color=_GREEN, lw=1, ls="--")
    ax2.axhline(0, color=_MUTE, lw=0.8)
    ax2.plot(x, dev, "s-", color=_AMBER, ms=4, lw=1.4)
    ax2.set_xlabel(f"Nivel [{lu}]", fontsize=8.5)
    ax2.set_ylabel("Desv.\n(%FS)", fontsize=8)
    lim = max(tol * 1.8, max((abs(d) for d in dev), default=1) * 1.3)
    ax2.set_ylim(-lim, lim)
    ax2.tick_params(labelsize=7); ax2.grid(True, alpha=0.3)
    if title:
        fig.suptitle(title, fontsize=10, fontweight="bold", color=_NAVY, y=0.99)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def frequency_png(analysis: Dict[str, Any], *, title: str = "") -> Optional[bytes]:
    """Respuesta en frecuencia: desviación en dB vs banda (semilog-x)."""
    try:
        plt = _mpl()
    except Exception:
        return None
    x = analysis["x"]; dev = analysis["dev_db"]; tol = analysis["tol_db"]
    band = analysis.get("band_hz")

    fig, ax = plt.subplots(figsize=(6.6, 3.6), dpi=170)
    ax.axhspan(-tol, tol, color="#eafaf0", zorder=0)
    ax.axhline(tol, color=_GREEN, lw=1, ls="--", label=f"±{tol:g} dB")
    ax.axhline(-tol, color=_GREEN, lw=1, ls="--")
    ax.axhline(0, color=_MUTE, lw=0.8)
    finite = [(x[i], dev[i]) for i in range(len(x)) if dev[i] not in (float("inf"), float("-inf"))]
    if finite:
        fx, fy = zip(*finite)
        ax.semilogx(fx, fy, "o-", color=_NAVY, ms=4, lw=1.5)
    if band:
        ax.axvline(band[0], color=_MUTE, lw=0.8, ls=":")
        ax.axvline(band[1], color=_MUTE, lw=0.8, ls=":")
    ax.set_xlabel("Frecuencia [Hz]", fontsize=8.5)
    ax.set_ylabel("Desviación [dB]", fontsize=8.5)
    ax.set_title("Respuesta en frecuencia", fontsize=9, color=_NAVY)
    ax.legend(fontsize=7.5, frameon=False)
    ax.tick_params(labelsize=7); ax.grid(True, alpha=0.3, which="both")
    if title:
        fig.suptitle(title, fontsize=10, fontweight="bold", color=_NAVY, y=1.02)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def curve_png(sensor_type: str, analysis: Dict[str, Any], *,
              kind: str = "linearity", title: str = "") -> Optional[bytes]:
    """Dispatcher: elige el gráfico según tipo de sensor / ensayo."""
    st = str(sensor_type).lower()
    if st == "proximity":
        return proximity_png(analysis, title=title)
    if kind == "frequency":
        return frequency_png(analysis, title=title)
    return amplitude_png(analysis, title=title)


__all__ = [
    "linearity_curve_svg", "proximity_png", "amplitude_png", "frequency_png",
    "curve_png",
]
