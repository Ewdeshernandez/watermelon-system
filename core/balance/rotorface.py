"""
core/balance/rotorface.py — Vista polar 2D (SVG) del plano de balanceo
======================================================================

Alternativa liviana e instantánea al rotor 3D: una "cara del rotor" en SVG
puro (sin Plotly, sin WebGL) que muestra, por plano:

  - El vector de VIBRACIÓN medido (flecha roja) — aparece al cargar el dato.
  - El CONTRAPESO de corrección (diamante del color del plano) — al calcular.
  - Rejilla polar con marcas 0/90/180/270° (0° arriba, TDC).

Se renderiza con st.markdown(svg, unsafe_allow_html=True). Cero costo: es una
imagen vectorial estática (no se puede "colgar").

Convención: 0° arriba; ángulo creciente en sentido horario en pantalla
(x = cx + L·sinθ, y = cy − L·cosθ).
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

_NAVY = "#0F1E3D"
_MUTE = "#94a3b8"
_GRID = "#e2e8f0"
_VIB = "#ef4444"
_COLORS = {"cyan": "#1AAEE5", "green": "#16a34a", "amber": "#D89B22"}
_FILLS = {"cyan": "#eaf7fd", "green": "#eafaf0", "amber": "#fdf6e7"}


def _pt(cx: float, cy: float, r: float, ang_deg: float) -> Tuple[float, float]:
    a = math.radians(float(ang_deg))
    return cx + r * math.sin(a), cy - r * math.cos(a)


def _plane_svg(cx: float, cy: float, r: float, pl: Dict[str, Any],
               vmax: float) -> str:
    color = _COLORS.get(pl.get("color", "cyan"), _COLORS["cyan"])
    fill = _FILLS.get(pl.get("color", "cyan"), "#eef2f7")
    s: List[str] = []
    # cara del plano
    s.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="{fill}" '
             f'stroke="{color}" stroke-width="2.6"/>')
    # ejes
    s.append(f'<line x1="{cx:.1f}" y1="{cy-r:.1f}" x2="{cx:.1f}" y2="{cy+r:.1f}" '
             f'stroke="{_GRID}" stroke-width="1"/>')
    s.append(f'<line x1="{cx-r:.1f}" y1="{cy:.1f}" x2="{cx+r:.1f}" y2="{cy:.1f}" '
             f'stroke="{_GRID}" stroke-width="1"/>')
    s.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="3.2" fill="{_NAVY}"/>')
    # marcas de ángulo
    for a, lbl in [(0, "0°"), (90, "90°"), (180, "180°"), (270, "270°")]:
        tx, ty = _pt(cx, cy, r * 1.16, a)
        s.append(f'<text x="{tx:.1f}" y="{ty+3:.1f}" text-anchor="middle" '
                 f'font-size="11" fill="{_MUTE}">{lbl}</text>')

    # vector de vibración
    vib = pl.get("vib")
    if vib and vib[0] and float(vib[0]) > 0 and vmax > 0:
        vmag, vang = float(vib[0]), float(vib[1])
        L = r * 0.82 * (vmag / vmax)
        vx, vy = _pt(cx, cy, L, vang)
        s.append(f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{vx:.1f}" y2="{vy:.1f}" '
                 f'stroke="{_VIB}" stroke-width="3" marker-end="url(#vibarrow)"/>')
        anchor = "start" if vx >= cx else "end"
        dx = 8 if vx >= cx else -8
        s.append(f'<text x="{vx+dx:.1f}" y="{vy-6:.1f}" text-anchor="{anchor}" '
                 f'font-size="12" font-weight="700" fill="{_VIB}">'
                 f'{vmag:.2f} {pl.get("vib_unit","")} ∠{vang:.0f}°</text>')

    # contrapeso
    wa = pl.get("weight_ang")
    if wa is not None:
        wx, wy = _pt(cx, cy, r, float(wa))
        d = 9
        s.append(f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{wx:.1f}" y2="{wy:.1f}" '
                 f'stroke="{color}" stroke-width="3"/>')
        s.append(f'<polygon points="{wx:.1f},{wy-d} {wx+d:.1f},{wy:.1f} '
                 f'{wx:.1f},{wy+d} {wx-d:.1f},{wy:.1f}" fill="{color}" '
                 f'stroke="white" stroke-width="1.4"/>')
        anchor = "start" if wx >= cx else "end"
        dx = 12 if wx >= cx else -12
        s.append(f'<text x="{wx+dx:.1f}" y="{wy+4:.1f}" text-anchor="{anchor}" '
                 f'font-size="13" font-weight="800" fill="{_NAVY}">'
                 f'{pl.get("weight_label","")}</text>')

    # nombre del plano
    s.append(f'<text x="{cx:.1f}" y="{cy+r+34:.1f}" text-anchor="middle" '
             f'font-size="13" font-weight="800" fill="{_NAVY}">'
             f'{pl.get("name","Plano")}</text>')
    return "".join(s)


def rotor_face_svg(planes: List[Dict[str, Any]], height: int = 300) -> str:
    """SVG de la(s) cara(s) del rotor con vibración + contrapeso."""
    n = max(1, len(planes))
    cell = 360
    width = cell * n
    cy = height * 0.44
    r = min(cell * 0.5, height * 0.5) * 0.62
    vmax = max([float(p["vib"][0]) for p in planes
                if p.get("vib") and p["vib"][0] and float(p["vib"][0]) > 0],
               default=0.0)

    body = "".join(
        _plane_svg(cell * (i + 0.5), cy, r, pl, vmax)
        for i, pl in enumerate(planes))

    return (
        f'<div style="max-width:{width}px;margin:4px auto 0 auto;">'
        f'<svg viewBox="0 0 {width} {height}" width="100%" '
        f'style="max-height:{height}px" xmlns="http://www.w3.org/2000/svg" '
        f'font-family="Inter,system-ui,sans-serif">'
        f'<defs><marker id="vibarrow" markerWidth="9" markerHeight="9" '
        f'refX="6" refY="3" orient="auto" markerUnits="strokeWidth">'
        f'<path d="M0,0 L6,3 L0,6 Z" fill="{_VIB}"/></marker></defs>'
        f'{body}</svg></div>'
    )


def build_planes_1p(vib: Optional[Tuple[float, float]], vib_unit: str,
                    weight_ang: Optional[float], weight_label: str) -> List[Dict[str, Any]]:
    return [{"name": "Plano de corrección", "color": "cyan", "vib": vib,
             "vib_unit": vib_unit, "weight_ang": weight_ang,
             "weight_label": weight_label}]


def build_planes_2p(vibA, vibB, vib_unit, waA, wlA, waB, wlB) -> List[Dict[str, Any]]:
    return [
        {"name": "Plano A", "color": "cyan", "vib": vibA, "vib_unit": vib_unit,
         "weight_ang": waA, "weight_label": wlA},
        {"name": "Plano B", "color": "green", "vib": vibB, "vib_unit": vib_unit,
         "weight_ang": waB, "weight_label": wlB},
    ]


__all__ = ["rotor_face_svg", "build_planes_1p", "build_planes_2p"]
