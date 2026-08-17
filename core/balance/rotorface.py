"""
core/balance/rotorface.py — Vista polar 2D (SVG) del plano de balanceo
======================================================================

"Cara del rotor" en SVG puro (sin Plotly/WebGL) que muestra por plano:

  - Vector de VIBRACIÓN medido (flecha roja fina) — al cargar el dato.
  - CONTRAPESO de corrección (diamante del color del plano) — al calcular.
  - Rejilla polar 0/90/180/270° con la escala orientada **contra el sentido de
    giro** de la máquina (convención de campo del balanceo).
  - Flecha del sentido de giro (CW/CCW) y nota de convención.

Importante: la escala se orienta contra el giro solo para el DIBUJO (para que
el técnico ubique el peso mirando el eje). El cálculo es convención-agnóstico
(el coeficiente de influencia se mide de la corrida de prueba), así que los
números no cambian con el sentido de giro.

Se renderiza con st.markdown(svg, unsafe_allow_html=True).
"""
from __future__ import annotations

import math
import secrets
from typing import Any, Dict, List, Optional, Tuple

_NAVY = "#0F1E3D"
_MUTE = "#94a3b8"
_GRID = "#e2e8f0"
_VIB = "#ef4444"
_ROT = "#64748b"
_COLORS = {"cyan": "#1AAEE5", "green": "#16a34a", "amber": "#D89B22"}
_FILLS = {"cyan": "#eaf7fd", "green": "#eafaf0", "amber": "#fdf6e7"}


def _pt(cx: float, cy: float, r: float, ang_deg: float, s: int) -> Tuple[float, float]:
    """Punto en la cara. s = +1 (escala horaria) o -1 (espejo = antihoraria)."""
    a = math.radians(float(ang_deg))
    return cx + s * r * math.sin(a), cy - r * math.cos(a)


def _rotation_arrow(cx: float, cy: float, r: float, rotation: str,
                    marker_id: str) -> str:
    """Arco con flecha que indica el sentido de giro FÍSICO (no espejado)."""
    rr = r * 0.42
    xl, yl = cx + rr * math.sin(math.radians(-52)), cy - rr * math.cos(math.radians(-52))
    xr, yr = cx + rr * math.sin(math.radians(52)), cy - rr * math.cos(math.radians(52))
    if rotation == "CW":
        path = f'M {xl:.1f},{yl:.1f} A {rr:.1f},{rr:.1f} 0 0 1 {xr:.1f},{yr:.1f}'
    else:
        path = f'M {xr:.1f},{yr:.1f} A {rr:.1f},{rr:.1f} 0 0 0 {xl:.1f},{yl:.1f}'
    return (f'<path d="{path}" fill="none" stroke="{_ROT}" stroke-width="2" '
            f'marker-end="url(#{marker_id})"/>'
            f'<text x="{cx:.1f}" y="{cy - rr - 6:.1f}" text-anchor="middle" '
            f'font-size="10" fill="{_ROT}">giro {rotation}</text>')


def _plane_svg(cx: float, cy: float, r: float, pl: Dict[str, Any], vmax: float,
               arrow_id: str, rot_marker_id: str, s: int, rotation: str) -> str:
    color = _COLORS.get(pl.get("color", "cyan"), _COLORS["cyan"])
    fill = _FILLS.get(pl.get("color", "cyan"), "#eef2f7")
    out: List[str] = []
    # cara del plano
    out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="{fill}" '
               f'stroke="{color}" stroke-width="2.6"/>')
    out.append(f'<line x1="{cx:.1f}" y1="{cy-r:.1f}" x2="{cx:.1f}" y2="{cy+r:.1f}" '
               f'stroke="{_GRID}" stroke-width="1"/>')
    out.append(f'<line x1="{cx-r:.1f}" y1="{cy:.1f}" x2="{cx+r:.1f}" y2="{cy:.1f}" '
               f'stroke="{_GRID}" stroke-width="1"/>')
    # sentido de giro (físico)
    out.append(_rotation_arrow(cx, cy, r, rotation, rot_marker_id))
    out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="3.2" fill="{_NAVY}"/>')
    # marcas de ángulo (escala CONTRA el giro)
    for a, lbl in [(0, "0°"), (90, "90°"), (180, "180°"), (270, "270°")]:
        tx, ty = _pt(cx, cy, r * 1.16, a, s)
        out.append(f'<text x="{tx:.1f}" y="{ty+3:.1f}" text-anchor="middle" '
                   f'font-size="11" fill="{_MUTE}">{lbl}</text>')

    # vector de vibración
    vib = pl.get("vib")
    if vib and vib[0] and float(vib[0]) > 0 and vmax > 0:
        vmag, vang = float(vib[0]), float(vib[1])
        L = r * 0.82 * (vmag / vmax)
        vx, vy = _pt(cx, cy, L, vang, s)
        out.append(f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{vx:.1f}" y2="{vy:.1f}" '
                   f'stroke="{_VIB}" stroke-width="2" stroke-linecap="round" '
                   f'marker-end="url(#{arrow_id})"/>')
        anchor = "start" if vx >= cx else "end"
        dx = 8 if vx >= cx else -8
        out.append(f'<text x="{vx+dx:.1f}" y="{vy-6:.1f}" text-anchor="{anchor}" '
                   f'font-size="12" font-weight="700" fill="{_VIB}">'
                   f'{vmag:.2f} {pl.get("vib_unit","")} ∠{vang:.0f}°</text>')

    # contrapeso
    wa = pl.get("weight_ang")
    if wa is not None:
        wx, wy = _pt(cx, cy, r, float(wa), s)
        d = 9
        out.append(f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{wx:.1f}" y2="{wy:.1f}" '
                   f'stroke="{color}" stroke-width="3"/>')
        out.append(f'<polygon points="{wx:.1f},{wy-d} {wx+d:.1f},{wy:.1f} '
                   f'{wx:.1f},{wy+d} {wx-d:.1f},{wy:.1f}" fill="{color}" '
                   f'stroke="white" stroke-width="1.4"/>')
        anchor = "start" if wx >= cx else "end"
        dx = 12 if wx >= cx else -12
        out.append(f'<text x="{wx+dx:.1f}" y="{wy+4:.1f}" text-anchor="{anchor}" '
                   f'font-size="13" font-weight="800" fill="{_NAVY}">'
                   f'{pl.get("weight_label","")}</text>')

    # nombre del plano
    out.append(f'<text x="{cx:.1f}" y="{cy+r+34:.1f}" text-anchor="middle" '
               f'font-size="13" font-weight="800" fill="{_NAVY}">'
               f'{pl.get("name","Plano")}</text>')
    return "".join(out)


def rotor_face_svg(planes: List[Dict[str, Any]], rotation: str = "CCW",
                   height: int = 300) -> str:
    """SVG de la(s) cara(s) del rotor. rotation = 'CW' | 'CCW' orienta la
    escala angular contra el giro (convención de balanceo)."""
    rotation = "CW" if str(rotation).upper() == "CW" else "CCW"
    # Escala CONTRA el giro: máquina CW → números CCW (espejo); CCW → horario.
    s = -1 if rotation == "CW" else 1

    uid = secrets.token_hex(3)
    arrow_id = "vibarrow_" + uid
    rot_id = "rotarrow_" + uid

    n = max(1, len(planes))
    cell = 360
    pad = 72          # margen lateral para que las etiquetas no se recorten
    width = cell * n + 2 * pad
    cy = height * 0.44
    r = min(cell * 0.5, height * 0.5) * 0.62
    vmax = max([float(p["vib"][0]) for p in planes
                if p.get("vib") and p["vib"][0] and float(p["vib"][0]) > 0],
               default=0.0)

    body = "".join(
        _plane_svg(pad + cell * (i + 0.5), cy, r, pl, vmax, arrow_id, rot_id, s, rotation)
        for i, pl in enumerate(planes))

    return (
        f'<div style="max-width:{width}px;margin:4px auto 0 auto;">'
        f'<svg viewBox="0 0 {width} {height + 18}" width="100%" '
        f'style="max-height:{height + 18}px" xmlns="http://www.w3.org/2000/svg" '
        f'font-family="Inter,system-ui,sans-serif">'
        f'<defs>'
        f'<marker id="{arrow_id}" markerWidth="10" markerHeight="8" refX="6.5" '
        f'refY="3" orient="auto" markerUnits="strokeWidth">'
        f'<path d="M0,0 L7,3 L0,6 L1.8,3 Z" fill="{_VIB}"/></marker>'
        f'<marker id="{rot_id}" markerWidth="8" markerHeight="8" refX="4" refY="3" '
        f'orient="auto" markerUnits="strokeWidth">'
        f'<path d="M0,0 L5,3 L0,6 Z" fill="{_ROT}"/></marker>'
        f'</defs>{body}'
        f'<text x="{width/2:.1f}" y="{height + 13:.0f}" text-anchor="middle" '
        f'font-size="10" fill="{_MUTE}">Ángulos medidos contra el sentido de '
        f'giro ({rotation})</text>'
        f'</svg></div>'
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
