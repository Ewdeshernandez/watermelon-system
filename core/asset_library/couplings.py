"""
core.asset_library.couplings
============================

Iconografía 2D de acoples flexibles y rígidos.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from core.asset_library.primitives import COLORS, shaft_line


def coupling_flexible(
    x_offset: float = 0,
    y_offset: float = 0,
    width: float = 80,
    height: float = 200,
    label: str = "Acople flexible",
) -> Tuple[str, Dict[str, Any]]:
    """Acople flexible tipo disc-pack o membrana — 2 flanges + paquete central."""
    W = width
    H = height
    cy = y_offset + H / 2
    stroke = COLORS["coupling_stroke"]
    fill = COLORS["coupling_fill"]

    flange_w = 12
    spacer_x1 = x_offset + 18
    spacer_x2 = x_offset + W - 18

    parts = [
        # Flange izquierda (lado driver)
        f'<rect x="{x_offset + 8:.1f}" y="{cy - 28:.1f}" width="{flange_w:.1f}" height="56" '
        f'rx="2" fill="{stroke}"/>',
        # Flange derecha (lado driven)
        f'<rect x="{x_offset + W - 20:.1f}" y="{cy - 28:.1f}" width="{flange_w:.1f}" height="56" '
        f'rx="2" fill="{stroke}"/>',
        # Spacer / disc-pack central
        f'<rect x="{spacer_x1:.1f}" y="{cy - 18:.1f}" width="{spacer_x2 - spacer_x1:.1f}" height="36" '
        f'rx="6" fill="{fill}" stroke="{stroke}" stroke-width="2"/>',
        # Bulones (representación: 4 puntos)
        *[
            f'<circle cx="{x_offset + 14:.1f}" cy="{cy - 20 + i * 13:.1f}" r="2" fill="{stroke}"/>'
            for i in range(4)
        ],
        *[
            f'<circle cx="{x_offset + W - 14:.1f}" cy="{cy - 20 + i * 13:.1f}" r="2" fill="{stroke}"/>'
            for i in range(4)
        ],
        # Eje atravesando todo el coupling
        shaft_line(x_offset, cy, x_offset + W, cy),
        # Label abajo
        f'<text x="{x_offset + W / 2:.1f}" y="{y_offset + H - 14:.1f}" text-anchor="middle" '
        f'font-size="9" font-weight="600" fill="{stroke}" '
        f'font-family="-apple-system, sans-serif">{label}</text>',
    ]
    anchors = {
        "shaft_in": (x_offset, cy),
        "shaft_out": (x_offset + W, cy),
        "viewbox_w": W,
        "viewbox_h": H,
    }
    return "".join(parts), anchors


def coupling_rigid(
    x_offset: float = 0,
    y_offset: float = 0,
    width: float = 60,
    height: float = 200,
    label: str = "Acople rígido",
) -> Tuple[str, Dict[str, Any]]:
    """Acople rígido (sleeve / hub-fitted)."""
    W = width
    H = height
    cy = y_offset + H / 2
    stroke = COLORS["coupling_stroke"]
    fill = COLORS["coupling_fill"]

    parts = [
        # Cuerpo cilíndrico (sleeve)
        f'<rect x="{x_offset + 8:.1f}" y="{cy - 16:.1f}" width="{W - 16:.1f}" height="32" '
        f'rx="3" fill="{fill}" stroke="{stroke}" stroke-width="2"/>',
        # 2 líneas verticales (representan los grub screws)
        f'<line x1="{x_offset + W / 2 - 8:.1f}" y1="{cy - 14:.1f}" x2="{x_offset + W / 2 - 8:.1f}" y2="{cy + 14:.1f}" '
        f'stroke="{stroke}" stroke-width="1.5"/>',
        f'<line x1="{x_offset + W / 2 + 8:.1f}" y1="{cy - 14:.1f}" x2="{x_offset + W / 2 + 8:.1f}" y2="{cy + 14:.1f}" '
        f'stroke="{stroke}" stroke-width="1.5"/>',
        # Eje
        shaft_line(x_offset, cy, x_offset + W, cy),
        f'<text x="{x_offset + W / 2:.1f}" y="{y_offset + H - 14:.1f}" text-anchor="middle" '
        f'font-size="9" font-weight="600" fill="{stroke}" '
        f'font-family="-apple-system, sans-serif">{label}</text>',
    ]
    anchors = {
        "shaft_in": (x_offset, cy),
        "shaft_out": (x_offset + W, cy),
        "viewbox_w": W,
        "viewbox_h": H,
    }
    return "".join(parts), anchors


__all__ = ["coupling_flexible", "coupling_rigid"]
