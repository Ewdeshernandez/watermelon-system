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
    ]
    anchors = {
        "shaft_in": (x_offset, cy),
        "shaft_out": (x_offset + W, cy),
        "viewbox_w": W,
        "viewbox_h": H,
    }
    return "".join(parts), anchors


def coupling_rigid_quill(
    x_offset: float = 0,
    y_offset: float = 0,
    width: float = 70,
    height: float = 200,
    label: str = "Acople rígido quill",
) -> Tuple[str, Dict[str, Any]]:
    """
    Acople rígido tipo quill-shaft (LM6000-style). Eje fino atravesando
    entre dos hubs cortos a cada extremo, sin sleeve grueso visible.
    Geometría correcta para tren aero-derivative: el shaft del power
    turbine output entra al hub del generador con un eje delgado.

    Visualmente: 2 hub-flanges chicos a izq/der + eje fino atravesando
    largo + anillo central de marriage flange opcional.
    """
    W = width
    H = height
    cy = y_offset + H / 2
    stroke = COLORS["coupling_stroke"]
    fill = COLORS["coupling_fill"]

    # Hub izquierdo (lado driver / power turbine output)
    hub_l_x = x_offset + 4
    hub_l_w = 14
    # Hub derecho (lado driven / generator input)
    hub_r_x = x_offset + W - 18
    hub_r_w = 14
    # Marriage flange central (opcional, da carácter de quill)
    mid_x = x_offset + W / 2

    parts = [
        # Hub izquierdo (flange acoplada al driver)
        f'<rect x="{hub_l_x:.1f}" y="{cy - 18:.1f}" width="{hub_l_w:.1f}" height="36" '
        f'rx="2" fill="{stroke}"/>',
        # Hub derecho (flange acoplada al driven)
        f'<rect x="{hub_r_x:.1f}" y="{cy - 18:.1f}" width="{hub_r_w:.1f}" height="36" '
        f'rx="2" fill="{stroke}"/>',
        # Marriage flange central — anillo delgado, denota junta rígida
        f'<circle cx="{mid_x:.1f}" cy="{cy:.1f}" r="9" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="1.5"/>',
        f'<circle cx="{mid_x:.1f}" cy="{cy:.1f}" r="3.5" fill="{stroke}"/>',
        # Eje QUILL — fino, atravesando todo el largo del coupling
        # Stroke más grueso que un eje normal pero más fino que un sleeve
        f'<line x1="{x_offset:.1f}" y1="{cy:.1f}" x2="{x_offset + W:.1f}" y2="{cy:.1f}" '
        f'stroke="{COLORS["shaft"]}" stroke-width="6" stroke-linecap="round"/>',
        # Highlight stripe en eje (cilíndrico feel)
        f'<line x1="{x_offset + 2:.1f}" y1="{cy - 1.5:.1f}" x2="{x_offset + W - 2:.1f}" y2="{cy - 1.5:.1f}" '
        f'stroke="white" stroke-width="0.8" stroke-opacity="0.4" stroke-linecap="round"/>',
        # Pequeños bolts radiales en el flange central (4 puntos)
        f'<circle cx="{mid_x - 6:.1f}" cy="{cy - 6:.1f}" r="1.2" fill="{stroke}"/>',
        f'<circle cx="{mid_x + 6:.1f}" cy="{cy - 6:.1f}" r="1.2" fill="{stroke}"/>',
        f'<circle cx="{mid_x - 6:.1f}" cy="{cy + 6:.1f}" r="1.2" fill="{stroke}"/>',
        f'<circle cx="{mid_x + 6:.1f}" cy="{cy + 6:.1f}" r="1.2" fill="{stroke}"/>',
    ]
    anchors = {
        "shaft_in": (x_offset, cy),
        "shaft_out": (x_offset + W, cy),
        "viewbox_w": W,
        "viewbox_h": H,
    }
    return "".join(parts), anchors


__all__ = ["coupling_flexible", "coupling_rigid", "coupling_rigid_quill"]
