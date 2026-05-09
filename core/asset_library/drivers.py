"""
core.asset_library.drivers
==========================

Iconografía 2D de máquinas motrices (drivers): motor eléctrico,
turbinas, motores reciprocantes.

Cada función devuelve (svg_str, anchors) donde anchors es un dict con
las coordenadas (x, y) de los puntos relevantes para colocar sensor
dots automáticamente. Las coordenadas son ABSOLUTAS dentro del
viewBox del icono (incluyen el x_offset/y_offset si se pasaron).

Estándar de tamaño: cada driver ocupa 320 x 200 px de viewBox.
El shaft (eje) sale por el lado derecho a y = y_offset + 100.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from core.asset_library.primitives import (
    COLORS, shaft_line, bearing_circle, machine_body, cooling_fins,
    cylinder_vertical, label_top, side_label, crankshaft_box,
)


# ============================================================
# Motor eléctrico (sleeve bearings)
# ============================================================

def electric_motor_sleeve(
    label: str = "Motor AC (sleeve)",
    x_offset: float = 0,
    y_offset: float = 0,
) -> Tuple[str, Dict[str, Any]]:
    """Motor eléctrico horizontal con cojinetes planos hidrodinámicos."""
    W, H = 320, 200
    body_x = x_offset + 30
    body_y = y_offset + 50
    body_w = 240
    body_h = 100
    cy = y_offset + 100  # eje
    parts = [
        machine_body(body_x, body_y, body_w, body_h, "driver"),
        cooling_fins(body_x + 30, body_y + 6, body_w - 60, body_h - 12, n_fins=8, role="driver"),
        # Caja de bornes arriba
        f'<rect x="{x_offset + 130:.1f}" y="{y_offset + 26:.1f}" width="60" height="22" '
        f'rx="3" fill="{COLORS["driver_accent"]}" stroke="{COLORS["driver_stroke"]}" stroke-width="2"/>',
        f'<text x="{x_offset + 160:.1f}" y="{y_offset + 41:.1f}" text-anchor="middle" '
        f'font-size="8" font-weight="700" fill="{COLORS["driver_stroke"]}">TERMINAL</text>',
        # Eje saliendo por la derecha
        shaft_line(x_offset + body_x + body_w - 6, cy, x_offset + W, cy),
        # Bearings
        bearing_circle(body_x + 14, cy, r=14, label="NDE", color="driver"),
        bearing_circle(body_x + body_w - 14, cy, r=14, label="DE", color="driver"),
        # Label
        label_top(x_offset + W / 2, y_offset + 24, label, "driver"),
        # Tag categoría
        f'<text x="{x_offset + W / 2:.1f}" y="{y_offset + H - 8:.1f}" text-anchor="middle" '
        f'font-size="9" font-weight="500" fill="{COLORS["text_muted"]}" '
        f'font-family="-apple-system, sans-serif">Sleeve bearings · API 670</text>',
    ]
    anchors = {
        "DE":  (body_x + body_w - 14, cy),
        "NDE": (body_x + 14, cy),
        "shaft_out": (x_offset + W, cy),
        "viewbox_w": W,
        "viewbox_h": H,
    }
    return "".join(parts), anchors


# ============================================================
# Motor eléctrico (rolling element)
# ============================================================

def electric_motor_rolling(
    label: str = "Motor AC (rolling)",
    x_offset: float = 0,
    y_offset: float = 0,
) -> Tuple[str, Dict[str, Any]]:
    """Motor eléctrico con rodamientos (más común para <500 HP)."""
    W, H = 320, 200
    body_x = x_offset + 30
    body_y = y_offset + 50
    body_w = 240
    body_h = 100
    cy = y_offset + 100
    parts = [
        machine_body(body_x, body_y, body_w, body_h, "driver"),
        cooling_fins(body_x + 30, body_y + 6, body_w - 60, body_h - 12, n_fins=10, role="driver"),
        # Ventilador NDE (más obvio en motores rolling)
        f'<circle cx="{body_x + 8:.1f}" cy="{cy:.1f}" r="22" '
        f'fill="none" stroke="{COLORS["driver_stroke"]}" stroke-width="1.5" stroke-dasharray="4,2"/>',
        f'<text x="{body_x + 8:.1f}" y="{cy + 38:.1f}" text-anchor="middle" '
        f'font-size="8" fill="{COLORS["text_muted"]}" font-family="-apple-system, sans-serif">FAN</text>',
        # Eje
        shaft_line(x_offset + body_x + body_w - 6, cy, x_offset + W, cy),
        bearing_circle(body_x + 14, cy, r=12, label="NDE", color="driver"),
        bearing_circle(body_x + body_w - 14, cy, r=12, label="DE", color="driver"),
        label_top(x_offset + W / 2, y_offset + 24, label, "driver"),
        f'<text x="{x_offset + W / 2:.1f}" y="{y_offset + H - 8:.1f}" text-anchor="middle" '
        f'font-size="9" fill="{COLORS["text_muted"]}" font-family="-apple-system, sans-serif">'
        f'Rolling element bearings · ISO 20816-3</text>',
    ]
    anchors = {
        "DE": (body_x + body_w - 14, cy),
        "NDE": (body_x + 14, cy),
        "shaft_out": (x_offset + W, cy),
        "viewbox_w": W,
        "viewbox_h": H,
    }
    return "".join(parts), anchors


# ============================================================
# Gas turbine aero (LM6000-style: split frames CRF + TRF)
# ============================================================

def gas_turbine_aero(
    label: str = "GE LM6000",
    x_offset: float = 0,
    y_offset: float = 0,
) -> Tuple[str, Dict[str, Any]]:
    """
    Turbina aero-derivativa con compresor (compressor section, izquierda)
    + combustor + turbina (turbine section, derecha). Bearing locations:
    CRF (Compressor Rear Frame) y TRF (Turbine Rear Frame).
    """
    W, H = 380, 200
    body_y = y_offset + 50
    body_h = 100
    cy = y_offset + 100
    fill = COLORS["driver_fill"]
    accent = COLORS["driver_accent"]
    stroke = COLORS["driver_stroke"]

    # Compresor (izquierda) — cuerpo cónico desde fan hasta combustor
    fan_x = x_offset + 25
    comp_end = x_offset + 165
    # Combustor central
    comb_start = x_offset + 175
    comb_end = x_offset + 200
    # Turbina (derecha)
    turb_end = x_offset + 350

    parts = [
        # Compresor — trapecio (de pequeño a grande)
        f'<polygon points="'
        f'{fan_x:.1f},{cy - 30:.1f} '
        f'{comp_end:.1f},{cy - 50:.1f} '
        f'{comp_end:.1f},{cy + 50:.1f} '
        f'{fan_x:.1f},{cy + 30:.1f}'
        f'" fill="{accent}" stroke="{stroke}" stroke-width="2.5"/>',

        # Aletas del compresor (vertical lines)
        *[
            f'<line x1="{fan_x + 15 + i * 20:.1f}" y1="{cy - 28 + i * 1:.1f}" '
            f'x2="{fan_x + 15 + i * 20:.1f}" y2="{cy + 28 - i * 1:.1f}" '
            f'stroke="{stroke}" stroke-width="0.8" stroke-opacity="0.5"/>'
            for i in range(7)
        ],

        # Combustor (cylinder marrón/ámbar)
        f'<rect x="{comb_start:.1f}" y="{cy - 55:.1f}" width="{comb_end - comb_start:.1f}" '
        f'height="110" fill="{COLORS["coupling_fill"]}" stroke="{COLORS["coupling_stroke"]}" '
        f'stroke-width="2.5" rx="4"/>',
        f'<text x="{(comb_start + comb_end) / 2:.1f}" y="{cy + 70:.1f}" text-anchor="middle" '
        f'font-size="8" font-weight="600" fill="{COLORS["coupling_stroke"]}">COMB</text>',

        # Turbina — trapecio (de grande a chico)
        f'<polygon points="'
        f'{comb_end:.1f},{cy - 50:.1f} '
        f'{turb_end:.1f},{cy - 30:.1f} '
        f'{turb_end:.1f},{cy + 30:.1f} '
        f'{comb_end:.1f},{cy + 50:.1f}'
        f'" fill="{fill}" stroke="{stroke}" stroke-width="2.5"/>',
        # Aletas de la turbina
        *[
            f'<line x1="{comb_end + 10 + i * 20:.1f}" y1="{cy - 48 + i * 2:.1f}" '
            f'x2="{comb_end + 10 + i * 20:.1f}" y2="{cy + 48 - i * 2:.1f}" '
            f'stroke="{stroke}" stroke-width="0.8" stroke-opacity="0.5"/>'
            for i in range(7)
        ],

        # Eje sale por la derecha
        shaft_line(turb_end - 4, cy, x_offset + W, cy),

        # Bearings: CRF (compressor rear frame, lado combustor del compresor)
        bearing_circle(comp_end - 14, cy, r=12, label="CRF", color="driver"),
        # TRF (turbine rear frame, lado salida de la turbina)
        bearing_circle(turb_end - 14, cy, r=12, label="TRF", color="driver"),

        # Label
        label_top(x_offset + W / 2, y_offset + 24, label, "driver"),
        f'<text x="{x_offset + W / 2:.1f}" y="{y_offset + H - 8:.1f}" text-anchor="middle" '
        f'font-size="9" fill="{COLORS["text_muted"]}" font-family="-apple-system, sans-serif">'
        f'Aero gas turbine · 2-shaft · OEM thresholds</text>',
    ]
    anchors = {
        "DE": (turb_end - 14, cy),       # TRF (lado output)
        "NDE": (comp_end - 14, cy),       # CRF (lado intake)
        "TRF": (turb_end - 14, cy),
        "CRF": (comp_end - 14, cy),
        "shaft_out": (x_offset + W, cy),
        "viewbox_w": W,
        "viewbox_h": H,
    }
    return "".join(parts), anchors


# ============================================================
# Gas turbine industrial (SGT-300 / Frame 9 — single shaft, casing único)
# ============================================================

def gas_turbine_industrial(
    label: str = "Siemens SGT-300",
    x_offset: float = 0,
    y_offset: float = 0,
) -> Tuple[str, Dict[str, Any]]:
    """Turbina industrial heavy-duty single-shaft (Siemens SGT, GE Frame)."""
    W, H = 360, 200
    body_x = x_offset + 30
    body_y = y_offset + 45
    body_w = 270
    body_h = 110
    cy = y_offset + 100
    stroke = COLORS["driver_stroke"]
    fill = COLORS["driver_fill"]
    accent = COLORS["driver_accent"]

    parts = [
        # Cuerpo principal
        f'<rect x="{body_x:.1f}" y="{body_y:.1f}" width="{body_w:.1f}" height="{body_h:.1f}" '
        f'rx="14" fill="{fill}" stroke="{stroke}" stroke-width="2.5"/>',
        # Sección compresor (izquierda) más alta
        f'<rect x="{body_x + 14:.1f}" y="{body_y + 8:.1f}" width="{body_w * 0.4:.1f}" height="{body_h - 16:.1f}" '
        f'fill="{accent}"/>',
        # Combustor exterior (cilindro lateral arriba, típico Siemens)
        f'<rect x="{body_x + 60:.1f}" y="{body_y - 20:.1f}" width="{body_w * 0.4:.1f}" height="22" '
        f'rx="6" fill="{COLORS["coupling_fill"]}" stroke="{COLORS["coupling_stroke"]}" stroke-width="2"/>',
        f'<text x="{body_x + 60 + body_w * 0.2:.1f}" y="{body_y - 5:.1f}" text-anchor="middle" '
        f'font-size="8" font-weight="600" fill="{COLORS["coupling_stroke"]}">COMBUSTOR</text>',
        # Línea conexión combustor → casing
        f'<line x1="{body_x + 80:.1f}" y1="{body_y - 4:.1f}" x2="{body_x + 80:.1f}" y2="{body_y + 4:.1f}" '
        f'stroke="{COLORS["coupling_stroke"]}" stroke-width="1.5"/>',
        # Aletas de la turbina (lado derecho)
        *[
            f'<line x1="{body_x + body_w * 0.5 + i * 18:.1f}" y1="{body_y + 12:.1f}" '
            f'x2="{body_x + body_w * 0.5 + i * 18:.1f}" y2="{body_y + body_h - 12:.1f}" '
            f'stroke="{stroke}" stroke-width="0.8" stroke-opacity="0.5"/>'
            for i in range(6)
        ],
        # Eje
        shaft_line(body_x + body_w - 6, cy, x_offset + W, cy),
        # Bearings DE/NDE
        bearing_circle(body_x + 14, cy, r=13, label="NDE", color="driver"),
        bearing_circle(body_x + body_w - 14, cy, r=13, label="DE", color="driver"),
        # Label
        label_top(x_offset + W / 2, y_offset + 24, label, "driver"),
        f'<text x="{x_offset + W / 2:.1f}" y="{y_offset + H - 8:.1f}" text-anchor="middle" '
        f'font-size="9" fill="{COLORS["text_muted"]}" font-family="-apple-system, sans-serif">'
        f'Industrial gas turbine · single shaft</text>',
    ]
    anchors = {
        "DE": (body_x + body_w - 14, cy),
        "NDE": (body_x + 14, cy),
        "shaft_out": (x_offset + W, cy),
        "viewbox_w": W,
        "viewbox_h": H,
    }
    return "".join(parts), anchors


# ============================================================
# Steam turbine (Siemens SST / GE BB-series / Mitsubishi)
# ============================================================

def steam_turbine(
    label: str = "Steam Turbine",
    x_offset: float = 0,
    y_offset: float = 0,
) -> Tuple[str, Dict[str, Any]]:
    """Turbina a vapor — casing horizontal con válvula de admisión arriba y exhaust."""
    W, H = 360, 200
    body_x = x_offset + 30
    body_y = y_offset + 55
    body_w = 270
    body_h = 90
    cy = y_offset + 100
    stroke = COLORS["driver_stroke"]
    fill = COLORS["driver_fill"]
    accent = COLORS["driver_accent"]

    parts = [
        # Casing principal (horizontal, slightly conical)
        f'<polygon points="'
        f'{body_x:.1f},{body_y + 20:.1f} '
        f'{body_x + body_w * 0.35:.1f},{body_y:.1f} '
        f'{body_x + body_w:.1f},{body_y - 5:.1f} '
        f'{body_x + body_w:.1f},{body_y + body_h + 5:.1f} '
        f'{body_x + body_w * 0.35:.1f},{body_y + body_h:.1f} '
        f'{body_x:.1f},{body_y + body_h - 20:.1f}'
        f'" fill="{fill}" stroke="{stroke}" stroke-width="2.5"/>',
        # Admisión de vapor (arriba, izquierda) con flecha
        f'<rect x="{body_x + 30:.1f}" y="{body_y - 32:.1f}" width="36" height="32" '
        f'fill="{accent}" stroke="{stroke}" stroke-width="2" rx="4"/>',
        f'<text x="{body_x + 48:.1f}" y="{body_y - 17:.1f}" text-anchor="middle" '
        f'font-size="8" font-weight="700" fill="{stroke}">ADMISIÓN</text>',
        f'<text x="{body_x + 48:.1f}" y="{body_y - 7:.1f}" text-anchor="middle" '
        f'font-size="7" font-weight="500" fill="{stroke}">vapor →</text>',
        # Aletas (etapas de la turbina)
        *[
            f'<line x1="{body_x + 70 + i * 22:.1f}" y1="{body_y + 5:.1f}" '
            f'x2="{body_x + 70 + i * 22:.1f}" y2="{body_y + body_h - 5:.1f}" '
            f'stroke="{stroke}" stroke-width="1" stroke-opacity="0.55"/>'
            for i in range(8)
        ],
        # Exhaust (abajo derecha)
        f'<rect x="{body_x + body_w - 90:.1f}" y="{body_y + body_h:.1f}" width="60" height="20" '
        f'fill="{accent}" stroke="{stroke}" stroke-width="2" rx="3"/>',
        f'<text x="{body_x + body_w - 60:.1f}" y="{body_y + body_h + 14:.1f}" text-anchor="middle" '
        f'font-size="8" font-weight="600" fill="{stroke}">EXHAUST ↓</text>',
        # Eje
        shaft_line(body_x + body_w - 6, cy, x_offset + W, cy),
        # Bearings
        bearing_circle(body_x + 14, cy, r=13, label="NDE", color="driver"),
        bearing_circle(body_x + body_w - 14, cy, r=13, label="DE", color="driver"),
        label_top(x_offset + W / 2, y_offset + 24, label, "driver"),
        f'<text x="{x_offset + W / 2:.1f}" y="{y_offset + H - 8:.1f}" text-anchor="middle" '
        f'font-size="9" fill="{COLORS["text_muted"]}" font-family="-apple-system, sans-serif">'
        f'Steam turbine · multi-stage</text>',
    ]
    anchors = {
        "DE": (body_x + body_w - 14, cy),
        "NDE": (body_x + 14, cy),
        "shaft_out": (x_offset + W, cy),
        "viewbox_w": W,
        "viewbox_h": H,
    }
    return "".join(parts), anchors


# ============================================================
# Recip engine 8-cyl in-line (Caterpillar G3500, Waukesha 8L)
# ============================================================

def recip_engine_8cyl_inline(
    label: str = "Recip Engine 8-cyl",
    x_offset: float = 0,
    y_offset: float = 0,
) -> Tuple[str, Dict[str, Any]]:
    """Motor reciprocante 8 cilindros en línea (gas natural)."""
    W, H = 420, 200
    base_y = y_offset + 110
    crank_h = 50
    crank_y = base_y
    cy = base_y + crank_h / 2
    stroke = COLORS["driver_stroke"]
    body_x = x_offset + 25
    body_w = 360
    n_cyl = 8

    parts: list = [
        # Crankcase (caja del cigüeñal)
        crankshaft_box(body_x, crank_y, body_w, crank_h, "driver"),
    ]
    # Cilindros verticales (arriba del crankcase)
    cyl_spacing = (body_w - 30) / n_cyl
    cyl_bore = cyl_spacing * 0.62
    for i in range(n_cyl):
        cx = body_x + 15 + cyl_spacing * (i + 0.5)
        parts.append(
            cylinder_vertical(
                cx, crank_y, length=58, bore=cyl_bore,
                role="driver",
                label=f"C{i + 1}",
                direction="up",
            )
        )
    # Eje del cigüeñal sale por la derecha
    parts.append(shaft_line(body_x + body_w - 4, cy, x_offset + W, cy))
    # Bearings principales (DE y NDE) — lado del flywheel y lado opposite
    parts.append(bearing_circle(body_x + 12, cy, r=12, label="NDE", color="driver"))
    parts.append(bearing_circle(body_x + body_w - 12, cy, r=12, label="DE", color="driver"))
    parts.append(label_top(x_offset + W / 2, y_offset + 12, label, "driver"))
    parts.append(
        f'<text x="{x_offset + W / 2:.1f}" y="{y_offset + H - 8:.1f}" text-anchor="middle" '
        f'font-size="9" fill="{COLORS["text_muted"]}" font-family="-apple-system, sans-serif">'
        f'Recip engine · 8 cylinders in-line</text>'
    )
    anchors = {
        "DE": (body_x + body_w - 12, cy),
        "NDE": (body_x + 12, cy),
        "shaft_out": (x_offset + W, cy),
        "viewbox_w": W,
        "viewbox_h": H,
    }
    return "".join(parts), anchors


# ============================================================
# Recip engine 16-cyl in-line (Cooper-Bessemer, Waukesha 16V)
# ============================================================

def recip_engine_16cyl_inline(
    label: str = "Recip Engine 16-cyl",
    x_offset: float = 0,
    y_offset: float = 0,
) -> Tuple[str, Dict[str, Any]]:
    """Motor reciprocante 16 cilindros en línea (caso integral Cooper-Bessemer)."""
    W, H = 580, 200
    crank_h = 48
    crank_y = y_offset + 110
    cy = crank_y + crank_h / 2
    body_x = x_offset + 25
    body_w = 520
    n_cyl = 16

    parts: list = [
        crankshaft_box(body_x, crank_y, body_w, crank_h, "driver"),
    ]
    cyl_spacing = (body_w - 30) / n_cyl
    cyl_bore = cyl_spacing * 0.7
    for i in range(n_cyl):
        cx = body_x + 15 + cyl_spacing * (i + 0.5)
        parts.append(
            cylinder_vertical(
                cx, crank_y, length=56, bore=cyl_bore,
                role="driver",
                label=f"{i + 1}",
                direction="up",
            )
        )
    parts.append(shaft_line(body_x + body_w - 4, cy, x_offset + W, cy))
    parts.append(bearing_circle(body_x + 12, cy, r=12, label="NDE", color="driver"))
    parts.append(bearing_circle(body_x + body_w - 12, cy, r=12, label="DE", color="driver"))
    parts.append(label_top(x_offset + W / 2, y_offset + 12, label, "driver"))
    parts.append(
        f'<text x="{x_offset + W / 2:.1f}" y="{y_offset + H - 8:.1f}" text-anchor="middle" '
        f'font-size="9" fill="{COLORS["text_muted"]}" font-family="-apple-system, sans-serif">'
        f'Recip engine · 16 cylinders in-line</text>'
    )
    anchors = {
        "DE": (body_x + body_w - 12, cy),
        "NDE": (body_x + 12, cy),
        "shaft_out": (x_offset + W, cy),
        "viewbox_w": W,
        "viewbox_h": H,
    }
    return "".join(parts), anchors


__all__ = [
    "electric_motor_sleeve",
    "electric_motor_rolling",
    "gas_turbine_aero",
    "gas_turbine_industrial",
    "steam_turbine",
    "recip_engine_8cyl_inline",
    "recip_engine_16cyl_inline",
]
