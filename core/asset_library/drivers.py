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
    Turbina aero-derivativa estilo Bently System1 (Ciclo 23.20).
    Silueta compacta sin álabes-rayados, con secciones definidas por
    fills sutiles. Bandas azules en bearings (CRF, TRF) y banda roja
    en combustor crítico.

    Layout (de izq a der):
      Bell-mouth intake → Compressor cone → CRF (banda azul) →
      Combustor (banda roja) → Power turbine cone → TRF (banda azul) →
      Outlet cone → Shaft out.
    """
    W, H = 360, 200
    cy = y_offset + 100

    # Paleta neutral System1-style
    body_light = "#f1f5f9"     # gris claro (fondo)
    body_mid = "#cbd5e1"       # gris medio (sombras + cuerpo principal)
    body_outline = "#64748b"   # gris outline
    bearing_band = "#3b82f6"   # azul vivo bearings
    bearing_dark = "#1e40af"   # azul oscuro borde
    combustor_red = "#dc2626"  # rojo combustor
    combustor_dark = "#7f1d1d" # rojo oscuro borde

    # Estaciones a lo largo del eje (todas continuas, sin gaps)
    intake_x1 = x_offset + 10
    intake_x2 = x_offset + 50
    crf_x = x_offset + 155     # banda CRF
    comb_x1 = x_offset + 165
    comb_x2 = x_offset + 200
    trf_x = x_offset + 295     # banda TRF
    outlet_x2 = x_offset + 345

    # Radios verticales en cada estación
    r_intake_in = 16     # bell-mouth angosto
    r_intake_out = 38    # bell-mouth se abre
    r_crf = 50           # compresor max
    r_comb = 58          # combustor más ancho
    r_pt_max = 50        # power turbine entry
    r_trf = 36           # power turbine exit
    r_outlet = 14        # outlet converge a shaft

    # SILUETA CONTINUA — un único path desde intake (izq) hasta outlet (der)
    # Top: intake_in → bell-mouth → compresor expande → CRF → combustor más alto
    #      → PT entry → PT converge → TRF → outlet converge → shaft
    # Bottom: el reverso por simetría
    body_path = (
        f'M {intake_x1:.1f},{cy - r_intake_in:.1f} '
        f'L {intake_x2:.1f},{cy - r_intake_out:.1f} '
        f'L {crf_x:.1f},{cy - r_crf:.1f} '
        f'L {comb_x1:.1f},{cy - r_comb:.1f} '
        f'L {comb_x2:.1f},{cy - r_comb:.1f} '
        f'L {comb_x2 + 10:.1f},{cy - r_pt_max:.1f} '
        f'L {trf_x:.1f},{cy - r_trf:.1f} '
        f'L {trf_x + 10:.1f},{cy - r_trf:.1f} '
        f'L {outlet_x2:.1f},{cy - r_outlet:.1f} '
        f'L {outlet_x2:.1f},{cy + r_outlet:.1f} '
        f'L {trf_x + 10:.1f},{cy + r_trf:.1f} '
        f'L {trf_x:.1f},{cy + r_trf:.1f} '
        f'L {comb_x2 + 10:.1f},{cy + r_pt_max:.1f} '
        f'L {comb_x2:.1f},{cy + r_comb:.1f} '
        f'L {comb_x1:.1f},{cy + r_comb:.1f} '
        f'L {crf_x:.1f},{cy + r_crf:.1f} '
        f'L {intake_x2:.1f},{cy + r_intake_out:.1f} '
        f'L {intake_x1:.1f},{cy + r_intake_in:.1f} '
        f'Z'
    )

    # Path interior del compresor (para fill medio sin cortar el body)
    compressor_overlay = (
        f'M {intake_x2:.1f},{cy - r_intake_out:.1f} '
        f'L {crf_x:.1f},{cy - r_crf:.1f} '
        f'L {crf_x:.1f},{cy + r_crf:.1f} '
        f'L {intake_x2:.1f},{cy + r_intake_out:.1f} Z'
    )

    # Path interior power turbine
    pt_overlay = (
        f'M {comb_x2 + 10:.1f},{cy - r_pt_max:.1f} '
        f'L {trf_x:.1f},{cy - r_trf:.1f} '
        f'L {trf_x:.1f},{cy + r_trf:.1f} '
        f'L {comb_x2 + 10:.1f},{cy + r_pt_max:.1f} Z'
    )

    parts = [
        # Cuerpo único — silueta continua de toda la turbina (gris claro)
        f'<path d="{body_path}" fill="{body_light}" stroke="{body_outline}" '
        f'stroke-width="1.5" stroke-linejoin="round"/>',

        # Overlay compresor — fill medio para diferenciar la zona
        f'<path d="{compressor_overlay}" fill="{body_mid}" stroke="none" opacity="0.85"/>',
        # Marcas de etapas de compresor (sutiles)
        *[
            f'<line x1="{intake_x2 + 8 + i * 14:.1f}" '
            f'y1="{cy - (r_intake_out + (r_crf - r_intake_out) * (8 + i * 14) / (crf_x - intake_x2)):.1f}" '
            f'x2="{intake_x2 + 8 + i * 14:.1f}" '
            f'y2="{cy + (r_intake_out + (r_crf - r_intake_out) * (8 + i * 14) / (crf_x - intake_x2)):.1f}" '
            f'stroke="{body_outline}" stroke-width="0.5" stroke-opacity="0.35"/>'
            for i in range(7)
        ],

        # Overlay power turbine
        f'<path d="{pt_overlay}" fill="{body_mid}" stroke="none" opacity="0.85"/>',
        *[
            f'<line x1="{comb_x2 + 18 + i * 13:.1f}" '
            f'y1="{cy - (r_pt_max - (r_pt_max - r_trf) * (8 + i * 13) / (trf_x - comb_x2 - 10)):.1f}" '
            f'x2="{comb_x2 + 18 + i * 13:.1f}" '
            f'y2="{cy + (r_pt_max - (r_pt_max - r_trf) * (8 + i * 13) / (trf_x - comb_x2 - 10)):.1f}" '
            f'stroke="{body_outline}" stroke-width="0.5" stroke-opacity="0.35"/>'
            for i in range(6)
        ],

        # Combustor — banda roja embebida en la silueta (no sobresale)
        f'<rect x="{comb_x1:.1f}" y="{cy - r_comb + 1:.1f}" '
        f'width="{comb_x2 - comb_x1:.1f}" height="{2 * r_comb - 2:.1f}" '
        f'fill="{combustor_red}" stroke="{combustor_dark}" stroke-width="0.8"/>',
        # Combustor cans (3 líneas verticales sutiles)
        *[
            f'<line x1="{comb_x1 + (comb_x2 - comb_x1) * (0.25 + i * 0.25):.1f}" '
            f'y1="{cy - r_comb + 5:.1f}" '
            f'x2="{comb_x1 + (comb_x2 - comb_x1) * (0.25 + i * 0.25):.1f}" '
            f'y2="{cy + r_comb - 5:.1f}" '
            f'stroke="{combustor_dark}" stroke-width="0.7" stroke-opacity="0.55"/>'
            for i in range(3)
        ],

        # Banda CRF azul (bearing 1 / NDE) — encima del body
        f'<rect x="{crf_x - 5:.1f}" y="{cy - r_crf - 4:.1f}" width="10" '
        f'height="{2 * r_crf + 8:.1f}" fill="{bearing_band}" stroke="{bearing_dark}" '
        f'stroke-width="1" rx="2"/>',

        # Banda TRF azul (bearing 2 / DE) — encima del body
        f'<rect x="{trf_x - 5:.1f}" y="{cy - r_trf - 4:.1f}" width="10" '
        f'height="{2 * r_trf + 8:.1f}" fill="{bearing_band}" stroke="{bearing_dark}" '
        f'stroke-width="1" rx="2"/>',

        # Eje sale por la derecha
        shaft_line(outlet_x2, cy, x_offset + W, cy),

        # Label arriba
        label_top(x_offset + W / 2, y_offset + 24, label, "driver"),
    ]

    anchors = {
        "DE": (trf_x, cy),     # TRF (lado output / coupling)
        "NDE": (crf_x, cy),    # CRF (lado intake / libre)
        "TRF": (trf_x, cy),
        "CRF": (crf_x, cy),
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
