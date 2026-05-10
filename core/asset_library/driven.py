"""
core.asset_library.driven
=========================

Iconografía 2D de máquinas accionadas (driven): generador, compresores,
bombas, gearbox, compresores reciprocantes.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from core.asset_library.primitives import (
    COLORS, shaft_line, bearing_circle, machine_body, cooling_fins,
    cylinder_horizontal, cylinder_vertical, cylinder_vertical_recip,
    label_top, side_label, crankshaft_box, crankcase_recip_box,
)


# ============================================================
# Generator synchronous (Brush BDAX / GE / Westinghouse)
# ============================================================

def generator_synchronous(
    label: str = "Generador Síncrono",
    x_offset: float = 0,
    y_offset: float = 0,
) -> Tuple[str, Dict[str, Any]]:
    """
    Generador síncrono industrial estilo Bently System1 (Ciclo 23.20).
    Cilindro horizontal compacto con esquinas redondeadas (no semicírculos),
    bandas azules verticales en bearings DE/NDE, bus duct gris sutil arriba.
    """
    W, H = 320, 200
    cy = y_offset + 100

    # Paleta neutral System1-style
    body_light = "#f1f5f9"
    body_mid = "#e2e8f0"
    body_outline = "#64748b"
    body_dark = "#94a3b8"
    bearing_band = "#3b82f6"
    bearing_dark = "#1e40af"

    body_x = x_offset + 25
    body_y = y_offset + 55
    body_w = 250
    body_h = 100
    de_x = body_x + 16          # banda DE (lado coupling, izq)
    nde_x = body_x + body_w - 16  # banda NDE (lado libre, der)

    # Stator slots
    slot_xs = [body_x + 30 + i * 22 for i in range(9)]

    parts = [
        # Label arriba
        label_top(x_offset + W / 2, y_offset + 16, label, "driven"),

        # Bus duct (caja de bornes)
        f'<rect x="{body_x + body_w / 2 - 45:.1f}" y="{body_y - 13:.1f}" width="90" height="13" '
        f'rx="2" fill="{body_mid}" stroke="{body_outline}" stroke-width="1"/>',
        f'<text x="{body_x + body_w / 2:.1f}" y="{body_y - 3:.1f}" text-anchor="middle" '
        f'font-size="8" font-weight="700" fill="#475569">BUS DUCT 13.8 kV</text>',

        # Cilindro principal — esquinas redondeadas suaves (rx=10), no semicírculos
        f'<rect x="{body_x:.1f}" y="{body_y:.1f}" width="{body_w:.1f}" height="{body_h:.1f}" '
        f'rx="10" fill="{body_light}" stroke="{body_outline}" stroke-width="1.5"/>',

        # Stator slots — líneas verticales internas
        *[
            f'<line x1="{x:.1f}" y1="{body_y + 16:.1f}" x2="{x:.1f}" y2="{body_y + body_h - 16:.1f}" '
            f'stroke="{body_dark}" stroke-width="0.7" stroke-opacity="0.6"/>'
            for x in slot_xs
        ],

        # Banda azul DE (bearing lado coupling)
        f'<rect x="{de_x - 5:.1f}" y="{body_y - 4:.1f}" width="10" height="{body_h + 8:.1f}" '
        f'fill="{bearing_band}" stroke="{bearing_dark}" stroke-width="1" rx="2"/>',

        # Banda azul NDE (bearing lado libre)
        f'<rect x="{nde_x - 5:.1f}" y="{body_y - 4:.1f}" width="10" height="{body_h + 8:.1f}" '
        f'fill="{bearing_band}" stroke="{bearing_dark}" stroke-width="1" rx="2"/>',

        # Eje entra por la izquierda
        shaft_line(x_offset, cy, body_x, cy),
    ]
    anchors = {
        "DE":  (de_x, cy),
        "NDE": (nde_x, cy),
        "shaft_in": (x_offset, cy),
        "viewbox_w": W,
        "viewbox_h": H,
    }
    return "".join(parts), anchors


# ============================================================
# Centrifugal compressor (Solar Mars, MAN RB, Atlas Copco)
# ============================================================

def centrifugal_compressor(
    label: str = "Compresor Centrífugo",
    x_offset: float = 0,
    y_offset: float = 0,
) -> Tuple[str, Dict[str, Any]]:
    """Compresor centrífugo single-stage (típico inyección gas, planta criogénica)."""
    W, H = 320, 200
    body_x = x_offset + 30
    body_y = y_offset + 50
    body_w = 240
    body_h = 100
    cy = y_offset + 100
    stroke = COLORS["driven_stroke"]
    fill = COLORS["driven_fill"]
    accent = COLORS["driven_accent"]

    parts = [
        # Voluta (forma de caracol)
        f'<circle cx="{body_x + body_w * 0.55:.1f}" cy="{cy:.1f}" r="48" '
        f'fill="{accent}" stroke="{stroke}" stroke-width="2.5"/>',
        # Hub central (impeller)
        f'<circle cx="{body_x + body_w * 0.55:.1f}" cy="{cy:.1f}" r="22" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>',
        # Aletas del impeller (curvadas hacia atrás)
        *[
            f'<path d="M {body_x + body_w * 0.55:.1f} {cy:.1f} '
            f'L {body_x + body_w * 0.55 + 18 * (i % 2 * 2 - 1) * (1 if i < 4 else 0.7):.1f} '
            f'{cy + (i - 3.5) * 5:.1f}" '
            f'stroke="{stroke}" stroke-width="1.5" fill="none" stroke-linecap="round"/>'
            for i in range(8)
        ],
        # Boquilla descarga (arriba derecha)
        f'<rect x="{body_x + body_w - 30:.1f}" y="{body_y - 18:.1f}" width="32" height="22" '
        f'fill="{accent}" stroke="{stroke}" stroke-width="2" rx="3"/>',
        f'<text x="{body_x + body_w - 14:.1f}" y="{body_y - 4:.1f}" text-anchor="middle" '
        f'font-size="8" font-weight="700" fill="{stroke}">DISCH</text>',
        # Boquilla succión (abajo)
        f'<rect x="{body_x + body_w * 0.4:.1f}" y="{body_y + body_h - 4:.1f}" width="32" height="22" '
        f'fill="{accent}" stroke="{stroke}" stroke-width="2" rx="3"/>',
        f'<text x="{body_x + body_w * 0.4 + 16:.1f}" y="{body_y + body_h + 12:.1f}" text-anchor="middle" '
        f'font-size="8" font-weight="700" fill="{stroke}">SUCT</text>',
        # Eje entra por la izquierda
        shaft_line(x_offset, cy, body_x + body_w * 0.55 - 22, cy),
        # Bearings
        bearing_circle(body_x + 14, cy, r=12, label="DE", color="driven"),
        bearing_circle(body_x + body_w - 14, cy, r=12, label="NDE", color="driven"),
        label_top(x_offset + W / 2, y_offset + 24, label, "driven"),
        f'<text x="{x_offset + W / 2:.1f}" y="{y_offset + H - 8:.1f}" text-anchor="middle" '
        f'font-size="9" fill="{COLORS["text_muted"]}" font-family="-apple-system, sans-serif">'
        f'Centrifugal compressor · API 617</text>',
    ]
    anchors = {
        "DE":  (body_x + 14, cy),
        "NDE": (body_x + body_w - 14, cy),
        "shaft_in": (x_offset, cy),
        "viewbox_w": W,
        "viewbox_h": H,
    }
    return "".join(parts), anchors


# ============================================================
# Centrifugal pump (single-stage, KSB, Sulzer OH)
# ============================================================

def centrifugal_pump_single(
    label: str = "Bomba Centrífuga",
    x_offset: float = 0,
    y_offset: float = 0,
) -> Tuple[str, Dict[str, Any]]:
    """Bomba centrífuga horizontal de 1 etapa (overhung, end-suction)."""
    W, H = 290, 200
    body_y = y_offset + 50
    body_h = 100
    cy = y_offset + 100
    stroke = COLORS["driven_stroke"]
    fill = COLORS["driven_fill"]
    accent = COLORS["driven_accent"]

    # Volume cae a la derecha
    spool_x = x_offset + 20
    spool_w = 80
    casing_cx = x_offset + 200
    casing_r = 50

    parts = [
        # Spool del bearing housing (izquierda)
        f'<rect x="{spool_x:.1f}" y="{body_y + 20:.1f}" width="{spool_w:.1f}" height="{body_h - 40:.1f}" '
        f'rx="6" fill="{fill}" stroke="{stroke}" stroke-width="2.5"/>',
        # Patas
        f'<rect x="{spool_x + 10:.1f}" y="{body_y + body_h - 18:.1f}" width="14" height="18" '
        f'fill="{stroke}"/>',
        f'<rect x="{spool_x + spool_w - 24:.1f}" y="{body_y + body_h - 18:.1f}" width="14" height="18" '
        f'fill="{stroke}"/>',
        # Voluta circular
        f'<circle cx="{casing_cx:.1f}" cy="{cy:.1f}" r="{casing_r}" '
        f'fill="{accent}" stroke="{stroke}" stroke-width="2.5"/>',
        # Impeller (núcleo)
        f'<circle cx="{casing_cx:.1f}" cy="{cy:.1f}" r="22" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>',
        # Aspas curvas
        *[
            f'<path d="M {casing_cx + 5 * (i % 2 * 2 - 1):.1f} {cy + (i - 2) * 4:.1f} '
            f'q 8 -6 18 4" '
            f'stroke="{stroke}" stroke-width="1.5" fill="none" stroke-linecap="round"/>'
            for i in range(5)
        ],
        # Succión (lado derecho)
        f'<rect x="{casing_cx + casing_r:.1f}" y="{cy - 12:.1f}" width="22" height="24" '
        f'fill="{accent}" stroke="{stroke}" stroke-width="2" rx="3"/>',
        f'<text x="{casing_cx + casing_r + 11:.1f}" y="{cy + 3:.1f}" text-anchor="middle" '
        f'font-size="7" font-weight="700" fill="{stroke}">SUCT</text>',
        # Descarga (arriba)
        f'<rect x="{casing_cx - 14:.1f}" y="{body_y - 18:.1f}" width="28" height="24" '
        f'fill="{accent}" stroke="{stroke}" stroke-width="2" rx="3"/>',
        f'<text x="{casing_cx:.1f}" y="{body_y - 4:.1f}" text-anchor="middle" '
        f'font-size="7" font-weight="700" fill="{stroke}">DISCH</text>',
        # Eje entra por la izquierda hasta el centro de la voluta
        shaft_line(x_offset, cy, casing_cx - 22, cy),
        # Bearings (DE = lado coupling, NDE = lado opposite)
        bearing_circle(spool_x + 14, cy, r=12, label="DE", color="driven"),
        bearing_circle(spool_x + spool_w - 14, cy, r=12, label="NDE", color="driven"),
        label_top(x_offset + W / 2, y_offset + 24, label, "driven"),
        f'<text x="{x_offset + W / 2:.1f}" y="{y_offset + H - 8:.1f}" text-anchor="middle" '
        f'font-size="9" fill="{COLORS["text_muted"]}" font-family="-apple-system, sans-serif">'
        f'Centrifugal pump · single stage · API 610</text>',
    ]
    anchors = {
        "DE":  (spool_x + 14, cy),
        "NDE": (spool_x + spool_w - 14, cy),
        "shaft_in": (x_offset, cy),
        "viewbox_w": W,
        "viewbox_h": H,
    }
    return "".join(parts), anchors


# ============================================================
# Multistage centrifugal pump (Flowserve DMX/DDM, Sulzer MSD)
# ============================================================

def centrifugal_pump_multistage(
    label: str = "Bomba Multietapa",
    x_offset: float = 0,
    y_offset: float = 0,
) -> Tuple[str, Dict[str, Any]]:
    """
    Bomba centrífuga horizontal multietapa (BB3 type, Flowserve DMX,
    Sulzer MSD) — barrel casing con N etapas en serie.
    """
    W, H = 380, 200
    body_x = x_offset + 30
    body_y = y_offset + 60
    body_w = 320
    body_h = 80
    cy = y_offset + 100
    stroke = COLORS["driven_stroke"]
    fill = COLORS["driven_fill"]
    accent = COLORS["driven_accent"]

    n_stages = 6

    parts = [
        # Barrel casing (cuerpo cilíndrico)
        f'<rect x="{body_x:.1f}" y="{body_y:.1f}" width="{body_w:.1f}" height="{body_h:.1f}" '
        f'rx="40" fill="{fill}" stroke="{stroke}" stroke-width="2.5"/>',
        # Tapa succión (izq)
        f'<rect x="{body_x:.1f}" y="{body_y:.1f}" width="22" height="{body_h:.1f}" '
        f'rx="40" fill="{accent}" stroke="{stroke}" stroke-width="2"/>',
        # Tapa descarga (der)
        f'<rect x="{body_x + body_w - 22:.1f}" y="{body_y:.1f}" width="22" height="{body_h:.1f}" '
        f'rx="40" fill="{accent}" stroke="{stroke}" stroke-width="2"/>',
        # Etapas (separadores verticales en el barrel)
        *[
            f'<line x1="{body_x + 30 + i * (body_w - 60) / n_stages:.1f}" y1="{body_y + 10:.1f}" '
            f'x2="{body_x + 30 + i * (body_w - 60) / n_stages:.1f}" y2="{body_y + body_h - 10:.1f}" '
            f'stroke="{stroke}" stroke-width="1.5" stroke-opacity="0.7"/>'
            for i in range(1, n_stages)
        ],
        # Etiquetas etapas
        *[
            f'<text x="{body_x + 30 + (i - 0.5) * (body_w - 60) / n_stages:.1f}" '
            f'y="{cy + 4:.1f}" text-anchor="middle" font-size="9" font-weight="700" '
            f'fill="{stroke}" font-family="SF Mono, monospace">E{i}</text>'
            for i in range(1, n_stages + 1)
        ],
        # Boquilla succión (abajo izq)
        f'<rect x="{body_x + 28:.1f}" y="{body_y + body_h - 4:.1f}" width="32" height="20" '
        f'fill="{accent}" stroke="{stroke}" stroke-width="2" rx="3"/>',
        f'<text x="{body_x + 44:.1f}" y="{body_y + body_h + 11:.1f}" text-anchor="middle" '
        f'font-size="8" font-weight="700" fill="{stroke}">SUCT</text>',
        # Boquilla descarga (arriba der)
        f'<rect x="{body_x + body_w - 60:.1f}" y="{body_y - 18:.1f}" width="32" height="22" '
        f'fill="{accent}" stroke="{stroke}" stroke-width="2" rx="3"/>',
        f'<text x="{body_x + body_w - 44:.1f}" y="{body_y - 4:.1f}" text-anchor="middle" '
        f'font-size="8" font-weight="700" fill="{stroke}">DISCH</text>',
        # Eje
        shaft_line(x_offset, cy, body_x + 6, cy),
        shaft_line(body_x + body_w - 6, cy, x_offset + W, cy),
        # Bearings
        bearing_circle(body_x + 14, cy, r=12, label="DE", color="driven"),
        bearing_circle(body_x + body_w - 14, cy, r=12, label="NDE", color="driven"),
        label_top(x_offset + W / 2, y_offset + 24, label, "driven"),
        f'<text x="{x_offset + W / 2:.1f}" y="{y_offset + H - 8:.1f}" text-anchor="middle" '
        f'font-size="9" fill="{COLORS["text_muted"]}" font-family="-apple-system, sans-serif">'
        f'Multistage pump · BB3 · API 610 · {n_stages} etapas</text>',
    ]
    anchors = {
        "DE":  (body_x + 14, cy),
        "NDE": (body_x + body_w - 14, cy),
        "shaft_in": (x_offset, cy),
        "viewbox_w": W,
        "viewbox_h": H,
    }
    return "".join(parts), anchors


# ============================================================
# Gearbox parallel-shaft (Lufkin, Voith, Renk)
# ============================================================

def gearbox_parallel(
    label: str = "Gearbox",
    x_offset: float = 0,
    y_offset: float = 0,
) -> Tuple[str, Dict[str, Any]]:
    """Caja reductora paralela (input arriba, output abajo, o single-shaft)."""
    W, H = 260, 200
    body_x = x_offset + 30
    body_y = y_offset + 50
    body_w = 200
    body_h = 100
    cy = y_offset + 100
    stroke = COLORS["driven_stroke"]
    fill = COLORS["driven_fill"]
    accent = COLORS["driven_accent"]

    parts = [
        # Cuerpo
        f'<rect x="{body_x:.1f}" y="{body_y:.1f}" width="{body_w:.1f}" height="{body_h:.1f}" '
        f'rx="10" fill="{fill}" stroke="{stroke}" stroke-width="2.5"/>',
        # Engranaje grande (centro)
        f'<circle cx="{body_x + body_w * 0.55:.1f}" cy="{cy + 10:.1f}" r="34" '
        f'fill="{accent}" stroke="{stroke}" stroke-width="2"/>',
        # Engranaje chico (input, arriba)
        f'<circle cx="{body_x + body_w * 0.35:.1f}" cy="{cy - 22:.1f}" r="20" '
        f'fill="{accent}" stroke="{stroke}" stroke-width="2"/>',
        # Dientes (representación simple)
        *[
            f'<line x1="{body_x + body_w * 0.55 + 30 * 1:.1f}" y1="{cy + 10:.1f}" '
            f'x2="{body_x + body_w * 0.55 + 38:.1f}" y2="{cy + 10:.1f}" stroke="{stroke}" stroke-width="2"/>'
            for _ in [0]  # placeholder para no romper la lista
        ],
        # Eje input (top, izquierda)
        f'<line x1="{x_offset:.1f}" y1="{cy - 22:.1f}" x2="{body_x + body_w * 0.35:.1f}" '
        f'y2="{cy - 22:.1f}" stroke="{COLORS["shaft"]}" stroke-width="3.5"/>',
        # Eje output (bottom, derecha)
        f'<line x1="{body_x + body_w * 0.55:.1f}" y1="{cy + 10:.1f}" x2="{x_offset + W:.1f}" '
        f'y2="{cy + 10:.1f}" stroke="{COLORS["shaft"]}" stroke-width="4"/>',
        # Bearings input
        bearing_circle(body_x + 14, cy - 22, r=10, label="In", color="driven"),
        # Bearings output
        bearing_circle(body_x + body_w - 14, cy + 10, r=10, label="Out", color="driven"),
        label_top(x_offset + W / 2, y_offset + 24, label, "driven"),
        f'<text x="{x_offset + W / 2:.1f}" y="{y_offset + H - 8:.1f}" text-anchor="middle" '
        f'font-size="9" fill="{COLORS["text_muted"]}" font-family="-apple-system, sans-serif">'
        f'Parallel-shaft gearbox · API 613</text>',
    ]
    anchors = {
        "DE":  (body_x + 14, cy - 22),         # input DE
        "NDE": (body_x + body_w - 14, cy + 10), # output NDE
        "input_DE": (body_x + 14, cy - 22),
        "output_DE": (body_x + body_w - 14, cy + 10),
        "shaft_in": (x_offset, cy - 22),
        "shaft_out": (x_offset + W, cy + 10),
        "viewbox_w": W,
        "viewbox_h": H,
    }
    return "".join(parts), anchors


# ============================================================
# Recip compressor 2-cyl boxer opposed (Ariel JGE-2, Burckhardt 2P)
# ============================================================

def recip_compressor_boxer_2cyl(
    label: str = "Compresor Recip 2-cyl",
    x_offset: float = 0,
    y_offset: float = 0,
) -> Tuple[str, Dict[str, Any]]:
    """
    Compresor reciprocante 2 cilindros boxer (verticalmente opuestos).
    Geometría tipo Ariel JGE-2 / Burckhardt 2P:
      - Cigüeñal sale por el centro horizontal del crankcase.
      - C1 vertical apuntando arriba, C2 vertical apuntando abajo, opuestos.
      - Shaft entra por la izquierda (acoplado al motor con coupling rígido).
    """
    W, H = 360, 240
    # cy fijo en 100 para que el shaft quede colineal con driver + coupling
    # (todos los iconos del catálogo comparten esa convención).
    cy = y_offset + 100
    crank_h = 50
    crank_y = cy - crank_h / 2
    crank_x = x_offset + 110
    crank_w = 140
    stroke = COLORS["driven_stroke"]

    # Posiciones de los cilindros (centrados horizontalmente sobre el crank)
    cyl_cx = crank_x + crank_w / 2
    cyl_length = 60   # ajustado a 60 para que cilindros queden dentro del viewBox
    cyl_bore = 40
    c1_cy = crank_y - 10           # extremo inferior del cilindro arriba
    c2_cy = crank_y + crank_h + 10  # extremo superior del cilindro abajo

    parts = [
        # Eje entra por la izquierda al centro del crankcase (acople rígido al motor)
        shaft_line(x_offset, cy, crank_x, cy),
        # Crankcase central con detalle: flange con bolts + mounting feet + cigüeñal
        crankcase_recip_box(crank_x, crank_y, crank_w, crank_h, "driven"),
        # Bielas (connecting rods) verticales — más anchas, con cabeza de biela visible
        f'<rect x="{cyl_cx - 3:.1f}" y="{c1_cy:.1f}" width="6" height="{cy - c1_cy:.1f}" '
        f'fill="{stroke}" fill-opacity="0.85"/>',
        f'<rect x="{cyl_cx - 3:.1f}" y="{cy:.1f}" width="6" height="{c2_cy - cy:.1f}" '
        f'fill="{stroke}" fill-opacity="0.85"/>',
        # C1 — cilindro arriba con cabeza+válvulas+flange detallado
        cylinder_vertical_recip(
            cx=cyl_cx, cy=c1_cy, length=cyl_length, bore=cyl_bore,
            role="driven", label="C1", direction="up",
        ),
        # C2 — cilindro abajo con cabeza+válvulas+flange detallado
        cylinder_vertical_recip(
            cx=cyl_cx, cy=c2_cy, length=cyl_length, bore=cyl_bore,
            role="driven", label="C2", direction="down",
        ),
        # Bearings del cigüeñal (DE = lado coupling, NDE = lado opuesto)
        bearing_circle(crank_x + 14, cy, r=11, label="DE", color="driven"),
        bearing_circle(crank_x + crank_w - 14, cy, r=11, label="NDE", color="driven"),
        # Label arriba (encima de C1)
        label_top(x_offset + W / 2, y_offset + 16, label, "driven"),
        # Subtitle abajo
        f'<text x="{x_offset + W / 2:.1f}" y="{y_offset + H - 6:.1f}" text-anchor="middle" '
        f'font-size="9" fill="{COLORS["text_muted"]}" font-family="-apple-system, sans-serif">'
        f'Recip compressor · 2-cyl boxer · API 618</text>',
    ]
    anchors = {
        "DE":  (crank_x + 14, cy),
        "NDE": (crank_x + crank_w - 14, cy),
        # Anchors de crosshead — para acelerómetros API 618 sobre cada cilindro
        "C1": (cyl_cx, c1_cy - cyl_length / 2),
        "C2": (cyl_cx, c2_cy + cyl_length / 2),
        "shaft_in": (x_offset, cy),
        "viewbox_w": W,
        "viewbox_h": H,
    }
    return "".join(parts), anchors


# ============================================================
# Recip compressor 4-cyl boxer (Ariel JGM-4, Burckhardt 4P)
# ============================================================

def recip_compressor_boxer_4cyl(
    label: str = "Compresor Recip 4-cyl",
    x_offset: float = 0,
    y_offset: float = 0,
) -> Tuple[str, Dict[str, Any]]:
    """
    Compresor reciprocante 4 cilindros boxer (Ariel JGM-4 / KBK/4 / Burckhardt 4P).

    Geometría real del Ariel KBK/4:
      - Cigüeñal sale por el centro horizontal del crankcase.
      - Pareja izquierda: C1 arriba + C2 abajo (verticalmente opuestos).
      - Pareja derecha:   C3 arriba + C4 abajo (verticalmente opuestos).
      - Shaft entra por la izquierda con acople rígido al motor eléctrico.
      - 4 acelerómetros crosshead — uno por cilindro (API 618).
    """
    W, H = 420, 240
    # cy fijo en 100 para alinear con shaft del driver + coupling.
    cy = y_offset + 100
    crank_h = 50
    crank_y = cy - crank_h / 2
    crank_x = x_offset + 90
    crank_w = 240
    stroke = COLORS["driven_stroke"]

    # Posiciones de los 4 cilindros (2 pares: izq y der)
    cyl_left_cx = crank_x + 60
    cyl_right_cx = crank_x + crank_w - 60
    cyl_length = 60   # ajustado para que cilindros queden dentro del viewBox H=240
    cyl_bore = 36
    top_cy = crank_y - 10           # extremo inferior de cilindro arriba
    bot_cy = crank_y + crank_h + 10  # extremo superior de cilindro abajo

    parts = [
        # Eje entra por la izquierda al centro del crankcase (acople rígido al motor)
        shaft_line(x_offset, cy, crank_x, cy),
        # Crankcase central con detalle: flange con bolts + mounting feet + cigüeñal
        crankcase_recip_box(crank_x, crank_y, crank_w, crank_h, "driven"),
        # Bielas (rods) verticales con grosor — pareja izquierda
        f'<rect x="{cyl_left_cx - 3:.1f}" y="{top_cy:.1f}" width="6" height="{cy - top_cy:.1f}" fill="{stroke}" fill-opacity="0.85"/>',
        f'<rect x="{cyl_left_cx - 3:.1f}" y="{cy:.1f}" width="6" height="{bot_cy - cy:.1f}" fill="{stroke}" fill-opacity="0.85"/>',
        # Pareja derecha
        f'<rect x="{cyl_right_cx - 3:.1f}" y="{top_cy:.1f}" width="6" height="{cy - top_cy:.1f}" fill="{stroke}" fill-opacity="0.85"/>',
        f'<rect x="{cyl_right_cx - 3:.1f}" y="{cy:.1f}" width="6" height="{bot_cy - cy:.1f}" fill="{stroke}" fill-opacity="0.85"/>',
        # 4 cilindros con detalle técnico (cabeza+válvulas+flange+bolts)
        cylinder_vertical_recip(
            cx=cyl_left_cx, cy=top_cy, length=cyl_length, bore=cyl_bore,
            role="driven", label="C1", direction="up",
        ),
        cylinder_vertical_recip(
            cx=cyl_left_cx, cy=bot_cy, length=cyl_length, bore=cyl_bore,
            role="driven", label="C2", direction="down",
        ),
        cylinder_vertical_recip(
            cx=cyl_right_cx, cy=top_cy, length=cyl_length, bore=cyl_bore,
            role="driven", label="C3", direction="up",
        ),
        cylinder_vertical_recip(
            cx=cyl_right_cx, cy=bot_cy, length=cyl_length, bore=cyl_bore,
            role="driven", label="C4", direction="down",
        ),
        # Bearings del cigüeñal
        bearing_circle(crank_x + 14, cy, r=11, label="DE", color="driven"),
        bearing_circle(crank_x + crank_w - 14, cy, r=11, label="NDE", color="driven"),
        # Label arriba
        label_top(x_offset + W / 2, y_offset + 16, label, "driven"),
        # Subtitle abajo
        f'<text x="{x_offset + W / 2:.1f}" y="{y_offset + H - 6:.1f}" text-anchor="middle" '
        f'font-size="9" fill="{COLORS["text_muted"]}" font-family="-apple-system, sans-serif">'
        f'Recip compressor · 4-cyl boxer (Ariel KBK/4 style) · API 618</text>',
    ]
    anchors = {
        "DE":  (crank_x + 14, cy),
        "NDE": (crank_x + crank_w - 14, cy),
        # Anchors de crosshead — uno por cilindro para acelerómetros API 618
        "C1": (cyl_left_cx, top_cy - cyl_length / 2),
        "C2": (cyl_left_cx, bot_cy + cyl_length / 2),
        "C3": (cyl_right_cx, top_cy - cyl_length / 2),
        "C4": (cyl_right_cx, bot_cy + cyl_length / 2),
        "shaft_in": (x_offset, cy),
        "viewbox_w": W,
        "viewbox_h": H,
    }
    return "".join(parts), anchors


__all__ = [
    "generator_synchronous",
    "centrifugal_compressor",
    "centrifugal_pump_single",
    "centrifugal_pump_multistage",
    "gearbox_parallel",
    "recip_compressor_boxer_2cyl",
    "recip_compressor_boxer_4cyl",
]
