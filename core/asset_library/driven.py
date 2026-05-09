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
    cylinder_horizontal, label_top, side_label, crankshaft_box,
)


# ============================================================
# Generator synchronous (Brush BDAX / GE / Westinghouse)
# ============================================================

def generator_synchronous(
    label: str = "Generador Síncrono",
    x_offset: float = 0,
    y_offset: float = 0,
) -> Tuple[str, Dict[str, Any]]:
    """Generador síncrono industrial (Brush, GE, Westinghouse)."""
    W, H = 340, 200
    body_x = x_offset + 30
    body_y = y_offset + 45
    body_w = 260
    body_h = 110
    cy = y_offset + 100
    stroke = COLORS["driven_stroke"]
    fill = COLORS["driven_fill"]
    accent = COLORS["driven_accent"]

    parts = [
        # Estator (cuerpo principal)
        f'<rect x="{body_x:.1f}" y="{body_y:.1f}" width="{body_w:.1f}" height="{body_h:.1f}" '
        f'rx="14" fill="{fill}" stroke="{stroke}" stroke-width="2.5"/>',
        # Tapas DE/NDE
        f'<rect x="{body_x:.1f}" y="{body_y:.1f}" width="20" height="{body_h:.1f}" '
        f'rx="14" fill="{accent}" stroke="{stroke}" stroke-width="2"/>',
        f'<rect x="{body_x + body_w - 20:.1f}" y="{body_y:.1f}" width="20" height="{body_h:.1f}" '
        f'rx="14" fill="{accent}" stroke="{stroke}" stroke-width="2"/>',
        # Slots del estator (6 líneas dentro del cuerpo)
        *[
            f'<line x1="{body_x + 30 + i * 30:.1f}" y1="{body_y + 14:.1f}" '
            f'x2="{body_x + 30 + i * 30:.1f}" y2="{body_y + body_h - 14:.1f}" '
            f'stroke="{stroke}" stroke-width="1" stroke-opacity="0.5"/>'
            for i in range(7)
        ],
        # Caja de bornes (bus duct) arriba
        f'<rect x="{body_x + body_w / 2 - 50:.1f}" y="{body_y - 22:.1f}" width="100" height="22" '
        f'rx="3" fill="{accent}" stroke="{stroke}" stroke-width="2"/>',
        f'<text x="{body_x + body_w / 2:.1f}" y="{body_y - 7:.1f}" text-anchor="middle" '
        f'font-size="9" font-weight="700" fill="{stroke}">BUS DUCT 13.8 kV</text>',
        # Eje (entra por la izquierda desde el coupling)
        shaft_line(x_offset, cy, body_x + 6, cy),
        # Bearings DE (lado coupling, izquierda) y NDE (derecha)
        bearing_circle(body_x + 14, cy, r=13, label="DE", color="driven"),
        bearing_circle(body_x + body_w - 14, cy, r=13, label="NDE", color="driven"),
        label_top(x_offset + W / 2, y_offset + 24, label, "driven"),
        f'<text x="{x_offset + W / 2:.1f}" y="{y_offset + H - 8:.1f}" text-anchor="middle" '
        f'font-size="9" fill="{COLORS["text_muted"]}" font-family="-apple-system, sans-serif">'
        f'Synchronous generator · 3-phase · 60 Hz</text>',
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
    """Compresor reciprocante 2 cilindros horizontalmente opuestos."""
    W, H = 360, 200
    crank_x = x_offset + 130
    crank_w = 100
    crank_h = 70
    crank_y = y_offset + 65
    cy = crank_y + crank_h / 2
    stroke = COLORS["driven_stroke"]

    parts = [
        crankshaft_box(crank_x, crank_y, crank_w, crank_h, "driven"),
        # Cilindro izquierdo
        cylinder_horizontal(
            cx=x_offset + 70, cy=cy, length=110, bore=36,
            role="driven", label="C1",
        ),
        # Cilindro derecho
        cylinder_horizontal(
            cx=x_offset + 290, cy=cy, length=110, bore=36,
            role="driven", label="C2",
        ),
        # Líneas de bielas (rod) entre cilindros y crankcase
        f'<line x1="{x_offset + 125:.1f}" y1="{cy:.1f}" x2="{crank_x + 4:.1f}" '
        f'y2="{cy:.1f}" stroke="{stroke}" stroke-width="2"/>',
        f'<line x1="{crank_x + crank_w - 4:.1f}" y1="{cy:.1f}" x2="{x_offset + 235:.1f}" '
        f'y2="{cy:.1f}" stroke="{stroke}" stroke-width="2"/>',
        # Eje principal entra por la izquierda (desde el coupling)
        shaft_line(x_offset, cy, crank_x + 6, cy),
        # Bearings principales del cigüeñal (DE = lado coupling, NDE = otro lado)
        bearing_circle(crank_x + 12, cy, r=12, label="DE", color="driven"),
        bearing_circle(crank_x + crank_w - 12, cy, r=12, label="NDE", color="driven"),
        label_top(x_offset + W / 2, y_offset + 24, label, "driven"),
        f'<text x="{x_offset + W / 2:.1f}" y="{y_offset + H - 8:.1f}" text-anchor="middle" '
        f'font-size="9" fill="{COLORS["text_muted"]}" font-family="-apple-system, sans-serif">'
        f'Recip compressor · 2-cyl boxer · API 618</text>',
    ]
    anchors = {
        "DE":  (crank_x + 12, cy),
        "NDE": (crank_x + crank_w - 12, cy),
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
    Compresor reciprocante 4 cilindros horizontalmente opuestos.
    Convención boxer: 2 cilindros arriba (C1, C3) + 2 abajo (C2, C4).
    """
    W, H = 360, 200
    crank_x = x_offset + 130
    crank_w = 100
    crank_h = 70
    crank_y = y_offset + 65
    cy = crank_y + crank_h / 2
    stroke = COLORS["driven_stroke"]

    parts = [
        crankshaft_box(crank_x, crank_y, crank_w, crank_h, "driven"),
        # 4 cilindros: 2 izquierda (top + bottom), 2 derecha (top + bottom)
        # Lado izquierdo top (C1)
        cylinder_horizontal(
            cx=x_offset + 70, cy=cy - 22, length=100, bore=28,
            role="driven", label="C1",
        ),
        # Lado izquierdo bottom (C2)
        cylinder_horizontal(
            cx=x_offset + 70, cy=cy + 22, length=100, bore=28,
            role="driven", label="C2",
        ),
        # Lado derecho top (C3)
        cylinder_horizontal(
            cx=x_offset + 290, cy=cy - 22, length=100, bore=28,
            role="driven", label="C3",
        ),
        # Lado derecho bottom (C4)
        cylinder_horizontal(
            cx=x_offset + 290, cy=cy + 22, length=100, bore=28,
            role="driven", label="C4",
        ),
        # Líneas de biela (4)
        f'<line x1="{x_offset + 120:.1f}" y1="{cy - 22:.1f}" x2="{crank_x + 4:.1f}" y2="{cy - 5:.1f}" '
        f'stroke="{stroke}" stroke-width="1.8"/>',
        f'<line x1="{x_offset + 120:.1f}" y1="{cy + 22:.1f}" x2="{crank_x + 4:.1f}" y2="{cy + 5:.1f}" '
        f'stroke="{stroke}" stroke-width="1.8"/>',
        f'<line x1="{crank_x + crank_w - 4:.1f}" y1="{cy - 5:.1f}" x2="{x_offset + 240:.1f}" y2="{cy - 22:.1f}" '
        f'stroke="{stroke}" stroke-width="1.8"/>',
        f'<line x1="{crank_x + crank_w - 4:.1f}" y1="{cy + 5:.1f}" x2="{x_offset + 240:.1f}" y2="{cy + 22:.1f}" '
        f'stroke="{stroke}" stroke-width="1.8"/>',
        # Eje principal
        shaft_line(x_offset, cy, crank_x + 6, cy),
        # Bearings principales
        bearing_circle(crank_x + 12, cy, r=12, label="DE", color="driven"),
        bearing_circle(crank_x + crank_w - 12, cy, r=12, label="NDE", color="driven"),
        label_top(x_offset + W / 2, y_offset + 24, label, "driven"),
        f'<text x="{x_offset + W / 2:.1f}" y="{y_offset + H - 8:.1f}" text-anchor="middle" '
        f'font-size="9" fill="{COLORS["text_muted"]}" font-family="-apple-system, sans-serif">'
        f'Recip compressor · 4-cyl boxer · API 618</text>',
    ]
    anchors = {
        "DE":  (crank_x + 12, cy),
        "NDE": (crank_x + crank_w - 12, cy),
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
