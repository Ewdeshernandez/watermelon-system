"""
core.asset_library.primitives
=============================

Helpers SVG reutilizables para construir iconografía industrial:
cuerpos de máquina, cojinetes, cilindros, ejes, acoples, labels.
Todos los helpers devuelven strings SVG componibles.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

# Paleta corporativa estandarizada
COLORS = {
    "driver_fill":   "#dbeafe",
    "driver_stroke": "#1e3a8a",
    "driver_dark":   "#1e40af",
    "driver_accent": "#bfdbfe",
    "driven_fill":   "#dcfce7",
    "driven_stroke": "#14532d",
    "driven_dark":   "#166534",
    "driven_accent": "#bbf7d0",
    "coupling_fill":   "#fde68a",
    "coupling_stroke": "#92400e",
    "shaft":         "#374151",
    "bearing_fill":  "#ffffff",
    "warning":       "#fbbf24",
    "danger":        "#ef4444",
    "text_dark":     "#0f172a",
    "text_muted":    "#64748b",
}


def shaft_line(x1: float, y1: float, x2: float, y2: float, width: float = 4.5) -> str:
    """Eje horizontal (línea gruesa)."""
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{COLORS["shaft"]}" stroke-width="{width}" stroke-linecap="round"/>'
    )


def bearing_circle(
    cx: float,
    cy: float,
    r: float = 14,
    label: str = "",
    color: str = "driver",
) -> str:
    """
    Círculo blanco con borde — representa un cojinete.
    color = 'driver' / 'driven' / custom hex.
    """
    if color == "driver":
        stroke = COLORS["driver_stroke"]
    elif color == "driven":
        stroke = COLORS["driven_stroke"]
    else:
        stroke = color
    out = (
        f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" '
        f'fill="{COLORS["bearing_fill"]}" stroke="{stroke}" stroke-width="2.5"/>'
    )
    if label:
        out += (
            f'<text x="{cx:.1f}" y="{cy + 4:.1f}" text-anchor="middle" '
            f'font-size="{int(r * 0.7)}" font-weight="700" '
            f'font-family="SF Mono, Menlo, monospace" fill="{stroke}">{label}</text>'
        )
    return out


def machine_body(
    x: float,
    y: float,
    w: float,
    h: float,
    role: str = "driver",
    rx: float = 8,
    has_endcaps: bool = True,
) -> str:
    """
    Cuerpo principal rectangular (motor, turbina, generador). Si
    has_endcaps=True añade dos tapas más oscuras a los extremos
    representando las DE/NDE bearings housing.
    """
    if role == "driver":
        fill = COLORS["driver_fill"]
        stroke = COLORS["driver_stroke"]
        accent = COLORS["driver_accent"]
    else:
        fill = COLORS["driven_fill"]
        stroke = COLORS["driven_stroke"]
        accent = COLORS["driven_accent"]

    parts: List[str] = []
    parts.append(
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
        f'rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="2.5"/>'
    )
    if has_endcaps:
        cap_w = max(14, w / 14)
        # Tapa izquierda
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{cap_w:.1f}" height="{h:.1f}" '
            f'rx="{rx}" fill="{accent}" stroke="{stroke}" stroke-width="2"/>'
        )
        # Tapa derecha
        parts.append(
            f'<rect x="{x + w - cap_w:.1f}" y="{y:.1f}" width="{cap_w:.1f}" height="{h:.1f}" '
            f'rx="{rx}" fill="{accent}" stroke="{stroke}" stroke-width="2"/>'
        )
    return "".join(parts)


def cooling_fins(
    x: float,
    y: float,
    w: float,
    h: float,
    n_fins: int = 6,
    role: str = "driver",
) -> str:
    """Dibuja lineas verticales paralelas (aletas de enfriamiento de un motor)."""
    stroke = COLORS["driver_stroke"] if role == "driver" else COLORS["driven_stroke"]
    parts: List[str] = []
    if n_fins < 1:
        return ""
    spacing = w / (n_fins + 1)
    for i in range(1, n_fins + 1):
        fx = x + spacing * i
        parts.append(
            f'<line x1="{fx:.1f}" y1="{y:.1f}" x2="{fx:.1f}" y2="{y + h:.1f}" '
            f'stroke="{stroke}" stroke-width="1" stroke-opacity="0.55"/>'
        )
    return "".join(parts)


def cylinder_horizontal(
    cx: float,
    cy: float,
    length: float,
    bore: float,
    role: str = "driven",
    label: str = "",
) -> str:
    """
    Cilindro horizontal (compresor reciprocante / motor reciprocante visto
    desde arriba). cx,cy = centro del cilindro; length = largo total;
    bore = diámetro del cilindro.
    """
    if role == "driver":
        fill = COLORS["driver_fill"]
        stroke = COLORS["driver_stroke"]
    else:
        fill = COLORS["driven_fill"]
        stroke = COLORS["driven_stroke"]

    half_l = length / 2
    half_b = bore / 2
    parts: List[str] = []
    # Cuerpo del cilindro (rectángulo redondeado)
    parts.append(
        f'<rect x="{cx - half_l:.1f}" y="{cy - half_b:.1f}" '
        f'width="{length:.1f}" height="{bore:.1f}" rx="6" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>'
    )
    # Cabeza del cilindro (extremo)
    head_w = bore * 0.3
    parts.append(
        f'<rect x="{cx + half_l - head_w:.1f}" y="{cy - half_b - 3:.1f}" '
        f'width="{head_w:.1f}" height="{bore + 6:.1f}" '
        f'fill="{stroke}" rx="3"/>'
    )
    if label:
        parts.append(
            f'<text x="{cx:.1f}" y="{cy + 3:.1f}" text-anchor="middle" '
            f'font-size="9" font-weight="700" fill="{stroke}" '
            f'font-family="SF Mono, monospace">{label}</text>'
        )
    return "".join(parts)


def cylinder_vertical(
    cx: float,
    cy: float,
    length: float,
    bore: float,
    role: str = "driven",
    label: str = "",
    direction: str = "up",
) -> str:
    """Cilindro vertical (típico recip motor in-line / boxer arriba/abajo)."""
    if role == "driver":
        fill = COLORS["driver_fill"]
        stroke = COLORS["driver_stroke"]
    else:
        fill = COLORS["driven_fill"]
        stroke = COLORS["driven_stroke"]

    half_b = bore / 2
    parts: List[str] = []
    if direction == "up":
        top = cy - length
        parts.append(
            f'<rect x="{cx - half_b:.1f}" y="{top:.1f}" '
            f'width="{bore:.1f}" height="{length:.1f}" rx="5" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>'
        )
        # Cabeza arriba
        parts.append(
            f'<rect x="{cx - half_b - 3:.1f}" y="{top - 6:.1f}" '
            f'width="{bore + 6:.1f}" height="6" rx="2" fill="{stroke}"/>'
        )
    else:  # down
        parts.append(
            f'<rect x="{cx - half_b:.1f}" y="{cy:.1f}" '
            f'width="{bore:.1f}" height="{length:.1f}" rx="5" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>'
        )
        # Cabeza abajo
        parts.append(
            f'<rect x="{cx - half_b - 3:.1f}" y="{cy + length:.1f}" '
            f'width="{bore + 6:.1f}" height="6" rx="2" fill="{stroke}"/>'
        )
    if label:
        ly = (cy - length / 2) if direction == "up" else (cy + length / 2)
        parts.append(
            f'<text x="{cx:.1f}" y="{ly + 3:.1f}" text-anchor="middle" '
            f'font-size="8" font-weight="700" fill="{stroke}" '
            f'font-family="SF Mono, monospace">{label}</text>'
        )
    return "".join(parts)


def label_top(cx: float, cy: float, text: str, role: str = "driver", size: int = 14) -> str:
    """Texto label arriba del icono (nombre del activo)."""
    color = COLORS["driver_stroke"] if role == "driver" else COLORS["driven_stroke"]
    return (
        f'<text x="{cx:.1f}" y="{cy:.1f}" text-anchor="middle" '
        f'font-size="{size}" font-weight="800" fill="{color}" '
        f'font-family="-apple-system, Segoe UI, Roboto, sans-serif">{text}</text>'
    )


def side_label(x: float, y: float, text: str, role: str = "driver") -> str:
    """Etiqueta DE/NDE chica."""
    color = COLORS["driver_stroke"] if role == "driver" else COLORS["driven_stroke"]
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" '
        f'font-size="9" font-weight="700" fill="{color}" '
        f'font-family="SF Mono, monospace">{text}</text>'
    )


def crankshaft_box(
    x: float,
    y: float,
    w: float,
    h: float,
    role: str = "driver",
) -> str:
    """Caja del cigüeñal — representación 2D del crankcase de un recip."""
    if role == "driver":
        fill = COLORS["driver_fill"]
        stroke = COLORS["driver_stroke"]
    else:
        fill = COLORS["driven_fill"]
        stroke = COLORS["driven_stroke"]
    parts = [
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
        f'rx="6" fill="{fill}" stroke="{stroke}" stroke-width="2.5"/>'
    ]
    # Línea central horizontal (donde corre el cigüeñal)
    cy = y + h / 2
    parts.append(
        f'<line x1="{x + 6:.1f}" y1="{cy:.1f}" x2="{x + w - 6:.1f}" y2="{cy:.1f}" '
        f'stroke="{stroke}" stroke-width="1.2" stroke-dasharray="4,3" stroke-opacity="0.55"/>'
    )
    return "".join(parts)


def axial_marker(x: float, y: float, role: str = "driver") -> str:
    """Pequeño marker triangular para sensor axial (thrust)."""
    color = COLORS["driver_stroke"] if role == "driver" else COLORS["driven_stroke"]
    return (
        f'<polygon points="{x:.1f},{y - 5:.1f} {x + 8:.1f},{y:.1f} {x:.1f},{y + 5:.1f}" '
        f'fill="{color}" fill-opacity="0.4" stroke="{color}" stroke-width="1"/>'
    )


__all__ = [
    "COLORS",
    "shaft_line",
    "bearing_circle",
    "machine_body",
    "cooling_fins",
    "cylinder_horizontal",
    "cylinder_vertical",
    "label_top",
    "side_label",
    "crankshaft_box",
    "axial_marker",
]
