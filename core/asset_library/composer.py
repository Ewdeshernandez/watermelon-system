"""
core.asset_library.composer
===========================

Compone el tren acoplado completo (driver + coupling + driven) en un
único SVG con:

  - Iconos del catalog para cada parte
  - Acople central
  - Sensor dots overlayados en los anchor points correctos
  - Status colors por sensor + label

API:

    full_svg = compose_train(
        driver_key="gas_turbine_aero",
        driven_key="generator_synchronous",
        driver_label="GE LM6000",
        driven_label="Brush BDAX 7-290ER",
        coupling="flexible",
        sensors_with_status=[
            {"label": "1Y_V", "anchor": "TRF", "side": "driver", "status": "Normal", "value": "0.78"},
            ...
        ],
    )
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from core.asset_library import get_icon
from core.asset_library.couplings import coupling_flexible, coupling_rigid


SEVERITY_COLORS = {
    "Danger":    "#ef4444",
    "Alarma":    "#f59e0b",
    "Normal":    "#22c55e",
    "Sin Norma": "#94a3b8",
    "No Data":   "#64748b",
}

SEVERITY_ANIM = {
    "Danger":  '<animate attributeName="r" values="2.6;3.6;2.6" dur="1.2s" repeatCount="indefinite"/>',
    "Alarma":  '<animate attributeName="opacity" values="1;0.55;1" dur="1.6s" repeatCount="indefinite"/>',
}


def _render_sensor_dot(
    cx: float,
    cy: float,
    label: str,
    value: str = "",
    unit: str = "",
    status: str = "Normal",
    title: str = "",
    text_above: bool = True,
) -> str:
    """
    SVG de un sensor dot con texto 'LABEL VALOR' inline. Por default el
    texto va arriba del dot; si `text_above=False` el texto va debajo
    (caso típico: probetas X/Y ortogonales API 670 — la Y arriba, la X
    abajo, ambas en el mismo cojinete pero textos no apilados).
    """
    color = SEVERITY_COLORS.get(status, "#64748b")
    anim = SEVERITY_ANIM.get(status, "")
    inline = f"{label} {value}".strip() if value and value != "—" else label

    if text_above:
        text_y = cy - 12
        unit_y = cy + 18
    else:
        text_y = cy + 16
        unit_y = cy + 26  # debajo del label, si hay unidad

    parts = [
        f'<g><title>{title or label}</title>',
        # Halo
        f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="9" fill="{color}" '
        f'fill-opacity="0.18" stroke="{color}" stroke-width="0.8" stroke-opacity="0.55"/>',
        # Dot principal
        f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5" fill="{color}" stroke="white" '
        f'stroke-width="1.8">{anim}</circle>',
        # Texto inline (arriba o abajo según text_above)
        f'<text x="{cx:.1f}" y="{text_y:.1f}" text-anchor="middle" '
        f'font-size="9" font-weight="800" font-family="SF Mono, Menlo, monospace" '
        f'fill="{color}" letter-spacing="-0.04" '
        f'style="paint-order:stroke;stroke:white;stroke-width:2.5;stroke-linejoin:round;">'
        f'{inline}</text>',
    ]
    if unit and value and value != "—":
        parts.append(
            f'<text x="{cx:.1f}" y="{unit_y:.1f}" text-anchor="middle" '
            f'font-size="7" font-weight="600" font-family="SF Mono, monospace" '
            f'fill="#475569" '
            f'style="paint-order:stroke;stroke:white;stroke-width:2;stroke-linejoin:round;">'
            f'{unit}</text>'
        )
    parts.append('</g>')
    return "".join(parts)


def compose_train(
    driver_key: str,
    driven_key: str,
    driver_label: str = "",
    driven_label: str = "",
    coupling: str = "flexible",
    sensors_with_status: Optional[List[Dict[str, Any]]] = None,
    bg_color: str = "#ffffff",
) -> str:
    """
    Compone un SVG con el tren acoplado completo.

    sensors_with_status: lista de sensores a overlayar. Cada uno:
        {
            "label": "1Y_V",         # texto a mostrar
            "side":  "driver",        # 'driver' | 'driven' | 'coupling'
            "anchor": "DE",           # key del anchors_dict del icono
            "status": "Normal",       # severidad
            "value":  "0.78",         # valor numérico (string)
            "unit":   "in/s pk",      # opcional
            "title":  "tooltip text", # opcional, default = label
        }

    Devuelve un SVG completo (no <g>) listo para insertar en HTML.
    """
    if sensors_with_status is None:
        sensors_with_status = []

    # Generar driver
    driver_svg, driver_anchors = get_icon(driver_key, label=driver_label, x_offset=0, y_offset=0)
    driver_w = driver_anchors.get("viewbox_w", 320)

    # Coupling
    coupling_w = 80 if coupling == "flexible" else 60
    coupling_x = driver_w
    if coupling == "rigid":
        coupling_svg, coupling_anchors = coupling_rigid(
            x_offset=coupling_x, y_offset=0, width=coupling_w,
        )
    else:
        coupling_svg, coupling_anchors = coupling_flexible(
            x_offset=coupling_x, y_offset=0, width=coupling_w,
        )

    # Driven
    driven_x = coupling_x + coupling_w
    driven_svg, driven_anchors = get_icon(
        driven_key, label=driven_label, x_offset=driven_x, y_offset=0,
    )
    driven_w = driven_anchors.get("viewbox_w", 320)

    total_w = driver_w + coupling_w + driven_w
    total_h = max(
        driver_anchors.get("viewbox_h", 200),
        driven_anchors.get("viewbox_h", 200),
        200,
    )

    # Sensor dots overlayados.
    # Si dos sensores caen en el mismo (side, anchor) — caso típico de
    # probetas de proximidad ortogonales X/Y a 90° en API 670, o pares
    # acelerómetro+velocímetro en el mismo cojinete — los desplazamos
    # horizontalmente para que ambos labels queden legibles.
    counts: Dict[Tuple[str, str], int] = {}
    for s in sensors_with_status:
        key = (s.get("side", "driver"), s.get("anchor", "DE"))
        counts[key] = counts.get(key, 0) + 1
    seen: Dict[Tuple[str, str], int] = {}

    dots_svg_parts: List[str] = []
    for s in sensors_with_status:
        side = s.get("side", "driver")
        anchor_name = s.get("anchor", "DE")
        if side == "driver":
            anchors = driver_anchors
        elif side == "driven":
            anchors = driven_anchors
        elif side == "coupling":
            anchors = coupling_anchors
        else:
            anchors = driver_anchors

        pos = anchors.get(anchor_name)
        if not pos or not isinstance(pos, tuple):
            continue
        cx, cy = pos

        # Resolver layout cuando hay múltiples sensores en el mismo anchor.
        # Caso típico API 670: probetas proximity X/Y a 90° en el mismo
        # cojinete, o pares accel+velocímetro. Para que los labels NO se
        # apilen unos sobre otros:
        #   N=1 → tal cual (texto arriba)
        #   N=2 → primer sensor texto ARRIBA, segundo texto ABAJO,
        #         ambos compartiendo el mismo dot (sin offset horizontal).
        #   N>=3 → distribución horizontal con texto arriba (cascada).
        key = (side, anchor_name)
        n_total = counts[key]
        text_above = True
        if n_total == 2:
            idx = seen.get(key, 0)
            seen[key] = idx + 1
            text_above = (idx == 0)  # primer sensor arriba, segundo abajo
        elif n_total >= 3:
            idx = seen.get(key, 0)
            seen[key] = idx + 1
            spread = 22  # px de separación entre dots cuando hay 3+
            offset_x = (idx - (n_total - 1) / 2) * spread
            cx = cx + offset_x

        dots_svg_parts.append(
            _render_sensor_dot(
                cx=cx, cy=cy,
                label=s.get("label", ""),
                value=s.get("value", ""),
                unit=s.get("unit", ""),
                status=s.get("status", "Normal"),
                title=s.get("title", ""),
                text_above=text_above,
            )
        )

    full_svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {total_w:.0f} {total_h:.0f}" '
        f'style="background:{bg_color};width:100%;height:auto;display:block;">'
        f'{driver_svg}{coupling_svg}{driven_svg}'
        f'{"".join(dots_svg_parts)}'
        f'</svg>'
    )
    return full_svg


__all__ = ["compose_train"]
