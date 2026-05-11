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

import math
from typing import Any, Dict, List, Optional, Tuple

from core.asset_library import get_icon
from core.asset_library.couplings import (
    coupling_flexible,
    coupling_rigid,
    coupling_rigid_quill,
)


SEVERITY_COLORS = {
    # Ciclo 23.34 — bumped Normal del #22c55e (contrast 2.5:1 sobre blanco,
    # FAIL WCAG AA) a #15803d (forest green, contrast 4.7:1 PASS WCAG AA).
    # Resto de severidades ya cumplen contrast.
    "Danger":    "#dc2626",   # red-600 (era #ef4444 red-500)
    "Alarma":    "#b45309",   # amber-700 (era #f59e0b amber-500)
    "Normal":    "#15803d",   # green-700 (era #22c55e green-500)
    "Sin Norma": "#475569",   # slate-600 (era #94a3b8 slate-400)
    "No Data":   "#334155",   # slate-700 (era #64748b slate-500)
}

SEVERITY_ANIM = {
    "Danger":  '<animate attributeName="r" values="2.6;3.6;2.6" dur="1.2s" repeatCount="indefinite"/>',
    "Alarma":  '<animate attributeName="opacity" values="1;0.55;1" dur="1.6s" repeatCount="indefinite"/>',
}


def _render_threshold_bar(
    cx: float,
    cy: float,
    value: Optional[float],
    alarm: Optional[float],
    danger: Optional[float],
    color: str,
    width: float = 64,
    height: float = 7,
) -> str:
    """
    Barra horizontal continua con 3 zonas (verde/amarillo/rojo) sin
    bordes ni separadores visibles — solo color (Ciclo 23.35).
    Marker del valor actual: triangulito chiquito sobre la barra
    apuntando hacia abajo, sin halo agresivo.

    User feedback v3.31.42 → "dejar una sola barra con colores sin
    dividirlas con líneas, solo con colores sin marco".
    """
    if alarm is None or danger is None or alarm <= 0 or danger <= 0 or alarm >= danger:
        return ""
    # Escala: 0 → 1.2 * danger (deja un poco de espacio post-danger)
    scale_max = danger * 1.2
    norm_w = width * (alarm / scale_max)
    alarm_w = width * ((danger - alarm) / scale_max)
    danger_w = max(0.0, width - norm_w - alarm_w)
    x0 = cx - width / 2
    y0 = cy - height / 2

    parts = [
        # Zonas continuas SIN strokes — solo colores. Las esquinas
        # redondeadas solo en los extremos (izq de verde, der de rojo)
        # para feel pill cohesivo.
        # Zona normal (verde) — esquina izq redondeada
        f'<path d="M {x0+3:.1f} {y0:.1f} '
        f'L {x0+norm_w:.1f} {y0:.1f} '
        f'L {x0+norm_w:.1f} {y0+height:.1f} '
        f'L {x0+3:.1f} {y0+height:.1f} '
        f'Q {x0:.1f} {y0+height:.1f} {x0:.1f} {y0+height-3:.1f} '
        f'L {x0:.1f} {y0+3:.1f} '
        f'Q {x0:.1f} {y0:.1f} {x0+3:.1f} {y0:.1f} Z" '
        f'fill="#22c55e"/>',
        # Zona alarma (amber) — sin esquinas
        f'<rect x="{x0+norm_w:.1f}" y="{y0:.1f}" width="{alarm_w:.1f}" '
        f'height="{height:.1f}" fill="#f59e0b"/>',
        # Zona danger (red) — esquina der redondeada
        f'<path d="M {x0+norm_w+alarm_w:.1f} {y0:.1f} '
        f'L {x0+width-3:.1f} {y0:.1f} '
        f'Q {x0+width:.1f} {y0:.1f} {x0+width:.1f} {y0+3:.1f} '
        f'L {x0+width:.1f} {y0+height-3:.1f} '
        f'Q {x0+width:.1f} {y0+height:.1f} {x0+width-3:.1f} {y0+height:.1f} '
        f'L {x0+norm_w+alarm_w:.1f} {y0+height:.1f} Z" '
        f'fill="#ef4444"/>',
    ]

    # Marker — triangulito apuntando hacia abajo desde arriba de la barra.
    # Mucho más sutil que las dos líneas con halo de la versión anterior.
    if value is not None and value >= 0:
        marker_pos = min(width, max(0.0, width * (value / scale_max)))
        mx = x0 + marker_pos
        parts.append(
            f'<polygon points="'
            f'{mx-3:.1f},{y0-4:.1f} '
            f'{mx+3:.1f},{y0-4:.1f} '
            f'{mx:.1f},{y0:.1f}" '
            f'fill="#0f172a"/>'
        )
    return "".join(parts)


def _render_sparkline(
    cx: float,
    cy: float,
    values: List[float],
    color: str,
    width: float = 64,
    height: float = 18,
    value_num: Optional[float] = None,
    danger: Optional[float] = None,
) -> str:
    """
    Onda sinusoidal data-driven (Ciclo 23.51) — la AMPLITUD de la onda
    refleja qué tan cerca está el valor del danger threshold. Un sensor
    en Normal (value bajo) muestra una onda casi plana. Un sensor cerca
    de Alarma muestra una onda mediana. Un sensor en Danger muestra una
    onda casi llenando el box.

    Esto es signature visual: el operador puede leer el estado del
    activo a 1 metro de distancia mirando SHAPES, no leyendo números.
    Diferenciador real vs System1/Bently/Emerson (que solo muestran
    barras o números planos).

    Si no hay value_num/danger útiles, fallback a onda mediana (50%).
    """
    if not values or len(values) < 2:
        return ""
    # Amplitud data-driven
    max_amplitude = (height - 6) / 2
    if value_num is not None and danger and danger > 0:
        # ratio 0..1.2 (post-danger sigue creciendo levemente para visual feedback)
        ratio = min(1.2, max(0.18, value_num / danger))
    else:
        ratio = 0.5  # fallback decorativo
    amplitude = max_amplitude * ratio
    # Frecuencia: constante por ahora. Future: ratio = RPM / 3600.
    n_cycles = 3
    n_points = 36
    y_center = height / 2
    pts = []
    for i in range(n_points):
        t = i / (n_points - 1)
        x = t * (width - 6) + 3
        y = y_center - amplitude * math.sin(2 * math.pi * n_cycles * t)
        pts.append(f"{x:.2f},{y:.2f}")
    pts_str = " ".join(pts)
    x0 = cx - width / 2
    y0 = cy - height / 2
    return (
        f'<g transform="translate({x0:.1f},{y0:.1f})">'
        f'<rect width="{width:.1f}" height="{height:.1f}" rx="9" '
        f'fill="white" stroke="{color}" stroke-width="1" opacity="0.92"/>'
        f'<polyline points="{pts_str}" fill="none" stroke="{color}" '
        f'stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>'
        f'</g>'
    )


def _render_sensor_dot(
    cx: float,
    cy: float,
    label: str,
    value: str = "",
    unit: str = "",
    status: str = "Normal",
    title: str = "",
    text_above: bool = True,
    spark_values: Optional[List[float]] = None,
    alarm: Optional[float] = None,
    danger: Optional[float] = None,
    value_num: Optional[float] = None,
    link: Optional[str] = None,
    is_stale: bool = False,
) -> str:
    """
    SVG de un sensor dot con:
      - halo + dot principal con animation por severidad
      - label + valor + unidad con stroke blanco para legibilidad
      - threshold bar (verde/amarillo/rojo + marker) — Ciclo 23.23
      - sparkline mini de tendencia — Ciclo 23.23

    text_above: si True, sparkline+threshold+texto arriba del dot;
    si False, debajo (caso típico probetas X/Y ortogonales API 670).

    spark_values: últimas N lecturas para la sparkline (None = no se renderiza).
    alarm/danger/value_num: para la threshold bar (None = no se renderiza).
    """
    color = SEVERITY_COLORS.get(status, "#64748b")
    anim = SEVERITY_ANIM.get(status, "")
    has_value = bool(value) and value != "—"
    inline = f"{label} {value}".strip() if has_value else label

    # Layout vertical extendido (Ciclo 23.52) — fix CRITICO:
    # ANCHOR del sensor está en CENTRO del body (cy=100 in icon coords).
    # Body radius es 48-52 → body top edge en cy-48 a cy-52.
    # ANTES: text_y=cy-45 caía DENTRO del body (45 < 48). Por eso "tocaba".
    # AHORA: text_y=cy-75, unit_y=cy-60 → ambos por encima de cy-52
    # con buffer de 8-10px. Equipo queda limpio sin etiquetas encima.
    # Mismo patrón mirror para text_above=False.
    has_spark = spark_values is not None and len(spark_values) >= 2
    has_threshold = (
        alarm is not None and danger is not None
        and alarm > 0 and danger > alarm
    )
    if text_above:
        text_y = cy - 75 if (has_spark or has_threshold) else cy - 16
        unit_y = cy - 60 if (has_spark or has_threshold) else cy - 28
        threshold_y = cy - 90
        spark_y = cy - 110
    else:
        # Ciclo 23.53 — bottom positions bajadas 15px más para
        # simetría con top (ambos lados con 25px clearance del body
        # edge). User feedback: "faltaria bajar un poco mas las de
        # abajo si las ves".
        text_y = cy + 75 if (has_spark or has_threshold) else cy + 20
        unit_y = cy + 90 if (has_spark or has_threshold) else cy + 32
        threshold_y = cy + 105
        spark_y = cy + 125

    # Click-to-drill (Ciclo 23.26) — wrapper SVG <a> con href absoluto.
    # Iteraciones previas:
    #   v3.31.31: <a href="?sensor=X"> — se rompía con base href de
    #             Streamlit, mandaba al usuario al home.
    #   v3.31.32: <g onclick="window.location.search=..."> — onclick
    #             quedaba sin disparar (Streamlit lo strip o el SVG no
    #             propaga el evento al window).
    # Esta iteración: SVG <a> con href absoluto "/Live_Monitoring?sensor=X".
    # El path absoluto sortea el base href; <a> es respetado por el
    # sanitizer; el browser navega normalmente y Streamlit detecta el
    # cambio de URL → rerun → query_params lee `sensor`.
    # Ciclo 23.31 — SVG read-only (drilldown via dropdown nativo Streamlit).
    # Ciclo 23.34 — agregado stale data signaling: si el sensor lleva más
    # de N segundos sin reportar, aplicamos filter wm-stale (grayscale +
    # opacity reducida) para que el operador vea de un vistazo que esa
    # lectura no es actual.
    if is_stale:
        open_tag = (
            f'<g filter="url(#wm-stale)">'
            f'<title>{title or label} — DATA STALE</title>'
        )
    else:
        open_tag = f'<g><title>{title or label}</title>'
    close_tag = '</g>'

    parts = [
        open_tag,
        # Halo (círculo translúcido alrededor del dot — efecto de glow)
        f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="10" fill="{color}" '
        f'fill-opacity="0.22" stroke="{color}" stroke-width="1" stroke-opacity="0.6"/>',
        # Dot principal con borde blanco grueso
        f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5.5" fill="{color}" stroke="white" '
        f'stroke-width="2">{anim}</circle>',
        # Texto LABEL + VALOR con halo blanco grueso para legibilidad
        f'<text x="{cx:.1f}" y="{text_y:.1f}" text-anchor="middle" '
        f'font-size="10" font-weight="800" font-family="SF Mono, Menlo, monospace" '
        f'fill="{color}" letter-spacing="-0.04" '
        f'style="paint-order:stroke;stroke:white;stroke-width:3.5;stroke-linejoin:round;">'
        f'{inline}</text>',
    ]
    if unit and has_value:
        # Estimar ancho de la pill según largo de la unidad
        pill_w = max(28, len(unit) * 4.2 + 8)
        pill_h = 9
        parts.append(
            # Pill background
            f'<rect x="{cx - pill_w / 2:.1f}" y="{unit_y - pill_h + 2:.1f}" '
            f'width="{pill_w:.1f}" height="{pill_h:.1f}" rx="4" '
            f'fill="white" stroke="{color}" stroke-width="0.8" stroke-opacity="0.6"/>'
        )
        parts.append(
            # Unidad text
            f'<text x="{cx:.1f}" y="{unit_y - 1:.1f}" text-anchor="middle" '
            f'font-size="7" font-weight="700" font-family="SF Mono, monospace" '
            f'fill="#475569" letter-spacing="0.03">{unit}</text>'
        )
    # Threshold bar (Ciclo 23.23) — contexto visual de qué tan cerca está
    # el valor de los setpoints alarm/danger.
    if has_threshold:
        parts.append(
            _render_threshold_bar(
                cx=cx, cy=threshold_y,
                value=value_num,
                alarm=alarm, danger=danger,
                color=color,
            )
        )
    # Sparkline data-driven (Ciclo 23.51) — amplitud de la onda
    # sinusoidal proporcional al value_num/danger ratio. Operador lee
    # SHAPES en vez de números a 1 metro de distancia.
    if has_spark:
        parts.append(
            _render_sparkline(
                cx=cx, cy=spark_y,
                values=spark_values,
                color=color,
                value_num=value_num,
                danger=danger,
            )
        )
    parts.append(close_tag)
    return "".join(parts)


def compose_train(
    driver_key: str,
    driven_key: str,
    driver_label: str = "",
    driven_label: str = "",
    coupling: str = "flexible",
    sensors_with_status: Optional[List[Dict[str, Any]]] = None,
    bg_color: str = "#ffffff",
    return_overlays: bool = False,
) -> Any:
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
            "spark_values": [...],    # opcional, últimas N lecturas para sparkline
            "alarm": 3.5,             # opcional, setpoint alarm para threshold bar
            "danger": 5.0,            # opcional, setpoint danger para threshold bar
            "value_num": 2.33,        # opcional, valor numérico para marker en threshold bar
        }

    Devuelve un SVG completo (no <g>) listo para insertar en HTML.
    """
    if sensors_with_status is None:
        sensors_with_status = []

    # Generar driver — Ciclo 23.28: usamos label=" " (single space) para
    # suprimir el label interno SIN disparar el fallback de get_icon
    # (que con "" mete default_label del catálogo). El texto " " produce
    # un <text> con un espacio — invisible. Renderizamos el título limpio
    # externamente en titles_layer a y=100 (entre sparkline bottom y
    # body top).
    driver_svg, driver_anchors = get_icon(driver_key, label=" ", x_offset=0, y_offset=0)
    driver_w = driver_anchors.get("viewbox_w", 320)

    # Coupling — Ciclo 23.23: detectar tren aero-derivative (gas_turbine_aero)
    # con coupling rigid → usar quill-shaft style (LM6000 real-world).
    # User-specified explicit "rigid_quill" también soportado.
    is_aero_rigid = (
        driver_key == "gas_turbine_aero"
        and coupling in ("rigid", "rigid_quill")
    )
    if coupling == "rigid_quill" or is_aero_rigid:
        coupling_w = 70
    elif coupling == "flexible":
        coupling_w = 80
    else:
        coupling_w = 60
    coupling_x = driver_w
    if coupling == "rigid_quill" or is_aero_rigid:
        coupling_svg, coupling_anchors = coupling_rigid_quill(
            x_offset=coupling_x, y_offset=0, width=coupling_w,
        )
    elif coupling == "rigid":
        coupling_svg, coupling_anchors = coupling_rigid(
            x_offset=coupling_x, y_offset=0, width=coupling_w,
        )
    else:
        coupling_svg, coupling_anchors = coupling_flexible(
            x_offset=coupling_x, y_offset=0, width=coupling_w,
        )

    # Driven — mismo patrón: " " para suprimir interno sin fallback.
    driven_x = coupling_x + coupling_w
    driven_svg, driven_anchors = get_icon(
        driven_key, label=" ", x_offset=driven_x, y_offset=0,
    )
    driven_w = driven_anchors.get("viewbox_w", 320)

    total_w = driver_w + coupling_w + driven_w
    base_h = max(
        driver_anchors.get("viewbox_h", 200),
        driven_anchors.get("viewbox_h", 200),
        200,
    )

    # Padding vertical (Ciclo 23.24) — si los sensores traen sparkline o
    # threshold bar, las decoraciones extienden ~72px por encima/debajo del
    # dot. Sin padding las cards quedan tocando el equipo o salen del
    # viewBox. Calculamos pad solo si hace falta.
    needs_decoration_pad = any(
        (s.get("spark_values") and len(s.get("spark_values", []))) >= 2
        or (
            s.get("alarm") is not None
            and s.get("danger") is not None
            and (s.get("alarm") or 0) > 0
            and (s.get("danger") or 0) > (s.get("alarm") or 0)
        )
        for s in sensors_with_status
    )
    # Ciclo 23.53: bumped 130 → 140 para acomodar bottom decorations
    # bajadas (sparkline bottom edge ahora a cy+134, necesita pad de 140).
    vert_pad = 140 if needs_decoration_pad else 0
    total_h = base_h + 2 * vert_pad

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
    # Overlays (Ciclo 23.32): por cada sensor calculamos el bbox (en %
    # del viewBox) de la sparkline para que el caller pueda dibujar
    # `<a>` HTML transparentes encima → clickeable real.
    overlays: List[Dict[str, Any]] = []
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
        # Aplicar padding vertical: anchors vienen del coord system del icono
        # (0..base_h). El equipo se shifta abajo por vert_pad, así que los
        # dots también deben shiftarse para quedar en su anchor físico.
        cy = cy + vert_pad

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
                spark_values=s.get("spark_values"),
                alarm=s.get("alarm"),
                danger=s.get("danger"),
                value_num=s.get("value_num"),
                link=s.get("link"),
                is_stale=bool(s.get("is_stale")),
            )
        )
        # Calcular bbox del overlay clickeable. Cubrimos toda la zona
        # "decorada" del sensor: sparkline + threshold + texto + dot.
        # Eso es más ergonómico que solo el rect de la sparkline.
        # Range vertical:
        #   text_above=True:  spark top @ cy-101, dot bottom @ cy+8 → range cy-101..cy+10
        #   text_above=False: dot top @ cy-10, spark bottom @ cy+103
        if text_above:
            y_min = cy - 101
            y_max = cy + 10
        else:
            y_min = cy - 10
            y_max = cy + 103
        x_min = cx - 32
        x_max = cx + 32
        sensor_lbl_for_link = s.get("sensor_label") or s.get("label") or ""
        overlays.append({
            "sensor_label": sensor_lbl_for_link,
            "x_pct": (x_min / total_w) * 100 if total_w else 0,
            "y_pct": (y_min / total_h) * 100 if total_h else 0,
            "w_pct": ((x_max - x_min) / total_w) * 100 if total_w else 0,
            "h_pct": ((y_max - y_min) / total_h) * 100 if total_h else 0,
        })

    # Defs globales (Ciclo 23.23) — drop-shadow filter aplicable a equipos
    # para dar profundidad. Se aplica via filter="url(#wm-shadow)" en la
    # capa de driver+coupling+driven. ID con prefijo wm- para evitar
    # colisiones con defs internos de cada icono.
    defs_block = (
        '<defs>'
        '<filter id="wm-shadow" x="-10%" y="-10%" width="120%" height="130%">'
        '<feDropShadow dx="0" dy="3" stdDeviation="2.5" flood-opacity="0.18"/>'
        '</filter>'
        # Ciclo 23.34 — filter para sensores stale (data vieja). Convierte
        # colores a grayscale via feColorMatrix + reduce opacity. Aplicado
        # en _render_sensor_dot cuando is_stale=True.
        '<filter id="wm-stale">'
        '<feColorMatrix type="matrix" values="'
        '0.33 0.33 0.33 0 0 '
        '0.33 0.33 0.33 0 0 '
        '0.33 0.33 0.33 0 0 '
        '0 0 0 0.45 0"/>'
        '</filter>'
        '<pattern id="wm-grid" width="22" height="22" patternUnits="userSpaceOnUse">'
        '<path d="M 22 0 L 0 0 0 22" fill="none" stroke="#e2e8f0" stroke-width="0.5"/>'
        '</pattern>'
        '</defs>'
    )
    # Background grid sutil — feel de plano técnico
    bg_layer = (
        f'<rect x="0" y="0" width="{total_w:.0f}" height="{total_h:.0f}" '
        f'fill="url(#wm-grid)" opacity="0.4"/>'
    )
    # Equipo + coupling envueltos en filter de sombra + translate para
    # vertical padding (Ciclo 23.24). El padding empuja el equipo hacia
    # abajo dentro del viewBox para dejar headroom para sparklines.
    equipment_layer = (
        f'<g filter="url(#wm-shadow)" transform="translate(0,{vert_pad})">'
        f'{driver_svg}{coupling_svg}{driven_svg}'
        f'</g>'
    )

    # Titles externos (Ciclo 23.28) — re-introducidos en y=100, posición
    # SAFE entre el bottom de las sparklines (~y=86) y el top del body
    # del equipo (~y=115). Antes los probé en y=22 (muy arriba, lejos
    # del equipo) y eliminados (chocaba con el fallback de get_icon).
    # Ahora el icono se le pasa label=" " para no duplicar.
    titles_layer = ""
    if driver_label:
        cx = driver_w / 2
        titles_layer += (
            f'<text x="{cx:.1f}" y="100" text-anchor="middle" '
            f'font-size="14" font-weight="800" fill="#1e3a8a" '
            f'font-family="-apple-system, Segoe UI, Roboto, sans-serif">'
            f'{driver_label}</text>'
        )
    if driven_label:
        cx = driver_w + coupling_w + driven_w / 2
        titles_layer += (
            f'<text x="{cx:.1f}" y="100" text-anchor="middle" '
            f'font-size="14" font-weight="800" fill="#14532d" '
            f'font-family="-apple-system, Segoe UI, Roboto, sans-serif">'
            f'{driven_label}</text>'
        )

    full_svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'viewBox="0 0 {total_w:.0f} {total_h:.0f}" '
        f'style="background:{bg_color};width:100%;height:auto;display:block;">'
        f'{defs_block}'
        f'{bg_layer}'
        f'{titles_layer}'
        f'{equipment_layer}'
        f'{"".join(dots_svg_parts)}'
        f'</svg>'
    )
    if return_overlays:
        return full_svg, overlays
    return full_svg


__all__ = ["compose_train"]
