"""
core.sensor_diagram
===================

Renderizado visual del Sensor Map (Ciclo 14c.2).

Genera un diagrama del tren acoplado con dos componentes:

  1. **Vista lateral del tren**: bloques DRIVER y DRIVEN con cojinetes
     numerados según convención API 670 / ISO 20816-1 (driver=1,2 →
     driven=3,4 →...). Keyphasor visible en coupling.

  2. **Vista polar por plano**: para cada cojinete, un círculo dividido
     en hemisferio L y R (mirando desde extremo del driver hacia el
     driven, con 0° arriba), con las sondas marcadas en sus ángulos
     físicos. Facilita verificar de un vistazo si el setup tiene las
     sondas X-Y a 45°/45° (estándar API 670) o algún offset distinto.

El render se devuelve como **bytes PNG** para mostrar en Streamlit con
``st.image()`` o embed en el PDF de Reports.
"""

from __future__ import annotations

import math
from io import BytesIO
from typing import Any, Dict, List, Optional


# Paleta sobria coherente con el Reports Watermelon
_COLOR_DRIVER = "#3b82f6"     # azul
_COLOR_DRIVEN = "#10b981"     # verde
_COLOR_BEARING = "#475569"    # slate
_COLOR_KEYPHASOR = "#f59e0b"  # ámbar
_COLOR_PROXIMITY = "#8b5cf6"  # violeta
_COLOR_VELOCITY = "#06b6d4"   # cian
_COLOR_ACCELEROMETER = "#ef4444"  # rojo
_COLOR_BG = "#ffffff"
_COLOR_GRID = "#e2e8f0"
_COLOR_TEXT = "#0f172a"

# Ciclo 15.1 — paleta de severidad para Machine Map (coherente con
# las cintas de severidad de los expanders Cat IV).
_COLOR_SEVERITY = {
    "Normal": "#16a34a",                # verde — CONDICIÓN ACEPTABLE
    "CONDICIÓN ACEPTABLE": "#16a34a",
    "Alarm": "#f59e0b",                 # ámbar — ATENCIÓN
    "ATENCIÓN": "#f59e0b",
    "Danger": "#dc2626",                # rojo — ACCIÓN REQUERIDA / CRÍTICA
    "ACCIÓN REQUERIDA": "#dc2626",
    "CRÍTICA": "#dc2626",
    "No Data": "#94a3b8",               # gris — sin CSV matched
    "": "#94a3b8",
}


def _color_for_sensor_type(sensor_type: str) -> str:
    t = (sensor_type or "").lower()
    if t == "proximity":
        return _COLOR_PROXIMITY
    if t == "velocity":
        return _COLOR_VELOCITY
    if t == "accelerometer":
        return _COLOR_ACCELEROMETER
    if t == "keyphasor":
        return _COLOR_KEYPHASOR
    return _COLOR_BEARING


def _marker_for_sensor_type(sensor_type: str) -> str:
    t = (sensor_type or "").lower()
    if t == "proximity":
        return "o"
    if t == "velocity":
        return "s"
    if t == "accelerometer":
        return "^"
    if t == "keyphasor":
        return "*"
    return "."


# ============================================================
# Plane label normalization (Ciclo 15.1.3)
# ------------------------------------------------------------
# Cuando varios sensores comparten un mismo plano fisico (ej.
# velocidad + acelerometro en el TRF de una turbina), cada uno
# puede traer un plane_label distinto del estilo "TRF Vel" /
# "TRF Accel" / "TRF Prox". Para el diagrama queremos mostrar
# UNA etiqueta de plano (la ubicacion fisica), sin que tape
# que hay varios tipos de sensor en ese plano.
#
# Estrategia:
#   1. Para cada plano, juntar todas las plane_labels de sus
#      sensores.
#   2. Quitar tokens del tipo de sensor (Vel, Accel, Prox y
#      sus variantes en español).
#   3. Si las labels limpias coinciden, usar esa.
#   4. Si difieren, usar la mas corta (suele ser la
#      ubicacion pura).
# Por separado el diagrama agrega chips circulares de color
# bajo el numero de cojinete con cada tipo de sensor presente
# (violeta=prox, cian=vel, rojo=accel) — asi el ingeniero ve
# de un vistazo que esta instrumentado en cada plano.
# ============================================================
_SENSOR_TYPE_TOKENS_TO_STRIP = (
    "accelerometer", "acelerometro", "acelerómetro",
    "accel", "acel", "acell", "ace",
    "velocity", "velocidad", "velocimetro", "velocímetro",
    "vel", "vt",
    "proximity", "prox", "proxim", "proxí",
    "displacement", "desplazamiento", "despl",
)


def _normalize_plane_label(raw: str) -> str:
    """Quita tokens de tipo de sensor y normaliza espacios."""
    txt = (raw or "").strip()
    if not txt:
        return ""
    # Comparar tokens en lowercase pero conservar el casing original.
    parts = txt.split()
    keep = []
    for tok in parts:
        if tok.lower().rstrip(".:,") in _SENSOR_TYPE_TOKENS_TO_STRIP:
            continue
        keep.append(tok)
    cleaned = " ".join(keep).strip(" -·_/")
    return cleaned or txt


def _plane_display_label(sensors_in_plane: List[Dict[str, Any]]) -> str:
    """Etiqueta de plano consolidada para el diagrama."""
    if not sensors_in_plane:
        return ""
    raw_labels = [str(s.get("plane_label", "") or "") for s in sensors_in_plane]
    cleaned = sorted({_normalize_plane_label(r) for r in raw_labels if r}, key=len)
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    # Si hay varias limpias distintas (raro), tomar la mas corta —
    # suele ser la ubicacion pura "TRF" vs "TRF C".
    return cleaned[0]


def _sensor_types_in_plane(sensors_in_plane: List[Dict[str, Any]]) -> List[str]:
    """Lista ordenada de tipos de sensor presentes en el plano."""
    order = ["proximity", "velocity", "accelerometer", "keyphasor"]
    present = {str(s.get("sensor_type", "") or "").lower() for s in sensors_in_plane}
    return [t for t in order if t in present]


def _worst_status_for_plane(
    sensors_in_plane: List[Dict[str, Any]],
    severity_by_label: Optional[Dict[str, str]],
) -> str:
    """
    Devuelve el status peor entre los sensores del plano:
    Danger > Alarm > Normal > No Data. Si severity_by_label es None
    devuelve "" (sin color de severidad).
    """
    if not severity_by_label:
        return ""
    rank = {"Danger": 3, "Alarm": 2, "Normal": 1, "No Data": 0, "": 0}
    worst = ""
    worst_rank = -1
    # Import diferido para evitar ciclo
    from core.sensor_map import sensor_label as _slabel
    for s in sensors_in_plane:
        lbl = _slabel(s)
        st = severity_by_label.get(lbl, "No Data")
        r = rank.get(st, 0)
        if r > worst_rank:
            worst_rank = r
            worst = st
    return worst


def render_sensor_map_diagram(
    sensors: List[Dict[str, Any]],
    *,
    train_label: str = "",
    driver_label: str = "Driver",
    driven_label: str = "Driven",
    figure_width_in: float = 12.0,
    severity_by_label: Optional[Dict[str, str]] = None,
    compact: bool = False,
) -> Optional[bytes]:
    """
    Devuelve PNG bytes con el diagrama del Sensor Map.

    Args:
        sensors: lista de sensores (dicts).
        train_label: subtítulo (ej. "Turbogenerador GE LM6000 + Brush 54 MW").
        driver_label / driven_label: etiquetas de las dos máquinas.
        figure_width_in: ancho del figure en pulgadas.
        severity_by_label: (Ciclo 15.1) dict opcional mapeando sensor_label
            → status ("Normal" / "Alarm" / "Danger" / "No Data"). Cuando
            se provee, los markers se colorean por severidad. Cuando no,
            usan el color por tipo de sensor (modo "configuración").
        compact: (Ciclo 15.1.1) si True, renderiza SOLO la vista lateral
            del tren, con cada cojinete coloreado según el peor status
            de los sensores en ese plano (worst-of). Pensado como banner
            arriba del Tabular List. Sin polar por plano. Si
            severity_by_label es None en compact, los cojinetes salen
            todos neutros.

    Returns:
        Bytes PNG o None si matplotlib no está disponible.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except Exception:
        return None

    if not sensors:
        return None

    # Separar sensores por máquina (driver vs driven). Convención: planos 1-N
    # del driver vienen primero; planos siguientes son del driven. Detectamos
    # el "corte" por max plane que tiene el driver según plane_labels.
    driver_planes = sorted({
        int(s.get("plane", 0))
        for s in sensors
        if s.get("plane", 0) > 0 and "driver" in str(s.get("plane_label", "")).lower()
    })
    driven_planes = sorted({
        int(s.get("plane", 0))
        for s in sensors
        if s.get("plane", 0) > 0 and "driven" in str(s.get("plane_label", "")).lower()
    })
    # Fallback: si plane_labels no indican driver/driven, partir por mediana.
    # Tambien cubrimos el caso real (Ciclo 15.1.3) donde solo los sensores
    # del driven traen "Driven NDE/DE" y los del driver traen labels como
    # "TRF Accel"/"CRF Vel" (sin token "driver"). En ese caso driver_planes
    # arrancaba vacio y los cojinetes del driver no se dibujaban.
    all_planes = sorted({int(s.get("plane", 0)) for s in sensors if s.get("plane", 0) > 0})
    if not driver_planes and not driven_planes and all_planes:
        mid = len(all_planes) // 2
        driver_planes = all_planes[:mid] if mid > 0 else all_planes
        driven_planes = all_planes[mid:] if mid > 0 else []
    elif not driver_planes and driven_planes and all_planes:
        # Driver = todos los planos que no son del driven
        driven_set = set(driven_planes)
        driver_planes = sorted([p for p in all_planes if p not in driven_set])
    elif driver_planes and not driven_planes and all_planes:
        driver_set = set(driver_planes)
        driven_planes = sorted([p for p in all_planes if p not in driver_set])

    keyphasor_sensors = [s for s in sensors if (s.get("sensor_type") or "").lower() == "keyphasor"]

    n_planes_total = len(driver_planes) + len(driven_planes)
    if n_planes_total == 0:
        return None

    # Layout: en modo full hay panel superior (tren) + panel inferior
    # (polar por plano). En modo compact solo hay tren, mas chico.
    if compact:
        fig = plt.figure(
            figsize=(figure_width_in, 2.6),
            facecolor=_COLOR_BG,
        )
        gs = fig.add_gridspec(1, 1)
        ax_top = fig.add_subplot(gs[0, 0])
    else:
        fig = plt.figure(
            figsize=(figure_width_in, 5.5 + 1.5 * ((n_planes_total + 3) // 4)),
            facecolor=_COLOR_BG,
        )
        gs = fig.add_gridspec(
            2, max(n_planes_total, 4),
            height_ratios=[1.5, 2.0],
            hspace=0.45, wspace=0.35,
        )

        # ========================================================
        # Panel superior: vista lateral del tren
        # ========================================================
        ax_top = fig.add_subplot(gs[0, :])
    ax_top.set_xlim(0, 10)
    ax_top.set_ylim(0, 4)
    ax_top.set_aspect("equal")
    ax_top.axis("off")

    # ============================================================
    # Ciclo 15.1.3 — silueta turbomachinery en lugar de cajas planas
    # Driver = turbina aero-derivativa (cono que se ensancha hacia
    # el inlet y estrecha hacia el coupling, con anillos de stages).
    # Driven = generador (cilindro con end shields y vanes radiales).
    # Coupling = disco flexible con tornilleria.
    # ============================================================
    n_drv = max(1, len(driver_planes))
    n_dvn = max(1, len(driven_planes))
    drv_w = 2.6 + 0.55 * (n_drv - 1)
    dvn_w = 2.6 + 0.55 * (n_dvn - 1)
    coupling_w = 0.55
    total_w = drv_w + coupling_w + dvn_w
    x_start = 5.0 - total_w / 2.0  # centrar
    dvn_x = x_start + drv_w + coupling_w

    # Linea de centro del rotor (eje del tren). En el lateral los cojinetes
    # se posan EN esta linea — es la geometria correcta de un tren acoplado.
    rotor_y = 2.0
    ax_top.plot(
        [x_start - 0.35, dvn_x + dvn_w + 0.35], [rotor_y, rotor_y],
        color="#0f172a", linewidth=2.2, zorder=2,
    )

    # ----- Driver: silueta de turbina -----
    # Polygon con un perfil tipo aero-turbine: inlet ancho a la izquierda,
    # carcaza de combustión, anillos de turbina HP/LP, exhaust hacia el
    # coupling. Es generico (no replica un modelo especifico).
    drv_left = x_start
    drv_right = x_start + drv_w
    inlet_h = 1.25       # alto del inlet (lado outboard)
    body_h = 1.05        # alto principal del gas generator
    exhaust_h = 0.85     # alto cerca del coupling
    drv_top = rotor_y + inlet_h
    drv_btm = rotor_y - inlet_h

    # Vertices definidos en sentido horario, simetricos sobre rotor_y
    inlet_x = drv_left
    body_start_x = drv_left + 0.45
    body_end_x = drv_right - 0.45
    drv_polygon = [
        (inlet_x,         rotor_y + inlet_h),
        (inlet_x + 0.18,  rotor_y + inlet_h),
        (body_start_x,    rotor_y + body_h),
        (body_end_x,      rotor_y + body_h * 0.85),
        (drv_right,       rotor_y + exhaust_h),
        (drv_right,       rotor_y - exhaust_h),
        (body_end_x,      rotor_y - body_h * 0.85),
        (body_start_x,    rotor_y - body_h),
        (inlet_x + 0.18,  rotor_y - inlet_h),
        (inlet_x,         rotor_y - inlet_h),
    ]
    drv_patch = mpatches.Polygon(
        drv_polygon, closed=True,
        facecolor=_COLOR_DRIVER, alpha=0.16,
        edgecolor=_COLOR_DRIVER, linewidth=1.6, zorder=1,
    )
    ax_top.add_patch(drv_patch)

    # Anillo de inlet (vanes guías) — barras verticales finas
    for _vx in (inlet_x + 0.05, inlet_x + 0.10, inlet_x + 0.15):
        ax_top.plot([_vx, _vx], [rotor_y - inlet_h * 0.85, rotor_y + inlet_h * 0.85],
                    color=_COLOR_DRIVER, linewidth=0.8, alpha=0.55, zorder=2)

    # Stage rings dentro del cuerpo: 2-3 lineas verticales sutiles
    n_stages = 3
    for k in range(1, n_stages + 1):
        sx = body_start_x + (body_end_x - body_start_x) * k / (n_stages + 1)
        # alto local interpolado
        frac = (sx - body_start_x) / max(1e-6, body_end_x - body_start_x)
        local_h = body_h - (body_h - body_h * 0.85) * frac
        ax_top.plot([sx, sx], [rotor_y - local_h * 0.95, rotor_y + local_h * 0.95],
                    color=_COLOR_DRIVER, linewidth=0.7, alpha=0.45, zorder=2)

    # Cono de exhaust (transicion al coupling)
    ax_top.plot([body_end_x, drv_right], [rotor_y + body_h * 0.85, rotor_y + exhaust_h],
                color=_COLOR_DRIVER, linewidth=1.2, alpha=0.7, zorder=2)
    ax_top.plot([body_end_x, drv_right], [rotor_y - body_h * 0.85, rotor_y - exhaust_h],
                color=_COLOR_DRIVER, linewidth=1.2, alpha=0.7, zorder=2)

    # Etiqueta del driver arriba del cuerpo
    ax_top.text(x_start + drv_w / 2, rotor_y + inlet_h + 0.30, driver_label,
                fontsize=12, fontweight="bold", color=_COLOR_DRIVER,
                ha="center", va="center")

    # ----- Driven: silueta de generador / motor electrico -----
    # Cilindro con end shields a ambos lados. Vanes radiales en el lado
    # outboard (lejos del coupling) que sugieren el rotor de aire/cooling.
    dvn_left = dvn_x
    dvn_right = dvn_x + dvn_w
    gen_h = 1.10
    es_w = 0.20  # ancho de end shields

    # Cuerpo principal (cilindro)
    dvn_body = mpatches.Rectangle(
        (dvn_left + es_w, rotor_y - gen_h),
        dvn_w - 2 * es_w, 2 * gen_h,
        facecolor=_COLOR_DRIVEN, alpha=0.18,
        edgecolor=_COLOR_DRIVEN, linewidth=1.6, zorder=1,
    )
    ax_top.add_patch(dvn_body)

    # End shields (pequenos rectangulos achaflanados a ambos lados)
    for _esx in (dvn_left, dvn_right - es_w):
        es_patch = mpatches.FancyBboxPatch(
            (_esx, rotor_y - gen_h * 0.92), es_w, gen_h * 1.84,
            boxstyle="round,pad=0.01,rounding_size=0.05",
            facecolor=_COLOR_DRIVEN, alpha=0.28,
            edgecolor=_COLOR_DRIVEN, linewidth=1.4, zorder=2,
        )
        ax_top.add_patch(es_patch)

    # Vanes radiales en el lado outboard (derecho — lejos del coupling)
    fan_cx = dvn_right - es_w * 0.5
    for _ang_deg in range(0, 360, 30):
        _a = math.radians(_ang_deg)
        _x0 = fan_cx
        _y0 = rotor_y
        _x1 = fan_cx + 0.10 * math.cos(_a)
        _y1 = rotor_y + gen_h * 0.55 * math.sin(_a)
        ax_top.plot([_x0, _x1], [_y0, _y1], color=_COLOR_DRIVEN,
                    linewidth=0.8, alpha=0.55, zorder=3)

    # Etiqueta del driven
    ax_top.text(dvn_x + dvn_w / 2, rotor_y + gen_h + 0.42, driven_label,
                fontsize=12, fontweight="bold", color=_COLOR_DRIVEN,
                ha="center", va="center")

    # ----- Coupling: disco flexible entre las dos maquinas -----
    coup_x = x_start + drv_w + coupling_w / 2
    # Dos discos verticales con tornilleria (puntos pequeños)
    for _cx in (coup_x - 0.10, coup_x + 0.10):
        ax_top.add_patch(mpatches.Rectangle(
            (_cx - 0.04, rotor_y - 0.55), 0.08, 1.10,
            facecolor="#cbd5e1", edgecolor="#475569", linewidth=1.0, zorder=3,
        ))
    # Tornilleria (4 puntos por disco)
    for _cx in (coup_x - 0.10, coup_x + 0.10):
        for _by in (-0.40, -0.13, 0.13, 0.40):
            ax_top.add_patch(mpatches.Circle(
                (_cx, rotor_y + _by), 0.025,
                facecolor="#0f172a", edgecolor="#0f172a", zorder=4,
            ))

    # Cojinetes coloreados por la peor severidad de los sensores en ese
    # plano (worst-of). En compact es la unica fuente de severidad. En
    # full coexiste con la severidad detallada por sensor en el panel
    # polar — el lateral funciona como mini-heatmap del tren y el polar
    # como drill-down por sonda.
    def _bearing_facecolor_for_plane(plane_num: int) -> str:
        if not severity_by_label:
            return "white"
        plane_sensors = [s for s in sensors if int(s.get("plane", 0)) == plane_num]
        worst = _worst_status_for_plane(plane_sensors, severity_by_label)
        return _COLOR_SEVERITY.get(worst, "white")

    # En compact los cojinetes son ligeramente mas grandes y el numero
    # va en blanco si el fondo es de color (mejor contraste).
    bearing_radius = 0.24 if compact else 0.20

    def _bearing_text_color(face: str) -> str:
        # Si el fondo es severidad (no blanco), usar blanco. Si es blanco,
        # usar slate (color del bearing).
        return "white" if face != "white" else _COLOR_BEARING

    # Helper: dibuja un cojinete + numero + chips de tipo + label de plano.
    def _draw_bearing(plane_num: int, bx: float):
        face = _bearing_facecolor_for_plane(plane_num)
        ax_top.add_patch(mpatches.Circle(
            (bx, rotor_y), bearing_radius, facecolor=face,
            edgecolor=_COLOR_BEARING, linewidth=1.8, zorder=4,
        ))
        ax_top.text(bx, rotor_y, str(plane_num), fontsize=9, fontweight="bold",
                    ha="center", va="center",
                    color=_bearing_text_color(face), zorder=5)

        # Sensores en este plano
        plane_sensors = [s for s in sensors if int(s.get("plane", 0)) == plane_num]

        # Ciclo 15.1.3: chips de tipo de sensor debajo del numero — uno
        # por cada tipo presente en el plano (proximity / velocity /
        # accelerometer). Hace visible que en el TRF de la turbina hay
        # AMBOS velocity Y accelerometer, no solo el accel que daba la
        # plane_label.
        present_types = _sensor_types_in_plane(plane_sensors)
        chip_types = [t for t in present_types if t != "keyphasor"]
        if chip_types:
            chip_w = 0.13
            chips_total_w = chip_w * len(chip_types) + 0.04 * (len(chip_types) - 1)
            chip_x0 = bx - chips_total_w / 2 + chip_w / 2
            chip_y = rotor_y - bearing_radius - 0.18
            for k, t in enumerate(chip_types):
                cx = chip_x0 + k * (chip_w + 0.04)
                ax_top.add_patch(mpatches.Circle(
                    (cx, chip_y), chip_w / 2,
                    facecolor=_color_for_sensor_type(t),
                    edgecolor=_COLOR_TEXT, linewidth=0.4, zorder=4,
                ))
            label_y_top = chip_y - chip_w / 2 - 0.08
        else:
            label_y_top = rotor_y - bearing_radius - 0.10

        # Plane label normalizada (sin tokens del tipo de sensor)
        plane_lbl = _plane_display_label(plane_sensors)
        if plane_lbl:
            ax_top.text(bx, label_y_top, plane_lbl,
                        fontsize=7.2, ha="center", va="top",
                        color=_COLOR_TEXT, alpha=0.85, fontweight="bold")

    # Cojinetes del driver
    for i, p in enumerate(driver_planes):
        bx = x_start + (i + 0.7) * (drv_w / (n_drv + 0.4))
        _draw_bearing(p, bx)

    # Cojinetes del driven
    for i, p in enumerate(driven_planes):
        bx = dvn_x + (i + 0.5) * (dvn_w / n_dvn)
        _draw_bearing(p, bx)

    # Coupling label
    ax_top.text(coup_x, rotor_y - 1.10, "Coupling", fontsize=7, ha="center", va="top",
                color=_COLOR_TEXT, alpha=0.75, style="italic")

    # Keyphasor mark si existe — sobre el coupling
    if keyphasor_sensors:
        ax_top.plot(coup_x, rotor_y + 1.05, marker="*", markersize=16,
                    color=_COLOR_KEYPHASOR, markeredgecolor=_COLOR_TEXT,
                    markeredgewidth=0.6, zorder=5)
        ax_top.text(coup_x + 0.22, rotor_y + 1.05, "kp", fontsize=7.5,
                    color=_COLOR_KEYPHASOR, fontweight="bold", va="center")

    # Título superior — en compact lo dejamos vacio (el banner del
    # Tabular ya tiene su propia cabecera con el tag del activo).
    if not compact:
        title_text = "Mapa de Sensores"
        if train_label:
            title_text += f" — {train_label}"
        ax_top.set_title(title_text, fontsize=12, fontweight="bold",
                         color=_COLOR_TEXT, pad=8)

    # ============================================================
    # Panel inferior: vista polar por plano (R/L con sondas).
    # En compact se omite por completo.
    # ============================================================
    if compact:
        plane_axes_planes = []
    else:
        plane_axes_planes = list(driver_planes) + list(driven_planes)
    cols = max(n_planes_total, 4)
    for i, plane_num in enumerate(plane_axes_planes):
        ax = fig.add_subplot(gs[1, i % cols])
        ax.set_xlim(-1.4, 1.4)
        ax.set_ylim(-1.4, 1.4)
        ax.set_aspect("equal")
        ax.axis("off")

        # Círculo del eje (sección transversal)
        circle = mpatches.Circle((0, 0), 1.0, facecolor="white",
                                  edgecolor=_COLOR_BEARING, linewidth=2.0, zorder=2)
        ax.add_patch(circle)

        # Línea divisoria horizontal L/R y vertical 0°
        ax.plot([-1.05, 1.05], [0, 0], color=_COLOR_GRID, linewidth=0.8, zorder=1, linestyle="--")
        ax.plot([0, 0], [-1.05, 1.05], color=_COLOR_GRID, linewidth=0.8, zorder=1, linestyle="--")

        # Etiquetas L / R / 0° / 180°
        ax.text(0, 1.18, "0°", fontsize=7, ha="center", va="bottom", color="#94a3b8")
        ax.text(0, -1.18, "180°", fontsize=7, ha="center", va="top", color="#94a3b8")
        ax.text(-1.18, 0, "L", fontsize=8, ha="right", va="center",
                color="#475569", fontweight="bold")
        ax.text(1.18, 0, "R", fontsize=8, ha="left", va="center",
                color="#475569", fontweight="bold")

        # Sensores en este plano
        plane_sensors = [s for s in sensors if int(s.get("plane", 0)) == plane_num]
        for s in plane_sensors:
            angle_deg = float(s.get("angle_deg", 0.0) or 0.0)
            side = str(s.get("side", "")).strip().lower()
            sensor_type = (s.get("sensor_type") or "").lower()

            # Convertir ángulo + lado a coordenadas:
            # 0° = arriba (12 o'clock). Lado L = hemisferio izquierdo (negativo X),
            # R = hemisferio derecho (positivo X). top/bottom direct.
            if side == "l":
                # Mirando hacia el driver, L = izquierda. Ángulo mide desde 0° arriba
                # rotando hacia la izquierda (sentido antihorario en este plot).
                theta_rad = math.radians(90.0 + angle_deg)
            elif side == "r":
                theta_rad = math.radians(90.0 - angle_deg)
            elif side == "top":
                theta_rad = math.radians(90.0)
            elif side == "bottom":
                theta_rad = math.radians(-90.0)
            else:
                theta_rad = math.radians(90.0 - angle_deg)

            r_marker = 1.05  # radio del marker (justo afuera del círculo)
            x = r_marker * math.cos(theta_rad)
            y = r_marker * math.sin(theta_rad)

            # Ciclo 15.1 — color por severidad si está disponible,
            # si no por tipo de sensor (modo configuración).
            from core.sensor_map import sensor_label as _slabel
            lbl = _slabel(s)
            if severity_by_label and lbl in severity_by_label:
                sev_status = severity_by_label[lbl]
                color = _COLOR_SEVERITY.get(sev_status, _COLOR_SEVERITY.get("No Data", "#94a3b8"))
            else:
                color = _color_for_sensor_type(sensor_type)
            marker = _marker_for_sensor_type(sensor_type)
            ax.plot(x, y, marker=marker, markersize=12, color=color,
                    markeredgecolor=_COLOR_TEXT, markeredgewidth=0.8, zorder=4)

            # Label sensor (1Y_D, 2_RAD_A, etc.)
            label_r = 1.32
            lx = label_r * math.cos(theta_rad)
            ly = label_r * math.sin(theta_rad)
            ax.text(lx, ly, lbl, fontsize=6.5, color=color, fontweight="bold",
                    ha="center", va="center")

        # Título del plano
        plane_lbl = next((s.get("plane_label", "") for s in plane_sensors), "")
        title = f"Plano {plane_num}"
        if plane_lbl:
            title += f"\n{plane_lbl}"
        ax.set_title(title, fontsize=8, fontweight="bold", color=_COLOR_TEXT, pad=4)

    # Ciclo 15.1 — leyenda según modo (severidad vs configuración).
    # En compact omitimos la leyenda: el banner del Tabular ya muestra
    # 4 metricas KPI con los mismos colores y labels arriba.
    if not compact:
        if severity_by_label:
            legend_handles = [
                mpatches.Patch(color=_COLOR_SEVERITY["Normal"], label="CONDICIÓN ACEPTABLE"),
                mpatches.Patch(color=_COLOR_SEVERITY["Alarm"], label="ATENCIÓN"),
                mpatches.Patch(color=_COLOR_SEVERITY["Danger"], label="ACCIÓN REQUERIDA"),
                mpatches.Patch(color=_COLOR_SEVERITY["No Data"], label="Sin datos"),
            ]
        else:
            legend_handles = [
                mpatches.Patch(color=_COLOR_PROXIMITY, label="Proximity"),
                mpatches.Patch(color=_COLOR_VELOCITY, label="Velocity"),
                mpatches.Patch(color=_COLOR_ACCELEROMETER, label="Accelerometer"),
                mpatches.Patch(color=_COLOR_KEYPHASOR, label="Keyphasor"),
            ]
        fig.legend(handles=legend_handles, loc="lower center", ncol=4,
                   frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.0))

    if compact:
        fig.tight_layout(pad=0.4)
    else:
        fig.tight_layout(rect=[0, 0.03, 1, 1])
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=_COLOR_BG)
    plt.close(fig)
    return buf.getvalue()


__all__ = ["render_sensor_map_diagram"]
