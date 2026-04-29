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


def render_sensor_map_diagram(
    sensors: List[Dict[str, Any]],
    *,
    train_label: str = "",
    driver_label: str = "Driver",
    driven_label: str = "Driven",
    figure_width_in: float = 12.0,
    severity_by_label: Optional[Dict[str, str]] = None,
) -> Optional[bytes]:
    """
    Devuelve PNG bytes con el diagrama completo del Sensor Map.

    Args:
        sensors: lista de sensores (dicts).
        train_label: subtítulo (ej. "Turbogenerador GE LM6000 + Brush 54 MW").
        driver_label / driven_label: etiquetas de las dos máquinas.
        figure_width_in: ancho del figure en pulgadas.
        severity_by_label: (Ciclo 15.1) dict opcional mapeando sensor_label
            → status ("Normal" / "Alarm" / "Danger" / "No Data"). Cuando
            se provee, los markers se colorean por severidad. Cuando no,
            usan el color por tipo de sensor (modo "configuración").

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
    # Fallback: si plane_labels no indican driver/driven, partir por mediana
    all_planes = sorted({int(s.get("plane", 0)) for s in sensors if s.get("plane", 0) > 0})
    if not driver_planes and not driven_planes and all_planes:
        mid = len(all_planes) // 2
        driver_planes = all_planes[:mid] if mid > 0 else all_planes
        driven_planes = all_planes[mid:] if mid > 0 else []

    keyphasor_sensors = [s for s in sensors if (s.get("sensor_type") or "").lower() == "keyphasor"]

    n_planes_total = len(driver_planes) + len(driven_planes)
    if n_planes_total == 0:
        return None

    # Layout: panel superior con el tren, panel inferior con polar views
    fig = plt.figure(
        figsize=(figure_width_in, 5.5 + 1.5 * ((n_planes_total + 3) // 4)),
        facecolor=_COLOR_BG,
    )
    gs = fig.add_gridspec(
        2, max(n_planes_total, 4),
        height_ratios=[1.5, 2.0],
        hspace=0.45, wspace=0.35,
    )

    # ============================================================
    # Panel superior: vista lateral del tren
    # ============================================================
    ax_top = fig.add_subplot(gs[0, :])
    ax_top.set_xlim(0, 10)
    ax_top.set_ylim(0, 4)
    ax_top.set_aspect("equal")
    ax_top.axis("off")

    # Bloques driver y driven
    n_drv = max(1, len(driver_planes))
    n_dvn = max(1, len(driven_planes))
    drv_w = 2.5 + 0.6 * (n_drv - 1)
    dvn_w = 2.5 + 0.6 * (n_dvn - 1)
    coupling_w = 0.6
    total_w = drv_w + coupling_w + dvn_w
    x_start = 5.0 - total_w / 2.0  # centrar

    drv_box = mpatches.FancyBboxPatch(
        (x_start, 1.2), drv_w, 1.6,
        boxstyle="round,pad=0.04,rounding_size=0.12",
        facecolor=_COLOR_DRIVER, alpha=0.18, edgecolor=_COLOR_DRIVER, linewidth=1.6,
    )
    ax_top.add_patch(drv_box)
    ax_top.text(x_start + drv_w / 2, 2.95, driver_label,
                fontsize=12, fontweight="bold", color=_COLOR_DRIVER,
                ha="center", va="center")

    dvn_x = x_start + drv_w + coupling_w
    dvn_box = mpatches.FancyBboxPatch(
        (dvn_x, 1.2), dvn_w, 1.6,
        boxstyle="round,pad=0.04,rounding_size=0.12",
        facecolor=_COLOR_DRIVEN, alpha=0.18, edgecolor=_COLOR_DRIVEN, linewidth=1.6,
    )
    ax_top.add_patch(dvn_box)
    ax_top.text(dvn_x + dvn_w / 2, 2.95, driven_label,
                fontsize=12, fontweight="bold", color=_COLOR_DRIVEN,
                ha="center", va="center")

    # Eje horizontal (línea negra que pasa por los cojinetes)
    ax_top.plot(
        [x_start - 0.2, dvn_x + dvn_w + 0.2], [2.0, 2.0],
        color="#0f172a", linewidth=2.5, zorder=2,
    )

    # Cojinetes del driver
    for i, p in enumerate(driver_planes):
        bx = x_start + (i + 0.5) * (drv_w / n_drv)
        ax_top.add_patch(mpatches.Circle(
            (bx, 2.0), 0.18, facecolor="white",
            edgecolor=_COLOR_BEARING, linewidth=1.8, zorder=3,
        ))
        ax_top.text(bx, 2.0, str(p), fontsize=9, fontweight="bold",
                    ha="center", va="center", color=_COLOR_BEARING, zorder=4)
        # plane label abajo
        plane_lbl = next((s.get("plane_label", "") for s in sensors if s.get("plane") == p), "")
        ax_top.text(bx, 1.45, plane_lbl, fontsize=7, ha="center", va="top",
                    color=_COLOR_TEXT, alpha=0.75)

    # Cojinetes del driven
    for i, p in enumerate(driven_planes):
        bx = dvn_x + (i + 0.5) * (dvn_w / n_dvn)
        ax_top.add_patch(mpatches.Circle(
            (bx, 2.0), 0.18, facecolor="white",
            edgecolor=_COLOR_BEARING, linewidth=1.8, zorder=3,
        ))
        ax_top.text(bx, 2.0, str(p), fontsize=9, fontweight="bold",
                    ha="center", va="center", color=_COLOR_BEARING, zorder=4)
        plane_lbl = next((s.get("plane_label", "") for s in sensors if s.get("plane") == p), "")
        ax_top.text(bx, 1.45, plane_lbl, fontsize=7, ha="center", va="top",
                    color=_COLOR_TEXT, alpha=0.75)

    # Coupling marker (entre las dos máquinas)
    coup_x = x_start + drv_w + coupling_w / 2
    ax_top.plot([coup_x, coup_x], [1.8, 2.2], color="#475569", linewidth=2.0, zorder=3)
    ax_top.text(coup_x, 1.05, "Coupling", fontsize=7, ha="center", va="top",
                color=_COLOR_TEXT, alpha=0.75)

    # Keyphasor mark si existe
    if keyphasor_sensors:
        ax_top.plot(coup_x, 2.0, marker="*", markersize=18,
                    color=_COLOR_KEYPHASOR, markeredgecolor=_COLOR_TEXT,
                    markeredgewidth=0.6, zorder=5)
        ax_top.text(coup_x + 0.2, 2.45, "kp", fontsize=8,
                    color=_COLOR_KEYPHASOR, fontweight="bold", va="center")

    # Título superior
    title_text = "Mapa de Sensores"
    if train_label:
        title_text += f" — {train_label}"
    ax_top.set_title(title_text, fontsize=12, fontweight="bold",
                     color=_COLOR_TEXT, pad=8)

    # ============================================================
    # Panel inferior: vista polar por plano (R/L con sondas)
    # ============================================================
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

    # Ciclo 15.1 — leyenda según modo (severidad vs configuración)
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

    fig.tight_layout(rect=[0, 0.03, 1, 1])
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=_COLOR_BG)
    plt.close(fig)
    return buf.getvalue()


__all__ = ["render_sensor_map_diagram"]
