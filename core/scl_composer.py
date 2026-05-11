"""
Watermelon System — Shaft Centerline Plot composer (Ciclo 23.73).

El SCL (Shaft Centerline Plot, también "Average Shaft Position") grafica
la posición promedio del eje rotor dentro del clearance del cojinete a
partir del DC gap voltage de proximity probes ortogonales (X/Y).

Para qué sirve:
  • Lift-off curve durante startup
  • Detección de misalignment (paralelo y angular)
  • Bearing wear (deriva lenta del centro operativo)
  • Loss of preload en tilting pad
  • Thermal growth durante run-up
  • Wipe / contact (eje cruza el clearance boundary)

Output principal:
  compose_shaft_centerline_plot(...) → plotly.graph_objects.Figure

Convención de signos (API 670):
  • X horizontal: positivo a la derecha del eje (mirando al rotor desde
    el lado driven, según convención Bently System1)
  • Y vertical: positivo hacia arriba
  • Gap voltage Bently 3300XL: más negativo = eje más cerca del probe
    (polaridad invertida del 7200/8mm series legacy)
  • Cold reference: eje apoyado en babbitt inferior por gravedad → (0, -Ca/2)
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Tuple, Dict, Any
import math


# Reutilizamos constantes y conversiones del helper de clearance
from core.bearing_clearance import (
    BearingClearance,
    BENTLY_3300XL_MV_PER_UM,
    UM_PER_MIL,
)


def position_from_gap_voltages(
    gap_x_v: float,
    gap_y_v: float,
    cold_ref_x_v: float,
    cold_ref_y_v: float,
    sensitivity_mv_per_um: float = BENTLY_3300XL_MV_PER_UM,
    flip_x: bool = False,
    flip_y: bool = False,
) -> Tuple[float, float]:
    """Convierte un par de gap voltages (V) a posición (x_um, y_um).

    Asunción de signo: Bently 3300XL convention. Cuando el eje se aleja
    del probe X horizontal montado a la derecha del shaft, gap voltage
    se vuelve MENOS negativo (más cercano a 0). Por eso:

        delta_gap_v (running − cold_ref) > 0  → eje se ALEJÓ del probe
                                              → posición negativa hacia el lado del probe
                                              → necesita flip de signo

    El `flip_x`/`flip_y` permite acomodar instalaciones donde el probe
    está montado en orientación inversa (Westinghouse vs GE, por ej).

    Args:
        gap_x_v / gap_y_v: voltage actual (Volts, no mV).
        cold_ref_x_v / cold_ref_y_v: voltage en reposo (Volts).
        sensitivity_mv_per_um: 7.874 mV/μm para Bently 3300XL.
        flip_x / flip_y: True si el probe está montado al revés.

    Returns:
        (x_um, y_um) en μm relativo al cold reference.
    """
    if sensitivity_mv_per_um <= 0:
        raise ValueError("sensitivity_mv_per_um debe ser > 0")
    dx_um = (gap_x_v - cold_ref_x_v) * 1000.0 / sensitivity_mv_per_um
    dy_um = (gap_y_v - cold_ref_y_v) * 1000.0 / sensitivity_mv_per_um
    # API 670 / Bently: gap más negativo = eje más cerca del probe.
    # Por defecto, delta positivo = eje se alejó del probe → en convención
    # de plot, eso significa posición negativa hacia el lado del probe.
    # Por eso negamos los deltas (probe X a la derecha: dx_um negativo
    # significa eje a la izquierda, lejos del probe).
    x_um = -dx_um if not flip_x else dx_um
    y_um = -dy_um if not flip_y else dy_um
    return x_um, y_um


def current_severity(
    position_x_um: float,
    position_y_um: float,
    clearance: BearingClearance,
) -> Tuple[str, str, float]:
    """Devuelve (severity, color, distance_pct).

    severity: "normal" | "alarm" | "danger" | "outside_clearance"
    color: hex CSS para el punto en el plot
    distance_pct: distancia del cold reference / radio del clearance × 100
    """
    cold_x_um = 0.0
    cold_y_um = clearance.cold_reference_y_um  # típicamente -Ca/2
    dx = position_x_um - cold_x_um
    dy = position_y_um - cold_y_um
    distance_um = math.sqrt(dx * dx + dy * dy)

    # Radio efectivo del clearance boundary respecto al cold ref:
    # cold ref está en (0, -Ca/2), boundary llega hasta (0, +Ca/2) en
    # el extremo superior → distancia desde cold ref al borde superior = Ca
    boundary_radius_um = clearance.assembled_clearance_um_pp

    distance_pct = (distance_um / boundary_radius_um) * 100.0 if boundary_radius_um else 0.0

    if distance_pct >= 100.0:
        return ("outside_clearance", "#dc2626", distance_pct)
    if distance_pct >= 60.0:
        return ("danger", "#dc2626", distance_pct)
    if distance_pct >= 40.0:
        return ("alarm", "#b45309", distance_pct)
    return ("normal", "#15803d", distance_pct)


def compose_shaft_centerline_plot(
    clearance: BearingClearance,
    position_x_um: float,
    position_y_um: float,
    history: Optional[List[Tuple[float, float]]] = None,
    bearing_label: str = "BRG",
    show_alarm_rings: bool = True,
) -> Any:
    """Construye un plotly figure del SCL plot.

    Args:
        clearance: BearingClearance del helper bearing_clearance.compute_*
        position_x_um / position_y_um: posición actual del eje (μm)
        history: lista de (x_um, y_um) de posiciones recientes para la
            traza desvaneciente. Si None, no se dibuja traza.
        bearing_label: ej. "BRG #1 (INLET)" — para el title del plot
        show_alarm_rings: dibujar círculos de 40% y 60% del boundary

    Returns:
        plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go

    Ca = clearance.assembled_clearance_um_pp
    cold_y = clearance.cold_reference_y_um

    severity, color, dist_pct = current_severity(
        position_x_um, position_y_um, clearance
    )

    fig = go.Figure()

    # 1. Clearance boundary — círculo gris (Ca diametral)
    # Centro del bearing: (0, 0). Radio = Ca (boundary que ve el eje
    # desde su posición de cold ref es Ca, porque cold ref está a -Ca/2
    # del centro y el extremo opuesto del clearance está a +Ca/2).
    # Realmente el clearance es un círculo centrado en el bearing, radio Ca/2.
    # Pero por convención API 670, el plot dibuja el círculo de DIÁMETRO Ca,
    # es decir radio Ca/2.
    R = Ca / 2.0
    _add_circle(fig, 0, 0, R, color="#94a3b8", width=2, dash="solid",
                name="Clearance boundary (Ca)")

    if show_alarm_rings:
        # 2. Anillo de alarm (40% Ca radio)
        _add_circle(fig, 0, 0, R * 0.40, color="#f59e0b", width=1, dash="dot",
                    name="Alarm boundary (40% Ca)")
        # 3. Anillo de danger (60% Ca radio)
        _add_circle(fig, 0, 0, R * 0.60, color="#dc2626", width=1, dash="dot",
                    name="Danger boundary (60% Ca)")

    # 4. Cold reference (eje apoyado en babbitt inferior por gravedad)
    fig.add_trace(go.Scatter(
        x=[0], y=[cold_y],
        mode="markers+text",
        marker=dict(symbol="circle-open", size=18, color="#64748b", line=dict(width=2)),
        text=["Cold ref"],
        textposition="bottom center",
        textfont=dict(size=10, color="#64748b"),
        name="Cold reference",
        hovertemplate=(
            "<b>Cold reference</b><br>"
            "Posición en reposo<br>"
            f"x = 0, y = {cold_y:.1f} μm<extra></extra>"
        ),
    ))

    # 5. Trayectoria histórica (línea desvaneciente)
    if history:
        hist_x = [p[0] for p in history]
        hist_y = [p[1] for p in history]
        fig.add_trace(go.Scatter(
            x=hist_x, y=hist_y,
            mode="lines",
            line=dict(color="#3b82f6", width=1.2),
            opacity=0.4,
            name="Trayectoria histórica",
            hovertemplate="x=%{x:.1f}, y=%{y:.1f} μm<extra></extra>",
        ))

    # 6. Posición actual (punto grande, color por severidad)
    fig.add_trace(go.Scatter(
        x=[position_x_um], y=[position_y_um],
        mode="markers+text",
        marker=dict(symbol="circle", size=18, color=color, line=dict(width=2, color="white")),
        text=[f"{dist_pct:.0f}%"],
        textposition="top center",
        textfont=dict(size=11, color=color, family="ui-monospace"),
        name=f"Posición actual ({severity})",
        hovertemplate=(
            f"<b>Posición actual</b><br>"
            f"x = {position_x_um:.1f} μm<br>"
            f"y = {position_y_um:.1f} μm<br>"
            f"Distancia desde cold ref: {dist_pct:.0f}% del clearance<br>"
            f"Severidad: {severity}<extra></extra>"
        ),
    ))

    # Layout
    # Aspect ratio 1:1 IS CRITICAL — sin esto el círculo se distorsiona
    # y la lectura del plot se vuelve incorrecta.
    range_pad = R * 1.3
    fig.update_layout(
        margin=dict(l=10, r=10, t=40, b=10),
        height=480,
        plot_bgcolor="white",
        title=dict(
            text=(
                f"<b>Shaft Centerline — {bearing_label}</b><br>"
                f"<span style='font-size:11px;color:#64748b;'>"
                f"Ø {clearance.shaft_dia_mm:.0f} mm · "
                f"Ca = {Ca:.0f} μm pp · "
                f"{clearance.bearing_type}</span>"
            ),
            font=dict(size=13),
            x=0.02, xanchor="left",
        ),
        xaxis=dict(
            title="Horizontal (X) μm",
            range=[-range_pad, range_pad],
            zeroline=True, zerolinecolor="#cbd5e1", zerolinewidth=1,
            showgrid=True, gridcolor="#f1f5f9",
            scaleanchor="y", scaleratio=1,  # aspect ratio 1:1
        ),
        yaxis=dict(
            title="Vertical (Y) μm",
            range=[-range_pad, range_pad],
            zeroline=True, zerolinecolor="#cbd5e1", zerolinewidth=1,
            showgrid=True, gridcolor="#f1f5f9",
        ),
        showlegend=False,
        hovermode="closest",
    )

    return fig


def _add_circle(fig, cx: float, cy: float, r: float,
                color: str, width: float, dash: str, name: str) -> None:
    """Helper: agrega un círculo (mediante 60 puntos en scatter line)."""
    n = 60
    theta = [2 * math.pi * i / n for i in range(n + 1)]
    xs = [cx + r * math.cos(t) for t in theta]
    ys = [cy + r * math.sin(t) for t in theta]
    import plotly.graph_objects as go
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines",
        line=dict(color=color, width=width, dash=dash),
        name=name,
        hoverinfo="skip",
        showlegend=False,
    ))


def detect_bearing_pair(sensor_label: str) -> Optional[Tuple[str, str, int]]:
    """Detecta el bearing number y el pair X/Y a partir del sensor_label.

    Convención Watermelon:
        1XD / 1YD → BRG #1 (displacement, proximity probe)
        2XD / 2YD → BRG #2
        3XD / 3YD → BRG #3
        4XD / 4YD → BRG #4

    Args:
        sensor_label: ej. "3XD" o "3YD" o "1YA" (accelerometer)

    Returns:
        (x_label, y_label, bearing_number) si es un displacement pair válido.
        None si el sensor no tiene pair de displacement (ej. accelerometer 1YA).
    """
    if not sensor_label or len(sensor_label) < 3:
        return None
    try:
        bearing_n = int(sensor_label[0])
    except ValueError:
        return None
    axis = sensor_label[1].upper()
    metric = sensor_label[2].upper()
    if metric != "D":
        # No es displacement (típicamente "A" acelerómetro o "V" velocity)
        return None
    if axis not in ("X", "Y"):
        return None
    x_label = f"{bearing_n}XD"
    y_label = f"{bearing_n}YD"
    return (x_label, y_label, bearing_n)
