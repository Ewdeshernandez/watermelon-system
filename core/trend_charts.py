"""
core.trend_charts
=================

Renderizado de mini line charts para mostrar la evolución temporal del
Overall de un sensor a través de los snapshots históricos. Pensados para
embeber en el PDF Reports (Ciclo 16.3) en la sección EVOLUCIÓN, debajo
de la tabla de cambios significativos.

Cada chart es pequeño (figsize ~4x2.2 in) y muestra:

  - Línea negra con markers redondos coloreados por Status del punto.
  - Líneas horizontales discontinuas para Alarm (ámbar) y Danger (rojo).
  - Etiquetas A / D al lado de las threshold lines.
  - Eje X: timestamp (formato corto).
  - Eje Y: unidad nativa del sensor.
  - Título: ``{sensor_label} · {plane_label}``.

El renderer es robusto a:
  - Pocos puntos (1-2): muestra el chart con markers grandes.
  - Snapshots con timestamps malformados: los descarta silencioso.
  - Falta de matplotlib en el entorno: devuelve None y el caller cae al
    fallback (omitir el grid).
"""

from __future__ import annotations

from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List, Optional


_COLOR_STATUS = {
    "Normal": "#16a34a",
    "Alarm": "#f59e0b",
    "Danger": "#dc2626",
    "No Data": "#94a3b8",
    "": "#94a3b8",
}


def _parse_ts(ts: str) -> Optional[datetime]:
    """Parse ISO8601 robust to suffixes (Z) y precisión variable."""
    if not ts:
        return None
    s = ts.strip().replace("Z", "")
    try:
        return datetime.fromisoformat(s)
    except Exception:
        # Intentar formatos alternativos
        for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.strptime(s[: len(fmt) + 6], fmt)
            except Exception:
                continue
    return None


def render_sensor_trend_chart(
    history_points: List[Dict[str, Any]],
    *,
    sensor_label: str,
    plane_label: str = "",
    alarm: float = 0.0,
    danger: float = 0.0,
    unit: str = "",
    figure_width_in: float = 4.6,
    figure_height_in: float = 2.4,
) -> Optional[bytes]:
    """
    Devuelve PNG bytes con un mini line chart del histórico de un sensor.
    Devuelve None si matplotlib no está disponible o no hay puntos válidos.

    Args:
        history_points: lista de dicts con keys timestamp, overall, status.
            La que devuelve get_sensor_history en core.instance_history.
        sensor_label: ej. "1_RAD_A"
        plane_label: ej. "CRF Accel"
        alarm: setpoint de Alarm (línea ámbar). 0 = no dibujar.
        danger: setpoint de Danger (línea roja). 0 = no dibujar.
        unit: unidad para el eje Y (ej. "g peak").
        figure_width_in / figure_height_in: tamaño del chart.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except Exception:
        return None

    if not history_points:
        return None

    # Parse y filtrar puntos validos
    xs: List[datetime] = []
    ys: List[float] = []
    sts: List[str] = []
    labels: List[str] = []
    for p in history_points:
        ts = _parse_ts(str(p.get("timestamp", "") or ""))
        if ts is None:
            continue
        try:
            ov = float(p.get("overall", 0))
        except Exception:
            continue
        xs.append(ts)
        ys.append(ov)
        sts.append(str(p.get("status", "") or ""))
        labels.append(str(p.get("corrida_label", "") or ""))

    if not xs:
        return None

    fig, ax = plt.subplots(
        figsize=(figure_width_in, figure_height_in),
        dpi=130,
        facecolor="white",
    )

    # Línea conectora con markers (debajo de los markers de color)
    ax.plot(xs, ys, color="#0f172a", linewidth=1.4, zorder=2, alpha=0.85)

    # Markers coloreados por status
    for x, y, st in zip(xs, ys, sts):
        col = _COLOR_STATUS.get(st, "#94a3b8")
        ax.plot(
            x, y, marker="o", markersize=8, color=col,
            markeredgecolor="#0f172a", markeredgewidth=0.6, zorder=4,
        )

    # Threshold lines
    if danger > 0:
        ax.axhline(
            danger, color=_COLOR_STATUS["Danger"], linewidth=0.9,
            linestyle="--", alpha=0.7, zorder=1,
        )
    if alarm > 0:
        ax.axhline(
            alarm, color=_COLOR_STATUS["Alarm"], linewidth=0.9,
            linestyle="--", alpha=0.7, zorder=1,
        )

    # Anotar D y A a la derecha de las threshold lines
    if xs:
        x_max = xs[-1]
        if danger > 0:
            ax.text(
                x_max, danger, " D",
                fontsize=7.5, color=_COLOR_STATUS["Danger"],
                va="center", ha="left", fontweight="bold",
            )
        if alarm > 0:
            ax.text(
                x_max, alarm, " A",
                fontsize=7.5, color=_COLOR_STATUS["Alarm"],
                va="center", ha="left", fontweight="bold",
            )

    # Cosmetics
    title = sensor_label
    if plane_label:
        title = f"{sensor_label} · {plane_label}"
    ax.set_title(title, fontsize=10.5, fontweight="bold",
                 color="#0f172a", pad=4, loc="left")
    if unit:
        ax.set_ylabel(unit, fontsize=8.5, color="#475569")

    ax.tick_params(axis="both", labelsize=7.5, colors="#475569")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color("#cbd5e1")
    ax.spines["bottom"].set_color("#cbd5e1")
    ax.grid(True, axis="y", linestyle=":", alpha=0.35, color="#94a3b8")

    # Format X axis con fechas cortas
    if len(xs) > 1:
        loc = mdates.AutoDateLocator(minticks=2, maxticks=5)
        fmt = mdates.DateFormatter("%d-%b")
        ax.xaxis.set_major_locator(loc)
        ax.xaxis.set_major_formatter(fmt)
    else:
        # Un solo punto: mostrarlo con su fecha
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))

    fig.autofmt_xdate(rotation=20, ha="right")

    # Padding suave a la derecha para que las labels A/D no se cortan
    if xs:
        x_min = xs[0]
        x_max = xs[-1]
        if x_min == x_max:
            from datetime import timedelta
            delta = timedelta(days=1)
            ax.set_xlim(x_min - delta, x_max + delta)
        else:
            span = (x_max - x_min)
            ax.set_xlim(x_min, x_max + span * 0.08)

    fig.tight_layout(pad=0.4)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    return buf.getvalue()


__all__ = ["render_sensor_trend_chart"]
