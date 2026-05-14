"""
core/modal/modal_animator.py — Visualización de mode shapes
============================================================

Genera visualizaciones de los mode shapes identificados:
  · Nivel 1 — Bar chart 2D (magnitud + fase por DOF)
  · Nivel 2 — Flechas 3D sobre wireframe (planar machine layout)
  · Nivel 3 — Mesh3D animado con colormap (próximo sprint)

Norma aplicable
---------------
ISO 7626-6 §7.2 — Visualización de mode shapes. Requiere indicación clara
de la magnitud relativa y la fase entre puntos de medición.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Union
import numpy as np


def build_bar_chart_mode_shape(
    mode_shape: np.ndarray,
    channel_names: List[str],
    mode_label: str = "",
):
    """
    Nivel 1 — Bar chart con magnitud + fase del mode shape.

    Args:
        mode_shape: vector complejo (N_DOFs,)
        channel_names: etiquetas de cada DOF
        mode_label: título descriptivo del modo (e.g. "Modo 1 · 53.5 Hz · ζ=1.6%")

    Returns:
        plotly.graph_objects.Figure con 2 subplots: magnitud y fase
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:
        raise ImportError("plotly requerido") from exc

    shape = np.asarray(mode_shape, dtype=complex)
    mag = np.abs(shape)
    # Normalizar magnitud para visualización (max = 1)
    if mag.max() > 0:
        mag = mag / mag.max()
    phase_deg = np.degrees(np.angle(shape))

    # Color por fase — naranja para 0°, azul para 180° (anti-fase)
    colors = ["#1AAEE5" if abs(p) > 90 else "#D89B22" for p in phase_deg]

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Magnitud (normalizada)", "Fase (°)"),
        vertical_spacing=0.18,
        shared_xaxes=True,
    )
    fig.add_trace(go.Bar(
        x=channel_names, y=mag, marker_color=colors,
        text=[f"{m:.2f}" for m in mag], textposition="outside",
        showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=channel_names, y=phase_deg, marker_color=colors,
        text=[f"{p:.0f}°" for p in phase_deg], textposition="outside",
        showlegend=False,
    ), row=2, col=1)
    fig.update_yaxes(title_text="|φ| (norm)", row=1, col=1, range=[0, 1.15])
    fig.update_yaxes(title_text="∠φ (°)", row=2, col=1, range=[-200, 200])
    fig.update_xaxes(title_text="DOF / Sensor", row=2, col=1)
    fig.update_layout(
        title=mode_label or "Mode shape — Nivel 1 (Bar chart)",
        height=460,
        template="plotly_white",
        margin=dict(l=50, r=20, t=70, b=40),
    )
    return fig


def build_arrows_2d_layout(
    mode_shape: np.ndarray,
    channel_positions: Sequence[tuple],  # [(x, y), ...] en plano del activo
    channel_directions: Sequence[tuple],  # [(dx, dy), ...] vectores unitarios
    channel_names: List[str],
    mode_label: str = "",
):
    """
    Nivel 2 — Flechas 2D sobre layout del activo.

    Cada DOF se renderiza como flecha en su posición, escalada por la
    magnitud del mode shape y orientada según la fase (positiva/negativa).

    Args:
        mode_shape: vector complejo (N_DOFs,)
        channel_positions: lista de (x, y) por DOF
        channel_directions: lista de (dx, dy) — eje sensible de cada sensor
        channel_names: etiquetas
        mode_label: título

    Returns:
        plotly Figure con quiver-style arrows
    """
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError("plotly requerido") from exc

    shape = np.asarray(mode_shape, dtype=complex)
    mag = np.abs(shape)
    if mag.max() > 0:
        mag = mag / mag.max()
    # Signo por fase: ~ 0° → +, ~ 180° → -
    phase_deg = np.degrees(np.angle(shape))
    sign = np.where(np.abs(phase_deg) < 90, 1.0, -1.0)
    arrow_lengths = mag * sign

    fig = go.Figure()

    # Base: puntos de cada sensor
    xs = [p[0] for p in channel_positions]
    ys = [p[1] for p in channel_positions]
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers+text",
        marker=dict(size=12, color="#0F1E3D", symbol="square"),
        text=channel_names, textposition="bottom center",
        textfont=dict(size=10),
        name="Sensores", showlegend=False,
    ))

    # Flechas: usando go.Scatter con annotation arrows
    arrow_scale = 0.4  # factor de escala visual
    for i, (pos, dir_v, m_signed, name) in enumerate(zip(
        channel_positions, channel_directions, arrow_lengths, channel_names
    )):
        x0, y0 = pos
        dx_unit, dy_unit = dir_v[0], dir_v[1]
        x1 = x0 + arrow_scale * dx_unit * m_signed
        y1 = y0 + arrow_scale * dy_unit * m_signed
        color = "#16a34a" if m_signed > 0 else "#dc2626"
        fig.add_annotation(
            x=x1, y=y1, ax=x0, ay=y0,
            xref="x", yref="y", axref="x", ayref="y",
            arrowhead=3, arrowsize=1.5, arrowwidth=2.5,
            arrowcolor=color, showarrow=True, text="",
        )

    fig.update_layout(
        title=mode_label or "Mode shape — Nivel 2 (flechas 2D)",
        xaxis_title="X (m)", yaxis_title="Y (m)",
        height=500,
        template="plotly_white",
        margin=dict(l=50, r=20, t=70, b=40),
        showlegend=False,
    )
    # Mantener aspect ratio
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def build_arrows_3d_wireframe(
    mode_shape: np.ndarray,
    channel_positions_3d: Sequence[tuple],   # [(x, y, z), ...]
    channel_directions_3d: Sequence[tuple],  # [(dx, dy, dz), ...]
    channel_names: List[str],
    mode_label: str = "",
):
    """
    Nivel 2.5 — Flechas 3D usando Plotly Cone.

    Versión 3D del arrows layout. Requiere position_3d + dof_direction
    poblados en el Sensor Map del activo.
    """
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError("plotly requerido") from exc

    shape = np.asarray(mode_shape, dtype=complex)
    mag = np.abs(shape)
    if mag.max() > 0:
        mag = mag / mag.max()
    phase_deg = np.degrees(np.angle(shape))
    sign = np.where(np.abs(phase_deg) < 90, 1.0, -1.0)
    scaled_mag = mag * sign

    xs, ys, zs = zip(*channel_positions_3d)

    # Cone glyphs por sensor
    arrow_scale = 0.4
    u = [scaled_mag[i] * channel_directions_3d[i][0] * arrow_scale
          for i in range(len(channel_names))]
    v = [scaled_mag[i] * channel_directions_3d[i][1] * arrow_scale
          for i in range(len(channel_names))]
    w = [scaled_mag[i] * channel_directions_3d[i][2] * arrow_scale
          for i in range(len(channel_names))]

    fig = go.Figure()
    fig.add_trace(go.Cone(
        x=xs, y=ys, z=zs, u=u, v=v, w=w,
        sizemode="absolute", sizeref=0.3,
        anchor="tail",
        colorscale="RdBu_r",
        cmin=-1, cmax=1,
        cmid=0,
        showscale=True,
        colorbar=dict(title="Mode shape<br>magnitud × signo"),
    ))
    # Puntos base
    fig.add_trace(go.Scatter3d(
        x=xs, y=ys, z=zs, mode="markers+text",
        marker=dict(size=4, color="#0F1E3D"),
        text=channel_names, textposition="top center",
        showlegend=False,
    ))
    fig.update_layout(
        title=mode_label or "Mode shape 3D — Nivel 2 (flechas wireframe)",
        scene=dict(
            xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)",
            aspectmode="data",
        ),
        height=600,
        template="plotly_white",
        margin=dict(l=20, r=20, t=70, b=20),
    )
    return fig
