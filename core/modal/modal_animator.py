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


def build_complexity_polar_plot(
    mode_shape,
    channel_names: List[str],
    mode_label: str = "",
):
    """
    Complexity Polar Plot estilo Artemis (Figura 10).

    Cada componente del mode shape se dibuja como vector radial en el plano
    complejo (Real horizontal, Imag vertical). Si todos los vectores son
    colineales (apuntan en la misma dirección o opuestas), el modo es real.
    Si están dispersos, el modo es complejo.

    Cumple ISO 7626-6 §7.2 — visualización geométrica del MPC.

    Args:
        mode_shape: vector complejo (N_DOFs,) — np.ndarray
        channel_names: etiquetas
        mode_label: título descriptivo
    """
    try:
        import plotly.graph_objects as go
        import numpy as _np
    except ImportError as exc:
        raise ImportError("plotly requerido") from exc

    phi = _np.asarray(mode_shape, dtype=complex).flatten()
    # Normalizar al máximo módulo para escala consistente
    max_mag = float(_np.abs(phi).max())
    if max_mag > 0:
        phi = phi / max_mag

    fig = go.Figure()
    # Círculo unidad de referencia
    theta = _np.linspace(0, 2 * _np.pi, 100)
    fig.add_trace(go.Scatter(
        x=_np.cos(theta), y=_np.sin(theta),
        mode="lines",
        line=dict(color="#cbd5e1", width=1, dash="dot"),
        showlegend=False, hoverinfo="skip",
    ))
    # Ejes Real e Imag
    fig.add_shape(type="line", x0=-1.1, x1=1.1, y0=0, y1=0,
                   line=dict(color="#94a3b8", width=1))
    fig.add_shape(type="line", x0=0, x1=0, y0=-1.1, y1=1.1,
                   line=dict(color="#94a3b8", width=1))

    # Cada componente del mode shape como vector desde origen
    for i, (val, name) in enumerate(zip(phi, channel_names)):
        re = float(_np.real(val))
        im = float(_np.imag(val))
        fig.add_annotation(
            x=re, y=im, ax=0, ay=0,
            xref="x", yref="y", axref="x", ayref="y",
            arrowhead=3, arrowsize=1.5, arrowwidth=2,
            arrowcolor="#1AAEE5", showarrow=True, text="",
        )
        # Etiqueta al final del vector
        offset = 0.08
        norm = (re ** 2 + im ** 2) ** 0.5
        if norm > 1e-3:
            xt = re * (1 + offset / norm)
            yt = im * (1 + offset / norm)
        else:
            xt, yt = re, im
        fig.add_annotation(
            x=xt, y=yt, text=name,
            showarrow=False, font=dict(size=10, color="#0F1E3D"),
        )

    fig.update_layout(
        title=mode_label or "Complexity Polar Plot — mode shape vectors",
        xaxis=dict(title="Re(φ)", range=[-1.3, 1.3],
                    zeroline=True, gridcolor="#e2e8f0",
                    scaleanchor="y", scaleratio=1),
        yaxis=dict(title="Im(φ)", range=[-1.3, 1.3],
                    zeroline=True, gridcolor="#e2e8f0"),
        height=460, template="plotly_white",
        margin=dict(l=50, r=50, t=70, b=40),
        showlegend=False,
    )
    return fig


def build_mac_matrix_plot(
    mac_matrix,
    mode_labels: List[str],
    title: str = "AutoMAC Matrix",
    use_3d: bool = True,
):
    """
    Visualización de la matriz MAC.

    Si use_3d=True → barras 3D (estilo Figura 9 Artemis).
    Si use_3d=False → heatmap 2D (más rápido y siempre legible).

    Args:
        mac_matrix: numpy array (N, N) con valores MAC ∈ [0, 1]
        mode_labels: etiquetas de cada modo (e.g. "21.77 Hz")
        title: título
        use_3d: usar bar chart 3D estilo Artemis

    Returns:
        plotly Figure
    """
    try:
        import plotly.graph_objects as go
        import numpy as _np
    except ImportError as exc:
        raise ImportError("plotly requerido") from exc

    M = _np.asarray(mac_matrix)
    n = M.shape[0]

    if use_3d and n > 1:
        # Bar3d via Mesh3d cubes — más complejo pero replica visual Artemis
        # Versión simplificada: surface plot 3D estilo "stem"
        x_grid, y_grid = _np.meshgrid(_np.arange(n), _np.arange(n))
        fig = go.Figure()
        # Cada barra como Mesh3d cube
        for i in range(n):
            for j in range(n):
                z_val = float(M[i, j])
                if z_val < 0.05:
                    continue  # skip barras muy bajas para legibilidad
                fig.add_trace(go.Mesh3d(
                    x=[i-0.4, i+0.4, i+0.4, i-0.4, i-0.4, i+0.4, i+0.4, i-0.4],
                    y=[j-0.4, j-0.4, j+0.4, j+0.4, j-0.4, j-0.4, j+0.4, j+0.4],
                    z=[0, 0, 0, 0, z_val, z_val, z_val, z_val],
                    i=[0, 0, 0, 1, 4, 4, 4, 5, 5, 6, 6, 7],
                    j=[1, 2, 3, 2, 5, 6, 7, 6, 1, 7, 2, 3],
                    k=[2, 3, 4, 3, 6, 7, 0, 7, 6, 4, 3, 4],
                    intensity=[z_val] * 8,
                    colorscale="Jet",
                    cmin=0, cmax=1,
                    showscale=(i == 0 and j == 0),
                    opacity=0.95,
                    hovertemplate=(
                        f"<b>Mode {i+1} vs Mode {j+1}</b><br>"
                        f"{mode_labels[i] if i < len(mode_labels) else ''} vs "
                        f"{mode_labels[j] if j < len(mode_labels) else ''}<br>"
                        f"MAC = {z_val:.3f}<extra></extra>"
                    ),
                ))
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis=dict(title="Mode i",
                            tickvals=list(range(n)),
                            ticktext=mode_labels[:n]),
                yaxis=dict(title="Mode j",
                            tickvals=list(range(n)),
                            ticktext=mode_labels[:n]),
                zaxis=dict(title="MAC", range=[0, 1]),
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
            ),
            height=600,
            template="plotly_white",
            margin=dict(l=20, r=20, t=70, b=20),
        )
    else:
        # 2D heatmap fallback
        fig = go.Figure(data=go.Heatmap(
            z=M, x=mode_labels[:n], y=mode_labels[:n],
            colorscale="Jet", zmin=0, zmax=1,
            text=_np.round(M, 2), texttemplate="%{text}",
            textfont=dict(size=10),
            hovertemplate="Mode %{y} vs Mode %{x}<br>MAC = %{z:.3f}<extra></extra>",
        ))
        fig.update_layout(
            title=title,
            xaxis_title="Mode j", yaxis_title="Mode i",
            height=500, template="plotly_white",
            margin=dict(l=80, r=20, t=70, b=80),
        )
    return fig


def build_campbell_diagram(
    natural_frequencies_hz: List[float],
    natural_freq_labels: Optional[List[str]] = None,
    rpm_min: float = 0.0,
    rpm_max: float = 4000.0,
    operating_rpm: Optional[float] = None,
    n_orders: int = 6,
    classification: Optional[List[str]] = None,
    title: str = "Diagrama de Campbell",
):
    """
    Genera el Diagrama de Campbell — cruza modos identificados vs velocidad
    operativa para identificar velocidades críticas (intersecciones modo ↔ orden).

    Estándar en rotodinámica (API 684 §1.6).

    Args:
        natural_frequencies_hz: lista de frecuencias naturales identificadas (Hz)
        natural_freq_labels: etiquetas opcionales ("Modo 1", "Modo 2", ...)
        rpm_min, rpm_max: rango de velocidad operativa a mostrar
        operating_rpm: si se da, dibuja vline destacada en esa velocidad
        n_orders: número de armónicas a dibujar (1×, 2×, ..., N×)
        classification: lista paralela a natural_frequencies_hz con
          "natural" | "harmonic" | "spurious" para colorear

    Returns:
        plotly Figure con líneas horizontales (modos) + líneas inclinadas (órdenes)
    """
    try:
        import plotly.graph_objects as go
        import numpy as _np
    except ImportError as exc:
        raise ImportError("plotly requerido") from exc

    if natural_freq_labels is None:
        natural_freq_labels = [f"Modo {i+1}" for i in range(len(natural_frequencies_hz))]
    if classification is None:
        classification = ["natural"] * len(natural_frequencies_hz)

    fig = go.Figure()

    rpm_range = _np.linspace(rpm_min, rpm_max, 50)
    # Órdenes de velocidad — líneas inclinadas (y = order * rpm/60)
    for order in range(1, n_orders + 1):
        y_order = order * rpm_range / 60.0
        fig.add_trace(go.Scatter(
            x=rpm_range, y=y_order,
            mode="lines",
            name=f"{order}× rpm",
            line=dict(color="#6B7280", width=1, dash="dot"),
            hovertemplate=f"{order}× rpm<br>%{{x:.0f}} rpm → %{{y:.1f}} Hz<extra></extra>",
        ))
        # Etiqueta al final de cada línea
        fig.add_annotation(
            x=rpm_max * 0.97, y=y_order[-1],
            text=f"{order}×", showarrow=False,
            font=dict(size=10, color="#6B7280"),
        )

    # Líneas horizontales = modos identificados
    class_colors = {
        "natural": "#16a34a",
        "harmonic": "#dc2626",
        "spurious": "#9ca3af",
    }
    for fn, label, cls in zip(natural_frequencies_hz, natural_freq_labels, classification):
        color = class_colors.get(cls, "#0F7FB0")
        fig.add_trace(go.Scatter(
            x=[rpm_min, rpm_max], y=[fn, fn],
            mode="lines",
            name=f"{label} · {fn:.1f} Hz",
            line=dict(color=color, width=2),
            hovertemplate=f"{label}<br>fn = {fn:.2f} Hz<extra></extra>",
        ))

    # Línea vertical en operating speed (si se da)
    if operating_rpm and rpm_min <= operating_rpm <= rpm_max:
        fig.add_vline(
            x=operating_rpm,
            line=dict(color="#D89B22", width=2, dash="dash"),
            annotation_text=f"Operativa: {operating_rpm:.0f} rpm",
            annotation_position="top",
            annotation_font_color="#D89B22",
        )

    # Detectar intersecciones (velocidades críticas)
    # Para cada modo natural, encontrar dónde cruza con cada orden
    critical_speeds = []
    for fn, label, cls in zip(natural_frequencies_hz, natural_freq_labels, classification):
        if cls != "natural":
            continue
        for order in range(1, n_orders + 1):
            # y = order * rpm/60 = fn → rpm = 60 * fn / order
            critical_rpm = 60.0 * fn / order
            if rpm_min <= critical_rpm <= rpm_max:
                critical_speeds.append((critical_rpm, fn, order, label))
                fig.add_trace(go.Scatter(
                    x=[critical_rpm], y=[fn],
                    mode="markers",
                    marker=dict(color="#dc2626", size=10, symbol="x",
                                 line=dict(width=2, color="#7f1d1d")),
                    name=f"Crítica {critical_rpm:.0f} rpm",
                    showlegend=False,
                    hovertemplate=(f"<b>Velocidad crítica</b><br>"
                                     f"{critical_rpm:.0f} rpm<br>"
                                     f"{label} cruza con {order}× rpm<br>"
                                     f"fn = {fn:.2f} Hz<extra></extra>"),
                ))

    fig.update_layout(
        title=title,
        xaxis_title="Velocidad operativa (rpm)",
        yaxis_title="Frecuencia (Hz)",
        height=500,
        template="plotly_white",
        margin=dict(l=60, r=40, t=70, b=50),
        showlegend=True,
        legend=dict(orientation="h", y=-0.18, x=0),
        hovermode="closest",
    )
    fig.update_xaxes(range=[rpm_min, rpm_max])
    fig.update_yaxes(range=[0, max(natural_frequencies_hz) * 1.3 if natural_frequencies_hz else 200])

    return fig, critical_speeds


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
