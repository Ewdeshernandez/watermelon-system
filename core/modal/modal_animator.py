"""
core/modal/modal_animator.py — Animación 3D de mode shapes
============================================================

Genera animaciones interactivas Plotly Mesh3d que muestran cómo se deforma
la geometría 3D del activo para cada modo natural identificado.

Este es el "wow factor" que hace que Watermelon Modal compita visualmente
con Artemis Modal. Tres niveles de fidelidad:

Nivel 1 — Bar chart 2D
  Por cada DOF, muestra magnitud + fase como barras. Útil para validación
  técnica pero no impresionante visualmente.

Nivel 2 — Wireframe 3D con flechas
  Cada sensor es una flecha 3D escalada por la magnitud del modo y orientada
  según la fase. Plotly Cone3D + Scatter3d.

Nivel 3 — Mesh3D animado con colormap (objetivo V1)
  · Mesh3D con vertices = nodes de la geometría
  · Interpolación de modal amplitude entre los DOFs medidos
  · Animación frame-by-frame: 30-60 frames por ciclo
  · Colormap: rojo = max desplazamiento, azul = mínimo
  · Export como GIF/MP4 para inclusión en reportes PDF

Norma aplicable
---------------
ISO 7626-6 §7.2 — Visualización de mode shapes. Requiere indicación clara
de la fase y la magnitud relativa entre puntos de medición.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List
import numpy as np


@dataclass
class AnimationConfig:
    """Parámetros para la animación 3D."""
    n_frames: int = 60        # Frames por ciclo completo
    fps: int = 30             # Frames por segundo
    amplitude_scale: float = 1.0  # Factor de escala para hacer visible la deformación
    colormap: str = "RdBu_r"  # Plotly colorscale
    show_undeformed: bool = True  # Muestra geometría original como referencia


def build_bar_chart_2d(mode_shape: np.ndarray, dof_labels: List[str]):
    """
    Nivel 1 — Mode shape como bar chart de magnitud + fase.

    Returns:
        plotly.graph_objects.Figure
    """
    raise NotImplementedError("Fase scaffolding")


def build_wireframe_3d_arrows(
    geometry,  # Wireframe3D
    mode_shape: np.ndarray,
    dof_positions: np.ndarray,
    dof_directions: np.ndarray,
    config: AnimationConfig = AnimationConfig(),
):
    """
    Nivel 2 — Wireframe con flechas vectoriales en cada sensor.

    Returns:
        plotly.graph_objects.Figure (estático, no animado)
    """
    raise NotImplementedError("Fase scaffolding")


def build_mesh3d_animation(
    geometry,  # Wireframe3D
    mode_shape: np.ndarray,
    dof_positions: np.ndarray,
    config: AnimationConfig = AnimationConfig(),
):
    """
    Nivel 3 — Animación completa Mesh3D con colormap.

    Args:
        geometry: Wireframe3D con nodes + faces
        mode_shape: Vector complejo de N DOFs
        dof_positions: Matriz (N_DOFs, 3) con posiciones xyz de cada sensor
        config: Parámetros de animación

    Returns:
        plotly.graph_objects.Figure con frames de animación
    """
    raise NotImplementedError("Fase scaffolding")


def export_animation_gif(fig, output_path: str, duration_s: float = 3.0):
    """
    Exporta la animación como GIF para inclusión en reportes.

    Args:
        fig: Figura Plotly con frames
        output_path: Ruta del GIF a generar
        duration_s: Duración total del GIF (loop)
    """
    raise NotImplementedError("Fase scaffolding")


def export_animation_mp4(fig, output_path: str, duration_s: float = 3.0):
    """Exporta la animación como MP4 (requiere ffmpeg)."""
    raise NotImplementedError("Fase scaffolding")
