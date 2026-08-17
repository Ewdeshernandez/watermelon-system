"""
core/balance/rotor3d.py — Rotor 3D (vista fija) con vector de vibración + contrapeso
====================================================================================

Figura Plotly 3D de un rotor con su(s) plano(s) de balanceo. Diseño:

  - Eje limpio (sin apoyos) para no restarle protagonismo a lo importante.
  - **Vector de vibración** (rojo) en cada plano: aparece apenas se carga el
    dato medido (magnitud + ángulo), antes de calcular.
  - **Contrapeso** (diamante del color del plano) en el aro: aparece con la
    posición angular al calcular la corrección.
  - **Vista fija tipo imagen**: la página la renderiza con staticPlot=True
    (no se puede rotar/mover/zoom).

Convención angular: 0° arriba (TDC). Posición a ángulo θ:
    y = R·sin(θ),  z = R·cos(θ)   (θ=0 → arriba).

Headless (no depende de Streamlit).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go


_NAVY = "#0F1E3D"
_SHAFT = "#64748b"
_VIB = "#ef4444"          # vibración (rojo)
_COLORS = {"cyan": "#1AAEE5", "green": "#16a34a", "amber": "#D89B22"}


def _circle(xp: float, R: float, n: int = 40):
    th = np.linspace(0, 2 * np.pi, n)
    return np.full(n, xp), R * np.sin(th), R * np.cos(th)


def _disk_mesh(xp: float, R: float, color: str, n: int = 32) -> go.Mesh3d:
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ys = np.concatenate([[0.0], R * np.sin(th)])
    zs = np.concatenate([[0.0], R * np.cos(th)])
    xs = np.full(len(ys), xp)
    i = [0] * n
    j = list(range(1, n + 1))
    k = list(range(2, n + 1)) + [1]
    return go.Mesh3d(x=xs, y=ys, z=zs, i=i, j=j, k=k, color=color,
                     opacity=0.20, hoverinfo="skip", showscale=False)


def rotor_3d_figure(planes: List[Dict[str, Any]],
                    shaft_len: float = 10.0, R: float = 1.0,
                    height: int = 340) -> go.Figure:
    """Rotor 3D con vector de vibración y/o contrapeso por plano.

    planes: lista de dicts con:
        name, x_frac (0..1), color ("cyan"/"green"),
        vib: (mag, ang) | None      → flecha roja de vibración
        vib_unit: str
        weight_ang: float | None    → posición del contrapeso
        weight_label: str           → ej. "47.2 g"
    """
    fig = go.Figure()
    rs = R * 0.15  # eje fino y limpio

    # --- Eje (cilindro) ---
    theta = np.linspace(0, 2 * np.pi, 24)
    xline = np.linspace(0, shaft_len, 2)
    th, X = np.meshgrid(theta, xline)
    fig.add_trace(go.Surface(
        x=X, y=rs * np.sin(th), z=rs * np.cos(th), showscale=False, opacity=1.0,
        colorscale=[[0, _SHAFT], [1, _SHAFT]], hoverinfo="skip",
        lighting=dict(ambient=0.62, diffuse=0.72, specular=0.15)))

    # escala común de los vectores de vibración (el mayor ocupa ~0.82·R)
    vmax = max([float(p["vib"][0]) for p in planes
                if p.get("vib") and p["vib"][0] and float(p["vib"][0]) > 0],
               default=0.0)

    for pl in planes:
        xp = float(pl.get("x_frac", 0.5)) * shaft_len
        color = _COLORS.get(pl.get("color", "cyan"), _COLORS["cyan"])
        name = pl.get("name", "Plano")

        # disco + aro
        fig.add_trace(_disk_mesh(xp, R, color))
        cx, cy, cz = _circle(xp, R)
        fig.add_trace(go.Scatter3d(x=cx, y=cy, z=cz, mode="lines",
                                   line=dict(color=color, width=5),
                                   hoverinfo="skip", showlegend=False))
        # ticks 0/90/180/270
        for a, lbl in [(0, "0°"), (90, "90°"), (180, "180°"), (270, "270°")]:
            ar = np.deg2rad(a)
            fig.add_trace(go.Scatter3d(
                x=[xp], y=[R * 1.17 * np.sin(ar)], z=[R * 1.17 * np.cos(ar)],
                mode="text", text=[lbl], textfont=dict(size=9, color="#94a3b8"),
                hoverinfo="skip", showlegend=False))

        # --- Vector de vibración (rojo) ---
        vib = pl.get("vib")
        if vib and vib[0] and float(vib[0]) > 0 and vmax > 0:
            vmag, vang = float(vib[0]), float(vib[1])
            ar = np.deg2rad(vang)
            L = R * 0.82 * (vmag / vmax)
            vy, vz = L * np.sin(ar), L * np.cos(ar)
            fig.add_trace(go.Scatter3d(
                x=[xp, xp], y=[0, vy], z=[0, vz], mode="lines",
                line=dict(color=_VIB, width=6), hoverinfo="skip", showlegend=False))
            fig.add_trace(go.Scatter3d(
                x=[xp], y=[vy], z=[vz], mode="markers+text",
                marker=dict(size=6, color=_VIB),
                text=[f"  {vmag:.2f} {pl.get('vib_unit', '')} ∠{vang:.0f}°"],
                textposition="middle right", textfont=dict(size=10, color=_VIB),
                hoverinfo="skip", showlegend=False))

        # --- Contrapeso (diamante) ---
        wa = pl.get("weight_ang")
        if wa is not None:
            ar = np.deg2rad(float(wa))
            wy, wz = R * np.sin(ar), R * np.cos(ar)
            fig.add_trace(go.Scatter3d(
                x=[xp, xp], y=[0, wy], z=[0, wz], mode="lines",
                line=dict(color=color, width=7), hoverinfo="skip", showlegend=False))
            fig.add_trace(go.Scatter3d(
                x=[xp], y=[wy], z=[wz], mode="markers+text",
                marker=dict(size=10, color=color, symbol="diamond",
                            line=dict(color="white", width=1)),
                text=[f"  {pl.get('weight_label', '')}"], textposition="top center",
                textfont=dict(size=12, color=_NAVY),
                hoverinfo="skip", showlegend=False))

        # etiqueta del plano
        fig.add_trace(go.Scatter3d(
            x=[xp], y=[0], z=[-R * 1.42], mode="text", text=[f"<b>{name}</b>"],
            textfont=dict(size=11, color=_NAVY), hoverinfo="skip", showlegend=False))

    fig.update_layout(
        height=height, margin=dict(l=0, r=0, t=6, b=0),
        paper_bgcolor="rgba(0,0,0,0)", showlegend=False, uirevision="fixed",
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            zaxis=dict(visible=False), aspectmode="manual",
            aspectratio=dict(x=2.4, y=1, z=1),
            # Vista fija: sin drag (giro) en la escena 3D.
            dragmode=False,
            camera=dict(eye=dict(x=1.45, y=1.6, z=0.95),
                        projection=dict(type="orthographic")),
        ),
    )
    return fig


def build_planes_1p(vib: Optional[Tuple[float, float]], vib_unit: str,
                    weight_ang: Optional[float], weight_label: str) -> List[Dict[str, Any]]:
    return [{
        "name": "Plano de corrección", "x_frac": 0.5, "color": "cyan",
        "vib": vib, "vib_unit": vib_unit,
        "weight_ang": weight_ang, "weight_label": weight_label,
    }]


def build_planes_2p(vibA, vibB, vib_unit, waA, wlA, waB, wlB) -> List[Dict[str, Any]]:
    return [
        {"name": "Plano A", "x_frac": 0.30, "color": "cyan", "vib": vibA,
         "vib_unit": vib_unit, "weight_ang": waA, "weight_label": wlA},
        {"name": "Plano B", "x_frac": 0.70, "color": "green", "vib": vibB,
         "vib_unit": vib_unit, "weight_ang": waB, "weight_label": wlB},
    ]


__all__ = ["rotor_3d_figure", "build_planes_1p", "build_planes_2p"]
