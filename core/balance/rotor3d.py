"""
core/balance/rotor3d.py — Rotor 3D con plano(s) de corrección
=============================================================

Figura Plotly 3D de un rotor entre cojinetes, mostrando el/los plano(s) de
corrección como discos y, en cada uno, la posición angular donde va el peso
(ángulo + gramos). Da al analista una lectura inmediata de "dónde poner el
peso" — el diferencial visual frente a Bently/Emerson/SKF.

Convención angular: 0° arriba (TDC), sentido del giro definido en campo.
Posición del peso a ángulo θ:  y = R·sin(θ),  z = R·cos(θ)  (θ=0 → arriba).

Headless (no depende de Streamlit). La página hace st.plotly_chart(fig).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import plotly.graph_objects as go


# Paleta consistente con core.balance.ui
_NAVY = "#0F1E3D"
_SHAFT = "#475569"
_COLORS = {"cyan": "#1AAEE5", "green": "#16a34a", "amber": "#D89B22"}


def _circle(xp: float, R: float, n: int = 60):
    th = np.linspace(0, 2 * np.pi, n)
    y = R * np.sin(th)
    z = R * np.cos(th)
    x = np.full(n, xp)
    return x, y, z


def _disk_mesh(xp: float, R: float, color: str, n: int = 48) -> go.Mesh3d:
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ys = np.concatenate([[0.0], R * np.sin(th)])
    zs = np.concatenate([[0.0], R * np.cos(th)])
    xs = np.full(len(ys), xp)
    i, j, k = [], [], []
    for p in range(1, n):
        i.append(0); j.append(p); k.append(p + 1)
    i.append(0); j.append(n); k.append(1)  # cierra el abanico
    return go.Mesh3d(x=xs, y=ys, z=zs, i=i, j=j, k=k, color=color,
                     opacity=0.22, hoverinfo="skip", showscale=False)


def rotor_3d_figure(planes: List[Dict[str, Any]],
                    shaft_len: float = 10.0, R: float = 1.0,
                    height: int = 380) -> go.Figure:
    """Rotor 3D con los planos de corrección.

    planes: [{"name", "x_frac"(0..1), "angle_deg", "weight_label", "color"}]
    """
    fig = go.Figure()
    rs = R * 0.22  # radio del eje

    # --- Eje (cilindro) ---
    theta = np.linspace(0, 2 * np.pi, 40)
    xline = np.linspace(0, shaft_len, 2)
    th, X = np.meshgrid(theta, xline)
    Ys = rs * np.sin(th); Zs = rs * np.cos(th)
    fig.add_trace(go.Surface(
        x=X, y=Ys, z=Zs, showscale=False, opacity=1.0,
        colorscale=[[0, _SHAFT], [1, _SHAFT]], hoverinfo="skip",
        lighting=dict(ambient=0.6, diffuse=0.7, specular=0.2),
    ))

    # --- Cojinetes (bloques en los extremos) ---
    for xb in (0.0, shaft_len):
        fig.add_trace(go.Scatter3d(
            x=[xb], y=[0], z=[-R * 1.25], mode="markers",
            marker=dict(size=9, color=_NAVY, symbol="square"),
            hovertext="Cojinete", hoverinfo="text", showlegend=False))

    # --- Planos de corrección ---
    for pl in planes:
        xp = float(pl.get("x_frac", 0.5)) * shaft_len
        color = _COLORS.get(pl.get("color", "cyan"), _COLORS["cyan"])
        ang = pl.get("angle_deg")
        wlabel = pl.get("weight_label", "")
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
                x=[xp], y=[R * 1.16 * np.sin(ar)], z=[R * 1.16 * np.cos(ar)],
                mode="text", text=[lbl],
                textfont=dict(size=9, color="#94a3b8"),
                hoverinfo="skip", showlegend=False))

        # marcador del peso de corrección (si hay ángulo)
        if ang is not None:
            ar = np.deg2rad(float(ang))
            wy, wz = R * np.sin(ar), R * np.cos(ar)
            fig.add_trace(go.Scatter3d(
                x=[xp, xp], y=[0, wy], z=[0, wz], mode="lines",
                line=dict(color=color, width=7), hoverinfo="skip", showlegend=False))
            fig.add_trace(go.Scatter3d(
                x=[xp], y=[wy], z=[wz], mode="markers+text",
                marker=dict(size=9, color=color, symbol="diamond",
                            line=dict(color="white", width=1)),
                text=[f"  {wlabel}"], textposition="top center",
                textfont=dict(size=12, color=_NAVY),
                hovertext=f"{name}: {wlabel} ∠ {float(ang):.0f}°",
                hoverinfo="text", showlegend=False))

        # etiqueta del plano (bajo el disco)
        fig.add_trace(go.Scatter3d(
            x=[xp], y=[0], z=[-R * 1.55], mode="text", text=[f"<b>{name}</b>"],
            textfont=dict(size=11, color=_NAVY), hoverinfo="skip", showlegend=False))

    fig.update_layout(
        height=height, margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            zaxis=dict(visible=False), aspectmode="manual",
            aspectratio=dict(x=2.4, y=1, z=1),
            camera=dict(eye=dict(x=1.6, y=1.7, z=1.1)),
        ),
        showlegend=False,
    )
    return fig


def rotor_3d_1plane(angle_deg: Optional[float], weight_label: str) -> go.Figure:
    return rotor_3d_figure([{
        "name": "Plano de corrección", "x_frac": 0.5,
        "angle_deg": angle_deg, "weight_label": weight_label, "color": "cyan",
    }])


def rotor_3d_2plane(a_ang: Optional[float], a_w: str,
                    b_ang: Optional[float], b_w: str) -> go.Figure:
    return rotor_3d_figure([
        {"name": "Plano A", "x_frac": 0.30, "angle_deg": a_ang,
         "weight_label": a_w, "color": "cyan"},
        {"name": "Plano B", "x_frac": 0.70, "angle_deg": b_ang,
         "weight_label": b_w, "color": "green"},
    ])


__all__ = ["rotor_3d_figure", "rotor_3d_1plane", "rotor_3d_2plane"]
