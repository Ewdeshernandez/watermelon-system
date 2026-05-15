"""
core/modal/geometry_3d.py — Geometria 3D para analisis modal
=============================================================

Define la representacion de la geometria 3D del activo (compresor, generador,
turbina, etc.) que se usa como soporte visual para los mode shapes y como
editor estilo Artemis Modal.

Modelo de datos (v3.31.170)
---------------------------
Un `ModalGeometry` consiste en:
  - `blocks`: secciones del tren mecanico (motor, coupling, compresor)
    rendereadas como cilindros o cajas via Plotly Mesh3d
  - `shaft_*`: eje longitudinal — linea fina central
  - `sensors`: puntos de medicion con direccion DOF (+X/-X/+Y/-Y/+Z/-Z)
    rendereados como Scatter3d marker + Cone para la flecha de direccion

Persistencia
------------
JSON en `data/modal/geometries/<asset_id>.json`. Si el usuario hace analisis
ad-hoc sin activo registrado, la geometria vive en session_state.

Norma aplicable
---------------
ISO 7626-6 §6 — Identificacion de DOF y orientacion espacial de los puntos
de medicion debe ser claramente documentada en el reporte modal.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple
import json
import math


# ---------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------

@dataclass
class GeometryBlock:
    """Seccion mecanica del tren — cilindro o caja."""
    id: str
    name: str
    shape: Literal["cylinder", "box"] = "cylinder"
    x_start: float = 0.0
    x_end: float = 100.0
    radius: float = 100.0       # solo para cylinder
    half_width: float = 100.0   # solo para box (Y)
    half_height: float = 100.0  # solo para box (Z)
    color: str = "#475569"
    opacity: float = 0.35


@dataclass
class GeometrySensor:
    """Punto de medicion con direccion de DOF."""
    id: str
    name: str
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    dof: str = "+Y"   # "+X" "-X" "+Y" "-Y" "+Z" "-Z"
    sensor_type: str = "accelerometer"  # accelerometer | proximity | velocity


@dataclass
class ModalGeometry:
    """Geometria 3D completa del activo."""
    asset_id: str = ""
    name: str = "Activo sin nombre"
    units: str = "mm"
    blocks: List[GeometryBlock] = field(default_factory=list)
    sensors: List[GeometrySensor] = field(default_factory=list)
    shaft_radius: float = 30.0
    shaft_start: float = 0.0
    shaft_end: float = 1000.0
    shaft_color: str = "#0F1E3D"

    # ----- serialization -----
    def to_dict(self) -> Dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "name": self.name,
            "units": self.units,
            "shaft_radius": self.shaft_radius,
            "shaft_start": self.shaft_start,
            "shaft_end": self.shaft_end,
            "shaft_color": self.shaft_color,
            "blocks": [asdict(b) for b in self.blocks],
            "sensors": [asdict(s) for s in self.sensors],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModalGeometry":
        return cls(
            asset_id=data.get("asset_id", ""),
            name=data.get("name", "Activo sin nombre"),
            units=data.get("units", "mm"),
            shaft_radius=float(data.get("shaft_radius", 30.0)),
            shaft_start=float(data.get("shaft_start", 0.0)),
            shaft_end=float(data.get("shaft_end", 1000.0)),
            shaft_color=data.get("shaft_color", "#0F1E3D"),
            blocks=[GeometryBlock(**b) for b in data.get("blocks", [])],
            sensors=[GeometrySensor(**s) for s in data.get("sensors", [])],
        )


# ---------------------------------------------------------------------
# Persistencia
# ---------------------------------------------------------------------

_GEOMETRY_DIR = Path("data/modal/geometries")


def _path_for(asset_id: str) -> Path:
    safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in asset_id)
    return _GEOMETRY_DIR / f"{safe_id}.json"


def save_geometry(geom: ModalGeometry) -> Path:
    if not geom.asset_id:
        raise ValueError("save_geometry requiere asset_id no vacio")
    _GEOMETRY_DIR.mkdir(parents=True, exist_ok=True)
    p = _path_for(geom.asset_id)
    with open(p, "w") as f:
        f.write(geom.to_json())
    return p


def load_geometry(asset_id: str) -> Optional[ModalGeometry]:
    p = _path_for(asset_id)
    if not p.exists():
        return None
    with open(p, "r") as f:
        return ModalGeometry.from_dict(json.load(f))


# ---------------------------------------------------------------------
# Templates pre-cargados (mas comunes en O&G y power gen)
# ---------------------------------------------------------------------

def template_motor_compressor() -> ModalGeometry:
    """Motor electrico + coupling + compresor centrifugo (gas process).

    Layout industrial tipico:
      - Motor: 2 acelerometros (DE = lado acople, NDE = lado libre)
      - Compresor: 4 proxies ortogonales (CE = lado acople X+Y, NCE = lado libre X+Y)
    """
    return ModalGeometry(
        name="Motor + Compresor",
        shaft_start=0.0,
        shaft_end=1600.0,
        shaft_radius=40.0,
        blocks=[
            GeometryBlock(id="motor", name="Motor",
                          shape="box", x_start=0.0, x_end=500.0,
                          half_width=250.0, half_height=250.0,
                          color="#0F1E3D", opacity=0.35),
            GeometryBlock(id="coupling", name="Coupling",
                          shape="cylinder", x_start=500.0, x_end=650.0,
                          radius=90.0, color="#D89B22", opacity=0.55),
            GeometryBlock(id="compressor", name="Compresor",
                          shape="cylinder", x_start=650.0, x_end=1600.0,
                          radius=300.0, color="#1AAEE5", opacity=0.30),
        ],
        sensors=[
            # Motor — 2 acelerometros (NDE + DE)
            GeometrySensor(id="s1", name="MOT-NDE", x=30.0, y=250.0, z=0.0,
                            dof="+Y", sensor_type="accelerometer"),
            GeometrySensor(id="s2", name="MOT-DE",  x=470.0, y=250.0, z=0.0,
                            dof="+Y", sensor_type="accelerometer"),
            # Compresor — 4 proximidades ortogonales
            GeometrySensor(id="s3", name="COMP-CE-Y",  x=700.0, y=300.0, z=0.0,
                            dof="+Y", sensor_type="proximity"),
            GeometrySensor(id="s4", name="COMP-CE-X",  x=700.0, y=0.0,   z=300.0,
                            dof="+Z", sensor_type="proximity"),
            GeometrySensor(id="s5", name="COMP-NCE-Y", x=1570.0, y=300.0, z=0.0,
                            dof="+Y", sensor_type="proximity"),
            GeometrySensor(id="s6", name="COMP-NCE-X", x=1570.0, y=0.0,   z=300.0,
                            dof="+Z", sensor_type="proximity"),
        ],
    )


def template_turbine_generator() -> ModalGeometry:
    """Turbina de gas aeroderivada (LM6000) + coupling + generador (Brush).

    Layout TES1 SIGA:
      - Turbina: 2 acelerometros (CRF = lado libre, TRF = lado acople)
      - Generador: 4 proxies ortogonales (CE X+Y lado acople, NCE X+Y lado libre)
    """
    return ModalGeometry(
        name="Turbina + Generador (LM6000 + Brush)",
        shaft_start=0.0,
        shaft_end=2200.0,
        shaft_radius=60.0,
        blocks=[
            GeometryBlock(id="turbine", name="Turbina LM6000",
                          shape="cylinder", x_start=0.0, x_end=1000.0,
                          radius=380.0, color="#0F1E3D", opacity=0.30),
            GeometryBlock(id="coupling", name="Coupling",
                          shape="cylinder", x_start=1000.0, x_end=1150.0,
                          radius=100.0, color="#D89B22", opacity=0.55),
            GeometryBlock(id="generator", name="Generador Brush",
                          shape="box", x_start=1150.0, x_end=2200.0,
                          half_width=320.0, half_height=350.0,
                          color="#1AAEE5", opacity=0.30),
        ],
        sensors=[
            # Turbina — 2 acelerometros (CRF lado libre + TRF lado acople)
            GeometrySensor(id="s1", name="TRB-CRF", x=80.0,  y=380.0, z=0.0,
                            dof="+Y", sensor_type="accelerometer"),
            GeometrySensor(id="s2", name="TRB-TRF", x=950.0, y=380.0, z=0.0,
                            dof="+Y", sensor_type="accelerometer"),
            # Generador — 4 proxies ortogonales en ambos bearings
            GeometrySensor(id="s3", name="GEN-CE-Y",  x=1200.0, y=320.0, z=0.0,
                            dof="+Y", sensor_type="proximity"),
            GeometrySensor(id="s4", name="GEN-CE-X",  x=1200.0, y=0.0,   z=350.0,
                            dof="+Z", sensor_type="proximity"),
            GeometrySensor(id="s5", name="GEN-NCE-Y", x=2150.0, y=320.0, z=0.0,
                            dof="+Y", sensor_type="proximity"),
            GeometrySensor(id="s6", name="GEN-NCE-X", x=2150.0, y=0.0,   z=350.0,
                            dof="+Z", sensor_type="proximity"),
        ],
    )


def template_pump_motor() -> ModalGeometry:
    """Motor electrico + coupling + bomba centrifuga.

    Layout industrial tipico:
      - Motor: 2 acelerometros (DE + NDE)
      - Bomba: 2 proxies (DE Y + DE X)
    """
    return ModalGeometry(
        name="Bomba + Motor",
        shaft_start=0.0,
        shaft_end=1300.0,
        shaft_radius=30.0,
        blocks=[
            GeometryBlock(id="motor", name="Motor",
                          shape="box", x_start=0.0, x_end=500.0,
                          half_width=200.0, half_height=200.0,
                          color="#0F1E3D", opacity=0.35),
            GeometryBlock(id="coupling", name="Coupling",
                          shape="cylinder", x_start=500.0, x_end=600.0,
                          radius=70.0, color="#D89B22", opacity=0.55),
            GeometryBlock(id="pump", name="Bomba",
                          shape="cylinder", x_start=600.0, x_end=1300.0,
                          radius=230.0, color="#1AAEE5", opacity=0.30),
        ],
        sensors=[
            GeometrySensor(id="s1", name="MOT-NDE",  x=30.0,  y=200.0, z=0.0,
                            dof="+Y", sensor_type="accelerometer"),
            GeometrySensor(id="s2", name="MOT-DE",   x=470.0, y=200.0, z=0.0,
                            dof="+Y", sensor_type="accelerometer"),
            GeometrySensor(id="s3", name="PMP-DE-Y", x=650.0, y=230.0, z=0.0,
                            dof="+Y", sensor_type="proximity"),
            GeometrySensor(id="s4", name="PMP-DE-X", x=650.0, y=0.0,   z=230.0,
                            dof="+Z", sensor_type="proximity"),
        ],
    )


TEMPLATES: Dict[str, Any] = {
    "motor_compressor": template_motor_compressor,
    "turbine_generator": template_turbine_generator,
    "pump_motor": template_pump_motor,
}


# ---------------------------------------------------------------------
# Plotly renderer
# ---------------------------------------------------------------------

def _cylinder_mesh(x0: float, x1: float, radius: float,
                    n_theta: int = 24) -> Tuple[list, list, list, list, list, list]:
    """Genera vertices (x,y,z) + faces (i,j,k) para un cilindro alineado a X."""
    import numpy as np
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    # bottom ring (x0) + top ring (x1) + 2 center points
    xs, ys, zs = [], [], []
    for t in theta:
        ys.append(radius * math.cos(t))
        zs.append(radius * math.sin(t))
        xs.append(x0)
    for t in theta:
        ys.append(radius * math.cos(t))
        zs.append(radius * math.sin(t))
        xs.append(x1)
    # centers
    xs += [x0, x1]
    ys += [0.0, 0.0]
    zs += [0.0, 0.0]
    n = n_theta
    c0, c1 = 2 * n, 2 * n + 1
    i, j, k = [], [], []
    # side quads as 2 triangles
    for a in range(n):
        b = (a + 1) % n
        i += [a, a]
        j += [b, n + b]
        k += [n + b, n + a]
    # bottom cap
    for a in range(n):
        b = (a + 1) % n
        i.append(a); j.append(b); k.append(c0)
    # top cap
    for a in range(n):
        b = (a + 1) % n
        i.append(n + a); j.append(c1); k.append(n + b)
    return xs, ys, zs, i, j, k


def _box_mesh(x0: float, x1: float, hw: float,
               hh: float) -> Tuple[list, list, list, list, list, list]:
    """Caja rectangular alineada a X (Y=hw, Z=hh)."""
    xs = [x0, x1, x1, x0, x0, x1, x1, x0]
    ys = [-hw, -hw, hw, hw, -hw, -hw, hw, hw]
    zs = [-hh, -hh, -hh, -hh, hh, hh, hh, hh]
    # 12 triangles (2 per face)
    i = [0, 0, 1, 1, 4, 4, 5, 5, 0, 0, 2, 2]
    j = [1, 2, 2, 5, 5, 7, 6, 1, 4, 7, 3, 7]
    k = [2, 3, 5, 6, 6, 4, 7, 0, 7, 4, 7, 6]
    return xs, ys, zs, i, j, k


def _dof_to_vector(dof: str) -> Tuple[float, float, float]:
    mapping = {
        "+X": (1, 0, 0), "-X": (-1, 0, 0),
        "+Y": (0, 1, 0), "-Y": (0, -1, 0),
        "+Z": (0, 0, 1), "-Z": (0, 0, -1),
    }
    return mapping.get(dof, (0, 1, 0))


def build_geometry_figure(geom: ModalGeometry,
                            arrow_size: Optional[float] = None,
                            show_legend: bool = True):
    """Construye una figura Plotly 3D del activo + sensores + flechas DOF."""
    import plotly.graph_objects as go

    fig = go.Figure()

    # 1) Bloques
    for b in geom.blocks:
        if b.shape == "cylinder":
            xs, ys, zs, i, j, k = _cylinder_mesh(b.x_start, b.x_end, b.radius)
        else:
            xs, ys, zs, i, j, k = _box_mesh(b.x_start, b.x_end,
                                              b.half_width, b.half_height)
        fig.add_trace(go.Mesh3d(
            x=xs, y=ys, z=zs, i=i, j=j, k=k,
            color=b.color, opacity=b.opacity,
            name=b.name, showlegend=show_legend, hoverinfo="name",
            flatshading=True,
        ))

    # 2) Eje (shaft) — cilindro fino centrado
    xs, ys, zs, i, j, k = _cylinder_mesh(geom.shaft_start, geom.shaft_end,
                                           geom.shaft_radius, n_theta=16)
    fig.add_trace(go.Mesh3d(
        x=xs, y=ys, z=zs, i=i, j=j, k=k,
        color=geom.shaft_color, opacity=0.95,
        name="Eje", showlegend=show_legend, hoverinfo="name",
        flatshading=True,
    ))

    # 3) Sensores — Scatter3d + Cone para DOF
    if geom.sensors:
        sx = [s.x for s in geom.sensors]
        sy = [s.y for s in geom.sensors]
        sz = [s.z for s in geom.sensors]
        labels = [f"{s.name} ({s.dof})" for s in geom.sensors]
        # color por tipo de sensor
        color_map = {"accelerometer": "#16a34a",
                       "proximity": "#dc2626",
                       "velocity": "#1AAEE5"}
        colors = [color_map.get(s.sensor_type, "#16a34a") for s in geom.sensors]
        fig.add_trace(go.Scatter3d(
            x=sx, y=sy, z=sz, mode="markers+text",
            marker=dict(size=8, color=colors, line=dict(width=2, color="#0F1E3D")),
            text=[s.name for s in geom.sensors],
            textposition="top center",
            textfont=dict(size=11, color="#0F1E3D"),
            hovertext=labels, hoverinfo="text",
            name="Sensores", showlegend=show_legend,
        ))

        # Cones para DOF — tamano relativo al span del eje
        if arrow_size is None:
            arrow_size = max((geom.shaft_end - geom.shaft_start) * 0.06, 50.0)
        cone_u, cone_v, cone_w = [], [], []
        cone_x, cone_y, cone_z = [], [], []
        for s in geom.sensors:
            ux, uy, uz = _dof_to_vector(s.dof)
            cone_x.append(s.x + ux * arrow_size * 0.5)
            cone_y.append(s.y + uy * arrow_size * 0.5)
            cone_z.append(s.z + uz * arrow_size * 0.5)
            cone_u.append(ux * arrow_size)
            cone_v.append(uy * arrow_size)
            cone_w.append(uz * arrow_size)
        fig.add_trace(go.Cone(
            x=cone_x, y=cone_y, z=cone_z,
            u=cone_u, v=cone_v, w=cone_w,
            sizemode="absolute", sizeref=arrow_size,
            colorscale=[[0, "#D89B22"], [1, "#D89B22"]],
            showscale=False, name="DOF", showlegend=show_legend,
            hoverinfo="skip",
        ))

    # Layout
    _span_for_layout = max(geom.shaft_end - geom.shaft_start, 100.0)
    fig.update_layout(
        scene=dict(
            xaxis=dict(title=f"X ({geom.units})", showgrid=True,
                        gridcolor="#e5e7eb", zerolinecolor="#cbd5e1"),
            yaxis=dict(title=f"Y ({geom.units})", showgrid=True,
                        gridcolor="#e5e7eb", zerolinecolor="#cbd5e1"),
            zaxis=dict(title=f"Z ({geom.units})", showgrid=True,
                        gridcolor="#e5e7eb", zerolinecolor="#cbd5e1"),
            aspectmode="data",
            camera=dict(eye=dict(x=1.4, y=1.2, z=0.9)),
            bgcolor="#f8fafc",
        ),
        title=dict(text=geom.name, font=dict(size=14, color="#0F1E3D")),
        margin=dict(l=0, r=0, t=40, b=0),
        height=560,
        paper_bgcolor="white",
    )
    return fig


# ---------------------------------------------------------------------
# Mode shape overlay sobre la geometria
# ---------------------------------------------------------------------

def build_geometry_with_mode_shape(
    geom: "ModalGeometry",
    mode_shape: Any,
    channel_names: List[str],
    mode_label: str = "",
    animate: bool = False,
    n_frames: int = 24,
    frame_duration_ms: int = 60,
):
    """
    Construye una figura 3D con la geometria del activo + flechas de mode shape
    coloreadas por fase (verde cofase / rojo anti-fase) en cada sensor que
    haga match con un canal del FDD/EMA por nombre.

    Si animate=True, la figura incluye frames Plotly que oscilan las flechas
    segun cos(phase + omega·t), equivalente a la animacion Artemis Modal donde
    el mode shape se ve "pulsar" a la frecuencia del modo (frecuencia visual
    es escalada, no la real).

    Args:
        geom: ModalGeometry con sensores ya posicionados
        mode_shape: array complejo (N_channels,) del modo a visualizar
        channel_names: lista de nombres de canal del FDD result
        mode_label: texto descriptivo del modo
        animate: si True, agrega frames + Play button + slider de fase
        n_frames: numero de frames por ciclo (default 24 = ~15 fps fluido)
        frame_duration_ms: ms por frame (60 ms = ciclo de ~1.5 s a 24 frames)

    Returns:
        plotly.graph_objects.Figure
    """
    import numpy as np
    import plotly.graph_objects as go

    fig = go.Figure()

    # 1) Bloques (estructura del tren) — opacidad mas baja para que las flechas
    #    destaquen
    for b in geom.blocks:
        if b.shape == "cylinder":
            xs, ys, zs, i, j, k = _cylinder_mesh(b.x_start, b.x_end, b.radius)
        else:
            xs, ys, zs, i, j, k = _box_mesh(b.x_start, b.x_end,
                                              b.half_width, b.half_height)
        fig.add_trace(go.Mesh3d(
            x=xs, y=ys, z=zs, i=i, j=j, k=k,
            color=b.color, opacity=max(b.opacity * 0.6, 0.15),
            name=b.name, showlegend=False, hoverinfo="skip",
            flatshading=True,
        ))

    # 2) Eje
    xs, ys, zs, i, j, k = _cylinder_mesh(geom.shaft_start, geom.shaft_end,
                                           geom.shaft_radius, n_theta=16)
    fig.add_trace(go.Mesh3d(
        x=xs, y=ys, z=zs, i=i, j=j, k=k,
        color=geom.shaft_color, opacity=0.85,
        name="Eje", showlegend=False, hoverinfo="skip",
        flatshading=True,
    ))

    # 3) Matching geometria.sensors <-> channel_names (case-insensitive)
    by_name: Dict[str, GeometrySensor] = {
        s.name.strip().upper(): s for s in geom.sensors
    }
    arr = np.asarray(mode_shape, dtype=complex).flatten()

    matched_positions: List[Tuple[float, float, float]] = []
    matched_directions: List[Tuple[float, float, float]] = []
    matched_signed_mag: List[float] = []
    matched_names: List[str] = []
    matched_phases_deg: List[float] = []

    for idx, ch in enumerate(channel_names):
        key = ch.strip().upper()
        if key not in by_name or idx >= len(arr):
            continue
        s = by_name[key]
        complex_val = complex(arr[idx])
        amp = float(abs(complex_val))
        phase_deg = float(math.degrees(math.atan2(complex_val.imag, complex_val.real)))
        # Signo de proyeccion sobre el eje real (fase ~0° = positivo, ~180° = negativo)
        sign = 1.0 if abs(phase_deg) <= 90.0 else -1.0
        ux, uy, uz = _dof_to_vector(s.dof)
        matched_positions.append((s.x, s.y, s.z))
        matched_directions.append((ux * sign, uy * sign, uz * sign))
        matched_signed_mag.append(amp * sign)
        matched_names.append(s.name)
        matched_phases_deg.append(phase_deg)

    if not matched_positions:
        # No matches — dejar la geometria sin flechas y un title de aviso
        fig.update_layout(
            title=dict(text=f"{mode_label}<br>"
                              "<sub>Sin matches de sensor por nombre — revisa "
                              "los sensores en Tab Setup → Geometría 3D</sub>",
                       font=dict(size=14, color="#dc2626")),
            scene=dict(aspectmode="data",
                         camera=dict(eye=dict(x=1.4, y=1.2, z=0.9)),
                         bgcolor="#f8fafc"),
            margin=dict(l=0, r=0, t=70, b=0), height=560,
            paper_bgcolor="white",
        )
        return fig

    # Normalizacion para tamano de flecha proporcional a la amplitud del modo
    span = max(geom.shaft_end - geom.shaft_start, 100.0)
    max_abs = max(abs(m) for m in matched_signed_mag) or 1.0
    base_arrow = span * 0.08

    # Para cada sensor i, el mode shape oscila como:
    #   d_i(t) = |phi_i| * cos(phase_i + omega·t) * dof_dir
    # En t=0 esto se reduce a la proyeccion sobre el eje real del componente
    # complejo (que ya estaba siendo calculada como amp*sign). Generalizamos
    # para soportar tanto el render estatico (t=0) como la animacion.
    matched_amps = [float(abs(m)) for m in matched_signed_mag]
    # phase_rad[i] es el angulo de la componente compleja
    matched_phases_rad = [math.radians(p) for p in matched_phases_deg]

    def _cones_at_phase(theta_rad: float):
        """Devuelve dos traces (cofase verde + anti-fase rojo) en el frame θ."""
        cx_pos, cy_pos, cz_pos = [], [], []
        cu_pos, cv_pos, cw_pos = [], [], []
        tx_pos: List[str] = []
        cx_neg, cy_neg, cz_neg = [], [], []
        cu_neg, cv_neg, cw_neg = [], [], []
        tx_neg: List[str] = []
        for (px, py, pz), (ux, uy, uz), amp, ph_rad, name in zip(
            matched_positions, matched_directions, matched_amps,
            matched_phases_rad, matched_names
        ):
            # ux,uy,uz ya tienen el signo de la proyeccion en t=0.
            # Eliminarlo: divide por el signo aplicado previamente.
            # Pero ux,uy,uz fue construido como _dof_to_vector(s.dof) * sign(t=0).
            # Para animacion, recuperamos el dof base y aplicamos cos(phase+theta).
            # Reconstruimos: dir_dof = (ux,uy,uz) si abs(ph_rad)<=pi/2 (signo +)
            #               (-ux,-uy,-uz) si abs(ph_rad)>pi/2 (signo -)
            # i.e. dir_dof = (ux,uy,uz) / sign_t0  con sign_t0 = cos(ph_rad)>=0?+1:-1
            sign_t0 = 1.0 if abs(math.cos(ph_rad)) >= 0 and math.cos(ph_rad) >= 0 else -1.0
            # Reconstruccion exacta:
            sign_t0 = 1.0 if math.cos(ph_rad) >= 0 else -1.0
            dux = ux / sign_t0 if sign_t0 != 0 else ux
            duy = uy / sign_t0 if sign_t0 != 0 else uy
            duz = uz / sign_t0 if sign_t0 != 0 else uz
            # Componente instantanea del mode shape en este frame:
            disp = amp * math.cos(ph_rad + theta_rad)
            sc = base_arrow * abs(disp) / max_abs
            cu = dux * (1.0 if disp >= 0 else -1.0) * sc
            cv = duy * (1.0 if disp >= 0 else -1.0) * sc
            cw = duz * (1.0 if disp >= 0 else -1.0) * sc
            label = (f"{name} · |phi|={amp:.3f} · phi={math.degrees(ph_rad):.0f}deg "
                     f"· disp={disp:+.3f}")
            if disp >= 0:
                cx_pos.append(px); cy_pos.append(py); cz_pos.append(pz)
                cu_pos.append(cu); cv_pos.append(cv); cw_pos.append(cw)
                tx_pos.append(label)
            else:
                cx_neg.append(px); cy_neg.append(py); cz_neg.append(pz)
                cu_neg.append(cu); cv_neg.append(cv); cw_neg.append(cw)
                tx_neg.append(label)
        # Crear traces — incluso si vacias, se agregan placeholders para que
        # los indices en frames sean estables.
        trace_pos = go.Cone(
            x=cx_pos or [0], y=cy_pos or [0], z=cz_pos or [0],
            u=cu_pos or [0], v=cv_pos or [0], w=cw_pos or [0],
            sizemode="absolute", sizeref=base_arrow,
            colorscale=[[0, "#16a34a"], [1, "#16a34a"]],
            showscale=False, name="Cofase (+)", showlegend=True,
            hoverinfo="text", text=tx_pos or [""],
            visible=bool(cx_pos),
        )
        trace_neg = go.Cone(
            x=cx_neg or [0], y=cy_neg or [0], z=cz_neg or [0],
            u=cu_neg or [0], v=cv_neg or [0], w=cw_neg or [0],
            sizemode="absolute", sizeref=base_arrow,
            colorscale=[[0, "#dc2626"], [1, "#dc2626"]],
            showscale=False, name="Anti-fase (−)", showlegend=True,
            hoverinfo="text", text=tx_neg or [""],
            visible=bool(cx_neg),
        )
        return trace_pos, trace_neg

    # Trace inicial (theta=0)
    cone_pos_0, cone_neg_0 = _cones_at_phase(0.0)
    cone_pos_idx = len(fig.data)        # indice del trace cone positivo
    cone_neg_idx = cone_pos_idx + 1
    fig.add_trace(cone_pos_0)
    fig.add_trace(cone_neg_0)

    # 5) Labels de los sensores
    fig.add_trace(go.Scatter3d(
        x=[p[0] for p in matched_positions],
        y=[p[1] for p in matched_positions],
        z=[p[2] for p in matched_positions],
        mode="text",
        text=matched_names,
        textposition="top center",
        textfont=dict(size=11, color="#0F1E3D"),
        showlegend=False, hoverinfo="skip",
    ))

    # Layout
    fig.update_layout(
        scene=dict(
            xaxis=dict(title=f"X ({geom.units})", showgrid=True,
                        gridcolor="#e5e7eb", zerolinecolor="#cbd5e1"),
            yaxis=dict(title=f"Y ({geom.units})", showgrid=True,
                        gridcolor="#e5e7eb", zerolinecolor="#cbd5e1"),
            zaxis=dict(title=f"Z ({geom.units})", showgrid=True,
                        gridcolor="#e5e7eb", zerolinecolor="#cbd5e1"),
            aspectmode="data",
            camera=dict(eye=dict(x=1.4, y=1.2, z=0.9)),
            bgcolor="#f8fafc",
        ),
        title=dict(text=f"{mode_label}<br>"
                          f"<sub>{len(matched_positions)}/{len(channel_names)} "
                          "sensores con match · verde: cofase · rojo: anti-fase</sub>",
                   font=dict(size=14, color="#0F1E3D")),
        margin=dict(l=0, r=0, t=70, b=0),
        height=580 if animate else 560,
        paper_bgcolor="white",
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.05),
    )

    # -------------------------------------------------------------------
    # Animacion Artemis-style: las flechas pulsan oscilando con la fase
    # del modo. Los bloques/eje/labels quedan estaticos; solo los 2 cones
    # se reemplazan por frame (indices cone_pos_idx, cone_neg_idx).
    # -------------------------------------------------------------------
    if animate:
        frames = []
        for k in range(n_frames):
            theta = 2.0 * math.pi * k / n_frames
            cone_pos_k, cone_neg_k = _cones_at_phase(theta)
            frames.append(go.Frame(
                data=[cone_pos_k, cone_neg_k],
                traces=[cone_pos_idx, cone_neg_idx],
                name=f"{int(math.degrees(theta)):03d}",
            ))
        fig.frames = frames

        # Play / Pause buttons + slider de fase manual
        fig.update_layout(
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                x=0.02, y=1.06, xanchor="left", yanchor="top",
                bgcolor="#0F1E3D", bordercolor="#0F1E3D",
                font=dict(color="white", size=12),
                buttons=[
                    dict(label="▶ Play",
                          method="animate",
                          args=[None, dict(
                              frame=dict(duration=frame_duration_ms, redraw=True),
                              fromcurrent=True, mode="immediate",
                              transition=dict(duration=0),
                          )]),
                    dict(label="⏸ Pause",
                          method="animate",
                          args=[[None], dict(
                              frame=dict(duration=0, redraw=False),
                              mode="immediate",
                              transition=dict(duration=0),
                          )]),
                ],
            )],
            sliders=[dict(
                active=0, currentvalue=dict(prefix="Fase del modo: ", suffix="°"),
                pad=dict(t=40, b=10),
                len=0.78, x=0.18, xanchor="left",
                steps=[
                    dict(method="animate", label=f.name,
                          args=[[f.name], dict(
                              frame=dict(duration=0, redraw=True),
                              mode="immediate",
                              transition=dict(duration=0),
                          )])
                    for f in frames
                ],
            )],
        )

    return fig
