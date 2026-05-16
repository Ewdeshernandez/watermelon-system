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
    """Seccion mecanica del tren — cilindro o caja.

    kind define que tipo de sensores deforman este bloque en la animacion:
      - "casing": deformado por sensores de carcasa (accels, velocidad)
      - "shaft":  deformado por sensores de eje (proximidades eddy)
      - "coupling": estatico o interpolado entre vecinos del mismo kind
    """
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
    kind: Literal["casing", "shaft", "coupling"] = "casing"


@dataclass
class GeometrySensor:
    """Punto de medicion con direccion de DOF.

    mounting define que mide el sensor y por lo tanto que capa anima:
      - "casing":         accelerometro o velocimetro montado en la carcasa
      - "shaft_proximity": eddy current probe montado en el bearing
                           viendo directamente el eje
    Si se deja en blanco, se infiere de sensor_type automaticamente.
    """
    id: str
    name: str
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    dof: str = "+Y"   # "+X" "-X" "+Y" "-Y" "+Z" "-Z"
    sensor_type: str = "accelerometer"  # accelerometer | proximity | velocity
    mounting: Literal["", "casing", "shaft_proximity"] = ""

    def effective_mounting(self) -> str:
        """Devuelve el mounting explicito si esta seteado, sino lo infiere."""
        if self.mounting:
            return self.mounting
        if self.sensor_type == "proximity":
            return "shaft_proximity"
        return "casing"  # accelerometer y velocity por default


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
                          color="#0F1E3D", opacity=0.35, kind="casing"),
            GeometryBlock(id="coupling", name="Coupling",
                          shape="cylinder", x_start=500.0, x_end=650.0,
                          radius=90.0, color="#D89B22", opacity=0.55,
                          kind="coupling"),
            GeometryBlock(id="compressor", name="Compresor",
                          shape="cylinder", x_start=650.0, x_end=1600.0,
                          radius=300.0, color="#1AAEE5", opacity=0.30,
                          kind="shaft"),
        ],
        sensors=[
            # Motor — 2 acelerometros sobre la carcasa
            GeometrySensor(id="s1", name="MOT-NDE", x=30.0, y=250.0, z=0.0,
                            dof="+Y", sensor_type="accelerometer",
                            mounting="casing"),
            GeometrySensor(id="s2", name="MOT-DE",  x=470.0, y=250.0, z=0.0,
                            dof="+Y", sensor_type="accelerometer",
                            mounting="casing"),
            # Compresor — 4 proximidades ortogonales mirando al eje
            GeometrySensor(id="s3", name="COMP-CE-Y",  x=700.0, y=300.0, z=0.0,
                            dof="+Y", sensor_type="proximity",
                            mounting="shaft_proximity"),
            GeometrySensor(id="s4", name="COMP-CE-X",  x=700.0, y=0.0,   z=300.0,
                            dof="+Z", sensor_type="proximity",
                            mounting="shaft_proximity"),
            GeometrySensor(id="s5", name="COMP-NCE-Y", x=1570.0, y=300.0, z=0.0,
                            dof="+Y", sensor_type="proximity",
                            mounting="shaft_proximity"),
            GeometrySensor(id="s6", name="COMP-NCE-X", x=1570.0, y=0.0,   z=300.0,
                            dof="+Z", sensor_type="proximity",
                            mounting="shaft_proximity"),
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
                          radius=380.0, color="#0F1E3D", opacity=0.30,
                          kind="casing"),
            GeometryBlock(id="coupling", name="Coupling",
                          shape="cylinder", x_start=1000.0, x_end=1150.0,
                          radius=100.0, color="#D89B22", opacity=0.55,
                          kind="coupling"),
            GeometryBlock(id="generator", name="Generador Brush",
                          shape="box", x_start=1150.0, x_end=2200.0,
                          half_width=320.0, half_height=350.0,
                          color="#1AAEE5", opacity=0.30,
                          kind="shaft"),
        ],
        sensors=[
            # Turbina — 2 acelerometros sobre la carcasa
            GeometrySensor(id="s1", name="TRB-CRF", x=80.0,  y=380.0, z=0.0,
                            dof="+Y", sensor_type="accelerometer",
                            mounting="casing"),
            GeometrySensor(id="s2", name="TRB-TRF", x=950.0, y=380.0, z=0.0,
                            dof="+Y", sensor_type="accelerometer",
                            mounting="casing"),
            # Generador — 4 proxies mirando al eje en los 2 bearings
            GeometrySensor(id="s3", name="GEN-CE-Y",  x=1200.0, y=320.0, z=0.0,
                            dof="+Y", sensor_type="proximity",
                            mounting="shaft_proximity"),
            GeometrySensor(id="s4", name="GEN-CE-X",  x=1200.0, y=0.0,   z=350.0,
                            dof="+Z", sensor_type="proximity",
                            mounting="shaft_proximity"),
            GeometrySensor(id="s5", name="GEN-NCE-Y", x=2150.0, y=320.0, z=0.0,
                            dof="+Y", sensor_type="proximity",
                            mounting="shaft_proximity"),
            GeometrySensor(id="s6", name="GEN-NCE-X", x=2150.0, y=0.0,   z=350.0,
                            dof="+Z", sensor_type="proximity",
                            mounting="shaft_proximity"),
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
                          color="#0F1E3D", opacity=0.35, kind="casing"),
            GeometryBlock(id="coupling", name="Coupling",
                          shape="cylinder", x_start=500.0, x_end=600.0,
                          radius=70.0, color="#D89B22", opacity=0.55,
                          kind="coupling"),
            GeometryBlock(id="pump", name="Bomba",
                          shape="cylinder", x_start=600.0, x_end=1300.0,
                          radius=230.0, color="#1AAEE5", opacity=0.30,
                          kind="shaft"),
        ],
        sensors=[
            GeometrySensor(id="s1", name="MOT-NDE",  x=30.0,  y=200.0, z=0.0,
                            dof="+Y", sensor_type="accelerometer",
                            mounting="casing"),
            GeometrySensor(id="s2", name="MOT-DE",   x=470.0, y=200.0, z=0.0,
                            dof="+Y", sensor_type="accelerometer",
                            mounting="casing"),
            GeometrySensor(id="s3", name="PMP-DE-Y", x=650.0, y=230.0, z=0.0,
                            dof="+Y", sensor_type="proximity",
                            mounting="shaft_proximity"),
            GeometrySensor(id="s4", name="PMP-DE-X", x=650.0, y=0.0,   z=230.0,
                            dof="+Z", sensor_type="proximity",
                            mounting="shaft_proximity"),
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
                    n_theta: int = 28, n_x: int = 16
                    ) -> Tuple[list, list, list, list, list, list]:
    """Cilindro alineado a X subdividido en n_x anillos longitudinales.

    Subdivision = mas vertices = la deformacion por spline luce como una
    viga que flexiona suavemente, no como bloque que se inclina.
    """
    import numpy as np
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    x_levels = np.linspace(x0, x1, n_x + 1)
    xs, ys, zs = [], [], []
    # Generar n_x+1 anillos de n_theta vertices cada uno
    for x_val in x_levels:
        for t in theta:
            xs.append(float(x_val))
            ys.append(radius * math.cos(t))
            zs.append(radius * math.sin(t))
    n = n_theta
    # Centros para los caps (solo extremos)
    xs += [float(x_levels[0]), float(x_levels[-1])]
    ys += [0.0, 0.0]
    zs += [0.0, 0.0]
    c0 = (n_x + 1) * n
    c1 = c0 + 1
    i, j, k = [], [], []
    # Side quads entre anillos consecutivos (2 triangles por quad)
    for ring in range(n_x):
        base0 = ring * n
        base1 = (ring + 1) * n
        for a in range(n):
            b = (a + 1) % n
            i += [base0 + a, base0 + a]
            j += [base0 + b, base1 + b]
            k += [base1 + b, base1 + a]
    # Cap bottom (anillo 0)
    for a in range(n):
        b = (a + 1) % n
        i.append(a); j.append(b); k.append(c0)
    # Cap top (anillo n_x)
    top_base = n_x * n
    for a in range(n):
        b = (a + 1) % n
        i.append(top_base + a); j.append(c1); k.append(top_base + b)
    return xs, ys, zs, i, j, k


def _box_mesh(x0: float, x1: float, hw: float, hh: float,
               n_x: int = 16) -> Tuple[list, list, list, list, list, list]:
    """Caja rectangular alineada a X subdividida en n_x cross-sections.

    Cada cross-section tiene 4 corner vertices (bottom-left, bottom-right,
    top-right, top-left en orden 0,1,2,3). Las caras laterales se construyen
    con quads entre cross-sections consecutivas. Caps solo en los extremos.

    Subdivision = bending suave de la carcasa cuando se anima la deformacion.
    """
    import numpy as np
    x_levels = np.linspace(x0, x1, n_x + 1)
    xs, ys, zs = [], [], []
    # 4 corner vertices por cross-section
    # Orden: 0=(-hw,-hh) 1=(+hw,-hh) 2=(+hw,+hh) 3=(-hw,+hh)
    corners = [(-hw, -hh), (+hw, -hh), (+hw, +hh), (-hw, +hh)]
    for x_val in x_levels:
        for (y, z) in corners:
            xs.append(float(x_val)); ys.append(y); zs.append(z)
    i, j, k = [], [], []
    # Side faces: 4 caras laterales (bottom, right, top, left), cada una
    # subdividida en n_x quads (2 triangles cada uno)
    for sec in range(n_x):
        b0 = sec * 4         # cross-section sec
        b1 = (sec + 1) * 4   # siguiente cross-section
        # 4 quads laterales (entre vertices a y a+1)
        for a in range(4):
            an = (a + 1) % 4
            # Quad (b0+a, b0+an, b1+an, b1+a) → 2 triangles
            i += [b0 + a, b0 + a]
            j += [b0 + an, b1 + an]
            k += [b1 + an, b1 + a]
    # Caps en los extremos (cross-section 0 y n_x)
    # Cap inicial: 2 triangles (0,1,2), (0,2,3)
    i += [0, 0]; j += [1, 2]; k += [2, 3]
    # Cap final: 2 triangles
    last = n_x * 4
    i += [last, last]; j += [last + 2, last + 3]; k += [last + 1, last + 2]
    return xs, ys, zs, i, j, k


# ---------------------------------------------------------------------
# Interpolacion PCHIP (Piecewise Cubic Hermite, Fritsch-Carlson)
# para construir splines complejos u(x) a partir de los sensores
# ---------------------------------------------------------------------

def _pchip_tangents(xs: List[float], ys: List[complex]) -> List[complex]:
    n = len(xs)
    if n < 2:
        return [complex(0.0)] * n
    h = [xs[i + 1] - xs[i] for i in range(n - 1)]
    d = [(ys[i + 1] - ys[i]) / h[i] for i in range(n - 1)]
    m: List[complex] = [complex(0.0)] * n
    m[0] = d[0]
    m[-1] = d[-1]
    for i in range(1, n - 1):
        # Monotonicidad de Fritsch-Carlson generalizada a complejos:
        # aplicamos por componentes real e imaginaria
        for component in ("real", "imag"):
            d_im1 = getattr(d[i - 1], component)
            d_i = getattr(d[i], component)
            if d_im1 * d_i <= 0:
                val = 0.0
            else:
                w1 = 2 * h[i] + h[i - 1]
                w2 = h[i] + 2 * h[i - 1]
                val = (w1 + w2) / (w1 / d_im1 + w2 / d_i)
            if component == "real":
                m[i] = complex(val, m[i].imag)
            else:
                m[i] = complex(m[i].real, val)
    return m


def _pchip_eval(xs: List[float], ys: List[complex],
                 tans: List[complex], x: float) -> complex:
    if not xs:
        return complex(0.0)
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    i = 0
    for k in range(len(xs) - 1):
        if x <= xs[k + 1]:
            i = k
            break
    h = xs[i + 1] - xs[i]
    t = (x - xs[i]) / h
    h00 = 2 * t**3 - 3 * t**2 + 1
    h10 = t**3 - 2 * t**2 + t
    h01 = -2 * t**3 + 3 * t**2
    h11 = t**3 - t**2
    return h00 * ys[i] + h10 * h * tans[i] + h01 * ys[i + 1] + h11 * h * tans[i + 1]


def _build_mode_spline_by_mounting(
    sensors_data: List[Dict[str, Any]],
    mounting_filter: str,
    axis: str,    # "y" o "z"
) -> Optional[Tuple[List[float], List[complex], List[complex]]]:
    """Construye (xs, ys_complex, tangents) para una capa mounting+axis.

    sensors_data: lista de dicts con keys: x, mounting, dof_unit_vec (3-tuple),
                  modal_complex (complex amp)
    Devuelve None si hay < 2 sensores compatibles (no se puede spline).
    """
    axis_idx = {"x": 0, "y": 1, "z": 2}[axis]
    sample_points: List[Tuple[float, complex]] = []
    for s in sensors_data:
        if s["mounting"] != mounting_filter:
            continue
        # Solo contribuye si el DOF unit vector tiene componente significativa
        # en el eje de interes
        comp = s["dof_unit_vec"][axis_idx]
        if abs(comp) < 0.1:
            continue
        # Proyeccion del modal complex sobre el eje
        contribution = complex(s["modal_complex"]) * comp
        sample_points.append((float(s["x"]), contribution))
    if len(sample_points) < 2:
        # Si hay 1 sensor, usar constante; si hay 0, no hay capa
        if len(sample_points) == 1:
            xs = [sample_points[0][0] - 1.0, sample_points[0][0] + 1.0]
            ys = [sample_points[0][1], sample_points[0][1]]
            tans = [complex(0.0), complex(0.0)]
            return xs, ys, tans
        return None
    # Ordenar por x
    sample_points.sort(key=lambda p: p[0])
    # Promediar duplicados en mismo X (caso accel+accel coincidentes)
    xs_uniq: List[float] = []
    ys_uniq: List[complex] = []
    for x, c in sample_points:
        if xs_uniq and abs(x - xs_uniq[-1]) < 1e-6:
            ys_uniq[-1] = (ys_uniq[-1] + c) / 2.0
        else:
            xs_uniq.append(x)
            ys_uniq.append(c)
    if len(xs_uniq) < 2:
        xs_uniq = [xs_uniq[0] - 1.0, xs_uniq[0] + 1.0]
        ys_uniq = [ys_uniq[0], ys_uniq[0]]
    tans = _pchip_tangents(xs_uniq, ys_uniq)
    return xs_uniq, ys_uniq, tans


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
                                           geom.shaft_radius, n_theta=20, n_x=24)
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
            dragmode="turntable",
            camera=dict(eye=dict(x=0.0, y=2.2, z=0.6),
                          up=dict(x=0, y=0, z=1)),
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
    n_frames: int = 36,
    frame_duration_ms: int = 180,
    show_arrows: bool = False,
    show_ghost: bool = True,
    colormap: str = "RdBu_r",
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

    # -------------------------------------------------------------------
    # Paso 1: Matching sensores <-> channel_names + datos por mounting
    # -------------------------------------------------------------------
    by_name: Dict[str, GeometrySensor] = {
        s.name.strip().upper(): s for s in geom.sensors
    }
    arr = np.asarray(mode_shape, dtype=complex).flatten()

    matched_positions: List[Tuple[float, float, float]] = []
    matched_directions: List[Tuple[float, float, float]] = []
    matched_amps: List[float] = []
    matched_phases_rad: List[float] = []
    matched_names: List[str] = []
    sensors_for_spline: List[Dict[str, Any]] = []

    for idx, ch in enumerate(channel_names):
        key = ch.strip().upper()
        if key not in by_name or idx >= len(arr):
            continue
        s = by_name[key]
        complex_val = complex(arr[idx])
        amp = float(abs(complex_val))
        phase_rad = math.atan2(complex_val.imag, complex_val.real)
        ux, uy, uz = _dof_to_vector(s.dof)
        matched_positions.append((s.x, s.y, s.z))
        matched_directions.append((ux, uy, uz))
        matched_amps.append(amp)
        matched_phases_rad.append(phase_rad)
        matched_names.append(s.name)
        sensors_for_spline.append({
            "x": s.x,
            "mounting": s.effective_mounting(),
            "dof_unit_vec": (ux, uy, uz),
            "modal_complex": complex_val,
        })

    # -------------------------------------------------------------------
    # Paso 2: Splines doble capa
    # Construye u_y(x), u_z(x) por separado para casing y shaft_proximity
    # -------------------------------------------------------------------
    splines: Dict[Tuple[str, str], Optional[Tuple[List[float], List[complex], List[complex]]]] = {}
    for mnt in ("casing", "shaft_proximity"):
        for axis in ("y", "z"):
            splines[(mnt, axis)] = _build_mode_spline_by_mounting(
                sensors_for_spline, mnt, axis
            )

    def _eval_spline(mnt: str, axis: str, x: float) -> complex:
        """Devuelve el valor complejo de la spline (mnt, axis) en x, o 0 si no hay."""
        sp = splines.get((mnt, axis))
        if sp is None:
            return complex(0.0)
        xs, ys, tans = sp
        return _pchip_eval(xs, ys, tans, x)

    # Compute amplitude max global para escalar deformacion visual
    sample_xs = np.linspace(geom.shaft_start, geom.shaft_end, 32)
    max_disp = 1e-12
    for mnt in ("casing", "shaft_proximity"):
        for axis in ("y", "z"):
            for x in sample_xs:
                max_disp = max(max_disp, abs(_eval_spline(mnt, axis, float(x))))

    span = max(geom.shaft_end - geom.shaft_start, 100.0)
    # Deformacion visual: hasta 18% del span — exageracion estetica para que
    # la flexion sea claramente visible (Artemis usa ~20% por default)
    deform_scale = (span * 0.18) / max_disp if max_disp > 1e-9 else 0.0

    def _kind_to_mounting(kind: str) -> str:
        """Mapea kind del bloque a mounting de sensores que lo deforman."""
        if kind == "shaft":
            return "shaft_proximity"
        if kind == "coupling":
            # El coupling esta en el shaft: usa shaft_proximity si hay,
            # sino fallback a casing
            return ("shaft_proximity"
                     if splines.get(("shaft_proximity", "y")) is not None
                     else "casing")
        return "casing"  # default

    def _displace_mesh(xs: List[float], ys: List[float], zs: List[float],
                        kind: str, theta_rad: float
                        ) -> Tuple[List[float], List[float], List[float], List[float]]:
        """Deforma los mesh nodes y devuelve tambien |displacement| por vertex.

        Returns:
            (new_xs, new_ys, new_zs, intensity_per_vertex)
        """
        mnt = _kind_to_mounting(kind)
        sp_y = splines.get((mnt, "y"))
        sp_z = splines.get((mnt, "z"))
        intensity = [0.0] * len(xs)
        if sp_y is None and sp_z is None:
            return xs, ys, zs, intensity
        new_ys = list(ys); new_zs = list(zs)
        cos_t = math.cos(theta_rad); sin_t = math.sin(theta_rad)
        for i, x in enumerate(xs):
            disp_y = 0.0; disp_z = 0.0
            if sp_y is not None:
                cy = _eval_spline(mnt, "y", float(x))
                disp_y = (cy * (cos_t + 1j * sin_t)).real
                new_ys[i] = ys[i] + disp_y * deform_scale
            if sp_z is not None:
                cz = _eval_spline(mnt, "z", float(x))
                disp_z = (cz * (cos_t + 1j * sin_t)).real
                new_zs[i] = zs[i] + disp_z * deform_scale
            # Magnitud SIGNED de la proyeccion total — usamos como intensidad
            # para que el colormap RdBu_r muestre +/-: rojo cofase, azul anti-fase
            intensity[i] = disp_y + disp_z  # suma signed
        return xs, new_ys, new_zs, intensity

    # -------------------------------------------------------------------
    # Paso 3: Construir mesh base de cada bloque + shaft (a theta=0)
    # Guardamos la connectividad i,j,k y los vertices base para deformar
    # despues en cada frame.
    # -------------------------------------------------------------------
    block_meshes_base = []  # cada item: (xs0, ys0, zs0, i, j, k, kind, color, opacity, name)
    for b in geom.blocks:
        if b.shape == "cylinder":
            xs, ys, zs, ii, jj, kk = _cylinder_mesh(b.x_start, b.x_end, b.radius)
        else:
            xs, ys, zs, ii, jj, kk = _box_mesh(b.x_start, b.x_end,
                                                 b.half_width, b.half_height)
        block_meshes_base.append((xs, ys, zs, ii, jj, kk,
                                    b.kind, b.color, b.opacity, b.name))

    # Shaft (cilindro central) — siempre kind="shaft"
    xs, ys, zs, ii, jj, kk = _cylinder_mesh(geom.shaft_start, geom.shaft_end,
                                              geom.shaft_radius, n_theta=20, n_x=24)
    shaft_base = (xs, ys, zs, ii, jj, kk, "shaft",
                   geom.shaft_color, 0.85, "Eje")

    # Intensity range global (para que el colormap sea estable a traves de frames)
    int_max = 0.0
    for (xs0, ys0, zs0, ii, jj, kk, kind, _c, _o, _n) in block_meshes_base + [
            (shaft_base[0], shaft_base[1], shaft_base[2], shaft_base[3],
             shaft_base[4], shaft_base[5], shaft_base[6], shaft_base[7],
             shaft_base[8], shaft_base[9])]:
        for k_test in (0, n_frames // 4):
            th = 2.0 * math.pi * k_test / max(n_frames, 1)
            _, _, _, intens = _displace_mesh(xs0, ys0, zs0, kind, th)
            int_max = max(int_max, max((abs(v) for v in intens), default=0.0))
    if int_max < 1e-9:
        int_max = 1.0

    # GHOST: contornos de la geometria sin deformar (semi-transparentes)
    if show_ghost:
        for (xs0, ys0, zs0, ii, jj, kk, kind, color, opacity, name) in block_meshes_base:
            fig.add_trace(go.Mesh3d(
                x=xs0, y=ys0, z=zs0, i=ii, j=jj, k=kk,
                color="#94a3b8", opacity=0.10,
                name=f"{name} (ghost)", showlegend=False, hoverinfo="skip",
                flatshading=True,
            ))
        # Shaft ghost
        s_xs, s_ys, s_zs, s_i, s_j, s_k = shaft_base[0:6]
        fig.add_trace(go.Mesh3d(
            x=s_xs, y=s_ys, z=s_zs, i=s_i, j=s_j, k=s_k,
            color="#64748b", opacity=0.20,
            name="Eje (ghost)", showlegend=False, hoverinfo="skip",
            flatshading=True,
        ))

    # Trace indices para frames — los bloques van primero, luego shaft, luego labels, luego cones
    block_trace_indices = []
    for idx, (xs0, ys0, zs0, ii, jj, kk, kind, color, opacity, name) in enumerate(block_meshes_base):
        new_x, new_y, new_z, intensity = _displace_mesh(xs0, ys0, zs0, kind, 0.0)
        # Mostrar colorbar SOLO en el primer trace (escala global compartida)
        _show_cb = (idx == 0)
        fig.add_trace(go.Mesh3d(
            x=new_x, y=new_y, z=new_z, i=ii, j=jj, k=kk,
            intensity=intensity, intensitymode="vertex",
            colorscale=colormap, cmin=-int_max, cmax=int_max,
            showscale=_show_cb,
            colorbar=dict(
                title=dict(text="Δ desplaz.<br>(visual)",
                            font=dict(size=11, color="#0F1E3D")),
                thickness=14, len=0.55, x=1.02, xanchor="left",
                tickfont=dict(size=10, color="#0F1E3D"),
                outlinewidth=0,
            ) if _show_cb else None,
            opacity=0.94,
            name=name, showlegend=False, hoverinfo="skip",
            flatshading=False,
            lighting=dict(ambient=0.45, diffuse=0.85, specular=0.35,
                            roughness=0.55, fresnel=0.15),
            lightposition=dict(x=2000, y=2500, z=2500),
        ))
        block_trace_indices.append(len(fig.data) - 1)

    # Shaft (deformado por shaft spline en t=0) — color uniforme oscuro para que destaque
    xs0, ys0, zs0, ii, jj, kk, kind, color, opacity, name = shaft_base
    new_x, new_y, new_z, intensity = _displace_mesh(xs0, ys0, zs0, kind, 0.0)
    fig.add_trace(go.Mesh3d(
        x=new_x, y=new_y, z=new_z, i=ii, j=jj, k=kk,
        intensity=intensity, intensitymode="vertex",
        colorscale=colormap, cmin=-int_max, cmax=int_max,
        showscale=False, opacity=1.0,
        name=name, showlegend=False, hoverinfo="skip",
        flatshading=False,
        lighting=dict(ambient=0.5, diffuse=0.9, specular=0.5,
                        roughness=0.4, fresnel=0.2),
        lightposition=dict(x=2000, y=2500, z=2500),
    ))
    shaft_trace_idx = len(fig.data) - 1

    # -------------------------------------------------------------------
    # Paso 4: Manejo de "sin matches"
    # -------------------------------------------------------------------
    if not matched_positions:
        fig.update_layout(
            title=dict(text=f"{mode_label}<br>"
                              "<sub>Sin matches de sensor por nombre — revisa "
                              "los sensores en Tab Setup → Geometría 3D</sub>",
                       font=dict(size=14, color="#dc2626")),
            scene=dict(aspectmode="data",
                         camera=dict(eye=dict(x=0.0, y=2.2, z=0.6),
                          up=dict(x=0, y=0, z=1)),
                         bgcolor="#f8fafc"),
            margin=dict(l=0, r=0, t=70, b=0), height=560,
            paper_bgcolor="white",
        )
        return fig

    max_amp = max(matched_amps) or 1.0
    # Flechas mas chicas (3%) para no opacar los bloques deformados
    base_arrow = span * 0.03

    # -------------------------------------------------------------------
    # Paso 5: Flechas (cones) — coloreadas verde/rojo segun fase instantanea
    # -------------------------------------------------------------------
    def _cones_at_phase(theta_rad: float):
        """Dos traces (cofase + anti-fase) en el frame θ."""
        cx_pos, cy_pos, cz_pos = [], [], []
        cu_pos, cv_pos, cw_pos = [], [], []
        tx_pos: List[str] = []
        cx_neg, cy_neg, cz_neg = [], [], []
        cu_neg, cv_neg, cw_neg = [], [], []
        tx_neg: List[str] = []
        for (px, py, pz), (dux, duy, duz), amp, ph_rad, name in zip(
            matched_positions, matched_directions, matched_amps,
            matched_phases_rad, matched_names
        ):
            disp = amp * math.cos(ph_rad + theta_rad)
            sc = base_arrow * abs(disp) / max_amp
            sign = 1.0 if disp >= 0 else -1.0
            cu = dux * sign * sc
            cv = duy * sign * sc
            cw = duz * sign * sc
            label = (f"{name} · |φ|={amp:.3f} · "
                     f"φ={math.degrees(ph_rad):.0f}° · disp={disp:+.3f}")
            if disp >= 0:
                cx_pos.append(px); cy_pos.append(py); cz_pos.append(pz)
                cu_pos.append(cu); cv_pos.append(cv); cw_pos.append(cw)
                tx_pos.append(label)
            else:
                cx_neg.append(px); cy_neg.append(py); cz_neg.append(pz)
                cu_neg.append(cu); cv_neg.append(cv); cw_neg.append(cw)
                tx_neg.append(label)
        trace_pos = go.Cone(
            x=cx_pos or [0], y=cy_pos or [0], z=cz_pos or [0],
            u=cu_pos or [0], v=cv_pos or [0], w=cw_pos or [0],
            sizemode="absolute", sizeref=base_arrow,
            colorscale=[[0, "#16a34a"], [1, "#16a34a"]],
            showscale=False, name="Cofase (+)", showlegend=True,
            hoverinfo="text", text=tx_pos or [""], visible=bool(cx_pos),
        )
        trace_neg = go.Cone(
            x=cx_neg or [0], y=cy_neg or [0], z=cz_neg or [0],
            u=cu_neg or [0], v=cv_neg or [0], w=cw_neg or [0],
            sizemode="absolute", sizeref=base_arrow,
            colorscale=[[0, "#dc2626"], [1, "#dc2626"]],
            showscale=False, name="Anti-fase (−)", showlegend=True,
            hoverinfo="text", text=tx_neg or [""], visible=bool(cx_neg),
        )
        return trace_pos, trace_neg

    cone_pos_idx = -1
    cone_neg_idx = -1
    if show_arrows:
        cone_pos_0, cone_neg_0 = _cones_at_phase(0.0)
        cone_pos_idx = len(fig.data)
        cone_neg_idx = cone_pos_idx + 1
        fig.add_trace(cone_pos_0)
        fig.add_trace(cone_neg_0)

    # Markers + labels de los sensores (estaticos, siempre visibles)
    fig.add_trace(go.Scatter3d(
        x=[p[0] for p in matched_positions],
        y=[p[1] for p in matched_positions],
        z=[p[2] for p in matched_positions],
        mode="markers+text",
        marker=dict(size=5, color="#0F1E3D",
                      line=dict(width=1, color="#ffffff")),
        text=matched_names,
        textposition="top center",
        textfont=dict(size=10, color="#0F1E3D"),
        showlegend=False, hoverinfo="text",
        hovertext=[f"{n} · |φ|={a:.3f} · φ={math.degrees(p):.0f}°"
                    for n, a, p in zip(matched_names, matched_amps,
                                          matched_phases_rad)],
    ))

    # -------------------------------------------------------------------
    # Layout
    # -------------------------------------------------------------------
    # Status summary para el subtitle
    n_casing = sum(1 for s in sensors_for_spline if s["mounting"] == "casing")
    n_shaft = sum(1 for s in sensors_for_spline if s["mounting"] == "shaft_proximity")
    subtitle = (f"{len(matched_positions)}/{len(channel_names)} sensores · "
                f"capa carcasa: {n_casing} accel · capa eje: {n_shaft} prox · "
                f"colormap: rojo = +amp, azul = −amp")

    fig.update_layout(
        scene=dict(
            xaxis=dict(title=f"X ({geom.units})", showgrid=True,
                        gridcolor="#e5e7eb", zerolinecolor="#cbd5e1"),
            yaxis=dict(title=f"Y ({geom.units})", showgrid=True,
                        gridcolor="#e5e7eb", zerolinecolor="#cbd5e1"),
            zaxis=dict(title=f"Z ({geom.units})", showgrid=True,
                        gridcolor="#e5e7eb", zerolinecolor="#cbd5e1"),
            aspectmode="data",
            dragmode="turntable",
            camera=dict(eye=dict(x=0.0, y=2.2, z=0.6),
                          up=dict(x=0, y=0, z=1)),
            bgcolor="#f8fafc",
        ),
        title=dict(text=f"{mode_label}<br><sub>{subtitle}</sub>",
                   font=dict(size=14, color="#0F1E3D")),
        margin=dict(l=0, r=0, t=70, b=0),
        height=600 if animate else 560,
        paper_bgcolor="white",
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.05),
    )

    # -------------------------------------------------------------------
    # Animacion: replace blocks + shaft + cones per frame
    # -------------------------------------------------------------------
    if animate:
        frames = []
        animated_indices = list(block_trace_indices) + [shaft_trace_idx]
        if show_arrows and cone_pos_idx >= 0:
            animated_indices.extend([cone_pos_idx, cone_neg_idx])
        for k in range(n_frames):
            theta = 2.0 * math.pi * k / n_frames
            frame_data: List[Any] = []
            # Bloques deformados con intensity colormap
            for (xs0, ys0, zs0, ii, jj, kk, kind, color, opacity, name) in block_meshes_base:
                new_x, new_y, new_z, intensity = _displace_mesh(xs0, ys0, zs0, kind, theta)
                frame_data.append(go.Mesh3d(
                    x=new_x, y=new_y, z=new_z, i=ii, j=jj, k=kk,
                    intensity=intensity, intensitymode="vertex",
                    colorscale=colormap, cmin=-int_max, cmax=int_max,
                    showscale=False, opacity=0.92,
                    name=name, showlegend=False, hoverinfo="skip",
                    flatshading=False,
                ))
            # Shaft deformado con intensity
            xs0, ys0, zs0, ii, jj, kk, kind, color, opacity, name = shaft_base
            new_x, new_y, new_z, intensity = _displace_mesh(xs0, ys0, zs0, kind, theta)
            frame_data.append(go.Mesh3d(
                x=new_x, y=new_y, z=new_z, i=ii, j=jj, k=kk,
                intensity=intensity, intensitymode="vertex",
                colorscale=colormap, cmin=-int_max, cmax=int_max,
                showscale=False, opacity=1.0,
                name=name, showlegend=False, hoverinfo="skip",
                flatshading=False,
            ))
            # Cones (opcional)
            if show_arrows and cone_pos_idx >= 0:
                cone_pos_k, cone_neg_k = _cones_at_phase(theta)
                frame_data.append(cone_pos_k)
                frame_data.append(cone_neg_k)
            frames.append(go.Frame(
                data=frame_data,
                traces=animated_indices,
                name=f"{int(math.degrees(theta)):03d}",
            ))
        fig.frames = frames

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
                              transition=dict(duration=frame_duration_ms // 2,
                                                easing="linear"),
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
                active=0,
                currentvalue=dict(prefix="Fase del modo: ", suffix="°"),
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


# ---------------------------------------------------------------------
# Export animado con header KPI integrado (para enviar a cliente)
# Soporta GIF (universal) y MP4 H.264 (mejor calidad/peso, profesional)
# ---------------------------------------------------------------------

def _render_mode_shape_frames(
    geom: "ModalGeometry",
    mode_shape: Any,
    channel_names: List[str],
    mode_number: int,
    freq_hz: float,
    damping_pct: float,
    running_rpm: float,
    classification: str,
    mpc_pct: float,
    n_frames: int,
    width_px: int,
    height_px: int,
    colormap: str,
    show_ghost: bool,
    asset_name: str,
) -> List[Any]:
    """Helper compartido: genera los frames PIL.Image con header KPI."""
    import io
    import math as _math
    import plotly.io as pio
    from PIL import Image, ImageDraw, ImageFont

    # Dimensiones del header
    header_h = 110
    plot_h = height_px - header_h

    # Pre-compute KPIs constantes
    fn_cpm = freq_hz * 60.0
    order = fn_cpm / max(running_rpm, 1.0)
    q_factor = 1.0 / (2 * max(damping_pct / 100.0, 1e-6))
    cls_color = {"natural": "#16a34a",
                  "harmonic": "#D89B22",
                  "spurious": "#dc2626"}.get(classification, "#475569")

    # Font setup (fallback a default si no encuentra)
    try:
        font_xl = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 28)
        font_lg = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)
        font_md = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
        font_sm = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 11)
    except (OSError, IOError):
        try:
            font_xl = ImageFont.truetype("DejaVuSans-Bold.ttf", 28)
            font_lg = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
            font_md = ImageFont.truetype("DejaVuSans.ttf", 14)
            font_sm = ImageFont.truetype("DejaVuSans.ttf", 11)
        except (OSError, IOError):
            font_xl = ImageFont.load_default()
            font_lg = font_xl; font_md = font_xl; font_sm = font_xl

    def _draw_header(canvas: Image.Image, phase_deg: float):
        d = ImageDraw.Draw(canvas)
        # Background gradient (simple navy)
        d.rectangle([0, 0, width_px, header_h], fill="#0F1E3D")
        # Logo / titulo izquierda
        d.text((20, 14), "Watermelon Modal", font=font_md, fill="#1AAEE5")
        d.text((20, 32), asset_name, font=font_lg, fill="white")
        # KPI cards
        x0 = 280
        col_w = (width_px - x0 - 20) // 5
        kpis = [
            ("MODO", f"#{mode_number}", ""),
            ("FRECUENCIA", f"{freq_hz:.2f} Hz",
             f"{fn_cpm:,.0f} CPM · {order:.3f}× run"),
            ("DAMPING", f"{damping_pct:.3f}%", f"Q = {q_factor:.1f}"),
            ("MPC", f"{mpc_pct:.1f}%", "complejidad"),
            ("CLASE", classification.upper(), f"fase {phase_deg:.0f}°"),
        ]
        for i, (lbl, val, sub) in enumerate(kpis):
            x = x0 + i * col_w
            d.text((x, 18), lbl, font=font_sm, fill="#94a3b8")
            color = cls_color if i == 4 else "white"
            d.text((x, 32), val, font=font_lg, fill=color)
            if sub:
                d.text((x, 64), sub, font=font_sm, fill="#94a3b8")
            # Divider vertical
            if i > 0:
                d.line([(x - 10, 22), (x - 10, header_h - 18)],
                        fill="#1e3a5f", width=1)
        # Footer ribbon
        d.rectangle([0, header_h - 4, width_px, header_h], fill="#1AAEE5")

    # Renderizar frames
    frames: List[Image.Image] = []
    for k in range(n_frames):
        theta = 2.0 * _math.pi * k / n_frames
        phase_deg = _math.degrees(theta)
        # Generar figura estatica en este theta — re-uso del builder con animate=False
        # pero forzando el frame θ usando una copia complex rotada
        import numpy as np
        ms_rot = np.asarray(mode_shape, dtype=complex).flatten()
        ms_rot = ms_rot * (np.cos(theta) + 1j * np.sin(theta))
        # Render estatico en esta fase
        fig = build_geometry_with_mode_shape(
            geom=geom, mode_shape=ms_rot, channel_names=channel_names,
            mode_label="", animate=False,
            show_arrows=False, show_ghost=show_ghost, colormap=colormap,
        )
        # Quitar titulo y margenes
        fig.update_layout(title=None, margin=dict(l=0, r=20, t=0, b=0),
                           height=plot_h, width=width_px)
        png_bytes = pio.to_image(fig, format="png",
                                   width=width_px, height=plot_h, scale=1)
        plot_img = Image.open(io.BytesIO(png_bytes))
        # Composicion: header arriba + plot abajo
        canvas = Image.new("RGB", (width_px, height_px), "white")
        _draw_header(canvas, phase_deg)
        canvas.paste(plot_img, (0, header_h))
        frames.append(canvas)

    return frames


# ---------------------------------------------------------------------
# Export GIF (universal, soporta WhatsApp/email/cualquier visor)
# ---------------------------------------------------------------------

def export_mode_shape_gif(
    geom: "ModalGeometry",
    mode_shape: Any,
    channel_names: List[str],
    mode_number: int,
    freq_hz: float,
    damping_pct: float,
    running_rpm: float = 3600.0,
    classification: str = "natural",
    mpc_pct: float = 0.0,
    n_frames: int = 36,
    frame_duration_ms: int = 280,
    width_px: int = 1280,
    height_px: int = 720,
    colormap: str = "RdBu_r",
    show_ghost: bool = True,
    asset_name: str = "Activo",
) -> bytes:
    """Renderiza GIF animado del mode shape con header KPI integrado."""
    import io
    frames = _render_mode_shape_frames(
        geom=geom, mode_shape=mode_shape, channel_names=channel_names,
        mode_number=mode_number, freq_hz=freq_hz, damping_pct=damping_pct,
        running_rpm=running_rpm, classification=classification,
        mpc_pct=mpc_pct, n_frames=n_frames,
        width_px=width_px, height_px=height_px, colormap=colormap,
        show_ghost=show_ghost, asset_name=asset_name,
    )
    buf = io.BytesIO()
    frames[0].save(
        buf, format="GIF", save_all=True, append_images=frames[1:],
        duration=frame_duration_ms, loop=0, optimize=False,
    )
    return buf.getvalue()


# ---------------------------------------------------------------------
# Export MP4 H.264 (profesional, mejor calidad/peso, ideal cliente)
# ---------------------------------------------------------------------

def export_mode_shape_mp4(
    geom: "ModalGeometry",
    mode_shape: Any,
    channel_names: List[str],
    mode_number: int,
    freq_hz: float,
    damping_pct: float,
    running_rpm: float = 3600.0,
    classification: str = "natural",
    mpc_pct: float = 0.0,
    n_frames: int = 48,
    fps: int = 12,
    width_px: int = 1280,
    height_px: int = 720,
    colormap: str = "RdBu_r",
    show_ghost: bool = True,
    asset_name: str = "Activo",
    quality: int = 8,
) -> bytes:
    """Renderiza MP4 H.264 del mode shape con header KPI integrado.

    Usa imageio + imageio-ffmpeg (binarios ffmpeg auto-descargados al install).
    H.264 yuv420p para maxima compatibilidad (WhatsApp, iPhone, Android, web).

    Args:
        fps: frames por segundo (12 = ciclo de 4 segundos con 48 frames)
        quality: 0-10, mayor=mejor calidad y mas peso (8 = optimo cliente)

    Returns: bytes del MP4 listo para st.download_button con mime=video/mp4.
    """
    import io
    import tempfile
    import os
    import numpy as np
    try:
        import imageio.v2 as imageio  # type: ignore[import-untyped]
    except ImportError:
        import imageio  # type: ignore[import-untyped]

    frames = _render_mode_shape_frames(
        geom=geom, mode_shape=mode_shape, channel_names=channel_names,
        mode_number=mode_number, freq_hz=freq_hz, damping_pct=damping_pct,
        running_rpm=running_rpm, classification=classification,
        mpc_pct=mpc_pct, n_frames=n_frames,
        width_px=width_px, height_px=height_px, colormap=colormap,
        show_ghost=show_ghost, asset_name=asset_name,
    )

    # Loop suave: agregamos los frames de nuevo invertidos para evitar
    # el "salto" al volver al frame 0 (estilo Artemis MP4 ping-pong)
    np_frames = [np.array(f) for f in frames]
    # Aseguramos dimensiones pares (H.264 requiere alturas y anchos pares)
    h, w = np_frames[0].shape[:2]
    if h % 2 or w % 2:
        new_h = h - (h % 2); new_w = w - (w % 2)
        np_frames = [f[:new_h, :new_w] for f in np_frames]

    # Escribir MP4 a tempfile (ffmpeg necesita path, no stream)
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
        tmp_path = tf.name
    try:
        imageio.mimwrite(
            tmp_path, np_frames,
            fps=fps,
            codec="libx264",
            quality=quality,
            macro_block_size=1,    # evita warnings de divisibilidad
            pixelformat="yuv420p", # maxima compatibilidad
        )
        with open(tmp_path, "rb") as f:
            data = f.read()
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
    return data
