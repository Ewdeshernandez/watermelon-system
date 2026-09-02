"""
core/modal/oma_layout.py — Layout de ensayo OMA: puntos de medición + geometría
===============================================================================

Modela DÓNDE va cada acelerómetro en la máquina y con qué REFERENCIA física, tal
como se hace en un ensayo modal profesional: cada canal es un "punto de medición"
ubicado en un componente (Motor / Bomba / Skid / Tubería) con una referencia de
posición (lado libre / lado acople / centro …) y una dirección (DOF: +X/+Y/+Z).

Esto es lo que da sentido físico a las formas modales y a la animación: sin la
geometría y las referencias, un modo es solo un número.

Mapeo de hardware: chasis NI cDAQ (p.ej. 9178, 8 slots) con módulos NI 9234
(4 canales IEPE c/u). 24 canales = 6 módulos (slots 1..6). Todo configurable.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence

# Componentes típicos de un tren motor-bomba con skid y tuberías
DEFAULT_COMPONENTS = ["Electric motor", "Coupling", "Single-stage pump", "Suction pipe", "Discharge pipe"]

# Standard position references (editable). NDE = free end, DE = drive/coupling end (API 670 / ISO 20816)
POSITION_REFS = ["NDE (free end)", "DE (coupling end)", "Center", "Top", "Bottom",
                 "Suction", "Discharge", "Base", "Flange"]

# Measurement directions (DOFs)
DOFS = ["+X", "+Y", "+Z", "-X", "-Y", "-Z", "H", "V", "A"]  # H=horizontal V=vertical A=axial

# Measurement type per point (ISO 7626): A=acceleration, V=velocity, D=displacement
MEAS_TYPES = ["A", "V", "D"]
MEAS_TYPE_NAME = {"A": "Acceleration", "V": "Velocity", "D": "Displacement"}

# Equipment types that can be drawn (solid boxes on the skid) — enriched database
COMPONENT_KINDS = [
    "Electric motor", "Combustion engine", "Gas turbine", "Steam turbine",
    "Axial compressor", "Centrifugal compressor", "Reciprocating compressor", "Screw compressor",
    "Multistage pump", "Screw pump", "Single-stage pump",
    "Gearbox", "Generator", "Fan / Blower", "Coupling", "Bearing housing",
    "Skid 1", "Skid 2", "Suction pipe", "Discharge pipe",
]

# Rotating machines (carry NDE/DE bearings) — used by norm-based placement
ROTATING_KINDS = ["Electric motor", "Combustion engine", "Gas turbine", "Steam turbine",
                  "Axial compressor", "Centrifugal compressor", "Reciprocating compressor",
                  "Screw compressor", "Multistage pump", "Screw pump", "Single-stage pump",
                  "Gearbox", "Generator", "Fan / Blower"]

DEFAULT_SENSITIVITY_MV_PER_G = 100.0


def is_pipe(kind: str) -> bool:
    return "pipe" in (kind or "").lower()


def component_default_box(kind: str, x0: float):
    """Caja por defecto (x0,x1,y0,y1,depth) según el tipo de equipo."""
    k = (kind or "").lower()
    if is_pipe(kind):
        if "suction" in k:
            return (x0, x0 + 0.32, 0.02, 0.08, 0.04)
        return (x0, x0 + 0.18, 0.08, 0.42, 0.04)
    if "coupling" in k:
        return (x0, x0 + 0.06, 0.05, 0.14, 0.06)
    if "skid" in k:
        return (x0, x0 + 0.40, -0.10, 0.00, 0.22)
    if "gearbox" in k or "bearing" in k:
        return (x0, x0 + 0.18, 0.00, 0.16, 0.11)
    if "turbine" in k or "compressor" in k or "engine" in k:
        return (x0, x0 + 0.38, 0.00, 0.20, 0.12)   # más largos
    return (x0, x0 + 0.30, 0.00, 0.19, 0.11)        # motor / bomba / generador / fan


# Recomendaciones de adquisición por norma (ISO 7626 EMA / ISO 20816 OMA)
def recommended_acquisition(test_mode: str) -> dict:
    """Devuelve parámetros recomendados por norma + una nota explicativa."""
    if (test_mode or "").upper().startswith("EMA"):
        return {"fs_hz": 2048.0, "block_size": 4096, "fmax_hz": 800.0, "duration_s": 30.0,
                "window": "force+exp", "averages": 5,
                "note": ("ISO 7626-5 (martillo): Fmax que cubra los modos de interés, ventana de "
                         "fuerza + exponencial, 3–5 promedios por punto, coherencia ≥ 0.8 en banda. "
                         "Δf fino (block grande) para damping.")}
    return {"fs_hz": 1280.0, "block_size": 4096, "fmax_hz": 200.0, "duration_s": 600.0,
            "window": "hanning", "averages": 1,
            "note": ("ISO 20816 / OMA (Brincker & Ventura): registro LARGO (≥ 1000–2000 ciclos del "
                     "modo más bajo, típ. 5–10 min), muestreo simultáneo, Fmax según banda de interés, "
                     "sin ventana de fuerza (excitación ambiental/operacional).")}


@dataclass
class MachineComponent:
    """Un equipo dibujado como caja sobre el skid (para armar la máquina)."""
    kind: str                    # Motor / Acople / Turbina / Bomba / Generador / Gear box / Tubería …
    label: str = ""              # nombre visible (por defecto = kind)
    x0: float = 0.0              # a lo largo del eje del tren (X), editable "a medida"
    x1: float = 0.18
    y0: float = -0.30            # altura (Z en el dibujo 3D): base..tope
    y1: float = 0.30
    depth: float = 0.16          # semi-profundidad (Y) para el sólido 3D

    def display(self) -> str:
        return self.label or self.kind


@dataclass
class MeasPoint:
    """Un punto de medición = un acelerómetro ubicado en la máquina."""
    idx: int                      # número de punto (1..N)
    component: str                # Motor / Bomba / Skid / Tubería …
    position_ref: str             # referencia física (LL, LA, Centro, Succión…)
    dof: str                      # dirección (+X/+Y/+Z…)
    module_slot: int              # slot del módulo NI (1..8)
    channel_index: int            # canal dentro del módulo (0..3 en el 9234)
    sensitivity_mv_per_g: float = DEFAULT_SENSITIVITY_MV_PER_G
    unit: str = "g"
    coupling: str = "IEPE"
    reference_sensor: bool = False   # ¿es sensor de referencia (fijo) del OMA?
    active: bool = True
    number: int = 0               # número de punto para la etiqueta (1,2,3…); 0 = usa idx
    meas_type: str = "A"          # A=aceleración, V=velocidad, D=desplazamiento
    # posición esquemática en el dibujo de la máquina (0..1 a lo largo del tren, y por DOF)
    x_norm: float = 0.0
    y_norm: float = 0.0

    @property
    def axis(self) -> str:
        """Eje sin signo: +Y->Y, -X->X …"""
        return (self.dof or "").replace("+", "").replace("-", "").strip() or "Y"

    @property
    def code(self) -> str:
        """Etiqueta tipo instrumento: número + eje + tipo, ej. '1XA' (punto 1, X, aceleración)."""
        n = self.number or self.idx
        return f"{n}{self.axis}{self.meas_type}"

    @property
    def bnc(self) -> int:
        """BNC global 1..N (para el rótulo físico)."""
        return (self.module_slot - 1) * 4 + self.channel_index + 1

    @property
    def label(self) -> str:
        """Etiqueta corta del punto, ej. 'Motor LL +Y'."""
        ref = self.position_ref.split(" ")[0]     # 'LL (lado libre)' -> 'LL'
        return f"{self.component} {ref} {self.dof}".strip()

    @property
    def tag(self) -> str:
        """Identificador de canal para datos/análisis (sin espacios)."""
        ref = self.position_ref.split(" ")[0]
        return f"P{self.idx:02d}_{self.component[:4]}_{ref}_{self.dof}".replace("+", "p").replace("-", "m")


@dataclass
class OMALayout:
    """Configuración completa del ensayo modal (máquina del cliente + puntos +
    adquisición). Sirve para EMA y OMA."""
    name: str = "Tren Motor-Bomba"
    # --- ficha del activo del cliente ---
    machine_type: str = ""               # ej. "Motor-Bomba centrífuga"
    tag: str = ""                        # tag / placa
    client: str = ""                     # cliente
    location: str = ""                   # planta / ubicación
    components: List[str] = field(default_factory=lambda: list(DEFAULT_COMPONENTS))
    machine_components: List[MachineComponent] = field(default_factory=list)  # dibujo (cajas)
    points: List[MeasPoint] = field(default_factory=list)
    # --- adquisición (compartida EMA/OMA) ---
    test_modes: List[str] = field(default_factory=lambda: ["OMA"])  # ["EMA"], ["OMA"] o ambos
    test_type: str = "OMA"               # (compat) modo principal
    fs_hz: float = 1280.0
    block_size: int = 4096               # EMA: muestras por golpe
    fmax_hz: float = 200.0
    duration_s: float = 300.0            # OMA: registros largos
    chassis: str = "cDAQ-9178"
    module_model: str = "NI 9234"
    running_speed_rpm: float = 1185.0

    # ---- utilidades ----
    def active_points(self) -> List[MeasPoint]:
        return [p for p in self.points if p.active]

    def channel_names(self) -> List[str]:
        # usa el código instrumento (1XA…); de-duplica si hace falta
        names: List[str] = []
        seen: dict = {}
        for p in self.active_points():
            c = p.code
            if c in seen:
                seen[c] += 1; c = f"{c}_{seen[c]}"
            else:
                seen[c] = 0
            names.append(c)
        return names

    def n_channels(self) -> int:
        return len(self.active_points())

    def references(self) -> List[MeasPoint]:
        return [p for p in self.active_points() if p.reference_sensor]

    def validate(self) -> List[str]:
        """Devuelve lista de problemas (vacía = OK)."""
        errs: List[str] = []
        pts = self.active_points()
        if not pts:
            errs.append("No hay puntos de medición activos.")
        # BNC/slot-canal duplicados
        seen = {}
        for p in pts:
            key = (p.module_slot, p.channel_index)
            if key in seen:
                errs.append(f"Canal duplicado: {p.label} y {seen[key]} usan slot {p.module_slot} "
                            f"canal {p.channel_index}.")
            seen[key] = p.label
            if p.sensitivity_mv_per_g <= 0:
                errs.append(f"{p.label}: sensibilidad inválida.")
        if not self.references():
            errs.append("OMA recomienda al menos un sensor de REFERENCIA fijo (marcá 'reference').")
        return errs

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @staticmethod
    def from_dict(d: dict) -> "OMALayout":
        from dataclasses import fields as _f
        d = dict(d or {})
        okp = {f.name for f in _f(MeasPoint)}
        d["points"] = [MeasPoint(**{k: v for k, v in (p or {}).items() if k in okp})
                       for p in d.get("points", [])]
        okc = {f.name for f in _f(MachineComponent)}
        d["machine_components"] = [MachineComponent(**{k: v for k, v in (c or {}).items() if k in okc})
                                   for c in d.get("machine_components", [])]
        okl = {f.name for f in _f(OMALayout)}
        return OMALayout(**{k: v for k, v in d.items() if k in okl})


def default_components() -> List[MachineComponent]:
    """In-line train: electric motor → coupling → pump + piping (elongated 3D solids)."""
    return [
        MachineComponent("Electric motor", "Electric motor", 0.03, 0.34, 0.00, 0.19, depth=0.11),
        MachineComponent("Coupling", "Coupling", 0.34, 0.40, 0.05, 0.14, depth=0.06),
        MachineComponent("Single-stage pump", "Pump", 0.40, 0.66, 0.00, 0.17, depth=0.10),
        MachineComponent("Suction pipe", "Suction pipe", 0.66, 0.98, 0.02, 0.08, depth=0.04),
        MachineComponent("Discharge pipe", "Discharge pipe", 0.66, 0.82, 0.08, 0.42, depth=0.04),
    ]


# --- guardado LOCAL (JSON) ---
def layouts_dir() -> str:
    d = os.environ.get("WM_MODAL_DIR") or os.path.join(os.path.expanduser("~"), ".watermelon", "modal")
    os.makedirs(d, exist_ok=True)
    return d


def _slug(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in (name or "modal")).strip("_")


def save_layout_local(layout: "OMALayout") -> str:
    p = os.path.join(layouts_dir(), f"{_slug(layout.name)}.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(layout.to_dict(), f, ensure_ascii=False, indent=2)
    return p


def list_layouts_local() -> List[str]:
    d = layouts_dir()
    return sorted(f[:-5] for f in os.listdir(d) if f.endswith(".json"))


def load_layout_local(name_or_slug: str) -> "OMALayout":
    p = os.path.join(layouts_dir(), f"{_slug(name_or_slug)}.json")
    with open(p, "r", encoding="utf-8") as f:
        return OMALayout.from_dict(json.load(f))


def auto_place_by_norm(layout: "OMALayout",
                       sensitivity: float = DEFAULT_SENSITIVITY_MV_PER_G) -> None:
    """Ubica los puntos según norma (API 670 / ISO 20816-1): en cada equipo
    rotativo, cojinetes NDE (lado libre) y DE (lado acople), con X/Y/Z, numerados
    de conductor a conducido. Modifica layout.points in-place."""
    rot = sorted([c for c in layout.machine_components if c.kind in ROTATING_KINDS],
                 key=lambda c: c.x0)
    pts: List[MeasPoint] = []; number = 1; n = 0
    for j, c in enumerate(rot):
        if j == 0:
            ends = [(c.x0, "NDE (lado libre)"), (c.x1, "DE (lado acople)")]
        elif j == len(rot) - 1:
            ends = [(c.x0, "DE (lado acople)"), (c.x1, "NDE (lado libre)")]
        else:
            ends = [(c.x0, "DE (lado acople)"), (c.x1, "DE (lado acople)")]
        ztop = c.y1
        for (xe, ref) in ends:
            for ax, off in (("X", 0.0), ("Y", 0.035), ("Z", -0.035)):
                slot = n // 4 + 1; ch = n % 4
                pts.append(MeasPoint(
                    idx=n + 1, component=c.kind, position_ref=ref, dof="+" + ax,
                    module_slot=slot, channel_index=ch, sensitivity_mv_per_g=sensitivity,
                    number=number, meas_type="A", reference_sensor=(number == 1 and ax == "X"),
                    x_norm=float(xe), y_norm=float(ztop + off)))
                n += 1
            number += 1
    layout.points = pts


# =====================================================================
# Plantilla: tren motor-bomba con 24 canales (6 módulos NI 9234)
# =====================================================================
def default_24ch_layout(name: str = "Tren Motor-Bomba P-762007",
                        sensitivity: float = DEFAULT_SENSITIVITY_MV_PER_G) -> OMALayout:
    """24 acelerómetros distribuidos en motor, bomba, skid y tuberías, con
    referencias físicas y DOFs, mapeados a 6 módulos NI 9234 (slots 1..6)."""
    # (componente, referencia, [dofs], x_norm) — x_norm ubica el punto a lo largo del tren
    plan = [
        ("Motor", "LA (lado acople)", ["+Y", "+X"], 0.18),
        ("Motor", "LL (lado libre)",  ["+Y", "+X"], 0.06),
        ("Bomba", "LA (lado acople)", ["+Y", "+X"], 0.34),
        ("Bomba", "LL (lado libre)",  ["+Y", "+X"], 0.46),
        ("Skid",  "Motor",            ["+Y", "+Z"], 0.12),
        ("Skid",  "Bomba",            ["+Y", "+Z"], 0.40),
        ("Skid",  "Centro",           ["+Y", "+Z"], 0.26),
        ("Tubería succión",  "Brida", ["+Y", "+X"], 0.60),
        ("Tubería succión",  "Centro",["+Y", "+Z"], 0.72),
        ("Tubería descarga", "Brida", ["+Y", "+X"], 0.84),
        ("Tubería descarga", "Centro",["+Y", "+Z"], 0.94),
        ("Bomba", "Superior",         ["+Z", "+Y"], 0.40),
    ]
    points: List[MeasPoint] = []
    n = 0
    for comp, ref, dofs, x in plan:
        pt_number = len({p.component + p.position_ref for p in points}) + 1
        for d in dofs:
            if n >= 24:
                break
            slot = n // 4 + 1
            ch = n % 4
            yv = {"+Y": 0.0, "+X": 0.25, "+Z": -0.25}.get(d, 0.0)
            points.append(MeasPoint(
                idx=n + 1, component=comp, position_ref=ref, dof=d,
                module_slot=slot, channel_index=ch, sensitivity_mv_per_g=sensitivity,
                number=pt_number, meas_type="A",       # acelerómetros → aceleración
                x_norm=x, y_norm=yv,
                reference_sensor=(n in (0, 4))))
            n += 1
    return OMALayout(name=name, points=points, machine_components=default_components())
