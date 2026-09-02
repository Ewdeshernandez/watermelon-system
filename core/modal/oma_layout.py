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

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence

# Componentes típicos de un tren motor-bomba con skid y tuberías
DEFAULT_COMPONENTS = ["Motor", "Bomba", "Skid", "Tubería succión", "Tubería descarga"]

# Referencias de posición estándar (editable). LL=lado libre (NDE), LA=lado acople (DE)
POSITION_REFS = ["LL (lado libre)", "LA (lado acople)", "Centro", "Superior", "Inferior",
                 "Succión", "Descarga", "Base", "Brida"]

# Direcciones de medición (grados de libertad)
DOFS = ["+X", "+Y", "+Z", "-X", "-Y", "-Z", "H", "V", "A"]  # H=horizontal V=vertical A=axial

DEFAULT_SENSITIVITY_MV_PER_G = 100.0


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
    # posición esquemática en el dibujo de la máquina (0..1 a lo largo del tren, y por DOF)
    x_norm: float = 0.0
    y_norm: float = 0.0

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
    points: List[MeasPoint] = field(default_factory=list)
    # --- adquisición (compartida EMA/OMA) ---
    test_type: str = "OMA"               # "EMA" | "OMA"
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
        return [p.label for p in self.active_points()]

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
        ok = {f.name for f in _f(MeasPoint)}
        d["points"] = [MeasPoint(**{k: v for k, v in (p or {}).items() if k in ok})
                       for p in d.get("points", [])]
        okl = {f.name for f in _f(OMALayout)}
        return OMALayout(**{k: v for k, v in d.items() if k in okl})


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
        for d in dofs:
            if n >= 24:
                break
            slot = n // 4 + 1
            ch = n % 4
            yv = {"+Y": 0.0, "+X": 0.25, "+Z": -0.25}.get(d, 0.0)
            points.append(MeasPoint(
                idx=n + 1, component=comp, position_ref=ref, dof=d,
                module_slot=slot, channel_index=ch, sensitivity_mv_per_g=sensitivity,
                x_norm=x, y_norm=yv,
                reference_sensor=(n in (0, 4))))   # 2 sensores de referencia fijos
            n += 1
    return OMALayout(name=name, points=points)
