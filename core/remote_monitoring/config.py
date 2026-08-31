"""
core/remote_monitoring/config.py — Config amigable de máquina + canales
=======================================================================

Modelo de configuración estilo ADRE 408 / System1 pero simple:

  MachineConfig  — propiedades del tren (rpm, sentido de giro, speed control,
                   cojinete). Equivale a "Machine Properties" de System1
                   (Tarea 2 del taller T00336).
  ChannelRow     — una fila del grid de canales: BNC físico → punto de
                   medición (tipo, sensib, unidad, coupling, ángulo, lado,
                   Alert/Danger). Equivale al "Mapping" de System1 (Tarea 3)
                   fundido con el punto de medición.
  AcqSetup       — máquina + canales. La unidad que se guarda y desde la que
                   arranca el stream.

Fuente única de verdad: `setup_to_sensor_map()` produce dicts compatibles
con core.sensor_map.new_sensor(), y `setup_to_channel_configs()` produce
los ChannelConfig de adquisición. Así esta UI amigable NO crea un modelo
paralelo — escribe al mismo modelo que consume el resto del sistema.

Marco: API 670 (protección/transductores), API 684 (rotordinámica),
ISO 20816 (evaluación). Las validaciones siguen esos criterios.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field, asdict, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# =====================================================================
# Catálogos (rangos físicos típicos para validación API 670)
# =====================================================================
SENSOR_TYPES = ["proximity", "velometer", "accelerometer", "keyphasor"]
COUPLINGS = ["AC", "DC", "IEPE"]
ROTATIONS = ["CW", "CCW"]
SPEED_CONTROLS = ["constant", "variable"]
BEARING_TYPES = ["plain", "tilting_pad", "rolling", "mixed"]

# Unidades válidas por tipo de sensor (la 1ª es el default).
UNITS_BY_TYPE = {
    "proximity": ["mil pp", "µm pp", "mil 0-pk", "µm 0-pk"],
    "velometer": ["mm/s rms", "mm/s pk", "in/s pk", "in/s rms"],
    "accelerometer": ["g rms", "g pk", "m/s² rms", "m/s² pk"],
    "keyphasor": ["rpm", "pulses/rev"],
}
# Unión ordenada de todas las unidades (para el dropdown del grid).
ALL_UNITS = list(dict.fromkeys(u for lst in UNITS_BY_TYPE.values() for u in lst))


def valid_units_for(sensor_type: str) -> List[str]:
    return UNITS_BY_TYPE.get(sensor_type, [])


def default_unit(sensor_type: str) -> str:
    lst = UNITS_BY_TYPE.get(sensor_type, [""])
    return lst[0] if lst else ""


# Unidad nativa por defecto según tipo de sensor
_DEFAULT_UNIT = {t: (lst[0] if lst else "") for t, lst in UNITS_BY_TYPE.items()}
# Coupling por defecto según tipo
_DEFAULT_COUPLING = {
    "proximity": "AC",
    "velometer": "AC",
    "accelerometer": "IEPE",
    "keyphasor": "DC",
}
# Sensibilidad por defecto (mV/EU)
_DEFAULT_SENS = {
    "proximity": 200.0,     # mV/mil (Bently 3300/3500)
    "velometer": 100.0,     # mV/(mm/s) — depende, valor razonable
    "accelerometer": 100.0, # mV/g
    "keyphasor": 1.0,
}
# Rango físico razonable de sensibilidad por tipo (para warning)
_SENS_RANGE = {
    "proximity": (100.0, 300.0),
    "velometer": (50.0, 600.0),
    "accelerometer": (10.0, 1000.0),
    "keyphasor": (0.0, 1e9),
}

# Tokens de keyphasor (mismo criterio que stream_source)
_KPH_TOKENS = ("kph", "keyphasor", "keyph", "tach", "tacho", "trigger")


# =====================================================================
# Dataclasses
# =====================================================================
@dataclass
class MachineConfig:
    name: str = "Máquina ad-hoc"
    template_id: str = ""
    rpm_nominal: float = 3600.0
    rpm_min: float = 0.0
    rpm_max: float = 0.0
    rotation: str = "CCW"            # CW | CCW
    speed_control: str = "constant"  # constant | variable
    bearing_type: str = "plain"      # plain | tilting_pad | rolling | mixed
    n_bearings: int = 2
    iso_norm: str = ""
    # --- Ficha del activo (homólogo a la web / ADRE 408) ---
    machine_type: str = ""           # ej. "Turbogenerator", "Motor+Pump"
    tag: str = ""                    # tag / placa de la máquina
    client: str = ""                 # cliente
    location: str = ""               # planta / ubicación

    def __post_init__(self) -> None:
        if self.rotation not in ROTATIONS:
            self.rotation = "CCW"
        if self.speed_control not in SPEED_CONTROLS:
            self.speed_control = "constant"


@dataclass
class ChannelRow:
    bnc_port: int
    point_label: str
    plane: int = 1
    sensor_type: str = "proximity"
    sensitivity_mv_per_eu: float = 200.0
    unit_native: str = "mil pp"
    coupling: str = "AC"
    angle_deg: float = 0.0
    side: str = ""
    alarm: float = 0.0
    danger: float = 0.0
    # --- Fase campos ADRE 408 / API 670 ---
    full_scale: float = 0.0        # rango de medición (EU). 0 = auto
    gap_bias_v: float = 0.0        # voltaje gap/bias nominal (V) — sondas prox DC
    active: bool = True            # canal activo (recolecta) — ADRE "Active"
    events_per_rev: int = 1        # keyphasor: pulsos por revolución
    trigger_v: float = 0.0         # keyphasor: umbral de disparo (V)
    notch_type: str = ""           # keyphasor: muesca | proyección | rueda dentada
    keyphasor_ref: str = ""        # punto keyphasor asociado (referencia de fase 1X)
    pair_ref: str = ""             # punto par X/Y para la órbita

    def is_keyphasor(self) -> bool:
        if self.sensor_type == "keyphasor":
            return True
        nm = (self.point_label or "").lower()
        return any(tok in nm for tok in _KPH_TOKENS)


# Parámetros de adquisición GLOBALES (por medición, no por canal) — igual que
# System1 "Spectrums & Waveforms". Definen la calidad del espectro/bode/cascade.
# Ventanas estándar de análisis de vibración (3): Hanning (general),
# Flat-Top (exactitud de amplitud), Uniform/Rectangular (transitorios/impacto).
WINDOWS = ["hanning", "flattop", "uniform"]
LINES_OPTIONS = [400, 800, 1600, 3200, 6400]
WAVEFORM_MODES = ["synchronous", "asynchronous"]
COMMON_ORDERS = [0.5, 1.0, 2.0, 3.0]        # órdenes ×rpm típicos a trackear
NOTCH_TYPES = ["", "muesca", "proyección", "rueda dentada"]
FREQ_UNITS = ["cpm", "hz"]                  # unidad de display de frecuencia


def freq_label(unit: str) -> str:
    return "CPM" if (unit or "").lower() == "cpm" else "Hz"


def hz_to_display(hz: float, unit: str) -> float:
    return hz * 60.0 if (unit or "").lower() == "cpm" else hz


def display_to_hz(val: float, unit: str) -> float:
    return val / 60.0 if (unit or "").lower() == "cpm" else val


@dataclass
class AcquisitionParams:
    fmax_hz: float = 1000.0        # frecuencia máxima del espectro (span) — SIEMPRE en Hz interno
    fmin_hz: float = 2.0           # corte pasa-altos (quita DC/deriva) — Hz interno
    lines: int = 1600              # líneas de resolución → Δf
    averages: int = 4              # promedios espectrales
    window: str = "hanning"        # ventana FFT
    samples_per_rev: int = 0       # muestreo síncrono (0 = auto)
    waveform_mode: str = "synchronous"   # synchronous | asynchronous
    orders: List[float] = field(default_factory=lambda: [1.0, 2.0])  # 1X, 2X por defecto (ADRE)
    freq_unit: str = "cpm"         # cpm | hz — preferencia de display (CPM por defecto)

    def delta_f(self) -> float:
        return self.fmax_hz / self.lines if self.lines else 0.0


# Tipos con espectro (el keyphasor no lleva params espectrales).
SPECTRAL_TYPES = ["proximity", "velometer", "accelerometer"]


def default_acq_for_type(sensor_type: str) -> "AcquisitionParams":
    """Defaults de adquisición según la física del sensor:
    proximidad = baja frecuencia (eje, 1X–10X); acelerómetro = alta
    frecuencia (rodamientos/engrane); velocidad = intermedio."""
    if sensor_type == "accelerometer":
        return AcquisitionParams(fmax_hz=10000.0, fmin_hz=10.0, lines=3200)
    if sensor_type == "velometer":
        return AcquisitionParams(fmax_hz=2000.0, fmin_hz=2.0, lines=1600)
    return AcquisitionParams(fmax_hz=1000.0, fmin_hz=2.0, lines=1600)  # proximity


@dataclass
class AcqSetup:
    machine: MachineConfig = field(default_factory=MachineConfig)
    channels: List[ChannelRow] = field(default_factory=list)
    # Adquisición GLOBAL del tren (waveform_mode, orders, samples_per_rev,
    # freq_unit). Sus fmax/fmin/lines sirven de fallback.
    acquisition: AcquisitionParams = field(default_factory=AcquisitionParams)
    # Adquisición POR TIPO de sensor (fmax/fmin/lines/window/averages) —
    # una máquina mixta prox+accel usa bandas distintas.
    acquisition_by_type: Dict[str, AcquisitionParams] = field(
        default_factory=lambda: {t: default_acq_for_type(t) for t in SPECTRAL_TYPES})

    def keyphasor_row(self) -> Optional[ChannelRow]:
        for ch in self.channels:
            if ch.is_keyphasor():
                return ch
        return None

    def acq_for(self, sensor_type: str) -> AcquisitionParams:
        """Params de adquisición efectivos para un tipo de sensor."""
        p = self.acquisition_by_type.get(sensor_type)
        return p if p is not None else self.acquisition


# =====================================================================
# Auto-layout — genera el mapa estándar desde la plantilla
# =====================================================================
def auto_layout(machine: MachineConfig, with_keyphasor: bool = True) -> List[ChannelRow]:
    """Genera pares de proximidad X/Y por cojinete + keyphasor, estilo
    layout estándar Bently. Convención: Y=0° (TDC), X=90° (lado R).

    El usuario luego ajusta en el grid. Esto mata la densidad del wizard:
    un click y tenés el layout base.
    """
    rows: List[ChannelRow] = []
    bnc = 1
    n = max(1, int(machine.n_bearings))
    for brg in range(1, n + 1):
        # Convención Bently Nevada (API 670): ángulo desde TDC (arriba),
        # Y a 45° izquierda, X a 45° derecha → 90° entre sí.
        rows.append(ChannelRow(bnc_port=bnc, point_label=f"{brg}Y", plane=brg,
                               sensor_type="proximity", sensitivity_mv_per_eu=200.0,
                               unit_native="mil pp", coupling="AC",
                               angle_deg=45.0, side="L", alarm=2.5, danger=4.0,
                               full_scale=10.0, gap_bias_v=-9.5, active=True))
        bnc += 1
        rows.append(ChannelRow(bnc_port=bnc, point_label=f"{brg}X", plane=brg,
                               sensor_type="proximity", sensitivity_mv_per_eu=200.0,
                               unit_native="mil pp", coupling="AC",
                               angle_deg=45.0, side="R", alarm=2.5, danger=4.0,
                               full_scale=10.0, gap_bias_v=-9.5, active=True))
        bnc += 1
    if with_keyphasor:
        rows.append(ChannelRow(bnc_port=bnc, point_label="KPH", plane=0,
                               sensor_type="keyphasor", sensitivity_mv_per_eu=1.0,
                               unit_native="pulses/rev", coupling="DC",
                               angle_deg=0.0, side="", alarm=0.0, danger=0.0,
                               active=True, events_per_rev=1, trigger_v=-7.0))
    return rows


def absolute_angle(angle_deg: float, side: str) -> float:
    """Ángulo ABSOLUTO de la sonda medido desde TDC (arriba), sentido horario.

    Convención Bently Nevada / API 670: el ángulo se cuenta desde arriba
    (TDC = 0°) hacia la Derecha (horario) o Izquierda (antihorario).
      · 45° R  → 45°   (1:30 h)
      · 45° L  → 315°  (10:30 h)  → separación 90° respecto a 45° R ✓
      · sin lado → se interpreta como horario desde TDC (igual que R).
    """
    s = (side or "").strip().upper()
    a = float(angle_deg) % 360.0
    if s == "L":
        return (-float(angle_deg)) % 360.0
    return a


def angular_separation(a_abs: float, b_abs: float) -> float:
    """Separación no orientada entre dos ángulos absolutos (0..180)."""
    d = abs((a_abs - b_abs) % 360.0)
    return min(d, 360.0 - d)


def defaults_for_type(sensor_type: str) -> Dict[str, Any]:
    """Devuelve unidad/coupling/sensib por defecto para un tipo de sensor.
    Útil para que el grid auto-complete al cambiar el tipo."""
    return {
        "unit_native": _DEFAULT_UNIT.get(sensor_type, ""),
        "coupling": _DEFAULT_COUPLING.get(sensor_type, "AC"),
        "sensitivity_mv_per_eu": _DEFAULT_SENS.get(sensor_type, 100.0),
    }


# =====================================================================
# Validación API 670 / ISO 20816
# =====================================================================
@dataclass
class Finding:
    level: str   # "error" | "warn" | "ok"
    code: str
    message: str


def validate_setup(setup: AcqSetup) -> List[Finding]:
    """Valida el setup contra criterios API 670 / ISO. Devuelve findings
    ordenados error → warn → ok."""
    out: List[Finding] = []
    chans = setup.channels

    if not chans:
        return [Finding("error", "no_channels", "No hay canales configurados.")]

    # BNC único
    seen: Dict[int, str] = {}
    for ch in chans:
        if ch.bnc_port in seen:
            out.append(Finding("error", "dup_bnc",
                               f"BNC {ch.bnc_port} duplicado ({seen[ch.bnc_port]} y {ch.point_label})."))
        else:
            seen[ch.bnc_port] = ch.point_label

    # Keyphasor
    kph = [c for c in chans if c.is_keyphasor()]
    if not kph:
        out.append(Finding("warn", "no_keyphasor",
                           "Sin keyphasor: no habrá gráficos síncronos (bode/polar/órbita/1X)."))
    elif len(kph) > 1:
        out.append(Finding("warn", "multi_keyphasor",
                           f"{len(kph)} canales keyphasor — normalmente basta 1 por tren."))

    # Pares X/Y ortogonales por cojinete (API 670: sondas radiales a 90°)
    by_plane: Dict[int, List[ChannelRow]] = {}
    for c in chans:
        if c.is_keyphasor():
            continue
        by_plane.setdefault(c.plane, []).append(c)
    for plane, group in by_plane.items():
        radials = [c for c in group if c.sensor_type in ("proximity", "velometer", "accelerometer")]
        if len(radials) >= 2:
            # Ángulo ABSOLUTO (considerando lado L/R) — convención Bently.
            abs0 = absolute_angle(radials[0].angle_deg, radials[0].side)
            abs1 = absolute_angle(radials[1].angle_deg, radials[1].side)
            sep = angular_separation(abs0, abs1)
            if abs(sep - 90.0) > 5.0:
                d0 = f"{radials[0].angle_deg:.0f}°{radials[0].side}".strip()
                d1 = f"{radials[1].angle_deg:.0f}°{radials[1].side}".strip()
                out.append(Finding("warn", "xy_not_orthogonal",
                                   f"Cojinete {plane}: sondas a {d0}/{d1} "
                                   f"(separación {sep:.0f}°, se esperan 90°±5° para órbita)."))

    # Alert < Danger; sensibilidad en rango físico; unidades por tipo
    for ch in chans:
        if ch.is_keyphasor():
            continue
        if ch.danger > 0 and ch.alarm > 0 and ch.alarm >= ch.danger:
            out.append(Finding("error", "alert_ge_danger",
                               f"{ch.point_label}: Alert ({ch.alarm}) ≥ Danger ({ch.danger})."))
        lo, hi = _SENS_RANGE.get(ch.sensor_type, (0.0, 1e9))
        if not (lo <= ch.sensitivity_mv_per_eu <= hi):
            out.append(Finding("warn", "sens_out_of_range",
                               f"{ch.point_label}: sensib {ch.sensitivity_mv_per_eu} mV/EU fuera "
                               f"del rango típico [{lo:.0f}–{hi:.0f}] para {ch.sensor_type}."))
        valid_u = valid_units_for(ch.sensor_type)
        if valid_u and ch.unit_native and ch.unit_native not in valid_u:
            out.append(Finding("warn", "unit_mismatch",
                               f"{ch.point_label}: unidad '{ch.unit_native}' no es típica de "
                               f"{ch.sensor_type} ({' / '.join(valid_u)})."))
        # Gap/bias de sondas de proximidad (Bently -24V: típico -2 a -18 VDC)
        if ch.sensor_type == "proximity" and ch.gap_bias_v != 0.0:
            if not (-18.0 <= ch.gap_bias_v <= -2.0):
                out.append(Finding("warn", "gap_out_of_range",
                                   f"{ch.point_label}: gap {ch.gap_bias_v} V fuera del rango "
                                   f"típico [-18, -2] VDC de un proximitor."))
        # Danger no debe exceder el full-scale del transductor
        if ch.full_scale > 0 and ch.danger > 0 and ch.danger > ch.full_scale:
            out.append(Finding("warn", "danger_over_fullscale",
                               f"{ch.point_label}: Danger ({ch.danger}) supera el "
                               f"full-scale ({ch.full_scale} {ch.unit_native})."))

    # Parámetros de adquisición globales
    acq = setup.acquisition
    if acq.fmin_hz >= acq.fmax_hz:
        out.append(Finding("error", "acq_freq_range",
                           f"Fmin ({acq.fmin_hz} Hz) debe ser menor que Fmax ({acq.fmax_hz} Hz)."))
    kph_rows = [c for c in chans if c.is_keyphasor()]
    for k in kph_rows:
        if k.events_per_rev < 1:
            out.append(Finding("warn", "kph_events",
                               f"{k.point_label}: eventos/rev debe ser ≥ 1."))

    errors = [f for f in out if f.level == "error"]
    warns = [f for f in out if f.level == "warn"]
    if not errors and not warns:
        out.append(Finding("ok", "all_good", "Configuración válida — sin observaciones."))
    return sorted(out, key=lambda f: {"error": 0, "warn": 1, "ok": 2}[f.level])


def is_setup_valid(setup: AcqSetup) -> bool:
    """True si no hay findings de nivel error."""
    return not any(f.level == "error" for f in validate_setup(setup))


# =====================================================================
# Puentes a los modelos existentes (fuente única de verdad)
# =====================================================================
def orbit_pairs(channels: List[ChannelRow]) -> List[Tuple[str, str]]:
    """Pares X/Y explícitos para órbita, desde pair_ref (bidireccional, sin
    duplicar). Ej: si 1XD.pair_ref='1YD' → par ('1XD','1YD')."""
    by_name = {c.point_label: c for c in channels}
    seen: set = set()
    pairs: List[Tuple[str, str]] = []
    for c in channels:
        if c.is_keyphasor():
            continue
        partner = (c.pair_ref or "").strip()
        if partner and partner != c.point_label and partner in by_name:
            key = tuple(sorted([c.point_label, partner]))
            if key not in seen:
                seen.add(key)
                pairs.append((c.point_label, partner))
    return pairs


def setup_to_channel_configs(setup: AcqSetup) -> List:
    """AcqSetup → List[ChannelConfig] de core.modal.acq_backend (adquisición)."""
    from core.modal.acq_backend import ChannelConfig
    out = []
    for ch in setup.channels:
        coup = ch.coupling if ch.coupling in COUPLINGS else "AC"
        out.append(ChannelConfig(
            name=ch.point_label,
            coupling=coup,
            sensitivity_mv_per_eu=float(ch.sensitivity_mv_per_eu or 1.0),
            bnc_port=int(ch.bnc_port),
            units=ch.unit_native or "",
        ))
    return out


def setup_to_sensor_map(setup: AcqSetup) -> List[Dict[str, Any]]:
    """AcqSetup → lista de dicts compatibles con core.sensor_map.new_sensor().
    Este es el puente para persistir al modelo del activo (fuente única)."""
    from core.sensor_map import new_sensor
    sensors = []
    for ch in setup.channels:
        # Inferir direction (Y/X/RAD/AX/keyphasor) del label del punto
        lbl = (ch.point_label or "").upper()
        if ch.is_keyphasor():
            direction = "keyphasor"
        elif lbl.endswith("Y") or "Y" in lbl[-2:]:
            direction = "Y"
        elif lbl.endswith("X") or "X" in lbl[-2:]:
            direction = "X"
        elif lbl.endswith("A") or "AX" in lbl:
            direction = "AX"
        else:
            direction = "RAD"
        coup = ch.coupling if ch.coupling in COUPLINGS else ""
        sensors.append(new_sensor(
            plane=int(ch.plane),
            plane_label=ch.point_label,
            side=ch.side or "L",
            angle_deg=float(ch.angle_deg),
            direction=direction,
            sensor_type=ch.sensor_type,
            unit_native=ch.unit_native,
            coupling=coup,
            sensitivity_mv_per_eu=float(ch.sensitivity_mv_per_eu or 0.0),
            alarm=float(ch.alarm or 0.0),
            danger=float(ch.danger or 0.0),
        ))
    return sensors


# =====================================================================
# Persistencia local (JSON, durable-dir aware) — Fase 1
# =====================================================================
def _setups_dir() -> Path:
    pd = os.environ.get("WM_PERSIST_DIR")
    if pd:
        return Path(pd) / "remote_monitoring" / "setups"
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent / "data" / "remote_monitoring" / "setups"
    return Path("data/remote_monitoring/setups")


def _slug(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in (name or "setup")).strip("_") or "setup"


def save_setup(setup: AcqSetup) -> Path:
    d = _setups_dir()
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{_slug(setup.machine.name)}.json"
    payload = {"machine": asdict(setup.machine),
               "channels": [asdict(c) for c in setup.channels],
               "acquisition": asdict(setup.acquisition),
               "acquisition_by_type": {t: asdict(p) for t, p in setup.acquisition_by_type.items()}}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _filter_fields(cls, data: dict) -> dict:
    """Filtra un dict a los campos válidos del dataclass (tolerante a JSON viejo
    con campos faltantes o extra)."""
    valid = {f.name for f in fields(cls)}
    return {k: v for k, v in (data or {}).items() if k in valid}


def load_setup(name: str) -> Optional[AcqSetup]:
    path = _setups_dir() / f"{_slug(name)}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    machine = MachineConfig(**_filter_fields(MachineConfig, data.get("machine", {})))
    channels = [ChannelRow(**_filter_fields(ChannelRow, c)) for c in data.get("channels", [])]
    acq = AcquisitionParams(**_filter_fields(AcquisitionParams, data.get("acquisition", {})))
    # Por tipo: back-compat → si no está, defaults por tipo
    raw_bt = data.get("acquisition_by_type") or {}
    by_type = {t: default_acq_for_type(t) for t in SPECTRAL_TYPES}
    for t, p in raw_bt.items():
        by_type[t] = AcquisitionParams(**_filter_fields(AcquisitionParams, p))
    return AcqSetup(machine=machine, channels=channels, acquisition=acq,
                    acquisition_by_type=by_type)


def list_setups() -> List[str]:
    d = _setups_dir()
    if not d.exists():
        return []
    return sorted(p.stem for p in d.glob("*.json"))


def delete_setup(name: str) -> bool:
    """Borra un setup guardado. Devuelve True si borró algo."""
    p = _setups_dir() / f"{_slug(name)}.json"
    try:
        p.unlink()
        return True
    except Exception:  # noqa: BLE001
        return False


# =====================================================================
# Serialización + persistencia en la NUBE (Supabase) del AcqSetup
# ---------------------------------------------------------------------
# La máquina de Remote Monitoring / módulo de campo es un AcqSetup. Este es el
# "hogar en la nube" para que el CAMPO (nativo) y la WEB compartan la MISMA
# máquina: tabla `rm_setups` (id, name, metadata jsonb, updated_at). Reusa el
# cliente Supabase del recorder (credenciales embebidas en el .exe).
# =====================================================================
_RM_SETUPS_TABLE = os.environ.get("WM_RM_SETUPS_TABLE", "rm_setups")


def setup_to_dict(setup: "AcqSetup") -> Dict[str, Any]:
    """AcqSetup → dict serializable (mismo formato que el JSON local)."""
    return {"machine": asdict(setup.machine),
            "channels": [asdict(c) for c in setup.channels],
            "acquisition": asdict(setup.acquisition),
            "acquisition_by_type": {t: asdict(p) for t, p in setup.acquisition_by_type.items()}}


def setup_from_dict(data: Dict[str, Any]) -> "AcqSetup":
    """dict → AcqSetup (tolerante a campos faltantes / JSON viejo)."""
    machine = MachineConfig(**_filter_fields(MachineConfig, data.get("machine", {})))
    channels = [ChannelRow(**_filter_fields(ChannelRow, c)) for c in data.get("channels", [])]
    acq = AcquisitionParams(**_filter_fields(AcquisitionParams, data.get("acquisition", {})))
    by_type = {t: default_acq_for_type(t) for t in SPECTRAL_TYPES}
    for t, p in (data.get("acquisition_by_type") or {}).items():
        by_type[t] = AcquisitionParams(**_filter_fields(AcquisitionParams, p))
    return AcqSetup(machine=machine, channels=channels, acquisition=acq, acquisition_by_type=by_type)


def _rm_client():
    """Cliente Supabase (reusa el del recorder: env vars → _cloud_config embebido)."""
    try:
        from core.remote_monitoring.recorder import _sb_client
        return _sb_client()
    except Exception:  # noqa: BLE001
        return None


def save_setup_cloud(setup: "AcqSetup") -> Dict[str, Any]:
    """Sube (upsert) la máquina a la nube. Offline-first: si no hay cliente/internet
    devuelve {ok:False}. La misma máquina la ve la web (Remote Monitoring)."""
    client = _rm_client()
    if client is None:
        return {"ok": False, "reason": "offline"}
    try:
        from datetime import datetime
        name = setup.machine.name or "Machine"
        row = {"id": _slug(name), "name": name, "metadata": setup_to_dict(setup),
               "updated_at": datetime.now().isoformat(timespec="seconds")}
        try:
            client.table(_RM_SETUPS_TABLE).upsert(row).execute()
        except Exception:  # noqa: BLE001
            client.table(_RM_SETUPS_TABLE).insert(row).execute()
        return {"ok": True, "id": row["id"], "name": name}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "reason": f"{type(e).__name__}: {e}"}


def list_setups_cloud() -> List[Dict[str, Any]]:
    client = _rm_client()
    if client is None:
        return []
    try:
        res = client.table(_RM_SETUPS_TABLE).select("id, name, updated_at").execute()
        return sorted(res.data or [], key=lambda r: r.get("updated_at", ""), reverse=True)
    except Exception:  # noqa: BLE001
        return []


def load_setup_cloud(name_or_slug: str) -> Optional["AcqSetup"]:
    client = _rm_client()
    if client is None:
        return None
    try:
        res = (client.table(_RM_SETUPS_TABLE).select("metadata")
               .eq("id", _slug(name_or_slug)).single().execute())
        if res.data and res.data.get("metadata"):
            return setup_from_dict(res.data["metadata"])
    except Exception:  # noqa: BLE001
        return None
    return None
