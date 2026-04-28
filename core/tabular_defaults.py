"""
core.tabular_defaults
=====================

Helper centralizado para derivar los **defaults de Tabular List**
desde el ``Instance.header`` de la máquina activa en Machinery
Library. Reduce el setup manual del usuario a casi cero.

Lógica:

  - **Criterion** (ISO 7919-3 / ISO 20816-3 / ISO 20816-4 / etc.) se
    elige según ``support_type`` + ``nominal_power_mw`` del activo.
  - **Family** (Proximity / Velocity / Acceleration) se elige según
    ``support_type``: fluid_film → Proximity, rolling_element →
    Velocity.
  - **Overall mode** (RMS / PP / Peak) se elige por familia:
    Proximity → PP (peak-to-peak es el estándar API 670), Velocity y
    Acceleration → RMS.
  - **Alarm / Danger thresholds** vienen primero de ``alert_level`` /
    ``danger_level`` del activo si están definidos (setpoints reales
    del DCS del cliente). Si no, se calculan de las tablas ISO según
    iso_part + class + RPM nominal y se convierten a la unidad
    apropiada (mil pp para shaft, mm/s RMS para casing).

Ciclo 14b.2 — exporta:

  - :func:`derive_tabular_defaults` ``(instance) -> dict``
  - :data:`GENERIC_DEFAULTS` (cuando no hay instancia)
"""

from __future__ import annotations

from typing import Any, Dict, Optional

try:
    from core.instance_state import Instance
except Exception:
    Instance = None  # type: ignore


# Constante para conversión de µm pp a mil pp (1 mil = 25.4 µm)
_UM_PER_MIL = 25.4


# Defaults genéricos cuando no hay instancia activa
GENERIC_DEFAULTS: Dict[str, Any] = {
    "criterion": "ISO 20816-3",
    "criterion_explanation": "Default genérico ISO 20816-3 (sin instancia activa).",
    "family": "Auto",
    "overall_mode": "RMS",
    "alarm": 4.5,
    "danger": 7.1,
    "alarm_source": "default",
    "danger_source": "default",
    "iso_part_internal": "20816-3",
    "machine_class": "class_iii",
    "rpm": 3600.0,
    "unit_hint": "mm/s RMS",
}


def _criterion_from_supports(support_type: str, nominal_power_mw: float) -> tuple[str, str, str]:
    """
    Devuelve (iso_part_internal, criterion_label, measurement_type).

    Mapping:
      fluid_film  → ISO 7919-3 / ISO 20816-3, shaft_displacement
      rolling_element → ISO 20816-3 (general) o ISO 20816-4 (aero turbinas)
                        según contexto, casing_velocity
      magnetic    → API 670, shaft_displacement
      mixed       → ISO 7919-3 / ISO 20816-3 (default conservador)
    """
    sup = (support_type or "").lower()
    if sup == "fluid_film":
        return "20816-3", "ISO 7919-3 / ISO 20816-3", "shaft_displacement"
    if sup == "rolling_element":
        return "20816-3", "ISO 20816-3", "casing_velocity"
    if sup == "magnetic":
        return "20816-3", "API 670 + ISO 20816-3", "shaft_displacement"
    if sup == "mixed":
        return "20816-3", "ISO 7919-3 / ISO 20816-3", "shaft_displacement"
    # Default fallback
    return "20816-3", "ISO 20816-3", "casing_velocity"


def _machine_class_from_power(nominal_power_mw: float) -> str:
    """
    ISO 20816-3 class según potencia nominal:
      Class I:   < 15 kW          (motores pequeños)
      Class II:  15–300 kW        (medianas industriales)
      Class III: > 300 kW rigid foundation
      Class IV:  > 300 kW flexible foundation, large turbo-machinery
    """
    p = nominal_power_mw or 0.0
    if p < 0.015:
        return "class_i"
    if p < 0.300:
        return "class_ii"
    if p < 50.0:
        return "class_iii"
    return "class_iv"  # ≥ 50 MW: turbo-generadores grandes (Brush 54 MW va acá)


def _family_from_supports(support_type: str) -> str:
    sup = (support_type or "").lower()
    if sup in ("fluid_film", "magnetic", "mixed"):
        return "Proximity"
    if sup == "rolling_element":
        return "Velocity"
    return "Auto"


def _overall_mode_default(family: str) -> str:
    f = (family or "").lower()
    if f == "proximity":
        return "PP"
    if f == "acceleration":
        return "RMS"
    return "RMS"


def _read_iso_thresholds(
    iso_part_internal: str,
    machine_class: str,
    measurement_type: str,
    rpm: float,
) -> Optional[tuple[float, float, float]]:
    """
    Lee los thresholds (A/B, B/C, C/D) de las tablas ISO. Si la entrada
    exacta (class, rpm) no existe, busca el RPM más cercano dentro de
    las claves disponibles para esa class.
    Devuelve (ab, bc, cd) en la unidad nativa de la tabla (µm pp para
    shaft, mm/s RMS para casing) o None si no se pudo resolver.
    """
    try:
        from core.rotordynamics import ISO_PART_TABLES
    except Exception:
        return None

    part_tables = ISO_PART_TABLES.get(iso_part_internal)
    if part_tables is None:
        return None
    measurement_table = part_tables.get(measurement_type)
    if measurement_table is None:
        return None

    # Buscar todas las keys de la class solicitada
    candidates = [
        (rpm_key, vals)
        for ((cls, rpm_key), vals) in measurement_table.items()
        if cls == machine_class
    ]
    if not candidates:
        return None

    # Match por RPM más cercano
    best = min(candidates, key=lambda c: abs(c[0] - rpm) if c[0] > 0 else 0)
    return best[1]


def derive_tabular_defaults(inst: Optional[Any]) -> Dict[str, Any]:
    """
    Calcula los defaults de Tabular List desde la instancia activa.

    Args:
        inst: Instance del Machinery Library (o None). Si es None,
            devuelve GENERIC_DEFAULTS.

    Returns:
        Dict con todas las claves que Tabular List necesita para
        configurarse automáticamente: criterion, family, overall_mode,
        alarm, danger, alarm_source, danger_source, iso_part_internal,
        machine_class, rpm, unit_hint, criterion_explanation.
    """
    if inst is None:
        return dict(GENERIC_DEFAULTS)

    iso_part_internal, criterion_label, measurement_type = _criterion_from_supports(
        getattr(inst, "support_type", "") or "",
        float(getattr(inst, "nominal_power_mw", 0.0) or 0.0),
    )
    machine_class = _machine_class_from_power(
        float(getattr(inst, "nominal_power_mw", 0.0) or 0.0)
    )
    family = _family_from_supports(getattr(inst, "support_type", "") or "")
    overall_mode = _overall_mode_default(family)
    rpm = float(getattr(inst, "nominal_rpm", 0.0) or 0.0) or 3600.0

    # Setpoints — preferir los del activo si existen y son distintos de cero
    alert_level = float(getattr(inst, "alert_level", 0.0) or 0.0)
    danger_level = float(getattr(inst, "danger_level", 0.0) or 0.0)
    setpoint_unit = (getattr(inst, "setpoint_unit", "") or "").strip()

    if alert_level > 0 and danger_level > 0:
        alarm = alert_level
        danger = danger_level
        unit_hint = setpoint_unit or "(unidad sin especificar)"
        alarm_source = f"setpoints reales del activo ({unit_hint})"
        danger_source = alarm_source
    else:
        # Calcular de tablas ISO
        thresholds = _read_iso_thresholds(
            iso_part_internal, machine_class, measurement_type, rpm
        )
        if thresholds is not None:
            ab, bc, cd = thresholds
            # Convención: Alert = B/C (límite Aceptable→Marginal)
            #             Danger = C/D (límite Marginal→Inaceptable)
            if measurement_type == "shaft_displacement":
                # Tabla en µm pp → convertir a mil pp para coherencia con CSVs típicos API 670
                alarm = float(bc) / _UM_PER_MIL
                danger = float(cd) / _UM_PER_MIL
                unit_hint = "mil pp"
            else:  # casing_velocity
                alarm = float(bc)
                danger = float(cd)
                unit_hint = "mm/s RMS"
            class_label = machine_class.replace("class_", "Class ").upper().replace("CLASS ", "Class ")
            alarm_source = (
                f"Tabla {criterion_label} {class_label} @ {rpm:.0f} rpm "
                f"(zona B/C → Alert)"
            )
            danger_source = (
                f"Tabla {criterion_label} {class_label} @ {rpm:.0f} rpm "
                f"(zona C/D → Danger)"
            )
        else:
            # Fallback genérico
            alarm = GENERIC_DEFAULTS["alarm"]
            danger = GENERIC_DEFAULTS["danger"]
            alarm_source = "default (no se encontró tabla ISO matching)"
            danger_source = alarm_source
            unit_hint = ""

    explanation = (
        f"{criterion_label} aplicable a "
        f"{machine_class.replace('class_', 'Class ').upper().replace('CLASS ', 'Class ')} "
        f"({getattr(inst, 'support_type', '') or 'support type sin definir'}, "
        f"{float(getattr(inst, 'nominal_power_mw', 0.0) or 0.0):.0f} MW @ {rpm:.0f} rpm)."
    )

    return {
        "criterion": criterion_label,
        "criterion_explanation": explanation,
        "family": family,
        "overall_mode": overall_mode,
        "alarm": float(alarm),
        "danger": float(danger),
        "alarm_source": alarm_source,
        "danger_source": danger_source,
        "iso_part_internal": iso_part_internal,
        "machine_class": machine_class,
        "rpm": float(rpm),
        "unit_hint": unit_hint,
    }


__all__ = [
    "derive_tabular_defaults",
    "GENERIC_DEFAULTS",
]
