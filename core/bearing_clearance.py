"""
Watermelon System — Bearing clearance calculator (Ciclo 23.72, scope re-aclarado 23.74).

Calcula el clearance boundary diametral y radial de un cojinete radial
(tilting pad o sleeve) usando rule-of-thumb API 616 / API 670 — ÚNICO
camino disponible cuando no se tiene datasheet del fabricante.

NOTA IMPORTANTE (Ciclo 23.74):
Para el módulo SCL principal (pages/09_Shaft_Centerline.py + core/scl_diagnostics.py)
la fuente preferida del clearance es `derive_radial_clearance_from_vault()`,
que prioriza datos OEM capturados del datasheet del fabricante (en
core/document_vault.py) sobre cualquier rule-of-thumb.

Este módulo (`bearing_clearance`) queda como:
  • Helper standalone para cálculos rápidos sin necesitar Streamlit
  • Pre-relleno en el wizard de creación de activos cuando NO hay datasheet
  • Soporte de preload explícito (tilting pad), no cubierto por la heurística
    actual del vault path

Output usado por:
  • Wizard de creación de activos — pre-relleno de Cb / Ca
  • Alarm preload — thresholds en displacement (mil pp, μm pp) a partir
    de % del assembled clearance (API 670 §6.10)
  • Reportes técnicos que necesiten justificar el cálculo rule-of-thumb

Fuentes / convenciones:
  • API 616 5th (gas turbines) y 617 8th (compressors) sugieren para
    tilting pad: diametral clearance machined 1.5 a 2.5 mils per inch
    of shaft diameter (~0.0015 a 0.0025 D), target 2.0.
  • Sleeve bearings: 1.0 a 1.5 mils per inch (~0.0010 a 0.0015 D).
  • Preload tilting pad (m = 1 − Ca/Cb): rango 0.0 a 0.5; típico 0.3 a 0.5.
    m = 0.5 → assembled clearance = 50% del machined.
  • Bently 3300XL proximity probe sensitivity: 200 mV/mil = 7.874 mV/μm.
  • Cold reference position en SCL: eje apoyado en el babbitt inferior
    por gravedad → (x=0, y=-Ca/2) en convención API 670.
  • Alarm preliminar (Alert) API 670 §6.10: 25-40 % of Ca.
  • Danger preliminar (Trip): 50-60 % of Ca.

Importante: estos cálculos son ORIENTATIVOS. Los valores reales
vienen del datasheet API-670 que acompaña al cojinete (lo emite el
fabricante: John Crane, Kingsbury, Waukesha, etc). El usuario debe
poder overridear con valor manual cuando tenga el dato real.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Literal, Dict, Any


# Conversión y constantes
MM_PER_INCH = 25.4
UM_PER_MIL = 25.4
BENTLY_3300XL_MV_PER_UM = 7.874  # 200 mV / mil → mV/μm
BENTLY_3300XL_MV_PER_MIL = 200.0

BearingType = Literal["tilting_pad", "sleeve", "tilting_pad_offset_60"]
Rule = Literal["min", "typical", "max"]


# Reglas de clearance en mils/inch de shaft diameter por tipo de cojinete
_RULE_MIL_PER_INCH: Dict[str, Dict[str, float]] = {
    "tilting_pad": {
        "min":     1.5,
        "typical": 2.0,
        "max":     2.5,
    },
    "tilting_pad_offset_60": {
        # Offset pivot (60% trailing) suele tener clearance similar
        # a centre pivot. Mantenemos la misma tabla. Si en datasheet
        # difiere, override manual.
        "min":     1.5,
        "typical": 2.0,
        "max":     2.5,
    },
    "sleeve": {
        "min":     1.0,
        "typical": 1.25,
        "max":     1.5,
    },
}


@dataclass(frozen=True)
class BearingClearance:
    """Resultado del cálculo de clearance para un cojinete radial."""

    # Inputs (echo back)
    shaft_dia_mm: float
    shaft_dia_in: float
    bearing_type: str
    preload: float
    rule_applied: str

    # Outputs principales (diametral)
    bore_clearance_um_pp: float        # Cb — machined diametral clearance
    bore_clearance_mil_pp: float
    assembled_clearance_um_pp: float   # Ca — after preload  (Cb × (1 − m))
    assembled_clearance_mil_pp: float

    # Outputs derivados
    radial_clearance_um: float         # Ca / 2 — for orbit / SCL
    cold_reference_y_um: float         # -Ca/2 (shaft en babbitt inferior)

    # Alarmas preliminares (% de Ca, displacement pk-pk)
    alarm_um_pp: float                 # 40 % Ca (Alert)
    alarm_mil_pp: float
    danger_um_pp: float                # 60 % Ca (Trip)
    danger_mil_pp: float

    # Bently 3300XL conversion helpers (constantes pero útiles tenerlas
    # acompañando para no duplicar lógica en composers)
    bently_sensitivity_mv_per_um: float = BENTLY_3300XL_MV_PER_UM

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_bearing_clearance(
    shaft_dia_mm: float,
    bearing_type: BearingType = "tilting_pad",
    preload: float = 0.4,
    rule: Rule = "typical",
) -> BearingClearance:
    """Calcula clearance boundary para un cojinete radial.

    Args:
        shaft_dia_mm: diámetro del eje en mm (ej. 100 para SGT-300 BRG#1)
        bearing_type: "tilting_pad", "sleeve", o "tilting_pad_offset_60"
        preload: m = 1 - Ca/Cb. 0.0 = sin preload, 0.5 = típico tilting pad
        rule: "min" / "typical" / "max" de la tabla rule-of-thumb

    Returns:
        BearingClearance dataclass con Cb, Ca, alarm/danger preliminares.

    Raises:
        ValueError: shaft_dia_mm ≤ 0, preload fuera de [0, 0.7] (físicamente
            imposible >0.7), bearing_type desconocido, rule desconocido.
    """
    if shaft_dia_mm <= 0:
        raise ValueError(f"shaft_dia_mm debe ser > 0 (recibido: {shaft_dia_mm})")
    if not 0.0 <= preload <= 0.7:
        raise ValueError(
            f"preload debe estar en [0, 0.7] (recibido: {preload}). "
            f"Tilting pad típico es 0.3-0.5; >0.5 es agresivo."
        )
    if bearing_type not in _RULE_MIL_PER_INCH:
        raise ValueError(
            f"bearing_type desconocido: {bearing_type!r}. "
            f"Opciones: {list(_RULE_MIL_PER_INCH.keys())}"
        )
    if rule not in ("min", "typical", "max"):
        raise ValueError(f"rule debe ser min/typical/max (recibido: {rule!r})")

    shaft_dia_in = shaft_dia_mm / MM_PER_INCH
    mil_per_in = _RULE_MIL_PER_INCH[bearing_type][rule]

    # Diametral clearance machined (Cb)
    bore_mil_pp = mil_per_in * shaft_dia_in
    bore_um_pp = bore_mil_pp * UM_PER_MIL

    # Assembled clearance Ca = Cb × (1 − m)
    assembled_um_pp = bore_um_pp * (1.0 - preload)
    assembled_mil_pp = bore_mil_pp * (1.0 - preload)

    # Alarmas (API 670 §6.10 — preliminar, ajustar con baseline real)
    ALARM_PCT = 0.40
    DANGER_PCT = 0.60
    alarm_um_pp = assembled_um_pp * ALARM_PCT
    danger_um_pp = assembled_um_pp * DANGER_PCT

    rule_label_map = {"min": "mínimo", "typical": "típico", "max": "máximo"}
    rule_applied = (
        f"API 616/670 {rule_label_map[rule]} — "
        f"{mil_per_in} mil/in × preload {preload}"
    )

    return BearingClearance(
        shaft_dia_mm=shaft_dia_mm,
        shaft_dia_in=round(shaft_dia_in, 4),
        bearing_type=bearing_type,
        preload=preload,
        rule_applied=rule_applied,
        bore_clearance_um_pp=round(bore_um_pp, 1),
        bore_clearance_mil_pp=round(bore_mil_pp, 3),
        assembled_clearance_um_pp=round(assembled_um_pp, 1),
        assembled_clearance_mil_pp=round(assembled_mil_pp, 3),
        radial_clearance_um=round(assembled_um_pp / 2.0, 1),
        cold_reference_y_um=round(-assembled_um_pp / 2.0, 1),
        alarm_um_pp=round(alarm_um_pp, 1),
        alarm_mil_pp=round(alarm_um_pp / UM_PER_MIL, 3),
        danger_um_pp=round(danger_um_pp, 1),
        danger_mil_pp=round(danger_um_pp / UM_PER_MIL, 3),
    )


def gap_voltage_to_position_um(
    delta_gap_v: float,
    sensitivity_mv_per_um: float = BENTLY_3300XL_MV_PER_UM,
) -> float:
    """Convierte un delta de gap voltage (V) a delta de posición (μm).

    Bently 3300XL: 200 mV/mil = 7.874 mV/μm. Polaridad negativa (más
    negativo = más cerca del probe). El caller pasa delta absoluto;
    el signo de "cerca/lejos" lo maneja el composer del plot.

    Args:
        delta_gap_v: diferencia de voltage en VOLTS (no mV).
            ej. -8.5 V cold ref vs -8.2 V running → delta_gap_v = +0.3
        sensitivity_mv_per_um: por defecto Bently 3300XL.

    Returns:
        delta_position_um (positivo = eje se alejó del probe; negativo
        = eje se acercó al probe).
    """
    if sensitivity_mv_per_um <= 0:
        raise ValueError("sensitivity_mv_per_um debe ser > 0")
    return (delta_gap_v * 1000.0) / sensitivity_mv_per_um


def manual_clearance(
    shaft_dia_mm: float,
    assembled_clearance_um_pp: float,
    bearing_type: str = "manual",
) -> BearingClearance:
    """Construye un BearingClearance a partir de un valor manual de Ca.

    Útil cuando el usuario tiene el datasheet API-670 del fabricante y
    quiere overridear el cálculo rule-of-thumb. El preload no aplica
    porque ya está implícito en el Ca manual.

    Args:
        shaft_dia_mm: diámetro del eje (sigue siendo necesario para el
            cold reference position).
        assembled_clearance_um_pp: Ca diametral en μm pp (del datasheet).
        bearing_type: etiqueta libre para el rule_applied.

    Returns:
        BearingClearance con Cb = Ca (sin preload asumido), alarmas
        calculadas igual sobre Ca.
    """
    if shaft_dia_mm <= 0:
        raise ValueError(f"shaft_dia_mm debe ser > 0")
    if assembled_clearance_um_pp <= 0:
        raise ValueError("assembled_clearance_um_pp debe ser > 0")

    ALARM_PCT = 0.40
    DANGER_PCT = 0.60
    alarm_um_pp = assembled_clearance_um_pp * ALARM_PCT
    danger_um_pp = assembled_clearance_um_pp * DANGER_PCT
    assembled_mil = assembled_clearance_um_pp / UM_PER_MIL

    return BearingClearance(
        shaft_dia_mm=shaft_dia_mm,
        shaft_dia_in=round(shaft_dia_mm / MM_PER_INCH, 4),
        bearing_type=bearing_type,
        preload=0.0,  # no aplica con override manual
        rule_applied="Manual (datasheet API-670 del fabricante)",
        bore_clearance_um_pp=round(assembled_clearance_um_pp, 1),
        bore_clearance_mil_pp=round(assembled_mil, 3),
        assembled_clearance_um_pp=round(assembled_clearance_um_pp, 1),
        assembled_clearance_mil_pp=round(assembled_mil, 3),
        radial_clearance_um=round(assembled_clearance_um_pp / 2.0, 1),
        cold_reference_y_um=round(-assembled_clearance_um_pp / 2.0, 1),
        alarm_um_pp=round(alarm_um_pp, 1),
        alarm_mil_pp=round(alarm_um_pp / UM_PER_MIL, 3),
        danger_um_pp=round(danger_um_pp, 1),
        danger_mil_pp=round(danger_um_pp / UM_PER_MIL, 3),
    )


if __name__ == "__main__":
    # Smoke test — Siemens SGT-300 datasheet (HI4088)
    print("=== Watermelon Bearing Clearance Calculator ===\n")
    for label, D, btype in [
        ("BRG #1 — INLET (Centre pivot)", 100.0, "tilting_pad"),
        ("BRG #2 — EXIT (Offset 60%)",    150.0, "tilting_pad_offset_60"),
    ]:
        c = compute_bearing_clearance(D, btype, preload=0.4, rule="typical")
        print(f"{label}")
        print(f"  Shaft Ø:    {c.shaft_dia_mm:.0f} mm ({c.shaft_dia_in:.3f} in)")
        print(f"  Cb (mach):  {c.bore_clearance_um_pp:.0f} μm pp  "
              f"({c.bore_clearance_mil_pp:.2f} mil pp)")
        print(f"  Ca (asbl):  {c.assembled_clearance_um_pp:.0f} μm pp  "
              f"({c.assembled_clearance_mil_pp:.2f} mil pp)")
        print(f"  Cold ref:   y = {c.cold_reference_y_um:.0f} μm")
        print(f"  Alarm:      {c.alarm_um_pp:.0f} μm pp  ({c.alarm_mil_pp:.2f} mil pp)")
        print(f"  Danger:     {c.danger_um_pp:.0f} μm pp  ({c.danger_mil_pp:.2f} mil pp)")
        print(f"  Rule:       {c.rule_applied}\n")
