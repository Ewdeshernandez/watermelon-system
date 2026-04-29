"""
core.machine_severity
=====================

Helpers compartidos entre **Machine Map** (pagina 01b) y el **Mini
Machine Map** del Tabular List (Ciclo 15.1.1) para clasificar el
estado de cada sensor del Sensor Map activo contra los CSVs cargados
en sesion.

La logica vive aca para que Tabular y Machine Map produzcan el MISMO
estado por sensor — no queremos que el banner del Tabular diga
"verde" cuando el Machine Map dice "ámbar". Una sola fuente de
verdad para severidad.

Funciones publicas:

  - ``classify_severity(overall, alarm, danger)``: Normal / Alarm /
    Danger / No Data segun thresholds del sensor.
  - ``compute_signal_overall_rms(signal_obj)``: RMS robusto contra
    NaN / arrays vacios.
  - ``convert_rms_to_unit(rms, unit_native)``: convierte RMS al modo
    de display que indique unit_native ("X pp" → 2√2·RMS,
    "X peak" → √2·RMS, "X RMS" o nada → RMS).
  - ``build_severity_table(sensors, signals)``: dataframe con una
    fila por sensor del mapa, su signal matched, overall en unidad
    nativa y status. La columna ``Status`` toma los strings
    canonicos ``Normal`` / ``Alarm`` / ``Danger`` / ``No Data``.
  - ``count_status(df)``: dict con ``total`` / ``normal`` / ``alarm``
    / ``danger`` / ``no_data`` para los KPIs de la cabecera.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

import pandas as pd

from core.sensor_map import (
    resolve_sensor_for_point,
    sensor_label as sensor_label_fn,
    sensor_unit_family,
)


# ============================================================
# CLASIFICACION
# ============================================================

def classify_severity(overall: float, alarm: float, danger: float) -> str:
    """Clasifica una amplitud contra alarm/danger del sensor."""
    try:
        ov = float(overall)
        a = float(alarm or 0.0)
        d = float(danger or 0.0)
    except Exception:
        return "No Data"
    if d > 0 and ov >= d:
        return "Danger"
    if a > 0 and ov >= a:
        return "Alarm"
    return "Normal"


def _safe_float(v) -> float:
    try:
        return float(v)
    except Exception:
        return 0.0


# ============================================================
# CALCULO DE OVERALL POR SIGNAL
# ============================================================

def compute_signal_overall_rms(signal_obj: Any) -> float:
    """Calcula overall RMS robusto a NaN. Devuelve 0.0 si no se puede."""
    import numpy as np
    try:
        amp = signal_obj.amplitude if hasattr(signal_obj, "amplitude") else signal_obj.get("y")
        amp = np.asarray(amp, dtype=float)
        amp = amp[np.isfinite(amp)]
        if amp.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(amp ** 2)))
    except Exception:
        return 0.0


def convert_rms_to_unit(rms_value: float, unit_native: str) -> float:
    """
    Convierte RMS al modo de display indicado por unit_native:
      - "X pp" / "X p-p" / "X peak-to-peak" → RMS × 2√2
      - "X peak" / "X pk"                    → RMS × √2
      - "X RMS" / vacio                      → RMS
    """
    u = (unit_native or "").lower()
    if "pp" in u or "p-p" in u or "peak-to-peak" in u:
        return rms_value * 2.0 * math.sqrt(2.0)
    if "peak" in u or "pk" in u:
        return rms_value * math.sqrt(2.0)
    return rms_value


# ============================================================
# TABLA DE SEVERIDAD POR SENSOR DEL MAPA
# ============================================================

def build_severity_table(
    sensors: List[Dict[str, Any]],
    signals: Dict[str, Any],
) -> pd.DataFrame:
    """
    Para cada sensor del mapa, busca el signal cargado que matchea
    su csv_match_pattern, calcula overall en la unidad nativa del
    sensor, y clasifica severidad contra alarm/danger.

    Devuelve un DataFrame con columnas:
      Label, Plane, Plane Label, Type, Family, Unit,
      Alarm, Danger, Overall, Status, Source

    ``Status`` toma "Normal" / "Alarm" / "Danger" / "No Data".
    """
    rows: List[Dict[str, Any]] = []
    for s in sensors:
        lbl = sensor_label_fn(s)
        family = sensor_unit_family(s)
        unit_native = s.get("unit_native", "")
        alarm = _safe_float(s.get("alarm"))
        danger = _safe_float(s.get("danger"))

        matched_signal = None
        matched_source = ""
        for signame, sigobj in (signals or {}).items():
            try:
                metadata = (
                    getattr(sigobj, "metadata", None)
                    or (sigobj.get("metadata") if isinstance(sigobj, dict) else {})
                    or {}
                )
                point = str(metadata.get("Point", "") or "")
                variable = str(metadata.get("Variable", "") or "")
                csv_unit = str(
                    metadata.get("Y-Axis Unit", "")
                    or metadata.get("Unit", "")
                    or ""
                )
                m = resolve_sensor_for_point([s], point, variable, csv_unit)
                if m is not None:
                    matched_signal = sigobj
                    matched_source = signame
                    break
            except Exception:
                continue

        if matched_signal is not None:
            rms = compute_signal_overall_rms(matched_signal)
            overall_in_unit = convert_rms_to_unit(rms, unit_native)
            status = classify_severity(overall_in_unit, alarm, danger)
        else:
            overall_in_unit = 0.0
            status = "No Data"

        rows.append({
            "Label": lbl,
            "Plane": s.get("plane", 0),
            "Plane Label": s.get("plane_label", ""),
            "Type": s.get("sensor_type", ""),
            "Family": family,
            "Unit": unit_native,
            "Alarm": alarm,
            "Danger": danger,
            "Overall": overall_in_unit,
            "Status": status,
            "Source": matched_source,
        })

    return pd.DataFrame(rows)


def count_status(df: pd.DataFrame) -> Dict[str, int]:
    """Devuelve dict con total y desglose por status para los KPIs."""
    if df is None or df.empty:
        return {"total": 0, "normal": 0, "alarm": 0, "danger": 0, "no_data": 0}
    total = len(df)
    return {
        "total": total,
        "normal": int((df["Status"] == "Normal").sum()),
        "alarm": int((df["Status"] == "Alarm").sum()),
        "danger": int((df["Status"] == "Danger").sum()),
        "no_data": int((df["Status"] == "No Data").sum()),
    }


__all__ = [
    "classify_severity",
    "compute_signal_overall_rms",
    "convert_rms_to_unit",
    "build_severity_table",
    "count_status",
]
