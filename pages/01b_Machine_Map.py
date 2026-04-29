"""
pages/01b_Machine_Map.py
========================

Machine Map (Ciclo 15.1) — visualización del estado actual de cada
sensor del activo monitoreado, con heatmap de severidad por punto.

Toma:
  - El **Sensor Map** configurado en Machinery Library (lista de
    sensores con su ubicación física y thresholds individuales).
  - Las **señales cargadas** en Load Data.
  - El motor de matching ``resolve_sensor_for_point`` para asociar
    cada CSV a su sensor del mapa.

Y produce:
  - Vista lateral del tren con cojinetes numerados (driver→driven
    convención API 670).
  - Vista polar por plano con sondas en sus ángulos físicos
    coloreadas según severidad (verde / amarillo / rojo / gris).
  - Lista de sensores con atención requerida para drill-down.
  - Resumen estadístico (cuántos en cada zona).

El render se hace con el mismo helper ``render_sensor_map_diagram``
que usa Machinery Library, pero en modo ``severity_by_label``
(diferentes colores).
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from core.auth import require_login, render_user_menu

from core.instance_selector import render_instance_selector
from core.instance_state import (
    get_instance,
    get_instance_document_bytes,
    compose_train_description,
)
from core.sensor_map import (
    resolve_sensor_for_point,
    sensor_label as sensor_label_fn,
    sensor_unit_family,
)
from core.sensor_diagram import render_sensor_map_diagram
from core.ui_theme import apply_watermelon_page_style, page_header


st.set_page_config(page_title="Watermelon System | Machine Map", layout="wide")
require_login()
render_user_menu()
apply_watermelon_page_style()


# ============================================================
# HELPERS
# ============================================================

def _classify_severity(overall: float, alarm: float, danger: float) -> str:
    """Clasifica una amplitud según los thresholds del sensor."""
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


def _compute_signal_overall_rms(signal_obj: Any) -> float:
    """Calcula overall RMS de un signal (similar al Tabular List)."""
    import numpy as np
    try:
        amp = signal_obj.amplitude if hasattr(signal_obj, "amplitude") else signal_obj.get("y")
        amp = np.asarray(amp, dtype=float)
        amp = amp[np.isfinite(amp)]
        if amp.size == 0:
            return 0.0
        rms = float(np.sqrt(np.mean(amp ** 2)))
        return rms
    except Exception:
        return 0.0


def _convert_rms_to_unit(rms_value: float, unit_native: str) -> float:
    """
    Convierte RMS al modo de display que indique unit_native:
    - "X pp" / "X p-p" / "X peak-to-peak" → RMS × 2√2
    - "X peak" / "X pk" → RMS × √2
    - "X RMS" o nada → RMS directo
    """
    import math
    u = (unit_native or "").lower()
    if "pp" in u or "p-p" in u or "peak-to-peak" in u:
        return rms_value * 2.0 * math.sqrt(2.0)
    if "peak" in u or "pk" in u:
        return rms_value * math.sqrt(2.0)
    return rms_value


def _build_severity_table(
    sensors: List[Dict[str, Any]],
    signals: Dict[str, Any],
) -> pd.DataFrame:
    """
    Para cada sensor del mapa, busca el signal que matchea, calcula su
    overall en la unidad nativa del sensor, y clasifica severidad.
    Devuelve un DataFrame con: Label, Plane, Plane Label, Type, Unit,
    Alarm, Danger, Overall (matched), Status, Source signal.
    """
    rows = []
    for s in sensors:
        lbl = sensor_label_fn(s)
        family = sensor_unit_family(s)
        unit_native = s.get("unit_native", "")
        alarm = _safe_float(s.get("alarm"))
        danger = _safe_float(s.get("danger"))

        # Buscar el signal que matchea este sensor
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
                csv_unit = str(metadata.get("Y-Axis Unit", "") or metadata.get("Unit", "") or "")
                m = resolve_sensor_for_point([s], point, variable, csv_unit)
                if m is not None:
                    matched_signal = sigobj
                    matched_source = signame
                    break
            except Exception:
                continue

        if matched_signal is not None:
            rms = _compute_signal_overall_rms(matched_signal)
            overall_in_unit = _convert_rms_to_unit(rms, unit_native)
            status = _classify_severity(overall_in_unit, alarm, danger)
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


# ============================================================
# RENDER
# ============================================================

page_header(
    title="Machine Map",
    subtitle=(
        "Mapa visual de severidad por sensor sobre el tren acoplado. "
        "Cada sonda se colorea según el estado actual de su medición "
        "contra los setpoints individuales del Sensor Map."
    ),
)

with st.sidebar:
    st.markdown("---")
    _instance_state = render_instance_selector(module_name="machine_map")

_active_id = _instance_state.get("instance_id") or ""
_active_instance = get_instance(_active_id) if _active_id else None

if _active_instance is None:
    st.error(
        "🚨 **No hay máquina activa.** Andá a Machinery Library, "
        "activá un activo y volvé acá."
    )
    st.stop()

if not _active_instance.sensors:
    st.warning(
        "El activo activo no tiene **Sensor Map** configurado. "
        "Andá a Machinery Library → sección 'Mapa de Sensores' → "
        "'Generar mapa estándar' para configurarlo."
    )
    st.stop()

# Banner con info del activo
with st.container(border=True):
    bcols = st.columns([1.0, 3.5])
    with bcols[0]:
        if _active_instance.schematic_png:
            try:
                _png = get_instance_document_bytes(
                    _active_instance.instance_id, _active_instance.schematic_png
                )
                if _png:
                    st.image(_png, use_container_width=True)
            except Exception:
                pass
    with bcols[1]:
        _tag = _active_instance.tag or _active_instance.instance_id
        st.markdown(f"### 🟢 Machine Map · **{_tag}**")
        st.caption(compose_train_description(_active_instance) or "(sin descripción)")
        st.caption(
            f"Sensor Map: **{len(_active_instance.sensors)} sensores configurados** · "
            f"Cliente: {_active_instance.client or '—'} · "
            f"Sitio: {_active_instance.site or _active_instance.location or '—'}"
        )

# Construir tabla de severidad usando los signals cargados
signals = st.session_state.get("signals", {}) or {}

if not signals:
    st.info(
        "ℹ️ No hay señales cargadas en sesión. Andá a **Load Data** "
        "para cargar CSVs y volvé al Machine Map. Los marcadores van a "
        "aparecer en gris (Sin datos) hasta entonces."
    )

df_severity = _build_severity_table(_active_instance.sensors, signals)

# Resumen estadístico
total = len(df_severity)
n_normal = int((df_severity["Status"] == "Normal").sum())
n_alarm = int((df_severity["Status"] == "Alarm").sum())
n_danger = int((df_severity["Status"] == "Danger").sum())
n_nodata = int((df_severity["Status"] == "No Data").sum())

cols_summary = st.columns(4)
cols_summary[0].metric("✅ CONDICIÓN ACEPTABLE", f"{n_normal}", f"de {total}")
cols_summary[1].metric("⚠️ ATENCIÓN", f"{n_alarm}", f"de {total}")
cols_summary[2].metric("🚨 ACCIÓN REQUERIDA", f"{n_danger}", f"de {total}")
cols_summary[3].metric("◌ Sin datos", f"{n_nodata}", f"de {total}")

# Diagrama con heatmap
st.markdown("### 🎯 Heatmap de severidad por plano")
severity_by_label: Dict[str, str] = dict(
    zip(df_severity["Label"].astype(str), df_severity["Status"].astype(str))
)

try:
    _drv_lbl = " ".join(p for p in [
        _active_instance.driver_manufacturer, _active_instance.driver_model
    ] if p) or "Driver"
    _dvn_lbl = " ".join(p for p in [
        _active_instance.driven_manufacturer, _active_instance.driven_model
    ] if p) or "Driven"
    _diag_png = render_sensor_map_diagram(
        _active_instance.sensors,
        train_label=compose_train_description(_active_instance) or "",
        driver_label=_drv_lbl,
        driven_label=_dvn_lbl,
        severity_by_label=severity_by_label,
    )
    if _diag_png:
        st.image(_diag_png, use_container_width=True)
    else:
        st.warning("No se pudo renderizar el diagrama.")
except Exception as e:
    st.warning(f"Error al renderizar diagrama: {e}")

# Drill-down: sensores con atención requerida
critical_df = df_severity[df_severity["Status"].isin(["Alarm", "Danger"])].copy()
if not critical_df.empty:
    st.markdown("### 🚨 Sensores con atención requerida")
    critical_display = critical_df[[
        "Label", "Plane Label", "Type", "Overall", "Alarm", "Danger",
        "Unit", "Status", "Source",
    ]].sort_values(by=["Status", "Overall"], ascending=[True, False])
    # Formatear números
    critical_display["Overall"] = critical_display["Overall"].map(lambda x: f"{x:.3f}")
    critical_display["Alarm"] = critical_display["Alarm"].map(lambda x: f"{x:.3f}")
    critical_display["Danger"] = critical_display["Danger"].map(lambda x: f"{x:.3f}")
    st.dataframe(critical_display, use_container_width=True, hide_index=True)
else:
    if total > 0 and n_nodata < total:
        st.success(
            "✅ Todos los sensores con datos están en zona aceptable "
            "(por debajo de Alarm)."
        )

# Tabla completa colapsada
with st.expander(f"Tabla completa de sensores ({total} configurados)", expanded=False):
    full_display = df_severity[[
        "Label", "Plane", "Plane Label", "Type", "Family", "Unit",
        "Alarm", "Danger", "Overall", "Status", "Source",
    ]].copy()
    full_display["Overall"] = full_display["Overall"].map(
        lambda x: f"{x:.3f}" if isinstance(x, (int, float)) and x != 0 else "—"
    )
    full_display["Alarm"] = full_display["Alarm"].map(lambda x: f"{x:.3f}")
    full_display["Danger"] = full_display["Danger"].map(lambda x: f"{x:.3f}")
    st.dataframe(full_display, use_container_width=True, hide_index=True)
