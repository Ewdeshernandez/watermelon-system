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
from core.machine_severity import (
    build_severity_table as _shared_build_severity_table,
    count_status as _shared_count_status,
)
from core.sensor_diagram import render_sensor_map_diagram
from core.ui_theme import apply_watermelon_page_style, page_header


st.set_page_config(page_title="Watermelon System | Machine Map", layout="wide")
require_login()
render_user_menu()
apply_watermelon_page_style()


# ============================================================
# HELPERS — desde Ciclo 15.1.1 viven en core.machine_severity
# para compartir con el Mini Machine Map del Tabular List.
# Aquí solo dejamos alias finos para no tocar el render más abajo.
# ============================================================

_build_severity_table = _shared_build_severity_table


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

# Resumen estadístico (mismo helper compartido que usa el Mini Map)
_counts = _shared_count_status(df_severity)
total = _counts["total"]
n_normal = _counts["normal"]
n_alarm = _counts["alarm"]
n_danger = _counts["danger"]
n_nodata = _counts["no_data"]

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
