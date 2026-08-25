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

from core.auth import require_login, render_user_menu, require_role

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
# Ciclo 17.16 — Machine Map es para staff
require_role(allowed_roles=("admin", "specialist"))
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
        "Visual severity map per sensor across the coupled train. "
        "Each probe is colored according to the current state of its measurement "
        "against the individual setpoints of the Sensor Map."
    ),
)

with st.sidebar:
    st.markdown("---")
    _instance_state = render_instance_selector(module_name="machine_map")

_active_id = _instance_state.get("instance_id") or ""
_active_instance = get_instance(_active_id) if _active_id else None

if _active_instance is None:
    st.error(
        "🚨 **No active machine.** Go to Machinery Library, "
        "activate an asset and come back here."
    )
    st.stop()

if not _active_instance.sensors:
    st.warning(
        "The active asset has no **Sensor Map** configured. "
        "Go to Machinery Library → 'Sensor Map' section → "
        "'Generate standard map' to configure it."
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
        st.caption(compose_train_description(_active_instance) or "(no description)")
        st.caption(
            f"Sensor Map: **{len(_active_instance.sensors)} sensors configured** · "
            f"Client: {_active_instance.client or '—'} · "
            f"Site: {_active_instance.site or _active_instance.location or '—'}"
        )

# Construir tabla de severidad usando los signals cargados
signals = st.session_state.get("signals", {}) or {}

if not signals:
    st.info(
        "ℹ️ No signals loaded in session. Go to **Load Data** "
        "to load CSVs and come back to the Machine Map. Markers will "
        "appear gray (No data) until then."
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
cols_summary[0].metric("✅ ACCEPTABLE CONDITION", f"{n_normal}", f"of {total}")
cols_summary[1].metric("⚠️ ATTENTION", f"{n_alarm}", f"of {total}")
cols_summary[2].metric("🚨 ACTION REQUIRED", f"{n_danger}", f"of {total}")
cols_summary[3].metric("◌ No data", f"{n_nodata}", f"of {total}")

# Diagrama con heatmap
st.markdown("### 🎯 Severity heatmap by plane")
severity_by_label: Dict[str, str] = dict(
    zip(df_severity["Label"].astype(str), df_severity["Status"].astype(str))
)
overall_by_label: Dict[str, float] = {}
unit_by_label: Dict[str, str] = {}
for _, _row in df_severity.iterrows():
    try:
        _lbl = str(_row["Label"])
        overall_by_label[_lbl] = float(_row.get("Overall") or 0.0)
        unit_by_label[_lbl] = str(_row.get("Unit") or "")
    except Exception:
        pass

# Ciclo 15.2 — preferimos render sobre la foto/dibujo real del activo si
# el usuario configuro x_pct/y_pct via click-to-place. Si no, caemos al
# render generico turbomachinery silhouette.
_used_real_schematic = False
_diag_png = None
if _active_instance.schematic_png:
    try:
        from core.sensor_diagram import render_on_schematic
        _sch_bytes_mm = get_instance_document_bytes(
            _active_instance.instance_id, _active_instance.schematic_png
        )
        if _sch_bytes_mm:
            _diag_png_real = render_on_schematic(
                _sch_bytes_mm, _active_instance.sensors,
                severity_by_label=severity_by_label,
                overall_by_label=overall_by_label,
                unit_by_label=unit_by_label,
            )
            if _diag_png_real:
                _diag_png = _diag_png_real
                _used_real_schematic = True
    except Exception:
        pass

if _diag_png is None:
    try:
        _drv_lbl = " ".join(p for p in [
            _active_instance.driver_manufacturer, _active_instance.driver_model
        ] if p) or "Driver"
        _dvn_lbl = " ".join(p for p in [
            _active_instance.driven_manufacturer, _active_instance.driven_model
        ] if p) or "Driven"
        # Ciclo 17.5.11 — kind adaptativo (motor/recip/turbine/etc.)
        try:
            from core.sensor_diagram import _infer_machine_kind as _ifk_mm
            _mm_drv_kind = _ifk_mm(_drv_lbl) or _ifk_mm(getattr(_active_instance, "asset_class", "")) or "turbine"
            _mm_dvn_kind = _ifk_mm(_dvn_lbl) or _ifk_mm(getattr(_active_instance, "asset_class", "")) or "generator"
        except Exception:
            _mm_drv_kind = "turbine"
            _mm_dvn_kind = "generator"
        _diag_png = render_sensor_map_diagram(
            _active_instance.sensors,
            train_label=compose_train_description(_active_instance) or "",
            driver_label=_drv_lbl,
            driven_label=_dvn_lbl,
            severity_by_label=severity_by_label,
            overall_by_label=overall_by_label,
            unit_by_label=unit_by_label,
            driver_kind=_mm_drv_kind,
            driven_kind=_mm_dvn_kind,
        )
    except Exception as e:
        st.warning(f"Error rendering diagram: {e}")

if _diag_png:
    st.image(_diag_png, use_container_width=True)
    if _used_real_schematic:
        st.caption(
            "Heatmap rendered over the asset's real schematic "
            "(coordinates configured in Machinery Library → Place "
            "sensors on the schematic)."
        )
else:
    st.warning("Could not render the diagram.")

# Drill-down: sensores con atención requerida
critical_df = df_severity[df_severity["Status"].isin(["Alarm", "Danger"])].copy()
if not critical_df.empty:
    st.markdown("### 🚨 Sensors requiring attention")
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
            "✅ All sensors with data are in the acceptable zone "
            "(below Alarm)."
        )

# Tabla completa colapsada
with st.expander(f"Full sensor table ({total} configured)", expanded=False):
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


# ============================================================
# Ciclo 15.1.6 — Diagnóstico self-service para sensores Sin Datos
# ------------------------------------------------------------
# Cuando un sensor del Sensor Map sale en "Sin datos" puede ser
# porque (a) el CSV no se cargó, (b) el csv_match_pattern del
# sensor no matchea el Point name del CSV cargado, o (c) el tipo
# de sensor / unit_native no coincide con el tipo del CSV
# (cross-type guard del matcher).
#
# Esta sección lista, para cada sensor sin datos:
#   - Su patrón esperado.
#   - Los Point names actualmente cargados en sesión.
#   - Una sugerencia de pattern que SI matchearia.
# ============================================================
nodata_df = df_severity[df_severity["Status"] == "No Data"].copy()
if not nodata_df.empty and signals:
    with st.expander(
        f"🔍 Diagnostics — why {len(nodata_df)} "
        f"{'sensor shows' if len(nodata_df) == 1 else 'sensors show'} "
        f"no data",
        expanded=False,
    ):
        # Recolectar Point names + variables + units de los signals en sesion
        point_inventory = []
        for signame, sigobj in (signals or {}).items():
            try:
                metadata = (
                    getattr(sigobj, "metadata", None)
                    or (sigobj.get("metadata") if isinstance(sigobj, dict) else {})
                    or {}
                )
                point_inventory.append({
                    "File": signame,
                    "Point": str(metadata.get("Point", "") or ""),
                    "Variable": str(metadata.get("Variable", "") or ""),
                    "Y-Axis Unit": str(
                        metadata.get("Y-Axis Unit", "")
                        or metadata.get("Unit", "")
                        or ""
                    ),
                })
            except Exception:
                continue

        if not point_inventory:
            st.info(
                "No signals with available metadata in session to "
                "diagnose. Reload the CSVs in Load Data."
            )
        else:
            st.caption(
                "The matcher pairs CSVs with Sensor Map sensors "
                "using the sensor's `csv_match_pattern` against the "
                "CSV Point name. The CSV unit must also be "
                "compatible with the sensor's family."
            )

            inv_df = pd.DataFrame(point_inventory)
            st.markdown("**Signals loaded in session:**")
            st.dataframe(inv_df, use_container_width=True, hide_index=True)

            st.markdown("**Sensors with no data and expected patterns:**")

            # Para cada sensor sin datos, mostrar patron + intento manual
            from core.sensor_map import resolve_sensor_for_point as _diag_resolve
            diag_rows = []
            for _, sr in nodata_df.iterrows():
                # Buscar el sensor original en _active_instance.sensors
                sensor_obj = None
                for _s in _active_instance.sensors:
                    try:
                        from core.sensor_map import sensor_label as _slbl
                        if _slbl(_s) == sr["Label"]:
                            sensor_obj = _s
                            break
                    except Exception:
                        pass
                pattern = (sensor_obj.get("csv_match_pattern", "")
                           if sensor_obj else "")
                stype = (sensor_obj.get("sensor_type", "")
                         if sensor_obj else "")
                unit_native = (sensor_obj.get("unit_native", "")
                               if sensor_obj else "")

                # Probar el matcher contra cada signal
                hit_signal = ""
                for inv in point_inventory:
                    try:
                        if sensor_obj is not None:
                            m = _diag_resolve(
                                [sensor_obj],
                                inv["Point"], inv["Variable"], inv["Y-Axis Unit"],
                            )
                            if m is not None:
                                hit_signal = inv["File"]
                                break
                    except Exception:
                        continue

                diag_rows.append({
                    "Sensor": sr["Label"],
                    "Plane": sr["Plane Label"] or sr["Plane"],
                    "Type": stype,
                    "Expected unit": unit_native,
                    "csv_match_pattern": pattern or "(empty)",
                    "Any loaded signal matches?": (
                        f"✓ {hit_signal}" if hit_signal else "✗ none"
                    ),
                })

            st.dataframe(
                pd.DataFrame(diag_rows),
                use_container_width=True,
                hide_index=True,
            )

            st.info(
                "💡 **How to fix it:**\n\n"
                "• If your CSV has a Point name different from the expected one, "
                "edit the sensor's `csv_match_pattern` in "
                "**Machinery Library → Sensor Map** so it "
                "matches. Example: if the real Point is `VE5809`, "
                "you can use pattern `*5809*` or `VE58*`.\n\n"
                "• If your CSV is not loaded, go to **Load Data** "
                "and upload it.\n\n"
                "• If the sensor is a proximity probe but the CSV is "
                "in `g` or `in/s`, it is not that signal — you must load "
                "the CSV in `mil pp` or `µm pp` matching the sensor."
            )
