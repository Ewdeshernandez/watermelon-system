"""
pages/_machinery_wizard.py
==========================

Wizard guiado para crear activos en Watermelon (Ciclo 21).

Inspirado en System1 / AMS Suite / @ptitude: el usuario va paso a
paso por una secuencia clara que lo guía desde "qué máquina es" hasta
"qué sensores tiene y en qué unidades reportan".

5 pasos:
  1. Tipo de máquina + plantilla LATAM (opcional)
  2. Tren mecánico (driver + driven, cojinetes, acople)
  3. Instrumentación (proximidad XY / acelerómetro / velocímetro)
  4. Unidades y setpoints por canal
  5. Datos del activo (ID, tag, cliente, sitio, RPM)

Convive con pages/00_Machinery_Library.py — el flujo legacy NO se
modifica. Cuando este wizard quede validado, se promueve como default
desde NAV_ITEMS sin tocar el archivo viejo.

Solo admin + specialist pueden crear activos nuevos.
"""

from __future__ import annotations

from core.auth import require_login, render_user_menu, require_role

require_login()
render_user_menu()
require_role(allowed_roles=("admin", "specialist"))

import re
from datetime import datetime
from typing import Any, Dict, List, Optional

import streamlit as st

from core.instance_state import (
    create_instance,
    get_instance,
    list_instances,
    update_instance_header,
)
from core.machine_profiles import PROFILES
from core.machine_templates import (
    get_template,
    list_categories,
    list_templates,
    list_templates_by_category,
    suggest_profile_key_for_template,
)
from core.sensor_map import generate_standard_sensor_map


st.set_page_config(
    page_title="Crear activo · Wizard — Watermelon",
    page_icon="🧙",
    layout="wide",
)


# =============================================================
# Estado del wizard en session_state
# =============================================================

WIZ_KEY = "wm_wizard_state_v3"
TOTAL_STEPS = 6


def _wizard_state() -> Dict[str, Any]:
    if WIZ_KEY not in st.session_state:
        st.session_state[WIZ_KEY] = _default_state()
    return st.session_state[WIZ_KEY]


def _default_state() -> Dict[str, Any]:
    return {
        "step": 1,
        # Paso 1
        "category": "",
        "template_id": "",
        "template_label": "",
        # Paso 2
        "driver_type": "",
        "driver_planes": 2,
        "driver_bearing_kind": "plain",  # plain / rolling
        "driver_bearing_model": "",
        # Ciclo 23.13 — iconografía 2D vectorial (asset library)
        "driver_icon_key": "",
        "driven_icon_key": "",
        "driven_type": "",
        "driven_planes": 2,
        "driven_bearing_kind": "rolling",
        "driven_bearing_model": "",
        "coupling_class": "flexible",
        # Gearbox intermedio (Ciclo 21.2)
        "include_gearbox": False,
        "gearbox_type": "",
        "gearbox_planes": 2,
        "gearbox_bearing_kind": "rolling",
        "gearbox_bearing_model": "",
        "gearbox_instrumentation": "axial_accel",
        # Reciprocantes (Ciclo 21.3)
        "cylinders_count": 4,        # 2 / 4 / 6 / 8
        "include_rod_drop": True,    # 1 sensor de rod drop por cilindro
        # Override de sensores (Ciclo 21.1) — se llena en el paso 4
        "sensors_override": None,
        # Paso 3
        "driver_instrumentation": "proximity_xy",
        "driven_instrumentation": "proximity_xy",
        "include_keyphasor": True,
        "channels_per_sensor": 1,
        # Paso 4
        "displacement_unit": "mil pp",
        "velocity_unit": "mm/s pk",
        "acceleration_unit": "g pk",
        "proximity_alarm_mil_pp": 4.0,
        "proximity_danger_mil_pp": 6.0,
        "accel_alarm_g": 4.5,
        "accel_danger_g": 9.0,
        "velocity_alarm_mm_s": 4.5,
        "velocity_danger_mm_s": 11.2,
        # Paso 5
        "instance_id": "",
        "tag": "",
        "client": "",
        "site": "",
        "location": "",
        "asset_class": "",
        "driver_manufacturer": "",
        "driver_model": "",
        "driven_manufacturer": "",
        "driven_model": "",
        "nominal_rpm": 0.0,
        "nominal_power_mw": 0.0,
        "iso_norm_code": "",
        "iso_norm_class": "",
        "notes": "",
        "profile_key": "custom_manual",
    }


def _reset_wizard():
    st.session_state[WIZ_KEY] = _default_state()


def _go_next():
    s = _wizard_state()
    s["step"] = min(s["step"] + 1, TOTAL_STEPS)


def _go_prev():
    s = _wizard_state()
    s["step"] = max(s["step"] - 1, 1)


# =============================================================
# Helpers (deben ir antes de los pasos que los usan)
# =============================================================

def _slug_default(state: Dict[str, Any]) -> str:
    seed = state.get("tag") or state.get("template_id") or state.get("driver_type") or ""
    seed = seed.lower()
    seed = re.sub(r"[^a-z0-9]+", "_", seed)
    return seed.strip("_")[:40]


def _build_full_sensor_map(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Construye el mapa completo de sensores. Soporta:
      - turbomáquina estándar (driver + driven con cojinetes radiales)
      - tren con gearbox intermedio (HSS + LSS)
      - compresor reciprocante (frame velocity + crosshead accel + rod drop)
    """
    from core.sensor_map import new_sensor

    is_recip = state.get("category") == "reciprocating_compressor"

    # ===== Caso reciprocante =====
    if is_recip:
        return _build_reciprocating_sensor_map(state)

    # ===== Caso estándar turbomáquina =====
    sensors = generate_standard_sensor_map(
        driver_planes=int(state.get("driver_planes", 2)),
        driver_instrumentation=state.get("driver_instrumentation", "proximity_xy"),
        driven_planes=int(state.get("driven_planes", 2)),
        driven_instrumentation=state.get("driven_instrumentation", "proximity_xy"),
        include_keyphasor=bool(state.get("include_keyphasor", True)),
        proximity_alarm_mil_pp=float(state.get("proximity_alarm_mil_pp", 4.0)),
        proximity_danger_mil_pp=float(state.get("proximity_danger_mil_pp", 6.0)),
        accel_alarm_g_rms=float(state.get("accel_alarm_g", 4.5)),
        accel_danger_g_rms=float(state.get("accel_danger_g", 9.0)),
        velocity_alarm_mm_s=float(state.get("velocity_alarm_mm_s", 4.5)),
        velocity_danger_mm_s=float(state.get("velocity_danger_mm_s", 11.2)),
    )

    # Si NO hay gearbox, devolver tal cual
    if not state.get("include_gearbox"):
        return sensors

    # Insertar sensores del gearbox entre driver y driven
    # Tomamos como base la cantidad de planos de driver para offset
    driver_planes = int(state.get("driver_planes", 2))
    gearbox_planes = int(state.get("gearbox_planes", 2))
    gearbox_instrum = state.get("gearbox_instrumentation", "axial_accel")

    gearbox_sensors: List[Dict[str, Any]] = []
    for i in range(gearbox_planes):
        plane_idx = driver_planes + i + 1
        if i == 0:
            label = "HSS gearbox"
        elif i == gearbox_planes - 1:
            label = "LSS gearbox"
        else:
            label = f"Intermedio {i} gearbox"

        if gearbox_instrum == "proximity_xy":
            # Par X-Y a 45°
            for direction, side in (("X", "R"), ("Y", "L")):
                gearbox_sensors.append(new_sensor(
                    plane=plane_idx, plane_label=label, side=side,
                    angle_deg=45.0, direction=direction,
                    sensor_type="proximity", unit_native="mil pp",
                    alarm=float(state.get("proximity_alarm_mil_pp", 4.0)),
                    danger=float(state.get("proximity_danger_mil_pp", 6.0)),
                    csv_match_pattern=f"*gear*{label.replace(' ', '_')}*",
                ))
        elif gearbox_instrum == "accel_plus_velocity":
            gearbox_sensors.append(new_sensor(
                plane=plane_idx, plane_label=label, side="T",
                angle_deg=0.0, direction="RAD",
                sensor_type="accelerometer", unit_native="g rms",
                alarm=float(state.get("accel_alarm_g", 4.5)),
                danger=float(state.get("accel_danger_g", 9.0)),
                csv_match_pattern=f"*gear*acc*{label.replace(' ', '_')}*",
            ))
            gearbox_sensors.append(new_sensor(
                plane=plane_idx, plane_label=label, side="T",
                angle_deg=0.0, direction="RAD",
                sensor_type="velometer", unit_native="mm/s rms",
                alarm=float(state.get("velocity_alarm_mm_s", 4.5)),
                danger=float(state.get("velocity_danger_mm_s", 11.2)),
                csv_match_pattern=f"*gear*vel*{label.replace(' ', '_')}*",
            ))
        else:  # axial_accel (default)
            gearbox_sensors.append(new_sensor(
                plane=plane_idx, plane_label=label, side="T",
                angle_deg=0.0, direction="RAD",
                sensor_type="accelerometer", unit_native="g rms",
                alarm=float(state.get("accel_alarm_g", 4.5)),
                danger=float(state.get("accel_danger_g", 9.0)),
                csv_match_pattern=f"*gear*{label.replace(' ', '_')}*",
            ))

    # Insertar gearbox después de driver (antes de driven y keyphasor)
    # Identificamos el primer índice de driven para insertar
    insert_idx = 0
    for idx, s in enumerate(sensors):
        plane_label = s.get("plane_label", "")
        if "driven" in plane_label.lower():
            insert_idx = idx
            break
    if insert_idx == 0:
        # Si no encontramos driven, insertamos antes del keyphasor o al final
        for idx, s in enumerate(sensors):
            if s.get("sensor_type") == "keyphasor":
                insert_idx = idx
                break
        else:
            insert_idx = len(sensors)

    # Re-numerar planes de los sensores driven/keyphasor después del gearbox
    plane_offset = gearbox_planes
    for s in sensors[insert_idx:]:
        s["plane"] = int(s.get("plane", 0)) + plane_offset

    sensors = sensors[:insert_idx] + gearbox_sensors + sensors[insert_idx:]
    return sensors


def _build_reciprocating_sensor_map(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Sensor map para compresor reciprocante (API 618 / ISO 20816-8) con
    coordenadas físicas (x_pct, y_pct) coherentes con el schematic
    generado en core.recip_schematic.
    """
    from core.recip_schematic import sensor_default_position
    from core.sensor_map import new_sensor

    n_cylinders_state = int(state.get("cylinders_count", 4))
    n_motor_planes_state = int(state.get("driver_planes", 2))

    def _with_position(sensor: Dict[str, Any]) -> Dict[str, Any]:
        x_pct, y_pct = sensor_default_position(
            sensor, n_cylinders=n_cylinders_state,
            n_motor_planes=n_motor_planes_state,
        )
        sensor["x_pct"] = float(x_pct)
        sensor["y_pct"] = float(y_pct)
        return sensor

    sensors: List[Dict[str, Any]] = []
    plane_idx = 1

    # 1. Driver (motor eléctrico) — usa su instrumentación normal
    driver_planes = int(state.get("driver_planes", 2))
    driver_instr = state.get("driver_instrumentation", "axial_accel")
    for i in range(driver_planes):
        side = "DE" if i == 0 else "NDE" if i == 1 else f"P{i+1}"
        plane_label = f"{side} motor"

        if driver_instr == "proximity_xy":
            for direction, side_xy in (("X", "R"), ("Y", "L")):
                sensors.append(_with_position(new_sensor(
                    plane=plane_idx, plane_label=plane_label, side=side_xy,
                    angle_deg=45.0, direction=direction,
                    sensor_type="proximity", unit_native="mil pp",
                    alarm=float(state.get("proximity_alarm_mil_pp", 4.0)),
                    danger=float(state.get("proximity_danger_mil_pp", 6.0)),
                    csv_match_pattern=f"*motor*{side}*{direction}*",
                )))
        else:  # axial_accel u otro
            sensors.append(_with_position(new_sensor(
                plane=plane_idx, plane_label=plane_label, side="T",
                angle_deg=0.0, direction="RAD",
                sensor_type="accelerometer", unit_native="g rms",
                alarm=float(state.get("accel_alarm_g", 4.5)),
                danger=float(state.get("accel_danger_g", 9.0)),
                csv_match_pattern=f"*motor*{side}*",
            )))
        plane_idx += 1

    # 2. Frame del compresor — 2 velocímetros (top + side)
    # Patterns más permisivos (Ciclo 22.1a): cubren más variantes de naming
    # de Bently Nevada, CSI 2140, ADRE, etc.
    frame_plane = plane_idx
    sensors.append(_with_position(new_sensor(
        plane=frame_plane, plane_label="Frame top", side="T",
        angle_deg=0.0, direction="RAD",
        sensor_type="velometer", unit_native="mm/s rms",
        alarm=float(state.get("velocity_alarm_mm_s", 4.5)),
        danger=float(state.get("velocity_danger_mm_s", 11.2)),
        csv_match_pattern="*frame*top*, *ftop*, *frame*tope*",
    )))
    sensors.append(_with_position(new_sensor(
        plane=frame_plane, plane_label="Frame side", side="L",
        angle_deg=90.0, direction="RAD",
        sensor_type="velometer", unit_native="mm/s rms",
        alarm=float(state.get("velocity_alarm_mm_s", 4.5)),
        danger=float(state.get("velocity_danger_mm_s", 11.2)),
        csv_match_pattern="*frame*side*, *frame*lat*, *fside*",
    )))
    plane_idx += 1

    # 3. Crosshead accelerometer + rod drop por cilindro
    n_cyl = int(state.get("cylinders_count", 4))
    include_rod_drop = bool(state.get("include_rod_drop", True))
    for c in range(1, n_cyl + 1):
        cyl_label = f"Cilindro {c}"
        # Crosshead acelerómetro — pattern OR cubre múltiples órdenes y abreviaciones
        crosshead_pat = (
            f"*crosshead*cyl{c}*, *cyl{c}*crosshead*, *crosshead*c{c}*, "
            f"*c{c}*crosshead*, *acc*cyl{c}*, *acel*c{c}*, *cylinder{c}*acc*"
        )
        sensors.append(_with_position(new_sensor(
            plane=plane_idx, plane_label=cyl_label, side="T",
            angle_deg=0.0, direction="RAD",
            sensor_type="accelerometer", unit_native="g pk",
            alarm=float(state.get("accel_alarm_g", 4.5)),
            danger=float(state.get("accel_danger_g", 9.0)),
            csv_match_pattern=crosshead_pat,
        )))
        # Rod drop (opcional)
        if include_rod_drop:
            rd_pat = (
                f"*rod*drop*cyl{c}*, *roddrop*{c}*, *rd*cyl{c}*, "
                f"*rod*{c}*, *rd*c{c}*"
            )
            sensors.append(_with_position(new_sensor(
                plane=plane_idx, plane_label=f"{cyl_label} rod drop", side="B",
                angle_deg=270.0, direction="Z",
                sensor_type="proximity", unit_native="mil pp",
                alarm=15.0, danger=25.0,
                csv_match_pattern=rd_pat,
            )))
        plane_idx += 1

    # 4. Keyphasor opcional
    if state.get("include_keyphasor"):
        sensors.append(_with_position(new_sensor(
            plane=plane_idx, plane_label="Keyphasor", side="",
            angle_deg=0.0, direction="",
            sensor_type="keyphasor", unit_native="",
            csv_match_pattern="*kphgen*, *keyph*, *kp*",
        )))

    return sensors


def _execute_creation(state: Dict[str, Any]) -> None:
    """Crea instance + persiste sensors + parámetros base."""
    inst_id_raw = (state["instance_id"] or "").strip()
    if not inst_id_raw:
        raise ValueError("El ID del activo es obligatorio.")
    if get_instance(inst_id_raw) is not None:
        raise ValueError(f"Ya existe un activo con ID '{inst_id_raw}'.")

    # 1. Crear instance base.
    # IMPORTANTE: create_instance slugifica internamente (ej. 'TES 1' → 'tes_1').
    # Tomamos el id REAL post-slugify del Instance retornado, que es el que
    # quedó persistido. Si seguimos usando inst_id_raw, todas las operaciones
    # posteriores (update_instance_header, etc.) buscarían un id que no existe
    # y fallarían en silencio — por eso los sensores no se guardaban.
    inst = create_instance(
        instance_id=inst_id_raw,
        profile_key=state.get("profile_key", "custom_manual"),
        tag=state.get("tag", ""),
        serial_number="",
        location=state.get("location", ""),
        notes=state.get("notes", ""),
        seed_from_profile=True,
    )
    inst_id = inst.instance_id  # ← ID realmente persistido (post-slugify)
    state["instance_id"] = inst_id  # ← actualizar state para que el mensaje de éxito muestre el slug real

    # 2. Sensores: usar el override del paso 4 si existe (editado por user),
    #    si no, regenerar el mapa estándar (incluyendo gearbox si aplica).
    sensors = state.get("sensors_override")
    if not sensors:
        sensors = _build_full_sensor_map(state)

    # 2b. Para reciprocantes, generar y persistir el schematic_png
    schematic_png_filename = ""
    if state.get("category") == "reciprocating_compressor":
        try:
            from core.recip_schematic import generate_recip_png
            from core.instance_repository import get_active_repository
            png_bytes = generate_recip_png(
                n_cylinders=int(state.get("cylinders_count", 4)),
                n_motor_planes=int(state.get("driver_planes", 2)),
                motor_label=state.get("driver_type") or "Motor",
                compressor_label=state.get("driven_type") or "Compresor",
            )
            if png_bytes:
                schematic_png_filename = "schematic_recip.png"
                repo = get_active_repository()
                repo.upload_document_bytes(inst_id, schematic_png_filename, png_bytes)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(
                "No pude persistir schematic recip: %s", e
            )

    # 3. Persistir header extendido + sensores.
    # Defense in depth: pasamos tag/notes/location aunque create_instance ya
    # los haya guardado, así si algo se reinicializa intermedio queda igual.
    update_instance_header(
        instance_id=inst_id,
        tag=state.get("tag", ""),
        location=state.get("location", ""),
        notes=state.get("notes", ""),
        profile_key=state.get("profile_key", "custom_manual"),
        client=state.get("client", ""),
        site=state.get("site", ""),
        asset_class=state.get("asset_class", "") or state.get("driver_type", ""),
        driver_manufacturer=state.get("driver_manufacturer", ""),
        driver_model=state.get("driver_type", ""),
        driver_serial="",
        driven_manufacturer=state.get("driven_manufacturer", ""),
        driven_model=state.get("driven_type", ""),
        driven_serial="",
        nominal_power_mw=float(state.get("nominal_power_mw", 0.0)),
        nominal_rpm=float(state.get("nominal_rpm", 0.0)),
        coupling_class=state.get("coupling_class", "flexible"),
        # Ciclo 23.13 — iconografía 2D vectorial (asset library)
        driver_icon_key=state.get("driver_icon_key", ""),
        driven_icon_key=state.get("driven_icon_key", ""),
        iso_norm_code=state.get("iso_norm_code", ""),
        iso_norm_class=state.get("iso_norm_class", ""),
        sensors=sensors,
        schematic_png=schematic_png_filename,
    )


# =============================================================
# Helpers de UI — DEFINIDOS ANTES de los pasos para que existan
# cuando los bloques `elif current == 4:` los llamen.
# =============================================================

def _render_icon_picker(
    state: Dict[str, Any],
    role: str,        # 'driver' | 'driven'
    state_key: str,   # 'driver_icon_key' | 'driven_icon_key'
    column,
) -> Optional[Dict[str, Any]]:
    """
    Selector visual del icono del activo (Ciclo 23.13).

    Usa core.asset_library.list_by_category() para mostrar dropdowns
    estilo System1 / Emerson AMS — primero categoría, después icono
    específico, con OEM examples y pre-fill de planos / cojinete.

    Devuelve la metadata del icono seleccionado (o None si no eligió).
    """
    try:
        from core.asset_library import list_by_category, get_asset_meta
    except ImportError:
        return None

    by_cat = list_by_category(role=role)  # {"Motores eléctricos": [...], ...}
    if not by_cat:
        return None

    with column:
        st.markdown(f"**Icono visual** ({role})")
        # Construir lista plana ordenada por categoría
        categories = sorted(by_cat.keys())

        # Dropdown 1: categoría
        cat_options = ["— Sin icono —"] + categories
        current_key = state.get(state_key, "")
        current_meta = get_asset_meta(current_key) if current_key else None
        current_cat = current_meta.get("category") if current_meta else "— Sin icono —"
        if current_cat not in cat_options:
            current_cat = "— Sin icono —"

        cat_pick = st.selectbox(
            f"Categoría",
            options=cat_options,
            index=cat_options.index(current_cat),
            key=f"wiz_icon_cat_{role}",
        )

        if cat_pick == "— Sin icono —":
            state[state_key] = ""
            return None

        # Dropdown 2: icono específico dentro de la categoría
        items = by_cat.get(cat_pick, [])
        item_keys = [it["key"] for it in items]
        item_labels = {
            it["key"]: f"{it['default_label']}  ·  {', '.join(it.get('oem_examples', [])[:2])}"
            for it in items
        }

        # Default al item actual si está en esa categoría, si no al primero
        default_idx = item_keys.index(current_key) if current_key in item_keys else 0

        icon_pick = st.selectbox(
            f"Modelo",
            options=item_keys,
            format_func=lambda k: item_labels.get(k, k),
            index=default_idx,
            key=f"wiz_icon_pick_{role}",
        )

        # Persistir + retornar metadata
        state[state_key] = icon_pick
        meta = get_asset_meta(icon_pick)

        # Pre-fill liviano: si el textfield del nombre está vacío, pre-llenar
        # con el primer OEM example. Si el user ya escribió algo lo respetamos.
        if meta:
            type_field = f"{role}_type"
            if not (state.get(type_field) or "").strip():
                ex = meta.get("oem_examples") or []
                state[type_field] = ex[0] if ex else meta.get("default_label", "")
            # Pre-fill planes y bearing_kind solo si están en defaults vírgenes
            planes_field = f"{role}_planes"
            if int(state.get(planes_field, 2)) in (2,) and meta.get("typical_planes"):
                state[planes_field] = int(meta["typical_planes"])
            bearing_field = f"{role}_bearing_kind"
            support_type = meta.get("support_type", "")
            if support_type == "fluid_film":
                state[bearing_field] = "plain"
            elif support_type == "rolling_element":
                state[bearing_field] = "rolling"

        return meta


def _render_train_preview(state: Dict[str, Any]) -> None:
    """
    Preview SVG en vivo del tren acoplado armado con la asset library.
    Se renderiza solo si el user eligió ambos iconos (driver + driven).
    """
    drv = state.get("driver_icon_key", "")
    drvn = state.get("driven_icon_key", "")
    if not drv or not drvn:
        return
    try:
        from core.asset_library.composer import compose_train
        svg = compose_train(
            driver_key=drv,
            driven_key=drvn,
            driver_label=state.get("driver_type", "") or "Driver",
            driven_label=state.get("driven_type", "") or "Driven",
            coupling=state.get("coupling_class", "flexible"),
            sensors_with_status=[],
        )
        st.markdown("**Preview del tren acoplado**")
        st.markdown(
            f'<div style="background:#ffffff;border:1px solid #e2e8f0;'
            f'border-radius:10px;padding:14px;">{svg}</div>',
            unsafe_allow_html=True,
        )
    except Exception as e:
        st.caption(f"(Preview no disponible: {e})")


def _render_sensors_table_editor(state: Dict[str, Any], sensors: List[Dict[str, Any]]) -> None:
    """Render del data_editor de pandas con la lista de sensores."""
    import pandas as pd
    df_sensors = pd.DataFrame([
        {
            "plane_label": s.get("plane_label", ""),
            "side": s.get("side", "L"),
            "angle_deg": s.get("angle_deg", 45.0),
            "direction": s.get("direction", "Y"),
            "sensor_type": s.get("sensor_type", "proximity"),
            "unit_native": s.get("unit_native", ""),
            "alarm": s.get("alarm", 0.0),
            "danger": s.get("danger", 0.0),
            "csv_match_pattern": s.get("csv_match_pattern", ""),
        }
        for s in sensors
    ])

    edited = st.data_editor(
        df_sensors,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "plane_label": st.column_config.TextColumn("Plano", width="medium"),
            "side": st.column_config.SelectboxColumn(
                "Lado", options=["L", "R", "T", "B", ""], width="small",
            ),
            "angle_deg": st.column_config.NumberColumn(
                "Ángulo (°)", min_value=0.0, max_value=360.0, step=15.0,
                width="small",
            ),
            "direction": st.column_config.SelectboxColumn(
                "Dirección", options=["X", "Y", "Z", "RAD", "AX", ""],
                width="small",
            ),
            "sensor_type": st.column_config.SelectboxColumn(
                "Tipo",
                options=["proximity", "accelerometer", "velometer", "keyphasor"],
                width="medium",
            ),
            "unit_native": st.column_config.TextColumn("Unidad", width="small"),
            "alarm": st.column_config.NumberColumn("Alarm", width="small"),
            "danger": st.column_config.NumberColumn("Danger", width="small"),
            "csv_match_pattern": st.column_config.TextColumn(
                "Pattern CSV (opcional)", width="medium",
            ),
        },
        key="wiz_sensors_editor",
    )
    state["_wizard_table_edited"] = edited


def _infer_icon_side_anchor(
    sensor: Dict[str, Any],
    drv_icon_key: str,
    drvn_icon_key: str,
) -> tuple:
    """
    Heurística Bently / API 670 para mapear un sensor del wizard a
    (icon_side, icon_anchor) del SVG library. Corregida en Ciclo 23.19
    para reflejar la convención real: bearings se numeran desde el
    extremo libre del driver hacia el generador.

      plane_label contains "driver/motor/turbina/engine"  → driver
      plane_label contains "driven/compresor/generador/bomba/frame/cilindro" → driven
      plane==1  → driver / NDE  (CRF en aero-derivative) = lado libre
      plane==2  → driver / DE   (TRF en aero-derivative) = lado coupling
      plane==3  → driven / DE   = lado coupling
      plane==4  → driven / NDE  = lado libre

    Devuelve ('', '') si no se pudo inferir → el especialista lo asigna
    manualmente con los selectboxes del editor.
    """
    plane_l = ((sensor.get("plane_label") or "")).lower()
    plane_num = int(sensor.get("plane") or 0)
    is_aero = "aero" in (drv_icon_key or "").lower()

    side = ""
    if any(t in plane_l for t in ("driver", "motor", "turbina", "engine")):
        side = "driver"
    elif any(t in plane_l for t in (
        "driven", "compresor", "compressor", "generador", "generator",
        "bomba", "pump", "frame", "cilindro", "cylinder",
    )):
        side = "driven"
    elif plane_num in (1, 2):
        side = "driver"
    elif plane_num >= 3:
        side = "driven"

    anchor = ""
    if "nde" in plane_l:
        anchor = "CRF" if (side == "driver" and is_aero) else "NDE"
    elif "de" in plane_l:
        anchor = "TRF" if (side == "driver" and is_aero) else "DE"
    elif plane_num == 1:
        anchor = "CRF" if is_aero else "NDE"  # lado libre
    elif plane_num == 2:
        anchor = "TRF" if is_aero else "DE"   # lado coupling
    elif plane_num == 3:
        anchor = "DE"
    elif plane_num == 4:
        anchor = "NDE"

    return side, anchor


def _autopopulate_icon_anchors(
    state: Dict[str, Any],
    sensors: List[Dict[str, Any]],
) -> None:
    """Llena icon_side / icon_anchor en cada sensor que no los tenga."""
    drv = state.get("driver_icon_key", "")
    drvn = state.get("driven_icon_key", "")
    if not drv or not drvn:
        return
    for s in sensors:
        if s.get("icon_side") and s.get("icon_anchor"):
            continue
        side, anchor = _infer_icon_side_anchor(s, drv, drvn)
        if side:
            s.setdefault("icon_side", side)
            if not s.get("icon_side"):
                s["icon_side"] = side
        if anchor:
            s.setdefault("icon_anchor", anchor)
            if not s.get("icon_anchor"):
                s["icon_anchor"] = anchor


def _render_visual_editor_library(
    state: Dict[str, Any],
    sensors: List[Dict[str, Any]],
) -> None:
    """
    Editor visual SVG basado en core.asset_library (Ciclo 23.13).

    Para cada sensor, el especialista elige (icon_side, icon_anchor)
    en dos selectboxes — los anchors disponibles dependen del icon_key
    del lado elegido. El SVG del tren se redibuja con los sensor dots
    en sus anchors físicos. Lo que queda persistido es lo que después
    Live Monitoring rinde idéntico, sin click-to-place de coordenadas
    arbitrarias.
    """
    from core.asset_library import available_anchors
    from core.asset_library.composer import compose_train

    drv = state.get("driver_icon_key", "")
    drvn = state.get("driven_icon_key", "")
    drv_anchors = available_anchors(drv)
    drvn_anchors = available_anchors(drvn)

    # Auto-poblar para sensores nuevos (idempotente)
    _autopopulate_icon_anchors(state, sensors)

    col_svg, col_list = st.columns([3, 2])

    with col_svg:
        # Build sensors_with_status para el preview
        s_for_svg = []
        from core.sensor_map import sensor_label as _sensor_label_fn
        for s in sensors:
            side = (s.get("icon_side") or "").strip()
            anchor = (s.get("icon_anchor") or "").strip()
            if not side or not anchor:
                continue
            try:
                lbl = _sensor_label_fn(s)
            except Exception:
                lbl = s.get("plane_label", "?")
            # Display label sin underscore para el SVG (Ciclo 23.18) — '2Y_A' → '2YA'.
            display_lbl = lbl.replace("_", "")
            s_for_svg.append({
                "label": display_lbl, "side": side, "anchor": anchor,
                "status": "Sin Norma",  # sin readings live en el wizard
                "value": "", "unit": "",
                "title": f"{lbl} · {s.get('sensor_type','')}",
            })

        try:
            svg = compose_train(
                driver_key=drv,
                driven_key=drvn,
                driver_label=state.get("driver_type") or "Driver",
                driven_label=state.get("driven_type") or "Driven",
                coupling=state.get("coupling_class", "flexible"),
                sensors_with_status=s_for_svg,
            )
            st.markdown(
                f'<div style="background:#ffffff;border:1px solid #e2e8f0;'
                f'border-radius:10px;padding:14px;">{svg}</div>',
                unsafe_allow_html=True,
            )
        except Exception as e:
            st.warning(f"No se pudo generar el SVG library: {e}")

        n_mapped = len(s_for_svg)
        n_total = len(sensors)
        st.caption(
            f"📍 **{n_mapped} de {n_total}** sensores asignados a un anchor. "
            "Los no asignados no aparecen en el preview ni en Live Monitoring."
        )
        if st.button("🪄 Auto-asignar todos por heurística (Bently / API 670)",
                     key="wiz_lib_autoassign", use_container_width=True):
            for s in sensors:
                side, anchor = _infer_icon_side_anchor(s, drv, drvn)
                if side:
                    s["icon_side"] = side
                if anchor:
                    s["icon_anchor"] = anchor
            state["sensors_override"] = sensors
            st.rerun()

    with col_list:
        st.markdown("**Asignación de cada sensor a un cojinete físico**")
        st.caption(
            "Side = en qué máquina del tren (driver / driven). "
            "Anchor = qué cojinete físico del icono (DE, NDE, TRF, CRF...)."
        )

        for idx, s in enumerate(sensors):
            try:
                from core.sensor_map import sensor_label as _slbl
                label = _slbl(s)
            except Exception:
                label = s.get("plane_label", f"sensor {idx+1}")
            stype = s.get("sensor_type", "")

            with st.container(border=True):
                st.markdown(f"**#{idx+1} · {label}** · `{stype}`")
                c1, c2 = st.columns(2)
                with c1:
                    side_options = ["", "driver", "driven", "coupling"]
                    side_cur = (s.get("icon_side") or "").strip()
                    side_idx = side_options.index(side_cur) if side_cur in side_options else 0
                    side_pick = st.selectbox(
                        "Side", side_options, index=side_idx,
                        key=f"wiz_lib_side_{idx}",
                        label_visibility="collapsed",
                        format_func=lambda x: x or "— side —",
                    )
                    s["icon_side"] = side_pick

                with c2:
                    if side_pick == "driver":
                        anch_options = [""] + drv_anchors
                    elif side_pick == "driven":
                        anch_options = [""] + drvn_anchors
                    else:
                        anch_options = [""]
                    anch_cur = (s.get("icon_anchor") or "").strip()
                    anch_idx = anch_options.index(anch_cur) if anch_cur in anch_options else 0
                    anch_pick = st.selectbox(
                        "Anchor", anch_options, index=anch_idx,
                        key=f"wiz_lib_anchor_{idx}",
                        label_visibility="collapsed",
                        format_func=lambda x: x or "— anchor —",
                    )
                    s["icon_anchor"] = anch_pick

        state["sensors_override"] = sensors


def _render_visual_editor(state: Dict[str, Any], sensors: List[Dict[str, Any]]) -> None:
    """
    Editor visual del sensor map.

    Ciclo 23.13 — Si la instancia tiene driver_icon_key + driven_icon_key,
    se rinde el SVG vectorial del library con dropdowns side/anchor por
    sensor. Si no, fallback al PNG legacy con click-to-place.

      - reciprocating_compressor → core.recip_schematic.generate_recip_png
      - turbomachinery / motor_pump / generic → core.train_schematic.generate_train_png

    Antes (pre Ciclo 23.8) este editor solo se mostraba para recip,
    dejando a los demás activos sin posicionamiento visual de sensores.
    """
    # Camino preferido: SVG library (System1-style)
    if state.get("driver_icon_key") and state.get("driven_icon_key"):
        _render_visual_editor_library(state, sensors)
        return

    # Fallback legacy: PNG generado + click-to-place de x_pct/y_pct
    try:
        from streamlit_image_coordinates import streamlit_image_coordinates
        _HAS_COORDS = True
    except ImportError:
        _HAS_COORDS = False

    category = state.get("category", "")
    is_recip = category == "reciprocating_compressor"

    if is_recip:
        from core.recip_schematic import generate_recip_png
        n_cyl = int(state.get("cylinders_count", 4))
        n_motor = int(state.get("driver_planes", 2))
        png_bytes = generate_recip_png(
            n_cylinders=n_cyl, n_motor_planes=n_motor,
            has_distance_piece=True,
            motor_label=state.get("driver_type") or "Motor",
            compressor_label=state.get("driven_type") or "Compresor",
        )
    else:
        # Tren acoplado genérico (turbo, motor+bomba, motor+generador, etc.)
        from core.train_schematic import generate_train_png
        n_d = int(state.get("driver_planes", 2))
        n_dn = int(state.get("driven_planes", 2))
        png_bytes = generate_train_png(
            driver_label=state.get("driver_type") or "Driver",
            driven_label=state.get("driven_type") or "Driven",
            n_driver_planes=n_d,
            n_driven_planes=n_dn,
        )

    if not png_bytes:
        st.warning("Pillow no está disponible — no puedo generar el schematic.")
        return

    from PIL import Image, ImageDraw, ImageFont
    import io as _io
    base_img = Image.open(_io.BytesIO(png_bytes)).convert("RGBA")
    width, height = base_img.size

    overlay = base_img.copy()
    draw = ImageDraw.Draw(overlay)
    try:
        font_marker = ImageFont.truetype("Arial.ttf", 16)
    except Exception:
        font_marker = ImageFont.load_default()

    color_by_type = {
        "proximity": "#ef4444",
        "accelerometer": "#a855f7",
        "velometer": "#3b82f6",
        "keyphasor": "#f59e0b",
    }

    selected_idx = state.get("_wiz_selected_sensor_idx", -1)

    # Si los sensores no tienen x_pct/y_pct seteados, computamos defaults
    # heurísticos según el plano + dirección para que el primer render
    # tenga sensores ya distribuidos lógicamente sobre el activo (no todos
    # apilados en 50%, 50%).
    if not is_recip:
        from core.train_schematic import sensor_default_position as _train_default_pos
    n_d_planes = int(state.get("driver_planes", 2))
    n_dn_planes = int(state.get("driven_planes", 2))

    for idx, s in enumerate(sensors):
        # Resolver coordenadas iniciales si están en None / faltantes
        raw_x = s.get("x_pct")
        raw_y = s.get("y_pct")
        if raw_x is None or raw_y is None:
            if not is_recip:
                dx, dy = _train_default_pos(s, n_d_planes, n_dn_planes)
            else:
                dx, dy = 50.0, 50.0
            if raw_x is None:
                s["x_pct"] = dx
            if raw_y is None:
                s["y_pct"] = dy
        try:
            x_pct = float(s.get("x_pct") or 50.0)
        except Exception:
            x_pct = 50.0
        try:
            y_pct = float(s.get("y_pct") or 50.0)
        except Exception:
            y_pct = 50.0
        x = int(width * x_pct / 100)
        y = int(height * y_pct / 100)
        c = color_by_type.get(s.get("sensor_type", ""), "#64748b")
        r = 11
        if idx == selected_idx:
            draw.ellipse((x - r - 4, y - r - 4, x + r + 4, y + r + 4),
                         outline="#000000", width=3)
        draw.ellipse((x - r, y - r, x + r, y + r), fill=c, outline="#ffffff", width=2)
        draw.text((x - 5, y - 9), str(idx + 1), fill="#ffffff", font=font_marker)

    col_img, col_list = st.columns([3, 2])

    with col_img:
        if _HAS_COORDS:
            coords = streamlit_image_coordinates(
                overlay, key="wiz_recip_canvas",
                width=min(width, 1100),
            )
            if coords and selected_idx >= 0:
                rendered_w = coords.get("width", width)
                rendered_h = coords.get("height", height)
                cx = coords.get("x", 0)
                cy = coords.get("y", 0)
                new_x_pct = (cx / rendered_w) * 100
                new_y_pct = (cy / rendered_h) * 100
                sensors[selected_idx]["x_pct"] = round(new_x_pct, 1)
                sensors[selected_idx]["y_pct"] = round(new_y_pct, 1)
                state["sensors_override"] = sensors
                state["_wiz_selected_sensor_idx"] = -1
                st.rerun()
        else:
            st.image(overlay)
            st.caption("(streamlit_image_coordinates no disponible — solo visualización)")

    with col_list:
        st.markdown("**Sensores** (click para seleccionar y reposicionar)")
        for idx, s in enumerate(sensors):
            label = s.get("plane_label", "") or ""
            stype = s.get("sensor_type", "") or ""
            # Defensive: dict.get(key, default) NO usa default cuando el key
            # existe con valor None (caso normal cuando _build_full_sensor_map
            # no setea x_pct/y_pct iniciales). Forzamos float con fallback.
            try:
                xp = float(s.get("x_pct") or 50.0)
            except Exception:
                xp = 50.0
            try:
                yp = float(s.get("y_pct") or 50.0)
            except Exception:
                yp = 50.0
            sel_str = "🎯 " if idx == selected_idx else ""
            btn_label = f"{sel_str}#{idx+1} · {label} · {stype} ({xp:.0f}%, {yp:.0f}%)"
            if st.button(btn_label, key=f"wiz_sel_sensor_{idx}",
                         use_container_width=True):
                state["_wiz_selected_sensor_idx"] = idx
                st.rerun()
        if st.button("❌ Deseleccionar", key="wiz_deselect_sensor"):
            state["_wiz_selected_sensor_idx"] = -1
            st.rerun()


# =============================================================
# Header del wizard
# =============================================================

st.title("Crear activo · Wizard guiado")
st.caption(
    "Configurá una máquina nueva en 5 pasos. El sistema arma automáticamente "
    "el mapa de sensores y los setpoints según las normas ISO/API correspondientes."
)

state = _wizard_state()
current = state["step"]

# Stepper visual.
# Ciclo 23.6 — orden corregido: primero unidades & setpoints, después editor
# de sensores (así el editor ya muestra los setpoints OEM elegidos por el
# usuario y no hay que ir-volver entre pasos).
step_labels = [
    "1 · Tipo",
    "2 · Tren",
    "3 · Instrumentación",
    "4 · Unidades & setpoints",
    "5 · Editar sensores",
    "6 · Datos del activo",
]
step_cols = st.columns(TOTAL_STEPS)
for i, (col, label) in enumerate(zip(step_cols, step_labels), start=1):
    with col:
        if i < current:
            st.markdown(f"<div style='padding:8px 12px;border-radius:8px;"
                        f"background:#dcfce7;color:#166534;font-weight:600;"
                        f"font-size:12px;'>✓ {label}</div>",
                        unsafe_allow_html=True)
        elif i == current:
            st.markdown(f"<div style='padding:8px 12px;border-radius:8px;"
                        f"background:#dbeafe;color:#1d4ed8;font-weight:700;"
                        f"font-size:12px;border:2px solid #3b82f6;'>{label}</div>",
                        unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='padding:8px 12px;border-radius:8px;"
                        f"background:#f1f5f9;color:#64748b;font-size:12px;'>"
                        f"{label}</div>",
                        unsafe_allow_html=True)

st.markdown("---")


# =============================================================
# PASO 1 — Tipo de máquina + plantilla LATAM
# =============================================================

if current == 1:
    st.subheader("Paso 1 · ¿Qué tipo de máquina vas a monitorear?")

    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.markdown("**A) Empezar desde cero**")
        st.caption("Configurá toda la máquina manualmente.")
        category_options = {
            "turbomachinery": "🌀 Turbomáquina (turbina/turbogenerador)",
            "centrifugal_compressor": "⚙️ Compresor centrífugo",
            "reciprocating_compressor": "🔄 Compresor reciprocante",
            "centrifugal_pump": "💧 Bomba centrífuga",
            "electric_motor": "⚡ Motor eléctrico",
            "fan_blower": "🌬️ Ventilador / soplador",
            "custom": "🛠️ Otro / personalizado",
        }
        cat_pick = st.radio(
            "Categoría",
            options=list(category_options.keys()),
            format_func=lambda k: category_options[k],
            index=list(category_options.keys()).index(state["category"])
                  if state["category"] in category_options else 0,
            key="wiz_step1_cat",
        )

    with col_b:
        st.markdown("**B) Usar una plantilla LATAM**")
        st.caption(
            "Pre-carga 20+ máquinas comunes en O&G/generación con sus "
            "rodamientos típicos, normas y esquemas de sensores recomendados."
        )
        cat_filter = state.get("template_filter_cat", "Todas")
        cats = ["Todas"] + list_categories()
        cat_filter = st.selectbox(
            "Filtrar por categoría",
            options=cats,
            index=cats.index(cat_filter) if cat_filter in cats else 0,
            key="wiz_template_cat",
        )
        if cat_filter == "Todas":
            templates = list_templates()
        else:
            templates = list_templates_by_category(cat_filter)
        template_options = {"": "— Sin plantilla —"}
        for t in templates:
            template_options[t.id] = t.label
        sel_template = st.selectbox(
            "Plantilla",
            options=list(template_options.keys()),
            format_func=lambda k: template_options[k],
            index=list(template_options.keys()).index(state["template_id"])
                  if state["template_id"] in template_options else 0,
            key="wiz_template_id",
        )

    # Botón Siguiente
    col_nav = st.columns([3, 1])
    with col_nav[1]:
        if st.button("Siguiente →", type="primary", use_container_width=True,
                     key="wiz_step1_next"):
            state["category"] = cat_pick
            state["template_id"] = sel_template
            # Si eligió plantilla, pre-cargar paso 5 + sugerencias
            if sel_template:
                t = get_template(sel_template)
                if t:
                    state["template_label"] = t.label
                    state["asset_class"] = t.label
                    state["driver_manufacturer"] = t.manufacturer
                    state["driver_model"] = t.model
                    state["nominal_rpm"] = float(t.operating_rpm_nominal or 0)
                    if t.rated_power_kw and len(t.rated_power_kw) >= 1:
                        state["nominal_power_mw"] = float(t.rated_power_kw[-1]) / 1000.0
                    state["iso_norm_code"] = t.iso_norm_recommended or ""
                    state["iso_norm_class"] = t.iso_class_recommended or ""
                    state["notes"] = t.notes or ""
                    state["driver_bearing_kind"] = (
                        "plain" if "plain" in (t.bearing_type or "").lower() else "rolling"
                    )
                    state["category"] = t.category
                    suggested = suggest_profile_key_for_template(t.id)
                    if suggested:
                        state["profile_key"] = suggested
            _go_next()
            st.rerun()


# =============================================================
# PASO 2 — Tren mecánico
# =============================================================

elif current == 2:
    st.subheader("Paso 2 · Tren mecánico")
    st.caption("Cuántas máquinas hay acopladas, qué tipo de cojinete, qué acople.")

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("### Driver (motriz)")
        # Selector visual de icono (asset library System1-style)
        _render_icon_picker(state, role="driver", state_key="driver_icon_key", column=col_l)
        state["driver_type"] = st.text_input(
            "Tipo / descripción",
            value=state.get("driver_type") or state.get("driver_model") or "",
            placeholder="Ej: GE LM6000, Solar Mars 100, Motor 4 polos",
            key="wiz_driver_type",
        )
        state["driver_planes"] = st.number_input(
            "Cantidad de cojinetes (planos de medición)",
            min_value=1, max_value=6,
            value=int(state.get("driver_planes", 2) or 2),
            key="wiz_driver_planes",
        )
        state["driver_bearing_kind"] = st.radio(
            "Tipo de cojinete",
            options=["plain", "rolling"],
            format_func=lambda k: {"plain": "🧱 Cojinete plano (fluid film)",
                                   "rolling": "⚙️ Rodamiento (rolling element)"}[k],
            index=["plain", "rolling"].index(state.get("driver_bearing_kind", "plain")),
            key="wiz_driver_bearing_kind",
        )
        if state["driver_bearing_kind"] == "rolling":
            state["driver_bearing_model"] = st.text_input(
                "Modelo de rodamiento típico (opcional)",
                value=state.get("driver_bearing_model", ""),
                placeholder="Ej: SKF 6319, NU 220",
                key="wiz_driver_bearing_model",
            )

    is_recip = state.get("category") == "reciprocating_compressor"

    with col_r:
        if is_recip:
            st.markdown("### Driven · Compresor reciprocante")
            # Selector visual de icono (asset library System1-style)
            _render_icon_picker(state, role="driven", state_key="driven_icon_key", column=col_r)
            state["driven_type"] = st.text_input(
                "Modelo / fabricante",
                value=state.get("driven_type") or state.get("driven_model") or "",
                placeholder="Ej: Ariel KBK/4, Burckhardt Process, Dresser-Rand HOS",
                key="wiz_driven_type",
            )
            state["cylinders_count"] = st.select_slider(
                "Número de cilindros",
                options=[2, 4, 6, 8],
                value=int(state.get("cylinders_count", 4) or 4),
                help="Frecuencia 1X = RPM/60; cada cilindro suma armónicos según "
                     "configuración (single/double-acting, etapas).",
                key="wiz_recip_cylinders",
            )
            state["include_rod_drop"] = st.checkbox(
                "Incluir sensores de rod drop (1 por cilindro)",
                value=bool(state.get("include_rod_drop", True)),
                help="Detección de desgaste de packing/rider band. Estándar API 618.",
                key="wiz_recip_rod_drop",
            )
            # En reciprocantes el "driven_planes" representa frame planes (top + side)
            state["driven_planes"] = 2
            state["driven_bearing_kind"] = "plain"
            st.caption(
                "ℹ️ En reciprocantes los sensores van en el frame del compresor "
                "y en cada crosshead, no en cojinetes radiales. ISO 20816-8."
            )
        else:
            st.markdown("### Driven (accionada)")
            # Selector visual de icono (asset library System1-style)
            _render_icon_picker(state, role="driven", state_key="driven_icon_key", column=col_r)
            state["driven_type"] = st.text_input(
                "Tipo / descripción",
                value=state.get("driven_type") or state.get("driven_model") or "",
                placeholder="Ej: Generador Brush 54MW, Compresor Ariel KBK/4",
                key="wiz_driven_type",
            )
            state["driven_planes"] = st.number_input(
                "Cantidad de cojinetes (planos de medición)",
                min_value=1, max_value=6,
                value=int(state.get("driven_planes", 2) or 2),
                key="wiz_driven_planes",
            )
            state["driven_bearing_kind"] = st.radio(
                "Tipo de cojinete",
                options=["plain", "rolling"],
                format_func=lambda k: {"plain": "🧱 Cojinete plano (fluid film)",
                                       "rolling": "⚙️ Rodamiento (rolling element)"}[k],
                index=["plain", "rolling"].index(state.get("driven_bearing_kind", "rolling")),
                key="wiz_driven_bearing_kind",
            )
            if state["driven_bearing_kind"] == "rolling":
                state["driven_bearing_model"] = st.text_input(
                    "Modelo de rodamiento típico (opcional)",
                    value=state.get("driven_bearing_model", ""),
                    placeholder="Ej: SKF 6319, NU 220",
                    key="wiz_driven_bearing_model",
                )

    st.markdown("---")

    # ===== Gearbox opcional (Ciclo 21.2) =====
    state["include_gearbox"] = st.checkbox(
        "⚙️ Incluir gearbox / multiplicador entre driver y driven",
        value=bool(state.get("include_gearbox", False)),
        help="Activá si hay una caja reductora/multiplicadora intermedia "
             "(ej: turbina + gearbox + generador, motor + reductor + bomba).",
        key="wiz_include_gearbox",
    )

    if state["include_gearbox"]:
        with st.container(border=True):
            st.markdown("### Gearbox / Caja reductora")
            gc1, gc2 = st.columns(2)
            with gc1:
                state["gearbox_type"] = st.text_input(
                    "Tipo / fabricante (opcional)",
                    value=state.get("gearbox_type", ""),
                    placeholder="Ej: Lufkin, Voith, Flender",
                    key="wiz_gearbox_type",
                )
                state["gearbox_planes"] = st.number_input(
                    "Cojinetes del gearbox",
                    min_value=2, max_value=6,
                    value=int(state.get("gearbox_planes", 2)),
                    help="Típico 2 (HSS = high speed shaft + LSS = low speed shaft). "
                         "Hasta 4 si tiene piñones intermedios.",
                    key="wiz_gearbox_planes",
                )
            with gc2:
                state["gearbox_bearing_kind"] = st.radio(
                    "Tipo de cojinete del gearbox",
                    options=["plain", "rolling"],
                    format_func=lambda k: {"plain": "🧱 Plano",
                                           "rolling": "⚙️ Rodamiento"}[k],
                    index=["plain", "rolling"].index(
                        state.get("gearbox_bearing_kind", "rolling")
                    ),
                    horizontal=True,
                    key="wiz_gearbox_bearing_kind",
                )
                if state["gearbox_bearing_kind"] == "rolling":
                    state["gearbox_bearing_model"] = st.text_input(
                        "Modelo de rodamiento típico (opcional)",
                        value=state.get("gearbox_bearing_model", ""),
                        placeholder="Ej: SKF NU 220, Timken 32220",
                        key="wiz_gearbox_bearing_model",
                    )

    st.markdown("### Acople")
    state["coupling_class"] = st.radio(
        "Tipo de acople",
        options=["rigid", "flexible", "fluid"],
        format_func=lambda k: {
            "rigid": "🔗 Rígido",
            "flexible": "🌀 Flexible (gear/disk/diaphragm)",
            "fluid": "💧 Hidrodinámico / fluid coupling",
        }[k],
        index=["rigid", "flexible", "fluid"].index(state.get("coupling_class", "flexible")),
        horizontal=True,
        key="wiz_coupling",
    )

    # Preview SVG del tren acoplado (Ciclo 23.13) — solo si el usuario eligió
    # ambos iconos. Si no, mostramos un hint para que use los selectores.
    st.markdown("---")
    if state.get("driver_icon_key") and state.get("driven_icon_key"):
        _render_train_preview(state)
    else:
        st.caption(
            "💡 Elegí icono visual del driver y driven arriba para ver el "
            "preview del tren acoplado completo (estilo System1 / AMS)."
        )

    col_nav = st.columns([1, 2, 1])
    with col_nav[0]:
        if st.button("← Atrás", use_container_width=True, key="wiz_step2_prev"):
            _go_prev()
            st.rerun()
    with col_nav[2]:
        if st.button("Siguiente →", type="primary", use_container_width=True,
                     key="wiz_step2_next"):
            _go_next()
            st.rerun()


# =============================================================
# PASO 3 — Instrumentación
# =============================================================

elif current == 3:
    st.subheader("Paso 3 · Instrumentación")
    st.caption(
        "Qué tipo de sensores hay en cada máquina. El sistema arma el mapa "
        "completo según las prácticas API 670 / ISO 20816."
    )

    instrum_options = {
        "proximity_xy": "🎯 Proximidad XY (no contacto, par X-Y a 45°)",
        "axial_accel": "📍 Acelerómetro carcasa (1 radial top por plano)",
        "accel_plus_velocity": "📊 Acelerómetro + Velocímetro (carcasa, completo)",
    }

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("### Driver — instrumentación")
        state["driver_instrumentation"] = st.radio(
            "Tipo de sensores en el driver",
            options=list(instrum_options.keys()),
            format_func=lambda k: instrum_options[k],
            index=list(instrum_options.keys()).index(
                state.get("driver_instrumentation", "proximity_xy")
            ),
            key="wiz_driver_instrum",
        )

    with col_r:
        is_recip = state.get("category") == "reciprocating_compressor"
        if is_recip:
            st.markdown("### Driven · Compresor reciprocante")
            st.info(
                "**Instrumentación API 618 / ISO 20816-8:**\n"
                "- 1 velocímetro frame top (radial)\n"
                "- 1 velocímetro frame side (lateral)\n"
                f"- 1 acelerómetro crosshead × {state.get('cylinders_count', 4)} cilindros\n"
                + (f"- 1 rod drop × {state.get('cylinders_count', 4)} cilindros\n"
                   if state.get("include_rod_drop", True) else "")
            )
            state["driven_instrumentation"] = "reciprocating"
        else:
            st.markdown("### Driven — instrumentación")
            state["driven_instrumentation"] = st.radio(
                "Tipo de sensores en el driven",
                options=list(instrum_options.keys()),
                format_func=lambda k: instrum_options[k],
                index=list(instrum_options.keys()).index(
                    state.get("driven_instrumentation", "proximity_xy")
                ) if state.get("driven_instrumentation") in instrum_options else 0,
                key="wiz_driven_instrum",
            )

    # ===== Instrumentación del gearbox (Ciclo 21.2) =====
    if state.get("include_gearbox"):
        st.markdown("### Gearbox — instrumentación")
        state["gearbox_instrumentation"] = st.radio(
            "Tipo de sensores en el gearbox",
            options=list(instrum_options.keys()),
            format_func=lambda k: instrum_options[k],
            index=list(instrum_options.keys()).index(
                state.get("gearbox_instrumentation", "axial_accel")
            ),
            help="Lo más común en gearboxes industriales es axial_accel "
                 "(acelerómetro carcasa por cojinete).",
            key="wiz_gearbox_instrum",
        )

    st.markdown("---")
    st.markdown("### Referencia 1X (keyphasor)")
    state["include_keyphasor"] = st.checkbox(
        "Incluir keyphasor (sensor de fase 1X en el coupling)",
        value=bool(state.get("include_keyphasor", True)),
        help="Sensor de proximidad apuntando a una marca en el eje. "
             "Necesario para análisis de fase, órbita filtrada y Bode/Polar.",
        key="wiz_keyphasor",
    )

    st.markdown("### Canales por sensor")
    state["channels_per_sensor"] = st.select_slider(
        "Cantidad de señales por sensor",
        options=[1, 2, 3],
        value=int(state.get("channels_per_sensor", 1) or 1),
        help="1 canal: solo amplitud RMS. 2 canales: RMS + waveform. 3 canales: + spectrum.",
        key="wiz_channels",
    )

    # Preview rápido del mapa (incluye gearbox)
    with st.expander("👁️ Vista previa del mapa de sensores que se va a generar",
                     expanded=True):
        try:
            preview = _build_full_sensor_map(state)
            st.markdown(f"**Total de sensores:** {len(preview)}")
            for s in preview:
                plane = s.get("plane_label", f"plano {s.get('plane', '?')}")
                stype = s.get("sensor_type", "?")
                direction = s.get("direction", "?")
                st.markdown(f"- `{plane}` · {stype} · dir: {direction}")
        except Exception as e:
            st.warning(f"No se pudo previsualizar: {e}")

    col_nav = st.columns([1, 2, 1])
    with col_nav[0]:
        if st.button("← Atrás", use_container_width=True, key="wiz_step3_prev"):
            _go_prev()
            st.rerun()
    with col_nav[2]:
        if st.button("Siguiente →", type="primary", use_container_width=True,
                     key="wiz_step3_next"):
            # Ciclo 23.6 — el mapa ya NO se genera acá. Se genera al pasar
            # del nuevo paso 4 (Unidades & setpoints) al nuevo paso 5
            # (Editor de sensores) para que los setpoints elegidos por el
            # usuario se apliquen desde la primera renderización del editor.
            _go_next()
            st.rerun()


# =============================================================
# PASO 4 — Unidades & setpoints (orden corregido Ciclo 23.6)
# Antes era el paso 5. Lo movimos antes del editor de sensores
# para que el editor muestre los setpoints reales elegidos por el
# usuario desde la primera renderización (no defaults ISO).
# =============================================================

elif current == 4:
    st.subheader("Paso 4 · Unidades y setpoints")
    st.caption(
        "Define las unidades en que reporta cada tipo de sensor, y los niveles "
        "alarm/danger. Estos valores son la fuente de verdad — Tabular List, "
        "Trends y Reports los respetan. En el siguiente paso podrás ajustar "
        "sensor por sensor si alguno tiene umbrales distintos al estándar."
    )

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("### Desplazamiento (proximidad)")
        state["displacement_unit"] = st.selectbox(
            "Unidad",
            options=["mil pp", "µm pp"],
            index=["mil pp", "µm pp"].index(
                state.get("displacement_unit", "mil pp")
            ),
            key="wiz_disp_unit",
        )
        state["proximity_alarm_mil_pp"] = st.number_input(
            f"Alarm (mil pp)",
            min_value=0.1, max_value=50.0,
            value=float(state.get("proximity_alarm_mil_pp", 4.0)),
            step=0.5,
            key="wiz_prox_alarm",
        )
        state["proximity_danger_mil_pp"] = st.number_input(
            "Danger (mil pp)",
            min_value=0.1, max_value=80.0,
            value=float(state.get("proximity_danger_mil_pp", 6.0)),
            step=0.5,
            key="wiz_prox_danger",
        )
        if state["displacement_unit"] == "µm pp":
            st.caption("(Equivale a {:.0f} / {:.0f} µm pp)".format(
                state["proximity_alarm_mil_pp"] * 25.4,
                state["proximity_danger_mil_pp"] * 25.4,
            ))

    with col_b:
        st.markdown("### Velocidad")
        state["velocity_unit"] = st.selectbox(
            "Unidad",
            options=["mm/s pk", "mm/s rms", "in/s pk"],
            index=["mm/s pk", "mm/s rms", "in/s pk"].index(
                state.get("velocity_unit", "mm/s pk")
            ),
            key="wiz_vel_unit",
        )
        state["velocity_alarm_mm_s"] = st.number_input(
            "Alarm (mm/s)",
            min_value=0.1, max_value=50.0,
            value=float(state.get("velocity_alarm_mm_s", 4.5)),
            step=0.5,
            key="wiz_vel_alarm",
        )
        state["velocity_danger_mm_s"] = st.number_input(
            "Danger (mm/s)",
            min_value=0.1, max_value=80.0,
            value=float(state.get("velocity_danger_mm_s", 11.2)),
            step=0.5,
            key="wiz_vel_danger",
        )

    with col_c:
        st.markdown("### Aceleración")
        state["acceleration_unit"] = st.selectbox(
            "Unidad",
            options=["g pk", "g rms", "m/s² rms"],
            index=["g pk", "g rms", "m/s² rms"].index(
                state.get("acceleration_unit", "g pk")
            ),
            key="wiz_acc_unit",
        )
        state["accel_alarm_g"] = st.number_input(
            "Alarm (g)",
            min_value=0.1, max_value=50.0,
            value=float(state.get("accel_alarm_g", 4.5)),
            step=0.5,
            key="wiz_acc_alarm",
        )
        state["accel_danger_g"] = st.number_input(
            "Danger (g)",
            min_value=0.1, max_value=100.0,
            value=float(state.get("accel_danger_g", 9.0)),
            step=0.5,
            key="wiz_acc_danger",
        )

    st.info(
        "💡 Si la plantilla LATAM tiene una norma ISO recomendada, los "
        "setpoints ya vienen ajustados. Podés sobreescribirlos arriba."
    )

    col_nav = st.columns([1, 2, 1])
    with col_nav[0]:
        if st.button("← Atrás", use_container_width=True, key="wiz_step4_prev"):
            _go_prev()
            st.rerun()
    with col_nav[2]:
        if st.button("Siguiente →", type="primary", use_container_width=True,
                     key="wiz_step4_next"):
            # Ciclo 23.6 — al pasar de Unidades & setpoints (nuevo paso 4) al
            # editor de sensores (nuevo paso 5):
            #   1) Si todavía no hay sensores_override, construirlo con
            #      _build_full_sensor_map (que ya lee las unidades / setpoints
            #      de state).
            #   2) Si ya existe (usuario volvió desde paso 5 a re-ajustar
            #      setpoints), recorrer la lista actual y propagar las
            #      nuevas unidades + alarm/danger por familia, preservando
            #      los demás campos (lados, ángulos, csv_match_pattern, etc.).
            sensors_list = state.get("sensors_override") or []
            if not sensors_list:
                sensors_list = _build_full_sensor_map(state)

            disp_unit = state.get("displacement_unit", "mil pp")
            vel_unit = state.get("velocity_unit", "mm/s pk")
            acc_unit = state.get("acceleration_unit", "g pk")
            prox_alarm = float(state.get("proximity_alarm_mil_pp", 4.0))
            prox_danger = float(state.get("proximity_danger_mil_pp", 6.0))
            vel_alarm = float(state.get("velocity_alarm_mm_s", 4.5))
            vel_danger = float(state.get("velocity_danger_mm_s", 11.2))
            acc_alarm = float(state.get("accel_alarm_g", 4.5))
            acc_danger = float(state.get("accel_danger_g", 9.0))

            for s in sensors_list:
                stype = (s.get("sensor_type") or "").lower()
                if stype == "proximity":
                    s["unit_native"] = disp_unit
                    s["alarm"] = prox_alarm
                    s["danger"] = prox_danger
                elif stype in ("velocity", "velometer"):
                    s["unit_native"] = vel_unit
                    s["alarm"] = vel_alarm
                    s["danger"] = vel_danger
                elif stype == "accelerometer":
                    s["unit_native"] = acc_unit
                    s["alarm"] = acc_alarm
                    s["danger"] = acc_danger
                # keyphasor: no toca setpoints (es referencia, no medición)
            state["sensors_override"] = sensors_list

            _go_next()
            st.rerun()


# =============================================================
# PASO 5 — Editor de sensores generados (orden corregido Ciclo 23.6)
# Sensores ya vienen con unidades y setpoints del paso 4. Acá el
# usuario puede ajustar sensor por sensor (lados, ángulos, csv match
# pattern). Si cambia setpoints/unidades por sensor, esos overrides
# se respetan (no se sobreescriben con los del paso 4).
# =============================================================

elif current == 5:
    st.subheader("Paso 5 · Ajustar sensores generados")
    st.caption(
        "Sensores generados con las unidades y setpoints del paso anterior. "
        "Acá podés ajustar individualmente lados, ángulos, tipos, unidades y "
        "patrones de match al CSV. Si un sensor tiene umbrales distintos al "
        "estándar (ej. un canal calibrado distinto), editalo en la tabla."
    )

    sensors_override = state.get("sensors_override")
    if not sensors_override:
        st.warning("No hay sensores generados. Volvé al paso 4 (unidades) y dale Siguiente.")
    else:
        import pandas as pd

        # Ciclo 23.8 — Editor visual click-to-place para TODAS las categorías.
        # Antes solo aparecía para reciprocating_compressor, dejando a las
        # demás máquinas sin posicionamiento visual.
        tab_visual, tab_table = st.tabs([
            "🎨 Editor visual (click para reposicionar)",
            "📋 Tabla de sensores",
        ])
        with tab_visual:
            is_recip = state.get("category") == "reciprocating_compressor"
            if is_recip:
                st.caption(
                    "Hacé click sobre la imagen para mover el sensor seleccionado a "
                    "esa posición. El esquema se genera según N cilindros + N "
                    "cojinetes del motor."
                )
            else:
                st.caption(
                    "Hacé click sobre la imagen para mover el sensor seleccionado a "
                    "esa posición. El esquema muestra el tren acoplado driver + "
                    "acople + driven con los cojinetes numerados."
                )
            _render_visual_editor(state, sensors_override)
        with tab_table:
            _render_sensors_table_editor(state, sensors_override)

        col_actions = st.columns([1, 1, 2])
        with col_actions[0]:
            if st.button("🔄 Regenerar desde paso 4",
                         help="Descarta cambios y reconstruye el mapa con las unidades del paso 4",
                         key="wiz_regen_sensors"):
                state["sensors_override"] = _build_full_sensor_map(state)
                # Aplicar los setpoints del paso 4 al nuevo mapa
                disp_unit = state.get("displacement_unit", "mil pp")
                vel_unit = state.get("velocity_unit", "mm/s pk")
                acc_unit = state.get("acceleration_unit", "g pk")
                prox_alarm = float(state.get("proximity_alarm_mil_pp", 4.0))
                prox_danger = float(state.get("proximity_danger_mil_pp", 6.0))
                vel_alarm = float(state.get("velocity_alarm_mm_s", 4.5))
                vel_danger = float(state.get("velocity_danger_mm_s", 11.2))
                acc_alarm = float(state.get("accel_alarm_g", 4.5))
                acc_danger = float(state.get("accel_danger_g", 9.0))
                for s in state["sensors_override"]:
                    stype = (s.get("sensor_type") or "").lower()
                    if stype == "proximity":
                        s["unit_native"] = disp_unit
                        s["alarm"] = prox_alarm
                        s["danger"] = prox_danger
                    elif stype in ("velocity", "velometer"):
                        s["unit_native"] = vel_unit
                        s["alarm"] = vel_alarm
                        s["danger"] = vel_danger
                    elif stype == "accelerometer":
                        s["unit_native"] = acc_unit
                        s["alarm"] = acc_alarm
                        s["danger"] = acc_danger
                st.rerun()

    col_nav = st.columns([1, 2, 1])
    with col_nav[0]:
        if st.button("← Atrás", use_container_width=True, key="wiz_step5_prev"):
            _go_prev()
            st.rerun()
    with col_nav[2]:
        if st.button("Siguiente →", type="primary", use_container_width=True,
                     key="wiz_step5_next"):
            # Persistir edits del data_editor antes de avanzar.
            edited_df = state.get("_wizard_table_edited")
            if edited_df is not None and state.get("sensors_override"):
                edited_records = edited_df.to_dict(orient="records")
                originals_by_idx = {i: s for i, s in enumerate(state["sensors_override"])}
                final_sensors = []
                for i, row in enumerate(edited_records):
                    base = dict(originals_by_idx.get(i, {}))
                    base.update({
                        "plane_label": row.get("plane_label", ""),
                        "side": row.get("side", "L"),
                        "angle_deg": float(row.get("angle_deg", 45.0)),
                        "direction": row.get("direction", "Y"),
                        "sensor_type": row.get("sensor_type", "proximity"),
                        "unit_native": row.get("unit_native", ""),
                        "alarm": float(row.get("alarm", 0.0)),
                        "danger": float(row.get("danger", 0.0)),
                        "csv_match_pattern": row.get("csv_match_pattern", ""),
                    })
                    final_sensors.append(base)
                state["sensors_override"] = final_sensors
            _go_next()
            st.rerun()


# =============================================================
# PASO 6 — Datos del activo + crear
# =============================================================

elif current == 6:
    st.subheader("Paso 6 · Datos del activo")
    st.caption("Última info y creamos.")

    col_l, col_r = st.columns(2)

    with col_l:
        suggested_id = state.get("instance_id") or _slug_default(state)
        state["instance_id"] = st.text_input(
            "ID único del activo (slug)",
            value=suggested_id,
            help="Solo letras, números, guiones y guiones bajos. Ej: 'tes1', 'parex_c200c'.",
            key="wiz_instance_id",
        )
        state["tag"] = st.text_input(
            "Tag interno del cliente",
            value=state.get("tag", ""),
            placeholder="Ej: TES1, C-200-C, SGT300A",
            key="wiz_tag",
        )
        state["client"] = st.text_input(
            "Cliente",
            value=state.get("client", ""),
            placeholder="Ej: Ecopetrol — Magnex, Parex",
            key="wiz_client",
        )
        state["site"] = st.text_input(
            "Sitio / planta",
            value=state.get("site", ""),
            placeholder="Ej: Termosuria Villavicencio",
            key="wiz_site",
        )
        state["location"] = st.text_input(
            "Ubicación física (opcional)",
            value=state.get("location", ""),
            placeholder="Ej: Planta La Belleza, Plato, Magdalena",
            key="wiz_location",
        )

    with col_r:
        state["nominal_rpm"] = st.number_input(
            "RPM nominal",
            min_value=0.0,
            value=float(state.get("nominal_rpm", 0.0) or 0.0),
            step=100.0,
            key="wiz_rpm",
        )
        state["nominal_power_mw"] = st.number_input(
            "Potencia nominal (MW)",
            min_value=0.0,
            value=float(state.get("nominal_power_mw", 0.0) or 0.0),
            step=1.0,
            key="wiz_power",
        )
        # Profile legacy (necesario para create_instance)
        profile_options = sorted(PROFILES.keys())
        suggested_profile = state.get("profile_key", "custom_manual")
        if suggested_profile not in profile_options:
            suggested_profile = "custom_manual" if "custom_manual" in profile_options else profile_options[0]
        state["profile_key"] = st.selectbox(
            "Profile (familia técnica)",
            options=profile_options,
            index=profile_options.index(suggested_profile),
            format_func=lambda k: PROFILES[k].label,
            key="wiz_profile",
        )
        state["iso_norm_code"] = st.text_input(
            "Norma ISO recomendada (opcional)",
            value=state.get("iso_norm_code", ""),
            help="Ej: ISO_20816_2 o ISO_10816_7",
            key="wiz_iso",
        )
        state["notes"] = st.text_area(
            "Notas técnicas",
            value=state.get("notes", ""),
            height=100,
            key="wiz_notes",
        )

    st.markdown("---")
    st.markdown("### Resumen")

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Categoría", state.get("category", "—"))
    s2.metric("Cojinetes", f"D:{state['driver_planes']} + A:{state['driven_planes']}")
    s3.metric("Driver instr.", state["driver_instrumentation"].replace("_", " "))
    s4.metric("Driven instr.", state["driven_instrumentation"].replace("_", " "))

    s5, s6, s7, s8 = st.columns(4)
    s5.metric("Despl.", state["displacement_unit"])
    s6.metric("Vel.", state["velocity_unit"])
    s7.metric("Acel.", state["acceleration_unit"])
    s8.metric("Keyphasor", "Sí" if state["include_keyphasor"] else "No")

    col_nav = st.columns([1, 2, 1])
    with col_nav[0]:
        if st.button("← Atrás", use_container_width=True, key="wiz_step6_prev"):
            _go_prev()
            st.rerun()
    with col_nav[2]:
        if st.button("✅ Crear activo", type="primary", use_container_width=True,
                     key="wiz_create"):
            try:
                _execute_creation(state)
                # Globitos PRIMERO (antes del reset y del success).
                # Si reseteamos antes, st.balloons a veces queda "stuck"
                # porque el rerun lo cancela.
                st.balloons()
                created_id = state['instance_id']
                _reset_wizard()
                st.success(
                    f"✅ Activo '{created_id}' creado correctamente. "
                    f"Sensores generados automáticamente. "
                    f"Lo encontrás en Machinery Library."
                )
            except Exception as e:
                st.error(f"❌ Error al crear el activo: {e}")


# =============================================================
# Footer — botón cancelar / debug
# =============================================================

st.markdown("---")
col_foot = st.columns([3, 1])
with col_foot[1]:
    if st.button("🔄 Reiniciar wizard", key="wiz_reset_btn"):
        _reset_wizard()
        st.rerun()
