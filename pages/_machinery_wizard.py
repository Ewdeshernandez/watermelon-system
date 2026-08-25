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

# v3.31.243 — set_page_config primero, sino el sidebar pierde estilos.
import streamlit as st
st.set_page_config(
    page_title="Create asset · Wizard — Watermelon",
    page_icon="🧙",
    layout="wide",
)

from core.auth import require_login, render_user_menu, require_role

require_login()
render_user_menu()
require_role(allowed_roles=("admin", "specialist"))

import re
from datetime import datetime
from typing import Any, Dict, List, Optional

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
        raise ValueError("The asset ID is required.")
    if get_instance(inst_id_raw) is not None:
        raise ValueError(f"An asset with ID '{inst_id_raw}' already exists.")

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
        st.markdown(f"**Visual icon** ({role})")
        # Construir lista plana ordenada por categoría
        categories = sorted(by_cat.keys())

        # Dropdown 1: categoría
        cat_options = ["— No icon —"] + categories
        current_key = state.get(state_key, "")
        current_meta = get_asset_meta(current_key) if current_key else None
        current_cat = current_meta.get("category") if current_meta else "— No icon —"
        if current_cat not in cat_options:
            current_cat = "— No icon —"

        cat_pick = st.selectbox(
            f"Category",
            options=cat_options,
            index=cat_options.index(current_cat),
            key=f"wiz_icon_cat_{role}",
        )

        if cat_pick == "— No icon —":
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
            f"Model",
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


def _gearbox_compose_kwargs(state: Dict[str, Any]) -> Dict[str, Any]:
    """Deriva los kwargs de gearbox para compose_train desde el estado del
    wizard. Si no hay gearbox, devuelve {} (compose_train queda idéntico)."""
    if not state.get("include_gearbox"):
        return {}
    label = (state.get("gearbox_type") or "").strip() or "Gearbox / reductor"
    return {
        "gearbox_key": state.get("gearbox_icon_key") or "gearbox_inline",
        "gearbox_label": label,
        "coupling2": state.get("coupling_class", "flexible"),
    }


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
            **_gearbox_compose_kwargs(state),
        )
        st.markdown("**Coupled train preview**")
        st.markdown(
            f'<div style="background:#ffffff;border:1px solid #e2e8f0;'
            f'border-radius:10px;padding:14px;">{svg}</div>',
            unsafe_allow_html=True,
        )
    except Exception as e:
        st.caption(f"(Preview not available: {e})")


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
            "plane_label": st.column_config.TextColumn("Plane", width="medium"),
            "side": st.column_config.SelectboxColumn(
                "Side", options=["L", "R", "T", "B", ""], width="small",
            ),
            "angle_deg": st.column_config.NumberColumn(
                "Angle (°)", min_value=0.0, max_value=360.0, step=15.0,
                width="small",
            ),
            "direction": st.column_config.SelectboxColumn(
                "Direction", options=["X", "Y", "Z", "RAD", "AX", ""],
                width="small",
            ),
            "sensor_type": st.column_config.SelectboxColumn(
                "Type",
                options=["proximity", "accelerometer", "velometer", "keyphasor"],
                width="medium",
            ),
            "unit_native": st.column_config.TextColumn("Unit", width="small"),
            "alarm": st.column_config.NumberColumn("Alarm", width="small"),
            "danger": st.column_config.NumberColumn("Danger", width="small"),
            "csv_match_pattern": st.column_config.TextColumn(
                "CSV pattern (optional)", width="medium",
            ),
        },
        # La key incluye una "generación": al Regenerar se incrementa y el
        # data_editor arranca de cero (descarta added_rows/edited_rows fantasma).
        key=f"wiz_sensors_editor_{state.get('_sensor_editor_gen', 0)}",
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


# =====================================================================
# Ciclo 23.148 — Editor de configuración modal por sensor
# =====================================================================
# Expander opcional dentro del editor visual del wizard que permite al
# analista capturar los 4 campos requeridos para análisis modal (EMA/OMA):
#
#   · sensitivity_mv_per_eu  (mV/EU) — sensibilidad del transductor
#   · coupling               (IEPE/AC/DC) — modo de acoplamiento al módulo de adquisición
#   · position_3d            ([x, y, z] metros) — pos. en frame del activo
#   · dof_direction          ([dx, dy, dz] unitario) — eje sensible
#
# Defaults inteligentes por sensor_type:
#   · accelerometer (IEPE): 100 mV/g
#   · proximity     (Bently)  : 200 mV/mil · AC
#   · velocity      (VS)      : 100 mV/(in/s) · IEPE
#
# Si el analista no toca el expander, los campos quedan vacíos y el
# sistema modal SKIPS el sensor (no se rompe nada).
# =====================================================================

def _render_modal_config_expander(
    s: Dict[str, Any],
    idx: int,
    sensor_type: str,
) -> None:
    """Editor inline de la configuración modal de un sensor."""
    from core.sensor_map import (
        _DEFAULT_SENSITIVITY_BY_TYPE,
        _DEFAULT_COUPLING_BY_TYPE,
    )

    _has_modal = bool(
        s.get("sensitivity_mv_per_eu") is not None
        or s.get("coupling")
        or s.get("position_3d")
        or s.get("dof_direction")
    )
    _expander_title = (
        "⚙ Modal configuration · captured ✓"
        if _has_modal
        else "⚙ Modal configuration (optional · for EMA / OMA)"
    )

    with st.expander(_expander_title, expanded=False):
        st.caption(
            "Required only if you are going to use this sensor in the Modal Analysis module. "
            "If you leave it empty, the sensor keeps working for Live Monitoring, "
            "Spectrum and Orbit as always."
        )

        # Defaults
        _sens_default = _DEFAULT_SENSITIVITY_BY_TYPE.get(sensor_type.lower(), 100.0)
        _coup_default = _DEFAULT_COUPLING_BY_TYPE.get(sensor_type.lower(), "AC")

        # Sensitivity + coupling
        mc1, mc2 = st.columns(2)
        with mc1:
            _cur_sens = s.get("sensitivity_mv_per_eu")
            sens_val = st.number_input(
                "Sensitivity (mV/EU)",
                min_value=0.0,
                max_value=10000.0,
                value=float(_cur_sens) if _cur_sens is not None else float(_sens_default),
                step=10.0,
                key=f"wiz_modal_sens_{idx}",
                help="IEPE accelerometer: 100 mV/g · Proximity probe: 200 mV/mil · Modal hammer: 2.4 mV/N",
            )
            # Solo guardamos si el valor difiere significativamente de None;
            # si el analista lo dejó en el default y NO completó posición 3D,
            # asumimos que no quiere configurar modal (no contaminamos data)
            if sens_val > 0:
                s["sensitivity_mv_per_eu"] = float(sens_val)

        with mc2:
            _coup_opts = ["", "IEPE", "AC", "DC"]
            _cur_coup = (s.get("coupling") or _coup_default).upper()
            _coup_idx = _coup_opts.index(_cur_coup) if _cur_coup in _coup_opts else 0
            coup_pick = st.selectbox(
                "Coupling to the system",
                _coup_opts,
                index=_coup_idx,
                key=f"wiz_modal_coup_{idx}",
                format_func=lambda x: x or "— none —",
                help="IEPE: accel / hammer · AC: Bently with DC blocker · DC: direct measurement",
            )
            if coup_pick:
                s["coupling"] = coup_pick

        # Position 3D
        st.markdown("**Sensor 3D position** — origin: crankcase center · X axial driver→driven")
        _pos = s.get("position_3d") or [0.0, 0.0, 0.0]
        pc1, pc2, pc3 = st.columns(3)
        with pc1:
            px = st.number_input("x (m)", value=float(_pos[0]),
                                 step=0.1, key=f"wiz_modal_px_{idx}")
        with pc2:
            py = st.number_input("y (m)", value=float(_pos[1]),
                                 step=0.1, key=f"wiz_modal_py_{idx}")
        with pc3:
            pz = st.number_input("z (m)", value=float(_pos[2]),
                                 step=0.1, key=f"wiz_modal_pz_{idx}")
        # Solo guardar si al menos una coord es no-cero (gesto explícito del usuario)
        if abs(px) + abs(py) + abs(pz) > 1e-9:
            s["position_3d"] = [float(px), float(py), float(pz)]

        # DOF direction
        st.markdown("**DOF direction** — unit vector of the sensitive axis")
        _dof = s.get("dof_direction") or [0.0, 1.0, 0.0]  # default Y
        dc1, dc2, dc3 = st.columns(3)
        with dc1:
            dx = st.number_input("dx", value=float(_dof[0]), step=0.1,
                                 min_value=-1.0, max_value=1.0,
                                 key=f"wiz_modal_dx_{idx}")
        with dc2:
            dy = st.number_input("dy", value=float(_dof[1]), step=0.1,
                                 min_value=-1.0, max_value=1.0,
                                 key=f"wiz_modal_dy_{idx}")
        with dc3:
            dz = st.number_input("dz", value=float(_dof[2]), step=0.1,
                                 min_value=-1.0, max_value=1.0,
                                 key=f"wiz_modal_dz_{idx}")
        if abs(dx) + abs(dy) + abs(dz) > 1e-9:
            s["dof_direction"] = [float(dx), float(dy), float(dz)]

        # Hint de inferencia automática por convención de naming
        _plbl = str(s.get("plane_label", "")).upper()
        if not s.get("dof_direction"):
            if "YA" in _plbl or "YV" in _plbl:
                st.caption("💡 Convention: sensor with 'Y' → `dof_direction` typically [0, 1, 0]")
            elif "XA" in _plbl or "XV" in _plbl:
                st.caption("💡 Convention: sensor with 'X' → `dof_direction` typically [1, 0, 0]")


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
        from core.sensor_map import gearbox_overlay_anchor as _gbx_ov
        for s in sensors:
            side = (s.get("icon_side") or "").strip()
            anchor = (s.get("icon_anchor") or "").strip()
            if not side or not anchor:
                _gov = _gbx_ov(s)
                if _gov:
                    side, anchor = _gov
                else:
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
                **_gearbox_compose_kwargs(state),
            )
            st.markdown(
                f'<div style="background:#ffffff;border:1px solid #e2e8f0;'
                f'border-radius:10px;padding:14px;">{svg}</div>',
                unsafe_allow_html=True,
            )
        except Exception as e:
            st.warning(f"Could not generate the SVG library: {e}")

        n_mapped = len(s_for_svg)
        n_total = len(sensors)
        st.caption(
            f"📍 **{n_mapped} of {n_total}** sensors assigned to an anchor. "
            "Unassigned ones do not appear in the preview or in Live Monitoring."
        )
        if st.button("🪄 Auto-assign all by heuristic (Bently / API 670)",
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
        st.markdown("**Assignment of each sensor to a physical bearing**")
        st.caption(
            "Side = which machine in the train (driver / driven). "
            "Anchor = which physical bearing on the icon (DE, NDE, TRF, CRF...)."
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

                # Ciclo 23.148 — Configuración modal opcional por sensor.
                # Si está vacía, el sistema modal SKIP este sensor (no se rompe nada).
                # Si está completa, queda disponible para EMA / OMA / 3D animación.
                _render_modal_config_expander(s, idx, stype)

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
        st.warning("Pillow is not available — I cannot generate the schematic.")
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
            st.caption("(streamlit_image_coordinates not available — display only)")

    with col_list:
        st.markdown("**Sensors** (click to select and reposition)")
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
        if st.button("❌ Deselect", key="wiz_deselect_sensor"):
            state["_wiz_selected_sensor_idx"] = -1
            st.rerun()


# =============================================================
# Header del wizard
# =============================================================

st.title("Create asset · Guided wizard")
st.caption(
    "Set up a new machine in 5 steps. The system automatically builds "
    "the sensor map and setpoints according to the applicable ISO/API standards."
)

state = _wizard_state()
current = state["step"]

# Stepper visual.
# Ciclo 23.6 — orden corregido: primero unidades & setpoints, después editor
# de sensores (así el editor ya muestra los setpoints OEM elegidos por el
# usuario y no hay que ir-volver entre pasos).
step_labels = [
    "1 · Type",
    "2 · Train",
    "3 · Instrumentation",
    "4 · Units & setpoints",
    "5 · Edit sensors",
    "6 · Asset data",
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
    st.subheader("Step 1 · What type of machine will you monitor?")

    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.markdown("**A) Start from scratch**")
        st.caption("Set up the whole machine manually.")
        category_options = {
            "turbomachinery": "🌀 Turbomachine (turbine/turbogenerator)",
            "centrifugal_compressor": "⚙️ Centrifugal compressor",
            "reciprocating_compressor": "🔄 Reciprocating compressor",
            "centrifugal_pump": "💧 Centrifugal pump",
            "electric_motor": "⚡ Electric motor",
            "fan_blower": "🌬️ Fan / blower",
            "custom": "🛠️ Other / custom",
        }
        cat_pick = st.radio(
            "Category",
            options=list(category_options.keys()),
            format_func=lambda k: category_options[k],
            index=list(category_options.keys()).index(state["category"])
                  if state["category"] in category_options else 0,
            key="wiz_step1_cat",
        )

    with col_b:
        st.markdown("**B) Use a LATAM template**")
        st.caption(
            "Preloads 20+ machines common in O&G/power generation with their "
            "typical bearings, standards and recommended sensor schemes."
        )
        cat_filter = state.get("template_filter_cat", "All")
        cats = ["All"] + list_categories()
        cat_filter = st.selectbox(
            "Filter by category",
            options=cats,
            index=cats.index(cat_filter) if cat_filter in cats else 0,
            key="wiz_template_cat",
        )
        if cat_filter == "All":
            templates = list_templates()
        else:
            templates = list_templates_by_category(cat_filter)
        template_options = {"": "— No template —"}
        for t in templates:
            template_options[t.id] = t.label
        sel_template = st.selectbox(
            "Template",
            options=list(template_options.keys()),
            format_func=lambda k: template_options[k],
            index=list(template_options.keys()).index(state["template_id"])
                  if state["template_id"] in template_options else 0,
            key="wiz_template_id",
        )

    # Botón Siguiente
    col_nav = st.columns([3, 1])
    with col_nav[1]:
        if st.button("Next →", type="primary", use_container_width=True,
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
    st.subheader("Step 2 · Mechanical train")
    st.caption("How many machines are coupled, what bearing type, what coupling.")

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("### Driver (driving)")
        # Selector visual de icono (asset library System1-style)
        _render_icon_picker(state, role="driver", state_key="driver_icon_key", column=col_l)
        state["driver_type"] = st.text_input(
            "Type / description",
            value=state.get("driver_type") or state.get("driver_model") or "",
            placeholder="e.g. GE LM6000, Solar Mars 100, 4-pole motor",
            key="wiz_driver_type",
        )
        state["driver_planes"] = st.number_input(
            "Number of bearings (measurement planes)",
            min_value=1, max_value=6,
            value=int(state.get("driver_planes", 2) or 2),
            key="wiz_driver_planes",
        )
        state["driver_bearing_kind"] = st.radio(
            "Bearing type",
            options=["plain", "rolling"],
            format_func=lambda k: {"plain": "🧱 Plain bearing (fluid film)",
                                   "rolling": "⚙️ Rolling bearing (rolling element)"}[k],
            index=["plain", "rolling"].index(state.get("driver_bearing_kind", "plain")),
            key="wiz_driver_bearing_kind",
        )
        if state["driver_bearing_kind"] == "rolling":
            state["driver_bearing_model"] = st.text_input(
                "Typical bearing model (optional)",
                value=state.get("driver_bearing_model", ""),
                placeholder="e.g. SKF 6319, NU 220",
                key="wiz_driver_bearing_model",
            )

    is_recip = state.get("category") == "reciprocating_compressor"

    with col_r:
        if is_recip:
            st.markdown("### Driven · Reciprocating compressor")
            # Selector visual de icono (asset library System1-style)
            _render_icon_picker(state, role="driven", state_key="driven_icon_key", column=col_r)
            state["driven_type"] = st.text_input(
                "Model / manufacturer",
                value=state.get("driven_type") or state.get("driven_model") or "",
                placeholder="e.g. Ariel KBK/4, Burckhardt Process, Dresser-Rand HOS",
                key="wiz_driven_type",
            )
            state["cylinders_count"] = st.select_slider(
                "Number of cylinders",
                options=[2, 4, 6, 8],
                value=int(state.get("cylinders_count", 4) or 4),
                help="1X frequency = RPM/60; each cylinder adds harmonics depending "
                     "on configuration (single/double-acting, stages).",
                key="wiz_recip_cylinders",
            )
            state["include_rod_drop"] = st.checkbox(
                "Include rod drop sensors (1 per cylinder)",
                value=bool(state.get("include_rod_drop", True)),
                help="Packing/rider band wear detection. API 618 standard.",
                key="wiz_recip_rod_drop",
            )
            # En reciprocantes el "driven_planes" representa frame planes (top + side)
            state["driven_planes"] = 2
            state["driven_bearing_kind"] = "plain"
            st.caption(
                "ℹ️ In reciprocating machines the sensors go on the compressor frame "
                "and on each crosshead, not on radial bearings. ISO 20816-8."
            )
        else:
            st.markdown("### Driven (driven machine)")
            # Selector visual de icono (asset library System1-style)
            _render_icon_picker(state, role="driven", state_key="driven_icon_key", column=col_r)
            state["driven_type"] = st.text_input(
                "Type / description",
                value=state.get("driven_type") or state.get("driven_model") or "",
                placeholder="e.g. Brush 54MW generator, Ariel KBK/4 compressor",
                key="wiz_driven_type",
            )
            state["driven_planes"] = st.number_input(
                "Number of bearings (measurement planes)",
                min_value=1, max_value=6,
                value=int(state.get("driven_planes", 2) or 2),
                key="wiz_driven_planes",
            )
            state["driven_bearing_kind"] = st.radio(
                "Bearing type",
                options=["plain", "rolling"],
                format_func=lambda k: {"plain": "🧱 Plain bearing (fluid film)",
                                       "rolling": "⚙️ Rolling bearing (rolling element)"}[k],
                index=["plain", "rolling"].index(state.get("driven_bearing_kind", "rolling")),
                key="wiz_driven_bearing_kind",
            )
            if state["driven_bearing_kind"] == "rolling":
                state["driven_bearing_model"] = st.text_input(
                    "Typical bearing model (optional)",
                    value=state.get("driven_bearing_model", ""),
                    placeholder="e.g. SKF 6319, NU 220",
                    key="wiz_driven_bearing_model",
                )

    st.markdown("---")

    # ===== Gearbox opcional (Ciclo 21.2) =====
    state["include_gearbox"] = st.checkbox(
        "⚙️ Include gearbox / step-up between driver and driven",
        value=bool(state.get("include_gearbox", False)),
        help="Enable if there is an intermediate reduction/step-up gearbox "
             "(e.g. turbine + gearbox + generator, motor + gearbox + pump).",
        key="wiz_include_gearbox",
    )

    if state["include_gearbox"]:
        with st.container(border=True):
            st.markdown("### Gearbox")
            gc1, gc2 = st.columns(2)
            with gc1:
                state["gearbox_type"] = st.text_input(
                    "Type / manufacturer (optional)",
                    value=state.get("gearbox_type", ""),
                    placeholder="e.g. Lufkin, Voith, Flender",
                    key="wiz_gearbox_type",
                )
                state["gearbox_planes"] = st.number_input(
                    "Gearbox bearings",
                    min_value=2, max_value=6,
                    value=int(state.get("gearbox_planes", 2)),
                    help="Typically 2 (HSS = high speed shaft + LSS = low speed shaft). "
                         "Up to 4 if it has intermediate pinions.",
                    key="wiz_gearbox_planes",
                )
            with gc2:
                state["gearbox_bearing_kind"] = st.radio(
                    "Gearbox bearing type",
                    options=["plain", "rolling"],
                    format_func=lambda k: {"plain": "🧱 Plain",
                                           "rolling": "⚙️ Rolling"}[k],
                    index=["plain", "rolling"].index(
                        state.get("gearbox_bearing_kind", "rolling")
                    ),
                    horizontal=True,
                    key="wiz_gearbox_bearing_kind",
                )
                if state["gearbox_bearing_kind"] == "rolling":
                    state["gearbox_bearing_model"] = st.text_input(
                        "Typical bearing model (optional)",
                        value=state.get("gearbox_bearing_model", ""),
                        placeholder="e.g. SKF NU 220, Timken 32220",
                        key="wiz_gearbox_bearing_model",
                    )

    st.markdown("### Coupling")
    state["coupling_class"] = st.radio(
        "Coupling type",
        options=["rigid", "flexible", "fluid"],
        format_func=lambda k: {
            "rigid": "🔗 Rigid",
            "flexible": "🌀 Flexible (gear/disk/diaphragm)",
            "fluid": "💧 Hydrodynamic / fluid coupling",
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
            "💡 Pick a visual icon for the driver and driven above to see the "
            "full coupled-train preview (System1 / AMS style)."
        )

    col_nav = st.columns([1, 2, 1])
    with col_nav[0]:
        if st.button("← Back", use_container_width=True, key="wiz_step2_prev"):
            _go_prev()
            st.rerun()
    with col_nav[2]:
        if st.button("Next →", type="primary", use_container_width=True,
                     key="wiz_step2_next"):
            _go_next()
            st.rerun()


# =============================================================
# PASO 3 — Instrumentación
# =============================================================

elif current == 3:
    st.subheader("Step 3 · Instrumentation")
    st.caption(
        "What type of sensors are on each machine. The system builds the full "
        "map following API 670 / ISO 20816 practices."
    )

    instrum_options = {
        "proximity_xy": "🎯 Proximity XY (non-contact, X-Y pair at 45°)",
        "axial_accel": "📍 Casing accelerometer (1 radial top per plane)",
        "accel_plus_velocity": "📊 Accelerometer + Velometer (casing, full)",
    }

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("### Driver — instrumentation")
        state["driver_instrumentation"] = st.radio(
            "Sensor type on the driver",
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
            st.markdown("### Driven · Reciprocating compressor")
            st.info(
                "**API 618 / ISO 20816-8 instrumentation:**\n"
                "- 1 frame top velometer (radial)\n"
                "- 1 frame side velometer (lateral)\n"
                f"- 1 crosshead accelerometer × {state.get('cylinders_count', 4)} cylinders\n"
                + (f"- 1 rod drop × {state.get('cylinders_count', 4)} cylinders\n"
                   if state.get("include_rod_drop", True) else "")
            )
            state["driven_instrumentation"] = "reciprocating"
        else:
            st.markdown("### Driven — instrumentation")
            state["driven_instrumentation"] = st.radio(
                "Sensor type on the driven",
                options=list(instrum_options.keys()),
                format_func=lambda k: instrum_options[k],
                index=list(instrum_options.keys()).index(
                    state.get("driven_instrumentation", "proximity_xy")
                ) if state.get("driven_instrumentation") in instrum_options else 0,
                key="wiz_driven_instrum",
            )

    # ===== Instrumentación del gearbox (Ciclo 21.2) =====
    if state.get("include_gearbox"):
        st.markdown("### Gearbox — instrumentation")
        state["gearbox_instrumentation"] = st.radio(
            "Sensor type on the gearbox",
            options=list(instrum_options.keys()),
            format_func=lambda k: instrum_options[k],
            index=list(instrum_options.keys()).index(
                state.get("gearbox_instrumentation", "axial_accel")
            ),
            help="The most common on industrial gearboxes is axial_accel "
                 "(casing accelerometer per bearing).",
            key="wiz_gearbox_instrum",
        )

    st.markdown("---")
    st.markdown("### 1X reference (keyphasor)")
    state["include_keyphasor"] = st.checkbox(
        "Include keyphasor (1X phase sensor on the coupling)",
        value=bool(state.get("include_keyphasor", True)),
        help="Proximity sensor aimed at a mark on the shaft. "
             "Required for phase analysis, filtered orbit and Bode/Polar.",
        key="wiz_keyphasor",
    )

    st.markdown("### Channels per sensor")
    state["channels_per_sensor"] = st.select_slider(
        "Number of signals per sensor",
        options=[1, 2, 3],
        value=int(state.get("channels_per_sensor", 1) or 1),
        help="1 channel: RMS amplitude only. 2 channels: RMS + waveform. 3 channels: + spectrum.",
        key="wiz_channels",
    )

    # Preview rápido del mapa (incluye gearbox)
    with st.expander("👁️ Preview of the sensor map that will be generated",
                     expanded=True):
        try:
            preview = _build_full_sensor_map(state)
            st.markdown(f"**Total sensors:** {len(preview)}")
            for s in preview:
                plane = s.get("plane_label", f"plano {s.get('plane', '?')}")
                stype = s.get("sensor_type", "?")
                direction = s.get("direction", "?")
                st.markdown(f"- `{plane}` · {stype} · dir: {direction}")
        except Exception as e:
            st.warning(f"Could not preview: {e}")

    col_nav = st.columns([1, 2, 1])
    with col_nav[0]:
        if st.button("← Back", use_container_width=True, key="wiz_step3_prev"):
            _go_prev()
            st.rerun()
    with col_nav[2]:
        if st.button("Next →", type="primary", use_container_width=True,
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
    st.subheader("Step 4 · Units and setpoints")
    st.caption(
        "Define the units each sensor type reports in, and the alarm/danger "
        "levels. These values are the source of truth — Tabular List, "
        "Trends and Reports respect them. In the next step you can adjust "
        "sensor by sensor if any has thresholds different from the standard."
    )

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("### Displacement (proximity)")
        state["displacement_unit"] = st.selectbox(
            "Unit",
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
            st.caption("(Equivalent to {:.0f} / {:.0f} µm pp)".format(
                state["proximity_alarm_mil_pp"] * 25.4,
                state["proximity_danger_mil_pp"] * 25.4,
            ))

    with col_b:
        st.markdown("### Velocity")
        state["velocity_unit"] = st.selectbox(
            "Unit",
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
        st.markdown("### Acceleration")
        state["acceleration_unit"] = st.selectbox(
            "Unit",
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
        "💡 If the LATAM template has a recommended ISO standard, the "
        "setpoints come pre-adjusted. You can override them above."
    )

    col_nav = st.columns([1, 2, 1])
    with col_nav[0]:
        if st.button("← Back", use_container_width=True, key="wiz_step4_prev"):
            _go_prev()
            st.rerun()
    with col_nav[2]:
        if st.button("Next →", type="primary", use_container_width=True,
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
    st.subheader("Step 5 · Adjust generated sensors")
    st.caption(
        "Sensors generated with the units and setpoints from the previous step. "
        "Here you can individually adjust sides, angles, types, units and "
        "CSV match patterns. If a sensor has thresholds different from the "
        "standard (e.g. a differently calibrated channel), edit it in the table."
    )

    sensors_override = state.get("sensors_override")
    if not sensors_override:
        st.warning("No sensors generated. Go back to step 4 (units) and click Next.")
    else:
        import pandas as pd

        # Ciclo 23.8 — Editor visual click-to-place para TODAS las categorías.
        # Antes solo aparecía para reciprocating_compressor, dejando a las
        # demás máquinas sin posicionamiento visual.
        tab_visual, tab_table = st.tabs([
            "🎨 Visual editor (click to reposition)",
            "📋 Sensor table",
        ])
        with tab_visual:
            is_recip = state.get("category") == "reciprocating_compressor"
            if is_recip:
                st.caption(
                    "Click on the image to move the selected sensor to "
                    "that position. The schematic is generated from N cylinders + N "
                    "motor bearings."
                )
            else:
                st.caption(
                    "Click on the image to move the selected sensor to "
                    "that position. The schematic shows the coupled train driver + "
                    "coupling + driven with the bearings numbered."
                )
            _render_visual_editor(state, sensors_override)
        with tab_table:
            _render_sensors_table_editor(state, sensors_override)

        col_actions = st.columns([1, 1, 2])
        with col_actions[0]:
            if st.button("🔄 Regenerate from step 4",
                         help="Discards changes and rebuilds the map with the step 4 units",
                         key="wiz_regen_sensors"):
                # Fuerza un data_editor nuevo (limpia added_rows/edited_rows
                # que sobrevivían a la regeneración y metían filas fantasma).
                state["_sensor_editor_gen"] = state.get("_sensor_editor_gen", 0) + 1
                state.pop("_wizard_table_edited", None)
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
        if st.button("← Back", use_container_width=True, key="wiz_step5_prev"):
            _go_prev()
            st.rerun()
    with col_nav[2]:
        if st.button("Next →", type="primary", use_container_width=True,
                     key="wiz_step5_next"):
            # Persistir edits del data_editor antes de avanzar.
            edited_df = state.get("_wizard_table_edited")
            if edited_df is not None and state.get("sensors_override"):
                edited_records = edited_df.to_dict(orient="records")
                originals_by_idx = {i: s for i, s in enumerate(state["sensors_override"])}
                def _num(v, default):
                    """float() tolerante: None, NaN o '' → default. Evita el
                    TypeError 'float() ... NoneType' cuando el data_editor deja
                    una fila con celdas vacías (p. ej. la fila fantasma que
                    agrega num_rows='dynamic')."""
                    try:
                        if v is None:
                            return float(default)
                        f = float(v)
                        return f if f == f else float(default)  # descarta NaN
                    except (TypeError, ValueError):
                        return float(default)

                final_sensors = []
                for i, row in enumerate(edited_records):
                    plane = (row.get("plane_label") or "").strip()
                    stype = (row.get("sensor_type") or "").strip()
                    # Salta la fila fantasma del data_editor dinámico: sin plano
                    # NI tipo no es un sensor real (antes rompía en float(None)).
                    if not plane and not stype:
                        continue
                    base = dict(originals_by_idx.get(i, {}))
                    base.update({
                        "plane_label": plane,
                        "side": row.get("side") or "L",
                        "angle_deg": _num(row.get("angle_deg"), 45.0),
                        "direction": row.get("direction") or "Y",
                        "sensor_type": stype or "proximity",
                        "unit_native": row.get("unit_native") or "",
                        "alarm": _num(row.get("alarm"), 0.0),
                        "danger": _num(row.get("danger"), 0.0),
                        "csv_match_pattern": row.get("csv_match_pattern") or "",
                    })
                    final_sensors.append(base)
                state["sensors_override"] = final_sensors
            _go_next()
            st.rerun()


# =============================================================
# PASO 6 — Datos del activo + crear
# =============================================================

elif current == 6:
    st.subheader("Step 6 · Asset data")
    st.caption("Final info and we create it.")

    col_l, col_r = st.columns(2)

    with col_l:
        suggested_id = state.get("instance_id") or _slug_default(state)
        state["instance_id"] = st.text_input(
            "Unique asset ID (slug)",
            value=suggested_id,
            help="Only letters, numbers, hyphens and underscores. E.g. 'tes1', 'parex_c200c'.",
            key="wiz_instance_id",
        )
        state["tag"] = st.text_input(
            "Client internal tag",
            value=state.get("tag", ""),
            placeholder="e.g. TES1, C-200-C, SGT300A",
            key="wiz_tag",
        )
        state["client"] = st.text_input(
            "Client",
            value=state.get("client", ""),
            placeholder="e.g. Ecopetrol — Magnex, Parex",
            key="wiz_client",
        )
        state["site"] = st.text_input(
            "Site / plant",
            value=state.get("site", ""),
            placeholder="e.g. Termosuria Villavicencio",
            key="wiz_site",
        )
        state["location"] = st.text_input(
            "Physical location (optional)",
            value=state.get("location", ""),
            placeholder="e.g. La Belleza plant, Plato, Magdalena",
            key="wiz_location",
        )

    with col_r:
        state["nominal_rpm"] = st.number_input(
            "Nominal RPM",
            min_value=0.0,
            value=float(state.get("nominal_rpm", 0.0) or 0.0),
            step=100.0,
            key="wiz_rpm",
        )
        state["nominal_power_mw"] = st.number_input(
            "Nominal power (MW)",
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
            "Profile (technical family)",
            options=profile_options,
            index=profile_options.index(suggested_profile),
            format_func=lambda k: PROFILES[k].label,
            key="wiz_profile",
        )
        state["iso_norm_code"] = st.text_input(
            "Recommended ISO standard (optional)",
            value=state.get("iso_norm_code", ""),
            help="E.g. ISO_20816_2 or ISO_10816_7",
            key="wiz_iso",
        )
        state["notes"] = st.text_area(
            "Technical notes",
            value=state.get("notes", ""),
            height=100,
            key="wiz_notes",
        )

    st.markdown("---")
    st.markdown("### Summary")

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Category", state.get("category", "—"))
    s2.metric("Bearings", f"D:{state['driver_planes']} + A:{state['driven_planes']}")
    s3.metric("Driver instr.", state["driver_instrumentation"].replace("_", " "))
    s4.metric("Driven instr.", state["driven_instrumentation"].replace("_", " "))

    s5, s6, s7, s8 = st.columns(4)
    s5.metric("Displ.", state["displacement_unit"])
    s6.metric("Vel.", state["velocity_unit"])
    s7.metric("Accel.", state["acceleration_unit"])
    s8.metric("Keyphasor", "Yes" if state["include_keyphasor"] else "No")

    col_nav = st.columns([1, 2, 1])
    with col_nav[0]:
        if st.button("← Back", use_container_width=True, key="wiz_step6_prev"):
            _go_prev()
            st.rerun()
    with col_nav[2]:
        if st.button("✅ Create asset", type="primary", use_container_width=True,
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
                    f"✅ Asset '{created_id}' created successfully. "
                    f"Sensors generated automatically. "
                    f"You can find it in Machinery Library."
                )
            except Exception as e:
                st.error(f"❌ Error creating the asset: {e}")


# =============================================================
# Footer — botón cancelar / debug
# =============================================================

st.markdown("---")
col_foot = st.columns([3, 1])
with col_foot[1]:
    if st.button("🔄 Reset wizard", key="wiz_reset_btn"):
        _reset_wizard()
        st.rerun()
