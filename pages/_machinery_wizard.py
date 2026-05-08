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

WIZ_KEY = "wm_wizard_state_v1"
TOTAL_STEPS = 5


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
        "driven_type": "",
        "driven_planes": 2,
        "driven_bearing_kind": "rolling",
        "driven_bearing_model": "",
        "coupling_class": "flexible",
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


def _execute_creation(state: Dict[str, Any]) -> None:
    """Crea instance + persiste sensors + parámetros base."""
    inst_id = (state["instance_id"] or "").strip()
    if not inst_id:
        raise ValueError("El ID del activo es obligatorio.")
    if get_instance(inst_id) is not None:
        raise ValueError(f"Ya existe un activo con ID '{inst_id}'.")

    # 1. Crear instance base
    inst = create_instance(
        instance_id=inst_id,
        profile_key=state.get("profile_key", "custom_manual"),
        tag=state.get("tag", ""),
        serial_number="",
        location=state.get("location", ""),
        notes=state.get("notes", ""),
        seed_from_profile=True,
    )

    # 2. Generar sensores
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

    # 3. Persistir header extendido + sensores
    update_instance_header(
        instance_id=inst_id,
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
        iso_norm_code=state.get("iso_norm_code", ""),
        iso_norm_class=state.get("iso_norm_class", ""),
        sensors=sensors,
    )


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

# Stepper visual
step_labels = [
    "1 · Tipo de máquina",
    "2 · Tren mecánico",
    "3 · Instrumentación",
    "4 · Unidades & setpoints",
    "5 · Datos del activo",
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

    with col_r:
        st.markdown("### Driven (accionada)")
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
        st.markdown("### Driven — instrumentación")
        state["driven_instrumentation"] = st.radio(
            "Tipo de sensores en el driven",
            options=list(instrum_options.keys()),
            format_func=lambda k: instrum_options[k],
            index=list(instrum_options.keys()).index(
                state.get("driven_instrumentation", "proximity_xy")
            ),
            key="wiz_driven_instrum",
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

    # Preview rápido del mapa
    with st.expander("👁️ Vista previa del mapa de sensores que se va a generar",
                     expanded=True):
        try:
            preview = generate_standard_sensor_map(
                driver_planes=state["driver_planes"],
                driver_instrumentation=state["driver_instrumentation"],
                driven_planes=state["driven_planes"],
                driven_instrumentation=state["driven_instrumentation"],
                include_keyphasor=state["include_keyphasor"],
            )
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
            _go_next()
            st.rerun()


# =============================================================
# PASO 4 — Unidades & setpoints
# =============================================================

elif current == 4:
    st.subheader("Paso 4 · Unidades y setpoints")
    st.caption(
        "Define las unidades en que reporta cada tipo de sensor, y los niveles "
        "alarm/danger. Estos valores son la fuente de verdad — Tabular List, "
        "Trends y Reports los respetan."
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
            _go_next()
            st.rerun()


# =============================================================
# PASO 5 — Datos del activo + crear
# =============================================================

elif current == 5:
    st.subheader("Paso 5 · Datos del activo")
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
        if st.button("← Atrás", use_container_width=True, key="wiz_step5_prev"):
            _go_prev()
            st.rerun()
    with col_nav[2]:
        if st.button("✅ Crear activo", type="primary", use_container_width=True,
                     key="wiz_create"):
            try:
                _execute_creation(state)
                st.success(
                    f"✅ Activo '{state['instance_id']}' creado correctamente. "
                    f"Sensores generados automáticamente. "
                    f"Lo encontrás en Machinery Library."
                )
                _reset_wizard()
                st.balloons()
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
