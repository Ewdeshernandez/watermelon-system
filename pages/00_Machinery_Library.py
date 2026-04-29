"""
pages/00_Machinery_Library.py
=============================

Machinery Library — cockpit central del sistema (Ciclo 14a).
Renombrado desde pages/17_Asset_Documents.py + promovido a primera
página post-login. Cada instancia (máquina física específica, ej.
"TES1" del cliente Ecopetrol) tiene su propio perfil técnico
extendido + Vault con manuales y parámetros físicos del cojinete.

Diferencia clave con la versión anterior (Ciclo 7):

  Antes: los datos se asociaban al "profile" (familia/tipo de
  máquina). Esto significaba que si tenías dos turbogeneradores
  Brush idénticos, compartían los mismos manuales y parámetros —
  un bug grave.

  Ahora (Ciclo 8): los datos viven por **instance_id**, identificador
  único de la máquina física. Dos instancias del mismo profile son
  independientes: TES1 puede tener un clearance ligeramente distinto
  al de TES2 después de un rebabbiting, y no se pisan.

Adicionalmente: el formulario de parámetros muestra en vivo los
valores derivados (Cd calculado de los diámetros, Cr, L/D, carga
unitaria) usando core/bearing_calculations, sin que el usuario
tenga que hacer las cuentas a mano.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from core.auth import require_login, render_user_menu
from core.bearing_calculations import compute_all_derived
from core.document_vault import CAPTURED_PARAMETER_FIELDS, DOCUMENT_TYPES
from core.instance_selector import render_instance_selector
from core.instance_state import (
    add_uploaded_file_to_instance,
    compose_train_description,
    create_instance,
    delete_instance,
    get_instance,
    get_instance_document_bytes,
    list_instances,
    remove_instance_document,
    update_instance_header,
    update_instance_parameters_bulk,
)
from core.machine_profiles import PROFILES as MACHINE_PROFILES, get_profile
from core.ui_theme import apply_watermelon_page_style, page_header


st.set_page_config(page_title="Watermelon System | Machinery Library", layout="wide")
require_login()
apply_watermelon_page_style()


# ============================================================
# HELPERS UI
# ============================================================

def _bytes_to_human(n: int) -> str:
    if not n:
        return "—"
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    if n < 1024 * 1024 * 1024:
        return f"{n / (1024 * 1024):.1f} MB"
    return f"{n / (1024 * 1024 * 1024):.2f} GB"


def _format_date(iso_str: str) -> str:
    if not iso_str:
        return "—"
    try:
        return datetime.fromisoformat(iso_str).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return iso_str


# ============================================================
# RENDER SECCIONES
# ============================================================

def render_create_instance_section() -> None:
    """Formulario inline para crear una nueva instancia."""
    with st.expander("➕ Crear nueva instancia de activo", expanded=False):
        st.caption(
            "Una instancia representa una máquina física específica que estás "
            "monitoreando. Por ejemplo, si tienes dos turbogeneradores Brush "
            "del mismo modelo, cada uno es una instancia distinta. La instancia "
            "agrupa parámetros, manuales y reportes específicos de esa unidad."
        )

        with st.form("create_instance_form", clear_on_submit=True):
            col1, col2 = st.columns(2)
            with col1:
                inst_id = st.text_input(
                    "ID de la instancia (slug único)",
                    placeholder="brush_tes1",
                    help=(
                        "Identificador interno único, usá solo letras, números, "
                        "guiones bajos y guiones. Ejemplo: brush_tes1, "
                        "siemens_sgt300_planta_b, motor_2pol_a."
                    ),
                )
                tag = st.text_input(
                    "Tag interno del cliente",
                    placeholder="TES1",
                    help="Tag/código corto que el cliente usa para esta máquina.",
                )

            with col2:
                profile_options = sorted(MACHINE_PROFILES.keys())
                profile_labels = {
                    pk: f"{MACHINE_PROFILES[pk].label}" for pk in profile_options
                }
                selected_profile = st.selectbox(
                    "Profile (familia / tipo de máquina)",
                    options=profile_options,
                    format_func=lambda pk: profile_labels.get(pk, pk),
                    help=(
                        "Familia técnica del activo. Define qué normas ISO/API "
                        "aplican y qué módulos son relevantes. Si el profile "
                        "tiene seed (Ciclo 7), los parámetros base se "
                        "pre-llenan automáticamente."
                    ),
                )
                serial = st.text_input(
                    "Número de serie OEM",
                    placeholder="Ej: GE-12345-A",
                )

            location = st.text_input(
                "Ubicación física",
                placeholder="Ej: Planta Térmica Atlántico, Cartagena",
            )
            notes = st.text_area(
                "Notas libres",
                placeholder="Información adicional sobre esta instancia (cliente, contrato, etc.)",
                height=70,
            )

            submitted = st.form_submit_button("Crear instancia", width="stretch")

            if submitted:
                if not inst_id.strip():
                    st.error("El ID es obligatorio.")
                    return
                if get_instance(inst_id.strip()) is not None:
                    st.error(f"Ya existe una instancia con ID '{inst_id.strip()}'. Elegí otro.")
                    return
                inst = create_instance(
                    instance_id=inst_id.strip(),
                    profile_key=selected_profile,
                    tag=tag.strip(),
                    serial_number=serial.strip(),
                    location=location.strip(),
                    notes=notes.strip(),
                    seed_from_profile=True,
                )
                st.success(
                    f"Instancia '{inst.instance_id}' creada. "
                    f"Profile: {profile_labels.get(selected_profile)}. "
                    f"Parámetros heredados del seed: {len(inst.captured_parameters)}."
                )
                st.session_state["wm_active_instance_id"] = inst.instance_id
                st.rerun()


def render_instance_header(state: Dict[str, Any]) -> None:
    """
    Header de la instancia + formulario completo de metadata (Ciclo 14a).
    Tabs por categoría: Identificación · Tren · Operación · Soportes ·
    Sondas · Setpoints · Mantenimiento · Esquemático.
    """
    instance_id = state["instance_id"]
    profile_label = state["profile_label"]
    profile = get_profile(state["profile_key"])

    inst = get_instance(instance_id)
    if inst is None:
        st.warning("Instancia no encontrada.")
        return

    # Cabecera resumen
    title_text = inst.tag or instance_id
    if inst.driver_model:
        title_text += f" · {inst.driver_model}"
    if inst.driven_model and "generador" in inst.driven_model.lower():
        title_text += f" + {inst.driven_model}"
    st.markdown(f"## {title_text}")

    sub_parts = [profile_label]
    if profile:
        sub_parts.append(f"ISO {profile.iso_part}")
        sub_parts.append(f"{profile.operating_rpm:.0f} rpm nominal")
        sub_parts.append(profile.bearing_type)
    st.caption(" · ".join(sub_parts))

    if inst.client or inst.site:
        loc_parts = [p for p in [inst.client, inst.site or inst.location] if p]
        st.caption(f"📍 {' · '.join(loc_parts)}")
    if inst.notes:
        with st.expander("Notas de la instancia", expanded=False):
            st.write(inst.notes)

    # Preview del esquemático (si está cargado)
    if inst.schematic_png:
        try:
            png_bytes = get_instance_document_bytes(instance_id, inst.schematic_png)
            if png_bytes:
                st.image(png_bytes, caption="Esquemático del tren acoplado", width=480)
        except Exception:
            pass

    with st.expander("Editar metadata completa de esta instancia", expanded=False):
        tab_id, tab_train, tab_op, tab_sup, tab_pr, tab_set, tab_mnt, tab_sch = st.tabs([
            "Identificación", "Tren acoplado", "Operación", "Soportes",
            "Sondas", "Setpoints", "Mantenimiento", "Esquemático",
        ])

        with st.form(f"edit_header_{instance_id}"):
            with tab_id:
                c1, c2 = st.columns(2)
                with c1:
                    new_tag = st.text_input("Tag", value=inst.tag or "", help="Identificador corto operativo, ej. TES1")
                    new_client = st.text_input("Cliente", value=inst.client or "", help="ej. ECOPETROL - MAGNEX")
                    new_site = st.text_input("Sitio / Planta", value=inst.site or "", help="ej. TERMOSURIA - VILLAVICENCIO")
                with c2:
                    new_asset_class = st.text_input("Clase de activo", value=inst.asset_class or "", help="ej. TURBOGENERADOR")
                    new_loc = st.text_input("Ubicación (legacy)", value=inst.location or "", help="campo libre antiguo")
                new_notes = st.text_area("Notas", value=inst.notes or "", height=70)

            with tab_train:
                st.markdown("**Driver (máquina motriz)**")
                d1, d2, d3 = st.columns(3)
                with d1:
                    new_drv_mfr = st.text_input("Fabricante driver", value=inst.driver_manufacturer or "")
                with d2:
                    new_drv_mdl = st.text_input("Modelo driver", value=inst.driver_model or "")
                with d3:
                    new_drv_ser = st.text_input("S/N driver (interno)", value=inst.driver_serial or "")
                st.markdown("**Driven (máquina accionada)**")
                e1, e2, e3 = st.columns(3)
                with e1:
                    new_dvn_mfr = st.text_input("Fabricante driven", value=inst.driven_manufacturer or "")
                with e2:
                    new_dvn_mdl = st.text_input("Modelo driven", value=inst.driven_model or "")
                with e3:
                    new_dvn_ser = st.text_input("S/N driven (interno)", value=inst.driven_serial or "")
                p1, p2 = st.columns(2)
                with p1:
                    new_power = st.number_input("Potencia nominal (MW)", value=float(inst.nominal_power_mw or 0.0), min_value=0.0, max_value=2000.0, step=1.0)
                with p2:
                    new_coupling = st.selectbox(
                        "Clase de acople",
                        ["", "rigid", "flexible", "fluid"],
                        index=["", "rigid", "flexible", "fluid"].index(inst.coupling_class) if inst.coupling_class in ["", "rigid", "flexible", "fluid"] else 0,
                    )

            with tab_op:
                o1, o2 = st.columns(2)
                with o1:
                    new_nom_rpm = st.number_input("RPM nominal", value=float(inst.nominal_rpm or 0.0), min_value=0.0, max_value=200000.0, step=10.0)
                    new_min_rpm = st.number_input("Min RPM operativo", value=float(inst.min_rpm or 0.0), min_value=0.0, max_value=200000.0, step=10.0)
                with o2:
                    new_max_rpm = st.number_input("Max RPM operativo", value=float(inst.max_rpm or 0.0), min_value=0.0, max_value=200000.0, step=10.0)
                    new_trip_rpm = st.number_input("Trip RPM (overspeed)", value=float(inst.trip_rpm or 0.0), min_value=0.0, max_value=200000.0, step=10.0)
                new_iso_group = st.text_input("ISO group", value=inst.iso_group or "", help="rigid / flexible")

            with tab_sup:
                s1, s2 = st.columns(2)
                with s1:
                    new_sup_type = st.selectbox(
                        "Tipo de soporte",
                        ["", "fluid_film", "rolling_element", "magnetic", "mixed"],
                        index=["", "fluid_film", "rolling_element", "magnetic", "mixed"].index(inst.support_type) if inst.support_type in ["", "fluid_film", "rolling_element", "magnetic", "mixed"] else 0,
                    )
                with s2:
                    new_sup_count = st.number_input("Cantidad de soportes", value=int(inst.support_count or 0), min_value=0, max_value=20, step=1)
                new_sup_detail = st.text_area(
                    "Detalle (texto libre)",
                    value=inst.support_detail or "",
                    height=80,
                    help="ej. '4 cojinetes planos tilting pad 5 zapatas, ID 254mm, clearance 8mil'",
                )

            with tab_pr:
                p1, p2 = st.columns(2)
                with p1:
                    new_px = st.number_input("Orientación sonda X (°)", value=float(inst.probe_x_orientation_deg or 0.0), min_value=-180.0, max_value=180.0, step=1.0, help="típico 45° (XL) o 0° (vertical)")
                with p2:
                    new_py = st.number_input("Orientación sonda Y (°)", value=float(inst.probe_y_orientation_deg or 0.0), min_value=-180.0, max_value=180.0, step=1.0, help="típico -45° (YR) o 90° (horizontal)")

            with tab_set:
                st.caption("Si están definidos, el motor de severidad usa estos thresholds reales antes que ISO genérico.")
                a1, a2, a3 = st.columns(3)
                with a1:
                    new_alert = st.number_input("Alert level", value=float(inst.alert_level or 0.0), min_value=0.0, step=0.1)
                with a2:
                    new_danger = st.number_input("Danger level", value=float(inst.danger_level or 0.0), min_value=0.0, step=0.1)
                with a3:
                    new_trip = st.number_input("Trip level", value=float(inst.trip_level or 0.0), min_value=0.0, step=0.1)
                new_sp_unit = st.text_input("Unidad", value=inst.setpoint_unit or "", help="ej. mil pp / mm/s rms")

            with tab_mnt:
                m1, m2 = st.columns(2)
                with m1:
                    new_lb = st.text_input("Último balanceo (YYYY-MM-DD)", value=inst.last_balance_date or "")
                    new_la = st.text_input("Último alineamiento", value=inst.last_alignment_date or "")
                with m2:
                    new_lo = st.text_input("Último overhaul mayor", value=inst.last_overhaul_date or "")
                    new_co = st.text_input("Fecha de comisionamiento", value=inst.commissioning_date or "")

            with tab_sch:
                st.caption(
                    "Cualquier imagen del Vault del activo (PNG / JPG / JPEG / GIF / "
                    "WEBP / SVG) aparece como opción acá, sin importar el 'document_type' "
                    "con que la hayas subido. Seleccioná cuál usar como esquemático "
                    "principal del tren para que aparezca en el Resumen Ejecutivo del PDF."
                )
                # Filtro permisivo (Ciclo 14a hotfix 6): acepta documentos que
                # sean imágenes por extensión, además del tipo 'schematic'.
                # Así el usuario no tiene que re-subir si eligió otro tipo
                # cuando lo cargó.
                _SCH_TYPES = ("schematic", "esquematico", "diagram")
                _SCH_EXTS = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".bmp", ".tiff")
                schematic_options = [("", "(sin esquemático)")]
                for d in inst.documents:
                    dtype = (d.get("document_type") or "").lower()
                    fname = (d.get("filename") or "").lower()
                    is_type_match = dtype in _SCH_TYPES
                    is_ext_match = any(fname.endswith(ext) for ext in _SCH_EXTS)
                    if is_type_match or is_ext_match:
                        label = d.get("title") or d.get("filename") or "—"
                        schematic_options.append((d.get("id", ""), label))
                option_ids = [o[0] for o in schematic_options]
                option_labels = [o[1] for o in schematic_options]
                current_idx = option_ids.index(inst.schematic_png) if inst.schematic_png in option_ids else 0
                new_sch_idx = st.selectbox(
                    "Esquemático principal",
                    options=range(len(option_ids)),
                    format_func=lambda i: option_labels[i],
                    index=current_idx,
                )
                new_sch_id = option_ids[new_sch_idx]
                if len(schematic_options) == 1:
                    st.warning(
                        "El Vault de este activo no tiene aún ninguna imagen. "
                        "Subí un PNG/JPG en la sección 'Cargar nuevo documento' "
                        "más abajo y volvé acá."
                    )

            saved = st.form_submit_button("💾 Actualizar metadata completa", width="stretch")
            if saved:
                update_instance_header(
                    instance_id,
                    tag=new_tag.strip(),
                    client=new_client.strip(),
                    site=new_site.strip(),
                    asset_class=new_asset_class.strip(),
                    location=new_loc.strip(),
                    notes=new_notes.strip(),
                    driver_manufacturer=new_drv_mfr.strip(),
                    driver_model=new_drv_mdl.strip(),
                    driver_serial=new_drv_ser.strip(),
                    driven_manufacturer=new_dvn_mfr.strip(),
                    driven_model=new_dvn_mdl.strip(),
                    driven_serial=new_dvn_ser.strip(),
                    nominal_power_mw=float(new_power),
                    coupling_class=new_coupling.strip(),
                    nominal_rpm=float(new_nom_rpm),
                    min_rpm=float(new_min_rpm),
                    max_rpm=float(new_max_rpm),
                    trip_rpm=float(new_trip_rpm),
                    iso_group=new_iso_group.strip(),
                    support_type=new_sup_type.strip(),
                    support_count=int(new_sup_count),
                    support_detail=new_sup_detail.strip(),
                    probe_x_orientation_deg=float(new_px),
                    probe_y_orientation_deg=float(new_py),
                    alert_level=float(new_alert),
                    danger_level=float(new_danger),
                    trip_level=float(new_trip),
                    setpoint_unit=new_sp_unit.strip(),
                    last_balance_date=new_lb.strip(),
                    last_alignment_date=new_la.strip(),
                    last_overhaul_date=new_lo.strip(),
                    commissioning_date=new_co.strip(),
                    schematic_png=new_sch_id.strip(),
                )
                st.success("Metadata actualizada.")
                st.rerun()


# ============================================================
# Ciclo 14c.1 — SENSOR MAP (mapa de sensores per-instancia)
# ============================================================

def render_sensor_map_section(instance_id: str) -> None:
    """
    Sección "📍 Mapa de Sensores" — editable in-place con st.data_editor.

    Cada fila describe un sensor de vibración con su ubicación física
    (plano + lado + ángulo + dirección) + tipo + unidad nativa +
    setpoints individuales + patrón para matchear el Point del CSV.

    Botones:
      - Generar mapa estándar: pre-llena 8 sensores típicos
        (4 cojinetes × 2 sondas X-Y proximity API 670 a 45° R/L).
      - Limpiar mapa: borra todos los sensores.
      - Guardar mapa: persiste los cambios del data_editor.
    """
    from core.sensor_map import generate_standard_sensor_map, sensor_label

    inst = get_instance(instance_id)
    if inst is None:
        return

    st.markdown("### 📍 Mapa de Sensores")
    st.caption(
        "Configurá una sola vez los sensores físicos del activo: ubicación API 670 / "
        "ISO 20816-1 (planos numerados de **driver → driven**), dirección X/Y a 45° R/L, "
        "tipo (proximity / velocity / accelerometer), unidad nativa y setpoints "
        "individuales del DCS. Después Tabular List clasifica cada CSV cargado con "
        "los thresholds correctos del sensor que matchea."
    )

    # Form de generación + botón limpiar
    # Ciclo 14c.1.1 — antes el botón "Generar mapa estándar" asumía un layout
    # único (8 proxímetros + 2 acelerómetros). Ahora pregunta el tipo de
    # soporte de driver y driven por separado, soportando trenes mixtos como
    # turbina aero (rolling_element con TRF/CRF) + generador (fluid_film X-Y).

    with st.expander(
        "🪄 Generar mapa estándar (configurable)",
        expanded=(len(inst.sensors) == 0),
    ):
        st.caption(
            "Configurá driver y driven por separado. "
            "**Driver = máquina motriz** (turbina, motor). "
            "**Driven = máquina accionada** (generador, bomba, compresor). "
            "Para cada lado elegís cuántos planos (cojinetes) tiene y qué tipo "
            "de soporte: `fluid_film` genera par X-Y proxímetros a 45° R/L "
            "(API 670); `rolling_element` genera 1 acelerómetro radial por plano."
        )

        # 3 modos de instrumentación. Para que sean amigables, traducimos
        # internamente los keys técnicos a etiquetas claras.
        _MODE_LABELS = {
            "proximity_xy": "Proxímetros X-Y (API 670, fluid_film)",
            "axial_accel": "Acelerómetro radial (1 por cojinete)",
            "accel_plus_velocity": "Acelerómetro + Velocímetro (turbinas aero, TRF/CRF)",
        }
        _MODE_KEYS = list(_MODE_LABELS.keys())
        _MODE_LABEL_LIST = list(_MODE_LABELS.values())

        gcol1, gcol2 = st.columns(2)
        with gcol1:
            st.markdown("**Driver (motriz)**")
            gen_driver_planes = st.number_input(
                "Planos del driver",
                min_value=1, max_value=8, value=2, step=1,
                key=f"gen_driver_planes_{instance_id}",
            )
            # Default modo según support_type ya configurado en la instancia
            _sup = (inst.support_type or "").lower()
            _driver_default_mode_idx = (
                _MODE_KEYS.index("accel_plus_velocity") if _sup == "rolling_element"
                else _MODE_KEYS.index("proximity_xy")
            )
            _gen_driver_mode_label = st.selectbox(
                "Instrumentación driver",
                options=_MODE_LABEL_LIST,
                index=_driver_default_mode_idx,
                key=f"gen_driver_mode_{instance_id}",
                help=(
                    "**Proxímetros X-Y**: cojinetes planos hidrodinámicos (Brush, Siemens grandes). "
                    "**Acelerómetro radial**: rodamientos simples (motores chicos, bombas). "
                    "**Accel + Velocity**: turbinas aero modernas (LM6000, TM2500) con "
                    "instrumentación completa en TRF y CRF."
                ),
            )
            gen_driver_mode = _MODE_KEYS[_MODE_LABEL_LIST.index(_gen_driver_mode_label)]
            gen_driver_prefix = ""
            if gen_driver_mode in ("axial_accel", "accel_plus_velocity"):
                gen_driver_prefix = st.text_input(
                    "Prefijo Point CSV (acelerómetros)",
                    value="acell",
                    key=f"gen_driver_prefix_{instance_id}",
                    help="Texto que aparece en el Point del CSV. Ej. 'TRF', 'CRF', 'BRG', 'casing', 'acell'. "
                         "Si tu equipo tiene CRF y TRF (LM6000), generá una primera vez con prefijo 'CRF' y "
                         "después editá manualmente el segundo plano para que su pattern diga 'TRF'.",
                )

        with gcol2:
            st.markdown("**Driven (accionada)**")
            gen_driven_planes = st.number_input(
                "Planos del driven",
                min_value=1, max_value=8, value=2, step=1,
                key=f"gen_driven_planes_{instance_id}",
            )
            _gen_driven_mode_label = st.selectbox(
                "Instrumentación driven",
                options=_MODE_LABEL_LIST,
                index=_MODE_KEYS.index("proximity_xy"),
                key=f"gen_driven_mode_{instance_id}",
                help="Generadores grandes y compresores centrífugos típicamente = "
                     "Proxímetros X-Y. Bombas chicas y motores = Accel radial.",
            )
            gen_driven_mode = _MODE_KEYS[_MODE_LABEL_LIST.index(_gen_driven_mode_label)]
            gen_driven_prefix = ""
            if gen_driven_mode in ("axial_accel", "accel_plus_velocity"):
                gen_driven_prefix = st.text_input(
                    "Prefijo Point CSV (acelerómetros driven)",
                    value="acell",
                    key=f"gen_driven_prefix_{instance_id}",
                )

        # Keyphasor (referencia 1X de fase)
        gen_include_keyphasor = st.checkbox(
            "Incluir keyphasor en coupling (referencia 1X para Polar/Bode)",
            value=False,
            key=f"gen_keyphasor_{instance_id}",
            help="Sensor de fase montado típicamente en el lado acople "
                 "entre driver y driven. Genera 1 pulso por revolución y se usa "
                 "como referencia angular para los plots polares y diagramas Bode.",
        )

        # Confirmación si ya hay sensores configurados
        confirm_overwrite = True
        if len(inst.sensors) > 0:
            st.warning(
                f"⚠️ Ya hay **{len(inst.sensors)} sensores** configurados. "
                "Al generar uno nuevo se reemplazan TODOS los existentes."
            )
            confirm_overwrite = st.checkbox(
                "Confirmo sobreescribir el mapa actual",
                key=f"confirm_overwrite_{instance_id}",
            )

        if st.button(
            "🪄 Generar mapa con esta configuración",
            key=f"gen_sensor_map_{instance_id}",
            type="primary",
            disabled=(len(inst.sensors) > 0 and not confirm_overwrite),
        ):
            new_map = generate_standard_sensor_map(
                driver_planes=int(gen_driver_planes),
                driver_instrumentation=gen_driver_mode,
                driver_accel_prefix=gen_driver_prefix.strip() or "acell",
                driven_planes=int(gen_driven_planes),
                driven_instrumentation=gen_driven_mode,
                driven_accel_prefix=gen_driven_prefix.strip() or "acell",
                include_keyphasor=gen_include_keyphasor,
            )
            update_instance_header(instance_id, sensors=new_map)
            st.success(f"Mapa generado con {len(new_map)} sensores.")
            st.rerun()

    # Botón limpiar (separado, siempre disponible)
    if st.button(
        "🗑️ Limpiar mapa de sensores",
        key=f"clear_sensor_map_{instance_id}",
        disabled=len(inst.sensors) == 0,
    ):
        update_instance_header(instance_id, sensors=[])
        st.success("Mapa de sensores limpiado.")
        st.rerun()

    if not inst.sensors:
        st.info(
            "Este activo no tiene sensores configurados. "
            "Expandí **🪄 Generar mapa estándar** arriba para empezar con un "
            "layout configurable según tipo de soporte de driver y driven, "
            "o configurá sensor por sensor manualmente con el editor de abajo."
        )

    # Editor in-place del mapa
    df_sensors = pd.DataFrame(inst.sensors)
    if df_sensors.empty:
        # Skeleton de columnas para que el data_editor permita agregar filas
        df_sensors = pd.DataFrame(columns=[
            "plane", "plane_label", "side", "angle_deg", "direction",
            "sensor_type", "unit_native", "alarm", "danger",
            "csv_match_pattern", "notes",
        ])

    edited_df = st.data_editor(
        df_sensors,
        num_rows="dynamic",
        key=f"sensor_map_editor_{instance_id}",
        column_config={
            "plane": st.column_config.NumberColumn(
                "Plano", min_value=1, max_value=20, step=1, default=1,
                help="Número correlativo desde driver (1) a driven (último). API 670.",
            ),
            "plane_label": st.column_config.TextColumn(
                "Etiqueta plano",
                help="ej. 'DE driver', 'NDE driven'. Opcional, para display en UI.",
            ),
            "side": st.column_config.SelectboxColumn(
                "Lado",
                options=["L", "R", "top", "bottom", "—"],
                default="L",
                help="Hemisferio visto desde el extremo del driver. L=izquierdo, R=derecho.",
            ),
            "angle_deg": st.column_config.NumberColumn(
                "Ángulo (°)", min_value=-180.0, max_value=180.0, step=1.0, default=45.0,
                help="0° = arriba. Sondas X-Y API 670 típicas: ±45°.",
            ),
            "direction": st.column_config.SelectboxColumn(
                "Dir",
                options=["X", "Y", "radial", "axial"],
                default="Y",
            ),
            "sensor_type": st.column_config.SelectboxColumn(
                "Tipo",
                options=["proximity", "velocity", "accelerometer", "keyphasor"],
                default="proximity",
                help=(
                    "proximity → Desplazamiento (mil pp / µm pp). "
                    "velocity → Velocidad (mm/s RMS / in/s peak). "
                    "accelerometer → Aceleración (g RMS / m/s² RMS). "
                    "keyphasor → Referencia 1X de fase (pulses/rev), "
                    "típicamente en coupling."
                ),
            ),
            "unit_native": st.column_config.SelectboxColumn(
                "Unidad",
                options=[
                    # Desplazamiento
                    "mil pp",
                    "µm pp",
                    "mm pp",
                    # Velocidad
                    "mm/s RMS",
                    "mm/s peak",
                    "in/s RMS",
                    "in/s peak",
                    # Aceleración
                    "g RMS",
                    "g peak",
                    "m/s² RMS",
                    "m/s² peak",
                ],
                default="mil pp",
                help=(
                    "Unidad nativa del sensor según su tipo. Las opciones están "
                    "agrupadas: las primeras 3 son desplazamiento (proximity), "
                    "las siguientes 4 velocidad (velocity), las últimas 4 "
                    "aceleración (accelerometer). Elegí la que use tu DCS / OEM."
                ),
            ),
            "alarm": st.column_config.NumberColumn(
                "Alarm", min_value=0.0, step=0.1, format="%.3f", default=4.0,
                help="Setpoint de alerta en la unidad nativa. Cuando la amplitud supera este valor, status = ATENCIÓN.",
            ),
            "danger": st.column_config.NumberColumn(
                "Danger", min_value=0.0, step=0.1, format="%.3f", default=6.0,
                help="Setpoint de disparo en la unidad nativa. Cuando la amplitud supera este valor, status = ACCIÓN REQUERIDA / CRÍTICA.",
            ),
            "csv_match_pattern": st.column_config.TextColumn(
                "Texto Point CSV",
                help=(
                    "Texto que aparece en el campo Point del CSV cargado. "
                    "Tres formatos válidos:\n"
                    "  • Substring simple: 'VE5807' matchea 'VE5807 (Y)' o 'VE5807-Y'.\n"
                    "  • Lista por comas: 'VE5807 (Y), VE5807-Y, 5807_Y' matchea cualquiera de las variantes.\n"
                    "  • Glob: '*5807*y*' usa comodines para casos avanzados.\n"
                    "El match es case-insensitive."
                ),
            ),
            "notes": st.column_config.TextColumn("Notas"),
        },
        width="stretch",
    )

    if st.button(
        "💾 Guardar mapa de sensores",
        key=f"save_sensor_map_{instance_id}",
        type="primary",
        width="stretch",
    ):
        # Convertir DataFrame a lista de dicts limpios. Ciclo 15.2 —
        # preservar x_pct/y_pct (coordenadas click-to-place) que no
        # estan en el data_editor pero viven en el sensor original.
        # Mapeamos por (plane, direction, side, sensor_type) que en
        # conjunto identifican univocamente al sensor.
        existing_coords: Dict[tuple, tuple] = {}
        for _s in (inst.sensors or []):
            try:
                k = (
                    int(_s.get("plane", 0) or 0),
                    str(_s.get("direction", "") or ""),
                    str(_s.get("side", "") or ""),
                    str(_s.get("sensor_type", "") or ""),
                )
                xp = _s.get("x_pct")
                yp = _s.get("y_pct")
                if xp is not None or yp is not None:
                    existing_coords[k] = (xp, yp)
            except Exception:
                continue

        new_sensors = []
        for _, row in edited_df.iterrows():
            try:
                sensor_dict = {
                    "plane": int(row.get("plane", 1) or 1),
                    "plane_label": str(row.get("plane_label", "") or ""),
                    "side": str(row.get("side", "L") or "L"),
                    "angle_deg": float(row.get("angle_deg", 45.0) or 45.0),
                    "direction": str(row.get("direction", "Y") or "Y"),
                    "sensor_type": str(row.get("sensor_type", "proximity") or "proximity"),
                    "unit_native": str(row.get("unit_native", "mil pp") or "mil pp"),
                    "alarm": float(row.get("alarm", 0.0) or 0.0),
                    "danger": float(row.get("danger", 0.0) or 0.0),
                    "csv_match_pattern": str(row.get("csv_match_pattern", "") or ""),
                    "notes": str(row.get("notes", "") or ""),
                }
                # Re-asociar coordenadas previas si las habia
                k = (
                    sensor_dict["plane"],
                    sensor_dict["direction"],
                    sensor_dict["side"],
                    sensor_dict["sensor_type"],
                )
                if k in existing_coords:
                    sensor_dict["x_pct"] = existing_coords[k][0]
                    sensor_dict["y_pct"] = existing_coords[k][1]
                else:
                    sensor_dict["x_pct"] = None
                    sensor_dict["y_pct"] = None
                new_sensors.append(sensor_dict)
            except Exception:
                continue
        update_instance_header(instance_id, sensors=new_sensors)
        st.success(f"Mapa guardado con {len(new_sensors)} sensores.")
        st.rerun()

    # Preview del mapa actual con labels formateados + diagrama visual
    if inst.sensors:
        with st.expander(f"Preview del mapa actual ({len(inst.sensors)} sensores)", expanded=False):
            preview_lines = []
            for s in inst.sensors:
                lbl = sensor_label(s)
                ploc = (
                    f"plano {s.get('plane', '?')} ({s.get('plane_label', '')}) · "
                    f"{s.get('side', '')} {s.get('angle_deg', 0):.0f}° · "
                    f"{s.get('direction', '')}"
                )
                tinfo = (
                    f"{s.get('sensor_type', '')} ({s.get('unit_native', '')}) · "
                    f"A={s.get('alarm', 0):.2f} D={s.get('danger', 0):.2f}"
                )
                pat = s.get('csv_match_pattern', '') or '(sin pattern)'
                preview_lines.append(f"- **{lbl}** · {ploc} · {tinfo} · match=`{pat}`")
            st.markdown("\n".join(preview_lines))

        # Ciclo 14c.2 — diagrama visual del mapa de sensores
        st.markdown("#### 🎯 Diagrama visual del mapa")
        st.caption(
            "Vista lateral del tren con cojinetes numerados (convención API 670 / "
            "ISO 20816-1 driver→driven) y vista polar por plano con sondas en sus "
            "ángulos físicos. R/L vistos desde el extremo del driver, 0° arriba."
        )
        try:
            from core.sensor_diagram import render_sensor_map_diagram
            _train_lbl = compose_train_description(inst) or ""
            _drv_lbl = " ".join(p for p in [inst.driver_manufacturer, inst.driver_model] if p) or "Driver"
            _dvn_lbl = " ".join(p for p in [inst.driven_manufacturer, inst.driven_model] if p) or "Driven"
            _diag_png = render_sensor_map_diagram(
                inst.sensors,
                train_label=_train_lbl,
                driver_label=_drv_lbl,
                driven_label=_dvn_lbl,
            )
            if _diag_png:
                st.image(_diag_png, use_container_width=True)
            else:
                st.warning(
                    "No se pudo renderizar el diagrama. "
                    "Verificá que matplotlib esté disponible en el entorno."
                )
        except Exception as e:
            st.warning(f"Error al renderizar diagrama: {e}")

        # ============================================================
        # Ciclo 15.2 — Click-to-place sobre el schematic_png real
        # ------------------------------------------------------------
        # Permite asignar coordenadas (x_pct, y_pct) a cada sensor del
        # mapa haciendo clic en la imagen del activo. Una vez
        # configurado, el Resumen Ejecutivo del PDF y la pagina
        # Machine Map renderizan los markers de severidad + valores
        # Overall sobre la foto/dibujo real en lugar del esquematico
        # generico turbomachinery.
        #
        # Si no hay schematic_png cargado para esta instancia, se omite
        # la seccion. Si no hay streamlit_image_coordinates instalado,
        # se ofrece un fallback de inputs numericos (defensivo).
        # ============================================================
        if inst.schematic_png:
            st.markdown("---")
            st.markdown("#### 📍 Posicionar sensores sobre el esquemático")
            st.caption(
                "Ubicá cada cojinete sobre la foto/dibujo del activo. Una vez "
                "posicionados, los reportes muestran los valores de vibración "
                "Overall coloreados por severidad sobre tu esquemático real, "
                "no sobre el genérico turbomachinery."
            )

            try:
                _sch_bytes = get_instance_document_bytes(
                    inst.instance_id, inst.schematic_png
                )
            except Exception:
                _sch_bytes = None

            if not _sch_bytes:
                st.info("No se pudo cargar el esquemático del activo.")
            else:
                # Inventario de planos del Sensor Map (un boton por plano —
                # no por sensor — porque los sensores que comparten plano
                # comparten posicion fisica).
                planes_map: Dict[int, Dict[str, Any]] = {}
                for _s in inst.sensors:
                    p = int(_s.get("plane", 0) or 0)
                    if p <= 0 or str(_s.get("sensor_type", "")).lower() == "keyphasor":
                        # Keyphasor lo manejamos aparte (suele ir en coupling)
                        if str(_s.get("sensor_type", "")).lower() == "keyphasor":
                            planes_map.setdefault("KP", {
                                "is_kp": True,
                                "plane_label": "Keyphasor",
                                "x_pct": _s.get("x_pct"),
                                "y_pct": _s.get("y_pct"),
                            })
                        continue
                    if p not in planes_map:
                        planes_map[p] = {
                            "is_kp": False,
                            "plane_label": _s.get("plane_label", "") or f"Plano {p}",
                            "x_pct": _s.get("x_pct"),
                            "y_pct": _s.get("y_pct"),
                        }

                if not planes_map:
                    st.info(
                        "Configurá primero los sensores del mapa arriba "
                        "para poder posicionarlos sobre el esquemático."
                    )
                else:
                    # UI de seleccion: que plano vamos a posicionar.
                    # Sort: planos numericos primero, KP al final.
                    plane_keys_sorted = sorted(
                        planes_map.keys(),
                        key=lambda k: (1, 999) if k == "KP" else (0, k),
                    )
                    plane_options = []
                    for k in plane_keys_sorted:
                        info = planes_map[k]
                        coord_status = (
                            f" · ✓ posicionado ({info['x_pct']:.1f}%, {info['y_pct']:.1f}%)"
                            if info["x_pct"] is not None and info["y_pct"] is not None
                            else " · ✗ sin posicionar"
                        )
                        if k == "KP":
                            plane_options.append(("KP", f"⭐ Keyphasor{coord_status}"))
                        else:
                            plane_options.append((k, f"Plano {k} · {info['plane_label']}{coord_status}"))

                    # Mantener seleccion entre reruns mediante session_state.
                    # Usamos la KEY del plano (no la label) porque la label
                    # cambia cuando se guarda una posicion (pasa de "sin
                    # posicionar" a "posicionado") y eso hacia que Streamlit
                    # no pudiera matchear el value previo y caia al indice 0
                    # (= Keyphasor con sort viejo).
                    _ctp_state_key = f"ctp_selected_plane_key_{instance_id}"
                    keys_in_order = [k for k, _ in plane_options]
                    if _ctp_state_key not in st.session_state or \
                       st.session_state[_ctp_state_key] not in keys_in_order:
                        st.session_state[_ctp_state_key] = keys_in_order[0]
                    default_idx = keys_in_order.index(st.session_state[_ctp_state_key])

                    selected_label = st.selectbox(
                        "Plano a posicionar (clic en la imagen abajo)",
                        [lbl for _, lbl in plane_options],
                        index=default_idx,
                        key=f"ctp_plane_select_widget_{instance_id}",
                    )
                    sel_label_to_key = {lbl: k for k, lbl in plane_options}
                    selected_plane = sel_label_to_key[selected_label]
                    # Persistir la key seleccionada para sobrevivir reruns
                    st.session_state[_ctp_state_key] = selected_plane

                    # Render con streamlit_image_coordinates
                    captured_xy: Optional[tuple] = None
                    img_w_px: Optional[int] = None
                    img_h_px: Optional[int] = None
                    try:
                        from streamlit_image_coordinates import streamlit_image_coordinates
                        from PIL import Image as PILImage
                        from io import BytesIO as _BIO

                        # Renderizar overlay con TODOS los sensores ya
                        # posicionados (modo configuracion — sin severity)
                        from core.sensor_diagram import render_on_schematic
                        _preview_png = render_on_schematic(
                            _sch_bytes, inst.sensors,
                            severity_by_label=None,
                            overall_by_label=None,
                            unit_by_label=None,
                            show_values=False,
                            show_labels=True,
                        ) or _sch_bytes

                        # streamlit_image_coordinates exige un PIL.Image o
                        # path/numpy array — no acepta bytes crudos. Lo
                        # decodificamos antes de pasarlo.
                        _preview_pil = PILImage.open(_BIO(_preview_png))
                        img_w_px, img_h_px = _preview_pil.size

                        coords = streamlit_image_coordinates(
                            _preview_pil,
                            key=f"ctp_canvas_{instance_id}",
                            use_column_width=True,
                        )
                        if coords is not None:
                            # streamlit_image_coordinates devuelve coords en
                            # pixeles relativos al tamaño REAL de la imagen
                            # (no el display) desde v0.1.6+.
                            try:
                                cx = float(coords["x"])
                                cy = float(coords["y"])
                                xp_pct = (cx / img_w_px) * 100.0
                                yp_pct = (cy / img_h_px) * 100.0
                                xp_pct = max(0.0, min(100.0, xp_pct))
                                yp_pct = max(0.0, min(100.0, yp_pct))
                                captured_xy = (xp_pct, yp_pct)
                            except Exception:
                                captured_xy = None
                    except ImportError:
                        st.warning(
                            "El paquete `streamlit-image-coordinates` no está "
                            "instalado. Usá el fallback numérico abajo."
                        )

                    # Fallback / edicion manual + confirmacion
                    cur_xp = planes_map[selected_plane].get("x_pct")
                    cur_yp = planes_map[selected_plane].get("y_pct")

                    cols_xy = st.columns([1, 1, 1])
                    new_xp = cols_xy[0].number_input(
                        "X (%)", min_value=0.0, max_value=100.0,
                        value=float(captured_xy[0]) if captured_xy else (
                            float(cur_xp) if cur_xp is not None else 50.0
                        ),
                        step=0.5, format="%.1f",
                        key=f"ctp_x_{instance_id}_{selected_plane}",
                    )
                    new_yp = cols_xy[1].number_input(
                        "Y (%)", min_value=0.0, max_value=100.0,
                        value=float(captured_xy[1]) if captured_xy else (
                            float(cur_yp) if cur_yp is not None else 50.0
                        ),
                        step=0.5, format="%.1f",
                        key=f"ctp_y_{instance_id}_{selected_plane}",
                    )
                    if cols_xy[2].button(
                        "💾 Guardar posición de este plano",
                        key=f"ctp_save_{instance_id}_{selected_plane}",
                        type="primary",
                        width="stretch",
                    ):
                        # Aplicar coords a TODOS los sensores que comparten el plano
                        updated_sensors = []
                        for _s in inst.sensors:
                            _s2 = dict(_s)
                            if selected_plane == "KP":
                                if str(_s.get("sensor_type", "")).lower() == "keyphasor":
                                    _s2["x_pct"] = float(new_xp)
                                    _s2["y_pct"] = float(new_yp)
                            else:
                                if int(_s.get("plane", 0) or 0) == int(selected_plane):
                                    _s2["x_pct"] = float(new_xp)
                                    _s2["y_pct"] = float(new_yp)
                            updated_sensors.append(_s2)
                        update_instance_header(instance_id, sensors=updated_sensors)
                        st.success(
                            f"Posición guardada para "
                            f"{'Keyphasor' if selected_plane == 'KP' else f'Plano {selected_plane}'}"
                            f" → ({new_xp:.1f}%, {new_yp:.1f}%)"
                        )
                        st.rerun()

                    # Boton para limpiar todas las coords (rehacer desde cero)
                    if any(
                        v.get("x_pct") is not None for v in planes_map.values()
                    ):
                        if st.button(
                            "🧹 Borrar todas las posiciones",
                            key=f"ctp_clear_{instance_id}",
                        ):
                            cleared = []
                            for _s in inst.sensors:
                                _s2 = dict(_s)
                                _s2["x_pct"] = None
                                _s2["y_pct"] = None
                                cleared.append(_s2)
                            update_instance_header(instance_id, sensors=cleared)
                            st.info("Coordenadas limpiadas.")
                            st.rerun()


def render_documents_section(instance_id: str) -> None:
    """Lista de documentos cargados de la instancia + acciones."""
    inst = get_instance(instance_id)
    if inst is None:
        return
    docs = list(inst.documents)
    docs.sort(key=lambda d: d.get("uploaded_at", ""), reverse=True)

    st.markdown("### Documentos del activo")
    if not docs:
        st.info(
            "Esta instancia aún no tiene documentos. Subí manuales OEM, "
            "reportes de mantenimiento, certificados o especificaciones "
            "desde la sección 'Cargar nuevo documento' más abajo."
        )
        return

    df = pd.DataFrame([
        {
            "Título": d.get("title", d.get("filename", "—")),
            "Tipo": DOCUMENT_TYPES.get(d.get("document_type", "other"), d.get("document_type", "—")),
            "Archivo": d.get("filename", "—"),
            "Tamaño": _bytes_to_human(int(d.get("size_bytes", 0))),
            "Subido": _format_date(d.get("uploaded_at", "")),
            "Tags": ", ".join(d.get("tags", [])) if d.get("tags") else "—",
        }
        for d in docs
    ])
    st.dataframe(df, width="stretch", hide_index=True)

    st.markdown("**Acciones**")
    for d in docs:
        with st.expander(f"📄 {d.get('title') or d.get('filename')}"):
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.caption(d.get("description", "") or "_(sin descripción)_")
                if d.get("tags"):
                    st.caption("Tags: " + ", ".join(d["tags"]))
                st.caption(f"ID interno: `{d.get('id')}`")
            with col2:
                # Bytes streaming-friendly: funciona igual con backend Local
                # o Supabase sin diferencias de UX para el usuario.
                file_bytes = get_instance_document_bytes(instance_id, d["id"])
                if file_bytes is not None:
                    st.download_button(
                        "Descargar",
                        data=file_bytes,
                        file_name=d.get("filename", "document"),
                        key=f"dl_{instance_id}_{d['id']}",
                        width="stretch",
                    )
                else:
                    st.caption("Archivo no disponible")
            with col3:
                if st.button("Eliminar", key=f"del_{instance_id}_{d['id']}", width="stretch"):
                    remove_instance_document(instance_id, d["id"])
                    st.success(f"Documento '{d.get('title')}' eliminado.")
                    st.rerun()


def render_upload_section(instance_id: str) -> None:
    """Formulario de upload de nuevo documento a la instancia activa."""
    st.markdown("### Cargar nuevo documento")
    st.caption(
        f"Los documentos cargados aquí quedan asociados exclusivamente a "
        f"la instancia activa (`{instance_id}`). No se comparten con otras "
        f"instancias del mismo profile."
    )

    with st.form("upload_form", clear_on_submit=True):
        uploaded_file = st.file_uploader(
            "Archivo (PDF, imagen, documento, hoja de datos)",
            type=None,
            accept_multiple_files=False,
        )

        col1, col2 = st.columns(2)
        with col1:
            doc_title = st.text_input(
                "Título descriptivo",
                placeholder="Ej: Reporte rebabbiting cojinetes (Wersin, oct 2018)",
            )
            doc_type_label_to_key = {label: key for key, label in DOCUMENT_TYPES.items()}
            doc_type_label = st.selectbox("Tipo de documento", list(DOCUMENT_TYPES.values()))
            doc_type = doc_type_label_to_key[doc_type_label]
        with col2:
            doc_description = st.text_area(
                "Descripción / contexto",
                placeholder="Resumen breve del contenido o contexto del documento.",
                height=100,
            )
            doc_tags_str = st.text_input(
                "Tags (separados por coma)",
                placeholder="bearing, rebabbiting, wersin, 2018",
            )

        submitted = st.form_submit_button("Cargar documento", width="stretch")

        if submitted:
            if uploaded_file is None:
                st.error("Selecciona un archivo antes de cargar.")
                return
            tags = [t.strip() for t in doc_tags_str.split(",") if t.strip()]
            doc_id = add_uploaded_file_to_instance(
                instance_id,
                uploaded_file,
                title=doc_title or uploaded_file.name,
                document_type=doc_type,
                description=doc_description,
                tags=tags,
            )
            if doc_id:
                st.success(
                    f"Documento '{uploaded_file.name}' cargado en instancia "
                    f"'{instance_id}'. ID: `{doc_id}`"
                )
                st.rerun()
            else:
                st.error("No fue posible cargar el documento.")


def render_captured_parameters_section(instance_id: str) -> None:
    """Form de parámetros estructurados con auto-cálculos en vivo."""
    inst = get_instance(instance_id)
    if inst is None:
        return

    st.markdown("### Parámetros técnicos del activo")
    st.caption(
        "Captura los parámetros físicos del activo extraídos de los manuales "
        "OEM o de mediciones de campo. Estos valores alimentan los módulos "
        "de análisis (Shaft Centerline, Polar, Bode) cuando requieren datos "
        "específicos del cojinete o del rotor."
    )

    current_values = dict(inst.captured_parameters)

    # Panel de auto-cálculos en vivo (solo lectura)
    derived = compute_all_derived(current_values)
    if derived:
        st.markdown("#### 🧮 Valores calculados automáticamente")
        st.caption(
            "Derivados en vivo de los parámetros ingresados. Si tipeás "
            "Cd manualmente abajo, ese valor manual gana sobre el cálculo."
        )
        cols = st.columns(min(len(derived), 4) or 1)
        col_idx = 0
        for key, info in derived.items():
            with cols[col_idx % len(cols)]:
                if key == "diametral_clearance":
                    st.metric(
                        "Cd diametral",
                        f"{info['value_mm']:.3f} mm",
                        delta=f"{info['value_mil']:.2f} mil pp",
                        help=info["explanation"],
                    )
                elif key == "radial_clearance":
                    st.metric(
                        "Cr radial",
                        f"{info['value_mm']:.3f} mm",
                        delta=f"{info['value_mil']:.2f} mil pp",
                        help=info["explanation"],
                    )
                elif key == "l_over_d":
                    st.metric(
                        "L/D",
                        f"{info['value']:.2f}",
                        delta=info["interpretation"],
                        delta_color="off",
                        help=info["explanation"],
                    )
                elif key == "unit_load":
                    st.metric(
                        "Carga unitaria",
                        f"{info['value_mpa']:.2f} MPa",
                        delta=info["interpretation"],
                        delta_color="off",
                        help=info["explanation"],
                    )
                elif key == "lift_off_speed":
                    st.metric(
                        "Lift-off est.",
                        f"{info['value_rpm']:.0f} rpm",
                        help=info["explanation"],
                    )
            col_idx += 1
        st.markdown("---")

    # Agrupar campos por categoría
    by_category: Dict[str, List[tuple]] = {}
    for field_key, field_def in CAPTURED_PARAMETER_FIELDS.items():
        cat = field_def.get("category", "Otros")
        by_category.setdefault(cat, []).append((field_key, field_def))

    new_values: Dict[str, Any] = {}

    with st.form("captured_params_form"):
        for category in sorted(by_category.keys()):
            with st.expander(
                f"📋 {category}",
                expanded=(category in ("Cojinete - geometría", "Identificación")),
            ):
                fields = by_category[category]
                cols = st.columns(2)
                for idx, (field_key, field_def) in enumerate(fields):
                    with cols[idx % 2]:
                        ftype = field_def.get("type", "str")
                        label = field_def.get("label", field_key)
                        help_text = field_def.get("help", "")
                        current = current_values.get(field_key)

                        if ftype == "float":
                            raw = st.text_input(
                                label,
                                value=str(current) if current is not None else "",
                                key=f"param_{instance_id}_{field_key}",
                                help=help_text,
                            )
                            if raw.strip() == "":
                                new_values[field_key] = None
                            else:
                                try:
                                    new_values[field_key] = float(raw.replace(",", "."))
                                except ValueError:
                                    new_values[field_key] = current
                        elif ftype == "date":
                            raw_date = st.text_input(
                                label,
                                value=str(current) if current is not None else "",
                                key=f"param_{instance_id}_{field_key}",
                                placeholder="YYYY-MM-DD",
                                help=help_text or "Formato YYYY-MM-DD",
                            )
                            new_values[field_key] = raw_date.strip() if raw_date.strip() else None
                        elif ftype == "text":
                            raw = st.text_area(
                                label,
                                value=str(current) if current is not None else "",
                                key=f"param_{instance_id}_{field_key}",
                                help=help_text,
                            )
                            new_values[field_key] = raw if raw.strip() else None
                        else:
                            raw = st.text_input(
                                label,
                                value=str(current) if current is not None else "",
                                key=f"param_{instance_id}_{field_key}",
                                help=help_text,
                            )
                            new_values[field_key] = raw if raw.strip() else None

        submitted = st.form_submit_button("Guardar parámetros", width="stretch")

    if submitted:
        update_instance_parameters_bulk(instance_id, new_values)
        st.success(
            "Parámetros guardados en la instancia. Los auto-cálculos se actualizan "
            "al recargar la página."
        )
        st.rerun()


def render_danger_zone(instance_id: str) -> None:
    """Acciones destructivas sobre la instancia (eliminar)."""
    with st.expander("⚠️ Zona peligrosa", expanded=False):
        st.warning(
            "Eliminar la instancia borra todos sus parámetros y documentos "
            "asociados. Operación irreversible."
        )
        confirm = st.text_input(
            f"Para confirmar, escribí el ID de la instancia (`{instance_id}`):",
            key=f"confirm_delete_{instance_id}",
        )
        if st.button(
            "Eliminar instancia permanentemente",
            disabled=(confirm.strip() != instance_id),
            key=f"delete_btn_{instance_id}",
        ):
            ok = delete_instance(instance_id)
            if ok:
                st.session_state.pop("wm_active_instance_id", None)
                st.success(f"Instancia '{instance_id}' eliminada.")
                st.rerun()
            else:
                st.error("No se pudo eliminar la instancia.")


# ============================================================
# Ciclo 14a — GRID DE MÁQUINAS (cockpit)
# ============================================================

def _set_active_instance(target_instance_id: str) -> None:
    """
    Callback del botón "Activar" en cada card del grid.

    Hotfix 8: actualiza AMBAS keys porque están separadas:
    - 'wm_active_instance_id' es la key persistente que get_active_instance_id()
      lee desde otras páginas (no atada a ningún widget).
    - 'wm_instance_select_library' es la key del selectbox del sidebar de
      esta página específica; al setearla acá, el selectbox al
      re-renderizarse en el próximo cycle se posiciona en la nueva activa.

    Los callbacks corren en una fase pre-render donde session_state
    se puede modificar libremente — incluso keys de widgets ya
    instanciados en el cycle anterior.
    """
    st.session_state["wm_active_instance_id"] = target_instance_id
    st.session_state["wm_instance_select_library"] = target_instance_id


def render_machinery_grid() -> None:
    """
    Grilla de cards con todas las máquinas registradas. Cada card resume
    tag · driver · driven · cliente · sitio · cantidad de docs/parámetros.
    Click → activa esa instancia y dispara rerun.
    """
    instances = list_instances()
    if not instances:
        return

    st.markdown("### Máquinas registradas")
    st.caption(
        f"{len(instances)} máquina(s) en el sistema. "
        "Click en cualquier card para activarla en todos los módulos de análisis."
    )

    cards_per_row = 3
    rows = [instances[i:i + cards_per_row] for i in range(0, len(instances), cards_per_row)]
    for row in rows:
        cols = st.columns(cards_per_row)
        for idx, summary in enumerate(row):
            with cols[idx]:
                inst_id = summary.get("instance_id", "")
                inst = get_instance(inst_id)
                if inst is None:
                    continue

                tag = inst.tag or inst_id
                driver_part = " ".join(p for p in [inst.driver_manufacturer, inst.driver_model] if p) or "(sin driver)"
                driven_part = " ".join(p for p in [inst.driven_manufacturer, inst.driven_model] if p)
                client = inst.client or "(sin cliente)"
                site = inst.site or inst.location or ""
                n_docs = len(inst.documents)
                power_str = f"{inst.nominal_power_mw:.0f} MW" if inst.nominal_power_mw > 0 else ""
                rpm_str = f"{inst.nominal_rpm:.0f} rpm" if inst.nominal_rpm > 0 else ""

                # Card con el esquemático si existe
                with st.container(border=True):
                    if inst.schematic_png:
                        try:
                            png = get_instance_document_bytes(inst_id, inst.schematic_png)
                            if png:
                                st.image(png, use_container_width=True)
                        except Exception:
                            pass
                    st.markdown(f"**{tag}**")
                    st.caption(driver_part)
                    if driven_part:
                        st.caption(f"+ {driven_part}")
                    meta_bits = [b for b in [power_str, rpm_str] if b]
                    if meta_bits:
                        st.caption(" · ".join(meta_bits))
                    if client or site:
                        st.caption(" · ".join(p for p in [client, site] if p))
                    st.caption(f"📄 {n_docs} documento(s)")
                    # Ciclo 14a — badge claro de estado del esquemático.
                    # Le dice al usuario de un vistazo si esta maquina ya
                    # tiene el esquematico vinculado para que aparezca en
                    # el Resumen Ejecutivo del PDF.
                    if inst.schematic_png:
                        st.caption("🖼️ esquemático vinculado")
                    else:
                        st.caption("⚠️ sin esquemático principal")

                    # Indicador si esta es la activa
                    if st.session_state.get("wm_active_instance_id") == inst_id:
                        st.success("✓ activa", icon="🟢")
                    else:
                        # Usamos on_click callback porque la key
                        # 'wm_active_instance_id' ya está instanciada por el
                        # selectbox del sidebar; modificarla directo con
                        # st.session_state[...] = ... lanzaría
                        # 'cannot be modified after widget instantiated'.
                        # Los callbacks corren en una fase especial donde
                        # session_state se puede escribir libremente.
                        st.button(
                            "Activar",
                            key=f"activate_{inst_id}",
                            on_click=_set_active_instance,
                            args=(inst_id,),
                            width="stretch",
                        )

    st.markdown("---")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    require_login()
    render_user_menu()

    page_header(
        title="Machinery Library",
        subtitle="Cockpit central de máquinas — perfil técnico, esquemáticos, manuales OEM y parámetros físicos por instancia de activo.",
    )

    with st.sidebar:
        st.markdown("---")
        # Pasamos module_name="documents" (alias histórico) en vez de
        # "library" porque el sistema de profiles valida que el module_name
        # esté declarado en core/machine_profiles. La Library es universal
        # (toda máquina debe ser configurable acá), así que reusamos el
        # alias 'documents' que ya está aceptado por todos los profiles.
        state = render_instance_selector(module_name="documents")

    instance_id = state.get("instance_id")

    # Ciclo 14a — Grid de cards de TODAS las máquinas registradas (cockpit)
    render_machinery_grid()

    # Sección siempre visible: crear instancia nueva
    render_create_instance_section()

    if not instance_id:
        st.info(
            "No hay máquina activa. Creá una desde el formulario de arriba "
            "o seleccioná una desde el grid de máquinas (sidebar)."
        )
        return

    st.markdown("---")
    render_instance_header(state)

    st.markdown("---")
    render_sensor_map_section(instance_id)

    st.markdown("---")
    render_captured_parameters_section(instance_id)

    st.markdown("---")
    render_documents_section(instance_id)

    st.markdown("---")
    render_upload_section(instance_id)

    st.markdown("---")
    render_danger_zone(instance_id)


if __name__ == "__main__":
    main()
