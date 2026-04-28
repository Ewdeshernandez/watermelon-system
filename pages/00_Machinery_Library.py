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
                    "Subí el esquemático PNG/JPG en la sección 'Cargar nuevo documento' "
                    "más abajo, con tipo='schematic'. Después seleccioná aquí cuál es el esquemático "
                    "principal del tren para que aparezca en el Resumen Ejecutivo del PDF."
                )
                schematic_options = [("", "(sin esquemático)")]
                for d in inst.documents:
                    if d.get("document_type", "").lower() in ("schematic", "esquematico", "diagram"):
                        schematic_options.append((d.get("id", ""), d.get("title") or d.get("filename") or "—"))
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

                    # Indicador si esta es la activa
                    if st.session_state.get("wm_active_instance_id") == inst_id:
                        st.success("✓ activa", icon="🟢")
                    else:
                        if st.button("Activar", key=f"activate_{inst_id}", width="stretch"):
                            st.session_state["wm_active_instance_id"] = inst_id
                            st.rerun()

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
    render_captured_parameters_section(instance_id)

    st.markdown("---")
    render_documents_section(instance_id)

    st.markdown("---")
    render_upload_section(instance_id)

    st.markdown("---")
    render_danger_zone(instance_id)


if __name__ == "__main__":
    main()
