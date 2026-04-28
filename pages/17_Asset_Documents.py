"""
pages/17_Asset_Documents.py
============================

Asset Documents — gestión de instancias de activos físicos. Cada
instancia (máquina física específica, ej. "TES1" del cliente
Atlántico) tiene su propio Vault con manuales, reportes y
parámetros físicos del cojinete.

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


st.set_page_config(page_title="Watermelon System | Asset Documents", layout="wide")
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
    """Cabecera con metadata editable de la instancia activa."""
    instance_id = state["instance_id"]
    profile_label = state["profile_label"]
    profile = get_profile(state["profile_key"])

    st.markdown(f"## {state.get('tag') or instance_id}")
    sub_parts = [profile_label]
    if profile:
        sub_parts.append(f"ISO {profile.iso_part}")
        sub_parts.append(f"{profile.operating_rpm:.0f} rpm nominal")
        sub_parts.append(profile.bearing_type)
    st.caption(" · ".join(sub_parts))
    if state.get("location"):
        st.caption(f"📍 {state['location']}")
    if state.get("serial_number"):
        st.caption(f"S/N: {state['serial_number']}")
    if state.get("notes"):
        with st.expander("Notas de la instancia", expanded=False):
            st.write(state["notes"])

    with st.expander("Editar metadata de esta instancia", expanded=False):
        with st.form(f"edit_header_{instance_id}"):
            c1, c2 = st.columns(2)
            with c1:
                new_tag = st.text_input("Tag", value=state.get("tag") or "")
                new_serial = st.text_input("Número de serie", value=state.get("serial_number") or "")
            with c2:
                new_loc = st.text_input("Ubicación", value=state.get("location") or "")
            new_notes = st.text_area("Notas", value=state.get("notes") or "", height=80)
            saved = st.form_submit_button("Actualizar metadata", width="stretch")
            if saved:
                update_instance_header(
                    instance_id,
                    tag=new_tag.strip(),
                    serial_number=new_serial.strip(),
                    location=new_loc.strip(),
                    notes=new_notes.strip(),
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
# MAIN
# ============================================================

def main() -> None:
    require_login()
    render_user_menu()

    page_header(
        title="Asset Documents",
        subtitle="Vault de manuales, reportes y parámetros físicos por instancia de activo.",
    )

    with st.sidebar:
        st.markdown("---")
        state = render_instance_selector(module_name="documents")

    instance_id = state.get("instance_id")

    # Sección siempre visible: crear instancia nueva (al tope para que sea accesible)
    render_create_instance_section()

    if not instance_id:
        st.info(
            "No hay instancia activa. Creá una desde el formulario de arriba "
            "o seleccioná una existente desde la sidebar."
        )
        # Debug visual: cuántas instancias hay
        n = len(list_instances())
        if n:
            st.caption(f"Hay {n} instancia(s) registrada(s).")
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
