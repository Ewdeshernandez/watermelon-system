"""
pages/17_Asset_Documents.py
============================

Document Vault — gestión de manuales, reportes y especificaciones por
asset profile. Permite cargar PDFs/imágenes/documentos asociados a la
máquina activa, listarlos, descargarlos, eliminarlos, y capturar los
parámetros técnicos clave (dimensiones del cojinete, propiedades del
aceite, umbrales del fabricante) que después alimentan los módulos de
análisis (Shaft Centerline, Polar, Bode).

Filosofía: la app no es solo un visor de datos vibratorios — es un
asset management platform donde cada máquina lleva consigo su
documentación técnica, su histórico de mantenimiento y sus parámetros
de referencia, todo conectado al perfil ISO/API correspondiente.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from core.auth import require_login, render_user_menu
from core.document_vault import (
    CAPTURED_PARAMETER_FIELDS,
    DOCUMENT_TYPES,
    add_document,
    delete_document,
    estimate_diametral_clearance_mm,
    get_captured_parameters,
    get_document_bytes,
    list_documents,
    update_captured_parameters_bulk,
)
from core.machine_profiles import get_profile
from core.profile_state import render_profile_selector
from core.ui_theme import apply_watermelon_page_style, page_header


st.set_page_config(page_title="Watermelon System | Asset Documents", layout="wide")
require_login()
apply_watermelon_page_style()


# ============================================================
# HELPERS UI
# ============================================================

def _bytes_to_human(n: int) -> str:
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

def render_documents_section(profile_key: str) -> None:
    """Lista de documentos cargados con acciones de download/delete."""
    docs = list_documents(profile_key)

    st.markdown("### Documentos del activo")
    if not docs:
        st.info(
            "Este activo aún no tiene documentos cargados. Sube manuales OEM, "
            "reportes de mantenimiento, certificados o especificaciones desde "
            "la sección 'Cargar nuevo documento' más abajo."
        )
        return

    df = pd.DataFrame([
        {
            "Origen": "🔒 Fábrica" if d.get("is_seed") else "👤 Usuario",
            "Título": d.get("title", d.get("filename", "—")),
            "Tipo": DOCUMENT_TYPES.get(
                d.get("document_type") or d.get("type", "other"),
                d.get("document_type") or d.get("type", "—"),
            ),
            "Archivo": d.get("filename", "—"),
            "Tamaño": _bytes_to_human(int(d.get("size_bytes", 0))) if d.get("size_bytes") else "—",
            "Subido": _format_date(d.get("uploaded_at", "")),
            "Tags": ", ".join(d.get("tags", [])) if d.get("tags") else "—",
        }
        for d in docs
    ])
    st.dataframe(df, width="stretch", hide_index=True)

    st.markdown("**Acciones**")
    for d in docs:
        is_seed = bool(d.get("is_seed"))
        title_prefix = "🔒 " if is_seed else "📄 "
        with st.expander(f"{title_prefix}{d.get('title') or d.get('filename')}"):
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                if is_seed:
                    st.caption(
                        "**Documento de fábrica** — viene committeado al repo "
                        "y sobrevive cualquier reinicio del despliegue. No "
                        "puede borrarse desde la UI; solo el equipo de "
                        "ingeniería puede modificarlo o reemplazarlo en el "
                        "código fuente."
                    )
                st.caption(d.get("description", "") or "_(sin descripción)_")
                if d.get("tags"):
                    st.caption("Tags: " + ", ".join(d["tags"]))
                st.caption(f"ID interno: `{d.get('id')}`")

            with col2:
                file_bytes = get_document_bytes(profile_key, d["id"])
                if file_bytes is not None:
                    st.download_button(
                        "Descargar",
                        data=file_bytes,
                        file_name=d.get("filename", "document"),
                        key=f"dl_{d['id']}",
                        width="stretch",
                    )
                else:
                    st.caption("Archivo no disponible")

            with col3:
                if is_seed:
                    st.button(
                        "🔒 Permanente",
                        key=f"del_{d['id']}",
                        width="stretch",
                        disabled=True,
                        help="Los documentos de fábrica son permanentes y no "
                             "pueden borrarse desde la UI.",
                    )
                else:
                    if st.button("Eliminar", key=f"del_{d['id']}", width="stretch"):
                        delete_document(profile_key, d["id"])
                        st.success(f"Documento '{d.get('title')}' eliminado.")
                        st.rerun()


def render_upload_section(profile_key: str) -> None:
    """Formulario de upload de nuevo documento."""
    st.markdown("### Cargar nuevo documento")

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
            doc_type_label = st.selectbox(
                "Tipo de documento",
                options=list(DOCUMENT_TYPES.values()),
                index=list(DOCUMENT_TYPES.keys()).index("bearing_repair_report")
                if "bearing_repair_report" in DOCUMENT_TYPES else 0,
            )
            doc_type = doc_type_label_to_key.get(doc_type_label, "other")

        with col2:
            doc_description = st.text_area(
                "Descripción / contexto",
                placeholder="Inspección y rebabbitado por Talleres Wersin SAS, oct 2018, "
                "cliente Proelectrica Cartagena. Material babbitt ASTM B-23 Grado 2. "
                "Resultado: 0% sin adherencia, todos aceptados clase B2.",
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

        try:
            tags = [t.strip() for t in doc_tags_str.split(",") if t.strip()]
            doc_id = add_document(
                profile_key,
                uploaded_file,
                title=doc_title or uploaded_file.name,
                document_type=doc_type,
                description=doc_description,
                tags=tags,
            )
            st.success(
                f"Documento '{uploaded_file.name}' cargado correctamente. "
                f"ID: `{doc_id}`"
            )
            st.rerun()
        except Exception as e:
            st.error(f"Error al cargar: {e}")


def render_captured_parameters_section(profile_key: str) -> None:
    """Formulario de captura/edición de parámetros estructurados del activo."""
    st.markdown("### Parámetros técnicos del activo")
    st.caption(
        "Captura los parámetros del activo extraídos de los documentos cargados "
        "o suministrados directamente por el fabricante. Estos valores alimentan "
        "los módulos de análisis (Shaft Centerline, Polar, Bode) cuando requieren "
        "datos físicos específicos."
    )

    current_values = get_captured_parameters(profile_key)

    # Agrupar campos por categoría
    by_category: Dict[str, List[tuple]] = {}
    for field_key, field_def in CAPTURED_PARAMETER_FIELDS.items():
        cat = field_def.get("category", "Otros")
        by_category.setdefault(cat, []).append((field_key, field_def))

    # Render formulario por categoría
    new_values: Dict[str, Any] = {}

    with st.form("captured_params_form"):
        for category in sorted(by_category.keys()):
            with st.expander(f"📋 {category}", expanded=(category in ("Cojinete - geometría", "Identificación"))):
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
                                key=f"param_{field_key}",
                                help=help_text,
                            )
                            if raw.strip() == "":
                                new_values[field_key] = None
                            else:
                                try:
                                    new_values[field_key] = float(raw.replace(",", "."))
                                except ValueError:
                                    new_values[field_key] = current  # mantener valor previo
                        elif ftype == "date":
                            raw_date = st.text_input(
                                label,
                                value=str(current) if current is not None else "",
                                key=f"param_{field_key}",
                                placeholder="YYYY-MM-DD",
                                help=help_text or "Formato YYYY-MM-DD",
                            )
                            new_values[field_key] = raw_date.strip() if raw_date.strip() else None
                        elif ftype == "text":
                            raw = st.text_area(
                                label,
                                value=str(current) if current is not None else "",
                                key=f"param_{field_key}",
                                help=help_text,
                            )
                            new_values[field_key] = raw if raw.strip() else None
                        else:  # str
                            raw = st.text_input(
                                label,
                                value=str(current) if current is not None else "",
                                key=f"param_{field_key}",
                                help=help_text,
                            )
                            new_values[field_key] = raw if raw.strip() else None

        submitted = st.form_submit_button("Guardar parámetros", width="stretch")

    if submitted:
        update_captured_parameters_bulk(profile_key, new_values)
        st.success("Parámetros guardados.")
        st.rerun()

    # Helper de estimación de clearance si tiene diámetro pero no clearance
    diam = current_values.get("bearing_inner_diameter_mm")
    cd = current_values.get("diametral_clearance_mm")
    if diam and not cd:
        st.markdown("#### Sugerencia automática")
        cd_est_typ = estimate_diametral_clearance_mm(float(diam), severity="typical")
        cd_est_tight = estimate_diametral_clearance_mm(float(diam), severity="tight")
        cd_est_loose = estimate_diametral_clearance_mm(float(diam), severity="loose")
        st.info(
            f"Tienes el diámetro interno ({diam} mm) pero no el clearance diametral. "
            f"Estimaciones heurísticas para cojinete plano hidrodinámico: "
            f"**tight** Cd ≈ {cd_est_tight:.3f} mm ({cd_est_tight*1000:.0f} µm), "
            f"**typical** Cd ≈ {cd_est_typ:.3f} mm ({cd_est_typ*1000:.0f} µm), "
            f"**loose** Cd ≈ {cd_est_loose:.3f} mm ({cd_est_loose*1000:.0f} µm). "
            f"Usa el dato del fabricante si está disponible (Brush, OEM manual)."
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    require_login()
    render_user_menu()

    page_header(
        title="Asset Documents",
        subtitle="Vault de manuales, reportes y especificaciones por máquina monitoreada.",
    )

    with st.sidebar:
        st.markdown("---")
        profile_state = render_profile_selector(module_name="documents")

    profile_key = profile_state["profile_key"]
    profile_label = profile_state["profile_label"]
    profile = get_profile(profile_key)

    st.markdown(f"## {profile_label}")
    st.caption(
        f"Categoría: {profile.category} · ISO {profile.iso_part} · "
        f"Operación nominal: {profile.operating_rpm:.0f} rpm · "
        f"Cojinete: {profile.bearing_type}"
    )
    if profile.notes:
        st.caption(profile.notes)

    st.markdown("---")
    render_documents_section(profile_key)

    st.markdown("---")
    render_upload_section(profile_key)

    st.markdown("---")
    render_captured_parameters_section(profile_key)


if __name__ == "__main__":
    main()
