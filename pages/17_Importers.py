"""
pages/17_Importers.py
=====================

Hub de importadores universales y plantillas de máquinas LATAM.

Ciclo 18.2: enchufa a la UI las capacidades que el Ciclo 18.1 dejó
listas en backend (core/loaders/* + core/machine_templates).

Esta página es 100% NUEVA — no modifica ninguna página existente.
Si algo se rompe, basta con borrar este archivo y la app vuelve al
estado v3.14.0 sin secuelas.

Dos tabs:
  - 📥 Importar CSV  : sube un archivo CSI 2140 / ADRE 408 / UFF y
                       muestra preview + metadata + opción de pasarlo
                       a session_state["signals"] como Watermelon Signal.
  - 📚 Plantillas LATAM : navega las 20 plantillas pre-cargadas con
                          rodamientos, normas ISO/API, sensores
                          recomendados. Útil para crear un activo
                          nuevo en Machinery Library con un click.
"""

from __future__ import annotations

# v3.31.243 — set_page_config primero, sino el sidebar pierde estilos.
import streamlit as st
st.set_page_config(
    page_title="Importers & Templates — Watermelon",
    page_icon="📥",
    layout="wide",
)

from core.auth import require_login, render_user_menu, require_role

require_login()
render_user_menu()
require_role(allowed_roles=("admin", "specialist"))


import io
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from core.loaders.adre408 import parse_adre408
from core.loaders.base import LoadedSignal, loaded_to_signal
from core.loaders.csi2140 import parse_csi2140
from core.loaders.uff import parse_uff
from core.machine_templates import (
    get_template,
    list_categories,
    list_templates,
    list_templates_by_category,
    suggest_profile_key_for_template,
)

from core.ui_theme import page_header as _wm_page_header  # hero compartido (v3.31.313)
_wm_page_header(
    "Importers & LATAM Templates",
    "Load Emerson CSI 2140, Bently Nevada ADRE 408 or Universal File "
    "Format (.uff/.unv) files, or choose a pre-loaded template from the 20 most "
    "common machines in LATAM O&G, power generation and petrochemicals.",
)


tab_import, tab_templates = st.tabs([
    "📥 Import CSV (CSI 2140 / ADRE 408 / UFF)",
    "📚 LATAM Templates",
])


# =============================================================
# TAB 1 — Importadores universales
# =============================================================

with tab_import:
    st.subheader("Import a file from another vendor")

    col1, col2 = st.columns([1, 2])

    with col1:
        vendor = st.selectbox(
            "File format",
            options=["csi2140", "adre408", "uff"],
            format_func=lambda v: {
                "csi2140": "Emerson CSI 2140 (CSV)",
                "adre408": "Bently Nevada ADRE 408 (CSV)",
                "uff":     "Universal File Format (.uff/.unv)",
            }[v],
            key="importer_vendor",
        )

        st.caption({
            "csi2140": "Machinery Health Analyzer exports (AMS Suite). Time or spectrum.",
            "adre408": "ADREsoftware exports (System1 predecessor). Time or spectrum.",
            "uff":     "SDRC/IDEAS dataset 58 standard (ASCII). Time response or spectrum.",
        }[vendor])

    with col2:
        uploaded = st.file_uploader(
            "Upload your file",
            type=["csv", "txt", "uff", "unv", "asc"],
            key="importer_upload",
            help="Maximum size 200 MB. Load multiple files one at a time.",
        )

    if uploaded is not None:
        with st.spinner(f"Parsing {uploaded.name} as {vendor}..."):
            try:
                if vendor == "csi2140":
                    loaded = parse_csi2140(uploaded, file_name=uploaded.name)
                elif vendor == "adre408":
                    loaded = parse_adre408(uploaded, file_name=uploaded.name)
                else:
                    loaded = parse_uff(uploaded, file_name=uploaded.name)
                loaded.validate()
            except Exception as e:
                st.error(f"❌ Error parsing the file: {e}")
                st.stop()

        st.success(f"✅ File parsed successfully as {vendor}")

        # Resumen
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Samples", f"{loaded.x.size:,}")
        m2.metric("Domain", loaded.domain)
        m3.metric("fs (Hz)", f"{loaded.fs:.1f}" if loaded.fs else "—")
        m4.metric("RPM", f"{loaded.rpm:.0f}" if loaded.rpm else "—")

        # Preview gráfico
        st.markdown("#### Preview")
        if loaded.domain == "time" and loaded.time is not None:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=loaded.time[:5000],   # cap para preview rápido
                y=loaded.x[:5000],
                mode="lines",
                line=dict(width=1),
                name=loaded.file_name,
            ))
            fig.update_layout(
                xaxis_title="Time (s)",
                yaxis_title=f"Amplitude ({loaded.units or '—'})",
                template="plotly_white",
                height=400,
                margin=dict(l=40, r=20, t=30, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)
            if loaded.x.size > 5000:
                st.caption(f"Showing first 5,000 points of {loaded.x.size:,} total.")
        elif loaded.domain == "spectrum":
            freq_axis = loaded.metadata.get("axis_freq_hz", [])
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(freq_axis),
                y=loaded.x.tolist(),
                mode="lines",
                line=dict(width=1),
            ))
            fig.update_layout(
                xaxis_title="Frequency (Hz)",
                yaxis_title=f"Amplitude ({loaded.units or '—'})",
                template="plotly_white",
                height=400,
                margin=dict(l=40, r=20, t=30, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)

        # Metadata expandida
        with st.expander("🔍 Parsed raw metadata", expanded=False):
            md_display = {k: v for k, v in loaded.metadata.items() if k != "axis_freq_hz"}
            st.json(md_display)

        # Acción: convertir a Signal y guardar en session_state
        st.markdown("---")
        col_a, col_b = st.columns([1, 3])
        with col_a:
            do_inject = st.button(
                "➕ Load as Watermelon Signal",
                type="primary",
                use_container_width=True,
            )
        with col_b:
            st.caption(
                "Injects this file into `st.session_state['signals']` so the "
                "rest of the pages (Spectrum, Trends, Orbit, etc.) see it just "
                "as if it had been loaded through the native flow."
            )

        if do_inject:
            try:
                signal = loaded_to_signal(loaded)
                signals = st.session_state.get("signals", {}) or {}
                # Evitar pisar — sufijo si ya existe
                key = loaded.file_name
                idx = 1
                while key in signals:
                    idx += 1
                    key = f"{loaded.file_name} ({idx})"
                signals[key] = signal
                st.session_state["signals"] = signals
                st.success(
                    f"✅ Injected as '{key}'. It is now available on the "
                    f"other pages (Spectrum, Time Waveforms, Trends, etc.)."
                )
            except Exception as e:
                st.error(f"❌ Could not inject into session_state: {e}")


# =============================================================
# TAB 2 — Plantillas LATAM
# =============================================================

with tab_templates:
    st.subheader("LATAM template catalog (20 machines)")
    st.caption(
        "Pre-loaded common machines from LATAM O&G, power generation and "
        "petrochemicals with typical bearings, recommended ISO/API standards "
        "and sensor layout. Designed to reduce the time-to-value of a new asset."
    )

    cats = list_categories()
    col_a, col_b = st.columns([1, 3])
    with col_a:
        category_filter = st.selectbox(
            "Filter by category",
            options=["All"] + cats,
            key="template_cat_filter",
        )

    if category_filter == "All":
        templates = list_templates()
    else:
        templates = list_templates_by_category(category_filter)

    with col_b:
        template_options = {t.id: t.label for t in templates}
        selected_id = st.selectbox(
            "Template",
            options=list(template_options.keys()),
            format_func=lambda i: template_options.get(i, i),
            key="template_picker",
        )

    if not selected_id:
        st.info("No templates in this category.")
        st.stop()

    t = get_template(selected_id)
    if t is None:
        st.warning("Template not found.")
        st.stop()

    st.markdown(f"### {t.label}")

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Manufacturer", t.manufacturer or "—")
    k2.metric("Category", t.category or "—")
    k3.metric("Nominal RPM", f"{t.operating_rpm_nominal:,.0f}" if t.operating_rpm_nominal else "—")
    k4.metric("Bearing type", t.bearing_type or "—")

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("**Application**")
        st.write(", ".join(t.application) if t.application else "—")

        st.markdown("**Power (kW)**")
        if t.rated_power_kw and len(t.rated_power_kw) >= 2:
            st.write(f"{t.rated_power_kw[0]:,.0f} – {t.rated_power_kw[-1]:,.0f} kW")
        else:
            st.write("—")

        st.markdown("**RPM range**")
        if t.operating_rpm_range and len(t.operating_rpm_range) == 2:
            st.write(f"{t.operating_rpm_range[0]:,.0f} – {t.operating_rpm_range[1]:,.0f}")
        else:
            st.write("—")

    with col_r:
        st.markdown("**Recommended ISO standard**")
        st.write(f"{t.iso_norm_recommended} ({t.iso_class_recommended})" if t.iso_norm_recommended else "—")

        st.markdown("**Recommended API standard**")
        st.write(t.api_norm_recommended or "—")

        st.markdown("**Typical bearings**")
        st.write(", ".join(t.common_bearings) if t.common_bearings else "—")

    st.markdown("**Recommended sensor layout**")
    if t.sensor_layout:
        st.json(t.sensor_layout)
    else:
        st.write("—")

    if t.notes:
        st.markdown("**Technical notes**")
        st.info(t.notes)

    # =========================================================
    # Ciclo 18.3 — Crear activo desde esta plantilla
    # =========================================================
    st.markdown("---")
    with st.expander("➕ Create asset from this template", expanded=False):
        from core.instance_state import create_instance, get_instance
        from core.machine_profiles import PROFILES as MACHINE_PROFILES

        st.caption(
            "Auto-fills the fields with the template values. "
            "You can edit everything before saving — the template is only a "
            "suggestion, not a straitjacket."
        )

        suggested_profile = suggest_profile_key_for_template(t.id) or "custom_manual"
        all_profiles = sorted(MACHINE_PROFILES.keys())
        try:
            default_idx = all_profiles.index(suggested_profile)
        except ValueError:
            default_idx = 0

        # Notas pre-llenadas con la metadata clave de la plantilla
        prefilled_notes_lines = [
            f"Base template: {t.label}",
            f"Manufacturer: {t.manufacturer}",
            f"Nominal RPM: {t.operating_rpm_nominal:,.0f}" if t.operating_rpm_nominal else "",
            f"Recommended ISO standard: {t.iso_norm_recommended} ({t.iso_class_recommended})" if t.iso_norm_recommended else "",
            f"API standard: {t.api_norm_recommended}" if t.api_norm_recommended else "",
            f"Typical bearings: {', '.join(t.common_bearings)}" if t.common_bearings else "",
            "",
            t.notes or "",
        ]
        prefilled_notes = "\n".join(line for line in prefilled_notes_lines if line is not None)

        with st.form(f"create_from_template_{t.id}", clear_on_submit=False):
            col_a, col_b = st.columns(2)
            with col_a:
                inst_id_in = st.text_input(
                    "Asset ID (unique slug)",
                    placeholder=f"{t.id}_plant_x",
                    help="Only letters, numbers, hyphens and underscores.",
                )
                tag_in = st.text_input(
                    "Client internal tag",
                    placeholder="C200C, Mars-1, etc.",
                )
            with col_b:
                profile_in = st.selectbox(
                    "Profile (technical family)",
                    options=all_profiles,
                    index=default_idx,
                    format_func=lambda pk: f"{MACHINE_PROFILES[pk].label}",
                    help="Suggested from the template's category and RPM. Editable.",
                )
                serial_in = st.text_input(
                    "OEM serial number",
                    placeholder="GE-12345-A",
                )

            location_in = st.text_input(
                "Physical location",
                placeholder="e.g. Planta La Belleza, Plato, Magdalena",
            )
            notes_in = st.text_area(
                "Notes",
                value=prefilled_notes,
                height=160,
            )

            submitted = st.form_submit_button(
                "✅ Create asset",
                type="primary",
                use_container_width=True,
            )

            if submitted:
                clean_id = inst_id_in.strip()
                if not clean_id:
                    st.error("The ID is required.")
                elif get_instance(clean_id) is not None:
                    st.error(f"An asset with ID '{clean_id}' already exists. Choose another.")
                else:
                    try:
                        inst = create_instance(
                            instance_id=clean_id,
                            profile_key=profile_in,
                            tag=tag_in.strip(),
                            serial_number=serial_in.strip(),
                            location=location_in.strip(),
                            notes=notes_in.strip(),
                            seed_from_profile=True,
                        )
                        st.success(
                            f"✅ Asset '{inst.instance_id}' created from template "
                            f"'{t.label}'. Profile: {profile_in}. "
                            f"You can find it in Machinery Library."
                        )
                        st.session_state["wm_active_instance_id"] = inst.instance_id
                    except Exception as e:
                        st.error(f"Could not create the asset: {e}")
