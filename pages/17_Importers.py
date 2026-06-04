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
    page_title="Importadores & Plantillas — Watermelon",
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
    "Importadores & Plantillas LATAM",
    "Carga archivos de Emerson CSI 2140, Bently Nevada ADRE 408 o Universal File "
    "Format (.uff/.unv), o elige una plantilla pre-cargada de las 20 máquinas más "
    "comunes en O&G, generación y petroquímica LATAM.",
)


tab_import, tab_templates = st.tabs([
    "📥 Importar CSV (CSI 2140 / ADRE 408 / UFF)",
    "📚 Plantillas LATAM",
])


# =============================================================
# TAB 1 — Importadores universales
# =============================================================

with tab_import:
    st.subheader("Importar archivo de otro vendor")

    col1, col2 = st.columns([1, 2])

    with col1:
        vendor = st.selectbox(
            "Formato del archivo",
            options=["csi2140", "adre408", "uff"],
            format_func=lambda v: {
                "csi2140": "Emerson CSI 2140 (CSV)",
                "adre408": "Bently Nevada ADRE 408 (CSV)",
                "uff":     "Universal File Format (.uff/.unv)",
            }[v],
            key="importer_vendor",
        )

        st.caption({
            "csi2140": "Exports del Machinery Health Analyzer (AMS Suite). Tiempo o espectro.",
            "adre408": "Exports de ADREsoftware (precursor System1). Tiempo o espectro.",
            "uff":     "Estándar SDRC/IDEAS dataset 58 (ASCII). Time response o spectrum.",
        }[vendor])

    with col2:
        uploaded = st.file_uploader(
            "Sube tu archivo",
            type=["csv", "txt", "uff", "unv", "asc"],
            key="importer_upload",
            help="Tamaño máximo 200 MB. Múltiples archivos cargas uno por uno.",
        )

    if uploaded is not None:
        with st.spinner(f"Parseando {uploaded.name} como {vendor}..."):
            try:
                if vendor == "csi2140":
                    loaded = parse_csi2140(uploaded, file_name=uploaded.name)
                elif vendor == "adre408":
                    loaded = parse_adre408(uploaded, file_name=uploaded.name)
                else:
                    loaded = parse_uff(uploaded, file_name=uploaded.name)
                loaded.validate()
            except Exception as e:
                st.error(f"❌ Error parseando el archivo: {e}")
                st.stop()

        st.success(f"✅ Archivo parseado correctamente como {vendor}")

        # Resumen
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Muestras", f"{loaded.x.size:,}")
        m2.metric("Dominio", loaded.domain)
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
                xaxis_title="Tiempo (s)",
                yaxis_title=f"Amplitud ({loaded.units or '—'})",
                template="plotly_white",
                height=400,
                margin=dict(l=40, r=20, t=30, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)
            if loaded.x.size > 5000:
                st.caption(f"Mostrando primeros 5,000 puntos de {loaded.x.size:,} totales.")
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
                xaxis_title="Frecuencia (Hz)",
                yaxis_title=f"Amplitud ({loaded.units or '—'})",
                template="plotly_white",
                height=400,
                margin=dict(l=40, r=20, t=30, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)

        # Metadata expandida
        with st.expander("🔍 Metadata cruda parseada", expanded=False):
            md_display = {k: v for k, v in loaded.metadata.items() if k != "axis_freq_hz"}
            st.json(md_display)

        # Acción: convertir a Signal y guardar en session_state
        st.markdown("---")
        col_a, col_b = st.columns([1, 3])
        with col_a:
            do_inject = st.button(
                "➕ Cargar como Signal Watermelon",
                type="primary",
                use_container_width=True,
            )
        with col_b:
            st.caption(
                "Inyecta este archivo al `st.session_state['signals']` para que "
                "el resto de páginas (Spectrum, Trends, Orbit, etc.) lo vean "
                "igual que si hubiera sido cargado por el flujo nativo."
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
                    f"✅ Inyectado como '{key}'. Ya está disponible en las "
                    f"otras páginas (Spectrum, Time Waveforms, Trends, etc.)."
                )
            except Exception as e:
                st.error(f"❌ No se pudo inyectar al session_state: {e}")


# =============================================================
# TAB 2 — Plantillas LATAM
# =============================================================

with tab_templates:
    st.subheader("Catálogo de plantillas LATAM (20 máquinas)")
    st.caption(
        "Pre-carga de máquinas comunes en O&G, generación y petroquímica LATAM "
        "con rodamientos típicos, normas ISO/API recomendadas y esquema de "
        "sensores. Pensado para reducir el time-to-value de un activo nuevo."
    )

    cats = list_categories()
    col_a, col_b = st.columns([1, 3])
    with col_a:
        category_filter = st.selectbox(
            "Filtrar por categoría",
            options=["Todas"] + cats,
            key="template_cat_filter",
        )

    if category_filter == "Todas":
        templates = list_templates()
    else:
        templates = list_templates_by_category(category_filter)

    with col_b:
        template_options = {t.id: t.label for t in templates}
        selected_id = st.selectbox(
            "Plantilla",
            options=list(template_options.keys()),
            format_func=lambda i: template_options.get(i, i),
            key="template_picker",
        )

    if not selected_id:
        st.info("No hay plantillas en esta categoría.")
        st.stop()

    t = get_template(selected_id)
    if t is None:
        st.warning("Plantilla no encontrada.")
        st.stop()

    st.markdown(f"### {t.label}")

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Fabricante", t.manufacturer or "—")
    k2.metric("Categoría", t.category or "—")
    k3.metric("RPM nominal", f"{t.operating_rpm_nominal:,.0f}" if t.operating_rpm_nominal else "—")
    k4.metric("Tipo cojinete", t.bearing_type or "—")

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("**Aplicación**")
        st.write(", ".join(t.application) if t.application else "—")

        st.markdown("**Potencia (kW)**")
        if t.rated_power_kw and len(t.rated_power_kw) >= 2:
            st.write(f"{t.rated_power_kw[0]:,.0f} – {t.rated_power_kw[-1]:,.0f} kW")
        else:
            st.write("—")

        st.markdown("**Rango de RPM**")
        if t.operating_rpm_range and len(t.operating_rpm_range) == 2:
            st.write(f"{t.operating_rpm_range[0]:,.0f} – {t.operating_rpm_range[1]:,.0f}")
        else:
            st.write("—")

    with col_r:
        st.markdown("**Norma ISO recomendada**")
        st.write(f"{t.iso_norm_recommended} ({t.iso_class_recommended})" if t.iso_norm_recommended else "—")

        st.markdown("**Norma API recomendada**")
        st.write(t.api_norm_recommended or "—")

        st.markdown("**Rodamientos típicos**")
        st.write(", ".join(t.common_bearings) if t.common_bearings else "—")

    st.markdown("**Esquema de sensores recomendado**")
    if t.sensor_layout:
        st.json(t.sensor_layout)
    else:
        st.write("—")

    if t.notes:
        st.markdown("**Notas técnicas**")
        st.info(t.notes)

    # =========================================================
    # Ciclo 18.3 — Crear activo desde esta plantilla
    # =========================================================
    st.markdown("---")
    with st.expander("➕ Crear activo desde esta plantilla", expanded=False):
        from core.instance_state import create_instance, get_instance
        from core.machine_profiles import PROFILES as MACHINE_PROFILES

        st.caption(
            "Auto-rellena los campos con los valores de la plantilla. "
            "Podés editar todo antes de guardar — la plantilla es solo una "
            "sugerencia, no una camisa de fuerza."
        )

        suggested_profile = suggest_profile_key_for_template(t.id) or "custom_manual"
        all_profiles = sorted(MACHINE_PROFILES.keys())
        try:
            default_idx = all_profiles.index(suggested_profile)
        except ValueError:
            default_idx = 0

        # Notas pre-llenadas con la metadata clave de la plantilla
        prefilled_notes_lines = [
            f"Plantilla base: {t.label}",
            f"Fabricante: {t.manufacturer}",
            f"RPM nominal: {t.operating_rpm_nominal:,.0f}" if t.operating_rpm_nominal else "",
            f"Norma ISO recomendada: {t.iso_norm_recommended} ({t.iso_class_recommended})" if t.iso_norm_recommended else "",
            f"Norma API: {t.api_norm_recommended}" if t.api_norm_recommended else "",
            f"Rodamientos típicos: {', '.join(t.common_bearings)}" if t.common_bearings else "",
            "",
            t.notes or "",
        ]
        prefilled_notes = "\n".join(line for line in prefilled_notes_lines if line is not None)

        with st.form(f"create_from_template_{t.id}", clear_on_submit=False):
            col_a, col_b = st.columns(2)
            with col_a:
                inst_id_in = st.text_input(
                    "ID del activo (slug único)",
                    placeholder=f"{t.id}_planta_x",
                    help="Solo letras, números, guiones y guiones bajos.",
                )
                tag_in = st.text_input(
                    "Tag interno del cliente",
                    placeholder="C200C, Mars-1, etc.",
                )
            with col_b:
                profile_in = st.selectbox(
                    "Profile (familia técnica)",
                    options=all_profiles,
                    index=default_idx,
                    format_func=lambda pk: f"{MACHINE_PROFILES[pk].label}",
                    help="Sugerido a partir de la categoría y RPM de la plantilla. Editable.",
                )
                serial_in = st.text_input(
                    "Número de serie OEM",
                    placeholder="GE-12345-A",
                )

            location_in = st.text_input(
                "Ubicación física",
                placeholder="Ej: Planta La Belleza, Plato, Magdalena",
            )
            notes_in = st.text_area(
                "Notas",
                value=prefilled_notes,
                height=160,
            )

            submitted = st.form_submit_button(
                "✅ Crear activo",
                type="primary",
                use_container_width=True,
            )

            if submitted:
                clean_id = inst_id_in.strip()
                if not clean_id:
                    st.error("El ID es obligatorio.")
                elif get_instance(clean_id) is not None:
                    st.error(f"Ya existe un activo con ID '{clean_id}'. Elegí otro.")
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
                            f"✅ Activo '{inst.instance_id}' creado desde plantilla "
                            f"'{t.label}'. Profile: {profile_in}. "
                            f"Lo encontrás en Machinery Library."
                        )
                        st.session_state["wm_active_instance_id"] = inst.instance_id
                    except Exception as e:
                        st.error(f"No se pudo crear el activo: {e}")
