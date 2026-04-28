"""
core.profile_state
==================

Helper compartido entre módulos (Polar, Bode, SCL, etc.) para gestionar
la selección de Asset Profile. Persiste el profile activo en
st.session_state para que el usuario lo elija una vez y aplique a toda
la sesión.

Uso típico en una página:

    from core.profile_state import render_profile_selector

    with st.sidebar:
        profile_state = render_profile_selector(module_name="polar")

    # profile_state contiene:
    #   - operating_rpm: float
    #   - iso_part: str
    #   - machine_group: str
    #   - custom_thresholds: Optional[Tuple[float, float, float]]
    #   - profile_label: str
    #   - is_applicable: bool (False si el profile no aplica al módulo)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import streamlit as st

from core.machine_profiles import (
    DEFAULT_PROFILE_KEY,
    PROFILES,
    get_profile,
    is_module_applicable,
    list_profile_options,
    module_not_applicable_message,
)


SESSION_KEY = "wm_active_profile_key"
CUSTOM_RPM_KEY = "wm_active_profile_custom_rpm"
CUSTOM_GROUP_KEY = "wm_active_profile_custom_group"
CUSTOM_AB_KEY = "wm_active_profile_custom_ab"
CUSTOM_BC_KEY = "wm_active_profile_custom_bc"
CUSTOM_CD_KEY = "wm_active_profile_custom_cd"
THRESHOLD_SOURCE_KEY = "wm_active_profile_threshold_source"
OEM_AB_KEY = "wm_active_profile_oem_ab"
OEM_BC_KEY = "wm_active_profile_oem_bc"
OEM_CD_KEY = "wm_active_profile_oem_cd"


def render_profile_selector(module_name: str = "polar") -> Dict[str, Any]:
    """
    Renderiza la sección "Asset Profile" en la sidebar y retorna el
    estado activo. Llamar dentro de un bloque ``with st.sidebar:``.

    Args:
        module_name: nombre del módulo actual ("polar", "bode",
            "shaft_centerline", "spectrum", "trends", etc.)
            Se usa para evaluar si el profile elegido es aplicable.

    Returns:
        Dict con campos:
            - operating_rpm (float)
            - iso_part (str)
            - machine_group (str)
            - measurement_type (str)
            - custom_thresholds (Optional[Tuple[float,float,float]])
            - threshold_source ("iso" / "oem" / "data_baseline" / "custom")
            - profile_key, profile_label
            - is_applicable (bool)
            - applicability_message (str, vacío si aplica)
            - profile_notes (str)
    """
    options = list_profile_options()

    # Inicializar default
    if SESSION_KEY not in st.session_state:
        st.session_state[SESSION_KEY] = DEFAULT_PROFILE_KEY

    st.markdown("### Asset Profile")
    label_to_key = {opt["label"]: opt["key"] for opt in options}
    labels = list(label_to_key.keys())
    current_key = st.session_state[SESSION_KEY]
    current_label = next(
        (lbl for lbl, k in label_to_key.items() if k == current_key),
        labels[0],
    )

    selected_label = st.selectbox(
        "Active machine profile",
        options=labels,
        index=labels.index(current_label) if current_label in labels else 0,
        key="wm_active_profile_selectbox",
        help=(
            "Define la máquina que estás analizando. Determina la velocidad "
            "operativa nominal, la parte ISO 20816 aplicable, el grupo/clase "
            "de la norma y los módulos donde el análisis es técnicamente "
            "relevante. Persiste entre páginas."
        ),
    )
    selected_key = label_to_key[selected_label]
    st.session_state[SESSION_KEY] = selected_key

    profile = get_profile(selected_key)

    # Mostrar info del profile
    st.caption(f"**Categoría:** {profile.category}")
    st.caption(f"**ISO part:** {profile.iso_part} · **Grupo:** {profile.machine_group}")
    st.caption(
        f"**Velocidad operativa default:** {profile.operating_rpm:.0f} rpm · "
        f"**Cojinete:** {profile.bearing_type}"
    )
    if profile.notes:
        st.caption(profile.notes)

    # ---------------------------------------------------------
    # Modo Custom: campos manuales
    # ---------------------------------------------------------
    operating_rpm = profile.operating_rpm
    iso_part = profile.iso_part
    machine_group = profile.machine_group
    custom_thresholds: Optional[Tuple[float, float, float]] = None
    threshold_source = "iso"

    if profile.key == "custom_manual":
        st.markdown("**Configuración manual**")

        operating_rpm = st.number_input(
            "Operating RPM",
            min_value=10.0,
            max_value=60000.0,
            value=float(st.session_state.get(CUSTOM_RPM_KEY, profile.operating_rpm)),
            step=50.0,
            key=CUSTOM_RPM_KEY,
        )

        iso_part = st.selectbox(
            "ISO 20816 part",
            options=["20816-2", "20816-3", "20816-4", "20816-7", "custom"],
            index=0,
            key="wm_custom_iso_part_select",
        )

        machine_group = st.text_input(
            "Machine group / class",
            value=st.session_state.get(CUSTOM_GROUP_KEY, "group2"),
            key=CUSTOM_GROUP_KEY,
            help="group1 / group2 (parte 2), class_iv (parte 3), rolling_aero (parte 4), category_i/ii (parte 7) o lo que aplique.",
        )

        threshold_source = st.selectbox(
            "Threshold source",
            options=["iso", "oem", "custom"],
            index=0,
            key=THRESHOLD_SOURCE_KEY,
            help=(
                "iso: usar las tablas de la norma seleccionada arriba. "
                "oem: ingresar los umbrales del fabricante (Brush, Siemens, GE…). "
                "custom: definir umbrales A/B, B/C, C/D libremente."
            ),
        )

        if threshold_source in ("oem", "custom"):
            st.markdown(f"**Umbrales {threshold_source.upper()} (en µm pp para shaft displacement, mm/s RMS para casing)**")
            ab = st.number_input(
                "A/B threshold",
                min_value=0.0,
                value=float(st.session_state.get(OEM_AB_KEY, 75.0)),
                step=5.0,
                key=OEM_AB_KEY,
            )
            bc = st.number_input(
                "B/C threshold",
                min_value=0.0,
                value=float(st.session_state.get(OEM_BC_KEY, 150.0)),
                step=5.0,
                key=OEM_BC_KEY,
            )
            cd = st.number_input(
                "C/D threshold",
                min_value=0.0,
                value=float(st.session_state.get(OEM_CD_KEY, 240.0)),
                step=5.0,
                key=OEM_CD_KEY,
            )
            if ab > 0 and bc > ab and cd > bc:
                custom_thresholds = (ab, bc, cd)

    # ---------------------------------------------------------
    # Aplicabilidad al módulo
    # ---------------------------------------------------------
    is_applicable = is_module_applicable(selected_key, module_name)
    applicability_message = ""
    if not is_applicable:
        applicability_message = module_not_applicable_message(selected_key, module_name)

    # measurement_type por defecto según parte ISO
    if iso_part in ("20816-4", "20816-7"):
        measurement_type = "casing_velocity"
    else:
        measurement_type = "shaft_displacement"

    return {
        "profile_key": selected_key,
        "profile_label": profile.label,
        "profile_notes": profile.notes,
        "category": profile.category,
        "operating_rpm": float(operating_rpm),
        "iso_part": iso_part,
        "machine_group": machine_group,
        "measurement_type": measurement_type,
        "custom_thresholds": custom_thresholds,
        "threshold_source": threshold_source,
        "is_applicable": is_applicable,
        "applicability_message": applicability_message,
        "applicable_modules": profile.applicable_modules,
    }


__all__ = [
    "render_profile_selector",
    "SESSION_KEY",
]
