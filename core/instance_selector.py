"""
core.instance_selector
======================

UI helper de Streamlit para que cada módulo (Polar, Bode, SCL,
Asset Documents) seleccione la **instancia activa** (máquina física
específica) con la que va a trabajar. Antes el sistema solo elegía
profile (familia/tipo de máquina), lo que mezclaba data entre
máquinas físicamente distintas del mismo modelo.

Funcionamiento:

1. Lista todas las instances registradas (core/instance_state).
2. Si no hay ninguna pero existen profiles con seed (Ciclo 7), ofrece
   crear una "instance por defecto" automática para el primer profile
   con seed (típicamente "brush_tes1" desde el seed del Brush).
3. Permite crear instancias nuevas inline (formulario expandible).
4. Persiste la selección en st.session_state para que sobreviva entre
   navegación de páginas.
5. Devuelve un dict con toda la info que el módulo necesita:
       {
         "instance_id":         str,
         "instance_label":      str,    # legible para mostrar al usuario
         "profile_key":         str,
         "profile_label":       str,
         "operating_rpm":       float,
         "iso_part":            str,
         "machine_group":       str,
         "is_applicable":       bool,
         "applicability_message": str,
         "tag":                 str,
         "serial_number":       str,
         "location":            str,
         "captured_parameters": dict,
         "documents":           list,
       }

Para mantener compatibilidad con módulos que aún esperan el formato
de profile_state.render_profile_selector(), todos los campos del
selector original están presentes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import streamlit as st

from core.instance_state import (
    list_instances,
    get_instance,
    get_instance_parameters,
    create_instance,
    get_active_backend_name,
)
from core.machine_profiles import (
    PROFILES as MACHINE_PROFILES,
    get_profile,
    module_not_applicable_message,
)
from core.vault_seeds import VAULT_SEEDS, has_seed


SESSION_KEY_INSTANCE = "wm_active_instance_id"
SESSION_KEY_PROFILE = "wm_active_profile_key"  # legacy, mantenido para compat


# =============================================================
# AUTO-BOOTSTRAP DE INSTANCIAS DESDE SEEDS
# =============================================================

def _suggest_instance_id_for_profile(profile_key: str) -> str:
    """
    Sugiere un instance_id legible derivado del profile_key.
    Ejemplo:
        brush_turbogenerator_54mw_3600 → brush_default
        siemens_sgt300_3600            → siemens_sgt300_default
    """
    base = profile_key.split("_")[0]
    return f"{base}_default"


def _ensure_seeded_instances_exist() -> None:
    """
    Para cada profile_key con semilla en core/vault_seeds.py, garantiza
    que exista la instance "por defecto" sugerida (ej. brush_default).
    No verifica si hay otras instances del mismo profile — solo crea
    la default si la suggested_id específica falta. Esto permite que el
    usuario tenga su instance custom (ej. brush_tes1) y al mismo tiempo
    la default automática del seed.
    """
    for profile_key in VAULT_SEEDS.keys():
        suggested_id = _suggest_instance_id_for_profile(profile_key)
        if get_instance(suggested_id) is not None:
            continue
        profile = get_profile(profile_key)
        prof_label = profile.label if profile else profile_key
        try:
            create_instance(
                instance_id=suggested_id,
                profile_key=profile_key,
                tag="(default)",
                serial_number="",
                location="",
                notes=(
                    f"Instance por defecto auto-creada desde el seed del profile "
                    f"'{prof_label}' (Ciclo 7). Edita los datos para reflejar "
                    f"la máquina física real, o creá una instance nueva con tag "
                    f"propio (ej. TES1) y dejá esta como referencia."
                ),
                seed_from_profile=True,
            )
        except Exception:
            continue


# =============================================================
# SELECTOR PRINCIPAL
# =============================================================

def render_instance_selector(module_name: str = "module") -> Dict[str, Any]:
    """
    Renderiza el selector de instancia activa en la sidebar y devuelve
    el contexto completo para el módulo que lo invoca.

    Args:
        module_name: clave del módulo que llama (polar, bode,
            shaft_centerline, asset_documents). Se usa para chequear si
            el profile asociado a la instancia tiene ese módulo en su
            applicable_modules.

    Returns:
        dict con instance_id, profile_key, parameters, documents y todos
        los campos del legacy profile_state.render_profile_selector.
    """
    # 1. Asegurar que existan instances de seed (auto-bootstrap)
    _ensure_seeded_instances_exist()

    # 2. Listar instances disponibles
    instances = list_instances()

    st.markdown("### Activo monitoreado")

    # Badge del backend activo (Local efímero vs Supabase persistente)
    backend = get_active_backend_name()
    if backend == "supabase":
        st.caption("☁️ Persistencia Supabase activa — los datos sobreviven cualquier redeploy.")
    else:
        st.caption(
            "💾 Storage local — los datos se pierden en redeploy de Streamlit Cloud. "
            "Configurá Supabase en secrets para persistencia real."
        )

    if not instances:
        st.warning(
            "No hay instancias de activo registradas. Andá a "
            "**Asset Documents** para crear la primera."
        )
        return _empty_state()

    # 3. Determinar la selección actual desde session_state
    current_id = st.session_state.get(SESSION_KEY_INSTANCE)
    instance_ids = [i["instance_id"] for i in instances]

    if current_id not in instance_ids:
        current_id = instance_ids[0]

    # 4. Etiquetas legibles para el dropdown
    def _label_for(i: Dict[str, Any]) -> str:
        tag = i.get("tag", "").strip()
        loc = i.get("location", "").strip()
        prof_key = i.get("profile_key", "")
        prof = get_profile(prof_key)
        prof_label = prof.label if prof else prof_key
        parts = [i["instance_id"]]
        if tag and tag != "(default)":
            parts.append(f"({tag})")
        suffix = f" — {prof_label}"
        if loc:
            suffix += f" · {loc}"
        return " ".join(parts) + suffix

    label_map = {i["instance_id"]: _label_for(i) for i in instances}

    # Ciclo 14a hotfix — usar SESSION_KEY_INSTANCE como key del widget
    # para que (a) sea fuente única de verdad entre todos los módulos
    # (Polar/Bode/SCL/Library/etc.) y (b) callbacks externos (como el
    # botón "Activar" del grid) puedan setear esa key vía on_click sin
    # caer en el error "cannot be modified after the widget is instantiated".
    # Si la key ya existe en session_state, Streamlit la usa y ignora index.
    # Si no existe, usa el index= para inicializar.
    selected_id = st.selectbox(
        "Instancia activa",
        options=instance_ids,
        index=instance_ids.index(current_id),
        format_func=lambda iid: label_map.get(iid, iid),
        key=SESSION_KEY_INSTANCE,
        help=(
            "Cada instancia representa una máquina física específica. "
            "Los datos del Vault (parámetros, manuales) y los reportes "
            "se asocian a la instancia, no al tipo de máquina. "
            "Crea instancias nuevas desde la Machinery Library."
        ),
    )

    # 5. La línea de "persistir selección" ya no hace falta — el widget
    #    escribe directo a SESSION_KEY_INSTANCE porque es su propia key.
    inst = get_instance(selected_id)
    if inst is None:
        return _empty_state()

    st.session_state[SESSION_KEY_PROFILE] = inst.profile_key  # legacy compat

    # 6. Cargar profile y validar aplicabilidad del módulo
    profile = get_profile(inst.profile_key)
    if profile is None:
        st.error(
            f"La instancia '{inst.instance_id}' referencia un profile "
            f"'{inst.profile_key}' que no existe en MACHINE_PROFILES. "
            f"Editá la instancia para asignarle un profile válido."
        )
        return _empty_state()

    is_applicable = module_name in profile.applicable_modules
    applicability_message = (
        "" if is_applicable
        else module_not_applicable_message(profile.key, module_name)
    )

    # 7. Mostrar resumen visual
    cap_count = len(inst.captured_parameters)
    doc_count = len(inst.documents)
    extras = []
    if cap_count: extras.append(f"{cap_count} parám.")
    if doc_count: extras.append(f"{doc_count} doc.")
    extras_str = " · ".join(extras) if extras else "sin parámetros ni documentos"
    if inst.location:
        st.caption(f"📍 {inst.location} · {extras_str}")
    else:
        st.caption(f"{extras_str}")

    if not is_applicable:
        st.warning(applicability_message)

    return {
        "instance_id": inst.instance_id,
        "instance_label": label_map.get(inst.instance_id, inst.instance_id),
        "profile_key": inst.profile_key,
        "profile_label": profile.label,
        "operating_rpm": float(profile.operating_rpm),
        "iso_part": profile.iso_part,
        "machine_group": profile.machine_group,
        "is_applicable": is_applicable,
        "applicability_message": applicability_message,
        "tag": inst.tag,
        "serial_number": inst.serial_number,
        "location": inst.location,
        "notes": inst.notes,
        "captured_parameters": dict(inst.captured_parameters),
        "documents": list(inst.documents),
        "custom_thresholds": getattr(profile, "custom_thresholds", None),
    }


def _empty_state() -> Dict[str, Any]:
    """Estado vacío cuando no hay instance seleccionable."""
    return {
        "instance_id": "",
        "instance_label": "",
        "profile_key": "",
        "profile_label": "",
        "operating_rpm": 3600.0,
        "iso_part": "20816-2",
        "machine_group": "class_iv",
        "is_applicable": False,
        "applicability_message": "Selecciona una instancia de activo primero.",
        "tag": "",
        "serial_number": "",
        "location": "",
        "notes": "",
        "captured_parameters": {},
        "documents": [],
        "custom_thresholds": None,
    }


def get_active_instance_id() -> Optional[str]:
    """Acceso directo al instance_id activo desde session_state."""
    return st.session_state.get(SESSION_KEY_INSTANCE)


__all__ = [
    "render_instance_selector",
    "get_active_instance_id",
    "SESSION_KEY_INSTANCE",
    "SESSION_KEY_PROFILE",
]
