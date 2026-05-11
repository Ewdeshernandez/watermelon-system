"""
core.snapshot_save_widget
=========================

Widget reusable para guardar snapshots de análisis a Live Monitoring
(Ciclo 23.81). Disponible desde cualquier página de análisis de Watermelon:

    from core.snapshot_save_widget import render_save_snapshot_button

    render_save_snapshot_button(
        instance_id=active_instance_id,
        snapshot_type="waveform",  # o "spectrum", "orbit", "tabular", etc.
        data_builder=lambda: dict(
            sensors_data=[...],  # kwargs que requiere save_X_snapshot
        ),
    )

El widget:
1. Muestra un botón discreto "💾 Guardar para Live Monitoring"
2. Al click, abre un popover con campos:
   - Label de la corrida (texto libre)
   - Notas/observaciones (opcional)
   - Botón confirmar "Guardar snapshot"
3. Llama al save_X_snapshot apropiado
4. Confirma con st.success

Soporta los 8 snapshot types definidos en core.history_storage.KNOWN_TYPES.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional
import streamlit as st


_SAVE_FUNCTIONS = {
    "scl":      ("core.scl_history",       "save_scl_snapshot"),
    "polar":    ("core.polar_history",     "save_polar_snapshot"),
    "bode":     ("core.bode_history",      "save_bode_snapshot"),
    "trend":    ("core.trend_history",     "save_trend_corrida"),  # legacy API
    "waveform": ("core.waveform_history",  "save_waveform_snapshot"),
    "spectrum": ("core.spectrum_history",  "save_spectrum_snapshot"),
    "orbit":    ("core.orbit_history",     "save_orbit_snapshot"),
    "tabular":  ("core.tabular_history",   "save_tabular_snapshot"),
}


def _resolve_save_fn(snapshot_type: str) -> Optional[Callable]:
    info = _SAVE_FUNCTIONS.get(snapshot_type)
    if info is None:
        return None
    module_path, fn_name = info
    try:
        import importlib
        mod = importlib.import_module(module_path)
        return getattr(mod, fn_name, None)
    except Exception:
        return None


def render_save_snapshot_button(
    instance_id: str,
    snapshot_type: str,
    data_builder: Callable[[], Dict[str, Any]],
    *,
    label_button: str = "💾 Guardar para Live Monitoring",
    help_text: Optional[str] = None,
    key_suffix: str = "",
) -> bool:
    """Renderiza botón + popover para guardar snapshot.

    Args:
        instance_id: ID del activo. Si vacío o None, el botón aparece
            deshabilitado con caption explicativo.
        snapshot_type: uno de scl/polar/bode/trend/waveform/spectrum/orbit/tabular
        data_builder: callable() → dict de kwargs para save_X_snapshot.
            Se ejecuta SOLO al confirmar (lazy), para evitar trabajo si
            el usuario no guarda.
        label_button: texto del botón principal
        help_text: tooltip del botón
        key_suffix: para distinguir múltiples botones en la misma página

    Returns:
        True si el snapshot se guardó exitosamente, False de lo contrario.
    """
    if not instance_id:
        st.caption(
            "_💾 Para guardar este análisis a Live Monitoring, primero "
            "seleccioná una instancia activa (Machinery Library)._"
        )
        return False

    save_fn = _resolve_save_fn(snapshot_type)
    if save_fn is None:
        st.caption(f"_💾 Snapshot type `{snapshot_type}` no soportado._")
        return False

    saved = False
    key_prefix = f"snapshot_save_{snapshot_type}_{key_suffix}"

    # Streamlit popover (>=1.32). Fallback a expander si no disponible.
    try:
        popover_ctx = st.popover(label_button, help=help_text, use_container_width=False)
        use_popover = True
    except AttributeError:
        popover_ctx = st.expander(label_button)
        use_popover = False

    with popover_ctx:
        st.markdown(
            f"<div style='font-size:12px;color:#64748b;margin-bottom:6px;'>"
            f"Guardar análisis <b>{snapshot_type.upper()}</b> del activo "
            f"<code>{instance_id}</code> a histórico Supabase para que aparezca "
            f"en Live Monitoring.</div>",
            unsafe_allow_html=True,
        )
        corrida_label = st.text_input(
            "Label de la corrida",
            placeholder="ej. Inspección semanal 11-may, Post-mantenimiento, ...",
            key=f"{key_prefix}_label",
        )
        notes = st.text_area(
            "Notas (opcional)",
            placeholder="Observaciones del especialista sobre esta corrida",
            key=f"{key_prefix}_notes",
            height=80,
        )
        confirm = st.button(
            "✓ Guardar snapshot",
            type="primary",
            use_container_width=True,
            key=f"{key_prefix}_confirm",
        )
        if confirm:
            try:
                kwargs = data_builder() or {}
                # Inyectar label + notes (overriding caller defaults si existen)
                kwargs["corrida_label"] = corrida_label or ""
                kwargs["notes"] = notes or ""
                sid = save_fn(instance_id=instance_id, **kwargs)
                if sid:
                    st.success(
                        f"✓ Snapshot guardado: `{sid}`. Ya visible en Live Monitoring."
                    )
                    saved = True
                else:
                    st.error("No se pudo guardar el snapshot. Revisá los datos.")
            except Exception as e:
                st.error(f"Error guardando: {e}")

    return saved


__all__ = ["render_save_snapshot_button"]
