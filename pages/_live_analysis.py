"""
pages/_live_analysis.py
=======================

Análisis Avanzado del Live Monitoring (Ciclo 23.156) — vista cliente.

Página OCULTA del nav (prefijo `_`). Se llega con el botón 🍉 desde
Live Monitoring. Una sola vista a la vez, seleccionable con el control
segmentado a la derecha: Espectro · Forma de onda · Órbita.
(Más adelante: Polar, Shaft Centerline.)

Estándares de visualización (clase System1/AMS):
  • Espectro: full-scale 60.000 CPM, cursores 1X/2X/3X, escala Y común
    por familia (velocidad / aceleración / proximidad).
  • Forma de onda: escala Y simétrica común por familia.
  • Órbita: aspecto 1:1 por cojinete.
"""
from __future__ import annotations

import streamlit as st

from core.auth import require_login, render_user_menu

# =============================================================
# CONFIG + GUARDS
# =============================================================

st.set_page_config(
    page_title="Watermelon System | Análisis Avanzado",
    layout="wide",
)
require_login()
render_user_menu()

# =============================================================
# HEADER (hero maestro)
# =============================================================

from core.ui_theme import page_header

page_header(
    "Análisis Avanzado",
    subtitle="Espectro · Forma de onda · Órbita — último snapshot del activo",
)

# =============================================================
# Activo objetivo (viene de Live Monitoring vía session_state)
# =============================================================

instance_id = (
    st.session_state.get("_live_analysis_instance")
    or st.session_state.get("live_asset_v3")
)
if not instance_id:
    st.info("Entrá desde **Live Monitoring** con el botón 🍉 Análisis avanzado.")
    st.stop()

_tag = str(instance_id).upper()
try:
    from core.instance_state import get_instance
    _inst = get_instance(instance_id)
    _tag = (getattr(_inst, "tag", None) or _tag)
except Exception:
    pass

# =============================================================
# Selector de vista (derecha) + chip del activo (izquierda)
# =============================================================

_VIEWS = ["📊 Espectro", "🌊 Forma de onda", "🌀 Órbita"]

c_left, c_right = st.columns([3, 2])
with c_left:
    st.markdown(
        f"<div style='padding-top:6px;'>"
        f"<span style='background:#0f172a;color:#f1f5f9;border-radius:8px;"
        f"padding:4px 12px;font-weight:700;font-size:13px;"
        f"letter-spacing:0.06em;'>{_tag}</span></div>",
        unsafe_allow_html=True,
    )
with c_right:
    try:
        _view = st.segmented_control(
            "Vista", _VIEWS, default=_VIEWS[0],
            key="wm_la_view", label_visibility="collapsed",
        )
    except Exception:
        _view = st.radio(
            "Vista", _VIEWS, horizontal=True,
            key="wm_la_view_radio", label_visibility="collapsed",
        )
_view = _view or _VIEWS[0]

st.markdown("")

# =============================================================
# Render de la vista seleccionada
# =============================================================

from core.recent_analyses_widget import (
    _render_orbit_detail,
    _render_spectrum_detail,
    _render_waveform_detail,
    load_latest_payload,
)

_KEY_BY_VIEW = {
    _VIEWS[0]: ("spectrum", _render_spectrum_detail),
    _VIEWS[1]: ("waveform", _render_waveform_detail),
    _VIEWS[2]: ("orbit", _render_orbit_detail),
}
_akey, _render = _KEY_BY_VIEW[_view]

with st.spinner("Cargando snapshot…"):
    _payload, _snap_id = load_latest_payload(instance_id, _akey)

if not _payload:
    st.info(
        "Aún no hay un snapshot de este tipo para el activo. "
        "Se genera desde Load Data / módulos de análisis."
    )
else:
    _render(_payload)
