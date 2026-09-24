"""
pages/15_Report_Center.py
=========================

**Report Center** — pestaña única para todo lo de reportes en Watermelon System:

  · Approval — genera el borrador, revisa/edita recomendaciones, firma
    (Prepared / Reviewed) y al aprobar envía el PDF final al cliente.
  · Delivery scheduling — configura por Cliente → Activo el/los correo(s),
    WhatsApp(s), el horario del briefing (semanal/mensual) y el aviso
    automático por alarma. Único lugar donde se programa el envío
    (antes estaba disperso en Machinery Library).

La lógica de render vive en core.report_center_ui (compartida).
"""
from __future__ import annotations

import streamlit as st

from core.auth import require_login, render_user_menu, require_role

st.set_page_config(
    page_title="Watermelon System | Report Center",
    layout="wide",
    initial_sidebar_state="expanded",
)
require_login()
render_user_menu()
require_role(("admin", "specialist"))

from core.ui_theme import apply_watermelon_page_style, page_header

apply_watermelon_page_style()
page_header(
    "Report Center",
    subtitle="Review, sign and send reports · centralized delivery scheduling "
             "per client and asset.",
)

from core.report_center_ui import inject_css, render_approval, render_delivery

inject_css()

try:
    from core.auth import get_current_user as _gcu
    _me = (_gcu() or {})
    _me_name = _me.get("full_name") or _me.get("username") or ""
except Exception:
    _me_name = ""

# Sub-navegación (segmented control con el mismo look del design system)
st.markdown('<div class="bf-label">View</div>', unsafe_allow_html=True)
_view = st.radio(
    "View", ["Approval", "Delivery scheduling"], horizontal=True,
    key="rc_view", label_visibility="collapsed",
)
st.markdown("")

if _view == "Approval":
    render_approval(_me_name)
else:
    render_delivery()
