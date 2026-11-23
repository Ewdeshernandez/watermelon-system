"""
pages/20_Administracion.py — Administración (solo admin)
========================================================

Diseño simple, tipo Ajustes: arriba el estado de la cuenta en 4 cifras y los
avisos que piden acción; abajo la lista de secciones (izquierda) y el detalle
(derecha). Se renderiza SOLO la sección activa (branch).

Secciones:
  Clientes       → core.admin.clients_home   (editor técnico en "Avanzado")
  Usuarios       → core.admin.people         (panel completo en "Avanzado")
  Licencias      → core.admin.licenses
  Suscripciones  → core.admin.subscriptions
  Conocimiento   → core.admin.knowledge
"""
from __future__ import annotations

import html

import streamlit as st

st.set_page_config(page_title="Watermelon System | Administración", page_icon="🛡️",
                   layout="wide")

from core.auth import get_current_user, render_user_menu, require_login, require_role  # noqa: E402
from core.ui_theme import apply_watermelon_page_style  # noqa: E402

require_login()
render_user_menu()
apply_watermelon_page_style()
require_role(allowed_roles=("admin",))

from core.admin import ADMIN_SECTIONS, overview, ui  # noqa: E402
from core.ui_industrial import inject_industrial_css  # noqa: E402

inject_industrial_css()
ui.inject()

_me = get_current_user() or {}
_first = str(_me.get("full_name") or "").split(" ")[0] or "admin"
st.markdown(f'<div class="adm-h1">Administración</div>'
            f'<div class="adm-hi">Hola, {html.escape(_first)}. Todo lo de la cuenta en un solo lugar.</div>',
            unsafe_allow_html=True)

data = overview.collect()

_ni = data.get("next_invoice")
_tiles = [
    (data.get("clients", 0), f"Cliente{'s' if data.get('clients') != 1 else ''} · "
                             f"{data.get('machines', 0)} máquinas"),
    (data.get("users") if data.get("users") is not None else "—", "Usuarios"),
    (data.get("lic_active") if data.get("lic_active") is not None else "—",
     f"Licencias activas · {data.get('lic_machines') or 0} equipos"),
]
if _ni:
    from core.billing.periods import fmt_date
    _d = fmt_date(_ni[0]).rsplit(" ", 1)[0]
    _tiles.append((_d, f"Próxima factura · {overview.short_money(float(_ni[1]), _ni[2])}"))
else:
    _tiles.append(("—", "Próxima factura"))
st.markdown('<div class="adm-tiles">' + "".join(
    f'<div class="adm-tile"><div class="adm-tn">{html.escape(str(n))}</div>'
    f'<div class="adm-tl">{html.escape(l)}</div></div>' for n, l in _tiles) + "</div>",
    unsafe_allow_html=True)

_KEYS = [k for k, _ in ADMIN_SECTIONS]
_LBL = dict(ADMIN_SECTIONS)
_COUNTS = {"clientes": data.get("clients"), "usuarios": data.get("users"),
           "licencias": len(data.get("_licenses") or []) or None,
           "suscripciones": len(data.get("_subs") or []) or None,
           "conocimiento": data.get("docs")}


def _go(section: str) -> None:
    st.session_state["adm_nav"] = section


for i, (sev, text, dest) in enumerate(overview.alerts(data)[:4]):
    a, b = st.columns([12, 1.4])
    a.markdown(f'<div class="adm-alert {sev}">{html.escape(text)}</div>', unsafe_allow_html=True)
    b.button("Ver", key=f"adm_alert_{i}", on_click=_go, args=(dest,), use_container_width=True)

if st.session_state.get("adm_nav") not in _KEYS:
    st.session_state["adm_nav"] = "clientes"

left, right = st.columns([1, 3.2], gap="large")
with left:
    sel = st.radio("Sección", _KEYS, key="adm_nav", label_visibility="collapsed",
                   format_func=lambda k: _LBL[k] + (f"  ·  {_COUNTS[k]}" if _COUNTS.get(k) else ""))
with right:
    if sel == "clientes":
        from core.admin.clients_home import render as _render
    elif sel == "usuarios":
        from core.admin.people import render as _render
    elif sel == "licencias":
        from core.admin.licenses import render as _render
    elif sel == "suscripciones":
        from core.admin.subscriptions import render as _render
    else:
        from core.admin.knowledge import render as _render
    _render()
