"""
pages/20_Administracion.py — Hub de Administración (una sola página)
===================================================================

Consolida las 3 secciones de administración en pestañas de colores (estilo
Calibración/Reports): Clientes · Licencias Planta · Usuarios. Solo admin.

Diseño anti-riesgo: el selector es un radio estilizado como barra de pestañas
y se renderiza SOLO la sección activa (branch, no st.tabs) — así no hay
colisiones de llaves de widgets ni se ejecutan las 3 secciones a la vez. Cada
sección vive en core/admin/*.py como render() sin efectos de import.

Las páginas viejas (pages/_admin_clients.py, 20_License_Admin.py,
_admin_users.py) quedan como envoltorios finos que llaman al mismo render(),
así sus URLs siguen sirviendo.
"""
from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="Watermelon System | Administration",
    page_icon="🛡️",
    layout="wide",
)

from core.auth import (
    require_login, render_user_menu, require_role, get_current_user,
)
from core.ui_theme import apply_watermelon_page_style
from core.admin import ADMIN_SECTIONS

require_login()
render_user_menu()
apply_watermelon_page_style()
require_role(allowed_roles=("admin",))

_NAVY = "#12305e"
_CYAN = "#274b7d"

# --- Look industrial compartido (mismo del Report Center) ---
from core.ui_industrial import inject_industrial_css, industrial_band

inject_industrial_css()

# KPIs del registro (admins / specialists / clientes) — best-effort.
_kpis = None
try:
    from core.clients import list_admins, list_clients, list_specialists, reload_registry
    reload_registry()
    from core.ui_industrial import dot as _dot
    _kpis = [(f'{_dot("dang")} {len(list_admins() or [])}', "Admins"),
             (f'{_dot("warn")} {len(list_specialists() or [])}', "Specialists"),
             (f'{_dot("info")} {len(list_clients() or [])}', "Clientes")]
except Exception:
    _kpis = None

industrial_band(
    "SIGA Internal · Administration", "Administration Panel",
    "Clientes &amp; roles · Watermelon Plant Licenses · System users · Conocimiento",
    kpis=_kpis,
)

# --- Selector tipo pestañas plano con puntos de color (estilo Calibración) ---
st.markdown(
    f"""
    <style>
    div[data-testid="stRadio"] > div[role="radiogroup"] {{
        flex-direction:row !important; gap:30px; flex-wrap:wrap;
        border-bottom:1px solid #E5EAF0; margin-bottom:16px;
    }}
    div[data-testid="stRadio"] label {{
        background:transparent !important; border:none !important; border-radius:0 !important;
        box-shadow:none; padding:6px 1px 11px 1px !important; margin:0 !important;
        font-weight:500; font-size:14px; color:#64748B; cursor:pointer; transition:color .15s;
    }}
    div[data-testid="stRadio"] label:hover {{ color:{_NAVY}; }}
    div[data-testid="stRadio"] label > div:first-of-type {{ display:none !important; }}
    div[data-testid="stRadio"] label:has(input:checked) {{
        color:{_NAVY}; box-shadow:inset 0 -3px 0 {_CYAN};
    }}
    div[data-testid="stRadio"] label::before {{
        content:"●"; font-size:13px; margin-right:8px; vertical-align:middle;
        position:relative; top:-1px; opacity:0.95;
    }}
    div[data-testid="stRadio"] label:nth-of-type(1)::before {{ color:#1AAEE5; }}
    div[data-testid="stRadio"] label:nth-of-type(2)::before {{ color:#D89B22; }}
    div[data-testid="stRadio"] label:nth-of-type(3)::before {{ color:#16A34A; }}
    </style>
    """,
    unsafe_allow_html=True,
)

_LABELS = [lbl for _, lbl in ADMIN_SECTIONS]
_KEYS = [key for key, _ in ADMIN_SECTIONS]
_choice = st.radio("Section", _LABELS, horizontal=True, key="admin_section_choice",
                   label_visibility="collapsed")
_sel = _KEYS[_LABELS.index(_choice)] if _choice in _LABELS else "clientes"

# --- Render SOLO la sección activa (branch) ---
if _sel == "clientes":
    from core.admin.clients import render as _render
elif _sel == "licencias":
    from core.admin.licenses import render as _render
elif _sel == "conocimiento":
    from core.admin.knowledge import render as _render
else:
    from core.admin.users import render as _render

_render()
