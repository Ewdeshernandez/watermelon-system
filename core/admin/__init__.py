"""
core.admin — secciones de Administración (pages/20_Administracion.py).

Cada módulo expone `render()` SIN efectos de import (no llama set_page_config,
require_login, render_user_menu ni require_role — de eso se encarga el hub una
sola vez). El hub renderiza SOLO la sección activa (branch, no st.tabs).
"""
from __future__ import annotations

ADMIN_SECTIONS = [
    ("clientes", "Clientes"),
    ("usuarios", "Usuarios"),
    ("licencias", "Licencias de campo"),
    ("suscripciones", "Suscripciones"),
    ("conocimiento", "Conocimiento"),
]

__all__ = ["ADMIN_SECTIONS"]
