"""
core.admin — Render functions de las secciones de administración.

Cada módulo expone `render()` SIN efectos de import (no llama set_page_config,
require_login, render_user_menu ni require_role — de eso se encarga el hub
pages/20_Administracion.py una sola vez). Así las secciones se pueden componer
en una sola página con pestañas de colores, renderizando SOLO la activa
(patrón branch, no st.tabs) para evitar colisiones de llaves de widgets.
"""
from __future__ import annotations

ADMIN_SECTIONS = [
    ("clientes", "Clientes"),
    ("licencias", "Licencias Planta"),
    ("usuarios", "Usuarios"),
]

__all__ = ["ADMIN_SECTIONS"]
