"""Idioma del portal (ES / EN) para las vistas que ve el cliente.

Regla (ver web-language-policy): la UI del analista va en inglés; lo que ve el
cliente va en español con opción de inglés. El idioma vive en
`st.session_state["wm_lang"]`; si no se ha elegido, depende del rol:
cliente → "es", personal SIGA → "en".

Uso:  from core.portal_i18n import T
      st.caption(T("Última lectura", "Last reading"))
"""
from __future__ import annotations

LANG_KEY = "wm_lang"
CLIENT_PREVIEW_KEY = "live_view_as_client"


def _ss():
    try:
        import streamlit as st
        return st.session_state
    except Exception:  # noqa: BLE001
        return {}


def is_client_view() -> bool:
    """True si la página se está dibujando para un cliente (rol client, o un
    admin/especialista en vista previa "View as client")."""
    ss = _ss()
    role = str(ss.get("auth_role", "") or "").strip().lower()
    return role == "client" or bool(ss.get(CLIENT_PREVIEW_KEY))


def lang() -> str:
    ss = _ss()
    v = str(ss.get(LANG_KEY, "") or "").lower()
    if v in ("es", "en"):
        return v
    return "es" if is_client_view() else "en"


def T(es: str, en: str) -> str:
    """Texto en el idioma activo."""
    return en if lang() == "en" else es


def num(value: float, decimals: int = 1) -> str:
    """Número con separadores del idioma: ES 14 072 · 59,7 — EN 14,072 · 59.7."""
    try:
        txt = f"{float(value):,.{decimals}f}"
    except (TypeError, ValueError):
        return "—"
    if lang() == "en":
        return txt
    return txt.replace(",", " ").replace(".", ",")
