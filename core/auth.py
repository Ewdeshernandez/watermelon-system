from __future__ import annotations

import hashlib
import hmac
import secrets
import time
from typing import Any, Dict, Optional

import streamlit as st


PBKDF2_PREFIX = "pbkdf2_sha256"
DEFAULT_ITERATIONS = 260000
DEFAULT_SESSION_TIMEOUT_MINUTES = 480


def make_password_hash(password: str, iterations: int = DEFAULT_ITERATIONS) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        iterations,
    ).hex()
    return f"{PBKDF2_PREFIX}${iterations}${salt}${digest}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        algorithm, iterations_str, salt, expected_digest = stored_hash.split("$", 3)
        if algorithm != PBKDF2_PREFIX:
            return False
        iterations = int(iterations_str)
    except Exception:
        return False

    computed_digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        iterations,
    ).hex()

    return hmac.compare_digest(computed_digest, expected_digest)


def _get_auth_config() -> Dict[str, Any]:
    try:
        auth_cfg = st.secrets["auth"]
        return dict(auth_cfg)
    except Exception:
        return {}


def _get_users_map() -> Dict[str, Dict[str, Any]]:
    auth_cfg = _get_auth_config()
    users = auth_cfg.get("users", {})
    return dict(users) if users else {}


def _find_user(identifier: str) -> Optional[Dict[str, Any]]:
    if not identifier:
        return None

    identifier_norm = identifier.strip().lower()
    users_map = _get_users_map()

    for username, user_data in users_map.items():
        user_record = dict(user_data)
        email = str(user_record.get("email", "")).strip().lower()
        username_norm = str(username).strip().lower()

        if identifier_norm == username_norm or identifier_norm == email:
            user_record["username"] = username
            return user_record

    return None


def _session_timeout_seconds() -> int:
    auth_cfg = _get_auth_config()
    minutes = int(auth_cfg.get("session_timeout_minutes", DEFAULT_SESSION_TIMEOUT_MINUTES))
    return max(1, minutes) * 60


def _now() -> int:
    return int(time.time())


def _hide_streamlit_navigation() -> None:
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {
            display: none !important;
        }
        [data-testid="collapsedControl"] {
            display: none !important;
        }
        [data-testid="stSidebarNav"] {
            display: none !important;
        }
        section[data-testid="stSidebar"] {
            display: none !important;
        }
        .stAppHeader {
            background: transparent !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _show_authenticated_layout_tweaks() -> None:
    st.markdown(
        """
        <style>
        .stAppHeader {
            background: transparent !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def is_authenticated() -> bool:
    if not st.session_state.get("auth_ok", False):
        return False

    expires_at = int(st.session_state.get("auth_expires_at", 0))
    if _now() >= expires_at:
        logout(silent=True)
        return False

    st.session_state["auth_expires_at"] = _now() + _session_timeout_seconds()
    return True


def get_current_user() -> Optional[Dict[str, Any]]:
    if not is_authenticated():
        return None

    return {
        "username": st.session_state.get("auth_username"),
        "email": st.session_state.get("auth_email"),
        "full_name": st.session_state.get("auth_full_name"),
        "role": st.session_state.get("auth_role"),
    }


def login(identifier: str, password: str) -> tuple[bool, str]:
    user = _find_user(identifier)
    if user is None:
        return False, "Usuario o correo no encontrado."

    stored_hash = str(user.get("password_hash", "")).strip()
    if not stored_hash:
        return False, "El usuario no tiene hash configurado."

    if not verify_password(password, stored_hash):
        return False, "Clave inválida."

    st.session_state["auth_ok"] = True
    st.session_state["auth_username"] = user.get("username", "")
    st.session_state["auth_email"] = user.get("email", "")
    st.session_state["auth_full_name"] = user.get("full_name", user.get("username", ""))
    st.session_state["auth_role"] = user.get("role", "viewer")
    st.session_state["auth_expires_at"] = _now() + _session_timeout_seconds()

    return True, "Acceso concedido."


def logout(silent: bool = False) -> None:
    keys_to_remove = [
        "auth_ok",
        "auth_username",
        "auth_email",
        "auth_full_name",
        "auth_role",
        "auth_expires_at",
    ]
    for key in keys_to_remove:
        if key in st.session_state:
            del st.session_state[key]

    if not silent:
        st.toast("Sesión cerrada", icon="🔒")


def require_login() -> None:
    if is_authenticated():
        _show_authenticated_layout_tweaks()
        return

    _hide_streamlit_navigation()
    st.warning("Debes iniciar sesión para acceder al demo.")

    try:
        st.switch_page("pages/00_Login.py")
    except Exception:
        st.stop()

    st.stop()


def render_login_shell() -> None:
    _hide_streamlit_navigation()

    st.markdown(
        """
        <style>
        .block-container {
            max-width: 1200px !important;
            padding-top: 3rem !important;
            padding-bottom: 2rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_user_menu() -> None:
    user = get_current_user()
    if not user:
        return

    _show_authenticated_layout_tweaks()

    with st.sidebar:
        st.markdown("### 🔐 Sesión")
        st.caption(f"Usuario: {user.get('username', '-')}")
        if user.get("full_name"):
            st.caption(f"Nombre: {user.get('full_name')}")
        if user.get("email"):
            st.caption(f"Correo: {user.get('email')}")
        st.caption(f"Rol: {user.get('role', 'viewer')}")

        if st.button("Cerrar sesión", use_container_width=True, type="secondary"):
            logout()
            try:
                st.switch_page("pages/00_Login.py")
            except Exception:
                st.rerun()