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


NAV_ITEMS = [
    {"label": "Home", "page": "pages/_landing.py"},
    # Ciclo 14a — Machinery Library queda como segundo del menú, justo
    # después de Home: el flujo correcto es seleccionar primero la máquina
    # activa y después cargar sus CSVs en Load Data.
    {"label": "Machinery Library", "page": "pages/00_Machinery_Library.py"},
    {"label": "Load Data", "page": "pages/01_Load_Data.py"},
    {"label": "Tabular List", "page": "pages/01__Tabular_List.py"},
    # Ciclo 15.1 — Machine Map (heatmap de severidad por sensor)
    {"label": "Machine Map", "page": "pages/01b_Machine_Map.py"},
    {"label": "Time Waveforms", "page": "pages/02_Time_Waveforms.py"},
    {"label": "Spectrum", "page": "pages/03_Spectrum.py"},
    {"label": "Trends", "page": "pages/04_Trends.py"},
    {"label": "Orbit Analysis", "page": "pages/05_Orbit_Analysis.py"},
    {"label": "Polar Plot", "page": "pages/06_Polar_Plot.py"},
    {"label": "Bode Plot", "page": "pages/07_Bode_Plot.py"},
    {"label": "Shaft Centerline", "page": "pages/09_Shaft_Centerline.py"},  # ✅ FIX
    {"label": "Phase Analysis", "page": "pages/13_Phase_Analysis.py"},
    {"label": "Diagnostics", "page": "pages/15_Diagnostics.py"},
    {"label": "Reports", "page": "pages/16_Reports.py"},
]


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
        [data-testid="stSidebarNav"] {
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

        [data-testid="stSidebarNav"] {
            display: none !important;
        }

        /* ===== SIDEBAR EXPANDED ===== */
        section[data-testid="stSidebar"][aria-expanded="true"] {
            width: 320px !important;
            min-width: 320px !important;
            max-width: 320px !important;
            background: linear-gradient(180deg, #67b7ff 0%, #4298ee 48%, #1f6fd1 100%) !important;
            border-right: 1px solid rgba(255,255,255,0.14);
        }

        section[data-testid="stSidebar"][aria-expanded="true"] > div {
            background: transparent !important;
            padding-top: 0.6rem !important;
        }

        /* ===== SIDEBAR COLLAPSED ===== */
        section[data-testid="stSidebar"][aria-expanded="false"] {
            width: 0 !important;
            min-width: 0 !important;
            max-width: 0 !important;
        }

        section[data-testid="stSidebar"][aria-expanded="false"] > div {
            width: 0 !important;
            min-width: 0 !important;
            max-width: 0 !important;
            overflow: visible !important;
        }

        div[data-testid="stSidebarUserContent"] {
            padding-top: 0 !important;
        }

        .wm-side-brand {
            font-size: 1.95rem;
            font-weight: 300;
            letter-spacing: -0.04em;
            line-height: 1.0;
            color: #ffffff;
            margin: 0.1rem 0 1.15rem 0;
        }

        .wm-side-section {
            font-size: 0.92rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.11em;
            color: rgba(255,255,255,0.90);
            margin: 1.1rem 0 0.7rem 0;
        }

        .wm-side-divider {
            height: 1px;
            width: 100%;
            background: rgba(255,255,255,0.22);
            border-radius: 999px;
            margin: 0.95rem 0 1rem 0;
        }

        .wm-user-card {
            padding: 15px 16px 14px 16px;
            border-radius: 18px;
            background: rgba(255,255,255,0.10);
            border: 1px solid rgba(255,255,255,0.16);
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.08);
            backdrop-filter: blur(10px);
            margin-bottom: 0.8rem;
        }

        .wm-user-line {
            color: rgba(255,255,255,0.97);
            font-size: 0.92rem;
            margin: 0.34rem 0;
            line-height: 1.45;
        }

        .wm-nav-wrap {
            margin-top: 0.15rem;
            margin-bottom: 0.65rem;
        }

        div[data-testid="stSidebar"] div[data-testid="stButton"] {
            margin-bottom: 0.62rem !important;
        }

        div[data-testid="stSidebar"] div[data-testid="stButton"] > button {
            width: 100% !important;
            min-height: 3.1rem !important;
            border-radius: 18px !important;
            border: 1px solid rgba(209, 223, 241, 0.95) !important;
            background: rgba(255,255,255,0.96) !important;
            color: #0f172a !important;
            font-weight: 600 !important;
            font-size: 1rem !important;
            text-align: left !important;
            justify-content: flex-start !important;
            padding: 0.85rem 1rem !important;
            box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06) !important;
            transition: all 0.18s ease !important;
        }

        div[data-testid="stSidebar"] div[data-testid="stButton"] > button:hover {
            background: #ffffff !important;
            border-color: #ffffff !important;
            box-shadow: 0 10px 20px rgba(15, 23, 42, 0.10) !important;
        }

        div[data-testid="stSidebar"] div[data-testid="stButton"] > button:focus,
        div[data-testid="stSidebar"] div[data-testid="stButton"] > button:focus-visible {
            box-shadow: 0 0 0 2px rgba(255,255,255,0.22) !important;
            outline: none !important;
        }

        div[data-testid="stSidebar"] div[data-testid="stButton"] > button *,
        div[data-testid="stSidebar"] div[data-testid="stButton"] > button p,
        div[data-testid="stSidebar"] div[data-testid="stButton"] > button span,
        div[data-testid="stSidebar"] div[data-testid="stButton"] > button div {
            color: #0f172a !important;
            fill: #0f172a !important;
            opacity: 1 !important;
            font-weight: 600 !important;
        }

        .wm-logout-spacer {
            margin-top: 0.35rem;
        }

        .wm-logout-label {
            font-size: 0.92rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.11em;
            color: rgba(255,255,255,0.90);
            margin: 0.1rem 0 0.7rem 0;
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
        "username":  st.session_state.get("auth_username"),
        "email":     st.session_state.get("auth_email"),
        "full_name": st.session_state.get("auth_full_name"),
        "role":      st.session_state.get("auth_role"),
        # Ciclo 17.14 — campos nuevos para identificación + ACL
        "user_id":   st.session_state.get("auth_user_id", ""),
        "is_admin":  bool(st.session_state.get("auth_is_admin", False)),
        "source":    st.session_state.get("auth_source", "legacy"),
    }


def login(identifier: str, password: str) -> tuple[bool, str]:
    """Intenta autenticar.

    Ciclo 17.14: prioridad a Supabase Auth (usuarios reales).
    Si Supabase no está configurado o el usuario no existe ahí, fallback
    al sistema viejo de usuarios hardcoded en .streamlit/secrets.toml
    (back-compat para no romper sesiones existentes durante la migración).
    """
    # ─── Try 1: Supabase Auth (Ciclo 17.14)
    try:
        from core.supabase_auth import is_supabase_auth_enabled, signin_user
        if is_supabase_auth_enabled():
            result = signin_user(identifier, password)
            if result.get("ok"):
                u = result["user"]
                st.session_state["auth_ok"] = True
                st.session_state["auth_username"] = u.get("email", "")
                st.session_state["auth_email"] = u.get("email", "")
                st.session_state["auth_full_name"] = u.get("full_name", "") or u.get("email", "")
                st.session_state["auth_role"] = u.get("role", "client")
                st.session_state["auth_user_id"] = u.get("id", "")
                st.session_state["auth_is_admin"] = bool(u.get("is_admin"))
                st.session_state["auth_source"] = "supabase"
                st.session_state["auth_expires_at"] = _now() + _session_timeout_seconds()
                return True, "Acceso concedido."
            # Si el error es "credenciales inválidas", NO caer al fallback —
            # significa que Supabase respondió pero no aceptó. Para evitar
            # confusión donde un usuario legacy haga login después de migrar.
            err = result.get("error", "")
            if any(s in err.lower() for s in ("incorrectos", "bloqueada", "no confirmada")):
                return False, err
            # Si el error es de config / red, continuar al fallback legacy
    except Exception:
        # Cualquier problema con Supabase: no romper, ir al fallback legacy
        pass

    # ─── Try 2: sistema legacy (.streamlit/secrets.toml hardcoded)
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
    st.session_state["auth_user_id"] = ""
    st.session_state["auth_is_admin"] = (user.get("role") == "admin")
    st.session_state["auth_source"] = "legacy"
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
        # Ciclo 17.14
        "auth_user_id",
        "auth_is_admin",
        "auth_source",
        "_supabase_admin_client",
    ]
    for key in keys_to_remove:
        if key in st.session_state:
            del st.session_state[key]

    if not silent:
        st.toast("Sesión cerrada")


def require_login() -> None:
    if is_authenticated():
        _show_authenticated_layout_tweaks()
        return

    _hide_streamlit_navigation()
    st.warning("Debes iniciar sesión para acceder al demo.")
    st.switch_page("pages/00_Login.py")


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
        st.markdown('<div class="wm-side-brand">Watermelon</div>', unsafe_allow_html=True)
        st.markdown('<div class="wm-side-divider"></div>', unsafe_allow_html=True)

        st.markdown('<div class="wm-side-section">Sesión</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="wm-user-card">
                <div class="wm-user-line"><b>Usuario:</b> {user.get('username', '-')}</div>
                <div class="wm-user-line"><b>Nombre:</b> {user.get('full_name', '-')}</div>
                <div class="wm-user-line"><b>Correo:</b> {user.get('email', '-')}</div>
                <div class="wm-user-line"><b>Rol:</b> {user.get('role', 'viewer')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="wm-side-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="wm-side-section">Navegación</div>', unsafe_allow_html=True)
        st.markdown('<div class="wm-nav-wrap"></div>', unsafe_allow_html=True)

        # Ciclo 17.14 — Filtrar nav según role.
        # client: vista limitada (sin Load Data, Reports edit, Library edit, etc.)
        #   Eso lo aplicamos en 17.16. Por ahora client ve el menú completo.
        # admin/specialist: menú completo + botón Admin Panel para admin
        for item in NAV_ITEMS:
            if st.button(item["label"], use_container_width=True, key=f"nav_{item['page']}"):
                st.switch_page(item["page"])

        # Ciclo 17.14 — Botón "Admin Panel" SOLO para admin único
        if user.get("is_admin"):
            st.markdown('<div class="wm-side-divider"></div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="wm-side-section">Administración</div>',
                unsafe_allow_html=True,
            )
            if st.button("👥  Admin · Usuarios", use_container_width=True,
                         key="nav_admin_users"):
                try:
                    st.switch_page("pages/_admin_users.py")
                except Exception:
                    st.warning("Admin Panel todavía no está disponible.")

        st.markdown('<div class="wm-side-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="wm-logout-spacer"></div>', unsafe_allow_html=True)
        st.markdown('<div class="wm-logout-label">Sesión</div>', unsafe_allow_html=True)

        if st.button("Cerrar sesión", use_container_width=True, key="logout_button"):
            logout()
            st.switch_page("pages/00_Login.py")

        # Ciclo 17.7 — versión del sistema al pie del sidebar.
        # Sutil, monoespaciada, color tenue para no competir con la
        # navegación pero permitir trazabilidad del build (la versión
        # se autoderiva de git tags vía core.version).
        try:
            from core.version import get_version_info as _gvi
            _v = _gvi()
            _env = _v.get("environment", "")
            _env_color = {
                "production":  "#10b981",
                "staging":     "#f59e0b",
                "development": "#0ea5e9",
            }.get(_env, "#94a3b8")
            _commit_part = f" · {_v['commit']}" if _v.get("commit") else ""
            st.markdown(
                f"""
                <div style="margin-top:0.7rem;padding-top:0.6rem;
                            border-top:1px solid rgba(15,23,42,0.06);
                            font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
                            font-size:0.68rem;color:#94a3b8;line-height:1.5;
                            text-align:center;">
                    <span style="color:#475569;font-weight:700;">{_v['version']}</span>
                    <span style="display:inline-block;width:6px;height:6px;
                                 border-radius:999px;background:{_env_color};
                                 margin:0 0.25rem;vertical-align:middle;"></span>
                    <span>{_env}{_commit_part}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        except Exception:
            pass