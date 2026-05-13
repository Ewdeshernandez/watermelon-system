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


# =============================================================
# Ciclo 23.40 — Navegación agrupada por dominio del análisis
# vibracional. Reemplaza la lista plana NAV_ITEMS por NAV_GROUPS
# (lista de secciones), espejando la organización de System1 / AMS
# y reduciendo carga cognitiva en sidebars con 15+ páginas.
#
# Iconos: Unicode geométricos (◉ ▣ ⊕ etc.) — peso cero, look "engineering
# CAD instrument" en vez de emojis rainbow consumer-app.
# =============================================================

NAV_GROUPS = [
    {
        "section": "Overview",
        "items": [
            {"label": "◉  Home", "page": "pages/_landing.py"},
            # Machinery Library segundo: flujo correcto es seleccionar
            # primero la máquina y después cargar CSVs en Load Data.
            {"label": "▣  Machinery Library", "page": "pages/00_Machinery_Library.py"},
            {"label": "▤  Tabular List", "page": "pages/01__Tabular_List.py"},
        ],
    },
    {
        "section": "Ingest",
        "items": [
            {"label": "⤓  Load Data", "page": "pages/01_Load_Data.py"},
            # Ciclo 18.2 — Importers & Plantillas LATAM hub
            {"label": "⇪  Importers & Plantillas", "page": "pages/17_Importers.py"},
            # Ciclo 21 — Wizard guiado para crear activos
            {"label": "✦  Crear activo (wizard)", "page": "pages/_machinery_wizard.py"},
        ],
    },
    {
        "section": "Live Operations",
        "items": [
            # Ciclo 23.1 — Live Monitoring (Tier 0 A): vectores 1X/2X live.
            # Diferenciador vs System1/AMS Suite. 🔴 se queda como signature
            # signaling "live data here".
            {"label": "🔴  Live Monitoring", "page": "pages/02_Live_Monitoring.py"},
            # Ciclo 15.1 — Machine Map (heatmap de severidad por sensor)
            {"label": "⌖  Machine Map", "page": "pages/01b_Machine_Map.py"},
        ],
    },
    {
        "section": "Time Domain",
        "items": [
            {"label": "∿  Time Waveforms", "page": "pages/02_Time_Waveforms.py"},
        ],
    },
    {
        "section": "Frequency Domain",
        "items": [
            {"label": "▥  Spectrum", "page": "pages/03_Spectrum.py"},
        ],
    },
    {
        "section": "Rotordynamics",
        "items": [
            {"label": "⊕  Orbit Analysis", "page": "pages/05_Orbit_Analysis.py"},
            {"label": "◔  Polar Plot", "page": "pages/06_Polar_Plot.py"},
            {"label": "⌭  Bode Plot", "page": "pages/07_Bode_Plot.py"},
            {"label": "─  Shaft Centerline", "page": "pages/09_Shaft_Centerline.py"},
        ],
    },
    {
        "section": "Trends & AI",
        "items": [
            {"label": "📈  Trends", "page": "pages/04_Trends.py"},
            # Ciclo 17.27 — AI Assistant Q&A sobre archivo histórico
            {"label": "✧  AI Assistant", "page": "pages/_ai_assistant.py"},
            # Ciclo 17.31 — Briefing Mensual Ejecutivo PDF al VP cliente
            {"label": "✉  Briefing Mensual", "page": "pages/_monthly_briefing.py"},
        ],
    },
    {
        "section": "Reports",
        "items": [
            # Ciclo 17.33 — eliminados Phase Analysis y Diagnostics legacy
            {"label": "⎙  Reports", "page": "pages/16_Reports.py"},
        ],
    },
    {
        "section": "Administration",
        "items": [
            # Ciclo 20B — Admin de clientes/roles (solo admin)
            {"label": "◇  Admin · Clientes", "page": "pages/_admin_clients.py"},
        ],
    },
]

# Backward-compat: lista plana derivada de los grupos. Cualquier código
# legacy que importe NAV_ITEMS sigue funcionando.
NAV_ITEMS = [item for g in NAV_GROUPS for item in g["items"]]


# =============================================================
# Ciclo 17.16 — Páginas restringidas para role=client
# =============================================================
# Los clientes externos NO deberían poder editar instancias, cargar
# nuevos CSVs ni hacer diagnostics. Solo ven módulos de visualización
# y el archivo histórico de Reports (read-only). Estas páginas se
# OCULTAN del menú lateral Y tienen un guard al inicio que tira
# "acceso denegado" si role=client (defensa en profundidad).

CLIENT_BLOCKED_PAGES = {
    "pages/00_Machinery_Library.py",
    "pages/01_Load_Data.py",
    # pages/15_Diagnostics.py removido del producto en Ciclo 17.33.
    "pages/01b_Machine_Map.py",  # tiene comandos para editar el map
    # Ciclo 18.2 — Importers no para client (sube data y crea activos)
    "pages/17_Importers.py",
    # Ciclo 20B — Admin de clientes solo admin (specialists tampoco)
    "pages/_admin_clients.py",
    # Ciclo 21 — Wizard de activos no para client (solo admin/specialist crean)
    "pages/_machinery_wizard.py",
    # Ciclo 17.27 — AI Assistant queda restringido a admin/specialist
    # en su versión inicial. Cuando expongamos Q&A para clientes,
    # se removerá de esta lista y se ajustará la página para
    # filtrar consultas a "shared_with_client=True" únicamente.
    "pages/_ai_assistant.py",
    # Ciclo 17.31 — Briefing Mensual: solo admin/specialist generan
    # y envían briefings al cliente. El cliente no debe poder
    # acceder a la herramienta de generación (ve los briefings que
    # le llegan por email pero no los genera él mismo).
    "pages/_monthly_briefing.py",
}


# Ciclo 23.130 — Páginas accesibles para el cliente pero OCULTAS del nav.
# El cliente puede llegar a estos módulos vía st.switch_page (redirect
# desde las cards de Live Monitoring → hero banner pro) pero no aparecen
# como entradas en el sidebar. Diseño: el nav del cliente queda con solo
# Home + Live Monitoring + Reports — clase mundial enterprise, menos
# saturado que el menú completo del analista.
CLIENT_HIDE_FROM_NAV = {
    "pages/01__Tabular_List.py",
    "pages/02_Time_Waveforms.py",
    "pages/03_Spectrum.py",
    "pages/05_Orbit_Analysis.py",
    "pages/06_Polar_Plot.py",
    "pages/07_Bode_Plot.py",
    "pages/09_Shaft_Centerline.py",
    "pages/04_Trends.py",
}


def is_page_allowed_for_role(page: str, role: str) -> bool:
    """Devuelve True si el role puede ACCEDER a la página (incluye
    acceso vía st.switch_page redirect). No afecta visibilidad en nav."""
    role = (role or "").strip().lower()
    if role in ("admin", "specialist", "viewer"):
        return True
    if role == "client":
        return page not in CLIENT_BLOCKED_PAGES
    # default conservador: solo páginas no bloqueadas
    return page not in CLIENT_BLOCKED_PAGES


def is_page_visible_in_nav_for_role(page: str, role: str) -> bool:
    """Devuelve True si la página debe APARECER en el sidebar nav.

    Cliente: solo Home + Live Monitoring + Reports en el menú lateral;
    los módulos de análisis se acceden vía las cards de Live Monitoring
    (redirect con switch_page).
    Admin/specialist/viewer: ven todo lo que `is_page_allowed_for_role`
    permite.
    """
    role = (role or "").strip().lower()
    if role == "client":
        if page in CLIENT_BLOCKED_PAGES or page in CLIENT_HIDE_FROM_NAV:
            return False
        return True
    return is_page_allowed_for_role(page, role)


def require_role(allowed_roles: tuple = ("admin", "specialist")) -> None:
    """Guard que pueden usar las páginas restringidas al inicio.
    Si el role del usuario no está en `allowed_roles`, muestra error
    y st.stop().
    """
    user = get_current_user() or {}
    role = (user.get("role", "") or "").strip().lower()
    if role not in allowed_roles:
        st.error(
            f"🚫 **Acceso denegado.** Esta sección requiere role "
            f"`{' / '.join(allowed_roles)}`. Tu role actual es "
            f"`{role or 'sin asignar'}`.\n\n"
            "Si pensás que esto es un error, contactá al administrador."
        )
        st.stop()


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

        /* ===== SIDEBAR EXPANDED — Ciclo 23.39 "Royal blue" =====
           Iteraciones:
             Original:  bright blue #67b7ff→#1f6fd1 (consumer/app-móvil, bloated)
             23.38:     dark slate #0f172a→#1e293b (funeral, sin personalidad)
             23.39:     Royal blue #1e3a8a→#2563eb (sweet spot, enterprise alive)
           256px width (Linear/Notion/Stripe standard), navy → royal blue
           con radial accent — look LinkedIn/Microsoft Azure premium. */
        section[data-testid="stSidebar"][aria-expanded="true"] {
            width: 256px !important;
            min-width: 256px !important;
            max-width: 256px !important;
            background:
                radial-gradient(circle at 0% 0%, rgba(59,130,246,0.30) 0%, transparent 55%),
                linear-gradient(180deg, #1e3a8a 0%, #2563eb 60%, #1e3a8a 100%) !important;
            border-right: 1px solid rgba(255,255,255,0.10);
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

        /* Ciclo 17.22 — Avatar mini al fondo del sidebar (Opción A UX).
           Reemplaza el wm-user-card grande de arriba: identidad accesible
           pero sin ocupar 25% del scroll inicial del sidebar. */
        .wm-user-mini {
            display: flex;
            align-items: center;
            gap: 0.65rem;
            padding: 0.55rem 0.4rem 0.65rem 0.4rem;
            margin-bottom: 0.35rem;
            border-radius: 12px;
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.10);
        }
        .wm-user-mini-avatar {
            width: 34px;
            height: 34px;
            border-radius: 50%;
            background: linear-gradient(135deg, #38bdf8 0%, #0ea5e9 100%);
            color: #0f172a;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 0.95rem;
            flex-shrink: 0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.18);
            letter-spacing: 0;
        }
        .wm-user-mini-text {
            display: flex;
            flex-direction: column;
            overflow: hidden;
            min-width: 0;
        }
        .wm-user-mini-name {
            font-weight: 600;
            font-size: 0.92rem;
            color: rgba(255,255,255,0.96);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            line-height: 1.15;
        }
        .wm-user-mini-role {
            font-size: 0.72rem;
            color: rgba(255,255,255,0.66);
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-top: 1px;
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

        /* Ciclo 23.38 — Sidebar nav buttons slim "international" style.
           Antes: 50px alto, 16px fuente, blanco con sombra → look "iPad app".
           Ahora: 38px alto, 13.5px fuente, translucent dark → look Linear/Notion. */
        div[data-testid="stSidebar"] div[data-testid="stButton"] {
            margin-bottom: 0.22rem !important;
        }

        div[data-testid="stSidebar"] div[data-testid="stButton"] > button {
            width: 100% !important;
            min-height: 2.35rem !important;
            border-radius: 8px !important;
            border: 1px solid transparent !important;
            background: transparent !important;
            color: rgba(241, 245, 249, 0.85) !important;
            font-weight: 500 !important;
            font-size: 0.84rem !important;
            letter-spacing: 0.005em !important;
            text-align: left !important;
            justify-content: flex-start !important;
            padding: 0.5rem 0.8rem !important;
            box-shadow: none !important;
            transition: all 0.15s ease !important;
        }

        div[data-testid="stSidebar"] div[data-testid="stButton"] > button:hover {
            background: rgba(255,255,255,0.08) !important;
            border-color: rgba(255,255,255,0.10) !important;
            color: #ffffff !important;
        }

        div[data-testid="stSidebar"] div[data-testid="stButton"] > button:focus,
        div[data-testid="stSidebar"] div[data-testid="stButton"] > button:focus-visible {
            background: rgba(255,255,255,0.10) !important;
            border-color: rgba(96,165,250,0.45) !important;
            box-shadow: 0 0 0 2px rgba(96,165,250,0.18) !important;
            outline: none !important;
        }

        div[data-testid="stSidebar"] div[data-testid="stButton"] > button *,
        div[data-testid="stSidebar"] div[data-testid="stButton"] > button p,
        div[data-testid="stSidebar"] div[data-testid="stButton"] > button span,
        div[data-testid="stSidebar"] div[data-testid="stButton"] > button div {
            color: inherit !important;
            fill: inherit !important;
            opacity: 1 !important;
            font-weight: 500 !important;
        }

        /* Brand mark + section labels en el sidebar slim */
        .wm-side-brand {
            font-size: 1.15rem !important;
            font-weight: 700 !important;
            letter-spacing: -0.02em !important;
            color: #f8fafc !important;
            margin: 0.2rem 0 1rem 0 !important;
        }
        .wm-side-section {
            font-size: 0.66rem !important;
            font-weight: 800 !important;
            text-transform: uppercase;
            letter-spacing: 0.12em !important;
            color: rgba(241,245,249,0.55) !important;
            margin: 1rem 0 0.5rem 0 !important;
            padding-left: 0.2rem;
        }
        .wm-side-divider {
            height: 1px;
            background: rgba(255,255,255,0.10) !important;
            margin: 0.7rem 0 !important;
        }
        /* User mini card más compacto */
        .wm-user-mini {
            padding: 0.45rem 0.4rem !important;
            border-radius: 9px !important;
        }
        .wm-user-mini-avatar {
            width: 28px !important;
            height: 28px !important;
            font-size: 0.82rem !important;
        }
        .wm-user-mini-name {
            font-size: 0.82rem !important;
        }
        .wm-user-mini-role {
            font-size: 0.62rem !important;
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

        # Ciclo 17.22 — Card de usuario movido al fondo del sidebar (Opción A).
        # Acá arranca DIRECTO la navegación. La identidad del usuario va abajo
        # como avatar mini, junto a "Cambiar mi password" y "Cerrar sesión".
        #
        # Ciclo 17.23 — Sacamos el header "Navegación" (Linear/Notion/Stripe
        # tampoco lo ponen): los botones SON la navegación, no necesitan
        # título que lo diga. El divider de arriba ya da separación visual.
        st.markdown('<div class="wm-nav-wrap"></div>', unsafe_allow_html=True)

        # Ciclo 23.40 — Render nav AGRUPADO por dominio.
        # Iteración: por cada grupo, primero filtramos qué items son
        # visibles para el role actual. Si el grupo queda con 0 items
        # visibles (caso típico: client no ve "Ingest" ni "Administration"),
        # NO emitimos el header de la sección — evita "Section sin items"
        # que se ve mal.
        # client: vista limitada; specialist/admin: menú completo.
        _user_role = (user.get("role", "") or "").strip().lower()

        for group in NAV_GROUPS:
            # Ciclo 23.130 — usar is_page_visible_in_nav_for_role para que
            # los módulos de análisis del cliente (accesibles vía redirect)
            # NO aparezcan en el sidebar — solo Home + Live Monitoring + Reports.
            visible_items = [
                it for it in group["items"]
                if is_page_visible_in_nav_for_role(it["page"], _user_role)
            ]
            if not visible_items:
                continue  # toda la sección oculta para este role → skip header
            st.markdown(
                f'<div class="wm-side-section">{group["section"]}</div>',
                unsafe_allow_html=True,
            )
            for item in visible_items:
                if st.button(
                    item["label"],
                    use_container_width=True,
                    key=f"nav_{item['page']}",
                ):
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

        # Ciclo 17.22 — Avatar mini con identidad del usuario (Opción A UX).
        # Reemplaza el card grande de arriba. El email completo aparece como
        # tooltip al hover sobre el bloque (atributo title=).
        _full_name = (user.get("full_name") or user.get("email") or "?").strip()
        _email = user.get("email", "") or user.get("username", "") or ""
        _role = (user.get("role", "") or "viewer").lower()
        _initial = (_full_name[:1] if _full_name else "?").upper()
        st.markdown(
            f"""
            <div class="wm-user-mini" title="{_email}">
                <div class="wm-user-mini-avatar">{_initial}</div>
                <div class="wm-user-mini-text">
                    <div class="wm-user-mini-name">{_full_name}</div>
                    <div class="wm-user-mini-role">{_role}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Ciclo 17.16 — Cambiar mi password (visible para todos los roles
        # excepto los del sistema legacy hardcoded, que cambian via secrets).
        if st.session_state.get("auth_source") == "supabase":
            with st.popover("🔑  Cambiar mi password", use_container_width=True):
                st.markdown("**Elegí una nueva contraseña**")
                _new = st.text_input(
                    "Nueva password (mín. 8 chars)",
                    type="password",
                    key="wm_self_pwd_new",
                )
                _confirm = st.text_input(
                    "Confirmar",
                    type="password",
                    key="wm_self_pwd_confirm",
                )
                if st.button("Cambiar", key="wm_self_pwd_submit",
                             type="primary", use_container_width=True):
                    if not _new or len(_new) < 8:
                        st.error("Mínimo 8 caracteres.")
                    elif _new != _confirm:
                        st.error("Las dos passwords no coinciden.")
                    else:
                        try:
                            from core.supabase_auth import reset_user_password
                            uid = st.session_state.get("auth_user_id", "")
                            if not uid:
                                st.error("No tengo tu user_id de sesión.")
                            else:
                                res = reset_user_password(uid, _new)
                                if res.get("ok"):
                                    st.success(
                                        "✓ Password cambiada. Tu sesión "
                                        "actual sigue activa hasta que "
                                        "cierres."
                                    )
                                    st.session_state.pop("wm_self_pwd_new", None)
                                    st.session_state.pop("wm_self_pwd_confirm", None)
                                else:
                                    st.error(f"Falló: {res.get('error', 'error')}")
                        except Exception as e:
                            st.error(f"Error: {e}")

        if st.button("Cerrar sesión", use_container_width=True, key="logout_button"):
            logout()
            st.switch_page("pages/00_Login.py")

        # Ciclo 17.7 — versión del sistema al pie del sidebar.
        # Ciclo 17.24 — versión LIMPIA: solo el dot del entorno + número.
        # El env y el commit hash quedan como tooltip al hover, así no
        # contaminan visualmente pero siguen accesibles para troubleshoot.
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
            _tooltip = f"{_env}{_commit_part}".strip(" ·")
            st.markdown(
                f"""
                <div title="{_tooltip}"
                     style="margin-top:0.7rem;padding-top:0.6rem;
                            border-top:1px solid rgba(15,23,42,0.06);
                            font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
                            font-size:0.72rem;color:#94a3b8;line-height:1.5;
                            text-align:center;cursor:default;">
                    <span style="display:inline-block;width:7px;height:7px;
                                 border-radius:999px;background:{_env_color};
                                 margin-right:0.4rem;vertical-align:middle;
                                 box-shadow:0 0 6px {_env_color}66;"></span>
                    <span style="color:#475569;font-weight:600;">{_v['version']}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        except Exception:
            pass