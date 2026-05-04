from __future__ import annotations

from pathlib import Path

import streamlit as st

from core.auth import is_authenticated, login as wm_login, render_login_shell
from core.version import get_version_info

st.set_page_config(
    page_title="Watermelon System | Login",
    page_icon="🍉",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =========================================================
# PATHS
# =========================================================
ROOT_DIR = Path(__file__).resolve().parents[1]
ASSETS_DIR = ROOT_DIR / "assets"
LOGO_PATH = ASSETS_DIR / "watermelon_logo.png"


def asset_exists(path: Path) -> bool:
    try:
        return path.exists() and path.is_file()
    except Exception:
        return False


# =========================================================
# AUTH
# =========================================================
if is_authenticated():
    st.switch_page("00_Home.py")

render_login_shell()

# =========================================================
# CSS
# =========================================================
st.markdown(
    """
    <style>
    header, #MainMenu, footer {visibility: hidden;}
    [data-testid="stSidebar"] {display: none !important;}

    /* =========================================================
       Ciclo 17.6 — Login premium "international"
       Paleta sobria azul corporativo (no cyan brillante).
       Glassmorphism muy sutil en la card. Se elimina el recuadro
       rojo que Streamlit pone al campo de password en focus
       (forzando bordes gris/azul sin importar estado).
       ========================================================= */

    .stApp {
        background:
            radial-gradient(circle at 78% 0%, rgba(33,71,140,0.10) 0%, transparent 32%),
            radial-gradient(circle at 0% 100%, rgba(33,71,140,0.06) 0%, transparent 28%),
            linear-gradient(180deg, #f3f6fb 0%, #e9eef6 100%);
        color: #0e1a30;
    }

    .block-container {
        max-width: 1280px !important;
        padding-top: 3rem !important;
        padding-bottom: 2.4rem !important;
        padding-left: 1.8rem !important;
        padding-right: 1.8rem !important;
    }

    /* columnas */
    [data-testid="column"] {
        display: flex;
        align-items: center;
    }

    /* ---------- IZQUIERDA: hero corporativo ---------- */
    .wm-eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.45rem 0.95rem;
        border-radius: 999px;
        background: rgba(33,71,140,0.07);
        border: 1px solid rgba(33,71,140,0.18);
        color: #21478c;
        font-size: 0.74rem;
        font-weight: 800;
        letter-spacing: 0.18em;
        text-transform: uppercase;
    }
    .wm-eyebrow .dot {
        width: 6px; height: 6px;
        border-radius: 999px;
        background: #21478c;
        box-shadow: 0 0 0 3px rgba(33,71,140,0.18);
    }

    .wm-brand-row {
        display: flex;
        align-items: center;
        gap: 0.95rem;
        margin-top: 1.1rem;
        margin-bottom: 1.1rem;
    }

    .wm-logo-box {
        width: 56px;
        height: 56px;
        border-radius: 14px;
        background: #ffffff;
        border: 1px solid rgba(15, 23, 42, 0.07);
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
        display: flex;
        align-items: center;
        justify-content: center;
        overflow: hidden;
        flex-shrink: 0;
    }

    .wm-brand-title {
        color: #0e1a30;
        font-size: 0.92rem;
        font-weight: 800;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        margin: 0;
    }

    .wm-brand-subtitle {
        color: #5d6d85;
        font-size: 0.88rem;
        font-weight: 600;
        margin-top: 0.15rem;
    }

    .wm-hero {
        color: #07142b;
        font-size: 3.7rem;
        line-height: 1.02;
        font-weight: 800;
        letter-spacing: -0.035em;
        max-width: 650px;
        margin: 0;
    }
    .wm-hero .accent {
        background: linear-gradient(90deg, #21478c 0%, #2a6dd1 100%);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
    }

    .wm-left-note {
        margin-top: 1rem;
        color: #5d6d85;
        font-size: 1.01rem;
        line-height: 1.65;
        max-width: 500px;
    }

    .wm-trust-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.65rem;
        margin-top: 1.6rem;
    }
    .wm-trust-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.38rem;
        padding: 0.35rem 0.7rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.65);
        border: 1px solid rgba(33,71,140,0.10);
        color: #3d4f6e;
        font-size: 0.78rem;
        font-weight: 600;
    }

    /* ---------- DERECHA: tarjeta login glassmorphism ----------
       FIX 17.6.1: antes envolvía el form en <div class="wm-login-card">
       pero Streamlit renderiza cada st.markdown como bloque
       independiente, así que el div se mostraba VACÍO arriba del
       form (caja blanca fantasma). Ahora estilamos directamente
       el [data-testid="stForm"] para que el form SEA la card.

       Usamos :has() para detectar nuestro marker .wm-login-marker
       dentro del column derecho y aplicar el estilo a TODO el
       column como card. */
    [data-testid="column"]:has(.wm-login-marker) {
        background: rgba(255, 255, 255, 0.88);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid rgba(33, 71, 140, 0.10);
        border-radius: 22px;
        box-shadow:
            0 22px 50px rgba(15, 23, 42, 0.10),
            0 2px 8px rgba(15, 23, 42, 0.03);
        padding: 2rem 1.85rem 1.55rem 1.85rem !important;
        max-width: 480px;
        margin-left: auto;
        align-items: stretch !important;
    }
    .wm-login-marker { display: none; }
    .wm-logo-marker { display: none; }

    /* Logo: dejar la imagen plana, sin "caja" envolvente fantasma */
    [data-testid="column"]:has(.wm-logo-marker) {
        align-items: center !important;
    }
    [data-testid="column"]:has(.wm-logo-marker) [data-testid="stImage"] img {
        width: 56px !important;
        height: 56px !important;
        border-radius: 14px;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
    }

    .wm-login-top {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        color: #21478c;
        font-size: 0.74rem;
        font-weight: 800;
        letter-spacing: 0.16em;
        text-transform: uppercase;
    }

    .wm-login-title {
        margin-top: 0.6rem;
        color: #0e1a30;
        font-size: 1.85rem;
        line-height: 1.05;
        font-weight: 800;
        letter-spacing: -0.025em;
    }

    .wm-login-copy {
        margin-top: 0.55rem;
        margin-bottom: 1.3rem;
        color: #5d6d85;
        font-size: 0.93rem;
        line-height: 1.55;
    }

    /* Forms — eliminar el recuadro rojo del field de password */
    div[data-testid="stForm"] {
        background: transparent !important;
        border: 0 !important;
        padding: 0 !important;
    }

    div[data-testid="stTextInput"] label {
        color: #1e2c47 !important;
        font-size: 0.88rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.01em !important;
    }

    /* =========================================================
       FIX 17.6.1 — KILL del recuadro rojo en username/password.
       El borde rojo viene de DOS sitios:
         (a) Streamlit BaseWeb pinta border-color:rgb(255,75,75)
             cuando el input está "needs attention" o vacío
             dentro de un st.form sin submit.
         (b) Chrome pone outline rojizo en autofill +
             :focus-visible (especialmente con password manager).
       Estrategia: override TOTAL en TODOS los selectores de
       BaseWeb (data-baseweb=input, base-input, input) y kill
       de webkit-autofill/focus-visible.
       ========================================================= */

    /* =========================================================
       FIX 17.6.2 — Doble borde resuelto.
       BaseWeb tiene 2 capas anidadas que ambas pintaban borde:
         [data-baseweb="input"]      ← OUTER (le ponemos el borde)
           [data-baseweb="base-input"]  ← INNER (transparente)
       Antes le ponía borde a las DOS y se veía doble (uno
       afuera del otro). Ahora solo el OUTER tiene borde + bg,
       el INNER queda transparente sin borde.
       ========================================================= */

    /* OUTER: el único que pinta borde + background */
    div[data-testid="stTextInput"] [data-baseweb="input"] {
        background: #f8fafd !important;
        border: 1px solid #d3dde9 !important;
        border-radius: 12px !important;
        min-height: 52px !important;
        box-shadow: none !important;
        outline: none !important;
        transition: border-color 0.18s ease, box-shadow 0.18s ease !important;
    }
    /* INNER: transparente, sin borde, sin radius (hereda del outer) */
    div[data-testid="stTextInput"] [data-baseweb="base-input"] {
        background: transparent !important;
        border: 0 !important;
        border-radius: 0 !important;
        box-shadow: none !important;
        outline: none !important;
        min-height: 50px !important;
    }
    /* Hover (solo outer) */
    div[data-testid="stTextInput"] [data-baseweb="input"]:hover {
        border-color: #b8c6d8 !important;
        box-shadow: none !important;
    }
    /* Focus (solo outer): azul corporativo, NO rojo */
    div[data-testid="stTextInput"] [data-baseweb="input"]:focus-within {
        border-color: #21478c !important;
        box-shadow: 0 0 0 3px rgba(33,71,140,0.15) !important;
    }
    /* Asegurar que el inner NO pinte un focus ring propio */
    div[data-testid="stTextInput"] [data-baseweb="base-input"]:focus-within {
        border: 0 !important;
        box-shadow: none !important;
    }
    /* Override TOTAL de cualquier rojo (sólo en el outer ahora) */
    div[data-testid="stTextInput"][aria-invalid] [data-baseweb="input"],
    div[data-testid="stTextInput"] [data-baseweb="input"][aria-invalid="true"],
    div[data-testid="stTextInput"] [data-baseweb="input"][style*="rgb(255"],
    div[data-testid="stTextInput"]:has(input:invalid) [data-baseweb="input"],
    div[data-testid="stTextInput"]:has(input:-webkit-autofill) [data-baseweb="input"] {
        border-color: #d3dde9 !important;
        box-shadow: none !important;
        background-color: #f8fafd !important;
    }
    /* Quitar el outline rojo de :focus-visible */
    div[data-testid="stTextInput"] *:focus,
    div[data-testid="stTextInput"] *:focus-visible {
        outline: none !important;
        outline-color: transparent !important;
    }
    /* Chrome autofill — quitar el fondo amarillo/rojizo */
    div[data-testid="stTextInput"] input:-webkit-autofill,
    div[data-testid="stTextInput"] input:-webkit-autofill:hover,
    div[data-testid="stTextInput"] input:-webkit-autofill:focus {
        -webkit-text-fill-color: #0e1a30 !important;
        -webkit-box-shadow: 0 0 0 1000px #f8fafd inset !important;
        box-shadow: 0 0 0 1000px #f8fafd inset !important;
        transition: background-color 5000s ease-in-out 0s !important;
    }

    div[data-testid="stTextInput"] input {
        color: #0e1a30 !important;
        font-size: 0.98rem !important;
        background: transparent !important;
    }

    div[data-testid="stTextInput"] input::placeholder {
        color: #8a9bb3 !important;
        opacity: 1 !important;
    }

    /* "Press Enter to submit" — texto que sale al lado del input.
       Lo dejamos pero más sutil. */
    div[data-testid="InputInstructions"] {
        color: #94a3b8 !important;
        font-size: 0.78rem !important;
        font-weight: 500 !important;
    }

    div[data-testid="stFormSubmitButton"] > button {
        width: 100% !important;
        height: 52px !important;
        margin-top: 1rem !important;
        border: 0 !important;
        border-radius: 12px !important;
        font-size: 1rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.02em !important;
        color: #ffffff !important;
        background: linear-gradient(135deg, #21478c 0%, #2a6dd1 100%) !important;
        box-shadow: 0 10px 26px rgba(33,71,140,0.28) !important;
        transition: all 0.2s ease !important;
    }

    div[data-testid="stFormSubmitButton"] > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 14px 32px rgba(33,71,140,0.36) !important;
        background: linear-gradient(135deg, #1d3f7c 0%, #2562bf 100%) !important;
    }

    .wm-footer-note {
        margin-top: 1.4rem;
        padding-top: 0.95rem;
        border-top: 1px solid rgba(33,71,140,0.10);
        color: #7a8aa1;
        font-size: 0.82rem;
        line-height: 1.5;
        display: flex;
        flex-direction: column;
        gap: 0.22rem;
    }

    .wm-footer-build {
        color: #a3b1c7;
        font-size: 0.75rem;
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    }

    div[data-testid="stAlert"] {
        border-radius: 12px !important;
        border: 1px solid rgba(15, 23, 42, 0.08) !important;
    }

    @media (max-width: 1100px) {
        .wm-hero {
            font-size: 2.7rem;
            max-width: 100%;
        }

        .wm-login-card {
            max-width: 100%;
            margin-left: 0;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# LAYOUT
# =========================================================
left_col, right_col = st.columns([1.2, 0.95], gap="large")

with left_col:
    st.markdown(
        """
        <div class="wm-eyebrow"><span class="dot"></span>Watermelon System</div>
        """,
        unsafe_allow_html=True,
    )

    logo_col, text_col = st.columns([0.12, 0.88], gap="small")

    with logo_col:
        # Marker invisible que sirve para que el CSS detecte ESTA
        # columna vía :has() y aplique los estilos del logo.
        # Antes envolvíamos con <div class="wm-logo-box"> pero
        # Streamlit renderizaba el div VACÍO porque la imagen va
        # como bloque aparte, dejando una caja blanca fantasma.
        st.markdown('<span class="wm-logo-marker"></span>', unsafe_allow_html=True)
        if asset_exists(LOGO_PATH):
            st.image(str(LOGO_PATH), use_container_width=True)
        else:
            st.markdown("<div style='font-size:1.5rem;'>🍉</div>", unsafe_allow_html=True)

    with text_col:
        st.markdown(
            """
            <div class="wm-brand-row" style="margin-top:0;">
                <div>
                    <div class="wm-brand-title">Industrial Vibration Intelligence</div>
                    <div class="wm-brand-subtitle">Rotating machinery diagnostics platform</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="wm-hero">
            Diagnóstico avanzado para
            <span class="accent">máquinas críticas.</span>
        </div>
        <div class="wm-left-note">
            Plataforma corporativa para monitoreo y análisis rotodinámico
            de turbomaquinaria, generación, compresión y bombeo —
            alineada con API 670 / API 684 / ISO 20816.
        </div>
        <div class="wm-trust-row">
            <span class="wm-trust-chip">🔒 SSO-ready</span>
            <span class="wm-trust-chip">📊 API 670 / 684</span>
            <span class="wm-trust-chip">🌐 ISO 20816</span>
            <span class="wm-trust-chip">⚙️ Multi-instance</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with right_col:
    # Marker invisible para que :has() detecte ESTA columna y le
    # aplique los estilos de card glassmorphism. Antes envolvíamos
    # con <div class="wm-login-card"> pero Streamlit lo renderizaba
    # VACÍO arriba del form (caja blanca fantasma).
    st.markdown('<span class="wm-login-marker"></span>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="wm-login-top">🔐 Secure access</div>
        <div class="wm-login-title">Ingresar</div>
        <div class="wm-login-copy">Acceso con credenciales corporativas. Las sesiones quedan auditadas para trazabilidad.</div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("wm_login_form", clear_on_submit=False):
        username = st.text_input(
            "Correo corporativo",
            placeholder="nombre.apellido@sigasas.com",
            key="wm_login_username",
            autocomplete="email",
            help=(
                "Para usuarios SIGASAS: tu correo @sigasas.com. "
                "Para clientes: el correo registrado por tu administrador."
            ),
        )

        password = st.text_input(
            "Contraseña",
            placeholder="••••••••••••",
            type="password",
            key="wm_login_password",
            autocomplete="current-password",
        )

        submit = st.form_submit_button("Iniciar sesión", use_container_width=True)

    if submit:
        ok, msg = wm_login(username.strip(), password)
        if ok:
            st.success(msg)
            st.switch_page("pages/_landing.py")
        else:
            st.error(msg)

    # =====================================================================
    # Ciclo 17.16 — "Olvidé mi contraseña"
    # =====================================================================
    # Link discreto debajo del form de login. Al expandir, pide email y
    # llama core.password_reset.request_reset. Por seguridad, NO revela
    # si el email existe o no — siempre muestra mensaje genérico.
    with st.expander("¿Olvidaste tu contraseña?", expanded=False):
        _reset_email = st.text_input(
            "Tu correo corporativo",
            placeholder="nombre.apellido@sigasas.com",
            key="wm_reset_email",
            help="Te enviaremos un link para elegir nueva clave. Válido por 1 hora.",
        ).strip().lower()
        if st.button("Enviar link de recuperación",
                     use_container_width=True,
                     key="wm_reset_request_btn"):
            if not _reset_email or "@" not in _reset_email:
                st.error("Ingresá un email válido.")
            else:
                try:
                    from core.password_reset import request_reset
                    # Detectar la URL base de la app actual
                    _base_url = ""
                    try:
                        from streamlit.runtime.scriptrunner import get_script_run_ctx
                        # Streamlit no expone fácilmente la URL pública;
                        # usamos un secret opcional o el default conocido.
                        try:
                            _base_url = (
                                st.secrets.get("app", {}).get("base_url", "")
                                or "https://wm-home-final-2026.streamlit.app"
                            )
                        except Exception:
                            _base_url = "https://wm-home-final-2026.streamlit.app"
                    except Exception:
                        _base_url = "https://wm-home-final-2026.streamlit.app"

                    res = request_reset(_reset_email, base_url=_base_url)
                    if res.get("ok"):
                        st.success(
                            "✓ " + res.get("message", "Si el email existe, "
                            "recibirás instrucciones en breve.")
                        )
                        if res.get("_debug"):
                            # Solo visible si hay un secret app.debug=true
                            try:
                                if st.secrets.get("app", {}).get("debug"):
                                    st.caption(f"🔧 debug: {res['_debug']}")
                            except Exception:
                                pass
                    else:
                        st.error(
                            "No se pudo iniciar el reset: "
                            + res.get("error", "error desconocido")
                        )
                except Exception as e:
                    st.error(f"Error inesperado: {e}")

    # Ciclo 17.6.3 — versión real desde core.version (deriva de
    # git tags, no hardcoded). Pinta un chip con el environment
    # coloreado para distinguir production / development a simple
    # vista.
    _vinfo = get_version_info()
    _env_color = {
        "production":  ("#10b981", "#ecfdf5"),  # verde (prod estable)
        "staging":     ("#f59e0b", "#fef3c7"),  # ámbar
        "development": ("#0ea5e9", "#e0f2fe"),  # azul (dev)
    }.get(_vinfo["environment"], ("#64748b", "#f1f5f9"))
    _env_chip = (
        f"<span style='display:inline-flex;align-items:center;gap:0.3rem;"
        f"padding:0.12rem 0.5rem;border-radius:999px;"
        f"background:{_env_color[1]};color:{_env_color[0]};"
        f"font-size:0.68rem;font-weight:700;text-transform:uppercase;"
        f"letter-spacing:0.06em;border:1px solid {_env_color[0]}33;'>"
        f"{_vinfo['environment']}</span>"
    )
    _build_extras = []
    if _vinfo["commit"]:
        _build_extras.append(_vinfo["commit"])
    if _vinfo["date"]:
        _build_extras.append(_vinfo["date"])
    _build_line = " · ".join(_build_extras)

    st.markdown(
        f"""
        <div class="wm-footer-note">
            <div>Watermelon System — Industrial monitoring software</div>
            <div class="wm-footer-build">
                <b>{_vinfo['version']}</b> {_env_chip}
                {('· ' + _build_line) if _build_line else ''}
            </div>
            <div class="wm-footer-build" style="font-size:0.7rem;opacity:0.7;">
                © 2026 SIGASAS · All rights reserved
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )