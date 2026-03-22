from __future__ import annotations

from pathlib import Path

import streamlit as st

from core.auth import is_authenticated, login as wm_login, render_login_shell

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

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(30,167,255,0.10) 0%, transparent 24%),
            linear-gradient(180deg, #edf3fa 0%, #e8eef6 100%);
        color: #0f172a;
    }

    .block-container {
        max-width: 1280px !important;
        padding-top: 2.4rem !important;
        padding-bottom: 2.4rem !important;
        padding-left: 1.8rem !important;
        padding-right: 1.8rem !important;
    }

    /* columnas */
    [data-testid="column"] {
        display: flex;
        align-items: center;
    }

    /* izquierda */
    .wm-eyebrow {
        display: inline-block;
        padding: 0.45rem 0.8rem;
        border-radius: 999px;
        background: rgba(30,167,255,0.08);
        border: 1px solid rgba(30,167,255,0.16);
        color: #148df0;
        font-size: 0.76rem;
        font-weight: 800;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }

    .wm-brand-row {
        display: flex;
        align-items: center;
        gap: 0.95rem;
        margin-top: 1rem;
        margin-bottom: 1.25rem;
    }

    .wm-logo-box {
        width: 58px;
        height: 58px;
        border-radius: 16px;
        background: #ffffff;
        border: 1px solid rgba(15, 23, 42, 0.06);
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.05);
        display: flex;
        align-items: center;
        justify-content: center;
        overflow: hidden;
        flex-shrink: 0;
    }

    .wm-brand-title {
        color: #0f172a;
        font-size: 0.95rem;
        font-weight: 800;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        margin: 0;
    }

    .wm-brand-subtitle {
        color: #617287;
        font-size: 0.92rem;
        font-weight: 600;
        margin-top: 0.12rem;
    }

    .wm-hero {
        color: #081226;
        font-size: 4.4rem;
        line-height: 0.94;
        font-weight: 900;
        letter-spacing: -0.06em;
        max-width: 650px;
        margin: 0;
    }

    .wm-left-note {
        margin-top: 1rem;
        color: #617287;
        font-size: 1rem;
        line-height: 1.65;
        max-width: 480px;
    }

    /* tarjeta login */
    .wm-login-card {
        width: 100%;
        max-width: 460px;
        margin-left: auto;
        background: #ffffff;
        border: 1px solid rgba(15, 23, 42, 0.07);
        border-radius: 24px;
        box-shadow:
            0 18px 46px rgba(15, 23, 42, 0.08),
            0 2px 8px rgba(15, 23, 42, 0.03);
        padding: 2rem 1.8rem;
    }

    .wm-login-top {
        color: #148df0;
        font-size: 0.76rem;
        font-weight: 800;
        letter-spacing: 0.13em;
        text-transform: uppercase;
    }

    .wm-login-title {
        margin-top: 0.55rem;
        color: #0f172a;
        font-size: 2rem;
        line-height: 1.0;
        font-weight: 900;
        letter-spacing: -0.04em;
    }

    .wm-login-copy {
        margin-top: 0.7rem;
        margin-bottom: 1.2rem;
        color: #64748b;
        font-size: 0.95rem;
        line-height: 1.6;
    }

    div[data-testid="stForm"] {
        background: transparent !important;
        border: 0 !important;
        padding: 0 !important;
    }

    div[data-testid="stTextInput"] label {
        color: #243244 !important;
        font-size: 0.91rem !important;
        font-weight: 700 !important;
    }

    div[data-testid="stTextInput"] > div > div {
        background: #f8fbff !important;
        border: 1px solid #d8e3ef !important;
        border-radius: 14px !important;
        min-height: 52px !important;
        transition: all 0.18s ease !important;
    }

    div[data-testid="stTextInput"] > div > div:focus-within {
        border: 1px solid #53baff !important;
        box-shadow: 0 0 0 4px rgba(83,186,255,0.14) !important;
    }

    div[data-testid="stTextInput"] input {
        color: #0f172a !important;
        font-size: 1rem !important;
    }

    div[data-testid="stTextInput"] input::placeholder {
        color: #8ca1b8 !important;
        opacity: 1 !important;
    }

    div[data-testid="stFormSubmitButton"] > button {
        width: 100% !important;
        height: 52px !important;
        margin-top: 0.55rem !important;
        border: 0 !important;
        border-radius: 14px !important;
        font-size: 1rem !important;
        font-weight: 800 !important;
        color: #ffffff !important;
        background: linear-gradient(90deg, #1593ff 0%, #39c1ff 100%) !important;
        box-shadow: 0 12px 28px rgba(21,147,255,0.22) !important;
        transition: all 0.18s ease !important;
    }

    div[data-testid="stFormSubmitButton"] > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 16px 32px rgba(21,147,255,0.28) !important;
    }

    .wm-footer-note {
        margin-top: 1rem;
        padding-top: 0.9rem;
        border-top: 1px solid #e7edf5;
        color: #7a8da3;
        font-size: 0.86rem;
        line-height: 1.5;
    }

    div[data-testid="stAlert"] {
        border-radius: 14px !important;
        border: 1px solid rgba(15, 23, 42, 0.08) !important;
    }

    @media (max-width: 1100px) {
        .wm-hero {
            font-size: 3.2rem;
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
        <div class="wm-eyebrow">● Watermelon System</div>
        """,
        unsafe_allow_html=True,
    )

    logo_col, text_col = st.columns([0.12, 0.88], gap="small")

    with logo_col:
        st.markdown('<div class="wm-logo-box">', unsafe_allow_html=True)
        if asset_exists(LOGO_PATH):
            st.image(str(LOGO_PATH), use_container_width=True)
        else:
            st.markdown("<div style='font-size:1.5rem;'>🍉</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with text_col:
        st.markdown(
            """
            <div class="wm-brand-row" style="margin-top:0;">
                <div>
                    <div class="wm-brand-title">Industrial Vibration Intelligence</div>
                    <div class="wm-brand-subtitle">Modern monitoring platform</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="wm-hero">
            Inicia sesión
            en Watermelon.
        </div>
        <div class="wm-left-note">
            Plataforma industrial para análisis, monitoreo y diagnóstico de vibraciones.
        </div>
        """,
        unsafe_allow_html=True,
    )

with right_col:
    st.markdown('<div class="wm-login-card">', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="wm-login-top">Secure Access</div>
        <div class="wm-login-title">Ingresar</div>
        <div class="wm-login-copy">Accede con tus credenciales corporativas.</div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("wm_login_form", clear_on_submit=False):
        username = st.text_input(
            "Usuario o correo",
            placeholder="usuario o correo",
            key="wm_login_username",
        )

        password = st.text_input(
            "Clave",
            placeholder="Ingresa tu contraseña",
            type="password",
            key="wm_login_password",
        )

        submit = st.form_submit_button("Entrar", use_container_width=True)

    if submit:
        ok, msg = wm_login(username.strip(), password)
        if ok:
            st.success(msg)
            st.switch_page("00_Home.py")
        else:
            st.error(msg)

    st.markdown(
        """
        <div class="wm-footer-note">
            Watermelon System · Industrial monitoring software
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)