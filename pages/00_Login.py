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
            radial-gradient(circle at top left, rgba(30,167,255,0.10) 0%, transparent 26%),
            linear-gradient(180deg, #eef4fb 0%, #e8eef7 100%);
        color: #0f172a;
    }

    .block-container {
        max-width: 1320px !important;
        padding-top: 2.2rem !important;
        padding-bottom: 2.2rem !important;
        padding-left: 1.6rem !important;
        padding-right: 1.6rem !important;
    }

    .wm-stage {
        min-height: 88vh;
        display: flex;
        align-items: center;
    }

    .wm-left {
        padding-right: 2.5rem;
    }

    .wm-kicker {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.45rem 0.82rem;
        border-radius: 999px;
        background: rgba(30,167,255,0.08);
        border: 1px solid rgba(30,167,255,0.16);
        color: #1187ea;
        font-size: 0.76rem;
        font-weight: 800;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }

    .wm-brand-row {
        display: flex;
        align-items: center;
        gap: 0.9rem;
        margin-top: 1rem;
    }

    .wm-logo-box {
        width: 56px;
        height: 56px;
        border-radius: 16px;
        background: #ffffff;
        border: 1px solid rgba(15, 23, 42, 0.06);
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.05);
        display: flex;
        align-items: center;
        justify-content: center;
        overflow: hidden;
    }

    .wm-brand-title {
        font-size: 0.95rem;
        font-weight: 800;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: #0f172a;
        margin: 0;
    }

    .wm-brand-subtitle {
        font-size: 0.92rem;
        color: #5b6b80;
        font-weight: 600;
        margin-top: 0.12rem;
    }

    .wm-title {
        margin-top: 1.7rem;
        font-size: 4.25rem;
        line-height: 0.96;
        font-weight: 900;
        letter-spacing: -0.055em;
        color: #091224;
        max-width: 620px;
    }

    .wm-copy {
        margin-top: 1rem;
        max-width: 520px;
        color: #5b6b80;
        font-size: 1.02rem;
        line-height: 1.7;
    }

    .wm-tags {
        display: flex;
        flex-wrap: wrap;
        gap: 0.65rem;
        margin-top: 1.4rem;
    }

    .wm-tag {
        padding: 0.52rem 0.82rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.78);
        border: 1px solid rgba(15, 23, 42, 0.06);
        color: #334155;
        font-size: 0.80rem;
        font-weight: 700;
    }

    .wm-card {
        background: #ffffff;
        border: 1px solid rgba(15, 23, 42, 0.07);
        border-radius: 24px;
        box-shadow:
            0 20px 50px rgba(15, 23, 42, 0.08),
            0 2px 8px rgba(15, 23, 42, 0.03);
        padding: 2rem 1.8rem;
        max-width: 440px;
        margin-left: auto;
    }

    .wm-card-topline {
        color: #1187ea;
        font-size: 0.75rem;
        font-weight: 800;
        letter-spacing: 0.14em;
        text-transform: uppercase;
    }

    .wm-card-title {
        margin-top: 0.55rem;
        color: #0f172a;
        font-size: 2rem;
        line-height: 1.0;
        font-weight: 900;
        letter-spacing: -0.04em;
    }

    .wm-card-copy {
        margin-top: 0.7rem;
        color: #64748b;
        font-size: 0.95rem;
        line-height: 1.6;
        margin-bottom: 1.2rem;
    }

    div[data-testid="stForm"] {
        background: transparent !important;
        border: 0 !important;
        padding: 0 !important;
    }

    div[data-testid="stTextInput"] label {
        color: #243244 !important;
        font-size: 0.90rem !important;
        font-weight: 700 !important;
    }

    div[data-testid="stTextInput"] > div > div {
        background: #f8fbff !important;
        border: 1px solid #d9e4f1 !important;
        border-radius: 14px !important;
        min-height: 52px !important;
        transition: all 0.18s ease !important;
    }

    div[data-testid="stTextInput"] > div > div:focus-within {
        border: 1px solid #51b7ff !important;
        box-shadow: 0 0 0 4px rgba(81,183,255,0.14) !important;
    }

    div[data-testid="stTextInput"] input {
        color: #0f172a !important;
        font-size: 1rem !important;
    }

    div[data-testid="stTextInput"] input::placeholder {
        color: #8aa0b8 !important;
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
        background: linear-gradient(90deg, #1593ff 0%, #32c2ff 100%) !important;
        box-shadow: 0 12px 28px rgba(21,147,255,0.22) !important;
        transition: all 0.18s ease !important;
    }

    div[data-testid="stFormSubmitButton"] > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 16px 32px rgba(21,147,255,0.28) !important;
    }

    .wm-note {
        margin-top: 1rem;
        padding-top: 0.9rem;
        border-top: 1px solid #e7eef6;
        color: #7a8da3;
        font-size: 0.86rem;
        line-height: 1.55;
    }

    div[data-testid="stAlert"] {
        border-radius: 14px !important;
        border: 1px solid rgba(15, 23, 42, 0.08) !important;
    }

    @media (max-width: 1100px) {
        .wm-left {
            padding-right: 0.5rem;
            margin-bottom: 2rem;
        }

        .wm-title {
            font-size: 3.2rem;
            max-width: 100%;
        }

        .wm-card {
            margin-left: 0;
            max-width: 100%;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# LAYOUT
# =========================================================
st.markdown('<div class="wm-stage">', unsafe_allow_html=True)

left_col, right_col = st.columns([1.25, 0.95], gap="large")

with left_col:
    st.markdown('<div class="wm-left">', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="wm-kicker">● Watermelon System</div>
        """,
        unsafe_allow_html=True,
    )

    brand_logo_col, brand_text_col = st.columns([0.12, 0.88], gap="small")

    with brand_logo_col:
        st.markdown('<div class="wm-logo-box">', unsafe_allow_html=True)
        if asset_exists(LOGO_PATH):
            st.image(str(LOGO_PATH), use_container_width=True)
        else:
            st.markdown("<div style='font-size:1.5rem;'>🍉</div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with brand_text_col:
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
        <div class="wm-title">
            Acceso premium,
            serio y limpio.
        </div>
        <div class="wm-copy">
            Una entrada sobria para una plataforma industrial moderna.
        </div>
        <div class="wm-tags">
            <div class="wm-tag">Waveform</div>
            <div class="wm-tag">Orbit</div>
            <div class="wm-tag">FFT</div>
            <div class="wm-tag">Trends</div>
            <div class="wm-tag">Diagnostics</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="wm-card">', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="wm-card-topline">Secure Access</div>
        <div class="wm-card-title">Ingresar</div>
        <div class="wm-card-copy">Accede con tus credenciales corporativas.</div>
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
        <div class="wm-note">
            Watermelon System · Industrial monitoring software
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)