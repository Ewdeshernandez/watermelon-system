from __future__ import annotations

from pathlib import Path

import streamlit as st

from core.auth import (
    is_authenticated,
    login as wm_login,
    render_login_shell,
)

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
            radial-gradient(circle at 10% 12%, rgba(77, 208, 255, 0.16) 0%, transparent 24%),
            radial-gradient(circle at 90% 18%, rgba(103, 114, 255, 0.10) 0%, transparent 20%),
            linear-gradient(180deg, #f7fbff 0%, #eef5ff 100%);
        color: #0f172a;
    }

    .block-container {
        max-width: 1380px !important;
        padding-top: 1.2rem !important;
        padding-bottom: 1.2rem !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
    }

    .wm-shell {
        min-height: 90vh;
        border-radius: 28px;
        overflow: hidden;
        border: 1px solid rgba(15, 23, 42, 0.06);
        background:
            linear-gradient(135deg, rgba(255,255,255,0.92), rgba(255,255,255,0.78));
        box-shadow:
            0 24px 80px rgba(32, 84, 146, 0.12),
            inset 0 1px 0 rgba(255,255,255,0.85);
        padding: 1rem;
    }

    .wm-left-panel {
        min-height: 82vh;
        padding: 3.2rem 3rem;
        border-radius: 24px;
        background:
            linear-gradient(180deg, rgba(255,255,255,0.76), rgba(255,255,255,0.56));
        border: 1px solid rgba(15, 23, 42, 0.06);
        box-shadow:
            inset 0 1px 0 rgba(255,255,255,0.8),
            0 12px 36px rgba(57, 93, 135, 0.08);
    }

    .wm-right-panel {
        min-height: 82vh;
        padding: 2rem 1.8rem;
        border-radius: 24px;
        background:
            linear-gradient(180deg, #0f172a 0%, #111c32 100%);
        border: 1px solid rgba(255,255,255,0.06);
        box-shadow:
            0 22px 60px rgba(15, 23, 42, 0.24),
            inset 0 1px 0 rgba(255,255,255,0.04);
    }

    .wm-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.48rem 0.85rem;
        border-radius: 999px;
        background: rgba(30,167,255,0.10);
        border: 1px solid rgba(30,167,255,0.18);
        color: #1192ff;
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }

    .wm-brand-row {
        display: flex;
        align-items: center;
        gap: 0.9rem;
        margin-top: 1.2rem;
    }

    .wm-logo-box {
        width: 62px;
        height: 62px;
        border-radius: 18px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: linear-gradient(180deg, #ffffff, #eef6ff);
        border: 1px solid rgba(15, 23, 42, 0.06);
        box-shadow: 0 10px 24px rgba(57, 93, 135, 0.10);
        overflow: hidden;
    }

    .wm-brand-name {
        color: #0f172a;
        font-size: 0.95rem;
        font-weight: 800;
        letter-spacing: 0.16em;
        text-transform: uppercase;
    }

    .wm-brand-sub {
        color: #4b5563;
        font-size: 0.92rem;
        font-weight: 600;
        margin-top: 0.2rem;
    }

    .wm-title {
        margin-top: 1.5rem;
        font-size: 4.25rem;
        line-height: 0.95;
        font-weight: 950;
        letter-spacing: -0.055em;
        color: #0b1324;
        max-width: 760px;
    }

    .wm-subtitle {
        margin-top: 1rem;
        max-width: 640px;
        color: #475569;
        font-size: 1.05rem;
        line-height: 1.75;
    }

    .wm-chips {
        display: flex;
        gap: 0.65rem;
        flex-wrap: wrap;
        margin-top: 1.35rem;
    }

    .wm-chip {
        padding: 0.55rem 0.82rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.72);
        border: 1px solid rgba(15, 23, 42, 0.06);
        color: #334155;
        font-size: 0.80rem;
        font-weight: 700;
    }

    .wm-preview-card {
        margin-top: 2rem;
        border-radius: 22px;
        padding: 1.1rem;
        background: linear-gradient(180deg, #ffffff, #f5f9ff);
        border: 1px solid rgba(15, 23, 42, 0.06);
        box-shadow: 0 14px 34px rgba(57, 93, 135, 0.08);
    }

    .wm-preview-head {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.95rem;
    }

    .wm-preview-title {
        color: #0f172a;
        font-size: 1rem;
        font-weight: 800;
    }

    .wm-preview-status {
        padding: 0.38rem 0.7rem;
        border-radius: 999px;
        background: rgba(16,185,129,0.10);
        border: 1px solid rgba(16,185,129,0.18);
        color: #059669;
        font-size: 0.74rem;
        font-weight: 800;
        letter-spacing: 0.10em;
        text-transform: uppercase;
    }

    .wm-chart {
        position: relative;
        height: 240px;
        border-radius: 18px;
        overflow: hidden;
        background:
            linear-gradient(180deg, #eaf4ff 0%, #f7fbff 100%);
        border: 1px solid rgba(15, 23, 42, 0.05);
    }

    .wm-chart-grid {
        position: absolute;
        inset: 0;
        background-image:
            linear-gradient(rgba(15, 23, 42, 0.05) 1px, transparent 1px),
            linear-gradient(90deg, rgba(15, 23, 42, 0.05) 1px, transparent 1px);
        background-size: 42px 42px;
    }

    .wm-wave {
        position: absolute;
        left: 4%;
        width: 92%;
        border-radius: 999px;
    }

    .wm-wave-1 {
        top: 28%;
        height: 72px;
        background: linear-gradient(90deg, #28a8ff 0%, #78d7ff 100%);
        clip-path: polygon(0% 62%, 8% 48%, 16% 66%, 24% 35%, 32% 54%, 40% 42%, 48% 64%, 56% 31%, 64% 58%, 72% 43%, 80% 62%, 88% 39%, 100% 54%, 100% 100%, 0% 100%);
        opacity: 0.96;
    }

    .wm-wave-2 {
        top: 56%;
        height: 62px;
        background: linear-gradient(90deg, #ff5f8f 0%, #ff96b2 100%);
        clip-path: polygon(0% 58%, 10% 42%, 20% 60%, 30% 32%, 40% 55%, 50% 38%, 60% 62%, 70% 37%, 80% 57%, 90% 40%, 100% 52%, 100% 100%, 0% 100%);
        opacity: 0.92;
    }

    .wm-stats {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.75rem;
        margin-top: 0.9rem;
    }

    .wm-stat {
        padding: 0.9rem;
        border-radius: 18px;
        background: rgba(255,255,255,0.84);
        border: 1px solid rgba(15, 23, 42, 0.06);
    }

    .wm-stat-value {
        color: #0f172a;
        font-size: 1.15rem;
        font-weight: 900;
        letter-spacing: -0.03em;
    }

    .wm-stat-label {
        margin-top: 0.18rem;
        color: #64748b;
        font-size: 0.82rem;
    }

    .wm-login-kicker {
        color: #53c4ff;
        font-size: 0.80rem;
        font-weight: 800;
        letter-spacing: 0.14em;
        text-transform: uppercase;
    }

    .wm-login-title {
        margin-top: 0.5rem;
        color: #f8fbff;
        font-size: 2rem;
        font-weight: 900;
        letter-spacing: -0.04em;
    }

    .wm-login-copy {
        margin-top: 0.7rem;
        color: #9eb0c8;
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
        color: #e5eefb !important;
        font-weight: 700 !important;
        font-size: 0.92rem !important;
    }

    div[data-testid="stTextInput"] > div > div {
        background: rgba(255,255,255,0.07) !important;
        border: 1px solid rgba(255,255,255,0.10) !important;
        border-radius: 16px !important;
        min-height: 54px !important;
        transition: all 0.18s ease !important;
    }

    div[data-testid="stTextInput"] > div > div:focus-within {
        border: 1px solid rgba(76, 201, 255, 0.70) !important;
        box-shadow:
            0 0 0 3px rgba(76, 201, 255, 0.14),
            0 0 20px rgba(76, 201, 255, 0.18) !important;
    }

    div[data-testid="stTextInput"] input {
        color: #f8fbff !important;
        font-size: 1rem !important;
    }

    div[data-testid="stTextInput"] input::placeholder {
        color: #91a5bf !important;
        opacity: 1 !important;
    }

    div[data-testid="stFormSubmitButton"] > button {
        width: 100% !important;
        height: 54px !important;
        margin-top: 0.55rem !important;
        border: 0 !important;
        border-radius: 16px !important;
        color: #ffffff !important;
        font-weight: 900 !important;
        font-size: 1rem !important;
        background: linear-gradient(90deg, #1293ff 0%, #3bc3ff 55%, #ff5f8f 100%) !important;
        box-shadow:
            0 16px 34px rgba(18, 147, 255, 0.24),
            0 8px 22px rgba(255, 95, 143, 0.14) !important;
        transition: all 0.18s ease !important;
    }

    div[data-testid="stFormSubmitButton"] > button:hover {
        transform: translateY(-1px);
        box-shadow:
            0 20px 40px rgba(18, 147, 255, 0.30),
            0 10px 26px rgba(255, 95, 143, 0.18) !important;
    }

    .wm-note {
        margin-top: 1rem;
        padding: 0.95rem 1rem;
        border-radius: 16px;
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.08);
        color: #a8bad1;
        font-size: 0.88rem;
        line-height: 1.55;
    }

    div[data-testid="stAlert"] {
        border-radius: 16px !important;
        border: 1px solid rgba(255,255,255,0.10) !important;
    }

    @media (max-width: 1200px) {
        .wm-title {
            font-size: 3.3rem;
        }
        .wm-stats {
            grid-template-columns: 1fr;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# LAYOUT
# =========================================================
st.markdown('<div class="wm-shell">', unsafe_allow_html=True)

left_col, right_col = st.columns([1.55, 0.92], gap="large")

with left_col:
    st.markdown('<div class="wm-left-panel">', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="wm-badge">● Watermelon System</div>
        """,
        unsafe_allow_html=True,
    )

    logo_col, brand_col = st.columns([0.12, 0.88], gap="small")

    with logo_col:
        st.markdown('<div class="wm-logo-box">', unsafe_allow_html=True)
        if asset_exists(LOGO_PATH):
            st.image(str(LOGO_PATH), use_container_width=True)
        else:
            st.markdown(
                "<div style='font-size:1.8rem;'>🍉</div>",
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    with brand_col:
        st.markdown(
            """
            <div class="wm-brand-row" style="margin-top:0;">
                <div>
                    <div class="wm-brand-name">Industrial Vibration Intelligence</div>
                    <div class="wm-brand-sub">Modern monitoring platform</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="wm-title">
            Clean.<br>
            Fast.<br>
            Industrial.
        </div>
        <div class="wm-subtitle">
            Una entrada más liviana, moderna y premium para Watermelon System.
        </div>
        <div class="wm-chips">
            <div class="wm-chip">Waveform</div>
            <div class="wm-chip">Orbit</div>
            <div class="wm-chip">FFT</div>
            <div class="wm-chip">Trends</div>
            <div class="wm-chip">Diagnostics</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="wm-preview-card">
            <div class="wm-preview-head">
                <div class="wm-preview-title">System Preview</div>
                <div class="wm-preview-status">Online</div>
            </div>
            <div class="wm-chart">
                <div class="wm-chart-grid"></div>
                <div class="wm-wave wm-wave-1"></div>
                <div class="wm-wave wm-wave-2"></div>
            </div>
            <div class="wm-stats">
                <div class="wm-stat">
                    <div class="wm-stat-value">Fast UI</div>
                    <div class="wm-stat-label">Interfaz clara y rápida</div>
                </div>
                <div class="wm-stat">
                    <div class="wm-stat-value">Modern UX</div>
                    <div class="wm-stat-label">Diseño premium industrial</div>
                </div>
                <div class="wm-stat">
                    <div class="wm-stat-value">HD Export</div>
                    <div class="wm-stat-label">Salida profesional</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="wm-right-panel">', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="wm-login-kicker">Secure Access</div>
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
        <div class="wm-note">
            Monitoreo y análisis de vibraciones con experiencia visual moderna.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)