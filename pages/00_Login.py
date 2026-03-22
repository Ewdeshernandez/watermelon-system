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
# HELPERS
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
# AUTH STATE
# =========================================================
if is_authenticated():
    st.switch_page("00_Home.py")

render_login_shell()

# =========================================================
# CSS - ULTRA PREMIUM LOGIN
# =========================================================
st.markdown(
    """
    <style>
    header, #MainMenu, footer {visibility: hidden;}
    [data-testid="stSidebar"] {display: none !important;}

    .stApp {
        background:
            radial-gradient(circle at 8% 15%, rgba(37, 203, 255, 0.20) 0%, transparent 22%),
            radial-gradient(circle at 88% 82%, rgba(255, 72, 113, 0.18) 0%, transparent 28%),
            radial-gradient(circle at 75% 22%, rgba(101, 255, 179, 0.10) 0%, transparent 16%),
            linear-gradient(135deg, #040b16 0%, #08111f 32%, #0b1630 65%, #101326 100%);
        color: #f6fbff;
        overflow: hidden;
    }

    .block-container {
        max-width: 1480px !important;
        padding-top: 1.35rem !important;
        padding-bottom: 1.35rem !important;
        padding-left: 1.75rem !important;
        padding-right: 1.75rem !important;
    }

    .wm-page-shell {
        position: relative;
        min-height: 92vh;
        border-radius: 30px;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.07);
        background:
            linear-gradient(180deg, rgba(255,255,255,0.035), rgba(255,255,255,0.018));
        box-shadow:
            0 30px 100px rgba(0,0,0,0.50),
            inset 0 1px 0 rgba(255,255,255,0.04);
        padding: 1rem;
    }

    .wm-page-shell::before {
        content: "";
        position: absolute;
        inset: 0;
        background:
            linear-gradient(120deg, rgba(255,255,255,0.02), transparent 32%),
            radial-gradient(circle at 18% 18%, rgba(30,167,255,0.10), transparent 25%),
            radial-gradient(circle at 82% 78%, rgba(255,77,109,0.10), transparent 24%);
        pointer-events: none;
    }

    .wm-grid-bg {
        position: absolute;
        inset: 0;
        opacity: 0.18;
        background-image:
            linear-gradient(rgba(255,255,255,0.06) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255,255,255,0.06) 1px, transparent 1px);
        background-size: 42px 42px;
        mask-image: radial-gradient(circle at center, black 40%, transparent 92%);
        -webkit-mask-image: radial-gradient(circle at center, black 40%, transparent 92%);
        pointer-events: none;
    }

    .wm-left-panel {
        position: relative;
        min-height: 84vh;
        padding: 3.2rem 3rem 2.6rem 3rem;
        border-radius: 26px;
        background:
            linear-gradient(180deg, rgba(10, 20, 36, 0.88), rgba(7, 14, 26, 0.78));
        border: 1px solid rgba(255,255,255,0.06);
        box-shadow:
            inset 0 1px 0 rgba(255,255,255,0.04),
            0 22px 50px rgba(0,0,0,0.26);
        overflow: hidden;
    }

    .wm-left-panel::before {
        content: "";
        position: absolute;
        width: 520px;
        height: 520px;
        right: -170px;
        top: -120px;
        background: radial-gradient(circle, rgba(30,167,255,0.22) 0%, transparent 62%);
        filter: blur(24px);
        pointer-events: none;
    }

    .wm-left-panel::after {
        content: "";
        position: absolute;
        width: 420px;
        height: 420px;
        left: -120px;
        bottom: -170px;
        background: radial-gradient(circle, rgba(255,77,109,0.16) 0%, transparent 62%);
        filter: blur(28px);
        pointer-events: none;
    }

    .wm-topline {
        display: inline-flex;
        align-items: center;
        gap: 0.55rem;
        padding: 0.5rem 0.95rem;
        border-radius: 999px;
        background: rgba(30,167,255,0.11);
        border: 1px solid rgba(30,167,255,0.28);
        color: #87ddff;
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        backdrop-filter: blur(10px);
    }

    .wm-brand-row {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin-top: 1rem;
        margin-bottom: 0.45rem;
    }

    .wm-logo-wrap {
        width: 66px;
        height: 66px;
        border-radius: 18px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: linear-gradient(180deg, rgba(255,255,255,0.12), rgba(255,255,255,0.04));
        border: 1px solid rgba(255,255,255,0.09);
        box-shadow: 0 12px 28px rgba(0,0,0,0.22);
        overflow: hidden;
    }

    .wm-brand-text {
        color: #dff3ff;
        font-size: 0.96rem;
        font-weight: 700;
        letter-spacing: 0.18em;
        text-transform: uppercase;
    }

    .wm-title {
        margin-top: 1.15rem;
        font-size: 4.45rem;
        line-height: 0.93;
        font-weight: 950;
        letter-spacing: -0.05em;
        color: #f7fbff;
        max-width: 780px;
    }

    .wm-subtitle {
        margin-top: 1.15rem;
        max-width: 720px;
        color: #aebcd0;
        font-size: 1.08rem;
        line-height: 1.78;
    }

    .wm-value-strip {
        display: flex;
        gap: 0.7rem;
        flex-wrap: wrap;
        margin-top: 1.35rem;
    }

    .wm-chip {
        padding: 0.58rem 0.88rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.08);
        color: #c4d4e6;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.01em;
    }

    .wm-preview-card {
        position: relative;
        margin-top: 2rem;
        border-radius: 24px;
        padding: 1.1rem;
        background: linear-gradient(180deg, rgba(255,255,255,0.045), rgba(255,255,255,0.02));
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
        overflow: hidden;
    }

    .wm-preview-head {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.95rem;
    }

    .wm-preview-title {
        color: #f2f8ff;
        font-size: 1rem;
        font-weight: 800;
        letter-spacing: -0.02em;
    }

    .wm-preview-badge {
        padding: 0.36rem 0.7rem;
        border-radius: 999px;
        background: rgba(101,255,179,0.10);
        border: 1px solid rgba(101,255,179,0.26);
        color: #8effbf;
        font-size: 0.74rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.12em;
    }

    .wm-chart {
        position: relative;
        height: 250px;
        border-radius: 20px;
        background:
            linear-gradient(180deg, rgba(7,16,31,0.95), rgba(10,18,36,0.88));
        border: 1px solid rgba(255,255,255,0.07);
        overflow: hidden;
    }

    .wm-chart-grid {
        position: absolute;
        inset: 0;
        background-image:
            linear-gradient(rgba(255,255,255,0.06) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255,255,255,0.06) 1px, transparent 1px);
        background-size: 40px 40px;
        opacity: 0.32;
    }

    .wm-chart-glow {
        position: absolute;
        inset: 0;
        background:
            radial-gradient(circle at 12% 70%, rgba(30,167,255,0.16), transparent 25%),
            radial-gradient(circle at 86% 35%, rgba(255,77,109,0.14), transparent 24%);
    }

    .wm-line {
        position: absolute;
        left: 4%;
        width: 92%;
        height: 3px;
        border-radius: 999px;
        opacity: 0.95;
        filter: drop-shadow(0 0 10px rgba(0,0,0,0.2));
    }

    .wm-line-1 {
        top: 30%;
        background: linear-gradient(90deg, transparent 0%, #18b4ff 14%, #7ad7ff 55%, #18b4ff 100%);
        clip-path: polygon(0% 55%, 6% 46%, 12% 58%, 18% 30%, 24% 64%, 30% 40%, 36% 52%, 42% 26%, 48% 61%, 54% 43%, 60% 49%, 66% 35%, 72% 57%, 78% 44%, 84% 60%, 90% 38%, 100% 48%, 100% 100%, 0% 100%);
        height: 84px;
    }

    .wm-line-2 {
        top: 52%;
        background: linear-gradient(90deg, transparent 0%, #ff5c7a 12%, #ff9ab0 58%, #ff5c7a 100%);
        clip-path: polygon(0% 60%, 8% 48%, 16% 65%, 24% 36%, 32% 54%, 40% 43%, 48% 67%, 56% 31%, 64% 58%, 72% 42%, 80% 63%, 88% 39%, 100% 57%, 100% 100%, 0% 100%);
        height: 92px;
    }

    .wm-metrics-row {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.8rem;
        margin-top: 0.95rem;
    }

    .wm-metric {
        padding: 0.85rem 0.9rem;
        border-radius: 18px;
        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(255,255,255,0.07);
    }

    .wm-metric-value {
        font-size: 1.18rem;
        font-weight: 900;
        color: #ffffff;
        letter-spacing: -0.03em;
    }

    .wm-metric-label {
        margin-top: 0.18rem;
        font-size: 0.82rem;
        color: #9fb3cb;
    }

    .wm-login-panel {
        position: relative;
        min-height: 84vh;
        padding: 2rem 1.8rem;
        border-radius: 26px;
        background:
            linear-gradient(180deg, rgba(10, 18, 34, 0.97), rgba(9, 15, 29, 0.93));
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow:
            0 22px 54px rgba(0,0,0,0.36),
            inset 0 1px 0 rgba(255,255,255,0.03);
        overflow: hidden;
    }

    .wm-login-panel::before {
        content: "";
        position: absolute;
        inset: auto -80px -90px auto;
        width: 260px;
        height: 260px;
        background: radial-gradient(circle, rgba(255,77,109,0.18), transparent 62%);
        filter: blur(24px);
        pointer-events: none;
    }

    .wm-login-kicker {
        color: #53c4ff;
        font-size: 0.82rem;
        font-weight: 800;
        letter-spacing: 0.14em;
        text-transform: uppercase;
    }

    .wm-login-title {
        margin-top: 0.45rem;
        color: #f4f8ff;
        font-size: 2.15rem;
        line-height: 1.02;
        font-weight: 900;
        letter-spacing: -0.04em;
    }

    .wm-login-copy {
        margin-top: 0.8rem;
        color: #98abc1;
        font-size: 0.97rem;
        line-height: 1.65;
        margin-bottom: 1.1rem;
    }

    .wm-status-row {
        display: flex;
        gap: 0.55rem;
        flex-wrap: wrap;
        margin-bottom: 1.2rem;
    }

    .wm-status-pill {
        padding: 0.45rem 0.72rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.08);
        color: #b7c8dc;
        font-size: 0.76rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    div[data-testid="stForm"] {
        background: transparent !important;
        border: 0 !important;
        padding: 0 !important;
    }

    div[data-testid="stTextInput"] label {
        color: #dce9f8 !important;
        font-weight: 700 !important;
        font-size: 0.92rem !important;
    }

    div[data-testid="stTextInput"] > div > div {
        background:
            linear-gradient(180deg, rgba(13,22,39,0.98), rgba(11,18,32,0.96)) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 18px !important;
        min-height: 56px !important;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
        transition: all 0.18s ease !important;
    }

    div[data-testid="stTextInput"] > div > div:focus-within {
        border: 1px solid rgba(30,167,255,0.65) !important;
        box-shadow:
            0 0 0 3px rgba(30,167,255,0.12),
            0 0 24px rgba(30,167,255,0.16) !important;
    }

    div[data-testid="stTextInput"] input {
        color: #f5f9ff !important;
        font-size: 1rem !important;
    }

    div[data-testid="stTextInput"] input::placeholder {
        color: #6f849e !important;
        opacity: 1 !important;
    }

    .stButton > button,
    div[data-testid="stFormSubmitButton"] > button {
        width: 100% !important;
        height: 56px !important;
        margin-top: 0.55rem !important;
        border: 0 !important;
        border-radius: 18px !important;
        color: #ffffff !important;
        font-weight: 900 !important;
        font-size: 1rem !important;
        letter-spacing: 0.01em !important;
        background:
            linear-gradient(90deg, #1293ff 0%, #1fb1ff 38%, #4cc9ff 62%, #ff5676 100%) !important;
        box-shadow:
            0 16px 34px rgba(18, 131, 255, 0.28),
            0 8px 24px rgba(255, 86, 118, 0.16) !important;
        transition: all 0.18s ease !important;
    }

    .stButton > button:hover,
    div[data-testid="stFormSubmitButton"] > button:hover {
        transform: translateY(-1px);
        box-shadow:
            0 20px 40px rgba(18, 131, 255, 0.34),
            0 12px 26px rgba(255, 86, 118, 0.20) !important;
    }

    .wm-login-note {
        margin-top: 1rem;
        padding: 0.92rem 1rem;
        border-radius: 18px;
        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(255,255,255,0.07);
        color: #9fb3cb;
        font-size: 0.88rem;
        line-height: 1.6;
    }

    div[data-testid="stAlert"] {
        border-radius: 18px !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
    }

    @media (max-width: 1200px) {
        .wm-title {
            font-size: 3.5rem;
        }
        .wm-metrics-row {
            grid-template-columns: 1fr;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# SHELL
# =========================================================
st.markdown(
    """
    <div class="wm-page-shell">
        <div class="wm-grid-bg"></div>
    </div>
    """,
    unsafe_allow_html=True,
)

left_col, right_col = st.columns([1.7, 0.95], gap="large")

# =========================================================
# LEFT PANEL
# =========================================================
with left_col:
    st.markdown('<div class="wm-left-panel">', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="wm-topline">● Watermelon System · Enterprise Access</div>
        """,
        unsafe_allow_html=True,
    )

    brand_a, brand_b = st.columns([0.12, 0.88], gap="small")

    with brand_a:
        if asset_exists(LOGO_PATH):
            st.markdown('<div class="wm-logo-wrap">', unsafe_allow_html=True)
            st.image(str(LOGO_PATH), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                """
                <div class="wm-logo-wrap" style="font-size:1.9rem;">🍉</div>
                """,
                unsafe_allow_html=True,
            )

    with brand_b:
        st.markdown(
            """
            <div class="wm-brand-row" style="margin-top:0;">
                <div class="wm-brand-text">Industrial Vibration Intelligence</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="wm-title">
            Control room<br>
            experience for<br>
            modern vibration teams.
        </div>
        <div class="wm-subtitle">
            Watermelon System transforma la entrada a la plataforma en una experiencia de producto real:
            rápida, premium y orientada a diagnóstico industrial serio. Más claridad visual, más presencia,
            más nivel de software desde el primer segundo.
        </div>
        <div class="wm-value-strip">
            <div class="wm-chip">Time Waveform</div>
            <div class="wm-chip">Orbit</div>
            <div class="wm-chip">FFT</div>
            <div class="wm-chip">Trends</div>
            <div class="wm-chip">Diagnostics</div>
            <div class="wm-chip">Reports HD</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="wm-preview-card">
            <div class="wm-preview-head">
                <div class="wm-preview-title">Live Monitoring Preview</div>
                <div class="wm-preview-badge">Online</div>
            </div>
            <div class="wm-chart">
                <div class="wm-chart-grid"></div>
                <div class="wm-chart-glow"></div>
                <div class="wm-line wm-line-1"></div>
                <div class="wm-line wm-line-2"></div>
            </div>
            <div class="wm-metrics-row">
                <div class="wm-metric">
                    <div class="wm-metric-value">4.8 ms</div>
                    <div class="wm-metric-label">UI response target</div>
                </div>
                <div class="wm-metric">
                    <div class="wm-metric-value">24/7</div>
                    <div class="wm-metric-label">Industrial readiness</div>
                </div>
                <div class="wm-metric">
                    <div class="wm-metric-value">HD Export</div>
                    <div class="wm-metric-label">Premium reporting layer</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# RIGHT PANEL
# =========================================================
with right_col:
    st.markdown('<div class="wm-login-panel">', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="wm-login-kicker">Secure Access</div>
        <div class="wm-login-title">Ingresar al sistema</div>
        <div class="wm-login-copy">
            Accede con tus credenciales corporativas y entra al entorno premium de monitoreo y análisis industrial.
        </div>
        <div class="wm-status-row">
            <div class="wm-status-pill">Enterprise UI</div>
            <div class="wm-status-pill">Secure Session</div>
            <div class="wm-status-pill">Industrial Ready</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("wm_login_form", clear_on_submit=False):
        username = st.text_input(
            "Usuario o correo",
            placeholder="usuario o correo corporativo",
            key="wm_login_username",
        )

        password = st.text_input(
            "Clave",
            placeholder="Ingresa tu contraseña",
            type="password",
            key="wm_login_password",
        )

        submit = st.form_submit_button("Ingresar", use_container_width=True)

    if submit:
        ok, msg = wm_login(username.strip(), password)
        if ok:
            st.success(msg)
            st.switch_page("00_Home.py")
        else:
            st.error(msg)

    st.markdown(
        """
        <div class="wm-login-note">
            Plataforma premium de análisis de vibraciones, visualización avanzada y diagnóstico industrial moderno.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)