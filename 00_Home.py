from __future__ import annotations

import streamlit as st

from core.auth import render_user_menu, require_login
from core.ui.theme import apply_watermelon_theme
from core.version import VERSION


st.set_page_config(
    page_title="Watermelon System | Home",
    page_icon="🍉",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.set_option("client.toolbarMode", "minimal")
st.set_option("client.showSidebarNavigation", False)

require_login()
render_user_menu()
apply_watermelon_theme()


ACTIVE_MODULES = [
    {
        "icon": "📥",
        "kicker": "Input",
        "title": "Load Data",
        "subtitle": "Cargue y prepare datasets para análisis.",
        "status": "LIVE",
        "button": "Open Load Data",
        "page": "pages/01_Load_Data.py",
    },
    {
        "icon": "🌊",
        "kicker": "Time Domain",
        "title": "Time Waveforms",
        "subtitle": "Inspección rápida de señales en el dominio del tiempo.",
        "status": "LIVE",
        "button": "Open Time Waveforms",
        "page": "pages/02_Time_Waveforms.py",
    },
    {
        "icon": "📈",
        "kicker": "Frequency",
        "title": "Spectrum",
        "subtitle": "FFT clara y veloz para diagnóstico base.",
        "status": "LIVE",
        "button": "Open Spectrum",
        "page": "pages/03_Spectrum.py",
    },
    {
        "icon": "📊",
        "kicker": "Monitoring",
        "title": "Trends",
        "subtitle": "Comparación histórica de variables y comportamiento.",
        "status": "LIVE",
        "button": "Open Trends",
        "page": "pages/04_Trends.py",
    },
    {
        "icon": "🌀",
        "kicker": "Rotor",
        "title": "Orbit Analysis",
        "subtitle": "Movimiento orbital y análisis visual del eje.",
        "status": "LIVE",
        "button": "Open Orbit Analysis",
        "page": "pages/05_Orbit_Analysis.py",
    },
    {
        "icon": "🩺",
        "kicker": "Engineering",
        "title": "Diagnostics",
        "subtitle": "Interpretación técnica y soporte de decisión.",
        "status": "LIVE",
        "button": "Open Diagnostics",
        "page": "pages/15_Diagnostics.py",
    },
]


def apply_home_style() -> None:
    st.markdown(
        """
        <style>
        #MainMenu {visibility: hidden !important;}
        header {visibility: hidden !important;}
        footer {visibility: hidden !important;}

        [data-testid="stToolbar"] {display: none !important;}
        [data-testid="stDecoration"] {display: none !important;}
        [data-testid="stStatusWidget"] {display: none !important;}
        [data-testid="stHeaderActionElements"] {display: none !important;}
        [data-testid="stAppDeployButton"] {display: none !important;}

        .stAppHeader {
            background: transparent !important;
        }

        .block-container {
            max-width: 1520px !important;
            padding-top: 1.4rem !important;
            padding-bottom: 2.0rem !important;
        }

        .wm-topbar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 14px;
            flex-wrap: wrap;
            margin-bottom: 1rem;
            padding: 14px 18px;
            border-radius: 22px;
            background: linear-gradient(180deg, rgba(255,255,255,0.98) 0%, rgba(247,250,252,0.96) 100%);
            border: 1px solid #d9e2ec;
            box-shadow:
                0 14px 34px rgba(15, 23, 42, 0.05),
                inset 0 1px 0 rgba(255,255,255,0.86);
        }

        .wm-topbar-left {
            display: flex;
            align-items: center;
            gap: 10px;
            flex-wrap: wrap;
        }

        .wm-pill {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 14px;
            border-radius: 999px;
            border: 1px solid #d7e6ff;
            background: linear-gradient(180deg, #f8fbff 0%, #eef6ff 100%);
            color: #1d4ed8;
            font-size: 0.84rem;
            font-weight: 900;
            letter-spacing: 0.02em;
        }

        .wm-pill-green {
            border: 1px solid #ccebd7;
            background: linear-gradient(180deg, #f3fcf6 0%, #e9f9ee 100%);
            color: #15803d;
        }

        .wm-topbar-text {
            color: #607083;
            font-size: 0.94rem;
            font-weight: 800;
        }

        .wm-hero {
            position: relative;
            overflow: hidden;
            background:
                radial-gradient(circle at top right, rgba(37,99,235,0.11) 0%, rgba(37,99,235,0.03) 26%, transparent 54%),
                linear-gradient(135deg, rgba(255,255,255,0.98) 0%, rgba(247,250,252,0.97) 50%, rgba(241,245,249,0.99) 100%);
            border: 1px solid #d9e2ec;
            border-radius: 30px;
            padding: 30px 30px 26px 30px;
            margin-bottom: 1.15rem;
            box-shadow:
                0 20px 50px rgba(15, 23, 42, 0.06),
                inset 0 1px 0 rgba(255,255,255,0.84);
        }

        .wm-hero::before {
            content: "";
            position: absolute;
            inset: 0 auto 0 0;
            width: 7px;
            background: linear-gradient(180deg, #2563eb 0%, #38bdf8 100%);
            border-radius: 30px 0 0 30px;
        }

        .wm-hero-kicker {
            font-size: 0.84rem;
            font-weight: 900;
            letter-spacing: 0.10em;
            text-transform: uppercase;
            color: #2563eb;
            margin-bottom: 10px;
        }

        .wm-hero-title {
            font-size: clamp(3rem, 5vw, 4.8rem);
            line-height: 0.94;
            font-weight: 950;
            letter-spacing: -0.05em;
            color: #1f2937;
            margin: 0 0 0.7rem 0;
        }

        .wm-hero-subtitle {
            font-size: 1.02rem;
            color: #5d6d80;
            font-weight: 700;
            margin: 0;
            max-width: 840px;
        }

        .wm-section-title {
            font-size: 1.04rem;
            font-weight: 900;
            color: #1f2937;
            margin: 0.15rem 0 0.9rem 0;
            letter-spacing: -0.02em;
        }

        .wm-module-card {
            background: linear-gradient(180deg, rgba(255,255,255,0.98) 0%, rgba(248,250,252,0.95) 100%);
            border: 1px solid #dbe3ec;
            border-radius: 26px;
            padding: 20px 20px 18px 20px;
            min-height: 182px;
            box-shadow:
                0 14px 34px rgba(15,23,42,0.04),
                inset 0 1px 0 rgba(255,255,255,0.84);
        }

        .wm-module-top {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            margin-bottom: 14px;
        }

        .wm-module-icon {
            width: 46px;
            height: 46px;
            border-radius: 16px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.35rem;
            background: linear-gradient(180deg, #f8fbff 0%, #eef6ff 100%);
            border: 1px solid #d7e6ff;
        }

        .wm-status-badge {
            display: inline-flex;
            align-items: center;
            padding: 6px 10px;
            border-radius: 999px;
            border: 1px solid #ccebd7;
            background: linear-gradient(180deg, #f3fcf6 0%, #e9f9ee 100%);
            color: #15803d;
            font-size: 0.72rem;
            font-weight: 900;
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }

        .wm-module-kicker {
            font-size: 0.76rem;
            font-weight: 900;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            color: #2563eb;
            margin-bottom: 8px;
        }

        .wm-module-title {
            font-size: 1.16rem;
            font-weight: 900;
            color: #1f2937;
            letter-spacing: -0.02em;
            margin-bottom: 10px;
        }

        .wm-module-subtitle {
            font-size: 0.92rem;
            color: #607083;
            font-weight: 700;
            line-height: 1.4;
            margin-bottom: 14px;
            min-height: 2.6rem;
        }

        .wm-module-divider {
            height: 1px;
            width: 100%;
            background: linear-gradient(90deg, rgba(59,130,246,0.14) 0%, rgba(203,213,225,0.42) 45%, rgba(203,213,225,0.08) 100%);
            border-radius: 999px;
            margin: 0 0 14px 0;
        }

        div[data-testid="stButton"] > button {
            border-radius: 18px !important;
            font-weight: 800 !important;
            min-height: 3.1rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_topbar() -> None:
    user = st.session_state.get("auth_username", "internal")
    st.markdown(
        f"""
        <div class="wm-topbar">
            <div class="wm-topbar-left">
                <div class="wm-pill">🍉 {VERSION}</div>
                <div class="wm-pill wm-pill-green">● System Online</div>
            </div>
            <div class="wm-topbar-text">Usuario activo: {user}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        """
        <div class="wm-hero">
            <div class="wm-hero-kicker">Watermelon System</div>
            <div class="wm-hero-title">Industrial Vibration Intelligence</div>
            <div class="wm-hero-subtitle">
                Plataforma interna para análisis de vibraciones con una experiencia más limpia,
                rápida y moderna que las suites tradicionales.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_module_card(module: dict[str, str], button_key: str) -> None:
    st.markdown(
        f"""
        <div class="wm-module-card">
            <div class="wm-module-top">
                <div class="wm-module-icon">{module["icon"]}</div>
                <div class="wm-status-badge">{module["status"]}</div>
            </div>
            <div class="wm-module-kicker">{module["kicker"]}</div>
            <div class="wm-module-title">{module["title"]}</div>
            <div class="wm-module-subtitle">{module["subtitle"]}</div>
            <div class="wm-module-divider"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button(module["button"], use_container_width=True, key=button_key):
        st.switch_page(module["page"])


apply_home_style()
render_topbar()
render_hero()

st.markdown('<div class="wm-section-title">Active Modules</div>', unsafe_allow_html=True)

row1 = st.columns(3, gap="large")
with row1[0]:
    render_module_card(ACTIVE_MODULES[0], "home_open_load_data")
with row1[1]:
    render_module_card(ACTIVE_MODULES[1], "home_open_time_waveforms")
with row1[2]:
    render_module_card(ACTIVE_MODULES[2], "home_open_spectrum")

row2 = st.columns(3, gap="large")
with row2[0]:
    render_module_card(ACTIVE_MODULES[3], "home_open_trends")
with row2[1]:
    render_module_card(ACTIVE_MODULES[4], "home_open_orbit_analysis")
with row2[2]:
    render_module_card(ACTIVE_MODULES[5], "home_open_diagnostics")