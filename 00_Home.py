from __future__ import annotations

import streamlit as st

from core.ui.theme import apply_watermelon_theme
from core.version import VERSION


st.set_page_config(
    page_title="Watermelon System | Home",
    page_icon="🍉",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Refuerzo en runtime para Community Cloud
st.set_option("client.toolbarMode", "minimal")
st.set_option("client.showSidebarNavigation", False)

# DEBUG TEMPORAL: auth desactivada para validar entrypoint y navegación
# from core.auth import render_user_menu, require_login
# require_login()
# render_user_menu()
apply_watermelon_theme()


MODULES = [
    {
        "kicker": "Input",
        "title": "Load Data",
        "stats": ["CSV", "Fast", "Ready"],
        "button": "Open Load Data",
        "page": "pages/01_Load_Data.py",
    },
    {
        "kicker": "Time",
        "title": "Time Waveforms",
        "stats": ["Raw", "1X", "HD"],
        "button": "Open Time Waveforms",
        "page": "pages/02_Time_Waveforms.py",
    },
    {
        "kicker": "Frequency",
        "title": "Spectrum",
        "stats": ["FFT", "Fast", "Clear"],
        "button": "Open Spectrum",
        "page": "pages/03_Spectrum.py",
    },
    {
        "kicker": "Monitoring",
        "title": "Trends",
        "stats": ["Trend", "Compare", "Inspect"],
        "button": "Open Trends",
        "page": "pages/04_Trends.py",
    },
    {
        "kicker": "Rotor",
        "title": "Orbit Analysis",
        "stats": ["Orbit", "Phase", "Motion"],
        "button": "Open Orbit Analysis",
        "page": "pages/05_Orbit_Analysis.py",
    },
    {
        "kicker": "Engineering",
        "title": "Diagnostics",
        "stats": ["Faults", "Evidence", "Action"],
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
        [data-testid="collapsedControl"] {display: none !important;}

        .stAppHeader {
            background: transparent !important;
        }

        .block-container {
            padding-top: 1.3rem !important;
            padding-bottom: 2rem !important;
            max-width: 1480px !important;
        }

        .wm-version-bar {
            display: flex;
            align-items: center;
            justify-content: flex-start;
            gap: 12px;
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

        .wm-version-pill {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 14px;
            border-radius: 999px;
            border: 1px solid #d7e6ff;
            background: linear-gradient(180deg, #f8fbff 0%, #eef6ff 100%);
            color: #1d4ed8;
            font-size: 0.86rem;
            font-weight: 900;
            letter-spacing: 0.02em;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.9);
        }

        .wm-version-text {
            color: #607083;
            font-size: 0.94rem;
            font-weight: 800;
        }

        .wm-hero {
            position: relative;
            overflow: hidden;
            background:
                radial-gradient(circle at top right, rgba(37,99,235,0.10) 0%, rgba(37,99,235,0.025) 26%, transparent 52%),
                linear-gradient(135deg, rgba(255,255,255,0.98) 0%, rgba(247,250,252,0.96) 48%, rgba(241,245,249,0.98) 100%);
            border: 1px solid #d9e2ec;
            border-radius: 30px;
            padding: 28px 28px 24px 28px;
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
            font-size: 0.86rem;
            font-weight: 900;
            letter-spacing: 0.10em;
            text-transform: uppercase;
            color: #2563eb;
            margin-bottom: 10px;
        }

        .wm-hero-title {
            font-size: clamp(2.8rem, 5vw, 4.6rem);
            line-height: 0.94;
            font-weight: 950;
            letter-spacing: -0.05em;
            color: #1f2937;
            margin: 0 0 0.55rem 0;
        }

        .wm-hero-subtitle {
            font-size: 1rem;
            color: #5d6d80;
            font-weight: 700;
            margin: 0;
        }

        .wm-section-title {
            font-size: 1.02rem;
            font-weight: 900;
            color: #1f2937;
            margin: 0.2rem 0 0.8rem 0;
            letter-spacing: -0.02em;
        }

        .wm-nav-card {
            background: linear-gradient(180deg, rgba(255,255,255,0.98) 0%, rgba(248,250,252,0.95) 100%);
            border: 1px solid #dbe3ec;
            border-radius: 26px;
            padding: 20px 20px 18px 20px;
            min-height: 140px;
            box-shadow:
                0 14px 34px rgba(15,23,42,0.04),
                inset 0 1px 0 rgba(255,255,255,0.84);
        }

        .wm-nav-card-kicker {
            display: inline-flex;
            align-items: center;
            padding: 6px 10px;
            border-radius: 999px;
            border: 1px solid #d7e6ff;
            background: linear-gradient(180deg, #f8fbff 0%, #eef6ff 100%);
            color: #2563eb;
            font-size: 0.74rem;
            font-weight: 900;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            margin-bottom: 12px;
        }

        .wm-nav-card-title {
            font-size: 1.12rem;
            font-weight: 900;
            color: #1f2937;
            margin-bottom: 14px;
            letter-spacing: -0.02em;
        }

        .wm-nav-divider {
            height: 1px;
            width: 100%;
            background: linear-gradient(90deg, rgba(59,130,246,0.14) 0%, rgba(203,213,225,0.42) 45%, rgba(203,213,225,0.08) 100%);
            border-radius: 999px;
            margin: 0 0 14px 0;
        }

        .wm-mini-stat-wrap {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }

        .wm-mini-stat {
            display: inline-flex;
            align-items: center;
            padding: 6px 10px;
            border-radius: 999px;
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            color: #475569;
            font-size: 0.79rem;
            font-weight: 800;
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


def render_version_bar() -> None:
    st.markdown(
        f"""
        <div class="wm-version-bar">
            <div class="wm-version-pill">🍉 {VERSION}</div>
            <div class="wm-version-text">Private internal demo</div>
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
            <div class="wm-hero-subtitle">Clean. Fast. Modern.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_module_card(module: dict[str, object], button_key: str) -> None:
    stats_html = "".join(
        f'<span class="wm-mini-stat">{item}</span>'
        for item in module["stats"]  # type: ignore[index]
    )

    st.markdown(
        f"""
        <div class="wm-nav-card">
            <div class="wm-nav-card-kicker">{module["kicker"]}</div>
            <div class="wm-nav-card-title">{module["title"]}</div>
            <div class="wm-nav-divider"></div>
            <div class="wm-mini-stat-wrap">
                {stats_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button(module["button"], use_container_width=True, key=button_key):  # type: ignore[index]
        st.switch_page(module["page"])  # type: ignore[index]


apply_home_style()
render_version_bar()
render_hero()

st.markdown('<div class="wm-section-title">Modules</div>', unsafe_allow_html=True)

row_1_col_1, row_1_col_2 = st.columns(2, gap="large")
with row_1_col_1:
    render_module_card(MODULES[0], "home_open_load_data")
with row_1_col_2:
    render_module_card(MODULES[1], "home_open_time_waveforms")

row_2_col_1, row_2_col_2 = st.columns(2, gap="large")
with row_2_col_1:
    render_module_card(MODULES[2], "home_open_spectrum")
with row_2_col_2:
    render_module_card(MODULES[3], "home_open_trends")

row_3_col_1, row_3_col_2 = st.columns(2, gap="large")
with row_3_col_1:
    render_module_card(MODULES[4], "home_open_orbit_analysis")
with row_3_col_2:
    render_module_card(MODULES[5], "home_open_diagnostics")