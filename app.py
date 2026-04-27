from __future__ import annotations

import streamlit as st

from core.auth import render_user_menu, require_login
from core.ui.theme import apply_theme


# =============================================================
# Watermelon System — Main entry point
# =============================================================
# app.py es el script principal de Streamlit. Tras autenticación
# renderiza el Home directamente. El Home antes vivía en
# 00_Home.py pero st.switch_page de Streamlit moderno solo acepta
# el main file y archivos en pages/, por eso se inlinea aquí.
#
# Si no hay sesión activa, require_login() redirige a la página
# de login (pages/00_Login.py).
# =============================================================

st.set_page_config(
    page_title="Watermelon System",
    page_icon="🍉",
    layout="wide",
)

require_login()
render_user_menu()
apply_theme()


# -------------------------------------------------------------
# HOME LAYOUT
# -------------------------------------------------------------
st.markdown(
    """
    <style>
    .block-container {
        max-width: 1550px !important;
        padding-top: 1.4rem !important;
        padding-bottom: 2.2rem !important;
    }

    /* HERO */
    .wm-hero {
        border-radius: 34px;
        padding: 70px 64px;
        margin-bottom: 40px;
        background: linear-gradient(135deg, #08111f 0%, #0c1628 40%, #101c31 70%, #162742 100%);
        border: 1px solid rgba(148, 163, 184, 0.14);
        box-shadow: 0 24px 70px rgba(15, 23, 42, 0.20);
    }

    .wm-chip {
        display: inline-block;
        padding: 10px 16px;
        border-radius: 999px;
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.12);
        color: rgba(255,255,255,0.88);
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        margin-bottom: 22px;
    }

    .wm-title {
        color: #f8fbff;
        font-size: 80px;
        line-height: 0.92;
        font-weight: 800;
        letter-spacing: -0.06em;
        margin: 0;
    }

    .wm-subtitle {
        margin-top: 20px;
        color: rgba(226, 232, 240, 0.75);
        font-size: 18px;
        max-width: 700px;
    }

    /* IMAGE SECTION */
    .wm-image {
        border-radius: 30px;
        overflow: hidden;
        border: 1px solid #dbe5f0;
        box-shadow: 0 18px 50px rgba(15, 23, 42, 0.12);
    }

    .wm-image img {
        width: 100%;
        display: block;
    }

    @media (max-width: 1000px) {
        .wm-title {
            font-size: 50px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# HERO
st.markdown(
    """
    <div class="wm-hero">
        <div class="wm-chip">Watermelon System</div>
        <h1 class="wm-title">Industrial Vibration Intelligence</h1>
        <div class="wm-subtitle">
            Plataforma moderna para análisis, monitoreo y diagnóstico de vibraciones industriales.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# IMAGEN PROTAGONISTA
st.markdown('<div class="wm-image">', unsafe_allow_html=True)
st.image("assets/turbina.png", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)
