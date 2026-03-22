from __future__ import annotations

import streamlit as st

from core.auth import require_login, render_user_menu

st.set_page_config(
    page_title="Watermelon System | Home",
    page_icon="🍉",
    layout="wide",
    initial_sidebar_state="expanded",
)

require_login()
render_user_menu()

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1450px !important;
        padding-top: 1.2rem !important;
        padding-bottom: 2rem !important;
        padding-left: 1.2rem !important;
        padding-right: 1.2rem !important;
    }

    .wm-hero {
        border-radius: 24px;
        padding: 1.5rem 1.6rem;
        background: linear-gradient(135deg, #0f172a 0%, #13213b 100%);
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 18px 40px rgba(15,23,42,0.18);
        margin-bottom: 1rem;
    }

    .wm-kicker {
        color: #66c2ff;
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }

    .wm-title {
        color: white;
        font-size: 2.2rem;
        font-weight: 900;
        letter-spacing: -0.04em;
        margin-top: 0.25rem;
    }

    .wm-copy {
        color: #b9c6d8;
        font-size: 0.98rem;
        margin-top: 0.4rem;
    }

    .wm-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 1rem;
        margin-top: 1rem;
    }

    .wm-card {
        background: white;
        border: 1px solid #e5edf5;
        border-radius: 22px;
        padding: 1rem 1.1rem;
        box-shadow: 0 10px 24px rgba(15,23,42,0.05);
    }

    .wm-card-title {
        color: #0f172a;
        font-size: 1.05rem;
        font-weight: 800;
        margin-bottom: 0.35rem;
    }

    .wm-card-copy {
        color: #64748b;
        font-size: 0.92rem;
        line-height: 1.6;
    }

    @media (max-width: 1200px) {
        .wm-grid {
            grid-template-columns: 1fr;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="wm-hero">
        <div class="wm-kicker">Watermelon System</div>
        <div class="wm-title">Industrial Vibration Intelligence</div>
        <div class="wm-copy">
            Plataforma central para análisis, monitoreo, visualización, diagnóstico y construcción de reportes.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        <div class="wm-card">
            <div class="wm-card-title">Análisis</div>
            <div class="wm-card-copy">
                Accede a módulos de formas de onda, espectro, órbita, fase, Bode, waterfall, centerline y TSA.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
        <div class="wm-card">
            <div class="wm-card-title">Diagnóstico</div>
            <div class="wm-card-copy">
                Centraliza señales, tendencias y visualizaciones para soporte técnico y análisis de condición.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        """
        <div class="wm-card">
            <div class="wm-card-title">Reportes</div>
            <div class="wm-card-copy">
                Construye reportes técnicos con bloques seleccionables para convertir análisis en entregables.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()

st.subheader("Accesos rápidos")

q1, q2, q3, q4 = st.columns(4)

with q1:
    if st.button("Load Data", use_container_width=True):
        st.switch_page("pages/01_Load_Data.py")

with q2:
    if st.button("Time Waveforms", use_container_width=True):
        st.switch_page("pages/02_Time_Waveforms.py")

with q3:
    if st.button("Diagnostics", use_container_width=True):
        st.switch_page("pages/15_Diagnostics.py")

with q4:
    if st.button("Reports", use_container_width=True):
        st.switch_page("pages/16_Reports.py")