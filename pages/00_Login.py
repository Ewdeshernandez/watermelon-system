from __future__ import annotations

import streamlit as st

from core.auth import get_current_user, is_authenticated, login, render_login_shell


st.set_page_config(
    page_title="Watermelon System | Login",
    page_icon="🍉",
    layout="wide",
    initial_sidebar_state="collapsed",
)

render_login_shell()

# Si ya hay sesión, SIEMPRE mandar a Home, no a Trends
if is_authenticated():
    st.switch_page("00_Home.py")


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
    [data-testid="collapsedControl"] {display: none !important;}

    .wm-login-wrap {
        max-width: 520px;
        margin: 3rem auto 0 auto;
        padding: 2rem;
        border-radius: 24px;
        border: 1px solid #d9e2ec;
        background: linear-gradient(180deg, rgba(255,255,255,0.98) 0%, rgba(247,250,252,0.96) 100%);
        box-shadow:
            0 20px 50px rgba(15, 23, 42, 0.06),
            inset 0 1px 0 rgba(255,255,255,0.84);
    }

    .wm-login-kicker {
        font-size: 0.85rem;
        font-weight: 900;
        letter-spacing: 0.10em;
        text-transform: uppercase;
        color: #2563eb;
        margin-bottom: 0.5rem;
    }

    .wm-login-title {
        font-size: 2.5rem;
        line-height: 0.95;
        font-weight: 950;
        letter-spacing: -0.04em;
        color: #1f2937;
        margin-bottom: 0.6rem;
    }

    .wm-login-subtitle {
        color: #5d6d80;
        font-size: 1rem;
        font-weight: 700;
        margin-bottom: 1.4rem;
    }
    </style>
    <div class="wm-login-wrap">
        <div class="wm-login-kicker">Watermelon System</div>
        <div class="wm-login-title">Industrial Vibration Intelligence</div>
        <div class="wm-login-subtitle">Acceso interno</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# formulario
with st.container():
    col_l, col_c, col_r = st.columns([1, 1.35, 1])

    with col_c:
        with st.form("wm_login_form", clear_on_submit=False):
            identifier = st.text_input("Usuario o correo", placeholder="admin o demo")
            password = st.text_input("Clave", type="password", placeholder="••••••••")
            submitted = st.form_submit_button("Ingresar", use_container_width=True)

        if submitted:
            ok, msg = login(identifier, password)
            if ok:
                st.success(msg)
                st.switch_page("00_Home.py")
            else:
                st.error(msg)

        user = get_current_user()
        if user:
            st.info(f"Sesión activa: {user.get('username', '-')}")