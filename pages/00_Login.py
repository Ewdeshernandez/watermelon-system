from __future__ import annotations

import streamlit as st

from core.auth import get_current_user, is_authenticated, login, render_login_shell


st.set_page_config(
    page_title="Watermelon System | Login",
    page_icon="🍉",
    layout="wide",
)

render_login_shell()

if is_authenticated():
    st.markdown(
        """
        <style>
        .wm-center-card {
            max-width: 760px;
            margin: 8vh auto 0 auto;
            padding: 2.2rem;
            border-radius: 26px;
            background: linear-gradient(180deg, rgba(14,20,30,0.96) 0%, rgba(10,14,24,0.98) 100%);
            border: 1px solid rgba(255,255,255,0.08);
            box-shadow: 0 24px 60px rgba(0,0,0,0.25);
        }
        .wm-ok-title {
            font-size: 2rem;
            font-weight: 800;
            color: white;
            margin-bottom: 0.45rem;
            letter-spacing: -0.03em;
        }
        .wm-ok-sub {
            color: rgba(255,255,255,0.72);
            margin-bottom: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    user = get_current_user()

    st.markdown('<div class="wm-center-card">', unsafe_allow_html=True)
    st.markdown('<div class="wm-ok-title">🍉 Acceso activo</div>', unsafe_allow_html=True)
    st.markdown('<div class="wm-ok-sub">Ya tienes una sesión activa en Watermelon System.</div>', unsafe_allow_html=True)
    if user:
        st.caption(f"Conectado como: {user.get('full_name') or user.get('username')}")

    col_a, col_b, col_c = st.columns([1, 1.2, 1])
    with col_b:
        if st.button("Entrar al demo", use_container_width=True, type="primary"):
            try:
                st.switch_page("pages/04_Trends.py")
            except Exception:
                st.info("Navega al módulo desde el menú lateral.")
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

st.markdown(
    """
    <style>
    .wm-login-wrap {
        max-width: 560px;
        margin: 7vh auto 0 auto;
        padding: 2.4rem 2.2rem 1.7rem 2.2rem;
        border-radius: 26px;
        background: linear-gradient(180deg, rgba(16,24,40,0.95) 0%, rgba(9,14,24,0.98) 100%);
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 24px 60px rgba(0,0,0,0.30);
    }
    .wm-login-title {
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.04em;
        margin-bottom: 0.35rem;
        color: white;
    }
    .wm-login-sub {
        color: rgba(255,255,255,0.72);
        margin-bottom: 1.4rem;
        line-height: 1.45;
    }
    .wm-login-foot {
        color: rgba(255,255,255,0.5);
        text-align: center;
        margin-top: 1rem;
        font-size: 0.92rem;
    }
    div[data-testid="stForm"] {
        border: none !important;
        padding: 0 !important;
        background: transparent !important;
        box-shadow: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="wm-login-wrap">', unsafe_allow_html=True)
st.markdown('<div class="wm-login-title">🍉 Watermelon System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="wm-login-sub">Demo privado. Ingresa con usuario o correo autorizado para acceder al sistema.</div>',
    unsafe_allow_html=True,
)

with st.form("wm_login_form", clear_on_submit=False):
    identifier = st.text_input(
        "Usuario o correo",
        placeholder="ej: admin o admin@watermelon.com",
    )
    password = st.text_input(
        "Clave",
        type="password",
        placeholder="Ingresa tu clave",
    )
    submitted = st.form_submit_button("Ingresar", use_container_width=True, type="primary")

if submitted:
    ok, message = login(identifier=identifier, password=password)
    if ok:
        st.success(message)
        try:
            st.switch_page("pages/04_Trends.py")
        except Exception:
            st.rerun()
    else:
        st.error(message)

st.markdown(
    '<div class="wm-login-foot">Private demo · Watermelon System · Restricted access</div>',
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)