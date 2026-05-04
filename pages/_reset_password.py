"""
pages/_reset_password.py
========================

Landing del email link de reset de password (Ciclo 17.16).

URL: https://wm-home-final-2026.streamlit.app/reset_password?token=xxx

Flujo:
  1. Usuario hace click en el link del email
  2. Esta página lee ?token=xxx desde st.query_params
  3. Valida el token vía core.password_reset.validate_token
  4. Si OK, muestra form: nueva password + confirmar
  5. Si el form es válido (>=8 chars, las dos coinciden), llama
     consume_token → cambia la pwd en Supabase Auth + invalida token
  6. Mensaje verde con email + botón "Ir a login"

Si el token está expirado/inválido/consumido, mostrar error claro
con opción de volver a "olvidé mi contraseña" desde el Login.
"""

from __future__ import annotations

import streamlit as st


st.set_page_config(
    page_title="Watermelon System | Restablecer contraseña",
    page_icon="🍉",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# =============================================================
# CSS / esconder navegación
# =============================================================
st.markdown(
    """
    <style>
    header, #MainMenu, footer {visibility: hidden;}
    [data-testid="stSidebar"] {display: none !important;}
    [data-testid="stSidebarNav"] {display: none !important;}
    .stApp {
        background:
            radial-gradient(circle at 78% 0%, rgba(33,71,140,0.10) 0%, transparent 32%),
            radial-gradient(circle at 0% 100%, rgba(33,71,140,0.06) 0%, transparent 28%),
            linear-gradient(180deg, #f3f6fb 0%, #e9eef6 100%);
    }
    .block-container {
        max-width: 540px !important;
        padding-top: 4rem !important;
    }
    .wmr-card {
        background: rgba(255,255,255,0.95);
        border: 1px solid rgba(33,71,140,0.10);
        border-radius: 22px;
        padding: 2.4rem 2rem;
        box-shadow: 0 22px 50px rgba(15,23,42,0.10);
    }
    .wmr-pill {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 999px;
        background: rgba(33,71,140,0.08);
        border: 1px solid rgba(33,71,140,0.18);
        color: #21478c;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        margin-bottom: 1rem;
    }
    .wmr-title {
        font-size: 24px;
        font-weight: 800;
        color: #0f172a;
        margin: 0 0 6px 0;
    }
    .wmr-sub {
        color: #475569;
        font-size: 14px;
        line-height: 1.5;
        margin-bottom: 1.4rem;
    }
    div[data-testid="stFormSubmitButton"] > button {
        width: 100% !important;
        height: 50px !important;
        border-radius: 12px !important;
        font-size: 1rem !important;
        font-weight: 700 !important;
        color: #ffffff !important;
        background: linear-gradient(135deg, #21478c 0%, #2a6dd1 100%) !important;
        border: 0 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================
# Leer token desde URL
# =============================================================
def _get_query_token() -> str:
    try:
        qp = st.query_params
        v = qp.get("token", "")
        if isinstance(v, list):
            return v[0] if v else ""
        return str(v or "")
    except Exception:
        # Compat con versiones viejas de Streamlit
        try:
            qp = st.experimental_get_query_params()
            return qp.get("token", [""])[0]
        except Exception:
            return ""


_token = _get_query_token().strip()


# =============================================================
# UI
# =============================================================
st.markdown('<div class="wmr-card">', unsafe_allow_html=True)
st.markdown('<span class="wmr-pill">🔐 Restablecer contraseña</span>',
            unsafe_allow_html=True)
st.markdown('<div class="wmr-title">Elegí una nueva contraseña</div>',
            unsafe_allow_html=True)

if not _token:
    st.markdown(
        '<div class="wmr-sub">Este link no incluye un token válido.</div>',
        unsafe_allow_html=True,
    )
    st.error(
        "🚫 No detectamos un token de reset en la URL. "
        "Por favor pedí un nuevo link desde la página de Login → "
        "'¿Olvidaste tu contraseña?'"
    )
    if st.button("Volver al Login", use_container_width=True):
        st.switch_page("pages/00_Login.py")
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# Validar token
try:
    from core.password_reset import validate_token, consume_token
except Exception as e:
    st.error(f"Error de configuración: no se pudo cargar password_reset ({e})")
    st.stop()

_val = validate_token(_token)
if not _val.get("valid"):
    st.markdown(
        f'<div class="wmr-sub">Token: <code>{_token[:24]}…</code></div>',
        unsafe_allow_html=True,
    )
    st.error(
        f"🚫 **{_val.get('error', 'Token inválido.')}**\n\n"
        "Si necesitás un nuevo link, pedilo desde la página de Login → "
        "'¿Olvidaste tu contraseña?'"
    )
    if st.button("Volver al Login", use_container_width=True):
        st.switch_page("pages/00_Login.py")
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# Token válido — mostrar form
_email = _val.get("email", "")
_full_name = _val.get("full_name", "")
_expires = _val.get("expires_at", "")[:16].replace("T", " ")

st.markdown(
    f'<div class="wmr-sub">'
    f"Hola <b>{_full_name or _email}</b>,<br/>"
    f"Estás a punto de restablecer la contraseña de la cuenta "
    f"<code>{_email}</code>.<br/>"
    f"<small style='color:#94a3b8;'>Link válido hasta: {_expires}</small>"
    "</div>",
    unsafe_allow_html=True,
)

# Si ya cambió OK, mostrar pantalla de éxito en vez del form
if st.session_state.get("wm_reset_success"):
    st.success(
        f"✅ **Tu contraseña se actualizó correctamente.**\n\n"
        f"Ya podés iniciar sesión con la nueva clave en `{_email}`."
    )
    if st.button("Ir al Login ahora", use_container_width=True, type="primary"):
        st.switch_page("pages/00_Login.py")
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

with st.form("wm_reset_form", clear_on_submit=False):
    new_pwd = st.text_input(
        "Nueva contraseña",
        type="password",
        placeholder="Mínimo 8 caracteres",
        key="wm_reset_new_pwd",
        autocomplete="new-password",
    )
    confirm_pwd = st.text_input(
        "Confirmar nueva contraseña",
        type="password",
        placeholder="Volvé a tipearla",
        key="wm_reset_confirm_pwd",
        autocomplete="new-password",
    )
    st.caption(
        "💡 Recomendación: usá una password de al menos 12 caracteres con "
        "mayúsculas, minúsculas y números. Considerá usar un password "
        "manager (1Password, Bitwarden, Apple Passwords)."
    )
    submit = st.form_submit_button("Cambiar contraseña", use_container_width=True)

if submit:
    if not new_pwd or len(new_pwd) < 8:
        st.error("La password debe tener al menos 8 caracteres.")
    elif new_pwd != confirm_pwd:
        st.error("Las dos passwords no coinciden.")
    else:
        try:
            res = consume_token(_token, new_pwd)
            if res.get("ok"):
                st.session_state["wm_reset_success"] = True
                st.rerun()
            else:
                st.error(f"No se pudo cambiar la password: {res.get('error')}")
        except Exception as e:
            st.error(f"Error inesperado: {e}")

st.markdown('</div>', unsafe_allow_html=True)
