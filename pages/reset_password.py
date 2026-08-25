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
    page_title="Watermelon System | Reset password",
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
st.markdown('<span class="wmr-pill">🔐 Reset password</span>',
            unsafe_allow_html=True)
st.markdown('<div class="wmr-title">Choose a new password</div>',
            unsafe_allow_html=True)

if not _token:
    st.markdown(
        '<div class="wmr-sub">This link does not include a valid token.</div>',
        unsafe_allow_html=True,
    )
    st.error(
        "🚫 We could not detect a reset token in the URL. "
        "Please request a new link from the Login page → "
        "'Forgot your password?'"
    )
    if st.button("Back to Login", use_container_width=True):
        st.switch_page("pages/00_Login.py")
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# Validar token
try:
    from core.password_reset import validate_token, consume_token
except Exception as e:
    st.error(f"Configuration error: could not load password_reset ({e})")
    st.stop()

_val = validate_token(_token)
if not _val.get("valid"):
    st.markdown(
        f'<div class="wmr-sub">Token: <code>{_token[:24]}…</code></div>',
        unsafe_allow_html=True,
    )
    st.error(
        f"🚫 **{_val.get('error', 'Invalid token.')}**\n\n"
        "If you need a new link, request it from the Login page → "
        "'Forgot your password?'"
    )
    if st.button("Back to Login", use_container_width=True):
        st.switch_page("pages/00_Login.py")
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# Token válido — mostrar form
_email = _val.get("email", "")
_full_name = _val.get("full_name", "")
_expires = _val.get("expires_at", "")[:16].replace("T", " ")

st.markdown(
    f'<div class="wmr-sub">'
    f"Hello <b>{_full_name or _email}</b>,<br/>"
    f"You are about to reset the password for the account "
    f"<code>{_email}</code>.<br/>"
    f"<small style='color:#94a3b8;'>Link valid until: {_expires}</small>"
    "</div>",
    unsafe_allow_html=True,
)

# Si ya cambió OK, mostrar pantalla de éxito en vez del form
if st.session_state.get("wm_reset_success"):
    st.success(
        f"✅ **Your password was updated successfully.**\n\n"
        f"You can now sign in with the new password using `{_email}`."
    )
    if st.button("Go to Login now", use_container_width=True, type="primary"):
        st.switch_page("pages/00_Login.py")
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

with st.form("wm_reset_form", clear_on_submit=False):
    new_pwd = st.text_input(
        "New password",
        type="password",
        placeholder="At least 8 characters",
        key="wm_reset_new_pwd",
        autocomplete="new-password",
    )
    confirm_pwd = st.text_input(
        "Confirm new password",
        type="password",
        placeholder="Type it again",
        key="wm_reset_confirm_pwd",
        autocomplete="new-password",
    )
    st.caption(
        "💡 Recommendation: use a password of at least 12 characters with "
        "uppercase, lowercase and numbers. Consider using a password "
        "manager (1Password, Bitwarden, Apple Passwords)."
    )
    submit = st.form_submit_button("Change password", use_container_width=True)

if submit:
    if not new_pwd or len(new_pwd) < 8:
        st.error("The password must be at least 8 characters long.")
    elif new_pwd != confirm_pwd:
        st.error("The two passwords do not match.")
    else:
        try:
            res = consume_token(_token, new_pwd)
            if res.get("ok"):
                st.session_state["wm_reset_success"] = True
                st.rerun()
            else:
                st.error(f"Could not change the password: {res.get('error')}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")

st.markdown('</div>', unsafe_allow_html=True)
