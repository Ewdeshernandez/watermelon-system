import streamlit as st

st.set_page_config(
    page_title="Watermelon System | Login",
    page_icon="🍉",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =========================================================
# CSS PREMIUM LOGIN - WATERMELON SYSTEM
# =========================================================
st.markdown(
    """
    <style>
    /* ---------- Hide Streamlit chrome ---------- */
    header, #MainMenu, footer {
        visibility: hidden;
    }

    [data-testid="stSidebar"] {
        display: none;
    }

    /* ---------- Global background ---------- */
    .stApp {
        background:
            radial-gradient(circle at 12% 18%, rgba(30,167,255,0.22) 0%, transparent 28%),
            radial-gradient(circle at 88% 82%, rgba(255,45,85,0.18) 0%, transparent 30%),
            radial-gradient(circle at 70% 20%, rgba(74,222,128,0.10) 0%, transparent 22%),
            linear-gradient(135deg, #07111f 0%, #0a1426 45%, #0d172b 100%);
        color: #f3f7ff;
    }

    .block-container {
        max-width: 1380px;
        padding-top: 2.1rem;
        padding-bottom: 1.5rem;
        padding-left: 2.5rem;
        padding-right: 2.5rem;
    }

    /* ---------- Custom shells ---------- */
    .wm-shell {
        position: relative;
        min-height: 82vh;
        border-radius: 28px;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.08);
        background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02));
        box-shadow:
            0 30px 80px rgba(0,0,0,0.45),
            inset 0 1px 0 rgba(255,255,255,0.05);
    }

    .wm-shell::before {
        content: "";
        position: absolute;
        inset: 0;
        background:
            linear-gradient(135deg, rgba(255,255,255,0.03), transparent 35%),
            radial-gradient(circle at 20% 20%, rgba(30,167,255,0.12), transparent 22%),
            radial-gradient(circle at 80% 75%, rgba(255,45,85,0.12), transparent 24%);
        pointer-events: none;
    }

    .wm-left-card {
        position: relative;
        min-height: 74vh;
        padding: 3.3rem 3.1rem 3.0rem 3.1rem;
        border-radius: 24px;
        background:
            linear-gradient(180deg, rgba(11, 20, 38, 0.88), rgba(8, 15, 28, 0.76));
        border: 1px solid rgba(255,255,255,0.06);
        box-shadow:
            inset 0 1px 0 rgba(255,255,255,0.05),
            0 18px 44px rgba(0,0,0,0.28);
        overflow: hidden;
    }

    .wm-left-card::after {
        content: "";
        position: absolute;
        width: 420px;
        height: 420px;
        right: -140px;
        top: -110px;
        background: radial-gradient(circle, rgba(30,167,255,0.22) 0%, transparent 60%);
        filter: blur(18px);
        pointer-events: none;
    }

    .wm-login-card {
        position: relative;
        margin-top: 2.2rem;
        padding: 2.0rem 1.8rem 1.7rem 1.8rem;
        border-radius: 24px;
        background:
            linear-gradient(180deg, rgba(12, 19, 34, 0.96), rgba(9, 16, 30, 0.90));
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow:
            0 20px 50px rgba(0,0,0,0.42),
            inset 0 1px 0 rgba(255,255,255,0.04);
    }

    /* ---------- Typography ---------- */
    .wm-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.45rem 0.85rem;
        border-radius: 999px;
        background: rgba(30,167,255,0.10);
        border: 1px solid rgba(30,167,255,0.28);
        color: #82d8ff;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }

    .wm-eyebrow {
        margin-top: 0.6rem;
        color: #49b8ff;
        font-size: 0.92rem;
        font-weight: 700;
        letter-spacing: 0.18em;
        text-transform: uppercase;
    }

    .wm-title {
        margin-top: 1rem;
        font-size: 4rem;
        line-height: 0.97;
        font-weight: 900;
        color: #f5f9ff;
        letter-spacing: -0.04em;
    }

    .wm-subtitle {
        margin-top: 1.1rem;
        max-width: 760px;
        font-size: 1.08rem;
        line-height: 1.75;
        color: #a9b7cb;
    }

    .wm-kpis {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.95rem;
        margin-top: 2.0rem;
    }

    .wm-kpi {
        padding: 1rem 1rem 0.9rem 1rem;
        border-radius: 18px;
        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(255,255,255,0.06);
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
    }

    .wm-kpi-value {
        font-size: 1.45rem;
        font-weight: 900;
        color: #ffffff;
        letter-spacing: -0.03em;
    }

    .wm-kpi-label {
        margin-top: 0.25rem;
        font-size: 0.86rem;
        color: #9db0c8;
    }

    .wm-right-head {
        color: #eef5ff;
        font-size: 1.75rem;
        font-weight: 850;
        letter-spacing: -0.03em;
        margin-bottom: 0.35rem;
    }

    .wm-right-copy {
        color: #93a6bf;
        font-size: 0.98rem;
        line-height: 1.65;
        margin-bottom: 1.35rem;
    }

    /* ---------- Inputs ---------- */
    div[data-testid="stTextInput"] label {
        color: #dbe7f7 !important;
        font-weight: 650 !important;
        font-size: 0.92rem !important;
    }

    div[data-testid="stTextInput"] > div > div {
        background: linear-gradient(180deg, rgba(15,24,42,0.98), rgba(12,20,36,0.95)) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 16px !important;
        min-height: 54px !important;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
        transition: all 0.2s ease;
    }

    div[data-testid="stTextInput"] > div > div:focus-within {
        border: 1px solid rgba(30,167,255,0.65) !important;
        box-shadow:
            0 0 0 3px rgba(30,167,255,0.14),
            0 0 22px rgba(30,167,255,0.16) !important;
    }

    div[data-testid="stTextInput"] input {
        color: #f3f7ff !important;
        font-size: 1rem !important;
    }

    div[data-testid="stTextInput"] input::placeholder {
        color: #6f849e !important;
        opacity: 1 !important;
    }

    /* ---------- Button ---------- */
    .stButton > button {
        width: 100%;
        height: 54px;
        margin-top: 0.55rem;
        border: 0 !important;
        border-radius: 16px !important;
        color: white !important;
        font-weight: 800 !important;
        font-size: 1rem !important;
        letter-spacing: 0.01em;
        background:
            linear-gradient(90deg, #1593ff 0%, #1ea7ff 35%, #24c6ff 60%, #ff4d6d 100%) !important;
        box-shadow:
            0 14px 30px rgba(18, 131, 255, 0.24),
            0 6px 20px rgba(255, 77, 109, 0.16);
        transition: all 0.22s ease;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow:
            0 18px 34px rgba(18, 131, 255, 0.34),
            0 8px 24px rgba(255, 77, 109, 0.22);
    }

    /* ---------- Alert box ---------- */
    .wm-note {
        margin-top: 1rem;
        padding: 0.9rem 1rem;
        border-radius: 16px;
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.06);
        color: #98acc5;
        font-size: 0.92rem;
        line-height: 1.6;
    }

    /* ---------- Divider stats ---------- */
    .wm-mini-row {
        display: flex;
        gap: 0.65rem;
        flex-wrap: wrap;
        margin-top: 1.15rem;
    }

    .wm-chip {
        padding: 0.55rem 0.85rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.07);
        color: #b7c7da;
        font-size: 0.82rem;
        font-weight: 650;
    }

    /* ---------- Responsive ---------- */
    @media (max-width: 1100px) {
        .wm-title {
            font-size: 3rem;
        }
        .wm-kpis {
            grid-template-columns: 1fr;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# SESSION DEFAULTS
# =========================================================
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

# =========================================================
# AUTH FUNCTION
# REEMPLAZA SOLO EL CUERPO INTERNO SI YA TIENES AUTH REAL
# =========================================================
def authenticate_user(username: str, password: str) -> bool:
    """
    Reemplaza esta función por tu auth real si ya existe.
    Mantén la firma para no tocar el resto del archivo.
    """
    demo_users = {
        "admin": "admin",
        "demo": "demo",
    }
    return demo_users.get(username) == password


# Si ya está autenticado, no mostramos el login
if st.session_state.get("authenticated", False):
    st.success("Sesión activa.")
    st.stop()

# =========================================================
# MAIN LAYOUT
# =========================================================
st.markdown('<div class="wm-shell">', unsafe_allow_html=True)

left_col, right_col = st.columns([1.65, 0.92], gap="large")

with left_col:
    st.markdown('<div class="wm-left-card">', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="wm-badge">● Watermelon System · Premium Platform</div>
        <div class="wm-eyebrow">Industrial Vibration Intelligence</div>
        <div class="wm-title">
            Diagnóstico moderno.<br>
            Monitoreo serio.<br>
            Presencia de producto.
        </div>
        <div class="wm-subtitle">
            Plataforma SaaS industrial para análisis, monitoreo, visualización y diagnóstico de vibraciones.
            Diseñada para una experiencia superior: más clara, más rápida y mucho más premium que el software industrial tradicional.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="wm-kpis">
            <div class="wm-kpi">
                <div class="wm-kpi-value">HD Analytics</div>
                <div class="wm-kpi-label">Visualización premium para señales, órbitas, espectros y diagnóstico.</div>
            </div>
            <div class="wm-kpi">
                <div class="wm-kpi-value">Modern UX</div>
                <div class="wm-kpi-label">Diseño enterprise limpio, veloz y comercialmente impecable.</div>
            </div>
            <div class="wm-kpi">
                <div class="wm-kpi-value">Industrial Ready</div>
                <div class="wm-kpi-label">Arquitectura orientada a confiabilidad, exportación y escalabilidad.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="wm-mini-row">
            <div class="wm-chip">Time Waveform</div>
            <div class="wm-chip">Orbit</div>
            <div class="wm-chip">FFT</div>
            <div class="wm-chip">Trend</div>
            <div class="wm-chip">Diagnostics</div>
            <div class="wm-chip">Reports HD</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="wm-login-card">', unsafe_allow_html=True)

    st.markdown('<div class="wm-right-head">Acceso seguro</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="wm-right-copy">Ingresa a la plataforma con tus credenciales internas. '
        'Entorno corporativo para monitoreo y análisis avanzado.</div>',
        unsafe_allow_html=True,
    )

    username = st.text_input(
        "Usuario o correo",
        placeholder="usuario@empresa.com",
        key="login_username",
    )

    password = st.text_input(
        "Clave",
        placeholder="Ingresa tu contraseña",
        type="password",
        key="login_password",
    )

    login_clicked = st.button("Ingresar al sistema", use_container_width=True)

    if login_clicked:
        if authenticate_user(username.strip(), password):
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Credenciales inválidas. Verifica usuario y contraseña.")

    st.markdown(
        """
        <div class="wm-note">
            Watermelon System está diseñado para ofrecer una experiencia de análisis industrial más moderna,
            más clara y más poderosa que el software legacy tradicional.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)