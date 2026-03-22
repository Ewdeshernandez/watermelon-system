import streamlit as st

st.set_page_config(
    page_title="Watermelon System | Login",
    page_icon="🍉",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =========================================================
# SESSION DEFAULTS
# =========================================================
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

# =========================================================
# AUTH BRIDGE
# IMPORTANTE:
# AQUÍ CONECTAS TU AUTH REAL.
# NO METO AUTH DEMO PARA NO DAÑARTE EL INGRESO.
# =========================================================
def do_login(username: str, password: str) -> bool:
    """
    Conecta aquí tu validación real.
    Ejemplos:
        return authenticate_user(username, password)
        return login_user(username, password)
        return auth_manager.login(username, password)

    TEMPORAL:
    Si quieres probar visualmente mientras conectas auth real,
    descomenta la línea del return de abajo.
    """

    # ====== PEGA AQUÍ TU LÓGICA REAL ======
    # return authenticate_user(username, password)

    # ====== SOLO PARA PRUEBA VISUAL, BORRAR DESPUÉS ======
    return False


# Si ya hay sesión, no mostramos login
if st.session_state.get("authenticated", False):
    st.success("Sesión activa")
    st.stop()

# =========================================================
# CSS
# =========================================================
st.markdown(
    """
    <style>
    header, #MainMenu, footer {visibility: hidden;}
    [data-testid="stSidebar"] {display: none;}

    .stApp {
        background:
            radial-gradient(circle at 12% 18%, rgba(30,167,255,0.20) 0%, transparent 28%),
            radial-gradient(circle at 88% 82%, rgba(255,45,85,0.16) 0%, transparent 30%),
            linear-gradient(135deg, #07111f 0%, #0a1426 45%, #0d172b 100%);
        color: #f3f7ff;
    }

    .block-container {
        max-width: 1380px;
        padding-top: 2rem;
        padding-bottom: 1.5rem;
        padding-left: 2.5rem;
        padding-right: 2.5rem;
    }

    .wm-shell {
        min-height: 82vh;
        border-radius: 28px;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.08);
        background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02));
        box-shadow: 0 30px 80px rgba(0,0,0,0.45);
        padding: 1.2rem;
    }

    .wm-left-card {
        min-height: 74vh;
        padding: 3rem;
        border-radius: 24px;
        background: linear-gradient(180deg, rgba(11, 20, 38, 0.88), rgba(8, 15, 28, 0.76));
        border: 1px solid rgba(255,255,255,0.06);
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.05), 0 18px 44px rgba(0,0,0,0.28);
    }

    .wm-login-card {
        margin-top: 2rem;
        padding: 1.8rem;
        border-radius: 24px;
        background: linear-gradient(180deg, rgba(12, 19, 34, 0.96), rgba(9, 16, 30, 0.90));
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 20px 50px rgba(0,0,0,0.42);
    }

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
        margin-top: 0.8rem;
        color: #49b8ff;
        font-size: 0.92rem;
        font-weight: 700;
        letter-spacing: 0.18em;
        text-transform: uppercase;
    }

    .wm-title {
        margin-top: 1rem;
        font-size: 3.8rem;
        line-height: 0.98;
        font-weight: 900;
        color: #f5f9ff;
        letter-spacing: -0.04em;
    }

    .wm-subtitle {
        margin-top: 1rem;
        max-width: 720px;
        font-size: 1.02rem;
        line-height: 1.7;
        color: #a9b7cb;
    }

    .wm-kpis {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.95rem;
        margin-top: 1.8rem;
    }

    .wm-kpi {
        padding: 1rem;
        border-radius: 18px;
        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(255,255,255,0.06);
    }

    .wm-kpi-value {
        font-size: 1.12rem;
        font-weight: 800;
        color: #ffffff;
    }

    .wm-kpi-label {
        margin-top: 0.25rem;
        font-size: 0.84rem;
        color: #9db0c8;
        line-height: 1.45;
    }

    .wm-right-head {
        color: #eef5ff;
        font-size: 1.55rem;
        font-weight: 850;
        letter-spacing: -0.03em;
        margin-bottom: 0.2rem;
    }

    .wm-right-copy {
        color: #93a6bf;
        font-size: 0.93rem;
        line-height: 1.55;
        margin-bottom: 1rem;
    }

    div[data-testid="stTextInput"] label {
        color: #dbe7f7 !important;
        font-weight: 650 !important;
        font-size: 0.92rem !important;
    }

    div[data-testid="stTextInput"] > div > div {
        background: linear-gradient(180deg, rgba(15,24,42,0.98), rgba(12,20,36,0.95)) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 16px !important;
        min-height: 52px !important;
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

    .stButton > button {
        width: 100%;
        height: 52px;
        margin-top: 0.35rem;
        border: 0 !important;
        border-radius: 16px !important;
        color: white !important;
        font-weight: 800 !important;
        font-size: 1rem !important;
        background: linear-gradient(90deg, #1593ff 0%, #1ea7ff 35%, #24c6ff 60%, #ff4d6d 100%) !important;
        box-shadow: 0 14px 30px rgba(18, 131, 255, 0.24), 0 6px 20px rgba(255, 77, 109, 0.16);
        transition: all 0.22s ease;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 18px 34px rgba(18, 131, 255, 0.34), 0 8px 24px rgba(255, 77, 109, 0.22);
    }

    .wm-note {
        margin-top: 0.9rem;
        padding: 0.9rem 1rem;
        border-radius: 16px;
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.06);
        color: #98acc5;
        font-size: 0.88rem;
        line-height: 1.55;
    }

    .wm-mini-row {
        display: flex;
        gap: 0.65rem;
        flex-wrap: wrap;
        margin-top: 1.1rem;
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

    @media (max-width: 1100px) {
        .wm-title { font-size: 3rem; }
        .wm-kpis { grid-template-columns: 1fr; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# LAYOUT
# =========================================================
st.markdown('<div class="wm-shell">', unsafe_allow_html=True)

left_col, right_col = st.columns([1.65, 0.92], gap="large")

with left_col:
    st.markdown('<div class="wm-left-card">', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="wm-badge">● Watermelon System</div>
        <div class="wm-eyebrow">Industrial Vibration Intelligence</div>
        <div class="wm-title">
            El nuevo estándar<br>
            en monitoreo<br>
            de vibraciones.
        </div>
        <div class="wm-subtitle">
            Plataforma industrial premium para análisis, visualización y diagnóstico moderno.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="wm-kpis">
            <div class="wm-kpi">
                <div class="wm-kpi-value">Más claro</div>
                <div class="wm-kpi-label">Visuales premium para análisis técnico serio.</div>
            </div>
            <div class="wm-kpi">
                <div class="wm-kpi-value">Más rápido</div>
                <div class="wm-kpi-label">Experiencia moderna enfocada en operación real.</div>
            </div>
            <div class="wm-kpi">
                <div class="wm-kpi-value">Más potente</div>
                <div class="wm-kpi-label">Arquitectura lista para escalar módulos y diagnóstico.</div>
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
            <div class="wm-chip">Reports</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="wm-login-card">', unsafe_allow_html=True)

    st.markdown('<div class="wm-right-head">Acceso seguro</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="wm-right-copy">Ingresa con tus credenciales.</div>',
        unsafe_allow_html=True,
    )

    with st.form("login_form", clear_on_submit=False):
        username = st.text_input(
            "Usuario o correo",
            placeholder="usuario",
            key="login_username",
        )

        password = st.text_input(
            "Clave",
            placeholder="••••••••",
            type="password",
            key="login_password",
        )

        login_clicked = st.form_submit_button("Ingresar al sistema", use_container_width=True)

    if login_clicked:
        ok = do_login(username.strip(), password)
        if ok:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Credenciales inválidas.")

    st.markdown(
        """
        <div class="wm-note">
            Plataforma premium de análisis industrial.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)