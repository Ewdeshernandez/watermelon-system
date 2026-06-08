import streamlit as st

def apply_theme():
    st.markdown("""
    <style>

    /* ===== GLOBAL ===== */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        color: #0f172a;
    }

    /* ===== SIDEBAR ===== Ciclo 23.38 — paleta navy dark
       (consistente con el upgrade del sidebar en core/auth.py).
       Reglas básicas; el detail styling vive en auth.py. */
    section[data-testid="stSidebar"] * {
        color: #f1f5f9;
        font-weight: 500;
    }
    .stSidebar button {
        background: transparent;
        border: none;
        text-align: left;
        padding: 0.5rem 0.8rem;
        border-radius: 8px;
        font-size: 0.84rem;
        transition: 0.15s;
    }
    .stSidebar button:hover {
        background: rgba(255,255,255,0.08);
    }

    /* ===== CHROME STREAMLIT OCULTO ===== Ciclo 23.161 — feedback cliente
       (Juan Mora): el menú de 3 puntos (theme/Print/Record screen/"Made
       with Streamlit") no aporta y resta seriedad. Se oculta el toolbar,
       el menú principal, el footer y el botón de deploy. */
    #MainMenu {visibility: hidden;}
    [data-testid="stMainMenu"] {display: none !important;}
    [data-testid="stToolbarActions"] {display: none !important;}
    [data-testid="stDecoration"] {display: none !important;}
    header[data-testid="stHeader"] {background: transparent;}
    footer {visibility: hidden; height: 0;}
    .stDeployButton {display: none !important;}
    #stDecoration {display: none !important;}

    /* ===== LOADER SANDÍA ===== Ciclo 23.161 — feedback cliente: el ícono
       nativo de "RUNNING" (bici / nadador) se reemplaza por una 🍉 que
       late, tanto en el status widget (arriba derecha) como en st.spinner. */
    [data-testid="stStatusWidget"] svg,
    [data-testid="stStatusWidget"] img {display: none !important;}
    [data-testid="stStatusWidget"] > div:first-child::before {
        content: "🍉";
        font-size: 17px;
        display: inline-block;
        animation: wmPulse 1s ease-in-out infinite;
        margin-right: 2px;
    }
    [data-testid="stSpinner"] svg {display: none !important;}
    [data-testid="stSpinner"] > div::before {
        content: "🍉";
        font-size: 20px;
        display: inline-block;
        animation: wmSpin 1.1s linear infinite;
        margin-right: 8px;
    }
    @keyframes wmPulse {
        0%, 100% {transform: scale(1); opacity: 1;}
        50% {transform: scale(1.25); opacity: 0.6;}
    }
    @keyframes wmSpin {
        0% {transform: rotate(0deg);}
        100% {transform: rotate(360deg);}
    }

    /* ===== HEADER LIMPIO ===== */
    .block-container {
        padding-top: 2rem;
    }

    /* ===== HERO ===== */
    .wm-hero {
        background: #f8fafc;
        border-radius: 16px;
        padding: 40px;
        border: 1px solid #e2e8f0;
    }

    .wm-title {
        font-size: 42px;
        font-weight: 700;
        letter-spacing: -0.5px;
    }

    .wm-subtitle {
        font-size: 16px;
        color: #64748b;
        margin-top: 8px;
    }

    /* ===== GRID ===== */
    .wm-grid {
        margin-top: 30px;
    }

    .wm-card {
        background: white;
        border-radius: 14px;
        padding: 20px;
        border: 1px solid #e2e8f0;
        transition: 0.2s;
    }

    .wm-card:hover {
        border: 1px solid #cbd5f5;
        transform: translateY(-2px);
    }

    .wm-card-title {
        font-size: 16px;
        font-weight: 600;
    }

    .wm-card-button button {
        width: 100%;
        border-radius: 10px;
        height: 38px;
        font-weight: 500;
    }

    </style>
    """, unsafe_allow_html=True)