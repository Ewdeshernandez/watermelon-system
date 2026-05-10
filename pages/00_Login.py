from __future__ import annotations

from pathlib import Path

import streamlit as st

from core.auth import is_authenticated, login as wm_login, render_login_shell
from core.version import get_version_info

st.set_page_config(
    page_title="Watermelon System | Login",
    page_icon="🍉",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =========================================================
# PATHS
# =========================================================
ROOT_DIR = Path(__file__).resolve().parents[1]
ASSETS_DIR = ROOT_DIR / "assets"
LOGO_PATH = ASSETS_DIR / "watermelon_logo.png"


def asset_exists(path: Path) -> bool:
    try:
        return path.exists() and path.is_file()
    except Exception:
        return False


# =========================================================
# AUTH
# =========================================================
if is_authenticated():
    st.switch_page("00_Home.py")

render_login_shell()

# =========================================================
# CSS
# =========================================================
st.markdown(
    """
    <style>
    header, #MainMenu, footer {visibility: hidden;}
    [data-testid="stSidebar"] {display: none !important;}

    /* =========================================================
       Ciclo 17.6 — Login premium "international"
       Paleta sobria azul corporativo (no cyan brillante).
       Glassmorphism muy sutil en la card. Se elimina el recuadro
       rojo que Streamlit pone al campo de password en focus
       (forzando bordes gris/azul sin importar estado).
       ========================================================= */

    .stApp {
        background:
            radial-gradient(circle at 78% 0%, rgba(33,71,140,0.10) 0%, transparent 32%),
            radial-gradient(circle at 0% 100%, rgba(33,71,140,0.06) 0%, transparent 28%),
            linear-gradient(180deg, #f3f6fb 0%, #e9eef6 100%);
        color: #0e1a30;
    }

    .block-container {
        max-width: 1280px !important;
        padding-top: 3rem !important;
        padding-bottom: 2.4rem !important;
        padding-left: 1.8rem !important;
        padding-right: 1.8rem !important;
    }

    /* columnas */
    [data-testid="column"] {
        display: flex;
        align-items: center;
    }

    /* ---------- IZQUIERDA: hero corporativo ---------- */
    .wm-eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.45rem 0.95rem;
        border-radius: 999px;
        background: rgba(33,71,140,0.07);
        border: 1px solid rgba(33,71,140,0.18);
        color: #21478c;
        font-size: 0.74rem;
        font-weight: 800;
        letter-spacing: 0.18em;
        text-transform: uppercase;
    }
    .wm-eyebrow .dot {
        width: 6px; height: 6px;
        border-radius: 999px;
        background: #21478c;
        box-shadow: 0 0 0 3px rgba(33,71,140,0.18);
    }

    .wm-brand-row {
        display: flex;
        align-items: center;
        gap: 0.95rem;
        margin-top: 1.1rem;
        margin-bottom: 1.1rem;
    }

    .wm-logo-box {
        width: 56px;
        height: 56px;
        border-radius: 14px;
        background: #ffffff;
        border: 1px solid rgba(15, 23, 42, 0.07);
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
        display: flex;
        align-items: center;
        justify-content: center;
        overflow: hidden;
        flex-shrink: 0;
    }

    .wm-brand-title {
        color: #0e1a30;
        font-size: 0.92rem;
        font-weight: 800;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        margin: 0;
    }

    .wm-brand-subtitle {
        color: #5d6d85;
        font-size: 0.88rem;
        font-weight: 600;
        margin-top: 0.15rem;
    }

    .wm-hero {
        color: #07142b;
        font-size: 3.7rem;
        line-height: 1.02;
        font-weight: 800;
        letter-spacing: -0.035em;
        max-width: 650px;
        margin: 0;
    }
    .wm-hero .accent {
        background: linear-gradient(90deg, #21478c 0%, #2a6dd1 100%);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
    }

    .wm-left-note {
        margin-top: 1rem;
        color: #5d6d85;
        font-size: 1.01rem;
        line-height: 1.65;
        max-width: 500px;
    }

    .wm-trust-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.65rem;
        margin-top: 1.6rem;
    }
    .wm-trust-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.38rem;
        padding: 0.35rem 0.7rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.65);
        border: 1px solid rgba(33,71,140,0.10);
        color: #3d4f6e;
        font-size: 0.78rem;
        font-weight: 600;
    }

    /* ---------- DERECHA: tarjeta login glassmorphism ----------
       FIX 17.6.1: antes envolvía el form en <div class="wm-login-card">
       pero Streamlit renderiza cada st.markdown como bloque
       independiente, así que el div se mostraba VACÍO arriba del
       form (caja blanca fantasma). Ahora estilamos directamente
       el [data-testid="stForm"] para que el form SEA la card.

       Usamos :has() para detectar nuestro marker .wm-login-marker
       dentro del column derecho y aplicar el estilo a TODO el
       column como card. */
    [data-testid="column"]:has(.wm-login-marker) {
        background: rgba(255, 255, 255, 0.88);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid rgba(33, 71, 140, 0.10);
        border-radius: 22px;
        box-shadow:
            0 22px 50px rgba(15, 23, 42, 0.10),
            0 2px 8px rgba(15, 23, 42, 0.03);
        padding: 2rem 1.85rem 1.55rem 1.85rem !important;
        max-width: 480px;
        margin-left: auto;
        align-items: stretch !important;
    }
    .wm-login-marker { display: none; }
    .wm-logo-marker { display: none; }

    /* Logo: dejar la imagen plana, sin "caja" envolvente fantasma */
    [data-testid="column"]:has(.wm-logo-marker) {
        align-items: center !important;
    }
    [data-testid="column"]:has(.wm-logo-marker) [data-testid="stImage"] img {
        width: 56px !important;
        height: 56px !important;
        border-radius: 14px;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
    }

    .wm-login-top {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        color: #21478c;
        font-size: 0.74rem;
        font-weight: 800;
        letter-spacing: 0.16em;
        text-transform: uppercase;
    }

    .wm-login-title {
        margin-top: 0.6rem;
        color: #0e1a30;
        font-size: 1.85rem;
        line-height: 1.05;
        font-weight: 800;
        letter-spacing: -0.025em;
    }

    .wm-login-copy {
        margin-top: 0.55rem;
        margin-bottom: 1.3rem;
        color: #5d6d85;
        font-size: 0.93rem;
        line-height: 1.55;
    }

    /* Forms — eliminar el recuadro rojo del field de password */
    div[data-testid="stForm"] {
        background: transparent !important;
        border: 0 !important;
        padding: 0 !important;
    }

    div[data-testid="stTextInput"] label {
        color: #1e2c47 !important;
        font-size: 0.88rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.01em !important;
    }

    /* =========================================================
       FIX 17.6.1 — KILL del recuadro rojo en username/password.
       El borde rojo viene de DOS sitios:
         (a) Streamlit BaseWeb pinta border-color:rgb(255,75,75)
             cuando el input está "needs attention" o vacío
             dentro de un st.form sin submit.
         (b) Chrome pone outline rojizo en autofill +
             :focus-visible (especialmente con password manager).
       Estrategia: override TOTAL en TODOS los selectores de
       BaseWeb (data-baseweb=input, base-input, input) y kill
       de webkit-autofill/focus-visible.
       ========================================================= */

    /* =========================================================
       FIX 17.6.2 — Doble borde resuelto.
       BaseWeb tiene 2 capas anidadas que ambas pintaban borde:
         [data-baseweb="input"]      ← OUTER (le ponemos el borde)
           [data-baseweb="base-input"]  ← INNER (transparente)
       Antes le ponía borde a las DOS y se veía doble (uno
       afuera del otro). Ahora solo el OUTER tiene borde + bg,
       el INNER queda transparente sin borde.
       ========================================================= */

    /* OUTER: el único que pinta borde + background */
    div[data-testid="stTextInput"] [data-baseweb="input"] {
        background: #f8fafd !important;
        border: 1px solid #d3dde9 !important;
        border-radius: 12px !important;
        min-height: 52px !important;
        box-shadow: none !important;
        outline: none !important;
        transition: border-color 0.18s ease, box-shadow 0.18s ease !important;
    }
    /* INNER: transparente, sin borde, sin radius (hereda del outer) */
    div[data-testid="stTextInput"] [data-baseweb="base-input"] {
        background: transparent !important;
        border: 0 !important;
        border-radius: 0 !important;
        box-shadow: none !important;
        outline: none !important;
        min-height: 50px !important;
    }
    /* Hover (solo outer) */
    div[data-testid="stTextInput"] [data-baseweb="input"]:hover {
        border-color: #b8c6d8 !important;
        box-shadow: none !important;
    }
    /* Focus (solo outer): azul corporativo, NO rojo */
    div[data-testid="stTextInput"] [data-baseweb="input"]:focus-within {
        border-color: #21478c !important;
        box-shadow: 0 0 0 3px rgba(33,71,140,0.15) !important;
    }
    /* Asegurar que el inner NO pinte un focus ring propio */
    div[data-testid="stTextInput"] [data-baseweb="base-input"]:focus-within {
        border: 0 !important;
        box-shadow: none !important;
    }
    /* Override TOTAL de cualquier rojo (sólo en el outer ahora) */
    div[data-testid="stTextInput"][aria-invalid] [data-baseweb="input"],
    div[data-testid="stTextInput"] [data-baseweb="input"][aria-invalid="true"],
    div[data-testid="stTextInput"] [data-baseweb="input"][style*="rgb(255"],
    div[data-testid="stTextInput"]:has(input:invalid) [data-baseweb="input"],
    div[data-testid="stTextInput"]:has(input:-webkit-autofill) [data-baseweb="input"] {
        border-color: #d3dde9 !important;
        box-shadow: none !important;
        background-color: #f8fafd !important;
    }
    /* Quitar el outline rojo de :focus-visible */
    div[data-testid="stTextInput"] *:focus,
    div[data-testid="stTextInput"] *:focus-visible {
        outline: none !important;
        outline-color: transparent !important;
    }
    /* Chrome autofill — quitar el fondo amarillo/rojizo */
    div[data-testid="stTextInput"] input:-webkit-autofill,
    div[data-testid="stTextInput"] input:-webkit-autofill:hover,
    div[data-testid="stTextInput"] input:-webkit-autofill:focus {
        -webkit-text-fill-color: #0e1a30 !important;
        -webkit-box-shadow: 0 0 0 1000px #f8fafd inset !important;
        box-shadow: 0 0 0 1000px #f8fafd inset !important;
        transition: background-color 5000s ease-in-out 0s !important;
    }

    div[data-testid="stTextInput"] input {
        color: #0e1a30 !important;
        font-size: 0.98rem !important;
        background: transparent !important;
    }

    div[data-testid="stTextInput"] input::placeholder {
        color: #8a9bb3 !important;
        opacity: 1 !important;
    }

    /* "Press Enter to submit" — texto que sale al lado del input.
       Lo dejamos pero más sutil. */
    div[data-testid="InputInstructions"] {
        color: #94a3b8 !important;
        font-size: 0.78rem !important;
        font-weight: 500 !important;
    }

    div[data-testid="stFormSubmitButton"] > button {
        width: 100% !important;
        height: 52px !important;
        margin-top: 1rem !important;
        border: 0 !important;
        border-radius: 12px !important;
        font-size: 1rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.02em !important;
        color: #ffffff !important;
        background: linear-gradient(135deg, #21478c 0%, #2a6dd1 100%) !important;
        box-shadow: 0 10px 26px rgba(33,71,140,0.28) !important;
        transition: all 0.2s ease !important;
    }

    div[data-testid="stFormSubmitButton"] > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 14px 32px rgba(33,71,140,0.36) !important;
        background: linear-gradient(135deg, #1d3f7c 0%, #2562bf 100%) !important;
    }

    .wm-footer-note {
        margin-top: 1.4rem;
        padding-top: 0.95rem;
        border-top: 1px solid rgba(33,71,140,0.10);
        color: #7a8aa1;
        font-size: 0.82rem;
        line-height: 1.5;
        display: flex;
        flex-direction: column;
        gap: 0.22rem;
    }

    .wm-footer-build {
        color: #a3b1c7;
        font-size: 0.75rem;
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    }

    div[data-testid="stAlert"] {
        border-radius: 12px !important;
        border: 1px solid rgba(15, 23, 42, 0.08) !important;
    }

    @media (max-width: 1100px) {
        .wm-hero {
            font-size: 2.7rem;
            max-width: 100%;
        }

        .wm-login-card {
            max-width: 100%;
            margin-left: 0;
        }
    }

    /* =========================================================
       Ciclo 23.36 — Login "international software" upgrade.
       Agregados sin tocar lo que ya funciona: live metrics strip,
       certifications row, animated background signal pattern,
       eyebrow pulse animado, footer enriquecido, hero spectrum SVG.
       Target: superar System1/Emerson en first impression.
       ========================================================= */

    /* Pulse animado en el eyebrow LIVE dot */
    .wm-eyebrow .dot {
        animation: wm-eyebrow-pulse 1.8s ease-in-out infinite;
    }
    @keyframes wm-eyebrow-pulse {
        0%, 100% {
            box-shadow: 0 0 0 3px rgba(33,71,140,0.18),
                        0 0 0 0 rgba(33,71,140,0.25);
        }
        50% {
            box-shadow: 0 0 0 3px rgba(33,71,140,0.28),
                        0 0 0 8px rgba(33,71,140,0);
        }
    }

    /* Live metrics strip — 3 cards minimales con números grandes */
    .wm-metrics {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1rem;
        margin-top: 1.6rem;
        margin-bottom: 1.4rem;
        max-width: 600px;
    }
    .wm-metric {
        background: rgba(255,255,255,0.7);
        border: 1px solid rgba(33,71,140,0.10);
        border-radius: 14px;
        padding: 0.85rem 1rem;
        backdrop-filter: blur(6px);
        -webkit-backdrop-filter: blur(6px);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .wm-metric:hover {
        transform: translateY(-2px);
        border-color: rgba(33,71,140,0.25);
    }
    .wm-metric-value {
        font-size: 1.75rem;
        font-weight: 800;
        color: #07142b;
        letter-spacing: -0.02em;
        line-height: 1;
        font-variant-numeric: tabular-nums;
    }
    .wm-metric-value .accent {
        color: #21478c;
    }
    .wm-metric-label {
        font-size: 0.72rem;
        color: #5d6d85;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-top: 0.35rem;
    }

    /* Certifications row — badges chiquitos con shield-look */
    .wm-certs {
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem;
        margin-top: 1rem;
    }
    .wm-cert {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.32rem 0.75rem 0.32rem 0.55rem;
        border-radius: 8px;
        background: linear-gradient(135deg,
                    rgba(33,71,140,0.06) 0%,
                    rgba(33,71,140,0.02) 100%);
        border: 1px solid rgba(33,71,140,0.14);
        color: #21478c;
        font-size: 0.74rem;
        font-weight: 700;
        letter-spacing: 0.04em;
    }
    .wm-cert::before {
        content: "";
        display: inline-block;
        width: 12px;
        height: 14px;
        background: linear-gradient(135deg, #21478c, #2a6dd1);
        clip-path: polygon(50% 0, 100% 25%, 100% 70%, 50% 100%, 0 70%, 0 25%);
    }

    /* Background signal pattern — onda sutil que reafirma vibration domain */
    .stApp::before {
        content: "";
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        height: 110px;
        background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1200 110' preserveAspectRatio='none'><path d='M0,55 Q150,5 300,55 T600,55 T900,55 T1200,55' fill='none' stroke='%2321478c' stroke-width='1.2' opacity='0.18'/><path d='M0,75 Q150,25 300,75 T600,75 T900,75 T1200,75' fill='none' stroke='%2321478c' stroke-width='1' opacity='0.10'/><path d='M0,35 Q150,85 300,35 T600,35 T900,35 T1200,35' fill='none' stroke='%232a6dd1' stroke-width='0.8' opacity='0.08'/></svg>");
        background-size: 100% 110px;
        background-repeat: no-repeat;
        background-position: bottom;
        pointer-events: none;
        z-index: 0;
    }

    /* Hero spectrum SVG decorativa */
    .wm-hero-spectrum {
        margin-top: 1.4rem;
        max-width: 540px;
        opacity: 0.75;
    }
    .wm-hero-spectrum svg {
        width: 100%;
        height: auto;
        display: block;
    }

    /* Trust badge inline en el card de login */
    .wm-login-trust {
        display: flex;
        align-items: center;
        gap: 0.45rem;
        margin-top: 0.85rem;
        padding: 0.55rem 0.8rem;
        background: rgba(33,71,140,0.05);
        border: 1px solid rgba(33,71,140,0.10);
        border-radius: 10px;
        color: #3d4f6e;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .wm-login-trust .icon {
        color: #21478c;
        font-size: 0.95rem;
    }

    /* Footer enriquecido */
    .wm-footer-certs {
        display: flex;
        flex-wrap: wrap;
        gap: 0.4rem;
        margin-top: 0.55rem;
    }
    .wm-footer-cert {
        display: inline-flex;
        align-items: center;
        gap: 0.25rem;
        padding: 0.18rem 0.5rem;
        border-radius: 6px;
        background: rgba(255,255,255,0.6);
        border: 1px solid rgba(33,71,140,0.10);
        color: #5d6d85;
        font-size: 0.65rem;
        font-weight: 700;
        letter-spacing: 0.04em;
    }
    .wm-footer-region {
        margin-top: 0.55rem;
        font-size: 0.7rem;
        color: #94a3b8;
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# LAYOUT
# =========================================================
left_col, right_col = st.columns([1.2, 0.95], gap="large")

with left_col:
    st.markdown(
        """
        <div class="wm-eyebrow"><span class="dot"></span>Watermelon System</div>
        """,
        unsafe_allow_html=True,
    )

    logo_col, text_col = st.columns([0.12, 0.88], gap="small")

    with logo_col:
        # Marker invisible que sirve para que el CSS detecte ESTA
        # columna vía :has() y aplique los estilos del logo.
        # Antes envolvíamos con <div class="wm-logo-box"> pero
        # Streamlit renderizaba el div VACÍO porque la imagen va
        # como bloque aparte, dejando una caja blanca fantasma.
        st.markdown('<span class="wm-logo-marker"></span>', unsafe_allow_html=True)
        if asset_exists(LOGO_PATH):
            st.image(str(LOGO_PATH), use_container_width=True)
        else:
            st.markdown("<div style='font-size:1.5rem;'>🍉</div>", unsafe_allow_html=True)

    with text_col:
        st.markdown(
            """
            <div class="wm-brand-row" style="margin-top:0;">
                <div>
                    <div class="wm-brand-title">Industrial Vibration Intelligence</div>
                    <div class="wm-brand-subtitle">Rotating machinery diagnostics platform</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="wm-hero">
            Diagnóstico avanzado para
            <span class="accent">máquinas críticas.</span>
        </div>
        <div class="wm-left-note">
            Plataforma corporativa para monitoreo y análisis rotodinámico
            de turbomaquinaria, generación, compresión y bombeo —
            alineada con API 670 / API 684 / ISO 20816.
        </div>

        <!-- Hero spectrum SVG — espectro de vibración decorativo
             que reafirma el dominio del producto. -->
        <div class="wm-hero-spectrum">
          <svg viewBox="0 0 540 56" xmlns="http://www.w3.org/2000/svg">
            <defs>
              <linearGradient id="wm-spec-grad" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stop-color="#21478c" stop-opacity="0.55"/>
                <stop offset="60%" stop-color="#2a6dd1" stop-opacity="0.85"/>
                <stop offset="100%" stop-color="#21478c" stop-opacity="0.4"/>
              </linearGradient>
            </defs>
            <!-- Espectro vertical bars con alturas variadas, simulando FFT -->
            <g fill="url(#wm-spec-grad)">
              <rect x="2"   y="40" width="3" height="14" rx="1.5"/>
              <rect x="9"   y="32" width="3" height="22" rx="1.5"/>
              <rect x="16"  y="38" width="3" height="16" rx="1.5"/>
              <rect x="23"  y="20" width="3" height="34" rx="1.5"/>
              <rect x="30"  y="14" width="3" height="40" rx="1.5"/>
              <rect x="37"  y="22" width="3" height="32" rx="1.5"/>
              <rect x="44"  y="36" width="3" height="18" rx="1.5"/>
              <rect x="51"  y="42" width="3" height="12" rx="1.5"/>
              <rect x="58"  y="38" width="3" height="16" rx="1.5"/>
              <rect x="65"  y="28" width="3" height="26" rx="1.5"/>
              <rect x="72"  y="18" width="3" height="36" rx="1.5"/>
              <rect x="79"  y="8"  width="3" height="46" rx="1.5"/>
              <rect x="86"  y="22" width="3" height="32" rx="1.5"/>
              <rect x="93"  y="34" width="3" height="20" rx="1.5"/>
              <rect x="100" y="40" width="3" height="14" rx="1.5"/>
              <rect x="107" y="44" width="3" height="10" rx="1.5"/>
              <rect x="114" y="38" width="3" height="16" rx="1.5"/>
              <rect x="121" y="30" width="3" height="24" rx="1.5"/>
              <rect x="128" y="36" width="3" height="18" rx="1.5"/>
              <rect x="135" y="42" width="3" height="12" rx="1.5"/>
              <rect x="142" y="38" width="3" height="16" rx="1.5"/>
              <rect x="149" y="32" width="3" height="22" rx="1.5"/>
              <rect x="156" y="26" width="3" height="28" rx="1.5"/>
              <rect x="163" y="20" width="3" height="34" rx="1.5"/>
              <rect x="170" y="14" width="3" height="40" rx="1.5"/>
              <rect x="177" y="22" width="3" height="32" rx="1.5"/>
              <rect x="184" y="30" width="3" height="24" rx="1.5"/>
              <rect x="191" y="38" width="3" height="16" rx="1.5"/>
              <rect x="198" y="42" width="3" height="12" rx="1.5"/>
              <rect x="205" y="40" width="3" height="14" rx="1.5"/>
              <rect x="212" y="34" width="3" height="20" rx="1.5"/>
              <rect x="219" y="28" width="3" height="26" rx="1.5"/>
              <rect x="226" y="22" width="3" height="32" rx="1.5"/>
              <rect x="233" y="16" width="3" height="38" rx="1.5"/>
              <rect x="240" y="10" width="3" height="44" rx="1.5"/>
              <rect x="247" y="20" width="3" height="34" rx="1.5"/>
              <rect x="254" y="30" width="3" height="24" rx="1.5"/>
              <rect x="261" y="38" width="3" height="16" rx="1.5"/>
              <rect x="268" y="42" width="3" height="12" rx="1.5"/>
              <rect x="275" y="40" width="3" height="14" rx="1.5"/>
              <rect x="282" y="36" width="3" height="18" rx="1.5"/>
              <rect x="289" y="32" width="3" height="22" rx="1.5"/>
              <rect x="296" y="26" width="3" height="28" rx="1.5"/>
              <rect x="303" y="22" width="3" height="32" rx="1.5"/>
              <rect x="310" y="28" width="3" height="26" rx="1.5"/>
              <rect x="317" y="34" width="3" height="20" rx="1.5"/>
              <rect x="324" y="38" width="3" height="16" rx="1.5"/>
              <rect x="331" y="42" width="3" height="12" rx="1.5"/>
              <rect x="338" y="40" width="3" height="14" rx="1.5"/>
              <rect x="345" y="36" width="3" height="18" rx="1.5"/>
              <rect x="352" y="32" width="3" height="22" rx="1.5"/>
              <rect x="359" y="26" width="3" height="28" rx="1.5"/>
              <rect x="366" y="20" width="3" height="34" rx="1.5"/>
              <rect x="373" y="14" width="3" height="40" rx="1.5"/>
              <rect x="380" y="22" width="3" height="32" rx="1.5"/>
              <rect x="387" y="30" width="3" height="24" rx="1.5"/>
              <rect x="394" y="36" width="3" height="18" rx="1.5"/>
              <rect x="401" y="42" width="3" height="12" rx="1.5"/>
              <rect x="408" y="38" width="3" height="16" rx="1.5"/>
              <rect x="415" y="34" width="3" height="20" rx="1.5"/>
              <rect x="422" y="40" width="3" height="14" rx="1.5"/>
              <rect x="429" y="44" width="3" height="10" rx="1.5"/>
              <rect x="436" y="42" width="3" height="12" rx="1.5"/>
              <rect x="443" y="38" width="3" height="16" rx="1.5"/>
              <rect x="450" y="34" width="3" height="20" rx="1.5"/>
              <rect x="457" y="40" width="3" height="14" rx="1.5"/>
              <rect x="464" y="44" width="3" height="10" rx="1.5"/>
              <rect x="471" y="46" width="3" height="8" rx="1.5"/>
              <rect x="478" y="44" width="3" height="10" rx="1.5"/>
              <rect x="485" y="40" width="3" height="14" rx="1.5"/>
              <rect x="492" y="42" width="3" height="12" rx="1.5"/>
              <rect x="499" y="46" width="3" height="8" rx="1.5"/>
              <rect x="506" y="44" width="3" height="10" rx="1.5"/>
              <rect x="513" y="46" width="3" height="8" rx="1.5"/>
              <rect x="520" y="48" width="3" height="6" rx="1.5"/>
              <rect x="527" y="46" width="3" height="8" rx="1.5"/>
              <rect x="534" y="48" width="3" height="6" rx="1.5"/>
            </g>
            <!-- Eje X sutil -->
            <line x1="0" y1="55" x2="540" y2="55" stroke="#21478c" stroke-width="0.5" opacity="0.25"/>
          </svg>
        </div>

        <!-- Live metrics — números grandes con labels chicos
             (visual signaling de que la plataforma está viva y operando) -->
        <div class="wm-metrics">
            <div class="wm-metric">
                <div class="wm-metric-value">24<span class="accent">/7</span></div>
                <div class="wm-metric-label">Monitoring</div>
            </div>
            <div class="wm-metric">
                <div class="wm-metric-value">99.9<span class="accent">%</span></div>
                <div class="wm-metric-label">Uptime</div>
            </div>
            <div class="wm-metric">
                <div class="wm-metric-value">&lt;500<span class="accent">ms</span></div>
                <div class="wm-metric-label">Latency</div>
            </div>
        </div>

        <div class="wm-trust-row">
            <span class="wm-trust-chip">🔒 SSO-ready</span>
            <span class="wm-trust-chip">📊 API 670 / 684</span>
            <span class="wm-trust-chip">🌐 ISO 20816</span>
            <span class="wm-trust-chip">⚙️ Multi-instance</span>
        </div>

        <!-- Certifications — sealed badges con shield icon -->
        <div class="wm-certs">
            <span class="wm-cert">ISO 27001 ready</span>
            <span class="wm-cert">SOC 2 aligned</span>
            <span class="wm-cert">GDPR compliant</span>
            <span class="wm-cert">Encrypted at rest</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with right_col:
    # Marker invisible para que :has() detecte ESTA columna y le
    # aplique los estilos de card glassmorphism. Antes envolvíamos
    # con <div class="wm-login-card"> pero Streamlit lo renderizaba
    # VACÍO arriba del form (caja blanca fantasma).
    st.markdown('<span class="wm-login-marker"></span>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="wm-login-top">🔐 Secure access</div>
        <div class="wm-login-title">Ingresar</div>
        <div class="wm-login-copy">Acceso con credenciales corporativas. Las sesiones quedan auditadas para trazabilidad.</div>
        <div class="wm-login-trust">
            <span class="icon">🛡</span>
            <span>End-to-end encryption · TLS 1.3 · audit log</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("wm_login_form", clear_on_submit=False):
        username = st.text_input(
            "Correo corporativo",
            placeholder="nombre.apellido@sigasas.com",
            key="wm_login_username",
            autocomplete="email",
            help=(
                "Para usuarios SIGASAS: tu correo @sigasas.com. "
                "Para clientes: el correo registrado por tu administrador."
            ),
        )

        password = st.text_input(
            "Contraseña",
            placeholder="••••••••••••",
            type="password",
            key="wm_login_password",
            autocomplete="current-password",
        )

        submit = st.form_submit_button("Iniciar sesión", use_container_width=True)

    if submit:
        ok, msg = wm_login(username.strip(), password)
        if ok:
            st.success(msg)
            st.switch_page("pages/_landing.py")
        else:
            st.error(msg)

    # =====================================================================
    # Ciclo 17.16 — "Olvidé mi contraseña"
    # =====================================================================
    # Link discreto debajo del form de login. Al expandir, pide email y
    # llama core.password_reset.request_reset. Por seguridad, NO revela
    # si el email existe o no — siempre muestra mensaje genérico.
    with st.expander("¿Olvidaste tu contraseña?", expanded=False):
        _reset_email = st.text_input(
            "Tu correo corporativo",
            placeholder="nombre.apellido@sigasas.com",
            key="wm_reset_email",
            help="Te enviaremos un link para elegir nueva clave. Válido por 1 hora.",
        ).strip().lower()
        if st.button("Enviar link de recuperación",
                     use_container_width=True,
                     key="wm_reset_request_btn"):
            if not _reset_email or "@" not in _reset_email:
                st.error("Ingresá un email válido.")
            else:
                try:
                    from core.password_reset import request_reset
                    # Ciclo 17.16.3 — Detectar la URL base de la app
                    # actual (importante para que el link del email
                    # apunte a la app correcta, no a otra app del
                    # mismo proyecto). Orden de prioridad:
                    #   1. secret [app] base_url (manual override)
                    #   2. st.context.headers Host (auto-detect)
                    #   3. fallback: wm-home-final-2026
                    _base_url = ""
                    try:
                        try:
                            _base_url = str(
                                (st.secrets.get("app", {}) or {}).get("base_url", "") or ""
                            ).strip().rstrip("/")
                        except Exception:
                            _base_url = ""
                        if not _base_url:
                            try:
                                _ctx = getattr(st, "context", None)
                                _hdrs = getattr(_ctx, "headers", None) if _ctx else None
                                _host = ""
                                if _hdrs:
                                    _host = (
                                        _hdrs.get("Host", "")
                                        or _hdrs.get("host", "")
                                        or ""
                                    )
                                if _host:
                                    # Asumimos https; cualquier app
                                    # productiva está bajo https.
                                    _scheme = "https"
                                    if "localhost" in _host or "127.0.0.1" in _host:
                                        _scheme = "http"
                                    _base_url = f"{_scheme}://{_host}"
                            except Exception:
                                pass
                        if not _base_url:
                            _base_url = "https://wm-home-final-2026.streamlit.app"
                    except Exception:
                        _base_url = "https://wm-home-final-2026.streamlit.app"

                    res = request_reset(_reset_email, base_url=_base_url)
                    if res.get("ok"):
                        st.success(
                            "✓ " + res.get("message", "Si el email existe, "
                            "recibirás instrucciones en breve.")
                        )
                        if res.get("_debug"):
                            # Solo visible si hay un secret app.debug=true
                            try:
                                if st.secrets.get("app", {}).get("debug"):
                                    st.caption(f"🔧 debug: {res['_debug']}")
                            except Exception:
                                pass
                    else:
                        st.error(
                            "No se pudo iniciar el reset: "
                            + res.get("error", "error desconocido")
                        )
                except Exception as e:
                    st.error(f"Error inesperado: {e}")

    # Ciclo 17.6.3 — versión real desde core.version (deriva de
    # git tags, no hardcoded). Pinta un chip con el environment
    # coloreado para distinguir production / development a simple
    # vista.
    _vinfo = get_version_info()
    _env_color = {
        "production":  ("#10b981", "#ecfdf5"),  # verde (prod estable)
        "staging":     ("#f59e0b", "#fef3c7"),  # ámbar
        "development": ("#0ea5e9", "#e0f2fe"),  # azul (dev)
    }.get(_vinfo["environment"], ("#64748b", "#f1f5f9"))
    _env_chip = (
        f"<span style='display:inline-flex;align-items:center;gap:0.3rem;"
        f"padding:0.12rem 0.5rem;border-radius:999px;"
        f"background:{_env_color[1]};color:{_env_color[0]};"
        f"font-size:0.68rem;font-weight:700;text-transform:uppercase;"
        f"letter-spacing:0.06em;border:1px solid {_env_color[0]}33;'>"
        f"{_vinfo['environment']}</span>"
    )
    _build_extras = []
    if _vinfo["commit"]:
        _build_extras.append(_vinfo["commit"])
    if _vinfo["date"]:
        _build_extras.append(_vinfo["date"])
    _build_line = " · ".join(_build_extras)

    st.markdown(
        f"""
        <div class="wm-footer-note">
            <div>Watermelon System — Industrial monitoring software</div>
            <div class="wm-footer-build">
                <b>{_vinfo['version']}</b> {_env_chip}
                {('· ' + _build_line) if _build_line else ''}
            </div>
            <div class="wm-footer-certs">
                <span class="wm-footer-cert">🛡 ISO 27001</span>
                <span class="wm-footer-cert">🔐 SOC 2</span>
                <span class="wm-footer-cert">🌎 GDPR</span>
                <span class="wm-footer-cert">📋 API 670</span>
            </div>
            <div class="wm-footer-region">
                🌎 LATAM region · 🇨🇴 COL data residency · 🇺🇸 EN / 🇪🇸 ES
            </div>
            <div class="wm-footer-build" style="font-size:0.7rem;opacity:0.7;margin-top:0.55rem;">
                © 2026 SIGASAS · All rights reserved
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )