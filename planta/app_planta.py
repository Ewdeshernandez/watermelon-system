"""
planta/app_planta.py — Watermelon Planta Edition · Entry Point
================================================================

Mini-app Streamlit standalone para captura modal en planta sin internet.
Diseñada para correr en laptop de campo con la maleta Watermelon de
adquisición de alta precisión conectada por USB.

NO requiere:
- Internet (todo el flujo es local)
- Supabase / login (sin auth)
- API externa (sin Anthropic, sin OpenAI)
- Acceso al repo completo de Watermelon (solo necesita esta carpeta y
  reusable de core/modal/)

SÍ requiere:
- Python 3.10+ instalado en el PC
- Drivers de adquisición Watermelon instalados en el sistema
- numpy, pandas, streamlit, plotly

Uso típico:
    streamlit run app_planta.py
    → abre browser en localhost:8501
    → técnico configura canales + dispara captura
    → TDMS queda en planta/data/captures/

Después en oficina (con internet):
    upload los TDMS al Watermelon Cloud manualmente (FASE A)
    o el sync uploader auto-detecta y sube (FASE B, próximo sprint)
"""
from __future__ import annotations

import streamlit as st
from pathlib import Path
import sys

# Agregar el repo root al path para poder importar core.modal.*
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

st.set_page_config(
    page_title="Watermelon Planta · Captura NI",
    page_icon="🍉",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Asegurar carpetas de datos
_CAPTURES_DIR = Path(__file__).parent / "data" / "captures"
_CAPTURES_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================================
# Tema visual global (v3.31.214 — FASE E branding consistente)
# Tipografía Inter / system fonts + colores SIGA + spacing pulido
#
# v3.31.246 — MOVIDO antes del check de licencia para que la pantalla
# bloqueante de licencia inválida también herede el styling. Antes el
# st.stop() del bloqueador cortaba la ejecución y la pantalla salía con
# look default de Streamlit (header "Deploy" visible, sin sidebar azul,
# tipografía cruda). Cliente reportó visual "horrible y asqueroso".
# =====================================================================
st.markdown("""
<style>
    /* Tipografía consistente — Inter / SF Pro / Segoe */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI',
                      Inter, 'Helvetica Neue', sans-serif !important;
    }
    /* Botones primary con gradient SIGA (verde teal) */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #0f766e 0%, #0d9488 100%) !important;
        border: none !important;
        box-shadow: 0 2px 8px rgba(15,118,110,0.25) !important;
        font-weight: 600 !important;
        transition: all 0.15s ease !important;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(15,118,110,0.35) !important;
    }
    /* Metrics con borde sutil */
    [data-testid="stMetric"] {
        background: rgba(15,118,110,0.04) !important;
        border-radius: 10px !important;
        padding: 14px 16px !important;
        border: 1px solid rgba(15,118,110,0.10) !important;
    }
    /* Quitar el "Made with Streamlit" footer default */
    footer {visibility: hidden !important;}
    /* Reducir padding top global */
    .block-container {
        padding-top: 2rem !important;
        max-width: 1400px !important;
    }

    /* ============================================================
       SIDEBAR — clase mundial industrial (FASE H2.6 v3.31.219)
       ============================================================ */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #fafbfc 0%, #f1f5f9 100%) !important;
        border-right: 1px solid rgba(15,118,110,0.08) !important;
    }
    section[data-testid="stSidebar"] > div {
        padding-top: 1rem !important;
    }
    /* OCULTAR la nav default de Streamlit (app planta / Captura Modal) */
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] {
        display: none !important;
    }
    section[data-testid="stSidebar"] h5 {
        font-size: 10px !important;
        font-weight: 700 !important;
        letter-spacing: 2.5px !important;
        text-transform: uppercase !important;
        color: #475569 !important;
        margin: 0 0 10px 4px !important;
        padding-top: 4px !important;
    }
    section[data-testid="stSidebar"] hr {
        margin: 16px 0 !important;
        border-color: rgba(15,118,110,0.08) !important;
    }
    /* Pulse animation para status dot */
    @keyframes wm-pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50%      { opacity: 0.55; transform: scale(1.15); }
    }
    .wm-pulse-dot {
        animation: wm-pulse 1.6s ease-in-out infinite;
        display: inline-block;
    }
    /* Botones de acceso rápido en sidebar — premium gradient cards */
    section[data-testid="stSidebar"] .stButton > button {
        border-radius: 10px !important;
        font-size: 13px !important;
        font-weight: 700 !important;
        letter-spacing: 0.4px !important;
        transition: all 0.18s cubic-bezier(.4,0,.2,1) !important;
        padding: 14px 12px !important;
        background: linear-gradient(135deg, #ffffff 0%, #f0fdfa 100%) !important;
        border: 1px solid rgba(15,118,110,0.20) !important;
        color: #0f766e !important;
        box-shadow: 0 1px 3px rgba(15,118,110,0.08),
                    inset 0 1px 0 rgba(255,255,255,0.5) !important;
        text-align: center !important;
        line-height: 1.3 !important;
    }
    section[data-testid="stSidebar"] .stButton > button:hover:not(:disabled) {
        background: linear-gradient(135deg, #0f766e 0%, #0d9488 100%) !important;
        color: #ffffff !important;
        border-color: #0f766e !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 16px rgba(15,118,110,0.28),
                    inset 0 1px 0 rgba(255,255,255,0.18) !important;
    }
    section[data-testid="stSidebar"] .stButton > button:disabled {
        background: #f8fafc !important;
        color: #94a3b8 !important;
        border-color: rgba(148,163,184,0.25) !important;
        box-shadow: none !important;
    }
    /* Page links en sidebar */
    section[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"] {
        border-radius: 8px !important;
        padding: 8px 12px !important;
        transition: all 0.15s ease !important;
    }
    section[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"]:hover {
        background: rgba(15,118,110,0.08) !important;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================================
# Verificación de licencia (v3.31.215 — FASE D)
# Bloquea la app si la licencia no es válida. Cachea en session_state
# para no releer el disco en cada rerun.
#
# v3.31.246 — Este bloque se ejecuta DESPUÉS del CSS global para que
# la pantalla bloqueante de licencia inválida salga con sidebar azul,
# tipografía Inter y "Deploy" oculto.
# =====================================================================
try:
    from license_manager import (
        get_cached_license,
        render_license_blocker,
        render_license_status_chip,
    )
    _LIC = get_cached_license()
    if not _LIC.valid:
        # Mostrar pantalla bloqueante y detener toda la app
        render_license_blocker(_LIC)
        st.stop()
except ImportError:
    # license_manager.py debe existir siempre. Si no está, instalación corrupta.
    st.error(
        "⚠ Instalación corrupta: falta `license_manager.py`.\n\n"
        "Contacta a SIGA GROUP para reinstalar Watermelon Planta."
    )
    st.stop()

# =====================================================================
# Header premium clase mundial (v3.31.214 — FASE E branding)
# Diseño: gradient navy→teal + logo SVG inline + tagline + meta chips
# =====================================================================
_app_version = "v1.0.0"
try:
    _ver_file = _REPO_ROOT / "VERSION"
    if _ver_file.exists():
        _app_version = _ver_file.read_text().strip()
except Exception:
    pass

st.markdown(
    f"""
    <div style="
        background:linear-gradient(135deg,#1e3a8a 0%,#1e40af 45%,#0f766e 100%);
        padding:28px 32px;border-radius:14px;color:white;
        margin-bottom:24px;
        box-shadow:0 6px 24px rgba(15,118,110,0.22),
                   0 2px 8px rgba(30,58,138,0.18);
        position:relative;overflow:hidden;">
        <!-- Pattern decorativo sutil arriba derecha -->
        <div style="position:absolute;top:-40px;right:-40px;
                    width:200px;height:200px;border-radius:50%;
                    background:radial-gradient(circle,
                                                rgba(255,255,255,0.08) 0%,
                                                rgba(255,255,255,0) 70%);
                    pointer-events:none;"></div>
        <!-- Logo inline + título -->
        <div style="display:flex;align-items:center;gap:18px;position:relative;">
            <svg width="56" height="56" viewBox="0 0 256 256"
                 xmlns="http://www.w3.org/2000/svg"
                 style="flex-shrink:0;
                        filter:drop-shadow(0 2px 6px rgba(0,0,0,0.25));">
                <defs>
                    <linearGradient id="hdrRind" x1="0%" y1="100%"
                                    x2="0%" y2="0%">
                        <stop offset="0%" stop-color="#14532d"/>
                        <stop offset="100%" stop-color="#4ade80"/>
                    </linearGradient>
                    <linearGradient id="hdrMeat" x1="0%" y1="100%"
                                    x2="0%" y2="0%">
                        <stop offset="0%" stop-color="#9f1239"/>
                        <stop offset="100%" stop-color="#fda4af"/>
                    </linearGradient>
                </defs>
                <circle cx="128" cy="128" r="124"
                        fill="rgba(255,255,255,0.12)"/>
                <g transform="translate(128 140)">
                    <path d="M -88 0 A 88 88 0 0 0 88 0 L 80 0 A 80 80 0 0 1 -80 0 Z"
                          fill="url(#hdrRind)"/>
                    <path d="M -80 0 A 80 80 0 0 0 80 0 L 74 0 A 74 74 0 0 1 -74 0 Z"
                          fill="#bbf7d0"/>
                    <path d="M -74 0 A 74 74 0 0 0 74 0 Z"
                          fill="url(#hdrMeat)"/>
                    <ellipse cx="-40" cy="22" rx="4" ry="7"
                             fill="#1f2937" transform="rotate(-15 -40 22)"/>
                    <ellipse cx="-12" cy="38" rx="4" ry="7" fill="#1f2937"/>
                    <ellipse cx="16" cy="36" rx="4" ry="7"
                             fill="#1f2937" transform="rotate(8 16 36)"/>
                    <ellipse cx="40" cy="22" rx="4" ry="7"
                             fill="#1f2937" transform="rotate(15 40 22)"/>
                </g>
            </svg>
            <div style="flex-grow:1;">
                <div style="font-size:13px;font-weight:600;
                            color:#a7f3d0;letter-spacing:3px;
                            text-transform:uppercase;margin-bottom:2px;">
                    SIGA GROUP · Modal Analysis Edition
                </div>
                <div style="font-size:28px;font-weight:800;
                            letter-spacing:-0.4px;line-height:1.1;">
                    Watermelon Planta
                </div>
                <div style="font-size:13px;opacity:0.85;margin-top:6px;
                            font-weight:400;">
                    Captura modal offline · Sistema Watermelon de alta precisión · ISO 7626 / 20816
                </div>
            </div>
            <!-- Chip version a la derecha -->
            <div style="background:rgba(255,255,255,0.15);
                        border:1px solid rgba(255,255,255,0.25);
                        border-radius:8px;padding:6px 12px;
                        font-size:11px;font-weight:600;
                        letter-spacing:1px;white-space:nowrap;
                        backdrop-filter:blur(8px);">
                {_app_version}
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =====================================================================
# Chip de status de licencia (v3.31.215 — FASE D)
# =====================================================================
render_license_status_chip(_LIC)
if _LIC.expires_soon:
    st.warning(
        f"⚠ **Tu licencia vence en {_LIC.days_until_expiry} días** "
        f"({_LIC.expires_at.strftime('%d/%m/%Y')}). "
        f"Contacta a ehernandez@sigasas.com para renovar y evitar "
        f"interrupciones en el servicio."
    )

# =====================================================================
# Auto-update checker (v3.31.216 — FASE F)
# Chequea GitHub Releases una vez por sesión + cache 24h.
# Silencioso si no hay internet (Planta es offline-first).
# =====================================================================
try:
    from updater import (
        get_cached_check, render_update_banner, run_auto_update_ui,
    )
    _UPDATE_INFO = get_cached_check(_app_version)
    render_update_banner(_UPDATE_INFO)
    # v3.31.398 — AUTO-UPDATE: si hay versión nueva y corremos como .exe,
    # botón que descarga el installer y lo corre /VERYSILENT (la app se
    # cierra, se actualiza y se reabre sola — sin reinstalar a mano).
    # Con data/.auto_update_on_start.flag lo hace SOLO al detectar internet.
    run_auto_update_ui(_UPDATE_INFO)
except ImportError:
    _UPDATE_INFO = None  # updater es opcional, no romper si falta

# =====================================================================
# Welcome Onboarding (v3.31.214 — FASE E3)
# Se muestra SOLO la primera vez. Después se esconde con flag local.
# =====================================================================
_ONBOARD_FLAG = Path(__file__).parent / "data" / ".onboarded.flag"

if not _ONBOARD_FLAG.exists():
    # v3.31.256 — Onboarding industrial/sobrio (antes amber juguetón).
    # Estilo coherente con el header SIGA (navy/teal) en vez de naranja.
    st.markdown("""
    <div style="background:linear-gradient(135deg,#f8fafc 0%,#f0fdfa 100%);
                border:1px solid rgba(15,118,110,0.18);border-radius:14px;
                padding:24px 28px;margin-bottom:20px;
                box-shadow:0 2px 8px rgba(15,23,42,0.05);">
        <div style="display:flex;align-items:center;gap:14px;
                    margin-bottom:14px;">
            <div style="width:42px;height:42px;border-radius:50%;
                        background:linear-gradient(135deg,#0f766e 0%,#0d9488 100%);
                        display:flex;align-items:center;justify-content:center;
                        color:white;font-size:20px;flex-shrink:0;">✦</div>
            <div>
                <div style="font-size:11px;font-weight:700;letter-spacing:1.5px;
                            text-transform:uppercase;color:#0f766e;">
                    Primera ejecución
                </div>
                <div style="font-size:17px;font-weight:700;color:#0f172a;
                            margin-top:2px;">
                    Bienvenido a Watermelon Planta Edition
                </div>
            </div>
        </div>
        <div style="color:#475569;font-size:13.5px;line-height:1.55;
                    margin-bottom:16px;">
            Tres pasos para empezar a capturar análisis modal:
        </div>
        <div style="display:grid;grid-template-columns:repeat(3,1fr);
                    gap:12px;">
            <div style="background:white;border:1px solid #e2e8f0;
                        border-radius:10px;padding:14px;">
                <div style="display:flex;align-items:center;gap:8px;
                            margin-bottom:8px;">
                    <div style="width:22px;height:22px;border-radius:50%;
                                background:#0f766e;color:white;font-size:11px;
                                font-weight:700;display:flex;
                                align-items:center;justify-content:center;">1</div>
                    <div style="font-weight:700;color:#0f172a;font-size:13.5px;">
                        Conecta el equipo
                    </div>
                </div>
                <div style="font-size:12px;color:#64748b;line-height:1.45;">
                    Conecta el equipo Watermelon al puerto USB y espera a
                    que los indicadores enciendan.
                </div>
            </div>
            <div style="background:white;border:1px solid #e2e8f0;
                        border-radius:10px;padding:14px;">
                <div style="display:flex;align-items:center;gap:8px;
                            margin-bottom:8px;">
                    <div style="width:22px;height:22px;border-radius:50%;
                                background:#0f766e;color:white;font-size:11px;
                                font-weight:700;display:flex;
                                align-items:center;justify-content:center;">2</div>
                    <div style="font-weight:700;color:#0f172a;font-size:13.5px;">
                        Captura tu ensayo
                    </div>
                </div>
                <div style="font-size:12px;color:#64748b;line-height:1.45;">
                    Selecciona modo EMA u OMA, configura los canales y
                    dispara la adquisición.
                </div>
            </div>
            <div style="background:white;border:1px solid #e2e8f0;
                        border-radius:10px;padding:14px;">
                <div style="display:flex;align-items:center;gap:8px;
                            margin-bottom:8px;">
                    <div style="width:22px;height:22px;border-radius:50%;
                                background:#0f766e;color:white;font-size:11px;
                                font-weight:700;display:flex;
                                align-items:center;justify-content:center;">3</div>
                    <div style="font-weight:700;color:#0f172a;font-size:13.5px;">
                        Sincroniza al Cloud
                    </div>
                </div>
                <div style="font-size:12px;color:#64748b;line-height:1.45;">
                    Cuando tengas internet, autentícate y sube las capturas
                    para procesarlas en Watermelon Cloud.
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    _onb_col1, _onb_col2 = st.columns([3, 1])
    with _onb_col2:
        if st.button("Entendido", type="primary",
                       use_container_width=True, key="dismiss_onboard"):
            _ONBOARD_FLAG.parent.mkdir(parents=True, exist_ok=True)
            _ONBOARD_FLAG.write_text("dismissed")
            st.rerun()
    with _onb_col1:
        st.caption(
            "Este mensaje no volverá a aparecer. Para soporte: ehernandez@sigasas.com"
        )
    st.divider()

# Estado del sistema
col1, col2, col3 = st.columns(3)
with col1:
    # Test internet con timeout corto
    import socket
    def _has_internet(timeout=1.5):
        try:
            socket.setdefaulttimeout(timeout)
            socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(
                ("8.8.8.8", 53)
            )
            return True
        except OSError:
            return False
    online = _has_internet()
    if online:
        st.success("🌐 **Online** · puedes sincronizar capturas al Cloud")
    else:
        st.warning("📴 **Offline** · capturas se guardan local, sync cuando vuelva la red")

with col2:
    try:
        from core.modal.acq_backend import discover_acq_modules
        modules = discover_acq_modules("cDAQ1")
        if modules:
            st.success(
                f"✓ **Equipo Watermelon conectado** "
                f"→ {len(modules)*4} canales disponibles"
            )
        else:
            st.warning("⚠ Equipo Watermelon no detectado — verifica USB")
    except ImportError:
        # v3.31.256 — Sanitizado. Antes decía "INSTALAR.bat" (concepto
        # interno) y exponía info que asustaba al cliente final.
        st.warning(
            "⚠ Componente de adquisición pendiente — reinstala el instalador "
            "completo más reciente"
        )
    except Exception:
        st.warning("⚠ No se pudo detectar el equipo. Verifica USB y reinicia.")

with col3:
    n_captures = len(list(_CAPTURES_DIR.glob("*.tdms")))
    st.info(f"📁 **{n_captures} captura(s)** guardada(s) en `{_CAPTURES_DIR.name}/`")

st.divider()

# =====================================================================
# Sidebar industrial (v3.31.217 — FASE H: clase mundial sin hardware leak)
# Estilo System1/AMS — branding + status sin nombres de libs/marcas
# =====================================================================

# Pre-computar status del sistema DAQ (silencioso, sin nombres de libs)
_daq_status = "down"
try:
    from core.modal.acq_backend import discover_acq_modules as _disc
    _mods = _disc("cDAQ1")
    _daq_status = "ok" if _mods else "disconnected"
    _daq_channels = len(_mods) * 4 if _mods else 0
except ImportError:
    _daq_status = "no_drivers"
    _daq_channels = 0
except Exception:
    _daq_status = "error"
    _daq_channels = 0

# Mapa de status a (label, color, dot)
_DAQ_STATUS = {
    "ok":           ("Operativo",            "#10b981", "●"),
    "disconnected": ("Equipo no conectado",  "#f59e0b", "●"),
    "no_drivers":   ("Componente pendiente", "#f59e0b", "●"),
    "error":        ("Reinicia el equipo",   "#f59e0b", "●"),
    "down":         ("No disponible",        "#94a3b8", "○"),
}
_lbl, _col, _dot = _DAQ_STATUS[_daq_status]

_n_capturas = len(list(_CAPTURES_DIR.glob('*.tdms')))
# Animación pulsante solo si el sistema está en falla — atrae la atención
_dot_class = "wm-pulse-dot" if _daq_status in ("no_drivers", "error") else ""

# Mini detalle textual del status (segunda línea bajo el label)
_DAQ_HINT = {
    "ok":           f"Listo · {_daq_channels} canales",
    "disconnected": "Conecta por USB",
    "no_drivers":   "Reinstala el instalador completo",
    "error":        "Reinicia el equipo",
    "down":         "—",
}
_hint = _DAQ_HINT[_daq_status]

with st.sidebar:
    # ================================================================
    # 1. BRANDING HEADER — dark hero card con logo + identidad
    # ================================================================
    st.markdown(
        """
        <div style="background:linear-gradient(135deg,#0f172a 0%,#1e3a8a 50%,#0f766e 100%);
                    border-radius:12px;padding:16px;margin:-8px 0 18px 0;
                    box-shadow:0 4px 14px rgba(15,118,110,0.18),
                               inset 0 1px 0 rgba(255,255,255,0.1);
                    position:relative;overflow:hidden;">
            <!-- glow decorativo arriba derecha -->
            <div style="position:absolute;top:-30px;right:-30px;width:120px;height:120px;
                        border-radius:50%;
                        background:radial-gradient(circle,
                            rgba(255,255,255,0.15) 0%,
                            rgba(255,255,255,0) 70%);
                        pointer-events:none;"></div>
            <div style="display:flex;align-items:center;gap:11px;position:relative;">
                <svg width="42" height="42" viewBox="0 0 256 256"
                     xmlns="http://www.w3.org/2000/svg"
                     style="flex-shrink:0;
                            filter:drop-shadow(0 2px 4px rgba(0,0,0,0.3));">
                    <defs>
                        <linearGradient id="sbRind" x1="0%" y1="100%" x2="0%" y2="0%">
                            <stop offset="0%" stop-color="#14532d"/>
                            <stop offset="100%" stop-color="#4ade80"/>
                        </linearGradient>
                        <linearGradient id="sbMeat" x1="0%" y1="100%" x2="0%" y2="0%">
                            <stop offset="0%" stop-color="#9f1239"/>
                            <stop offset="100%" stop-color="#fb7185"/>
                        </linearGradient>
                    </defs>
                    <circle cx="128" cy="128" r="120" fill="rgba(255,255,255,0.15)"/>
                    <g transform="translate(128 142)">
                        <path d="M -78 0 A 78 78 0 0 0 78 0 L 71 0 A 71 71 0 0 1 -71 0 Z"
                              fill="url(#sbRind)"/>
                        <path d="M -71 0 A 71 71 0 0 0 71 0 L 65 0 A 65 65 0 0 1 -65 0 Z"
                              fill="#bbf7d0"/>
                        <path d="M -65 0 A 65 65 0 0 0 65 0 Z" fill="url(#sbMeat)"/>
                        <ellipse cx="-32" cy="20" rx="3.5" ry="6" fill="#1f2937"
                                 transform="rotate(-15 -32 20)"/>
                        <ellipse cx="-10" cy="32" rx="3.5" ry="6" fill="#1f2937"/>
                        <ellipse cx="14"  cy="32" rx="3.5" ry="6" fill="#1f2937"
                                 transform="rotate(8 14 32)"/>
                        <ellipse cx="34"  cy="20" rx="3.5" ry="6" fill="#1f2937"
                                 transform="rotate(15 34 20)"/>
                    </g>
                </svg>
                <div style="flex:1;min-width:0;">
                    <div style="font-size:15px;font-weight:800;color:#ffffff;
                                letter-spacing:0.2px;line-height:1.1;">
                        Watermelon
                    </div>
                    <div style="font-size:9px;font-weight:700;color:#a7f3d0;
                                letter-spacing:2.5px;text-transform:uppercase;
                                margin-top:3px;">
                        Planta · Modal
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ================================================================
    # 2. ESTADO DEL SISTEMA — health panel con dot pulsante
    # ================================================================
    st.markdown("##### Estado del sistema")

    st.markdown(
        f"""
        <div style="background:#ffffff;border-radius:10px;padding:14px;
                    border:1px solid {_col}33;
                    box-shadow:0 1px 3px rgba(0,0,0,0.04);
                    margin-bottom:10px;position:relative;overflow:hidden;">
            <div style="display:flex;align-items:flex-start;justify-content:space-between;
                        gap:8px;">
                <div style="flex:1;min-width:0;">
                    <div style="font-size:9px;color:#94a3b8;font-weight:700;
                                text-transform:uppercase;letter-spacing:1.5px;
                                margin-bottom:3px;">
                        Maleta Watermelon
                    </div>
                    <div style="font-size:14px;color:#0f172a;font-weight:700;
                                line-height:1.2;">
                        {_lbl}
                    </div>
                    <div style="font-size:11px;color:#64748b;margin-top:2px;">
                        {_hint}
                    </div>
                </div>
                <div style="color:{_col};font-size:22px;line-height:1;flex-shrink:0;
                            margin-top:2px;" class="{_dot_class}">
                    {_dot}
                </div>
            </div>
        </div>

        <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;
                    margin-bottom:14px;">
            <div style="background:#ffffff;border-radius:8px;padding:10px;
                        border:1px solid rgba(15,118,110,0.10);
                        text-align:center;">
                <div style="font-size:9px;color:#94a3b8;font-weight:700;
                            text-transform:uppercase;letter-spacing:1.5px;">
                    Canales
                </div>
                <div style="font-size:20px;font-weight:800;color:#0f766e;
                            line-height:1.1;margin-top:4px;">
                    {_daq_channels if _daq_channels else '—'}
                </div>
            </div>
            <div style="background:#ffffff;border-radius:8px;padding:10px;
                        border:1px solid rgba(15,118,110,0.10);
                        text-align:center;">
                <div style="font-size:9px;color:#94a3b8;font-weight:700;
                            text-transform:uppercase;letter-spacing:1.5px;">
                    Capturas
                </div>
                <div style="font-size:20px;font-weight:800;color:#0f766e;
                            line-height:1.1;margin-top:4px;">
                    {_n_capturas}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # ================================================================
    # 3. ACCESO RÁPIDO — cards Artemis con icono SVG + label + descripción
    # ================================================================
    st.markdown("##### Acceso rápido")

    _qa1, _qa2 = st.columns(2)
    with _qa1:
        if _LIC.has_module("ema"):
            if st.button("🔨  EMA", key="qa_ema",
                          use_container_width=True,
                          help="Captura sincronizada con martillo modal · ISO 7626-5"):
                st.switch_page("pages/01_Captura_Modal.py")
        else:
            st.button("🔒  EMA", key="qa_ema_lock",
                      use_container_width=True, disabled=True,
                      help="No incluido en tu plan — contacta a SIGA para upgrade")
    with _qa2:
        if _LIC.has_module("oma"):
            if st.button("🌊  OMA", key="qa_oma",
                          use_container_width=True,
                          help="Captura continua bajo operación · ISO 20816"):
                st.session_state["_planta_mode_preselect"] = "oma"
                st.switch_page("pages/01_Captura_Modal.py")
        else:
            st.button("🔒  OMA", key="qa_oma_lock",
                      use_container_width=True, disabled=True,
                      help="No incluido en tu plan — contacta a SIGA para upgrade")

    # Sub-cards descriptivos debajo de los botones — explican qué hace cada uno
    st.markdown(
        """
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;
                    margin-top:8px;margin-bottom:4px;">
            <div style="padding:8px 10px;background:rgba(15,118,110,0.04);
                        border-radius:6px;border-left:2px solid #0f766e;">
                <div style="font-size:9px;color:#0f766e;font-weight:700;
                            text-transform:uppercase;letter-spacing:1px;">
                    Impact Hammer
                </div>
                <div style="font-size:10px;color:#475569;line-height:1.4;
                            margin-top:2px;">
                    Máquina parada · ISO 7626-5
                </div>
            </div>
            <div style="padding:8px 10px;background:rgba(30,58,138,0.04);
                        border-radius:6px;border-left:2px solid #1e3a8a;">
                <div style="font-size:9px;color:#1e3a8a;font-weight:700;
                            text-transform:uppercase;letter-spacing:1px;">
                    Continuous
                </div>
                <div style="font-size:10px;color:#475569;line-height:1.4;
                            margin-top:2px;">
                    Máquina rotando · ISO 20816
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # ================================================================
    # 4. LICENCIA — premium card con badge del plan
    # ================================================================
    st.markdown("##### Licencia")
    _vence = (_LIC.expires_at.strftime('%d/%m/%Y')
               if _LIC.expires_at else '—')
    _modules_str = ' · '.join(m.upper() for m in _LIC.modules) \
        if _LIC.modules else '—'

    # Badge color según plan
    _PLAN_COLORS = {
        "enterprise": ("#eab308", "rgba(234,179,8,0.10)"),  # dorado
        "pro":        ("#0f766e", "rgba(15,118,110,0.10)"),
        "basic":      ("#475569", "rgba(71,85,105,0.10)"),
        "trial":      ("#3b82f6", "rgba(59,130,246,0.10)"),
    }
    _plan_col, _plan_bg = _PLAN_COLORS.get(_LIC.plan, ("#0f766e", "rgba(15,118,110,0.10)"))

    st.markdown(
        f"""
        <div style="background:#ffffff;border-radius:10px;padding:14px;
                    border:1px solid rgba(15,118,110,0.10);
                    box-shadow:0 1px 3px rgba(0,0,0,0.04);
                    margin-bottom:14px;position:relative;overflow:hidden;">
            <!-- Badge del plan en esquina top-right -->
            <div style="position:absolute;top:10px;right:10px;
                        background:{_plan_bg};color:{_plan_col};
                        padding:3px 8px;border-radius:5px;
                        font-size:9px;font-weight:800;letter-spacing:1.5px;
                        text-transform:uppercase;
                        border:1px solid {_plan_col}66;">
                {_LIC.plan.upper() if _LIC.plan else '—'}
            </div>
            <div style="font-size:9px;color:#94a3b8;font-weight:700;
                        text-transform:uppercase;letter-spacing:1.5px;
                        margin-bottom:3px;">
                Cliente
            </div>
            <div style="font-size:13px;color:#0f172a;font-weight:700;
                        line-height:1.2;margin-bottom:10px;
                        word-break:break-word;padding-right:60px;">
                {_LIC.customer or '—'}
            </div>
            <div style="display:flex;justify-content:space-between;align-items:center;
                        padding-top:10px;border-top:1px solid rgba(15,118,110,0.08);">
                <div>
                    <div style="font-size:9px;color:#94a3b8;font-weight:700;
                                text-transform:uppercase;letter-spacing:1.2px;">
                        Vence
                    </div>
                    <div style="font-size:12px;color:#0f172a;font-weight:600;">
                        {_vence}
                    </div>
                </div>
                <div style="text-align:right;">
                    <div style="font-size:9px;color:#94a3b8;font-weight:700;
                                text-transform:uppercase;letter-spacing:1.2px;">
                        Canales
                    </div>
                    <div style="font-size:12px;color:#0f172a;font-weight:600;">
                        hasta {_LIC.max_channels}
                    </div>
                </div>
            </div>
            <div style="margin-top:10px;padding-top:10px;
                        border-top:1px solid rgba(15,118,110,0.08);">
                <div style="font-size:9px;color:#94a3b8;font-weight:700;
                            text-transform:uppercase;letter-spacing:1.2px;
                            margin-bottom:4px;">
                    Módulos activos
                </div>
                <div style="font-size:10px;color:#0f766e;font-weight:600;
                            font-family:'SF Mono',Menlo,Consolas,monospace;
                            line-height:1.4;">
                    {_modules_str}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # ================================================================
    # 5. ACTUALIZACIONES — botón compacto + caption
    # ================================================================
    st.markdown("##### Actualizaciones")
    try:
        from updater import render_update_check_button
        render_update_check_button(_app_version)
    except ImportError:
        pass

    st.divider()

    # ================================================================
    # 6. FOOTER — version chip + soporte
    # ================================================================
    st.markdown(
        f"""
        <div style="padding:14px 12px;background:#ffffff;border-radius:10px;
                    border:1px solid rgba(15,118,110,0.10);
                    box-shadow:0 1px 3px rgba(0,0,0,0.04);">
            <div style="display:flex;align-items:center;gap:8px;
                        padding-bottom:10px;
                        border-bottom:1px solid rgba(15,118,110,0.08);
                        margin-bottom:10px;">
                <span style="display:inline-block;width:8px;height:8px;
                             border-radius:50%;background:#0f766e;
                             box-shadow:0 0 0 3px rgba(15,118,110,0.18);"></span>
                <span style="font-size:11px;color:#0f172a;font-weight:700;
                             font-family:'SF Mono',Menlo,Consolas,monospace;">
                    {_app_version}
                </span>
                <span style="font-size:9px;color:#94a3b8;font-weight:600;
                             text-transform:uppercase;letter-spacing:1px;
                             margin-left:auto;">
                    Estable
                </span>
            </div>
            <div style="font-size:10px;color:#475569;line-height:1.6;">
                <div style="margin-bottom:3px;font-weight:700;color:#0f172a;
                            font-size:11px;">
                    Soporte SIGA
                </div>
                <div>
                    <a href="mailto:ehernandez@sigasas.com"
                       style="color:#0f766e;text-decoration:none;">
                        ehernandez@sigasas.com
                    </a>
                </div>
                <div>
                    <a href="https://watermelonsys.net" target="_blank"
                       style="color:#0f766e;text-decoration:none;">
                        watermelonsys.net
                    </a>
                </div>
            </div>
            <div style="margin-top:10px;padding-top:10px;
                        border-top:1px solid rgba(15,118,110,0.08);
                        font-size:9px;color:#94a3b8;text-align:center;
                        letter-spacing:0.5px;">
                © 2026 SIGA GROUP S.A.S
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Navegación principal
st.markdown("## 📥 Selecciona tipo de captura")
st.caption(
    "Elige el tipo de ensayo que vas a hacer. Cada uno tiene su flujo "
    "validado contra normas internacionales."
)

cap_col1, cap_col2 = st.columns(2)

with cap_col1:
    st.markdown(
        """
        <div style="border:2px solid #0f766e;border-radius:10px;padding:20px;
                    background:#f0fdfa;height:200px;">
            <div style="font-size:32px;">🔨</div>
            <div style="font-weight:700;font-size:18px;margin-top:8px;">
                EMA · Impact Hammer
            </div>
            <div style="font-size:13px;color:#475569;margin-top:6px;">
                Captura sincronizada de impacto + respuesta. Para ensayo
                en máquina parada con martillo modal. Conforme ISO 7626-5.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if _LIC.has_module("ema"):
        if st.button("▶ Iniciar captura EMA", type="primary",
                      use_container_width=True, key="goto_ema"):
            st.switch_page("pages/01_Captura_Modal.py")
    else:
        st.button(
            "🔒 EMA no incluido en tu plan",
            disabled=True, use_container_width=True, key="ema_locked",
            help="Tu licencia no incluye EMA. Contacta a ehernandez@sigasas.com "
                 "para upgrade.",
        )

with cap_col2:
    st.markdown(
        """
        <div style="border:2px solid #1e3a8a;border-radius:10px;padding:20px;
                    background:#eff6ff;height:200px;">
            <div style="font-size:32px;">🌊</div>
            <div style="font-weight:700;font-size:18px;margin-top:8px;">
                OMA · Continuous
            </div>
            <div style="font-size:13px;color:#475569;margin-top:6px;">
                Captura continua bajo condiciones operacionales. Para
                máquina rotando. Streaming hasta 32 canales.
                Conforme ISO 20816 / Brincker & Ventura 2015.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if _LIC.has_module("oma"):
        if st.button("▶ Iniciar captura OMA", type="primary",
                      use_container_width=True, key="goto_oma"):
            st.session_state["_planta_mode_preselect"] = "oma"
            st.switch_page("pages/01_Captura_Modal.py")
    else:
        st.button(
            "🔒 OMA no incluido en tu plan",
            disabled=True, use_container_width=True, key="oma_locked",
            help="Tu licencia no incluye OMA. Contacta a ehernandez@sigasas.com "
                 "para upgrade a plan Pro o Enterprise.",
        )

st.divider()

# ------------------------------------------------------------------
# Sección SYNC con Watermelon Cloud (v3.31.209 — FASE B)
# ------------------------------------------------------------------
st.markdown("## ☁ Sincronización con Watermelon Cloud")

try:
    from auth_planta import (
        current_user, logout, request_login_code, verify_login_code,
    )
    from sync_uploader import (
        list_pending, list_uploaded, sync_all, get_sync_stats,
    )
    _SYNC_AVAILABLE = True
except ImportError as _exc:
    _SYNC_AVAILABLE = False
    st.warning(
        f"⚠ Módulo de sync no disponible: {_exc}\n\n"
        f"Si querés sincronizar, instala supabase con: `pip install supabase`"
    )

if _SYNC_AVAILABLE:
    _user = current_user()
    _stats = get_sync_stats(_CAPTURES_DIR)

    if _user is None:
        # No logueado — login por CÓDIGO OTP (v3.31.398, igual que la app
        # Cloud: ya no hay passwords). Paso 1: email → enviar código.
        # Paso 2: ingresar el código de 6 dígitos que llega al correo.
        with st.expander("🔐 Inicia sesión para sincronizar capturas al Cloud",
                          expanded=False):
            st.caption(
                "Una sola vez (necesitas internet ahora). Te enviamos un "
                "código de 6 dígitos a tu correo — sin contraseña. Tu sesión "
                "queda guardada y se renueva automáticamente."
            )
            _otp_email = st.session_state.get("_planta_otp_email", "")
            if not _otp_email:
                with st.form("planta_otp_request"):
                    _email = st.text_input("Email", placeholder="tu@email.com")
                    if st.form_submit_button("📧 Enviarme el código",
                                             type="primary"):
                        if not _email or "@" not in _email:
                            st.error("Escribe un email válido.")
                        else:
                            try:
                                request_login_code(_email)
                                st.session_state["_planta_otp_email"] = \
                                    _email.strip().lower()
                                st.rerun()
                            except (RuntimeError, ValueError) as exc:
                                st.error(str(exc))
            else:
                st.info(f"📧 Código enviado a **{_otp_email}** — revisa tu "
                        f"correo (también spam).")
                with st.form("planta_otp_verify"):
                    _code = st.text_input(
                        "Código de 6 dígitos", max_chars=8,
                        placeholder="123456")
                    _cv1, _cv2 = st.columns(2)
                    _do_verify = _cv1.form_submit_button(
                        "✓ Verificar e iniciar sesión", type="primary")
                    _do_back = _cv2.form_submit_button("← Cambiar email")
                if _do_verify:
                    try:
                        verify_login_code(_otp_email, _code)
                        st.session_state.pop("_planta_otp_email", None)
                        st.success("✓ Logueado. Recargando...")
                        st.rerun()
                    except (RuntimeError, ValueError) as exc:
                        st.error(str(exc))
                if _do_back:
                    st.session_state.pop("_planta_otp_email", None)
                    st.rerun()

        if _stats["pending"] > 0:
            st.info(
                f"📥 Tienes **{_stats['pending']} capturas** "
                f"({_stats['pending_mb']:.1f} MB) esperando ser subidos. "
                f"Inicia sesión arriba para sincronizar."
            )
        else:
            st.caption("No hay capturas pendientes de sincronizar.")
    else:
        # Logueado — mostrar stats + botón sync
        _login_col, _logout_col = st.columns([5, 1])
        with _login_col:
            st.success(f"✓ Logueado como **{_user['email']}**")
        with _logout_col:
            if st.button("Cerrar sesión", key="logout_btn"):
                logout()
                st.rerun()

        _stat_col1, _stat_col2, _stat_col3 = st.columns(3)
        _stat_col1.metric("Pendientes", _stats["pending"])
        _stat_col2.metric("Ya en Cloud", _stats["uploaded"])
        _stat_col3.metric("MB pendientes", f"{_stats['pending_mb']:.1f}")

        if _stats["pending"] == 0:
            st.info("🎉 Todo sincronizado — no hay capturas pendientes")
        else:
            _pending_files = list_pending(_CAPTURES_DIR)
            with st.expander(f"Ver las {len(_pending_files)} capturas pendientes"):
                for p in _pending_files[:50]:
                    st.text(f"  · {p.name}  ({p.stat().st_size/(1024*1024):.2f} MB)")
                if len(_pending_files) > 50:
                    st.caption(f"...y {len(_pending_files) - 50} más")

            if st.button(
                f"🔄 **Sync ahora** — subir {_stats['pending']} capturas al Cloud",

                type="primary",
                use_container_width=True,
                key="sync_now_btn",
            ):
                _progress = st.progress(0.0, text="Iniciando sync...")
                _status_area = st.empty()

                def _on_file_done(idx, total, fname, msg):
                    _progress.progress(idx / max(total, 1),
                                        text=f"{idx}/{total} · {fname}")
                    with _status_area.container():
                        st.text(f"  {msg}")

                with st.spinner("Subiendo capturas al Cloud..."):
                    result = sync_all(
                        _CAPTURES_DIR,
                        _user["email"],
                        _user["access_token"],
                        on_file_done=_on_file_done,
                    )

                _progress.empty()
                if result["failed"] == 0:
                    st.success(
                        f"✓ **{result['uploaded']}/{result['total']} subidos** "
                        f"exitosamente al Cloud."
                    )
                    st.balloons()
                else:
                    st.warning(
                        f"⚠ {result['uploaded']} subidos · "
                        f"{result['failed']} fallaron"
                    )
                    with st.expander("Ver errores"):
                        for e in result["errors"]:
                            st.error(f"**{e['file']}**: {e['error']}")
                # Refrescar stats
                import time as _t
                _t.sleep(2)
                st.rerun()

st.divider()

# Lista de capturas previas
st.markdown("## 📂 Capturas previas en este equipo")
captures = sorted(_CAPTURES_DIR.glob("*.tdms"),
                   key=lambda p: p.stat().st_mtime, reverse=True)

if not captures:
    st.info(
        "Aún no hay capturas en este equipo. Las que generes con los botones "
        "de arriba aparecerán acá."
    )
else:
    from datetime import datetime
    # v3.31.388 — tabla en Markdown (ya NO st.dataframe) para no depender del
    # componente DataFrame en el .exe (el mismo chunk de CSS que rompía en
    # Captura Modal). Markdown se renderiza nativo, sin chunks lazy.
    _md = "| Archivo | Tamaño (MB) | Fecha |\n| --- | --- | --- |\n"
    for cap in captures[:20]:  # mostrar últimos 20
        stat = cap.stat()
        size_mb = stat.st_size / (1024 * 1024)
        ts = datetime.fromtimestamp(stat.st_mtime)
        _name = cap.name.replace("|", "\\|")
        _md += f"| {_name} | {size_mb:.2f} | {ts.strftime('%Y-%m-%d %H:%M:%S')} |\n"
    st.markdown(_md)
    st.caption(
        f"Mostrando {min(len(captures), 20)} de {len(captures)} capturas. "
        f"Todas en `{_CAPTURES_DIR}`."
    )

# =====================================================================
# Footer corporativo (v3.31.214 — FASE E branding)
# =====================================================================
st.markdown(
    f"""
    <div style="margin-top:60px;padding-top:24px;
                border-top:1px solid rgba(0,0,0,0.08);
                display:flex;align-items:center;justify-content:space-between;
                font-size:12px;color:#64748b;font-family:-apple-system,'Segoe UI',Inter,sans-serif;">
        <div style="display:flex;align-items:center;gap:10px;">
            <span style="display:inline-block;width:8px;height:8px;
                         border-radius:50%;background:#0f766e;"></span>
            <span><b style="color:#0f766e;">Watermelon Planta Edition</b>
            &middot; {_app_version}</span>
        </div>
        <div style="opacity:0.85;">
            © 2026 <b>SIGA GROUP S.A.S</b> &middot;
            Modal Analysis ISO 7626 / 20816 &middot;
            <a href="https://watermelonsys.net" target="_blank"
               style="color:#0f766e;text-decoration:none;">watermelonsys.net</a>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
