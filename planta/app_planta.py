"""
planta/app_planta.py — Watermelon Planta Edition · Entry Point
================================================================

Mini-app Streamlit standalone para captura modal en planta sin internet.
Diseñada para correr en laptop de campo con maleta NI cDAQ-9178 + NI-9234
conectada por USB.

NO requiere:
- Internet (todo el flujo es local)
- Supabase / login (sin auth)
- API externa (sin Anthropic, sin OpenAI)
- Acceso al repo completo de Watermelon (solo necesita esta carpeta y
  reusable de core/modal/)

SÍ requiere:
- Python 3.10+ instalado en el PC
- nidaqmx (driver NI-DAQmx instalado en sistema + pip)
- npTDMS, numpy, pandas, streamlit, plotly

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
</style>
""", unsafe_allow_html=True)

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
                    Captura modal offline · NI cDAQ-9178 + NI-9234 · ISO 7626 / 20816
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
# Welcome Onboarding (v3.31.214 — FASE E3)
# Se muestra SOLO la primera vez. Después se esconde con flag local.
# =====================================================================
_ONBOARD_FLAG = Path(__file__).parent / "data" / ".onboarded.flag"

if not _ONBOARD_FLAG.exists():
    st.markdown("""
    <div style="background:linear-gradient(135deg,#fef3c7 0%,#fde68a 100%);
                border:2px solid #f59e0b;border-radius:12px;
                padding:24px 28px;margin-bottom:24px;
                box-shadow:0 4px 14px rgba(245,158,11,0.18);">
        <div style="display:flex;align-items:center;gap:12px;
                    margin-bottom:12px;">
            <span style="font-size:28px;">👋</span>
            <span style="font-size:20px;font-weight:800;color:#92400e;">
                ¡Bienvenido a Watermelon Planta Edition!
            </span>
        </div>
        <div style="color:#78350f;font-size:14px;line-height:1.6;
                    margin-bottom:14px;">
            Esta es la <b>primera vez</b> que abres la app.
            Te dejamos 3 pasos breves para que empieces a capturar
            análisis modal en menos de 5 minutos.
        </div>
        <div style="display:grid;grid-template-columns:repeat(3,1fr);
                    gap:14px;margin-top:18px;">
            <div style="background:white;border-radius:10px;padding:14px;
                        box-shadow:0 1px 4px rgba(0,0,0,0.08);">
                <div style="font-size:24px;margin-bottom:6px;">🔌</div>
                <div style="font-weight:700;color:#1f2937;font-size:14px;">
                    1. Conecta la maleta
                </div>
                <div style="font-size:12px;color:#64748b;margin-top:4px;">
                    Conecta tu NI cDAQ-9178 al USB.
                    Espera que el LED de power esté verde.
                </div>
            </div>
            <div style="background:white;border-radius:10px;padding:14px;
                        box-shadow:0 1px 4px rgba(0,0,0,0.08);">
                <div style="font-size:24px;margin-bottom:6px;">🎙</div>
                <div style="font-weight:700;color:#1f2937;font-size:14px;">
                    2. Captura tu primer ensayo
                </div>
                <div style="font-size:12px;color:#64748b;margin-top:4px;">
                    Elige EMA o OMA abajo. Configura canales.
                    Click "Iniciar captura".
                </div>
            </div>
            <div style="background:white;border-radius:10px;padding:14px;
                        box-shadow:0 1px 4px rgba(0,0,0,0.08);">
                <div style="font-size:24px;margin-bottom:6px;">☁</div>
                <div style="font-weight:700;color:#1f2937;font-size:14px;">
                    3. Sync al Cloud (opcional)
                </div>
                <div style="font-size:12px;color:#64748b;margin-top:4px;">
                    Cuando tengas internet, login + "Sync ahora"
                    para procesar en Watermelon Cloud.
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    _onb_col1, _onb_col2 = st.columns([3, 1])
    with _onb_col2:
        if st.button("✓ Entendido", type="primary",
                       use_container_width=True, key="dismiss_onboard"):
            _ONBOARD_FLAG.parent.mkdir(parents=True, exist_ok=True)
            _ONBOARD_FLAG.write_text("dismissed")
            st.rerun()
    with _onb_col1:
        st.caption(
            "Este mensaje no se mostrará otra vez. Si necesitas ayuda "
            "después, consulta `README_PLANTA.txt` o contacta soporte@sigasas.com."
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
        st.success("🌐 **Online** · puedes sincronizar TDMS al Cloud")
    else:
        st.warning("📴 **Offline** · TDMS se guardan local, sync cuando vuelva la red")

with col2:
    try:
        from core.modal.ni_daq import discover_ni9234_modules
        modules = discover_ni9234_modules("cDAQ1")
        if modules:
            st.success(
                f"✓ **{len(modules)} módulo(s) NI-9234** detectado(s) "
                f"→ {len(modules)*4} canales disponibles"
            )
        else:
            st.error("✗ Sin maleta NI conectada — verifica USB")
    except ImportError:
        st.error("✗ nidaqmx no instalado — corre INSTALAR.bat")
    except Exception as exc:
        st.error(f"✗ Error discovery: {exc}")

with col3:
    n_captures = len(list(_CAPTURES_DIR.glob("*.tdms")))
    st.info(f"📁 **{n_captures} captura(s)** guardada(s) en `{_CAPTURES_DIR.name}/`")

st.divider()

# Sidebar con info técnica
with st.sidebar:
    st.markdown("### 🛠 Watermelon Planta")
    st.caption(f"Carpeta de capturas: `{_CAPTURES_DIR}`")
    st.divider()
    st.markdown("**Versiones**")
    try:
        ver_file = _REPO_ROOT / "VERSION"
        if ver_file.exists():
            st.code(f"Watermelon: {ver_file.read_text().strip()}", language="text")
    except Exception:
        pass
    try:
        import nidaqmx
        st.code(f"nidaqmx:    {nidaqmx.__version__}", language="text")
    except (ImportError, AttributeError):
        st.code("nidaqmx:    NO INSTALADO", language="text")
    try:
        import nptdms
        st.code(f"npTDMS:     {nptdms.__version__}", language="text")
    except (ImportError, AttributeError):
        st.code("npTDMS:     NO INSTALADO", language="text")
    st.divider()
    st.markdown("**Ayuda**")
    st.caption(
        "Sigue los pasos en el README_PLANTA.txt del USB. "
        "Para reportar problemas: ehernandez@sigasas.com"
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
    if st.button("▶ Iniciar captura EMA", type="primary",
                  use_container_width=True, key="goto_ema"):
        st.switch_page("pages/01_Captura_Modal.py")

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
                máquina rotando. Streaming TDMS hasta 32 canales.
                Conforme ISO 20816 / Brincker & Ventura 2015.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("▶ Iniciar captura OMA", type="primary",
                  use_container_width=True, key="goto_oma"):
        st.session_state["_planta_mode_preselect"] = "oma"
        st.switch_page("pages/01_Captura_Modal.py")

st.divider()

# ------------------------------------------------------------------
# Sección SYNC con Watermelon Cloud (v3.31.209 — FASE B)
# ------------------------------------------------------------------
st.markdown("## ☁ Sincronización con Watermelon Cloud")

try:
    from auth_planta import current_user, login, logout
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
        # No logueado — mostrar formulario de login
        with st.expander("🔐 Inicia sesión para sincronizar TDMS al Cloud",
                          expanded=False):
            st.caption(
                "Una sola vez (necesitas internet ahora). Tu sesión queda "
                "guardada y se renueva automáticamente cuando esté disponible."
            )
            with st.form("planta_login"):
                _email = st.text_input("Email", placeholder="tu@email.com")
                _pwd = st.text_input("Password", type="password")
                if st.form_submit_button("Iniciar sesión", type="primary"):
                    if not _email or not _pwd:
                        st.error("Email y password requeridos")
                    else:
                        try:
                            login(_email, _pwd)
                            st.success("✓ Logueado. Recargando...")
                            st.rerun()
                        except RuntimeError as exc:
                            st.error(f"Login falló: {exc}")

        if _stats["pending"] > 0:
            st.info(
                f"📥 Tienes **{_stats['pending']} TDMS** "
                f"({_stats['pending_mb']:.1f} MB) esperando ser subidos. "
                f"Inicia sesión arriba para sincronizar."
            )
        else:
            st.caption("No hay TDMS pendientes de sincronizar.")
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
            st.info("🎉 Todo sincronizado — no hay TDMS pendientes")
        else:
            _pending_files = list_pending(_CAPTURES_DIR)
            with st.expander(f"Ver los {len(_pending_files)} TDMS pendientes"):
                for p in _pending_files[:50]:
                    st.text(f"  · {p.name}  ({p.stat().st_size/(1024*1024):.2f} MB)")
                if len(_pending_files) > 50:
                    st.caption(f"...y {len(_pending_files) - 50} más")

            if st.button(
                f"🔄 **Sync ahora** — subir {_stats['pending']} TDMS al Cloud",
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

                with st.spinner("Subiendo TDMS al Cloud..."):
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
    import pandas as pd
    from datetime import datetime
    rows = []
    for cap in captures[:20]:  # mostrar últimos 20
        stat = cap.stat()
        size_mb = stat.st_size / (1024 * 1024)
        ts = datetime.fromtimestamp(stat.st_mtime)
        rows.append({
            "Archivo": cap.name,
            "Tamaño (MB)": f"{size_mb:.2f}",
            "Fecha": ts.strftime("%Y-%m-%d %H:%M:%S"),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
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
