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

# Header con identidad de planta (visualmente distinto del Watermelon Cloud)
st.markdown(
    """
    <div style="background:linear-gradient(135deg,#1e3a8a 0%,#0f766e 100%);
                padding:24px 28px;border-radius:12px;color:white;
                margin-bottom:24px;box-shadow:0 4px 18px rgba(0,0,0,0.15);">
        <div style="display:flex;align-items:center;gap:14px;">
            <div style="font-size:36px;">🍉</div>
            <div>
                <div style="font-size:24px;font-weight:800;letter-spacing:-0.3px;">
                    Watermelon · Planta Edition
                </div>
                <div style="font-size:13px;opacity:0.85;margin-top:2px;">
                    Captura modal offline para hardware NI cDAQ-9178 + NI-9234
                </div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

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
