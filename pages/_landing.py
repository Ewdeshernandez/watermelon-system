"""
pages/_landing.py
=================

Página de aterrizaje (Home) — Ciclo 17.11 (Nivel 1+2) +
Ciclo 17.12 (Nivel 3: gauge + omnibox + modo turno).

De brochure → HMI de centro de control.

Layout (de arriba a abajo):

  ┌─ HERO COMPACTO ──────────────────────────────────────────┐
  │  Saludo personalizado + reloj turno + status flota        │
  └───────────────────────────────────────────────────────────┘

  ┌─ OMNIBOX (Cmd+K visual) ─────────────────────────────────┐
  │  🔍 Buscar instancia, reporte o norma...                  │
  │  [Resultados live debajo cuando hay query]                │
  └───────────────────────────────────────────────────────────┘

  ┌─ KPI BAND ────────────────────────────────────────────────┐
  │  Total · Danger · Warning · Healthy   (con sparkline 7d)  │
  └───────────────────────────────────────────────────────────┘

  ┌─ QUICK ACTIONS ──────────────────────────────────────────┐
  │  [📤 Cargar CSV] [📈 Trends] [📄 Reporte] [📚] [🔬 Diag]  │
  └───────────────────────────────────────────────────────────┘

  ┌─ ACTIVE ASSETS GRID ──────────┐  ┌─ ACTIVITY FEED ──────┐
  │  Card por instancia con GAUGE │  │  Últimas 10          │
  │  semicircular 0-100 (Bently)  │  │  acciones del usuario│
  └───────────────────────────────┘  └──────────────────────┘

  ┌─ SCADA STATUS BAR ───────────────────────────────────────┐
  │  ENV · version · vault · sync · sesión                    │
  └───────────────────────────────────────────────────────────┘

Modo turno: si la hora actual cae en turno noche (22-06), todo el
hero + footer se renderizan con un acento rojo en lugar del verde
watermelon (señal visual de "estás en modo guardia nocturna").

Filosofía: en un ciclo el operador entra, ve la flota en 2 segundos,
sabe a qué activo ir, y arranca a trabajar sin clics extra.
"""

from __future__ import annotations

from datetime import datetime

import streamlit as st

from core.auth import get_current_user, render_user_menu, require_login
from core.briefing import generate_and_save_briefing
from core.health_score import (
    compute_health_score,
    render_score_gauge,
)
from core.home_metrics import (
    activity_sparkline,
    compute_fleet_status,
    get_personalized_greeting,
    get_system_health,
    list_recent_activity,
    severity_sparkline,
)
from core.omnibox_search import kind_color, kind_label, omnibox_search
from core.ui.theme import apply_theme


st.set_page_config(
    page_title="Watermelon System",
    page_icon="🍉",
    layout="wide",
)

require_login()
render_user_menu()
apply_theme()

# =============================================================
# DATOS PARA EL HOME
# =============================================================
_user = get_current_user() or {}
_full_name = _user.get("full_name", "") or _user.get("username", "")

# Ciclo 17.24.5 — detectar timezone del browser del usuario.
# st.context.timezone (Streamlit 1.31+) devuelve un string IANA tipo
# "America/Bogota" basado en el browser. Si está disponible, calculamos
# el reloj en la zona del usuario. Si no (versión vieja o falla),
# fallback a la hora del server (UTC en Streamlit Cloud).
_user_tz = ""
try:
    _ctx = getattr(st, "context", None)
    if _ctx is not None:
        _tz_attr = getattr(_ctx, "timezone", None)
        if _tz_attr:
            _user_tz = str(_tz_attr).strip()
except Exception:
    _user_tz = ""

_greet = get_personalized_greeting(_full_name, tz_name=_user_tz)
# Ciclo 17.15 — activity feed filtrable por usuario
_my_email = (_user.get("email", "") or "").strip().lower()
_my_role  = (_user.get("role", "")  or "").strip().lower()

# Ciclo 23.131 — Scoping: si role=client, pasar email para filtrar
# fleet por asset_tags del cliente registrado en clients.json.
_fleet = compute_fleet_status(client_email=_my_email if _my_role == "client" else "")
_health = get_system_health()

# Toggle para admin/specialist: "Mi actividad" vs "Toda la actividad"
# Default: "toda" para admin (ven movimiento del equipo entero), "mía"
# para specialist (su trabajo). Client siempre ve solo la suya.
_default_scope = "all" if _my_role == "admin" else "mine"
_activity_scope = st.session_state.get("wm_activity_scope", _default_scope)

_activity = list_recent_activity(
    limit=10,
    viewer_email=_my_email,
    viewer_role=_my_role,
    owner_filter=_my_email if _activity_scope == "mine" else "",
)

# =============================================================
# MODO TURNO (Ciclo 17.12 — Nivel 3)
# =============================================================
# El "modo noche" se activa entre 22:00 y 06:00 y vira el accent
# del hero+footer del verde watermelon a un rojo guardia. No es
# dark mode pleno (eso requiere theme.toml de Streamlit), es una
# señal visual sutil para que el operador sepa que está en turno
# nocturno cuando entra al sistema.
_is_night = _greet["shift"] == "Turno noche"
_HERO_BG_DAY = (
    "radial-gradient(circle at 12% 10%, rgba(74,222,128,0.10) 0%, transparent 35%),"
    "radial-gradient(circle at 92% 90%, rgba(56,189,248,0.10) 0%, transparent 38%),"
    "linear-gradient(135deg, #0b1426 0%, #0f1d36 50%, #0b1426 100%)"
)
_HERO_BG_NIGHT = (
    "radial-gradient(circle at 12% 10%, rgba(239,68,68,0.14) 0%, transparent 35%),"
    "radial-gradient(circle at 92% 90%, rgba(168,85,247,0.10) 0%, transparent 38%),"
    "linear-gradient(135deg, #1a0a14 0%, #2a0e1f 50%, #1a0a14 100%)"
)
_ACCENT_COLOR = "#ef4444" if _is_night else "#10b981"
_HERO_BG = _HERO_BG_NIGHT if _is_night else _HERO_BG_DAY


# =============================================================
# ESTILOS GLOBALES
# =============================================================
st.markdown(
    """
    <style>
    .block-container {
        max-width: 1600px !important;
        padding-top: 1.0rem !important;
        padding-bottom: 2.0rem !important;
    }

    /* ───── HERO (v3.31.265 — minimalista enterprise) ───── */
    .wmh-hero {
        border-radius: 14px;
        padding: 22px 28px;
        margin-bottom: 20px;
        background: linear-gradient(135deg, #0b1426 0%, #0f1d36 100%);
        border: 1px solid rgba(148,163,184,0.12);
        border-left: 4px solid #10b981; /* acento color de estado */
        box-shadow: 0 4px 16px rgba(15,23,42,0.10);
        color: #f8fafc;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 28px;
        flex-wrap: wrap;
        position: relative;
    }
    /* ::before / ::after removidos en v3.31.265 — el scan animado y
       los radial-gradients sobrecargados se quitan para un look mas
       sobrio tipo System1/AMS. El border-left coloreado ya da el
       acento de estado sin necesidad de capas extra. */
    .wmh-hero-left { flex: 1 1 60%; min-width: 320px; }
    .wmh-hero-right { flex: 0 0 auto; text-align: right; min-width: 240px; }

    .wmh-pill {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 999px;
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.14);
        color: rgba(248,250,252,0.85);
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        margin-bottom: 10px;
    }
    .wmh-title {
        font-size: 36px;
        font-weight: 800;
        line-height: 1.05;
        letter-spacing: -0.02em;
        margin: 0 0 6px 0;
        background: linear-gradient(90deg, #f8fafc 0%, #cbd5e1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .wmh-greeting {
        font-size: 18px;
        font-weight: 600;
        color: #e2e8f0;
        margin: 4px 0 2px 0;
    }
    .wmh-status-line {
        font-size: 13px;
        color: rgba(226,232,240,0.78);
        font-weight: 500;
    }
    .wmh-status-line .dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin: 0 6px 0 4px;
        vertical-align: middle;
        animation: wmh-status-pulse 1.8s ease-in-out infinite;
    }
    /* Ciclo 23.38 — pulse animado en el status dot del hero.
       Si hay activos en atención/danger, transmite urgencia visual. */
    @keyframes wmh-status-pulse {
        0%, 100% { box-shadow: 0 0 0 0 currentColor; opacity: 1; }
        50%      { box-shadow: 0 0 0 6px rgba(255,255,255,0); opacity: 0.55; }
    }

    /* Ciclo 17.24 — Línea "Último reporte: hace 2h" debajo del status,
       tiempo relativo calculado en JS para mostrar respecto a la zona
       horaria del browser del usuario. */
    .wmh-last-report {
        margin-top: 10px;
        font-size: 12px;
        color: rgba(226,232,240,0.62);
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 10px;
        border-radius: 999px;
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.07);
    }
    .wmh-last-report .wmh-lr-icon { font-size: 13px; }

    /* Ciclo 17.24 — Countdown "próximo turno X en Yh Zmin" debajo de
       la fecha. JS lo refresca cada 30s. */
    .wmh-next-shift {
        margin-top: 6px;
        font-size: 11px;
        color: rgba(226,232,240,0.50);
        font-style: italic;
        letter-spacing: 0.01em;
    }

    .wmh-clock {
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        font-size: 38px;
        font-weight: 700;
        color: #f8fafc;
        line-height: 1;
        letter-spacing: -0.02em;
    }
    .wmh-shift {
        font-size: 12px;
        color: rgba(226,232,240,0.78);
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-top: 6px;
    }
    .wmh-date {
        font-size: 12px;
        color: rgba(226,232,240,0.55);
        margin-top: 2px;
    }

    /* ───── KPI BAND (v3.31.265 — flat / minimal) ─────
       Antes: cards con hover lift +3px, glow shadows, border-top 4px,
       value 44px. Look "dashboard hackathon".
       Ahora: cards flat sin lift, border-top 2px sutil, value 32px
       tabular-nums, hover solo cambia border-color. Look enterprise
       enterprise tipo Linear/Notion/AMS. */
    .wmh-kpi {
        background: #ffffff;
        border: 1px solid #e6ebf2;
        border-radius: 10px;
        padding: 16px 18px;
        box-shadow: none;
        height: 100%;
        transition: border-color 0.15s ease;
        position: relative;
    }
    .wmh-kpi:hover {
        border-color: #cbd5e1;
    }
    .wmh-kpi-label {
        font-size: 10px;
        font-weight: 800;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: #64748b;
        margin-bottom: 8px;
    }
    .wmh-kpi-value {
        font-size: 32px;
        font-weight: 800;
        line-height: 1;
        letter-spacing: -0.02em;
        color: #0f172a;
        font-variant-numeric: tabular-nums;
    }
    .wmh-kpi-sub {
        font-size: 11px;
        color: #94a3b8;
        margin-top: 6px;
        line-height: 1.4;
    }
    .wmh-kpi-spark {
        margin-top: 8px;
        opacity: 0.55; /* sparkline mas tenue, no compete con el value */
    }
    .wmh-kpi.danger  .wmh-kpi-value { color: #dc2626; }
    .wmh-kpi.warning .wmh-kpi-value { color: #d97706; }
    .wmh-kpi.healthy .wmh-kpi-value { color: #059669; }
    /* Border-top accent fino (2px) — discreto pero leible */
    .wmh-kpi.danger  { border-top: 2px solid #ef4444; }
    .wmh-kpi.warning { border-top: 2px solid #f59e0b; }
    .wmh-kpi.healthy { border-top: 2px solid #10b981; }
    .wmh-kpi.total   { border-top: 2px solid #0ea5e9; }

    /* Ciclo 17.24 — Botón "Ver →" debajo de cada card KPI.
       Estilo "link sutil": sin fondo ni borde, texto pequeño con
       color tenue. Hover acentúa el color. */
    div[data-testid="stButton"] > button[kind="secondary"][data-wm-kpi-link] {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        color: #64748b !important;
        font-size: 11px !important;
        font-weight: 600 !important;
        letter-spacing: 0.04em !important;
        padding: 6px 0 0 0 !important;
        margin: 0 !important;
        text-align: left !important;
        justify-content: flex-start !important;
    }
    div[data-testid="stButton"] > button[kind="secondary"][data-wm-kpi-link]:hover {
        color: #0ea5e9 !important;
        background: transparent !important;
    }

    /* ───── SECTION HEADERS ───── */
    .wmh-sec {
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: #475569;
        margin: 26px 0 12px 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .wmh-sec .bar {
        flex: 1;
        height: 1px;
        background: linear-gradient(90deg, #cbd5e1 0%, transparent 100%);
    }

    /* ───── ASSET CARDS ───── */
    .wmh-asset {
        background: #ffffff;
        border: 1px solid #e6ebf2;
        border-radius: 14px;
        padding: 14px 16px;
        height: 100%;
        transition: transform 0.12s ease, box-shadow 0.12s ease;
    }
    .wmh-asset:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 22px rgba(15,23,42,0.10);
    }
    .wmh-asset-head {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 6px;
    }
    .wmh-asset-tag {
        font-size: 16px;
        font-weight: 800;
        color: #0f172a;
        letter-spacing: -0.01em;
    }
    .wmh-asset-class {
        font-size: 11px;
        color: #64748b;
        margin-bottom: 8px;
        line-height: 1.4;
    }
    .wmh-asset-meta {
        font-size: 11px;
        color: #94a3b8;
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        margin-top: 6px;
    }
    .wmh-sev-pill {
        font-size: 10px;
        font-weight: 700;
        padding: 3px 9px;
        border-radius: 999px;
        letter-spacing: 0.06em;
    }
    .wmh-sev-pill.healthy { background: #d1fae5; color: #047857; }
    .wmh-sev-pill.warning { background: #fef3c7; color: #b45309; }
    .wmh-sev-pill.danger  { background: #fee2e2; color: #b91c1c; }
    .wmh-sev-pill.unknown { background: #f1f5f9; color: #475569; }

    /* ───── HEALTH SCORE GAUGE WRAPPER ───── */
    .wmh-gauge-wrap {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 2px;
        padding: 6px 0 0 0;
    }
    .wmh-gauge-band {
        font-size: 10px;
        font-weight: 800;
        letter-spacing: 0.10em;
        text-transform: uppercase;
    }
    .wmh-gauge-tip {
        font-size: 10px;
        color: #94a3b8;
        text-align: center;
        line-height: 1.3;
        max-width: 130px;
    }

    /* ───── OMNIBOX (Cmd+K visual) ───── Ciclo 23.38 — más prominente.
       Border + box-shadow más marcados, focus-within accent, Cmd+K
       badge a la derecha del input (estilo Linear/Notion). */
    .wmh-omni-wrap {
        background: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 14px;
        padding: 8px 14px;
        margin-bottom: 22px;
        box-shadow: 0 6px 18px rgba(15,23,42,0.07);
        transition: border-color 0.18s ease, box-shadow 0.18s ease;
        position: relative;
    }
    .wmh-omni-wrap:focus-within {
        border-color: #2563eb;
        box-shadow: 0 8px 24px rgba(37,99,235,0.14);
    }
    .wmh-omni-wrap::after {
        content: "⌘ K";
        position: absolute;
        right: 18px;
        top: 50%;
        transform: translateY(-50%);
        padding: 3px 8px;
        background: #f1f5f9;
        border: 1px solid #cbd5e1;
        border-radius: 6px;
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        font-size: 11px;
        font-weight: 700;
        color: #64748b;
        letter-spacing: 0.03em;
        pointer-events: none;
        line-height: 1;
    }
    .wmh-omni-wrap:focus-within::after {
        opacity: 0;
    }
    .wmh-omni-hint {
        font-size: 11px;
        color: #94a3b8;
        margin-top: 4px;
        margin-left: 2px;
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    }
    .wmh-omni-results {
        margin-top: 8px;
        max-height: 320px;
        overflow-y: auto;
        border-top: 1px solid #f1f5f9;
        padding-top: 6px;
    }
    .wmh-omni-hit {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 8px 8px;
        border-radius: 8px;
        font-size: 13px;
        cursor: pointer;
        transition: background 0.08s ease;
    }
    .wmh-omni-hit:hover { background: #f8fafc; }
    .wmh-omni-hit-icon { font-size: 16px; flex: 0 0 auto; }
    .wmh-omni-hit-body { flex: 1 1 auto; }
    .wmh-omni-hit-title { color: #0f172a; font-weight: 700; }
    .wmh-omni-hit-sub {
        color: #94a3b8; font-size: 11px;
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    }
    .wmh-omni-hit-kind {
        font-size: 10px;
        font-weight: 800;
        letter-spacing: 0.10em;
        padding: 2px 8px;
        border-radius: 999px;
        flex: 0 0 auto;
    }

    /* ───── ACTIVITY FEED ───── */
    .wmh-feed {
        background: #ffffff;
        border: 1px solid #e6ebf2;
        border-radius: 14px;
        padding: 14px 16px;
    }
    .wmh-feed-item {
        display: flex;
        gap: 10px;
        padding: 10px 0;
        border-bottom: 1px solid #f1f5f9;
        font-size: 13px;
    }
    .wmh-feed-item:last-child { border-bottom: none; }
    .wmh-feed-icon { flex: 0 0 auto; font-size: 16px; line-height: 1.4; }
    .wmh-feed-body { flex: 1 1 auto; }
    .wmh-feed-title { color: #0f172a; font-weight: 600; line-height: 1.3; }
    .wmh-feed-sub {
        color: #94a3b8; font-size: 11px;
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    }
    .wmh-feed-age {
        color: #64748b; font-size: 11px; margin-top: 2px;
    }

    /* ───── SCADA STATUS BAR (footer) ───── */
    .wmh-scada {
        margin-top: 26px;
        padding: 10px 16px;
        border-radius: 10px;
        background: #0f172a;
        color: #cbd5e1;
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        font-size: 11px;
        letter-spacing: 0.04em;
        display: flex;
        gap: 18px;
        flex-wrap: wrap;
        align-items: center;
        border: 1px solid rgba(148,163,184,0.18);
    }
    .wmh-scada .seg { display: inline-flex; align-items: center; gap: 6px; }
    .wmh-scada .seg b { color: #f8fafc; font-weight: 700; }
    .wmh-scada .seg .dot {
        display: inline-block;
        width: 8px; height: 8px;
        border-radius: 50%;
    }
    .wmh-scada .sep { color: #475569; }

    @media (max-width: 900px) {
        .wmh-title { font-size: 28px; }
        .wmh-clock { font-size: 30px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================================================
# CSS DINÁMICO — modo turno (Ciclo 17.12)
# =============================================================
# Override condicional del background del hero y del accent.
# Si _is_night, vira a tonos rojo/púrpura "guardia nocturna".
st.markdown(
    f"""
    <style>
    .wmh-hero {{ background: {_HERO_BG} !important;
                  border-left-color: {_ACCENT_COLOR} !important; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================
# HERO COMPACTO
# =============================================================
_total = _fleet["total"]
_dang = _fleet["by_severity"].get("danger", 0)
_warn = _fleet["by_severity"].get("warning", 0)
_heal = _fleet["by_severity"].get("healthy", 0)
_unk  = _fleet["by_severity"].get("unknown", 0)

# Línea de status: prioriza la peor noticia
if _dang > 0:
    _status_dot, _status_color, _status_text = "⬤", "#ef4444", f"{_dang} activo(s) en estado crítico"
elif _warn > 0:
    _status_dot, _status_color, _status_text = "⬤", "#f59e0b", f"{_warn} activo(s) requieren atención"
elif _total == 0:
    _status_dot, _status_color, _status_text = "⬤", "#94a3b8", "Sin activos cargados — empezá creando uno en Machinery Library"
else:
    _status_dot, _status_color, _status_text = "⬤", "#10b981", f"{_total} activo(s) monitoreado(s) · sin alertas críticas"


# Ciclo 17.24 — Último reporte archivado (para mostrar en el hero).
# Ciclo 17.24.1 hotfix: tiempo relativo calculado en Python (no JS) porque
# Streamlit no permite <script> dentro de st.markdown.
_last_report_line = ""
try:
    from core.reports_archive import list_archived_reports
    _archived = list_archived_reports(
        viewer_email=_user.get("email", "") or _user.get("username", ""),
        viewer_role=_user.get("role", "viewer"),
        limit=1,
    )
    if _archived:
        _last = _archived[0]
        _archived_at_iso = _last.get("archived_at", "") or ""
        _last_asset = _last.get("asset_name", "") or _last.get("client_name", "") or "reporte"
        # Calcular tiempo relativo en Python (server-side)
        _rel = ""
        try:
            _then = datetime.fromisoformat(_archived_at_iso.replace("Z", "+00:00"))
            if _then.tzinfo is None:
                # Asumir hora local del server si no hay TZ
                _diff = (datetime.now() - _then).total_seconds()
            else:
                from datetime import timezone as _tz
                _diff = (datetime.now(_tz.utc) - _then).total_seconds()
            _diff = max(0, _diff)
            if _diff < 60:
                _rel = "hace un instante"
            elif _diff < 3600:
                _rel = f"hace {int(_diff // 60)} min"
            elif _diff < 86400:
                _rel = f"hace {int(_diff // 3600)} h"
            elif _diff < 86400 * 7:
                _rel = f"hace {int(_diff // 86400)} días"
            else:
                _rel = _then.strftime("%d %b %Y").lower()
        except Exception:
            _rel = ""
        if _rel:
            # Ciclo 17.24.2 — pasamos también el ISO al frontend en
            # data-archived-at para que el JS separado pueda re-calcular
            # el "hace Xh" relativo a la zona del browser del usuario.
            # El _rel server-side queda como fallback inicial.
            _last_report_line = (
                f'<div class="wmh-last-report" '
                f'data-archived-at="{_archived_at_iso}" '
                f'data-asset="{_last_asset}">'
                f'<span class="wmh-lr-icon">📄</span> '
                f'<span class="wmh-lr-text">último reporte: {_last_asset} · {_rel}</span>'
                f'</div>'
            )
except Exception:
    pass


st.markdown(
    f"""
    <div class="wmh-hero"><div class="wmh-hero-left"><span class="wmh-pill">🍉 Watermelon · Industrial Vibration Intelligence</span><div class="wmh-greeting">{_greet['greeting']}</div><div class="wmh-status-line"><span class="dot" style="background:{_status_color};"></span>{_status_text}</div>{_last_report_line}</div><div class="wmh-hero-right"><div class="wmh-clock" id="wm-clock-live">{_greet['time_hhmm']}</div><div class="wmh-shift" id="wm-shift-live">{_greet['shift_emoji']} {_greet['shift']}</div><div class="wmh-date" id="wm-date-live">{_greet['date_long']}</div><div class="wmh-next-shift" id="wm-next-shift">&nbsp;</div></div></div>
    """,
    unsafe_allow_html=True,
)
# Ciclo 17.24.3 — IMPORTANTE: NO meter comentarios HTML <!-- ... --> con
# la palabra "<script>" adentro dentro del bloque markdown del hero.
# Streamlit los interpreta como código y rompe TODO el render del hero
# (lo muestra como bloque de código en pantalla, no como HTML). Los
# comentarios de explicación van como comentarios Python, fuera del
# st.markdown.


# =============================================================
# OMNIBOX (Ciclo 17.12 — Nivel 3) — REMOVIDO en v3.31.264
# =============================================================
# La barra de búsqueda global ocupaba demasiado espacio visual en el
# Home. Acá ya queda removida — el usuario navega via sidebar o KPIs.
# Si en algún momento volvemos a necesitar búsqueda global, se puede
# implementar como modal/dialog accesible con Cmd+K sin ocupar layout
# permanente.
_omni_q = ""  # placeholder para que el código siguiente no falle

# v3.31.264 — Search JS hotkey + results loop removidos (junto con la
# barra visible arriba). El home queda sin búsqueda global, foco en
# KPIs + activos + actividad reciente.


# =============================================================
# KPI BAND (4 cards con sparkline)
# =============================================================
def _spark_svg(values: list, color: str = "#0ea5e9", width: int = 140, height: int = 32) -> str:
    """Genera un sparkline SVG inline a partir de una lista de ints."""
    if not values:
        return ""
    n = len(values)
    vmax = max(values) if max(values) > 0 else 1
    # Coordenadas
    pts = []
    for i, v in enumerate(values):
        x = (i / max(n - 1, 1)) * (width - 4) + 2
        y = height - 2 - (v / vmax) * (height - 6)
        pts.append(f"{x:.1f},{y:.1f}")
    poly = " ".join(pts)
    # Área debajo
    area = f"M2,{height} L" + poly.replace(" ", " L") + f" L{width-2},{height} Z"
    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'xmlns="http://www.w3.org/2000/svg" style="display:block;">'
        f'<path d="{area}" fill="{color}" opacity="0.12"/>'
        f'<polyline points="{poly}" fill="none" stroke="{color}" '
        f'stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>'
        f'</svg>'
    )


_spark_total   = activity_sparkline(7)
_spark_dang    = severity_sparkline("danger", 7)
_spark_warn    = severity_sparkline("warning", 7)
_spark_heal    = severity_sparkline("healthy", 7)


def _kpi_card(klass: str, label: str, value: int, sub: str, spark_html: str) -> str:
    return f"""
    <div class="wmh-kpi {klass}">
        <div class="wmh-kpi-label">{label}</div>
        <div class="wmh-kpi-value">{value}</div>
        <div class="wmh-kpi-sub">{sub}</div>
        <div class="wmh-kpi-spark">{spark_html}</div>
    </div>
    """


k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(_kpi_card(
        "total", "Activos en flota", _total,
        "vault local · monitoreados",
        _spark_svg(_spark_total, "#0ea5e9"),
    ), unsafe_allow_html=True)
    # Ciclo 17.24 — card clickable: navega a Machinery Library
    if st.button("Ver flota →", key="kpi_btn_total", use_container_width=True):
        st.switch_page("pages/00_Machinery_Library.py")
with k2:
    st.markdown(_kpi_card(
        "danger", "Críticos", _dang,
        "requieren intervención",
        _spark_svg(_spark_dang, "#ef4444"),
    ), unsafe_allow_html=True)
    # Ciclo 17.32 — Diagnostics fue eliminado del producto. Redirigimos
    # al Machinery Library con filtro por severidad para que el usuario
    # pueda inspeccionar y atender los activos críticos desde ahí.
    if st.button("Ver críticos →", key="kpi_btn_danger", use_container_width=True):
        st.session_state["wm_lib_filter_status"] = "danger"
        st.switch_page("pages/00_Machinery_Library.py")
with k3:
    st.markdown(_kpi_card(
        "warning", "En atención", _warn,
        "vigilancia o config pendiente",
        _spark_svg(_spark_warn, "#f59e0b"),
    ), unsafe_allow_html=True)
    # Ciclo 17.32 — Diagnostics fue eliminado del producto. Redirigimos
    # al Machinery Library con filtro por severidad warning.
    if st.button("Ver en atención →", key="kpi_btn_warning", use_container_width=True):
        st.session_state["wm_lib_filter_status"] = "warning"
        st.switch_page("pages/00_Machinery_Library.py")
with k4:
    _heal_or_unk = _heal if _heal > 0 else _unk
    _heal_label = "Saludables" if _heal > 0 else "Sin clasificar"
    _heal_sub = "norma asignada · baseline OK" if _heal > 0 else "asignar norma en Library"
    st.markdown(_kpi_card(
        "healthy" if _heal > 0 else "total", _heal_label, _heal_or_unk,
        _heal_sub,
        _spark_svg(_spark_heal if _heal > 0 else _spark_total, "#10b981" if _heal > 0 else "#94a3b8"),
    ), unsafe_allow_html=True)
    # Ciclo 17.24 — card clickable: navega a Machinery Library
    _btn_label = "Ver saludables →" if _heal > 0 else "Asignar normas →"
    if st.button(_btn_label, key="kpi_btn_heal", use_container_width=True):
        if _heal > 0:
            st.session_state["wm_lib_filter_status"] = "healthy"
        else:
            st.session_state["wm_lib_filter_status"] = "unclassified"
        st.switch_page("pages/00_Machinery_Library.py")


# =============================================================
# QUICK ACTIONS — REMOVIDO en v3.31.264
# =============================================================
# Los 6 botones de acciones rápidas (Cargar CSV, Trends, Machinery
# Lib, AI Assistant, Reports, Briefing del día) eran redundantes con
# el sidebar que ya tiene todas esas opciones. Quitarlas hace el Home
# más minimalista, look enterprise (Baker / Emerson AMS).
#
# El briefing del día sigue accesible desde Reports. Si se necesita
# como atajo, agregar entrada en sidebar (no en el Home grid).


# =============================================================
# ACTIVE ASSETS GRID + ACTIVITY FEED
# =============================================================
left, right = st.columns([2, 1], gap="large")


with left:
    st.markdown(
        '<div class="wmh-sec">🛡️ Activos en monitoreo <div class="bar"></div></div>',
        unsafe_allow_html=True,
    )

    if _total == 0:
        st.info(
            "🔍 No hay instancias creadas todavía. "
            "Ve a **Machinery Library → Crear nueva instancia** para empezar."
        )
    else:
        # v3.31.265 — Tabla compacta minimalista (reemplaza grid de
        # cards 3×N con gauges grandes). Look enterprise tipo
        # System1/AMS — densidad alta, escaneable de un vistazo.
        # Health score se muestra como número + dot coloreado, sin gauge.
        st.markdown("""
        <style>
        .wmh-asset-table {
            background: white;
            border: 1px solid #e6ebf2;
            border-radius: 10px;
            overflow: hidden;
            margin-bottom: 8px;
        }
        .wmh-asset-row {
            display: grid;
            grid-template-columns: 1.6fr 1.6fr 0.7fr 1.1fr 1.1fr;
            gap: 14px;
            padding: 12px 18px;
            align-items: center;
            border-bottom: 1px solid #f1f5f9;
            transition: background 0.12s ease;
        }
        .wmh-asset-row:last-child { border-bottom: none; }
        .wmh-asset-row:hover { background: #f8fafc; }
        .wmh-asset-row.is-header {
            background: #f8fafc;
            font-size: 10px;
            font-weight: 800;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: #64748b;
            padding: 10px 18px;
        }
        .wmh-asset-row.is-header:hover { background: #f8fafc; }
        .wmh-row-tag {
            font-size: 14px;
            font-weight: 800;
            color: #0f172a;
            letter-spacing: -0.01em;
        }
        .wmh-row-loc {
            font-size: 11px;
            color: #94a3b8;
            font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
            margin-top: 2px;
        }
        .wmh-row-class {
            font-size: 12px;
            color: #1f2937;
            line-height: 1.35;
        }
        .wmh-row-meta {
            font-size: 11px;
            color: #94a3b8;
            margin-top: 2px;
        }
        .wmh-row-score {
            font-size: 18px;
            font-weight: 800;
            font-variant-numeric: tabular-nums;
            color: #0f172a;
            text-align: right;
        }
        .wmh-row-score-sub {
            font-size: 10px;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            text-align: right;
            margin-top: 1px;
        }
        .wmh-row-sev-pill {
            font-size: 10px;
            font-weight: 800;
            padding: 3px 10px;
            border-radius: 999px;
            letter-spacing: 0.06em;
            text-transform: uppercase;
        }
        </style>
        <div class="wmh-asset-table">
          <div class="wmh-asset-row is-header">
            <div>Activo</div>
            <div>Tren</div>
            <div style="text-align:right;">Health</div>
            <div>Estado</div>
            <div style="text-align:right;">Acción</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        instances = _fleet["instances"][:9]
        for inst in instances:
            sev_class = inst.severity
            asset_line = inst.asset_class or inst.profile_key or "—"
            loc_line = inst.location or "sin ubicación"

            # Health score (carga la instancia full)
            try:
                from core.instance_state import get_instance
                _full = get_instance(inst.instance_id)
                if _full is not None:
                    _hs_data = {
                        "tag": inst.tag,
                        "iso_norm_code": getattr(_full, "iso_norm_code", "") or "",
                        "last_balance_date": getattr(_full, "last_balance_date", "") or "",
                        "documents": getattr(_full, "documents", []) or [],
                        "setpoint_warning_override": getattr(_full, "setpoint_warning_override", 0.0) or 0.0,
                        "setpoint_danger_override": getattr(_full, "setpoint_danger_override", 0.0) or 0.0,
                        "override_justification": getattr(_full, "override_justification", "") or "",
                    }
                    _hs = compute_health_score(_hs_data)
                else:
                    _hs = compute_health_score({"tag": inst.tag, "iso_norm_code": "",
                                                 "last_balance_date": "", "documents": []})
            except Exception:
                _hs = compute_health_score({"tag": inst.tag, "iso_norm_code": "",
                                             "last_balance_date": "", "documents": []})

            # Fila con st.columns (necesario para botón interactivo)
            c1, c2, c3, c4, c5 = st.columns([1.6, 1.6, 0.7, 1.1, 1.1])
            with c1:
                st.markdown(
                    f'<div class="wmh-row-tag">{inst.severity_dot} {inst.tag}</div>'
                    f'<div class="wmh-row-loc">📍 {loc_line}</div>',
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown(
                    f'<div class="wmh-row-class">{asset_line}</div>'
                    f'<div class="wmh-row-meta">{inst.n_documents} docs · {inst.last_seen_human}</div>',
                    unsafe_allow_html=True,
                )
            with c3:
                st.markdown(
                    f'<div class="wmh-row-score" style="color:{_hs.color};">{_hs.score}</div>'
                    f'<div class="wmh-row-score-sub" style="color:{_hs.color};">{_hs.band_label}</div>',
                    unsafe_allow_html=True,
                )
            with c4:
                st.markdown(
                    f'<span class="wmh-row-sev-pill wmh-sev-pill {sev_class}">'
                    f'{inst.severity_label}</span>',
                    unsafe_allow_html=True,
                )
            with c5:
                if st.button("Abrir →", key=f"asset_open_{inst.instance_id}",
                             use_container_width=True):
                    st.session_state["wm_active_instance"] = inst.instance_id
                    try:
                        st.switch_page("pages/00_Machinery_Library.py")
                    except Exception:
                        pass

        if _fleet["total"] > 9:
            st.caption(f"+ {_fleet['total'] - 9} activos más en Machinery Library →")


with right:
    st.markdown(
        '<div class="wmh-sec">📰 Actividad reciente <div class="bar"></div></div>',
        unsafe_allow_html=True,
    )

    # Ciclo 17.15 — Toggle "Mi actividad / Toda la actividad" para
    # admin/specialist. Client siempre ve solo su actividad.
    if _my_role in ("admin", "specialist"):
        _scope_label = {"mine": "🙋 Solo mía", "all": "🌐 Toda la actividad"}
        _picked = st.radio(
            "Alcance",
            options=["mine", "all"],
            format_func=lambda s: _scope_label[s],
            index=0 if _activity_scope == "mine" else 1,
            horizontal=True,
            key="wm_activity_scope_radio",
            label_visibility="collapsed",
        )
        if _picked != _activity_scope:
            st.session_state["wm_activity_scope"] = _picked
            st.rerun()

    if not _activity:
        st.markdown(
            '<div class="wmh-feed">'
            '<div class="wmh-feed-item">'
            '<div class="wmh-feed-icon">💤</div>'
            '<div class="wmh-feed-body">'
            '<div class="wmh-feed-title">Sin actividad reciente</div>'
            '<div class="wmh-feed-age">Empezá cargando un CSV o creando una instancia.</div>'
            '</div></div></div>',
            unsafe_allow_html=True,
        )
    else:
        # Ciclo 17.15 — avatar de iniciales según owner_email del evento
        def _initials_avatar(email: str) -> str:
            if not email:
                return ""
            name = email.split("@")[0].strip()
            parts = name.replace(".", " ").replace("_", " ").replace("-", " ").split()
            if len(parts) >= 2 and parts[0] and parts[1]:
                ini = (parts[0][0] + parts[1][0]).upper()
            else:
                ini = name[:2].upper() if name else "?"
            # Color determinístico por hash del email
            h = sum(ord(c) for c in email) % 6
            colors = ["#0ea5e9", "#84cc16", "#a855f7", "#f59e0b", "#ec4899", "#14b8a6"]
            return (
                f'<span style="display:inline-flex;align-items:center;'
                f'justify-content:center;width:24px;height:24px;border-radius:999px;'
                f'background:{colors[h]};color:white;font-size:9px;font-weight:800;'
                f'letter-spacing:0;flex-shrink:0;margin-right:6px;'
                f'vertical-align:middle;">{ini}</span>'
            )

        feed_html = ['<div class="wmh-feed">']
        for ev in _activity:
            avatar_html = _initials_avatar(getattr(ev, "owner_email", "") or "")
            feed_html.append(
                f'<div class="wmh-feed-item">'
                f'<div class="wmh-feed-icon">{ev.icon}</div>'
                f'<div class="wmh-feed-body">'
                f'<div class="wmh-feed-title">{avatar_html}{ev.title}</div>'
                f'<div class="wmh-feed-sub">{ev.subtitle}</div>'
                f'<div class="wmh-feed-age">{ev.age_human}</div>'
                f'</div></div>'
            )
        feed_html.append('</div>')
        st.markdown("".join(feed_html), unsafe_allow_html=True)


# =============================================================
# SCADA STATUS BAR (footer)
# =============================================================
st.markdown(
    f"""
    <div class="wmh-scada">
        <span class="seg">
            <span class="dot" style="background:{_health['env_color']};"></span>
            <b>{_health['env'].upper()}</b>
        </span>
        <span class="sep">│</span>
        <span class="seg"><b>{_health['version']}</b> · {_health['commit']}</span>
        <span class="sep">│</span>
        <span class="seg">VAULT
            <span class="dot" style="background:{'#10b981' if _health['vault_status']=='OK' else '#94a3b8'};"></span>
            <b>{_health['vault_status']}</b> · {_health['vault_n']} activos
        </span>
        <span class="sep">│</span>
        <span class="seg">Última sync · <b>{_health['last_data_age']}</b></span>
        <span class="sep">│</span>
        <span class="seg">Sesión: <b>{_full_name or 'anónimo'}</b></span>
    </div>
    """,
    unsafe_allow_html=True,
)
