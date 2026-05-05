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

_greet = get_personalized_greeting(_full_name)
_fleet = compute_fleet_status()
_health = get_system_health()
# Ciclo 17.15 — activity feed filtrable por usuario
_my_email = (_user.get("email", "") or "").strip().lower()
_my_role  = (_user.get("role", "")  or "").strip().lower()

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

    /* ───── HERO ───── */
    .wmh-hero {
        border-radius: 22px;
        padding: 26px 34px;
        margin-bottom: 22px;
        background:
            radial-gradient(circle at 12% 10%, rgba(74,222,128,0.10) 0%, transparent 35%),
            radial-gradient(circle at 92% 90%, rgba(248,113,113,0.10) 0%, transparent 38%),
            linear-gradient(135deg, #0b1426 0%, #0f1d36 50%, #0b1426 100%);
        border: 1px solid rgba(148,163,184,0.16);
        box-shadow: 0 18px 60px rgba(15,23,42,0.22);
        color: #f8fafc;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 28px;
        flex-wrap: wrap;
        position: relative;
    }
    .wmh-hero::before {
        content: "";
        position: absolute; left: 0; top: 0; bottom: 0;
        width: 4px; border-radius: 22px 0 0 22px;
        opacity: 0.85;
    }
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

    /* ───── KPI BAND ───── */
    .wmh-kpi {
        background: #ffffff;
        border: 1px solid #e6ebf2;
        border-radius: 14px;
        padding: 16px 18px;
        box-shadow: 0 2px 8px rgba(15,23,42,0.04);
        height: 100%;
    }
    .wmh-kpi-label {
        font-size: 10px;
        font-weight: 800;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: #64748b;
        margin-bottom: 8px;
    }
    .wmh-kpi-value {
        font-size: 36px;
        font-weight: 800;
        line-height: 1;
        letter-spacing: -0.02em;
        color: #0f172a;
    }
    .wmh-kpi-sub {
        font-size: 11px;
        color: #94a3b8;
        margin-top: 4px;
    }
    .wmh-kpi-spark { margin-top: 10px; }
    .wmh-kpi.danger  .wmh-kpi-value { color: #dc2626; }
    .wmh-kpi.warning .wmh-kpi-value { color: #d97706; }
    .wmh-kpi.healthy .wmh-kpi-value { color: #059669; }
    .wmh-kpi.danger  { border-top: 3px solid #ef4444; }
    .wmh-kpi.warning { border-top: 3px solid #f59e0b; }
    .wmh-kpi.healthy { border-top: 3px solid #10b981; }
    .wmh-kpi.total   { border-top: 3px solid #0ea5e9; }

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

    /* ───── OMNIBOX (Cmd+K visual) ───── */
    .wmh-omni-wrap {
        background: #ffffff;
        border: 1px solid #e6ebf2;
        border-radius: 14px;
        padding: 8px 14px 8px 14px;
        margin-bottom: 22px;
        box-shadow: 0 4px 14px rgba(15,23,42,0.05);
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
    .wmh-hero {{ background: {_HERO_BG} !important; }}
    .wmh-hero::before {{ background: {_ACCENT_COLOR}; }}
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
#
# El reloj LIVE se inyecta abajo en su propio st.markdown separado
# (igual al patrón del Cmd+K que funciona desde Ciclo 17.13). El script
# accede a window.parent.document porque Streamlit corre en iframe.

# Ciclo 17.24.2 — Reloj LIVE con auto-zona-horaria del browser.
# IMPORTANTE: este st.markdown está SEPARADO del hero (sigue el patrón
# del Cmd+K del Ciclo 17.13 que funciona desde hace meses). NO mezclar
# scripts con HTML grande en el mismo st.markdown — Streamlit lo
# sanitiza y rompe todo. Este bloque es chico y aislado, no se sanitiza.
#
# El script accede a window.parent.document porque Streamlit corre
# dentro de un iframe, mientras los elementos con id viven en el
# documento padre. Refresca cada 30 segundos.
st.markdown(
    """
    <script>
    (function() {
        function pad(n) { return String(n).padStart(2, '0'); }
        function fmtClock(d) {
            let h = d.getHours();
            const m = pad(d.getMinutes());
            const ampm = h >= 12 ? 'pm' : 'am';
            h = h % 12;
            if (h === 0) h = 12;
            return h + ':' + m + ' ' + ampm;
        }
        function shiftFor(h) {
            if (h >= 6 && h < 14)  return {emoji:'☀️',  text:'TURNO MAÑANA',  end:14, nextEmoji:'🌇', nextText:'tarde'};
            if (h >= 14 && h < 22) return {emoji:'🌇', text:'TURNO TARDE',   end:22, nextEmoji:'🌙', nextText:'noche'};
            return {emoji:'🌙', text:'TURNO NOCHE', end: (h >= 22 ? 30 : 6), nextEmoji:'☀️', nextText:'mañana'};
        }
        function dateLong(d) {
            const dias = ['domingo','lunes','martes','miércoles','jueves','viernes','sábado'];
            const meses = ['ene','feb','mar','abr','may','jun','jul','ago','sep','oct','nov','dic'];
            return dias[d.getDay()] + ' · ' + pad(d.getDate()) + ' ' + meses[d.getMonth()] + ' ' + d.getFullYear();
        }
        function nextShiftStr(d) {
            const h = d.getHours();
            const m = d.getMinutes();
            const s = shiftFor(h);
            const nowMins = h * 60 + m;
            let endMins = s.end * 60;
            if (endMins <= nowMins) endMins += 24 * 60;
            const diff = endMins - nowMins;
            const dh = Math.floor(diff / 60);
            const dm = diff % 60;
            const dur = dh > 0 ? (dh + 'h ' + dm + 'min') : (dm + ' min');
            return 'próximo turno ' + s.nextEmoji + ' ' + s.nextText + ' en ' + dur;
        }
        function relativeTime(isoStr) {
            if (!isoStr) return '';
            const then = new Date(isoStr);
            if (isNaN(then)) return '';
            const diffSec = Math.max(0, (Date.now() - then.getTime()) / 1000);
            if (diffSec < 60)        return 'hace un instante';
            if (diffSec < 3600)      return 'hace ' + Math.floor(diffSec / 60) + ' min';
            if (diffSec < 86400)     return 'hace ' + Math.floor(diffSec / 3600) + ' h';
            if (diffSec < 86400 * 7) return 'hace ' + Math.floor(diffSec / 86400) + ' días';
            const meses = ['ene','feb','mar','abr','may','jun','jul','ago','sep','oct','nov','dic'];
            return then.getDate() + ' ' + meses[then.getMonth()] + ' ' + then.getFullYear();
        }
        function getDoc() {
            // Streamlit corre en iframe; los IDs viven en el documento padre.
            try { return window.parent.document; } catch(e) { return document; }
        }
        function tick() {
            const doc = getDoc();
            const d = new Date();
            const h = d.getHours();
            const cl = doc.getElementById('wm-clock-live');
            const sh = doc.getElementById('wm-shift-live');
            const dt = doc.getElementById('wm-date-live');
            const ns = doc.getElementById('wm-next-shift');
            if (cl) cl.textContent = fmtClock(d);
            if (sh) {
                const s = shiftFor(h);
                sh.textContent = s.emoji + ' ' + s.text;
            }
            if (dt) dt.textContent = dateLong(d);
            if (ns) ns.textContent = nextShiftStr(d);
            // Último reporte: tiempo relativo
            const lr = doc.querySelector('.wmh-last-report');
            if (lr) {
                const at = lr.getAttribute('data-archived-at');
                const asset = lr.getAttribute('data-asset') || 'reporte';
                const rel = relativeTime(at);
                const txtEl = lr.querySelector('.wmh-lr-text');
                if (txtEl && rel) txtEl.textContent = 'último reporte: ' + asset + ' · ' + rel;
            }
        }
        // Re-engancha tras rerender de Streamlit (igual al patrón del Cmd+K)
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', tick);
        } else {
            tick();
        }
        setTimeout(tick, 800);
        setInterval(tick, 30000);
    })();
    </script>
    """,
    unsafe_allow_html=True,
)


# =============================================================
# OMNIBOX (Ciclo 17.12 — Nivel 3)
# =============================================================
# Búsqueda global fuzzy contra: instancias del Vault, drafts de
# reportes y normas ISO/API. Resultados live debajo.
# Si la query está vacía → no muestra nada (no satura el home).
st.markdown('<div class="wmh-omni-wrap">', unsafe_allow_html=True)
_omni_q = st.text_input(
    label="Búsqueda global",
    placeholder="🔍  Buscar activo, reporte o norma — ej. \"TES1\", \"ISO 20816\", \"684\", \"balanceo\"… (Cmd+K)",
    key="wmh_omnibox_q",
    label_visibility="collapsed",
)
st.markdown(
    '<div class="wmh-omni-hint">tip: tipea ≥2 caracteres · '
    'enter para buscar · click en un resultado para ir directo · '
    '<b>Cmd+K</b> (o Ctrl+K) para enfocar desde cualquier lugar</div>',
    unsafe_allow_html=True,
)

# Ciclo 17.13 — Cmd+K real: hotkey global que enfoca el omnibox
# sin necesidad de instalar streamlit-shortcuts. Inyecta JS inline
# que escucha keydown en el iframe parent y enfoca el text_input
# por su placeholder (que sirve de selector estable).
st.markdown(
    """
    <script>
    (function() {
        function attachOmniHotkey() {
            // Streamlit corre dentro de iframe; intentar window.parent también.
            const targets = [window];
            try { if (window.parent && window.parent !== window) targets.push(window.parent); } catch(e){}
            const focusOmni = function() {
                // Selector: input cuyo placeholder empiece con el lupa+texto
                let nodes = document;
                try { nodes = window.parent.document; } catch(e){}
                const el = nodes.querySelector('input[placeholder^="\\u{1F50D}"]');
                if (el) { el.focus(); el.select && el.select(); return true; }
                return false;
            };
            const handler = function(ev) {
                const isMac = navigator.platform.toUpperCase().indexOf('MAC') >= 0;
                const cmdK = (isMac ? ev.metaKey : ev.ctrlKey) && (ev.key === 'k' || ev.key === 'K');
                if (cmdK) {
                    ev.preventDefault();
                    ev.stopPropagation();
                    focusOmni();
                }
            };
            targets.forEach(function(t) {
                try { t.addEventListener('keydown', handler, true); } catch(e){}
            });
        }
        // Re-enganchar tras rerender de Streamlit
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', attachOmniHotkey);
        } else {
            attachOmniHotkey();
        }
        setTimeout(attachOmniHotkey, 800);
    })();
    </script>
    """,
    unsafe_allow_html=True,
)

if _omni_q and len(_omni_q.strip()) >= 2:
    _hits = omnibox_search(_omni_q, limit=8)
    if not _hits:
        st.markdown(
            '<div class="wmh-omni-results">'
            '<div style="padding:10px;color:#94a3b8;font-size:13px;">'
            'Sin resultados para esa búsqueda.</div></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div class="wmh-omni-results">', unsafe_allow_html=True)
        for i, h in enumerate(_hits):
            kcol = kind_color(h.kind)
            klabel = kind_label(h.kind).upper()
            cols = st.columns([0.07, 0.78, 0.15])
            with cols[0]:
                st.markdown(
                    f'<div style="font-size:18px;text-align:center;'
                    f'padding-top:6px;">{h.icon}</div>',
                    unsafe_allow_html=True,
                )
            with cols[1]:
                st.markdown(
                    f'<div style="line-height:1.25;padding-top:4px;">'
                    f'<div style="color:#0f172a;font-weight:700;font-size:13px;">{h.title}</div>'
                    f'<div style="color:#94a3b8;font-size:11px;'
                    f'font-family:ui-monospace,SFMono-Regular,Menlo,monospace;">'
                    f'{h.subtitle}</div></div>',
                    unsafe_allow_html=True,
                )
            with cols[2]:
                _btn_key = f"omni_hit_{i}_{h.kind}_{h.payload.get('instance_id','') or h.payload.get('draft_name','') or h.payload.get('norm_code','')}"
                _btn_label = "Abrir →"
                if st.button(_btn_label, key=_btn_key, use_container_width=True):
                    if h.kind == "instance":
                        st.session_state["wm_active_instance"] = h.payload.get("instance_id", "")
                    elif h.kind == "report":
                        st.session_state["wm_active_draft_name"] = h.payload.get("draft_name", "")
                    elif h.kind == "norm":
                        st.session_state["wm_omnibox_focus_norm"] = h.payload.get("norm_code", "")
                    if h.target_page:
                        try:
                            st.switch_page(h.target_page)
                        except Exception:
                            pass
            # Pill kind small
            st.markdown(
                f'<div style="margin:-4px 0 6px 38px;">'
                f'<span class="wmh-omni-hit-kind" style="background:{kcol}1A;color:{kcol};">'
                f'{klabel}</span></div>',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)


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
    # Ciclo 17.24 — card clickable: navega a Diagnostics
    if st.button("Ver críticos →", key="kpi_btn_danger", use_container_width=True):
        st.session_state["wm_diag_filter_severity"] = "danger"
        st.switch_page("pages/15_Diagnostics.py")
with k3:
    st.markdown(_kpi_card(
        "warning", "En atención", _warn,
        "vigilancia o config pendiente",
        _spark_svg(_spark_warn, "#f59e0b"),
    ), unsafe_allow_html=True)
    # Ciclo 17.24 — card clickable: navega a Diagnostics
    if st.button("Ver en atención →", key="kpi_btn_warning", use_container_width=True):
        st.session_state["wm_diag_filter_severity"] = "warning"
        st.switch_page("pages/15_Diagnostics.py")
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
# QUICK ACTIONS
# =============================================================
st.markdown(
    '<div class="wmh-sec">⚡ Acciones rápidas <div class="bar"></div></div>',
    unsafe_allow_html=True,
)

qa_cols = st.columns(6)
_QUICK = [
    ("📤  Cargar CSV",    "pages/01_Load_Data.py",          "page"),
    ("📈  Trends",        "pages/04_Trends.py",             "page"),
    ("📚  Machinery Lib", "pages/00_Machinery_Library.py",  "page"),
    ("🔬  Diagnostics",   "pages/15_Diagnostics.py",        "page"),
    ("📄  Reports",       "pages/16_Reports.py",            "page"),
    ("📰  Briefing día",  "_briefing_action",               "briefing"),
]
for col, (label, target, kind) in zip(qa_cols, _QUICK):
    with col:
        if st.button(label, use_container_width=True, key=f"qa_{target}"):
            if kind == "briefing":
                # Ciclo 17.13 — Genera el PDF del día al instante.
                # Snapshot, comparativo vs ayer, top 3, vencimientos.
                try:
                    with st.spinner("Generando briefing del día…"):
                        _pdf_b, _pdf_p = generate_and_save_briefing()
                    st.session_state["wm_briefing_pdf_bytes"] = _pdf_b
                    st.session_state["wm_briefing_pdf_path"] = str(_pdf_p)
                    st.session_state["wm_briefing_pdf_at"] = datetime.now().strftime("%H:%M")
                except Exception as _e:
                    st.session_state["wm_briefing_pdf_bytes"] = None
                    st.session_state["wm_briefing_pdf_error"] = str(_e)
            else:
                try:
                    st.switch_page(target)
                except Exception:
                    st.warning(f"No pude abrir {target}")

# Download del briefing si recién se generó
_brief_bytes = st.session_state.get("wm_briefing_pdf_bytes")
_brief_err = st.session_state.get("wm_briefing_pdf_error")
if _brief_err:
    st.error(f"No pude generar el briefing: {_brief_err}")
    st.session_state.pop("wm_briefing_pdf_error", None)
elif _brief_bytes:
    _at = st.session_state.get("wm_briefing_pdf_at", "")
    _date_str = datetime.now().strftime("%Y-%m-%d")
    st.success(
        f"✓ Briefing diario listo (generado {_at}). "
        f"Tamaño {len(_brief_bytes)/1024:.1f} KB."
    )
    st.download_button(
        label=f"⬇️  Descargar briefing_{_date_str}.pdf",
        data=_brief_bytes,
        file_name=f"briefing_{_date_str}.pdf",
        mime="application/pdf",
        use_container_width=True,
        key="wm_briefing_dl",
    )


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
        # Grid 3 columnas, máximo 9 cards visibles
        # Cada card incluye Health Score gauge SVG (Ciclo 17.12 Nivel 3)
        instances = _fleet["instances"][:9]
        for row_start in range(0, len(instances), 3):
            cols = st.columns(3, gap="medium")
            for col, inst in zip(cols, instances[row_start:row_start + 3]):
                with col:
                    sev_class = inst.severity
                    asset_line = inst.asset_class or inst.profile_key or "—"
                    loc_line = inst.location or "sin ubicación"

                    # Calcular health score (carga la instancia full)
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

                    _gauge_html = render_score_gauge(_hs.score, _hs.color, size=110)
                    _band_html = (
                        f'<div class="wmh-gauge-band" style="color:{_hs.color};">'
                        f'{_hs.band_label}</div>'
                    )

                    # IMPORTANTE: HTML colapsado a sin saltos internos
                    # para que el markdown parser de Streamlit no corte
                    # el bloque HTML por blank lines y muestre crudo.
                    _card_html = (
                        f'<div class="wmh-asset">'
                        f'<div class="wmh-asset-head">'
                        f'<span class="wmh-asset-tag">{inst.severity_dot} {inst.tag}</span>'
                        f'<span class="wmh-sev-pill {sev_class}">{inst.severity_label}</span>'
                        f'</div>'
                        f'<div class="wmh-asset-class">{asset_line}</div>'
                        f'<div class="wmh-gauge-wrap">{_gauge_html}{_band_html}'
                        f'<div class="wmh-gauge-tip">{_hs.one_liner}</div>'
                        f'</div>'
                        f'<div class="wmh-asset-meta">'
                        f'📍 {loc_line} · 📁 {inst.n_documents} docs · 🕒 {inst.last_seen_human}'
                        f'</div>'
                        f'</div>'
                    )
                    st.markdown(_card_html, unsafe_allow_html=True)
                    bcols = st.columns(2)
                    with bcols[0]:
                        if st.button("Trends", key=f"asset_tr_{inst.instance_id}",
                                     use_container_width=True):
                            st.session_state["wm_active_instance"] = inst.instance_id
                            try:
                                st.switch_page("pages/04_Trends.py")
                            except Exception:
                                pass
                    with bcols[1]:
                        if st.button("Editar", key=f"asset_ed_{inst.instance_id}",
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
