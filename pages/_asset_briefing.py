"""
pages/_asset_briefing.py
========================

Briefing por ACTIVO — Semanal / Mensual (F4 UI).

Página OCULTA del nav (admin/specialist). Genera a demanda el briefing
figura-rico de un activo (o de todos) reusando el motor F1-F3
(core.briefing_builder). El especialista lo revisa y descarga; el envío
automático lo hace el cron del lunes 6am (scripts/send_weekly_briefing.py).

v3.31.350 — Refinamiento estético clase enterprise (System1 / AMS-grade):
controles como segmented controls, panel de configuración contenido,
ledger de resultados con chips de severidad. Misma paleta del design
system (core/ui_theme). Funcionalidad idéntica.
"""
from __future__ import annotations

import streamlit as st

from core.auth import require_login, render_user_menu, require_role

st.set_page_config(
    page_title="Watermelon System | Briefing por activo",
    layout="wide",
    initial_sidebar_state="expanded",
)
require_login()
render_user_menu()
require_role(("admin", "specialist"))

from core.ui_theme import apply_watermelon_page_style, page_header
apply_watermelon_page_style()
page_header(
    "Briefing por activo",
    subtitle="Resumen gerencial + figuras (tendencia · espectro · onda · órbita) "
             "por activo · Semanal o Mensual · con el formato de tus reportes.",
)

# -----------------------------------------------------------------
# Estilo local — refinamiento enterprise (scoped a esta página).
# CSS inyectado por st.markdown solo aplica al render de ESTE módulo,
# así que podemos refinar radios/botón sin afectar el resto de la app.
# -----------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Micro-label de sección — mismo lenguaje que el hero pill */
    .bf-label {
        font-size: 10.5px; font-weight: 700; letter-spacing: 0.16em;
        text-transform: uppercase; color: #94a3b8;
        margin: 0 0 9px 2px;
    }
    .bf-label .bf-accent { color: #e11d48; }

    /* ---- Radios como segmented control ---- */
    div[data-testid="stRadio"] [role="radiogroup"] { gap: 8px; }
    div[data-testid="stRadio"] [role="radiogroup"] > label {
        border: 1px solid #e2e8f0;
        background: #f8fafc;
        border-radius: 11px;
        padding: 9px 16px;
        margin: 0 !important;
        transition: all .16s ease;
        cursor: pointer;
    }
    div[data-testid="stRadio"] [role="radiogroup"] > label:hover {
        border-color: #cbd5e1; background: #ffffff;
    }
    /* Oculta el círculo nativo; dejamos solo el texto */
    div[data-testid="stRadio"] [role="radiogroup"] > label > div:first-child {
        display: none !important;
    }
    div[data-testid="stRadio"] [role="radiogroup"] > label p {
        font-weight: 600; color: #475569; font-size: 0.92rem;
    }
    /* Estado seleccionado (Chrome soporta :has) */
    div[data-testid="stRadio"] [role="radiogroup"] > label:has(input:checked) {
        border-color: #0f1d36;
        background: linear-gradient(180deg, #0f1d36 0%, #16284a 100%);
        box-shadow: 0 6px 16px rgba(15,29,54,0.18);
    }
    div[data-testid="stRadio"] [role="radiogroup"] > label:has(input:checked) p {
        color: #f8fafc;
    }

    /* ---- Botón primario: rojo Watermelon refinado (no neón) ---- */
    section.main [data-testid="stButton"] button[kind="primary"],
    section.main [data-testid="stBaseButton-primary"] {
        background: linear-gradient(180deg, #f43f5e 0%, #e11d48 100%) !important;
        border: 1px solid #be123c !important;
        color: #ffffff !important;
        border-radius: 14px !important;
        min-height: 54px !important;
        font-weight: 700 !important;
        letter-spacing: 0.01em !important;
        box-shadow: 0 12px 26px rgba(225,29,72,0.26) !important;
        transition: all .18s ease !important;
    }
    section.main [data-testid="stButton"] button[kind="primary"]:hover,
    section.main [data-testid="stBaseButton-primary"]:hover {
        background: linear-gradient(180deg, #fb5071 0%, #d11842 100%) !important;
        box-shadow: 0 16px 32px rgba(225,29,72,0.34) !important;
        transform: translateY(-1px);
    }

    /* ---- Ledger de resultados ---- */
    .bf-row {
        display: flex; align-items: center; gap: 14px;
        border: 1px solid #e6ecf5;
        border-left: 4px solid var(--bf-dot, #94a3b8);
        background: linear-gradient(180deg, #ffffff 0%, #fbfcfe 100%);
        border-radius: 14px;
        padding: 13px 18px;
        box-shadow: 0 6px 18px rgba(15,23,42,0.04);
    }
    .bf-row-main { flex: 1 1 auto; min-width: 0; }
    .bf-row-tag {
        font-size: 1.02rem; font-weight: 800; color: #0f172a;
        letter-spacing: -0.01em;
    }
    .bf-row-sev { font-size: 0.85rem; font-weight: 600; color: #475569; }
    .bf-chips { display: flex; flex-wrap: wrap; gap: 7px; margin-top: 7px; }
    .bf-chip {
        border: 1px solid #e2e8f0; background: #f8fafc; color: #334155;
        border-radius: 999px; padding: 3px 11px;
        font-size: 0.78rem; font-weight: 600;
    }
    .bf-dot {
        width: 11px; height: 11px; border-radius: 50%; flex: 0 0 auto;
        box-shadow: 0 0 0 4px var(--bf-dot-soft, rgba(148,163,184,0.16));
    }
    .bf-summary {
        font-size: 0.86rem; color: #475569; font-weight: 600;
        padding: 4px 0 2px 2px;
    }
    .bf-foot {
        display: flex; align-items: flex-start; gap: 9px;
        border: 1px solid #e6ecf5; background: #f8fafc;
        border-radius: 12px; padding: 11px 15px; margin-top: 6px;
        color: #64748b; font-size: 0.86rem; line-height: 1.5;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------
# Panel de configuración
# -----------------------------------------------------------------
with st.container(border=True):
    cfg1, cfg2 = st.columns(2)
    with cfg1:
        st.markdown('<div class="bf-label">Periodo</div>', unsafe_allow_html=True)
        _period = st.radio(
            "Periodo", ["Semanal", "Mensual"], horizontal=True,
            key="briefing_period", label_visibility="collapsed",
        )
    with cfg2:
        st.markdown('<div class="bf-label">Alcance</div>', unsafe_allow_html=True)
        _scope = st.radio(
            "Alcance", ["Un activo", "Todos los activos"], horizontal=True,
            key="briefing_scope", label_visibility="collapsed",
        )

    _target_iid = None
    if _scope == "Un activo":
        st.markdown('<div class="bf-label" style="margin-top:14px;">Activo</div>',
                    unsafe_allow_html=True)
        try:
            from core.instance_state import list_instances
            _rows = list_instances() or []
            _opts = []
            for r in _rows:
                iid = r.get("instance_id") if isinstance(r, dict) else getattr(r, "instance_id", "")
                tag = (r.get("tag") if isinstance(r, dict) else getattr(r, "tag", "")) or iid
                if iid:
                    _opts.append((iid, tag))
            if _opts:
                _label_map = {f"{tag} ({iid})": iid for iid, tag in _opts}
                _sel = st.selectbox("Activo", list(_label_map.keys()),
                                    key="briefing_asset", label_visibility="collapsed")
                _target_iid = _label_map.get(_sel)
            else:
                st.info("No hay activos registrados.")
        except Exception as e:
            st.error(f"No se pudieron listar los activos: {e}")

    st.markdown('<div class="bf-label" style="margin-top:14px;">Redacción</div>',
                unsafe_allow_html=True)
    _use_ai = st.toggle(
        "Usar IA para redacción (si hay credenciales)", value=True,
        key="briefing_use_ai",
        help="Si está activo y hay credenciales, la IA mejora el borrador. "
             "Si no, se usa el borrador determinístico (siempre funciona).",
    )

st.markdown("")

# -----------------------------------------------------------------
# Generación
# -----------------------------------------------------------------
if st.button("📄  Generar briefing", type="primary", use_container_width=True):
    from core.briefing_builder import build_asset_briefing, build_all_briefings
    # Portada profesional: firma "Preparado por" con el usuario logueado
    # (mismo criterio que Reports). El revisor firma al validar.
    _meta_extra = {}
    try:
        from core.auth import get_current_user
        _u = get_current_user() or {}
        _meta_extra["prepared_by"] = _u.get("full_name") or _u.get("username") or ""
    except Exception:
        pass
    results = []
    with st.spinner("Generando briefing(s)… (figuras + redacción + PDF)"):
        if _scope == "Todos los activos":
            for iid, pdf, meta in build_all_briefings(_period, use_ai=_use_ai,
                                                      meta_extra=_meta_extra):
                results.append((iid, pdf, meta))
        elif _target_iid:
            pdf, meta = build_asset_briefing(_target_iid, _period, use_ai=_use_ai,
                                             meta_extra=_meta_extra)
            results.append((_target_iid, pdf, meta))
        else:
            st.warning("Selecciona un activo.")

    _ok = [r for r in results if r[1]]
    if results:
        st.markdown("")
        st.markdown(
            f'<div class="bf-label" style="margin-top:6px;">'
            f'Briefings generados <span class="bf-accent">·</span> '
            f'{len(_ok)} de {len(results)}</div>',
            unsafe_allow_html=True,
        )

    for iid, pdf, meta in results:
        tag = meta.get("tag", iid)
        if not pdf:
            st.warning(f"**{tag}** — {meta.get('status', 'sin datos')} (no se generó)")
            continue

        _sev = meta.get("status", "—")
        _dot, _soft = (
            ("#dc2626", "rgba(220,38,38,0.16)") if "rít" in _sev else
            ("#d97706", "rgba(217,119,6,0.16)") if "tenci" in _sev else
            ("#10b981", "rgba(16,185,129,0.16)")
        )
        cc1, cc2 = st.columns([3.2, 1])
        with cc1:
            st.markdown(
                f"""
                <div class="bf-row" style="--bf-dot:{_dot};--bf-dot-soft:{_soft};">
                    <span class="bf-dot"></span>
                    <div class="bf-row-main">
                        <span class="bf-row-tag">{tag}</span>
                        &nbsp;<span class="bf-row-sev">· {_sev}</span>
                        <div class="bf-chips">
                            <span class="bf-chip">Salud {meta.get('score','—')}</span>
                            <span class="bf-chip">{meta.get('alarms',0)} alarma(s)</span>
                            <span class="bf-chip">{meta.get('n_figures',0)} figuras</span>
                            <span class="bf-chip">{_period}</span>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with cc2:
            st.download_button(
                "⬇  Descargar PDF", data=pdf,
                file_name=f"Briefing_{tag}_{_period}.pdf",
                mime="application/pdf", key=f"dl_{iid}",
                use_container_width=True)
        st.markdown("")

st.markdown(
    '<div class="bf-foot">💡&nbsp;El briefing automático del <b>lunes 6am</b> '
    'corre para todos los activos y se envía al especialista para revisión '
    'antes de remitirlo al cliente.</div>',
    unsafe_allow_html=True,
)
