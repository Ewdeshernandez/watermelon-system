"""
pages/_asset_briefing.py
========================

Briefing por ACTIVO — Semanal / Mensual (F4 UI).

Página OCULTA del nav (admin/specialist). Genera a demanda el briefing
figura-rico de un activo (o de todos) reusando el motor F1-F3
(core.briefing_builder). El especialista lo revisa y descarga; el envío
automático lo hace el cron del lunes 6am (scripts/send_weekly_briefing.py).
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
# Controles
# -----------------------------------------------------------------
c1, c2 = st.columns([1, 2])
with c1:
    _period = st.radio("Periodo", ["Semanal", "Mensual"], horizontal=True,
                       key="briefing_period")
with c2:
    _scope = st.radio("Alcance", ["Un activo", "Todos los activos"],
                      horizontal=True, key="briefing_scope")

_target_iid = None
if _scope == "Un activo":
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
            _sel = st.selectbox("Activo", list(_label_map.keys()), key="briefing_asset")
            _target_iid = _label_map.get(_sel)
        else:
            st.info("No hay activos registrados.")
    except Exception as e:
        st.error(f"No se pudieron listar los activos: {e}")

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
if st.button("📄 Generar briefing", type="primary", use_container_width=True):
    from core.briefing_builder import build_asset_briefing, build_all_briefings
    results = []
    with st.spinner("Generando briefing(s)… (figuras + redacción + PDF)"):
        if _scope == "Todos los activos":
            for iid, pdf, meta in build_all_briefings(_period, use_ai=_use_ai):
                results.append((iid, pdf, meta))
        elif _target_iid:
            pdf, meta = build_asset_briefing(_target_iid, _period, use_ai=_use_ai)
            results.append((_target_iid, pdf, meta))
        else:
            st.warning("Selecciona un activo.")

    _ok = [r for r in results if r[1]]
    if results:
        st.success(f"{len(_ok)} de {len(results)} briefing(s) generados.")
    for iid, pdf, meta in results:
        tag = meta.get("tag", iid)
        if not pdf:
            st.warning(f"**{tag}** — {meta.get('status', 'sin datos')} (no se generó)")
            continue
        _sev = meta.get("status", "—")
        _dot = ("#dc2626" if "rít" in _sev else
                "#d97706" if "tenci" in _sev else "#10b981")
        with st.container():
            cc1, cc2 = st.columns([3, 1])
            with cc1:
                st.markdown(
                    f"<div style='display:flex;align-items:center;gap:8px;'>"
                    f"<span style='width:10px;height:10px;border-radius:50%;"
                    f"background:{_dot};'></span>"
                    f"<b>{tag}</b> · {_sev} · salud {meta.get('score','—')} · "
                    f"{meta.get('alarms',0)} alarma(s) · {meta.get('n_figures',0)} figuras"
                    f"</div>", unsafe_allow_html=True)
            with cc2:
                st.download_button(
                    "⬇ Descargar PDF", data=pdf,
                    file_name=f"Briefing_{tag}_{_period}.pdf",
                    mime="application/pdf", key=f"dl_{iid}",
                    use_container_width=True)
        st.markdown("---")

st.caption(
    "💡 El briefing automático del **lunes 6am** corre para todos los activos y "
    "se envía al especialista para revisión antes de remitirlo al cliente.")
