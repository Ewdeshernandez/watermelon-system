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
            from core.instance_state import get_instances_version, list_instances

            # Cachear la lista de activos: sin esto, CADA rerun (cada edición
            # del data_editor de recomendaciones) hace un round-trip a
            # Supabase y la página se siente lenta. Se invalida sola cuando
            # alguien muta una instancia (get_instances_version).
            @st.cache_data(show_spinner=False)
            def _cached_instances(_v: int):
                return list_instances() or []

            _rows = _cached_instances(get_instances_version())
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
# Generación — SOLO vía flujo de aprobación (v3.31.400)
# No existe generación directa de PDF: todo reporte nace como BORRADOR
# en la cola, se revisa (solo recomendaciones editables), se firma y al
# aprobar se genera el PDF final y se envía al cliente.
# -----------------------------------------------------------------
if st.button("📝  Generar reporte y enviarlo a APROBACIÓN",
             type="primary", use_container_width=True, key="bfq_manual"):
    from core.briefing_builder import build_asset_draft, build_all_drafts
    _ms = []
    with st.spinner("Generando borrador(es) — datos + redacción IA…"):
        if _scope == "Todos los activos":
            _ms = build_all_drafts(_period, use_ai=_use_ai)
        elif _target_iid:
            _ms = [build_asset_draft(_target_iid, _period, use_ai=_use_ai)]
        else:
            st.warning("Selecciona un activo.")
    if _ms:
        _ok = [m for m in _ms if m.get("ok")]
        _bad = [m for m in _ms if not m.get("ok")]
        for m in _bad:
            st.warning(f"{m.get('tag', m.get('instance_id'))}: "
                       f"{m.get('status','?')} (no se creó borrador)")
        if _ok:
            from core.briefing_queue import list_pending
            st.session_state["bfq_cache"] = list_pending()
            st.success(f"✅ {len(_ok)} borrador(es) en la cola — apruébalos "
                       f"abajo en 'Pendientes de aprobación'.")

# -----------------------------------------------------------------
# Programación AUTOMÁTICA (v3.31.401) — el sistema genera el reporte
# el día y hora configurados y lo deja en la cola de aprobación solo.
# La config vive en Supabase; el cron horario de Render la lee
# (send_weekly_briefing.py --check-schedule).
# -----------------------------------------------------------------
with st.expander("⏰ Programación automática — generar y enviar a "
                 "aprobación solo", expanded=False):
    from core.briefing_queue import get_schedule, save_schedule

    if "bfq_sched" not in st.session_state:
        st.session_state["bfq_sched"] = get_schedule()
    _cfg = st.session_state["bfq_sched"]

    _DIAS = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes",
             "Sábado", "Domingo"]
    _en = st.toggle("Generación automática activa",
                    value=bool(_cfg.get("enabled")), key="bfq_sched_en")
    _sc1, _sc2, _sc3 = st.columns([2, 1, 1])
    with _sc1:
        _days_sel = st.multiselect(
            "Día(s)", _DIAS,
            default=[_DIAS[d] for d in (_cfg.get("days") or [0])
                     if 0 <= int(d) <= 6],
            key="bfq_sched_days")
    with _sc2:
        _hour_sel = st.selectbox(
            "Hora (Bogotá)", list(range(24)),
            index=int(_cfg.get("hour", 5)),
            format_func=lambda h: f"{h:02d}:00", key="bfq_sched_hour")
    with _sc3:
        _per_sel = st.selectbox(
            "Periodo", ["Semanal", "Mensual"],
            index=(1 if str(_cfg.get("period", "")).startswith("Mensual") else 0),
            key="bfq_sched_period")
    if st.button("💾 Guardar programación", key="bfq_sched_save"):
        _new_cfg = {
            "enabled": _en,
            "days": [_DIAS.index(d) for d in (_days_sel or ["Lunes"])],
            "hour": int(_hour_sel),
            "period": _per_sel,
        }
        if save_schedule(_new_cfg):
            st.session_state["bfq_sched"] = _new_cfg
            _dias_txt = ", ".join(_days_sel or ["Lunes"])
            st.success(f"✅ Programado: {_dias_txt} a las {_hour_sel:02d}:00 "
                       f"({_per_sel}). Los borradores llegarán solos a "
                       f"'Pendientes de aprobación' y verás la alerta al "
                       f"entrar a la app."
                       if _en else "Programación guardada (desactivada).")
        else:
            st.error("No se pudo guardar la programación.")

# -----------------------------------------------------------------
# PENDIENTES DE APROBACIÓN — lista compacta (v3.31.399)
# El módulo abre LIMPIO: arriba se crea el reporte nuevo; aquí abajo
# solo se listan los pendientes. El detalle (recomendaciones editables
# + firmas) se abre ÚNICAMENTE al presionar "Aprobar…" en una fila.
# Render mínimo por rerun = página rápida.
# -----------------------------------------------------------------
st.markdown("")
try:
    from core.auth import get_current_user as _gcu
    _me = (_gcu() or {})
    _me_name = _me.get("full_name") or _me.get("username") or ""
except Exception:
    _me_name = ""

if "bfq_cache" not in st.session_state:
    from core.briefing_queue import list_pending
    st.session_state["bfq_cache"] = list_pending()
_pending = st.session_state["bfq_cache"]

_qh1, _qh2 = st.columns([4, 1])
with _qh1:
    st.markdown(
        f'<div class="bf-label">Pendientes de aprobación '
        f'<span class="bf-accent">·</span> {len(_pending)} reporte(s)</div>',
        unsafe_allow_html=True,
    )
with _qh2:
    if st.button("🔄 Actualizar", key="bfq_refresh", use_container_width=True):
        from core.briefing_queue import list_pending
        st.session_state["bfq_cache"] = list_pending()
        st.session_state.pop("bfq_open", None)
        st.rerun()

if not _pending:
    st.caption("No hay reportes pendientes. El cron del lunes 6am los deja "
               "aquí; también puedes crear uno arriba con "
               "'Generar BORRADOR'.")

_open_iid = st.session_state.get("bfq_open", "")

for _iid, _tag, _d in _pending:
    _r1, _r2 = st.columns([4, 1])
    with _r1:
        _st_txt = (_d.get("kpis") or {}).get("status", "—")
        st.markdown(
            f"""
            <div class="bf-row" style="--bf-dot:#94a3b8;">
                <span class="bf-dot"></span>
                <div class="bf-row-main">
                    <span class="bf-row-tag">{_tag}</span>
                    &nbsp;<span class="bf-row-sev">· Reporte {_d.get('period','Semanal')}
                    · {_st_txt}</span>
                    <div class="bf-chips">
                        <span class="bf-chip">{_d.get('consecutive','') or 'sin consecutivo'}</span>
                        <span class="bf-chip">borrador {( _d.get('created_at','') or '')[:16]}</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with _r2:
        if _open_iid == _iid:
            if st.button("✖ Cerrar", key=f"bfq_close_{_iid}",
                         use_container_width=True):
                st.session_state.pop("bfq_open", None)
                st.rerun()
        else:
            if st.button("✅ Aprobar…", key=f"bfq_openbtn_{_iid}",
                         use_container_width=True):
                st.session_state["bfq_open"] = _iid
                st.rerun()

    if _open_iid != _iid:
        st.markdown("")
        continue

    # ============= DETALLE — solo del reporte abierto =============
    _sum = _d.get("summary", "")
    _diag = _d.get("diagnosis", "")
    with st.container(border=True):
        with st.expander("👁 Resumen ejecutivo y diagnóstico (solo lectura)",
                          expanded=False):
            st.markdown(_sum or "_(sin resumen)_")
            st.markdown("---")
            st.markdown(_diag or "_(sin diagnóstico)_")

        # Recomendaciones: lo ÚNICO editable en aprobación. Vienen del
        # reporte anterior con su fecha. Carga perezosa (solo al abrir).
        st.markdown("**Recomendaciones** (traídas del reporte anterior — "
                    "edita, agrega o borra las que el cliente ya ejecutó):")
        from datetime import date as _qdate

        import pandas as _qpd

        from core.briefing_recommendations import (
            list_recommendations as _qlist,
            save_recommendations as _qsave,
        )
        _q_ss = f"recs_cache_{_iid}"
        if _q_ss not in st.session_state:
            st.session_state[_q_ss] = _qlist(_iid)

        def _q_to_date(s):
            try:
                return _qdate.fromisoformat(str(s)[:10])
            except Exception:
                return _qdate.today()

        _qdf = _qpd.DataFrame(
            [{"id": r["id"], "Recomendación": r["text"],
              "Fecha de inicio": _q_to_date(r["started_at"])}
             for r in st.session_state[_q_ss]],
            columns=["id", "Recomendación", "Fecha de inicio"],
        )
        _qedited = st.data_editor(
            _qdf, key=f"bfq_recs_{_iid}", num_rows="dynamic",
            use_container_width=True, hide_index=True,
            column_config={
                "id": None,
                "Recomendación": st.column_config.TextColumn(
                    "Recomendación", width="large", required=True),
                "Fecha de inicio": st.column_config.DateColumn(
                    "Fecha de inicio", format="YYYY-MM-DD",
                    default=_qdate.today()),
            },
        )

        _s1, _s2 = st.columns(2)
        with _s1:
            _elab = st.text_input("Elaborado por", value=_me_name,
                                  key=f"bfq_elab_{_iid}")
            _elab_rol = st.text_input("Cargo (elaboró)", value="",
                                      key=f"bfq_elabr_{_iid}",
                                      placeholder="opcional")
        with _s2:
            _aprb = st.text_input("Aprobado por", value="",
                                  key=f"bfq_aprb_{_iid}",
                                  placeholder="obligatorio para aprobar")
            _aprb_rol = st.text_input("Cargo (aprobó)", value="",
                                      key=f"bfq_aprbr_{_iid}",
                                      placeholder="opcional")

        _b2, _b3 = st.columns([1, 1.6])
        with _b2:
            if st.button("👁 Vista previa PDF", key=f"bfq_prev_{_iid}",
                         use_container_width=True):
                # Persistir recomendaciones en pantalla antes de rasterizar
                try:
                    _qsave(_iid, [{"id": r.get("id") or "",
                                   "text": r.get("Recomendación") or "",
                                   "started_at": r.get("Fecha de inicio")}
                                  for _, r in _qedited.iterrows()])
                    st.session_state[_q_ss] = _qlist(_iid)
                except Exception:
                    pass
                with st.spinner("Generando vista previa…"):
                    from core.briefing_builder import build_asset_briefing
                    _pdf, _m = build_asset_briefing(
                        _iid, _d.get("period", "Semanal"), use_ai=False,
                        sections_override={"summary": _sum, "diagnosis": _diag},
                        meta_extra={
                            "prepared_by": _elab or _me_name,
                            "reviewed_by": _aprb,
                            "prepared_label": "Elaborado por:",
                            "reviewed_label": "Aprobado por:",
                            "consecutive": _d.get("consecutive", ""),
                        },
                    )
                st.session_state[f"bfq_pdf_{_iid}"] = _pdf
        with _b3:
            if st.button("✅ Aprobar y enviar al cliente", key=f"bfq_go_{_iid}",
                         type="primary", use_container_width=True):
                if not (_aprb or "").strip():
                    st.error("Falta 'Aprobado por' — el reporte siempre debe "
                             "llevar elaborado y aprobado.")
                elif not (_elab or "").strip():
                    st.error("Falta 'Elaborado por'.")
                else:
                    from core.briefing_queue import approve_and_send
                    # Persistir la tabla de recomendaciones tal como está en
                    # pantalla (lo ÚNICO editable en aprobación).
                    try:
                        _qsave(_iid, [{"id": r.get("id") or "",
                                       "text": r.get("Recomendación") or "",
                                       "started_at": r.get("Fecha de inicio")}
                                      for _, r in _qedited.iterrows()])
                        st.session_state[_q_ss] = _qlist(_iid)
                    except Exception:
                        pass
                    with st.spinner("Aprobando, generando PDF final y "
                                    "enviando al cliente…"):
                        _res = approve_and_send(
                            _iid, prepared_by=_elab, approved_by=_aprb,
                            prepared_role=_elab_rol, approved_role=_aprb_rol,
                            send=True,
                        )
                    if _res.get("ok"):
                        _dv = _res.get("delivery") or {}
                        if _dv.get("any_ok"):
                            st.success(f"✅ {_tag} aprobado y ENVIADO al cliente.")
                        else:
                            st.warning(
                                f"Aprobado, pero el envío falló o el activo no "
                                f"tiene canales configurados: "
                                f"{_dv.get('error', _dv)}. Descarga el PDF y "
                                f"envíalo manualmente.")
                        st.session_state[f"bfq_pdf_{_iid}"] = _res.get("pdf")
                        st.session_state.pop("bfq_open", None)
                        from core.briefing_queue import list_pending
                        st.session_state["bfq_cache"] = list_pending()
                    else:
                        st.error(f"No se pudo aprobar: {_res.get('error')}")

        if st.session_state.get(f"bfq_pdf_{_iid}"):
            st.download_button(
                "⬇ Descargar PDF (última versión generada)",
                data=st.session_state[f"bfq_pdf_{_iid}"],
                file_name=f"Reporte_{_tag}_{_d.get('period','Semanal')}.pdf",
                mime="application/pdf", key=f"bfq_dl_{_iid}",
                use_container_width=True,
            )
    st.markdown("")

st.markdown(
    '<div class="bf-foot">💡&nbsp;Flujo: el <b>lunes 6am</b> el sistema deja el '
    'borrador de cada activo en <b>Pendientes de aprobación</b> → el '
    'especialista lo revisa/edita → firma <b>Elaborado por</b> y '
    '<b>Aprobado por</b> → al aprobar, el PDF final se envía automáticamente '
    'al cliente por los canales del activo.</div>',
    unsafe_allow_html=True,
)
