"""
core/report_center_ui.py
========================

Render helpers del **Report Center** (pestaña "Report Center" del nav).

Un solo lugar para todo lo de reportes:
  · render_approval(me_name)  → generar borrador + cola de PENDIENTES +
    detalle de revisión/firma/aprobación (la lógica que vivía en
    pages/_asset_briefing.py, ahora compartida).
  · render_delivery()         → programación de ENVÍOS centralizada por
    Cliente → Activo: correo(s), WhatsApp(s), horario del briefing
    (semanal/mensual) y aviso automático por alarma. Reemplaza la config
    dispersa que estaba en Machinery Library.

No hace set_page_config ni auth — eso lo hace la página que lo importa.
"""
from __future__ import annotations

import streamlit as st


# -----------------------------------------------------------------
# CSS compartido (mismo lenguaje enterprise System1/AMS del briefing)
# -----------------------------------------------------------------
def inject_css() -> None:
    st.markdown(
        """
        <style>
        .bf-label {
            font-size: 10.5px; font-weight: 700; letter-spacing: 0.16em;
            text-transform: uppercase; color: #94a3b8; margin: 0 0 9px 2px;
        }
        .bf-label .bf-accent { color: #e11d48; }
        div[data-testid="stRadio"] [role="radiogroup"] { gap: 8px; }
        div[data-testid="stRadio"] [role="radiogroup"] > label {
            border: 1px solid #e2e8f0; background: #f8fafc; border-radius: 11px;
            padding: 9px 16px; margin: 0 !important; transition: all .16s ease; cursor: pointer;
        }
        div[data-testid="stRadio"] [role="radiogroup"] > label:hover {
            border-color: #cbd5e1; background: #ffffff;
        }
        div[data-testid="stRadio"] [role="radiogroup"] > label > div:first-child { display: none !important; }
        div[data-testid="stRadio"] [role="radiogroup"] > label p { font-weight: 600; color: #475569; font-size: 0.92rem; }
        div[data-testid="stRadio"] [role="radiogroup"] > label:has(input:checked) {
            border-color: #0f1d36; background: linear-gradient(180deg, #0f1d36 0%, #16284a 100%);
            box-shadow: 0 6px 16px rgba(15,29,54,0.18);
        }
        div[data-testid="stRadio"] [role="radiogroup"] > label:has(input:checked) p { color: #f8fafc; }
        .bf-row {
            display: flex; align-items: center; gap: 14px;
            border: 1px solid #e6ecf5; border-left: 4px solid var(--bf-dot, #94a3b8);
            background: linear-gradient(180deg, #ffffff 0%, #fbfcfe 100%);
            border-radius: 14px; padding: 13px 18px; box-shadow: 0 6px 18px rgba(15,23,42,0.04);
        }
        .bf-row-main { flex: 1 1 auto; min-width: 0; }
        .bf-row-tag { font-size: 1.02rem; font-weight: 800; color: #0f172a; letter-spacing: -0.01em; }
        .bf-row-sev { font-size: 0.85rem; font-weight: 600; color: #475569; }
        .bf-chips { display: flex; flex-wrap: wrap; gap: 7px; margin-top: 7px; }
        .bf-chip { border: 1px solid #e2e8f0; background: #f8fafc; color: #334155;
            border-radius: 999px; padding: 3px 11px; font-size: 0.78rem; font-weight: 600; }
        .bf-dot { width: 11px; height: 11px; border-radius: 50%; flex: 0 0 auto;
            box-shadow: 0 0 0 4px var(--bf-dot-soft, rgba(148,163,184,0.16)); }
        .bf-foot { display: flex; align-items: flex-start; gap: 9px;
            border: 1px solid #e6ecf5; background: #f8fafc; border-radius: 12px;
            padding: 11px 15px; margin-top: 6px; color: #64748b; font-size: 0.86rem; line-height: 1.5; }
        /* Delivery center */
        .dc-client { font-size: 1.05rem; font-weight: 800; color: #0f172a; letter-spacing: -0.01em; }
        .dc-meta { color: #94a3b8; font-size: 0.82rem; font-weight: 600; }
        .dc-asset-tag { font-size: 1.0rem; font-weight: 800; color: #0f172a; }
        .dc-asset-sub { color: #64748b; font-size: 0.82rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# =================================================================
# VISTA 1 — APROBACIÓN (generar borrador + cola + detalle)
# =================================================================
_DIAS_EN = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
            "Saturday", "Sunday"]


def _cached_instances():
    from core.instance_state import get_instances_version, list_instances

    @st.cache_data(show_spinner=False)
    def _inner(_v: int):
        return list_instances() or []

    return _inner(get_instances_version())


def render_approval(me_name: str = "") -> None:
    """Panel de generación + cola de pendientes + detalle de aprobación."""
    from core.briefing_queue import list_pending

    # ---- Panel de generación ----
    with st.container(border=True):
        cfg1, cfg2 = st.columns(2)
        with cfg1:
            st.markdown('<div class="bf-label">Period</div>', unsafe_allow_html=True)
            _period = st.radio("Period", ["Semanal", "Mensual"], horizontal=True,
                               key="rc_period", label_visibility="collapsed")
        with cfg2:
            st.markdown('<div class="bf-label">Scope</div>', unsafe_allow_html=True)
            _scope = st.radio("Scope", ["Un activo", "Todos los activos"], horizontal=True,
                              key="rc_scope", label_visibility="collapsed")

        _target_iid = None
        if _scope == "Un activo":
            st.markdown('<div class="bf-label" style="margin-top:14px;">Asset</div>',
                        unsafe_allow_html=True)
            _rows = _cached_instances()
            _opts = []
            for r in _rows:
                iid = r.get("instance_id") if isinstance(r, dict) else getattr(r, "instance_id", "")
                tag = (r.get("tag") if isinstance(r, dict) else getattr(r, "tag", "")) or iid
                if iid:
                    _opts.append((iid, tag))
            if _opts:
                _label_map = {f"{tag} ({iid})": iid for iid, tag in _opts}
                _sel = st.selectbox("Asset", list(_label_map.keys()),
                                    key="rc_asset", label_visibility="collapsed")
                _target_iid = _label_map.get(_sel)
            else:
                st.info("No assets registered.")

        st.markdown('<div class="bf-label" style="margin-top:14px;">Drafting</div>',
                    unsafe_allow_html=True)
        _use_ai = st.toggle("Use AI for drafting (grounded in the knowledge base)",
                            value=True, key="rc_use_ai",
                            help="If enabled and credentials are available, the AI improves "
                                 "the draft (grounded in courses/standards/manuals). Otherwise "
                                 "the deterministic draft is used (always works).")

    if st.button("📝  Generate report and send it to APPROVAL", type="primary",
                 use_container_width=True, key="rc_gen"):
        from core.briefing_builder import build_all_drafts, build_asset_draft
        _ms = []
        with st.spinner("Generating draft(s) — data + AI drafting…"):
            if _scope == "Todos los activos":
                _ms = build_all_drafts(_period, use_ai=_use_ai)
            elif _target_iid:
                _ms = [build_asset_draft(_target_iid, _period, use_ai=_use_ai)]
            else:
                st.warning("Select an asset.")
        if _ms:
            _ok = [m for m in _ms if m.get("ok")]
            for m in [m for m in _ms if not m.get("ok")]:
                st.warning(f"{m.get('tag', m.get('instance_id'))}: "
                           f"{m.get('status','?')} (draft not created)")
            if _ok:
                st.session_state["rc_cache"] = list_pending()
                st.success(f"✅ {len(_ok)} draft(s) in the queue — approve them below.")

    st.markdown("")

    # ---- Cola de pendientes ----
    if "rc_cache" not in st.session_state:
        st.session_state["rc_cache"] = list_pending()
    _pending = st.session_state["rc_cache"]

    _qh1, _qh2 = st.columns([4, 1])
    with _qh1:
        st.markdown(f'<div class="bf-label">Pending approval '
                    f'<span class="bf-accent">·</span> {len(_pending)} report(s)</div>',
                    unsafe_allow_html=True)
    with _qh2:
        if st.button("🔄 Refresh", key="rc_refresh", use_container_width=True):
            st.session_state["rc_cache"] = list_pending()
            st.session_state.pop("rc_open", None)
            st.rerun()

    if not _pending:
        st.caption("No pending reports. The scheduler leaves them here on each "
                   "asset's day; you can also create one above.")

    _open_iid = st.session_state.get("rc_open", "")
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
                        &nbsp;<span class="bf-row-sev">· {_d.get('period','Semanal')} report · {_st_txt}</span>
                        <div class="bf-chips">
                            <span class="bf-chip">{_d.get('consecutive','') or 'no consecutive'}</span>
                            <span class="bf-chip">draft {( _d.get('created_at','') or '')[:16]}</span>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with _r2:
            if _open_iid == _iid:
                if st.button("✖ Close", key=f"rc_close_{_iid}", use_container_width=True):
                    st.session_state.pop("rc_open", None)
                    st.rerun()
            else:
                if st.button("✅ Approve…", key=f"rc_openbtn_{_iid}", use_container_width=True):
                    st.session_state["rc_open"] = _iid
                    st.rerun()

        if _open_iid == _iid:
            _render_approval_detail(_iid, _tag, _d, me_name)
        st.markdown("")

    st.markdown(
        '<div class="bf-foot">💡&nbsp;Flow: the system leaves each asset\'s draft in '
        '<b>Pending approval</b> on its scheduled day → the specialist reviews/edits it → '
        'signs off <b>Prepared by</b> and <b>Approved by</b> → on approval, the final PDF is '
        'sent to the client through the channels set in <b>Delivery scheduling</b>.</div>',
        unsafe_allow_html=True,
    )


def _render_approval_detail(_iid: str, _tag: str, _d: dict, me_name: str) -> None:
    """Detalle editable de un borrador: recomendaciones + firmas + acciones."""
    from datetime import date as _qdate

    import pandas as _qpd

    from core.briefing_queue import approve_and_send, list_pending
    from core.briefing_recommendations import (
        add_recommendation as _q_add,
        clear_dismissed as _q_cleardis,
        dismiss_proposal as _q_dismiss,
        list_recommendations as _qlist,
        save_recommendations as _qsave,
    )

    _sum = _d.get("summary", "")
    _diag = _d.get("diagnosis", "")
    with st.container(border=True):
        with st.expander("👁 Executive summary and diagnosis (read-only)", expanded=False):
            st.markdown(_sum or "_(no summary)_")
            st.markdown("---")
            st.markdown(_diag or "_(no diagnosis)_")

        st.markdown("**Recommendations** (brought from the previous report — edit, add, "
                    "or delete the ones the client has already done):")

        _q_ss = f"rc_recs_cache_{_iid}"
        if _q_ss not in st.session_state:
            st.session_state[_q_ss] = _qlist(_iid)

        from core.briefing_builder import system_recommendation_proposals
        _props_ss = f"rc_reco_props_{_iid}"
        if _props_ss not in st.session_state:
            try:
                st.session_state[_props_ss] = system_recommendation_proposals(_iid)
            except Exception:
                st.session_state[_props_ss] = []
        _props = st.session_state[_props_ss] or []
        if _props:
            with st.expander(f"💡 System proposals ({len(_props)}) — adopt or dismiss",
                             expanded=True):
                st.caption("The system proposes recommendations from the current findings. "
                           "'Adopt' moves it into your editable list; 'Dismiss' hides it.")
                for _pi, _ptxt in enumerate(_props):
                    _pc1, _pc2, _pc3 = st.columns([0.72, 0.14, 0.14])
                    _pc1.write(_ptxt)
                    if _pc2.button("Adopt", key=f"rc_adopt_{_iid}_{_pi}", use_container_width=True):
                        _q_add(_iid, _ptxt)
                        st.session_state[_q_ss] = _qlist(_iid)
                        st.session_state.pop(_props_ss, None)
                        st.rerun()
                    if _pc3.button("Dismiss", key=f"rc_dismiss_{_iid}_{_pi}", use_container_width=True):
                        _q_dismiss(_iid, _ptxt)
                        st.session_state.pop(_props_ss, None)
                        st.rerun()
        else:
            _pc = st.columns([0.8, 0.2])
            _pc[0].caption("No new system proposals (all adopted or dismissed).")
            if _pc[1].button("Reset dismissed", key=f"rc_reco_reset_{_iid}", use_container_width=True):
                _q_cleardis(_iid)
                st.session_state.pop(_props_ss, None)
                st.rerun()

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
            _qdf, key=f"rc_recs_{_iid}", num_rows="dynamic",
            use_container_width=True, hide_index=True,
            column_config={
                "id": None,
                "Recomendación": st.column_config.TextColumn("Recommendation", width="large", required=True),
                "Fecha de inicio": st.column_config.DateColumn("Start date", format="YYYY-MM-DD",
                                                               default=_qdate.today()),
            },
        )

        _s1, _s2 = st.columns(2)
        with _s1:
            _elab = st.text_input("Prepared by", value="Ángel Daniel Leiva", key=f"rc_elab_{_iid}")
            _elab_rol = st.text_input("Role (prepared)",
                                      value="Senior Machinery Diagnostics Engineer",
                                      key=f"rc_elabr_{_iid}", placeholder="optional")
        with _s2:
            _aprb = st.text_input("Reviewed by", value="Ewdes Andrés Hernández",
                                  key=f"rc_aprb_{_iid}", placeholder="required to approve")
            _aprb_rol = st.text_input("Role (reviewed)", value="Machinery Diagnostics Champion",
                                      key=f"rc_aprbr_{_iid}", placeholder="optional")

        _b2, _b3 = st.columns([1, 1.6])
        with _b2:
            if st.button("👁 PDF preview", key=f"rc_prev_{_iid}", use_container_width=True):
                try:
                    _qsave(_iid, [{"id": r.get("id") or "", "text": r.get("Recomendación") or "",
                                   "started_at": r.get("Fecha de inicio")}
                                  for _, r in _qedited.iterrows()])
                    st.session_state[_q_ss] = _qlist(_iid)
                except Exception:
                    pass
                with st.spinner("Generating preview…"):
                    from core.briefing_builder import build_asset_briefing
                    _pdf, _m = build_asset_briefing(
                        _iid, _d.get("period", "Semanal"), use_ai=False,
                        sections_override={"summary": _sum, "diagnosis": _diag},
                        meta_extra={
                            "prepared_by": _elab or me_name, "reviewed_by": _aprb,
                            "prepared_role": _elab_rol, "reviewed_role": _aprb_rol,
                            "prepared_label": "Preparado por:", "reviewed_label": "Revisado por:",
                            "consecutive": _d.get("consecutive", ""),
                        },
                    )
                st.session_state[f"rc_pdf_{_iid}"] = _pdf
        with _b3:
            if st.button("✅ Approve and send to client", key=f"rc_go_{_iid}",
                         type="primary", use_container_width=True):
                if not (_aprb or "").strip():
                    st.error("'Approved by' is missing — the report must carry both signatures.")
                elif not (_elab or "").strip():
                    st.error("'Prepared by' is missing.")
                else:
                    try:
                        _qsave(_iid, [{"id": r.get("id") or "", "text": r.get("Recomendación") or "",
                                       "started_at": r.get("Fecha de inicio")}
                                      for _, r in _qedited.iterrows()])
                        st.session_state[_q_ss] = _qlist(_iid)
                    except Exception:
                        pass
                    with st.spinner("Approving, generating final PDF and sending to client…"):
                        _res = approve_and_send(_iid, prepared_by=_elab, approved_by=_aprb,
                                                prepared_role=_elab_rol, approved_role=_aprb_rol,
                                                send=True)
                    if _res.get("ok"):
                        _dv = _res.get("delivery") or {}
                        if _dv.get("any_ok"):
                            st.success(f"✅ {_tag} approved and SENT to the client.")
                        else:
                            st.warning(f"Approved, but sending failed or the asset has no "
                                       f"channels configured: {_dv.get('error', _dv)}. "
                                       f"Download the PDF and send it manually.")
                        st.session_state[f"rc_pdf_{_iid}"] = _res.get("pdf")
                        st.session_state.pop("rc_open", None)
                        st.session_state["rc_cache"] = list_pending()
                    else:
                        st.error(f"Could not approve: {_res.get('error')}")

        if st.session_state.get(f"rc_pdf_{_iid}"):
            st.download_button("⬇ Download PDF (latest generated version)",
                               data=st.session_state[f"rc_pdf_{_iid}"],
                               file_name=f"Report_{_tag}_{_d.get('period','Semanal')}.pdf",
                               mime="application/pdf", key=f"rc_dl_{_iid}",
                               use_container_width=True)


# =================================================================
# VISTA 2 — DELIVERY SCHEDULING (por Cliente → Activo)
# =================================================================
def render_delivery() -> None:
    """Programación de envíos centralizada: correo(s), WhatsApp(s), horario del
    briefing (semanal/mensual) y aviso automático por alarma — por activo,
    agrupado por cliente. Reemplaza la config dispersa de Machinery Library."""
    from core.briefing_queue import get_schedule, save_schedule
    from core.instance_state import get_instance, update_instance_header

    _rows = _cached_instances()
    if not _rows:
        st.info("No assets registered. Create one in Machinery Library first.")
        return

    # Agrupar por cliente
    groups: dict = {}
    for r in _rows:
        iid = r.get("instance_id") if isinstance(r, dict) else getattr(r, "instance_id", "")
        if not iid:
            continue
        tag = (r.get("tag") if isinstance(r, dict) else getattr(r, "tag", "")) or iid
        client = (r.get("client") if isinstance(r, dict) else getattr(r, "client", "")) or "— Sin cliente —"
        groups.setdefault(client, []).append((iid, tag))

    # Filtro por cliente
    _clients = ["All clients"] + sorted(groups.keys())
    _fc1, _fc2 = st.columns([2, 3])
    with _fc1:
        _fclient = st.selectbox("Client", _clients, key="rc_dc_client")
    with _fc2:
        st.caption("Set here who receives each asset's reports and when. This is the "
                   "single source the schedulers read — no longer in Machinery Library.")

    for client, assets in sorted(groups.items()):
        if _fclient != "All clients" and client != _fclient:
            continue
        st.markdown(f'<div class="dc-client">{client}</div>'
                    f'<div class="dc-meta">{len(assets)} asset(s)</div>',
                    unsafe_allow_html=True)
        st.markdown("")

        for iid, tag in sorted(assets, key=lambda x: x[1]):
            inst = get_instance(iid)
            if inst is None:
                continue
            _cfg = get_schedule(iid) or {}
            _emails = getattr(inst, "client_email", "") or ""
            _wa = getattr(inst, "whatsapp_number", "") or ""
            _alarm = bool(getattr(inst, "alarm_send_enabled", False))

            with st.container(border=True):
                st.markdown(f'<span class="dc-asset-tag">{tag}</span> '
                            f'<span class="dc-asset-sub">· {iid}</span>',
                            unsafe_allow_html=True)

                # Recipients
                _rc1, _rc2 = st.columns(2)
                with _rc1:
                    _email_in = st.text_input(
                        "Email recipient(s)", value=_emails, key=f"rc_dc_email_{iid}",
                        placeholder="a@client.com, b@client.com",
                        help="Comma / semicolon / newline separated for multiple.")
                with _rc2:
                    _wa_in = st.text_input(
                        "WhatsApp number(s)", value=_wa, key=f"rc_dc_wa_{iid}",
                        placeholder="573001234567, 573007654321",
                        help="E.164 without '+'. Comma separated for multiple.")

                # Schedule (weekly/monthly briefing)
                _sc1, _sc2, _sc3, _sc4 = st.columns([1.1, 2, 1, 1.2])
                with _sc1:
                    _en = st.toggle("Scheduled", value=bool(_cfg.get("enabled")),
                                    key=f"rc_dc_en_{iid}",
                                    help="Auto-generate the report to the approval queue on the "
                                         "day/hour below.")
                with _sc2:
                    _days_sel = st.multiselect(
                        "Day(s)", _DIAS_EN,
                        default=[_DIAS_EN[d] for d in (_cfg.get("days") or [0]) if 0 <= int(d) <= 6],
                        key=f"rc_dc_days_{iid}")
                with _sc3:
                    _hour_sel = st.selectbox("Hour", list(range(24)),
                                             index=int(_cfg.get("hour", 5)),
                                             format_func=lambda h: f"{h:02d}:00",
                                             key=f"rc_dc_hour_{iid}")
                with _sc4:
                    _per_sel = st.selectbox(
                        "Period", ["Semanal", "Mensual"],
                        index=(1 if str(_cfg.get("period", "")).startswith("Mensual") else 0),
                        key=f"rc_dc_per_{iid}")

                _ac1, _ac2 = st.columns([2, 1])
                with _ac1:
                    _alarm_in = st.toggle(
                        "Auto-send on alarm / danger (immediate 1-page report)",
                        value=_alarm, key=f"rc_dc_alarm_{iid}")
                with _ac2:
                    if st.button("💾 Save", key=f"rc_dc_save_{iid}", use_container_width=True):
                        _new_cfg = {
                            "enabled": bool(_en),
                            "days": [_DIAS_EN.index(d) for d in (_days_sel or ["Monday"])],
                            "hour": int(_hour_sel),
                            "period": _per_sel,
                        }
                        _ok1 = save_schedule(iid, _new_cfg)
                        _ok2 = update_instance_header(
                            iid,
                            client_email=_email_in.strip(),
                            whatsapp_number=_wa_in.strip(),
                            alarm_send_enabled=bool(_alarm_in),
                            # mantener consistencia del flag legacy de "envío programado"
                            report_send_enabled=bool(_en),
                        )
                        if _ok1 and _ok2:
                            st.success(f"Saved delivery config for {tag}.")
                            # invalidar cache de instancias para reflejar cambios
                            try:
                                _cached_instances.clear()  # type: ignore[attr-defined]
                            except Exception:
                                pass
                        else:
                            st.error("Could not save (check the asset exists).")
            st.markdown("")
