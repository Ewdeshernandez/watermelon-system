"""
core/report_center_ui.py
========================

Render helpers del **Report Center** (pestaña "Report Center" del nav).

Un solo lugar para todo lo de reportes:
  · render_approval(me_name)  → generar borrador + cola de PENDIENTES +
    detalle de revisión/firma/aprobación. Los firmantes se pre-cargan de la
    config del activo (get_signers).
  · render_delivery()         → configuración POR MÁQUINA (como la de envío):
    correo(s), WhatsApp(s), los distintos reportes que le llegan (Semanal /
    Mensual, cada uno con su día y hora), aviso por alarma, y quién ELABORA /
    quién REVISA-aprueba. Un solo lugar por cliente/activo. Reemplaza la
    config dispersa que estaba en Machinery Library.

No hace set_page_config ni auth — eso lo hace la página que lo importa.
"""
from __future__ import annotations

import streamlit as st

_DIAS_EN = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
            "Saturday", "Sunday"]
_DIAS_ABBR = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
_REPORT_TYPES = ["Semanal", "Mensual"]


# -----------------------------------------------------------------
# CSS — look industrial / internacional (steel navy + amber accent)
# -----------------------------------------------------------------
def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600&display=swap');

        :root{
          --rc-ink:#0b1f3a; --rc-ink2:#3a4c66; --rc-mut:#8090a6;
          --rc-line:#e2e8f2; --rc-panel:#ffffff; --rc-panel2:#f4f7fb;
          --rc-navy:#12305e; --rc-steel:#274b7d;
          --rc-amber:#e8890c; --rc-amber-soft:#fff4e2;
          --rc-ok:#1f9d55; --rc-warn:#e8890c; --rc-dang:#dc3545; --rc-off:#8a97a8;
          --rc-shadow:0 1px 2px rgba(11,31,58,.05), 0 8px 24px rgba(11,31,58,.06);
        }
        /* Banda de encabezado por vista */
        .rc-band{
          position:relative; border-radius:16px; padding:18px 22px; margin:2px 0 18px;
          background:linear-gradient(120deg,#0e2547 0%,#12305e 46%,#274b7d 100%);
          color:#eaf1fb; overflow:hidden; box-shadow:0 10px 30px rgba(12,32,78,.22);
        }
        .rc-band::after{content:"";position:absolute;inset:0;
          background:repeating-linear-gradient(135deg,rgba(255,255,255,.045) 0 2px,transparent 2px 9px);
          pointer-events:none;}
        .rc-band .kick{font:600 10.5px/1 'IBM Plex Mono',monospace;letter-spacing:.28em;
          text-transform:uppercase;color:#8fb4e6;}
        .rc-band h2{margin:6px 0 2px;font:700 22px/1.15 'IBM Plex Sans',sans-serif;letter-spacing:-.4px;}
        .rc-band p{margin:0;color:#b9cbe6;font:500 13px/1.4 'IBM Plex Sans',sans-serif;max-width:70ch;}
        .rc-band .rc-strip{position:absolute;top:0;right:0;height:100%;width:6px;
          background:linear-gradient(180deg,var(--rc-amber),#c56f05);}
        .rc-kpis{display:flex;gap:10px;flex-wrap:wrap;margin-top:14px;}
        .rc-kpi{background:rgba(255,255,255,.09);border:1px solid rgba(255,255,255,.14);
          border-radius:11px;padding:8px 14px;min-width:96px;}
        .rc-kpi .n{font:700 20px/1 'IBM Plex Sans',sans-serif;display:flex;align-items:center;gap:8px;}
        .rc-kpi .l{font:500 10.5px/1.2 'IBM Plex Sans';color:#a9c1e2;margin-top:4px;
          text-transform:uppercase;letter-spacing:.05em;}

        .bf-label{font:700 10.5px/1 'IBM Plex Mono',monospace;letter-spacing:.18em;
          text-transform:uppercase;color:var(--rc-mut);margin:0 0 9px 2px;}
        .bf-label .bf-accent{color:var(--rc-amber);}

        /* Radios como segmented control (más sobrio, steel) */
        div[data-testid="stRadio"] [role="radiogroup"]{gap:8px;}
        div[data-testid="stRadio"] [role="radiogroup"] > label{
          border:1px solid var(--rc-line);background:var(--rc-panel2);border-radius:11px;
          padding:9px 18px;margin:0 !important;transition:all .16s ease;cursor:pointer;}
        div[data-testid="stRadio"] [role="radiogroup"] > label:hover{border-color:#c4d2e6;background:#fff;}
        div[data-testid="stRadio"] [role="radiogroup"] > label > div:first-child{display:none !important;}
        div[data-testid="stRadio"] [role="radiogroup"] > label p{font-weight:600;color:var(--rc-ink2);font-size:.92rem;}
        div[data-testid="stRadio"] [role="radiogroup"] > label:has(input:checked){
          border-color:#12305e;background:linear-gradient(180deg,#12305e 0%,#1c4478 100%);
          box-shadow:0 6px 16px rgba(18,48,94,.22);}
        div[data-testid="stRadio"] [role="radiogroup"] > label:has(input:checked) p{color:#f2f7ff;}

        /* Fila de la cola (approval) */
        .bf-row{display:flex;align-items:center;gap:14px;
          border:1px solid var(--rc-line);border-left:4px solid var(--bf-dot,#8a97a8);
          background:linear-gradient(180deg,#ffffff 0%,#fafcff 100%);
          border-radius:14px;padding:14px 18px;box-shadow:var(--rc-shadow);}
        .bf-row-main{flex:1 1 auto;min-width:0;}
        .bf-row-tag{font:800 1.04rem/1 'IBM Plex Sans';color:var(--rc-ink);letter-spacing:-.01em;}
        .bf-row-sev{font-size:.85rem;font-weight:600;color:var(--rc-ink2);}
        .bf-chips{display:flex;flex-wrap:wrap;gap:7px;margin-top:8px;}
        .bf-chip{border:1px solid var(--rc-line);background:var(--rc-panel2);color:var(--rc-ink2);
          border-radius:999px;padding:3px 11px;font:600 .76rem/1.3 'IBM Plex Mono',monospace;}
        .bf-dot{width:11px;height:11px;border-radius:50%;flex:0 0 auto;
          box-shadow:0 0 0 4px var(--bf-dot-soft,rgba(138,151,168,.16));}
        .bf-foot{display:flex;align-items:flex-start;gap:9px;border:1px solid var(--rc-line);
          background:var(--rc-panel2);border-radius:12px;padding:12px 16px;margin-top:8px;
          color:var(--rc-ink2);font-size:.86rem;line-height:1.5;}

        /* Delivery center */
        .dc-client{display:flex;align-items:center;gap:10px;margin:6px 0 2px;}
        .dc-client .cn{font:800 1.12rem/1 'IBM Plex Sans';color:var(--rc-ink);letter-spacing:-.01em;}
        .dc-client .cm{font:600 .78rem/1 'IBM Plex Mono',monospace;color:var(--rc-mut);
          background:var(--rc-panel2);border:1px solid var(--rc-line);border-radius:999px;padding:3px 10px;}
        .dc-client .rule{flex:1;height:1px;background:linear-gradient(90deg,var(--rc-line),transparent);}
        .dc-asset{display:flex;align-items:baseline;gap:10px;margin-bottom:2px;}
        .dc-asset .tg{font:800 1.02rem/1 'IBM Plex Sans';color:var(--rc-ink);
          padding:2px 0;border-bottom:2px solid var(--rc-amber);}
        .dc-asset .sb{font:500 .8rem/1 'IBM Plex Mono',monospace;color:var(--rc-mut);}
        .dc-sec{font:700 10px/1 'IBM Plex Mono',monospace;letter-spacing:.16em;text-transform:uppercase;
          color:var(--rc-steel);margin:6px 0 2px;}
        .dc-badge{display:inline-flex;align-items:center;gap:7px;font:600 .78rem/1 'IBM Plex Sans';
          border-radius:8px;padding:5px 11px;border:1px solid var(--rc-line);background:var(--rc-panel2);color:var(--rc-ink2);}
        .dc-badge.wk{background:#eaf1fb;border-color:#c8dcf4;color:#1c4478;}
        .dc-badge.mo{background:var(--rc-amber-soft);border-color:#f3d9ad;color:#9a5b06;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _band(kick: str, title: str, sub: str, kpis=None) -> None:
    _k = ""
    if kpis:
        cells = "".join(
            f'<div class="rc-kpi"><div class="n">{n}</div><div class="l">{l}</div></div>'
            for n, l in kpis)
        _k = f'<div class="rc-kpis">{cells}</div>'
    st.markdown(
        f'<div class="rc-band"><span class="rc-strip"></span>'
        f'<div class="kick">{kick}</div><h2>{title}</h2><p>{sub}</p>{_k}</div>',
        unsafe_allow_html=True,
    )


def _dot(sev: str) -> str:
    c = {"ok": "#1f9d55", "warn": "#e8890c", "dang": "#dc3545", "off": "#8a97a8"}.get(sev, "#8a97a8")
    return f'<span style="color:{c};font-size:20px;line-height:0;">●</span>'


def _cached_instances():
    from core.instance_state import get_instances_version, list_instances

    @st.cache_data(show_spinner=False)
    def _inner(_v: int):
        return list_instances() or []

    return _inner(get_instances_version())


# =================================================================
# VISTA 1 — APPROVAL
# =================================================================
def render_approval(me_name: str = "") -> None:
    from core.briefing_queue import list_pending

    if "rc_cache" not in st.session_state:
        st.session_state["rc_cache"] = list_pending()
    _pending = st.session_state["rc_cache"]

    _band("Report Center · Approval", "Review, sign &amp; send",
          "Generate the draft, review the recommendations, sign off and approve — "
          "the final signed PDF is delivered through each asset's configured channels.",
          kpis=[(f'{_dot("warn")} {len(_pending)}', "Pending approval")])

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
                                 "the draft (grounded in courses/standards/manuals).")

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
                <div class="bf-row" style="--bf-dot:#274b7d;">
                    <span class="bf-dot" style="background:#274b7d;"></span>
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
    from datetime import date as _qdate

    import pandas as _qpd

    from core.briefing_queue import approve_and_send, get_signers, list_pending
    from core.briefing_recommendations import (
        add_recommendation as _q_add,
        clear_dismissed as _q_cleardis,
        dismiss_proposal as _q_dismiss,
        list_recommendations as _qlist,
        save_recommendations as _qsave,
    )

    _sum = _d.get("summary", "")
    _diag = _d.get("diagnosis", "")
    _sg = get_signers(_iid)  # firmantes por defecto del activo
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
                st.caption("The system proposes recommendations from the current findings.")
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

        st.markdown('<div class="bf-label" style="margin-top:6px;">Signatures '
                    '<span class="bf-accent">·</span> defaults from the asset config</div>',
                    unsafe_allow_html=True)
        _s1, _s2 = st.columns(2)
        with _s1:
            _elab = st.text_input("Prepared by", value=_sg.get("prepared_by", ""), key=f"rc_elab_{_iid}")
            _elab_rol = st.text_input("Role (prepared)", value=_sg.get("prepared_role", ""),
                                      key=f"rc_elabr_{_iid}", placeholder="optional")
        with _s2:
            _aprb = st.text_input("Reviewed by", value=_sg.get("reviewed_by", ""),
                                  key=f"rc_aprb_{_iid}", placeholder="required to approve")
            _aprb_rol = st.text_input("Role (reviewed)", value=_sg.get("reviewed_role", ""),
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
# VISTA 2 — DELIVERY SCHEDULING (config por Cliente → Activo)
# =================================================================
def render_delivery() -> None:
    """Config por máquina: destinatarios, los distintos reportes que le llegan
    (Semanal/Mensual con su día y hora), aviso por alarma, y quién elabora /
    revisa-aprueba. Un solo lugar (antes disperso en Machinery Library)."""
    from core.briefing_queue import get_schedules, get_signers, save_schedules, save_signers
    from core.instance_state import get_instance, update_instance_header

    _rows = _cached_instances()
    if not _rows:
        st.info("No assets registered. Create one in Machinery Library first.")
        return

    groups: dict = {}
    for r in _rows:
        iid = r.get("instance_id") if isinstance(r, dict) else getattr(r, "instance_id", "")
        if not iid:
            continue
        tag = (r.get("tag") if isinstance(r, dict) else getattr(r, "tag", "")) or iid
        client = (r.get("client") if isinstance(r, dict) else getattr(r, "client", "")) or "— Sin cliente —"
        groups.setdefault(client, []).append((iid, tag))

    # Celebración tras guardar (sandías subiendo + mensaje), una sola vez.
    _saved = st.session_state.pop("rc_saved_tag", None)
    if _saved:
        _celebrate_saved(_saved)

    _n_assets = sum(len(a) for a in groups.values())
    _band("Report Center · Delivery scheduling", "Per-machine delivery config",
          "Pick a machine and set who receives its reports, which reports it gets "
          "(weekly / monthly, each with its own day &amp; time), alarm auto-send, and "
          "who prepares and who approves. This is the single source the schedulers read.",
          kpis=[(f'{_dot("ok")} {len(groups)}', "Clients"),
                (f'{_dot("warn")} {_n_assets}', "Assets")])

    # Catálogo de clientes para ligar activos: registrados + los ya usados.
    _reg = []
    try:
        from core.clients import list_clients
        _reg = [c.display_name for c in (list_clients() or []) if c.display_name]
    except Exception:
        _reg = []
    _existing = [c for c in groups.keys() if c and c != "— Sin cliente —"]
    _client_opts = sorted(set(_reg) | set(_existing))

    _clients = ["All clients"] + sorted(groups.keys())
    _fc1, _fc2 = st.columns([2, 1])
    with _fc1:
        _fclient = st.selectbox("Filter by client", _clients, key="rc_dc_client")
    with _fc2:
        _tags_all = sorted({t for a in groups.values() for _, t in a})
        _ftag = st.selectbox("Filter by asset", ["All assets"] + _tags_all, key="rc_dc_tag")

    if "— Sin cliente —" in groups:
        st.info("Some assets are not linked to a client yet — set the **Client** field "
                "in each card below so you can filter and group them.")

    for client, assets in sorted(groups.items()):
        if _fclient != "All clients" and client != _fclient:
            continue
        _visible = [(iid, tag) for iid, tag in assets
                    if _ftag == "All assets" or tag == _ftag]
        if not _visible:
            continue
        st.markdown(f'<div class="dc-client"><span class="cn">{client}</span>'
                    f'<span class="cm">{len(_visible)} asset(s)</span>'
                    f'<span class="rule"></span></div>', unsafe_allow_html=True)
        st.markdown("")

        for iid, tag in sorted(_visible, key=lambda x: x[1]):
            _render_delivery_asset(iid, tag, _client_opts, get_instance, get_schedules,
                                   get_signers, save_schedules, save_signers,
                                   update_instance_header)
            st.markdown("")


def _split_items(raw: str):
    import re
    return [p for p in re.split(r"[,;\n\r\t ]+", str(raw or "")) if p.strip()]


def _valid_email(v: str) -> bool:
    import re
    return bool(re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]{2,}", (v or "").strip()))


def _valid_phone(v: str) -> bool:
    import re
    return bool(re.fullmatch(r"\d{8,15}", (v or "").strip().replace("+", "").replace(" ", "")))


def _multi_input(ss_key: str, init_vals, validator, placeholder: str, add_label: str):
    """Editor de lista: una casilla por ítem, ✓ verde si válido / ✕ rojo si no,
    botón ✕ para eliminar y '+ añadir'. Devuelve (valid_list, n_invalid)."""
    ids_key = f"{ss_key}__ids"
    nxt_key = f"{ss_key}__next"
    if ids_key not in st.session_state:
        vals = list(init_vals) or [""]
        st.session_state[ids_key] = list(range(len(vals)))
        st.session_state[nxt_key] = len(vals)
        for i, v in enumerate(vals):
            st.session_state[f"{ss_key}__v_{i}"] = v

    ids = st.session_state[ids_key]
    valid_list, n_invalid = [], 0
    for _id in list(ids):
        vkey = f"{ss_key}__v_{_id}"
        st.session_state.setdefault(vkey, "")
        c1, c2, c3 = st.columns([8, 0.9, 0.9])
        c1.text_input("item", key=vkey, placeholder=placeholder,
                      label_visibility="collapsed")
        cur = (st.session_state.get(vkey, "") or "").strip()
        if not cur:
            c2.markdown('<div style="padding-top:8px;color:#8a97a8;">–</div>',
                        unsafe_allow_html=True)
        elif validator(cur):
            c2.markdown('<div style="padding-top:6px;font-size:18px;color:#1f9d55;">✔</div>',
                        unsafe_allow_html=True)
            valid_list.append(cur)
        else:
            c2.markdown('<div style="padding-top:6px;font-size:18px;color:#dc3545;">✕</div>',
                        unsafe_allow_html=True)
            n_invalid += 1
        if c3.button("✕", key=f"{ss_key}__rm_{_id}", help="Remove"):
            ids.remove(_id)
            st.session_state.pop(vkey, None)
            st.rerun()

    if st.button(add_label, key=f"{ss_key}__add"):
        nid = st.session_state[nxt_key]
        st.session_state[nxt_key] = nid + 1
        st.session_state[ids_key] = ids + [nid]
        st.session_state[f"{ss_key}__v_{nid}"] = ""
        st.rerun()

    return valid_list, n_invalid


def _h12(h: int) -> str:
    """Hora 0-23 → formato 12h, ej. 6 → '6:00 AM', 18 → '6:00 PM'."""
    h = int(h) % 24
    ampm = "AM" if h < 12 else "PM"
    hr = h % 12 or 12
    return f"{hr}:00 {ampm}"


def _slot_editor(ss_key: str, init_slots):
    """Editor de horarios: una fila por (día + hora), cada día su propia hora,
    con ✕ para eliminar y '+ Add day/time'. Hora en formato 12h. Devuelve la
    lista [[dow, hour], …]."""
    ids_key = f"{ss_key}__ids"
    nxt_key = f"{ss_key}__next"
    if ids_key not in st.session_state:
        slots = [list(s) for s in (init_slots or [])] or [[0, 6]]
        st.session_state[ids_key] = list(range(len(slots)))
        st.session_state[nxt_key] = len(slots)
        for i, (d, h) in enumerate(slots):
            st.session_state[f"{ss_key}__d_{i}"] = _DIAS_EN[int(d) % 7]
            st.session_state[f"{ss_key}__h_{i}"] = int(h)

    ids = st.session_state[ids_key]
    out = []
    for _id in list(ids):
        dkey, hkey = f"{ss_key}__d_{_id}", f"{ss_key}__h_{_id}"
        st.session_state.setdefault(dkey, _DIAS_EN[0])
        st.session_state.setdefault(hkey, 6)
        c1, c2, c3 = st.columns([3.2, 2.2, 0.8])
        c1.selectbox("Day", _DIAS_EN, key=dkey, label_visibility="collapsed")
        c2.selectbox("Time", list(range(24)), key=hkey, format_func=_h12,
                     label_visibility="collapsed")
        if c3.button("✕", key=f"{ss_key}__rm_{_id}", help="Remove this day/time"):
            ids.remove(_id)
            st.rerun()
        out.append([_DIAS_EN.index(st.session_state[dkey]), int(st.session_state[hkey])])

    if st.button("＋ Add day / time", key=f"{ss_key}__add"):
        nid = st.session_state[nxt_key]
        st.session_state[nxt_key] = nid + 1
        st.session_state[ids_key] = ids + [nid]
        st.session_state[f"{ss_key}__d_{nid}"] = _DIAS_EN[0]
        st.session_state[f"{ss_key}__h_{nid}"] = 6
        st.rerun()

    return out


def _celebrate_saved(tag: str) -> None:
    """Sandías subiendo + mensaje de guardado (se muestra una vez tras salvar)."""
    st.success(f"✅ Se han guardado correctamente los cambios de **{tag}**.")
    try:
        st.toast(f"🍉 Guardado: {tag}", icon="🍉")
    except Exception:
        pass
    _mel = "".join(
        f'<span style="left:{4 + i * 8}%;animation-delay:{(i % 6) * .12:.2f}s;'
        f'font-size:{22 + (i % 4) * 8}px;">🍉</span>' for i in range(12))
    st.markdown(
        f"""
        <style>
        @keyframes rcRise {{
          0%   {{ transform: translateY(20vh) rotate(0deg); opacity:0; }}
          15%  {{ opacity:1; }}
          100% {{ transform: translateY(-96vh) rotate(220deg); opacity:0; }}
        }}
        .rc-mel {{ position:fixed; inset:0; pointer-events:none; z-index:9999; overflow:hidden; }}
        .rc-mel span {{ position:absolute; bottom:-40px; animation:rcRise 2.1s ease-in forwards; }}
        </style>
        <div class="rc-mel">{_mel}</div>
        """,
        unsafe_allow_html=True,
    )


def _render_delivery_asset(iid, tag, client_opts, get_instance, get_schedules, get_signers,
                           save_schedules, save_signers, update_instance_header) -> None:
    inst = get_instance(iid)
    if inst is None:
        return
    _entries = {e.get("period", "Semanal"): e for e in (get_schedules(iid) or [])}
    _sg = get_signers(iid)
    _emails = getattr(inst, "client_email", "") or ""
    _wa = getattr(inst, "whatsapp_number", "") or ""
    _alarm = bool(getattr(inst, "alarm_send_enabled", False))
    _cur_client = (getattr(inst, "client", "") or "").strip()

    with st.container(border=True):
        st.markdown(f'<div class="dc-asset"><span class="tg">{tag}</span>'
                    f'<span class="sb">· {iid}</span></div>', unsafe_allow_html=True)

        # --- Client link ---
        st.markdown('<div class="dc-sec">Client</div>', unsafe_allow_html=True)
        _NEW = "➕ New client…"
        _opts = list(dict.fromkeys(([_cur_client] if _cur_client else [])
                                   + client_opts + [_NEW]))
        _cc1, _cc2 = st.columns([2, 2])
        with _cc1:
            _client_sel = st.selectbox("Linked client", _opts,
                                       index=(_opts.index(_cur_client) if _cur_client in _opts else 0),
                                       key=f"rc_dc_clientsel_{iid}",
                                       label_visibility="collapsed")
        _client_val = _cur_client
        with _cc2:
            if _client_sel == _NEW:
                _client_val = st.text_input("New client name", value="",
                                            key=f"rc_dc_clientnew_{iid}",
                                            placeholder="Client name",
                                            label_visibility="collapsed").strip()
            else:
                _client_val = _client_sel

        # --- Recipients (una casilla por ítem, con validación ✓/✕) ---
        st.markdown('<div class="dc-sec">Email recipients '
                    '<span style="color:#8a97a8;font-weight:500;text-transform:none;'
                    'letter-spacing:0;">· one box each · ✔ valid · ✕ removes</span></div>',
                    unsafe_allow_html=True)
        _emails_valid, _em_bad = _multi_input(
            f"rc_dc_em_{iid}", _split_items(_emails), _valid_email,
            "name@client.com", "＋ Add email")

        st.markdown('<div class="dc-sec">WhatsApp numbers '
                    '<span style="color:#8a97a8;font-weight:500;text-transform:none;'
                    'letter-spacing:0;">· digits only, country code, no +</span></div>',
                    unsafe_allow_html=True)
        _wa_valid, _wa_bad = _multi_input(
            f"rc_dc_wa_{iid}", _split_items(_wa), _valid_phone,
            "573001234567", "＋ Add number")
        try:
            from core.whatsapp_sender import whatsapp_status
            _ws = whatsapp_status()
        except Exception:
            _ws = {"configured": False}
        if _wa_valid and not _ws.get("configured"):
            st.warning("⚠️ WhatsApp channel is **not set up yet** — these numbers are saved, "
                       "but reports won't be delivered by WhatsApp until the Meta WhatsApp "
                       "Business API is configured (one-time, in secrets). Email is unaffected.")
        elif _wa_valid and _ws.get("configured"):
            st.caption(f"WhatsApp ready · mode: {_ws.get('mode','—')}"
                       + (f" · template {_ws.get('template_name')}" if _ws.get('template_name') else ""))

        # --- Reports this machine gets (one row per type; per-day time) ---
        st.markdown('<div class="dc-sec">Reports this machine receives '
                    '<span style="color:#8a97a8;font-weight:500;text-transform:none;'
                    'letter-spacing:0;">· each day can have its own time</span></div>',
                    unsafe_allow_html=True)
        _draft_entries = {}
        for _per in _REPORT_TYPES:
            _cur = _entries.get(_per, {})
            _badge = "wk" if _per == "Semanal" else "mo"
            _plabel = "Weekly" if _per == "Semanal" else "Monthly"
            _init_slots = _cur.get("slots") or [[d, int(_cur.get("hour", 6))]
                                                for d in (_cur.get("days") or [0])]
            _tc1, _tc2 = st.columns([1.4, 4])
            with _tc1:
                st.markdown(f'<span class="dc-badge {_badge}">{_dot("ok")} {_plabel}</span>',
                            unsafe_allow_html=True)
                _en = st.toggle("Enabled", value=bool(_cur.get("enabled")),
                                key=f"rc_dc_en_{iid}_{_per}")
            with _tc2:
                if _en:
                    _slots = _slot_editor(f"rc_dc_slots_{iid}_{_per}", _init_slots)
                else:
                    st.caption("Off — enable to set day(s) and time(s).")
                    _slots = [list(s) for s in _init_slots]
            _draft_entries[_per] = {"enabled": bool(_en), "period": _per, "slots": _slots}

        # --- Daily quick report (1-page Live, DIRECT — no approval) ---
        from core.briefing_queue import get_quick_schedule
        _q = get_quick_schedule(iid)
        st.markdown('<div class="dc-sec">Daily quick report '
                    '<span style="color:#8a97a8;font-weight:500;text-transform:none;'
                    'letter-spacing:0;">· 1-page Live snapshot · sent DIRECT (no approval) '
                    'by email &amp; WhatsApp</span></div>', unsafe_allow_html=True)
        _qc1, _qc2 = st.columns([1.4, 4])
        with _qc1:
            st.markdown(f'<span class="dc-badge" style="background:#e8f6ee;border-color:#bfe6cd;'
                        f'color:#1b6b3a;">{_dot("ok")} Quick</span>', unsafe_allow_html=True)
            _q_en = st.toggle("Enabled", value=bool(_q.get("enabled")), key=f"rc_dc_qen_{iid}")
        with _qc2:
            if _q_en:
                _q_slots = _slot_editor(f"rc_dc_qslots_{iid}", _q.get("slots") or [[0, 7]])
            else:
                st.caption("Off — enable to send the 1-page live report on set day(s)/time(s).")
                _q_slots = [list(s) for s in (_q.get("slots") or [])]

        # --- Alarm ---
        st.markdown('<div class="dc-sec">Alarm</div>', unsafe_allow_html=True)
        _alarm_in = st.toggle("Auto-send on alarm / danger (immediate 1-page report, checked every 15 min)",
                              value=_alarm, key=f"rc_dc_alarm_{iid}")

        # --- Signers (who prepares / who approves) ---
        st.markdown('<div class="dc-sec">Signatures — who prepares · who approves</div>',
                    unsafe_allow_html=True)
        _sc1, _sc2 = st.columns(2)
        with _sc1:
            _prep = st.text_input("Prepared by", value=_sg.get("prepared_by", ""),
                                  key=f"rc_dc_prep_{iid}")
            _prep_r = st.text_input("Role (prepared)", value=_sg.get("prepared_role", ""),
                                    key=f"rc_dc_prepr_{iid}", placeholder="optional")
        with _sc2:
            _rev = st.text_input("Reviewed / approved by", value=_sg.get("reviewed_by", ""),
                                 key=f"rc_dc_rev_{iid}")
            _rev_r = st.text_input("Role (reviewed)", value=_sg.get("reviewed_role", ""),
                                   key=f"rc_dc_revr_{iid}", placeholder="optional")

        # --- Save ---
        if _em_bad or _wa_bad:
            _bits = []
            if _em_bad:
                _bits.append(f"{_em_bad} email(s) missing @ or domain")
            if _wa_bad:
                _bits.append(f"{_wa_bad} phone(s) not digits-only")
            st.warning("Fix or remove: " + " · ".join(_bits)
                       + ". Invalid entries won't be saved.")
        if st.button("💾 Save machine config", key=f"rc_dc_save_{iid}",
                     type="primary", use_container_width=True):
            if _em_bad or _wa_bad:
                st.error("There are invalid entries (marked ✕). Fix or remove them "
                         "before saving.")
            else:
                from core.briefing_queue import save_quick_schedule
                _ok1 = save_schedules(iid, list(_draft_entries.values()))
                _okq = save_quick_schedule(iid, {"enabled": bool(_q_en), "slots": _q_slots})
                _ok2 = save_signers(iid, {
                    "prepared_by": _prep, "prepared_role": _prep_r,
                    "reviewed_by": _rev, "reviewed_role": _rev_r,
                })
                _any_sched = any(e["enabled"] for e in _draft_entries.values())
                _ok3 = update_instance_header(
                    iid, client_email=", ".join(_emails_valid),
                    whatsapp_number=", ".join(_wa_valid),
                    alarm_send_enabled=bool(_alarm_in),
                    report_send_enabled=bool(_any_sched),
                    client=(_client_val or "").strip(),
                )
                if _ok1 and _okq and _ok2 and _ok3:
                    st.session_state["rc_saved_tag"] = tag
                    st.rerun()
                else:
                    st.error("Could not save (check the asset exists).")
