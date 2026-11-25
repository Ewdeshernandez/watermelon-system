"""
core/client_report_archive.py
=============================

Archivo de reportes — lo que ve el CLIENTE (Reports, rol client) y la vista
de administración del archivo en Report Center (admin/specialist).

  · Cliente: sus reportes Semanales y Mensuales aprobados, agrupados por mes,
    con filtros (tipo, máquina, año), estado con punto de color y descarga del
    PDF bajo demanda (no se leen 20 PDF de 5 MB para pintar la lista).
  · Analista: tabla de todo el archivo + carga de reportes ANTERIORES (los
    semanales enviados antes del 2026-11-23 no se archivaban) + retirar uno
    cargado por error.

La visibilidad la decide core.reports_archive._can_view (un cliente solo ve
los reportes de SU cliente). Este módulo no relaja ese filtro.
"""
from __future__ import annotations

import html as _html
import re
from datetime import date, datetime
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio",
         "agosto", "septiembre", "octubre", "noviembre", "diciembre"]

KIND_WEEKLY = "Semanal"
KIND_MONTHLY = "Mensual"
KIND_TECH = "Técnico"

_SEV_COLORS = {"ok": "#1f9d55", "warn": "#e8890c", "dang": "#dc3545",
               "info": "#2563eb", "off": "#8090a6"}


# ---------------------------------------------------------------------------
# Helpers puros (testeados en tests/test_client_report_archive.py)
# ---------------------------------------------------------------------------

def _rm(sc: Dict[str, Any]) -> Dict[str, Any]:
    rm = sc.get("report_meta")
    return rm if isinstance(rm, dict) else {}


def report_kind(sc: Dict[str, Any]) -> str:
    """Semanal / Mensual / Técnico. Usa report_meta.period (desde 2026-11-25);
    para los archivados antes, lo deduce del título/consecutivo/notas."""
    rm = _rm(sc)
    period = str(rm.get("period", "") or "").lower()
    blob = " ".join(str(x or "") for x in (
        period, rm.get("asset_class"), rm.get("consecutive"),
        sc.get("extra_notes"))).lower()
    if "mensual" in blob or "monthly" in blob:
        return KIND_MONTHLY
    if "semanal" in blob or "weekly" in blob:
        return KIND_WEEKLY
    return KIND_TECH


def sev_class(text: str) -> str:
    """Texto de severidad ejecutiva → clase de color (ok/warn/dang/off/info)."""
    t = (text or "").strip().lower()
    if not t or t in ("—", "-"):
        return "off"
    if any(k in t for k in ("detenid", "fuera de línea", "fuera de linea",
                            "offline", "sin datos", "parada", "sin conexión")):
        return "off"
    if any(k in t for k in ("peligro", "crít", "crit", "danger", "severo",
                            "alto riesgo")):
        return "dang"
    if any(k in t for k in ("alerta", "alarma", "precauc", "vigil", "atención",
                            "atencion", "warning", "alert")):
        return "warn"
    if any(k in t for k in ("normal", "satisf", "aceptable", "ok", "bueno")):
        return "ok"
    return "info"


def report_date(sc: Dict[str, Any]) -> str:
    """Fecha del reporte (YYYY-MM-DD): report_date del meta, o la de archivo."""
    d = str(_rm(sc).get("report_date", "") or "").strip()[:10]
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", d):
        return d
    return str(sc.get("archived_at", "") or "")[:10]


def machine_of(sc: Dict[str, Any]) -> str:
    rm = _rm(sc)
    tag = str(rm.get("instance_tag", "") or "").strip()
    if tag:
        return tag
    ac = str(rm.get("asset_class", "") or "").strip()
    if "·" in ac:                       # "Reporte Semanal · SGT300B"
        return ac.split("·")[-1].strip()
    return ac or "—"


def month_label(ymd: str) -> str:
    try:
        y, m = int(ymd[:4]), int(ymd[5:7])
        return f"{MESES[m - 1].capitalize()} {y}"
    except Exception:  # noqa: BLE001
        return "Sin fecha"


def fmt_date(ymd: str) -> str:
    try:
        y, m, d = int(ymd[:4]), int(ymd[5:7]), int(ymd[8:10])
        return f"{d} {MESES[m - 1][:3]} {y}"
    except Exception:  # noqa: BLE001
        return ymd or "—"


def group_by_month(reports: List[Dict[str, Any]]) -> List[Tuple[str, List[Dict[str, Any]]]]:
    """[(YYYY-MM, [reportes más nuevos primero])], meses más nuevos primero."""
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for sc in reports:
        groups.setdefault(report_date(sc)[:7] or "0000-00", []).append(sc)
    out = []
    for ym in sorted(groups, reverse=True):
        items = sorted(groups[ym],
                       key=lambda s: (report_date(s), s.get("archived_at", "")),
                       reverse=True)
        out.append((ym, items))
    return out


def filter_reports(reports: List[Dict[str, Any]], *, kind: str = "",
                   machine: str = "", year: str = "") -> List[Dict[str, Any]]:
    out = []
    for sc in reports:
        if kind and report_kind(sc) != kind:
            continue
        if machine and machine_of(sc) != machine:
            continue
        if year and not report_date(sc).startswith(year):
            continue
        out.append(sc)
    return out


def download_name(sc: Dict[str, Any]) -> str:
    rm = _rm(sc)
    base = (str(rm.get("consecutive", "") or "").strip()
            or f"{report_kind(sc)}_{machine_of(sc)}_{report_date(sc)}")
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("_") or "reporte"
    return f"{base}.pdf"


def _pdf_loader(sc: Dict[str, Any], email: str, role: str):
    """Callable sin argumentos para st.download_button (lectura diferida)."""
    return partial(_read_pdf, sc.get("_pdf_path", ""), sc.get("archive_id", ""),
                   email, role)


def _read_pdf(pdf_path: str, archive_id: str, email: str, role: str) -> bytes:
    try:
        if pdf_path and Path(pdf_path).exists():
            return Path(pdf_path).read_bytes()
    except Exception:  # noqa: BLE001
        pass
    from core.reports_archive import get_archived_pdf_bytes
    return get_archived_pdf_bytes(archive_id, viewer_email=email,
                                  viewer_role=role) or b""


# ---------------------------------------------------------------------------
# Vista CLIENTE (Reports, rol client) — español
# ---------------------------------------------------------------------------

_CLIENT_CSS = """
<style>
.ra-month{display:flex;align-items:baseline;gap:12px;margin:22px 0 8px;}
.ra-month .t{font:700 15px/1 'IBM Plex Sans',sans-serif;color:#0b1f3a;letter-spacing:-.2px;}
.ra-month .n{font:600 10.5px/1 'IBM Plex Mono',monospace;letter-spacing:.14em;
  text-transform:uppercase;color:#8090a6;}
.ra-month .rule{flex:1;height:1px;background:#e2e8f2;}
.ra-cell{font-family:'IBM Plex Sans',sans-serif;line-height:1.3;padding-top:4px;}
.ra-no{font:600 13px/1.25 'IBM Plex Mono',monospace;color:#12305e;word-break:break-all;}
.ra-sub{font:500 11.5px/1.35 'IBM Plex Sans',sans-serif;color:#8090a6;margin-top:3px;}
.ra-k{display:inline-block;font:600 10.5px/1 'IBM Plex Mono',monospace;letter-spacing:.08em;
  text-transform:uppercase;padding:5px 8px;border-radius:6px;}
.ra-k.w{background:#e8f0fb;color:#274b7d;}
.ra-k.m{background:#fdf1e1;color:#a65f05;}
.ra-k.t{background:#eef2f7;color:#3a4c66;}
.ra-m{font:600 13.5px/1.3 'IBM Plex Sans',sans-serif;color:#0b1f3a;}
.ra-d{font:500 13px/1.3 'IBM Plex Sans',sans-serif;color:#3a4c66;}
.ra-s{display:flex;align-items:center;gap:8px;font:500 13px/1.3 'IBM Plex Sans',sans-serif;color:#0b1f3a;}
.ra-s i{width:10px;height:10px;border-radius:50%;flex:0 0 10px;display:inline-block;}
.ra-empty{border:1px dashed #cdd6e3;border-radius:12px;padding:28px 22px;text-align:center;
  font:500 14px/1.5 'IBM Plex Sans',sans-serif;color:#3a4c66;background:#f7f9fc;}
.ra-empty b{display:block;color:#0b1f3a;font-size:15px;margin-bottom:4px;}
.ra-foot{font:500 12px/1.5 'IBM Plex Sans',sans-serif;color:#8090a6;margin-top:22px;}
</style>
"""

_KIND_CLS = {KIND_WEEKLY: "w", KIND_MONTHLY: "m", KIND_TECH: "t"}


def _esc(x: Any) -> str:
    return _html.escape(str(x if x is not None else ""))


def _client_name(email: str) -> str:
    try:
        from core.clients import get_client_for_email
        c = get_client_for_email(email)
        return (c.display_name or "").strip() if c else ""
    except Exception:  # noqa: BLE001
        return ""


def _row(sc: Dict[str, Any], email: str, role: str, key_prefix: str) -> None:
    import streamlit as st
    rm = _rm(sc)
    kind = report_kind(sc)
    sev_txt = str(rm.get("executive_severity", "") or "").strip()
    color = _SEV_COLORS[sev_class(sev_txt)]
    consec = str(rm.get("consecutive", "") or "").strip() or "Sin consecutivo"
    signers = []
    if rm.get("prepared_by"):
        signers.append(f"Elaboró {rm['prepared_by']}")
    if rm.get("reviewed_by"):
        signers.append(f"Aprobó {rm['reviewed_by']}")

    c1, c2, c3, c4, c5 = st.columns([2.6, 1.0, 1.1, 1.7, 1.25],
                                    vertical_alignment="center")
    with c1:
        st.markdown(
            f'<div class="ra-cell"><div class="ra-no">{_esc(consec)}</div>'
            f'<div class="ra-sub">{_esc(" · ".join(signers)) or "&nbsp;"}</div></div>',
            unsafe_allow_html=True)
    with c2:
        st.markdown(
            f'<div class="ra-cell"><span class="ra-k {_KIND_CLS[kind]}">{_esc(kind)}</span></div>',
            unsafe_allow_html=True)
    with c3:
        st.markdown(
            f'<div class="ra-cell"><div class="ra-m">{_esc(machine_of(sc))}</div>'
            f'<div class="ra-sub">{_esc(fmt_date(report_date(sc)))}</div></div>',
            unsafe_allow_html=True)
    with c4:
        # Reportes anteriores (cargados desde el correo) no traen estado: celda vacía.
        st.markdown(
            (f'<div class="ra-cell"><div class="ra-s"><i style="background:{color}"></i>'
             f'{_esc(sev_txt)}</div></div>') if sev_txt else '<div class="ra-cell"></div>',
            unsafe_allow_html=True)
    with c5:
        st.download_button(
            "Descargar PDF",
            data=_pdf_loader(sc, email, role),
            file_name=download_name(sc),
            mime="application/pdf",
            key=f"{key_prefix}_{sc.get('archive_id', '')}",
            use_container_width=True,
        )


def render_client_archive(email: str, role: str = "client", *,
                          as_client: Any = None) -> None:
    """Vista completa del archivo para un usuario cliente (solo lectura).

    as_client: un core.clients.Client — vista previa admin "ver como cliente":
    lista con los permisos del admin y filtra con la MISMA regla que aplica a
    los usuarios de ese cliente (sidecar_matches_client)."""
    import streamlit as st
    from core.ui_industrial import inject_industrial_css, industrial_band
    from core.reports_archive import list_archived_reports, sidecar_matches_client

    inject_industrial_css()
    st.markdown(_CLIENT_CSS, unsafe_allow_html=True)

    try:
        reports = list_archived_reports(viewer_email=email, viewer_role=role,
                                        limit=2000)
    except Exception:  # noqa: BLE001
        reports = []
    if as_client is not None:
        reports = [r for r in reports if sidecar_matches_client(r, as_client)]

    cname = ((getattr(as_client, "display_name", "") or "").strip()
             if as_client is not None else _client_name(email))
    n_w = sum(1 for r in reports if report_kind(r) == KIND_WEEKLY)
    n_m = sum(1 for r in reports if report_kind(r) == KIND_MONTHLY)
    last = max((report_date(r) for r in reports), default="")
    industrial_band(
        "Archivo de reportes" + (f" · {_esc(cname)}" if cname else ""),
        "Reportes de su flota",
        "Todos los reportes semanales y mensuales aprobados por su especialista, "
        "en el orden en que se emitieron. Descargue cualquiera en PDF.",
        kpis=[(str(len(reports)), "Reportes"), (str(n_w), "Semanales"),
              (str(n_m), "Mensuales"),
              (_esc(fmt_date(last)) if last else "—", "Último emitido")],
    )

    if not reports:
        st.markdown(
            '<div class="ra-empty"><b>Aún no hay reportes en su archivo.</b>'
            'Cada reporte semanal o mensual que su especialista apruebe y le '
            'envíe quedará guardado aquí automáticamente.</div>',
            unsafe_allow_html=True)
        return

    kinds = ["Todos", KIND_WEEKLY, KIND_MONTHLY] + (
        [KIND_TECH] if any(report_kind(r) == KIND_TECH for r in reports) else [])
    machines = sorted({machine_of(r) for r in reports})
    years = sorted({report_date(r)[:4] for r in reports if report_date(r)},
                   reverse=True)

    f1, f2, f3 = st.columns([2.2, 1.4, 1.0], vertical_alignment="bottom")
    with f1:
        kind = st.segmented_control("Tipo de reporte", kinds, default="Todos",
                                    key="ra_cli_kind") or "Todos"
    with f2:
        machine = st.selectbox("Máquina", ["Todas"] + machines, key="ra_cli_machine")
    with f3:
        year = st.selectbox("Año", ["Todos"] + years, key="ra_cli_year")

    shown = filter_reports(
        reports,
        kind="" if kind == "Todos" else kind,
        machine="" if machine == "Todas" else machine,
        year="" if year == "Todos" else year,
    )
    if not shown:
        st.markdown('<div class="ra-empty"><b>Ningún reporte coincide con el filtro.</b>'
                    'Cambie el tipo, la máquina o el año.</div>',
                    unsafe_allow_html=True)
        return

    for ym, items in group_by_month(shown):
        st.markdown(
            f'<div class="ra-month"><span class="t">{_esc(month_label(ym + "-01"))}</span>'
            f'<span class="n">{len(items)} reporte{"s" if len(items) != 1 else ""}</span>'
            f'<span class="rule"></span></div>',
            unsafe_allow_html=True)
        with st.container(border=True):
            for i, sc in enumerate(items):
                if i:
                    st.markdown('<div style="height:1px;background:#eef2f7;margin:2px 0;"></div>',
                                unsafe_allow_html=True)
                _row(sc, email, role, "ra_cli_dl")

    st.markdown(
        '<div class="ra-foot">Los reportes del archivo son copias fieles del PDF que '
        'recibió por correo y no se modifican. Si le falta alguno, pídalo a su '
        'especialista.</div>',
        unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Vista ANALISTA (Report Center → Archive) — inglés (UI analista)
# ---------------------------------------------------------------------------

_STATUS_OPTS = ["Operación normal", "Alerta", "Peligro", "Máquina detenida"]


def _instances() -> List[Tuple[str, str]]:
    # Solo máquinas EN SERVICIO (las de baja/demo no reciben reportes).
    try:
        from core.instance_state import list_reportable_instances
        rows = list_reportable_instances() or []
    except Exception:  # noqa: BLE001
        rows = []
    out = []
    for r in rows:
        iid = (r.get("instance_id") if isinstance(r, dict) else "") or ""
        tag = (r.get("tag") if isinstance(r, dict) else "") or iid
        if iid:
            out.append((iid, tag))
    return sorted(out, key=lambda x: x[1].lower())


def render_archive_admin(me_email: str, me_role: str) -> None:
    import streamlit as st
    from core.ui_industrial import html_table, dot
    from core.reports_archive import (archive_report_pdf, list_archived_reports,
                                      delete_archived_report)

    if (me_role or "").lower() == "admin":
        try:
            from core.clients import list_clients
            clients = [c for c in list_clients() if (c.display_name or "").strip()]
        except Exception:  # noqa: BLE001
            clients = []
        full = "Full archive"
        pick = st.selectbox(
            "View as", [full] + [c.display_name for c in clients],
            key="rc_archive_view_as",
            help="Preview the archive exactly as a client's users see it in Reports.")
        if pick != full:
            client = next(c for c in clients if c.display_name == pick)
            st.caption(f"Preview — this is exactly what {pick} users see in Reports "
                       "(same visibility rule, read-only).")
            render_client_archive(me_email, me_role, as_client=client)
            return

    reports = list_archived_reports(viewer_email=me_email, viewer_role=me_role,
                                    limit=2000)
    reports = sorted(reports, key=lambda s: (report_date(s), s.get("archived_at", "")),
                     reverse=True)

    st.markdown('<div class="bf-label">Client archive</div>', unsafe_allow_html=True)
    st.caption("Every weekly and monthly report approved and sent from Report Center "
               "is archived automatically. Client users see the reports of their own "
               "company in Reports.")
    if reports:
        rows = []
        for sc in reports:
            rm = _rm(sc)
            sev = str(rm.get("executive_severity", "") or "")
            rows.append([
                rm.get("client", "") or "—",
                machine_of(sc),
                report_kind(sc),
                rm.get("consecutive", "") or "—",
                report_date(sc),
                f'{dot(sev_class(sev))} {_esc(sev or "—")}',
                "Yes" if sc.get("shared_with_client") else "No",
                sc.get("size_human", ""),
            ])
        html_table(["Client", "Machine", "Type", "Report no.", "Date", "Status",
                    "Shared flag", "Size"], rows, raw_cols=[5])
    else:
        st.info("The archive is empty.")

    # ---- Carga de reportes anteriores -------------------------------------
    st.markdown('<div class="bf-label" style="margin-top:18px;">Add an earlier report</div>',
                unsafe_allow_html=True)
    st.caption("Reports sent before 23 Nov 2026 were not archived. Upload the PDF the "
               "client received so it appears in their archive. Use the original date "
               "and report number.")
    insts = _instances()
    if not insts:
        st.warning("No machines found.")
        return
    with st.form("ra_backfill", clear_on_submit=True):
        a1, a2, a3 = st.columns([1.4, 1, 1])
        with a1:
            pick = st.selectbox("Machine", range(len(insts)),
                                format_func=lambda i: f"{insts[i][1]}  ·  {insts[i][0]}")
        with a2:
            period = st.selectbox("Type", [KIND_WEEKLY, KIND_MONTHLY])
        with a3:
            rdate = st.date_input("Report date", value=date.today(),
                                  max_value=date.today(), format="YYYY-MM-DD")
        b1, b2 = st.columns([1.4, 2])
        with b1:
            consec = st.text_input("Report no.", placeholder="SIGA-REP-TEC-WM-…-014")
        with b2:
            status = st.selectbox("Status on the report", _STATUS_OPTS)
        c1, c2 = st.columns(2)
        with c1:
            prepared = st.text_input("Prepared by")
        with c2:
            reviewed = st.text_input("Approved by")
        pdf = st.file_uploader("Report PDF", type=["pdf"])
        sub = st.form_submit_button("Add to client archive", type="primary")

    if sub:
        iid, tag = insts[pick]
        consec = (consec or "").strip()
        if pdf is None:
            st.error("Attach the PDF.")
            return
        data = pdf.getvalue()
        if not data.startswith(b"%PDF"):
            st.error("That file is not a PDF.")
            return
        if consec and any(
                (_rm(r).get("consecutive", "") or "").strip().lower() == consec.lower()
                and (_rm(r).get("instance_id") or "") == iid for r in reports):
            st.error(f"{consec} is already in the archive for {tag}.")
            return
        try:
            from core.instance_state import get_instance
            inst = get_instance(iid)
        except Exception:  # noqa: BLE001
            inst = None
        meta = {
            "client": (getattr(inst, "client", "") or "").strip(),
            "site": (getattr(inst, "site", "") or getattr(inst, "location", "") or "").strip(),
            "asset_class": f"Reporte {period} · {tag}",
            "instance_tag": tag, "instance_id": iid,
            "report_date": rdate.isoformat(),
            "consecutive": consec,
            "executive_severity": status,
            "period": period,
            "prepared_by": (prepared or "").strip(),
            "reviewed_by": (reviewed or "").strip(),
        }
        if not meta["client"]:
            st.error(f"{tag} has no client assigned — the client would not see it.")
            return
        res = archive_report_pdf(
            pdf_bytes=data, meta=meta, owner_email=(me_email or "report-center@sigasas.com"),
            archived_at=datetime(rdate.year, rdate.month, rdate.day, 12, 0, 0),
            shared_with_client=True,
            extra_notes=f"Earlier report added to the archive by {me_email}.")
        if res.get("ok"):
            st.success(f"Added: {consec or tag} · {rdate.isoformat()}. "
                       "The client sees it in Reports now.")
            st.rerun()
        else:
            st.error(f"Could not archive: {res.get('error', 'unknown error')}")

    # ---- Retirar uno cargado por error (solo admin) ------------------------
    if (me_role or "").lower() == "admin" and reports:
        with st.expander("Remove a report from the archive", expanded=False):
            st.caption("Deletes the PDF from the archive for everyone, including the "
                       "client. This cannot be undone.")
            idx = st.selectbox(
                "Report", range(len(reports)),
                format_func=lambda i: (f"{machine_of(reports[i])} · "
                                       f"{_rm(reports[i]).get('consecutive') or '—'} · "
                                       f"{report_date(reports[i])}"),
                key="ra_del_pick")
            ok = st.checkbox("I understand this deletes it permanently", key="ra_del_ok")
            if st.button("Delete report", disabled=not ok, key="ra_del_btn"):
                r = delete_archived_report(reports[idx].get("archive_id", ""),
                                           viewer_email=me_email, viewer_role=me_role)
                if r.get("ok"):
                    st.success("Deleted.")
                    st.rerun()
                else:
                    st.error(r.get("error", "Could not delete."))
