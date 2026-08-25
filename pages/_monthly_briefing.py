"""
pages/_monthly_briefing.py
==========================

Briefing Mensual Ejecutivo — UI (Ciclo 17.31).

Página oculta (admin + specialist) donde se genera y envía por email
el briefing ejecutivo mensual al VP de Operaciones / CFO del cliente
final. Reusa core.ai_briefing + core.briefing_monthly_pdf +
core.email_sender.

Flujo:
  1. Especialista selecciona cliente del listado de clientes con
     reportes archivados accesibles.
  2. Selecciona el mes a cubrir (default: mes anterior completo).
  3. Click 'Generar briefing' → spinner mientras Claude redacta.
  4. Preview del briefing en pantalla (markdown + tabla de activos).
  5. Click 'Descargar PDF' o 'Enviar por email' (con destinatarios
     editables, separados por coma).
  6. Si Send: usa core.email_sender (Microsoft Graph) con el PDF
     adjunto y un body HTML breve.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any, Dict, List

import streamlit as st

from core.auth import (
    get_current_user,
    render_user_menu,
    require_login,
)
from core.ai_briefing import (
    generate_monthly_briefing,
    SEVERITY_COLORS,
)
from core.briefing_monthly_pdf import (
    generate_monthly_briefing_pdf,
    _format_month_es,
)
from core.ai_diagnostic import is_ai_available
from core.email_sender import send_email, get_email_backend_status
from core.reports_archive import list_archived_reports


# =============================================================
# CONFIG + GUARDS
# =============================================================

st.set_page_config(
    page_title="Watermelon System | Briefing Mensual",
    layout="wide",
)
require_login()
render_user_menu()

_user = get_current_user() or {}
_user_email = (_user.get("email") or "").strip()
_user_role = (_user.get("role") or "").strip().lower()

if _user_role not in ("admin", "specialist"):
    st.error(
        "Restricted access. The Monthly Briefing is available "
        "to **admin** and **specialist** roles."
    )
    st.stop()


# =============================================================
# HEADER
# =============================================================

st.markdown(
    """
    <div style='padding:14px 18px; border-radius:14px;
                background:linear-gradient(135deg,#0f172a 0%, #1e3a8a 100%);
                color:#f1f5f9; margin-bottom:16px;'>
        <div style='font-size:0.95rem; font-weight:600; letter-spacing:0.04em;
                    text-transform:uppercase; color:#cbd5e1;'>
            Watermelon System · Executive Briefing
        </div>
        <div style='font-size:1.6rem; font-weight:700; margin-top:4px;'>
            Monthly briefing for the client's VP of Operations
        </div>
        <div style='font-size:0.95rem; color:#cbd5e1; margin-top:6px;'>
            One-page PDF with the consolidated status of the client's
            asset portfolio, ready to email to the C-level decision
            maker who does not log in to the system.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not is_ai_available():
    st.warning(
        "**AI unavailable.** `[anthropic] api_key` is not configured "
        "in the secrets. The briefing requires Claude to synthesize."
    )
    st.stop()


# =============================================================
# SESSION STATE
# =============================================================

if "wm_brief_result" not in st.session_state:
    st.session_state["wm_brief_result"] = None
if "wm_brief_pdf_bytes" not in st.session_state:
    st.session_state["wm_brief_pdf_bytes"] = None


# =============================================================
# DESCUBRIR CLIENTES DISPONIBLES
# =============================================================

@st.cache_data(show_spinner=False, ttl=120)
def _discover_clients(viewer_email: str, viewer_role: str) -> List[str]:
    """Scrapea los reportes archivados accesibles y devuelve la lista
    de clientes únicos (ordenada alfabéticamente)."""
    sidecars = list_archived_reports(
        viewer_email=viewer_email,
        viewer_role=viewer_role,
        limit=500,
    )
    clients: set = set()
    for sc in sidecars:
        rm = sc.get("report_meta", {}) or {}
        c = (rm.get("client") or "").strip()
        if c:
            clients.add(c)
    return sorted(clients)


_clients = _discover_clients(_user_email, _user_role)

if not _clients:
    st.info(
        "There are no clients with archived reports in the system. "
        "Once reports are archived from Reports → Archive report, "
        "the client will appear here."
    )
    st.stop()


# =============================================================
# FORMULARIO DE SELECCIÓN
# =============================================================

st.markdown("### 1. Briefing configuration")

_form_cols = st.columns([2.5, 1.5, 1.5])

with _form_cols[0]:
    selected_client = st.selectbox(
        "Client",
        options=_clients,
        key="wm_brief_client",
        help="Choose the client you want to generate the briefing for.",
    )

with _form_cols[1]:
    # Default: mes anterior completo
    today = date.today()
    if today.month == 1:
        default_y, default_m = today.year - 1, 12
    else:
        default_y, default_m = today.year, today.month - 1
    # Generamos lista de últimos 12 meses para elegir
    months_options: List[str] = []
    cy, cm = today.year, today.month
    for i in range(12):
        m = cm - i
        y = cy
        while m <= 0:
            m += 12
            y -= 1
        months_options.append(f"{y:04d}-{m:02d}")
    default_iso = f"{default_y:04d}-{default_m:02d}"
    if default_iso not in months_options:
        months_options.insert(0, default_iso)
    selected_month = st.selectbox(
        "Month to cover",
        options=months_options,
        index=months_options.index(default_iso) if default_iso in months_options else 0,
        format_func=lambda x: _format_month_es(x),
        key="wm_brief_month",
        help=(
            "Defaults to the full previous month. The briefing "
            "covers all of the client's archived reports within "
            "the selected month's range."
        ),
    )

with _form_cols[2]:
    st.caption("&nbsp;", unsafe_allow_html=True)  # spacer alineación
    generate_clicked = st.button(
        "Generate briefing",
        type="primary",
        use_container_width=True,
        key="wm_brief_generate_btn",
    )


# =============================================================
# GENERACIÓN DEL BRIEFING
# =============================================================

if generate_clicked:
    with st.spinner(
        "Claude reading all of the month's reports and "
        "synthesizing the executive briefing... (10-30 sec)"
    ):
        try:
            _brief_res = generate_monthly_briefing(
                client_filter=selected_client,
                month_iso=selected_month,
                viewer_email=_user_email,
                viewer_role=_user_role,
                use_cache=True,
            )
        except Exception as exc:
            _brief_res = {
                "ok": False,
                "markdown": (
                    f"_Unexpected error:_\n\n```\n"
                    f"{type(exc).__name__}: {exc}\n```"
                ),
                "asset_aggregates": [],
                "n_reports": 0,
                "n_assets": 0,
                "month_iso": selected_month,
                "client_filter": selected_client,
                "model": "",
                "fallback_used": False,
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_usd": 0.0,
            }
    st.session_state["wm_brief_result"] = _brief_res
    # Reset PDF cache (se re-genera al pedirlo)
    st.session_state["wm_brief_pdf_bytes"] = None

# =============================================================
# RENDER DEL RESULTADO
# =============================================================

_stored = st.session_state.get("wm_brief_result")
if _stored is None:
    st.info(
        "Set the client and month above, then click "
        "**Generate briefing**. Claude will read all of the period's "
        "archived reports and write the executive synthesis."
    )
    st.stop()

if not _stored.get("ok"):
    st.error(_stored.get("markdown", "Failed to generate the briefing."))
    if st.button("Retry"):
        st.session_state["wm_brief_result"] = None
        st.session_state["wm_brief_pdf_bytes"] = None
        st.rerun()
    st.stop()


# =============================================================
# Preview en pantalla
# =============================================================

st.markdown("### 2. Briefing preview")

if _stored.get("fallback_used"):
    st.info(
        "Briefing generated with the fallback model (Haiku 4.5). "
        "Slightly lower quality."
    )

# Caption con metadata de la consulta
_n_reports = _stored.get("n_reports", 0)
_n_assets = _stored.get("n_assets", 0)
_cost = _stored.get("cost_usd", 0.0)
_in_tok = _stored.get("input_tokens", 0)
_out_tok = _stored.get("output_tokens", 0)
_model_used = _stored.get("model", "")
st.caption(
    f"{_n_reports} reports · {_n_assets} assets · "
    f"Model: `{_model_used}` · Tokens: {_in_tok:,} → {_out_tok:,} · "
    f"Cost: ~${_cost:.4f}"
)

# Render del markdown del AI
with st.container(border=True):
    st.markdown(_stored.get("markdown", ""))

# Tabla de activos (vista rápida)
_aggs = _stored.get("asset_aggregates", []) or []
if _aggs:
    with st.expander(f"Status by asset ({len(_aggs)} assets)", expanded=False):
        for ag in _aggs:
            sev = ag.get("latest_severity", "—")
            color = SEVERITY_COLORS.get(sev, "#475569")
            st.markdown(
                f"<div style='padding:6px 10px; border-radius:6px; "
                f"background:#f8fafc; border-left:4px solid {color}; "
                f"margin-bottom:5px;'>"
                f"<b>{ag.get('asset_blob') or ag.get('instance_tag') or '—'}</b> "
                f"· {ag.get('n_reports_in_month', 0)} report(s) in the month · "
                f"<span style='color:{color}; font-weight:bold;'>{sev}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )


# =============================================================
# Generar PDF (lazy)
# =============================================================

st.markdown("### 3. Download or send")

_pdf_cols = st.columns([1.5, 1.5, 4])

with _pdf_cols[0]:
    if st.button(
        "Generate PDF",
        use_container_width=True,
        key="wm_brief_pdf_btn",
        type="secondary",
    ):
        with st.spinner("Generating PDF..."):
            try:
                _pdf_bytes = generate_monthly_briefing_pdf(_stored)
                st.session_state["wm_brief_pdf_bytes"] = _pdf_bytes
                st.success(f"PDF generated ({len(_pdf_bytes) // 1024} KB)")
            except Exception as exc:
                st.error(f"Error generating PDF: {exc}")

_pdf_bytes_cached = st.session_state.get("wm_brief_pdf_bytes")

with _pdf_cols[1]:
    if _pdf_bytes_cached:
        _safe_client = (
            (selected_client or "cliente").lower()
            .replace(" ", "_").replace("-", "_")[:30]
        )
        _fname = f"briefing_{_safe_client}_{selected_month}.pdf"
        st.download_button(
            "Download PDF",
            data=_pdf_bytes_cached,
            file_name=_fname,
            mime="application/pdf",
            use_container_width=True,
            key="wm_brief_dl_btn",
            type="primary",
        )
    else:
        st.button(
            "Download PDF",
            disabled=True,
            use_container_width=True,
            help="Generate the PDF first.",
        )


# =============================================================
# Envío por email
# =============================================================

st.markdown("---")
st.markdown("### 4. Send to client via email")

_email_status = get_email_backend_status()
if not _email_status.get("configured"):
    st.warning(
        f"📧 **Email not configured.** {_email_status.get('details', '')}\n\n"
        "To send briefings by email, configure Microsoft Graph "
        "or SMTP in the Streamlit Cloud secrets (`[email]` section)."
    )
else:
    st.caption(
        f"Configured email backend: **{_email_status.get('backend', 'N/A')}**"
    )

    _email_cols = st.columns([3, 1.5])
    with _email_cols[0]:
        recipients_input = st.text_input(
            "Recipients (comma-separated)",
            value="",
            placeholder="vp.operations@client.com, maintenance.manager@client.com",
            key="wm_brief_recipients",
            help=(
                "Comma-separated list of emails. Each recipient "
                "receives an individual email with the PDF attached."
            ),
        )
    with _email_cols[1]:
        st.caption("&nbsp;", unsafe_allow_html=True)
        send_disabled = (
            not _email_status.get("configured")
            or not recipients_input.strip()
            or not _pdf_bytes_cached
        )
        send_help = None
        if not _pdf_bytes_cached:
            send_help = "Generate the PDF first."
        elif not recipients_input.strip():
            send_help = "Add at least one recipient."
        send_clicked = st.button(
            "Send email",
            use_container_width=True,
            disabled=send_disabled,
            help=send_help,
            type="primary" if not send_disabled else "secondary",
            key="wm_brief_send_btn",
        )

    if send_clicked and _pdf_bytes_cached:
        recipients = [
            e.strip() for e in recipients_input.split(",") if e.strip()
        ]
        valid_recipients = [e for e in recipients if "@" in e and "." in e]
        invalid_recipients = [e for e in recipients if e not in valid_recipients]

        if invalid_recipients:
            st.warning(
                f"Invalid emails ignored: {', '.join(invalid_recipients)}"
            )

        if not valid_recipients:
            st.error("No valid recipients.")
        else:
            month_label = _format_month_es(selected_month)
            subject = f"Executive Briefing — {selected_client} · {month_label}"

            body_text = f"""Estimado equipo,

Adjuntamos el Briefing Ejecutivo Mensual correspondiente a {month_label} \
para el cliente {selected_client}.

El presente documento sintetiza el estado del portafolio de activos \
monitoreados durante el periodo, incluyendo top 3 prioridades operativas \
y recomendación global del mes.

Para detalle técnico completo, ingresá a la plataforma Watermelon System.

Cordialmente,
Equipo de Machinery Diagnostics
SIGA SAS
"""
            body_html = f"""<!DOCTYPE html>
<html>
<body style="font-family:Arial,sans-serif;color:#0f172a;line-height:1.5;max-width:600px;">
  <div style="background:linear-gradient(135deg,#0f172a 0%, #1e3a8a 100%);color:#fff;padding:18px 22px;border-radius:10px 10px 0 0;">
    <div style="font-size:11px;letter-spacing:0.06em;text-transform:uppercase;color:#cbd5e1;">
      Watermelon System · Briefing Ejecutivo
    </div>
    <div style="font-size:20px;font-weight:700;margin-top:4px;">
      {selected_client}
    </div>
    <div style="font-size:14px;color:#cbd5e1;margin-top:2px;">
      {month_label}
    </div>
  </div>
  <div style="padding:16px 22px;background:#fff;border:1px solid #e6ebf2;border-top:none;border-radius:0 0 10px 10px;">
    <p>Estimado equipo,</p>
    <p>Adjuntamos el <b>Briefing Ejecutivo Mensual</b> correspondiente a
       <b>{month_label}</b> para el cliente <b>{selected_client}</b>.</p>
    <p>El presente documento sintetiza el estado del portafolio de activos
       monitoreados durante el periodo, incluyendo:</p>
    <ul>
      <li>Resumen ejecutivo del mes</li>
      <li>Top 3 prioridades operativas</li>
      <li>Estado consolidado de cada activo</li>
      <li>Recomendación global del mes</li>
    </ul>
    <p>Para detalle técnico completo de cada análisis, ingresá a la
       plataforma <a href="https://wm-home-final-2026.streamlit.app" style="color:#0ea5e9;">Watermelon System</a>.</p>
    <p style="margin-top:18px;">Cordialmente,<br/>
       <b>Equipo de Machinery Diagnostics</b><br/>
       <span style="color:#64748b;font-size:13px;">SIGA SAS · Cajicá, Cundinamarca</span>
    </p>
  </div>
  <div style="font-size:10px;color:#94a3b8;text-align:center;padding:10px;">
    Generado automáticamente por Watermelon System · {datetime.now().strftime('%Y-%m-%d %H:%M')}
  </div>
</body>
</html>
"""
            attachment_filename = (
                f"briefing_{(selected_client or 'cliente').lower().replace(' ', '_').replace('-', '_')[:30]}"
                f"_{selected_month}.pdf"
            )

            sent_ok = 0
            sent_fail = 0
            with st.spinner(
                f"Sending to {len(valid_recipients)} recipient(s)..."
            ):
                for recipient in valid_recipients:
                    res = send_email(
                        to=recipient,
                        subject=subject,
                        body_text=body_text,
                        body_html=body_html,
                        attachments=[(
                            attachment_filename,
                            _pdf_bytes_cached,
                            "application/pdf",
                        )],
                    )
                    if res.get("ok"):
                        sent_ok += 1
                    else:
                        sent_fail += 1
                        st.error(
                            f"Failed to send to {recipient}: {res.get('error', '')}"
                        )

            if sent_ok > 0:
                st.success(
                    f"Sent to {sent_ok} recipient(s) "
                    f"via {_email_status.get('backend', 'N/A')}."
                )
            if sent_fail > 0:
                st.warning(
                    f"{sent_fail} send(s) failed. Check the "
                    f"errors above and retry."
                )


# =============================================================
# FOOTER
# =============================================================

st.markdown("---")
st.caption(
    "To automate the monthly send (on the 1st of each month), you "
    "can configure a cron job that calls the generation and send "
    "endpoint. For now, sending is manual from this page."
)
