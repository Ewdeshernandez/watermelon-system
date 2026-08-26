from __future__ import annotations

import gc
import sys
import textwrap
from datetime import date, datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import plotly.graph_objects as go
import streamlit as st

try:
    from PIL import Image as PILImage  # Ciclo 17.21 — downsize para PDF
    _HAS_PIL = True
except Exception:
    PILImage = None  # type: ignore
    _HAS_PIL = False


# =============================================================
# Ciclo 17.21 — Downsize de imágenes para generación de PDF
# =============================================================
# Bug que matamos:
#   Al generar el PDF con 10+ imágenes, reportlab cargaba TODAS las
#   imágenes a memoria simultáneamente para construir el documento.
#   Cada imagen Plotly export con scale=2 pesa ~2-3 MB sin comprimir
#   en RAM. 10 imgs × 3 MB + buffer del PDF + libs Python (~300 MB)
#   excedía los 1 GB de Streamlit Cloud → OOMKilled al "Preparar PDF".
#
# Fix:
#   Antes de meter cada imagen al PDF, la pasamos por Pillow:
#     1) Si es más ancha que MAX_PDF_IMG_WIDTH (1500 px), downsize
#        manteniendo aspect ratio
#     2) Re-encodear como PNG optimizado
#     3) gc.collect() para liberar buffers intermedios
#
#   Resultado: imagen del PDF queda en max ~500 KB en memoria
#   en lugar de 2-3 MB. Pico al generar PDF baja de ~50 MB a ~5 MB.
#
# Calidad visual:
#   1500 px de ancho en una página A4 (21 cm) son 180 DPI.
#   Para impresión profesional son suficientes 150 DPI. Usuario
#   no nota diferencia visible.
# =============================================================

MAX_PDF_IMG_WIDTH = 1500   # px — ancho máximo aceptable en PDF


def _pdf_safe_image_bytes(raw_bytes: Optional[bytes],
                            max_width: int = MAX_PDF_IMG_WIDTH) -> Optional[bytes]:
    """Devuelve bytes optimizados para meter al PDF.

    - Si Pillow no está disponible o la imagen ya es chica, devuelve
      los bytes originales sin tocar.
    - Si la imagen excede `max_width`, la redimensiona manteniendo
      aspect ratio.
    - Re-encodea como PNG optimizado para reducir el peso en RAM
      cuando reportlab la abra.
    - Llama gc.collect() al final para liberar buffers intermedios.

    En cualquier excepción, devuelve los bytes originales (no rompe
    la generación del PDF si el downsize falla).
    """
    if not raw_bytes:
        return raw_bytes
    if not _HAS_PIL or PILImage is None:
        return raw_bytes
    try:
        with PILImage.open(BytesIO(raw_bytes)) as im:
            w, h = im.size
            if w <= max_width:
                # Imagen ya está en tamaño aceptable, no re-encodeamos
                return raw_bytes
            # Downsize manteniendo aspect ratio
            new_w = max_width
            new_h = int(round(h * (max_width / w)))
            # LANCZOS = filtro de mejor calidad para downscaling
            im_resized = im.resize((new_w, new_h), PILImage.LANCZOS)
            # Si la imagen tiene canal alpha, la dejamos como RGBA;
            # si no, RGB para que el PNG quede lo más liviano posible
            out_buf = BytesIO()
            if im_resized.mode in ("RGBA", "LA"):
                im_resized.save(out_buf, format="PNG", optimize=True)
            else:
                im_resized.convert("RGB").save(
                    out_buf, format="PNG", optimize=True,
                )
            out = out_buf.getvalue()
        # Liberar buffers intermedios
        del im_resized, out_buf
        gc.collect()
        # Solo usar el resized si efectivamente quedó más liviano
        if len(out) < len(raw_bytes):
            return out
        return raw_bytes
    except Exception as e:
        try:
            print(f"[WM_PDF_DOWNSIZE] WARN · downsize falló: {e}",
                  file=sys.stderr, flush=True)
        except Exception:
            pass
        return raw_bytes
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents

from core.auth import require_login, render_user_menu, get_current_user
from core.report_state import (
    clear_report_state,
    delete_named_report_draft,
    ensure_report_state_loaded,
    list_report_drafts,
    list_all_users_with_state,
    load_named_report_draft,
    load_report_state,
    read_item_image_bytes,   # Ciclo 17.20 — lazy loading de PNGs
    save_named_report_draft,
    save_report_state,
)
from core.reports_archive import (
    archive_report_pdf,
    list_archived_reports,
    get_archived_pdf_bytes,
    delete_archived_report,
    share_with_client,
    get_archive_stats,
)
from core.ai_diagnostic import (  # Ciclo 17.26 P5+ — síntesis ejecutiva AI
    generate_executive_summary,
    is_ai_available,
)
from core.ai_runcompare import (  # Ciclo 17.28 — Run-vs-Run comparison
    find_previous_report,
    generate_run_comparison,
)
from core.ai_rul import (  # Ciclo 17.30 — RUL Predictivo
    find_asset_history,
    generate_rul_estimate,
    MIN_HISTORY_FOR_RUL,
)
from core.ai_patterns import (  # Ciclo 17.34 — Pattern Memory
    find_similar_patterns,
)


st.set_page_config(page_title="Watermelon System | Reports", layout="wide")
require_login()
render_user_menu()


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = PROJECT_ROOT / "assets"
FONTS_DIR = ASSETS_DIR / "fonts"
WATERMELON_LOGO = ASSETS_DIR / "watermelon_logo.png"


def _register_unicode_fonts() -> Tuple[str, str]:
    """
    Registra una fuente Unicode TrueType desde assets/fonts/, en orden de
    preferencia. La primera familia disponible gana:

      1. IBM Plex Sans  — recomendada para reportes técnicos (look engineering
         pro, claridad metrológica). SIL Open Font License.
      2. DejaVu Sans   — fallback robusto, ya bundled.
      3. Helvetica     — último recurso (sin glifos extendidos).

    Devuelve (regular_name, bold_name) ya registrados y con familia mapeada
    para que <b>...</b> resuelva al peso bold.

    Para activar IBM Plex Sans, deja en assets/fonts/:
        IBMPlexSans-Regular.ttf
        IBMPlexSans-Bold.ttf
    (Descarga: github.com/IBM/plex / Google Fonts.)
    """
    candidates = (
        ("IBMPlexSans",  "IBMPlexSans-Regular.ttf",  "IBMPlexSans-Bold.ttf"),
        ("DejaVuSans",   "DejaVuSans.ttf",           "DejaVuSans-Bold.ttf"),
    )

    for family, regular_file, bold_file in candidates:
        try:
            regular_path = FONTS_DIR / regular_file
            bold_path = FONTS_DIR / bold_file
            if not (regular_path.exists() and bold_path.exists()):
                continue
            bold_name = f"{family}-Bold"
            if family not in pdfmetrics.getRegisteredFontNames():
                pdfmetrics.registerFont(TTFont(family, str(regular_path)))
            if bold_name not in pdfmetrics.getRegisteredFontNames():
                pdfmetrics.registerFont(TTFont(bold_name, str(bold_path)))
            registerFontFamily(
                family,
                normal=family,
                bold=bold_name,
                italic=family,
                boldItalic=bold_name,
            )
            return family, bold_name
        except Exception:
            continue
    return "Helvetica", "Helvetica-Bold"


PDF_FONT_REGULAR, PDF_FONT_BOLD = _register_unicode_fonts()

SIGA_WATERMARK_CANDIDATES = [
    ASSETS_DIR / "siga_watermark.png",
    ASSETS_DIR / "SIGA_watermark.png",
    ASSETS_DIR / "watermark_logo_transparent_background.png",
]

TODAY_STR = date.today().strftime("%Y-%m-%d")


st.markdown(
    """
    <style>
        .wm-page-title {
            font-size: 2.08rem;
            font-weight: 800;
            color: #f5f7fb;
            margin-bottom: 0.18rem;
            letter-spacing: 0.2px;
        }
        .wm-page-subtitle {
            color: #9aa6b2;
            font-size: 0.98rem;
            margin-bottom: 1.10rem;
        }
        .wm-card {
            background: linear-gradient(180deg, rgba(18,24,34,0.96) 0%, rgba(12,17,25,0.96) 100%);
            border: 1px solid rgba(90,110,140,0.22);
            border-radius: 20px;
            padding: 1rem 1rem 0.95rem 1rem;
            box-shadow: 0 12px 32px rgba(0,0,0,0.22);
            margin-bottom: 1rem;
        }
        .wm-kpi {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 14px;
            padding: 0.8rem 0.9rem;
            min-height: 82px;
        }
        .wm-kpi-label {
            color: #8fa0b5;
            font-size: 0.83rem;
            margin-bottom: 0.2rem;
        }
        .wm-kpi-value {
            color: #ffffff;
            font-size: 1.15rem;
            font-weight: 700;
        }
        .wm-section-title {
            color: #ffffff;
            font-weight: 800;
            font-size: 1.08rem;
            margin-top: 0.15rem;
            margin-bottom: 0.75rem;
        }
        .wm-block-title {
            color: #f5f7fb;
            font-size: 1.05rem;
            font-weight: 800;
            margin-bottom: 0.15rem;
        }
        .wm-block-subtitle {
            color: #95a2b1;
            font-size: 0.90rem;
            margin-bottom: 0.80rem;
        }
        .wm-divider {
            height: 1px;
            background: linear-gradient(90deg, rgba(255,255,255,0.08), rgba(255,255,255,0.02));
            margin: 0.85rem 0 1rem 0;
        }
        .wm-badge {
            display: inline-block;
            padding: 0.22rem 0.55rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 700;
            margin-right: 0.35rem;
            border: 1px solid transparent;
        }
        .wm-badge-spectrum {
            background: rgba(59, 130, 246, 0.14);
            color: #93c5fd;
            border-color: rgba(59, 130, 246, 0.28);
        }
        .wm-badge-waveform {
            background: rgba(16, 185, 129, 0.14);
            color: #86efac;
            border-color: rgba(16, 185, 129, 0.28);
        }
        .wm-badge-orbit {
            background: rgba(168, 85, 247, 0.14);
            color: #d8b4fe;
            border-color: rgba(168, 85, 247, 0.28);
        }
        .wm-badge-tabular {
            background: rgba(245, 158, 11, 0.14);
            color: #fcd34d;
            border-color: rgba(245, 158, 11, 0.28);
        }
        .wm-badge-trends {
            background: rgba(236, 72, 153, 0.14);
            color: #f9a8d4;
            border-color: rgba(236, 72, 153, 0.28);
        }
        .wm-badge-generic {
            background: rgba(148, 163, 184, 0.14);
            color: #cbd5e1;
            border-color: rgba(148, 163, 184, 0.28);
        }
        .wm-muted {
            color: #93a1b3;
            font-size: 0.9rem;
        }
        .wm-note {
            color: #b8c3cf;
            font-size: 0.92rem;
            line-height: 1.55;
        }
        .wm-figure-card {
            background: rgba(255,255,255,0.02);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 16px;
            padding: 0.9rem;
            margin-bottom: 1rem;
        }
        .wm-preview-card {
            background: rgba(255,255,255,0.02);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 16px;
            padding: 0.85rem 0.95rem;
            margin-bottom: 0.8rem;
        }
        .wm-meta-hint {
            color: #8fa0b5;
            font-size: 0.84rem;
            margin-top: -0.25rem;
            margin-bottom: 0.75rem;
        }
        .wm-highlight-box {
            background: rgba(14, 165, 233, 0.08);
            border: 1px solid rgba(14, 165, 233, 0.18);
            border-radius: 14px;
            padding: 0.8rem 0.95rem;
            color: #dbeafe;
            font-size: 0.92rem;
            line-height: 1.55;
        }
        .wm-signature-help {
            color: #8fa0b5;
            font-size: 0.83rem;
            margin-top: -0.15rem;
            margin-bottom: 0.85rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


DEFAULT_REPORT_META = {
    "report_title": "REPORTE DE MONITOREO EN LÍNEA",
    "client": "",
    "asset": "",
    "unit": "",
    "location": "",
    "prepared_by": "",
    "reviewed_by": "",
    # Cargos y ciudad pre-llenados con el default profesional de Watermelon.
    # El usuario los puede editar libremente — son solo punto de partida
    # para evitar campos vacíos en la portada.
    "prepared_role": "Junior Condition Monitoring Engineer",
    "reviewed_role": "Machinery Diagnostic Champion",
    "prepared_city": "Cajicá, Cundinamarca · Colombia",
    "reviewed_city": "Cajicá, Cundinamarca · Colombia",
    "period": "",
    "report_date": TODAY_STR,
    "consecutive": "",
    "service_objective": "",
    "service_development": "",
    "recommendations": "",
    "executive_summary": "",
    "train_description": "",
    # Ciclo 10A — campos SIGA-style para bloque grande del activo en portada
    "asset_class": "",         # ej. "TURBOGENERADOR"
    "asset_model": "",         # ej. "LM5000"
    # Format control SIGA-style (header de cada página)
    "format_code": "WMS-FMT-001",  # equivalente al SIGA-FMT-178
    "format_version": "1",
    "format_date": "2026-04-28",
    # Ciclo 14a — esquemático del tren (proviene de Asset Instance activa)
    "schematic_doc_id": "",       # doc_id en el Vault de la instancia
    "schematic_instance_id": "",  # id de la instancia para resolver el doc
}

# Ciclo 17.5.6 — ahora delegamos en ensure_report_state_loaded()
# que respeta items en memoria (bug histórico: si un módulo
# añadía items antes de visitar Reports, esta página los
# sobrescribía con los del disco). El helper compartido hace
# merge correcto.
ensure_report_state_loaded()

# =====================================================================
# Ciclo 17.14.1 HOTFIX — Banners de recovery / pérdida de datos
# =====================================================================
# Si load_report_state tuvo que recuperar desde un backup (porque el
# JSON principal estaba corrupto), avisamos al usuario VISIBLEMENTE.
# Esto era el bug histórico: antes el sistema silenciosamente devolvía
# {} y el usuario perdía un día de trabajo sin saber por qué.
_rec_from = st.session_state.get("wm_report_recovered_from")
_load_err = st.session_state.get("wm_report_load_error")

if _rec_from and not st.session_state.get("wm_report_recovery_dismissed"):
    _rec_n = st.session_state.get("wm_report_recovered_n_items", 0)
    _rec_at = st.session_state.get("wm_report_recovered_at", "")
    st.warning(
        f"**Your report was recovered from an automatic backup.** "
        f"The main file had a problem (Streamlit was probably "
        f"interrupted while saving) and we restored "
        f"from **`{_rec_from}`** with **{_rec_n} items**.\n\n"
        f"**Please check** that all the items you expected are present. "
        f"If something from the last moment is missing, it may have been lost "
        f"between the last successful save and the corruption.",
        icon="",
    )
    if st.button("Got it — dismiss this notice",
                 key="wm_report_recovery_dismiss"):
        st.session_state["wm_report_recovery_dismissed"] = True
        st.session_state.pop("wm_report_recovered_from", None)
        st.rerun()

if _load_err and not st.session_state.get("wm_report_load_err_dismissed"):
    st.error(
        f"**I couldn't load your report, even from backups.** "
        f"The file `data/report_state.json` and all of its backups (.bak.1 to "
        f".bak.{5}) are corrupted or unreadable.\n\n"
        f"**Technical error:** `{_load_err[:200]}`\n\n"
        f"**Suggested action:** manually check the "
        f"`data/` folder in case there is a recoverable file. Otherwise, you'll "
        f"have to rebuild the report from scratch. This should NOT happen "
        f"again — the new system keeps 5 rotating backups.",
        icon="",
    )
    if st.button("Got it — start a clean report",
                 key="wm_report_load_err_dismiss"):
        st.session_state["wm_report_load_err_dismissed"] = True
        st.session_state.pop("wm_report_load_error", None)
        st.rerun()
# Asegurar que el meta tenga los defaults del módulo Reports
# (campos como report_date) sin pisar lo que ya había.
_loaded_meta = st.session_state.get("report_meta", {}) or {}
_merged_meta = dict(DEFAULT_REPORT_META)
if isinstance(_loaded_meta, dict):
    _merged_meta.update(_loaded_meta)
if not _merged_meta.get("report_date"):
    _merged_meta["report_date"] = TODAY_STR

# =====================================================================
# Ciclo 17.15 — Inyectar owner_email en meta automáticamente
# =====================================================================
# El meta ahora incluye owner_email para identificar al autor del
# reporte y aplicar permisos. Si el meta no tiene owner_email,
# se asigna al usuario activo (primera vez que se persiste).
_wm_user = get_current_user() or {}
_wm_my_email = (_wm_user.get("email", "") or "").strip().lower()
_wm_my_role = (_wm_user.get("role", "") or "").strip().lower()
if not _merged_meta.get("owner_email"):
    _merged_meta["owner_email"] = _wm_my_email
st.session_state["report_meta"] = _merged_meta

# =====================================================================
# Ciclo 17.15 — Selector "¿De quién es el reporte que estás viendo?"
# =====================================================================
# Por default cada usuario carga SU propio reporte (vive en
# data/users/{email_slug}/report_state.json). Pero admin/specialist
# pueden inspeccionar el reporte de otro colega @sigasas.com en modo
# READ-ONLY, sin pisar el suyo. Si quieren editarlo, usan "Duplicar
# a mi reporte".
_owner_of_this_report = (_merged_meta.get("owner_email") or _wm_my_email).strip().lower()
_is_my_own = (_owner_of_this_report == _wm_my_email) or not _owner_of_this_report
_can_inspect_others = _wm_my_role in ("admin", "specialist")

# =====================================================================
# Ciclo 17.16 — Modo CLIENT: solo archivo histórico, sin editor
# =====================================================================
# Si el role es 'client', no debería poder editar reportes — solo ver
# los PDFs archivados que están marcados shared_with_client. Renderizamos
# una vista limitada y st.stop() para no ejecutar todo el editor.
if _wm_my_role == "client":
    st.markdown(
        f"""
        <div style="background:linear-gradient(135deg,#0f172a,#1e293b);
                    color:white;padding:20px 28px;border-radius:14px;
                    margin-bottom:18px;">
          <div style="font-size:11px;font-weight:800;letter-spacing:0.18em;
                      text-transform:uppercase;color:#a5b4fc;">
            🔐 Acceso de cliente — Solo lectura
          </div>
          <div style="font-size:22px;font-weight:800;margin:6px 0;">
            📚 Archivo histórico de reportes
          </div>
          <div style="color:rgba(226,232,240,0.85);font-size:13px;">
            Bienvenido <b>{_wm_my_email}</b>. Acá podés consultar y descargar
            los reportes técnicos que SIGASAS compartió con vos.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Ciclo 23.144 — Vista cliente minimalista:
    # Removidos: métricas "Reportes disponibles" + "Espacio total" (info interna SIGA,
    # no aporta al cliente). Filtro "por cliente" también removido — el cliente ya
    # está scopeado a su propio match_strings, filtrar por sí mismo es redundante.
    # Quedan solo filtros útiles: por activo + por año.
    cf1, cf2 = st.columns([2, 1])
    with cf1:
        _cf_asset = st.text_input("Filtrar por activo",
                                   placeholder="ej. C-200C",
                                   key="wm_cli_arch_asset").strip()
    with cf2:
        _cf_year = st.selectbox(
            "Año",
            options=["(todos)"] + [str(y) for y in range(datetime.now().year, 2023, -1)],
            index=0, key="wm_cli_arch_year",
        )
    _cli_from = ""; _cli_to = ""
    if _cf_year and _cf_year != "(todos)":
        _cli_from = f"{_cf_year}-01-01"
        _cli_to = f"{_cf_year}-12-31"

    _cli_archived = list_archived_reports(
        viewer_email=_wm_my_email,
        viewer_role=_wm_my_role,
        client_filter="",  # cliente no filtra por cliente — ya está scopeado por backend
        asset_filter=_cf_asset,
        date_from=_cli_from,
        date_to=_cli_to,
        limit=100,
    )
    if not _cli_archived:
        st.info(
            "📭 No hay reportes compartidos contigo todavía. "
            "Cuando SIGASAS publique un nuevo análisis, aparecerá acá."
        )
    else:
        st.caption(f"{len(_cli_archived)} reporte(s) disponibles")
        for sc in _cli_archived:
            rm = sc.get("report_meta", {}) or {}
            _aid = sc.get("archive_id", "")
            _client = rm.get("client", "—")
            _asset = rm.get("asset_class") or rm.get("instance_tag") or "—"
            _sev = rm.get("executive_severity", "")
            _date = sc.get("archived_at", "")[:16]
            _size = sc.get("size_human", "")

            # Ciclo 23.144 — Fix render `</div>` literal en cliente:
            # La f-string anidada {f'<span...>' if _sev else ''} dentro del
            # f"""...""" triple-quoted producía un bug donde el `</div>` del
            # contenedor derecho aparecía como texto literal. Solución:
            # construir el badge fuera del template y embedearlo como variable
            # simple. Si no hay severidad, simplemente omitimos el span.
            _sev_badge = (
                f'<span style="background:#fee2e2;color:#b91c1c;'
                f'padding:4px 10px;border-radius:999px;font-size:10px;'
                f'font-weight:800;">{_sev}</span>'
                if _sev else ""
            )
            st.markdown(
                f"""
                <div style="background:white;border:1px solid #e6ebf2;
                            border-radius:12px;padding:14px 18px;margin-bottom:8px;
                            display:flex;justify-content:space-between;align-items:center;">
                  <div>
                    <div style="font-weight:800;color:#0f172a;font-size:15px;">
                      {_client} · {_asset}
                    </div>
                    <div style="color:#475569;font-size:12px;margin-top:2px;">
                      Publicado {_date} · {_size}
                    </div>
                  </div>
                  <div>{_sev_badge}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            _pb = get_archived_pdf_bytes(_aid, viewer_email=_wm_my_email,
                                           viewer_role=_wm_my_role)
            if _pb:
                st.download_button(
                    "Descargar este reporte",
                    data=_pb,
                    file_name=f"{_aid.split('/')[-1]}.pdf",
                    mime="application/pdf",
                    key=f"cli_dl_{_aid}",
                    use_container_width=True,
                )

    st.divider()
    st.caption(
        "¿Necesitás un reporte que no aparece acá? Contactá a tu "
        "especialista SIGASAS para solicitar la publicación."
    )
    st.stop()


# =====================================================================
# Resto del page (editor + archivo) solo se ejecuta para admin/specialist
# =====================================================================

# Selector solo si admin/specialist Y hay al menos otro usuario con estado
if _can_inspect_others:
    _all_users = list_all_users_with_state()
    _other_users = [
        u for u in _all_users
        if u.get("owner_email") and u["owner_email"].lower() != _wm_my_email
    ]
    if _other_users:
        with st.expander(
            "Inspect another specialist's report (read-only)",
            expanded=False,
        ):
            _opts = ["(my own report)"] + [
                f"{u['owner_email']}  ·  {u['n_items']} items  ·  "
                f"last saved: {u.get('last_saved', '')[:16]}"
                for u in _other_users
            ]
            _emails = [_wm_my_email] + [u["owner_email"] for u in _other_users]
            _current_idx = 0
            if _owner_of_this_report and _owner_of_this_report != _wm_my_email:
                if _owner_of_this_report in _emails:
                    _current_idx = _emails.index(_owner_of_this_report)
            _pick_idx = st.selectbox(
                "Whose?",
                options=range(len(_opts)),
                format_func=lambda i: _opts[i],
                index=_current_idx,
                key="wm_inspect_other_report",
            )
            _new_target = _emails[_pick_idx]
            if _new_target != _owner_of_this_report:
                # Cargar el reporte del otro usuario en modo lectura
                _other_state = load_report_state(email=_new_target)
                st.session_state["report_items"] = _other_state.get("items", [])
                _other_meta = _other_state.get("meta", {}) or {}
                if not _other_meta.get("owner_email"):
                    _other_meta["owner_email"] = _new_target
                st.session_state["report_meta"] = _other_meta
                st.session_state["report_state_loaded_for"] = _new_target
                st.rerun()

# Recalcular flags después del posible switch
_owner_of_this_report = (
    st.session_state.get("report_meta", {}).get("owner_email", "") or _wm_my_email
).strip().lower()
_is_my_own = (_owner_of_this_report == _wm_my_email)

# Banner de modo lectura si no es tu reporte
if not _is_my_own and _owner_of_this_report:
    bcols = st.columns([0.7, 0.3])
    with bcols[0]:
        st.warning(
            f"️ **You are viewing `{_owner_of_this_report}`'s report "
            f"in READ-ONLY mode.** Any changes you make will not be "
            f"saved. To edit it, use the 'Duplicate to my report' button.",
            icon="",
        )
    with bcols[1]:
        if st.button("Duplicate to my report",
                     use_container_width=True,
                     key="wm_dup_to_mine",
                     type="primary"):
            # Copiar items + meta al espacio del usuario actual
            _items_to_copy = list(st.session_state.get("report_items", []) or [])
            _meta_to_copy = dict(st.session_state.get("report_meta", {}) or {})
            _meta_to_copy["owner_email"] = _wm_my_email
            _meta_to_copy["duplicated_from"] = _owner_of_this_report
            _meta_to_copy["duplicated_at"] = datetime.now().isoformat(timespec="seconds")
            save_report_state(
                items=_items_to_copy,
                meta=_meta_to_copy,
                email=_wm_my_email,
            )
            st.session_state["report_meta"] = _meta_to_copy
            st.session_state["report_state_loaded_for"] = _wm_my_email
            st.success(
                f"✓ Duplicated to your workspace ({len(_items_to_copy)} items). "
                "You can now edit it freely."
            )
            st.rerun()

if "report_items" not in st.session_state:
    st.session_state["report_items"] = []

if "report_pdf_bytes" not in st.session_state:
    st.session_state["report_pdf_bytes"] = None
if "report_pdf_error" not in st.session_state:
    st.session_state["report_pdf_error"] = None
if "report_draft_name_value" not in st.session_state:
    st.session_state["report_draft_name_value"] = ""

if "report_meta" not in st.session_state:
    st.session_state["report_meta"] = dict(DEFAULT_REPORT_META)

if not st.session_state["report_meta"].get("report_date"):
    st.session_state["report_meta"]["report_date"] = TODAY_STR


# =============================================================
# Ciclo 14a — Auto-fill desde Asset Instance activa
# =============================================================
# Cuando hay una máquina seleccionada en la Machinery Library,
# pre-llenamos los campos de portada del reporte (cliente, sitio,
# clase, modelo, descripción del tren, esquemático). Sólo aplica
# si los campos están vacíos: NO sobreescribe lo que el ingeniero
# ya tipeó. Eso permite que el usuario haga override manual sin
# que el auto-fill se lo pise en cada rerun.
def _autofill_report_meta_from_active_instance() -> None:
    try:
        from core.instance_selector import get_active_instance_id
        from core.instance_state import get_instance, compose_train_description
    except Exception:
        return

    inst_id = get_active_instance_id()
    if not inst_id:
        return
    inst = get_instance(inst_id)
    if inst is None:
        return

    meta = st.session_state["report_meta"]

    # Ciclo 17.5.10 — DETECCIÓN DE CAMBIO DE INSTANCIA
    # ===============================================================
    # Bug reportado: cliente activo TES1 pero el reporte muestra
    # cliente, sitio, clase, modelo, train_description y esquemático
    # de C200C (la instancia previa). Causa: el back-fill original
    # solo escribía cuando el meta[key] estaba VACÍO, lo que dejaba
    # los datos pegados al cambiar de instancia.
    #
    # Fix: si la instancia activa cambió desde el último auto-fill,
    # los campos heredados de la instancia anterior están stale y
    # deben sobrescribirse. Marcamos el origen del autofill en
    # meta["_autofilled_from_instance_id"] para detectar el cambio.
    last_filled_from = (meta.get("_autofilled_from_instance_id") or "").strip()
    instance_changed = bool(last_filled_from) and last_filled_from != inst_id

    def _maybe_set(key: str, value: str) -> None:
        """Escribe meta[key]=value si está vacío O si cambió la
        instancia activa. Sincroniza también el widget key
        report_meta_<key> para que el textbox refleje el cambio."""
        value = (value or "").strip()
        if not value:
            return
        if instance_changed or not (meta.get(key) or "").strip():
            meta[key] = value
            _wkey = f"report_meta_{key}"
            if _wkey in st.session_state:
                st.session_state[_wkey] = value

    # Campos directos de la instancia
    _maybe_set("client", inst.client or "")
    _maybe_set("asset_class", inst.asset_class or "")
    _maybe_set("asset_model", inst.driver_model or inst.driven_model or "")
    _maybe_set("location", inst.site or inst.location or "")
    _maybe_set("asset", inst.tag or "")
    _maybe_set("unit", inst.tag or "")

    # Train description compuesta
    composed = compose_train_description(inst)
    _maybe_set("train_description", composed or "")

    # Esquemático principal: si la instancia cambió, hay que
    # invalidar el cached del meta antes de re-leer.
    if instance_changed:
        meta.pop("schematic_doc_id", None)
        meta.pop("schematic_instance_id", None)
    if inst.schematic_png:
        if instance_changed or not (meta.get("schematic_doc_id") or "").strip():
            meta["schematic_doc_id"] = inst.schematic_png
            meta["schematic_instance_id"] = inst.instance_id

    # Cuando cambió la instancia el resumen ejecutivo cached pierde
    # validez (puede mencionar la máquina vieja). Lo invalidamos para
    # forzar regeneración con findings de la nueva instancia.
    if instance_changed:
        meta["executive_summary"] = ""
        if "report_meta_executive_summary" in st.session_state:
            st.session_state["report_meta_executive_summary"] = ""

    # Marcar el origen para detectar futuros cambios
    meta["_autofilled_from_instance_id"] = inst_id

    # Persistir a disco si hubo cambio efectivo
    if instance_changed:
        try:
            save_report_state(
                items=st.session_state.get("report_items", []) or [],
                meta=meta,
            )
        except Exception:
            pass


_autofill_report_meta_from_active_instance()


def _normalize_report_items(raw_items: Any) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []

    if not isinstance(raw_items, list):
        return items

    for idx, item in enumerate(raw_items):
        if not isinstance(item, dict):
            continue

        fig = item.get("figure")
        # Ciclo 17.20 lazy loading: leer PNG desde disco solo cuando se necesite
        image_bytes = read_item_image_bytes(item)
        safe_fig = None

        if fig is not None:
            try:
                safe_fig = go.Figure(fig)
            except Exception:
                safe_fig = None

        if safe_fig is None and image_bytes is None:
            continue

        normalized = {
            "id": str(item.get("id") or f"report_item_{idx+1}"),
            "type": str(item.get("type") or "figure"),
            "title": str(item.get("title") or f"Figura {idx+1}"),
            "notes": str(item.get("notes") or ""),
            "signal_id": str(item.get("signal_id") or ""),
            "machine": str(item.get("machine") or ""),
            "point": str(item.get("point") or ""),
            "variable": str(item.get("variable") or ""),
            "timestamp": str(item.get("timestamp") or ""),
            "figure": safe_fig,
            "image_bytes": image_bytes,
        }
        items.append(normalized)

    return items


def _persist_items(items: List[Dict[str, Any]]) -> None:
    st.session_state["report_items"] = items


def _get_items() -> List[Dict[str, Any]]:
    items = _normalize_report_items(st.session_state.get("report_items", []))
    _persist_items(items)
    save_report_state(items=items, meta=st.session_state.get("report_meta", {}))
    return items


def _move_item(item_id: str, direction: int) -> None:
    items = _get_items()
    idx = next((i for i, item in enumerate(items) if item["id"] == item_id), None)
    if idx is None:
        return
    new_idx = idx + direction
    if new_idx < 0 or new_idx >= len(items):
        return
    items[idx], items[new_idx] = items[new_idx], items[idx]
    _persist_items(items)


def _remove_item(item_id: str) -> None:
    items = [item for item in _get_items() if item["id"] != item_id]
    _persist_items(items)


def _clear_all_items() -> None:
    st.session_state["report_items"] = []
    clear_report_state()


def _type_badge(item_type: str) -> str:
    return item_type.replace("_", " ").title()


def _type_badge_class(item_type: str) -> str:
    normalized = (item_type or "").strip().lower()
    mapping = {
        "spectrum": "wm-badge-spectrum",
        "waveform": "wm-badge-waveform",
        "orbit": "wm-badge-orbit",
        "tabular": "wm-badge-tabular",
        "trends": "wm-badge-trends",
    }
    return mapping.get(normalized, "wm-badge-generic")


def _source_line(item: Dict[str, Any]) -> str:
    parts = [
        item.get("machine", "").strip(),
        item.get("point", "").strip(),
        item.get("variable", "").strip(),
        item.get("timestamp", "").strip(),
    ]
    parts = [p for p in parts if p]
    return " | ".join(parts) if parts else "No associated metadata"


def _count_by_type(items: List[Dict[str, Any]], item_type: str) -> int:
    return sum(1 for item in items if item.get("type", "").lower() == item_type.lower())


def _first_existing_watermark() -> Optional[Path]:
    for p in SIGA_WATERMARK_CANDIDATES:
        if p.exists():
            return p
    return None


def _paragraph_safe(text: str) -> str:
    """
    Escapa caracteres especiales para insertar texto en un Paragraph de
    ReportLab, pero rehabilita un set acotado de tags inline soportados por
    ReportLab (negrita, itálica, subíndice, superíndice). Esto permite que
    las narrativas auto-redactadas usen <b>...</b> para sub-headers sin
    inyectar HTML peligroso desde fuentes no controladas.
    """
    escaped = (
        (text or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br/>")
    )
    # Rehabilitar tags whitelisted (ya escapados a &lt;tag&gt;)
    for opener, closer in (("b", "b"), ("i", "i"), ("sub", "sub"), ("sup", "sup")):
        escaped = escaped.replace(f"&lt;{opener}&gt;", f"<{opener}>")
        escaped = escaped.replace(f"&lt;/{closer}&gt;", f"</{closer}>")
    return escaped


def _figure_png_bytes(fig: go.Figure) -> bytes:
    export_fig = go.Figure(fig)
    export_fig.update_layout(
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f8fafc",
        font=dict(color="#0f172a"),
    )
    # Ciclo 23.155 — anti-OOM: vía helper compartido (decima + scale=1).
    from core.plot_export import fig_to_png_bytes
    png, _ = fig_to_png_bytes(export_fig, width=2400, height=1250, scale=1)
    return png or b""


def _fit_image_dimensions(img_bytes: bytes, max_width: float, max_height: float) -> Tuple[float, float]:
    reader = ImageReader(BytesIO(img_bytes))
    img_w, img_h = reader.getSize()
    scale = min(max_width / img_w, max_height / img_h)
    return img_w * scale, img_h * scale


def _split_notes_and_summary_table(notes: str) -> Tuple[str, Optional[List[List[str]]]]:
    """
    Si el bloque de notes contiene un marcador '--- RESUMEN ---' seguido de una
    tabla en formato to_string() de pandas, la separa y la devuelve como
    matriz [[col1, col2, ...], [val11, val12, ...], ...].

    Devuelve (notes_sin_tabla, tabla_o_None).
    """
    if not notes or "--- RESUMEN ---" not in notes:
        return notes, None

    parts = notes.split("--- RESUMEN ---", 1)
    main_text = parts[0].rstrip()
    raw_block = parts[1].strip()

    if not raw_block:
        return main_text, None

    raw_lines = [ln for ln in raw_block.splitlines() if ln.strip()]
    if len(raw_lines) < 2:
        return main_text, None

    # Para tablas tipo df.to_string(): la primera línea son cabeceras separadas
    # por múltiples espacios; las restantes son filas. Se usa un split por
    # ≥ 2 espacios para preservar valores que contengan un solo espacio.
    import re
    rows: List[List[str]] = []
    for ln in raw_lines:
        cells = re.split(r"\s{2,}", ln.strip())
        rows.append(cells)

    # Si las filas no tienen el mismo ancho, abandonamos y devolvemos texto plano.
    width = len(rows[0])
    if any(len(r) != width for r in rows[1:]):
        return notes, None

    return main_text, rows


# =============================================================================
# Ciclo 17.26 — Bloque clínico AI: parser markdown → ReportLab nativo
# =============================================================================
# El módulo AI emite un diagnóstico en markdown. El reporte al cliente NO
# debe mostrar marcas "AI" ni markdown crudo (### o **bold**); debe leerse
# como un informe técnico continuo firmado por el especialista. Estos
# helpers parsean el markdown del AI y lo convierten en flowables nativos
# de ReportLab (Paragraphs con estilos clínicos, listas con sangría
# francesa, negritas e itálicas inline).
#
# Convención de marcador en `notes`:
#   <<<WM_AI_BLOCK>>>          ← inicio absoluto: a partir de acá se
#                                suprime la narrativa determinística y
#                                manda la voz del AI.
#   Parámetro|Valor            ← (opcional) tabla cuantitativa de evidencia.
#   Overall|0.30 mil pp          Una fila por línea, separadas por '|'.
#   1X|1.13 mil pp @ 3601 CPM    El parser intenta tabularizar todo lo que
#   ...                          esté entre WM_AI_BLOCK y WM_AI_NARRATIVE.
#   <<<WM_AI_NARRATIVE>>>      ← desde acá hasta el fin del bloque va el
#   ### Hallazgos principales    markdown del AI.
#   - bullet ...
# =============================================================================

_WM_AI_BLOCK_MARKER = "<<<WM_AI_BLOCK>>>"
_WM_AI_NARRATIVE_MARKER = "<<<WM_AI_NARRATIVE>>>"


def _md_inline_to_rl(text: str) -> str:
    """Convierte markdown inline (**bold**, *italic*, `code`) a inline-tags
    compatibles con ReportLab Paragraph. Escapa el resto de manera segura."""
    import re as _re
    escaped = (
        (text or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    # **bold**
    escaped = _re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", escaped)
    # *italic* (single asterisks, no overlap with **)
    escaped = _re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", escaped)
    # `code` → fuente monoespaciada
    escaped = _re.sub(r"`([^`]+)`", r'<font name="Courier">\1</font>', escaped)
    return escaped


def _split_ai_clinical_block(
    notes: str,
) -> Tuple[str, Optional[List[List[str]]], Optional[str]]:
    """Si `notes` contiene el marcador <<<WM_AI_BLOCK>>>, separa en:
      - pre_block_text: texto previo al marcador (debería estar vacío en el
        flujo nuevo; se conserva por defensa para casos legacy donde el
        especialista escribió notas manuales antes de generar el AI).
      - quant_rows: matriz [[col1, col2, ...], ...] derivada de las líneas
        'A|B' entre WM_AI_BLOCK y WM_AI_NARRATIVE. None si no hay tabla.
      - ai_md: markdown del AI desde <<<WM_AI_NARRATIVE>>> al final.

    Si no hay marcador, devuelve (notes, None, None) y el caller renderiza
    de la forma legacy (un solo Paragraph WMFigureText)."""
    if not notes or _WM_AI_BLOCK_MARKER not in notes:
        return notes, None, None

    pre, after_block = notes.split(_WM_AI_BLOCK_MARKER, 1)

    if _WM_AI_NARRATIVE_MARKER in after_block:
        quant_part, ai_part = after_block.split(_WM_AI_NARRATIVE_MARKER, 1)
    else:
        quant_part = ""
        ai_part = after_block

    # Parsear tabla cuantitativa (líneas con '|')
    quant_rows: List[List[str]] = []
    for line in quant_part.strip().splitlines():
        if "|" in line:
            cells = [c.strip() for c in line.split("|")]
            if len(cells) >= 2 and any(c for c in cells):
                quant_rows.append(cells)
    quant_or_none: Optional[List[List[str]]] = (
        quant_rows if len(quant_rows) >= 2 else None
    )

    return pre.rstrip(), quant_or_none, ai_part.strip()


def _render_ai_clinical_flowables(
    ai_md: str,
    styles,
) -> List[Any]:
    """Parsea el markdown del AI a flowables ReportLab nativos.

    Bloques soportados:
      ### Heading       → Paragraph WMClinicalHeading
      - bullet          → Paragraph WMClinicalBullet con '• ' y sangría
      1. numbered       → Paragraph WMClinicalNumbered con número en
                           negrita y sangría francesa
      párrafo regular   → Paragraph WMClinicalBody
      ---               → Spacer
      línea en blanco   → ignorada (los Paragraphs ya tienen spaceAfter)
    Inline:
      **bold**, *italic*, `code` (vía _md_inline_to_rl).
    """
    import re as _re
    flowables: List[Any] = []

    if not ai_md:
        return flowables

    lines = ai_md.splitlines()
    n = len(lines)
    i = 0

    while i < n:
        stripped = lines[i].strip()

        # Línea en blanco
        if not stripped:
            i += 1
            continue

        # Heading ### / ## / #
        if stripped.startswith("###"):
            head_txt = stripped.lstrip("#").strip()
            flowables.append(Paragraph(
                _md_inline_to_rl(head_txt), styles["WMClinicalHeading"]
            ))
            i += 1
            continue
        if stripped.startswith("##"):
            head_txt = stripped.lstrip("#").strip()
            flowables.append(Paragraph(
                _md_inline_to_rl(head_txt), styles["WMClinicalHeading"]
            ))
            i += 1
            continue

        # Regla horizontal
        if stripped in ("---", "***", "___"):
            flowables.append(Spacer(1, 0.18 * cm))
            i += 1
            continue

        # Bullet list (- / * / +)
        if _re.match(r"^[-*+]\s+", stripped):
            while i < n and _re.match(r"^[-*+]\s+", lines[i].strip()):
                content = _re.sub(r"^[-*+]\s+", "", lines[i].strip())
                # Continuación: líneas que no son block-starters
                j = i + 1
                while j < n:
                    nxt = lines[j].strip()
                    if not nxt:
                        break
                    if _re.match(r"^[-*+]\s+", nxt):
                        break
                    if _re.match(r"^\d+\.\s+", nxt):
                        break
                    if nxt.startswith("#"):
                        break
                    if nxt in ("---", "***", "___"):
                        break
                    content += " " + nxt
                    j += 1
                flowables.append(Paragraph(
                    "•&nbsp;&nbsp;" + _md_inline_to_rl(content),
                    styles["WMClinicalBullet"],
                ))
                i = j
            continue

        # Numbered list (1. 2. 3. ...)
        if _re.match(r"^\d+\.\s+", stripped):
            while i < n and _re.match(r"^\d+\.\s+", lines[i].strip()):
                m = _re.match(r"^(\d+)\.\s+(.*)", lines[i].strip())
                if not m:
                    break
                num = m.group(1)
                content = m.group(2)
                j = i + 1
                while j < n:
                    nxt = lines[j].strip()
                    if not nxt:
                        break
                    if _re.match(r"^[-*+]\s+", nxt):
                        break
                    if _re.match(r"^\d+\.\s+", nxt):
                        break
                    if nxt.startswith("#"):
                        break
                    if nxt in ("---", "***", "___"):
                        break
                    content += " " + nxt
                    j += 1
                flowables.append(Paragraph(
                    f"<b>{num}.</b>&nbsp;&nbsp;{_md_inline_to_rl(content)}",
                    styles["WMClinicalNumbered"],
                ))
                i = j
            continue

        # Párrafo regular: acumular hasta blank o block-starter
        para = stripped
        j = i + 1
        while j < n:
            nxt = lines[j].strip()
            if not nxt:
                break
            if nxt.startswith("#"):
                break
            if _re.match(r"^[-*+]\s+", nxt):
                break
            if _re.match(r"^\d+\.\s+", nxt):
                break
            if nxt in ("---", "***", "___"):
                break
            para += " " + nxt
            j += 1
        flowables.append(Paragraph(
            _md_inline_to_rl(para), styles["WMClinicalBody"]
        ))
        i = j

    return flowables


def _render_quant_evidence_table(
    rows: List[List[str]],
    styles,
    usable_width: float,
) -> Any:
    """Construye una tabla compacta de evidencia cuantitativa para acompañar
    el diagnóstico clínico. Header oscuro, 2-4 filas con los números
    relevantes (overall, 1X, RPM, severidad ISO, etc.)."""
    if not rows or len(rows) < 2:
        return None
    n_cols = max(len(r) for r in rows)
    # Distribución: primera columna 35%, resto reparten 65%
    if n_cols >= 2:
        col_w = [
            usable_width * 0.32,
        ] + [
            (usable_width * 0.66) / (n_cols - 1),
        ] * (n_cols - 1)
    else:
        col_w = [usable_width]

    header = [Paragraph(_paragraph_safe(c), styles["WMTableHeader"]) for c in rows[0]]
    body = [
        [Paragraph(_paragraph_safe(c), styles["WMTableCell"]) for c in r]
        for r in rows[1:]
    ]
    tbl = Table([header] + body, colWidths=col_w, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("FONTNAME", (0, 0), (-1, 0), PDF_FONT_BOLD),
        ("FONTNAME", (0, 1), (-1, -1), PDF_FONT_REGULAR),
        ("FONTSIZE", (0, 0), (-1, -1), 8.4),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
        ("TOPPADDING", (0, 0), (-1, 0), 6),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 4),
        ("TOPPADDING", (0, 1), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f1f5f9"), colors.white]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
    ]))
    return tbl


def _render_notes_flowables(
    notes_main: str,
    styles,
    summary_table: Optional[List[List[str]]],
    usable_width: float,
) -> List[Any]:
    """
    Construye la secuencia de flowables para el bloque de notas de una figura:
    el texto principal como Paragraph y, si aplica, el RESUMEN como Table
    nativa de ReportLab con cabecera coloreada y zebra striping.

    Ciclo 17.26 — si las notas contienen el marcador <<<WM_AI_BLOCK>>>, se
    suprime la narrativa determinística previa, se renderiza la tabla
    cuantitativa de evidencia (si viene) y luego el bloque clínico del AI
    parseado como flowables nativos. El cliente lee texto continuo de
    especialista, sin marcas de "AI" ni markdown crudo.
    """
    flowables: List[Any] = []

    # Detección del bloque clínico AI (Ciclo 17.26).
    pre_text, quant_rows, ai_md = _split_ai_clinical_block(notes_main)

    if ai_md is not None:
        # AI presente → manda. La narrativa determinística previa se
        # descarta. Solo conservamos el `pre_text` si realmente trae
        # notas manuales del especialista (no la narrativa Cat IV
        # autogenerada — que también se descarta porque el AI ya hizo
        # ese trabajo y mejor).
        # En el flujo nuevo `pre_text` viene vacío.
        if quant_rows:
            tbl = _render_quant_evidence_table(quant_rows, styles, usable_width)
            if tbl is not None:
                flowables.append(tbl)
                flowables.append(Spacer(1, 0.30 * cm))

        ai_flows = _render_ai_clinical_flowables(ai_md, styles)
        flowables.extend(ai_flows)
        flowables.append(Spacer(1, 0.20 * cm))

        # Si después del bloque AI hay un --- RESUMEN --- legacy, se
        # renderiza igual abajo (manteniendo compat con flujos viejos).
        if summary_table and len(summary_table) >= 1:
            # Se cae al render de summary_table de abajo, mismo código
            pass
        else:
            return flowables

    elif notes_main.strip():
        flowables.append(Paragraph(_paragraph_safe(notes_main), styles["WMFigureText"]))

    if summary_table and len(summary_table) >= 1:
        header = [Paragraph(_paragraph_safe(c), styles["WMTableHeader"]) for c in summary_table[0]]
        body_rows = []
        for r in summary_table[1:]:
            body_rows.append([Paragraph(_paragraph_safe(c), styles["WMTableCell"]) for c in r])

        n_cols = len(summary_table[0])
        # distribuir el ancho disponible de forma uniforme (con padding)
        col_w = (usable_width - 0.6 * cm) / max(n_cols, 1)
        col_widths = [col_w] * n_cols

        table_data = [header] + body_rows
        tbl = Table(table_data, colWidths=col_widths, repeatRows=1)
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("FONTNAME", (0, 0), (-1, 0), PDF_FONT_BOLD),
            ("FONTNAME", (0, 1), (-1, -1), PDF_FONT_REGULAR),
            ("FONTSIZE", (0, 0), (-1, -1), 8.4),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
            ("TOPPADDING", (0, 0), (-1, 0), 6),
            ("BOTTOMPADDING", (0, 1), (-1, -1), 4),
            ("TOPPADDING", (0, 1), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f1f5f9"), colors.white]),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
        ]))
        flowables.append(Spacer(1, 0.15 * cm))
        flowables.append(tbl)
        flowables.append(Spacer(1, 0.10 * cm))

    return flowables


# =============================================================
# Tabla de Contenido (Ciclo 10A.4)
# =============================================================
# WMDocTemplate subclasea SimpleDocTemplate para que `multiBuild` pueda
# llamar `afterFlowable` y registrar entradas TOC con número de página.
# Estrategia:
#   * Los Paragraphs de las 5 secciones principales (RESUMEN EJECUTIVO,
#     RECOMENDACIONES, OBJETIVO, DESARROLLO, FIGURAS) usan estilo
#     'WMTOC1' (visualmente idéntico a 'WMSection') → entran al TOC
#     como nivel 0.
#   * Los captions de cada figura usan 'WMTOC2' → nivel 1 (sub-entradas
#     bajo "FIGURAS Y ANÁLISIS").
#   * Los headings que NO deben aparecer en el TOC (e.g. 'TABLA DE
#     CONTENIDO' propio, sub-bloques internos) siguen usando 'WMSection'
#     o 'WMFigureCaption' originales — invisibles al TOC.
# `multiBuild` corre 2-3 pasadas hasta que los números de página
# convergen. `bookmarkPage` permite que cada entrada del TOC sea un
# link interno clickeable (PDF nativo).
class WMDocTemplate(SimpleDocTemplate):
    def afterFlowable(self, flowable):
        if not isinstance(flowable, Paragraph):
            return
        try:
            style_name = flowable.style.name
        except Exception:
            return
        if style_name == "WMTOC1":
            level = 0
        elif style_name == "WMTOC2":
            level = 1
        else:
            return
        text = flowable.getPlainText()
        # Key estable basado en id(flowable): el mismo objeto vive en
        # todas las pasadas de multiBuild → mismo key → el TOC compara
        # entries igualadas y converge en 2 pasadas. Si reseteáramos un
        # contador (1, 2, 3...) los keys cambiarían entre pasadas y
        # multiBuild fallaría con "Index entries not resolved".
        key = f"toc-{level}-{id(flowable):x}"
        self.canv.bookmarkPage(key)
        self.notify("TOCEntry", (level, text, self.page, key))


def _build_pdf_bytes(meta: Dict[str, str], items: List[Dict[str, Any]]) -> bytes:
    buffer = BytesIO()
    page_width, page_height = A4

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="WMTitle", parent=styles["Title"], fontName=PDF_FONT_BOLD, fontSize=15, leading=18, alignment=TA_LEFT, textColor=colors.HexColor("#0f172a"), spaceAfter=6))
    styles.add(ParagraphStyle(name="WMSubTitle", parent=styles["Normal"], fontName=PDF_FONT_BOLD, fontSize=12.5, leading=15, alignment=TA_LEFT, textColor=colors.HexColor("#111827"), spaceAfter=5))
    styles.add(ParagraphStyle(name="WMBody", parent=styles["BodyText"], fontName=PDF_FONT_REGULAR, fontSize=10.5, leading=15.5, alignment=TA_JUSTIFY, textColor=colors.HexColor("#111827"), spaceAfter=10))
    styles.add(ParagraphStyle(name="WMMeta", parent=styles["Normal"], fontName=PDF_FONT_REGULAR, fontSize=10.4, leading=14.2, alignment=TA_LEFT, textColor=colors.HexColor("#111827"), spaceAfter=5))
    styles.add(ParagraphStyle(name="WMSection", parent=styles["Heading2"], fontName=PDF_FONT_BOLD, fontSize=14.6, leading=18.5, alignment=TA_LEFT, textColor=colors.HexColor("#0f172a"), spaceBefore=6, spaceAfter=11))
    styles.add(ParagraphStyle(name="WMFigureCaption", parent=styles["Normal"], fontName=PDF_FONT_BOLD, fontSize=10.5, leading=13.5, alignment=TA_CENTER, textColor=colors.HexColor("#111827"), spaceBefore=6, spaceAfter=8))
    styles.add(ParagraphStyle(name="WMFigureText", parent=styles["BodyText"], fontName=PDF_FONT_REGULAR, fontSize=10.2, leading=14.8, alignment=TA_JUSTIFY, textColor=colors.HexColor("#111827"), spaceAfter=16))
    styles.add(ParagraphStyle(name="WMSignLine", parent=styles["Normal"], fontName=PDF_FONT_REGULAR, fontSize=9.6, leading=12, alignment=TA_CENTER, textColor=colors.HexColor("#111827"), spaceAfter=2))
    styles.add(ParagraphStyle(name="WMTableCell", parent=styles["Normal"], fontName=PDF_FONT_REGULAR, fontSize=8.4, leading=11, alignment=TA_LEFT, textColor=colors.HexColor("#111827")))
    styles.add(ParagraphStyle(name="WMTableHeader", parent=styles["Normal"], fontName=PDF_FONT_BOLD, fontSize=8.5, leading=11, alignment=TA_LEFT, textColor=colors.HexColor("#ffffff")))

    # Ciclo 17.26 — estilos del bloque clínico (renderiza el markdown del
    # módulo AI como ReportLab nativo: headers, párrafos justificados,
    # bullets con sangría francesa, listas numeradas). El cliente lee
    # texto continuo de un especialista; no ve markdown crudo.
    styles.add(ParagraphStyle(
        name="WMClinicalHeading",
        parent=styles["Normal"],
        fontName=PDF_FONT_BOLD,
        fontSize=10.6,
        leading=14,
        alignment=TA_LEFT,
        textColor=colors.HexColor("#0f172a"),
        spaceBefore=8,
        spaceAfter=4,
    ))
    styles.add(ParagraphStyle(
        name="WMClinicalBody",
        parent=styles["BodyText"],
        fontName=PDF_FONT_REGULAR,
        fontSize=10.2,
        leading=14.8,
        alignment=TA_JUSTIFY,
        textColor=colors.HexColor("#111827"),
        spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        name="WMClinicalBullet",
        parent=styles["BodyText"],
        fontName=PDF_FONT_REGULAR,
        fontSize=10.2,
        leading=14.6,
        alignment=TA_JUSTIFY,
        textColor=colors.HexColor("#111827"),
        leftIndent=14,
        firstLineIndent=-14,
        spaceAfter=5,
    ))
    styles.add(ParagraphStyle(
        name="WMClinicalNumbered",
        parent=styles["BodyText"],
        fontName=PDF_FONT_REGULAR,
        fontSize=10.2,
        leading=14.6,
        alignment=TA_JUSTIFY,
        textColor=colors.HexColor("#111827"),
        leftIndent=18,
        firstLineIndent=-18,
        spaceAfter=5,
    ))

    # Ciclo 10A.4 — estilos para entradas que SÍ entran al TOC.
    # Visualmente idénticos a WMSection / WMFigureCaption respectivamente,
    # pero con nombre distinto para que afterFlowable los detecte.
    styles.add(ParagraphStyle(name="WMTOC1", parent=styles["WMSection"]))
    styles.add(ParagraphStyle(name="WMTOC2", parent=styles["WMFigureCaption"]))

    # Ciclo 10A.4 — estilos del PROPIO TOC (cómo se ven las entradas
    # dentro de la página de Tabla de Contenido). H1 negrita, H2 indentada.
    toc_level0_style = ParagraphStyle(
        name="WMTOCLevel0",
        fontName=PDF_FONT_BOLD,
        fontSize=11,
        leading=16,
        leftIndent=0,
        firstLineIndent=0,
        spaceBefore=8,
        spaceAfter=2,
        textColor=colors.HexColor("#0f172a"),
    )
    toc_level1_style = ParagraphStyle(
        name="WMTOCLevel1",
        fontName=PDF_FONT_REGULAR,
        fontSize=10,
        leading=14,
        leftIndent=18,
        firstLineIndent=0,
        spaceBefore=2,
        spaceAfter=1,
        textColor=colors.HexColor("#334155"),
    )

    logo_watermark = _first_existing_watermark()

    doc = WMDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=2.1 * cm,
        rightMargin=2.1 * cm,
        topMargin=1.6 * cm,
        bottomMargin=1.5 * cm,
        title=meta.get("report_title") or "Watermelon System Report",
        author=meta.get("prepared_by") or "Watermelon System",
    )

    def _draw_cover_page(canvas, doc):
        # Portada SIGA-style: fondo blanco completamente limpio, sin cintas
        # de colores ni acentos — la sobriedad es el branding. La estructura
        # del contenido (logo centrado, bloque del activo en jerarquía
        # tipográfica, firmas paralelas) es la que aporta el peso visual,
        # no decoración cromática agresiva.
        canvas.saveState()
        canvas.setFillColor(colors.HexColor("#ffffff"))
        canvas.rect(0, 0, page_width, page_height, fill=1, stroke=0)

        # Header SIGA-style (mismo formato que en páginas internas para
        # consistencia): código de formato controlado a la izquierda,
        # número de página a la derecha. Línea fina cyan debajo, sutil.
        format_code = meta.get("format_code") or "WMS-FMT-001"
        format_version = meta.get("format_version") or "1"
        format_date = meta.get("format_date") or "2026-04-28"
        format_header = f"{format_code} | Versión {format_version} | Fecha {format_date}"

        internal_left = 2.1 * cm
        internal_right = 2.1 * cm
        internal_width_end = page_width - internal_right

        canvas.setFillColor(colors.HexColor("#0f172a"))
        canvas.setFont(PDF_FONT_BOLD, 7.8)
        canvas.drawString(internal_left, page_height - 1.0 * cm, format_header)

        canvas.setFont(PDF_FONT_BOLD, 9.0)
        canvas.drawRightString(
            page_width - internal_right,
            page_height - 1.0 * cm,
            f"Página {doc.page}",
        )

        # Línea fina cyan separadora arriba — único acento de color en la portada
        canvas.setStrokeColor(colors.HexColor("#0ea5e9"))
        canvas.setLineWidth(0.8)
        canvas.line(internal_left, page_height - 1.35 * cm, internal_width_end, page_height - 1.35 * cm)

        # Footer disclaimer (mismo de SIGA, idéntico a páginas internas)
        footer = (
            "INFORME VÁLIDO ÚNICAMENTE PARA LAS CONDICIONES PRESENTES "
            "DURANTE EL SERVICIO. NO PODRÁ SER COPIADO PARCIAL O TOTALMENTE "
            "SIN PREVIA AUTORIZACIÓN."
        )
        canvas.setStrokeColor(colors.HexColor("#0ea5e9"))
        canvas.setLineWidth(0.8)
        canvas.line(internal_left, 0.95 * cm, internal_width_end, 0.95 * cm)
        canvas.setFillColor(colors.HexColor("#475569"))
        canvas.setFont(PDF_FONT_REGULAR, 6.4)
        canvas.drawCentredString(
            (internal_left + internal_width_end) / 2,
            0.55 * cm,
            footer,
        )

        # Ciclo 17.7 — version stamp en la PORTADA del PDF.
        # Trazabilidad: cualquier cliente que abra el reporte sabe
        # exactamente con qué build del sistema fue generado. Se
        # imprime en la esquina inferior derecha, fuente pequeña
        # gris muy tenue para no compitir con el disclaimer.
        try:
            from core.version import get_version_info as _gvi_pdf
            _vinfo_pdf = _gvi_pdf()
            _ver_line = (
                f"Generado con Watermelon System "
                f"{_vinfo_pdf['version']}"
            )
            if _vinfo_pdf.get("commit"):
                _ver_line += f" · build {_vinfo_pdf['commit']}"
            if _vinfo_pdf.get("date"):
                _ver_line += f" · {_vinfo_pdf['date']}"
            canvas.setFillColor(colors.HexColor("#94a3b8"))
            canvas.setFont(PDF_FONT_REGULAR, 5.6)
            canvas.drawRightString(
                page_width - internal_right,
                0.30 * cm,
                _ver_line,
            )
        except Exception:
            pass

        canvas.restoreState()

    def _draw_internal_page(canvas, doc):
        canvas.saveState()
        canvas.setFillColor(colors.HexColor("#ffffff"))
        canvas.rect(0, 0, page_width, page_height, fill=1, stroke=0)

        canvas.setFont(PDF_FONT_BOLD, 11)
        canvas.setFillColor(colors.HexColor("#111827"))
        canvas.drawRightString(page_width - 1.2 * cm, page_height - 1.0 * cm, f"Página {doc.page}")

        internal_left = 2.1 * cm
        internal_right = 2.1 * cm
        internal_width_end = page_width - internal_right

        canvas.setStrokeColor(colors.HexColor("#0ea5e9"))
        canvas.setLineWidth(1.1)
        canvas.line(internal_left, page_height - 1.35 * cm, internal_width_end, page_height - 1.35 * cm)

        # Header SIGA-style (Ciclo 10A): código de formato controlado a la
        # izquierda + título del reporte centrado. El consecutivo va arriba a
        # la derecha junto al número de página, lo arma _draw_internal_page
        # justo después.
        format_code = meta.get("format_code") or "WMS-FMT-001"
        format_version = meta.get("format_version") or "1"
        format_date = meta.get("format_date") or "2026-04-28"
        format_header = f"{format_code} | Versión {format_version} | Fecha {format_date}"
        canvas.setFillColor(colors.HexColor("#0f172a"))
        canvas.setFont(PDF_FONT_BOLD, 7.8)
        canvas.drawString(internal_left, page_height - 1.0 * cm, format_header)
        canvas.setFont(PDF_FONT_REGULAR, 7.8)
        canvas.drawString(
            internal_left + 7.2 * cm,
            page_height - 1.0 * cm,
            f"| {meta.get('report_title') or 'Reporte técnico'}",
        )

        footer = "INFORME VÁLIDO ÚNICAMENTE PARA LAS CONDICIONES PRESENTES DURANTE EL SERVICIO. NO PODRÁ SER COPIADO PARCIAL O TOTALMENTE SIN PREVIA AUTORIZACIÓN."
        canvas.setStrokeColor(colors.HexColor("#0ea5e9"))
        canvas.setLineWidth(1.0)
        canvas.line(internal_left, 0.95 * cm, internal_width_end, 0.95 * cm)

        canvas.setFillColor(colors.HexColor("#111827"))
        canvas.setFont(PDF_FONT_REGULAR, 6.4)
        canvas.drawCentredString((internal_left + internal_width_end) / 2, 0.55 * cm, footer)

        # Ciclo 17.7 — version stamp en páginas internas también
        try:
            from core.version import get_version_short as _gvs_pdf
            _ver_short = _gvs_pdf()
            canvas.setFillColor(colors.HexColor("#94a3b8"))
            canvas.setFont(PDF_FONT_REGULAR, 5.6)
            canvas.drawRightString(
                page_width - internal_right,
                0.30 * cm,
                f"Watermelon System {_ver_short}",
            )
        except Exception:
            pass

        canvas.restoreState()

    story: List[Any] = []

    # ============================================================
    # PORTADA SIGA-STYLE — todo centrado, sobrio, simétrico.
    # ============================================================
    from reportlab.platypus import HRFlowable

    # 1. Logo Watermelon centrado arriba
    if WATERMELON_LOGO.exists():
        cover_logo = Image(str(WATERMELON_LOGO), width=5.8 * cm, height=2.7 * cm)
        cover_logo.hAlign = "CENTER"
        story.append(Spacer(1, 0.40 * cm))
        story.append(cover_logo)
        story.append(Spacer(1, 0.85 * cm))

    # 2. Eyebrow centrado, color sobrio
    story.append(
        Paragraph(
            "Machinery Diagnostics Engineering",
            ParagraphStyle(
                name="WMCoverEyebrow",
                parent=styles["Normal"],
                fontName=PDF_FONT_BOLD,
                fontSize=11,
                leading=14,
                alignment=TA_CENTER,
                textColor=colors.HexColor("#475569"),
                spaceAfter=6,
            ),
        )
    )

    # 3. Título grande del reporte, centrado
    story.append(
        Paragraph(
            _paragraph_safe(meta.get("report_title") or "REPORTE TÉCNICO"),
            ParagraphStyle(
                name="WMCoverReportTitle",
                parent=styles["Normal"],
                fontName=PDF_FONT_BOLD,
                fontSize=20,
                leading=24,
                alignment=TA_CENTER,
                textColor=colors.HexColor("#0f172a"),
                spaceAfter=4,
            ),
        )
    )

    # 4. Sub-marca "Watermelon System" centrada (igual al SIGA)
    story.append(
        Paragraph(
            "Watermelon System",
            ParagraphStyle(
                name="WMCoverBrand",
                parent=styles["Normal"],
                fontName=PDF_FONT_REGULAR,
                fontSize=12,
                leading=15,
                alignment=TA_CENTER,
                textColor=colors.HexColor("#475569"),
                spaceAfter=20,
            ),
        )
    )

    # ===== Bloque grande del activo (estilo SIGA) =====
    # En el reporte SIGA original se ve algo como:
    #     TURBOGENERADOR TES1
    #     LM5000
    #     VILLAVICENCIO
    #     TERMOSURIA
    # Cada línea grande, centrada (o alineada a izquierda según diseño).
    # Mantenemos alineación a izquierda para que case con el resto de la
    # portada que ya tiene logo + Machinery Diagnostics Engineering.
    asset_class = (meta.get("asset_class") or "").strip()
    asset_name = (meta.get("asset") or "").strip()
    unit_name = (meta.get("unit") or "").strip()
    asset_model = (meta.get("asset_model") or "").strip()
    location_name = (meta.get("location") or "").strip()
    client_name = (meta.get("client") or "").strip()

    # Línea 1: clase + tag/unidad ("TURBOGENERADOR TES1")
    line1_parts = []
    if asset_class:
        line1_parts.append(asset_class)
    if unit_name:
        line1_parts.append(unit_name)
    elif asset_name and not asset_class:
        line1_parts.append(asset_name)
    line1 = " ".join(line1_parts).strip().upper()

    # Líneas siguientes: modelo, ubicación, cliente
    cover_block_lines = [
        line1,
        asset_model.upper() if asset_model else "",
        location_name.upper() if location_name else "",
        client_name.upper() if client_name else "",
    ]
    cover_block_lines = [ln for ln in cover_block_lines if ln]

    # Separador horizontal sutil arriba del bloque del activo
    story.append(
        HRFlowable(
            width="40%",
            thickness=0.7,
            color=colors.HexColor("#94a3b8"),
            spaceBefore=4,
            spaceAfter=14,
            hAlign="CENTER",
        )
    )

    for idx, line in enumerate(cover_block_lines):
        # La primera línea (clase + tag) va más grande, las demás un poco menores
        font_size = 24 if idx == 0 else 16
        leading = 28 if idx == 0 else 20
        story.append(
            Paragraph(
                _paragraph_safe(line),
                ParagraphStyle(
                    name=f"WMCoverBlock_{idx}",
                    parent=styles["Normal"],
                    fontName=PDF_FONT_BOLD,
                    fontSize=font_size,
                    leading=leading,
                    alignment=TA_CENTER,
                    textColor=colors.HexColor("#0f172a"),
                    spaceAfter=2,
                ),
            )
        )

    # Si hay descripción del tren acoplado, sub-cabecera centrada en regular
    train_text = (meta.get("train_description") or "").strip()
    if train_text:
        story.append(Spacer(1, 0.30 * cm))
        story.append(
            Paragraph(
                _paragraph_safe(train_text),
                ParagraphStyle(
                    name="WMCoverTrain",
                    parent=styles["Normal"],
                    fontName=PDF_FONT_REGULAR,
                    fontSize=10.5,
                    leading=14,
                    alignment=TA_CENTER,
                    textColor=colors.HexColor("#475569"),
                    spaceBefore=4,
                    spaceAfter=3,
                ),
            )
        )

    # Separador horizontal sutil abajo del bloque del activo
    story.append(
        HRFlowable(
            width="40%",
            thickness=0.7,
            color=colors.HexColor("#94a3b8"),
            spaceBefore=14,
            spaceAfter=14,
            hAlign="CENTER",
        )
    )

    # Aire grande antes del bloque de firmas (estética SIGA: las firmas
    # quedan en el tercio inferior del cover, no apretadas al activo)
    story.append(Spacer(1, 3.50 * cm))

    prepared_by = (meta.get("prepared_by") or "").strip()
    prepared_role = (meta.get("prepared_role") or "Junior Condition Monitoring Engineer").strip()
    prepared_city = (meta.get("prepared_city") or "Cajicá, Cundinamarca · Colombia").strip()
    reviewed_by = (meta.get("reviewed_by") or "").strip()
    reviewed_role = (meta.get("reviewed_role") or "Machinery Diagnostic Champion").strip()
    reviewed_city = (meta.get("reviewed_city") or "Cajicá, Cundinamarca · Colombia").strip()
    report_date_value = meta.get("report_date") or TODAY_STR
    period_value = (meta.get("period") or "").strip()
    consecutive_value = (meta.get("consecutive") or "").strip()

    # Bloque de firmas en DOS COLUMNAS PARALELAS centradas (estilo SIGA).
    # Cada columna: "Preparado/Revisado por:" en bold + nombre + cargo + ciudad.
    # Si solo hay uno (preparado o revisado), la otra columna queda vacía.
    sig_label_style = ParagraphStyle(
        name="WMCoverSigLabel",
        parent=styles["Normal"],
        fontName=PDF_FONT_BOLD,
        fontSize=10.2,
        leading=13,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=4,
    )
    sig_name_style = ParagraphStyle(
        name="WMCoverSigName",
        parent=styles["Normal"],
        fontName=PDF_FONT_BOLD,
        fontSize=11,
        leading=14,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=2,
    )
    sig_role_style = ParagraphStyle(
        name="WMCoverSigRole",
        parent=styles["Normal"],
        fontName=PDF_FONT_REGULAR,
        fontSize=9.5,
        leading=12,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#374151"),
        spaceAfter=2,
    )
    sig_city_style = ParagraphStyle(
        name="WMCoverSigCity",
        parent=styles["Normal"],
        fontName=PDF_FONT_REGULAR,
        fontSize=9.0,
        leading=11.5,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#64748b"),
    )

    def _build_signature_cell(label: str, name: str, role: str, city: str) -> List[Any]:
        cell: List[Any] = []
        if not name:
            return [Paragraph("", sig_label_style)]
        cell.append(Paragraph(label, sig_label_style))
        cell.append(Paragraph(_paragraph_safe(name), sig_name_style))
        if role:
            cell.append(Paragraph(_paragraph_safe(role), sig_role_style))
        if city:
            cell.append(Paragraph(_paragraph_safe(city), sig_city_style))
        return cell

    if prepared_by or reviewed_by:
        sig_table = Table(
            [[
                _build_signature_cell("Preparado por:", prepared_by, prepared_role, prepared_city),
                _build_signature_cell("Revisado por:", reviewed_by, reviewed_role, reviewed_city),
            ]],
            colWidths=[8.3 * cm, 8.3 * cm],
        )
        sig_table.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        sig_table.hAlign = "CENTER"
        story.append(sig_table)
        # Aire amplio entre firmas y mini-tabla de fecha/consecutivo: que
        # ese bloque quede empujado contra el pie de la portada (estética SIGA).
        # Casi pegado al disclaimer del footer.
        story.append(Spacer(1, 4.50 * cm))

    # Bloque de fecha/periodo/consecutivo como mini-tabla 2 columnas — más
    # profesional y compacto que un párrafo plano. "Periodo evaluado" se
    # OCULTA cuando viene vacío o "No aplica" (estética SIGA).
    meta_rows: List[List[Any]] = []
    label_style = ParagraphStyle(
        name="WMCoverMetaLabel",
        parent=styles["WMMeta"],
        fontName=PDF_FONT_BOLD,
        fontSize=10.0,
        textColor=colors.HexColor("#0f172a"),
    )
    value_style = ParagraphStyle(
        name="WMCoverMetaValue",
        parent=styles["WMMeta"],
        fontName=PDF_FONT_REGULAR,
        fontSize=10.0,
        textColor=colors.HexColor("#111827"),
    )
    meta_rows.append([
        Paragraph("Fecha del reporte", label_style),
        Paragraph(_paragraph_safe(report_date_value), value_style),
    ])
    if period_value and period_value.lower() not in ("no aplica", "n/a", "-"):
        meta_rows.append([
            Paragraph("Periodo evaluado", label_style),
            Paragraph(_paragraph_safe(period_value), value_style),
        ])
    if consecutive_value:
        meta_rows.append([
            Paragraph("Consecutivo", label_style),
            Paragraph(_paragraph_safe(consecutive_value), value_style),
        ])

    if meta_rows:
        # Tabla CENTRADA (estilo SIGA): label bold + valor regular, columna
        # 1 angosta para alinear con la columna 2 amplia. Líneas finas
        # arriba y abajo, sin colores fuertes.
        meta_tbl = Table(meta_rows, colWidths=[4.4 * cm, 6.6 * cm])
        meta_tbl.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LINEABOVE", (0, 0), (-1, 0), 0.6, colors.HexColor("#cbd5e1")),
            ("LINEBELOW", (0, -1), (-1, -1), 0.6, colors.HexColor("#cbd5e1")),
        ]))
        meta_tbl.hAlign = "CENTER"
        story.append(meta_tbl)

    # (El disclaimer legal se imprime con el footer del canvas — no hace
    # falta repetirlo acá. Eso lo deja consistente con páginas internas.)

    story.append(PageBreak())

    # =========================================================
    # PÁGINA 2 — TABLA DE CONTENIDO (Ciclo 10A.4)
    # =========================================================
    # Se construye automáticamente en la 2ª pasada de multiBuild.
    # El header del título usa WMSection (no WMTOC1) para que NO entre
    # al TOC como auto-referencia.
    story.append(Paragraph("TABLA DE CONTENIDO", styles["WMSection"]))
    story.append(Spacer(1, 0.20 * cm))

    toc = TableOfContents()
    toc.levelStyles = [toc_level0_style, toc_level1_style]
    # Dot leaders + número de página alineado a la derecha.
    # ReportLab dibuja esto automáticamente cuando el levelStyle no
    # define justify especial; el separador se controla con
    # toc.dotsMinLevel.
    toc.dotsMinLevel = 0
    story.append(toc)

    story.append(PageBreak())

    # RESUMEN EJECUTIVO — página inicial después del TOC.
    # Es el "elevator pitch" del reporte: lo primero que el cliente lee
    # de fondo del análisis (después de la portada y la TOC).
    #
    # Hotfix Ciclo 15.1.3 — si el usuario no llenó el Resumen Ejecutivo
    # manualmente, lo auto-redactamos a partir de las figuras cargadas.
    # Antes, si estaba vacio, se OMITIA la seccion entera y el reporte
    # llegaba al cliente sin elevator pitch. Eso es peor que un draft.
    executive_text = (meta.get("executive_summary") or "").strip()

    # Ciclo 17.5.8 — recomputar severidad LIVE en cada generación
    # de PDF. Antes el badge se extraía del texto cacheado en meta
    # (`for known in ...: if known in executive_text`), lo que dejaba
    # la severidad pegada al draft viejo aunque el usuario hubiera
    # añadido figuras nuevas (caso reportado: Trend con Strong change
    # llegaba al PDF pero el Resumen Ejecutivo seguía diciendo
    # "CONDICIÓN ACEPTABLE" porque la prosa cached era previa a la
    # adición del item).
    severity_live = ""
    severity_live_color = ""
    if items:
        try:
            _findings_live = _extract_findings_from_items(items)
            severity_live, severity_live_color = _global_severity(_findings_live)
        except Exception:
            severity_live = ""

    # Si el draft cached menciona una severidad distinta a la live,
    # el draft está stale → regeneramos automáticamente para que la
    # prosa se alinee. Preservamos la edición manual cuando severidad
    # cached coincide con la live (asumimos que el usuario sigue OK
    # con la conclusión).
    cached_severity = ""
    for known in ("CRÍTICA", "ACCIÓN REQUERIDA", "ATENCIÓN", "VIGILANCIA", "CONDICIÓN ACEPTABLE"):
        if executive_text and known in executive_text:
            cached_severity = known
            break

    needs_redraft = bool(items) and (
        not executive_text
        or (severity_live and cached_severity and severity_live != cached_severity)
    )
    if needs_redraft:
        try:
            executive_text = (_autodraft_executive_summary(meta, items) or "").strip()
        except Exception:
            pass

    # Ciclo 17.26 P5+ — Si hay síntesis ejecutiva AI generada, manda
    # ese contenido para el bloque RESUMEN EJECUTIVO. Se renderiza con
    # los estilos clínicos (markdown → ReportLab nativo) en lugar del
    # Paragraph plano del flujo legacy.
    ai_executive_md = (meta.get("ai_executive_summary") or "").strip()

    # Ciclo 17.31 (v3.31.236) — DETECCIÓN DE MISMATCH AI vs LIVE.
    # `needs_redraft` arriba solo regenera el texto AUTODRAFT (legacy).
    # Pero `ai_executive_md` viene de meta["ai_executive_summary"] que
    # se generó en otra sesión y puede tener severidad obsoleta.
    # Si detectamos mismatch entre severidad LIVE y la del texto AI,
    # blanqueamos `ai_executive_md` para forzar regeneración (caller
    # mostrará warning + autodraft del legacy hasta que el usuario
    # regenere la narrativa AI).
    if ai_executive_md and severity_live:
        ai_severity_detected = ""
        for known in ("CRÍTICA", "ACCIÓN REQUERIDA", "ATENCIÓN",
                      "VIGILANCIA", "CONDICIÓN ACEPTABLE"):
            if known in ai_executive_md:
                ai_severity_detected = known
                break
        if ai_severity_detected and ai_severity_detected != severity_live:
            # Mismatch: la narrativa AI hablaba de otra severidad.
            # Descartar el contenido AI y caer al texto legacy
            # (ya regenerado arriba si needs_redraft).
            try:
                meta["ai_executive_summary_stale"] = ai_executive_md
            except Exception:
                pass
            ai_executive_md = ""

    if executive_text or ai_executive_md:
        story.append(Paragraph("RESUMEN EJECUTIVO", styles["WMTOC1"]))

        # Cinta de severidad: usamos SIEMPRE la live (live wins).
        severity_label = severity_live or cached_severity
        if severity_label:
            color_map = {
                "CRÍTICA": "#dc2626",
                "ACCIÓN REQUERIDA": "#ea580c",
                "ATENCIÓN": "#f59e0b",
                "VIGILANCIA": "#84cc16",
                "CONDICIÓN ACEPTABLE": "#16a34a",
            }
            severity_color = color_map.get(severity_label, "#475569")
            severity_style = ParagraphStyle(
                name="WMExecSeverity",
                parent=styles["Normal"],
                fontName=PDF_FONT_BOLD,
                fontSize=11.5,
                leading=14,
                alignment=TA_CENTER,
                textColor=colors.white,
                backColor=colors.HexColor(severity_color),
                borderPadding=(8, 10, 8, 10),
                spaceAfter=12,
            )
            story.append(Paragraph(f"Estado global: {severity_label}", severity_style))

        # Ciclo 15.1.5 — esquemático VIVO en el Resumen Ejecutivo:
        # cuando hay una Asset Instance activa con Sensor Map y
        # severidad disponible (sesion del Tabular o cálculo legacy),
        # mostramos el HEATMAP del tren con valores Overall por plano
        # coloreados por severidad, EN LUGAR DE la imagen estática.
        # El esquemático asi entrega información — el cliente abre la
        # primera página y ve de un vistazo cuáles cojinetes están en
        # alarma con sus valores numéricos, no solo decoración.
        #
        # Si no hay Sensor Map / sesión de Tabular / signals → fallback
        # al schematic_png estático del Vault (Ciclo 14a).
        sch_doc_id = (meta.get("schematic_doc_id") or "").strip()
        sch_inst_id = (meta.get("schematic_instance_id") or "").strip()

        # Helper local para insertar caption y cerrar el bloque
        def _emit_train_caption(label_text: str, alive: bool):
            sch_caption_style = ParagraphStyle(
                name="WMSchematicCaption_RE",
                parent=styles["WMMeta"],
                fontName=PDF_FONT_REGULAR,
                fontSize=8.8,
                leading=11,
                alignment=TA_CENTER,
                textColor=colors.HexColor("#475569"),
                spaceBefore=2,
                spaceAfter=10,
            )
            story.append(Paragraph(label_text, sch_caption_style))

        rendered_alive_schematic = False

        if sch_inst_id:
            try:
                from core.instance_state import (
                    get_instance,
                    compose_train_description,
                    get_instance_document_bytes,
                )
                from core.sensor_diagram import (
                    render_sensor_map_diagram,
                    render_on_schematic,
                )
                from core.machine_severity import build_severity_table

                _re_inst = get_instance(sch_inst_id)
                if _re_inst is not None and getattr(_re_inst, "sensors", None):
                    _re_signals = st.session_state.get("signals", {}) or {}
                    _re_df = build_severity_table(_re_inst.sensors, _re_signals)
                    if _re_df is not None and not _re_df.empty:
                        _re_sev = dict(zip(
                            _re_df["Label"].astype(str),
                            _re_df["Status"].astype(str),
                        ))
                        _re_overall = {}
                        _re_unit = {}
                        for _, _r in _re_df.iterrows():
                            try:
                                _lbl = str(_r["Label"])
                                _re_overall[_lbl] = float(_r.get("Overall") or 0.0)
                                _re_unit[_lbl] = str(_r.get("Unit") or "")
                            except Exception:
                                pass

                        # Ciclo 17.35 (v3.31.242) — INVERSIÓN DE PRIORIDAD.
                        # Antes: si había schematic_png en el Vault, se usaba
                        # esa foto/dibujo como base y los sensores se
                        # pintaban encima. Problema: las fotos viejas del
                        # Vault son típicamente feas o desactualizadas y
                        # tapaban al diagrama vivo que es clase mundial
                        # (compresor Ariel KBK/4 4-cyl, motor Hyundai HNP2,
                        # turbinas LM6000 dibujados con detalle realista).
                        #
                        # Ahora: render_sensor_map_diagram() es el CANON.
                        # Solo se usa el schematic_png estático del Vault
                        # como fallback de último recurso si el diagrama
                        # vivo retorna None (caso rarísimo).
                        _re_drv = " ".join(p for p in [
                            getattr(_re_inst, "driver_manufacturer", ""),
                            getattr(_re_inst, "driver_model", ""),
                        ] if p) or "Driver"
                        _re_dvn = " ".join(p for p in [
                            getattr(_re_inst, "driven_manufacturer", ""),
                            getattr(_re_inst, "driven_model", ""),
                        ] if p) or "Driven"

                        try:
                            from core.sensor_diagram import _infer_machine_kind as _ifk
                            _re_drv_kind = (
                                _ifk(_re_drv) or _ifk(getattr(_re_inst, "asset_class", "")) or "turbine"
                            )
                            _re_dvn_kind = (
                                _ifk(_re_dvn) or _ifk(getattr(_re_inst, "asset_class", "")) or "generator"
                            )
                        except Exception:
                            _re_drv_kind = "turbine"
                            _re_dvn_kind = "generator"

                        # 1) Diagrama vivo (canon)
                        _re_png = render_sensor_map_diagram(
                            _re_inst.sensors,
                            train_label="",
                            driver_label=_re_drv,
                            driven_label=_re_dvn,
                            severity_by_label=_re_sev,
                            overall_by_label=_re_overall,
                            unit_by_label=_re_unit,
                            driver_kind=_re_drv_kind,
                            driven_kind=_re_dvn_kind,
                            figure_width_in=11.5,
                            compact=True,
                        )
                        _used_real_schematic = False

                        # 2) Fallback de último recurso: foto del Vault
                        # con overlays. Solo si el diagrama vivo falló.
                        if _re_png is None and _re_inst.schematic_png:
                            try:
                                _sch_bytes = get_instance_document_bytes(
                                    _re_inst.instance_id, _re_inst.schematic_png
                                )
                                if _sch_bytes:
                                    _re_png_real = render_on_schematic(
                                        _sch_bytes, _re_inst.sensors,
                                        severity_by_label=_re_sev,
                                        overall_by_label=_re_overall,
                                        unit_by_label=_re_unit,
                                    )
                                    if _re_png_real:
                                        _re_png = _re_png_real
                                        _used_real_schematic = True
                            except Exception:
                                pass
                        if _re_png:
                            usable_w = A4[0] - doc.leftMargin - doc.rightMargin
                            target_w = min(15.0 * cm, usable_w)
                            target_h = 7.0 * cm
                            fitted_w, fitted_h = _fit_image_dimensions(
                                _re_png, target_w, target_h
                            )
                            # Ciclo 17.21 — downsize antes del PDF
                            _re_img = Image(
                                BytesIO(_pdf_safe_image_bytes(_re_png) or _re_png),
                                width=fitted_w, height=fitted_h,
                            )
                            _re_img.hAlign = "CENTER"
                            story.append(Spacer(1, 0.10 * cm))
                            story.append(_re_img)
                            gc.collect()  # liberar buffers intermedios
                            train_lbl = (meta.get("train_description")
                                         or compose_train_description(_re_inst)
                                         or "").strip()
                            # Ciclo 15.2.3 — caption corta y ejecutiva.
                            # La imagen + colores + valores hablan solos;
                            # detalle ingenieril va en MAPA DE SENSORES.
                            if train_lbl:
                                cap = f"Estado actual del tren · {train_lbl}"
                            else:
                                cap = "Estado actual del tren acoplado"
                            _emit_train_caption(cap, alive=True)
                            rendered_alive_schematic = True
            except Exception:
                # Si algo falla en el esquematico vivo, caemos al estatico.
                rendered_alive_schematic = False

        if not rendered_alive_schematic and sch_doc_id and sch_inst_id:
            # Fallback Ciclo 14a — esquematico estatico del Vault de la
            # instancia (foto/dibujo subido por el usuario).
            try:
                from core.instance_state import get_instance_document_bytes
                sch_bytes = get_instance_document_bytes(sch_inst_id, sch_doc_id)
                if sch_bytes:
                    usable_w = A4[0] - doc.leftMargin - doc.rightMargin
                    target_w = min(12.5 * cm, usable_w)
                    target_h = 6.0 * cm
                    fitted_w, fitted_h = _fit_image_dimensions(
                        sch_bytes, target_w, target_h
                    )
                    # Ciclo 17.21 — downsize antes del PDF
                    sch_img = Image(
                        BytesIO(_pdf_safe_image_bytes(sch_bytes) or sch_bytes),
                        width=fitted_w, height=fitted_h,
                    )
                    sch_img.hAlign = "CENTER"
                    story.append(Spacer(1, 0.10 * cm))
                    story.append(sch_img)
                    gc.collect()
                    train_lbl = (meta.get("train_description") or "").strip()
                    if train_lbl:
                        _emit_train_caption(
                            f"Esquemático del tren · {train_lbl}",
                            alive=False,
                        )
                    else:
                        _emit_train_caption(
                            "Esquemático del tren acoplado",
                            alive=False,
                        )
            except Exception:
                pass

        # Ciclo 17.26 P5+ — Si hay síntesis ejecutiva AI, la
        # renderizamos con estilos clínicos (markdown → flowables
        # ReportLab nativos: headings, bullets numerados con sangría,
        # negritas reales, párrafos justificados). Se ve como un
        # informe técnico profesional, no como markdown plano.
        # Si no hay AI, fallback al executive_text legacy en un
        # único Paragraph (comportamiento anterior).
        if ai_executive_md:
            ai_exec_flowables = _render_ai_clinical_flowables(
                ai_executive_md, styles
            )
            for fl in ai_exec_flowables:
                story.append(fl)
        elif executive_text:
            story.append(Paragraph(_paragraph_safe(executive_text), styles["WMBody"]))
        story.append(Spacer(1, 0.30 * cm))

        # Ciclo 17.28 — Si el especialista generó la comparación
        # Run-vs-Run, agregamos una nueva sección 'EVOLUCIÓN DESDE
        # LA ÚLTIMA CORRIDA' después del Resumen Ejecutivo. Esto
        # contextualiza al lector con el delta mecánico antes de que
        # entre al detalle de las figuras individuales.
        run_comparison_md = (meta.get("ai_run_comparison") or "").strip()
        if run_comparison_md:
            story.append(PageBreak())
            story.append(Paragraph(
                "EVOLUCIÓN DESDE LA ÚLTIMA CORRIDA", styles["WMTOC1"]
            ))
            # Caption con metadata de comparación: contra qué reporte
            # comparamos y cuántos días transcurrieron.
            run_meta = meta.get("ai_run_comparison_meta") or {}
            prev_consec = str(run_meta.get("prev_consecutive", "") or "")
            prev_date = str(run_meta.get("prev_archived_at", "") or "")
            days_elapsed = run_meta.get("days_elapsed")
            cmp_caption_bits: List[str] = []
            if prev_consec:
                cmp_caption_bits.append(f"Reporte previo: {prev_consec}")
            if prev_date:
                cmp_caption_bits.append(f"Archivado: {prev_date}")
            if days_elapsed is not None:
                cmp_caption_bits.append(
                    f"Intervalo: {days_elapsed} días operativos"
                )
            if cmp_caption_bits:
                _cmp_caption_style = ParagraphStyle(
                    name="WMRunCmpCaption",
                    parent=styles["Normal"],
                    fontName=PDF_FONT_REGULAR,
                    fontSize=9.5,
                    leading=12,
                    alignment=TA_LEFT,
                    textColor=colors.HexColor("#475569"),
                    spaceAfter=10,
                )
                story.append(Paragraph(
                    " · ".join(cmp_caption_bits),
                    _cmp_caption_style,
                ))
            # Render del delta forense con los estilos clínicos
            # (mismo parser que usamos para Síntesis Ejecutiva AI)
            run_cmp_flowables = _render_ai_clinical_flowables(
                run_comparison_md, styles
            )
            for fl in run_cmp_flowables:
                story.append(fl)
            story.append(Spacer(1, 0.30 * cm))

        # Ciclo 17.30 — Proyección de Vida Útil Restante (RUL)
        # Si el especialista generó la estimación, agregamos otra
        # sección después de la Evolución. El cliente VP / Mantto
        # ve el horizonte de intervención sugerido.
        rul_estimate_md = (meta.get("ai_rul_estimate") or "").strip()
        if rul_estimate_md:
            story.append(PageBreak())
            story.append(Paragraph(
                "PROYECCIÓN DE VIDA ÚTIL RESTANTE", styles["WMTOC1"]
            ))
            rul_meta_dict = meta.get("ai_rul_meta") or {}
            rul_n = int(rul_meta_dict.get("n_history", 0) or 0)
            rul_days = int(rul_meta_dict.get("history_days_covered", 0) or 0)
            rul_mono = bool(rul_meta_dict.get("monotonic", False))
            rul_first_sev = str(rul_meta_dict.get("first_severity", "") or "")
            rul_last_sev = str(rul_meta_dict.get("last_severity", "") or "")
            rul_caption_bits: List[str] = []
            if rul_n > 0:
                rul_caption_bits.append(f"Base estadística: {rul_n} reportes")
            if rul_days > 0:
                rul_caption_bits.append(f"Cubriendo {rul_days} días operativos")
            if rul_first_sev and rul_last_sev:
                rul_caption_bits.append(
                    f"Trayectoria: {rul_first_sev} → {rul_last_sev}"
                )
            rul_caption_bits.append(
                "Monotónica" if rul_mono
                else "No-monotónica (validación operativa requerida)"
            )
            if rul_caption_bits:
                _rul_caption_style = ParagraphStyle(
                    name="WMRulCaption",
                    parent=styles["Normal"],
                    fontName=PDF_FONT_REGULAR,
                    fontSize=9.5,
                    leading=12,
                    alignment=TA_LEFT,
                    textColor=colors.HexColor("#475569"),
                    spaceAfter=10,
                )
                story.append(Paragraph(
                    " · ".join(rul_caption_bits),
                    _rul_caption_style,
                ))
            rul_flowables = _render_ai_clinical_flowables(
                rul_estimate_md, styles
            )
            for fl in rul_flowables:
                story.append(fl)
            story.append(Spacer(1, 0.30 * cm))

        # Ciclo 17.34 — Patrones reconocidos en archivo histórico.
        # Si el especialista buscó patterns y hubo matches, agregamos
        # nueva sección. Cada match es un mini-bloque con badge de
        # similarity score + contexto + resolución del caso histórico.
        patterns_matches = meta.get("ai_patterns_matches") or []
        if patterns_matches:
            story.append(PageBreak())
            story.append(Paragraph(
                "PATRONES RECONOCIDOS EN ARCHIVO HISTÓRICO",
                styles["WMTOC1"],
            ))
            patterns_meta = meta.get("ai_patterns_meta") or {}
            n_searched = int(patterns_meta.get("n_history_searched", 0) or 0)
            global_p = str(patterns_meta.get("global_assessment", "") or "")
            _pat_caption_style = ParagraphStyle(
                name="WMPatCaption",
                parent=styles["Normal"],
                fontName=PDF_FONT_REGULAR,
                fontSize=9.5,
                leading=12,
                alignment=TA_LEFT,
                textColor=colors.HexColor("#475569"),
                spaceAfter=8,
            )
            if n_searched > 0:
                story.append(Paragraph(
                    f"Archivo histórico searcheado: {n_searched} reportes "
                    f"accesibles. Memoria institucional aplicada al caso "
                    f"actual.",
                    _pat_caption_style,
                ))
            if global_p:
                _pat_assessment_style = ParagraphStyle(
                    name="WMPatAssessment",
                    parent=styles["WMClinicalBody"],
                    fontName=PDF_FONT_BOLD,
                    spaceAfter=10,
                )
                story.append(Paragraph(
                    _paragraph_safe(global_p),
                    _pat_assessment_style,
                ))
            for idx, m in enumerate(patterns_matches[:5], 1):
                score = int(m.get("similarity_score", 0) or 0)
                band = str(m.get("similarity_band", "") or "")
                color_hex = str(m.get("similarity_color", "#475569") or "#475569")
                consec = str(m.get("consecutive", "") or "")
                date_p = str(m.get("date", "") or "")
                asset_p = str(m.get("asset", "") or "")
                sev_p = str(m.get("severity", "") or "")
                rationale = str(m.get("rationale", "") or "")
                resolution = str(m.get("resolution_summary", "") or "")
                applicability = str(m.get("applicability", "") or "")

                # Header del match: badge de score + identificadores
                header_html = (
                    f"<font color='{color_hex}'><b>{score}% · {band}</b></font>"
                    f" &nbsp;·&nbsp; <b>{consec or 'reporte'}</b> · "
                    f"{date_p} · {asset_p} · severidad {sev_p}"
                )
                story.append(Paragraph(
                    header_html, styles["WMClinicalHeading"]
                ))
                if rationale:
                    story.append(Paragraph(
                        f"<b>Por qué son similares:</b> "
                        f"{_paragraph_safe(rationale)}",
                        styles["WMClinicalBody"],
                    ))
                if resolution and "no documentada" not in resolution.lower():
                    story.append(Paragraph(
                        f"<b>Resolución del caso histórico:</b> "
                        f"{_paragraph_safe(resolution)}",
                        styles["WMClinicalBody"],
                    ))
                if applicability:
                    story.append(Paragraph(
                        f"<b>Aplicabilidad al caso actual:</b> "
                        f"{_paragraph_safe(applicability)}",
                        styles["WMClinicalBody"],
                    ))
                story.append(Spacer(1, 0.12 * cm))
            story.append(Spacer(1, 0.20 * cm))

        story.append(PageBreak())

    # ============================================================
    # Ciclo 15.1.2 — SECCIÓN MAPA DE SENSORES (Machine Map)
    # ------------------------------------------------------------
    # Va inmediatamente después del Resumen Ejecutivo y antes de
    # Recomendaciones. Aparece automáticamente cuando hay una
    # Asset Instance activa (schematic_instance_id en meta) con
    # Sensor Map configurado. Si no hay, se omite limpio.
    #
    # Composición:
    #   - Título "MAPA DE SENSORES" en WMTOC1 (entra al TOC).
    #   - Párrafo de síntesis con totales por zona en prosa
    #     ("De los N sensores configurados, X están en condición
    #      aceptable, Y en atención y Z requieren acción inmediata.").
    #   - Heatmap full (lateral + polar por plano) renderizado con
    #     render_sensor_map_diagram en modo severity_by_label.
    #   - Caption corto bajo la figura.
    #   - Tabla drill-down de sensores con atención requerida
    #     (Alarm + Danger), solo si hay alguno. Si todo está
    #     aceptable, se omite la tabla y se cierra con una línea
    #     positiva.
    # ============================================================
    sm_inst_id = (meta.get("schematic_instance_id") or "").strip()
    if sm_inst_id:
        try:
            from core.instance_state import get_instance, compose_train_description
            from core.sensor_diagram import render_sensor_map_diagram
            from core.machine_severity import build_severity_table, count_status

            sm_instance = get_instance(sm_inst_id)
        except Exception:
            sm_instance = None

        if sm_instance is not None and getattr(sm_instance, "sensors", None):
            try:
                sm_signals = st.session_state.get("signals", {}) or {}
                sm_df = build_severity_table(sm_instance.sensors, sm_signals)
                sm_counts = count_status(sm_df)
                sm_total = sm_counts["total"]

                story.append(Paragraph("MAPA DE SENSORES", styles["WMTOC1"]))

                # Ciclo 15.1.4 — sintesis ingenieril basada en la columna
                # Overall + Status del Tabular (misma data, no calculo
                # paralelo). Mencionamos por NOMBRE los sensores con mayor
                # consumo de margen para que el reporte sea diagnostico,
                # no descriptivo.
                #
                # Calcular % de danger consumido por sensor para ranking
                def _pct_of_danger(row):
                    try:
                        d = float(row.get("Danger", 0) or 0)
                        o = float(row.get("Overall", 0) or 0)
                        return (o / d * 100.0) if d > 0 else 0.0
                    except Exception:
                        return 0.0
                sm_df = sm_df.copy()
                sm_df["_pct_danger"] = sm_df.apply(_pct_of_danger, axis=1)

                # Sensor con mayor consumo (cualquier estado, excepto No Data)
                with_data = sm_df[sm_df["Status"] != "No Data"].copy()
                top_consumer = None
                if not with_data.empty:
                    top_consumer = with_data.sort_values(
                        "_pct_danger", ascending=False
                    ).iloc[0]

                # Construir prosa en bloques
                paras: List[str] = []

                # Encabezado factual con totales — Ciclo 15.1.5: separamos
                # keyphasor (referencia de fase) de los sensores de vibración.
                # Reportarlos como un "sensor de vibración más" da una imagen
                # incorrecta del Sensor Map al cliente.
                n_vib = sm_counts.get("vibration_total", sm_total)
                n_kp = sm_counts.get("keyphasor", 0)

                head_intro = (
                    f"El Sensor Map configurado para esta unidad consta de "
                    f"{n_vib} "
                    f"{'sensor' if n_vib == 1 else 'sensores'} de vibración "
                    f"de monitoreo continuo distribuidos a lo largo del tren "
                    f"acoplado"
                )
                if n_kp > 0:
                    head_intro += (
                        f", complementados por "
                        f"{n_kp} "
                        f"{'señal de referencia de fase (keyphasor) instalada' if n_kp == 1 else 'señales de referencia de fase (keyphasors) instaladas'} "
                        f"sobre el eje del rotor para sincronización de "
                        f"medidas vectoriales (orbits, Polar y Bode) y "
                        f"diagnóstico de fenómenos rotodinámicos según "
                        f"API 670"
                    )

                if sm_counts["no_data"] == n_vib:
                    head_intro += (
                        ". Ninguno de los sensores de vibración cuenta con "
                        "señal cargada en la sesión actual; los marcadores "
                        "del heatmap aparecen en estado neutro hasta que se "
                        "carguen los CSV correspondientes."
                    )
                else:
                    n_eval = n_vib - sm_counts["no_data"]
                    head_intro += (
                        f". De los sensores de vibración, {n_eval} "
                        f"{'cuenta' if n_eval == 1 else 'cuentan'} con señal "
                        f"cargada en la sesión y "
                        f"{'fue evaluado' if n_eval == 1 else 'fueron evaluados'} "
                        f"contra los umbrales individuales de Alarm y Danger "
                        f"definidos por sensor en el Sensor Map (los mismos "
                        f"setpoints que utiliza el módulo Tabular List)."
                    )
                paras.append(head_intro)

                # Distribución de severidad — orientada a lo critico primero
                if sm_counts["danger"] > 0 or sm_counts["alarm"] > 0:
                    sev_parts = []
                    if sm_counts["danger"] > 0:
                        sev_parts.append(
                            f"{sm_counts['danger']} "
                            f"{'sensor supera' if sm_counts['danger'] == 1 else 'sensores superan'} "
                            f"el umbral de Danger y "
                            f"{'requiere' if sm_counts['danger'] == 1 else 'requieren'} "
                            f"acción inmediata"
                        )
                    if sm_counts["alarm"] > 0:
                        sev_parts.append(
                            f"{sm_counts['alarm']} "
                            f"{'se encuentra' if sm_counts['alarm'] == 1 else 'se encuentran'} "
                            f"en zona de Atención (entre Alarm y Danger)"
                        )
                    sev_parts.append(
                        f"{sm_counts['normal']} "
                        f"{'mantiene' if sm_counts['normal'] == 1 else 'mantienen'} "
                        f"condición aceptable por debajo del setpoint de Alarm"
                    )
                    paras.append(
                        "La distribución de severidad indica que " +
                        ", ".join(sev_parts) + "."
                    )

                    # Top consumer por nombre
                    if top_consumer is not None:
                        paras.append(
                            f"El sensor con mayor margen consumido del setpoint "
                            f"de Danger es {top_consumer.get('Label', '')} "
                            f"({top_consumer.get('Plane Label', '') or 'plano '+str(top_consumer.get('Plane',''))}) "
                            f"con un Overall de {float(top_consumer.get('Overall', 0)):.3f} "
                            f"{top_consumer.get('Unit', '')} sobre un Danger de "
                            f"{float(top_consumer.get('Danger', 0)):.3f} "
                            f"{top_consumer.get('Unit', '')}, equivalente al "
                            f"{float(top_consumer.get('_pct_danger', 0)):.0f}% del "
                            f"umbral. Se recomienda priorizar la verificación de "
                            f"este punto en el siguiente ciclo de inspección."
                        )
                else:
                    # Todo aceptable — pero igual mencionar el de mayor margen consumido
                    if top_consumer is not None:
                        paras.append(
                            f"El conjunto de {sm_counts['normal']} "
                            f"{'sensor evaluado se mantiene' if sm_counts['normal'] == 1 else 'sensores evaluados se mantienen'} "
                            f"por debajo del umbral de Alarm definido por sensor; "
                            f"el de mayor margen consumido del setpoint de Danger "
                            f"es {top_consumer.get('Label','')} "
                            f"({top_consumer.get('Plane Label','') or 'plano '+str(top_consumer.get('Plane',''))}) "
                            f"al {float(top_consumer.get('_pct_danger', 0)):.0f}% del "
                            f"setpoint, lo cual se considera dentro del margen de "
                            f"operación normal."
                        )
                    else:
                        paras.append(
                            "No hay sensores en zona de Atención ni de Acción "
                            "Requerida en la sesión actual."
                        )

                # Cierre — explica que es el heatmap
                paras.append(
                    "El heatmap a continuación ubica cada sonda en su posición "
                    "física sobre el tren acoplado y la colorea según el estado "
                    "actual contra los umbrales de Alarm y Danger del propio "
                    "sensor. Los chips circulares bajo cada cojinete indican los "
                    "tipos de sensor presentes en ese plano (sondas de proximidad, "
                    "transductores de velocidad y/o acelerómetros)."
                )

                for _p in paras:
                    story.append(Paragraph(_paragraph_safe(_p), styles["WMBody"]))

                # Heatmap full
                try:
                    sm_drv = " ".join(p for p in [
                        getattr(sm_instance, "driver_manufacturer", ""),
                        getattr(sm_instance, "driver_model", ""),
                    ] if p) or "Driver"
                    sm_dvn = " ".join(p for p in [
                        getattr(sm_instance, "driven_manufacturer", ""),
                        getattr(sm_instance, "driven_model", ""),
                    ] if p) or "Driven"
                    sev_by_label = dict(zip(
                        sm_df["Label"].astype(str),
                        sm_df["Status"].astype(str),
                    ))
                    # Ciclo 17.5.11 — kind adaptativo
                    try:
                        from core.sensor_diagram import _infer_machine_kind as _ifk2
                        _sm_drv_kind = _ifk2(sm_drv) or _ifk2(getattr(sm_instance, "asset_class", "")) or "turbine"
                        _sm_dvn_kind = _ifk2(sm_dvn) or _ifk2(getattr(sm_instance, "asset_class", "")) or "generator"
                    except Exception:
                        _sm_drv_kind = "turbine"
                        _sm_dvn_kind = "generator"
                    sm_png = render_sensor_map_diagram(
                        sm_instance.sensors,
                        train_label=compose_train_description(sm_instance) or "",
                        driver_label=sm_drv,
                        driven_label=sm_dvn,
                        severity_by_label=sev_by_label,
                        driver_kind=_sm_drv_kind,
                        driven_kind=_sm_dvn_kind,
                    )
                    if sm_png:
                        usable_w = A4[0] - doc.leftMargin - doc.rightMargin
                        target_w = min(15.5 * cm, usable_w)
                        target_h = 11.0 * cm
                        fitted_w, fitted_h = _fit_image_dimensions(
                            sm_png, target_w, target_h
                        )
                        # Ciclo 17.21 — downsize antes del PDF
                        sm_img = Image(
                            BytesIO(_pdf_safe_image_bytes(sm_png) or sm_png),
                            width=fitted_w, height=fitted_h,
                        )
                        sm_img.hAlign = "CENTER"
                        story.append(Spacer(1, 0.10 * cm))
                        story.append(sm_img)
                        gc.collect()
                        sm_caption_style = ParagraphStyle(
                            name="WMSensorMapCaption",
                            parent=styles["WMMeta"],
                            fontName=PDF_FONT_REGULAR,
                            fontSize=8.8,
                            leading=11,
                            alignment=TA_CENTER,
                            textColor=colors.HexColor("#475569"),
                            spaceBefore=2,
                            spaceAfter=10,
                        )
                        story.append(Paragraph(
                            "Heatmap del Sensor Map · vista lateral del tren "
                            "y vista polar por plano, coloreadas por severidad "
                            "actual frente a los setpoints individuales del DCS.",
                            sm_caption_style,
                        ))
                except Exception:
                    pass

                # Drill-down de sensores con atención requerida
                critical_df = sm_df[sm_df["Status"].isin(["Alarm", "Danger"])].copy()
                if not critical_df.empty:
                    critical_df = critical_df.sort_values(
                        by=["Status", "Overall"], ascending=[True, False]
                    )
                    table_data = [[
                        Paragraph("<b>Sensor</b>", styles["WMTableHeader"]),
                        Paragraph("<b>Plano</b>", styles["WMTableHeader"]),
                        Paragraph("<b>Tipo</b>", styles["WMTableHeader"]),
                        Paragraph("<b>Overall</b>", styles["WMTableHeader"]),
                        Paragraph("<b>Alarm</b>", styles["WMTableHeader"]),
                        Paragraph("<b>Danger</b>", styles["WMTableHeader"]),
                        Paragraph("<b>Unidad</b>", styles["WMTableHeader"]),
                        Paragraph("<b>Estado</b>", styles["WMTableHeader"]),
                    ]]
                    status_color = {
                        "Alarm": "#f59e0b",
                        "Danger": "#dc2626",
                    }
                    for _, r in critical_df.iterrows():
                        st_color = status_color.get(r["Status"], "#475569")
                        st_label = (
                            "ATENCIÓN" if r["Status"] == "Alarm"
                            else "ACCIÓN REQUERIDA" if r["Status"] == "Danger"
                            else r["Status"]
                        )
                        table_data.append([
                            Paragraph(str(r["Label"]), styles["WMTableCell"]),
                            Paragraph(str(r["Plane Label"] or r["Plane"]), styles["WMTableCell"]),
                            Paragraph(str(r["Type"]).capitalize(), styles["WMTableCell"]),
                            Paragraph(f"{float(r['Overall']):.3f}", styles["WMTableCell"]),
                            Paragraph(f"{float(r['Alarm']):.3f}", styles["WMTableCell"]),
                            Paragraph(f"{float(r['Danger']):.3f}", styles["WMTableCell"]),
                            Paragraph(str(r["Unit"]), styles["WMTableCell"]),
                            Paragraph(
                                f'<font color="{st_color}"><b>{st_label}</b></font>',
                                styles["WMTableCell"],
                            ),
                        ])
                    sm_tbl = Table(
                        table_data,
                        colWidths=[
                            2.6 * cm, 2.4 * cm, 2.0 * cm, 1.7 * cm,
                            1.6 * cm, 1.6 * cm, 1.7 * cm, 2.6 * cm,
                        ],
                        repeatRows=1,
                    )
                    sm_tbl.setStyle(TableStyle([
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                         [colors.HexColor("#ffffff"), colors.HexColor("#f8fafc")]),
                        ("BOX", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
                        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#e2e8f0")),
                        ("LEFTPADDING", (0, 0), (-1, -1), 4),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                        ("TOPPADDING", (0, 0), (-1, -1), 4),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ]))
                    story.append(Spacer(1, 0.15 * cm))
                    drill_caption = ParagraphStyle(
                        name="WMDrillCaption",
                        parent=styles["WMMeta"],
                        fontName=PDF_FONT_BOLD,
                        fontSize=10.0,
                        leading=13,
                        textColor=colors.HexColor("#0f172a"),
                        spaceBefore=2,
                        spaceAfter=4,
                    )
                    story.append(Paragraph(
                        "Sensores con atención requerida",
                        drill_caption,
                    ))
                    story.append(sm_tbl)
                else:
                    if sm_total > 0 and sm_counts["no_data"] < sm_total:
                        story.append(Paragraph(
                            "Todos los sensores con dato cargado en la sesión "
                            "se mantienen en zona aceptable, por debajo del "
                            "umbral de Alarm.",
                            styles["WMBody"],
                        ))

                story.append(Spacer(1, 0.20 * cm))
                story.append(PageBreak())
            except Exception:
                # Si falla cualquier paso de la sección Machine Map,
                # no bloqueamos el reporte.
                pass

    # ============================================================
    # Ciclo 16.2 — SECCIÓN EVOLUCIÓN (comparativo multi-fecha)
    # ------------------------------------------------------------
    # Justo después del MAPA DE SENSORES, antes de Recomendaciones.
    # Aparece automáticamente cuando hay al menos 1 snapshot anterior
    # guardado para esta instancia. Compara la corrida actual contra
    # la última corrida snapshoteada y reporta tendencias por sensor.
    # ============================================================
    if sm_inst_id:
        try:
            from core.instance_history import (
                get_previous_snapshot,
                compare_to_previous,
                trend_arrow,
                save_snapshot as _hist_save_snapshot,
            )
            from core.machine_severity import build_severity_table as _hist_bst
            from core.instance_state import get_instance as _hist_get_instance

            _hist_inst = _hist_get_instance(sm_inst_id)
            if _hist_inst is not None and getattr(_hist_inst, "sensors", None):
                _curr_sev_df = _hist_bst(
                    _hist_inst.sensors,
                    st.session_state.get("signals", {}) or {},
                )
                # Ciclo 16.2.1 — saltea snapshots cuyas lecturas son
                # identicas a la corrida actual (caso: usuario guarda
                # manualmente la corrida actual y despues genera el PDF).
                _prev_snap = get_previous_snapshot(
                    sm_inst_id,
                    skip_identical_to=_curr_sev_df,
                )

                if _prev_snap is not None and _curr_sev_df is not None and not _curr_sev_df.empty:
                    cmp_df = compare_to_previous(_curr_sev_df, _prev_snap)
                    if cmp_df is not None and not cmp_df.empty:
                        story.append(Paragraph("EVOLUCIÓN DESDE LA CORRIDA ANTERIOR", styles["WMTOC1"]))

                        _prev_lbl = _prev_snap.get("corrida_label", "—")
                        _prev_ts = (_prev_snap.get("timestamp", "") or "")[:10]

                        # Síntesis en prosa
                        _trends = cmp_df["Trend"].value_counts().to_dict()
                        _n_total = int(sum(_trends.values()))
                        _n_up_crit = int(_trends.get("up_critical", 0))
                        _n_up = int(_trends.get("up", 0))
                        _n_stable = int(_trends.get("stable", 0))
                        _n_down = int(_trends.get("down", 0))
                        _n_down_g = int(_trends.get("down_good", 0))
                        _n_no_prev = int(_trends.get("no_prev", 0))

                        _intro = (
                            f"Esta corrida se compara contra la corrida anterior "
                            f"registrada bajo la etiqueta «{_prev_lbl}» del "
                            f"{_prev_ts}. La comparación se realiza sensor por "
                            f"sensor sobre la misma instancia y respeta los "
                            f"setpoints individuales del Sensor Map."
                        )
                        story.append(Paragraph(_paragraph_safe(_intro), styles["WMBody"]))

                        # Hallazgos cuantitativos
                        _findings_parts = []
                        if _n_up_crit > 0:
                            _findings_parts.append(
                                f"{_n_up_crit} "
                                f"{'sensor presenta' if _n_up_crit == 1 else 'sensores presentan'} "
                                f"alza significativa (≥20% o cambio de zona hacia "
                                f"Atención/Acción Requerida)"
                            )
                        if _n_up > 0:
                            _findings_parts.append(
                                f"{_n_up} con tendencia ascendente moderada (+5 a +20%)"
                            )
                        if _n_stable > 0:
                            _findings_parts.append(
                                f"{_n_stable} estables (variación menor al 5%)"
                            )
                        if _n_down > 0 or _n_down_g > 0:
                            _down_total = _n_down + _n_down_g
                            _findings_parts.append(
                                f"{_down_total} con tendencia descendente"
                            )
                        if _n_no_prev > 0:
                            _findings_parts.append(
                                f"{_n_no_prev} sin lectura previa para comparar"
                            )

                        if _findings_parts:
                            _findings_text = (
                                "Distribución por tendencia: " +
                                ", ".join(_findings_parts) + "."
                            )
                            story.append(Paragraph(
                                _paragraph_safe(_findings_text), styles["WMBody"]))

                        # Mencionar el sensor con mayor incremento por nombre
                        _crits = cmp_df[cmp_df["Trend"] == "up_critical"].copy()
                        if not _crits.empty and "Delta_pct" in _crits.columns:
                            _crits = _crits.dropna(subset=["Delta_pct"]).sort_values(
                                "Delta_pct", ascending=False
                            )
                            if not _crits.empty:
                                _top = _crits.iloc[0]
                                _lbl = str(_top.get("Label", ""))
                                _pl = str(_top.get("Plane Label", "") or "")
                                _ov_prev = float(_top.get("Overall_prev", 0) or 0)
                                _ov_curr = float(_top.get("Overall", 0) or 0)
                                _dp = float(_top.get("Delta_pct", 0) or 0)
                                _unit = str(_top.get("Unit", "") or "")
                                _st_prev = str(_top.get("Status_prev", "") or "")
                                _st_curr = str(_top.get("Status", "") or "")
                                _msg = (
                                    f"El mayor incremento se observa en el sensor "
                                    f"{_lbl} ({_pl}), que pasó de {_ov_prev:.3f} a "
                                    f"{_ov_curr:.3f} {_unit} ({_dp:+.1f}%)"
                                )
                                if _st_prev != _st_curr:
                                    _msg += (
                                        f", cruzando además del estado «{_st_prev}» "
                                        f"al estado «{_st_curr}»"
                                    )
                                _msg += (
                                    ". Se recomienda priorizar la verificación de "
                                    "este punto y revisar el histórico de Trends "
                                    "para descartar comportamiento transitorio."
                                )
                                story.append(Paragraph(
                                    _paragraph_safe(_msg), styles["WMBody"]))

                        # Tabla compacta solo con sensores con cambio significativo
                        _show = cmp_df[
                            cmp_df["Trend"].isin(["up_critical", "up", "down", "down_good"])
                        ].copy()
                        if not _show.empty:
                            # Orden por trend criticidad
                            _trend_order = {
                                "up_critical": 0, "up": 1,
                                "down": 2, "down_good": 3,
                            }
                            _show = _show.sort_values(
                                by="Trend",
                                key=lambda col: col.map(_trend_order),
                            )
                            _table_data = [[
                                Paragraph("<b>Sensor</b>", styles["WMTableHeader"]),
                                Paragraph("<b>Plano</b>", styles["WMTableHeader"]),
                                Paragraph("<b>Anterior</b>", styles["WMTableHeader"]),
                                Paragraph("<b>Actual</b>", styles["WMTableHeader"]),
                                Paragraph("<b>Δ</b>", styles["WMTableHeader"]),
                                Paragraph("<b>Δ %</b>", styles["WMTableHeader"]),
                                Paragraph("<b>Tendencia</b>", styles["WMTableHeader"]),
                                Paragraph("<b>Estado</b>", styles["WMTableHeader"]),
                            ]]
                            _trend_color_map = {
                                "up_critical": "#dc2626",
                                "up": "#f59e0b",
                                "stable": "#475569",
                                "down": "#16a34a",
                                "down_good": "#059669",
                            }
                            for _, _r in _show.iterrows():
                                _trend = str(_r["Trend"])
                                _color = _trend_color_map.get(_trend, "#475569")
                                _dp_v = _r.get("Delta_pct")
                                _dp_str = f"{float(_dp_v):+.1f}%" if _dp_v is not None and pd.notna(_dp_v) else "—"
                                _d_v = _r.get("Delta")
                                _d_str = f"{float(_d_v):+.3f}" if _d_v is not None and pd.notna(_d_v) else "—"
                                _ovp = _r.get("Overall_prev")
                                _ovp_str = f"{float(_ovp):.3f}" if _ovp is not None and pd.notna(_ovp) else "—"
                                _ovc = _r.get("Overall")
                                _ovc_str = f"{float(_ovc):.3f}" if _ovc is not None and pd.notna(_ovc) else "—"
                                _st_curr = str(_r["Status"] or "")
                                _table_data.append([
                                    Paragraph(str(_r["Label"]), styles["WMTableCell"]),
                                    Paragraph(str(_r["Plane Label"] or _r["Plane"]), styles["WMTableCell"]),
                                    Paragraph(_ovp_str, styles["WMTableCell"]),
                                    Paragraph(_ovc_str, styles["WMTableCell"]),
                                    Paragraph(_d_str, styles["WMTableCell"]),
                                    Paragraph(
                                        f'<font color="{_color}"><b>{_dp_str}</b></font>',
                                        styles["WMTableCell"],
                                    ),
                                    Paragraph(
                                        f'<font color="{_color}"><b>{trend_arrow(_trend)}</b></font>',
                                        styles["WMTableCell"],
                                    ),
                                    Paragraph(_st_curr, styles["WMTableCell"]),
                                ])
                            _evo_tbl = Table(
                                _table_data,
                                colWidths=[
                                    2.5 * cm, 2.6 * cm, 1.9 * cm, 1.9 * cm,
                                    1.7 * cm, 1.6 * cm, 1.4 * cm, 2.4 * cm,
                                ],
                                repeatRows=1,
                            )
                            _evo_tbl.setStyle(TableStyle([
                                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
                                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                                ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                                 [colors.HexColor("#ffffff"), colors.HexColor("#f8fafc")]),
                                ("BOX", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
                                ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#e2e8f0")),
                                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                                ("TOPPADDING", (0, 0), (-1, -1), 4),
                                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                            ]))
                            story.append(Spacer(1, 0.10 * cm))
                            _evo_caption_style = ParagraphStyle(
                                name="WMEvolutionCaption",
                                parent=styles["WMMeta"],
                                fontName=PDF_FONT_BOLD,
                                fontSize=10.0,
                                leading=13,
                                textColor=colors.HexColor("#0f172a"),
                                spaceBefore=2,
                                spaceAfter=4,
                            )
                            story.append(Paragraph(
                                "Sensores con cambio significativo respecto a la "
                                "corrida anterior",
                                _evo_caption_style,
                            ))
                            story.append(_evo_tbl)
                        else:
                            story.append(Paragraph(
                                "Ningún sensor presentó variación significativa "
                                "(±5%) entre corridas. La condición global se "
                                "mantiene estable.",
                                styles["WMBody"],
                            ))

                        # ============================================
                        # Ciclo 16.3 — TRENDS multi-snapshot grid
                        # --------------------------------------------
                        # Mini line charts por sensor crítico mostrando
                        # los últimos N snapshots con threshold lines
                        # Alarm/Danger. Permite ver la trajectoria del
                        # sensor en el tiempo, no solo el delta vs la
                        # corrida anterior.
                        # ============================================
                        try:
                            from core.trend_charts import render_sensor_trend_chart
                            from core.instance_history import get_sensor_history

                            # Top sensores criticos: up_critical y up
                            _crit_for_trends = cmp_df[
                                cmp_df["Trend"].isin(["up_critical", "up"])
                            ].copy()
                            if not _crit_for_trends.empty:
                                _crit_for_trends["__pct"] = _crit_for_trends["Delta_pct"].fillna(0)
                                _crit_for_trends = _crit_for_trends.sort_values(
                                    "__pct", ascending=False
                                ).head(6)

                                # Renderizar chart por sensor
                                _chart_imgs = []
                                for _, _cr in _crit_for_trends.iterrows():
                                    _lbl = str(_cr["Label"])
                                    _plbl = str(_cr.get("Plane Label", "") or "")
                                    _alarm = float(_cr.get("Alarm") or 0)
                                    _danger = float(_cr.get("Danger") or 0)
                                    _unit = str(_cr.get("Unit", "") or "")
                                    # Buscar alarm/danger del sensor en el
                                    # current_sev_df (cmp_df no las trae)
                                    _curr_row = _curr_sev_df[
                                        _curr_sev_df["Label"] == _lbl
                                    ]
                                    if not _curr_row.empty:
                                        _curr_row_d = _curr_row.iloc[0]
                                        _alarm = float(_curr_row_d.get("Alarm") or 0)
                                        _danger = float(_curr_row_d.get("Danger") or 0)
                                        _unit = str(_curr_row_d.get("Unit", "") or "")

                                    # Histórico + corrida actual al final
                                    _current_reading = {
                                        "overall": float(_cr.get("Overall") or 0),
                                        "status": str(_cr.get("Status", "") or ""),
                                        "alarm": _alarm,
                                        "danger": _danger,
                                        "unit": _unit,
                                        "corrida_label": "Actual",
                                    }
                                    _hist = get_sensor_history(
                                        sm_inst_id, _lbl,
                                        max_snapshots=8,
                                        current_reading=_current_reading,
                                    )
                                    if len(_hist) < 2:
                                        # Necesitamos al menos 2 puntos para una tendencia
                                        continue
                                    _png = render_sensor_trend_chart(
                                        _hist,
                                        sensor_label=_lbl,
                                        plane_label=_plbl,
                                        alarm=_alarm, danger=_danger,
                                        unit=_unit,
                                        figure_width_in=4.6,
                                        figure_height_in=2.4,
                                    )
                                    if _png:
                                        _chart_imgs.append(_png)

                                if _chart_imgs:
                                    story.append(Spacer(1, 0.30 * cm))
                                    story.append(Paragraph(
                                        "Trayectoria histórica de los sensores con mayor evolución",
                                        _evo_caption_style,
                                    ))
                                    story.append(Paragraph(
                                        "Cada gráfico muestra el Overall del sensor a "
                                        "lo largo de las últimas corridas snapshoteadas, "
                                        "con líneas horizontales para los setpoints "
                                        "individuales de Alarm (ámbar) y Danger (rojo). "
                                        "El último punto corresponde a la corrida actual. "
                                        "Los markers se colorean por estado de cada "
                                        "snapshot (verde aceptable, ámbar atención, rojo "
                                        "acción requerida).",
                                        styles["WMBody"],
                                    ))

                                    # Grid 2 cols × N filas con los charts
                                    usable_w = A4[0] - doc.leftMargin - doc.rightMargin
                                    cell_w = usable_w / 2.0 - 0.2 * cm
                                    target_h = 4.5 * cm
                                    grid_data = []
                                    for i in range(0, len(_chart_imgs), 2):
                                        row = []
                                        for j in range(2):
                                            if i + j < len(_chart_imgs):
                                                _bytes = _chart_imgs[i + j]
                                                fitted_w, fitted_h = _fit_image_dimensions(
                                                    _bytes, cell_w, target_h
                                                )
                                                # Ciclo 17.21 — downsize antes del PDF
                                                _img = Image(
                                                    BytesIO(_pdf_safe_image_bytes(_bytes) or _bytes),
                                                    width=fitted_w,
                                                    height=fitted_h,
                                                )
                                                _img.hAlign = "CENTER"
                                                gc.collect()
                                                row.append(_img)
                                            else:
                                                row.append("")
                                        grid_data.append(row)

                                    grid_tbl = Table(
                                        grid_data,
                                        colWidths=[cell_w + 0.2 * cm, cell_w + 0.2 * cm],
                                    )
                                    grid_tbl.setStyle(TableStyle([
                                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                                        ("LEFTPADDING", (0, 0), (-1, -1), 2),
                                        ("RIGHTPADDING", (0, 0), (-1, -1), 2),
                                        ("TOPPADDING", (0, 0), (-1, -1), 4),
                                        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                                    ]))
                                    story.append(Spacer(1, 0.10 * cm))
                                    story.append(grid_tbl)
                        except Exception:
                            # Si trends no se puede renderizar, seguimos
                            # con el resto del reporte sin bloquear.
                            pass

                        story.append(Spacer(1, 0.20 * cm))
                        story.append(PageBreak())

                # AUTO-SNAPSHOT: al final, guardar la corrida actual como
                # snapshot para que la próxima corrida tenga referencia.
                # Solo si hay datos (no snapshotear corridas vacías).
                if _curr_sev_df is not None and not _curr_sev_df.empty:
                    try:
                        _auto_label = (
                            (meta.get("consecutive") or "").strip()
                            or f"Reporte {meta.get('report_date', '')}"
                        )
                        _hist_save_snapshot(
                            sm_inst_id,
                            _curr_sev_df,
                            corrida_label=_auto_label,
                            notes="Snapshot automático al generar PDF.",
                        )
                    except Exception:
                        pass
        except Exception:
            # Si falla cualquier cosa de evolución, no bloqueamos el reporte.
            pass

    # Orden SIGA-style (Ciclo 10A): RECOMENDACIONES primero — es lo que el
    # cliente abre y lee de inmediato. Objetivo y Desarrollo van después.
    # Secciones que están vacías se ocultan y la numeración se compacta.
    section_idx = 1
    objective_text = (meta.get("service_objective") or "").strip()
    recommendations_text = (meta.get("recommendations") or "").strip()
    development_text = (meta.get("service_development") or "").strip()

    if recommendations_text:
        story.append(Paragraph(f"{section_idx}. RECOMENDACIONES", styles["WMTOC1"]))
        story.append(Paragraph(_paragraph_safe(recommendations_text), styles["WMBody"]))
        story.append(Spacer(1, 0.12 * cm))
        section_idx += 1

    if objective_text:
        story.append(Paragraph(f"{section_idx}. OBJETIVO DEL SERVICIO", styles["WMTOC1"]))
        story.append(Paragraph(_paragraph_safe(objective_text), styles["WMBody"]))
        story.append(Spacer(1, 0.12 * cm))
        section_idx += 1

    if development_text:
        story.append(Paragraph(f"{section_idx}. DESARROLLO DEL SERVICIO", styles["WMTOC1"]))
        story.append(Paragraph(_paragraph_safe(development_text), styles["WMBody"]))
        story.append(Spacer(1, 0.15 * cm))
        section_idx += 1

    story.append(Paragraph(f"{section_idx}. FIGURAS Y ANÁLISIS", styles["WMTOC1"]))
    story.append(Spacer(1, 0.08 * cm))

    usable_width = A4[0] - doc.leftMargin - doc.rightMargin
    max_img_width = usable_width - 0.6 * cm
    max_img_height = 8.9 * cm

    for idx, item in enumerate(items, start=1):
        png_bytes = None
        figure_render_error = None

        # En cloud SOLO usamos image_bytes ya preparados desde el módulo origen.
        # Ciclo 17.20: lazy load — leer PNG desde disco si no está en memoria.
        png_bytes = read_item_image_bytes(item)
        if png_bytes is None:
            figure_render_error = "La figura no traía image_bytes pre-renderizados"

        caption = f"Figura {idx}. {item.get('title') or f'Figura {idx}'}"
        notes = (item.get("notes") or "").strip()

        # Detecta y separa el bloque "--- RESUMEN ---" para renderizarlo como
        # tabla nativa de ReportLab en vez de texto monoespaciado.
        notes_main, summary_table = _split_notes_and_summary_table(notes)
        notes_flowables = _render_notes_flowables(
            notes_main, styles, summary_table, usable_width
        )

        if png_bytes is not None:
            # Ciclo 17.21 — downsize antes del PDF (este es el render de cada
            # figura del reporte — el caso más impactante para reducir RAM)
            png_bytes_pdf = _pdf_safe_image_bytes(png_bytes) or png_bytes
            img_w, img_h = _fit_image_dimensions(png_bytes_pdf, max_img_width, max_img_height)
            img = Image(BytesIO(png_bytes_pdf), width=img_w, height=img_h)
            img.hAlign = "CENTER"
            gc.collect()

            block = [
                Spacer(1, 0.18 * cm),
                img,
                Paragraph(_paragraph_safe(caption), styles["WMTOC2"]),
                *notes_flowables,
                Spacer(1, 0.24 * cm),
            ]
        else:
            error_text = "No fue posible renderizar esta figura como imagen dentro del entorno de despliegue."
            if figure_render_error:
                error_text += f" Detalle técnico: {figure_render_error}"

            block = [
                Spacer(1, 0.18 * cm),
                Paragraph(_paragraph_safe(caption), styles["WMTOC2"]),
                Paragraph(_paragraph_safe(error_text), styles["WMFigureText"]),
                *notes_flowables,
                Spacer(1, 0.24 * cm),
            ]

        story.append(KeepTogether(block))

    story.append(Spacer(1, 0.40 * cm))
    # Ciclo 10A.4 — multiBuild: 2-3 pasadas para que el TableOfContents
    # converja con los números de página correctos. La primera pasada
    # registra las entradas (afterFlowable las captura); la segunda las
    # imprime con los page numbers reales.
    doc.multiBuild(story, onFirstPage=_draw_cover_page, onLaterPages=_draw_internal_page)
    return buffer.getvalue()


items = _get_items()
meta = st.session_state["report_meta"]

if not meta.get("report_date"):
    meta["report_date"] = TODAY_STR
if not meta.get("prepared_role"):
    meta["prepared_role"] = "Ingeniero de diagnóstico"
if not meta.get("reviewed_role"):
    meta["reviewed_role"] = "Revisión técnica"


from core.ui_theme import page_header as _wm_page_header  # hero compartido (v3.31.313)
_wm_page_header(
    "Reports",
    "Premium editor for technical deliverables. Organize real figures from Spectrum, "
    "Waveform, Orbit and Tabular List, and export a corporate PDF ready for the client.",
)

# =============================================================
# Aviso de DISCO LLENO (ENOSPC). El autosave ya no crashea la página; acá
# avisamos y ofrecemos liberar espacio (borra backups .bak.* y temporales).
# =============================================================
try:
    from core.report_state import (
        PERSIST_LAST_ERROR as _WM_PERSIST_ERR,
        free_disk_space_best_effort as _wm_free_space,
    )
    if _WM_PERSIST_ERR.get("enospc"):
        st.warning("⚠️ **Disk full on the server.** Autosave is "
                   "degraded (your work is still in memory). Free up space and "
                   "save again.")
        if st.button("🧹 Free up disk space", key="wm_free_disk"):
            _freed = _wm_free_space()
            st.success(f"Freed: {_freed.get('backups', 0)} backups and "
                       f"{_freed.get('tmp', 0)} temporary files. Try saving again.")
            _WM_PERSIST_ERR.clear()
            st.rerun()
except Exception:
    pass

# =============================================================
# v3.31.483 — Familias de reporte (pestañas). El "Reporte del Sistema"
# (todo el builder de vibraciones de abajo) queda INTACTO; las familias de
# campo (diario/preliminar/boroscopia/alineación/mecánico) se renderizan
# arriba y hacen st.stop() para no ejecutar el cuerpo del sistema.
# =============================================================
from core.reports_ext.ui import report_family_selector as _rep_family_selector
from core.reports_ext.page import render_report_family as _render_report_family

_rep_family = _rep_family_selector()
if _rep_family != "sistema":
    _render_report_family(_rep_family)
    st.stop()

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f'<div class="wm-kpi"><div class="wm-kpi-label">Figures in report</div><div class="wm-kpi-value">{len(items):,}</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="wm-kpi"><div class="wm-kpi-label">Spectrum blocks</div><div class="wm-kpi-value">{_count_by_type(items, "spectrum"):,}</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="wm-kpi"><div class="wm-kpi-label">Prepared by</div><div class="wm-kpi-value">{meta["prepared_by"] or "-"}</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="wm-kpi"><div class="wm-kpi-label">Consecutive</div><div class="wm-kpi-value">{meta["consecutive"] or "-"}</div></div>', unsafe_allow_html=True)

st.markdown('<div class="wm-divider"></div>', unsafe_allow_html=True)

# =============================================================
# Ciclo 23.168 — AUTO-GUARDADO cada 2 min (pedido Ewdes).
# El estado ya se persiste en cada interacción; este fragment fuerza un
# guardado periódico aunque el especialista esté inactivo, y muestra un
# indicador con la hora del último guardado para dar tranquilidad. Si la
# página se cae / cierra / se va el internet, al reentrar el reporte se
# recupera automáticamente (load_report_state + backups rotativos).
# 2 min (no 5): el reporte es trabajo "vital"; minimiza la pérdida y el
# guardado es barato.
# =============================================================
@st.fragment(run_every="120s")
def _wm_report_autosave_indicator() -> None:
    import datetime as _dt
    try:
        save_report_state(items=st.session_state.get("report_items", []),
                          meta=st.session_state.get("report_meta", {}))
        _ts = _dt.datetime.now().strftime("%H:%M:%S")
        st.session_state["_wm_report_autosave_ts"] = _ts
        ok = True
    except Exception:
        ok = False
    _ts = st.session_state.get("_wm_report_autosave_ts", "—")
    _color = "#10b981" if ok else "#d97706"
    _msg = "Autosaved" if ok else "Retrying save"
    st.markdown(
        f"<div style='display:inline-flex;align-items:center;gap:7px;"
        f"font-size:12px;color:#475569;background:#f1f5f9;border:1px solid #e2e8f0;"
        f"border-radius:8px;padding:5px 12px;margin-bottom:6px;'>"
        f"<span style='width:8px;height:8px;border-radius:50%;background:{_color};'></span>"
        f"💾 {_msg} every 2 min · last: <b>{_ts}</b></div>",
        unsafe_allow_html=True,
    )

_wm_report_autosave_indicator()

st.markdown('<div class="wm-section-title">Report actions</div>', unsafe_allow_html=True)

ga1, ga2, ga3, ga4 = st.columns([1.2, 1.2, 1.2, 3.4])
with ga1:
    if st.button("Refresh figures", use_container_width=True):
        _persist_items(_get_items())
        st.rerun()
with ga2:
    clear_disabled = len(items) == 0
    if st.button("Clear report", use_container_width=True, disabled=clear_disabled):
        _clear_all_items()
        st.rerun()

# Ciclo 17.32 (v3.31.237) — fingerprint de items + meta clave para
# detectar cuando la síntesis AI cached quedó obsoleta. Se incluye:
#   - id + type + signal_id + timestamp de cada item (orden estable),
#   - severidad live actual,
#   - report_date + instance_id (cambios de meta crítico).
# El resultado es un sha1[:10] determinístico. Si el fingerprint actual
# difiere del guardado en wm_ai_exec_summary, sabemos que la narrativa
# AI no corresponde al estado actual del reporte y mostramos warning.
def _compute_items_fingerprint(
    _items: list,
    _meta: dict,
    _severity_live: str = "",
) -> str:
    import hashlib
    import json as _json
    sig = []
    for _it in (_items or []):
        if not isinstance(_it, dict):
            continue
        sig.append((
            str(_it.get("id") or ""),
            str(_it.get("type") or ""),
            str(_it.get("signal_id") or ""),
            str(_it.get("machine") or ""),
            str(_it.get("point") or ""),
            str(_it.get("variable") or ""),
            str(_it.get("timestamp") or ""),
        ))
    sig.sort()  # orden estable independiente del orden de items
    payload = {
        "items": sig,
        "severity_live": _severity_live or "",
        "report_date": (_meta or {}).get("report_date", "")[:10],
        "instance_id": (_meta or {}).get("instance_id", ""),
        "consecutive": (_meta or {}).get("consecutive", ""),
    }
    blob = _json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:10]


# Severidad live actual (idéntico cálculo que en el PDF builder, simplificado).
# Si no hay forma de calcularla, queda "" y el fingerprint solo dependerá de items+meta.
def _read_severity_live() -> str:
    try:
        for known in ("CRÍTICA", "ACCIÓN REQUERIDA", "ATENCIÓN",
                      "VIGILANCIA", "CONDICIÓN ACEPTABLE"):
            if known in (meta.get("severity_live") or ""):
                return known
    except Exception:
        pass
    return ""


# Ciclo 17.26 P5+ — Síntesis Ejecutiva AI
# El botón vive en ga4 para no chocar con los 3 botones existentes.
# El resultado se persiste en session_state['wm_ai_exec_summary'] y
# se inyecta a meta['ai_executive_summary'] justo antes de
# _build_pdf_bytes para que el PDF render lo use en lugar del
# resumen ejecutivo determinístico.
with ga4:
    _ai_exec_disabled = (len(items) == 0) or (not is_ai_available())
    _ai_exec_help = None
    if not is_ai_available():
        _ai_exec_help = (
            "AI unavailable. Configure [anthropic] api_key in the "
            "Streamlit secrets to enable it."
        )
    elif len(items) == 0:
        _ai_exec_help = (
            "Add at least one figure from Spectrum, Trends, etc. "
            "before generating the executive synthesis."
        )
    if st.button(
        "Generate AI Executive Synthesis",
        use_container_width=True,
        disabled=_ai_exec_disabled,
        type="primary" if not _ai_exec_disabled else "secondary",
        help=_ai_exec_help,
        key="wm_ai_exec_btn",
    ):
        with st.spinner("Claude reading all the figures and synthesizing... (8-20 sec)"):
            try:
                _exec_result = generate_executive_summary(
                    items, meta=meta, use_cache=True
                )
            except Exception as exc:
                _exec_result = {
                    "ok": False,
                    "markdown": (
                        f"_Unexpected error:_\n\n```\n"
                        f"{type(exc).__name__}: {exc}\n```"
                    ),
                    "error": str(exc)[:500],
                    "model": "",
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "fallback_used": False,
                }
        # Ciclo 17.32 — guardar fingerprint del estado en el momento de
        # la generación. Si en un rerun futuro el fingerprint cambia
        # (usuario agregó/quitó items, cambió severidad, etc.) la UI
        # detectará la divergencia y avisará al usuario que regenere.
        try:
            _exec_result["fingerprint"] = _compute_items_fingerprint(
                items, meta, _read_severity_live()
            )
        except Exception:
            _exec_result["fingerprint"] = ""
        st.session_state["wm_ai_exec_summary"] = _exec_result

# Mostrar el resultado de la síntesis ejecutiva si está generado
_exec_stored = st.session_state.get("wm_ai_exec_summary")
if _exec_stored is not None:
    with st.expander(
        "AI Executive Synthesis (preview, goes at the start of the PDF)",
        expanded=True,
    ):
        if _exec_stored.get("ok"):
            if _exec_stored.get("fallback_used"):
                st.info(
                    "Generated with the fallback model (Haiku 4.5). "
                    "You can regenerate later when Sonnet recovers capacity."
                )
            # Ciclo 17.32 (v3.31.237) — STALENESS CHECK.
            # Comparar fingerprint guardado vs estado actual del reporte.
            # Si difieren, la síntesis AI describe un estado que ya no
            # corresponde (usuario agregó/quitó items, cambió severidad,
            # cambió report_date, etc.). Mostramos warning prominente
            # con CTA a Regenerar.
            try:
                _fp_now = _compute_items_fingerprint(
                    items, meta, _read_severity_live()
                )
                _fp_saved = str(_exec_stored.get("fingerprint") or "")
                _is_stale = bool(_fp_saved) and (_fp_saved != _fp_now)
            except Exception:
                _is_stale = False

            if _is_stale:
                st.warning(
                    "⚠️ This synthesis was generated with an earlier state "
                    "of the report (items, severity or date changed). "
                    "Click **Regenerate** so it reflects the "
                    "current state before exporting the PDF."
                )

            st.markdown(_exec_stored.get("markdown", ""))
            _model_used_exec = str(_exec_stored.get("model", "") or "")
            if _model_used_exec.startswith("claude-haiku"):
                _in_p_exec, _out_p_exec = 1.0, 5.0
            else:
                _in_p_exec, _out_p_exec = 3.0, 15.0
            _cost_exec = (
                _exec_stored.get("input_tokens", 0) * _in_p_exec
                + _exec_stored.get("output_tokens", 0) * _out_p_exec
            ) / 1_000_000
            _exec_btn_cols = st.columns([1, 1, 5])
            with _exec_btn_cols[0]:
                if st.button(
                    "Regenerate",
                    key="wm_ai_exec_regen",
                    use_container_width=True,
                ):
                    with st.spinner("Regenerating synthesis..."):
                        try:
                            _exec_new = generate_executive_summary(
                                items, meta=meta, use_cache=False
                            )
                            # Ciclo 17.32 — refresh fingerprint con estado actual.
                            try:
                                _exec_new["fingerprint"] = (
                                    _compute_items_fingerprint(
                                        items, meta, _read_severity_live()
                                    )
                                )
                            except Exception:
                                _exec_new["fingerprint"] = ""
                            st.session_state["wm_ai_exec_summary"] = _exec_new
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Error: {exc}")
            with _exec_btn_cols[1]:
                if st.button(
                    "Discard",
                    key="wm_ai_exec_clear",
                    use_container_width=True,
                    help="Reverts to the legacy deterministic executive summary.",
                ):
                    st.session_state["wm_ai_exec_summary"] = None
                    st.rerun()
            with _exec_btn_cols[2]:
                _fb_tag = (
                    " · fallback model"
                    if _exec_stored.get("fallback_used") else ""
                )
                st.caption(
                    f"Model: `{_model_used_exec}` · "
                    f"Tokens: {_exec_stored.get('input_tokens', 0)} → "
                    f"{_exec_stored.get('output_tokens', 0)} · "
                    f"Cost: ~${_cost_exec:.4f}{_fb_tag}"
                )
        else:
            st.error(
                _exec_stored.get("markdown", "Failed to generate the executive synthesis.")
            )
            if st.button("Retry", key="wm_ai_exec_retry"):
                st.session_state["wm_ai_exec_summary"] = None
                st.rerun()

# =============================================================
# Ciclo 17.28 — AI Run-vs-Run Comparison
# =============================================================
# Cuando el reporte actual corresponde a un activo que ya tiene
# reportes archivados, ofrecemos generar el "delta forense" — una
# narrativa de evolución mecánica que compara el reporte actual
# contra el último archivado del mismo instance_id. Aparece como
# botón separado (al lado del de Síntesis Ejecutiva), con preview
# expandible y se inyecta al PDF como sección 'EVOLUCIÓN DESDE LA
# ÚLTIMA CORRIDA' después del Resumen Ejecutivo.
_runcmp_iid = (meta.get("instance_id") or "").strip()
_runcmp_itag = (meta.get("instance_tag") or "").strip()
_runcmp_rdate = (meta.get("report_date") or
                 datetime.now().strftime("%Y-%m-%d"))[:10]

# Buscar reporte anterior una sola vez por sesión por activo.
#
# Ciclo 17.31 (v3.31.236) — antes la key era
# `wm_ai_runcmp_prev_{instance_id}`. Eso causaba dos bugs:
#   1. Si el usuario cambiaba la report_date en el meta del reporte
#      actual, el cache seguía devolviendo el "reporte previo" calculado
#      con la fecha vieja → comparación contra el reporte equivocado.
#   2. Si el usuario navegaba entre instances dentro de la misma
#      sesión, las keys eran únicas, pero al editar otros campos del
#      meta (instance_tag) sin cambiar instance_id, el cache no se
#      invalidaba.
# Solución: incluir report_date + instance_tag en la key para que
# cualquier cambio relevante fuerce un nuevo find_previous_report.
_runcmp_prev_key = (
    f"wm_ai_runcmp_prev_{_runcmp_iid}_{_runcmp_itag}_{_runcmp_rdate}"
)
if _runcmp_prev_key not in st.session_state:
    try:
        st.session_state[_runcmp_prev_key] = find_previous_report(
            viewer_email=_wm_my_email,
            viewer_role=_wm_my_role,
            instance_id=_runcmp_iid,
            instance_tag=_runcmp_itag,
            before_date=_runcmp_rdate,
        )
    except Exception:
        st.session_state[_runcmp_prev_key] = None

_runcmp_prev = st.session_state.get(_runcmp_prev_key)

# UI del botón Run-vs-Run + render del resultado
_runcmp_disabled = (
    not is_ai_available()
    or len(items) == 0
    or _runcmp_prev is None
)
_runcmp_help = None
if not is_ai_available():
    _runcmp_help = (
        "AI unavailable. Configure [anthropic] api_key in the secrets."
    )
elif len(items) == 0:
    _runcmp_help = "Add figures to the report first."
elif _runcmp_prev is None:
    if not _runcmp_iid and not _runcmp_itag:
        _runcmp_help = (
            "No asset selected in the report meta. "
            "Activate an instance from Machinery Library."
        )
    else:
        _runcmp_help = (
            f"No previous archived report was found for asset "
            f"'{_runcmp_itag or _runcmp_iid}'. This is the first report "
            f"for this asset, or the previous one is not archived yet."
        )

_runcmp_label_btn = "Compare with previous report"
if _runcmp_prev:
    _prev_consec = (_runcmp_prev.get("report_meta", {}) or {}).get(
        "consecutive", ""
    )
    _prev_date = _runcmp_prev.get("archived_at", "")[:10]
    if _prev_consec or _prev_date:
        _runcmp_label_btn = (
            f"Compare with previous report "
            f"({_prev_consec or _prev_date})"
        )

_runcmp_btn_cols = st.columns([2.4, 3.6])
with _runcmp_btn_cols[0]:
    if st.button(
        _runcmp_label_btn,
        key="wm_ai_runcmp_btn",
        use_container_width=True,
        disabled=_runcmp_disabled,
        help=_runcmp_help,
        type="primary" if not _runcmp_disabled else "secondary",
    ):
        with st.spinner(
            "Claude comparing with the previous report... (8-20 sec)"
        ):
            try:
                _runcmp_result = generate_run_comparison(
                    _runcmp_prev,
                    current_meta=meta,
                    current_items=items,
                    use_cache=True,
                )
            except Exception as exc:
                _runcmp_result = {
                    "ok": False,
                    "markdown": (
                        f"_Unexpected error:_\n\n```\n"
                        f"{type(exc).__name__}: {exc}\n```"
                    ),
                    "error": str(exc)[:500],
                    "model": "",
                    "cached": False,
                    "fallback_used": False,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cost_usd": 0.0,
                    "prev_archive_id": "",
                    "prev_archived_at": "",
                    "prev_consecutive": "",
                    "days_elapsed": None,
                }
        st.session_state["wm_ai_runcmp_result"] = _runcmp_result

with _runcmp_btn_cols[1]:
    if _runcmp_prev is not None:
        _prev_meta = _runcmp_prev.get("report_meta", {}) or {}
        _prev_sev = _prev_meta.get("executive_severity", "—")
        _prev_iid = _prev_meta.get("instance_tag", "")
        _prev_date_disp = _runcmp_prev.get("archived_at", "")[:10]
        st.caption(
            f"Previous report detected: **{_prev_meta.get('consecutive', '—')}** · "
            f"{_prev_date_disp} · Previous severity: **{_prev_sev}** · "
            f"Asset: {_prev_iid or 'N/A'}"
        )

# Render del resultado del run-compare
_runcmp_stored = st.session_state.get("wm_ai_runcmp_result")
if _runcmp_stored is not None:
    with st.expander(
        "Evolution since the last run (preview, goes to the PDF)",
        expanded=True,
    ):
        if _runcmp_stored.get("ok"):
            if _runcmp_stored.get("fallback_used"):
                st.info(
                    "This comparison was generated with the fallback "
                    "model (Haiku 4.5). Slightly lower quality."
                )
            _days = _runcmp_stored.get("days_elapsed")
            _prev_consec_disp = _runcmp_stored.get("prev_consecutive", "")
            if _days is not None and _prev_consec_disp:
                st.caption(
                    f"Compared against report **{_prev_consec_disp}** "
                    f"({_days} days ago)"
                )
            st.markdown(_runcmp_stored.get("markdown", ""))
            _model_used_rc = str(_runcmp_stored.get("model", "") or "")
            _cost_rc = _runcmp_stored.get("cost_usd", 0.0)
            _rc_btn_cols = st.columns([1, 1, 5])
            with _rc_btn_cols[0]:
                if st.button(
                    "Regenerate",
                    key="wm_ai_runcmp_regen",
                    use_container_width=True,
                ):
                    with st.spinner("Regenerating..."):
                        try:
                            _rc_new = generate_run_comparison(
                                _runcmp_prev,
                                current_meta=meta,
                                current_items=items,
                                use_cache=False,
                            )
                            st.session_state["wm_ai_runcmp_result"] = _rc_new
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Error: {exc}")
            with _rc_btn_cols[1]:
                if st.button(
                    "Discard",
                    key="wm_ai_runcmp_clear",
                    use_container_width=True,
                ):
                    st.session_state["wm_ai_runcmp_result"] = None
                    st.rerun()
            with _rc_btn_cols[2]:
                _fb_tag_rc = (
                    " · fallback model"
                    if _runcmp_stored.get("fallback_used") else ""
                )
                st.caption(
                    f"Model: `{_model_used_rc}` · "
                    f"Tokens: {_runcmp_stored.get('input_tokens', 0)} → "
                    f"{_runcmp_stored.get('output_tokens', 0)} · "
                    f"Cost: ~${_cost_rc:.4f}{_fb_tag_rc}"
                )
        else:
            st.error(
                _runcmp_stored.get(
                    "markdown", "Failed to generate the run-vs-run comparison."
                )
            )
            if st.button("Retry", key="wm_ai_runcmp_retry"):
                st.session_state["wm_ai_runcmp_result"] = None
                st.rerun()

# =============================================================
# Ciclo 17.30 — AI RUL Predictivo (Remaining Useful Life)
# =============================================================
# Estima cuántos días operativos le quedan al activo antes de cruzar
# a la próxima zona de severidad (típicamente CRÍTICA / zona D ISO
# 20816). Requiere al menos 3 reportes históricos del mismo activo
# para emitir percentiles cuantitativos; con menos, el AI emite
# análisis cualitativo y declara "datos insuficientes".
_rul_iid = (meta.get("instance_id") or "").strip()
_rul_itag = (meta.get("instance_tag") or "").strip()

# Cargar la historia del activo una vez por sesión
_rul_history_key = f"wm_ai_rul_history_{_rul_iid or _rul_itag}"
if _rul_history_key not in st.session_state:
    try:
        st.session_state[_rul_history_key] = find_asset_history(
            viewer_email=_wm_my_email,
            viewer_role=_wm_my_role,
            instance_id=_rul_iid,
            instance_tag=_rul_itag,
            limit=30,
        )
    except Exception:
        st.session_state[_rul_history_key] = []

_rul_history = st.session_state.get(_rul_history_key) or []

_rul_disabled = (
    not is_ai_available()
    or len(items) == 0
    or len(_rul_history) < 1
)
_rul_help = None
if not is_ai_available():
    _rul_help = "AI unavailable. Configure [anthropic] api_key."
elif len(items) == 0:
    _rul_help = "Add figures to the report first."
elif len(_rul_history) == 0:
    if not _rul_iid and not _rul_itag:
        _rul_help = (
            "Activate an instance from Machinery Library to "
            "enable the predictive analysis."
        )
    else:
        _rul_help = (
            f"No archived history for asset "
            f"'{_rul_itag or _rul_iid}'. The RUL projection requires "
            f"at least one previous report (ideally 3+ for "
            f"percentiles)."
        )

_rul_label_btn = "Estimate Remaining Useful Life (RUL)"
if _rul_history:
    _rul_label_btn = (
        f"Estimate Remaining Useful Life "
        f"({len(_rul_history)} historical reports)"
    )

_rul_btn_cols = st.columns([2.4, 3.6])
with _rul_btn_cols[0]:
    if st.button(
        _rul_label_btn,
        key="wm_ai_rul_btn",
        use_container_width=True,
        disabled=_rul_disabled,
        help=_rul_help,
        type="primary" if not _rul_disabled else "secondary",
    ):
        with st.spinner(
            "Claude analyzing the historical trajectory and projecting "
            "useful life... (10-25 sec)"
        ):
            try:
                _rul_result = generate_rul_estimate(
                    _rul_history,
                    current_meta=meta,
                    current_items=items,
                    use_cache=True,
                )
            except Exception as exc:
                _rul_result = {
                    "ok": False,
                    "markdown": (
                        f"_Unexpected error:_\n\n```\n"
                        f"{type(exc).__name__}: {exc}\n```"
                    ),
                    "error": str(exc)[:500],
                    "model": "",
                    "cached": False,
                    "fallback_used": False,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cost_usd": 0.0,
                    "n_history": 0,
                    "history_days_covered": 0,
                    "monotonic": False,
                }
        st.session_state["wm_ai_rul_result"] = _rul_result

with _rul_btn_cols[1]:
    if _rul_history:
        _rul_n = len(_rul_history)
        _rul_min = (_rul_history[0].get("archived_at", "") or "")[:10]
        _rul_max = (_rul_history[-1].get("archived_at", "") or "")[:10]
        _rul_quality = (
            "Sufficient for percentiles"
            if _rul_n >= MIN_HISTORY_FOR_RUL
            else f"Qualitative analysis only (minimum {MIN_HISTORY_FOR_RUL} reports for percentiles)"
        )
        st.caption(
            f"History detected: **{_rul_n} reports** between "
            f"{_rul_min} and {_rul_max}. {_rul_quality}"
        )

# Render del resultado RUL
_rul_stored = st.session_state.get("wm_ai_rul_result")
if _rul_stored is not None:
    with st.expander(
        "Remaining useful life projection (preview, goes to the PDF)",
        expanded=True,
    ):
        if _rul_stored.get("ok"):
            if _rul_stored.get("fallback_used"):
                st.info(
                    "This projection was generated with the fallback "
                    "model (Haiku 4.5). Slightly lower quality."
                )
            _n_h = _rul_stored.get("n_history", 0)
            _days_cov = _rul_stored.get("history_days_covered", 0)
            _mono = _rul_stored.get("monotonic", False)
            st.caption(
                f"Statistical base: {_n_h} reports · "
                f"{_days_cov} days covered · "
                f"Monotonic trajectory: {'yes' if _mono else 'no (requires validation)'}"
            )
            st.markdown(_rul_stored.get("markdown", ""))
            _model_used_rul = str(_rul_stored.get("model", "") or "")
            _cost_rul = _rul_stored.get("cost_usd", 0.0)
            _rul_btn_cols2 = st.columns([1, 1, 5])
            with _rul_btn_cols2[0]:
                if st.button(
                    "Regenerate",
                    key="wm_ai_rul_regen",
                    use_container_width=True,
                ):
                    with st.spinner("Regenerating..."):
                        try:
                            _rul_new = generate_rul_estimate(
                                _rul_history,
                                current_meta=meta,
                                current_items=items,
                                use_cache=False,
                            )
                            st.session_state["wm_ai_rul_result"] = _rul_new
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Error: {exc}")
            with _rul_btn_cols2[1]:
                if st.button(
                    "Discard",
                    key="wm_ai_rul_clear",
                    use_container_width=True,
                ):
                    st.session_state["wm_ai_rul_result"] = None
                    st.rerun()
            with _rul_btn_cols2[2]:
                _fb_tag_rul = (
                    " · fallback model"
                    if _rul_stored.get("fallback_used") else ""
                )
                st.caption(
                    f"Model: `{_model_used_rul}` · "
                    f"Tokens: {_rul_stored.get('input_tokens', 0)} → "
                    f"{_rul_stored.get('output_tokens', 0)} · "
                    f"Cost: ~${_cost_rul:.4f}{_fb_tag_rul}"
                )
        else:
            st.error(
                _rul_stored.get(
                    "markdown", "Failed to generate the RUL projection."
                )
            )
            if st.button("Retry", key="wm_ai_rul_retry"):
                st.session_state["wm_ai_rul_result"] = None
                st.rerun()

# =============================================================
# Ciclo 17.34 — Pattern Memory (memoria institucional)
# =============================================================
# Busca en TODO el archivo histórico accesible patrones mecánicos
# similares al reporte que se está preparando — sin importar que
# sean del mismo activo o cliente. Convierte el archivo en un
# cerebro colectivo: cada reporte que se archiva suma valor a
# todos los próximos análisis.
_patterns_disabled = (
    not is_ai_available()
    or len(items) == 0
)
_patterns_help = None
if not is_ai_available():
    _patterns_help = "AI unavailable. Configure [anthropic] api_key."
elif len(items) == 0:
    _patterns_help = "Add figures to the report first."

_patterns_btn_cols = st.columns([2.4, 3.6])
with _patterns_btn_cols[0]:
    if st.button(
        "Search for similar patterns in the report archive",
        key="wm_ai_patterns_btn",
        use_container_width=True,
        disabled=_patterns_disabled,
        help=_patterns_help,
        type="primary" if not _patterns_disabled else "secondary",
    ):
        with st.spinner(
            "Claude searching for similar mechanical patterns in the "
            "report archive... (10-25 sec)"
        ):
            try:
                _patterns_result = find_similar_patterns(
                    current_meta=meta,
                    current_items=items,
                    viewer_email=_wm_my_email,
                    viewer_role=_wm_my_role,
                    top_k=5,
                    use_cache=True,
                )
            except Exception as exc:
                _patterns_result = {
                    "ok": False,
                    "matches": [],
                    "global_assessment": (
                        f"_Unexpected error:_\n\n```\n"
                        f"{type(exc).__name__}: {exc}\n```"
                    ),
                    "n_history_searched": 0,
                    "model": "",
                    "cached": False,
                    "fallback_used": False,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cost_usd": 0.0,
                }
        st.session_state["wm_ai_patterns_result"] = _patterns_result

with _patterns_btn_cols[1]:
    st.caption(
        "Recognizes similar mechanical patterns in other archived "
        "reports (same client, other assets). Institutional "
        "memory: every report you archive adds value to future "
        "analyses."
    )

# Render del resultado
_patterns_stored = st.session_state.get("wm_ai_patterns_result")
if _patterns_stored is not None:
    with st.expander(
        "Patterns recognized in the report archive (preview, goes to the PDF)",
        expanded=True,
    ):
        if _patterns_stored.get("ok"):
            if _patterns_stored.get("fallback_used"):
                st.info(
                    "Results generated with the fallback model "
                    "(Haiku 4.5). Slightly lower quality."
                )
            _n_hist_p = _patterns_stored.get("n_history_searched", 0)
            _global_p = _patterns_stored.get("global_assessment", "")
            st.caption(
                f"Archive searched: {_n_hist_p} accessible reports."
            )
            if _global_p:
                st.markdown(f"**{_global_p}**")
            _matches_p = _patterns_stored.get("matches", []) or []
            if not _matches_p:
                st.info(
                    "No patterns with significant similarity (>40%) "
                    "were identified in the accessible archive. "
                    "This may be a novel case for the monitoring "
                    "program, or the archive needs more comparable "
                    "reports."
                )
            else:
                for i, m in enumerate(_matches_p, 1):
                    _score = m.get("similarity_score", 0)
                    _band = m.get("similarity_band", "")
                    _color = m.get("similarity_color", "#475569")
                    _consec = m.get("consecutive", "")
                    _date = m.get("date", "")
                    _asset = m.get("asset", "")
                    _sev = m.get("severity", "")
                    _rationale = m.get("rationale", "")
                    _resolution = m.get("resolution_summary", "")
                    _applicability = m.get("applicability", "")
                    _archive_id = m.get("archive_id", "")
                    # Header del match con badge de score
                    st.markdown(
                        f"<div style='padding:10px 14px; "
                        f"border-radius:8px; background:#f8fafc; "
                        f"border-left:5px solid {_color}; "
                        f"margin-bottom:6px;'>"
                        f"<span style='display:inline-block; "
                        f"padding:3px 10px; border-radius:999px; "
                        f"background:{_color}; color:white; "
                        f"font-weight:700; font-size:0.85rem; "
                        f"margin-right:10px;'>"
                        f"{_score}% · {_band}</span>"
                        f"<b>{_consec or 'report'}</b> · {_date}"
                        f" · {_asset} · severity {_sev}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    if _rationale:
                        st.markdown(f"**Why they are similar:** {_rationale}")
                    if _resolution and _resolution != "(resolución no documentada en el archivo)":
                        st.markdown(f"**Resolution of the historical case:** {_resolution}")
                    if _applicability:
                        st.markdown(f"**Applicability to the current case:** {_applicability}")
                    # Botón de descarga del PDF citado
                    try:
                        from core.reports_archive import get_archived_pdf_bytes
                        _ref_pdf = get_archived_pdf_bytes(
                            archive_id=_archive_id,
                            viewer_email=_wm_my_email,
                            viewer_role=_wm_my_role,
                        )
                    except Exception:
                        _ref_pdf = None
                    if _ref_pdf:
                        _safe_fname = (_consec or _archive_id.replace("/", "_"))[:40]
                        st.download_button(
                            f"Download the historical case PDF",
                            data=_ref_pdf,
                            file_name=f"{_safe_fname}.pdf",
                            mime="application/pdf",
                            key=f"wm_pat_dl_{_archive_id}_{i}",
                            use_container_width=False,
                        )
                    if i < len(_matches_p):
                        st.markdown("---")
            # Caption final con metadata técnica
            _model_used_p = str(_patterns_stored.get("model", "") or "")
            _cost_p = _patterns_stored.get("cost_usd", 0.0)
            _fb_tag_p = (
                " · modelo de respaldo"
                if _patterns_stored.get("fallback_used") else ""
            )
            _patterns_btn_cols2 = st.columns([1, 1, 5])
            with _patterns_btn_cols2[0]:
                if st.button(
                    "Regenerate",
                    key="wm_ai_patterns_regen",
                    use_container_width=True,
                ):
                    with st.spinner("Regenerating..."):
                        try:
                            _new = find_similar_patterns(
                                current_meta=meta,
                                current_items=items,
                                viewer_email=_wm_my_email,
                                viewer_role=_wm_my_role,
                                top_k=5,
                                use_cache=False,
                            )
                            st.session_state["wm_ai_patterns_result"] = _new
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Error: {exc}")
            with _patterns_btn_cols2[1]:
                if st.button(
                    "Discard",
                    key="wm_ai_patterns_clear",
                    use_container_width=True,
                ):
                    st.session_state["wm_ai_patterns_result"] = None
                    st.rerun()
            with _patterns_btn_cols2[2]:
                st.caption(
                    f"Model: `{_model_used_p}` · "
                    f"Tokens: {_patterns_stored.get('input_tokens', 0)} → "
                    f"{_patterns_stored.get('output_tokens', 0)} · "
                    f"Cost: ~${_cost_p:.4f}{_fb_tag_p}"
                )
        else:
            st.error(
                _patterns_stored.get(
                    "global_assessment",
                    "Failed to search for similar patterns.",
                )
            )
            if st.button("Retry", key="wm_ai_patterns_retry"):
                st.session_state["wm_ai_patterns_result"] = None
                st.rerun()

pdf_ready = len(items) > 0
pdf_error = None
pdf_bytes: Optional[bytes] = None
meta = st.session_state["report_meta"]

# =============================================================
# Ciclo 14a — Panel de status del auto-fill (debug visual)
# =============================================================
# Justo antes del botón "Preparar PDF" mostramos qué se rellenó
# desde la instancia activa. Esto le permite al ingeniero confirmar
# CON SUS PROPIOS OJOS que el esquemático está vinculado y va a
# aparecer en el Resumen Ejecutivo, sin tener que generar el PDF
# y verificar a posteriori.
try:
    from core.instance_selector import get_active_instance_id
    from core.instance_state import get_instance
    _active_id = get_active_instance_id()
    _active_inst = get_instance(_active_id) if _active_id else None
except Exception:
    _active_id = None
    _active_inst = None

with st.expander("Auto-fill from monitored asset", expanded=True):
    if _active_inst is None:
        st.warning(
            "No monitored asset is active. Go to Machinery Library "
            "and activate a machine so its data auto-fills here."
        )
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Asset:** {_active_inst.tag or _active_inst.instance_id}")
            st.caption(f"Client · {meta.get('client') or '—'}")
            st.caption(f"Site · {meta.get('location') or '—'}")
            st.caption(f"Class · {meta.get('asset_class') or '—'}")
            st.caption(f"Model · {meta.get('asset_model') or '—'}")
        with c2:
            train_d = (meta.get('train_description') or '').strip()
            if train_d:
                st.caption("**Train description:**")
                st.caption(train_d)
            else:
                st.caption("**Train description:** —")

            # Estado del esquemático — diagnóstico crítico
            sch_doc = (meta.get("schematic_doc_id") or "").strip()
            sch_inst = (meta.get("schematic_instance_id") or "").strip()
            inst_sch = (_active_inst.schematic_png or "").strip()

            if sch_doc and sch_inst:
                # Validar que el doc realmente exista y traiga bytes
                try:
                    from core.instance_state import get_instance_document_bytes
                    test_bytes = get_instance_document_bytes(sch_inst, sch_doc)
                    if test_bytes:
                        st.success(
                            f"✓ Schematic ready for the Executive Summary "
                            f"({len(test_bytes) // 1024} KB)"
                        )
                    else:
                        st.error(
                            f"✗ schematic_doc_id present ({sch_doc}) but "
                            f"the Vault file could not be read. "
                            f"Was the document deleted?"
                        )
                except Exception as e:
                    st.error(f"✗ Error reading schematic: {e}")
            elif inst_sch:
                # La instancia tiene schematic_png pero el meta no se rellenó
                st.warning(
                    f"The asset has schematic_png={inst_sch[:20]}... "
                    f"but the report meta didn't take it. Click "
                    f"'Reset auto-fill' below to force a reload."
                )
                if st.button("Reset auto-fill from asset", key="reset_autofill"):
                    meta["schematic_doc_id"] = ""
                    meta["schematic_instance_id"] = ""
                    _autofill_report_meta_from_active_instance()
                    st.rerun()
            else:
                st.error(
                    "✗ The asset has NO main schematic linked. "
                    "Go to Machinery Library → your active machine → "
                    "Edit metadata → Schematic tab → select your PNG/JPG → save."
                )

with ga3:
    if st.button("Prepare PDF", use_container_width=True, disabled=not pdf_ready):
        try:
            # Ciclo 17.26 P5+ — Si el especialista generó síntesis
            # ejecutiva AI, la inyectamos al meta para que el PDF la
            # use en el bloque RESUMEN EJECUTIVO.
            _exec_for_pdf = st.session_state.get("wm_ai_exec_summary")
            if (_exec_for_pdf
                    and _exec_for_pdf.get("ok")
                    and _exec_for_pdf.get("markdown")):
                meta["ai_executive_summary"] = str(
                    _exec_for_pdf.get("markdown", "")
                ).strip()
            else:
                meta["ai_executive_summary"] = ""

            # Ciclo 17.28 — Si el especialista generó la comparación
            # Run-vs-Run, inyectamos el markdown + metadata para que
            # el PDF agregue la sección 'EVOLUCIÓN DESDE LA ÚLTIMA
            # CORRIDA' después del RESUMEN EJECUTIVO.
            _runcmp_for_pdf = st.session_state.get("wm_ai_runcmp_result")
            if (_runcmp_for_pdf
                    and _runcmp_for_pdf.get("ok")
                    and _runcmp_for_pdf.get("markdown")):
                meta["ai_run_comparison"] = str(
                    _runcmp_for_pdf.get("markdown", "")
                ).strip()
                meta["ai_run_comparison_meta"] = {
                    "prev_consecutive": _runcmp_for_pdf.get(
                        "prev_consecutive", ""
                    ),
                    "prev_archived_at": _runcmp_for_pdf.get(
                        "prev_archived_at", ""
                    ),
                    "days_elapsed": _runcmp_for_pdf.get("days_elapsed"),
                }
            else:
                meta["ai_run_comparison"] = ""
                meta["ai_run_comparison_meta"] = {}

            # Ciclo 17.30 — Si el especialista generó la proyección RUL,
            # inyectamos al meta para que el PDF agregue la sección
            # 'PROYECCIÓN DE VIDA ÚTIL RESTANTE' después de Evolución.
            _rul_for_pdf = st.session_state.get("wm_ai_rul_result")
            if (_rul_for_pdf
                    and _rul_for_pdf.get("ok")
                    and _rul_for_pdf.get("markdown")):
                meta["ai_rul_estimate"] = str(
                    _rul_for_pdf.get("markdown", "")
                ).strip()
                meta["ai_rul_meta"] = {
                    "n_history": _rul_for_pdf.get("n_history", 0),
                    "history_days_covered": _rul_for_pdf.get(
                        "history_days_covered", 0
                    ),
                    "monotonic": _rul_for_pdf.get("monotonic", False),
                    "first_severity": _rul_for_pdf.get("first_severity", ""),
                    "last_severity": _rul_for_pdf.get("last_severity", ""),
                }
            else:
                meta["ai_rul_estimate"] = ""
                meta["ai_rul_meta"] = {}

            # Ciclo 17.34 — Si el especialista buscó patrones similares
            # en archivo histórico, inyectamos los matches al meta para
            # que el PDF agregue la sección 'PATRONES RECONOCIDOS EN
            # ARCHIVO HISTÓRICO' después de RUL.
            _patterns_for_pdf = st.session_state.get("wm_ai_patterns_result")
            if (_patterns_for_pdf
                    and _patterns_for_pdf.get("ok")
                    and _patterns_for_pdf.get("matches")):
                meta["ai_patterns_matches"] = _patterns_for_pdf.get(
                    "matches", []
                )
                meta["ai_patterns_meta"] = {
                    "n_history_searched": _patterns_for_pdf.get(
                        "n_history_searched", 0
                    ),
                    "global_assessment": _patterns_for_pdf.get(
                        "global_assessment", ""
                    ),
                }
            else:
                meta["ai_patterns_matches"] = []
                meta["ai_patterns_meta"] = {}

            pdf_bytes = _build_pdf_bytes(meta, items)
            st.session_state["report_pdf_bytes"] = pdf_bytes
            st.session_state["report_pdf_error"] = None

            # ─────────────────────────────────────────────────────
            # Ciclo 17.13 — Persistir severidad ejecutiva al Vault
            # ─────────────────────────────────────────────────────
            # Después de generar el PDF con éxito, recomputamos la
            # severity_live (igual que hace _build_pdf_bytes internamente)
            # y la persistimos en metadata.json del activo activo. Esto
            # alimenta el Home con datos REALES (no heurística) para
            # mostrar dot rojo si el activo está en CRÍTICA, etc.
            try:
                _active_iid = (
                    st.session_state.get("wm_active_instance")
                    or meta.get("instance_id", "")
                    or ""
                ).strip()
                if _active_iid:
                    _findings = _extract_findings_from_items(items)
                    _sev_label, _sev_color = _global_severity(_findings)
                    _summary = (
                        meta.get("executive_summary", "")
                        or _findings.get("executive_oneliner", "")
                        or ""
                    )
                    from core.instance_state import update_instance_executive_severity
                    update_instance_executive_severity(
                        instance_id=_active_iid,
                        severity=_sev_label,
                        summary=_summary,
                    )
            except Exception:
                # No interrumpir el flujo del PDF si la persistencia falla
                pass
        except Exception as e:
            st.session_state["report_pdf_bytes"] = None
            st.session_state["report_pdf_error"] = str(e)

pdf_bytes = st.session_state.get("report_pdf_bytes")
pdf_error = st.session_state.get("report_pdf_error")

if pdf_bytes is not None:
    dl_cols = st.columns([0.5, 0.5])
    with dl_cols[0]:
        st.download_button(
            "Download PDF",
            data=pdf_bytes,
            file_name=(meta.get("consecutive") or "watermelon_report").replace(" ", "_") + ".pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    with dl_cols[1]:
        # Ciclo 17.15 — Archivar PDF como copia inmutable
        # Solo el OWNER del reporte puede archivar (no read-only viewers)
        _can_archive = _is_my_own and _wm_my_role in ("admin", "specialist")
        if not _can_archive:
            st.button(
                "Archive report",
                disabled=True, use_container_width=True,
                help="Only the report author can archive it.",
            )
        else:
            with st.popover("Archive report", use_container_width=True):
                st.markdown("**Archive an immutable copy of the PDF**")
                st.caption(
                    "Once archived, the PDF is stored permanently "
                    "in the report repository (it is not overwritten). "
                    "You'll be able to consult it from the 'Report archive' tab."
                )
                _share = st.checkbox(
                    "Share with client (visible to users with role=client)",
                    value=False,
                    key="wm_archive_share_cb",
                )
                _notes = st.text_area(
                    "Notes for this version (optional)",
                    placeholder="e.g. final revision after client feedback; "
                                "includes analysis of Thursday's second trip",
                    key="wm_archive_notes",
                    height=80,
                )
                if st.button("Confirm archive",
                             key="wm_archive_do",
                             type="primary",
                             use_container_width=True):
                    _arch_res = archive_report_pdf(
                        pdf_bytes=pdf_bytes,
                        meta=meta,
                        owner_email=_wm_my_email,
                        shared_with_client=_share,
                        extra_notes=_notes,
                    )
                    if _arch_res.get("ok"):
                        st.success(
                            f"✓ Archived as `{_arch_res['archive_id']}` · "
                            f"{_arch_res['size_human']}"
                        )
                    else:
                        st.error(f"Failed: {_arch_res.get('error', 'error')}")

if pdf_error:
    st.warning(f"PDF export error: {pdf_error}")

st.markdown('<div class="wm-divider"></div>', unsafe_allow_html=True)

drafts = list_report_drafts()

st.markdown('<div class="wm-section-title">Report drafts</div>', unsafe_allow_html=True)

d1, d2, d3, d4 = st.columns([1.9, 1.1, 1.1, 1.1])
with d1:
    default_draft_name = (
        st.session_state.get("report_draft_name_value", "").strip()
        or meta.get("consecutive")
        or meta.get("asset")
        or "reporte_actual"
    )
    if not st.session_state.get("report_draft_name_value"):
        st.session_state["report_draft_name_value"] = default_draft_name

    draft_name = st.text_input(
        "Draft name",
        key="report_draft_name_value",
    )
with d2:
    st.write("")
    st.write("")
    if st.button("Save draft", use_container_width=True):
        saved_name = save_named_report_draft(
            draft_name=draft_name,
            items=st.session_state.get("report_items", []),
            meta=st.session_state.get("report_meta", {}),
        )
        save_report_state(items=st.session_state.get("report_items", []), meta=st.session_state.get("report_meta", {}))
        st.success(f"Draft saved: {saved_name}")
        st.rerun()
with d3:
    st.write("")
    st.write("")
    if st.button("Duplicate draft", use_container_width=True):
        base_name = (draft_name or "reporte_actual").strip()
        duplicate_name = f"{base_name}_copia"
        saved_name = save_named_report_draft(
            draft_name=duplicate_name,
            items=st.session_state.get("report_items", []),
            meta=st.session_state.get("report_meta", {}),
        )
        st.success(f"Draft duplicated: {saved_name}")
        st.rerun()
with d4:
    st.write("")
    st.write("")
    if st.button("New report", use_container_width=True):
        st.session_state["report_items"] = []
        st.session_state["report_meta"] = dict(DEFAULT_REPORT_META)
        st.session_state["report_pdf_bytes"] = None
        st.session_state["report_pdf_error"] = None
        clear_report_state()
        save_report_state(items=st.session_state["report_items"], meta=st.session_state["report_meta"])
        st.rerun()

d5, d6, d7 = st.columns([2.2, 1.1, 1.1])
with d5:
    selected_draft = st.selectbox(
        "Existing drafts",
        options=drafts if drafts else ["—"],
        index=0,
        key="report_selected_draft",
    )
with d6:
    st.write("")
    st.write("")
    if st.button("Load draft", use_container_width=True, disabled=not drafts or selected_draft == "—"):
        loaded = load_named_report_draft(selected_draft)
        merged_meta = dict(DEFAULT_REPORT_META)
        if isinstance(loaded.get("meta"), dict):
            merged_meta.update(loaded["meta"])
        if not merged_meta.get("report_date"):
            merged_meta["report_date"] = TODAY_STR

        st.session_state["report_items"] = loaded.get("items", [])
        st.session_state["report_meta"] = merged_meta
        st.session_state["report_pdf_bytes"] = None
        st.session_state["report_pdf_error"] = None
        save_report_state(items=st.session_state["report_items"], meta=st.session_state["report_meta"])
        st.success(f"Draft loaded: {selected_draft}")
        st.rerun()
with d7:
    st.write("")
    st.write("")
    if st.button("Delete draft", use_container_width=True, disabled=not drafts or selected_draft == "—"):
        delete_named_report_draft(selected_draft)
        if st.session_state.get("report_draft_name_value") == selected_draft:
            pass
        st.success(f"Draft deleted: {selected_draft}")
        st.rerun()

st.markdown('<div class="wm-divider"></div>', unsafe_allow_html=True)

st.markdown('<div class="wm-section-title">Report metadata</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="wm-meta-hint">The report date is loaded automatically with the current date. The evaluated period is optional and worthwhile when the service corresponds to a campaign, operating window or date range.</div>',
    unsafe_allow_html=True,
)

m1, m2, m3 = st.columns(3)
with m1:
    meta["report_title"] = st.text_input("Report title", key="report_meta_report_title", value=meta["report_title"])
with m2:
    meta["client"] = st.text_input("Client", key="report_meta_client", value=meta["client"])
with m3:
    meta["asset"] = st.text_input("Asset / machine", key="report_meta_asset", value=meta["asset"])

m4, m5, m6 = st.columns(3)
with m4:
    meta["unit"] = st.text_input("Unit", key="report_meta_unit", value=meta["unit"])
with m5:
    meta["location"] = st.text_input("Location", key="report_meta_location", value=meta["location"])
with m6:
    meta["consecutive"] = st.text_input("Consecutive", key="report_meta_consecutive", value=meta["consecutive"])

# Ciclo 10A — bloque grande SIGA-style del activo en la portada
m_sa1, m_sa2 = st.columns(2)
with m_sa1:
    meta["asset_class"] = st.text_input(
        "Asset class (cover)",
        key="report_meta_asset_class",
        value=meta.get("asset_class", ""),
        placeholder="TURBOGENERADOR, MOTOR-BOMBA, COMPRESOR…",
        help=(
            "Technical asset class in uppercase — appears large "
            "on the PDF cover, next to the unit. SIGA report style."
        ),
    )
with m_sa2:
    meta["asset_model"] = st.text_input(
        "Model / configuration (cover)",
        key="report_meta_asset_model",
        value=meta.get("asset_model", ""),
        placeholder="LM5000, SGT-300, Brush 54 MW…",
        help="Asset model/configuration. Printed large below the class.",
    )

# Composición del tren acoplado — la mayoría de máquinas reales son trenes
# acoplados (turbina + generador, motor + bomba, motor + compresor, etc.)
meta["train_description"] = st.text_area(
    "Coupled train composition (optional)",
    key="report_meta_train_description",
    value=meta.get("train_description", ""),
    height=80,
    placeholder=(
        "Example: a GE LM6000 aeroderivative turbine coupled through a "
        "double-helical gearbox to a 54 MW Brush electric generator at 3600 rpm. "
        "This text replaces the 'Asset' field on the cover and in the narratives "
        "when filled in, allowing you to describe complete mechanical trains."
    ),
)

m7, m8 = st.columns(2)
with m7:
    meta["prepared_by"] = st.text_input("Prepared by", key="report_meta_prepared_by", value=meta["prepared_by"])
    meta["prepared_role"] = st.text_input("Role of the preparer", key="report_meta_prepared_role", value=meta["prepared_role"])
    meta["prepared_city"] = st.text_input(
        "City / country of the preparer",
        key="report_meta_prepared_city",
        value=meta.get("prepared_city", ""),
        placeholder="Cajicá, Cundinamarca · Colombia",
        help="City and country shown below the name/role on the cover.",
    )
with m8:
    meta["reviewed_by"] = st.text_input("Reviewed by", key="report_meta_reviewed_by", value=meta["reviewed_by"])
    meta["reviewed_role"] = st.text_input("Role of the reviewer", key="report_meta_reviewed_role", value=meta["reviewed_role"])
    meta["reviewed_city"] = st.text_input(
        "City / country of the reviewer",
        key="report_meta_reviewed_city",
        value=meta.get("reviewed_city", ""),
        placeholder="Cajicá, Cundinamarca · Colombia",
    )

st.markdown(
    '<div class="wm-signature-help">These roles will also be shown in the final approval block of the PDF.</div>',
    unsafe_allow_html=True,
)

m9, m10 = st.columns(2)
with m9:
    meta["report_date"] = st.text_input("Report date", key="report_meta_report_date", value=meta["report_date"] or TODAY_STR)
with m10:
    meta["period"] = st.text_input("Evaluated period (optional)", key="report_meta_period", value=meta["period"], placeholder="Example: 2026-04-01 to 2026-04-07")

st.markdown(
    '<div class="wm-highlight-box"><b>Editorial tip</b><br>If the service corresponds to a one-off visit, you can leave the evaluated period empty and use only the report date. If it covers a trend, campaign or operating window, it is worth filling in.</div>',
    unsafe_allow_html=True,
)

meta["report_date"] = meta["report_date"] or TODAY_STR


def _autodraft_sections_from_items(meta_dict: Dict[str, Any], current_items: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    Genera un draft inicial de Objetivo, Desarrollo y Recomendaciones a partir
    de las figuras enviadas al reporte. Sirve para que el ingeniero parta de
    una base prosa-coherente y solo ajuste matices.
    """
    asset = (meta_dict.get("asset") or "").strip()
    client = (meta_dict.get("client") or "").strip()
    train = (meta_dict.get("train_description") or "").strip()
    n = len(current_items)
    types_seen = sorted({(it.get("type") or "figure").strip().lower() for it in current_items})
    type_label_map = {
        "figure": "figuras de análisis", "spectrum": "espectros",
        "waveform": "formas de onda", "orbit": "órbitas",
        "trends": "tendencias", "tabular": "tablas tabulares",
    }
    type_phrase = ", ".join(type_label_map.get(t, t) for t in types_seen) or "figuras de análisis"

    # Cláusula del activo: prioriza la descripción del tren acoplado si la hay,
    # cae al asset simple si no.
    asset_clause = train if train else (asset or "[activo]")
    if not train and asset:
        asset_clause = f"activo {asset}"
    elif train:
        asset_clause = f"tren acoplado conformado por {train}"
    else:
        asset_clause = "activo [activo]"
    client_clause = f" del cliente {client}" if client else ""

    objective = (
        f"Evaluar la condición rotodinámica del {asset_clause}{client_clause} "
        f"a partir de {n} {type_phrase} adquiridas en condición operativa "
        f"mediante el sistema de monitoreo en línea y remoto Watermelon System, "
        f"con el propósito de identificar hallazgos rotodinámicos relevantes y "
        f"emitir recomendaciones técnicas alineadas con las normas internacionales "
        f"aplicables al análisis avanzado de rotordinámica: API 670 para "
        f"instrumentación con sondas de proximidad, API 684 para análisis "
        f"rotodinámico, ISO 20816 para evaluación de severidad de vibración "
        f"mecánica e ISO 21940 para criterios de balanceo."
    )

    development_lines = [
        "El servicio se ejecutó bajo la metodología de diagnóstico avanzado del "
        "sistema Watermelon System, plataforma de monitoreo en línea y remoto "
        "para máquinas rotativas críticas, conforme a las siguientes etapas:",

        "<b>1. Adquisición de datos.</b> La data fue capturada de forma continua "
        "por el sistema Watermelon System a través de las sondas de proximidad y "
        "acelerómetros instalados en el tren mecánico, registrando simultáneamente "
        "las variables operativas relevantes desde el sistema de control distribuido "
        "(DCS) del proceso. Las señales fueron validadas en cuanto a integridad de "
        "estado, continuidad temporal y consistencia de unidades de origen, "
        "preservando la trazabilidad metrológica del registro original.",

        "<b>2. Procesamiento analítico.</b> Cada corrida fue analizada en los "
        "módulos especializados de Watermelon System según la naturaleza del "
        "fenómeno a caracterizar: análisis de respuesta sincrónica 1X (Polar y "
        "Bode con detección automática de velocidades críticas y factor de "
        "amplificación Q según API 684), análisis de posición DC del muñón en el "
        "cojinete (Shaft Centerline con cálculo de eccentricity ratio, attitude "
        "angle, lift-off speed y migración multi-fecha conforme práctica estándar "
        "API 670 para cojinetes hidrodinámicos), y evaluación de severidad de "
        "vibración según ISO 20816 en la parte aplicable a la familia del activo. "
        "Todo el procesamiento respeta la unidad de la fuente original sin "
        "conversiones forzadas que pudieran introducir error de redondeo en la "
        "narrativa.",

        "<b>3. Comparación temporal.</b> Cuando se dispone de más de una corrida "
        "del mismo activo, el sistema realiza comparativos multi-fecha para "
        "detectar evolución del eccentricity ratio, migración del centerline del "
        "muñón, deriva de fase y cambios del factor de amplificación entre "
        "corridas, lo que permite distinguir hallazgos transitorios de tendencias "
        "sostenidas de degradación.",

        "<b>4. Síntesis y recomendaciones.</b> Los hallazgos individuales se "
        "consolidan en un resumen ejecutivo de severidad global, y se emiten "
        "recomendaciones técnicas priorizadas por horizonte de acción "
        "(inmediato, corto plazo y vigilancia rutinaria), correlacionadas con la "
        "información del Document Vault del activo (manuales de fabricante, "
        "reportes históricos de mantenimiento, dimensiones de cojinetes y "
        "parámetros físicos validados).",
    ]
    development = "\n\n".join(development_lines)

    bullets: List[str] = []
    seen = set()
    for it in current_items:
        notes = (it.get("notes") or "").strip()
        if not notes:
            continue
        for line in notes.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            # Captura líneas que ya vienen numeradas en la narrativa de cada figura
            import re
            m = re.match(r"^(\d+)\.\s+(.*)", stripped)
            if not m:
                continue
            text = m.group(2).strip()
            key = text.lower()[:120]
            if key in seen:
                continue
            seen.add(key)
            bullets.append(text)
            if len(bullets) >= 8:
                break
        if len(bullets) >= 8:
            break

    if bullets:
        rec_intro = (
            "A partir de los hallazgos consolidados de las figuras del reporte, "
            "se emiten las siguientes recomendaciones técnicas priorizadas:"
        )
        rec_body = "\n\n".join(f"{i}. {b}" for i, b in enumerate(bullets, start=1))
        recommendations = f"{rec_intro}\n\n{rec_body}"
    else:
        recommendations = (
            "Se recomienda mantener el seguimiento periódico de las variables "
            "monitoreadas y correlacionar contra histórico de mantenimiento y "
            "condición operativa registrada en el DCS."
        )

    return {
        "service_objective": objective,
        "service_development": development,
        "recommendations": recommendations,
    }


# =============================================================
# RESUMEN EJECUTIVO AUTO-REDACTADO
# =============================================================

# Severidad: ranking ordinal para clasificar el estado global del activo
# a partir de los hallazgos individuales de cada figura.
_SCL_SEVERITY_RANK = {
    "HEALTHY": 0,
    "STABLE": 0,
    "MARGINAL_LOW": 1,
    "MARGINAL_HIGH": 2,
    "MINOR_DRIFT": 1,
    "MODERATE_DRIFT": 2,
    "WHIRL_RISK": 3,
    "MAJOR_DRIFT": 3,
    "WIPE_RISK": 4,
}

_ISO_ZONE_RANK = {"A": 0, "B": 1, "C": 2, "D": 3}


def _extract_findings_from_items(current_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Recorre cada figura del reporte y extrae métricas estructuradas a partir
    del texto de las narrativas Cat IV ya escritas. Funciona como lector
    semántico de las narrativas que produjimos en SCL/Polar/Bode.

    Returns:
        Dict con listas de findings clasificadas por tipo. Útil para que
        _compose_executive_summary() arme la prosa de síntesis.
    """
    import re

    findings: Dict[str, Any] = {
        "scl_states": [],         # {classification, e_c, alpha, fig_title}
        "scl_migrations": [],     # {classification, pct_clearance, fig_title}
        "critical_speeds": [],    # {rpm, q, fig_title}
        "iso_zones": [],          # {zone, fig_title}
        "lift_off": [],           # {rpm, margin_pct, fig_title}
        "trend_states": [],       # {status, headline, fig_title} — Ciclo 17.5.7
        "high_priority_actions": [],
        "n_figures": len(current_items),
    }

    for it in current_items:
        # Ciclo 17.5.7 — Trend items tienen un campo estructurado
        # `autodiagnostic` con {status, status_label, headline,
        # prose, recommendations} y un `behavior_summary` con
        # {top_classification}. Lo leemos antes del parsing por
        # regex para que el Resumen Ejecutivo pueda escalar la
        # severidad global cuando haya alarm/action o Strong
        # change. Antes esto se perdía porque el extractor solo
        # leía SCL/Polar/Bode/ISO.
        if str(it.get("type", "")).lower() == "trends":
            autodiag = it.get("autodiagnostic") or {}
            status_str = str(autodiag.get("status") or "").lower()
            behav = it.get("behavior_summary") or {}
            top_class = str(behav.get("top_classification") or "")

            if status_str in ("watch", "alarm", "action") or top_class == "Strong change":
                findings["trend_states"].append({
                    "status": status_str or "unknown",
                    "headline": str(autodiag.get("headline") or "").strip(),
                    "behavior": top_class,
                    "fig_title": str(it.get("title") or ""),
                })

            # Si el autodiag dice action/alarm, levantamos high_priority_actions
            # para que entren al cálculo de _global_severity sin tocar más código.
            if status_str in ("alarm", "action"):
                findings["high_priority_actions"].append({
                    "text": (
                        f"PRIORIDAD ALTA — {autodiag.get('headline','').strip()}"
                        if autodiag.get("headline") else "PRIORIDAD ALTA — Trend en zona Atención/Acción"
                    ),
                    "fig_title": str(it.get("title") or ""),
                })

        notes = (it.get("notes") or "")
        title = (it.get("title") or "")
        if not notes:
            continue

        # Patrón numérico estricto para evitar capturar puntos finales/comas
        NUM = r"(\d+(?:\.\d+)?)"

        # SCL Cat IV classification + e/c + α (formato del comparativo)
        for m in re.finditer(
            rf"eccentricity ratio (?:de\s+|e/c\s*=\s*){NUM}[^.]*?attitude angle (?:de\s+){NUM}°[^.]*?clasificación\s+(\w+)",
            notes,
        ):
            findings["scl_states"].append({
                "e_c": float(m.group(1)),
                "alpha": float(m.group(2)),
                "classification": m.group(3),
                "fig_title": title,
            })

        # SCL panel individual: extrae todas las pares (e/c, classification)
        # cuando la classification aparece en la misma figura aunque no esté
        # explícito el word "clasificación".
        already_titled = any(s["fig_title"] == title for s in findings["scl_states"])
        if not already_titled:
            ec_match = re.search(rf"eccentricity ratio de {NUM}", notes)
            cls_hits = re.findall(r"\b(WIPE_RISK|HEALTHY|MARGINAL_HIGH|MARGINAL_LOW|WHIRL_RISK)\b", notes)
            if ec_match and cls_hits:
                findings["scl_states"].append({
                    "e_c": float(ec_match.group(1)),
                    "alpha": None,
                    "classification": cls_hits[0],
                    "fig_title": title,
                })
            elif ec_match:
                # Inferir desde rango: 0.40-0.70 healthy, <0.30 whirl risk, etc.
                e_c = float(ec_match.group(1))
                if e_c < 0.30:
                    inferred = "WHIRL_RISK"
                elif e_c < 0.40:
                    inferred = "MARGINAL_LOW"
                elif e_c <= 0.70:
                    inferred = "HEALTHY"
                elif e_c <= 0.85:
                    inferred = "MARGINAL_HIGH"
                else:
                    inferred = "WIPE_RISK"
                findings["scl_states"].append({
                    "e_c": e_c, "alpha": None,
                    "classification": inferred, "fig_title": title,
                })

        # SCL migration
        mig = re.search(
            rf"[Mm]igraci[oó]n (\w+) del centerline\s*\({NUM}%\s*del clearance",
            notes,
        )
        if mig:
            severity_word = mig.group(1).lower()
            mig_class = {
                "estable": "STABLE", "menor": "MINOR_DRIFT",
                "moderada": "MODERATE_DRIFT", "mayor": "MAJOR_DRIFT",
            }.get(severity_word, "MINOR_DRIFT")
            findings["scl_migrations"].append({
                "classification": mig_class,
                "pct_clearance": float(mig.group(2)),
                "fig_title": title,
            })

        # Critical speeds + Q factor (Polar/Bode narrative)
        for m in re.finditer(
            rf"velocidad cr[ií]tica.*?(\d[\d,]*)\s*rpm.*?(?:factor\s+Q|Q\s*=)\s*(?:de\s+)?{NUM}",
            notes,
        ):
            try:
                rpm_val = float(m.group(1).replace(",", ""))
                q_val = float(m.group(2))
                findings["critical_speeds"].append({
                    "rpm": rpm_val, "q": q_val, "fig_title": title,
                })
            except ValueError:
                pass

        # ISO 20816 zone (A/B/C/D)
        iso_zone = re.search(r"zona\s+ISO\s+([ABCD])\b", notes)
        if iso_zone:
            findings["iso_zones"].append({
                "zone": iso_zone.group(1), "fig_title": title,
            })

        # Lift-off
        lo = re.search(
            rf"lift[\-\s]off.*?(\d[\d,]*)\s*rpm.*?margen del\s+{NUM}%",
            notes,
        )
        if lo:
            try:
                rpm_lo = float(lo.group(1).replace(",", ""))
                margin = float(lo.group(2))
                findings["lift_off"].append({
                    "rpm": rpm_lo, "margin_pct": margin, "fig_title": title,
                })
            except ValueError:
                pass

        # PRIORIDAD ALTA actions
        for line in notes.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if "PRIORIDAD ALTA" in stripped.upper():
                findings["high_priority_actions"].append({
                    "text": stripped, "fig_title": title,
                })

    return findings


def _global_severity(findings: Dict[str, Any]) -> Tuple[str, str]:
    """
    Calcula severidad global del activo combinando todos los findings.
    Devuelve (severity_label, color_hex).
    """
    rank = 0

    for s in findings["scl_states"]:
        rank = max(rank, _SCL_SEVERITY_RANK.get(s["classification"], 0))
    for m in findings["scl_migrations"]:
        rank = max(rank, _SCL_SEVERITY_RANK.get(m["classification"], 0))
    for z in findings["iso_zones"]:
        rank = max(rank, _ISO_ZONE_RANK.get(z["zone"], 0))
    # Ciclo 17.5.7 — Trend autodiag escala el severity global.
    _TREND_STATUS_RANK = {
        "ok": 0, "watch": 1, "alarm": 2, "action": 3,
    }
    for t in findings.get("trend_states", []) or []:
        rank = max(rank, _TREND_STATUS_RANK.get(t.get("status", "ok"), 0))
        if t.get("behavior") == "Strong change":
            rank = max(rank, 2)  # Strong change → al menos ATENCIÓN
    if findings["high_priority_actions"]:
        rank = max(rank, 3)

    label_map = {
        0: ("CONDICIÓN ACEPTABLE", "#16a34a"),
        1: ("VIGILANCIA", "#84cc16"),
        2: ("ATENCIÓN", "#f59e0b"),
        3: ("ACCIÓN REQUERIDA", "#ea580c"),
        4: ("CRÍTICA", "#dc2626"),
    }
    return label_map.get(rank, label_map[0])


def _compose_executive_summary(meta_dict: Dict[str, Any], findings: Dict[str, Any]) -> str:
    """
    Convierte los findings estructurados en prosa de resumen ejecutivo
    estilo Cat IV. Cuatro bloques: estado global, hallazgos principales,
    severidad y acciones críticas.
    """
    if findings["n_figures"] == 0:
        return ""

    asset = (meta_dict.get("asset") or "").strip()
    train = (meta_dict.get("train_description") or "").strip()
    if train:
        asset_clause = f"tren acoplado conformado por {train}"
    elif asset:
        asset_clause = f"activo {asset}"
    else:
        asset_clause = "activo en evaluación"
    severity_label, _ = _global_severity(findings)

    paragraphs: List[str] = []

    # Bloque 1: estado global
    n_fig = findings["n_figures"]
    n_scl = len(findings["scl_states"])
    n_mig = len(findings["scl_migrations"])
    n_crit = len(findings["critical_speeds"])

    components = []
    if n_scl:
        components.append(f"{n_scl} análisis de Shaft Centerline")
    if n_crit:
        components.append(f"{n_crit} detección{'es' if n_crit != 1 else ''} de velocidades críticas")
    if n_mig:
        components.append(f"{n_mig} comparativ{'os' if n_mig != 1 else 'o'} de migración multi-fecha")
    # Ciclo 17.5.8 — pluralización correcta de "figura" / "figuras"
    _fig_word = "figura" if n_fig == 1 else "figuras"
    composition_clause = ", ".join(components) if components else f"{n_fig} {_fig_word} de análisis"

    paragraphs.append(
        f"El presente reporte sintetiza la condición rotodinámica del "
        f"{asset_clause} a partir de {n_fig} {_fig_word} de análisis adquirida"
        f"{'s' if n_fig != 1 else ''} mediante el sistema de monitoreo en línea "
        f"y remoto Watermelon System, incluyendo {composition_clause}. La "
        f"evaluación combinada de los hallazgos según los criterios técnicos "
        f"aplicables (API 670 / API 684 para análisis rotodinámico, ISO 20816 "
        f"para severidad de vibración) arroja una clasificación global de "
        f"{severity_label}."
    )

    # Bloque 2: hallazgos principales
    hallazgos: List[str] = []

    # SCL states (peor primero)
    if findings["scl_states"]:
        scl_sorted = sorted(
            findings["scl_states"],
            key=lambda s: -_SCL_SEVERITY_RANK.get(s["classification"], 0),
        )
        worst = scl_sorted[0]
        if worst["classification"] == "HEALTHY":
            hallazgos.append(
                f"el centerline del muñón opera en zona hidrodinámica sana "
                f"(e/c = {worst['e_c']:.2f}), lo que indica buen amortiguamiento "
                f"y condición de cojinete adecuada"
            )
        elif worst["classification"] == "WIPE_RISK":
            hallazgos.append(
                f"se detectó eccentricity ratio crítico (e/c = {worst['e_c']:.2f}) "
                f"con riesgo de wipe del babbitt — requiere acción prioritaria"
            )
        elif worst["classification"] == "WHIRL_RISK":
            hallazgos.append(
                f"el centerline presenta eccentricity ratio bajo "
                f"(e/c = {worst['e_c']:.2f}) sugestivo de riesgo de oil whirl, "
                f"lo que amerita verificación del espectro subsíncrono"
            )
        elif worst["classification"] == "MARGINAL_HIGH":
            hallazgos.append(
                f"el centerline opera con eccentricity ratio elevado "
                f"(e/c = {worst['e_c']:.2f}), cerca del límite del clearance"
            )
        else:
            hallazgos.append(
                f"el centerline presenta eccentricity ratio de "
                f"{worst['e_c']:.2f} en condición de margen reducido"
            )

    # Migration
    if findings["scl_migrations"]:
        mig_sorted = sorted(
            findings["scl_migrations"],
            key=lambda m: -_SCL_SEVERITY_RANK.get(m["classification"], 0),
        )
        worst_mig = mig_sorted[0]
        mig_word = {
            "STABLE": "estable", "MINOR_DRIFT": "menor",
            "MODERATE_DRIFT": "moderada", "MAJOR_DRIFT": "mayor",
        }.get(worst_mig["classification"], "menor")
        hallazgos.append(
            f"la comparación multi-fecha del centerline muestra una migración "
            f"{mig_word} ({worst_mig['pct_clearance']:.1f}% del clearance radial)"
        )

    # Critical speeds
    if findings["critical_speeds"]:
        crit_sorted = sorted(findings["critical_speeds"], key=lambda c: -c["q"])
        worst_crit = crit_sorted[0]
        q_descriptor = (
            "con factor Q elevado, indicando bajo amortiguamiento"
            if worst_crit["q"] >= 5.0 else
            "con factor Q moderado, dentro de rangos aceptables"
            if worst_crit["q"] >= 2.5 else
            "con factor Q bajo, indicando buen amortiguamiento"
        )
        hallazgos.append(
            f"se identificó velocidad crítica en {worst_crit['rpm']:.0f} rpm "
            f"con factor Q de {worst_crit['q']:.2f}, {q_descriptor}"
        )

    # ISO zones
    if findings["iso_zones"]:
        worst_zone = max(findings["iso_zones"], key=lambda z: _ISO_ZONE_RANK.get(z["zone"], 0))
        zone_text = {
            "A": "zona A (recién comisionado / aceptable)",
            "B": "zona B (operación sostenida aceptable)",
            "C": "zona C (operación restringida en tiempo)",
            "D": "zona D (no se permite operación sostenida)",
        }.get(worst_zone["zone"], f"zona {worst_zone['zone']}")
        hallazgos.append(
            f"los niveles de vibración se ubican en {zone_text} según ISO 20816"
        )

    # Ciclo 17.5.7 — Trend autodiag findings al hallazgo principal
    _trend_states = findings.get("trend_states", []) or []
    if _trend_states:
        _ts_rank = {"action": 3, "alarm": 2, "watch": 1, "ok": 0, "unknown": 0}
        _worst_t = max(_trend_states, key=lambda t: _ts_rank.get(t.get("status", "ok"), 0))
        _worst_status = _worst_t.get("status", "ok")
        _worst_behavior = _worst_t.get("behavior", "")
        _n_alarming = sum(1 for t in _trend_states if _ts_rank.get(t.get("status", "ok"), 0) >= 2)
        _n_strong = sum(1 for t in _trend_states if t.get("behavior") == "Strong change")

        if _worst_status == "action":
            hallazgos.append(
                "el análisis de tendencias reporta al menos una señal en zona "
                "ACCIÓN REQUERIDA (umbral Danger superado) que demanda intervención inmediata"
            )
        elif _worst_status == "alarm":
            hallazgos.append(
                f"el análisis de tendencias reporta {_n_alarming} señal(es) en zona ATENCIÓN "
                f"(umbral Warning superado) que requieren monitoreo intensivo"
            )
        elif _worst_status == "watch":
            hallazgos.append(
                "el análisis de tendencias muestra al menos una señal en zona de vigilancia "
                "(85–100% del Warning), conviene aumentar la frecuencia de monitoreo"
            )

        if _n_strong > 0 and _worst_behavior == "Strong change":
            hallazgos.append(
                f"el detector de cambio de régimen identifica {_n_strong} transición(es) "
                f"fuerte(s) de comportamiento entre la línea base y la corrida actual"
            )

    if hallazgos:
        paragraphs.append(
            "Los hallazgos principales del análisis combinado son: " +
            "; ".join(hallazgos) + "."
        )

    # Bloque 3: lift-off y soporte
    if findings["lift_off"]:
        lo_avg = sum(l["margin_pct"] for l in findings["lift_off"]) / len(findings["lift_off"])
        lo_min = min(l["margin_pct"] for l in findings["lift_off"])
        if lo_avg >= 80.0:
            lo_clause = (
                f"La velocidad de lift-off detectada deja un margen promedio de "
                f"{lo_avg:.0f}% respecto a la velocidad operativa, lo que confirma "
                f"el establecimiento adecuado del régimen hidrodinámico durante el "
                f"arranque del rotor."
            )
        else:
            lo_clause = (
                f"El margen entre la velocidad de lift-off y la velocidad operativa "
                f"({lo_min:.0f}% mínimo) está por debajo del rango sano típico "
                f"(80–95%), lo que sugiere transición tardía al régimen hidrodinámico."
            )
        paragraphs.append(lo_clause)

    # Bloque 4: acciones críticas
    if findings["high_priority_actions"]:
        actions_text: List[str] = []
        seen_actions = set()
        for a in findings["high_priority_actions"][:3]:
            key = a["text"].lower()[:80]
            if key in seen_actions:
                continue
            seen_actions.add(key)
            actions_text.append(a["text"])
        if actions_text:
            paragraphs.append(
                "El análisis identifica las siguientes acciones de prioridad alta "
                "que deben ser ejecutadas en el corto plazo:\n\n" +
                "\n\n".join(f"• {a}" for a in actions_text)
            )
    else:
        if severity_label in ("CONDICIÓN ACEPTABLE", "VIGILANCIA"):
            paragraphs.append(
                "No se identifican acciones de prioridad alta en el análisis. Se "
                "recomienda mantener la frecuencia actual de monitoreo y conservar "
                "el presente reporte como línea base de aceptación para comparaciones "
                "en próximas corridas."
            )
        else:
            paragraphs.append(
                "Aunque no se identifican acciones explícitamente clasificadas como "
                "PRIORIDAD ALTA en las narrativas, la severidad global de "
                f"{severity_label} amerita seguimiento estrecho, correlación con "
                "datos de proceso del DCS y revisión de las recomendaciones "
                "numeradas dentro de cada figura del reporte."
            )

    return "\n\n".join(paragraphs)


def _autodraft_executive_summary(meta_dict: Dict[str, Any], current_items: List[Dict[str, Any]]) -> str:
    """Wrapper público: extrae findings y compone el resumen ejecutivo."""
    findings = _extract_findings_from_items(current_items)
    return _compose_executive_summary(meta_dict, findings)


def _autodraft_single_section(
    section: str,
    meta_dict: Dict[str, Any],
    current_items: List[Dict[str, Any]],
) -> str:
    """Ciclo 17.6 — Devuelve UN solo campo regenerado del draft.

    section: 'executive_summary' | 'service_objective' |
             'service_development' | 'recommendations'
    """
    if section == "executive_summary":
        return _autodraft_executive_summary(meta_dict, current_items)
    full = _autodraft_sections_from_items(meta_dict, current_items)
    return full.get(section, "")


st.markdown('<div class="wm-section-title">Narrative sections</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="wm-meta-hint">If you leave the three fields empty, those sections are hidden in the PDF and the figures are renumbered from 1. Use "Auto-draft from figures" to generate an initial draft from the narratives of each loaded figure.</div>',
    unsafe_allow_html=True,
)

# =============================================================
# Ciclo 17.6 — Editor de secciones UNIFICADO
# =============================================================
# Cada una de las 4 secciones (Resumen ejecutivo, Objetivo,
# Desarrollo, Recomendaciones) tiene su propia fila full-width
# con su botón de Auto-redactar individual + textarea propio.
# Antes el layout era inconsistente: 2 textareas full-width y
# 2 en columnas; auto-redactar global vs auto-redactar
# resumen ejecutivo separado. Ahora todo coherente.

st.markdown(
    '<div class="wm-muted">Each section can be regenerated individually with the <b>Auto-draft</b> button. The draft is based on the report metadata (client, asset) and on findings extracted from the narratives of each loaded figure.</div>',
    unsafe_allow_html=True,
)

_section_specs = [
    {
        "key": "executive_summary",
        "label": "Executive summary",
        "hint": "First page of the PDF, after the cover",
        "height": 220,
        "placeholder": "A 4–5 paragraph synthesis: overall status, key findings, severity and critical actions. What the client reads first when opening the PDF.",
        "btn_label": "Auto-draft Executive Summary",
    },
    {
        "key": "service_objective",
        "label": "Service objective",
        "hint": "What was evaluated and under which standards?",
        "height": 150,
        "placeholder": "Evaluate the rotordynamic condition of the asset according to API 670 / API 684 / ISO 20816...",
        "btn_label": "Auto-draft Objective",
    },
    {
        "key": "service_development",
        "label": "Service development",
        "hint": "Methodology — acquisition, processing, temporal comparison, synthesis",
        "height": 220,
        "placeholder": "Stages of the service executed by the Watermelon System...",
        "btn_label": "Auto-draft Development",
    },
    {
        "key": "recommendations",
        "label": "Recommendations",
        "hint": "Prioritized actions based on the consolidated findings",
        "height": 220,
        "placeholder": "1. Investigate points in the Alarm zone. 2. Verify...",
        "btn_label": "Auto-draft Recommendations",
    },
]

for _spec in _section_specs:
    _key = _spec["key"]
    _wkey = f"report_meta_{_key}"

    # Encabezado de la sección + botón Auto-redactar a la derecha
    _h_left, _h_right = st.columns([0.78, 0.22])
    with _h_left:
        st.markdown(
            f'<div class="wm-block-title" style="margin-top:0.6rem;">{_spec["label"]}</div>'
            f'<div class="wm-block-subtitle" style="margin-bottom:0.45rem;">{_spec["hint"]}</div>',
            unsafe_allow_html=True,
        )
    with _h_right:
        if st.button(
            "🪄 Auto-draft",
            use_container_width=True,
            disabled=len(items) == 0,
            key=f"autodraft_btn_{_key}",
            help=_spec["btn_label"],
        ):
            try:
                _new_text = _autodraft_single_section(_key, meta, items)
                meta[_key] = _new_text
                st.session_state[_wkey] = _new_text
                st.session_state["report_meta"] = meta
                save_report_state(
                    items=st.session_state.get("report_items", []),
                    meta=meta,
                )
                st.success(f"«{_spec['label']}» regenerated.")
                st.rerun()
            except Exception as _exc:
                st.error(f"Could not auto-draft: {_exc}")

    # Textarea full-width
    # Ciclo 17.20 fix: NO podemos pasar value= si _wkey ya está en
    # st.session_state (Streamlit revienta con "widget created with
    # default value but also had its value set via Session State API"
    # → "Oh no" en producción). Inicializamos session_state ANTES de
    # crear el widget, sin value=.
    if _wkey not in st.session_state:
        st.session_state[_wkey] = meta.get(_key, "") or ""
    meta[_key] = st.text_area(
        label=_spec["label"],
        label_visibility="collapsed",
        key=_wkey,
        height=_spec["height"],
        placeholder=_spec["placeholder"],
    )

st.session_state["report_meta"] = meta
save_report_state(items=st.session_state.get("report_items", []), meta=st.session_state["report_meta"])

st.markdown('<div class="wm-divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="wm-section-title">Report structure</div>', unsafe_allow_html=True)

if not items:
    st.info("There are no figures in the report yet. Go to Spectrum, Waveform, Orbit or Tabular List and use the 'Send to Report' button.")
else:
    for index, item in enumerate(items, start=1):
        st.markdown('<div class="wm-card"><div class="wm-figure-card">', unsafe_allow_html=True)
        badge_class = _type_badge_class(item["type"])
        st.markdown(f'<div class="wm-block-title">Figure {index}. {item["title"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="wm-block-subtitle"><span class="wm-badge {badge_class}">{_type_badge(item["type"])}</span>{_source_line(item)}</div>', unsafe_allow_html=True)

        tcol1, tcol2, tcol3, tcol4 = st.columns([2.4, 0.8, 0.8, 0.8])
        with tcol1:
            new_title = st.text_input("Figure title", value=item["title"], key=f"report_title_{item['id']}")
            item["title"] = new_title
        with tcol2:
            st.write("")
            st.write("")
            if st.button("↑ Up", key=f"report_up_{item['id']}", use_container_width=True, disabled=index == 1):
                _move_item(item["id"], -1)
                st.rerun()
        with tcol3:
            st.write("")
            st.write("")
            if st.button("↓ Down", key=f"report_down_{item['id']}", use_container_width=True, disabled=index == len(items)):
                _move_item(item["id"], +1)
                st.rerun()
        with tcol4:
            st.write("")
            st.write("")
            if st.button("Remove", key=f"report_remove_{item['id']}", use_container_width=True):
                _remove_item(item["id"])
                st.rerun()

        if item.get("figure") is not None:
            st.plotly_chart(
                item["figure"],
                use_container_width=True,
                config={"displaylogo": False},
                key=f"report_plot_{item['id']}",
            )
        else:
            # Ciclo 17.20 lazy: leer PNG desde disco SOLO ahora, no antes.
            # Esto evita tener N image_bytes simultáneos en session_state
            # cuando hay muchas figuras (causaba OOM en Streamlit Cloud).
            _img_bytes = read_item_image_bytes(item)
            if _img_bytes is not None:
                st.image(
                    _img_bytes,
                    use_container_width=True,
                )

        new_notes = st.text_area(
            f"Technical interpretation of figure {index}",
            value=item["notes"],
            key=f"report_notes_{item['id']}",
            height=150,
            placeholder="Write here the technical analysis that will go below this figure in the final report.",
        )
        item["notes"] = new_notes

        st.markdown("</div></div>", unsafe_allow_html=True)

    _persist_items(items)

st.markdown('<div class="wm-divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="wm-section-title">Report preview</div>', unsafe_allow_html=True)

p1, p2 = st.columns([1.12, 1.88])
with p1:
    st.markdown('<div class="wm-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="wm-block-title">{meta["report_title"] or "Technical vibration report"}</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="wm-note">
            <strong>Client:</strong> {meta["client"] or "-"}<br>
            <strong>Asset:</strong> {meta["asset"] or "-"}<br>
            <strong>Unit:</strong> {meta["unit"] or "-"}<br>
            <strong>Location:</strong> {meta["location"] or "-"}<br>
            <strong>Prepared by:</strong> {meta["prepared_by"] or "-"}<br>
            <strong>Role:</strong> {meta["prepared_role"] or "-"}<br>
            <strong>Reviewed by:</strong> {meta["reviewed_by"] or "-"}<br>
            <strong>Reviewer role:</strong> {meta["reviewed_role"] or "-"}<br>
            <strong>Report date:</strong> {meta["report_date"] or TODAY_STR}<br>
            <strong>Evaluated period:</strong> {meta["period"] or "N/A"}<br>
            <strong>Consecutive:</strong> {meta["consecutive"] or "-"}<br>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="wm-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="wm-block-subtitle">Executive summary (first page of the PDF)</div>', unsafe_allow_html=True)
    st.write(meta.get("executive_summary") or "—")
    st.markdown('<div class="wm-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="wm-block-subtitle">Service objective</div>', unsafe_allow_html=True)
    st.write(meta["service_objective"] or "—")
    st.markdown('<div class="wm-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="wm-block-subtitle">Recommendations</div>', unsafe_allow_html=True)
    st.write(meta["recommendations"] or "—")
    st.markdown('<div class="wm-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="wm-block-subtitle">Service development</div>', unsafe_allow_html=True)
    st.write(meta["service_development"] or "—")
    st.markdown("</div>", unsafe_allow_html=True)

with p2:
    st.markdown('<div class="wm-card">', unsafe_allow_html=True)
    st.markdown('<div class="wm-block-title">Ordered figure summary</div>', unsafe_allow_html=True)

    if not items:
        st.markdown('<div class="wm-note">No figures added yet.</div>', unsafe_allow_html=True)
    else:
        for index, item in enumerate(items, start=1):
            summary_note = item["notes"][:240] + ("..." if len(item["notes"]) > 240 else "") if item["notes"] else ""
            badge_class = _type_badge_class(item["type"])
            st.markdown(
                f"""
                <div class="wm-preview-card">
                    <span class="wm-badge {badge_class}">{_type_badge(item["type"])}</span>
                    <span class="wm-badge wm-badge-generic">Figure {index}</span>
                    <strong>{item["title"]}</strong><br>
                    <span class="wm-muted">{_source_line(item)}</span><br><br>
                    <span class="wm-note">{summary_note}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("</div>", unsafe_allow_html=True)

st.caption(
    "Current flow: Spectrum, Waveform, Orbit and Tabular List push real content to the report via st.session_state['report_items']. "
    "Reports acts as a premium editor of the technical deliverable and a professional PDF exporter, without rebuilding the visual engines."
)


# =====================================================================
# Ciclo 17.15 P5 — Tab "Archivo histórico" de PDFs aprobados
# =====================================================================
# Lista los reportes archivados que el usuario actual puede ver según
# su role (admin todo / specialist mismo dominio / client solo shared).
# Filtros multi-criterio + descarga directa + opciones de admin
# (compartir con cliente, eliminar).
st.divider()
st.markdown(
    '<div class="wm-divider"></div><br/>'
    '<h2 style="margin-top:0;font-weight:800;color:#0f172a;">'
    '📚 Report archive</h2>',
    unsafe_allow_html=True,
)

_archive_stats = get_archive_stats()
ah_k1, ah_k2, ah_k3 = st.columns(3)
with ah_k1:
    st.metric("Total archived", _archive_stats["total"])
with ah_k2:
    st.metric("Total space", _archive_stats["total_size_human"])
with ah_k3:
    st.metric("Authors with archive", len(_archive_stats["by_owner"]))

# Filtros
fc1, fc2, fc3, fc4 = st.columns([0.27, 0.27, 0.27, 0.19])
with fc1:
    _f_client = st.text_input("Filter client", placeholder="e.g. PAREX",
                               key="wm_arch_client").strip()
with fc2:
    _f_asset = st.text_input("Filter asset", placeholder="e.g. C-200C",
                              key="wm_arch_asset").strip()
with fc3:
    _f_owner = st.text_input("Filter author", placeholder="e.g. jsuarez",
                              key="wm_arch_owner").strip()
with fc4:
    _f_year = st.selectbox(
        "Year",
        options=["(todos)"] + [str(y) for y in range(datetime.now().year, 2023, -1)],
        index=0,
        key="wm_arch_year",
    )

date_from = ""
date_to = ""
if _f_year and _f_year != "(todos)":
    date_from = f"{_f_year}-01-01"
    date_to = f"{_f_year}-12-31"

_archived = list_archived_reports(
    viewer_email=_wm_my_email,
    viewer_role=_wm_my_role,
    owner_filter=_f_owner,
    client_filter=_f_client,
    asset_filter=_f_asset,
    date_from=date_from,
    date_to=date_to,
    limit=100,
)

st.caption(
    f"Showing **{len(_archived)}** archived reports visible to your role "
    f"(`{_wm_my_role}`)"
)

if not _archived:
    st.info(
        "There are no archived reports matching your filters. "
        "When you generate a PDF and click 'Archive report', it will appear here."
    )
else:
    for sc in _archived:
        rm = sc.get("report_meta", {}) or {}
        _aid = sc.get("archive_id", "")
        _client = rm.get("client", "—")
        _asset = rm.get("asset_class") or rm.get("instance_tag") or "—"
        _sev = rm.get("executive_severity", "")
        _date = sc.get("archived_at", "")[:16]
        _owner = sc.get("owner_email", "—")
        _shared = sc.get("shared_with_client", False)
        _size = sc.get("size_human", "")

        with st.container():
            # Hotfix v3.21.1: el f-string con indentación de 16+ espacios hacía que
            # Streamlit/CommonMark tratara las líneas internas como bloques de código,
            # mostrando los <div> como texto crudo. textwrap.dedent normaliza la
            # indentación al mínimo común (0 acá) y el HTML se renderiza limpio.
            _shared_str = " · shared with client" if _shared else ""
            _sev_pill = (
                f'<span style="background:#fee2e2;color:#b91c1c;'
                f'padding:4px 10px;border-radius:999px;font-size:10px;'
                f'font-weight:800;">{_sev}</span>'
            ) if _sev else ""
            _card_html = textwrap.dedent(f"""\
            <div style="background:white;border:1px solid #e6ebf2;border-radius:12px;padding:14px 18px;margin-bottom:8px;">
              <div style="display:flex;justify-content:space-between;align-items:center;">
                <div>
                  <div style="font-weight:800;color:#0f172a;font-size:15px;">{_client} · {_asset}</div>
                  <div style="color:#475569;font-size:12px;margin-top:2px;">{_owner} · {_date} · {_size}{_shared_str}</div>
                  <div style="margin-top:4px;font-size:11px;color:#64748b;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;">ID: {_aid}</div>
                </div>
                <div>{_sev_pill}</div>
              </div>
            </div>
            """)
            st.markdown(_card_html, unsafe_allow_html=True)
            ac1, ac2, ac3 = st.columns([0.5, 0.25, 0.25])
            with ac1:
                # Descarga directa
                _pdf_b = get_archived_pdf_bytes(
                    _aid, viewer_email=_wm_my_email, viewer_role=_wm_my_role,
                )
                if _pdf_b:
                    st.download_button(
                        "Download PDF",
                        data=_pdf_b,
                        file_name=f"{_aid.split('/')[-1]}.pdf",
                        mime="application/pdf",
                        key=f"dl_{_aid}",
                        use_container_width=True,
                    )
                else:
                    st.button("Download PDF", disabled=True,
                              key=f"dl_dis_{_aid}", use_container_width=True)
            with ac2:
                # Toggle compartir con cliente (solo owner o admin)
                _can_share = (_owner.lower() == _wm_my_email
                              or _wm_my_role == "admin")
                if _can_share:
                    if _shared:
                        if st.button("Unpublish",
                                     key=f"unsh_{_aid}",
                                     use_container_width=True):
                            r = share_with_client(_aid, False,
                                                   viewer_email=_wm_my_email,
                                                   viewer_role=_wm_my_role)
                            if r.get("ok"):
                                st.rerun()
                            else:
                                st.error(r.get("error"))
                    else:
                        if st.button("Share",
                                     key=f"sh_{_aid}",
                                     use_container_width=True):
                            r = share_with_client(_aid, True,
                                                   viewer_email=_wm_my_email,
                                                   viewer_role=_wm_my_role)
                            if r.get("ok"):
                                st.rerun()
                            else:
                                st.error(r.get("error"))
            with ac3:
                # Eliminar (solo owner o admin)
                _can_del = (_owner.lower() == _wm_my_email
                            or _wm_my_role == "admin")
                if _can_del:
                    with st.popover("️  Delete", use_container_width=True):
                        st.warning(f"You are about to delete: `{_aid}`")
                        if st.button("Confirm deletion",
                                     key=f"del_{_aid}",
                                     type="primary"):
                            r = delete_archived_report(_aid,
                                                        viewer_email=_wm_my_email,
                                                        viewer_role=_wm_my_role)
                            if r.get("ok"):
                                st.success("Deleted.")
                                st.rerun()
                            else:
                                st.error(r.get("error"))
