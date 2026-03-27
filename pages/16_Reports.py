from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import plotly.graph_objects as go
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer

from core.auth import require_login, render_user_menu


st.set_page_config(page_title="Watermelon System | Reports", layout="wide")
require_login()
render_user_menu()


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = PROJECT_ROOT / "assets"
WATERMELON_LOGO = ASSETS_DIR / "watermelon_logo.png"

SIGA_LOGO_CANDIDATES = [
    ASSETS_DIR / "siga_logo.png",
    ASSETS_DIR / "SIGA_logo.png",
    ASSETS_DIR / "siga_group_logo.png",
    ASSETS_DIR / "SIGA_GROUP_logo.png",
    ASSETS_DIR / "siga.png",
]

SIGA_WATERMARK_CANDIDATES = [
    ASSETS_DIR / "siga_watermark.png",
    ASSETS_DIR / "SIGA_watermark.png",
    ASSETS_DIR / "watermark_logo_transparent_background.png",
]


st.markdown(
    """
    <style>
        .wm-page-title {
            font-size: 2rem;
            font-weight: 700;
            color: #f5f7fb;
            margin-bottom: 0.20rem;
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
            border-radius: 18px;
            padding: 1rem 1rem 0.95rem 1rem;
            box-shadow: 0 10px 28px rgba(0,0,0,0.18);
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
            font-weight: 700;
            font-size: 1.08rem;
            margin-top: 0.15rem;
            margin-bottom: 0.75rem;
        }
        .wm-block-title {
            color: #f5f7fb;
            font-size: 1.05rem;
            font-weight: 700;
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
    </style>
    """,
    unsafe_allow_html=True,
)


if "report_items" not in st.session_state:
    st.session_state["report_items"] = []

if "report_meta" not in st.session_state:
    st.session_state["report_meta"] = {
        "report_title": "REPORTE DE MONITOREO EN LINEA",
        "client": "",
        "asset": "",
        "unit": "",
        "location": "",
        "prepared_by": "",
        "reviewed_by": "",
        "period": "",
        "report_date": "",
        "consecutive": "",
        "service_development": "",
        "recommendations": "",
    }


def _normalize_report_items(raw_items: Any) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []

    if not isinstance(raw_items, list):
        return items

    for idx, item in enumerate(raw_items):
        if not isinstance(item, dict):
            continue

        fig = item.get("figure")
        image_bytes = item.get("image_bytes")
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
            "title": str(item.get("title") or f"Figure {idx+1}"),
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
    return " | ".join(parts) if parts else "Sin metadata asociada"


def _count_by_type(items: List[Dict[str, Any]], item_type: str) -> int:
    return sum(1 for item in items if item.get("type", "").lower() == item_type.lower())


def _first_existing_logo() -> Optional[Path]:
    for p in SIGA_LOGO_CANDIDATES:
        if p.exists():
            return p
    return None


def _first_existing_watermark() -> Optional[Path]:
    for p in SIGA_WATERMARK_CANDIDATES:
        if p.exists():
            return p
    return None


def _paragraph_safe(text: str) -> str:
    return (
        (text or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br/>")
    )


def _figure_png_bytes(fig: go.Figure) -> bytes:
    export_fig = go.Figure(fig)
    export_fig.update_layout(
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f8fafc",
        font=dict(color="#0f172a"),
    )
    return export_fig.to_image(format="png", width=2400, height=1250, scale=2)


def _fit_image_dimensions(img_bytes: bytes, max_width: float, max_height: float) -> Tuple[float, float]:
    reader = ImageReader(BytesIO(img_bytes))
    img_w, img_h = reader.getSize()
    scale = min(max_width / img_w, max_height / img_h)
    return img_w * scale, img_h * scale


def _build_pdf_bytes(meta: Dict[str, str], items: List[Dict[str, Any]]) -> bytes:
    buffer = BytesIO()
    page_width, page_height = A4

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="WMTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=22, leading=26, alignment=TA_LEFT, textColor=colors.HexColor("#0f172a"), spaceAfter=8))
    styles.add(ParagraphStyle(name="WMSubTitle", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=12, leading=15, alignment=TA_LEFT, textColor=colors.HexColor("#111827"), spaceAfter=4))
    styles.add(ParagraphStyle(name="WMBody", parent=styles["BodyText"], fontName="Helvetica", fontSize=10.4, leading=15, alignment=TA_JUSTIFY, textColor=colors.HexColor("#111827"), spaceAfter=8))
    styles.add(ParagraphStyle(name="WMMeta", parent=styles["Normal"], fontName="Helvetica", fontSize=10.4, leading=14, alignment=TA_LEFT, textColor=colors.HexColor("#111827"), spaceAfter=4))
    styles.add(ParagraphStyle(name="WMSection", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=14, leading=18, alignment=TA_LEFT, textColor=colors.HexColor("#0f172a"), spaceBefore=4, spaceAfter=10))
    styles.add(ParagraphStyle(name="WMFigureCaption", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=10.2, leading=13, alignment=TA_CENTER, textColor=colors.HexColor("#111827"), spaceBefore=5, spaceAfter=8))
    styles.add(ParagraphStyle(name="WMFigureText", parent=styles["BodyText"], fontName="Helvetica", fontSize=10.2, leading=14, alignment=TA_JUSTIFY, textColor=colors.HexColor("#111827"), spaceAfter=14))

    logo_siga = _first_existing_logo()
    logo_watermark = _first_existing_watermark()

    doc = SimpleDocTemplate(
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
        canvas.saveState()
        canvas.setFillColor(colors.HexColor("#ffffff"))
        canvas.rect(0, 0, page_width, page_height, fill=1, stroke=0)

        canvas.setFillColor(colors.HexColor("#38bdf8"))
        canvas.rect(page_width - 4.5 * cm, 0, 4.5 * cm, page_height, fill=1, stroke=0)

        canvas.setFillColor(colors.HexColor("#0284c7"))
        canvas.roundRect(page_width - 4.95 * cm, page_height - 6.8 * cm, 0.42 * cm, 4.8 * cm, 0.2 * cm, fill=1, stroke=0)
        canvas.roundRect(page_width - 4.95 * cm, 1.0 * cm, 0.42 * cm, 2.3 * cm, 0.2 * cm, fill=1, stroke=0)

        if logo_watermark and logo_watermark.exists():
            try:
                wm_w = 7.8 * cm
                wm_h = 7.8 * cm
                wm_x = page_width - 8.0 * cm
                wm_y = 0.15 * cm
                canvas.saveState()
                canvas.setFillAlpha(0.20)
                canvas.drawImage(str(logo_watermark), wm_x, wm_y, width=wm_w, height=wm_h, mask='auto', preserveAspectRatio=True, anchor='c')
                canvas.restoreState()
            except Exception:
                pass

        canvas.setFont("Helvetica-Bold", 10.5)
        canvas.setFillColor(colors.HexColor("#111827"))
        canvas.drawRightString(page_width - 1.15 * cm, page_height - 1.0 * cm, f"Página {doc.page}")

        footer = "INFORME VALIDO UNICAMENTE PARA LAS CONDICIONES PRESENTES DURANTE EL SERVICIO. NO PODRA SER COPIADO PARCIAL O TOTALMENTE SIN PREVIA AUTORIZACION."
        canvas.setFillColor(colors.HexColor("#334155"))
        canvas.setFont("Helvetica-Bold", 5.8)
        canvas.drawCentredString((page_width - 4.5 * cm) / 2, 0.52 * cm, footer)
        canvas.restoreState()

    def _draw_internal_page(canvas, doc):
        canvas.saveState()
        canvas.setFillColor(colors.HexColor("#ffffff"))
        canvas.rect(0, 0, page_width, page_height, fill=1, stroke=0)

        canvas.setFont("Helvetica-Bold", 11)
        canvas.setFillColor(colors.HexColor("#111827"))
        canvas.drawRightString(page_width - 1.2 * cm, page_height - 1.0 * cm, f"Página {doc.page}")

        internal_left = 2.1 * cm
        internal_right = 2.1 * cm
        internal_width_end = page_width - internal_right

        canvas.setStrokeColor(colors.HexColor("#0ea5e9"))
        canvas.setLineWidth(1.1)
        canvas.line(internal_left, page_height - 1.35 * cm, internal_width_end, page_height - 1.35 * cm)

        canvas.setFillColor(colors.HexColor("#0f172a"))
        canvas.setFont("Helvetica-Bold", 8.2)
        canvas.drawString(internal_left, page_height - 1.0 * cm, "Machinery Diagnostics Engineering")

        footer = "INFORME VALIDO UNICAMENTE PARA LAS CONDICIONES PRESENTES DURANTE EL SERVICIO. NO PODRA SER COPIADO PARCIAL O TOTALMENTE SIN PREVIA AUTORIZACION."
        canvas.setStrokeColor(colors.HexColor("#0ea5e9"))
        canvas.setLineWidth(1.0)
        canvas.line(internal_left, 0.95 * cm, internal_width_end, 0.95 * cm)

        canvas.setFillColor(colors.HexColor("#111827"))
        canvas.setFont("Helvetica", 6.4)
        canvas.drawCentredString((internal_left + internal_width_end) / 2, 0.55 * cm, footer)
        canvas.restoreState()

    story: List[Any] = []

    header_code = meta.get("consecutive") or "SIGA-FMT-178 | Version 3 | Fecha 19-06-2024"
    story.append(Paragraph(_paragraph_safe(header_code), ParagraphStyle(name="WMCoverHead", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=9.3, leading=12, textColor=colors.HexColor("#0ea5e9"), alignment=TA_LEFT, spaceAfter=10)))

    if WATERMELON_LOGO.exists():
        story.append(Image(str(WATERMELON_LOGO), width=4.2 * cm, height=2.0 * cm))
        story.append(Spacer(1, 0.35 * cm))

    story.append(Spacer(1, 0.18 * cm))
    story.append(Paragraph("Machinery Diagnostics Engineering", styles["WMSubTitle"]))
    story.append(Spacer(1, 0.55 * cm))
    story.append(Paragraph(_paragraph_safe(meta.get("report_title") or "REPORTE TECNICO"), styles["WMTitle"]))
    story.append(Paragraph("Watermelon System", ParagraphStyle(name="WMBrandSub", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=15.5, leading=19, textColor=colors.HexColor("#111827"), spaceAfter=18)))

    cover_lines = [
        meta.get("asset") or "-",
        meta.get("unit") or "-",
        meta.get("location") or "-",
        meta.get("client") or "-",
    ]
    for line in cover_lines:
        story.append(Paragraph(_paragraph_safe(line), ParagraphStyle(name=f"WMCoverLine_{len(story)}", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=12.8, leading=16, textColor=colors.HexColor("#111827"), spaceAfter=3)))

    story.append(Spacer(1, 0.85 * cm))
    story.append(Paragraph(f"<b>Preparado por:</b><br/>{_paragraph_safe(meta.get('prepared_by') or '-')}", styles["WMMeta"]))
    story.append(Spacer(1, 0.50 * cm))
    story.append(Paragraph(f"<b>Revisado por:</b><br/>{_paragraph_safe(meta.get('reviewed_by') or '-')}", styles["WMMeta"]))
    story.append(Spacer(1, 0.85 * cm))
    story.append(Paragraph(f"<b>Periodo evaluado:</b> {_paragraph_safe(meta.get('period') or '-')}", styles["WMMeta"]))
    story.append(Paragraph(f"<b>Fecha de reporte:</b> {_paragraph_safe(meta.get('report_date') or '-')}", styles["WMMeta"]))
    story.append(Paragraph(f"<b>Consecutivo:</b> {_paragraph_safe(meta.get('consecutive') or '-')}", styles["WMMeta"]))
    story.append(PageBreak())

    story.append(Paragraph("1. RECOMENDACIONES", styles["WMSection"]))
    story.append(Paragraph(_paragraph_safe(meta.get("recommendations") or "Sin recomendaciones registradas."), styles["WMBody"]))
    story.append(Spacer(1, 0.20 * cm))
    story.append(Paragraph("2. DESARROLLO DEL SERVICIO", styles["WMSection"]))
    story.append(Paragraph(_paragraph_safe(meta.get("service_development") or "Sin desarrollo del servicio registrado."), styles["WMBody"]))
    story.append(Spacer(1, 0.35 * cm))

    usable_width = A4[0] - doc.leftMargin - doc.rightMargin
    max_img_width = usable_width - 1.0 * cm
    max_img_height = 8.6 * cm

    for idx, item in enumerate(items, start=1):
        if item.get("figure") is not None:
            png_bytes = _figure_png_bytes(item["figure"])
        elif item.get("image_bytes") is not None:
            png_bytes = item["image_bytes"]
        else:
            continue

        img_w, img_h = _fit_image_dimensions(png_bytes, max_img_width, max_img_height)
        img = Image(BytesIO(png_bytes), width=img_w, height=img_h)
        img.hAlign = "CENTER"
        story.append(Spacer(1, 0.15 * cm))
        story.append(img)

        caption = f"Figura {idx}. {item.get('title') or f'Figura {idx}'}"
        story.append(Paragraph(_paragraph_safe(caption), styles["WMFigureCaption"]))

        notes = item.get("notes") or "Sin interpretación técnica todavía."
        story.append(Paragraph(_paragraph_safe(notes), styles["WMFigureText"]))
        story.append(Spacer(1, 0.20 * cm))

    doc.build(story, onFirstPage=_draw_cover_page, onLaterPages=_draw_internal_page)
    return buffer.getvalue()


items = _get_items()
meta = st.session_state["report_meta"]


st.markdown('<div class="wm-page-title">Reports</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="wm-page-subtitle">Editor de entregables técnicos premium. Este módulo consume figuras reales enviadas desde Spectrum y otros módulos, y exporta PDF técnico profesional.</div>',
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f'<div class="wm-kpi"><div class="wm-kpi-label">Figures in Report</div><div class="wm-kpi-value">{len(items):,}</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="wm-kpi"><div class="wm-kpi-label">Spectrum Blocks</div><div class="wm-kpi-value">{_count_by_type(items, "spectrum"):,}</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="wm-kpi"><div class="wm-kpi-label">Prepared By</div><div class="wm-kpi-value">{meta["prepared_by"] or "-"}</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="wm-kpi"><div class="wm-kpi-label">Consecutive</div><div class="wm-kpi-value">{meta["consecutive"] or "-"}</div></div>', unsafe_allow_html=True)

st.markdown('<div class="wm-divider"></div>', unsafe_allow_html=True)

st.markdown('<div class="wm-section-title">Report Actions</div>', unsafe_allow_html=True)

ga1, ga2, ga3, ga4 = st.columns([1.2, 1.2, 1.2, 3.4])
with ga1:
    if st.button("Refresh Items", use_container_width=True):
        _persist_items(_get_items())
        st.rerun()
with ga2:
    clear_disabled = len(items) == 0
    if st.button("Clear Report", use_container_width=True, disabled=clear_disabled):
        _clear_all_items()
        st.rerun()

pdf_ready = len(items) > 0
pdf_error = None
pdf_bytes: Optional[bytes] = None
if pdf_ready:
    try:
        pdf_bytes = _build_pdf_bytes(meta, items)
    except Exception as e:
        pdf_error = str(e)

with ga3:
    if pdf_bytes is not None:
        st.download_button(
            "Export PDF",
            data=pdf_bytes,
            file_name=(meta.get("consecutive") or "watermelon_report").replace(" ", "_") + ".pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    else:
        st.button("Export PDF", use_container_width=True, disabled=True)

if pdf_error:
    st.warning(f"PDF export error: {pdf_error}")

st.markdown('<div class="wm-divider"></div>', unsafe_allow_html=True)

st.markdown('<div class="wm-section-title">Report Metadata</div>', unsafe_allow_html=True)

m1, m2, m3 = st.columns(3)
with m1:
    meta["report_title"] = st.text_input("Report Title", value=meta["report_title"])
with m2:
    meta["client"] = st.text_input("Client", value=meta["client"])
with m3:
    meta["asset"] = st.text_input("Asset / Machine", value=meta["asset"])

m4, m5, m6 = st.columns(3)
with m4:
    meta["unit"] = st.text_input("Unit", value=meta["unit"])
with m5:
    meta["location"] = st.text_input("Location", value=meta["location"])
with m6:
    meta["consecutive"] = st.text_input("Consecutive", value=meta["consecutive"])

m7, m8, m9 = st.columns(3)
with m7:
    meta["prepared_by"] = st.text_input("Prepared By", value=meta["prepared_by"])
with m8:
    meta["reviewed_by"] = st.text_input("Reviewed By", value=meta["reviewed_by"])
with m9:
    meta["report_date"] = st.text_input("Report Date", value=meta["report_date"])

m10, m11 = st.columns(2)
with m10:
    meta["period"] = st.text_input("Evaluation Period", value=meta["period"])
with m11:
    st.write("")

t1, t2 = st.columns(2)
with t1:
    meta["service_development"] = st.text_area("Desarrollo del servicio", value=meta["service_development"], height=180)
with t2:
    meta["recommendations"] = st.text_area("Recomendaciones", value=meta["recommendations"], height=180)

st.session_state["report_meta"] = meta

st.markdown('<div class="wm-divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="wm-section-title">Report Structure</div>', unsafe_allow_html=True)

if not items:
    st.info("Todavía no hay figuras en el reporte. Entra a Spectrum, Waveforms, Orbit o Tabular List y usa el botón 'Enviar a Reporte'.")
else:
    for index, item in enumerate(items, start=1):
        st.markdown('<div class="wm-card">', unsafe_allow_html=True)
        badge_class = _type_badge_class(item["type"])
        st.markdown(f'<div class="wm-block-title">Figura {index}. {item["title"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="wm-block-subtitle"><span class="wm-badge {badge_class}">{_type_badge(item["type"])}</span>{_source_line(item)}</div>', unsafe_allow_html=True)

        tcol1, tcol2, tcol3, tcol4 = st.columns([2.4, 0.8, 0.8, 0.8])
        with tcol1:
            new_title = st.text_input("Figure Title", value=item["title"], key=f"report_title_{item['id']}")
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
        elif item.get("image_bytes") is not None:
            st.image(
                item["image_bytes"],
                use_container_width=True,
            )

        new_notes = st.text_area(
            f"Technical interpretation for Figure {index}",
            value=item["notes"],
            key=f"report_notes_{item['id']}",
            height=140,
            placeholder="Escribe aquí el análisis técnico que irá debajo de esta figura en el reporte final.",
        )
        item["notes"] = new_notes

        st.markdown("</div>", unsafe_allow_html=True)

    _persist_items(items)

st.markdown('<div class="wm-divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="wm-section-title">Report Preview</div>', unsafe_allow_html=True)

p1, p2 = st.columns([1.15, 1.85])
with p1:
    st.markdown('<div class="wm-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="wm-block-title">{meta["report_title"] or "Technical Vibration Report"}</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="wm-note">
            <strong>Client:</strong> {meta["client"] or "-"}<br>
            <strong>Asset:</strong> {meta["asset"] or "-"}<br>
            <strong>Unit:</strong> {meta["unit"] or "-"}<br>
            <strong>Location:</strong> {meta["location"] or "-"}<br>
            <strong>Prepared By:</strong> {meta["prepared_by"] or "-"}<br>
            <strong>Reviewed By:</strong> {meta["reviewed_by"] or "-"}<br>
            <strong>Period:</strong> {meta["period"] or "-"}<br>
            <strong>Report Date:</strong> {meta["report_date"] or "-"}<br>
            <strong>Consecutive:</strong> {meta["consecutive"] or "-"}<br>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="wm-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="wm-block-subtitle">Recomendaciones</div>', unsafe_allow_html=True)
    st.write(meta["recommendations"] or "—")
    st.markdown('<div class="wm-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="wm-block-subtitle">Desarrollo del servicio</div>', unsafe_allow_html=True)
    st.write(meta["service_development"] or "—")
    st.markdown("</div>", unsafe_allow_html=True)

with p2:
    st.markdown('<div class="wm-card">', unsafe_allow_html=True)
    st.markdown('<div class="wm-block-title">Ordered Figure Summary</div>', unsafe_allow_html=True)

    if not items:
        st.markdown('<div class="wm-note">No hay figuras agregadas todavía.</div>', unsafe_allow_html=True)
    else:
        for index, item in enumerate(items, start=1):
            summary_note = item["notes"][:220] + ("..." if len(item["notes"]) > 220 else "") if item["notes"] else "Sin interpretación técnica todavía."
            badge_class = _type_badge_class(item["type"])
            st.markdown(
                f"""
                <div class="wm-note">
                    <span class="wm-badge {badge_class}">{_type_badge(item["type"])}</span>
                    <span class="wm-badge wm-badge-generic">Figura {index}</span>
                    <strong>{item["title"]}</strong><br>
                    {_source_line(item)}<br>
                    {summary_note}
                </div>
                <div class="wm-divider"></div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("</div>", unsafe_allow_html=True)

st.caption(
    "Flujo actual: Spectrum, Waveforms, Orbit y Tabular List empujan contenido real al reporte mediante st.session_state['report_items']. "
    "Reports actúa como editor premium de entregable técnico y exportador PDF profesional, sin reconstruir motores visuales."
)
