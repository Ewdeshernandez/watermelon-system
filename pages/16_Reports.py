from __future__ import annotations

from datetime import date
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

from core.auth import require_login, render_user_menu
from core.report_state import (
    clear_report_state,
    delete_named_report_draft,
    list_report_drafts,
    load_named_report_draft,
    load_report_state,
    save_named_report_draft,
    save_report_state,
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

if "report_state_loaded" not in st.session_state:
    persisted_state = load_report_state()

    persisted_items = persisted_state.get("items", [])
    persisted_meta = persisted_state.get("meta", {})

    st.session_state["report_items"] = persisted_items if isinstance(persisted_items, list) else []
    merged_meta = dict(DEFAULT_REPORT_META)
    if isinstance(persisted_meta, dict):
        merged_meta.update(persisted_meta)
    if not merged_meta.get("report_date"):
        merged_meta["report_date"] = TODAY_STR
    st.session_state["report_meta"] = merged_meta
    st.session_state["report_state_loaded"] = True

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

    # Mapa instance.field → meta key. Sólo se rellena si meta[key] está
    # vacío (back-fill no destructivo).
    mappings = {
        "client": (inst.client or "").strip(),
        "asset_class": (inst.asset_class or "").strip(),
        "asset_model": (inst.driver_model or inst.driven_model or "").strip(),
        "location": (inst.site or inst.location or "").strip(),
        "asset": (inst.tag or "").strip(),
        "unit": (inst.tag or "").strip(),
    }
    for k, v in mappings.items():
        if v and not (meta.get(k) or "").strip():
            meta[k] = v

    # Train description: si el meta no la tiene, la componemos de
    # los campos driver/driven de la instance.
    if not (meta.get("train_description") or "").strip():
        composed = compose_train_description(inst)
        if composed:
            meta["train_description"] = composed

    # Schematic: doc_id del esquemático en el Vault de la instance.
    # El render del PDF lo resuelve a bytes vía get_instance_document_bytes.
    if inst.schematic_png and not (meta.get("schematic_doc_id") or "").strip():
        meta["schematic_doc_id"] = inst.schematic_png
        meta["schematic_instance_id"] = inst.instance_id


_autofill_report_meta_from_active_instance()


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
    return " | ".join(parts) if parts else "Sin metadata asociada"


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
    return export_fig.to_image(format="png", width=2400, height=1250, scale=2)


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
    """
    flowables: List[Any] = []

    if notes_main.strip():
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
    executive_text = (meta.get("executive_summary") or "").strip()
    if executive_text:
        story.append(Paragraph("RESUMEN EJECUTIVO", styles["WMTOC1"]))

        # Cinta de severidad: una franja con estado global y color (si se puede
        # detectar desde el primer párrafo del resumen). Si no, omitida.
        severity_label = ""
        for known in ("CRÍTICA", "ACCIÓN REQUERIDA", "ATENCIÓN", "VIGILANCIA", "CONDICIÓN ACEPTABLE"):
            if known in executive_text:
                severity_label = known
                break
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

        # Ciclo 14a — Esquemático del tren acoplado (debajo del badge de
        # severidad, sobre el cuerpo del Resumen Ejecutivo). Aparece
        # automáticamente cuando hay una Asset Instance activa con
        # schematic_png cargado en su Vault. Si no hay, omite limpio.
        sch_doc_id = (meta.get("schematic_doc_id") or "").strip()
        sch_inst_id = (meta.get("schematic_instance_id") or "").strip()
        if sch_doc_id and sch_inst_id:
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
                    sch_img = Image(BytesIO(sch_bytes), width=fitted_w, height=fitted_h)
                    sch_img.hAlign = "CENTER"
                    story.append(Spacer(1, 0.10 * cm))
                    story.append(sch_img)
                    sch_caption_style = ParagraphStyle(
                        name="WMSchematicCaption",
                        parent=styles["WMMeta"],
                        fontName=PDF_FONT_REGULAR,
                        fontSize=8.8,
                        leading=11,
                        alignment=TA_CENTER,
                        textColor=colors.HexColor("#475569"),
                        spaceBefore=2,
                        spaceAfter=10,
                    )
                    train_lbl = (meta.get("train_description") or "").strip()
                    if train_lbl:
                        story.append(Paragraph(
                            f"Esquemático del tren · {train_lbl}",
                            sch_caption_style,
                        ))
                    else:
                        story.append(Paragraph(
                            "Esquemático del tren acoplado",
                            sch_caption_style,
                        ))
            except Exception:
                # Si falla cualquier paso (instancia borrada, doc roto,
                # imagen corrupta) silenciamos para no bloquear el reporte.
                pass

        story.append(Paragraph(_paragraph_safe(executive_text), styles["WMBody"]))
        story.append(Spacer(1, 0.30 * cm))
        story.append(PageBreak())

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
        if item.get("image_bytes") is not None:
            png_bytes = item["image_bytes"]
        else:
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
            img_w, img_h = _fit_image_dimensions(png_bytes, max_img_width, max_img_height)
            img = Image(BytesIO(png_bytes), width=img_w, height=img_h)
            img.hAlign = "CENTER"

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


st.markdown('<div class="wm-page-title">Reports</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="wm-page-subtitle">Editor premium de entregables técnicos. Este módulo organiza figuras reales enviadas desde Spectrum, Waveform, Orbit y Tabular List, y exporta un PDF corporativo listo para cliente.</div>',
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f'<div class="wm-kpi"><div class="wm-kpi-label">Figuras en reporte</div><div class="wm-kpi-value">{len(items):,}</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="wm-kpi"><div class="wm-kpi-label">Bloques Spectrum</div><div class="wm-kpi-value">{_count_by_type(items, "spectrum"):,}</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="wm-kpi"><div class="wm-kpi-label">Preparado por</div><div class="wm-kpi-value">{meta["prepared_by"] or "-"}</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="wm-kpi"><div class="wm-kpi-label">Consecutivo</div><div class="wm-kpi-value">{meta["consecutive"] or "-"}</div></div>', unsafe_allow_html=True)

st.markdown('<div class="wm-divider"></div>', unsafe_allow_html=True)

st.markdown('<div class="wm-section-title">Acciones del reporte</div>', unsafe_allow_html=True)

ga1, ga2, ga3, ga4 = st.columns([1.2, 1.2, 1.2, 3.4])
with ga1:
    if st.button("Actualizar figuras", use_container_width=True):
        _persist_items(_get_items())
        st.rerun()
with ga2:
    clear_disabled = len(items) == 0
    if st.button("Vaciar reporte", use_container_width=True, disabled=clear_disabled):
        _clear_all_items()
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

with st.expander("📋 Auto-fill desde activo monitoreado", expanded=True):
    if _active_inst is None:
        st.warning(
            "No hay activo monitoreado activo. Anda a Machinery Library "
            "y activa una máquina para que sus datos se auto-llenen acá."
        )
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Activo:** {_active_inst.tag or _active_inst.instance_id}")
            st.caption(f"Cliente · {meta.get('client') or '—'}")
            st.caption(f"Sitio · {meta.get('location') or '—'}")
            st.caption(f"Clase · {meta.get('asset_class') or '—'}")
            st.caption(f"Modelo · {meta.get('asset_model') or '—'}")
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
                            f"✓ Esquemático listo para Resumen Ejecutivo "
                            f"({len(test_bytes) // 1024} KB)"
                        )
                    else:
                        st.error(
                            f"✗ schematic_doc_id presente ({sch_doc}) pero "
                            f"no se pudo leer el archivo del Vault. "
                            f"¿El documento fue borrado?"
                        )
                except Exception as e:
                    st.error(f"✗ Error leyendo esquemático: {e}")
            elif inst_sch:
                # La instancia tiene schematic_png pero el meta no se rellenó
                st.warning(
                    f"⚠️ El activo tiene schematic_png={inst_sch[:20]}... "
                    f"pero el meta del reporte no lo tomó. Click en "
                    f"'Reset auto-fill' abajo para forzar recarga."
                )
                if st.button("Reset auto-fill desde activo", key="reset_autofill"):
                    meta["schematic_doc_id"] = ""
                    meta["schematic_instance_id"] = ""
                    _autofill_report_meta_from_active_instance()
                    st.rerun()
            else:
                st.error(
                    "✗ El activo NO tiene esquemático principal vinculado. "
                    "Andá a Machinery Library → tu máquina activa → "
                    "Editar metadata → tab Esquemático → seleccioná tu PNG/JPG → guardar."
                )

with ga3:
    if st.button("Preparar PDF", use_container_width=True, disabled=not pdf_ready):
        try:
            pdf_bytes = _build_pdf_bytes(meta, items)
            st.session_state["report_pdf_bytes"] = pdf_bytes
            st.session_state["report_pdf_error"] = None
        except Exception as e:
            st.session_state["report_pdf_bytes"] = None
            st.session_state["report_pdf_error"] = str(e)

pdf_bytes = st.session_state.get("report_pdf_bytes")
pdf_error = st.session_state.get("report_pdf_error")

if pdf_bytes is not None:
    st.download_button(
        "Descargar PDF",
        data=pdf_bytes,
        file_name=(meta.get("consecutive") or "watermelon_report").replace(" ", "_") + ".pdf",
        mime="application/pdf",
        use_container_width=True,
    )

if pdf_error:
    st.warning(f"PDF export error: {pdf_error}")

st.markdown('<div class="wm-divider"></div>', unsafe_allow_html=True)

drafts = list_report_drafts()

st.markdown('<div class="wm-section-title">Borradores del reporte</div>', unsafe_allow_html=True)

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
        "Nombre del borrador",
        key="report_draft_name_value",
    )
with d2:
    st.write("")
    st.write("")
    if st.button("Guardar borrador", use_container_width=True):
        saved_name = save_named_report_draft(
            draft_name=draft_name,
            items=st.session_state.get("report_items", []),
            meta=st.session_state.get("report_meta", {}),
        )
        save_report_state(items=st.session_state.get("report_items", []), meta=st.session_state.get("report_meta", {}))
        st.success(f"Borrador guardado: {saved_name}")
        st.rerun()
with d3:
    st.write("")
    st.write("")
    if st.button("Duplicar borrador", use_container_width=True):
        base_name = (draft_name or "reporte_actual").strip()
        duplicate_name = f"{base_name}_copia"
        saved_name = save_named_report_draft(
            draft_name=duplicate_name,
            items=st.session_state.get("report_items", []),
            meta=st.session_state.get("report_meta", {}),
        )
        st.success(f"Borrador duplicado: {saved_name}")
        st.rerun()
with d4:
    st.write("")
    st.write("")
    if st.button("Nuevo reporte", use_container_width=True):
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
        "Borradores existentes",
        options=drafts if drafts else ["—"],
        index=0,
        key="report_selected_draft",
    )
with d6:
    st.write("")
    st.write("")
    if st.button("Cargar borrador", use_container_width=True, disabled=not drafts or selected_draft == "—"):
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
        st.success(f"Borrador cargado: {selected_draft}")
        st.rerun()
with d7:
    st.write("")
    st.write("")
    if st.button("Eliminar borrador", use_container_width=True, disabled=not drafts or selected_draft == "—"):
        delete_named_report_draft(selected_draft)
        if st.session_state.get("report_draft_name_value") == selected_draft:
            pass
        st.success(f"Borrador eliminado: {selected_draft}")
        st.rerun()

st.markdown('<div class="wm-divider"></div>', unsafe_allow_html=True)

st.markdown('<div class="wm-section-title">Metadatos del reporte</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="wm-meta-hint">La fecha del reporte se carga automáticamente con la fecha actual. El periodo evaluado es opcional y vale la pena cuando el servicio corresponde a una campaña, ventana operativa o rango de fechas.</div>',
    unsafe_allow_html=True,
)

m1, m2, m3 = st.columns(3)
with m1:
    meta["report_title"] = st.text_input("Título del reporte", key="report_meta_report_title", value=meta["report_title"])
with m2:
    meta["client"] = st.text_input("Cliente", key="report_meta_client", value=meta["client"])
with m3:
    meta["asset"] = st.text_input("Activo / máquina", key="report_meta_asset", value=meta["asset"])

m4, m5, m6 = st.columns(3)
with m4:
    meta["unit"] = st.text_input("Unidad", key="report_meta_unit", value=meta["unit"])
with m5:
    meta["location"] = st.text_input("Ubicación", key="report_meta_location", value=meta["location"])
with m6:
    meta["consecutive"] = st.text_input("Consecutivo", key="report_meta_consecutive", value=meta["consecutive"])

# Ciclo 10A — bloque grande SIGA-style del activo en la portada
m_sa1, m_sa2 = st.columns(2)
with m_sa1:
    meta["asset_class"] = st.text_input(
        "Clase de activo (portada)",
        key="report_meta_asset_class",
        value=meta.get("asset_class", ""),
        placeholder="TURBOGENERADOR, MOTOR-BOMBA, COMPRESOR…",
        help=(
            "Clase técnica del activo en mayúsculas — aparece en grande "
            "en la portada del PDF, junto a la unidad. Estilo reporte SIGA."
        ),
    )
with m_sa2:
    meta["asset_model"] = st.text_input(
        "Modelo / configuración (portada)",
        key="report_meta_asset_model",
        value=meta.get("asset_model", ""),
        placeholder="LM5000, SGT-300, Brush 54 MW…",
        help="Modelo/configuración del activo. Se imprime en grande debajo de la clase.",
    )

# Composición del tren acoplado — la mayoría de máquinas reales son trenes
# acoplados (turbina + generador, motor + bomba, motor + compresor, etc.)
meta["train_description"] = st.text_area(
    "Composición del tren acoplado (opcional)",
    key="report_meta_train_description",
    value=meta.get("train_description", ""),
    height=80,
    placeholder=(
        "Ejemplo: una turbina aeroderivada GE LM6000 acoplada por reductor "
        "doble-helicoidal a un generador eléctrico Brush de 54 MW a 3600 rpm. "
        "Este texto reemplaza al campo 'Activo' en la portada y narrativas "
        "cuando se completa, permitiendo describir trenes mecánicos completos."
    ),
)

m7, m8 = st.columns(2)
with m7:
    meta["prepared_by"] = st.text_input("Preparado por", key="report_meta_prepared_by", value=meta["prepared_by"])
    meta["prepared_role"] = st.text_input("Cargo de quien prepara", key="report_meta_prepared_role", value=meta["prepared_role"])
    meta["prepared_city"] = st.text_input(
        "Ciudad / país de quien prepara",
        key="report_meta_prepared_city",
        value=meta.get("prepared_city", ""),
        placeholder="Cajicá, Cundinamarca · Colombia",
        help="Ciudad y país que aparecen debajo del nombre/cargo en la portada.",
    )
with m8:
    meta["reviewed_by"] = st.text_input("Revisado por", key="report_meta_reviewed_by", value=meta["reviewed_by"])
    meta["reviewed_role"] = st.text_input("Cargo de quien revisa", key="report_meta_reviewed_role", value=meta["reviewed_role"])
    meta["reviewed_city"] = st.text_input(
        "Ciudad / país de quien revisa",
        key="report_meta_reviewed_city",
        value=meta.get("reviewed_city", ""),
        placeholder="Cajicá, Cundinamarca · Colombia",
    )

st.markdown(
    '<div class="wm-signature-help">Estos cargos también se mostrarán en el bloque final de aprobación del PDF.</div>',
    unsafe_allow_html=True,
)

m9, m10 = st.columns(2)
with m9:
    meta["report_date"] = st.text_input("Fecha del reporte", key="report_meta_report_date", value=meta["report_date"] or TODAY_STR)
with m10:
    meta["period"] = st.text_input("Periodo evaluado (opcional)", key="report_meta_period", value=meta["period"], placeholder="Ejemplo: 2026-04-01 a 2026-04-07")

st.markdown(
    '<div class="wm-highlight-box"><b>Sugerencia editorial</b><br>Si el servicio corresponde a una visita puntual, puedes dejar vacío el periodo evaluado y usar solo la fecha del reporte. Si cubre tendencia, campaña o ventana de operación, sí conviene llenarlo.</div>',
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
        "high_priority_actions": [],
        "n_figures": len(current_items),
    }

    for it in current_items:
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
    composition_clause = ", ".join(components) if components else f"{n_fig} figuras de análisis"

    paragraphs.append(
        f"El presente reporte sintetiza la condición rotodinámica del "
        f"{asset_clause} a partir de {n_fig} figuras de análisis adquiridas "
        f"mediante el sistema de monitoreo en línea y remoto Watermelon System, "
        f"incluyendo {composition_clause}. La evaluación combinada de los "
        f"hallazgos según los criterios técnicos aplicables (API 670 / API 684 "
        f"para análisis rotodinámico, ISO 20816 para severidad de vibración) "
        f"arroja una clasificación global de {severity_label}."
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


st.markdown('<div class="wm-section-title">Secciones narrativas</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="wm-meta-hint">Si dejas vacíos los tres campos, esas secciones se ocultan en el PDF y las figuras pasan a numerarse desde 1. Usa "Auto-redactar desde figuras" para generar un draft inicial a partir de las narrativas de cada figura cargada.</div>',
    unsafe_allow_html=True,
)

ad1, ad2, ad3 = st.columns([1.4, 1.6, 3.0])
with ad1:
    if st.button("Auto-redactar secciones 1/2/3", use_container_width=True, disabled=len(items) == 0):
        draft = _autodraft_sections_from_items(meta, items)
        for k, v in draft.items():
            meta[k] = v
            st.session_state[f"report_meta_{k}"] = v
        st.session_state["report_meta"] = meta
        save_report_state(items=st.session_state.get("report_items", []), meta=meta)
        st.success("Secciones 1/2/3 redactadas como draft. Ajusta los matices que quieras.")
        st.rerun()
with ad2:
    if st.button("Auto-redactar resumen ejecutivo", use_container_width=True, disabled=len(items) == 0):
        exec_draft = _autodraft_executive_summary(meta, items)
        meta["executive_summary"] = exec_draft
        st.session_state["report_meta_executive_summary"] = exec_draft
        st.session_state["report_meta"] = meta
        save_report_state(items=st.session_state.get("report_items", []), meta=meta)
        st.success("Resumen ejecutivo generado. Aparece como página inicial del PDF, después de la portada.")
        st.rerun()
with ad3:
    st.markdown(
        '<div class="wm-muted">El draft se basa en metadatos del reporte (cliente, activo) y en hallazgos extraídos de las narrativas de cada figura. El resumen ejecutivo sintetiza estado global, hallazgos clave, severidad y acciones críticas.</div>',
        unsafe_allow_html=True,
    )

exec_col = st.columns(1)[0]
with exec_col:
    meta["executive_summary"] = st.text_area(
        "Resumen ejecutivo (página inicial del PDF, después de portada)",
        key="report_meta_executive_summary",
        value=meta.get("executive_summary", ""),
        height=220,
        placeholder="Síntesis de 4–5 párrafos: estado global, hallazgos clave, severidad y acciones críticas. Lo que el cliente lee primero al abrir el PDF.",
    )

t0 = st.columns(1)[0]
with t0:
    meta["service_objective"] = st.text_area("Objetivo del servicio", key="report_meta_service_objective", value=meta["service_objective"], height=120)

t1, t2 = st.columns(2)
with t1:
    meta["service_development"] = st.text_area("Desarrollo del servicio", key="report_meta_service_development", value=meta["service_development"], height=190)
with t2:
    meta["recommendations"] = st.text_area("Recomendaciones", key="report_meta_recommendations", value=meta["recommendations"], height=190)

st.session_state["report_meta"] = meta
save_report_state(items=st.session_state.get("report_items", []), meta=st.session_state["report_meta"])

st.markdown('<div class="wm-divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="wm-section-title">Estructura del reporte</div>', unsafe_allow_html=True)

if not items:
    st.info("Todavía no hay figuras en el reporte. Entra a Spectrum, Waveform, Orbit o Tabular List y usa el botón 'Enviar a Reporte'.")
else:
    for index, item in enumerate(items, start=1):
        st.markdown('<div class="wm-card"><div class="wm-figure-card">', unsafe_allow_html=True)
        badge_class = _type_badge_class(item["type"])
        st.markdown(f'<div class="wm-block-title">Figura {index}. {item["title"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="wm-block-subtitle"><span class="wm-badge {badge_class}">{_type_badge(item["type"])}</span>{_source_line(item)}</div>', unsafe_allow_html=True)

        tcol1, tcol2, tcol3, tcol4 = st.columns([2.4, 0.8, 0.8, 0.8])
        with tcol1:
            new_title = st.text_input("Título de la figura", value=item["title"], key=f"report_title_{item['id']}")
            item["title"] = new_title
        with tcol2:
            st.write("")
            st.write("")
            if st.button("↑ Subir", key=f"report_up_{item['id']}", use_container_width=True, disabled=index == 1):
                _move_item(item["id"], -1)
                st.rerun()
        with tcol3:
            st.write("")
            st.write("")
            if st.button("↓ Bajar", key=f"report_down_{item['id']}", use_container_width=True, disabled=index == len(items)):
                _move_item(item["id"], +1)
                st.rerun()
        with tcol4:
            st.write("")
            st.write("")
            if st.button("Eliminar", key=f"report_remove_{item['id']}", use_container_width=True):
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
            f"Interpretación técnica de la figura {index}",
            value=item["notes"],
            key=f"report_notes_{item['id']}",
            height=150,
            placeholder="Escribe aquí el análisis técnico que irá debajo de esta figura en el reporte final.",
        )
        item["notes"] = new_notes

        st.markdown("</div></div>", unsafe_allow_html=True)

    _persist_items(items)

st.markdown('<div class="wm-divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="wm-section-title">Vista previa del reporte</div>', unsafe_allow_html=True)

p1, p2 = st.columns([1.12, 1.88])
with p1:
    st.markdown('<div class="wm-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="wm-block-title">{meta["report_title"] or "Reporte técnico de vibraciones"}</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="wm-note">
            <strong>Cliente:</strong> {meta["client"] or "-"}<br>
            <strong>Activo:</strong> {meta["asset"] or "-"}<br>
            <strong>Unidad:</strong> {meta["unit"] or "-"}<br>
            <strong>Ubicación:</strong> {meta["location"] or "-"}<br>
            <strong>Preparado por:</strong> {meta["prepared_by"] or "-"}<br>
            <strong>Cargo:</strong> {meta["prepared_role"] or "-"}<br>
            <strong>Revisado por:</strong> {meta["reviewed_by"] or "-"}<br>
            <strong>Cargo revisión:</strong> {meta["reviewed_role"] or "-"}<br>
            <strong>Fecha del reporte:</strong> {meta["report_date"] or TODAY_STR}<br>
            <strong>Periodo evaluado:</strong> {meta["period"] or "No aplica"}<br>
            <strong>Consecutivo:</strong> {meta["consecutive"] or "-"}<br>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="wm-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="wm-block-subtitle">Resumen ejecutivo (página inicial del PDF)</div>', unsafe_allow_html=True)
    st.write(meta.get("executive_summary") or "—")
    st.markdown('<div class="wm-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="wm-block-subtitle">Objetivo del servicio</div>', unsafe_allow_html=True)
    st.write(meta["service_objective"] or "—")
    st.markdown('<div class="wm-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="wm-block-subtitle">Recomendaciones</div>', unsafe_allow_html=True)
    st.write(meta["recommendations"] or "—")
    st.markdown('<div class="wm-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="wm-block-subtitle">Desarrollo del servicio</div>', unsafe_allow_html=True)
    st.write(meta["service_development"] or "—")
    st.markdown("</div>", unsafe_allow_html=True)

with p2:
    st.markdown('<div class="wm-card">', unsafe_allow_html=True)
    st.markdown('<div class="wm-block-title">Resumen ordenado de figuras</div>', unsafe_allow_html=True)

    if not items:
        st.markdown('<div class="wm-note">No hay figuras agregadas todavía.</div>', unsafe_allow_html=True)
    else:
        for index, item in enumerate(items, start=1):
            summary_note = item["notes"][:240] + ("..." if len(item["notes"]) > 240 else "") if item["notes"] else ""
            badge_class = _type_badge_class(item["type"])
            st.markdown(
                f"""
                <div class="wm-preview-card">
                    <span class="wm-badge {badge_class}">{_type_badge(item["type"])}</span>
                    <span class="wm-badge wm-badge-generic">Figura {index}</span>
                    <strong>{item["title"]}</strong><br>
                    <span class="wm-muted">{_source_line(item)}</span><br><br>
                    <span class="wm-note">{summary_note}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("</div>", unsafe_allow_html=True)

st.caption(
    "Flujo actual: Spectrum, Waveform, Orbit y Tabular List empujan contenido real al reporte mediante st.session_state['report_items']. "
    "Reports actúa como editor premium del entregable técnico y exportador PDF profesional, sin reconstruir motores visuales."
)
