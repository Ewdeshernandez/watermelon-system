"""
core.reports_ext.common
=======================

Helpers compartidos para los reportes de campo (headless, sin Streamlit).
Se construyen sobre `core.report_pdf_shell` para conservar el formato SIGA
(portada + TOC clickeable + banda/pie con version stamp).

Incluye:
  - Flowables: section/subsection (van al TOC), párrafos, kv-table, grid-table,
    bullets, plan de trabajo numerado.
  - Imágenes: `safe_image` (redimensiona preservando aspecto con PIL) y
    `photo_grid` (registro fotográfico en cuadrícula con leyendas).
  - `severity_table`: tabla de hallazgos de boroscopia (acceso / hallazgos /
    severidad-comentarios / imagen) con la severidad coloreada.
  - `signatures_block`: firmas Contratista / Contratante.
  - `autofill_base_meta`: precarga cliente/planta/equipo/fecha desde el activo
    activo de Live Monitoring.
"""
from __future__ import annotations

import io
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import Image, Paragraph, Spacer, Table, TableStyle

from core.report_pdf_shell import make_styles, paragraph_safe  # noqa: F401


def _photo_caption_style() -> ParagraphStyle:
    """Estilo de leyenda de figura en CURSIVA (Helvetica-Oblique = itálica real;
    la fuente embebida no trae variante itálica)."""
    return ParagraphStyle(
        name="WMPhotoCaptionItalic", fontName="Helvetica-Oblique",
        fontSize=10, leading=13, alignment=TA_CENTER,
        textColor=colors.HexColor("#111827"), spaceBefore=5, spaceAfter=3)


def _photo_credit_style() -> ParagraphStyle:
    """Crédito discreto bajo cada figura (pequeño, gris, cursiva)."""
    return ParagraphStyle(
        name="WMPhotoCredit", fontName="Helvetica-Oblique",
        fontSize=7.5, leading=9.5, alignment=TA_CENTER,
        textColor=colors.HexColor("#94a3b8"), spaceAfter=8)


def photo_credit(year: Optional[int] = None) -> str:
    """Texto de crédito de fotografía (propiedad SIGA)."""
    y = year or datetime.now().year
    return f"Fotografía tomada por SIGA, {y}"

_HEADER_BG = "#0f4c81"
_OK = "#16a34a"
_WARN = "#D89B22"
_BAD = "#dc2626"
_INK = "#111827"


# ---------------------------------------------------------------------
# Flowables de texto
# ---------------------------------------------------------------------
def p(text: str, styles, style: str = "WMBody"):
    return Paragraph(paragraph_safe(str(text)), styles[style])


def section(title: str, styles):
    """Heading nivel 0 (entra al TOC)."""
    return Paragraph(paragraph_safe(title), styles["WMTOC1"])


def subsection(title: str, styles):
    """Heading nivel 1 (entra al TOC)."""
    return Paragraph(paragraph_safe(title), styles["WMTOC2"])


def bullets(items: List[str], styles) -> List[Any]:
    out: List[Any] = []
    for it in items:
        if str(it).strip():
            out.append(Paragraph("•&nbsp;&nbsp;" + paragraph_safe(str(it)),
                                  styles["WMClinicalBullet"]))
    return out


def numbered_plan(sections: List[Dict[str, Any]], styles) -> List[Any]:
    """Plan de trabajo: [{title, items:[...]}] → secciones numeradas con
    sub-bullets, estilo del formato SIGA-FMT-136."""
    out: List[Any] = []
    for i, sec in enumerate(sections, 1):
        title = str(sec.get("title", "")).strip()
        if title:
            # Sin negrita (pedido Ewdes): número + descripción en peso normal.
            out.append(Paragraph(f"{i}.&nbsp;&nbsp;{paragraph_safe(title)}",
                                  styles["WMClinicalNumbered"]))
        for it in sec.get("items", []) or []:
            if str(it).strip():
                out.append(Paragraph("•&nbsp;&nbsp;" + paragraph_safe(str(it)),
                                      styles["WMClinicalBullet"]))
        out.append(Spacer(1, 0.15 * cm))
    return out


def kv_table(rows: List[Tuple[str, str]], styles,
             col_widths: Optional[List[float]] = None) -> Table:
    data = [[Paragraph(f"<b>{paragraph_safe(k)}</b>", styles["WMTableCell"]),
             Paragraph(paragraph_safe(str(v)), styles["WMTableCell"])]
            for k, v in rows]
    t = Table(data, colWidths=col_widths or [5.0 * cm, 11.2 * cm])
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#e2e8f0")),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


def two_col_kv(rows: List[Tuple[str, str, str, str]], styles) -> Table:
    """Fila = (label_izq, valor_izq, label_der, valor_der). Formato de
    'DATOS DEL SERVICIO' en dos columnas como el SIGA-FMT-136."""
    data = []
    for la, va, lb, vb in rows:
        data.append([
            Paragraph(f"<b>{paragraph_safe(la)}</b>", styles["WMTableCell"]),
            Paragraph(paragraph_safe(str(va)), styles["WMTableCell"]),
            Paragraph(f"<b>{paragraph_safe(lb)}</b>" if lb else "", styles["WMTableCell"]),
            Paragraph(paragraph_safe(str(vb)) if lb else "", styles["WMTableCell"]),
        ])
    t = Table(data, colWidths=[3.2 * cm, 4.9 * cm, 3.2 * cm, 4.9 * cm])
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#e2e8f0")),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


def grid_table(headers: List[str], rows: List[List[Any]], styles,
               col_widths: Optional[List[float]] = None) -> Table:
    head = [Paragraph(f"<b>{paragraph_safe(h)}</b>", styles["WMTableHeader"]) for h in headers]
    body = [[Paragraph(paragraph_safe(str(c)), styles["WMTableCell"]) for c in r] for r in rows]
    t = Table([head] + body, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(_HEADER_BG)),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f1f5f9")]),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1")),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


# ---------------------------------------------------------------------
# Imágenes
# ---------------------------------------------------------------------
def _img_size(raw: bytes) -> Optional[Tuple[int, int]]:
    try:
        from PIL import Image as PILImage
        im = PILImage.open(io.BytesIO(raw))
        return im.size
    except Exception:
        return None


def safe_image(raw: bytes, max_w_cm: float, max_h_cm: float) -> Optional[Image]:
    """reportlab Image redimensionada dentro de (max_w, max_h) preservando
    el aspecto. None si no se puede leer."""
    if not raw:
        return None
    size = _img_size(raw)
    if not size:
        try:
            return Image(io.BytesIO(raw), width=max_w_cm * cm, height=max_h_cm * cm)
        except Exception:
            return None
    w, h = size
    if w <= 0 or h <= 0:
        return None
    ratio = min(max_w_cm * cm / w, max_h_cm * cm / h)
    try:
        return Image(io.BytesIO(raw), width=w * ratio, height=h * ratio)
    except Exception:
        return None


def photo_grid(photos: List[Dict[str, Any]], styles, cols: int = 2,
               max_h_cm: float = 6.0, credit: Optional[str] = None) -> List[Any]:
    """Registro fotográfico en cuadrícula. photos: [{bytes, caption}].
    credit: crédito discreto bajo cada figura (ej. 'Fotografía tomada por SIGA,
    2026'). None = sin crédito."""
    out: List[Any] = []
    if not photos:
        return out
    cell_w = 16.0 / cols       # cm por celda
    cap_style = _photo_caption_style()      # leyendas en cursiva
    credit_style = _photo_credit_style()    # crédito pequeño gris
    row: List[Any] = []
    grid_rows: List[List[Any]] = []
    for ph in photos:
        img = safe_image(ph.get("bytes"), cell_w - 0.4, max_h_cm)
        cap = Paragraph(paragraph_safe(ph.get("caption", "")), cap_style)
        cell = [img, cap] if img else [cap]
        if credit:
            cell.append(Paragraph(paragraph_safe(credit), credit_style))
        row.append(cell)
        if len(row) == cols:
            grid_rows.append(row); row = []
    if row:
        while len(row) < cols:
            row.append([Paragraph("", styles["WMFigureCaption"])])
        grid_rows.append(row)
    t = Table(grid_rows, colWidths=[cell_w * cm] * cols)
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    out.append(t)
    return out


# ---------------------------------------------------------------------
# Boroscopia — tabla de severidad
# ---------------------------------------------------------------------
def _severity_color(sev: str) -> str:
    s = (sev or "").strip().lower()
    if s.startswith(("no operativo", "no serviciable", "unservi")):
        return _BAD
    if s.startswith(("operativo", "serviciable", "servi")):
        return _OK
    return _WARN


def severity_table(rows: List[Dict[str, Any]], styles) -> Table:
    """Tabla de hallazgos de boroscopia. row: {access, findings, severity,
    comment, image_bytes}. Columnas: Acceso/Ubicación | Hallazgos evidenciados |
    Severidad / Comentarios | Imagen."""
    head = [Paragraph(f"<b>{h}</b>", styles["WMTableHeader"]) for h in
            ["Acceso / Ubicación", "Hallazgos evidenciados",
             "Severidad / Comentarios", "Imagen"]]
    data = [head]
    for r in rows:
        sev = str(r.get("severity", ""))
        col = _severity_color(sev)
        sev_html = (f'<b><font color="{col}">{paragraph_safe(sev)}</font></b><br/>'
                    f'{paragraph_safe(str(r.get("comment","")))}')
        img = safe_image(r.get("image_bytes"), 4.2, 4.0)
        data.append([
            Paragraph(paragraph_safe(str(r.get("access", ""))), styles["WMTableCell"]),
            Paragraph(paragraph_safe(str(r.get("findings", ""))), styles["WMTableCell"]),
            Paragraph(sev_html, styles["WMTableCell"]),
            img or Paragraph("—", styles["WMTableCell"]),
        ])
    t = Table(data, colWidths=[3.0 * cm, 4.6 * cm, 4.2 * cm, 4.4 * cm], repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(_HEADER_BG)),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def severity_legend(styles) -> Table:
    """Tabla 'Condición de severidades' (Operativo / No operativo)."""
    rows = [
        ["Operativo (Serviciable)",
         "El equipo puede seguir operando sin ninguna restricción de forma segura."],
        ["No operativo (Unserviciable)", "El equipo no puede ser operado."],
    ]
    data = [[Paragraph(f'<b><font color="{_OK if i==0 else _BAD}">{r[0]}</font></b>',
                       styles["WMTableCell"]),
             Paragraph(r[1], styles["WMTableCell"])] for i, r in enumerate(rows)]
    t = Table(data, colWidths=[5.0 * cm, 11.2 * cm])
    t.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


# ---------------------------------------------------------------------
# Firmas Contratista / Contratante (formato campo)
# ---------------------------------------------------------------------
def signatures_block(left: Dict[str, str], right: Dict[str, str], styles) -> Table:
    def _cell(sig: Dict[str, str], role_label: str):
        return [
            Paragraph("_______________________________", styles["WMTableCell"]),
            Paragraph(f"<b>{role_label}: {paragraph_safe(sig.get('org',''))}</b>",
                      styles["WMTableCell"]),
            Paragraph(f"Nombre: {paragraph_safe(sig.get('name',''))}", styles["WMTableCell"]),
            Paragraph(f"Cargo: {paragraph_safe(sig.get('role',''))}", styles["WMTableCell"]),
        ]
    t = Table([[_cell(left, "Contratista"), _cell(right, "Contratante")]],
              colWidths=[8.1 * cm, 8.1 * cm])
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    return t


# ---------------------------------------------------------------------
# Autofill desde el activo activo (Live Monitoring)
# ---------------------------------------------------------------------
def autofill_base_meta() -> Dict[str, str]:
    """Precarga cliente / planta / equipo / ubicación / fecha desde el activo
    activo. Devuelve dict con claves: client, plant, equipo, location, asset,
    train_description. Vacío si no hay activo."""
    base = {"client": "", "plant": "", "equipo": "", "location": "",
            "asset": "", "train_description": ""}
    try:
        from core.instance_selector import get_active_instance_id
        from core.instance_state import get_instance, compose_train_description
    except Exception:
        return base
    try:
        iid = get_active_instance_id()
        if not iid:
            return base
        inst = get_instance(iid)
        if inst is None:
            return base
        base["client"] = inst.client or ""
        base["plant"] = inst.site or inst.location or ""
        base["location"] = inst.location or inst.site or ""
        base["equipo"] = inst.tag or ""
        base["asset"] = inst.tag or ""
        try:
            base["train_description"] = compose_train_description(inst) or ""
        except Exception:
            pass
    except Exception:
        return base
    return base


def today_str() -> str:
    return datetime.now().strftime("%d/%m/%Y")


# ---------------------------------------------------------------------
# Consecutivo automático (ISO 9001:2015 §7.5.2 — identificación única)
# ---------------------------------------------------------------------
# Esquema: SIGA-{TIPO}-{AÑO}-{NNNN}. Contador persistente por tipo y año.
# ISO 9001:2015 §7.5.2 exige que la información documentada tenga
# identificación única (título, fecha, autor, número de referencia) y deja
# libre el formato; este esquema es trazable, ordenable y único.
TYPE_CODES = {
    "diario": "DIA", "preliminar": "PRE", "boroscopia": "BOR",
    "alineacion": "ALI", "mecanico": "MEC",
}

# Autoridades habilitadas para revisar/aprobar reportes (solo dos).
REVIEWERS = {
    "Ewdes A. Hernández B.": "Machinery Diagnostic Champion",
    "Ángel Leiva": "Ingeniero CBM",
}


def _state_dir() -> Optional[str]:
    """Directorio persistente (disco de Render vía WM_PERSIST_DIR) o local."""
    d = os.environ.get("WM_PERSIST_DIR") or os.path.join(
        os.path.expanduser("~"), ".watermelon_state")
    try:
        os.makedirs(d, exist_ok=True)
        return d
    except Exception:
        return None


def _counter_file() -> Optional[str]:
    d = _state_dir()
    return os.path.join(d, "report_consecutives.json") if d else None


def _load_counters() -> Dict[str, int]:
    f = _counter_file()
    if f and os.path.exists(f):
        try:
            with open(f, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return {}
    return {}


def _save_counters(c: Dict[str, int]) -> bool:
    f = _counter_file()
    if not f:
        return False
    try:
        with open(f, "w", encoding="utf-8") as fh:
            json.dump(c, fh)
        return True
    except Exception:
        return False


def _cons_key(family: str) -> str:
    return f"{TYPE_CODES.get(family, family)}-{datetime.now().year}"


def _fmt_cons(family: str, n: int) -> str:
    return f"SIGAGROUP-{TYPE_CODES.get(family, family)}-{datetime.now().year}-{n:04d}"


def peek_consecutive(family: str) -> str:
    """Siguiente consecutivo SIN incrementar (para mostrar en el formulario)."""
    if _counter_file() is None:
        return f"SIGAGROUP-{TYPE_CODES.get(family, family)}-{datetime.now():%Y-%m%d%H%M}"
    n = int(_load_counters().get(_cons_key(family), 0)) + 1
    return _fmt_cons(family, n)


def commit_consecutive(family: str) -> str:
    """Incrementa el contador y devuelve el consecutivo asignado."""
    if _counter_file() is None:
        return f"SIGAGROUP-{TYPE_CODES.get(family, family)}-{datetime.now():%Y-%m%d%H%M}"
    c = _load_counters()
    k = _cons_key(family)
    n = int(c.get(k, 0)) + 1
    c[k] = n
    _save_counters(c)
    return _fmt_cons(family, n)


__all__ = [
    "make_styles", "p", "section", "subsection", "bullets", "numbered_plan",
    "kv_table", "two_col_kv", "grid_table", "safe_image", "photo_grid",
    "severity_table", "severity_legend", "signatures_block",
    "autofill_base_meta", "today_str", "photo_credit",
    "TYPE_CODES", "REVIEWERS", "peek_consecutive", "commit_consecutive",
]
