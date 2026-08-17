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
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import Image, Paragraph, Spacer, Table, TableStyle

from core.report_pdf_shell import make_styles, paragraph_safe  # noqa: F401

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
            out.append(Paragraph(f"<b>{i}. {paragraph_safe(title)}</b>",
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
               max_h_cm: float = 6.0) -> List[Any]:
    """Registro fotográfico en cuadrícula. photos: [{bytes, caption}]."""
    out: List[Any] = []
    if not photos:
        return out
    cell_w = 16.0 / cols       # cm por celda
    row: List[Any] = []
    grid_rows: List[List[Any]] = []
    for ph in photos:
        img = safe_image(ph.get("bytes"), cell_w - 0.4, max_h_cm)
        cap = Paragraph(paragraph_safe(ph.get("caption", "")), styles["WMFigureCaption"])
        cell = [img, cap] if img else [cap]
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


__all__ = [
    "make_styles", "p", "section", "subsection", "bullets", "numbered_plan",
    "kv_table", "two_col_kv", "grid_table", "safe_image", "photo_grid",
    "severity_table", "severity_legend", "signatures_block",
    "autofill_base_meta", "today_str",
]
