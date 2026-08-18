"""
core.reports_ext — Reportes de campo adicionales de Watermelon.

Extiende el módulo Reportes (que ya tiene el reporte del sistema de vibraciones)
con reportes de campo bajo el MISMO formato SIGA (portada + TOC + secciones del
shell `core.report_pdf_shell`):

  - daily        → Reporte Diario (SIGA-FMT-136)
  - preliminary  → Reporte Preliminar
  - borescope    → Reporte de Inspección Boroscópica (SIGA-FMT-178)
  - alignment    → Reporte de Alineación
  - mechanical   → Reporte Mecánico

Cada builder es headless (no depende de Streamlit) y recibe metadatos + su
contenido; la UI (page.py) hace el wiring con formularios y carga de fotos.
"""
from __future__ import annotations

REPORT_FAMILIES = [
    ("sistema", "Reporte del Sistema"),
    ("diario", "Diario"),
    ("preliminar", "Preliminar"),
    ("boroscopia", "Boroscopia"),
    ("alineacion", "Alineación"),
    ("mecanico", "Mecánico"),
]

__all__ = ["REPORT_FAMILIES"]
