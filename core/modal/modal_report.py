"""
core/modal/modal_report.py — Builder de bloques modal para Reports Watermelon
==============================================================================

NO genera PDF independiente. En su lugar, prepara los bloques (tabla modal,
plots, hallazgos) en el formato que consume el sistema Reports existente de
Watermelon (`core/reports_archive.py` + `pages/16_Reports.py`).

El PDF final se genera con el template Watermelon estándar — los reportes
modales son una SECCIÓN dentro de un reporte Watermelon, no un PDF aparte.

Sprint pendiente
----------------
Wire la siguiente función al sistema Reports existente:
  · Recibir un modal_run (resultado de EMA o OMA)
  · Construir bloques compatibles con el report_state actual
  · Plots a PNG via plotly + kaleido
  · Tabla modal como bloque tabular
  · Caption + sección normativa

Norma aplicable
---------------
ISO 7626-6 §8 — Documentación de resultados modales (sin formato específico,
solo requiere fn, ζ, mode shape, método aplicado, validación)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ModalReportBlock:
    """Un bloque de contenido modal listo para inyectar al sistema Reports."""
    title: str
    kind: str  # "modal_table" | "modal_plot" | "modal_text" | "modal_compliance"
    data: Dict[str, Any] = field(default_factory=dict)
    caption: str = ""
    norm_ref: str = ""


def build_modal_blocks(
    modes: List[Dict],
    method: str,                       # "EMA" | "OMA"
    compliance_report: Optional[Dict] = None,
    figures_png: Optional[Dict[str, bytes]] = None,
) -> List[ModalReportBlock]:
    """
    Construye los bloques modales para inyectar al sistema Reports.

    Args:
        modes: lista de modos identificados
        method: "EMA" o "OMA"
        compliance_report: checklist ISO 7626-5 si aplica
        figures_png: dict {nombre: PNG bytes} con plots a embedir

    Returns:
        Lista de ModalReportBlock que el sistema Reports puede consumir.
    """
    blocks: List[ModalReportBlock] = []

    # Bloque 1: Tabla modal
    if modes:
        blocks.append(ModalReportBlock(
            title=f"Tabla modal — {method}",
            kind="modal_table",
            data={"modes": modes, "method": method},
            caption=f"Resultados de identificación modal por método {method}.",
            norm_ref="ISO 7626-6 §6.3",
        ))

    # Bloque 2: Compliance ISO 7626-5 (solo EMA)
    if compliance_report and method.upper() == "EMA":
        blocks.append(ModalReportBlock(
            title="Validación ISO 7626-5",
            kind="modal_compliance",
            data={"report": compliance_report},
            caption=f"{compliance_report.get('n_passed', 0)}/"
                      f"{compliance_report.get('n_total', 0)} checks aprobados.",
            norm_ref="ISO 7626-5 §6 + §7",
        ))

    # Bloques 3+: Plots
    if figures_png:
        for name, png in figures_png.items():
            blocks.append(ModalReportBlock(
                title=name,
                kind="modal_plot",
                data={"png_bytes": png},
                caption=f"Figura — {name}",
                norm_ref="",
            ))

    return blocks


def figure_to_png_bytes(fig, width: int = 1200, height: int = 700) -> bytes:
    """
    Convierte una figura Plotly a PNG bytes para insertar en Reports.

    Requiere kaleido (ya en requirements.txt).
    """
    try:
        import plotly.io as pio
        return pio.to_image(fig, format="png", width=width, height=height, scale=2)
    except (ImportError, Exception):
        return b""
