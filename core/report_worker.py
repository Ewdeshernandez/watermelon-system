"""Punto de entrada liviano para armar el reporte PDF en un proceso aparte.

La API (api/app.py) lo ejecuta con ProcessPoolExecutor(spawn): el hijo importa
SOLO esto (no la app FastAPI), arma el PDF y termina — la memoria vuelve al
sistema. Ver _report_pdf_isolated en api/app.py.
"""
from __future__ import annotations


def build_report(asset: str):
    from core.instance_state import get_instance
    from core.live_report_builder import build_report_for_instance
    return build_report_for_instance(asset, get_instance(asset))
