"""
core/modal/run_report.py — Reporte OMA desde una corrida subida por el campo
============================================================================

Cierra el ciclo campo → nube → reporte: reconstruye una corrida OMA (payload de
`modal_runs`) a un objeto tipo FDDResult y arma el PDF SIGA con
`oma_siga_report.build_oma_siga_pdf` (Campbell + correlación EMA↔OMA incluidos).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np


@dataclass
class _Mode:
    natural_frequency_hz: float
    damping_ratio_pct: float
    complexity_pct: float
    classification: str
    mode_shape: np.ndarray


@dataclass
class _FDD:
    frequencies_hz: np.ndarray
    singular_values: np.ndarray
    modes: List[_Mode]
    channel_names: List[str]


def _fdd_from_run(run: Dict[str, Any]) -> _FDD:
    svd = run.get("svd") or {}
    freqs = np.asarray(svd.get("freqs", []), float)
    sv1 = np.asarray(svd.get("sv1", []), float)
    sv = sv1[None, :] if sv1.size else np.zeros((1, freqs.size))
    modes = []
    for m in run.get("modes", []):
        sh = m.get("shape") or {}
        vec = np.asarray(sh.get("re", []), float) + 1j * np.asarray(sh.get("im", []), float)
        modes.append(_Mode(float(m.get("fn", 0)), float(m.get("zeta", 0)),
                           float(m.get("complexity", 0)), m.get("class", "natural"), vec))
    chn = run.get("channel_names") or [f"P{i+1}" for i in range(
        len(modes[0].mode_shape) if modes else 0)]
    return _FDD(freqs, sv, modes, chn)


def build_report_from_run(run: Dict[str, Any], bilingual_es: bool = True) -> bytes:
    """Genera el PDF OMA SIGA desde el payload de una corrida (`modal_runs`)."""
    from core.modal.oma_siga_report import build_oma_siga_pdf
    from core.modal.campbell import SpeedBand
    from core.modal.ema_oma_correlation import correlate

    fdd = _fdd_from_run(run)
    rpm = float(run.get("running_rpm", 1185.0)) or 1185.0
    modes_hz = [m.natural_frequency_hz for m in fdd.modes]

    campbell = None
    if modes_hz:
        campbell = {
            "modes_hz": modes_hz, "rpm_min": 0.0, "rpm_max": max(rpm * 1.4, 1500.0),
            "operating_rpm": rpm,
            "bands": [{"center_rpm": rpm, "tol_rpm": 0.15 * rpm, "label": f"Operación {rpm:.0f}±15%"},
                      {"center_rpm": rpm / 2, "tol_rpm": 0.15 * rpm / 2, "label": "½ velocidad"}],
            "mode_labels": [f"Modo {i+1}" for i in range(len(modes_hz))],
        }

    ema_oma = None
    ema = run.get("ema_modes") or []
    if ema and modes_hz:
        ema_oma = correlate(ema, modes_hz, tol_hz=2.5,
                            oma_labels=[f"{f:.3f}" for f in modes_hz])

    meta = {
        "report_title": "Reporte Análisis Modal Operacional (OMA)",
        "format_code": "SIGA-FMT-179", "format_version": "1",
        "asset": run.get("asset") or run.get("name") or "Equipo",
        "client": run.get("client", ""), "location": run.get("location", ""),
        "prepared_by": "Watermelon System", "prepared_role": "Machinery Diagnostics",
    }
    return build_oma_siga_pdf(
        meta=meta,
        conditions=[{"label": run.get("name", "Condición operacional"), "fdd_result": fdd,
                     "notes": "Procesamiento FDD de la corrida capturada en campo."}],
        campbell=campbell, ema_oma=ema_oma,
        findings=run.get("findings"), recommendations=run.get("recommendations"))
