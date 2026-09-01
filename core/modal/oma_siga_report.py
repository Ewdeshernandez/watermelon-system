"""
core/modal/oma_siga_report.py — Reporte OMA estilo SIGA (independiente de ARTeMIS)
=================================================================================

Ensambla un PDF de Análisis Modal Operacional con la MISMA estructura de tus
reportes (SIGA-FMT), usando el shell corporativo (portada + banda de formato +
firmas + TOC) y generando TODOS los gráficos desde el propio software:

  σ(f) (valores singulares FDD) · tabla de candidatos (freq/damping/complexity) ·
  complexity plots · matriz MAC · formas modales · Diagrama de Campbell con
  detección automática de cruces (API 684) · correlación EMA–OMA · normativa.

Reusa:
  - core.report_pdf_shell.render_report_pdf / make_styles   (motor SIGA)
  - core.reports_ext.common (section/p/grid_table/safe_image/…)
  - core.modal.modal_report._plotly_to_png                  (Plotly→PNG, kaleido)
  - core.modal.campbell / ema_oma_correlation / modal_animator / oma_engine
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

# Normativa por defecto (la de tus reportes)
DEFAULT_NORMS = [
    "API 684 – Tutorial on Rotor Dynamics and Balance, 4th Ed.",
    "API 670 – Machinery Protection Systems, 6th Ed.",
    "ISO 20816 / ISO 10816 – Evaluación de vibración en partes no rotativas.",
    "ISO 7626 (serie) – Determinación experimental de movilidad mecánica.",
    "ISO 7626-1 – Requisitos básicos de medición de FRF.",
    "ISO 7626-6 – Presentación de datos e identificación de parámetros modales.",
    "ISO 17359 – Monitoreo de condición y diagnóstico de máquinas.",
    "ISO 18436-2 – Requisitos de certificación de personal.",
]


def _png(fig, w: int = 1280, h: int = 620, scale: float = 1.5) -> Optional[bytes]:
    """Plotly → PNG (kaleido). Defensivo: None si falla (el PDF igual se arma)."""
    try:
        from core.modal.modal_report import _plotly_to_png
        return _plotly_to_png(fig, width=w, height=h, scale=scale)
    except Exception:  # noqa: BLE001
        return None


def _svd_figure(fdd, title: str = "Valores singulares de las densidades espectrales"):
    """Curva σ(f) en dB con SV1..SV3 y los modos marcados (estilo FDD)."""
    import plotly.graph_objects as go
    freqs = np.asarray(getattr(fdd, "frequencies_hz"))
    sv = np.asarray(getattr(fdd, "singular_values"))          # (N_ch, n_freq)
    if sv.ndim == 1:
        sv = sv[None, :]
    fig = go.Figure()
    colors = ["#0F1E3D", "#1AAEE5", "#94a3b8"]
    for i in range(min(3, sv.shape[0])):
        fig.add_trace(go.Scatter(x=freqs, y=10.0 * np.log10(np.maximum(sv[i], 1e-30)),
                                 mode="lines", name=f"SV{i+1}",
                                 line=dict(color=colors[i], width=1.6 if i == 0 else 1.0)))
    for m in getattr(fdd, "modes", []) or []:
        fn = float(getattr(m, "natural_frequency_hz", 0.0))
        j = int(np.argmin(np.abs(freqs - fn)))
        fig.add_trace(go.Scatter(x=[fn], y=[10.0 * np.log10(max(sv[0, j], 1e-30))],
                                 mode="markers", showlegend=False,
                                 marker=dict(size=9, symbol="circle-open",
                                             color="#dc2626", line=dict(width=2))))
    fig.update_layout(title=title, xaxis_title="Frecuencia (Hz)",
                      yaxis_title="Magnitud (dB)", template="plotly_white",
                      height=420, margin=dict(l=60, r=30, t=60, b=50),
                      legend=dict(orientation="h", y=-0.22))
    return fig


def _candidate_rows(fdd) -> List[List[Any]]:
    rows = []
    for i, m in enumerate(getattr(fdd, "modes", []) or [], 1):
        rows.append([
            i,
            f"{float(getattr(m, 'natural_frequency_hz', 0)):.3f}",
            f"{float(getattr(m, 'damping_ratio_pct', 0)):.3f}",
            f"{float(getattr(m, 'complexity_pct', 0)):.3f}",
            getattr(m, "classification", "natural"),
        ])
    return rows


def build_oma_siga_pdf(
    *,
    meta: Dict[str, Any],
    conditions: Sequence[Dict[str, Any]],
    campbell: Optional[Dict[str, Any]] = None,
    ema_oma: Optional[Sequence[Any]] = None,
    intro: str = "",
    background: str = "",
    findings: Optional[Sequence[str]] = None,
    recommendations: Optional[Sequence[str]] = None,
    instrumentation: str = "",
    norms: Optional[Sequence[str]] = None,
    max_shape_modes: int = 3,
) -> bytes:
    """Arma el PDF OMA SIGA. `conditions` = [{label, fdd_result, notes?}, ...].

    `campbell` = {modes_hz, rpm_min, rpm_max, bands?, operating_rpm?, mode_labels?,
                  classification?}. `ema_oma` = lista de ModeMatch (o resultado de
    ema_oma_correlation.correlate)."""
    from core.report_pdf_shell import render_report_pdf, make_styles
    from core.reports_ext.common import (section, subsection, p, numbered_list,
                                          bullets, grid_table, safe_image)
    from core.modal import modal_animator
    from core.modal.oma_engine import compute_mac_matrix

    styles = make_styles()
    body: List[Any] = []

    def _fig(fig, w_cm=16.5, h_cm=8.5, caption: str = ""):
        png = _png(fig)
        img = safe_image(png, w_cm, h_cm) if png else None
        if img is not None:
            body.append(img)
            if caption:
                body.append(p(caption, styles, "WMFigureCaption"))

    # 1 · Introducción
    body.append(section("1. Introducción y alcance", styles))
    body.append(p(intro or (
        "Se realizó un Análisis Modal Operacional (OMA) sobre el conjunto evaluado, con "
        "el objetivo de identificar las principales frecuencias naturales, factores de "
        "amortiguamiento y formas modales a partir de las respuestas vibratorias adquiridas "
        "durante la operación del equipo."), styles))

    # 2 · Antecedentes
    if background:
        body.append(section("2. Antecedentes", styles))
        body.append(p(background, styles))

    # 3 · Hallazgos
    if findings:
        body.append(section("3. Hallazgos", styles))
        body.extend(numbered_list(list(findings), styles))

    # 4 · Recomendaciones
    if recommendations:
        body.append(section("4. Recomendaciones finales", styles))
        body.extend(numbered_list(list(recommendations), styles))

    # 5 · Desarrollo / instrumentación
    body.append(section("5. Desarrollo del servicio", styles))
    body.append(p(instrumentation or (
        "La instrumentación se instaló siguiendo las buenas prácticas de medición estructural "
        "(API 684, ISO 20816), empleando acelerómetros piezoeléctricos calibrados con montaje "
        "rígido. Se verificó la fijación, el cableado y la sincronización de canales."), styles))

    # 6 · Resultados OMA — por condición
    body.append(section("6. Resultados – Análisis Modal Operacional (OMA)", styles))
    for k, cond in enumerate(conditions, 1):
        fdd = cond.get("fdd_result")
        label = cond.get("label", f"Condición {k}")
        body.append(subsection(f"6.{k} {label}", styles))
        if cond.get("notes"):
            body.append(p(cond["notes"], styles))
        # σ(f)
        try:
            _fig(_svd_figure(fdd), caption=f"Figura. Valores singulares – {label}.")
        except Exception:  # noqa: BLE001
            pass
        # tabla candidatos (NATIVA)
        rows = _candidate_rows(fdd)
        if rows:
            body.append(grid_table(
                ["#", "Frecuencia [Hz]", "Amortiguamiento [%]", "Complexity [%]", "Clasificación"],
                rows, styles))
            body.append(p(f"Tabla. Parámetros modales identificados – {label}.",
                          styles, "WMFigureCaption"))
        # figuras por modo (complexity polar + forma modal) para los primeros modos
        modes = list(getattr(fdd, "modes", []) or [])
        chn = getattr(fdd, "channel_names", None) or [f"CH{i+1}" for i in range(
            len(getattr(modes[0], "mode_shape", [])) if modes else 0)]
        for m in modes[:max_shape_modes]:
            fn = float(getattr(m, "natural_frequency_hz", 0))
            try:
                _fig(modal_animator.build_complexity_polar_plot(
                    getattr(m, "mode_shape"), chn, mode_label=f"{fn:.3f} Hz"),
                    w_cm=9.0, h_cm=8.0,
                    caption=f"Figura. Complexity Plot – modo {fn:.3f} Hz "
                            f"(complejidad {float(getattr(m,'complexity_pct',0)):.1f}%).")
            except Exception:  # noqa: BLE001
                pass
            try:
                _fig(modal_animator.build_bar_chart_mode_shape(
                    getattr(m, "mode_shape"), chn, mode_label=f"{fn:.3f} Hz"),
                    caption=f"Figura. Forma modal – modo {fn:.3f} Hz.")
            except Exception:  # noqa: BLE001
                pass
        # MAC de la condición
        try:
            if len(modes) >= 2:
                mac = compute_mac_matrix(modes)
                labels = [f"{float(getattr(mm,'natural_frequency_hz',0)):.1f}" for mm in modes]
                _fig(modal_animator.build_mac_matrix_plot(mac, labels,
                     title=f"Matriz MAC – {label}", use_3d=False),
                     w_cm=12.0, h_cm=10.0, caption=f"Figura. Matriz MAC – {label}.")
        except Exception:  # noqa: BLE001
            pass

    # 6.N · Campbell (auto-cruces)
    if campbell:
        from core.modal.campbell import (build_campbell_figure, compute_crossings,
                                          crossings_table, summarize as camp_summary,
                                          SpeedBand)
        bands = campbell.get("bands")
        if bands and bands and not isinstance(bands[0], SpeedBand):
            bands = [SpeedBand(**b) if isinstance(b, dict) else SpeedBand(*b) for b in bands]
        cx = compute_crossings(
            campbell["modes_hz"], campbell.get("rpm_min", 0.0), campbell["rpm_max"],
            orders=campbell.get("orders") or (0.5, 1.0, 2.0, 3.0, 4.0),
            bands=bands, mode_labels=campbell.get("mode_labels"),
            classification=campbell.get("classification"))
        body.append(section("6.C Evaluación mediante Diagrama de Campbell", styles))
        fig = build_campbell_figure(
            campbell["modes_hz"], campbell.get("rpm_min", 0.0), campbell["rpm_max"],
            orders=campbell.get("orders") or (0.5, 1.0, 2.0, 3.0, 4.0), bands=bands,
            mode_labels=campbell.get("mode_labels"), classification=campbell.get("classification"),
            operating_rpm=campbell.get("operating_rpm"), crossings=cx)
        _fig(fig, w_cm=17.0, h_cm=9.5, caption="Figura. Diagrama de Campbell con cruces detectados.")
        tbl = crossings_table(cx)
        coin = [r for r in tbl if r["Estado"] in ("Coincidencia", "Cercano")][:14]
        if coin:
            body.append(grid_table(
                ["Modo", "fn [Hz]", "Orden", "RPM cruce", "Banda", "Margen [%]", "Estado"],
                [[r["Modo"], r["fn [Hz]"], r["Orden"], r["RPM cruce"], r["Banda"],
                  r["Margen [%]"], r["Estado"]] for r in coin], styles))
            body.append(p("Tabla. Cruces fn ↔ orden dentro de las bandas de operación.",
                          styles, "WMFigureCaption"))
        body.append(p(camp_summary(cx), styles))

    # 7 · Correlación EMA–OMA
    if ema_oma:
        from core.modal.ema_oma_correlation import correlation_table, summarize as corr_summary
        body.append(section("7. Correlación dinámica complementaria EMA–OMA", styles))
        crows = correlation_table(ema_oma)
        if crows:
            heads = list(crows[0].keys())
            body.append(grid_table(heads, [[r[h] for h in heads] for r in crows], styles))
            body.append(p("Tabla. Correspondencia entre respuestas EMA y modos OMA.",
                          styles, "WMFigureCaption"))
        body.append(p(corr_summary(ema_oma), styles))

    # 8 · Normativa
    body.append(section("8. Normativa", styles))
    body.extend(bullets(list(norms or DEFAULT_NORMS), styles))

    return render_report_pdf(meta, body)
