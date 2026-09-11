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
    # Todas las curvas de valores singulares (SV1..SVn), no sólo SV1: revelan modos
    # cercanos. El payload trae `sv` = lista de curvas; `sv1` es el fallback legacy.
    _svm = svd.get("sv")
    if _svm is not None and len(_svm):
        sv = np.asarray(_svm, float)
        if sv.ndim == 1:
            sv = sv[None, :]
    else:
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


def mode_confidence(fn: float, zeta_pct: float, complexity_pct: float, cls: str,
                    ssi_freqs, rpm: float = 0.0) -> str:
    """Confianza de un modo (Alta/Media/Baja) combinando: confirmación cruzada FDD↔SSI,
    complejidad (MPC), amortiguación física y coincidencia con armónicos de giro.

    - Alta: confirmado por SSI, complejidad < 15%, damping físico (0.1–8%), no armónico.
    - Media: confirmado por SSI o complejidad < 15%, damping físico, no muy complejo.
    - Baja: complejidad > 40%, sin confirmación SSI, damping no físico (≈0 o > 15%),
            armónico de giro, o clasificado spurious/harmonic.
    """
    fn = float(fn or 0.0); z = float(zeta_pct or 0.0); cx = float(complexity_pct or 0.0)
    conf_ssi = any(abs(fn - float(sf)) <= max(0.02 * fn, 1.0) for sf in (ssi_freqs or []))
    harmonic = False
    if rpm:
        order = fn / (float(rpm) / 60.0) if rpm else 0.0
        k = round(order)
        harmonic = (k >= 1 and abs(order - k) <= 0.03)
    damp_bad = (z < 0.05) or (z > 15.0)
    if str(cls or "").lower() in ("spurious", "harmonic") or damp_bad or cx > 40.0 or harmonic:
        return "Baja"
    if conf_ssi and cx < 15.0 and (0.1 <= z <= 8.0):
        return "Alta"
    if (conf_ssi or cx < 15.0) and cx < 40.0:
        return "Media"
    return "Baja"


def _ssi_freqs_of(run: Dict[str, Any]):
    return [float(m.get("fn", 0)) for m in ((run.get("ssi") or {}).get("modes") or []) if m.get("fn")]


def _ssi_png_from_run(run: Dict[str, Any], fdd) -> "bytes | None":
    """Diagrama de estabilización SSI (fortaleza del análisis) desde run['ssi'].
    Polos estables (verde) vs espurios (gris) + densidad espectral (SV1) de fondo."""
    ssi = run.get("ssi") or {}
    diagram = ssi.get("diagram") or []
    if not diagram:
        return None
    try:
        import plotly.graph_objects as go
        mode_freqs = [float(m.get("fn", 0)) for m in (ssi.get("modes") or [])] or \
                     [float(getattr(m, "natural_frequency_hz", 0)) for m in getattr(fdd, "modes", [])]
        sx, sy, ux, uy = [], [], [], []
        omax = 40
        for (order, fr, mask) in diagram:
            omax = max(omax, int(order))
            for f, mk in zip(np.asarray(fr, float), mask):
                (sx if mk else ux).append(float(f)); (sy if mk else uy).append(int(order))
        xmax = max([max(sx) if sx else 0, max(ux) if ux else 0,
                    max(mode_freqs) if mode_freqs else 0, 50]) * 1.05
        fig = go.Figure()
        for f in mode_freqs:
            fig.add_vrect(x0=f * 0.985, x1=f * 1.015, fillcolor="rgba(22,163,74,.12)", line_width=0)
        # densidad espectral (SV1) de fondo
        _fr = np.asarray(getattr(fdd, "frequencies_hz", []), float)
        _sv = np.asarray(getattr(fdd, "singular_values", []), float)
        if _fr.size and _sv.size:
            _sv0 = _sv[0] if _sv.ndim > 1 else _sv
            fig.add_trace(go.Scatter(x=_fr, y=10.0 * np.log10(np.maximum(_sv0, 1e-30)),
                          mode="lines", name="Densidad espectral (SV1)",
                          line=dict(color="rgba(37,99,235,.45)", width=1.5), yaxis="y2", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=ux, y=uy, mode="markers", name="Polo espurio (numérico)",
                      marker=dict(size=4, color="#cbd5e1"), hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=sx, y=sy, mode="markers", name="Polo estable",
                      marker=dict(size=7, color="#16a34a", line=dict(width=.5, color="white"))))
        for f in mode_freqs:
            fig.add_annotation(x=f, y=omax, text=f"<b>{f:.1f}</b>", showarrow=False,
                               font=dict(size=9, color="#166534"), yanchor="bottom",
                               bgcolor="rgba(255,255,255,.85)")
        fig.update_layout(height=470, template="plotly_white",
                          xaxis=dict(range=[0, xmax], title="Frecuencia (Hz)"),
                          yaxis=dict(title="Orden del modelo", range=[0, omax * 1.12]),
                          yaxis2=dict(overlaying="y", side="right", showgrid=False, showticklabels=False),
                          legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
                          margin=dict(l=60, r=30, t=30, b=50))
        return fig.to_image(format="png", width=1100, height=560, scale=2)
    except Exception:  # noqa: BLE001
        return None


def _sensor_check_from_run(run: Dict[str, Any]):
    """(png, rows) de la verificación de sensores (fortaleza) desde run['sensor_check']."""
    sc = run.get("sensor_check") or {}
    if not isinstance(sc, dict):
        return None, None
    rows = sc.get("rows") or None
    png = None
    _b64 = sc.get("png_b64") or sc.get("png")
    if _b64:
        try:
            import base64
            png = base64.b64decode(_b64) if isinstance(_b64, str) else bytes(_b64)
        except Exception:  # noqa: BLE001
            png = None
    return png, rows


def build_report_from_run(run: Dict[str, Any], bilingual_es: bool = True,
                          shape_pngs=None, findings=None, recommendations=None,
                          meta_extra=None, config_png=None, sensor_png=None,
                          sensor_rows=None, ssi_png=None, max_shape_modes: int = 3,
                          filter_reliable: bool = True) -> bytes:
    """Genera el PDF OMA SIGA desde el payload de una corrida (`modal_runs`).

    `shape_pngs`: lista opcional de PNGs (vista 3D de geometría) por modo, en orden;
    reemplazan el diagrama de barras de la forma modal manteniendo el formato SIGA.
    `findings`/`recommendations`: si se pasan, sobrescriben los del payload.
    `meta_extra`: dict que se fusiona en `meta` (consecutive, prepared_by, reviewed_by,
    report_date, roles, ciudad, etc.) para portada/firmas/consecutivo del shell SIGA.
    `config_png`/`sensor_png`/`sensor_rows`: figuras extra (config 3D, verificación de
    sensores) embebidas en el reporte."""
    from core.modal.oma_siga_report import build_oma_siga_pdf
    from core.modal.campbell import SpeedBand
    from core.modal.ema_oma_correlation import correlate

    fdd = _fdd_from_run(run)
    rpm = float(run.get("running_rpm", 1185.0)) or 1185.0

    # --- Confianza por modo + FILTRO de confiabilidad (por defecto) -----------
    # EFDD ya viene en los modos capturados (default de captura desde v0.9.55). Aquí,
    # como haría un analista antes de emitir, dejamos SÓLO los modos confiables:
    # se descartan spurious/harmonic y los de confianza Baja (complejidad > 40%,
    # damping no físico, armónico de giro, sin confirmación SSI). Guarda de seguridad:
    # si TODOS resultaran Baja, no se vacía el reporte (se listan todos).
    _run_modes = list(run.get("modes") or [])
    _ssif = _ssi_freqs_of(run)
    _conf_all = [mode_confidence(mm.get("fn", 0), mm.get("zeta", 0), mm.get("complexity", 0),
                                 mm.get("class", "natural"), _ssif, rpm)
                 for mm in _run_modes]
    _conf = _conf_all
    if filter_reliable and _run_modes and any(c != "Baja" for c in _conf_all):
        _keep = [i for i, (mm, c) in enumerate(zip(_run_modes, _conf_all))
                 if c != "Baja"
                 and str(mm.get("class", "natural")).lower() not in ("spurious", "harmonic")]
        if _keep:
            fdd.modes = [fdd.modes[i] for i in _keep if i < len(fdd.modes)]
            _conf = [_conf_all[i] for i in _keep]

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

    # SSI (fortaleza): si no lo pasaron, generarlo automáticamente desde run["ssi"].
    if ssi_png is None:
        ssi_png = _ssi_png_from_run(run, fdd)
    # Verificación de sensores (fortaleza): idem desde run["sensor_check"].
    if sensor_png is None and sensor_rows is None:
        sensor_png, sensor_rows = _sensor_check_from_run(run)

    meta = {
        "report_title": "Reporte Análisis Modal Operacional (OMA)",
        "hide_format_band": True,        # sin banda/título de formato (FMT) — pedido del usuario
        "asset": run.get("asset") or run.get("name") or "Equipo",
        "client": run.get("client", ""), "location": run.get("location", ""),
        "prepared_by": "Watermelon System", "prepared_role": "Machinery Diagnostics",
    }
    if meta_extra:
        meta.update({k: v for k, v in meta_extra.items() if v not in (None, "")})
    # Técnica automática: EMA si la corrida es de impacto (sin modos OMA), OMA si no.
    _kind = str(run.get("kind", "") or "").lower()
    _technique = "EMA" if (_kind.startswith("ema") or (not fdd.modes and (run.get("ema_modes")))) else "OMA"
    # Tipo de sensor automático desde el layout: proximidad (meas_type "D") vs acelerómetro.
    _apts = [q for q in ((run.get("layout") or {}).get("points") or []) if q.get("active", True)]
    _nprox = sum(1 for q in _apts if str(q.get("meas_type", "A")).upper() == "D")
    if _apts and _nprox >= max(2, len(_apts) // 2):
        _sensor_kind = "proximity"
    elif _nprox > 0:
        _sensor_kind = "mixed"
    else:
        _sensor_kind = "accel"
    # (_conf ya calculado y filtrado arriba, alineado a fdd.modes)
    return build_oma_siga_pdf(
        meta=meta,
        conditions=[{"label": run.get("name", "Condición operacional"), "fdd_result": fdd,
                     "notes": "Procesamiento FDD de la corrida capturada en campo.",
                     "mode_confidence": _conf}],
        campbell=campbell, ema_oma=ema_oma, technique=_technique, sensor_kind=_sensor_kind,
        findings=findings if findings is not None else run.get("findings"),
        recommendations=recommendations if recommendations is not None else run.get("recommendations"),
        mode_shape_pngs=shape_pngs, config_png=config_png, sensor_png=sensor_png,
        sensor_rows=sensor_rows, ssi_png=ssi_png,
        max_shape_modes=max(int(max_shape_modes), len(shape_pngs or [])))
