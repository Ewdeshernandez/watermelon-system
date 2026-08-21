"""
core/remote_monitoring/analysis.py — funciones de análisis compartidas
======================================================================

Puras (numpy), reusadas por la app web y el módulo nativo:
  · detect_criticals   — picos de resonancia (críticas) de una curva amp vs rpm
  · half_power_af       — factor de amplificación (AF/SAF) por media potencia (API 684)
  · cascade_diagnosis   — auto-diagnóstico: modos + oil whirl / oil whip / ½X
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np


def detect_criticals(rr, aa, thr_frac: float = 0.4, merge_frac: float = 0.08,
                     top: int = 4) -> List[int]:
    """Índices de picos de resonancia (locales prominentes, fusionando cercanos),
    ordenados por rpm ascendente."""
    aa = np.asarray(aa, float)
    rr = np.asarray(rr, float)
    if len(aa) < 3:
        return [int(np.argmax(aa))] if len(aa) else []
    mx = float(np.max(aa))
    cand = [i for i in range(1, len(aa) - 1)
            if aa[i] >= aa[i - 1] and aa[i] >= aa[i + 1] and aa[i] > thr_frac * mx]
    span = (rr[-1] - rr[0]) if len(rr) > 1 else 1.0
    out: List[int] = []
    for i in cand:
        if out and abs(rr[i] - rr[out[-1]]) < merge_frac * span:
            if aa[i] > aa[out[-1]]:
                out[-1] = i
        else:
            out.append(i)
    out = sorted(sorted(out, key=lambda i: -aa[i])[:top], key=lambda i: rr[i])
    return out or [int(np.argmax(aa))]


def half_power_af(rr, aa, ipk):
    """Factor de amplificación (AF/SAF) por ancho de banda de media potencia
    (API 684): AF = Nc / (N2 − N1), con N1,N2 donde amp = pico/√2.
    Devuelve (AF, N1, N2, h) o None."""
    rr = np.asarray(rr, float)
    aa = np.asarray(aa, float)
    apk = aa[ipk]
    if apk <= 0:
        return None
    h = apk / np.sqrt(2.0)
    N1 = N2 = None
    for i in range(ipk, 0, -1):
        if aa[i] >= h > aa[i - 1]:
            N1 = float(np.interp(h, [aa[i - 1], aa[i]], [rr[i - 1], rr[i]])); break
    for i in range(ipk, len(aa) - 1):
        if aa[i] >= h > aa[i + 1]:
            N2 = float(np.interp(h, [aa[i + 1], aa[i]], [rr[i + 1], rr[i]])); break
    if N1 is None or N2 is None or N2 <= N1:
        return None
    return rr[ipk] / (N2 - N1), N1, N2, h


def cascade_diagnosis(rpms, freqs, mat, crit_rpms) -> List[Tuple[str, str, str]]:
    """Auto-diagnóstico de la cascada: velocidades críticas (modos) +
    inestabilidades subsíncronas con nombre propio (oil whirl / oil whip / ½X).
    Devuelve lista de (nivel, título, detalle). nivel ∈ {info, warn, danger}."""
    out: List[Tuple[str, str, str]] = []
    for nc in crit_rpms:
        out.append(("info", f"Velocidad crítica ≈ {nc:.0f} rpm",
                    "Resonancia de un modo de flexión del rotor (se amplifica el 1X). "
                    "Verificar margen de separación (API 684)."))
    freqs = np.asarray(freqs, float)
    rpms = np.asarray(rpms, float)
    mat = np.asarray(mat, float)
    if not len(rpms) or not len(freqs):
        return out
    df = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 1.0
    gmax = float(np.max(mat)) if mat.size else 0.0
    pts = []
    for i, rp in enumerate(rpms):
        f1 = rp / 60.0
        if f1 <= 0:
            continue
        band = (freqs >= 0.15 * f1) & (freqs <= 0.9 * f1)
        if not band.any():
            continue
        sub = mat[i][band]
        fb = freqs[band]
        j = int(np.argmax(sub))
        af, ff = float(sub[j]), float(fb[j])
        b1 = np.abs(freqs - f1) <= max(df, 0.05 * f1)
        a1 = float(mat[i][b1].max()) if b1.any() else 0.0
        if af > 0.25 * max(a1, 1e-9) and af > 0.06 * max(gmax, 1e-9):
            pts.append((rp, ff, ff / f1, af))
    if len(pts) < max(4, len(rpms) // 8):
        return out
    P = np.array(pts, float)
    ffreq, orders = P[:, 1], P[:, 2]
    mo = float(np.median(orders))
    f_cv = float(np.std(ffreq) / (np.mean(ffreq) + 1e-9))
    o_cv = float(np.std(orders) / (np.mean(orders) + 1e-9))
    near_crit = bool(len(crit_rpms)) and any(
        abs(np.mean(ffreq) * 60.0 - nc) < 0.18 * nc for nc in crit_rpms)
    if f_cv < 0.14 and o_cv > 0.18 and near_crit:
        out.append(("danger", "⚠ OIL WHIP (latigazo de aceite)",
                    f"Subsíncrono ENGANCHADO a una natural (~{np.mean(ffreq):.0f} Hz ≈ 1ª crítica), "
                    f"fijo mientras sube la velocidad. Inestabilidad de película SEVERA (API 684) — actuar."))
    elif 0.35 <= mo <= 0.49 and o_cv < 0.16:
        out.append(("danger", "⚠ OIL WHIRL (remolino de aceite)",
                    f"Subsíncrono a ~{mo:.2f}X que SIGUE la velocidad: inestabilidad de la película "
                    f"de aceite del cojinete (API 684). Puede degenerar en oil whip."))
    elif 0.47 <= mo <= 0.53 and o_cv < 0.14:
        out.append(("warn", "Subarmónico ½X",
                    "Componente a ~0.5X que sigue la velocidad → roce (rub) o holgura mecánica, "
                    "NO película de aceite."))
    else:
        out.append(("warn", f"Subsíncrono ~{mo:.2f}X",
                    "Energía por debajo del 1X; vigilar su evolución con la velocidad."))
    return out
