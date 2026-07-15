"""
core/modal/oma_interpret.py — Diagnóstico e interpretación automática de OMA
============================================================================

Lee un FDDResult y genera OBSERVACIONES automáticas en lenguaje llano + una
conclusión técnica, para que el usuario entienda qué significan los resultados
y qué tan válidos son — sin necesidad de ser experto en análisis modal.

Motivación (feedback de campo v3.31.437): los resultados OMA muestran valores
(frecuencia, damping, MPC, AutoMAC) pero no explican su significado ni la
CALIDAD/VALIDEZ del modo, y esto es crítico cuando se ensaya con UN solo
sensor (donde mode shape, MPC y AutoMAC no tienen sentido espacial).

Sin dependencias de numpy — trabaja solo con los escalares del resultado.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Literal

Severity = Literal["ok", "info", "warn", "crit"]


@dataclass
class Observation:
    severity: Severity
    title: str
    detail: str


# Umbrales físicos razonables (estructural / rotodinámico)
_DAMP_HIGH = 10.0     # % — por encima: sospechoso (registro corto / pico ancho)
_DAMP_VERY_HIGH = 20.0
_DAMP_LOW = 0.2       # % — por debajo: sospechosamente bajo
_DAMP_OK_LO, _DAMP_OK_HI = 0.5, 5.0
_MIN_SENSORS_SHAPE = 3   # mínimo para empezar a resolver forma espacial


def _natural_modes(fdd: Any) -> List[Any]:
    return [m for m in getattr(fdd, "modes", [])
            if getattr(m, "classification", "natural") == "natural"]


def svd_detection_report(freqs, sv1_db, f_min_hz: float, f_max_hz: float,
                         prominence_db: float, top: int = 6) -> dict:
    """Analiza la curva SVD y reporta los picos CANDIDATOS y por qué se
    aceptan/descartan, para dar transparencia al criterio del FDD (feedback
    v3.31.444). Devuelve dict con: n_at_threshold (picos que superan la
    prominencia), candidates [(hz, prominencia_db)] (los más marcados), best_hz,
    best_prom, noise_floor_db. Nunca lanza."""
    try:
        import numpy as np
        from scipy.signal import find_peaks
    except Exception:  # noqa: BLE001
        return {}
    try:
        f = np.asarray(freqs, dtype=float)
        y = np.asarray(sv1_db, dtype=float)
        mask = (f >= f_min_hz) & (f <= f_max_hz)
        fb, yb = f[mask], y[mask]
        if fb.size < 5:
            return {"n_at_threshold": 0, "candidates": [], "best_hz": None,
                    "best_prom": 0.0, "noise_floor_db": None}
        pk_thr, _ = find_peaks(yb, prominence=max(prominence_db, 0.01))
        pk_all, props_all = find_peaks(yb, prominence=0.5)
        proms = props_all.get("prominences", np.array([]))
        order = list(np.argsort(proms)[::-1][:top]) if proms.size else []
        candidates = [(float(fb[pk_all[i]]), float(proms[i])) for i in order]
        # Detalle por candidato: incluye el valor singular (dB) en el pico y si
        # supera el umbral de prominencia — alimenta la tabla de candidatos con
        # el motivo de aceptación/rechazo (v3.31.451).
        candidates_detail = [{
            "freq_hz": float(fb[pk_all[i]]),
            "prominence_db": float(proms[i]),
            "sv_db": float(yb[pk_all[i]]),
            "passes": bool(float(proms[i]) >= prominence_db),
        } for i in order]
        best_hz = candidates[0][0] if candidates else None
        best_prom = candidates[0][1] if candidates else 0.0
        return {"n_at_threshold": int(pk_thr.size), "candidates": candidates,
                "candidates_detail": candidates_detail,
                "best_hz": best_hz, "best_prom": best_prom,
                "noise_floor_db": float(np.median(yb))}
    except Exception:  # noqa: BLE001
        return {}


def interpret_fdd(fdd: Any) -> dict:
    """Devuelve {'observations': [Observation], 'conclusion': str,
    'single_sensor': bool}. Nunca lanza — si algo falta, degrada."""
    obs: List[Observation] = []
    try:
        n_ch = int(getattr(fdd, "n_channels", 0) or 0)
    except Exception:  # noqa: BLE001
        n_ch = 0
    modes = list(getattr(fdd, "modes", []) or [])
    naturals = _natural_modes(fdd)
    n_nat = len(naturals)
    n_harm = sum(1 for m in modes
                 if getattr(m, "classification", "") == "harmonic")
    dur = float(getattr(fdd, "duration_s", 0.0) or 0.0)
    single = n_ch <= 1

    # --- 1) Nº de sensores: el condicionante más importante ---------------
    if single:
        obs.append(Observation(
            "crit", "Ensayo con UN solo sensor — alcance limitado",
            "Con un único sensor solo se pueden estimar **frecuencias "
            "naturales** y, de forma aproximada, el **amortiguamiento**. "
            "La **forma modal (mode shape), el MPC/complejidad y la matriz "
            "AutoMAC NO tienen significado espacial**: el mode shape es un "
            "único punto (por eso se ve una sola barra y fase 0°), el MPC da "
            "0% de forma trivial, y el AutoMAC da 1.00 entre todos los modos "
            "(todos son colineales al tener 1 grado de libertad). Para "
            "resolver formas modales reales usa **≥3–4 sensores** repartidos "
            "en la estructura."))
    elif n_ch < _MIN_SENSORS_SHAPE:
        obs.append(Observation(
            "warn", f"Pocos sensores ({n_ch}) para forma modal",
            "Con 2 sensores la forma modal es muy pobre y el AutoMAC entre "
            "modos tiende a ser alto (falsos 'redundantes'). Suma sensores "
            "para una identificación espacial confiable."))
    else:
        obs.append(Observation(
            "ok", f"{n_ch} sensores — forma modal resoluble",
            "Con esta cantidad de canales el mode shape, el MPC y el AutoMAC "
            "sí aportan información espacial útil."))

    # --- 2) Duración del registro ----------------------------------------
    if dur and dur < 200.0:
        obs.append(Observation(
            "warn", f"Registro corto ({dur:.0f} s)",
            "OMA/FDD asume excitación ambiental estacionaria y necesita "
            "registros largos (norma ISO 16649: al menos 1000×T_low, "
            "recomendado 2000×T_low). Con registros cortos el "
            "**amortiguamiento** tiene mucha varianza y las frecuencias "
            "bajas quedan mal resueltas. Para reporte, recaptura con más "
            "tiempo."))
    elif dur:
        obs.append(Observation(
            "ok", f"Duración de registro {dur:.0f} s",
            "Duración suficiente para una estimación estable en el rango "
            "identificado."))

    # --- 3) Amortiguamiento por modo natural -----------------------------
    for m in naturals:
        z = float(getattr(m, "damping_ratio_pct", 0.0) or 0.0)
        fn = float(getattr(m, "natural_frequency_hz", 0.0) or 0.0)
        if z >= _DAMP_VERY_HIGH:
            obs.append(Observation(
                "warn", f"Damping muy alto en {fn:.1f} Hz (ζ={z:.1f}%)",
                "Un amortiguamiento tan alto casi nunca es físico en "
                "estructura o rotor: suele indicar **registro corto**, un "
                "pico ancho/mal separado, o dos modos solapados leídos como "
                "uno. Interpreta esta frecuencia con cautela y recaptura con "
                "más tiempo."))
        elif z >= _DAMP_HIGH:
            obs.append(Observation(
                "info", f"Damping elevado en {fn:.1f} Hz (ζ={z:.1f}%)",
                "Por encima de ~10% conviene confirmar con un registro más "
                "largo; el valor típico estructural es 0.5–5%."))
        elif 0 < z < _DAMP_LOW:
            obs.append(Observation(
                "info", f"Damping muy bajo en {fn:.1f} Hz (ζ={z:.2f}%)",
                "Amortiguamiento sospechosamente bajo — puede ser un pico muy "
                "agudo bien definido, o un artefacto. Verifica que sea un modo "
                "y no una armónica."))

    # --- 4) Armónicas presentes ------------------------------------------
    if n_harm:
        obs.append(Observation(
            "info", f"{n_harm} pico(s) clasificado(s) como armónica",
            "Se detectaron picos alineados con múltiplos de la velocidad de "
            "giro: son **armónicas** (forzadas por la máquina), NO modos "
            "naturales. Se excluyen del conteo de modos. Si la máquina estaba "
            "parada, revisa el valor de 'running speed'."))

    # --- 5) Confianza promedio -------------------------------------------
    if naturals:
        conf = sum(float(getattr(m, "confidence", 0.0)) for m in naturals) / n_nat
        if conf < 0.6:
            obs.append(Observation(
                "warn", f"Confianza promedio baja ({conf*100:.0f}%)",
                "Los modos identificados tienen baja nitidez/coherencia. "
                "Recaptura con más tiempo y/o revisa el montaje del sensor."))

    # --- Conclusión técnica ----------------------------------------------
    if n_nat == 0:
        conclusion = (
            "No se identificaron modos naturales claros en este ensayo. "
            "Revisa el rango de frecuencia (f_min/f_max), la prominencia, y "
            "que el sensor estuviera bien acoplado y midiendo vibración real.")
    else:
        freqs = ", ".join(f"{float(m.natural_frequency_hz):.1f} Hz"
                          for m in sorted(naturals,
                                          key=lambda x: x.natural_frequency_hz))
        if single:
            conclusion = (
                f"Se identificaron **{n_nat} frecuencia(s) natural(es)**: "
                f"{freqs}. Con un solo sensor esta es la conclusión válida del "
                "ensayo: **las frecuencias a las que la estructura resuena**. "
                "Para caracterizar la FORMA de cada modo (y validar con "
                "AutoMAC/MPC) se requieren varios sensores.")
        else:
            conclusion = (
                f"Se identificaron **{n_nat} modo(s) natural(es)** en {freqs}. "
                "Prioriza para reporte los modos con damping 0.5–5%, MPC/"
                "complejidad baja y AutoMAC off-diagonal < 0.7 (bien "
                "separados de los demás).")

    return {"observations": obs, "conclusion": conclusion,
            "single_sensor": single}
