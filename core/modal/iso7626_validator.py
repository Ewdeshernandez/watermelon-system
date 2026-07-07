"""
core/modal/iso7626_validator.py — Validador automático de ensayos modales
==========================================================================

Aplica los criterios objetivos de ISO 7626-5 (impact hammer testing) a un
conjunto input + output(s) capturados y devuelve un checklist con
pass/fail por cada criterio.

Esto es lo que diferencia Watermelon de Artemis / CSI Emerson / Bently:
ninguno de esos te dice "tu ensayo es conforme ISO 7626" — solo te dan
los datos. Watermelon valida automáticamente y emite un veredicto.

Criterios implementados
-----------------------
1. **Single peak input** — ISO 7626-5 secc. 7.3.2
     El impacto debe ser una sola descarga, sin "doble golpe". Validamos
     que el segundo pico mayor en el input sea al menos 6× menor que el
     primario (rejection ratio).

2. **Input spectrum flat** — ISO 7626-5 secc. 6.3
     El espectro del martillo debe ser plano (±10 dB) hasta la frecuencia
     objetivo. Si decae prematuramente, la banda alta no está bien excitada
     y los modos allí no son confiables.

3. **Output decay** — ISO 7626-5 secc. 7.3.3
     La respuesta debe decaer ≥ 90% antes del fin del record. Si no, hay
     leakage espectral y los modos quedan distorsionados.

4. **Coherence band** — ISO 7626-5 secc. 7.4
     γ²(f) ≥ 0.8 (típico) o ≥ 0.9 (estricto) en la banda de interés.
     Si falla en alguna sub-banda, los modos allí no son confiables.

5. **Number of averages** — ISO 7626-5 secc. 7.3.4
     Mínimo 5 impactos promediados por punto de medición.

Cada check devuelve un CheckResult dataclass con:
  · passed: bool
  · severity: "ok" | "warning" | "fail"
  · title: str — etiqueta corta
  · detail: str — explicación con números reales
  · norm_ref: str — referencia explícita a la sección de la norma
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math
import numpy as np


# =====================================================================
# Tipos
# =====================================================================

@dataclass
class CheckResult:
    """Resultado de un check individual del estándar."""
    passed: bool
    severity: str            # "ok" | "warning" | "fail"
    title: str
    detail: str
    norm_ref: str
    measured_value: Optional[float] = None
    threshold: Optional[float] = None


@dataclass
class ComplianceReport:
    """Reporte completo de cumplimiento ISO 7626-5."""
    checks: List[CheckResult] = field(default_factory=list)
    test_setup_name: str = ""
    norm: str = "ISO 7626-5"

    @property
    def n_passed(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    @property
    def n_total(self) -> int:
        return len(self.checks)

    @property
    def overall_pass(self) -> bool:
        return self.n_passed == self.n_total and self.n_total > 0

    @property
    def has_fails(self) -> bool:
        return any(c.severity == "fail" for c in self.checks)

    @property
    def has_warnings(self) -> bool:
        return any(c.severity == "warning" for c in self.checks)


# =====================================================================
# Checks individuales
# =====================================================================

def check_single_peak_input(
    input_signal: np.ndarray,
    sample_rate_hz: float,
    rejection_ratio: float = 6.0,
    min_separation_ms: float = 5.0,
) -> CheckResult:
    """
    ISO 7626-5 secc. 7.3.2 — Validar que el impacto sea único.

    Busca el peak global del input. Luego busca el segundo peak más alto
    en ventanas separadas del primero. Si el ratio peak1/peak2 < rejection_ratio
    → doble golpe → FAIL.
    """
    x = np.abs(np.asarray(input_signal, dtype=float))
    if x.size < 16:
        return CheckResult(False, "fail", "Single peak input",
                           "Señal demasiado corta para validar",
                           "ISO 7626-5 secc. 7.3.2")

    peak1_idx = int(np.argmax(x))
    peak1 = float(x[peak1_idx])
    if peak1 < 1e-12:
        return CheckResult(False, "fail", "Single peak input",
                           "Señal de input plana (sin impacto detectable)",
                           "ISO 7626-5 secc. 7.3.2")

    # Buscar segundo peak fuera de una ventana de ±min_separation_ms del primero
    window_samples = max(1, int(min_separation_ms * 1e-3 * sample_rate_hz))
    masked = x.copy()
    start = max(0, peak1_idx - window_samples)
    end = min(x.size, peak1_idx + window_samples)
    masked[start:end] = 0.0
    peak2 = float(masked.max()) if masked.size > 0 else 0.0

    ratio = peak1 / max(peak2, peak1 * 1e-9)
    passed = ratio >= rejection_ratio
    severity = "ok" if passed else ("warning" if ratio >= 3.0 else "fail")
    detail = (f"Peak primario {peak1:.3f}, peak secundario {peak2:.3f}, "
              f"ratio {ratio:.1f}× (mínimo {rejection_ratio}×).")
    if not passed:
        detail += " ⚠ Posible doble golpe — re-tomar impacto."
    return CheckResult(passed, severity, "Single peak input", detail,
                       "ISO 7626-5 secc. 7.3.2",
                       measured_value=ratio, threshold=rejection_ratio)


def check_input_spectrum_flat(
    input_signal: np.ndarray,
    sample_rate_hz: float,
    f_target_hz: float,
    flatness_db: float = 10.0,
    f_min_hz: float = 5.0,
) -> CheckResult:
    """
    ISO 7626-5 secc. 6.3 — El auto-espectro del martillo debe ser plano (±flatness_db)
    desde f_min hasta f_target.
    """
    try:
        from scipy.signal import welch
    except ImportError:
        return CheckResult(False, "fail", "Input spectrum flat",
                           "scipy no disponible", "ISO 7626-5 secc. 6.3")

    x = np.asarray(input_signal, dtype=float)
    if x.size < 64:
        return CheckResult(False, "fail", "Input spectrum flat",
                           "Señal demasiado corta", "ISO 7626-5 secc. 6.3")

    nperseg = min(x.size, 1024)
    freq, psd = welch(x, fs=sample_rate_hz, nperseg=nperseg)
    mag_db = 10.0 * np.log10(np.maximum(psd, 1e-30))

    mask = (freq >= f_min_hz) & (freq <= f_target_hz)
    if not np.any(mask):
        return CheckResult(False, "fail", "Input spectrum flat",
                           "Banda f_min→f_target no cubierta", "ISO 7626-5 secc. 6.3")

    band = mag_db[mask]
    deviation = float(band.max() - band.min())
    passed = deviation <= flatness_db
    severity = "ok" if passed else ("warning" if deviation <= flatness_db * 1.5 else "fail")
    detail = (f"Variación en banda {f_min_hz:.0f}–{f_target_hz:.0f} Hz: "
              f"{deviation:.1f} dB (máximo permitido {flatness_db:.0f} dB).")
    if not passed:
        detail += (" ⚠ Martillo con cabeza muy dura o muy blanda — ajustar "
                   "para que excite uniformemente la banda objetivo.")
    return CheckResult(passed, severity, "Input spectrum flat", detail,
                       "ISO 7626-5 secc. 6.3",
                       measured_value=deviation, threshold=flatness_db)


def check_response_decay(
    output_signal: np.ndarray,
    decay_fraction: float = 0.10,
) -> CheckResult:
    """
    ISO 7626-5 secc. 7.3.3 — La respuesta debe decaer significativamente antes
    del fin del record. Si los últimos 10% del record tienen RMS > decay_fraction
    del RMS máximo → leakage probable → FAIL.
    """
    y = np.abs(np.asarray(output_signal, dtype=float))
    n = y.size
    if n < 100:
        return CheckResult(False, "fail", "Response decay",
                           "Señal demasiado corta", "ISO 7626-5 secc. 7.3.3")

    # RMS en ventana inicial (primer 20%) vs ventana final (último 10%)
    n_init = max(10, n // 5)
    n_final = max(10, n // 10)
    rms_init = float(np.sqrt((y[:n_init] ** 2).mean()))
    rms_final = float(np.sqrt((y[-n_final:] ** 2).mean()))

    if rms_init < 1e-12:
        return CheckResult(False, "fail", "Response decay",
                           "Respuesta inicial nula", "ISO 7626-5 secc. 7.3.3")

    ratio = rms_final / rms_init
    passed = ratio <= decay_fraction
    severity = "ok" if passed else ("warning" if ratio <= 0.25 else "fail")
    detail = (f"RMS final / inicial = {ratio*100:.1f}% "
              f"(máximo permitido {decay_fraction*100:.0f}%).")
    if not passed:
        detail += (" ⚠ La respuesta no decayó suficiente — aumentar duración "
                   "del record o aplicar exponential window.")
    return CheckResult(passed, severity, "Response decay", detail,
                       "ISO 7626-5 secc. 7.3.3",
                       measured_value=ratio, threshold=decay_fraction)


def check_coherence_band(
    coherence: np.ndarray,
    frequencies_hz: np.ndarray,
    f_min_hz: float = 5.0,
    f_max_hz: float = 1000.0,
    threshold: float = 0.8,
    min_passing_fraction: float = 0.85,
) -> CheckResult:
    """
    ISO 7626-5 secc. 7.4 — Coherencia γ² ≥ threshold en la banda de interés.

    Permite que hasta (1 - min_passing_fraction) de la banda esté por debajo
    del threshold (típicamente en bins de modos donde por definición γ² baja).
    Si la fracción que pasa es menor a min_passing_fraction → FAIL.
    """
    f = np.asarray(frequencies_hz, dtype=float)
    g = np.asarray(coherence, dtype=float)
    mask = (f >= f_min_hz) & (f <= f_max_hz)
    if not np.any(mask):
        return CheckResult(False, "fail", "Coherencia γ²",
                           "Banda no cubierta", "ISO 7626-5 secc. 7.4")

    band = g[mask]
    passing_fraction = float(np.mean(band >= threshold))
    mean_g = float(band.mean())
    min_g = float(band.min())
    passed = passing_fraction >= min_passing_fraction
    severity = ("ok" if passed
                else ("warning" if passing_fraction >= 0.70 else "fail"))
    detail = (f"γ² promedio = {mean_g:.2f}, mínimo = {min_g:.2f}, "
              f"fracción de bins con γ² ≥ {threshold}: {passing_fraction*100:.1f}% "
              f"(mínimo {min_passing_fraction*100:.0f}%).")
    if not passed:
        detail += (" ⚠ Coherencia insuficiente — aumentar N° de impactos, "
                   "verificar masa montaje del sensor o reducir ruido ambiente.")
    return CheckResult(passed, severity, "Coherencia γ²", detail,
                       "ISO 7626-5 secc. 7.4",
                       measured_value=passing_fraction,
                       threshold=min_passing_fraction)


def check_n_averages(
    n_averages: int,
    minimum: int = 5,
) -> CheckResult:
    """
    ISO 7626-5 secc. 7.3.4 — Mínimo 5 promedios para reducir varianza estadística.
    """
    passed = n_averages >= minimum
    severity = "ok" if passed else ("warning" if n_averages >= 3 else "fail")
    detail = (f"Promedios = {n_averages} (mínimo recomendado {minimum}).")
    if not passed:
        detail += (" ⚠ Pocos promedios — la varianza de la FRF será alta y "
                   "puede ocultar modos cercanos a ruido.")
    return CheckResult(passed, severity, "N° de promedios", detail,
                       "ISO 7626-5 secc. 7.3.4",
                       measured_value=float(n_averages),
                       threshold=float(minimum))


# =====================================================================
# Compliance report builder
# =====================================================================

def build_compliance_report(
    input_signal: np.ndarray,
    output_signal: np.ndarray,
    coherence: np.ndarray,
    coherence_frequencies_hz: np.ndarray,
    sample_rate_hz: float,
    f_target_hz: float = 500.0,
    n_averages: int = 1,
    coherence_threshold: float = 0.8,
    test_setup_name: str = "",
) -> ComplianceReport:
    """
    Construye un reporte completo aplicando los 5 checks principales.

    Args:
        input_signal: Señal del martillo en el tiempo (en N o V crudos)
        output_signal: Señal del acelerómetro/sensor en el tiempo
        coherence: γ²(f) ya calculada
        coherence_frequencies_hz: Eje de frecuencia de la coherencia
        sample_rate_hz: Tasa de muestreo
        f_target_hz: Frecuencia objetivo del ensayo (banda alta de interés)
        n_averages: Número de impactos promediados
        coherence_threshold: γ² mínimo aceptable
        test_setup_name: Nombre del test setup para el reporte

    Returns:
        ComplianceReport con 5 checks
    """
    report = ComplianceReport(test_setup_name=test_setup_name)

    report.checks.append(check_single_peak_input(input_signal, sample_rate_hz))
    report.checks.append(check_input_spectrum_flat(
        input_signal, sample_rate_hz, f_target_hz=f_target_hz
    ))
    report.checks.append(check_response_decay(output_signal))
    report.checks.append(check_coherence_band(
        coherence, coherence_frequencies_hz,
        f_max_hz=f_target_hz, threshold=coherence_threshold,
    ))
    report.checks.append(check_n_averages(n_averages))

    return report
