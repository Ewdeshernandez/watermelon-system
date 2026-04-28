"""
core.spectrum_scale
===================

Helpers para auto-escalar el rango de frecuencia (eje X) del análisis
espectral según la **naturaleza física** de la cantidad medida:

  - Desplazamiento (mil pp, µm pp): el contenido relevante está en
    bajas frecuencias (1X-10X de la velocidad de giro). Para
    turbomaquinaria y rotores grandes con sondas de proximidad, el
    rango útil típico es hasta ~1000 Hz = 60.000 CPM.

  - Velocidad (in/s pk, mm/s pk, mm/s RMS): rango medio, captura tanto
    las componentes sincrónicas como armónicos altos asociados a
    holgura mecánica, soltura, defectos de engranajes en el rango
    medio. Tope típico ~2000 Hz = 120.000 CPM (cubre ISO 20816 y
    análisis general industrial).

  - Aceleración (g pk, g RMS): rango ALTO, necesario para detectar
    fallas tempranas de rodamientos (BPFO/BPFI/BSF/FTF) y resonancias
    de elementos rodantes que viven a alta frecuencia. Tope típico
    ~10.000 Hz = 600.000 CPM.

Estos defaults siguen prácticas Cat IV reales (lo que usa un ingeniero
rotodinámico senior cuando elige el rango de su FFT). El usuario
siempre puede overridear el valor manualmente desde la sidebar.
"""

from __future__ import annotations

from typing import Tuple, Optional


# CPM = Hz × 60. Los topes están definidos en CPM para integrar
# directo con el resto del módulo Spectrum que ya trabaja en CPM.

DEFAULT_MAX_CPM_DISPLACEMENT = 60_000.0   # ~1.000 Hz
DEFAULT_MAX_CPM_VELOCITY = 120_000.0      # ~2.000 Hz
DEFAULT_MAX_CPM_ACCELERATION = 600_000.0  # ~10.000 Hz
DEFAULT_MAX_CPM_FALLBACK = 60_000.0       # genérico si no se puede inferir


def classify_amplitude_quantity(unit_text: str) -> str:
    """
    Devuelve la familia física a partir del texto de unidad.

    Returns: "displacement" | "velocity" | "acceleration" | "unknown"

    Ejemplos:
      "mil pp"     → displacement
      "µm pp"      → displacement
      "um pk-pk"   → displacement
      "in/s pk"    → velocity
      "mm/s rms"   → velocity
      "g pk"       → acceleration
      "g rms"      → acceleration
    """
    if not unit_text:
        return "unknown"
    u = unit_text.strip().lower().replace("μ", "u").replace("µ", "u")

    # Aceleración (chequear primero — "g" es muy corto)
    if u.startswith("g") or " g " in f" {u} " or u.endswith(" g") or "g pk" in u or "g rms" in u:
        return "acceleration"

    # Velocidad (in/s, mm/s, m/s)
    if "in/s" in u or "mm/s" in u or "m/s" in u or "ips" in u:
        return "velocity"

    # Desplazamiento (mil, mils, um, micras, micrón)
    if "mil" in u or "um" in u or "micras" in u or "micron" in u:
        return "displacement"

    return "unknown"


def suggest_max_cpm_for_unit(
    unit_text: str,
    *,
    rpm: Optional[float] = None,
) -> Tuple[float, str]:
    """
    Devuelve un (max_cpm_sugerido, razon_textual) basado en la unidad
    de la amplitud y opcionalmente el RPM operativo.

    Si la unidad es desconocida, cae al fallback genérico (10× rpm si
    hay rpm, o 60.000 CPM).

    El segundo valor del tuple es una razón legible para mostrar al
    usuario en la UI ("displacement → 60.000 CPM" etc.) que aporta
    transparencia sobre qué decisión tomó el sistema.
    """
    family = classify_amplitude_quantity(unit_text)

    if family == "displacement":
        return DEFAULT_MAX_CPM_DISPLACEMENT, (
            "desplazamiento (sondas de proximidad) → 60.000 CPM (~1 kHz). "
            "Rango típico para análisis sincrónico 1X-10X de turbomaquinaria."
        )
    if family == "velocity":
        return DEFAULT_MAX_CPM_VELOCITY, (
            "velocidad → 120.000 CPM (~2 kHz). Rango ISO 20816 + análisis "
            "armónico para holgura mecánica y desalineación."
        )
    if family == "acceleration":
        return DEFAULT_MAX_CPM_ACCELERATION, (
            "aceleración → 600.000 CPM (~10 kHz). Rango alto para detección "
            "de fallas tempranas de rodamientos (BPFO/BPFI/BSF/FTF) y "
            "resonancias de elementos rodantes."
        )

    # Unknown: si hay RPM razonable, usar 10× operating rpm; sino fallback
    if rpm is not None and rpm > 0:
        suggested = max(rpm * 10.0, 1_000.0)
        return suggested, (
            f"unidad no clasificada → 10× velocidad operativa "
            f"({rpm:.0f} rpm × 10 = {suggested:.0f} CPM)."
        )

    return DEFAULT_MAX_CPM_FALLBACK, (
        f"unidad no clasificada y sin RPM → fallback genérico "
        f"{DEFAULT_MAX_CPM_FALLBACK:.0f} CPM."
    )


__all__ = [
    "classify_amplitude_quantity",
    "suggest_max_cpm_for_unit",
    "DEFAULT_MAX_CPM_DISPLACEMENT",
    "DEFAULT_MAX_CPM_VELOCITY",
    "DEFAULT_MAX_CPM_ACCELERATION",
    "DEFAULT_MAX_CPM_FALLBACK",
]
