from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


BPFO_COLOR = "#ef4444"
BPFI_COLOR = "#f59e0b"
BSF_COLOR = "#8b5cf6"
FTF_COLOR = "#10b981"

FAULT_COLORS = {
    "BPFO": BPFO_COLOR,
    "BPFI": BPFI_COLOR,
    "BSF": BSF_COLOR,
    "FTF": FTF_COLOR,
}

BEARING_CATALOG: Dict[str, Dict[str, Any]] = {
    "SKF 6319": {
        "manufacturer": "SKF",
        "model": "6319",
        "display_name": "SKF 6319",
        "factors": {"BPFO": 3.0960, "BPFI": 4.9040, "BSF": 4.1980, "FTF": 0.3870},
    },
    "SKF 6319/C3": {
        "manufacturer": "SKF",
        "model": "6319/C3",
        "display_name": "SKF 6319/C3",
        "factors": {"BPFO": 3.0960, "BPFI": 4.9040, "BSF": 4.1980, "FTF": 0.3870},
    },
}


def list_bearing_catalog_options() -> List[str]:
    return sorted(BEARING_CATALOG.keys())


def format_number(value: Any, digits: int = 3, fallback: str = "—") -> str:
    if value is None:
        return fallback
    try:
        val = float(value)
        if not math.isfinite(val):
            return fallback
        return f"{val:.{digits}f}"
    except Exception:
        return fallback


def get_catalog_entry(selected_name: str) -> Optional[Dict[str, Any]]:
    name = str(selected_name or "").strip()
    if not name:
        return None
    return BEARING_CATALOG.get(name)


def build_bearing_fault_overlay(
    selected_name: str,
    rpm: Optional[float],
    harmonic_count: int = 3,
) -> Dict[str, Any]:
    entry = get_catalog_entry(selected_name)
    model_display = str(selected_name or "").strip()

    if rpm is None or float(rpm) <= 0:
        return {
            "available": False,
            "model_display": model_display or "—",
            "families": [],
            "lines": [],
            "message": "No se pudo calcular frecuencias de falla de rodamiento porque la señal no tiene RPM válido.",
        }

    if entry is None:
        return {
            "available": False,
            "model_display": model_display or "—",
            "families": [],
            "lines": [],
            "message": "No se encontró el rodamiento seleccionado en el catálogo interno.",
        }

    factors = dict(entry["factors"])
    rpm_value = float(rpm)
    harmonic_count = max(1, int(harmonic_count))

    families: List[Dict[str, Any]] = []
    flat_lines: List[Dict[str, Any]] = []

    for family in ["BPFO", "BPFI", "BSF", "FTF"]:
        factor = float(factors[family])
        base_freq_cpm = rpm_value * factor
        color = FAULT_COLORS[family]
        lines = []
        for harmonic in range(1, harmonic_count + 1):
            freq_cpm = base_freq_cpm * harmonic
            line = {
                "family": family,
                "harmonic": harmonic,
                "factor": factor,
                "base_freq_cpm": base_freq_cpm,
                "freq_cpm": freq_cpm,
                "label": f"{harmonic}x {family}",
                "color": color,
            }
            lines.append(line)

            flat_lines.append(line)

        families.append(
            {
                "family": family,
                "factor": factor,
                "base_freq_cpm": base_freq_cpm,
                "color": color,
                "lines": lines,
            }
        )

    return {
        "available": True,
        "model_display": entry.get("display_name", model_display or "—"),
        "families": families,
        "lines": flat_lines,
        "message": "",
    }


def _find_local_peak_near(
    freq_cpm: np.ndarray,
    amp_peak: np.ndarray,
    target_cpm: float,
    tolerance_pct: float,
) -> Tuple[Optional[float], Optional[float]]:
    if target_cpm <= 0 or freq_cpm.size == 0 or amp_peak.size == 0:
        return None, None

    tol = abs(target_cpm) * (float(tolerance_pct) / 100.0)
    tol = max(tol, 60.0)

    mask = np.isfinite(freq_cpm) & np.isfinite(amp_peak) & (freq_cpm >= (target_cpm - tol)) & (freq_cpm <= (target_cpm + tol))
    if not np.any(mask):
        return None, None

    idx_candidates = np.where(mask)[0]
    idx = int(idx_candidates[np.argmax(amp_peak[mask])])
    return float(freq_cpm[idx]), float(amp_peak[idx])


def build_bearing_fault_assessment(
    freq_cpm: np.ndarray,
    amp_peak: np.ndarray,
    overlay: Dict[str, Any],
    tolerance_pct: float = 3.0,
    min_relative_amp: float = 0.12,
) -> Dict[str, Any]:
    if not overlay.get("available", False):
        return {
            "matched_families": [],
            "unmatched_families": [],
            "narrative": overlay.get("message") or "No fue posible evaluar frecuencias de falla de rodamiento.",
        }

    if freq_cpm.size == 0 or amp_peak.size == 0:
        return {
            "matched_families": [],
            "unmatched_families": [fam["family"] for fam in overlay.get("families", [])],
            "narrative": (
                f"Se calcularon las frecuencias de falla del rodamiento {overlay.get('model_display', '—')}, "
                "pero no hay datos espectrales válidos para compararlas."
            ),
        }

    finite_amp = amp_peak[np.isfinite(amp_peak)]
    ref_amp = float(np.max(finite_amp)) if finite_amp.size else 0.0
    amplitude_threshold = ref_amp * max(0.0, float(min_relative_amp))

    matched_families: List[Dict[str, Any]] = []
    unmatched_families: List[str] = []

    for family_block in overlay.get("families", []):
        family = family_block["family"]
        hits: List[Dict[str, Any]] = []

        for line in family_block.get("lines", []):
            peak_freq, peak_amp_val = _find_local_peak_near(
                freq_cpm=freq_cpm,
                amp_peak=amp_peak,
                target_cpm=float(line["freq_cpm"]),
                tolerance_pct=tolerance_pct,
            )
            if peak_freq is None or peak_amp_val is None:
                continue
            if peak_amp_val < amplitude_threshold:
                continue

            deviation_pct = abs(peak_freq - float(line["freq_cpm"])) / max(abs(float(line["freq_cpm"])), 1e-9) * 100.0
            hits.append(
                {
                    "harmonic": int(line["harmonic"]),
                    "expected_cpm": float(line["freq_cpm"]),
                    "found_cpm": float(peak_freq),
                    "amp_peak": float(peak_amp_val),
                    "deviation_pct": float(deviation_pct),
                }
            )

        if len(hits) >= 1:
            matched_families.append(
                {
                    "family": family,
                    "factor": float(family_block["factor"]),
                    "base_freq_cpm": float(family_block["base_freq_cpm"]),
                    "hit_count": len(hits),
                    "harmonic_count": len(family_block.get("lines", [])),
                    "hits": hits,
                }
            )
        else:
            unmatched_families.append(family)

    model_display = overlay.get("model_display", "—")
    family_factors_text = ", ".join(
        f"{fam['family']}={format_number(fam['base_freq_cpm'], 1)} CPM"
        for fam in overlay.get("families", [])
    )

    if matched_families:
        matched_txt = "; ".join(
            f"{fam['family']} ({fam['hit_count']}/{fam['harmonic_count']} armónicos)"
            for fam in matched_families
        )
        if unmatched_families:
            unmatched_txt = ", ".join(unmatched_families)
            narrative = (
                f"Se calcularon las frecuencias características del rodamiento {model_display} "
                f"({family_factors_text}). Se observa coincidencia espectral con {matched_txt}. "
                f"No se observa coincidencia clara con {unmatched_txt}."
            )
        else:
            narrative = (
                f"Se calcularon las frecuencias características del rodamiento {model_display} "
                f"({family_factors_text}). Se observa coincidencia espectral con {matched_txt}."
            )
    else:
        narrative = (
            f"Se calcularon las frecuencias características del rodamiento {model_display} "
            f"({family_factors_text}). No se observa coincidencia clara con BPFO, BPFI, BSF ni FTF "
            f"dentro de una tolerancia de ±{format_number(tolerance_pct, 1)}%."
        )

    return {
        "matched_families": matched_families,
        "unmatched_families": unmatched_families,
        "narrative": narrative,
    }
