from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = PROJECT_ROOT / "data" / "bearing_catalog.csv"


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


def normalize_bearing_text(text: str) -> str:
    return "".join(ch for ch in str(text or "").upper() if ch.isalnum())


def load_bearing_catalog() -> pd.DataFrame:
    if not CATALOG_PATH.exists():
        return pd.DataFrame(
            columns=[
                "manufacturer", "model", "alias1", "alias2", "alias3",
                "bpfo_factor", "bpfi_factor", "bsf_factor", "ftf_factor",
            ]
        )
    df = pd.read_csv(CATALOG_PATH)
    expected = [
        "manufacturer", "model", "alias1", "alias2", "alias3",
        "bpfo_factor", "bpfi_factor", "bsf_factor", "ftf_factor",
    ]
    for col in expected:
        if col not in df.columns:
            df[col] = ""
    return df[expected].copy()


def list_bearing_catalog_options() -> List[str]:
    df = load_bearing_catalog()
    if df.empty:
        return []
    options: List[str] = []
    for _, row in df.iterrows():
        manufacturer = str(row.get("manufacturer", "") or "").strip()
        model = str(row.get("model", "") or "").strip()
        if manufacturer and model:
            options.append(f"{manufacturer} {model}")
        elif model:
            options.append(model)
    return sorted(dict.fromkeys(options))


def _row_display_name(row: pd.Series) -> str:
    manufacturer = str(row.get("manufacturer", "") or "").strip()
    model = str(row.get("model", "") or "").strip()
    if manufacturer and model:
        return f"{manufacturer} {model}"
    return model or "Unknown Bearing"


def find_bearing_catalog_entry(selected_name: str) -> Optional[Dict[str, Any]]:
    normalized = normalize_bearing_text(selected_name)
    if not normalized:
        return None

    df = load_bearing_catalog()
    if df.empty:
        return None

    for _, row in df.iterrows():
        aliases = [
            _row_display_name(row),
            str(row.get("model", "") or ""),
            str(row.get("alias1", "") or ""),
            str(row.get("alias2", "") or ""),
            str(row.get("alias3", "") or ""),
        ]
        alias_norms = {normalize_bearing_text(a) for a in aliases if str(a).strip()}
        if normalized in alias_norms:
            try:
                return {
                    "display_name": _row_display_name(row),
                    "manufacturer": str(row.get("manufacturer", "") or "").strip(),
                    "model": str(row.get("model", "") or "").strip(),
                    "bpfo_factor": float(row["bpfo_factor"]),
                    "bpfi_factor": float(row["bpfi_factor"]),
                    "bsf_factor": float(row["bsf_factor"]),
                    "ftf_factor": float(row["ftf_factor"]),
                }
            except Exception:
                return None

    return None


def build_bearing_fault_overlay_from_catalog(
    selected_name: str,
    rpm: Optional[float],
    harmonic_count: int = 3,
) -> Dict[str, Any]:
    entry = find_bearing_catalog_entry(selected_name)

    if rpm is None or float(rpm) <= 0:
        return {
            "available": False,
            "mode": "catalog",
            "model_display": str(selected_name or "").strip() or "—",
            "families": [],
            "lines": [],
            "message": "No se pudo calcular frecuencias de falla de rodamiento porque la señal no tiene RPM válido.",
        }

    if entry is None:
        return {
            "available": False,
            "mode": "catalog",
            "model_display": str(selected_name or "").strip() or "—",
            "families": [],
            "lines": [],
            "message": "No se encontró el rodamiento seleccionado en el catálogo interno.",
        }

    return _build_overlay_from_factors(
        model_display=entry["display_name"],
        rpm=float(rpm),
        harmonic_count=harmonic_count,
        factors={
            "BPFO": float(entry["bpfo_factor"]),
            "BPFI": float(entry["bpfi_factor"]),
            "BSF": float(entry["bsf_factor"]),
            "FTF": float(entry["ftf_factor"]),
        },
        mode="catalog",
    )


def build_bearing_fault_overlay_from_nb(
    nb: int,
    rpm: Optional[float],
    harmonic_count: int = 3,
) -> Dict[str, Any]:
    if rpm is None or float(rpm) <= 0:
        return {
            "available": False,
            "mode": "approximate",
            "model_display": f"Nb={nb}",
            "families": [],
            "lines": [],
            "message": "No se pudo calcular frecuencias de falla aproximadas porque la señal no tiene RPM válido.",
        }

    nb = int(nb)
    if nb <= 0:
        return {
            "available": False,
            "mode": "approximate",
            "model_display": f"Nb={nb}",
            "families": [],
            "lines": [],
            "message": "El número de elementos rodantes debe ser mayor que cero.",
        }

    factors = {
        "BPFI": (nb / 2.0) + 1.2,
        "BPFO": (nb / 2.0) - 1.2,
        "BSF": 0.5 * ((nb / 2.0) - (1.2 / nb)),
        "FTF": 0.5 - (1.2 / nb),
    }

    return _build_overlay_from_factors(
        model_display=f"Aprox. Nb={nb}",
        rpm=float(rpm),
        harmonic_count=harmonic_count,
        factors=factors,
        mode="approximate",
    )


def _build_overlay_from_factors(
    model_display: str,
    rpm: float,
    harmonic_count: int,
    factors: Dict[str, float],
    mode: str,
) -> Dict[str, Any]:
    harmonic_count = max(1, int(harmonic_count))
    families: List[Dict[str, Any]] = []
    flat_lines: List[Dict[str, Any]] = []

    for family in ["BPFO", "BPFI", "BSF", "FTF"]:
        factor = float(factors[family])
        base_freq_cpm = rpm * factor
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
        "mode": mode,
        "model_display": model_display,
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
                f"Se calcularon las frecuencias características de rodamiento para {model_display} "
                f"({family_factors_text}). Se observa coincidencia espectral con {matched_txt}. "
                f"No se observa coincidencia clara con {unmatched_txt}."
            )
        else:
            narrative = (
                f"Se calcularon las frecuencias características de rodamiento para {model_display} "
                f"({family_factors_text}). Se observa coincidencia espectral con {matched_txt}."
            )
    else:
        narrative = (
            f"Se calcularon las frecuencias características de rodamiento para {model_display} "
            f"({family_factors_text}). No se observa coincidencia clara con BPFO, BPFI, BSF ni FTF "
            f"dentro de una tolerancia de ±{format_number(tolerance_pct, 1)}%."
        )

    return {
        "matched_families": matched_families,
        "unmatched_families": unmatched_families,
        "narrative": narrative,
    }


# ----------------------------------------------------------------------
# AI DIAGNOSIS (Industrial Level)
# ----------------------------------------------------------------------
def build_bearing_fault_ai_diagnosis(assessment: Dict[str, Any]) -> Dict[str, Any]:
    matched = assessment.get("matched_families", [])
    if not matched:
        return {
            "fault_type": "No clear bearing fault",
            "severity": "Normal",
            "confidence": 0.0,
            "message": "No se identifican patrones claros de falla de rodamiento."
        }

    best = sorted(matched, key=lambda x: (x["hit_count"], sum(h["amp_peak"] for h in x["hits"])), reverse=True)[0]

    family = best["family"]
    hit_count = best["hit_count"]
    harmonic_count = best["harmonic_count"]

    total_amp = sum(h["amp_peak"] for h in best["hits"])
    avg_amp = total_amp / max(len(best["hits"]), 1)

    ratio = hit_count / max(harmonic_count, 1)

    if ratio >= 0.7 and hit_count >= 3:
        severity = "Severa"
    elif ratio >= 0.4:
        severity = "Moderada"
    else:
        severity = "Incipiente"

    fault_map = {
        "BPFO": "Outer race defect",
        "BPFI": "Inner race defect",
        "BSF": "Rolling element defect",
        "FTF": "Cage defect",
    }

    fault_type = fault_map.get(family, family)

    confidence = min(1.0, 0.4 + ratio + (avg_amp / (avg_amp + 1e-6)) * 0.2)

    message = (
        f"{fault_type} con severidad {severity}. "
        f"Se detectaron {hit_count} armónicos relevantes de {family}, "
        f"indicando patrón consistente de falla."
    )

    return {
        "fault_type": fault_type,
        "severity": severity,
        "confidence": confidence,
        "message": message,
    }
