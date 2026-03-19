import re
from typing import Any, Dict, Optional

import numpy as np


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None:
            return default
        if isinstance(value, (int, float, np.integer, np.floating)):
            return float(value)
        text = str(value).strip()
        if text == "":
            return default
        match = re.search(r"[-+]?\d*\.?\d+", text.replace(",", ""))
        if match:
            return float(match.group())
        return default
    except Exception:
        return default


def _wrap_360(angle_deg):
    return np.mod(angle_deg, 360.0)


def _circular_mean_deg(angles_deg: np.ndarray) -> float:
    if len(angles_deg) == 0:
        return 0.0
    z = np.mean(np.exp(1j * np.radians(angles_deg)))
    return float(_wrap_360(np.degrees(np.angle(z))))


def _circular_std_deg(angles_deg: np.ndarray) -> float:
    if len(angles_deg) == 0:
        return 0.0
    z = np.mean(np.exp(1j * np.radians(angles_deg)))
    r = float(np.abs(z))
    r = float(np.clip(r, 1e-12, 1.0))
    return float(np.degrees(np.sqrt(-2.0 * np.log(r))))


def _median_dt_ms(time_array: np.ndarray) -> Optional[float]:
    if time_array is None or len(time_array) < 2:
        return None
    diffs = np.diff(np.asarray(time_array, dtype=float))
    diffs = diffs[np.isfinite(diffs)]
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return None
    return float(np.median(diffs))


def _extract_rpm(metadata: Dict[str, Any]) -> Optional[float]:
    keys = [
        "Sample Speed",
        "sample_speed",
        "RPM",
        "rpm",
        "speed_rpm",
        "machine_speed_rpm",
    ]
    for key in keys:
        if key in metadata:
            rpm = _safe_float(metadata.get(key))
            if rpm is not None and rpm > 0:
                return float(rpm)
    return None


def _extract_number_of_revs(metadata: Dict[str, Any]) -> Optional[int]:
    keys = [
        "Number of Revs",
        "number_of_revs",
        "n_revs",
        "revs",
        "revolutions",
    ]
    for key in keys:
        if key in metadata:
            value = _safe_float(metadata.get(key))
            if value is not None and value > 0:
                return int(round(value))
    return None


def _extract_variable_hint(metadata: Dict[str, Any]) -> Dict[str, Optional[int]]:
    variable = str(metadata.get("Variable", ""))
    match = re.search(
        r"\((\d+)\s*X\s*/\s*(\d+)\s*revs\)",
        variable,
        flags=re.IGNORECASE,
    )
    if not match:
        return {"hint_samples_per_rev": None, "hint_revs": None}
    return {
        "hint_samples_per_rev": int(match.group(1)),
        "hint_revs": int(match.group(2)),
    }


def _fit_1x_per_revolution(x_rev: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Ajuste robusto por revolución:
        x(theta) ≈ a*cos(theta) + b*sin(theta) + c

    Amplitud 1X (PP) = 2 * sqrt(a^2 + b^2)

    Convención de fase usada:
        phase = atan2(-Im(C1), Re(C1)) envuelta a 0..360
    equivalente aquí a:
        phase = atan2(b, a)
    """
    n_revs, samples_per_rev = x_rev.shape

    theta = 2.0 * np.pi * np.arange(samples_per_rev) / samples_per_rev
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    ones = np.ones(samples_per_rev)

    design = np.column_stack([cos_t, sin_t, ones])
    pinv = np.linalg.pinv(design)

    coeffs = (pinv @ x_rev.T).T
    a = coeffs[:, 0]
    b = coeffs[:, 1]

    amp_pk = np.sqrt(a * a + b * b)
    amp_pp = 2.0 * amp_pk
    phase_deg = _wrap_360(np.degrees(np.arctan2(b, a)))

    # Complejo equivalente a la componente 1X
    c1 = a - 1j * b

    return {
        "amp_pp_per_rev": amp_pp,
        "phase_per_rev_deg": phase_deg,
        "complex_1x_per_rev": c1,
        "a_coef": a,
        "b_coef": b,
    }


def _score_candidate_geometry(x: np.ndarray, n_revs: int) -> Optional[Dict[str, Any]]:
    """
    Score físico:
    - coherencia circular de fase
    - amplitud 1X consistente
    - fuerza de la componente 1X
    """
    n_samples = len(x)
    if n_revs is None or n_revs <= 0:
        return None

    spr_float = n_samples / n_revs
    if abs(spr_float - round(spr_float)) > 0.05:
        return None

    spr = int(round(spr_float))
    usable = spr * n_revs
    if usable <= 0 or usable > n_samples:
        return None

    x_use = x[:usable]
    x_rev = x_use.reshape(n_revs, spr)

    comp = _fit_1x_per_revolution(x_rev)

    amp_pp = comp["amp_pp_per_rev"]
    phase_deg = comp["phase_per_rev_deg"]

    amp_mean = float(np.mean(amp_pp))
    phase_mean = float(_circular_mean_deg(phase_deg))
    phase_std = float(_circular_std_deg(phase_deg))
    amp_cv = float(np.std(amp_pp) / max(abs(amp_mean), 1e-12))

    score = (250.0 - phase_std) + (12.0 * amp_mean) - (30.0 * amp_cv)

    return {
        "n_revs": int(n_revs),
        "samples_per_rev": int(spr),
        "usable_samples": int(usable),
        "score": float(score),
        "amp_mean": amp_mean,
        "phase_mean": phase_mean,
        "phase_std": phase_std,
        "amp_cv": amp_cv,
        "amp_pp_per_rev": amp_pp,
        "phase_per_rev_deg": phase_deg,
        "complex_1x_per_rev": comp["complex_1x_per_rev"],
    }


def _infer_geometry(signal) -> Dict[str, Any]:
    """
    Sincronización por geometría:
    1) probar candidatos físicos
    2) escoger el de mejor coherencia de fase y amplitud
    3) usar header/time solo como evidencia auxiliar
    """
    x = np.asarray(signal.x, dtype=float)
    metadata = getattr(signal, "metadata", {}) or {}
    time_array = getattr(signal, "time", None)

    rpm = _extract_rpm(metadata)
    number_of_revs = _extract_number_of_revs(metadata)
    hints = _extract_variable_hint(metadata)
    dt_ms = _median_dt_ms(np.asarray(time_array, dtype=float)) if time_array is not None else None

    inferred_revs_from_time = None
    if dt_ms is not None and rpm is not None and rpm > 0 and len(x) > 1:
        total_time_s = (dt_ms * (len(x) - 1)) / 1000.0
        inferred_revs_from_time = int(round(total_time_s * rpm / 60.0))

    candidates = []

    for candidate_revs in [
        number_of_revs,
        inferred_revs_from_time,
        hints["hint_revs"],
        8,
        16,
        32,
        64,
    ]:
        result = _score_candidate_geometry(x, candidate_revs)
        if result is not None:
            candidates.append(result)

    if not candidates:
        return {
            "is_synchronous": False,
            "samples_per_rev": None,
            "n_revs": None,
            "usable_samples": len(x),
            "rpm": rpm,
            "dt_ms": dt_ms,
            "source": "fallback_exact_frequency",
            "header_number_of_revs": number_of_revs,
            "inferred_revs_from_time": inferred_revs_from_time,
            "variable_hint_samples_per_rev": hints["hint_samples_per_rev"],
            "variable_hint_revs": hints["hint_revs"],
            "candidate_table": [],
        }

    best = max(candidates, key=lambda item: item["score"])

    candidate_table = [
        {
            "n_revs": c["n_revs"],
            "samples_per_rev": c["samples_per_rev"],
            "score": round(c["score"], 6),
            "amp_mean": round(c["amp_mean"], 6),
            "phase_mean": round(c["phase_mean"], 6),
            "phase_std": round(c["phase_std"], 6),
            "amp_cv": round(c["amp_cv"], 6),
        }
        for c in sorted(candidates, key=lambda item: item["score"], reverse=True)
    ]

    return {
        "is_synchronous": True,
        "samples_per_rev": best["samples_per_rev"],
        "n_revs": best["n_revs"],
        "usable_samples": best["usable_samples"],
        "rpm": rpm,
        "dt_ms": dt_ms,
        "source": "best_physical_candidate",
        "header_number_of_revs": number_of_revs,
        "inferred_revs_from_time": inferred_revs_from_time,
        "variable_hint_samples_per_rev": hints["hint_samples_per_rev"],
        "variable_hint_revs": hints["hint_revs"],
        "candidate_table": candidate_table,
        "precomputed_amp_pp_per_rev": best["amp_pp_per_rev"],
        "precomputed_phase_per_rev_deg": best["phase_per_rev_deg"],
        "precomputed_complex_1x_per_rev": best["complex_1x_per_rev"],
        "precomputed_amp_mean": best["amp_mean"],
        "precomputed_phase_mean": best["phase_mean"],
        "precomputed_phase_std": best["phase_std"],
    }


def _compute_fallback_exact_frequency(signal) -> Dict[str, Any]:
    """
    Fallback sin pseudo:
    ajuste global directo a cos/sin usando RPM del header
    """
    x = np.asarray(signal.x, dtype=float)
    t = np.asarray(signal.time, dtype=float)
    metadata = getattr(signal, "metadata", {}) or {}

    rpm = _extract_rpm(metadata)
    if rpm is None or len(t) != len(x) or len(x) < 2:
        return {
            "mode": "fallback_invalid",
            "amplitude_pp_per_rev": np.array([0.0]),
            "phase_per_rev_deg": np.array([0.0]),
            "complex_1x_per_rev": np.array([0.0 + 0.0j]),
            "mean_amplitude_pp": 0.0,
            "mean_phase_deg": 0.0,
            "phase_stability_deg": 0.0,
            "n_revs": 1,
            "samples_per_rev": len(x),
            "mean_rpm": 0.0,
            "debug": {
                "sync_source": "fallback_invalid",
                "phase_convention": "phase = atan2(b, a) on fitted 1X",
                "candidate_table": [],
            },
        }

    x0 = x - np.mean(x)
    t0_ms = t - t[0]
    omega = 2.0 * np.pi * (rpm / 60.0)
    theta = omega * (t0_ms / 1000.0)

    design = np.column_stack([np.cos(theta), np.sin(theta), np.ones(len(theta))])
    coeffs, _, _, _ = np.linalg.lstsq(design, x0, rcond=None)
    a = float(coeffs[0])
    b = float(coeffs[1])

    amp_pp = 2.0 * float(np.sqrt(a * a + b * b))
    phase_deg = float(_wrap_360(np.degrees(np.arctan2(b, a))))
    c1 = np.array([a - 1j * b], dtype=np.complex128)

    return {
        "mode": "exact_frequency_fit",
        "amplitude_pp_per_rev": np.array([amp_pp]),
        "phase_per_rev_deg": np.array([phase_deg]),
        "complex_1x_per_rev": c1,
        "mean_amplitude_pp": amp_pp,
        "mean_phase_deg": phase_deg,
        "phase_stability_deg": 0.0,
        "n_revs": 1,
        "samples_per_rev": len(x),
        "mean_rpm": float(rpm),
        "debug": {
            "sync_source": "exact_frequency_fit",
            "phase_convention": "phase = atan2(b, a) on fitted 1X",
            "candidate_table": [],
        },
    }


def analyze_phase(signal) -> Dict[str, Any]:
    geom = _infer_geometry(signal)

    if not geom["is_synchronous"]:
        return _compute_fallback_exact_frequency(signal)

    amp_pp_per_rev = np.asarray(geom["precomputed_amp_pp_per_rev"], dtype=float)
    phase_per_rev_deg = np.asarray(geom["precomputed_phase_per_rev_deg"], dtype=float)
    complex_1x_per_rev = np.asarray(geom["precomputed_complex_1x_per_rev"], dtype=np.complex128)

    mean_amp_pp = float(geom["precomputed_amp_mean"])
    mean_phase_deg = float(geom["precomputed_phase_mean"])
    phase_stability_deg = float(geom["precomputed_phase_std"])

    samples_per_rev = int(geom["samples_per_rev"])
    n_revs = int(geom["n_revs"])

    return {
        "mode": "geometry_sync_1x_fit",
        "amplitude_pp_per_rev": amp_pp_per_rev,
        "phase_per_rev_deg": phase_per_rev_deg,
        "complex_1x_per_rev": complex_1x_per_rev,
        "mean_amplitude_pp": mean_amp_pp,
        "mean_phase_deg": mean_phase_deg,
        "phase_stability_deg": phase_stability_deg,
        "n_revs": n_revs,
        "samples_per_rev": samples_per_rev,
        "mean_rpm": float(geom["rpm"]) if geom["rpm"] is not None else 0.0,
        "debug": {
            "sync_source": geom["source"],
            "header_number_of_revs": geom["header_number_of_revs"],
            "inferred_revs_from_time": geom["inferred_revs_from_time"],
            "variable_hint_samples_per_rev": geom["variable_hint_samples_per_rev"],
            "variable_hint_revs": geom["variable_hint_revs"],
            "phase_convention": "phase = atan2(b, a) on fitted 1X",
            "candidate_table": geom["candidate_table"],
        },
    }