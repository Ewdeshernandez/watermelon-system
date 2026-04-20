from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import numpy as np


SAFE_COLOR = "#16a34a"
WARNING_COLOR = "#f59e0b"
DANGER_COLOR = "#dc2626"
NEUTRAL_COLOR = "#2563eb"


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


def get_semaforo_status(
    value: float,
    safe_limit: float = 60.0,
    warning_limit: float = 85.0,
) -> Tuple[str, str]:
    value = float(value)
    if value < safe_limit:
        return "SAFE", SAFE_COLOR
    if value < warning_limit:
        return "WARNING", WARNING_COLOR
    return "DANGER", DANGER_COLOR


def remaining_margin_pct(util_pct: float) -> float:
    return max(0.0, 100.0 - float(util_pct))


def boundary_utilization_pct(
    px: float,
    py: float,
    center_x: float,
    center_y: float,
    clearance_x: float,
    clearance_y: float,
) -> float:
    nx = (float(px) - float(center_x)) / max(float(clearance_x), 1e-9)
    ny = (float(py) - float(center_y)) / max(float(clearance_y), 1e-9)
    return math.sqrt(nx * nx + ny * ny) * 100.0


def build_clearance_diagnostics(
    x: np.ndarray,
    y: np.ndarray,
    center_x: float,
    center_y: float,
    clearance_x: float,
    clearance_y: float,
    safe_limit: float = 60.0,
    warning_limit: float = 85.0,
) -> Dict[str, Any]:
    if len(x) == 0 or len(y) == 0:
        status, color = get_semaforo_status(0.0, safe_limit, warning_limit)
        return {
            "util_max": 0.0,
            "util_min": 0.0,
            "util_mean": 0.0,
            "margin_min": 100.0,
            "status": status,
            "color": color,
        }

    utils = np.array(
        [
            boundary_utilization_pct(px, py, center_x, center_y, clearance_x, clearance_y)
            for px, py in zip(x, y)
        ],
        dtype=float,
    )

    util_max = float(np.max(utils))
    util_min = float(np.min(utils))
    util_mean = float(np.mean(utils))
    margin_min = remaining_margin_pct(util_max)
    status, color = get_semaforo_status(util_max, safe_limit, warning_limit)

    return {
        "util_max": util_max,
        "util_min": util_min,
        "util_mean": util_mean,
        "margin_min": margin_min,
        "status": status,
        "color": color,
    }


def detect_early_rub(
    x: np.ndarray,
    y: np.ndarray,
    speed: np.ndarray,
    center_x: float,
    center_y: float,
    clearance_x: float,
    clearance_y: float,
    warning_util_pct: float = 80.0,
    danger_util_pct: float = 95.0,
) -> Dict[str, Any]:
    if len(x) < 3:
        return {
            "triggered": False,
            "severity": "SAFE",
            "color": SAFE_COLOR,
            "message": "Insufficient points",
            "max_util_pct": 0.0,
            "contact_points": 0,
            "warning_points": 0,
            "trend_score": 0.0,
            "first_warning_speed": None,
            "first_danger_speed": None,
        }

    utils = np.array(
        [
            boundary_utilization_pct(px, py, center_x, center_y, clearance_x, clearance_y)
            for px, py in zip(x, y)
        ],
        dtype=float,
    )

    warning_mask = utils >= float(warning_util_pct)
    danger_mask = utils >= float(danger_util_pct)

    warning_points = int(np.sum(warning_mask))
    contact_points = int(np.sum(danger_mask))

    if len(utils) >= 5:
        idx = np.arange(len(utils), dtype=float)
        slope = float(np.polyfit(idx, utils, 1)[0])
    else:
        slope = 0.0

    last_n = min(8, len(utils))
    tail_mean = float(np.mean(utils[-last_n:])) if last_n else 0.0
    max_util = float(np.max(utils)) if len(utils) else 0.0

    if contact_points > 0 or max_util >= float(danger_util_pct):
        severity = "DANGER"
        color = DANGER_COLOR
        triggered = True
        message = "Early rub risk high"
    elif warning_points >= 2 or tail_mean >= float(warning_util_pct) or slope > 1.5:
        severity = "WARNING"
        color = WARNING_COLOR
        triggered = True
        message = "Possible early rub tendency"
    else:
        severity = "SAFE"
        color = SAFE_COLOR
        triggered = False
        message = "No early rub tendency detected"

    first_warning_speed = None
    first_danger_speed = None

    if np.any(warning_mask):
        first_warning_speed = float(speed[np.argmax(warning_mask)])
    if np.any(danger_mask):
        first_danger_speed = float(speed[np.argmax(danger_mask)])

    return {
        "triggered": triggered,
        "severity": severity,
        "color": color,
        "message": message,
        "max_util_pct": max_util,
        "contact_points": contact_points,
        "warning_points": warning_points,
        "trend_score": slope,
        "first_warning_speed": first_warning_speed,
        "first_danger_speed": first_danger_speed,
    }


# ============================================================
# TEXTUAL DIAGNOSTICS
# ============================================================
def build_polar_text_diagnostics(
    *,
    status: str,
    critical_speeds: List[Dict[str, float]],
    max_amp: float,
) -> Dict[str, str]:
    candidate_count = len(critical_speeds)

    if candidate_count == 0:
        headline = "No clear critical speed candidate detected"
        detail = (
            "The polar response does not show a strong peak-phase combination in the current speed range. "
            "Behavior looks relatively stable for this run."
        )
        action = "Continue monitoring during future run-up or coast-down events."
        return {"headline": headline, "detail": detail, "action": action}

    cs1 = critical_speeds[0]
    cs1_speed = int(round(float(cs1["speed"])))
    cs1_amp = float(cs1["amp"])
    cs1_phase = abs(float(cs1["phase_delta"]))

    if status == "SAFE":
        headline = f"Critical speed candidate detected near {cs1_speed} rpm, but response is still controlled"
        detail = (
            f"A candidate appears near {cs1_speed} rpm with amplitude around {cs1_amp:.3f} and "
            f"phase change of {cs1_phase:.1f}°. The response is present but not yet severe."
        )
        action = "Track the same zone in the next startup and compare growth trend."
        return {"headline": headline, "detail": detail, "action": action}

    if status == "WARNING":
        headline = f"Possible proximity to critical speed near {cs1_speed} rpm"
        detail = (
            f"The polar path shows a relevant response near {cs1_speed} rpm with amplitude around {cs1_amp:.3f} "
            f"and phase shift of {cs1_phase:.1f}°. This suggests dynamic amplification, but not a fully developed severe resonance."
        )
        action = "Monitor behavior during ramp-up, compare with Bode phase/amplitude, and verify if the peak repeats consistently."
        return {"headline": headline, "detail": detail, "action": action}

    headline = f"Strong critical speed behavior near {cs1_speed} rpm"
    detail = (
        f"The polar path shows a dominant response near {cs1_speed} rpm with amplitude around {cs1_amp:.3f} "
        f"and phase shift of {cs1_phase:.1f}°. This is consistent with a highly significant dynamic event."
    )
    action = "Review operating avoidance, confirm with Bode/orbit behavior, and evaluate mechanical margin before repeated runs."
    return {"headline": headline, "detail": detail, "action": action}


def build_shaft_text_diagnostics(
    *,
    status: str,
    util_max: float,
    margin_min: float,
    first_warning_speed: float | None = None,
    first_danger_speed: float | None = None,
) -> Dict[str, str]:
    if status == "SAFE":
        headline = "Shaft position is operating with healthy clearance margin"
        detail = (
            f"Maximum clearance utilization is {util_max:.1f}% and minimum remaining margin is {margin_min:.1f}%. "
            "The shaft centerline remains well inside the configured boundary."
        )
        action = "Keep trending clearance utilization and compare future runs for drift."
        return {"headline": headline, "detail": detail, "action": action}

    if status == "WARNING":
        speed_text = ""
        if first_warning_speed is not None:
            speed_text = f" First warning tendency appears near {first_warning_speed:.0f} rpm."
        headline = "Shaft position is approaching the configured clearance boundary"
        detail = (
            f"Maximum clearance utilization is {util_max:.1f}% and minimum remaining margin is {margin_min:.1f}%."
            f"{speed_text} This suggests reduced dynamic margin or possible bearing condition change."
        )
        action = "Review boundary definition, compare with past centerline plots, and verify if the trend is repeatable."
        return {"headline": headline, "detail": detail, "action": action}

    speed_text = ""
    if first_danger_speed is not None:
        speed_text = f" Boundary overutilization tendency begins near {first_danger_speed:.0f} rpm."
    headline = "Shaft position is extremely close to the configured clearance limit"
    detail = (
        f"Maximum clearance utilization is {util_max:.1f}% and minimum remaining margin is {margin_min:.1f}%."
        f"{speed_text} The shaft centerline is operating with very low geometric margin."
    )
    action = "Treat as high priority: verify bearing clearance assumptions, inspect machine condition, and avoid repeated operation until reviewed."
    return {"headline": headline, "detail": detail, "action": action}
