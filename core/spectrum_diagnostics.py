from __future__ import annotations

from typing import Dict, List, Optional

SAFE_COLOR = "#16a34a"
WARNING_COLOR = "#f59e0b"
DANGER_COLOR = "#dc2626"


def build_spectrum_report_notes(text_diag: Dict[str, str]) -> str:
    headline = str(text_diag.get("headline", "") or "").strip()
    detail = str(text_diag.get("detail", "") or "").strip()
    action = str(text_diag.get("action", "") or "").strip()

    blocks = []
    if headline:
        blocks.append(f"Resumen diagnóstico: {headline}")
    if detail:
        blocks.append(f"Detalle: {detail}")
    if action:
        blocks.append(f"Acción recomendada: {action}")
    return "\n\n".join(blocks).strip()


def _amp_of_order(harmonics: List[dict], order: int) -> float:
    for h in harmonics:
        if int(h.get("order", -1)) == int(order):
            try:
                return float(h.get("amp_peak", 0.0))
            except Exception:
                return 0.0
    return 0.0


def _max_order_from(harmonics: List[dict], min_order: int) -> float:
    vals = []
    for h in harmonics:
        try:
            order = int(h.get("order", -1))
            amp = float(h.get("amp_peak", 0.0))
            if order >= min_order:
                vals.append(amp)
        except Exception:
            pass
    return max(vals) if vals else 0.0


def _harmonic_count_above(harmonics: List[dict], threshold: float) -> int:
    c = 0
    for h in harmonics:
        try:
            if float(h.get("amp_peak", 0.0)) >= threshold:
                c += 1
        except Exception:
            pass
    return c


def evaluate_spectrum_diagnostic(
    *,
    one_x_amp: Optional[float],
    harmonics: List[dict],
    overall_spec_rms: Optional[float],
    dominant_peak_freq_cpm: Optional[float],
    dominant_peak_amp: Optional[float],
    rpm: Optional[float],
) -> Dict[str, str]:
    one_x = float(one_x_amp) if one_x_amp is not None else 0.0
    overall = float(overall_spec_rms) if overall_spec_rms is not None else 0.0
    dom_freq = float(dominant_peak_freq_cpm) if dominant_peak_freq_cpm is not None else 0.0
    dom_amp = float(dominant_peak_amp) if dominant_peak_amp is not None else 0.0

    two_x = _amp_of_order(harmonics, 2)
    three_x = _amp_of_order(harmonics, 3)
    high_harm = _max_order_from(harmonics, 4)

    ref = max(one_x, dom_amp, 1e-9)
    ratio_2x = two_x / ref
    ratio_3x = three_x / ref
    ratio_high = high_harm / ref

    significant_threshold = ref * 0.25
    strong_harmonic_count = _harmonic_count_above(harmonics, significant_threshold)

    near_1x = False
    if rpm is not None and rpm > 0 and dom_freq > 0:
        near_1x = abs(dom_freq - float(rpm)) <= max(0.08 * float(rpm), 60.0)

    status = "SAFE"
    color = SAFE_COLOR

    # 1) Mechanical looseness / harmonic-rich spectrum
    if strong_harmonic_count >= 4 and ratio_high >= 0.20:
        status = "DANGER"
        color = DANGER_COLOR
        return {
            "status": status,
            "color": color,
            "headline": "Mechanical looseness pattern suspected",
            "detail": (
                "Multiple harmonic components are present with significant high-order content. "
                "This spectrum shape is consistent with looseness or structural non-linearity."
            ),
            "action": (
                "Inspect support rigidity, hold-down condition, pedestal / base looseness, "
                "and compare with time waveform for impact or modulation behavior."
            ),
        }

    # 2) Misalignment tendency
    if ratio_2x >= 0.35 or (ratio_2x >= 0.25 and ratio_3x >= 0.18):
        status = "WARNING"
        color = WARNING_COLOR
        return {
            "status": status,
            "color": color,
            "headline": "Misalignment tendency suspected",
            "detail": (
                "The 2X component is significant relative to the synchronous component, "
                "with possible contribution from higher harmonics. This may indicate coupling "
                "or shaft train misalignment."
            ),
            "action": (
                "Review alignment condition, verify coupling behavior, and correlate with axial / radial "
                "measurements and phase behavior."
            ),
        }

    # 3) Unbalance tendency
    if one_x > 0 and near_1x and ratio_2x < 0.25 and ratio_high < 0.15:
        status = "WARNING"
        color = WARNING_COLOR
        return {
            "status": status,
            "color": color,
            "headline": "Synchronous unbalance signature suspected",
            "detail": (
                "The spectrum is dominated by a strong 1X component near running speed with low relative "
                "harmonic content. This is consistent with unbalance-type behavior."
            ),
            "action": (
                "Verify balance condition, review phase consistency across startups, and correlate with "
                "Polar / Bode response before corrective action."
            ),
        }

    # 4) Broadband / distributed energy
    if overall > 0 and dom_amp > 0 and (overall / max(dom_amp, 1e-9)) > 0.65 and strong_harmonic_count <= 2:
        status = "WARNING"
        color = WARNING_COLOR
        return {
            "status": status,
            "color": color,
            "headline": "Broadband energy pattern observed",
            "detail": (
                "Overall spectral energy is elevated relative to the dominant discrete peak, "
                "suggesting distributed broadband content rather than a purely synchronous signature."
            ),
            "action": (
                "Review process condition, flow-induced excitation, cavitation / turbulence possibility, "
                "or non-stationary excitation sources."
            ),
        }

    # 5) Default
    return {
        "status": status,
        "color": color,
        "headline": "No dominant abnormal spectral pattern identified",
        "detail": (
            "The current spectrum does not show a strong harmonic pattern clearly associated with "
            "unbalance, misalignment, or looseness."
        ),
        "action": (
            "Continue trending the spectrum and compare with waveform, Bode, Polar, and operating condition "
            "before concluding a machine fault mechanism."
        ),
    }
