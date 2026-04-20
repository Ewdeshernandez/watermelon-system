from __future__ import annotations

from typing import Any, Dict, List, Optional


def _status_rank(status: str) -> int:
    mapping = {"SAFE": 0, "WARNING": 1, "DANGER": 2}
    return mapping.get(str(status or "").upper(), 0)


def _bearing_severity_rank(severity: str) -> int:
    mapping = {"Normal": 0, "Incipiente": 1, "Moderada": 2, "Severa": 3}
    return mapping.get(str(severity or "").strip(), 0)


def _global_severity_name(level: int) -> str:
    mapping = {0: "Normal", 1: "Alerta", 2: "Moderada", 3: "Severa"}
    return mapping.get(int(level), "Alerta")


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _unique_texts(values: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for v in values:
        txt = _safe_text(v)
        key = txt.lower()
        if txt and key not in seen:
            out.append(txt)
            seen.add(key)
    return out


def build_unified_spectrum_ai_diagnosis(
    spectrum_diag: Dict[str, Any],
    *,
    bearing_enabled: bool,
    bearing_ai: Optional[Dict[str, Any]] = None,
    bearing_assessment: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    bearing_ai = bearing_ai or {}
    bearing_assessment = bearing_assessment or {}

    spectrum_headline = _safe_text(spectrum_diag.get("headline"))
    spectrum_narrative = _safe_text(spectrum_diag.get("narrative"))
    spectrum_status = _safe_text(spectrum_diag.get("status")).upper()

    bearing_fault_type = _safe_text(bearing_ai.get("fault_type"))
    bearing_fault_message = _safe_text(bearing_ai.get("message"))
    bearing_confidence = float(bearing_ai.get("confidence", 0.0) or 0.0)
    bearing_severity = _safe_text(bearing_ai.get("severity")) or "Normal"

    has_bearing_fault = (
        bearing_enabled
        and bearing_fault_type
        and bearing_fault_type != "No clear bearing fault"
        and bearing_confidence >= 0.45
    )

    matched_families = bearing_assessment.get("matched_families", []) or []
    matched_family_names = [
        str(item.get("family") or "")
        for item in matched_families
        if item.get("family")
    ]
    matched_family_names = _unique_texts(matched_family_names)

    spectrum_rank = _status_rank(spectrum_status)
    bearing_rank = _bearing_severity_rank(bearing_severity)
    global_rank = max(spectrum_rank, bearing_rank)
    global_severity = _global_severity_name(global_rank)

    primary_fault = spectrum_headline or "Sin patrón dominante"
    secondary_fault = ""

    if has_bearing_fault and spectrum_status in {"WARNING", "DANGER"}:
        if bearing_rank >= 3:
            primary_fault = bearing_fault_type
            secondary_fault = spectrum_headline
        else:
            primary_fault = spectrum_headline
            secondary_fault = bearing_fault_type
    elif has_bearing_fault:
        primary_fault = bearing_fault_type
        secondary_fault = ""
    else:
        primary_fault = spectrum_headline or "Sin patrón dominante"

    confidence = 0.55
    if spectrum_status == "SAFE":
        confidence = 0.60
    elif spectrum_status == "WARNING":
        confidence = 0.78
    elif spectrum_status == "DANGER":
        confidence = 0.90

    if has_bearing_fault:
        confidence = max(confidence, min(0.97, 0.50 + bearing_confidence * 0.45))

    confidence_pct = int(round(confidence * 100.0))

    title = primary_fault
    if secondary_fault:
        title = f"{primary_fault} + {secondary_fault}"

    parts: List[str] = []

    if has_bearing_fault and secondary_fault:
        parts.append(
            f"la condición dominante del espectro es consistente con {primary_fault.lower()}, "
            f"con evidencia secundaria compatible con {secondary_fault.lower()}."
        )
    elif has_bearing_fault:
        parts.append(
            f"el patrón más relevante es compatible con {primary_fault.lower()}."
        )
    elif spectrum_headline:
        parts.append(
            f"la condición dominante del espectro es consistente con {primary_fault.lower()}."
        )

    if spectrum_narrative:
        parts.append(spectrum_narrative)

    if has_bearing_fault and bearing_fault_message:
        parts.append(bearing_fault_message)

    assessment_narrative = _safe_text(bearing_assessment.get("narrative"))
    if has_bearing_fault and assessment_narrative:
        parts.append(assessment_narrative)

    if has_bearing_fault and matched_family_names:
        parts.append(
            "Familias coincidentes de rodamiento detectadas: " + ", ".join(matched_family_names) + "."
        )

    narrative = "\n\n".join([p for p in parts if p]).strip()
    if not narrative:
        narrative = "No se identificó un patrón suficientemente definido para emitir diagnóstico unificado."

    contributors: List[str] = []
    if spectrum_headline:
        contributors.append(spectrum_headline)
    if has_bearing_fault:
        contributors.append(bearing_fault_type)
    contributors = _unique_texts(contributors)

    return {
        "title": title,
        "severity": global_severity,
        "confidence_pct": confidence_pct,
        "primary_fault": primary_fault,
        "secondary_fault": secondary_fault,
        "contributors": contributors,
        "headline": title,
        "narrative": narrative,
    }
