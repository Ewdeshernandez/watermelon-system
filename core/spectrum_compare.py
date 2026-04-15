from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import math
import pandas as pd


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


def safe_pct_change(new_value: Optional[float], old_value: Optional[float]) -> Optional[float]:
    if new_value is None or old_value is None:
        return None
    try:
        new_v = float(new_value)
        old_v = float(old_value)
    except Exception:
        return None
    if not math.isfinite(new_v) or not math.isfinite(old_v):
        return None
    if abs(old_v) < 1e-12:
        return None
    return ((new_v - old_v) / abs(old_v)) * 100.0


def parse_compare_timestamp(ts: Optional[str]) -> Optional[pd.Timestamp]:
    if ts is None:
        return None
    raw = str(ts).strip()
    if not raw:
        return None
    try:
        parsed = pd.to_datetime(raw, errors="coerce", utc=False)
        if pd.isna(parsed):
            return None
        return parsed
    except Exception:
        return None


def order_compare_records_by_time(records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Optional[pd.Timestamp], Optional[pd.Timestamp], Optional[int]]:
    if len(records) != 2:
        return records, None, None, None

    pairs: List[Tuple[Optional[pd.Timestamp], Dict[str, Any]]] = []
    for rec in records:
        pairs.append((parse_compare_timestamp(rec.get("timestamp")), rec))

    if all(ts is not None for ts, _ in pairs):
        pairs_sorted = sorted(pairs, key=lambda item: item[0])
        ordered_records = [pairs_sorted[0][1], pairs_sorted[1][1]]
        ts_a = pairs_sorted[0][0]
        ts_b = pairs_sorted[1][0]
        delta_days = int((ts_b - ts_a).total_seconds() // 86400)
        return ordered_records, ts_a, ts_b, delta_days

    return records, pairs[0][0], pairs[1][0], None


def format_compare_timestamp(ts: Optional[pd.Timestamp], fallback: Optional[str] = None) -> str:
    if ts is None:
        return fallback or "—"
    try:
        return ts.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return fallback or "—"


def build_compare_time_label(ts_a: Optional[pd.Timestamp], ts_b: Optional[pd.Timestamp], delta_days: Optional[int]) -> str:
    left = format_compare_timestamp(ts_a, "—")
    right = format_compare_timestamp(ts_b, "—")
    if delta_days is not None and delta_days >= 0:
        return f"{left} → {right} | Δt: {delta_days} días"
    return f"{left} → {right}"


def build_compare_assessment(
    summary_a: Dict[str, Any],
    summary_b: Dict[str, Any],
    *,
    delta_days: Optional[int] = None,
) -> Dict[str, Any]:
    warnings: List[str] = []

    unit_a = summary_a.get("amplitude_unit")
    unit_b = summary_b.get("amplitude_unit")
    if unit_a != unit_b:
        warnings.append("Las unidades base no coinciden entre A y B.")

    rpm_delta_pct = safe_pct_change(summary_b.get("rpm"), summary_a.get("rpm"))
    if rpm_delta_pct is not None and abs(rpm_delta_pct) > 3.0:
        warnings.append(f"RPM diferentes entre A y B ({format_number(rpm_delta_pct, 1)}%).")

    fs_delta_pct = safe_pct_change(summary_b.get("sample_rate_hz"), summary_a.get("sample_rate_hz"))
    if fs_delta_pct is not None and abs(fs_delta_pct) > 5.0:
        warnings.append(f"Frecuencia de muestreo diferente ({format_number(fs_delta_pct, 1)}%).")

    dur_delta_pct = safe_pct_change(summary_b.get("duration_s"), summary_a.get("duration_s"))
    if dur_delta_pct is not None and abs(dur_delta_pct) > 20.0:
        warnings.append(f"Duración de señal diferente ({format_number(dur_delta_pct, 1)}%).")

    peak_delta_pct = safe_pct_change(summary_b.get("peak_amp"), summary_a.get("peak_amp"))
    overall_delta_pct = safe_pct_change(summary_b.get("overall"), summary_a.get("overall"))
    one_x_delta_pct = safe_pct_change(summary_b.get("one_x_amp"), summary_a.get("one_x_amp"))
    two_x_delta_pct = safe_pct_change(summary_b.get("two_x_amp"), summary_a.get("two_x_amp"))
    three_x_delta_pct = safe_pct_change(summary_b.get("three_x_amp"), summary_a.get("three_x_amp"))
    high_harm_delta_pct = safe_pct_change(summary_b.get("high_harm_amp"), summary_a.get("high_harm_amp"))

    comparability_penalty = 0
    comparability_penalty += 15 if any("RPM diferentes" in w for w in warnings) else 0
    comparability_penalty += 10 if any("Frecuencia de muestreo" in w for w in warnings) else 0
    comparability_penalty += 10 if any("Duración de señal" in w for w in warnings) else 0
    comparability_penalty += 10 if any("unidades base" in w for w in warnings) else 0

    severity = "Normal"
    severity_color = "#16a34a"
    title = "Sin cambio espectral dominante"
    narrative = (
        "La comparación A vs B no muestra un cambio dominante claramente asociado a una evolución mecánica específica. "
        "Se recomienda conservar esta comparación como referencia base y correlacionar con Orbit, Bode, Trends y condición operativa."
    )
    confidence = 82 - comparability_penalty

    if (
        one_x_delta_pct is not None and one_x_delta_pct >= 20
        and (two_x_delta_pct is None or two_x_delta_pct < 15)
        and (high_harm_delta_pct is None or high_harm_delta_pct < 15)
    ):
        severity = "Alerta"
        severity_color = "#f59e0b"
        title = "Incremento dominante en 1X"
        narrative = (
            "El espectro B presenta incremento dominante de la componente 1X respecto a A, sin crecimiento proporcional en 2X ni en armónicos altos. "
            "Este patrón es consistente con aumento de respuesta sincrónica, compatible con progresión de desbalance si la condición operativa es comparable."
        )
        confidence = max(confidence, 86)

    if (
        one_x_delta_pct is not None and one_x_delta_pct >= 20
        and two_x_delta_pct is not None and two_x_delta_pct >= 20
    ):
        severity = "Alerta"
        severity_color = "#f59e0b"
        title = "Incremento simultáneo de 1X y 2X"
        narrative = (
            "El espectro B incrementa simultáneamente las componentes 1X y 2X respecto a A. "
            "Este comportamiento sugiere evolución del fenómeno sincrónico con posible transición desde una condición dominada por 1X "
            "hacia desalineación, incremento de carga dinámica o mayor efecto del tren de potencia."
        )
        confidence = max(confidence, 89)

    if (
        two_x_delta_pct is not None and two_x_delta_pct >= 20
        and three_x_delta_pct is not None and three_x_delta_pct >= 10
    ):
        severity = "Alerta"
        severity_color = "#f59e0b"
        title = "Mayor contenido en 2X y 3X"
        narrative = (
            "El espectro B incrementa las componentes 2X y 3X respecto a A. "
            "El patrón es compatible con evolución hacia desalineación, incremento del efecto del acople o cambios dinámicos del tren de potencia."
        )
        confidence = max(confidence, 88)

    if high_harm_delta_pct is not None and high_harm_delta_pct >= 25:
        severity = "Severa"
        severity_color = "#dc2626"
        title = "Aumento de armónicos altos"
        narrative = (
            "El espectro B muestra incremento relevante en armónicos altos respecto a A. "
            "Este patrón apunta a progresión hacia holgura mecánica, no linealidad estructural o degradación de rigidez."
        )
        confidence = max(confidence, 90)

    if (
        overall_delta_pct is not None and overall_delta_pct >= 20
        and (peak_delta_pct is None or peak_delta_pct < 12)
    ):
        severity = "Alerta"
        severity_color = "#f59e0b"
        title = "Mayor energía de banda ancha"
        narrative = (
            "La energía global del espectro en B aumenta más que el pico dominante. "
            "Esto sugiere crecimiento de contenido distribuido o banda ancha, compatible con proceso, flujo, fricción o excitación no puramente sincrónica."
        )
        confidence = max(confidence, 84)

    if peak_delta_pct is not None and abs(peak_delta_pct) <= 8 and (overall_delta_pct is None or abs(overall_delta_pct) <= 8):
        severity = "Normal"
        severity_color = "#16a34a"
        title = "Espectros comparables sin variación fuerte"
        narrative = (
            "A y B se mantienen cercanos en pico dominante y energía global. "
            "No se observa una evolución espectral fuerte entre ambos estados bajo esta comparación."
        )
        confidence = max(confidence, 80)

    if delta_days is not None and delta_days > 0 and narrative:
        narrative = f"En un periodo de {delta_days} días, {narrative[0].lower() + narrative[1:]}"

    confidence = max(45, min(96, int(round(confidence))))

    chips = [
        (f"Severidad: {severity}", severity_color),
        (f"Confianza: {confidence}%", None),
        (f"Δ Peak: {format_number(peak_delta_pct, 1)}%", None),
        (f"Δ Overall: {format_number(overall_delta_pct, 1)}%", None),
        (f"Δ 1X: {format_number(one_x_delta_pct, 1)}%", None),
        (f"Δ 2X: {format_number(two_x_delta_pct, 1)}%", None),
    ]

    return {
        "severity": severity,
        "severity_color": severity_color,
        "title": title,
        "narrative": narrative,
        "confidence_pct": confidence,
        "chips": chips,
        "warnings": warnings,
        "peak_delta_pct": peak_delta_pct,
        "overall_delta_pct": overall_delta_pct,
        "one_x_delta_pct": one_x_delta_pct,
        "two_x_delta_pct": two_x_delta_pct,
        "three_x_delta_pct": three_x_delta_pct,
        "high_harm_delta_pct": high_harm_delta_pct,
    }


def build_compare_report_notes(
    compare_assessment: Dict[str, Any],
    summary_a: Dict[str, Any],
    summary_b: Dict[str, Any],
    time_label: str = "",
) -> str:
    blocks = []

    title = str(compare_assessment.get("title") or "").strip()
    narrative = str(compare_assessment.get("narrative") or "").strip()

    if time_label:
        blocks.append(time_label)
    if title:
        blocks.append(title)
    if narrative:
        blocks.append(narrative)

    blocks.append(
        (
            "Resumen comparativo:\n"
            f"- A Peak: {format_number(summary_a.get('peak_amp'), 3)} @ {format_number(summary_a.get('peak_freq_cpm'), 1)} CPM\n"
            f"- B Peak: {format_number(summary_b.get('peak_amp'), 3)} @ {format_number(summary_b.get('peak_freq_cpm'), 1)} CPM\n"
            f"- Δ Peak: {format_number(compare_assessment.get('peak_delta_pct'), 1)}%\n"
            f"- Δ Overall: {format_number(compare_assessment.get('overall_delta_pct'), 1)}%\n"
            f"- Δ 1X: {format_number(compare_assessment.get('one_x_delta_pct'), 1)}%\n"
            f"- Δ 2X: {format_number(compare_assessment.get('two_x_delta_pct'), 1)}%"
        )
    )

    warnings = compare_assessment.get("warnings", [])
    if warnings:
        blocks.append("Advertencias de comparabilidad:\n- " + "\n- ".join(str(w) for w in warnings))

    return "\n\n".join(blocks).strip()
