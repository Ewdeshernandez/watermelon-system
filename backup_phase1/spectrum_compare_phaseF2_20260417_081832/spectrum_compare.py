from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import math
import re
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
    primary_fault = "Sin patrón dominante"
    secondary_fault = "—"
    executive_summary = "No se observa un cambio dominante claramente atribuible a un mecanismo de falla específico."
    technical_basis = (
        "La comparación A vs B no muestra una variación fuerte y consistente en 1X, 2X, 3X o armónicos altos "
        "que justifique inferir una evolución mecánica dominante con buena confianza."
    )
    recommendation = "Mantener esta comparación como referencia base y correlacionar con Orbit, Bode, Trends y condición operativa."
    confidence = 82 - comparability_penalty

    # INCREMENTOS
    if (
        one_x_delta_pct is not None and one_x_delta_pct >= 20
        and (two_x_delta_pct is None or two_x_delta_pct < 15)
        and (high_harm_delta_pct is None or high_harm_delta_pct < 15)
    ):
        severity = "Alerta"
        severity_color = "#f59e0b"
        title = "Incremento dominante en 1X"
        primary_fault = "Cambio sincrónico tipo desbalance"
        secondary_fault = "Sin evidencia fuerte de armónicos altos"
        executive_summary = "B presenta crecimiento dominante en 1X respecto a A."
        technical_basis = (
            "La componente 1X aumenta sin crecimiento proporcional en 2X ni en armónicos altos, "
            "patrón consistente con aumento de respuesta sincrónica, compatible con progresión de desbalance "
            "si la condición operativa entre ambas mediciones es comparable."
        )
        recommendation = "Verificar balanceo, condición del rotor y estabilidad de fase; correlacionar con módulos Polar y Bode."
        confidence = max(confidence, 86)

    if (
        one_x_delta_pct is not None and one_x_delta_pct >= 20
        and two_x_delta_pct is not None and two_x_delta_pct >= 20
    ):
        severity = "Alerta"
        severity_color = "#f59e0b"
        title = "Incremento simultáneo de 1X y 2X"
        primary_fault = "Evolución sincrónica compuesta"
        secondary_fault = "Posible desalineación / tren de potencia"
        executive_summary = "B incrementa simultáneamente las componentes 1X y 2X frente a A."
        technical_basis = (
            "El crecimiento conjunto de 1X y 2X sugiere evolución del fenómeno sincrónico con posible transición "
            "desde una condición dominada por 1X hacia desalineación, incremento de carga dinámica o mayor influencia del tren de potencia."
        )
        recommendation = "Revisar alineación, condición del acople, rigidez de soportes y correlacionar con mediciones axiales y análisis de fase."
        confidence = max(confidence, 89)

    if (
        two_x_delta_pct is not None and two_x_delta_pct >= 20
        and three_x_delta_pct is not None and three_x_delta_pct >= 10
    ):
        severity = "Alerta"
        severity_color = "#f59e0b"
        title = "Mayor contenido en 2X y 3X"
        primary_fault = "Firma compatible con desalineación"
        secondary_fault = "Efecto del tren de potencia"
        executive_summary = "B incrementa 2X y 3X respecto a A."
        technical_basis = (
            "El aumento conjunto de 2X y 3X es compatible con evolución hacia desalineación, "
            "incremento del efecto del acople o cambios dinámicos del tren de potencia."
        )
        recommendation = "Revisar alineación, acople, fijaciones mecánicas y confirmar con forma de onda y fase."
        confidence = max(confidence, 88)

    if high_harm_delta_pct is not None and high_harm_delta_pct >= 25:
        severity = "Severa"
        severity_color = "#dc2626"
        title = "Aumento de armónicos altos"
        primary_fault = "Posible holgura / no linealidad"
        secondary_fault = "Degradación de rigidez"
        executive_summary = "B muestra crecimiento importante en armónicos altos respecto a A."
        technical_basis = (
            "El aumento de armónicos altos apunta a progresión hacia holgura mecánica, "
            "no linealidad estructural o pérdida de rigidez."
        )
        recommendation = "Inspeccionar bases, pernos, soportes, pedestal, holguras y comportamiento de impactos en forma de onda."
        confidence = max(confidence, 90)

    if (
        overall_delta_pct is not None and overall_delta_pct >= 20
        and (peak_delta_pct is None or peak_delta_pct < 12)
    ):
        severity = "Alerta"
        severity_color = "#f59e0b"
        title = "Mayor energía de banda ancha"
        primary_fault = "Incremento de contenido distribuido"
        secondary_fault = "Proceso / flujo / fricción"
        executive_summary = "La energía global crece más que el pico dominante."
        technical_basis = (
            "Esto sugiere aumento de contenido distribuido o banda ancha, compatible con proceso, "
            "flujo, fricción o excitación no puramente sincrónica."
        )
        recommendation = "Revisar condición de proceso, cavitación, turbulencia, roce o excitaciones no estacionarias."
        confidence = max(confidence, 84)

    # DISMINUCIONES
    if (
        one_x_delta_pct is not None and one_x_delta_pct <= -20
        and overall_delta_pct is not None and overall_delta_pct <= -20
        and (two_x_delta_pct is None or two_x_delta_pct > -20)
    ):
        severity = "Normal"
        severity_color = "#16a34a"
        title = "Reducción de respuesta sincrónica"
        primary_fault = "Disminución de firma tipo desbalance"
        secondary_fault = "Menor respuesta 1X"
        executive_summary = "B presenta reducción importante de la componente 1X y de la energía global respecto a A."
        technical_basis = (
            "La disminución simultánea de 1X y del overall indica reducción de la respuesta sincrónica de la máquina, "
            "compatible con disminución de la firma asociada a desbalance o con una condición operativa menos severa."
        )
        recommendation = "Validar que RPM, carga y condiciones de proceso sean comparables antes de interpretar esta disminución como mejora mecánica real."
        confidence = max(confidence, 84)

    if (
        one_x_delta_pct is not None and one_x_delta_pct <= -20
        and two_x_delta_pct is not None and two_x_delta_pct <= -20
        and overall_delta_pct is not None and overall_delta_pct <= -20
    ):
        severity = "Normal"
        severity_color = "#16a34a"
        title = "Reducción de firma sincrónica global"
        primary_fault = "Disminución de 1X y 2X"
        secondary_fault = "Menor excitación dinámica"
        executive_summary = "B muestra reducción simultánea de 1X, 2X y energía global respecto a A."
        technical_basis = (
            "La reducción conjunta de 1X, 2X y overall sugiere disminución de la firma sincrónica de la máquina, "
            "compatible con menor severidad de desbalance/desalineación o con cambio favorable en condición operativa."
        )
        recommendation = "Correlacionar con condición operativa, fase, carga y antecedentes de mantenimiento para diferenciar mejora mecánica real de cambio de proceso."
        confidence = max(confidence, 87)

    if (
        two_x_delta_pct is not None and two_x_delta_pct <= -20
        and three_x_delta_pct is not None and three_x_delta_pct <= -10
    ):
        severity = "Normal"
        severity_color = "#16a34a"
        title = "Reducción de contenido 2X y 3X"
        primary_fault = "Disminución de firma compatible con desalineación"
        secondary_fault = "Menor efecto del tren de potencia"
        executive_summary = "B reduce 2X y 3X respecto a A."
        technical_basis = (
            "La disminución de 2X y 3X sugiere reducción de la firma asociada a desalineación o menor excitación del tren de potencia."
        )
        recommendation = "Confirmar si existió intervención mecánica o si las condiciones de carga/proceso cambiaron entre mediciones."
        confidence = max(confidence, 84)

    if high_harm_delta_pct is not None and high_harm_delta_pct <= -25:
        severity = "Normal"
        severity_color = "#16a34a"
        title = "Reducción de armónicos altos"
        primary_fault = "Menor no linealidad / holgura aparente"
        secondary_fault = "Disminución de contenido armónico alto"
        executive_summary = "B reduce el contenido de armónicos altos respecto a A."
        technical_basis = "La disminución de armónicos altos puede indicar reducción de no linealidad, impactos o holgura aparente."
        recommendation = "Correlacionar con forma de onda e historial de intervención para confirmar si hubo mejora estructural o mecánica."
        confidence = max(confidence, 82)

    if (
        overall_delta_pct is not None and overall_delta_pct <= -20
        and (peak_delta_pct is None or peak_delta_pct <= -12)
    ):
        severity = "Normal"
        severity_color = "#16a34a"
        title = "Reducción global de energía"
        primary_fault = "Disminución general de severidad"
        secondary_fault = "Menor excitación global"
        executive_summary = "B presenta reducción global de energía respecto a A."
        technical_basis = "La caída del overall y del pico dominante sugiere una condición menos energética en la medición más reciente."
        recommendation = "Validar comparabilidad operativa antes de concluir mejora mecánica definitiva."
        confidence = max(confidence, 80)

    if peak_delta_pct is not None and abs(peak_delta_pct) <= 8 and (overall_delta_pct is None or abs(overall_delta_pct) <= 8):
        severity = "Normal"
        severity_color = "#16a34a"
        title = "Espectros comparables sin variación fuerte"
        primary_fault = "Sin evolución dominante"
        secondary_fault = "—"
        executive_summary = "A y B se mantienen cercanos en pico dominante y energía global."
        technical_basis = "No se observa una evolución espectral fuerte entre ambos estados bajo esta comparación."
        recommendation = "Mantener monitoreo y usar esta comparación como línea base."
        confidence = max(confidence, 80)

    confidence = max(45, min(96, int(round(confidence))))
    comparability_score = max(0, min(100, 100 - comparability_penalty))

    condition_summary = build_compare_condition_summary(
        {
            "peak_delta_pct": peak_delta_pct,
            "overall_delta_pct": overall_delta_pct,
            "one_x_delta_pct": one_x_delta_pct,
            "two_x_delta_pct": two_x_delta_pct,
            "high_harm_delta_pct": high_harm_delta_pct,
            "comparability_score": comparability_score,
        }
    )

    narrative_parts: List[str] = []
    if delta_days is not None and delta_days > 0:
        narrative_parts.append(f"En un periodo de {delta_days} días, {executive_summary[0].lower() + executive_summary[1:]}")
    else:
        narrative_parts.append(executive_summary)
    narrative_parts.append(technical_basis)
    narrative_parts.append(recommendation)
    narrative = " ".join(part.strip() for part in narrative_parts if part and part.strip())

    chips = [
        (f"Severidad: {severity}", severity_color),
        (f"Confianza: {confidence}%", None),
        (f"Compare Score: {condition_summary['compare_score']}", condition_summary["traffic_color"]),
        (f"Trend: {condition_summary['condition_trend']}", None),
        (f"Semáforo: {condition_summary['traffic_light']}", condition_summary["traffic_color"]),
        (f"Comparabilidad: {comparability_score}%", None),
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
        "executive_summary": executive_summary,
        "technical_basis": technical_basis,
        "recommendation": recommendation,
        "primary_fault": primary_fault,
        "secondary_fault": secondary_fault,
        "comparability_score": comparability_score,
        "compare_score": condition_summary["compare_score"],
        "condition_trend": condition_summary["condition_trend"],
        "traffic_light": condition_summary["traffic_light"],
        "traffic_color": condition_summary["traffic_color"],
        "condition_text": condition_summary["condition_text"],
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

def build_compare_narrative(
    compare_assessment: Dict[str, Any],
    delta_days: Optional[int],
) -> str:
    exec_sum = str(compare_assessment.get("executive_summary") or "").strip()
    tech = str(compare_assessment.get("technical_basis") or "").strip()
    reco = str(compare_assessment.get("recommendation") or "").strip()
    condition_text = str(compare_assessment.get("condition_text") or "").strip()
    trend = str(compare_assessment.get("condition_trend") or "").strip()
    score = compare_assessment.get("compare_score")

    intro = ""
    if delta_days is not None and delta_days > 0:
        intro = f"En un periodo de {delta_days} días, "

    score_text = ""
    if score is not None and trend:
        score_text = f" El compare score es {score}/100 y la tendencia estimada de condición es {trend}."

    narrative = f"{intro}{exec_sum} {tech} {reco}{score_text} {condition_text}".strip()
    narrative = re.sub(r"\s+", " ", narrative)
    return narrative

def build_compare_report_notes(
    compare_assessment: Dict[str, Any],
    summary_a: Dict[str, Any],
    summary_b: Dict[str, Any],
    time_label: str = "",
    insights_df: Optional[pd.DataFrame] = None,
    delta_days: Optional[int] = None,
) -> str:
    blocks: List[str] = []

    if time_label:
        blocks.append(time_label)

    title = str(compare_assessment.get("title") or "").strip()
    if title:
        blocks.append(title)

    narrative = build_compare_narrative(compare_assessment, delta_days)
    if narrative:
        blocks.append(narrative)

    if insights_df is not None and not insights_df.empty:
        top = insights_df.head(3)
        insight_lines: List[str] = []
        for _, row in top.iterrows():
            insight_lines.append(
                f"- {row['Insight']}: Δ {row['Δ % vs A']}% — {row['Interpretation']}"
            )
        if insight_lines:
            blocks.append(
                "Principales cambios detectados:\n" +
                "\n".join(insight_lines)
            )

    blocks.append(
        "Resumen comparativo cuantitativo:\n"
        f"- A Peak: {format_number(summary_a.get('peak_amp'), 3)} @ {format_number(summary_a.get('peak_freq_cpm'), 1)} CPM\n"
        f"- B Peak: {format_number(summary_b.get('peak_amp'), 3)} @ {format_number(summary_b.get('peak_freq_cpm'), 1)} CPM\n"
        f"- Δ Peak: {format_number(compare_assessment.get('peak_delta_pct'), 1)}%\n"
        f"- Δ Overall: {format_number(compare_assessment.get('overall_delta_pct'), 1)}%\n"
        f"- Δ 1X: {format_number(compare_assessment.get('one_x_delta_pct'), 1)}%\n"
        f"- Δ 2X: {format_number(compare_assessment.get('two_x_delta_pct'), 1)}%\n"
        f"- Falla primaria: {compare_assessment.get('primary_fault') or '—'}\n"
        f"- Falla secundaria: {compare_assessment.get('secondary_fault') or '—'}\n"
        f"- Compare Score: {format_number(compare_assessment.get('compare_score'), 0)}/100\n"
        f"- Condition Trend: {compare_assessment.get('condition_trend') or '—'}\n"
        f"- Semáforo: {compare_assessment.get('traffic_light') or '—'}\n"
        f"- Comparabilidad: {format_number(compare_assessment.get('comparability_score'), 0)}%\n"
        f"- Confianza diagnóstica: {format_number(compare_assessment.get('confidence_pct'), 0)}%"
    )

    warnings = compare_assessment.get("warnings", [])
    if warnings:
        blocks.append(
            "Advertencias de comparabilidad:\n- " +
            "\n- ".join(str(w) for w in warnings)
        )

    return "\n\n".join(blocks).strip()
def build_compare_metric_table(
    summary_a: Dict[str, Any],
    summary_b: Dict[str, Any],
    compare_assessment: Dict[str, Any],
) -> pd.DataFrame:
    rows = [
        {
            "Metric": "Dominant Peak Frequency (CPM)",
            "A": format_number(summary_a.get("peak_freq_cpm"), 1),
            "B": format_number(summary_b.get("peak_freq_cpm"), 1),
            "Δ % vs A": format_number(safe_pct_change(summary_b.get("peak_freq_cpm"), summary_a.get("peak_freq_cpm")), 1),
        },
        {
            "Metric": "Dominant Peak Amplitude",
            "A": format_number(summary_a.get("peak_amp"), 3),
            "B": format_number(summary_b.get("peak_amp"), 3),
            "Δ % vs A": format_number(compare_assessment.get("peak_delta_pct"), 1),
        },
        {
            "Metric": "Spectrum Overall",
            "A": format_number(summary_a.get("overall"), 3),
            "B": format_number(summary_b.get("overall"), 3),
            "Δ % vs A": format_number(compare_assessment.get("overall_delta_pct"), 1),
        },
        {
            "Metric": "1X Amplitude",
            "A": format_number(summary_a.get("one_x_amp"), 3),
            "B": format_number(summary_b.get("one_x_amp"), 3),
            "Δ % vs A": format_number(compare_assessment.get("one_x_delta_pct"), 1),
        },
        {
            "Metric": "2X Amplitude",
            "A": format_number(summary_a.get("two_x_amp"), 3),
            "B": format_number(summary_b.get("two_x_amp"), 3),
            "Δ % vs A": format_number(compare_assessment.get("two_x_delta_pct"), 1),
        },
        {
            "Metric": "3X Amplitude",
            "A": format_number(summary_a.get("three_x_amp"), 3),
            "B": format_number(summary_b.get("three_x_amp"), 3),
            "Δ % vs A": format_number(compare_assessment.get("three_x_delta_pct"), 1),
        },
        {
            "Metric": "High Harmonic Amplitude (≥4X)",
            "A": format_number(summary_a.get("high_harm_amp"), 3),
            "B": format_number(summary_b.get("high_harm_amp"), 3),
            "Δ % vs A": format_number(compare_assessment.get("high_harm_delta_pct"), 1),
        },
    ]
    return pd.DataFrame(rows)


def build_compare_validation_table(
    summary_a: Dict[str, Any],
    summary_b: Dict[str, Any],
) -> pd.DataFrame:
    rpm_delta = safe_pct_change(summary_b.get("rpm"), summary_a.get("rpm"))
    fs_delta = safe_pct_change(summary_b.get("sample_rate_hz"), summary_a.get("sample_rate_hz"))
    dur_delta = safe_pct_change(summary_b.get("duration_s"), summary_a.get("duration_s"))

    rows = [
        {
            "Validation Parameter": "Amplitude unit",
            "A": str(summary_a.get("amplitude_unit") or "—"),
            "B": str(summary_b.get("amplitude_unit") or "—"),
            "Δ / Status": "OK" if summary_a.get("amplitude_unit") == summary_b.get("amplitude_unit") else "Mismatch",
        },
        {
            "Validation Parameter": "RPM",
            "A": format_number(summary_a.get("rpm"), 0),
            "B": format_number(summary_b.get("rpm"), 0),
            "Δ / Status": (format_number(rpm_delta, 1) + "%") if rpm_delta is not None else "—",
        },
        {
            "Validation Parameter": "Sample Rate (Hz)",
            "A": format_number(summary_a.get("sample_rate_hz"), 2),
            "B": format_number(summary_b.get("sample_rate_hz"), 2),
            "Δ / Status": (format_number(fs_delta, 1) + "%") if fs_delta is not None else "—",
        },
        {
            "Validation Parameter": "Duration (s)",
            "A": format_number(summary_a.get("duration_s"), 3),
            "B": format_number(summary_b.get("duration_s"), 3),
            "Δ / Status": (format_number(dur_delta, 1) + "%") if dur_delta is not None else "—",
        },
    ]
    return pd.DataFrame(rows)


def _insight_interpretation(label: str, delta_pct: Optional[float]) -> str:
    if delta_pct is None:
        return "Sin referencia suficiente para comparar."

    abs_delta = abs(float(delta_pct))

    if label == "1X":
        if delta_pct >= 20:
            return "Crecimiento sincrónico fuerte; compatible con desbalance o aumento de respuesta 1X."
        if delta_pct <= -20:
            return "Disminución de respuesta 1X respecto a la referencia; posible reducción de firma sincrónica."
        return "Cambio menor en componente sincrónica."

    if label == "2X":
        if delta_pct >= 20:
            return "Crecimiento relevante en 2X; compatible con desalineación o efecto del tren de potencia."
        if delta_pct <= -20:
            return "Reducción importante en 2X; posible disminución de firma asociada a desalineación."
        return "Cambio menor en 2X."

    if label == "3X":
        if delta_pct >= 15:
            return "Crecimiento en 3X; refuerza lectura de desalineación / comportamiento del tren."
        if delta_pct <= -15:
            return "Disminución clara en 3X."
        return "Cambio menor en 3X."

    if label == "Overall":
        if delta_pct >= 20:
            return "Aumento global de energía; revisar si el crecimiento es concentrado o distribuido."
        if delta_pct <= -20:
            return "Reducción global de energía respecto a la referencia."
        return "Cambio global moderado."

    if label == "High Harmonics":
        if delta_pct >= 25:
            return "Crecimiento en armónicos altos; posible holgura, no linealidad o pérdida de rigidez."
        if delta_pct <= -25:
            return "Disminución clara de armónicos altos."
        return "Cambio moderado en contenido armónico alto."

    if abs_delta >= 20:
        return "Cambio relevante."
    return "Cambio moderado o bajo."


def build_compare_insight_table(
    summary_a: Dict[str, Any],
    summary_b: Dict[str, Any],
    compare_assessment: Dict[str, Any],
) -> pd.DataFrame:
    insight_rows = [
        {
            "Component": "1X",
            "A": summary_a.get("one_x_amp"),
            "B": summary_b.get("one_x_amp"),
            "DeltaPct": compare_assessment.get("one_x_delta_pct"),
            "Priority": 1,
        },
        {
            "Component": "2X",
            "A": summary_a.get("two_x_amp"),
            "B": summary_b.get("two_x_amp"),
            "DeltaPct": compare_assessment.get("two_x_delta_pct"),
            "Priority": 2,
        },
        {
            "Component": "3X",
            "A": summary_a.get("three_x_amp"),
            "B": summary_b.get("three_x_amp"),
            "DeltaPct": compare_assessment.get("three_x_delta_pct"),
            "Priority": 3,
        },
        {
            "Component": "Overall",
            "A": summary_a.get("overall"),
            "B": summary_b.get("overall"),
            "DeltaPct": compare_assessment.get("overall_delta_pct"),
            "Priority": 4,
        },
        {
            "Component": "High Harmonics",
            "A": summary_a.get("high_harm_amp"),
            "B": summary_b.get("high_harm_amp"),
            "DeltaPct": compare_assessment.get("high_harm_delta_pct"),
            "Priority": 5,
        },
    ]

    normalized_rows: List[Dict[str, Any]] = []
    for row in insight_rows:
        delta_pct = row["DeltaPct"]
        normalized_rows.append(
            {
                "Insight": row["Component"],
                "A": format_number(row["A"], 3),
                "B": format_number(row["B"], 3),
                "Δ % vs A": format_number(delta_pct, 1),
                "Interpretation": _insight_interpretation(row["Component"], delta_pct),
                "_abs_delta": abs(float(delta_pct)) if delta_pct is not None else -1.0,
                "_priority": row["Priority"],
            }
        )

    df = pd.DataFrame(normalized_rows)
    if df.empty:
        return df

    df = df.sort_values(by=["_abs_delta", "_priority"], ascending=[False, True]).reset_index(drop=True)
    df = df.drop(columns=["_abs_delta", "_priority"])
    return df


def build_compare_top_findings(compare_insights_df: pd.DataFrame, max_items: int = 3) -> List[str]:
    findings: List[str] = []
    if compare_insights_df is None or compare_insights_df.empty:
        return findings

    for _, row in compare_insights_df.head(max_items).iterrows():
        insight = str(row.get("Insight") or "—")
        delta_pct = str(row.get("Δ % vs A") or "—")
        interpretation = str(row.get("Interpretation") or "").strip()
        findings.append(f"{insight}: Δ {delta_pct}% — {interpretation}")

    return findings


def build_compare_condition_summary(
    compare_assessment: Dict[str, Any],
) -> Dict[str, Any]:
    peak_delta = compare_assessment.get("peak_delta_pct")
    overall_delta = compare_assessment.get("overall_delta_pct")
    one_x_delta = compare_assessment.get("one_x_delta_pct")
    two_x_delta = compare_assessment.get("two_x_delta_pct")
    high_harm_delta = compare_assessment.get("high_harm_delta_pct")
    comparability = float(compare_assessment.get("comparability_score") or 0.0)

    worsening_score = 0.0
    improving_score = 0.0

    def add_component(delta: Optional[float], weight: float) -> None:
        nonlocal worsening_score, improving_score
        if delta is None:
            return
        try:
            val = float(delta)
        except Exception:
            return
        if val > 0:
            worsening_score += min(abs(val), 100.0) * weight
        elif val < 0:
            improving_score += min(abs(val), 100.0) * weight

    add_component(one_x_delta, 0.34)
    add_component(two_x_delta, 0.24)
    add_component(overall_delta, 0.24)
    add_component(high_harm_delta, 0.12)
    add_component(peak_delta, 0.06)

    raw_balance = worsening_score - improving_score

    if comparability < 85:
        raw_balance *= 0.90
    if comparability < 70:
        raw_balance *= 0.82
    if comparability < 50:
        raw_balance *= 0.72

    compare_score = 50.0 + raw_balance / 3.0
    compare_score = max(0.0, min(100.0, compare_score))

    if compare_score >= 60:
        condition_trend = "Worsening"
        traffic_light = "Red"
        traffic_color = "#dc2626"
        condition_text = "La condición comparativa sugiere deterioro respecto a la referencia."
    elif compare_score <= 40:
        condition_trend = "Improving"
        traffic_light = "Green"
        traffic_color = "#16a34a"
        condition_text = "La condición comparativa sugiere mejora o reducción de severidad respecto a la referencia."
    else:
        condition_trend = "Stable"
        traffic_light = "Yellow"
        traffic_color = "#f59e0b"
        condition_text = "La condición comparativa se mantiene estable o con cambio moderado."

    return {
        "compare_score": int(round(compare_score)),
        "condition_trend": condition_trend,
        "traffic_light": traffic_light,
        "traffic_color": traffic_color,
        "condition_text": condition_text,
    }

