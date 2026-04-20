from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import math
import pandas as pd


# ============================================================
# F5 EDITORIAL NARRATIVE
# ============================================================
def build_trend_editorial_narrative(assessment: dict) -> str:
    trend = str(assessment.get("trend_label") or "").lower()
    driver = str(assessment.get("top_driver") or "").upper()
    driver_pct = assessment.get("top_driver_pct")

    try:
        driver_pct_txt = f"{abs(float(driver_pct)):.1f}%"
    except:
        driver_pct_txt = "significativo"

    if trend == "improving":
        return (
            f"La tendencia multitemporal evidencia una mejora de la condición dinámica de la máquina, "
            f"caracterizada por la reducción de la respuesta vibratoria, especialmente en la componente {driver} "
            f"({driver_pct_txt}). Este comportamiento es consistente con una disminución de excitaciones dinámicas "
            f"como el desbalanceo y una menor severidad vibratoria global. "
            f"Se recomienda validar comparabilidad operativa antes de concluir una mejora mecánica definitiva."
        )

    if trend == "deteriorating":
        return (
            f"La tendencia multitemporal muestra un deterioro progresivo de la condición dinámica, con incremento "
            f"de la respuesta vibratoria, destacándose la componente {driver} ({driver_pct_txt}). "
            f"Este comportamiento puede estar asociado a aumento de excitaciones mecánicas como desbalanceo, "
            f"desalineación u otros fenómenos dinámicos."
        )

    if trend == "stable":
        return (
            f"La evaluación multitemporal no evidencia cambios significativos en la condición dinámica. "
            f"La respuesta vibratoria se mantiene estable sin variaciones dominantes."
        )

    if trend == "volatile":
        return (
            f"La tendencia presenta comportamiento variable sin patrón dominante claro. "
            f"Esto puede estar influenciado por condiciones operativas cambiantes."
        )

    return "No fue posible construir una interpretación técnica consistente."


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


def parse_trend_timestamp(ts: Optional[str]) -> Optional[pd.Timestamp]:
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


def format_trend_timestamp(ts: Optional[pd.Timestamp], fallback: str = "—") -> str:
    if ts is None:
        return fallback
    try:
        return ts.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return fallback


def order_trend_records_by_time(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def sort_key(rec: Dict[str, Any]) -> Tuple[int, Any]:
        ts = parse_trend_timestamp(rec.get("timestamp"))
        if ts is None:
            return (1, str(rec.get("timestamp") or ""))
        return (0, ts)

    return sorted(records, key=sort_key)


def build_trend_condition_summary(record_summary: Dict[str, Any], comparability_score: float = 100.0) -> Dict[str, Any]:
    peak = record_summary.get("peak_amp")
    overall = record_summary.get("overall")
    one_x = record_summary.get("one_x_amp")
    two_x = record_summary.get("two_x_amp")
    high_harm = record_summary.get("high_harm_amp")

    def safe_float(v: Any) -> float:
        try:
            x = float(v)
            if math.isfinite(x):
                return x
            return 0.0
        except Exception:
            return 0.0

    severity_raw = (
        safe_float(one_x) * 0.34
        + safe_float(two_x) * 0.22
        + safe_float(overall) * 0.24
        + safe_float(high_harm) * 0.14
        + safe_float(peak) * 0.06
    )

    if comparability_score < 85:
        severity_raw *= 0.95
    if comparability_score < 70:
        severity_raw *= 0.90

    return {
        "severity_raw": severity_raw,
        "comparability_score": comparability_score,
    }



def _derive_trend_direction(
    score_deltas: List[float],
    first_record: Optional[Dict[str, Any]] = None,
    last_record: Optional[Dict[str, Any]] = None,
) -> str:
    if not score_deltas and (first_record is None or last_record is None):
        return "Stable"

    positive = sum(1 for x in score_deltas if x > 2.5)
    negative = sum(1 for x in score_deltas if x < -2.5)

    one_x_change = safe_pct_change(
        last_record.get("one_x_amp") if last_record else None,
        first_record.get("one_x_amp") if first_record else None,
    )
    two_x_change = safe_pct_change(
        last_record.get("two_x_amp") if last_record else None,
        first_record.get("two_x_amp") if first_record else None,
    )
    overall_change = safe_pct_change(
        last_record.get("overall") if last_record else None,
        first_record.get("overall") if first_record else None,
    )
    high_harm_change = safe_pct_change(
        last_record.get("high_harm_amp") if last_record else None,
        first_record.get("high_harm_amp") if first_record else None,
    )

    worsening_votes = 0
    improving_votes = 0

    for value, threshold in [
        (one_x_change, 18.0),
        (two_x_change, 18.0),
        (overall_change, 18.0),
        (high_harm_change, 22.0),
    ]:
        if value is None:
            continue
        if value >= threshold:
            worsening_votes += 1
        elif value <= -threshold:
            improving_votes += 1

    if positive > 0 and negative > 0:
        if abs(positive - negative) <= 1:
            return "Volatile"

    if improving_votes >= 2 and worsening_votes == 0:
        return "Improving"

    if worsening_votes >= 2 and improving_votes == 0:
        return "Worsening"

    if positive >= 2 and negative == 0:
        return "Worsening"

    if negative >= 2 and positive == 0:
        return "Improving"

    if positive > 0 and negative > 0:
        return "Volatile"

    if improving_votes == 1 and worsening_votes == 0 and negative >= 1:
        return "Improving"

    if worsening_votes == 1 and improving_votes == 0 and positive >= 1:
        return "Worsening"

    return "Stable"

def deduplicate_trend_records(records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    ordered = order_trend_records_by_time(records)

    seen = set()
    unique_records: List[Dict[str, Any]] = []
    duplicate_count = 0

    def _round(v):
        try:
            return round(float(v), 5)
        except:
            return None

    for rec in ordered:
        key = (
            str(rec.get("timestamp") or "").strip(),
            _round(rec.get("one_x_amp")),
            _round(rec.get("two_x_amp")),
            _round(rec.get("overall")),
            _round(rec.get("peak_amp")),
        )

        if key in seen:
            duplicate_count += 1
            continue

        seen.add(key)
        unique_records.append(rec)

    return unique_records, duplicate_count

def build_trend_series_table(records: List[Dict[str, Any]]) -> pd.DataFrame:
    ordered, _duplicate_count = deduplicate_trend_records(records)

    if not ordered:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    previous_score: Optional[float] = None

    for idx, rec in enumerate(ordered, start=1):
        ts = parse_trend_timestamp(rec.get("timestamp"))
        score_info = build_trend_condition_summary(
            rec,
            comparability_score=float(rec.get("comparability_score", 100.0) or 100.0),
        )
        score = score_info["severity_raw"]

        delta_score = None
        if previous_score is not None:
            delta_score = score - previous_score

        rows.append(
            {
                "Order": idx,
                "Timestamp": format_trend_timestamp(ts, str(rec.get("timestamp") or "—")),
                "Signal": str(rec.get("name") or "—"),
                "RPM": format_number(rec.get("rpm"), 0),
                "Peak": format_number(rec.get("peak_amp"), 3),
                "Overall": format_number(rec.get("overall"), 3),
                "1X": format_number(rec.get("one_x_amp"), 3),
                "2X": format_number(rec.get("two_x_amp"), 3),
                "High Harmonics": format_number(rec.get("high_harm_amp"), 3),
                "Trend Score": format_number(score, 2),
                "Δ Score vs Prev": format_number(delta_score, 2),
            }
        )

        previous_score = score

    return pd.DataFrame(rows)


def build_trend_assessment(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    ordered, duplicate_count = deduplicate_trend_records(records)

    if len(ordered) < 3:
        return {
            "trend_label": "Insufficient Data",
            "traffic_light": "Yellow",
            "traffic_color": "#f59e0b",
            "headline": "No hay suficientes fechas para análisis multitemporal.",
            "narrative": "Se requieren al menos 3 mediciones únicas para evaluar tendencia multitemporal de condición.",
            "score_deltas": [],
            "first_timestamp": None,
            "last_timestamp": None,
            "days_span": None,
            "top_driver": "—",
            "latest_score": None,
            "duplicate_count": duplicate_count,
            "series_count": len(ordered),
        }

    scores: List[float] = []
    score_deltas: List[float] = []

    for rec in ordered:
        score_info = build_trend_condition_summary(
            rec,
            comparability_score=float(rec.get("comparability_score", 100.0) or 100.0),
        )
        scores.append(float(score_info["severity_raw"]))

    for i in range(1, len(scores)):
        score_deltas.append(scores[i] - scores[i - 1])

    first = ordered[0]
    last = ordered[-1]

    trend_label = _derive_trend_direction(
        score_deltas,
        first_record=first,
        last_record=last,
    )

    first_ts = parse_trend_timestamp(ordered[0].get("timestamp"))
    last_ts = parse_trend_timestamp(ordered[-1].get("timestamp"))
    days_span = None
    if first_ts is not None and last_ts is not None:
        days_span = int((last_ts - first_ts).total_seconds() // 86400)

    first = ordered[0]
    last = ordered[-1]

    one_x_change = safe_pct_change(last.get("one_x_amp"), first.get("one_x_amp"))
    two_x_change = safe_pct_change(last.get("two_x_amp"), first.get("two_x_amp"))
    overall_change = safe_pct_change(last.get("overall"), first.get("overall"))
    high_harm_change = safe_pct_change(last.get("high_harm_amp"), first.get("high_harm_amp"))

    driver_candidates = {
        "1X": {
            "delta_pct": one_x_change,
            "amplitude": last.get("one_x_amp"),
            "priority": 1.00,
        },
        "2X": {
            "delta_pct": two_x_change,
            "amplitude": last.get("two_x_amp"),
            "priority": 0.88,
        },
        "Overall": {
            "delta_pct": overall_change,
            "amplitude": last.get("overall"),
            "priority": 0.82,
        },
        "High Harmonics": {
            "delta_pct": high_harm_change,
            "amplitude": last.get("high_harm_amp"),
            "priority": 0.72,
        },
    }

    top_driver = "—"
    top_driver_val = None
    top_driver_score = None

    for key, payload in driver_candidates.items():
        delta_pct = payload.get("delta_pct")
        amplitude = payload.get("amplitude")
        priority = payload.get("priority", 1.0)

        if delta_pct is None:
            continue

        try:
            delta_abs = abs(float(delta_pct))
        except Exception:
            continue

        try:
            amp_val = abs(float(amplitude)) if amplitude is not None else 0.0
        except Exception:
            amp_val = 0.0

        weighted_score = delta_abs * priority * (1.0 + min(amp_val, 10.0) / 10.0)

        if top_driver_score is None or weighted_score > top_driver_score:
            top_driver_score = weighted_score
            top_driver = key
            top_driver_val = float(delta_pct)

    # Regla editorial/mecánica:
    # si 1X y 2X están muy cerca en % pero 1X domina en amplitud, priorizar 1X
    try:
        one_x_amp = abs(float(last.get("one_x_amp"))) if last.get("one_x_amp") is not None else 0.0
    except Exception:
        one_x_amp = 0.0

    try:
        two_x_amp = abs(float(last.get("two_x_amp"))) if last.get("two_x_amp") is not None else 0.0
    except Exception:
        two_x_amp = 0.0

    try:
        one_x_abs = abs(float(one_x_change)) if one_x_change is not None else None
    except Exception:
        one_x_abs = None

    try:
        two_x_abs = abs(float(two_x_change)) if two_x_change is not None else None
    except Exception:
        two_x_abs = None

    if (
        one_x_abs is not None
        and two_x_abs is not None
        and abs(one_x_abs - two_x_abs) <= 5.0
        and one_x_amp >= (two_x_amp * 3.0)
    ):
        top_driver = "1X"
        top_driver_val = float(one_x_change)

    if trend_label == "Worsening":
        traffic_light = "Red"
        traffic_color = "#dc2626"
        headline = "La condición de la máquina muestra tendencia al deterioro con crecimiento acumulado de componentes dinámicos."
    elif trend_label == "Improving":
        traffic_light = "Green"
        traffic_color = "#16a34a"
        headline = "La condición de la máquina muestra tendencia a la mejora con reducción acumulada de severidad dinámica."
    elif trend_label == "Volatile":
        traffic_light = "Yellow"
        traffic_color = "#f59e0b"
        headline = "La condición de la máquina muestra comportamiento inestable."
    else:
        traffic_light = "Yellow"
        traffic_color = "#f59e0b"
        headline = "La condición de la máquina se mantiene relativamente estable sin cambio acumulado dominante."

    span_text = f" en {days_span} días" if days_span is not None and days_span >= 0 else ""
    driver_text = ""
    if top_driver_val is not None:
        driver_text = f" El componente con mayor cambio acumulado es {top_driver} ({format_number(top_driver_val, 1)}%)."

    duplicate_text = ""
    if duplicate_count > 0:
        duplicate_text = f" Se omitieron {duplicate_count} registros duplicados antes del análisis."

    latest_score = scores[-1] if scores else None

    faults = infer_trend_faults(first, last)

    narrative = (
        f"{headline}{span_text} "
        f"Se evaluaron {len(ordered)} mediciones únicas ordenadas cronológicamente."
        f"{driver_text}"
        f"{duplicate_text} "
        "Esta lectura multitemporal permite distinguir si la condición evoluciona, mejora o fluctúa entre campañas de medición."
    ).strip()

    return {
        "trend_label": trend_label,
        "traffic_light": traffic_light,
        "traffic_color": traffic_color,
        "headline": headline,
        "narrative": narrative,
        "score_deltas": score_deltas,
        "first_timestamp": format_trend_timestamp(first_ts) if first_ts is not None else first.get("timestamp"),
        "last_timestamp": format_trend_timestamp(last_ts) if last_ts is not None else last.get("timestamp"),
        "days_span": days_span,
        "top_driver": top_driver,
        "top_driver_pct": top_driver_val,
        "latest_score": latest_score,
        "primary_fault": faults.get("primary_fault"),
        "secondary_fault": faults.get("secondary_fault"),
        "series_count": len(ordered),
        "duplicate_count": duplicate_count,
    }



def infer_trend_faults(
    first: Dict[str, Any],
    last: Dict[str, Any],
) -> Dict[str, str]:

    def pct(a, b):
        try:
            return ((float(a) - float(b)) / abs(float(b))) * 100.0
        except:
            return None

    one_x = pct(last.get("one_x_amp"), first.get("one_x_amp"))
    two_x = pct(last.get("two_x_amp"), first.get("two_x_amp"))
    overall = pct(last.get("overall"), first.get("overall"))
    high = pct(last.get("high_harm_amp"), first.get("high_harm_amp"))

    primary = "Sin cambio dominante"
    secondary = "Sin patrón secundario claro"

    # --- DESBALANCEO ---
    if one_x is not None and abs(one_x) > 20:
        if one_x > 0:
            primary = "Incremento de desbalanceo"
        else:
            primary = "Reducción de desbalanceo"

    # --- DESALINEACIÓN ---
    if two_x is not None and abs(two_x) > 20:
        secondary = "Comportamiento compatible con desalineación"

    # --- FALLA INCIPIENTE ---
    if high is not None and abs(high) > 25:
        secondary = "Incremento de componentes de alta frecuencia (posible daño incipiente)"

    # --- ENERGÍA GLOBAL ---
    if overall is not None and abs(overall) > 20:
        if overall > 0:
            secondary = "Incremento global de energía"
        else:
            secondary = "Reducción global de energía"

    return {
        "primary_fault": primary,
        "secondary_fault": secondary,
    }


def build_trend_executive_card(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    assessment = build_trend_assessment(records)
    return {
        "headline": str(assessment.get("headline") or "Trend Assessment").strip(),
        "trend_label": str(assessment.get("trend_label") or "—").strip(),
        "traffic_light": str(assessment.get("traffic_light") or "—").strip(),
        "traffic_color": str(assessment.get("traffic_color") or "#64748b").strip(),
        "narrative": str(assessment.get("narrative") or "").strip(),
        "top_driver": str(assessment.get("top_driver") or "—").strip(),
        "top_driver_pct": assessment.get("top_driver_pct"),
        "latest_score": assessment.get("latest_score"),
        "series_count": int(assessment.get("series_count") or 0),
        "first_timestamp": assessment.get("first_timestamp"),
        "last_timestamp": assessment.get("last_timestamp"),
        "days_span": assessment.get("days_span"),
        "duplicate_count": int(assessment.get("duplicate_count") or 0),
    }


def build_trend_report_notes(records: List[Dict[str, Any]]) -> str:
    assessment = build_trend_assessment(records)
    series_df = build_trend_series_table(records)

    blocks: List[str] = []
    blocks.append(str(assessment.get("narrative") or "").strip())

    blocks.append(
        "Interpretación técnica:\n"
        f"{build_trend_editorial_narrative(assessment)}"
    )

    blocks.append(
        "Resumen de tendencia:\n"
        f"- Trend Label: {assessment.get('trend_label') or '—'}\n"
        f"- Semáforo: {assessment.get('traffic_light') or '—'}\n"
        f"- Primer registro: {assessment.get('first_timestamp') or '—'}\n"
        f"- Último registro: {assessment.get('last_timestamp') or '—'}\n"
        f"- Horizonte: {assessment.get('days_span') if assessment.get('days_span') is not None else '—'} días\n"
        f"- Driver dominante: {assessment.get('top_driver') or '—'}\n"
        f"- Cambio del driver: {format_number(assessment.get('top_driver_pct'), 1)}%\n"
        f"- Último trend score: {format_number(assessment.get('latest_score'), 2)}\n"
        f"- Duplicados omitidos: {assessment.get('duplicate_count') or 0}"
    )

    if not series_df.empty:
        lines = []
        for _, row in series_df.iterrows():
            lines.append(
                f"- {row['Timestamp']} | Score {row['Trend Score']} | 1X {row['1X']} | 2X {row['2X']} | Overall {row['Overall']}"
            )
        blocks.append("Serie temporal resumida:\n" + "\n".join(lines))

    return "\n\n".join(block for block in blocks if block.strip())