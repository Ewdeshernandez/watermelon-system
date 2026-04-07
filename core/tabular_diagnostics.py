from __future__ import annotations

from typing import Dict

import pandas as pd

SAFE_COLOR = "#16a34a"
WARNING_COLOR = "#f59e0b"
DANGER_COLOR = "#dc2626"


def build_tabular_report_notes(text_diag: Dict[str, str]) -> str:
    headline = str(text_diag.get("headline", "") or "").strip()
    narrative = str(text_diag.get("narrative", "") or "").strip()

    blocks = []
    if headline:
        blocks.append(f"Resumen diagnóstico: {headline}")
    if narrative:
        blocks.append(narrative)
    return "\n\n".join(blocks).strip()


def _safe_float(v) -> float:
    try:
        return float(v)
    except Exception:
        return 0.0


def _top_assets(df: pd.DataFrame, n: int = 3) -> str:
    if df.empty:
        return "sin activos críticos identificados"

    d = df.copy()
    d["Overall_num"] = pd.to_numeric(d["Overall"], errors="coerce")
    d["Danger_num"] = pd.to_numeric(d["Danger"], errors="coerce")
    d["severity_ratio"] = d["Overall_num"] / d["Danger_num"].replace(0, pd.NA)
    d["severity_ratio"] = pd.to_numeric(d["severity_ratio"], errors="coerce").fillna(0.0)

    d = d.sort_values(["severity_ratio", "Overall_num"], ascending=False)
    top = d.head(n)

    items = []
    for _, row in top.iterrows():
        items.append(f"{row['Machine']} / {row['Point']}")
    return ", ".join(items) if items else "sin activos críticos identificados"


def _machine_scope_text(df: pd.DataFrame) -> str:
    machines = sorted({str(x) for x in df["Machine"].dropna().astype(str).tolist() if str(x).strip()})
    if not machines:
        return "los equipos analizados"
    if len(machines) == 1:
        return f"la máquina {machines[0]}"
    if len(machines) <= 3:
        return "las máquinas " + ", ".join(machines)
    return f"{len(machines)} máquinas analizadas"


def evaluate_tabular_diagnostic(df: pd.DataFrame) -> Dict[str, str]:
    if df is None or df.empty:
        return {
            "status": "SAFE",
            "color": SAFE_COLOR,
            "headline": "Sin datos disponibles para diagnóstico",
            "narrative": (
                "La tabla no contiene señales válidas para generar un resumen automático. "
                "Se recomienda verificar la carga de señales y la configuración del módulo."
            ),
            "normal_count": 0,
            "alarm_count": 0,
            "danger_count": 0,
            "primary_pattern": "Sin datos",
        }

    work = df.copy()

    for col in ["Overall", "Danger", "Alarm", "0.5X Amp", "1X Amp", "2X Amp"]:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    status_series = work["Status"].astype(str).fillna("No Data")
    normal_count = int((status_series == "Normal").sum())
    alarm_count = int((status_series == "Alarm").sum())
    danger_count = int((status_series == "Danger").sum())

    sig_counts = {"0.5X": 0, "1X": 0, "2X": 0}
    valid_rows = 0

    for _, row in work.iterrows():
        amps = {
            "0.5X": _safe_float(row.get("0.5X Amp")),
            "1X": _safe_float(row.get("1X Amp")),
            "2X": _safe_float(row.get("2X Amp")),
        }
        max_key = max(amps, key=amps.get)
        max_val = amps[max_key]
        if max_val > 0:
            sig_counts[max_key] += 1
            valid_rows += 1

    primary_pattern = "Sin firma dominante" if valid_rows == 0 else max(sig_counts, key=sig_counts.get)

    critical_df = work[work["Status"].isin(["Alarm", "Danger"])].copy()
    top_assets = _top_assets(critical_df, n=3)
    machine_scope = _machine_scope_text(work)
    signal_count = len(work)

    if danger_count > 0:
        status = "DANGER"
        color = DANGER_COLOR
    elif alarm_count > 0:
        status = "WARNING"
        color = WARNING_COLOR
    else:
        status = "SAFE"
        color = SAFE_COLOR

    if primary_pattern == "1X":
        pattern_headline = "Predomina una firma tipo desbalance"
        pattern_text = (
            "La mayor parte de las señales presenta predominio en 1X, comportamiento consistente con una condición "
            "tipo desbalance cuando esta firma se acompaña de bajo contenido relativo en otros armónicos."
        )
        recommendation = (
            "Se recomienda verificar condición de balanceo, revisar consistencia de fase entre arranques "
            "y correlacionar con los módulos Polar y Bode antes de definir una intervención."
        )
    elif primary_pattern == "2X":
        pattern_headline = "Predomina una firma tipo desalineación"
        pattern_text = (
            "La mayor parte de las señales presenta una contribución relevante en 2X, lo cual puede indicar "
            "desalineación o comportamiento asociado al tren de potencia."
        )
        recommendation = (
            "Se recomienda revisar alineación, condición del acople y correlacionar con mediciones radiales, "
            "axiales y análisis de fase."
        )
    elif primary_pattern == "0.5X":
        pattern_headline = "Predomina una firma subarmónica"
        pattern_text = (
            "La tabla muestra presencia dominante de 0.5X en varias señales, lo cual puede estar asociado "
            "a inestabilidad, holgura o fenómenos no lineales dependiendo de la máquina."
        )
        recommendation = (
            "Se recomienda investigar inestabilidad subarmónica, posible holgura o interacción con el proceso, "
            "y validar con forma de onda y espectro detallado."
        )
    else:
        pattern_headline = "No se identifica una firma dominante única"
        pattern_text = (
            "La tabla no muestra una sola firma armónica claramente dominante entre 0.5X, 1X y 2X; "
            "la condición parece mixta o distribuida entre varios mecanismos."
        )
        recommendation = (
            "Se recomienda mantener la tendencia, revisar los puntos más cargados y correlacionar con los demás módulos "
            "antes de concluir un mecanismo de falla."
        )

    if status == "SAFE":
        headline = "Condición general estable en Tabular List"
        narrative = (
            f"Se analizaron {signal_count} señales de vibración correspondientes a {machine_scope}. "
            f"Actualmente {normal_count} señales se encuentran en condición Normal, {alarm_count} en Alarm "
            f"y {danger_count} en Danger. {pattern_text} {recommendation}"
        )
    elif status == "WARNING":
        headline = f"Se identifican señales en alarma. {pattern_headline}"
        narrative = (
            f"Se analizaron {signal_count} señales de vibración correspondientes a {machine_scope}. "
            f"Actualmente {normal_count} señales se encuentran en condición Normal, {alarm_count} en Alarm "
            f"y {danger_count} en Danger. Los activos que requieren mayor atención son: {top_assets}. "
            f"{pattern_text} {recommendation}"
        )
    else:
        headline = f"Se identifican señales en peligro. {pattern_headline}"
        narrative = (
            f"Se analizaron {signal_count} señales de vibración correspondientes a {machine_scope}. "
            f"Actualmente {normal_count} señales se encuentran en condición Normal, {alarm_count} en Alarm "
            f"y {danger_count} en Danger. Los activos más críticos son: {top_assets}. "
            f"{pattern_text} Se recomienda atender primero los puntos en Danger y validar la condición mecánica "
            f"antes de operación repetida. {recommendation}"
        )

    return {
        "status": status,
        "color": color,
        "headline": headline,
        "narrative": narrative,
        "normal_count": normal_count,
        "alarm_count": alarm_count,
        "danger_count": danger_count,
        "primary_pattern": primary_pattern,
    }
