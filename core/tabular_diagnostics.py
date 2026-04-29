from __future__ import annotations

from typing import Dict, List

import pandas as pd

SAFE_COLOR = "#16a34a"
WARNING_COLOR = "#f59e0b"
DANGER_COLOR = "#dc2626"


def build_tabular_report_notes(text_diag: Dict[str, str]) -> str:
    headline = str(text_diag.get("headline", "") or "").strip()
    narrative = str(text_diag.get("narrative", "") or "").strip()

    blocks = []
    if headline:
        blocks.append(headline)
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

    # =========================================================
    # Ciclo 14c.3 — análisis enriquecido Cat IV
    # =========================================================

    # Distribución por familia de sensor
    family_series = work.get("Family", pd.Series(dtype=str)).astype(str).fillna("")
    family_counts: Dict[str, int] = {}
    for f in family_series:
        f_clean = f.strip()
        if not f_clean:
            continue
        family_counts[f_clean] = family_counts.get(f_clean, 0) + 1
    family_summary_parts = [f"{c} {f}" for f, c in sorted(family_counts.items(), key=lambda x: -x[1])]
    family_summary = " + ".join(family_summary_parts) if family_summary_parts else "sin clasificación de familia"

    # Citas a normas según familias presentes
    norms_cited = set()
    fams_lower = {k.lower() for k in family_counts.keys()}
    if "proximity" in fams_lower:
        norms_cited.add("API 670 (sondas de proximidad)")
        norms_cited.add("ISO 7919-3 / ISO 20816-3 (shaft displacement)")
    if "velocity" in fams_lower:
        norms_cited.add("ISO 20816-3 (casing velocity)")
    if "acceleration" in fams_lower:
        norms_cited.add("ISO 20816-3 / ISO 13373-1 (envelope acceleration)")
    norms_text = " · ".join(sorted(norms_cited)) if norms_cited else "ISO 20816-3"

    # Margen de severidad de los puntos críticos (% del danger)
    severity_summary = ""
    if danger_count > 0 or alarm_count > 0:
        crit_df = work[work["Status"].isin(["Alarm", "Danger"])].copy()
        crit_df["margin_pct"] = (
            pd.to_numeric(crit_df["Overall"], errors="coerce")
            / pd.to_numeric(crit_df["Danger"], errors="coerce").replace(0, pd.NA)
            * 100.0
        )
        crit_df = crit_df.sort_values("margin_pct", ascending=False).head(3)
        margin_lines = []
        for _, row in crit_df.iterrows():
            mp = row.get("margin_pct")
            if mp is None or pd.isna(mp):
                continue
            margin_lines.append(
                f"{row['Machine']}/{row['Point']} a {float(mp):.0f}% del danger"
            )
        if margin_lines:
            severity_summary = "; ".join(margin_lines)

    # Recomendaciones priorizadas (lista numerada estilo Cat IV)
    recommendations: List[str] = [] if False else []  # type ignore
    recommendations: list = []

    if danger_count > 0:
        recommendations.append(
            "PRIORIDAD CRÍTICA: programar verificación inmediata de los puntos "
            "en zona Danger. Restringir operación sostenida hasta confirmar "
            "diagnóstico. Correlacionar con Time Waveform (impactos), Spectrum "
            "(firmas mecánicas) y Polar/Bode 1X antes de cualquier intervención."
        )
    if alarm_count > 0:
        recommendations.append(
            "Investigar puntos en zona Alarm: comparar contra histórico (Trends) "
            "para distinguir condición transitoria de degradación sostenida."
        )

    if primary_pattern == "1X":
        recommendations.append(
            "Verificar condición de balanceo según ISO 21940-12 nivel G 2.5 "
            "para turbomaquinaria de proceso. Confirmar consistencia de fase "
            "entre arranques antes de programar balanceo en sitio."
        )
    elif primary_pattern == "2X":
        recommendations.append(
            "Inspeccionar alineación del tren acoplado (eje del driver respecto "
            "al driven) según API 686. Verificar condición del acople: "
            "desgaste de dientes, juego radial, deformación de elementos "
            "elastoméricos."
        )
    elif primary_pattern == "0.5X":
        recommendations.append(
            "Investigar inestabilidad sub-síncrona (oil whirl / oil whip / rub) "
            "según API 684. Validar régimen hidrodinámico del cojinete, "
            "presiones de aceite, holguras radiales y temperatura de babbitt."
        )

    if status == "SAFE":
        recommendations.append(
            "Mantener la frecuencia actual de monitoreo y conservar este "
            "reporte como línea base de aceptación para comparaciones en "
            "próximas corridas."
        )

    rec_block = ""
    if recommendations:
        rec_block = "\n\nRecomendaciones técnicas priorizadas:\n" + "\n".join(
            f"{i + 1}. {r}" for i, r in enumerate(recommendations)
        )

    # Narrativa final por status
    base_intro = (
        f"El módulo Tabular List analizó {signal_count} señales de vibración "
        f"correspondientes a {machine_scope}, distribuidas por familia como "
        f"{family_summary}. La evaluación se realizó conforme a los criterios "
        f"técnicos aplicables: {norms_text}. "
    )

    base_status = (
        f"Distribución global de severidad: {normal_count} señal(es) en "
        f"CONDICIÓN ACEPTABLE, {alarm_count} en ATENCIÓN, {danger_count} en "
        f"ACCIÓN REQUERIDA / CRÍTICA. "
    )

    if status == "SAFE":
        headline = "Condición global ACEPTABLE en todos los puntos analizados"
        narrative = (
            base_intro + base_status +
            f"{pattern_text} La firma armónica predominante es {primary_pattern}, "
            f"sin excursiones por encima de los setpoints individuales del "
            f"Sensor Map." + rec_block
        )
    elif status == "WARNING":
        headline = f"Señales en zona ATENCIÓN — {pattern_headline.lower()}"
        crit_text = f"Activos con mayor margen consumido: {severity_summary}." if severity_summary else f"Activos relevantes: {top_assets}."
        narrative = (
            base_intro + base_status +
            crit_text + " " + pattern_text + rec_block
        )
    else:
        headline = f"Señales en zona DANGER — {pattern_headline.lower()}"
        crit_text = f"Activos críticos con mayor margen consumido: {severity_summary}." if severity_summary else f"Activos críticos: {top_assets}."
        narrative = (
            base_intro + base_status +
            crit_text + " " + pattern_text + rec_block
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
        "family_summary": family_summary,
        "norms_cited": norms_text,
    }
