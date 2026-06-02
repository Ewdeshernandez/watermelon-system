"""
core.live_report_builder
========================

Generación HEADLESS del reporte ejecutivo PDF de un activo — sin sesión
Streamlit. Lo usa el cron de envíos programados (scripts/send_scheduled_reports.py)
y cualquier proceso que necesite armar el PDF fuera de la página.

La página `pages/02_Live_Monitoring.py` tiene su propia copia de esta lógica
(`_build_live_report_pdf` + helpers) acoplada al render en vivo. Acá la
replicamos en forma PURA. Si cambia la lógica de severidad/health/eventos en
la página, actualizar también este módulo (son fuentes paralelas a propósito
para no importar la página Streamlit en un proceso headless).

API:
    build_report_for_instance(instance_id, instance_obj=None)
        -> (pdf_bytes | None, meta: dict)

    meta = {"instance_id", "status", "score", "zone", "alarms"}
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


# =============================================================
# Helpers de tiempo (puros) — portados de la página
# =============================================================

def _parse_captured_at(captured_at: Any) -> Optional[datetime]:
    if captured_at is None:
        return None
    if isinstance(captured_at, datetime):
        if captured_at.tzinfo is None:
            return captured_at.replace(tzinfo=timezone.utc)
        return captured_at
    if isinstance(captured_at, str):
        try:
            return datetime.fromisoformat(captured_at.replace("Z", "+00:00"))
        except Exception:
            return None
    return None


def _format_age(captured_at: Any) -> str:
    captured = _parse_captured_at(captured_at)
    if captured is None:
        return "—"
    delta = (datetime.now(timezone.utc) - captured).total_seconds()
    if delta < 0:
        return "ahora"
    if delta < 60:
        return f"{int(delta)} s"
    if delta < 3600:
        return f"{int(delta / 60)} min"
    if delta < 86400:
        return f"{int(delta / 3600)} h"
    return f"{int(delta / 86400)} d"


def _seconds_since(captured_at: Any) -> float:
    captured = _parse_captured_at(captured_at)
    if captured is None:
        return 999999.0
    return (datetime.now(timezone.utc) - captured).total_seconds()


# =============================================================
# Severidad / lookup / health / eventos (puros)
# =============================================================

def _compute_severity(value, sensor_match, unit, instance_obj=None):
    from core.severity import compute_severity as _core
    return _core(value=value, sensor_match=sensor_match, unit=unit,
                 instance_obj=instance_obj, sensor_type_hint="")


def _build_sensor_lookup(instance_obj) -> Dict[str, Dict[str, Any]]:
    if instance_obj is None or not getattr(instance_obj, "sensors", None):
        return {}
    try:
        from core.sensor_map import sensor_label as _sensor_label_fn
    except Exception:
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for s in instance_obj.sensors or []:
        try:
            out[_sensor_label_fn(s)] = s
        except Exception:
            continue
    return out


def _compute_rendered_rows(
    latest: List[Dict[str, Any]],
    sensor_lookup: Dict[str, Dict[str, Any]],
    instance_obj: Any = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    direct_rows = [
        r for r in latest
        if r.get("metric") == "Direct"
        and not (r.get("variable") or "").lower().startswith("velocidad")
    ]
    summary = {"Normal": 0, "Alarma": 0, "Danger": 0, "Sin Norma": 0, "No Data": 0}
    rendered: List[Dict[str, Any]] = []
    for r in direct_rows:
        sensor_label = r.get("sensor_label") or "—"
        sensor_match = sensor_lookup.get(sensor_label)
        unit = r.get("unit") or ""
        sev = _compute_severity(r.get("value"), sensor_match, unit, instance_obj)
        summary[sev["status"]] = summary.get(sev["status"], 0) + 1
        rendered.append({
            "sensor_label": sensor_label,
            "plane_label": (sensor_match or {}).get("plane_label", ""),
            "variable": r.get("variable"),
            "value": r.get("value"),
            "unit": unit,
            "age": _format_age(r.get("captured_at", "")),
            "status": sev["status"], "fg": sev["fg"], "bg": sev["bg"],
            "alarm_used": sev["alarm"], "danger_used": sev["danger"],
            "_sort_key": (
                {"Danger": 0, "Alarma": 1, "Sin Norma": 2, "Normal": 3, "No Data": 4}.get(sev["status"], 9),
                sensor_label,
            ),
        })
    rendered.sort(key=lambda r: r["_sort_key"])
    return rendered, summary


def _compute_health_score(
    severity_summary: Optional[Dict[str, int]],
    latest: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[Optional[int], str, str]:
    if not latest:
        return None, "Sin datos", "#94a3b8"
    s = severity_summary or {}
    n_normal = s.get("Normal", 0)
    n_alarm = s.get("Alarma", 0)
    n_danger = s.get("Danger", 0)
    n_eval = n_normal + n_alarm + n_danger
    if n_eval == 0:
        return None, "Sin norma", "#94a3b8"
    penalty = (n_alarm * 18 + n_danger * 45) / n_eval
    score = int(round(max(0.0, 100.0 - penalty)))
    if n_danger > 0:
        score = min(score, 49)
    elif n_alarm > 0:
        score = min(score, 74)
    if score >= 90:
        return score, "Zona A · Normal", "#1D9E75"
    if score >= 75:
        return score, "Zona B · Vigilancia", "#1D9E75"
    if score >= 50:
        return score, "Zona C · Alerta", "#EF9F27"
    return score, "Zona D · Peligro", "#E24B4A"


def _detect_severity_events(
    spark_data: Dict[str, List[Dict[str, Any]]],
    sensor_lookup: Dict[str, Dict[str, Any]],
    instance_obj: Any = None,
    max_events: int = 8,
) -> List[Dict[str, Any]]:
    rank = {"Normal": 0, "Sin Norma": 0, "No Data": 0, "Alarma": 1, "Danger": 2}
    events: List[Dict[str, Any]] = []
    for sensor_label, history in (spark_data or {}).items():
        if not history or len(history) < 2:
            continue
        sensor_match = sensor_lookup.get(sensor_label)
        prev_status: Optional[str] = None
        for h in history:
            val = h.get("value")
            unit = h.get("unit") or ""
            if val is None:
                continue
            sev = _compute_severity(val, sensor_match, unit, instance_obj)
            status = sev["status"]
            if prev_status is not None and rank.get(status, 0) != rank.get(prev_status, 0):
                if rank.get(status, 0) > 0 or rank.get(prev_status, 0) > 0:
                    rising = rank.get(status, 0) > rank.get(prev_status, 0)
                    events.append({
                        "sensor_label": sensor_label, "from": prev_status,
                        "to": status, "rising": rising, "value": val, "unit": unit,
                        "captured_at": h.get("captured_at"),
                        "fg": sev["fg"], "bg": sev["bg"],
                    })
            prev_status = status
    events.sort(key=lambda e: e.get("captured_at") or "", reverse=True)
    return events[:max_events]


# =============================================================
# Armado del PDF (espejo de _build_live_report_pdf de la página)
# =============================================================

def current_severity_level(
    instance_id: str,
    instance_obj: Any = None,
) -> Tuple[int, str, Dict[str, int]]:
    """Nivel de severidad ACTUAL del activo SIN armar el PDF (barato — solo
    lee latest + computa severidad). Para el cron de alarmas, que chequea cada
    15 min y solo arma el PDF si hay que avisar.

    Devuelve (level, status, summary):
        level: 0 = Normal / Sin datos · 1 = Alarma · 2 = Danger
    """
    from core.live_readings import latest_for_instance
    if instance_obj is None:
        from core.instance_state import get_instance
        instance_obj = get_instance(instance_id)
    latest = latest_for_instance(instance_id) or []
    if not latest:
        return 0, "Sin datos", {}
    sensor_lookup = _build_sensor_lookup(instance_obj)
    _, summary = _compute_rendered_rows(latest, sensor_lookup, instance_obj)
    if summary.get("Danger", 0):
        return 2, "Crítica", summary
    if summary.get("Alarma", 0):
        return 1, "Atención", summary
    return 0, "Operación normal", summary


def build_report_for_instance(
    instance_id: str,
    instance_obj: Any = None,
) -> Tuple[Optional[bytes], Dict[str, Any]]:
    """Genera el PDF ejecutivo del activo en forma headless.

    Devuelve (pdf_bytes, meta). Si no hay lecturas o falla, (None, meta_parcial)."""
    from core.live_readings import latest_for_instance, recent_history_all_direct
    from core.live_report_pdf import generate_live_report_pdf, render_trend_png

    if instance_obj is None:
        from core.instance_state import get_instance
        instance_obj = get_instance(instance_id)

    latest = latest_for_instance(instance_id) or []
    if not latest:
        return None, {"instance_id": instance_id, "status": "Sin datos",
                      "score": None, "zone": "Sin datos", "alarms": 0}

    sensor_lookup = _build_sensor_lookup(instance_obj)
    rendered_rows, severity_summary = _compute_rendered_rows(latest, sensor_lookup, instance_obj)
    spark_data = recent_history_all_direct(instance_id, n_per_sensor=30) or {}

    # Health + KPIs
    score, zone, zcolor = _compute_health_score(severity_summary, latest)
    speed_row = next((r for r in latest
                      if (r.get("variable") or "").lower().startswith("velocidad")), None)
    speed_txt = (f"{float(speed_row['value']):.0f} rpm"
                 if speed_row and speed_row.get("value") is not None else "—")
    n_danger = severity_summary.get("Danger", 0)
    n_alarm = severity_summary.get("Alarma", 0)
    status = "Crítica" if n_danger else ("Atención" if n_alarm else "Operación normal")
    last_txt = "—"
    try:
        oldest = min(latest, key=lambda r: _seconds_since(r.get("captured_at")))
        last_txt = f"hace {_format_age(oldest.get('captured_at'))}"
    except Exception:
        pass

    health = {"score": score, "zone": zone, "color": zcolor}
    kpis = {"speed": speed_txt, "status": status, "alarms": n_danger + n_alarm, "last": last_txt}
    meta = {"instance_id": instance_id, "status": status, "score": score,
            "zone": zone, "alarms": n_danger + n_alarm}

    # Canales con 1X/2X
    vec: Dict[str, Dict[str, Any]] = {}
    for r in latest:
        s, m = r.get("sensor_label"), r.get("metric")
        if s and m in ("1X_Ampl", "1X_Phase", "2X_Ampl", "2X_Phase"):
            vec.setdefault(s, {})[m] = r.get("value")

    def _a(v):
        try:
            return f"{float(v):.2f}" if v is not None and float(v) >= 1e-4 else "—"
        except Exception:
            return "—"

    def _p(a, p):
        try:
            if a is None or float(a) < 1e-4 or p is None or abs(float(p)) < 1e-30:
                return "—"
            return f"{float(p):.0f}°"
        except Exception:
            return "—"

    channels = []
    for r in rendered_rows:
        sl = r["sensor_label"]
        v = vec.get(sl, {})
        try:
            val = f"{float(r['value']):.2f}" if r["value"] is not None else "—"
        except Exception:
            val = "—"
        channels.append({
            "sensor_label": sl, "plane_label": r.get("plane_label") or "—",
            "value": val, "unit": r["unit"], "status": r["status"],
            "x1_amp": _a(v.get("1X_Ampl")), "x1_ph": _p(v.get("1X_Ampl"), v.get("1X_Phase")),
            "x2_amp": _a(v.get("2X_Ampl")), "x2_ph": _p(v.get("2X_Ampl"), v.get("2X_Phase")),
        })

    # Eventos
    def _ev_val(v):
        try:
            return f"{float(v):.2f}"
        except Exception:
            return "—"
    ev = _detect_severity_events(spark_data, sensor_lookup, instance_obj, max_events=8)
    events = [{"sensor_label": e["sensor_label"], "to": e["to"], "value": _ev_val(e["value"]),
               "unit": e["unit"], "age": _format_age(e.get("captured_at", "")), "rising": e["rising"]}
              for e in ev]

    # Tendencia PNG (canales que comparten unidad)
    trend_png = None
    try:
        by_unit: Dict[str, List[str]] = {}
        for r in rendered_rows:
            by_unit.setdefault(r["unit"], []).append(r["sensor_label"])
        if by_unit:
            unit_grp = max(by_unit.values(), key=len)[:4]
            palette = ["#1e40af", "#0891b2", "#7c3aed", "#be185d"]
            series = []
            for i, sl in enumerate(unit_grp):
                hist = spark_data.get(sl, [])
                xs = [h.get("captured_at") for h in hist if h.get("value") is not None]
                ys = [h.get("value") for h in hist if h.get("value") is not None]
                if len(ys) >= 2:
                    series.append({"label": sl, "x": xs, "y": ys, "color": palette[i % len(palette)]})
            if series:
                rr0 = next((r for r in rendered_rows if r["sensor_label"] in unit_grp), None)
                trend_png = render_trend_png(
                    series,
                    alarm=(rr0.get("alarm_used", 0) or 0) if rr0 else 0,
                    danger=(rr0.get("danger_used", 0) or 0) if rr0 else 0,
                    y_title=rr0["unit"] if rr0 else "valor",
                )
    except Exception:
        trend_png = None

    try:
        pdf_bytes = generate_live_report_pdf(instance_id, instance_obj, health, kpis,
                                             channels, events, trend_png)
        return pdf_bytes, meta
    except Exception:
        return None, meta


__all__ = ["build_report_for_instance", "current_severity_level"]
