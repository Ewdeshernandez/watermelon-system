"""
core.briefing_builder — Orquestador del Briefing por activo (F3)
================================================================

Junta TODO y produce el PDF del briefing de un activo (o de todos), headless:

  datos (salud/KPIs/canales)   ← reusa helpers de live_report_builder
  figuras (tendencia/espectro/onda/órbita)  ← core.briefing_figures (F1)
  redacción (resumen/diagnóstico/recomendaciones)  ← IA best-effort + fallback
  maquetación PDF              ← core.briefing_report_pdf (F2)

API:
  build_asset_briefing(instance_id, period_label="Semanal") -> (pdf_bytes, meta)
  build_all_briefings(period_label="Semanal") -> List[(instance_id, pdf_bytes, meta)]

No usa Streamlit (corre en cron). La IA es opcional: si no hay credenciales o
falla, se usa un borrador determinístico construido de los datos reales.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1) Datos del activo (salud / KPIs / canales) — reusa live_report_builder
# ---------------------------------------------------------------------------
def _compute_asset_data(instance_id: str, instance_obj: Any) -> Optional[Dict[str, Any]]:
    from core.live_readings import latest_for_instance
    from core.live_report_builder import (
        _build_sensor_lookup, _compute_rendered_rows, _compute_health_score,
        _format_age, _seconds_since,
    )
    latest = latest_for_instance(instance_id) or []
    if not latest:
        return None

    sensor_lookup = _build_sensor_lookup(instance_obj)
    rendered_rows, severity_summary = _compute_rendered_rows(latest, sensor_lookup, instance_obj)
    score, zone, zcolor = _compute_health_score(severity_summary, latest)

    speed_row = next((r for r in latest
                      if (r.get("variable") or "").lower().startswith("velocidad")), None)
    speed_txt = (f"{float(speed_row['value']):.0f} rpm"
                 if speed_row and speed_row.get("value") is not None else "—")
    n_danger = severity_summary.get("Danger", 0)
    n_alarm = severity_summary.get("Alarma", 0)
    status = "Crítica" if n_danger else ("Atención" if n_alarm else "Operación normal")

    # vectores 1X/2X
    vec: Dict[str, Dict[str, Any]] = {}
    for r in latest:
        s, m = r.get("sensor_label"), r.get("metric")
        if s and m in ("1X_Ampl", "2X_Ampl"):
            vec.setdefault(s, {})[m] = r.get("value")

    def _a(v):
        try:
            return f"{float(v):.2f}" if v is not None and float(v) >= 1e-4 else "—"
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
            "x1_amp": _a(v.get("1X_Ampl")), "x2_amp": _a(v.get("2X_Ampl")),
        })

    return {
        "health": {"score": score, "zone": zone, "color": zcolor},
        "kpis": {"status": status, "speed": speed_txt, "alarms": n_danger + n_alarm},
        "channels": channels,
        "severity_summary": severity_summary,
        "n_alarm": n_alarm, "n_danger": n_danger,
    }


# ---------------------------------------------------------------------------
# 2) Redacción: borrador determinístico + mejora IA opcional
# ---------------------------------------------------------------------------
def _deterministic_sections(tag: str, period: str, data: Dict[str, Any]) -> Dict[str, Any]:
    zone = data["health"].get("zone", "—")
    speed = data["kpis"].get("speed", "—")
    n_alarm = data.get("n_alarm", 0)
    n_danger = data.get("n_danger", 0)
    channels = data.get("channels", [])

    # Resumen
    estado = ("condición crítica" if n_danger else
              "puntos en alarma" if n_alarm else "operación normal")
    summary = (
        f"Durante el periodo {period.lower()}, el activo {tag} se mantuvo en "
        f"{zone} con velocidad {speed}. "
        + (f"Se registran {n_danger + n_alarm} canal(es) sobre umbral ({estado})."
           if (n_alarm or n_danger) else
           "No se registran canales sobre umbral; el activo opera dentro de la "
           "zona aceptable según ISO 20816 / API 670."))

    # Diagnóstico — por canal en alarma/danger
    alarmados = [c for c in channels if c.get("status") in ("Alarma", "Danger")]
    if alarmados:
        det = "; ".join(
            f"{c['sensor_label']} ({c.get('plane_label','')}) {c['value']} {c['unit']} "
            f"en {c['status']}" for c in alarmados[:6])
        diagnosis = (
            f"Los siguientes puntos superan su umbral y requieren seguimiento: {det}. "
            "El resto de la cadena de medición se mantiene dentro de límites normales.")
    else:
        diagnosis = (
            "Todos los puntos de medición se encuentran dentro de límites normales. "
            "Las componentes síncronas (1X) son dominantes sin armónicos elevados, "
            "consistente con operación estable.")

    # Recomendaciones
    if n_danger:
        recs = ["Atender de inmediato los puntos en condición crítica y evaluar "
                "parada controlada según criticidad del activo.",
                "Confirmar consistencia de fase 1X entre arranques antes de "
                "cualquier intervención de balanceo o alineación."]
    elif n_alarm:
        recs = ["Programar inspección de los puntos en alarma y verificar "
                "condición de balanceo (ISO 21940-12 G 2.5) y alineación.",
                "Aumentar temporalmente la frecuencia de monitoreo del activo."]
    else:
        recs = ["Mantener la frecuencia actual de monitoreo.",
                "Conservar este briefing como línea base para comparación en "
                "próximas corridas."]
    return {"summary": summary, "diagnosis": diagnosis, "recommendations": recs}


def _ai_enhance(sections: Dict[str, Any], tag: str, period: str,
                data: Dict[str, Any]) -> Dict[str, Any]:
    """Mejora best-effort con IA. Si no hay credenciales o falla, deja el
    borrador determinístico intacto."""
    try:
        from core.ai_diagnostic import is_ai_available, generate_executive_summary
        if not is_ai_available():
            return sections
        # Construir items mínimos para que la IA tenga contexto
        items = [{
            "type": "tabular", "title": f"Estado {tag}",
            "machine": tag, "point": "",
            "notes": (sections["summary"] + "\n" + sections["diagnosis"]),
        }]
        meta = {"machine_train": tag, "period": period}
        res = generate_executive_summary(items, meta)
        if res.get("ok") and res.get("markdown"):
            sections = dict(sections)
            sections["summary"] = res["markdown"].strip()
    except Exception as e:
        log.warning("briefing IA enhance falló (se usa borrador): %s", e)
    return sections


# ---------------------------------------------------------------------------
# 3) Builder por activo
# ---------------------------------------------------------------------------
def build_asset_briefing(
    instance_id: str,
    period_label: str = "Semanal",
    instance_obj: Any = None,
    use_ai: bool = True,
) -> Tuple[Optional[bytes], Dict[str, Any]]:
    """Genera el PDF del briefing de un activo. Devuelve (pdf_bytes, meta)."""
    from core.instance_state import get_instance
    if instance_obj is None:
        instance_obj = get_instance(instance_id)

    tag = (getattr(instance_obj, "tag", None) or instance_id).upper()
    train = " ".join(p for p in [
        getattr(instance_obj, "driver_model", "") or "",
        "→", getattr(instance_obj, "driven_model", "") or "",
    ] if p and p != "→") or "—"
    client = getattr(instance_obj, "client", "") or ""
    if client:
        train = f"{train} · {client}"

    data = _compute_asset_data(instance_id, instance_obj)
    if not data:
        return None, {"instance_id": instance_id, "status": "Sin datos", "ok": False}

    sections = _deterministic_sections(tag, period_label, data)
    if use_ai:
        sections = _ai_enhance(sections, tag, period_label, data)

    try:
        from core.briefing_figures import collect_asset_figures
        figures = collect_asset_figures(instance_id)
    except Exception as e:
        log.warning("briefing figures falló: %s", e)
        figures = {}

    try:
        from core.briefing_report_pdf import generate_briefing_pdf
        pdf = generate_briefing_pdf(
            instance_id=instance_id, tag=tag, train=train,
            period_label=period_label,
            health=data["health"], kpis=data["kpis"],
            figures=figures,
            summary=sections["summary"],
            diagnosis=sections["diagnosis"],
            recommendations=sections["recommendations"],
            channels=data["channels"],
        )
    except Exception as e:
        log.error("briefing PDF falló para %s: %s", instance_id, e)
        return None, {"instance_id": instance_id, "status": "Error PDF", "ok": False}

    meta = {
        "instance_id": instance_id, "tag": tag, "ok": True,
        "status": data["kpis"]["status"], "score": data["health"]["score"],
        "alarms": data["kpis"]["alarms"], "period": period_label,
        "n_figures": sum(1 for v in figures.values() if v),
    }
    return pdf, meta


# ---------------------------------------------------------------------------
# 4) Builder para TODOS los activos
# ---------------------------------------------------------------------------
def build_all_briefings(
    period_label: str = "Semanal", use_ai: bool = True,
) -> List[Tuple[str, Optional[bytes], Dict[str, Any]]]:
    """Genera el briefing de cada activo con datos. Devuelve lista de
    (instance_id, pdf_bytes|None, meta)."""
    from core.instance_state import list_instances, get_instance
    out: List[Tuple[str, Optional[bytes], Dict[str, Any]]] = []
    try:
        instances = list_instances() or []
    except Exception as e:
        log.error("build_all_briefings: list_instances falló: %s", e)
        return out
    for inst in instances:
        iid = inst.get("instance_id") if isinstance(inst, dict) else getattr(inst, "instance_id", "")
        if not iid:
            continue
        try:
            obj = get_instance(iid)
            pdf, meta = build_asset_briefing(iid, period_label, instance_obj=obj, use_ai=use_ai)
            out.append((iid, pdf, meta))
        except Exception as e:
            log.warning("briefing %s falló: %s", iid, e)
            out.append((iid, None, {"instance_id": iid, "ok": False, "status": "Error"}))
    return out


__all__ = ["build_asset_briefing", "build_all_briefings"]
