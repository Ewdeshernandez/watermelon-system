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


def _speed_shaft_label(sensor_label: Optional[str], variable: Optional[str]) -> str:
    """Etiqueta legible del eje a partir del canal (KPH TURBINA / KPH GENERADOR…)."""
    t = f"{sensor_label or ''} {variable or ''}".lower()
    if "turbin" in t:
        return "Turbina"
    if "gener" in t:
        return "Generador"
    if "motor" in t:
        return "Motor"
    if "bomba" in t or "pump" in t:
        return "Bomba"
    return (sensor_label or "").strip() or "Eje"


def pick_speeds(latest: List[Dict[str, Any]]):
    """Velocidad(es) del activo. CLAVE: un tren turbo-generador reporta DOS
    velocidades (KPH TURBINA ~14000 rpm y KPH GENERADOR ~1800 rpm). Tomar la
    'primera' mostraba la del generador (1800) como si fuera la máquina → engañoso.

    Devuelve (primary_txt, all_txt, primary_val):
      · primary_txt: velocidad del EJE PRINCIPAL (la mayor; en turbo-gen = turbina)
      · all_txt: todas las velocidades etiquetadas ('Turbina 14040 · Generador 1800 rpm')
      · primary_val: float de la principal (o None)
    """
    best: Dict[str, float] = {}   # label -> mayor valor por eje (dedupe)
    for r in (latest or []):
        var = (r.get("variable") or "").lower()
        slab = (r.get("sensor_label") or "").lower()
        is_speed = var.startswith("velocidad") or var.startswith("speed") or "kph" in slab
        if not is_speed:
            continue
        v = r.get("value")
        if v is None:
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        lab = _speed_shaft_label(r.get("sensor_label"), r.get("variable"))
        if lab not in best or fv > best[lab]:
            best[lab] = fv
    if not best:
        return "—", "—", None
    items = sorted(best.items(), key=lambda kv: -kv[1])   # mayor primero (turbina)
    primary_val = items[0][1]
    primary_txt = f"{primary_val:.0f} rpm"
    if len(items) == 1:
        return primary_txt, primary_txt, primary_val
    all_txt = " · ".join(f"{lab} {val:.0f}" for lab, val in items) + " rpm"
    return primary_txt, all_txt, primary_val


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


def _local_dt_str(captured_at: Any) -> str:
    """captured_at (UTC) → 'YYYY-MM-DD HH:MM' en hora local del cliente."""
    captured = _parse_captured_at(captured_at)
    if captured is None:
        return ""
    try:
        from zoneinfo import ZoneInfo
        return captured.astimezone(ZoneInfo("America/Bogota")).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return captured.strftime("%Y-%m-%d %H:%M")


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
    alarm_focus: bool = False,
    offline_age_min: Optional[float] = None,
) -> Tuple[Optional[bytes], Dict[str, Any]]:
    """Genera el PDF ejecutivo del activo en forma headless.

    offline_age_min: si se pasa (minutos sin reportar), el reporte se marca
    FUERA DE LÍNEA — banner gris + nota honesta; muestra los ÚLTIMOS datos
    medidos (no simula condición actual). Para activos con servicio contratado
    que están parados o sin enlace (ej. SGT300A).

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
    speed_txt, speed_all, _speed_val = pick_speeds(latest)   # turbina primero (no el generador)
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
    kpis = {"speed": speed_txt, "speeds_all": speed_all,
            "status": status, "alarms": n_danger + n_alarm, "last": last_txt}
    meta = {"instance_id": instance_id, "status": status, "score": score,
            "zone": zone, "alarms": n_danger + n_alarm}

    # FUERA DE LÍNEA — activo con servicio contratado pero sin reportar.
    # Reusa los ÚLTIMOS datos medidos; solo cambia banner/estado a gris honesto.
    if offline_age_min is not None:
        try:
            newest = min(latest, key=lambda r: _seconds_since(r.get("captured_at")))
            since_txt = _local_dt_str(newest.get("captured_at"))
            age_txt = f"hace {_format_age(newest.get('captured_at'))}"
        except Exception:
            since_txt, age_txt = "", f"hace {offline_age_min/60.0:.1f} h"
        health = {"score": score, "zone": "Fuera de línea", "color": "#475569"}
        kpis.update({"status": "Fuera de línea", "offline": True,
                     "offline_since": since_txt, "offline_age": age_txt,
                     "last": age_txt})
        meta.update({"status": "Fuera de línea", "offline": True})

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

    # Reporte de ALARMA (alarm_focus=True, cron de alarmas): la tendencia
    # muestra EXCLUSIVAMENTE los canales en Alarma/Danger, últimos 7 DÍAS
    # (máx por hora), con sus límites de alarma/danger. Ventana ampliada de
    # 48 h → 7 d (2026-09-21, decisión del usuario) para distinguir escalón
    # súbito vs deriva lenta. Usa el rollup horario (instantáneo); si un canal
    # no está en el rollup, cae al bucketed por hora sobre la data cruda.
    trend_png = None
    trend_title = "Tendencia overall"
    if alarm_focus:
        try:
            from datetime import timedelta
            from core.live_readings import history_bucketed, history_rollup
            _alarm_rows = [r for r in rendered_rows
                           if r.get("status") in ("Alarma", "Danger")]
            if _alarm_rows:
                _unit0 = _alarm_rows[0]["unit"]
                _alarm_rows = [r for r in _alarm_rows if r["unit"] == _unit0][:4]
                _var_by_sensor: Dict[str, str] = {}
                for r in latest:
                    if (r.get("sensor_label") and r.get("variable")
                            and (r.get("metric") or "") == "Direct"):
                        _var_by_sensor.setdefault(r["sensor_label"], r["variable"])
                _from_iso = (datetime.now(timezone.utc)
                             - timedelta(days=7)).isoformat()
                _palette = ["#dc2626", "#d97706", "#7c3aed", "#0891b2"]
                _series = []
                for i, rr in enumerate(_alarm_rows):
                    _var = _var_by_sensor.get(rr["sensor_label"])
                    if not _var:
                        continue
                    _pts = history_rollup(instance_id, _var, "Direct",
                                          _from_iso, "1 hour") or []
                    if len([p for p in _pts if p.get("max_val") is not None]) < 2:
                        # Fallback: rollup no cubre este canal → bucketed crudo 1 h
                        _pts = history_bucketed(instance_id, _var, "Direct",
                                                _from_iso, "1 hour") or []
                    xs = [b.get("bucket") for b in _pts
                          if b.get("max_val") is not None]
                    ys = [b.get("max_val") for b in _pts
                          if b.get("max_val") is not None]
                    if len(ys) >= 2:
                        _series.append({"label": rr["sensor_label"], "x": xs,
                                        "y": ys,
                                        "color": _palette[i % len(_palette)]})
                if _series:
                    _rr0 = _alarm_rows[0]
                    trend_png = render_trend_png(
                        _series,
                        alarm=(_rr0.get("alarm_used", 0) or 0),
                        danger=(_rr0.get("danger_used", 0) or 0),
                        y_title=f"{_unit0} (máx / hora)",
                    )
                    if trend_png:
                        trend_title = "Canales en alarma — tendencia últimos 7 días"
        except Exception:
            trend_png = None

    # Tendencia PNG (canales que comparten unidad) — programado o fallback
    try:
        by_unit: Dict[str, List[str]] = {}
        for r in rendered_rows:
            by_unit.setdefault(r["unit"], []).append(r["sensor_label"])
        if by_unit and trend_png is None:
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
        pass  # no pisar una tendencia de alarma ya generada

    try:
        _train = None
        try:                                   # DIAGRAMA LIVE (vector, preferido)
            from core.train_svg import build_train_svg, svg_to_train_drawing
            _svg = build_train_svg(instance_obj, latest, sensor_lookup)
            _train = svg_to_train_drawing(_svg) if _svg else None
        except Exception:  # noqa: BLE001
            _train = None
        schematic_png = None
        if _train is None:                     # fallback: PNG matplotlib
            try:
                from core.briefing_builder import _render_sensor_map
                schematic_png = _render_sensor_map(instance_obj, channels)
            except Exception:  # noqa: BLE001
                schematic_png = None
        pdf_bytes = generate_live_report_pdf(instance_id, instance_obj, health, kpis,
                                             channels, events, trend_png,
                                             trend_title=trend_title,
                                             schematic_png=schematic_png, train_drawing=_train)
        return pdf_bytes, meta
    except Exception:
        return None, meta


__all__ = ["build_report_for_instance", "current_severity_level"]
