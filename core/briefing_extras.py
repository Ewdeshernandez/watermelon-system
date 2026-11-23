"""
core/briefing_extras.py
=======================

Secciones nuevas del reporte Semanal / Mensual por activo (sin quitar nada de
lo que ya trae):

  · Franja de indicadores (misma del reporte diario): velocidades por eje,
    vibración de eje máxima, última lectura.
  · Puntos a vigilar: las 3 sondas más cerca de su alerta + su cambio.
  · Estadística del periodo por sonda: máximo, promedio y mínimo EN OPERACIÓN
    (eje principal ≥ 90 % de su velocidad normal — la misma regla de la app
    móvil), % de alerta y Δ contra el periodo anterior.
  · Eventos de alarma del periodo resumidos por sonda.
  · Solo mensual: operación del mes (horas en operación, paradas, arranques,
    disponibilidad).

Todo sale del rollup horario (trend_rollup) y de severity_events: no cambia
datos, solo los pone en contexto. Funciones puras separadas de las que leen
la base para poder probarlas.
"""

from __future__ import annotations

import logging
import statistics
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

log = logging.getLogger(__name__)

TURNING_RPM = 100.0      # por debajo: eje parado
RUNNING_FRAC = 0.90      # en operación = ≥ 90 % de la velocidad normal


# ---------------------------------------------------------------------------
# Puras (testeables)
# ---------------------------------------------------------------------------

def _ts(bucket: Any) -> Optional[datetime]:
    try:
        d = datetime.fromisoformat(str(bucket).replace("Z", "+00:00"))
        return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
    except Exception:  # noqa: BLE001
        return None


def _f(v: Any) -> Optional[float]:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if x == x else None    # NaN fuera


def running_hours(rpm_pts: List[Dict[str, Any]]) -> Tuple[set, float]:
    """(set de timestamps en operación, velocidad normal) desde la serie
    horaria de velocidad del eje principal."""
    pts = [(_ts(p.get("bucket")), _f(p.get("avg_val"))) for p in rpm_pts or []]
    pts = [(t, v) for t, v in pts if t is not None and v is not None]
    turning = sorted(v for _, v in pts if v > TURNING_RPM)
    if not turning:
        return set(), 0.0
    normal = turning[len(turning) // 2]
    return {t for t, v in pts if v >= RUNNING_FRAC * normal}, normal


def operation_summary(rpm_pts: List[Dict[str, Any]], bucket_h: float = 1.0) -> Dict[str, Any]:
    """Horas en operación, paradas, arranques y disponibilidad del periodo.

    · en operación = ≥ 90 % de la velocidad normal
    · girando = > 100 rpm (incluye arranque / parada a medias)
    · arranque = paso de parado → girando; parada = girando → parado
    · disponibilidad = horas girando / horas con dato
    """
    pts = sorted(((_ts(p.get("bucket")), _f(p.get("avg_val"))) for p in rpm_pts or []),
                 key=lambda x: x[0] or datetime.min.replace(tzinfo=timezone.utc))
    pts = [(t, v) for t, v in pts if t is not None and v is not None]
    if not pts:
        return {}
    running, normal = running_hours(rpm_pts)
    turning = [v > TURNING_RPM for _, v in pts]
    starts = sum(1 for a, b in zip(turning, turning[1:]) if (not a) and b)
    stops = sum(1 for a, b in zip(turning, turning[1:]) if a and (not b))
    n = len(pts)
    return {
        "hours_data": n * bucket_h,
        "hours_running": len(running) * bucket_h,
        "hours_turning": sum(turning) * bucket_h,
        "hours_stopped": (n - sum(turning)) * bucket_h,
        "starts": starts, "stops": stops,
        "availability": (sum(turning) / n * 100.0) if n else None,
        "normal_rpm": normal,
    }


def channel_stats(pts: List[Dict[str, Any]], running: Optional[set]) -> Optional[Dict[str, Any]]:
    """Máximo del periodo (todo) + promedio y mínimo EN OPERACIÓN.

    El mínimo es el del PROMEDIO por intervalo: una lectura suelta en 0
    (corte de comunicación) no lo arrastra. Sin dato de rpm → todo el periodo.
    """
    rows = []
    for p in pts or []:
        t = _ts(p.get("bucket"))
        mx, av = _f(p.get("max_val")), _f(p.get("avg_val"))
        if t is None or mx is None:
            continue
        rows.append((t, mx, av if av is not None else mx))
    if not rows:
        return None
    top = max(rows, key=lambda r: r[1])
    op = [r for r in rows if running and r[0] in running] if running else []
    base = op or rows
    return {
        "max": top[1], "max_at": top[0],
        "avg": statistics.fmean(r[2] for r in base),
        "min": min(r[2] for r in base),
        "running": bool(op),
        "n": len(rows),
    }


def pct_change(now: Optional[float], prev: Optional[float]) -> Optional[float]:
    if now is None or prev is None or prev <= 0:
        return None
    return (now - prev) / prev * 100.0


_RANK = {"Normal": 0, "Sin Norma": 0, "No Data": 0, "Alarma": 1, "Danger": 2}


def summarize_events(events: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Cruces de alarma por sonda: cuántas entradas a alarma/peligro, peor
    nivel, valor máximo y la última vez. Ordenado por peor nivel y cantidad."""
    by: Dict[str, Dict[str, Any]] = {}
    for e in events or []:
        to, fr = e.get("to_status"), e.get("from_status")
        if _RANK.get(to, 0) <= _RANK.get(fr, 0) or _RANK.get(to, 0) == 0:
            continue                      # solo ENTRADAS (subidas) a alarma/peligro
        sl = str(e.get("sensor_label") or "—")
        d = by.setdefault(sl, {"sensor_label": sl, "display": sl.replace("_", ""),
                               "count": 0, "worst": "Alarma", "max": None,
                               "unit": e.get("unit") or "", "last": None})
        d["count"] += 1
        if _RANK.get(to, 0) > _RANK.get(d["worst"], 0):
            d["worst"] = to
        v = _f(e.get("value"))
        if v is not None and (d["max"] is None or v > d["max"]):
            d["max"] = v
        t = _ts(e.get("crossed_at"))
        if t is not None and (d["last"] is None or t > d["last"]):
            d["last"] = t
    return sorted(by.values(), key=lambda d: (-_RANK.get(d["worst"], 0), -d["count"]))


def watch_points(channels: List[Dict[str, Any]], n: int = 3) -> List[Dict[str, Any]]:
    """Las n sondas más cerca de su alerta (% del valor actual sobre la alerta)."""
    pool = [c for c in channels if c.get("pct") is not None]
    return sorted(pool, key=lambda c: -c["pct"])[:n]


# ---------------------------------------------------------------------------
# Lectura de la base
# ---------------------------------------------------------------------------

def _hourly(instance_id: str, variable: str, frm: datetime) -> List[Dict[str, Any]]:
    from core.live_readings import history_bucketed, history_rollup
    pts = history_rollup(instance_id, variable, "Direct", frm.isoformat(), "1 hour") or []
    if len([p for p in pts if p.get("max_val") is not None]) < 2:
        pts = history_bucketed(instance_id, variable, "Direct", frm.isoformat(), "1 hour") or []
    return [p for p in pts if p.get("max_val") is not None]


def _speed_variables(latest: List[Dict[str, Any]]) -> List[Tuple[str, str, float]]:
    """[(variable, etiqueta eje, valor)] de velocidades, mayor primero."""
    from core.live_report_builder import _speed_shaft_label
    best: Dict[str, Tuple[str, str, float]] = {}
    for r in latest or []:
        var = str(r.get("variable") or "")
        lv, slab = var.lower(), str(r.get("sensor_label") or "").lower()
        if not (lv.startswith("velocidad") or lv.startswith("speed") or "kph" in slab
                or "rpm" in str(r.get("unit") or "").lower()):
            continue
        v = _f(r.get("value"))
        if v is None:
            continue
        lab = _speed_shaft_label(r.get("sensor_label"), var)
        if lab not in best or v > best[lab][2]:
            best[lab] = (var, lab, v)
    return sorted(best.values(), key=lambda x: -x[2])


def period_extras(instance_id: str, channels: List[Dict[str, Any]],
                  period_label: str = "Semanal",
                  now: Optional[datetime] = None) -> Dict[str, Any]:
    """Todo lo nuevo del reporte del periodo. `channels` = los de la tabular
    (briefing_builder); se ENRIQUECEN in place (orden del tren Y→X, auxiliares
    agrupados, % de alerta) y se devuelven en `channels`.
    Nunca lanza: lo que falle queda vacío y el reporte sale igual."""
    from core.live_readings import latest_for_instance
    from core.report_extras import aux_groups_for, order_channels

    monthly = str(period_label or "").lower().startswith("mensual")
    days = 30 if monthly else 7
    now = now or datetime.now(timezone.utc)
    t_now0 = now - timedelta(days=days)
    t_prev0 = now - timedelta(days=2 * days)
    out: Dict[str, Any] = {"monthly": monthly, "days": days, "channels": channels,
                           "stats": [], "events": [], "watch": [], "operation": {},
                           "band": {}}
    try:
        latest = latest_for_instance(instance_id) or []
    except Exception:  # noqa: BLE001
        latest = []

    # Orden del tren + auxiliares + % de alerta
    try:
        aux, aux_order = aux_groups_for(instance_id)
        channels = order_channels(channels, aux, aux_order)
        out["channels"] = channels
    except Exception as e:  # noqa: BLE001
        log.warning("briefing_extras orden falló: %s", e)
    for c in channels:
        c.setdefault("display", str(c.get("sensor_label") or "").replace("_", ""))
        c.setdefault("aux_group", "")
        v, a = _f(c.get("value")), _f(c.get("alarm"))
        c["pct"] = (v / a * 100.0) if (v is not None and a) else None

    # Velocidades y franja
    speeds = _speed_variables(latest)
    band: Dict[str, Any] = {"speeds": [(lab, val) for _, lab, val in speeds]}
    shaft = [(c, _f(c.get("value"))) for c in channels
             if any(u in str(c.get("unit") or "").lower() for u in ("µm", "um", "mil"))]
    shaft = [(c, v) for c, v in shaft if v is not None]
    if shaft:
        c, v = max(shaft, key=lambda x: x[1])
        band["shaft_max"] = (v, c.get("display"), c.get("unit") or "µm pp")
    out["band"] = band

    # Operación (rpm del eje principal) en el periodo actual y el anterior
    running: set = set()
    try:
        if speeds:
            rpm_pts = _hourly(instance_id, speeds[0][0], t_prev0)
            running, _ = running_hours(rpm_pts)
            cur = [p for p in rpm_pts if (_ts(p.get("bucket")) or now) >= t_now0]
            out["operation"] = operation_summary(cur)
    except Exception as e:  # noqa: BLE001
        log.warning("briefing_extras operación falló: %s", e)

    # Estadística por sonda (periodo actual vs anterior)
    var_by: Dict[str, str] = {}
    for r in latest:
        if r.get("sensor_label") and r.get("variable") and (r.get("metric") or "") == "Direct":
            var_by.setdefault(r["sensor_label"], r["variable"])
    stats = []
    for c in channels:
        var = var_by.get(c.get("sensor_label"))
        if not var:
            continue
        try:
            pts = _hourly(instance_id, var, t_prev0)
        except Exception:  # noqa: BLE001
            continue
        cur = [p for p in pts if (_ts(p.get("bucket")) or now) >= t_now0]
        prv = [p for p in pts if (_ts(p.get("bucket")) or now) < t_now0]
        s_now = channel_stats(cur, running)
        if not s_now:
            continue
        s_prev = channel_stats(prv, running)
        prev_avg = s_prev["avg"] if s_prev else None
        row = {
            "sensor_label": c.get("sensor_label"), "display": c.get("display"),
            "aux_group": c.get("aux_group", ""), "unit": c.get("unit", ""),
            "alarm": _f(c.get("alarm")), "danger": _f(c.get("danger")),
            **s_now, "prev_avg": prev_avg,
            "delta": pct_change(s_now["avg"], prev_avg),
            "pct_max": (s_now["max"] / _f(c.get("alarm")) * 100.0) if _f(c.get("alarm")) else None,
        }
        c["delta"] = row["delta"]
        stats.append(row)
    out["stats"] = stats

    # Eventos de alarma del periodo
    try:
        from core.live_readings import _get_supabase_client
        cl = _get_supabase_client()
        if cl is not None:
            res = (cl.table("severity_events")
                   .select("sensor_label,from_status,to_status,value,unit,crossed_at")
                   .eq("instance_id", instance_id).gte("crossed_at", t_now0.isoformat())
                   .order("crossed_at", desc=True).limit(1000).execute())
            out["events"] = summarize_events(getattr(res, "data", []) or [])
    except Exception as e:  # noqa: BLE001
        log.warning("briefing_extras eventos falló: %s", e)

    out["watch"] = watch_points(channels)
    return out
