"""
core/report_extras.py
=====================

Extras del reporte de condición (diario / descarga desde Live):

  · Canales ordenados como el tren: cojinete 1..N, **Y antes que X** (Bently:
    Y = canal 1), luego otras; los AUXILIARES de Machines (bomba de lubricación,
    arrancador…) al final, agrupados con su nombre.
  · % de la alerta por canal y **Δ 24 h** (promedio de la última hora vs la misma
    hora de ayer, desde el rollup horario).
  · **Puntos a vigilar**: las 3 sondas más cerca de su alerta (aviso temprano).
  · Tendencias de **24 h** (máx por hora): vibración de eje (µm) y el punto más
    cercano a su alerta con sus líneas de alerta/peligro.

Nada de esto cambia los datos: solo los ordena y los pone en contexto.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple


def _axis_rank(label: str) -> int:
    t = (label or "").replace("_", "")[1:]
    return 0 if "Y" in t[:1] else (1 if "X" in t[:1] else 2)


def _bearing(label: str) -> int:
    m = re.match(r"^(\d+)", label or "")
    return int(m.group(1)) if m else 99


def aux_groups_for(instance_id: str) -> Tuple[Dict[str, str], List[str]]:
    """{label_display: nombre del grupo} + orden de grupos, desde Machines."""
    try:
        from core.live_readings import _get_supabase_client
        from core.live_train import machines_train_for
        mt = machines_train_for(instance_id, client=_get_supabase_client())
    except Exception:  # noqa: BLE001
        mt = None
    if not mt:
        return {}, []
    groups: Dict[str, str] = {}
    order: List[str] = []
    for a in (mt[1].get("aux") or []):
        name = str(a.get("name") or "Auxiliar")
        order.append(name)
        for p in a.get("probes") or []:
            groups[str(p.get("n") if isinstance(p, dict) else p).upper()] = name
    return groups, order


def order_channels(channels: List[Dict[str, Any]], aux: Dict[str, str],
                   aux_order: List[str]) -> List[Dict[str, Any]]:
    for c in channels:
        disp = str(c.get("sensor_label") or "").replace("_", "")
        c["display"] = disp
        c["aux_group"] = aux.get(disp.upper(), "")

    def key(c):
        lbl = str(c.get("sensor_label") or "")
        if c["aux_group"]:
            gi = aux_order.index(c["aux_group"]) if c["aux_group"] in aux_order else 99
            return (1, gi, _axis_rank(lbl), lbl)
        return (0, _bearing(lbl), _axis_rank(lbl), lbl)
    return sorted(channels, key=key)


def _hourly(instance_id: str, variable: str, hours: int = 26) -> List[Dict[str, Any]]:
    from core.live_readings import history_bucketed, history_rollup
    frm = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    pts = history_rollup(instance_id, variable, "Direct", frm, "1 hour") or []
    if len([p for p in pts if p.get("max_val") is not None]) < 2:
        pts = history_bucketed(instance_id, variable, "Direct", frm, "1 hour") or []
    return [p for p in pts if p.get("max_val") is not None]


def _ts(p: Dict[str, Any]) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(str(p.get("bucket")).replace("Z", "+00:00"))
    except Exception:  # noqa: BLE001
        return None


def delta_24h(pts: List[Dict[str, Any]]) -> Optional[float]:
    """% de cambio: promedio de la última hora vs la hora equivalente de ayer."""
    if len(pts) < 2:
        return None
    pts = sorted(pts, key=lambda p: _ts(p) or datetime.min.replace(tzinfo=timezone.utc))
    last = pts[-1]
    lt = _ts(last)
    if lt is None:
        return None
    target = lt - timedelta(hours=24)
    old = min(pts, key=lambda p: abs(((_ts(p) or lt) - target).total_seconds()))
    if abs(((_ts(old) or lt) - target).total_seconds()) > 3 * 3600:
        return None                     # no hay dato de ayer cerca
    a = last.get("avg_val") if last.get("avg_val") is not None else last.get("max_val")
    b = old.get("avg_val") if old.get("avg_val") is not None else old.get("max_val")
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return None
    if b <= 0:
        return None
    return (a - b) / b * 100.0


def enrich(instance_id: str, latest: List[Dict[str, Any]], rendered_rows: List[Dict[str, Any]],
           channels: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Agrega alerta/%/Δ24h/aux a `channels` (in place, ordenados) y devuelve
    {'channels', 'watch', 'shaft_series', 'watch_series'}."""
    by_row = {r["sensor_label"]: r for r in rendered_rows}
    var_by = {}
    for r in latest or []:
        if r.get("sensor_label") and r.get("variable") and (r.get("metric") or "") == "Direct":
            var_by.setdefault(r["sensor_label"], r["variable"])
    aux, aux_order = aux_groups_for(instance_id)
    channels = order_channels(channels, aux, aux_order)
    hist: Dict[str, List[Dict[str, Any]]] = {}
    for c in channels:
        sl = c.get("sensor_label")
        r = by_row.get(sl) or {}
        c["alarm"] = float(r.get("alarm_used") or 0) or None
        c["danger"] = float(r.get("danger_used") or 0) or None
        try:
            v = float(r.get("value"))
        except (TypeError, ValueError):
            v = None
        c["pct"] = (v / c["alarm"] * 100.0) if (v is not None and c["alarm"]) else None
        c["delta"] = None
        var = var_by.get(sl)
        if var:
            try:
                pts = _hourly(instance_id, var)
                hist[sl] = pts
                c["delta"] = delta_24h(pts)
            except Exception:  # noqa: BLE001
                pass
    watch = sorted([c for c in channels if c.get("pct") is not None],
                   key=lambda c: -c["pct"])[:3]

    def series(sl: str, color: str) -> Optional[Dict[str, Any]]:
        pts = [p for p in hist.get(sl, []) if (_ts(p) or datetime.min.replace(tzinfo=timezone.utc))
               >= datetime.now(timezone.utc) - timedelta(hours=24)]
        if len(pts) < 2:
            return None
        return {"label": str(sl).replace("_", ""), "x": [p.get("bucket") for p in pts],
                "y": [p.get("max_val") for p in pts], "color": color}

    shaft = [c for c in channels if "µm" in str(c.get("unit") or "") or "mil" in str(c.get("unit") or "")]
    shaft = sorted(shaft, key=lambda c: -(float(c.get("value")) if str(c.get("value")).replace(".", "", 1).isdigit() else 0))[:6]
    pal = ["#1e40af", "#0891b2", "#7c3aed", "#be185d", "#0f766e", "#b45309"]
    shaft_series = [s for s in (series(c["sensor_label"], pal[i % len(pal)]) for i, c in enumerate(shaft)) if s]
    watch_series = None
    if watch:
        watch_series = series(watch[0]["sensor_label"], "#d97706")
    return {"channels": channels, "watch": watch, "shaft_series": shaft_series,
            "watch_series": watch_series}
