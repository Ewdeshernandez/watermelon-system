"""Live Monitoring — datos de valor para el cliente.

* Tendencia por punto (30 días) y cambio contra la semana pasada.
* Horas en operación / arranques en los últimos 30 días (desde las rpm).
* Recomendaciones del especialista con estado (parámetro del activo).
* Solicitudes de servicio del cliente (tabla `service_requests` + correo a SIGA).

Funciones puras arriba (tests en tests/test_live_insights.py); acceso a datos
abajo, siempre a prueba de fallos.

DDL (Supabase SQL editor):
    create table if not exists public.service_requests (
      id bigserial primary key,
      created_at timestamptz not null default now(),
      email text not null, client text, instance_id text, tag text,
      kind text, urgency text, description text, phone text,
      status text not null default 'open', handled_by text, handled_at timestamptz
    );
    alter table public.service_requests enable row level security;
"""
from __future__ import annotations

import logging
import statistics
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence

log = logging.getLogger(__name__)

RECS_KEY = "recommendations"
REC_STATUSES = ("pending", "in_progress", "done")
REC_LABELS = {"pending": ("Pendiente", "Pending"), "in_progress": ("En curso", "In progress"),
              "done": ("Cerrada", "Closed")}
SERVICE_KINDS = {
    "remote": ("Análisis remoto de un punto o evento", "Remote analysis of a point or event"),
    "visit": ("Visita técnica en sitio", "On-site technical visit"),
    "balance": ("Balanceo", "Balancing"),
    "modal": ("Análisis modal / resonancia", "Modal / resonance analysis"),
    "other": ("Otro", "Other"),
}
FLAT_PCT = 8.0          # |cambio| menor a esto → "estable"


def _ts(v: Any) -> Optional[datetime]:
    try:
        d = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
        return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Tendencia por punto
# ---------------------------------------------------------------------------

def running_values(rows: Sequence[Dict[str, Any]]) -> List[float]:
    """Promedios diarios descartando días con la máquina parada o parcial
    (valor < 70 % de la mediana) — evitan falsas 'caídas' por arranques."""
    vals = []
    for r in rows or []:
        try:
            vals.append(float(r.get("avg_val")))
        except (TypeError, ValueError):
            continue
    if not vals:
        return []
    pos = [v for v in vals if v > 0]
    if not pos:
        return []
    med = statistics.median(pos)
    return [v for v in vals if v >= 0.70 * med]


def trend_delta(rows: Sequence[Dict[str, Any]], now: Optional[datetime] = None) -> Dict[str, Any]:
    """Cambio del promedio de los últimos 2 días contra el de hace 7–8 días.
    rows = buckets diarios de history_rollup (bucket, avg_val)."""
    now = now or datetime.now(timezone.utc)
    good = []
    vals = running_values(rows)
    keep = set(vals)
    for r in rows or []:
        t = _ts(r.get("bucket"))
        try:
            v = float(r.get("avg_val"))
        except (TypeError, ValueError):
            continue
        if t is not None and v in keep:
            good.append((t, v))
    good.sort()
    spark = [v for _, v in good][-30:]
    recent = [v for t, v in good if t >= now - timedelta(days=2)]
    before = [v for t, v in good if now - timedelta(days=9) <= t < now - timedelta(days=6)]
    if not recent or not before or statistics.mean(before) == 0:
        return {"spark": spark, "pct": None, "dir": "flat"}
    pct = (statistics.mean(recent) - statistics.mean(before)) / abs(statistics.mean(before)) * 100.0
    d = "up" if pct >= FLAT_PCT else ("down" if pct <= -FLAT_PCT else "flat")
    return {"spark": spark, "pct": round(pct, 1), "dir": d}


def point_trends(instance_id: str, points: Sequence[Dict[str, str]], days: int = 30) -> Dict[str, Dict[str, Any]]:
    """{tag: trend_delta} para cada punto {tag, variable}. Una RPC por punto, en paralelo."""
    from core.live_readings import history_rollup
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

    def fetch(p):
        try:
            return history_rollup(instance_id, p["variable"], "Direct", since, "1 day")
        except Exception:  # noqa: BLE001
            return []

    pts = [p for p in points if p.get("variable")]
    with ThreadPoolExecutor(max_workers=4) as ex:
        raw = dict(zip([p["tag"] for p in pts], ex.map(fetch, pts)))
    for p in pts:                       # el cliente HTTP compartido a veces falla en paralelo
        if not raw.get(p["tag"]):
            raw[p["tag"]] = fetch(p)
    return {tag: trend_delta(rows) for tag, rows in raw.items()}


# ---------------------------------------------------------------------------
# Horas en operación
# ---------------------------------------------------------------------------

def operation_stats(rows: Sequence[Dict[str, Any]], *, days: int = 30,
                    now: Optional[datetime] = None) -> Optional[Dict[str, Any]]:
    """Desde buckets HORARIOS de rpm del eje principal:
    horas operando (≥ 90 % de la rpm normal), horas con dato, arranques y paradas.
    La rpm normal = mediana de las horas con rpm > 100."""
    now = now or datetime.now(timezone.utc)
    pts = []
    for r in rows or []:
        t = _ts(r.get("bucket"))
        try:
            v = float(r.get("avg_val"))
        except (TypeError, ValueError):
            continue
        if t is not None:
            pts.append((t, v))
    if not pts:
        return None
    pts.sort()
    moving = [v for _, v in pts if v > 100]
    if not moving:
        return {"days": days, "hours_total": days * 24, "hours_data": len(pts), "hours_running": 0,
                "pct_running": 0.0, "starts": 0, "stops": 0, "normal_rpm": None,
                "hours_nodata": max(0, days * 24 - len(pts))}
    normal = statistics.median(moving)
    running = [v >= 0.9 * normal for _, v in pts]
    stopped = [v < 0.1 * normal for _, v in pts]
    starts = stops = 0
    state = None                      # None / "run" / "stop" (ignora transitorios intermedios)
    for i in range(len(pts)):
        s = "run" if running[i] else ("stop" if stopped[i] else None)
        if s is None:
            continue
        if state == "stop" and s == "run":
            starts += 1
        elif state == "run" and s == "stop":
            stops += 1
        state = s
    h_run = sum(running)
    return {"days": days, "hours_total": days * 24, "hours_data": len(pts), "hours_running": h_run,
            "pct_running": round(h_run / len(pts) * 100.0, 1), "starts": starts, "stops": stops,
            "normal_rpm": round(normal), "hours_nodata": max(0, days * 24 - len(pts))}


def main_speed_variable(latest: Sequence[Dict[str, Any]]) -> Optional[str]:
    """La variable 'Velocidad …' de mayor rpm (eje principal)."""
    best = None
    for r in latest or []:
        var = str(r.get("variable") or "")
        if r.get("metric") == "Direct" and var.lower().startswith("velocidad"):
            try:
                v = float(r.get("value") or 0)
            except (TypeError, ValueError):
                v = 0
            if best is None or v > best[1]:
                best = (var, v)
    return best[0] if best else None


def load_operation_stats(instance_id: str, speed_var: str, days: int = 30) -> Optional[Dict[str, Any]]:
    try:
        from core.live_readings import history_rollup
        since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        return operation_stats(history_rollup(instance_id, speed_var, "Direct", since, "1 hour"), days=days)
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Recomendaciones del especialista
# ---------------------------------------------------------------------------

def sort_recs(recs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    order = {"pending": 0, "in_progress": 1, "done": 2}
    return sorted(recs or [], key=lambda r: (order.get(r.get("status"), 3),
                                             -_epoch(r.get("updated_at") or r.get("created_at"))))


def _epoch(v: Any) -> float:
    t = _ts(v)
    return t.timestamp() if t else 0.0


def get_recommendations(instance_id: str) -> List[Dict[str, Any]]:
    try:
        from core.instance_state import get_instance_parameters
        v = (get_instance_parameters(instance_id) or {}).get(RECS_KEY)
        return [r for r in v if isinstance(r, dict) and r.get("text")] if isinstance(v, list) else []
    except Exception:  # noqa: BLE001
        return []


def save_recommendations(instance_id: str, recs: List[Dict[str, Any]]) -> bool:
    try:
        from core.instance_state import update_instance_parameter
        return bool(update_instance_parameter(instance_id, RECS_KEY, recs))
    except Exception:  # noqa: BLE001
        return False


def add_recommendation(recs: List[Dict[str, Any]], text: str, by: str, source: str = "") -> List[Dict[str, Any]]:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return list(recs) + [{"id": uuid.uuid4().hex[:10], "text": text.strip()[:400], "status": "pending",
                          "source": source.strip()[:80], "by": by, "created_at": now, "updated_at": now}]


def set_rec_status(recs: List[Dict[str, Any]], rec_id: str, status: str, by: str) -> List[Dict[str, Any]]:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out = []
    for r in recs:
        if r.get("id") == rec_id and status in REC_STATUSES:
            r = dict(r, status=status, updated_at=now, by=by)
        out.append(r)
    return out


# ---------------------------------------------------------------------------
# Solicitudes de servicio
# ---------------------------------------------------------------------------

def _sb():
    try:
        from core.live_readings import _get_supabase_client
        return _get_supabase_client()
    except Exception:  # noqa: BLE001
        return None


def create_service_request(*, email: str, client: str, instance_id: str, tag: str, kind: str,
                           urgency: str, description: str, phone: str = "") -> Dict[str, Any]:
    """Guarda la solicitud y avisa a SIGA por correo. Devuelve {"ok", "id"?, "error"?}."""
    row = {"email": email, "client": client, "instance_id": instance_id, "tag": tag, "kind": kind,
           "urgency": urgency, "description": description.strip()[:2000], "phone": phone.strip()[:40]}
    c = _sb()
    rid = None
    if c is not None:
        try:
            r = c.table("service_requests").insert(row).execute()
            rid = (r.data or [{}])[0].get("id")
        except Exception as e:  # noqa: BLE001
            log.warning("service_requests insert failed: %s", e)
    mail_ok = False
    try:
        from core.email_sender import send_email
        from core.supabase_auth import ADMIN_EMAIL
        kind_txt = SERVICE_KINDS.get(kind, (kind, kind))[0]
        subj = f"[Watermelon] Solicitud de servicio · {client} · {tag}" + (" · URGENTE" if urgency == "urgent" else "")
        body = (f"Cliente: {client}\nUsuario: {email}\nTeléfono: {phone or '—'}\nMáquina: {tag}\n"
                f"Tipo: {kind_txt}\nUrgencia: {'Urgente' if urgency == 'urgent' else 'Normal'}\n\n"
                f"{description.strip()}\n\nVer en Live Monitoring (vista SIGA).")
        res = send_email(ADMIN_EMAIL, subj, body, reply_to=email, from_name="Watermelon Live Monitoring")
        mail_ok = bool(res.get("ok"))
    except Exception as e:  # noqa: BLE001
        log.warning("service request email failed: %s", e)
    if rid is None and not mail_ok:
        return {"ok": False, "error": "no se pudo registrar ni enviar"}
    return {"ok": True, "id": rid, "mailed": mail_ok}


def open_service_requests(limit: int = 50) -> Optional[List[Dict[str, Any]]]:
    c = _sb()
    if c is None:
        return None
    try:
        r = (c.table("service_requests").select("*").neq("status", "done")
             .order("created_at", desc=True).limit(limit).execute())
        return list(r.data or [])
    except Exception as e:  # noqa: BLE001
        log.info("service_requests read failed: %s", e)
        return None


def requests_for_user(email: str, limit: int = 5) -> List[Dict[str, Any]]:
    c = _sb()
    if c is None or not email:
        return []
    try:
        r = (c.table("service_requests").select("id,created_at,tag,kind,status")
             .eq("email", email.lower()).order("created_at", desc=True).limit(limit).execute())
        return list(r.data or [])
    except Exception:  # noqa: BLE001
        return []


def set_request_status(req_id: Any, status: str, by: str) -> bool:
    c = _sb()
    if c is None:
        return False
    try:
        c.table("service_requests").update({
            "status": status, "handled_by": by,
            "handled_at": datetime.now(timezone.utc).isoformat()}).eq("id", req_id).execute()
        return True
    except Exception:  # noqa: BLE001
        return False
