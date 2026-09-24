"""
core.briefing_queue — Cola de revisión y aprobación del Briefing
================================================================

Flujo (v3.31.393):

    1. El CRON del lunes genera el BORRADOR de cada activo (secciones con IA
       + datos) y lo deja PENDIENTE en esta cola. Ya no se envía nada al
       cliente automáticamente.
    2. El ESPECIALISTA lo ve en "Briefing por activo": edita resumen,
       diagnóstico y recomendaciones, descarga vista previa.
    3. Al APROBAR: se firma con "Elaborado por" (quien preparó/editó) y
       "Aprobado por" (quien aprueba), se genera el PDF final y AHÍ SÍ se
       envía al cliente por los canales del activo (deliver_report).

Persistencia: captured_parameters["briefing_draft"] de la instancia
(repositorio activo → Supabase). UN borrador vigente por activo (el nuevo
cron reemplaza al anterior si aún no fue aprobado; si ya fue aprobado y
enviado, simplemente arranca el ciclo siguiente).

Modelo del borrador:
    {
      "period": "Semanal", "created_at": ISO,
      "status": "pendiente" | "aprobado",
      "summary": str, "diagnosis": str,
      "prepared_by": str, "prepared_role": str,
      "approved_by": str, "approved_role": str,
      "approved_at": ISO, "sent_at": ISO, "sent_result": str,
      "health": {...}, "kpis": {...},          # snapshot informativo
    }
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

PARAM_KEY = "briefing_draft"

STATUS_PENDING = "pendiente"
STATUS_APPROVED = "aprobado"


def _now_iso() -> str:
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("America/Bogota")).isoformat(timespec="minutes")
    except Exception:
        return datetime.now().isoformat(timespec="minutes")


def get_draft(instance_id: str) -> Optional[Dict[str, Any]]:
    try:
        from core.instance_state import get_instance_parameters
        d = get_instance_parameters(instance_id).get(PARAM_KEY)
        return dict(d) if isinstance(d, dict) else None
    except Exception as e:
        log.warning("get_draft(%s) falló: %s", instance_id, e)
        return None


def save_draft(instance_id: str, draft: Dict[str, Any]) -> bool:
    try:
        from core.instance_state import update_instance_parameter
        return update_instance_parameter(instance_id, PARAM_KEY, dict(draft))
    except Exception as e:
        log.warning("save_draft(%s) falló: %s", instance_id, e)
        return False


def update_draft(instance_id: str, **fields: Any) -> bool:
    d = get_draft(instance_id) or {}
    d.update({k: v for k, v in fields.items() if v is not None})
    return save_draft(instance_id, d)


def clear_draft(instance_id: str) -> bool:
    try:
        from core.instance_state import update_instance_parameter
        return update_instance_parameter(instance_id, PARAM_KEY, None)
    except Exception as e:
        log.warning("clear_draft(%s) falló: %s", instance_id, e)
        return False


def new_pending_draft(instance_id: str, period: str,
                      summary: str, diagnosis: str,
                      health: Optional[Dict[str, Any]] = None,
                      kpis: Optional[Dict[str, Any]] = None,
                      consecutive: str = "") -> bool:
    """Crea/reemplaza el borrador PENDIENTE del activo (lo llama el cron)."""
    return save_draft(instance_id, {
        "period": period, "created_at": _now_iso(),
        "status": STATUS_PENDING,
        "summary": summary or "", "diagnosis": diagnosis or "",
        "consecutive": consecutive or "",
        "prepared_by": "", "prepared_role": "",
        "approved_by": "", "approved_role": "",
        "approved_at": "", "sent_at": "", "sent_result": "",
        "health": health or {}, "kpis": kpis or {},
    })


def list_pending() -> List[Tuple[str, str, Dict[str, Any]]]:
    """[(instance_id, tag, draft)] de todos los activos con borrador
    pendiente de aprobación."""
    out: List[Tuple[str, str, Dict[str, Any]]] = []
    try:
        from core.instance_state import list_instances
        for r in list_instances() or []:
            iid = r.get("instance_id") if isinstance(r, dict) else getattr(r, "instance_id", "")
            tag = (r.get("tag") if isinstance(r, dict) else getattr(r, "tag", "")) or iid
            if not iid:
                continue
            d = get_draft(iid)
            if d and d.get("status") == STATUS_PENDING:
                out.append((iid, tag, d))
    except Exception as e:
        log.warning("list_pending falló: %s", e)
    out.sort(key=lambda t: t[2].get("created_at", ""), reverse=True)
    return out


def approve_and_send(instance_id: str, *,
                     prepared_by: str, approved_by: str,
                     prepared_role: str = "", approved_role: str = "",
                     send: bool = True) -> Dict[str, Any]:
    """Aprueba el borrador: genera el PDF FINAL firmado (Elaborado por /
    Aprobado por) con las secciones editadas + recomendaciones vigentes,
    y si send=True lo envía al cliente por los canales del activo.

    Devuelve {"ok", "pdf", "meta", "delivery", "error"}."""
    out: Dict[str, Any] = {"ok": False, "pdf": None, "meta": {},
                           "delivery": None, "error": ""}
    draft = get_draft(instance_id)
    if not draft:
        out["error"] = "No hay borrador para este activo."
        return out
    if not (prepared_by or "").strip() or not (approved_by or "").strip():
        out["error"] = "El briefing requiere 'Elaborado por' y 'Aprobado por'."
        return out

    try:
        from core.briefing_builder import build_asset_briefing
        from core.instance_state import get_instance
        inst = get_instance(instance_id)
        meta_extra = {
            "prepared_by": prepared_by.strip(),
            "reviewed_by": approved_by.strip(),
            "prepared_label": "Preparado por:",
            "reviewed_label": "Revisado por:",
        }
        # Consecutivo definitivo: el reclamado al crear el borrador; si el
        # borrador es viejo y no trae, se reclama uno nuevo aquí.
        _consec = (draft.get("consecutive") or "").strip()
        if not _consec:
            from core.briefing_builder import next_consecutive
            _consec = next_consecutive(instance_id,
                                       getattr(inst, "tag", "") or "",
                                       claim=True)
        meta_extra["consecutive"] = _consec
        if (prepared_role or "").strip():
            meta_extra["prepared_role"] = prepared_role.strip()
        if (approved_role or "").strip():
            meta_extra["reviewed_role"] = approved_role.strip()
        pdf, meta = build_asset_briefing(
            instance_id, draft.get("period", "Semanal"),
            instance_obj=inst, use_ai=False,
            meta_extra=meta_extra,
            sections_override={
                "summary": draft.get("summary", ""),
                "diagnosis": draft.get("diagnosis", ""),
            },
        )
        if not pdf:
            out["error"] = f"No se pudo generar el PDF: {meta.get('status', '?')}"
            return out
        out["pdf"], out["meta"] = pdf, meta
    except Exception as e:
        log.error("approve_and_send(%s) PDF falló: %s", instance_id, e)
        out["error"] = f"Error generando PDF: {e}"
        return out

    delivery = None
    if send:
        try:
            from core.report_delivery import deliver_report
            delivery = deliver_report(inst, pdf, meta={
                "instance_id": instance_id,
                "status": meta.get("status", "—"),
                "score": meta.get("score"),
                "alarms": meta.get("alarms", 0),
            })
            out["delivery"] = delivery
        except Exception as e:
            log.error("approve_and_send(%s) envío falló: %s", instance_id, e)
            out["delivery"] = {"any_ok": False, "error": str(e)}

    update_draft(
        instance_id,
        status=STATUS_APPROVED,
        prepared_by=prepared_by.strip(), approved_by=approved_by.strip(),
        prepared_role=(prepared_role or "").strip(),
        approved_role=(approved_role or "").strip(),
        approved_at=_now_iso(),
        sent_at=(_now_iso() if (send and delivery and delivery.get("any_ok")) else ""),
        sent_result=(str({k: v for k, v in (delivery or {}).items() if k != "any_ok"})
                     if send else "no enviado"),
    )
    out["ok"] = True
    return out


# ---------------------------------------------------------------------------
# Programación automática (v3.31.401) — día(s) + hora en que el sistema
# genera los borradores y los manda a la cola SOLO. Config GLOBAL, persistida
# en captured_parameters de cada instancia (el cron corre headless en Render
# sin disco compartido → Supabase es la única fuente común).
#   {"enabled": bool, "days": [0..6] (0=Lunes), "hour": 0-23,
#    "period": "Semanal"|"Mensual"}
# ---------------------------------------------------------------------------
SCHED_KEY = "briefing_schedule"      # legacy: 1 sola programación (dict)
SCHEDS_KEY = "briefing_schedules"    # v24: lista de programaciones por activo

_DEFAULT_SCHED = {"enabled": False, "days": [0], "hour": 5, "period": "Semanal"}


def _clean_entry(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Normaliza una programación. v24.2: cada reporte tiene una lista de SLOTS
    (día, hora) → permite hora distinta por día. Migra el formato viejo
    (days[] + hour único) a slots. Conserva days/hour derivados por compat."""
    slots = cfg.get("slots")
    if slots:
        cs = sorted({(max(0, min(6, int(s[0]))), max(0, min(23, int(s[1]))))
                     for s in slots if isinstance(s, (list, tuple)) and len(s) >= 2})
    else:
        days = [int(d) for d in (cfg.get("days") or [0]) if 0 <= int(d) <= 6] or [0]
        hour = max(0, min(23, int(cfg.get("hour", 5))))
        cs = sorted({(d, hour) for d in days})
    cs = [[d, h] for d, h in cs] or [[0, 5]]
    return {
        "enabled": bool(cfg.get("enabled")),
        "period": ("Mensual" if str(cfg.get("period", "")).lower().startswith("mensual")
                   else "Semanal"),
        "slots": cs,
        "days": sorted({d for d, _ in cs}),   # compat
        "hour": cs[0][1],                      # compat
    }


def get_schedule(instance_id: str) -> Dict[str, Any]:
    """Config de programación DEL ACTIVO — legacy (una sola). Devuelve la
    entrada Semanal si hay lista nueva; si no, la config vieja."""
    try:
        entries = get_schedules(instance_id)
        if entries:
            for e in entries:
                if str(e.get("period", "")).startswith("Semanal"):
                    return dict(e)
            return dict(entries[0])
    except Exception as e:
        log.warning("get_schedule(%s) falló: %s", instance_id, e)
    return dict(_DEFAULT_SCHED)


def get_schedules(instance_id: str) -> List[Dict[str, Any]]:
    """Lista de programaciones del activo (v24). Migra transparentemente la
    config vieja (SCHED_KEY, un solo dict) a lista de 1 elemento."""
    try:
        from core.instance_state import get_instance_parameters
        params = get_instance_parameters(instance_id) or {}
        raw = params.get(SCHEDS_KEY)
        if isinstance(raw, list):
            return [_clean_entry(e) for e in raw if isinstance(e, dict)]
        legacy = params.get(SCHED_KEY)
        if isinstance(legacy, dict):
            return [_clean_entry(legacy)]
    except Exception as e:
        log.warning("get_schedules(%s) falló: %s", instance_id, e)
    return []


def save_schedule(instance_id: str, cfg: Dict[str, Any]) -> bool:
    """Guarda UNA programación (legacy). Reemplaza la entrada del mismo periodo
    en la lista nueva y conserva las demás."""
    entry = _clean_entry(cfg)
    others = [e for e in get_schedules(instance_id)
              if e.get("period") != entry.get("period")]
    return save_schedules(instance_id, others + [entry])


def save_schedules(instance_id: str, entries: List[Dict[str, Any]]) -> bool:
    """Guarda la LISTA de programaciones del activo. Escribe la clave nueva y
    espeja la entrada Semanal (o la primera) en SCHED_KEY para compat de
    lectores viejos."""
    try:
        from core.instance_state import update_instance_parameter
        clean = [_clean_entry(e) for e in (entries or []) if isinstance(e, dict)]
        ok = update_instance_parameter(instance_id, SCHEDS_KEY, clean)
        # Espejo legacy: primera Semanal habilitada, o primera, o default OFF.
        mirror = next((e for e in clean if e.get("enabled")
                       and e.get("period", "").startswith("Semanal")), None) \
            or (clean[0] if clean else dict(_DEFAULT_SCHED))
        update_instance_parameter(instance_id, SCHED_KEY, dict(mirror))
        return ok
    except Exception as e:
        log.warning("save_schedules(%s) falló: %s", instance_id, e)
        return False


def list_schedules() -> List[Tuple[str, str, Dict[str, Any]]]:
    """[(instance_id, tag, cfg)] — legacy, una entrada por activo."""
    out: List[Tuple[str, str, Dict[str, Any]]] = []
    try:
        from core.instance_state import list_instances
        for r in list_instances() or []:
            iid = r.get("instance_id") if isinstance(r, dict) else getattr(r, "instance_id", "")
            tag = (r.get("tag") if isinstance(r, dict) else getattr(r, "tag", "")) or iid
            if not iid:
                continue
            for e in get_schedules(iid):
                out.append((iid, tag, e))
                break
    except Exception as e:
        log.warning("list_schedules falló: %s", e)
    return out


def list_schedule_entries() -> List[Tuple[str, str, Dict[str, Any]]]:
    """[(instance_id, tag, cfg)] — UNA fila por CADA programación (v24). El
    cron itera esto para soportar Semanal + Mensual en el mismo activo."""
    out: List[Tuple[str, str, Dict[str, Any]]] = []
    try:
        from core.instance_state import list_instances
        for r in list_instances() or []:
            iid = r.get("instance_id") if isinstance(r, dict) else getattr(r, "instance_id", "")
            tag = (r.get("tag") if isinstance(r, dict) else getattr(r, "tag", "")) or iid
            if not iid:
                continue
            for e in get_schedules(iid):
                out.append((iid, tag, e))
    except Exception as e:
        log.warning("list_schedule_entries falló: %s", e)
    return out


QUICK_KEY = "quick_report_schedule"  # v24: reporte RÁPIDO 1-pág (envío directo)


def get_quick_schedule(instance_id: str) -> Dict[str, Any]:
    """Programación del reporte RÁPIDO (1 página de Live Monitoring, envío
    DIRECTO sin aprobación). {enabled, slots:[[dow,hour],…]}."""
    try:
        from core.instance_state import get_instance_parameters
        cfg = get_instance_parameters(instance_id).get(QUICK_KEY)
        if isinstance(cfg, dict):
            ce = _clean_entry({"enabled": cfg.get("enabled"),
                               "slots": cfg.get("slots"),
                               "days": cfg.get("days"), "hour": cfg.get("hour"),
                               "period": "Rápido"})
            return {"enabled": ce["enabled"], "slots": ce["slots"]}
    except Exception as e:
        log.warning("get_quick_schedule(%s) falló: %s", instance_id, e)
    return {"enabled": False, "slots": [[0, 7]]}


def save_quick_schedule(instance_id: str, cfg: Dict[str, Any]) -> bool:
    try:
        from core.instance_state import update_instance_parameter
        ce = _clean_entry({"enabled": cfg.get("enabled"), "slots": cfg.get("slots"),
                           "period": "Rápido"})
        return update_instance_parameter(
            instance_id, QUICK_KEY, {"enabled": ce["enabled"], "slots": ce["slots"]})
    except Exception as e:
        log.warning("save_quick_schedule(%s) falló: %s", instance_id, e)
        return False


SIGNERS_KEY = "report_signers"       # v24: firmantes por activo

_DEFAULT_SIGNERS = {
    "prepared_by": "Ángel Daniel Leiva",
    "prepared_role": "Senior Machinery Diagnostics Engineer",
    "reviewed_by": "Ewdes Andrés Hernández",
    "reviewed_role": "Machinery Diagnostics Champion",
}


def get_signers(instance_id: str) -> Dict[str, Any]:
    """Firmantes por activo: quién ELABORA y quién REVISA/aprueba (nombre +
    rol). Se configura junto al envío en el Report Center; el Approval los
    toma por defecto. Si no hay config, usa el default de la casa."""
    try:
        from core.instance_state import get_instance_parameters
        cfg = get_instance_parameters(instance_id).get(SIGNERS_KEY)
        if isinstance(cfg, dict):
            out = dict(_DEFAULT_SIGNERS)
            out.update({k: str(v) for k, v in cfg.items() if k in _DEFAULT_SIGNERS})
            return out
    except Exception as e:
        log.warning("get_signers(%s) falló: %s", instance_id, e)
    return dict(_DEFAULT_SIGNERS)


def save_signers(instance_id: str, cfg: Dict[str, Any]) -> bool:
    """Guarda los firmantes del activo."""
    try:
        from core.instance_state import update_instance_parameter
        clean = {k: str(cfg.get(k, _DEFAULT_SIGNERS[k]) or "").strip()
                 for k in _DEFAULT_SIGNERS}
        return update_instance_parameter(instance_id, SIGNERS_KEY, clean)
    except Exception as e:
        log.warning("save_signers(%s) falló: %s", instance_id, e)
        return False


def schedule_due(cfg: Dict[str, Any], now: Optional[datetime] = None) -> bool:
    """¿Coincide AHORA (hora Bogotá) con el día+hora programados?"""
    if not cfg or not cfg.get("enabled"):
        return False
    if now is None:
        try:
            from zoneinfo import ZoneInfo
            now = datetime.now(ZoneInfo("America/Bogota"))
        except Exception:
            now = datetime.now()
    try:
        slots = cfg.get("slots") or [[d, int(cfg.get("hour", -1))]
                                     for d in (cfg.get("days") or [])]
        return any(now.weekday() == int(s[0]) and now.hour == int(s[1])
                   for s in slots if len(s) >= 2)
    except Exception:
        return False


__all__ = ["get_draft", "save_draft", "update_draft", "clear_draft",
           "new_pending_draft", "list_pending", "approve_and_send",
           "get_schedule", "get_schedules", "save_schedule", "save_schedules",
           "list_schedules", "list_schedule_entries", "schedule_due",
           "get_signers", "save_signers",
           "get_quick_schedule", "save_quick_schedule",
           "STATUS_PENDING", "STATUS_APPROVED", "PARAM_KEY", "SCHED_KEY",
           "SCHEDS_KEY", "SIGNERS_KEY", "QUICK_KEY"]
