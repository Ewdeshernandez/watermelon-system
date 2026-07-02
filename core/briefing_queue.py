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
            "prepared_label": "Elaborado por:",
            "reviewed_label": "Aprobado por:",
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


__all__ = ["get_draft", "save_draft", "update_draft", "clear_draft",
           "new_pending_draft", "list_pending", "approve_and_send",
           "STATUS_PENDING", "STATUS_APPROVED", "PARAM_KEY"]
