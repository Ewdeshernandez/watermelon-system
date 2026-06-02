"""
scripts/send_scheduled_reports.py
=================================

Cron headless de envío automático del reporte ejecutivo al cliente
(Fase 1b). Corre SIN sesión Streamlit, pensado para un Render Cron Job
que se dispara cada hora en punto.

Lógica:
    1. Lista todos los activos.
    2. Para cada uno con `report_send_enabled = True`, chequea si AHORA
       (hora local América/Bogotá) coincide con su `report_send_day`
       (0=Lunes…6=Domingo) y `report_send_hour` (0-23).
    3. Si coincide, genera el PDF (core.live_report_builder) y lo envía
       por email y/o WhatsApp (core.report_delivery).

Uso (cron de Render, cada hora en punto → schedule "0 * * * *"):
    mkdir -p .streamlit && cp /etc/secrets/secrets.toml .streamlit/secrets.toml \
        && python scripts/send_scheduled_reports.py

Flags para pruebas manuales:
    --instance <id>   Procesa SOLO ese activo.
    --force           Ignora el día/hora programados (envía igual).
    --dry-run         Genera el PDF pero NO envía (solo loggea).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime

# Permitir importar core.* cuando se corre como script desde la raíz del repo
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("wm.scheduled_reports")

_TZ_NAME = "America/Bogota"


def _now_local() -> datetime:
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo(_TZ_NAME))
    except Exception:
        return datetime.now()


def _is_due(inst, now: datetime) -> bool:
    """¿Coincide AHORA con el día/hora programados del activo?"""
    if not getattr(inst, "report_send_enabled", False):
        return False
    try:
        day = int(getattr(inst, "report_send_day", 0) or 0)
        hour = int(getattr(inst, "report_send_hour", 6) or 6)
    except Exception:
        return False
    # weekday(): Lunes=0 … Domingo=6 (coincide con nuestra convención)
    return now.weekday() == day and now.hour == hour


def _has_recipient(inst) -> bool:
    email = (getattr(inst, "client_email", "") or "").strip()
    wa = (getattr(inst, "whatsapp_number", "") or "").strip()
    return bool((email and "@" in email) or wa)


def process(only_instance: str = "", force: bool = False, dry_run: bool = False) -> int:
    from core.instance_state import list_instances, get_instance
    from core.live_report_builder import build_report_for_instance
    from core.report_delivery import deliver_report

    now = _now_local()
    log.info("Ejecutando envíos programados · ahora local=%s (weekday=%d hour=%d)",
             now.strftime("%Y-%m-%d %H:%M %Z"), now.weekday(), now.hour)

    # Resolver lista de activos a evaluar
    if only_instance:
        ids = [only_instance]
    else:
        try:
            ids = [row.get("instance_id") for row in (list_instances() or [])
                   if row.get("instance_id")]
        except Exception as e:
            log.error("No se pudo listar activos: %s", e)
            return 1

    sent_ok = 0
    sent_fail = 0
    skipped = 0

    for iid in ids:
        inst = get_instance(iid)
        if inst is None:
            log.warning("Activo %s no existe — salteado.", iid)
            continue

        due = force or _is_due(inst, now)
        if not due:
            skipped += 1
            continue
        if not _has_recipient(inst):
            log.warning("Activo %s programado pero SIN email/WhatsApp — salteado.", iid)
            skipped += 1
            continue

        tag = getattr(inst, "tag", "") or iid
        log.info("→ Generando reporte de %s (%s)…", tag, iid)
        pdf_bytes, meta = build_report_for_instance(iid, inst)
        if not pdf_bytes:
            log.error("   %s: no se pudo generar el PDF (¿sin lecturas?). meta=%s", tag, meta)
            sent_fail += 1
            continue

        if dry_run:
            log.info("   %s: PDF OK (%d bytes) — DRY RUN, no se envía.", tag, len(pdf_bytes))
            sent_ok += 1
            continue

        res = deliver_report(inst, pdf_bytes, meta)
        em = res.get("email")
        wa = res.get("whatsapp")
        if em is not None:
            log.info("   %s email: %s", tag, "OK" if em.get("ok") else f"FALLA · {em.get('error')}")
        if wa is not None:
            log.info("   %s whatsapp: %s", tag, "OK" if wa.get("ok") else f"FALLA · {wa.get('error')}")
        if res.get("any_ok"):
            sent_ok += 1
        else:
            sent_fail += 1
            if em is None and wa is None:
                log.error("   %s: %s", tag, res.get("error", "no se envió por ningún canal"))

    log.info("Listo · enviados=%d fallidos=%d salteados=%d", sent_ok, sent_fail, skipped)
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description="Envío programado del reporte ejecutivo")
    p.add_argument("--instance", default="", help="Procesar SOLO este activo (id)")
    p.add_argument("--force", action="store_true", help="Ignorar día/hora programados")
    p.add_argument("--dry-run", action="store_true", help="Generar PDF pero NO enviar")
    args = p.parse_args()
    rc = process(only_instance=args.instance, force=args.force, dry_run=args.dry_run)
    sys.exit(rc)


if __name__ == "__main__":
    main()
