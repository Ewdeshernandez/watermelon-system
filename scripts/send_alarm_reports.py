"""
scripts/send_alarm_reports.py
=============================

Cron headless de envío automático POR ALARMA (Fase 3). Pensado para un
Render Cron Job que corre cada 15 minutos (schedule "*/15 * * * *").

Lógica "1 aviso por episodio":
    - Para cada activo con `alarm_send_enabled = True` y destinatario,
      calcula el nivel de severidad ACTUAL (barato, sin armar PDF):
          0 = Normal/Sin datos · 1 = Alarma · 2 = Danger
    - Compara con `alarm_alert_level` (último nivel ya avisado, persistido).
    - Si el nivel EMPEORA (level > alarm_alert_level): genera el PDF y lo
      envía marcado como ALERTA, y guarda alarm_alert_level = level.
      → Esto avisa al ENTRAR en Alarma y otra vez si ESCALA a Danger.
    - Si vuelve a Normal (level == 0) y antes estaba avisado: resetea
      alarm_alert_level = 0 (sin enviar). Así el próximo cruce vuelve a avisar.
    - Si sigue igual o mejora pero todavía en alarma: NO reenvía (anti-spam).

Uso (Render Cron Job, cada 15 min):
    mkdir -p .streamlit && cp /etc/secrets/secrets.toml .streamlit/secrets.toml \
        && python scripts/send_alarm_reports.py

Flags para pruebas:
    --instance <id>   Procesa SOLO ese activo.
    --force           Envía si hay alarma (level>0) ignorando el estado guardado.
    --dry-run         Evalúa y loggea, pero NO envía ni persiste estado.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("wm.alarm_reports")

_LEVEL_NAME = {0: "Normal", 1: "Alarma", 2: "Danger"}


def _has_recipient(inst) -> bool:
    email = (getattr(inst, "client_email", "") or "").strip()
    wa = (getattr(inst, "whatsapp_number", "") or "").strip()
    return bool((email and "@" in email) or wa)


def process(only_instance: str = "", force: bool = False, dry_run: bool = False) -> int:
    from core.instance_state import list_instances, get_instance, update_instance_header
    from core.live_report_builder import current_severity_level, build_report_for_instance
    from core.report_delivery import deliver_report

    if only_instance:
        ids = [only_instance]
    else:
        try:
            ids = [row.get("instance_id") for row in (list_instances() or [])
                   if row.get("instance_id")]
        except Exception as e:
            log.error("No se pudo listar activos: %s", e)
            return 1

    log.info("Chequeo de alarmas · %d activo(s) a evaluar", len(ids))
    sent = skipped = reset = errors = 0

    for iid in ids:
        inst = get_instance(iid)
        if inst is None:
            continue
        if not getattr(inst, "alarm_send_enabled", False):
            skipped += 1
            continue
        if not _has_recipient(inst):
            log.warning("Activo %s con alarma activa pero SIN email/WhatsApp — salteado.", iid)
            skipped += 1
            continue

        tag = getattr(inst, "tag", "") or iid
        stored = int(getattr(inst, "alarm_alert_level", 0) or 0)
        level, status, summary = current_severity_level(iid, inst)

        # ¿Hay que avisar? Solo si EMPEORA respecto a lo ya avisado
        # (o --force con cualquier alarma activa).
        should_alert = (level > stored) or (force and level > 0)

        if should_alert:
            log.info("→ %s: nivel %s (antes %s) — generando aviso…",
                     tag, _LEVEL_NAME.get(level, level), _LEVEL_NAME.get(stored, stored))
            pdf_bytes, meta = build_report_for_instance(iid, inst)
            if not pdf_bytes:
                log.error("   %s: no se pudo generar el PDF (¿sin lecturas?).", tag)
                errors += 1
                continue
            if dry_run:
                log.info("   %s: DRY RUN — no se envía (habría avisado nivel %s).",
                         tag, _LEVEL_NAME.get(level, level))
                continue
            res = deliver_report(inst, pdf_bytes, meta, alert=True)
            em, wa = res.get("email"), res.get("whatsapp")
            if em is not None:
                log.info("   %s email: %s", tag, "OK" if em.get("ok") else f"FALLA · {em.get('error')}")
            if wa is not None:
                log.info("   %s whatsapp: %s", tag, "OK" if wa.get("ok") else f"FALLA · {wa.get('error')}")
            if res.get("any_ok"):
                sent += 1
                update_instance_header(iid, alarm_alert_level=level)
            else:
                errors += 1

        elif level == 0 and stored != 0:
            # Volvió a Normal → resetear el estado para que el próximo cruce avise
            if not dry_run:
                update_instance_header(iid, alarm_alert_level=0)
            log.info("✓ %s: normalizado — estado de alarma reseteado.", tag)
            reset += 1
        else:
            skipped += 1

    log.info("Listo · avisos=%d reset=%d salteados=%d errores=%d", sent, reset, skipped, errors)
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description="Envío automático por alarma")
    p.add_argument("--instance", default="", help="Procesar SOLO este activo (id)")
    p.add_argument("--force", action="store_true", help="Avisar si hay alarma, ignorando estado")
    p.add_argument("--dry-run", action="store_true", help="Evaluar sin enviar ni persistir")
    args = p.parse_args()
    sys.exit(process(only_instance=args.instance, force=args.force, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
