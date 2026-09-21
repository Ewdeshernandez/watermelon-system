"""
scripts/send_weekly_briefing.py
===============================

Cron headless del BRIEFING por activo (F4). Pensado para un Render Cron Job
que corre los LUNES 6:00 AM (hora Bogotá):

    schedule:  0 11 * * 1     (11:00 UTC = 06:00 America/Bogota, lunes)

Qué hace (v3.31.393 — flujo de APROBACIÓN):
    1. Genera el BORRADOR del briefing de TODOS los activos con datos
       (resumen + diagnóstico con IA) y lo deja PENDIENTE en la cola de
       revisión (core.briefing_queue). NADA se envía al cliente aquí.
    2. Notifica por email al ESPECIALISTA que hay borradores pendientes:
       los revisa/edita/aprueba en la app ("Briefing por activo") y al
       aprobar se firma (Elaborado por / Aprobado por) y AHÍ SÍ se envía
       al cliente.

El destinatario de revisión sale de la env var WM_BRIEFING_REVIEW_EMAIL
(o el default abajo). Se puede pasar varios separados por coma.

Uso (cron Render):
    mkdir -p .streamlit && cp /etc/secrets/secrets.toml .streamlit/secrets.toml \
        && python scripts/send_weekly_briefing.py --period Semanal

Flags:
    --period Semanal|Mensual   (default Semanal)
    --instance <id>            Solo ese activo (pruebas)
    --to <email[,email]>       Override del destinatario de revisión
    --dry-run                  Genera pero NO envía
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
log = logging.getLogger("wm.weekly_briefing")

_DEFAULT_REVIEW_EMAIL = "ehernandez@sigasas.com"


def _review_recipients(override: str = "") -> list:
    raw = (override or os.environ.get("WM_BRIEFING_REVIEW_EMAIL", "")
           or _DEFAULT_REVIEW_EMAIL)
    return [e.strip() for e in raw.split(",") if e.strip() and "@" in e]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", default="Semanal", choices=["Semanal", "Mensual"])
    ap.add_argument("--instance", default="")
    ap.add_argument("--to", default="")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--auto-send", action="store_true",
                    help="Aprueba y ENVIA el briefing directo al cliente SIN revision "
                         "humana (modo autonomo, sin especialista). Firma 'Watermelon "
                         "System (automatico)'.")
    ap.add_argument("--on-weekday", type=int, default=None,
                    help="Solo ejecuta si hoy (America/Bogota) == este dia (0=Lun..6=Dom). "
                         "Permite disparar el semanal desde un cron HORARIO sin crear otro servicio.")
    ap.add_argument("--on-hour", type=int, default=None,
                    help="Solo ejecuta si la hora local (America/Bogota) == esta (0-23).")
    ap.add_argument("--check-schedule", action="store_true",
                    help="Modo cron HORARIO ('0 * * * *'): solo genera si la "
                         "hora/día actuales coinciden con la programación "
                         "configurada en la app (briefing_schedule). El día, "
                         "la hora y el periodo salen de esa config.")
    ap.add_argument("--first-business-day", action="store_true",
                    help="Solo ejecuta si hoy (America/Bogota) es el PRIMER día "
                         "hábil del mes (lun-vie; salta fines de semana). Para el "
                         "briefing MENSUAL disparado desde un cron horario. "
                         "Combínalo con --on-hour.")
    args = ap.parse_args()

    # Gate por día/hora local (para dispararlo desde un cron HORARIO): si hoy no
    # coincide con on-weekday/on-hour (America/Bogota), no hace nada.
    if (args.on_weekday is not None or args.on_hour is not None
            or args.first_business_day):
        from datetime import datetime, date
        try:
            from zoneinfo import ZoneInfo
            now = datetime.now(ZoneInfo("America/Bogota"))
        except Exception:  # noqa: BLE001
            now = datetime.utcnow()  # fallback (peor caso: UTC)
        if args.on_weekday is not None and now.weekday() != args.on_weekday:
            log.info("No es el día programado (hoy=%d, on-weekday=%d) — nada que hacer.",
                     now.weekday(), args.on_weekday)
            return 0
        if args.first_business_day:
            # Primer día hábil = hoy es lun-vie y ningún lun-vie ocurrió antes
            # este mes.
            if now.weekday() >= 5:
                log.info("Hoy es fin de semana — no es primer día hábil.")
                return 0
            _earlier_bday = any(
                date(now.year, now.month, d).weekday() < 5
                for d in range(1, now.day))
            if _earlier_bday:
                log.info("Ya hubo un día hábil antes este mes — no es el primero.")
                return 0
        if args.on_hour is not None and now.hour != args.on_hour:
            log.info("No es la hora programada (hora=%d, on-hour=%d) — nada que hacer.",
                     now.hour, args.on_hour)
            return 0

    from core.briefing_builder import build_asset_draft, build_all_drafts

    # Modo programado desde la UI (v3.31.402 — POR ACTIVO): cada equipo tiene
    # su propio día/hora/periodo en Supabase. Este cron corre cada hora en
    # punto y genera SOLO los borradores de los activos cuya programación
    # coincide con ahora.
    if args.check_schedule:
        from core.briefing_queue import list_schedules, schedule_due
        due = [(iid, tag, cfg) for iid, tag, cfg in list_schedules()
               if schedule_due(cfg)]
        if not due:
            log.info("Ninguna programación coincide ahora — nada que hacer.")
            return 0
        results = []
        for iid, tag, cfg in due:
            log.info("Programación COINCIDE para %s (%s) → borrador %s.",
                     tag, iid, cfg.get("period", "Semanal"))
            results.append(build_asset_draft(iid, cfg.get("period", "Semanal")))
    else:
        log.info("Generando BORRADORES de briefing (%s) para la cola de "
                 "revisión…", args.period)
        if args.instance:
            results = [build_asset_draft(args.instance, args.period)]
        else:
            results = build_all_drafts(args.period)

    ok = [m for m in results if m.get("ok")]
    log.info("Borradores en cola: %d de %d activos.", len(ok), len(results))
    if not ok:
        log.warning("No quedó ningún borrador pendiente.")
        return 0

    # --- MODO AUTÓNOMO: aprueba y envía directo al cliente, sin especialista ---
    if args.auto_send:
        from core.briefing_queue import approve_and_send
        sent = 0
        for m in ok:
            iid = m.get("instance_id")
            if args.dry_run:
                log.info("[DRY-RUN] auto-send %s (%s)", m.get("tag", iid), iid)
                continue
            r = approve_and_send(
                iid, prepared_by="Watermelon System",
                approved_by="Watermelon System (automático)",
                prepared_role="Motor de monitoreo", approved_role="Envío automático",
                send=True)
            deliv = r.get("delivery") or {}
            if r.get("ok") and deliv.get("any_ok"):
                sent += 1
                log.info("AUTO-ENVIADO %s → cliente.", m.get("tag", iid))
            else:
                log.error("Auto-send %s falló: %s", iid,
                          r.get("error") or deliv.get("error") or "sin canal/entrega")
        log.info("Briefing %s auto-enviado a %d/%d activos.", args.period, sent, len(ok))
        return 0

    recipients = _review_recipients(args.to)
    if not recipients:
        log.error("Sin destinatario de revisión (WM_BRIEFING_REVIEW_EMAIL).")
        return 1

    lines = [f"• {m.get('tag', m.get('instance_id'))}: {m.get('status','—')} · "
             f"salud {m.get('score','—')} · {m.get('alarms',0)} alarma(s)"
             for m in ok]

    subject = (f"Briefing {args.period} — {len(ok)} borrador(es) PENDIENTES "
               f"de aprobación")
    body = (
        f"El briefing {args.period.lower()} quedó generado como BORRADOR y está "
        f"pendiente de tu revisión y aprobación en la app "
        f"(Briefing por activo → Pendientes de aprobación):\n\n"
        + "\n".join(lines) +
        "\n\nFlujo: revisa/edita resumen, diagnóstico y recomendaciones → "
        "firma 'Elaborado por' y 'Aprobado por' → al aprobar, el PDF final se "
        "envía automáticamente al cliente por los canales del activo.\n\n"
        "— Watermelon System")

    if args.dry_run:
        log.info("[DRY-RUN] No se notifica. Destinatarios: %s", recipients)
        return 0

    try:
        from core.email_sender import send_email
        sent = 0
        for to in recipients:
            r = send_email(to, subject, body)
            if r and (r.get("ok") if isinstance(r, dict) else r):
                sent += 1
                log.info("Notificación enviada a %s", to)
            else:
                log.error("Fallo notificando a %s: %s", to, r)
        return 0 if sent else 1
    except Exception as e:
        log.error("Notificación falló: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
