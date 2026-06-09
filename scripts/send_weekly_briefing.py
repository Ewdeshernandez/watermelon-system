"""
scripts/send_weekly_briefing.py
===============================

Cron headless del BRIEFING por activo (F4). Pensado para un Render Cron Job
que corre los LUNES 6:00 AM (hora Bogotá):

    schedule:  0 11 * * 1     (11:00 UTC = 06:00 America/Bogota, lunes)

Qué hace:
    1. Genera el briefing figura-rico de TODOS los activos con datos
       (core.briefing_builder.build_all_briefings).
    2. Envía los PDFs por email al ESPECIALISTA para que los revise y los
       remita al cliente (no se mandan directo al cliente).

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
    args = ap.parse_args()

    from core.briefing_builder import build_asset_briefing, build_all_briefings

    log.info("Generando briefings (%s)…", args.period)
    if args.instance:
        pdf, meta = build_asset_briefing(args.instance, args.period)
        results = [(args.instance, pdf, meta)]
    else:
        results = build_all_briefings(args.period)

    ok = [(iid, pdf, meta) for iid, pdf, meta in results if pdf]
    log.info("Briefings generados: %d de %d activos.", len(ok), len(results))
    if not ok:
        log.warning("Nada que enviar.")
        return 0

    recipients = _review_recipients(args.to)
    if not recipients:
        log.error("Sin destinatario de revisión (WM_BRIEFING_REVIEW_EMAIL).")
        return 1

    # Adjuntar todos los PDFs en un solo correo de revisión
    attachments = []
    lines = []
    for iid, pdf, meta in ok:
        tag = meta.get("tag", iid)
        fname = f"Briefing_{tag}_{args.period}.pdf"
        attachments.append((fname, pdf, "application/pdf"))
        lines.append(f"• {tag}: {meta.get('status','—')} · salud "
                     f"{meta.get('score','—')} · {meta.get('alarms',0)} alarma(s)")

    subject = f"Briefing {args.period} — {len(ok)} activo(s) · revisión"
    body = (
        f"Briefing {args.period} generado automáticamente para revisión del "
        f"especialista antes de remitir al cliente.\n\n"
        + "\n".join(lines) +
        "\n\nRevisa los PDF adjuntos, ajusta diagnóstico/recomendaciones si hace "
        "falta y reenvía al cliente.\n\n— Watermelon System")

    if args.dry_run:
        log.info("[DRY-RUN] No se envía. Destinatarios: %s · adjuntos: %d",
                 recipients, len(attachments))
        return 0

    try:
        from core.email_sender import send_email
        sent = 0
        for to in recipients:
            r = send_email(to, subject, body, attachments=attachments)
            if r and (r.get("ok") if isinstance(r, dict) else r):
                sent += 1
                log.info("Briefing enviado a %s", to)
            else:
                log.error("Fallo enviando a %s: %s", to, r)
        return 0 if sent else 1
    except Exception as e:
        log.error("Envío falló: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
