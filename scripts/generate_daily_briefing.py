#!/usr/bin/env python3
"""
scripts/generate_daily_briefing.py
==================================

Script standalone para generar el briefing diario del sistema sin
necesidad de abrir Streamlit. Pensado para ejecutarse via cron,
GitHub Actions, scheduler de Streamlit Cloud, o cualquier otro
mecanismo automatizado.

Uso típico (cron):
    # Todos los días a las 6:30 AM hora de Bogotá
    30 6 * * *  cd /path/to/WatermelonSystem && \\
                /usr/bin/python3 scripts/generate_daily_briefing.py

Uso con email (requiere configurar SMTP via vars de entorno):
    python scripts/generate_daily_briefing.py \\
        --email cliente@empresa.com \\
        --smtp-host smtp.gmail.com --smtp-port 587 \\
        --smtp-user user@example.com --smtp-pass APP_PASSWORD

Uso solo guardar a disco (lo que hace por default):
    python scripts/generate_daily_briefing.py
    → genera data/briefings/briefing_YYYY-MM-DD.pdf

Uso con output personalizado:
    python scripts/generate_daily_briefing.py \\
        --output /tmp/briefing_hoy.pdf
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path


# Permitir ejecución desde cualquier cwd: agregar root del proyecto al path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def _send_email(pdf_bytes: bytes, to_email: str, subject: str,
                body: str, smtp_host: str, smtp_port: int,
                smtp_user: str, smtp_pass: str) -> bool:
    """Envía el PDF como adjunto por SMTP."""
    try:
        import smtplib
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText
        from email.mime.application import MIMEApplication

        msg = MIMEMultipart()
        msg["From"] = smtp_user
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain", "utf-8"))

        attachment = MIMEApplication(pdf_bytes, _subtype="pdf")
        attachment.add_header(
            "Content-Disposition", "attachment",
            filename=f"briefing_{datetime.now().strftime('%Y-%m-%d')}.pdf",
        )
        msg.attach(attachment)

        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.send_message(msg)
        return True
    except Exception as e:
        print(f"  ✗ Error enviando email: {e}", file=sys.stderr)
        return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Genera el briefing diario de Watermelon System (PDF de 1 página)."
    )
    parser.add_argument("--output", type=str, default="",
                        help="Path del PDF output (default: data/briefings/briefing_YYYY-MM-DD.pdf)")
    parser.add_argument("--email", type=str, default="",
                        help="Email destinatario (opcional)")
    parser.add_argument("--smtp-host", type=str, default=os.environ.get("WM_SMTP_HOST", ""))
    parser.add_argument("--smtp-port", type=int, default=int(os.environ.get("WM_SMTP_PORT", "587") or 587))
    parser.add_argument("--smtp-user", type=str, default=os.environ.get("WM_SMTP_USER", ""))
    parser.add_argument("--smtp-pass", type=str, default=os.environ.get("WM_SMTP_PASS", ""))
    parser.add_argument("--quiet", action="store_true", help="Suprime salida exitosa")
    args = parser.parse_args()

    try:
        from core.briefing import generate_and_save_briefing
    except ImportError as e:
        print(f"✗ Error importando core.briefing: {e}", file=sys.stderr)
        print("  Asegurate de correr desde el root del proyecto, o de tener "
              "instalado reportlab.", file=sys.stderr)
        return 2

    now = datetime.now()
    pdf_bytes, default_path = generate_and_save_briefing(now)

    # Output personalizado si se pidió
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(pdf_bytes)
    else:
        target = default_path

    if not args.quiet:
        print(f"✓ Briefing generado: {target}")
        print(f"  Tamaño: {len(pdf_bytes)} bytes ({len(pdf_bytes)/1024:.1f} KB)")

    # Envío por email si se pidió
    if args.email:
        missing = [k for k in ("smtp_host","smtp_user","smtp_pass") if not getattr(args, k)]
        if missing:
            print(f"✗ Faltan credenciales SMTP: {missing}", file=sys.stderr)
            return 3
        ok = _send_email(
            pdf_bytes=pdf_bytes,
            to_email=args.email,
            subject=f"Briefing diario Watermelon — {now.strftime('%Y-%m-%d')}",
            body=(
                "Hola,\n\nAdjunto el briefing diario de la flota generado "
                f"automáticamente el {now.strftime('%Y-%m-%d %H:%M')}.\n\n"
                "Saludos,\nWatermelon System (SIGASAS)\n"
            ),
            smtp_host=args.smtp_host, smtp_port=args.smtp_port,
            smtp_user=args.smtp_user, smtp_pass=args.smtp_pass,
        )
        if ok and not args.quiet:
            print(f"✓ Email enviado a {args.email}")
        elif not ok:
            return 4

    return 0


if __name__ == "__main__":
    sys.exit(main())
