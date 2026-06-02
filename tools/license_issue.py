#!/usr/bin/env python3
"""
tools/license_issue.py — Emisor de licencias RSA-signed para Watermelon Planta
==============================================================================

Script INTERNO de SIGA GROUP. Lo corre el equipo comercial cada vez que se
vende/renueva una licencia. NUNCA se distribuye al cliente. La lógica vive en
`tools/license_core.py` (compartida con la app `license_admin.py`).

USO:
    cd watermelon-system
    python tools/license_issue.py \\
        --customer "Termoeléctrica Norte SAS" \\
        --email "ingenieria@termonorte.com" \\
        --expires "2027-05-21" \\
        --modules ema,oma,fea \\
        --max-channels 32 \\
        --plan pro

OUTPUTS (en tools/licenses_issued/<customer-slug>/):
    license.token        ← enviar al cliente
    README_CLIENTE.txt   ← enviar al cliente (instrucciones)
    license.json         ← registro interno SIGA (NO enviar)

¿Preferís una interfaz visual para crear/listar/renovar/verificar licencias?
    streamlit run tools/license_admin.py
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from license_core import PLANS, issue_license  # noqa: E402


def _parse_date(date_str: str) -> datetime:
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"Fecha inválida '{date_str}'. Usá formato YYYY-MM-DD."
        ) from e


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Emite una licencia RSA-signed para Watermelon Planta.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--customer", required=True, help='Razón social. Ej: "Parex Resources"')
    parser.add_argument("--email", required=True, help="Email de contacto técnico.")
    parser.add_argument("--plan", choices=list(PLANS.keys()), default="pro",
                        help="Plan comercial (define defaults). Default: pro.")
    parser.add_argument("--expires", type=_parse_date, default=None,
                        help="Vencimiento (YYYY-MM-DD). Si no se da, usa default del plan.")
    parser.add_argument("--modules", default=None,
                        help='Módulos separados por coma. Ej: "ema,oma,fea".')
    parser.add_argument("--max-channels", type=int, default=None,
                        help="Máx. de canales simultáneos. Default: del plan.")
    parser.add_argument("--notes", default="", help="Notas internas (no van al cliente).")
    args = parser.parse_args()

    mods = None
    if args.modules:
        mods = [m.strip().lower() for m in args.modules.split(",") if m.strip()]

    try:
        res = issue_license(
            customer=args.customer, email=args.email, plan=args.plan,
            expires_dt=args.expires, modules=mods,
            max_channels=args.max_channels, notes=args.notes,
        )
    except (ValueError, RuntimeError) as e:
        print(f"ERROR: {e}")
        return 1

    rec = res.record
    print()
    print("=" * 64)
    print("  ✓ LICENCIA EMITIDA")
    print("=" * 64)
    print()
    print(f"  Cliente:      {rec['customer']}")
    print(f"  Email:        {rec['email']}")
    print(f"  Plan:         {rec['plan_label']}")
    print(f"  Módulos:      {', '.join(rec['modules'])}")
    print(f"  Max canales:  {rec['max_channels']}")
    print(f"  Vence:        {rec['expires_at_utc'][:10]}")
    print(f"  License ID:   {rec['license_id']}")
    print()
    print("  ARCHIVOS GENERADOS:")
    print(f"    {res.token_path}")
    print(f"    {res.readme_path}  (instrucciones para el cliente)")
    print(f"    {res.json_path}    (registro interno SIGA)")
    print()
    print("  ENVIAR AL CLIENTE: license.token + README_CLIENTE.txt")
    print("  NO enviar license.json (registro interno).")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
