#!/usr/bin/env python3
"""
tools/license_issue.py — Emisor de licencias RSA-signed para Watermelon Planta
==============================================================================

Script INTERNO de SIGA GROUP. Lo corre el equipo de comercial cada vez que
se vende una licencia a un cliente nuevo. NUNCA se distribuye al cliente.

Flujo:
  1. SIGA cierra venta con cliente X (ej: "Termoeléctrica Norte SAS").
  2. SIGA corre este script con los datos del cliente.
  3. Genera un archivo `license.token` (JWT firmado con RSA-2048).
  4. SIGA envía el archivo `license.token` al cliente por email/USB/etc.
  5. Cliente lo pega en `<install_dir>/planta/data/license.token`.
  6. Al iniciar Watermelon Planta, `license_manager.py` verifica el token
     con la public key embebida y desbloquea las features compradas.

USO:
    cd watermelon-system
    python tools/license_issue.py \\
        --customer "Termoeléctrica Norte SAS" \\
        --email "ingenieria@termonorte.com" \\
        --expires "2027-05-21" \\
        --modules ema,oma,fea \\
        --max-channels 32 \\
        --plan pro

OUTPUTS:
    tools/licenses_issued/<customer-slug>/license.token   ← enviar al cliente
    tools/licenses_issued/<customer-slug>/license.json    ← registro interno

REQUIERE:
    pip install pyjwt cryptography
    tools/.keys/private_key.pem  ← debe existir (generar con license_keygen.py)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path


# ============================================================================
# CONFIGURACIÓN DE PLANES COMERCIALES
# ============================================================================
# Cada plan define qué features están disponibles. Esto se sincroniza con
# la lista de pricing pública. Si agregas un plan nuevo, actualiza también
# planta/license_manager.py:_PLAN_FEATURES.

PLANS = {
    "trial": {
        "label": "Trial 30 días",
        "default_modules": ["ema"],
        "default_max_channels": 4,
        "default_duration_days": 30,
    },
    "basic": {
        "label": "Basic — EMA only",
        "default_modules": ["ema"],
        "default_max_channels": 8,
        "default_duration_days": 365,
    },
    "pro": {
        "label": "Pro — EMA + OMA",
        "default_modules": ["ema", "oma"],
        "default_max_channels": 16,
        "default_duration_days": 365,
    },
    "enterprise": {
        "label": "Enterprise — EMA + OMA + FEA + 3D + Reports",
        "default_modules": ["ema", "oma", "fea", "modes3d", "reports"],
        "default_max_channels": 32,
        "default_duration_days": 365,
    },
}

VALID_MODULES = {"ema", "oma", "fea", "modes3d", "reports", "sync"}


# ============================================================================
# HELPERS
# ============================================================================

def slugify(text: str) -> str:
    """Convierte 'Termoeléctrica Norte SAS' → 'termoelectrica-norte-sas'."""
    text = text.lower().strip()
    # Quitar acentos básicos
    repl = {"á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u", "ñ": "n"}
    for src, dst in repl.items():
        text = text.replace(src, dst)
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")


def parse_date(date_str: str) -> datetime:
    """'2027-05-21' → datetime UTC."""
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return dt.replace(tzinfo=timezone.utc)
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"Fecha inválida '{date_str}'. Usa formato YYYY-MM-DD."
        ) from e


# ============================================================================
# MAIN
# ============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Emite una licencia RSA-signed para Watermelon Planta.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--customer", required=True,
        help='Razón social del cliente. Ej: "Termoeléctrica Norte SAS"',
    )
    parser.add_argument(
        "--email", required=True,
        help="Email de contacto técnico del cliente.",
    )
    parser.add_argument(
        "--plan", choices=list(PLANS.keys()), default="pro",
        help="Plan comercial (define defaults). Default: pro.",
    )
    parser.add_argument(
        "--expires", type=parse_date, default=None,
        help="Fecha de vencimiento (YYYY-MM-DD). Si no se da, usa default del plan.",
    )
    parser.add_argument(
        "--modules", default=None,
        help='Módulos habilitados, separados por coma. '
             'Ej: "ema,oma,fea". Si no se da, usa default del plan.',
    )
    parser.add_argument(
        "--max-channels", type=int, default=None,
        help="Máximo de canales NI simultáneos. Default: del plan.",
    )
    parser.add_argument(
        "--notes", default="",
        help="Notas internas (no van al cliente, solo al registro json).",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------------
    # 1. Validar private key existe
    # ------------------------------------------------------------------------
    priv_path = Path(__file__).parent / ".keys" / "private_key.pem"
    if not priv_path.exists():
        print(f"ERROR: No existe la private key en {priv_path}")
        print()
        print("Primero genera el par de claves con:")
        print("    python tools/license_keygen.py")
        return 1

    # ------------------------------------------------------------------------
    # 2. Validar dependencias
    # ------------------------------------------------------------------------
    try:
        import jwt  # PyJWT
        from cryptography.hazmat.primitives import serialization
    except ImportError as e:
        print(f"ERROR: faltan dependencias: {e}")
        print()
        print("Instala con:")
        print("    pip install pyjwt cryptography")
        return 1

    # ------------------------------------------------------------------------
    # 3. Resolver defaults del plan
    # ------------------------------------------------------------------------
    plan_cfg = PLANS[args.plan]

    if args.modules is None:
        modules = list(plan_cfg["default_modules"])
    else:
        modules = [m.strip().lower() for m in args.modules.split(",") if m.strip()]
        invalid = set(modules) - VALID_MODULES
        if invalid:
            print(f"ERROR: módulos inválidos: {invalid}")
            print(f"  Válidos: {sorted(VALID_MODULES)}")
            return 1

    max_channels = args.max_channels if args.max_channels is not None \
        else plan_cfg["default_max_channels"]

    if args.expires is None:
        from datetime import timedelta
        expires_dt = datetime.now(timezone.utc) + timedelta(
            days=plan_cfg["default_duration_days"]
        )
    else:
        expires_dt = args.expires

    # ------------------------------------------------------------------------
    # 4. Construir el payload del JWT
    # ------------------------------------------------------------------------
    now = datetime.now(timezone.utc)
    license_id = str(uuid.uuid4())

    payload = {
        # Claims standard JWT
        "iss": "SIGA GROUP SAS",          # issuer
        "sub": args.email,                # subject (email del cliente)
        "aud": "watermelon-planta",       # audience (app que verifica)
        "iat": int(now.timestamp()),      # issued at
        "exp": int(expires_dt.timestamp()),  # expiration
        "jti": license_id,                # JWT ID (único, para revocación)

        # Claims custom Watermelon
        "customer": args.customer,
        "plan": args.plan,
        "plan_label": plan_cfg["label"],
        "modules": modules,
        "max_channels": max_channels,
    }

    # ------------------------------------------------------------------------
    # 5. Cargar private key y firmar JWT con RS256
    # ------------------------------------------------------------------------
    priv_pem = priv_path.read_bytes()
    private_key = serialization.load_pem_private_key(priv_pem, password=None)

    token = jwt.encode(payload, private_key, algorithm="RS256")

    # ------------------------------------------------------------------------
    # 6. Guardar outputs
    # ------------------------------------------------------------------------
    customer_slug = slugify(args.customer)
    out_dir = Path(__file__).parent / "licenses_issued" / customer_slug
    out_dir.mkdir(parents=True, exist_ok=True)

    token_path = out_dir / "license.token"
    json_path = out_dir / "license.json"
    readme_path = out_dir / "README_CLIENTE.txt"

    token_path.write_text(token, encoding="utf-8")

    # Registro interno (con datos sensibles + notas)
    record = {
        "license_id": license_id,
        "customer": args.customer,
        "customer_slug": customer_slug,
        "email": args.email,
        "plan": args.plan,
        "plan_label": plan_cfg["label"],
        "modules": modules,
        "max_channels": max_channels,
        "issued_at_utc": now.isoformat(),
        "expires_at_utc": expires_dt.isoformat(),
        "issued_by": "license_issue.py",
        "internal_notes": args.notes,
        "payload": payload,
    }
    json_path.write_text(
        json.dumps(record, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # README para enviar al cliente (instrucciones de instalación)
    readme = f"""WATERMELON PLANTA EDITION — LICENCIA ACTIVADA
================================================================

Cliente:    {args.customer}
Email:      {args.email}
Plan:       {plan_cfg["label"]}
Módulos:    {", ".join(modules)}
Canales:    hasta {max_channels} simultáneos
Vence:      {expires_dt.strftime("%d / %m / %Y")}

------------------------------------------------------------------
CÓMO INSTALAR LA LICENCIA
------------------------------------------------------------------

1. Localiza la carpeta de instalación de Watermelon Planta.
   Por defecto está en:
       C:\\Program Files\\Watermelon Planta\\

2. Adentro hay una carpeta llamada "data". Entra ahí.

3. Copia el archivo "license.token" (adjunto en este envío)
   dentro de esa carpeta "data".

   El resultado debe verse así:
       C:\\Program Files\\Watermelon Planta\\data\\license.token

4. Abre Watermelon Planta normalmente.
   En la pantalla de inicio verás:
       ✓ Licencia válida — {args.customer}
       Vence: {expires_dt.strftime("%d/%m/%Y")}

5. Si tu licencia está por vencer (menos de 30 días),
   contáctanos para renovar:
       ehernandez@sigasas.com

------------------------------------------------------------------
SOPORTE TÉCNICO
------------------------------------------------------------------

  SIGA GROUP SAS — Watermelon Division
  Email:   ehernandez@sigasas.com

Esta licencia es PERSONAL e INTRANSFERIBLE. No la compartas con
terceros. Cada licencia tiene un ID único de auditoría.

License ID: {license_id}
"""
    readme_path.write_text(readme, encoding="utf-8")

    # ------------------------------------------------------------------------
    # 7. Resumen en consola
    # ------------------------------------------------------------------------
    print()
    print("=" * 64)
    print("  ✓ LICENCIA EMITIDA")
    print("=" * 64)
    print()
    print(f"  Cliente:        {args.customer}")
    print(f"  Email:          {args.email}")
    print(f"  Plan:           {plan_cfg['label']}")
    print(f"  Módulos:        {', '.join(modules)}")
    print(f"  Max canales:    {max_channels}")
    print(f"  Vence:          {expires_dt.strftime('%Y-%m-%d')}")
    print(f"  License ID:     {license_id}")
    print()
    print(f"  Token size:     {len(token)} bytes")
    print()
    print("  ARCHIVOS GENERADOS:")
    print(f"    {token_path}")
    print(f"    {json_path}    (registro interno SIGA)")
    print(f"    {readme_path}  (instrucciones para el cliente)")
    print()
    print("=" * 64)
    print("  ENVIAR AL CLIENTE")
    print("=" * 64)
    print()
    print(f"  Enviar SOLAMENTE estos 2 archivos al cliente:")
    print(f"    1. license.token       (la licencia)")
    print(f"    2. README_CLIENTE.txt  (instrucciones de instalación)")
    print()
    print(f"  NO enviar license.json (es registro interno con notas).")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
