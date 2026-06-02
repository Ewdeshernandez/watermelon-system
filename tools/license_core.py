#!/usr/bin/env python3
"""
tools/license_core.py — Núcleo reutilizable de licencias Watermelon Planta
==========================================================================

Lógica compartida entre el CLI (`license_issue.py`) y la herramienta de
administración (`license_admin.py`). Centraliza:

  * Definición de planes comerciales (PLANS).
  * Emisión de licencias firmadas RSA-2048 (issue_license).
  * Listado de licencias ya emitidas con su estado (list_issued_licenses).
  * Verificación de un token contra la public key (verify_token).

⚠️ SEGURIDAD: este módulo usa la PRIVATE KEY (tools/.keys/private_key.pem)
para firmar. NUNCA debe correr en la nube ni distribuirse al cliente. Es
exclusivamente para uso interno de SIGA en una máquina de confianza.
"""
from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


_KEYS_DIR = Path(__file__).parent / ".keys"
_PRIVATE_KEY_PATH = _KEYS_DIR / "private_key.pem"
_PUBLIC_KEY_PATH = _KEYS_DIR / "public_key.pem"
_ISSUED_DIR = Path(__file__).parent / "licenses_issued"

_ISSUER = "SIGA GROUP SAS"
_AUDIENCE = "watermelon-planta"
_SUPPORT_EMAIL = "ehernandez@sigasas.com"


# ============================================================================
# PLANES COMERCIALES
# ============================================================================

PLANS: Dict[str, Dict[str, Any]] = {
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

# Descripción legible de cada módulo (para el README del cliente)
MODULE_LABELS = {
    "ema": "Análisis Modal Experimental (EMA)",
    "oma": "Análisis Modal Operacional (OMA)",
    "fea": "Correlación con elementos finitos (FEA)",
    "modes3d": "Visualización 3D de modos animados",
    "reports": "Reportes profesionales",
    "sync": "Sincronización con la nube",
}


# ============================================================================
# HELPERS
# ============================================================================

def slugify(text: str) -> str:
    """'Termoeléctrica Norte SAS' → 'termoelectrica-norte-sas'."""
    text = (text or "").lower().strip()
    for src, dst in {"á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u", "ñ": "n"}.items():
        text = text.replace(src, dst)
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")


def keys_exist() -> bool:
    return _PRIVATE_KEY_PATH.exists() and _PUBLIC_KEY_PATH.exists()


def _require_deps():
    try:
        import jwt  # noqa: F401  (PyJWT)
        from cryptography.hazmat.primitives import serialization  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            f"Faltan dependencias ({e}). Instalá con: pip install pyjwt cryptography"
        ) from e


# ============================================================================
# EMISIÓN
# ============================================================================

@dataclass
class IssueResult:
    license_id: str
    customer: str
    customer_slug: str
    token: str
    record: Dict[str, Any]
    token_path: Path
    json_path: Path
    readme_path: Path
    readme_text: str


def _build_readme(customer, email, plan_label, modules, max_channels,
                  expires_dt, license_id) -> str:
    mod_lines = "\n".join(f"     · {MODULE_LABELS.get(m, m)}" for m in modules)
    return f"""WATERMELON PLANTA EDITION — LICENCIA ACTIVADA
================================================================

Cliente:    {customer}
Email:      {email}
Plan:       {plan_label}
Módulos habilitados:
{mod_lines}
Canales:    hasta {max_channels} simultáneos
Vence:      {expires_dt.strftime("%d / %m / %Y")}

------------------------------------------------------------------
CÓMO INSTALAR LA LICENCIA
------------------------------------------------------------------

1. Localizá la carpeta de instalación de Watermelon Planta.
   Por defecto está en:
       C:\\Program Files\\Watermelon Planta\\

2. Adentro hay una carpeta llamada "data". Entrá ahí.

3. Copiá el archivo "license.token" (adjunto en este envío)
   dentro de esa carpeta "data". Debe quedar así:
       C:\\Program Files\\Watermelon Planta\\data\\license.token

4. Abrí Watermelon Planta normalmente. En la pantalla de inicio verás:
       ✓ Licencia válida — {customer}
       Vence: {expires_dt.strftime("%d/%m/%Y")}

5. Si tu licencia está por vencer (menos de 30 días), contactanos
   para renovarla: {_SUPPORT_EMAIL}

------------------------------------------------------------------
SOPORTE TÉCNICO
------------------------------------------------------------------

  SIGA GROUP SAS — Watermelon Division
  Email:   {_SUPPORT_EMAIL}

Esta licencia es PERSONAL e INTRANSFERIBLE. No la compartas con
terceros. Cada licencia tiene un ID único de auditoría.

License ID: {license_id}
"""


def issue_license(
    customer: str,
    email: str,
    plan: str = "pro",
    expires_dt: Optional[datetime] = None,
    modules: Optional[List[str]] = None,
    max_channels: Optional[int] = None,
    notes: str = "",
) -> IssueResult:
    """Emite (firma + guarda) una licencia. Devuelve IssueResult.

    Lanza RuntimeError/ValueError si falta la key, deps, o args inválidos.
    """
    if not customer or not email or "@" not in email:
        raise ValueError("Cliente y email válido son obligatorios.")
    if plan not in PLANS:
        raise ValueError(f"Plan inválido '{plan}'. Válidos: {list(PLANS)}")
    if not _PRIVATE_KEY_PATH.exists():
        raise RuntimeError(
            f"No existe la private key en {_PRIVATE_KEY_PATH}. "
            "Generá el par con: python tools/license_keygen.py"
        )
    _require_deps()

    import jwt
    from cryptography.hazmat.primitives import serialization

    plan_cfg = PLANS[plan]
    mods = list(modules) if modules else list(plan_cfg["default_modules"])
    mods = [m.strip().lower() for m in mods if m.strip()]
    invalid = set(mods) - VALID_MODULES
    if invalid:
        raise ValueError(f"Módulos inválidos: {invalid}. Válidos: {sorted(VALID_MODULES)}")
    if not mods:
        raise ValueError("Debe haber al menos un módulo habilitado.")

    chans = int(max_channels) if max_channels is not None else int(plan_cfg["default_max_channels"])

    now = datetime.now(timezone.utc)
    if expires_dt is None:
        expires_dt = now + timedelta(days=plan_cfg["default_duration_days"])
    if expires_dt.tzinfo is None:
        expires_dt = expires_dt.replace(tzinfo=timezone.utc)
    if expires_dt <= now:
        raise ValueError("La fecha de vencimiento debe ser futura.")

    license_id = str(uuid.uuid4())
    payload = {
        "iss": _ISSUER,
        "sub": email,
        "aud": _AUDIENCE,
        "iat": int(now.timestamp()),
        "exp": int(expires_dt.timestamp()),
        "jti": license_id,
        "customer": customer,
        "plan": plan,
        "plan_label": plan_cfg["label"],
        "modules": mods,
        "max_channels": chans,
    }

    private_key = serialization.load_pem_private_key(
        _PRIVATE_KEY_PATH.read_bytes(), password=None
    )
    token = jwt.encode(payload, private_key, algorithm="RS256")

    customer_slug = slugify(customer)
    out_dir = _ISSUED_DIR / customer_slug
    out_dir.mkdir(parents=True, exist_ok=True)
    token_path = out_dir / "license.token"
    json_path = out_dir / "license.json"
    readme_path = out_dir / "README_CLIENTE.txt"

    record = {
        "license_id": license_id,
        "customer": customer,
        "customer_slug": customer_slug,
        "email": email,
        "plan": plan,
        "plan_label": plan_cfg["label"],
        "modules": mods,
        "max_channels": chans,
        "issued_at_utc": now.isoformat(),
        "expires_at_utc": expires_dt.isoformat(),
        "issued_by": "license_core.issue_license",
        "internal_notes": notes,
        "payload": payload,
    }
    readme_text = _build_readme(customer, email, plan_cfg["label"], mods,
                                chans, expires_dt, license_id)

    token_path.write_text(token, encoding="utf-8")
    json_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    readme_path.write_text(readme_text, encoding="utf-8")

    return IssueResult(
        license_id=license_id, customer=customer, customer_slug=customer_slug,
        token=token, record=record, token_path=token_path, json_path=json_path,
        readme_path=readme_path, readme_text=readme_text,
    )


# ============================================================================
# LISTADO + ESTADO
# ============================================================================

def _status_for(expires_at_utc: str) -> Dict[str, Any]:
    try:
        exp = datetime.fromisoformat(expires_at_utc)
        if exp.tzinfo is None:
            exp = exp.replace(tzinfo=timezone.utc)
    except Exception:
        return {"status": "DESCONOCIDO", "days_left": None}
    days_left = (exp - datetime.now(timezone.utc)).days
    if days_left < 0:
        status = "VENCIDA"
    elif days_left <= 30:
        status = "POR VENCER"
    else:
        status = "VIGENTE"
    return {"status": status, "days_left": days_left}


def list_issued_licenses() -> List[Dict[str, Any]]:
    """Lista todas las licencias emitidas (lee tools/licenses_issued/*/license.json)."""
    out: List[Dict[str, Any]] = []
    if not _ISSUED_DIR.exists():
        return out
    for child in sorted(_ISSUED_DIR.iterdir()):
        meta = child / "license.json"
        if not meta.exists():
            continue
        try:
            rec = json.loads(meta.read_text(encoding="utf-8"))
        except Exception:
            continue
        st = _status_for(rec.get("expires_at_utc", ""))
        rec["_status"] = st["status"]
        rec["_days_left"] = st["days_left"]
        rec["_dir"] = str(child)
        out.append(rec)
    # Ordenar: por vencer/vencidas primero
    order = {"VENCIDA": 0, "POR VENCER": 1, "VIGENTE": 2, "DESCONOCIDO": 3}
    out.sort(key=lambda r: (order.get(r.get("_status"), 9), r.get("_days_left") or 0))
    return out


# ============================================================================
# VERIFICACIÓN (con public key — igual que hace el cliente)
# ============================================================================

def verify_token(token: str) -> Dict[str, Any]:
    """Verifica un token contra la public key. Devuelve {valid, reason, claims}."""
    _require_deps()
    import jwt
    if not _PUBLIC_KEY_PATH.exists():
        return {"valid": False, "reason": "No existe public_key.pem", "claims": None}
    pub_pem = _PUBLIC_KEY_PATH.read_text(encoding="utf-8")
    try:
        claims = jwt.decode(
            token.strip(), pub_pem, algorithms=["RS256"],
            audience=_AUDIENCE, issuer=_ISSUER,
            options={"require": ["exp", "iat", "iss", "aud"]},
            leeway=0,
        )
        return {"valid": True, "reason": "Licencia válida", "claims": claims}
    except Exception as e:  # jwt.ExpiredSignatureError, InvalidToken, etc.
        return {"valid": False, "reason": f"{type(e).__name__}: {e}", "claims": None}


__all__ = [
    "PLANS", "VALID_MODULES", "MODULE_LABELS", "IssueResult",
    "issue_license", "list_issued_licenses", "verify_token",
    "slugify", "keys_exist",
]
