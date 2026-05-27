"""
planta/license_manager.py — Verificador de licencias RSA (runtime cliente)
============================================================================

Se ejecuta en cada arranque de Watermelon Planta Edition en el equipo del
cliente. Verifica que el archivo `data/license.token` exista, sea un JWT
RS256 firmado por la private key de SIGA, esté vigente, y devuelve la
estructura de features que el cliente compró.

ARQUITECTURA:
  • Public key EMBEBIDA en este archivo (hardcoded como string Python).
    Cuando se hace `pyinstaller`, queda dentro del .exe — no editable
    sin recompilar y firmar de nuevo.
  • Token JWT vive en `<install_dir>/data/license.token`.
  • Verificación OFFLINE: no requiere red, todo es criptografía local.
  • Si el token está corrupto / vencido / no existe → app entra en modo
    bloqueado con instrucciones de contacto.

SEGURIDAD:
  • RS256 (RSA-2048) — atacante necesita la private key de SIGA para
    falsificar una licencia. Sin ella, modificar el token rompe la firma.
  • `aud=watermelon-planta` — token de otra app no funciona aquí.
  • `iss=SIGA GROUP SAS` — verificamos issuer.
  • Tolerancia de reloj: 0 (sin leeway). Cliente con reloj atrasado >
    fecha de emisión = inválido. Esto evita ataques de roll-back.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional


# ============================================================================
# PUBLIC KEY EMBEBIDA (RSA-2048)
# ============================================================================
# IMPORTANTE: NO MODIFICAR A MANO. Si hay que rotar:
#   1. Generar nuevo par con tools/license_keygen.py
#   2. Reemplazar el contenido entre BEGIN/END abajo
#   3. RECOMPILAR el .exe (todas las licencias viejas dejarán de funcionar)
#   4. Reemitir TODAS las licencias activas con la nueva private key

_EMBEDDED_PUBLIC_KEY = """-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA2pghc0DWF6nfNvD+dC8i
zs7rT/eeZJj9S4GKRzFY3zmDPiAkehJnZcqcsYWkV2flRxn4E5+qLi2l2il31Peh
FnuQThqv4CdETOTUZR7yYotyLc4dpjq+ZEVJcRN305GsUV1EZlojUZCh8E9j4GNS
e6AClPf8FYWIHJJwYso1FyT0RZWQAFrpAreOPbPQD9e7C/XfGEJxicytkafr4UUP
Pdl1zjeX7Lsfu4hIjPfP+CJcc806GqMr1fhB9JFJKRmGWTPRSmWBVX43eD/sRszm
aCOLAgemyCKDawsKoZn1jdVntqgSoDz/IzaNbuzUqR2tU/8bzA3BIcm1xDFXaFfo
lQIDAQAB
-----END PUBLIC KEY-----"""


# ============================================================================
# CONFIGURACIÓN
# ============================================================================
# Path del token relativo a este archivo. Cuando PyInstaller empaqueta,
# `planta/` queda dentro del .exe pero `data/` queda externa (writable).
# Para soportar ambos modos (dev + bundled), buscamos el data dir así:
#   1. Si existe variable de entorno WATERMELON_DATA_DIR → usarla
#   2. Si estamos en exe (sys.frozen) → <exe_dir>/data/
#   3. Default → <repo_root>/planta/data/

LICENSE_FILENAME = "license.token"
EXPIRY_WARNING_DAYS = 30  # Avisar al user cuando quedan ≤30 días

ALL_MODULES = ["ema", "oma", "fea", "modes3d", "reports", "sync"]

# ============================================================================
# REVOCACIÓN ONLINE (FASE J v3.31.221)
# ============================================================================
# Endpoint público de SIGA que devuelve si una licencia fue revocada.
# Planta llama acá al arrancar si hay internet — silencioso si no hay red.
REVOCATION_CHECK_URL = (
    "https://yxeqwkhybueelmkrdkgq.supabase.co/functions/v1/license-check"
)
REVOCATION_CHECK_TIMEOUT_S = 5  # No bloqueamos el arranque más de 5s
# Cache local del último check exitoso (en data/.revocation_cache.json)
REVOCATION_CACHE_FILENAME = ".revocation_cache.json"
# Frecuencia: chequeamos online máximo 1 vez cada 24h
REVOCATION_CHECK_INTERVAL_HOURS = 24
# Grace periods: si no podemos validar en N días, qué hacer
REVOCATION_GRACE_WARN_DAYS = 7    # Warning amarillo
REVOCATION_GRACE_BLOCK_DAYS = 30  # Bloqueo total (evita ataque offline)


# ============================================================================
# DATA CLASS
# ============================================================================

@dataclass
class LicenseInfo:
    """Resultado de verificar una licencia."""
    valid: bool
    customer: str = ""
    email: str = ""
    plan: str = ""
    plan_label: str = ""
    modules: list = field(default_factory=list)
    max_channels: int = 0
    issued_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    license_id: str = ""
    # Si valid=False, este campo explica por qué (para mostrar al user)
    error_reason: str = ""
    # Si valid=True pero quedan pocos días, este flag se prende
    expires_soon: bool = False
    days_until_expiry: int = 0
    # Estado de revocación online (FASE J v3.31.221)
    # Si valid=False y revoked=True, la causa fue revocación remota por SIGA
    revoked: bool = False
    revocation_reason: str = ""
    # Si valid=True pero hace muchos días que no podemos validar online
    revocation_stale_days: int = 0
    revocation_stale_warning: bool = False

    def has_module(self, module_name: str) -> bool:
        """¿La licencia incluye este módulo? Ej: license.has_module('oma')."""
        return self.valid and module_name in self.modules

    def to_dict(self) -> dict:
        """Serializable para st.session_state o caching."""
        return {
            "valid": self.valid,
            "revoked": self.revoked,
            "revocation_reason": self.revocation_reason,
            "revocation_stale_days": self.revocation_stale_days,
            "revocation_stale_warning": self.revocation_stale_warning,
            "customer": self.customer,
            "email": self.email,
            "plan": self.plan,
            "plan_label": self.plan_label,
            "modules": list(self.modules),
            "max_channels": self.max_channels,
            "issued_at": self.issued_at.isoformat() if self.issued_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "license_id": self.license_id,
            "error_reason": self.error_reason,
            "expires_soon": self.expires_soon,
            "days_until_expiry": self.days_until_expiry,
        }


# ============================================================================
# RESOLVER PATHS
# ============================================================================

def _get_data_dir() -> Path:
    """Determina dónde está la carpeta data/ tanto en dev como en .exe bundled."""
    import os
    import sys

    # 1. Override por variable de entorno (útil para tests / multi-install)
    env_dir = os.environ.get("WATERMELON_DATA_DIR")
    if env_dir:
        return Path(env_dir)

    # 2. Modo bundled (PyInstaller): <exe_dir>/data/
    if getattr(sys, "frozen", False):
        # sys.executable apunta al .exe; data/ está al lado
        exe_dir = Path(sys.executable).parent
        return exe_dir / "data"

    # 3. Modo dev: <repo>/planta/data/
    return Path(__file__).parent / "data"


def get_license_path() -> Path:
    """Devuelve la ruta esperada del archivo license.token."""
    return _get_data_dir() / LICENSE_FILENAME


def _get_revocation_cache_path() -> Path:
    """Path del cache de revocación (en data/, gitignored)."""
    return _get_data_dir() / REVOCATION_CACHE_FILENAME


# ============================================================================
# REVOCACIÓN ONLINE (FASE J v3.31.221)
# ============================================================================

def _read_revocation_cache() -> Optional[dict]:
    """Lee el cache local. None si no existe o está corrupto."""
    p = _get_revocation_cache_path()
    if not p.exists():
        return None
    try:
        import json
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _write_revocation_cache(license_id: str, status: str,
                              reason: str = "") -> None:
    """Guarda el resultado del último check exitoso."""
    p = _get_revocation_cache_path()
    try:
        import json
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps({
                "license_id": license_id,
                "status": status,        # "active" o "revoked"
                "reason": reason,
                "checked_at": datetime.now(timezone.utc).isoformat(),
            }, indent=2),
            encoding="utf-8",
        )
    except OSError:
        pass  # no es fatal


def check_revocation_status(license_id: str) -> dict:
    """
    Chequea si la licencia fue revocada por SIGA.

    Comportamiento:
      1. Si hay cache reciente (<24h) → usa cache
      2. Si no, llama al endpoint SIGA (timeout 5s)
      3. Si endpoint responde "revoked" → cachea y devuelve revoked
      4. Si endpoint responde "active" → cachea y devuelve active
      5. Si endpoint falla (sin internet, server down) → usa cache viejo
         o devuelve "unknown"

    Returns:
        {
            "status": "active" | "revoked" | "unknown",
            "reason": str,
            "stale_days": int,       # días desde último check exitoso
            "from_cache": bool,
            "checked_online": bool,
        }
    """
    cache = _read_revocation_cache()

    # 1. Verificar si cache es del mismo license_id y reciente
    if cache and cache.get("license_id") == license_id:
        try:
            cached_at = datetime.fromisoformat(cache["checked_at"])
            age = datetime.now(timezone.utc) - cached_at
            if age < timedelta(hours=REVOCATION_CHECK_INTERVAL_HOURS):
                # Cache fresco — usarlo directo
                return {
                    "status": cache.get("status", "unknown"),
                    "reason": cache.get("reason", ""),
                    "stale_days": age.days,
                    "from_cache": True,
                    "checked_online": False,
                }
        except (KeyError, ValueError):
            pass  # cache corrupto, ignorar

    # 2. Hacer la request HTTP al endpoint SIGA
    import urllib.parse
    import urllib.request
    import urllib.error
    import json

    url = (
        f"{REVOCATION_CHECK_URL}?"
        f"{urllib.parse.urlencode({'jti': license_id})}"
    )

    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "WatermelonPlanta-LicenseCheck"},
        )
        with urllib.request.urlopen(req, timeout=REVOCATION_CHECK_TIMEOUT_S) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        status = data.get("status", "unknown")
        reason = data.get("reason", "")

        # 3. Cachear el resultado
        _write_revocation_cache(license_id, status, reason)

        return {
            "status": status,
            "reason": reason,
            "stale_days": 0,
            "from_cache": False,
            "checked_online": True,
        }

    except (urllib.error.URLError, urllib.error.HTTPError,
            TimeoutError, ConnectionError, OSError,
            json.JSONDecodeError, Exception):  # noqa: BLE001
        # 4. Fallback al cache si lo hay (puede estar viejo)
        if cache and cache.get("license_id") == license_id:
            try:
                cached_at = datetime.fromisoformat(cache["checked_at"])
                age = datetime.now(timezone.utc) - cached_at
                return {
                    "status": cache.get("status", "unknown"),
                    "reason": cache.get("reason", ""),
                    "stale_days": age.days,
                    "from_cache": True,
                    "checked_online": False,
                }
            except (KeyError, ValueError):
                pass

        # Sin cache + sin internet — desconocido
        return {
            "status": "unknown",
            "reason": "",
            "stale_days": 0,
            "from_cache": False,
            "checked_online": False,
        }


# ============================================================================
# VERIFICACIÓN
# ============================================================================

def verify_license(token_path: Optional[Path] = None) -> LicenseInfo:
    """
    Verifica la licencia activa.

    Args:
        token_path: si se da, usar ese path. Si no, autodetectar.

    Returns:
        LicenseInfo con valid=True/False y todos los campos parseados.
        Si valid=False, error_reason explica el problema en español.
    """
    if token_path is None:
        token_path = get_license_path()

    # ------------------------------------------------------------------------
    # 1. ¿Existe el archivo?
    # ------------------------------------------------------------------------
    if not token_path.exists():
        return LicenseInfo(
            valid=False,
            error_reason=(
                f"No se encontró el archivo de licencia.\n\n"
                f"Esperaba encontrarlo en:\n  {token_path}\n\n"
                f"Para activar Watermelon Planta, pega el archivo "
                f"'license.token' que recibiste de SIGA GROUP en la carpeta "
                f"'data/' de tu instalación."
            ),
        )

    # ------------------------------------------------------------------------
    # 2. ¿Está PyJWT disponible?
    # ------------------------------------------------------------------------
    try:
        import jwt
    except ImportError:
        return LicenseInfo(
            valid=False,
            error_reason=(
                "Falta la dependencia 'pyjwt' en el sistema.\n"
                "Esto indica una instalación corrupta de Watermelon Planta.\n"
                "Contacta a SIGA GROUP para reinstalar."
            ),
        )

    # ------------------------------------------------------------------------
    # 3. Leer token
    # ------------------------------------------------------------------------
    try:
        token = token_path.read_text(encoding="utf-8").strip()
    except Exception as e:  # noqa: BLE001
        return LicenseInfo(
            valid=False,
            error_reason=(
                f"No se pudo leer el archivo de licencia.\n\n"
                f"Error técnico: {e}\n\n"
                f"Verifica que el archivo no esté corrupto o bloqueado por "
                f"permisos del sistema."
            ),
        )

    if not token:
        return LicenseInfo(
            valid=False,
            error_reason="El archivo de licencia está vacío.",
        )

    # ------------------------------------------------------------------------
    # 4. Verificar firma + claims standard
    # ------------------------------------------------------------------------
    try:
        payload = jwt.decode(
            token,
            _EMBEDDED_PUBLIC_KEY,
            algorithms=["RS256"],
            audience="watermelon-planta",
            issuer="SIGA GROUP SAS",
            options={
                "require": ["exp", "iat", "iss", "aud", "sub"],
                "verify_exp": True,
                "verify_iat": True,
                "verify_iss": True,
                "verify_aud": True,
                "verify_signature": True,
            },
            leeway=0,  # Sin tolerancia de reloj — evita roll-back attacks
        )
    except jwt.ExpiredSignatureError:
        return LicenseInfo(
            valid=False,
            error_reason=(
                "Tu licencia ha vencido.\n\n"
                "Para renovar, contacta a SIGA GROUP:\n"
                "  ehernandez@sigasas.com"
            ),
        )
    except jwt.InvalidAudienceError:
        return LicenseInfo(
            valid=False,
            error_reason=(
                "El token no corresponde a Watermelon Planta.\n"
                "Verifica que estás usando la licencia correcta."
            ),
        )
    except jwt.InvalidIssuerError:
        return LicenseInfo(
            valid=False,
            error_reason=(
                "El token no fue emitido por SIGA GROUP.\n"
                "Esto indica una licencia falsificada o manipulada."
            ),
        )
    except jwt.InvalidSignatureError:
        return LicenseInfo(
            valid=False,
            error_reason=(
                "La firma de la licencia no es válida.\n\n"
                "Esto puede pasar si:\n"
                "  • El archivo fue editado manualmente\n"
                "  • Es una licencia de otra versión de Watermelon\n"
                "  • Hubo corrupción de datos\n\n"
                "Contacta a SIGA GROUP para reemitir tu licencia."
            ),
        )
    except jwt.PyJWTError as e:
        return LicenseInfo(
            valid=False,
            error_reason=(
                f"La licencia no es válida.\n\n"
                f"Detalle técnico: {type(e).__name__}: {e}\n\n"
                f"Contacta a SIGA GROUP si crees que es un error."
            ),
        )

    # ------------------------------------------------------------------------
    # 5. Parsear claims custom + computar warnings
    # ------------------------------------------------------------------------
    now = datetime.now(timezone.utc)

    try:
        issued_at = datetime.fromtimestamp(int(payload["iat"]), tz=timezone.utc)
        expires_at = datetime.fromtimestamp(int(payload["exp"]), tz=timezone.utc)
    except (KeyError, ValueError, TypeError) as e:
        return LicenseInfo(
            valid=False,
            error_reason=f"Fechas de licencia corruptas: {e}",
        )

    modules_raw = payload.get("modules", [])
    if not isinstance(modules_raw, list):
        modules_raw = []
    modules = [str(m).lower() for m in modules_raw if isinstance(m, str)]

    try:
        max_channels = int(payload.get("max_channels", 0))
    except (ValueError, TypeError):
        max_channels = 0

    days_left = (expires_at - now).days
    expires_soon = 0 <= days_left <= EXPIRY_WARNING_DAYS
    license_id = str(payload.get("jti", ""))

    # ------------------------------------------------------------------------
    # 6. Verificar revocación online (FASE J v3.31.221)
    # ------------------------------------------------------------------------
    revocation_info = check_revocation_status(license_id) if license_id else {
        "status": "unknown", "reason": "", "stale_days": 0,
        "from_cache": False, "checked_online": False,
    }

    # Si SIGA la revocó explícitamente → invalidar la licencia
    if revocation_info["status"] == "revoked":
        return LicenseInfo(
            valid=False,
            revoked=True,
            revocation_reason=revocation_info["reason"] or "Sin motivo especificado",
            error_reason=(
                f"Tu licencia fue revocada por SIGA GROUP.\n\n"
                f"Motivo: {revocation_info['reason'] or 'No especificado'}\n\n"
                f"Si crees que es un error, contacta a:\n"
                f"  ehernandez@sigasas.com\n\n"
                f"License ID: {license_id}"
            ),
            customer=str(payload.get("customer", "")),
            email=str(payload.get("sub", "")),
            license_id=license_id,
        )

    # Si llevamos demasiados días sin poder validar online → bloquear
    if revocation_info["stale_days"] >= REVOCATION_GRACE_BLOCK_DAYS:
        return LicenseInfo(
            valid=False,
            error_reason=(
                f"No se ha podido validar tu licencia con SIGA GROUP en los "
                f"últimos {revocation_info['stale_days']} días.\n\n"
                f"Por seguridad, Watermelon Planta requiere conectarse a "
                f"internet al menos una vez cada {REVOCATION_GRACE_BLOCK_DAYS} "
                f"días para validar.\n\n"
                f"Conecta este equipo a internet temporalmente para "
                f"reactivar tu licencia."
            ),
            customer=str(payload.get("customer", "")),
            email=str(payload.get("sub", "")),
            license_id=license_id,
        )

    # Warning si llevamos varios días sin validar (no bloquea aún)
    stale_warning = (
        revocation_info["stale_days"] >= REVOCATION_GRACE_WARN_DAYS
        and not revocation_info["checked_online"]
    )

    return LicenseInfo(
        valid=True,
        customer=str(payload.get("customer", "")),
        email=str(payload.get("sub", "")),
        plan=str(payload.get("plan", "")),
        plan_label=str(payload.get("plan_label", "")),
        modules=modules,
        max_channels=max_channels,
        issued_at=issued_at,
        expires_at=expires_at,
        license_id=license_id,
        expires_soon=expires_soon,
        days_until_expiry=days_left,
        revocation_stale_days=revocation_info["stale_days"],
        revocation_stale_warning=stale_warning,
    )


# ============================================================================
# UI HELPERS (Streamlit)
# ============================================================================

def render_license_blocker(info: LicenseInfo) -> None:
    """
    Pantalla bloqueante cuando la licencia es inválida.

    v3.31.255 — Rediseño completo clase mundial (tipo Adobe/Autodesk).
    Hero card con gradient, logo SVG, mensaje claro, ruta del archivo
    y contacto. Sin st.error feo, sin markdown crudo.

    Llamar al inicio de app_planta.py:

        info = verify_license()
        if not info.valid:
            render_license_blocker(info)
            st.stop()
    """
    import streamlit as st
    import html as _html

    expected_path = get_license_path()

    # Ocultar TODO el chrome de Streamlit en esta pantalla — solo el card
    st.markdown(
        """
        <style>
            [data-testid="stHeader"] {display: none !important;}
            [data-testid="stToolbar"] {display: none !important;}
            section[data-testid="stSidebar"] {display: none !important;}
            .block-container {
                max-width: 720px !important;
                padding-top: 4rem !important;
                padding-bottom: 2rem !important;
            }
            footer {visibility: hidden !important;}
            #MainMenu {visibility: hidden !important;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    _reason = _html.escape(info.error_reason or "Licencia no encontrada o inválida")
    _path = _html.escape(str(expected_path))

    st.markdown(
        f"""
        <div style="
            background:linear-gradient(135deg,#1e3a8a 0%,#1e40af 45%,#0f766e 100%);
            border-radius:20px;
            padding:40px 44px 36px;
            color:white;
            box-shadow:0 12px 40px rgba(15,118,110,0.28),
                       0 4px 16px rgba(30,58,138,0.22);
            position:relative;
            overflow:hidden;
            margin-bottom:24px;">
            <!-- Pattern decorativo -->
            <div style="position:absolute;top:-60px;right:-60px;
                        width:240px;height:240px;border-radius:50%;
                        background:radial-gradient(circle,
                                                    rgba(255,255,255,0.10) 0%,
                                                    rgba(255,255,255,0) 70%);
                        pointer-events:none;"></div>
            <!-- Logo + título -->
            <div style="display:flex;align-items:center;gap:18px;
                        position:relative;margin-bottom:24px;">
                <svg width="48" height="48" viewBox="0 0 256 256"
                     xmlns="http://www.w3.org/2000/svg" style="flex-shrink:0;">
                    <defs>
                        <linearGradient id="wmg" x1="0" y1="0" x2="1" y2="1">
                            <stop offset="0%" stop-color="#f97316"/>
                            <stop offset="100%" stop-color="#ef4444"/>
                        </linearGradient>
                    </defs>
                    <circle cx="128" cy="128" r="100"
                            fill="url(#wmg)" opacity="0.95"/>
                    <circle cx="128" cy="128" r="100"
                            fill="none" stroke="white"
                            stroke-width="6" opacity="0.85"/>
                    <path d="M48 128 Q128 60 208 128"
                          fill="none" stroke="white"
                          stroke-width="5" opacity="0.7"/>
                </svg>
                <div>
                    <div style="font-size:22px;font-weight:800;
                                letter-spacing:-0.3px;line-height:1.1;">
                        Watermelon Planta Edition
                    </div>
                    <div style="font-size:12px;opacity:0.75;
                                letter-spacing:1.2px;text-transform:uppercase;
                                margin-top:4px;font-weight:600;">
                        SIGA GROUP · Análisis de Vibraciones Industrial
                    </div>
                </div>
            </div>
            <!-- Mensaje principal -->
            <div style="background:rgba(0,0,0,0.18);
                        border:1px solid rgba(255,255,255,0.12);
                        border-radius:12px;padding:20px 22px;
                        position:relative;">
                <div style="display:flex;align-items:center;gap:10px;
                            margin-bottom:8px;">
                    <div style="width:10px;height:10px;border-radius:50%;
                                background:#fbbf24;
                                box-shadow:0 0 12px rgba(251,191,36,0.6);"></div>
                    <div style="font-size:11px;font-weight:700;
                                letter-spacing:1.8px;text-transform:uppercase;
                                color:#fbbf24;">
                        Acceso bloqueado · Licencia requerida
                    </div>
                </div>
                <div style="font-size:16px;font-weight:600;line-height:1.45;
                            color:white;">
                    {_reason}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Card de instrucciones
    st.markdown(
        f"""
        <div style="
            background:white;
            border:1px solid #e2e8f0;
            border-radius:14px;
            padding:24px 26px;
            box-shadow:0 1px 3px rgba(0,0,0,0.04),
                       0 4px 12px rgba(15,23,42,0.05);
            margin-bottom:18px;">
            <div style="font-size:11px;font-weight:700;letter-spacing:1.5px;
                        text-transform:uppercase;color:#0f766e;
                        margin-bottom:14px;">
                Activar licencia
            </div>
            <div style="font-size:15px;color:#0f172a;line-height:1.55;
                        margin-bottom:18px;font-weight:500;">
                Copia tu archivo <code style="background:#f1f5f9;
                padding:2px 7px;border-radius:5px;font-size:13px;
                color:#0f766e;font-weight:600;">license.token</code>
                en la siguiente ubicación y reinicia la aplicación:
            </div>
            <div style="background:#0f172a;color:#5eead4;
                        padding:14px 18px;border-radius:10px;
                        font-family:'SF Mono','Consolas','Monaco',monospace;
                        font-size:13px;word-break:break-all;
                        border-left:3px solid #14b8a6;">
                {_path}
            </div>
        </div>

        <div style="
            background:#f8fafc;
            border:1px solid #e2e8f0;
            border-radius:14px;
            padding:22px 26px;">
            <div style="font-size:11px;font-weight:700;letter-spacing:1.5px;
                        text-transform:uppercase;color:#475569;
                        margin-bottom:14px;">
                ¿Necesitas ayuda?
            </div>
            <div style="display:grid;grid-template-columns:1fr;gap:12px;">
                <div style="display:flex;align-items:flex-start;gap:12px;
                            padding:12px 14px;background:white;
                            border:1px solid #e2e8f0;border-radius:10px;">
                    <div style="font-size:18px;flex-shrink:0;">📧</div>
                    <div>
                        <div style="font-size:11px;color:#64748b;
                                    font-weight:600;letter-spacing:0.5px;
                                    text-transform:uppercase;">
                            Soporte técnico
                        </div>
                        <div style="font-size:14px;color:#0f172a;
                                    font-weight:600;margin-top:2px;">
                            <a href="mailto:ehernandez@sigasas.com"
                               style="color:#0f766e;text-decoration:none;">
                                ehernandez@sigasas.com
                            </a>
                        </div>
                    </div>
                </div>
                <div style="display:flex;align-items:flex-start;gap:12px;
                            padding:12px 14px;background:white;
                            border:1px solid #e2e8f0;border-radius:10px;">
                    <div style="font-size:18px;flex-shrink:0;">🏢</div>
                    <div>
                        <div style="font-size:11px;color:#64748b;
                                    font-weight:600;letter-spacing:0.5px;
                                    text-transform:uppercase;">
                            Cliente nuevo
                        </div>
                        <div style="font-size:14px;color:#0f172a;
                                    margin-top:2px;line-height:1.45;">
                            Contacta a SIGA GROUP para tu cotización y la
                            generación de tu licencia personalizada.
                        </div>
                    </div>
                </div>
                <div style="display:flex;align-items:flex-start;gap:12px;
                            padding:12px 14px;background:white;
                            border:1px solid #e2e8f0;border-radius:10px;">
                    <div style="font-size:18px;flex-shrink:0;">📨</div>
                    <div>
                        <div style="font-size:11px;color:#64748b;
                                    font-weight:600;letter-spacing:0.5px;
                                    text-transform:uppercase;">
                            Cliente existente
                        </div>
                        <div style="font-size:14px;color:#0f172a;
                                    margin-top:2px;line-height:1.45;">
                            Revisa tu correo — la licencia fue enviada como
                            archivo adjunto llamado
                            <code style="background:#f1f5f9;padding:1px 6px;
                            border-radius:4px;font-size:12px;color:#0f766e;
                            font-weight:600;">license.token</code>.
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <div style="text-align:center;margin-top:24px;
                    font-size:11px;color:#94a3b8;
                    letter-spacing:1.2px;text-transform:uppercase;
                    font-weight:600;">
            © SIGA GROUP S.A.S · Watermelon System
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_license_status_chip(info: LicenseInfo) -> None:
    """
    Chip compacto para mostrar en el header (cuando la licencia ES válida).

    Si quedan ≤30 días, muestra warning amarillo.
    Si es enterprise, muestra dorado. Otros: verde.
    """
    import streamlit as st

    if not info.valid:
        return

    if info.expires_soon:
        color = "#f59e0b"  # ámbar
        bg = "rgba(245, 158, 11, 0.12)"
        msg = f"Licencia vence en {info.days_until_expiry} días"
    elif info.plan == "enterprise":
        color = "#eab308"  # dorado
        bg = "rgba(234, 179, 8, 0.10)"
        msg = f"Enterprise · {info.customer}"
    elif info.plan == "trial":
        color = "#3b82f6"  # azul
        bg = "rgba(59, 130, 246, 0.10)"
        msg = f"Trial · {info.days_until_expiry} días restantes"
    else:
        color = "#10b981"  # verde
        bg = "rgba(16, 185, 129, 0.10)"
        msg = f"{info.plan_label} · {info.customer}"

    expires_str = info.expires_at.strftime("%d/%m/%Y") if info.expires_at else "—"

    st.markdown(
        f"""
        <div style="
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 6px 12px;
            background: {bg};
            border: 1px solid {color};
            border-radius: 16px;
            font-size: 13px;
            color: {color};
            font-weight: 600;
        ">
            <span>✓</span>
            <span>{msg}</span>
            <span style="opacity: 0.7;">· vence {expires_str}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================================
# CACHE EN SESSION STATE (opcional, para evitar releer disco en cada rerun)
# ============================================================================

_SESSION_KEY = "_watermelon_license_info"


def get_cached_license() -> LicenseInfo:
    """
    Devuelve la licencia desde st.session_state si está cacheada,
    o la verifica y cachea. Llamar desde Streamlit.
    """
    try:
        import streamlit as st
    except ImportError:
        return verify_license()

    if _SESSION_KEY not in st.session_state:
        st.session_state[_SESSION_KEY] = verify_license()
    return st.session_state[_SESSION_KEY]


def clear_license_cache() -> None:
    """Forzar reverificación (ej: después de pegar un token nuevo)."""
    try:
        import streamlit as st
        if _SESSION_KEY in st.session_state:
            del st.session_state[_SESSION_KEY]
    except ImportError:
        pass


# ============================================================================
# CLI DE DEBUG
# ============================================================================

if __name__ == "__main__":
    import sys

    info = verify_license()
    print()
    print("=" * 60)
    print("  WATERMELON PLANTA — Verificación de licencia")
    print("=" * 60)
    print()
    print(f"  Token esperado en: {get_license_path()}")
    print()
    if info.valid:
        print("  ✓ LICENCIA VÁLIDA")
        print()
        print(f"    Cliente:      {info.customer}")
        print(f"    Email:        {info.email}")
        print(f"    Plan:         {info.plan_label}")
        print(f"    Módulos:      {', '.join(info.modules)}")
        print(f"    Max canales:  {info.max_channels}")
        print(f"    Emitida:      {info.issued_at}")
        print(f"    Vence:        {info.expires_at}")
        print(f"    Días left:    {info.days_until_expiry}")
        if info.expires_soon:
            print(f"    ⚠ VENCE PRONTO ({info.days_until_expiry} días)")
        print(f"    License ID:   {info.license_id}")
        sys.exit(0)
    else:
        print("  ✗ LICENCIA NO VÁLIDA")
        print()
        print(f"  Razón:")
        for line in info.error_reason.split("\n"):
            print(f"    {line}")
        sys.exit(1)
