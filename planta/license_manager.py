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
from datetime import datetime, timezone
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

    def has_module(self, module_name: str) -> bool:
        """¿La licencia incluye este módulo? Ej: license.has_module('oma')."""
        return self.valid and module_name in self.modules

    def to_dict(self) -> dict:
        """Serializable para st.session_state o caching."""
        return {
            "valid": self.valid,
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
        license_id=str(payload.get("jti", "")),
        expires_soon=expires_soon,
        days_until_expiry=days_left,
    )


# ============================================================================
# UI HELPERS (Streamlit)
# ============================================================================

def render_license_blocker(info: LicenseInfo) -> None:
    """
    Pantalla bloqueante cuando la licencia es inválida.
    Llamar al inicio de app_planta.py:

        info = verify_license()
        if not info.valid:
            render_license_blocker(info)
            st.stop()
    """
    import streamlit as st

    expected_path = get_license_path()

    st.error("Licencia no válida — Watermelon Planta bloqueado")
    st.markdown(f"### {info.error_reason}")

    st.markdown("---")

    st.markdown("#### ¿Dónde poner el archivo `license.token`?")
    st.code(str(expected_path), language="text")

    st.markdown("#### ¿No tienes tu licencia?")
    st.markdown(
        """
- **Cliente nuevo:** Contacta a SIGA GROUP para tu cotización y licencia.
- **Cliente existente:** Revisa tu email — la licencia fue enviada como
  archivo adjunto llamado `license.token`.
- **Soporte técnico:** ehernandez@sigasas.com
        """
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
