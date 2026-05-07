"""
api.auth
========

Validación de API keys para el servicio público.

Estrategia v1.0: API keys estáticas en variable de entorno
WATERMELON_API_KEYS (lista separada por comas) o en st.secrets.api_keys.
v1.1: rotación dinámica desde Supabase.

Uso:
    from api.auth import require_api_key

    # Desde un router FastAPI:
    @router.get("/v1/protected")
    def protected(api_key: str = Depends(require_api_key)):
        return {"ok": True, "key_id": api_key}
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
from typing import List, Optional


log = logging.getLogger(__name__)


def _load_configured_keys() -> List[str]:
    """
    Carga las API keys válidas. Prioridad:
      1. variable de entorno WATERMELON_API_KEYS  ("k1,k2,k3")
      2. st.secrets.api_keys                       (lista o string)
    """
    raw = os.environ.get("WATERMELON_API_KEYS", "").strip()
    keys: List[str] = []
    if raw:
        keys.extend(k.strip() for k in raw.split(",") if k.strip())

    # Soporte opcional de Streamlit secrets si la app corre dentro de
    # un proceso Streamlit. No forzamos import — fallback graceful.
    try:
        import streamlit as st  # type: ignore
        secret_block = st.secrets.get("api_keys", None)  # type: ignore
        if secret_block:
            if isinstance(secret_block, str):
                keys.append(secret_block)
            elif isinstance(secret_block, (list, tuple)):
                keys.extend(str(k) for k in secret_block)
    except Exception:
        # No streamlit en el proceso — está bien.
        pass

    return [k for k in keys if k]


def is_valid_api_key(provided: Optional[str]) -> bool:
    """
    Compara la key recibida contra la whitelist usando hmac.compare_digest
    para evitar timing attacks.
    """
    if not provided:
        return False
    keys = _load_configured_keys()
    if not keys:
        log.warning("API key check ejecutándose sin keys configuradas. "
                    "Configurar WATERMELON_API_KEYS antes de exponer públicamente.")
        return False
    for valid in keys:
        if hmac.compare_digest(provided, valid):
            return True
    return False


def hash_for_log(api_key: str) -> str:
    """
    SHA-256 truncado para loggear referencias a una key SIN imprimirla
    en plano. Útil en audit logs y métricas por cliente.
    """
    if not api_key:
        return ""
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:12]


# Helper para FastAPI; no se ejecuta si FastAPI no está instalado.
def require_api_key(*args, **kwargs):
    """
    Dependency placeholder. La función real se construye dinámicamente
    en api/app.py una vez que tenemos FastAPI/Header importables. Esto
    permite que api.services y api.auth sean importables sin FastAPI.
    """
    raise NotImplementedError(
        "require_api_key debe usarse desde api.app vía FastAPI Depends. "
        "Para validar manualmente usar is_valid_api_key()."
    )


__all__ = ["is_valid_api_key", "hash_for_log", "require_api_key"]
