"""
core.clients
============

Registry multi-tenant de Watermelon System (Ciclo 20A).

Carga `data/clients.json` y expone resolvers para identificar
quién está consultando (admin, specialist o cliente externo) y
calcular el filtro de visibilidad sobre el archivo de reportes.

Tres roles:
  - admin       → ve TODO. Puede actuar en nombre de cualquier cliente.
  - specialist  → ve TODO (equipo SIGA Cat IV).
  - client      → ve solo reportes cuyo report_meta matchee con sus
                  match_strings (case-insensitive substring).

Resoluciones soportadas:
  - resolve_by_phone(phone)     → CallerScope (para el bot WhatsApp)
  - resolve_by_api_key(api_key) → CallerScope (para clientes API directos)
  - get_client_by_id(client_id) → Client | None (para X-Client-Filter)
  - filter_matches(report_meta, scope) → bool

Robustez:
  - Si data/clients.json no existe o es inválido, devolvemos un
    registry mínimo con admin único (env WATERMELON_API_KEYS).
  - Cache lru por proceso; reload_registry() limpia.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = PROJECT_ROOT / "data" / "clients.json"


# =============================================================
# Modelos
# =============================================================

@dataclass(frozen=True)
class CallerScope:
    """
    Resultado de resolver un phone o api_key contra el registry.
    """
    role: str = "unknown"          # admin | specialist | client | unknown
    name: str = ""
    email: str = ""
    client_id: str = ""             # solo cuando role="client"
    client_display: str = ""        # display_name del cliente
    match_strings: Tuple[str, ...] = ()
    phone: str = ""

    @property
    def is_authorized(self) -> bool:
        return self.role in ("admin", "specialist", "client")

    @property
    def sees_everything(self) -> bool:
        """admin y specialist ven todo el archivo."""
        return self.role in ("admin", "specialist")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "name": self.name,
            "email": self.email,
            "client_id": self.client_id,
            "client_display": self.client_display,
            "match_strings": list(self.match_strings),
        }


@dataclass(frozen=True)
class Client:
    id: str
    display_name: str
    match_strings: Tuple[str, ...] = ()
    asset_tags: Tuple[str, ...] = ()
    whatsapp_numbers: Tuple[str, ...] = ()
    api_key: str = ""
    owner_emails: Tuple[str, ...] = ()

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "display_name": self.display_name,
            "match_strings": list(self.match_strings),
            "asset_tags": list(self.asset_tags),
            "whatsapp_numbers": list(self.whatsapp_numbers),
            "owner_emails": list(self.owner_emails),
            # api_key NUNCA se expone en serializaciones públicas
        }


# =============================================================
# Loader
# =============================================================

def _normalize_phone(p: str) -> str:
    """Quita '+' y espacios. Aceptamos '573008888883' como canónico."""
    return (p or "").strip().lstrip("+").replace(" ", "").replace("-", "")


@lru_cache(maxsize=1)
def _load_registry() -> Dict[str, Any]:
    if not REGISTRY_PATH.exists():
        log.warning("clients.json no encontrado en %s — usando registry vacío", REGISTRY_PATH)
        return {"admins": [], "specialists": [], "clients": []}

    try:
        with REGISTRY_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        log.error("clients.json inválido (%s) — usando registry vacío", e)
        return {"admins": [], "specialists": [], "clients": []}

    if not isinstance(data, dict):
        return {"admins": [], "specialists": [], "clients": []}

    data.setdefault("admins", [])
    data.setdefault("specialists", [])
    data.setdefault("clients", [])
    return data


def reload_registry() -> None:
    """Limpia la cache. Útil cuando se edita clients.json en runtime."""
    _load_registry.cache_clear()


def save_registry(data: Dict[str, Any]) -> None:
    """
    Persiste el registry a disco. Usado por el Admin UI (Ciclo 20B).
    Escribe atómicamente (tmp file + rename) para evitar corrupciones
    si el proceso muere mid-write. Limpia la lru_cache después.

    Args:
        data: dict completo con keys 'admins', 'specialists', 'clients'.
              _meta se genera/preserva automáticamente.

    Raises:
        ValueError si la estructura es inválida.
    """
    if not isinstance(data, dict):
        raise ValueError("Registry data debe ser dict")

    # Validación básica
    for key in ("admins", "specialists", "clients"):
        val = data.get(key, [])
        if not isinstance(val, list):
            raise ValueError(f"'{key}' debe ser lista, recibido {type(val).__name__}")

    # Preservar _meta y bumpear last_updated
    from datetime import datetime
    raw = _load_registry()
    meta = dict(raw.get("_meta", {}))
    meta["last_updated"] = datetime.now().strftime("%Y-%m-%d")
    meta.setdefault("version", "1.0.0")

    payload = {
        "_meta": meta,
        "admins": list(data.get("admins", [])),
        "specialists": list(data.get("specialists", [])),
        "clients": list(data.get("clients", [])),
    }

    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = REGISTRY_PATH.with_suffix(".json.tmp")
    tmp_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    tmp_path.replace(REGISTRY_PATH)
    reload_registry()


# =============================================================
# API pública
# =============================================================

def list_admins() -> List[Dict[str, Any]]:
    return list(_load_registry().get("admins", []))


def list_specialists() -> List[Dict[str, Any]]:
    return list(_load_registry().get("specialists", []))


def list_clients() -> List[Client]:
    out: List[Client] = []
    for c in _load_registry().get("clients", []):
        if not isinstance(c, dict):
            continue
        cid = (c.get("id") or "").strip()
        if not cid:
            continue
        out.append(Client(
            id=cid,
            display_name=c.get("display_name", cid),
            match_strings=tuple(s.lower() for s in (c.get("match_strings") or [])),
            asset_tags=tuple(c.get("asset_tags") or []),
            whatsapp_numbers=tuple(_normalize_phone(p) for p in (c.get("whatsapp_numbers") or [])),
            api_key=(c.get("api_key") or "").strip(),
            owner_emails=tuple((e or "").lower().strip() for e in (c.get("owner_emails") or [])),
        ))
    return out


def get_client_by_id(client_id: str) -> Optional[Client]:
    target = (client_id or "").strip()
    if not target:
        return None
    for c in list_clients():
        if c.id == target:
            return c
    return None


def get_client_for_email(email: str) -> Optional[Client]:
    """Devuelve el Client cuyo owner_emails contiene el email dado.

    Returns None si el email no está asignado a ningún cliente — útil
    para detectar usuarios role=client que no tienen asset_tags
    asignados (caso fallback: ven nada o todo, el caller decide).
    """
    target = (email or "").strip().lower()
    if not target:
        return None
    for c in list_clients():
        if target in c.owner_emails:
            return c
    return None


def assign_client_to_email(client_id: str, email: str) -> bool:
    """Agrega `email` a owner_emails del cliente `client_id` en clients.json.

    Returns True si se hizo el cambio (escrito y registry refrescado).
    Si el email ya estaba presente o el client_id no existe, devuelve False.
    """
    target_id = (client_id or "").strip()
    target_email = (email or "").strip().lower()
    if not target_id or not target_email:
        return False

    if not REGISTRY_PATH.exists():
        return False
    try:
        with REGISTRY_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False

    clients = data.get("clients") or []
    touched = False
    for c in clients:
        if str(c.get("id", "")).strip() == target_id:
            existing = [str(e).strip().lower() for e in (c.get("owner_emails") or [])]
            if target_email in existing:
                return False  # ya estaba — no-op
            c["owner_emails"] = (c.get("owner_emails") or []) + [target_email]
            touched = True
            break

    if not touched:
        return False

    try:
        with REGISTRY_PATH.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
    except OSError:
        return False

    reload_registry()
    return True


def filter_instances_for_email(instances: List[Dict[str, Any]],
                                email: str) -> List[Dict[str, Any]]:
    """Filtra una lista de instances dict por los asset_tags del cliente
    al que pertenece `email`.

    - Si el email NO está en ningún owner_emails → devuelve [] (cliente
      sin assets asignados).
    - Si el cliente tiene asset_tags=[] → devuelve [] (no le asignaron
      activos todavía).
    - Si el email pertenece a un cliente con asset_tags configurados →
      filtra instances cuyo `tag` (case-insensitive) está en asset_tags.
    """
    c = get_client_for_email(email)
    if c is None or not c.asset_tags:
        return []
    allowed_tags_lc = {t.strip().lower() for t in c.asset_tags if t}
    out = []
    for inst in instances:
        tag = str(inst.get("tag", "") or inst.get("instance_id", "")).strip().lower()
        if tag in allowed_tags_lc:
            out.append(inst)
    return out


def resolve_by_phone(phone: str) -> CallerScope:
    """
    Resuelve un número WhatsApp contra el registry. Returns CallerScope
    con role unknown si no está en ninguna lista (caller debe rechazar).
    """
    p_norm = _normalize_phone(phone)
    if not p_norm:
        return CallerScope(phone=p_norm)

    for a in list_admins():
        for raw in (a.get("whatsapp_numbers") or []):
            if _normalize_phone(raw) == p_norm:
                return CallerScope(
                    role="admin",
                    name=a.get("name", ""),
                    email=(a.get("email") or "").lower(),
                    phone=p_norm,
                )

    for s in list_specialists():
        for raw in (s.get("whatsapp_numbers") or []):
            if _normalize_phone(raw) == p_norm:
                return CallerScope(
                    role="specialist",
                    name=s.get("name", ""),
                    email=(s.get("email") or "").lower(),
                    phone=p_norm,
                )

    for c in list_clients():
        if p_norm in c.whatsapp_numbers:
            return CallerScope(
                role="client",
                name=c.display_name,
                client_id=c.id,
                client_display=c.display_name,
                match_strings=c.match_strings,
                phone=p_norm,
            )

    return CallerScope(phone=p_norm)


def resolve_by_api_key(api_key: str, admin_keys: Optional[List[str]] = None) -> CallerScope:
    """
    Resuelve una API key:
      - Si está en admin_keys (env var WATERMELON_API_KEYS) → admin global
      - Si matchea c.api_key de algún cliente → client(c.id)
      - Otherwise → unknown
    """
    key = (api_key or "").strip()
    if not key:
        return CallerScope()

    # Admin keys vienen del env (lista). El bot usa una de éstas.
    if admin_keys is None:
        raw = os.environ.get("WATERMELON_API_KEYS", "").strip()
        admin_keys = [k.strip() for k in raw.split(",") if k.strip()]

    if key in admin_keys:
        return CallerScope(role="admin", name="api_admin", email="api@watermelon")

    for c in list_clients():
        if c.api_key and c.api_key == key:
            return CallerScope(
                role="client",
                name=c.display_name,
                client_id=c.id,
                client_display=c.display_name,
                match_strings=c.match_strings,
            )

    return CallerScope()


# =============================================================
# Filter
# =============================================================

def filter_matches(report_meta: Dict[str, Any], scope: CallerScope) -> bool:
    """
    True si el reporte es visible para este scope.
      - admin / specialist → siempre True
      - client → True si alguno de los match_strings aparece (case-insensitive)
                 en client / instance_tag / asset_class / train_description.
      - unknown → False
    """
    if scope.sees_everything:
        return True
    if scope.role != "client" or not scope.match_strings:
        return False

    rm = report_meta if isinstance(report_meta, dict) else {}
    text = " ".join([
        str(rm.get("client", "")),
        str(rm.get("instance_tag", "")),
        str(rm.get("asset_class", "")),
        str(rm.get("train_description", "")),
        str(rm.get("instance_id", "")),
        str(rm.get("asset_model", "")),
        str(rm.get("site", "")),
    ]).lower()

    return any(ms in text for ms in scope.match_strings if ms)


__all__ = [
    "CallerScope",
    "Client",
    "list_admins",
    "list_specialists",
    "list_clients",
    "get_client_by_id",
    "resolve_by_phone",
    "resolve_by_api_key",
    "filter_matches",
    "reload_registry",
    "save_registry",
]
