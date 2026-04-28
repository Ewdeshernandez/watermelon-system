"""
core.instance_state
===================

Asset Instance: representa una **máquina física específica** monitoreada
por el sistema (ej. el turbogenerador Brush 54 MW de la unidad TES1
ubicado en la planta del Atlántico). Es el nivel correcto de granularidad
para guardar parámetros, documentos y mediciones, porque dos máquinas
del mismo TIPO/profile (ej. dos Brush 54 MW idénticos) son físicamente
distintas y no deben compartir su data.

Diferencia conceptual entre Profile e Instance:

  Profile (familia)                    Instance (unidad física)
  ───────────────────                  ─────────────────────────
  Brush turbogenerator 54 MW           TES1 (Atlántico)
  Brush turbogenerator 54 MW           TES2 (Casanare)
  Siemens SGT-300 (gas)                Unidad gas A
  Siemens SGT-300 (gas)                Unidad gas B

  Define: ISO part aplicable,         Define: Ubicación, serial, tag,
  módulos disponibles, defaults       parámetros físicos medidos del
  por familia.                         cojinete real, manuales OEM
                                       específicos de esta unidad,
                                       histórico de mantenimiento.

Storage:
  Las funciones de este módulo NO tocan filesystem ni Supabase
  directamente — delegan en core/instance_repository.get_active_repository(),
  que selecciona automáticamente el backend (Local en dev, Supabase en
  producción cuando hay credenciales en st.secrets).

  Backend Local:    data/instances/{instance_id}/metadata.json
                    data/instances/{instance_id}/documents/{file}
  Backend Supabase: tabla 'instances' + bucket 'instance-documents'

Cada instance tiene un ID único (slug) elegido por el usuario al
crearla (ej. "brush_tes1", "siemens_sgt300_planta_b"). El profile_key
queda como referencia dentro de metadata.json para que cada módulo
sepa qué normas/ISO part/módulos aplicar.

Backwards compatibility con Ciclo 7 (vault_seeds + profile_key):
- Si una instancia se crea desde un profile que tiene seed en
  core/vault_seeds.py, los parámetros del seed se inyectan como
  defaults iniciales (el usuario puede sobrescribirlos).
- Si el sistema arranca sin instancias creadas, ofrece auto-crear una
  por cada profile con seed disponible (con instance_id sugerido).
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.instance_repository import get_active_repository, INSTANCES_DIR


# =============================================================
# DATA MODEL
# =============================================================

@dataclass
class Instance:
    """Representa una máquina física registrada en el sistema."""
    instance_id: str          # slug único, ej. "brush_tes1"
    profile_key: str           # referencia al profile (familia/tipo)
    tag: str = ""              # tag interno del cliente, ej. "TES1"
    serial_number: str = ""    # número de serie OEM
    location: str = ""         # ubicación física (planta, ciudad)
    notes: str = ""            # notas libres del operador
    captured_parameters: Dict[str, Any] = field(default_factory=dict)
    documents: List[Dict[str, Any]] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Instance":
        """Construye desde un dict serializado (de filesystem o Supabase)."""
        return cls(
            instance_id=data.get("instance_id", ""),
            profile_key=data.get("profile_key", ""),
            tag=data.get("tag", ""),
            serial_number=data.get("serial_number", ""),
            location=data.get("location", ""),
            notes=data.get("notes", ""),
            captured_parameters=dict(data.get("captured_parameters", {})),
            documents=list(data.get("documents", [])),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
        )


# =============================================================
# UTILIDADES
# =============================================================

def _slugify(name: str) -> str:
    """Convierte un nombre arbitrario a un slug seguro para filesystem/URLs."""
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9_-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:80] if s else "unnamed"


# =============================================================
# CRUD DE INSTANCIAS
# =============================================================

def list_instances() -> List[Dict[str, Any]]:
    """
    Devuelve resumen liviano de todas las instances registradas,
    ordenadas por fecha de actualización descendente.
    """
    return get_active_repository().list_instances()


def get_instance(instance_id: str) -> Optional[Instance]:
    """Devuelve la Instance completa o None si no existe."""
    data = get_active_repository().load_instance(instance_id)
    if data is None:
        return None
    return Instance.from_dict(data)


def _save_instance(inst: Instance) -> None:
    """Persiste una Instance al backend activo."""
    get_active_repository().save_instance(asdict(inst))


def create_instance(
    *,
    instance_id: str,
    profile_key: str,
    tag: str = "",
    serial_number: str = "",
    location: str = "",
    notes: str = "",
    seed_from_profile: bool = True,
) -> Instance:
    """
    Crea una nueva Instance. Si seed_from_profile=True y existe semilla
    en core/vault_seeds.py para el profile_key, los parámetros del seed
    se inyectan como defaults iniciales en captured_parameters.
    """
    inst_id = _slugify(instance_id)
    seeded_params: Dict[str, Any] = {}
    if seed_from_profile:
        try:
            from core.vault_seeds import get_seed_parameters
            seeded_params = get_seed_parameters(profile_key)
        except Exception:
            seeded_params = {}

    inst = Instance(
        instance_id=inst_id,
        profile_key=profile_key,
        tag=tag,
        serial_number=serial_number,
        location=location,
        notes=notes,
        captured_parameters=dict(seeded_params),
        documents=[],
    )
    _save_instance(inst)
    return inst


def update_instance_header(
    instance_id: str,
    *,
    tag: Optional[str] = None,
    serial_number: Optional[str] = None,
    location: Optional[str] = None,
    notes: Optional[str] = None,
    profile_key: Optional[str] = None,
) -> bool:
    """Actualiza solo los campos de header (sin tocar parámetros ni docs)."""
    inst = get_instance(instance_id)
    if inst is None:
        return False
    if tag is not None: inst.tag = tag
    if serial_number is not None: inst.serial_number = serial_number
    if location is not None: inst.location = location
    if notes is not None: inst.notes = notes
    if profile_key is not None: inst.profile_key = profile_key
    _save_instance(inst)
    return True


def delete_instance(instance_id: str) -> bool:
    """Borra completamente una Instance del backend (metadata + documentos)."""
    return get_active_repository().delete_instance(instance_id)


# =============================================================
# PARÁMETROS CAPTURADOS PER-INSTANCIA
# =============================================================

def get_instance_parameters(instance_id: str) -> Dict[str, Any]:
    """Devuelve los parámetros capturados de una instancia, o {} si no existe."""
    inst = get_instance(instance_id)
    return dict(inst.captured_parameters) if inst else {}


def update_instance_parameters_bulk(
    instance_id: str,
    values: Dict[str, Any],
) -> bool:
    """
    Actualiza múltiples parámetros. Valores None o "" eliminan el
    parámetro existente.
    """
    inst = get_instance(instance_id)
    if inst is None:
        return False
    for k, v in values.items():
        if v is None or v == "":
            inst.captured_parameters.pop(k, None)
        else:
            inst.captured_parameters[k] = v
    _save_instance(inst)
    return True


def update_instance_parameter(
    instance_id: str,
    key: str,
    value: Any,
) -> bool:
    """Actualiza un parámetro individual."""
    return update_instance_parameters_bulk(instance_id, {key: value})


# =============================================================
# DOCUMENTOS PER-INSTANCIA
# =============================================================

def add_uploaded_file_to_instance(
    instance_id: str,
    file_obj: Any,
    *,
    title: str = "",
    document_type: str = "other",
    description: str = "",
    tags: Optional[List[str]] = None,
) -> Optional[str]:
    """
    Acepta un UploadedFile de Streamlit, lo persiste vía backend activo
    (Local o Supabase Storage) y lo indexa en metadata.
    """
    inst = get_instance(instance_id)
    if inst is None:
        return None

    original = getattr(file_obj, "name", "document")
    safe_name = _slugify(original.rsplit(".", 1)[0])
    ext = original.rsplit(".", 1)[-1] if "." in original else "bin"
    storage_id = uuid.uuid4().hex[:12]
    storage_filename = f"{storage_id}__{safe_name}.{ext}"

    # Leer bytes del file_obj
    try:
        if hasattr(file_obj, "seek"):
            file_obj.seek(0)
        data = file_obj.read() if hasattr(file_obj, "read") else file_obj
        if isinstance(data, str):
            data = data.encode("utf-8")
    except Exception:
        return None

    repo = get_active_repository()
    if not repo.upload_document_bytes(instance_id, storage_filename, data):
        return None

    return add_instance_document(
        instance_id,
        storage_filename=storage_filename,
        original_filename=original,
        title=title,
        document_type=document_type,
        description=description,
        tags=tags,
        size_bytes=len(data),
    )


def add_instance_document(
    instance_id: str,
    *,
    storage_filename: str,
    original_filename: str,
    title: str = "",
    document_type: str = "other",
    description: str = "",
    tags: Optional[List[str]] = None,
    size_bytes: int = 0,
) -> Optional[str]:
    """
    Indexa un documento ya persistido (vía repo.upload_document_bytes)
    en la metadata de la instancia. Devuelve el doc_id generado.
    """
    inst = get_instance(instance_id)
    if inst is None:
        return None
    doc_id = uuid.uuid4().hex[:12]
    record = {
        "id": doc_id,
        "storage_filename": storage_filename,
        "filename": original_filename,
        "title": title or original_filename,
        "document_type": document_type,
        "description": description,
        "tags": list(tags or []),
        "size_bytes": int(size_bytes),
        "uploaded_at": datetime.now().isoformat(timespec="seconds"),
    }
    inst.documents.append(record)
    _save_instance(inst)
    return doc_id


def remove_instance_document(instance_id: str, doc_id: str) -> bool:
    """Quita el documento del índice y borra el archivo del backend."""
    inst = get_instance(instance_id)
    if inst is None:
        return False
    target = next((d for d in inst.documents if d.get("id") == doc_id), None)
    if target is None:
        return False
    storage = target.get("storage_filename", "")
    if storage:
        get_active_repository().delete_document_file(instance_id, storage)
    inst.documents = [d for d in inst.documents if d.get("id") != doc_id]
    _save_instance(inst)
    return True


def get_instance_document_path(instance_id: str, doc_id: str) -> Optional[Path]:
    """
    Devuelve un Path local al archivo del documento. Para el backend
    Local, es el path real en disco. Para Supabase, descarga el archivo
    a un tempfile y devuelve ese path (caché de sesión).

    Si el módulo solo necesita los bytes, preferí get_instance_document_bytes
    que evita el round-trip por filesystem.
    """
    inst = get_instance(instance_id)
    if inst is None:
        return None
    target = next((d for d in inst.documents if d.get("id") == doc_id), None)
    if target is None:
        return None
    storage = target.get("storage_filename", "")
    if not storage:
        return None

    repo = get_active_repository()
    # Si es local, podemos devolver el path directo
    if repo.backend_name == "local_filesystem":
        path = INSTANCES_DIR / instance_id / "documents" / storage
        return path if path.exists() else None

    # Si es Supabase, descargamos a tempfile
    data = repo.download_document_bytes(instance_id, storage)
    if data is None:
        return None
    import tempfile
    tmp = Path(tempfile.gettempdir()) / f"wm_{instance_id}_{doc_id}_{storage}"
    try:
        with open(tmp, "wb") as f:
            f.write(data)
        return tmp
    except Exception:
        return None


def get_instance_document_bytes(instance_id: str, doc_id: str) -> Optional[bytes]:
    """
    Devuelve los bytes del archivo del documento, leyéndolos del backend
    activo. Más eficiente que get_instance_document_path para servir
    descargas (evita el round-trip por tempfile en backend Supabase).
    """
    inst = get_instance(instance_id)
    if inst is None:
        return None
    target = next((d for d in inst.documents if d.get("id") == doc_id), None)
    if target is None:
        return None
    storage = target.get("storage_filename", "")
    if not storage:
        return None
    return get_active_repository().download_document_bytes(instance_id, storage)


# =============================================================
# DIAGNÓSTICOS DE BACKEND (para mostrar en sidebar)
# =============================================================

def get_active_backend_name() -> str:
    """Devuelve el nombre del backend activo ('local_filesystem' o 'supabase')."""
    return get_active_repository().backend_name


__all__ = [
    "Instance",
    "INSTANCES_DIR",
    "list_instances",
    "get_instance",
    "create_instance",
    "update_instance_header",
    "delete_instance",
    "get_instance_parameters",
    "update_instance_parameters_bulk",
    "update_instance_parameter",
    "add_instance_document",
    "add_uploaded_file_to_instance",
    "remove_instance_document",
    "get_instance_document_path",
    "get_instance_document_bytes",
    "get_active_backend_name",
]
