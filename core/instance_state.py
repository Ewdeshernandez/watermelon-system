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

Storage en disco:

    data/
      instances/
        {instance_id}/
          metadata.json       ← info del activo + parámetros + índice docs
          documents/
            {doc_id}__{filename}

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

import json
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
INSTANCES_DIR = DATA_DIR / "instances"


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


# =============================================================
# HELPERS DE FILESYSTEM
# =============================================================

def _slugify(name: str) -> str:
    """Convierte un nombre arbitrario a un slug seguro para filesystem."""
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9_-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:80] if s else "unnamed"


def _instance_dir(instance_id: str) -> Path:
    return INSTANCES_DIR / instance_id


def _metadata_path(instance_id: str) -> Path:
    return _instance_dir(instance_id) / "metadata.json"


def _documents_dir(instance_id: str) -> Path:
    return _instance_dir(instance_id) / "documents"


def _ensure_dirs(instance_id: str) -> None:
    INSTANCES_DIR.mkdir(parents=True, exist_ok=True)
    _instance_dir(instance_id).mkdir(parents=True, exist_ok=True)
    _documents_dir(instance_id).mkdir(parents=True, exist_ok=True)


# =============================================================
# CRUD DE INSTANCIAS
# =============================================================

def list_instances() -> List[Dict[str, Any]]:
    """
    Devuelve resumen de todas las instances registradas, ordenadas por
    fecha de actualización descendente. Cada entry tiene los campos
    del header (instance_id, profile_key, tag, serial_number, location,
    notes, created_at, updated_at) — sin parámetros ni documentos para
    que sea liviano.
    """
    if not INSTANCES_DIR.exists():
        return []
    entries: List[Dict[str, Any]] = []
    for child in INSTANCES_DIR.iterdir():
        if not child.is_dir():
            continue
        meta_path = child / "metadata.json"
        if not meta_path.exists():
            continue
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            header = {
                "instance_id": data.get("instance_id", child.name),
                "profile_key": data.get("profile_key", ""),
                "tag": data.get("tag", ""),
                "serial_number": data.get("serial_number", ""),
                "location": data.get("location", ""),
                "notes": data.get("notes", ""),
                "created_at": data.get("created_at", ""),
                "updated_at": data.get("updated_at", ""),
                "n_documents": len(data.get("documents", [])),
                "n_parameters": len(data.get("captured_parameters", {})),
            }
            entries.append(header)
        except Exception:
            continue
    entries.sort(key=lambda e: e.get("updated_at", ""), reverse=True)
    return entries


def get_instance(instance_id: str) -> Optional[Instance]:
    """Devuelve la Instance completa o None si no existe."""
    path = _metadata_path(instance_id)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return Instance(
            instance_id=data.get("instance_id", instance_id),
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
    except Exception:
        return None


def _save_instance(inst: Instance) -> None:
    _ensure_dirs(inst.instance_id)
    inst.updated_at = datetime.now().isoformat(timespec="seconds")
    if not inst.created_at:
        inst.created_at = inst.updated_at
    payload = asdict(inst)
    with open(_metadata_path(inst.instance_id), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)


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
    Crea una nueva Instance en disco. Si seed_from_profile=True y existe
    una semilla en core/vault_seeds.py para el profile_key, los
    parámetros del seed se inyectan como defaults en captured_parameters.

    Args:
        instance_id: slug único (se sanitiza). Si ya existe, sobrescribe.
        profile_key: referencia al profile (debe existir en MACHINE_PROFILES).
        tag: tag interno del cliente (ej. "TES1").
        serial_number, location, notes: metadata libre.
        seed_from_profile: si True, pre-puebla parámetros desde
            core/vault_seeds.get_seed_parameters(profile_key).

    Returns:
        Instance recién creada y persistida.
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
    """
    Borra completamente una Instance del disco (metadata + todos sus
    documentos). Operación destructiva, sin papelera.
    """
    inst_dir = _instance_dir(instance_id)
    if not inst_dir.exists():
        return False
    try:
        import shutil
        shutil.rmtree(inst_dir)
        return True
    except Exception:
        return False


# =============================================================
# PARÁMETROS CAPTURADOS PER-INSTANCIA
# =============================================================

def get_instance_parameters(instance_id: str) -> Dict[str, Any]:
    """
    Devuelve los parámetros capturados de una instancia específica.
    Si la instance no existe, devuelve {} (no levanta).
    """
    inst = get_instance(instance_id)
    if inst is None:
        return {}
    return dict(inst.captured_parameters)


def update_instance_parameters_bulk(
    instance_id: str,
    values: Dict[str, Any],
) -> bool:
    """
    Actualiza múltiples parámetros de una instancia. Valores None o ""
    eliminan el parámetro existente.
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
    Acepta un UploadedFile de Streamlit (o cualquier objeto file-like
    con .name y .read()), lo persiste a disco bajo
    data/instances/{instance_id}/documents/ y lo indexa en metadata.json.
    Devuelve el doc_id generado, o None si la instance no existe o
    el upload falló.
    """
    inst = get_instance(instance_id)
    if inst is None:
        return None
    _ensure_dirs(instance_id)

    # Generar storage_filename con prefijo UUID para evitar colisiones
    import uuid
    original = getattr(file_obj, "name", "document")
    safe_name = _slugify(original.rsplit(".", 1)[0])
    ext = original.rsplit(".", 1)[-1] if "." in original else "bin"
    storage_id = uuid.uuid4().hex[:12]
    storage_filename = f"{storage_id}__{safe_name}.{ext}"
    target = _documents_dir(instance_id) / storage_filename

    try:
        # Streamlit UploadedFile: .read() devuelve bytes
        file_obj.seek(0) if hasattr(file_obj, "seek") else None
        data = file_obj.read() if hasattr(file_obj, "read") else file_obj
        if isinstance(data, str):
            data = data.encode("utf-8")
        with open(target, "wb") as f:
            f.write(data)
        size = target.stat().st_size
    except Exception:
        return None

    return add_instance_document(
        instance_id,
        storage_filename=storage_filename,
        original_filename=original,
        title=title,
        document_type=document_type,
        description=description,
        tags=tags,
        size_bytes=size,
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
    Agrega un documento al índice de una instancia. El archivo binario
    debe haberse copiado previamente a _documents_dir(instance_id) /
    storage_filename. Devuelve el doc_id generado, o None si la
    instancia no existe.
    """
    import uuid
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
    """Quita un documento del índice y borra el archivo del disco."""
    inst = get_instance(instance_id)
    if inst is None:
        return False
    target = next((d for d in inst.documents if d.get("id") == doc_id), None)
    if target is None:
        return False
    storage = target.get("storage_filename", "")
    if storage:
        path = _documents_dir(instance_id) / storage
        try:
            if path.exists():
                path.unlink()
        except Exception:
            pass
    inst.documents = [d for d in inst.documents if d.get("id") != doc_id]
    _save_instance(inst)
    return True


def get_instance_document_path(instance_id: str, doc_id: str) -> Optional[Path]:
    """Devuelve el path en disco del archivo de un documento, o None."""
    inst = get_instance(instance_id)
    if inst is None:
        return None
    target = next((d for d in inst.documents if d.get("id") == doc_id), None)
    if target is None:
        return None
    storage = target.get("storage_filename", "")
    if not storage:
        return None
    path = _documents_dir(instance_id) / storage
    return path if path.exists() else None


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
]
