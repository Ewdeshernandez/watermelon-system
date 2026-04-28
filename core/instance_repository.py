"""
core.instance_repository
========================

Capa de abstracción de storage para el modelo de Asset Instances.
Permite que el sistema persista los datos en dos backends distintos:

  - LocalFilesystemRepository:
      Lo que veníamos usando hasta el Ciclo 8. Guarda metadata.json
      en data/instances/{instance_id}/ y los archivos en
      data/instances/{instance_id}/documents/. Funciona sin internet,
      ideal para desarrollo local. Pero en Streamlit Cloud el
      filesystem es efímero — los datos se pierden en cada redeploy.

  - SupabaseRepository:
      Persistencia real para producción. Guarda metadata como JSONB
      en una tabla 'instances' de Supabase Postgres, y los archivos
      binarios (PDFs, imágenes) en Supabase Storage bucket
      'instance-documents'. Sobrevive cualquier redeploy / reboot.
      Free tier de Supabase alcanza para varios años de operación
      (500 MB DB + 1 GB storage gratis para siempre).

Selección automática:
  Si st.secrets contiene una sección [supabase] con url + service_key,
  el sistema usa SupabaseRepository. Si no, cae a Local. Esto permite
  desarrollar localmente sin configurar nada y al desplegar en
  Streamlit Cloud (donde se configuran los secrets en la UI) se
  active automáticamente la persistencia.

Public API:
  get_active_repository() -> InstanceRepository
      Devuelve la instancia singleton del repositorio activo,
      seleccionado en base a configuración. Las funciones de
      core/instance_state.py la usan internamente.

  Ambos backends implementan el mismo protocolo de métodos:

      list_instances() -> List[Dict[str, Any]]      # headers livianos
      load_instance(instance_id) -> Optional[Dict]  # metadata completa
      save_instance(instance_dict) -> None
      delete_instance(instance_id) -> bool

      upload_document_bytes(instance_id, storage_filename, data) -> bool
      download_document_bytes(instance_id, storage_filename) -> Optional[bytes]
      delete_document_file(instance_id, storage_filename) -> bool
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
INSTANCES_DIR = DATA_DIR / "instances"


class InstanceRepository(Protocol):
    """Protocolo (duck typing) para backends de storage de instancias."""

    def list_instances(self) -> List[Dict[str, Any]]: ...
    def load_instance(self, instance_id: str) -> Optional[Dict[str, Any]]: ...
    def save_instance(self, instance_dict: Dict[str, Any]) -> None: ...
    def delete_instance(self, instance_id: str) -> bool: ...

    def upload_document_bytes(
        self, instance_id: str, storage_filename: str, data: bytes
    ) -> bool: ...
    def download_document_bytes(
        self, instance_id: str, storage_filename: str
    ) -> Optional[bytes]: ...
    def delete_document_file(
        self, instance_id: str, storage_filename: str
    ) -> bool: ...

    @property
    def backend_name(self) -> str: ...


# =============================================================
# LOCAL FILESYSTEM BACKEND
# =============================================================

class LocalFilesystemRepository:
    """
    Backend que persiste todo en data/instances/. Es el comportamiento
    histórico (Ciclo 8). Sirve para desarrollo local. En Streamlit Cloud
    es efímero — los datos se pierden en redeploys del container.
    """

    backend_name = "local_filesystem"

    def _instance_dir(self, instance_id: str) -> Path:
        return INSTANCES_DIR / instance_id

    def _metadata_path(self, instance_id: str) -> Path:
        return self._instance_dir(instance_id) / "metadata.json"

    def _documents_dir(self, instance_id: str) -> Path:
        return self._instance_dir(instance_id) / "documents"

    def _ensure_dirs(self, instance_id: str) -> None:
        INSTANCES_DIR.mkdir(parents=True, exist_ok=True)
        self._instance_dir(instance_id).mkdir(parents=True, exist_ok=True)
        self._documents_dir(instance_id).mkdir(parents=True, exist_ok=True)

    def list_instances(self) -> List[Dict[str, Any]]:
        if not INSTANCES_DIR.exists():
            return []
        entries: List[Dict[str, Any]] = []
        for child in INSTANCES_DIR.iterdir():
            if not child.is_dir():
                continue
            meta = child / "metadata.json"
            if not meta.exists():
                continue
            try:
                with open(meta, "r", encoding="utf-8") as f:
                    data = json.load(f)
                entries.append({
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
                })
            except Exception:
                continue
        entries.sort(key=lambda e: e.get("updated_at", ""), reverse=True)
        return entries

    def load_instance(self, instance_id: str) -> Optional[Dict[str, Any]]:
        path = self._metadata_path(instance_id)
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def save_instance(self, instance_dict: Dict[str, Any]) -> None:
        instance_id = instance_dict["instance_id"]
        self._ensure_dirs(instance_id)
        instance_dict["updated_at"] = datetime.now().isoformat(timespec="seconds")
        if not instance_dict.get("created_at"):
            instance_dict["created_at"] = instance_dict["updated_at"]
        with open(self._metadata_path(instance_id), "w", encoding="utf-8") as f:
            json.dump(instance_dict, f, indent=2, ensure_ascii=False, default=str)

    def delete_instance(self, instance_id: str) -> bool:
        d = self._instance_dir(instance_id)
        if not d.exists():
            return False
        try:
            shutil.rmtree(d)
            return True
        except Exception:
            return False

    def upload_document_bytes(
        self, instance_id: str, storage_filename: str, data: bytes
    ) -> bool:
        self._ensure_dirs(instance_id)
        try:
            with open(self._documents_dir(instance_id) / storage_filename, "wb") as f:
                f.write(data)
            return True
        except Exception:
            return False

    def download_document_bytes(
        self, instance_id: str, storage_filename: str
    ) -> Optional[bytes]:
        path = self._documents_dir(instance_id) / storage_filename
        if not path.exists():
            return None
        try:
            with open(path, "rb") as f:
                return f.read()
        except Exception:
            return None

    def delete_document_file(
        self, instance_id: str, storage_filename: str
    ) -> bool:
        path = self._documents_dir(instance_id) / storage_filename
        if not path.exists():
            return False
        try:
            path.unlink()
            return True
        except Exception:
            return False


# =============================================================
# SUPABASE BACKEND
# =============================================================

class SupabaseRepository:
    """
    Backend que persiste contra Supabase Postgres + Storage. Sobrevive
    cualquier redeploy de Streamlit Cloud. Requiere:

      1. Cuenta Supabase + project creado (free tier alcanza)
      2. Tabla 'instances' con schema:
           id text PRIMARY KEY,
           metadata jsonb NOT NULL,
           updated_at timestamptz DEFAULT now()
      3. Bucket de Storage llamado 'instance-documents'
      4. Credenciales en st.secrets["supabase"]:
           url = "https://xxx.supabase.co"
           service_key = "ey..."
           bucket = "instance-documents"  (opcional, default este nombre)

    Si la tabla o el bucket no existen, los métodos fallan con
    excepciones claras.
    """

    backend_name = "supabase"

    def __init__(self, url: str, service_key: str, bucket: str = "instance-documents"):
        # Lazy import: solo si efectivamente se usa el backend
        try:
            from supabase import create_client
        except ImportError as e:
            raise ImportError(
                "El paquete 'supabase' no está instalado. "
                "Agregalo a requirements.txt: supabase>=2.0.0"
            ) from e
        self._client = create_client(url, service_key)
        self._bucket = bucket
        self._table = "instances"

    def list_instances(self) -> List[Dict[str, Any]]:
        try:
            res = self._client.table(self._table).select("id, metadata, updated_at").execute()
        except Exception:
            return []
        entries: List[Dict[str, Any]] = []
        for row in (res.data or []):
            md = row.get("metadata") or {}
            entries.append({
                "instance_id": md.get("instance_id", row.get("id", "")),
                "profile_key": md.get("profile_key", ""),
                "tag": md.get("tag", ""),
                "serial_number": md.get("serial_number", ""),
                "location": md.get("location", ""),
                "notes": md.get("notes", ""),
                "created_at": md.get("created_at", ""),
                "updated_at": md.get("updated_at", row.get("updated_at", "")),
                "n_documents": len(md.get("documents", [])),
                "n_parameters": len(md.get("captured_parameters", {})),
            })
        entries.sort(key=lambda e: e.get("updated_at", ""), reverse=True)
        return entries

    def load_instance(self, instance_id: str) -> Optional[Dict[str, Any]]:
        try:
            res = (
                self._client.table(self._table)
                .select("metadata")
                .eq("id", instance_id)
                .single()
                .execute()
            )
        except Exception:
            return None
        if not res.data:
            return None
        return res.data.get("metadata") or None

    def save_instance(self, instance_dict: Dict[str, Any]) -> None:
        instance_id = instance_dict["instance_id"]
        instance_dict["updated_at"] = datetime.now().isoformat(timespec="seconds")
        if not instance_dict.get("created_at"):
            instance_dict["created_at"] = instance_dict["updated_at"]
        payload = {"id": instance_id, "metadata": instance_dict}
        # upsert: insert si no existe, update si existe
        self._client.table(self._table).upsert(payload).execute()

    def delete_instance(self, instance_id: str) -> bool:
        try:
            # Borrar primero los documentos del bucket
            inst = self.load_instance(instance_id)
            if inst:
                for doc in inst.get("documents", []):
                    storage = doc.get("storage_filename")
                    if storage:
                        self.delete_document_file(instance_id, storage)
            self._client.table(self._table).delete().eq("id", instance_id).execute()
            return True
        except Exception:
            return False

    def _doc_path(self, instance_id: str, storage_filename: str) -> str:
        return f"{instance_id}/{storage_filename}"

    def upload_document_bytes(
        self, instance_id: str, storage_filename: str, data: bytes
    ) -> bool:
        path = self._doc_path(instance_id, storage_filename)
        try:
            self._client.storage.from_(self._bucket).upload(
                path, data, {"upsert": "true"}
            )
            return True
        except Exception:
            return False

    def download_document_bytes(
        self, instance_id: str, storage_filename: str
    ) -> Optional[bytes]:
        path = self._doc_path(instance_id, storage_filename)
        try:
            return self._client.storage.from_(self._bucket).download(path)
        except Exception:
            return None

    def delete_document_file(
        self, instance_id: str, storage_filename: str
    ) -> bool:
        path = self._doc_path(instance_id, storage_filename)
        try:
            self._client.storage.from_(self._bucket).remove([path])
            return True
        except Exception:
            return False


# =============================================================
# FACTORY: selección automática del backend
# =============================================================

_REPOSITORY_CACHE: Optional[InstanceRepository] = None


def _build_repository() -> InstanceRepository:
    """
    Inspecciona st.secrets para decidir qué backend usar:

      Si existe st.secrets['supabase']['url'] y
                st.secrets['supabase']['service_key'] → Supabase
      Si no                                          → Local filesystem

    Falla limpio: si Supabase está configurado pero la conexión inicial
    falla (paquete no instalado, credenciales inválidas), cae a Local
    con un warning visible en la sidebar.
    """
    try:
        import streamlit as st
        # st.secrets puede no existir si secrets.toml no está
        if "supabase" in st.secrets:
            cfg = st.secrets["supabase"]
            url = cfg.get("url", "").strip()
            key = cfg.get("service_key", "").strip()
            bucket = cfg.get("bucket", "instance-documents").strip()
            if url and key:
                try:
                    return SupabaseRepository(url=url, service_key=key, bucket=bucket)
                except Exception as e:
                    # No pudo conectarse — log y caer a local
                    try:
                        st.sidebar.warning(
                            f"Supabase configurado pero no accesible "
                            f"({type(e).__name__}). Usando filesystem local."
                        )
                    except Exception:
                        pass
    except Exception:
        # Streamlit no inicializado o secrets no disponibles
        pass

    return LocalFilesystemRepository()


def get_active_repository() -> InstanceRepository:
    """
    Devuelve el backend activo (singleton). Se cachea en el primer
    llamado dentro del proceso. Para forzar re-evaluación (ej. en
    tests), llamar reset_repository_cache().
    """
    global _REPOSITORY_CACHE
    if _REPOSITORY_CACHE is None:
        _REPOSITORY_CACHE = _build_repository()
    return _REPOSITORY_CACHE


def reset_repository_cache() -> None:
    """Invalida el cache (útil para tests o cambios de config)."""
    global _REPOSITORY_CACHE
    _REPOSITORY_CACHE = None


__all__ = [
    "InstanceRepository",
    "LocalFilesystemRepository",
    "SupabaseRepository",
    "get_active_repository",
    "reset_repository_cache",
]
