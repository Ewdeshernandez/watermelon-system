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
    """
    Representa una máquina física registrada en el sistema.

    Ciclo 14a — header extendido con identificación de tren acoplado
    (driver/driven), rango operativo, soportes, sondas, setpoints y
    metadata de mantenimiento. Todos los campos nuevos tienen default
    vacío → instancias creadas en ciclos previos siguen siendo válidas.
    """
    # Identificación core (Ciclo 8)
    instance_id: str          # slug único, ej. "brush_tes1"
    profile_key: str           # referencia al profile (familia/tipo)
    tag: str = ""              # tag corto operativo, ej. "TES1"
    serial_number: str = ""    # legacy — alias de driven_serial (back-compat)
    location: str = ""         # ubicación física (legacy single-string)
    notes: str = ""            # notas libres del operador

    # Ciclo 14a — Identificación extendida del activo
    client: str = ""           # ej. "ECOPETROL - MAGNEX"
    site: str = ""             # ej. "TERMOSURIA - VILLAVICENCIO"
    asset_class: str = ""      # ej. "TURBOGENERADOR"

    # Ciclo 14a — Tren acoplado (driver = motriz, driven = accionada)
    driver_manufacturer: str = ""   # ej. "GE"
    driver_model: str = ""          # ej. "LM6000"
    driver_serial: str = ""         # oculto del reporte; útil para garantías
    driven_manufacturer: str = ""   # ej. "Brush"
    driven_model: str = ""          # ej. "Generador 54 MW"
    driven_serial: str = ""         # oculto del reporte
    nominal_power_mw: float = 0.0   # ej. 54

    # Ciclo 14a — Operación y rango de velocidad
    nominal_rpm: float = 0.0
    min_rpm: float = 0.0            # rango operativo mínimo
    max_rpm: float = 0.0            # rango operativo máximo
    trip_rpm: float = 0.0           # velocidad de disparo (overspeed trip)
    iso_group: str = ""             # rigid | flexible | etc.

    # Ciclo 14a — Soportes (clave para diagnóstico)
    support_type: str = ""          # fluid_film | rolling_element | magnetic | mixed
    support_count: int = 0          # cantidad total de soportes en el tren
    support_detail: str = ""        # texto libre descriptivo

    # Ciclo 14a — Sondas de proximidad (orientación física)
    probe_x_orientation_deg: float = 0.0  # típico 45 (XL) o 0 (vertical)
    probe_y_orientation_deg: float = 0.0  # típico -45 (YR) o 90 (horizontal)

    # Ciclo 14a — Setpoints (si están definidos, tienen prioridad sobre ISO genérico)
    alert_level: float = 0.0
    danger_level: float = 0.0
    trip_level: float = 0.0
    setpoint_unit: str = ""         # ej. "mil pp" / "mm/s rms"

    # Ciclo 14a — Acople
    coupling_class: str = ""        # rigid | flexible | fluid

    # Ciclo 14a — Esquemático visual (PNG HD)
    schematic_png: str = ""         # storage_filename dentro del Document Vault de la instancia
                                    # (se resuelve via get_instance_document_bytes)

    # Ciclo 14a — Mantenimiento (oculto del cuerpo del reporte, contexto interno)
    last_balance_date: str = ""
    last_alignment_date: str = ""
    last_overhaul_date: str = ""
    commissioning_date: str = ""

    # Ciclo 14c.1 — Mapa de sensores de vibración por plano (API 670 / ISO 20816-1).
    # Cada sensor describe ubicación física + tipo + unidad + setpoints
    # individuales + patrón de match al Point del CSV cargado en Load Data.
    # Si está vacío, Tabular List cae a los defaults derivados de la instancia.
    sensors: List[Dict[str, Any]] = field(default_factory=list)

    # Datos capturados ad-hoc (legacy, sigue funcionando)
    captured_parameters: Dict[str, Any] = field(default_factory=dict)
    documents: List[Dict[str, Any]] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Instance":
        """
        Construye desde un dict serializado (de filesystem o Supabase).
        Resiliente a versiones previas: campos nuevos del Ciclo 14a
        usan default vacío si no están presentes en el dict.
        """
        def _f(k: str, default: Any = "") -> Any:
            v = data.get(k, default)
            return v if v is not None else default

        return cls(
            instance_id=_f("instance_id"),
            profile_key=_f("profile_key"),
            tag=_f("tag"),
            serial_number=_f("serial_number"),
            location=_f("location"),
            notes=_f("notes"),
            # extended
            client=_f("client"),
            site=_f("site"),
            asset_class=_f("asset_class"),
            driver_manufacturer=_f("driver_manufacturer"),
            driver_model=_f("driver_model"),
            driver_serial=_f("driver_serial"),
            driven_manufacturer=_f("driven_manufacturer"),
            driven_model=_f("driven_model"),
            driven_serial=_f("driven_serial"),
            nominal_power_mw=float(_f("nominal_power_mw", 0.0) or 0.0),
            nominal_rpm=float(_f("nominal_rpm", 0.0) or 0.0),
            min_rpm=float(_f("min_rpm", 0.0) or 0.0),
            max_rpm=float(_f("max_rpm", 0.0) or 0.0),
            trip_rpm=float(_f("trip_rpm", 0.0) or 0.0),
            iso_group=_f("iso_group"),
            support_type=_f("support_type"),
            support_count=int(_f("support_count", 0) or 0),
            support_detail=_f("support_detail"),
            probe_x_orientation_deg=float(_f("probe_x_orientation_deg", 0.0) or 0.0),
            probe_y_orientation_deg=float(_f("probe_y_orientation_deg", 0.0) or 0.0),
            alert_level=float(_f("alert_level", 0.0) or 0.0),
            danger_level=float(_f("danger_level", 0.0) or 0.0),
            trip_level=float(_f("trip_level", 0.0) or 0.0),
            setpoint_unit=_f("setpoint_unit"),
            coupling_class=_f("coupling_class"),
            schematic_png=_f("schematic_png"),
            last_balance_date=_f("last_balance_date"),
            last_alignment_date=_f("last_alignment_date"),
            last_overhaul_date=_f("last_overhaul_date"),
            commissioning_date=_f("commissioning_date"),
            sensors=list(data.get("sensors", []) or []),
            captured_parameters=dict(data.get("captured_parameters", {}) or {}),
            documents=list(data.get("documents", []) or []),
            created_at=_f("created_at"),
            updated_at=_f("updated_at"),
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
    **kwargs: Any,
) -> bool:
    """
    Actualiza campos de header (sin tocar parámetros capturados ni docs).
    Acepta todos los campos del Ciclo 14a vía kwargs — sólo se actualizan
    los que se pasan; los demás quedan intactos. Pasar None para un campo
    es no-op (no lo actualiza), pasar "" o 0 SÍ lo actualiza al valor vacío.
    """
    inst = get_instance(instance_id)
    if inst is None:
        return False

    # Campos legacy (Ciclo 8) + extendidos (Ciclo 14a) + sensores (Ciclo 14c.1)
    allowed = {
        "tag", "serial_number", "location", "notes", "profile_key",
        "client", "site", "asset_class",
        "driver_manufacturer", "driver_model", "driver_serial",
        "driven_manufacturer", "driven_model", "driven_serial",
        "nominal_power_mw", "nominal_rpm", "min_rpm", "max_rpm", "trip_rpm",
        "iso_group", "support_type", "support_count", "support_detail",
        "probe_x_orientation_deg", "probe_y_orientation_deg",
        "alert_level", "danger_level", "trip_level", "setpoint_unit",
        "coupling_class", "schematic_png",
        "last_balance_date", "last_alignment_date", "last_overhaul_date",
        "commissioning_date",
        "sensors",
    }
    for key, val in kwargs.items():
        if key in allowed and val is not None:
            setattr(inst, key, val)

    _save_instance(inst)
    return True


def compose_train_description(inst: Instance) -> str:
    """
    Compone una descripción narrativa del tren acoplado a partir de los
    campos driver/driven. Ejemplos:

      driver=GE LM6000, driven=Brush 54 MW
        → "Turbogenerador GE LM6000 acoplado a Generador Brush 54 MW"

      driver=GE TM2500, driven=Generador 25 MW
        → "Turbogenerador GE TM2500 acoplado a Generador 25 MW"

    Si los campos no están presentes, cae a notes/legacy. Si tampoco,
    devuelve string vacío para que el caller decida fallback.
    """
    asset_class = (inst.asset_class or "").strip()
    drv_mfr = (inst.driver_manufacturer or "").strip()
    drv_mdl = (inst.driver_model or "").strip()
    drvn_mfr = (inst.driven_manufacturer or "").strip()
    drvn_mdl = (inst.driven_model or "").strip()
    power = inst.nominal_power_mw or 0.0

    driver_part = " ".join(p for p in [drv_mfr, drv_mdl] if p).strip()
    driven_part = " ".join(p for p in [drvn_mfr, drvn_mdl] if p).strip()

    if asset_class and driver_part and driven_part:
        # Activo principal según la clase del tren
        head = asset_class.capitalize()
        if asset_class.upper() == "TURBOGENERADOR":
            head = "Turbogenerador"
        # No duplicar la palabra "Generador" en la frase final.
        # Casos típicos:
        #   driven="Brush Generador 54 MW" → contiene "generador" → no agrega prefijo
        #   driven="Brush 54 MW"           → no contiene → agrega prefijo "Generador"
        #   driven="Generador Brush 54 MW" → ya está → no agrega
        if asset_class.upper() == "TURBOGENERADOR":
            already_has = "generador" in driven_part.lower()
            driven_phrase = driven_part if already_has else f"Generador {driven_part}"
        else:
            driven_phrase = driven_part
        return f"{head} {driver_part} acoplado a {driven_phrase}"

    if driver_part and driven_part:
        return f"{driver_part} acoplado a {driven_part}"

    if driver_part:
        suffix = f" de {power:.0f} MW" if power > 0 else ""
        return f"{driver_part}{suffix}"

    # Fallback al legacy notes si existe
    return (inst.notes or "").strip()


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
    "compose_train_description",
]
