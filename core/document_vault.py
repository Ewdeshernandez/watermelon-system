"""
core.document_vault
===================

Almacenamiento e indexación de documentos técnicos por asset profile.
Permite que cada máquina monitoreada tenga adjuntos persistentes
(manuales OEM, reportes de mantenimiento, certificados, especificaciones,
fotografías) y que parámetros clave del activo (clearance del cojinete,
diámetro del journal, propiedades del aceite, umbrales del fabricante)
queden capturados como datos estructurados consultables desde cualquier
módulo de análisis.

Estructura en disco:

    data/
      asset_documents/
        {profile_key}/
          {doc_id}__{filename}
      asset_metadata/
        {profile_key}.json   ← índice de documentos + parámetros capturados

Cada documento se identifica por un UUID generado al subir. El nombre
del archivo en disco preserva el original con el UUID como prefijo para
evitar colisiones y mantener trazabilidad.

API principal:
    add_document(profile_key, file_obj, ...) -> doc_id
    list_documents(profile_key) -> List[Dict]
    get_document_path(profile_key, doc_id) -> Path
    delete_document(profile_key, doc_id) -> bool
    get_captured_parameters(profile_key) -> Dict
    update_captured_parameter(profile_key, key, value) -> None
    get_full_metadata(profile_key) -> Dict
"""

from __future__ import annotations

import json
import re
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DOCUMENTS_DIR = DATA_DIR / "asset_documents"
METADATA_DIR = DATA_DIR / "asset_metadata"


# =============================================================
# DOCUMENT TYPES PRE-DEFINIDOS
# =============================================================

DOCUMENT_TYPES = {
    "bearing_repair_report": "Reporte de reparación / rebabbitado de cojinete",
    "oem_manual": "Manual del fabricante (OEM)",
    "commissioning_report": "Reporte de comisionamiento",
    "vibration_baseline": "Línea base de vibración",
    "alignment_report": "Reporte de alineación",
    "balancing_report": "Reporte de balanceo de campo",
    "maintenance_history": "Historial de mantenimiento",
    "electrical_report": "Reporte eléctrico (motor, generador)",
    "thermography_report": "Termografía",
    "oil_analysis": "Análisis de aceite",
    "inspection_certificate": "Certificado de inspección",
    "rotor_data_sheet": "Hoja de datos del rotor",
    "bearing_data_sheet": "Hoja de datos del cojinete",
    "photograph": "Fotografía / registro visual",
    "other": "Otro",
}


# =============================================================
# CAPTURED PARAMETERS — campos estructurados que los módulos
# pueden leer del Vault para enriquecer su análisis
# =============================================================

CAPTURED_PARAMETER_FIELDS = {
    # Geometría del cojinete
    "bearing_inner_diameter_mm": {
        "label": "Diámetro interno del cojinete (mm)",
        "type": "float",
        "category": "Cojinete - geometría",
        "help": "Diámetro interno final mecanizado del cojinete (Φ del muñón con clearance).",
    },
    "shaft_journal_diameter_mm": {
        "label": "Diámetro del journal del eje (mm)",
        "type": "float",
        "category": "Cojinete - geometría",
        "help": "Diámetro del muñón del eje en la zona del cojinete.",
    },
    "bearing_length_mm": {
        "label": "Longitud axial del cojinete (mm)",
        "type": "float",
        "category": "Cojinete - geometría",
        "help": "Longitud axial efectiva del cojinete (L). Relación L/D típica 0.5-1.0.",
    },
    "diametral_clearance_mm": {
        "label": "Clearance diametral Cd (mm)",
        "type": "float",
        "category": "Cojinete - geometría",
        "help": "Clearance diametral total. Típico para cojinetes planos: 0.0010 - 0.0020 × Φ.",
    },

    # Material y condición del cojinete
    "babbitt_material": {
        "label": "Material babbitt",
        "type": "str",
        "category": "Cojinete - material",
        "help": "Aleación antifricción. Ej: ASTM B-23 Grade 2, BERA 90, etc.",
    },
    "last_rebabbiting_date": {
        "label": "Última fecha de rebabbitado",
        "type": "date",
        "category": "Cojinete - material",
        "help": "Fecha de la última intervención de rebabbitado del cojinete.",
    },
    "max_babbitt_temp_c": {
        "label": "Temperatura máxima admisible babbitt (°C)",
        "type": "float",
        "category": "Cojinete - material",
        "help": "Límite de temperatura del babbitt antes de degradación. Típico: 130 °C.",
    },

    # Rotor / carga
    "rotor_total_weight_kg": {
        "label": "Peso total del rotor (kg)",
        "type": "float",
        "category": "Rotor",
        "help": "Peso total del rotor según fabricante o pesaje en taller.",
    },
    "static_load_per_bearing_kn": {
        "label": "Carga estática por cojinete (kN)",
        "type": "float",
        "category": "Rotor",
        "help": "Reacción estática en cada cojinete. Para Sommerfeld y eccentricity calc.",
    },

    # Operación nominal
    "rated_power_mw": {
        "label": "Potencia nominal (MW)",
        "type": "float",
        "category": "Operación",
    },
    "rated_speed_rpm": {
        "label": "Velocidad nominal (rpm)",
        "type": "float",
        "category": "Operación",
    },
    "max_speed_rpm": {
        "label": "Velocidad máxima permitida (rpm)",
        "type": "float",
        "category": "Operación",
    },

    # Lubricación
    "oil_grade": {
        "label": "Grado del aceite",
        "type": "str",
        "category": "Lubricación",
        "help": "Ej: ISO VG 32, ISO VG 46, Mobil DTE Heavy Medium, etc.",
    },
    "oil_viscosity_cst_40c": {
        "label": "Viscosidad cinemática a 40 °C (cSt)",
        "type": "float",
        "category": "Lubricación",
    },
    "oil_inlet_temp_c": {
        "label": "Temperatura de entrada del aceite (°C)",
        "type": "float",
        "category": "Lubricación",
    },

    # Umbrales OEM (Brush, Siemens, Bently API 670, etc.)
    "oem_alert_um_pp": {
        "label": "OEM Alert (µm pp shaft displacement)",
        "type": "float",
        "category": "Umbrales OEM",
        "help": "Nivel Alert recomendado por el fabricante. Usar en Asset Profile threshold_source='oem'.",
    },
    "oem_danger_um_pp": {
        "label": "OEM Danger (µm pp shaft displacement)",
        "type": "float",
        "category": "Umbrales OEM",
    },
    "oem_alert_mm_s_rms": {
        "label": "OEM Alert (mm/s RMS casing velocity)",
        "type": "float",
        "category": "Umbrales OEM",
    },
    "oem_danger_mm_s_rms": {
        "label": "OEM Danger (mm/s RMS casing velocity)",
        "type": "float",
        "category": "Umbrales OEM",
    },

    # Historial
    "last_overhaul_date": {
        "label": "Última overhaul (fecha)",
        "type": "date",
        "category": "Historial",
    },
    "last_balance_date": {
        "label": "Último balance de campo (fecha)",
        "type": "date",
        "category": "Historial",
    },
    "total_running_hours": {
        "label": "Horas totales de operación",
        "type": "float",
        "category": "Historial",
    },

    # Notas libres
    "asset_serial_number": {
        "label": "Número de serie del activo",
        "type": "str",
        "category": "Identificación",
    },
    "asset_tag": {
        "label": "Tag interno del activo",
        "type": "str",
        "category": "Identificación",
        "help": "Ej: TES1 2026, U-101, etc.",
    },
    "asset_location": {
        "label": "Ubicación / planta",
        "type": "str",
        "category": "Identificación",
        "help": "Ej: Termoeléctrica Cartagena, Magnex Ecopetrol, etc.",
    },
    "asset_notes": {
        "label": "Notas adicionales del activo",
        "type": "text",
        "category": "Identificación",
    },
}


# =============================================================
# HELPERS DE FILESYSTEM
# =============================================================

def _safe_filename(name: str) -> str:
    """Convierte un nombre arbitrario a algo seguro para filesystem."""
    name = name.strip().replace(" ", "_")
    name = re.sub(r"[^A-Za-z0-9._-]", "", name)
    return name[:120] if name else "document"


def _ensure_dirs(profile_key: str) -> None:
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    (DOCUMENTS_DIR / profile_key).mkdir(parents=True, exist_ok=True)


def _metadata_path(profile_key: str) -> Path:
    return METADATA_DIR / f"{profile_key}.json"


def _load_metadata(profile_key: str) -> Dict[str, Any]:
    path = _metadata_path(profile_key)
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    return {
        "profile_key": profile_key,
        "documents": [],
        "captured_parameters": {},
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }


def _save_metadata(profile_key: str, meta: Dict[str, Any]) -> None:
    _ensure_dirs(profile_key)
    meta["updated_at"] = datetime.now().isoformat(timespec="seconds")
    path = _metadata_path(profile_key)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False, default=str)


# =============================================================
# API PÚBLICA
# =============================================================

def add_document(
    profile_key: str,
    file_obj: Any,
    *,
    title: str = "",
    document_type: str = "other",
    description: str = "",
    tags: Optional[List[str]] = None,
) -> str:
    """
    Guarda un archivo asociado al profile y lo agrega al índice.

    Args:
        profile_key: clave del asset profile (ej. 'brush_turbogenerator_54mw_3600')
        file_obj: archivo subido (UploadedFile de Streamlit, bytes, o path)
        title: título descriptivo (ej. 'Reporte Wersin rebabbiting cojinetes 2018')
        document_type: clave de DOCUMENT_TYPES
        description: descripción libre
        tags: lista de tags para búsqueda

    Returns:
        document_id (UUID generado)
    """
    _ensure_dirs(profile_key)

    doc_id = uuid.uuid4().hex[:12]

    # Resolver el contenido binario
    original_filename = "document"
    if hasattr(file_obj, "name"):
        original_filename = str(file_obj.name)
    elif isinstance(file_obj, (str, Path)):
        original_filename = Path(file_obj).name

    clean_name = _safe_filename(original_filename)
    storage_filename = f"{doc_id}__{clean_name}"
    storage_path = DOCUMENTS_DIR / profile_key / storage_filename

    # Escribir el archivo
    if hasattr(file_obj, "getvalue"):
        data = file_obj.getvalue()
        with open(storage_path, "wb") as f:
            f.write(data)
        size_bytes = len(data)
    elif hasattr(file_obj, "read"):
        try:
            file_obj.seek(0)
        except Exception:
            pass
        data = file_obj.read()
        with open(storage_path, "wb") as f:
            f.write(data if isinstance(data, bytes) else data.encode("utf-8"))
        size_bytes = storage_path.stat().st_size
    elif isinstance(file_obj, bytes):
        with open(storage_path, "wb") as f:
            f.write(file_obj)
        size_bytes = len(file_obj)
    elif isinstance(file_obj, (str, Path)):
        shutil.copyfile(file_obj, storage_path)
        size_bytes = storage_path.stat().st_size
    else:
        raise ValueError(f"Tipo de file_obj no soportado: {type(file_obj)}")

    # Actualizar índice
    meta = _load_metadata(profile_key)
    doc_record = {
        "id": doc_id,
        "filename": original_filename,
        "storage_filename": storage_filename,
        "title": title or original_filename,
        "type": document_type if document_type in DOCUMENT_TYPES else "other",
        "description": description,
        "tags": tags or [],
        "uploaded_at": datetime.now().isoformat(timespec="seconds"),
        "size_bytes": size_bytes,
    }
    meta.setdefault("documents", []).append(doc_record)
    _save_metadata(profile_key, meta)

    return doc_id


def list_documents(profile_key: str) -> List[Dict[str, Any]]:
    """Lista los documentos asociados a un profile, ordenados por fecha desc."""
    meta = _load_metadata(profile_key)
    docs = list(meta.get("documents", []))
    docs.sort(key=lambda d: d.get("uploaded_at", ""), reverse=True)
    return docs


def get_document_path(profile_key: str, doc_id: str) -> Optional[Path]:
    """Devuelve el path en disco del archivo, o None si no existe."""
    meta = _load_metadata(profile_key)
    for d in meta.get("documents", []):
        if d.get("id") == doc_id:
            path = DOCUMENTS_DIR / profile_key / d["storage_filename"]
            if path.exists():
                return path
            return None
    return None


def get_document_bytes(profile_key: str, doc_id: str) -> Optional[bytes]:
    """Lee el archivo del Vault y devuelve sus bytes."""
    path = get_document_path(profile_key, doc_id)
    if path is None:
        return None
    with open(path, "rb") as f:
        return f.read()


def delete_document(profile_key: str, doc_id: str) -> bool:
    """Borra el archivo del filesystem y lo remueve del índice."""
    meta = _load_metadata(profile_key)
    docs = meta.get("documents", [])
    target = None
    for d in docs:
        if d.get("id") == doc_id:
            target = d
            break
    if target is None:
        return False

    path = DOCUMENTS_DIR / profile_key / target["storage_filename"]
    try:
        if path.exists():
            path.unlink()
    except Exception:
        pass

    meta["documents"] = [d for d in docs if d.get("id") != doc_id]
    _save_metadata(profile_key, meta)
    return True


def get_captured_parameters(profile_key: str) -> Dict[str, Any]:
    """Devuelve los parámetros estructurados capturados para el profile."""
    meta = _load_metadata(profile_key)
    return dict(meta.get("captured_parameters", {}))


def update_captured_parameter(profile_key: str, key: str, value: Any) -> None:
    """Actualiza (o crea) un parámetro capturado y persiste."""
    meta = _load_metadata(profile_key)
    params = meta.setdefault("captured_parameters", {})
    if value is None or value == "":
        params.pop(key, None)
    else:
        params[key] = value
    _save_metadata(profile_key, meta)


def update_captured_parameters_bulk(profile_key: str, values: Dict[str, Any]) -> None:
    """Actualiza varios parámetros de una vez."""
    meta = _load_metadata(profile_key)
    params = meta.setdefault("captured_parameters", {})
    for k, v in values.items():
        if v is None or v == "":
            params.pop(k, None)
        else:
            params[k] = v
    _save_metadata(profile_key, meta)


def get_full_metadata(profile_key: str) -> Dict[str, Any]:
    """Acceso completo a la metadata del profile (documentos + parámetros)."""
    return _load_metadata(profile_key)


def estimate_diametral_clearance_mm(
    bearing_inner_diameter_mm: float,
    *,
    severity: str = "typical",
) -> float:
    """
    Estimación heurística de clearance diametral cuando no se tiene el dato OEM.

    Reglas del oficio (cojinetes hidrodinámicos planos):
        typical:    Cd ≈ 0.0015 × Φ
        tight:      Cd ≈ 0.0010 × Φ  (alta velocidad, low clearance design)
        loose:      Cd ≈ 0.0020 × Φ  (low speed, larger clearance)

    Args:
        bearing_inner_diameter_mm: diámetro interno del cojinete en mm
        severity: 'tight' / 'typical' / 'loose'

    Returns:
        clearance diametral estimado en mm
    """
    factor_map = {"tight": 0.0010, "typical": 0.0015, "loose": 0.0020}
    factor = factor_map.get(severity, 0.0015)
    return float(bearing_inner_diameter_mm) * factor


__all__ = [
    "DOCUMENT_TYPES",
    "CAPTURED_PARAMETER_FIELDS",
    "add_document",
    "list_documents",
    "get_document_path",
    "get_document_bytes",
    "delete_document",
    "get_captured_parameters",
    "update_captured_parameter",
    "update_captured_parameters_bulk",
    "get_full_metadata",
    "estimate_diametral_clearance_mm",
]
