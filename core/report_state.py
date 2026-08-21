from __future__ import annotations

import base64
import errno
import json
import os
import re
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Flag de fallo de persistencia (disco lleno). La UI puede leerlo para avisar.
PERSIST_LAST_ERROR: Dict[str, Any] = {}


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

# Ciclo 23.169 — Raíz PERSISTENTE para los datos de usuario (reportes:
# estado, drafts e imágenes). El filesystem de Render es efímero (se borra
# en cada redeploy/reinicio), así que en producción se monta un DISCO
# persistente y se setea la env var WM_PERSIST_DIR (ej. /var/data). Si está
# definida, todo lo de data/users/ vive ahí y sobrevive reinicios. En local
# (sin la var) sigue usando data/ del repo — comportamiento idéntico al actual.
_PERSIST_ROOT = Path(os.environ["WM_PERSIST_DIR"]) if os.environ.get("WM_PERSIST_DIR") else DATA_DIR
USERS_ROOT = _PERSIST_ROOT / "users"

# Paths LEGACY (Ciclo <17.15) — globales, sin namespace por usuario.
# Todavía se usan como fallback y para detectar/migrar estado viejo.
_LEGACY_REPORT_STATE_FILE = DATA_DIR / "report_state.json"
_LEGACY_REPORT_DRAFTS_DIR = DATA_DIR / "report_drafts"

# Re-exports para back-compat: módulos viejos pueden importar estas
# constantes y siguen funcionando (resuelven al path del usuario activo
# vía las funciones de helper si están en una sesión Streamlit).
REPORT_STATE_FILE = _LEGACY_REPORT_STATE_FILE
REPORT_DRAFTS_DIR = _LEGACY_REPORT_DRAFTS_DIR

# =============================================================
# Ciclo 17.14.1 HOTFIX — Anti-pérdida de trabajo
# =============================================================
# Bug crítico reportado: con 50-100 imágenes, se pierde TODO el
# trabajo del día porque:
#   1. save_report_state escribía al JSON directo (NO atómico).
#      Si Streamlit crasheaba mid-write (timeout/OOM con 67MB
#      de JSON con base64 inline) → archivo corrupto.
#   2. load_report_state silenciosamente devolvía {} sin
#      avisar al usuario que el JSON estaba roto.
#
# Fix:
#   - Write atómico vía tempfile + os.replace
#   - Backup rotativo (.bak.1 → .bak.5) ANTES de cada write
#     exitoso, así siempre tenemos N versiones de respaldo
#   - load_report_state ahora intenta recovery desde backup
#     si el archivo principal está corrupto, y EXPONE el flag
#     'recovered_from' para que la UI muestre banner al usuario
#   - cleanup_old_backups limita a max N backups por archivo
# =============================================================

MAX_BACKUPS = 5  # cantidad de versiones de respaldo a mantener


# =============================================================
# Ciclo 17.15 — Namespacing por owner_email
# =============================================================
# Reemplaza el storage global por uno per-usuario:
#
#   data/                            (legacy, pre 17.15)
#     report_state.json              ← compartido entre todos (BUG)
#     report_drafts/{name}.json
#
#   data/users/{email_slug}/         (Ciclo 17.15+)
#     report_state.json              ← privado de cada usuario
#     report_state.json.bak.1..5     ← backups rotativos
#     report_drafts/{name}.json      ← drafts privados
#
# Los módulos consumidores (Trends, Spectrum, Polar, Bode, etc.) NO
# necesitan cambiar nada — el namespacing se resuelve internamente
# leyendo `st.session_state["auth_email"]`.
#
# Migración automática: si existe el archivo legacy `data/report_state.json`
# Y el path del usuario activo es admin, se mueve al espacio del admin
# como su trabajo personal (asumiendo que era él quien tenía el reporte
# en curso). Para otros usuarios, el legacy queda intacto hasta que el
# admin lo gestione manualmente.

def _email_slug(email: str) -> str:
    """Convierte email a slug seguro para filesystem.
    'ehernandez@sigasas.com' → 'ehernandez_at_sigasas_com'
    """
    s = (email or "").strip().lower().replace("@", "_at_")
    s = re.sub(r"[^a-z0-9_-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "anonymous"


def _current_owner_email() -> str:
    """Lee el email del usuario activo desde session_state. Si no
    hay sesión Streamlit (corriendo standalone) devuelve string vacío.
    """
    try:
        import streamlit as st  # type: ignore
        v = st.session_state.get("auth_email", "") or ""
        return str(v).strip().lower()
    except Exception:
        return ""


def get_user_data_dir(email: Optional[str] = None) -> Path:
    """Devuelve `data/users/{slug}/` para el usuario indicado o el activo.
    Crea el directorio si no existe.
    """
    e = email if email is not None else _current_owner_email()
    slug = _email_slug(e or "anonymous")
    d = USERS_ROOT / slug
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_user_state_file(email: Optional[str] = None) -> Path:
    """Path al report_state.json del usuario indicado o activo."""
    return get_user_data_dir(email) / "report_state.json"


def get_user_drafts_dir(email: Optional[str] = None) -> Path:
    """Path a la carpeta de drafts nombrados del usuario."""
    d = get_user_data_dir(email) / "report_drafts"
    d.mkdir(parents=True, exist_ok=True)
    return d


# =============================================================
# Ciclo 17.17 — Image storage refactor
# =============================================================
# Bug crítico que matamos acá:
#   Antes: cada imagen se persistía como base64 INLINE dentro del
#   report_state.json. Con 50 imágenes el JSON pesaba 67MB, con
#   100 imágenes >130MB. Streamlit Cloud (1GB RAM) no aguantaba:
#     - json.dump tardaba 10-30s y bloqueaba el rerun
#     - si el usuario hacía algo mid-write → crash → JSON corrupto
#     - json.loads al cargar reventaba la RAM
#     - reportes perdidos en cada caída
#
# Ahora: cada imagen vive como PNG suelto en
#   data/users/{slug}/report_images/{item_id}.png
# y el JSON solo guarda `image_file` (el nombre del PNG).
#
# Beneficios:
#   - JSON queda en KBs en vez de MBs → write atómico real
#   - load no carga 100MB de strings de una vez
#   - una imagen corrupta no rompe el reporte entero
#   - reportes y drafts comparten el mismo image pool por item_id
#
# Migración transparente: cuando se carga un JSON con
# `image_bytes_b64` (formato viejo), restore_report_items decodifica
# los bytes, escribe el PNG a disco y deja `image_file` apuntando
# al nuevo archivo. La próxima vez que se guarde el reporte, el
# `image_bytes_b64` legacy se va y queda solo `image_file`.
# =============================================================

def get_user_images_dir(email: Optional[str] = None) -> Path:
    """Path a la carpeta donde viven los PNGs sueltos del usuario.
    Compartida entre el reporte actual y todos los drafts: si un
    item con id X aparece en current y en draft, ambos refieren al
    mismo PNG (deduplicación natural por item_id).
    """
    d = get_user_data_dir(email) / "report_images"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _safe_item_id(raw: Any) -> str:
    """Sanitiza un item id para usarlo como nombre de archivo.
    Solo deja [A-Za-z0-9_-], colapsa repetidos, recorta a 120 chars.
    """
    s = str(raw or "").strip()
    s = re.sub(r"[^A-Za-z0-9_-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:120] or "item"


def _atomic_write_png(target: Path, data: bytes) -> None:
    """Escribe `data` como PNG a `target` ATÓMICAMENTE (igual que el
    helper de JSON pero binario): tmp file → fsync → os.replace.
    Si crashea mid-write, el archivo final queda intacto en su
    versión previa (o no existe si era nuevo).
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_name = tempfile.mkstemp(
        prefix=target.stem + ".",
        suffix=".png.tmp",
        dir=str(target.parent),
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(tmp_fd, "wb") as f:
            f.write(data)
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                pass
        os.replace(str(tmp_path), str(target))
    except Exception:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
        raise


def _resolve_images_dir(images_dir: Optional[Path],
                          email: Optional[str]) -> Optional[Path]:
    """Devuelve el dir de imágenes a usar. Si el caller lo pasa, lo
    respeta; si no, intenta resolverlo desde el email (o el activo).
    Si no se puede resolver (ej. corriendo standalone sin sesión),
    devuelve None y el caller cae a base64 inline para no perder datos.
    """
    if images_dir is not None:
        try:
            images_dir.mkdir(parents=True, exist_ok=True)
            return images_dir
        except Exception:
            return None
    try:
        e = email if email is not None else _current_owner_email()
        if not e:
            return None
        return get_user_images_dir(e)
    except Exception:
        return None


def _maybe_migrate_legacy_to_user(email: str) -> None:
    """Si existe el report_state.json legacy global Y NO existe el del
    usuario actual, lo movemos al espacio del usuario.

    Solo migra para el ADMIN ÚNICO porque el archivo legacy era el
    "reporte global" del sistema viejo de admin/demo. Para otros
    usuarios, el legacy queda intacto.
    """
    if not email:
        return
    try:
        from core.supabase_auth import is_admin_email
        if not is_admin_email(email):
            return
    except Exception:
        # Si no podemos importar supabase_auth, no migrar
        return

    user_state = get_user_state_file(email)
    if user_state.exists():
        return  # ya tiene su archivo, no pisar
    if not _LEGACY_REPORT_STATE_FILE.exists():
        return  # no hay legacy

    try:
        # Copia (no movimiento) para que el legacy quede como referencia
        shutil.copy2(_LEGACY_REPORT_STATE_FILE, user_state)
        # Y los drafts también
        if _LEGACY_REPORT_DRAFTS_DIR.exists():
            user_drafts = get_user_drafts_dir(email)
            for p in _LEGACY_REPORT_DRAFTS_DIR.glob("*.json"):
                tgt = user_drafts / p.name
                if not tgt.exists():
                    shutil.copy2(p, tgt)
    except Exception:
        pass


def _encode_image_bytes(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return base64.b64encode(value).decode("utf-8")
    return ""


def _decode_image_bytes(value: Any) -> bytes | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return base64.b64decode(text.encode("utf-8"))
    except Exception:
        return None


def _safe_slug(text: Any) -> str:
    raw = str(text or "").strip().lower()
    raw = re.sub(r"[^a-z0-9_-]+", "_", raw)
    raw = re.sub(r"_+", "_", raw).strip("_")
    return raw or "draft"


def sanitize_report_items(items: Any, *,
                            email: Optional[str] = None,
                            images_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Convierte la lista de items en memoria al formato persistible en JSON.

    Ciclo 17.17: las imágenes ya NO se guardan como base64 inline. Se
    escriben como PNG sueltos en `data/users/{slug}/report_images/` y
    el JSON solo guarda `image_file` (el nombre del PNG, relativo a esa
    carpeta). Esto baja el JSON de 67MB a unos pocos KB con 50 imágenes.

    Args:
        items:        lista de items en memoria (con `image_bytes` bytes
                      o `image_bytes_b64` legacy o `image_file` ya migrado)
        email:        owner del reporte. Se usa para resolver el dir de
                      imágenes. Si no se pasa, intenta el activo de sesión.
        images_dir:   override explícito de la carpeta de PNGs. Útil para
                      tests o exports a archivos arbitrarios.

    Returns:
        Lista de dicts JSON-serializables. Cada item con imagen tiene
        `image_file` apuntando al PNG. Solo cae a `image_bytes_b64` si
        no pudo escribir el PNG (fallback anti-pérdida de datos).
    """
    out: List[Dict[str, Any]] = []

    if not isinstance(items, list):
        return out

    resolved_dir = _resolve_images_dir(images_dir, email)

    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            continue

        item_id = str(item.get("id") or f"report_item_{idx+1}")
        rec: Dict[str, Any] = {
            "id":        item_id,
            "type":      str(item.get("type") or "figure"),
            "title":     str(item.get("title") or f"Figura {idx+1}"),
            "notes":     str(item.get("notes") or ""),
            "signal_id": str(item.get("signal_id") or ""),
            "machine":   str(item.get("machine") or ""),
            "point":     str(item.get("point") or ""),
            "variable":  str(item.get("variable") or ""),
            "timestamp": str(item.get("timestamp") or ""),
        }

        # Resolver bytes de la imagen, vengan en cualquiera de los 3
        # formatos posibles (in-memory, b64 legacy, o image_file persistido)
        raw_bytes: Optional[bytes] = None
        rb = item.get("image_bytes")
        if isinstance(rb, (bytes, bytearray)) and len(rb) > 0:
            raw_bytes = bytes(rb)
        elif item.get("image_bytes_b64"):
            decoded = _decode_image_bytes(item.get("image_bytes_b64"))
            if decoded:
                raw_bytes = decoded

        # Caso 1: tenemos bytes Y carpeta de imágenes → PNG suelto
        if raw_bytes is not None and resolved_dir is not None:
            try:
                fname = _safe_item_id(item_id) + ".png"
                _atomic_write_png(resolved_dir / fname, raw_bytes)
                rec["image_file"] = fname
            except Exception:
                # Fallback duro: si el PNG no se pudo escribir, dejamos
                # el b64 inline para no perder la imagen. Mejor un JSON
                # gordo que un reporte sin imagen.
                rec["image_bytes_b64"] = _encode_image_bytes(raw_bytes)

        # Caso 2: tenemos bytes pero NO hay carpeta resoluble (corrida
        # standalone sin sesión Streamlit, o destino arbitrario)
        elif raw_bytes is not None:
            rec["image_bytes_b64"] = _encode_image_bytes(raw_bytes)

        # Caso 3: el item ya viene con image_file (cargado y re-serializado
        # sin tocar la imagen) → preservar el path tal cual
        elif item.get("image_file"):
            rec["image_file"] = str(item["image_file"])

        out.append(rec)

    return out


def restore_report_items(items: Any, *,
                           email: Optional[str] = None,
                           images_dir: Optional[Path] = None,
                           migrate_legacy: bool = True) -> List[Dict[str, Any]]:
    """Inverso de sanitize_report_items: reconstruye los items en
    memoria desde lo persistido en el JSON.

    Ciclo 17.20 ULTRA-IMPORTANTE — LAZY LOADING:
      Antes esta función cargaba TODOS los image_bytes de TODOS los
      items en memoria al hacer load_report_state. Con 10 items de
      ~1MB cada uno, eso son 10MB constantes en st.session_state que
      se replican en cada rerun. Más reportlab cargando todo de una
      al generar el PDF → 50-100 MB transitorios → OOM en Streamlit
      Cloud (1 GB).

      Ahora devuelve `image_bytes = None` y `image_file = "ruta.png"`.
      Los consumers (Reports render, PDF generation) deben usar
      `read_item_image_bytes(item)` para leer el PNG desde disco SOLO
      cuando lo van a usar.

      Esto baja la memoria de session_state de ~20MB a ~50KB con 10
      imágenes y elimina los crashes con muchas imágenes.

    Args:
        items:           lista de items leídos del JSON
        email:           owner del reporte (resuelve images_dir)
        images_dir:      override explícito de la carpeta de PNGs
        migrate_legacy:  si True (default), reescribe los b64 legacy
                         como PNGs sueltos durante la carga
    """
    out: List[Dict[str, Any]] = []

    if not isinstance(items, list):
        return out

    resolved_dir = _resolve_images_dir(images_dir, email)
    # Guardamos también el path absoluto del images_dir en cada item para
    # que `read_item_image_bytes` pueda resolverlo sin necesidad de saber
    # el email del owner (útil para drafts y exports cross-context).
    resolved_dir_str = str(resolved_dir) if resolved_dir is not None else ""

    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            continue

        item_id = str(item.get("id") or f"report_item_{idx+1}")
        rec: Dict[str, Any] = {
            "id":               item_id,
            "type":             str(item.get("type") or "figure"),
            "title":            str(item.get("title") or f"Figura {idx+1}"),
            "notes":            str(item.get("notes") or ""),
            "signal_id":        str(item.get("signal_id") or ""),
            "machine":          str(item.get("machine") or ""),
            "point":            str(item.get("point") or ""),
            "variable":         str(item.get("variable") or ""),
            "timestamp":        str(item.get("timestamp") or ""),
            "figure":           None,
            "image_bytes":      None,             # LAZY — leer con read_item_image_bytes
            "image_file":       "",
            "_images_dir":      resolved_dir_str,  # privado: para resolver el path
        }

        # 1) Formato nuevo: image_file → registramos path, NO cargamos bytes
        img_file = str(item.get("image_file") or "").strip()
        if img_file and resolved_dir is not None:
            png_path = resolved_dir / img_file
            if png_path.exists():
                rec["image_file"] = img_file
                # NO leemos los bytes acá. Se leen on-demand via
                # read_item_image_bytes() solo cuando se renderiza.
                out.append(rec)
                continue

        # 2) Legacy: image_bytes_b64 inline → MIGRAR a PNG y dejar lazy
        b64 = item.get("image_bytes_b64")
        if b64:
            decoded = _decode_image_bytes(b64)
            if decoded is not None:
                # Si podemos migrar a PNG, lo hacemos y NO mantenemos
                # los bytes en memoria. Si no, último recurso: bytes
                # inmediatos para no perder la imagen.
                if migrate_legacy and resolved_dir is not None:
                    try:
                        fname = _safe_item_id(item_id) + ".png"
                        _atomic_write_png(resolved_dir / fname, decoded)
                        rec["image_file"] = fname
                        out.append(rec)
                        continue
                    except Exception:
                        pass
                # Fallback: mantener bytes (legacy sin migrar)
                rec["image_bytes"] = decoded

        out.append(rec)

    return out


def read_item_image_bytes(item: Dict[str, Any]) -> Optional[bytes]:
    """Lee los bytes de la imagen de un report item ON-DEMAND.

    Ciclo 17.20: este es el accessor lazy que reemplaza el acceso
    directo a `item["image_bytes"]`. Lee el PNG desde disco solo
    cuando se llama, evitando mantener todos los bytes en memoria.

    Soporta los 3 formatos:
      1. item["image_bytes"] ya tiene bytes (legacy o caller que
         acaba de generar la imagen) → devolver directo
      2. item["image_file"] + item["_images_dir"] → leer PNG
      3. ninguno → None

    Args:
        item: dict del report item

    Returns:
        bytes del PNG, o None si no se pudo obtener.
    """
    if not isinstance(item, dict):
        return None

    # Caso 1: ya tiene bytes en memoria (item recién creado por un
    # módulo, antes del primer load_report_state)
    rb = item.get("image_bytes")
    if isinstance(rb, (bytes, bytearray)) and len(rb) > 0:
        return bytes(rb)

    # Caso 2: lazy desde disco
    img_file = str(item.get("image_file") or "").strip()
    if not img_file:
        return None

    # Resolver el dir de imágenes — primero del propio item, después
    # del usuario activo
    images_dir_str = str(item.get("_images_dir") or "").strip()
    if images_dir_str:
        png_path = Path(images_dir_str) / img_file
    else:
        try:
            png_path = get_user_images_dir() / img_file
        except Exception:
            return None

    if not png_path.exists():
        return None
    try:
        return png_path.read_bytes()
    except Exception:
        return None


def _serialize_state(*, items: Any, meta: Any,
                       email: Optional[str] = None,
                       images_dir: Optional[Path] = None) -> Dict[str, Any]:
    return {
        "items": sanitize_report_items(items, email=email, images_dir=images_dir),
        "meta": meta if isinstance(meta, dict) else {},
    }


def _restore_state(raw: Dict[str, Any],
                     email: Optional[str] = None,
                     images_dir: Optional[Path] = None) -> Dict[str, Any]:
    return {
        "items": restore_report_items(
            raw.get("items", []),
            email=email,
            images_dir=images_dir,
        ),
        "meta": raw.get("meta", {}) if isinstance(raw.get("meta", {}), dict) else {},
    }


# =============================================================
# Ciclo 17.14.1 — Helpers internos de write atómico + backups
# =============================================================

def _purge_backups(target: Path) -> int:
    """Borra TODOS los backups .bak.1..N de `target`. Devuelve cuántos borró.
    Los backups son solo para recovery; en disco lleno se pueden sacrificar."""
    n = 0
    for i in range(1, MAX_BACKUPS + 1):
        b = _backup_path(target, i)
        try:
            if b.exists():
                b.unlink(); n += 1
        except Exception:
            pass
    return n


def _purge_orphan_tmp(directory: Path) -> int:
    """Borra archivos temporales huérfanos (*.tmp) que quedaron de escrituras
    interrumpidas (típico cuando el disco se llenó a medio write)."""
    n = 0
    try:
        for p in Path(directory).glob("*.tmp"):
            try:
                p.unlink(); n += 1
            except Exception:
                pass
    except Exception:
        pass
    return n


def free_disk_space_best_effort() -> Dict[str, int]:
    """Libera espacio de forma SEGURA en todo el árbol de datos: elimina
    backups .bak.* y temporales .tmp huérfanos de todos los usuarios. No toca
    el estado vigente, drafts ni imágenes. Devuelve conteos."""
    baks = tmps = 0
    roots = {_PERSIST_ROOT, DATA_DIR}
    for root in roots:
        try:
            if not root.exists():
                continue
            for p in root.rglob("*.bak.*"):
                try:
                    p.unlink(); baks += 1
                except Exception:
                    pass
            for p in root.rglob("*.tmp"):
                try:
                    p.unlink(); tmps += 1
                except Exception:
                    pass
        except Exception:
            pass
    return {"backups": baks, "tmp": tmps}


def _backup_path(target: Path, idx: int) -> Path:
    """Devuelve el path del backup número `idx` (1..MAX_BACKUPS)
    para el archivo `target`. Ej: report_state.json → report_state.json.bak.1
    """
    return target.with_suffix(target.suffix + f".bak.{idx}")


def _rotate_backups(target: Path) -> None:
    """Rota los backups del archivo `target`:
       - .bak.5 se elimina (más viejo)
       - .bak.4 → .bak.5
       - .bak.3 → .bak.4
       - ...
       - .bak.1 → .bak.2
       - target → .bak.1 (más reciente)
    Si el target no existe (primer save), simplemente no rota nada.
    """
    if not target.exists():
        return
    # Eliminar el más viejo (.bak.MAX_BACKUPS) si existe
    oldest = _backup_path(target, MAX_BACKUPS)
    if oldest.exists():
        try:
            oldest.unlink()
        except Exception:
            pass
    # Rotar desde el (MAX-1) hacia atrás
    for i in range(MAX_BACKUPS - 1, 0, -1):
        src = _backup_path(target, i)
        dst = _backup_path(target, i + 1)
        if src.exists():
            try:
                src.replace(dst)
            except Exception:
                pass
    # target → .bak.1 (copia, no movimiento, para que el target
    # quede disponible mientras escribimos el nuevo)
    try:
        shutil.copy2(target, _backup_path(target, 1))
    except Exception:
        pass


def _atomic_write_json(target: Path, payload: Dict[str, Any]) -> None:
    """Escribe `payload` a `target` ATÓMICAMENTE:
       1. Escribe a un archivo temporal en el mismo directorio
       2. fsync para asegurar bytes en disco antes del rename
       3. os.replace (atómico en POSIX y Windows) → si crashea,
          el archivo final queda intacto en su versión previa
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_name = tempfile.mkstemp(
        prefix=target.stem + ".",
        suffix=".tmp",
        dir=str(target.parent),
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                pass
        # Rename atómico — si crashea aquí, target queda con
        # contenido viejo, NO con archivo corrupto a medio escribir
        os.replace(str(tmp_path), str(target))
    except Exception:
        # Si algo falló, asegurar limpieza del temp
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
        raise


# =============================================================
# API PÚBLICA
# =============================================================

def _resolve_state_file(filename: Optional[Path] = None,
                         email: Optional[str] = None) -> Path:
    """Ciclo 17.15: si el caller no pasa filename, devuelve el path
    per-usuario (data/users/{slug}/report_state.json). Si tampoco hay
    sesión Streamlit, cae al path legacy global para back-compat.
    """
    if filename is not None:
        return filename
    e = email if email is not None else _current_owner_email()
    if e:
        # Una sola vez por sesión, intentar migración legacy → user
        return get_user_state_file(e)
    return _LEGACY_REPORT_STATE_FILE


def save_report_state(*, items: Any, meta: Any,
                       filename: Path | None = None,
                       email: Optional[str] = None) -> None:
    """Persiste el estado del reporte de forma SEGURA.

    Ciclo 17.14.1 HOTFIX:
      - Antes del write, rota backups (mantiene 5 versiones)
      - Write ATÓMICO via tmp file + os.replace
      - Si crashea mid-write, el archivo final queda intacto
        y se puede recuperar desde el backup más reciente

    Ciclo 17.15: si no se pasa filename, persiste al espacio per-usuario
    (data/users/{email_slug}/report_state.json).
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    target = _resolve_state_file(filename, email)
    target.parent.mkdir(parents=True, exist_ok=True)

    # Ciclo 17.17: pasamos el email para que sanitize sepa dónde escribir
    # los PNGs sueltos (data/users/{slug}/report_images/). Si el caller
    # pasó un filename arbitrario (caso draft o export) Y no pasó email,
    # caemos al activo de sesión.
    effective_email = email if email is not None else _current_owner_email()

    def _do_save() -> None:
        payload = _serialize_state(items=items, meta=meta, email=effective_email)
        payload["_save_meta"] = {
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "n_items": len(payload.get("items", [])),
            "owner_email": effective_email or "",
        }
        _rotate_backups(target)
        _atomic_write_json(target, payload)

    # v3.31.5xx — Resiliencia a DISCO LLENO (ENOSPC). Antes, un disco lleno
    # hacía crashear TODA la página de Reports al arrancar (autosave). Ahora:
    # si no hay espacio, liberamos backups + temporales huérfanos y reintentamos
    # una vez; si aún falla, NO crasheamos — la sesión sigue viva, solo se
    # marca el fallo de persistencia para que la UI avise.
    try:
        _do_save()
        PERSIST_LAST_ERROR.clear()
    except OSError as exc:
        if getattr(exc, "errno", None) == errno.ENOSPC:
            _purge_backups(target)
            _purge_orphan_tmp(target.parent)
            try:
                free_disk_space_best_effort()
            except Exception:
                pass
            try:
                _do_save()
                PERSIST_LAST_ERROR.clear()
            except OSError as exc2:
                PERSIST_LAST_ERROR.clear()
                PERSIST_LAST_ERROR.update({
                    "enospc": True,
                    "message": "Disco lleno: no se pudo autoguardar. El trabajo "
                               "de esta sesión sigue en memoria; libera espacio "
                               "en el servidor para volver a persistir.",
                    "at": datetime.now().isoformat(timespec="seconds"),
                    "detail": str(exc2),
                })
            return
        raise


def load_report_state(*, filename: Path | None = None,
                       email: Optional[str] = None) -> Dict[str, Any]:
    """Carga el estado del reporte. Si el archivo principal está
    corrupto, intenta recovery automático desde los backups.

    Ciclo 17.14.1 HOTFIX:
      - Si el JSON principal falla → intenta .bak.1, luego .bak.2, etc.
      - Si recupera desde backup, agrega `_recovered_from` al dict
        para que la UI pueda mostrar un banner al usuario
      - Si TODOS fallan, devuelve {} pero con `_load_error` poblado
        para que la UI sepa que hubo un problema (no silent fail)

    Ciclo 17.15: si no se pasa filename, lee del espacio per-usuario.
    Migra automáticamente del path legacy global si corresponde.
    """
    e = email if email is not None else _current_owner_email()
    # Migración automática legacy → user (idempotente, solo para admin)
    if filename is None and e:
        _maybe_migrate_legacy_to_user(e)

    target = _resolve_state_file(filename, e)
    if not target.exists():
        return {"items": [], "meta": {}}

    # Intento principal
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
        result = _restore_state(raw, email=e)
        # No agregamos _recovered_from porque cargó normal
        return result
    except Exception as primary_err:
        # Recovery desde backups (1..MAX, del más reciente al más viejo)
        for i in range(1, MAX_BACKUPS + 1):
            bk = _backup_path(target, i)
            if not bk.exists():
                continue
            try:
                raw = json.loads(bk.read_text(encoding="utf-8"))
                result = _restore_state(raw, email=e)
                # Marcar de dónde se recuperó para que la UI avise
                result["_recovered_from"] = bk.name
                result["_recovered_at"] = datetime.now().isoformat(timespec="seconds")
                # Reescribir el archivo principal con el contenido del backup
                # para que las próximas cargas no tengan que recurrir al backup.
                # (Si esto falla, no es bloqueante — la lectura ya tuvo éxito)
                try:
                    _atomic_write_json(target, raw)
                except Exception:
                    pass
                return result
            except Exception:
                continue
        # Ningún backup salvable
        return {
            "items": [],
            "meta": {},
            "_load_error": str(primary_err),
            "_load_error_at": datetime.now().isoformat(timespec="seconds"),
        }


def clear_report_state(*, filename: Path | None = None,
                        email: Optional[str] = None) -> None:
    """Borra el estado actual del reporte.
    Ciclo 17.14.1: NO toca los backups (.bak.*) — siguen disponibles
    si después el usuario quiere recuperar.
    Ciclo 17.15: opera sobre el espacio per-usuario por default.
    """
    target = _resolve_state_file(filename, email)
    if target.exists():
        target.unlink()


def list_available_backups(filename: Path | None = None,
                            email: Optional[str] = None) -> List[Dict[str, Any]]:
    """Devuelve lista de backups disponibles con metadata útil.
    Útil para UI de "restaurar desde backup específico" si se necesita.
    Ciclo 17.15: opera sobre el espacio per-usuario por default.
    """
    target = _resolve_state_file(filename, email)
    out: List[Dict[str, Any]] = []
    for i in range(1, MAX_BACKUPS + 1):
        bk = _backup_path(target, i)
        if not bk.exists():
            continue
        try:
            stat = bk.stat()
            # Intentar leer el _save_meta del backup
            saved_at = ""
            n_items = 0
            try:
                raw = json.loads(bk.read_text(encoding="utf-8"))
                sm = raw.get("_save_meta", {}) or {}
                saved_at = sm.get("saved_at", "") or ""
                n_items = int(sm.get("n_items", 0) or 0)
            except Exception:
                pass
            out.append({
                "backup_idx": i,
                "filename": bk.name,
                "path": str(bk),
                "size_bytes": stat.st_size,
                "size_human": _human_size(stat.st_size),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
                "saved_at": saved_at,
                "n_items": n_items,
            })
        except Exception:
            continue
    return out


def _human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


# =============================================================
# Ciclo 17.5.6 — Helpers compartidos para envío al reporte
# =============================================================
# Bug histórico: cada módulo (Polar, Bode, SCL, Spectrum,
# Time Waveform, Trends, Tabular) hacía
# st.session_state.report_items.append(...) directo. La página
# Reports cargaba report_state.json desde disco SOLO la primera
# vez que se entraba a Reports en la sesión, vía un flag
# `report_state_loaded`. Si el usuario añadía items desde un
# módulo ANTES de visitar Reports por primera vez, al entrar a
# Reports se cargaba el estado de disco encima del estado en
# memoria — perdiendo los items recién agregados. Por eso la UX
# era "envío al reporte → no aparece → debo ir a Reports y
# volver y reenviar para que cargue".
#
# Solución: dos helpers compartidos que TODOS los módulos
# llaman antes/después de modificar report_items:
#
#   ensure_report_state_loaded()
#       Carga report_state.json a session_state si todavía no
#       se cargó. Idempotente vía flag. Hace merge con lo que
#       ya esté en memoria (priority a memoria sobre disco para
#       casos de race entre upload y load).
#
#   append_report_item_and_persist(item)
#       Asegura load + appende item + persiste a disco. Así la
#       fuente de verdad (disco) siempre queda sincronizada con
#       memoria, y aunque el usuario nunca abra Reports, el
#       reporte queda armado.

def ensure_report_state_loaded() -> None:
    """Asegura que `st.session_state['report_items']` y
    `st.session_state['report_meta']` reflejen el contenido
    persistido en disco. Idempotente — múltiples llamadas en la
    misma sesión sólo cargan una vez.

    Si el caller ya añadió items a memoria antes de la primera
    carga (caso del bug original), se hace merge: los items en
    memoria se preservan y se añaden los faltantes del disco
    detectados por id.
    """
    try:
        import streamlit as st  # type: ignore
    except Exception:
        return

    # Ciclo 17.15 — el flag de "loaded" ahora se asocia al email del
    # owner. Si el usuario cambia (logout + login con otro), el flag
    # se invalida automáticamente y se recarga el estado del nuevo.
    _current = _current_owner_email() or "anonymous"
    _loaded_for = st.session_state.get("report_state_loaded_for")
    if _loaded_for == _current:
        return

    persisted = load_report_state()
    disk_items = persisted.get("items", []) if isinstance(persisted, dict) else []
    disk_meta = persisted.get("meta", {}) if isinstance(persisted, dict) else {}

    # Ciclo 17.14.1 — Propagar flags de recovery/error a session_state
    # para que la UI de Reports pueda mostrar banners visibles al usuario.
    if isinstance(persisted, dict):
        rec_from = persisted.get("_recovered_from")
        rec_at = persisted.get("_recovered_at")
        load_err = persisted.get("_load_error")
        load_err_at = persisted.get("_load_error_at")
        if rec_from:
            st.session_state["wm_report_recovered_from"] = rec_from
            st.session_state["wm_report_recovered_at"] = rec_at or ""
            st.session_state["wm_report_recovered_n_items"] = len(disk_items)
        if load_err:
            st.session_state["wm_report_load_error"] = load_err
            st.session_state["wm_report_load_error_at"] = load_err_at or ""

    # Si memoria ya tiene items (race del bug), preservamos esos y
    # adjuntamos los faltantes del disco por id.
    in_memory = list(st.session_state.get("report_items", []) or [])
    if in_memory:
        seen_ids = {str(it.get("id") or "") for it in in_memory if it.get("id")}
        merged = list(in_memory)
        if isinstance(disk_items, list):
            for it in disk_items:
                if not isinstance(it, dict):
                    continue
                _iid = str(it.get("id") or "")
                if _iid and _iid in seen_ids:
                    continue
                merged.append(it)
        st.session_state["report_items"] = merged
    else:
        st.session_state["report_items"] = disk_items if isinstance(disk_items, list) else []

    if not st.session_state.get("report_meta"):
        st.session_state["report_meta"] = (
            disk_meta if isinstance(disk_meta, dict) else {}
        )

    st.session_state["report_state_loaded_for"] = _current
    # Mantener flag legacy para back-compat por si algún módulo lo lee
    st.session_state["report_state_loaded"] = True


def append_report_item_and_persist(item: Dict[str, Any]) -> bool:
    """Añade un item al reporte y lo persiste a disco al mismo
    tiempo. Devuelve True si la persistencia tuvo éxito (la
    adición a memoria siempre se hace).

    Pensado para uso desde cualquier módulo (Polar, Bode, SCL,
    Spectrum, Time Waveform, Trends, Tabular) — reemplaza el
    patrón frágil ``st.session_state.report_items.append(item)``
    que dependía de que Reports estuviera ya inicializado.
    """
    try:
        import streamlit as st  # type: ignore
    except Exception:
        return False

    ensure_report_state_loaded()
    st.session_state.setdefault("report_items", [])
    if isinstance(item, dict):
        st.session_state["report_items"].append(item)

    try:
        save_report_state(
            items=st.session_state["report_items"],
            meta=st.session_state.get("report_meta", {}) or {},
        )
        # Ciclo 17.20: después del save, recargamos los items en forma
        # LAZY (image_bytes → None, image_file → path al PNG ya escrito).
        # Esto libera la memoria de los image_bytes inmediatamente, sin
        # esperar al próximo rerun. Crítico para evitar OOM cuando se
        # envían muchas imágenes al reporte en la misma sesión.
        try:
            _persisted = load_report_state()
            _lazy_items = _persisted.get("items", [])
            if isinstance(_lazy_items, list):
                st.session_state["report_items"] = _lazy_items
        except Exception:
            pass  # no bloqueante — el save ya fue exitoso
        return True
    except Exception:
        # La memoria ya está actualizada, lo único que falla es
        # la persistencia a disco — no rompemos al usuario.
        return False


def _draft_path(draft_name: Any, email: Optional[str] = None) -> Path:
    """Ciclo 17.15: drafts ahora viven per-usuario en
    data/users/{email_slug}/report_drafts/{name}.json.
    """
    drafts_dir = get_user_drafts_dir(email) if (email is not None or _current_owner_email()) else _LEGACY_REPORT_DRAFTS_DIR
    drafts_dir.mkdir(parents=True, exist_ok=True)
    return drafts_dir / f"{_safe_slug(draft_name)}.json"


def list_report_drafts(email: Optional[str] = None) -> List[str]:
    """Lista los drafts del usuario indicado o activo.
    Ciclo 17.15: por usuario, no global.
    """
    if email is not None or _current_owner_email():
        drafts_dir = get_user_drafts_dir(email)
    else:
        drafts_dir = _LEGACY_REPORT_DRAFTS_DIR
    if not drafts_dir.exists():
        return []
    return [p.stem for p in sorted(drafts_dir.glob("*.json"))]


def save_named_report_draft(*, draft_name: Any, items: Any, meta: Any,
                              email: Optional[str] = None) -> str:
    """Guarda un draft nombrado. Ciclo 17.15: en el espacio del usuario.
    Ciclo 17.17: las imágenes se persisten como PNG sueltos en el image
    pool compartido del usuario (data/users/{slug}/report_images/), no
    inline en el JSON. Si dos drafts comparten un item por id, comparten
    también el PNG (deduplicación natural).
    """
    target = _draft_path(draft_name, email)
    effective_email = email if email is not None else _current_owner_email()
    payload = _serialize_state(items=items, meta=meta, email=effective_email)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str),
                      encoding="utf-8")
    return target.stem


def load_named_report_draft(draft_name: Any,
                             email: Optional[str] = None) -> Dict[str, Any]:
    target = _draft_path(draft_name, email)
    if not target.exists():
        return {"items": [], "meta": {}}
    return load_report_state(filename=target, email=email)


def delete_named_report_draft(draft_name: Any,
                               email: Optional[str] = None) -> None:
    target = _draft_path(draft_name, email)
    if target.exists():
        target.unlink()


# =============================================================
# Ciclo 17.15 — Helpers de visibilidad cross-usuario (admin)
# =============================================================

def list_all_users_with_state() -> List[Dict[str, Any]]:
    """Lista todos los usuarios que tienen report_state guardado en disco.
    Útil para que admin/specialist puedan ver "los reportes de quién están
    activos en el sistema".
    Devuelve [{email_slug, n_items, last_saved, n_drafts, total_size_bytes}].
    """
    users_root = USERS_ROOT
    if not users_root.exists():
        return []
    out: List[Dict[str, Any]] = []
    for user_dir in users_root.iterdir():
        if not user_dir.is_dir():
            continue
        state_file = user_dir / "report_state.json"
        drafts_dir = user_dir / "report_drafts"
        n_items = 0
        last_saved = ""
        owner_email = ""
        size = 0
        if state_file.exists():
            try:
                stat = state_file.stat()
                size += stat.st_size
                raw = json.loads(state_file.read_text(encoding="utf-8"))
                sm = raw.get("_save_meta", {}) or {}
                n_items = int(sm.get("n_items", 0) or 0)
                last_saved = sm.get("saved_at", "") or ""
                owner_email = sm.get("owner_email", "") or ""
            except Exception:
                pass
        n_drafts = 0
        if drafts_dir.exists():
            n_drafts = len(list(drafts_dir.glob("*.json")))
            for p in drafts_dir.glob("*.json"):
                try:
                    size += p.stat().st_size
                except Exception:
                    pass
        out.append({
            "email_slug": user_dir.name,
            "owner_email": owner_email,
            "n_items": n_items,
            "n_drafts": n_drafts,
            "last_saved": last_saved,
            "total_size_bytes": size,
            "total_size_human": _human_size(size),
        })
    out.sort(key=lambda x: x.get("last_saved", ""), reverse=True)
    return out
