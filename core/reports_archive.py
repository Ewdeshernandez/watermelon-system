"""
core.reports_archive
====================

Archivo histórico INMUTABLE de reportes PDF (Ciclo 17.15).

Cuando el especialista termina un reporte y lo "aprueba", el PDF se
copia a una carpeta de archivo permanente:

    data/reports_archive/{email_slug}/{YYYY}/{MM}/{ts}_{slug}.pdf
    data/reports_archive/{email_slug}/{YYYY}/{MM}/{ts}_{slug}.json   (metadata sidecar)

Características:
  - Inmutable: no se puede sobrescribir un PDF archivado. Si querés
    una nueva versión del mismo reporte, se guarda con timestamp nuevo.
  - Per-usuario: cada autor tiene su carpeta de archivo, identificable
    por email_slug.
  - Visibilidad por role:
      * admin: ve TODOS los archivos de TODOS los usuarios
      * specialist: ve los suyos + los de otros @sigasas.com (read-only)
      * client: ve SOLO los archivos marcados shared_with_client=True
  - Metadata sidecar con info esencial para listar/filtrar sin tener
    que abrir el PDF: cliente, sitio, activo, fecha del análisis,
    severidad ejecutiva, n_páginas (si la pasamos), shared_with_client.

API pública:
  - archive_report_pdf(pdf_bytes, meta, owner_email, ...) → archive_id
  - list_archived_reports(viewer_email, viewer_role, filters)
  - get_archived_pdf_bytes(archive_id, viewer_email, viewer_role)
  - get_archived_metadata(archive_id)
  - delete_archived_report(archive_id, viewer_email, viewer_role)
    (solo el owner o admin pueden borrar)
  - share_with_client(archive_id, shared: bool)
"""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ARCHIVE_ROOT = DATA_DIR / "reports_archive"


# =============================================================
# UTILS
# =============================================================

def _slug(text: str, max_len: int = 60) -> str:
    """Slug corto y safe para filesystem."""
    s = (text or "").strip().lower()
    s = re.sub(r"[^a-z0-9_-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return (s or "unnamed")[:max_len]


def _email_slug(email: str) -> str:
    s = (email or "").strip().lower().replace("@", "_at_")
    s = re.sub(r"[^a-z0-9_-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "anonymous"


def _human_size(n: int) -> str:
    fn = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if fn < 1024:
            return f"{fn:.1f} {unit}"
        fn /= 1024
    return f"{fn:.1f} TB"


# =============================================================
# ARCHIVE — guardar reporte aprobado
# =============================================================

def archive_report_pdf(
    *,
    pdf_bytes: bytes,
    meta: Dict[str, Any],
    owner_email: str,
    archived_at: Optional[datetime] = None,
    shared_with_client: bool = False,
    extra_notes: str = "",
) -> Dict[str, Any]:
    """Archiva un PDF como copia inmutable. Devuelve dict con archive_id
    + paths + status.

    Args:
        pdf_bytes:           el PDF generado por pages/16_Reports.py
        meta:                el meta dict del reporte (cliente, sitio,
                             activo, train_description, etc.)
        owner_email:         email del autor del reporte
        archived_at:         timestamp; default = now
        shared_with_client:  si True, el client role asociado al cliente
                             puede verlo desde su panel
        extra_notes:         texto libre del autor sobre esta versión

    Returns:
        {
          "ok": True,
          "archive_id": "ehernandez_at_sigasas_com/2026/05/2026-05-03_223045_parex_c200c",
          "pdf_path":   "/abs/path/..../report.pdf",
          "meta_path":  "/abs/path/..../report.json",
          "size_bytes": int,
        }
    """
    if not pdf_bytes:
        return {"ok": False, "error": "PDF vacío."}
    if not owner_email:
        return {"ok": False, "error": "Falta owner_email."}

    ts = archived_at or datetime.now()
    yyyy = ts.strftime("%Y")
    mm = ts.strftime("%m")
    ts_str = ts.strftime("%Y-%m-%d_%H%M%S")

    # Slugs de cliente + activo para nombre legible del archivo
    client = meta.get("client", "") or meta.get("client_name", "") or "sin_cliente"
    asset = (
        meta.get("asset_class", "")
        or meta.get("train_description", "")
        or meta.get("instance_tag", "")
        or "sin_activo"
    )
    file_slug = f"{ts_str}_{_slug(client, 30)}_{_slug(asset, 30)}"

    owner_slug = _email_slug(owner_email)
    target_dir = ARCHIVE_ROOT / owner_slug / yyyy / mm
    target_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = target_dir / f"{file_slug}.pdf"
    meta_path = target_dir / f"{file_slug}.json"

    # INMUTABLE: si por alguna razón ya existe (timestamp colisión), agregar sufijo
    counter = 1
    while pdf_path.exists():
        counter += 1
        pdf_path = target_dir / f"{file_slug}_v{counter}.pdf"
        meta_path = target_dir / f"{file_slug}_v{counter}.json"
    if counter > 1:
        file_slug = f"{file_slug}_v{counter}"

    # Escribir el PDF
    pdf_path.write_bytes(pdf_bytes)

    # Sidecar con metadata
    sidecar = {
        "archive_id": f"{owner_slug}/{yyyy}/{mm}/{file_slug}",
        "owner_email": owner_email,
        "owner_slug": owner_slug,
        "archived_at": ts.isoformat(timespec="seconds"),
        "year": yyyy,
        "month": mm,
        "size_bytes": len(pdf_bytes),
        "size_human": _human_size(len(pdf_bytes)),
        "shared_with_client": bool(shared_with_client),
        "extra_notes": (extra_notes or "").strip(),
        # Snapshot del meta del reporte (cliente, sitio, activo, etc.)
        "report_meta": {
            "client":          meta.get("client", ""),
            "site":            meta.get("site", ""),
            "asset_class":     meta.get("asset_class", ""),
            "asset_model":     meta.get("asset_model", ""),
            "train_description": meta.get("train_description", ""),
            "instance_tag":    meta.get("instance_tag", ""),
            "instance_id":     meta.get("instance_id", ""),
            "report_date":     meta.get("report_date", ""),
            "consecutive":     meta.get("consecutive", ""),
            "executive_summary": meta.get("executive_summary", ""),
            "executive_severity": meta.get("executive_severity", ""),
            "prepared_by":     meta.get("prepared_by", ""),
            "reviewed_by":     meta.get("reviewed_by", ""),
        },
    }
    meta_path.write_text(
        json.dumps(sidecar, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    return {
        "ok": True,
        "archive_id": sidecar["archive_id"],
        "pdf_path": str(pdf_path),
        "meta_path": str(meta_path),
        "size_bytes": len(pdf_bytes),
        "size_human": _human_size(len(pdf_bytes)),
    }


# =============================================================
# LIST — buscar reportes archivados con filtros + permisos
# =============================================================

def _can_view(viewer_email: str, viewer_role: str, sidecar: Dict[str, Any]) -> bool:
    """Reglas de visibilidad por role:
       admin       → TODO
       specialist  → suyos + de otros @sigasas.com
       client      → solo shared_with_client=True
       (otros)     → solo suyos
    """
    role = (viewer_role or "").strip().lower()
    owner = (sidecar.get("owner_email") or "").lower()
    viewer = (viewer_email or "").lower()

    if role == "admin":
        return True
    if role == "specialist":
        # Suyos + cualquier @sigasas.com
        if owner == viewer:
            return True
        return owner.endswith("@sigasas.com")
    if role == "client":
        return bool(sidecar.get("shared_with_client"))
    # Default conservador
    return owner == viewer


def list_archived_reports(
    *,
    viewer_email: str,
    viewer_role: str,
    owner_filter: str = "",
    client_filter: str = "",
    asset_filter: str = "",
    date_from: str = "",
    date_to: str = "",
    text_search: str = "",
    limit: int = 200,
) -> List[Dict[str, Any]]:
    """Lista reportes archivados visibles para el viewer, con filtros.

    Args:
        viewer_email: quien está mirando
        viewer_role:  admin | specialist | client (define visibilidad)
        owner_filter: filtrar por owner_email (substring)
        client_filter: filtrar por cliente (substring case-insensitive)
        asset_filter: filtrar por activo (substring)
        date_from / date_to: ISO YYYY-MM-DD
        text_search: busca en notes + meta del reporte
        limit:       máximo de resultados

    Returns:
        Lista de sidecars que el viewer puede ver, ordenados por
        archived_at desc.
    """
    if not ARCHIVE_ROOT.exists():
        return []

    out: List[Dict[str, Any]] = []
    owner_filter_l = (owner_filter or "").strip().lower()
    client_filter_l = (client_filter or "").strip().lower()
    asset_filter_l = (asset_filter or "").strip().lower()
    text_search_l = (text_search or "").strip().lower()

    for sidecar_path in ARCHIVE_ROOT.rglob("*.json"):
        try:
            sc = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not _can_view(viewer_email, viewer_role, sc):
            continue

        # Filtros
        if owner_filter_l and owner_filter_l not in (sc.get("owner_email") or "").lower():
            continue
        rm = sc.get("report_meta", {}) or {}
        if client_filter_l and client_filter_l not in (rm.get("client") or "").lower():
            continue
        if asset_filter_l:
            asset_blob = " ".join([
                rm.get("asset_class", ""),
                rm.get("asset_model", ""),
                rm.get("instance_tag", ""),
                rm.get("train_description", ""),
            ]).lower()
            if asset_filter_l not in asset_blob:
                continue
        ts = sc.get("archived_at", "")[:10]
        if date_from and ts < date_from:
            continue
        if date_to and ts > date_to:
            continue
        if text_search_l:
            blob = json.dumps(sc, ensure_ascii=False).lower()
            if text_search_l not in blob:
                continue

        # Agregar el path del PDF para que el caller pueda descargarlo
        pdf_path = sidecar_path.with_suffix(".pdf")
        sc["_pdf_path"] = str(pdf_path)
        sc["_sidecar_path"] = str(sidecar_path)
        sc["_pdf_exists"] = pdf_path.exists()
        out.append(sc)

    out.sort(key=lambda x: x.get("archived_at", ""), reverse=True)
    return out[:limit]


def get_archived_pdf_bytes(
    archive_id: str,
    *,
    viewer_email: str = "",
    viewer_role: str = "",
) -> Optional[bytes]:
    """Devuelve los bytes del PDF archivado si el viewer tiene permiso.
    archive_id es el path relativo: 'owner_slug/YYYY/MM/file_slug'
    """
    pdf_path = ARCHIVE_ROOT / f"{archive_id}.pdf"
    sidecar_path = ARCHIVE_ROOT / f"{archive_id}.json"
    if not pdf_path.exists() or not sidecar_path.exists():
        return None
    try:
        sc = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if viewer_email or viewer_role:
        if not _can_view(viewer_email, viewer_role, sc):
            return None
    try:
        return pdf_path.read_bytes()
    except Exception:
        return None


def get_archived_metadata(archive_id: str) -> Optional[Dict[str, Any]]:
    sidecar_path = ARCHIVE_ROOT / f"{archive_id}.json"
    if not sidecar_path.exists():
        return None
    try:
        return json.loads(sidecar_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def delete_archived_report(
    archive_id: str,
    *,
    viewer_email: str,
    viewer_role: str,
) -> Dict[str, Any]:
    """Elimina un reporte archivado. Solo el owner o admin pueden borrar.
    Aunque el sistema sea "inmutable", el admin necesita poder limpiar.
    """
    sidecar_path = ARCHIVE_ROOT / f"{archive_id}.json"
    pdf_path = ARCHIVE_ROOT / f"{archive_id}.pdf"
    if not sidecar_path.exists():
        return {"ok": False, "error": "Archivo no encontrado."}
    try:
        sc = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except Exception:
        return {"ok": False, "error": "Sidecar corrupto."}

    role = (viewer_role or "").lower()
    owner = (sc.get("owner_email") or "").lower()
    viewer = (viewer_email or "").lower()
    if role != "admin" and owner != viewer:
        return {"ok": False, "error": "Solo el autor o admin pueden eliminar este reporte."}

    try:
        if pdf_path.exists():
            pdf_path.unlink()
        sidecar_path.unlink()
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def share_with_client(
    archive_id: str,
    shared: bool,
    *,
    viewer_email: str,
    viewer_role: str,
) -> Dict[str, Any]:
    """Toggle del flag shared_with_client. Solo owner o admin."""
    sidecar_path = ARCHIVE_ROOT / f"{archive_id}.json"
    if not sidecar_path.exists():
        return {"ok": False, "error": "Archivo no encontrado."}
    try:
        sc = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except Exception:
        return {"ok": False, "error": "Sidecar corrupto."}

    role = (viewer_role or "").lower()
    owner = (sc.get("owner_email") or "").lower()
    viewer = (viewer_email or "").lower()
    if role != "admin" and owner != viewer:
        return {"ok": False, "error": "Solo el autor o admin pueden cambiar este flag."}

    sc["shared_with_client"] = bool(shared)
    sc["shared_changed_at"] = datetime.now().isoformat(timespec="seconds")
    sc["shared_changed_by"] = viewer_email
    sidecar_path.write_text(json.dumps(sc, indent=2, ensure_ascii=False, default=str),
                             encoding="utf-8")
    return {"ok": True, "shared": bool(shared)}


def get_archive_stats() -> Dict[str, Any]:
    """Estadística general del archivo (para Home admin / KPIs)."""
    if not ARCHIVE_ROOT.exists():
        return {"total": 0, "by_owner": {}, "total_size": 0, "total_size_human": "0 B"}
    total = 0
    total_size = 0
    by_owner: Dict[str, int] = {}
    for sidecar_path in ARCHIVE_ROOT.rglob("*.json"):
        try:
            sc = json.loads(sidecar_path.read_text(encoding="utf-8"))
            total += 1
            owner = sc.get("owner_email", "(unknown)")
            by_owner[owner] = by_owner.get(owner, 0) + 1
            total_size += int(sc.get("size_bytes", 0) or 0)
        except Exception:
            continue
    return {
        "total": total,
        "by_owner": by_owner,
        "total_size": total_size,
        "total_size_human": _human_size(total_size),
    }


__all__ = [
    "ARCHIVE_ROOT",
    "archive_report_pdf",
    "list_archived_reports",
    "get_archived_pdf_bytes",
    "get_archived_metadata",
    "delete_archived_report",
    "share_with_client",
    "get_archive_stats",
]
