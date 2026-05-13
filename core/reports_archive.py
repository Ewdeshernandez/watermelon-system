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
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ARCHIVE_ROOT = DATA_DIR / "reports_archive"

# Ciclo 17.29 CRÍTICO — Persistencia vía Supabase Storage.
# Streamlit Cloud usa contenedores efímeros: cada redeploy borra
# data/reports_archive/. Antes de este fix, todos los reportes
# archivados se perdían en cada redeploy. Ahora los subimos a
# Supabase Storage en un bucket dedicado, y los restauramos al
# filesystem en cold start.
ARCHIVE_BUCKET_NAME = "reports-archive"
_SUPABASE_CLIENT_CACHE: Any = None
_SUPABASE_CLIENT_TRIED: bool = False
_SYNC_FROM_DONE: bool = False  # Para llamar sync_from_supabase 1 vez por proceso


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
# SUPABASE STORAGE — persistencia entre redeploys (Ciclo 17.29)
# =============================================================
# Streamlit Cloud usa contenedores efímeros. Cada redeploy borra el
# filesystem local. Para que los reportes archivados sobrevivan,
# los persistimos en un bucket Supabase Storage dedicado.
#
# El usuario debe crear el bucket UNA VEZ en Supabase Dashboard:
#   Storage → New bucket → Name: 'reports-archive' → Public: NO
#   Las RLS policies se manejan con la service_key (admin).
#
# Layout en Supabase:
#   reports-archive/{owner_slug}/{YYYY}/{MM}/{file_slug}.pdf
#   reports-archive/{owner_slug}/{YYYY}/{MM}/{file_slug}.json
#
# Mismo layout que filesystem local → 1:1 mapping del archive_id.

def _get_archive_supabase_client() -> Any:
    """Devuelve un cliente Supabase para el bucket de archivo.
    None si supabase no está configurado o el SDK no está instalado.
    Cached lazy: 1 build por proceso.

    Resolución de credenciales (Ciclo 19 hotfix):
      1. st.secrets["supabase"] (cuando corre dentro de Streamlit Cloud)
      2. Variables de entorno SUPABASE_URL + SUPABASE_SERVICE_KEY
         (necesario para que la API REST en Render acceda al archivo
         sin Streamlit en el proceso).
    """
    global _SUPABASE_CLIENT_CACHE, _SUPABASE_CLIENT_TRIED
    if _SUPABASE_CLIENT_TRIED:
        return _SUPABASE_CLIENT_CACHE
    _SUPABASE_CLIENT_TRIED = True

    url = ""
    key = ""

    # Fuente 1: st.secrets (Streamlit)
    try:
        import streamlit as st
        if "supabase" in st.secrets:
            cfg = st.secrets["supabase"]
            url = (cfg.get("url", "") or "").strip()
            key = (cfg.get("service_key", "") or "").strip()
    except Exception:
        pass

    # Fuente 2: env vars (API REST en Render, sin Streamlit)
    if not url or not key:
        url = url or os.environ.get("SUPABASE_URL", "").strip()
        key = key or (
            os.environ.get("SUPABASE_SERVICE_KEY", "").strip()
            or os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip()
        )

    if not url or not key:
        return None

    try:
        from supabase import create_client
        _SUPABASE_CLIENT_CACHE = create_client(url, key)
        return _SUPABASE_CLIENT_CACHE
    except Exception as exc:
        print(
            f"[WM_ARCHIVE] No se pudo inicializar cliente Supabase: {exc}",
            file=sys.stderr, flush=True,
        )
        return None


def _supabase_path_for(archive_id: str, suffix: str) -> str:
    """Convierte archive_id en path dentro del bucket.
    archive_id = 'owner_slug/YYYY/MM/file_slug' → path = 'owner_slug/YYYY/MM/file_slug.pdf'."""
    return f"{archive_id}{suffix}"


def _upload_to_supabase(
    archive_id: str,
    pdf_bytes: bytes,
    sidecar_dict: Dict[str, Any],
) -> bool:
    """Sube PDF + sidecar JSON a Supabase Storage. Idempotente (upsert).
    Devuelve True si ambos uploads fueron OK."""
    client = _get_archive_supabase_client()
    if client is None:
        return False
    try:
        bucket = client.storage.from_(ARCHIVE_BUCKET_NAME)
        pdf_path = _supabase_path_for(archive_id, ".pdf")
        json_path = _supabase_path_for(archive_id, ".json")
        bucket.upload(
            pdf_path, pdf_bytes,
            {"upsert": "true", "content-type": "application/pdf"},
        )
        sidecar_bytes = json.dumps(
            sidecar_dict, indent=2, ensure_ascii=False, default=str
        ).encode("utf-8")
        bucket.upload(
            json_path, sidecar_bytes,
            {"upsert": "true", "content-type": "application/json"},
        )
        print(
            f"[WM_ARCHIVE] Supabase upload OK: {archive_id} "
            f"({_human_size(len(pdf_bytes))})",
            file=sys.stderr, flush=True,
        )
        return True
    except Exception as exc:
        print(
            f"[WM_ARCHIVE] Supabase upload FAIL para {archive_id}: {exc}",
            file=sys.stderr, flush=True,
        )
        return False


def _download_from_supabase(archive_id: str) -> Tuple[Optional[bytes], Optional[Dict[str, Any]]]:
    """Baja PDF + sidecar desde Supabase. Devuelve (pdf_bytes, sidecar_dict)
    o (None, None) si no se pudo bajar."""
    client = _get_archive_supabase_client()
    if client is None:
        return None, None
    try:
        bucket = client.storage.from_(ARCHIVE_BUCKET_NAME)
        pdf_bytes = bucket.download(_supabase_path_for(archive_id, ".pdf"))
        sidecar_raw = bucket.download(_supabase_path_for(archive_id, ".json"))
        sidecar_dict = (
            json.loads(sidecar_raw.decode("utf-8"))
            if sidecar_raw else None
        )
        return pdf_bytes, sidecar_dict
    except Exception as exc:
        print(
            f"[WM_ARCHIVE] Supabase download FAIL para {archive_id}: {exc}",
            file=sys.stderr, flush=True,
        )
        return None, None


def _delete_from_supabase(archive_id: str) -> bool:
    """Borra PDF + sidecar de Supabase. Idempotente."""
    client = _get_archive_supabase_client()
    if client is None:
        return False
    try:
        bucket = client.storage.from_(ARCHIVE_BUCKET_NAME)
        bucket.remove([
            _supabase_path_for(archive_id, ".pdf"),
            _supabase_path_for(archive_id, ".json"),
        ])
        return True
    except Exception as exc:
        print(
            f"[WM_ARCHIVE] Supabase delete FAIL para {archive_id}: {exc}",
            file=sys.stderr, flush=True,
        )
        return False


def _list_supabase_archive_files() -> List[str]:
    """Lista todos los .json (sidecars) del bucket. Devuelve archive_ids
    sin extensión. Recorre la jerarquía owner/YYYY/MM/file."""
    client = _get_archive_supabase_client()
    if client is None:
        return []
    out: List[str] = []
    try:
        bucket = client.storage.from_(ARCHIVE_BUCKET_NAME)
        # Walk: list by depth. Supabase no expone rglob; iteramos por nivel.
        owners_resp = bucket.list("", {"limit": 1000})
        owners = [
            o["name"] for o in (owners_resp or [])
            if o.get("name") and not o["name"].startswith(".")
        ]
        for owner in owners:
            try:
                years_resp = bucket.list(owner, {"limit": 100})
            except Exception:
                continue
            years = [
                y["name"] for y in (years_resp or [])
                if y.get("name") and y["name"].isdigit()
            ]
            for year in years:
                try:
                    months_resp = bucket.list(
                        f"{owner}/{year}", {"limit": 100}
                    )
                except Exception:
                    continue
                months = [
                    m["name"] for m in (months_resp or [])
                    if m.get("name") and m["name"].isdigit()
                ]
                for month in months:
                    try:
                        files_resp = bucket.list(
                            f"{owner}/{year}/{month}", {"limit": 1000}
                        )
                    except Exception:
                        continue
                    for f in (files_resp or []):
                        nm = f.get("name", "")
                        if nm.endswith(".json"):
                            archive_id = (
                                f"{owner}/{year}/{month}/"
                                f"{nm[:-len('.json')]}"
                            )
                            out.append(archive_id)
        return out
    except Exception as exc:
        print(
            f"[WM_ARCHIVE] Supabase list FAIL: {exc}",
            file=sys.stderr, flush=True,
        )
        return []


def sync_archive_from_supabase(*, force: bool = False) -> Dict[str, int]:
    """Restaura el archivo histórico desde Supabase al filesystem
    local. Llamada en cold start (lazy, una vez por proceso).

    Args:
        force: si True, re-baja todo aunque ya exista localmente.
               Default False = solo baja archive_ids ausentes.

    Returns:
        {"downloaded": int, "skipped": int, "failed": int}
    """
    global _SYNC_FROM_DONE
    if _SYNC_FROM_DONE and not force:
        return {"downloaded": 0, "skipped": 0, "failed": 0,
                "already_synced": True}

    client = _get_archive_supabase_client()
    if client is None:
        _SYNC_FROM_DONE = True
        return {"downloaded": 0, "skipped": 0, "failed": 0,
                "supabase_unavailable": True}

    archive_ids = _list_supabase_archive_files()
    stats = {"downloaded": 0, "skipped": 0, "failed": 0}

    for aid in archive_ids:
        local_pdf = ARCHIVE_ROOT / f"{aid}.pdf"
        local_json = ARCHIVE_ROOT / f"{aid}.json"
        if (not force and local_pdf.exists() and local_json.exists()):
            stats["skipped"] += 1
            continue
        pdf_bytes, sidecar_dict = _download_from_supabase(aid)
        if pdf_bytes is None or sidecar_dict is None:
            stats["failed"] += 1
            continue
        try:
            local_pdf.parent.mkdir(parents=True, exist_ok=True)
            local_pdf.write_bytes(pdf_bytes)
            local_json.write_text(
                json.dumps(sidecar_dict, indent=2, ensure_ascii=False,
                           default=str),
                encoding="utf-8",
            )
            stats["downloaded"] += 1
        except Exception as exc:
            print(
                f"[WM_ARCHIVE] Falló escritura local de {aid}: {exc}",
                file=sys.stderr, flush=True,
            )
            stats["failed"] += 1

    _SYNC_FROM_DONE = True
    if stats["downloaded"] > 0 or stats["failed"] > 0:
        print(
            f"[WM_ARCHIVE] sync_from_supabase: {stats}",
            file=sys.stderr, flush=True,
        )
    return stats


def sync_archive_to_supabase(*, force: bool = False) -> Dict[str, int]:
    """Sube todos los reportes locales que NO están en Supabase. Útil
    para migrar archivos existentes la primera vez que se configura el
    bucket. Idempotente.

    Args:
        force: si True, re-sube todo aunque ya esté en Supabase.

    Returns: {"uploaded": int, "skipped": int, "failed": int}.
    """
    client = _get_archive_supabase_client()
    if client is None:
        return {"uploaded": 0, "skipped": 0, "failed": 0,
                "supabase_unavailable": True}

    if not ARCHIVE_ROOT.exists():
        return {"uploaded": 0, "skipped": 0, "failed": 0}

    remote_ids = set(_list_supabase_archive_files()) if not force else set()
    stats = {"uploaded": 0, "skipped": 0, "failed": 0}

    for sidecar_path in ARCHIVE_ROOT.rglob("*.json"):
        try:
            sc = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        archive_id = sc.get("archive_id", "")
        if not archive_id:
            continue
        pdf_path = sidecar_path.with_suffix(".pdf")
        if not pdf_path.exists():
            continue
        if not force and archive_id in remote_ids:
            stats["skipped"] += 1
            continue
        try:
            pdf_bytes = pdf_path.read_bytes()
            ok = _upload_to_supabase(archive_id, pdf_bytes, sc)
            if ok:
                stats["uploaded"] += 1
            else:
                stats["failed"] += 1
        except Exception:
            stats["failed"] += 1
    print(
        f"[WM_ARCHIVE] sync_to_supabase: {stats}",
        file=sys.stderr, flush=True,
    )
    return stats


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

    # Ciclo 17.29 CRÍTICO — Subir a Supabase Storage para que sobreviva
    # al próximo redeploy de Streamlit Cloud. Si Supabase no está
    # configurado o falla, el archivo local sigue válido pero quedará
    # vulnerable a pérdida en el próximo redeploy. El upload es
    # best-effort: no bloquea el archivado local.
    supabase_uploaded = _upload_to_supabase(
        sidecar["archive_id"], pdf_bytes, sidecar
    )

    return {
        "ok": True,
        "archive_id": sidecar["archive_id"],
        "pdf_path": str(pdf_path),
        "meta_path": str(meta_path),
        "size_bytes": len(pdf_bytes),
        "size_human": _human_size(len(pdf_bytes)),
        "supabase_uploaded": supabase_uploaded,
    }


# =============================================================
# LIST — buscar reportes archivados con filtros + permisos
# =============================================================

def _can_view(viewer_email: str, viewer_role: str, sidecar: Dict[str, Any]) -> bool:
    """Reglas de visibilidad por role:
       admin       → TODO
       specialist  → suyos + de otros @sigasas.com
       client      → shared_with_client=True  OR  match_strings/asset_tags
                     coinciden con el cliente al que pertenece el viewer.
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
        # Ciclo 23.131 — Visibilidad ampliada:
        # (a) flag explícito shared_with_client, O
        # (b) el reporte matchea con los match_strings/asset_tags del
        #     cliente al que pertenece el viewer (auto-share por scope).
        if bool(sidecar.get("shared_with_client")):
            return True
        try:
            from core.clients import get_client_for_email
            c = get_client_for_email(viewer)
            if c is None:
                return False
            rm = (sidecar.get("report_meta") or {})
            # Texto donde buscamos match: client / instance_tag /
            # asset_class / train_description / asset
            haystacks = [
                str(rm.get("client", "")),
                str(rm.get("instance_tag", "")),
                str(rm.get("asset", "")),
                str(rm.get("asset_class", "")),
                str(rm.get("train_description", "")),
                str(sidecar.get("client", "")),
                str(sidecar.get("asset", "")),
            ]
            haystack_lc = " ".join(haystacks).lower()
            # Match por match_strings (substring case-insensitive)
            for s in c.match_strings:
                s_lc = str(s).strip().lower()
                if s_lc and s_lc in haystack_lc:
                    return True
            # Match por asset_tags (TES1, TES3, etc.)
            for t in c.asset_tags:
                t_lc = str(t).strip().lower()
                if t_lc and t_lc in haystack_lc:
                    return True
            return False
        except Exception:
            return False
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
    # Ciclo 17.29 CRÍTICO — En cold start (filesystem efímero recién
    # restaurado por Streamlit Cloud), el archive local está vacío.
    # Sincronizamos desde Supabase Storage la primera vez que se
    # llama esta función por proceso. Subsecuentes llamadas usan
    # filesystem (rápido). Idempotente.
    if not _SYNC_FROM_DONE:
        sync_archive_from_supabase()

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

    # Ciclo 17.29 — Si los archivos no están localmente (cold start o
    # primer acceso desde un proceso reciente), intentar bajar de
    # Supabase. Cubre el caso de un viewer que pidió un archive_id
    # antes de que sync_archive_from_supabase corriera por list_*.
    if not pdf_path.exists() or not sidecar_path.exists():
        pdf_bytes_remote, sc_remote = _download_from_supabase(archive_id)
        if pdf_bytes_remote and sc_remote:
            # Validar permisos contra el sidecar bajado
            if viewer_email or viewer_role:
                if not _can_view(viewer_email, viewer_role, sc_remote):
                    return None
            # Persistir local para futuros accesos
            try:
                pdf_path.parent.mkdir(parents=True, exist_ok=True)
                pdf_path.write_bytes(pdf_bytes_remote)
                sidecar_path.write_text(
                    json.dumps(sc_remote, indent=2, ensure_ascii=False,
                               default=str),
                    encoding="utf-8",
                )
            except Exception:
                pass
            return pdf_bytes_remote
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
        # Ciclo 17.29 — borrar también de Supabase Storage para que
        # no resucite en el próximo cold start.
        _delete_from_supabase(archive_id)
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
    # Ciclo 17.29 — re-subir el sidecar actualizado a Supabase (el PDF
    # no cambió pero el sidecar sí; lo subimos junto para mantenerlo
    # consistente entre filesystem y bucket).
    pdf_path = ARCHIVE_ROOT / f"{archive_id}.pdf"
    if pdf_path.exists():
        try:
            _upload_to_supabase(archive_id, pdf_path.read_bytes(), sc)
        except Exception:
            pass
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
