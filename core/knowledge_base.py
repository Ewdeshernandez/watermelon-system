"""
core.knowledge_base — Repositorio de conocimiento (RAG) admin-only
==================================================================

El administrador sube cursos de vibraciones + manuales de máquina; el motor
de reportes (briefing semanal/mensual) recupera pasajes relevantes y se APOYA
en ellos para redactar con la terminología, los criterios y los umbrales de
norma correctos.

⚠️ Copyright: el material es de REFERENCIA privada. La IA fundamenta y
sintetiza; NO pega texto del manual en el reporte del cliente (eso se blinda
en el prompt de recuperación, no aquí).

Pipeline:
    PDF → extract_pdf_text → _chunk_text → embed_texts (Voyage, dim 1024)
        → insert en knowledge_docs / knowledge_chunks (Supabase pgvector)
    query → embed (input_type='query') → RPC match_knowledge → top-K chunks

Embeddings: Voyage AI (voyage-3.5-lite por defecto, 1024 dims). La llave sale
de st.secrets['voyage']['api_key'] o la env VOYAGE_API_KEY.
"""
from __future__ import annotations

import json
import logging
import os
import urllib.request
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

VOYAGE_URL = "https://api.voyageai.com/v1/embeddings"
EMBED_MODEL = "voyage-3.5-lite"   # 1024 dims, barato; subir a voyage-3.5 si hace falta
EMBED_DIM = 1024
_MAX_BATCH = 96                   # Voyage admite hasta 128 inputs/request
DOCS_TABLE = "knowledge_docs"
CHUNKS_TABLE = "knowledge_chunks"


# ---------------------------------------------------------------------------
# Llave + cliente
# ---------------------------------------------------------------------------
def _voyage_key() -> str:
    key = os.environ.get("VOYAGE_API_KEY", "")
    if not key:
        try:
            import streamlit as st
            key = (st.secrets.get("voyage", {}) or {}).get("api_key", "")
        except Exception:
            key = ""
    return (key or "").strip()


def voyage_ready() -> bool:
    return bool(_voyage_key())


def _sb():
    from core.live_readings import _get_supabase_client
    return _get_supabase_client()


# ---------------------------------------------------------------------------
# Embeddings (Voyage)
# ---------------------------------------------------------------------------
def embed_texts(texts: List[str], input_type: str = "document",
                model: str = EMBED_MODEL) -> List[List[float]]:
    """Embeddings de una lista de textos. input_type='document' para indexar,
    'query' para buscar. Devuelve lista de vectores (1024). Lanza si falla."""
    key = _voyage_key()
    if not key:
        raise RuntimeError("Falta VOYAGE_API_KEY (secrets[voyage].api_key).")
    out: List[List[float]] = []
    for i in range(0, len(texts), _MAX_BATCH):
        batch = [t if (t or "").strip() else " " for t in texts[i:i + _MAX_BATCH]]
        body = json.dumps({"input": batch, "model": model,
                           "input_type": input_type}).encode()
        req = urllib.request.Request(
            VOYAGE_URL, data=body, method="POST",
            headers={"Authorization": f"Bearer {key}",
                     "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            resp = json.load(r)
        out.extend([d["embedding"] for d in resp.get("data", [])])
    return out


# ---------------------------------------------------------------------------
# Extracción + chunking
# ---------------------------------------------------------------------------
def extract_pdf_text(pdf_bytes: bytes) -> str:
    """Texto plano de un PDF (pypdf). Vacío si no se puede leer."""
    try:
        from io import BytesIO
        try:
            from pypdf import PdfReader
        except Exception:
            from PyPDF2 import PdfReader
        reader = PdfReader(BytesIO(pdf_bytes))
        parts = []
        for pg in reader.pages:
            try:
                parts.append(pg.extract_text() or "")
            except Exception:
                continue
        return "\n".join(parts)
    except Exception as e:
        log.warning("extract_pdf_text falló: %s", e)
        return ""


def _chunk_text(text: str, target_chars: int = 2800,
                overlap: int = 300) -> List[str]:
    """Parte el texto en trozos ~target_chars respetando párrafos, con solape
    para no perder contexto en los bordes. ~700-800 tokens por chunk."""
    text = (text or "").replace("\r", "")
    if not text.strip():
        return []
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    cur = ""
    for p in paras:
        if len(cur) + len(p) + 2 <= target_chars:
            cur = f"{cur}\n\n{p}" if cur else p
        else:
            if cur:
                chunks.append(cur)
            # párrafo más grande que un chunk → cortar duro
            if len(p) > target_chars:
                for j in range(0, len(p), target_chars - overlap):
                    chunks.append(p[j:j + target_chars])
                cur = ""
            else:
                cur = p
    if cur:
        chunks.append(cur)
    # Solape entre chunks consecutivos (cola del anterior al inicio del siguiente)
    if overlap > 0 and len(chunks) > 1:
        joined = []
        for i, c in enumerate(chunks):
            if i > 0:
                tail = chunks[i - 1][-overlap:]
                c = f"{tail}\n{c}"
            joined.append(c)
        chunks = joined
    return [c for c in chunks if c.strip()]


# ---------------------------------------------------------------------------
# Ingesta / gestión
# ---------------------------------------------------------------------------
def ingest_document(title: str, pdf_bytes: bytes, *,
                    source_type: str = "manual",
                    machine_model: str = "",
                    filename: str = "",
                    uploaded_by: str = "") -> Dict[str, Any]:
    """Ingesta un PDF: extrae, trocea, embebe e inserta en la BD. Devuelve
    {ok, doc_id, n_chunks, error}."""
    out: Dict[str, Any] = {"ok": False, "doc_id": None, "n_chunks": 0, "error": ""}
    text = extract_pdf_text(pdf_bytes)
    if not text.strip():
        out["error"] = "No se pudo extraer texto del PDF (¿es escaneado sin OCR?)."
        return out
    chunks = _chunk_text(text)
    if not chunks:
        out["error"] = "El PDF no produjo texto utilizable."
        return out
    try:
        embeddings = embed_texts(chunks, input_type="document")
    except Exception as e:
        out["error"] = f"Embeddings Voyage fallaron: {e}"
        return out
    if len(embeddings) != len(chunks):
        out["error"] = "Desajuste chunks/embeddings."
        return out
    try:
        sb = _sb()
        doc = sb.table(DOCS_TABLE).insert({
            "title": title.strip() or (filename or "documento"),
            "source_type": source_type or "manual",
            "machine_model": (machine_model or "").strip() or None,
            "filename": filename or None,
            "n_chunks": len(chunks),
            "uploaded_by": uploaded_by or None,
        }).execute()
        doc_id = doc.data[0]["id"]
        rows = [{"doc_id": doc_id, "chunk_index": i,
                 "content": c, "embedding": embeddings[i]}
                for i, c in enumerate(chunks)]
        for j in range(0, len(rows), 100):
            sb.table(CHUNKS_TABLE).insert(rows[j:j + 100]).execute()
        out.update(ok=True, doc_id=doc_id, n_chunks=len(chunks))
    except Exception as e:
        log.error("ingest_document falló: %s", e)
        out["error"] = f"Inserción en BD falló: {e}"
    return out


def list_documents() -> List[Dict[str, Any]]:
    try:
        sb = _sb()
        r = (sb.table(DOCS_TABLE).select("*")
             .order("created_at", desc=True).execute())
        return r.data or []
    except Exception as e:
        log.warning("list_documents falló: %s", e)
        return []


def delete_document(doc_id: str) -> bool:
    try:
        sb = _sb()
        # chunks tienen FK ON DELETE CASCADE → basta borrar el doc
        sb.table(DOCS_TABLE).delete().eq("id", doc_id).execute()
        return True
    except Exception as e:
        log.warning("delete_document falló: %s", e)
        return False


# ---------------------------------------------------------------------------
# Recuperación (para inyectar al prompt del reporte)
# ---------------------------------------------------------------------------
def retrieve(query: str, *, filter_model: str = "",
             k: int = 6, min_similarity: float = 0.25) -> List[Dict[str, Any]]:
    """Top-K pasajes más relevantes al query. filter_model prioriza el manual
    de esa máquina (los docs sin modelo siempre son elegibles). Devuelve
    [{title, source_type, machine_model, content, similarity}]."""
    if not (query or "").strip() or not voyage_ready():
        return []
    try:
        qvec = embed_texts([query], input_type="query")[0]
    except Exception as e:
        log.warning("retrieve embed falló: %s", e)
        return []
    try:
        sb = _sb()
        r = sb.rpc("match_knowledge", {
            "query_embedding": qvec,
            "match_count": k,
            "filter_model": (filter_model or "").strip() or None,
        }).execute()
        rows = r.data or []
        return [x for x in rows if (x.get("similarity") or 0) >= min_similarity]
    except Exception as e:
        log.warning("match_knowledge falló: %s", e)
        return []


def build_reference_context(query: str, *, filter_model: str = "",
                            k: int = 6, max_chars: int = 4000) -> str:
    """Bloque de texto de REFERENCIA para inyectar al prompt de la IA. Vacío si
    no hay material relevante. Marca cada pasaje con su fuente (para citar) y
    corta a max_chars."""
    hits = retrieve(query, filter_model=filter_model, k=k)
    if not hits:
        return ""
    out, total = [], 0
    for h in hits:
        src = h.get("title") or "referencia"
        piece = f"[Fuente: {src}]\n{(h.get('content') or '').strip()}"
        if total + len(piece) > max_chars:
            break
        out.append(piece)
        total += len(piece)
    return "\n\n---\n\n".join(out)


__all__ = ["voyage_ready", "embed_texts", "extract_pdf_text",
           "ingest_document", "list_documents", "delete_document",
           "retrieve", "build_reference_context",
           "EMBED_MODEL", "EMBED_DIM"]
