"""
core.ai_qa
==========

AI Q&A sobre el archivo histórico de reportes (Ciclo 17.27).

Convierte el archivo histórico de PDFs en una base de conocimiento
consultable. El usuario hace una pregunta en lenguaje natural y el
sistema responde citando reportes específicos.

Estrategia de retrieval (sin embeddings, sin vector store):
  1. list_archived_reports devuelve los sidecars accesibles al
     viewer. Cada sidecar trae metadata rica: cliente, activo,
     fecha, severidad ejecutiva, executive_summary preview.
  2. Para muchas preguntas, los sidecars solos alcanzan (ej: "qué
     activos están en CRÍTICA" → filtramos por severity_label).
  3. Cuando se necesita más detalle, extraemos texto del PDF
     individual con pypdf (cache local TTL 30 días).
  4. Pasamos los reportes filtrados como contexto a Claude con un
     system prompt de "asistente técnico de mantenimiento".
  5. Pedimos que cite los reportes con el formato fijo
     [REPORT:{archive_id}] para que el caller pueda extraer
     citaciones precisas.

Costo estimado por consulta:
  ~$0.03-0.10 según cuántos reportes entran en el contexto.
  Con cache local de PDFs extraídos, las consultas repetidas
  son virtualmente gratis (solo el output cuenta).

API pública:
  - query_archive(question, viewer_email, viewer_role, ...) → dict
  - extract_pdf_text(pdf_path) → str (con cache TTL)
  - clear_qa_cache() → int
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.ai_diagnostic import (
    DEFAULT_MODEL,
    FALLBACK_MODEL,
    DEFAULT_TIMEOUT_S,
    is_ai_available,
    _get_client,
    _get_model_name,
    _bump_stats,
)
from core.reports_archive import (
    list_archived_reports,
    get_archived_pdf_bytes,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
QA_CACHE_DIR = DATA_DIR / "cache" / "ai_qa"
PDF_TEXT_CACHE_DIR = DATA_DIR / "cache" / "pdf_text"

DEFAULT_QA_TTL_SECONDS = 30 * 24 * 3600
QA_PROMPT_VERSION = "v1_archive_qa_2026_05"

# Costos por modelo (idénticos a ai_diagnostic.py)
_MODEL_PRICING_IN = {"sonnet": 3.0, "haiku": 1.0}
_MODEL_PRICING_OUT = {"sonnet": 15.0, "haiku": 5.0}


# =============================================================
# SYSTEM PROMPT — Asistente Técnico de Mantenimiento
# =============================================================

_SYSTEM_PROMPT_QA = """\
Eres un asistente técnico de mantenimiento predictivo Cat IV ISO
18436-2 que responde preguntas sobre el archivo histórico de reportes
de monitoreo en línea de Watermelon System. Tu rol es ayudar al
especialista, ingeniero de mantenimiento o gerente del cliente a
encontrar información en su histórico, comparar activos, detectar
patrones y obtener síntesis cualitativas.

Vas a recibir una pregunta del usuario y un conjunto de reportes
archivados (con su metadata: cliente, activo, fecha, severidad,
extracto del resumen ejecutivo). Cada reporte está etiquetado con un
archive_id único entre corchetes que debés usar para citar.

REGLAS DE RESPUESTA:

1. RESPONDÉ SOLO CON BASE EN LOS REPORTES PROPORCIONADOS.
   No inventes datos. Si la pregunta no se puede responder con la
   información disponible, decilo explícitamente y sugerí qué filtros
   o información adicional ayudaría.

2. CITÁ CADA AFIRMACIÓN CON EL FORMATO FIJO [REPORT:archive_id]
   inmediatamente después de la frase que respaldás. Múltiples
   archive_ids se separan con comas dentro del corchete:
   [REPORT:owner/2026/05/file_slug, REPORT:owner/2026/04/other_file].
   Esto permite al sistema mostrarle al usuario links directos a los
   reportes citados.

3. ESTRUCTURA DE LA RESPUESTA según el tipo de pregunta:

   - LISTADO ("muéstrame todos los activos con X"):
       Lista en bullets con cita por item. Resumí al final con
       conteo y conclusión.

   - COMPARACIÓN ("¿cuál tiene peor severidad?", "compará TES1
     vs TES3"):
       Tabla mental implícita en prosa, con cifras concretas y citas.

   - TENDENCIA HISTÓRICA ("¿cómo evolucionó X?"):
       Cronología desde el más antiguo al más reciente, con citas.

   - SÍNTESIS ("resumime X"):
       Párrafo de 60-100 palabras con la conclusión principal,
       respaldado con 1-3 citas clave.

   - PREGUNTA ABIERTA / DIAGNÓSTICA ("¿qué piensas de Y?"):
       Análisis técnico con voz pasiva, citaciones, y declaración
       explícita de cuáles aspectos NO podés evaluar con la
       información disponible.

4. TONO Y ESTILO:
   - Voz pasiva técnica: "se observa", "se concluye", "se identifica".
   - Sin emojis. Sin lenguaje coloquial.
   - Técnicamente preciso: nombres correctos de las firmas mecánicas
     (oil whip, BPFO, sub-síncrono, desbalance residual con
     amplificación resonante, etc.).
   - Citá normas con cláusula cuando sea pertinente (API 670 §4.3.2,
     ISO 20816-2, etc.).

5. EXTENSIÓN:
   - Respuestas cortas para preguntas factuales (1-3 párrafos).
   - Respuestas más largas solo cuando la pregunta lo amerite
     (síntesis comparativas, tendencias multi-año, etc.).
   - Máximo 600 palabras totales por respuesta.

6. DISCLAIMER OBLIGATORIO al final de cualquier respuesta que
   incluya recomendaciones operativas:
   "Las observaciones derivadas de este histórico requieren
   validación del especialista responsable antes de usarse para
   decisiones operativas."

7. PRIVACIDAD:
   Solo respondés sobre los reportes que te fueron proporcionados.
   Si la pregunta sugiere acceso a información fuera de ese
   conjunto, declaralo y limitá la respuesta a lo accesible.
"""


# =============================================================
# CACHE DE TEXTO DE PDFs (para evitar re-extraer en cada query)
# =============================================================

def extract_pdf_text(pdf_path: str, *, max_chars: int = 30000) -> str:
    """Extrae texto de un PDF archivado con cache local.
    Si el PDF ya fue extraído antes y la mtime no cambió, devuelve
    desde cache. Si no, extrae con pypdf y persiste."""
    p = Path(pdf_path)
    if not p.exists():
        return ""

    try:
        mtime = p.stat().st_mtime
        size = p.stat().st_size
    except Exception:
        return ""

    # Cache key: hash de path + mtime + size
    cache_key = hashlib.sha256(
        f"{p.resolve()}|{mtime}|{size}".encode("utf-8")
    ).hexdigest()[:24]
    cache_path = PDF_TEXT_CACHE_DIR / f"{cache_key}.txt"

    if cache_path.exists():
        try:
            return cache_path.read_text(encoding="utf-8", errors="replace")[:max_chars]
        except Exception:
            pass

    # Extraer con pypdf
    text_parts: List[str] = []
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(p))
        for page in reader.pages[:50]:  # Máximo 50 páginas por defensa
            try:
                text_parts.append(page.extract_text() or "")
            except Exception:
                continue
    except Exception as exc:
        print(f"[WM_AI_QA] Falló extracción PDF {p.name}: {exc}",
              file=sys.stderr, flush=True)
        return ""

    full_text = "\n".join(text_parts).strip()
    full_text = re.sub(r"\n{3,}", "\n\n", full_text)  # cleanup
    full_text = full_text[:max_chars]

    # Persistir cache
    try:
        PDF_TEXT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(full_text, encoding="utf-8")
    except Exception:
        pass

    return full_text


def clear_qa_cache() -> int:
    """Borra todos los caches de Q&A (PDFs extraídos + respuestas).
    Devuelve cantidad eliminada."""
    n = 0
    for cache_dir in (QA_CACHE_DIR, PDF_TEXT_CACHE_DIR):
        if not cache_dir.exists():
            continue
        for pth in cache_dir.glob("*"):
            try:
                pth.unlink()
                n += 1
            except Exception:
                pass
    return n


# =============================================================
# CONSTRUCCIÓN DEL CONTEXTO PARA CLAUDE
# =============================================================

def _summarize_report_for_qa(
    sidecar: Dict[str, Any],
    *,
    include_pdf_text: bool = False,
    pdf_text_max_chars: int = 4000,
) -> str:
    """Convierte un sidecar archivado en un bloque de texto que el AI
    puede usar como contexto de un solo reporte."""
    rm = sidecar.get("report_meta", {}) or {}
    archive_id = sidecar.get("archive_id", "")
    archived_at = sidecar.get("archived_at", "")[:10]

    lines: List[str] = [f"### Reporte [REPORT:{archive_id}]"]
    if rm.get("consecutive"):
        lines.append(f"- Consecutivo: {rm.get('consecutive')}")
    if archived_at:
        lines.append(f"- Archivado: {archived_at}")
    if rm.get("report_date"):
        lines.append(f"- Fecha del análisis: {rm.get('report_date')}")
    if rm.get("client"):
        lines.append(f"- Cliente: {rm.get('client')}")
    if rm.get("site"):
        lines.append(f"- Sitio: {rm.get('site')}")
    asset_blob = " · ".join(filter(None, [
        rm.get("asset_class", ""),
        rm.get("asset_model", ""),
        rm.get("instance_tag", ""),
    ]))
    if asset_blob:
        lines.append(f"- Activo: {asset_blob}")
    if rm.get("train_description"):
        lines.append(f"- Tren: {rm.get('train_description')}")
    if rm.get("executive_severity"):
        lines.append(f"- Severidad ejecutiva: {rm.get('executive_severity')}")
    if rm.get("prepared_by"):
        lines.append(f"- Preparado por: {rm.get('prepared_by')}")
    exec_sum = (rm.get("executive_summary") or "").strip()
    if exec_sum:
        lines.append("")
        lines.append("**Resumen ejecutivo del reporte:**")
        lines.append(exec_sum[:1500])

    if include_pdf_text:
        pdf_path = sidecar.get("_pdf_path", "")
        if pdf_path:
            text = extract_pdf_text(pdf_path, max_chars=pdf_text_max_chars)
            if text:
                lines.append("")
                lines.append("**Contenido del PDF (extracto):**")
                lines.append(text[:pdf_text_max_chars])

    return "\n".join(lines)


def _build_qa_user_message(
    question: str,
    sidecars: List[Dict[str, Any]],
    *,
    include_pdf_text: bool = False,
) -> str:
    """Compone el user message: pregunta + N reportes resumidos."""
    parts: List[str] = []
    parts.append("# Pregunta del usuario")
    parts.append(question.strip())
    parts.append("")
    parts.append(f"# Archivo accesible al usuario ({len(sidecars)} reportes)")
    parts.append("")
    for sc in sidecars:
        parts.append(_summarize_report_for_qa(
            sc, include_pdf_text=include_pdf_text
        ))
        parts.append("")
        parts.append("---")
        parts.append("")
    parts.append("Por favor, respondé la pregunta basándote ÚNICAMENTE en estos "
                 "reportes. Citá cada afirmación con [REPORT:archive_id] "
                 "inmediatamente después de la frase que respaldás.")
    return "\n".join(parts)


# =============================================================
# DETECCIÓN DE FILTROS HEURÍSTICOS DESDE LA PREGUNTA
# =============================================================

_SEVERITY_KEYWORDS = {
    "crítica": "CRÍTICA",
    "critica": "CRÍTICA",
    "critico": "CRÍTICA",
    "crítico": "CRÍTICA",
    "acción requerida": "ACCIÓN REQUERIDA",
    "accion requerida": "ACCIÓN REQUERIDA",
    "atención": "ATENCIÓN",
    "atencion": "ATENCIÓN",
    "vigilancia": "VIGILANCIA",
    "aceptable": "CONDICIÓN ACEPTABLE",
}


def _infer_filters_from_question(question: str) -> Dict[str, str]:
    """Heurísticas baratas para detectar filtros en la pregunta del
    usuario. Esto reduce el contexto enviado a Claude (solo los
    reportes que potencialmente importan)."""
    q = (question or "").lower()
    filters: Dict[str, str] = {}

    # Cliente nombrado explícitamente (ej "para ECOPETROL")
    for client_kw in ("ecopetrol", "magnex", "pdvsa", "promigas", "argos",
                      "isa", "epm", "celsia", "termocol"):
        if client_kw in q:
            filters["client_filter"] = client_kw
            break

    # Año en la pregunta
    year_m = re.search(r"\b(20\d{2})\b", q)
    if year_m:
        year = year_m.group(1)
        filters["date_from"] = f"{year}-01-01"
        filters["date_to"] = f"{year}-12-31"

    # Frases típicas de rango temporal
    if "último mes" in q or "ultimo mes" in q or "30 días" in q or "30 dias" in q:
        from datetime import timedelta
        d_to = datetime.now()
        d_from = d_to - timedelta(days=30)
        filters["date_from"] = d_from.strftime("%Y-%m-%d")
        filters["date_to"] = d_to.strftime("%Y-%m-%d")
    elif "últimos 6 meses" in q or "ultimos 6 meses" in q or "6 meses" in q:
        from datetime import timedelta
        d_to = datetime.now()
        d_from = d_to - timedelta(days=180)
        filters["date_from"] = d_from.strftime("%Y-%m-%d")
        filters["date_to"] = d_to.strftime("%Y-%m-%d")
    elif "este año" in q or "este ano" in q:
        filters["date_from"] = f"{datetime.now().year}-01-01"

    return filters


def _parse_citations(
    answer_md: str,
    available_archive_ids: List[str],
) -> List[str]:
    """Extrae los archive_ids citados con [REPORT:...] del markdown
    de respuesta. Solo devuelve los que efectivamente existen en el
    conjunto que pasamos a Claude (filtra alucinaciones de IDs)."""
    cited: List[str] = []
    seen: set = set()
    available_set = set(available_archive_ids)

    for m in re.finditer(r"\[REPORT:\s*([^\]]+?)\s*\]", answer_md):
        raw = m.group(1)
        for token in raw.split(","):
            token_clean = token.strip().lstrip("REPORT:").strip()
            if token_clean and token_clean in available_set and token_clean not in seen:
                cited.append(token_clean)
                seen.add(token_clean)

    return cited


# =============================================================
# RETRY HELPER (heredado de ai_diagnostic.py)
# =============================================================

def _is_retryable_exc(exc: Exception) -> Tuple[bool, Optional[int]]:
    err_str = str(exc); err_low = err_str.lower()
    exc_type_name = type(exc).__name__
    status_code = getattr(exc, "status_code", None)
    _retryable = (429, 502, 503, 529)
    if status_code is None:
        for code in _retryable:
            if (f"Error code: {code}" in err_str
                    or f"'{code}'" in err_str
                    or f'"{code}"' in err_str):
                status_code = code; break
    if status_code in _retryable:
        return (True, status_code)
    if any(s in exc_type_name for s in (
            "APITimeoutError", "APIConnectionError", "TimeoutException",
            "ConnectTimeout", "ReadTimeout", "ConnectionError")):
        return (True, 408)
    if "timed out" in err_low or "timeout" in err_low or "interrupted" in err_low:
        return (True, 408)
    return (False, status_code)


# =============================================================
# API PÚBLICA — query_archive
# =============================================================

def query_archive(
    question: str,
    *,
    viewer_email: str,
    viewer_role: str,
    use_pdf_text: bool = False,
    max_reports_in_context: int = 30,
    use_cache: bool = True,
    cache_ttl_seconds: int = DEFAULT_QA_TTL_SECONDS,
    max_tokens: int = 2048,
) -> Dict[str, Any]:
    """Hace una pregunta sobre el archivo histórico al usuario.

    Args:
        question: pregunta en lenguaje natural.
        viewer_email / viewer_role: identidad del que consulta (define
            qué reportes están accesibles según los permisos de
            reports_archive.list_archived_reports).
        use_pdf_text: si True, extrae texto de cada PDF y lo incluye
            en el contexto. Más caro pero responde preguntas
            profundas. Por default False — la metadata del sidecar
            (incluyendo executive_summary) alcanza para ~80% de las
            preguntas.
        max_reports_in_context: límite de reportes incluidos en el
            contexto (por presupuesto de tokens). Si hay más, se
            ordenan por fecha desc y se cortan los más viejos.
        use_cache: si True, intenta servir respuestas cacheadas.
        cache_ttl_seconds: TTL del cache de respuestas.
        max_tokens: límite de output.

    Returns:
        {
            "ok": bool,
            "markdown": str,
            "reports_referenced": [        # lista de archive_ids citados
                {"archive_id": str, "consecutive": str, "client": str,
                 "asset": str, "date": str, "severity": str}
            ],
            "n_reports_in_context": int,
            "n_reports_in_archive": int,
            "model": str,
            "cached": bool,
            "fallback_used": bool,
            "input_tokens": int,
            "output_tokens": int,
            "cost_usd": float,
            "error": str,
            "generated_at": str,
        }
    """
    if not question or not question.strip():
        return _empty_qa_response("La pregunta está vacía.")

    if not is_ai_available():
        return _empty_qa_response(
            "_⚠️ AI no disponible — falta configurar `[anthropic] api_key` en "
            "los secrets de Streamlit, o no está instalado el paquete "
            "`anthropic`._"
        )

    # 1) Inferir filtros baratos de la pregunta
    filters = _infer_filters_from_question(question)

    # 2) Listar reportes accesibles con filtros aplicados
    sidecars_all = list_archived_reports(
        viewer_email=viewer_email,
        viewer_role=viewer_role,
        client_filter=filters.get("client_filter", ""),
        date_from=filters.get("date_from", ""),
        date_to=filters.get("date_to", ""),
        limit=200,
    )
    n_archive = len(sidecars_all)

    if not sidecars_all:
        return _empty_qa_response(
            f"_No se encontraron reportes archivados accesibles para tu "
            f"rol y filtros derivados de la pregunta. "
            f"Filtros aplicados: {filters or '(ninguno)'}_",
            n_archive=0,
        )

    # 3) Limitar al N más reciente para no explotar el contexto
    sidecars = sidecars_all[:max_reports_in_context]
    available_archive_ids = [sc.get("archive_id", "") for sc in sidecars]

    # 4) Cache HIT por hash de question + ids + version
    h = hashlib.sha256(
        json.dumps({
            "v": QA_PROMPT_VERSION,
            "q": question.strip().lower(),
            "ids": sorted(available_archive_ids),
            "viewer": viewer_email.lower() if viewer_email else "",
            "use_pdf": use_pdf_text,
        }, sort_keys=True).encode("utf-8")
    ).hexdigest()[:32]
    cache_path = QA_CACHE_DIR / f"{h}.json"

    if use_cache and cache_path.exists():
        try:
            age = time.time() - cache_path.stat().st_mtime
            if age <= cache_ttl_seconds:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                _bump_stats(0, 0, cached.get("model", ""), cached=True)
                return {**cached, "cached": True}
        except Exception:
            pass

    # 5) Llamada a Claude con retry + fallback
    client = _get_client()
    if client is None:
        return _empty_qa_response("_⚠️ No se pudo inicializar el cliente._")

    user_msg = _build_qa_user_message(
        question, sidecars, include_pdf_text=use_pdf_text
    )
    primary_model = _get_model_name()

    def _try(model_name: str, label: str) -> Tuple[Any, Optional[Exception]]:
        last = None
        for attempt in range(3):
            try:
                print(
                    f"[WM_AI_QA] CALL · {label} · model={model_name} · "
                    f"n_reports={len(sidecars)} · attempt={attempt + 1}/3 · "
                    f"hash={h[:8]}",
                    file=sys.stderr, flush=True,
                )
                resp = client.messages.create(
                    model=model_name,
                    max_tokens=max_tokens,
                    system=_SYSTEM_PROMPT_QA,
                    messages=[{"role": "user", "content": user_msg}],
                )
                return resp, None
            except Exception as exc:
                last = exc
                is_retry, st_code = _is_retryable_exc(exc)
                if is_retry and attempt < 2:
                    wait_s = 2 ** attempt
                    print(
                        f"[WM_AI_QA] RETRY · {label} · status={st_code} · "
                        f"wait={wait_s}s",
                        file=sys.stderr, flush=True,
                    )
                    time.sleep(wait_s)
                    continue
                print(
                    f"[WM_AI_QA] FAIL · {label} · {str(exc)[:200]}",
                    file=sys.stderr, flush=True,
                )
                return None, exc
        return None, last

    response, last_exc = _try(primary_model, "primary")
    fallback_used = False
    used_model = primary_model

    if response is None and last_exc is not None:
        is_retry, _ = _is_retryable_exc(last_exc)
        is_already_fallback = primary_model.startswith("claude-haiku")
        if is_retry and not is_already_fallback:
            print(
                f"[WM_AI_QA] FALLBACK · {primary_model} agotado, "
                f"intentando con {FALLBACK_MODEL}",
                file=sys.stderr, flush=True,
            )
            response, last_exc = _try(FALLBACK_MODEL, "fallback-haiku")
            if response is not None:
                fallback_used = True
                used_model = FALLBACK_MODEL

    if response is None:
        err = str(last_exc) if last_exc else "unknown"
        if "overloaded" in err.lower() or "529" in err:
            msg = ("_⚠️ Servidores Claude sobrecargados. Esperá 5-10 min y "
                   "reintentá._")
        elif "timeout" in err.lower() or "timed out" in err.lower():
            msg = "_⚠️ Timeout de conexión. Verificá tu red y reintentá._"
        else:
            msg = f"_⚠️ Error consultando archivo:_\n\n```\n{err}\n```"
        return _empty_qa_response(
            msg, n_archive=n_archive, error=err[:500],
            fallback_used=fallback_used,
        )

    # 6) Extraer respuesta y citaciones
    try:
        markdown = response.content[0].text
    except Exception:
        markdown = "_(no text in response)_"
    in_tok = getattr(response.usage, "input_tokens", 0) if response.usage else 0
    out_tok = getattr(response.usage, "output_tokens", 0) if response.usage else 0

    cited_ids = _parse_citations(markdown, available_archive_ids)
    reports_referenced: List[Dict[str, str]] = []
    sidecars_by_id = {sc.get("archive_id", ""): sc for sc in sidecars}
    for aid in cited_ids:
        sc = sidecars_by_id.get(aid)
        if not sc:
            continue
        rm = sc.get("report_meta", {}) or {}
        reports_referenced.append({
            "archive_id": aid,
            "consecutive": rm.get("consecutive", ""),
            "client": rm.get("client", ""),
            "asset": " · ".join(filter(None, [
                rm.get("asset_class", ""),
                rm.get("instance_tag", ""),
            ])),
            "date": sc.get("archived_at", "")[:10],
            "severity": rm.get("executive_severity", ""),
        })

    # 7) Pricing dinámico
    if used_model.startswith("claude-haiku"):
        in_p, out_p = _MODEL_PRICING_IN["haiku"], _MODEL_PRICING_OUT["haiku"]
    else:
        in_p, out_p = _MODEL_PRICING_IN["sonnet"], _MODEL_PRICING_OUT["sonnet"]
    cost_usd = (in_tok * in_p + out_tok * out_p) / 1_000_000

    print(
        f"[WM_AI_QA] OK · model={used_model} · in={in_tok} · out={out_tok} · "
        f"~${cost_usd:.4f} · n_cited={len(reports_referenced)}"
        f"{' · FALLBACK' if fallback_used else ''}",
        file=sys.stderr, flush=True,
    )

    result = {
        "ok": True,
        "markdown": markdown,
        "reports_referenced": reports_referenced,
        "n_reports_in_context": len(sidecars),
        "n_reports_in_archive": n_archive,
        "model": used_model,
        "cached": False,
        "fallback_used": fallback_used,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "cost_usd": round(cost_usd, 5),
        "error": "",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }

    # 8) Persistir cache
    if use_cache:
        try:
            QA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps(result, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:
            pass
    _bump_stats(in_tok, out_tok, used_model, cached=False)

    return result


def _empty_qa_response(
    md: str,
    *,
    n_archive: int = 0,
    error: str = "",
    fallback_used: bool = False,
) -> Dict[str, Any]:
    """Construye un dict de respuesta vacío con el markdown dado."""
    return {
        "ok": False,
        "markdown": md,
        "reports_referenced": [],
        "n_reports_in_context": 0,
        "n_reports_in_archive": n_archive,
        "model": "",
        "cached": False,
        "fallback_used": fallback_used,
        "input_tokens": 0,
        "output_tokens": 0,
        "cost_usd": 0.0,
        "error": error,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }


__all__ = [
    "query_archive",
    "extract_pdf_text",
    "clear_qa_cache",
]
