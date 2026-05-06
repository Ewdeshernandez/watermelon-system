"""
core.ai_patterns
================

Pattern Memory — memoria institucional con AI (Ciclo 17.34).

Cuando se prepara un reporte nuevo, el sistema busca automáticamente
en el archivo histórico patrones mecánicos similares (oil whip,
defectos de rodamiento etapa II-III, desbalance con resonancia, etc.)
en CUALQUIER activo del cliente, no solo el mismo. El archivo deja
de ser un repositorio pasivo de PDFs y se convierte en un cerebro
colectivo: cada reporte que se archiva suma valor a TODOS los
próximos análisis.

Estrategia de matching:
  En lugar de embeddings (Voyage AI, OpenAI, etc.), usamos a Claude
  Sonnet 4.5 como matcher semántico directo. Ventajas:
    - Cero infraestructura nueva (sin vector store, sin segunda key)
    - Claude entiende contexto técnico Cat IV nativamente — mejor
      que la similitud coseno de embeddings genéricos
    - Explica POR QUÉ son similares ("ambos comparten firma
      sub-síncrona a 0.949X + Q-factor elevado") — embeddings dan
      número pero no narrativa
    - Para escala de cientos de reportes cabe en contexto de Sonnet
      4.5 (200K tokens). Si la escala crece a miles, agregamos
      embeddings como pre-filtro en una iteración futura.

API pública:
  - compute_fingerprint(sidecar) → texto estructurado con la huella
    mecánica del reporte (asset, severidad, firmas, summary).
  - find_similar_patterns(current_meta, current_items, viewer_email,
    viewer_role, ...) → dict con lista de matches + score + por qué
    + resolución previa si está documentada + tokens/costo.
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
    is_ai_available,
    _get_client,
    _get_model_name,
    _bump_stats,
)
from core.ai_qa import _is_retryable_exc
from core.reports_archive import list_archived_reports


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PATTERNS_CACHE_DIR = DATA_DIR / "cache" / "ai_patterns"

DEFAULT_PATTERNS_TTL_SECONDS = 14 * 24 * 3600  # 14 días
PATTERNS_PROMPT_VERSION = "v1_institutional_memory_2026_05"

DEFAULT_TOP_K = 5
MAX_HISTORY_IN_CONTEXT = 60  # presupuesto de tokens

# Categorías cualitativas de score (para badges en UI)
SIMILARITY_BANDS = [
    (85, "Muy alta", "#dc2626"),    # rojo intenso
    (70, "Alta", "#ea580c"),         # naranja
    (55, "Moderada", "#f59e0b"),     # amarillo
    (40, "Baja", "#84cc16"),         # verde-lima
    (0,  "Marginal", "#94a3b8"),     # gris
]


def similarity_band(score: float) -> Tuple[str, str]:
    """Devuelve (label, hex_color) para un score 0-100."""
    for threshold, label, color in SIMILARITY_BANDS:
        if score >= threshold:
            return label, color
    return "Marginal", "#94a3b8"


# =============================================================
# SYSTEM PROMPT — Matcher Institucional de Patrones
# =============================================================

_SYSTEM_PROMPT_PATTERNS = """\
Eres un especialista de análisis de vibraciones Cat IV ISO 18436-2
con MEMORIA INSTITUCIONAL. Has revisado cientos de reportes técnicos
del archivo histórico de un programa de mantenimiento predictivo y
podés reconocer cuándo un patrón mecánico actual se parece a casos
previos (de cualquier activo, no solo del mismo) que ya pasaron por
el archivo. Tu rol específico: hacer pattern matching SEMÁNTICO entre
un reporte ACTUAL en preparación y N reportes archivados, devolviendo
los TOP K más similares con score cualitativo, explicación del por
qué, y la resolución documentada si la hay.

Vas a recibir:
  - Una FINGERPRINT del reporte actual (huella mecánica): asset,
    severidad, firmas detectadas, frecuencias clave, executive
    summary preliminar.
  - Una lista de N FINGERPRINTS de reportes históricos archivados,
    cada uno con su archive_id único y los mismos campos.

Tu tarea: identificar los TOP K (default 5) reportes históricos más
similares al actual desde el punto de vista MECÁNICO. La similitud
mecánica se basa en:
  - Tipo de firma dominante (oil whip, BPFO, desbalance, misalignment,
    rub, soltura, defecto de rodamiento etapa I/II/III).
  - Frecuencias características compartidas (sub-sincrónicas,
    1X dominante, BPFx, harmónicos altos).
  - Patrón de evolución (estable, monotónico ascendente, errático).
  - Combinaciones de causas raíz (ej: desbalance + resonancia en
    ambos casos).

NO importa si son del mismo activo, mismo cliente, o misma fecha —
SOLO importa el patrón mecánico. Un caso de TES1 de hace 18 meses
puede ser muy similar mecánicamente a un BL3 actual.

ESTRUCTURA OBLIGATORIA DE LA RESPUESTA — JSON estricto, sin markdown,
sin bloque de código, sin texto antes o después. Solo el JSON:

{
  "matches": [
    {
      "archive_id": "owner/2026/04/file_slug",
      "similarity_score": 87,
      "rationale": "Texto técnico de 1-3 frases explicando QUÉ firmas o patrones comparten ambos casos. Voz pasiva técnica. Sin emojis.",
      "resolution_summary": "Resolución del caso histórico si está documentada en el executive_summary. Si no hay resolución registrada, valor: '(resolución no documentada en el archivo)'.",
      "applicability": "Frase corta sobre qué tan aplicable es la resolución del caso histórico al caso actual."
    }
  ],
  "global_assessment": "Frase única evaluando el conjunto: si hay alta similitud (algún match >70%), declara 'Patrón con antecedentes relevantes en archivo'. Si no hay match >55%, declara 'Patrón sin antecedentes mecánicos similares en el archivo accesible'. Tono técnico."
}

REGLAS:

- Devolvé SOLO los TOP K matches (default 5, o menos si hay menos
  reportes archivados o si ninguno supera score 40%).
- Los scores son CUALITATIVOS basados en tu juicio experto, no
  cálculo matemático. Usá la escala:
    - 85-100: Muy alta similitud (mismas firmas, mismas frecuencias)
    - 70-84: Alta similitud (mismas firmas, distintos detalles)
    - 55-69: Moderada (algunos rasgos compartidos)
    - 40-54: Baja (algún elemento en común)
    - <40: Marginal (no incluir en la lista)
- Voz pasiva técnica. Sin emojis. Sin caracteres ornamentales.
- Si los datos del archive son escasos (n_history < 5), declaralo
  en global_assessment y emití menos matches con honestidad.
- Si la información de resolución del executive_summary del caso
  histórico es vaga, decilo: "resolución no documentada".
- NO inventes archive_ids. Solo cita los que vinieron en el input.
- El JSON debe ser estrictamente válido (parseable por json.loads).
"""


# =============================================================
# FINGERPRINT — extracción de huella mecánica
# =============================================================

_AI_BLOCK_REGEX = re.compile(
    r"<<<WM_AI_BLOCK>>>(.*?)<<<WM_AI_NARRATIVE>>>",
    re.DOTALL,
)


def _extract_quant_summary_from_notes(notes: str) -> str:
    """De las notas de un report_item, extrae el bloque cuantitativo
    formateado tipo 'Velocidad: 3601 RPM; Overall: 2.27 mil pp; ...'.
    Si no hay marcador, devuelve string vacío."""
    if not notes:
        return ""
    m = _AI_BLOCK_REGEX.search(notes)
    if not m:
        return ""
    quant_text = m.group(1).strip()
    pieces: List[str] = []
    for line in quant_text.splitlines():
        if "|" in line:
            cells = [c.strip() for c in line.split("|")]
            if len(cells) >= 2 and cells[0] and cells[0] != "Parámetro":
                pieces.append(f"{cells[0]}: {cells[1]}")
    return "; ".join(pieces)


def compute_fingerprint(sidecar: Dict[str, Any]) -> str:
    """Genera la huella mecánica de un reporte archivado en formato
    de texto estructurado para que Claude pueda compararlo."""
    if not sidecar:
        return ""
    rm = sidecar.get("report_meta", {}) or {}
    archive_id = sidecar.get("archive_id", "")
    archived_at = sidecar.get("archived_at", "")[:10]

    lines: List[str] = [f"archive_id: {archive_id}"]
    if archived_at:
        lines.append(f"archived_at: {archived_at}")
    asset_blob = " · ".join(filter(None, [
        rm.get("asset_class", ""),
        rm.get("asset_model", ""),
        rm.get("instance_tag", ""),
    ]))
    if asset_blob:
        lines.append(f"asset: {asset_blob}")
    if rm.get("client"):
        lines.append(f"client: {rm.get('client')}")
    sev = (rm.get("executive_severity") or "").strip()
    if sev:
        lines.append(f"severity: {sev}")
    exec_sum = (rm.get("executive_summary") or "").strip()
    if exec_sum:
        lines.append(f"executive_summary: {exec_sum[:1200]}")
    return "\n".join(lines)


def compute_fingerprint_for_current(
    current_meta: Dict[str, Any],
    current_items: List[Dict[str, Any]],
) -> str:
    """Genera la huella mecánica del reporte EN PREPARACIÓN. Usa el
    meta del reporte + las tablas cuantitativas + las notas de cada
    item (que contienen la narrativa Cat IV o AI cuando existe)."""
    lines: List[str] = ["archive_id: (current — in preparation)"]
    asset_blob = " · ".join(filter(None, [
        current_meta.get("asset_class", ""),
        current_meta.get("asset_model", ""),
        current_meta.get("instance_tag", ""),
    ]))
    if asset_blob:
        lines.append(f"asset: {asset_blob}")
    if current_meta.get("client"):
        lines.append(f"client: {current_meta.get('client')}")
    sev_curr = (current_meta.get("executive_severity") or "").strip()
    if sev_curr:
        lines.append(f"severity: {sev_curr}")

    # Concatenar evidencia cuantitativa de las figuras del reporte
    quant_pieces: List[str] = []
    narrative_pieces: List[str] = []
    for item in current_items[:8]:  # techo defensivo
        i_type = str(item.get("type", "") or "")
        notes = str(item.get("notes", "") or "")
        quant = _extract_quant_summary_from_notes(notes)
        if quant:
            quant_pieces.append(f"[{i_type}] {quant}")
        # Tomar también las primeras 600 chars de la narrativa para
        # capturar firmas mecánicas detectadas
        if "<<<WM_AI_NARRATIVE>>>" in notes:
            tail = notes.split("<<<WM_AI_NARRATIVE>>>", 1)[1]
            narrative_pieces.append(f"[{i_type}] {tail.strip()[:600]}")
        else:
            narrative_pieces.append(f"[{i_type}] {notes.strip()[:600]}")

    if quant_pieces:
        lines.append("quantitative_evidence:")
        for q in quant_pieces:
            lines.append(f"  - {q}")
    if narrative_pieces:
        lines.append("findings_narrative:")
        for n in narrative_pieces:
            lines.append(f"  - {n[:600]}")
    return "\n".join(lines)


# =============================================================
# BUILDER DEL USER MESSAGE
# =============================================================

def _build_patterns_user_message(
    current_fp: str,
    archive_fingerprints: List[str],
    top_k: int,
) -> str:
    parts: List[str] = []
    parts.append("# Reporte ACTUAL en preparación (huella mecánica)")
    parts.append("")
    parts.append(current_fp)
    parts.append("")
    parts.append("---")
    parts.append("")
    parts.append(
        f"# Archivo histórico accesible — {len(archive_fingerprints)} reportes"
    )
    parts.append("")
    for i, fp in enumerate(archive_fingerprints, 1):
        parts.append(f"## Histórico #{i}")
        parts.append(fp)
        parts.append("")
    parts.append("---")
    parts.append("")
    parts.append(
        f"Devolvé los TOP {top_k} matches del archivo histórico más "
        f"similares MECÁNICAMENTE al reporte actual, en formato JSON "
        f"estricto según la estructura definida. Si ninguno supera "
        f"score 40%, devolvé lista vacía y declaralo en "
        f"global_assessment."
    )
    return "\n".join(parts)


def _patterns_payload_hash(
    current_fp: str,
    archive_ids: List[str],
    top_k: int,
) -> str:
    blob = {
        "version": PATTERNS_PROMPT_VERSION,
        "model": _get_model_name(),
        "current_fp_hash": hashlib.sha256(
            current_fp.encode("utf-8")
        ).hexdigest()[:16],
        "archive_ids_sorted": sorted(archive_ids),
        "top_k": top_k,
    }
    j = json.dumps(blob, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(j.encode("utf-8")).hexdigest()[:32]


# =============================================================
# API PÚBLICA — find_similar_patterns
# =============================================================

def find_similar_patterns(
    current_meta: Dict[str, Any],
    current_items: List[Dict[str, Any]],
    *,
    viewer_email: str,
    viewer_role: str,
    top_k: int = DEFAULT_TOP_K,
    max_history_in_context: int = MAX_HISTORY_IN_CONTEXT,
    use_cache: bool = True,
    cache_ttl_seconds: int = DEFAULT_PATTERNS_TTL_SECONDS,
    max_tokens: int = 2048,
) -> Dict[str, Any]:
    """Busca patrones mecánicos similares en el archivo histórico al
    reporte que se está preparando.

    Args:
        current_meta: meta del reporte actual.
        current_items: figuras del reporte actual (sus notas tienen
            las tablas cuantitativas + narrativa Cat IV/AI).
        viewer_email/role: identidad del que consulta (define qué
            archivos son accesibles según permisos).
        top_k: cuántos matches devolver (default 5).
        max_history_in_context: techo de reportes archivados a meter
            en el prompt (default 60). Si hay más, se ordenan por
            recencia y se cortan los más viejos.

    Returns:
        {
            "ok": bool,
            "matches": [
                {
                    "archive_id": str,
                    "similarity_score": int (0-100),
                    "similarity_band": str,
                    "similarity_color": str (hex),
                    "rationale": str,
                    "resolution_summary": str,
                    "applicability": str,
                    # Enriquecido desde el sidecar:
                    "consecutive": str,
                    "client": str,
                    "asset": str,
                    "date": str,
                    "severity": str,
                }
            ],
            "global_assessment": str,
            "n_history_searched": int,
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
    if not is_ai_available():
        return _empty_patterns(
            "_AI no disponible — falta configurar `[anthropic] api_key`._"
        )
    if not current_items:
        return _empty_patterns(
            "_El reporte actual no tiene figuras para identificar patrones._"
        )

    # 1) Listar todo el archivo accesible
    sidecars_all = list_archived_reports(
        viewer_email=viewer_email,
        viewer_role=viewer_role,
        limit=500,
    )
    if not sidecars_all:
        return _empty_patterns(
            "_El archivo histórico está vacío. Pattern matching requiere "
            "al menos un reporte archivado para comparar._",
            n_history_searched=0,
        )

    # 2) Cortar al N más reciente para presupuesto de tokens
    sidecars = sidecars_all[:max_history_in_context]
    n_archive = len(sidecars)

    # 3) Construir fingerprints
    current_fp = compute_fingerprint_for_current(current_meta, current_items)
    archive_fps: List[str] = []
    sidecars_by_id: Dict[str, Dict[str, Any]] = {}
    for sc in sidecars:
        fp = compute_fingerprint(sc)
        if fp:
            archive_fps.append(fp)
            aid = sc.get("archive_id", "")
            if aid:
                sidecars_by_id[aid] = sc

    if not archive_fps:
        return _empty_patterns(
            "_No se pudieron construir huellas mecánicas de los "
            "reportes archivados (sidecars vacíos o corruptos)._",
            n_history_searched=0,
        )

    # 4) Cache HIT
    archive_ids_for_hash = [
        sc.get("archive_id", "") for sc in sidecars
    ]
    h = _patterns_payload_hash(current_fp, archive_ids_for_hash, top_k)
    cache_path = PATTERNS_CACHE_DIR / f"{h}.json"
    if use_cache and cache_path.exists():
        try:
            age = time.time() - cache_path.stat().st_mtime
            if age <= cache_ttl_seconds:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                _bump_stats(0, 0, cached.get("model", ""), cached=True)
                return {**cached, "cached": True}
        except Exception:
            pass

    # 5) Llamada con retry + fallback
    client = _get_client()
    if client is None:
        return _empty_patterns(
            "_No se pudo inicializar el cliente Claude._"
        )

    user_msg = _build_patterns_user_message(current_fp, archive_fps, top_k)
    primary_model = _get_model_name()

    def _try(model_name: str, label: str):
        last = None
        for attempt in range(3):
            try:
                print(
                    f"[WM_AI_PATTERNS] CALL · {label} · model={model_name} · "
                    f"n_history={n_archive} · top_k={top_k} · "
                    f"attempt={attempt + 1}/3 · hash={h[:8]}",
                    file=sys.stderr, flush=True,
                )
                resp = client.messages.create(
                    model=model_name,
                    max_tokens=max_tokens,
                    system=_SYSTEM_PROMPT_PATTERNS,
                    messages=[{"role": "user", "content": user_msg}],
                )
                return resp, None
            except Exception as exc:
                last = exc
                is_retry, st_code = _is_retryable_exc(exc)
                if is_retry and attempt < 2:
                    wait_s = 2 ** attempt
                    print(
                        f"[WM_AI_PATTERNS] RETRY · {label} · "
                        f"status={st_code} · wait={wait_s}s",
                        file=sys.stderr, flush=True,
                    )
                    time.sleep(wait_s)
                    continue
                print(
                    f"[WM_AI_PATTERNS] FAIL · {label} · {str(exc)[:200]}",
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
                f"[WM_AI_PATTERNS] FALLBACK · {primary_model} agotado, "
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
            msg = ("_Servidores Claude sobrecargados. Esperá 5-10 min "
                   "y reintentá._")
        elif "timeout" in err.lower() or "timed out" in err.lower():
            msg = "_Timeout de conexión. Verificá tu red y reintentá._"
        else:
            msg = f"_Error buscando patrones similares:_\n\n```\n{err}\n```"
        return _empty_patterns(
            msg, error=err[:500], fallback_used=fallback_used,
            n_history_searched=n_archive,
        )

    # 6) Parsear el JSON devuelto por Claude
    try:
        raw_text = response.content[0].text
    except Exception:
        raw_text = ""
    parsed = _parse_patterns_json(raw_text, sidecars_by_id)
    if parsed is None:
        # Como fallback defensivo: devolver el texto crudo en
        # global_assessment para que el usuario al menos vea algo.
        parsed = {
            "matches": [],
            "global_assessment": (
                "El modelo no devolvió un JSON parseable. Texto crudo:\n\n"
                + raw_text[:1000]
            ),
        }

    in_tok = getattr(response.usage, "input_tokens", 0) if response.usage else 0
    out_tok = getattr(response.usage, "output_tokens", 0) if response.usage else 0

    if used_model.startswith("claude-haiku"):
        in_p, out_p = 1.0, 5.0
    else:
        in_p, out_p = 3.0, 15.0
    cost_usd = (in_tok * in_p + out_tok * out_p) / 1_000_000

    print(
        f"[WM_AI_PATTERNS] OK · model={used_model} · in={in_tok} · "
        f"out={out_tok} · ~${cost_usd:.4f} · "
        f"matches={len(parsed.get('matches', []))}"
        f"{' · FALLBACK' if fallback_used else ''}",
        file=sys.stderr, flush=True,
    )

    result = {
        "ok": True,
        "matches": parsed.get("matches", []),
        "global_assessment": parsed.get("global_assessment", ""),
        "n_history_searched": n_archive,
        "model": used_model,
        "cached": False,
        "fallback_used": fallback_used,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "cost_usd": round(cost_usd, 5),
        "error": "",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }

    if use_cache:
        try:
            PATTERNS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps(result, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:
            pass
    _bump_stats(in_tok, out_tok, used_model, cached=False)

    return result


def _parse_patterns_json(
    raw_text: str,
    sidecars_by_id: Dict[str, Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Parsea el JSON devuelto por Claude. Tolera bloques de código
    markdown alrededor (```json ... ```). Enriquece cada match con
    metadata del sidecar correspondiente."""
    if not raw_text:
        return None
    text = raw_text.strip()
    # Limpiar bloque de código markdown si lo hay
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
    try:
        parsed = json.loads(text)
    except Exception:
        # Fallback: buscar el primer { ... } balanceado
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return None
        try:
            parsed = json.loads(m.group(0))
        except Exception:
            return None

    matches = parsed.get("matches", []) or []
    enriched: List[Dict[str, Any]] = []
    for m in matches:
        if not isinstance(m, dict):
            continue
        aid = (m.get("archive_id") or "").strip()
        sc = sidecars_by_id.get(aid)
        if not sc:
            continue  # filtrar alucinaciones de IDs inexistentes
        rm = sc.get("report_meta", {}) or {}
        try:
            score = int(m.get("similarity_score", 0))
        except Exception:
            score = 0
        score = max(0, min(100, score))
        if score < 40:
            continue  # threshold defensivo
        band, color = similarity_band(float(score))
        enriched.append({
            "archive_id": aid,
            "similarity_score": score,
            "similarity_band": band,
            "similarity_color": color,
            "rationale": str(m.get("rationale", "") or ""),
            "resolution_summary": str(m.get("resolution_summary", "") or ""),
            "applicability": str(m.get("applicability", "") or ""),
            "consecutive": rm.get("consecutive", ""),
            "client": rm.get("client", ""),
            "asset": " · ".join(filter(None, [
                rm.get("asset_class", ""),
                rm.get("instance_tag", ""),
            ])),
            "date": sc.get("archived_at", "")[:10],
            "severity": rm.get("executive_severity", ""),
        })
    enriched.sort(key=lambda x: -x["similarity_score"])

    return {
        "matches": enriched,
        "global_assessment": str(parsed.get("global_assessment", "") or ""),
    }


def _empty_patterns(
    md: str,
    *,
    n_history_searched: int = 0,
    error: str = "",
    fallback_used: bool = False,
) -> Dict[str, Any]:
    return {
        "ok": False,
        "matches": [],
        "global_assessment": md,
        "n_history_searched": n_history_searched,
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
    "find_similar_patterns",
    "compute_fingerprint",
    "compute_fingerprint_for_current",
    "similarity_band",
    "SIMILARITY_BANDS",
    "DEFAULT_TOP_K",
]
