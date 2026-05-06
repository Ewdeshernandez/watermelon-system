"""
core.ai_runcompare
==================

AI Run-vs-Run Comparison (Ciclo 17.28).

Cuando el especialista prepara un reporte de un activo que ya tuvo
reportes anteriores, este módulo lee automáticamente el ÚLTIMO
reporte archivado del mismo activo y genera un delta narrativo
forense: qué métricas cambiaron, qué firmas mecánicas son nuevas o
desaparecieron, hacia dónde va la tendencia, en qué ventana de
tiempo, y qué implicación operativa tiene.

Es lo que separa "monitoring" (mirar números) de "condition
monitoring" (entender la evolución mecánica del activo).

Diseño:
  1. find_previous_report(viewer, instance_id, before_date)
     - Busca en core.reports_archive el último reporte archivado
       cuyo report_meta.instance_id coincide.
     - Filtro por before_date para evitar comparar contra el
       reporte que se está preparando ahora.
     - Permisos heredados de list_archived_reports.

  2. generate_run_comparison(prev_sidecar, current_meta,
     current_items)
     - Extrae las tablas cuantitativas de las notas (marcador
       <<<WM_AI_BLOCK>>>) del reporte ACTUAL para tener números
       crudos comparables.
     - Extrae las mismas tablas del PDF anterior si están en su
       JSON (sidecar.report_meta.executive_summary contiene la
       severidad y la prosa cualitativa; las cuantitativas vienen
       del PDF si las extraemos).
     - Construye payload con: severidad anterior vs actual,
       executive_summary anterior, items actuales con sus tablas,
       días transcurridos.
     - Llama a Claude con system prompt "delta forense".
     - Devuelve respuesta + metadata (modelo, tokens, costo,
       fallback flag, días transcurridos, archive_id anterior).

API pública:
  - find_previous_report(viewer_email, viewer_role, instance_id,
                         instance_tag?, before_date?) → sidecar | None
  - generate_run_comparison(prev_sidecar, current_meta,
                            current_items, ...) → dict
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from datetime import datetime, timedelta
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
from core.ai_qa import _is_retryable_exc  # reusa la heurística
from core.reports_archive import list_archived_reports


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RUNCOMPARE_CACHE_DIR = DATA_DIR / "cache" / "ai_runcompare"

DEFAULT_CACHE_TTL_SECONDS = 30 * 24 * 3600
RUNCOMPARE_PROMPT_VERSION = "v1_delta_forense_2026_05"


# =============================================================
# SYSTEM PROMPT — Delta Forense
# =============================================================

_SYSTEM_PROMPT_RUNCOMPARE = """\
Eres un especialista de análisis de vibraciones Cat IV ISO 18436-2
senior, con 25 años de experiencia en programas de monitoreo
continuo. Tu rol específico aquí: comparar el reporte que se está
preparando AHORA contra el último reporte archivado del MISMO
activo, y producir el "delta forense" — la narrativa de evolución
mecánica que separa monitoring de condition monitoring real.

Vas a recibir:
  - Metadata + severidad + resumen ejecutivo del reporte ANTERIOR.
  - Metadata + figuras + datos cuantitativos del reporte ACTUAL.
  - Días transcurridos entre ambos.

ESTRUCTURA OBLIGATORIA DE LA RESPUESTA (sin emojis, sin caracteres
ornamentales):

PÁRRAFO 1 — LEAD DE EVOLUCIÓN (~70-100 palabras):
Empezá directo con la comparación de severidad ejecutiva (anterior
vs actual), días transcurridos, y la conclusión global del cambio
en una frase: ESCALAMIENTO / ESTABILIDAD / MEJORÍA. Mencioná el tag
del activo. Ejemplo: "En las últimas 30 días operativas, el
turbogenerador TES1 evolucionó de severidad ATENCIÓN a CRÍTICA, lo
que constituye un escalamiento confirmado del estado mecánico del
tren."

### Cambios cuantitativos detectados

Tres a seis bullets numerados (1., 2., 3., ...) cada uno con la
estructura: **[Métrica/Firma]**: [valor anterior] → [valor actual]
([delta absoluto o porcentual]). Comentario interpretativo breve
de una oración. Cubrí:
  - Cambios en overall RMS / amplitudes principales por canal.
  - Aparición o desaparición de firmas mecánicas (sub-sincrónicas,
    BPFx, harmónicos altos, modulación).
  - Cambios en Q-factor o ancho de banda de picos resonantes.
  - Cambios en posición DC del muñón si hay shaft centerline.
  - Cambios en ratio X/Y de órbitas si hay orbit data.
Cada bullet identifica explícitamente qué figura/canal del reporte
provee la evidencia.

### Interpretación clínica de la evolución

UN párrafo (~80-130 palabras) sin header bullet. Sintetizá la
"trayectoria mecánica" — ¿el activo está acelerando hacia falla, o
estabilizándose?, ¿cuál es la causa raíz que se está consolidando?,
¿hay nuevas hipótesis disparadas por los cambios? Citá normas
relevantes con cláusula (API 670 §4.3.2, API 684 §6.8.3, ISO 17359
anexo C, etc.) cuando sea pertinente. Voz pasiva técnica.

### Implicación operativa y ventana de acción

UN párrafo (~60-100 palabras). Cuál es la implicación INMEDIATA del
delta para el operador: ¿se mantiene el plan de mantenimiento
original, se debe acelerar la intervención, hay riesgo de
escalamiento si no se actúa? Si la trayectoria es lineal y se
extrapola, mencioná la ventana de tiempo estimada antes del
próximo nivel de severidad. Cierre con la frase obligatoria:
"La planificación final es responsabilidad del operador del activo
conforme a su sistema de gestión de integridad."

### Evaluación de confianza

UN párrafo único iniciando con "Confianza del diagnóstico
evolutivo: XX%" (entero). Explicá:
  - Cuántos puntos de comparación convergen (n_figuras comparables).
  - Limitaciones temporales (intervalo entre reportes — corto =
    poca tendencia, largo = sin trazabilidad fina).
  - Sesgos posibles (cambios operativos, cambios de instrumentación,
    estado del aceite/cojinetes no medido, etc.).
  - Qué información complementaria reduciría la incertidumbre.

REGLAS DE VOZ Y ESTILO:

- Voz pasiva técnica: "se observa", "se concluye", "se identifica".
- Sin emojis, sin caracteres ornamentales (===, ***, ---).
- NO uses primera persona ("yo creo") ni segunda ("debes").
- Si la magnitud del cambio es despreciable y la severidad se
  mantiene, declaralo explícitamente como ESTABILIDAD MECÁNICA, no
  inventes deltas para justificar la respuesta.
- Si hay menos de 3 figuras comparables, declaralo como limitación
  en el bloque de Confianza, no fuerces conclusiones débiles.
- Si las unidades de medición cambiaron entre reportes, advertí
  explícitamente que la comparación tiene un componente metodológico
  que requiere validación.
- Máximo 500 palabras totales. Densidad sobre extensión.
- Cita números reales del payload, no inventes valores.
"""


# =============================================================
# HELPERS DE BÚSQUEDA Y EXTRACCIÓN
# =============================================================

def find_previous_report(
    viewer_email: str,
    viewer_role: str,
    instance_id: str = "",
    instance_tag: str = "",
    *,
    before_date: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Busca el último reporte archivado del mismo activo.

    Estrategia:
      1. Listar reportes accesibles al viewer.
      2. Filtrar por instance_id (match exacto en report_meta).
      3. Si no hay match por instance_id, fallback a instance_tag.
      4. Filtrar por archived_at < before_date si se proveyó.
      5. Devolver el más reciente (la lista ya viene ordenada desc).

    Args:
        viewer_email/role: identidad del que consulta.
        instance_id: ID único del activo (preferido).
        instance_tag: tag legible (fallback si no hay instance_id).
        before_date: ISO YYYY-MM-DD; reportes posteriores se ignoran.

    Returns:
        sidecar dict completo (con _pdf_path, _sidecar_path) o None
        si no hay reporte anterior.
    """
    if not instance_id and not instance_tag:
        return None

    sidecars = list_archived_reports(
        viewer_email=viewer_email,
        viewer_role=viewer_role,
        limit=500,
    )

    # Filtrar por activo + fecha
    matches: List[Dict[str, Any]] = []
    instance_id_norm = (instance_id or "").strip()
    instance_tag_norm = (instance_tag or "").strip().lower()

    for sc in sidecars:
        rm = sc.get("report_meta", {}) or {}
        sc_iid = (rm.get("instance_id") or "").strip()
        sc_tag = (rm.get("instance_tag") or "").strip().lower()

        # Match exacto por instance_id (preferido)
        match_by_id = (
            instance_id_norm and sc_iid and sc_iid == instance_id_norm
        )
        # Fallback: match por tag normalizado
        match_by_tag = (
            not match_by_id
            and instance_tag_norm
            and sc_tag
            and sc_tag == instance_tag_norm
        )
        if not (match_by_id or match_by_tag):
            continue

        # Filtro temporal opcional
        archived_date = sc.get("archived_at", "")[:10]
        if before_date and archived_date >= before_date:
            continue

        matches.append(sc)

    if not matches:
        return None

    # list_archived_reports ya ordena por archived_at desc → el primero
    # del filtrado es el más reciente.
    return matches[0]


_AI_BLOCK_REGEX = re.compile(
    r"<<<WM_AI_BLOCK>>>(.*?)<<<WM_AI_NARRATIVE>>>",
    re.DOTALL,
)


def _extract_quant_table_from_notes(notes: str) -> List[Tuple[str, str]]:
    """Extrae los pares (parámetro, valor) de la tabla cuantitativa
    embebida en las notas con marcadores <<<WM_AI_BLOCK>>>. Si no hay
    marcador, devuelve lista vacía."""
    if not notes:
        return []
    m = _AI_BLOCK_REGEX.search(notes)
    if not m:
        return []
    quant_text = m.group(1).strip()
    rows: List[Tuple[str, str]] = []
    for line in quant_text.splitlines():
        if "|" not in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 2 and parts[0] and parts[0] != "Parámetro":
            rows.append((parts[0], parts[1]))
    return rows


def _summarize_current_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """Reduce un report_item a la metadata mínima necesaria para
    comparación: tipo, título, máquina, punto, tabla cuantitativa."""
    return {
        "type": str(item.get("type", "") or ""),
        "title": str(item.get("title", "") or ""),
        "machine": str(item.get("machine", "") or ""),
        "point": str(item.get("point", "") or ""),
        "variable": str(item.get("variable", "") or ""),
        "quant_table": _extract_quant_table_from_notes(
            str(item.get("notes", "") or "")
        ),
    }


def _build_runcompare_user_message(
    prev_sidecar: Dict[str, Any],
    current_meta: Dict[str, Any],
    current_items: List[Dict[str, Any]],
) -> str:
    """Compone el user message: previo + actual + delta calculado."""
    rm_prev = prev_sidecar.get("report_meta", {}) or {}
    archived_at = prev_sidecar.get("archived_at", "")[:10]
    current_date = (current_meta.get("report_date") or
                    datetime.now().strftime("%Y-%m-%d"))

    # Calcular días transcurridos
    days_elapsed: Optional[int] = None
    try:
        d_prev = datetime.fromisoformat(archived_at)
        d_curr = datetime.fromisoformat(current_date[:10])
        days_elapsed = (d_curr - d_prev).days
    except Exception:
        pass

    parts: List[str] = []
    parts.append("# Reporte ANTERIOR (archivado)")
    parts.append("")
    if rm_prev.get("consecutive"):
        parts.append(f"- Consecutivo: {rm_prev.get('consecutive')}")
    parts.append(f"- Archivado: {archived_at}")
    if rm_prev.get("report_date"):
        parts.append(f"- Fecha del análisis: {rm_prev.get('report_date')}")
    if rm_prev.get("client"):
        parts.append(f"- Cliente: {rm_prev.get('client')}")
    asset_blob_prev = " · ".join(filter(None, [
        rm_prev.get("asset_class", ""),
        rm_prev.get("asset_model", ""),
        rm_prev.get("instance_tag", ""),
    ]))
    if asset_blob_prev:
        parts.append(f"- Activo: {asset_blob_prev}")
    if rm_prev.get("train_description"):
        parts.append(f"- Tren: {rm_prev.get('train_description')}")
    sev_prev = rm_prev.get("executive_severity", "").strip()
    if sev_prev:
        parts.append(f"- Severidad ejecutiva ANTERIOR: {sev_prev}")
    exec_sum_prev = (rm_prev.get("executive_summary") or "").strip()
    if exec_sum_prev:
        parts.append("")
        parts.append("**Resumen ejecutivo del reporte anterior:**")
        parts.append(exec_sum_prev[:2000])

    parts.append("")
    parts.append("---")
    parts.append("")
    parts.append("# Reporte ACTUAL (en preparación)")
    parts.append("")
    if current_meta.get("consecutive"):
        parts.append(f"- Consecutivo: {current_meta.get('consecutive')}")
    parts.append(f"- Fecha del análisis: {current_date}")
    if current_meta.get("client"):
        parts.append(f"- Cliente: {current_meta.get('client')}")
    asset_blob_curr = " · ".join(filter(None, [
        current_meta.get("asset_class", ""),
        current_meta.get("asset_model", ""),
        current_meta.get("instance_tag", ""),
    ]))
    if asset_blob_curr:
        parts.append(f"- Activo: {asset_blob_curr}")
    sev_curr = (current_meta.get("executive_severity") or "").strip()
    if sev_curr:
        parts.append(f"- Severidad ejecutiva ACTUAL: {sev_curr}")
    if days_elapsed is not None:
        parts.append(f"- **Días transcurridos entre reportes: {days_elapsed}**")

    parts.append("")
    parts.append(f"**Figuras del reporte actual ({len(current_items)} total):**")
    parts.append("")
    for idx, item in enumerate(current_items, 1):
        summary = _summarize_current_item(item)
        parts.append(
            f"{idx}. **{summary['type'].upper()}** · "
            f"{summary['title']}"
        )
        if summary["machine"] or summary["point"]:
            parts.append(
                f"   - Ubicación: {summary['machine']} / {summary['point']}"
            )
        if summary["quant_table"]:
            parts.append("   - Tabla cuantitativa de evidencia:")
            for k, v in summary["quant_table"]:
                parts.append(f"     · {k}: {v}")
        parts.append("")

    parts.append("---")
    parts.append("")
    parts.append(
        "Por favor, generá el DELTA FORENSE siguiendo la estructura "
        "obligatoria (lead de evolución, cambios cuantitativos detectados, "
        "interpretación clínica, implicación operativa y ventana de "
        "acción, evaluación de confianza). Citá números reales del payload, "
        "no inventes valores."
    )
    return "\n".join(parts)


def _runcompare_payload_hash(
    prev_archive_id: str,
    current_meta: Dict[str, Any],
    current_items: List[Dict[str, Any]],
) -> str:
    """Hash determinístico para cache."""
    blob = {
        "version": RUNCOMPARE_PROMPT_VERSION,
        "model": _get_model_name(),
        "prev_id": prev_archive_id,
        "current_consecutive": current_meta.get("consecutive", ""),
        "current_date": current_meta.get("report_date", ""),
        "n_items": len(current_items),
        "items_sig": [
            hashlib.sha256(
                str(i.get("notes", "") or "").encode("utf-8")
            ).hexdigest()[:16]
            for i in current_items
        ],
    }
    j = json.dumps(blob, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(j.encode("utf-8")).hexdigest()[:32]


# =============================================================
# API PÚBLICA — generate_run_comparison
# =============================================================

def generate_run_comparison(
    prev_sidecar: Dict[str, Any],
    current_meta: Dict[str, Any],
    current_items: List[Dict[str, Any]],
    *,
    use_cache: bool = True,
    cache_ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS,
    max_tokens: int = 2048,
) -> Dict[str, Any]:
    """Genera el delta forense entre dos reportes del mismo activo.

    Args:
        prev_sidecar: sidecar JSON del reporte anterior (con
            archive_id, archived_at, report_meta, etc.).
        current_meta: meta dict del reporte que se está preparando.
        current_items: lista de report_items del reporte actual.

    Returns:
        {
            "ok": bool,
            "markdown": str,
            "prev_archive_id": str,
            "prev_archived_at": str,
            "prev_consecutive": str,
            "days_elapsed": int | None,
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
    if not prev_sidecar:
        return _empty_runcompare("_No hay reporte anterior para comparar._")
    if not current_items:
        return _empty_runcompare(
            "_El reporte actual no tiene figuras para comparar._"
        )
    if not is_ai_available():
        return _empty_runcompare(
            "_AI no disponible — falta configurar `[anthropic] api_key`._"
        )

    prev_archive_id = prev_sidecar.get("archive_id", "")
    prev_archived_at = prev_sidecar.get("archived_at", "")[:10]
    rm_prev = prev_sidecar.get("report_meta", {}) or {}
    prev_consecutive = rm_prev.get("consecutive", "")

    # Días transcurridos
    days_elapsed: Optional[int] = None
    try:
        d_prev = datetime.fromisoformat(prev_archived_at)
        current_date_str = (current_meta.get("report_date") or
                            datetime.now().strftime("%Y-%m-%d"))[:10]
        d_curr = datetime.fromisoformat(current_date_str)
        days_elapsed = max(0, (d_curr - d_prev).days)
    except Exception:
        pass

    # Cache HIT
    h = _runcompare_payload_hash(prev_archive_id, current_meta, current_items)
    cache_path = RUNCOMPARE_CACHE_DIR / f"{h}.json"
    if use_cache and cache_path.exists():
        try:
            age = time.time() - cache_path.stat().st_mtime
            if age <= cache_ttl_seconds:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                _bump_stats(0, 0, cached.get("model", ""), cached=True)
                return {**cached, "cached": True}
        except Exception:
            pass

    # Llamada con retry + fallback
    client = _get_client()
    if client is None:
        return _empty_runcompare("_No se pudo inicializar el cliente._")

    user_msg = _build_runcompare_user_message(
        prev_sidecar, current_meta, current_items
    )
    primary_model = _get_model_name()

    def _try(model_name: str, label: str):
        last = None
        for attempt in range(3):
            try:
                print(
                    f"[WM_AI_RUNCMP] CALL · {label} · model={model_name} · "
                    f"prev={prev_archive_id[:30]} · "
                    f"days={days_elapsed} · attempt={attempt + 1}/3 · "
                    f"hash={h[:8]}",
                    file=sys.stderr, flush=True,
                )
                resp = client.messages.create(
                    model=model_name,
                    max_tokens=max_tokens,
                    system=_SYSTEM_PROMPT_RUNCOMPARE,
                    messages=[{"role": "user", "content": user_msg}],
                )
                return resp, None
            except Exception as exc:
                last = exc
                is_retry, st_code = _is_retryable_exc(exc)
                if is_retry and attempt < 2:
                    wait_s = 2 ** attempt
                    print(
                        f"[WM_AI_RUNCMP] RETRY · {label} · "
                        f"status={st_code} · wait={wait_s}s",
                        file=sys.stderr, flush=True,
                    )
                    time.sleep(wait_s)
                    continue
                print(
                    f"[WM_AI_RUNCMP] FAIL · {label} · {str(exc)[:200]}",
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
                f"[WM_AI_RUNCMP] FALLBACK · {primary_model} agotado, "
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
            msg = ("_Servidores Claude sobrecargados. "
                   "Esperá 5-10 min y reintentá._")
        elif "timeout" in err.lower() or "timed out" in err.lower():
            msg = "_Timeout de conexión. Verificá tu red y reintentá._"
        else:
            msg = f"_Error generando comparación:_\n\n```\n{err}\n```"
        return _empty_runcompare(
            msg,
            prev_archive_id=prev_archive_id,
            prev_archived_at=prev_archived_at,
            prev_consecutive=prev_consecutive,
            days_elapsed=days_elapsed,
            error=err[:500],
            fallback_used=fallback_used,
        )

    try:
        markdown = response.content[0].text
    except Exception:
        markdown = "_(no text in response)_"
    in_tok = getattr(response.usage, "input_tokens", 0) if response.usage else 0
    out_tok = getattr(response.usage, "output_tokens", 0) if response.usage else 0

    # Pricing dinámico
    if used_model.startswith("claude-haiku"):
        in_p, out_p = 1.0, 5.0
    else:
        in_p, out_p = 3.0, 15.0
    cost_usd = (in_tok * in_p + out_tok * out_p) / 1_000_000

    print(
        f"[WM_AI_RUNCMP] OK · model={used_model} · in={in_tok} · "
        f"out={out_tok} · ~${cost_usd:.4f}"
        f"{' · FALLBACK' if fallback_used else ''}",
        file=sys.stderr, flush=True,
    )

    result = {
        "ok": True,
        "markdown": markdown,
        "prev_archive_id": prev_archive_id,
        "prev_archived_at": prev_archived_at,
        "prev_consecutive": prev_consecutive,
        "days_elapsed": days_elapsed,
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
            RUNCOMPARE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps(result, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:
            pass
    _bump_stats(in_tok, out_tok, used_model, cached=False)

    return result


def _empty_runcompare(
    md: str,
    *,
    prev_archive_id: str = "",
    prev_archived_at: str = "",
    prev_consecutive: str = "",
    days_elapsed: Optional[int] = None,
    error: str = "",
    fallback_used: bool = False,
) -> Dict[str, Any]:
    return {
        "ok": False,
        "markdown": md,
        "prev_archive_id": prev_archive_id,
        "prev_archived_at": prev_archived_at,
        "prev_consecutive": prev_consecutive,
        "days_elapsed": days_elapsed,
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
    "find_previous_report",
    "generate_run_comparison",
]
