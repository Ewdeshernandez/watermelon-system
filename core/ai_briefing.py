"""
core.ai_briefing
================

Briefing Mensual Ejecutivo automático para C-level del cliente
(Ciclo 17.31).

Diferente del briefing diario (`core/briefing.py`) que apunta al
operador/specialist. Este apunta al **VP de Operaciones / CFO /
gerencia de mantenimiento del cliente final**, que no se loguea al
sistema y necesita ver el estado consolidado de TODOS sus activos
en 1 página de PDF que llega por email cada mes.

Contenido:
  - Header con cliente + mes + n_activos cubiertos
  - Párrafo ejecutivo AI (~80 palabras) sintetizando el mes
  - Top 3 prioridades del mes (las severidades más altas o las
    tendencias más preocupantes)
  - Lista breve de TODOS los activos con su severidad final
  - Footer con disclaimer + branding

API pública:
  - generate_monthly_briefing(client_filter, month_iso, viewer_email,
    viewer_role, ...) → dict con markdown + asset_summary +
    top_priorities + meta de la consulta + tokens/costo.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import date, datetime, timedelta
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
BRIEFING_CACHE_DIR = DATA_DIR / "cache" / "ai_briefing"

DEFAULT_BRIEFING_TTL_SECONDS = 7 * 24 * 3600
BRIEFING_PROMPT_VERSION = "v1_executive_monthly_2026_05"

# Severidad numérica para ranking de prioridades
_SEVERITY_RANK = {
    "CRÍTICA": 5,
    "ACCIÓN REQUERIDA": 4,
    "ATENCIÓN": 3,
    "VIGILANCIA": 2,
    "CONDICIÓN ACEPTABLE": 1,
}

# Colores para la tabla del PDF
SEVERITY_COLORS = {
    "CRÍTICA": "#dc2626",
    "ACCIÓN REQUERIDA": "#ea580c",
    "ATENCIÓN": "#f59e0b",
    "VIGILANCIA": "#84cc16",
    "CONDICIÓN ACEPTABLE": "#16a34a",
}


# =============================================================
# SYSTEM PROMPT — Briefing Ejecutivo C-level
# =============================================================

_SYSTEM_PROMPT_BRIEFING = """\
Eres un asistente ejecutivo de comunicación técnica para programas
de mantenimiento predictivo. Tu rol específico: redactar el
**briefing mensual** que recibe en su email el VP de Operaciones /
Gerente de Mantenimiento / CFO del cliente final. Este lector NO
es especialista en vibraciones — es decisor estratégico que
necesita entender en 90 segundos qué está pasando con la flota de
activos críticos y qué decisiones requieren su atención.

Vas a recibir el listado de reportes archivados del cliente en el
mes seleccionado, agrupados por activo, con metadata (cliente,
sitio, activo, severidad ejecutiva, resumen ejecutivo del último
reporte de cada activo).

ESTRUCTURA OBLIGATORIA DE LA RESPUESTA (sin emojis, sin caracteres
ornamentales, sin markdown crudo de bullets si no es necesario):

PÁRRAFO 1 — APERTURA EJECUTIVA (~70-100 palabras):
Empezá directo con: cuántos reportes técnicos se generaron en el
mes, cuántos activos cubrieron, distribución de severidades.
Mencioná el nombre del cliente. Cerrá el párrafo con la conclusión
GLOBAL del mes en una sola frase: estado estable / mejorando /
con activos en escalamiento que requieren acción ejecutiva.

### Top 3 prioridades operativas

Tres bullets numerados (1., 2., 3.). Cada uno con la siguiente
estructura:

**[Activo · Severidad].** [Hallazgo principal en 1 frase técnica
pero accesible al lector no-especialista]. **Acción ejecutiva
recomendada:** [acción concreta de nivel ejecutivo — programar
parada, asignar presupuesto a balanceo, ventana de mantenimiento,
monitoreo aumentado, etc.].

Las prioridades se ordenan por severidad descendente (CRÍTICA
primero) y dentro de la misma severidad por fecha de detección más
reciente. Si hay menos de 3 activos en estado relevante, listá
solo los que apliquen y al final agregá un comentario sobre el
estado general del resto.

### Estado del portafolio

Un párrafo (~50-80 palabras) describiendo el estado del resto de
los activos del cliente que no aparecen en el Top 3: cuántos están
en condición aceptable, cuántos en vigilancia rutinaria, cuántos
con tendencias menores que conviene seguir. Tono tranquilizador
cuando aplique, alertando solo donde corresponda.

CIERRE — RECOMENDACIÓN GLOBAL DEL MES (1 frase):

Una sola oración cerrando con la recomendación estratégica: si
mantener la cadencia actual de monitoreo, si acelerar
intervenciones programadas, si revisar el plan de mantenimiento
preventivo, etc.

DISCLAIMER (línea fija, sin variantes):

El presente briefing ejecutivo se emite conforme a la metodología Cat IV ISO 18436-2 con base en los reportes técnicos del periodo. La planificación operativa final es responsabilidad del operador del activo conforme a su sistema de gestión de integridad.

REGLAS DE VOZ Y ESTILO:

- Voz pasiva técnica, pero ACCESIBLE al lector no-especialista.
  Evitá jerga muy específica salvo cuando es indispensable
  (ejemplos: "oil whip", "BPFO", "Q-factor" → mejor "inestabilidad
  fluido-dinámica", "defecto en pista de rodamiento", "factor de
  amplificación dinámica" la primera vez, citando el término
  técnico entre paréntesis si es necesario).
- Sin emojis. Sin caracteres ornamentales (===, ***, ---).
- Voz en tercera persona / pasiva. Nunca primera persona.
- Cifras en lenguaje natural cuando sea posible: "el 60% de los
  activos" en lugar de "3 de 5".
- Citá normas (API 670, ISO 20816) sólo cuando sea indispensable
  para justificar una recomendación; este lector no las conoce
  todas.
- Máximo 380 palabras totales. Densidad sobre extensión. El VP lee
  esto en 90 segundos.
- NO inventes datos. Si la información del payload es escasa,
  declaralo explícitamente en lugar de extrapolar.
"""


# =============================================================
# HELPERS DE BÚSQUEDA Y AGREGACIÓN
# =============================================================

def _month_range(month_iso: str) -> Tuple[str, str]:
    """De 'YYYY-MM' devuelve (date_from, date_to) en formato ISO YYYY-MM-DD
    cubriendo el mes completo."""
    try:
        y_str, m_str = month_iso.split("-")
        y = int(y_str)
        m = int(m_str)
    except Exception:
        # Fallback: mes actual
        d = datetime.now()
        y, m = d.year, d.month
    first = date(y, m, 1)
    if m == 12:
        next_first = date(y + 1, 1, 1)
    else:
        next_first = date(y, m + 1, 1)
    last = next_first - timedelta(days=1)
    return first.isoformat(), last.isoformat()


def _list_reports_for_briefing(
    client_filter: str,
    month_iso: str,
    viewer_email: str,
    viewer_role: str,
) -> List[Dict[str, Any]]:
    """Lista los reportes archivados que cubren el cliente + mes.
    Reusa list_archived_reports con filtros."""
    date_from, date_to = _month_range(month_iso)
    sidecars = list_archived_reports(
        viewer_email=viewer_email,
        viewer_role=viewer_role,
        client_filter=client_filter,
        date_from=date_from,
        date_to=date_to,
        limit=500,
    )
    return sidecars


def _aggregate_by_asset(
    sidecars: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Agrupa reportes por instance_id (o instance_tag fallback) y
    devuelve UN registro por activo con su estado más reciente.

    Returns:
        [
            {
                "instance_id": str,
                "instance_tag": str,
                "asset_blob": str,
                "n_reports_in_month": int,
                "latest_archived_at": str,
                "latest_consecutive": str,
                "latest_severity": str,
                "latest_severity_rank": int,
                "latest_executive_summary": str,
                "all_severities_in_month": list[str],
                "archive_ids": list[str],
            },
            ...
        ]
        Ordenada por severity_rank descendente, luego latest_archived_at desc.
    """
    by_asset: Dict[str, Dict[str, Any]] = {}

    for sc in sidecars:
        rm = sc.get("report_meta", {}) or {}
        iid = (rm.get("instance_id") or "").strip()
        itag = (rm.get("instance_tag") or "").strip()
        # Key de agrupación: instance_id si existe, sino instance_tag
        key = iid or itag.lower() or rm.get("consecutive", "")
        if not key:
            continue

        archived_at = sc.get("archived_at", "")[:10]
        sev = (rm.get("executive_severity") or "").strip().upper()
        consecutive = rm.get("consecutive", "")
        exec_sum = (rm.get("executive_summary") or "").strip()

        asset_blob = " · ".join(filter(None, [
            rm.get("asset_class", ""),
            rm.get("asset_model", ""),
            itag,
        ]))

        if key not in by_asset:
            by_asset[key] = {
                "instance_id": iid,
                "instance_tag": itag,
                "asset_blob": asset_blob,
                "n_reports_in_month": 0,
                "latest_archived_at": "",
                "latest_consecutive": "",
                "latest_severity": "",
                "latest_severity_rank": 0,
                "latest_executive_summary": "",
                "all_severities_in_month": [],
                "archive_ids": [],
            }
        rec = by_asset[key]
        rec["n_reports_in_month"] += 1
        if sev:
            rec["all_severities_in_month"].append(sev)
        if archived_at > rec["latest_archived_at"]:
            rec["latest_archived_at"] = archived_at
            rec["latest_consecutive"] = consecutive
            rec["latest_severity"] = sev
            rec["latest_severity_rank"] = _SEVERITY_RANK.get(sev, 0)
            rec["latest_executive_summary"] = exec_sum[:1500]
        if sc.get("archive_id"):
            rec["archive_ids"].append(sc["archive_id"])

    # Ordenar por severity_rank desc, latest_archived_at desc
    out = list(by_asset.values())
    out.sort(
        key=lambda r: (
            -r.get("latest_severity_rank", 0),
            r.get("latest_archived_at", ""),
        ),
        reverse=False,  # severity desc primero, fecha asc dentro de la misma severity
    )
    # Truco: para que dentro de la misma severidad sea por fecha DESC,
    # invertimos el listado y volvemos a ordenar con clave compuesta.
    out.sort(
        key=lambda r: (
            -r.get("latest_severity_rank", 0),
            -_date_to_int(r.get("latest_archived_at", "")),
        ),
    )
    return out


def _date_to_int(d: str) -> int:
    """Convierte 'YYYY-MM-DD' en int comparable."""
    try:
        return int(d.replace("-", ""))
    except Exception:
        return 0


def _build_briefing_user_message(
    client_filter: str,
    month_iso: str,
    sidecars: List[Dict[str, Any]],
    asset_aggregates: List[Dict[str, Any]],
) -> str:
    """Compone el user message con todo el contexto del mes."""
    parts: List[str] = []
    parts.append("# Briefing mensual ejecutivo")
    parts.append("")
    parts.append(f"- Cliente: {client_filter or '(todos los clientes accesibles)'}")
    parts.append(f"- Periodo: mes {month_iso}")
    parts.append(f"- Reportes técnicos generados: {len(sidecars)}")
    parts.append(f"- Activos cubiertos: {len(asset_aggregates)}")
    parts.append("")

    # Conteo por severidad
    sev_count: Dict[str, int] = {}
    for ag in asset_aggregates:
        sev = ag.get("latest_severity", "").strip() or "(sin severidad)"
        sev_count[sev] = sev_count.get(sev, 0) + 1
    if sev_count:
        parts.append("**Distribución de severidad ejecutiva (último estado de cada activo):**")
        for sev, n in sorted(
            sev_count.items(),
            key=lambda x: -_SEVERITY_RANK.get(x[0], 0),
        ):
            parts.append(f"- {sev}: {n} activo(s)")
        parts.append("")

    parts.append("# Detalle por activo")
    parts.append("")
    for i, ag in enumerate(asset_aggregates, 1):
        parts.append(
            f"## {i}. {ag.get('asset_blob') or ag.get('instance_tag') or ag.get('instance_id') or 'Activo'}"
        )
        if ag.get("instance_tag"):
            parts.append(f"- Tag: {ag.get('instance_tag')}")
        parts.append(f"- Reportes en el mes: {ag.get('n_reports_in_month', 0)}")
        if ag.get("latest_consecutive"):
            parts.append(
                f"- Último reporte: {ag.get('latest_consecutive')} "
                f"({ag.get('latest_archived_at')})"
            )
        if ag.get("latest_severity"):
            parts.append(
                f"- Severidad ejecutiva final del mes: "
                f"{ag.get('latest_severity')}"
            )
        all_sevs = ag.get("all_severities_in_month") or []
        if len(all_sevs) > 1:
            unique_sevs = list(dict.fromkeys(all_sevs))
            parts.append(
                f"- Trayectoria del mes: {' → '.join(unique_sevs)}"
            )
        exec_sum = ag.get("latest_executive_summary", "")
        if exec_sum:
            parts.append("")
            parts.append("**Resumen ejecutivo del último reporte del activo:**")
            parts.append(exec_sum[:1500])
        parts.append("")

    parts.append("---")
    parts.append("")
    parts.append(
        "Por favor, redactá el BRIEFING MENSUAL EJECUTIVO siguiendo "
        "la estructura obligatoria. Recordá: el lector es VP de "
        "Operaciones / CFO, NO es especialista. El briefing debe "
        "leerse en 90 segundos y debe responder dos preguntas: "
        "(a) ¿qué pasó con mi flota este mes?, (b) ¿qué decisiones "
        "ejecutivas requiere mi atención?"
    )
    return "\n".join(parts)


def _briefing_payload_hash(
    client_filter: str,
    month_iso: str,
    archive_ids: List[str],
) -> str:
    blob = {
        "version": BRIEFING_PROMPT_VERSION,
        "model": _get_model_name(),
        "client": (client_filter or "").strip().lower(),
        "month": month_iso,
        "ids": sorted(archive_ids),
    }
    j = json.dumps(blob, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(j.encode("utf-8")).hexdigest()[:32]


# =============================================================
# API PÚBLICA — generate_monthly_briefing
# =============================================================

def generate_monthly_briefing(
    *,
    client_filter: str,
    month_iso: str,
    viewer_email: str,
    viewer_role: str,
    use_cache: bool = True,
    cache_ttl_seconds: int = DEFAULT_BRIEFING_TTL_SECONDS,
    max_tokens: int = 2048,
) -> Dict[str, Any]:
    """Genera el briefing mensual ejecutivo del cliente.

    Args:
        client_filter: nombre (o substring) del cliente.
        month_iso: 'YYYY-MM' del mes a cubrir (ej. '2026-04').
        viewer_email/role: identidad del que ejecuta (define qué
            reportes son accesibles según permisos del archivo).

    Returns:
        {
            "ok": bool,
            "markdown": str,         # briefing redactado por AI
            "asset_aggregates": list,  # data tabular para el PDF
            "n_reports": int,
            "n_assets": int,
            "month_iso": str,
            "client_filter": str,
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
        return _empty_briefing(
            "_AI no disponible — falta configurar `[anthropic] api_key`._",
            client_filter=client_filter, month_iso=month_iso,
        )

    if not month_iso or len(month_iso) < 7:
        return _empty_briefing(
            "_Mes inválido. Use formato 'YYYY-MM' (ej. '2026-04')._",
            client_filter=client_filter, month_iso=month_iso,
        )

    # 1) Listar reportes del cliente en el mes
    sidecars = _list_reports_for_briefing(
        client_filter, month_iso, viewer_email, viewer_role
    )
    if not sidecars:
        return _empty_briefing(
            f"_No se encontraron reportes archivados para el cliente "
            f"'{client_filter}' en el mes {month_iso}. Verificá que "
            f"el cliente esté escrito correctamente y que haya "
            f"reportes archivados en ese periodo._",
            client_filter=client_filter, month_iso=month_iso,
        )

    # 2) Agregar por activo
    asset_aggregates = _aggregate_by_asset(sidecars)
    archive_ids = [sc.get("archive_id", "") for sc in sidecars]

    # 3) Cache HIT
    h = _briefing_payload_hash(client_filter, month_iso, archive_ids)
    cache_path = BRIEFING_CACHE_DIR / f"{h}.json"
    if use_cache and cache_path.exists():
        try:
            age = time.time() - cache_path.stat().st_mtime
            if age <= cache_ttl_seconds:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                _bump_stats(0, 0, cached.get("model", ""), cached=True)
                return {**cached, "cached": True}
        except Exception:
            pass

    # 4) Llamada a Claude con retry + fallback
    client = _get_client()
    if client is None:
        return _empty_briefing(
            "_No se pudo inicializar el cliente Claude._",
            client_filter=client_filter, month_iso=month_iso,
        )

    user_msg = _build_briefing_user_message(
        client_filter, month_iso, sidecars, asset_aggregates
    )
    primary_model = _get_model_name()

    def _try(model_name: str, label: str):
        last = None
        for attempt in range(3):
            try:
                print(
                    f"[WM_AI_BRIEF] CALL · {label} · model={model_name} · "
                    f"client={client_filter!r} · month={month_iso} · "
                    f"n_assets={len(asset_aggregates)} · "
                    f"attempt={attempt + 1}/3 · hash={h[:8]}",
                    file=sys.stderr, flush=True,
                )
                resp = client.messages.create(
                    model=model_name,
                    max_tokens=max_tokens,
                    system=_SYSTEM_PROMPT_BRIEFING,
                    messages=[{"role": "user", "content": user_msg}],
                )
                return resp, None
            except Exception as exc:
                last = exc
                is_retry, st_code = _is_retryable_exc(exc)
                if is_retry and attempt < 2:
                    wait_s = 2 ** attempt
                    print(
                        f"[WM_AI_BRIEF] RETRY · {label} · "
                        f"status={st_code} · wait={wait_s}s",
                        file=sys.stderr, flush=True,
                    )
                    time.sleep(wait_s)
                    continue
                print(
                    f"[WM_AI_BRIEF] FAIL · {label} · {str(exc)[:200]}",
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
                f"[WM_AI_BRIEF] FALLBACK · {primary_model} agotado, "
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
            msg = f"_Error generando briefing:_\n\n```\n{err}\n```"
        return _empty_briefing(
            msg, client_filter=client_filter, month_iso=month_iso,
            error=err[:500], fallback_used=fallback_used,
        )

    try:
        markdown = response.content[0].text
    except Exception:
        markdown = "_(no text in response)_"
    in_tok = getattr(response.usage, "input_tokens", 0) if response.usage else 0
    out_tok = getattr(response.usage, "output_tokens", 0) if response.usage else 0

    if used_model.startswith("claude-haiku"):
        in_p, out_p = 1.0, 5.0
    else:
        in_p, out_p = 3.0, 15.0
    cost_usd = (in_tok * in_p + out_tok * out_p) / 1_000_000

    print(
        f"[WM_AI_BRIEF] OK · model={used_model} · in={in_tok} · "
        f"out={out_tok} · ~${cost_usd:.4f}"
        f"{' · FALLBACK' if fallback_used else ''}",
        file=sys.stderr, flush=True,
    )

    result = {
        "ok": True,
        "markdown": markdown,
        "asset_aggregates": asset_aggregates,
        "n_reports": len(sidecars),
        "n_assets": len(asset_aggregates),
        "month_iso": month_iso,
        "client_filter": client_filter,
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
            BRIEFING_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps(result, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:
            pass
    _bump_stats(in_tok, out_tok, used_model, cached=False)

    return result


def _empty_briefing(
    md: str,
    *,
    client_filter: str = "",
    month_iso: str = "",
    error: str = "",
    fallback_used: bool = False,
) -> Dict[str, Any]:
    return {
        "ok": False,
        "markdown": md,
        "asset_aggregates": [],
        "n_reports": 0,
        "n_assets": 0,
        "month_iso": month_iso,
        "client_filter": client_filter,
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
    "generate_monthly_briefing",
    "SEVERITY_COLORS",
]
