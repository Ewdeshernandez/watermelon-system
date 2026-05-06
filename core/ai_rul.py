"""
core.ai_rul
===========

AI Remaining Useful Life (RUL) Predictivo (Ciclo 17.30).

Estima la vida útil restante de un activo combinando:
  - Historia completa de reportes archivados del mismo instance_id.
  - Severidades ejecutivas a lo largo del tiempo.
  - Intervalos entre corridas.
  - Tendencia cualitativa de los executive_summary.
  - Estado actual (figuras + diagnósticos AI del reporte que se está
    preparando).

Claude actúa como ACTUARIO PREDICTIVO DE MANTENIMIENTO Cat IV ISO
18436-2 y emite estimaciones P10/P50/P90 con ventana óptima de
intervención. Disclaimer legal fuerte: las estimaciones requieren
validación humana antes de decisiones costosas.

Diseño defensivo:
  - Si hay menos de 3 reportes históricos: declarar "datos
    insuficientes para proyección estadística" — Claude sólo emite
    análisis cualitativo, no percentiles.
  - Si la severidad oscila erráticamente (no monótona): declarar
    "patrón de degradación no consistente, requiere validación
    operativa antes de proyectar".
  - Disclaimer obligatorio de "no es decisión de mantenimiento, es
    insumo para el especialista".

API pública:
  - find_asset_history(viewer, instance_id, instance_tag, limit)
    → lista de sidecars del mismo activo, ordenada por fecha asc.
  - generate_rul_estimate(history, current_meta, current_items)
    → dict con markdown + percentiles + ventana de intervención.
"""
from __future__ import annotations

import hashlib
import json
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
from core.ai_qa import _is_retryable_exc
from core.reports_archive import list_archived_reports


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RUL_CACHE_DIR = DATA_DIR / "cache" / "ai_rul"

DEFAULT_RUL_TTL_SECONDS = 7 * 24 * 3600  # 7 días: data fresca
RUL_PROMPT_VERSION = "v1_actuarial_2026_05"
MIN_HISTORY_FOR_RUL = 3  # menos que esto = solo análisis cualitativo


# Mapeo numérico de severidad para ratio de cambio
_SEVERITY_NUMERIC = {
    "CONDICIÓN ACEPTABLE": 1,
    "VIGILANCIA": 2,
    "ATENCIÓN": 3,
    "ACCIÓN REQUERIDA": 4,
    "CRÍTICA": 5,
}


# =============================================================
# SYSTEM PROMPT — Actuario Predictivo de Mantenimiento
# =============================================================

_SYSTEM_PROMPT_RUL = """\
Eres un actuario predictivo de mantenimiento Cat IV ISO 18436-2 con
25 años de experiencia haciendo análisis de degradación, curvas P-F
(Potential Failure to Failure) y estimaciones de Remaining Useful
Life (RUL) en máquinas rotativas industriales críticas. Tu rol
específico: combinar la historia completa del activo (reportes
archivados) con su estado actual y emitir una proyección de vida
útil restante con intervalos de confianza P10/P50/P90.

REGLAS CRÍTICAS DE GOBERNANZA:

Esta proyección es un INSUMO TÉCNICO para el especialista, NO una
decisión de mantenimiento. Tu salida no autoriza paradas, no
programa intervenciones, no compromete recursos. La decisión final
es del operador del activo basada en su sistema de gestión de
integridad.

Si la historia es insuficiente (< 3 reportes archivados), o si la
trayectoria es errática (severidad que sube y baja sin patrón
monótono), declaralo EXPLÍCITAMENTE y NO emitas percentiles
numéricos. Mejor un análisis cualitativo honesto que percentiles
inventados que pueden costar millones si se tomaran como verdad.

ESTRUCTURA OBLIGATORIA DE LA RESPUESTA (sin emojis, sin caracteres
ornamentales):

PÁRRAFO 1 — SITUACIÓN ACTUAL Y BASELINE (~70-100 palabras):
Empezá directo identificando el activo, la severidad actual, y
contextualizando contra el baseline histórico (severidad inicial
del primer reporte archivado). Mencioná cuántos reportes del activo
están en el archivo y el intervalo temporal cubierto. Si la
historia es insuficiente, declaralo aquí.

### Trayectoria histórica observada

UN párrafo (~80-130 palabras) describiendo la evolución mecánica del
activo en el tiempo: ¿la severidad ha sido monótona ascendente,
oscilante, estable?, ¿qué firmas mecánicas aparecieron y cuáles se
mantuvieron?, ¿qué intervenciones registradas explican cambios
abruptos?, ¿hay patrones cíclicos por estación operativa? Voz
pasiva técnica, citá cláusulas API/ISO cuando aplique.

### Modelo de degradación inferido

UN párrafo (~70-100 palabras) clasificando la degradación según
patrón industrial estándar:
  - Linear (degradación constante por unidad de tiempo)
  - Exponencial / acelerante (taza de cambio aumenta con el tiempo)
  - Estable / oscilante (sin tendencia neta)
  - Stepwise (saltos discretos asociados a eventos)
  - Datos insuficientes para clasificación estadística

Justificá la clasificación con la evidencia de los reportes.

### Proyección RUL

SI Y SOLO SI hay datos suficientes (≥ 3 reportes monótonos), emití
los siguientes percentiles en un bloque numerado:

1. **Mejor caso (P10):** XX días operativos antes de cruzar a
   próxima zona de severidad (típicamente zona D ISO 20816 o
   estado CRÍTICO).
2. **Caso esperado (P50):** XX días operativos.
3. **Peor caso (P90):** XX días operativos.

Después un párrafo cerrando con la VENTANA ÓPTIMA DE INTERVENCIÓN:
"Con base en estos percentiles, se sugiere planificar la
intervención correctiva dentro del intervalo [P50/2, P50] días
operativos desde la fecha del presente reporte". Esta ventana
balancea el riesgo de falla prematura contra el costo de
intervención prematura.

SI los datos son insuficientes o la trayectoria es errática,
escribí en su lugar: "**Datos insuficientes para proyección
cuantitativa.** [Explicación de por qué — historia corta,
trayectoria no monótona, etc.]. Se sugiere [acción concreta para
mejorar la base de datos predictiva: agregar X reportes más, etc.]"

### Variables que afectan la incertidumbre

Tres bullets numerados explicando los factores que más impactan la
incertidumbre del estimado:
  - Cambios operativos del activo (variaciones de carga, RPM,
    temperatura, etc.) que modifican la tasa de degradación.
  - Calidad y frecuencia del monitoreo (gaps temporales, cambios
    de instrumentación).
  - Eventos extraordinarios no capturados (intervenciones de
    mantenimiento entre corridas, cambios de operador, etc.).

### Disclaimer técnico-legal

Cerrá SIEMPRE con estos dos párrafos exactos, sin variantes, sin
reformular, sin numerar:

La presente proyección de vida útil restante se emite conforme a la metodología Cat IV ISO 18436-2 y constituye un insumo técnico estadístico para el especialista responsable, basado en la información disponible en el archivo del activo al momento del análisis.

Esta proyección NO constituye una decisión de mantenimiento, NO autoriza paradas operativas, y NO compromete recursos. La planificación de intervenciones, la asignación de presupuesto y la decisión final sobre el activo son responsabilidad del operador conforme a su sistema de gestión de integridad y deben validarse contra el contexto operativo completo.

REGLAS DE VOZ Y ESTILO:

- Voz pasiva técnica.
- Sin emojis, sin caracteres ornamentales.
- NO inventes números. Si la historia es insuficiente, decilo.
- NO emitas percentiles que no estén respaldados por la
  trayectoria observada.
- Cita normas con cláusula cuando exista (ISO 17359 anexo C, API
  580 §3, etc.).
- Máximo 600 palabras totales. Densidad sobre extensión.
"""


# =============================================================
# HELPERS
# =============================================================

def find_asset_history(
    viewer_email: str,
    viewer_role: str,
    instance_id: str = "",
    instance_tag: str = "",
    *,
    limit: int = 30,
) -> List[Dict[str, Any]]:
    """Devuelve la timeline completa del activo, ordenada cronológicamente
    ASCENDENTE (más viejo primero, más reciente último). Filtra por
    instance_id (preferido) o instance_tag normalizado."""
    if not instance_id and not instance_tag:
        return []

    sidecars = list_archived_reports(
        viewer_email=viewer_email,
        viewer_role=viewer_role,
        limit=500,
    )

    matches: List[Dict[str, Any]] = []
    iid_norm = (instance_id or "").strip()
    itag_norm = (instance_tag or "").strip().lower()

    for sc in sidecars:
        rm = sc.get("report_meta", {}) or {}
        sc_iid = (rm.get("instance_id") or "").strip()
        sc_tag = (rm.get("instance_tag") or "").strip().lower()
        match_id = iid_norm and sc_iid and sc_iid == iid_norm
        match_tag = (
            not match_id and itag_norm and sc_tag and sc_tag == itag_norm
        )
        if match_id or match_tag:
            matches.append(sc)

    # list_archived_reports ya ordena por archived_at desc; invertimos
    # para asc cronológico (más viejo primero).
    matches.sort(key=lambda x: x.get("archived_at", ""))
    return matches[-limit:]  # quedarse con los N más recientes (asc)


def _compute_severity_progression(
    history: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Cuantifica la trayectoria de severidad: ratio de cambio,
    monotonicidad, días totales cubiertos."""
    points: List[Tuple[str, int, str]] = []
    for sc in history:
        date = sc.get("archived_at", "")[:10]
        sev_label = (
            (sc.get("report_meta", {}) or {}).get("executive_severity", "")
            or ""
        ).strip().upper()
        sev_num = _SEVERITY_NUMERIC.get(sev_label)
        if sev_num is not None:
            points.append((date, sev_num, sev_label))

    if not points:
        return {
            "n_points": 0,
            "monotonic_ascending": False,
            "monotonic_descending": False,
            "total_days_covered": 0,
            "first_severity": "",
            "last_severity": "",
            "first_date": "",
            "last_date": "",
        }

    n = len(points)
    monotonic_asc = all(
        points[i][1] >= points[i - 1][1] for i in range(1, n)
    )
    monotonic_desc = all(
        points[i][1] <= points[i - 1][1] for i in range(1, n)
    )
    try:
        d_first = datetime.fromisoformat(points[0][0])
        d_last = datetime.fromisoformat(points[-1][0])
        total_days = (d_last - d_first).days
    except Exception:
        total_days = 0

    return {
        "n_points": n,
        "monotonic_ascending": monotonic_asc,
        "monotonic_descending": monotonic_desc,
        "total_days_covered": total_days,
        "first_severity": points[0][2],
        "last_severity": points[-1][2],
        "first_date": points[0][0],
        "last_date": points[-1][0],
        "severity_curve": [(d, lbl) for d, _, lbl in points],
    }


def _build_rul_user_message(
    history: List[Dict[str, Any]],
    current_meta: Dict[str, Any],
    current_items: List[Dict[str, Any]],
    progression: Dict[str, Any],
) -> str:
    """Compone el user message: contexto + timeline + estado actual."""
    parts: List[str] = []

    # Contexto del activo
    parts.append("# Activo bajo análisis predictivo")
    parts.append("")
    asset_blob = " · ".join(filter(None, [
        current_meta.get("asset_class", ""),
        current_meta.get("asset_model", ""),
        current_meta.get("instance_tag", ""),
    ]))
    if asset_blob:
        parts.append(f"- Activo: {asset_blob}")
    if current_meta.get("client"):
        parts.append(f"- Cliente: {current_meta.get('client')}")
    if current_meta.get("train_description"):
        parts.append(f"- Tren: {current_meta.get('train_description')}")
    sev_curr = (current_meta.get("executive_severity") or "").strip()
    if sev_curr:
        parts.append(f"- Severidad ACTUAL: {sev_curr}")
    parts.append(f"- Reportes históricos en archivo: {len(history)}")
    parts.append(
        f"- Intervalo temporal cubierto: "
        f"{progression.get('total_days_covered', 0)} días"
    )
    parts.append(
        f"- Trayectoria monotónica ascendente: "
        f"{'SÍ' if progression.get('monotonic_ascending') else 'NO'}"
    )
    parts.append("")

    # Timeline histórica
    parts.append("# Timeline histórica del activo")
    parts.append("")
    if not history:
        parts.append("(Sin reportes archivados previos)")
    else:
        for i, sc in enumerate(history, 1):
            rm = sc.get("report_meta", {}) or {}
            archived_at = sc.get("archived_at", "")[:10]
            consec = rm.get("consecutive", "")
            sev = rm.get("executive_severity", "")
            parts.append(
                f"## Reporte {i} — {consec or '(sin consecutivo)'} · "
                f"{archived_at}"
            )
            parts.append(f"- Severidad: {sev or '(no registrada)'}")
            exec_sum = (rm.get("executive_summary") or "").strip()
            if exec_sum:
                parts.append("- Resumen ejecutivo previo:")
                parts.append(f"  {exec_sum[:1000]}")
            parts.append("")

    # Estado actual sintetizado (figuras del reporte en preparación)
    parts.append("# Estado actual del activo (reporte en preparación)")
    parts.append("")
    parts.append(f"- Figuras del reporte actual: {len(current_items)}")
    if current_items:
        parts.append("- Tipos de análisis presentes:")
        types: Dict[str, int] = {}
        for it in current_items:
            t = str(it.get("type", "") or "(sin tipo)")
            types[t] = types.get(t, 0) + 1
        for t, n in sorted(types.items(), key=lambda x: -x[1]):
            parts.append(f"  · {t}: {n}")
    parts.append("")

    # Pedido explícito
    parts.append("---")
    parts.append("")
    parts.append(
        "Por favor, generá la PROYECCIÓN DE VIDA ÚTIL RESTANTE siguiendo "
        "la estructura obligatoria. Si los datos históricos son "
        "insuficientes (< 3 reportes monótonos), declaralo "
        "explícitamente y NO emitas percentiles numéricos. La calidad "
        "del estimado depende directamente de la calidad de la base de "
        "datos predictiva — no fuerces conclusiones débiles."
    )
    return "\n".join(parts)


def _rul_payload_hash(
    instance_id: str,
    history: List[Dict[str, Any]],
    current_meta: Dict[str, Any],
) -> str:
    blob = {
        "version": RUL_PROMPT_VERSION,
        "model": _get_model_name(),
        "iid": instance_id,
        "n_hist": len(history),
        "hist_ids": [
            sc.get("archive_id", "") for sc in history
        ],
        "current_consec": current_meta.get("consecutive", ""),
        "current_date": current_meta.get("report_date", ""),
        "current_severity": current_meta.get("executive_severity", ""),
    }
    j = json.dumps(blob, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(j.encode("utf-8")).hexdigest()[:32]


# =============================================================
# API PÚBLICA — generate_rul_estimate
# =============================================================

def generate_rul_estimate(
    history: List[Dict[str, Any]],
    current_meta: Dict[str, Any],
    current_items: List[Dict[str, Any]],
    *,
    use_cache: bool = True,
    cache_ttl_seconds: int = DEFAULT_RUL_TTL_SECONDS,
    max_tokens: int = 2048,
) -> Dict[str, Any]:
    """Genera la estimación de Remaining Useful Life del activo.

    Args:
        history: timeline de reportes archivados del mismo activo
                 (orden ASC, ya filtrada por instance_id).
        current_meta: meta del reporte actual (con executive_severity,
                      asset_class, etc.).
        current_items: figuras del reporte actual.

    Returns:
        {
            "ok": bool,
            "markdown": str,
            "n_history": int,
            "history_days_covered": int,
            "monotonic": bool,
            "first_severity": str,
            "last_severity": str,
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
        return _empty_rul_response(
            "_⚠️ AI no disponible — falta configurar `[anthropic] api_key`._"
        )

    progression = _compute_severity_progression(history)
    n_hist = progression.get("n_points", 0)

    # Si la historia es muy corta, igual permitimos al AI emitir un
    # análisis cualitativo (el system prompt lo guía a no inventar
    # percentiles). No bloqueamos la generación porque hay valor en
    # decir "no hay data suficiente, agregá reportes".
    instance_id = current_meta.get("instance_id", "")

    # Cache HIT
    h = _rul_payload_hash(instance_id, history, current_meta)
    cache_path = RUL_CACHE_DIR / f"{h}.json"
    if use_cache and cache_path.exists():
        try:
            age = time.time() - cache_path.stat().st_mtime
            if age <= cache_ttl_seconds:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                _bump_stats(0, 0, cached.get("model", ""), cached=True)
                return {**cached, "cached": True}
        except Exception:
            pass

    client = _get_client()
    if client is None:
        return _empty_rul_response("_⚠️ No se pudo inicializar el cliente._")

    user_msg = _build_rul_user_message(
        history, current_meta, current_items, progression
    )
    primary_model = _get_model_name()

    def _try(model_name: str, label: str):
        last = None
        for attempt in range(3):
            try:
                print(
                    f"[WM_AI_RUL] CALL · {label} · model={model_name} · "
                    f"n_hist={n_hist} · attempt={attempt + 1}/3 · "
                    f"hash={h[:8]}",
                    file=sys.stderr, flush=True,
                )
                resp = client.messages.create(
                    model=model_name,
                    max_tokens=max_tokens,
                    system=_SYSTEM_PROMPT_RUL,
                    messages=[{"role": "user", "content": user_msg}],
                )
                return resp, None
            except Exception as exc:
                last = exc
                is_retry, st_code = _is_retryable_exc(exc)
                if is_retry and attempt < 2:
                    wait_s = 2 ** attempt
                    print(
                        f"[WM_AI_RUL] RETRY · {label} · status={st_code} · "
                        f"wait={wait_s}s",
                        file=sys.stderr, flush=True,
                    )
                    time.sleep(wait_s)
                    continue
                print(
                    f"[WM_AI_RUL] FAIL · {label} · {str(exc)[:200]}",
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
                f"[WM_AI_RUL] FALLBACK · {primary_model} agotado, "
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
            msg = ("_⚠️ Servidores Claude sobrecargados. "
                   "Esperá 5-10 min y reintentá._")
        elif "timeout" in err.lower() or "timed out" in err.lower():
            msg = "_⚠️ Timeout de conexión. Verificá tu red y reintentá._"
        else:
            msg = f"_⚠️ Error generando RUL:_\n\n```\n{err}\n```"
        return _empty_rul_response(
            msg, error=err[:500], fallback_used=fallback_used,
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
        f"[WM_AI_RUL] OK · model={used_model} · in={in_tok} · "
        f"out={out_tok} · ~${cost_usd:.4f}"
        f"{' · FALLBACK' if fallback_used else ''}",
        file=sys.stderr, flush=True,
    )

    result = {
        "ok": True,
        "markdown": markdown,
        "n_history": n_hist,
        "history_days_covered": progression.get("total_days_covered", 0),
        "monotonic": progression.get("monotonic_ascending", False),
        "first_severity": progression.get("first_severity", ""),
        "last_severity": progression.get("last_severity", ""),
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
            RUL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps(result, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:
            pass
    _bump_stats(in_tok, out_tok, used_model, cached=False)

    return result


def _empty_rul_response(
    md: str,
    *,
    error: str = "",
    fallback_used: bool = False,
) -> Dict[str, Any]:
    return {
        "ok": False,
        "markdown": md,
        "n_history": 0,
        "history_days_covered": 0,
        "monotonic": False,
        "first_severity": "",
        "last_severity": "",
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
    "find_asset_history",
    "generate_rul_estimate",
    "MIN_HISTORY_FOR_RUL",
]
