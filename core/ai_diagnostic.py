"""
core.ai_diagnostic
==================

Wrapper de Claude API (Anthropic) para generar diagnósticos clínicos
con IA en cada módulo de análisis. Ciclo 17.26.

Diseño (Opción C híbrido):
  - El diagnóstico técnico determinístico actual del sistema (basado
    en reglas ISO/API + thresholds) SE MANTIENE como evidencia
    auditable para reportes legales / de cliente.
  - Esta capa AGREGA un bloque adicional de "Interpretación clínica
    AI" en lenguaje natural, con análisis cualitativo y
    recomendaciones de acción concretas.
  - Los dos bloques conviven en el reporte: el técnico es la
    evidencia, el AI es la interpretación.

Costo aproximado:
  - Claude Sonnet 4.5: ~$3/MTok input, ~$15/MTok output
  - Diagnóstico promedio: 800 tok in + 500 tok out ≈ USD 0.010
  - 100 diagnósticos/mes ≈ USD 1
  - 1000 diagnósticos/mes ≈ USD 10

API pública:
  - generate_ai_diagnostic(payload, module_type, ...) → str (markdown)
  - is_ai_available() → bool (hay key configurada)
  - clear_diagnostic_cache() → int (limpia cache local)
  - get_ai_stats() → dict (n_calls, total_tokens, etc para admin)

Cache:
  Las requests se cachean por hash SHA256 del payload normalizado en
  data/cache/ai_diagnostics/{hash}.json con TTL configurable. Si el
  specialist genera 2 veces el mismo análisis, no pagamos 2 veces.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache" / "ai_diagnostics"
STATS_FILE = DATA_DIR / "ai_diagnostic_stats.json"

# TTL del cache: 30 días. Pasado ese tiempo, se re-genera.
DEFAULT_CACHE_TTL_SECONDS = 30 * 24 * 3600

# Modelo por default (configurable desde secrets [anthropic].model)
DEFAULT_MODEL = "claude-sonnet-4-5-20250929"
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TIMEOUT_S = 30

# Modelo de fallback: si el modelo principal (Sonnet 4.5 por default) falla
# 3 veces con errores transientes (529 overloaded, 503, 429), intentamos
# una vez más con Haiku 4.5. Haiku corre en infraestructura distinta y
# casi nunca se sobrecarga. Calidad ligeramente menor pero suficiente para
# vibraciones y mejor que mostrarle un error al cliente.
# Costo Haiku: ~$1/MTok input, ~$5/MTok output (vs Sonnet $3 / $15).
FALLBACK_MODEL = "claude-haiku-4-5-20251001"


# =============================================================
# CONFIGURACIÓN DESDE SECRETS
# =============================================================

def _read_anthropic_secret() -> Dict[str, Any]:
    """Lee [anthropic] de st.secrets. Devuelve {} si no hay sesión
    Streamlit o no está configurado.
    """
    try:
        import streamlit as st  # type: ignore
        if not hasattr(st, "secrets"):
            return {}
        sec = st.secrets
        if "anthropic" not in sec:
            return {}
        sub = sec["anthropic"]
        # AttrDict de Streamlit Cloud → dict puro
        try:
            return {k: sub[k] for k in sub}
        except Exception:
            try:
                return dict(sub)
            except Exception:
                return {}
    except Exception:
        return {}


def is_ai_available() -> bool:
    """True si hay API key configurada y el SDK anthropic está instalado."""
    cfg = _read_anthropic_secret()
    if not cfg.get("api_key", "").strip():
        return False
    try:
        import anthropic  # noqa: F401
        return True
    except ImportError:
        return False


def _get_client():
    """Devuelve cliente Anthropic configurado, o None si no hay key/SDK."""
    cfg = _read_anthropic_secret()
    api_key = str(cfg.get("api_key", "")).strip()
    if not api_key:
        return None
    try:
        from anthropic import Anthropic
        return Anthropic(api_key=api_key, timeout=DEFAULT_TIMEOUT_S)
    except Exception as e:
        print(f"[WM_AI] FAIL · no se pudo crear cliente Anthropic: {e}",
              file=sys.stderr, flush=True)
        return None


def _get_model_name() -> str:
    cfg = _read_anthropic_secret()
    return str(cfg.get("model", DEFAULT_MODEL)).strip() or DEFAULT_MODEL


# =============================================================
# CACHE LOCAL POR HASH DEL PAYLOAD
# =============================================================

def _payload_hash(payload: Dict[str, Any], module_type: str) -> str:
    """Hash determinístico del payload normalizado. Si el mismo análisis
    se pide 2 veces, sale del cache (no pagamos a Anthropic 2 veces).

    El hash incluye `_PROMPT_VERSION`: cuando el system prompt cambia,
    todos los diagnósticos cacheados con la versión anterior se
    re-generan automáticamente con la nueva voz / formato.
    """
    blob = {
        "module": module_type,
        "model": _get_model_name(),
        "prompt_version": _PROMPT_VERSION,
        "payload": payload,
    }
    j = json.dumps(blob, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(j.encode("utf-8")).hexdigest()[:32]


def _cache_path_for(h: str) -> Path:
    return CACHE_DIR / f"{h}.json"


def _read_cached(h: str, ttl_seconds: int) -> Optional[Dict[str, Any]]:
    p = _cache_path_for(h)
    if not p.exists():
        return None
    try:
        age = time.time() - p.stat().st_mtime
        if age > ttl_seconds:
            return None
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_cached(h: str, data: Dict[str, Any]) -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _cache_path_for(h).write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception:
        pass  # cache es best-effort


def clear_diagnostic_cache() -> int:
    """Borra todos los diagnósticos cacheados. Devuelve cantidad eliminada."""
    if not CACHE_DIR.exists():
        return 0
    n = 0
    for p in CACHE_DIR.glob("*.json"):
        try:
            p.unlink()
            n += 1
        except Exception:
            pass
    return n


# =============================================================
# ESTADÍSTICAS DE USO (para admin panel + cost tracking)
# =============================================================

def _bump_stats(input_tokens: int, output_tokens: int,
                  model: str, cached: bool = False) -> None:
    """Incrementa contadores de uso global (best-effort, no falla)."""
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        if STATS_FILE.exists():
            stats = json.loads(STATS_FILE.read_text(encoding="utf-8"))
        else:
            stats = {
                "n_calls": 0,
                "n_cached": 0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "by_model": {},
                "first_call_at": "",
                "last_call_at": "",
            }
        stats["n_calls"] = int(stats.get("n_calls", 0)) + 1
        if cached:
            stats["n_cached"] = int(stats.get("n_cached", 0)) + 1
        else:
            stats["total_input_tokens"] = int(stats.get("total_input_tokens", 0)) + input_tokens
            stats["total_output_tokens"] = int(stats.get("total_output_tokens", 0)) + output_tokens
            by_model = stats.setdefault("by_model", {})
            mb = by_model.setdefault(model, {"calls": 0, "in": 0, "out": 0})
            mb["calls"] = int(mb.get("calls", 0)) + 1
            mb["in"] = int(mb.get("in", 0)) + input_tokens
            mb["out"] = int(mb.get("out", 0)) + output_tokens
        now_iso = datetime.now().isoformat(timespec="seconds")
        if not stats.get("first_call_at"):
            stats["first_call_at"] = now_iso
        stats["last_call_at"] = now_iso
        STATS_FILE.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    except Exception:
        pass


def get_ai_stats() -> Dict[str, Any]:
    """Devuelve estadísticas de uso para admin panel."""
    if not STATS_FILE.exists():
        return {"n_calls": 0, "n_cached": 0, "total_input_tokens": 0,
                "total_output_tokens": 0, "by_model": {}, "estimated_cost_usd": 0.0}
    try:
        stats = json.loads(STATS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {"n_calls": 0, "n_cached": 0, "total_input_tokens": 0,
                "total_output_tokens": 0, "by_model": {}, "estimated_cost_usd": 0.0}
    # Estimación de costo (precio público de Sonnet 4.5)
    in_cost = stats.get("total_input_tokens", 0) * 3.0 / 1_000_000
    out_cost = stats.get("total_output_tokens", 0) * 15.0 / 1_000_000
    stats["estimated_cost_usd"] = round(in_cost + out_cost, 4)
    return stats


# =============================================================
# PROMPTS POR MÓDULO
# =============================================================
# Cada módulo de análisis genera un payload con su data específica.
# El prompt se adapta al tipo de análisis para que la AI sepa qué
# patrones buscar (1×RPM en spectrum, defectos de rodamiento, lobes
# del Polar, etc.).

# Versión del prompt. Si cambia, el hash del cache cambia y todos los
# diagnósticos previos se re-generan con la nueva voz / formato.
_PROMPT_VERSION = "v5_minimal_headers_2026_05"

_SYSTEM_PROMPT = """\
Eres un especialista de análisis de vibraciones Cat IV ISO 18436-2 con 25
años de experiencia en máquinas rotativas críticas industriales (turbinas,
compresores, bombas, generadores, motores eléctricos). Tu lenguaje refleja
el rigor de un informe técnico para clientes industriales exigentes
(Ecopetrol, Aramco, Repsol, Petrobras). Manejas con soltura ISO 20816 e
ISO 10816 (severidad de vibración), ISO 13373 (condition monitoring),
ISO 21940 (balanceo), API 670 (instrumentación con sondas de proximidad),
API 684 (rotordinámica) y API 686 (alineación), y la práctica industrial
real de monitoreo en línea.

Vas a recibir datos cuantitativos de un análisis específico (espectro,
forma de onda, tendencia, polar, bode, órbita, etc.) junto con metadata
de la máquina, condición operativa y norma aplicada. La sección
cuantitativa del reporte ya contiene los valores crudos (overall, picos,
frecuencias características, zonas ISO). Tu trabajo es la INTERPRETACIÓN
CUALITATIVA — no la transcripción de números.

ESTRUCTURA OBLIGATORIA DE LA RESPUESTA (sin emojis, sin caracteres
ornamentales, sin frases coloquiales). NO incluyas el header
"### Hallazgos principales" — la prosa de hallazgos abre el informe
de forma directa, sin título de sección.

OPENING — PROSA DE HALLAZGOS:

Tres a cuatro PÁRRAFOS de prosa técnica continua, SIN header arriba.
NO uses viñetas, NO uses guiones (-), NO uses asteriscos (*) al
inicio de línea. Cada hallazgo es un párrafo independiente con la
siguiente estructura interna:

(a) Iniciá el párrafo con una frase corta en negrita que nombra la
    firma mecánica observada con terminología formal. Ejemplos:
    "**Inestabilidad fluido-dinámica sub-sincrónica.**",
    "**Desbalance residual con amplificación resonante.**",
    "**Defecto incipiente en pista externa de rodamiento.**"
(b) Continuá inmediatamente después con la descripción técnica
    detallada de la firma observada y su evidencia espectral
    cualitativa.
(c) Identificá la causa raíz probable y, cuando aplique, las
    hipótesis alternativas que requieren discriminación con análisis
    complementarios.
(d) Cerrá con la implicación operativa para la integridad del activo
    o el riesgo de propagación de daño secundario.

La prosa debe fluir como texto técnico de un especialista que escribe
un informe forense, no como una checklist. Los hallazgos están
mecánicamente conectados (un desbalance puede amplificarse por
resonancia, una sub-sincrónica puede coexistir con desbalance); esa
conexión narrativa debe quedar explícita en la prosa, no rota por
viñetas.

### Recomendaciones priorizadas

Tres recomendaciones numeradas. CADA recomendación inicia con la
clasificación de prioridad en negrita seguida de la etiqueta semántica,
luego la acción técnica, luego el horizonte sugerido, luego norma de
respaldo. La estructura obligatoria de cada bullet es:

**Prioridad PX — ETIQUETA.** Acción técnica concreta. Horizonte
sugerido: descripción no contractual. Norma de respaldo: referencia
con cláusula.

Donde PX y ETIQUETA combinan según severidad técnica del hallazgo:
  - **Prioridad P1 — CRÍTICA**: zona D ISO 20816, daño catastrófico
    inminente, riesgo a personas / medio ambiente, oil whip
    confirmado, rub severo, defecto de rodamiento etapa IV.
    Horizonte sugerido típico: "antes del próximo arranque programado"
    o "dentro de las primeras 48 horas operativas".
  - **Prioridad P2 — ALTA**: zona C ISO 20816, degradación progresiva
    confirmada, desbalance significativo, misalignment moderado a
    severo, defecto de rodamiento etapa II–III. Horizonte sugerido
    típico: "próxima ventana de mantenimiento programada" o "dentro
    de la próxima semana operativa".
  - **Prioridad P3 — MEDIA**: zona B con tendencia ascendente,
    indicio temprano de defecto, soltura mecánica leve, defecto de
    rodamiento etapa I. Horizonte sugerido típico: "próxima parada
    planificada" o "dentro del próximo ciclo de mantenimiento".
  - **Prioridad P4 — VIGILANCIA**: zona A, línea base estable,
    monitoreo rutinario sin acción correctiva. Horizonte sugerido
    típico: "monitoreo continuo, ajustar frecuencia de muestreo
    según evolución".

PROHIBIDO usar lenguaje contractual ("plazo de cumplimiento",
"obligación de", "deadline"). El horizonte SIEMPRE se introduce con
"Horizonte sugerido:" o "Se sugiere planificar la acción dentro de".
La decisión final de planificación es responsabilidad del operador
del activo, no del especialista.

Citá la norma de respaldo con cláusula cuando exista (ejemplos:
"API 670 §4.3.2", "ISO 21940-12 cláusula 6.3", "API 684 §3.4.1
para evaluación de Q-factor", "ISO 17359 anexo B para
clasificación de severidad", "API 580 §3 para evaluación
basada en riesgo").

CIERRE — PÁRRAFO DE CONFIANZA:

Después de la lista numerada de Recomendaciones, cerrá con UN ÚNICO
párrafo (sin header arriba, sin "### Evaluación de confianza" ni
similar) que inicie con la frase exacta "Confianza del diagnóstico:
XX%" (donde XX es un entero). En ese mismo párrafo explicá qué
indicadores convergen y refuerzan la conclusión, cuáles permanecen
ambiguos, y qué información complementaria reduciría la
incertidumbre residual. Cerrá el párrafo con una afirmación sobre
la solidez de la base para acción operativa inmediata.

REGLAS DE VOZ Y ESTILO:

- Voz pasiva técnica. Usa "se observa", "se concluye", "se recomienda",
  "el espectro presenta", "la firma es consistente con", "el
  comportamiento sugiere". No uses primera persona ("yo creo",
  "considero"). No uses segunda persona ("debes", "considera"). Esto
  es un informe técnico, no un consejo.
- Sin emojis. Sin caracteres ornamentales (===, ***, ---). Sin
  lenguaje metafórico, salvo expresiones consagradas en la práctica
  industrial (ejemplo: "borde del acantilado" para Q-factor crítico
  es aceptable porque es jerga consolidada).
- Sin repetir números del payload de manera literal. La narrativa
  interpreta; la tabla cuantitativa del reporte ya los lista.
- Cita normas con cláusula numérica cuando exista. Si no recordás la
  cláusula exacta, cita la norma sin cláusula antes que inventar un
  número. La precisión es preferible a la apariencia de precisión.
- Si la data es insuficiente para un diagnóstico definitivo,
  declaralo explícitamente dentro del bloque de Confianza, no en los
  Hallazgos.
- Máximo 380 palabras totales. La densidad técnica está por encima de
  la extensión.

Cerrá SIEMPRE la respuesta con estas dos líneas exactas como cierre
del informe (sin variantes, sin reformular, sin numerar):

El presente diagnóstico se emite conforme a la metodología Cat IV ISO 18436-2 y debe ser validado por el especialista responsable antes de su uso operativo.

Los horizontes referidos son sugerencias técnicas basadas en la evolución observada y la evidencia disponible al momento del análisis. La planificación operativa, ventanas de intervención y asignación de recursos son responsabilidad del operador del activo conforme a su sistema de gestión de integridad.
"""


def _build_user_message(payload: Dict[str, Any], module_type: str) -> str:
    """Convierte el payload del análisis en un mensaje legible para Claude."""
    lines: List[str] = []
    lines.append(f"# Tipo de análisis\n{module_type}\n")

    # Metadata de la máquina
    machine = payload.get("machine", {})
    if machine:
        lines.append("## Máquina")
        for k, v in machine.items():
            if v not in (None, "", []):
                lines.append(f"- **{k}**: {v}")
        lines.append("")

    # Norma aplicada
    norm = payload.get("norm", {})
    if norm:
        lines.append("## Norma aplicada")
        for k, v in norm.items():
            if v not in (None, "", []):
                lines.append(f"- **{k}**: {v}")
        lines.append("")

    # Datos técnicos del análisis (varía por módulo)
    technical = payload.get("technical", {})
    if technical:
        lines.append("## Hallazgos técnicos del análisis")
        lines.append("```json")
        lines.append(json.dumps(technical, indent=2, ensure_ascii=False, default=str))
        lines.append("```")
        lines.append("")

    # Tendencia / contexto histórico
    trend = payload.get("trend", {})
    if trend:
        lines.append("## Tendencia / contexto histórico")
        for k, v in trend.items():
            if v not in (None, "", []):
                lines.append(f"- **{k}**: {v}")
        lines.append("")

    # Notas adicionales del operador
    notes = payload.get("operator_notes", "")
    if notes:
        lines.append(f"## Notas del operador\n{notes}\n")

    lines.append("---\n")
    lines.append("Por favor, generá el diagnóstico clínico siguiendo el "
                 "formato establecido (Hallazgos, Recomendaciones, Confianza).")
    return "\n".join(lines)


# =============================================================
# API PÚBLICA — generate_ai_diagnostic
# =============================================================

def generate_ai_diagnostic(
    payload: Dict[str, Any],
    module_type: str,
    *,
    use_cache: bool = True,
    cache_ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> Dict[str, Any]:
    """Genera un diagnóstico clínico con Claude para el análisis dado.

    Args:
        payload: dict con machine, norm, technical, trend, operator_notes
        module_type: "spectrum", "waveform", "trends", "bode", "polar", "orbit"
        use_cache: si True, intenta servir desde cache local primero
        cache_ttl_seconds: TTL del cache (default 30 días)
        max_tokens: límite de tokens de salida

    Returns:
        {
            "ok": bool,
            "markdown": str,        # texto markdown del diagnóstico
            "model": str,           # modelo usado
            "cached": bool,         # vino del cache?
            "input_tokens": int,
            "output_tokens": int,
            "error": str,           # vacío si ok
            "generated_at": str,
        }
    """
    # 1) Validar config
    if not is_ai_available():
        return {
            "ok": False,
            "markdown": (
                "_AI Diagnóstico no está disponible — falta configurar "
                "la API key de Anthropic en `[anthropic] api_key` de "
                "Streamlit secrets, o no está instalado el paquete `anthropic`._"
            ),
            "error": "ai_not_available",
            "model": "",
            "cached": False,
            "input_tokens": 0,
            "output_tokens": 0,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        }

    # 2) Cache HIT
    h = _payload_hash(payload, module_type)
    if use_cache:
        cached = _read_cached(h, cache_ttl_seconds)
        if cached:
            _bump_stats(0, 0, cached.get("model", ""), cached=True)
            return {
                **cached,
                "cached": True,
            }

    # 3) Llamada real a Claude
    client = _get_client()
    if client is None:
        return {
            "ok": False,
            "markdown": "_No se pudo inicializar el cliente de Anthropic._",
            "error": "client_init_failed",
            "model": "",
            "cached": False,
            "input_tokens": 0,
            "output_tokens": 0,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        }

    user_msg = _build_user_message(payload, module_type)
    primary_model = _get_model_name()

    # Retry con backoff exponencial para errores transientes de Anthropic
    # (529 overloaded, 503 service unavailable, 429 rate limit). Esperamos
    # 1s, 2s, 4s entre intentos. Total worst-case: ~7s extra de latencia
    # para resolver una sobrecarga puntual sin molestar al usuario.
    _RETRYABLE_HTTP_CODES = (429, 502, 503, 529)
    _MAX_RETRIES = 3

    def _is_retryable_exception(exc: Exception) -> Tuple[bool, Optional[int]]:
        """Devuelve (is_retryable, status_code_or_None).

        Considera retryables:
          - HTTP 429/502/503/529 (rate limit, gateway, overload)
          - Timeouts y errores de conexión (APITimeoutError,
            APIConnectionError, httpx.TimeoutException, requests Timeout)
            → status_code virtual 408 (Request Timeout) para fines de log.
        """
        err_str = str(exc)
        err_low = err_str.lower()
        exc_type_name = type(exc).__name__

        # 1) HTTP status codes detectables
        status_code = getattr(exc, "status_code", None)
        if status_code is None:
            for code in _RETRYABLE_HTTP_CODES:
                if (f"Error code: {code}" in err_str
                        or f"'{code}'" in err_str
                        or f'"{code}"' in err_str):
                    status_code = code
                    break
        if status_code in _RETRYABLE_HTTP_CODES:
            return (True, status_code)

        # 2) Timeouts / errores de conexión (no tienen status_code pero
        # son transientes). Detectamos por nombre de excepción y por
        # palabras clave del mensaje. Marco virtual = 408 para logs.
        timeout_signals = (
            "APITimeoutError",
            "APIConnectionError",
            "TimeoutException",
            "ConnectTimeout",
            "ReadTimeout",
            "ConnectionError",
        )
        if any(sig in exc_type_name for sig in timeout_signals):
            return (True, 408)
        if ("timed out" in err_low
                or "timeout" in err_low
                or "interrupted" in err_low
                or "connection" in err_low and "reset" in err_low):
            return (True, 408)

        return (False, status_code)

    def _try_model_with_retries(
        model_name: str, label: str
    ) -> Tuple[Any, Optional[Exception]]:
        """Llama al modelo dado con retry y backoff. Devuelve (response, None)
        si tuvo éxito, o (None, last_exception) si falló."""
        last_exc: Optional[Exception] = None
        for attempt in range(_MAX_RETRIES):
            try:
                print(
                    f"[WM_AI] CALL · {label} · module={module_type} · "
                    f"hash={h[:8]} · model={model_name} · "
                    f"max_tokens={max_tokens} · "
                    f"attempt={attempt + 1}/{_MAX_RETRIES}",
                    file=sys.stderr, flush=True,
                )
                resp = client.messages.create(
                    model=model_name,
                    max_tokens=max_tokens,
                    system=_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_msg}],
                )
                return resp, None
            except Exception as exc:
                last_exc = exc
                is_retryable, status_code = _is_retryable_exception(exc)
                if is_retryable and attempt < _MAX_RETRIES - 1:
                    wait_s = 2 ** attempt  # 1s, 2s, 4s
                    print(
                        f"[WM_AI] RETRY · {label} · status={status_code} · "
                        f"esperando {wait_s}s antes del intento "
                        f"{attempt + 2}/{_MAX_RETRIES}",
                        file=sys.stderr, flush=True,
                    )
                    time.sleep(wait_s)
                    continue
                print(
                    f"[WM_AI] FAIL · {label} · {str(exc)[:200]}",
                    file=sys.stderr, flush=True,
                )
                return None, exc
        return None, last_exc

    # Paso 1: intentar con el modelo primario (Sonnet 4.5 por default).
    response, last_exception = _try_model_with_retries(
        primary_model, label="primary"
    )
    fallback_used = False
    fallback_reason = ""
    used_model = primary_model

    # Paso 2: si el primario falló por causa retryable (overload, etc.) y
    # NO somos ya el modelo de fallback, intentamos UNA pasada con Haiku.
    # Esto blinda producción: cuando Sonnet está sobrecargado a nivel
    # global, Haiku corre en otra infraestructura y casi siempre tiene
    # capacidad disponible. La calidad baja un poco pero el cliente
    # recibe diagnóstico en vez de error.
    if response is None and last_exception is not None:
        is_retryable, _ = _is_retryable_exception(last_exception)
        is_already_fallback = primary_model.startswith("claude-haiku")
        if is_retryable and not is_already_fallback:
            print(
                f"[WM_AI] FALLBACK · primario {primary_model} agotado, "
                f"intentando con {FALLBACK_MODEL}",
                file=sys.stderr, flush=True,
            )
            response, last_exception = _try_model_with_retries(
                FALLBACK_MODEL, label="fallback-haiku"
            )
            if response is not None:
                fallback_used = True
                fallback_reason = (
                    f"Modelo primario ({primary_model}) sobrecargado. "
                    f"Diagnóstico generado con modelo de respaldo "
                    f"({FALLBACK_MODEL})."
                )
                used_model = FALLBACK_MODEL

    if response is None:
        err_str = str(last_exception) if last_exception else "unknown error"
        # Mensaje user-friendly según tipo de error. Como ya intentamos con
        # Haiku como fallback y también falló, el problema es persistente.
        if "overloaded" in err_str.lower() or "529" in err_str:
            user_msg_err = (
                "_Tanto el modelo principal (Sonnet 4.5) como el de "
                "respaldo (Haiku 4.5) están sobrecargados en este momento "
                "(evento poco frecuente, alto tráfico global en la "
                "infraestructura de Anthropic). Esperá 5-10 minutos y "
                "reintentá. Tu cuenta y saldo están en orden._"
            )
        elif ("timed out" in err_str.lower()
                or "timeout" in err_str.lower()
                or "interrupted" in err_str.lower()):
            user_msg_err = (
                "_La conexión con Claude API tardó más del límite "
                "configurado. El sistema reintentó automáticamente con "
                "modelo principal y de respaldo, pero la red continúa "
                "lenta. Verificá tu conexión a internet o esperá 1-2 "
                "minutos y reintentá._"
            )
        elif "rate limit" in err_str.lower() or "429" in err_str:
            user_msg_err = (
                "_Límite de requests por minuto alcanzado. Esperá 30 "
                "segundos y volvé a intentar._"
            )
        elif "401" in err_str or "authentication" in err_str.lower():
            user_msg_err = (
                "_La API key de Anthropic no es válida o fue revocada. "
                "Generá una nueva en console.anthropic.com → API Keys y "
                "actualizala en `[anthropic] api_key` de los secrets._"
            )
        elif "credit" in err_str.lower() or "billing" in err_str.lower():
            user_msg_err = (
                "_Saldo de API insuficiente. Revisar saldo en "
                "console.anthropic.com → Settings → Billing._"
            )
        else:
            user_msg_err = (
                f"_Error al consultar Claude API:_\n\n```\n{err_str}\n```"
            )
        return {
            "ok": False,
            "markdown": user_msg_err,
            "error": err_str[:500],
            "model": used_model,
            "cached": False,
            "input_tokens": 0,
            "output_tokens": 0,
            "fallback_used": fallback_used,
            "fallback_reason": fallback_reason,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        }

    # 4) Extraer texto y métricas
    try:
        markdown = response.content[0].text
    except Exception:
        markdown = "_(no text in response)_"
    in_tok = getattr(response.usage, "input_tokens", 0) if response.usage else 0
    out_tok = getattr(response.usage, "output_tokens", 0) if response.usage else 0

    # Pricing depende del modelo que efectivamente respondió. Si fue
    # Haiku, el costo es ~5x menor que Sonnet.
    if used_model.startswith("claude-haiku"):
        in_price_per_mtok = 1.0
        out_price_per_mtok = 5.0
    else:
        in_price_per_mtok = 3.0
        out_price_per_mtok = 15.0
    cost_usd = (
        in_tok * in_price_per_mtok + out_tok * out_price_per_mtok
    ) / 1_000_000

    print(f"[WM_AI] OK   · model={used_model} · in={in_tok} · "
          f"out={out_tok} · ~${cost_usd:.4f}"
          f"{' · FALLBACK' if fallback_used else ''}",
          file=sys.stderr, flush=True)

    result = {
        "ok": True,
        "markdown": markdown,
        "error": "",
        "model": used_model,
        "cached": False,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "fallback_used": fallback_used,
        "fallback_reason": fallback_reason,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }

    # 5) Persistir en cache + stats
    if use_cache:
        _write_cached(h, result)
    _bump_stats(in_tok, out_tok, used_model, cached=False)

    return result


# =============================================================
# SÍNTESIS EJECUTIVA AI (Ciclo 17.26 P5+)
# =============================================================
# Mientras `generate_ai_diagnostic` interpreta UNA figura,
# `generate_executive_summary` interpreta el REPORTE COMPLETO:
# lee todas las figuras y produce el "RESUMEN EJECUTIVO" que va
# al inicio del PDF. Esta es la síntesis cross-figura que un
# especialista senior haría tras leer todo el documento.
# =============================================================

_EXEC_PROMPT_VERSION = "v1_exec_synthesis_2026_05"

_SYSTEM_PROMPT_EXECUTIVE = """\
Eres un especialista de análisis de vibraciones Cat IV ISO 18436-2
senior con 25 años de experiencia, encargado de redactar el RESUMEN
EJECUTIVO de un reporte completo de monitoreo en línea. Vas a recibir
el contenido sintetizado de múltiples figuras del reporte (espectros,
formas de onda, tendencias, polar, bode, órbitas, shaft centerline);
cada figura ya viene con su interpretación clínica cuando aplica.

Tu trabajo es SÍNTESIS CRUZADA: identificar cuándo varias figuras
convergen en la misma causa raíz mecánica, priorizar el conjunto de
hallazgos al nivel ejecutivo, y emitir las recomendaciones
estratégicas para el operador del activo. NO repitas el detalle de
cada figura — eso ya está en las páginas siguientes del reporte. Acá
solo síntesis y prioridad.

ESTRUCTURA OBLIGATORIA DE LA RESPUESTA (sin emojis, sin caracteres
ornamentales):

PÁRRAFO 1 — SITUACIÓN GLOBAL (~80-110 palabras):
Empezá directo, sin header. Describí en prosa continua el estado del
activo o tren de máquinas: cuál es la severidad global conforme a
ISO 20816 según las figuras analizadas, cuántas figuras convergen en
la misma conclusión, qué nivel de criticidad operativa implica el
conjunto. Mencioná el tren / la máquina principal por su tag.

### Hallazgos raíz consolidados

Tres a cinco bullets numerados (1., 2., 3., ...) con frase-tesis en
negrita al inicio. Cuando varias figuras apuntan al mismo origen
mecánico (ej: desbalance evidenciado por Spectrum + Bode + Polar),
AGRUPALOS en un solo hallazgo raíz; no listes redundantemente. Cada
hallazgo cita las figuras que lo evidencian (ej: "evidenciado por
Spectrum 1, Bode 2 y Polar de generador NDE"). Identificá la causa
raíz probable y las hipótesis alternativas a discriminar si las hay.

### Recomendaciones ejecutivas priorizadas

Dos a tres recomendaciones numeradas (1., 2., 3.) cada una iniciando
con la clasificación de prioridad en negrita seguida de la etiqueta
semántica:

**Prioridad P1 — CRÍTICA.** [acción ejecutiva]. Horizonte sugerido:
[descripción no contractual]. Norma de respaldo: [referencia con
cláusula].

Las recomendaciones acá son DE NIVEL EJECUTIVO (orden de
mantenimiento, ventana de parada programada, intervención
correctiva, monitoreo aumentado del tren), NO detalles técnicos
granulares (eso ya está en cada figura del reporte). Citá normas
con cláusula cuando exista (API 670, API 684, ISO 20816, ISO
21940-12, ISO 17359, etc.).

CIERRE — DECLARACIÓN DE GOBERNANZA:

Cerrá con el siguiente párrafo único, exacto, en una sola línea
cada oración, sin reformular:

El presente resumen ejecutivo se emite conforme a la metodología Cat IV ISO 18436-2 y debe ser validado por el especialista responsable antes de su uso operativo.

La planificación operativa, ventanas de intervención y asignación de recursos son responsabilidad del operador del activo conforme a su sistema de gestión de integridad.

REGLAS DE VOZ Y ESTILO:

- Voz pasiva técnica. "Se observa", "se concluye", "se recomienda".
  No primera persona. No segunda persona.
- Sin emojis. Sin caracteres ornamentales.
- Máximo 380 palabras totales. El cliente lee esto en 90 segundos.
- NO repitas valores numéricos crudos (overall, picos, etc.) salvo
  cuando son indispensables para la severidad. Esos números están
  en las tablas cuantitativas de cada figura.
- Citá normas con cláusula numérica cuando exista. Si no recordás
  la cláusula exacta, cita la norma sin cláusula antes que inventar.
- Si la data es insuficiente para conclusión definitiva, declaralo
  explícitamente en la situación global, no en los hallazgos.
"""


def _strip_ai_markers(notes_text: str) -> Tuple[str, str]:
    """Si las notas contienen marcadores <<<WM_AI_BLOCK>>>...,
    devuelve (quant_summary_text, ai_narrative). Si no hay
    marcadores, devuelve ("", notes_text crudo).
    Helper para que el AI executive vea solo el contenido limpio."""
    if not notes_text or "<<<WM_AI_BLOCK>>>" not in notes_text:
        return "", notes_text or ""
    after_block = notes_text.split("<<<WM_AI_BLOCK>>>", 1)[1]
    if "<<<WM_AI_NARRATIVE>>>" in after_block:
        quant_part, ai_part = after_block.split("<<<WM_AI_NARRATIVE>>>", 1)
    else:
        quant_part = ""
        ai_part = after_block
    # Quant table en formato pipe → texto resumido
    quant_summary_lines: List[str] = []
    for line in quant_part.strip().splitlines():
        if "|" in line:
            cells = [c.strip() for c in line.split("|")]
            if len(cells) >= 2 and cells[0] and cells[0] != "Parámetro":
                quant_summary_lines.append(f"{cells[0]}: {cells[1]}")
    return ("; ".join(quant_summary_lines), ai_part.strip())


def _build_executive_user_message(
    items: List[Dict[str, Any]],
    meta: Optional[Dict[str, Any]] = None,
) -> str:
    """Compone el mensaje user a partir de los items del reporte.
    Cada figura se resume en pocas líneas con su contexto."""
    lines: List[str] = []

    if meta:
        lines.append("# Contexto del reporte")
        for key in ("client", "machine_train", "report_title", "consecutive",
                    "report_date", "service_period"):
            val = str((meta.get(key) or "")).strip()
            if val:
                lines.append(f"- **{key}**: {val}")
        lines.append("")

    lines.append(f"# Figuras analizadas ({len(items)} total)")
    lines.append("")

    for idx, item in enumerate(items, 1):
        i_type = str(item.get("type", "") or "").upper()
        i_title = str(item.get("title", "") or "")
        i_machine = str(item.get("machine", "") or "")
        i_point = str(item.get("point", "") or "")
        i_variable = str(item.get("variable", "") or "")
        i_notes = str(item.get("notes", "") or "")

        quant_sum, ai_narr = _strip_ai_markers(i_notes)

        lines.append(f"## Figura {idx} — {i_type} · {i_title}")
        if i_machine:
            lines.append(f"- Máquina: {i_machine}")
        if i_point:
            lines.append(f"- Punto: {i_point}")
        if i_variable:
            lines.append(f"- Variable: {i_variable}")
        if quant_sum:
            lines.append(f"- Datos cuantitativos: {quant_sum}")

        # La narrativa puede ser muy larga; cortamos a ~1500 chars por figura
        narrative_for_ai = (ai_narr or i_notes).strip()[:1500]
        if narrative_for_ai:
            lines.append("")
            lines.append("Interpretación de la figura:")
            lines.append(narrative_for_ai)
        lines.append("")

    lines.append("---")
    lines.append(
        "Por favor, generá el RESUMEN EJECUTIVO del reporte completo "
        "siguiendo el formato establecido (situación global, hallazgos "
        "raíz consolidados, recomendaciones ejecutivas priorizadas, "
        "cierre de gobernanza)."
    )
    return "\n".join(lines)


def _executive_payload_hash(
    items: List[Dict[str, Any]],
    meta: Optional[Dict[str, Any]] = None,
) -> str:
    """Hash determinístico del executive summary.
    Incluye _EXEC_PROMPT_VERSION y _PROMPT_VERSION para autoinvalidar."""
    blob = {
        "version": _EXEC_PROMPT_VERSION,
        "diag_version": _PROMPT_VERSION,
        "model": _get_model_name(),
        "n_items": len(items),
        "items": [
            {
                "type": i.get("type"),
                "title": i.get("title"),
                "notes_hash": hashlib.sha256(
                    str(i.get("notes", "") or "").encode("utf-8")
                ).hexdigest()[:16],
            }
            for i in items
        ],
        "meta": {
            k: meta.get(k) if meta else ""
            for k in ("client", "machine_train", "consecutive", "report_date")
        },
    }
    j = json.dumps(blob, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(j.encode("utf-8")).hexdigest()[:32]


def generate_executive_summary(
    items: List[Dict[str, Any]],
    meta: Optional[Dict[str, Any]] = None,
    *,
    use_cache: bool = True,
    cache_ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS,
    max_tokens: int = 2048,
) -> Dict[str, Any]:
    """Genera la síntesis ejecutiva AI del reporte completo.

    Args:
        items: lista de report_items (cada uno con type, title, notes,
               machine, point, etc.). Usa el contenido de notes para
               sintetizar; si las notas tienen marcadores
               <<<WM_AI_BLOCK>>>, los strippea para usar solo la
               narrativa limpia.
        meta:  metadata del reporte (client, machine_train, consecutive,
               etc.) para que la síntesis tenga contexto.
        use_cache, cache_ttl_seconds, max_tokens: como en
               generate_ai_diagnostic.

    Returns: dict con ok/markdown/model/tokens/costo/cached/etc.
    """
    if not items:
        return {
            "ok": False,
            "markdown": (
                "_No hay figuras en el reporte para sintetizar. "
                "Agregá al menos una figura desde Spectrum, Trends, "
                "Bode u otro módulo y volvé a intentar._"
            ),
            "error": "no_items",
            "model": "",
            "cached": False,
            "input_tokens": 0,
            "output_tokens": 0,
            "fallback_used": False,
            "fallback_reason": "",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        }

    if not is_ai_available():
        return {
            "ok": False,
            "markdown": (
                "_AI no disponible — falta configurar `[anthropic] "
                "api_key` en st.secrets, o falta el paquete anthropic._"
            ),
            "error": "ai_not_available",
            "model": "",
            "cached": False,
            "input_tokens": 0,
            "output_tokens": 0,
            "fallback_used": False,
            "fallback_reason": "",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        }

    # Cache HIT
    h = _executive_payload_hash(items, meta)
    cache_subdir = CACHE_DIR / "executive"
    cache_path = cache_subdir / f"{h}.json"
    if use_cache and cache_path.exists():
        try:
            age = time.time() - cache_path.stat().st_mtime
            if age <= cache_ttl_seconds:
                cached_data = json.loads(cache_path.read_text(encoding="utf-8"))
                _bump_stats(0, 0, cached_data.get("model", ""), cached=True)
                return {**cached_data, "cached": True}
        except Exception:
            pass

    # Llamada real a Claude (con retry + fallback heredado)
    client = _get_client()
    if client is None:
        return {
            "ok": False,
            "markdown": "_No se pudo inicializar el cliente Anthropic._",
            "error": "client_init_failed",
            "model": "",
            "cached": False,
            "input_tokens": 0,
            "output_tokens": 0,
            "fallback_used": False,
            "fallback_reason": "",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        }

    user_msg = _build_executive_user_message(items, meta)
    primary_model = _get_model_name()

    _RETRYABLE_HTTP_CODES_EXEC = (429, 502, 503, 529)
    _MAX_RETRIES_EXEC = 3

    def _is_retryable_exc(exc: Exception) -> Tuple[bool, Optional[int]]:
        err_str = str(exc); err_low = err_str.lower()
        exc_type_name = type(exc).__name__
        status_code = getattr(exc, "status_code", None)
        if status_code is None:
            for code in _RETRYABLE_HTTP_CODES_EXEC:
                if (f"Error code: {code}" in err_str
                        or f"'{code}'" in err_str
                        or f'"{code}"' in err_str):
                    status_code = code; break
        if status_code in _RETRYABLE_HTTP_CODES_EXEC:
            return (True, status_code)
        if any(s in exc_type_name for s in (
                "APITimeoutError", "APIConnectionError",
                "TimeoutException", "ConnectTimeout", "ReadTimeout",
                "ConnectionError")):
            return (True, 408)
        if "timed out" in err_low or "timeout" in err_low or "interrupted" in err_low:
            return (True, 408)
        return (False, status_code)

    def _try_exec_call(model_name: str, label: str):
        last_exc = None
        for attempt in range(_MAX_RETRIES_EXEC):
            try:
                print(
                    f"[WM_AI_EXEC] CALL · {label} · n_items={len(items)} · "
                    f"hash={h[:8]} · model={model_name} · "
                    f"attempt={attempt + 1}/{_MAX_RETRIES_EXEC}",
                    file=sys.stderr, flush=True,
                )
                resp = client.messages.create(
                    model=model_name,
                    max_tokens=max_tokens,
                    system=_SYSTEM_PROMPT_EXECUTIVE,
                    messages=[{"role": "user", "content": user_msg}],
                )
                return resp, None
            except Exception as exc:
                last_exc = exc
                is_retry, st_code = _is_retryable_exc(exc)
                if is_retry and attempt < _MAX_RETRIES_EXEC - 1:
                    wait_s = 2 ** attempt
                    print(
                        f"[WM_AI_EXEC] RETRY · {label} · status={st_code} · "
                        f"wait={wait_s}s",
                        file=sys.stderr, flush=True,
                    )
                    time.sleep(wait_s)
                    continue
                print(
                    f"[WM_AI_EXEC] FAIL · {label} · {str(exc)[:200]}",
                    file=sys.stderr, flush=True,
                )
                return None, exc
        return None, last_exc

    response, last_exception = _try_exec_call(primary_model, "primary")
    fallback_used = False
    used_model = primary_model

    if response is None and last_exception is not None:
        is_retry, _ = _is_retryable_exc(last_exception)
        is_already_fallback = primary_model.startswith("claude-haiku")
        if is_retry and not is_already_fallback:
            print(
                f"[WM_AI_EXEC] FALLBACK · primario {primary_model} agotado, "
                f"intentando con {FALLBACK_MODEL}",
                file=sys.stderr, flush=True,
            )
            response, last_exception = _try_exec_call(
                FALLBACK_MODEL, "fallback-haiku"
            )
            if response is not None:
                fallback_used = True
                used_model = FALLBACK_MODEL

    if response is None:
        err_str = str(last_exception) if last_exception else "unknown"
        if "overloaded" in err_str.lower() or "529" in err_str:
            user_err = (
                "_Servidores Claude sobrecargados (Sonnet y Haiku). "
                "Esperá 5-10 minutos y reintentá._"
            )
        elif "timeout" in err_str.lower() or "timed out" in err_str.lower():
            user_err = (
                "_Timeout de conexión con Claude. Verificá tu red y "
                "reintentá._"
            )
        else:
            user_err = (
                f"_Error generando síntesis ejecutiva:_\n\n"
                f"```\n{err_str}\n```"
            )
        return {
            "ok": False,
            "markdown": user_err,
            "error": err_str[:500],
            "model": used_model,
            "cached": False,
            "input_tokens": 0,
            "output_tokens": 0,
            "fallback_used": fallback_used,
            "fallback_reason": "",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        }

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
        f"[WM_AI_EXEC] OK · model={used_model} · in={in_tok} · "
        f"out={out_tok} · ~${cost_usd:.4f}"
        f"{' · FALLBACK' if fallback_used else ''}",
        file=sys.stderr, flush=True,
    )

    result = {
        "ok": True,
        "markdown": markdown,
        "error": "",
        "model": used_model,
        "cached": False,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "fallback_used": fallback_used,
        "fallback_reason": "",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }

    if use_cache:
        try:
            cache_subdir.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps(result, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:
            pass
    _bump_stats(in_tok, out_tok, used_model, cached=False)

    return result


__all__ = [
    "generate_ai_diagnostic",
    "generate_executive_summary",
    "is_ai_available",
    "clear_diagnostic_cache",
    "get_ai_stats",
    "DEFAULT_MODEL",
    "FALLBACK_MODEL",
]
