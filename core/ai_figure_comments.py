"""
core.ai_figure_comments — Comentario de figura nivel Cat. IV (IA + RAG)

Genera, con IA y apoyándose en el repositorio de conocimiento (cursos +
manuales, vía RAG), un comentario experto por cada figura del briefing
(tendencia, espectro, forma de onda, órbita). Devuelve {key: comentario}.

Blindaje: la IA fundamenta y sintetiza; NO copia texto del material ni inventa
componentes inexistentes (un generador eléctrico no tiene álabes).
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

_TIPO = {"spectrum": "espectro", "waveform": "forma de onda", "orbit": "órbita"}

_SYSTEM = (
    "Eres un analista de vibraciones Categoría IV (ISO 18436-2) comentando las "
    "figuras de un reporte de condición de maquinaria rotativa. Escribes con "
    "criterio experto y te APOYAS en el material de referencia entregado. "
    "REGLAS: (1) 2 a 4 frases por figura, claras y técnicas; (2) cita la "
    "evidencia visible (1X/2X y armónicos, bandas laterales, forma de la órbita, "
    "factor de cresta, tendencia) y relaciónala con el mecanismo probable "
    "(desbalance, desalineación, holgura, roce, rodamiento/engrane, "
    "inestabilidad hidrodinámica); (3) NO inventes componentes que no existen "
    "en esa máquina (un GENERADOR eléctrico no tiene álabes ni compresor); "
    "(4) NO copies texto literal del material (es material con derechos: "
    "sintetiza con tus palabras); (5) si la figura no muestra anomalía, dilo. "
    "Responde EXCLUSIVAMENTE un objeto JSON {\"<key>\": \"comentario\", ...} "
    "usando las 'key' EXACTAS que te doy."
)


def generate_figure_comments(machine_train: str, machine_ctx: str,
                             figures_spec: Dict[str, Any],
                             model: Optional[str] = None) -> Dict[str, str]:
    """figures_spec: {'trends':[{'key','label','seed'}...], 'spectrum':bool,
    'spectrum_seed':str, 'waveform':..., 'orbit':...}. Devuelve {key: texto}.
    {} si no hay IA o falla (el caller conserva los comentarios base)."""
    try:
        from core.ai_diagnostic import _get_client, _get_model_name, is_ai_available
    except Exception:
        return {}
    if not is_ai_available():
        return {}
    client = _get_client()
    if client is None:
        return {}

    figs: List[Dict[str, str]] = []
    for t in figures_spec.get("trends", []) or []:
        figs.append({"key": t["key"], "tipo": "tendencia",
                     "figura": t.get("label", "Tendencia"),
                     "observacion_base": t.get("seed", "")})
    for k in ("spectrum", "waveform", "orbit"):
        if figures_spec.get(k):
            figs.append({"key": k, "tipo": _TIPO[k], "figura": _TIPO[k],
                         "observacion_base": figures_spec.get(f"{k}_seed", "")})
    if not figs:
        return {}

    user = (f"MÁQUINA (tren): {machine_train}\n\n"
            f"{machine_ctx}\n\n"
            f"FIGURAS A COMENTAR (usa la 'key' exacta en el JSON de salida):\n"
            f"{json.dumps(figs, ensure_ascii=False, indent=1)}\n\n"
            "Devuelve un JSON con un comentario nivel Cat. IV por cada 'key'.")
    import time
    _mdl = model or _get_model_name()
    for attempt in range(3):
        try:
            resp = client.messages.create(
                model=_mdl, max_tokens=1600, system=_SYSTEM,
                messages=[{"role": "user", "content": user}],
                timeout=90.0)
            txt = "".join(getattr(b, "text", "") for b in resp.content
                          if getattr(b, "type", "") == "text")
            m = re.search(r"\{.*\}", txt, re.S)
            if not m:
                return {}
            data = json.loads(m.group(0))
            return {str(k): str(v).strip() for k, v in data.items()
                    if str(v).strip()}
        except Exception as e:
            log.warning("generate_figure_comments intento %d/3 falló: %s",
                        attempt + 1, e)
            if attempt < 2:
                time.sleep(2.0 * (attempt + 1))
                continue
            return {}
    return {}


__all__ = ["generate_figure_comments"]
