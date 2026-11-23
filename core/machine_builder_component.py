"""
core/machine_builder_component.py
=================================

Componente Streamlit bidireccional que embebe el constructor de tren del campo
(native/field_ui/builder.html) — fuente ÚNICA para web y campo. El shim
native/field_ui/index.html traduce el puente postMessage del builder al
protocolo de componentes de Streamlit.

    value = machine_builder(spec, nonce="SGT300B:v3", key="mb")
    # → None hasta que el usuario pulsa "Construir máquina";
    #   luego {"spec": {...}, "ts": 1732300000000}
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit.components.v1 as components

_DIR = Path(__file__).resolve().parents[1] / "native" / "field_ui"
_component = components.declare_component("wm_machine_builder", path=str(_DIR))


def machine_builder(spec: Optional[Dict[str, Any]], nonce: str, key: str,
                    skin: str = "light") -> Optional[Dict[str, Any]]:
    """Dibuja el constructor. `spec` = tren a cargar (layout guardado); se
    re-carga solo cuando cambia `nonce` (máquina / versión seleccionada)."""
    return _component(page="builder.html", spec=spec or {}, nonce=nonce, skin=skin,
                      key=key, default=None)


def machine_config(msgs: List[Dict[str, Any]], nonce: str, key: str,
                   skin: str = "light") -> Optional[Dict[str, Any]]:
    """Dibuja la configuración de 7 pasos (native/field_ui/config.html) —
    la MISMA del campo. `msgs` (host→página, ver config_bridge) se entregan
    cuando cambia `nonce`. Devuelve la última acción {wm:'action', name, setup, ..., ts}."""
    return _component(page="config.html", msgs=msgs, nonce=nonce, skin=skin,
                      key=key, default=None)


def machine_live_train(layout: Dict[str, Any], live: Dict[str, Any], key: str,
                       skin: str = "light") -> None:
    """Tren de Machines EN VIVO (builder.html#live) para el hero de Live
    Monitoring. El tren se carga una vez; los valores llegan en cada rerun por
    postMessage {wm:'live'} sin recargar el iframe (sin parpadeo)."""
    import hashlib
    import json as _json
    nonce = hashlib.sha1(_json.dumps(layout, sort_keys=True).encode()).hexdigest()[:12]
    _component(page="builder.html", mode="live", spec=layout, live=live, nonce=nonce,
               skin=skin, key=key, default=None)

