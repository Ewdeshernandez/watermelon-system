"""
core/signature_render.py
========================

Genera una FIRMA cursiva (PNG transparente) a partir del nombre del firmante,
server-side con PIL + fuente Great Vibes (OFL, en assets/fonts). Se usa para
poner la firma sobre "Preparado por" / "Revisado por" en el reporte, sin
depender de subir imágenes ni de un navegador headless (funciona en Cloud).
"""
from __future__ import annotations

import logging
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

_FONT_PATH = Path(__file__).resolve().parent.parent / "assets" / "fonts" / "GreatVibes-Regular.ttf"
_INK = (18, 48, 94, 255)  # #12305e navy


@lru_cache(maxsize=64)
def render_signature_png(name: str, size: int = 96) -> Optional[bytes]:
    """Devuelve PNG (bytes) con el nombre en cursiva sobre fondo transparente,
    recortado a su contenido. None si algo falla (el reporte cae a solo texto)."""
    name = (name or "").strip()
    if not name:
        return None
    try:
        from PIL import Image, ImageDraw, ImageFont
        font = ImageFont.truetype(str(_FONT_PATH), size)
        # Lienzo amplio → medir → recortar al bounding box real.
        pad = size // 3
        tmp = Image.new("RGBA", (len(name) * size, size * 2), (0, 0, 0, 0))
        d = ImageDraw.Draw(tmp)
        d.text((pad, pad), name, font=font, fill=_INK)
        bbox = tmp.getbbox()
        if not bbox:
            return None
        x0, y0, x1, y1 = bbox
        crop = tmp.crop((max(0, x0 - pad // 2), max(0, y0 - pad // 2),
                         min(tmp.width, x1 + pad // 2), min(tmp.height, y1 + pad // 2)))
        buf = BytesIO()
        crop.save(buf, format="PNG")
        return buf.getvalue()
    except Exception as e:  # noqa: BLE001
        log.warning("render_signature_png(%r) falló: %s", name, e)
        return None
