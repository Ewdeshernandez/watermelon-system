"""
core.train_schematic
====================

Genera schematics PNG genéricos de trenes acoplados de 2 máquinas
(no-reciprocantes). Útil para:

  - Turbomáquinas (turbogenerador, turboexpansor, turbo-compresor)
  - Motor + bomba
  - Motor + compresor centrífugo
  - Cualquier driver + driven con eje horizontal y N cojinetes por lado.

Para reciprocantes ver :mod:`core.recip_schematic` (con cilindros
horizontales opuestos boxer).

API:
    generate_train_png(driver_label, driven_label, n_driver_planes,
                       n_driven_planes) -> bytes (PNG)
    sensor_default_position(sensor_dict, n_driver_planes,
                            n_driven_planes) -> (x_pct, y_pct)

Diseño:
  - Driver (azul) a la izquierda, ocupa ~30% del ancho
  - Acople (ámbar) en el centro
  - Driven (verde) a la derecha, ocupa ~42% del ancho
  - Eje (línea negra) atraviesa ambas máquinas
  - Cojinetes representados como círculos sobre el eje, numerados
    correlativamente (driver 1..N, driven N+1..)
"""

from __future__ import annotations

import io
from typing import Any, Dict, Optional, Tuple

try:
    from PIL import Image, ImageDraw, ImageFont
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False
    Image = None  # type: ignore


# Layout constants (porcentajes del canvas 1400x500)
LAYOUT = {
    "driver_x_pct":  (6, 36),
    "driver_y_pct":  (30, 70),
    "coupling_x_pct": (37, 47),
    "driven_x_pct":  (48, 95),
    "driven_y_pct":  (28, 72),
    "shaft_y_pct":    50,
}


def generate_train_png(
    driver_label: str = "Driver",
    driven_label: str = "Driven",
    n_driver_planes: int = 2,
    n_driven_planes: int = 2,
    width: int = 1400,
    height: int = 500,
    bg_color: str = "#fafbfc",
) -> bytes:
    """
    Genera el schematic de un tren acoplado genérico.

    Devuelve bytes PNG. Si PIL no está disponible, devuelve b"".
    """
    if not _HAS_PIL:
        return b""

    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)

    # Fonts (con fallback graceful)
    try:
        font_big = ImageFont.truetype("Arial Bold.ttf", 28)
    except Exception:
        try:
            font_big = ImageFont.truetype("Arial.ttf", 28)
        except Exception:
            font_big = ImageFont.load_default()
    try:
        font_med = ImageFont.truetype("Arial.ttf", 18)
        font_sm = ImageFont.truetype("Arial.ttf", 13)
    except Exception:
        font_med = font_big
        font_sm = font_big

    def pct_xy(x_pct: float, y_pct: float) -> Tuple[int, int]:
        return (int(width * x_pct / 100), int(height * y_pct / 100))

    shaft_y = int(height * LAYOUT["shaft_y_pct"] / 100)

    # ===== Eje horizontal completo =====
    eje_x1 = int(width * LAYOUT["driver_x_pct"][0] / 100)
    eje_x2 = int(width * LAYOUT["driven_x_pct"][1] / 100)
    draw.line([(eje_x1, shaft_y), (eje_x2, shaft_y)], fill="#374151", width=5)

    # ===== Driver box =====
    dx1, dy1 = pct_xy(LAYOUT["driver_x_pct"][0], LAYOUT["driver_y_pct"][0])
    dx2, dy2 = pct_xy(LAYOUT["driver_x_pct"][1], LAYOUT["driver_y_pct"][1])
    # Body
    draw.rectangle(
        [dx1, dy1, dx2, dy2],
        outline="#1e3a8a", width=3, fill="#dbeafe",
    )
    # Tapa DE (lado del acople, más oscura)
    de_w = max(14, (dx2 - dx1) // 12)
    draw.rectangle(
        [dx2 - de_w, dy1, dx2, dy2],
        outline="#1e3a8a", width=3, fill="#bfdbfe",
    )
    # Tapa NDE (extremo opuesto)
    draw.rectangle(
        [dx1, dy1, dx1 + de_w, dy2],
        outline="#1e3a8a", width=3, fill="#bfdbfe",
    )
    # Label
    label_text = (driver_label or "Driver").strip()[:34]
    draw.text(
        ((dx1 + dx2) // 2 - len(label_text) * 7, dy1 - 28),
        label_text, fill="#1e3a8a", font=font_med,
    )
    # Etiqueta DE / NDE arriba del cuerpo
    draw.text((dx1 + de_w + 4, dy1 + 4), "NDE", fill="#1e3a8a", font=font_sm)
    draw.text((dx2 - de_w - 30, dy1 + 4), "DE", fill="#1e3a8a", font=font_sm)

    # ===== Driven box =====
    nx1, ny1 = pct_xy(LAYOUT["driven_x_pct"][0], LAYOUT["driven_y_pct"][0])
    nx2, ny2 = pct_xy(LAYOUT["driven_x_pct"][1], LAYOUT["driven_y_pct"][1])
    draw.rectangle(
        [nx1, ny1, nx2, ny2],
        outline="#14532d", width=3, fill="#dcfce7",
    )
    de_w_n = max(14, (nx2 - nx1) // 14)
    # Tapa DE driven (lado acople = izquierda)
    draw.rectangle(
        [nx1, ny1, nx1 + de_w_n, ny2],
        outline="#14532d", width=3, fill="#bbf7d0",
    )
    # Tapa NDE driven (extremo derecho)
    draw.rectangle(
        [nx2 - de_w_n, ny1, nx2, ny2],
        outline="#14532d", width=3, fill="#bbf7d0",
    )
    label_text2 = (driven_label or "Driven").strip()[:42]
    draw.text(
        ((nx1 + nx2) // 2 - len(label_text2) * 7, ny1 - 28),
        label_text2, fill="#14532d", font=font_med,
    )
    draw.text((nx1 + de_w_n + 4, ny1 + 4), "DE", fill="#14532d", font=font_sm)
    draw.text((nx2 - de_w_n - 32, ny1 + 4), "NDE", fill="#14532d", font=font_sm)

    # ===== Acople (ámbar, entre driver y driven) =====
    cx1, _ = pct_xy(LAYOUT["coupling_x_pct"][0], 50)
    cx2, _ = pct_xy(LAYOUT["coupling_x_pct"][1], 50)
    coup_h = 38
    # Cuerpo
    draw.rectangle(
        [cx1, shaft_y - coup_h, cx2, shaft_y + coup_h],
        outline="#92400e", width=3, fill="#fde68a",
    )
    # Detalle: 3 líneas verticales (representan flanges)
    third = (cx2 - cx1) // 3
    for i in (1, 2):
        x = cx1 + third * i
        draw.line(
            [(x, shaft_y - coup_h + 4), (x, shaft_y + coup_h - 4)],
            fill="#92400e", width=2,
        )
    draw.text(
        ((cx1 + cx2) // 2 - 26, shaft_y + coup_h + 6),
        "Acople", fill="#92400e", font=font_sm,
    )

    # ===== Cojinetes del driver =====
    n_d = max(1, n_driver_planes)
    drv_inner_x1 = dx1 + de_w
    drv_inner_x2 = dx2 - de_w
    plane_idx = 0
    for i in range(n_d):
        # Distribuir uniformemente en el cuerpo (excluyendo tapas)
        bx = drv_inner_x1 + (drv_inner_x2 - drv_inner_x1) * (i + 1) // (n_d + 1)
        plane_idx += 1
        # Círculo del cojinete (overlap parcial sobre el eje)
        r = 22
        draw.ellipse(
            [bx - r, shaft_y - r, bx + r, shaft_y + r],
            outline="#1e3a8a", width=3, fill="white",
        )
        draw.text((bx - 6, shaft_y - 9), str(plane_idx),
                  fill="#1e3a8a", font=font_med)
        side = "DE" if i == n_d - 1 else "NDE" if i == 0 else f"P{plane_idx}"
        draw.text((bx - 12, shaft_y + r + 4), side,
                  fill="#1e3a8a", font=font_sm)

    # ===== Cojinetes del driven =====
    n_dn = max(1, n_driven_planes)
    drv2_inner_x1 = nx1 + de_w_n
    drv2_inner_x2 = nx2 - de_w_n
    for i in range(n_dn):
        bx = drv2_inner_x1 + (drv2_inner_x2 - drv2_inner_x1) * (i + 1) // (n_dn + 1)
        plane_idx += 1
        r = 22
        draw.ellipse(
            [bx - r, shaft_y - r, bx + r, shaft_y + r],
            outline="#14532d", width=3, fill="white",
        )
        draw.text((bx - 6, shaft_y - 9), str(plane_idx),
                  fill="#14532d", font=font_med)
        side = "DE" if i == 0 else "NDE" if i == n_dn - 1 else f"P{plane_idx}"
        draw.text((bx - 12, shaft_y + r + 4), side,
                  fill="#14532d", font=font_sm)

    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


def sensor_default_position(
    sensor: Dict[str, Any],
    n_driver_planes: int = 2,
    n_driven_planes: int = 2,
) -> Tuple[float, float]:
    """
    Devuelve (x_pct, y_pct) iniciales para un sensor según su plano y
    tipo. Para acelerómetros radiales se ubica arriba del cojinete;
    para proximidad X/Y a 45° R/L sobre el cojinete; etc.
    """
    plane = int(sensor.get("plane", 1) or 1)
    direction = (sensor.get("direction", "") or "").upper()
    sensor_type = (sensor.get("sensor_type", "") or "").lower()

    n_d = max(1, n_driver_planes)
    n_dn = max(1, n_driven_planes)

    # Determinar lado driver vs driven por número de plano
    if plane <= n_d:
        # Driver
        x_start, x_end = LAYOUT["driver_x_pct"]
        local_idx = plane
        local_total = n_d
    else:
        x_start, x_end = LAYOUT["driven_x_pct"]
        local_idx = plane - n_d
        local_total = n_dn

    # Distribuir uniformemente en el cuerpo de la máquina
    inner_pad = 2.5  # margen para no solapar las tapas
    inner_start = x_start + inner_pad
    inner_end = x_end - inner_pad
    x_pct = inner_start + (inner_end - inner_start) * local_idx / (local_total + 1)

    shaft_y = LAYOUT["shaft_y_pct"]
    if sensor_type == "keyphasor":
        # Keyphasor en el coupling
        cx1, cx2 = LAYOUT["coupling_x_pct"]
        return ((cx1 + cx2) / 2, shaft_y + 8)
    if direction == "X":
        # X arriba del eje (45° R)
        return (x_pct + 1.0, shaft_y - 7)
    if direction == "Y":
        # Y debajo del eje (45° L)
        return (x_pct - 1.0, shaft_y + 7)
    if direction in ("RADIAL", "RAD"):
        return (x_pct, shaft_y - 11)  # arriba del cojinete (carcasa)
    if direction in ("AXIAL", "AX"):
        return (x_pct + 4, shaft_y)
    return (x_pct, shaft_y - 5)


__all__ = [
    "generate_train_png",
    "sensor_default_position",
    "LAYOUT",
]
