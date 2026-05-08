"""
core.recip_schematic
====================

Genera schematics PNG para compresores reciprocantes (Ciclo 21.4/21.5).

Sin esto, el render genérico de `core/sensor_diagram.py` numeraba los
sensores en línea horizontal sin distinguir entre cojinetes del motor,
frame del compresor, crossheads y rod drops — lo que producía dibujos
incorrectos como "DE motor / NDE motor / Frame top / Cilindro 1" todos
en fila.

Aquí dibujamos el activo en sus partes físicas reales:
  - Motor a la izquierda (rectángulo) con N cojinetes
  - Pieza de distancia (opcional) entre motor y crankshaft
  - Cigüeñal (línea horizontal central)
  - Compresor a la derecha con N cilindros sobre cuerpo
  - Acople

API:
    generate_recip_png(n_cylinders, n_motor_planes, ...) -> bytes
    sensor_default_position(sensor, n_cylinders, ...) -> (x_pct, y_pct)
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


# Layout constants (unidades = % del ancho/alto del canvas)
LAYOUT = {
    "motor_x_pct":   (4, 32),     # rect motor: x_left a x_right en %
    "motor_y_pct":   (35, 75),    # rect motor: y_top a y_bottom en %
    "distance_x_pct": (33, 42),
    "distance_y_pct": (40, 70),
    "compressor_x_pct": (43, 96),
    "compressor_y_pct": (40, 80),
    "cylinder_y_pct":   (15, 38),  # cuadrados arriba del cuerpo
    "shaft_y_pct":      57,         # línea cigüeñal
}


def generate_recip_png(
    n_cylinders: int,
    n_motor_planes: int = 2,
    has_distance_piece: bool = True,
    motor_label: str = "Motor",
    compressor_label: str = "Compresor",
    width: int = 1400,
    height: int = 500,
    bg_color: str = "#ffffff",
) -> bytes:
    """
    Genera un PNG con el dibujo del tren reciprocante.
    Devuelve bytes PNG. Si PIL no está disponible, devuelve b"".
    """
    if not _HAS_PIL:
        return b""

    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)

    try:
        font_big = ImageFont.truetype("Arial.ttf", 28)
        font_med = ImageFont.truetype("Arial.ttf", 18)
        font_sm = ImageFont.truetype("Arial.ttf", 14)
    except Exception:
        font_big = ImageFont.load_default()
        font_med = font_big
        font_sm = font_big

    def pct_to_xy(x_pct: float, y_pct: float) -> Tuple[int, int]:
        return (int(width * x_pct / 100), int(height * y_pct / 100))

    def pct_box(x1, x2, y1, y2):
        return (*pct_to_xy(x1, y1), *pct_to_xy(x2, y2))

    # ===== MOTOR =====
    mx1, mx2 = LAYOUT["motor_x_pct"]
    my1, my2 = LAYOUT["motor_y_pct"]
    draw.rectangle(pct_box(mx1, mx2, my1, my2),
                   fill="#dbeafe", outline="#1e40af", width=3)
    # Label motor
    label_xy = pct_to_xy((mx1 + mx2) / 2 - 4, my1 - 8)
    draw.text(label_xy, motor_label, fill="#1e40af", font=font_big)

    # Cojinetes del motor (círculos numerados)
    motor_centerline_y = (my1 + my2) / 2
    bearing_spacing = (mx2 - mx1 - 6) / max(n_motor_planes, 1)
    for i in range(n_motor_planes):
        bx_pct = mx1 + 3 + bearing_spacing * (i + 0.5)
        bx, by = pct_to_xy(bx_pct, motor_centerline_y)
        r = 18
        draw.ellipse((bx - r, by - r, bx + r, by + r),
                     fill="#ffffff", outline="#1e40af", width=2)
        draw.text((bx - 6, by - 12), str(i + 1), fill="#1e40af", font=font_med)
        # Label below
        side = "DE" if i == 0 else ("NDE" if i == 1 else f"P{i+1}")
        draw.text((bx - 14, by + r + 6), side, fill="#475569", font=font_sm)

    # ===== PIEZA DE DISTANCIA =====
    if has_distance_piece:
        dx1, dx2 = LAYOUT["distance_x_pct"]
        dy1, dy2 = LAYOUT["distance_y_pct"]
        draw.rectangle(pct_box(dx1, dx2, dy1, dy2),
                       fill="#fef3c7", outline="#a16207", width=2)
        draw.text(pct_to_xy(dx1 + 0.5, dy2 + 1),
                  "Pieza\ndistancia", fill="#a16207", font=font_sm)

    # ===== COMPRESOR =====
    cx1, cx2 = LAYOUT["compressor_x_pct"]
    cy1, cy2 = LAYOUT["compressor_y_pct"]
    draw.rectangle(pct_box(cx1, cx2, cy1, cy2),
                   fill="#dcfce7", outline="#15803d", width=3)
    label_cxy = pct_to_xy((cx1 + cx2) / 2 - 5, cy1 - 8)
    draw.text(label_cxy, compressor_label, fill="#15803d", font=font_big)

    # ===== CILINDROS =====
    cyl_y1_pct, cyl_y2_pct = LAYOUT["cylinder_y_pct"]
    cyl_h = (cyl_y2_pct - cyl_y1_pct)
    cyl_w_pct = (cx2 - cx1) / (n_cylinders + 1) * 0.8
    for c in range(n_cylinders):
        cx_center_pct = cx1 + (c + 1) * (cx2 - cx1) / (n_cylinders + 1)
        cx_left = cx_center_pct - cyl_w_pct / 2
        cx_right = cx_center_pct + cyl_w_pct / 2
        draw.rectangle(pct_box(cx_left, cx_right, cyl_y1_pct, cyl_y2_pct),
                       fill="#ffffff", outline="#15803d", width=2)
        # Label C1, C2, ...
        lbl_xy = pct_to_xy(cx_center_pct - 1, cyl_y1_pct + cyl_h / 2 - 1)
        draw.text(lbl_xy, f"C{c+1}", fill="#15803d", font=font_med)
        # Línea descendente al cigüeñal (cuello de cilindro)
        shaft_y = LAYOUT["shaft_y_pct"]
        line_top_xy = pct_to_xy(cx_center_pct, cyl_y2_pct)
        line_bot_xy = pct_to_xy(cx_center_pct, shaft_y)
        draw.line([line_top_xy, line_bot_xy], fill="#15803d", width=3)

    # ===== CIGÜEÑAL (línea horizontal central) =====
    shaft_y_pct = LAYOUT["shaft_y_pct"]
    sx1, _ = pct_to_xy(mx2, shaft_y_pct)
    sx2, _ = pct_to_xy(cx1, shaft_y_pct)
    sy = pct_to_xy(0, shaft_y_pct)[1]
    draw.line([(sx1, sy), (sx2, sy)], fill="#1e293b", width=4)
    # Acople (líneas verticales en el medio de la pieza distancia)
    if has_distance_piece:
        dx1, dx2 = LAYOUT["distance_x_pct"]
        coup_x = (dx1 + dx2) / 2
        cy_top, cy_bot = pct_to_xy(coup_x, shaft_y_pct - 4)[1], pct_to_xy(coup_x, shaft_y_pct + 4)[1]
        cx_pos, _ = pct_to_xy(coup_x, 0)
        for off in (-3, 0, 3):
            draw.line([(cx_pos + off, cy_top), (cx_pos + off, cy_bot)],
                      fill="#475569", width=2)

    # Footer label
    draw.text((width - 250, height - 24),
              f"{n_cylinders} cilindros · {n_motor_planes} cojinetes motor",
              fill="#64748b", font=font_sm)

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def sensor_default_position(
    sensor: Dict[str, Any],
    n_cylinders: int = 4,
    n_motor_planes: int = 2,
) -> Tuple[float, float]:
    """
    Devuelve (x_pct, y_pct) sugerido para un sensor según su rol.
    Diseñado para coincidir con el layout que produce generate_recip_png().
    """
    plane_label = (sensor.get("plane_label") or "").lower()
    sensor_type = sensor.get("sensor_type", "")

    mx1, mx2 = LAYOUT["motor_x_pct"]
    my_center = sum(LAYOUT["motor_y_pct"]) / 2
    cx1, cx2 = LAYOUT["compressor_x_pct"]
    cy1, cy2 = LAYOUT["compressor_y_pct"]
    cyl_y1, cyl_y2 = LAYOUT["cylinder_y_pct"]

    # Motor
    if "motor" in plane_label:
        if "de" in plane_label.split() or plane_label.startswith("de"):
            return ((mx1 + mx2) * 0.40, my_center)
        if "nde" in plane_label.split() or plane_label.startswith("nde"):
            return ((mx1 + mx2) * 0.60, my_center)
        # planos extra del motor
        return ((mx1 + mx2) / 2, my_center)

    # Frame
    if "frame top" in plane_label:
        return ((cx1 + cx2) / 2, cy1 - 5)
    if "frame side" in plane_label:
        return (cx1 + 2, (cy1 + cy2) / 2)

    # Crosshead / cilindro
    if "cilindro" in plane_label or "crosshead" in plane_label:
        # Extraer número del cilindro
        import re
        m = re.search(r"cilindro\s*(\d+)", plane_label)
        if not m:
            m = re.search(r"(\d+)", plane_label)
        cyl_num = int(m.group(1)) if m else 1
        cyl_num = max(1, min(cyl_num, n_cylinders))
        cx_center = cx1 + cyl_num * (cx2 - cx1) / (n_cylinders + 1)
        if "rod drop" in plane_label:
            # debajo del cigüeñal
            return (cx_center, LAYOUT["shaft_y_pct"] + 8)
        else:
            # crosshead: justo bajo el cuadrado del cilindro
            return (cx_center, cyl_y2 + 4)

    # Keyphasor en el acople
    if sensor_type == "keyphasor":
        dx1, dx2 = LAYOUT["distance_x_pct"]
        return ((dx1 + dx2) / 2, LAYOUT["shaft_y_pct"] - 8)

    # Default: centro
    return (50.0, 50.0)


__all__ = ["generate_recip_png", "sensor_default_position", "LAYOUT"]
