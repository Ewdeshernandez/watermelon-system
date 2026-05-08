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
# Ciclo 21.4 v2: eliminada pieza distancia, acople = 3 líneas verticales,
# cilindros todos arriba alineados (estilo Ariel KBK / Burckhardt real).
LAYOUT = {
    "motor_x_pct":   (5, 35),
    "motor_y_pct":   (40, 78),
    "coupling_x_pct": (36, 41),
    "compressor_x_pct": (42, 96),
    "compressor_y_pct": (50, 78),
    "cylinder_y_pct":   (20, 48),  # cilindros arriba — más altos
    "shaft_y_pct":      59,
}


def generate_recip_png(
    n_cylinders: int,
    n_motor_planes: int = 2,
    has_distance_piece: bool = False,  # deprecated — siempre acople directo
    motor_label: str = "Motor",
    compressor_label: str = "Compresor",
    width: int = 1400,
    height: int = 520,
    bg_color: str = "#fafbfc",
) -> bytes:
    """
    Genera un PNG con el dibujo del tren reciprocante.
    Layout estilo Ariel KBK / Burckhardt real:
      - Motor a la izquierda (rectángulo con tapas)
      - Acople directo (3 líneas verticales)
      - Compresor con cilindros TODOS ARRIBA alineados
      - Cigüeñal interno al cuerpo del compresor
    """
    if not _HAS_PIL:
        return b""

    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)

    try:
        font_big = ImageFont.truetype("Arial Bold.ttf", 26)
    except Exception:
        try:
            font_big = ImageFont.truetype("Arial.ttf", 26)
        except Exception:
            font_big = ImageFont.load_default()
    try:
        font_med = ImageFont.truetype("Arial.ttf", 17)
        font_sm = ImageFont.truetype("Arial.ttf", 13)
    except Exception:
        font_med = font_big
        font_sm = font_big

    def pct_to_xy(x_pct: float, y_pct: float) -> Tuple[int, int]:
        return (int(width * x_pct / 100), int(height * y_pct / 100))

    def pct_box(x1, x2, y1, y2):
        return (*pct_to_xy(x1, y1), *pct_to_xy(x2, y2))

    # ===== MOTOR (cuerpo principal + tapas DE/NDE) =====
    mx1, mx2 = LAYOUT["motor_x_pct"]
    my1, my2 = LAYOUT["motor_y_pct"]
    # Tapa DE (izquierda, más oscura)
    cap_w = (mx2 - mx1) * 0.06
    draw.rectangle(pct_box(mx1, mx1 + cap_w, my1 - 2, my2 + 2),
                   fill="#93c5fd", outline="#1e3a8a", width=2)
    # Tapa NDE (derecha)
    draw.rectangle(pct_box(mx2 - cap_w, mx2, my1 - 2, my2 + 2),
                   fill="#93c5fd", outline="#1e3a8a", width=2)
    # Cuerpo principal
    draw.rectangle(pct_box(mx1 + cap_w, mx2 - cap_w, my1, my2),
                   fill="#dbeafe", outline="#1e40af", width=3)

    # Aletas / nervaduras decorativas en el cuerpo del motor
    body_x1 = mx1 + cap_w + 1
    body_x2 = mx2 - cap_w - 1
    fin_count = 5
    for i in range(1, fin_count + 1):
        fx = body_x1 + (body_x2 - body_x1) * i / (fin_count + 1)
        draw.line([pct_to_xy(fx, my1 + 2), pct_to_xy(fx, my2 - 2)],
                  fill="#60a5fa", width=2)

    # Label motor
    draw.text(pct_to_xy(mx1 + 2, my1 - 8), motor_label, fill="#1e3a8a", font=font_big)

    # Cojinetes del motor (círculos numerados — solo los configurados)
    motor_centerline_y = (my1 + my2) / 2
    bearing_xs = []
    if n_motor_planes == 1:
        bearing_xs = [(mx1 + mx2) / 2]
    elif n_motor_planes == 2:
        bearing_xs = [mx1 + (mx2 - mx1) * 0.30, mx1 + (mx2 - mx1) * 0.70]
    else:
        for i in range(n_motor_planes):
            bearing_xs.append(mx1 + (mx2 - mx1) * (0.18 + 0.64 * i / max(n_motor_planes - 1, 1)))
    for i, bx_pct in enumerate(bearing_xs):
        bx, by = pct_to_xy(bx_pct, motor_centerline_y)
        r = 17
        draw.ellipse((bx - r, by - r, bx + r, by + r),
                     fill="#ffffff", outline="#1e40af", width=2)
        draw.text((bx - 5, by - 11), str(i + 1), fill="#1e40af", font=font_med)
        side = "DE" if i == 0 else ("NDE" if i == 1 else f"P{i+1}")
        draw.text((bx - 11, by + r + 5), side, fill="#475569", font=font_sm)

    # ===== ACOPLE (3 líneas verticales) =====
    coup_x1, coup_x2 = LAYOUT["coupling_x_pct"]
    coup_center = (coup_x1 + coup_x2) / 2
    shaft_y = LAYOUT["shaft_y_pct"]
    coup_top_y = shaft_y - 5
    coup_bot_y = shaft_y + 5
    for off_pct in (-1.0, 0.0, 1.0):
        cx, _ = pct_to_xy(coup_center + off_pct, 0)
        cy_top = pct_to_xy(0, coup_top_y)[1]
        cy_bot = pct_to_xy(0, coup_bot_y)[1]
        draw.line([(cx, cy_top), (cx, cy_bot)], fill="#475569", width=3)
    # Texto "Acople" debajo
    draw.text(pct_to_xy(coup_center - 2.5, coup_bot_y + 2),
              "Acople", fill="#64748b", font=font_sm)

    # ===== COMPRESOR (cuerpo + frame del cigüeñal) =====
    cx1, cx2 = LAYOUT["compressor_x_pct"]
    cy1, cy2 = LAYOUT["compressor_y_pct"]
    # Frame del compresor
    draw.rectangle(pct_box(cx1, cx2, cy1, cy2),
                   fill="#dcfce7", outline="#15803d", width=3)

    # Label compresor
    draw.text(pct_to_xy(cx1 + 2, cy1 - 8), compressor_label,
              fill="#14532d", font=font_big)

    # ===== CILINDROS — TODOS ARRIBA, ALINEADOS =====
    cyl_y1_pct, cyl_y2_pct = LAYOUT["cylinder_y_pct"]
    # Width disponible para cilindros: cx1 + 2% a cx2 - 2%
    cyl_zone_x1 = cx1 + 2
    cyl_zone_x2 = cx2 - 2
    cyl_total_w = cyl_zone_x2 - cyl_zone_x1
    cyl_w_pct = cyl_total_w / n_cylinders * 0.85
    for c in range(n_cylinders):
        cx_center_pct = cyl_zone_x1 + (c + 0.5) * cyl_total_w / n_cylinders
        cx_left = cx_center_pct - cyl_w_pct / 2
        cx_right = cx_center_pct + cyl_w_pct / 2
        # Cilindro: rectángulo con redondeo simulado por dos rectángulos
        draw.rectangle(pct_box(cx_left, cx_right, cyl_y1_pct, cyl_y2_pct),
                       fill="#ffffff", outline="#15803d", width=2)
        # Tope del cilindro (válvulas) — barra superior
        draw.rectangle(pct_box(cx_left - 0.3, cx_right + 0.3,
                               cyl_y1_pct - 1.5, cyl_y1_pct + 1),
                       fill="#bbf7d0", outline="#15803d", width=1)
        # Label centrado
        cyl_h = cyl_y2_pct - cyl_y1_pct
        lbl_xy = pct_to_xy(cx_center_pct - 1.2, cyl_y1_pct + cyl_h / 2 - 1.3)
        draw.text(lbl_xy, f"C{c+1}", fill="#14532d", font=font_med)
        # Conexión vertical al cuerpo del compresor (cuello pistón)
        line_top_xy = pct_to_xy(cx_center_pct, cyl_y2_pct)
        line_bot_xy = pct_to_xy(cx_center_pct, cy1 + 1)
        draw.line([line_top_xy, line_bot_xy], fill="#15803d", width=2)

    # ===== CIGÜEÑAL (línea horizontal interna al frame del compresor) =====
    shaft_y_pct = LAYOUT["shaft_y_pct"]
    sx_motor, _ = pct_to_xy(mx2, shaft_y_pct)
    sx_comp, _ = pct_to_xy(cx1, shaft_y_pct)
    sy = pct_to_xy(0, shaft_y_pct)[1]
    # Eje motor → acople
    draw.line([(sx_motor, sy), pct_to_xy(coup_x1, shaft_y_pct)],
              fill="#1e293b", width=4)
    # Eje acople → compresor
    draw.line([pct_to_xy(coup_x2, shaft_y_pct), (sx_comp, sy)],
              fill="#1e293b", width=4)
    # Cigüeñal dentro del compresor (línea más gruesa horizontal)
    sx_in1 = pct_to_xy(cx1 + 1, shaft_y_pct)
    sx_in2 = pct_to_xy(cx2 - 1, shaft_y_pct)
    draw.line([sx_in1, sx_in2], fill="#0f172a", width=5)
    draw.text(pct_to_xy(cx1 + 2, shaft_y_pct + 2),
              "cigüeñal", fill="#475569", font=font_sm)

    # Footer label
    draw.text((width - 280, height - 22),
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
        dx1, dx2 = LAYOUT["coupling_x_pct"]
        return ((dx1 + dx2) / 2, LAYOUT["shaft_y_pct"] - 8)

    # Default: centro
    return (50.0, 50.0)


__all__ = ["generate_recip_png", "sensor_default_position", "LAYOUT"]
