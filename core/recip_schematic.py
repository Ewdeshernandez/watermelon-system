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
# Ciclo 21.4 v3: cilindros HORIZONTALMENTE OPUESTOS (boxer style Ariel
# KBK/KBT, Burckhardt) + acople con flanges y bulones.
LAYOUT = {
    "motor_x_pct":   (5, 33),
    "motor_y_pct":   (38, 70),
    "coupling_x_pct": (34, 41),     # zona del acople (más ancha para flanges)
    "compressor_x_pct": (42, 96),
    "compressor_y_pct": (45, 63),    # frame del compresor más angosto (centro)
    "cylinder_top_y_pct":    (10, 38),  # cilindros opuestos arriba
    "cylinder_bottom_y_pct": (70, 98),  # cilindros opuestos abajo (espejo)
    "shaft_y_pct":      54,
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

    # ===== ACOPLE (2 flanges con bulones — estilo gear coupling real) =====
    coup_x1, coup_x2 = LAYOUT["coupling_x_pct"]
    shaft_y = LAYOUT["shaft_y_pct"]
    flange_w_pct = 1.6
    flange_h_pct = 12  # altura de los flanges
    flange_y1 = shaft_y - flange_h_pct / 2
    flange_y2 = shaft_y + flange_h_pct / 2
    coup_center = (coup_x1 + coup_x2) / 2

    # Flange izquierdo (lado motor)
    fl_x1 = coup_center - 2.5
    fl_x2 = fl_x1 + flange_w_pct
    draw.rectangle(pct_box(fl_x1, fl_x2, flange_y1, flange_y2),
                   fill="#cbd5e1", outline="#334155", width=2)
    # Flange derecho (lado compresor)
    fr_x1 = coup_center + 0.9
    fr_x2 = fr_x1 + flange_w_pct
    draw.rectangle(pct_box(fr_x1, fr_x2, flange_y1, flange_y2),
                   fill="#cbd5e1", outline="#334155", width=2)
    # Bulones — 4 puntos en cada flange
    for fl_xc in [(fl_x1 + fl_x2) / 2, (fr_x1 + fr_x2) / 2]:
        for off_y_pct in (-3.5, -1.2, 1.2, 3.5):
            bx, by = pct_to_xy(fl_xc, shaft_y + off_y_pct)
            draw.ellipse((bx - 2, by - 2, bx + 2, by + 2), fill="#1e293b")
    # Conector central entre flanges (gear teeth simulation)
    mid_x1 = fl_x2
    mid_x2 = fr_x1
    draw.rectangle(pct_box(mid_x1, mid_x2, shaft_y - 2, shaft_y + 2),
                   fill="#94a3b8", outline="#334155", width=1)
    # Texto "Acople" debajo
    draw.text(pct_to_xy(coup_center - 2.0, flange_y2 + 1),
              "Acople", fill="#475569", font=font_sm)

    # ===== COMPRESOR (cuerpo + frame del cigüeñal) =====
    cx1, cx2 = LAYOUT["compressor_x_pct"]
    cy1, cy2 = LAYOUT["compressor_y_pct"]
    # Frame del compresor
    draw.rectangle(pct_box(cx1, cx2, cy1, cy2),
                   fill="#dcfce7", outline="#15803d", width=3)

    # Label compresor
    draw.text(pct_to_xy(cx1 + 2, cy1 - 8), compressor_label,
              fill="#14532d", font=font_big)

    # ===== CILINDROS — HORIZONTALMENTE OPUESTOS (boxer/Ariel KBK style) =====
    # Convención: cilindros impares (1, 3, 5, 7) ARRIBA; pares (2, 4, 6, 8) ABAJO,
    # alineados en pares por posición x.
    cyl_top_y1, cyl_top_y2 = LAYOUT["cylinder_top_y_pct"]
    cyl_bot_y1, cyl_bot_y2 = LAYOUT["cylinder_bottom_y_pct"]
    n_pairs = (n_cylinders + 1) // 2  # n=4 → 2 pares; n=3 → 2 pares (último arriba solo)

    cyl_zone_x1 = cx1 + 3
    cyl_zone_x2 = cx2 - 3
    cyl_zone_w = cyl_zone_x2 - cyl_zone_x1
    pair_w = cyl_zone_w / n_pairs
    cyl_w_pct = pair_w * 0.55  # ancho de cada cilindro

    def _draw_cylinder(cx_center_pct: float, cy_y1: float, cy_y2: float,
                       label: str, is_top: bool):
        cx_left = cx_center_pct - cyl_w_pct / 2
        cx_right = cx_center_pct + cyl_w_pct / 2
        draw.rectangle(pct_box(cx_left, cx_right, cy_y1, cy_y2),
                       fill="#ffffff", outline="#15803d", width=2)
        # Tope (cabeza del cilindro con válvulas) — del lado externo
        if is_top:
            draw.rectangle(pct_box(cx_left - 0.3, cx_right + 0.3,
                                   cy_y1 - 1.5, cy_y1 + 1),
                           fill="#bbf7d0", outline="#15803d", width=1)
        else:
            draw.rectangle(pct_box(cx_left - 0.3, cx_right + 0.3,
                                   cy_y2 - 1, cy_y2 + 1.5),
                           fill="#bbf7d0", outline="#15803d", width=1)
        # Label centrado
        cyl_h = cy_y2 - cy_y1
        lbl_xy = pct_to_xy(cx_center_pct - 1.2, cy_y1 + cyl_h / 2 - 1.3)
        draw.text(lbl_xy, label, fill="#14532d", font=font_med)
        # Conexión al frame (cuello del pistón)
        if is_top:
            line_top_xy = pct_to_xy(cx_center_pct, cy_y2)
            line_bot_xy = pct_to_xy(cx_center_pct, cy1 + 0.5)
        else:
            line_top_xy = pct_to_xy(cx_center_pct, cy2 - 0.5)
            line_bot_xy = pct_to_xy(cx_center_pct, cy_y1)
        draw.line([line_top_xy, line_bot_xy], fill="#15803d", width=2)

    for pair_idx in range(n_pairs):
        cx_center = cyl_zone_x1 + (pair_idx + 0.5) * pair_w
        # Cilindro impar (arriba) — siempre presente si pair_idx < ceil(n/2)
        cyl_top_num = pair_idx * 2 + 1
        if cyl_top_num <= n_cylinders:
            _draw_cylinder(cx_center, cyl_top_y1, cyl_top_y2,
                           f"C{cyl_top_num}", is_top=True)
        # Cilindro par (abajo) — espejo
        cyl_bot_num = pair_idx * 2 + 2
        if cyl_bot_num <= n_cylinders:
            _draw_cylinder(cx_center, cyl_bot_y1, cyl_bot_y2,
                           f"C{cyl_bot_num}", is_top=False)

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

    # Footer label removido (Ciclo 21.4 v4) — se superponía con los
    # cilindros inferiores en algunos layouts. La info de cilindros y
    # cojinetes ya queda visible en los labels de los componentes.

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
    # Layout boxer: cilindros impares arriba, pares abajo
    cyl_y1_top, cyl_y2_top = LAYOUT["cylinder_top_y_pct"]
    cyl_y1_bot, cyl_y2_bot = LAYOUT["cylinder_bottom_y_pct"]

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

    # Crosshead / cilindro — config boxer (opuestos)
    if "cilindro" in plane_label or "crosshead" in plane_label:
        import re
        m = re.search(r"cilindro\s*(\d+)", plane_label)
        if not m:
            m = re.search(r"(\d+)", plane_label)
        cyl_num = int(m.group(1)) if m else 1
        cyl_num = max(1, min(cyl_num, n_cylinders))
        # Boxer: par arriba (impar) + abajo (par)
        n_pairs = (n_cylinders + 1) // 2
        pair_idx = (cyl_num - 1) // 2
        cyl_zone_x1 = cx1 + 3
        cyl_zone_w = (cx2 - 3) - cyl_zone_x1
        pair_w = cyl_zone_w / max(n_pairs, 1)
        cx_center = cyl_zone_x1 + (pair_idx + 0.5) * pair_w
        is_top = (cyl_num % 2 == 1)  # impar arriba
        if "rod drop" in plane_label:
            # rod drop al lado del cigüeñal
            if is_top:
                return (cx_center, LAYOUT["shaft_y_pct"] - 3)
            else:
                return (cx_center, LAYOUT["shaft_y_pct"] + 3)
        else:
            # crosshead: justo entre el cilindro y el frame
            if is_top:
                cyl_y_top1, cyl_y_top2 = LAYOUT["cylinder_top_y_pct"]
                return (cx_center, cyl_y_top2 + 3)
            else:
                cyl_y_bot1, cyl_y_bot2 = LAYOUT["cylinder_bottom_y_pct"]
                return (cx_center, cyl_y_bot1 - 3)

    # Keyphasor en el acople
    if sensor_type == "keyphasor":
        dx1, dx2 = LAYOUT["coupling_x_pct"]
        return ((dx1 + dx2) / 2, LAYOUT["shaft_y_pct"] - 8)

    # Default: centro
    return (50.0, 50.0)


__all__ = ["generate_recip_png", "sensor_default_position", "LAYOUT"]
