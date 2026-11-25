"""Compone el video de ventas (ES / EN) desde las capturas reales de SGT300B.
Uso: python compose.py <shots_dir> <out.mp4> <es|en>"""
import os
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFilter, ImageFont

SHOTS, OUT, LANG = sys.argv[1], sys.argv[2], sys.argv[3]
ROOT = Path(__file__).resolve().parents[2]
FONTS = str(ROOT / "assets" / "fonts") + "/"
LOGO = str(ROOT / "assets" / "watermelon_logo.png")
# Máquina mostrada en la portada (ej. "SGT300B · <cliente>"); se pasa por entorno
# para no dejar nombres de cliente en el repo.
ASSET = os.environ.get("WM_TOUR_ASSET", "")
W, H, FPS = 1600, 1000, 30
SRC_W, SRC_H = 2880, 1800
NAVY, INK, PAPER, GREEN, BLUE = (11, 31, 58), (18, 48, 94), (244, 247, 251), (34, 197, 94), (39, 75, 125)
XF = 0.55  # crossfade (s)


def font(name, size):
    return ImageFont.truetype(FONTS + name, size)


F_KICK = font("IBMPlexMono-SemiBold.ttf", 19)
F_TITLE = font("IBMPlexSans-SemiBold.ttf", 40)
F_SUB = font("IBMPlexSans-Regular.ttf", 24)
F_BIG = font("IBMPlexSans-Bold.ttf", 62)
F_MID = font("IBMPlexSans-Regular.ttf", 30)
F_SMALL = font("IBMPlexMono-Medium.ttf", 22)

T = {
    "es": {
        "intro": ("Monitoreo de turbomaquinaria en vivo",
                  "Recorrido real" + (f" · {ASSET}" if ASSET else " · datos reales")),
        "s": [
            ("EN VIVO", "Su turbogenerador, en vivo",
             "Velocidad, vibración y salud del activo, actualizados cada 10 segundos."),
            ("TREN DE MÁQUINA", "Cada sensor sobre el tren real",
             "Turbina, caja y generador con lectura y severidad en cada punto."),
            ("API 670", "Tabla de vibración por canal",
             "Zona ISO, % de alarma, tendencia y vector 1X de cada sensor."),
            ("HISTÓRICO", "Tendencias de días, meses y años",
             "Arranques, disparos y cambios de carga quedan a la vista."),
            ("EVENTOS", "Bitácora de cada cruce de umbral",
             "Hora, valor y duración en alarma: nada se pierde."),
            ("ROTODINÁMICA", "Forma de onda real",
             "Procesada por nuestro motor de rotodinámica, con keyphasor por vuelta."),
            ("ROTODINÁMICA", "Espectro FFT con órdenes 1X–4X",
             "Base para diagnosticar desbalanceo, desalineación, holguras y más."),
            ("ROTODINÁMICA", "Órbitas de los cinco cojinetes",
             "Misma escala, sondas en su ángulo real y sentido de giro API 670."),
            ("ROTODINÁMICA", "Forma y tamaño de cada órbita",
             "Eje mayor, eje menor, elipticidad y Smax por cojinete."),
            ("REPORTES", "Reporte semanal firmado por especialista",
             "Hallazgos, recomendaciones, órbitas, Bode y polar en un PDF."),
            ("APP MÓVIL", "Watermelon Live en el celular",
             "Flota, salud, pronóstico y alertas en el bolsillo."),
            ("SUSCRIPCIÓN", "Su plan, facturas y pagos",
             "PSE, Bre-B, tarjeta, Nequi o transferencia, en un solo lugar."),
        ],
        "outro": ("Piloto de 30 días sin costo",
                  "Sensores incluidos en el plan · contrato desde 12 meses",
                  "watermelonsystem.com   ·   WhatsApp +57 300 888 8883"),
    },
    "en": {
        "intro": ("Live turbomachinery monitoring",
                  "Real walkthrough" + (f" · {ASSET}" if ASSET else " · real data")),
        "s": [
            ("LIVE", "Your turbogenerator, live",
             "Speed, vibration and asset health, refreshed every 10 seconds."),
            ("MACHINE TRAIN", "Every sensor on the real train",
             "Turbine, gearbox and generator with reading and severity per point."),
            ("API 670", "Vibration table per channel",
             "ISO zone, % of alarm, trend and 1X vector for every sensor."),
            ("HISTORY", "Trends over days, months and years",
             "Start-ups, trips and load changes in plain sight."),
            ("EVENTS", "A log of every threshold crossing",
             "Time, value and duration in alarm: nothing is missed."),
            ("ROTORDYNAMICS", "Real waveform",
             "Processed by our rotordynamics engine, keyphasor on every turn."),
            ("ROTORDYNAMICS", "FFT spectrum with 1X–4X orders",
             "The basis to diagnose unbalance, misalignment, looseness and more."),
            ("ROTORDYNAMICS", "Orbits of all five bearings",
             "Same scale, probes at their real angle, API 670 rotation."),
            ("ROTORDYNAMICS", "Shape and size of every orbit",
             "Major axis, minor axis, ellipticity and Smax per bearing."),
            ("REPORTS", "Weekly report signed by a specialist",
             "Findings, recommendations, orbits, Bode and polar in one PDF."),
            ("MOBILE APP", "Watermelon Live on your phone",
             "Fleet, health, forecast and alerts in your pocket."),
            ("SUBSCRIPTION", "Your plan, invoices and payments",
             "Card, bank transfer and local methods, all in one place."),
        ],
        "outro": ("30-day pilot at no cost",
                  "Sensors included in the plan · 12-month minimum term",
                  "watermelonsystem.com   ·   WhatsApp +57 300 888 8883"),
    },
}[LANG]

FULL = (560, 0, 2880, 1450)


def ease(t):
    return t * t * (3 - 2 * t)


def lerp_rect(a, b, t):
    return tuple(a[i] + (b[i] - a[i]) * t for i in range(4))


def kb_frame(img, r):
    x0, y0, x1, y1 = r
    return img.resize((W, H), Image.BICUBIC, box=(x0, y0, x1, y1))


# ---------- lienzos compuestos ----------
def canvas_report():
    c = Image.new("RGB", (SRC_W, SRC_H), NAVY)
    g = Image.new("L", (SRC_W, SRC_H), 0)
    ImageDraw.Draw(g).ellipse((600, -400, 2600, 1400), fill=90)
    c.paste(Image.new("RGB", c.size, BLUE), (0, 0), g.filter(ImageFilter.GaussianBlur(260)))
    pages = ["rep_05.png", "rep_07.png", "rep_13.png", "rep_27.png"]
    h = 1500
    xs = [180, 840, 1500, 2160]
    for i, (p, x) in enumerate(zip(pages, xs)):
        im = Image.open(f"{SHOTS}/{p}").convert("RGB")
        im = im.resize((int(im.width * h / im.height), h), Image.LANCZOS)
        y = 170 if i % 2 == 0 else 230
        sh = Image.new("L", (im.width + 120, im.height + 120), 0)
        ImageDraw.Draw(sh).rectangle((60, 70, im.width + 60, im.height + 70), fill=150)
        c.paste((0, 0, 0), (x - 60, y - 60), sh.filter(ImageFilter.GaussianBlur(30)))
        c.paste(im, (x, y))
    return c


def phone(path, h):
    im = Image.open(f"{SHOTS}/{path}").convert("RGB")
    im = im.resize((int(im.width * h / im.height), h), Image.LANCZOS)
    pad = 26
    fr = Image.new("RGB", (im.width + 2 * pad, im.height + 2 * pad), (20, 24, 32))
    m = Image.new("L", fr.size, 0)
    ImageDraw.Draw(m).rounded_rectangle((0, 0, fr.width - 1, fr.height - 1), 90, fill=255)
    sm = Image.new("L", im.size, 0)
    ImageDraw.Draw(sm).rounded_rectangle((0, 0, im.width - 1, im.height - 1), 64, fill=255)
    fr.paste(im, (pad, pad), sm)
    return fr, m


def canvas_mobile():
    c = Image.new("RGB", (SRC_W, SRC_H), NAVY)
    g = Image.new("L", (SRC_W, SRC_H), 0)
    ImageDraw.Draw(g).ellipse((400, 200, 2500, 1900), fill=110)
    c.paste(Image.new("RGB", c.size, BLUE), (0, 0), g.filter(ImageFilter.GaussianBlur(280)))
    shots = ["11_mob_fleet.png", "11_mob_asset_0.png", "11_mob_asset_1.png"]
    h = 1420
    phones = [phone(s, h) for s in shots]
    gap = 110
    total = sum(p.width for p, _ in phones) + gap * 2
    x = (SRC_W - total) // 2
    for i, (p, m) in enumerate(phones):
        y = 150 if i == 1 else 230
        sh = Image.new("L", (p.width + 160, p.height + 160), 0)
        ImageDraw.Draw(sh).rounded_rectangle((80, 100, p.width + 80, p.height + 100), 90, fill=170)
        c.paste((0, 0, 0), (x - 80, y - 80), sh.filter(ImageFilter.GaussianBlur(40)))
        c.paste(p, (x, y), m)
        x += p.width + gap
    return c


def card(title, sub, extra=None, logo_h=210):
    c = Image.new("RGB", (W, H), PAPER)
    d = ImageDraw.Draw(c)
    lg = Image.open(LOGO).convert("RGBA")
    lg = lg.resize((int(lg.width * logo_h * 2 / lg.height), logo_h * 2), Image.LANCZOS)
    lg = lg.resize((lg.width // 2, lg.height // 2), Image.LANCZOS)
    top = 220 if extra is None else 170
    c.paste(lg, ((W - lg.width) // 2, top), lg)
    y = top + lg.height + 50
    d.text((W // 2, y), title, font=F_BIG, fill=INK, anchor="ma")
    d.text((W // 2, y + 92), sub, font=F_MID, fill=(80, 98, 128), anchor="ma")
    if extra:
        bw = d.textlength(extra, font=F_SMALL) + 70
        bx = (W - bw) // 2
        d.rounded_rectangle((bx, y + 170, bx + bw, y + 236), 33, fill=NAVY)
        d.text((W // 2, y + 203), extra, font=F_SMALL, fill=(255, 255, 255), anchor="mm")
    d.rectangle((0, H - 10, W, H), fill=GREEN)
    return c


def caption(frame, kick, title, sub, alpha):
    if alpha <= 0:
        return frame
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    tw = max(d.textlength(title, font=F_TITLE), d.textlength(sub, font=F_SUB)) + 96
    x0, y1 = 44, H - 44
    y0 = y1 - 172
    a = int(235 * alpha)
    d.rounded_rectangle((x0, y0, x0 + tw, y1), 22, fill=NAVY + (a,))
    d.ellipse((x0 + 36, y0 + 38, x0 + 50, y0 + 52), fill=GREEN + (int(255 * alpha),))
    d.text((x0 + 64, y0 + 33), kick, font=F_KICK, fill=(130, 200, 160, int(255 * alpha)))
    d.text((x0 + 36, y0 + 64), title, font=F_TITLE, fill=(255, 255, 255, int(255 * alpha)))
    d.text((x0 + 36, y0 + 120), sub, font=F_SUB, fill=(190, 208, 232, int(255 * alpha)))
    out = frame.convert("RGBA")
    out.alpha_composite(ov)
    return out.convert("RGB")


# ---------- escenas ----------
def img(name):
    return Image.open(f"{SHOTS}/{name}").convert("RGB")


S = T["s"]


def sub_clean():
    """Suscripción sin la oferta comercial ni recomendaciones de su reporte (privado del cliente)."""
    im = img("10_sub_a.png")
    bg = im.getpixel((2860, 1700))
    ImageDraw.Draw(im).rectangle((520, 1360, 2880, 1800), fill=bg)
    return im
scenes = [
    # (fuente, dur, rect_ini, rect_fin, caption_idx)
    (img("01_top.png"), 5.2, (0, 0, 2880, 1800), (640, 400, 2720, 1700), 0),
    (img("02_train.png"), 5.0, (560, 0, 2800, 1400), (700, 60, 2660, 1285), 1),
    (img("03_tabular.png"), 5.0, FULL, (660, 80, 2260, 1080), 2),
    (img("04_trend.png"), 5.0, FULL, (640, 400, 2880, 1800), 3),
    (img("05_events.png"), 4.6, FULL, (660, 60, 2260, 1060), 4),
    (img("06_waveform.png"), 5.0, FULL, (620, 180, 2880, 1592), 5),
    (img("07_spectrum.png"), 5.0, FULL, (700, 300, 2300, 1300), 6),
    (img("08_orbit_a.png"), 5.0, FULL, (640, 180, 2880, 1580), 7),
    (img("08_orbit_b.png"), 4.4, FULL, (560, 350, 2880, 1800), 8),
    (canvas_report(), 5.4, (0, 0, 2880, 1800), (300, 120, 2580, 1545), 9),
    (canvas_mobile(), 5.4, (0, 0, 2880, 1800), (240, 60, 2640, 1560), 10),
    (sub_clean(), 5.0, FULL, (640, 300, 2760, 1625), 11),
]

intro = card(*T["intro"])
outro = card(T["outro"][0], T["outro"][1], T["outro"][2], logo_h=170)


def scene_frames(src, dur, r0, r1, ci):
    n = int(dur * FPS)
    k, ti, su = S[ci]
    for i in range(n):
        t = i / (n - 1)
        f = kb_frame(src, lerp_rect(r0, r1, ease(t)))
        ca = min(1.0, max(0.0, (i / FPS - 0.45) / 0.45)) * min(1.0, max(0.0, (dur - i / FPS - 0.15) / 0.4))
        yield caption(f, k, ti, su, ca)


def still(c, dur):
    for _ in range(int(dur * FPS)):
        yield c


def timeline():
    """Encadena escenas con fundido de XF s, guardando solo nb frames en memoria."""
    from collections import deque
    gens = [still(intro, 3.2)] + [scene_frames(*s) for s in scenes] + [still(outro, 4.8)]
    nb = int(XF * FPS)
    tail = deque()
    for gi, g in enumerate(gens):
        g = iter(g)
        if tail:                                   # fundido con la escena anterior
            for i in range(len(tail)):
                fb = next(g)
                yield Image.blend(tail.popleft(), fb, (i + 1) / (nb + 1))
        last = gi == len(gens) - 1
        for fr in g:
            tail.append(fr)
            if len(tail) > (0 if last else nb):
                yield tail.popleft()
    yield from tail


ff = imageio_ffmpeg.get_ffmpeg_exe()
cmd = [ff, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
       "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "27", "-pix_fmt", "yuv420p",
       "-movflags", "+faststart", "-an", OUT]
p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
nfr = 0
for fr in timeline():
    p.stdin.write(fr.tobytes())
    nfr += 1
p.stdin.close()
p.wait()
print(OUT, nfr, "frames", round(nfr / FPS, 1), "s")
