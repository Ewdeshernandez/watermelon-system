#!/usr/bin/env python3
"""
Watermelon System1 RPA Exporter — reemplaza a la persona
========================================================

Automatiza el 'clic derecho → Export to CSV' de las formas de onda en el cliente
System1, para que NADIE tenga que hacerlo a mano. Corre en el server Parex, cada
hora (Task Scheduler), y deja los CSV en la carpeta que el agente sube
(s1_agent.py --csv).

Por qué pywinauto: pone el nombre de archivo por **UIA SetValue**, NO por
teclado — así esquiva el remapeo de layout del RDP/VMware anidado.

Flujo por cada punto configurado:
  1. Seleccionar el tag de onda en el árbol de System1 (por nombre).
  2. Clic derecho sobre la gráfica → menú → 'Export to CSV'.
  3. En el diálogo Save As: fijar carpeta + nombre → Save.

Comandos:
  python s1_rpa_export.py --inspect     # vuelca el árbol UIA de System1 (para
                                         # afinar selectores la 1ª vez)
  python s1_rpa_export.py --run          # exporta todos los puntos del config
  python s1_rpa_export.py --run --dry    # simula (no exporta), loguea pasos

Config: config.toml → [rpa] (ver config.example.toml).

Requiere: pip install pywinauto  (solo Windows).
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List

try:
    import tomllib as _toml  # py311+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as _toml    # type: ignore

HERE = Path(__file__).resolve().parent
LOG_DIR = HERE / "logs"
TPL_DIR = HERE / "templates"          # plantillas de etiquetas de nodo (PNG)
log = logging.getLogger("s1_rpa")

WINDOW_TITLE_RE = r".*System 1.*"
EXPORT_MENU_TEXT = "Export to CSV"


def _find_template(screen_img, tpl_path: Path, thr: float = 0.80):
    """Busca una plantilla en la captura de pantalla (matchTemplate).

    Devuelve (score, cx, cy) del mejor match en coords de PANTALLA (= pixeles de
    la captura de escritorio ImageGrab). None si no supera el umbral.
    """
    import numpy as np
    import cv2
    if not tpl_path.exists():
        log.warning("plantilla no existe: %s", tpl_path)
        return None
    scr = cv2.cvtColor(np.array(screen_img), cv2.COLOR_RGB2GRAY)
    tpl = cv2.imread(str(tpl_path), cv2.IMREAD_GRAYSCALE)
    if tpl is None:
        return None
    res = cv2.matchTemplate(scr, tpl, cv2.TM_CCOEFF_NORMED)
    _minv, maxv, _minl, maxl = cv2.minMaxLoc(res)
    h, w = tpl.shape[:2]
    cx, cy = maxl[0] + w // 2, maxl[1] + h // 2
    if maxv < thr:
        return (maxv, cx, cy)   # devuelve igual, el llamador decide por score
    return (maxv, cx, cy)


def _load_cfg(path: str | None) -> dict:
    p = Path(path) if path else (HERE / "config.toml")
    if not p.exists():
        return {}
    with open(p, "rb") as fh:
        return _toml.load(fh)


def _connect():
    from pywinauto import Application
    app = Application(backend="uia").connect(title_re=WINDOW_TITLE_RE, timeout=20)
    win = app.window(title_re=WINDOW_TITLE_RE)
    win.set_focus()
    return app, win


def inspect(cfg: dict) -> int:
    """Lista TreeItems (tags de onda) y panes grandes (candidatos a gráfica).

    En vez de volcar todo el árbol (enorme y lento), enumera lo que el RPA
    necesita: los nodos seleccionables del árbol (con su rectángulo) y los
    contenedores grandes donde se dibuja la onda (para fijar plot_xy).
    """
    app, win = _connect()
    r = win.rectangle()
    print("== System1 window ==")
    print("WINDOW rect L%d T%d R%d B%d  (w=%d h=%d)" % (
        r.left, r.top, r.right, r.bottom, r.width(), r.height()))

    def _rect(c):
        try:
            return c.rectangle()
        except Exception:  # noqa: BLE001
            return None

    def _txt(c):
        try:
            t = c.window_text()
        except Exception:  # noqa: BLE001
            t = ""
        return (t or "").strip()

    # --- TreeItems: los tags de onda seleccionables ---
    print("\n== TreeItems (tag candidatos) ==")
    n = 0
    for ct in ("TreeItem", "ListItem"):
        try:
            items = win.descendants(control_type=ct)
        except Exception as exc:  # noqa: BLE001
            print("  (err %s: %s)" % (ct, exc))
            continue
        for it in items:
            t = _txt(it)
            rc = _rect(it)
            if not t:
                continue
            n += 1
            if rc is not None:
                print("  [%s] '%s'  @cx=%d cy=%d" % (
                    ct, t, (rc.left + rc.right) // 2, (rc.top + rc.bottom) // 2))
            else:
                print("  [%s] '%s'" % (ct, t))
    if n == 0:
        print("  (ninguno — el árbol puede ser Custom/Pane; ver panes abajo)")

    # --- Panes/Custom grandes: candidatos a área de gráfica (para plot_xy) ---
    print("\n== Contenedores grandes (candidatos a gráfica, plot_xy) ==")
    cands = []
    for ct in ("Pane", "Custom", "Document", "Image", "Group"):
        try:
            for c in win.descendants(control_type=ct):
                rc = _rect(c)
                if rc is None:
                    continue
                area = rc.width() * rc.height()
                if area <= 0:
                    continue
                cands.append((area, ct, _txt(c), rc))
        except Exception:  # noqa: BLE001
            pass
    cands.sort(key=lambda c: c[0], reverse=True)
    for area, ct, t, rc in cands[:12]:
        # offset del CENTRO del control relativo a la esquina de la ventana
        ox = (rc.left + rc.right) // 2 - r.left
        oy = (rc.top + rc.bottom) // 2 - r.top
        print("  [%s] area=%d  rect(L%d T%d R%d B%d)  plot_xy=[%d,%d]  '%s'" % (
            ct, area, rc.left, rc.top, rc.right, rc.bottom, ox, oy, t[:40]))

    # --- TODOS los controles con texto (para hallar selectores de onda) ---
    print("\n== Controles con texto (buscar tags de onda tipo 1XD/1YD) ==")
    seen = 0
    try:
        alld = win.descendants()
    except Exception as exc:  # noqa: BLE001
        alld = []
        print("  (err descendants: %s)" % exc)
    for c in alld:
        t = _txt(c)
        if not t or len(t) > 60:
            continue
        try:
            ct = c.element_info.control_type
        except Exception:  # noqa: BLE001
            ct = "?"
        rc = _rect(c)
        seen += 1
        if seen > 120:
            print("  ... (cortado en 120)")
            break
        if rc is not None:
            print("  [%s] '%s'  @cx=%d cy=%d" % (
                ct, t, (rc.left + rc.right) // 2, (rc.top + rc.bottom) // 2))
        else:
            print("  [%s] '%s'" % (ct, t))
    return 0


def _select_tree_tag(win, tag_name: str) -> bool:
    """Selecciona en el árbol el tag de onda por nombre (substring)."""
    from pywinauto.findwindows import ElementNotFoundError
    try:
        item = win.child_window(title_re=f".*{tag_name}.*",
                                control_type="TreeItem")
        item.click_input()
        time.sleep(0.6)
        return True
    except (ElementNotFoundError, Exception) as exc:  # noqa: BLE001
        log.warning("No hallé el tag '%s' en el árbol: %s", tag_name, exc)
        return False


def _export_current_plot(app, win, plot_xy, out_dir: Path, out_name: str,
                         dry: bool) -> bool:
    """Clic derecho en la gráfica → 'Export to CSV' → Save As con nombre."""
    from pywinauto import mouse
    x, y = plot_xy
    rect = win.rectangle()
    px, py = rect.left + x, rect.top + y
    if dry:
        log.info("[dry] right-click (%d,%d) → %s → %s\\%s.csv",
                 px, py, EXPORT_MENU_TEXT, out_dir, out_name)
        return True
    mouse.right_click(coords=(px, py))
    time.sleep(0.5)
    # menú contextual
    try:
        win.child_window(title=EXPORT_MENU_TEXT,
                         control_type="MenuItem").click_input()
    except Exception as exc:  # noqa: BLE001
        log.warning("No hallé '%s' en el menú: %s", EXPORT_MENU_TEXT, exc)
        return False
    time.sleep(0.8)
    # diálogo Save As
    try:
        dlg = app.window(title_re="Save As", top_level_only=False)
        full = str(out_dir / f"{out_name}.csv")
        edit = dlg.child_window(class_name="Edit")
        edit.set_edit_text(full)      # UIA SetValue → sin teclado
        time.sleep(0.2)
        dlg.child_window(title="Save", control_type="Button").click_input()
        time.sleep(0.6)
        # posible confirmación de sobreescritura
        try:
            conf = app.window(title_re="(Confirm Save As|Save As)")
            conf.child_window(title="Yes", control_type="Button").click_input()
        except Exception:  # noqa: BLE001
            pass
        return True
    except Exception as exc:  # noqa: BLE001
        log.error("Falló el Save As de %s: %s", out_name, exc)
        return False


def run(cfg: dict, dry: bool) -> int:
    rpa = cfg.get("rpa", {})
    points: List[dict] = rpa.get("points", [])
    out_dir = Path(rpa.get("out_folder",
                           str(Path.home() / "Desktop" / "CSV")))
    plot_xy = tuple(rpa.get("plot_xy", [640, 360]))  # offset dentro de la ventana
    if not points:
        log.error("Config [rpa].points vacío — nada que exportar.")
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)
    app, win = _connect()
    ok = fail = 0
    for pt in points:
        tag = pt.get("tag", "")
        name = pt.get("name", tag)
        log.info("Exportando %s (tag=%s)", name, tag)
        if not dry and not _select_tree_tag(win, tag):
            fail += 1
            continue
        if _export_current_plot(app, win, plot_xy, out_dir, name, dry):
            ok += 1
        else:
            fail += 1
        time.sleep(0.5)
    log.info("RPA export: %d OK, %d fallidos → %s", ok, fail, out_dir)
    return 0 if fail == 0 else 1


def shot(cfg: dict, out_path: str) -> int:
    """Captura la ventana de System1 a PNG (pixeles nativos, para mapear coords).

    UIA no expone el contenido de System1 (canvas propietario); el RPA irá por
    coordenadas de pantalla. Esta imagen sirve para leer la posición de cada
    nodo del árbol y el centro de la gráfica. El origen (0,0) del PNG = esquina
    top-left de la ventana; coord de pantalla = px_img + win.left.
    """
    app, win = _connect()
    r = win.rectangle()
    img = win.capture_as_image()
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(p))
    print("SHOT %s  (%dx%d)  win.left=%d win.top=%d" % (
        p, img.width, img.height, r.left, r.top))
    print("MAP: screen_x = img_x + (%d) ; screen_y = img_y + (%d)" % (
        r.left, r.top))
    return 0


def rclick(cfg: dict, x: int, y: int) -> int:
    """Clic derecho en (x,y) pantalla y vuelca los MenuItem del popup.

    Sirve para descubrir el texto exacto del 'Export to CSV' (y si hay submenú).
    """
    from pywinauto import mouse, Desktop
    _connect()  # trae System1 al frente
    time.sleep(0.4)
    mouse.right_click(coords=(x, y))
    time.sleep(0.8)
    _grab_screen(r"C:\WM_wave\s1.png")   # ve el menú (ventana aparte)
    print("== MenuItems tras right-click en (%d,%d) ==" % (x, y))
    seen = 0
    try:
        for w in Desktop(backend="uia").windows():
            try:
                for mi in w.descendants(control_type="MenuItem"):
                    t = (mi.window_text() or "").strip()
                    if t:
                        seen += 1
                        print("  MenuItem: '%s'" % t)
            except Exception:  # noqa: BLE001
                pass
    except Exception as exc:  # noqa: BLE001
        print("  (err: %s)" % exc)
    if seen == 0:
        print("  (ninguno — el menú puede ser owner-drawn/no-UIA)")
    return 0


def _grab_screen(path: str):
    """Captura TODO el escritorio (incluye popups/menús/diálogos, que son
    ventanas aparte y NO salen en win.capture_as_image()). Requiere sesión
    interactiva (la tarea /it la provee)."""
    from PIL import ImageGrab
    img = ImageGrab.grab(all_screens=True)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(p))
    return img


def click_probe(cfg: dict, x: int, y: int, out_png: str) -> int:
    """Clic izquierdo en (x,y) y captura de pantalla COMPLETA (ve popups)."""
    from pywinauto import mouse
    app, win = _connect()
    time.sleep(0.3)
    mouse.click(coords=(x, y))
    time.sleep(0.8)
    img = _grab_screen(out_png)
    print("CLICK (%d,%d) -> pantalla completa %s (%dx%d)" % (
        x, y, out_png, img.width, img.height))
    return 0


def _tree_scroll_top(win, tree_x: int, tree_y: int, ticks: int = 40) -> None:
    """Lleva el árbol de System1 a su tope con la rueda del mouse.

    El árbol es un canvas sin scroll UIA; scrolleamos por rueda para tener un
    estado determinista (los offsets de cada nodo son estables desde el tope).
    """
    from pywinauto import mouse
    r = win.rectangle()
    sx, sy = r.left + tree_x, r.top + tree_y
    mouse.move(coords=(sx, sy))
    time.sleep(0.2)
    for _ in range(ticks):
        mouse.scroll(coords=(sx, sy), wheel_dist=1)  # +1 = arriba
        time.sleep(0.15)                              # lento: evita aceleración
    time.sleep(0.4)


def treetop(cfg: dict, out_png: str) -> int:
    """Scroll del árbol al tope + recaptura (para calibrar coords fijas)."""
    app, win = _connect()
    # tree_x/tree_y = offset dentro de la ventana sobre el panel del árbol
    tx = int(cfg.get("rpa", {}).get("tree_x", 100))
    ty = int(cfg.get("rpa", {}).get("tree_y", 300))
    _tree_scroll_top(win, tx, ty)
    img = win.capture_as_image()
    p = Path(out_png)
    p.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(p))
    print("TREETOP scroll@(%d,%d) -> %s (%dx%d)" % (tx, ty, p, img.width, img.height))
    return 0


def pick(cfg: dict, downticks: int, x: int, y: int, out_png: str) -> int:
    """Topea el árbol -> baja N ticks -> clic (x,y) -> recaptura.

    Estado determinista: siempre parte del tope (scroll saturado), baja un número
    fijo de ticks (0 = turbina visible desde el tope; N = trae el generador a una
    posición fija), y clica la coord del nodo. Así la coord es reproducible sin
    importar el estado previo del árbol.
    """
    from pywinauto import mouse
    app, win = _connect()
    tx = int(cfg.get("rpa", {}).get("tree_x", 100))
    ty = int(cfg.get("rpa", {}).get("tree_y", 300))
    _tree_scroll_top(win, tx, ty)                       # canónico: tope
    if downticks:
        r = win.rectangle()
        sx, sy = r.left + tx, r.top + ty
        for _ in range(abs(downticks)):
            mouse.scroll(coords=(sx, sy),
                         wheel_dist=-1 if downticks > 0 else 1)  # -1 = abajo
            time.sleep(0.15)                          # lento: scroll determinista
        time.sleep(0.3)
    mouse.click(coords=(x, y))                          # x,y = pantalla absoluta
    time.sleep(0.8)
    img = win.capture_as_image()
    p = Path(out_png)
    p.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(p))
    print("PICK down=%d click(%d,%d) -> %s (%dx%d)" % (
        downticks, x, y, p, img.width, img.height))
    return 0


def maketpl(cfg: dict, name: str, x0: int, y0: int, x1: int, y1: int) -> int:
    """Captura escritorio y recorta [x0:x1, y0:y1] como templates/NAME.png."""
    _connect()
    time.sleep(0.3)
    img = _grab_screen(r"C:\WM_wave\s1.png")   # también deja la full para ver
    TPL_DIR.mkdir(parents=True, exist_ok=True)
    crop = img.crop((x0, y0, x1, y1))
    out = TPL_DIR / f"{name}.png"
    crop.save(str(out))
    print("MAKETPL %s = [%d,%d,%d,%d] -> %s (%dx%d)" % (
        name, x0, y0, x1, y1, out, crop.width, crop.height))
    return 0


def findtpl(cfg: dict, name: str) -> int:
    """Busca templates/NAME.png en la pantalla actual; imprime score+centro."""
    _connect()
    time.sleep(0.3)
    img = _grab_screen(r"C:\WM_wave\s1.png")
    r = _find_template(img, TPL_DIR / f"{name}.png", thr=0.0)
    if r is None:
        print("FIND %s: plantilla ausente" % name)
        return 1
    score, cx, cy = r
    print("FIND %s: score=%.3f centro=(%d,%d)" % (name, score, cx, cy))
    return 0


def _select_by_template(win, cfg: dict, name: str,
                        thr: float = 0.80, max_scrolls: int = 14) -> bool:
    """Trae el nodo a la vista y lo clica, usando búsqueda por imagen.

    Satura hacia arriba (tope), luego en cada paso: captura → busca la etiqueta;
    si score>=thr clica su centro; si no, scrollea unos ticks abajo y reintenta.
    Tolera el scroll no determinista de System1.
    """
    from pywinauto import mouse
    tx = int(cfg.get("rpa", {}).get("tree_x", 100))
    ty = int(cfg.get("rpa", {}).get("tree_y", 300))
    r = win.rectangle()
    sx, sy = r.left + tx, r.top + ty
    _tree_scroll_top(win, tx, ty)
    tpl = TPL_DIR / f"{name}.png"
    for _i in range(max_scrolls):
        img = _grab_screen(r"C:\WM_wave\s1.png")
        m = _find_template(img, tpl, thr)
        if m is not None and m[0] >= thr:
            score, cx, cy = m
            mouse.click(coords=(cx, cy))
            time.sleep(0.7)
            log.info("select %s: score=%.3f @ (%d,%d)", name, score, cx, cy)
            return True
        for _ in range(3):
            mouse.scroll(coords=(sx, sy), wheel_dist=-1)
            time.sleep(0.15)
        time.sleep(0.2)
    log.warning("select %s: NO encontrado tras %d scrolls", name, max_scrolls)
    return False


def keys_probe(cfg: dict, seq: str, out_png: str) -> int:
    """Enfoca el árbol (clic) y envía una secuencia de teclas; recaptura.

    Las teclas virtuales (HOME/DOWN/ENTER) no sufren el remapeo de layout del
    RDP anidado (eso solo afecta al TEXTO). Sirve para validar navegación por
    teclado del árbol: {HOME}{DOWN 9}{ENTER} debería seleccionar 2YD TURBINA NDE.
    """
    from pywinauto import mouse
    from pywinauto.keyboard import send_keys
    app, win = _connect()
    tx = int(cfg.get("rpa", {}).get("tree_x", 100))
    ty = int(cfg.get("rpa", {}).get("tree_y", 300))
    r = win.rectangle()
    mouse.click(coords=(r.left + tx, r.top + ty))   # foco al árbol
    time.sleep(0.4)
    send_keys(seq, pause=0.03)
    time.sleep(0.8)
    img = win.capture_as_image()
    p = Path(out_png)
    p.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(p))
    print("KEYS '%s' -> recaptura %s (%dx%d)" % (seq, p, img.width, img.height))
    return 0


def _setup_logging():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(LOG_DIR / "s1_rpa.log", encoding="utf-8")])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Watermelon System1 RPA Exporter")
    ap.add_argument("--config")
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--shot", metavar="PNG", nargs="?",
                    const=r"C:\WM_wave\s1.png",
                    help="captura la ventana de System1 a PNG y sale")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--rclick", nargs=2, type=int, metavar=("X", "Y"),
                    help="right-click en (X,Y) pantalla y volcar menú")
    ap.add_argument("--click", nargs=2, type=int, metavar=("X", "Y"),
                    help="left-click en (X,Y) pantalla y recapturar")
    ap.add_argument("--treetop", action="store_true",
                    help="scroll del árbol al tope + recaptura")
    ap.add_argument("--keys", metavar="SEQ",
                    help="clic al árbol + send_keys(SEQ) + recaptura")
    ap.add_argument("--pick", nargs=3, type=int, metavar=("DOWNTICKS", "X", "Y"),
                    help="topea árbol + baja N ticks + clic (X,Y) + recaptura")
    ap.add_argument("--maketpl", nargs=5,
                    metavar=("NAME", "X0", "Y0", "X1", "Y1"),
                    help="recorta la pantalla y guarda templates/NAME.png")
    ap.add_argument("--find", metavar="NAME",
                    help="busca templates/NAME.png en pantalla; imprime score")
    ap.add_argument("--seltest", metavar="NAME",
                    help="selecciona el nodo NAME por imagen + recaptura")
    ap.add_argument("--topshot", action="store_true",
                    help="satura scroll arriba (tope) + captura de escritorio")
    args = ap.parse_args(argv)
    _setup_logging()
    cfg = _load_cfg(args.config)
    if args.shot:
        return shot(cfg, args.shot)
    if args.treetop:
        return treetop(cfg, r"C:\WM_wave\s1.png")
    if args.keys:
        return keys_probe(cfg, args.keys, r"C:\WM_wave\s1.png")
    if args.pick:
        return pick(cfg, args.pick[0], args.pick[1], args.pick[2],
                    r"C:\WM_wave\s1.png")
    if args.maketpl:
        n, x0, y0, x1, y1 = args.maketpl
        return maketpl(cfg, n, int(x0), int(y0), int(x1), int(y1))
    if args.find:
        return findtpl(cfg, args.find)
    if args.topshot:
        app, win = _connect()
        tx = int(cfg.get("rpa", {}).get("tree_x", 100))
        ty = int(cfg.get("rpa", {}).get("tree_y", 300))
        _tree_scroll_top(win, tx, ty)
        img = _grab_screen(r"C:\WM_wave\s1.png")
        print("TOPSHOT -> C:\\WM_wave\\s1.png (%dx%d)" % (img.width, img.height))
        return 0
    if args.seltest:
        app, win = _connect()
        ok = _select_by_template(win, cfg, args.seltest)
        _grab_screen(r"C:\WM_wave\s1.png")
        print("SELTEST %s: %s" % (args.seltest, "OK" if ok else "NO"))
        return 0 if ok else 1
    if args.rclick:
        return rclick(cfg, args.rclick[0], args.rclick[1])
    if args.click:
        return click_probe(cfg, args.click[0], args.click[1],
                           r"C:\WM_wave\s1.png")
    if args.inspect:
        return inspect(cfg)
    if args.run:
        return run(cfg, dry=args.dry)
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
