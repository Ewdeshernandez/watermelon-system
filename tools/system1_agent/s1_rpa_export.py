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


def _hide_console():
    """Minimiza la ventana de consola (cmd.exe de la tarea) para que NO tape
    System1 en la captura ni intercepte los clics."""
    try:
        import ctypes
        hwnd = ctypes.windll.kernel32.GetConsoleWindow()
        if hwnd:
            ctypes.windll.user32.ShowWindow(hwnd, 6)   # SW_MINIMIZE
    except Exception:  # noqa: BLE001
        pass


def _force_foreground(win):
    """Trae System1 al FRENTE aunque otra ventana esté encima. Windows bloquea
    SetForegroundWindow desde background, así que se adjunta el hilo de la
    ventana en primer plano + ALT truco, se maximiza y se sube al tope."""
    try:
        import win32gui, win32con, win32process, win32api, ctypes
        hwnd = int(win.handle)
        try:
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
            win32gui.ShowWindow(hwnd, win32con.SW_MAXIMIZE)
        except Exception:  # noqa: BLE001
            pass
        fg = win32gui.GetForegroundWindow()
        if fg and fg != hwnd:
            t_fg, _ = win32process.GetWindowThreadProcessId(fg)
            t_me = win32api.GetCurrentThreadId()
            t_wn, _ = win32process.GetWindowThreadProcessId(hwnd)
            for a, b in ((t_me, t_fg), (t_me, t_wn)):
                try:
                    ctypes.windll.user32.AttachThreadInput(a, b, True)
                except Exception:  # noqa: BLE001
                    pass
            try:
                # ALT despierta el permiso de foreground
                win32api.keybd_event(0x12, 0, 0, 0)
                win32api.keybd_event(0x12, 0, win32con.KEYEVENTF_KEYUP, 0)
                win32gui.BringWindowToTop(hwnd)
                win32gui.SetForegroundWindow(hwnd)
            except Exception:  # noqa: BLE001
                pass
            for a, b in ((t_me, t_fg), (t_me, t_wn)):
                try:
                    ctypes.windll.user32.AttachThreadInput(a, b, False)
                except Exception:  # noqa: BLE001
                    pass
    except Exception:  # noqa: BLE001
        pass


def _connect():
    from pywinauto import Application
    _hide_console()
    app = Application(backend="uia").connect(title_re=WINDOW_TITLE_RE, timeout=20)
    win = app.window(title_re=WINDOW_TITLE_RE)
    try:
        win.maximize()
    except Exception:  # noqa: BLE001
        pass
    _force_foreground(win)
    try:
        win.set_focus()
    except Exception:  # noqa: BLE001
        pass
    time.sleep(0.6)
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


def _grab():
    """Captura TODO el escritorio a memoria (sin tocar disco). Incluye
    popups/menús/diálogos (ventanas aparte que NO salen en capture_as_image).
    Requiere sesión interactiva (la tarea /it la provee)."""
    from PIL import ImageGrab
    return ImageGrab.grab(all_screens=True)


def _grab_screen(path: str):
    """Igual que _grab() pero guarda a disco (para inspección). Reintenta si el
    archivo está bloqueado (lo lee el host por SMB)."""
    img = _grab()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(5):
        try:
            img.save(str(p))
            break
        except PermissionError:
            time.sleep(0.4)
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


# Clases Chromium (pgAdmin/Azure) tienen árboles UIA enormes → lentísimas de
# enumerar. NO contienen el menú de System1, así que se saltan. La ventana
# principal de System1 SÍ se enumera: el popup del menú vive en su árbol WPF.
_HEAVY_CLASSES = {"Chrome_WidgetWin_1"}


def _click_menuitem(text: str, app=None, timeout: float = 3.0) -> bool:
    """Clic en un MenuItem por título, RÁPIDO: itera Desktop saltando ventanas
    pesadas (Chromium) y la principal de System1 (árbol WPF enorme). El popup del
    menú es liviano → se halla y se invoca al instante, antes de que se cierre.
    invoke() dispara sin mover mouse/foco (click_input cerraría el popup)."""
    from pywinauto import Desktop
    end = time.time() + timeout
    while time.time() < end:
        try:
            wins = Desktop(backend="uia").windows()
        except Exception:  # noqa: BLE001
            wins = []
        for w in wins:
            try:
                cls = w.element_info.class_name
            except Exception:  # noqa: BLE001
                cls = ""
            if cls in _HEAVY_CLASSES:
                continue
            try:
                items = w.descendants(control_type="MenuItem")
            except Exception:  # noqa: BLE001
                continue
            for mi in items:
                try:
                    if (mi.window_text() or "").strip() != text:
                        continue
                except Exception:  # noqa: BLE001
                    continue
                for how in ("invoke", "select", "click"):
                    try:
                        if how == "invoke":
                            mi.invoke()
                        elif how == "select":
                            mi.select()
                        else:
                            mi.click_input()
                        return True
                    except Exception:  # noqa: BLE001
                        continue
        time.sleep(0.2)
    return False


def _dismiss_dialogs(app=None):
    """Cierra un Save As colgado (Cancel/ESC) de una corrida previa, incluido
    el diálogo Win32 #32770 (que UIA a veces no ve)."""
    from pywinauto import Application
    from pywinauto.keyboard import send_keys
    # win32: cerrar #32770 con Cancel (o ESC)
    for _ in range(3):
        try:
            a = Application(backend="win32").connect(
                class_name="#32770", timeout=0.6)
            d = a.window(class_name="#32770")
            if not d.exists():
                break
            done = False
            for tt in ("Cancel", "&Cancel", "Cancelar", "&Cancelar"):
                try:
                    b = d.child_window(title=tt, class_name="Button")
                    if b.exists():
                        b.click_input()
                        done = True
                        break
                except Exception:  # noqa: BLE001
                    continue
            if not done:
                try:
                    d.set_focus()
                    send_keys("{ESC}")
                except Exception:  # noqa: BLE001
                    pass
            time.sleep(0.5)
        except Exception:  # noqa: BLE001
            break


def _find_save_button(w):
    """Devuelve el botón Save/Guardar de una ventana (Button o SplitButton)."""
    for ct in ("Button", "SplitButton"):
        for title in ("Save", "Guardar", "&Save", "&Guardar"):
            try:
                b = w.child_window(title=title, control_type=ct)
                if b.exists(timeout=0.15):
                    return b
            except Exception:  # noqa: BLE001
                continue
    return None


def _is_file_dialog(w) -> bool:
    """True si w es un diálogo de archivo estándar de Windows (class #32770)
    o tiene botón Save/Guardar."""
    try:
        if w.element_info.class_name == "#32770":
            return True
    except Exception:  # noqa: BLE001
        pass
    return _find_save_button(w) is not None


def _find_file_dialog(app):
    """Busca el diálogo Save As (#32770) vía app y Desktop. None si no está."""
    from pywinauto import Desktop
    # 1) rápido: por clase, vía app (incluye diálogos propios) y Desktop
    for finder in (
        lambda: app.window(class_name="#32770") if app is not None else None,
        lambda: Desktop(backend="uia").window(class_name="#32770"),
    ):
        try:
            d = finder()
            if d is not None and d.exists(timeout=0.3):
                return d
        except Exception:  # noqa: BLE001
            continue
    # 2) fallback: iterar top-level, clase primero (rápido), botón después
    try:
        for w in Desktop(backend="uia").windows():
            try:
                if w.element_info.class_name == "#32770":
                    return w
            except Exception:  # noqa: BLE001
                pass
    except Exception:  # noqa: BLE001
        pass
    return None


def _save_as_win32(full_path: str, timeout: float = 25.0) -> bool:
    """Maneja el diálogo Save As con backend WIN32 (el diálogo de archivo es un
    #32770 clásico Win32; el backend uia lo ve mal). Fija el nombre en la caja
    Edit y da Save. Fallback: teclear el path + Enter en el campo enfocado."""
    from pywinauto import Application
    from pywinauto.keyboard import send_keys
    end = time.time() + timeout
    dlg = None
    while time.time() < end and dlg is None:
        try:
            app32 = Application(backend="win32").connect(
                class_name="#32770", timeout=1)
            dlg = app32.window(class_name="#32770")
            if not dlg.exists():
                dlg = None
        except Exception:  # noqa: BLE001
            dlg = None
            time.sleep(0.5)
    if dlg is None:
        log.error("win32: no apareció el diálogo #32770")
        return False
    try:
        dlg.set_focus()
    except Exception:  # noqa: BLE001
        pass
    time.sleep(0.3)
    # 1) caja de nombre: Edit clásico
    filled = False
    for finder in (
        lambda: dlg.child_window(class_name="Edit"),
        lambda: dlg.child_window(class_name="ComboBox").child_window(
            class_name="Edit"),
    ):
        try:
            e = finder()
            if e.exists():
                e.set_edit_text(full_path)
                filled = True
                break
        except Exception:  # noqa: BLE001
            continue
    # 2) fallback: teclear en el campo enfocado (el diálogo abre con foco ahí)
    if not filled:
        try:
            send_keys("^a{DELETE}")
            send_keys(full_path.replace("(", "{(}").replace(")", "{)}"),
                      with_spaces=True, pause=0.01)
            filled = True
        except Exception:  # noqa: BLE001
            pass
    time.sleep(0.4)

    def _confirm_overwrite():
        try:
            a = Application(backend="win32").connect(
                class_name="#32770", timeout=0.6)
            c = a.window(class_name="#32770")
            if c.exists():
                for tt in ("&Yes", "Yes", "&Sí", "Sí"):
                    try:
                        yb = c.child_window(title=tt, class_name="Button")
                        if yb.exists():
                            yb.click_input()
                            return
                    except Exception:  # noqa: BLE001
                        continue
        except Exception:  # noqa: BLE001
            pass

    # Guardar: clic REAL en el botón Save (b.click() por mensaje no dispara).
    # Verifico por existencia del archivo; reintento con Alt+S y Enter.
    tgt = Path(full_path)
    for attempt in range(4):
        clicked = False
        for title in ("&Save", "Save", "&Guardar", "Guardar"):
            try:
                b = dlg.child_window(title=title, class_name="Button")
                if b.exists():
                    b.click_input()
                    clicked = True
                    break
            except Exception:  # noqa: BLE001
                continue
        if not clicked:
            try:
                dlg.set_focus()
                send_keys("%s")            # Alt+S = acelerador Save
            except Exception:  # noqa: BLE001
                pass
        time.sleep(0.6)
        _confirm_overwrite()
        # esperar a que aparezca el archivo
        for _ in range(6):
            if tgt.exists():
                return True
            time.sleep(0.4)
        # aún no: reintento con Enter en el diálogo enfocado
        try:
            dlg.set_focus()
            send_keys("{ENTER}")
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.6)
        _confirm_overwrite()
        if tgt.exists():
            return True
    return tgt.exists()


def _save_as(app, full_path: str, timeout: float = 25.0) -> bool:
    """Intenta primero backend WIN32 (fiable para #32770). Si falla, cae al
    método UIA."""
    if _save_as_win32(full_path, timeout=timeout):
        return True
    from pywinauto import Desktop
    end = time.time() + 4
    dlg = None
    save_btn = None
    while time.time() < end and dlg is None:
        dlg = _find_file_dialog(app)
        if dlg is not None:
            save_btn = _find_save_button(dlg)
        if dlg is None:
            time.sleep(0.3)
    if dlg is None:
        log.error("no apareció el diálogo Save As. Ventanas top-level:")
        try:
            for w in Desktop(backend="uia").windows():
                try:
                    log.error("  win: '%s' class=%s", w.window_text(),
                              w.element_info.class_name)
                except Exception:  # noqa: BLE001
                    pass
        except Exception:  # noqa: BLE001
            pass
        return False
    # caja de nombre de archivo: ComboBox/Edit "File name" o el primer Edit
    edit = None
    for finder in (
        lambda: dlg.child_window(title_re="(File name|Nombre).*",
                                 control_type="Edit"),
        lambda: dlg.child_window(title_re="(File name|Nombre).*",
                                 control_type="ComboBox"),
        lambda: dlg.child_window(class_name="Edit"),
    ):
        try:
            e = finder()
            if e.exists(timeout=0.4):
                edit = e
                break
        except Exception:  # noqa: BLE001
            continue
    if edit is None:
        try:
            edit = dlg.descendants(control_type="Edit")[0]
        except Exception:  # noqa: BLE001
            edit = None
    if edit is None:
        log.error("no hallé la caja de nombre en Save As")
        return False
    ok_set = False
    for setter in ("set_edit_text", "set_text", "type_keys"):
        try:
            if setter == "type_keys":
                edit.click_input()
                edit.type_keys(full_path, with_spaces=True, set_foreground=True)
            else:
                getattr(edit, setter)(full_path)
            ok_set = True
            break
        except Exception:  # noqa: BLE001
            continue
    if not ok_set:
        log.error("no pude fijar el nombre")
        return False
    time.sleep(0.4)
    try:
        save_btn.click_input()
        return True
    except Exception:  # noqa: BLE001
        try:
            from pywinauto.keyboard import send_keys
            send_keys("{ENTER}")
            return True
        except Exception:  # noqa: BLE001
            return False


def export_one(cfg: dict, x: int, y: int, name: str, out_dir: str) -> int:
    """Exporta UNA gráfica: clic-derecho (x,y) → Export to CSV → Save As nombre."""
    from pywinauto import mouse
    app, win = _connect()
    _dismiss_dialogs(app)                    # cierra Save As colgado previo
    win.set_focus()
    time.sleep(0.3)
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)
    full = str(outp / f"{name}.csv")
    try:
        if outp.joinpath(f"{name}.csv").exists():
            outp.joinpath(f"{name}.csv").unlink()   # limpio para verificar
    except Exception:  # noqa: BLE001
        pass
    ok = False
    clicked_any = False
    for intento in range(2):
        mouse.right_click(coords=(x, y))
        time.sleep(0.7)
        clicked = _click_menuitem(EXPORT_MENU_TEXT, app=app)
        clicked_any = clicked_any or clicked
        if not clicked:
            continue
        time.sleep(2.0)
        ok = _save_as(app, full)
        if ok:
            break
    if not clicked_any:
        print("EXPORT %s: no hallé 'Export to CSV' en el menú" % name)
    time.sleep(1.2)
    exists = Path(full).exists()
    print("EXPORT %s: menu=%s saveas=%s archivo=%s (%s)" % (
        name, clicked_any, ok, exists, full))
    _grab_screen(r"C:\WM_wave\s1.png")
    return 0 if exists else 1


def cropfile(src: str, name: str, x0: int, y0: int, x1: int, y1: int) -> int:
    """Recorta [x0:x1,y0:y1] de un PNG existente → templates/NAME.png.
    Sirve para plantillas de menús (el menú se cierra si se regraba)."""
    from PIL import Image
    img = Image.open(src)
    TPL_DIR.mkdir(parents=True, exist_ok=True)
    crop = img.crop((x0, y0, x1, y1))
    out = TPL_DIR / f"{name}.png"
    crop.save(str(out))
    print("CROPFILE %s [%d,%d,%d,%d] -> %s (%dx%d)" % (
        name, x0, y0, x1, y1, out, crop.width, crop.height))
    return 0


def _menu_click_by_template(name: str, thr: float = 0.72) -> bool:
    """Con el menú abierto: captura, halla la etiqueta del item por imagen y
    hace un click de MOUSE REAL en su centro (dispara el comando WPF; invoke UIA
    no lo hacía)."""
    from pywinauto import mouse
    tpl = TPL_DIR / f"{name}.png"
    if not tpl.exists():
        return False
    img = _grab()
    m = _find_template(img, tpl, thr)
    if m is not None and m[0] >= thr:
        _s, cx, cy = m
        mouse.click(coords=(cx, cy))
        return True
    return False


# Grilla 4x2 de la hoja "Export csv1" de System1 (coords de pantalla del CENTRO
# de cada gráfica). Fila1 = turbina, Fila2 = generador. Calibrado 1740x882.
DEFAULT_TILES = [
    {"name": "1yd", "x": 384, "y": 405},
    {"name": "1xd", "x": 742, "y": 405},
    {"name": "2yd", "x": 1100, "y": 405},
    {"name": "2xd", "x": 1458, "y": 405},
    {"name": "5yd", "x": 384, "y": 680},
    {"name": "5xd", "x": 742, "y": 680},
    {"name": "6yd", "x": 1100, "y": 680},
    {"name": "6xd", "x": 1458, "y": 680},
]


def _activate_sheet(win, sheet_name: str, cfg: dict | None = None) -> bool:
    """Hace clic en la pestaña de plots (ej 'Export csv1') para GARANTIZAR que la
    hoja correcta esté activa antes de exportar.

    Por qué: alguien pudo dejar OTRA hoja abierta y encima (órbita, overview,
    Export csv2…). El RPA exporta por coordenada fija de la parrilla 4×2 de
    'Export csv1'; si está otra hoja, el clic-derecho cae en la gráfica
    equivocada y su menú NO tiene 'Export to CSV' → 0/8. Aquí buscamos el
    control por su TEXTO en el árbol UIA de System1 (robusto al tamaño de
    ventana) y lo clicamos. Fallback: coord de pantalla en cfg[rpa].sheet_xy.
    """
    if not sheet_name:
        return False
    target = sheet_name.strip().lower()
    try:
        cands = win.descendants()
    except Exception:  # noqa: BLE001
        cands = []
    best = None                       # (area, ctrl, rect) — el más chico = pestaña
    for c in cands:
        try:
            t = (c.window_text() or "").strip()
        except Exception:  # noqa: BLE001
            continue
        if not t:
            continue
        tl = t.lower()
        # exacto o 'Export csv1' contenido; NO matchea 'Export csv2'
        if tl == target or target in tl:
            try:
                rc = c.rectangle()
                if rc.width() <= 0 or rc.height() <= 0:
                    continue
            except Exception:  # noqa: BLE001
                continue
            area = rc.width() * rc.height()
            if best is None or area < best[0]:
                best = (area, c, rc)
    from pywinauto import mouse
    if best is not None:
        _area, ctrl, rc = best
        cx, cy = (rc.left + rc.right) // 2, (rc.top + rc.bottom) // 2
        clicked = False
        try:
            ctrl.click_input()
            clicked = True
        except Exception:  # noqa: BLE001
            try:
                mouse.click(coords=(cx, cy))
                clicked = True
            except Exception:  # noqa: BLE001
                clicked = False
        if clicked:
            time.sleep(1.2)            # deja que la hoja se dibuje
            log.info("pestaña '%s' activada @ (%d,%d)", sheet_name, cx, cy)
            print("SHEET '%s': activada @ (%d,%d)" % (sheet_name, cx, cy))
            return True
    # fallback: coordenada de pantalla calibrada en config
    xy = (cfg or {}).get("rpa", {}).get("sheet_xy")
    if xy:
        try:
            mouse.click(coords=(int(xy[0]), int(xy[1])))
            time.sleep(1.2)
            print("SHEET '%s': activada por coord %s" % (sheet_name, tuple(xy)))
            return True
        except Exception:  # noqa: BLE001
            pass
    log.warning("no pude activar la pestaña '%s'", sheet_name)
    print("SHEET '%s': NO activada (UIA no la vio; fija cfg[rpa].sheet_xy)"
          % sheet_name)
    return False


def export_all(cfg: dict, out_dir: str, sheet: str | None = None) -> int:
    """Exporta las 8 gráficas de la hoja (clic-derecho→Export→Save As) por
    coordenada fija. Sin árbol ni scroll.

    Antes de nada ACTIVA la hoja correcta ('Export csv1' por defecto) por si
    quedó otra encima — así el robot no depende de que un humano la deje abierta.
    """
    rpa = cfg.get("rpa", {})
    sheet = sheet or rpa.get("sheet", "Export csv1")
    tiles = rpa.get("tiles", DEFAULT_TILES)
    # traer System1 al frente + activar la hoja de exportación
    try:
        app, win = _connect()
        _dismiss_dialogs(app)                 # cierra Save As colgado previo
        if sheet:
            _activate_sheet(win, sheet, cfg)
    except Exception as exc:  # noqa: BLE001
        log.warning("preparación de hoja falló: %s", exc)
    ok = 0
    results = []
    for t in tiles:
        rc = export_one(cfg, int(t["x"]), int(t["y"]), t["name"], out_dir)
        good = (rc == 0)
        results.append("%s=%s" % (t["name"], "OK" if good else "FAIL"))
        if good:
            ok += 1
        time.sleep(0.5)
    print("EXPORTALL: %d/%d  [%s]" % (ok, len(tiles), " ".join(results)))
    return 0 if ok == len(tiles) else 1


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
                        thr: float = 0.78, max_iter: int = 44) -> bool:
    """Trae el nodo a la vista y lo clica, por búsqueda de imagen BIDIRECCIONAL.

    El wheel scroll de System1 es de dirección/monto inconsistente, así que no
    se asume ni tope ni dirección: en cada paso busca la etiqueta; si no está,
    scrollea; si la región del árbol NO cambió (llegó a un extremo), invierte la
    dirección. Así recorre todo el árbol y encuentra el nodo esté donde esté.
    """
    from pywinauto import mouse
    import numpy as np
    tx = int(cfg.get("rpa", {}).get("tree_x", 100))
    ty = int(cfg.get("rpa", {}).get("tree_y", 300))
    tree_w = int(cfg.get("rpa", {}).get("tree_w", 340))  # ancho panel árbol (px)
    r = win.rectangle()
    sx, sy = r.left + tx, r.top + ty
    tpl = TPL_DIR / f"{name}.png"
    sign = -1
    prev_tree = None
    for _i in range(max_iter):
        img = _grab()                    # memoria: NO tocar disco en el loop
        m = _find_template(img, tpl, thr)
        if m is not None and m[0] >= thr:
            score, cx, cy = m
            mouse.click(coords=(cx, cy))
            time.sleep(0.7)
            log.info("select %s: score=%.3f @ (%d,%d) iter=%d",
                     name, score, cx, cy, _i)
            return True
        tree_region = np.array(img)[:, :tree_w].copy()
        if prev_tree is not None and np.array_equal(tree_region, prev_tree):
            sign = -sign                 # árbol no se movió: extremo → invertir
            log.info("select %s: extremo, invierto dirección -> %d", name, sign)
        prev_tree = tree_region
        for _ in range(3):
            mouse.scroll(coords=(sx, sy), wheel_dist=sign)
            time.sleep(0.15)
        time.sleep(0.25)
    log.warning("select %s: NO encontrado tras %d iter", name, max_iter)
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
    ap.add_argument("--grab", action="store_true",
                    help="solo captura de escritorio (sin interactuar)")
    ap.add_argument("--exp1", nargs=4, metavar=("X", "Y", "NAME", "OUTDIR"),
                    help="exporta UNA gráfica: right-click (X,Y)->CSV->NAME")
    ap.add_argument("--expall", metavar="OUTDIR",
                    help="exporta las 8 gráficas de la hoja a OUTDIR")
    ap.add_argument("--sheet", metavar="NAME", default=None,
                    help="pestaña de plots a activar antes de exportar "
                         "(def 'Export csv1'); a prueba de 'Export csv2'")
    ap.add_argument("--actsheet", metavar="NAME",
                    help="SOLO activa la pestaña NAME + screenshot (diagnóstico)")
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
    if args.exp1:
        x, y, name, outd = args.exp1
        return export_one(cfg, int(x), int(y), name, outd)
    if args.actsheet:
        app, win = _connect()
        ok = _activate_sheet(win, args.actsheet, cfg)
        _grab_screen(r"C:\WM_wave\s1.png")
        print("ACTSHEET %s: %s" % (args.actsheet, "OK" if ok else "NO"))
        return 0 if ok else 1
    if args.expall:
        return export_all(cfg, args.expall, sheet=args.sheet)
    if args.grab:
        _connect()
        img = _grab_screen(r"C:\WM_wave\s1.png")
        print("GRAB -> C:\\WM_wave\\s1.png (%dx%d)" % (img.width, img.height))
        return 0
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
