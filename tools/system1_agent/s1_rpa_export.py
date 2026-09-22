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
log = logging.getLogger("s1_rpa")

WINDOW_TITLE_RE = r".*System 1.*"
EXPORT_MENU_TEXT = "Export to CSV"


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
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args(argv)
    _setup_logging()
    cfg = _load_cfg(args.config)
    if args.inspect:
        return inspect(cfg)
    if args.run:
        return run(cfg, dry=args.dry)
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
