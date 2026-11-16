"""
Watermelon Field — piel HTML embebida (híbrido) + bridge de datos en vivo.

La piel (`skin.html`) se renderiza dentro de un QWebEngineView. Los datos en vivo
se EMPUJAN de Python a JS con `page().runJavaScript("window.applyLive(...)")`
(unidireccional, sin qwebchannel.js). La piel misma nunca se rompe por un payload
malo, y en navegador standalone queda estática (animación CSS).

Fase: BRIDGE PROOF — alimenta la vista Monitoreo con datos SIMULADOS (ondas +
valores + zonas) a ~12 fps. En el app real, `FieldSkin.push()` recibirá las
lecturas del agente/adquisición en vez del simulador.

Uso:
    python native/field_ui/web_skin.py                 # ventana viva (sim)
    python native/field_ui/web_skin.py --frames 3 OUT  # graba 3 PNGs y cierra
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from PySide6 import QtCore, QtWidgets
from PySide6.QtWebEngineWidgets import QWebEngineView


def _skin_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    for base in (here, getattr(sys, "_MEIPASS", ""), os.getcwd()):
        if not base:
            continue
        for rel in ("skin.html", os.path.join("field_ui", "skin.html")):
            p = os.path.join(base, rel)
            if os.path.exists(p):
                return p
    return os.path.join(here, "skin.html")


# Canales de la vista Monitoreo, en el MISMO orden que las lanes del scope.
# (nombre, amplitud base µm, clase-zona CSS, ciclos por ventana)
_SCOPE = [
    ("6XD", 44.1, "vd", 3.0),
    ("6YD", 42.7, "vd", 3.0),
    ("5XD", 38.2, "vc", 3.2),
    ("1XD", 11.8, "va", 4.0),
]
_BRG = [  # panel derecho, orden 1,2,caja,5,6
    (11.8, "va"), (13.1, "va"), (2.34, "vb"), (38.2, "vc"), (44.1, "vd"),
]
_N = 150  # muestras por ventana de scope


class FieldSkin(QtWidgets.QWidget):
    """Piel HTML embebida. Reusable como pestaña dentro del app de campo."""

    def __init__(self, parent=None, embed: bool = True):
        super().__init__(parent)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.view = QWebEngineView(self)
        lay.addWidget(self.view)
        self._ph = 0.0
        self._timer = None
        url = QtCore.QUrl.fromLocalFile(_skin_path())
        if embed:               # dentro de un tab del app → sin chrome propio (no duplicar)
            url.setQuery("embed=1")
        self.view.load(url)

    # --- API de datos ---------------------------------------------------
    def push(self, payload: dict) -> None:
        """Empuja un payload a la piel (Python -> JS)."""
        try:
            js = "window.applyLive && window.applyLive(" + json.dumps(payload) + ")"
            self.view.page().runJavaScript(js)
        except Exception:  # noqa: BLE001
            pass

    def _sim_payload(self) -> dict:
        self._ph += 0.33
        x = np.linspace(0.0, 1.0, _N)
        subs = ["Gen NDE", "Gen NDE", "Gen DE", "Turbina DE"]
        channels = []
        for (name, base, cls, cyc), sub in zip(_SCOPE, subs):
            s = np.sin(2 * np.pi * cyc * x + self._ph) \
                + 0.12 * np.sin(2 * np.pi * cyc * 2 * x + 1.7 * self._ph) \
                + 0.06 * np.random.randn(_N)
            s = np.clip(s / 1.25, -1.0, 1.0)
            val = base + np.random.randn() * 0.25
            channels.append({"name": name, "sub": sub, "cls": cls, "unit": "µm pp",
                             "val": f"{val:.1f}", "samples": [round(float(v), 3) for v in s]})
        # keyphasor: tren de pulsos que se desplaza
        kp = np.full(_N, -0.55)
        for c in range(3):
            pos = int(((c / 3.0) + (self._ph * 0.05)) % 1.0 * _N)
            kp[pos:pos + 3] = 0.6
        channels.append({"name": "KPH", "sub": "keyphasor", "cls": "", "unit": "rpm",
                         "val": "1803", "samples": [round(float(v), 3) for v in kp]})

        zmap = {"va": ("A", "a"), "vb": ("B", "b"), "vc": ("C", "c"), "vd": ("D", "d")}
        names = ["1 Turbina DE", "2 Turbina NDE", "Caja engrane", "5 Gen DE", "6 Gen NDE"]
        bearings = []
        for (base, cls), nm in zip(_BRG, names):
            jitter = base + np.random.randn() * (0.25 if base > 5 else 0.03)
            z, zc = zmap.get(cls, ("A", "a"))
            bearings.append({"zone": z, "zcls": zc, "name": nm, "cls": cls,
                             "unit": "µm pp" if base > 5 else "g RMS",
                             "val": f"{jitter:.1f}" if base > 5 else f"{jitter:.2f}"})
        return {"rpm": 1803, "channels": channels, "bearings": bearings}

    def start_sim(self, fps: int = 12) -> None:
        if self._timer is not None:
            return
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(lambda: self.push(self._sim_payload()))
        self._timer.start(max(40, int(1000 / fps)))


def main() -> int:
    frames, out = 0, "skin_frame.png"
    if "--frames" in sys.argv:
        i = sys.argv.index("--frames")
        frames = int(sys.argv[i + 1]) if i + 1 < len(sys.argv) else 3
        out = sys.argv[i + 2] if i + 2 < len(sys.argv) else out

    app = QtWidgets.QApplication(sys.argv)
    win = QtWidgets.QMainWindow()
    win.setWindowTitle("Watermelon Rotordynamics — Campo (piel viva)")
    skin = FieldSkin()
    win.setCentralWidget(skin)
    win.resize(1320, 860)
    win.show()

    def _go(ok):
        skin.start_sim(12)
        if frames:
            base, ext = os.path.splitext(out)
            state = {"i": 0}

            def _shoot():
                p = f"{base}_{state['i']}{ext or '.png'}"
                win.grab().save(p)
                print("FRAME", p)
                state["i"] += 1
                if state["i"] < frames:
                    QtCore.QTimer.singleShot(420, _shoot)
                else:
                    app.quit()
            QtCore.QTimer.singleShot(900, _shoot)

    skin.view.loadFinished.connect(_go)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
