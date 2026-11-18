"""
Watermelon Field — Constructor de tren (librería de equipos) embebido.

Carga `builder.html` (librería de equipos premium + canvas del tren + config de
cojinetes/tipo que recomienda sensores por API 670 / ISO 20816) en un
QWebEngineView. Cuando el usuario arma el tren y da "Construir máquina", el HTML
deja el spec en `window.__wm_build`; este widget lo recoge por polling y emite
`machineBuilt(dict)`.

El spec tiene forma:
    {"machines":[{"type","name","bearings","bearing_type","bno":[...]}...],
     "sensors":[{"n","k","ang","comp","rack"}...]}
"""
from __future__ import annotations

import json
import os
import sys

from PySide6 import QtCore, QtWidgets
from PySide6.QtWebEngineWidgets import QWebEngineView


def _builder_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    for base in (here, getattr(sys, "_MEIPASS", ""), os.getcwd()):
        if not base:
            continue
        for rel in ("builder.html", os.path.join("field_ui", "builder.html")):
            p = os.path.join(base, rel)
            if os.path.exists(p):
                return p
    return os.path.join(here, "builder.html")


class TrainBuilder(QtWidgets.QWidget):
    """Constructor de tren embebido. Emite machineBuilt(spec) al dar Construir."""

    machineBuilt = QtCore.Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.view = QWebEngineView(self)
        lay.addWidget(self.view)
        self.view.load(QtCore.QUrl.fromLocalFile(_builder_path()))
        # Polling del spec (JS -> Python, sin qwebchannel.js)
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._poll)
        self._timer.start(500)

    def _poll(self):
        try:
            self.view.page().runJavaScript(
                "(function(){var b=window.__wm_build;window.__wm_build='';return b||'';})()",
                self._got)
        except Exception:  # noqa: BLE001
            pass

    def _got(self, s):
        if not s:
            return
        try:
            spec = json.loads(s)
        except Exception:  # noqa: BLE001
            return
        if isinstance(spec, dict) and spec.get("machines"):
            self.machineBuilt.emit(spec)


class TrainViewer(QtWidgets.QWidget):
    """Vista read-only del tren premium — MISMO render que el Constructor.

    Carga `builder.html#view` (oculta librería/buildbar/config, solo el tren) y
    expone `load_spec(spec)` que llama a `window.wmLoadSpec`. Se embebe en
    'Sensors & layout' para que el dibujo sea idéntico al que armó el usuario.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.view = QWebEngineView(self)
        lay.addWidget(self.view)
        self._ready = False
        self._pending = None
        self.view.loadFinished.connect(self._on_load)
        url = QtCore.QUrl.fromLocalFile(_builder_path())
        url.setFragment("view")                 # modo visor read-only
        self.view.load(url)

    def _on_load(self, ok):
        self._ready = bool(ok)
        if self._ready and self._pending is not None:
            self.load_spec(self._pending)
            self._pending = None

    def load_spec(self, spec: dict):
        """Pinta el tren del spec. Si la página aún no cargó, lo deja pendiente."""
        if not self._ready:
            self._pending = spec
            return
        try:
            self.view.page().runJavaScript("window.wmLoadSpec(" + json.dumps(spec) + ");")
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = QtWidgets.QMainWindow()
    b = TrainBuilder()
    b.machineBuilt.connect(lambda sp: print("MACHINE BUILT:", len(sp.get("machines", [])),
                                            "máquinas,", len(sp.get("sensors", [])), "sondas"))
    w.setCentralWidget(b)
    w.resize(1320, 860)
    w.show()
    raise SystemExit(app.exec())
