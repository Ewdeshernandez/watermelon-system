"""
Watermelon Field — módulo NATIVO industrial de adquisición y monitoreo
======================================================================

App de escritorio (PySide6 + pyqtgraph): menú, barra de herramientas, pestañas
(Configuración · Monitoreo · Tabular · Espectro) y barra de estado — estilo
estación de análisis (System1/ADRE), pero abierto y con nube. Reusa el motor de
core/remote_monitoring. Tiempo real nativo, sin navegador.

Correr:
    python native/watermelon_field.py --sim          # demo (Mac/dev)
    python native/watermelon_field.py --sens 100 --fs 5120   # NI real (Windows)
"""
from __future__ import annotations

import argparse
import html
import os
import sys
import threading

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.remote_monitoring.agent import AcqAgent
from core.remote_monitoring.stream_source import (ChannelConfig, StreamConfig,
                                                  SimulatedStreamSource, is_keyphasor_channel)

# --- Paleta System1 (paridad con la WEB): tema claro instrumento ---
BG = "#eef2f7"       # fondo app (gris azulado claro, como la web)
PANEL = "#ffffff"    # paneles / plots (blancos, como System1)
PANEL2 = "#e7edf6"   # hover / panel elevado
LINE = "#d6deea"     # bordes / grid
INK = "#1f2937"      # texto principal
MUTE = "#64748b"     # texto secundario
ACC = "#2f6fb0"      # acento azul (web)
NAVY = "#0F1E3D"     # chrome (menú/toolbar/status) + títulos
BLUE = "#2f6fb0"
CORN = "#4f8fd0"     # traza primaria (cornflower, System1)
AMBER = "#e08a1e"    # traza secundaria (ámbar) para fase/espectro
GREEN = "#2fa36b"    # órbita
REDL = "#c0392b"     # 1X / peligro
KPH = "#12467f"      # puntos keyphasor (azul profundo, System1)
# Colores por TIPO de sensor (idénticos a la web: bolitas de la sección)
SENSOR_COLORS = {"prox": "#8b5cf6", "vel": "#22b8cf", "accel": "#ef4444", "keyphasor": "#f59e0b"}
SENSOR_LABELS = {"prox": "Proximity", "vel": "Velocity", "accel": "Accelerometer", "keyphasor": "Keyphasor"}


def _asset_path(rel: str) -> str:
    """Ruta a un asset tanto en dev como en el .exe (PyInstaller _MEIPASS)."""
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(base, rel)


_LOGO_URI_CACHE = {}


def _logo_data_uri() -> str:
    """Logo Watermelon como data URI (base64) para incrustar en el HTML del reporte.
    Cachea el resultado; devuelve '' si no se encuentra el asset."""
    if "uri" in _LOGO_URI_CACHE:
        return _LOGO_URI_CACHE["uri"]
    uri = ""
    try:
        import base64
        for rel in ("assets/watermelon_logo.png", "watermelon_logo.png"):
            p = _asset_path(rel)
            if os.path.exists(p):
                with open(p, "rb") as f:
                    uri = "data:image/png;base64," + base64.b64encode(f.read()).decode("ascii")
                break
    except Exception:  # noqa: BLE001
        uri = ""
    _LOGO_URI_CACHE["uri"] = uri
    return uri


def build_agent(args) -> AcqAgent:
    # Máquina simulada editable (archivo JSON de la biblioteca) — v0.4
    if getattr(args, "machine_file", ""):
        from core.remote_monitoring.sim_machine import SimMachine
        m = SimMachine.load(args.machine_file)
        args.fs = m.fs                     # la máquina manda el muestreo
        args.machine = m.name
        return AcqAgent(SimulatedStreamSource(m.to_stream_config()), instance_id=m.name)

    # Banco de pruebas: escenario sim con nombre (proximidad/accel/velocidad/faults)
    if getattr(args, "scenario", ""):
        from core.remote_monitoring.sim_scenarios import build_scenario
        cfg = build_scenario(args.scenario, fs=args.fs)
        return AcqAgent(SimulatedStreamSource(cfg), instance_id=args.machine)

    idxs = [int(x) for x in args.chans.split(",") if x.strip() != ""]
    names = [s.strip() for s in args.names.split(",")]
    coup = "DC" if args.prox else "IEPE"
    units = "mil pp" if args.prox else "g rms"
    chans = []
    for k, i in enumerate(idxs):
        chans.append(ChannelConfig(name=names[k] if k < len(names) else f"CH{i}", coupling=coup,
                                   sensitivity_mv_per_eu=args.sens, bnc_port=(i + 1), units=units))
    if args.kph_bnc:
        chans.append(ChannelConfig(name="KPH", coupling="DC", sensitivity_mv_per_eu=1.0,
                                   bnc_port=args.kph_bnc, units="pulses/rev"))
    cfg = StreamConfig(sample_rate_hz=args.fs, channels=chans, block_seconds=0.1,
                       buffer_seconds=4.0, rpm=args.rpm, chassis_name=args.chassis,
                       defect=(getattr(args, "defect", "") or "none"))
    if args.sim or getattr(args, "defect", ""):
        source = SimulatedStreamSource(cfg)
    else:
        from core.remote_monitoring.ni_stream_source import NIStreamSource
        source = NIStreamSource(cfg)
    return AcqAgent(source, instance_id=args.machine)


def _spectrum(x, fs):
    x = x - np.mean(x)
    w = np.hanning(len(x))
    mag = np.abs(np.fft.rfft(x * w)) / (np.sum(w) / 2)
    return np.fft.rfftfreq(len(x), 1.0 / fs), mag


def _stylesheet(scale: float = 1.0) -> str:
    # Todas las tipografías/paddings derivan de `scale` → la UI se ajusta al tamaño
    # de pantalla (auto) o a la escala que elija el usuario (menú View → UI scale).
    f = max(10, round(13 * scale))          # texto base
    ft = max(11, round(13 * scale))         # pestañas / botones
    fh = max(9, round(11 * scale))          # encabezados de tabla
    pv = max(4, round(7 * scale)); ph = max(6, round(14 * scale))
    return f"""
    QMainWindow, QDialog {{ background: {BG}; }}
    QWidget {{ background: {BG}; color: {INK};
        font-family: 'Segoe UI', 'Inter', Arial, sans-serif; font-size: {f}px; }}
    QLabel {{ background: transparent; color: {INK}; }}
    QMenuBar {{ background: {NAVY}; color: #eaf1fb; padding: 2px; }}
    QMenuBar::item {{ padding: 5px 10px; background: transparent; color: #eaf1fb; }}
    QMenuBar::item:selected {{ background: {ACC}; border-radius: 5px; color: white; }}
    QMenu {{ background: {PANEL}; border: 1px solid {LINE}; color: {INK}; }}
    QMenu::item:selected {{ background: {ACC}; color: white; }}
    QToolBar {{ background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
        stop:0 #16294a, stop:1 #0d1a30); spacing: 7px; padding: 8px 12px;
        border-bottom: 1px solid #24406e; }}
    QToolBar::separator {{ background: #2b3d5f; width: 1px; margin: 5px 7px; }}
    /* Cada acción como CHIP visible (estilo estación de análisis) — el nombre
       nunca se pierde sobre el fondo oscuro. */
    QToolBar QToolButton {{ color: #eaf1fb; background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.14); padding: {pv}px {ph}px;
        border-radius: 8px; font-weight: 700; font-size: {ft}px; }}
    QToolBar QToolButton:hover {{ background: {ACC}; border-color: {ACC}; color: white; }}
    QToolBar QToolButton:pressed {{ background: {KPH}; border-color: {KPH}; }}
    QToolBar QToolButton:disabled {{ color: #6f7f9c;
        background: rgba(255,255,255,0.03); border-color: rgba(255,255,255,0.06); }}
    QTabWidget::pane {{ border: 1px solid {LINE}; background: {PANEL};
        border-radius: 8px; top: -1px; }}
    QTabBar {{ qproperty-drawBase: 0; }}
    QTabBar::tab {{ background: transparent; color: {MUTE}; padding: 9px 20px;
        margin-right: 3px; border: none; font-weight: 600; font-size: {ft}px;
        border-top-left-radius: 7px; border-top-right-radius: 7px; }}
    QTabBar::tab:hover {{ color: {INK}; background: {PANEL}; }}
    QTabBar::tab:selected {{ background: {PANEL}; color: {ACC};
        border-bottom: 2px solid {ACC}; }}
    QStatusBar {{ background: {NAVY}; color: #b7c6de; border-top: 1px solid {NAVY}; }}
    QStatusBar QLabel {{ color: #b7c6de; padding: 2px 8px; }}
    QPushButton {{ background: {PANEL2}; color: {INK}; border: 1px solid {LINE};
        padding: 7px 15px; border-radius: 7px; font-weight: 600; }}
    QPushButton:hover {{ background: {ACC}; color: {NAVY}; border-color: {ACC}; }}
    QPushButton:disabled {{ background: {PANEL}; color: {MUTE}; }}
    QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox {{ background: {PANEL};
        color: {INK}; border: 1px solid {LINE}; border-radius: 6px; padding: 5px 8px;
        selection-background-color: {ACC}; selection-color: {NAVY}; }}
    QComboBox:hover, QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover {{
        border-color: {ACC}; }}
    QComboBox QAbstractItemView {{ background: {PANEL}; color: {INK};
        selection-background-color: {ACC}; selection-color: {NAVY};
        border: 1px solid {LINE}; }}
    QComboBox::drop-down {{ border: none; width: 20px; }}
    QTableWidget {{ background: {PANEL}; color: {INK}; gridline-color: #eef2f8;
        border: 1px solid {LINE}; border-radius: 10px;
        alternate-background-color: #f6f9fd; }}
    QTableWidget::item {{ padding: 6px 8px; }}
    QTableWidget::item:selected {{ background: #dbe8f7; color: {INK}; }}
    QHeaderView::section {{ background: {NAVY}; color: #8ec3ef; padding: 9px 8px;
        border: none; border-right: 1px solid #24344f; font-weight: 700; font-size: {fh}px;
        text-transform: uppercase; letter-spacing: .04em; }}
    QHeaderView::section:first {{ border-top-left-radius: 9px; }}
    QHeaderView::section:last {{ border-top-right-radius: 9px; border-right: none; }}
    QTextEdit {{ background: {PANEL}; color: {INK}; border: 1px solid {LINE};
        border-radius: 8px; }}
    QScrollBar:vertical {{ background: {BG}; width: 11px; margin: 0; }}
    QScrollBar::handle:vertical {{ background: {LINE}; border-radius: 5px; min-height: 30px; }}
    QScrollBar::handle:vertical:hover {{ background: {ACC}; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
    """


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--scenario", default="",
                    help="banco de pruebas sim: prox_6brg, prox_runup_whip, accel_bpfo, "
                         "accel_bpfi, accel_bsf, gear_mesh, vel_iso_6brg, coastdown, ...")
    ap.add_argument("--defect", default="",
                    help="inyecta defecto (modo manual --sim): unbalance|misalignment|"
                         "looseness|rub|oil_whirl|bearing_bpfo|bearing_bpfi|bearing_bsf|gear_mesh")
    ap.add_argument("--machine-file", default="",
                    help="corre una máquina simulada guardada (JSON de la biblioteca) — v0.4")
    ap.add_argument("--machine", default="Rotor_Kit_Field")
    ap.add_argument("--chassis", default="cDAQ1")
    ap.add_argument("--chans", default="0,1")
    ap.add_argument("--names", default="1YA,1XA")
    ap.add_argument("--sens", type=float, default=100.0)
    ap.add_argument("--fs", type=float, default=5120.0)
    ap.add_argument("--rpm", type=float, default=1475.0)
    ap.add_argument("--prox", action="store_true")
    ap.add_argument("--kph-bnc", type=int, default=0)
    ap.add_argument("--alarm", type=float, default=0.0, help="Nivel de alarma (unid. del canal)")
    ap.add_argument("--danger", type=float, default=0.0, help="Nivel de peligro (unid. del canal)")
    args = ap.parse_args()

    try:
        from PySide6 import QtWidgets, QtCore, QtGui
        import pyqtgraph as pg
    except ImportError:
        print("Install native deps:  pip install -r native/requirements.txt")
        return 2

    agent = build_agent(args)
    vib = [(i, c) for i, c in enumerate(agent.channels) if not is_keyphasor_channel(c)]
    from core.remote_monitoring.recorder import TransientRecorder, upload_recording
    from core.remote_monitoring.transient import TransientCapture, TransientConfig
    from core.remote_monitoring import analysis as diag
    kph_glob = agent.source.config.keyphasor_index()
    from core.remote_monitoring.sim_machine import MODES, MODE_TO_PROFILE
    # Internal mode keys stay Spanish (stored in JSON / used by the engine); the UI
    # shows English labels and maps back to the key.
    MODE_LABELS = {"estable": "Steady", "arranque": "Startup", "parada": "Coastdown",
                   "arranque_parada": "Startup / Coastdown"}
    LABEL_TO_MODE = {v: k for k, v in MODE_LABELS.items()}
    _mode_labels = [MODE_LABELS.get(m, m) for m in MODES]
    tc = TransientCapture(TransientConfig(fmax_hz=min(2000.0, args.fs / 2.5)))

    pg.setConfigOptions(antialias=True, background=PANEL, foreground=MUTE)
    # High-DPI: que Qt escale por densidad de pantalla (monitores 4K/escalados).
    try:
        QtGui.QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
            QtCore.Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    except Exception:  # noqa: BLE001
        pass
    app = QtWidgets.QApplication(sys.argv)

    # --- Auto-ajuste a la pantalla: deriva una escala de UI del tamaño de pantalla ---
    ui_scale = {"v": 1.0}

    def _auto_scale() -> float:
        try:
            scr = app.primaryScreen()
            h = scr.availableGeometry().height()
            # menos alto → UI más compacta; pantallas grandes → un poco más grande
            sc = 0.80 if h < 800 else 0.90 if h < 950 else 1.0 if h < 1150 else \
                 1.12 if h < 1500 else 1.25
            return sc
        except Exception:  # noqa: BLE001
            return 1.0

    def apply_scale(sc):
        ui_scale["v"] = float(sc)
        app.setStyleSheet(_stylesheet(ui_scale["v"]))

    apply_scale(_auto_scale())

    win = QtWidgets.QMainWindow()
    win.setWindowTitle(f"Watermelon Field — {args.machine}")
    win.resize(1360, 860)
    rec_state = {"rec": None}

    # ---------------- Menú ----------------
    mb = win.menuBar()
    m_file = mb.addMenu("&File")
    m_view = mb.addMenu("&View")
    m_help = mb.addMenu("&Help")
    act_start = QtGui.QAction("Start", win)
    act_stop = QtGui.QAction("Stop", win); act_stop.setEnabled(False)
    act_save = QtGui.QAction("Save data", win)
    act_sync = QtGui.QAction("Upload to cloud", win)
    act_clear = QtGui.QAction("Delete data", win)
    act_quit = QtGui.QAction("Quit", win)
    for a in (act_start, act_stop, act_save, act_sync, act_clear):
        m_file.addAction(a)
    m_file.addSeparator(); m_file.addAction(act_quit)
    act_about = QtGui.QAction("About Watermelon Field", win)
    m_help.addAction(act_about)

    # View → UI scale (auto-ajuste a la pantalla; el usuario puede forzar una escala)
    m_scale = m_view.addMenu("UI scale")
    _scale_grp = QtGui.QActionGroup(win); _scale_grp.setExclusive(True)
    _scale_opts = [("Auto (fit screen)", None), ("90%", 0.90), ("100%", 1.0),
                   ("110%", 1.10), ("125%", 1.25), ("150%", 1.50)]

    def _mk_scale(sc):
        def _do():
            apply_scale(_auto_scale() if sc is None else sc)
        return _do
    for _lbl, _sc in _scale_opts:
        _a = QtGui.QAction(_lbl, win, checkable=True); _a.setActionGroup(_scale_grp)
        if _lbl.startswith("Auto"):
            _a.setChecked(True)
        _a.triggered.connect(_mk_scale(_sc))
        m_scale.addAction(_a)

    # ---------------- Toolbar ----------------
    tb = win.addToolBar("Main")
    tb.setMovable(False)
    tb.addAction(act_start); tb.addAction(act_stop); tb.addAction(act_save)
    tb.addSeparator()
    tb.addAction(act_sync); tb.addAction(act_clear)
    tb.addSeparator()
    lbl_modo = QtWidgets.QLabel(" Mode: "); lbl_modo.setStyleSheet("color:white;")
    tb.addWidget(lbl_modo)
    cb_run = QtWidgets.QComboBox(); cb_run.addItems(_mode_labels); tb.addWidget(cb_run)
    spacer = QtWidgets.QWidget(); spacer.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                                       QtWidgets.QSizePolicy.Preferred)
    tb.addWidget(spacer)
    lbl_machine = QtWidgets.QLabel(f"  {args.machine}   ")
    lbl_machine.setStyleSheet("color:white; font-weight:700; font-size:14px;")
    tb.addWidget(lbl_machine)

    # ---------------- Pestañas ----------------
    tabs = QtWidgets.QTabWidget(); win.setCentralWidget(tabs)

    # --- Configuración (editor de máquina simulada — v0.4) ---
    from core.remote_monitoring.sim_machine import (SimMachine, SensorSpec, MODES, PHENOMENA,
                                                    MODE_TO_PROFILE, save_to_library,
                                                    list_machines, load_from_library)
    _KIND_LABELS = [("Proximity", "prox"), ("Velocity", "vel"),
                    ("Accelerometer", "accel"), ("Keyphasor", "keyphasor")]
    _LABEL_BY_KIND = {k: l for l, k in _KIND_LABELS}
    _KIND_BY_LABEL = {l: k for l, k in _KIND_LABELS}
    _SIDES = ["—", "R", "L"]

    def _machine_from_agent() -> SimMachine:
        """Máquina inicial a partir de los canales con que arrancó el app."""
        sens = []
        for c in agent.channels:
            if is_keyphasor_channel(c):
                k = "keyphasor"
            elif "mil" in (c.units or "").lower():
                k = "prox"
            elif "mm/s" in (c.units or "").lower():
                k = "vel"
            else:
                k = "accel"
            sens.append(SensorSpec(c.name, k, int(c.bnc_port or 1),
                                   float(c.sensitivity_mv_per_eu or 100.0)))
        cf = agent.source.config
        return SimMachine(name=args.machine, fs=float(args.fs), sensors=sens,
                          rpm=float(getattr(cf, "rpm", 3000.0)),
                          crit1=float(getattr(cf, "sim_critical_rpm", 0.0)),
                          crit2=float(getattr(cf, "sim_critical_rpm2", 0.0)),
                          phenomena=dict(getattr(cf, "defect_by_kind", {}) or {}),
                          severity=float(getattr(cf, "sim_severity", 1.0)))

    cfg_outer = QtWidgets.QWidget(); cfg_ol = QtWidgets.QVBoxLayout(cfg_outer)
    cfg_ol.setContentsMargins(4, 4, 4, 4); cfg_ol.setSpacing(8)
    # Config amigable estilo web: letra chica, casillas prolijas, pestañas internas.
    cfg_outer.setStyleSheet(
        "QWidget { font-size: 12px; }"
        "QLabel { color:#475569; font-weight:600; }"
        "QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {"
        "  background:#ffffff; border:1px solid #d6deea; border-radius:7px;"
        "  padding:4px 8px; min-height:24px; }"
        "QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {"
        "  border:1px solid #4f8fd0; }"
        "QTableWidget { border:1px solid #e6ebf2; border-radius:10px; gridline-color:#eef2f8; }"
        "QTableWidget::item { padding:3px 6px; }")

    _is_sim = type(agent.source).__name__ == "SimulatedStreamSource"

    def _dsp(mn, mx, val, step=100.0):
        s = QtWidgets.QDoubleSpinBox(); s.setRange(mn, mx); s.setValue(val); s.setSingleStep(step); return s

    def _wrap(layout):
        w = QtWidgets.QWidget(); layout.setContentsMargins(0, 0, 0, 0); w.setLayout(layout); return w

    cfg_tabs = QtWidgets.QTabWidget(); cfg_ol.addWidget(cfg_tabs, 1)

    # ---------- Pestaña 1: Machine ----------
    pg_machine = QtWidgets.QWidget(); ml = QtWidgets.QVBoxLayout(pg_machine); ml.setSpacing(9)

    def _subhdr(txt):
        h = QtWidgets.QLabel(txt)
        h.setStyleSheet("color:#94a3b8; font-weight:800; font-size:10px; letter-spacing:.08em;"
                        " text-transform:uppercase; margin-top:4px;")
        return h

    # Cargar una máquina YA configurada (local o de la web ☁) y editarla
    ml.addWidget(_subhdr("Load an existing machine (library + cloud ☁)"))
    r_lib = QtWidgets.QHBoxLayout()
    cb_lib = QtWidgets.QComboBox(); cb_lib.setMinimumWidth(280); r_lib.addWidget(cb_lib, 1)
    btn_load = QtWidgets.QPushButton("Load"); r_lib.addWidget(btn_load); r_lib.addStretch(1)
    ml.addLayout(r_lib)

    # Ficha del activo (homólogo a la web)
    ml.addWidget(_subhdr("Asset record"))
    fr = QtWidgets.QFormLayout(); fr.setHorizontalSpacing(14); fr.setVerticalSpacing(7)
    ed_name = QtWidgets.QLineEdit(); ed_name.setPlaceholderText("e.g. TG-1, Compressor K-101")
    ed_type = QtWidgets.QLineEdit(); ed_type.setPlaceholderText("e.g. Turbogenerator, Motor+Pump")
    ed_tag = QtWidgets.QLineEdit(); ed_tag.setPlaceholderText("asset tag / nameplate")
    ed_client = QtWidgets.QLineEdit(); ed_client.setPlaceholderText("client")
    ed_loc = QtWidgets.QLineEdit(); ed_loc.setPlaceholderText("plant / location")
    _r1 = QtWidgets.QHBoxLayout(); _r1.addWidget(ed_name, 1); _r1.addSpacing(10)
    _r1.addWidget(QtWidgets.QLabel("Type:")); _r1.addWidget(ed_type, 1)
    fr.addRow("Machine name:", _wrap(_r1))
    _r2 = QtWidgets.QHBoxLayout(); _r2.addWidget(ed_tag, 1); _r2.addSpacing(10)
    _r2.addWidget(QtWidgets.QLabel("Client:")); _r2.addWidget(ed_client, 1); _r2.addSpacing(10)
    _r2.addWidget(QtWidgets.QLabel("Location:")); _r2.addWidget(ed_loc, 1)
    fr.addRow("Tag:", _wrap(_r2))
    ml.addLayout(fr)

    # Operación + geometría (con Nº de cojinetes → layout recomendado)
    ml.addWidget(_subhdr("Operation & geometry"))
    r2 = QtWidgets.QHBoxLayout()
    r2.addWidget(QtWidgets.QLabel("Mode:"))
    cb_mode = QtWidgets.QComboBox(); cb_mode.addItems(_mode_labels); r2.addWidget(cb_mode)
    r2.addWidget(QtWidgets.QLabel("RPM:")); sp_rpm = _dsp(0, 60000, 3000); r2.addWidget(sp_rpm)
    r2.addWidget(QtWidgets.QLabel("Rotation:"))
    cb_rot = QtWidgets.QComboBox(); cb_rot.addItems(["CCW", "CW"]); r2.addWidget(cb_rot)
    r2.addWidget(QtWidgets.QLabel("Bearing type:"))
    cb_brg = QtWidgets.QComboBox(); cb_brg.addItems(["plain", "tilting_pad", "rolling", "mixed"])
    r2.addWidget(cb_brg); r2.addStretch(1); ml.addLayout(r2)
    r2b = QtWidgets.QHBoxLayout()
    r2b.addWidget(QtWidgets.QLabel("No. of bearings:"))
    sp_nbrg = QtWidgets.QSpinBox(); sp_nbrg.setRange(1, 12); sp_nbrg.setValue(4); r2b.addWidget(sp_nbrg)
    btn_autolay = QtWidgets.QPushButton("Auto-generate sensor layout")
    btn_autolay.setStyleSheet(
        "QPushButton{background:#eef5ff;color:#12467f;border:1px solid #bcd6f5;font-weight:700;"
        "padding:6px 14px;border-radius:7px;} QPushButton:hover{background:#dbe8f7;}")
    r2b.addWidget(btn_autolay)
    r2b.addWidget(QtWidgets.QLabel("<i style='color:#64748b'>→ fills a recommended X/Y layout you can edit</i>"))
    r2b.addStretch(1); ml.addLayout(r2b)
    ml.addStretch(1)
    cfg_tabs.addTab(pg_machine, "Machine")

    # ---------- Pestaña 2: Sensors & layout ----------
    pg_sensors = QtWidgets.QWidget(); sl = QtWidgets.QVBoxLayout(pg_sensors)
    leg = QtWidgets.QLabel(
        f"<span style='color:{SENSOR_COLORS['prox']}'>●</span> Proximity&nbsp;&nbsp;"
        f"<span style='color:{SENSOR_COLORS['vel']}'>●</span> Velocity&nbsp;&nbsp;"
        f"<span style='color:{SENSOR_COLORS['accel']}'>●</span> Accelerometer&nbsp;&nbsp;"
        f"<span style='color:{SENSOR_COLORS['keyphasor']}'>●</span> Keyphasor"
        f"&nbsp;&nbsp;&nbsp;<span style='color:#64748b'>API 670 angle: from TDC · "
        f"R=clockwise · L=counter-clockwise (45°L+45°R=90°)</span>")
    sl.addWidget(leg)
    canv = QtWidgets.QHBoxLayout()
    # 9 columnas visibles + 6 OCULTAS (ADRE 408): full_scale, active, coupling, unit,
    # keyphasor_ref, pair_ref. La tabla es la fuente única; el "Channel editor" edita todo.
    _COL_FS, _COL_ACT, _COL_COUP, _COL_UNIT, _COL_KPH, _COL_PAIR = 9, 10, 11, 12, 13, 14
    tblc = QtWidgets.QTableWidget(0, 15)
    tblc.setHorizontalHeaderLabels(["Channel", "Type", "BNC", "Sensit. (mV/EU)",
                                    "Angle°", "Side", "Gap (V)", "Alarm", "Danger",
                                    "FullScale", "Active", "Coupling", "Unit", "Keyphasor", "Pair"])
    for _c in (_COL_FS, _COL_ACT, _COL_COUP, _COL_UNIT, _COL_KPH, _COL_PAIR):
        tblc.setColumnHidden(_c, True)          # datos ADRE ocultos (se editan en Channel editor)
    for _c in range(9):
        tblc.horizontalHeader().setSectionResizeMode(_c, QtWidgets.QHeaderView.Stretch)
    tblc.verticalHeader().setVisible(False)
    tblc.setAlternatingRowColors(True)
    tblc.setShowGrid(False)
    tblc.verticalHeader().setDefaultSectionSize(30)
    tblc.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
    tblc.setMinimumHeight(240)
    canv.addWidget(tblc, 3)
    brg_plot = pg.PlotWidget(); brg_plot.setBackground("w"); brg_plot.setAspectLocked(True)
    brg_plot.hideAxis("left"); brg_plot.hideAxis("bottom"); brg_plot.setMenuEnabled(False)
    brg_plot.setMinimumWidth(340)
    brg_plot.setTitle("Machine layout", color=NAVY, size="10pt")
    _brg_vb = brg_plot.getViewBox()              # diagrama FIJO (sin pan/zoom/rueda)
    _brg_vb.setMouseEnabled(x=False, y=False); _brg_vb.setMenuEnabled(False)
    _brg_vb.wheelEvent = lambda ev, axis=None: None
    try:
        brg_plot.getPlotItem().hideButtons()
    except Exception:  # noqa: BLE001
        pass
    canv.addWidget(brg_plot, 2)
    sl.addLayout(canv, 1)
    r_sb = QtWidgets.QHBoxLayout()
    btn_add = QtWidgets.QPushButton("+ Sensor"); btn_del = QtWidgets.QPushButton("– Remove")
    for b in (btn_add, btn_del): r_sb.addWidget(b)
    r_sb.addStretch(1); sl.addLayout(r_sb)
    cfg_tabs.addTab(pg_sensors, "Sensors & layout")

    # ---------- Pestaña 3: Acquisition (avanzada, estilo System1/ADRE) ----------
    pg_acq = QtWidgets.QWidget(); al = QtWidgets.QVBoxLayout(pg_acq); al.setSpacing(9)
    al.addWidget(_subhdr("Train-wide"))
    ra = QtWidgets.QHBoxLayout()
    ra.addWidget(QtWidgets.QLabel("Sampling (Hz):"))
    sp_fs = QtWidgets.QSpinBox(); sp_fs.setRange(256, 102400); sp_fs.setSingleStep(1280); ra.addWidget(sp_fs)
    ra.addWidget(QtWidgets.QLabel("Frequency:"))
    cb_frequ = QtWidgets.QComboBox(); cb_frequ.addItems(["CPM", "Hz"]); ra.addWidget(cb_frequ)
    ra.addWidget(QtWidgets.QLabel("Waveform:"))
    cb_wfmode = QtWidgets.QComboBox(); cb_wfmode.addItems(["synchronous", "asynchronous"]); ra.addWidget(cb_wfmode)
    ra.addWidget(QtWidgets.QLabel("Samples/rev:"))
    sp_sprev = QtWidgets.QSpinBox(); sp_sprev.setRange(0, 4096); sp_sprev.setValue(0); ra.addWidget(sp_sprev)
    ra.addWidget(QtWidgets.QLabel("Averages:"))
    sp_avg = QtWidgets.QSpinBox(); sp_avg.setRange(1, 64); sp_avg.setValue(4); ra.addWidget(sp_avg)
    ra.addWidget(QtWidgets.QLabel("Window:"))
    cb_win = QtWidgets.QComboBox(); cb_win.addItems(["hanning", "flattop", "uniform"]); ra.addWidget(cb_win)
    ra.addStretch(1); al.addLayout(ra)
    r_ord = QtWidgets.QHBoxLayout()
    r_ord.addWidget(QtWidgets.QLabel("Orders (×rpm):"))
    o_half = QtWidgets.QCheckBox("½X"); o_1x = QtWidgets.QCheckBox("1X"); o_1x.setChecked(True)
    o_2x = QtWidgets.QCheckBox("2X"); o_2x.setChecked(True); o_3x = QtWidgets.QCheckBox("3X")
    for _c in (o_half, o_1x, o_2x, o_3x):
        r_ord.addWidget(_c)
    r_ord.addStretch(1); al.addLayout(r_ord)

    # Bandas por tipo de sensor (proximidad baja freq · acelerómetro alta freq)
    al.addWidget(_subhdr("Acquisition band per sensor type (ISO / System1)"))
    acq_band = {}
    _band_def = {"proximity": (1000, 2, "1600"), "velometer": (2000, 2, "1600"),
                 "accelerometer": (10000, 10, "3200")}
    for _t, _lab in [("proximity", "Proximity"), ("velometer", "Velocity"),
                     ("accelerometer", "Accelerometer")]:
        _fx, _fn, _ln = _band_def[_t]
        rb = QtWidgets.QHBoxLayout()
        _dot = SENSOR_COLORS.get({"proximity": "prox", "velometer": "vel",
                                  "accelerometer": "accel"}[_t], "#8b5cf6")
        _l = QtWidgets.QLabel(f"<span style='color:{_dot}'>●</span> {_lab}"); _l.setMinimumWidth(120)
        rb.addWidget(_l)
        rb.addWidget(QtWidgets.QLabel("Fmax (Hz):")); _wfx = _dsp(50, 40000, _fx, 100); rb.addWidget(_wfx)
        rb.addWidget(QtWidgets.QLabel("Fmin (Hz):")); _wfn = _dsp(0, 2000, _fn, 1); rb.addWidget(_wfn)
        rb.addWidget(QtWidgets.QLabel("Lines:"))
        _wln = QtWidgets.QComboBox(); _wln.addItems(["400", "800", "1600", "3200", "6400"])
        _wln.setCurrentText(_ln); rb.addWidget(_wln)
        rb.addStretch(1); al.addLayout(rb)
        acq_band[_t] = (_wfx, _wfn, _wln)

    lbl_acq_df = QtWidgets.QLabel(""); lbl_acq_df.setStyleSheet("color:#64748b;")
    al.addWidget(lbl_acq_df)
    al.addWidget(_subhdr("Configured channels — pairing (edit in Channel editor) + band per type"))
    tbl_acq = QtWidgets.QTableWidget(0, 6)
    tbl_acq.setHorizontalHeaderLabels(["Channel", "Type", "Pair (X/Y)", "Keyphasor",
                                       "Fmax (Hz)", "Lines"])
    tbl_acq.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
    tbl_acq.verticalHeader().setVisible(False)
    tbl_acq.setAlternatingRowColors(True); tbl_acq.setShowGrid(False)
    tbl_acq.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
    tbl_acq.verticalHeader().setDefaultSectionSize(26)
    al.addWidget(tbl_acq, 1)
    cfg_tabs.addTab(pg_acq, "Acquisition")

    def _band_ap(t):
        """AcquisitionParams del tipo t desde los widgets de banda + train-wide."""
        from core.remote_monitoring.config import AcquisitionParams
        fx, fn, ln = acq_band[t]
        orders = [v for v, c in ((0.5, o_half), (1.0, o_1x), (2.0, o_2x), (3.0, o_3x)) if c.isChecked()]
        return AcquisitionParams(
            fmax_hz=float(fx.value()), fmin_hz=float(fn.value()), lines=int(ln.currentText()),
            averages=int(sp_avg.value()), window=cb_win.currentText(),
            samples_per_rev=int(sp_sprev.value()), waveform_mode=cb_wfmode.currentText(),
            orders=orders or [1.0], freq_unit=("cpm" if cb_frequ.currentText() == "CPM" else "hz"))

    def _acq_cell(text, bold=False, fg=None):
        it = QtWidgets.QTableWidgetItem(text)
        it.setFlags(QtCore.Qt.ItemIsEnabled)
        if fg:
            it.setForeground(QtGui.QColor(fg))
        if bold:
            f = it.font(); f.setBold(True); it.setFont(f)
        return it

    def _refresh_acq_info():
        # Read-only: muestra los canales EN ORDEN con su pareo/keyphasor (que se EDITAN
        # en 'Channel editor') y la banda de adquisición por tipo de sensor (editable arriba).
        try:
            _pfx, _pfn, _pln = acq_band["proximity"]
            fx = float(_pfx.value()); ln = int(_pln.currentText())
            df = fx / ln if ln else 0.0
            orders = [s for s, c in (("½X", o_half), ("1X", o_1x), ("2X", o_2x), ("3X", o_3x))
                      if c.isChecked()]
            lbl_acq_df.setText(f"Proximity Δf = {df:.2f} Hz · {cb_wfmode.currentText()} · "
                               f"{sp_avg.value()} avg · orders {', '.join(orders) or '—'} · "
                               f"{cb_frequ.currentText()}")
            _WT = {"prox": "proximity", "vel": "velometer", "accel": "accelerometer",
                   "keyphasor": "keyphasor"}
            rows = []
            for r in range(tblc.rowCount()):
                it = tblc.item(r, 0)
                if not it:
                    continue
                w = tblc.cellWidget(r, 1)
                lbl = w.currentText() if w else "Accelerometer"
                rows.append((r, it.text(), _KIND_BY_LABEL.get(lbl, "accel")))
            names = {nm for _, nm, _ in rows}
            kph_name = next((nm for _, nm, k in rows if k == "keyphasor"), "")
            tbl_acq.setRowCount(len(rows))
            for i, (r, nm, kind) in enumerate(rows):
                tbl_acq.setItem(i, 0, _acq_cell(nm, bold=True,
                                fg=SENSOR_COLORS.get(kind, "#8b5cf6")))
                tbl_acq.setItem(i, 1, _acq_cell(_WT.get(kind, kind).capitalize()))
                if kind == "keyphasor":
                    for c in (2, 3, 4, 5):
                        tbl_acq.setItem(i, c, _acq_cell("—"))
                    continue
                pv = tblc.item(r, _COL_PAIR); pair = pv.text() if pv and pv.text() else ""
                if not pair:
                    sib = _sibling_name(nm); pair = sib if sib in names else "— (no pair)"
                kv = tblc.item(r, _COL_KPH); kphr = kv.text() if kv and kv.text() else (kph_name or "— (none)")
                _bfx, _bfn, _bln = acq_band.get(_WT.get(kind, "proximity"), acq_band["proximity"])
                tbl_acq.setItem(i, 2, _acq_cell(pair, fg=("#b45309" if pair.startswith("—") else None)))
                tbl_acq.setItem(i, 3, _acq_cell(kphr))
                tbl_acq.setItem(i, 4, _acq_cell(f"{_bfx.value():.0f}"))
                tbl_acq.setItem(i, 5, _acq_cell(_bln.currentText()))
        except Exception:  # noqa: BLE001
            pass
    for _bw in acq_band.values():
        _bw[0].valueChanged.connect(lambda *_: _refresh_acq_info())
        _bw[2].currentTextChanged.connect(lambda *_: _refresh_acq_info())
    for _cw in (o_half, o_1x, o_2x, o_3x):
        _cw.stateChanged.connect(lambda *_: _refresh_acq_info())
    for _w in (cb_wfmode, cb_frequ):
        _w.currentTextChanged.connect(lambda *_: _refresh_acq_info())
    sp_avg.valueChanged.connect(lambda *_: _refresh_acq_info())
    cfg_tabs.currentChanged.connect(lambda *_: _refresh_acq_info())

    # ---------- Pestaña: Channel editor (maestro-detalle, paridad ADRE 408) ----------
    pg_ched = QtWidgets.QWidget(); cl = QtWidgets.QVBoxLayout(pg_ched); cl.setSpacing(8)
    _ched_row = {"r": None, "busy": False}
    hrow = QtWidgets.QHBoxLayout()
    hrow.addWidget(QtWidgets.QLabel("Channel:"))
    cb_ched = QtWidgets.QComboBox(); cb_ched.setMinimumWidth(160); hrow.addWidget(cb_ched)
    btn_ch_prev = QtWidgets.QPushButton("◀"); btn_ch_next = QtWidgets.QPushButton("▶")
    for _b in (btn_ch_prev, btn_ch_next):
        _b.setFixedWidth(34); hrow.addWidget(_b)
    lbl_ch_bearing = QtWidgets.QLabel(""); lbl_ch_bearing.setStyleSheet("color:#64748b;")
    hrow.addWidget(lbl_ch_bearing); hrow.addStretch(1)
    cl.addLayout(hrow)

    def _mkspin(mn, mx, step=1.0, dec=2, val=0.0):
        s = QtWidgets.QDoubleSpinBox(); s.setDecimals(dec); s.setRange(mn, mx)
        s.setSingleStep(step); s.setValue(val); return s

    e_point = QtWidgets.QLineEdit()
    e_bnc = QtWidgets.QSpinBox(); e_bnc.setRange(1, 64)
    e_active = QtWidgets.QCheckBox("Active (collects data)"); e_active.setChecked(True)
    e_type = QtWidgets.QComboBox(); e_type.addItems([l for l, _ in _KIND_LABELS])
    e_sens = _mkspin(0, 100000, 10, 2, 200)
    e_unit = QtWidgets.QComboBox(); e_unit.addItems(["mil pp", "um pp", "mm/s rms", "in/s pk", "g rms", "pulses/rev"])
    e_coup = QtWidgets.QComboBox(); e_coup.addItems(["DC", "AC", "IEPE"])
    e_full = _mkspin(0, 100000, 1, 2, 0)
    e_gap = _mkspin(-30, 30, 0.1, 2, 0)
    e_ang = _mkspin(0, 360, 5, 1, 0)
    e_side = QtWidgets.QComboBox(); e_side.addItems(_SIDES)
    e_kph = QtWidgets.QComboBox(); e_pair = QtWidgets.QComboBox()
    e_alert = _mkspin(0, 100000, 0.1, 3, 0)
    e_danger = _mkspin(0, 100000, 0.1, 3, 0)

    def _grp(title, pairs):
        cl.addWidget(_subhdr(title))
        f = QtWidgets.QFormLayout(); f.setHorizontalSpacing(16); f.setVerticalSpacing(7)
        row = QtWidgets.QHBoxLayout()
        for i, (lab, w) in enumerate(pairs):
            if lab:
                row.addWidget(QtWidgets.QLabel(lab))
            row.addWidget(w, 1)
            if i < len(pairs) - 1:
                row.addSpacing(12)
        row.addStretch(0)
        cl.addLayout(row)

    _grp("Identification (API 670)", [("Point:", e_point), ("BNC:", e_bnc), ("", e_active)])
    _grp("Transducer", [("Type:", e_type), ("Sensitivity mV/EU:", e_sens),
                        ("Unit:", e_unit), ("Coupling:", e_coup)])
    _grp("", [("Full-scale (EU):", e_full), ("Gap/Bias (V):", e_gap)])
    _grp("Orientation (TDC top · R clockwise · L counter-clockwise)",
         [("Angle°:", e_ang), ("Side:", e_side)])
    _grp("Associations (phase reference + orbit pair)",
         [("Associated keyphasor:", e_kph), ("X/Y pair (orbit):", e_pair)])
    _grp("Alarms (API 670 / ISO 20816)", [("Alert:", e_alert), ("Danger:", e_danger)])

    btn_ch_apply = QtWidgets.QPushButton("✓ Apply to channel")
    btn_ch_apply.setStyleSheet(
        "QPushButton{background:#10b981;color:white;border:none;font-weight:800;"
        "padding:9px 18px;border-radius:8px;} QPushButton:hover{background:#0e9f6e;}")
    lbl_ch_ok = QtWidgets.QLabel(""); lbl_ch_ok.setStyleSheet("color:#166534;font-weight:700;")
    arow = QtWidgets.QHBoxLayout(); arow.addStretch(1)
    arow.addWidget(lbl_ch_ok); arow.addWidget(btn_ch_apply)
    cl.addLayout(arow); cl.addStretch(1)
    cfg_tabs.insertTab(2, pg_ched, "Channel editor")

    def _cell(r, c):
        it = tblc.item(r, c)
        return it.text() if it else ""

    def _all_names():
        return [_cell(r, 0) for r in range(tblc.rowCount()) if tblc.item(r, 0)]

    def _kph_names():
        out = []
        for r in range(tblc.rowCount()):
            w = tblc.cellWidget(r, 1)
            if w and _KIND_BY_LABEL.get(w.currentText()) == "keyphasor":
                out.append(_cell(r, 0))
        return out

    def _ched_load_row(r):
        if r is None or r < 0 or r >= tblc.rowCount():
            return
        _ched_row["busy"] = True
        _ched_row["r"] = r
        names = _all_names()
        e_point.setText(_cell(r, 0))
        w = tblc.cellWidget(r, 1)
        e_type.setCurrentText(w.currentText() if w else "Accelerometer")
        e_bnc.setValue(int(float(_cell(r, 2) or 1)))
        e_sens.setValue(float(_cell(r, 3) or 0))
        e_ang.setValue(float(_cell(r, 4) or 0))
        sw = tblc.cellWidget(r, 5)
        e_side.setCurrentText(sw.currentText() if sw else "—")
        e_gap.setValue(float(_cell(r, 6) or 0))
        e_alert.setValue(float(_cell(r, 7) or 0))
        e_danger.setValue(float(_cell(r, 8) or 0))
        e_full.setValue(float(_cell(r, _COL_FS) or 0))
        e_active.setChecked((_cell(r, _COL_ACT) or "1") != "0")
        e_coup.setCurrentText(_cell(r, _COL_COUP) or "IEPE")
        e_unit.setCurrentText(_cell(r, _COL_UNIT) or "g rms")
        e_kph.clear(); e_kph.addItems(["—"] + _kph_names())
        e_kph.setCurrentText(_cell(r, _COL_KPH) or "—")
        e_pair.clear(); e_pair.addItems(["—"] + [n for n in names if n != _cell(r, 0)])
        e_pair.setCurrentText(_cell(r, _COL_PAIR) or "—")
        lbl_ch_bearing.setText(f"Bearing {_bearing_no(_cell(r, 0)) or '—'}")
        lbl_ch_ok.setText("")
        _ched_row["busy"] = False

    def _ched_apply():
        r = _ched_row["r"]
        if r is None or r >= tblc.rowCount():
            return
        def _set(c, v):
            it = tblc.item(r, c)
            if it:
                it.setText(v)
            else:
                tblc.setItem(r, c, QtWidgets.QTableWidgetItem(v))
        _set(0, e_point.text() or f"CH{r}")
        w = tblc.cellWidget(r, 1)
        if w:
            w.setCurrentText(e_type.currentText())
        _set(2, str(e_bnc.value())); _set(3, f"{e_sens.value():g}"); _set(4, f"{e_ang.value():g}")
        sw = tblc.cellWidget(r, 5)
        if sw:
            sw.setCurrentText(e_side.currentText())
        _set(6, f"{e_gap.value():g}"); _set(7, f"{e_alert.value():g}"); _set(8, f"{e_danger.value():g}")
        _set(_COL_FS, f"{e_full.value():g}"); _set(_COL_ACT, "1" if e_active.isChecked() else "0")
        _set(_COL_COUP, e_coup.currentText()); _set(_COL_UNIT, e_unit.currentText())
        _set(_COL_KPH, "" if e_kph.currentText() == "—" else e_kph.currentText())
        _set(_COL_PAIR, "" if e_pair.currentText() == "—" else e_pair.currentText())
        _color_name_cell(r); draw_bearing()
        _ched_refresh_selector(); _refresh_acq_info()
        lbl_ch_ok.setText("✓ Applied")

    def _ched_refresh_selector():
        cur = cb_ched.currentText()
        cb_ched.blockSignals(True); cb_ched.clear(); cb_ched.addItems(_all_names())
        if cur in _all_names():
            cb_ched.setCurrentText(cur)
        cb_ched.blockSignals(False)

    cb_ched.currentIndexChanged.connect(lambda i: _ched_load_row(i))
    btn_ch_apply.clicked.connect(_ched_apply)
    btn_ch_prev.clicked.connect(lambda: cb_ched.setCurrentIndex(max(0, cb_ched.currentIndex() - 1)))
    btn_ch_next.clicked.connect(
        lambda: cb_ched.setCurrentIndex(min(cb_ched.count() - 1, cb_ched.currentIndex() + 1)))

    def _sensors_selected(r):
        if r is not None and r >= 0:
            cb_ched.setCurrentIndex(r)

    tblc.itemSelectionChanged.connect(lambda: _sensors_selected(tblc.currentRow()))

    # ---------- Pestaña: Validation (API 670 / ISO 20816) ----------
    pg_val = QtWidgets.QWidget(); vl = QtWidgets.QVBoxLayout(pg_val); vl.setSpacing(8)
    vtop = QtWidgets.QHBoxLayout()
    vtop.addWidget(QtWidgets.QLabel(
        "<b>Validation</b> <span style='color:#64748b'>— API 670 / ISO 20816 · runs on the "
        "current configuration</span>"))
    vtop.addStretch(1)
    btn_validate = QtWidgets.QPushButton("↻ Validate now")
    btn_validate.setStyleSheet(
        "QPushButton{background:#12467f;color:white;border:none;font-weight:700;"
        "padding:7px 15px;border-radius:7px;} QPushButton:hover{background:#0e3a6b;}")
    vtop.addWidget(btn_validate); vl.addLayout(vtop)
    val_summary = QtWidgets.QLabel(""); val_summary.setStyleSheet("font-size:14px;font-weight:700;")
    vl.addWidget(val_summary)
    val_out = QtWidgets.QTextEdit(); val_out.setReadOnly(True); vl.addWidget(val_out, 1)

    def _run_validation():
        try:
            from core.remote_monitoring.config import validate_setup
            setup = _form_to_setup()
            findings = validate_setup(setup)
        except Exception as e:  # noqa: BLE001
            val_summary.setText(""); val_out.setHtml(f"<i style='color:#b91c1c'>Error: {e}</i>")
            return
        ne = sum(1 for f in findings if f.level == "error")
        nw = sum(1 for f in findings if f.level == "warn")
        if ne:
            val_summary.setText(f"🔴 {ne} error(s) · 🟡 {nw} warning(s) — fix errors before measuring")
            val_summary.setStyleSheet("font-size:14px;font-weight:800;color:#b91c1c;")
        elif nw:
            val_summary.setText(f"🟡 {nw} warning(s) — review recommended")
            val_summary.setStyleSheet("font-size:14px;font-weight:800;color:#b45309;")
        else:
            val_summary.setText("🟢 Configuration valid — no findings")
            val_summary.setStyleSheet("font-size:14px;font-weight:800;color:#166534;")
        _ic = {"error": ("🔴", "#b91c1c"), "warn": ("🟡", "#b45309"), "ok": ("🟢", "#166534")}
        rows = []
        for f in findings:
            ic, col = _ic.get(f.level, ("•", "#334155"))
            rows.append(f"<div style='margin:4px 0'><span>{ic}</span> "
                        f"<span style='color:{col}'>{html.escape(f.message)}</span></div>")
        val_out.setHtml("<div style='font-family:Segoe UI,Arial;font-size:12.5px'>"
                        + "".join(rows) + "</div>")
    btn_validate.clicked.connect(_run_validation)
    cfg_tabs.insertTab(4, pg_val, "Validation")
    cfg_tabs.currentChanged.connect(
        lambda *_: _run_validation() if cfg_tabs.currentWidget() is pg_val else None)

    # ---------- Pestaña 4: Simulator (solo con simulador) ----------
    pg_sim = QtWidgets.QWidget(); sml = QtWidgets.QVBoxLayout(pg_sim); sml.setSpacing(10)
    rr = QtWidgets.QHBoxLayout()
    rr.addWidget(QtWidgets.QLabel("Startup/Coastdown range  Start→"))
    sp_r0 = _dsp(0, 60000, 300); rr.addWidget(sp_r0)
    sp_r1 = _dsp(0, 60000, 6000); rr.addWidget(sp_r1)
    rr.addWidget(QtWidgets.QLabel("Ramp(s):")); sp_ramp = _dsp(1, 3600, 90, 5); rr.addWidget(sp_ramp)
    rr.addStretch(1); sml.addLayout(rr)
    r3 = QtWidgets.QHBoxLayout()
    r3.addWidget(QtWidgets.QLabel("Critical 1:")); sp_c1 = _dsp(0, 60000, 0); r3.addWidget(sp_c1)
    r3.addWidget(QtWidgets.QLabel("Critical 2:")); sp_c2 = _dsp(0, 60000, 0); r3.addWidget(sp_c2)
    r3.addWidget(QtWidgets.QLabel("Severity:")); sp_sev = _dsp(0, 3, 1.0, 0.25); r3.addWidget(sp_sev)
    r3.addWidget(QtWidgets.QLabel("Prox:"))
    cb_ph_p = QtWidgets.QComboBox(); cb_ph_p.addItems(PHENOMENA["prox"]); r3.addWidget(cb_ph_p)
    r3.addWidget(QtWidgets.QLabel("Vel:"))
    cb_ph_v = QtWidgets.QComboBox(); cb_ph_v.addItems(PHENOMENA["vel"]); r3.addWidget(cb_ph_v)
    r3.addWidget(QtWidgets.QLabel("Accel:"))
    cb_ph_a = QtWidgets.QComboBox(); cb_ph_a.addItems(PHENOMENA["accel"]); r3.addWidget(cb_ph_a)
    r3.addStretch(1); sml.addLayout(r3)
    sml.addWidget(QtWidgets.QLabel(
        "<i style='color:#64748b'>Test-bench only: injects faults/criticals to validate the "
        "software without hardware. Not shown in field mode.</i>"))
    sml.addStretch(1)
    if _is_sim:
        cfg_tabs.addTab(pg_sim, "Simulator")

    # ---------- Botones de acción (siempre visibles, debajo de las pestañas) ----------
    rb = QtWidgets.QHBoxLayout()
    btn_save = QtWidgets.QPushButton("Save configuration")
    btn_apply = QtWidgets.QPushButton("Apply & measure")
    _redbtn = ("QPushButton{background:#f5484a;color:white;border:none;font-weight:700;"
               "padding:8px 16px;border-radius:7px;} QPushButton:hover{background:#d63c3e;}")
    btn_save.setStyleSheet(_redbtn); btn_apply.setStyleSheet(_redbtn)
    btn_cloud = QtWidgets.QPushButton("Save machine to Watermelon System")
    btn_cloud.setStyleSheet(
        "QPushButton{background:#10b981;color:white;border:none;font-weight:700;"
        "padding:8px 16px;border-radius:7px;} QPushButton:hover{background:#0e9f6e;}")
    rb.addStretch(1); rb.addWidget(btn_cloud); rb.addWidget(btn_save); rb.addWidget(btn_apply)
    cfg_ol.addLayout(rb)
    tabs.addTab(cfg_outer, "Setup")

    import math as _math

    def _bearing_no(nm):
        d = "".join(ch for ch in (nm or "") if ch.isdigit())
        return int(d) if d else 0

    def _sibling_name(nm):
        """Nombre del sensor PAR (mismo cojinete, eje opuesto): 1Y↔1X, 1V↔1H, 2YA↔2XA."""
        s = nm or ""
        swap = {"Y": "X", "X": "Y", "V": "H", "H": "V",
                "y": "x", "x": "y", "v": "h", "h": "v"}
        for i, ch in enumerate(s):
            if ch in swap:
                return s[:i] + swap[ch] + s[i + 1:]
        return ""

    def _disp_angle(s, idx):
        """Ángulo para DIBUJAR el sensor: si tiene ángulo configurado, ese; si no,
        convención por nombre (Y=45°L, X=45°R, V=arriba, H=der.) para que no se
        encimen con el default 0°."""
        if abs(float(getattr(s, "angle", 0.0) or 0.0)) > 1e-6:
            return s.abs_angle()
        nm = (s.name or "").upper()
        if "Y" in nm: return 315.0
        if "X" in nm: return 45.0
        if "V" in nm: return 0.0
        if "H" in nm: return 90.0
        return (idx * 40) % 360

    def draw_bearing():
        """Esquema HORIZONTAL de la máquina: eje + un anillo por cojinete + TODOS los
        sensores como bolitas de color en su ángulo, con etiqueta. Diagrama fijo.
        Círculos y letras grandes para que se lea bien; separación amplia entre
        cojinetes para que las etiquetas no se encimen."""
        try:
            brg_plot.clear()
            m = read_form()
            meas = [s for s in m.sensors if s.kind != "keyphasor"]
            kph = [s for s in m.sensors if s.kind == "keyphasor"]
            brs = sorted(set(_bearing_no(s.name) for s in meas)) or [1]
            S = 1.9                                   # separación entre cojinetes
            xpos = {b: i * S for i, b in enumerate(brs)}
            n = len(brs); xmax = (n - 1) * S
            th = np.linspace(0, 2 * np.pi, 90); R = 0.46
            brg_plot.plot([-0.95, xmax + 0.95], [0, 0], pen=pg.mkPen("#c9d6e8", width=9))  # eje
            for b in brs:
                cx = xpos[b]
                brg_plot.plot(cx + R * np.sin(th), R * np.cos(th), pen=pg.mkPen(NAVY, width=9))
                lb = pg.TextItem(html=f"<div style='font-size:10pt;color:#64748b;"
                                 f"font-weight:700'>Brg {b}</div>", anchor=(0.5, 0))
                lb.setPos(cx, -R - 0.22); brg_plot.addItem(lb)
                bs = [s for s in meas if _bearing_no(s.name) == b]
                for idx, s in enumerate(bs):
                    a = _math.radians(_disp_angle(s, idx))
                    dx, dy = _math.sin(a), _math.cos(a)
                    px, py = cx + (R + 0.02) * dx, (R + 0.02) * dy
                    col = SENSOR_COLORS.get(s.kind, "#8b5cf6")
                    brg_plot.addItem(pg.ScatterPlotItem([px], [py], symbol="o", size=22,
                                                        brush=col, pen=pg.mkPen("w", width=2)))
                    t = pg.TextItem(html=f"<div style='font-size:11pt;color:#0F1E3D;"
                                    f"font-weight:700'>{s.name.replace('_', '')}</div>",
                                    anchor=(0.5, 0.5))
                    t.setPos(cx + (R + 0.42) * dx, (R + 0.42) * dy); brg_plot.addItem(t)
            if kph:                                   # keyphasor a la izquierda del tren
                brg_plot.addItem(pg.ScatterPlotItem([-0.9], [0.0], symbol="t", size=20,
                                 brush=SENSOR_COLORS["keyphasor"], pen=pg.mkPen("w", width=1.5)))
                tk = pg.TextItem(html="<div style='font-size:10pt;color:#64748b;"
                                 "font-weight:700'>KPH</div>", anchor=(0.5, 1.2))
                tk.setPos(-0.9, 0.14); brg_plot.addItem(tk)
            rot = pg.TextItem(html=f"<div style='font-size:12pt;color:#0F1E3D;font-weight:800'>"
                              f"{'CW ↻' if m.rotation == 'CW' else 'CCW ↺'}</div>", anchor=(1, 1))
            rot.setPos(xmax + 0.95, 1.15); brg_plot.addItem(rot)
            # Auto-encuadre honrando aspect-lock → se ven TODOS los cojinetes (con 5-6
            # antes se cortaba y solo mostraba los del medio).
            brg_plot.getViewBox().setRange(xRange=(-1.2, xmax + 1.2), yRange=(-1.1, 1.25),
                                           padding=0, disableAutoRange=True)
            brg_plot.getViewBox().autoRange(padding=0.06)
        except Exception:  # noqa: BLE001
            pass

    def _color_name_cell(r):
        w = tblc.cellWidget(r, 1)
        kind = _KIND_BY_LABEL.get(w.currentText(), "accel") if w else "accel"
        it = tblc.item(r, 0)
        if it:
            it.setForeground(QtGui.QColor(SENSOR_COLORS.get(kind, "#8b5cf6")))
            f = it.font(); f.setBold(True); it.setFont(f)

    _KIND_UC = {"prox": ("mil pp", "DC"), "vel": ("mm/s rms", "AC"),
                "accel": ("g rms", "IEPE"), "keyphasor": ("pulses/rev", "DC")}

    def _add_sensor_row(s: SensorSpec):
        r = tblc.rowCount(); tblc.insertRow(r)
        tblc.setItem(r, 0, QtWidgets.QTableWidgetItem(s.name))
        cbk = QtWidgets.QComboBox(); cbk.addItems([l for l, _ in _KIND_LABELS])
        cbk.setCurrentText(_LABEL_BY_KIND.get(s.kind, "Accelerometer"))
        cbk.currentTextChanged.connect(lambda _t, rr=r: (_color_name_cell(rr), draw_bearing()))
        tblc.setCellWidget(r, 1, cbk)
        tblc.setItem(r, 2, QtWidgets.QTableWidgetItem(str(s.bnc)))
        tblc.setItem(r, 3, QtWidgets.QTableWidgetItem(f"{s.sensitivity:g}"))
        tblc.setItem(r, 4, QtWidgets.QTableWidgetItem(f"{s.angle:g}"))
        cbs = QtWidgets.QComboBox(); cbs.addItems(_SIDES)
        cbs.setCurrentText(s.side if s.side in ("R", "L") else "—")
        cbs.currentTextChanged.connect(lambda _t: draw_bearing())
        tblc.setCellWidget(r, 5, cbs)
        tblc.setItem(r, 6, QtWidgets.QTableWidgetItem(f"{s.gap:g}"))
        tblc.setItem(r, 7, QtWidgets.QTableWidgetItem(f"{s.alarm:g}"))
        tblc.setItem(r, 8, QtWidgets.QTableWidgetItem(f"{s.danger:g}"))
        # columnas ADRE ocultas (defaults por tipo). SensorSpec no las trae → derivadas.
        _u, _cp = _KIND_UC.get(s.kind, ("g rms", "IEPE"))
        tblc.setItem(r, _COL_FS, QtWidgets.QTableWidgetItem(f"{getattr(s, 'full_scale', 0.0):g}"))
        tblc.setItem(r, _COL_ACT, QtWidgets.QTableWidgetItem("1"))
        tblc.setItem(r, _COL_COUP, QtWidgets.QTableWidgetItem(_cp))
        tblc.setItem(r, _COL_UNIT, QtWidgets.QTableWidgetItem(_u))
        tblc.setItem(r, _COL_KPH, QtWidgets.QTableWidgetItem(getattr(s, "keyphasor_ref", "") or ""))
        tblc.setItem(r, _COL_PAIR, QtWidgets.QTableWidgetItem(getattr(s, "pair_ref", "") or ""))
        _color_name_cell(r)

    def fill_form(m: SimMachine):
        ed_name.setText(m.name); sp_fs.setValue(int(m.fs))
        cb_rot.setCurrentText(getattr(m, "rotation", "CCW"))
        cb_brg.setCurrentText(getattr(m, "bearing_type", "plain"))
        cb_mode.setCurrentText(MODE_LABELS.get(m.mode, MODE_LABELS["estable"]))
        sp_rpm.setValue(m.rpm); sp_r0.setValue(m.rpm_start); sp_r1.setValue(m.rpm_end)
        sp_ramp.setValue(m.ramp_s); sp_c1.setValue(m.crit1); sp_c2.setValue(m.crit2)
        sp_sev.setValue(m.severity)
        cb_ph_p.setCurrentText(m.phenomena.get("prox", "none"))
        cb_ph_v.setCurrentText(m.phenomena.get("vel", "none"))
        cb_ph_a.setCurrentText(m.phenomena.get("accel", "none"))
        tblc.setRowCount(0)
        for s in m.sensors: _add_sensor_row(s)
        draw_bearing()
        _ched_refresh_selector()
        if tblc.rowCount():
            _ched_load_row(0)

    def read_form() -> SimMachine:
        sens = []
        for r in range(tblc.rowCount()):
            nm = tblc.item(r, 0).text() if tblc.item(r, 0) else f"CH{r}"
            lbl = tblc.cellWidget(r, 1).currentText() if tblc.cellWidget(r, 1) else "Accelerometer"
            kd = _KIND_BY_LABEL.get(lbl, "accel")
            sd = tblc.cellWidget(r, 5).currentText() if tblc.cellWidget(r, 5) else "—"

            def _num(col, dv):
                try:
                    return float(tblc.item(r, col).text())
                except Exception:  # noqa: BLE001
                    return dv
            sens.append(SensorSpec(nm, kd, int(_num(2, r + 1)), _num(3, 100.0), _num(4, 0.0),
                                   side=("" if sd == "—" else sd),
                                   gap=_num(6, 0.0), alarm=_num(7, 0.0), danger=_num(8, 0.0)))
        ph = {"prox": cb_ph_p.currentText(), "vel": cb_ph_v.currentText(), "accel": cb_ph_a.currentText()}
        return SimMachine(name=ed_name.text() or "Machine", fs=float(sp_fs.value()), sensors=sens,
                          rotation=cb_rot.currentText(), bearing_type=cb_brg.currentText(),
                          mode=LABEL_TO_MODE.get(cb_mode.currentText(), "estable"), rpm=sp_rpm.value(),
                          rpm_start=sp_r0.value(), rpm_end=sp_r1.value(), ramp_s=sp_ramp.value(),
                          crit1=sp_c1.value(), crit2=sp_c2.value(), severity=sp_sev.value(),
                          phenomena={k: v for k, v in ph.items() if v != "none"})

    def _setup_to_sim(setup) -> SimMachine:
        """AcqSetup (nube/RM) → SimMachine (formato del nativo) para poder editar y
        medir en el campo una máquina configurada en la web."""
        KMAP = {"proximity": "prox", "velometer": "vel", "accelerometer": "accel",
                "keyphasor": "keyphasor"}
        sens = []
        for ch in setup.channels:
            sens.append(SensorSpec(
                ch.point_label, KMAP.get(ch.sensor_type, "accel"), int(ch.bnc_port or 1),
                float(ch.sensitivity_mv_per_eu or 100.0), float(ch.angle_deg or 0.0),
                side=(ch.side or ""), gap=float(getattr(ch, "gap_bias_v", 0.0) or 0.0),
                alarm=float(ch.alarm or 0.0), danger=float(ch.danger or 0.0)))
        mc = setup.machine
        mode = "arranque" if getattr(mc, "speed_control", "constant") == "variable" else "estable"
        return SimMachine(name=mc.name, fs=float(getattr(args, "fs", 5120.0)) or 5120.0,
                          sensors=sens, rotation=getattr(mc, "rotation", "CCW"),
                          bearing_type=getattr(mc, "bearing_type", "plain"), mode=mode,
                          rpm=float(getattr(mc, "rpm_nominal", 3000.0)),
                          rpm_start=float(getattr(mc, "rpm_min", 0.0) or 300.0),
                          rpm_end=float(getattr(mc, "rpm_max", 0.0) or 6000.0))

    def _cloud_machine_names():
        try:
            from core.remote_monitoring.config import list_setups_cloud
            return [(r.get("name") or r.get("id") or "") for r in list_setups_cloud()
                    if (r.get("name") or r.get("id"))]
        except Exception:  # noqa: BLE001
            return []

    def refresh_lib():
        local = list_machines() or []
        cloud_only = [n for n in _cloud_machine_names() if n not in local]
        items = local + [f"☁ {n}" for n in cloud_only]
        cb_lib.clear(); cb_lib.addItems(items or ["(empty)"])

    def do_load_lib():
        nm = cb_lib.currentText()
        if not nm or nm == "(empty)":
            return
        if nm.startswith("☁"):                     # máquina de la NUBE (rm_setups)
            name = nm[1:].strip()
            try:
                from core.remote_monitoring.config import load_setup_cloud
                setup = load_setup_cloud(name)
            except Exception:  # noqa: BLE001
                setup = None
            if setup is None:
                QtWidgets.QMessageBox.warning(win, "Load", f"Could not download '{name}' from the cloud.")
                return
            fill_form(_setup_to_sim(setup))
            mc = setup.machine
            ed_type.setText(getattr(mc, "machine_type", "") or "")
            ed_tag.setText(getattr(mc, "tag", "") or "")
            ed_client.setText(getattr(mc, "client", "") or "")
            ed_loc.setText(getattr(mc, "location", "") or "")
            if getattr(mc, "n_bearings", 0):
                sp_nbrg.setValue(int(mc.n_bearings))
            aq = setup.acquisition
            cb_win.setCurrentText(aq.window); sp_avg.setValue(int(aq.averages))
            sp_sprev.setValue(int(getattr(aq, "samples_per_rev", 0) or 0))
            cb_wfmode.setCurrentText(getattr(aq, "waveform_mode", "synchronous"))
            cb_frequ.setCurrentText("CPM" if getattr(aq, "freq_unit", "cpm") == "cpm" else "Hz")
            _ords = set(getattr(aq, "orders", []) or [])
            o_half.setChecked(0.5 in _ords); o_1x.setChecked(1.0 in _ords)
            o_2x.setChecked(2.0 in _ords); o_3x.setChecked(3.0 in _ords)
            for _t in ("proximity", "velometer", "accelerometer"):
                bp = (setup.acquisition_by_type or {}).get(_t) or aq
                fx, fn, ln = acq_band[_t]
                fx.setValue(float(bp.fmax_hz)); fn.setValue(float(bp.fmin_hz))
                ln.setCurrentText(str(int(bp.lines)))
        else:
            fill_form(load_from_library(nm))

    def do_save_lib():
        m = read_form()
        try:
            save_to_library(m); refresh_lib()
            QtWidgets.QMessageBox.information(win, "Library", f"Machine '{m.name}' saved.")
        except Exception as e:  # noqa: BLE001
            QtWidgets.QMessageBox.warning(win, "Library", f"Could not save: {e}")

    def do_apply():
        """Guarda la máquina y RELANZA el app midiéndola (evita reconstruir plots)."""
        import subprocess, tempfile, types
        m = read_form()
        path = os.path.join(tempfile.gettempdir(), "wm_apply_machine.json")
        m.save(path)
        # 1) Validar en ESTE proceso que la máquina se puede construir (si falla,
        #    mostramos el error acá y NO cerramos nada).
        try:
            _a = types.SimpleNamespace(machine_file=path, scenario="", defect="", sim=True,
                                       fs=m.fs, machine=m.name, chans="", names="", sens=100.0,
                                       rpm=m.rpm, prox=False, kph_bnc=0, chassis="cDAQ1")
            build_agent(_a)
        except Exception as e:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(
                win, "Apply & measure", f"The configuration has a problem and cannot be "
                f"measured:\n\n{type(e).__name__}: {e}")
            return
        # 2) Relanzar en una ventana que QUEDA ABIERTA si hay error (para verlo).
        try:
            if getattr(sys, "frozen", False):
                bat = os.path.join(tempfile.gettempdir(), "wm_run_machine.bat")
                with open(bat, "w") as f:
                    f.write(f'@echo off\r\n"{sys.executable}" --machine-file "{path}"\r\n'
                            f'if errorlevel 1 (echo. & echo *** ERROR on startup *** & pause)\r\n')
                os.startfile(bat)  # noqa: S606  (Windows)
            else:
                subprocess.Popen([sys.executable, os.path.abspath(__file__),
                                  "--machine-file", path])
            win.close()
        except Exception as e:  # noqa: BLE001
            QtWidgets.QMessageBox.warning(win, "Apply", f"Could not relaunch: {e}")

    def _form_to_setup():
        """Convierte la máquina del formulario (SimMachine) al formato ÚNICO de
        Watermelon System / Remote Monitoring (AcqSetup) — para guardarla/subirla
        como una máquina nueva que la web comparte."""
        from core.remote_monitoring.config import (AcqSetup, MachineConfig, ChannelRow,
                                                    AcquisitionParams)
        m = read_form()
        KMAP = {"prox": "proximity", "vel": "velometer", "accel": "accelerometer",
                "keyphasor": "keyphasor"}
        UMAP = {"prox": "mil pp", "vel": "mm/s rms", "accel": "g rms", "keyphasor": "pulses/rev"}
        CMAP = {"prox": "DC", "vel": "AC", "accel": "IEPE", "keyphasor": "DC"}
        chans = []
        for r, s in enumerate(m.sensors):        # r == fila de tblc (mismo orden)
            def _cx(col, dv=""):
                it = tblc.item(r, col); return it.text() if it else dv
            try:
                fs = float(_cx(_COL_FS) or 0)
            except Exception:  # noqa: BLE001
                fs = 0.0
            chans.append(ChannelRow(
                bnc_port=int(s.bnc), point_label=s.name, plane=(_bearing_no(s.name) or 1),
                sensor_type=KMAP.get(s.kind, "proximity"),
                sensitivity_mv_per_eu=float(s.sensitivity),
                unit_native=(_cx(_COL_UNIT) or UMAP.get(s.kind, "")),
                coupling=(_cx(_COL_COUP) or CMAP.get(s.kind, "AC")),
                angle_deg=float(s.angle), side=(s.side or ""),
                alarm=float(s.alarm), danger=float(s.danger), gap_bias_v=float(s.gap),
                full_scale=fs, active=(_cx(_COL_ACT, "1") != "0"),
                keyphasor_ref=_cx(_COL_KPH), pair_ref=_cx(_COL_PAIR), events_per_rev=1))
        # Completar pareo/keyphasor faltantes por inferencia (si el usuario no los definió
        # en el Channel editor): mismo cojinete, eje opuesto (1Y↔1X); keyphasor del tren.
        kph_name = next((c.point_label for c in chans if c.sensor_type == "keyphasor"), "")
        names = {c.point_label for c in chans}
        for c in chans:
            if c.sensor_type == "keyphasor":
                continue
            if not c.keyphasor_ref:
                c.keyphasor_ref = kph_name
            if not c.pair_ref:
                sib = _sibling_name(c.point_label)
                if sib and sib in names:
                    c.pair_ref = sib
        nb = len({_bearing_no(s.name) for s in m.sensors if s.kind != "keyphasor"}) or 1
        speed = "variable" if m.mode in ("arranque", "parada", "arranque_parada") else "constant"
        rng = [v for v in (m.rpm_start, m.rpm_end) if v > 0]
        mc = MachineConfig(name=m.name, rpm_nominal=float(m.rpm),
                           rpm_min=float(min(rng)) if rng else 0.0,
                           rpm_max=float(max(rng)) if rng else 0.0,
                           rotation=m.rotation, speed_control=speed,
                           bearing_type=m.bearing_type, n_bearings=int(sp_nbrg.value()) or nb,
                           machine_type=ed_type.text().strip(), tag=ed_tag.text().strip(),
                           client=ed_client.text().strip(), location=ed_loc.text().strip())
        try:
            by_type = {t: _band_ap(t) for t in ("proximity", "velometer", "accelerometer")}
            acq = _band_ap("proximity")     # global (fallback) = banda de proximidad
        except Exception:  # noqa: BLE001
            acq = AcquisitionParams(); by_type = None
        setup = AcqSetup(machine=mc, channels=chans, acquisition=acq)
        if by_type:
            setup.acquisition_by_type = by_type
        return setup

    def do_save_cloud_machine():
        try:
            setup = _form_to_setup()
        except Exception as e:  # noqa: BLE001
            _nice("Save machine", f"<b style='color:#b91c1c'>Config error</b><br>{e}",
                  QtWidgets.QMessageBox.Warning)
            return
        if not setup.channels:
            _nice("Save machine", "<b>Add at least one sensor</b> before saving the machine.")
            return
        btn_cloud.setEnabled(False); btn_cloud.setText("Uploading…")
        res = {}

        def _work():
            try:
                from core.remote_monitoring.config import save_setup_cloud
                res.update(save_setup_cloud(setup))
            except Exception as e:  # noqa: BLE001
                res.update({"ok": False, "reason": f"{type(e).__name__}: {e}"})
            res["done"] = True
        threading.Thread(target=_work, daemon=True).start()

        def _check():
            if not res.get("done"):
                QtCore.QTimer.singleShot(400, _check); return
            btn_cloud.setEnabled(True); btn_cloud.setText("Save machine to Watermelon System")
            if res.get("ok"):
                _nice("Machine saved",
                      "<div style='font-size:16px'><b style='color:#166534'>☁ Machine saved to "
                      "Watermelon System</b></div>"
                      f"<div style='color:#0F1E3D;margin-top:6px'><b>{res.get('name','')}</b> · "
                      f"{len(setup.channels)} channels</div>"
                      "<div style='color:#64748b;margin-top:8px'>Now available in the web "
                      "(Remote Monitoring). 🍉</div>")
            elif res.get("reason") == "offline":
                _nice("Save machine", "<b>No cloud connection.</b> Check internet and try again "
                      "(the machine is still in your local library).", QtWidgets.QMessageBox.Warning)
            else:
                _nice("Save machine", f"<b style='color:#b91c1c'>Could not save</b><br>"
                      f"<span style='color:#334155'>{res.get('reason','?')}</span>",
                      QtWidgets.QMessageBox.Warning)
        QtCore.QTimer.singleShot(300, _check)

    def _do_autolayout():
        # Nº de cojinetes → layout recomendado (KPH + X/Y por cojinete). El usuario edita.
        nb = int(sp_nbrg.value())
        fill_form(SimMachine.plantilla_prox_train(n_bearings=nb, name=(ed_name.text() or "Machine")))
        cfg_tabs.setCurrentWidget(pg_sensors)          # mostrar el resultado
        _refresh_acq_info()
        _nice("Layout ready",
              "<div style='font-size:15px'><b style='color:#166534'>✅ Recommended layout generated</b></div>"
              f"<div style='color:#0F1E3D;margin-top:6px'>{nb} bearings · keyphasor + X/Y per bearing "
              f"({nb * 2 + 1} channels).</div>"
              "<div style='color:#64748b;margin-top:8px'>Review/edit angles, alarms and gaps in "
              "<b>Sensors &amp; layout</b>, then continue. 🍉</div>")

    def _add_sensor_click():
        _add_sensor_row(SensorSpec("CHn", "accel", tblc.rowCount() + 1))
        draw_bearing(); _ched_refresh_selector()

    def _del_sensor_click():
        if tblc.currentRow() >= 0:
            tblc.removeRow(tblc.currentRow())
            draw_bearing(); _ched_refresh_selector()

    btn_add.clicked.connect(_add_sensor_click)
    btn_del.clicked.connect(_del_sensor_click)
    btn_autolay.clicked.connect(_do_autolayout)
    btn_load.clicked.connect(do_load_lib)
    btn_save.clicked.connect(do_save_lib)
    btn_cloud.clicked.connect(do_save_cloud_machine)
    btn_apply.clicked.connect(do_apply)
    refresh_lib(); fill_form(_machine_from_agent())
    tblc.itemChanged.connect(lambda *_: draw_bearing())   # redibuja al editar ángulo/nombre
    cb_rot.currentTextChanged.connect(lambda *_: draw_bearing())
    # Redibujar el diagrama al entrar a "Sensors & layout" → evita que quede viejo
    # (ej. cambiabas rotación a CW en Machine y el diagrama seguía mostrando CCW).
    cfg_tabs.currentChanged.connect(
        lambda *_: draw_bearing() if cfg_tabs.currentWidget() is pg_sensors else None)

    # --- Monitoreo (adquisición + estado + tabular rápido, estilo web) ---
    from core.remote_monitoring.stream_source import channel_kind as _ckind
    from core.remote_monitoring.recorder import (sync_pending, local_usage, free_bytes,
                                                 pending_count)
    ALARM_DEF = {"prox": (2.5, 4.0), "vel": (2.8, 4.5), "accel": (4.5, 7.1)}

    def _alarm_for(c):
        if args.alarm or args.danger:
            return args.alarm, args.danger
        return ALARM_DEF.get(_ckind(c), (0.0, 0.0))

    def _amp3(eu0, fs, f1, kind):
        """Overall + 1X + 2X en la convención de la NORMA del sensor:
        proximidad → pp (API 670/ISO 7919); velocidad/acel → RMS (ISO 20816).
        Amplitudes 1X/2X del pico del espectro (no colapsan); fase por proyección."""
        if kind == "prox":
            ov = float(eu0.max() - eu0.min()); k = 2.0            # 0-pk → pp
        else:
            ov = float(np.sqrt(np.mean(eu0 ** 2))); k = 1.0 / np.sqrt(2.0)  # 0-pk → rms
        fr, mag = _spectrum(eu0, fs)
        def _order(o):
            if not f1:
                return 0.0, 0.0
            ft = o * f1; b = (fr >= ft * 0.8) & (fr <= ft * 1.2)
            amp = float(mag[b].max()) * k if b.any() else 0.0
            _, ph = one_x_vector(eu0, fs, ft)
            return amp, ph
        a1, p1 = _order(1.0); a2, p2 = _order(2.0)
        return ov, a1, p1, a2, p2

    mon_w = QtWidgets.QWidget(); mon_l = QtWidgets.QVBoxLayout(mon_w)
    # tira de estado (RPM · 1X · Estado · Ventana · Samples · Vectores · Guardados · Tamaño)
    strip = QtWidgets.QFrame(); strip.setObjectName("statStrip")
    # selector con # → el borde queda SOLO en el panel, no en cada celda (bug anterior)
    strip.setStyleSheet(f"QFrame#statStrip {{ background:white; border:1px solid {LINE}; "
                        f"border-radius:10px; }}")
    sl = QtWidgets.QHBoxLayout(strip); sl.setContentsMargins(6, 8, 6, 8); sl.setSpacing(0)
    sv = {}

    def _statcell(key, label, first=False):
        w = QtWidgets.QWidget(); v = QtWidgets.QVBoxLayout(w)
        v.setContentsMargins(18, 2, 18, 2); v.setSpacing(2)
        if not first:                                   # divisor sutil entre métricas
            w.setStyleSheet("border-left:1px solid #eef2f7;")
        lab = QtWidgets.QLabel(label)
        lab.setStyleSheet("border:none; color:#8a97ab; font-size:10px; font-weight:700;"
                          " letter-spacing:.06em;")
        val = QtWidgets.QLabel("—")
        val.setStyleSheet(f"border:none; color:{NAVY}; font-size:18px; font-weight:800;"
                          " font-family:'Consolas','SF Mono',monospace;")
        v.addWidget(lab); v.addWidget(val); sv[key] = val
        sl.addWidget(w)
    _cells = [("rpm", "RPM"), ("x1", "1X"), ("estado", "STATUS"), ("srate", "SAMPLING"),
              ("vent", "WINDOW"), ("samp", "SAMPLES"), ("vect", "VECTORS"),
              ("guard", "SAVED"), ("size", "SIZE")]
    for _n, (_k, _l) in enumerate(_cells):
        _statcell(_k, _l, first=(_n == 0))
    sl.addStretch(1); mon_l.addWidget(strip)
    # (Start/Stop/Save/Upload/Delete live in the top toolbar)
    ctl = QtWidgets.QHBoxLayout()
    lbl_disk = QtWidgets.QLabel("Disk: —"); lbl_disk.setStyleSheet("color:#64748b;")
    ctl.addStretch(1); ctl.addWidget(lbl_disk)
    mon_l.addLayout(ctl)
    # tabular list — current values (fast)
    tblt = QtWidgets.QTableWidget(len(vib), 10)
    tblt.setHorizontalHeaderLabels(["Sensor", "Gap", "Overall", "1X", "1X phase",
                                    "2X", "2X phase", "Alarm", "Danger", "Status"])
    tblt.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
    tblt.verticalHeader().setVisible(False)
    tblt.setAlternatingRowColors(True)
    tblt.setShowGrid(False)
    tblt.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
    tblt.setFocusPolicy(QtCore.Qt.NoFocus)
    mon_l.addWidget(tblt, 1)
    mon_l.addWidget(QtWidgets.QLabel(
        "<i style='color:#64748b'>Amplitudes by standard: displacement in pp (API 670 · "
        "ISO 7919), velocity/acceleration in RMS (ISO 20816).</i>"))
    tabs.addTab(mon_w, "Monitoring")

    # --- Onda (ANÁLISIS: formas de onda + espectro) ---
    ond_w = QtWidgets.QWidget(); ond_l = QtWidgets.QVBoxLayout(ond_w)
    top = QtWidgets.QHBoxLayout()
    top.addWidget(QtWidgets.QLabel("<b>Waveforms</b>"))
    top.addWidget(QtWidgets.QLabel(
        "<i style='color:#64748b'>double-click a waveform = view it alone with its spectrum (FFT) · "
        "double-click again = back to all</i>"))
    top.addStretch(1); ond_l.addLayout(top)
    gl = pg.GraphicsLayoutWidget(); ond_l.addWidget(gl, 1)
    gl.ci.setContentsMargins(2, 2, 2, 2); gl.ci.setSpacing(2)
    wave_curves = []; wave_plots = []; wave_stats = []; wave_pills = []
    for r, (i, c) in enumerate(vib):
        p = gl.addPlot(row=r, col=0); p.showGrid(x=False, y=True, alpha=0.06)  # limpio, casi blanco
        col = SENSOR_COLORS.get(_ckind(c), CORN)
        p.setLabel("left", c.name, color=col)
        p.getAxis("bottom").setStyle(showValues=(r == len(vib) - 1))
        wave_curves.append(p.plot(pen=pg.mkPen(col, width=1.5)))
        pill = pg.TextItem(c.name, color="w", anchor=(0, 0), fill=pg.mkBrush(col))
        p.addItem(pill); wave_pills.append(pill)
        stt = pg.TextItem("", color=MUTE, anchor=(1, 0)); p.addItem(stt); wave_stats.append(stt)
        wave_plots.append(p)
    for _p in wave_plots[1:]:
        _p.setXLink(wave_plots[0])          # todas las ondas ALINEADAS en X (arrancan en 0)
    p_spec = gl.addPlot(row=len(vib), col=0); p_spec.showGrid(x=False, y=True, alpha=0.06)
    p_spec.setLabel("left", "amplitude"); p_spec.setLabel("bottom", "Frequency (CPM)")
    p_spec.setTitle("Spectrum (FFT)", color=NAVY, size="9pt")
    spec_curve = p_spec.plot(pen=pg.mkPen(AMBER, width=1.4))
    def _ordline(col):
        ln = pg.InfiniteLine(angle=90, pen=pg.mkPen(col, width=1, style=QtCore.Qt.DashLine))
        p_spec.addItem(ln); return ln
    v1x = _ordline(REDL); v2x = _ordline("#8b5cf6"); v3x = _ordline("#2fa36b")
    spec_info = pg.TextItem("", color=NAVY, anchor=(1, 0)); p_spec.addItem(spec_info)
    # Cursor del espectro: mové el mouse y muestra "valor unidad @ CPM"
    spec_data = {}
    spec_cursor = pg.InfiniteLine(angle=90, pen=pg.mkPen("#334155", width=1))
    spec_cur_txt = pg.TextItem("", color="#0F1E3D", anchor=(0, 1))
    p_spec.addItem(spec_cursor); p_spec.addItem(spec_cur_txt); spec_cursor.hide()

    def _spec_mouse(pos):
        try:
            cpm = spec_data.get("cpm"); mag = spec_data.get("mag")
            if cpm is None or not len(cpm) or not p_spec.sceneBoundingRect().contains(pos):
                spec_cursor.hide(); spec_cur_txt.setText(""); return
            vx = float(p_spec.vb.mapSceneToView(pos).x())
            j = int(np.abs(cpm - vx).argmin())
            spec_cursor.setPos(float(cpm[j])); spec_cursor.show()
            spec_cur_txt.setText(f"{mag[j]:.3g} {spec_data.get('unit','')} @ {cpm[j]:.0f} CPM")
            spec_cur_txt.setPos(float(cpm[j]), float(mag[j]))
        except Exception:  # noqa: BLE001
            pass
    p_spec.scene().sigMouseMoved.connect(_spec_mouse)
    tabs.addTab(ond_w, "Waveform")

    onda_focus = {"i": None}    # None = todas; idx = solo esa onda

    def _apply_onda_focus():
        fi = onda_focus["i"]
        for idx, p in enumerate(wave_plots):
            p.setVisible(fi is None or idx == fi)
        p_spec.setVisible(fi is not None)          # espectro SOLO al enfocar (doble clic)
    p_spec.setVisible(False)                        # arranca oculto (grilla = solo ondas)

    def _onda_dblclick(ev):
        try:
            if not ev.double():
                return
            pos = ev.scenePos()
            for idx, p in enumerate(wave_plots):
                if p.isVisible() and p.vb.sceneBoundingRect().contains(pos):
                    onda_focus["i"] = None if onda_focus["i"] == idx else idx
                    _apply_onda_focus(); return
            onda_focus["i"] = None; _apply_onda_focus()   # doble clic fuera → volver
        except Exception:  # noqa: BLE001
            pass
    gl.scene().sigMouseClicked.connect(_onda_dblclick)

    # --- Órbita (grilla de pares X/Y orientados por ángulo de sonda, estilo web) ---
    import math as _mo

    def _probe_angle(name):
        nm = (name or "").upper()
        if "Y" in nm: return 315.0     # 45°L
        if "X" in nm: return 45.0      # 45°R
        if "V" in nm: return 0.0       # vertical (TDC)
        if "H" in nm: return 90.0      # horizontal (lado R)
        return 0.0
    _byb = {}
    for i, c in vib:
        nm = c.name.upper()
        dg = "".join(ch for ch in nm if ch.isdigit()); brg = int(dg) if dg else 0
        ax = "Y" if ("Y" in nm or "V" in nm) else ("X" if ("X" in nm or "H" in nm) else "?")
        if ax != "?":
            _byb.setdefault(brg, {})[ax] = (i, c)
    orb_pairs = [(brg, _byb[brg]["Y"], _byb[brg]["X"])
                 for brg in sorted(_byb) if "Y" in _byb[brg] and "X" in _byb[brg]]
    orb_ok = len(orb_pairs) >= 1
    orb_items = []
    orb_focus = {"k": None}
    if orb_ok:
        orb_w = QtWidgets.QWidget(); orb_l = QtWidgets.QVBoxLayout(orb_w)
        orb_l.addWidget(QtWidgets.QLabel(
            "<b>Orbits</b> <i style='color:#64748b'>— oriented to the probe angle "
            "(Y=45°L · X=45°R) · double-click = view one · double-click again = back</i>"))
        gl_orb = pg.GraphicsLayoutWidget(); orb_l.addWidget(gl_orb, 1)
        gl_orb.ci.setContentsMargins(2, 2, 2, 2); gl_orb.ci.setSpacing(2)
        ncol = 2 if len(orb_pairs) > 1 else 1
        for k, (brg, (yi, yc), (xi, xc)) in enumerate(orb_pairs):
            p = gl_orb.addPlot(row=k // ncol, col=k % ncol)
            p.setAspectLocked(True); p.showGrid(x=True, y=True, alpha=0.12)
            p.hideAxis("left"); p.hideAxis("bottom"); p.setMenuEnabled(False)
            p.addLine(x=0, pen=pg.mkPen("#e6ecf5")); p.addLine(y=0, pen=pg.mkPen("#e6ecf5"))
            it = dict(
                p=p, yi=yi, xi=xi, yc=yc, xc=xc, brg=brg,
                aY=_mo.radians(_probe_angle(yc.name)), aX=_mo.radians(_probe_angle(xc.name)),
                curve=p.plot(pen=pg.mkPen(CORN, width=1.7)),
                smax=p.plot(pen=pg.mkPen(REDL, width=1.1, style=QtCore.Qt.DashLine)),
                smax_txt=pg.TextItem("", color=REDL, anchor=(0.5, 1.2)),
                kph=p.plot(pen=None, symbol="o", symbolBrush=KPH, symbolSize=7, symbolPen=pg.mkPen("w", width=1)),
                kph1=p.plot(pen=None, symbol="o", symbolBrush=REDL, symbolSize=12, symbolPen=pg.mkPen("w", width=2)),
                pill=pg.TextItem(f"Bearing {brg}", color="w", anchor=(0, 0), fill=pg.mkBrush(NAVY)))
            p.addItem(it["smax_txt"]); p.addItem(it["pill"])
            orb_items.append(it)
        tabs.addTab(orb_w, "Orbit")

        def _apply_orb_focus():
            fk = orb_focus["k"]
            for idx, it in enumerate(orb_items):
                it["p"].setVisible(fk is None or idx == fk)

        def _orb_dblclick(ev):
            try:
                if not ev.double():
                    return
                pos = ev.scenePos()
                for idx, it in enumerate(orb_items):
                    if it["p"].isVisible() and it["p"].vb.sceneBoundingRect().contains(pos):
                        orb_focus["k"] = None if orb_focus["k"] == idx else idx
                        _apply_orb_focus(); return
                orb_focus["k"] = None; _apply_orb_focus()
            except Exception:  # noqa: BLE001
                pass
        gl_orb.scene().sigMouseClicked.connect(_orb_dblclick)

    def _grid_focus(cells, focus, gl):
        """Doble clic para enfocar una celda / volver a la grilla (patrón común)."""
        def _apply():
            fk = focus["k"]
            for k, cl in enumerate(cells):
                cl["p"].setVisible(fk is None or k == fk)

        def _dbl(ev):
            try:
                if not ev.double():
                    return
                pos = ev.scenePos()
                for k, cl in enumerate(cells):
                    if cl["p"].isVisible() and cl["p"].vb.sceneBoundingRect().contains(pos):
                        focus["k"] = None if focus["k"] == k else k
                        _apply(); return
                focus["k"] = None; _apply()
            except Exception:  # noqa: BLE001
                pass
        gl.scene().sigMouseClicked.connect(_dbl)
        return _apply

    # --- Bode (grilla por canal: amplitud 1X vs rpm; doble clic = uno + cursor) ---
    bode_w = QtWidgets.QWidget(); bode_l = QtWidgets.QVBoxLayout(bode_w)
    bode_l.addWidget(QtWidgets.QLabel(
        "<b>Bode</b> <i style='color:#64748b'>— 1X amplitude vs rpm per channel (fills during run-up) · "
        "double-click = view one · move the mouse = amplitude/phase/rpm</i>"))
    gl_bode = pg.GraphicsLayoutWidget(); bode_l.addWidget(gl_bode, 1)
    gl_bode.ci.setContentsMargins(2, 2, 2, 2); gl_bode.ci.setSpacing(2)
    bode_cells = []; bode_focus = {"k": None}; _ncb = 2 if len(vib) > 1 else 1
    for k, (i, c) in enumerate(vib):
        p = gl_bode.addPlot(row=k // _ncb, col=k % _ncb)
        p.showGrid(x=True, y=True, alpha=0.12); p.setLabel("bottom", "RPM"); p.setLabel("left", f"1X {c.units}")
        col = SENSOR_COLORS.get(_ckind(c), CORN)
        pill = pg.TextItem(c.name, color="w", anchor=(0, 0), fill=pg.mkBrush(col)); p.addItem(pill)
        cur = pg.InfiniteLine(angle=90, pen=pg.mkPen("#334155", width=1)); p.addItem(cur); cur.hide()
        txt = pg.TextItem("", color="#0F1E3D", anchor=(0, 1)); p.addItem(txt)
        bode_cells.append(dict(
            p=p, name=c.name, unit=c.units, cur=cur, txt=txt, data={},
            amp=p.plot(pen=pg.mkPen(col, width=1.6), symbol="o", symbolSize=3, symbolBrush=col, symbolPen=None),
            pill=pill))
    tabs.addTab(bode_w, "Bode")
    _grid_focus(bode_cells, bode_focus, gl_bode)

    def _bode_mouse(pos):
        try:
            fk = bode_focus["k"]
            if fk is None:
                return
            cl = bode_cells[fk]; rr = cl["data"].get("rr")
            if rr is None or not len(rr) or not cl["p"].vb.sceneBoundingRect().contains(pos):
                cl["cur"].hide(); cl["txt"].setText(""); return
            vx = float(cl["p"].vb.mapSceneToView(pos).x())
            j = int(np.abs(rr - vx).argmin())
            cl["cur"].setPos(float(rr[j])); cl["cur"].show()
            cl["txt"].setText(f"1X {cl['data']['am'][j]:.2f} {cl['unit']} @ "
                              f"{cl['data']['ph'][j]:.0f}° @ {rr[j]:.0f} rpm")
            cl["txt"].setPos(float(rr[j]), float(cl['data']['am'][j]))
        except Exception:  # noqa: BLE001
            pass
    gl_bode.scene().sigMouseMoved.connect(_bode_mouse)

    # --- Polar (0° = ángulo de la sonda · fase contra el giro; grilla + doble clic) ---
    pol_w = QtWidgets.QWidget(); pol_l = QtWidgets.QVBoxLayout(pol_w)
    ptop = QtWidgets.QHBoxLayout()
    ptop.addWidget(QtWidgets.QLabel("<b>Polar</b>"))
    ptop.addWidget(QtWidgets.QLabel("Rotation:"))
    cb_giro = QtWidgets.QComboBox(); cb_giro.addItems(["CCW", "CW"]); ptop.addWidget(cb_giro)
    ptop.addWidget(QtWidgets.QLabel(
        "<i style='color:#64748b'>0° = probe angle · phase AGAINST rotation · "
        "double-click = full polar (rpm, Ncrit, data)</i>"))
    ptop.addStretch(1); pol_l.addLayout(ptop)
    gl_pol = pg.GraphicsLayoutWidget(); pol_l.addWidget(gl_pol, 1)
    gl_pol.ci.setContentsMargins(2, 2, 2, 2); gl_pol.ci.setSpacing(2)
    pol_cells = []; pol_focus = {"k": None}
    _ncp = 4 if len(vib) >= 5 else (2 if len(vib) > 1 else 1)   # cuadrados: más columnas = más grandes
    _pth = np.linspace(0, 2 * np.pi, 120)
    for k, (i, c) in enumerate(vib):
        p = gl_pol.addPlot(row=k // _ncp, col=k % _ncp)
        p.setAspectLocked(True); p.hideAxis("left"); p.hideAxis("bottom"); p.setMenuEnabled(False)
        for _r in (0.25, 0.5, 0.75, 1.0):                       # anillos de amplitud
            p.plot(_r * np.sin(_pth), _r * np.cos(_pth),
                   pen=pg.mkPen("#d6deea", width=1, style=QtCore.Qt.DashLine))
        pa = _probe_angle(c.name)                               # 0° del polar = ángulo de la sonda
        deg_items = []
        for d in range(0, 360, 30):                             # radios a ángulos FÍSICOS
            a = _mo.radians(pa + d)
            p.plot([0, _mo.sin(a)], [0, _mo.cos(a)], pen=pg.mkPen("#eef2f8", width=1))
            t = pg.TextItem("", color=MUTE, anchor=(0.5, 0.5))
            t.setPos(1.13 * _mo.sin(a), 1.13 * _mo.cos(a)); p.addItem(t)
            deg_items.append((t, d))
        p.setXRange(-1.35, 1.35); p.setYRange(-1.35, 1.35)
        col = SENSOR_COLORS.get(_ckind(c), CORN)
        pill = pg.TextItem(c.name, color="w", anchor=(0, 0), fill=pg.mkBrush(col))
        p.addItem(pill); pill.setPos(-1.3, 1.3)
        curve = p.plot(pen=pg.mkPen(col, width=1.6))
        pts = p.plot(pen=None, symbol="o", symbolSize=4, symbolBrush=KPH, symbolPen=None)
        ncrit = pg.ScatterPlotItem(size=16, symbol="star", brush=pg.mkBrush(REDL), pen=pg.mkPen("w", width=1))
        op = pg.ScatterPlotItem(size=12, brush=pg.mkBrush("#16a34a"), pen=pg.mkPen("w", width=1))
        p.addItem(ncrit); p.addItem(op)
        ncrit_txt = pg.TextItem("", color=REDL, anchor=(0, 0.5)); p.addItem(ncrit_txt)
        box = pg.TextItem("", color=NAVY, anchor=(0, 1)); p.addItem(box); box.setPos(-1.3, -1.1)
        rpm_lbls = [pg.TextItem("", color="#64748b", anchor=(0.5, 0.5)) for _ in range(8)]
        for t in rpm_lbls:
            p.addItem(t)
        arc = p.plot(pen=pg.mkPen("#5b6b86", width=2.0))     # arco de giro (visible pero prolijo)
        arw = pg.ArrowItem(angle=0, tipAngle=30, headLen=10, brush="#5b6b86", pen=None); p.addItem(arw)
        pol_cells.append(dict(p=p, name=c.name, unit=c.units, pill=pill, curve=curve, pts=pts,
                              ncrit=ncrit, op=op, ncrit_txt=ncrit_txt, box=box, rpm_lbls=rpm_lbls,
                              pa=pa, deg_items=deg_items, arc=arc, arw=arw))
    tabs.addTab(pol_w, "Polar")
    _grid_focus(pol_cells, pol_focus, gl_pol)

    def _polar_relabel():
        cw = (cb_giro.currentText() == "CW")     # CW → ángulos de fase al revés
        R = 1.18
        # arco de giro cerca del tope: CW = horario, CCW = antihorario
        degs = np.linspace(-32, 32, 16) if cw else np.linspace(32, -32, 16)
        xs = R * np.sin(np.radians(degs)); ys = R * np.cos(np.radians(degs))
        for cl in pol_cells:
            for t, d in cl["deg_items"]:
                t.setText(f"{(360 - d) % 360 if cw else d}°")
            cl["arc"].setData(xs, ys)
            cl["arw"].setPos(float(xs[-1]), float(ys[-1]))
            dx = xs[-1] - xs[-2]; dy = ys[-1] - ys[-2]
            cl["arw"].setStyle(angle=_mo.degrees(_mo.atan2(dy, -dx)))   # punta en el sentido de avance
    _polar_relabel()
    cb_giro.currentTextChanged.connect(lambda *_: _polar_relabel())

    # --- Cascada (grilla por canal: espectros apilados; doble clic = uno solo) ---
    casc_w = QtWidgets.QWidget(); casc_l = QtWidgets.QVBoxLayout(casc_w)
    casc_l.addWidget(QtWidgets.QLabel(
        "<b>Cascade</b> <i style='color:#64748b'>— spectra stacked by rpm per channel (run-up) · "
        "double-click = view one</i>"))
    gl_casc = pg.GraphicsLayoutWidget(); casc_l.addWidget(gl_casc, 1)
    gl_casc.ci.setContentsMargins(2, 2, 2, 2); gl_casc.ci.setSpacing(2)
    casc_cells = []; casc_focus = {"k": None}; _NCC = 26
    for k, (i, c) in enumerate(vib):
        p = gl_casc.addPlot(row=k // _ncb, col=k % _ncb)
        p.showGrid(x=True, y=True, alpha=0.15); p.setLabel("bottom", "Frequency (Hz)"); p.setLabel("left", "RPM")
        col = SENSOR_COLORS.get(_ckind(c), CORN)
        pill = pg.TextItem(c.name, color="w", anchor=(0, 0), fill=pg.mkBrush(col)); p.addItem(pill)
        casc_cells.append(dict(p=p, name=c.name, pill=pill,
                               curves=[p.plot(pen=pg.mkPen(col, width=0.8)) for _ in range(_NCC)]))
    tabs.addTab(casc_w, "Cascade")
    _grid_focus(casc_cells, casc_focus, gl_casc)

    # --- Shaft Centerline (posición del muñón en el cojinete vs rpm) — estilo web ---
    scl_items = []
    scl_track = {}          # brg -> {rpm_bucket: (x, y)}
    try:
        _scl_cmap = pg.colormap.get("turbo")
    except Exception:  # noqa: BLE001
        _scl_cmap = None
    if orb_ok:
        scl_w = QtWidgets.QWidget(); scl_l = QtWidgets.QVBoxLayout(scl_w)
        scl_l.addWidget(QtWidgets.QLabel(
            "<b>Shaft Centerline</b> <i style='color:#64748b'>— journal position within the bearing "
            "clearance as speed changes. Track colored by rpm · REST = at rest (bottom) · "
            "large dot = current.</i>"))
        gl_scl = pg.GraphicsLayoutWidget(); scl_l.addWidget(gl_scl, 1)
        gl_scl.ci.setContentsMargins(2, 2, 2, 2); gl_scl.ci.setSpacing(2)
        ncol = 2 if len(orb_pairs) > 1 else 1
        _th = np.linspace(0, 2 * np.pi, 160); Cclr = 8.0     # juego dibujado (mil)
        for k, (brg, (yi, yc), (xi, xc)) in enumerate(orb_pairs):
            p = gl_scl.addPlot(row=k // ncol, col=k % ncol)
            p.setAspectLocked(True); p.showGrid(x=True, y=True, alpha=0.06)
            p.hideAxis("left"); p.hideAxis("bottom"); p.setMenuEnabled(False)
            p.plot(Cclr * np.sin(_th), Cclr * np.cos(_th),
                   pen=pg.mkPen("#94a3b8", width=1.5, style=QtCore.Qt.DashLine))     # juego
            p.addLine(x=0, pen=pg.mkPen("#eef2f8")); p.addLine(y=0, pen=pg.mkPen("#eef2f8"))
            p.addItem(pg.ScatterPlotItem([0], [0], symbol="+", size=14, pen=pg.mkPen(NAVY, width=2)))  # centro
            rest = pg.TextItem("REST", color=REDL, anchor=(0.5, 0)); rest.setPos(0, -Cclr * 0.99)
            p.addItem(rest)
            line = p.plot(pen=pg.mkPen("#8aa0bd", width=1))
            trk = pg.ScatterPlotItem(size=7, pen=None); p.addItem(trk)
            cur = pg.ScatterPlotItem(size=14, brush=pg.mkBrush(REDL), pen=pg.mkPen("w", width=2)); p.addItem(cur)
            pill = pg.TextItem(f"Bearing {brg}", color="w", anchor=(0, 0), fill=pg.mkBrush(NAVY))
            p.addItem(pill); pill.setPos(-Cclr * 1.04, Cclr * 1.06)
            p.setXRange(-Cclr * 1.12, Cclr * 1.12); p.setYRange(-Cclr * 1.12, Cclr * 1.12)
            scl_items.append(dict(
                p=p, yi=yi, xi=xi, yc=yc, xc=xc, brg=brg,
                aY=_mo.radians(_probe_angle(yc.name)), aX=_mo.radians(_probe_angle(xc.name)),
                line=line, trk=trk, cur=cur))
            scl_track[brg] = {}
        tabs.addTab(scl_w, "Shaft Centerline")

    # --- Diagnostics (whirl/whip + criticals) ---
    diag_w = QtWidgets.QWidget(); diag_l = QtWidgets.QVBoxLayout(diag_w)
    drow = QtWidgets.QHBoxLayout()
    btn_diag = QtWidgets.QPushButton("Generate preliminary report")
    btn_diag_pdf = QtWidgets.QPushButton("Save PDF…")
    btn_diag_save = QtWidgets.QPushButton("Save HTML…")
    btn_diag_cloud = QtWidgets.QPushButton("Upload report to cloud")
    _greenbtn = ("QPushButton{background:#10b981;color:white;border:none;font-weight:700;"
                 "padding:7px 15px;border-radius:7px;} QPushButton:hover{background:#0e9f6e;}")
    btn_diag_pdf.setStyleSheet(_greenbtn)
    drow.addWidget(btn_diag); drow.addWidget(btn_diag_pdf)
    drow.addWidget(btn_diag_save); drow.addWidget(btn_diag_cloud)
    drow.addSpacing(16)
    drow.addWidget(QtWidgets.QLabel("Report language:"))
    cb_lang = QtWidgets.QComboBox(); cb_lang.addItems(["English", "Español"]); drow.addWidget(cb_lang)
    drow.addStretch(1)
    diag_l.addLayout(drow)
    diag_txt = QtWidgets.QTextEdit(); diag_txt.setReadOnly(True)
    diag_l.addWidget(diag_txt, 1)
    diag_state = {"html": ""}
    tabs.addTab(diag_w, "Diagnostics")

    lbl_rpm = QtWidgets.QLabel("RPM: —"); lbl_state = QtWidgets.QLabel("stopped")
    lbl_rec = QtWidgets.QLabel("")
    win.statusBar().addWidget(lbl_state)
    win.statusBar().addPermanentWidget(lbl_rec)
    win.statusBar().addPermanentWidget(lbl_rpm)

    # ---------------- Lógica ----------------
    from core.remote_monitoring.keyphasor import one_x_vector

    def update():
        # indicador en vivo de captura (se graba desde Iniciar) — así se ve que NO se pierde data
        _sess = rec_state.get("session")
        if _sess is not None and getattr(_sess, "open", False):
            lbl_rec.setText(f"● capturing · {_sess.status.duration_s:.0f}s · {_sess.status.size_mb:.1f} MB")
        snap = agent.snapshot(2.0)
        if snap.shape[1] < 16:
            return
        fs = agent.sample_rate_hz
        rpm = agent.estimate_rpm(snap)
        if not rpm:      # sin keyphasor → rpm del pico dominante de vibración (1X)
            fr0, mag0 = _spectrum(snap[vib[0][0]], fs)
            band = (fr0 > 3) & (fr0 < 300)
            if band.any() and float(mag0[band].max()) > 1e-9:
                rpm = float(fr0[band][np.argmax(mag0[band])] * 60.0)
        lbl_rpm.setText(f"RPM: {rpm:.0f}" if rpm else "RPM: —")
        f1 = (rpm / 60.0) if rpm else None
        # Alimentar el capturador de transitorio para Bode/Polar/Cascada. Con el sim
        # ya a tiempo real, cada ~4 refrescos (~0.36 s) da muchos puntos en el arranque.
        rec_state["fn"] = rec_state.get("fn", 0) + 1
        if rpm and rec_state["fn"] % 4 == 0:
            try:
                tc.feed(snap, rpm, fs, vib, kph_idx=kph_glob)
            except Exception:  # noqa: BLE001
                pass
            # Shaft Centerline: acumular la posición del muñón por rpm SIEMPRE (no solo
            # con la pestaña abierta) para que la traza se arme durante el arranque.
            if scl_items:
                bk = int(rpm // 100) * 100
                for it in scl_items:
                    mX = float((snap[it["xi"]] * 1000.0 / (it["xc"].sensitivity_mv_per_eu or 1.0)).mean())
                    mY = float((snap[it["yi"]] * 1000.0 / (it["yc"].sensitivity_mv_per_eu or 1.0)).mean())
                    cx = mX * _mo.sin(it["aX"]) + mY * _mo.sin(it["aY"])
                    cy = mX * _mo.cos(it["aX"]) + mY * _mo.cos(it["aY"])
                    scl_track[it["brg"]][bk] = (cx, cy)
        cur = tabs.tabText(tabs.currentIndex())
        if cur == "Monitoring":
            # global state (steady/run-up/coast-down) from rpm variation
            prev = rec_state.get("prev_rpm")
            if rpm and prev:
                d = rpm - prev
                estado_g = "Startup" if d > 15 else ("Coastdown" if d < -15 else "Steady")
            else:
                estado_g = "—"
            rec_state["prev_rpm"] = rpm
            # tira de estado
            try:
                block = agent.source.config.block_samples
                total = int(getattr(agent, "blocks_read", 0)) * block
            except Exception:  # noqa: BLE001
                total = snap.shape[1]
            try:
                vect = len(tc.bode(vib[0][1].name)[0])
            except Exception:  # noqa: BLE001
                vect = 0
            sv["rpm"].setText(f"{rpm:.0f}" if rpm else "—")
            sv["x1"].setText(f"{f1:.1f} Hz" if f1 else "—")
            sv["estado"].setText(estado_g)
            sv["estado"].setStyleSheet(
                "border:none; font-size:18px; font-weight:800;"
                " font-family:'Consolas','SF Mono',monospace; color:"
                + ("#16a34a" if estado_g == "Steady" else "#b45309"))
            sv["srate"].setText(f"{fs/1000:.1f} kS/s" if fs >= 1000 else f"{fs:.0f} S/s")
            sv["vent"].setText(f"{snap.shape[1] / fs:.1f} s")
            sv["samp"].setText(f"{total:,}")
            sv["vect"].setText(str(vect))
            sv["guard"].setText(str(int(rec_state.get("guard", 0))))
            sv["size"].setText(f"{len(agent.channels) * snap.shape[1] * 8 / 1e6:.2f} MB")
            if rec_state["fn"] % 32 == 0:
                _refresh_disk()
            # tabular list
            for r, (i, c) in enumerate(vib):
                eu = snap[i] * 1000.0 / (c.sensitivity_mv_per_eu or 1.0)
                eu0 = eu - eu.mean()
                gapv = float(snap[i].mean())            # V (prox real ~-9.5; sim ~0)
                ov, a1, p1, a2, p2 = _amp3(eu0, fs, f1, _ckind(c))
                al, dg = _alarm_for(c)
                if dg and ov >= dg:
                    estado, bgc, fgc = "DANGER", "#fde2e2", "#991b1b"
                elif al and ov >= al:
                    estado, bgc, fgc = "ALERT", "#fdf0d5", "#92400e"
                else:
                    estado, bgc, fgc = "OK", "#e6f4ea", "#166534"
                vals = [c.name, f"{gapv:.2f} V", f"{ov:.3g} {c.units}", f"{a1:.3g}",
                        f"{p1:.0f}°", f"{a2:.3g}", f"{p2:.0f}°",
                        f"{al:g}", f"{dg:g}", estado]
                for cc, v in enumerate(vals):
                    it = QtWidgets.QTableWidgetItem(v)
                    f = it.font()
                    if cc == 0:                                  # sensor: cornflower como la web
                        it.setForeground(QtGui.QColor(CORN)); f.setBold(True); it.setFont(f)
                    elif cc == 7:                                # Alarma: ámbar
                        it.setForeground(QtGui.QColor("#e08a1e"))
                    elif cc == 8:                                # Danger: rojo
                        it.setForeground(QtGui.QColor("#c0392b"))
                    elif cc == 9:                                # Estado: chip de color
                        it.setBackground(QtGui.QColor(bgc)); it.setForeground(QtGui.QColor(fgc))
                        f.setBold(True); it.setFont(f)
                        it.setTextAlignment(QtCore.Qt.AlignCenter)
                    tblt.setItem(r, cc, it)
        elif cur == "Waveform":
            fi = onda_focus["i"]
            nshow = min(snap.shape[1], int(0.6 * fs))          # 600 ms (estándar)
            tms = np.arange(nshow) / fs * 1000.0
            for idx, ((i, c), curve) in enumerate(zip(vib, wave_curves)):
                if fi is not None and idx != fi:
                    continue                                   # foco: solo la elegida
                eu = snap[i, -nshow:] * 1000.0 / (c.sensitivity_mv_per_eu or 1.0)
                eu0 = eu - eu.mean()
                curve.setData(tms, eu0)
                pp = float(eu0.max() - eu0.min()); rms = float(np.sqrt(np.mean(eu0 ** 2)))
                cf = (float(np.max(np.abs(eu0))) / rms) if rms > 1e-9 else 0.0
                wave_stats[idx].setText(f"pp  {pp:.2f} {c.units}\nrms {rms:.2f}\nCF  {cf:.2f}")
                wave_stats[idx].setPos(tms[-1], eu0.max())
                wave_pills[idx].setPos(tms[0], eu0.max())
            if len(tms):
                wave_plots[0].setXRange(0.0, float(tms[-1]), padding=0)  # X desde 0, todas alineadas
            # Espectro: SOLO cuando hay una onda enfocada (doble clic). En la grilla no.
            if fi is not None:
                sel, csel = vib[fi]
                sig = snap[sel] * 1000.0 / (csel.sensitivity_mv_per_eu or 1.0)
                sig0 = sig - sig.mean()
                fr, mag = _spectrum(sig0, fs)
                kind_s = _ckind(csel)
                fmax_hz = 1000.0 if kind_s in ("prox", "vel") else min(0.4 * fs, 10000.0)
                fmax_hz = min(fmax_hz, float(fr[-1]))
                keep = fr <= fmax_hz
                kconv = 2.0 if kind_s == "prox" else (1.0 / np.sqrt(2.0))
                cpm = fr[keep] * 60.0                                   # eje en CPM
                magd = mag[keep] * kconv                                # convención del sensor (pp/rms)
                spec_curve.setData(cpm, magd)
                spec_data["cpm"] = cpm; spec_data["mag"] = magd; spec_data["unit"] = csel.units
                ymax = float(magd.max()) if magd.size else 1.0
                p_spec.setXRange(0, fmax_hz * 60.0, padding=0)         # origen X en 0
                p_spec.setYRange(0, ymax * 1.08 + 1e-9, padding=0)     # base Y en 0
                def _ordamp(o):
                    if not f1:
                        return 0.0
                    ft = o * f1; b = (fr >= ft * 0.8) & (fr <= ft * 1.2)
                    return float(mag[b].max()) * kconv if b.any() else 0.0
                overall = float(sig0.max() - sig0.min()) if kind_s == "prox" else float(np.sqrt(np.mean(sig0 ** 2)))
                if f1:
                    for ln, o in ((v1x, 1), (v2x, 2), (v3x, 3)):
                        ln.setPos(o * f1 * 60.0); ln.show()               # órdenes en CPM
                    spec_info.setText(
                        f"{csel.name}\nOverall {overall:.2f} {csel.units}\n"
                        f"1X  {_ordamp(1):.2f}\n2X  {_ordamp(2):.2f}\n3X  {_ordamp(3):.2f}")
                    spec_info.setPos(fmax_hz * 60.0, ymax)
                else:
                    for ln in (v1x, v2x, v3x):
                        ln.hide()
                    spec_info.setText("")
        elif orb_ok and cur == "Orbit":
            fk = orb_focus["k"]
            nrev = min(snap.shape[1], int((8 * fs / max(rpm, 1)) * 60) if rpm else int(0.4 * fs))
            for idx, it in enumerate(orb_items):
                if fk is not None and idx != fk:
                    continue
                yc, xc = it["yc"], it["xc"]
                Y = snap[it["yi"], -nrev:] * 1000.0 / (yc.sensitivity_mv_per_eu or 1.0)
                X = snap[it["xi"], -nrev:] * 1000.0 / (xc.sensitivity_mv_per_eu or 1.0)
                Y = Y - Y.mean(); X = X - X.mean()
                # orientar al marco físico: proyectar cada sonda a su ángulo (desde TDC)
                px = X * _mo.sin(it["aX"]) + Y * _mo.sin(it["aY"])
                py = X * _mo.cos(it["aX"]) + Y * _mo.cos(it["aY"])
                it["curve"].setData(px, py)
                if len(px):
                    rad = np.hypot(px, py); j = int(np.argmax(rad))
                    it["smax"].setData([0, px[j]], [0, py[j]])
                    it["smax_txt"].setText(f"Smax {rad[j]:.2f} {xc.units}")
                    it["smax_txt"].setPos(px[j], py[j])
                    it["pill"].setPos(float(px.min()), float(py.max()))
                if f1 and len(px):
                    spr = max(1, int(fs / f1))
                    kx, ky = px[::spr], py[::spr]
                    it["kph"].setData(kx, ky)
                    it["kph1"].setData(kx[:1], ky[:1])   # 1er pulso = referencia (rojo grande)
        elif cur == "Bode":
            fk = bode_focus["k"]
            for k, cl in enumerate(bode_cells):
                if fk is not None and k != fk:
                    continue
                rr, am, ph = tc.bode(cl["name"])
                if len(rr) >= 2:
                    amp = np.asarray(am, float) * 2.0
                    cl["amp"].setData(rr, amp)
                    cl["data"] = {"rr": np.asarray(rr, float), "am": amp, "ph": np.asarray(ph, float)}
                    cl["pill"].setPos(float(rr[0]), float(amp.max()))
        elif cur == "Polar":
            fk = pol_focus["k"]
            for k, cl in enumerate(pol_cells):
                if fk is not None and k != fk:
                    continue
                rich = (fk is not None and k == fk)
                rr, am, ph = tc.bode(cl["name"])
                if len(rr) >= 3:
                    rrn = np.asarray(rr, float)
                    a = np.asarray(am, float) * 2.0
                    # ángulo FÍSICO = sonda + fase contra el giro (CW → signo negativo)
                    sign = -1.0 if cb_giro.currentText() == "CW" else 1.0
                    phys = np.radians(cl["pa"] + sign * np.asarray(ph, float))
                    amax = float(a.max()) or 1.0
                    rn = a / amax
                    x = rn * np.sin(phys); y = rn * np.cos(phys)
                    cl["curve"].setData(x, y); cl["pts"].setData(x, y)
                    jc = int(np.argmax(a))                        # Ncrit = máx amplitud
                    if rich:
                        cl["ncrit"].setData([x[jc]], [y[jc]]); cl["op"].setData([x[-1]], [y[-1]])
                        cl["ncrit_txt"].setText(f"Ncrit {rrn[jc]:.0f}")
                        cl["ncrit_txt"].setPos(x[jc], y[jc])
                        af = ""
                        try:
                            _r2 = diag.half_power_af(rrn, a, jc)
                            if _r2:
                                af = f" · AF {_r2[0]:.1f}"
                        except Exception:  # noqa: BLE001
                            pass
                        cl["box"].setText(f"{cl['name']}   Ncrit {rrn[jc]:.0f} rpm · "
                                          f"Amp {a[jc]:.1f} {cl['unit']} · "
                                          f"Fase {float(np.asarray(ph, float)[jc]):.0f}°{af}")
                        idxs = np.unique(np.linspace(0, len(rrn) - 1, 8).round().astype(int))
                        for t, jj in zip(cl["rpm_lbls"], idxs):
                            t.setText(f"{rrn[jj]:.0f}"); t.setPos(float(x[jj]), float(y[jj]))
                        for t in cl["rpm_lbls"][len(idxs):]:
                            t.setText("")
                    else:
                        cl["ncrit"].setData([], []); cl["op"].setData([], [])
                        cl["ncrit_txt"].setText(""); cl["box"].setText("")
                        for t in cl["rpm_lbls"]:
                            t.setText("")
        elif scl_items and cur == "Shaft Centerline":
            for it in scl_items:
                rrk = sorted(scl_track[it["brg"]])       # ya acumulado arriba (siempre)
                if not rrk:
                    continue
                xs = [scl_track[it["brg"]][r][0] for r in rrk]
                ys = [scl_track[it["brg"]][r][1] for r in rrk]
                it["cur"].setData([xs[-1]], [ys[-1]])    # posición actual (último rpm)
                it["line"].setData(xs, ys)
                if _scl_cmap is not None and len(rrk) >= 2:
                    rlo, rhi = rrk[0], max(rrk[-1], rrk[0] + 1)
                    brs = [pg.mkBrush(_scl_cmap.map((r - rlo) / (rhi - rlo), mode="qcolor"))
                           for r in rrk]
                    it["trk"].setData(xs, ys, brush=brs)
                else:
                    it["trk"].setData(xs, ys, brush=pg.mkBrush(CORN))
        elif cur == "Cascade":
            fk = casc_focus["k"]
            for k, cl in enumerate(casc_cells):
                if fk is not None and k != fk:
                    continue
                rr, fr, mat = tc.cascade(cl["name"])
                for cv in cl["curves"]:
                    cv.setData([], [])
                if len(rr) >= 2 and mat.size:
                    idx = np.unique(np.linspace(0, len(rr) - 1, min(len(cl["curves"]), len(rr)))
                                    .round().astype(int))
                    span = float(rr[-1] - rr[0]) or 1.0
                    pk = float(mat.max()) or 1.0
                    sc = (span / max(1, len(idx))) * 1.5 / pk
                    for cv, i in zip(cl["curves"], idx):
                        cv.setData(fr, rr[i] + mat[i] * sc)

    # --- Bilingual strings for the preliminary report (EN default, ES for the client) ---
    REPORT_T = {
        "en": dict(
            need_acq="<i>Start acquisition (▶ Start) to generate the report.</i>",
            title="Preliminary vibration report", machine="Machine", sampling="Sampling",
            verdict="Verdict", verdict_note="(overall vs ISO 20816 / configured levels)",
            verdicts=[("NO FINDINGS", "#166534"), ("WATCH — ALERT level", "#b45309"),
                      ("ACTION — DANGER level", "#b91c1c")],
            s1="1 · Current levels by sensor",
            th=["Sensor", "Overall", "1X", "1X phase", "2X", "Alarm", "Danger", "Status"],
            st=["OK", "ALERT", "DANGER"],
            s2="2 · Critical speeds (API 684)",
            no_crit="<i>No criticals detected (run a run-up/coast-down to evaluate them).</i>",
            s3="3 · Instabilities / sub-synchronous",
            no_sub="<i>No relevant sub-synchronous components.</i>",
            s4="4 · Recommendation",
            rec_danger="<b>DANGER level</b>: schedule shutdown / inspection; check balancing, "
                       "alignment and bearings.",
            rec_alert="<b>ALERT</b> level: increase monitoring frequency and plan corrective action.",
            rec_ok="Levels within standard; continue routine monitoring.",
            rec_whip="<b>Oil whip</b>: severe film instability — act on the bearing.",
            rec_crit="Verify the <b>separation margin</b> to criticals (API 684) at run-up/coast-down.",
            footer="Automatic preliminary report — requires specialist validation. Watermelon System.",
            dfmt="%m/%d/%Y %H:%M"),
        "es": dict(
            need_acq="<i>Iniciá la adquisición (▶ Iniciar) para generar el reporte.</i>",
            title="Reporte preliminar de vibraciones", machine="Máquina", sampling="Muestreo",
            verdict="Veredicto", verdict_note="(overall vs ISO 20816 / niveles configurados)",
            verdicts=[("SIN NOVEDAD", "#166534"), ("OBSERVAR — nivel de ALERTA", "#b45309"),
                      ("ACCIÓN — nivel de PELIGRO", "#b91c1c")],
            s1="1 · Niveles actuales por sensor",
            th=["Sensor", "Overall", "1X", "1X fase", "2X", "Alarma", "Peligro", "Estado"],
            st=["OK", "ALERTA", "PELIGRO"],
            s2="2 · Velocidades críticas (API 684)",
            no_crit="<i>Sin críticas detectadas (hacé un arranque/parada para evaluarlas).</i>",
            s3="3 · Inestabilidades / subsíncronos",
            no_sub="<i>Sin subsíncronos relevantes.</i>",
            s4="4 · Recomendación",
            rec_danger="<b>Nivel de PELIGRO</b>: programar parada / inspección; verificar balanceo, "
                       "alineación y cojinetes.",
            rec_alert="Nivel de <b>ALERTA</b>: aumentar frecuencia de monitoreo y planificar corrección.",
            rec_ok="Niveles dentro de norma; continuar monitoreo de rutina.",
            rec_whip="<b>Oil whip</b>: inestabilidad de película severa — actuar sobre el cojinete.",
            rec_crit="Verificar el <b>margen de separación</b> a las críticas (API 684) en arranque/parada.",
            footer="Reporte automático preliminar — requiere validación de especialista. Watermelon System.",
            dfmt="%d/%m/%Y %H:%M"),
    }

    def run_diag():
        import time as _t
        T = REPORT_T["es" if cb_lang.currentText() == "Español" else "en"]
        snap = agent.snapshot()
        if snap.shape[1] < 16:
            diag_txt.setHtml(T["need_acq"])
            return
        fs = agent.sample_rate_hz
        rpm = agent.estimate_rpm(snap) or 0.0
        f1 = (rpm / 60.0) if rpm else None
        # 1) Current levels by sensor (per-type standard) + status
        rows = []; worst = 0
        for i, c in vib:
            eu = snap[i] * 1000.0 / (c.sensitivity_mv_per_eu or 1.0); eu0 = eu - eu.mean()
            ov, a1, p1, a2, p2 = _amp3(eu0, fs, f1, _ckind(c))
            al, dg = _alarm_for(c)
            if dg and ov >= dg:
                st, scol = T["st"][2], "#b91c1c"; worst = max(worst, 2)
            elif al and ov >= al:
                st, scol = T["st"][1], "#b45309"; worst = max(worst, 1)
            else:
                st, scol = T["st"][0], "#166534"
            rows.append((c.name, ov, c.units, a1, p1, a2, p2, al, dg, st, scol))
        # 2) Critical speeds + AF (from the transient)
        crit_lines = []; crits = set()
        for i, c in vib:
            rb, ab = np.asarray(tc.bode(c.name)[0], float), np.asarray(tc.bode(c.name)[1], float)
            if len(rb) >= 3:
                for j in diag.detect_criticals(rb, ab):
                    nc = float(rb[j]); crits.add(round(nc / 50) * 50)
                    af = ""
                    _r2 = diag.half_power_af(rb, ab, j)
                    if _r2:
                        af = f" · AF {_r2[0]:.1f}"
                    crit_lines.append(f"{c.name}: <b>{nc:.0f} rpm</b>{af}")
        # 3) Sub-synchronous (whirl / whip / ½X)
        rrk, frk, matk = tc.cascade(vib[0][1].name)
        subs = diag.cascade_diagnosis(rrk, frk, matk, sorted(float(x) for x in crits)) if len(rrk) >= 3 else []
        # --- HTML report ---
        verdict = T["verdicts"][worst]
        thr = "".join(f"<th>{x}</th>" for x in T["th"])
        # Branded header (Watermelon System) — white card, green accent bar, navy title.
        # Clean/print-friendly for the report deliverable (the app chrome stays navy).
        logo = _logo_data_uri()
        logo_img = (f"<img src='{logo}' width='44' height='44' "
                    "style='vertical-align:middle;margin-right:12px'>") if logo else "🍉 "
        header = (
            "<table width='100%' cellspacing='0' cellpadding='0' "
            "style='border:1px solid #d6deea'><tr>"
            "<td width='8' style='background:#10b981'>&nbsp;</td>"
            f"<td style='background:#ffffff;padding:12px 16px'>{logo_img}"
            "<span style='color:#0F1E3D;font-size:22px;font-weight:800;vertical-align:middle;"
            "letter-spacing:-.02em'>Watermelon System</span>"
            "<span style='color:#0e9f6e;font-size:13px;font-weight:700;margin-left:12px'>"
            "Vibration &amp; Rotordynamics</span></td></tr></table>")
        h = [f"<div style='font-family:Segoe UI,Arial'>",
             header,
             f"<h2 style='color:{NAVY};margin:10px 0 0 0'>{T['title']}</h2>",
             f"<div style='color:#64748b;font-size:12px'>{T['machine']} <b>{args.machine}</b> · "
             f"{_t.strftime(T['dfmt'])} · RPM {rpm:.0f} · {T['sampling']} {fs/1000:.1f} kS/s</div>",
             f"<p style='font-size:15px'>{T['verdict']}: <b style='color:{verdict[1]}'>{verdict[0]}</b> "
             f"<span style='color:#64748b;font-size:12px'>{T['verdict_note']}</span></p>",
             f"<h3 style='color:#0F1E3D'>{T['s1']}</h3>",
             "<table cellspacing='0' cellpadding='5' border='1' style='border-collapse:collapse;"
             "border-color:#d6deea;font-size:12px'>",
             f"<tr style='background:#0F1E3D;color:#8ec3ef'>{thr}</tr>"]
        for nm, ov, un, a1, p1, a2, p2, al, dg, st, scol in rows:
            h.append(f"<tr><td><b>{nm}</b></td><td>{ov:.2f} {un}</td><td>{a1:.2f}</td>"
                     f"<td>{p1:.0f}°</td><td>{a2:.2f}</td><td>{al:g}</td><td>{dg:g}</td>"
                     f"<td style='color:{scol}'><b>{st}</b></td></tr>")
        h.append("</table>")
        h.append(f"<h3 style='color:#0F1E3D'>{T['s2']}</h3>")
        h.append("<p>" + ("<br>".join(crit_lines) if crit_lines else T["no_crit"]) + "</p>")
        h.append(f"<h3 style='color:#0F1E3D'>{T['s3']}</h3>")
        col = {"info": ACC, "warn": "#b45309", "danger": "#b91c1c"}
        if subs:
            for lvl, title, detail in subs:
                h.append(f"<p style='color:{col.get(lvl,'#333')}'><b>{title}</b><br>{detail}</p>")
        else:
            h.append(f"<p>{T['no_sub']}</p>")
        h.append(f"<h3 style='color:#0F1E3D'>{T['s4']}</h3><ul>")
        if worst >= 2:
            h.append(f"<li>{T['rec_danger']}</li>")
        elif worst == 1:
            h.append(f"<li>{T['rec_alert']}</li>")
        else:
            h.append(f"<li>{T['rec_ok']}</li>")
        if any("WHIP" in (t or "") for _l, t, _d in subs):
            h.append(f"<li>{T['rec_whip']}</li>")
        if crit_lines:
            h.append(f"<li>{T['rec_crit']}</li>")
        h.append(f"</ul><div style='color:#94a3b8;font-size:11px'>{T['footer']}</div></div>")
        html = "".join(h)
        diag_state["html"] = html
        diag_txt.setHtml(html)

    def do_save_report():
        if not diag_state.get("html"):
            _nice("Save report", "<b>Generate the report first</b> with «Generate preliminary report».")
            return
        import time as _t
        from core.remote_monitoring.recorder import _persist_root
        rdir = os.path.join(os.path.dirname(_persist_root()), "reports")
        os.makedirs(rdir, exist_ok=True)
        path = os.path.join(rdir, f"report_{args.machine}_{_t.strftime('%Y%m%d_%H%M%S')}.html")
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("<html><meta charset='utf-8'><body>" + diag_state["html"] + "</body></html>")
            _nice("Report saved",
                  "<div style='font-size:15px'><b style='color:#166534'>✅ Report saved</b></div>"
                  f"<div style='color:#334155;font-family:monospace;margin-top:6px'>{path}</div>"
                  "<div style='color:#64748b;margin-top:8px'>Open it in your browser; you can print it to PDF.</div>")
        except Exception as e:  # noqa: BLE001
            _nice("Save report", f"<b style='color:#b91c1c'>Could not save</b><br>{e}",
                  QtWidgets.QMessageBox.Warning)

    def do_save_pdf():
        """Exporta el reporte a PDF directamente (sin navegador) — listo para enviar."""
        if not diag_state.get("html"):
            _nice("Save PDF", "<b>Generate the report first</b> with «Generate preliminary report».")
            return
        import time as _t
        from core.remote_monitoring.recorder import _persist_root
        rdir = os.path.join(os.path.dirname(_persist_root()), "reports")
        os.makedirs(rdir, exist_ok=True)
        default = os.path.join(rdir, f"report_{args.machine}_{_t.strftime('%Y%m%d_%H%M%S')}.pdf")
        path, _f = QtWidgets.QFileDialog.getSaveFileName(win, "Save report as PDF", default,
                                                         "PDF (*.pdf)")
        if not path:
            return
        try:
            doc = QtGui.QTextDocument()
            doc.setHtml("<div style='font-family:Segoe UI,Arial'>" + diag_state["html"] + "</div>")
            writer = QtGui.QPdfWriter(path)
            writer.setPageSize(QtGui.QPageSize(QtGui.QPageSize.A4))
            try:
                writer.setPageMargins(QtCore.QMarginsF(14, 14, 14, 14),
                                      QtGui.QPageLayout.Unit.Millimeter)
            except Exception:  # noqa: BLE001
                pass
            doc.print_(writer)
            _nice("PDF saved",
                  "<div style='font-size:15px'><b style='color:#166534'>✅ PDF saved</b></div>"
                  f"<div style='color:#334155;font-family:monospace;margin-top:6px'>{path}</div>"
                  "<div style='color:#64748b;margin-top:8px'>Ready to send by email / WhatsApp.</div>")
        except Exception as e:  # noqa: BLE001
            _nice("Save PDF", f"<b style='color:#b91c1c'>Could not export PDF</b><br>{e}",
                  QtWidgets.QMessageBox.Warning)

    def do_upload_report():
        """Sube el reporte (HTML) a la nube junto a las grabaciones (bucket transients)."""
        if not diag_state.get("html"):
            _nice("Upload report", "<b>Generate the report first</b> with «Generate preliminary report».")
            return
        import time as _t
        try:
            from core.remote_monitoring.recorder import _sb_client, _BUCKET
            client = _sb_client()
        except Exception:  # noqa: BLE001
            client = None
        if client is None:
            _nice("Upload report", "<b>No cloud connection.</b> Check your internet and try again; "
                  "the report is still available to save as PDF/HTML locally.",
                  QtWidgets.QMessageBox.Warning)
            return
        slug = "".join(c if c.isalnum() or c in "-_" else "_" for c in args.machine)
        key = f"reports/{slug}/report_{_t.strftime('%Y%m%d_%H%M%S')}.html"
        payload = ("<html><meta charset='utf-8'><body>" + diag_state["html"] + "</body></html>").encode("utf-8")
        btn_diag_cloud.setEnabled(False); btn_diag_cloud.setText("Uploading…")
        res = {}

        def _work():
            try:
                store = client.storage.from_(_BUCKET)
                try:
                    store.upload(key, payload, {"upsert": "true", "content-type": "text/html"})
                except Exception:  # noqa: BLE001
                    store.update(key, payload)
                res["ok"] = True
            except Exception as e:  # noqa: BLE001
                res["err"] = f"{type(e).__name__}: {e}"
            res["done"] = True
        threading.Thread(target=_work, daemon=True).start()

        def _check():
            if not res.get("done"):
                QtCore.QTimer.singleShot(400, _check); return
            btn_diag_cloud.setEnabled(True); btn_diag_cloud.setText("Upload report to cloud")
            if res.get("ok"):
                _nice("Report uploaded",
                      "<div style='font-size:15px'><b style='color:#166534'>☁ Report uploaded</b></div>"
                      f"<div style='color:#334155;font-family:monospace;margin-top:6px'>{_BUCKET}/{key}</div>")
            else:
                _nice("Upload report", f"<b style='color:#b91c1c'>Upload failed</b><br>"
                      f"<span style='color:#334155'>{res.get('err','?')}</span>",
                      QtWidgets.QMessageBox.Warning)
        QtCore.QTimer.singleShot(300, _check)

    timer = QtCore.QTimer(); timer.timeout.connect(update)
    btn_diag.clicked.connect(run_diag)
    btn_diag_pdf.clicked.connect(do_save_pdf)
    btn_diag_save.clicked.connect(do_save_report)
    btn_diag_cloud.clicked.connect(do_upload_report)

    def do_start():
        # Pide nombre/consecutivo de la corrida y GRABA DESDE EL INICIO a disco
        # (así no se pierde nada, aunque guardes/subas después).
        import time as _t
        default = f"{args.machine}_{_t.strftime('%Y%m%d_%H%M%S')}"
        tag, ok = QtWidgets.QInputDialog.getText(
            win, "Start run", "Run name / sequence number:", text=default)
        if not ok:
            return
        try:
            agent.start()
        except Exception as e:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(win, "Error", f"Could not start: {e}")
            return
        try:
            ch_meta = [{"name": c.name, "units": c.units, "coupling": c.coupling,
                        "bnc_port": c.bnc_port, "sensitivity_mv_per_eu": float(c.sensitivity_mv_per_eu or 0)}
                       for c in agent.channels]
            rec = TransientRecorder(agent.instance_id, agent.sample_rate_hz, ch_meta,
                                    machine=args.machine, rec_id=(tag.strip() or None))
            agent.on_block = rec.append          # captura CADA bloque desde el inicio
            rec_state["session"] = rec; rec_state["saved"] = False
        except Exception:  # noqa: BLE001
            rec_state["session"] = None
        act_start.setEnabled(False); act_stop.setEnabled(True)
        lbl_state.setText("● capturing data (from the start)")
        timer.start(90)     # ~11 fps: smooth and lighter on CPU/RAM (modest PCs)

    def do_stop():
        timer.stop()
        try:
            agent.stop()
        except Exception:  # noqa: BLE001
            pass
        agent.on_block = None
        rec = rec_state.get("session")
        if rec and getattr(rec, "open", False):
            rec.stop()
        act_start.setEnabled(True); act_stop.setEnabled(False)
        lbl_rec.setText("")
        if rec:
            lbl_state.setText(f"stopped · {rec.status.duration_s:.0f}s · "
                              f"{rec.status.size_mb:.1f} MB · ready to 💾 Save data")
        else:
            lbl_state.setText("stopped")
        _refresh_disk()

    def _refresh_disk():
        try:
            cnt, used = local_usage(agent.instance_id)
            free = free_bytes()
            lbl_disk.setText(f"Disk: {cnt} recording(s) · {used / 1e6:.0f} MB used · "
                             f"{free / 1e6:.0f} MB free")
            pend = pending_count(agent.instance_id)
            act_sync.setText(f"Upload to cloud ({pend})" if pend else "Upload to cloud")
            act_sync.setEnabled(pend > 0)
        except Exception:  # noqa: BLE001
            pass

    def do_sync():
        # 1) ¿hay credenciales/cliente de nube?
        try:
            from core.remote_monitoring.recorder import _sb_client
            client = _sb_client()
        except Exception:  # noqa: BLE001
            client = None
        if client is None:
            QtWidgets.QMessageBox.warning(
                win, "Upload — no cloud connection",
                "Could not connect to the cloud, so recordings were not uploaded "
                "(they stay saved locally and nothing is lost).\n\n"
                "Please check:\n"
                "1) This computer has an internet connection.\n"
                "2) A firewall/proxy is not blocking the connection.\n\n"
                "You can retry the upload later — the data is kept on disk.")
            _refresh_disk(); return
        # URL que realmente se está usando (para diagnosticar)
        used_url = os.environ.get("WM_SUPABASE_URL", "")
        if not used_url:
            try:
                from core.remote_monitoring import _cloud_config as _cc
                used_url = getattr(_cc, "SUPABASE_URL", "")
            except Exception:  # noqa: BLE001
                used_url = "(no embebida)"
        # 2) subir EN HILO DE FONDO (no congela la UI)
        act_sync.setEnabled(False); act_sync.setText("Uploading…")
        res = {}

        res["ok"] = res["fail"] = 0

        def _work():
            try:
                from core.remote_monitoring.recorder import (list_recordings, is_synced,
                                                             upload_recording)
                pend = [m for m in list_recordings(agent.instance_id) if not is_synced(m["_dir"])]
                res["total"] = len(pend)
                for k, m in enumerate(pend):
                    res["progress"] = f"Uploading {k + 1} of {len(pend)}…"
                    r = upload_recording(m["_dir"])
                    if r.get("ok"):
                        res["ok"] += 1
                    else:
                        res["fail"] += 1
                        res["reason"] = f"{r.get('reason', '?')} · URL: {used_url or '(vacía)'}"
            except Exception as e:  # noqa: BLE001
                res["err"] = f"{type(e).__name__}: {e}"
            res["done"] = True
        threading.Thread(target=_work, daemon=True).start()

        def _check():
            if not res.get("done"):
                lbl_rec.setText(res.get("progress", "uploading…"))
                QtCore.QTimer.singleShot(400, _check); return
            lbl_rec.setText("")
            act_sync.setEnabled(True); act_sync.setText("Upload to cloud"); _refresh_disk()
            ok, fail = res.get("ok", 0), res.get("fail", 0)
            if res.get("err"):
                _nice("Upload", f"<b style='color:#b91c1c'>Upload failed</b><br>"
                      f"<span style='color:#334155'>{res['err']}</span>", QtWidgets.QMessageBox.Warning)
            elif fail == 0 and ok > 0:
                _nice("Uploaded",
                      "<div style='font-size:17px'><b style='color:#166534'>☁ Data synced to the cloud</b></div>"
                      f"<div style='color:#334155;margin-top:6px'>{ok} recording(s) uploaded.</div>"
                      "<div style='color:#0F1E3D;margin-top:10px;font-size:14px'>You can now <b>start the "
                      "diagnosis in Watermelon System</b> (web → Analysis → Reprocess). 🍉</div>")
            elif ok == 0 and fail == 0:
                _nice("Upload", "<b>No pending recordings.</b> Everything is in the cloud.")
            else:
                _nice("Upload",
                      f"<b>Uploaded {ok} · failed {fail}.</b><br>"
                      f"<span style='color:#b45309'>{res.get('reason','')}</span>",
                      QtWidgets.QMessageBox.Warning)
        QtCore.QTimer.singleShot(300, _check)

    def _nice(title, html, icon=QtWidgets.QMessageBox.Information):
        m = QtWidgets.QMessageBox(win)
        m.setWindowTitle(title); m.setIcon(icon)
        m.setTextFormat(QtCore.Qt.RichText)
        m.setText(html)
        m.exec()

    def do_save():
        # 1) Si TODAVÍA está adquiriendo → hay que detener primero
        if act_stop.isEnabled():
            _nice("Save data",
                  "<div style='font-size:15px'><b style='color:#b45309'>⏸ Stop acquisition first</b></div>"
                  "<div style='color:#334155;margin-top:4px'>Press <b>■ Stop</b> and then "
                  "<b>💾 Save data</b>.</div>", QtWidgets.QMessageBox.Warning)
            return
        # 2) is there a stopped run?
        rec = rec_state.get("session")
        if not rec:
            _nice("Save data",
                  "<div style='font-size:15px'><b>No run to save</b></div>"
                  "<div style='color:#334155;margin-top:4px'>Do: <b>▶ Start</b> → measure → "
                  "<b>■ Stop</b> → <b>💾 Save data</b>.</div>")
            return
        # 3) La corrida YA está en disco (se graba desde Iniciar). Guardar = confirmar local.
        if not rec_state.get("saved"):
            rec_state["guard"] = int(rec_state.get("guard", 0)) + 1
            rec_state["saved"] = True
        _refresh_disk()
        pend = 0
        try:
            from core.remote_monitoring.recorder import pending_count
            pend = pending_count(agent.instance_id)
        except Exception:  # noqa: BLE001
            pass
        _nice("Data saved",
              "<div style='font-size:17px'><b style='color:#166534'>✅ Data saved on this computer</b></div>"
              f"<div style='color:#0F1E3D;font-family:monospace;margin-top:6px'>{rec.rec_id}<br>"
              f"{rec.status.duration_s:.0f} s · {rec.status.size_mb:.1f} MB</div>"
              "<div style='color:#b45309;margin-top:10px;font-size:14px'>⏳ <b>Pending cloud upload</b> — "
              f"press <b>↑ Upload to cloud ({pend})</b> when you have internet.</div>")

    def set_mode_live(mode):
        """Change the operating MODE live (steady/run-up/coast-down) on the same
        machine: adjust the profile, rewind the rotor clock and reset the transient
        capture so Bode/Cascade come out clean. `mode` arrives as the UI label."""
        nonlocal tc
        label = mode
        mode = LABEL_TO_MODE.get(mode, mode)     # UI label → internal key
        cf = agent.source.config
        if hasattr(cf, "speed_profile"):
            cf.speed_profile = MODE_TO_PROFILE.get(mode, "constant")
            # Orientar SIEMPRE el barrido según el modo: arranque sube (lo→hi),
            # parada baja (hi→lo). Deriva el rango de la máquina o de la crítica.
            if mode in ("arranque", "parada", "arranque_parada"):
                vals = [v for v in (cf.rpm_start, cf.rpm_end) if v > 0]
                lo = min(vals) if vals else 300.0
                hi = max([cf.rpm_start, cf.rpm_end, cf.rpm * 2.0,
                          (cf.sim_critical_rpm or 0) * 2.5, (cf.sim_critical_rpm2 or 0) * 1.4])
                if hi <= lo:
                    lo, hi = 300.0, max(6000.0, cf.rpm * 2.0)
                cf.rpm_start, cf.rpm_end = (hi, lo) if mode == "parada" else (lo, hi)
                cf.ramp_seconds = max(cf.ramp_seconds, 60.0)
        if hasattr(agent.source, "rewind"):
            agent.source.rewind()
        tc = TransientCapture(TransientConfig(fmax_hz=min(2000.0, agent.sample_rate_hz / 2.5)))
        for cl in casc_cells:
            for cv in cl["curves"]:
                cv.setData([], [])
        for cl in bode_cells:
            cl["amp"].setData([], []); cl["data"] = {}; cl["cur"].hide(); cl["txt"].setText("")
        for cl in pol_cells:
            cl["curve"].setData([], []); cl["pts"].setData([], [])
            cl["ncrit"].setData([], []); cl["op"].setData([], [])
            cl["ncrit_txt"].setText(""); cl["box"].setText("")
            for _t in cl["rpm_lbls"]:
                _t.setText("")
        for _b in scl_track:                 # reiniciar la traza del shaft centerline
            scl_track[_b].clear()
        lbl_state.setText(f"● {label}" if agent.source.is_running() else f"mode: {label}")

    _prof0 = getattr(agent.source.config, "speed_profile", "constant")
    _inv = {v: k for k, v in MODE_TO_PROFILE.items()}
    cb_run.blockSignals(True)
    cb_run.setCurrentText(MODE_LABELS.get(_inv.get(_prof0, "estable"), "Steady"))
    cb_run.blockSignals(False)
    cb_run.currentTextChanged.connect(set_mode_live)
    act_start.triggered.connect(do_start)
    act_stop.triggered.connect(do_stop)
    act_save.triggered.connect(do_save)     # barra: Guardar datos
    act_sync.triggered.connect(do_sync)     # barra: Subir a la nube

    def do_clear():
        from core.remote_monitoring.recorder import clear_recordings, pending_count, local_usage
        cnt, used = local_usage(agent.instance_id)
        if cnt == 0:
            QtWidgets.QMessageBox.information(win, "Delete local data", "No local recordings.")
            return
        pend = pending_count(agent.instance_id)
        m = QtWidgets.QMessageBox(win)
        m.setWindowTitle("Delete local recordings")
        m.setIcon(QtWidgets.QMessageBox.Warning)
        m.setText(f"Delete {cnt} local recording(s) ({used/1e6:.0f} MB)?")
        m.setInformativeText(
            (f"⚠ {pend} are NOT uploaded to the cloud: if you delete them, they are lost.\n\n"
             if pend else "The ones already in the cloud stay in the cloud.\n\n")
            + "Continue?")
        only = None
        if pend:
            bt_all = m.addButton("Delete ALL", QtWidgets.QMessageBox.DestructiveRole)
            bt_synced = m.addButton("Only uploaded ones", QtWidgets.QMessageBox.AcceptRole)
            m.addButton("Cancel", QtWidgets.QMessageBox.RejectRole)
            m.exec()
            cl = m.clickedButton()
            if cl == bt_all: only = False
            elif cl == bt_synced: only = True
            else: return
        else:
            m.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
            if m.exec() != QtWidgets.QMessageBox.Yes:
                return
            only = False
        n, freed = clear_recordings(agent.instance_id, only_synced=bool(only))
        rec_state["guard"] = 0
        _refresh_disk()
        QtWidgets.QMessageBox.information(win, "Delete local data",
                                         f"Deleted {n} recording(s) · freed {freed/1e6:.0f} MB.")
    act_clear.triggered.connect(do_clear)   # barra: Borrar datos
    _refresh_disk()
    act_quit.triggered.connect(win.close)
    act_about.triggered.connect(lambda: QtWidgets.QMessageBox.about(
        win, "Watermelon Field", "Watermelon Field — native acquisition module.\n"
        "Rotordynamics API 670/684 · integrated cloud.\n© SIGA"))

    win.showMaximized()      # use the full screen (adapts to any resolution)
    ret = app.exec()
    try:
        agent.stop()
    except Exception:  # noqa: BLE001
        pass
    return ret


def _run_with_crashlog() -> int:
    """Corre main() y, si algo revienta al iniciar, guarda el traceback en un
    archivo junto al .exe y lo muestra en un diálogo (para diagnosticar en campo)."""
    try:
        return main()
    except SystemExit:
        raise
    except BaseException:  # noqa: BLE001
        import traceback
        base = os.path.dirname(sys.executable if getattr(sys, "frozen", False)
                               else os.path.abspath(__file__))
        logp = os.path.join(base, "watermelon_error.log")
        tb = traceback.format_exc()
        try:
            with open(logp, "a", encoding="utf-8") as f:
                f.write("\n==== startup error ====\n" + tb + "\n")
        except Exception:  # noqa: BLE001
            pass
        try:
            from PySide6 import QtWidgets
            _a = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
            QtWidgets.QMessageBox.critical(None, "Watermelon Field — startup error",
                                           f"{tb}\n\nDetails saved to:\n{logp}")
        except Exception:  # noqa: BLE001
            print(tb)
        return 1


if __name__ == "__main__":
    sys.exit(_run_with_crashlog())
