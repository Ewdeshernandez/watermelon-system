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
SENSOR_ES = {"prox": "Proximidad", "vel": "Velocidad", "accel": "Acelerómetro", "keyphasor": "Keyphasor"}


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


def _stylesheet() -> str:
    return f"""
    QMainWindow, QDialog {{ background: {BG}; }}
    QWidget {{ background: {BG}; color: {INK};
        font-family: 'Segoe UI', 'Inter', Arial, sans-serif; font-size: 13px; }}
    QLabel {{ background: transparent; color: {INK}; }}
    QMenuBar {{ background: {NAVY}; color: #eaf1fb; padding: 2px; }}
    QMenuBar::item {{ padding: 5px 10px; background: transparent; color: #eaf1fb; }}
    QMenuBar::item:selected {{ background: {ACC}; border-radius: 5px; color: white; }}
    QMenu {{ background: {PANEL}; border: 1px solid {LINE}; color: {INK}; }}
    QMenu::item:selected {{ background: {ACC}; color: white; }}
    QToolBar {{ background: {NAVY}; spacing: 8px; padding: 7px 10px;
        border-bottom: 1px solid {NAVY}; }}
    QToolBar::separator {{ background: #2b3d5f; width: 1px; margin: 4px 6px; }}
    QToolBar QToolButton {{ color: #eaf1fb; padding: 7px 14px; border-radius: 7px;
        font-weight: 600; }}
    QToolBar QToolButton:hover {{ background: {ACC}; color: white; }}
    QToolBar QToolButton:disabled {{ color: #6b7d9c; }}
    QTabWidget::pane {{ border: 1px solid {LINE}; background: {PANEL};
        border-radius: 8px; top: -1px; }}
    QTabBar {{ qproperty-drawBase: 0; }}
    QTabBar::tab {{ background: transparent; color: {MUTE}; padding: 9px 20px;
        margin-right: 3px; border: none; font-weight: 600;
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
        border: none; border-right: 1px solid #24344f; font-weight: 700;
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
        print("Instalá deps del nativo:  pip install -r native/requirements.txt")
        return 2

    agent = build_agent(args)
    vib = [(i, c) for i, c in enumerate(agent.channels) if not is_keyphasor_channel(c)]
    from core.remote_monitoring.recorder import TransientRecorder, upload_recording
    from core.remote_monitoring.transient import TransientCapture, TransientConfig
    from core.remote_monitoring import analysis as diag
    kph_glob = agent.source.config.keyphasor_index()
    from core.remote_monitoring.sim_machine import MODES, MODE_TO_PROFILE
    tc = TransientCapture(TransientConfig(fmax_hz=min(2000.0, args.fs / 2.5)))

    pg.setConfigOptions(antialias=True, background=PANEL, foreground=MUTE)
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(_stylesheet())

    win = QtWidgets.QMainWindow()
    win.setWindowTitle(f"Watermelon Field — {args.machine}")
    win.resize(1360, 860)
    rec_state = {"rec": None}

    # ---------------- Menú ----------------
    mb = win.menuBar()
    m_file = mb.addMenu("&Archivo")
    m_view = mb.addMenu("&Ver")
    m_help = mb.addMenu("A&yuda")
    act_start = QtGui.QAction("Iniciar", win)
    act_stop = QtGui.QAction("Detener", win); act_stop.setEnabled(False)
    act_save = QtGui.QAction("Guardar datos", win)
    act_sync = QtGui.QAction("Subir a la nube", win)
    act_clear = QtGui.QAction("Borrar datos", win)
    act_quit = QtGui.QAction("Salir", win)
    for a in (act_start, act_stop, act_save, act_sync, act_clear):
        m_file.addAction(a)
    m_file.addSeparator(); m_file.addAction(act_quit)
    act_about = QtGui.QAction("Acerca de Watermelon Field", win)
    m_help.addAction(act_about)

    # ---------------- Toolbar ----------------
    tb = win.addToolBar("Principal")
    tb.setMovable(False)
    tb.addAction(act_start); tb.addAction(act_stop); tb.addAction(act_save)
    tb.addSeparator()
    tb.addAction(act_sync); tb.addAction(act_clear)
    tb.addSeparator()
    lbl_modo = QtWidgets.QLabel(" Modo: "); lbl_modo.setStyleSheet("color:white;")
    tb.addWidget(lbl_modo)
    cb_run = QtWidgets.QComboBox(); cb_run.addItems(MODES); tb.addWidget(cb_run)
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
    _KIND_LABELS = [("Proximidad", "prox"), ("Velocidad", "vel"),
                    ("Acelerómetro", "accel"), ("Keyphasor", "keyphasor")]
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
    cfg_scroll = QtWidgets.QScrollArea(); cfg_scroll.setWidgetResizable(True)
    cfg_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
    cfg_w = QtWidgets.QWidget(); cfg_l = QtWidgets.QVBoxLayout(cfg_w)
    cfg_scroll.setWidget(cfg_w); cfg_ol.addWidget(cfg_scroll)

    def _sec(txt):
        lb = QtWidgets.QLabel(txt)
        lb.setStyleSheet(f"background:{NAVY}; color:white; font-weight:700; font-size:13px;"
                         f"padding:8px 13px; border-radius:6px; margin-top:6px;")
        return lb

    cfg_l.addWidget(_sec("1 · Máquina  —  tren (API 684)"))
    # fila 1: nombre, fs, biblioteca
    r1 = QtWidgets.QHBoxLayout()
    r1.addWidget(QtWidgets.QLabel("Máquina:"))
    ed_name = QtWidgets.QLineEdit(); r1.addWidget(ed_name, 2)
    r1.addWidget(QtWidgets.QLabel("Muestreo (Hz):"))
    sp_fs = QtWidgets.QSpinBox(); sp_fs.setRange(256, 102400); sp_fs.setSingleStep(1280); r1.addWidget(sp_fs)
    r1.addWidget(QtWidgets.QLabel("Biblioteca:"))
    cb_lib = QtWidgets.QComboBox(); cb_lib.setMinimumWidth(160); r1.addWidget(cb_lib)
    btn_load = QtWidgets.QPushButton("Cargar"); r1.addWidget(btn_load)
    cfg_l.addLayout(r1)
    # fila 2: operación
    r2 = QtWidgets.QHBoxLayout()
    r2.addWidget(QtWidgets.QLabel("Modo:"))
    cb_mode = QtWidgets.QComboBox(); cb_mode.addItems(MODES); r2.addWidget(cb_mode)
    def _dsp(mn, mx, val, step=100.0):
        s = QtWidgets.QDoubleSpinBox(); s.setRange(mn, mx); s.setValue(val); s.setSingleStep(step); return s
    r2.addWidget(QtWidgets.QLabel("RPM:")); sp_rpm = _dsp(0, 60000, 3000); r2.addWidget(sp_rpm)
    r2.addWidget(QtWidgets.QLabel("Arranque→")); sp_r0 = _dsp(0, 60000, 300); r2.addWidget(sp_r0)
    sp_r1 = _dsp(0, 60000, 6000); r2.addWidget(sp_r1)
    r2.addWidget(QtWidgets.QLabel("Rampa(s):")); sp_ramp = _dsp(1, 3600, 90, 5); r2.addWidget(sp_ramp)
    cfg_l.addLayout(r2)
    # fila 2b: máquina (rotación / cojinete)
    r3b = QtWidgets.QHBoxLayout()
    r3b.addWidget(QtWidgets.QLabel("Sentido de giro:"))
    cb_rot = QtWidgets.QComboBox(); cb_rot.addItems(["CCW", "CW"]); r3b.addWidget(cb_rot)
    r3b.addWidget(QtWidgets.QLabel("Tipo de cojinete:"))
    cb_brg = QtWidgets.QComboBox(); cb_brg.addItems(["plain", "tilting_pad", "rolling", "mixed"])
    r3b.addWidget(cb_brg); r3b.addStretch(1); cfg_l.addLayout(r3b)

    cfg_l.addWidget(_sec("Fenómenos y transitorio  —  inyectá fallas por tipo de sensor"))
    r3 = QtWidgets.QHBoxLayout()
    r3.addWidget(QtWidgets.QLabel("Crítica 1:")); sp_c1 = _dsp(0, 60000, 0); r3.addWidget(sp_c1)
    r3.addWidget(QtWidgets.QLabel("Crítica 2:")); sp_c2 = _dsp(0, 60000, 0); r3.addWidget(sp_c2)
    r3.addWidget(QtWidgets.QLabel("Severidad:")); sp_sev = _dsp(0, 3, 1.0, 0.25); r3.addWidget(sp_sev)
    r3.addWidget(QtWidgets.QLabel("Prox:"))
    cb_ph_p = QtWidgets.QComboBox(); cb_ph_p.addItems(PHENOMENA["prox"]); r3.addWidget(cb_ph_p)
    r3.addWidget(QtWidgets.QLabel("Vel:"))
    cb_ph_v = QtWidgets.QComboBox(); cb_ph_v.addItems(PHENOMENA["vel"]); r3.addWidget(cb_ph_v)
    r3.addWidget(QtWidgets.QLabel("Accel:"))
    cb_ph_a = QtWidgets.QComboBox(); cb_ph_a.addItems(PHENOMENA["accel"]); r3.addWidget(cb_ph_a)
    r3.addStretch(1); cfg_l.addLayout(r3)

    cfg_l.addWidget(_sec("2 · Canales  —  BNC → punto de medición"))
    # leyenda de colores por tipo (idéntica a la web) + convención de ángulo
    leg = QtWidgets.QLabel(
        f"<span style='color:{SENSOR_COLORS['prox']}'>●</span> Proximidad&nbsp;&nbsp;"
        f"<span style='color:{SENSOR_COLORS['vel']}'>●</span> Velocidad&nbsp;&nbsp;"
        f"<span style='color:{SENSOR_COLORS['accel']}'>●</span> Acelerómetro&nbsp;&nbsp;"
        f"<span style='color:{SENSOR_COLORS['keyphasor']}'>●</span> Keyphasor"
        f"&nbsp;&nbsp;&nbsp;<span style='color:#64748b'>Ángulo API 670: desde TDC · "
        f"R=horario · L=antihorario (45°L+45°R=90°)</span>")
    cfg_l.addWidget(leg)
    # tabla + diagrama de sección lado a lado
    canv = QtWidgets.QHBoxLayout()
    tblc = QtWidgets.QTableWidget(0, 9)
    tblc.setHorizontalHeaderLabels(["Canal", "Tipo", "BNC", "Sensib (mV/EU)",
                                    "Ángulo°", "Lado", "Gap (V)", "Alarma", "Peligro"])
    tblc.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
    canv.addWidget(tblc, 3)
    brg_plot = pg.PlotWidget(); brg_plot.setBackground("w"); brg_plot.setAspectLocked(True)
    brg_plot.hideAxis("left"); brg_plot.hideAxis("bottom"); brg_plot.setMenuEnabled(False)
    brg_plot.setMinimumWidth(280)
    canv.addWidget(brg_plot, 2)
    cfg_l.addLayout(canv, 1)
    rb = QtWidgets.QHBoxLayout()
    btn_add = QtWidgets.QPushButton("+ Sensor"); btn_del = QtWidgets.QPushButton("– Quitar")
    btn_tpl = QtWidgets.QPushButton("Plantilla motor+bomba")
    btn_save = QtWidgets.QPushButton("Guardar configuración")
    btn_apply = QtWidgets.QPushButton("Aplicar y medir")
    _redbtn = ("QPushButton{background:#f5484a;color:white;border:none;font-weight:700;"
               "padding:8px 16px;border-radius:7px;} QPushButton:hover{background:#d63c3e;}")
    btn_save.setStyleSheet(_redbtn); btn_apply.setStyleSheet(_redbtn)
    for b in (btn_add, btn_del, btn_tpl): rb.addWidget(b)
    rb.addStretch(1); rb.addWidget(btn_save); rb.addWidget(btn_apply)
    cfg_l.addLayout(rb)
    tabs.addTab(cfg_outer, "Configuración")

    import math as _math

    def draw_bearing():
        """Dibuja la sección del cojinete (bolitas de color en su ángulo) como la web."""
        try:
            brg_plot.clear()
            th = np.linspace(0, 2 * np.pi, 200)
            R = 1.0
            brg_plot.plot(R * np.sin(th), R * np.cos(th), pen=pg.mkPen(NAVY, width=14))  # anillo
            brg_plot.plot(0.34 * np.sin(th), 0.34 * np.cos(th),
                          pen=pg.mkPen("#c9d6e8", width=2))                                # eje
            tdc = pg.ScatterPlotItem([0], [R + 0.14], symbol="t", size=15, brush=ACC, pen=None)
            brg_plot.addItem(tdc)
            t0 = pg.TextItem("TDC 0°", color=NAVY, anchor=(0.5, 1.2)); t0.setPos(0, R + 0.16)
            brg_plot.addItem(t0)
            m = read_form()
            for s in m.sensors:
                a = _math.radians(s.abs_angle())
                x, y = R * _math.sin(a), R * _math.cos(a)     # 0°=arriba, R=horario
                col = SENSOR_COLORS.get(s.kind, "#8b5cf6")
                brg_plot.addItem(pg.ScatterPlotItem([x], [y], symbol="o", size=26,
                                                    brush=col, pen=pg.mkPen("w", width=2)))
                lb = pg.TextItem(s.name, color="w", anchor=(0.5, 0.5)); lb.setPos(x, y)
                brg_plot.addItem(lb)
            cx = pg.TextItem(("CW ↻" if m.rotation == "CW" else "CCW ↺"),
                             color=NAVY, anchor=(0.5, 0.5)); cx.setPos(0, 0)
            brg_plot.addItem(cx)
            brg_plot.setXRange(-1.5, 1.5); brg_plot.setYRange(-1.4, 1.4)
        except Exception:  # noqa: BLE001
            pass

    def _color_name_cell(r):
        w = tblc.cellWidget(r, 1)
        kind = _KIND_BY_LABEL.get(w.currentText(), "accel") if w else "accel"
        it = tblc.item(r, 0)
        if it:
            it.setForeground(QtGui.QColor(SENSOR_COLORS.get(kind, "#8b5cf6")))
            f = it.font(); f.setBold(True); it.setFont(f)

    def _add_sensor_row(s: SensorSpec):
        r = tblc.rowCount(); tblc.insertRow(r)
        tblc.setItem(r, 0, QtWidgets.QTableWidgetItem(s.name))
        cbk = QtWidgets.QComboBox(); cbk.addItems([l for l, _ in _KIND_LABELS])
        cbk.setCurrentText(_LABEL_BY_KIND.get(s.kind, "Acelerómetro"))
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
        _color_name_cell(r)

    def fill_form(m: SimMachine):
        ed_name.setText(m.name); sp_fs.setValue(int(m.fs))
        cb_rot.setCurrentText(getattr(m, "rotation", "CCW"))
        cb_brg.setCurrentText(getattr(m, "bearing_type", "plain"))
        cb_mode.setCurrentText(m.mode if m.mode in MODES else "estable")
        sp_rpm.setValue(m.rpm); sp_r0.setValue(m.rpm_start); sp_r1.setValue(m.rpm_end)
        sp_ramp.setValue(m.ramp_s); sp_c1.setValue(m.crit1); sp_c2.setValue(m.crit2)
        sp_sev.setValue(m.severity)
        cb_ph_p.setCurrentText(m.phenomena.get("prox", "none"))
        cb_ph_v.setCurrentText(m.phenomena.get("vel", "none"))
        cb_ph_a.setCurrentText(m.phenomena.get("accel", "none"))
        tblc.setRowCount(0)
        for s in m.sensors: _add_sensor_row(s)
        draw_bearing()

    def read_form() -> SimMachine:
        sens = []
        for r in range(tblc.rowCount()):
            nm = tblc.item(r, 0).text() if tblc.item(r, 0) else f"CH{r}"
            lbl = tblc.cellWidget(r, 1).currentText() if tblc.cellWidget(r, 1) else "Acelerómetro"
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
        return SimMachine(name=ed_name.text() or "Maquina", fs=float(sp_fs.value()), sensors=sens,
                          rotation=cb_rot.currentText(), bearing_type=cb_brg.currentText(),
                          mode=cb_mode.currentText(), rpm=sp_rpm.value(),
                          rpm_start=sp_r0.value(), rpm_end=sp_r1.value(), ramp_s=sp_ramp.value(),
                          crit1=sp_c1.value(), crit2=sp_c2.value(), severity=sp_sev.value(),
                          phenomena={k: v for k, v in ph.items() if v != "none"})

    def refresh_lib():
        cb_lib.clear(); cb_lib.addItems(list_machines() or ["(vacía)"])

    def do_load_lib():
        nm = cb_lib.currentText()
        if nm and nm != "(vacía)":
            fill_form(load_from_library(nm))

    def do_save_lib():
        m = read_form()
        try:
            save_to_library(m); refresh_lib()
            QtWidgets.QMessageBox.information(win, "Biblioteca", f"Máquina '{m.name}' guardada.")
        except Exception as e:  # noqa: BLE001
            QtWidgets.QMessageBox.warning(win, "Biblioteca", f"No se pudo guardar: {e}")

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
                win, "Aplicar y medir", f"La configuración tiene un problema y no se puede "
                f"medir:\n\n{type(e).__name__}: {e}")
            return
        # 2) Relanzar en una ventana que QUEDA ABIERTA si hay error (para verlo).
        try:
            if getattr(sys, "frozen", False):
                bat = os.path.join(tempfile.gettempdir(), "wm_run_machine.bat")
                with open(bat, "w") as f:
                    f.write(f'@echo off\r\n"{sys.executable}" --machine-file "{path}"\r\n'
                            f'if errorlevel 1 (echo. & echo *** ERROR al iniciar *** & pause)\r\n')
                os.startfile(bat)  # noqa: S606  (Windows)
            else:
                subprocess.Popen([sys.executable, os.path.abspath(__file__),
                                  "--machine-file", path])
            win.close()
        except Exception as e:  # noqa: BLE001
            QtWidgets.QMessageBox.warning(win, "Aplicar", f"No se pudo relanzar: {e}")

    btn_add.clicked.connect(lambda: _add_sensor_row(SensorSpec("CHn", "accel", tblc.rowCount() + 1)))
    btn_del.clicked.connect(lambda: tblc.removeRow(tblc.currentRow()) if tblc.currentRow() >= 0 else None)
    btn_tpl.clicked.connect(lambda: fill_form(SimMachine.plantilla_motor_bomba()))
    btn_load.clicked.connect(do_load_lib)
    btn_save.clicked.connect(do_save_lib)
    btn_apply.clicked.connect(do_apply)
    refresh_lib(); fill_form(_machine_from_agent())
    tblc.itemChanged.connect(lambda *_: draw_bearing())   # redibuja al editar ángulo/nombre
    cb_rot.currentTextChanged.connect(lambda *_: draw_bearing())

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
    _cells = [("rpm", "RPM"), ("x1", "1X"), ("estado", "ESTADO"), ("srate", "MUESTREO"),
              ("vent", "VENTANA"), ("samp", "SAMPLES"), ("vect", "VECTORES"),
              ("guard", "GUARDADOS"), ("size", "TAMAÑO")]
    for _n, (_k, _l) in enumerate(_cells):
        _statcell(_k, _l, first=(_n == 0))
    sl.addStretch(1); mon_l.addWidget(strip)
    # (Iniciar/Detener/Guardar/Subir a la nube/Borrar datos viven en la barra superior)
    ctl = QtWidgets.QHBoxLayout()
    lbl_disk = QtWidgets.QLabel("Disco: —"); lbl_disk.setStyleSheet("color:#64748b;")
    ctl.addStretch(1); ctl.addWidget(lbl_disk)
    mon_l.addLayout(ctl)
    # tabular list — valores actuales (rápido)
    tblt = QtWidgets.QTableWidget(len(vib), 10)
    tblt.setHorizontalHeaderLabels(["Sensor", "Gap", "Overall", "1X", "1X fase",
                                    "2X", "2X fase", "Alarma", "Danger", "Estado"])
    tblt.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
    tblt.verticalHeader().setVisible(False)
    tblt.setAlternatingRowColors(True)
    tblt.setShowGrid(False)
    tblt.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
    tblt.setFocusPolicy(QtCore.Qt.NoFocus)
    mon_l.addWidget(tblt, 1)
    mon_l.addWidget(QtWidgets.QLabel(
        "<i style='color:#64748b'>Amplitudes por norma: desplazamiento en pp (API 670 · "
        "ISO 7919), velocidad/aceleración en RMS (ISO 20816).</i>"))
    tabs.addTab(mon_w, "Monitoreo")

    # --- Onda (ANÁLISIS: formas de onda + espectro) ---
    ond_w = QtWidgets.QWidget(); ond_l = QtWidgets.QVBoxLayout(ond_w)
    top = QtWidgets.QHBoxLayout()
    top.addWidget(QtWidgets.QLabel("<b>Formas de onda</b>"))
    top.addWidget(QtWidgets.QLabel(
        "<i style='color:#64748b'>doble clic en una onda = verla sola con su espectro (FFT) · "
        "doble clic otra vez = volver a todas</i>"))
    top.addStretch(1); ond_l.addLayout(top)
    gl = pg.GraphicsLayoutWidget(); ond_l.addWidget(gl, 1)
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
    p_spec.setLabel("left", "amplitud"); p_spec.setLabel("bottom", "Frecuencia (CPM)")
    p_spec.setTitle("Espectro (FFT)", color=NAVY, size="9pt")
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
    tabs.addTab(ond_w, "Onda")

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
            "<b>Órbitas</b> <i style='color:#64748b'>— orientadas al ángulo de sonda "
            "(Y=45°L · X=45°R) · doble clic = ver una sola · doble clic otra vez = volver</i>"))
        gl_orb = pg.GraphicsLayoutWidget(); orb_l.addWidget(gl_orb, 1)
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
                pill=pg.TextItem(f"Cojinete {brg}", color="w", anchor=(0, 0), fill=pg.mkBrush(NAVY)))
            p.addItem(it["smax_txt"]); p.addItem(it["pill"])
            orb_items.append(it)
        tabs.addTab(orb_w, "Órbita")

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

    # --- Bode (amp + fase vs rpm, se llena en runup) ---
    bode_w = QtWidgets.QWidget(); bode_l = QtWidgets.QVBoxLayout(bode_w)
    bh = QtWidgets.QHBoxLayout(); bh.addWidget(QtWidgets.QLabel("Canal:"))
    cb_bode = QtWidgets.QComboBox(); cb_bode.addItems([c.name for _, c in vib]); bh.addWidget(cb_bode)
    bh.addStretch(1); bode_l.addLayout(bh)
    gl_b = pg.GraphicsLayoutWidget(); bode_l.addWidget(gl_b, 1)
    p_ph = gl_b.addPlot(row=0, col=0); p_ph.setLabel("left", "Fase 1X (°)"); p_ph.showGrid(x=True, y=True, alpha=0.25)
    p_ph.getAxis("bottom").setStyle(showValues=False); p_ph.invertY(True)
    c_ph = p_ph.plot(pen=pg.mkPen(AMBER, width=1.8), symbol="o", symbolSize=4,
                     symbolBrush=AMBER, symbolPen=None)
    p_am = gl_b.addPlot(row=1, col=0); p_am.setLabel("left", "1X"); p_am.setLabel("bottom", "RPM")
    p_am.showGrid(x=True, y=True, alpha=0.25)
    c_am = p_am.plot(pen=pg.mkPen(CORN, width=1.8), symbol="o", symbolSize=4,
                     symbolBrush=CORN, symbolPen=None)
    bode_hint = pg.TextItem("Poné Modo → arranque (o parada) para llenar el Bode",
                            color=MUTE, anchor=(0.5, 0.5)); p_am.addItem(bode_hint)
    tabs.addTab(bode_w, "Bode")

    # --- Polar (Nyquist del vector 1X vs rpm, se llena en runup) ---
    pol_w = QtWidgets.QWidget(); pol_l = QtWidgets.QVBoxLayout(pol_w)
    ph2 = QtWidgets.QHBoxLayout(); ph2.addWidget(QtWidgets.QLabel("Canal:"))
    cb_pol = QtWidgets.QComboBox(); cb_pol.addItems([c.name for _, c in vib]); ph2.addWidget(cb_pol)
    ph2.addStretch(1); pol_l.addLayout(ph2)
    p_pol = pg.PlotWidget(); p_pol.setAspectLocked(True); p_pol.showGrid(x=True, y=True, alpha=0.12)
    p_pol.addLine(x=0, pen=pg.mkPen("#e6ecf5")); p_pol.addLine(y=0, pen=pg.mkPen("#e6ecf5"))
    pol_curve = p_pol.plot(pen=pg.mkPen(CORN, width=1.8))
    pol_pts = p_pol.plot(pen=None, symbol="o", symbolSize=5, symbolBrush=KPH, symbolPen=None)
    pol_hint = pg.TextItem("Poné Modo → arranque para llenar el Polar", color=MUTE, anchor=(0.5, 0.5))
    p_pol.addItem(pol_hint)
    pol_l.addWidget(p_pol, 1)
    tabs.addTab(pol_w, "Polar")

    # --- Cascada (espectros apilados) ---
    casc_w = QtWidgets.QWidget(); casc_l = QtWidgets.QVBoxLayout(casc_w)
    ch2 = QtWidgets.QHBoxLayout(); ch2.addWidget(QtWidgets.QLabel("Canal:"))
    cb_casc = QtWidgets.QComboBox(); cb_casc.addItems([c.name for _, c in vib]); ch2.addWidget(cb_casc)
    ch2.addStretch(1); casc_l.addLayout(ch2)
    p_casc = pg.PlotWidget(); p_casc.setLabel("bottom", "Frecuencia (Hz)"); p_casc.setLabel("left", "RPM")
    p_casc.showGrid(x=True, y=True, alpha=0.2)
    casc_curves = [p_casc.plot(pen=pg.mkPen(CORN, width=0.9)) for _ in range(40)]
    casc_l.addWidget(p_casc, 1)
    tabs.addTab(casc_w, "Cascada")

    # --- Shaft Centerline (posición del muñón en el cojinete vs rpm) ---
    scl_items = []
    scl_track = {}          # brg -> {rpm_bucket: (x, y)}
    if orb_ok:
        scl_w = QtWidgets.QWidget(); scl_l = QtWidgets.QVBoxLayout(scl_w)
        scl_l.addWidget(QtWidgets.QLabel(
            "<b>Shaft Centerline</b> <i style='color:#64748b'>— posición del muñón en el juego del "
            "cojinete al variar la velocidad (arranque/parada). El punto rojo = actual.</i>"))
        gl_scl = pg.GraphicsLayoutWidget(); scl_l.addWidget(gl_scl, 1)
        ncol = 2 if len(orb_pairs) > 1 else 1
        _th = np.linspace(0, 2 * np.pi, 120); Cclr = 10.0     # juego dibujado (mil)
        for k, (brg, (yi, yc), (xi, xc)) in enumerate(orb_pairs):
            p = gl_scl.addPlot(row=k // ncol, col=k % ncol)
            p.setAspectLocked(True); p.showGrid(x=True, y=True, alpha=0.1)
            p.hideAxis("left"); p.hideAxis("bottom"); p.setMenuEnabled(False)
            p.plot(Cclr * np.sin(_th), Cclr * np.cos(_th), pen=pg.mkPen(NAVY, width=2))  # juego
            p.addLine(x=0, pen=pg.mkPen("#e6ecf5")); p.addLine(y=0, pen=pg.mkPen("#e6ecf5"))
            it = dict(p=p, yi=yi, xi=xi, yc=yc, xc=xc, brg=brg,
                      aY=_mo.radians(_probe_angle(yc.name)), aX=_mo.radians(_probe_angle(xc.name)),
                      track=p.plot(pen=pg.mkPen(CORN, width=1.5)),
                      cur=p.plot(pen=None, symbol="o", symbolSize=12, symbolBrush=REDL,
                                 symbolPen=pg.mkPen("w", width=2)),
                      pill=pg.TextItem(f"Cojinete {brg}", color="w", anchor=(0, 0), fill=pg.mkBrush(NAVY)))
            p.addItem(it["pill"]); p.setXRange(-Cclr * 1.2, Cclr * 1.2); p.setYRange(-Cclr * 1.2, Cclr * 1.2)
            it["pill"].setPos(-Cclr * 1.1, Cclr * 1.1)
            scl_items.append(it); scl_track[brg] = {}
        tabs.addTab(scl_w, "Shaft Centerline")

    # --- Diagnóstico (whirl/whip + críticas) ---
    diag_w = QtWidgets.QWidget(); diag_l = QtWidgets.QVBoxLayout(diag_w)
    btn_diag = QtWidgets.QPushButton("Diagnosticar (API 684)")
    diag_txt = QtWidgets.QTextEdit(); diag_txt.setReadOnly(True)
    diag_l.addWidget(btn_diag); diag_l.addWidget(diag_txt, 1)
    tabs.addTab(diag_w, "Diagnóstico")

    lbl_rpm = QtWidgets.QLabel("RPM: —"); lbl_state = QtWidgets.QLabel("detenido")
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
            lbl_rec.setText(f"● capturando · {_sess.status.duration_s:.0f}s · {_sess.status.size_mb:.1f} MB")
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
        cur = tabs.tabText(tabs.currentIndex())
        if cur == "Monitoreo":
            # estado global (estable/arranque/parada) por variación de rpm
            prev = rec_state.get("prev_rpm")
            if rpm and prev:
                d = rpm - prev
                estado_g = "Arranque" if d > 15 else ("Parada" if d < -15 else "Estable")
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
                + ("#16a34a" if estado_g == "Estable" else "#b45309"))
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
        elif cur == "Onda":
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
        elif orb_ok and cur == "Órbita":
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
            rr, am, ph = tc.bode(cb_bode.currentText())
            if len(rr) >= 2:
                bode_hint.setText("")
                c_am.setData(rr, np.asarray(am) * 2.0)      # pp aprox
                c_ph.setData(rr, np.asarray(ph))
            else:
                bode_hint.setText("Poné Modo → arranque (o parada) para llenar el Bode")
                bode_hint.setPos(float(rr[0]) if len(rr) else 3000.0, 0.0)
        elif cur == "Polar":
            rr, am, ph = tc.bode(cb_pol.currentText())
            if len(rr) >= 3:
                pol_hint.setText("")
                th = np.radians(np.asarray(ph, float)); a = np.asarray(am, float) * 2.0
                x = a * np.cos(th); y = a * np.sin(th)
                pol_curve.setData(x, y); pol_pts.setData(x, y)
            else:
                pol_curve.setData([], []); pol_pts.setData([], [])
                pol_hint.setText("Poné Modo → arranque para llenar el Polar")
        elif scl_items and cur == "Shaft Centerline":
            for it in scl_items:
                Y = snap[it["yi"]] * 1000.0 / (it["yc"].sensitivity_mv_per_eu or 1.0)
                X = snap[it["xi"]] * 1000.0 / (it["xc"].sensitivity_mv_per_eu or 1.0)
                mX, mY = float(X.mean()), float(Y.mean())          # posición media (gap, mil)
                # a marco físico por el ángulo de sonda
                cx = mX * _mo.sin(it["aX"]) + mY * _mo.sin(it["aY"])
                cy = mX * _mo.cos(it["aX"]) + mY * _mo.cos(it["aY"])
                it["cur"].setData([cx], [cy])
                if rpm:
                    bk = int(rpm // 100) * 100
                    scl_track[it["brg"]][bk] = (cx, cy)
                    pts = [scl_track[it["brg"]][r] for r in sorted(scl_track[it["brg"]])]
                    if pts:
                        it["track"].setData([p[0] for p in pts], [p[1] for p in pts])
        elif cur == "Cascada":
            rr, fr, mat = tc.cascade(cb_casc.currentText())
            for cv in casc_curves:
                cv.setData([], [])
            if len(rr) >= 2 and mat.size:
                idx = np.unique(np.linspace(0, len(rr) - 1, min(len(casc_curves), len(rr)))
                                .round().astype(int))
                span = float(rr[-1] - rr[0]) or 1.0
                pk = float(mat.max()) or 1.0
                sc = (span / max(1, len(idx))) * 1.5 / pk
                for cv, i in zip(casc_curves, idx):
                    cv.setData(fr, rr[i] + mat[i] * sc)

    def run_diag():
        rr, fr, mat = tc.cascade(cb_casc.currentText())
        if len(rr) < 3:
            diag_txt.setHtml("<i>Corré un runup (variá la velocidad) para tener datos que diagnosticar.</i>")
            return
        rb, ab, _ph = tc.bode(cb_casc.currentText())
        crit = [float(rb[i]) for i in diag.detect_criticals(np.asarray(rb, float), np.asarray(ab, float))]
        found = diag.cascade_diagnosis(rr, fr, mat, crit)
        html = [f"<h3 style='color:{NAVY}'>Auto-diagnóstico (API 684)</h3>"]
        col = {"info": ACC, "warn": "#b45309", "danger": "#b91c1c"}
        for lvl, title, detail in found:
            html.append(f"<p style='color:{col.get(lvl,'#333')}'><b>{title}</b><br>{detail}</p>")
        diag_txt.setHtml("".join(html) if found else "<i>Sin hallazgos.</i>")

    timer = QtCore.QTimer(); timer.timeout.connect(update)
    btn_diag.clicked.connect(run_diag)

    def do_start():
        # Pide nombre/consecutivo de la corrida y GRABA DESDE EL INICIO a disco
        # (así no se pierde nada, aunque guardes/subas después).
        import time as _t
        default = f"{args.machine}_{_t.strftime('%Y%m%d_%H%M%S')}"
        tag, ok = QtWidgets.QInputDialog.getText(
            win, "Iniciar corrida", "Nombre / consecutivo de la corrida:", text=default)
        if not ok:
            return
        try:
            agent.start()
        except Exception as e:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(win, "Error", f"No se pudo iniciar: {e}")
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
        lbl_state.setText("● capturando datos (desde el inicio)")
        timer.start(90)     # ~11 fps: fluido y más liviano de CPU/RAM (PCs modestas)

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
            lbl_state.setText(f"detenido · {rec.status.duration_s:.0f}s · "
                              f"{rec.status.size_mb:.1f} MB · listo para 💾 Guardar datos")
        else:
            lbl_state.setText("detenido")
        _refresh_disk()

    def _refresh_disk():
        try:
            cnt, used = local_usage(agent.instance_id)
            free = free_bytes()
            lbl_disk.setText(f"Disco: {cnt} grabación(es) · {used / 1e6:.0f} MB usados · "
                             f"{free / 1e6:.0f} MB libres")
            pend = pending_count(agent.instance_id)
            act_sync.setText(f"Subir a la nube ({pend})" if pend else "Subir a la nube")
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
            has_url = bool(os.environ.get("WM_SUPABASE_URL"))
            has_key = bool(os.environ.get("WM_SUPABASE_KEY"))
            QtWidgets.QMessageBox.warning(
                win, "Sincronizar — sin conexión a la nube",
                "No se pudo conectar a Supabase, por eso las grabaciones no suben "
                "(quedan guardadas local).\n\n"
                f"• WM_SUPABASE_URL {'OK' if has_url else 'FALTA'}\n"
                f"• WM_SUPABASE_KEY {'OK' if has_key else 'FALTA'}\n\n"
                "Pasos:\n"
                "1) Editá 'Nube__EDITAR_credenciales.bat' con tu Project URL y service_role key "
                "(Supabase → Project Settings → API).\n"
                "2) Abrí el programa con 'SIMULADOR_con_NUBE.bat' (ese carga las credenciales).\n"
                "3) Verificá que haya internet.")
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
        act_sync.setEnabled(False); act_sync.setText("Subiendo…")
        res = {}

        res["ok"] = res["fail"] = 0

        def _work():
            try:
                from core.remote_monitoring.recorder import (list_recordings, is_synced,
                                                             upload_recording)
                pend = [m for m in list_recordings(agent.instance_id) if not is_synced(m["_dir"])]
                res["total"] = len(pend)
                for k, m in enumerate(pend):
                    res["progress"] = f"Subiendo {k + 1} de {len(pend)}…"
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
                lbl_rec.setText(res.get("progress", "subiendo…"))
                QtCore.QTimer.singleShot(400, _check); return
            lbl_rec.setText("")
            act_sync.setEnabled(True); act_sync.setText("Subir a la nube"); _refresh_disk()
            ok, fail = res.get("ok", 0), res.get("fail", 0)
            if res.get("err"):
                _nice("Sincronizar", f"<b style='color:#b91c1c'>No se pudo subir</b><br>"
                      f"<span style='color:#334155'>{res['err']}</span>", QtWidgets.QMessageBox.Warning)
            elif fail == 0 and ok > 0:
                _nice("Sincronizado",
                      "<div style='font-size:17px'><b style='color:#166534'>☁ Datos sincronizados en la nube</b></div>"
                      f"<div style='color:#334155;margin-top:6px'>{ok} grabación(es) subida(s).</div>"
                      "<div style='color:#0F1E3D;margin-top:10px;font-size:14px'>Ya podés <b>iniciar el "
                      "diagnóstico en Watermelon System</b> (web → Análisis → Reprocesar). 🍉</div>")
            elif ok == 0 and fail == 0:
                _nice("Sincronizar", "<b>No hay grabaciones pendientes.</b> Todo está en la nube.")
            else:
                _nice("Sincronizar",
                      f"<b>Subidas {ok} · fallidas {fail}.</b><br>"
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
            _nice("Guardar datos",
                  "<div style='font-size:15px'><b style='color:#b45309'>⏸ Primero detené la adquisición</b></div>"
                  "<div style='color:#334155;margin-top:4px'>Dale <b>■ Detener</b> y después "
                  "<b>💾 Guardar datos</b>.</div>", QtWidgets.QMessageBox.Warning)
            return
        # 2) ¿hay una corrida detenida?
        rec = rec_state.get("session")
        if not rec:
            _nice("Guardar datos",
                  "<div style='font-size:15px'><b>No hay una corrida para guardar</b></div>"
                  "<div style='color:#334155;margin-top:4px'>Hacé: <b>▶ Iniciar</b> → medí → "
                  "<b>■ Detener</b> → <b>💾 Guardar datos</b>.</div>")
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
        _nice("Datos guardados",
              "<div style='font-size:17px'><b style='color:#166534'>✅ Datos guardados en el equipo</b></div>"
              f"<div style='color:#0F1E3D;font-family:monospace;margin-top:6px'>{rec.rec_id}<br>"
              f"{rec.status.duration_s:.0f} s · {rec.status.size_mb:.1f} MB</div>"
              "<div style='color:#b45309;margin-top:10px;font-size:14px'>⏳ <b>Pendiente de subir a la nube</b> — "
              f"apretá <b>↑ Subir pendientes ({pend})</b> cuando tengas internet.</div>")

    def set_mode_live(mode):
        """Cambia el MODO de operación en vivo (estable/arranque/parada) sobre la
        misma máquina: ajusta el perfil, rebobina el reloj del rotor y reinicia el
        capturador de transitorio para que Bode/Cascada salgan limpios."""
        nonlocal tc
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
        for cv in casc_curves:
            cv.setData([], [])
        c_am.setData([], []); c_ph.setData([], [])
        pol_curve.setData([], []); pol_pts.setData([], [])
        for _b in scl_track:                 # reiniciar la traza del shaft centerline
            scl_track[_b].clear()
        lbl_state.setText(f"● {mode}" if agent.source.is_running() else f"modo: {mode}")

    _prof0 = getattr(agent.source.config, "speed_profile", "constant")
    _inv = {v: k for k, v in MODE_TO_PROFILE.items()}
    cb_run.blockSignals(True); cb_run.setCurrentText(_inv.get(_prof0, "estable")); cb_run.blockSignals(False)
    cb_run.currentTextChanged.connect(set_mode_live)
    act_start.triggered.connect(do_start)
    act_stop.triggered.connect(do_stop)
    act_save.triggered.connect(do_save)     # barra: Guardar datos
    act_sync.triggered.connect(do_sync)     # barra: Subir a la nube

    def do_clear():
        from core.remote_monitoring.recorder import clear_recordings, pending_count, local_usage
        cnt, used = local_usage(agent.instance_id)
        if cnt == 0:
            QtWidgets.QMessageBox.information(win, "Limpiar locales", "No hay grabaciones locales.")
            return
        pend = pending_count(agent.instance_id)
        m = QtWidgets.QMessageBox(win)
        m.setWindowTitle("Limpiar grabaciones locales")
        m.setIcon(QtWidgets.QMessageBox.Warning)
        m.setText(f"Borrar {cnt} grabación(es) locales ({used/1e6:.0f} MB)?")
        m.setInformativeText(
            (f"⚠ Hay {pend} SIN subir a la nube: si las borrás, se pierden.\n\n"
             if pend else "Las que ya están en la nube quedan en la nube.\n\n")
            + "¿Continuar?")
        only = None
        if pend:
            bt_all = m.addButton("Borrar TODAS", QtWidgets.QMessageBox.DestructiveRole)
            bt_synced = m.addButton("Solo las ya subidas", QtWidgets.QMessageBox.AcceptRole)
            m.addButton("Cancelar", QtWidgets.QMessageBox.RejectRole)
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
        QtWidgets.QMessageBox.information(win, "Limpiar locales",
                                         f"Borradas {n} grabación(es) · liberados {freed/1e6:.0f} MB.")
    act_clear.triggered.connect(do_clear)   # barra: Borrar datos
    _refresh_disk()
    act_quit.triggered.connect(win.close)
    act_about.triggered.connect(lambda: QtWidgets.QMessageBox.about(
        win, "Watermelon Field", "Watermelon Field — módulo nativo de adquisición.\n"
        "Rotodinámica API 670/684 · nube integrada.\n© SIGA"))

    win.show()
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
                f.write("\n==== error al iniciar ====\n" + tb + "\n")
        except Exception:  # noqa: BLE001
            pass
        try:
            from PySide6 import QtWidgets
            _a = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
            QtWidgets.QMessageBox.critical(None, "Watermelon Field — error al iniciar",
                                           f"{tb}\n\nDetalle guardado en:\n{logp}")
        except Exception:  # noqa: BLE001
            print(tb)
        return 1


if __name__ == "__main__":
    sys.exit(_run_with_crashlog())
