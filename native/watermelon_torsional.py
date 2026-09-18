"""
Watermelon Torsional — app de campo (native)
============================================

Field app (PySide6 + pyqtgraph), hermana de Watermelon Modal / Rotordynamics.
Analiza el par medido con el sistema de telemetría Binsfeld TorqueTrak 10K:
la salida ±10 V del receptor RX10K se lee con una tarjeta NI de voltaje DC
(9229 preferida — ±60 V/24 bit/simultáneo — o 9215) y se convierte a torque.

Espeja a `native/watermelon_modal.py` en estructura, estilo, i18n (T/EN·ES),
banner de hardware y selector de idioma. Toda la matemática vive en el núcleo
compartido `core.torsional.*` (lo mismo que consume el módulo web).

P0: geometría→escalado, torque en vivo (simulado), verificación de shunt y
runup (order tracking). Adquisición real (NIStreamSource) = paso Windows.
"""
from __future__ import annotations

import argparse
import sys
import traceback
from collections import deque
from typing import Optional

import numpy as np

try:
    from PySide6 import QtCore, QtGui, QtWidgets
    import pyqtgraph as pg
except Exception as exc:  # noqa: BLE001
    print("Missing PySide6/pyqtgraph:", exc)
    raise

from core.torsional.scaling import (
    ShaftGeometry, GageConfig, BridgeType, TorqueScaling, full_scale_torque,
)
from core.torsional.sim_source import (
    TorsionalStreamConfig, SimulatedTorsionalSource, make_torsional_channels,
)
from core.torsional.analysis import (
    torque_metrics, torque_spectrum, order_amplitudes,
    keyphasor_to_rpm, order_tracking,
)
from core.torsional.shunt_cal import REF1_100UE, REF2_500UE, verify_shunt

__version__ = "0.1.0"
DAQ_NAME = "Watermelon DAQ"
NAVY = "#0F1E3D"; ACC = "#1AAEE5"; GREEN = "#10b981"; AMBER = "#f59e0b"; RED = "#ef4444"


# =====================================================================
# i18n — bilingüe EN/ES (mismo patrón que watermelon_modal.py)
# =====================================================================
_LANG = "en"


def T(en, es=None):
    """Devuelve el texto en el idioma activo (en por defecto)."""
    return es if (es is not None and _LANG == "es") else en


def _load_lang():
    try:
        v = QtCore.QSettings("WatermelonSystem", "Torsional").value("lang", "en")
        return "es" if str(v).lower().startswith("es") else "en"
    except Exception:  # noqa: BLE001
        return "en"


def _save_lang(v):
    try:
        QtCore.QSettings("WatermelonSystem", "Torsional").setValue("lang", "es" if v == "es" else "en")
    except Exception:  # noqa: BLE001
        pass


def _stylesheet() -> str:
    return f"""
    QWidget {{ font-family: 'Segoe UI', Arial; font-size: 12px; color: {NAVY}; }}
    QMainWindow, QTabWidget::pane {{ background: #f5f8fd; }}
    QTabWidget::pane {{ border: 1px solid #dbe4f0; border-radius: 10px; top: -1px; }}
    QTabBar::tab {{ background: #e6ecf5; color: #334155; padding: 7px 11px; margin-right: 2px;
        border-top-left-radius: 8px; border-top-right-radius: 8px; font-weight: 700; font-size: 12px; }}
    QTabBar::tab:hover {{ background: #d4deee; }}
    QTabBar::tab:selected {{ background: {NAVY}; color: white; }}
    QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{ background: white;
        border: 1px solid #cbd5e1; border-radius: 6px; padding: 4px 8px; selection-background-color: {ACC}; }}
    QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{ border: 1px solid {ACC}; }}
    QSpinBox, QDoubleSpinBox {{ padding-right: 20px; }}
    QComboBox::drop-down {{ border: none; width: 20px; }}
    QPushButton {{ background: {NAVY}; color: white; border: none; font-weight: 700;
        padding: 8px 15px; border-radius: 8px; }}
    QPushButton:hover {{ background: #0e3a6b; }}
    QPushButton:pressed {{ background: #0b2e56; }}
    QPushButton:disabled {{ background: #94a3b8; }}
    QGroupBox {{ font-weight: 700; border: 1px solid #dbe4f0; border-radius: 8px; margin-top: 8px; padding-top: 8px; }}
    QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; }}
    QToolTip {{ background: {NAVY}; color: white; border: none; padding: 6px 8px; border-radius: 6px; }}
    """


def _detect_dc_channels():
    """¿Hay una NI de voltaje DC (9229/9215) conectada AHORA? Devuelve nº de canales."""
    try:
        from core.modal.acq_backend import list_available_devices
        n = 0
        for d in list_available_devices():
            pt = str(d.get("product_type", ""))
            if "9229" in pt or "9239" in pt:
                n += 4
            elif "9215" in pt:
                n += 4
        return n
    except Exception:  # noqa: BLE001
        return 0


def _wl(t):
    import html
    return html.escape(str(t or ""))


class _UpdateChecker(QtCore.QThread):
    """Consulta los Releases de GitHub en segundo plano (sin congelar la UI)."""
    found = QtCore.Signal(object)

    def __init__(self, current_version, parent=None):
        super().__init__(parent); self._ver = current_version

    def run(self):
        try:
            from core.torsional.updater import check_for_update
            info = check_for_update(self._ver)
            if info:
                self.found.emit(info)
        except Exception:  # noqa: BLE001
            pass


def _show_update_banner(win, info):
    """Aviso de actualización + botón para actualizar de una (descarga + instala)."""
    try:
        ver = info.get("version", "?")
        box = QtWidgets.QMessageBox(win)
        box.setWindowTitle("Watermelon Torsional — update available")
        box.setIcon(QtWidgets.QMessageBox.Information)
        box.setText(f"<b>A newer version is available: v{ver}</b>")
        box.setInformativeText(T(
            "Update now? The installer will be downloaded and applied over the current "
            "installation. The app will close to finish the update.",
            "¿Actualizar ahora? El instalador se descarga y se aplica sobre la instalación "
            "actual. La app se cerrará para terminar.") + "\n\n" + _wl((info.get("notes", "") or "")[:400]))
        b_now = box.addButton(T("Update now", "Actualizar ahora"), QtWidgets.QMessageBox.AcceptRole)
        box.addButton(T("Later", "Después"), QtWidgets.QMessageBox.RejectRole)
        box.exec()
        if box.clickedButton() is not b_now:
            return
        from core.torsional import updater
        url = info.get("setup_url") or info.get("zip_url")
        if not url:
            if info.get("html_url"):
                QtGui.QDesktopServices.openUrl(QtCore.QUrl(info["html_url"]))
            return
        dlg = QtWidgets.QProgressDialog(T("Downloading update…", "Descargando actualización…"),
                                        "Cancel", 0, 100, win)
        dlg.setWindowTitle("Updating"); dlg.setModal(True); dlg.setMinimumDuration(0); dlg.show()

        def _prog(fr):
            dlg.setValue(int(fr * 100)); QtWidgets.QApplication.processEvents()
        path = updater.download_file(url, on_progress=_prog)
        dlg.close()
        if not path:
            QtWidgets.QMessageBox.warning(win, "Update",
                T("Could not download the update.", "No se pudo descargar la actualización."))
            return
        if path.lower().endswith("setup.exe"):
            updater.launch_installer(path)
            QtWidgets.QApplication.quit()      # el instalador reemplaza y relanza
        else:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(path))
    except Exception:  # noqa: BLE001
        pass


def build_app(simulated: bool = True):
    global _LANG
    _LANG = _load_lang()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(_stylesheet())
    pg.setConfigOptions(antialias=True)
    win = QtWidgets.QMainWindow()
    win.setWindowTitle(f"Watermelon Torsional v{__version__}")
    _scr = app.primaryScreen()
    _av = _scr.availableGeometry() if _scr else None
    if _av is not None:
        _w = min(1320, _av.width() - 40); _h = min(850, _av.height() - 80)
        win.resize(max(900, _w), max(560, _h)); win.setMinimumSize(820, 520)
        win.move(_av.left() + (_av.width() - win.width()) // 2,
                 _av.top() + (_av.height() - win.height()) // 2)
    else:
        win.resize(1320, 850)

    # Estado compartido de la sesión
    st = {
        "scaling": None,      # TorqueScaling actual
        "gage": None,         # GageConfig actual (para shunt)
        "source": None,       # SimulatedTorsionalSource
        "fs": 2560.0,
        "buf_torque": deque(maxlen=1),
        "buf_kph": deque(maxlen=1),
        "buf_secs": 6.0,
        "running": False,
    }

    # ---- Toolbar: marca + versión + idioma + banner de hardware ----
    tb = win.addToolBar("main"); tb.setMovable(False)
    tb.setStyleSheet(f"QToolBar {{ background: {NAVY}; padding: 7px 14px; spacing:0px; }}")
    brand = QtWidgets.QLabel()
    brand.setText(
        "<span style='color:#ffffff; font-weight:800; letter-spacing:2.5px; font-size:16px;'>WATERMELON</span>"
        "<span style='color:#1AAEE5; font-weight:800; letter-spacing:2.5px; font-size:16px;'>&nbsp;TORSIONAL</span>"
        "<span style='color:#5b6b86; font-weight:600; letter-spacing:1px; font-size:9.5px;'>&nbsp;&nbsp;™</span>")
    brand.setTextFormat(QtCore.Qt.RichText); tb.addWidget(brand)
    ver_lbl = QtWidgets.QLabel(f"v{__version__}")
    ver_lbl.setStyleSheet("color:#cbd5e1; background:#1e3a5f; border-radius:9px; padding:2px 10px; "
                          "font-weight:700; font-size:11px; margin-left:12px;")
    tb.addWidget(ver_lbl)
    spc = QtWidgets.QWidget(); spc.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
    tb.addWidget(spc)

    def _switch_lang(new):
        if new == _LANG:
            return
        _save_lang(new)
        QtWidgets.QMessageBox.information(win, "Watermelon Torsional",
            T("Language set. The app will restart to apply it.",
              "Idioma cambiado. La app se reiniciará para aplicarlo."))
        try:
            QtCore.QProcess.startDetached(QtWidgets.QApplication.applicationFilePath(), sys.argv[1:])
        except Exception:  # noqa: BLE001
            pass
        app.quit()

    lang_wrap = QtWidgets.QWidget()
    lang_wrap.setStyleSheet("QWidget{background:#16233b; border:1px solid #2a3a57; border-radius:9px;}")
    _lh = QtWidgets.QHBoxLayout(lang_wrap); _lh.setContentsMargins(3, 2, 3, 2); _lh.setSpacing(2)
    _glob = QtWidgets.QLabel("\U0001F310"); _glob.setStyleSheet("background:transparent; border:none; font-size:12px;")
    _lh.addWidget(_glob)
    for _code in ("EN", "ES"):
        _b = QtWidgets.QToolButton(); _b.setText(_code); _b.setCheckable(True)
        _b.setChecked(_code.lower() == _LANG); _b.setCursor(QtCore.Qt.PointingHandCursor)
        _b.setStyleSheet(
            "QToolButton{background:transparent; color:#8ea0bd; border:none; border-radius:6px;"
            "padding:3px 12px; font-weight:800; font-size:11px; letter-spacing:1px;}"
            "QToolButton:hover{color:#dbe6f5;}"
            "QToolButton:checked{background:#1AAEE5; color:#08243a;}")
        _b.clicked.connect(lambda _=0, c=_code.lower(): _switch_lang(c))
        _lh.addWidget(_b)
    tb.addWidget(lang_wrap)
    _sp2 = QtWidgets.QWidget(); _sp2.setFixedWidth(12); tb.addWidget(_sp2)

    mode_lbl = QtWidgets.QLabel("")
    mode_lbl.setToolTip(T("Auto-detected NI voltage module (9229/9215) for the RX10K analog output.",
                          "Módulo NI de voltaje (9229/9215) autodetectado para la salida analógica del RX10K."))
    tb.addWidget(mode_lbl)

    def _refresh_hw_banner():
        n = _detect_dc_channels()
        if n > 0:
            mode_lbl.setText(f"● LIVE — {DAQ_NAME} · {n} ch   ")
            mode_lbl.setStyleSheet("color:#34d399; font-weight:700;")
        else:
            mode_lbl.setText(T("● SIMULATED — no DAQ connected   ",
                               "● SIMULADO — sin DAQ conectado   "))
            mode_lbl.setStyleSheet("color:#fbbf24; font-weight:700;")
    _refresh_hw_banner()
    _hw_timer = QtCore.QTimer(win); _hw_timer.setInterval(5000)
    _hw_timer.timeout.connect(_refresh_hw_banner); _hw_timer.start()

    tabs = QtWidgets.QTabWidget(); win.setCentralWidget(tabs)

    # =================================================================
    # Helpers de escalado
    # =================================================================
    def _current_units():
        return "nm" if cb_units.currentIndex() == 0 else "ftlb"

    def _bridge():
        return [BridgeType.TORQUE, BridgeType.AXIAL, BridgeType.QUARTER][cb_bridge.currentIndex()]

    def _rebuild_scaling():
        """Reconstruye TorqueScaling/GageConfig desde los campos de Configuración."""
        try:
            shaft = ShaftGeometry(
                outer_diameter_in=sb_do.value(),
                inner_diameter_in=sb_di.value(),
                modulus_psi=sb_e.value() * 1e6,
                poisson=sb_nu.value(),
            )
            gage = GageConfig(
                gage_factor=sb_gf.value(),
                transmitter_gain=int(cb_gxmt.currentText()),
                gage_resistance_ohm=sb_rg.value(),
                bridge=_bridge(),
            )
            units = _current_units()
            z = sb_z.value()
            sc = TorqueScaling.from_geometry(shaft, gage, units=units, scale_factor_z=z)
            st["scaling"] = sc; st["gage"] = gage
            u = "N·m" if units == "nm" else "ft-lb"
            lbl_tfs.setText(
                T(f"Full-scale torque (10 V): <b>{sc.full_scale_torque:,.1f} {u}</b>"
                  f"  ·  <b>{sc.eu_per_volt:,.2f} {u}/V</b>",
                  f"Par de fondo de escala (10 V): <b>{sc.full_scale_torque:,.1f} {u}</b>"
                  f"  ·  <b>{sc.eu_per_volt:,.2f} {u}/V</b>"))
            lbl_tfs.setStyleSheet(f"color:{NAVY}; font-size:13px;")
            return True
        except Exception as exc:  # noqa: BLE001
            lbl_tfs.setText(f"⚠ {exc}"); lbl_tfs.setStyleSheet(f"color:{RED};")
            st["scaling"] = None
            return False

    # =================================================================
    # TAB 1 — Configuration (geometría del eje + galga → escalado)
    # =================================================================
    pg_cfg = QtWidgets.QWidget(); cfg_l = QtWidgets.QVBoxLayout(pg_cfg)
    intro = QtWidgets.QLabel(T(
        "Enter the shaft and strain-gage parameters. Watermelon computes the volts→torque "
        "calibration (TorqueTrak 10K, Appendix B).",
        "Ingresa los parámetros del eje y la galga. Watermelon calcula la calibración "
        "volts→torque (TorqueTrak 10K, Appendix B)."))
    intro.setWordWrap(True); cfg_l.addWidget(intro)

    form_wrap = QtWidgets.QHBoxLayout(); cfg_l.addLayout(form_wrap)

    gb_shaft = QtWidgets.QGroupBox(T("Shaft", "Eje")); f1 = QtWidgets.QFormLayout(gb_shaft)
    sb_do = QtWidgets.QDoubleSpinBox(); sb_do.setRange(0.1, 100.0); sb_do.setDecimals(3); sb_do.setValue(3.0); sb_do.setSuffix(" in")
    sb_di = QtWidgets.QDoubleSpinBox(); sb_di.setRange(0.0, 99.0); sb_di.setDecimals(3); sb_di.setValue(0.0); sb_di.setSuffix(" in")
    sb_e = QtWidgets.QDoubleSpinBox(); sb_e.setRange(1.0, 60.0); sb_e.setDecimals(1); sb_e.setValue(30.0); sb_e.setSuffix(" ×10⁶ psi")
    sb_nu = QtWidgets.QDoubleSpinBox(); sb_nu.setRange(0.1, 0.5); sb_nu.setDecimals(2); sb_nu.setValue(0.30)
    f1.addRow(T("Outer Ø (Do)", "Ø exterior (Do)"), sb_do)
    f1.addRow(T("Inner Ø (Di)", "Ø interior (Di)"), sb_di)
    f1.addRow(T("Modulus E", "Módulo E"), sb_e)
    f1.addRow(T("Poisson ν", "Poisson ν"), sb_nu)
    form_wrap.addWidget(gb_shaft)

    gb_gage = QtWidgets.QGroupBox(T("Gage / transmitter", "Galga / transmisor")); f2 = QtWidgets.QFormLayout(gb_gage)
    sb_gf = QtWidgets.QDoubleSpinBox(); sb_gf.setRange(1.0, 3.0); sb_gf.setDecimals(3); sb_gf.setValue(2.0)
    cb_gxmt = QtWidgets.QComboBox(); cb_gxmt.addItems(["500", "1000", "2000", "4000", "8000", "16000"]); cb_gxmt.setCurrentText("4000")
    sb_rg = QtWidgets.QDoubleSpinBox(); sb_rg.setRange(100.0, 1000.0); sb_rg.setDecimals(0); sb_rg.setValue(350.0); sb_rg.setSuffix(" Ω")
    cb_bridge = QtWidgets.QComboBox(); cb_bridge.addItems([T("Torque (full, 4)", "Torque (completo, 4)"),
                                                           T("Axial (full, 2.6)", "Axial (completo, 2.6)"),
                                                           T("¼ bridge (1)", "¼ puente (1)")])
    sb_z = QtWidgets.QDoubleSpinBox(); sb_z.setRange(0.25, 4.0); sb_z.setDecimals(4); sb_z.setValue(1.0)
    cb_units = QtWidgets.QComboBox(); cb_units.addItems(["N·m", "ft-lb"])
    f2.addRow(T("Gage factor GF", "Factor de galga GF"), sb_gf)
    f2.addRow(T("Transmitter gain GXMT", "Ganancia transmisor GXMT"), cb_gxmt)
    f2.addRow(T("Gage resistance RG", "Resistencia galga RG"), sb_rg)
    f2.addRow(T("Bridge", "Puente"), cb_bridge)
    f2.addRow(T("System-gain scale Z", "Escala System-gain Z"), sb_z)
    f2.addRow(T("Units", "Unidades"), cb_units)
    form_wrap.addWidget(gb_gage)

    lbl_tfs = QtWidgets.QLabel(""); lbl_tfs.setWordWrap(True)
    lbl_tfs.setStyleSheet(f"background:white; border:1px solid #dbe4f0; border-radius:8px; padding:10px;")
    cfg_l.addWidget(lbl_tfs)
    cfg_l.addStretch(1)
    for _w in (sb_do, sb_di, sb_e, sb_nu, sb_gf, sb_rg, sb_z):
        _w.valueChanged.connect(lambda _=0: _rebuild_scaling())
    cb_gxmt.currentIndexChanged.connect(lambda _=0: _rebuild_scaling())
    cb_bridge.currentIndexChanged.connect(lambda _=0: _rebuild_scaling())
    cb_units.currentIndexChanged.connect(lambda _=0: _rebuild_scaling())
    tabs.addTab(pg_cfg, T("Configuration", "Configuración"))

    # =================================================================
    # TAB 2 — Live torque
    # =================================================================
    pg_live = QtWidgets.QWidget(); live_l = QtWidgets.QVBoxLayout(pg_live)

    # Controles del simulador
    gb_sim = QtWidgets.QGroupBox(T("Simulated signal", "Señal simulada")); simf = QtWidgets.QHBoxLayout(gb_sim)
    sb_rpm = QtWidgets.QDoubleSpinBox(); sb_rpm.setRange(60, 12000); sb_rpm.setValue(1800); sb_rpm.setSuffix(" rpm")
    sb_mean = QtWidgets.QDoubleSpinBox(); sb_mean.setRange(0, 1e6); sb_mean.setValue(1000); sb_mean.setSuffix(" EU")
    sb_a1 = QtWidgets.QDoubleSpinBox(); sb_a1.setRange(0, 1e5); sb_a1.setValue(60)
    sb_a2 = QtWidgets.QDoubleSpinBox(); sb_a2.setRange(0, 1e5); sb_a2.setValue(25)
    sb_noise = QtWidgets.QDoubleSpinBox(); sb_noise.setRange(0, 1e4); sb_noise.setValue(5)
    for lbl, w in [(T("RPM", "RPM"), sb_rpm), (T("Mean", "Media"), sb_mean),
                   ("1×", sb_a1), ("2×", sb_a2), (T("Noise", "Ruido"), sb_noise)]:
        simf.addWidget(QtWidgets.QLabel(lbl)); simf.addWidget(w)
    btn_start = QtWidgets.QPushButton(T("▶ Start", "▶ Iniciar"))
    btn_stop = QtWidgets.QPushButton(T("■ Stop", "■ Detener")); btn_stop.setEnabled(False)
    simf.addStretch(1); simf.addWidget(btn_start); simf.addWidget(btn_stop)
    live_l.addWidget(gb_sim)

    # Lecturas grandes
    read_row = QtWidgets.QHBoxLayout(); live_l.addLayout(read_row)
    def _readout(title):
        box = QtWidgets.QGroupBox(title); bl = QtWidgets.QVBoxLayout(box)
        val = QtWidgets.QLabel("—"); val.setAlignment(QtCore.Qt.AlignCenter)
        val.setStyleSheet(f"font-size:22px; font-weight:800; color:{NAVY};")
        bl.addWidget(val); return box, val
    b1, v_mean = _readout(T("Mean torque", "Par medio"))
    b2, v_pp = _readout(T("Peak-peak", "Pico-pico"))
    b3, v_ripple = _readout(T("Ripple %", "Rizado %"))
    b4, v_rpm = _readout("RPM")
    for b in (b1, b2, b3, b4):
        read_row.addWidget(b)

    plots_row = QtWidgets.QHBoxLayout(); live_l.addLayout(plots_row, 1)
    p_time = pg.PlotWidget(); p_time.setBackground("w"); p_time.showGrid(x=True, y=True, alpha=0.3)
    p_time.setLabel("bottom", T("time", "tiempo"), "s"); p_time.setTitle(T("Torque vs time", "Par vs tiempo"))
    curve_t = p_time.plot(pen=pg.mkPen(ACC, width=2))
    p_spec = pg.PlotWidget(); p_spec.setBackground("w"); p_spec.showGrid(x=True, y=True, alpha=0.3)
    p_spec.setLabel("bottom", T("frequency", "frecuencia"), "Hz"); p_spec.setTitle(T("Torque spectrum", "Espectro de par"))
    curve_s = p_spec.plot(pen=pg.mkPen(NAVY, width=2))
    plots_row.addWidget(p_time, 1); plots_row.addWidget(p_spec, 1)

    # Barras de órdenes
    p_ord = pg.PlotWidget(); p_ord.setBackground("w"); p_ord.setMaximumHeight(150)
    p_ord.setTitle(T("Orders (× running speed)", "Órdenes (× velocidad)")); p_ord.showGrid(y=True, alpha=0.3)
    bar_ord = pg.BarGraphItem(x=[1, 2, 3, 4, 5], height=[0] * 5, width=0.6, brush=ACC)
    p_ord.addItem(bar_ord); p_ord.getAxis("bottom").setTicks([[(i, f"{i}×") for i in range(1, 6)]])
    live_l.addWidget(p_ord)

    live_timer = QtCore.QTimer(win); live_timer.setInterval(120)

    def _start_live():
        if not _rebuild_scaling():
            QtWidgets.QMessageBox.warning(win, "Watermelon Torsional",
                T("Fix the configuration first.", "Corrige la configuración primero."))
            return
        fs = st["fs"]
        units = st["scaling"].units
        cfg = TorsionalStreamConfig(
            sample_rate_hz=fs, rpm=sb_rpm.value(),
            channels=make_torsional_channels(units=units),
            block_seconds=0.1, buffer_seconds=st["buf_secs"],
            mean_torque=sb_mean.value(),
            orders=((1.0, sb_a1.value(), 0.0), (2.0, sb_a2.value(), 0.0)),
            torque_noise_rms_eu=sb_noise.value(),
            scaling=st["scaling"], torque_units=units,
        )
        src = SimulatedTorsionalSource(cfg); src.start()
        st["source"] = src
        maxlen = int(st["buf_secs"] * fs)
        st["buf_torque"] = deque(maxlen=maxlen); st["buf_kph"] = deque(maxlen=maxlen)
        st["running"] = True
        btn_start.setEnabled(False); btn_stop.setEnabled(True)
        live_timer.start()

    def _stop_live():
        live_timer.stop(); st["running"] = False
        if st["source"] is not None:
            st["source"].stop()
        btn_start.setEnabled(True); btn_stop.setEnabled(False)

    def _tick():
        src = st["source"]; sc = st["scaling"]
        if src is None or sc is None:
            return
        block = src.read_block()          # (n_channels, n)
        cfg = src.config
        kph_i = cfg.keyphasor_index()
        torque_i = next(i for i in range(cfg.n_channels) if i != kph_i)
        from core.torsional.scaling import voltage_to_torque
        torque_eu = voltage_to_torque(block[torque_i], sc)
        st["buf_torque"].extend(torque_eu.tolist())
        st["buf_kph"].extend(block[kph_i].tolist())

        arr = np.asarray(st["buf_torque"]); kph = np.asarray(st["buf_kph"])
        fs = st["fs"]
        if arr.size < 32:
            return
        t = np.arange(arr.size) / fs
        curve_t.setData(t, arr)

        m = torque_metrics(arr)
        u = "N·m" if sc.units == "nm" else "ft-lb"
        v_mean.setText(f"{m.mean:,.1f} {u}")
        v_pp.setText(f"{m.peak_to_peak:,.1f} {u}")
        v_ripple.setText("∞" if m.ripple_pct == float("inf") else f"{m.ripple_pct:.1f} %")

        freqs, amp = torque_spectrum(arr, fs)
        mask = freqs <= 600.0                      # techo de banda del equipo (500 Hz)
        curve_s.setData(freqs[mask], amp[mask])

        # RPM del keyphasor
        _, rpm = keyphasor_to_rpm(kph, fs)
        rpm_now = float(np.median(rpm)) if rpm.size else sb_rpm.value()
        v_rpm.setText(f"{rpm_now:,.0f}")
        oa = order_amplitudes(arr, fs, rpm_now, orders=(1, 2, 3, 4, 5))
        bar_ord.setOpts(height=[oa[float(o)][0] for o in range(1, 6)])

    btn_start.clicked.connect(_start_live)
    btn_stop.clicked.connect(_stop_live)
    live_timer.timeout.connect(_tick)
    tabs.addTab(pg_live, T("Live torque", "Torque en vivo"))

    # =================================================================
    # TAB 3 — Shunt check (verificación de calibración)
    # =================================================================
    pg_sh = QtWidgets.QWidget(); sh_l = QtWidgets.QVBoxLayout(pg_sh)
    sh_intro = QtWidgets.QLabel(T(
        "Trigger a reference shunt on the TX10K-S with the RM10K remote. Watermelon computes the "
        "expected voltage and checks the reading against it (field cal, TorqueTrak 10K step 12).",
        "Activa un shunt de referencia en el TX10K-S con el control RM10K. Watermelon calcula el "
        "voltaje esperado y contrasta la lectura (cal de campo, TorqueTrak 10K paso 12)."))
    sh_intro.setWordWrap(True); sh_l.addWidget(sh_intro)

    sh_btns = QtWidgets.QHBoxLayout(); sh_l.addLayout(sh_btns)
    btn_ref1 = QtWidgets.QPushButton(T("Apply Ref 1 (100 µε)", "Aplicar Ref 1 (100 µε)"))
    btn_ref2 = QtWidgets.QPushButton(T("Apply Ref 2 (500 µε)", "Aplicar Ref 2 (500 µε)"))
    sh_btns.addWidget(btn_ref1); sh_btns.addWidget(btn_ref2); sh_btns.addStretch(1)
    sh_result = QtWidgets.QTextBrowser(); sh_result.setMaximumHeight(220); sh_l.addWidget(sh_result)
    sh_l.addStretch(1)

    def _run_shunt(ref):
        if not _rebuild_scaling() or st["gage"] is None:
            QtWidgets.QMessageBox.warning(win, "Watermelon Torsional",
                T("Fix the configuration first.", "Corrige la configuración primero."))
            return
        sc = st["scaling"]; gage = st["gage"]
        # En simulado: el RX10K reproduce el shunt con un pequeño error realista.
        from core.torsional.shunt_cal import expected_shunt_voltage, full_scale_strain_torque
        eps_fs = full_scale_strain_torque(gage)
        expected = expected_shunt_voltage(ref.simulated_ue, eps_fs, sc.scale_factor_z)
        rng = np.random.default_rng()
        measured = expected * (1.0 + rng.normal(0, 0.002)) + rng.normal(0, 0.003)
        chk = verify_shunt(measured, ref, gage, scale_factor_z=sc.scale_factor_z)
        color = GREEN if chk.passed else AMBER
        status = T("PASS", "OK") if chk.passed else T("OUT OF TOL", "FUERA DE TOL")
        sh_result.setHtml(
            f"<div style='font-size:13px'>"
            f"<b>{ref.name}</b> — {ref.simulated_ue:.0f} µε<br>"
            f"{T('Expected','Esperado')}: <b>{chk.expected_v:.4f} V</b> &nbsp; · &nbsp; "
            f"{T('Measured','Medido')}: <b>{chk.measured_v:.4f} V</b><br>"
            f"{T('Error','Error')}: <b>{chk.error_pct:+.3f} %FS</b> &nbsp; "
            f"<span style='color:{color}; font-weight:800'>● {status}</span><br>"
            f"{T('Effective Z revealed by shunt','Z efectivo del shunt')}: <b>{chk.suggested_z:.4f}</b> "
            f"({T('current','actual')}: {sc.scale_factor_z:.4f})</div>")

    btn_ref1.clicked.connect(lambda: _run_shunt(REF1_100UE))
    btn_ref2.clicked.connect(lambda: _run_shunt(REF2_500UE))
    tabs.addTab(pg_sh, T("Shunt check", "Verificación shunt"))

    # =================================================================
    # TAB 4 — Runup (order tracking / Campbell de torque)
    # =================================================================
    pg_ru = QtWidgets.QWidget(); ru_l = QtWidgets.QVBoxLayout(pg_ru)
    ru_ctrl = QtWidgets.QHBoxLayout(); ru_l.addLayout(ru_ctrl)
    sb_r0 = QtWidgets.QDoubleSpinBox(); sb_r0.setRange(60, 12000); sb_r0.setValue(600); sb_r0.setSuffix(" rpm")
    sb_r1 = QtWidgets.QDoubleSpinBox(); sb_r1.setRange(60, 12000); sb_r1.setValue(3600); sb_r1.setSuffix(" rpm")
    sb_res = QtWidgets.QDoubleSpinBox(); sb_res.setRange(0, 500); sb_res.setValue(30); sb_res.setSuffix(" Hz")
    for lbl, w in [(T("Start", "Inicio"), sb_r0), (T("End", "Fin"), sb_r1),
                   (T("Torsional natural", "Natural torsional"), sb_res)]:
        ru_ctrl.addWidget(QtWidgets.QLabel(lbl)); ru_ctrl.addWidget(w)
    btn_run = QtWidgets.QPushButton(T("▶ Run simulated run-up", "▶ Correr runup simulado"))
    ru_ctrl.addStretch(1); ru_ctrl.addWidget(btn_run)
    p_camp = pg.PlotWidget(); p_camp.setBackground("w"); p_camp.showGrid(x=True, y=True, alpha=0.3)
    p_camp.setLabel("bottom", "RPM"); p_camp.setLabel("left", T("order amplitude", "amplitud de orden"))
    p_camp.setTitle(T("Order tracking — amplitude vs speed", "Order tracking — amplitud vs velocidad"))
    p_camp.addLegend()
    ru_l.addWidget(p_camp, 1)

    def _run_runup():
        if not _rebuild_scaling():
            return
        sc = st["scaling"]; fs = st["fs"]; units = sc.units
        ramp = 4.0
        cfg = TorsionalStreamConfig(
            sample_rate_hz=fs, channels=make_torsional_channels(units=units),
            block_seconds=0.1, buffer_seconds=ramp + 1.0,
            speed_profile="runup", rpm_start=sb_r0.value(), rpm_end=sb_r1.value(), ramp_seconds=ramp,
            mean_torque=500.0, orders=((1.0, 100.0, 0.0), (2.0, 45.0, 0.0)),
            torsional_res_hz=sb_res.value(), torque_noise_rms_eu=4.0,
            scaling=sc, torque_units=units,
        )
        src = SimulatedTorsionalSource(cfg); src.start()
        n_blocks = int(ramp / cfg.block_seconds)
        data = np.concatenate([src.read_block() for _ in range(n_blocks)], axis=1)
        kph_i = cfg.keyphasor_index(); torque_i = next(i for i in range(cfg.n_channels) if i != kph_i)
        from core.torsional.scaling import voltage_to_torque
        torque = voltage_to_torque(data[torque_i], sc)
        t_rev, rpm_inst = keyphasor_to_rpm(data[kph_i], fs)
        if rpm_inst.size < 3:
            return
        tt = np.arange(torque.size) / fs
        rpm_ps = np.interp(tt, t_rev, rpm_inst, left=rpm_inst[0], right=rpm_inst[-1])
        tracks = order_tracking(torque, fs, rpm_ps, orders=(1, 2, 3), n_segments=28)
        p_camp.clear()
        for tr, col in zip(tracks, (ACC, GREEN, AMBER)):
            p_camp.plot(tr.rpm, tr.amplitude, pen=pg.mkPen(col, width=2),
                        name=f"{int(tr.order)}×")
        # línea de la resonancia (rpm donde 1× = f_res)
        if sb_res.value() > 0:
            rc = sb_res.value() * 60.0
            p_camp.addItem(pg.InfiniteLine(pos=rc, angle=90,
                           pen=pg.mkPen(RED, width=1, style=QtCore.Qt.DashLine)))

    btn_run.clicked.connect(_run_runup)
    tabs.addTab(pg_ru, T("Run-up", "Runup"))

    # =================================================================
    # TAB 5 — Updates (auto-actualización por red, como el Modal)
    # =================================================================
    pg_upd = QtWidgets.QWidget(); ul = QtWidgets.QVBoxLayout(pg_upd)
    _cur = QtWidgets.QLabel(T(f"Installed version: <b>v{__version__}</b>",
                              f"Versión instalada: <b>v{__version__}</b>"))
    _cur.setTextFormat(QtCore.Qt.RichText)
    _ustatus = QtWidgets.QLabel(T("Press <b>Check for updates</b> to see if a newer version is available.",
                                  "Pulsa <b>Buscar actualizaciones</b> para ver si hay una versión más nueva."))
    _ustatus.setWordWrap(True); _ustatus.setTextFormat(QtCore.Qt.RichText); _ustatus.setStyleSheet("color:#64748b;")
    _unotes = QtWidgets.QTextBrowser(); _unotes.setMaximumHeight(180); _unotes.hide()
    _ubrow = QtWidgets.QPushButton(T("🔍  Check for updates", "🔍  Buscar actualizaciones"))
    _ubgo = QtWidgets.QPushButton(T("⬇  Update now", "⬇  Actualizar ahora"))
    _ubgo.setStyleSheet(f"QPushButton{{background:{GREEN};}}QPushButton:hover{{background:#12833a;}}")
    _ubgo.hide()
    urow = QtWidgets.QHBoxLayout(); urow.addWidget(_ubrow); urow.addWidget(_ubgo); urow.addStretch(1)
    _ufoot = QtWidgets.QLabel(T(
        "Updates download and install automatically over the network; the app restarts when done. "
        "No files to send — the field PC just needs internet.",
        "Las actualizaciones se descargan e instalan automáticamente por red; la app se reinicia al terminar. "
        "Sin enviar archivos — el PC de campo solo necesita internet."))
    _ufoot.setWordWrap(True); _ufoot.setStyleSheet("color:#94a3b8;")
    for w in (_cur, _ustatus, _unotes):
        ul.addWidget(w)
    ul.addLayout(urow); ul.addWidget(_ufoot); ul.addStretch(1)
    st["_pending_update"] = None

    def _upd_check():
        _ubrow.setEnabled(False); _ubrow.setText(T("🔍  Checking…", "🔍  Buscando…"))
        QtWidgets.QApplication.processEvents()
        try:
            from core.torsional.updater import diagnose
            info, msg = diagnose(__version__)
        except Exception as exc:  # noqa: BLE001
            info, msg = None, f"Error: {type(exc).__name__}: {exc}"
        _ubrow.setEnabled(True); _ubrow.setText(T("🔍  Check for updates", "🔍  Buscar actualizaciones"))
        st["_pending_update"] = info
        if info:
            _ustatus.setText(T(f"✅ <b style='color:{GREEN}'>New version available: v{info['version']}</b>",
                               f"✅ <b style='color:{GREEN}'>Nueva versión disponible: v{info['version']}</b>"))
            _unotes.setPlainText((info.get("notes") or "").strip()); _unotes.show(); _ubgo.show()
        else:
            _ustatus.setText(_wl(msg).replace("\n", "<br>")); _unotes.hide(); _ubgo.hide()

    def _upd_go():
        if st.get("_pending_update"):
            _show_update_banner(win, st["_pending_update"])
    _ubrow.clicked.connect(_upd_check); _ubgo.clicked.connect(_upd_go)
    tabs.addTab(pg_upd, T("Updates", "Actualizaciones"))

    _rebuild_scaling()
    return app, win


def main(argv=None):
    ap = argparse.ArgumentParser(description="Watermelon Torsional — TorqueTrak 10K (native)")
    ap.add_argument("--sim", action="store_true", default=True)
    args = ap.parse_args(argv)
    try:
        app, win = build_app(simulated=True); win.showMaximized()
        # Auto-actualizador: al conectar a internet, avisa si hay versión nueva (por red).
        try:
            _chk = _UpdateChecker(__version__, win)
            _chk.found.connect(lambda info: _show_update_banner(win, info))
            win._update_checker = _chk           # mantener referencia viva
            QtCore.QTimer.singleShot(3000, _chk.start)
        except Exception:  # noqa: BLE001
            pass
        sys.exit(app.exec())
    except Exception:  # noqa: BLE001
        err = traceback.format_exc()
        try:
            with open("watermelon_torsional_error.log", "w", encoding="utf-8") as fh:
                fh.write(err)
            _a = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
            QtWidgets.QMessageBox.critical(None, "Watermelon Torsional — startup error", err[-1500:])
        except Exception:  # noqa: BLE001
            print(err)
        sys.exit(1)


if __name__ == "__main__":
    main()
