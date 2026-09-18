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
    voltage_to_torque,
)
from core.torsional.sim_source import (
    TorsionalStreamConfig, SimulatedTorsionalSource, make_torsional_channels,
)
from core.torsional.analysis import (
    torque_metrics, torque_spectrum, order_amplitudes,
    keyphasor_to_rpm, order_tracking, fatigue_ranges,
)
from core.torsional.shunt_cal import REF1_100UE, REF2_500UE, verify_shunt

__version__ = "0.4.0"
DAQ_NAME = "Watermelon DAQ"
NAVY = "#0F1E3D"; ACC = "#1AAEE5"; GREEN = "#10b981"; AMBER = "#f59e0b"; RED = "#ef4444"

# Galgas comunes Vishay Micro-Measurements para telemetría de torque
# (Binsfeld "Shaft Strain Gaging Guide", p.6). bridge_idx: 0=torque, 1=axial, None=flexión.
# (part#, medida, config, ohms, piezas, bridge_idx, nota_en, nota_es)
_GAGES = [
    ("CEA-06-250US-350", "torque", "full", 350, 1, 0,
     "Torque · full-bridge in ONE piece — use one per shaft.",
     "Torque · puente completo en UNA pieza — una por eje."),
    ("CEA-06-187UV-350", "torque", "half", 350, 2, 0,
     "Torque, bending-insensitive · half-bridge · TWO pieces 180° apart → full bridge.",
     "Torque, insensible a flexión · media puente · DOS piezas a 180° → puente completo."),
    ("CEA-06-250UT-350", "axial", "half", 350, 2, 1,
     "Axial (thrust/tension) · half-bridge · TWO pieces 180° apart → full bridge.",
     "Axial (empuje/tensión) · media puente · DOS piezas a 180° → puente completo."),
    ("EA-06-250MQ-350", "bending", "half", 350, 2, None,
     "Bending, torque-insensitive · half-bridge · TWO pieces 180° apart.",
     "Flexión, insensible a torque · media puente · DOS piezas a 180°."),
]
# Materiales del eje: E en ×10⁶ psi, ν (Poisson). La galga STC 06 está
# compensada térmicamente para ACERO; otros materiales cambian la compensación.
_MATERIALS = [
    ("Steel / Acero", 30.0, 0.30),
    ("Stainless / Inoxidable", 28.0, 0.30),
    ("Aluminum / Aluminio", 10.0, 0.33),
    ("Titanium / Titanio", 16.5, 0.34),
    ("Brass / Bronce", 15.0, 0.34),
    ("Copper / Cobre", 17.0, 0.34),
    ("Custom / Personalizado", None, None),
]


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
    cb_material = QtWidgets.QComboBox(); cb_material.addItems([m[0] for m in _MATERIALS])
    sb_do = QtWidgets.QDoubleSpinBox(); sb_do.setRange(0.1, 100.0); sb_do.setDecimals(3); sb_do.setValue(3.0); sb_do.setSuffix(" in")
    sb_di = QtWidgets.QDoubleSpinBox(); sb_di.setRange(0.0, 99.0); sb_di.setDecimals(3); sb_di.setValue(0.0); sb_di.setSuffix(" in")
    sb_e = QtWidgets.QDoubleSpinBox(); sb_e.setRange(1.0, 60.0); sb_e.setDecimals(1); sb_e.setValue(30.0); sb_e.setSuffix(" ×10⁶ psi")
    sb_nu = QtWidgets.QDoubleSpinBox(); sb_nu.setRange(0.1, 0.5); sb_nu.setDecimals(2); sb_nu.setValue(0.30)
    f1.addRow(T("Material", "Material"), cb_material)
    f1.addRow(T("Outer Ø (Do)", "Ø exterior (Do)"), sb_do)
    f1.addRow(T("Inner Ø (Di)", "Ø interior (Di)"), sb_di)
    f1.addRow(T("Modulus E", "Módulo E"), sb_e)
    f1.addRow(T("Poisson ν", "Poisson ν"), sb_nu)
    form_wrap.addWidget(gb_shaft)

    def _on_material(_=0):
        name, e, nu = _MATERIALS[cb_material.currentIndex()]
        custom = (e is None)
        if not custom:
            sb_e.setValue(e); sb_nu.setValue(nu)
        sb_e.setEnabled(custom); sb_nu.setEnabled(custom)
    cb_material.currentIndexChanged.connect(_on_material)

    gb_gage = QtWidgets.QGroupBox(T("Gage / transmitter", "Galga / transmisor")); f2 = QtWidgets.QFormLayout(gb_gage)
    cb_gage = QtWidgets.QComboBox(); cb_gage.addItems([g[0] for g in _GAGES] + [T("Custom", "Personalizado")])
    lbl_gage = QtWidgets.QLabel(""); lbl_gage.setWordWrap(True); lbl_gage.setStyleSheet("color:#475569; font-size:11px;")
    sb_gf = QtWidgets.QDoubleSpinBox(); sb_gf.setRange(1.0, 3.0); sb_gf.setDecimals(3); sb_gf.setValue(2.10)
    sb_gf.setToolTip(T("Read the EXACT gage factor from the gage package/box.",
                       "Lee el factor de galga EXACTO de la caja/hoja de la galga."))
    cb_gxmt = QtWidgets.QComboBox(); cb_gxmt.addItems(["500", "1000", "2000", "4000", "8000", "16000"]); cb_gxmt.setCurrentText("4000")
    sb_rg = QtWidgets.QDoubleSpinBox(); sb_rg.setRange(100.0, 1000.0); sb_rg.setDecimals(0); sb_rg.setValue(350.0); sb_rg.setSuffix(" Ω")
    cb_bridge = QtWidgets.QComboBox(); cb_bridge.addItems([T("Torque (full bridge)", "Torque (puente completo)"),
                                                           T("Axial (full bridge)", "Axial (puente completo)"),
                                                           T("¼ bridge (1 grid)", "¼ puente (1 grilla)")])
    sb_z = QtWidgets.QDoubleSpinBox(); sb_z.setRange(0.25, 4.0); sb_z.setDecimals(4); sb_z.setValue(1.0)
    cb_units = QtWidgets.QComboBox(); cb_units.addItems(["N·m", "ft-lb"])
    f2.addRow(T("Strain gage", "Galga"), cb_gage)
    f2.addRow("", lbl_gage)
    f2.addRow(T("Gage factor GF", "Factor de galga GF"), sb_gf)
    f2.addRow(T("Transmitter gain GXMT", "Ganancia transmisor GXMT"), cb_gxmt)
    f2.addRow(T("Gage resistance RG", "Resistencia galga RG"), sb_rg)
    f2.addRow(T("Bridge", "Puente"), cb_bridge)
    f2.addRow(T("System-gain scale Z", "Escala System-gain Z"), sb_z)
    f2.addRow(T("Units", "Unidades"), cb_units)
    form_wrap.addWidget(gb_gage)

    def _on_gage(_=0):
        idx = cb_gage.currentIndex()
        if idx >= len(_GAGES):     # Custom
            lbl_gage.setText(T("Custom gage — set RG and bridge manually.",
                               "Galga personalizada — ajusta RG y puente a mano."))
            sb_rg.setEnabled(True); cb_bridge.setEnabled(True); return
        part, meas, cfgc, ohms, pcs, br, note_en, note_es = _GAGES[idx]
        sb_rg.setValue(float(ohms)); sb_rg.setEnabled(False)
        if br is not None:
            cb_bridge.setCurrentIndex(br)
        cb_bridge.setEnabled(br is None)   # bending: deja elegir (no calcula torque)
        _pcs = T(f"{pcs} piece(s) per shaft", f"{pcs} pieza(s) por eje")
        lbl_gage.setText(f"<b>{part}</b> · {_pcs}<br>{note_es if _LANG == 'es' else note_en}"
                         + ("" if br is not None else T("<br>⚠ Bending gage — not for torque scaling.",
                                                        "<br>⚠ Galga de flexión — no calcula torque.")))
    cb_gage.currentIndexChanged.connect(_on_gage)

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
    cb_preset = QtWidgets.QComboBox()
    cb_preset.addItems([T("Custom", "Personalizado"),
                        T("Reciprocating engine (1800 rpm)", "Motor recíprocante (1800 rpm)"),
                        T("Gearbox / VFD", "Caja / VFD")])
    sb_rpm = QtWidgets.QDoubleSpinBox(); sb_rpm.setRange(60, 12000); sb_rpm.setValue(1800); sb_rpm.setSuffix(" rpm")
    sb_mean = QtWidgets.QDoubleSpinBox(); sb_mean.setRange(0, 1e6); sb_mean.setValue(1000); sb_mean.setSuffix(" EU")
    sb_a1 = QtWidgets.QDoubleSpinBox(); sb_a1.setRange(0, 1e5); sb_a1.setValue(60)
    sb_a2 = QtWidgets.QDoubleSpinBox(); sb_a2.setRange(0, 1e5); sb_a2.setValue(25)
    sb_noise = QtWidgets.QDoubleSpinBox(); sb_noise.setRange(0, 1e4); sb_noise.setValue(5)
    simf.addWidget(QtWidgets.QLabel(T("Preset", "Preset"))); simf.addWidget(cb_preset)
    for lbl, w in [(T("RPM", "RPM"), sb_rpm), (T("Mean", "Media"), sb_mean),
                   ("1×", sb_a1), ("2×", sb_a2), (T("Noise", "Ruido"), sb_noise)]:
        simf.addWidget(QtWidgets.QLabel(lbl)); simf.addWidget(w)
    btn_start = QtWidgets.QPushButton(T("▶ Start", "▶ Iniciar"))
    btn_stop = QtWidgets.QPushButton(T("■ Stop", "■ Detener")); btn_stop.setEnabled(False)
    simf.addStretch(1); simf.addWidget(btn_start); simf.addWidget(btn_stop)
    live_l.addWidget(gb_sim)

    def _preset():
        """Parámetros de simulación según el preset (dict con rpm/mean/orders/…).
        Un motor recíprocante tiene firma torsional rica: medio-orden (0.5×) fuerte +
        armónicos de la velocidad (firing). Gearbox añade el orden de engrane (GMF)."""
        idx = cb_preset.currentIndex()
        if idx == 1:   # Reciprocating engine @ 1800 rpm
            return dict(rpm=1800.0, mean=1500.0, noise=10.0, res_hz=45.0, gear_teeth=0, gear_amp=0.0,
                        orders=((0.5, 220.0, 0.0), (1.0, 320.0, 0.0), (1.5, 140.0, 30.0),
                                (2.0, 260.0, 0.0), (3.0, 120.0, 0.0), (4.0, 70.0, 0.0)))
        if idx == 2:   # Gearbox / VFD
            return dict(rpm=1800.0, mean=1200.0, noise=6.0, res_hz=60.0, gear_teeth=23, gear_amp=90.0,
                        orders=((1.0, 80.0, 0.0), (2.0, 40.0, 0.0), (6.0, 45.0, 0.0)))
        return dict(rpm=sb_rpm.value(), mean=sb_mean.value(), noise=sb_noise.value(), res_hz=0.0,
                    gear_teeth=0, gear_amp=0.0,
                    orders=((1.0, sb_a1.value(), 0.0), (2.0, sb_a2.value(), 0.0)))

    def _on_preset(_=0):
        p = _preset()
        sb_rpm.setValue(p["rpm"]); sb_mean.setValue(p["mean"]); sb_noise.setValue(p["noise"])
        _custom = cb_preset.currentIndex() == 0
        for w in (sb_rpm, sb_mean, sb_a1, sb_a2, sb_noise):
            w.setEnabled(_custom)
    cb_preset.currentIndexChanged.connect(_on_preset)

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

    def _preflight():
        """Checklist de campo — el operador confirma qué colocar/ajustar en el
        equipo ANTES de capturar. Cero fallas. Debe marcar todo para continuar."""
        items = [
            T("Strain gage bonded & wired per diagram; resistance checked; no short to shaft.",
              "Galga pegada y cableada según diagrama; resistencia verificada; sin corto al eje."),
            T("TX10K-S powered — status LED solid; fresh 9 V battery; antenna attached.",
              "TX10K-S energizado — LED de estado sólido; batería 9 V fresca; antena puesta."),
            T("RX10K RF channel matches the TX10K-S channel; signal strength OK.",
              "Canal RF del RX10K = canal del TX10K-S; intensidad de señal OK."),
            T(f"Transmitter gain (GXMT) on the TX10K-S = {cb_gxmt.currentText()}.",
              f"Ganancia del transmisor (GXMT) en el TX10K-S = {cb_gxmt.currentText()}."),
            T("AutoZero applied on the RX10K with NO load on the shaft.",
              "AutoZero aplicado en el RX10K SIN carga en el eje."),
            T("Shunt calibration verified (Ref 1 / Ref 2) — see the Shunt check tab.",
              "Calibración por shunt verificada (Ref 1 / Ref 2) — ver pestaña Verificación shunt."),
            T("RX10K analog output ±10 V wired to NI 9229 (banana→BNC). Analog OR digital, not both.",
              "Salida analógica ±10 V del RX10K a la NI 9229 (banana→BNC). Analógica O digital, no ambas."),
        ]
        dlg = QtWidgets.QDialog(win)
        dlg.setWindowTitle(T("Field checklist — confirm before capture",
                             "Checklist de campo — confirmar antes de capturar"))
        v = QtWidgets.QVBoxLayout(dlg)
        hdr = QtWidgets.QLabel(T("Confirm each item is set on the equipment. Zero failures.",
                                 "Confirma cada punto en el equipo. Cero fallas."))
        hdr.setStyleSheet(f"font-weight:800; color:{NAVY};"); v.addWidget(hdr)
        checks = []
        for it in items:
            c = QtWidgets.QCheckBox(it); v.addWidget(c); checks.append(c)
        bb = QtWidgets.QDialogButtonBox()
        ok = bb.addButton(T("Confirm & continue", "Confirmar y continuar"), QtWidgets.QDialogButtonBox.AcceptRole)
        bb.addButton(T("Cancel", "Cancelar"), QtWidgets.QDialogButtonBox.RejectRole)
        ok.setEnabled(False)
        chk_all = QtWidgets.QCheckBox(T("Check all", "Marcar todo"))
        v.addWidget(chk_all); v.addWidget(bb)

        def _upd(_=0):
            ok.setEnabled(all(c.isChecked() for c in checks))
        for c in checks:
            c.stateChanged.connect(_upd)
        chk_all.stateChanged.connect(lambda s: [c.setChecked(bool(s)) for c in checks])
        bb.accepted.connect(dlg.accept); bb.rejected.connect(dlg.reject)
        return dlg.exec() == QtWidgets.QDialog.Accepted

    def _start_live():
        if not _rebuild_scaling():
            QtWidgets.QMessageBox.warning(win, "Watermelon Torsional",
                T("Fix the configuration first.", "Corrige la configuración primero."))
            return
        if not _preflight():
            return
        fs = st["fs"]
        units = st["scaling"].units
        p = _preset()
        cfg = TorsionalStreamConfig(
            sample_rate_hz=fs, rpm=p["rpm"],
            channels=make_torsional_channels(units=units),
            block_seconds=0.1, buffer_seconds=st["buf_secs"],
            mean_torque=p["mean"], orders=p["orders"],
            gear_teeth=p["gear_teeth"], gear_amp_eu=p["gear_amp"],
            torsional_res_hz=p["res_hz"], torque_noise_rms_eu=p["noise"],
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
    # TAB 5 — Campbell / interference (API 684)
    # =================================================================
    pg_cb = QtWidgets.QWidget(); cb_l = QtWidgets.QVBoxLayout(pg_cb)
    cb_l.addWidget(QtWidgets.QLabel(T(
        "Simulated run-up → torsional naturals vs excitation orders (API 684). Red × = coincidence in the operating band.",
        "Runup simulado → naturales torsionales vs órdenes de excitación (API 684). × roja = coincidencia en banda de operación.")))
    cb_ctrl = QtWidgets.QHBoxLayout()
    sb_cb_rpm = QtWidgets.QDoubleSpinBox(); sb_cb_rpm.setRange(60, 12000); sb_cb_rpm.setValue(1800); sb_cb_rpm.setSuffix(" rpm")
    cb_margin = QtWidgets.QComboBox(); cb_margin.addItems(["±10% (API 684)", "±15% (ISO 22266)", "±5%"])
    cb_ctrl.addWidget(QtWidgets.QLabel(T("Operating speed", "Velocidad de operación"))); cb_ctrl.addWidget(sb_cb_rpm)
    cb_ctrl.addWidget(QtWidgets.QLabel(T("Permissible band", "Franja permisible"))); cb_ctrl.addWidget(cb_margin)
    btn_cb = QtWidgets.QPushButton(T("▶ Run Campbell", "▶ Correr Campbell"))
    cb_ctrl.addWidget(btn_cb); cb_ctrl.addStretch(1)
    cb_l.addLayout(cb_ctrl)
    p_cb = pg.PlotWidget(); p_cb.setBackground("w"); p_cb.showGrid(x=True, y=True, alpha=0.3)
    p_cb.setLabel("bottom", "RPM"); p_cb.setLabel("left", T("frequency (Hz)", "frecuencia (Hz)"))
    p_cb.setTitle(T("Campbell / interference diagram", "Diagrama de Campbell / interferencia"))
    cb_l.addWidget(p_cb, 1)
    cb_table = QtWidgets.QTableWidget(0, 6)
    cb_table.setHorizontalHeaderLabels([T("Natural", "Natural"), T("Freq", "Frec"), T("Order", "Orden"),
                                        T("Crossing rpm", "RPM cruce"), T("Margin", "Margen"), T("Status", "Estado")])
    cb_table.horizontalHeader().setStretchLastSection(True); cb_table.setMaximumHeight(180)
    cb_l.addWidget(cb_table)

    def _run_campbell():
        if not _rebuild_scaling():
            return
        p = _preset(); fs = st["fs"]; sc = st["scaling"]; units = sc.units
        rpm = sb_cb_rpm.value(); res = p["res_hz"] or 45.0     # velocidad editable por el usuario
        _mrg = [0.10, 0.15, 0.05][cb_margin.currentIndex()]    # franja permisible por norma
        ramp = 4.0; r0 = max(300.0, rpm * 0.35); r1 = rpm * 1.6
        cfg = TorsionalStreamConfig(
            sample_rate_hz=fs, channels=make_torsional_channels(units=units), block_seconds=0.1,
            buffer_seconds=ramp + 1, speed_profile="runup", rpm_start=r0, rpm_end=r1, ramp_seconds=ramp,
            mean_torque=p["mean"], orders=p["orders"], gear_teeth=p["gear_teeth"], gear_amp_eu=p["gear_amp"],
            torsional_res_hz=res, torque_noise_rms_eu=p["noise"], scaling=sc, torque_units=units)
        src = SimulatedTorsionalSource(cfg); src.start()
        data = np.concatenate([src.read_block() for _ in range(int(ramp / cfg.block_seconds))], axis=1)
        ki = cfg.keyphasor_index(); ti = next(i for i in range(cfg.n_channels) if i != ki)
        torque = voltage_to_torque(data[ti], sc); tr, ri = keyphasor_to_rpm(data[ki], fs)
        if ri.size < 3:
            return
        tt = np.arange(torque.size) / fs; rps = np.interp(tt, tr, ri, left=ri[0], right=ri[-1])
        tracks = order_tracking(torque, fs, rps, orders=(1, 2, 3), n_segments=28)
        fgr = np.linspace(2, 200, 600); acc = np.zeros_like(fgr)
        for x in tracks:
            fa = x.order * x.rpm / 60.0; s = np.argsort(fa)
            acc += np.interp(fgr, fa[s], x.amplitude[s], left=0, right=0)
        naturals = []
        if acc.max() > 0:
            thr = 0.35 * acc.max()
            for i in range(1, len(acc) - 1):
                if acc[i] > thr and acc[i] >= acc[i - 1] and acc[i] > acc[i + 1]:
                    fv = float(fgr[i])
                    if not any(abs(fv - y) < 3 for y in naturals):
                        naturals.append(fv)
        naturals = naturals or [res]
        from core.modal.campbell import compute_crossings, SpeedBand
        rpm_max = r1 * 1.05; band = SpeedBand(rpm, _mrg * rpm, "Op")
        crossings = compute_crossings(naturals, 0.0, rpm_max, (1., 2., 3., 4., 6.), bands=[band],
                                      mode_labels=[f"TNF{i+1}" for i in range(len(naturals))])
        st["camp"] = {"naturals": naturals, "crossings": crossings, "rpm": rpm, "rpm_max": rpm_max,
                      "margin": _mrg}
        p_cb.clear()
        _ymax = (max(naturals) * 1.35) if naturals else 100.0
        # Franja PERMISIBLE (banda de operación por norma) — región sombreada ámbar.
        _reg = pg.LinearRegionItem(values=[band.low, band.high], orientation="vertical", movable=False,
                                   brush=pg.mkBrush(245, 158, 11, 45), pen=pg.mkPen(None))
        _reg.setZValue(-10); p_cb.addItem(_reg)
        _bl = pg.TextItem(T(f"Operating ±{_mrg*100:.0f}%", f"Operación ±{_mrg*100:.0f}%"),
                          color="#b45309", anchor=(0, 1))
        _bl.setPos(band.low, _ymax); p_cb.addItem(_bl)
        xr = np.linspace(0, rpm_max, 60)
        for o in (1., 2., 3., 4., 6.):
            p_cb.plot(xr, o * xr / 60.0, pen=pg.mkPen("#94a3b8", width=1, style=QtCore.Qt.DotLine))
        for fn in naturals:
            p_cb.plot([0, rpm_max], [fn, fn], pen=pg.mkPen(GREEN, width=2))
        p_cb.addItem(pg.InfiniteLine(pos=rpm, angle=90, pen=pg.mkPen(NAVY, width=2, style=QtCore.Qt.DashLine)))
        _cc = {"coincidence": RED, "near": AMBER, "clear": "#94a3b8"}
        for c in crossings:
            p_cb.addItem(pg.ScatterPlotItem([c.crossing_rpm], [c.mode_hz], symbol="x", size=14,
                                            pen=pg.mkPen(_cc[c.severity], width=3)))
        _stx = {"coincidence": T("Coincidence", "Coincidencia"), "near": T("Near", "Cercano"),
                "clear": T("Clear", "Libre")}
        cb_table.setRowCount(len(crossings))
        for r, c in enumerate(crossings):
            for cix, v in enumerate([c.mode_label, f"{c.mode_hz:.1f} Hz", f"{c.order:g}×",
                                     f"{c.crossing_rpm:.0f}", f"{c.sep_margin_pct:.0f}%", _stx[c.severity]]):
                cb_table.setItem(r, cix, QtWidgets.QTableWidgetItem(v))
    btn_cb.clicked.connect(_run_campbell)
    tabs.addTab(pg_cb, "Campbell")

    # =================================================================
    # TAB 6 — Fatigue (rainflow ASTM E1049)
    # =================================================================
    pg_ft = QtWidgets.QWidget(); ft_l = QtWidgets.QVBoxLayout(pg_ft)
    ft_l.addWidget(QtWidgets.QLabel(T(
        "Rainflow cycle counting (ASTM E1049) on a captured torque history — input for shaft fatigue / Goodman.",
        "Conteo rainflow (ASTM E1049) sobre el historial de par — entrada para fatiga / Goodman del eje.")))
    btn_ft = QtWidgets.QPushButton(T("▶ Capture & count (rainflow)", "▶ Capturar y contar (rainflow)"))
    ft_l.addWidget(btn_ft)
    ft_read = QtWidgets.QLabel("—"); ft_read.setStyleSheet(f"font-weight:800; color:{NAVY};")
    ft_l.addWidget(ft_read)
    p_ft = pg.PlotWidget(); p_ft.setBackground("w"); p_ft.showGrid(x=True, y=True, alpha=0.3)
    p_ft.setLabel("bottom", T("torque range", "rango de par")); p_ft.setLabel("left", T("cycle count", "conteo"))
    p_ft.setTitle(T("Rainflow histogram", "Histograma rainflow"))
    ft_l.addWidget(p_ft, 1)

    def _run_fatigue():
        if not _rebuild_scaling():
            return
        p = _preset(); fs = st["fs"]; sc = st["scaling"]; units = sc.units
        cfg = TorsionalStreamConfig(
            sample_rate_hz=fs, rpm=p["rpm"], channels=make_torsional_channels(units=units),
            block_seconds=0.25, buffer_seconds=8, mean_torque=p["mean"], orders=p["orders"],
            gear_teeth=p["gear_teeth"], gear_amp_eu=p["gear_amp"], torsional_res_hz=p["res_hz"],
            torque_noise_rms_eu=p["noise"], scaling=sc, torque_units=units)
        src = SimulatedTorsionalSource(cfg); src.start()
        data = np.concatenate([src.read_block() for _ in range(32)], axis=1)
        ki = cfg.keyphasor_index(); ti = next(i for i in range(cfg.n_channels) if i != ki)
        torque = voltage_to_torque(data[ti], sc)
        ranges = fatigue_ranges(torque)
        u = "N·m" if units == "nm" else "ft-lb"
        st["fat"] = {"ranges": ranges, "units": u}
        p_ft.clear()
        if ranges:
            rr = np.array([r for r, _ in ranges]); cc = np.array([c for _, c in ranges])
            nb = int(np.clip(len(rr), 8, 24)); edges = np.linspace(0, rr.max() * 1.0001, nb + 1)
            hist, _ = np.histogram(rr, bins=edges, weights=cc)
            ctr = 0.5 * (edges[:-1] + edges[1:])
            p_ft.addItem(pg.BarGraphItem(x=ctr, height=hist, width=(edges[1] - edges[0]) * 0.9, brush=NAVY))
            ft_read.setText(T(f"Total cycles: {cc.sum():,.0f}  ·  Largest range: {rr.max():,.1f} {u}",
                              f"Ciclos totales: {cc.sum():,.0f}  ·  Rango máximo: {rr.max():,.1f} {u}"))
    btn_ft.clicked.connect(_run_fatigue)
    tabs.addTab(pg_ft, "Fatigue")

    # =================================================================
    # TAB 7 — Preliminary report (PDF de campo, como el Modal)
    # =================================================================
    pg_rp = QtWidgets.QWidget(); rp_l = QtWidgets.QVBoxLayout(pg_rp)
    rp_l.addWidget(QtWidgets.QLabel(T(
        "Quick same-day field PDF: metrics + orders + Campbell + fatigue. The full SIGA report is generated on the web.",
        "PDF de campo del mismo día: métricas + órdenes + Campbell + fatiga. El reporte SIGA completo se genera en la web.")))
    rform = QtWidgets.QFormLayout()
    ed_asset = QtWidgets.QLineEdit(); ed_client = QtWidgets.QLineEdit(); ed_prep = QtWidgets.QLineEdit()
    rform.addRow(T("Asset / Tag", "Activo / Tag"), ed_asset)
    rform.addRow(T("Client", "Cliente"), ed_client)
    rform.addRow(T("Prepared by", "Realizado por"), ed_prep)
    rp_l.addLayout(rform)
    btn_rp = QtWidgets.QPushButton(T("📄 Generate preliminary report (PDF)", "📄 Generar reporte preliminar (PDF)"))
    rp_l.addWidget(btn_rp)
    rp_status = QtWidgets.QLabel(""); rp_status.setWordWrap(True); rp_l.addWidget(rp_status)
    rp_l.addStretch(1)

    def _grab_png(widget):
        try:
            pm = widget.grab()
            ba = QtCore.QByteArray(); buf = QtCore.QBuffer(ba); buf.open(QtCore.QIODevice.WriteOnly)
            pm.save(buf, "PNG"); return bytes(ba)
        except Exception:  # noqa: BLE001
            return None

    def _gen_report():
        if not _rebuild_scaling():
            return
        # asegura que Campbell y Fatiga estén corridos
        if "camp" not in st:
            _run_campbell()
        if "fat" not in st:
            _run_fatigue()
        sc = st["scaling"]; u = "N·m" if sc.units == "nm" else "ft-lb"
        p = _preset(); rpm = p["rpm"]
        # métricas de una captura estable
        cfg = TorsionalStreamConfig(sample_rate_hz=st["fs"], rpm=rpm,
            channels=make_torsional_channels(units=sc.units), block_seconds=0.25, buffer_seconds=6,
            mean_torque=p["mean"], orders=p["orders"], gear_teeth=p["gear_teeth"], gear_amp_eu=p["gear_amp"],
            torsional_res_hz=p["res_hz"], torque_noise_rms_eu=p["noise"], scaling=sc, torque_units=sc.units)
        src = SimulatedTorsionalSource(cfg); src.start()
        data = np.concatenate([src.read_block() for _ in range(24)], axis=1)
        ki = cfg.keyphasor_index(); ti = next(i for i in range(cfg.n_channels) if i != ki)
        torque = voltage_to_torque(data[ti], sc); mm = torque_metrics(torque)
        oa = order_amplitudes(torque, st["fs"], rpm, orders=(1, 2, 3, 4, 5))
        camp = st.get("camp", {}); crossings = camp.get("crossings", [])
        coincid = [c for c in crossings if c.severity == "coincidence"]
        worst = min((c.sep_margin_pct for c in crossings), default=float("inf"))
        ranges = st.get("fat", {}).get("ranges", [])
        rmax = max((r for r, _ in ranges), default=0.0)
        _rip = "∞" if mm.ripple_pct == float("inf") else f"{mm.ripple_pct:.1f}%"

        quality = [
            (T("Signal", "Señal"), "OK", T(f"Mean {mm.mean:,.0f} {u}, pp {mm.peak_to_peak:,.0f} {u}",
                                           f"Media {mm.mean:,.0f} {u}, pp {mm.peak_to_peak:,.0f} {u}")),
            (T("Separation margin (API 684)", "Margen de separación (API 684)"),
             "NO-GO" if coincid else ("REVIEW" if worst < 10 else "GO"),
             T(f"Worst {worst:.0f}% (target ≥10%)", f"Mínimo {worst:.0f}% (objetivo ≥10%)")),
        ]
        analysis = [
            T(f"Dominant order {max(range(1,6), key=lambda k: oa[float(k)][0])}× at {rpm:,.0f} rpm; ripple {_rip}.",
              f"Orden dominante {max(range(1,6), key=lambda k: oa[float(k)][0])}× a {rpm:,.0f} rpm; rizado {_rip}."),
            T(f"Torsional naturals: {', '.join(f'{x:.1f} Hz' for x in camp.get('naturals', [])) or '—'}.",
              f"Naturales torsionales: {', '.join(f'{x:.1f} Hz' for x in camp.get('naturals', [])) or '—'}."),
        ]
        findings = [T(f"{len(coincid)} order coincidence(s) in the operating band (worst margin {worst:.0f}%).",
                      f"{len(coincid)} coincidencia(s) de orden en la banda de operación (margen mínimo {worst:.0f}%).")
                    if coincid else T("No coincidences in the operating band.",
                                      "Sin coincidencias en la banda de operación."),
                    T(f"Largest rainflow torque range {rmax:,.0f} {u} (ASTM E1049).",
                      f"Mayor rango rainflow del par {rmax:,.0f} {u} (ASTM E1049).")]
        recs = [T("Confirm coincidences with an operating amplitude/phase run (API 684).",
                  "Confirmar coincidencias con corrida de amplitud/fase en operación (API 684)."),
                T("Evaluate shaft fatigue against the Goodman diagram at the gage location.",
                  "Evaluar fatiga del eje contra el diagrama de Goodman en la galga.")]
        ord_rows = [[f"{o}×", f"{o*rpm/60:.2f} Hz", f"{oa[float(o)][0]:.1f} {u}"] for o in range(1, 6)]
        sections = [
            {"title": T("Order spectrum", "Espectro de órdenes"),
             "figures": [(T("Live torque spectrum", "Espectro de par"), _grab_png(p_spec))],
             "table": {"headers": [T("Order", "Orden"), T("Freq", "Frec"), T("Amplitude", "Amplitud")], "rows": ord_rows}},
            {"title": T("Campbell / interference (API 684)", "Campbell / interferencia (API 684)"),
             "figures": [(T("Orders vs torsional naturals", "Órdenes vs naturales"), _grab_png(p_cb))]},
            {"title": T("Fatigue (rainflow)", "Fatiga (rainflow)"),
             "figures": [(T("Rainflow histogram", "Histograma rainflow"), _grab_png(p_ft))]},
        ]
        meta = {"title": T("Preliminary Torsional Report", "Reporte Torsional Preliminar"),
                "asset": ed_asset.text() or "—", "client": ed_client.text() or "—",
                "prep": ed_prep.text() or "—", "rpm": f"{rpm:,.0f}", "equip": "TorqueTrak 10K + NI 9229"}
        try:
            from core.modal.preliminary_report import build_preliminary_pdf
            _es = (_LANG == "es")
            pdf = build_preliminary_pdf(meta=meta, quality=quality, sections=sections, analysis=analysis,
                                        findings=findings, recommendations=recs,
                                        run_id=f"TOR-{ed_asset.text() or 'run'}", lang=("es" if _es else "en"))
        except Exception as exc:  # noqa: BLE001
            rp_status.setText(f"❌ {type(exc).__name__}: {exc}"); return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(win, T("Save report", "Guardar reporte"),
                                                        f"Torsional_{ed_asset.text() or 'run'}.pdf", "PDF (*.pdf)")
        if not path:
            return
        with open(path, "wb") as fh:
            fh.write(pdf)
        rp_status.setText(T(f"✅ Saved: {path}", f"✅ Guardado: {path}"))
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(path))
    btn_rp.clicked.connect(_gen_report)
    tabs.addTab(pg_rp, T("Report", "Reporte"))

    # =================================================================
    # TAB 8 — Updates (auto-actualización por red, como el Modal)
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

    _on_material(); _on_gage()      # estado inicial (galga/material por defecto)
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
