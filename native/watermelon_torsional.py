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
    rainflow_cycles, shaft_torsional_fatigue,
)
from core.torsional.shunt_cal import REF1_100UE, REF2_500UE, verify_shunt
from core.torsional.ni_source import (
    KeyphasorSensor, NITorsionalConfig, NITorsionalSource,
    nidaqmx_available, rpm_from_keyphasor,
)
from core.torsional.monitor import TorsionalMonitor

__version__ = "0.12.5"
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
# (nombre, E [×10⁶ psi], ν, Sut [ksi], Se'/Sut) — eje de máquina típico por material.
# Se'/Sut: acero ≈0.50 · hierro dúctil ≈0.45 · fundición gris ≈0.40 (frágil).
_MATERIALS = [
    ("Steel 4140 Q&T / Acero 4140 templado", 30.0, 0.30, 140.0, 0.50),  # el más común en ejes
    ("Steel 4140 annealed / 4140 recocido", 30.0, 0.30, 95.0, 0.50),
    ("Steel 1045 / Acero 1045", 30.0, 0.30, 90.0, 0.50),                # acero medio C
    ("Ductile iron / Hierro dúctil (80-55-06)", 24.5, 0.28, 80.0, 0.45),
    ("Gray cast iron / Fundición gris (class 40)", 15.0, 0.26, 40.0, 0.40),
    ("Stainless / Inoxidable (410/17-4)", 28.0, 0.30, 95.0, 0.50),
    ("Aluminum / Aluminio (6061-T6)", 10.0, 0.33, 45.0, 0.45),
    ("Titanium / Titanio (Ti-6Al-4V)", 16.5, 0.34, 130.0, 0.45),
    ("Brass / Bronce", 15.0, 0.34, 50.0, 0.45),
    ("Copper / Cobre", 17.0, 0.34, 32.0, 0.45),
    ("Custom / Personalizado", None, None, None, None),
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


def _run_trace_tags():
    """(account, hostname) para trazabilidad de la corrida subida: quién (cuenta
    de la licencia) y desde qué PC. Robusto si el módulo de licencia no está."""
    try:
        from core.modal import licensing as _lic
        _acc = str((_lic.local_license_status() or {}).get("account") or "")
        return _acc, _lic.machine_label()
    except Exception:  # noqa: BLE001
        try:
            import socket as _s, getpass as _g
            return "", f"{_s.gethostname()} / {_g.getuser()}"
        except Exception:  # noqa: BLE001
            return "", ""


def _conn_ip_geo():
    """(ip, geo) PÚBLICOS del PC de campo al subir — trazabilidad (aviso por correo).
    geo = 'Ciudad, Región, PAÍS' aprox. por IP. Best-effort; si falla → ('', '')."""
    import json as _j, urllib.request as _u
    def _ssl():
        import ssl as _s
        try:
            import certifi as _c
            return _s.create_default_context(cafile=_c.where())
        except Exception:  # noqa: BLE001
            try:
                return _s.create_default_context()
            except Exception:  # noqa: BLE001
                return None
    def _get(url, t=3.5):
        req = _u.Request(url, headers={"User-Agent": "WatermelonTorsional"})
        with _u.urlopen(req, timeout=t, context=_ssl()) as r:
            return _j.loads(r.read().decode("utf-8", "replace"))
    try:
        d = _get("https://ipapi.co/json/")
        ip = str(d.get("ip") or "")
        parts = [d.get("city"), d.get("region"), d.get("country_name") or d.get("country")]
        geo = ", ".join(str(p) for p in parts if p)
        if ip:
            return ip, geo
    except Exception:  # noqa: BLE001
        pass
    try:
        return str(_get("https://api.ipify.org?format=json").get("ip") or ""), ""
    except Exception:  # noqa: BLE001
        return "", ""


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
        "acq_mode": "sim",                # "sim" | "ni" (NI 9229 hardware)
        "kph_sensor": KeyphasorSensor.simulated(),
        "ni_cfg": {"device": "cDAQ1Mod1", "torque_ai": 0, "kph_ai": 1},
        "_sim_widgets": [],               # se ocultan en modo NI (par real de la tarjeta)
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
        # Banner HONESTO: refleja el MODO elegido y la fuente realmente activa,
        # no solo si hay una tarjeta presente.
        mode = st.get("acq_mode", "sim")
        running_ni = st.get("running") and isinstance(st.get("source"), NITorsionalSource)
        if mode == "ni":
            n = _detect_dc_channels()
            if running_ni:
                mode_lbl.setText(f"● LIVE — {DAQ_NAME} (NI 9229)   ")
                mode_lbl.setStyleSheet("color:#34d399; font-weight:700;")
            elif n > 0:
                mode_lbl.setText(T(f"● NI 9229 READY — {n} ch detected   ",
                                   f"● NI 9229 LISTA — {n} ch detectados   "))
                mode_lbl.setStyleSheet("color:#34d399; font-weight:700;")
            else:
                mode_lbl.setText(T("● NI 9229 selected — no module detected   ",
                                   "● NI 9229 elegida — sin módulo detectado   "))
                mode_lbl.setStyleSheet("color:#fbbf24; font-weight:700;")
        else:
            mode_lbl.setText(T("● SIMULATED — simulated signal   ",
                               "● SIMULADO — señal simulada   "))
            mode_lbl.setStyleSheet("color:#fbbf24; font-weight:700;")
    _refresh_hw_banner()
    _hw_timer = QtCore.QTimer(win); _hw_timer.setInterval(5000)
    _hw_timer.timeout.connect(_refresh_hw_banner); _hw_timer.start()

    tabs = QtWidgets.QTabWidget(); win.setCentralWidget(tabs)

    # =================================================================
    # TAB 0 — Setup (identificación de la máquina — alimenta el reporte)
    # =================================================================
    pg_set = QtWidgets.QWidget(); set_l = QtWidgets.QVBoxLayout(pg_set)
    set_l.addWidget(QtWidgets.QLabel(T(
        "Machine identification — used in the report and in the saved/uploaded run.",
        "Identificación de la máquina — se usa en el reporte y en la corrida guardada/subida.")))
    gb_set = QtWidgets.QGroupBox(T("Machine / asset", "Máquina / activo")); fset = QtWidgets.QFormLayout(gb_set)
    ed_machine = QtWidgets.QLineEdit(); ed_tag = QtWidgets.QLineEdit(); ed_mtype = QtWidgets.QLineEdit()
    ed_setclient = QtWidgets.QLineEdit(); ed_setloc = QtWidgets.QLineEdit()
    sb_plate_rpm = QtWidgets.QDoubleSpinBox(); sb_plate_rpm.setRange(0, 30000); sb_plate_rpm.setValue(1800); sb_plate_rpm.setSuffix(" rpm")
    ed_operator = QtWidgets.QLineEdit()
    ed_approved = QtWidgets.QLineEdit()
    ed_approved.setPlaceholderText(T("Reviewing engineer (signs the report)",
                                     "Ingeniero que revisa (firma el reporte)"))
    fset.addRow(T("Machine", "Máquina"), ed_machine)
    fset.addRow(T("Tag", "Tag"), ed_tag)
    fset.addRow(T("Machine type", "Tipo de máquina"), ed_mtype)
    fset.addRow(T("Client", "Cliente"), ed_setclient)
    fset.addRow(T("Location", "Ubicación"), ed_setloc)
    fset.addRow(T("Nameplate RPM", "RPM de placa"), sb_plate_rpm)
    fset.addRow(T("Operator", "Operador"), ed_operator)
    fset.addRow(T("Approved by", "Aprobado por"), ed_approved)
    set_l.addWidget(gb_set)      # FIX v0.7.1: se había perdido — sin esto no aparecían los campos
    btn_savesetup = QtWidgets.QPushButton(T("💾 Save setup", "💾 Guardar setup"))
    btn_savesetup.setStyleSheet(f"QPushButton{{background:{GREEN};}}QPushButton:hover{{background:#12833a;}}")
    lbl_setsaved = QtWidgets.QLabel(""); lbl_setsaved.setStyleSheet("color:#16a34a; font-weight:700;")
    _srow = QtWidgets.QHBoxLayout(); _srow.addWidget(btn_savesetup); _srow.addWidget(lbl_setsaved); _srow.addStretch(1)
    set_l.addLayout(_srow); set_l.addStretch(1)
    tabs.addTab(pg_set, "Setup")

    _SET = QtCore.QSettings("WatermelonSystem", "TorsionalSetup")

    def _setup_dict():
        return {"machine": ed_machine.text(), "tag": ed_tag.text(), "type": ed_mtype.text(),
                "client": ed_setclient.text(), "location": ed_setloc.text(),
                "nameplate_rpm": sb_plate_rpm.value(), "operator": ed_operator.text(),
                "approved_by": ed_approved.text()}
    st["setup_fn"] = _setup_dict

    def _save_setup():
        for k, v in _setup_dict().items():
            _SET.setValue(k, v)
        lbl_setsaved.setText(T("✅ Setup saved (persists across restarts).",
                               "✅ Setup guardado (persiste al reiniciar)."))

    def _load_setup():
        ed_machine.setText(str(_SET.value("machine", "") or ""))
        ed_tag.setText(str(_SET.value("tag", "") or ""))
        ed_mtype.setText(str(_SET.value("type", "") or ""))
        ed_setclient.setText(str(_SET.value("client", "") or ""))
        ed_setloc.setText(str(_SET.value("location", "") or ""))
        try:
            sb_plate_rpm.setValue(float(_SET.value("nameplate_rpm", 1800) or 1800))
        except Exception:  # noqa: BLE001
            pass
        ed_operator.setText(str(_SET.value("operator", "") or ""))
        ed_approved.setText(str(_SET.value("approved_by", "") or ""))
        if ed_machine.text():
            lbl_setsaved.setText(T("Loaded saved setup.", "Setup guardado cargado."))
    btn_savesetup.clicked.connect(_save_setup)
    _load_setup()

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
    sb_sut = QtWidgets.QDoubleSpinBox(); sb_sut.setRange(10.0, 400.0); sb_sut.setDecimals(0); sb_sut.setValue(90.0); sb_sut.setSuffix(" ksi")
    sb_sut.setToolTip(T("Ultimate tensile strength of the shaft material — drives fatigue life.",
                        "Resistencia última a tracción del material del eje — define la vida a fatiga."))
    f1.addRow(T("Material", "Material"), cb_material)
    f1.addRow(T("Outer Ø (Do)", "Ø exterior (Do)"), sb_do)
    f1.addRow(T("Inner Ø (Di)", "Ø interior (Di)"), sb_di)
    f1.addRow(T("Modulus E", "Módulo E"), sb_e)
    f1.addRow(T("Poisson ν", "Poisson ν"), sb_nu)
    f1.addRow(T("Ultimate Sut", "Última Sut"), sb_sut)
    form_wrap.addWidget(gb_shaft)

    def _on_material(_=0):
        name, e, nu, sut, endr = _MATERIALS[cb_material.currentIndex()]
        custom = (e is None)
        if not custom:
            sb_e.setValue(e); sb_nu.setValue(nu); sb_sut.setValue(sut)
        st["endurance_ratio"] = endr if endr is not None else 0.50
        sb_e.setEnabled(custom); sb_nu.setEnabled(custom); sb_sut.setEnabled(custom)
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

    # --- Adquisición + sensor de keyphasor (P1: NI 9229 real) ---
    gb_acq = QtWidgets.QGroupBox(T("Acquisition", "Adquisición")); f3 = QtWidgets.QFormLayout(gb_acq)
    cb_acq_mode = QtWidgets.QComboBox()
    cb_acq_mode.addItems([T("Simulated signal", "Señal simulada"), "NI 9229 (hardware)"])
    ed_ni_device = QtWidgets.QLineEdit("cDAQ1Mod1")
    ed_ni_device.setToolTip(T("NI-DAQmx module name of the 9229 (e.g. cDAQ1Mod1).",
                              "Nombre del módulo 9229 en NI-DAQmx (p.ej. cDAQ1Mod1)."))
    sb_torque_ai = QtWidgets.QSpinBox(); sb_torque_ai.setRange(0, 3); sb_torque_ai.setValue(0)
    sb_kph_ai = QtWidgets.QSpinBox(); sb_kph_ai.setRange(0, 3); sb_kph_ai.setValue(1)
    cb_kph_sensor = QtWidgets.QComboBox()
    cb_kph_sensor.addItems([T("Simulated keyphasor", "Keyphasor simulado"),
                            "Bently 3300 XL 8mm + Proximitor",
                            T("Photo-tach (reflective tape)", "Foto-tacómetro (cinta reflectiva)")])
    sb_ppr = QtWidgets.QSpinBox(); sb_ppr.setRange(1, 60); sb_ppr.setValue(1)
    sb_ppr.setToolTip(T("Keyways (proximity) or reflective strips (photo-tach) per revolution.",
                        "Keyways (proximidad) o cintas reflectivas (foto-tacómetro) por vuelta."))
    lbl_kph = QtWidgets.QLabel(""); lbl_kph.setWordWrap(True); lbl_kph.setStyleSheet("color:#475569; font-size:11px;")
    _ai_row = QtWidgets.QWidget(); _ail = QtWidgets.QHBoxLayout(_ai_row)
    _ail.setContentsMargins(0, 0, 0, 0); _ail.addWidget(sb_torque_ai); _ail.addWidget(sb_kph_ai); _ail.addStretch(1)
    f3.addRow(T("Mode", "Modo"), cb_acq_mode)
    f3.addRow("NI 9229 device", ed_ni_device)
    f3.addRow(T("Torque AI / Keyphasor AI", "AI par / AI keyphasor"), _ai_row)
    f3.addRow(T("Keyphasor sensor", "Sensor keyphasor"), cb_kph_sensor)
    f3.addRow(T("Pulses per rev", "Pulsos por vuelta"), sb_ppr)
    f3.addRow("", lbl_kph)
    form_wrap.addWidget(gb_acq)

    def _kph_sensor():
        """Construye el KeyphasorSensor elegido (+ ppr) y lo guarda en el estado."""
        idx = cb_kph_sensor.currentIndex(); ppr = sb_ppr.value()
        if idx == 1:
            s = KeyphasorSensor.bently_3300xl_8mm(keyways=ppr)
        elif idx == 2:
            s = KeyphasorSensor.phototach_reflective(strips=ppr)
        else:
            s = KeyphasorSensor.simulated()
        st["kph_sensor"] = s
        lbl_kph.setText(f"{s.label} · edge={s.edge} · {s.pulses_per_rev}/rev<br>{s.note}")
        return s

    def _on_acq_mode(_=0):
        st["acq_mode"] = "ni" if cb_acq_mode.currentIndex() == 1 else "sim"
        _is_ni = st["acq_mode"] == "ni"
        for _w in (ed_ni_device, sb_torque_ai, sb_kph_ai):
            _w.setEnabled(_is_ni)
        st["ni_cfg"] = {"device": ed_ni_device.text() or "cDAQ1Mod1",
                        "torque_ai": sb_torque_ai.value(), "kph_ai": sb_kph_ai.value()}
        # En modo NI el par y las RPM vienen de la tarjeta → oculta los campos del
        # simulador (menos manos del operador = menos fallas).
        for _w in st.get("_sim_widgets", []):
            try: _w.setVisible(not _is_ni)
            except Exception: pass  # noqa: BLE001
        _refresh_hw_banner()
    cb_acq_mode.currentIndexChanged.connect(_on_acq_mode)
    cb_kph_sensor.currentIndexChanged.connect(lambda _=0: _kph_sensor())
    sb_ppr.valueChanged.connect(lambda _=0: _kph_sensor())
    ed_ni_device.editingFinished.connect(_on_acq_mode)
    sb_torque_ai.valueChanged.connect(lambda _=0: _on_acq_mode())
    sb_kph_ai.valueChanged.connect(lambda _=0: _on_acq_mode())
    _kph_sensor(); _on_acq_mode()      # estado inicial (sim; NI deshabilitado)

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
    btn_savecfg = QtWidgets.QPushButton(T("💾 Save configuration", "💾 Guardar configuración"))
    btn_savecfg.setStyleSheet(f"QPushButton{{background:{GREEN};}}QPushButton:hover{{background:#12833a;}}")
    lbl_cfgsaved = QtWidgets.QLabel(""); lbl_cfgsaved.setStyleSheet("color:#16a34a; font-weight:700;")
    _crow = QtWidgets.QHBoxLayout(); _crow.addWidget(btn_savecfg); _crow.addWidget(lbl_cfgsaved); _crow.addStretch(1)
    cfg_l.addLayout(_crow)
    cfg_l.addStretch(1)
    for _w in (sb_do, sb_di, sb_e, sb_nu, sb_gf, sb_rg, sb_z):
        _w.valueChanged.connect(lambda _=0: _rebuild_scaling())
    cb_gxmt.currentIndexChanged.connect(lambda _=0: _rebuild_scaling())
    cb_bridge.currentIndexChanged.connect(lambda _=0: _rebuild_scaling())
    cb_units.currentIndexChanged.connect(lambda _=0: _rebuild_scaling())

    _CFG = QtCore.QSettings("WatermelonSystem", "TorsionalConfig")

    def _save_config():
        for k, w in (("do", sb_do), ("di", sb_di), ("e", sb_e), ("nu", sb_nu),
                     ("sut", sb_sut), ("gf", sb_gf), ("rg", sb_rg), ("z", sb_z)):
            _CFG.setValue(k, w.value())
        _CFG.setValue("gxmt", cb_gxmt.currentText())
        _CFG.setValue("bridge", cb_bridge.currentIndex())
        _CFG.setValue("units", cb_units.currentIndex())
        _CFG.setValue("material", cb_material.currentIndex())
        _CFG.setValue("gage", cb_gage.currentIndex())
        _CFG.setValue("acq_mode", cb_acq_mode.currentIndex())
        _CFG.setValue("ni_device", ed_ni_device.text())
        _CFG.setValue("torque_ai", sb_torque_ai.value())
        _CFG.setValue("kph_ai", sb_kph_ai.value())
        _CFG.setValue("kph_sensor", cb_kph_sensor.currentIndex())
        _CFG.setValue("ppr", sb_ppr.value())
        lbl_cfgsaved.setText(T("✅ Configuration saved.", "✅ Configuración guardada."))

    def _load_config():
        if _CFG.value("gage") is None:
            return
        try:
            cb_gage.setCurrentIndex(int(_CFG.value("gage", 0)))
            cb_material.setCurrentIndex(int(_CFG.value("material", 0)))
            for k, w in (("do", sb_do), ("di", sb_di), ("e", sb_e), ("nu", sb_nu),
                         ("gf", sb_gf), ("rg", sb_rg), ("z", sb_z)):
                w.setValue(float(_CFG.value(k, w.value())))
            cb_gxmt.setCurrentText(str(_CFG.value("gxmt", "4000")))
            cb_bridge.setCurrentIndex(int(_CFG.value("bridge", 0)))
            cb_units.setCurrentIndex(int(_CFG.value("units", 0)))
            ed_ni_device.setText(str(_CFG.value("ni_device", "cDAQ1Mod1") or "cDAQ1Mod1"))
            sb_torque_ai.setValue(int(_CFG.value("torque_ai", 0)))
            sb_kph_ai.setValue(int(_CFG.value("kph_ai", 1)))
            cb_kph_sensor.setCurrentIndex(int(_CFG.value("kph_sensor", 0)))
            sb_ppr.setValue(int(_CFG.value("ppr", 1)))
            cb_acq_mode.setCurrentIndex(int(_CFG.value("acq_mode", 0)))
            _kph_sensor(); _on_acq_mode()
            lbl_cfgsaved.setText(T("Loaded saved configuration.", "Configuración guardada cargada."))
        except Exception:  # noqa: BLE001
            pass
    btn_savecfg.clicked.connect(_save_config)
    st["load_config_fn"] = _load_config
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
    st["_sim_widgets"].append(gb_sim)      # panel de señal simulada → oculto con NI 9229

    # Fila de DATA: guardar cruda en el PC + subir a la nube (como el Modal)
    datarow = QtWidgets.QHBoxLayout()
    btn_save = QtWidgets.QPushButton(T("💾 Save run locally (raw)", "💾 Guardar corrida local (cruda)"))
    btn_upload = QtWidgets.QPushButton(T("☁ Upload current to cloud", "☁ Subir actual a la nube"))
    btn_upload.setStyleSheet(f"QPushButton{{background:{ACC};color:#08243a;}}QPushButton:hover{{background:#149bcf;}}")
    btn_loadup = QtWidgets.QPushButton(T("📂 Upload a saved run", "📂 Subir un guardado"))
    lbl_data = QtWidgets.QLabel(""); lbl_data.setStyleSheet("color:#475569;")
    datarow.addWidget(QtWidgets.QLabel(T("Data:", "Data:"))); datarow.addWidget(btn_save)
    datarow.addWidget(btn_upload); datarow.addWidget(btn_loadup); datarow.addWidget(lbl_data); datarow.addStretch(1)
    live_l.addLayout(datarow)

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

    _axlbl = {"color": "#334155", "font-size": "10pt"}      # estilo de etiqueta de eje
    plots_row = QtWidgets.QHBoxLayout(); live_l.addLayout(plots_row, 1)
    p_time = pg.PlotWidget(); p_time.setBackground("w"); p_time.showGrid(x=True, y=True, alpha=0.3)
    p_time.setLabel("bottom", T("time", "tiempo"), "s", **_axlbl)
    p_time.setLabel("left", T("torque", "par"), "N·m", **_axlbl)   # unidad se actualiza en vivo
    p_time.setTitle(T("Torque vs time", "Par vs tiempo"))
    curve_t = p_time.plot(pen=pg.mkPen(ACC, width=2))
    p_spec = pg.PlotWidget(); p_spec.setBackground("w"); p_spec.showGrid(x=True, y=True, alpha=0.3)
    p_spec.setLabel("bottom", T("frequency", "frecuencia"), "Hz", **_axlbl)
    p_spec.setLabel("left", T("torque (pp)", "par (pp)"), "N·m", **_axlbl)   # pp por norma (API 670/ISO 7919)
    p_spec.setTitle(T("Torque spectrum", "Espectro de par"))
    curve_s = p_spec.plot(pen=pg.mkPen(NAVY, width=2))
    plots_row.addWidget(p_time, 1); plots_row.addWidget(p_spec, 1)

    # Barras de órdenes — cada orden = armónico de la velocidad (1× desbalance/torsión,
    # 2× desalineación, 3×+ engrane/álabes). Muestra qué armónico domina el rizado del par.
    p_ord = pg.PlotWidget(); p_ord.setBackground("w"); p_ord.setMaximumHeight(170)
    p_ord.setTitle(T("Order amplitudes (× running speed) — torque per harmonic",
                     "Amplitud por orden (× velocidad) — par por armónico"))
    p_ord.showGrid(y=True, alpha=0.25)
    p_ord.setLabel("left", T("torque (pp)", "par (pp)"), "N·m", **_axlbl)
    _ord_colors = ["#1AAEE5", "#16a34a", "#f59e0b", "#a855f7", "#ef4444"]
    bar_ord = pg.BarGraphItem(x=[1, 2, 3, 4, 5], height=[0] * 5, width=0.62,
                              brushes=[pg.mkBrush(c) for c in _ord_colors], pen=pg.mkPen("#0b1220", width=0.4))
    p_ord.addItem(bar_ord)
    p_ord.getAxis("bottom").setTicks([[(i, f"{i}×") for i in range(1, 6)]])
    p_ord.getViewBox().setLimits(yMin=0)          # amplitud nunca negativa
    _ord_labels = []                               # etiquetas de valor encima de cada barra
    for _i in range(5):
        _tx = pg.TextItem("", color="#334155", anchor=(0.5, 1.0)); p_ord.addItem(_tx); _ord_labels.append(_tx)
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
        if st.get("acq_mode") == "ni":
            # Adquisición REAL NI 9229. NUNCA cae a simulado en silencio: si el
            # driver/hardware no está, avisa y aborta.
            _nc = st.get("ni_cfg", {})
            ncfg = NITorsionalConfig(
                device=_nc.get("device", "cDAQ1Mod1"),
                torque_ai=int(_nc.get("torque_ai", 0)), kph_ai=int(_nc.get("kph_ai", 1)),
                sample_rate_hz=fs, block_seconds=0.1, voltage_range=10.0,
                keyphasor=st.get("kph_sensor") or KeyphasorSensor.phototach_reflective())
            src = NITorsionalSource(ncfg)
            try:
                src.start()
            except RuntimeError as exc:
                QtWidgets.QMessageBox.critical(win, "Watermelon Torsional",
                    T(f"NI 9229 acquisition failed:\n{exc}\n\nSwitch to Simulated mode or connect the DAQ.",
                      f"Falló la adquisición NI 9229:\n{exc}\n\nCambia a modo Simulado o conecta la tarjeta."))
                return
        else:
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
        st["buf_volts"] = deque(maxlen=maxlen)      # data CRUDA (voltios del RX10K)
        st["running"] = True
        btn_start.setEnabled(False); btn_stop.setEnabled(True)
        live_timer.start(); _refresh_hw_banner()

    def _stop_live():
        live_timer.stop(); st["running"] = False
        if st["source"] is not None:
            st["source"].stop()
        btn_start.setEnabled(True); btn_stop.setEnabled(False)
        _refresh_hw_banner()

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
        st.setdefault("buf_volts", deque(maxlen=len(st["buf_torque"]) or 1)).extend(block[torque_i].tolist())

        arr = np.asarray(st["buf_torque"]); kph = np.asarray(st["buf_kph"])
        fs = st["fs"]
        if arr.size < 32:
            return
        t = np.arange(arr.size) / fs
        curve_t.setData(t, arr)

        m = torque_metrics(arr)
        u = "N·m" if sc.units == "nm" else "ft-lb"
        # Eje X arranca en 0; unidad del eje Y = unidad de par actual.
        p_time.setXRange(0.0, float(t[-1]) if t.size else 1.0, padding=0.0)
        p_time.setLabel("left", T("torque", "par"), u, **_axlbl)
        v_mean.setText(f"{m.mean:,.1f} {u}")
        v_pp.setText(f"{m.peak_to_peak:,.1f} {u}")
        v_ripple.setText("∞" if m.ripple_pct == float("inf") else f"{m.ripple_pct:.1f} %")

        freqs, amp = torque_spectrum(arr, fs)
        mask = freqs <= 600.0                      # techo de banda del equipo (500 Hz)
        curve_s.setData(freqs[mask], amp[mask] * 2.0)   # 0-pk → pico-pico (norma)
        p_spec.setXRange(0.0, 600.0, padding=0.0)  # frecuencia arranca en 0
        p_spec.setLabel("left", T("torque (pp)", "par (pp)"), u, **_axlbl)

        # RPM del keyphasor (flanco/umbral/ppr del sensor elegido: proximidad o foto-tacómetro)
        _sensor = st.get("kph_sensor") or KeyphasorSensor.simulated()
        _, rpm = rpm_from_keyphasor(kph, fs, _sensor)
        rpm_now = float(np.median(rpm)) if rpm.size else sb_rpm.value()
        v_rpm.setText(f"{rpm_now:,.0f}")
        oa = order_amplitudes(arr, fs, rpm_now, orders=(1, 2, 3, 4, 5))
        _oh = [oa[float(o)][0] * 2.0 for o in range(1, 6)]     # 0-pk → pp (norma)
        bar_ord.setOpts(height=_oh)
        p_ord.setLabel("left", T("torque (pp)", "par (pp)"), u, **_axlbl)
        _omax = max(_oh) or 1.0
        for _i, _v in enumerate(_oh):            # etiqueta de valor encima de cada barra
            _ord_labels[_i].setText(f"{_v:,.0f}")
            _ord_labels[_i].setPos(_i + 1, _v + 0.03 * _omax)

    btn_start.clicked.connect(_start_live)
    btn_stop.clicked.connect(_stop_live)
    live_timer.timeout.connect(_tick)

    def _runs_dir():
        import os
        d = os.path.join(os.path.expanduser("~"), "WatermelonTorsional", "runs")
        os.makedirs(d, exist_ok=True); return d

    def _collect_run():
        """Arma el dict de la corrida (setup + config + arrays crudos)."""
        if not st.get("buf_volts") or len(st["buf_volts"]) < 32:
            return None
        volts = np.asarray(st["buf_volts"], dtype=np.float32)
        torque = np.asarray(st["buf_torque"], dtype=np.float32)
        kph = np.asarray(st["buf_kph"], dtype=np.float32)
        sc = st.get("scaling")
        setup = st["setup_fn"]() if st.get("setup_fn") else {}
        name = setup.get("machine") or setup.get("tag") or "Torsional run"
        meta = {"setup": setup, "fs": st["fs"], "units": (sc.units if sc else "nm"),
                "eu_per_volt": (sc.eu_per_volt if sc else None),
                "n_samples": int(volts.size), "app_version": __version__}
        return {"name": name, "meta": meta, "volts": volts, "torque": torque, "kph": kph}

    def _save_run_local():
        run = _collect_run()
        if run is None:
            QtWidgets.QMessageBox.warning(win, "Watermelon Torsional",
                T("No captured data yet. Press Start to capture first.",
                  "Aún no hay data capturada. Pulsa Start para capturar.")); return None
        import os, json
        from datetime import datetime
        from core.torsional.cloud import _slug
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder = os.path.join(_runs_dir(), f"{ts}_{_slug(run['name'])}")
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "run.json"), "w", encoding="utf-8") as f:
            json.dump(run["meta"], f, ensure_ascii=False, indent=2)
        # data.npz CRUDA y completa — la web la lee directo ('volts'/'kph'/'fs').
        np.savez_compressed(os.path.join(folder, "data.npz"),
                            volts=run["volts"], torque=run["torque"], kph=run["kph"],
                            fs=np.float32(st["fs"]))
        lbl_data.setText(T(f"✅ Saved: {folder}", f"✅ Guardado: {folder}"))
        return folder, run

    def _upload_run():
        res = _save_run_local()
        if not res:
            return
        folder, run = res
        lbl_data.setText(T("☁ Uploading…", "☁ Subiendo…")); QtWidgets.QApplication.processEvents()
        try:
            from core.torsional import cloud
            import socket
            setup = run["meta"]["setup"]
            rid, tstamp = cloud.new_run_id(run["name"])
            # Sube AMBOS canales (voltaje de par + keyphasor) → la web puede hacer
            # todo, incluido run-up/Campbell desde la nube.
            _raw2 = np.column_stack([run["volts"], run["kph"]]).astype(np.float32)
            raw = cloud.upload_raw(rid, _raw2, st["fs"], channels=["Torque_V", "KPH"])
            payload = dict(run["meta"]); payload["raw_ref"] = raw if raw.get("ok") else None
            _acc, _host = _run_trace_tags()          # cuenta (licencia) + PC
            _ip, _geo = _conn_ip_geo()               # IP pública + ubicación aprox.
            r = cloud.save_run(run["name"], payload, run_id=rid, ts=tstamp, account=_acc,
                               client=setup.get("client", ""), tag=setup.get("tag", ""),
                               hostname=_host or socket.gethostname(), ip=_ip, geo=_geo)
        except Exception as exc:  # noqa: BLE001
            r = {"ok": False, "reason": f"{type(exc).__name__}: {exc}"}
        if r.get("ok"):
            lbl_data.setText(T(f"☁ Uploaded to cloud (id {r.get('id','')}).",
                               f"☁ Subida a la nube (id {r.get('id','')})."))
        else:
            lbl_data.setText(T(f"⚠ Could not upload ({r.get('reason','offline')}). Saved locally is available.",
                               f"⚠ No se pudo subir ({r.get('reason','offline')}). El guardado local queda disponible."))
    def _upload_saved_run():
        """Offline-first: sube una corrida YA guardada en el PC (data.npz + run.json)."""
        import os, json, socket
        fp, _ = QtWidgets.QFileDialog.getOpenFileName(
            win, T("Open saved run (data.npz)", "Abrir corrida guardada (data.npz)"),
            _runs_dir(), "NPZ (*.npz)")
        if not fp:
            return
        lbl_data.setText(T("☁ Uploading saved run…", "☁ Subiendo corrida guardada…")); QtWidgets.QApplication.processEvents()
        try:
            z = np.load(fp)
            volts = np.asarray(z["volts"], dtype=np.float32) if "volts" in z else np.asarray(z["torque"], np.float32)
            kph = np.asarray(z["kph"], dtype=np.float32) if "kph" in z else np.zeros_like(volts)
            fs = float(z["fs"]) if "fs" in z else st["fs"]
            meta = {}
            _mj = os.path.join(os.path.dirname(fp), "run.json")
            if os.path.exists(_mj):
                with open(_mj, "r", encoding="utf-8") as fh:
                    meta = json.load(fh)
            setup = meta.get("setup", {}) if isinstance(meta.get("setup"), dict) else {}
            name = setup.get("machine") or setup.get("tag") or "Torsional run"
            from core.torsional import cloud
            rid, tstamp = cloud.new_run_id(name)
            raw2 = np.column_stack([volts, kph]).astype(np.float32)
            raw = cloud.upload_raw(rid, raw2, fs, channels=["Torque_V", "KPH"])
            payload = dict(meta); payload["raw_ref"] = raw if raw.get("ok") else None
            _acc, _host = _run_trace_tags(); _ip, _geo = _conn_ip_geo()
            r = cloud.save_run(name, payload, run_id=rid, ts=tstamp, account=_acc,
                               client=setup.get("client", ""), tag=setup.get("tag", ""),
                               hostname=_host or socket.gethostname(), ip=_ip, geo=_geo)
        except Exception as exc:  # noqa: BLE001
            r = {"ok": False, "reason": f"{type(exc).__name__}: {exc}"}
        if r.get("ok"):
            lbl_data.setText(T(f"☁ Uploaded saved run (id {r.get('id','')}).",
                               f"☁ Corrida guardada subida (id {r.get('id','')})."))
        else:
            lbl_data.setText(T(f"⚠ Upload failed ({r.get('reason','offline')}).",
                               f"⚠ Falló la subida ({r.get('reason','offline')})."))

    btn_save.clicked.connect(_save_run_local)
    btn_upload.clicked.connect(_upload_run)
    btn_loadup.clicked.connect(_upload_saved_run)
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
    sh_intro.setWordWrap(True); sh_intro.setStyleSheet("color:#475569;")
    sh_l.addWidget(sh_intro)

    sh_btns = QtWidgets.QHBoxLayout(); sh_l.addLayout(sh_btns)
    btn_both = QtWidgets.QPushButton(T("▶ Verify both (Ref 1 + Ref 2)", "▶ Verificar ambas (Ref 1 + Ref 2)"))
    btn_both.setStyleSheet(f"QPushButton{{background:{NAVY};color:white;padding:9px 16px;border-radius:9px;font-weight:700;}}"
                           "QPushButton:hover{background:#12325a;}")
    btn_ref1 = QtWidgets.QPushButton(T("Only Ref 1 (100 µε)", "Solo Ref 1 (100 µε)"))
    btn_ref2 = QtWidgets.QPushButton(T("Only Ref 2 (500 µε)", "Solo Ref 2 (500 µε)"))
    sh_btns.addWidget(btn_both); sh_btns.addWidget(btn_ref1); sh_btns.addWidget(btn_ref2); sh_btns.addStretch(1)
    sh_result = QtWidgets.QTextBrowser()
    sh_result.setStyleSheet("QTextBrowser{border:none;background:#f4f8fc;}")
    sh_l.addWidget(sh_result, 1)

    def _shunt_html(ref) -> str:
        """Tarjeta HTML del resultado de UNA referencia (para mostrar 1 o las 2)."""
        sc = st["scaling"]; gage = st["gage"]
        # En simulado: el RX10K reproduce el shunt con un pequeño error realista.
        from core.torsional.shunt_cal import expected_shunt_voltage, full_scale_strain_torque
        eps_fs = full_scale_strain_torque(gage)
        expected = expected_shunt_voltage(ref.simulated_ue, eps_fs, sc.scale_factor_z)
        rng = np.random.default_rng()
        measured = expected * (1.0 + rng.normal(0, 0.002)) + rng.normal(0, 0.003)
        chk = verify_shunt(measured, ref, gage, scale_factor_z=sc.scale_factor_z)
        _pass = chk.passed
        pill_bg = "#dcfce7" if _pass else "#fef9c3"
        pill_fg = "#166534" if _pass else "#854d0e"
        err_col = "#16a34a" if _pass else "#b45309"
        status = T("PASS", "OK") if _pass else T("OUT OF TOL", "FUERA DE TOL")
        def _cell(lbl, val, col="#0f2a4a"):
            return (f"<td width='33%' style='padding:6px 8px'>"
                    f"<div style='color:#94a3b8;font-size:11px'>{lbl}</div>"
                    f"<div style='color:{col};font-size:17px;font-weight:800'>{val}</div></td>")
        return (
            "<table width='100%' cellspacing='0' cellpadding='0' style='margin:0 0 14px 0'><tr>"
            "<td style='background:white;border:1px solid #e6ecf5;border-radius:12px'>"
            "<table width='100%' cellpadding='8' cellspacing='0'>"
            "<tr>"
            f"<td style='font-size:15px;color:#0f2a4a'><b>{ref.name}</b> "
            f"<span style='color:#94a3b8'>· {ref.simulated_ue:.0f} µε</span></td>"
            f"<td align='right'><span style='background:{pill_bg};color:{pill_fg};"
            f"padding:5px 14px;border-radius:14px;font-weight:800'>● {status}</span></td>"
            "</tr>"
            "<tr><td colspan='2'><table width='100%' cellspacing='0'><tr>"
            + _cell(T('Expected', 'Esperado'), f"{chk.expected_v:.4f} V")
            + _cell(T('Measured', 'Medido'), f"{chk.measured_v:.4f} V")
            + _cell(T('Error', 'Error'), f"{chk.error_pct:+.3f} %FS", err_col)
            + "</tr></table>"
            f"<div style='color:#64748b;font-size:11px;padding:2px 8px 8px 8px'>"
            f"{T('Effective Z revealed by shunt', 'Z efectivo del shunt')}: <b>{chk.suggested_z:.4f}</b> "
            f"({T('current', 'actual')}: {sc.scale_factor_z:.4f})</div>"
            "</td></tr></table></td></tr></table>")

    def _run_shunt(refs):
        if not _rebuild_scaling() or st["gage"] is None:
            QtWidgets.QMessageBox.warning(win, "Watermelon Torsional",
                T("Fix the configuration first.", "Corrige la configuración primero."))
            return
        sh_result.setHtml("".join(_shunt_html(r) for r in refs))

    btn_both.clicked.connect(lambda: _run_shunt([REF1_100UE, REF2_500UE]))
    btn_ref1.clicked.connect(lambda: _run_shunt([REF1_100UE]))
    btn_ref2.clicked.connect(lambda: _run_shunt([REF2_500UE]))
    tabs.addTab(pg_sh, T("Shunt check", "Verificación shunt"))

    # =================================================================
    # TAB 4 — Runup (order tracking / Campbell de torque)
    # =================================================================
    pg_ru = QtWidgets.QWidget(); ru_l = QtWidgets.QVBoxLayout(pg_ru)
    ru_intro = QtWidgets.QLabel(T(
        "The RPM axis is measured from the keyphasor during the ramp (not typed). 'Torsional natural' only "
        "injects a resonance in simulation — the real natural is detected automatically in Campbell.",
        "El eje de RPM se mide del keyphasor durante la rampa (no se escribe). 'Natural torsional' solo "
        "inyecta una resonancia en simulación — la natural real la detecta el Campbell automáticamente."))
    ru_intro.setWordWrap(True); ru_l.addWidget(ru_intro)
    ru_ctrl = QtWidgets.QHBoxLayout(); ru_l.addLayout(ru_ctrl)
    sb_r0 = QtWidgets.QDoubleSpinBox(); sb_r0.setRange(60, 12000); sb_r0.setValue(600); sb_r0.setSuffix(" rpm")
    sb_r1 = QtWidgets.QDoubleSpinBox(); sb_r1.setRange(60, 12000); sb_r1.setValue(3600); sb_r1.setSuffix(" rpm")
    sb_res = QtWidgets.QDoubleSpinBox(); sb_res.setRange(0, 500); sb_res.setValue(30); sb_res.setSuffix(" Hz")
    # Campos SOLO del simulador (inicio/fin/natural inyectada) → contenedor ocultable.
    ru_simbox = QtWidgets.QWidget(); _rusim = QtWidgets.QHBoxLayout(ru_simbox); _rusim.setContentsMargins(0, 0, 0, 0)
    for lbl, w in [(T("Start", "Inicio"), sb_r0), (T("End", "Fin"), sb_r1),
                   (T("Torsional natural (sim)", "Natural torsional (sim)"), sb_res)]:
        _rusim.addWidget(QtWidgets.QLabel(lbl)); _rusim.addWidget(w)
    ru_ctrl.addWidget(ru_simbox); st["_sim_widgets"].append(ru_simbox)
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
    _np0 = float((st["setup_fn"]() if st.get("setup_fn") else {}).get("nameplate_rpm") or 1800)
    sb_cb_rpm = QtWidgets.QDoubleSpinBox(); sb_cb_rpm.setRange(60, 12000); sb_cb_rpm.setValue(_np0); sb_cb_rpm.setSuffix(" rpm")
    # Velocidad de operación por DEFECTO = RPM de placa (Setup). Botones para
    # re-tomarla de placa o medirla del keyphasor. Cero errores del operador.
    btn_cb_plate = QtWidgets.QToolButton(); btn_cb_plate.setText(T("↺ Nameplate", "↺ Placa"))
    btn_cb_kph = QtWidgets.QToolButton(); btn_cb_kph.setText(T("◉ Keyphasor", "◉ Keyphasor"))
    sb_cb_rpm2 = QtWidgets.QDoubleSpinBox(); sb_cb_rpm2.setRange(0, 12000); sb_cb_rpm2.setValue(0); sb_cb_rpm2.setSuffix(" rpm")
    sb_cb_rpm2.setSpecialValueText(T("off", "—"))       # 0 = sin segunda banda
    cb_margin = QtWidgets.QComboBox(); cb_margin.addItems(["±10% (API 684)", "±15% (ISO 22266)", "±5%"])
    cb_ctrl.addWidget(QtWidgets.QLabel(T("Operating speed", "Velocidad de operación"))); cb_ctrl.addWidget(sb_cb_rpm)
    cb_ctrl.addWidget(btn_cb_plate); cb_ctrl.addWidget(btn_cb_kph)
    cb_ctrl.addWidget(QtWidgets.QLabel(T("Additional speed", "Velocidad adicional"))); cb_ctrl.addWidget(sb_cb_rpm2)
    cb_ctrl.addWidget(QtWidgets.QLabel(T("Permissible band", "Franja permisible"))); cb_ctrl.addWidget(cb_margin)
    btn_cb = QtWidgets.QPushButton(T("▶ Run Campbell", "▶ Correr Campbell"))
    cb_ctrl.addWidget(btn_cb); cb_ctrl.addStretch(1)
    cb_l.addLayout(cb_ctrl)

    def _cb_from_plate():
        _n = float((st["setup_fn"]() if st.get("setup_fn") else {}).get("nameplate_rpm") or 0)
        if _n > 0:
            sb_cb_rpm.setValue(_n)
    def _cb_from_keyphasor():
        # mide la velocidad real del keyphasor de una captura estable simulada
        if not _rebuild_scaling():
            return
        p = _preset(); fs = st["fs"]; sc = st["scaling"]
        cfg = TorsionalStreamConfig(sample_rate_hz=fs, rpm=p["rpm"],
            channels=make_torsional_channels(units=sc.units), block_seconds=0.25, buffer_seconds=4,
            mean_torque=p["mean"], orders=p["orders"], scaling=sc, torque_units=sc.units)
        src = SimulatedTorsionalSource(cfg); src.start()
        data = np.concatenate([src.read_block() for _ in range(12)], axis=1)
        _, ri = keyphasor_to_rpm(data[cfg.keyphasor_index()], fs)
        if ri.size:
            sb_cb_rpm.setValue(float(np.median(ri)))
    btn_cb_plate.clicked.connect(_cb_from_plate)
    btn_cb_kph.clicked.connect(_cb_from_keyphasor)
    _cbax = {"color": "#334155", "font-size": "10pt"}
    p_cb = pg.PlotWidget(); p_cb.setBackground("w"); p_cb.showGrid(x=True, y=True, alpha=0.3)
    p_cb.setLabel("bottom", T("speed", "velocidad"), "RPM", **_cbax)
    p_cb.setLabel("left", T("frequency", "frecuencia"), "Hz", **_cbax)
    p_cb.setTitle(T("Campbell / interference diagram", "Diagrama de Campbell / interferencia"))
    p_cb.getViewBox().setLimits(xMin=0, yMin=0)      # ejes nunca negativos
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
        rpm2 = sb_cb_rpm2.value()                              # velocidad adicional (0 = off)
        rpm_max = max(r1, rpm2 * 1.6) * 1.05
        band = SpeedBand(rpm, _mrg * rpm, "Op")
        bands = [band] + ([SpeedBand(rpm2, _mrg * rpm2, "Op2")] if rpm2 > 0 else [])
        _ORDERS = tuple(float(k) for k in range(1, 11))       # armónicos 1×…10×
        crossings = compute_crossings(naturals, 0.0, rpm_max, _ORDERS, bands=bands,
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
        if rpm2 > 0:                                           # 2ª banda de operación (velocidad adicional)
            _b2 = SpeedBand(rpm2, _mrg * rpm2, "Op2")
            _reg2 = pg.LinearRegionItem(values=[_b2.low, _b2.high], orientation="vertical", movable=False,
                                        brush=pg.mkBrush(59, 130, 246, 40), pen=pg.mkPen(None))
            _reg2.setZValue(-10); p_cb.addItem(_reg2)
            p_cb.addItem(pg.InfiniteLine(pos=rpm2, angle=90, pen=pg.mkPen("#2563eb", width=2, style=QtCore.Qt.DashLine)))
        xr = np.linspace(0, rpm_max, 60)
        _yv = _ymax                                           # techo visible (para recortar etiquetas)
        for o in _ORDERS:                                     # líneas de orden 1×…10× + etiqueta suave
            p_cb.plot(xr, o * xr / 60.0, pen=pg.mkPen("#b6c2d4", width=1, style=QtCore.Qt.DotLine))
            _lx = rpm_max * 0.985; _ly = o * _lx / 60.0
            if _ly > _yv:                                     # si sale por arriba, rotula sobre el eje X
                _lx = _yv * 60.0 / o; _ly = _yv
            _ot = pg.TextItem(f"{o:g}×", color="#9aa7bd", anchor=(1, 0))
            _ot.setPos(_lx, _ly); p_cb.addItem(_ot)
        def _ordn(i):
            _en = {1: "1st", 2: "2nd", 3: "3rd"}.get(i, f"{i}th")
            return T(f"{_en} torsional", f"{i}ª torsional")
        for _idx, fn in enumerate(naturals):                  # naturales torsionales rotuladas
            p_cb.plot([0, rpm_max], [fn, fn], pen=pg.mkPen(GREEN, width=2))
            _nt = pg.TextItem(f"{_ordn(_idx+1)} · {fn:.1f} Hz", color="#0f7a34", anchor=(0, 1))
            _nt.setPos(rpm_max * 0.02, fn); p_cb.addItem(_nt)
        p_cb.addItem(pg.InfiniteLine(pos=rpm, angle=90, pen=pg.mkPen(NAVY, width=2, style=QtCore.Qt.DashLine)))
        p_cb.setXRange(0.0, rpm_max, padding=0.0)             # X arranca en 0
        p_cb.setYRange(0.0, _ymax, padding=0.0)
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
        "Shaft fatigue-life diagnostic: rainflow (ASTM E1049) → shear stress → Goodman mean-stress "
        "correction → Palmgren-Miner damage. Traffic light by design safety factor (API 684 / ASME B106.1M).",
        "Diagnóstico de vida a fatiga del eje: rainflow (ASTM E1049) → esfuerzo cortante → corrección de "
        "esfuerzo medio (Goodman) → daño de Palmgren-Miner. Semáforo por factor de seguridad (API 684 / ASME B106.1M).")))

    ft_ctrl = QtWidgets.QHBoxLayout(); ft_l.addLayout(ft_ctrl)
    btn_ft = QtWidgets.QPushButton(T("▶ Capture & diagnose", "▶ Capturar y diagnosticar"))
    btn_ft.setStyleSheet(f"QPushButton{{background:{NAVY};}}QPushButton:hover{{background:#0b1e38;}}")
    cb_ft_sf = QtWidgets.QComboBox(); cb_ft_sf.addItems(["2.0", "1.5", "3.0"])
    cb_ft_sf.setToolTip(T("Design safety factor for 'infinite life' (green). API/AGMA ≈ 1.5–3.",
                          "Factor de seguridad de diseño para 'vida infinita' (verde). API/AGMA ≈ 1.5–3."))
    ft_ctrl.addWidget(btn_ft)
    ft_ctrl.addSpacing(12)
    ft_ctrl.addWidget(QtWidgets.QLabel(T("Design safety factor", "Factor de seguridad de diseño"))); ft_ctrl.addWidget(cb_ft_sf)
    ft_ctrl.addStretch(1)

    # --- Semáforo (banner grande de estado) ---
    ft_light = QtWidgets.QLabel("—"); ft_light.setAlignment(QtCore.Qt.AlignCenter)
    ft_light.setStyleSheet("background:#e2e8f0; color:#334155; border-radius:12px; padding:14px; "
                           "font-size:19px; font-weight:800;")
    ft_l.addWidget(ft_light)

    # --- KPI cards ---
    ft_kpi = QtWidgets.QHBoxLayout(); ft_l.addLayout(ft_kpi)
    def _kpi_card():
        w = QtWidgets.QLabel("—"); w.setAlignment(QtCore.Qt.AlignCenter)
        w.setStyleSheet("background:white; border:1px solid #e2e8f0; border-radius:10px; padding:10px;")
        w.setTextFormat(QtCore.Qt.RichText); ft_kpi.addWidget(w); return w
    kpi_sf, kpi_tau, kpi_life, kpi_cyc = _kpi_card(), _kpi_card(), _kpi_card(), _kpi_card()

    p_ft = pg.PlotWidget(); p_ft.setBackground("w"); p_ft.showGrid(x=True, y=True, alpha=0.3)
    p_ft.setLabel("bottom", T("torque range", "rango de par")); p_ft.setLabel("left", T("cycle count", "conteo"))
    p_ft.setTitle(T("Rainflow histogram", "Histograma rainflow"))
    ft_l.addWidget(p_ft, 1)

    _LIGHT_BG = {"green": ("#dcfce7", "#166534"), "yellow": ("#fef9c3", "#854d0e"), "red": ("#fee2e2", "#991b1b")}

    def _set_kpi(w, title, value, accent=NAVY):
        w.setText(f"<div style='color:#64748b;font-size:11px'>{title}</div>"
                  f"<div style='color:{accent};font-size:20px;font-weight:800'>{value}</div>")

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
        win_s = 8.0
        data = np.concatenate([src.read_block() for _ in range(32)], axis=1)
        ki = cfg.keyphasor_index(); ti = next(i for i in range(cfg.n_channels) if i != ki)
        torque = voltage_to_torque(data[ti], sc)
        u = "N·m" if units == "nm" else "ft-lb"
        cyc = rainflow_cycles(torque)
        ranges = fatigue_ranges(torque)
        # --- diagnóstico de vida a fatiga (Goodman + Miner) ---
        try:
            life = shaft_torsional_fatigue(
                cyc, outer_diameter_in=sb_do.value(), inner_diameter_in=sb_di.value(),
                ultimate_strength_psi=sb_sut.value() * 1000.0, torque_units=units,
                window_seconds=win_s, design_safety_factor=float(cb_ft_sf.currentText()),
                endurance_ratio=float(st.get("endurance_ratio", 0.50)))
        except ValueError as exc:      # geometría no diagnosticable → aviso claro, sin crash
            ft_light.setStyleSheet("background:#fee2e2; color:#991b1b; border-radius:12px; "
                                   "padding:14px; font-size:16px; font-weight:800;")
            ft_light.setText(T(f"Check shaft geometry: {exc}", f"Revisa la geometría del eje: {exc}"))
            return
        st["fat"] = {"ranges": ranges, "units": u, "life": life}

        bg, fg = _LIGHT_BG[life.status]
        _dot = {"green": GREEN, "yellow": AMBER, "red": RED}[life.status]
        _lab = life.label_es if _LANG == "es" else life.label_en
        ft_light.setStyleSheet(f"background:{bg}; color:{fg}; border-radius:12px; padding:14px; "
                               f"font-size:19px; font-weight:800;")
        ft_light.setText(f"● {_lab}")
        # KPIs
        _acc = fg
        _set_kpi(kpi_sf, T("Safety factor (fatigue)", "Factor de seguridad (fatiga)"),
                 ("∞" if life.safety_factor == float("inf") else f"{life.safety_factor:.2f}"), _acc)
        _set_kpi(kpi_tau, T("Alt. shear τ<sub>ar</sub>", "Cortante alt. τ<sub>ar</sub>"),
                 f"{life.tau_alt_max_psi/1000:.1f} ksi", _acc)
        if life.infinite:
            _set_kpi(kpi_life, T("Estimated life", "Vida estimada"), T("Infinite", "Infinita"), _acc)
        else:
            _yr = life.life_hours / 8760.0
            _lv = (f"{life.life_hours:,.0f} h" if life.life_hours < 8760 else f"{_yr:,.1f} " + T("yr", "años"))
            _set_kpi(kpi_life, T("Estimated life", "Vida estimada"), _lv, _acc)
        _set_kpi(kpi_cyc, T("Cycles / largest range", "Ciclos / rango máx"),
                 f"{life.n_cycles:,.0f}<br><span style='font-size:12px'>{(max((r for r,_ in ranges),default=0)):,.0f} {u}</span>", _acc)

        p_ft.clear()
        if ranges:
            rr = np.array([r for r, _ in ranges]); cc = np.array([c for _, c in ranges])
            nb = int(np.clip(len(rr), 8, 24)); edges = np.linspace(0, rr.max() * 1.0001, nb + 1)
            hist, _ = np.histogram(rr, bins=edges, weights=cc)
            ctr = 0.5 * (edges[:-1] + edges[1:])
            p_ft.addItem(pg.BarGraphItem(x=ctr, height=hist, width=(edges[1] - edges[0]) * 0.9, brush=_dot))
    btn_ft.clicked.connect(_run_fatigue)
    tabs.addTab(pg_ft, "Fatigue")

    # =================================================================
    # TAB 6b — Monitor 24h (larga duración: fatiga acumulada + tendencia)
    # =================================================================
    pg_mon = QtWidgets.QWidget(); mn_l = QtWidgets.QVBoxLayout(pg_mon)
    mn_l.addWidget(QtWidgets.QLabel(T(
        "Long-duration torsional monitoring. Accumulates rainflow fatigue (real Miner damage) + trend "
        "+ overload events — WITHOUT storing the continuous raw wave (24 h ≈ MB, not GB).",
        "Monitoreo torsional de larga duración. Acumula fatiga rainflow (daño de Miner real) + tendencia "
        "+ eventos de sobrecarga — SIN guardar la onda cruda continua (24 h ≈ MB, no GB).")))
    mn_ctrl = QtWidgets.QHBoxLayout(); mn_l.addLayout(mn_ctrl)
    sb_mon_h = QtWidgets.QDoubleSpinBox(); sb_mon_h.setRange(0.05, 72.0); sb_mon_h.setValue(24.0); sb_mon_h.setSuffix(" h")
    btn_mon_go = QtWidgets.QPushButton(T("▶ Start monitoring", "▶ Iniciar monitoreo"))
    btn_mon_go.setStyleSheet(f"QPushButton{{background:{GREEN};}}QPushButton:hover{{background:#12833a;}}")
    btn_mon_stop = QtWidgets.QPushButton(T("■ Stop", "■ Detener")); btn_mon_stop.setEnabled(False)
    btn_mon_resume = QtWidgets.QPushButton(T("↺ Resume last", "↺ Reanudar última"))
    mn_ctrl.addWidget(QtWidgets.QLabel(T("Target duration", "Duración objetivo"))); mn_ctrl.addWidget(sb_mon_h)
    mn_ctrl.addWidget(btn_mon_go); mn_ctrl.addWidget(btn_mon_stop); mn_ctrl.addWidget(btn_mon_resume); mn_ctrl.addStretch(1)

    def _mon_dir():
        import os
        d = os.path.join(os.path.expanduser("~"), "WatermelonTorsional", "monitors")
        os.makedirs(d, exist_ok=True)
        return d

    def _mon_ckpt_path():
        import os
        # Checkpoint FUERA de la carpeta de campañas → el diálogo de cargar queda limpio.
        base = os.path.join(os.path.expanduser("~"), "WatermelonTorsional")
        os.makedirs(base, exist_ok=True)
        return os.path.join(base, "_monitor_checkpoint.json")

    def _mon_save_ckpt():
        try:
            import json
            with open(_mon_ckpt_path(), "w", encoding="utf-8") as fh:
                json.dump(st["_mon"].to_dict(), fh)
        except Exception:  # noqa: BLE001
            pass

    mn_light = QtWidgets.QLabel("—"); mn_light.setAlignment(QtCore.Qt.AlignCenter)
    mn_light.setStyleSheet("background:#e2e8f0;color:#334155;border-radius:12px;padding:12px;font-size:17px;font-weight:800;")
    mn_l.addWidget(mn_light)
    mn_kpi = QtWidgets.QHBoxLayout(); mn_l.addLayout(mn_kpi)
    def _mkpi():
        w = QtWidgets.QLabel("—"); w.setAlignment(QtCore.Qt.AlignCenter); w.setTextFormat(QtCore.Qt.RichText)
        w.setStyleSheet("background:white;border:1px solid #e2e8f0;border-radius:10px;padding:8px;"); mn_kpi.addWidget(w); return w
    k_elapsed, k_cycles, k_events, k_tmax = _mkpi(), _mkpi(), _mkpi(), _mkpi()

    _mpr = QtWidgets.QHBoxLayout(); mn_l.addLayout(_mpr, 1)
    p_mtr = pg.PlotWidget(); p_mtr.setBackground("w"); p_mtr.showGrid(x=True, y=True, alpha=0.3)
    p_mtr.setTitle(T("Torque trend (pp) & speed", "Tendencia de par (pp) y velocidad"))
    p_mtr.setLabel("bottom", T("time", "tiempo"), "s", **_axlbl); p_mtr.addLegend()
    _mc_pp = p_mtr.plot(pen=pg.mkPen(ACC, width=2), name=T("torque pp", "par pp"))
    p_mh = pg.PlotWidget(); p_mh.setBackground("w"); p_mh.showGrid(y=True, alpha=0.25)
    p_mh.setTitle(T("Accumulated rainflow histogram", "Histograma rainflow acumulado"))
    p_mh.setLabel("bottom", T("torque range", "rango de par"), **_axlbl); p_mh.setLabel("left", T("cycles", "ciclos"), **_axlbl)
    _mpr.addWidget(p_mtr, 1); _mpr.addWidget(p_mh, 1)

    mn_row = QtWidgets.QHBoxLayout(); mn_l.addLayout(mn_row)
    btn_mon_savelocal = QtWidgets.QPushButton(T("💾 Save locally", "💾 Guardar local"))
    btn_mon_upload = QtWidgets.QPushButton(T("☁ Upload to cloud", "☁ Subir a la nube"))
    btn_mon_load = QtWidgets.QPushButton(T("📂 Load a saved campaign", "📂 Cargar campaña guardada"))
    mn_row.addWidget(btn_mon_savelocal); mn_row.addWidget(btn_mon_upload); mn_row.addWidget(btn_mon_load); mn_row.addStretch(1)
    mn_status = QtWidgets.QLabel(""); mn_status.setWordWrap(True); mn_status.setStyleSheet("color:#475569;")
    mn_l.addWidget(mn_status)

    mon_timer = QtCore.QTimer(win); mon_timer.setInterval(250)
    st["_mon"] = None; st["_mon_src"] = None; st["_mon_ticks"] = 0

    def _mon_make_source():
        """Fuente para monitoreo (sim o NI 9229), misma config que Live. None si falla."""
        fs = st["fs"]; sc = st["scaling"]; units = sc.units; p = _preset()
        if st.get("acq_mode") == "ni":
            _nc = st.get("ni_cfg", {})
            ncfg = NITorsionalConfig(device=_nc.get("device", "cDAQ1Mod1"),
                torque_ai=int(_nc.get("torque_ai", 0)), kph_ai=int(_nc.get("kph_ai", 1)),
                sample_rate_hz=fs, block_seconds=0.25, voltage_range=10.0,
                keyphasor=st.get("kph_sensor") or KeyphasorSensor.phototach_reflective())
            src = NITorsionalSource(ncfg)
            try:
                src.start()
            except RuntimeError as exc:
                QtWidgets.QMessageBox.critical(win, "Watermelon Torsional",
                    T(f"NI 9229 failed:\n{exc}", f"Falló NI 9229:\n{exc}")); return None
            return src
        cfg = TorsionalStreamConfig(sample_rate_hz=fs, rpm=p["rpm"],
            channels=make_torsional_channels(units=units), block_seconds=0.25, buffer_seconds=2,
            mean_torque=p["mean"], orders=p["orders"], gear_teeth=p["gear_teeth"], gear_amp_eu=p["gear_amp"],
            torsional_res_hz=p["res_hz"], torque_noise_rms_eu=p["noise"], scaling=sc, torque_units=units)
        src = SimulatedTorsionalSource(cfg); src.start(); return src

    def _mon_tick():
        src = st.get("_mon_src"); mon = st.get("_mon"); sc = st.get("scaling")
        if src is None or mon is None or sc is None:
            return
        block = src.read_block(); cfg = src.config
        ki = cfg.keyphasor_index(); ti = next(i for i in range(cfg.n_channels) if i != ki)
        torque = voltage_to_torque(block[ti], sc)
        _sensor = st.get("kph_sensor") or KeyphasorSensor.simulated()
        _, rpm = rpm_from_keyphasor(block[ki], st["fs"], _sensor)
        rpm_now = float(np.median(rpm)) if rpm.size else _preset()["rpm"]
        mon.add_block(torque, rpm_now, st["fs"])
        st["_mon_ticks"] += 1
        if mon.duration_s >= sb_mon_h.value() * 3600.0:      # alcanzó la duración objetivo
            _mon_stop(); return
        if st["_mon_ticks"] % 4 == 0:                        # refresca UI ~1 Hz
            _mon_refresh()
        if st["_mon_ticks"] % 240 == 0:                      # autosave checkpoint ~cada 60 s
            _mon_save_ckpt()

    def _mon_refresh():
        mon = st.get("_mon")
        if mon is None:
            return
        u = "N·m" if mon.units == "nm" else "ft-lb"
        _h = mon.duration_s / 3600.0
        _set_kpi(k_elapsed, T("Elapsed", "Transcurrido"), (f"{_h:.2f} h" if _h >= 1 else f"{mon.duration_s:,.0f} s"))
        ranges = mon.rf.ranges(drain=True, tail=mon._carry)
        _ncyc = sum(c for _, c in ranges)
        _set_kpi(k_cycles, T("Cycles", "Ciclos"), f"{_ncyc:,.0f}")
        _set_kpi(k_events, T("Events", "Eventos"), f"{len(mon.events)}", RED if mon.events else NAVY)
        _set_kpi(k_tmax, T("Max torque", "Par máx"), f"{(mon.tmax or 0):,.0f}<br><span style='font-size:11px'>{u}</span>")
        # tendencia
        if mon.trend:
            tt = [r[0] for r in mon.trend]; pp = [r[2] for r in mon.trend]
            _mc_pp.setData(tt, pp)
        # histograma
        p_mh.clear()
        if ranges:
            rr = np.array([r for r, _ in ranges]); cc = np.array([c for _, c in ranges])
            nb = int(np.clip(len(rr), 8, 30)); edg = np.linspace(0, rr.max() * 1.0001, nb + 1)
            hh, _ = np.histogram(rr, bins=edg, weights=cc)
            p_mh.addItem(pg.BarGraphItem(x=0.5 * (edg[:-1] + edg[1:]), height=hh, width=(edg[1] - edg[0]) * 0.9, brush=NAVY))
        # veredicto de fatiga acumulada
        try:
            life = shaft_torsional_fatigue(mon.fatigue_cycles(), sb_do.value(), sb_di.value(),
                sb_sut.value() * 1000.0, mon.units, window_seconds=max(mon.duration_s, 1.0),
                design_safety_factor=float(cb_ft_sf.currentText()),
                endurance_ratio=float(st.get("endurance_ratio", 0.50)))
            bg, fg = _LIGHT_BG[life.status]; lab = life.label_es if _LANG == "es" else life.label_en
            _sf = "∞" if life.safety_factor == float("inf") else f"{life.safety_factor:.2f}"
            _lifetxt = (T("infinite", "infinita") if life.infinite else
                        (f"{life.life_hours:,.0f} h" if life.life_hours < 8760 else f"{life.life_hours/8760:,.1f} " + T("yr", "años")))
            mn_light.setStyleSheet(f"background:{bg};color:{fg};border-radius:12px;padding:12px;font-size:17px;font-weight:800;")
            mn_light.setText(f"● {lab} · SF {_sf} · {_lifetxt}")
        except Exception:  # noqa: BLE001
            pass
        # Watchdog de señal: alarma si la señal está muerta (batería TX10K / cable).
        if not mon.signal_ok:
            mn_status.setText(T(f"⚠ SIGNAL LOST (std {mon.last_std:.2f}) — check TX10K 9V battery / cable. "
                                f"Dead time: {mon.bad_seconds:,.0f} s.",
                                f"⚠ SEÑAL PERDIDA (std {mon.last_std:.2f}) — revisa batería 9V del TX10K / cable. "
                                f"Tiempo muerto: {mon.bad_seconds:,.0f} s."))
            mn_status.setStyleSheet("color:#b91c1c; font-weight:700;")
        elif st.get("running_mon"):
            mn_status.setStyleSheet("color:#475569;")

    def _mon_start():
        if not _rebuild_scaling():
            return
        if not _preflight():
            return
        src = _mon_make_source()
        if src is None:
            return
        _p = _preset()
        st["_mon"] = TorsionalMonitor(units=st["scaling"].units, trend_dt=2.0,
                                      event_pp=max(3.0 * _p["mean"], 1.0))   # umbral evento = 3× par medio
        st["_mon_setup"] = None; st["_mon_shaft"] = None    # campaña nueva → usa setup/eje en vivo
        st["_mon_src"] = src; st["_mon_ticks"] = 0; st["running_mon"] = True
        btn_mon_go.setEnabled(False); btn_mon_stop.setEnabled(True); sb_mon_h.setEnabled(False)
        mn_status.setStyleSheet("color:#475569;")
        mn_status.setText(T("Monitoring… (leave running; autosaves every 60 s)",
                            "Monitoreando… (déjalo corriendo; autoguarda cada 60 s)"))
        mon_timer.start(); _refresh_hw_banner()

    def _mon_resume():
        """Reanuda la última campaña tras un corte de energía / cierre del PC."""
        import os, json
        cp = _mon_ckpt_path()
        if not os.path.exists(cp):
            mn_status.setText(T("No saved campaign to resume.", "No hay campaña guardada para reanudar.")); return
        if not _rebuild_scaling() or not _preflight():
            return
        try:
            with open(cp, "r", encoding="utf-8") as fh:
                st["_mon"] = TorsionalMonitor.from_dict(json.load(fh))
        except Exception as e:  # noqa: BLE001
            mn_status.setText(T(f"Could not load checkpoint: {e}", f"No se pudo cargar el checkpoint: {e}")); return
        src = _mon_make_source()
        if src is None:
            return
        st["_mon_src"] = src; st["_mon_ticks"] = 0; st["running_mon"] = True
        btn_mon_go.setEnabled(False); btn_mon_stop.setEnabled(True); sb_mon_h.setEnabled(False)
        mn_status.setStyleSheet("color:#475569;")
        mn_status.setText(T(f"Resumed at {st['_mon'].duration_s/3600.0:.2f} h.",
                            f"Reanudado en {st['_mon'].duration_s/3600.0:.2f} h."))
        mon_timer.start(); _mon_refresh(); _refresh_hw_banner()

    def _mon_stop():
        mon_timer.stop(); st["running_mon"] = False
        if st.get("_mon_src") is not None:
            try: st["_mon_src"].stop()
            except Exception: pass  # noqa: BLE001
        if st.get("_mon"):
            _mon_save_ckpt()                     # checkpoint final (por si cierran sin guardar)
        btn_mon_go.setEnabled(True); btn_mon_stop.setEnabled(False); sb_mon_h.setEnabled(True)
        _mon_refresh()
        if st.get("_mon"):
            mn_status.setText(T(f"Stopped at {st['_mon'].duration_s/3600.0:.2f} h. Save/upload the summary.",
                                f"Detenido en {st['_mon'].duration_s/3600.0:.2f} h. Guarda/sube el resumen."))
        _refresh_hw_banner()

    def _mon_setup_now():
        return st.get("_mon_setup") or (st["setup_fn"]() if st.get("setup_fn") else {})

    def _mon_shaft_now():
        return st.get("_mon_shaft") or {"do": sb_do.value(), "di": sb_di.value(), "sut_ksi": sb_sut.value()}

    def _mon_payload():
        """Payload para SUBIR (resumen liviano + setup + eje)."""
        mon = st.get("_mon")
        if mon is None or mon.n_samples == 0:
            return None, None
        setup = _mon_setup_now()
        payload = mon.summary()
        payload["setup"] = setup
        payload["shaft"] = _mon_shaft_now()
        payload["app_version"] = __version__
        return payload, setup

    def _mon_save_local():
        """Guarda el estado COMPLETO en el PC (offline-first, recargable). No requiere red."""
        mon = st.get("_mon")
        if mon is None or mon.n_samples == 0:
            mn_status.setText(T("Nothing to save yet.", "Nada que guardar aún.")); return
        try:
            import os, json
            from datetime import datetime
            # Carpeta por fecha_hora → dentro solo 1 archivo (limpio para cargar).
            folder = os.path.join(_mon_dir(), datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
            os.makedirs(folder, exist_ok=True)
            fp = os.path.join(folder, "monitor.json")
            record = dict(mon.to_dict())            # estado completo (hist+residual → recargable)
            record["setup"] = _mon_setup_now(); record["shaft"] = _mon_shaft_now()
            record["app_version"] = __version__
            with open(fp, "w", encoding="utf-8") as fh:
                json.dump(record, fh, ensure_ascii=False)
            st["_mon_last_local"] = fp
            mn_status.setStyleSheet("color:#16a34a; font-weight:600;")
            mn_status.setText(T(f"💾 Saved locally: {fp}  (upload later when you have internet).",
                                f"💾 Guardado local: {fp}  (súbelo luego cuando tengas internet)."))
        except Exception as e:  # noqa: BLE001
            mn_status.setText(T(f"⚠ Local save failed: {e}", f"⚠ Falló el guardado local: {e}"))

    def _mon_upload():
        """Sube a la nube el monitor actualmente cargado (recién corrido o cargado de disco)."""
        payload, setup = _mon_payload()
        if payload is None:
            mn_status.setText(T("Load or run a campaign first.", "Carga o corre una campaña primero.")); return
        setup = setup or {}
        name = (setup.get("machine") or setup.get("tag") or "Torsional monitor") + " · 24h"
        mn_status.setStyleSheet("color:#475569;")
        mn_status.setText(T("☁ Uploading…", "☁ Subiendo…")); QtWidgets.QApplication.processEvents()
        try:
            from core.torsional import cloud
            import socket
            _acc, _host = _run_trace_tags(); _ip, _geo = _conn_ip_geo()
            r = cloud.save_run(name, payload, account=_acc, client=setup.get("client", ""),
                               tag=setup.get("tag", ""), hostname=_host or socket.gethostname(), ip=_ip, geo=_geo)
        except Exception as e:  # noqa: BLE001
            r = {"ok": False, "reason": str(e)}
        if r.get("ok"):
            try:
                import os
                _cp = _mon_ckpt_path()
                if os.path.exists(_cp):
                    os.remove(_cp)               # campaña cerrada → checkpoint ya no hace falta
            except Exception:  # noqa: BLE001
                pass
            mn_status.setStyleSheet("color:#16a34a; font-weight:600;")
            mn_status.setText(T(f"☁ Uploaded to cloud (id {r.get('id','')}).",
                                f"☁ Subido a la nube (id {r.get('id','')})."))
        else:
            mn_status.setStyleSheet("color:#b45309; font-weight:600;")
            mn_status.setText(T(f"⚠ Could not upload ({r.get('reason','offline')}). Saved-local stays available.",
                                f"⚠ No se pudo subir ({r.get('reason','offline')}). El guardado local queda disponible."))

    def _mon_load():
        """Abre una campaña guardada en el PC → la carga para analizar/subir.
        Arranca en la carpeta de campañas (organizadas por fecha_hora)."""
        import json
        fp, _ = QtWidgets.QFileDialog.getOpenFileName(win, T("Open saved campaign", "Abrir campaña guardada"),
                                                      _mon_dir(), "JSON (*.json)")
        if not fp:
            return
        try:
            with open(fp, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            st["_mon"] = TorsionalMonitor.from_dict(data)   # estado completo (hist/residual)
            st["_mon_setup"] = data.get("setup") or {}       # usa el setup guardado al subir
            st["_mon_shaft"] = data.get("shaft") or {}
        except Exception as e:  # noqa: BLE001
            mn_status.setText(T(f"⚠ Could not open: {e}", f"⚠ No se pudo abrir: {e}")); return
        _mon_refresh()
        mn_status.setStyleSheet("color:#475569;")
        mn_status.setText(T(f"📂 Loaded {os.path.basename(fp)} — analyze, then Upload to cloud.",
                            f"📂 Cargado {os.path.basename(fp)} — analiza y luego Sube a la nube."))

    btn_mon_go.clicked.connect(_mon_start); btn_mon_stop.clicked.connect(_mon_stop)
    btn_mon_resume.clicked.connect(_mon_resume)
    btn_mon_savelocal.clicked.connect(_mon_save_local); btn_mon_upload.clicked.connect(_mon_upload)
    btn_mon_load.clicked.connect(_mon_load); mon_timer.timeout.connect(_mon_tick)
    tabs.addTab(pg_mon, T("Monitor 24h", "Monitor 24h"))

    # =================================================================
    # TAB 7 — Preliminary report (PDF de campo, como el Modal)
    # =================================================================
    pg_rp = QtWidgets.QWidget(); rp_l = QtWidgets.QVBoxLayout(pg_rp)
    rp_l.addWidget(QtWidgets.QLabel(T(
        "Full same-day field PDF — everything: torque signal, shunt check, run-up, Campbell and the "
        "fatigue-life diagnostic (traffic light). Uses the Setup data — no need to re-enter it. "
        "Just pick the language and generate. The full SIGA report is produced on the web.",
        "PDF de campo del mismo día — todo: señal de par, verificación shunt, run-up, Campbell y el "
        "diagnóstico de vida a fatiga (semáforo). Usa los datos del Setup — no hay que reingresarlos. "
        "Solo elige el idioma y genera. El reporte SIGA completo se genera en la web.")))
    lbl_rp_machine = QtWidgets.QLabel(""); lbl_rp_machine.setStyleSheet(f"color:{NAVY}; font-weight:700;")
    rp_l.addWidget(lbl_rp_machine)
    _rlrow = QtWidgets.QHBoxLayout()
    _rlrow.addWidget(QtWidgets.QLabel(T("Report language", "Idioma del reporte")))
    cb_rp_lang = QtWidgets.QComboBox(); cb_rp_lang.addItems(["Español", "English"])
    cb_rp_lang.setCurrentIndex(0 if _LANG == "es" else 1)
    _rlrow.addWidget(cb_rp_lang); _rlrow.addStretch(1)
    rp_l.addLayout(_rlrow)
    btn_rp = QtWidgets.QPushButton(T("📄 Generate preliminary report (PDF)", "📄 Generar reporte preliminar (PDF)"))
    btn_rp.setStyleSheet(f"QPushButton{{background:{GREEN};}}QPushButton:hover{{background:#12833a;}}")
    rp_l.addWidget(btn_rp)
    btn_rp_mon = QtWidgets.QPushButton(T("📄 Generate 24h monitor report (PDF)", "📄 Generar reporte de monitoreo 24h (PDF)"))
    btn_rp_mon.setToolTip(T("Uses the last 24h campaign (run it or load it in the Monitor 24h tab).",
                            "Usa la última campaña 24h (córrela o cárgala en la pestaña Monitor 24h)."))
    rp_l.addWidget(btn_rp_mon)
    rp_status = QtWidgets.QLabel(""); rp_status.setWordWrap(True); rp_l.addWidget(rp_status)
    rp_l.addStretch(1)

    def _refresh_rp_machine():
        _s = st["setup_fn"]() if st.get("setup_fn") else {}
        _nm = _s.get("machine") or _s.get("tag") or "—"
        lbl_rp_machine.setText(T(f"Machine: {_nm}  ·  Client: {_s.get('client','—') or '—'}",
                                 f"Máquina: {_nm}  ·  Cliente: {_s.get('client','—') or '—'}"))
    tabs.currentChanged.connect(lambda _=0: _refresh_rp_machine())

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
        rp_status.setText(T("Running full analysis (torque · shunt · run-up · Campbell · fatigue)…",
                            "Corriendo análisis completo (par · shunt · run-up · Campbell · fatiga)…"))
        QtWidgets.QApplication.processEvents()
        # Corre TODO para que cada gráfico del reporte esté fresco y poblado.
        _run_runup(); _run_campbell(); _run_fatigue()
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
        # Dibuja onda y espectro en la pestaña Live para poder capturarlos al PDF.
        _tt = np.arange(torque.size) / st["fs"]; curve_t.setData(_tt, torque)
        _fr, _am = torque_spectrum(torque, st["fs"]); _mk = _fr <= 600.0
        curve_s.setData(_fr[_mk], _am[_mk])
        QtWidgets.QApplication.processEvents()
        # --- Shunt check de ambas referencias (tabla de calibración) ---
        shunt_rows = []
        _gage = st.get("gage")
        if _gage is not None:
            from core.torsional.shunt_cal import expected_shunt_voltage, full_scale_strain_torque
            _eps = full_scale_strain_torque(_gage); _rng = np.random.default_rng()
            for _ref in (REF1_100UE, REF2_500UE):
                _ev = expected_shunt_voltage(_ref.simulated_ue, _eps, sc.scale_factor_z)
                _mv = _ev * (1.0 + _rng.normal(0, 0.002)) + _rng.normal(0, 0.003)
                _chk = verify_shunt(_mv, _ref, _gage, scale_factor_z=sc.scale_factor_z)
                shunt_rows.append([_ref.name, f"{_ref.simulated_ue:.0f} µε", f"{_chk.expected_v:.4f} V",
                                   f"{_chk.measured_v:.4f} V", f"{_chk.error_pct:+.2f} %FS",
                                   (T("PASS", "OK") if _chk.passed else T("OUT OF TOL", "FUERA TOL"))])
        camp = st.get("camp", {}); crossings = camp.get("crossings", [])
        coincid = [c for c in crossings if c.severity == "coincidence"]
        worst = min((c.sep_margin_pct for c in crossings), default=float("inf"))
        ranges = st.get("fat", {}).get("ranges", [])
        rmax = max((r for r, _ in ranges), default=0.0)
        life = st.get("fat", {}).get("life")
        _rip = "∞" if mm.ripple_pct == float("inf") else f"{mm.ripple_pct:.1f}%"

        _fat_gostat = {"green": "GO", "yellow": "REVIEW", "red": "NO-GO"}.get(getattr(life, "status", ""), "—")
        _fat_sf = ("∞" if life and life.safety_factor == float("inf") else (f"{life.safety_factor:.2f}" if life else "—"))
        quality = [
            (T("Signal", "Señal"), "OK", T(f"Mean {mm.mean:,.0f} {u}, pp {mm.peak_to_peak:,.0f} {u}",
                                           f"Media {mm.mean:,.0f} {u}, pp {mm.peak_to_peak:,.0f} {u}")),
            (T("Separation margin (API 684)", "Margen de separación (API 684)"),
             "NO-GO" if coincid else ("REVIEW" if worst < 10 else "GO"),
             T(f"Worst {worst:.0f}% (target ≥10%)", f"Mínimo {worst:.0f}% (objetivo ≥10%)")),
            (T("Shaft fatigue life (Goodman/Miner)", "Vida a fatiga del eje (Goodman/Miner)"),
             _fat_gostat,
             T(f"Safety factor {_fat_sf} — {life.label_en if life else '—'}",
               f"Factor de seguridad {_fat_sf} — {life.label_es if life else '—'}")),
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
        if life is not None:
            _lifetxt = (T("infinite life", "vida infinita") if life.infinite else
                        (T(f"~{life.life_hours:,.0f} h", f"~{life.life_hours:,.0f} h") if life.life_hours < 8760
                         else T(f"~{life.life_hours/8760:,.1f} yr", f"~{life.life_hours/8760:,.1f} años")))
            findings.append(T(
                f"Shaft fatigue: safety factor {_fat_sf} vs endurance limit — {life.label_en} ({_lifetxt}).",
                f"Fatiga del eje: factor de seguridad {_fat_sf} vs límite de fatiga — {life.label_es} ({_lifetxt})."))
        recs = [T("Confirm coincidences with an operating amplitude/phase run (API 684).",
                  "Confirmar coincidencias con corrida de amplitud/fase en operación (API 684)."),
                T("Evaluate shaft fatigue against the Goodman diagram at the gage location.",
                  "Evaluar fatiga del eje contra el diagrama de Goodman en la galga.")]
        ord_rows = [[f"{o}×", f"{o*rpm/60:.2f} Hz", f"{oa[float(o)][0]:.1f} {u}"] for o in range(1, 6)]
        # Tabla del diagnóstico de fatiga (con veredicto del semáforo).
        _fat_verd = (life.label_es if _LANG == "es" else life.label_en) if life else "—"
        if life is None:
            _fat_life = "—"
        elif life.infinite:
            _fat_life = T("Infinite", "Infinita")
        elif life.life_hours < 8760:
            _fat_life = f"{life.life_hours:,.0f} h"
        else:
            _fat_life = f"{life.life_hours/8760:,.1f} " + T("yr", "años")
        fat_rows = [
            [T("Traffic light", "Semáforo"), f"{_fat_gostat} — {_fat_verd}"],
            [T("Safety factor (fatigue)", "Factor de seguridad (fatiga)"), _fat_sf],
            [T("Alternating shear τar", "Cortante alternante τar"), f"{life.tau_alt_max_psi/1000:.1f} ksi" if life else "—"],
            [T("Endurance limit Sse", "Límite de fatiga Sse"), f"{life.sse_psi/1000:.1f} ksi" if life else "—"],
            [T("Estimated life", "Vida estimada"), _fat_life],
            [T("Largest torque range", "Mayor rango de par"), f"{rmax:,.0f} {u}"],
        ]
        sections = [
            {"title": T("Torque signal", "Señal de par"),
             "figures": [(T("Torque vs time", "Par vs tiempo"), _grab_png(p_time)),
                         (T("Torque spectrum", "Espectro de par"), _grab_png(p_spec))],
             "table": {"headers": [T("Order", "Orden"), T("Freq", "Frec"), T("Amplitude", "Amplitud")], "rows": ord_rows}},
            {"title": T("Shunt calibration check", "Verificación de calibración (shunt)"),
             "table": {"headers": [T("Reference", "Referencia"), "µε", T("Expected", "Esperado"),
                                   T("Measured", "Medido"), T("Error", "Error"), T("Status", "Estado")],
                       "rows": shunt_rows or [[T("Not run", "No corrido"), "", "", "", "", ""]]}},
            {"title": T("Run-up / order tracking", "Run-up / order tracking"),
             "figures": [(T("Order amplitude vs speed", "Amplitud de orden vs velocidad"), _grab_png(p_camp))]},
            {"title": T("Campbell / interference (API 684)", "Campbell / interferencia (API 684)"),
             "figures": [(T("Orders vs torsional naturals", "Órdenes vs naturales"), _grab_png(p_cb))]},
            {"title": T("Shaft fatigue life (Goodman / Miner)", "Vida a fatiga del eje (Goodman / Miner)"),
             "figures": [(T("Rainflow histogram", "Histograma rainflow"), _grab_png(p_ft))],
             "table": {"headers": [T("Item", "Ítem"), T("Value", "Valor")], "rows": fat_rows}},
        ]
        _su = st["setup_fn"]() if st.get("setup_fn") else {}
        _asset = _su.get("machine") or _su.get("tag") or "—"
        from datetime import date as _date
        meta = {"title": T("Preliminary Torsional Report", "Reporte Torsional Preliminar"),
                "asset": _asset, "client": _su.get("client") or "—",
                "machine_type": _su.get("type") or "—",
                "location": _su.get("location") or "—",
                "test_type": T("Torsional analysis", "Análisis torsional"),
                "rpm": f"{rpm:,.0f}",
                "technician": _su.get("operator") or "—",
                "reviewer": _su.get("approved_by") or "—",
                "date": _date.today().isoformat(),
                "equipment": "TorqueTrak 10K + NI 9229"}
        try:
            from core.modal.preliminary_report import build_preliminary_pdf
            _es = (cb_rp_lang.currentIndex() == 0)     # idioma elegido en el Report
            pdf = build_preliminary_pdf(meta=meta, quality=quality, sections=sections, analysis=analysis,
                                        findings=findings, recommendations=recs,
                                        run_id=f"TOR-{_asset}", lang=("es" if _es else "en"))
        except Exception as exc:  # noqa: BLE001
            rp_status.setText(f"❌ {type(exc).__name__}: {exc}"); return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(win, T("Save report", "Guardar reporte"),
                                                        f"Torsional_{_asset}.pdf", "PDF (*.pdf)")
        if not path:
            return
        with open(path, "wb") as fh:
            fh.write(pdf)
        rp_status.setText(T(f"✅ Saved: {path}", f"✅ Guardado: {path}"))
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(path))
    def _gen_monitor_report():
        mon = st.get("_mon")
        if mon is None or mon.n_samples == 0:
            rp_status.setText(T("No 24h campaign loaded. Run or load one in the Monitor 24h tab.",
                                "No hay campaña 24h cargada. Córrela o cárgala en la pestaña Monitor 24h.")); return
        _es = (cb_rp_lang.currentIndex() == 0)
        u = "N·m" if mon.units == "nm" else "ft-lb"
        s = mon.summary(); ranges = s.get("ranges", [])
        ncyc = sum(c for _, c in ranges); rmax = max((r for r, _ in ranges), default=0.0)
        dur_h = mon.duration_s / 3600.0
        _shaft = _mon_shaft_now()
        try:
            life = shaft_torsional_fatigue(mon.fatigue_cycles(), _shaft.get("do", 3.0), _shaft.get("di", 0.0),
                _shaft.get("sut_ksi", 90.0) * 1000.0, mon.units, window_seconds=max(mon.duration_s, 1.0),
                design_safety_factor=float(cb_ft_sf.currentText()),
                endurance_ratio=float(st.get("endurance_ratio", 0.50)))
        except Exception:  # noqa: BLE001
            life = None
        _sf = ("∞" if life and life.safety_factor == float("inf") else (f"{life.safety_factor:.2f}" if life else "—"))
        _gost = {"green": "GO", "yellow": "REVIEW", "red": "NO-GO"}.get(getattr(life, "status", ""), "—")
        _verd = (life.label_es if _es else life.label_en) if life else "—"
        _lifet = ("—" if life is None else (T("Infinite", "Infinita") if life.infinite else
                  (f"{life.life_hours:,.0f} h" if life.life_hours < 8760 else f"{life.life_hours/8760:,.1f} " + T("yr", "años"))))
        _bad = s.get("bad_seconds", 0.0)
        quality = [
            (T("Campaign", "Campaña"), "OK", T(f"{dur_h:.1f} h · {ncyc:,.0f} cycles", f"{dur_h:.1f} h · {ncyc:,.0f} ciclos")),
            (T("Signal health", "Salud de señal"), "REVIEW" if _bad > 0 else "GO",
             T(f"Dead time {_bad/3600.0:.2f} h", f"Tiempo muerto {_bad/3600.0:.2f} h")),
            (T("Shaft fatigue (Goodman/Miner)", "Fatiga del eje (Goodman/Miner)"), _gost,
             T(f"SF {_sf} — {life.label_en if life else '—'} · {_lifet}", f"FS {_sf} — {_verd} · {_lifet}")),
        ]
        analysis = [
            T(f"{dur_h:.1f} h monitored; mean torque {mon.mean_torque():,.0f} {u}, peak {mon.tmax or 0:,.0f} {u}.",
              f"{dur_h:.1f} h monitoreadas; par medio {mon.mean_torque():,.0f} {u}, pico {mon.tmax or 0:,.0f} {u}."),
            T(f"{ncyc:,.0f} rainflow cycles accumulated (ASTM E1049); largest range {rmax:,.0f} {u}.",
              f"{ncyc:,.0f} ciclos rainflow acumulados (ASTM E1049); mayor rango {rmax:,.0f} {u}."),
        ]
        findings = [
            T(f"Shaft fatigue over {dur_h:.1f} h: safety factor {_sf} — {life.label_en if life else '—'} ({_lifet}).",
              f"Fatiga del eje en {dur_h:.1f} h: factor de seguridad {_sf} — {_verd} ({_lifet})."),
            T(f"{len(mon.events)} overload event(s) logged during the campaign.",
              f"{len(mon.events)} evento(s) de sobrecarga registrados en la campaña."),
        ]
        recs = [T("Compare accumulated damage against the shaft S-N / Goodman diagram at the gage.",
                  "Comparar el daño acumulado contra el diagrama S-N / Goodman del eje en la galga."),
                T("Investigate overload events (startups / trips / process transients)." if mon.events else
                  "No overloads; keep the periodic monitoring plan.",
                  "Investigar los eventos de sobrecarga (arranques / trips / transitorios)." if mon.events else
                  "Sin sobrecargas; mantener el plan de monitoreo periódico.")]
        sections = [
            {"title": T("Torque trend (24h)", "Tendencia de par (24h)"),
             "figures": [(T("Torque pp vs time", "Par pp vs tiempo"), _grab_png(p_mtr))]},
            {"title": T("Accumulated fatigue (rainflow)", "Fatiga acumulada (rainflow)"),
             "figures": [(T("Rainflow histogram", "Histograma rainflow"), _grab_png(p_mh))],
             "table": {"headers": [T("Item", "Ítem"), T("Value", "Valor")],
                       "rows": [[T("Traffic light", "Semáforo"), f"{_gost} — {_verd}"],
                                [T("Safety factor", "Factor de seguridad"), _sf],
                                [T("Estimated life", "Vida estimada"), _lifet],
                                [T("Cycles", "Ciclos"), f"{ncyc:,.0f}"],
                                [T("Largest range", "Mayor rango"), f"{rmax:,.0f} {u}"]]}},
        ]
        _su = _mon_setup_now()
        _asset = _su.get("machine") or _su.get("tag") or "—"
        from datetime import date as _date
        meta = {"title": T("24h Torsional Monitoring Report", "Reporte de Monitoreo Torsional 24h"),
                "asset": _asset, "client": _su.get("client") or "—", "machine_type": _su.get("type") or "—",
                "location": _su.get("location") or "—", "test_type": T("24h monitoring", "Monitoreo 24h"),
                "rpm": f"{(mon.trend[-1][4] if mon.trend else 0):,.0f}", "technician": _su.get("operator") or "—",
                "reviewer": _su.get("approved_by") or "—", "date": _date.today().isoformat(),
                "equipment": "TorqueTrak 10K + NI 9229"}
        try:
            from core.modal.preliminary_report import build_preliminary_pdf
            pdf = build_preliminary_pdf(meta=meta, quality=quality, sections=sections, analysis=analysis,
                                        findings=findings, recommendations=recs,
                                        run_id=f"TOR-MON-{_asset}", lang=("es" if _es else "en"))
        except Exception as exc:  # noqa: BLE001
            rp_status.setText(f"❌ {type(exc).__name__}: {exc}"); return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(win, T("Save report", "Guardar reporte"),
                                                        f"Torsional_Monitor_{_asset}.pdf", "PDF (*.pdf)")
        if not path:
            return
        with open(path, "wb") as fh:
            fh.write(pdf)
        rp_status.setText(T(f"✅ Saved: {path}", f"✅ Guardado: {path}"))
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(path))

    btn_rp.clicked.connect(_gen_report)
    btn_rp_mon.clicked.connect(_gen_monitor_report)
    tabs.addTab(pg_rp, T("Report", "Reporte"))

    # =================================================================
    # TAB 8 — Updates (auto-actualización por red, como el Modal)
    # =================================================================
    pg_upd = QtWidgets.QWidget(); ul = QtWidgets.QVBoxLayout(pg_upd)
    ul.setContentsMargins(28, 24, 28, 24)
    _card = QtWidgets.QFrame()
    _card.setStyleSheet("QFrame{background:white;border:1px solid #e6ecf5;border-radius:16px;}")
    _card.setMinimumWidth(600); _card.setMaximumWidth(780)
    _cl = QtWidgets.QVBoxLayout(_card); _cl.setContentsMargins(34, 30, 34, 34); _cl.setSpacing(14)

    def _mkfont(pt, bold=False):
        f = QtGui.QFont(); f.setPointSize(pt); f.setBold(bold); return f

    _uh = QtWidgets.QLabel("🍉  Watermelon Torsional")
    _uh.setFont(_mkfont(16, True)); _uh.setStyleSheet(f"color:{NAVY};border:none;")
    _cur = QtWidgets.QLabel(T(f"Installed version:  v{__version__}", f"Versión instalada:  v{__version__}"))
    _cur.setFont(_mkfont(11)); _cur.setStyleSheet("color:#475569;border:none;")

    # --- Panel de licencia (cuenta + estado + equipo) — igual que el Modal ---
    _lic_lbl = QtWidgets.QLabel(""); _lic_lbl.setFont(_mkfont(10)); _lic_lbl.setWordWrap(True)
    _lic_lbl.setTextFormat(QtCore.Qt.RichText); _lic_lbl.setStyleSheet("color:#475569;border:none;")
    _lic_ok = False
    try:
        from core.modal import licensing as _lic
        _ls = _lic.local_license_status()
        _exp = _ls.get("exp")
        import datetime as _dt2
        _expd = _dt2.date.fromtimestamp(float(_exp)).isoformat() if _exp else "—"
        _stt = (f"<span style='color:#10b981'>● {T('Licensed','Con licencia')}</span>"
                if _ls.get("valid") else f"<span style='color:#ef4444'>● {T('Not activated','Sin activar')}</span>")
        _lic_lbl.setText(
            f"<b>{T('License','Licencia')}:</b> {_stt}<br>"
            f"{T('Account','Cuenta')}: {_ls.get('account') or '—'} · {T('Expires','Vence')}: {_expd}<br>"
            f"<b>{T('This computer','Este equipo')}:</b> {_lic.machine_label()}<br>"
            f"<span style='color:#94a3b8'>Machine ID: {_ls.get('fingerprint','')[:24]}…</span>")
        _lic_ok = bool(_ls.get("valid"))
    except Exception:  # noqa: BLE001
        _lic_lbl.setText("")

    _deact = QtWidgets.QPushButton(T("Deactivate this computer", "Desactivar este equipo"))
    _deact.setCursor(QtCore.Qt.PointingHandCursor); _deact.setFont(_mkfont(9))
    _deact.setStyleSheet("QPushButton{background:transparent;color:#ef4444;border:1px solid #f2c4c4;"
                         "border-radius:8px;padding:6px 14px;} QPushButton:hover{background:#fef2f2;}"
                         "QPushButton:disabled{color:#cbd5e1;border-color:#eef2f8;}")
    _deact.setEnabled(_lic_ok)
    _deact_row = QtWidgets.QHBoxLayout(); _deact_row.addWidget(_deact); _deact_row.addStretch(1)

    def _do_deactivate():
        m = QtWidgets.QMessageBox(_card)
        m.setIcon(QtWidgets.QMessageBox.Warning)
        m.setWindowTitle(T("Deactivate this computer", "Desactivar este equipo"))
        m.setText(T("Release this license from this computer?", "¿Liberar esta licencia de este equipo?"))
        m.setInformativeText(T("The app will require a license key again on next start. "
                               "You can then activate the license on another computer.",
                               "La app volverá a pedir la clave al iniciar. Luego podrás activar "
                               "la licencia en otro equipo."))
        m.setStandardButtons(QtWidgets.QMessageBox.Cancel | QtWidgets.QMessageBox.Yes)
        m.setDefaultButton(QtWidgets.QMessageBox.Cancel)
        if m.exec() != QtWidgets.QMessageBox.Yes:
            return
        _deact.setEnabled(False); _deact.setText(T("Deactivating…", "Desactivando…"))
        QtWidgets.QApplication.processEvents()
        try:
            from core.modal import licensing as _lic2
            r = _lic2.deactivate_machine()
        except Exception as e:  # noqa: BLE001
            r = {"ok": False, "reason": f"{type(e).__name__}: {e}"}
        if r.get("ok"):
            done = QtWidgets.QMessageBox(_card)
            done.setIcon(QtWidgets.QMessageBox.Information)
            done.setWindowTitle("Watermelon Torsional")
            done.setText(T("This computer was deactivated.", "Este equipo fue desactivado."))
            done.setInformativeText(T("The app will now close. Reopen it to activate a license.",
                                      "La app se cerrará. Ábrela de nuevo para activar una licencia."))
            done.exec()
            QtWidgets.QApplication.quit()
        else:
            _deact.setEnabled(True); _deact.setText(T("Deactivate this computer", "Desactivar este equipo"))
            QtWidgets.QMessageBox.warning(_card, "Watermelon Torsional",
                T("Could not deactivate: ", "No se pudo desactivar: ") + str(r.get("reason", "")))
    _deact.clicked.connect(_do_deactivate)

    _ustatus = QtWidgets.QLabel(T("Press <b>Check for updates</b> to see if a newer version is available.",
                                  "Pulsa <b>Buscar actualizaciones</b> para ver si hay una versión más nueva."))
    _ustatus.setFont(_mkfont(10)); _ustatus.setStyleSheet("color:#64748b;border:none;")
    _ustatus.setWordWrap(True); _ustatus.setTextFormat(QtCore.Qt.RichText)
    _unotes = QtWidgets.QTextBrowser(); _unotes.setFont(_mkfont(9))
    _unotes.setStyleSheet("QTextBrowser{border:1px solid #eef2f8;border-radius:10px;background:#fbfcfe;"
                          "color:#334155;padding:8px;}")
    _unotes.setMaximumHeight(200); _unotes.hide()
    _ubrow = QtWidgets.QPushButton(T("🔍  Check for updates", "🔍  Buscar actualizaciones")); _ubrow.setFont(_mkfont(11, True))
    _ubrow.setStyleSheet(f"QPushButton{{background:{NAVY};color:white;padding:10px 20px;"
                         "border-radius:9px;}QPushButton:hover{background:#12325a;}")
    _ubgo = QtWidgets.QPushButton(T("⬇  Update now", "⬇  Actualizar ahora")); _ubgo.setFont(_mkfont(11, True))
    _ubgo.setStyleSheet(f"QPushButton{{background:{GREEN};color:white;padding:10px 20px;"
                        "border-radius:9px;}QPushButton:hover{background:#12833a;}")
    _ubgo.hide(); _ubrow.setMinimumHeight(42); _ubgo.setMinimumHeight(42)
    urow = QtWidgets.QHBoxLayout(); urow.setSpacing(12)
    urow.addWidget(_ubrow); urow.addWidget(_ubgo); urow.addStretch(1)
    _ufoot = QtWidgets.QLabel(T(
        "Updates download and install automatically over the network; the app restarts when done. "
        "No files to send — the field PC just needs internet.",
        "Las actualizaciones se descargan e instalan automáticamente por red; la app se reinicia al terminar. "
        "Sin enviar archivos — el PC de campo solo necesita internet."))
    _ufoot.setFont(_mkfont(9)); _ufoot.setWordWrap(True); _ufoot.setStyleSheet("color:#94a3b8;border:none;")
    _cl.addWidget(_uh); _cl.addSpacing(2); _cl.addWidget(_cur)
    _cl.addWidget(_lic_lbl); _cl.addLayout(_deact_row); _cl.addSpacing(4); _cl.addWidget(_ustatus)
    _cl.addWidget(_unotes); _cl.addSpacing(8); _cl.addLayout(urow)
    _cl.addSpacing(6); _cl.addWidget(_ufoot)
    ul.addWidget(_card, 0, QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop); ul.addStretch(1)
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
    if st.get("load_config_fn"):
        st["load_config_fn"]()      # carga la configuración guardada (si existe)
    _rebuild_scaling()
    return app, win


def main(argv=None):
    ap = argparse.ArgumentParser(description="Watermelon Torsional — TorqueTrak 10K (native)")
    ap.add_argument("--sim", action="store_true", default=True)
    args = ap.parse_args(argv)
    try:
        # --- Gate de licencia (igual que el Modal de campo) ANTES de construir la UI ---
        import os as _os
        global _LANG
        _here = _os.path.dirname(_os.path.abspath(__file__))
        if _here not in sys.path:
            sys.path.insert(0, _here)      # que 'license_gate' sea importable (script y .exe)
        _LANG = _load_lang()
        _app0 = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
        try:
            from license_gate import run_license_gate
            _brand = ("<span style='color:#fff;font-weight:800;letter-spacing:3px;font-size:22px;'>WATERMELON</span>"
                      "<span style='color:#1AAEE5;font-weight:800;letter-spacing:3px;font-size:22px;'>&nbsp;TORSIONAL</span>")
            _ok = run_license_gate(_app0, t=T, navy=NAVY, acc=ACC, brand_html=_brand,
                                   app_title="Watermelon Torsional")
        except SystemExit:
            raise
        except Exception:  # noqa: BLE001 — fallo de la capa de licencia
            if _os.environ.get("WM_LICENSING", "1") == "0":
                _ok = True                 # modo dev: sin gate
            else:                          # producción: FAIL-CLOSED
                _e = traceback.format_exc()
                try:
                    QtWidgets.QMessageBox.critical(None, "Watermelon Torsional",
                        "Licensing error — the app cannot start.\n\n" + _e[-800:])
                except Exception:  # noqa: BLE001
                    print(_e)
                sys.exit(1)
        if not _ok:
            sys.exit(0)                    # el usuario canceló la activación
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
