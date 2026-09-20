"""
native/watermelon_balancing.py — Watermelon Balancing (app de campo)
====================================================================

App de campo para BALANCEO in-situ (1 y 2 planos) por coeficientes de
influencia. REUSA el motor PROBADO `core/balance/*` (engine, report, cloud) —
NO reimplementa fórmulas. Sigue el molde de Modal/Torsional [[module-creation-playbook]].

Fuentes de datos:
  · Manual — el cliente da el 1× (amplitud + fase) medido con otro equipo.
  · Simulado — desbalance de ejemplo para practicar.
  · (NI 9229 proximidad µm / NI 9234 acelerómetro→velocidad mm/s — próximo).

Norma de unidad: carcasa/absoluta (acelerómetro) → VELOCIDAD mm/s RMS
(ISO 20816); eje relativo (proximitor) → DESPLAZAMIENTO µm pp (API 670/ISO 7919).

Corre con licencia (modelo paquete). WM_LICENSING=0 salta el gate (dev).
"""
from __future__ import annotations

import argparse
import sys
import traceback

from PySide6 import QtCore, QtGui, QtWidgets

from core.balance.engine import (
    solve_1plane, solve_2plane, recommend_trial_weight_g,
    evaluate_iso_grades, to_complex, to_polar, calc_U_trial, calc_U_res_auto,
)
from core.balance.ni_balance import (
    extract_1x, one_x_accel_to_velocity, VibChannel, NIBalanceConfig, NIBalanceSource,
    keyphasor_power_note,
)
from core.torsional.ni_source import KeyphasorSensor, nidaqmx_available

__version__ = "0.4.0"

# Marca
NAVY = "#0f2a4a"; ACC = "#1AAEE5"; GREEN = "#16a34a"; AMBER = "#f59e0b"; RED = "#dc2626"

# --- idioma (bilingüe, como los otros módulos) ---
_LANG = "es"


def _load_lang() -> str:
    try:
        v = QtCore.QSettings("WatermelonSystem", "Balancing").value("lang", "es")
        return "en" if str(v).lower() == "en" else "es"
    except Exception:  # noqa: BLE001
        return "es"


def _save_lang(v: str) -> None:
    try:
        QtCore.QSettings("WatermelonSystem", "Balancing").setValue("lang", "es" if v == "es" else "en")
    except Exception:  # noqa: BLE001
        pass


def T(en: str, es: str) -> str:
    return es if _LANG == "es" else en


def _stylesheet() -> str:
    return f"""
    QWidget {{ font-size: 13px; color: #0f172a; }}
    QMainWindow, QWidget#page {{ background: #eef2f7; }}
    QGroupBox {{ background: white; border: 1px solid #dbe4f0; border-radius: 12px;
        margin-top: 12px; padding: 12px; font-weight: 700; }}
    QGroupBox::title {{ subcontrol-origin: margin; left: 12px; padding: 0 4px; color: {NAVY}; }}
    QPushButton {{ background: {NAVY}; color: white; border: none; border-radius: 9px;
        padding: 8px 14px; font-weight: 700; }}
    QPushButton:hover {{ background: #12325a; }}
    QLineEdit, QComboBox, QDoubleSpinBox, QSpinBox {{ background: white; border: 1px solid #cfdbec;
        border-radius: 8px; padding: 6px 8px; }}
    QTabBar::tab {{ padding: 8px 14px; }}
    """


def _dsb(minv, maxv, val, dec=2, suffix=""):
    w = QtWidgets.QDoubleSpinBox(); w.setRange(minv, maxv); w.setDecimals(dec); w.setValue(val)
    if suffix:
        w.setSuffix(suffix)
    return w


def build_app(simulated: bool = True):
    global _LANG
    _LANG = _load_lang()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(_stylesheet())
    win = QtWidgets.QMainWindow()
    win.setWindowTitle(f"Watermelon Balancing v{__version__}")
    win.resize(1180, 820)

    st = {"unit": "mils pk-pk", "r1p": None, "r2p": None, "iso": None,
          "setup_fn": None,
          "acq": {"mode": "sim", "vib_kind": "accel_9234",
                  "kph": KeyphasorSensor.phototach_reflective(),
                  "kph_device": "cDAQ1Mod1", "vib_device": "cDAQ1Mod2",
                  "sens": 100.0, "fs": 5120.0},
          # Simulador de rotor (para practicar el lazo completo sin equipo):
          # V_medida = alpha·(U_desbalance + pesos puestos).
          "sim": {"a1": to_complex(0.45, 35.0), "u1": to_complex(9.0, 110.0),
                  "aA1": to_complex(0.42, 20.0), "aA2": to_complex(0.14, 200.0),
                  "aB1": to_complex(0.11, 165.0), "aB2": to_complex(0.5, -25.0),
                  "uA": to_complex(7.0, 60.0), "uB": to_complex(5.0, 290.0)}}

    # ---- Toolbar: marca + versión + idioma ----
    tb = win.addToolBar("main"); tb.setMovable(False)
    tb.setStyleSheet(f"QToolBar {{ background: {NAVY}; padding: 7px 14px; }}")
    brand = QtWidgets.QLabel(
        "<span style='color:#fff;font-weight:800;letter-spacing:2.5px;font-size:16px;'>WATERMELON</span>"
        "<span style='color:#1AAEE5;font-weight:800;letter-spacing:2.5px;font-size:16px;'>&nbsp;BALANCING</span>")
    brand.setTextFormat(QtCore.Qt.RichText); tb.addWidget(brand)
    _ver = QtWidgets.QLabel(f"v{__version__}")
    _ver.setStyleSheet("color:#cbd5e1; background:#1e3a5f; border-radius:9px; padding:2px 10px; margin-left:12px;")
    tb.addWidget(_ver)
    _spc = QtWidgets.QWidget(); _spc.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
    tb.addWidget(_spc)

    def _switch_lang(new):
        if new == _LANG:
            return
        _save_lang(new)
        QtWidgets.QMessageBox.information(win, "Watermelon Balancing",
            T("Language set. Restart to apply.", "Idioma cambiado. Reinicia para aplicar."))
        try:
            QtCore.QProcess.startDetached(QtWidgets.QApplication.applicationFilePath(), sys.argv[1:])
        except Exception:  # noqa: BLE001
            pass
        app.quit()

    for _code in ("EN", "ES"):
        _b = QtWidgets.QToolButton(); _b.setText(_code); _b.setCheckable(True)
        _b.setChecked(_code.lower() == _LANG)
        _b.setStyleSheet("QToolButton{background:transparent;color:#8ea0bd;border:none;padding:3px 10px;font-weight:800;}"
                         "QToolButton:checked{background:#1AAEE5;color:#08243a;border-radius:6px;}")
        _b.clicked.connect(lambda _=0, c=_code.lower(): _switch_lang(c))
        tb.addWidget(_b)

    tabs = QtWidgets.QTabWidget(); win.setCentralWidget(tabs)

    def _scroll(inner):
        """Envuelve una página en scroll → nunca se pierde información en pantallas pequeñas."""
        sa = QtWidgets.QScrollArea(); sa.setWidgetResizable(True); sa.setFrameShape(QtWidgets.QFrame.NoFrame)
        sa.setWidget(inner); return sa

    # =================================================================
    # TAB 0 — Setup
    # =================================================================
    pg_set = QtWidgets.QWidget(); sl = QtWidgets.QVBoxLayout(pg_set)
    sl.addWidget(QtWidgets.QLabel(T("Machine identification — used in the report and the saved/uploaded run.",
                                    "Identificación de la máquina — se usa en el reporte y la corrida guardada/subida.")))
    gb = QtWidgets.QGroupBox(T("Machine / asset", "Máquina / activo")); f = QtWidgets.QFormLayout(gb)
    ed_machine = QtWidgets.QLineEdit(); ed_tag = QtWidgets.QLineEdit(); ed_client = QtWidgets.QLineEdit()
    ed_loc = QtWidgets.QLineEdit(); ed_operator = QtWidgets.QLineEdit(); ed_approved = QtWidgets.QLineEdit()
    sb_rpm = _dsb(0, 30000, 1800, 0, " rpm")
    cb_sensor = QtWidgets.QComboBox()
    cb_sensor.addItems([T("Casing (accelerometer → velocity, mm/s RMS)", "Carcasa (acelerómetro → velocidad, mm/s RMS)"),
                        T("Shaft (proximity, mils pk-pk)", "Eje (proximidad, mils pk-pk)")])
    for lbl, w in [(T("Machine", "Máquina"), ed_machine), (T("Tag", "Tag"), ed_tag),
                   (T("Client", "Cliente"), ed_client), (T("Location", "Ubicación"), ed_loc),
                   (T("Nameplate RPM", "RPM de placa"), sb_rpm),
                   (T("Vibration type", "Tipo de vibración"), cb_sensor),
                   (T("Operator", "Operador"), ed_operator), (T("Approved by", "Aprobado por"), ed_approved)]:
        f.addRow(lbl, w)
    sl.addWidget(gb)
    _lbl_unit = QtWidgets.QLabel(""); _lbl_unit.setStyleSheet(f"color:{NAVY};font-weight:700;")
    sl.addWidget(_lbl_unit)
    btn_saveset = QtWidgets.QPushButton(T("💾 Save setup", "💾 Guardar setup"))
    btn_saveset.setStyleSheet(f"QPushButton{{background:{GREEN};}}")
    _setmsg = QtWidgets.QLabel(""); _setmsg.setStyleSheet("color:#16a34a;font-weight:700;")
    _sr = QtWidgets.QHBoxLayout(); _sr.addWidget(btn_saveset); _sr.addWidget(_setmsg); _sr.addStretch(1)
    sl.addLayout(_sr); sl.addStretch(1)
    tabs.addTab(_scroll(pg_set), "Setup")

    _SET = QtCore.QSettings("WatermelonSystem", "BalancingSetup")

    def _unit():
        return "mm/s RMS" if cb_sensor.currentIndex() == 0 else "mils pk-pk"

    def _on_sensor(_=0):
        st["unit"] = _unit()
        _lbl_unit.setText(T(f"Balancing unit: {st['unit']}  ·  "
                            + ("casing/absolute → velocity (ISO 20816)" if cb_sensor.currentIndex() == 0
                               else "shaft/relative → displacement (API 670)"),
                            f"Unidad de balanceo: {st['unit']}  ·  "
                            + ("carcasa/absoluta → velocidad (ISO 20816)" if cb_sensor.currentIndex() == 0
                               else "eje/relativo → desplazamiento (API 670)")))
    cb_sensor.currentIndexChanged.connect(_on_sensor)

    def _setup_dict():
        return {"machine": ed_machine.text(), "tag": ed_tag.text(), "client": ed_client.text(),
                "location": ed_loc.text(), "nameplate_rpm": sb_rpm.value(),
                "operator": ed_operator.text(), "approved_by": ed_approved.text(),
                "sensor": cb_sensor.currentIndex(), "unit": _unit()}
    st["setup_fn"] = _setup_dict

    def _save_setup():
        for k, v in _setup_dict().items():
            _SET.setValue(k, v)
        _setmsg.setText(T("✅ Setup saved.", "✅ Setup guardado."))

    def _load_setup():
        if _SET.value("machine") is None:
            return
        ed_machine.setText(str(_SET.value("machine", "") or "")); ed_tag.setText(str(_SET.value("tag", "") or ""))
        ed_client.setText(str(_SET.value("client", "") or "")); ed_loc.setText(str(_SET.value("location", "") or ""))
        ed_operator.setText(str(_SET.value("operator", "") or "")); ed_approved.setText(str(_SET.value("approved_by", "") or ""))
        try: sb_rpm.setValue(float(_SET.value("nameplate_rpm", 1800) or 1800))
        except Exception: pass  # noqa: BLE001
        try: cb_sensor.setCurrentIndex(int(_SET.value("sensor", 0)))
        except Exception: pass  # noqa: BLE001
    btn_saveset.clicked.connect(_save_setup)

    # =================================================================
    # TAB — Configuration (adquisición: manual / simulado / NI)
    # =================================================================
    pg_cfg = QtWidgets.QWidget(); cl = QtWidgets.QVBoxLayout(pg_cfg)
    cl.addWidget(QtWidgets.QLabel(T(
        "Data source for the 1× vibration. Manual = the client gives you the 1× (amplitude+phase) "
        "from another instrument. Simulated = practice the full loop. NI = capture live from the card.",
        "Fuente del 1× de vibración. Manual = el cliente te da el 1× (amplitud+fase) de otro equipo. "
        "Simulado = practicar el lazo completo. NI = capturar en vivo de la tarjeta.")))
    gb_acq = QtWidgets.QGroupBox(T("Acquisition", "Adquisición")); fa = QtWidgets.QFormLayout(gb_acq)
    cb_planes = QtWidgets.QComboBox(); cb_planes.addItems([T("1 plane", "1 plano"), T("2 planes", "2 planos")])
    fa.addRow(T("Balancing planes", "Planos de balanceo"), cb_planes)
    cb_mode = QtWidgets.QComboBox()
    cb_mode.addItems([T("Manual (typed)", "Manual (escrito)"), T("Simulated", "Simulado"),
                      "NI 9229 (proximity µm)", "NI 9234 (accel → velocity mm/s)"])
    cb_mode.setCurrentIndex(1)
    cb_kph = QtWidgets.QComboBox()
    cb_kph.addItems([T("Photo-tach (reflective tape)", "Foto-tacómetro (cinta reflectiva)"),
                     "Bently 3300 XL 8mm + Proximitor"])
    sb_ppr = QtWidgets.QSpinBox(); sb_ppr.setRange(1, 60); sb_ppr.setValue(1)
    ed_kphdev = QtWidgets.QLineEdit("cDAQ1Mod1"); ed_vibdev = QtWidgets.QLineEdit("cDAQ1Mod2")
    sb_sens = _dsb(1, 5000, 100.0, 1, " mV/unit")
    lbl_pow = QtWidgets.QLabel(""); lbl_pow.setWordWrap(True); lbl_pow.setStyleSheet("color:#b45309;font-size:11px;")
    fa.addRow(T("Mode", "Modo"), cb_mode)
    fa.addRow(T("Keyphasor sensor (always channel 0)", "Sensor keyphasor (siempre canal 0)"), cb_kph)
    fa.addRow(T("Pulses per rev", "Pulsos por vuelta"), sb_ppr)
    fa.addRow(T("Keyphasor device", "Device keyphasor"), ed_kphdev)
    fa.addRow(T("Vibration device", "Device vibración"), ed_vibdev)
    fa.addRow(T("Sensitivity", "Sensibilidad"), sb_sens)
    fa.addRow("", lbl_pow)
    cl.addWidget(gb_acq)

    # --- Datos del rotor / balanceo (todo lo que necesita el cálculo) ---
    gb_rotor = QtWidgets.QGroupBox(T("Rotor / balance data", "Datos del rotor / balanceo")); fr = QtWidgets.QFormLayout(gb_rotor)
    iso_w = _dsb(0.01, 1e5, 500.0, 2, " kg")     # masa del rotor
    tw_w = _dsb(0.01, 1e5, 250.0, 2, " kg")      # carga por plano (≈ masa/2)
    tw_r = _dsb(1, 5000, 150.0, 1, " mm")        # radio de balanceo
    tw_k = _dsb(0.2, 2.0, 1.25, 2)               # factor peso de prueba
    iso_g = QtWidgets.QComboBox(); iso_g.addItems(["0.4", "1.0", "2.5", "6.3", "16.0"]); iso_g.setCurrentText("2.5")
    fr.addRow(T("Rotor mass", "Masa del rotor"), iso_w)
    fr.addRow(T("Load per plane", "Carga por plano"), tw_w)
    fr.addRow(T("Balance radius", "Radio de balanceo"), tw_r)
    fr.addRow(T("Trial factor k", "Factor de prueba k"), tw_k)
    fr.addRow(T("ISO grade G", "Grado ISO G"), iso_g)
    cl.addWidget(gb_rotor)

    def _autoload(_=0):
        tw_w.setValue(iso_w.value() / 2.0)       # carga por plano ≈ mitad del rotor (entre 2 cojinetes)
    iso_w.valueChanged.connect(_autoload)

    btn_savecfg = QtWidgets.QPushButton(T("💾 Save configuration", "💾 Guardar configuración"))
    btn_savecfg.setStyleSheet(f"QPushButton{{background:{GREEN};}}")
    _cfgmsg = QtWidgets.QLabel(""); _cfgmsg.setStyleSheet("color:#16a34a;font-weight:700;")
    _cr = QtWidgets.QHBoxLayout(); _cr.addWidget(btn_savecfg); _cr.addWidget(_cfgmsg); _cr.addStretch(1)
    cl.addLayout(_cr); cl.addStretch(1)
    tabs.addTab(_scroll(pg_cfg), "Configuration")

    _CFG = QtCore.QSettings("WatermelonSystem", "BalancingConfig")

    def _save_config():
        _CFG.setValue("planes", cb_planes.currentIndex()); _CFG.setValue("mode", cb_mode.currentIndex())
        _CFG.setValue("kph", cb_kph.currentIndex()); _CFG.setValue("ppr", sb_ppr.value())
        _CFG.setValue("kphdev", ed_kphdev.text()); _CFG.setValue("vibdev", ed_vibdev.text())
        _CFG.setValue("sens", sb_sens.value())
        _CFG.setValue("mass", iso_w.value()); _CFG.setValue("load", tw_w.value())
        _CFG.setValue("radius", tw_r.value()); _CFG.setValue("k", tw_k.value()); _CFG.setValue("G", iso_g.currentIndex())
        _cfgmsg.setText(T("✅ Configuration saved.", "✅ Configuración guardada."))

    def _load_config():
        if _CFG.value("mode") is None:
            return
        try:
            cb_planes.setCurrentIndex(int(_CFG.value("planes", 0)))
            cb_mode.setCurrentIndex(int(_CFG.value("mode", 1)))
            cb_kph.setCurrentIndex(int(_CFG.value("kph", 0))); sb_ppr.setValue(int(_CFG.value("ppr", 1)))
            ed_kphdev.setText(str(_CFG.value("kphdev", "cDAQ1Mod1") or "cDAQ1Mod1"))
            ed_vibdev.setText(str(_CFG.value("vibdev", "cDAQ1Mod2") or "cDAQ1Mod2"))
            sb_sens.setValue(float(_CFG.value("sens", 100.0) or 100.0))
            iso_w.setValue(float(_CFG.value("mass", 500.0) or 500.0)); tw_w.setValue(float(_CFG.value("load", 250.0) or 250.0))
            tw_r.setValue(float(_CFG.value("radius", 150.0) or 150.0)); tw_k.setValue(float(_CFG.value("k", 1.25) or 1.25))
            iso_g.setCurrentIndex(int(_CFG.value("G", 2)))
        except Exception:  # noqa: BLE001
            pass
    btn_savecfg.clicked.connect(_save_config)

    def _kph_sensor():
        idx = cb_kph.currentIndex(); ppr = sb_ppr.value()
        s = (KeyphasorSensor.bently_3300xl_8mm(keyways=ppr) if idx == 1
             else KeyphasorSensor.phototach_reflective(strips=ppr))
        st["acq"]["kph"] = s
        lbl_pow.setText("⚡ " + keyphasor_power_note(s))
        return s

    def _on_mode(_=0):
        i = cb_mode.currentIndex()
        st["acq"]["mode"] = ["manual", "sim", "ni_prox", "ni_accel"][i]
        st["acq"]["vib_kind"] = "proximity_9229" if i == 2 else "accel_9234"
        st["acq"]["kph_device"] = ed_kphdev.text() or "cDAQ1Mod1"
        st["acq"]["vib_device"] = ed_vibdev.text() or "cDAQ1Mod2"
        st["acq"]["sens"] = sb_sens.value()
        # la unidad de balanceo la fija el sensor (prox µm / accel→velocidad mm/s)
        if i == 2:
            st["unit"] = "mils pk-pk"; cb_sensor.setCurrentIndex(1)
        elif i == 3:
            st["unit"] = "mm/s RMS"; cb_sensor.setCurrentIndex(0)
    for _w in (cb_kph, sb_ppr):
        _w.currentIndexChanged.connect(lambda _=0: _kph_sensor()) if hasattr(_w, "currentIndexChanged") else \
            _w.valueChanged.connect(lambda _=0: _kph_sensor())
    sb_ppr.valueChanged.connect(lambda _=0: _kph_sensor())
    cb_mode.currentIndexChanged.connect(_on_mode)
    ed_kphdev.editingFinished.connect(_on_mode); ed_vibdev.editingFinished.connect(_on_mode)
    sb_sens.valueChanged.connect(lambda _=0: _on_mode())
    _kph_sensor(); _on_mode()

    # ---------- captura del 1× (simulado o NI) ----------
    def _ni_capture(n_vib):
        """Snapshot de NI: lee ~2 s, devuelve lista [(mag,ang)] por canal de vibración + rpm.
        None si falla (sin driver/hardware) — nunca simula en silencio."""
        acq = st["acq"]; fs = acq["fs"]
        chans = [VibChannel(f"V{i}", kind=acq["vib_kind"], device=acq["vib_device"], ai=i,
                            sensitivity_mv_per_unit=acq["sens"]) for i in range(n_vib)]
        cfg = NIBalanceConfig(keyphasor=acq["kph"], kph_device=acq["kph_device"], kph_ai=0,
                              vib_channels=chans, sample_rate_hz=fs, block_seconds=0.5)
        src = NIBalanceSource(cfg)
        try:
            src.start()
        except RuntimeError as exc:
            QtWidgets.QMessageBox.critical(win, "Watermelon Balancing",
                T(f"NI capture failed:\n{exc}", f"Falló la captura NI:\n{exc}")); return None
        import numpy as np
        try:
            data = np.concatenate([src.read_block() for _ in range(4)], axis=1)   # ~2 s
        finally:
            src.stop()
        kph = data[0]; out = []
        is_prox = acq["vib_kind"] == "proximity_9229"
        for i in range(n_vib):
            # extract_1x en 0-pico de la señal cruda del canal.
            amp0, ph, rpm = extract_1x(data[1 + i], kph, fs, acq["kph"], to_pp=False)
            if is_prox:
                # canal de voltaje → desplazamiento µm pp (sens en mV/µm). Verificar en campo.
                um0pk = amp0 * 1000.0 / max(acq["sens"], 1e-6)
                out.append((um0pk * 2.0, ph))          # mils pk-pk
            else:
                # nidaqmx accel chan devuelve g → m/s² → velocidad 1× RMS (mm/s).
                vamp0pk, vph = one_x_accel_to_velocity(amp0 * 9.80665, ph, rpm)
                out.append((vamp0pk / (2.0 ** 0.5), vph))   # mm/s RMS
        return out

    def _sim_measure_1p(w_complex):
        s = st["sim"]; z = s["a1"] * (s["u1"] + w_complex)
        return to_polar(z)

    def _sim_measure_2p(w1, w2):
        s = st["sim"]
        za = s["aA1"] * (s["uA"] + w1) + s["aA2"] * (s["uB"] + w2)
        zb = s["aB1"] * (s["uA"] + w1) + s["aB2"] * (s["uB"] + w2)
        return to_polar(za), to_polar(zb)

    # ---------- helper: par de campos (magnitud ∠ ángulo) ----------
    def _vec_row(form, label, magmax=1e6, magdef=0.0, unit_suffix=""):
        mag = _dsb(0, magmax, magdef, 3); ang = _dsb(0, 360, 0.0, 1, "°")
        row = QtWidgets.QWidget(); h = QtWidgets.QHBoxLayout(row); h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(mag); h.addWidget(QtWidgets.QLabel("∠")); h.addWidget(ang); h.addStretch(1)
        form.addRow(label, row)
        return mag, ang

    # =================================================================
    # TAB 1 — Trial weight (API 684) + ISO
    # =================================================================
    pg_tw = QtWidgets.QWidget(); twl = QtWidgets.QVBoxLayout(pg_tw)
    twl.addWidget(QtWidgets.QLabel(T("Uses the rotor data from Configuration (mass, load, radius, k, G) + RPM from Setup.",
                                     "Usa los datos del rotor de Configuration (masa, carga, radio, k, G) + RPM del Setup.")))
    gb_tw = QtWidgets.QGroupBox(T("Trial weight (API 684)", "Peso de prueba (API 684)")); ftw = QtWidgets.QFormLayout(gb_tw)
    btn_tw = QtWidgets.QPushButton(T("Compute suggested trial weight", "Calcular peso de prueba sugerido"))
    ftw.addRow("", btn_tw)
    tw_out = QtWidgets.QLabel("—"); tw_out.setStyleSheet(f"color:{NAVY};font-weight:800;font-size:15px;"); ftw.addRow("", tw_out)
    twl.addWidget(gb_tw)

    def _do_tw():
        Wt, Ut = recommend_trial_weight_g(tw_w.value(), sb_rpm.value(), tw_r.value(), tw_k.value())
        tw_out.setText(T(f"→ {Wt:,.2f} g  (U_trial {Ut:,.0f} g·mm)  @ {tw_r.value():.0f} mm",
                         f"→ {Wt:,.2f} g  (U_prueba {Ut:,.0f} g·mm)  @ {tw_r.value():.0f} mm"))
    btn_tw.clicked.connect(_do_tw)

    gb_iso = QtWidgets.QGroupBox(T("ISO 21940 quality check", "Chequeo de calidad ISO 21940")); fiso = QtWidgets.QFormLayout(gb_iso)
    iso_ures = _dsb(0, 1e7, 0.0, 1, " g·mm")
    fiso.addRow(T("Residual U_res (measured)", "Residual U_res (medido)"), iso_ures)
    btn_iso = QtWidgets.QPushButton(T("Evaluate ISO", "Evaluar ISO")); fiso.addRow("", btn_iso)
    iso_out = QtWidgets.QLabel("—"); iso_out.setWordWrap(True); iso_out.setStyleSheet("font-weight:700;"); fiso.addRow("", iso_out)
    twl.addWidget(gb_iso); twl.addStretch(1)

    def _do_iso():
        res = evaluate_iso_grades(iso_w.value(), sb_rpm.value(), iso_ures.value())
        st["iso"] = res
        g = float(iso_g.currentText())
        per = next((x for x in res["results"] if abs(x["G"] - g) < 1e-9), None)
        uper = per["U_per"] if per else 0.0
        ok = per["pass"] if per else None
        _v = "#16a34a" if ok else "#dc2626"
        iso_out.setText(
            f"<b>G{g:g}:</b> U_per {uper:,.0f} g·mm · residual {iso_ures.value():,.0f} g·mm → "
            f"<span style='color:{_v};font-weight:800'>"
            + (T("PASS", "CUMPLE") if ok else T("FAIL", "NO CUMPLE") if ok is not None else "—")
            + f"</span><br><span style='color:#64748b'>{res.get('summary_label','')}</span>")
    btn_iso.clicked.connect(_do_iso)
    tabs.addTab(_scroll(pg_tw), T("Trial + ISO", "Prueba + ISO"))

    # =================================================================
    # TAB 2 — 1 plano
    # =================================================================
    pg1 = QtWidgets.QWidget(); l1 = QtWidgets.QVBoxLayout(pg1)
    l1.addWidget(QtWidgets.QLabel(T("Single-plane balancing (influence coefficient, ISO 21940-12). "
                                    "Enter the 1× vibration (amplitude ∠ phase) and the trial weight.",
                                    "Balanceo en 1 plano (coef. de influencia, ISO 21940-12). "
                                    "Ingresa el 1× (amplitud ∠ fase) y el peso de prueba.")))
    gb1 = QtWidgets.QGroupBox(T("Measurements", "Mediciones")); f1 = QtWidgets.QFormLayout(gb1)
    v0m, v0a = _vec_row(f1, T("V0 — initial vibration", "V0 — vibración inicial"))
    twm, twa = _vec_row(f1, T("Trial weight (g ∠°)", "Peso de prueba (g ∠°)"), 1e5, 10.0)
    vtm, vta = _vec_row(f1, T("Vt — with trial weight", "Vt — con peso de prueba"))
    vfm, vfa = _vec_row(f1, T("Vf — final (optional)", "Vf — final (opcional)"))
    l1.addWidget(gb1)
    lbl_sugg1 = QtWidgets.QLabel(""); lbl_sugg1.setWordWrap(True); lbl_sugg1.setStyleSheet("color:#0f2a4a;font-size:12px;")
    l1.addWidget(lbl_sugg1)
    _cap1r = QtWidgets.QHBoxLayout()
    btn1_cref = QtWidgets.QPushButton(T("📷 Capture reference (V0)", "📷 Capturar referencia (V0)"))
    btn1_ctrial = QtWidgets.QPushButton(T("📷 Capture with trial (Vt)", "📷 Capturar con prueba (Vt)"))
    _cap1r.addWidget(btn1_cref); _cap1r.addWidget(btn1_ctrial); _cap1r.addStretch(1); l1.addLayout(_cap1r)
    _row1 = QtWidgets.QHBoxLayout()
    btn1 = QtWidgets.QPushButton(T("▶ Solve 1-plane", "▶ Resolver 1 plano"))
    btn1.setStyleSheet(f"QPushButton{{background:{GREEN};}}")
    btn1_demo = QtWidgets.QPushButton(T("Simulated example", "Ejemplo simulado"))
    _row1.addWidget(btn1); _row1.addWidget(btn1_demo); _row1.addStretch(1); l1.addLayout(_row1)

    def _cap1(with_trial):
        mode = st["acq"]["mode"]
        if mode == "manual":
            out1.setText(T("Manual mode — type the vectors, or switch mode in Configuration.",
                           "Modo manual — escribe los vectores, o cambia el modo en Configuration.")); return
        if mode.startswith("ni"):
            r = _ni_capture(1)
            if not r:
                return
            mag, ang = r[0]
        else:
            w = to_complex(twm.value(), twa.value()) if with_trial else 0 + 0j
            mag, ang = _sim_measure_1p(w)
        if with_trial:
            vtm.setValue(mag); vta.setValue(ang)
        else:
            v0m.setValue(mag); v0a.setValue(ang)
    btn1_cref.clicked.connect(lambda: _cap1(False)); btn1_ctrial.clicked.connect(lambda: _cap1(True))
    out1 = QtWidgets.QLabel("—"); out1.setWordWrap(True)
    out1.setStyleSheet("background:white;border:1px solid #dbe4f0;border-radius:10px;padding:12px;font-size:14px;")
    l1.addWidget(out1); l1.addStretch(1)

    def _solve1():
        try:
            r = solve_1plane(v0m.value(), v0a.value(), vtm.value(), vta.value(), twm.value(), twa.value())
        except ValueError as e:
            out1.setText(f"⚠ {e}"); return
        st["r1p"] = {"unit": st["unit"], "v0": (v0m.value(), v0a.value()),
                     "trial": (twm.value(), twa.value()), "vt": (vtm.value(), vta.value()),
                     "vf": ((vfm.value(), vfa.value()) if vfm.value() > 0 else None), "result": r}
        u = st["unit"]
        out1.setText(T(
            f"<b>Correction weight:</b> {r['corr_mass_g']:,.2f} g ∠ {r['corr_ang_deg']:.1f}°<br>"
            f"Predicted residual: {r['pred_mag']:.3f} {u} · model {r['quality']}<br>"
            f"<span style='color:#64748b'>{r['note']}</span>",
            f"<b>Peso de corrección:</b> {r['corr_mass_g']:,.2f} g ∠ {r['corr_ang_deg']:.1f}°<br>"
            f"Residual predicho: {r['pred_mag']:.3f} {u} · modelo {r['quality']}<br>"
            f"<span style='color:#64748b'>{r['note']}</span>"))

    def _demo1():
        v0m.setValue(8.60); v0a.setValue(63.0); twm.setValue(10.0); twa.setValue(0.0)
        vtm.setValue(6.50); vta.setValue(206.0); _solve1()
    btn1.clicked.connect(_solve1); btn1_demo.clicked.connect(_demo1)
    tabs.addTab(_scroll(pg1), T("1 plane", "1 plano"))

    # =================================================================
    # TAB 3 — 2 planos
    # =================================================================
    pg2 = QtWidgets.QWidget(); l2 = QtWidgets.QVBoxLayout(pg2)
    l2.addWidget(QtWidgets.QLabel(T("Two-plane balancing. Runs: 0 initial · 1 trial in plane A · 2 trial in plane B. "
                                    "Enter A/B vibration for each run and the trial weights.",
                                    "Balanceo en 2 planos. Corridas: 0 inicial · 1 prueba en plano A · 2 prueba en plano B. "
                                    "Ingresa la vibración A/B de cada corrida y los pesos de prueba.")))
    gb2 = QtWidgets.QGroupBox(T("Measurements", "Mediciones")); f2 = QtWidgets.QFormLayout(gb2)
    a0m, a0a = _vec_row(f2, "A0 — A inicial"); b0m, b0a = _vec_row(f2, "B0 — B inicial")
    wam, waa = _vec_row(f2, T("Trial A (g ∠°)", "Prueba A (g ∠°)"), 1e5, 10.0)
    a1m, a1a = _vec_row(f2, "A1 — A (trial A)"); b1m, b1a = _vec_row(f2, "B1 — B (trial A)")
    wbm, wba = _vec_row(f2, T("Trial B (g ∠°)", "Prueba B (g ∠°)"), 1e5, 10.0)
    a2m, a2a = _vec_row(f2, "A2 — A (trial B)"); b2m, b2a = _vec_row(f2, "B2 — B (trial B)")
    l2.addWidget(gb2)
    _cap2r = QtWidgets.QHBoxLayout()
    b2_ref = QtWidgets.QPushButton(T("📷 Reference", "📷 Referencia"))
    b2_ta = QtWidgets.QPushButton(T("📷 Trial A", "📷 Prueba A"))
    b2_tb = QtWidgets.QPushButton(T("📷 Trial B", "📷 Prueba B"))
    _cap2r.addWidget(b2_ref); _cap2r.addWidget(b2_ta); _cap2r.addWidget(b2_tb); _cap2r.addStretch(1)
    l2.addLayout(_cap2r)
    lbl_sugg2 = QtWidgets.QLabel(""); lbl_sugg2.setWordWrap(True); lbl_sugg2.setStyleSheet("color:#0f2a4a;font-size:12px;")
    l2.addWidget(lbl_sugg2)
    _srow2 = QtWidgets.QHBoxLayout()
    btn2_demo = QtWidgets.QPushButton(T("Simulated example", "Ejemplo simulado"))
    _srow2.addWidget(btn2_demo); _srow2.addStretch(1); l2.addLayout(_srow2)
    btn2 = QtWidgets.QPushButton(T("▶ Solve 2-plane", "▶ Resolver 2 planos"))
    btn2.setStyleSheet(f"QPushButton{{background:{GREEN};}}")
    l2.addWidget(btn2)

    def _cap2(run):
        mode = st["acq"]["mode"]
        if mode == "manual":
            out2.setText(T("Manual mode — type the vectors.", "Modo manual — escribe los vectores.")); return
        if mode.startswith("ni"):
            r = _ni_capture(2)
            if not r:
                return
            (amag, aang), (bmag, bang) = r[0], r[1]
        else:
            w1 = to_complex(wam.value(), waa.value()) if run == "trialA" else 0 + 0j
            w2 = to_complex(wbm.value(), wba.value()) if run == "trialB" else 0 + 0j
            (amag, aang), (bmag, bang) = _sim_measure_2p(w1, w2)
        if run == "ref":
            a0m.setValue(amag); a0a.setValue(aang); b0m.setValue(bmag); b0a.setValue(bang)
        elif run == "trialA":
            a1m.setValue(amag); a1a.setValue(aang); b1m.setValue(bmag); b1a.setValue(bang)
        else:
            a2m.setValue(amag); a2a.setValue(aang); b2m.setValue(bmag); b2a.setValue(bang)
    b2_ref.clicked.connect(lambda: _cap2("ref")); b2_ta.clicked.connect(lambda: _cap2("trialA"))
    b2_tb.clicked.connect(lambda: _cap2("trialB"))
    out2 = QtWidgets.QLabel("—"); out2.setWordWrap(True)
    out2.setStyleSheet("background:white;border:1px solid #dbe4f0;border-radius:10px;padding:12px;font-size:14px;")
    l2.addWidget(out2); l2.addStretch(1)

    def _solve2():
        cx = to_complex
        try:
            r = solve_2plane(cx(a0m.value(), a0a.value()), cx(b0m.value(), b0a.value()),
                             cx(a1m.value(), a1a.value()), cx(b1m.value(), b1a.value()),
                             cx(a2m.value(), a2a.value()), cx(b2m.value(), b2a.value()),
                             cx(wam.value(), waa.value()), cx(wbm.value(), wba.value()))
        except ValueError as e:
            out2.setText(f"⚠ {e}"); return
        st["r2p"] = {"unit": st["unit"],
                     "a0": (a0m.value(), a0a.value()), "b0": (b0m.value(), b0a.value()),
                     "a1": (a1m.value(), a1a.value()), "b1": (b1m.value(), b1a.value()),
                     "a2": (a2m.value(), a2a.value()), "b2": (b2m.value(), b2a.value()),
                     "wa": (wam.value(), waa.value()), "wb": (wbm.value(), wba.value()), "result": r}
        wca_m, wca_a = to_polar(r["WA_corr"]); wcb_m, wcb_a = to_polar(r["WB_corr"])
        aa_m, _ = to_polar(r["A_after"]); ba_m, _ = to_polar(r["B_after"]); u = st["unit"]
        out2.setText(T(
            f"<b>Plane A correction:</b> {wca_m:,.2f} g ∠ {wca_a:.1f}°<br>"
            f"<b>Plane B correction:</b> {wcb_m:,.2f} g ∠ {wcb_a:.1f}°<br>"
            f"Predicted residual: A {aa_m:.3f} · B {ba_m:.3f} {u} · model {r['quality']}",
            f"<b>Corrección plano A:</b> {wca_m:,.2f} g ∠ {wca_a:.1f}°<br>"
            f"<b>Corrección plano B:</b> {wcb_m:,.2f} g ∠ {wcb_a:.1f}°<br>"
            f"Residual predicho: A {aa_m:.3f} · B {ba_m:.3f} {u} · modelo {r['quality']}"))
    btn2.clicked.connect(_solve2)

    def _demo2():
        wt = to_complex(wam.value() or 10.0, waa.value()); wt2 = to_complex(wbm.value() or 10.0, wba.value())
        wam.setValue(to_polar(wt)[0]); wbm.setValue(to_polar(wt2)[0])
        (am, aa), (bm, bb) = _sim_measure_2p(0, 0)
        a0m.setValue(am); a0a.setValue(aa); b0m.setValue(bm); b0a.setValue(bb)
        (am, aa), (bm, bb) = _sim_measure_2p(wt, 0)
        a1m.setValue(am); a1a.setValue(aa); b1m.setValue(bm); b1a.setValue(bb)
        (am, aa), (bm, bb) = _sim_measure_2p(0, wt2)
        a2m.setValue(am); a2a.setValue(aa); b2m.setValue(bm); b2a.setValue(bb)
        _solve2()
    btn2_demo.clicked.connect(_demo2)
    tabs.addTab(_scroll(pg2), T("2 planes", "2 planos"))

    # =================================================================
    # TAB 4 — Report + save/cloud
    # =================================================================
    pg_rp = QtWidgets.QWidget(); rl = QtWidgets.QVBoxLayout(pg_rp)
    rl.addWidget(QtWidgets.QLabel(T("Preliminary balancing report (PDF) + save local / upload to cloud. "
                                    "Uses the last 1-plane / 2-plane / ISO results and the Setup data.",
                                    "Reporte preliminar de balanceo (PDF) + guardar local / subir a la nube. "
                                    "Usa los últimos resultados de 1/2 planos e ISO y los datos del Setup.")))
    _rlang = QtWidgets.QHBoxLayout(); _rlang.addWidget(QtWidgets.QLabel(T("Report language", "Idioma del reporte")))
    cb_rlang = QtWidgets.QComboBox(); cb_rlang.addItems(["Español", "English"]); cb_rlang.setCurrentIndex(0 if _LANG == "es" else 1)
    _rlang.addWidget(cb_rlang); _rlang.addStretch(1); rl.addLayout(_rlang)
    _rr = QtWidgets.QHBoxLayout()
    btn_pdf = QtWidgets.QPushButton(T("📄 Generate report (PDF)", "📄 Generar reporte (PDF)"))
    btn_pdf.setStyleSheet(f"QPushButton{{background:{GREEN};}}")
    btn_savelocal = QtWidgets.QPushButton(T("💾 Save locally", "💾 Guardar local"))
    btn_upload = QtWidgets.QPushButton(T("☁ Upload to cloud", "☁ Subir a la nube"))
    _rr.addWidget(btn_pdf); _rr.addWidget(btn_savelocal); _rr.addWidget(btn_upload); _rr.addStretch(1)
    rl.addLayout(_rr)
    rp_status = QtWidgets.QLabel(""); rp_status.setWordWrap(True); rp_status.setStyleSheet("color:#475569;")
    rl.addWidget(rp_status); rl.addStretch(1)

    def _meta():
        s = st["setup_fn"]() if st.get("setup_fn") else {}
        from datetime import date as _date
        return {"asset": s.get("machine") or s.get("tag") or "—", "client": s.get("client") or "—",
                "location": s.get("location") or "—", "specialist": s.get("operator") or "—",
                "reviewer": s.get("approved_by") or "—", "unit": st["unit"],
                "rpm": s.get("nameplate_rpm") or 0, "report_date": _date.today().strftime("%d/%m/%Y"),
                "rotation": "CCW"}

    def _build_pdf():
        if not st.get("r1p") and not st.get("r2p"):
            rp_status.setText(T("Solve a 1-plane or 2-plane case first.", "Resuelve un caso de 1 o 2 planos primero.")); return None
        try:
            from core.balance.report import build_balance_pdf
            pdf = build_balance_pdf(meta=_meta(), one_plane=st.get("r1p"), two_plane=st.get("r2p"), iso=st.get("iso"))
            return pdf
        except Exception as exc:  # noqa: BLE001
            rp_status.setText(f"❌ {type(exc).__name__}: {exc}"); return None

    def _gen_pdf():
        pdf = _build_pdf()
        if not pdf:
            return
        _asset = _meta()["asset"]
        path, _ = QtWidgets.QFileDialog.getSaveFileName(win, T("Save report", "Guardar reporte"),
                                                        f"Balanceo_{_asset}.pdf", "PDF (*.pdf)")
        if not path:
            return
        with open(path, "wb") as fh:
            fh.write(pdf)
        rp_status.setText(T(f"✅ Saved: {path}", f"✅ Guardado: {path}"))
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(path))

    def _payload():
        return {"kind": "balance", "setup": st["setup_fn"]() if st.get("setup_fn") else {},
                "unit": st["unit"], "one_plane": st.get("r1p"), "two_plane": st.get("r2p"),
                "iso": st.get("iso"), "app_version": __version__}

    def _save_local():
        if not st.get("r1p") and not st.get("r2p"):
            rp_status.setText(T("Nothing to save yet.", "Nada que guardar aún.")); return
        import os, json
        from datetime import datetime
        d = os.path.join(os.path.expanduser("~"), "WatermelonBalancing", "runs",
                         datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        os.makedirs(d, exist_ok=True)
        fp = os.path.join(d, "balance.json")
        with open(fp, "w", encoding="utf-8") as fh:
            json.dump(_payload(), fh, ensure_ascii=False)
        rp_status.setText(T(f"💾 Saved locally: {fp}", f"💾 Guardado local: {fp}"))

    def _upload():
        if not st.get("r1p") and not st.get("r2p"):
            rp_status.setText(T("Nothing to upload yet.", "Nada que subir aún.")); return
        rp_status.setText(T("☁ Uploading…", "☁ Subiendo…")); QtWidgets.QApplication.processEvents()
        try:
            from core.balance import cloud
            import socket
            s = st["setup_fn"]() if st.get("setup_fn") else {}
            name = (s.get("machine") or s.get("tag") or "Balance") + " · " + T("balancing", "balanceo")
            _acc = _host = ""
            try:
                from core.modal import licensing as _lic
                _acc = str((_lic.local_license_status() or {}).get("account") or "")
            except Exception:  # noqa: BLE001
                pass
            r = cloud.save_run(name, _payload(), account=_acc, client=s.get("client", ""),
                               tag=s.get("tag", ""), hostname=socket.gethostname())
        except Exception as e:  # noqa: BLE001
            r = {"ok": False, "reason": str(e)}
        rp_status.setText(T(f"☁ Uploaded (id {r.get('id','')}).", f"☁ Subido (id {r.get('id','')}).")
                          if r.get("ok") else
                          T(f"⚠ Upload failed ({r.get('reason','offline')}). Saved-local stays available.",
                            f"⚠ Falló la subida ({r.get('reason','offline')}). El guardado local queda disponible."))

    btn_pdf.clicked.connect(_gen_pdf); btn_savelocal.clicked.connect(_save_local); btn_upload.clicked.connect(_upload)
    tabs.addTab(_scroll(pg_rp), T("Report", "Reporte"))

    # =================================================================
    # TAB 5 — Updates
    # =================================================================
    pg_up = QtWidgets.QWidget(); ul = QtWidgets.QVBoxLayout(pg_up); ul.setContentsMargins(28, 24, 28, 24)
    _card = QtWidgets.QFrame()
    _card.setStyleSheet("QFrame{background:white;border:1px solid #e6ecf5;border-radius:16px;}")
    _card.setMinimumWidth(600); _card.setMaximumWidth(780)
    _cl2 = QtWidgets.QVBoxLayout(_card); _cl2.setContentsMargins(34, 30, 34, 34); _cl2.setSpacing(14)

    def _mkfont(pt, bold=False):
        fnt = QtGui.QFont(); fnt.setPointSize(pt); fnt.setBold(bold); return fnt

    _uh = QtWidgets.QLabel("🍉  Watermelon Balancing"); _uh.setFont(_mkfont(16, True)); _uh.setStyleSheet(f"color:{NAVY};border:none;")
    _cur = QtWidgets.QLabel(T(f"Installed version:  v{__version__}", f"Versión instalada:  v{__version__}"))
    _cur.setFont(_mkfont(11)); _cur.setStyleSheet("color:#475569;border:none;")
    _lic_lbl = QtWidgets.QLabel(""); _lic_lbl.setFont(_mkfont(10)); _lic_lbl.setWordWrap(True)
    _lic_lbl.setTextFormat(QtCore.Qt.RichText); _lic_lbl.setStyleSheet("color:#475569;border:none;")
    _lic_ok = False
    try:
        from core.modal import licensing as _lic
        _ls = _lic.local_license_status()
        import datetime as _dt2
        _expd = _dt2.date.fromtimestamp(float(_ls.get("exp"))).isoformat() if _ls.get("exp") else "—"
        _stt = (f"<span style='color:#10b981'>● {T('Licensed','Con licencia')}</span>" if _ls.get("valid")
                else f"<span style='color:#ef4444'>● {T('Not activated','Sin activar')}</span>")
        _lic_lbl.setText(f"<b>{T('License','Licencia')}:</b> {_stt}<br>"
                         f"{T('Account','Cuenta')}: {_ls.get('account') or '—'} · {T('Expires','Vence')}: {_expd}<br>"
                         f"<b>{T('This computer','Este equipo')}:</b> {_lic.machine_label()}<br>"
                         f"<span style='color:#94a3b8'>Machine ID: {_ls.get('fingerprint','')[:24]}…</span>")
        _lic_ok = bool(_ls.get("valid"))
    except Exception:  # noqa: BLE001
        _lic_lbl.setText("")
    _deact = QtWidgets.QPushButton(T("Deactivate this computer", "Desactivar este equipo")); _deact.setFont(_mkfont(9))
    _deact.setStyleSheet("QPushButton{background:transparent;color:#ef4444;border:1px solid #f2c4c4;border-radius:8px;padding:6px 14px;}"
                         "QPushButton:hover{background:#fef2f2;} QPushButton:disabled{color:#cbd5e1;border-color:#eef2f8;}")
    _deact.setEnabled(_lic_ok)
    _deact_row = QtWidgets.QHBoxLayout(); _deact_row.addWidget(_deact); _deact_row.addStretch(1)

    def _do_deact():
        m = QtWidgets.QMessageBox(_card); m.setIcon(QtWidgets.QMessageBox.Warning)
        m.setWindowTitle(T("Deactivate this computer", "Desactivar este equipo"))
        m.setText(T("Release this license from this computer?", "¿Liberar esta licencia de este equipo?"))
        m.setStandardButtons(QtWidgets.QMessageBox.Cancel | QtWidgets.QMessageBox.Yes)
        m.setDefaultButton(QtWidgets.QMessageBox.Cancel)
        if m.exec() != QtWidgets.QMessageBox.Yes:
            return
        try:
            from core.modal import licensing as _lic2
            r = _lic2.deactivate_machine()
        except Exception as e:  # noqa: BLE001
            r = {"ok": False, "reason": str(e)}
        if r.get("ok"):
            QtWidgets.QMessageBox.information(_card, "Watermelon Balancing",
                T("Deactivated. The app will close.", "Desactivado. La app se cerrará.")); QtWidgets.QApplication.quit()
        else:
            QtWidgets.QMessageBox.warning(_card, "Watermelon Balancing",
                T("Could not deactivate: ", "No se pudo desactivar: ") + str(r.get("reason", "")))
    _deact.clicked.connect(_do_deact)

    _ust = QtWidgets.QLabel(T("Press <b>Check for updates</b> to see if a newer version is available.",
                              "Pulsa <b>Buscar actualizaciones</b> para ver si hay una versión más nueva."))
    _ust.setFont(_mkfont(10)); _ust.setWordWrap(True); _ust.setTextFormat(QtCore.Qt.RichText); _ust.setStyleSheet("color:#64748b;border:none;")
    _unotes = QtWidgets.QTextBrowser(); _unotes.setFont(_mkfont(9)); _unotes.setMaximumHeight(200); _unotes.hide()
    _unotes.setStyleSheet("QTextBrowser{border:1px solid #eef2f8;border-radius:10px;background:#fbfcfe;color:#334155;padding:8px;}")
    _ub = QtWidgets.QPushButton(T("🔍  Check for updates", "🔍  Buscar actualizaciones")); _ub.setFont(_mkfont(11, True)); _ub.setMinimumHeight(42)
    _ub.setStyleSheet(f"QPushButton{{background:{NAVY};color:white;padding:10px 20px;border-radius:9px;}}QPushButton:hover{{background:#12325a;}}")
    _ubgo = QtWidgets.QPushButton(T("⬇  Update now", "⬇  Actualizar ahora")); _ubgo.setFont(_mkfont(11, True)); _ubgo.setMinimumHeight(42); _ubgo.hide()
    _ubgo.setStyleSheet(f"QPushButton{{background:{GREEN};color:white;padding:10px 20px;border-radius:9px;}}QPushButton:hover{{background:#12833a;}}")
    _urow = QtWidgets.QHBoxLayout(); _urow.setSpacing(12); _urow.addWidget(_ub); _urow.addWidget(_ubgo); _urow.addStretch(1)
    _foot = QtWidgets.QLabel(T("Updates download and install automatically over the network; the app restarts when done.",
                              "Las actualizaciones se descargan e instalan automáticamente por red; la app se reinicia al terminar."))
    _foot.setFont(_mkfont(9)); _foot.setWordWrap(True); _foot.setStyleSheet("color:#94a3b8;border:none;")
    _cl2.addWidget(_uh); _cl2.addWidget(_cur); _cl2.addWidget(_lic_lbl); _cl2.addLayout(_deact_row)
    _cl2.addSpacing(4); _cl2.addWidget(_ust); _cl2.addWidget(_unotes); _cl2.addSpacing(6); _cl2.addLayout(_urow)
    _cl2.addSpacing(6); _cl2.addWidget(_foot)
    ul.addWidget(_card, 0, QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop); ul.addStretch(1)
    st["_pending"] = None

    def _chk():
        _ub.setEnabled(False); _ub.setText(T("🔍  Checking…", "🔍  Buscando…")); QtWidgets.QApplication.processEvents()
        try:
            from core.balance.updater import diagnose
            info, msg = diagnose(__version__)
        except Exception as e:  # noqa: BLE001
            info, msg = None, f"Error: {e}"
        _ub.setEnabled(True); _ub.setText(T("🔍  Check for updates", "🔍  Buscar actualizaciones"))
        st["_pending"] = info
        if info:
            _ust.setText(T(f"✅ <b style='color:{GREEN}'>New version available: v{info['version']}</b>",
                           f"✅ <b style='color:{GREEN}'>Nueva versión disponible: v{info['version']}</b>"))
            _unotes.setPlainText((info.get("notes") or "").strip()); _unotes.show(); _ubgo.show()
        else:
            _ust.setText(str(msg).replace("\n", "<br>")); _unotes.hide(); _ubgo.hide()

    def _go():
        info = st.get("_pending")
        if not info:
            return
        from core.balance import updater
        url = info.get("setup_url") or info.get("zip_url")
        if not url:
            if info.get("html_url"):
                QtGui.QDesktopServices.openUrl(QtCore.QUrl(info["html_url"]))
            return
        dlg = QtWidgets.QProgressDialog(T("Downloading update…", "Descargando actualización…"), "Cancel", 0, 100, win)
        dlg.setWindowTitle("Updating"); dlg.setModal(True); dlg.setMinimumDuration(0); dlg.show()
        path = updater.download_file(url, on_progress=lambda fr: (dlg.setValue(int(fr * 100)), QtWidgets.QApplication.processEvents()))
        dlg.close()
        if path and path.lower().endswith("setup.exe"):
            updater.launch_installer(path); QtWidgets.QApplication.quit()
        elif path:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(path))
        else:
            QtWidgets.QMessageBox.warning(win, "Update", T("Could not download the update.", "No se pudo descargar la actualización."))
    _ub.clicked.connect(_chk); _ubgo.clicked.connect(_go)
    tabs.addTab(pg_up, T("Updates", "Actualizaciones"))

    # ---------- planos según configuración (#3) + peso sugerido (#4) ----------
    IDX_1P, IDX_2P = 3, 4       # índices de las pestañas 1 plano / 2 planos

    def _apply_planes(_=0):
        two = cb_planes.currentIndex() == 1
        tabs.setTabVisible(IDX_1P, not two)
        tabs.setTabVisible(IDX_2P, two)

    def _suggest_g():
        try:
            Wt, _u = recommend_trial_weight_g(tw_w.value(), sb_rpm.value(), tw_r.value(), tw_k.value())
            return float(Wt)
        except Exception:  # noqa: BLE001
            return 0.0

    def _refresh_suggest():
        g = _suggest_g()
        if g <= 0:
            return
        _txt = T(f"Suggested trial weight (API 684): <b>{g:,.2f} g</b> @ {tw_r.value():.0f} mm. "
                 "Edit it if you must — it is the field analyst's responsibility.",
                 f"Peso de prueba sugerido (API 684): <b>{g:,.2f} g</b> @ {tw_r.value():.0f} mm. "
                 "Cámbialo si lo requieres — es responsabilidad del analista de campo.")
        lbl_sugg1.setText(_txt); lbl_sugg2.setText(_txt)
        # prefill si el campo sigue en el default (10 g)
        for _m in (twm, wam, wbm):
            if abs(_m.value() - 10.0) < 1e-6 or _m.value() == 0.0:
                _m.setValue(g)

    def _warn_edit(field, base_label):
        def _f():
            g = _suggest_g()
            if g > 0 and abs(field.value() - g) / g > 0.05:
                base_label.setText(base_label.text() +
                    T("  ⚠ Modified from suggested — analyst's responsibility.",
                      "  ⚠ Modificado del sugerido — responsabilidad del analista."))
        return _f
    twm.editingFinished.connect(_warn_edit(twm, lbl_sugg1))
    wam.editingFinished.connect(_warn_edit(wam, lbl_sugg2))
    wbm.editingFinished.connect(_warn_edit(wbm, lbl_sugg2))
    for _w in (tw_w, tw_r, tw_k, sb_rpm):
        _w.valueChanged.connect(lambda _=0: _refresh_suggest())
    cb_planes.currentIndexChanged.connect(_apply_planes)
    tabs.currentChanged.connect(lambda i: (_refresh_suggest() if i in (IDX_1P, IDX_2P) else None))

    _on_sensor(); _load_setup(); _on_sensor()
    _load_config(); _on_mode(); _kph_sensor(); _apply_planes(); _refresh_suggest()
    return app, win


def main(argv=None):
    ap = argparse.ArgumentParser(description="Watermelon Balancing (native)")
    ap.add_argument("--sim", action="store_true", default=True)
    ap.parse_args(argv)
    try:
        import os as _os
        global _LANG
        _here = _os.path.dirname(_os.path.abspath(__file__))
        if _here not in sys.path:
            sys.path.insert(0, _here)
        _LANG = _load_lang()
        _app0 = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
        try:
            from license_gate import run_license_gate
            _brand = ("<span style='color:#fff;font-weight:800;letter-spacing:3px;font-size:22px;'>WATERMELON</span>"
                      "<span style='color:#1AAEE5;font-weight:800;letter-spacing:3px;font-size:22px;'>&nbsp;BALANCING</span>")
            _ok = run_license_gate(_app0, t=T, navy=NAVY, acc=ACC, brand_html=_brand, app_title="Watermelon Balancing")
        except SystemExit:
            raise
        except Exception:  # noqa: BLE001
            if _os.environ.get("WM_LICENSING", "1") == "0":
                _ok = True
            else:
                QtWidgets.QMessageBox.critical(None, "Watermelon Balancing",
                    "Licensing error — the app cannot start.\n\n" + traceback.format_exc()[-800:])
                sys.exit(1)
        if not _ok:
            sys.exit(0)
        app, win = build_app(simulated=True); win.showMaximized()
        sys.exit(app.exec())
    except Exception:  # noqa: BLE001
        err = traceback.format_exc()
        try:
            with open("watermelon_balancing_error.log", "w", encoding="utf-8") as fh:
                fh.write(err)
            QtWidgets.QMessageBox.critical(None, "Watermelon Balancing — startup error", err[-1500:])
        except Exception:  # noqa: BLE001
            print(err)
        sys.exit(1)


if __name__ == "__main__":
    main()
