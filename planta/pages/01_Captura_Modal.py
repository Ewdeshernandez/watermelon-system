"""
planta/pages/01_Captura_Modal.py — UI captura standalone
==========================================================

Pantalla principal de configuración + captura para Watermelon Planta
Edition. Reusa los componentes de core/modal/ del repo principal pero
sin requerir auth ni Supabase.

Flujo:
1. Auto-discovery de la maleta Watermelon (qué módulos están instalados)
2. Selección modo EMA / OMA
3. Parámetros normativos (fs, duración, trigger para EMA)
4. Grid editable con N canales según hardware detectado
5. Validación pre-captura
6. Botón "Iniciar captura" con progress bar
7. Al terminar, link directo al TDMS generado + opción de copiar comando
   companion equivalente para reproducir desde CLI
"""
from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime
import streamlit as st

# Importar el core del repo
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from core.modal.acq_backend import (  # noqa: E402
    AcquisitionConfig,
    ChannelConfig,
    capture as ni_capture,
    discover_acq_modules,
)

st.set_page_config(
    page_title="Watermelon Planta · Captura",
    page_icon="🍉",
    layout="wide",
    initial_sidebar_state="collapsed",
)

_CAPTURES_DIR = _REPO_ROOT / "planta" / "data" / "captures"
_CAPTURES_DIR.mkdir(parents=True, exist_ok=True)

# Botón volver
if st.button("← Volver al inicio", key="back_home"):
    st.switch_page("app_planta.py")

st.markdown(
    "<h2 style='margin-top:0;'>📥 Configurar y disparar captura</h2>",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# 1. Auto-discovery del hardware
# ------------------------------------------------------------------
_chassis = "cDAQ1"
try:
    _modules = discover_acq_modules(_chassis)
except ImportError as exc:
    st.error(
        f"Drivers de adquisición Watermelon no disponibles. "
        f"Corre INSTALAR.bat y reinicia. (detalle técnico: {exc})"
    )
    st.stop()
except Exception as exc:
    st.error(f"Detección de la maleta falló: {exc}")
    _modules = []

if not _modules:
    st.warning(
        "⚠ No se detectó la maleta Watermelon conectada. "
        "Verifica que esté conectada por USB y que sus indicadores "
        "estén encendidos. Puedes seguir configurando pero NO podrás "
        "capturar."
    )
    _installed_slots = set()
    _max_bnc = 32
else:
    _installed_slots = {m["slot"] for m in _modules}
    _max_bnc = max(m["bnc_range"][1] for m in _modules)
    st.success(
        f"✓ **Maleta Watermelon conectada** "
        f"→ BNC 1..{_max_bnc} disponibles ({_max_bnc} canales)"
    )

# ------------------------------------------------------------------
# 2. Selector modo
# ------------------------------------------------------------------
default_mode = st.session_state.get("_planta_mode_preselect", "ema")
mode_idx = 1 if default_mode == "oma" else 0
modo = st.radio(
    "**Modo de ensayo**",
    ["EMA · Impact Hammer", "OMA · Continuous"],
    index=mode_idx,
    horizontal=True,
    key="planta_modo",
    help="EMA = ensayo con martillo modal en máquina parada. "
         "OMA = captura continua bajo condiciones operacionales.",
)
is_oma = modo.startswith("OMA")

# Limpiar pre-select para no quedar pegado
if "_planta_mode_preselect" in st.session_state:
    del st.session_state["_planta_mode_preselect"]

# ------------------------------------------------------------------
# 3. Parámetros generales
# ------------------------------------------------------------------
st.divider()
st.markdown("### ⚙ Parámetros de captura")

p_col1, p_col2, p_col3 = st.columns(3)
with p_col1:
    fs = st.number_input(
        "Sample rate (Hz)",
        value=int(st.session_state.get("planta_fs", 5120 if is_oma else 12800)),
        min_value=1024, max_value=51200, step=1024,
        key="planta_fs",
        help="Hasta 51.2 kHz soportado. Típico EMA 12800 Hz, OMA 5120 Hz.",
    )

if is_oma:
    with p_col2:
        fn_low = st.number_input(
            "Frecuencia natural mínima esperada (Hz)",
            value=float(st.session_state.get("planta_fn_low", 5.0)),
            min_value=0.5, max_value=200.0, step=0.5,
            key="planta_fn_low",
            help="Brincker & Ventura 2015: T_min ≥ 2000/fn_low.",
        )
    with p_col3:
        _t_default = max(120.0, 2000.0 / max(fn_low, 0.1))
        duration = st.number_input(
            "Duración (s)",
            value=float(st.session_state.get("planta_dur_oma", _t_default)),
            min_value=30.0, max_value=3600.0, step=30.0,
            key="planta_dur_oma",
            help=f"T_min recomendado para fn_low={fn_low} Hz: "
                 f"{2000.0/fn_low:.0f} s",
        )
    averages = 1
    trigger_bnc = None
    trigger_level = 0.5
else:  # EMA
    with p_col2:
        duration = st.number_input(
            "Duración por impacto (s)",
            value=float(st.session_state.get("planta_dur_ema", 2.0)),
            min_value=0.5, max_value=10.0, step=0.5,
            key="planta_dur_ema",
            help="Window para que respuesta decaiga. Típico 1-2 s.",
        )
    with p_col3:
        averages = st.number_input(
            "N° de impactos a promediar",
            value=int(st.session_state.get("planta_avg_ema", 5)),
            min_value=1, max_value=30, step=1,
            key="planta_avg_ema",
            help="ISO 7626-5 §6.3: mínimo 3, recomendado 5-10.",
        )
    fn_low = 0.0

# ------------------------------------------------------------------
# 4. Grid editable de canales
# ------------------------------------------------------------------
st.divider()
st.markdown(f"### 📡 Canales · maleta `{_chassis}` (hasta {_max_bnc} BNC)")

import pandas as pd

_grid_key = f"planta_grid_{'oma' if is_oma else 'ema'}"
if _grid_key not in st.session_state:
    _rows = []
    for bnc in range(1, _max_bnc + 1):
        slot = (bnc - 1) // 4 + 1
        if not is_oma and bnc == 1:
            _rows.append({
                "BNC": bnc, "Slot": slot, "Habilitado": True,
                "Nombre": "Hammer", "Coupling": "IEPE",
                "Sens (mV/EU)": 2.25, "Unidad": "N",
            })
        else:
            _rows.append({
                "BNC": bnc, "Slot": slot,
                "Habilitado": (bnc <= 4 if not is_oma else bnc <= min(8, _max_bnc)),
                "Nombre": f"Ch{bnc:02d}", "Coupling": "IEPE",
                "Sens (mV/EU)": 100.0, "Unidad": "g",
            })
    st.session_state[_grid_key] = _rows

_df = pd.DataFrame(st.session_state[_grid_key])
_edited = st.data_editor(
    _df,
    key=f"planta_grid_editor_{'oma' if is_oma else 'ema'}",
    hide_index=True,
    use_container_width=True,
    num_rows="fixed",
    column_config={
        "BNC": st.column_config.NumberColumn(
            "BNC", disabled=True, width="small",
        ),
        "Slot": st.column_config.NumberColumn(
            "Slot", disabled=True, width="small",
        ),
        "Habilitado": st.column_config.CheckboxColumn(
            "✓", width="small",
        ),
        "Nombre": st.column_config.TextColumn(
            "Sensor", max_chars=20, width="medium",
        ),
        "Coupling": st.column_config.SelectboxColumn(
            "Coupling", options=["IEPE", "AC", "DC"], width="small",
        ),
        "Sens (mV/EU)": st.column_config.NumberColumn(
            "Sens", min_value=0.1, max_value=10000.0,
            step=0.1, format="%.2f", width="small",
        ),
        "Unidad": st.column_config.SelectboxColumn(
            "EU", options=["g", "mil", "N", "ips", "mm/s"], width="small",
        ),
    },
)
st.session_state[_grid_key] = _edited.to_dict("records")

_enabled = [r for r in st.session_state[_grid_key] if r.get("Habilitado")]

# Detectar martillo (sens < 10 mV/EU → es martillo)
hammer_bnc = None
if not is_oma:
    for r in _enabled:
        if (r.get("Coupling", "").upper() == "IEPE"
                and float(r.get("Sens (mV/EU)", 100.0)) < 10):
            hammer_bnc = int(r["BNC"])
            break

# KPIs
kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric("Canales habilitados", len(_enabled))
kpi2.metric("Slots requeridos",
              len({r["Slot"] for r in _enabled}))
if not is_oma and hammer_bnc:
    kpi3.metric("Martillo detectado", f"BNC {hammer_bnc}")
elif is_oma:
    _ram = (len(_enabled) * duration * fs * 4) / (1024 * 1024)
    kpi3.metric("Tamaño TDMS estimado", f"{_ram:.0f} MB")
else:
    kpi3.metric("Martillo", "⚠ no detectado",
                  help="Configura un canal con sensibilidad < 10 mV/N (típico martillo modal)")

# Validaciones
errors = []
if len(_enabled) == 0:
    errors.append("Habilita al menos 1 canal en la tabla")
if not is_oma and hammer_bnc is None:
    errors.append(
        "Modo EMA requiere 1 canal de martillo (sens < 10 mV/N). "
        "Configura el martillo en BNC 1 con sens 2.25 mV/N."
    )
if _installed_slots:
    missing = {r["Slot"] for r in _enabled} - _installed_slots
    if missing:
        errors.append(
            f"Canales habilitados en slots vacíos: {sorted(missing)}. "
            f"Solo tienes módulos en slots {sorted(_installed_slots)}."
        )

# ------------------------------------------------------------------
# 5. Botón captura
# ------------------------------------------------------------------
st.divider()

if errors:
    for e in errors:
        st.error(f"✗ {e}")
    st.button("🎙 Iniciar captura", disabled=True, type="primary",
                use_container_width=True)
else:
    st.success("✓ Todo OK — listo para capturar")
    if st.button("🎙 **Iniciar captura ahora**", type="primary",
                  use_container_width=True, key="planta_capture_btn"):
        # Construir ChannelConfigs
        channels = []
        for r in _enabled:
            channels.append(ChannelConfig(
                bnc_port=int(r["BNC"]),
                name=str(r.get("Nombre", f"Ch{r['BNC']:02d}")).strip(),
                coupling=str(r.get("Coupling", "IEPE")).upper(),
                sensitivity_mv_per_eu=float(r.get("Sens (mV/EU)", 100.0)),
                units=str(r.get("Unidad", "g")),
            ))

        # Output path
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode_token = "oma" if is_oma else "ema"
        out_path = _CAPTURES_DIR / f"planta_{mode_token}_{ts}.tdms"

        config = AcquisitionConfig(
            mode="oma_continuous" if is_oma else "ema_triggered",
            sample_rate_hz=float(fs),
            duration_s=float(duration),
            channels=channels,
            chassis_name=_chassis,
            trigger_channel=hammer_bnc if not is_oma else None,
            trigger_level_V=0.1,
            n_averages=int(averages) if not is_oma else 1,
            output_tdms_path=out_path,
        )

        progress_bar = st.progress(0.0, text="Iniciando...")
        def _cb(p, s):
            try:
                progress_bar.progress(min(max(p, 0.0), 1.0), text=s)
            except Exception:
                pass

        try:
            with st.spinner(f"Capturando {len(channels)} canales..."):
                result = ni_capture(config, on_progress=_cb)
            progress_bar.empty()
            st.balloons()
            st.success(
                f"✓ **Captura completa**\n\n"
                f"Archivo: `{result.name}`\n\n"
                f"Carpeta: `{result.parent}`"
            )
            st.caption(
                "El archivo TDMS está listo. Cuando tengas internet:\n"
                "1. Abre Watermelon Cloud → Modal Analysis → Adquisición → Importar archivo\n"
                "2. Sube este TDMS para procesarlo y validarlo contra ISO 7626-5\n"
                "3. O usa el sync uploader cuando esté disponible (próximo release)"
            )
        except Exception as exc:
            progress_bar.empty()
            st.error(f"✗ **Error en captura**: {exc}")
            st.caption(
                "Verifica: maleta conectada por USB, sensores bien conectados "
                "a los BNC correctos, sensibilidades correctas. Si el problema "
                "persiste, contacta soporte (ehernandez@sigasas.com)."
            )

# ------------------------------------------------------------------
# 6. Comando companion equivalente (para reproducir desde CLI si querés)
# ------------------------------------------------------------------
if _enabled:
    with st.expander("▸ Comando CLI equivalente (para reproducir desde Terminal)"):
        cmd_parts = [
            f"python scripts\\capture_companion\\capture.py",
            f"--mode {'oma' if is_oma else 'ema'}",
            f"--chassis {_chassis}",
            f"--fs {int(fs)}",
            f"--duration {float(duration)}",
        ]
        if is_oma:
            cmd_parts.append(f"--fn-low {float(fn_low)}")
        else:
            cmd_parts.append(f"--averages {int(averages)}")
            if hammer_bnc:
                cmd_parts.append(f"--trigger-bnc {hammer_bnc}")
        for r in _enabled:
            _nombre = r.get("Nombre") or f"Ch{int(r['BNC']):02d}"
            _bnc = int(r["BNC"])
            _coup = str(r.get("Coupling", "IEPE")).upper()
            _sens = float(r.get("Sens (mV/EU)", 100.0))
            cmd_parts.append(f"--channels {_nombre}:{_bnc}:{_coup}:{_sens:g}")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        cmd_parts.append(
            f"--output planta\\data\\captures\\manual_{'oma' if is_oma else 'ema'}_{ts}.tdms"
        )
        st.code(" \\\n    ".join(cmd_parts), language="powershell")
        st.caption(
            "Pega en PowerShell (en UNA sola línea, sin los `\\` ni saltos)."
        )
