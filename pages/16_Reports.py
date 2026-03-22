import streamlit as st
from pathlib import Path

from core.auth import require_login, render_user_menu

# IMPORTS DE TUS MÓDULOS (CLAVE)
from pages._02_Time_Waveforms import build_waveform_figure
from pages._03_Spectrum import (
    compute_spectrum_peak,
    build_spectrum_figure,
    convert_peak_to_mode,
)
from pages._05_Orbit_Analysis import build_orbit_figure
from pages._06_Trend import build_trend_figure

st.set_page_config(page_title="Watermelon System | Reports", layout="wide")

require_login()
render_user_menu()

# ============================================================
# LOAD DATA
# ============================================================

signals = st.session_state.get("signals", {})

if not signals:
    st.warning("No hay señales cargadas en el sistema.")
    st.stop()

# tomar la primera señal como demo (luego lo hacemos dinámico)
key = list(signals.keys())[0]
signal = signals[key]

logo_path = Path("assets/watermelon_logo.png")
logo_uri = None
if logo_path.exists():
    import base64
    logo_uri = f"data:image/png;base64,{base64.b64encode(logo_path.read_bytes()).decode()}"

st.title("📊 Industrial Report")

# ============================================================
# 1. WAVEFORM
# ============================================================

st.markdown("## Waveform Analysis")

fig_wave = build_waveform_figure(
    record=signal,
    cursor_a_s=0,
    cursor_b_s=0.01,
    x_axis_unit="ms",
    show_cursor_b=True,
    show_right_info_box=True,
    y_scale_mode="Auto",
    y_limit_abs=None,
    logo_uri=logo_uri,
    waveform_mode_label="Raw",
    show_cycle_start_markers=True,
)

st.plotly_chart(fig_wave, use_container_width=True)

# ============================================================
# 2. SPECTRUM
# ============================================================

st.markdown("## Spectrum Analysis")

spectrum = compute_spectrum_peak(
    time_s=signal.time,
    y=signal.x,
    window_name="Hanning",
    remove_dc=True,
    detrend=True,
    zero_padding=True,
    high_res_factor=8,
)

amp_display = convert_peak_to_mode(spectrum.amp_peak, "Peak-to-Peak")

fig_spec = build_spectrum_figure(
    record=signal,
    freq_cpm=spectrum.freq_cpm,
    amp_display=amp_display,
    amp_peak=spectrum.amp_peak,
    amplitude_mode="Peak-to-Peak",
    max_cpm=60000,
    y_axis_mode="Auto",
    y_axis_manual_max=None,
    show_harmonics=True,
    show_harmonic_amplitudes=True,
    harmonic_points_for_labels=[],
    show_right_info_box=True,
    fill_area=True,
    annotate_peak=True,
    logo_uri=logo_uri,
    spectrum_mode_label="Hanning",
    one_x_display_amp=None,
    one_x_display_freq_cpm=None,
    overall_spec_rms=None,
    resolution_cpm=spectrum.resolution_cpm,
    real_resolution_cpm=spectrum.real_resolution_cpm,
    interpolated_peak_freq_cpm=spectrum.peak_freq_cpm,
    interpolated_peak_amp_display=None,
)

st.plotly_chart(fig_spec, use_container_width=True)

# ============================================================
# 3. ORBIT
# ============================================================

st.markdown("## Orbit Analysis")

# aquí debes usar tu result real
# result = compute_orbit(...)
# fig_orbit = build_orbit_figure(...)

st.info("Orbit listo para integrar (usa tu compute actual)")

# ============================================================
# 4. TREND
# ============================================================

st.markdown("## Trend Analysis")

# records = [...]
# fig_trend = build_trend_figure(...)

st.info("Trend listo para integrar (usa tus CSV cargados)")