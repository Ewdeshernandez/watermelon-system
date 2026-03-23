
from pathlib import Path
import textwrap

path = Path("pages/03_Spectrum.py")
src = path.read_text(encoding="utf-8")

start_marker = """# ------------------------------------------------------------
# Session defaults
# ------------------------------------------------------------"""
end_marker = "if st.session_state.wm_sp_export_error:"

if start_marker not in src:
    raise SystemExit("No encontré el bloque 'Session defaults' en pages/03_Spectrum.py")

head = src.split(start_marker)[0]

new_tail = textwrap.dedent(r'''
# ------------------------------------------------------------
# Session defaults
# ------------------------------------------------------------
if "wm_sp_selected_signal_ids" not in st.session_state:
    st.session_state.wm_sp_selected_signal_ids = []
if "wm_sp_export_store" not in st.session_state:
    st.session_state.wm_sp_export_store = {}
if "report_items" not in st.session_state:
    st.session_state.report_items = []


# ------------------------------------------------------------
# Panel helpers
# ------------------------------------------------------------
def _signal_option_label(record: SignalRecord) -> str:
    parts = [
        str(record.machine or "").strip(),
        str(record.point or "").strip(),
        str(record.variable or "").strip(),
        str(record.name or "").strip(),
    ]
    parts = [p for p in parts if p]
    return " | ".join(parts) if parts else record.signal_id


def _safe_slug(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "spectrum"


def _queue_spectrum_to_report(
    record: SignalRecord,
    fig: go.Figure,
    panel_title: str,
) -> None:
    report_item_id = make_export_state_key(
        [
            "spectrum-report-item",
            record.signal_id,
            record.machine,
            record.point,
            record.variable,
            record.timestamp,
            panel_title,
            len(st.session_state.report_items),
        ]
    )

    st.session_state.report_items.append(
        {
            "id": report_item_id,
            "type": "spectrum",
            "title": panel_title,
            "notes": "",
            "signal_id": record.signal_id,
            "machine": record.machine,
            "point": record.point,
            "variable": record.variable,
            "timestamp": record.timestamp,
            "figure": go.Figure(fig),
        }
    )


def render_spectrum_panel(
    record: SignalRecord,
    panel_idx: int,
    *,
    window_name: str,
    amplitude_mode: str,
    remove_dc: bool,
    detrend: bool,
    zero_padding: bool,
    high_res_display: bool,
    high_res_factor: int,
    max_cpm: float,
    y_axis_mode: str,
    y_axis_manual_max: Optional[float],
    fill_area: bool,
    annotate_peak: bool,
    show_harmonics: bool,
    harmonic_count: int,
    harmonic_band_fraction: float,
    show_harmonic_amplitudes: bool,
    harmonic_label_mode: str,
    show_right_info_box: bool,
    logo_uri: Optional[str],
) -> None:
    spectrum = compute_spectrum_peak(
        time_s=record.time_s,
        y=record.amplitude,
        window_name=window_name,
        remove_dc=remove_dc,
        detrend=detrend,
        zero_padding=zero_padding,
        high_res_factor=high_res_factor,
        min_peak_cpm=1.0,
    )

    if spectrum.freq_cpm.size == 0 or spectrum.amp_peak.size == 0:
        st.warning(f"No fue posible calcular el espectro para: {_signal_option_label(record)}")
        return

    freq_cpm = spectrum.freq_cpm
    amp_peak = spectrum.amp_peak
    resolution_cpm = spectrum.resolution_cpm
    real_resolution_cpm = spectrum.real_resolution_cpm

    amp_display = convert_peak_to_mode(amp_peak, amplitude_mode)
    interpolated_peak_amp_display = convert_scalar_peak_to_mode(spectrum.peak_amp_peak, amplitude_mode)

    one_x_display_amp: Optional[float] = None
    one_x_display_freq_cpm: Optional[float] = None

    if record.rpm is not None and record.rpm > 0:
        one_x_freq_cpm = float(record.rpm)
        one_x_freq_hz = one_x_freq_cpm / 60.0

        one_x_peak_amp_from_waveform = estimate_harmonic_from_waveform_peak(
            time_s=record.time_s,
            y=record.amplitude,
            freq_hz=one_x_freq_hz,
            remove_mean=True,
        )

        one_x_local_freq_cpm, one_x_peak_amp_from_spectrum = find_local_peak_near_1x(
            freq_cpm=freq_cpm,
            amp_peak=amp_peak,
            one_x_cpm=one_x_freq_cpm,
            band_fraction=harmonic_band_fraction,
        )

        if one_x_peak_amp_from_waveform is not None:
            one_x_display_amp = convert_scalar_peak_to_mode(one_x_peak_amp_from_waveform, amplitude_mode)
            one_x_display_freq_cpm = one_x_freq_cpm
        elif one_x_peak_amp_from_spectrum is not None:
            one_x_display_amp = convert_scalar_peak_to_mode(one_x_peak_amp_from_spectrum, amplitude_mode)
            one_x_display_freq_cpm = one_x_local_freq_cpm

    all_harmonic_points = collect_harmonic_points(
        freq_cpm=freq_cpm,
        amp_peak=amp_peak,
        base_rpm=record.rpm,
        harmonic_count=harmonic_count,
        band_fraction=harmonic_band_fraction,
        max_cpm=max_cpm,
    )

    harmonic_points_for_labels = choose_harmonics_to_annotate(
        harmonic_points=all_harmonic_points,
        label_mode=harmonic_label_mode,
    )

    overall_spec_rms = compute_spectrum_overall_rms_parseval(
        time_s=record.time_s,
        y=record.amplitude,
        remove_dc=remove_dc,
        detrend=detrend,
        max_cpm=max_cpm,
    )

    fig = build_spectrum_figure(
        record=record,
        freq_cpm=freq_cpm,
        amp_display=amp_display,
        amp_peak=amp_peak,
        amplitude_mode=amplitude_mode,
        max_cpm=max_cpm,
        y_axis_mode=y_axis_mode,
        y_axis_manual_max=y_axis_manual_max,
        show_harmonics=show_harmonics,
        show_harmonic_amplitudes=show_harmonic_amplitudes,
        harmonic_points_for_labels=harmonic_points_for_labels if show_harmonics else [],
        show_right_info_box=show_right_info_box,
        fill_area=fill_area,
        annotate_peak=annotate_peak,
        logo_uri=logo_uri,
        spectrum_mode_label=window_name,
        one_x_display_amp=one_x_display_amp,
        one_x_display_freq_cpm=one_x_display_freq_cpm,
        overall_spec_rms=overall_spec_rms,
        resolution_cpm=resolution_cpm,
        real_resolution_cpm=real_resolution_cpm,
        interpolated_peak_freq_cpm=spectrum.peak_freq_cpm,
        interpolated_peak_amp_display=interpolated_peak_amp_display,
    )

    panel_key = make_export_state_key(
        [
            "wm-spectrum-panel",
            panel_idx,
            record.signal_id,
            record.machine,
            record.point,
            record.variable,
            record.timestamp,
            window_name,
            amplitude_mode,
            remove_dc,
            detrend,
            zero_padding,
            high_res_display,
            high_res_factor,
            max_cpm,
            y_axis_mode,
            y_axis_manual_max,
            fill_area,
            annotate_peak,
            show_harmonics,
            harmonic_count,
            harmonic_band_fraction,
            show_harmonic_amplitudes,
            harmonic_label_mode,
            show_right_info_box,
            record.rpm,
            float(np.nanmax(amp_display)) if amp_display.size else 0.0,
            float(np.nanmin(amp_display)) if amp_display.size else 0.0,
            amp_display.size,
            overall_spec_rms,
            resolution_cpm,
            real_resolution_cpm,
            spectrum.peak_freq_cpm,
            spectrum.peak_amp_peak,
            len(all_harmonic_points),
        ]
    )

    if panel_key not in st.session_state.wm_sp_export_store:
        st.session_state.wm_sp_export_store[panel_key] = {
            "cache_key": None,
            "png_bytes": None,
            "error": None,
        }

    if st.session_state.wm_sp_export_store[panel_key]["cache_key"] != panel_key:
        st.session_state.wm_sp_export_store[panel_key] = {
            "cache_key": panel_key,
            "png_bytes": None,
            "error": None,
        }

    display_title = f"Spectrum {panel_idx + 1} — {_signal_option_label(record)}"
    st.markdown(f"### {display_title}")

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displaylogo": False},
        key=f"wm_spectrum_plot_{panel_key}",
    )

    st.markdown('<div class="wm-export-actions"></div>', unsafe_allow_html=True)

    left_pad, col_export1, col_export2, col_report, right_pad = st.columns([1.5, 1.2, 1.2, 1.2, 1.5])

    with col_export1:
        if st.button("Prepare PNG HD", key=f"prepare_png_{panel_key}", use_container_width=True):
            with st.spinner("Generating HD export..."):
                png_bytes, export_error = build_export_png_bytes(fig=fig)
                st.session_state.wm_sp_export_store[panel_key]["png_bytes"] = png_bytes
                st.session_state.wm_sp_export_store[panel_key]["error"] = export_error

    with col_export2:
        panel_png = st.session_state.wm_sp_export_store[panel_key]["png_bytes"]
        if panel_png is not None:
            st.download_button(
                "Download PNG HD",
                data=panel_png,
                file_name=f"{_safe_slug(record.machine)}_{_safe_slug(record.point)}_{_safe_slug(record.variable)}_spectrum_hd.png",
                mime="image/png",
                key=f"download_png_{panel_key}",
                use_container_width=True,
            )
        else:
            st.button("Download PNG HD", key=f"download_disabled_{panel_key}", disabled=True, use_container_width=True)

    with col_report:
        if st.button("Enviar a Reporte", key=f"send_report_{panel_key}", use_container_width=True):
            _queue_spectrum_to_report(
                record=record,
                fig=fig,
                panel_title=display_title,
            )
            st.success("Spectrum enviado al reporte")

    panel_error = st.session_state.wm_sp_export_store[panel_key]["error"]
    if panel_error:
        st.warning(f"PNG export error: {panel_error}")

    if panel_idx < 999999:
        st.markdown("---")


# ------------------------------------------------------------
# Load signals
# ------------------------------------------------------------
records_all = load_signals_from_session()

if not records_all:
    st.warning("No se pudieron cargar señales válidas desde `st.session_state['signals']`.")
    st.stop()


# ------------------------------------------------------------
# Sidebar controls
# ------------------------------------------------------------
with st.sidebar:
    st.markdown("### Signal Selection")

    label_to_id = {_signal_option_label(r): r.signal_id for r in records_all}
    signal_labels = list(label_to_id.keys())
    valid_ids = {r.signal_id for r in records_all}

    current_selected_ids = [
        sig_id for sig_id in st.session_state.wm_sp_selected_signal_ids
        if sig_id in valid_ids
    ]

    if not current_selected_ids:
        current_selected_ids = [records_all[0].signal_id]

    current_selected_labels = [
        label for label, sig_id in label_to_id.items()
        if sig_id in current_selected_ids
    ]

    selected_labels = st.multiselect(
        "Spectra to display",
        options=signal_labels,
        default=current_selected_labels,
    )

    selected_signal_ids = [
        label_to_id[label]
        for label in selected_labels
        if label in label_to_id
    ]

    st.session_state.wm_sp_selected_signal_ids = selected_signal_ids

    st.markdown("### Spectrum Processing")

    window_name = st.selectbox(
        "Window",
        ["Hanning", "Hamming", "Blackman", "Rectangular"],
        index=0,
    )

    amplitude_mode = st.selectbox(
        "Amplitude mode",
        ["Peak", "Peak-to-Peak", "RMS"],
        index=1,
    )

    remove_dc = st.checkbox("Remove DC", value=True)
    detrend = st.checkbox("Detrend", value=True)
    zero_padding = st.checkbox("Zero padding", value=True)

    high_res_display = st.checkbox("High resolution display", value=True)
    high_res_factor = int(
        st.selectbox(
            "Display interpolation factor",
            options=[1, 2, 4, 8, 16],
            index=3,
            disabled=not high_res_display,
        )
    )
    if not high_res_display:
        high_res_factor = 1

    st.markdown("### Display")

    default_source_id = selected_signal_ids[0] if selected_signal_ids else records_all[0].signal_id
    default_source_record = next(r for r in records_all if r.signal_id == default_source_id)
    default_max_cpm = float(default_source_record.rpm * 10) if default_source_record.rpm is not None else 60000.0

    max_cpm = st.number_input(
        "Max frequency (CPM)",
        min_value=100.0,
        value=float(max(1000.0, default_max_cpm)),
        step=100.0,
        format="%.0f",
    )

    y_axis_mode = st.selectbox(
        "Y-axis scale",
        ["Auto", "Manual"],
        index=0,
    )

    y_axis_manual_max: Optional[float] = None
    if y_axis_mode == "Manual":
        y_axis_manual_max = float(
            st.number_input(
                "Manual Y max",
                min_value=0.001,
                value=3.0,
                step=0.1,
                format="%.3f",
            )
        )

    fill_area = st.checkbox("Fill area", value=True)
    annotate_peak = st.checkbox("Annotate dominant peak", value=True)
    show_right_info_box = st.checkbox("Show info box", value=True)

    st.markdown("### Harmonics")

    show_harmonics = st.checkbox("Show 1X harmonics", value=True)
    harmonic_count = int(
        st.number_input(
            "Harmonic count",
            min_value=1,
            max_value=30,
            value=8,
            step=1,
            disabled=not show_harmonics,
        )
    )

    harmonic_band_fraction = st.slider(
        "Harmonic search band (% of target)",
        min_value=5,
        max_value=40,
        value=20,
        step=1,
        disabled=not show_harmonics,
    ) / 100.0

    show_harmonic_amplitudes = st.checkbox(
        "Show harmonic amplitudes",
        value=True,
        disabled=not show_harmonics,
    )

    harmonic_label_mode = st.selectbox(
        "Harmonic label density",
        options=["1X only", "1X + Top 3", "Top 3 amplitudes", "All visible"],
        index=1,
        disabled=not (show_harmonics and show_harmonic_amplitudes),
    )


# ------------------------------------------------------------
# Render panels
# ------------------------------------------------------------
selected_signal_ids = [
    sig_id for sig_id in st.session_state.wm_sp_selected_signal_ids
    if sig_id in {r.signal_id for r in records_all}
]

if not selected_signal_ids:
    st.info("Selecciona uno o más espectros en la barra lateral.")
    st.stop()

records_by_id = {r.signal_id: r for r in records_all}
selected_records = [records_by_id[sig_id] for sig_id in selected_signal_ids if sig_id in records_by_id]

logo_uri = get_logo_data_uri(LOGO_PATH)

for panel_idx, record in enumerate(selected_records):
    render_spectrum_panel(
        record=record,
        panel_idx=panel_idx,
        window_name=window_name,
        amplitude_mode=amplitude_mode,
        remove_dc=remove_dc,
        detrend=detrend,
        zero_padding=zero_padding,
        high_res_display=high_res_display,
        high_res_factor=high_res_factor,
        max_cpm=max_cpm,
        y_axis_mode=y_axis_mode,
        y_axis_manual_max=y_axis_manual_max,
        fill_area=fill_area,
        annotate_peak=annotate_peak,
        show_harmonics=show_harmonics,
        harmonic_count=harmonic_count,
        harmonic_band_fraction=harmonic_band_fraction,
        show_harmonic_amplitudes=show_harmonic_amplitudes,
        harmonic_label_mode=harmonic_label_mode,
        show_right_info_box=show_right_info_box,
        logo_uri=logo_uri,
    )
''').lstrip()

new_src = head + new_tail
path.write_text(new_src, encoding="utf-8")
print("OK: pages/03_Spectrum.py actualizado")
PY