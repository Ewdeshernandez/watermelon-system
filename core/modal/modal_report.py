"""
core/modal/modal_report.py — Inyectar resultados modales al sistema Reports
==========================================================================

Genera items en formato Reports Watermelon estandar (type/title/notes/
image_bytes) a partir de los resultados de un FDD/EMA + geometria. Cada
item es una "figura" que el sistema Reports renderiza al PDF como cualquier
plot regular — NO requiere handlers PDF nuevos.

Por default filtra solo modos con classification="natural" (excluye
harmonics y spurious). Toggle include_non_natural=True para incluir todos.

Bloques generados por modo natural:
  - 3D snapshot del mode shape (via build_geometry_with_mode_shape + kaleido)
  - Bar chart 2D magnitud + fase por canal
  - Complexity polar plot
  + Bloque global:
  - AutoMAC heatmap matriz
  - Tabla resumen de todos los modos naturales

Norma aplicable
---------------
ISO 7626-6 §8 — Documentacion modal: cada modo debe incluir fn, zeta,
mode shape, metodo aplicado, validacion (MPC + AutoMAC).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def _natural_modes_filter(modes: List[Any],
                            include_non_natural: bool = False) -> List[Any]:
    """Filtra modos por classification. Default solo naturales."""
    if include_non_natural:
        return list(modes)
    return [m for m in modes
            if getattr(m, "classification", "natural") == "natural"]


def _plotly_to_png(fig, width: int = 1280, height: int = 720,
                    scale: float = 1.5) -> bytes:
    """Renderiza Plotly figure a PNG bytes via kaleido."""
    import plotly.io as pio
    return pio.to_image(fig, format="png", width=width, height=height,
                        scale=scale)


def _make_item(item_type: str, title: str, png_bytes: bytes,
                notes: str = "", machine: str = "",
                point: str = "", variable: str = "") -> Dict[str, Any]:
    """Construye un item compatible con report_state."""
    import uuid
    return {
        "id": f"modal_{uuid.uuid4().hex[:12]}",
        "type": item_type,
        "title": title,
        "notes": notes,
        "signal_id": "",
        "machine": machine,
        "point": point,
        "variable": variable,
        "timestamp": "",
        "figure": None,           # usamos image_bytes directo
        "image_bytes": png_bytes,
    }


def build_modal_report_items(
    fdd_result: Any,
    geom: Any,
    *,
    include_non_natural: bool = False,
    asset_name: str = "Activo",
    method: str = "OMA",
    running_rpm: float = 3600.0,
    colormap: str = "RdBu_r",
    camera_eye: Optional[Dict[str, float]] = None,
    camera_up: Optional[Dict[str, float]] = None,
    progress_cb: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """
    Genera lista de items para inyectar al report_state del sistema Reports.

    Args:
        fdd_result: FDDResult con .modes y .channel_names
        geom: ModalGeometry para los snapshots 3D
        include_non_natural: si True incluye harmonics+spurious
        asset_name: para el title/machine de cada item
        method: "OMA" o "EMA"
        running_rpm: para CPM y orden en titles
        colormap: para los plots heatmap
        camera_eye/up: preset de vista del 3D (lateral default si None)
        progress_cb: callable(idx, total, stage_str) opcional para UI feedback

    Returns: List[item dict] listo para append a st.session_state["report_items"]
    """
    from core.modal.geometry_3d import build_geometry_with_mode_shape
    from core.modal.modal_animator import (
        build_bar_chart_mode_shape,
        build_complexity_polar_plot,
        build_mac_matrix_plot,
    )
    from core.modal.oma_engine import compute_mac_matrix

    if not fdd_result or not getattr(fdd_result, "modes", None):
        return []

    natural_modes = _natural_modes_filter(
        fdd_result.modes, include_non_natural
    )
    if not natural_modes:
        return []

    items: List[Dict[str, Any]] = []
    total_steps = len(natural_modes) * 3 + 2
    step = 0

    def _bump(stage: str):
        nonlocal step
        if progress_cb:
            try:
                progress_cb(step, total_steps, stage)
            except Exception:
                pass
        step += 1

    # ------- Por cada modo natural: 3 plots PNG -------
    for mode in natural_modes:
        fn_hz = float(mode.natural_frequency_hz)
        fn_cpm = fn_hz * 60.0
        order = fn_cpm / max(running_rpm, 1.0)
        zeta = float(mode.damping_ratio_pct)
        mpc = float(getattr(mode, "complexity_pct", 0.0))
        cls = getattr(mode, "classification", "natural")

        mode_label = (f"Modo {mode.mode_number} · {fn_hz:.2f} Hz · "
                       f"{fn_cpm:,.0f} CPM · {order:.3f}x run")
        notes_base = (f"Damping zeta = {zeta:.3f}% · Q = "
                       f"{1.0 / (2 * max(zeta / 100.0, 1e-6)):.1f} · "
                       f"MPC = {mpc:.1f}% · Clasificacion: {cls.upper()} · "
                       f"Metodo: {method} · Norma: ISO 7626-6 §7.2")

        # 1) Snapshot 3D mode shape
        _bump(f"Modo {mode.mode_number} 3D")
        try:
            fig_3d = build_geometry_with_mode_shape(
                geom=geom,
                mode_shape=mode.mode_shape,
                channel_names=fdd_result.channel_names,
                mode_label=mode_label,
                animate=False,
                show_arrows=False,
                show_ghost=True,
                colormap=colormap,
                camera_eye=camera_eye,
                camera_up=camera_up,
            )
            png_3d = _plotly_to_png(fig_3d, width=1280, height=720, scale=1.5)
            items.append(_make_item(
                item_type="modal_3d",
                title=f"{mode_label} — Mode shape 3D",
                png_bytes=png_3d,
                notes=notes_base + " · Mesh con heatmap RdBu_r de amplitud "
                                    "(rojo cofase, azul anti-fase).",
                machine=asset_name,
                point="Train 3D",
                variable="Mode shape",
            ))
        except Exception as exc:  # noqa: BLE001
            pass

        # 2) Bar chart 2D magnitud + fase
        _bump(f"Modo {mode.mode_number} bar 2D")
        try:
            fig_bar = build_bar_chart_mode_shape(
                mode_shape=mode.mode_shape,
                channel_names=fdd_result.channel_names,
                mode_label=mode_label,
            )
            png_bar = _plotly_to_png(fig_bar, width=1280, height=560,
                                       scale=1.5)
            items.append(_make_item(
                item_type="modal_bar",
                title=f"{mode_label} — Bar chart magnitud + fase",
                png_bytes=png_bar,
                notes=("Magnitud normalizada y fase por canal del mode "
                       "shape complejo. ISO 7626-6 §7.2."),
                machine=asset_name,
                point="Sensores",
                variable="|phi| + arg(phi)",
            ))
        except Exception:  # noqa: BLE001
            pass

        # 3) Complexity polar plot
        _bump(f"Modo {mode.mode_number} polar")
        try:
            fig_pol = build_complexity_polar_plot(
                mode_shape=mode.mode_shape,
                channel_names=fdd_result.channel_names,
                mode_label=f"{mode_label} · MPC = {mpc:.1f}%",
            )
            png_pol = _plotly_to_png(fig_pol, width=720, height=720,
                                       scale=1.5)
            items.append(_make_item(
                item_type="modal_polar",
                title=f"{mode_label} — Complexity polar (MPC = {mpc:.1f}%)",
                png_bytes=png_pol,
                notes=("Vectores complejos del mode shape en el plano. "
                       "Colineales = modo natural. Dispersos = complejo o "
                       "espurio. Pappa & Eishan 1995."),
                machine=asset_name,
                point="Sensores",
                variable="Modal complexity",
            ))
        except Exception:  # noqa: BLE001
            pass

    # ------- Bloque global: AutoMAC heatmap -------
    _bump("AutoMAC")
    try:
        mac = compute_mac_matrix(natural_modes)
        labels = [f"M{m.mode_number} ({m.natural_frequency_hz:.1f} Hz)"
                  for m in natural_modes]
        fig_mac = build_mac_matrix_plot(mac, labels, title="AutoMAC",
                                          use_3d=False)
        png_mac = _plotly_to_png(fig_mac, width=960, height=720, scale=1.5)
        n = mac.shape[0]
        redundant = []
        for i in range(n):
            for j in range(i + 1, n):
                if mac[i, j] > 0.7:
                    redundant.append(
                        f"M{natural_modes[i].mode_number}-"
                        f"M{natural_modes[j].mode_number} ({mac[i,j]:.2f})"
                    )
        redundancy_note = (
            f"Pares redundantes: {', '.join(redundant)}"
            if redundant else
            "Sin pares redundantes — todos los modos son linealmente "
            "independientes."
        )
        items.append(_make_item(
            item_type="modal_automac",
            title=f"AutoMAC matrix — {len(natural_modes)} modos naturales",
            png_bytes=png_mac,
            notes=("Modal Assurance Criterion entre cada par de modos. "
                   "Diagonal = 1 (siempre). Off-diagonal > 0.7 indica modos "
                   "redundantes. ISO 7626-6 §6.5 + API 684 §1.6. · "
                   + redundancy_note),
            machine=asset_name,
            point="Set modal completo",
            variable="MAC matrix",
        ))
    except Exception:  # noqa: BLE001
        pass

    # ------- Bloque global: Tabla resumen como PNG -------
    _bump("Tabla resumen")
    try:
        import plotly.graph_objects as go
        header = ["Modo", "Freq (Hz)", "CPM", "Orden", "zeta (%)",
                   "Q", "MPC (%)", "Clase"]
        rows = []
        for m in natural_modes:
            _fh = float(m.natural_frequency_hz)
            _z = float(m.damping_ratio_pct)
            _mpc = float(getattr(m, "complexity_pct", 0.0))
            _cls = getattr(m, "classification", "natural")
            rows.append([
                f"M{m.mode_number}",
                f"{_fh:.2f}",
                f"{_fh * 60:,.0f}",
                f"{_fh * 60 / max(running_rpm, 1):.3f}x",
                f"{_z:.3f}",
                f"{1.0 / (2 * max(_z / 100, 1e-6)):.1f}",
                f"{_mpc:.1f}",
                _cls.upper(),
            ])
        cells_values = [[row[i] for row in rows] for i in range(len(header))]
        fig_tbl = go.Figure(data=[go.Table(
            header=dict(values=header, fill_color="#0F1E3D",
                          font=dict(color="white", size=12),
                          align="center"),
            cells=dict(values=cells_values,
                        fill_color=[["#f8fafc", "#e2e8f0"] * 50],
                        font=dict(color="#0F1E3D", size=11),
                        align="center", height=28),
        )])
        _tbl_h = max(200, 80 + len(rows) * 32)
        fig_tbl.update_layout(
            title=dict(text=f"Resumen modal — {asset_name} · {method}",
                        font=dict(size=14)),
            margin=dict(l=20, r=20, t=50, b=20),
            height=_tbl_h,
        )
        png_tbl = _plotly_to_png(fig_tbl, width=1280, height=_tbl_h,
                                    scale=1.5)
        items.append(_make_item(
            item_type="modal_summary_table",
            title=f"Resumen modal — {len(natural_modes)} modos naturales",
            png_bytes=png_tbl,
            notes=(f"Identificacion modal por {method}. Total: "
                   f"{len(natural_modes)} modos naturales. "
                   "Compliance ISO 7626-6 §8 (documentacion)."),
            machine=asset_name,
            point="Set modal completo",
            variable="Tabla modal",
        ))
    except Exception:  # noqa: BLE001
        pass

    return items


# -------------------------------------------------------------------
# Helper para inyectar items al report_state via session_state
# -------------------------------------------------------------------

def append_modal_items_to_report(items: List[Dict[str, Any]]) -> int:
    """Append items al session_state['report_items'] de Streamlit.

    Returns numero de items agregados.
    """
    import streamlit as st
    if not items:
        return 0
    existing = list(st.session_state.get("report_items", []))
    existing.extend(items)
    st.session_state["report_items"] = existing
    return len(items)


# ====================================================================
# Helpers Fase 3: render por categoria (Setup / Acq / EMA / OMA / FEA)
# Cada helper devuelve una List[item dict] lista para append.
# ====================================================================

def build_setup_items(geom: Any, asset_name: str = "Activo") -> List[Dict[str, Any]]:
    """Renderiza la geometria 3D del activo (sin deformacion) + tablas."""
    from core.modal.geometry_3d import build_geometry_figure
    items: List[Dict[str, Any]] = []
    if geom is None:
        return items
    # 1) Snapshot 3D estatico de la geometria
    try:
        fig = build_geometry_figure(geom)
        png = _plotly_to_png(fig, width=1280, height=720, scale=1.5)
        items.append(_make_item(
            item_type="modal_setup_geometry",
            title=f"Geometria 3D del activo · {asset_name}",
            png_bytes=png,
            notes=(f"Vista del tren mecanico con {len(geom.blocks)} bloques "
                   f"y {len(geom.sensors)} sensores instrumentados. "
                   "ISO 7626-6 §6 — DOF y orientacion espacial documentados."),
            machine=asset_name,
            point="Setup",
            variable="Geometria 3D",
        ))
    except Exception:  # noqa: BLE001
        pass

    # 2) Tabla de bloques + sensores como PNG
    try:
        import plotly.graph_objects as go
        # Bloques table
        if geom.blocks:
            rows_b = [[
                b.name, b.shape, b.kind,
                f"{b.x_start:.0f}", f"{b.x_end:.0f}",
                f"{b.radius:.0f}" if b.shape == "cylinder"
                else f"{b.half_width:.0f}, {b.half_height:.0f}",
            ] for b in geom.blocks]
            header_b = ["Nombre", "Forma", "Capa",
                         "x_start (mm)", "x_end (mm)", "R o hw,hh (mm)"]
            cells_b = [[r[i] for r in rows_b] for i in range(len(header_b))]
            fig_b = go.Figure(data=[go.Table(
                header=dict(values=header_b, fill_color="#0F1E3D",
                              font=dict(color="white", size=11),
                              align="center"),
                cells=dict(values=cells_b,
                            fill_color=[["#f8fafc", "#e2e8f0"] * 50],
                            font=dict(color="#0F1E3D", size=10),
                            align="center", height=24),
            )])
            _h_b = max(180, 60 + len(rows_b) * 28)
            fig_b.update_layout(
                title=dict(text=f"Bloques mecanicos · {len(geom.blocks)}",
                            font=dict(size=13)),
                margin=dict(l=20, r=20, t=40, b=20), height=_h_b,
            )
            png_b = _plotly_to_png(fig_b, width=1280, height=_h_b, scale=1.5)
            items.append(_make_item(
                item_type="modal_setup_blocks",
                title=f"Bloques mecanicos · {asset_name}",
                png_bytes=png_b,
                notes=f"Tabla de {len(geom.blocks)} secciones del tren.",
                machine=asset_name,
                point="Setup",
                variable="Bloques",
            ))
        # Sensores table
        if geom.sensors:
            rows_s = [[
                s.name, s.sensor_type, s.effective_mounting(),
                f"{s.x:.0f}", f"{s.y:.0f}", f"{s.z:.0f}", s.dof,
            ] for s in geom.sensors]
            header_s = ["Nombre", "Tipo", "Mounting",
                         "x (mm)", "y (mm)", "z (mm)", "DOF"]
            cells_s = [[r[i] for r in rows_s] for i in range(len(header_s))]
            fig_s = go.Figure(data=[go.Table(
                header=dict(values=header_s, fill_color="#0F1E3D",
                              font=dict(color="white", size=11),
                              align="center"),
                cells=dict(values=cells_s,
                            fill_color=[["#f8fafc", "#e2e8f0"] * 50],
                            font=dict(color="#0F1E3D", size=10),
                            align="center", height=24),
            )])
            _h_s = max(180, 60 + len(rows_s) * 28)
            fig_s.update_layout(
                title=dict(text=f"Sensores instrumentados · {len(geom.sensors)}",
                            font=dict(size=13)),
                margin=dict(l=20, r=20, t=40, b=20), height=_h_s,
            )
            png_s = _plotly_to_png(fig_s, width=1280, height=_h_s, scale=1.5)
            items.append(_make_item(
                item_type="modal_setup_sensors",
                title=f"Sensores · {asset_name}",
                png_bytes=png_s,
                notes=(f"Tabla de {len(geom.sensors)} sensores con tipo, "
                       "mounting (casing/shaft_proximity), posicion 3D y DOF."),
                machine=asset_name,
                point="Setup",
                variable="Sensores",
            ))
    except Exception:  # noqa: BLE001
        pass

    return items


def build_acquisition_items(tdms: Any, asset_name: str = "Activo",
                              max_channels: int = 6) -> List[Dict[str, Any]]:
    """Renderiza los waveforms time-series del TDMS cargado (uno por canal)."""
    items: List[Dict[str, Any]] = []
    if tdms is None or not getattr(tdms, "channels", None):
        return items
    try:
        import plotly.graph_objects as go
        import numpy as np
        fs = float(getattr(tdms, "sample_rate_hz", 5120.0))
        for ch_idx, ch in enumerate(tdms.channels[:max_channels]):
            data = np.asarray(ch.data, dtype=float)
            # Downsample para PDF (max 5000 puntos)
            if len(data) > 5000:
                stride = len(data) // 5000
                data_plot = data[::stride]
                t_plot = np.arange(len(data_plot)) * stride / fs
            else:
                data_plot = data
                t_plot = np.arange(len(data_plot)) / fs
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=t_plot, y=data_plot, mode="lines",
                line=dict(color="#0F1E3D", width=1),
                name=ch.name,
            ))
            unit = getattr(ch, "units", "") or ""
            fig.update_layout(
                title=dict(text=f"Waveform · {ch.name}",
                            font=dict(size=14, color="#0F1E3D")),
                xaxis=dict(title="Tiempo (s)", showgrid=True),
                yaxis=dict(title=f"Amplitud ({unit})" if unit else "Amplitud",
                            showgrid=True),
                margin=dict(l=60, r=20, t=40, b=50),
                height=380, paper_bgcolor="white", plot_bgcolor="#f8fafc",
            )
            png = _plotly_to_png(fig, width=1280, height=380, scale=1.5)
            items.append(_make_item(
                item_type="modal_acq_waveform",
                title=f"Waveform · {ch.name} ({len(data):,} muestras @ {fs:.0f} Hz)",
                png_bytes=png,
                notes=(f"Senal cruda del canal {ch.name} (unidad {unit}). "
                       f"Captura {len(data)/fs:.1f} s @ {fs:.0f} Hz. "
                       "ISO 7626-5 §6.4 — muestreo simultaneo."),
                machine=asset_name,
                point=ch.name,
                variable="Waveform raw",
            ))
    except Exception:  # noqa: BLE001
        pass
    return items


def build_ema_items(frfs: List[Any], peaks: List[Any],
                      asset_name: str = "Activo") -> List[Dict[str, Any]]:
    """Renderiza FRFs Bode + plot de magnitud con peaks marcados."""
    items: List[Dict[str, Any]] = []
    if not frfs and not peaks:
        return items
    try:
        import plotly.graph_objects as go
        import numpy as np
        # FRF principal (primera compleja, sino primera disponible)
        primary = None
        for f in frfs:
            if getattr(f, "is_complex_frf", False):
                primary = f; break
        if primary is None and frfs:
            primary = frfs[0]
        if primary is None:
            return items
        mag = primary.magnitude_linear() if hasattr(primary, "magnitude_linear") else None
        freqs = primary.frequencies_hz
        if mag is None or len(mag) == 0:
            return items
        mag_db = 20 * np.log10(np.maximum(mag, 1e-12))
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=freqs, y=mag_db, mode="lines",
            line=dict(color="#0F1E3D", width=1.5),
            name="FRF magnitud (dB)",
        ))
        # Marcar peaks
        if peaks:
            for p in peaks:
                fig.add_vline(x=p.frequency_hz, line=dict(color="#dc2626",
                                                            width=1, dash="dash"))
            peak_x = [p.frequency_hz for p in peaks]
            peak_y = [20 * np.log10(max(p.magnitude_peak, 1e-12))
                       if hasattr(p, "magnitude_peak") else 0 for p in peaks]
            fig.add_trace(go.Scatter(
                x=peak_x, y=peak_y, mode="markers",
                marker=dict(color="#dc2626", size=10, symbol="diamond"),
                name=f"Modos identificados ({len(peaks)})",
            ))
        fig.update_layout(
            title=dict(text=f"FRF EMA con {len(peaks)} modos identificados",
                        font=dict(size=14, color="#0F1E3D")),
            xaxis=dict(title="Frecuencia (Hz)", showgrid=True),
            yaxis=dict(title="Magnitud (dB)", showgrid=True),
            margin=dict(l=60, r=20, t=40, b=50),
            height=480, paper_bgcolor="white", plot_bgcolor="#f8fafc",
            legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.15),
        )
        png = _plotly_to_png(fig, width=1280, height=480, scale=1.5)
        items.append(_make_item(
            item_type="modal_ema_frf",
            title=f"FRF EMA · {len(peaks)} modos · Bode magnitud",
            png_bytes=png,
            notes=("FRF principal EMA en escala dB con peaks de modos "
                   "naturales marcados. ISO 7626-6 §6.3 — identificacion "
                   "por half-power band."),
            machine=asset_name,
            point="FRF",
            variable="EMA Bode",
        ))

        # Tabla peaks
        if peaks:
            header = ["Modo", "Freq (Hz)", "Damping (%)",
                       "Bandwidth (Hz)", "Q factor"]
            rows = [[
                f"P{i+1}",
                f"{p.frequency_hz:.2f}",
                f"{p.damping_ratio_pct:.3f}",
                f"{p.bandwidth_hz:.3f}",
                f"{p.quality_factor:.1f}",
            ] for i, p in enumerate(peaks)]
            cells = [[r[i] for r in rows] for i in range(len(header))]
            fig_t = go.Figure(data=[go.Table(
                header=dict(values=header, fill_color="#0F1E3D",
                              font=dict(color="white", size=11),
                              align="center"),
                cells=dict(values=cells,
                            fill_color=[["#f8fafc", "#e2e8f0"] * 50],
                            font=dict(color="#0F1E3D", size=10),
                            align="center", height=24),
            )])
            _h = max(180, 60 + len(rows) * 28)
            fig_t.update_layout(
                title=dict(text=f"Modos EMA · {len(peaks)} peaks",
                            font=dict(size=13)),
                margin=dict(l=20, r=20, t=40, b=20), height=_h,
            )
            png_t = _plotly_to_png(fig_t, width=1280, height=_h, scale=1.5)
            items.append(_make_item(
                item_type="modal_ema_peaks_table",
                title=f"Tabla peaks EMA · {len(peaks)} modos",
                png_bytes=png_t,
                notes="Identificacion por half-power method (ISO 7626-6 §6.3.2).",
                machine=asset_name,
                point="EMA",
                variable="Peaks table",
            ))
    except Exception:  # noqa: BLE001
        pass
    return items


def build_oma_items(fdd_result: Any, asset_name: str = "Activo",
                     ) -> List[Dict[str, Any]]:
    """Renderiza el SVD plot del FDD + tabla de modos OMA."""
    items: List[Dict[str, Any]] = []
    if not fdd_result or not getattr(fdd_result, "modes", None):
        return items
    try:
        import plotly.graph_objects as go
        import numpy as np
        # SVD plot
        if hasattr(fdd_result, "singular_values") and hasattr(fdd_result, "frequencies_hz"):
            svs = np.asarray(fdd_result.singular_values)
            freqs = np.asarray(fdd_result.frequencies_hz)
            sv_db = 20 * np.log10(np.maximum(svs[:, 0], 1e-12))
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=freqs, y=sv_db, mode="lines",
                line=dict(color="#0F1E3D", width=1.5),
                name="1st singular value (dB)",
            ))
            # Marcar modos identificados
            for m in fdd_result.modes:
                _color = {"natural": "#16a34a", "harmonic": "#D89B22",
                            "spurious": "#dc2626"}.get(
                    getattr(m, "classification", "natural"), "#64748b")
                fig.add_vline(x=m.natural_frequency_hz,
                                line=dict(color=_color, width=1, dash="dash"))
            fig.update_layout(
                title=dict(text=f"FDD First Singular Value · "
                                  f"{len(fdd_result.modes)} modos identificados",
                            font=dict(size=14, color="#0F1E3D")),
                xaxis=dict(title="Frecuencia (Hz)", showgrid=True),
                yaxis=dict(title="Magnitud SV (dB)", showgrid=True),
                margin=dict(l=60, r=20, t=40, b=50),
                height=480, paper_bgcolor="white", plot_bgcolor="#f8fafc",
            )
            png = _plotly_to_png(fig, width=1280, height=480, scale=1.5)
            items.append(_make_item(
                item_type="modal_oma_svd",
                title=(f"FDD · 1st Singular Value · "
                        f"{len(fdd_result.modes)} modos"),
                png_bytes=png,
                notes=("First singular value del cross-spectrum density "
                       "matrix decomposition. Peaks indican modos "
                       "estructurales. Brincker, Zhang, Andersen 2001."),
                machine=asset_name,
                point="FDD",
                variable="SVD",
            ))
    except Exception:  # noqa: BLE001
        pass
    return items


def build_fea_items(fea_result: Any, fdd_result: Any,
                     asset_name: str = "Activo") -> List[Dict[str, Any]]:
    """Renderiza Cross-MAC heatmap FEA <-> Experimental + tabla pareo."""
    items: List[Dict[str, Any]] = []
    if not fea_result or not getattr(fea_result, "modes", None):
        return items
    if not fdd_result or not getattr(fdd_result, "modes", None):
        return items
    try:
        from core.modal.fea_compare import (
            compute_fea_experimental_cross_mac,
            pair_modes,
            build_cross_mac_heatmap,
        )
        exp_shapes = [m.mode_shape for m in fdd_result.modes]
        exp_freqs = [m.natural_frequency_hz for m in fdd_result.modes]
        exp_channels = list(fdd_result.channel_names)
        exp_labels = [f"M{m.mode_number} ({m.natural_frequency_hz:.1f} Hz)"
                       for m in fdd_result.modes]
        fea_freqs = [m.freq_hz for m in fea_result.modes]
        fea_labels = [f"FEA M{m.mode_number} ({m.freq_hz:.1f} Hz)"
                       for m in fea_result.modes]
        mac = compute_fea_experimental_cross_mac(
            fea_modes=fea_result.modes,
            fea_dof_names=fea_result.dof_names,
            exp_mode_shapes=exp_shapes,
            exp_dof_names=exp_channels,
        )
        if mac is None:
            return items
        # 1) Heatmap PNG
        fig_mac = build_cross_mac_heatmap(
            mac=mac, fea_labels=fea_labels, exp_labels=exp_labels,
            title=f"Cross-MAC FEA <-> Experimental · {fea_result.model_name}",
        )
        png_mac = _plotly_to_png(fig_mac, width=1280, height=720, scale=1.5)
        items.append(_make_item(
            item_type="modal_fea_cross_mac",
            title=f"Cross-MAC FEA <-> Experimental · {fea_result.model_name}",
            png_bytes=png_mac,
            notes=(f"Modal Assurance Criterion entre {len(fea_result.modes)} "
                   f"modos FEA y {len(exp_shapes)} modos experimentales. "
                   "Diagonal alta indica buen match. API 684 §1.6."),
            machine=asset_name,
            point="FEA validation",
            variable="Cross-MAC",
        ))
        # 2) Tabla de pareo
        import plotly.graph_objects as go
        pairs = pair_modes(mac, fea_freqs, exp_freqs)
        status_map = {"valid": "OK", "shape_only": "Forma OK · freq fuera",
                       "freq_only": "Freq OK · forma debil",
                       "weak": "Debil", "no_match": "Sin match"}
        rows = [[
            f"M{p['fea_mode']} ({p['fea_freq']:.2f} Hz)",
            (f"M{p['exp_mode']} ({p['exp_freq']:.2f} Hz)"
              if p["exp_mode"] else "-"),
            f"{p['mac']:.3f}",
            (f"{p['delta_freq_pct']:.1f}"
              if p["delta_freq_pct"] is not None else "-"),
            status_map.get(p["status"], p["status"]),
        ] for p in pairs]
        header = ["FEA mode", "Experimental match", "MAC",
                   "Delta f (%)", "Estado"]
        cells = [[r[i] for r in rows] for i in range(len(header))]
        fig_p = go.Figure(data=[go.Table(
            header=dict(values=header, fill_color="#0F1E3D",
                          font=dict(color="white", size=11), align="center"),
            cells=dict(values=cells,
                        fill_color=[["#f8fafc", "#e2e8f0"] * 50],
                        font=dict(color="#0F1E3D", size=10),
                        align="center", height=26),
        )])
        _h = max(180, 60 + len(rows) * 30)
        fig_p.update_layout(
            title=dict(text=f"Pareo modos FEA <-> Experimental",
                        font=dict(size=13)),
            margin=dict(l=20, r=20, t=40, b=20), height=_h,
        )
        png_p = _plotly_to_png(fig_p, width=1280, height=_h, scale=1.5)
        items.append(_make_item(
            item_type="modal_fea_pairing",
            title=f"Pareo modos FEA <-> Experimental · {len(pairs)} pares",
            png_bytes=png_p,
            notes=("Pareo greedy con clasificacion: valid / shape_only / "
                   "freq_only / weak / no_match. API 684 §1.6 + Ewins 2000."),
            machine=asset_name,
            point="FEA validation",
            variable="Mode pairing",
        ))
    except Exception:  # noqa: BLE001
        pass
    return items


# ====================================================================
# Helper Fase 4: payload + item para análisis IA
# ====================================================================

def build_modal_ai_payload(
    fdd_result: Any,
    *,
    asset_name: str = "Activo",
    method: str = "OMA",
    running_rpm: float = 3600.0,
    operator_notes: str = "",
) -> Dict[str, Any]:
    """Construye payload para generate_ai_diagnostic con contexto modal."""
    if not fdd_result or not getattr(fdd_result, "modes", None):
        return {}

    all_modes = list(fdd_result.modes)
    natural_modes = [
        m for m in all_modes
        if getattr(m, "classification", "natural") == "natural"
    ]

    # Modos en formato estructurado para IA
    modes_data = []
    for m in all_modes:
        modes_data.append({
            "mode_number": int(m.mode_number),
            "freq_hz": round(float(m.natural_frequency_hz), 3),
            "freq_cpm": round(float(m.natural_frequency_hz) * 60.0, 1),
            "order_running": round(
                float(m.natural_frequency_hz) * 60.0 / max(running_rpm, 1.0),
                3,
            ),
            "damping_pct": round(float(m.damping_ratio_pct), 3),
            "Q_factor": round(
                1.0 / (2 * max(float(m.damping_ratio_pct) / 100.0, 1e-6)),
                1,
            ),
            "complexity_mpc_pct": round(
                float(getattr(m, "complexity_pct", 0.0)), 1,
            ),
            "classification": getattr(m, "classification", "natural"),
        })

    # MAC matrix para naturales
    mac_info = {}
    if len(natural_modes) >= 2:
        try:
            from core.modal.oma_engine import compute_mac_matrix
            mac = compute_mac_matrix(natural_modes)
            n = mac.shape[0]
            redundant_pairs = []
            for i in range(n):
                for j in range(i + 1, n):
                    if mac[i, j] > 0.7:
                        redundant_pairs.append({
                            "mode_i": natural_modes[i].mode_number,
                            "mode_j": natural_modes[j].mode_number,
                            "mac": round(float(mac[i, j]), 3),
                        })
            mac_info = {
                "n_modes_evaluated": int(n),
                "redundant_pairs_count": len(redundant_pairs),
                "redundant_pairs": redundant_pairs[:10],
            }
        except Exception:  # noqa: BLE001
            pass

    return {
        "machine": {
            "asset": asset_name,
            "running_speed_rpm": running_rpm,
            "method": method,
        },
        "norm": {
            "primary": "ISO 7626-1..6 (Modal analysis)",
            "operating_vibration": "ISO 20816",
            "rotor_dynamics": "API 684 §1.6",
            "compressor_separation": "API 618 §7.9.4.2.5.3.2",
        },
        "technical": {
            "method": method,
            "modes_total": len(all_modes),
            "modes_natural": len(natural_modes),
            "modes_harmonic": sum(
                1 for m in all_modes
                if getattr(m, "classification", "natural") == "harmonic"
            ),
            "modes_spurious": sum(
                1 for m in all_modes
                if getattr(m, "classification", "natural") == "spurious"
            ),
            "modes": modes_data,
            "automac_analysis": mac_info,
            "running_speed_rpm": running_rpm,
            "running_speed_hz": round(running_rpm / 60.0, 2),
            "frequency_band": (
                f"{fdd_result.frequencies_hz[0]:.1f} - "
                f"{fdd_result.frequencies_hz[-1]:.1f} Hz"
                if hasattr(fdd_result, "frequencies_hz") else "n/a"
            ),
        },
        "trend": {},
        "operator_notes": operator_notes,
    }


def build_ai_diagnostic_report_item(
    ai_result: Dict[str, Any],
    asset_name: str = "Activo",
    method: str = "OMA",
) -> Dict[str, Any]:
    """Convierte resultado de generate_ai_diagnostic a item del reporte.

    Render un PNG header navy con el título + la primera linea del markdown
    como teaser. El markdown completo va al campo 'notes' del item.
    """
    import io
    import uuid
    from PIL import Image, ImageDraw, ImageFont

    markdown = ai_result.get("markdown", "")
    model = ai_result.get("model", "unknown")
    cached = ai_result.get("cached", False)
    tokens_in = ai_result.get("input_tokens", 0)
    tokens_out = ai_result.get("output_tokens", 0)

    # Render header PNG nicely formatted
    width_px = 1280
    height_px = 280
    canvas = Image.new("RGB", (width_px, height_px), "white")
    d = ImageDraw.Draw(canvas)

    try:
        font_brand = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
        font_title = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
        font_body = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 13)
        font_meta = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 11)
    except (OSError, IOError):
        font_brand = ImageFont.load_default()
        font_title = font_brand; font_body = font_brand; font_meta = font_brand

    # Header navy
    d.rectangle([0, 0, width_px, 90], fill="#0F1E3D")
    d.text((20, 14), "Watermelon Modal · Análisis IA", font=font_brand,
           fill="#1AAEE5")
    d.text((20, 34), f"Diagnóstico interpretativo IA · {asset_name}",
           font=font_title, fill="white")
    d.text((20, 70), f"Método: {method} · modelo {model}"
                      + (" · CACHE HIT" if cached else ""),
           font=font_meta, fill="#94a3b8")
    d.rectangle([0, 86, width_px, 90], fill="#1AAEE5")

    # Primer párrafo del markdown como teaser (max 8 lineas)
    teaser_lines = []
    for line in markdown.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        teaser_lines.append(line)
        if len(teaser_lines) >= 8:
            break

    y = 110
    for line in teaser_lines:
        wrapped = []
        cur = ""
        for w in line.split():
            try:
                bbox = font_body.getbbox((cur + " " + w).strip())
                width_w = bbox[2] - bbox[0]
            except AttributeError:
                width_w = len((cur + " " + w).strip()) * 7
            if width_w <= width_px - 40 or not cur:
                cur = (cur + " " + w).strip()
            else:
                wrapped.append(cur); cur = w
        if cur: wrapped.append(cur)
        for wl in wrapped:
            d.text((20, y), wl, font=font_body, fill="#0F1E3D")
            y += 18
            if y > height_px - 30:
                break
        if y > height_px - 30:
            break

    d.text((20, height_px - 22),
           f"Texto completo en notes · tokens in={tokens_in} out={tokens_out}",
           font=font_meta, fill="#64748b")

    buf = io.BytesIO()
    canvas.save(buf, format="PNG", optimize=True)
    png_bytes = buf.getvalue()

    return {
        "id": f"modal_ai_{uuid.uuid4().hex[:12]}",
        "type": "modal_ai_diagnostic",
        "title": f"Análisis IA modal · {asset_name}",
        "notes": markdown,
        "signal_id": "",
        "machine": asset_name,
        "point": "Análisis modal",
        "variable": "IA interpretativa",
        "timestamp": "",
        "figure": None,
        "image_bytes": png_bytes,
    }
