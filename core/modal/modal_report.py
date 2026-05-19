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
