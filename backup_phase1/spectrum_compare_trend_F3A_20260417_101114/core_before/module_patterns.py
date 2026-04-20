from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple

import plotly.graph_objects as go
import streamlit as st

from core.ui_theme import card, kpi_card


def panel_card(
    title: str,
    subtitle: str,
    meta_html: str,
    chips: Sequence[str],
) -> None:
    card(title=title, subtitle=subtitle, meta_html=meta_html, chips=chips)


def helper_card(
    title: str,
    subtitle: str,
    chips: Sequence[Tuple[str, Optional[str]]],
) -> None:
    kpi_card(title=title, subtitle=subtitle, chips=chips)


def export_report_row(
    export_key: str,
    fig: go.Figure,
    export_builder: Callable[[go.Figure], tuple[bytes | None, str | None]],
    report_callback: Callable[[], None],
    file_name: str,
) -> None:
    if "wm_export_store" not in st.session_state:
        st.session_state["wm_export_store"] = {}

    if export_key not in st.session_state["wm_export_store"]:
        st.session_state["wm_export_store"][export_key] = {"png_bytes": None, "error": None}

    left_pad, col_export1, col_export2, col_report, right_pad = st.columns([2.0, 1.2, 1.2, 1.2, 2.0])

    with col_export1:
        if st.button("Prepare PNG HD", key=f"prepare_png_{export_key}", width="stretch"):
            with st.spinner("Generating HD export..."):
                png_bytes, export_error = export_builder(fig)
                st.session_state["wm_export_store"][export_key]["png_bytes"] = png_bytes
                st.session_state["wm_export_store"][export_key]["error"] = export_error

    with col_export2:
        png_bytes = st.session_state["wm_export_store"][export_key]["png_bytes"]
        if png_bytes is not None:
            st.download_button(
                "Download PNG HD",
                data=png_bytes,
                file_name=file_name,
                mime="image/png",
                key=f"download_png_{export_key}",
                width="stretch",
            )
        else:
            st.button(
                "Download PNG HD",
                disabled=True,
                key=f"download_disabled_{export_key}",
                width="stretch",
            )

    with col_report:
        if st.button("Enviar a Reporte", key=f"report_{export_key}", width="stretch"):
            report_callback()
            st.success("Elemento enviado al reporte.")
