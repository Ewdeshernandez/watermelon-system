from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List

import plotly.graph_objects as go
import streamlit as st

from core.auth import require_login, render_user_menu


# ============================================================
# Page config
# ============================================================

st.set_page_config(page_title="Watermelon System | Reports", layout="wide")
require_login()
render_user_menu()


# ============================================================
# Styles
# ============================================================

st.markdown(
    """
    <style>
        .wm-page-title {
            font-size: 2rem;
            font-weight: 700;
            color: #f5f7fb;
            margin-bottom: 0.20rem;
            letter-spacing: 0.2px;
        }
        .wm-page-subtitle {
            color: #9aa6b2;
            font-size: 0.98rem;
            margin-bottom: 1.10rem;
        }
        .wm-card {
            background: linear-gradient(180deg, rgba(18,24,34,0.96) 0%, rgba(12,17,25,0.96) 100%);
            border: 1px solid rgba(90,110,140,0.22);
            border-radius: 18px;
            padding: 1rem 1rem 0.95rem 1rem;
            box-shadow: 0 10px 28px rgba(0,0,0,0.18);
            margin-bottom: 1rem;
        }
        .wm-kpi {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 14px;
            padding: 0.8rem 0.9rem;
            min-height: 82px;
        }
        .wm-kpi-label {
            color: #8fa0b5;
            font-size: 0.83rem;
            margin-bottom: 0.2rem;
        }
        .wm-kpi-value {
            color: #ffffff;
            font-size: 1.15rem;
            font-weight: 700;
        }
        .wm-section-title {
            color: #ffffff;
            font-weight: 700;
            font-size: 1.08rem;
            margin-top: 0.15rem;
            margin-bottom: 0.75rem;
        }
        .wm-block-title {
            color: #f5f7fb;
            font-size: 1.05rem;
            font-weight: 700;
            margin-bottom: 0.15rem;
        }
        .wm-block-subtitle {
            color: #95a2b1;
            font-size: 0.90rem;
            margin-bottom: 0.80rem;
        }
        .wm-divider {
            height: 1px;
            background: linear-gradient(90deg, rgba(255,255,255,0.08), rgba(255,255,255,0.02));
            margin: 0.85rem 0 1rem 0;
        }
        .wm-badge {
            display: inline-block;
            padding: 0.22rem 0.55rem;
            border-radius: 999px;
            background: rgba(41, 182, 246, 0.12);
            color: #7fd7ff;
            border: 1px solid rgba(41, 182, 246, 0.24);
            font-size: 0.78rem;
            font-weight: 600;
            margin-right: 0.35rem;
        }
        .wm-muted {
            color: #93a1b3;
            font-size: 0.9rem;
        }
        .wm-note {
            color: #b8c3cf;
            font-size: 0.92rem;
            line-height: 1.55;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Session init
# ============================================================

if "report_items" not in st.session_state:
    st.session_state["report_items"] = []

if "report_meta" not in st.session_state:
    st.session_state["report_meta"] = {
        "report_title": "Technical Vibration Report",
        "client": "",
        "asset": "",
        "unit": "",
        "location": "",
        "prepared_by": "",
        "reviewed_by": "",
        "period": "",
        "report_date": "",
        "consecutive": "",
        "service_development": "",
        "recommendations": "",
    }


# ============================================================
# Helpers
# ============================================================

def _normalize_report_items(raw_items: Any) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []

    if not isinstance(raw_items, list):
        return items

    for idx, item in enumerate(raw_items):
        if not isinstance(item, dict):
            continue

        fig = item.get("figure")
        if fig is None:
            continue

        try:
            safe_fig = go.Figure(fig)
        except Exception:
            continue

        normalized = {
            "id": str(item.get("id") or f"report_item_{idx+1}"),
            "type": str(item.get("type") or "figure"),
            "title": str(item.get("title") or f"Figure {idx+1}"),
            "notes": str(item.get("notes") or ""),
            "signal_id": str(item.get("signal_id") or ""),
            "machine": str(item.get("machine") or ""),
            "point": str(item.get("point") or ""),
            "variable": str(item.get("variable") or ""),
            "timestamp": str(item.get("timestamp") or ""),
            "figure": safe_fig,
        }
        items.append(normalized)

    return items


def _persist_items(items: List[Dict[str, Any]]) -> None:
    st.session_state["report_items"] = items


def _get_items() -> List[Dict[str, Any]]:
    items = _normalize_report_items(st.session_state.get("report_items", []))
    _persist_items(items)
    return items


def _move_item(item_id: str, direction: int) -> None:
    items = _get_items()
    idx = next((i for i, item in enumerate(items) if item["id"] == item_id), None)
    if idx is None:
        return

    new_idx = idx + direction
    if new_idx < 0 or new_idx >= len(items):
        return

    items[idx], items[new_idx] = items[new_idx], items[idx]
    _persist_items(items)


def _remove_item(item_id: str) -> None:
    items = [item for item in _get_items() if item["id"] != item_id]
    _persist_items(items)


def _clear_all_items() -> None:
    st.session_state["report_items"] = []


def _type_badge(item_type: str) -> str:
    return item_type.replace("_", " ").title()


def _source_line(item: Dict[str, Any]) -> str:
    parts = [
        item.get("machine", "").strip(),
        item.get("point", "").strip(),
        item.get("variable", "").strip(),
        item.get("timestamp", "").strip(),
    ]
    parts = [p for p in parts if p]
    return " | ".join(parts) if parts else "Sin metadata asociada"


def _count_by_type(items: List[Dict[str, Any]], item_type: str) -> int:
    return sum(1 for item in items if item.get("type", "").lower() == item_type.lower())


items = _get_items()
meta = st.session_state["report_meta"]


# ============================================================
# Header
# ============================================================

st.markdown('<div class="wm-page-title">Reports</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="wm-page-subtitle">Editor de entregables técnicos premium. Este módulo consume directamente las figuras enviadas desde Spectrum y futuros módulos.</div>',
    unsafe_allow_html=True,
)


# ============================================================
# Top summary
# ============================================================

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(
        f"""
        <div class="wm-kpi">
            <div class="wm-kpi-label">Figures in Report</div>
            <div class="wm-kpi-value">{len(items):,}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        f"""
        <div class="wm-kpi">
            <div class="wm-kpi-label">Spectrum Blocks</div>
            <div class="wm-kpi-value">{_count_by_type(items, "spectrum"):,}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        f"""
        <div class="wm-kpi">
            <div class="wm-kpi-label">Prepared By</div>
            <div class="wm-kpi-value">{meta["prepared_by"] or "-"}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c4:
    st.markdown(
        f"""
        <div class="wm-kpi">
            <div class="wm-kpi-label">Consecutive</div>
            <div class="wm-kpi-value">{meta["consecutive"] or "-"}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown('<div class="wm-divider"></div>', unsafe_allow_html=True)


# ============================================================
# Global actions
# ============================================================

st.markdown('<div class="wm-section-title">Report Actions</div>', unsafe_allow_html=True)

ga1, ga2, ga3 = st.columns([1.2, 1.2, 4])

with ga1:
    if st.button("Refresh Items", use_container_width=True):
        _persist_items(_get_items())
        st.rerun()

with ga2:
    clear_disabled = len(items) == 0
    if st.button("Clear Report", use_container_width=True, disabled=clear_disabled):
        _clear_all_items()
        st.rerun()

st.markdown('<div class="wm-divider"></div>', unsafe_allow_html=True)


# ============================================================
# Metadata
# ============================================================

st.markdown('<div class="wm-section-title">Report Metadata</div>', unsafe_allow_html=True)

m1, m2, m3 = st.columns(3)
with m1:
    meta["report_title"] = st.text_input("Report Title", value=meta["report_title"])
with m2:
    meta["client"] = st.text_input("Client", value=meta["client"])
with m3:
    meta["asset"] = st.text_input("Asset / Machine", value=meta["asset"])

m4, m5, m6 = st.columns(3)
with m4:
    meta["unit"] = st.text_input("Unit", value=meta["unit"])
with m5:
    meta["location"] = st.text_input("Location", value=meta["location"])
with m6:
    meta["consecutive"] = st.text_input("Consecutive", value=meta["consecutive"])

m7, m8, m9 = st.columns(3)
with m7:
    meta["prepared_by"] = st.text_input("Prepared By", value=meta["prepared_by"])
with m8:
    meta["reviewed_by"] = st.text_input("Reviewed By", value=meta["reviewed_by"])
with m9:
    meta["report_date"] = st.text_input("Report Date", value=meta["report_date"])

m10, m11 = st.columns(2)
with m10:
    meta["period"] = st.text_input("Evaluation Period", value=meta["period"])
with m11:
    st.write("")

t1, t2 = st.columns(2)
with t1:
    meta["service_development"] = st.text_area(
        "Desarrollo del servicio",
        value=meta["service_development"],
        height=180,
        placeholder="Describe la intervención, alcance, metodología, comportamiento dinámico observado y hallazgos clave.",
    )
with t2:
    meta["recommendations"] = st.text_area(
        "Recomendaciones",
        value=meta["recommendations"],
        height=180,
        placeholder="Redacta recomendaciones técnicas, criticidad, acciones sugeridas y seguimiento.",
    )

st.session_state["report_meta"] = meta

st.markdown('<div class="wm-divider"></div>', unsafe_allow_html=True)


# ============================================================
# Report structure
# ============================================================

st.markdown('<div class="wm-section-title">Report Structure</div>', unsafe_allow_html=True)

if not items:
    st.info("Todavía no hay figuras en el reporte. Entra a Spectrum y usa el botón 'Enviar a Reporte'.")
else:
    for index, item in enumerate(items, start=1):
        st.markdown('<div class="wm-card">', unsafe_allow_html=True)
        st.markdown(
            f'<div class="wm-block-title">Figura {index}. {item["title"]}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="wm-block-subtitle"><span class="wm-badge">{_type_badge(item["type"])}</span>{_source_line(item)}</div>',
            unsafe_allow_html=True,
        )

        tcol1, tcol2, tcol3, tcol4 = st.columns([2.4, 0.8, 0.8, 0.8])

        with tcol1:
            new_title = st.text_input(
                "Figure Title",
                value=item["title"],
                key=f"report_title_{item['id']}",
            )
            item["title"] = new_title

        with tcol2:
            st.write("")
            st.write("")
            if st.button(
                "↑ Up",
                key=f"report_up_{item['id']}",
                use_container_width=True,
                disabled=index == 1,
            ):
                _move_item(item["id"], -1)
                st.rerun()

        with tcol3:
            st.write("")
            st.write("")
            if st.button(
                "↓ Down",
                key=f"report_down_{item['id']}",
                use_container_width=True,
                disabled=index == len(items),
            ):
                _move_item(item["id"], +1)
                st.rerun()

        with tcol4:
            st.write("")
            st.write("")
            if st.button(
                "Remove",
                key=f"report_remove_{item['id']}",
                use_container_width=True,
            ):
                _remove_item(item["id"])
                st.rerun()

        st.plotly_chart(
            item["figure"],
            use_container_width=True,
            config={"displaylogo": False},
            key=f"report_plot_{item['id']}",
        )

        new_notes = st.text_area(
            f"Technical interpretation for Figure {index}",
            value=item["notes"],
            key=f"report_notes_{item['id']}",
            height=140,
            placeholder="Escribe aquí el análisis técnico que irá debajo de esta figura en el reporte final.",
        )
        item["notes"] = new_notes

        st.markdown("</div>", unsafe_allow_html=True)

    _persist_items(items)


# ============================================================
# Report preview
# ============================================================

st.markdown('<div class="wm-divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="wm-section-title">Report Preview</div>', unsafe_allow_html=True)

p1, p2 = st.columns([1.15, 1.85])

with p1:
    st.markdown('<div class="wm-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="wm-block-title">{meta["report_title"] or "Technical Vibration Report"}</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="wm-note">
            <strong>Client:</strong> {meta["client"] or "-"}<br>
            <strong>Asset:</strong> {meta["asset"] or "-"}<br>
            <strong>Unit:</strong> {meta["unit"] or "-"}<br>
            <strong>Location:</strong> {meta["location"] or "-"}<br>
            <strong>Prepared By:</strong> {meta["prepared_by"] or "-"}<br>
            <strong>Reviewed By:</strong> {meta["reviewed_by"] or "-"}<br>
            <strong>Period:</strong> {meta["period"] or "-"}<br>
            <strong>Report Date:</strong> {meta["report_date"] or "-"}<br>
            <strong>Consecutive:</strong> {meta["consecutive"] or "-"}<br>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="wm-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="wm-block-subtitle">Recomendaciones</div>', unsafe_allow_html=True)
    st.write(meta["recommendations"] or "—")
    st.markdown('<div class="wm-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="wm-block-subtitle">Desarrollo del servicio</div>', unsafe_allow_html=True)
    st.write(meta["service_development"] or "—")
    st.markdown("</div>", unsafe_allow_html=True)

with p2:
    st.markdown('<div class="wm-card">', unsafe_allow_html=True)
    st.markdown('<div class="wm-block-title">Ordered Figure Summary</div>', unsafe_allow_html=True)

    if not items:
        st.markdown('<div class="wm-note">No hay figuras agregadas todavía.</div>', unsafe_allow_html=True)
    else:
        for index, item in enumerate(items, start=1):
            st.markdown(
                f"""
                <div class="wm-note">
                    <span class="wm-badge">Figura {index}</span>
                    <strong>{item["title"]}</strong><br>
                    {_source_line(item)}<br>
                    {item["notes"][:220] + ("..." if len(item["notes"]) > 220 else "") if item["notes"] else "Sin interpretación técnica todavía."}
                </div>
                <div class="wm-divider"></div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# Footer
# ============================================================

st.caption(
    "Flujo actual: Spectrum empuja figuras reales al reporte mediante st.session_state['report_items']. "
    "Reports actúa como editor de entregable técnico, sin reconstruir motores visuales."
)
