"""
core/ui_industrial.py
=====================

Look INDUSTRIAL compartido (el mismo del Report Center) para cualquier página
Streamlit: banda navy con textura + acento ámbar, tipografía IBM Plex, pestañas
con puntos de color (sin emojis), tablas y formularios pulidos.

Uso:
    from core.ui_industrial import inject_industrial_css, industrial_band
    inject_industrial_css()
    industrial_band("SIGA Internal · Administration", "Administration Panel",
                    "Clientes · Licencias · Usuarios",
                    kpis=[("● 1", "Admins"), ("● 2", "Specialists"), ("● 4", "Clientes")])
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import streamlit as st


def inject_industrial_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600&display=swap');
        :root{
          --wi-ink:#0b1f3a; --wi-ink2:#3a4c66; --wi-mut:#8090a6; --wi-line:#e2e8f2;
          --wi-navy:#12305e; --wi-steel:#274b7d; --wi-amber:#e8890c;
          --wi-ok:#1f9d55; --wi-warn:#e8890c; --wi-dang:#dc3545;
        }
        /* Banda de encabezado */
        .wi-band{position:relative;border-radius:16px;padding:20px 24px;margin:2px 0 20px;
          background:linear-gradient(120deg,#0e2547 0%,#12305e 46%,#274b7d 100%);
          color:#eaf1fb;overflow:hidden;box-shadow:0 10px 30px rgba(12,32,78,.22);}
        .wi-band::after{content:"";position:absolute;inset:0;
          background:repeating-linear-gradient(135deg,rgba(255,255,255,.045) 0 2px,transparent 2px 9px);pointer-events:none;}
        .wi-band .strip{position:absolute;top:0;right:0;height:100%;width:6px;
          background:linear-gradient(180deg,var(--wi-amber),#c56f05);}
        .wi-band .kick{font:600 10.5px/1 'IBM Plex Mono',monospace;letter-spacing:.28em;
          text-transform:uppercase;color:#8fb4e6;}
        .wi-band h2{margin:6px 0 2px;font:700 23px/1.15 'IBM Plex Sans',sans-serif;letter-spacing:-.4px;}
        .wi-band p{margin:0;color:#b9cbe6;font:500 13px/1.4 'IBM Plex Sans',sans-serif;}
        .wi-kpis{display:flex;gap:10px;flex-wrap:wrap;margin-top:14px;}
        .wi-kpi{background:rgba(255,255,255,.09);border:1px solid rgba(255,255,255,.14);
          border-radius:11px;padding:8px 14px;min-width:92px;}
        .wi-kpi .n{font:700 20px/1 'IBM Plex Sans';display:flex;align-items:center;gap:8px;}
        .wi-kpi .l{font:500 10.5px/1.2 'IBM Plex Sans';color:#a9c1e2;margin-top:4px;
          text-transform:uppercase;letter-spacing:.05em;}
        .wi-label{font:700 10.5px/1 'IBM Plex Mono',monospace;letter-spacing:.16em;
          text-transform:uppercase;color:var(--wi-steel);margin:8px 0 4px;}

        /* Pestañas (st.tabs) estilo subrayado + punto de color, sin caja */
        div[data-testid="stTabs"] div[role="tablist"]{gap:26px;border-bottom:1px solid var(--wi-line);}
        div[data-testid="stTabs"] button[role="tab"]{
          background:transparent !important;border:none !important;padding:6px 1px 11px !important;
          font:600 14px/1 'IBM Plex Sans',sans-serif !important;color:var(--wi-mut) !important;}
        div[data-testid="stTabs"] button[role="tab"]:hover{color:var(--wi-navy) !important;}
        div[data-testid="stTabs"] button[role="tab"][aria-selected="true"]{
          color:var(--wi-navy) !important;box-shadow:inset 0 -3px 0 var(--wi-amber);}
        div[data-testid="stTabs"] button[role="tab"]::before{content:"●";margin-right:8px;font-size:12px;
          position:relative;top:-1px;}
        div[data-testid="stTabs"] button[role="tab"]:nth-of-type(1)::before{color:#2563eb;}
        div[data-testid="stTabs"] button[role="tab"]:nth-of-type(2)::before{color:var(--wi-amber);}
        div[data-testid="stTabs"] button[role="tab"]:nth-of-type(3)::before{color:var(--wi-ok);}
        div[data-testid="stTabs"] button[role="tab"]:nth-of-type(4)::before{color:#8090a6;}
        div[data-testid="stTabs"] div[data-baseweb="tab-highlight"]{display:none;}

        /* Tablas (dataframe) con marco navy + esquinas suaves */
        div[data-testid="stDataFrame"]{border:1px solid var(--wi-line);border-radius:12px;overflow:hidden;
          box-shadow:0 1px 2px rgba(11,31,58,.05),0 6px 20px rgba(11,31,58,.05);}
        /* Subheaders → tipografía industrial */
        .stMarkdown h2, .stMarkdown h3{font-family:'IBM Plex Sans',sans-serif !important;
          letter-spacing:-.3px;color:var(--wi-ink);}
        </style>
        """,
        unsafe_allow_html=True,
    )


def industrial_band(kick: str, title: str, sub: str,
                    kpis: Optional[List[Tuple[str, str]]] = None) -> None:
    _k = ""
    if kpis:
        cells = "".join(
            f'<div class="wi-kpi"><div class="n">{n}</div><div class="l">{l}</div></div>'
            for n, l in kpis)
        _k = f'<div class="wi-kpis">{cells}</div>'
    st.markdown(
        f'<div class="wi-band"><span class="strip"></span>'
        f'<div class="kick">{kick}</div><h2>{title}</h2><p>{sub}</p>{_k}</div>',
        unsafe_allow_html=True,
    )


def dot(sev: str = "ok") -> str:
    c = {"ok": "#1f9d55", "warn": "#e8890c", "dang": "#dc3545",
         "info": "#2563eb", "off": "#8090a6"}.get(sev, "#8090a6")
    return f'<span style="color:{c};font-size:18px;line-height:0;">●</span>'


def html_table(columns: List[str], rows: List[List[object]]) -> None:
    """Tabla estilizada (header navy mono + zebra + marco redondeado), idéntica
    a las del Report Center. `rows` = lista de filas; cada celda se escapa. Los
    valores vacíos se muestran como '—'."""
    import html as _html
    _css = """
    <style>
    .wi-tbl-wrap{overflow-x:auto;border:1px solid #e2e8f2;border-radius:12px;
      box-shadow:0 1px 2px rgba(11,31,58,.05),0 6px 20px rgba(11,31,58,.05);margin:2px 0 6px;}
    table.wi-tbl{width:100%;border-collapse:collapse;font-family:'IBM Plex Sans',sans-serif;font-size:13px;}
    table.wi-tbl th{background:#12305e;color:#fff;text-align:left;padding:11px 14px;
      font:600 11px/1.3 'IBM Plex Mono',monospace;letter-spacing:.04em;white-space:nowrap;}
    table.wi-tbl td{padding:10px 14px;border-bottom:1px solid #eef2f7;color:#0b1f3a;vertical-align:top;}
    table.wi-tbl tr:last-child td{border-bottom:0;}
    table.wi-tbl tbody tr:nth-child(even) td{background:#f7f9fc;}
    table.wi-tbl .mono{font-family:'IBM Plex Mono',monospace;font-size:12px;color:#274b7d;}
    </style>
    """
    head = "".join(f"<th>{_html.escape(str(c))}</th>" for c in columns)
    body = []
    for r in rows:
        cells = []
        for i, cell in enumerate(r):
            val = "" if cell is None else str(cell)
            val = _html.escape(val) if val.strip() else "—"
            cls = ' class="mono"' if i == 0 else ""
            cells.append(f"<td{cls}>{val}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    st.markdown(
        _css + f'<div class="wi-tbl-wrap"><table class="wi-tbl"><thead><tr>{head}</tr>'
        f'</thead><tbody>{"".join(body)}</tbody></table></div>',
        unsafe_allow_html=True,
    )
