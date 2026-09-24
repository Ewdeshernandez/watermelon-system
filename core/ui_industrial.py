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
