"""Piezas visuales de Administración (estilo listas agrupadas, sobrio)."""
from __future__ import annotations

import html
from typing import Iterable, Tuple

import streamlit as st

CSS = """
<style>
.adm-h1{font:600 30px/1.15 'IBM Plex Sans',sans-serif;color:#0b1f3a;margin:2px 0 2px}
.adm-hi{font:400 14px/1.4 'IBM Plex Sans',sans-serif;color:#64748b;margin-bottom:18px}
.adm-tiles{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:0 0 14px}
@media (max-width:900px){.adm-tiles{grid-template-columns:repeat(2,minmax(0,1fr))}}
.adm-tile{background:#fff;border:1px solid #e6ebf2;border-radius:14px;padding:14px 16px}
.adm-tn{font:600 26px/1.1 'IBM Plex Sans',sans-serif;color:#0b1f3a}
.adm-tl{font:400 12.5px/1.4 'IBM Plex Sans',sans-serif;color:#64748b;margin-top:4px}
.adm-alert{display:flex;gap:10px;align-items:center;border-radius:12px;padding:10px 14px;
  font:400 13.5px/1.4 'IBM Plex Sans',sans-serif;margin-bottom:8px}
.adm-alert.warn{background:#fff5e6;color:#8a4b00}
.adm-alert.dang{background:#fdecec;color:#9b1c1c}
.adm-alert.info{background:#eef4fd;color:#1d4f91}
.adm-t{font:600 21px/1.2 'IBM Plex Sans',sans-serif;color:#0b1f3a;margin:2px 0 2px}
.adm-ts{font:400 13px/1.4 'IBM Plex Sans',sans-serif;color:#64748b;margin-bottom:14px}
.adm-gl{font:600 11px/1 'IBM Plex Mono',monospace;letter-spacing:.07em;color:#7b8aa0;
  text-transform:uppercase;margin:6px 0 7px}
.adm-grp{background:#fff;border:1px solid #e6ebf2;border-radius:12px;overflow:hidden;margin-bottom:14px}
.adm-li{display:flex;align-items:center;gap:10px;padding:11px 14px;border-bottom:1px solid #eef2f7;
  font:400 14px/1.4 'IBM Plex Sans',sans-serif;color:#0b1f3a}
.adm-li:last-child{border-bottom:none}
.adm-li .r{margin-left:auto;text-align:right;color:#64748b;font-size:13px}
.adm-mut{color:#7b8aa0;font-size:12.5px}
.adm-pill{display:inline-block;font:500 11.5px/1 'IBM Plex Sans',sans-serif;padding:4px 9px;
  border-radius:999px;white-space:nowrap}
.adm-pill.ok{background:#e6f6ec;color:#136c37}.adm-pill.wa{background:#fff1dc;color:#8a4b00}
.adm-pill.ne{background:#eef1f5;color:#5b6b80}.adm-pill.ac{background:#e7efff;color:#1d4f91}
.adm-pill.da{background:#fdecec;color:#9b1c1c}
/* Navegación: radio como lista agrupada */
div.st-key-adm_nav div[role="radiogroup"]{background:#fff;border:1px solid #e6ebf2;border-radius:14px;
  overflow:hidden;gap:0 !important}
div.st-key-adm_nav label{margin:0 !important;padding:12px 14px !important;border-bottom:1px solid #eef2f7;
  width:100%;cursor:pointer;font:500 14.5px/1.3 'IBM Plex Sans',sans-serif;color:#0b1f3a}
div.st-key-adm_nav label:last-of-type{border-bottom:none}
div.st-key-adm_nav label > div:first-of-type{display:none !important}
div.st-key-adm_nav label:has(input:checked){background:#e7efff;color:#1d4f91}
div.st-key-adm_nav label::before{content:"";display:inline-block;width:22px;height:22px;
  border-radius:6px;margin-right:10px;vertical-align:middle}
div.st-key-adm_nav label:nth-of-type(1)::before{background:#2563eb}
div.st-key-adm_nav label:nth-of-type(2)::before{background:#16a34a}
div.st-key-adm_nav label:nth-of-type(3)::before{background:#d97706}
div.st-key-adm_nav label:nth-of-type(4)::before{background:#7c3aed}
div.st-key-adm_nav label:nth-of-type(5)::before{background:#e5484d}
</style>
"""


def inject() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def pill(text: str, kind: str = "ne") -> str:
    return f'<span class="adm-pill {kind}">{html.escape(text)}</span>'


def title(t: str, sub: str = "") -> None:
    st.markdown(f'<div class="adm-t">{html.escape(t)}</div>'
                + (f'<div class="adm-ts">{html.escape(sub)}</div>' if sub else ""),
                unsafe_allow_html=True)


def group(label: str, rows: Iterable[Tuple[str, str]]) -> None:
    """Lista agrupada. Cada fila = (izquierda HTML, derecha HTML) ya escapadas."""
    items = "".join(f'<div class="adm-li"><span>{l}</span><span class="r">{r}</span></div>'
                    for l, r in rows)
    st.markdown((f'<div class="adm-gl">{html.escape(label)}</div>' if label else "")
                + f'<div class="adm-grp">{items}</div>', unsafe_allow_html=True)
