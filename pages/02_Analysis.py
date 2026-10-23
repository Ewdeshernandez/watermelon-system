"""
pages/02_Analysis.py — Workspace ÚNICO de análisis (Fase 2).
================================================================

Reemplaza (gradualmente) las 7 páginas sueltas (Waveform, Spectrum, Orbit,
Polar, Bode, Shaft Centerline, Trends). Flujo clase ADRE/System1: cargas UN
registro (CSV de ingesta manual) + eliges canal → navegas las VISTAS con un
segmented control, sin recargar el dato. Las vistas que necesitan par X/Y +
keyphasor se avisan si el CSV no los trae (honesto, no se fabrica).

Estado (2026-10-23): shell + Waveform + Espectro REALES; Orbit/Polar/Bode/
Shaft/Trends en migración. El dato vive 1 vez en session_state.
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.auth import require_login, render_user_menu

st.set_page_config(page_title="Watermelon System | Análisis", layout="wide",
                   initial_sidebar_state="collapsed")
require_login()
render_user_menu()

from core.ui_theme import apply_watermelon_page_style, page_header
from core.ui_industrial import inject_industrial_css

apply_watermelon_page_style()
inject_industrial_css()
page_header("Análisis", "Ingesta manual de CSV · todas las vistas en un solo lugar")

SS = st.session_state


# =========================================================
# Carga del registro (una sola vez → session_state)
# =========================================================
@st.cache_data(show_spinner=False)
def _read_csv(data: bytes) -> pd.DataFrame:
    # Autodetecta separador (coma / punto y coma) y saltea cabeceras de metadatos
    # de System1 (líneas 'Clave,Valor' antes de la tabla x,y).
    txt = data.decode("utf-8", "replace")
    # buscar la línea de encabezado de datos
    lines = txt.splitlines()
    start = 0
    for i, ln in enumerate(lines):
        low = ln.lower().replace(" ", "")
        if low.startswith("x-axis") or low.startswith("time") or low.startswith("t_s") \
                or ("," in ln and sum(c.isdigit() for c in ln) > 4 and i > 0):
            start = i
            break
    try:
        return pd.read_csv(io.StringIO("\n".join(lines[start:])), sep=None,
                           engine="python")
    except Exception:
        return pd.read_csv(io.StringIO(txt), sep=None, engine="python")


up = st.file_uploader("Registro CSV importado", type=["csv"],
                      help="Export de System1 o cualquier CSV con columna de tiempo + amplitud.")
if up is not None:
    try:
        SS["an_df"] = _read_csv(up.getvalue())
        SS["an_name"] = up.name
    except Exception as exc:  # noqa: BLE001
        st.error(f"No se pudo leer el CSV: {exc}")

df: pd.DataFrame | None = SS.get("an_df")
if df is None or df.empty:
    st.info("Carga un CSV para empezar. Las vistas (Waveform · Espectro · Órbita · "
            "Polar · Bode · Shaft Centerline · Trends) se habilitan según las columnas "
            "que traiga el archivo.")
    st.stop()

num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
if len(num_cols) < 1:
    st.warning("El CSV no tiene columnas numéricas que graficar.")
    st.stop()


# =========================================================
# Selección de señal (tiempo + canal X / Y)
# =========================================================
c1, c2, c3 = st.columns([1.3, 1, 1])
with c1:
    st.caption(f"Registro · {SS.get('an_name','')}  ·  {len(df):,} muestras")
    t_col = st.selectbox("Tiempo (eje X)", ["(índice)"] + list(df.columns), index=0)
with c2:
    x_col = st.selectbox("Canal X", num_cols, index=0)
with c3:
    _y_opts = ["(ninguno)"] + [c for c in num_cols if c != x_col]
    y_col = st.selectbox("Canal Y (para órbita)", _y_opts, index=0)

t = (df.index.to_numpy(dtype=float) if t_col == "(índice)"
     else pd.to_numeric(df[t_col], errors="coerce").to_numpy(dtype=float))
x = pd.to_numeric(df[x_col], errors="coerce").to_numpy(dtype=float)
y = (pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=float)
     if y_col != "(ninguno)" else None)
has_xy = y is not None
_fs = None
if t.size >= 2:
    dt = float(np.nanmedian(np.diff(t)))
    if dt > 0:
        _fs = 1.0 / dt


# =========================================================
# Vistas (segmented control) — dato compartido, no recarga
# =========================================================
VIEWS = ["Waveform", "Espectro", "Órbita", "Polar", "Bode", "Shaft Centerline", "Trends"]
view = st.segmented_control("Vista", VIEWS, default="Waveform",
                            selection_mode="single") or "Waveform"

_NAVY = "#12305e"


def _layout(fig, title):
    fig.update_layout(
        template="plotly_white", height=430, title=dict(text=title, font=dict(size=15)),
        margin=dict(l=60, r=24, t=48, b=48), font=dict(family="IBM Plex Sans"),
        showlegend=False)
    return fig


if view == "Waveform":
    fig = go.Figure(go.Scatter(x=t, y=x, mode="lines", line=dict(color=_NAVY, width=1.3)))
    fig.update_xaxes(title=("Tiempo" if t_col != "(índice)" else "Muestra"))
    fig.update_yaxes(title=x_col)
    st.plotly_chart(_layout(fig, f"Forma de onda · {x_col}"), use_container_width=True)

elif view == "Espectro":
    xv = x[np.isfinite(x)]
    if xv.size < 8 or _fs is None:
        st.warning("Espectro: faltan muestras o no se pudo estimar la frecuencia de "
                   "muestreo (revisa la columna de tiempo).")
    else:
        xw = xv - np.mean(xv)
        sp = np.abs(np.fft.rfft(xw * np.hanning(xw.size))) * (2.0 / xw.size)
        fr = np.fft.rfftfreq(xw.size, d=1.0 / _fs)
        fig = go.Figure(go.Scatter(x=fr, y=sp, mode="lines", line=dict(color=_NAVY, width=1.1)))
        fig.update_xaxes(title="Frecuencia (Hz)")
        fig.update_yaxes(title=f"Amplitud · {x_col}")
        st.plotly_chart(_layout(fig, f"Espectro (FFT) · {x_col}"), use_container_width=True)

elif view == "Órbita":
    if not has_xy:
        st.warning("Órbita: necesita par **X/Y** (proximidad). Selecciona un Canal Y arriba.")
    else:
        fig = go.Figure(go.Scatter(x=x, y=y, mode="lines", line=dict(color=_NAVY, width=1.3)))
        fig.update_xaxes(title=x_col, scaleanchor="y", scaleratio=1)
        fig.update_yaxes(title=y_col)
        st.plotly_chart(_layout(fig, f"Órbita · {x_col} / {y_col}"), use_container_width=True)

else:
    _need = {"Polar": "par X/Y + keyphasor y barrido de rpm (run-up/coast-down)",
             "Bode": "par X/Y + keyphasor y barrido de rpm",
             "Shaft Centerline": "gap DC de los proximitores X/Y",
             "Trends": "varios registros en el tiempo (histórico)"}
    st.info(f"**{view}** — en migración (Fase 2). Requiere: {_need.get(view, '')}. "
            "Por ahora disponible en su página clásica del menú.")

st.caption("Workspace unificado (Fase 2) · migración incremental desde las páginas clásicas.")
