"""Preview aislado de la vista de onda cruda dinámica (System1).
Solo para verificación visual local — no forma parte de la app.
    streamlit run _preview_dynamic_raw.py
"""
import streamlit as st

st.set_page_config(page_title="Dynamic Raw · Preview", layout="wide")
st.title("Dynamic analysis — System1 raw (preview)")
st.caption("Lee el bucket `dynamic_raw` y reconstruye onda / espectro / órbita.")

from core.dynamic_raw_view import render_dynamic_raw

render_dynamic_raw("SGT300B", "SGT300B")
