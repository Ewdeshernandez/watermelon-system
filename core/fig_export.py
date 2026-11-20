"""
core/fig_export.py — Descarga de figuras Plotly como PNG (botón clase mundial).
================================================================================

Reutilizable entre páginas (Rotodinámica, Live Monitoring, …) para que TODOS los
gráficos tengan el mismo botón de descarga 🍉 (data-URI → baja directo a Downloads
con el nombre correcto + lluvia de sandías). Antes vivía solo en pages/02_Analysis.
"""
from __future__ import annotations

import streamlit as st


@st.cache_data(show_spinner=False)
def fig_png_cached(fig_json: str) -> bytes:
    """Render PNG de una figura Plotly (kaleido). Cacheado por contenido."""
    import plotly.io as _pio
    return _pio.from_json(fig_json).to_image(format="png", scale=2)


def download_png_button(png_bytes: bytes, filename: str, key: str = "wmdl") -> None:
    """Botón de descarga: data-URI (baja DIRECTO a Downloads con el nombre correcto,
    sin pasar por el endpoint /media de Streamlit) + lluvia de sandías 🍉."""
    import base64
    import html as _h
    import re as _re2

    import streamlit.components.v1 as _components
    b64 = base64.b64encode(png_bytes).decode()
    fn = _h.escape(filename, quote=True)
    eid = "wmdl_" + _re2.sub(r"[^a-zA-Z0-9]", "", str(key))[:24]
    html = """
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@600&display=swap" rel="stylesheet">
<a id="__EID__" href="data:image/png;base64,__B64__" download="__FN__" onclick="wmMelons()"
   style="display:inline-flex;align-items:center;gap:9px;text-decoration:none;
          font-family:'IBM Plex Sans',system-ui,sans-serif;font-weight:600;font-size:13.5px;
          background:linear-gradient(135deg,#12305e,#274b7d);color:#fff;padding:10px 18px;
          border-radius:10px;border:1px solid #0e2547;box-shadow:0 2px 6px rgba(11,31,58,.18);
          transition:transform .08s ease,box-shadow .15s ease;cursor:pointer">
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#e8890c"
       stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
    <path d="M12 3v12"/><path d="m7 11 5 5 5-5"/><path d="M5 21h14"/></svg>
  Descargar imagen
</a>
<script>
const a=document.getElementById('__EID__');
a.onmouseenter=()=>{a.style.transform='translateY(-1px)';a.style.boxShadow='0 6px 16px rgba(11,31,58,.28)';};
a.onmouseleave=()=>{a.style.transform='none';a.style.boxShadow='0 2px 6px rgba(11,31,58,.18)';};
function wmMelons(){
  for(let i=0;i<30;i++){
    const m=document.createElement('div'); m.textContent='\U0001f349';
    m.style.cssText='position:fixed;left:'+(Math.random()*100)+'vw;top:-24px;'+
      'font-size:'+(15+Math.random()*20)+'px;z-index:99999;pointer-events:none;'+
      'transition:transform '+(1.3+Math.random()*1.1)+'s ease-in,opacity .5s ease-out';
    document.body.appendChild(m);
    requestAnimationFrame(()=>{m.style.transform='translateY(130vh) rotate('+(Math.random()*720-360)+'deg)';});
    setTimeout(()=>{m.style.opacity='0';},1400);
    setTimeout(()=>{m.remove();},2600);
  }
}
</script>
"""
    html = html.replace("__EID__", eid).replace("__B64__", b64).replace("__FN__", fn)
    _components.html(html, height=58)


def download_fig_button(fig, filename: str, key: str) -> None:
    """Descarga una figura Plotly como PNG con el botón+sandías. Cacheado por
    contenido → no re-renderiza en cada interacción (kaleido es lento)."""
    try:
        import plotly.io as _pio
        png = fig_png_cached(_pio.to_json(fig))
    except Exception:  # noqa: BLE001
        st.caption("Descarga PNG no disponible (kaleido).")
        return
    download_png_button(png, filename, key=key)
