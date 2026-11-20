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


def _round_png(png_bytes, radius=18, border="#e2e8f2", border_w=2):
    import io

    from PIL import Image, ImageDraw
    im = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    w, h = im.size
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, w - 1, h - 1], radius=radius, fill=255)
    im = Image.composite(im, Image.new("RGB", (w, h), "white"), mask)
    ImageDraw.Draw(im).rounded_rectangle([border_w // 2, border_w // 2,
                                          w - 1 - border_w // 2, h - 1 - border_w // 2],
                                         radius=radius, outline=border, width=border_w)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def table_png(cols, rows, title=""):
    """Tabla simple como PNG (header navy + zebra) para adjuntar bajo un gráfico."""
    import io

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ncol, nrow = len(cols), len(rows)
    fig_w = min(1.2 * ncol + 1.0, 16)
    fig_h = 0.56 * nrow + (0.95 if title else 0.45)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200)
    ax.axis("off"); ax.set_position([0, 0, 1, 1])
    _bot = min(max(0.36 / fig_h, 0.04), 0.12)
    _top = 1.0 - (0.40 / fig_h if title else 0.02)
    if title:
        fig.text(0.010, 0.985, title, fontsize=9, fontweight="normal",
                 color="#5a6b85", va="top", ha="left")
    tbl = ax.table(cellText=rows, colLabels=cols, cellLoc="center",
                   bbox=[0.006, _bot, 0.988, _top - _bot])
    tbl.auto_set_font_size(False); tbl.set_fontsize(8.4)
    for j in range(ncol):
        h = tbl[0, j]; h.set_facecolor("#12305e")
        h.get_text().set_color("white"); h.get_text().set_fontweight("bold")
    for i in range(nrow):
        for j in range(ncol):
            cc = tbl[i + 1, j]
            cc.set_facecolor("#f4f7fb" if i % 2 else "#ffffff")
            cc.get_text().set_color("#0b1f3a")
        tbl[i + 1, 0].get_text().set_fontweight("bold")
    colw = [max([len(str(cols[j]))] + [len(str(rows[i][j])) for i in range(nrow)])
            for j in range(ncol)]
    _tot = float(sum(colw)) or 1.0
    for j in range(ncol):
        for i in range(nrow + 1):
            tbl[i, j].set_width(colw[j] / _tot)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="white"); plt.close(fig)
    return _round_png(buf.getvalue())


def combine_vertical(png_top, png_bottom, gap=22, bottom_frac=1.0):
    """Apila dos PNG verticalmente (centrados, fondo blanco) → un solo PNG."""
    import io

    from PIL import Image
    a = Image.open(io.BytesIO(png_top)).convert("RGB")
    b = Image.open(io.BytesIO(png_bottom)).convert("RGB")
    if bottom_frac and bottom_frac < 1.0 and b.width > 1:
        _nw = max(1, int(a.width * bottom_frac))
        _nh = max(1, int(b.height * _nw / b.width))
        b = b.resize((_nw, _nh), Image.LANCZOS)
    w = max(a.width, b.width)
    out = Image.new("RGB", (w, a.height + gap + b.height), "white")
    out.paste(a, ((w - a.width) // 2, 0))
    out.paste(b, ((w - b.width) // 2, a.height + gap))
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return buf.getvalue()


def download_fig_table_button(fig, cols, rows, filename, key, table_title="",
                              table_frac=1.0) -> None:
    """Descarga gráfico + tabla apilados en una sola imagen (el 🍉 habitual)."""
    try:
        import plotly.io as _pio
        top = fig_png_cached(_pio.to_json(fig))
        combined = combine_vertical(top, table_png(cols, rows, table_title),
                                    bottom_frac=table_frac)
    except Exception:  # noqa: BLE001
        download_fig_button(fig, filename, key)
        return
    download_png_button(combined, filename, key=key)
