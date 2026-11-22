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


# CSS de la tabla densa de Live Monitoring (vive en el <style> de la página, que
# NO llega al iframe del componente) → lo embebemos para que el PNG salga igual.
_LIVE_TABLE_CSS = """<style>
table.wm-live-table{width:100%;border-collapse:separate;border-spacing:0;font-size:13px;
 background:#fff;border-radius:8px;overflow:hidden;border:1px solid #e5edf7;font-variant-numeric:tabular-nums;}
table.wm-live-table thead tr{background:#f8fafc;}
table.wm-live-table thead th{text-align:left;color:#475569;font-weight:800;font-size:10px;
 text-transform:uppercase;letter-spacing:.08em;padding:11px 14px;border-bottom:2px solid #d9e8fb;}
table.wm-live-table tbody td{padding:11px 14px;border-bottom:1px solid #f1f5f9;color:#0f172a;}
table.wm-live-table tbody tr:last-child td{border-bottom:none;}
table.wm-live-table tbody tr.row-alarm{background:#fff8e7;}
table.wm-live-table tbody tr.row-danger{background:#fef2f2;}
table.wm-live-table tbody tr.row-danger td:first-child{border-left:3px solid #ef4444;}
table.wm-live-table tbody tr.row-alarm td:first-child{border-left:3px solid #f59e0b;}
table.wm-live-table .col-num{text-align:right;font-weight:700;font-family:'SF Mono',Menlo,Consolas,monospace;}
table.wm-live-table .col-mono{font-family:'SF Mono',Menlo,Consolas,monospace;font-size:12px;color:#475569;}
table.wm-live-table .col-spark{width:100px;padding:6px 14px;}
</style>"""


def download_html_png_button(inner_html: str, filename: str, key: str,
                             title: str = "", extra_css: str = "",
                             width: int = 1200) -> None:
    """Captura un bloque HTML (tabla, event log…) a PNG con html2canvas y lo baja
    directo (nombre correcto + lluvia de sandías), para que CUALQUIER sección de
    Live Monitoring se pueda descargar/enviar igual que el trend. Fidelidad
    pixel-perfect a lo que ve el cliente: pills, sparklines SVG, glifos, colores.

    - inner_html: el MISMO HTML que se pinta en pantalla (con sus estilos inline).
    - title: encabezado opcional arriba del PNG (ej. tag de la máquina + fecha).
    - extra_css: <style>…</style> adicional si el HTML usa clases.
    """
    import base64
    import html as _h
    import re as _re2

    import streamlit.components.v1 as _components
    eid = "wmh_" + _re2.sub(r"[^a-zA-Z0-9]", "", str(key))[:24]
    hdr = (f'<div style="font:700 14px -apple-system,system-ui,sans-serif;color:#0f172a;'
           f'margin:0 0 12px 0;">{_h.escape(title)}</div>') if title else ""
    cap_html = (_LIVE_TABLE_CSS + (extra_css or "")
                + f'<div style="width:{int(width)}px;background:#fff;padding:16px 18px;box-sizing:border-box;'
                f'font-family:-apple-system,system-ui,sans-serif;">' + hdr + inner_html + '</div>')
    b64 = base64.b64encode(cap_html.encode("utf-8")).decode()
    fn = _h.escape(filename, quote=True)
    # NO html2canvas: el iframe de components.html es de ORIGEN NULO (sandbox sin
    # allow-same-origin), y html2canvas clona el DOM a un iframe propio que en ese
    # contexto es cross-origin → SecurityError → "Error PNG". En su lugar:
    # serializar el nodo a XHTML, meterlo en un <svg><foreignObject>, cargarlo como
    # <img> y pintarlo en un canvas. No clona iframes → funciona en origen nulo.
    # El <style> va DENTRO del foreignObject para que apliquen las clases
    # (wm-live-table, etc). Verificado en navegador real (origen nulo, sin taint).
    html = """
<div id="__EID___cap" style="position:absolute;left:-10000px;top:0;"></div>
<button id="__EID__" style="display:inline-flex;align-items:center;gap:9px;border:none;
   font-family:system-ui,-apple-system,sans-serif;font-weight:600;font-size:13.5px;
   background:linear-gradient(135deg,#12305e,#274b7d);color:#fff;padding:10px 18px;
   border-radius:10px;border:1px solid #0e2547;box-shadow:0 2px 6px rgba(11,31,58,.18);
   transition:transform .08s ease,box-shadow .15s ease;cursor:pointer">
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#e8890c"
       stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
    <path d="M12 3v12"/><path d="m7 11 5 5 5-5"/><path d="M5 21h14"/></svg>
  Descargar PNG
</button>
<script>
(function(){
  var cap=document.getElementById('__EID___cap');
  try{ cap.innerHTML=decodeURIComponent(escape(atob('__B64__'))); }catch(e){ cap.textContent='err'; }
  var b=document.getElementById('__EID__');
  b.onmouseenter=function(){b.style.transform='translateY(-1px)';b.style.boxShadow='0 6px 16px rgba(11,31,58,.28)';};
  b.onmouseleave=function(){b.style.transform='none';b.style.boxShadow='0 2px 6px rgba(11,31,58,.18)';};
  function wmMelons(){
    for(var i=0;i<24;i++){
      var m=document.createElement('div'); m.textContent='\\uD83C\\uDF49';
      m.style.cssText='position:fixed;left:'+(Math.random()*100)+'vw;top:-24px;'+
        'font-size:'+(15+Math.random()*20)+'px;z-index:99999;pointer-events:none;'+
        'transition:transform '+(1.3+Math.random()*1.1)+'s ease-in,opacity .5s ease-out';
      document.body.appendChild(m);
      (function(mm){requestAnimationFrame(function(){mm.style.transform='translateY(130vh) rotate('+(Math.random()*720-360)+'deg)';});
        setTimeout(function(){mm.style.opacity='0';},1400); setTimeout(function(){mm.remove();},2600);})(m);
    }
  }
  b.onclick=async function(){
    b.disabled=true; var old=b.innerHTML; b.innerHTML='Generando…';
    try{
      // Mover style + contenido a un div XHTML (namespace) para serializar válido.
      var wrap=document.createElement('div');
      wrap.setAttribute('xmlns','http://www.w3.org/1999/xhtml');
      for(var i=0;i<cap.childNodes.length;i++)wrap.appendChild(cap.childNodes[i].cloneNode(true));
      wrap.style.position='absolute'; wrap.style.left='-10000px'; wrap.style.top='0';
      document.body.appendChild(wrap);
      var target=wrap.querySelector('div')||wrap;
      await new Promise(function(r){requestAnimationFrame(function(){requestAnimationFrame(r);});});
      var w=Math.ceil(target.scrollWidth)||1000, h=Math.ceil(target.scrollHeight)||100;
      var xhtml=new XMLSerializer().serializeToString(wrap);
      var svg='<svg xmlns="http://www.w3.org/2000/svg" width="'+w+'" height="'+h+'">'+
              '<foreignObject x="0" y="0" width="'+w+'" height="'+h+'">'+xhtml+'</foreignObject></svg>';
      var img=new Image();
      await new Promise(function(res,rej){
        img.onload=res; img.onerror=function(){rej(new Error('svg img load'));};
        img.src='data:image/svg+xml;charset=utf-8,'+encodeURIComponent(svg);
      });
      var sc=2;
      var canvas=document.createElement('canvas'); canvas.width=w*sc; canvas.height=h*sc;
      var ctx=canvas.getContext('2d');
      ctx.fillStyle='#ffffff'; ctx.fillRect(0,0,canvas.width,canvas.height);
      ctx.scale(sc,sc); ctx.drawImage(img,0,0);
      var url=canvas.toDataURL('image/png');            // lanza si el canvas está tainted
      wrap.remove();
      if(!canvas.width||!canvas.height||url.length<256){throw new Error('canvas vacio');}
      var link=document.createElement('a');
      link.href=url; link.download='__FN__';
      document.body.appendChild(link); link.click(); link.remove(); wmMelons();
    }catch(e){ b.textContent='Error: '+((e&&e.message)||'PNG');
      setTimeout(function(){b.innerHTML=old;b.disabled=false;},2200); return; }
    b.innerHTML=old; b.disabled=false;
  };
})();
</script>
"""
    html = (html.replace("__EID__", eid).replace("__B64__", b64)
            .replace("__FN__", fn))
    _components.html(html, height=60)


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
