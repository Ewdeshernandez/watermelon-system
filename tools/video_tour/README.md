# Video del recorrido (ventas)

Graba el recorrido de 60 s de la plataforma (ES y EN) con datos reales, navegando
como cliente. Resultado: `watermelon_tour_es.mp4` / `watermelon_tour_en.mp4`
(~8 MB, 1600×1000) para watermelonsystem.com (`public/video/`) y WhatsApp.

## Requisitos (venv aparte, no en requirements del proyecto)

```bash
python3 -m venv /tmp/vvenv && /tmp/vvenv/bin/pip install playwright imageio-ffmpeg pillow numpy pymupdf
/tmp/vvenv/bin/python -m playwright install chromium
```

## 1. Lanzador local con sesión de cliente (NO commitear)

Crear `_video_app.py` en la raíz del repo (las páginas deben estar junto al script):

```python
import os, time
import streamlit as st
import core.auth as _a
_a._persist_session_cookie = lambda *a, **k: None
_now = int(time.time())
st.session_state.update({
    "auth_ok": True, "auth_email": os.environ["WM_VIDEO_EMAIL"],
    "auth_full_name": "Equipo", "auth_role": "client",
    "auth_started_at": _now, "auth_expires_at": _now + 6 * 3600,
    "live_asset_v3": os.environ["WM_VIDEO_INSTANCE"],
})
st.switch_page("pages/02_Live_Monitoring.py")
```

Correrlo en el puerto 8560 (config `video-app` de `.claude/launch.json`).
Solo es local: no hay login real.

## 2. Capturas

`WM_VIDEO_SCRUB` = lista JSON de textos que NO deben salir en un video público
(p. ej. la ubicación de planta). Los valores reales están en `.claude/napkin.md`.

```bash
export WM_VIDEO_SCRUB='["<ubicación>"]'
python cap_live.py shots/     # Live: arriba, tabla, tendencia, eventos, espejo
python cap_fix.py shots/      # tren completo + tendencia 30D
python cap_sub.py shots/      # Suscripción
python cap_mobile.py shots/   # app móvil (localhost:3200; requiere bypass dev temporal)
```

Páginas del reporte semanal: renderizar con pymupdf a `shots/rep_05.png`,
`rep_07.png`, `rep_13.png` y `rep_27.png` (hallazgos, Tabular, órbitas, Bode). No usar la portada.

## 3. Composición

```bash
WM_TOUR_ASSET="SGT300B · <cliente>" python compose.py shots/ watermelon_tour_es.mp4 es
WM_TOUR_ASSET="SGT300B · <client>"  python compose.py shots/ watermelon_tour_en.mp4 en
```

## Nunca en el video

- Ubicación de la planta.
- Recomendaciones del reporte del cliente.
- La oferta comercial de Suscripción. `compose.py` tapa esa zona.
- Portada del reporte.
- Indicador de desarrollo de Next.js.
- Íconos en las pestañas Waveform / Spectrum / Orbit.
