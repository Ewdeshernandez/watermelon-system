# Watermelon System — Project Context

Auto-loaded by Claude Code. Public-safe context. Sensitive client info lives in `.claude/napkin.md` (gitignored).

## Stack

- Streamlit Cloud frontend
- Supabase (Postgres) backend
- GitHub for code, this repo
- Versión actual: see `VERSION` file
- Convención de release: `VERSION` file + git tags + sync entre `dev` y `main`

## Convenciones de código

- `page_header()` requiere `subtitle` obligatorio, no opcional.
- Bently 3500 bearings: `1` = lado libre (CRF en aero), `2` = lado coupling (TRF).
- Sensor labels en SVG sin underscore en display (ej: `2YA`, no `2Y_A`). En código/datos sí va con underscore.

## Bugs recurrentes resueltos

- VERSION mismatch en scripts de push: usar `set -euo pipefail` + verificación de tag remoto antes de push.
- Confusión de bearings 1/2: ver convención arriba antes de tocar mapeos.

## Preferencias de colaboración

- Respuestas breves, estilo directo.
- Visual de monitoreo objetivo: clase System1/AMS, no PowerPoint.
- SVG vectorial OK, pero proporciones realistas.

## Próximo

- Mejorar Live Monitoring usando frontend-design skill.
- Integrar SVG profesional dibujado en Inkscape para diagramas de equipos.

## Qué NO va aquí

Nombres de cliente, ubicaciones de planta, modelos exactos de equipos en producción. Eso vive en `.claude/napkin.md` (local, gitignored).
