# Watermelon System

**Industrial Vibration Intelligence**
Plataforma de análisis, monitoreo y diagnóstico de vibraciones industriales con
soporte para análisis rotodinámico, diagnóstico de rodamientos, y generación de
reportes basados en normas internacionales.

> Repositorio en evolución activa. Producción está en `main` (deploy automático).
> Toda la nueva funcionalidad se desarrolla en `dev` antes de promover a `main`.

---

## Stack técnico

- **Lenguaje:** Python 3.10+
- **UI:** Streamlit (multi-página)
- **Numérica:** NumPy, SciPy, Pandas
- **Gráficos:** Plotly (con exportador Kaleido)
- **Reportes:** ReportLab (PDF)
- **Auth:** PBKDF2-SHA256 (260K iteraciones) + `hmac.compare_digest`

---

## Arranque rápido

```bash
# 1. Clonar
git clone https://github.com/Ewdeshernandez/watermelon-system.git
cd watermelon-system

# 2. Entorno virtual
python3 -m venv .venv
source .venv/bin/activate     # en Windows: .venv\Scripts\activate

# 3. Dependencias
pip install -r requirements.txt

# 4. Configurar secrets (NO commitear)
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Editar .streamlit/secrets.toml y reemplazar password_hash con hashes reales
python tools/generate_password_hash.py  # genera hashes PBKDF2

# 5. Correr la app
streamlit run app.py
```

La app abre en `http://localhost:8501`. Login obligatorio antes de acceder a cualquier
página de análisis.

---

## Estructura del proyecto

```
WatermelonSystem/
├── app.py                  # Router de entrada (login → home)
├── 00_Home.py              # Home autenticada
├── pages/                  # Páginas de análisis (Streamlit multipage)
│   ├── 00_Login.py
│   ├── 01_Load_Data.py     # Importación CSV (Bently Nevada, GE, etc.)
│   ├── 02_Time_Waveforms.py
│   ├── 03_Spectrum.py      # FFT + diagnóstico
│   ├── 04_Trends.py        # Análisis de tendencias multi-fecha
│   ├── 05_Orbit_Analysis.py
│   ├── 06_Polar_Plot.py
│   ├── 07_Bode_Plot.py
│   ├── 09_Shaft_Centerline.py
│   ├── 13_Phase_Analysis.py
│   ├── 15_Diagnostics.py
│   └── 16_Reports.py
├── core/                   # Lógica de análisis (sin Streamlit)
│   ├── orbit.py            # Órbita filtrada, precesión, geometría de sondas
│   ├── phase.py            # Análisis de fase 1X (sync geométrica)
│   ├── order_tracking.py   # Order tracking 1X-NX
│   ├── tsa.py              # Time Synchronous Average
│   ├── spectrum_*.py       # Análisis espectral y diagnóstico
│   ├── bearing_*.py        # Catálogo y frecuencias de falla
│   ├── waveform_*.py       # Métricas e impactos en waveform
│   ├── diagnostics.py      # Semáforos, narrativa, severidad
│   ├── auth.py             # Autenticación PBKDF2
│   └── ui/                 # Tema y header reutilizable
├── modules/                # (futuro) parsers reutilizables
├── tools/                  # Utilidades de mantenimiento
├── assets/                 # Logos, imágenes
├── data/                   # Catálogos (bearing_catalog.csv) y estado runtime
├── .streamlit/             # config y secrets
└── requirements.txt
```

---

## Normas de referencia

Watermelon System apunta a alinear sus diagnósticos con:

- **ISO 20816-2** (antigua ISO 7919-2 / ISO 10816-2): severidad de vibración
  en máquinas grandes con cojinetes planos (turbogeneradores >40 MW).
- **API 670**: cadena de medición con sondas de proximidad.
- **API 684**: rotodinámica, márgenes de separación, factor de amplificación Q.
- **ISO 21940**: balanceo residual y grados de balance G.
- **ISO 13373** (series): diagnóstico avanzado (espectro, órbita, demodulación).

---

## Flujo de desarrollo

```
main  ◄── (release controlado, deploy live)
  │
  └── dev  ◄── (integración estable)
       │
       ├── chore/repo-hygiene
       ├── feat/csv-loader-extract
       ├── feat/waterfall
       ├── feat/campbell
       └── ...
```

**Reglas:**
1. Toda feature/fix arranca en una branch desde `dev`.
2. Antes de cualquier cambio destructivo: crear tag `pre-<descripcion>-YYYYMMDD`.
3. Commits convencionales: `feat:`, `fix:`, `refactor:`, `chore:`, `docs:`.
4. Merge a `dev` solo tras pruebas locales con `streamlit run app.py`.
5. Merge a `main` solo tras revisión y tag de release `vX.Y.Z`.
6. NUNCA push directo a `main` sin pasar por `dev`.

---

## Tags de retorno

El repo mantiene tags como puntos de retorno seguros. Para volver a un punto:

```bash
git checkout <tag-name>           # explorar un estado pasado
git checkout -b rescue/<nombre>   # crear rama de rescate desde ese punto
```

---

## Licencia

Pendiente de definir. Por ahora todos los derechos reservados.

---

## Estado actual

- **Versión:** v0.1-demo-interno
- **Producción:** [watermelonsystem.app](https://watermelonsystem.app) (rama `main`)
- **Roadmap próximo:**
  - Extraer parser CSV a `core/csv_loader.py`
  - Implementar Waterfall (cascada FFT vs RPM)
  - Implementar Campbell Diagram con margen API 684
  - Implementar Envelope Spectrum (demodulación Hilbert)
  - Anclar diagnóstico a ISO 20816-2 zonas A/B/C/D
