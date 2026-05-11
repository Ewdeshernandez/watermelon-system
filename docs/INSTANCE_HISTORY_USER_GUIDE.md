# Watermelon System — Histórico de Análisis (v3.31.76-85)

Guía rápida del flow completo: especialista sube CSVs → cliente ve resultados en Live Monitoring → cliente recibe ZIP por email.

## Flow operativo

```
┌──────────────────────┐        ┌────────────────────────┐        ┌──────────────────────┐
│   Especialista       │        │   Supabase Storage     │        │   Cliente            │
│   (admin/specialist) │        │   bucket               │        │   (lectura)          │
│                      │        │   `instance-history`   │        │                      │
│  1. Sube CSVs en     │───────▶│   {instance_id}/       │───────▶│  Live Monitoring:    │
│     Load Data        │        │     waveform/          │        │   📊 Últimos         │
│  2. Selecciona       │        │     spectrum/          │        │      análisis        │
│     activo (TES1)    │        │     orbit/             │        │                      │
│  3. Click ✓Guardar   │        │     tabular/           │        │   Click "Ver         │
│     todo y procesar  │        │     {ts}.json.gz       │        │   detalle" →         │
│                      │        │                        │        │   plot Plotly        │
│  4. Live Monitoring: │        │                        │        │   inline             │
│     📦 Exportar →    │───────▶│   _exports/            │───────▶│  Email con link      │
│     ZIP o email      │        │     {id}_{ts}.zip      │        │  → descarga ZIP      │
└──────────────────────┘        └────────────────────────┘        └──────────────────────┘
```

## Setup inicial (1 sola vez)

Si querés que los snapshots persistan contra restart de Streamlit Cloud Free:

1. Corré el SQL en Supabase Dashboard → SQL Editor:
   ```
   ~/Documents/WatermelonSystem/data/storage_instance_history_setup.sql
   ```
   Crea 2 buckets: `instance-history` (público) + `instance-history-backups` (privado).

2. Verificá en Streamlit Cloud Settings → Secrets que esté configurado:
   ```toml
   [supabase]
   url = "https://...supabase.co"
   anon_key = "sb_publishable_..."
   service_key = "sb_secret_..."   # solo para backend, no se expone al cliente
   ```

Si **no** corres el SQL ni configurás `anon_key`, el sistema cae a **disco local** automáticamente (`data/instances/{instance_id}/history/`). Los snapshots se borran cuando Streamlit Cloud reinicia el container (~24h sin tráfico).

## Workflow del especialista (todos los días, mañana y tarde)

### Paso 1: subir CSVs

1. Andá a **Load Data** (sidebar Ingest)
2. Verificá que la **instancia activa** sea la correcta (Machinery Library selector)
3. Drag-and-drop los CSVs de la corrida (vibration, ops, etc.)
4. El sistema parsea automáticamente y muestra el ready box

### Paso 2: guardar para Live Monitoring (1 click)

5. Expander **"💾 Guardar como snapshot para Live Monitoring"**:
   - Por defecto los 4 tipos (Waveform, Spectrum, Orbit, Tabular) están marcados
   - Opcionalmente: label de la corrida (ej. "Inspección semanal 12-may")
   - Opcionalmente: notas
   - Opcionalmente: velocidad rotacional RPM (mejora vectores 1X/2X)
6. Click **"✓ Guardar todo y continuar al procesamiento"**
7. Spinner mientras computa FFT + métricas + auto-detect pares X/Y
8. Mensaje "✓ N snapshots guardados — ya visibles en Live Monitoring"

### Paso 3: procesar análisis (workflow normal)

9. Seguís al módulo de análisis que quieras (Spectrum, Waveform, Orbit, Tabular)
10. El especialista hace su trabajo normal — fine-tune FFT, drill-down, diagnóstico, etc.

## Workflow del cliente (cuando entra a la app)

1. Click **Live Monitoring** en el sidebar
2. Selecciona el activo de interés
3. Scroll hacia abajo del diagrama → sección **"📊 Últimos análisis del activo"**
4. 4 cards con:
   - Timestamp del más reciente ("hace 2 h")
   - Label de la corrida del especialista
   - Quick stats (# sensores, peak severity, etc.)
5. Click **"Ver detalle"** en cualquier card → plot Plotly inline:
   - **Waveform**: time-series + tabla de métricas
   - **Spectrum**: FFT con top peaks identificados
   - **Orbit**: scatter X-Y por bearing
   - **Tabular**: tabla completa de canales con severidad ISO

## Enviar ZIP al cliente por email (en momentos clave)

Cuando el especialista quiere mandarle al cliente todo el histórico:

1. Live Monitoring → seleccionar activo
2. Scroll hasta **"📦 Exportar histórico del activo"**
3. Dos modos:

   **A) Descargar ZIP local** (especialista guarda en su computadora):
   - Click "📥 Descargar ZIP" → popover
   - Marca "Incluir diagrama" para que el ZIP incluya el SVG del momento
   - Click "Generar ZIP" → spinner → botón "⬇ download"

   **B) Enviar al cliente directo**:
   - Click "📧 Enviar al cliente" → popover
   - Ingresar email del cliente + nota opcional
   - Click "Generar link y abrir email"
   - El sistema sube el ZIP a Supabase Storage (link válido 24h) y abre tu cliente de email con todo pre-cargado
   - El cliente recibe el email, hace click en el link, descarga el ZIP

## Contenido del ZIP

```
{instance_id}_history_{ts}.zip
├── manifest.json              ← metadata del export (cliente, fecha, total snapshots, etc.)
├── README.txt                 ← instrucciones de cómo abrir/interpretar
├── diagram.svg                ← snapshot visual del Live Monitoring (si se incluyó)
├── diagram.png                ← idem en PNG
└── snapshots/
    ├── scl/
    │   └── scl_20260511_153022.json
    ├── polar/
    ├── bode/
    ├── waveform/
    ├── spectrum/
    ├── orbit/
    └── tabular/
```

Los JSON están **descomprimidos** (no .gz) para que el cliente pueda abrirlos con cualquier editor de texto.

## Retención automática (LRU)

- **10 snapshots máx por (instance, tipo)**: al insertar el #11, el más viejo se borra automáticamente
- Sin acción manual necesaria
- Cubre típicamente 5 días de corridas (2/día mañana+tarde) en hot retention
- Si necesitás más histórico, exportá ZIPs periódicamente

## Troubleshooting

**Snapshots no aparecen en Live Monitoring después de guardar**:
- Verificá en Supabase Dashboard → Storage → bucket `instance-history` que los archivos estén ahí
- Si NO están: el sistema cayó a disco local (probable que falte `anon_key` en secrets)
- Hard reload de la página (`Cmd+Shift+R`)

**"Ver detalle" no muestra plot**:
- El snapshot puede estar corrupto. Mirá los logs de Streamlit (Manage app → Logs)
- O el payload no incluye time/values (algunos CSVs sin columna vibración)

**ZIP de export muy grande (>5 MB)**:
- El bucket Supabase corta uploads >5 MB
- Solución: descargá ZIP local y mandalo por WeTransfer/Google Drive
- O usá retención más agresiva (menos snapshots)

**Email no abre**:
- Algunos navegadores (Safari, Firefox 121+) bloquean `mailto:` desde iframe sandbox
- El sistema muestra el link en una caja code para que copies manualmente

## Modules involucrados (referencia técnica)

| Módulo | Responsabilidad |
|---|---|
| `core/history_storage.py` | Backend Supabase Storage + gzip + LRU |
| `core/snapshot_batch_builder.py` | Convierte parsed CSVs en payloads para los 4 tipos |
| `core/waveform_history.py` | Snapshots de waveforms |
| `core/spectrum_history.py` | Snapshots de spectrums FFT |
| `core/orbit_history.py` | Snapshots de orbits con auto-detect X/Y |
| `core/tabular_history.py` | Snapshots de tabular list |
| `core/scl_history.py` | Snapshots SCL (refactorizado al backend nuevo) |
| `core/polar_history.py` | Snapshots Polar (refactorizado) |
| `core/bode_history.py` | Snapshots Bode (refactorizado) |
| `core/recent_analyses_widget.py` | Sección "📊 Últimos análisis" en Live Monitoring |
| `core/history_export_widget.py` | Sección "📦 Exportar histórico" |
| `core/snapshot_save_widget.py` | Helper genérico para save desde cualquier módulo (opcional) |

## Próximos features pendientes

- `trend_history.py` → migrar al backend nuevo (CSVs binarios, requiere extension de `history_storage`)
- Backup automático semanal con pg_cron (descomentar Opción A en `storage_diagram_shares_setup.sql`)
- Cards comparativas entre snapshots históricos
- Export PDF (no solo ZIP)
