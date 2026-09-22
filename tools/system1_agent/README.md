# Watermelon System1 Agent

Trae la **onda cruda dinámica con keyphasor** de Bently System1 (VM Parex) al
Watermelon Cloud **cada hora, sin que nadie la exporte a mano**. Watermelon Live
· *Dynamic analysis* la reconstruye en forma de onda, espectro (FFT, órdenes
1X/2X/3X) y órbita.

```
System1 (VM Parex)
  ├─ s1_rpa_export.py  (RPA, cada hora)  → replica el 'clic derecho → Export to CSV'
  │      de cada onda → carpeta  Desktop\CSV\  (1xd.csv, 1yd.csv, ...)
  └─ s1_agent.py --csv  (cada hora)      → lee esos CSV, empareja X/Y por cojinete,
         arma el formato Watermelon (con keyphasor) y sube a Supabase: dynamic_raw
              └─ Watermelon Live · Dynamic analysis → onda / espectro / órbita
```

## Por qué así (lo que se verificó en el server, 2026-09-21)
- El backend de System1 es **SQL Server** (base `BNC_Databases`), **legible con
  Windows Auth sin password** — pero ahí solo vive la **configuración**. La
  **onda cruda NO está en SQL Server** (abrir una máquina no attacha ninguna base
  de datos de onda).
- La onda sale por el **`Export to CSV`** de cada gráfica (menú de clic derecho).
  Formato real: header `Clave,Valor` (Machine/Point/Number of Revs/Sample
  Speed/…) + `X-Axis Value,Y-Axis Value` + muestras (t en ms, amp). Keyphasor
  implícito: la onda arranca en la marca; cada `samples_per_rev` = 1 vuelta.
- Por eso el robot es **RPA** (automatiza ese export) + un **lector de carpeta**
  que convierte y sube. `pywinauto` fija el nombre de archivo por UIA (no por
  teclado) — esquiva el remapeo de layout del RDP/VMware anidado.

---

## Instalación en el server Parex (una vez)

1. **Copiar** esta carpeta a `C:\Watermelon\system1_agent\`.
2. **Python 3.9+** + deps: `python -m pip install -r requirements.txt`
3. **Supabase**: `config.example.toml` → `config.toml`, setear
   `[supabase] service_key` (o variables de entorno `SUPABASE_URL` /
   `SUPABASE_SERVICE_KEY`).
4. **Probar el canal a la nube** (sintético, sin tocar System1):
   ```
   python s1_agent.py --demo --once
   ```
   Abrir Watermelon Live → SGT300B → *Dynamic analysis*: deben verse
   onda/espectro/órbita. (Ya quedó verde desde el Mac.)
5. **Afinar el RPA** (1ª vez): con System1 abierto en la máquina/pantalla de onda:
   ```
   python s1_rpa_export.py --inspect
   ```
   Ajustar en `config.toml` → `[rpa]`: el `tag` de cada punto (texto del nodo del
   árbol), `plot_xy` (centro de la gráfica dentro de la ventana) y `out_folder`.
   Probar sin exportar: `python s1_rpa_export.py --run --dry`, luego `--run`.
6. **Convertir + subir** lo que el RPA dejó en la carpeta:
   ```
   python s1_agent.py --csv --once
   ```
7. **Agendar cada hora** (Task Scheduler → *Import Task…*):
   - `WatermelonS1Agent.xml` corre `run_agent.bat`, que hace primero el RPA y
     luego el `--csv`. Corre aunque nadie tenga sesión abierta.

---

## Comandos

| Comando | Qué hace |
|---|---|
| `python s1_agent.py --selftest` | Valida el formato CSV, sin red |
| `python s1_agent.py --demo --once` | Sube 1 ronda sintética (prueba el canal) |
| `python s1_rpa_export.py --inspect` | Vuelca el árbol UIA de System1 (afinar) |
| `python s1_rpa_export.py --run [--dry]` | Exporta las ondas a CSV (reemplaza persona) |
| `python s1_agent.py --csv --once` | Convierte la carpeta CSV y sube |
| `python s1_agent.py --discover` | (opcional) explora SQL Server = solo config |

## Estado e idempotencia
- El nombre remoto usa el **Timestamp** de la captura → subir de nuevo no
  duplica (`x-upsert`). `data/agent_state.db` guarda estado; logs en `logs/`.

## Seguridad
- Windows Auth para SQL (sin password). `service_key` de Supabase server-side
  (env o `config.toml`, que está en `.gitignore`).

## Formato de intercambio
El agente convierte el CSV de System1 al formato v1 de `core/dynamic_raw.py`
(un archivo = un cojinete con canales X, Y, KPH). La web reconstruye desde ahí.
