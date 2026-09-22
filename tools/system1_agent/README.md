# Watermelon System1 Agent

Robot **headless** que corre en el servidor de Bently System1 (VM Parex) y sube
la **onda cruda dinámica con keyphasor** al Watermelon Cloud **cada hora**.
Watermelon Live · *Análisis Avanzado* la reconstruye en forma de onda, espectro
(FFT, órdenes 1X/2X/3X) y órbita.

```
System1 PostgreSQL 14  (VM Parex)
   └─ s1_agent.py  (Task Scheduler, cada hora, incremental por timestamp)
        └─ CSV por punto/instante (X, Y, KPH)  →  Supabase Storage: dynamic_raw
             └─ Watermelon Live · Análisis Avanzado  →  onda / espectro / órbita
```

## Por qué esto funciona (y no es un black box)
System1 Premium guarda en **PostgreSQL 14** (base abierta) y trae **Database
Replication** + **Data Export** nativos. La data **no está encriptada**; se lee
con un usuario read-only. Este agente solo hace `SELECT`.

---

## Instalación en el server Parex (una vez)

1. **Copiar** esta carpeta a `C:\Watermelon\system1_agent\`.

2. **Python 3.9+** (si no está): instalar desde python.org y marcar *Add to PATH*.

3. **Dependencias**:
   ```
   cd C:\Watermelon\system1_agent
   python -m pip install -r requirements.txt
   ```

4. **Config**: copiar `config.example.toml` → `config.toml` y llenar:
   - `[supabase] service_key` (Supabase → Settings → API → `service_role`).
     *Mejor aún:* setear variables de entorno del sistema
     `SUPABASE_URL` y `SUPABASE_SERVICE_KEY` (así la key no queda en disco).

5. **Probar el pipeline SIN tocar System1** (sube capturas sintéticas):
   ```
   python s1_agent.py --demo --once
   ```
   Abrir Watermelon Live · Análisis Avanzado → deben verse onda/espectro/órbita
   del activo `SGT300B`. Si aparecen, el canal a la nube quedó probado.

6. **Descubrir la base de System1** (read-only):
   ```
   python s1_agent.py --discover
   ```
   Lista la DB, tablas candidatas de waveform y sus columnas. Con eso:
   - Llenar `[system1] dbname/user/password/points`.
   - Escribir `[system1.query].sql` con los nombres reales de tabla/columna.
     La query debe devolver:
     `point, channel, captured_at, unit, rpm, fs_hz, samples_per_rev, sample_index, value`

7. **Probar producción**:
   ```
   python s1_agent.py --once
   ```

8. **Agendar cada hora**: Task Scheduler → *Import Task…* → `WatermelonS1Agent.xml`
   (editar la ruta si la carpeta no es `C:\Watermelon\system1_agent\`).
   Corre cada hora, aunque nadie tenga sesión abierta.

---

## Comandos

| Comando | Qué hace |
|---|---|
| `python s1_agent.py --selftest` | Valida el formato CSV, sin red |
| `python s1_agent.py --demo --once` | Sube 1 ronda sintética (prueba el canal) |
| `python s1_agent.py --demo --once --dry` | Genera y valida, **sin** subir |
| `python s1_agent.py --discover` | Explora la DB de System1 (read-only) |
| `python s1_agent.py --once` | 1 ronda real (producción) |

## Estado e idempotencia
- `data/agent_state.db` (SQLite) guarda el último `captured_at` por punto →
  solo sube lo nuevo (incremental, barato para correr cada hora).
- Storage usa `x-upsert`: reintentar **no** duplica.
- Logs en `logs/s1_agent.log`.

## Seguridad
- Usar un **usuario PostgreSQL read-only** para el agente.
- El `service_key` de Supabase es server-side: mantenerlo en variable de entorno
  o en `config.toml` (que está en `.gitignore`), nunca en git.

## Formato de intercambio
Idéntico a `core/dynamic_raw.py` (v1). Un archivo = un punto en un instante,
canales muestreados en simultáneo. Header autodescriptivo `#clave=valor`, luego
`t_s,X,Y,KPH`.
