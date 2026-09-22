# GO-LIVE — System1 Agent (checklist final)

Todo lo del lado nube+web ya está construido y probado. Esto es lo que falta,
que corre EN el server Parex (necesita acceso a la DB de System1).

## 1. Copiar el agente al server
Copiar `tools/system1_agent/` completo a `C:\Watermelon\system1_agent\`.

## 2. Python + deps
```
cd C:\Watermelon\system1_agent
python -m pip install -r requirements.txt
```

## 3. Credenciales Supabase (destino)
`config.example.toml` → `config.toml`, y setear `[supabase] service_key`
(Supabase → Settings → API → `service_role`). O variables de entorno del sistema
`SUPABASE_URL` / `SUPABASE_SERVICE_KEY`.

## 4. Probar el canal a la nube SIN tocar System1
```
python s1_agent.py --demo --once
```
Abrir Watermelon Live · Live Monitoring · SGT300B → "Dynamic analysis". Si se ven
onda/espectro/órbita → el canal quedó probado. (Ya lo dejamos verde desde el Mac.)

## 5. Descubrir la DB de System1  ← EL PASO QUE FALTA
```
python s1_agent.py --discover
```
Copiar la salida (lista tablas de waveform y columnas). Con eso:
- Llenar `[system1] dbname / user / password / points`.
- Escribir `[system1.query].sql` con los nombres reales.

Query esperada (debe devolver estas columnas, en este orden):
```
point, channel, captured_at, unit, rpm, fs_hz, samples_per_rev, sample_index, value
ORDER BY captured_at, channel, sample_index
```
Parámetros: `%(since)s` (último captured_at) y `%(point)s`.

Ejemplo (ajustar tabla/columnas a lo que muestre --discover):
```sql
SELECT p.name          AS point,
       c.name          AS channel,
       w.timestamp     AS captured_at,
       c.unit          AS unit,
       w.rpm           AS rpm,
       w.sample_rate   AS fs_hz,
       w.samples_per_rev AS samples_per_rev,
       s.idx           AS sample_index,
       s.value         AS value
FROM   waveform w
JOIN   channel  c ON c.id = w.channel_id
JOIN   point    p ON p.id = c.point_id
JOIN   sample   s ON s.waveform_id = w.id
WHERE  p.name = %(point)s AND w.timestamp > %(since)s
ORDER  BY w.timestamp, c.name, s.idx;
```

## 6. (Recomendado) usuario read-only
`sql/create_readonly_role.sql` — crea `watermelon_ro` con solo SELECT. Usarlo en
`[system1] user/password` en vez de `postgres`.

## 7. Probar producción
```
python s1_agent.py --once
```

## 8. Agendar cada hora
Task Scheduler → Import Task… → `WatermelonS1Agent.xml`
(editar la ruta si la carpeta no es `C:\Watermelon\system1_agent\`).

---
Cuando tengas la salida de `--discover`, pásamela y te dejo el `[system1.query].sql`
final escrito y validado.
