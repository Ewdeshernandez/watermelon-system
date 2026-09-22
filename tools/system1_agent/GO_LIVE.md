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

## 5. Descubrir el esquema de System1  ← EL PASO QUE FALTA
Backend YA verificado: **SQL Server**, base **`BNC_Databases`**, **Windows Auth
sin password** (probado: `sqlcmd -S localhost -E` listó las 7 bases).
```
python s1_agent.py --discover
```
Copiar la salida (tablas de waveform + columnas). Con eso escribir
`[system1.query].sql` (T-SQL) con los nombres reales. Debe devolver:
```
point, channel, captured_at, unit, rpm, fs_hz, samples_per_rev, sample_index, value
ORDER BY captured_at, channel, sample_index
```
Placeholders posicionales `?` en orden: (since, point).

Alternativa por clics (sin teclear): **Azure Data Studio** (instalado) →
conectar a `localhost` con *Windows Authentication* → expandir `BNC_Databases`
→ Tables → clic derecho en la tabla de onda → *Select Top 1000*.

Ejemplo (ajustar a lo que muestre --discover):
```sql
SELECT p.Name AS point, c.Name AS channel, w.Timestamp AS captured_at,
       c.Unit AS unit, w.Rpm AS rpm, w.SampleRate AS fs_hz,
       w.SamplesPerRev AS samples_per_rev, s.Idx AS sample_index, s.Value AS value
FROM   Waveform w
JOIN   Channel c ON c.Id = w.ChannelId
JOIN   Point   p ON p.Id = c.PointId
JOIN   Sample  s ON s.WaveformId = w.Id
WHERE  w.Timestamp > ? AND p.Name = ?
ORDER  BY w.Timestamp, c.Name, s.Idx;
```
> OJO: System1 suele guardar la onda como **BLOB binario** (VARBINARY) por
> waveform, no fila-por-muestra. Si es así, la query trae el blob + metadatos y
> hay que decodificar el formato Bently (tarea de mapeo aparte).

## 6. (Opcional) login SQL Server read-only
Producción usa Windows Auth (el agente corre como el usuario del server, que ya
lee). Para endurecer, `sql/create_readonly_role.sql` crea un login `watermelon_ro`
con solo SELECT sobre `BNC_Databases`; entonces en config `trusted = false` +
`user`/`password`.

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
