# Instalar el robot en una VM nueva (otro cliente) — checklist

Tiempo estimado: ~20 min. Lo que hay que **instalar/configurar** en el server de
System1 del cliente para que la onda cruda llegue sola a Watermelon cada hora.

## 0. Requisitos en el server del cliente
- Windows con **System1 instalado** y la máquina del cliente **abierta** (con la
  forma de onda visible; así se puede exportar).
- **Internet** (que alcance `*.supabase.co` — el robot sube ahí).
- Permiso de administrador (para instalar Python y agendar la tarea).

## 1. Instalar Python 3.11+
Descargar de python.org → instalar → **marcar "Add Python to PATH"**.
Verificar: `python --version`

## 2. Copiar el robot
Copiar la carpeta `system1_agent\` a **`C:\Watermelon\system1_agent\`**.

## 3. Dependencias
```
cd C:\Watermelon\system1_agent
python -m pip install -r requirements.txt
```
(instala: numpy, supabase, pywinauto, pyodbc)

## 4. Config (`config.toml`)
Copiar `config.example.toml` → `config.toml` y llenar:
- `asset` = **instance_id del cliente en Watermelon** (mismo id que usa Live).
- `[supabase] service_key` = service_role del proyecto Supabase (o variables de
  entorno `SUPABASE_URL` / `SUPABASE_SERVICE_KEY`).
- `[system1_csv] folder` = carpeta donde se exportan los CSV (ej.
  `C:\Users\<user>\Desktop\CSV`).
- `[rpa] out_folder` = la misma carpeta.
- `[rpa] points` = un renglón por onda: `tag` (texto EXACTO del nodo en el árbol
  de System1) y `name` (nombre de archivo, ej. `1xd`). Emparejar X/Y por cojinete
  (1xd+1yd, 2xd+2yd, …) para tener órbita.

## 5. Crear el activo en Watermelon
El `asset`/instance_id debe existir en Watermelon Live (Machinery Library) para
que la vista *Dynamic analysis* encuentre las capturas.

## 6. Afinar el RPA (una vez, con System1 abierto)
```
python s1_rpa_export.py --inspect
```
Ajustar en `config.toml` → `[rpa]`:
- `plot_xy` = centro de la gráfica dentro de la ventana de System1 (offset x,y).
- `tag` de cada punto según el árbol real.
Probar: `python s1_rpa_export.py --run --dry`  → luego `python s1_rpa_export.py --run`

## 7. Probar el ciclo completo
```
python s1_rpa_export.py --run
python s1_agent.py --csv --once
```
Abrir Watermelon Live → activo del cliente → *Dynamic analysis* → onda/espectro/órbita.

## 8. Agendar cada hora
Task Scheduler → *Import Task…* → `WatermelonS1Agent.xml` (editar la ruta si la
carpeta no es `C:\Watermelon\system1_agent\`). Corre `run_agent.bat` (RPA → subir).

## 9. Teclado (si aplica)
El RPA escribe por UIA (no por teclado), así que **no** depende del layout. Solo
si algo se teclea manual y el server está en español, poner **English (US)** en
el indicador de idioma del taskbar.

## Notas
- Todo es `SELECT`/lectura + export sancionado de System1. No modifica la base.
- El backend de System1 es SQL Server (`BNC_Databases`), legible con Windows Auth
  sin password — pero la onda NO está ahí; sale por el Export to CSV (esto).
