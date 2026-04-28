# Supabase Setup — Persistencia real del Vault (Ciclo 9)

Esta guía configura Supabase como backend de persistencia para que las instancias, parámetros y documentos del Vault sobrevivan cualquier redeploy o reinicio del container de Streamlit Cloud.

Tiempo estimado: **5–10 minutos**.

---

## 1. Crear cuenta y proyecto Supabase

1. Andá a https://supabase.com y hacé "Start your project" (cuenta gratis con GitHub o email).
2. En el dashboard, "New Project":
   - **Name**: `watermelon-prod` (o el que prefieras)
   - **Database Password**: generá una fuerte y guardala
   - **Region**: la más cercana a tus usuarios (US East, South America - São Paulo si Colombia/CO)
   - **Pricing Plan**: Free (alcanza para varios años de operación: 500 MB DB + 1 GB Storage)
3. Dale "Create new project" — tarda ~1 minuto en aprovisionarse.

---

## 2. Crear el schema de la tabla

1. En tu proyecto, andá al menú lateral → **SQL Editor** → "New query".
2. Abrí el archivo `data/supabase_schema.sql` de este repo, copiá TODO el contenido.
3. Pegalo en el editor de Supabase y dale "Run" (o Cmd/Ctrl+Enter).
4. Vas a ver "Success. No rows returned" — todo OK.

Esto crea la tabla `instances` con índices apropiados y un trigger de `updated_at`.

---

## 3. Crear el bucket de Storage

1. Menú lateral → **Storage** → "New bucket".
2. **Name**: `instance-documents`
3. **Public bucket**: **DESACTIVADO** (debe quedar privado — accedemos vía service key)
4. **Allowed MIME types**: dejá vacío (acepta todo)
5. **File size limit**: 50 MB (suficiente para PDFs de manuales)
6. "Save"

---

## 4. Copiar las credenciales

1. Menú lateral → **Project Settings** (engranaje abajo) → **API**.
2. Copiá dos valores:
   - **Project URL**: tipo `https://abcdefghijklmnop.supabase.co`
   - **service_role key** (NO el anon key): tipo `eyJhbGciOi...` (largo, empieza con `eyJ`)

⚠️ El `service_role` key tiene acceso total a la base de datos — NUNCA lo expongas en frontend ni lo commitees al repo. Solo va en Streamlit Cloud secrets (encriptado).

---

## 5. Configurar el secret en Streamlit Cloud

### Para producción (Streamlit Cloud)

1. Andá a https://share.streamlit.io → tu app Watermelon → **Settings** → **Secrets**.
2. En el editor, pegá al final (sin tocar lo demás):

```toml
[supabase]
url         = "https://abcdefghijklmnop.supabase.co"
service_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
bucket      = "instance-documents"
```

3. "Save". La app se redespliega sola en ~30s.

### Para desarrollo local

Editá `.streamlit/secrets.toml` (NO `.streamlit/secrets.toml.example`) y agregá la misma sección. Ese archivo está en `.gitignore` y nunca se sube al repo.

---

## 6. Validación

Después de configurar y redesplegar:

1. Abrí la app y andá a **Asset Documents**.
2. En la sidebar arriba, debajo de "Activo monitoreado", debería aparecer:
   - ☁️ **Persistencia Supabase activa — los datos sobreviven cualquier redeploy.**
3. Si en cambio ves:
   - 💾 **Storage local — los datos se pierden en redeploy...**
   → algo del secret falló. Revisá las logs en Streamlit Cloud → Manage app → Logs.

4. Para test rápido: creá una instancia "test_supabase" desde el formulario, guardala. En Supabase → Table Editor → instances debería aparecer la nueva fila.

5. Subí un PDF como documento de esa instancia. En Supabase → Storage → instance-documents → carpeta `test_supabase/` debería aparecer el archivo.

---

## Consumo del free tier

- **Database**: 500 MB. Una instancia con muchos parámetros pesa ~5 KB. Podés tener decenas de miles de instancias antes de llenar.
- **Storage**: 1 GB. Un manual OEM típico pesa 2-10 MB. Podés guardar entre 100 y 500 PDFs.
- **API requests**: 50.000/mes para auth, ilimitadas para tabla y storage en free tier.
- **Costo cuando crezcas**: USD 25/mes el plan Pro (8 GB DB + 100 GB Storage + 250.000 API auth/mes).

---

## Troubleshooting

**"Supabase configurado pero no accesible"** en sidebar:
- Revisá que `url` y `service_key` estén bien copiados (sin espacios extra, comillas, etc.).
- Verificá que el proyecto Supabase esté activo (no pausado por inactividad — en free tier se pausa después de 7 días sin uso, basta con abrir el dashboard para reactivar).

**"relation 'instances' does not exist"**:
- No corriste el SQL del paso 2. Volvé al SQL Editor y corré `data/supabase_schema.sql`.

**"Bucket 'instance-documents' not found"**:
- Faltó el paso 3. Creá el bucket con ese nombre exacto.

**Migración de data local existente**:
- Si tenías instancias creadas localmente (en `data/instances/`) y querés migrarlas a Supabase: corré el script de migración (próximo entregable). Por ahora, recreá las pocas instancias que tengas desde la UI — los datos del Brush vienen del seed automáticamente.
