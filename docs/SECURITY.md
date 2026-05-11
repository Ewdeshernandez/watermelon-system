# Watermelon System — Security Playbook

Documento operativo de seguridad. Define el modelo de amenazas, las prácticas obligatorias, y los procedimientos de respuesta. Repo público, datos confidenciales en Supabase.

---

## 1. Modelo de seguridad

| Componente | Acceso | Riesgo si se compromete |
|---|---|---|
| Repo GitHub `watermelon-system` | Público | Cualquiera lee el código. No hay riesgo si **no hay secretos commiteados**. |
| Supabase Postgres | service_role key | Acceso TOTAL a todas las tablas. Si la key leak, el atacante lee/borra TODO. |
| Supabase Storage bucket `instance-documents` | service_role key | Mismo nivel: acceso total a archivos de clientes. |
| Streamlit Cloud app | Public URL + login app | Sin la auth de la app, no se ve nada. Auth via Supabase + session_state. |
| Render WhatsApp bot | Webhook público | Necesita verify_token + Meta signature validation. |

**Conclusión:** el activo crítico a proteger es la `service_role` key de Supabase. Toda la seguridad arranca por ahí.

---

## 2. Reglas absolutas (NO romper)

### 2.1 — Nada de secretos en el repo

NUNCA commitear archivos con:
- `service_role` keys de Supabase (JWT que empieza con `eyJ...`)
- `anon_key` de Supabase
- Database passwords
- Meta WhatsApp access tokens
- SMTP passwords
- API keys de servicios pagados

Los secretos viven en:
- **Producción Streamlit Cloud** → Settings → Secrets (encriptado)
- **Producción Render (WhatsApp bot)** → Environment Variables (encriptado)
- **Local desarrollo** → `.streamlit/secrets.toml` (gitignored)
- **CI/GitHub Actions** → Repository Secrets (encriptado)

### 2.2 — .gitignore obligatorio

Verificar que estos paths están en `.gitignore`:
- `.streamlit/secrets.toml`
- `.env`
- `.envrc`
- `*.pem`, `*.key`
- `data/*.csv` (datos de clientes)

### 2.3 — Antes de hacer commit

Si tocás algún archivo de configuración, correr:
```bash
git diff --cached | grep -iE "(eyJ[A-Za-z0-9_-]{20,}|sk-|password\s*=\s*['\"]|AIza)"
```

Si hay match: **STOP**, no commitear. Mover el secreto a env var.

### 2.4 — Rotación de keys

Rotar la `service_role` key de Supabase si:
- Apareció en un commit (aunque después se haya borrado — el historial git la conserva)
- Estuvo en un screenshot público (LinkedIn, demos)
- Se compartió en Slack/email/WhatsApp
- Alguien que tenía acceso al equipo se fue

**Cómo rotar:**
1. Supabase Dashboard → Settings → API → "Reset service_role key"
2. Actualizar en Streamlit Cloud Secrets + Render env vars
3. Re-deploy ambos servicios

---

## 3. Defensas activas

### 3.1 — CORS Supabase

Restringir origins de las API requests:

Supabase Dashboard → Settings → API → CORS allowed origins:
```
https://watermelonsys.net
https://wm-home-final-2026.streamlit.app
https://*.streamlit.app
http://localhost:8501
http://localhost:8502
http://localhost:8503
http://localhost:8504
http://localhost:8505
http://localhost:8506
http://localhost:8507
http://localhost:8508
http://localhost:8509
http://localhost:8510
```

**NO usar `*`** (default — permite cualquier origin).

### 3.2 — Row Level Security (RLS) — defensa en profundidad

Aunque el backend usa `service_role` (que bypasea RLS), enablar RLS protege contra:
- Una key `anon` que se exponga en futuro
- Un error en el código que termine usando anon_key en vez de service_key
- Auditorías de seguridad / compliance

Correr el script `data/security_hardening.sql` desde Supabase SQL Editor.

### 3.3 — Streamlit app auth

Confirmar que toda página crítica llama:
```python
require_login()
require_role(allowed_roles=("admin", "specialist"))  # según corresponda
```

Páginas que actualmente exigen login: las que importan `from core.auth import require_login`.

### 3.4 — Validación de webhook WhatsApp (Render bot)

El bot debe:
1. Validar `hub.verify_token` en GET (setup)
2. Validar `X-Hub-Signature-256` en POST (cada mensaje)

Si no valida la signature, **cualquiera puede mandar mensajes falsos** a tu bot.

---

## 4. Backup y recovery (anti-tumbar)

### 4.1 — Supabase backups

Plan **Free**: NO hay backups automáticos. Estrategia:
- Setup GitHub Actions semanal que corre `pg_dump` y guarda en un bucket privado (S3 o similar)
- Mínimo retener 4 backups semanales

Plan **Pro** ($25/mes): backups diarios automáticos retenidos 7 días. **Recomendado** una vez tengas clientes pagando.

### 4.2 — GitHub branch protection

Settings → Branches → `main` rules:
- ✅ Require pull request before merging (no push directo)
- ✅ Require status checks (si configurás CI)
- ✅ Do not allow force pushes

Esto evita que un commit hostil/accidental reescriba la historia de main.

### 4.3 — Streamlit Cloud / Render

Ambos tienen un solo botón "Reboot" en sus dashboards. Si algo se rompe:
1. Streamlit Cloud → Manage app → Reboot
2. Render → Service → Manual Deploy → Clear build cache & deploy

---

## 5. Checklist de respuesta a incidentes

Si sospechás compromiso:

1. **Inmediato:** Reset service_role key en Supabase. Actualizar Streamlit Cloud + Render.
2. **Inmediato:** Cambiar password de cuenta Supabase (no la DB password, la del dashboard).
3. **Inmediato:** Revisar Audit Log de Supabase (Dashboard → Database → Audit) para conexiones extrañas.
4. **30 min:** Notificar a clientes afectados si hubo data leak.
5. **24h:** Postmortem — qué pasó, cómo se previene.

---

## 6. Reportar vulnerabilidades

Si encontrás un bug de seguridad en Watermelon:
- Email: `security@sigasas.com`
- NO abrir issue público en GitHub
- Reportá con detalles + PoC; te respondo en 48h

---

**Última revisión:** v3.31.54 — Mayo 2026
