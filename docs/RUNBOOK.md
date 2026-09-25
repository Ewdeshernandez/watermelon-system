# RUNBOOK — Watermelon System (documentación dura / recuperación ante desastre)

> **Para qué sirve este documento:** si pierdes el PC, cambias de computadora, o alguien nuevo
> tiene que continuar el proyecto — aquí está TODO: dónde vive cada cosa, cómo se autentica, cómo
> se compila y despliega, y cómo recuperar el control.
>
> **Este archivo vive en GitHub** (`docs/RUNBOOK.md`) → sobrevive a la pérdida del PC.
> **NO contiene los valores de los secretos** (llaves privadas, contraseñas) — solo dónde están y
> cómo recuperarlos. Última actualización: 2026-09-14.

---

## 0) La idea en una frase

Nada crítico vive en tu PC. **El código está en GitHub** y **los datos + backend están en Supabase**
(ambos en la nube). Tu PC es **desechable**: si se pierde, clonas el repo en otro y sigues. Lo único
que hay que proteger de verdad son **las CUENTAS** (ver §7).

---

## 1) Qué es el proyecto

**Watermelon System** — plataforma de análisis de vibraciones y modal. Tres apps:
- **Rotordynamics** (campo, nativo Windows) — monitoreo/rotodinámica.
- **Watermelon Modal** (campo nativo + web) — análisis modal EMA/OMA. Compite contra ARTeMIS.
- **Watermelon Planta** (app con login OTP).

Arquitectura clave: **el app nativo de campo captura y sube DATA CRUDA a la nube; la WEB hace TODO
el análisis** con esa data cruda y genera el reporte.

---

## 2) Dónde vive TODO (el mapa)

| Componente | Dónde | Identificador |
|---|---|---|
| **Código fuente** | GitHub (privado) | `github.com/Ewdeshernandez/watermelon-system` |
| **Base de datos + Storage + Auth + Edge Functions** | Supabase | proyecto `watermelon-prod`, ref `yxeqwkhybueelmkrdkgq` |
| **Web (análisis modal, reportes)** | Streamlit Community Cloud | deploy automático al hacer push a `main` |
| **Builds del `.exe` (Windows)** | GitHub Actions | se disparan con tags `modal-v*` (modal) / field builds |
| **Releases (instaladores .exe)** | GitHub Releases | `github.com/Ewdeshernandez/watermelon-system/releases` |
| **Dominio** | watermelonsystem.app | (registrador — verificar en tu cuenta de dominios) |

El repo local en el Mac es solo una **copia de trabajo**. La fuente de verdad es GitHub.

---

## 3) Estructura del código (lo esencial)

```
native/watermelon_modal.py     # app modal de campo (un archivo, ~4000 líneas). __version__ arriba.
native/watermelon_field.py     # app de rotordinámica de campo
native/installer/*.iss         # instaladores Inno Setup
core/modal/                    # motor modal: licensing.py, modal_cloud.py, ssi.py, oma_engine.py, ...
core/remote_monitoring/        # recorder.py (_sb_client), _cloud_config.py (generado en CI)
pages/                         # web Streamlit (pages/18 = Watermelon Modal web)
server/licensing/              # backend de licencias: schema.sql, rls_modal_anon.sql, functions/
.github/workflows/             # build-modal-windows.yml, build-planta.yml, ...
docs/RUNBOOK.md                # este archivo
```

---

## 4) Base de datos (Supabase → Table Editor)

| Tabla | Para qué |
|---|---|
| `licenses` | **Licencias vendidas.** Una fila por licencia (`key`, `seats`, `features`, `expires_at`, `status`, `account`). |
| `activations` | **Máquinas activadas.** Una fila por PC (`machine_fp`, `hostname`, `revoked`, `last_seen`). Binding + kill-switch. |
| `modal_setups` | Configuración/geometría modal compartida campo↔web. |
| `modal_runs` | Corridas OMA subidas desde el campo (metadata + `raw_ref`). La web las analiza. |
| `rm_setups`, `live_readings`, `instances`, `transient_recordings` | Rotordinámica / monitoreo. |
| `planta_otp_challenges` | Retos OTP de la app Planta. |

**Storage (buckets):** `modal-raw` (data cruda gzip de las corridas, privado), `transients`,
`modal-captures`, etc.

---

## 5) Cómo se autentica cada cosa

### a) Licencias del `.exe` (anti-robo) — firma Ed25519
- El servidor **firma** un token con una **llave PRIVADA** (Ed25519). El `.exe` trae la **llave
  PÚBLICA** embebida y solo **verifica** (no puede falsificar).
- **Pública actual** (segura, va en el código): `ozjVKHml8OE4E-1h07evyw5IOcGnO-IbdB7lV_OUaeM`
  → en `core/modal/licensing.py` (`LICENSE_PUBKEY_B64`) y en la Edge Function `activate`.
- **Privada**: es el secret de Supabase `WM_LICENSE_PRIVKEY` (NO se puede leer de vuelta) + una copia
  local en el Mac `wm_license_private_NEW.pem` (gitignored). **Si se pierde la privada NO es
  catástrofe → se ROTA** (ver §9).
- Flujo: la app llama a la Edge Function `activate` con `{license_key, machine_fp}` → valida clave,
  cupo (`seats`) y binding → devuelve token firmado. El `.exe` re-chequea en CADA arranque
  (kill-switch inmediato). Funciona offline con token en caché hasta 30 días.

### b) Subida de data a la nube (uploads del campo) — anon key + RLS
- El `.exe` embebe la **anon/publishable key** (`sb_publishable_...`, pública por diseño), NO el
  service key. Con **RLS** (`server/licensing/rls_modal_anon.sql`) solo puede **subir** a
  `modal_setups`/`modal_runs`/`modal-raw`. Un binario robado NO tiene god-mode.
- La **web** lee/analiza con el **service key** del lado servidor (bypass RLS).

### c) Usuarios (web / Planta) — OTP
- Supabase Auth con OTP (código al correo). No hay contraseñas de usuario.

---

## 6) Llaves y secretos — DÓNDE están (no los valores)

| Secreto | Dónde vive | Se puede leer | Si se pierde |
|---|---|---|---|
| **Llave privada de licencias** (`WM_LICENSE_PRIVKEY`) | Supabase → Edge Functions → Secrets + `.pem` local | No (Supabase lo enmascara) | **Rotar** (§9) |
| **Service key de Supabase** | Supabase → Settings → API → Secret keys | Sí (dashboard) | Regenerar en Supabase |
| **Anon/publishable key** | Supabase → Settings → API → Publishable key | Sí (pública) | Copiar de nuevo |
| **Secrets de GitHub Actions** (`SUPABASE_URL`, `SUPABASE_ANON_KEY`, `SUPABASE_SERVICE_KEY`) | GitHub → repo → Settings → Secrets → Actions | No (write-only) | Volver a pegarlos desde Supabase |

> **Regla:** los valores reales NO van en este documento ni en git. Se recuperan desde el dashboard
> de Supabase (que sobrevive a la pérdida del PC).

---

## 7) Las CUENTAS (esto es lo único de verdad crítico)

Si proteges estas cuentas, **nunca pierdes el proyecto** aunque pierdas todos los PCs:

1. **GitHub** (`Ewdeshernandez`) — tiene el código + los Actions secrets.
2. **Supabase** — tiene la base de datos, storage, Edge Functions y sus secrets.
3. **Correo** `ehernandez@sigasas.com` — es la recuperación de las dos de arriba y el OTP.
4. **Streamlit Community Cloud** — despliega la web (se reconecta desde GitHub).
5. **Registrador del dominio** watermelonsystem.app.

**ACCIÓN OBLIGATORIA (hazla hoy):**
- Activa **2FA (verificación en dos pasos)** en GitHub, Supabase y el correo.
- Guarda los **recovery codes** de cada uno en un **gestor de contraseñas** (1Password / Bitwarden),
  **NO** en el PC ni en un .txt suelto. Ese gestor es tu caja fuerte.

---

## 8) Cómo compilar y desplegar

### Web (Streamlit)
- `git push origin main` → Streamlit Community Cloud redepliega solo. Nada más.

### App nativo modal (.exe Windows)
1. Sube `__version__` en `native/watermelon_modal.py`.
2. `git commit` + `git push origin main`.
3. `git tag modal-vX.Y.Z && git push origin modal-vX.Y.Z` → GitHub Actions compila y publica el
   instalador en Releases (~10-15 min).
- El workflow embebe las credenciales de nube desde los GitHub Secrets.

---

## 8b) Live Monitoring — reportes AUTOMÁTICOS (trabaja solo, sin especialista)

El módulo Live Monitoring manda avisos/reportes solo. Los dispara **GitHub Actions**
(no Render), workflow `.github/workflows/live_auto_reports.yml`:

| Cron (UTC) | Qué | Script |
|---|---|---|
| `*/15 * * * *` | **Alarma/Danger** — avisa al entrar/escalar (anti-spam 1×/episodio) + **heartbeat OFFLINE** (activo sin datos > `WM_OFFLINE_MINUTES`, def 60) | `scripts/send_alarm_reports.py` |
| `0 * * * *` | **Diario/programado** — envía a los activos cuya hora/día programados coinciden | `scripts/send_scheduled_reports.py` |
| `0 12 * * 1` (+ sáb noche) | **Semanal** — briefing por activo con FLUJO humano: genera BORRADOR → avisa por email al revisor (`WM_BRIEFING_REVIEW_EMAIL`, def ehernandez@sigasas.com) → el especialista revisa/firma/aprueba en la app ("Briefing por activo") → al aprobar se envía solo al cliente. Firmas: Preparado por Laura C. Garzón / Revisado por Ewdes A. Hernández. **NO usar `--auto-send`** (ese modo se salta la revisión y firma "Watermelon System (automático)"). | `scripts/send_weekly_briefing.py --period Semanal` |

**Secret requerido (una vez):** `STREAMLIT_SECRETS_TOML` en GitHub → Settings → Secrets →
Actions = contenido COMPLETO de `.streamlit/secrets.toml` (`[supabase]`, `[email]`,
`[whatsapp]`). El workflow lo reescribe en el runner. **Sin este secret los crons
corren pero NO envían nada (skip elegante).**

**Para que un activo reciba avisos:** en Machinery Library debe tener destinatario
(email/WhatsApp) y `alarm_send_enabled` / `report_send_enabled` en ON. Activo sin
destinatario = salteado (no se monitorea).

**Probar sin enviar:** Actions → "Live Monitoring — reportes automáticos" → Run
workflow → job `all`, dry-run ✔.

---

## 9) Cómo ROTAR la llave de licencias (si se filtra/pierde la privada)

1. Genera par nuevo (en Mac):
   ```bash
   openssl genpkey -algorithm ed25519 -out wm_license_private_NEW.pem
   python3 - <<'PY'
   from cryptography.hazmat.primitives import serialization as s; import base64
   k=s.load_pem_private_key(open("wm_license_private_NEW.pem","rb").read(),None)
   pub=k.public_key().public_bytes(s.Encoding.Raw,s.PublicFormat.Raw)
   priv=k.private_bytes(s.Encoding.Raw,s.PrivateFormat.Raw,s.NoEncryption())
   print("PUBLIC :", base64.urlsafe_b64encode(pub).decode().rstrip("="))
   print("PRIVATE:", base64.urlsafe_b64encode(priv).decode().rstrip("="))
   PY
   ```
2. Pon la **PÚBLICA** nueva en `core/modal/licensing.py` (`LICENSE_PUBKEY_B64`) y en la Edge Function
   `activate` (constante `LICENSE_PUBKEY`).
3. Pon la **PRIVADA** nueva en el secret de Supabase `WM_LICENSE_PRIVKEY` y re-despliega `activate`.
4. Sube versión + build nuevo. Los PCs se re-activan solos al actualizar.

---

## 10) Cómo CREAR y GESTIONAR licencias (consola web)

**Consola:** Watermelon System (web) → **Administración → Licencias**. Solo admin
`@sigasas.com`. Fuente de verdad: tablas Supabase `licenses` + `activations`. No hace
falta tocar SQL a mano.

Desde la consola:
- **Crear licencia** → nombre de cliente, cuenta (email), paquete, seats, vigencia →
  genera la clave `WM-XXXX-XXXX-XXXX` y la muestra para entregar al cliente.
- **Ver dónde vive** → por cada máquina: PC (hostname), IP, ubicación, módulo, VM y
  última conexión.
- **Renovar** → extiende la vigencia (y reactiva si estaba suspendida).
- **Suspender por falta de pago** → `status=suspended`; al próximo arranque online el
  cliente ve **"Licencia no renovada por falta de pago"** (edge `activate` → `payment_due`,
  el gate lo muestra de entrada).
- **Reactivar / Revocar** licencia completa; **Revocar / Reactivar / Liberar cupo** por
  máquina individual.

Notas:
- **seats = 1** → esa clave solo activa en **un** PC (compartirla → `no_seats`).
- **Se aplica** en el próximo arranque online del cliente (gate re-chequea en cada inicio).

**Fallback SQL** (si la consola no está disponible), Supabase → SQL Editor:

```sql
insert into public.licenses (key, account, customer, seats, features, plan, expires_at, status)
values ('WM-XXXX-XXXX-XXXX', 'cliente@empresa.com', 'Empresa SAS', 1,
        '{oma,ema,report}', 'Modal (OMA/EMA)', '2027-12-31T23:59:59Z', 'active');
```

- Suspender por pago: `update licenses set status='suspended',
  suspended_reason='Licencia no renovada por falta de pago' where key='WM-…';`
- Revocar todo: `licenses.status='revoked'`. Revocar un PC: `activations.revoked=true`.

**Requisito una sola vez:** aplicar `server/licensing/2026_09_licenses_console.sql`
(columnas IP/geo/PC/estado comercial) y re-desplegar la edge `activate`
(`supabase functions deploy activate`).

---

## 11) RECUPERACIÓN ANTE DESASTRE — "perdí/me robaron el PC" o "cambié de computadora"

**Respuesta corta: no pierdes nada.** Todo lo importante está en la nube. Pasos en el PC nuevo:

1. **Instala** Git + Python 3.12.
2. **Inicia sesión** en GitHub y Supabase (con 2FA / recovery codes del gestor — §7).
3. **Clona el repo:**
   ```bash
   git clone https://github.com/Ewdeshernandez/watermelon-system.git
   ```
4. Ya tienes **todo el código** de vuelta. La base de datos y los usuarios siguen intactos en
   Supabase. La web sigue arriba. Los clientes activados siguen funcionando.
5. Si necesitas re-compilar: solo haces push/tag (los secrets viven en GitHub, no en el PC).
6. **Único archivo local que no está en git:** `wm_license_private_NEW.pem`. Si lo perdiste y algún
   día necesitas la privada legible → **rota la llave** (§9). El sistema sigue firmando mientras
   tanto (el secret de Supabase no se pierde).

**Lo que SÍ te dejaría varado:** perder acceso a las **cuentas** (GitHub/Supabase/correo) sin 2FA ni
recovery codes. Por eso §7 es obligatorio.

---

## 12) Checklist de "estoy protegido"

- [ ] 2FA activo en GitHub, Supabase y correo `ehernandez@sigasas.com`.
- [ ] Recovery codes de los 3 guardados en un gestor de contraseñas (fuera del PC).
- [ ] El repo está pusheado a GitHub (este RUNBOOK incluido).
- [ ] `wm_license_private_NEW.pem` respaldado en el gestor/caja fuerte (o asumir que se rota si se pierde).
- [ ] Sé cómo crear una licencia (§10) y cómo revocar (§10).
