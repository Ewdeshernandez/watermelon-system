# Watermelon Modal — Licenciamiento (Fase 2, servidor)

Despliegue del emisor de licencias. El servidor firma tokens con una llave PRIVADA que
**nunca** sale de aquí; el `.exe` sólo trae la PÚBLICA (verifica, no puede forjar).

## 1) Genera el par de llaves (hazlo TÚ — Claude no debe ver la privada)

```bash
# Privada (guárdala en el servidor, NUNCA en el .exe ni en git)
openssl genpkey -algorithm ed25519 -out wm_license_private.pem
# Pública en base64url (esto va embebido en el .exe)
python3 - <<'PY'
from cryptography.hazmat.primitives import serialization as s
import base64
k=s.load_pem_private_key(open("wm_license_private.pem","rb").read(),None)
pub=k.public_key().public_bytes(s.Encoding.Raw,s.PublicFormat.Raw)
priv=k.private_bytes(s.Encoding.Raw,s.PrivateFormat.Raw,s.NoEncryption())
print("PUBLIC  (al .exe, env WM_LICENSE_PUBKEY):", base64.urlsafe_b64encode(pub).decode().rstrip("="))
print("PRIVATE (secreto del server):", base64.urlsafe_b64encode(priv).decode().rstrip("="))
PY
```

- La **PÚBLICA** → secret del build `WM_LICENSE_PUBKEY` (GitHub Actions) → queda en `licensing.LICENSE_PUBKEY_B64`.
- La **PRIVADA** → secret de Supabase: `supabase secrets set WM_LICENSE_PRIVKEY=<base64>`.

## 2) Base de datos + RLS
Aplica `schema.sql` (SQL editor de Supabase o `supabase db push`). Crea `licenses` y
`activations` con RLS: cada usuario ve SOLO lo suyo; nadie escribe tokens desde el cliente.

## 3) Edge Function `activate`
`supabase functions deploy activate`. El cliente llama con su JWT (login Supabase) + su
huella de máquina; la función valida cupo, registra la activación y devuelve el **token firmado**.

## 4) Migrar el `.exe` a anon key + RLS (CRÍTICO)
Hoy el build embebe `SUPABASE_SERVICE_KEY` (god-mode). Cambiar el secret del build a la
**anon key** y confiar en RLS. La subida de corridas (`modal_runs`, bucket `modal-raw`) debe
tener policies que permitan al usuario autenticado insertar lo suyo. Ver `schema.sql`.
