# Sistema de Licencias Watermelon Planta — Manual Interno SIGA

Solo para uso interno del equipo de SIGA GROUP. **NO** distribuir al cliente.

## TL;DR — Emitir una licencia para un cliente nuevo

```bash
cd watermelon-system
python tools/license_issue.py \
    --customer "Termoeléctrica Norte SAS" \
    --email "ingenieria@termonorte.com" \
    --plan pro \
    --expires 2027-05-21
```

Eso genera `tools/licenses_issued/termoelectrica-norte-sas/`:
- `license.token` → enviar al cliente
- `README_CLIENTE.txt` → enviar al cliente con instrucciones
- `license.json` → **NO ENVIAR**, queda como registro interno

Adjunta los dos primeros archivos por email y listo.

---

## Arquitectura

```
[SIGA, bóveda offline]                    [Cliente, en planta]
      private_key.pem                          public_key.pem
            |                                       ^
            v                                       |
     license_issue.py                       license_manager.py
            |                                       |
            v                                       v
       license.token  ─── email/USB ──>     planta/data/license.token
            |                                       |
            +─── (firma RSA-2048) ──────────────────+
                                                    |
                                              (verifica firma)
                                                    |
                                                    v
                                            Watermelon Planta arranca
```

**Componentes:**

| Archivo | Ubicación | Quién lo tiene | Función |
|---|---|---|---|
| `private_key.pem` | `tools/.keys/` | Solo SIGA (bóveda) | Firmar licencias |
| `public_key.pem` | `tools/.keys/` + hardcoded en `planta/license_manager.py` | SIGA + cliente | Verificar firmas |
| `license.token` | `tools/licenses_issued/<cliente>/` + cliente en `planta/data/` | SIGA + 1 cliente | El JWT firmado |
| `license_issue.py` | `tools/` | Solo SIGA | Generador |
| `license_manager.py` | `planta/` | Cliente (dentro del .exe) | Verificador runtime |

---

## Planes comerciales

Definidos en `tools/license_issue.py` → `PLANS`:

| Plan | Módulos | Max canales | Default duración |
|---|---|---|---|
| `trial` | EMA | 4 | 30 días |
| `basic` | EMA | 8 | 365 días |
| `pro` | EMA + OMA | 16 | 365 días |
| `enterprise` | EMA + OMA + FEA + 3D + Reports | 32 | 365 días |

Si necesitas un plan custom, usa `--modules` y `--max-channels` manualmente.

---

## Operaciones comunes

### Renovar una licencia que se vence

Igual que emitir nueva, con la misma data pero nuevo `--expires`:

```bash
python tools/license_issue.py \
    --customer "Termoeléctrica Norte SAS" \
    --email "ingenieria@termonorte.com" \
    --plan pro \
    --expires 2028-05-21 \
    --notes "Renovación anual #2 — pago confirmado 2027-05-15"
```

El nuevo `license.token` reemplaza al viejo en `<install_dir>/data/`. El cliente solo pega encima.

### Hacer upgrade de plan

```bash
python tools/license_issue.py \
    --customer "Termoeléctrica Norte SAS" \
    --email "ingenieria@termonorte.com" \
    --plan enterprise \
    --notes "Upgrade pro→enterprise — pedido OC-2027-441"
```

### Emitir trial de 30 días

```bash
python tools/license_issue.py \
    --customer "Cliente Demo" \
    --email "demo@cliente.com" \
    --plan trial
```

(usa defaults: 30 días, solo EMA, 4 canales)

### Revisar licencias emitidas

```bash
ls tools/licenses_issued/
```

Cada carpeta contiene el `license.json` con todos los detalles, incluyendo el `license_id` (UUID único) y las notas internas.

---

## Procedimiento de seguridad

### Dónde vive la `private_key.pem`

**Triple backup, todos OFFLINE:**

1. **USB encriptado en caja fuerte** de SIGA — primario
2. **Encrypted disk image** en NAS de SIGA — secundario
3. **Vault 1Password (cuenta corporativa, no personal)** — terciario

**NUNCA:**
- En Git (ya está en `.gitignore` pero verifica)
- En email
- En Slack / WhatsApp / Telegram
- En el laptop del dev sin encriptar
- En un cliente
- En un USB que sale de la oficina sin encriptación

### Qué hacer si `private_key.pem` se filtra

Es el escenario peor caso. Plan de contingencia:

1. **Confirmar la filtración.** ¿Apareció en GitHub público? ¿La envió alguien por email?
2. **Regenerar par de claves:**
   ```bash
   rm tools/.keys/private_key.pem tools/.keys/public_key.pem
   python tools/license_keygen.py
   ```
3. **Actualizar `planta/license_manager.py`:** copiar nueva `public_key.pem` al string `_EMBEDDED_PUBLIC_KEY`.
4. **Bump VERSION + rebuild .exe + nuevo installer.**
5. **Re-emitir TODAS las licencias activas** con la nueva private key:
   ```bash
   for cliente in tools/licenses_issued/*/license.json; do
       # Re-correr license_issue con los mismos datos
       # (escribir script helper)
   done
   ```
6. **Enviar a TODOS los clientes** el nuevo installer + nueva license.token.
7. **Postmortem interno** — ¿cómo se filtró? ¿qué cambia para que no pase de nuevo?

Documentar todo en `docs/incidents/<fecha>_key_leak.md`.

### Rotación preventiva de claves

Recomendable cada 2-3 años incluso sin incidente. Mismo procedimiento que filtración pero planificado.

---

## Cómo funciona la verificación (técnico)

`planta/license_manager.py` hace esto en cada arranque:

1. Lee `<data_dir>/license.token` como string.
2. Verifica la firma RSA-2048 contra `_EMBEDDED_PUBLIC_KEY`.
3. Valida claims standard JWT:
   - `iss == "SIGA GROUP SAS"` (issuer)
   - `aud == "watermelon-planta"` (audience)
   - `exp > now` (no vencido)
   - `iat <= now` (no es del futuro)
4. Parsea claims custom:
   - `customer`, `plan`, `modules`, `max_channels`, `jti`
5. Devuelve `LicenseInfo` con `valid=True` o `valid=False` + razón.

**Sin red. Sin Supabase. Sin nada externo.** Todo es criptografía local — no se puede falsificar sin la private key. Editar el token rompe la firma. Editar `license_manager.py` requiere rebuild del .exe (que tiene su propia integridad).

**Tolerancia 0 al reloj** (`leeway=0`): si el cliente atrasa su reloj, la licencia que aún no debía existir es inválida. Esto evita ataques tipo "atrasé la fecha para evitar el vencimiento".

---

## Smoke test end-to-end

```bash
# 1. Generar par (una vez)
python tools/license_keygen.py

# 2. Emitir licencia test
python tools/license_issue.py \
    --customer "SIGA Test Lab" \
    --email "test@sigasas.com" \
    --plan enterprise

# 3. Instalar token en planta/
cp tools/licenses_issued/siga-test-lab/license.token planta/data/license.token

# 4. Verificar
python planta/license_manager.py

# Output esperado:
#   ✓ LICENCIA VÁLIDA
#     Cliente: SIGA Test Lab
#     Plan: Enterprise — EMA + OMA + FEA + 3D + Reports
#     ...
```

---

## FAQ interna

**¿Por qué RSA y no AES o HMAC?**
Asymmetric crypto. La key que firma (private) no es la misma que verifica (public). Eso significa que la `public_key.pem` puede ir embebida en el .exe del cliente sin riesgo — aunque la extraiga, no puede firmar licencias nuevas.

**¿Por qué JWT y no formato propio?**
PyJWT está battle-tested, las claims standard son entendidas por toda la industria, y el formato es debuggable (base64 separable). No reinventamos rueda.

**¿Por qué no llamar a un servidor de SIGA para validar?**
Porque el cliente compró Watermelon Planta para usar **offline en planta industrial sin internet**. Si dependiéramos de un servidor, sería inútil.

**¿Y si el cliente cambia la hora del PC para evadir vencimiento?**
Aplicamos `leeway=0` → token con `iat > now` es inválido. Para evadir tendría que atrasar la hora a antes de la emisión, lo que rompe otras cosas del SO (certificados HTTPS, etc.).

**¿Puede un cliente compartir su `license.token` con otra empresa?**
Técnicamente sí, no hay binding al hostname. Pero el token tiene el `customer` y `email` del cliente original, así que sería evidente en cualquier audit. Por contrato comercial está prohibido. Si lo vemos en un cliente que no compró, revocamos.

**¿Cómo revoco una licencia?**
Hoy: no hay blacklist online (porque sería online). Mitigación: el .exe del cliente que el infractor tiene seguirá funcionando hasta el `exp`. Para el siguiente release del .exe, agregamos el `jti` a una lista negra hardcoded.

---

## Cambios futuros

- [ ] Implementar lista negra de `jti` revocados en `license_manager.py`
- [ ] Binding opcional a hostname/MAC para licencias enterprise
- [ ] Dashboard interno de SIGA mostrando licencias activas + vencimientos
- [ ] Email automático 30 días antes del vencimiento
- [ ] Generar `license.token` desde una mini-UI en lugar de CLI
