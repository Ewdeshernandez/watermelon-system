# Firma de código — Watermelon Planta (Azure Trusted / Artifact Signing)

Objetivo: firmar digitalmente `WatermelonPlanta.exe` y el instalador para que
**Smart App Control (SAC)** y SmartScreen dejen de bloquearlos en los PCs de los
clientes. El workflow (`.github/workflows/build-planta.yml`) ya tiene los pasos
de firma; **solo faltan las credenciales de Azure**. Mientras no existan, el
build corre igual pero SIN firmar.

> Nota: en enero 2026 Microsoft renombró "Trusted Signing" → **"Artifact
> Signing"**. Es el mismo servicio.

---

## Requisito previo
Identidad de la organización verificable. Para el perfil de certificado tipo
**"Public Trust"** Microsoft valida legalmente a la empresa (SIGA GROUP S.A.S).
La verificación tarda 1–5 días hábiles. Costo del servicio: ~US$9.99/mes.

## Paso 1 — Crear los recursos en Azure Portal
1. Entra a https://portal.azure.com con la cuenta de la empresa.
2. Busca **"Trusted Signing"** (o "Artifact Signing") → **Create**.
   - Crea un **Trusted Signing Account** (elige región, ej. `East US` o
     `West Europe`). Anota el **endpoint** que te muestra, ej.
     `https://eus.codesigning.azure.net`.
3. Dentro de la cuenta → **Identity validation** → verifica la organización
   (razón social, NIT, dirección). Espera la aprobación.
4. Ya aprobada → crea un **Certificate Profile** tipo **Public Trust**. Anota
   su **nombre**.

## Paso 2 — Crear el "App Registration" (identidad para GitHub)
1. Azure Portal → **Microsoft Entra ID** → **App registrations** → **New**.
   - Nombre: `watermelon-github-signing`. Créala.
2. Anota de la app: **Application (client) ID** y **Directory (tenant) ID**.
3. En la app → **Certificates & secrets** → **New client secret** → copia el
   **Value** (se muestra una sola vez).

## Paso 3 — Dar permiso de firma a esa identidad
1. Vuelve al **Trusted Signing Account** → **Access control (IAM)** →
   **Add role assignment**.
2. Rol: **Trusted Signing Certificate Profile Signer**.
3. Asigna a la app `watermelon-github-signing`.

## Paso 4 — Cargar los secrets en GitHub
Repo → **Settings → Secrets and variables → Actions → New repository secret**.
Crea estos 6:

| Secret | Valor |
|---|---|
| `AZURE_TENANT_ID` | Directory (tenant) ID |
| `AZURE_CLIENT_ID` | Application (client) ID |
| `AZURE_CLIENT_SECRET` | el Value del client secret |
| `AZURE_SIGNING_ENDPOINT` | ej. `https://eus.codesigning.azure.net` |
| `AZURE_SIGNING_ACCOUNT` | nombre del Trusted Signing Account |
| `AZURE_SIGNING_PROFILE` | nombre del Certificate Profile |

En cuanto exista `AZURE_CLIENT_ID`, los pasos de firma se activan solos en el
siguiente build.

## Paso 5 — Verificar
1. Haz un push que dispare el build (o Actions → Run workflow).
2. En los logs deben aparecer en verde los pasos **"Firmar
   WatermelonPlanta.exe"** y **"Firmar el instalador Setup.exe"**.
3. Descarga el instalador, clic derecho → **Propiedades → Firmas digitales**:
   debe listar a **SIGA GROUP S.A.S** con timestamp.
4. Instálalo en un PC con Smart App Control activo: ya **no** debe bloquearse.

---

## Notas
- SAC además de la firma usa reputación/ML. Con firma de Public Trust el
  bloqueo desaparece; en apps nuevas puede tardar unos días en ganar
  reputación plena, pero el "publisher no verificable" se elimina de inmediato.
- La versión de la acción usada es `azure/trusted-signing-action@v0.5.9`. Si
  falla por versión, revisa la última en
  https://github.com/Azure/trusted-signing-action/releases y actualiza el tag.
- Mientras se monta esto, el workaround para el técnico es apagar SAC
  (Configuración → Seguridad de Windows → Control de aplicaciones y navegador →
  Smart App Control → Desactivado); en Windows 11 24H2/25H2 se puede reactivar.
