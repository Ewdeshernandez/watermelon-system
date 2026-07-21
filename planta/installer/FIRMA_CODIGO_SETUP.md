# Firma de código — Watermelon Planta (SSL.com OV + eSigner cloud)

Objetivo: firmar digitalmente `WatermelonPlanta.exe` y el instalador para que
**Smart App Control (SAC)** y SmartScreen dejen de bloquearlos en los PCs de los
clientes. El workflow (`.github/workflows/build-planta.yml`) ya tiene los pasos
de firma con **SSL.com eSigner**; **solo faltan las credenciales** (los 4
secrets `SSL_*`). Mientras no existan, el build corre igual pero SIN firmar.

> Se descartó Azure Trusted Signing: NO está disponible para organizaciones de
> Colombia (solo EE.UU./Canadá/UE/UK). SSL.com sí emite a empresas colombianas.

## Producto comprado
- **OV Code Signing** (US$129/año) — pone el nombre de SIGA GROUP S.A.S en cada
  instalador y elimina "Editor no verificado".
- **eSigner Cloud Signing Tier 1** (US$180/año, 30 días gratis) — firma desde
  GitHub Actions sin token USB, vía HSM en la nube. 240 firmas/año.

---

## Paso 1 — Activar cuenta y validar la organización (TÚ)
1. Verificá el correo de activación de SSL.com en **ehernandez@sigasas.com**.
2. Completá la orden con los datos legales de **SIGA GROUP S.A.S**: razón social,
   NIT, dirección registrada, teléfono verificable (te llaman/mandan código).
   Tené a mano el Certificado de Cámara de Comercio.
3. SSL.com valida en **3–5 días hábiles**. El certificado OV se emite ahí.

## Paso 2 — Enrolar el certificado en eSigner + sacar credenciales (TÚ)
En el portal SSL.com Manager, una vez emitido el certificado:
1. Enrolá el certificado OV en **eSigner** (si no quedó activo desde la compra).
2. Configurá la **autenticación TOTP** para firma automatizada: SSL.com te da un
   **QR / secret TOTP** — guardá el **secret** (texto), NO solo el QR.
3. Anotá el **credential_id** del certificado (aparece en el detalle del cert /
   en la API de eSigner).

## Paso 3 — Cargar los 4 secrets en GitHub (TÚ, te guío)
Repo → **Settings → Secrets and variables → Actions → New repository secret**:

| Secret | Valor |
|---|---|
| `SSL_USERNAME` | tu usuario de SSL.com |
| `SSL_PASSWORD` | tu contraseña de SSL.com |
| `SSL_CREDENTIAL_ID` | el credential_id del certificado en eSigner |
| `SSL_TOTP_SECRET` | el secret TOTP (texto) de eSigner |

En cuanto exista `SSL_USERNAME`, los pasos de firma se activan solos en el
siguiente build.

## Paso 4 — Verificar (YO)
1. Disparás un build (push que toque VERSION/planta/core/modal, o Run workflow).
2. En los logs deben salir en verde **"Firmar WatermelonPlanta.exe (SSL.com
   eSigner)"** y **"Firmar el instalador Setup.exe (SSL.com eSigner)"**.
3. Descargá el instalador → clic derecho → **Propiedades → Firmas digitales**:
   debe listar **SIGA GROUP S.A.S** con timestamp.
4. Instalalo en un PC con Smart App Control activo: el bloqueo de "editor no
   verificable" desaparece. (La reputación plena ante SAC/SmartScreen se
   acumula con las descargas; puede tardar unos días/algunas descargas.)

---

## Notas
- Acción usada: `sslcom/esigner-codesign@develop`. Si falla por versión, revisá
  https://github.com/SSLcom/esigner-codesign y fijá un tag estable.
- El primer build firmado hay que verificarlo — puede requerir ajustar
  `file_path`/`override` según cómo devuelva la acción los archivos firmados.
- Tier 1 = 240 firmas/año. El workflow firma 2 archivos por release (.exe +
  instalador); alcanza para ~120 releases/año. Si se firma en cada push conviene
  gatear la firma solo en releases.
- Mientras se completa todo esto, el workaround para el técnico es apagar Smart
  App Control (Configuración → Seguridad de Windows → Control de aplicaciones y
  navegador → Smart App Control → Desactivado); en Win 11 24H2/25H2 se reactiva.
