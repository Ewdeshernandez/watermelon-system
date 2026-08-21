# Watermelon — Kit de campo (PC Windows + cDAQ NI)

La adquisición **corre local** en la PC conectada al cDAQ por USB (la nube no ve
el USB). Lo **offline es para GRABAR**: una vez instalado, captura sin internet y
sincroniza a Supabase cuando vuelve la conexión.

## Qué llevar en la USB
1. **Este repo** (Watermelon) — desde GitHub → *Code → Download ZIP*, o `git clone`.
2. **Instalador NI-DAQmx** (de ni.com, ~1.5 GB) — **obligatorio**, es el driver del cDAQ.
3. **Python 3.11** (instalador de python.org) — tildá *Add Python to PATH*.
4. Tu **`.streamlit/secrets.toml`** (con las credenciales Supabase) — copialo dentro del repo.
5. *(Opcional, solo si la PC NO tendrá internet nunca)* la carpeta `field/wheels`
   con los paquetes descargados (ver abajo).

## Pasos en la PC (una vez, con internet para el driver)
1. Instalar **NI-DAQmx** (el instalador de ni.com). Reiniciar si lo pide.
2. Instalar **Python 3.11** (Add to PATH).
3. Doble clic en **`field\setup_windows.bat`** → crea el entorno e instala todo.
4. Copiar tu **`.streamlit\secrets.toml`** al repo.
5. Doble clic en **`field\ni_check.bat`** → confirma que el cDAQ y las 5 señales
   llegan (keyphasor + 4 radiales) y que el keyphasor pulsa. **Guardá esa salida.**
6. Doble clic en **`field\run_watermelon.bat`** → abre la app en `localhost:8501`.
   Login → **Remote Monitoring** → Fuente **Campo** → máquina **Rotor_Kit_SIGA_1**
   → **▶ Iniciar** → órbitas/Bode reales. **⏺ Grabar** el runup.

## Después (captura offline)
- Sin internet: grabás igual; queda **pendiente**.
- Con internet: **☁ Subir pendientes** → Supabase. Otros lo reprocesan desde la web.

## Instalación 100% OFFLINE (opcional)
Si la PC no tendrá internet ni para el `pip install`, en una PC CON internet
(Windows) generá los wheels y llevalos en la USB dentro de `field/wheels`:

    pip download -r requirements.txt nidaqmx -d field\wheels

Luego `setup_windows.bat` los detecta e instala sin internet.
(El **driver NI-DAQmx** igual hay que instalarlo aparte — no es un wheel.)

## Recordá (hardware de este test)
- 9234 = **IEPE OFF** automático (canales AC) → seguro para proximidad, no daña nada.
- Tomás la **vibración dinámica** (órbitas/Bode/cascada). El **gap DC/centerline**
  espera al **9229**. El keyphasor en 9234 es marginal (AC) — si la fase sale con
  ruido, es por eso.
