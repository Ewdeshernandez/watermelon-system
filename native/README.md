# Watermelon Field — módulo NATIVO de adquisición (Windows/campo)

App de escritorio **nativa** (PySide6 + pyqtgraph) para la PC conectada al cDAQ.
Tiempo real sólido, **sin navegador → no se traba**. Reusa TODO el motor de
`core/remote_monitoring` (adquisición NI, FFT, order tracking, diagnóstico API
670/684). La **web queda solo para análisis y reportes**.

## Por qué nativo (y no web para el live)
Bently **ADRE 408 / System1**, Emerson **AMS**, SKF **Microlog** — todos son
**nativos**. El tiempo real (órbitas, Bode, live) necesita render nativo veloz,
no un navegador sobre websocket. La web es para análisis multi-usuario y reportes.

## Cómo lo superamos
- Un **solo motor Python** con **rotodinámica API 670/684** (whirl/whip, AF, Ncrit)
  + rodamientos/ISO — combina lo de Bently (rotor) y Emerson/SKF (predictivo).
- **Nube integrada** (multi-especialista, reportes, biblioteca) que ellos cobran aparte.
- **Abierto y a medida**, sin licencias por canal.

## Stack
- **PySide6** (GUI nativa) + **pyqtgraph** (plots en tiempo real, 15–30 FPS).
- **nidaqmx** (adquisición NI, 1 tarea + reloj compartido = simultáneo).
- Motor compartido: `core/remote_monitoring/*` (agent, stream_source, transient,
  keyphasor, recorder, diagnóstico).
- Empaquetado a **instalador Windows** con PyInstaller (fase de release).

## Correr
```bat
:: demo (Mac/dev, sin hardware)
python native/watermelon_field.py --sim

:: campo real (Windows, NI-DAQmx): 2 acelerómetros 100 mV/g en Mod1 ai0,ai1
python native/watermelon_field.py --sens 100 --fs 5120
```
Instalar deps: `pip install -r native/requirements.txt`

## Roadmap
- **v0.1 (actual):** live waveforms + espectro + barras de nivel + rpm + grabar. ✅
- **v0.2:** órbita en vivo (par X/Y), keyphasor, marcadores 1X/2X, alarmas ISO 20816.
- **v0.3:** Bode/Cascada/Polar en vivo durante runup + diagnóstico whirl/whip.
- **v0.4:** panel de configuración (canales/sensores) + carga desde Supabase.
- **v0.5:** empaquetado a instalador Windows + auto-sync de grabaciones a la nube.
- **v1.0:** rutas/predictivo (rodamientos, envelope) para pelear con Emerson/SKF.
