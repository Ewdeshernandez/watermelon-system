============================================================
  WATERMELON PLANTA EDITION
  Captura modal offline para hardware NI cDAQ-9178 + NI-9234
============================================================

Este folder contiene una mini-app standalone que corre en una
laptop de planta con la maleta NI conectada por USB. Permite al
tecnico configurar canales y disparar capturas SIN INTERNET.

Despues, cuando la laptop vuelva a tener conexion, los TDMS
generados se pueden subir al Watermelon Cloud (manualmente por
ahora; auto-sync viene en proximo release).


============================================================
  PRE-REQUISITOS (UNA SOLA VEZ)
============================================================

Ya instalados en este PC si seguiste el setup anterior:
  - Windows 10/11 64-bit
  - Python 3.10+ con "Add to PATH" marcado
  - NI-DAQmx driver instalado (NI MAX detecta la maleta)
  - Git for Windows (opcional)


============================================================
  PASO 1 - INSTALAR
============================================================

  1. Conecta la maleta NI cDAQ-9178 al USB del PC
  2. Verifica que aparece en NI MAX como "cDAQ1"
  3. Doble click en  INSTALAR.bat
  4. Espera ~1-2 min mientras instala dependencias Python
  5. Cuando termine, te dice "INSTALACION COMPLETA"

Si INSTALAR.bat da error sobre Python:
  - Verifica que python --version funciona en PowerShell
  - Si no, reinstala Python 3.12 marcando "Add to PATH"


============================================================
  PASO 2 - ABRIR LA APP
============================================================

  1. Doble click en  INICIAR.bat
  2. Se abre una ventana negra (PowerShell)
  3. Despues de 5-10 segundos, tu browser default abre
     automaticamente en  http://localhost:8501
  4. La app de Watermelon Planta aparece en el browser

Si el browser NO abre solo:
  - Abre Chrome o Firefox manualmente
  - Pega esta URL:  http://localhost:8501


============================================================
  PASO 3 - HACER UNA CAPTURA
============================================================

  En la pantalla principal de la app:

  1. Verifica que en la parte superior dice:
       "Online" o "Offline" (no importa cual)
       "N modulo(s) NI-9234 detectado(s)"
       "N captura(s) guardada(s)"

  2. Selecciona el tipo de ensayo:
       - EMA: para ensayo con martillo modal (maquina parada)
       - OMA: para captura continua (maquina rotando)

  3. En la siguiente pantalla:
       - Ajusta sample rate, duracion, etc.
       - En la tabla de canales, marca cuales vas a usar
       - Edita el nombre del sensor y la sensibilidad
       - El sistema valida automaticamente

  4. Click en  "Iniciar captura ahora"
       - Sigue las instrucciones en pantalla
       - Para EMA: dispara el martillo cuando dice
       - Para OMA: espera a que termine los segundos

  5. Cuando termine, te dice "Captura completa" con un
     mensaje verde y el nombre del archivo TDMS generado


============================================================
  PASO 4 - SUBIR AL WATERMELON CLOUD
============================================================

Cuando este PC vuelva a tener internet:

  Opcion A - Manual (HOY):
    1. Abre tu Mac o cualquier PC con internet
    2. Abre  https://wm-home-final-2026.streamlit.app
    3. Login con tu usuario
    4. Sidebar: Modal Analysis
    5. Pestana: Adquisicion
    6. Radio: "Importar archivo de captura existente"
    7. Sube el .tdms que esta en  planta\data\captures\
    8. Watermelon Cloud procesa y muestra modos modales

  Opcion B - Auto-sync (PROXIMO RELEASE):
    En el siguiente sprint, esta app va a detectar cuando
    vuelve el internet y subir automaticamente los TDMS
    pendientes al Watermelon Cloud, sin que tengas que
    hacer nada.


============================================================
  DONDE QUEDAN LOS ARCHIVOS
============================================================

Todos los TDMS generados quedan en:

   planta\data\captures\

Nombrados con timestamp:
   planta_ema_20260520_143055.tdms
   planta_oma_20260520_150812.tdms

Puedes copiarlos a un USB para llevarlos a oficina.


============================================================
  PROBLEMAS COMUNES
============================================================

* "Python no se reconoce..."
  -> Reinstala Python 3.12 marcando "Add to PATH"

* "Sin maleta NI conectada"
  -> Verifica cable USB, cambia puerto USB
  -> Abre NI MAX y mira si detecta la maleta
  -> Si NI MAX la ve pero la app no, reinicia el PC

* "Error de captura: ..."
  -> Verifica que los sensores esten bien conectados
  -> Verifica las sensibilidades configuradas
  -> Verifica que no estes habilitando canales en slots vacios

* "Browser no abre"
  -> Abre Chrome manualmente en  http://localhost:8501

* App lenta o cuelga
  -> Cierra la ventana negra (PowerShell)
  -> Vuelve a hacer doble click en INICIAR.bat


============================================================
  SOPORTE
============================================================

Para reportar problemas o pedir features nuevos:

  Ewdes Hernandez
  ehernandez@sigasas.com

Repositorio del codigo:
  https://github.com/Ewdeshernandez/watermelon-system
