============================================================
  WATERMELON PLANTA EDITION
  Captura modal offline · Sistema de adquisicion Watermelon
============================================================

Esta es una aplicacion standalone que corre en una laptop
de planta con la maleta Watermelon conectada por USB.
Permite al tecnico configurar canales y disparar capturas
SIN INTERNET.

Despues, cuando la laptop vuelva a tener conexion, los
archivos generados se sincronizan automaticamente al
Watermelon Cloud para procesamiento avanzado.


============================================================
  PRE-REQUISITOS (UNA SOLA VEZ)
============================================================

Ya instalados en este PC si seguiste el setup anterior:
  - Windows 10/11 64-bit
  - Python 3.10+ con "Add to PATH" marcado
  - Drivers de adquisicion Watermelon instalados
  - License token de Watermelon Planta (te lo entrega SIGA)


============================================================
  PASO 1 - INSTALAR
============================================================

  1. Conecta la maleta Watermelon al USB del PC
  2. Verifica que los indicadores de power esten encendidos
  3. Doble click en  INSTALAR.bat
  4. Espera ~1-2 min mientras instala dependencias
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
       "Maleta Watermelon conectada"
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
     mensaje verde y el nombre del archivo generado


============================================================
  PASO 4 - SINCRONIZAR AL WATERMELON CLOUD
============================================================

Cuando este PC vuelva a tener internet:

  1. En la pantalla principal de la app, abajo veras
     la seccion "Sincronizacion con Watermelon Cloud"

  2. Inicia sesion con tu usuario y password
     (la primera vez necesitas internet activo)

  3. Click en "Sync ahora" — sube las capturas pendientes

  4. Una vez en el Cloud, podes procesarlas con todas las
     herramientas avanzadas (EMA, OMA, FEA, Mode Shapes 3D)


============================================================
  DONDE QUEDAN LOS ARCHIVOS
============================================================

Todos los archivos generados quedan en:

   data\captures\

Nombrados con timestamp:
   planta_ema_20260520_143055.<formato>
   planta_oma_20260520_150812.<formato>

Puedes copiarlos a un USB para llevarlos a oficina si
preferis no usar el sync automatico.


============================================================
  PROBLEMAS COMUNES
============================================================

* "Python no se reconoce..."
  -> Reinstala Python 3.12 marcando "Add to PATH"

* "Sin maleta Watermelon conectada"
  -> Verifica cable USB, cambia puerto USB
  -> Verifica que los indicadores de power esten encendidos
  -> Reinicia el PC si el problema persiste

* "Drivers de adquisicion no detectados"
  -> Corre INSTALAR.bat de nuevo
  -> Si persiste, contacta soporte SIGA

* "Licencia no valida"
  -> Verifica que el archivo license.token este en data\
  -> Si tu licencia vencio, contacta a SIGA para renovar

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
  SOPORTE TECNICO
============================================================

Para reportar problemas o pedir features nuevos:

  SIGA GROUP S.A.S
  Email:  ehernandez@sigasas.com
  Web:    https://watermelonsys.net

(c) 2026 SIGA GROUP S.A.S — Todos los derechos reservados
