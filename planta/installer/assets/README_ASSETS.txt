============================================================
  ASSETS para Watermelon Planta Installer
============================================================

Archivos necesarios para que el build funcione:

  watermelon.ico      ← icono del .exe + installer (256x256 multi-res)
  watermelon-logo.svg ← logo vectorial fuente (ya está en este folder)
  splash.png          ← imagen del wizard installer (164x314 px)
  license.txt         ← EULA (ya está en este folder)


============================================================
  CÓMO GENERAR watermelon.ico (UNA SOLA VEZ)
============================================================

Opción A — Online (sin instalar nada, 2 min):

  1. Ve a https://convertio.co/svg-ico/
  2. Sube el archivo watermelon-logo.svg
  3. Click "Convert"
  4. Descarga el .ico generado
  5. Guárdalo en este folder como  watermelon.ico

Opción B — Con ImageMagick (CLI):

  magick watermelon-logo.svg -resize 256x256 ^
         -define icon:auto-resize=16,32,48,64,128,256 ^
         watermelon.ico


============================================================
  CÓMO GENERAR splash.png (UNA SOLA VEZ)
============================================================

Opción A — Online:

  1. Ve a https://convertio.co/svg-png/
  2. Sube watermelon-logo.svg
  3. Pon resolución 164x314 (ese es el tamaño que Inno Setup espera)
  4. Descarga como  splash.png  y guárdalo en este folder

Opción B — Con Inkscape (si lo tienes):

  inkscape watermelon-logo.svg --export-type=png ^
           --export-width=164 --export-height=314 ^
           --export-filename=splash.png

NOTA: Inno Setup viene con imágenes default (WizModernImage-IS.bmp y
WizModernSmallImage-IS.bmp) que se pueden usar si no querés generar el
splash custom. El installer.iss actual ya las referencia. Si querés tu
propio splash, edita installer.iss para apuntar al splash.png/splash.bmp.


============================================================
  TIMELINE
============================================================

1. Generar watermelon.ico  ← Lo más importante (sin esto el .exe no
   tiene icono y el installer falla)
2. (Opcional) Generar splash.png si querés branding completo
3. Ejecutar build_exe.bat → genera dist\WatermelonPlanta.exe
4. Ejecutar build_installer.bat → genera dist\WatermelonPlantaSetup-v1.0.exe
5. Ese .exe es el que mandas a clientes.

Para FASE C inicial: solo necesitamos watermelon.ico. El resto puede
quedar con defaults de Inno Setup.


============================================================
  CONTACTO
============================================================

Para cuestiones de branding/diseño profesional, contactar a un
diseñador. El logo SVG actual es funcional pero un diseñador podría
mejorarlo para v2.0.

soporte@sigasas.com
