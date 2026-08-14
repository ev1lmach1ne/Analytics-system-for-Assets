# Analytics System for Assets

Aplicación de escritorio (PyQt6) para importar, limpiar, analizar y backtestear datos históricos de activos financieros (cripto, forex, índices, acciones), con QuestDB como base de datos local.

## 🖥️ Requisitos del sistema

Para ejecutar **`AnalyticsSystem.exe`**. Es autocontenido: **no** hace falta instalar Python, Java, Docker ni QuestDB, y **no** pide permisos de administrador. Descargar y doble clic.

|                       | 🟡 Mínimos                                  | 🟢 Recomendados                                    |
|-----------------------|---------------------------------------------|----------------------------------------------------|
| **Sistema operativo** | Windows 10 de 64 bits                       | Windows 11 de 64 bits                              |
| **Procesador**        | 4 núcleos x86-64                            | 6-8 núcleos con buena frecuencia por núcleo        |
| **Memoria RAM**       | 8 GB                                        | 16 GB                                              |
| **Espacio libre**     | 5 GB                                        | 20 GB o más                                        |
| **Almacenamiento**    | Disco duro (HDD)                            | SSD, mejor NVMe                                    |
| **Gráficos**          | La integrada del equipo                     | La integrada; no se aprovecha una GPU dedicada     |
| **Pantalla**          | 1280 × 800                                  | 1400 × 900 o superior                              |
| **Conexión**          | Solo para el primer arranque y las descargas | Estable (4-8 descargas en paralelo)               |
| **Software previo**   | Ninguno                                     | Ninguno                                            |

**Mínimos** = la app arranca y se trabaja con uno o dos activos a la vez.
**Recomendados** = optimizaciones largas, comparador y varios históricos grandes abiertos a la vez.

### ⚠️ Dos cosas antes de ejecutarlo

1. **Déjalo en una carpeta donde puedas escribir.** El `.exe` guarda su `config.json`, tus sistemas y tus favoritos **junto a sí mismo**. Sirve el Escritorio, Documentos o una carpeta propia tipo `D:\AnalyticsSystem`. **No lo pongas en `C:\Program Files`**: ahí Windows no le deja escribir sin administrador y perderías la configuración.
2. **Windows SmartScreen avisará la primera vez.** El `.exe` no está firmado digitalmente, así que Windows muestra *"Windows protegió tu PC"*. Es **Más información → Ejecutar de todas formas**. Solo pasa la primera vez.

<details>
<summary><b>Por qué estas cifras</b> (desglose de disco y detalle técnico)</summary>

#### Espacio en disco, desglosado

| Qué                                      | Cuánto      | Dónde                                        |
|------------------------------------------|-------------|----------------------------------------------|
| El propio `AnalyticsSystem.exe`          | ~342 MB     | Donde lo dejes                               |
| Descompresión temporal en cada arranque  | ~394 MB     | `%TEMP%` (normalmente en `C:`)               |
| QuestDB (motor + Java embebido)          | ~120 MB     | `%LOCALAPPDATA%\AnalyticsSystemForAssets`    |
| Tus datos históricos                     | crece       | La misma carpeta de QuestDB                  |

Los datos son lo único que crece de verdad: en la máquina de desarrollo, con varios años de velas de 1 minuto, la carpeta de QuestDB va por **~1 GB**.

#### El resto

- **Arranque**: el `.exe` está construido en modo *onefile*, así que **cada vez que lo abres se descomprime entero (~394 MB) en `%TEMP%`** y se borra al cerrar. Por eso el primer arranque es lento (el antivirus escanea todo eso) y por eso hace falta espacio libre en la unidad del sistema, no solo donde guardes el `.exe`.
- **Procesador**: el optimizador y el backtest corren en un solo hilo (con [numba](https://numba.pydata.org/) compilando los bucles críticos), así que pesa más la velocidad de un núcleo que el número de núcleos. Los núcleos extra sí se aprovechan al descargar históricos (4-8 conexiones en paralelo según el proveedor) y para que la GUI siga fluida mientras QuestDB trabaja de fondo. Arquitectura **x86-64**: en equipos ARM (Windows on ARM) funciona por emulación.
- **RAM**: la app suma tres consumidores — la GUI de PyQt6 con gráficos QtWebEngine (Chromium), los DataFrames de pandas del activo cargado, y la JVM de QuestDB en segundo plano. Con 8 GB se trabaja bien sobre uno o dos activos; con 16 GB se puede tener varios históricos largos en memoria y el comparador abierto sin que el sistema empiece a paginar.
- **Disco SSD**: QuestDB lee sus tablas por *memory-mapped files* y en un disco mecánico se nota mucho al importar y al cargar rangos largos. La descompresión de cada arranque también agradece el SSD.
- **Pantalla**: la ventana pide 1400 × 900, y si no cabe se recorta automáticamente al 92 % del área visible, así que en portátiles pequeños funciona igual, solo con menos sitio para los gráficos.
- **Sistema operativo**: el mínimo de Windows 10 viene de Qt 6, sobre el que está construida la GUI (PyQt6).
- **Internet**: solo para la descarga inicial de QuestDB (~120 MB, una vez) y para bajar históricos. El análisis y el backtest funcionan sin conexión sobre los datos ya importados.

</details>

<details>
<summary><b>Otras formas de ejecutarlo</b> (macOS, Linux y código fuente)</summary>

- **macOS**: `empaquetar.py` genera un `.app` autocontenido. QuestDB se autoinstala igual (descarga además un JRE portátil de Eclipse Temurin). Hace falta macOS 11 o superior, también por Qt 6.
- **Linux**: `empaquetar.py` genera un `.tar.gz`. La app funciona, pero QuestDB hay que instalarlo y arrancarlo a mano (ver más abajo).
- **Desde el código fuente**: además de lo anterior necesitas **Python 3.10 o superior** (`instalar.bat` instala Python 3.12 si no encuentra ninguno) y **~1,2 GB extra** para el entorno virtual con las dependencias.

Los requisitos de equipo son equivalentes en los tres casos.

</details>

## Instalación

Tres formas, de la más simple a la más manual. **Si solo quieres usar la app, la opción 1 es la tuya.**

### ⚡ Opción 1 — Portable: descargar el `.exe` y abrirlo

Sin Python, sin comandos, sin instalador. Un único archivo.

1. Descarga **`AnalyticsSystem.exe`** de la [última versión publicada](https://github.com/ev1lmach1ne/Analytics-system-for-Assets/releases/latest).
2. Muévelo a una carpeta **donde puedas escribir**: el Escritorio, Documentos o una carpeta propia tipo `D:\AnalyticsSystem`. Recuerda: **no lo pongas en `C:\Program Files`** (ver [Requisitos del sistema](#-requisitos-del-sistema)).
3. Doble clic. La primera vez Windows preguntará: **Más información → Ejecutar de todas formas**.

Eso es todo. En el primer arranque la app pide la carpeta donde guardar los datos y se encarga sola de descargar y levantar QuestDB cuando haga falta.

> **El primer arranque tarda** (medio minuto o así): el `.exe` se descomprime entero en `%TEMP%` y el antivirus lo escanea. Los siguientes son más rápidos.

**Dónde deja sus cosas**: `config.json`, `Sistemas/` y `Favoritos/` se crean **junto al `.exe`**, así que si lo mueves, llévate la carpeta entera. QuestDB y los históricos viven aparte, en `%LOCALAPPDATA%\AnalyticsSystemForAssets`.

**Para actualizar**: sustituye el `.exe` por el nuevo en la misma carpeta. Tu configuración, sistemas y favoritos siguen ahí.

**Para desinstalar**: borra la carpeta del `.exe` y `%LOCALAPPDATA%\AnalyticsSystemForAssets`. No toca el registro ni deja nada más.

<details>
<summary>Generar el <code>.exe</code> tú mismo</summary>

Desde el código fuente (opción 2 o 3), con el entorno virtual activado:

```bash
python empaquetar.py
```

Detecta el sistema operativo y deja el resultado en `dist/`: `AnalyticsSystem.exe` en Windows, `AnalyticsSystem.app` en macOS y un `.tar.gz` en Linux. La primera vez instala PyInstaller si no lo tienes.

</details>

### Opción 2 — Instalador desde el código (Windows)

Para tener el código a mano y poder modificarlo, sin pelearte con la línea de comandos:

1. Descarga (o clona) esta carpeta.
2. Doble clic en **`instalar.bat`**.

El instalador localiza Python (y si no lo tienes, lo instala con winget), crea el entorno virtual, instala las dependencias y deja un acceso directo **"Analytics System"** en el escritorio. Se puede volver a ejecutar en cualquier momento: solo instala lo que falte.

### Opción 3 — Manual (Windows / macOS / Linux)

```bash
git clone <url-del-repo>
cd Analytics-system-for-Assets
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
```

## Actualizar a una versión nueva

**Con el `.exe` portable**: descarga el nuevo y sustituye el viejo en su carpeta. Nada más.

**Desde el código**: descarga la versión nueva **sobre la misma carpeta** (o `git pull`): así se conserva tu configuración (`config.json`, `.env`). Tus datos y QuestDB viven fuera del proyecto y no se tocan.

No hace falta reinstalar: al abrir la app con el acceso directo (o `launcher.vbs`), si la versión nueva trae dependencias nuevas se instalan solas antes de arrancar.

Los scripts sueltos de `library/scripts_utiles/` y `library/Backtests/` tienen dependencias extra propias en `library/requirements.txt` (solo instalarlas si vas a usar esos scripts).

## Configuración

**Si usas el `.exe` portable puedes saltarte esta sección entera**: la app pregunta la carpeta de datos en el primer arranque y los valores por defecto de QuestDB funcionan tal cual. Solo hace falta tocar nada de esto si quieres apuntar a una QuestDB remota.

1. Copia `.env.example` a `.env` (opcional: los valores por defecto de QuestDB ya funcionan para una instancia local autoinstalada).
2. Variables disponibles en `.env`:
   - `QUESTDB_HOST`, `QUESTDB_PG_PORT`, `QUESTDB_HTTP_PORT`, `QUESTDB_DATABASE`, `QUESTDB_USER`, `QUESTDB_PASSWORD` — conexión a QuestDB. Con los defaults (`localhost`, puertos 18812/19000) la app gestiona su propia instancia.
   - `BASE_DATA` — carpeta donde se guardan los datos limpios e informes. **Opcional**: si no se define, la app pregunta la carpeta en el primer arranque y la recuerda en `config.json`.

> **Cambiar la carpeta de datos desde Ajustes requiere reiniciar la aplicación.** La ruta nueva se guarda en `config.json`, pero se aplica por completo en el siguiente arranque: hasta entonces las pestañas abiertas siguen usando la carpeta anterior para no dejar archivos a medias entre dos rutas.

## Arranque

Con el **`.exe` portable**, doble clic y ya. Desde el código hay tres vías:

- **Opción A**: el acceso directo **"Analytics System"** del escritorio (lo crea `instalar.bat`).
- **Opción B**: doble clic en `launcher.vbs` — usa el venv del proyecto si existe (si no, el `pythonw.exe` del sistema) y actualiza las dependencias automáticamente si `requirements.txt` cambió.
- **Opción C**: con el entorno virtual activado, `python app.py` (o `pythonw app.py` para no abrir consola).

En el primer arranque la app pide la carpeta base de datos y crea dentro las subcarpetas `Limpiados/` y `Limpiados/Informes/`.

Para crear un acceso directo a mano: clic derecho → Nuevo → Acceso directo → apuntar a `launcher.vbs` con "Iniciar en" = la carpeta del proyecto, y asignar el icono `icon.ico` de la raíz si se desea.

## QuestDB

La app incluye un gestor automático ("QuestDB de bolsillo", `core/questdb_manager.py`):

- La primera vez que uses la pestaña **Importar** o abras **Ajustes**, si no hay un QuestDB accesible en `QUESTDB_HOST`, descarga el binario oficial (con Java embebido) desde GitHub Releases y lo arranca en segundo plano. No requiere permisos de administrador ni Docker.
- Windows y macOS: automático. **Linux**: instala QuestDB manualmente (https://questdb.io/download/) y configura host/puertos en `.env`.
- Si apuntas `QUESTDB_HOST` a un servidor remoto, la app usa ese servidor y no instala nada.

## Riesgo y Stop Loss en el backtest

- **Riesgo del setup** (`riesgo_pct`) es el límite de riesgo **nominal** de cada operación: `distancia al stop × volumen`, acotado por `equity × riesgo_pct`. Las comisiones y el slippage se descuentan aparte del resultado y no entran en ese presupuesto.
- **Stop Loss × ATR** tiene dos modos por setup:
  - **Fijo**: el stop se ancla a la primera entrada y no se mueve al añadir promedios.
  - **Dinámico por promedio**: tras cada entrada se reancla al precio medio de la posición con el ATR del momento, sin bajar nunca un stop ya mejorado por Break Even o Trailing.
- En ambos modos, cada promedio solo consume el riesgo que queda libre; el riesgo nominal total nunca supera el presupuesto del setup.
- El ATR de una entrada se calcula con la **última vela cerrada** (nunca la vela de entrada, que aún no existe al operar su apertura).
- **Gaps**: si una vela abre al otro lado del stop, el cierre se llena al precio de apertura (pérdida real por hueco), no al nivel teórico del stop. Un tramo pendiente no se ejecuta si la vela abre atravesando el stop.
- `Stop Loss = 0` significa "sin stop real": en ese caso no puede garantizarse una pérdida máxima.
- **Velas interpoladas**: el toggle `Excluir velas interpoladas de las señales` está encendido por defecto. Las velas siguen en la serie para conservar continuidad e indicadores, pero no pueden generar entradas nuevas ni órdenes límite; las salidas y la gestión de posiciones abiertas siguen activas. Al apagarlo, las velas interpoladas se tratan como cualquier otra y vuelven a poder generar señales.

## Estructura del repo

- `app.py` — punto de entrada de la GUI.
- `core/` — configuración, estrategias, backtest, proveedores de datos, gestor de QuestDB.
- `gui/` — ventanas, pestañas y diálogos (PyQt6).
- `tests/` — tests (`python -m pytest tests/ -q`).
- `library/` — scripts sueltos de análisis y backtesting. **Aviso**: algunos contienen rutas absolutas (`D:\...`) de la máquina original y requieren ajuste manual antes de usarse; no forman parte del flujo principal de la GUI.
