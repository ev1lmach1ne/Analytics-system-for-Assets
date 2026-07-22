# Analytics System for Assets

Aplicación de escritorio (PyQt6) para importar, limpiar, analizar y backtestear datos históricos de activos financieros (cripto, forex, índices, acciones), con QuestDB como base de datos local.

## Requisitos

- **Python 3.10 o superior** (probado en Windows con CPython).
- **Windows** es la plataforma principal. **macOS** tiene soporte parcial (QuestDB también se autoinstala). En **Linux** la app funciona pero QuestDB debe instalarse y arrancarse manualmente (ver más abajo).
- Conexión a internet en el primer uso de Importar/Ajustes (para la descarga automática de QuestDB) y para descargar históricos de los proveedores.

No hace falta instalar Java, Docker ni QuestDB por separado en Windows/macOS.

## Instalación

```bash
git clone <url-del-repo>
cd Analytics-system-for-Assets
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
```

Los scripts sueltos de `library/scripts_utiles/` y `library/Backtests/` tienen dependencias extra propias en `library/requirements.txt` (solo instalarlas si vas a usar esos scripts).

## Configuración

1. Copia `.env.example` a `.env` (opcional: los valores por defecto de QuestDB ya funcionan para una instancia local autoinstalada).
2. Variables disponibles en `.env`:
   - `QUESTDB_HOST`, `QUESTDB_PG_PORT`, `QUESTDB_HTTP_PORT`, `QUESTDB_DATABASE`, `QUESTDB_USER`, `QUESTDB_PASSWORD` — conexión a QuestDB. Con los defaults (`localhost`, puertos 18812/19000) la app gestiona su propia instancia.
   - `BASE_DATA` — carpeta donde se guardan los datos limpios e informes. **Opcional**: si no se define, la app pregunta la carpeta en el primer arranque y la recuerda en `config.json`.

## Arranque

- **Opción A**: doble clic en `launcher.vbs` (lanza la app sin ventana de consola; requiere que `pythonw.exe` esté en el PATH).
- **Opción B**: con el entorno virtual activado, `python app.py` (o `pythonw app.py` para no abrir consola).

En el primer arranque la app pide la carpeta base de datos y crea dentro las subcarpetas `Limpiados/` y `Limpiados/Informes/`.

Para crear un acceso directo en el escritorio: clic derecho → Nuevo → Acceso directo → apuntar a `launcher.vbs` (o a `pythonw.exe app.py` con "Iniciar en" = la carpeta del proyecto) y asignar el icono de `gui/resources` si se desea.

## QuestDB

La app incluye un gestor automático ("QuestDB de bolsillo", `core/questdb_manager.py`):

- La primera vez que uses la pestaña **Importar** o abras **Ajustes**, si no hay un QuestDB accesible en `QUESTDB_HOST`, descarga el binario oficial (con Java embebido) desde GitHub Releases y lo arranca en segundo plano. No requiere permisos de administrador ni Docker.
- Windows y macOS: automático. **Linux**: instala QuestDB manualmente (https://questdb.io/download/) y configura host/puertos en `.env`.
- Si apuntas `QUESTDB_HOST` a un servidor remoto, la app usa ese servidor y no instala nada.

## Estructura del repo

- `app.py` — punto de entrada de la GUI.
- `core/` — configuración, estrategias, backtest, proveedores de datos, gestor de QuestDB.
- `gui/` — ventanas, pestañas y diálogos (PyQt6).
- `tests/` — tests (`python -m pytest tests/ -q`).
- `library/` — scripts sueltos de análisis y backtesting. **Aviso**: algunos contienen rutas absolutas (`D:\...`) de la máquina original y requieren ajuste manual antes de usarse; no forman parte del flujo principal de la GUI.
