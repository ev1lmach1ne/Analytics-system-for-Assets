# Resumen Técnico — Analytics System for Assets

## 1. Propósito general del proyecto

Aplicación de escritorio (Windows/macOS/Linux) para **investigación cuantitativa de activos financieros** (cripto, forex, índices, acciones, futuros). Cubre el ciclo completo de trabajo de un trader/analista en una sola herramienta:

- **Descarga** de datos históricos OHLC de múltiples proveedores.
- **Importación y limpieza** de datos brutos (CSV u orígenes externos) hacia una base de datos local **QuestDB**.
- **Análisis descriptivo** del activo (métricas, gráficos, patrones de velas, calendario económico).
- **Diseño y backtest** de sistemas de trading multi-setup con optimización de parámetros y validación IS/OOS, Walk-Forward y Montecarlo.
- **Exportación** del sistema a código de plataformas de trading (TradingView Pine Script y MetaTrader 5 MQL5).

Es autocontenida y portable: el ejecutable empaquetado (PyInstaller `--onefile`) incluye su propio motor de base de datos (descarga e inicia QuestDB con Java embebido en el primer arranque), sin requerir Python, Java ni Docker preinstalados.

## 2. Módulos o servicios principales y sus responsabilidades

### Núcleo de lógica (`core/`)

| Módulo | Responsabilidad |
|---|---|
| `core/config.py` | Configuración global: rutas (base de datos, `config.json`, `Sistemas/`, `Favoritos/`), credenciales QuestDB (`.env`), timeframes, presets de fricción (slippage/comisión por clase de activo), factores de anualización de métricas. |
| `core/strategies.py` | Catálogo de indicadores y **registro declarativo de estrategias** (`ESTRATEGIAS`): cada una expone su spec de parámetros (de la que la GUI autogenera formularios), una función generadora de señales y una descripción legible. Soporta sistemas multi-setup (hasta 64) mediante bitmasks. |
| `core/backtest.py` | **Motor de backtest vectorizado** (numba): simulación realista (señal en vela `t` → ejecución al open de `t+1`), stop-loss con gestión de gaps, órdenes límite, entrada escalonada por tramos, break-even/trailing, cierres parciales, dimensionamiento por riesgo (`riesgo_pct`), stop por ATR (fijo/dinámico), split IS/OOS, Walk-Forward y Montecarlo, métricas (Sharpe, CAGR, etc.). |
| `core/optimizer.py` | Barrido de parámetros (grid cartesiano) para un setup, restringido al tramo **in-sample**: paraleliza simulaciones (numba `nogil`) en un pool de hilos con caché de señales y límite de combinaciones. |
| `core/metrics.py` | Funciones puras de métricas cuantitativas (Efficiency Ratio, KAMA, Hurst, SAR, percentiles rodantes) compiladas con numba `@njit(nogil=True)` para no bloquear la GUI. |
| `core/candle_patterns.py` | Detección vectorizada de patrones de velas japonesas y estadística de su rendimiento forward (hit rate, significancia binomial, edge vs. base). |
| `core/questdb_manager.py` | Gestión de QuestDB "de bolsillo": descarga, arranque y parada del motor local (con Java embebido en Windows, JRE portátil Temurin en macOS). Solo actúa si el host configurado es localhost; respeta instancias remotas. |
| `core/parsing.py` | Parseo tolerante de valores numéricos "sucios" (formatos europeo/americano, sufijos K/M/B) para importación. |
| `core/rf_registry.py` | Registro auxiliar (`.rf_registry.json`) de la tasa libre de riesgo por archivo limpio. |
| `core/questdb_errors.py` | Errores tipados y mensajes diagnósticos para operaciones con QuestDB. |
| `core/data_providers/` | Proveedores de datos con interfaz común (`BaseProvider`): **ccxt** (exchanges), **Dukascopy**, **yfinance**, **Hyperliquid** y calendario económico. |
| `core/connectors/` | Conectores opcionales (OANDA) y persistencia de configuración (`.connectors.json`). |
| `core/codegen/` | Exportador de sistemas a código de plataformas: capas `ir.py` (árbol neutro), `fidelidad.py` (informe de pérdidas al traducir), emisores `pine.py` (Pine v6) y `mql.py` (MQL5) con librerías runtime portadas. |

### Interfaz gráfica (`gui/`)

| Módulo | Responsabilidad |
|---|---|
| `gui/main_window.py` | Ventana principal con pestañas, precarga asíncrona de pestañas bajo overlay, tema oscuro propio y helpers de Win32. |
| `gui/widgets/tab_descargar.py` | Descarga de históricos desde los proveedores (catálogo, búsqueda, timeframes, ejecución del script como `QProcess`). |
| `gui/widgets/tab_importar.py` | Importación de CSVs a QuestDB: normalización, preseteo de tipo de activo, ejecución de scripts de limpieza, bootstrap de QuestDB. |
| `gui/widgets/tab_analisis.py` | Analizador del activo: métricas (tarjetas KPI), gráficos nativos matplotlib, patrones de velas y calendario económico. |
| `gui/widgets/tab_limpiados.py` | Explorador de datos limpios: vista en tabla, rangos de análisis y descarga de informes. |
| `gui/widgets/tab_comparador.py` | Comparador de múltiples activos/estrategias. |
| `gui/widgets/tab_backtest.py` | Constructor de sistemas, backtest y optimizador (3 sub-pestañas); gráficas de resultados. |
| `gui/widgets/tab_patrones.py` | Sub-pestaña de patrones de velas del Analizador (escaneo en QThread). |
| `gui/widgets/lwc_chart.py` | Visualización estilo TradingView (Lightweight Charts JS offline en QWebEngine) como alternativa a matplotlib. |
| `gui/widgets/console_widget.py` | Consola que ejecuta scripts como `QProcess` (rama `--run-script` del empaquetado). |
| `gui/dialogs/` | Diálogos: primera apertura, ajustes, tutorial, exportación de código (con informe de fidelidad). |
| `gui/questdb_bootstrap.py` | Envoltorio Qt del gestor de QuestDB (hilo con progreso). |

### Scripts y utilidades (`library/`)

- `library/scripts_utiles/` — scripts lanzados como subprocesos por la GUI: `descargar_datos.py`, `preparar_datos.py`, `limpieza_datos_er.py`, `limpieza_diag.py`, `analisis_descriptivo.py`, `categorias_comun.py`, reorganización de carpetas, etc. (algunos con rutas absolutas de la máquina original, fuera del flujo principal).
- `library/Backtests/` — backtests históricos sueltos de referencia (p. ej. `backtest_btc_1h`).

### Infraestructura

- `app.py` — punto de entrada: verificación temprana de dependencias, modo script empaquetado, splash, precarga de pestañas y tutorial.
- `tests/` — suite de pytest (motor de backtest, estrategias, patrones, codegen, config, QuestDB, GUI logic).
- `empaquetar.py` / `empaquetar.bat` — generación de ejecutable autocontenido (PyInstaller) por SO.
- `instalar.bat` / `launcher.vbs` — instalación y arranque desde el código fuente.
- `config.json`, `.connectors.json` — configuración persistente junto al ejecutable.

## 3. Tecnologías y librerías clave

| Capa | Tecnología |
|---|---|
| Lenguaje | Python 3.10+ (venv 3.14 en el entorno actual) |
| GUI | **PyQt6** (estilo Fusion, tema oscuro custom) + **PyQt6-WebEngine** (Chromium) para gráficos LWC |
| Datos / numérico | **pandas**, **numpy**, **numba** (JIT `nogil` para bucles críticos), **scipy**, **matplotlib** |
| Base de datos | **QuestDB** (PostgreSQL wire protocol, puerto 18812; HTTP 19000) vía `psycopg2-binary` |
| Proveedores de datos | **ccxt**, **yfinance**, Dukascopy (descargas paralelas), Hyperliquid, OANDA (`oandapyV20`) |
| APIs externas | TradingEconomics (calendario económico), Finnhub (opcional) |
| Estadística / informes | **statsmodels**, **seaborn** (scripts de análisis), PDF de informes |
| Configuración | `python-dotenv` (`.env`), `config.json` |
| Red | `requests`, `curl-cffi` (entorno), `websockets` |
| Empaquetado | **PyInstaller** (`--onefile`), `customtkinter` (utilidades) |
| Tests | pytest |

## 4. Flujo de datos principal entre componentes

```
┌──────────────┐   descarga (QProcess → core/data_providers)   ┌─────────────────────┐
│   Descargar  │ ─────────────────────────────────────────────▶│  data/<proveedor>/  │
│  (tab_gui)   │        (ccxt, Dukascopy, yfinance...)          │  <categoria>/<activo>│
└──────────────┘                                                └──────────┬──────────┘
                                                                           │
┌──────────────┐   importación + limpieza (scripts_utiles)      ┌──────────▼──────────┐
│   Importar   │ ──────────────────────────────────────────────▶│  QuestDB (local)    │
│  (tab_gui)   │  CSV → normalize/clean → tabla temporal        │  data/Limpiados/CSV │
└──────────────┘                                                └──────────┬──────────┘
                                                                           │
┌──────────────┐   lectura (pandas → DataFrame OHLC)                      │
│   Limpiados  │ ◀─────────────────────────────────────────────────────────┘
│  (tab_gui)   │
└──────┬───────┘
       │  archivo seleccionado
       ▼
┌──────────────┐   análisis (QThread + scripts)          ┌──────────────────────┐
│   Analizar   │ ───────────────────────────────────────▶│  Informes PDF + bundle │
│  (tab_gui)   │   métricas/gráficos/patrones/calendario  │  de arrays (gráficos) │
└──────┬───────┘                                          └──────────────────────┘
       │
       ▼
┌──────────────┐   core/strategies (señales) → core/backtest.simular()          ┌─────────────┐
│   Backtest   │ ─────────────────────────────────────────────────────────────▶│  Resultados  │
│  (tab_gui)   │   parámetros → core/optimizer (solo IS)  ◀── validación ─────▶│  + código    │
└──────┬───────┘                                                            │  (codegen)  │
       │                                                                     └─────────────┘
       ▼
┌──────────────┐   sistemas JSON en Sistemas/, favoritos en Favoritos/
│   Persistir  │   configuración en config.json (junto al .exe), datos en %LOCALAPPDATA%
└──────────────┘
```

**Resumen del ciclo:**

1. **Descarga** → los proveedores (`BaseProvider`) bajan OHLC y lo organizan por proveedor/categoría/activo bajo `data/`.
2. **Importación** → los scripts de limpieza transforman el CSV a formato canónico (con `.meta.json` y registro de tasa libre de riesgo) y lo cargan en **QuestDB**; una copia limpia queda en `data/Limpiados/`.
3. **Análisis** → los CSVs limpios se leen como DataFrames; el analizador calcula métricas y dibuja (matplotlib o LWC), emitiendo informes PDF con un bundle de arrays que la GUI reutiliza para la vista de gráficos (misma información, cero recálculo).
4. **Backtest** → la GUI construye la definición del sistema, `core/strategies.py` genera las señales y `core/backtest.py` (numba, vectorizado) las simula con riesgo/fricción realistas; el optimizador barre parámetros solo sobre IS y la validación completa (IS+OOS+WFA+Montecarlo) corre sobre la serie entera.
5. **Exportación** → `core/codegen/` traduce el sistema a Pine Script v6 / MQL5 con informe de fidelidad previo.
6. **Persistencia** → sistemas (`Sistemas/`), favoritos (`Favoritos/`), configuración (`config.json`) y conexiones (`.connectors.json`) viven junto al ejecutable; QuestDB y los datos, en `%LOCALAPPDATA%\AnalyticsSystemForAssets`.

Todos los procesos pesados (backtest, optimización, análisis, descarga, bootstrap de QuestDB) corren fuera del hilo de la GUI (QThread, `QProcess` o hilos con numba `nogil`), manteniendo la interfaz fluida.
