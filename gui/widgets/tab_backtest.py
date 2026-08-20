"""
Pestaña "Backtest": tres sub-pestañas, cada una con una única responsabilidad.

- Constructor: selección de activo (file explorer sobre LIMPIADOS_DIR) y
  constructor de SISTEMAS: lista de setups, cada uno con su plantilla
  (predefinida o custom por reglas), sus parámetros y su PROPIO riesgo %,
  stop ATR, take-profit y salida por tiempo; la definición (reglas) de cada
  plantilla se muestra interpolada con los parámetros. Cuenta global
  (capital/comisión/slippage), split IS/OOS con slider (defecto 70/30) y
  Walk-Forward opcional. Los sistemas se guardan/cargan en JSON. Desde aquí
  se lanza tanto "▶ Ejecutar backtest" (una configuración fija, serie
  completa) como "🔍 Prueba de parametrización (Solo IS)" (barrido de
  parámetros).
- Optimizador: resultado del barrido de parámetros — core/optimizer.py corre
  cada combinación ÚNICAMENTE sobre el tramo IS (nunca ve el tramo OOS) y
  esta pestaña las compara: scatter grande retorno-vs-riesgo + tabla
  compacta con sparkline de la equity IS, enlazados por selección. Elegir
  una combinación aquí la vuelca de vuelta en el Constructor.
- Resultados: métricas IS/OOS/Total, gráfico de log-return acumulado con
  flechas de compras/ventas y sombreado IS vs OOS, zoom por periodo (toolbar
  matplotlib + rango de fechas), tabla de trades, ventanas WFA y Montecarlo
  — siempre de UNA única configuración ya fijada, corrida sobre toda la
  serie con "▶ Ejecutar backtest".

Backtest y optimización corren en QThreads (las funciones numba usan nogil,
la GUI no se congela). Payload y señales calculadas con core/backtest,
core/optimizer y core/strategies.
"""
import copy
import html
import inspect
import json
import os
import re
import shutil
import time

import numpy as np
import pandas as pd
from PyQt6.QtCore import (Qt, QObject, QThread, pyqtSignal, QDate, QSize,
                          QEvent, QTimer, QRectF, QUrl, QPropertyAnimation,
                          QEasingCurve)
from PyQt6.QtGui import (QColor, QShortcut, QKeySequence, QPainter, QPen, QPalette,
                         QDesktopServices)
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QComboBox,
    QCheckBox, QScrollArea, QFrame, QTableWidget, QTableWidgetItem,
    QHeaderView, QPushButton, QSplitter, QLineEdit, QSpinBox, QDoubleSpinBox,
    QSlider, QListWidget, QListWidgetItem, QTabWidget, QFormLayout, QGroupBox,
    QDateEdit, QSizePolicy, QApplication, QDialog,
    QButtonGroup, QStyledItemDelegate, QStyle, QStackedWidget, QMenu,
    QRadioButton, QToolButton, QGraphicsOpacityEffect,
)
from gui.dialogs.base import DialogoBase, crear_dialogo
from gui.dialogs.mensajes import pedir_texto, confirmar, informacion
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from matplotlib.dates import date2num
import matplotlib.dates as mdates
from matplotlib.collections import PolyCollection, LineCollection
from matplotlib.patches import FancyArrowPatch
from matplotlib.ticker import MaxNLocator

from core.config import (
    LIMPIADOS_DIR, SISTEMAS_DIR, FAVORITOS_DIR, TF_PATTERN, tf_to_minutes,
    tipo_activo_de_csv, PRESETS_FRICCION, TIPO_MAP,
    velas_por_anio as velas_por_anio_config,
    velas_por_dia as velas_por_dia_config,
    get_selector_recientes, set_selector_recientes,
    velas_a_tiempo_legible, get_te_api_key,
)
from core.backtest import (
    simular, dividir_is_oos, calcular_metricas, montecarlo, resultado_filtrado,
    MOTIVOS_SALIDA, MECANISMOS_SALIDA, RESULTADOS_ORDEN, ORDEN_RELLENADA,
)
from core.optimizer import (
    optimizar_setup, n_combinaciones, LIMITE_COMBOS_DEFECTO, PREFIJO_RIESGO,
    estadisticas_conjunto, analisis_por_parametro, analisis_vecindad,
    fiabilidad_estadistica,
)
from core.strategies import (
    ESTRATEGIAS, params_por_defecto, generar_senales_sistema,
    describir, describir_setup, codigo_sistema, defaults_setup, MAX_SETUPS,
    _INDICADORES_REGLA, _OPERADORES_REGLA, _filtros_por_defecto,
    _mascaras_condiciones_dir, etapa_salida_por_defecto, filas_plantilla,
    trigger_etapa, salida_mecanismo_por_defecto, validar_parciales,
    validar_tramos, validar_setup, AVISO_EXCESO_PARCIALES, tramo_entrada_por_defecto, trigger_tramo,
    sma, ema, rsi, atr, bollinger, stochastic, williams_r, cci, _kama_serie,
    _sar_serie, donchian, _zigzag_pivotes, _ZIGZAG_DESVIACION_REGLA,
    _supertrend_serie, _macd_series, _adx_series, _aroon_series, _cmo_serie,
    _trix_serie, _stochrsi_series, _ichimoku_series, _keltner_series,
    _ttm_squeeze_series, _vwap_series,
    _entrada_por_defecto, NIVELES_FIB, tramos_zigzag_vigentes,
    preparar_eventos_noticias,
    _er_serie, _hurst_serie, ventana_regimen_defecto,
    METODOS_REGIMEN_ER, METODOS_REGIMEN_HURST,
    UMBRAL_ER_TENDENCIA, UMBRAL_ER_RUIDO,
    UMBRAL_HURST_TENDENCIA, UMBRAL_HURST_REVERSION,
    VENTANA_ER_DEFECTO, VENTANA_HURST_DEFECTO, VENTANA_HURST_MINIMA,
)
from core.candle_patterns import detectar_patrones
from core.data_providers import economic_calendar
from gui.widgets.file_explorer import FileExplorer
from gui.widgets.bombear import bombear_eventos
from gui.widgets.progress_animada import ProgressBarAnimada
from gui.widgets.loading_overlay import (mostrar_overlay_tab, ocultar_overlay_tab,
                                         apagar_overlay_tab)
from gui.widgets.tf_common import TF_LABELS, parsear_tf_custom, regla_de_tf
from gui.widgets.lwc_chart import LwcChart, WEBENGINE_OK
from gui.widgets.plot_common import (
    icono_ayuda as _icono_ayuda_popup,
    icono_ayuda_texto as _icono_ayuda_texto,
    panel_flotante_dialog as _panel_flotante_dialog,
    eje_fechas_ordinal,
)

# fuente del sistema, la misma pila que Lightweight Charts en Windows
# (Segoe UI -> Trebuchet MS -> la por defecto de matplotlib si no están).
# rcParams es global: toda la app comparte el mismo criterio tipográfico.
import matplotlib as _mpl
from matplotlib import font_manager as _font_manager
try:
    _fuentes_instaladas = {f.name for f in _font_manager.fontManager.ttflist}
except Exception:
    _fuentes_instaladas = set()
for _familia_lwc in ('Segoe UI', 'Trebuchet MS', 'Verdana'):
    if _familia_lwc in _fuentes_instaladas:
        _mpl.rcParams['font.family'] = [_familia_lwc] + list(
            _mpl.rcParams['font.family'])
        break

FIG_BG = '#0d1424'
AX_FG = '#c8d6e5'
GRID_C = '#253a60'
VERDE = '#2ecc71'
ROJO = '#e74c3c'
GRIS = '#5a7a9a'
AMBAR = '#f1c40f'
AZUL = '#4fc3f7'

# cruceta y último precio con la apariencia por defecto de Lightweight Charts
# (la vista moderna), para que la clásica se lea igual: líneas de cruceta
# #758696 semitransparentes y etiquetas con fondo #4c525e y texto blanco
CRUCETA_COLOR = '#758696'
CRUCETA_BG = '#4c525e'
CRUCETA_TX = '#ffffff'

# flechas de operaciones: tonos más saturados que las velas (VERDE/ROJO de
# arriba) para que los marcadores no se camuflen sobre cuerpos del mismo color.
# VERDE_FLECHA además se mantiene notablemente más oscuro que VERDE (vela
# alcista) para que la flecha de compra no se confunda con el color de vela.
VERDE_FLECHA = '#1b8a3a'   # compra (abre largo / cierra corto)
ROJO_FLECHA = '#ff1744'    # venta (abre corto / cierra largo)

# ZigZag: el azul por defecto del indicador de TradingView que se tomó como
# referencia. No se reutiliza AZUL porque es el primer color de la paleta de
# medias y la polilínea se confundiría con una media móvil.
AZUL_ZIGZAG = '#2962FF'
# tramos de Fibonacci: ámbar apagado, para que la banda quede por detrás de
# todo sin competir con las velas ni con los segmentos de las órdenes
AMARILLO_FIB = '#c9a227'

# color fijo por periodo para las medias (SMA/EMA) dibujadas en el gráfico de
# Resultados — así una media de 200 siempre se identifica por su color sin
# importar el orden en que aparezcan los demás setups/filtros.
COLOR_MEDIA_FIJO = {20: '#2B7FFF', 50: '#FF8904', 200: '#800000'}
# resto de medias: paleta rotatoria, en orden de periodo creciente
PALETA_MEDIA = [AZUL, AMBAR, '#2ecc71', '#9b59b6', '#e67e22']

# un color por familia de overlay. Viven aquí y no como literales dentro del
# dibujo porque el panel de capas los usa para su cuadradito: si cada sitio
# tuviera el suyo, el color del panel dejaría de corresponderse con la línea.
COLOR_BOLLINGER = '#9b59b6'
COLOR_KAMA = '#ab47bc'
COLOR_DONCHIAN = '#26a69a'
COLOR_SAR = '#8d6e63'
COLOR_SUPERTREND = '#d4a5f5'
COLOR_ICHIMOKU_TENKAN = '#00bcd4'
COLOR_ICHIMOKU_KIJUN = '#ff7043'
COLOR_ICHIMOKU_SENKOU = '#78909c'
COLOR_KELTNER = '#26a69a'
# grises del VWAP: tonos distintos por configuración (anclaje/k/modo) para
# que canales solapados no se apilen en una sola mancha oscura
_PALETA_VWAP = ['#9aa7b8', '#7a8ba3', '#b8c4d4', '#5d6f8a']
COLOR_PATRONES = '#2ecc71'

# ── paneles apilados del gráfico de operaciones (precio + osciladores) ──
# El precio y cada oscilador activo comparten una única Figure/Axes-stack
# (sharex) en vez de canvases separados, para que el zoom/pan del precio
# arrastre gratis a los osciladores. Las posiciones se fijan a mano vía
# ax.set_position() (ver _aplicar_pesos_paneles) en vez de dejarlas al
# GridSpec, para poder redimensionar en vivo arrastrando el borde entre
# paneles sin reconstruir los Axes en cada frame.
PESO_PRECIO_DEFECTO = 3.0
PESO_OSC_DEFECTO = 1.0
PESO_PANEL_MIN = 0.15
# margen ajustado a la esquina (sin RangeSlider ya no hace falta reservar
# BOTTOM_STACK para su franja) y RIGHT_PANEL casi al borde, igual que la
# escala de precio de Lightweight Charts en la vista moderna.
LEFT_PANEL, RIGHT_PANEL = 0.035, 0.95
TOP_PANEL, BOTTOM_STACK = 0.94, 0.075
GAP_PANEL = 0.018
ETIQUETA_PANEL = {'precio': 'Precio', 'rsi': 'RSI', 'atr': 'ATR',
                  'stoch': 'Estocástico', 'williams': 'Williams %R', 'cci': 'CCI',
                  'er': 'Régimen ER', 'hurst': 'Régimen Hurst',
                  'macd': 'MACD', 'adx': 'ADX (+DI/−DI)', 'aroon': 'Aroon',
                  'cmo': 'CMO', 'trix': 'TRIX', 'stochrsi': 'StochRSI',
                  'ttm': 'TTM Squeeze', 'vwap': 'Distancia VWAP (σ)'}
# primer color de la paleta de cada panel: es el que representa al oscilador en
# el cuadradito del panel de capas
COLOR_PANEL_OSC = {'rsi': '#ffffff', 'atr': '#2ecc71', 'stoch': '#26c6da',
                   'williams': '#ec407a', 'cci': '#5c6bc0',
                   'er': '#ff9800', 'hurst': '#ab47bc',
                   'macd': '#4fc3f7', 'adx': '#7986cb', 'aroon': '#1abc9c',
                   'cmo': '#ffa726', 'trix': '#ff7043', 'stochrsi': '#fd79a8',
                   'ttm': '#ffb300', 'vwap': '#9aa7b8'}
# (kind, clave en _recolectar_indicadores) de cada panel apilado bajo el precio,
# en el orden en que se apilan. Un solo sitio: lo recorren el dibujo y el
# catálogo de capas.
PANELES_OSCILADOR = (('rsi', 'rsis'), ('atr', 'atrs'), ('stoch', 'stochs'),
                     ('williams', 'williams'), ('cci', 'ccis'),
                     ('macd', 'macds'), ('adx', 'adxs'), ('aroon', 'aroones'),
                     ('cmo', 'cmos'), ('trix', 'trixes'),
                     ('stochrsi', 'stochrsis'), ('ttm', 'ttms'),
                     ('vwap', 'vwaps'),
                     ('er', 'ers'), ('hurst', 'hursts'))
# banda que el filtro ADMITE en cada método, para sombrearla en su panel
BANDA_REGIMEN = {
    'er_tendencia': (UMBRAL_ER_TENDENCIA, 1.0),
    'er_rango': (0.0, UMBRAL_ER_RUIDO),
    'hurst_tendencia': (UMBRAL_HURST_TENDENCIA, 1.0),
    'hurst_reversion': (0.0, UMBRAL_HURST_REVERSION),
}

MODOS_WFA = [
    'Retorno %',
    'Retorno acumulado',
    'Max DD %',
    'Win rate %',
    'Retorno vs Max DD',
    'Trades',
    'Curva de Equidad Combinada (OOS)',
]

MODOS_EQUITY = [
    'Retorno %',
    'Capital',
    'Log-retorno',
    'Drawdown',
]

MODOS_MFE_MAE = [
    'Eficiencia MFE/MAE',
    'Distribución MFE/MAE',
    'Eficiencia Entrada/Salida',
]

# modos del selector de dirección de la pestaña Resultados (ids del QButtonGroup)
_MODO_TODOS, _MODO_LARGOS, _MODO_CORTOS, _MODO_COMPARAR = range(4)

RUTA_ESTRATEGIAS_GUARDADAS_LEGACY = os.path.join(LIMPIADOS_DIR, 'backtest_estrategias.json')
_TIPO_MAP_INV = {v: k for k, v in TIPO_MAP.items()}   # 'CRYPTO' -> 'Crypto'


def _slug_sistema(nombre):
    """Nombre de carpeta seguro (bajo Sistemas/) a partir del nombre del sistema."""
    return re.sub(r'[^\w\-]+', '_', nombre.strip().lower()).strip('_') or 'sistema'


def _config_serializable(config):
    """Parte de la config del motor que se puede guardar en JSON.

    Además de la cuenta, config lleva las máscaras precomputadas de
    condiciones (parciales_masks_long y las otras cinco): listas de arrays
    numpy que json.dump no sabe serializar. Y como json.dump escribe A MEDIDA
    que recorre, al toparse con la primera dejaba el archivo cortado por la
    mitad; _leer_favoritos descarta esa carpeta al fallar el json.load, así
    que el favorito desaparecía sin un solo mensaje de error.

    Se descarta lo que no sea serializable en vez de nombrar las claves
    conocidas: así una máscara nueva no puede volver a romperlo en silencio.
    Lo que la recarga necesita son los escalares de cuenta (ver
    _cargar_favorito_nombre), que sobreviven al filtro."""
    limpio = {}
    for clave, valor in (config or {}).items():
        try:
            json.dumps(valor)
        except (TypeError, ValueError):
            continue
        limpio[clave] = valor
    return limpio


def _guardar_json_atomico(ruta, datos):
    """Escribe JSON a un temporal y lo renombra encima del destino.

    Un fallo a mitad de escritura deja el temporal a medias pero NO toca el
    archivo bueno, que es lo que convirtió el error de serialización en un
    favorito irrecuperable."""
    tmp = f"{ruta}.tmp"
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(datos, f, ensure_ascii=False, indent=2)
    os.replace(tmp, ruta)


def _nombre_activo_limpio(csv_path):
    """'BTCUSDT_1h_limpiado.csv' -> 'BTCUSDT' (quita tf + sufijo limpiado/limpio,
    que siempre están presentes en los CSV — ver TF_PATTERN)."""
    stem = os.path.splitext(os.path.basename(csv_path))[0]
    stem = TF_PATTERN.sub('', stem)
    return stem.replace('_', ' ').strip() or stem


def _titulo_activo_html(csv_path, tf):
    """Nombre de activo limpio en negrita + temporalidad en texto plano
    (sin recuadro), reutilizado en el título de Resultados y en el Constructor."""
    nombre = html.escape(_nombre_activo_limpio(csv_path))
    if not tf:
        return f"<b style='font-size:14px'>{nombre}</b>"
    return (f"<b style='font-size:14px'>{nombre}</b> "
            f"<span style='color:#4fc3f7'>· {html.escape(tf)}</span>")


def _texto_periodo(ts):
    """'01/01/2020 → 15/03/2025 · 5 años 2 meses' a partir del índice de
    tiempos del backtest. Cadena vacía si no hay velas.

    La duración es APROXIMADA (mes = 30,44 días, año = 12 de esos meses): es
    una etiqueta para dar escala al resto del título, no un dato del que
    dependa ninguna métrica — para eso está velas_anio, que sí sale del
    muestreo real.

    Se cuentan meses y de ahí se derivan los años, en vez de dividir por 365,25
    y quedarse con el resto: así un año justo (365 días) no se queda en
    "0 años 11 meses" por el redondeo hacia abajo.
    """
    if len(ts) == 0:
        return ''
    dias = (ts[-1] - ts[0]).total_seconds() / 86400
    if dias < 31:
        n = int(dias)
        dur = f"{n} día{'s' if n != 1 else ''}"
    else:
        meses_totales = max(1, round(dias / 30.44))
        anios, meses = divmod(meses_totales, 12)
        if anios:
            dur = f"{anios} año{'s' if anios != 1 else ''}"
            if meses:
                dur += f" {meses} mes{'es' if meses != 1 else ''}"
        else:
            dur = f"{meses} mes{'es' if meses != 1 else ''}"
    return (f"{ts[0].strftime('%d/%m/%Y')} → {ts[-1].strftime('%d/%m/%Y')} "
            f"· {dur}")


def _migrar_estrategias_legacy():
    """Migración one-shot: vuelca las entradas del viejo JSON único
    (Limpiados/backtest_estrategias.json) a carpetas individuales bajo
    Sistemas/, y renombra el JSON viejo para no volver a migrarlo."""
    if not os.path.exists(RUTA_ESTRATEGIAS_GUARDADAS_LEGACY):
        return
    try:
        with open(RUTA_ESTRATEGIAS_GUARDADAS_LEGACY, encoding='utf-8') as f:
            datos_legacy = json.load(f)
    except (OSError, json.JSONDecodeError):
        return
    for nombre, datos in datos_legacy.items():
        carpeta = os.path.join(SISTEMAS_DIR, _slug_sistema(nombre))
        destino = os.path.join(carpeta, 'sistema.json')
        if os.path.exists(destino):
            continue
        os.makedirs(carpeta, exist_ok=True)
        datos_migrados = dict(datos)
        datos_migrados['nombre'] = nombre
        with open(destino, 'w', encoding='utf-8') as f:
            json.dump(datos_migrados, f, ensure_ascii=False, indent=2)
    try:
        os.replace(RUTA_ESTRATEGIAS_GUARDADAS_LEGACY,
                   RUTA_ESTRATEGIAS_GUARDADAS_LEGACY + '.migrado')
    except OSError:
        pass

_CHEVRON_SVG = os.path.join(os.path.dirname(__file__), '..', 'assets',
                            'chevron-down.svg').replace('\\', '/')

STYLE_BACKTEST = """
QLabel#titulo   { color: #4fc3f7; font-size: 13px; font-weight: bold; }
QLabel#estado   { color: #5a7a9a; font-size: 11px; }
QLabel#campo    { color: #c8d6e5; font-size: 11px; }
QGroupBox {
    color: #4fc3f7; font-size: 11px; font-weight: bold;
    border: 1px solid #253a60; border-radius: 6px; margin-top: 8px;
    padding-top: 4px;
}
QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 3px; }
QTableWidget {
    background-color: #0d1424; alternate-background-color: #101a2e;
    color: #c8d6e5; gridline-color: #253a60;
    border: 1px solid #253a60; font-size: 11px;
}
QHeaderView::section {
    background-color: #141e30; color: #4fc3f7;
    border: 1px solid #253a60; padding: 3px; font-size: 11px;
}
QPushButton#run {
    background-color: #1c4a2e; color: #2ecc71; font-weight: bold;
    padding: 8px 16px; border: 1px solid #2ecc71; border-radius: 4px;
}
QPushButton#run:disabled { background-color: #16202f; color: #3a5a7a; border-color: #3a5a7a; }
QPushButton#tf {
    background-color: #1a2a45; color: #5a7a9a; padding: 4px 10px;
    font-size: 11px; min-width: 32px;
}
QPushButton#tf:checked {
    background-color: #2a4a6a; color: #4fc3f7; font-weight: bold;
}
QPushButton#tf:disabled {
    background-color: #12192a; color: #33465e;
}
QPushButton#edge {
    background-color: #1a2a45; color: #8fb3d9; padding: 6px 12px;
    border: 1px solid #253a60; border-radius: 4px; font-size: 11px;
}
QPushButton#edge:checked {
    background-color: #3a2f10; color: #f1c40f; font-weight: bold;
    border: 1px solid #f1c40f;
}
QSpinBox:disabled, QDoubleSpinBox:disabled {
    background-color: #0d1220; color: #33465e; border: 1px solid #1b2840;
}
QComboBox:disabled {
    background-color: #0d1220; color: #33465e; border: 1px solid #1b2840;
}
QLabel:disabled { color: #33465e; }
QToolTip {
    background-color: #101a2e; color: #c8d6e5;
    border: 1px solid #253a60; border-radius: 4px;
    padding: 6px; font-size: 11px;
}
QComboBox { background-color: #111828; combobox-popup: 0; }
QComboBox::drop-down { border: none; background: transparent; width: 22px; }
QComboBox::down-arrow {
    image: url("__CHEVRON__");
    width: 12px; height: 8px;
}
QComboBox::down-arrow:on { }
QComboBox QAbstractItemView {
    background-color: #1a2a45; color: #c8d6e5;
    selection-background-color: #2a4a6a; selection-color: #4fc3f7;
    border: 1px solid #253a60; outline: none; margin: 0px;
}
QComboBox QAbstractItemView::item {
    padding: 6px 10px; border-bottom: 1px solid #182030;
}
QComboBox QAbstractItemView::item:selected {
    background-color: #2a4a6a; color: #4fc3f7;
}
QComboBox QAbstractItemView::item:hover {
    background-color: #223755;
}
QComboBox QAbstractItemView::item:disabled {
    background-color: #0d1424; color: #3a5a7a;
}
QComboBox QAbstractItemView::item:disabled:hover {
    background-color: #0d1424;
}
"""


def _no_crash(fn):
    """Una excepción sin capturar en un slot PyQt6 aborta el proceso.

    Trunca los argumentos posicionales extra a la aridad real de `fn`: Qt
    reenvía al wrapper TODOS los argumentos de la señal (p.ej. el `bool
    checked` de `clicked`) porque `wrapper(self, *a, **kw)` acepta cualquier
    cantidad, y eso rompía con un TypeError silencioso (tragado más abajo)
    cualquier slot conectado a una señal con argumentos que no los declarara."""
    n_extra = sum(1 for p in list(inspect.signature(fn).parameters.values())[1:]
                  if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD))

    def wrapper(self, *a, **kw):
        try:
            return fn(self, *a[:n_extra], **kw)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[Backtest] Error en {fn.__name__}: {e}")
    wrapper.__name__ = fn.__name__
    return wrapper


_RUTA_PERF = os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), 'backtest_perf.log')


def _log_slow(nombre, ms):
    """Registra operaciones del Backtester que tarden >200 ms en
    backtest_perf.log, para diagnosticar congelaciones en equipos lentos."""
    if ms < 200:
        return
    try:
        with open(_RUTA_PERF, 'a', encoding='utf-8') as f:
            f.write(f"[{time.strftime('%H:%M:%S')}] {nombre}: {ms:.0f} ms\n")
    except Exception:
        pass


def _style_ax(ax):
    ax.set_facecolor(FIG_BG)
    ax.tick_params(colors=AX_FG, labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_C)
    # grid sólido como Lightweight Charts (mismo #253a60, sin atenuar)
    ax.grid(True, alpha=1.0, color=GRID_C, linewidth=0.8)


def _fmt(v, dec=2, sufijo=''):
    if v is None:
        return '—'
    if v == float('inf'):
        return '∞'
    return f"{v:.{dec}f}{sufijo}"


def _fmt_precio(v):
    """Formato del badge de último precio: decimales según la magnitud,
    como el formateador de la escala de Lightweight Charts."""
    av = abs(v)
    if av >= 100:
        dec = 2
    elif av >= 1:
        dec = 4
    elif av >= 0.01:
        dec = 4
    else:
        dec = 6
    return f"{v:,.{dec}f}"


def _marcar_niveles_eje(ax, niveles):
    """Escribe en el eje Y de un panel EXACTAMENTE los valores que significan
    algo (70/50/30 del RSI, umbrales del ER...), cada etiqueta del color de su
    línea horizontal.

    El repartidor automático de matplotlib elige múltiplos redondos — en el RSI,
    0-20-40-60-80-100 — así que justo los números que definen el indicador eran
    los únicos que no se leían: había que estimar a ojo dónde caía la línea de
    puntos de sobrecompra.

    Devuelve False (y no toca el eje) si no hay niveles que marcar o si son
    demasiados para un panel de ~90 px; en ese caso el llamador deja el
    localizador automático, que siempre da alguna referencia de escala.
    """
    MAX_MARCAS = 6
    if not niveles:
        return False
    por_valor = {}
    for valor, color in niveles:
        por_valor.setdefault(float(valor), color)
    if len(por_valor) > MAX_MARCAS:
        return False
    valores = sorted(por_valor)
    ax.set_yticks(valores)
    ax.set_yticklabels([f'{v:g}' for v in valores])
    for valor, etiqueta in zip(valores, ax.get_yticklabels()):
        etiqueta.set_color(por_valor[valor])
    return True


def _decimar_ohlc(x, o, h, l, c, x0, x1, max_velas=2500):
    """Recorta a la ventana visible [x0, x1] y, si hay más de max_velas
    puntos, dibuja una vela REAL de cada `paso` (submuestreo, no agregación
    en bloques): la vela que se ve de lejos es la MISMA que se ve de cerca —
    al hacer zoom solo aparecen las intermedias, ninguna cambia de color ni
    de forma. (La agregación por bloques creaba velas sintéticas cuyo
    open/high/low/close no era el de ninguna vela real, y el gráfico parecía
    enseñar "otras velas" según el zoom.)"""
    i0, i1 = np.searchsorted(x, [x0, x1])
    i0 = max(i0 - 1, 0)
    i1 = min(i1 + 1, len(x))
    if i1 <= i0:
        i1 = min(i0 + 1, len(x))
    xs, os_, hs, ls, cs = x[i0:i1], o[i0:i1], h[i0:i1], l[i0:i1], c[i0:i1]
    n = len(xs)
    if n <= max_velas:
        return xs, os_, hs, ls, cs
    paso = int(np.ceil(n / max_velas))
    idx = np.arange(0, n, paso)
    return xs[idx], os_[idx], hs[idx], ls[idx], cs[idx]


def _decimar_linea(x_all, y_all, x0, x1, max_puntos=2500):
    """Recorta una serie (x, y) de overlay/oscilador a la ventana visible
    [x0, x1] y, si quedan más de max_puntos, la submuestrea con el MISMO
    criterio que las velas (_decimar_ohlc): se dibujan puntos REALES de la
    serie, ninguno sintético, y al acercar solo aparecen los intermedios.

    El indexado con numpy conserva los NaN (huecos de las series con tramos
    sin valor), así que no deforma líneas discontinuas. `x_all` es el índice
    de vela (monótono), igual que `_x_full`."""
    i0 = max(int(np.searchsorted(x_all, x0)) - 1, 0)
    i1 = min(int(np.searchsorted(x_all, x1)) + 1, len(x_all))
    if i1 <= i0:
        i1 = min(i0 + 1, len(x_all))
    xs, ys = x_all[i0:i1], y_all[i0:i1]
    m = len(xs)
    if m <= max_puntos:
        return xs, ys
    paso = int(np.ceil(m / max_puntos))
    idx = np.arange(0, m, paso)
    return xs[idx], ys[idx]


# pasos máximos de la animación de zoom con la rueda (ver _avanzar_zoom):
# antes convergía tras ~20 pasos (~600 ms de animación con los frames de
# blit de ~30 ms, que era el lag/parpadeo al hacer scroll); con 6 pasos el
# easing se resuelve en ~180 ms, ágil pero sigue siendo animado.
MAX_PASOS_ZOOM = 6


def _cargar_ohlc(csv_path, regla_resample=None):
    """Lee timestamp/open/high/low/close de un CSV limpiado y, si se pide,
    lo reagrega a una temporalidad más gruesa (misma agregación que
    tab_patrones._ResampleThread). Compartido por _BacktestThread y
    _OptimizerThread."""
    cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
            'interpolado', 'anomalia']
    try:
        df = pd.read_csv(csv_path, usecols=cols, engine='pyarrow')
    except (ImportError, ValueError):
        df = pd.read_csv(csv_path, usecols=lambda c: c in cols)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp', 'close']).sort_values('timestamp')
    df = df.reset_index(drop=True)
    for nombre in ('interpolado', 'anomalia'):
        if nombre not in df.columns:
            df[nombre] = 0
        else:
            df[nombre] = (pd.to_numeric(df[nombre], errors='coerce')
                          .fillna(0).astype(np.int8))
    if 'volume' not in df.columns:
        df['volume'] = 0
    else:
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0)
    if regla_resample:
        agg = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
               'volume': 'sum',
               'interpolado': 'max', 'anomalia': 'max'}
        df = (df.set_index('timestamp')[
                ['open', 'high', 'low', 'close', 'volume',
                 'interpolado', 'anomalia']]
                .resample(regla_resample).agg(agg)
                .dropna(subset=['close'])
                .reset_index())
    return df


def _algun_setup_usa_noticias(setups):
    return any((s.get('filtros') or {}).get('noticias', {}).get('activo') for s in setups)


def _resolver_monedas_noticias(setups, csv_path):
    """Copia superficial de `setups` con filtros.noticias.monedas=None
    resuelto al instrumento del CSV (heurística por nombre de archivo). No se
    persiste en el setup original: un sistema guardado sigue siendo portable
    si se reutiliza con otro activo — la resolución es solo para esta
    corrida."""
    nombre_activo = os.path.splitext(os.path.basename(csv_path or ''))[0]
    resueltas = None
    out = []
    for s in setups:
        noticias = (s.get('filtros') or {}).get('noticias')
        if noticias and noticias.get('activo') and noticias.get('monedas') is None:
            if resueltas is None:
                resueltas = economic_calendar.monedas_de_instrumento(nombre_activo)
            s2 = dict(s)
            s2['filtros'] = dict(s['filtros'])
            s2['filtros']['noticias'] = dict(noticias)
            s2['filtros']['noticias']['monedas'] = resueltas
            out.append(s2)
        else:
            out.append(s)
    return out


def _cargar_eventos_noticias(df, csv_path, progress_callback=None):
    """Descarga/cachea (TradingEconomics) los eventos económicos que cubren
    el rango de `df`. Lanza la excepción tal cual si falla (sin API key, red,
    etc.) — el llamador decide cómo mostrarla."""
    return economic_calendar.obtener_eventos(
        df['timestamp'].iloc[0], df['timestamp'].iloc[-1],
        api_key=get_te_api_key(),
        progress_callback=progress_callback)


def _velas_por_anio(df, csv_path=None):
    """Mediana del paso temporal -> nº de velas/año, para anualizar retorno
    y Sharpe. El factor anual usa la sesión real de la clase de activo del
    CSV (CRYPTO 24/7/365, STOCK/FUTURO/FOREX con su calendario de trading);
    si no se puede determinar la clase, cae al supuesto 24/7/365 anterior."""
    difs = df['timestamp'].diff().dropna().dt.total_seconds() / 60.0
    minutos_vela = float(difs.median()) if len(difs) else 60.0
    tipo = tipo_activo_de_csv(csv_path) if csv_path else None
    return velas_por_anio_config(tipo, minutos_vela)


def _velas_por_dia(df, csv_path=None):
    """Nº de velas por día de trading de la clase de activo del CSV, para
    métricas por tiempo (p. ej. 'avg_trades_por_dia'). Misma sesión real
    (STOCK 390 min/día, CRYPTO/FUTURO/FOREX 1440) que _velas_por_anio."""
    difs = df['timestamp'].diff().dropna().dt.total_seconds() / 60.0
    minutos_vela = float(difs.median()) if len(difs) else 60.0
    tipo = tipo_activo_de_csv(csv_path) if csv_path else None
    return velas_por_dia_config(tipo, minutos_vela)


# ══════════════ hilo de backtest ══════════════
class _DetectarDiasThread(QThread):
    """Lee el timestamp del CSV en segundo plano para saber qué días de la
    semana tienen velas — leer el CSV entero en el hilo principal congela la
    UI al seleccionar un activo grande."""

    terminado = pyqtSignal(set)

    def __init__(self, path, parent=None):
        super().__init__(parent)
        self._path = path

    def run(self):
        try:
            try:
                df_ts = pd.read_csv(self._path, usecols=['timestamp'],
                                    engine='pyarrow')
            except (ImportError, ValueError):
                df_ts = pd.read_csv(self._path,
                                    usecols=lambda c: c == 'timestamp')
            dow = pd.to_datetime(df_ts['timestamp'],
                                 errors='coerce').dt.dayofweek.dropna()
            dias = set(int(d) for d in dow.unique()) or set(range(7))
        except Exception:
            dias = set(range(7))
        self.terminado.emit(dias)


class _BacktestThread(QThread):
    """Lee el CSV, genera las señales del SISTEMA (multi-setup) y corre
    motor + métricas + WFA + MC. IMPORTANTE: parentar al widget (destruir un
    QThread corriendo aborta el proceso, misma nota que tab_patrones)."""
    computed = pyqtSignal(object)
    estado = pyqtSignal(str)   # mensajes de fase (p.ej. descarga de eventos)
    progreso = pyqtSignal(int, int)   # hitos de fase (i, total=100)

    def __init__(self, csv_path, setups, config_global, pct_oos,
                 wfa_activo, wfa_ventanas, codigo='', tf_label=None,
                 regla_resample=None, excluir_interpoladas=False, parent=None):
        super().__init__(parent)
        self._csv = csv_path
        self._setups = setups
        self._cfg_global = config_global
        self._pct_oos = pct_oos
        self._wfa = wfa_activo
        self._wfa_n = wfa_ventanas
        self._codigo = codigo
        self._tf_label = tf_label
        self._regla_resample = regla_resample
        self._excluir_interpoladas = bool(excluir_interpoladas)

    def run(self):
        try:
            self.progreso.emit(5, 100)
            df = _cargar_ohlc(self._csv, self._regla_resample)
            n = len(df)
            if n < 50:
                self.computed.emit({'error': f'Serie demasiado corta ({n} velas)'})
                return
            if not self._setups:
                self.computed.emit({'error': 'El sistema no tiene ningún setup'})
                return
            self.progreso.emit(20, 100)

            df_eventos = None
            eventos_prep = None
            avisos_noticias = []
            setups_senales = self._setups
            if _algun_setup_usa_noticias(self._setups):
                try:
                    self.estado.emit("Descargando calendario económico (TradingEconomics)…")
                    df_eventos = _cargar_eventos_noticias(
                        df, self._csv,
                        progress_callback=lambda i, total: self.estado.emit(
                            f"Descargando calendario económico… ({i}/{total} meses)"))
                except Exception as e:
                    # Best-effort: el backtest corre igual; se avisa y se omite
                    # el filtro en vez de abortar toda la corrida.
                    avisos_noticias.append(f"Filtro de noticias NO aplicado: {e}")
                    df_eventos = None
                if df_eventos is not None and len(df_eventos) > 0:
                    eventos_prep = preparar_eventos_noticias(df_eventos)
                    setups_senales = _resolver_monedas_noticias(self._setups, self._csv)

            mascara_interpoladas = None
            if self._excluir_interpoladas:
                mascara_interpoladas = df['interpolado'].to_numpy(dtype=bool)
            senales = generar_senales_sistema(
                df, setups_senales, eventos_prep,
                mascara_entradas=mascara_interpoladas)
            self.progreso.emit(40, 100)

            # config del motor: cuenta global + riesgo/stop/TP/tiempo por setup
            config = dict(self._cfg_global)
            config['excluir_interpoladas'] = self._excluir_interpoladas
            config_por_setup = {}
            for k, setup in enumerate(self._setups):
                salida_n = int(setup.get('salida_n_velas', 0))
                if setup['plantilla'] == 'Patrones de velas' and salida_n == 0:
                    salida_n = int(setup.get('params', {}).get('lag_salida', 5))
                config_por_setup[k] = {
                    'riesgo_pct': float(setup.get('riesgo_pct', 0.01)),
                    'stop_atr': float(setup.get('stop_atr', 0.0)),
                    'stop_atr_modo': setup.get('stop_atr_modo', 'fijo'),
                    'tp_r': float(setup.get('tp_r', 0.0)),
                    'salida_n_velas': salida_n,
                    'be_atr': float(setup.get('be_atr', 0.0)),
                    'be_unidad': setup.get('be_unidad', 'atr'),
                    'trailing_atr': float(setup.get('trailing_atr', 0.0)),
                    'parciales': setup.get('parciales', []),
                    'tramos': setup.get('tramos', []),
                    'limite_vigencia_velas': int(setup.get('limite_vigencia_velas', 0)),
                    'limite_cancelar_avance_r': float(
                        setup.get('limite_cancelar_avance_r', 0.0)),
                }
                for clave_mec in MECANISMOS_SALIDA:
                    if setup.get(clave_mec):
                        config_por_setup[k][clave_mec] = setup[clave_mec]
            config['config_por_setup'] = config_por_setup

            # pre-computar máscaras de condiciones (salidas parciales y
            # entrada escalonada comparten el mismo patrón: una máscara por
            # etapa/tramo, True en toda la serie si no tiene condiciones)
            def _masks_por_etapas(clave_lista):
                masks_long, masks_short = [], []
                for setup in self._setups:
                    ml, ms = [], []
                    for etapa in setup.get(clave_lista, []):
                        conds = etapa.get('condiciones')
                        if conds and len(conds) > 0:
                            m_l, m_s = _mascaras_condiciones_dir(df, conds)
                            ml.append(np.asarray(m_l, dtype=bool))
                            ms.append(np.asarray(m_s, dtype=bool))
                        else:
                            ml.append(np.ones(n, dtype=bool))
                            ms.append(np.ones(n, dtype=bool))
                    masks_long.append(ml if ml else None)
                    masks_short.append(ms if ms else None)
                return masks_long, masks_short

            config['parciales_masks_long'], config['parciales_masks_short'] = \
                _masks_por_etapas('parciales')
            config['tramos_masks_long'], config['tramos_masks_short'] = \
                _masks_por_etapas('tramos')

            # los 4 mecanismos globales son dicts sueltos, no una lista: se
            # ordenan según MECANISMOS_SALIDA para casar con los índices que
            # espera el motor
            mec_long, mec_short = [], []
            for setup in self._setups:
                ml, ms = [], []
                for clave_mec in MECANISMOS_SALIDA:
                    conds = (setup.get(clave_mec) or {}).get('condiciones')
                    if conds and len(conds) > 0:
                        m_l, m_s = _mascaras_condiciones_dir(df, conds)
                        ml.append(np.asarray(m_l, dtype=bool))
                        ms.append(np.asarray(m_s, dtype=bool))
                    else:
                        ml.append(np.ones(n, dtype=bool))
                        ms.append(np.ones(n, dtype=bool))
                mec_long.append(ml)
                mec_short.append(ms)
            config['mecanismos_masks_long'] = mec_long
            config['mecanismos_masks_short'] = mec_short

            resultado = simular(df['open'].values, df['high'].values,
                                df['low'].values, df['close'].values,
                                senales, config)
            self.progreso.emit(60, 100)

            velas_anio = _velas_por_anio(df, self._csv)
            velas_dia = _velas_por_dia(df, self._csv)

            corte = dividir_is_oos(n, self._pct_oos)
            metricas = {
                'IS': calcular_metricas(resultado, 0, corte, velas_anio, velas_dia),
                'OOS': calcular_metricas(resultado, corte, n, velas_anio, velas_dia),
                'Total': calcular_metricas(resultado, 0, n, velas_anio, velas_dia),
            }

            # métricas por setup (valida si cada forma de entrada aporta)
            tr = resultado['trades']
            metricas_setup = []
            for k, setup in enumerate(self._setups):
                m = tr['setup'] == k
                pnl = tr['pnl'][m]
                r_setup = tr['r_multiple'][m]
                metricas_setup.append({
                    'nombre': setup.get('nombre') or setup['plantilla'],
                    'riesgo_pct': float(setup.get('riesgo_pct', 0.01)),
                    'n_trades': int(m.sum()),
                    'win_rate': float((pnl > 0).mean()) if len(pnl) else None,
                    'pnl_total': float(pnl.sum()),
                    'expectancy_pct': float(r_setup.mean()) if len(r_setup) else None,
                })

            wfa = None
            if self._wfa and self._wfa_n >= 2:
                bordes = np.linspace(0, n, self._wfa_n + 1).astype(int)
                ventanas = []
                for k in range(self._wfa_n):
                    met = calcular_metricas(resultado, bordes[k], bordes[k + 1],
                                            velas_anio, velas_dia)
                    met['idx_ini'] = int(bordes[k])
                    met['idx_fin'] = int(bordes[k + 1])
                    ventanas.append(met)
                wfa = ventanas

            mc = montecarlo(resultado['trades'],
                            config.get('capital_inicial', 10000.0),
                            n_sims=1000, semilla=1234)
            self.progreso.emit(85, 100)

            c = df['close'].values
            self.progreso.emit(100, 100)
            self.computed.emit({
                'timestamps': df['timestamp'].values,
                'open': df['open'].values,
                'high': df['high'].values,
                'low': df['low'].values,
                'close': c,
                'volume': df['volume'].values if 'volume' in df.columns
                          else np.zeros(n),
                'codigo': self._codigo,
                'log_ret_acum': np.log(c / c[0]),
                'resultado': resultado,
                'metricas': metricas,
                'metricas_setup': metricas_setup,
                'nombres_setup': [s.get('nombre') or s['plantilla']
                                  for s in self._setups],
                'corte': corte,
                'n_velas': n,
                'velas_anio': velas_anio,
                'velas_dia': velas_dia,
                'wfa': wfa,
                'montecarlo': mc,
                'estrategia': ' + '.join(s.get('nombre') or s['plantilla']
                                         for s in self._setups),
                'csv': self._csv,
                'config': config,
                'setups': self._setups,
                'tf': self._tf_label,
                'eventos_noticias': df_eventos,
                # filtros pedidos que no se pudieron aplicar (ver
                # _mascara_filtros_setup): sin esto, un backtest sin filtrar
                # es indistinguible de uno filtrado
                'avisos': (senales.get('avisos') or []) + avisos_noticias,
            })
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.computed.emit({'error': str(e)})


# ══════════════ hilo de optimización (barrido restringido a IS) ══════════════
class _OptimizerThread(QThread):
    """Barrido de parámetros de UN setup — ver core.optimizer.optimizar_setup:
    cada combinación se simula únicamente sobre el tramo IS. Misma carga/
    resample de CSV que _BacktestThread (mismo archivo, misma temporalidad)."""
    progreso = pyqtSignal(int, int)
    estado = pyqtSignal(str)
    terminado = pyqtSignal(object)

    def __init__(self, csv_path, setup_base, sweep_params, sweep_riesgo,
                 config_global, pct_oos, metrica, tf_label=None,
                 regla_resample=None, excluir_interpoladas=False, parent=None):
        super().__init__(parent)
        self._csv = csv_path
        self._setup_base = setup_base
        self._sweep_params = sweep_params
        self._sweep_riesgo = sweep_riesgo
        self._cfg_global = config_global
        self._pct_oos = pct_oos
        self._metrica = metrica
        self._tf_label = tf_label
        self._regla_resample = regla_resample
        self._excluir_interpoladas = bool(excluir_interpoladas)

    def run(self):
        try:
            # Progreso 0/N ya durante la carga del CSV (antes del bucle),
            # para que la UI no quede en "cargando" sin barra.
            sweep_prev = dict(self._sweep_params)
            sweep_prev.update(
                {PREFIJO_RIESGO + k: v for k, v in self._sweep_riesgo.items()})
            total_prev = n_combinaciones(sweep_prev) if sweep_prev else 1
            self.progreso.emit(0, total_prev)
            df = _cargar_ohlc(self._csv, self._regla_resample)
            n = len(df)
            if n < 50:
                self.terminado.emit({'error': f'Serie demasiado corta ({n} velas)'})
                return
            velas_anio = _velas_por_anio(df, self._csv)
            setup_base = self._setup_base
            eventos_prep = None
            mascara_interpoladas = None
            if self._excluir_interpoladas:
                mascara_interpoladas = df['interpolado'].to_numpy(dtype=bool)
            avisos_noticias = []
            if _algun_setup_usa_noticias([setup_base]):
                try:
                    self.estado.emit("Descargando calendario económico (TradingEconomics)…")
                    df_eventos = _cargar_eventos_noticias(
                        df, self._csv,
                        progress_callback=lambda i, total: self.estado.emit(
                            f"Descargando calendario económico… ({i}/{total} meses)"))
                except Exception as e:
                    # Best-effort: la optimización corre igual; se avisa y se
                    # omite el filtro en vez de abortar todo el barrido.
                    avisos_noticias.append(f"Filtro de noticias NO aplicado: {e}")
                    df_eventos = None
                if df_eventos is not None and len(df_eventos) > 0:
                    eventos_prep = preparar_eventos_noticias(df_eventos)
                    setup_base = _resolver_monedas_noticias([setup_base], self._csv)[0]
            resultados = optimizar_setup(
                df, setup_base, self._sweep_params, self._sweep_riesgo,
                self._cfg_global, self._pct_oos, metrica=self._metrica,
                velas_por_anio=velas_anio,
                progreso_cb=lambda i, total: self.progreso.emit(i, total),
                eventos_noticias=eventos_prep,
                mascara_entradas=mascara_interpoladas)
            self.terminado.emit({
                'resultados': resultados,
                'metrica': self._metrica,
                'setup_base': self._setup_base,
                'csv': self._csv,
                'tf': self._tf_label,
                'avisos': avisos_noticias,
            })
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.terminado.emit({'error': str(e)})


# ══════════════ ayuda contextual (icono "?" por sección) ══════════════
def _icono_ayuda(tooltip):
    """Badge «?» de cada título de sección de Constructor y Resultados.

    Responde a las DOS cosas: tooltip al pasar el ratón y popup con el mismo
    texto al hacer clic. Antes solo llevaba tooltip, y como es idéntico a los
    iconos de Análisis y de los gráficos —que sí abren un panel al clicarlos—,
    clicar uno del Backtester no hacía nada y se leía como un icono roto."""
    return _icono_ayuda_texto(tooltip)


def _fila_ayuda(tooltip):
    """Fila nueva (icono alineado a la derecha) para insertar como primer
    elemento de un QGroupBox con título nativo, sin tocar ese título."""
    fila = QHBoxLayout()
    fila.addStretch()
    fila.addWidget(_icono_ayuda(tooltip))
    return fila


def _fila_ayuda_popup(logica, significado, uso, resultados):
    """Como _fila_ayuda, pero con el icono_ayuda de plot_common (popup con
    4 pestañas) — para las secciones de gráficos de Resultados."""
    fila = QHBoxLayout()
    fila.addStretch()
    fila.addWidget(_icono_ayuda_popup(logica, significado, uso, resultados))
    return fila


def _fila_con_ayuda(form, texto, campo, ayuda, tooltip_campo=True):
    """addRow poniendo el bocadillo TAMBIÉN en la etiqueta de la fila.

    `form.addRow("texto", campo)` fabrica la QLabel por dentro y no la
    devuelve, así que el nombre del campo —justo donde el usuario lleva el
    ratón cuando no sabe qué es algo— se quedaba mudo por muy buen tooltip que
    tuviera el control.

    `tooltip_campo=False` para las filas cuyo control ya explica otra cosa más
    concreta que no conviene pisar (el combo de unidad del Break Even)."""
    etiqueta = QLabel(texto)
    etiqueta.setToolTip(ayuda)
    if tooltip_campo and isinstance(campo, QWidget):
        campo.setToolTip(ayuda)
    form.addRow(etiqueta, campo)
    return etiqueta


def _insertar_ayuda_form(form_layout, tooltip):
    """Variante de _fila_ayuda para un QGroupBox cuyo layout es QFormLayout
    (no admite insertLayout): envuelve la fila en un QWidget y la inserta
    como primera fila de ancho completo."""
    cont = QWidget()
    cont.setLayout(_fila_ayuda(tooltip))
    form_layout.insertRow(0, cont)


# ══════════════ resaltado de fila completa en tablas de etapas/condiciones ══════════════
# Qt solo resalta nativamente la celda bajo el ratón cuando es un
# QTableWidgetItem real; las columnas con un widget embebido (setCellWidget:
# combos, spins, botones) no participan en la selección nativa — un clic ahí
# no llega a informar a la tabla de qué fila se tocó. _seleccionar_fila_al_clic
# tapa ese hueco a mano, y _OverlayFilaSeleccionada dibuja el borde resultante.
_COLOR_BORDE_FILA = '#4fc3f7'

# Fade del combo de sesión al deshabilitarlo (TF diario): mismos tiempos que
# el fade del arranque de la app (app.py: 300ms in / 400ms out).
_FADE_IN_MS = 300
_FADE_OUT_MS = 400


def _seleccionar_fila_al_clic(widget, tabla, fila):
    """Envuelve el mousePressEvent de `widget` (una celda embebida de
    `tabla`, fila `fila`) para que, antes de su comportamiento normal (abrir
    el desplegable, etc.), seleccione esa fila en la tabla — así un clic en
    CUALQUIER columna resalta la fila entera, no solo la celda con un
    QTableWidgetItem plano."""
    original = widget.mousePressEvent
    def _clic(event):
        tabla.selectRow(fila)
        original(event)
    widget.mousePressEvent = _clic
    return widget


class _OverlayFilaSeleccionada(QWidget):
    """Dibuja UN único rectángulo alrededor de la fila seleccionada, de la
    primera columna a la última.

    Va como hijo del viewport, cubriéndolo entero y transparente al ratón, en
    vez de estilar celda a celda: los widgets embebidos son opacos y taparían
    los tramos de borde que caen bajo ellos, dejando un contorno partido y
    visible solo en algunas columnas."""

    def __init__(self, tabla):
        super().__init__(tabla.viewport())
        self.tabla = tabla
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setGeometry(tabla.viewport().rect())
        tabla.viewport().installEventFilter(self)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.Resize:
            self.setGeometry(obj.rect())
        return False

    def paintEvent(self, _evento):
        fila = self.tabla.currentRow()
        if fila < 0 or not self.tabla.selectionModel().selectedIndexes():
            return
        modelo = self.tabla.model()
        rect = self.tabla.visualRect(modelo.index(fila, 0)).united(
            self.tabla.visualRect(modelo.index(fila, self.tabla.columnCount() - 1)))
        # las columnas pueden sumar algo más que el viewport (redondeo del
        # reparto de anchos): sin recortar, el lado derecho caería fuera
        rect = rect.intersected(self.rect())
        if rect.isEmpty():
            return
        pintor = QPainter(self)
        pintor.setRenderHint(QPainter.RenderHint.Antialiasing)
        pintor.setPen(QPen(QColor(_COLOR_BORDE_FILA), 1))
        # medio píxel adentro: un trazo de 1px sobre coordenadas enteras cae
        # justo en la frontera entre dos filas de píxeles y el antialiasing lo
        # reparte a medias entre ambas, dejándolo descolorido
        pintor.drawRoundedRect(QRectF(rect).adjusted(0.5, 0.5, -0.5, -0.5), 3, 3)


def _repintar_seleccion_fila(tabla):
    """Redibuja el contorno de la fila seleccionada — conectado a
    tabla.itemSelectionChanged. El overlay se crea la primera vez y se sube
    por encima de las celdas en cada repintado: los widgets embebidos que
    añade cada recarga de la tabla entran al viewport por encima de él."""
    overlay = getattr(tabla, '_overlay_fila', None)
    if overlay is None:
        overlay = _OverlayFilaSeleccionada(tabla)
        tabla._overlay_fila = overlay
        overlay.show()
    overlay.raise_()
    overlay.update()


class _SinPintadoSeleccion(QStyledItemDelegate):
    """Pinta las celdas como si nunca estuvieran seleccionadas, dejando el
    contorno del overlay como único indicador de fila activa.

    El resaltado nativo solo alcanza a las celdas con item plano —en estas
    tablas, la del %—, que es precisamente el resaltado a medias que el
    contorno viene a sustituir; además reemplaza el color de texto propio del
    item (el de las filas de stop/TP/BE/trailing) por el de selección."""

    def initStyleOption(self, option, index):
        super().initStyleOption(option, index)
        option.state &= ~QStyle.StateFlag.State_Selected


def _activar_borde_fila(tabla):
    """Selección por fila señalada únicamente con el contorno del overlay.

    Hacen falta las dos piezas: la vista rellena la celda seleccionada con el
    brush Highlight de su paleta sin pasar por el delegate, y el delegate es
    quien decide el color del texto."""
    tabla.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
    paleta = tabla.palette()
    paleta.setColor(QPalette.ColorRole.Highlight, QColor(0, 0, 0, 0))
    tabla.setPalette(paleta)
    tabla.setItemDelegate(_SinPintadoSeleccion(tabla))
    tabla.itemSelectionChanged.connect(lambda: _repintar_seleccion_fila(tabla))


class _BorrarFilaPorTeclado(QObject):
    """EventFilter que borra la fila activa de una tabla al pulsar Supr o
    Retroceso, PERO solo cuando el foco está en la propia tabla (no en un
    widget embebido de celda).

    Las celdas de estas tablas llevan QComboBox/QSpinBox embebidos: si el
    atajo se capturara con QShortcut(WidgetWithChildrenShortcut) robaría la
    tecla también cuando el usuario está editando un spin — donde Retroceso
    debe borrar el último dígito, no la fila. Al filtrar en la tabla (que
    solo recibe el keyPress cuando ella misma tiene el foco), el combo/spin
    conserva su comportamiento normal de edición."""

    def __init__(self, tabla, accion, parent=None):
        super().__init__(parent)
        self._accion = accion
        tabla.installEventFilter(self)

    def eventFilter(self, obj, evento):
        if (evento.type() == QEvent.Type.KeyPress
                and evento.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace)):
            foco = obj.focusWidget()
            # Solo borra la fila cuando el foco está en la tabla o su
            # viewport (celda plana seleccionada). Si el foco está en un
            # widget embebido editable (combo/spin) o en el editor inline de
            # una celda, se deja pasar la tecla para no romper la edición.
            if foco is obj or foco is obj.viewport():
                self._accion()
                return True
        return super().eventFilter(obj, evento)


def _instalar_borrado_fila(tabla, accion):
    _BorrarFilaPorTeclado(tabla, accion)


# ══════════════ editor de reglas custom ══════════════
def _combo_regla(opciones, actual=None):
    cb = QComboBox()
    cb.addItems(opciones)
    if actual in opciones:
        cb.setCurrentText(actual)
    return cb


def _spin_regla(valor, minimo=1, maximo=100000, dec=None):
    if dec is None:
        sp = QSpinBox()
        sp.setRange(minimo, maximo)
        sp.setValue(int(valor))
    else:
        sp = QDoubleSpinBox()
        sp.setRange(-1e9, 1e9)
        sp.setDecimals(dec)
        sp.setValue(float(valor))
    return sp


# precios en bruto: su «Periodo» no significa nada (el motor lo ignora — ver
# _spec_regla/core.strategies._serie_indicador). Dejarlo activo confundía:
# parecía que el número importaba cuando no.
_PRECIOS_SIN_PERIODO = {'close', 'open', 'high', 'low'}

# series VWAP con anclaje/desviación/modo seleccionables en las condiciones
_TIPOS_VWAP_REGLA = ('VWAP_MEDIA', 'VWAP_SD', 'VWAP_SUP', 'VWAP_INF')
_MAPA_ANCLAJE = {'Sesión': 'D', 'Semana': 'W', 'Mes': 'M', 'Trimestre': 'T',
                 'Año': 'A', 'Década': '10Y', 'Siglo': '100Y'}
_MAPA_ANCLAJE_INV = {v: k for k, v in _MAPA_ANCLAJE.items()}
_MAPA_MODO_VWAP = {'SD': 'sd', '%': 'pct'}
_MAPA_MODO_VWAP_INV = {v: k for k, v in _MAPA_MODO_VWAP.items()}
_K_VWAP = ['0.5', '1', '2', '3']


def _es_vwap_regla(tipo):
    return tipo in _TIPOS_VWAP_REGLA


def _sincronizar_vwap(cmb_izq, cmb_der, ancla, k, modo):
    """Habilita/deshabilita las columnas Anclaje/Desviación/Modo según si el
    indicador de la fila (izq o der) es una serie VWAP configurable."""
    def _sync(_texto=None):
        activo = _es_vwap_regla(cmb_izq.currentText()) or \
            _es_vwap_regla(cmb_der.currentText())
        for w in (ancla, k, modo):
            w.setEnabled(activo)
            w.setToolTip('' if activo else
                         "Solo aplica a las series VWAP (anclaje y banda).")
    cmb_izq.currentTextChanged.connect(_sync)
    cmb_der.currentTextChanged.connect(_sync)
    _sync()


def _sincronizar_periodo(cmb, spin):
    """Liga un combo de indicador con su spin de periodo: al elegir un precio
    en bruto (close/open/high/low) el spin se deshabilita y lo explica; al
    volver a un indicador con ventana (SMA, RSI...) se reactiva."""
    def _sync(texto):
        if texto in _PRECIOS_SIN_PERIODO:
            spin.setEnabled(False)
            spin.setToolTip("Precio en bruto: sin periodo.")
        else:
            spin.setEnabled(True)
            spin.setToolTip(
                "Valor numérico de la comparación." if texto == 'Valor'
                else "Ventana (lookback) del indicador.")
    cmb.currentTextChanged.connect(_sync)
    _sync(cmb.currentText())


def _spec_regla(tipo, num, anclaje=None, k=None, modo=None):
    if tipo == 'Valor':
        return {'tipo': 'valor', 'valor': num}
    if tipo in ('close', 'open', 'high', 'low'):
        return {'tipo': tipo}
    spec = {'tipo': tipo, 'periodo': int(num)}
    # las series VWAP llevan anclaje + desviación + modo (seleccionables en
    # las columnas de la tabla de condiciones)
    if tipo in _TIPOS_VWAP_REGLA:
        spec['anclaje'] = anclaje or 'W'
        spec['k'] = float(k or 1.0)
        spec['modo'] = modo or 'sd'
    return spec


def _texto_spec_regla(spec):
    """Convierte un spec de regla ({'tipo':..., 'periodo':...} o
    {'tipo':'valor','valor':...}) a texto compacto: 'close', 'SMA(200)',
    '30', 'RSI(14)'..."""
    if not isinstance(spec, dict):
        return '—'
    t = spec.get('tipo', '')
    if t == 'valor':
        v = spec.get('valor', 0.0)
        return f'{v:g}' if float(v) == int(float(v)) else f'{v:g}'
    if t in ('close', 'open', 'high', 'low'):
        return t
    per = spec.get('periodo')
    if per:
        return f'{t}({per:g})'
    return t


def _texto_condicion(cond):
    """Una condición a texto legible: 'close > SMA(200)', 'RSI(14) < 30'..."""
    izq = _texto_spec_regla(cond.get('izq', {}))
    op = cond.get('op', '>')
    der = _texto_spec_regla(cond.get('der', {}))
    txt = f'{izq} {op} {der}'
    direccion = cond.get('direccion', 'ambas')
    if direccion == 'long':
        txt += ' · L'
    elif direccion == 'short':
        txt += ' · S'
    return txt


def _describir_condiciones(conds):
    """Texto resumido para el botón «Condiciones» de una fila: las dos
    primeras condiciones separadas por ' · ' y, si hay más, un '+N'. Vacío
    devuelve el placeholder '+ Cond' (se ve más amigable que '0 cond')."""
    conds = conds or []
    if not conds:
        return '+ Cond'
    textos = [_texto_condicion(c) for c in conds]
    resumen = ' · '.join(textos[:2])
    if len(textos) > 2:
        resumen += f' · +{len(textos) - 2}'
    return resumen


def _tooltip_condiciones(conds):
    """Tooltip con la lista COMPLETA de condiciones (una por línea), para
    cuando el resumen del botón se corta por el ancho de la celda."""
    conds = conds or []
    if not conds:
        return 'Añadir condiciones (todas AND)'
    lineas = [f'• {_texto_condicion(c)}' for c in conds]
    return 'Condiciones (todas AND):\n' + '\n'.join(lineas)


def _abrir_guia_calendario(parent):
    """Guía paso a paso para configurar la API key de TradingEconomics,
    necesaria para el filtro «Evitar noticias» (Constructor) y las líneas de
    eventos en Resultados. Devuelve True si al cerrar el diálogo ya hay una
    key configurada (el llamador puede entonces activar el checkbox)."""
    from gui.dialogs.settings_dialog import SettingsDialog

    dlg, contenido, _ = crear_dialogo(
        "Noticias económicas (TradingEconomics)", parent,
        subtitulo='API key gratuita (~100 peticiones al mes)', ancho=480)

    cuerpo = QLabel(
        "El filtro «Evitar noticias» y las líneas de eventos en Resultados "
        "necesitan datos del calendario económico de TradingEconomics, que "
        "requiere una API key gratuita (~100 peticiones al mes).")
    cuerpo.setWordWrap(True)
    contenido.addWidget(cuerpo)

    # Paso 1
    paso1 = QLabel("Paso 1 — Consigue la API key")
    paso1.setStyleSheet("color: #7aaccc; font-size: 12px; font-weight: bold;")
    contenido.addWidget(paso1)
    t1 = QLabel("Regístrate gratis en tradingeconomics.com; en «My Account » "
                "→ API tendrás tu key (el plan gratuito incluye el "
                "calendario). Sin tarjeta de crédito.")
    t1.setWordWrap(True)
    contenido.addWidget(t1)
    btn_web = QPushButton("Abrir tradingeconomics.com")
    btn_web.clicked.connect(
        lambda: QDesktopServices.openUrl(QUrl("https://tradingeconomics.com/api")))
    contenido.addWidget(btn_web)

    # Paso 2
    paso2 = QLabel("Paso 2 — Conéctala en la plataforma")
    paso2.setStyleSheet("color: #7aaccc; font-size: 12px; font-weight: bold;")
    contenido.addWidget(paso2)
    t2 = QLabel("En Ajustes → Calendario económico, pega tu key y pulsa "
                "«Probar conexión». Al cerrar Ajustes, esta opción quedará "
                "activa.")
    t2.setWordWrap(True)
    contenido.addWidget(t2)
    btn_ajustes = QPushButton("Abrir Ajustes")
    btn_ajustes.clicked.connect(lambda: SettingsDialog(parent).exec())
    contenido.addWidget(btn_ajustes)

    # Paso 3
    paso3 = QLabel("Paso 3 — Listo")
    paso3.setStyleSheet("color: #7aaccc; font-size: 12px; font-weight: bold;")
    contenido.addWidget(paso3)
    t3 = QLabel("Vuelve a marcar «Evitar noticias» (o «Mostrar noticias» en "
                "Resultados). Se descargarán los eventos del rango del backtest.")
    t3.setWordWrap(True)
    contenido.addWidget(t3)

    btn_cerrar = QPushButton("Cerrar")
    btn_cerrar.setObjectName("accionOk")
    btn_cerrar.clicked.connect(dlg.accept)
    contenido.addWidget(btn_cerrar, alignment=Qt.AlignmentFlag.AlignRight)

    dlg.exec()
    return bool(get_te_api_key())


def _acumular_indicador_spec(spec, mas, rsis, atrs, bbs, donchianes=None,
                             zigzags=None, macds=None, adxs=None,
                             supertrends=None, aroones=None, cmos=None,
                             trixes=None, stochrsis=None, ichimokus=None,
                             keltners=None, ttms=None, vwaps=None):
    """Clasifica un spec de indicador ({'tipo':'EMA','periodo':200}, etc.) en
    los sets que consumen ResultadosWidget._dibujar_principal/_dibujar_indicadores.
    Ignora specs sin indicador real (close/open/high/low/valor)."""
    t = spec.get('tipo', '')
    per = int(spec.get('periodo', 14))
    if t in ('SMA', 'EMA'):
        mas.add((t, per))
    elif t == 'RSI':
        rsis.add(per)
    elif t == 'ATR':
        atrs.add(per)
    elif t in ('BB_sup', 'BB_inf', 'BB_media'):
        bbs.add((per, float(spec.get('desv', 2.0))))
    elif t in ('DONCHIAN_SUP', 'DONCHIAN_INF') and donchianes is not None:
        donchianes.add(per)
    elif t == 'ZIGZAG' and zigzags is not None:
        # desde una regla la desviación es fija y el «Periodo» son las piernas
        zigzags.add((_ZIGZAG_DESVIACION_REGLA, per))
    elif t == 'SUPERTREND' and supertrends is not None:
        # en el editor el multiplicador es fijo (×3); la plantilla lo varía
        supertrends.add((per, 3.0))
    elif t in ('MACD_LINEA', 'MACD_SENAL', 'MACD_HIST') and macds is not None:
        # parámetros fijos 12/26/9 en el editor; la plantilla los varía
        macds.add((12, 26, 9))
    elif t in ('ADX', 'DI_PLUS', 'DI_MINUS') and adxs is not None:
        # el umbral de fuerza no viaja en la regla: se dibuja con el de 20
        adxs.add((per, 20.0))
    elif t in ('AROON_UP', 'AROON_DN') and aroones is not None:
        aroones.add(per)
    elif t == 'CMO' and cmos is not None:
        cmos.add(per)
    elif t == 'TRIX' and trixes is not None:
        trixes.add(per)
    elif t == 'STOCHRSI' and stochrsis is not None:
        stochrsis.add(per)
    elif t in ('ICHIMOKU_TENKAN', 'ICHIMOKU_KIJUN', 'ICHIMOKU_SENKOU_A',
               'ICHIMOKU_SENKOU_B', 'ICHIMOKU_CHIKOU') and ichimokus is not None:
        # parámetros fijos 9/26/52 en el editor; la plantilla los varía
        ichimokus.add((9, 26, 52))
    elif t in ('KELTNER_SUP', 'KELTNER_INF', 'KELTNER_MEDIA') and keltners is not None:
        keltners.add((per, 2.0))
    elif t in ('TTM_SQUEEZE', 'TTM_MOMENTUM') and ttms is not None:
        ttms.add((per, 2.0, 1.5))
    elif t in _TIPOS_VWAP_REGLA and vwaps is not None:
        vwaps.add((spec.get('anclaje', 'W'), float(spec.get('k', 1.0)),
                   spec.get('modo', 'sd')))


class EditorReglas(QGroupBox):
    """Tabla de condiciones: cada fila es una condición; las filas con la
    misma (regla, setup) se combinan con AND; setups distintos con OR."""
    _REGLAS = ['Entrada Long', 'Salida Long', 'Entrada Short', 'Salida Short']
    _CLAVES = {'Entrada Long': 'entradas_long', 'Salida Long': 'salidas_long',
               'Entrada Short': 'entradas_short', 'Salida Short': 'salidas_short'}

    def __init__(self, parent=None):
        super().__init__("Reglas custom (mismo setup = AND, setups distintos = OR)", parent)
        lay = QVBoxLayout(self)
        lay.insertLayout(0, _fila_ayuda(
            "Solo para la plantilla «Custom (reglas)»: construye las "
            "condiciones de entrada/salida a mano combinando indicadores. "
            "Las condiciones del mismo bloque se exigen todas a la vez "
            "(AND); bloques distintos son alternativas entre sí (OR)."))
        self.tabla = QTableWidget(0, 10)
        self.tabla.setHorizontalHeaderLabels(
            ['Regla', 'Setup', 'Indicador', 'Periodo', 'Operador',
             'Comparar con', 'Valor/Periodo', 'Anclaje', 'Desviación', 'Modo'])
        self.tabla.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.tabla.verticalHeader().setVisible(False)
        self.tabla.setMinimumHeight(120)
        lay.addWidget(self.tabla)
        fila = QHBoxLayout()
        btn_add = QPushButton("+ Condición")
        btn_add.clicked.connect(self._add_fila)
        btn_del = QPushButton("− Quitar")
        btn_del.clicked.connect(self._del_fila)
        fila.addWidget(btn_add)
        fila.addWidget(btn_del)
        fila.addStretch()
        lay.addLayout(fila)

    def _add_fila(self, _=False, datos=None):
        r = self.tabla.rowCount()
        self.tabla.insertRow(r)
        d = datos or {}
        self.tabla.setCellWidget(r, 0, _combo_regla(self._REGLAS, d.get('regla')))
        self.tabla.setCellWidget(r, 1, _spin_regla(d.get('setup', 0), 0, 9))
        cmb_izq = _combo_regla(_INDICADORES_REGLA, d.get('izq', 'close'))
        self.tabla.setCellWidget(r, 2, cmb_izq)
        sp_izq = _spin_regla(d.get('izq_periodo', 14), 1, 5000)
        self.tabla.setCellWidget(r, 3, sp_izq)
        self.tabla.setCellWidget(r, 4, _combo_regla(_OPERADORES_REGLA, d.get('op', '>')))
        cmb_der = _combo_regla(['Valor'] + _INDICADORES_REGLA,
                               d.get('der', 'Valor'))
        self.tabla.setCellWidget(r, 5, cmb_der)
        sp_der = _spin_regla(d.get('der_valor', 0.0), dec=4)
        self.tabla.setCellWidget(r, 6, sp_der)
        _sincronizar_periodo(cmb_izq, sp_izq)
        _sincronizar_periodo(cmb_der, sp_der)
        # columnas VWAP (anclaje / desviación / modo) — solo activas si la
        # fila usa una serie VWAP configurable
        cmb_ancla = _combo_regla(
            list(_MAPA_ANCLAJE), _MAPA_ANCLAJE_INV.get(d.get('anclaje', 'W'), 'Semana'))
        cmb_ancla.setToolTip("Periodo de referencia del VWAP: cuándo se "
                             "reinicia el acumulado (sesión/semana/mes/…).")
        cmb_k = _combo_regla(_K_VWAP, '{:g}'.format(float(d.get('k', 1.0))))
        cmb_k.setToolTip("Multiplicador de la banda: VWAP ± k·σ (o k% en modo "
                         "porcentaje).")
        cmb_modo = _combo_regla(list(_MAPA_MODO_VWAP),
                                _MAPA_MODO_VWAP_INV.get(d.get('modo', 'sd'), 'SD'))
        cmb_modo.setToolTip("Modo de la banda: desviación estándar del anclaje "
                            "o porcentaje del propio VWAP.")
        _sincronizar_vwap(cmb_izq, cmb_der, cmb_ancla, cmb_k, cmb_modo)
        self.tabla.setCellWidget(r, 7, cmb_ancla)
        self.tabla.setCellWidget(r, 8, cmb_k)
        self.tabla.setCellWidget(r, 9, cmb_modo)

    def _del_fila(self):
        r = self.tabla.currentRow()
        if r >= 0:
            self.tabla.removeRow(r)
        elif self.tabla.rowCount():
            self.tabla.removeRow(self.tabla.rowCount() - 1)

    def reglas(self):
        """Serializa la tabla a la estructura de _gen_custom."""
        grupos = {}   # (clave, setup) -> lista de condiciones
        for r in range(self.tabla.rowCount()):
            regla = self.tabla.cellWidget(r, 0).currentText()
            setup = self.tabla.cellWidget(r, 1).value()
            izq = self.tabla.cellWidget(r, 2).currentText()
            izq_per = self.tabla.cellWidget(r, 3).value()
            op = self.tabla.cellWidget(r, 4).currentText()
            der = self.tabla.cellWidget(r, 5).currentText()
            der_val = self.tabla.cellWidget(r, 6).value()
            anclaje = _MAPA_ANCLAJE[self.tabla.cellWidget(r, 7).currentText()]
            k = float(self.tabla.cellWidget(r, 8).currentText())
            modo = _MAPA_MODO_VWAP[self.tabla.cellWidget(r, 9).currentText()]
            cond = {'izq': _spec_regla(izq, izq_per, anclaje, k, modo),
                    'op': op,
                    'der': _spec_regla(der, der_val, anclaje, k, modo)}
            grupos.setdefault((self._CLAVES[regla], setup), []).append(cond)
        out = {'entradas_long': [], 'salidas_long': [],
               'entradas_short': [], 'salidas_short': []}
        for (clave, setup), condiciones in grupos.items():
            out[clave].append({'setup_id': setup, 'condiciones': condiciones})
        return out

    def cargar_reglas(self, reglas):
        self.tabla.setRowCount(0)
        inversa = {v: k for k, v in self._CLAVES.items()}
        n = 0
        for clave, setups in (reglas or {}).items():
            for setup in setups:
                for cond in setup.get('condiciones', []):
                    n += 1
                    if n % 5 == 0:
                        bombear_eventos()
                    izq, der = cond['izq'], cond['der']
                    self._add_fila(datos={
                        'regla': inversa.get(clave, 'Entrada Long'),
                        'setup': setup.get('setup_id', 0),
                        'izq': izq['tipo'], 'izq_periodo': izq.get('periodo', 14),
                        'op': cond['op'],
                        'der': 'Valor' if der['tipo'] == 'valor' else der['tipo'],
                        'der_valor': der.get('valor', der.get('periodo', 0.0)),
                        'anclaje': izq.get('anclaje') or der.get('anclaje') or 'W',
                        'k': izq.get('k') or der.get('k') or 1.0,
                        'modo': izq.get('modo') or der.get('modo') or 'sd',
                    })


class EditorCondiciones(QGroupBox):
    """Tabla de condiciones planas (todas AND entre sí) — versión simplificada
    de EditorReglas sin agrupación por regla/setup: se usa como filtro extra
    de un setup (entrada o salida), aplicable sobre CUALQUIER plantilla, no
    solo 'Custom (reglas)'.

    Por encima de esas condiciones del usuario puede pintar las filas de la
    SEÑAL de la plantilla (ver core/strategies.filas_plantilla): son la lógica
    que el sistema ya ejecuta de por sí, así que no se guardan como filtro
    —duplicarlas como AND sería redundante— pero sí son editables: cada celda
    mapeada escribe en el parámetro correspondiente de la plantilla."""

    COL_INDICADOR, COL_PERIODO, COL_OP, COL_DER, COL_VALOR, COL_DIR = range(6)
    COL_ANCLAJE, COL_K, COL_MODO = range(6, 9)
    # qué parámetro de la plantilla edita cada celda, según el 'mapeo' del
    # descriptor (ver core/strategies._fila)
    _CELDA_MAPEO = {COL_INDICADOR: 'izq.tipo', COL_PERIODO: 'izq.periodo',
                    COL_DER: 'der.tipo', COL_VALOR: 'der.valor'}
    _FONDO_PLANTILLA = QColor('#16233b')

    def __init__(self, titulo, parent=None):
        super().__init__(titulo, parent)
        lay = QVBoxLayout(self)
        self.tabla = QTableWidget(0, 9)
        self.tabla.setHorizontalHeaderLabels(
            ['Indicador', 'Periodo', 'Operador', 'Comparar con', 'Valor/Periodo',
             'Dirección', 'Anclaje', 'Desviación', 'Modo'])
        self.tabla.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.tabla.verticalHeader().setVisible(False)
        self.tabla.verticalHeader().setDefaultSectionSize(30)
        self.tabla.setMinimumHeight(90)
        _activar_borde_fila(self.tabla)
        _instalar_borrado_fila(self.tabla, self._del_fila)
        lay.addWidget(self.tabla)
        fila = QHBoxLayout()
        btn_add = QPushButton("+ Condición")
        btn_add.clicked.connect(self._add_fila)
        btn_del = QPushButton("− Quitar")
        btn_del.clicked.connect(self._del_fila)
        fila.addWidget(btn_add)
        fila.addWidget(btn_del)
        fila.addStretch()
        lay.addLayout(fila)
        # expuesto para que quien use esta tabla como "Entrada del setup"
        # (no en los diálogos transitorios de Condiciones) pueda añadir aquí
        # su propio botón de deshacer — ver OptimizadorWidget.__init__
        self.fila_botones = fila
        # filas de la señal de la plantilla, siempre al principio de la tabla
        self._filas_plantilla = []
        self._plantilla = None
        self._on_param = None
        self._sincronizando = False

    # ── filas de la señal de la plantilla ──
    @property
    def n_filas_plantilla(self):
        return len(self._filas_plantilla)

    def cargar(self, condiciones, filas=None, plantilla=None, on_param=None):
        """Repinta la tabla entera: primero las filas de la señal de la
        plantilla (`filas`, descriptores de core/strategies.filas_plantilla) y
        debajo las condiciones del usuario. `on_param(clave, valor)` recibe las
        ediciones que caen sobre un parámetro de la plantilla."""
        self._plantilla = plantilla
        self._on_param = on_param
        self._filas_plantilla = list(filas or [])
        self._sincronizando = True
        # setItem (fila de texto) emitiría cellChanged y realimentaría el
        # guardado del setup mientras se está repintando
        self.tabla.blockSignals(True)
        try:
            self.tabla.clearSpans()
            self.tabla.setRowCount(0)
            for i, desc in enumerate(self._filas_plantilla):
                if i and i % 4 == 0:
                    bombear_eventos()
                self._add_fila_plantilla(i, desc)
            self.cargar_condiciones(condiciones)
        finally:
            self.tabla.blockSignals(False)
            self._sincronizando = False
            _repintar_seleccion_fila(self.tabla)

    def _widget_param(self, clave, valor):
        """Widget para una celda mapeada a un parámetro de la plantilla, con
        el rango/opciones que declara ESTRATEGIAS. None si el parámetro no
        existe o no es editable con un widget simple."""
        specs = ESTRATEGIAS.get(self._plantilla, {}).get('params', [])
        spec = next((s for s in specs if s['clave'] == clave), None)
        if spec is None:
            return None, None
        if spec['tipo'] == 'choice':
            w = _combo_regla(spec['opciones'], str(valor))
            return w, w.currentText
        if spec['tipo'] == 'int':
            w = QSpinBox()
            w.setRange(spec.get('min', 1), spec.get('max', 100000))
            w.setValue(int(valor))
            return w, w.value
        if spec['tipo'] == 'float':
            w = QDoubleSpinBox()
            w.setRange(spec.get('min', -1e9), spec.get('max', 1e9))
            w.setDecimals(2)
            w.setValue(float(valor))
            return w, w.value
        return None, None

    def _add_fila_plantilla(self, r, desc):
        self.tabla.insertRow(r)
        mapeo = desc.get('mapeo') or {}
        tooltip = ("Esta es la señal de la plantilla: lo que el sistema hace de "
                   "por sí. Las celdas activas editan sus parámetros.")
        if desc.get('nota'):
            tooltip += f"\n⚠ {desc['nota']}"
        lados = [d.get('direccion') for d in self._filas_plantilla]
        if len(lados) == 2 and set(lados) == {'long', 'short'} and any(
                sp['clave'] == 'direccion'
                for sp in ESTRATEGIAS.get(self._plantilla, {}).get('params', [])):
            tooltip += ("\nSelecciona esta fila y pulsa «− Quitar» para dejar "
                        "solo el otro lado (Long/Short).")

        if desc.get('izq') is None:
            # la tabla no sabe representar este indicador (Stochastic, %R,
            # CCI, patrones): una sola celda de texto a lo ancho
            item = QTableWidgetItem(f"▸ señal de la plantilla · {desc['texto']}")
            item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            item.setBackground(self._FONDO_PLANTILLA)
            item.setToolTip(tooltip + "\n(no representable como fila de "
                            "indicador: edítala desde «Parámetros»)")
            self.tabla.setItem(r, 0, item)
            self.tabla.setSpan(r, 0, 1, 9)
            return

        izq, der = desc['izq'], desc['der']
        es_valor = der['tipo'] == 'valor'
        # 'der.valor' y 'der.periodo' comparten la misma celda (col. 4)
        mapeo = dict(mapeo)
        if 'der.periodo' in mapeo:
            mapeo['der.valor'] = mapeo.pop('der.periodo')
        base = {
            self.COL_INDICADOR: _combo_regla(_INDICADORES_REGLA, izq['tipo']),
            self.COL_PERIODO: _spin_regla(izq.get('periodo', 14), 1, 5000),
            self.COL_OP: _combo_regla(_OPERADORES_REGLA, desc['op']),
            self.COL_DER: _combo_regla(['Valor'] + _INDICADORES_REGLA,
                                       'Valor' if es_valor else der['tipo']),
            self.COL_VALOR: _spin_regla(
                der.get('valor', der.get('periodo', 0.0)), dec=4),
            self.COL_DIR: _combo_regla(
                list(_MAPA_DIRECCION),
                _MAPA_DIRECCION_INV.get(desc.get('direccion', 'ambas'), 'Ambas')),
        }
        for col, w in base.items():
            clave = mapeo.get(self._CELDA_MAPEO.get(col))
            if clave:
                editable, leer = self._widget_param(clave, self._valor_celda(col, desc))
                if editable is not None:
                    w = editable
                    w.setToolTip(tooltip)
                    self._conectar_param(w, clave, leer)
                else:
                    w.setEnabled(False)
            else:
                w.setEnabled(False)
                w.setToolTip(tooltip)
            self.tabla.setCellWidget(r, col, w)

    @staticmethod
    def _valor_celda(col, desc):
        izq, der = desc['izq'], desc['der']
        if col == EditorCondiciones.COL_INDICADOR:
            return izq['tipo']
        if col == EditorCondiciones.COL_PERIODO:
            return izq.get('periodo', 14)
        if col == EditorCondiciones.COL_DER:
            return der['tipo']
        return der.get('valor', der.get('periodo', 0.0))

    def _conectar_param(self, w, clave, leer):
        def _emitir(*_):
            if self._sincronizando or self._on_param is None:
                return
            self._on_param(clave, leer())
        if isinstance(w, QComboBox):
            w.currentTextChanged.connect(_emitir)
        else:
            w.valueChanged.connect(_emitir)

    def sincronizar_filas_plantilla(self, filas):
        """Refresca los valores de las filas de plantilla sin recrearlas
        (así no se destruye el widget que el usuario esté editando). Devuelve
        False si la estructura cambió —p. ej. al pasar de Ambas a solo Long— y
        hace falta un `cargar` completo."""
        filas = list(filas or [])
        if len(filas) != len(self._filas_plantilla):
            return False
        self._sincronizando = True
        try:
            for r, desc in enumerate(filas):
                anterior = self._filas_plantilla[r]
                if (desc.get('izq') is None) != (anterior.get('izq') is None):
                    return False
                if desc.get('izq') is None:
                    item = self.tabla.item(r, 0)
                    if item is not None:
                        item.setText(f"▸ señal de la plantilla · {desc['texto']}")
                    continue
                for col in (self.COL_INDICADOR, self.COL_PERIODO, self.COL_OP,
                            self.COL_DER, self.COL_VALOR):
                    w = self.tabla.cellWidget(r, col)
                    if w is None:
                        continue
                    if col == self.COL_OP:
                        w.setCurrentText(desc['op'])
                    elif col == self.COL_DER:
                        w.setCurrentText('Valor' if desc['der']['tipo'] == 'valor'
                                         else desc['der']['tipo'])
                    else:
                        valor = self._valor_celda(col, desc)
                        if isinstance(w, QComboBox):
                            w.setCurrentText(str(valor))
                        elif isinstance(w, QSpinBox):
                            w.setValue(int(valor))
                        else:
                            w.setValue(float(valor))
        finally:
            self._sincronizando = False
        self._filas_plantilla = filas
        return True

    def _add_fila(self, _=False, datos=None):
        r = self.tabla.rowCount()
        self.tabla.insertRow(r)
        d = datos or {}
        w0 = _combo_regla(_INDICADORES_REGLA, d.get('izq', 'close'))
        w1 = _spin_regla(d.get('izq_periodo', 14), 1, 5000)
        w2 = _combo_regla(_OPERADORES_REGLA, d.get('op', '>'))
        w3 = _combo_regla(['Valor'] + _INDICADORES_REGLA, d.get('der', 'Valor'))
        w4 = _spin_regla(d.get('der_valor', 0.0), dec=4)
        _sincronizar_periodo(w0, w1)
        _sincronizar_periodo(w3, w4)
        cmb_dir = _combo_regla(list(_MAPA_DIRECCION), d.get('direccion', 'Ambas'))
        cmb_dir.setToolTip(
            "A qué lado se aplica esta condición: 'Ambas' la exige tanto para "
            "entradas/salidas long como short (comportamiento clásico); "
            "'Long'/'Short' la restringe solo a ese lado, sin afectar al otro "
            "— p.ej. close > SMA(200) en Long y close < SMA(200) en Short "
            "dentro del mismo setup.")
        # columnas VWAP (anclaje / desviación / modo) — solo activas si la
        # fila usa una serie VWAP configurable
        ancla_ini = _MAPA_ANCLAJE_INV.get(d.get('anclaje', 'W'), 'Semana')
        cmb_ancla = _combo_regla(list(_MAPA_ANCLAJE), ancla_ini)
        cmb_ancla.setToolTip("Periodo de referencia del VWAP: cuándo se "
                             "reinicia el acumulado (sesión/semana/mes/…).")
        cmb_k = _combo_regla(_K_VWAP, '{:g}'.format(float(d.get('k', 1.0))))
        cmb_k.setToolTip("Multiplicador de la banda: VWAP ± k·σ (o k% en modo "
                         "porcentaje).")
        cmb_modo = _combo_regla(list(_MAPA_MODO_VWAP),
                                _MAPA_MODO_VWAP_INV.get(d.get('modo', 'sd'), 'SD'))
        cmb_modo.setToolTip("Modo de la banda: desviación estándar del anclaje "
                            "o porcentaje del propio VWAP.")
        _sincronizar_vwap(w0, w3, cmb_ancla, cmb_k, cmb_modo)
        for col, w in enumerate((w0, w1, w2, w3, w4, cmb_dir,
                                 cmb_ancla, cmb_k, cmb_modo)):
            _seleccionar_fila_al_clic(w, self.tabla, r)
            self.tabla.setCellWidget(r, col, w)
        _repintar_seleccion_fila(self.tabla)

    def _intentar_quitar_lado(self, r):
        """Si la fila r es una de las DOS filas de plantilla (long+short) y la
        plantilla tiene un parámetro 'direccion' que controla cuántas hay,
        borrarla equivale a poner ese parámetro al lado contrario — mismo
        resultado que el combo 'Dirección' de Parámetros, pero desde la
        tabla. Devuelve True si lo hizo (patrones de velas y Custom (reglas)
        no califican: no tienen un 'direccion' global que gobierne la fila)."""
        if self.n_filas_plantilla != 2 or self._on_param is None:
            return False
        lados = [d.get('direccion') for d in self._filas_plantilla]
        if set(lados) != {'long', 'short'}:
            return False
        specs = ESTRATEGIAS.get(self._plantilla, {}).get('params', [])
        if not any(sp['clave'] == 'direccion' for sp in specs):
            return False
        self._on_param('direccion', 'Short' if lados[r] == 'long' else 'Long')
        return True

    def _del_fila(self):
        # las filas de la señal de la plantilla no se pueden quitar en
        # general: son la lógica del sistema, no un filtro añadido — salvo
        # que sean las dos mitades (long/short) de una plantilla con
        # 'direccion', en cuyo caso borrar una equivale a dejar solo el otro
        # lado (ver _intentar_quitar_lado).
        primera = self.n_filas_plantilla
        r = self.tabla.currentRow()
        if 0 <= r < primera:
            self._intentar_quitar_lado(r)
            return
        if self.tabla.rowCount() <= primera:
            return
        self.tabla.removeRow(r if r >= primera else self.tabla.rowCount() - 1)

    def condiciones(self):
        """Lista plana de condiciones (AND entre sí dentro de su misma
        dirección; ver 'direccion' de cada una). Las filas de la señal de la
        plantilla quedan fuera: el motor ya las ejecuta, y repetirlas como
        filtro AND solo podría restringir de más."""
        out = []
        for r in range(self.n_filas_plantilla, self.tabla.rowCount()):
            if self.tabla.cellWidget(r, 0) is None:
                # fila recién insertada (rowsInserted dispara el guardado antes
                # de que _add_fila termine de poner sus cellWidget): se ignora
                # aquí, la próxima señal real ya la incluirá completa.
                continue
            izq = self.tabla.cellWidget(r, 0).currentText()
            izq_per = self.tabla.cellWidget(r, 1).value()
            op = self.tabla.cellWidget(r, 2).currentText()
            der = self.tabla.cellWidget(r, 3).currentText()
            der_val = self.tabla.cellWidget(r, 4).value()
            direccion = _MAPA_DIRECCION[self.tabla.cellWidget(r, 5).currentText()]
            anclaje = _MAPA_ANCLAJE[self.tabla.cellWidget(r, 6).currentText()]
            k = float(self.tabla.cellWidget(r, 7).currentText())
            modo = _MAPA_MODO_VWAP[self.tabla.cellWidget(r, 8).currentText()]
            out.append({'izq': _spec_regla(izq, izq_per, anclaje, k, modo),
                        'op': op,
                        'der': _spec_regla(der, der_val, anclaje, k, modo),
                        'direccion': direccion})
        return out

    def cargar_condiciones(self, condiciones):
        """Repuebla solo las condiciones del usuario, conservando las filas de
        la señal de la plantilla que haya encima."""
        while self.tabla.rowCount() > self.n_filas_plantilla:
            self.tabla.removeRow(self.tabla.rowCount() - 1)
        for i, cond in enumerate(condiciones or []):
            if i and i % 4 == 0:
                bombear_eventos()
            izq, der = cond['izq'], cond['der']
            self._add_fila(datos={
                'izq': izq['tipo'], 'izq_periodo': izq.get('periodo', 14),
                'op': cond['op'],
                'der': 'Valor' if der['tipo'] == 'valor' else der['tipo'],
                'der_valor': der.get('valor', der.get('periodo', 0.0)),
                'direccion': _MAPA_DIRECCION_INV.get(cond.get('direccion', 'ambas'), 'Ambas'),
                'anclaje': izq.get('anclaje') or der.get('anclaje') or 'W',
                'k': izq.get('k') or der.get('k') or 1.0,
                'modo': izq.get('modo') or der.get('modo') or 'sd',
            })


# ══════════════ sub-pestaña Optimizador ══════════════
def _setup_por_defecto(plantilla='Cruce de medias'):
    s = {'nombre': 'Setup 1', 'plantilla': plantilla,
         'params': params_por_defecto(plantilla),
         'riesgo_pct': 0.01, 'stop_atr': 2.0, 'stop_atr_modo': 'fijo',
         'tp_r': 0.0,
         'salida_n_velas': 0, 'edge': False,
         'be_atr': 0.0, 'be_unidad': 'atr', 'trailing_atr': 0.0,
         # la salida implícita de la plantilla, explícita y editable: 100% a
         # su señal de salida. Equivale a no tener etapas (ver core/backtest).
         'parciales': [etapa_salida_por_defecto()],
         # idem para la entrada: un solo tramo al 100% a la señal equivale a
         # no tener entrada escalonada.
         'tramos': [tramo_entrada_por_defecto()],
         'filtros': _filtros_por_defecto()}
    # la plantilla puede recomendar su propio stop/tp por defecto (p.ej. el
    # cruce de medias sale por cruce contrario: sin stop ATR)
    s.update(defaults_setup(plantilla))
    return s


# mapas etiqueta (UI, español) <-> valor interno del setup['filtros']
_MAPA_REGIMEN = {
    'Ninguno': 'ninguno', 'Tendencia (ER)': 'er_tendencia',
    'Rango (ER)': 'er_rango', 'Tendencia (Hurst)': 'hurst_tendencia',
    'Reversión (Hurst)': 'hurst_reversion',
}
_MAPA_REGIMEN_INV = {v: k for k, v in _MAPA_REGIMEN.items()}


def _normalizar_metodo_regimen(metodo):
    """Unifica identificadores antiguos al canónico del core.

    Versiones previas guardaban la reversión de Hurst como 'hurst_reversión'
    (con tilde), pero core/strategies.py solo reconoce 'hurst_reversion'.
    Sin esta migración, un setup guardado con la tilde cargaba como «Ninguno»
    y el filtro se perdía en silencio."""
    if metodo == 'hurst_reversión':
        return 'hurst_reversion'
    return metodo
# La entrada por orden límite está terminada en el motor pero aún no se abre al
# uso: se lista en gris y no se puede elegir (ver _bloquear_entrada_limite).
_ETIQUETA_LIMITE_FIB = 'Límite en nivel Fibonacci (próximamente)'
_MAPA_TIPO_ENTRADA = {
    'A mercado': 'mercado',
    _ETIQUETA_LIMITE_FIB: 'limite_fib',
}
_MAPA_TIPO_ENTRADA_INV = {v: k for k, v in _MAPA_TIPO_ENTRADA.items()}
_MAPA_VOLATILIDAD = {
    'Ninguna': 'ninguno',
    'ATR alto (percentil ≥)': 'atr_percentil_alto',
    'ATR bajo (percentil ≤)': 'atr_percentil_bajo',
    'Desv. estándar alta (percentil ≥)': 'stdev_percentil_alto',
    'Desv. estándar baja (percentil ≤)': 'stdev_percentil_bajo',
}
_MAPA_SESION = {
    'Ninguna': 'ninguna', 'Overnight': 'overnight', 'Londres': 'londres',
    'NY': 'ny', 'Personalizada': 'personalizada',
}
_MAPA_VOLATILIDAD_INV = {v: k for k, v in _MAPA_VOLATILIDAD.items()}
_MAPA_SESION_INV = {v: k for k, v in _MAPA_SESION.items()}
_MAPA_IMPACTO_NOTICIAS = {'Bajo': 'bajo', 'Medio': 'medio', 'Alto': 'alto'}
_MAPA_IMPACTO_NOTICIAS_INV = {v: k for k, v in _MAPA_IMPACTO_NOTICIAS.items()}
_MAPA_TF_SUPERIOR = {
    'Ninguno': 'ninguno', 'SMA': 'SMA', 'EMA': 'EMA', 'Supertrend': 'SUPERTREND',
}
_MAPA_TF_SUPERIOR_INV = {v: k for k, v in _MAPA_TF_SUPERIOR.items()}
_MAPA_RELACION_TF = {'Ambos': 'ambos', 'Solo largos': 'long', 'Solo cortos': 'short'}
_MAPA_RELACION_TF_INV = {v: k for k, v in _MAPA_RELACION_TF.items()}

# dirección de una condición de filtro (EditorCondiciones): a qué lado(s) se
# aplica. 'Ambas' es el valor por defecto y restringe long y short por igual
# (comportamiento previo a esta columna, 100% compatible con setups guardados
# sin el campo 'direccion').
_MAPA_DIRECCION = {'Ambas': 'ambas', 'Long': 'long', 'Short': 'short'}
_MAPA_DIRECCION_INV = {v: k for k, v in _MAPA_DIRECCION.items()}

# ── mecanismos globales de salida dentro de la tabla «Salida del setup» ──
# Son las cuatro redes de seguridad del setup (stop/TP/BE/trailing): no son
# etapas secuenciales, así que se pintan al final de la tabla, con un color
# propio para distinguirlas de un vistazo, y se marcan con este rol para que
# _leer_tabla_etapas y _validar_pct_parciales las ignoren al reconstruir
# 'parciales'.
# unidad de la distancia de activación del break-even. '× ATR' es el defecto
# por compatibilidad: es lo que hacían los setups guardados antes de existir
# esta opción (ver core/backtest, 'be_unidad').
_MAPA_BE_UNIDAD = {'× ATR': 'atr', 'R': 'r'}
_MAPA_BE_UNIDAD_INV = {v: k for k, v in _MAPA_BE_UNIDAD.items()}

# modo del Stop Loss ×ATR del setup (setup['stop_atr_modo']): 'fijo' ancla el
# stop a la primera entrada; 'dinamico_promedio' lo reancla al precio medio
# de la posición con el ATR del momento tras cada entrada (ver core/backtest).
_MAPA_STOP_MODO = {'Fijo': 'fijo',
                   'Dinámico por promedio': 'dinamico_promedio'}
_MAPA_STOP_MODO_INV = {v: k for k, v in _MAPA_STOP_MODO.items()}

_ROL_MECANISMO = 'mecanismo_salida'
# la clave del mecanismo va en la celda 0 para localizar su fila sin depender
# de comparar textos (ver _fila_de_mecanismo)
_ROL_CLAVE_MECANISMO = int(Qt.ItemDataRole.UserRole) + 1
ROJO_OSCURO = '#8b1a1a'   # trailing: rojo apagado, para no competir con el stop

# clave del setup -> (etiqueta, atributo del spin que lo activa, sufijo, color).
# sufijo None = lo decide un combo de la propia fila (el break-even, cuya
# unidad depende de cmb_be_unidad).
_MECANISMOS_SALIDA_UI = {
    'salida_stop':     ('Stop Loss', 'sp_stop', ' ×ATR', ROJO),
    'salida_tp':       ('Take Profit', 'sp_tp', ' R', VERDE),
    'salida_be':       ('Break Even', 'sp_be', None, AZUL),
    'salida_trailing': ('Trailing Stop', 'sp_trailing', ' ×ATR', ROJO_OSCURO),
    'salida_tiempo':   ('Tiempo', 'sp_tiempo', ' velas', AMBAR),
}
# mínimo del spin de la fila: nunca 0, para que la fila no se autodestruya
# mientras la editas (para apagar el mecanismo se usa su campo de arriba)
_MIN_VALOR_MECANISMO = {'salida_tiempo': 1}

# clave escalar del setup -> atributo del spin que la edita (ver _deshacer)
_ATTR_ESCALAR_MECANISMO = {
    'stop_atr': 'sp_stop', 'tp_r': 'sp_tp', 'salida_n_velas': 'sp_tiempo',
    'be_atr': 'sp_be', 'trailing_atr': 'sp_trailing'}

# disparador de una etapa de salida parcial (ver core/strategies.trigger_etapa)
_MAPA_TRIGGER = {'Señal de la plantilla': 'senal', 'R:R alcanzado': 'r',
                 'Solo condiciones': 'cond',
                 'Estancamiento (N velas sin R)': 'estancamiento'}
_MAPA_TRIGGER_INV = {v: k for k, v in _MAPA_TRIGGER.items()}
# unidad del spin 1 (siempre visible salvo señal/condiciones), por disparador
_SUFIJOS_TRIGGER_ETAPA = {'r': ' R', 'estancamiento': ' velas'}
# 'estancamiento' es el único disparador con un SEGUNDO número (N velas en el
# spin 1 de arriba + R mínimo aquí); campo2 = (sufijo del 2º spin, valor defecto)
_CAMPO2_TRIGGER_ETAPA = {'estancamiento': (' R', 1.0)}
# tooltip por opción del combo de disparador (etapas de Salida): al pasar el
# ratón por una opción concreta del desplegable, o al dejar el combo cerrado
# en esa opción, se muestra SOLO su propia explicación — no un bloque con
# las 4 juntas.
_TOOLTIPS_TRIGGER_ETAPA = {
    'senal': ("La etapa cierra su % cuando la plantilla da su señal de "
             "salida (el cruce contrario, la vuelta a la media...), al "
             "open de la vela siguiente."),
    'r': ("Cierra intra-vela al tocar ese múltiplo del riesgo inicial de "
         "la operación."),
    'cond': ("Cierra intra-vela en cuanto se cumplan las «Condiciones» de "
            "la etapa, al margen de la plantilla."),
    'estancamiento': ("Cierra si, tras el nº de velas indicado desde la "
                      "entrada, el precio no ha llegado a alcanzar (en su "
                      "mejor momento) el R mínimo indicado — para cortar "
                      "operaciones que no arrancan sin esperar al stop."),
}

# disparador de un tramo de entrada escalonada (ver core/strategies.trigger_tramo)
_MAPA_TRIGGER_ENTRADA = {
    'Señal de la plantilla': 'senal', 'A +N velas': 'velas',
    'Retroceso (promediar)': 'retroceso', 'Avance (pirámide)': 'avance',
    'Solo condiciones': 'cond',
}
_MAPA_TRIGGER_ENTRADA_INV = {v: k for k, v in _MAPA_TRIGGER_ENTRADA.items()}
_SUFIJOS_TRIGGER_ENTRADA = {'velas': ' velas', 'retroceso': ' ×ATR', 'avance': ' R'}
# igual que _TOOLTIPS_TRIGGER_ETAPA, pero para el combo de tramos de
# entrada escalonada
_TOOLTIPS_TRIGGER_ENTRADA = {
    'senal': ("El promedio se construye cuando se repite la señal de entrada "
               "(mismo lado), al open de la vela siguiente."),
    'velas': ("Entra N velas después de la 1ª entrada — reparte la "
               "ejecución para no depender de un solo open."),
    'retroceso': ("Promedia cuando el precio va NxATR en contra de la 1a "
                   "entrada (reversión a la media)."),
    'avance': "Piramida cuando el precio avanza +N R a favor (tendencia).",
    'cond': "Entra en cuanto se cumplan sus «Condiciones».",
}


class TemplateCard(QFrame):
    """Tarjeta visual seleccionable para elegir plantilla o sistema.
    `clicked` se emite con un solo click (usado por el diálogo «Elegir…» y
    por Favoritos/Guardados, donde un click ya reemplaza el sistema como
    acción deliberada de un solo paso). `activado` se emite solo con doble
    click o Enter/Return con la tarjeta enfocada — usado donde un click
    accidental no debe disparar la acción (recientes de Predeterminados,
    que AÑADEN un setup en vez de reemplazar)."""
    clicked = pyqtSignal(str)
    activado = pyqtSignal(str)

    def __init__(self, nombre, desc_corta, color, parent=None, categoria=None):
        super().__init__(parent)
        self._nombre = nombre
        self._selected = False
        self._color = color

        self.setMinimumWidth(130)
        self.setMinimumHeight(52)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        vlay = QVBoxLayout(self)
        vlay.setContentsMargins(10, 6, 10, 6)
        vlay.setSpacing(1)

        self.lbl_nombre = QLabel(nombre)
        self.lbl_nombre.setStyleSheet(
            "background: transparent; font-weight: bold; font-size: 12px; color: #c8d6e5;")
        self.lbl_nombre.setWordWrap(True)
        self.lbl_desc = QLabel(desc_corta)
        self.lbl_desc.setStyleSheet(
            "background: transparent; font-size: 10px; color: #8fb3d9;")
        self.lbl_desc.setWordWrap(True)

        vlay.addWidget(self.lbl_nombre)
        vlay.addWidget(self.lbl_desc)
        vlay.addStretch()
        # chip gris (sugerencia de categoría) en la esquina inferior derecha
        if categoria:
            self.lbl_categoria = QLabel(categoria)
            self.lbl_categoria.setStyleSheet(
                f"background: transparent; font-size: 8px; color: {GRIS};")
            self.lbl_categoria.setAlignment(Qt.AlignmentFlag.AlignRight)
            vlay.addWidget(self.lbl_categoria)
        self._update_style()

    def _update_style(self):
        c = self._color
        if self._selected:
            self.setStyleSheet(
                f"TemplateCard {{"
                f"  background-color: #1a2a4a;"
                f"  border: 2px solid {c};"
                f"  border-left: 5px solid {c};"
                f"  border-radius: 6px;"
                f"}}"
                f"TemplateCard:focus {{"
                f"  border: 1px solid {c};"
                f"  border-left: 5px solid {c};"
                f"}}")
        else:
            self.setStyleSheet(
                f"TemplateCard {{"
                f"  background-color: #0f1a30;"
                f"  border: 1px solid #253a60;"
                f"  border-left: 5px solid {c};"
                f"  border-radius: 6px;"
                f"}}"
                f"TemplateCard:hover {{"
                f"  background-color: #162545;"
                f"  border: 1px solid #3a5a8a;"
                f"  border-left: 5px solid {c};"
                f"}}"
                f"TemplateCard:focus {{"
                f"  border: 1px solid {c};"
                f"  border-left: 5px solid {c};"
                f"}}")

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.setFocus()
            self.clicked.emit(self._nombre)
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.activado.emit(self._nombre)
        super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.activado.emit(self._nombre)
        else:
            super().keyPressEvent(event)

    def setSelected(self, selected):
        self._selected = selected
        self._update_style()

    @property
    def nombre(self):
        return self._nombre


class DialogoSeleccionTarjeta(DialogoBase):
    """Ventana modal con todas las opciones como tarjetas en rejilla (3
    columnas) dentro de un scroll VERTICAL (barra ya tematizada). Un click en
    una tarjeta la elige y cierra el diálogo."""

    def __init__(self, opciones, actual, titulo, parent=None):
        super().__init__(titulo, parent,
                         subtitulo='Elige una opción con un clic',
                         ancho=480, alto=400)
        self.elegido = None

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        cont = QWidget()
        grid = QGridLayout(cont)
        grid.setSpacing(6)
        grid.setContentsMargins(0, 0, 0, 0)
        for i, (nombre, meta) in enumerate(opciones):
            card = TemplateCard(nombre, meta.get('desc_corta', nombre),
                                meta.get('color', '#607d8b'),
                                categoria=meta.get('categoria'))
            card.setSelected(nombre == actual)
            card.clicked.connect(self._elegir)
            grid.addWidget(card, i // 3, i % 3)
        scroll.setWidget(cont)
        self.contenido.addWidget(scroll, 1)

        btn_cancelar = QPushButton("Cancelar")
        btn_cancelar.setObjectName("accionNeutra")
        btn_cancelar.clicked.connect(self.reject)
        fila_btn = QHBoxLayout()
        fila_btn.addStretch()
        fila_btn.addWidget(btn_cancelar)
        self.contenido.addLayout(fila_btn)

    def _elegir(self, nombre):
        self.elegido = nombre
        self.accept()


class SelectorTarjetas(QWidget):
    """Selector compacto de alto fijo: muestra la tarjeta seleccionada + las
    usadas recientemente (acceso rápido) + un botón «Elegir…» que abre
    DialogoSeleccionTarjeta. Sustituye a la fila scrolleable de tarjetas para
    que no desborde el margen derecho."""
    seleccion_cambiada = pyqtSignal(str)

    def __init__(self, opciones, titulo, parent=None, actual_inicial=None,
                 max_recientes=3, clave_persistencia=None):
        super().__init__(parent)
        self._opciones = dict(opciones)          # nombre -> meta
        self._orden = [n for n, _ in opciones]   # orden original
        self._titulo = titulo
        self._max_recientes = max_recientes
        self._clave = clave_persistencia         # None -> solo en memoria
        self._actual = actual_inicial if actual_inicial in self._opciones else None
        self._card_actual = None                 # para tooltips
        # recientes persistidos entre reinicios: se filtran a opciones válidas,
        # se excluye el actual y se recorta al máximo
        recientes = get_selector_recientes(self._clave) if self._clave else []
        self._recientes = [
            n for n in recientes
            if n in self._opciones and n != self._actual][:self._max_recientes]

        self._lay = QHBoxLayout(self)
        self._lay.setContentsMargins(0, 0, 0, 0)
        self._lay.setSpacing(6)
        self._rebuild()

    def _guardar_recientes(self):
        # se persiste la actual (si hay) al frente + el resto de recientes,
        # así lo último elegido sobrevive aunque se cierre la app antes de
        # elegir una segunda tarjeta (self._actual nunca se restaura solo,
        # p.ej. el selector de "sistema" no recibe actual_inicial)
        if self._clave:
            persistidos = [self._actual] if self._actual is not None else []
            persistidos += [r for r in self._recientes if r != self._actual]
            set_selector_recientes(self._clave, persistidos[:self._max_recientes])

    def seleccion(self):
        return self._actual

    def limpiar_seleccion(self):
        """Quita el resaltado de la tarjeta actual sin emitir señal (se usa
        cuando otro selector pasa a ser la fuente del sistema activo)."""
        if self._actual is not None:
            self._actual = None
            self._rebuild()

    def set_seleccion(self, nombre, emitir=False):
        if nombre not in self._opciones or nombre == self._actual:
            if emitir and nombre in self._opciones:
                self.seleccion_cambiada.emit(nombre)
            return
        # el anterior pasa al frente de recientes (dedup, sin el nuevo actual)
        if self._actual is not None:
            self._recientes = [self._actual] + [
                r for r in self._recientes if r != self._actual]
        self._recientes = [r for r in self._recientes if r != nombre]
        self._recientes = self._recientes[:self._max_recientes]
        self._actual = nombre
        self._guardar_recientes()
        self._rebuild()
        if emitir:
            self.seleccion_cambiada.emit(nombre)

    def set_tooltip_actual(self, html):
        if self._card_actual is not None:
            self._card_actual.setToolTip(html)

    def _make_card(self, nombre, seleccionada):
        meta = self._opciones[nombre]
        card = TemplateCard(nombre, meta.get('desc_corta', nombre),
                            meta.get('color', '#607d8b'),
                            categoria=meta.get('categoria'))
        card.setSelected(seleccionada)
        return card

    def _rebuild(self):
        # limpiar layout
        while self._lay.count():
            item = self._lay.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._card_actual = None

        if self._actual is not None:
            # la tarjeta actual es solo indicativa: no abre nada al pulsarla,
            # el diálogo se abre únicamente con el botón «Elegir…»
            self._card_actual = self._make_card(self._actual, True)
            self._lay.addWidget(self._card_actual)

        for nombre in self._recientes:
            card = self._make_card(nombre, False)
            # doble click o Enter (no un solo click) para evitar añadir un
            # setup por un click accidental al pasar por la tarjeta
            card.activado.connect(lambda n: self.set_seleccion(n, emitir=True))
            self._lay.addWidget(card)

        btn = QPushButton("Elegir…")
        btn.setObjectName("run")
        btn.clicked.connect(self._abrir_dialogo)
        self._lay.addWidget(btn)
        self._lay.addStretch()

    def _abrir_dialogo(self):
        opciones = [(n, self._opciones[n]) for n in self._orden]
        dlg = DialogoSeleccionTarjeta(opciones, self._actual, self._titulo, self)
        if dlg.exec() and dlg.elegido:
            self.set_seleccion(dlg.elegido, emitir=True)


class OptimizadorWidget(QWidget):
    """Constructor de SISTEMAS: un sistema es una lista de setups, cada uno
    con su plantilla (predefinida o custom por reglas), sus parámetros y su
    PROPIO riesgo %, stop ATR, take-profit y salida por tiempo. La
    definición (reglas de entrada/salida) de la plantilla elegida se muestra
    siempre, interpolada con los parámetros actuales."""
    ejecutar = pyqtSignal()    # lo escucha TabBacktest
    optimizar = pyqtSignal()   # idem — abre el diálogo de barrido de parámetros

    def __init__(self, parent=None):
        super().__init__(parent)
        self.csv_path = None
        self._overlay = None
        self._setups = [_setup_por_defecto()]
        self._cargando = False    # guard anti-bucle al poblar el editor
        self._fila_editada = None  # setup cuyo estado reflejan los widgets
        self._dias_disponibles = set(range(7))   # se recalcula al cargar un CSV
        # temporalidad: nativa del archivo (detectada del nombre, igual que
        # Patrones) y la elegida para el backtest — solo se puede subir de
        # granularidad, nunca bajar
        self._tf_nativo = None
        self._tf_actual = None
        self._tf_custom = None
        self._tf_buttons = {}

        # Los cambios de widgets disparan _guardar_setup_actual, que refresca
        # por completo el editor (tablas de entrada/salida, definición, código,
        # botón ejecutable). Arrastrar un spin o girar la rueda del ratón lo
        # dispara a ráfagas: se aplane el trabajo en un timer para que toda la
        # ráfaga colapse en un único refresco.
        self._timer_ui = QTimer(self)
        self._timer_ui.setSingleShot(True)
        self._timer_ui.setInterval(150)
        self._timer_ui.timeout.connect(self._flush_refrescos_ui)
        self._refresco_pendiente = False
        self._detectar_thread = None
        self._detectar_gen = 0

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.addWidget(splitter, 1)

        # ── izquierda: explorador de activos ──
        panel_izq = QWidget()
        lay_izq = QVBoxLayout(panel_izq)
        lay_izq.setContentsMargins(0, 0, 0, 0)
        buscador = QLineEdit()
        buscador.setPlaceholderText("Buscar activo…")
        lay_izq.addWidget(buscador)
        self.explorer = FileExplorer(LIMPIADOS_DIR, mode='csv')
        self.explorer.file_chosen.connect(self._on_file)
        buscador.textChanged.connect(self.explorer.set_search_text)
        lay_izq.addWidget(self.explorer, 1)
        splitter.addWidget(panel_izq)
        bombear_eventos()

        # ── derecha: configuración (scrolleable) ──
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        cont = QWidget()
        lay = QVBoxLayout(cont)
        lay.setSpacing(8)

        self.lbl_activo = QLabel("Ningún activo seleccionado")
        self.lbl_activo.setObjectName("titulo")
        lay.addWidget(self.lbl_activo)

        # ── temporalidad del backtest: por defecto la nativa del archivo,
        # se puede subir a una TF más gruesa (nunca bajar) igual que en la
        # pestaña Patrones; los botones más finos que la nativa quedan
        # bloqueados ──
        grp_tf = QGroupBox("Temporalidad del backtest")
        lay_tf = QVBoxLayout(grp_tf)
        lay_tf.insertLayout(0, _fila_ayuda(
            "Elige la temporalidad de las velas que usará el backtest: la "
            "nativa del CSV cargado, o un múltiplo agregado (p.ej. 1h → 4h). "
            "No cambia el activo, solo cómo se agrupan sus velas."))
        fila_tf = QHBoxLayout()
        fila_tf.setSpacing(4)
        self._tf_group = QButtonGroup(self)
        self._tf_group.setExclusive(True)
        for i, tf_label in enumerate(TF_LABELS):
            btn = QPushButton(tf_label)
            btn.setObjectName("tf")
            btn.setCheckable(True)
            btn.setEnabled(False)
            self._tf_group.addButton(btn, i)
            self._tf_buttons[tf_label] = btn
            fila_tf.addWidget(btn)
        self.btn_tf_custom = QPushButton("Custom…")
        self.btn_tf_custom.setObjectName("tf")
        self.btn_tf_custom.setCheckable(True)
        self.btn_tf_custom.setEnabled(False)
        self.btn_tf_custom.setToolTip(
            "Backtestear en una temporalidad personalizada (p.ej. 20m, 2h, "
            "3d, 5d). Debe ser igual o más gruesa que la nativa del archivo")
        self._tf_group.addButton(self.btn_tf_custom, len(TF_LABELS))
        fila_tf.addWidget(self.btn_tf_custom)
        fila_tf.addStretch()
        self._tf_group.idClicked.connect(self._on_tf_clicked)
        lay_tf.addLayout(fila_tf)
        self.lbl_tf_info = QLabel("")
        self.lbl_tf_info.setObjectName("estado")
        lay_tf.addWidget(self.lbl_tf_info)
        lay.addWidget(grp_tf)
        bombear_eventos()

        # ── calidad de datos ──
        grp_calidad = QGroupBox("Calidad de datos")
        lay_calidad = QVBoxLayout(grp_calidad)
        self.chk_excluir_interpoladas = QCheckBox(
            "Excluir velas interpoladas de las señales")
        self.chk_excluir_interpoladas.setChecked(True)
        self.chk_excluir_interpoladas.setToolTip(
            "Activado por defecto: las velas marcadas como interpoladas no "
            "pueden generar entradas nuevas ni órdenes límite. Se mantienen "
            "en la serie para conservar la continuidad temporal, y las "
            "salidas y la gestión de una posición ya abierta siguen "
            "funcionando.\n\n"
            "Al desactivarlo, las velas interpoladas se tratan como "
            "cualquier otra y vuelven a poder generar señales.")
        lay_calidad.addWidget(self.chk_excluir_interpoladas)
        lay.addWidget(grp_calidad)
        bombear_eventos()

        # ── selector de sistema (predeterminados / guardados / nuevo) ──
        grp_sel = QGroupBox("Sistema")
        lay_sel = QVBoxLayout(grp_sel)
        lay_sel.insertLayout(0, _fila_ayuda(
            "Carga un sistema de partida: una plantilla predefinida (Cruce "
            "de medias, RSI...), uno de tus sistemas guardados, o un "
            "favorito marcado desde Resultados. Cargar uno reemplaza los "
            "setups actuales del constructor."))

        # selector de sistemas predeterminados (tarjeta actual + recientes +
        # botón «Elegir…» que abre la ventana con todos)
        lbl_pred = QLabel("Predeterminados:")
        lbl_pred.setStyleSheet("font-size: 10px; color: #5a7a9a;")
        lay_sel.addWidget(lbl_pred)
        opciones_estrategias = [(n, ESTRATEGIAS[n]) for n in ESTRATEGIAS]
        self._selector_sistema = SelectorTarjetas(
            opciones_estrategias, "Elegir sistema predeterminado",
            clave_persistencia='sistema')
        self._selector_sistema.seleccion_cambiada.connect(
            self._on_sistema_card_clicked)
        lay_sel.addWidget(self._selector_sistema)

        # selector de sistemas guardados (combo si muchos, cards si pocos)
        self._guardadas_cards = {}
        self._guardado_cargado = None          # último sistema guardado cargado
        self._guardadas_container = QWidget()
        self._guardadas_lay = QHBoxLayout(self._guardadas_container)
        self._guardadas_lay.setSpacing(6)
        self.cmb_guardadas = QComboBox()
        self.cmb_guardadas.activated.connect(self._cargar_guardado)
        fila_g = QHBoxLayout()
        lbl_g = QLabel("Guardados:")
        lbl_g.setStyleSheet("font-size: 10px; color: #5a7a9a;")
        fila_g.addWidget(lbl_g)
        fila_g.addWidget(self.cmb_guardadas, 1)
        btn_cargar = QPushButton("Cargar")
        btn_cargar.clicked.connect(self._cargar_guardado)
        fila_g.addWidget(btn_cargar)
        btn_nuevo = QPushButton("+ Nuevo custom")
        btn_nuevo.setObjectName("run")
        btn_nuevo.setStyleSheet(
            "QPushButton { font-size: 10px; padding: 3px 8px; }")
        btn_nuevo.clicked.connect(self._nuevo_sistema_custom)
        fila_g.addWidget(btn_nuevo)
        btn_guardar = QPushButton("Guardar…")
        btn_guardar.clicked.connect(self._guardar_sistema)
        fila_g.addWidget(btn_guardar)
        btn_elim_g = QPushButton("Eliminar")
        btn_elim_g.clicked.connect(self._eliminar_guardada_seleccion)
        fila_g.addWidget(btn_elim_g)
        fila_g.addStretch()
        lay_sel.addLayout(fila_g)
        lay_sel.addWidget(self._guardadas_container)
        self._guardadas_container.setVisible(False)

        # ⭐ favoritos: combinaciones activo+temporalidad+setup guardadas
        # desde la pestaña Resultados (distinto de "Guardados": ahí solo se
        # guarda la estrategia, aquí se guarda con qué activo/tf se corrió)
        self._favoritos_cards = {}
        self._favorito_cargado = None          # último favorito cargado
        self._favoritos_container = QWidget()
        self._favoritos_lay = QHBoxLayout(self._favoritos_container)
        self._favoritos_lay.setSpacing(6)
        self.cmb_favoritos = QComboBox()
        self.cmb_favoritos.activated.connect(self._cargar_favorito)
        fila_f = QHBoxLayout()
        lbl_f = QLabel("⭐ Favoritos:")
        lbl_f.setStyleSheet("font-size: 10px; color: #5a7a9a;")
        fila_f.addWidget(lbl_f)
        fila_f.addWidget(self.cmb_favoritos, 1)
        btn_cargar_fav = QPushButton("Cargar")
        btn_cargar_fav.clicked.connect(self._cargar_favorito)
        fila_f.addWidget(btn_cargar_fav)
        btn_elim_f = QPushButton("Eliminar")
        btn_elim_f.clicked.connect(self._eliminar_favorito_seleccion)
        fila_f.addWidget(btn_elim_f)
        fila_f.addStretch()
        lay_sel.addLayout(fila_f)
        lay_sel.addWidget(self._favoritos_container)
        self._favoritos_container.setVisible(False)
        lay.addWidget(grp_sel)
        bombear_eventos()

        # ── sistema: lista de setups ──
        grp_sis = QGroupBox("Setups del sistema (cada uno con su propio riesgo)")
        lay_sis = QVBoxLayout(grp_sis)
        lay_sis.insertLayout(0, _fila_ayuda(
            "Un sistema puede combinar varios setups (plantillas) a la vez, "
            "cada uno con su propio riesgo, stop, TP y filtros. Añade, "
            "duplica, reordena o elimina setups aquí; el backtest los corre "
            "todos juntos sobre el mismo activo."))
        self.lista_setups = QListWidget()
        self.lista_setups.setMaximumHeight(120)
        self.lista_setups.currentRowChanged.connect(self._on_setup_selected)
        lay_sis.addWidget(self.lista_setups)
        fila_s = QHBoxLayout()
        for texto, slot in [("+ Setup", self._add_setup),
                            ("− Quitar", self._del_setup),
                            ("Duplicar", self._dup_setup),
                            ("↑", self._subir_setup), ("↓", self._bajar_setup)]:
            b = QPushButton(texto)
            b.clicked.connect(slot)
            fila_s.addWidget(b)
        fila_s.addStretch()
        lay_sis.addLayout(fila_s)

        lay.addWidget(grp_sis)
        bombear_eventos()

        # ── editor del setup seleccionado ──
        self.grp_setup = QGroupBox("Setup seleccionado")
        f_set = QFormLayout(self.grp_setup)
        self._form_setup = f_set
        _insertar_ayuda_form(f_set,
            "Edición completa del setup marcado en la lista de arriba: "
            "plantilla y sus parámetros, riesgo, entrada, salida y filtros. "
            "Los cambios se guardan automáticamente al cambiar de setup.")
        self.txt_nombre = QLineEdit()
        self.txt_nombre.textEdited.connect(self._guardar_setup_actual)
        f_set.addRow("Nombre:", self.txt_nombre)
        # ── selección de plantilla (tarjeta actual + recientes + «Elegir…») ──
        self._plantilla_actual = list(ESTRATEGIAS)[0]
        lbl_plantilla = QLabel("Plantilla:")
        f_set.addRow(lbl_plantilla)
        self._selector_plantilla = SelectorTarjetas(
            opciones_estrategias, "Elegir plantilla",
            actual_inicial=self._plantilla_actual,
            clave_persistencia='plantilla')
        self._selector_plantilla.seleccion_cambiada.connect(
            self._on_plantilla_changed)
        f_set.addRow(self._selector_plantilla)

        # resumen compacto de la definición (tooltip en la card tiene el detalle)
        self.lbl_resumen = QLabel("")
        self.lbl_resumen.setStyleSheet(
            "color: #8fb3d9; font-size: 10px; background-color: #101a2e;"
            "border: 1px solid #253a60; border-radius: 4px; padding: 4px 6px;")
        self.lbl_resumen.setWordWrap(True)
        f_set.addRow("Definición:", self.lbl_resumen)

        # parámetros autogenerados de la plantilla
        self.form_params_cont = QWidget()
        self.form_params = QFormLayout(self.form_params_cont)
        self.form_params.setContentsMargins(0, 0, 0, 0)
        f_set.addRow(self.form_params_cont)
        bombear_eventos()

        self.editor_reglas = EditorReglas()
        self.editor_reglas.setVisible(False)
        self.editor_reglas.tabla.cellChanged.connect(
            lambda *_: self._guardar_setup_actual())
        f_set.addRow(self.editor_reglas)

        # ── cómo se EJECUTA la señal de entrada (a mercado o esperando un
        # retroceso con una orden límite) — aplica a cualquier plantilla ──
        self.cmb_tipo_entrada = QComboBox()
        self.cmb_tipo_entrada.addItems(list(_MAPA_TIPO_ENTRADA))
        self._bloquear_entrada_limite()
        self.cmb_tipo_entrada.setToolTip(
            "Qué hace el setup cuando su plantilla da señal:\n"
            "· A mercado — entra al open de la vela siguiente (lo de siempre).\n"
            "· Límite en nivel Fibonacci — no entra: deja una orden esperando\n"
            "  en el retroceso del último tramo del ZigZag. Si el precio no\n"
            "  vuelve a ese nivel, la operación sencillamente no ocurre.\n"
            "  Próximamente: por ahora no se puede seleccionar.")
        self.cmb_tipo_entrada.currentTextChanged.connect(self._on_tipo_entrada_changed)
        f_set.addRow("Tipo de entrada:", self.cmb_tipo_entrada)
        bombear_eventos()

        fila_fib = QHBoxLayout()
        self.cmb_nivel_fib = QComboBox()
        self.cmb_nivel_fib.addItems([f"{v:g}" for v in NIVELES_FIB])
        self.cmb_nivel_fib.setCurrentText('0.618')
        self.cmb_nivel_fib.setToolTip(
            "Profundidad del retroceso donde se coloca la orden. Cuanto más "
            "profundo, mejor precio pero menos probabilidad de que se rellene.")
        self.cmb_nivel_fib.currentTextChanged.connect(self._guardar_setup_actual)
        self.sp_zz_desviacion = QDoubleSpinBox()
        self.sp_zz_desviacion.setRange(0.01, 100.0)
        self.sp_zz_desviacion.setDecimals(2)
        self.sp_zz_desviacion.setValue(5.0)
        self.sp_zz_desviacion.setSuffix(" %")
        self.sp_zz_desviacion.setToolTip(
            "ZigZag: movimiento mínimo que debe recorrer el precio para "
            "reconocer un cambio de tramo.")
        self.sp_zz_desviacion.valueChanged.connect(self._guardar_setup_actual)
        self.sp_zz_piernas = QSpinBox()
        self.sp_zz_piernas.setRange(2, 500)
        self.sp_zz_piernas.setValue(10)
        self.sp_zz_piernas.setToolTip(
            "ZigZag: velas totales que confirman un pivote, la mitad a cada "
            "lado. Cuantas más, más tarde se confirma el tramo (y más tarde "
            "puede colocarse la orden), pero menos pivotes de ruido.")
        self.sp_zz_piernas.valueChanged.connect(self._guardar_setup_actual)
        fila_fib.addWidget(QLabel("nivel"))
        fila_fib.addWidget(self.cmb_nivel_fib)
        fila_fib.addWidget(QLabel("ZigZag"))
        fila_fib.addWidget(self.sp_zz_desviacion)
        fila_fib.addWidget(self.sp_zz_piernas)
        fila_fib.addStretch()
        self._fila_fib_widget = QWidget()
        self._fila_fib_widget.setLayout(fila_fib)
        f_set.addRow("Retroceso:", self._fila_fib_widget)

        fila_vig = QHBoxLayout()
        self.sp_vigencia_limite = QSpinBox()
        self.sp_vigencia_limite.setRange(0, 5000)
        self.sp_vigencia_limite.setValue(0)
        self.sp_vigencia_limite.setSuffix(" velas")
        self.sp_vigencia_limite.setSpecialValueText("Sin caducidad")
        self.sp_vigencia_limite.setToolTip(
            "Cancela la orden si no se ha rellenado en N velas "
            "(0 = vive hasta rellenarse o hasta que un tramo nuevo la anule).")
        self.sp_vigencia_limite.valueChanged.connect(self._guardar_setup_actual)
        self.sp_cancelar_avance_r = QDoubleSpinBox()
        self.sp_cancelar_avance_r.setRange(0.0, 50.0)
        self.sp_cancelar_avance_r.setDecimals(1)
        self.sp_cancelar_avance_r.setValue(0.0)
        self.sp_cancelar_avance_r.setSuffix(" R")
        self.sp_cancelar_avance_r.setSpecialValueText("No cancelar")
        self.sp_cancelar_avance_r.setToolTip(
            "Cancela la orden si el precio avanza esta distancia a favor sin "
            "haberla tocado: el movimiento se fue sin nosotros y el retroceso "
            "ya no va a llegar a ese nivel.")
        self.sp_cancelar_avance_r.valueChanged.connect(self._guardar_setup_actual)
        fila_vig.addWidget(QLabel("caduca a"))
        fila_vig.addWidget(self.sp_vigencia_limite)
        fila_vig.addWidget(QLabel("o si avanza"))
        fila_vig.addWidget(self.sp_cancelar_avance_r)
        fila_vig.addStretch()
        self._fila_vigencia_widget = QWidget()
        self._fila_vigencia_widget.setLayout(fila_vig)
        f_set.addRow("Vida de la orden:", self._fila_vigencia_widget)

        # modo edge: probar la señal desnuda, sin stop ni TP
        self.btn_edge = QPushButton("⚡ Prueba de Ventaja (Edge)")
        self.btn_edge.setObjectName("edge")
        self.btn_edge.setCheckable(True)
        self.btn_edge.setToolTip(
            "<div style='color: #c8d6e5; font-size: 11px;'>"
            "<p style='color: #f1c40f; font-weight: bold; margin: 0 0 6px 0;'>"
            "⚡ Prueba de Ventaja (Edge)</p>"
            "<p style='margin: 0 0 6px 0;'>Mide el <b style='color: #4fc3f7;'>"
            "edge puro</b> de la señal: desactiva stop-loss y take-profit "
            "(se ponen a 0) para que cada trade abra con la señal de entrada "
            "y cierre <b>solo</b> con la señal de salida del indicador.</p>"
            "<p style='margin: 0 0 6px 0;'>Así el win rate, profit factor y "
            "expectancy reflejan la calidad de la señal sin que la gestión "
            "la rescate o la estropee.</p>"
            "<p style='color: #8fb3d9; margin: 0 0 6px 0;'>El ATR se sigue "
            "usando únicamente para dimensionar la posición (2×ATR de "
            "referencia, para que el riesgo % signifique algo). Riesgo % y "
            "salida por tiempo siguen editables.</p>"
            "<p style='color: #5a7a9a; margin: 0;'>Al desactivarlo se "
            "restauran el stop y el TP que tenías antes.</p>"
            "</div>")
        self.btn_edge.toggled.connect(self._on_edge_toggled)
        f_set.addRow(self.btn_edge)

        # riesgo/stop/TP/salida PROPIOS del setup
        self.sp_riesgo = QDoubleSpinBox()
        self.sp_riesgo.setRange(0.01, 100)
        self.sp_riesgo.setDecimals(2)
        self.sp_riesgo.setValue(1.0)
        self.sp_riesgo.setSuffix(" %")
        self.sp_riesgo.valueChanged.connect(self._guardar_setup_actual)
        _fila_con_ayuda(
            f_set, "Riesgo del setup:", self.sp_riesgo,
            "Cuánta cuenta te juegas en CADA operación de este setup.\n"
            "\n"
            "Para qué sirve: es la palanca que decide el tamaño de la\n"
            "posición. El motor calcula cuántas unidades comprar para que,\n"
            "si salta el Stop Loss, la pérdida sea exactamente este % del\n"
            "equity — así el riesgo no depende de lo volátil que esté el\n"
            "activo ni de lo lejos que quede el stop.\n"
            "\n"
            "Si el setup no lleva Stop Loss se mide contra 2×ATR de\n"
            "referencia, para que el % siga significando algo.")
        self.sp_stop = QDoubleSpinBox()
        self.sp_stop.setRange(0, 20)
        self.sp_stop.setDecimals(1)
        self.sp_stop.setValue(2.0)
        self.sp_stop.setSpecialValueText("Sin Stop Loss")
        self.sp_stop.setToolTip(
            "Distancia del Stop Loss en multiplos del ATR. 0 = sin stop real.")
        self.sp_stop.valueChanged.connect(self._guardar_setup_actual)
        self.sp_stop.valueChanged.connect(self._actualizar_modo_stop_habilitado)
        self.cmb_stop_modo = QComboBox()
        self.cmb_stop_modo.addItems(list(_MAPA_STOP_MODO))
        self.cmb_stop_modo.setToolTip(
            "Modo del Stop Loss × ATR. Fijo ancla el stop a la primera "
            "entrada; Dinámico por promedio lo reancla al precio medio tras "
            "cada entrada. Solo se puede usar con un Stop Loss configurado.")
        self.cmb_stop_modo.currentTextChanged.connect(self._guardar_setup_actual)
        fila_stop = QHBoxLayout()
        fila_stop.setContentsMargins(0, 0, 0, 0)
        fila_stop.addWidget(self.sp_stop, 1)
        fila_stop.addWidget(self.cmb_stop_modo)
        self._stop_row = QWidget()
        self._stop_row.setLayout(fila_stop)
        _fila_con_ayuda(
            f_set, "Stop Loss (× ATR):", self._stop_row,
            "Dónde se corta la pérdida: la operación se cierra si el precio\n"
            "va en contra esta distancia, medida en múltiplos del ATR (la\n"
            "volatilidad típica de la vela).\n"
            "\n"
            "Para qué sirve: dos cosas. Acota lo máximo que puede costarte\n"
            "una operación, y define la unidad «R» — la distancia de riesgo—\n"
            "con la que se miden el Take Profit, el R múltiple de cada trade\n"
            "y la expectativa del sistema.\n"
            "\n"
            "En ×ATR y no en precio fijo para que el stop se adapte solo: en\n"
            "un tramo tranquilo queda cerca y en uno movido, lejos.\n"
            "0 = sin Stop Loss (la operación solo cierra por señal, Take\n"
            "Profit o tiempo).")
        self.sp_tp = QDoubleSpinBox()
        self.sp_tp.setRange(0, 20)
        self.sp_tp.setDecimals(1)
        self.sp_tp.setValue(0.0)
        self.sp_tp.setSpecialValueText("Sin Take Profit")
        self.sp_tp.valueChanged.connect(self._guardar_setup_actual)
        _fila_con_ayuda(
            f_set, "Take Profit (R):", self.sp_tp,
            "Objetivo de beneficio al que se cierra la operación, en\n"
            "múltiplos de R (tu distancia de riesgo, o sea el Stop Loss).\n"
            "2 R = ganar el doble de lo que arriesgabas.\n"
            "\n"
            "Para qué sirve: recoge el beneficio sin depender de que llegue\n"
            "la señal de salida. Sube el win rate y acorta las operaciones,\n"
            "pero también corta las que se habrían ido mucho más lejos: es\n"
            "el contrapeso natural del Trailing Stop.\n"
            "\n"
            "0 = sin Take Profit, se deja correr hasta la señal de salida.")
        self.sp_tiempo = QSpinBox()
        self.sp_tiempo.setRange(0, 10000)
        self.sp_tiempo.setValue(0)
        self.sp_tiempo.setSpecialValueText("Sin límite")
        self.sp_tiempo.valueChanged.connect(self._guardar_setup_actual)
        _fila_con_ayuda(
            f_set, "Salida por tiempo (velas):", self.sp_tiempo,
            "Cierra la posición N velas después de entrar, esté como esté.\n"
            "\n"
            "Para qué sirve: evita quedarte atrapado en operaciones que no\n"
            "se resuelven ni a favor ni en contra, comiendo capital y\n"
            "exposición. Es lo natural en señales con fecha de caducidad —\n"
            "un patrón de velas dice algo sobre las próximas velas, no sobre\n"
            "las próximas semanas.\n"
            "\n"
            "0 = sin límite. En Patrones de velas, 0 usa el lag de salida\n"
            "del propio patrón.")

        # ── gestión de posición: break-even + trailing stop ──
        self.sp_be = QDoubleSpinBox()
        self.sp_be.setRange(0, 10)
        self.sp_be.setDecimals(1)
        self.sp_be.setValue(0.0)
        self.sp_be.setSpecialValueText("Sin Break Even")
        self.sp_be.valueChanged.connect(self._guardar_setup_actual)
        self.cmb_be_unidad = QComboBox()
        self.cmb_be_unidad.addItems(list(_MAPA_BE_UNIDAD))
        self.cmb_be_unidad.setToolTip(
            "Unidad de la distancia de activación:\n"
            "· × ATR — múltiplos del ATR de la vela (comportamiento histórico).\n"
            "· R — múltiplos de tu distancia de riesgo real (stop × ATR), la\n"
            "  misma vara que el disparador «R:R alcanzado» de las salidas.\n"
            "Solo coinciden si el stop está a 1×ATR: con el stop a 2×ATR,\n"
            "«1×ATR de avance» es 0.5R, no 1R.")
        self.cmb_be_unidad.currentTextChanged.connect(self._guardar_setup_actual)
        fila_be = QHBoxLayout()
        fila_be.addWidget(self.sp_be)
        fila_be.addWidget(self.cmb_be_unidad)
        AYUDA_BE = (
            "Mueve el Stop Loss al precio de entrada en cuanto la operación\n"
            "avanza a tu favor esta distancia. La unidad la eliges al lado.\n"
            "\n"
            "Para qué sirve: a partir de ese momento la operación ya no\n"
            "puede perder — como mucho sale en tablas. Convierte un trade\n"
            "con riesgo en un trade gratis.\n"
            "\n"
            "El precio a pagar: un retroceso normal que antes se habría\n"
            "recuperado ahora te saca en el punto de entrada. Cuanto más\n"
            "pronto lo actives, más operaciones buenas cortarás de raíz.\n"
            "\n"
            "0 = sin Break Even, el stop se queda donde estaba.")
        # el bocadillo NO va al combo de al lado: ese explica otra cosa
        # (× ATR frente a R) y pisarlo sería perder información
        self.sp_be.setToolTip(AYUDA_BE)
        _fila_con_ayuda(f_set, "Break Even:", fila_be, AYUDA_BE)
        self.sp_trailing = QDoubleSpinBox()
        self.sp_trailing.setRange(0, 10)
        self.sp_trailing.setDecimals(1)
        self.sp_trailing.setValue(0.0)
        self.sp_trailing.setSpecialValueText("Sin Trailing Stop")
        self.sp_trailing.valueChanged.connect(self._guardar_setup_actual)
        _fila_con_ayuda(
            f_set, "Trailing Stop (× ATR):", self.sp_trailing,
            "El Stop Loss persigue al precio a esta distancia (×ATR) del\n"
            "mejor nivel alcanzado. Solo se mueve a favor, nunca en contra.\n"
            "\n"
            "Para qué sirve: dejar correr una tendencia sin renunciar a lo\n"
            "ya ganado. En vez de fijar de antemano dónde recoges —eso es el\n"
            "Take Profit—, dejas que sea el mercado quien te saque cuando se\n"
            "gire.\n"
            "\n"
            "Cuanto más corta la distancia, antes te saca un simple respiro\n"
            "del precio; cuanto más larga, más devuelves antes de salir.\n"
            "0 = sin Trailing Stop.")

        # ── entradas ──
        grp_entradas = QGroupBox("Entrada del setup")
        lay_ent = QVBoxLayout(grp_entradas)
        grp_entradas.setToolTip(
            "Arriba, la señal de la plantilla: lo que dispara la entrada por "
            "sí solo (el cruce, la ruptura de banda...). Sus celdas activas "
            "editan los parámetros de la plantilla.\n"
            "Debajo, tus condiciones extra: se exigen ADEMÁS de la señal (AND) "
            "y solo restringen entradas nuevas, nunca cierran una posición.")
        lay_ent.insertLayout(0, _fila_ayuda(grp_entradas.toolTip()))
        self.editor_cond_entrada = EditorCondiciones("")
        self.editor_cond_entrada.tabla.cellChanged.connect(
            lambda *_: self._guardar_setup_actual())
        # insertRow/removeRow (+ Condición / − Quitar) no emiten cellChanged
        # — sin esto, añadir/quitar una fila no queda guardado (ni, por
        # tanto, capturado por deshacer) hasta que otra edición lo dispare
        self.editor_cond_entrada.tabla.model().rowsInserted.connect(
            lambda *_: self._guardar_setup_actual())
        self.editor_cond_entrada.tabla.model().rowsRemoved.connect(
            lambda *_: self._guardar_setup_actual())
        self.btn_deshacer_entrada = self._crear_boton_deshacer()
        # insertar ANTES del stretch (índice 2: tras +Condición/−Quitar), no
        # al final — si no, el botón queda empujado al extremo derecho en
        # vez de pegado a "− Quitar" como en Entrada escalonada/Salida
        self.editor_cond_entrada.fila_botones.insertWidget(2, self.btn_deshacer_entrada)
        lay_ent.addWidget(self.editor_cond_entrada)
        self.lbl_resumen_entrada = QLabel("")
        self.lbl_resumen_entrada.setWordWrap(True)
        self.lbl_resumen_entrada.setStyleSheet(
            "color: #8fb3d9; font-size: 10px; padding-top: 2px;")
        lay_ent.addWidget(self.lbl_resumen_entrada)

        # ── entrada escalonada: construir la posicion en varios promedios ──
        grp_tramos = QGroupBox("Entrada por promedio")
        grp_tramos.setToolTip(
            "Construye la posición en varios promedios: por velas, promediando a la\n"
            "baja, o piramidando a favor.\n"
            "• «Entrar %» = % del RIESGO TOTAL (no del tamaño); cada promedio se\n"
            "  dimensiona contra el stop vigente en ese momento.\n"
            "• Por defecto: un promedio, 100% a la señal (igual que sin promedios).\n"
            "Ejemplo: Promedio 1 50% a la senal → Promedio 2 50% si retrocede 1×ATR.")
        lay_tram = QVBoxLayout(grp_tramos)
        lay_tram.insertLayout(0, _fila_ayuda(grp_tramos.toolTip()))
        self.tabla_tramos = QTableWidget(0, 4)
        self.tabla_tramos.setHorizontalHeaderLabels(
            ["Entrar %", "Disparador", "Condiciones", "Gestión"])
        self.tabla_tramos.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        _activar_borde_fila(self.tabla_tramos)
        _instalar_borrado_fila(self.tabla_tramos, self._del_tramo)
        self.tabla_tramos.verticalHeader().setDefaultSectionSize(30)
        # sin límite de altura fijo: ver _ajustar_alto_tabla, llamado al
        # recargar la tabla en _cargar_tramos_tabla
        self.tabla_tramos.cellChanged.connect(self._guardar_setup_actual)
        lay_tram.addWidget(self.tabla_tramos)
        fila_t = QHBoxLayout()
        for texto, slot in [("+ Promedio", self._add_tramo),
                            ("− Quitar", self._del_tramo)]:
            b = QPushButton(texto)
            b.clicked.connect(slot)
            fila_t.addWidget(b)
        self.btn_deshacer_tramos = self._crear_boton_deshacer()
        fila_t.addWidget(self.btn_deshacer_tramos)
        fila_t.addStretch()
        lay_tram.addLayout(fila_t)
        self.lbl_resumen_tramos = QLabel("")
        self.lbl_resumen_tramos.setWordWrap(True)
        self.lbl_resumen_tramos.setStyleSheet(
            "color: #8fb3d9; font-size: 10px; padding-top: 2px;")
        self.lbl_resumen_tramos.setVisible(False)
        lay_tram.addWidget(self.lbl_resumen_tramos)
        self.lbl_aviso_tramos = QLabel("")
        self.lbl_aviso_tramos.setWordWrap(True)
        self.lbl_aviso_tramos.setStyleSheet(
            "color: #e74c3c; font-size: 10px; font-weight: bold; padding-top: 2px;")
        self.lbl_aviso_tramos.setVisible(False)
        lay_tram.addWidget(self.lbl_aviso_tramos)
        lay_ent.addWidget(grp_tramos)

        f_set.addRow(grp_entradas)

        # ── salida del setup ──
        grp_parciales = QGroupBox("Salida del setup")
        grp_parciales.setToolTip(
            "Reparte el cierre en varias salidas: cada una cierra un % del TAMAÑO\n"
            "ORIGINAL de la posición, con su propio disparador (señal, R:R,\n"
            "condiciones o estancamiento) y gestión (BE/trailing) sobre el resto.\n"
            "Debajo, si están activos, también aparecen Stop/TP/BE/Trailing/Tiempo:\n"
            "esos cierran % de lo que QUEDE, no del tamaño original.\n"
            "Por defecto: una salida, 100% a la señal (igual que sin configurar nada).\n"
            "Ejemplo: Salida 1 50% a la señal → Salida 2 50% a 2R + BE.")
        lay_parc = QVBoxLayout(grp_parciales)
        lay_parc.insertLayout(0, _fila_ayuda(grp_parciales.toolTip()))
        self.tabla_parciales = QTableWidget(0, 4)
        self.tabla_parciales.setHorizontalHeaderLabels(
            ["Cerrar %", "Disparador", "Condiciones", "Gestión"])
        self.tabla_parciales.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        _activar_borde_fila(self.tabla_parciales)
        _instalar_borrado_fila(self.tabla_parciales, self._del_etapa_parcial)
        self.tabla_parciales.verticalHeader().setDefaultSectionSize(30)
        # sin límite de altura fijo: ver _ajustar_alto_tabla, llamado al
        # recargar la tabla en _cargar_parciales_tabla (tras pintar también
        # las filas de mecanismo)
        self.tabla_parciales.cellChanged.connect(self._guardar_setup_actual)
        lay_parc.addWidget(self.tabla_parciales)
        fila_p = QHBoxLayout()
        for texto, slot in [("+ Salida", self._add_etapa_parcial),
                            ("− Quitar", self._del_etapa_parcial)]:
            b = QPushButton(texto)
            b.clicked.connect(slot)
            if texto == "− Quitar":
                b.setToolTip(
                    "Borra la etapa de salida seleccionada, o — si seleccionas "
                    "la fila de Stop/TP/Break-even/Trailing/Tiempo — desactiva "
                    "ese mecanismo (lo pone a 0, igual que en su campo de arriba).")
            fila_p.addWidget(b)
        self.btn_deshacer_parciales = self._crear_boton_deshacer()
        fila_p.addWidget(self.btn_deshacer_parciales)
        fila_p.addStretch()
        QShortcut(QKeySequence("Ctrl+Z"), self, self._deshacer)
        lay_parc.addLayout(fila_p)
        self.lbl_resumen_parciales = QLabel("")
        self.lbl_resumen_parciales.setWordWrap(True)
        self.lbl_resumen_parciales.setStyleSheet(
            "color: #8fb3d9; font-size: 10px; padding-top: 2px;")
        self.lbl_resumen_parciales.setVisible(False)
        lay_parc.addWidget(self.lbl_resumen_parciales)
        self.lbl_aviso_parciales = QLabel("")
        self.lbl_aviso_parciales.setWordWrap(True)
        self.lbl_aviso_parciales.setStyleSheet(
            "color: #e74c3c; font-size: 10px; font-weight: bold; padding-top: 2px;")
        self.lbl_aviso_parciales.setVisible(False)
        lay_parc.addWidget(self.lbl_aviso_parciales)
        f_set.addRow(grp_parciales)

        # etiquetas de stop/TP: se oscurecen junto al campo en modo edge
        self._lbl_stop = f_set.labelForField(self._stop_row)
        self._lbl_tp = f_set.labelForField(self.sp_tp)

        # ── filtros de entrada del setup (no afectan a las salidas) ──
        grp_filtros = QGroupBox("Filtros del setup")
        grp_filtros.setToolTip(
            "Día/régimen/volatilidad/sesión restringen cuándo puede abrirse una posición "
            "NUEVA (nunca cierran una ya abierta). Las condiciones extra de "
            "entrada/salida, más abajo, sí pueden aplicarse también a la "
            "salida si así se configuran.")
        f_filtros = QFormLayout(grp_filtros)
        _insertar_ayuda_form(f_filtros, grp_filtros.toolTip())

        fila_dias = QHBoxLayout()
        self._chk_dias = []
        for etiqueta_dia in ['L', 'M', 'X', 'J', 'V', 'S', 'D']:
            chk = QCheckBox(etiqueta_dia)
            chk.setChecked(True)
            chk.toggled.connect(self._guardar_setup_actual)
            fila_dias.addWidget(chk)
            self._chk_dias.append(chk)
        f_filtros.addRow("Día de la semana:", fila_dias)

        self.cmb_regimen = QComboBox()
        self.cmb_regimen.addItems(list(_MAPA_REGIMEN))
        self.cmb_regimen.currentTextChanged.connect(self._on_regimen_changed)
        f_filtros.addRow("Régimen:", self.cmb_regimen)
        self.sp_regimen_periodo = QSpinBox()
        self.sp_regimen_periodo.setRange(5, 5000)
        self.sp_regimen_periodo.setValue(VENTANA_ER_DEFECTO)
        self.sp_regimen_periodo.setToolTip(
            "Ventana rodante con la que se calcula el ER o el Hurst de cada "
            "vela.\n"
            f"· ER: {VENTANA_ER_DEFECTO} (Kaufman). Con ventanas largas el ER "
            "decae y casi\n"
            f"  ninguna vela llega a {UMBRAL_ER_TENDENCIA}, así que el filtro "
            "de tendencia bloquea todo.\n"
            f"· Hurst: {VENTANA_HURST_DEFECTO}, y nunca por debajo de "
            f"{VENTANA_HURST_MINIMA} — con menos, el estimador\n"
            f"  supera {UMBRAL_HURST_TENDENCIA} incluso sobre ruido puro y el "
            "régimen que marca no es fiable.")
        self.sp_regimen_periodo.valueChanged.connect(self._guardar_setup_actual)
        f_filtros.addRow("Periodo (ventana):", self.sp_regimen_periodo)

        self.cmb_volatilidad = QComboBox()
        self.cmb_volatilidad.addItems(list(_MAPA_VOLATILIDAD))
        self.cmb_volatilidad.setToolTip(
            "Solo se abren posiciones NUEVAS cuando la volatilidad de la vela "
            "está en el tramo elegido de su propia historia reciente. El corte "
            "es un PERCENTIL, no un valor absoluto: 'ATR alto ≥ 70' significa "
            "«solo el 30% de velas más volátiles de la ventana», y vale igual "
            "para cualquier activo y timeframe sin recalibrar.")
        self.cmb_volatilidad.currentTextChanged.connect(self._on_volatilidad_changed)
        f_filtros.addRow("Volatilidad:", self.cmb_volatilidad)
        fila_vol = QHBoxLayout()
        self.sp_volatilidad_periodo = QSpinBox()
        self.sp_volatilidad_periodo.setRange(10, 5000)
        self.sp_volatilidad_periodo.setValue(100)
        self.sp_volatilidad_periodo.setToolTip(
            "Histórico contra el que se compara la volatilidad actual. El "
            "indicador (ATR o desviación estándar) se calcula siempre sobre 14 "
            "velas; esta ventana solo decide cuánto pasado se usa para situar "
            "su percentil.")
        self.sp_volatilidad_percentil = QDoubleSpinBox()
        self.sp_volatilidad_percentil.setRange(0.0, 100.0)
        self.sp_volatilidad_percentil.setDecimals(1)
        self.sp_volatilidad_percentil.setValue(50.0)
        self.sp_volatilidad_periodo.valueChanged.connect(self._guardar_setup_actual)
        self.sp_volatilidad_percentil.valueChanged.connect(self._guardar_setup_actual)
        fila_vol.addWidget(QLabel("ventana"))
        fila_vol.addWidget(self.sp_volatilidad_periodo)
        fila_vol.addWidget(QLabel("percentil"))
        fila_vol.addWidget(self.sp_volatilidad_percentil)
        fila_vol.addStretch()
        self._fila_volatilidad_widget = QWidget()
        self._fila_volatilidad_widget.setLayout(fila_vol)
        f_filtros.addRow("Umbral:", self._fila_volatilidad_widget)

        self.cmb_sesion = QComboBox()
        self.cmb_sesion.addItems(list(_MAPA_SESION))
        self.cmb_sesion.setToolTip(
            "Solo se abren posiciones NUEVAS mientras la vela cae dentro de "
            "la sesión elegida — fuera de ese horario no se abre nada (no es "
            "un filtro para EVITAR esa sesión). 'Ninguna' desactiva el "
            "filtro y permite operar a cualquier hora.")
        self.cmb_sesion.currentTextChanged.connect(self._on_sesion_changed)
        f_filtros.addRow("Sesión horaria:", self.cmb_sesion)
        fila_horas = QHBoxLayout()
        self.sp_hora_ini = QSpinBox()
        self.sp_hora_ini.setRange(0, 23)
        self.sp_hora_fin = QSpinBox()
        self.sp_hora_fin.setRange(0, 23)
        self.sp_hora_ini.valueChanged.connect(self._guardar_setup_actual)
        self.sp_hora_fin.valueChanged.connect(self._guardar_setup_actual)
        fila_horas.addWidget(QLabel("de"))
        fila_horas.addWidget(self.sp_hora_ini)
        fila_horas.addWidget(QLabel("a"))
        fila_horas.addWidget(self.sp_hora_fin)
        fila_horas.addWidget(QLabel("(hora UTC)"))
        fila_horas.addStretch()
        self._fila_horas_widget = QWidget()
        self._fila_horas_widget.setLayout(fila_horas)
        f_filtros.addRow("Horas:", self._fila_horas_widget)
        self._lbl_horas = f_filtros.labelForField(self._fila_horas_widget)
        if self._lbl_horas is not None:
            self._lbl_horas.setVisible(False)

        # ── filtro de tendencia de TF superior (multi-timeframe) ──
        self.cmb_tf_sup_indicador = QComboBox()
        self.cmb_tf_sup_indicador.addItems(list(_MAPA_TF_SUPERIOR))
        self.cmb_tf_sup_indicador.setToolTip(
            "Solo se abren posiciones NUEVAS cuando el precio está alineado "
            "con la tendencia de un timeframe superior: la serie se "
            "resamplea al TF elegido y se compara el cierre de cada vela "
            "con su media (SMA/EMA) o Supertrend de ese TF. Sin lookahead: "
            "solo se usa la vela del TF superior ya cerrada.")
        self.cmb_tf_sup_indicador.currentTextChanged.connect(
            self._on_tf_sup_changed)
        f_filtros.addRow("Tendencia TF sup.:", self.cmb_tf_sup_indicador)
        fila_tfs = QHBoxLayout()
        self.cmb_tf_sup_tf = QComboBox()
        self.cmb_tf_sup_tf.addItems(TF_LABELS)
        self.cmb_tf_sup_tf.setCurrentText('1h')
        self.cmb_tf_sup_tf.setToolTip(
            "Timeframe superior al de las velas del activo; si no es mayor, "
            "el filtro se ignora con aviso.")
        self.sp_tf_sup_periodo = QSpinBox()
        self.sp_tf_sup_periodo.setRange(2, 5000)
        self.sp_tf_sup_periodo.setValue(200)
        self.sp_tf_sup_periodo.setToolTip(
            "Periodo del indicador calculado sobre el TF superior (p. ej. "
            "SMA 200 de 1h filtrando velas de 5m).")
        self.cmb_tf_sup_relacion = QComboBox()
        self.cmb_tf_sup_relacion.addItems(list(_MAPA_RELACION_TF))
        self.cmb_tf_sup_relacion.setToolTip(
            "'Ambos': largos solo con precio ≥ tendencia y cortos solo con "
            "precio ≤ tendencia. 'Solo largos'/'Solo cortos' aplican la "
            "alineación a un único lado.")
        self.cmb_tf_sup_tf.currentTextChanged.connect(self._guardar_setup_actual)
        self.sp_tf_sup_periodo.valueChanged.connect(self._guardar_setup_actual)
        self.cmb_tf_sup_relacion.currentTextChanged.connect(self._guardar_setup_actual)
        fila_tfs.addWidget(QLabel("TF"))
        fila_tfs.addWidget(self.cmb_tf_sup_tf)
        fila_tfs.addWidget(QLabel("periodo"))
        fila_tfs.addWidget(self.sp_tf_sup_periodo)
        fila_tfs.addWidget(QLabel("relación"))
        fila_tfs.addWidget(self.cmb_tf_sup_relacion)
        fila_tfs.addStretch()
        self._fila_tf_sup_widget = QWidget()
        self._fila_tf_sup_widget.setLayout(fila_tfs)
        f_filtros.addRow("Parámetros:", self._fila_tf_sup_widget)
        self._lbl_tf_sup = f_filtros.labelForField(self._fila_tf_sup_widget)
        if self._lbl_tf_sup is not None:
            self._lbl_tf_sup.setVisible(False)

        self.chk_noticias = QCheckBox("Evitar noticias")
        self.chk_noticias.setToolTip(
            "Próximamente: el calendario económico está en revisión (sin "
            "proveedor gratuito fiable). Esta opción volverá cuando se "
            "estabilice la fuente de datos.")
        self.chk_noticias.setEnabled(False)
        f_filtros.addRow(self.chk_noticias)

        fila_noticias = QHBoxLayout()
        self.sp_noticias_antes = QSpinBox()
        self.sp_noticias_antes.setRange(0, 1440)
        self.sp_noticias_antes.setValue(30)
        self.sp_noticias_antes.setSuffix(" min")
        self.sp_noticias_después = QSpinBox()
        self.sp_noticias_después.setRange(0, 1440)
        self.sp_noticias_después.setValue(30)
        self.sp_noticias_después.setSuffix(" min")
        self.sp_noticias_antes.valueChanged.connect(self._guardar_setup_actual)
        self.sp_noticias_después.valueChanged.connect(self._guardar_setup_actual)
        fila_noticias.addWidget(QLabel("antes"))
        fila_noticias.addWidget(self.sp_noticias_antes)
        fila_noticias.addWidget(QLabel("después"))
        fila_noticias.addWidget(self.sp_noticias_después)
        fila_noticias.addWidget(QLabel("impacto mínimo"))
        self.cmb_noticias_impacto = QComboBox()
        self.cmb_noticias_impacto.addItems(['Bajo', 'Medio', 'Alto'])
        self.cmb_noticias_impacto.setCurrentText('Alto')
        self.cmb_noticias_impacto.currentTextChanged.connect(self._on_noticias_impacto_changed)
        fila_noticias.addWidget(self.cmb_noticias_impacto)
        self.chk_noticias_cerrar = QCheckBox("Cerrar posiciones abiertas antes de la noticia")
        self.chk_noticias_cerrar.setToolTip(
            "Próximamente: el calendario económico está en revisión.")
        self.chk_noticias_cerrar.setEnabled(False)
        fila_noticias.addWidget(self.chk_noticias_cerrar)
        fila_noticias.addStretch()
        self._fila_noticias_widget = QWidget()
        self._fila_noticias_widget.setLayout(fila_noticias)
        f_filtros.addRow("Ventana:", self._fila_noticias_widget)
        self.cmb_noticias_impacto.setToolTip(economic_calendar.DESCRIPCION_IMPACTO['alto'])

        f_set.addRow(grp_filtros)
        lay.addWidget(self.grp_setup)
        bombear_eventos()

        # ── código del sistema (siempre visible, se regenera en vivo) ──
        grp_cod = QGroupBox("Código del sistema (variables, entrada y salida de cada setup)")
        lay_cod = QVBoxLayout(grp_cod)
        lay_cod.insertLayout(0, _fila_ayuda(
            "Pseudocódigo generado automáticamente a partir de la "
            "configuración actual: variables de cada setup, condición de "
            "entrada, condición de salida y todas las reglas adicionales "
            "(stop, TP, parciales, filtros). Se actualiza en vivo — sirve "
            "para verificar que el sistema hace lo que crees que hace, "
            "antes de correrlo."))
        from PyQt6.QtWidgets import QPlainTextEdit
        self.txt_codigo = QPlainTextEdit()
        self.txt_codigo.setReadOnly(True)
        self.txt_codigo.setStyleSheet(
            "QPlainTextEdit { background-color: #0d1424; color: #8fb3d9;"
            "border: 1px solid #253a60; border-radius: 4px;"
            "font-family: Consolas, monospace; font-size: 11px; }")
        self.txt_codigo.setMinimumHeight(160)
        lay_cod.addWidget(self.txt_codigo)
        lay.addWidget(grp_cod)

        # ── cuenta (global, no del setup) ──
        grp_ej = QGroupBox("Cuenta")
        f_ej = QFormLayout(grp_ej)
        _insertar_ayuda_form(f_ej,
            "Configuración global de la cuenta simulada: capital inicial, "
            "comisión por lado y slippage. Se aplica igual a todos los "
            "setups del sistema.")
        self.sp_capital = QDoubleSpinBox()
        self.sp_capital.setRange(100, 1e9)
        self.sp_capital.setValue(10000)
        self.sp_capital.setDecimals(0)
        f_ej.addRow("Balance inicial:", self.sp_capital)
        self.sp_comision = QDoubleSpinBox()
        self.sp_comision.setRange(0, 1)
        self.sp_comision.setDecimals(4)
        self.sp_comision.setValue(0.05)
        self.sp_comision.setSuffix(" %")
        self.sp_comision.setToolTip(
            "Comisión por lado como % del nocional. Al elegir un activo se "
            "prellena según su clase (aproximado, por lado): crypto 0.03%, "
            "stock 0.07%, futuro-CFD y forex 0.03% — ajústalo a tu broker")
        f_ej.addRow("Comisión (por lado):", self.sp_comision)
        self.sp_slippage = QDoubleSpinBox()
        self.sp_slippage.setRange(0, 1)
        self.sp_slippage.setDecimals(4)
        self.sp_slippage.setValue(0.02)
        self.sp_slippage.setSuffix(" %")
        self.sp_slippage.setToolTip(
            "Deslizamiento de precio por lado como % (encarece la entrada y "
            "abarata la salida). Al elegir un activo se prellena según su "
            "clase (aproximado): crypto 0.10%, stock 0.07%, futuro-CFD y "
            "forex 0.02% — dentro de cada clase hay más y menos líquidos")
        f_ej.addRow("Slippage:", self.sp_slippage)
        self.lbl_friccion = QLabel("")
        self.lbl_friccion.setObjectName("estado")
        self.lbl_friccion.setWordWrap(True)
        f_ej.addRow(self.lbl_friccion)
        lay.addWidget(grp_ej)

        # ── IS/OOS + WFA ──
        grp_split = QGroupBox("Muestra (In-Sample / Out-Of-Sample)")
        lay_sp = QVBoxLayout(grp_split)
        lay_sp.insertLayout(0, _fila_ayuda(
            "Divide la serie histórica en dos tramos: In-Sample (donde "
            "ajustas el sistema) y Out-of-Sample (donde lo validas sin "
            "haberlo tocado). El slider cambia qué % del final de la serie "
            "se reserva como OOS."))
        self.lbl_split = QLabel("IS 70% / OOS 30%")
        self.lbl_split.setObjectName("campo")
        self.slider_oos = QSlider(Qt.Orientation.Horizontal)
        self.slider_oos.setRange(5, 50)         # % OOS
        self.slider_oos.setValue(30)
        self.slider_oos.valueChanged.connect(
            lambda v: self.lbl_split.setText(f"IS {100 - v}% / OOS {v}%"))
        lay_sp.addWidget(self.lbl_split)
        lay_sp.addWidget(self.slider_oos)
        fila_wfa = QHBoxLayout()
        self.chk_wfa = QCheckBox("Walk-Forward Analysis")
        self.chk_wfa.setChecked(True)
        self.chk_wfa.setToolTip(
            "Divide la serie en N ventanas consecutivas y calcula las "
            "métricas de cada una por separado: mide si el sistema es "
            "estable en el tiempo o solo funcionó en una época")
        self.sp_wfa = QSpinBox()
        self.sp_wfa.setRange(2, 20)
        self.sp_wfa.setValue(5)
        self.sp_wfa.setPrefix("ventanas: ")
        fila_wfa.addWidget(self.chk_wfa)
        fila_wfa.addWidget(self.sp_wfa)
        fila_wfa.addStretch()
        lay_sp.addLayout(fila_wfa)
        lay.addWidget(grp_split)

        self.btn_run = QPushButton("▶  Ejecutar backtest")
        self.btn_run.setObjectName("run")
        self.btn_run.setEnabled(False)
        self.btn_run.setToolTip(
            "Corre esta configuración fija sobre toda la serie (IS+OOS) y "
            "muestra el detalle completo en Resultados")
        self.btn_run.clicked.connect(self.ejecutar.emit)
        lay.addWidget(self.btn_run)

        self.btn_optimizar = QPushButton("🔍  Prueba de parametrización (Solo IS)")
        self.btn_optimizar.setEnabled(False)
        self.btn_optimizar.setToolTip(
            "Prueba muchas combinaciones de parámetros del setup seleccionado, "
            "simulando cada una SOLO sobre el tramo In-Sample, y las compara "
            "en la pestaña «Optimizador»")
        self.btn_optimizar.clicked.connect(self.optimizar.emit)
        lay.addWidget(self.btn_optimizar)

        self.lbl_estado = QLabel("")
        self.lbl_estado.setObjectName("estado")
        lay.addWidget(self.lbl_estado)

        self.progreso = ProgressBarAnimada()
        self.progreso.setRange(0, 0)
        self.progreso.setTextVisible(False)
        self.progreso.setFixedHeight(6)
        self.progreso.setVisible(False)
        lay.addWidget(self.progreso)
        lay.addStretch()

        scroll.setWidget(cont)
        splitter.addWidget(scroll)
        splitter.setSizes([260, 760])
        splitter.setStretchFactor(1, 1)
        bombear_eventos()

        self._param_widgets = {}
        self._refresh_lista(seleccionar=0)
        self._recargar_guardadas()
        self._recargar_favoritos()
        # el código incluye las variables de cuenta: refrescarlo si cambian
        for w in (self.sp_capital, self.sp_comision, self.sp_slippage):
            w.valueChanged.connect(lambda *_: self._agendar_refresco())
        self.slider_oos.valueChanged.connect(lambda *_: self._agendar_refresco())
        self._refresh_codigo()

    # ── activo ──
    def _set_overlay(self, overlay):
        """Referencia al LoadingOverlay de la ventana principal (si existe)
        para tapar la app con spinner+progreso mientras se cargan archivos,
        se ejecuta el backtest o se optimiza."""
        self._overlay = overlay

    def _overlay_begin(self, texto, con_barra=False):
        ov = getattr(self, '_overlay', None)
        if ov is not None:
            ov.begin(texto, con_barra=con_barra)

    def _overlay_end(self):
        ov = getattr(self, '_overlay', None)
        if ov is not None:
            ov.end()

    def _overlay_set_progreso(self, i, total):
        ov = getattr(self, '_overlay', None)
        if ov is not None:
            ov.set_progreso(i, total)

    @_no_crash
    def _on_file(self, path):
        if not path.lower().endswith('.csv'):
            return
        self.csv_path = path
        m = TF_PATTERN.search(os.path.basename(path))
        self._tf_nativo = (m.group(1) or m.group(2)) if m else None
        self.lbl_activo.setTextFormat(Qt.TextFormat.RichText)
        self.lbl_activo.setText(_titulo_activo_html(path, self._tf_nativo))
        # rueda de carga DENTRO de la pestaña mientras se lee el CSV en hilo;
        # diferida 250 ms para no parpadear con archivos que leen al instante
        self._pendiente_overlay_carga = f"Cargando {os.path.basename(path)}…"
        QTimer.singleShot(250, self._mostrar_overlay_carga_diferido)
        # días disponibles: se leen en un hilo para no congelar la UI al
        # seleccionar un CSV grande; mientras tanto no se restringe nada
        self._dias_disponibles = set(range(7))
        self._actualizar_dias_checkboxes()
        self._configurar_botones_tf()
        self._aplicar_preset_friccion(path)
        self._actualizar_boton_ejecutable()
        self._actualizar_visibilidad_filtros()
        self._lanzar_detectar_dias(path)

    def _mostrar_overlay_carga_diferido(self):
        """Muestra la rueda solo si la lectura del CSV sigue en marcha."""
        texto = getattr(self, '_pendiente_overlay_carga', None)
        self._pendiente_overlay_carga = None
        th = getattr(self, '_detectar_thread', None)
        if texto and th is not None and th.isRunning():
            mostrar_overlay_tab(self, texto)

    def _lanzar_detectar_dias(self, path):
        self._detectar_gen = getattr(self, '_detectar_gen', 0) + 1
        gen = self._detectar_gen
        th = _DetectarDiasThread(path, parent=self)
        th.terminado.connect(
            lambda dias, g=gen: self._on_dias_detectados(dias, g))
        self._detectar_thread = th
        th.start()

    def _on_dias_detectados(self, dias, gen):
        if gen != getattr(self, '_detectar_gen', 0):
            return   # resultado de una selección anterior: se descarta
        self._detectar_thread = None
        self._pendiente_overlay_carga = None
        self._dias_disponibles = dias
        self._actualizar_dias_checkboxes()
        ocultar_overlay_tab(self)

    def _apagar_hilos(self):
        """Espera a que termine la detección de días en vuelo antes de cerrar
        la ventana (si no, Qt destruiría un QThread vivo → abort del proceso).
        wait() bloqueante: bombear eventos en mitad del cierre corrompe el
        montón (el finished del hilo pinta un árbol parcialmente destruido)."""
        th = getattr(self, '_detectar_thread', None)
        if th is not None:
            try:
                th.requestInterruption()
                th.wait()
            except Exception:
                pass
            self._detectar_thread = None
            # destruir el objeto del hilo MIENTRAS el árbol sigue vivo (si se
            # deja para el teardown del intérprete, Qt aborta por corrupción)
            try:
                th.deleteLater()
            except Exception:
                pass
        # Entrega única de eventos pendientes (terminado del hilo + deleteLater)
        # con el árbol vivo — si se dejan para el teardown, Qt los entrega sobre
        # widgets ya destruidos.
        QApplication.processEvents()
        apagar_overlay_tab(self)

    def _validar_sistema(self):
        """(válido, motivo) del sistema COMPLETO: recorre todos los setups, no
        solo el abierto en el editor, porque el backtest los corre todos."""
        for k, setup in enumerate(self._setups):
            avisos = validar_setup(setup)
            if avisos:
                nombre = setup.get('nombre') or f'Setup {k + 1}'
                return False, f"«{nombre}»: {avisos[0]}"
        return True, ''

    def _actualizar_boton_ejecutable(self):
        """Habilita «Ejecutar backtest» y «Prueba de parametrización» solo si
        hay un CSV cargado Y ningún setup tiene una configuración imposible
        (etapas que cierran más del 100%, tramos que arriesgan de más): correr
        el motor con eso daría resultados engañosos."""
        if not hasattr(self, 'btn_run'):
            return   # aún construyendo la UI
        hay_csv = bool(self.csv_path)
        valido, motivo = self._validar_sistema()
        habilitado = hay_csv and valido
        self.btn_run.setEnabled(habilitado)
        self.btn_optimizar.setEnabled(habilitado)
        if hay_csv and not valido:
            self.lbl_estado.setText(f"⚠ No se puede ejecutar — {motivo}")

    def _aplicar_preset_friccion(self, path):
        """Prellena slippage/comisión con una aproximación según la clase de
        activo (crypto/stock/futuro-CFD/forex) detectada del CSV — el
        usuario puede seguir ajustando los spinboxes a mano."""
        tipo = tipo_activo_de_csv(path)
        preset = PRESETS_FRICCION.get(tipo)
        if preset is None:
            self.lbl_friccion.setText(
                "Tipo de activo no detectado — slippage/comisión sin cambios")
            return
        self.sp_slippage.setValue(preset['slippage_pct'])
        self.sp_comision.setValue(preset['comision_pct'])
        etiqueta = _TIPO_MAP_INV.get(tipo, tipo)
        self.lbl_friccion.setText(
            f"Preset {etiqueta}: slippage {preset['slippage_pct']:g}% · "
            f"comisión {preset['comision_pct']:g}% (aproximado — ajústalo a "
            f"tu broker)")

    # ── temporalidad: subir de granularidad respecto a la nativa del
    # archivo (nunca bajar), igual criterio que la pestaña Patrones ──
    def _configurar_botones_tf(self):
        nat_min = tf_to_minutes(self._tf_nativo) if self._tf_nativo else None
        for lbl, btn in self._tf_buttons.items():
            m = tf_to_minutes(lbl)
            btn.setEnabled(nat_min is None or (m is not None and m >= nat_min))
        self.btn_tf_custom.setEnabled(True)
        tf_sel = self._tf_nativo if self._tf_nativo in self._tf_buttons else None
        if tf_sel is None and nat_min is not None:
            for lbl in TF_LABELS:
                if tf_to_minutes(lbl) == nat_min:
                    tf_sel = lbl
                    break
        if tf_sel is None:
            tf_sel = next((l for l in TF_LABELS if self._tf_buttons[l].isEnabled()),
                         TF_LABELS[0])
        self._tf_group.blockSignals(True)
        self._tf_buttons[tf_sel].setChecked(True)
        self._tf_group.blockSignals(False)
        self._tf_actual = tf_sel
        self._refrescar_lbl_tf()

    def _refrescar_lbl_tf(self):
        if self.csv_path:
            self.lbl_activo.setTextFormat(Qt.TextFormat.RichText)
            self.lbl_activo.setText(_titulo_activo_html(
                self.csv_path, self._tf_actual or self._tf_nativo))
        if not self._tf_actual:
            self.lbl_tf_info.setText("")
        elif not self._tf_nativo:
            self.lbl_tf_info.setText(
                f"Temporalidad: {self._tf_actual} (no se detectó la nativa "
                f"del nombre del archivo)")
        elif self._tf_actual == self._tf_nativo:
            self.lbl_tf_info.setText(f"Temporalidad nativa: {self._tf_nativo}")
        else:
            self.lbl_tf_info.setText(
                f"Nativa: {self._tf_nativo} · backtest en {self._tf_actual} "
f"(velas agregadas al ejecutar)")

    @_no_crash
    def _on_tf_clicked(self, btn_id):
        if btn_id == len(TF_LABELS):
            self._on_tf_custom_clicked()
            return
        self._seleccionar_tf(TF_LABELS[btn_id])

    def _restaurar_boton_tf(self, tf_label):
        btn = self._tf_buttons.get(tf_label)
        if btn is None and tf_label and tf_label == self._tf_custom:
            btn = self.btn_tf_custom
        if btn is not None:
            self._tf_group.blockSignals(True)
            btn.setChecked(True)
            self._tf_group.blockSignals(False)

    @_no_crash
    def _on_tf_custom_clicked(self):
        previo = self._tf_actual
        texto, ok = pedir_texto(
            self, "Temporalidad personalizada",
            "Temporalidad (p.ej. 20m, 90m, 2h, 3d, 5d, 2w):",
            inicial=self._tf_custom or "")
        if not ok:
            self._restaurar_boton_tf(previo)
            return
        tf_label = parsear_tf_custom(texto)
        if tf_label is None:
            self.lbl_estado.setText(
                f"Temporalidad no válida: «{texto}» (formato: 20m, 2h, 3d, 2w…)")
            self._restaurar_boton_tf(previo)
            return
        nat_min = tf_to_minutes(self._tf_nativo) if self._tf_nativo else None
        if nat_min is not None and tf_to_minutes(tf_label) < nat_min:
            self.lbl_estado.setText(
                f"{tf_label} es más fina que la temporalidad nativa del "
                f"archivo ({self._tf_nativo}): no se puede bajar de granularidad")
            self._restaurar_boton_tf(previo)
            return
        if tf_label in self._tf_buttons:
            self._restaurar_boton_tf(tf_label)
            self._seleccionar_tf(tf_label)
            return
        self._tf_custom = tf_label
        self.btn_tf_custom.setText(f"Custom: {tf_label}")
        self._seleccionar_tf(tf_label)

    def _seleccionar_tf(self, tf_label):
        self._tf_actual = tf_label
        self._refrescar_lbl_tf()
        self._actualizar_visibilidad_filtros()

    def tf_resample(self):
        """(tf_label, regla_pandas) a aplicar antes del backtest, o
        (tf_nativo, None) si se ejecuta en la temporalidad del archivo tal
        cual."""
        if not self._tf_actual or self._tf_actual == self._tf_nativo:
            return self._tf_actual, None
        return self._tf_actual, regla_de_tf(self._tf_actual)

    # ── modelo: lista de setups ──
    def setups(self):
        # volcar lo que haya en los widgets antes de leer (la tabla de
        # reglas custom no emite señales al editar sus combos internos)
        self._guardar_setup_actual()
        resultado = []
        for s in self._setups:
            copia = dict(s, params=dict(s['params']))
            copia.pop('_deshacer', None)   # historial de sesión, no del sistema
            resultado.append(copia)
        return resultado

    def _setup_actual(self):
        fila = self._fila_editada
        if fila is None:
            fila = self.lista_setups.currentRow()
        if 0 <= fila < len(self._setups):
            return self._setups[fila]
        return None

    def setup_seleccionado(self):
        """El setup actualmente abierto en el editor (con lo que haya en los
        widgets ya volcado), para usarlo como base del barrido de
        optimización — ver DialogoOptimizacion / _OptimizerThread."""
        self._guardar_setup_actual()
        s = self._setup_actual()
        return dict(s, params=dict(s['params'])) if s is not None else None

    def cargar_setup_en_constructor(self, setup):
        """Vuelca una configuración elegida en la pestaña Optimizador
        (leaderboard del barrido) sobre el setup actualmente seleccionado
        aquí — conserva el nombre que ya tenía el setup."""
        fila = self._fila_editada
        if fila is None:
            fila = self.lista_setups.currentRow()
        if not (0 <= fila < len(self._setups)):
            return
        nombre = self._setups[fila].get('nombre', setup.get('nombre', 'Setup'))
        self._setups[fila] = dict(setup, params=dict(setup['params']), nombre=nombre)
        self._refresh_lista(seleccionar=fila)
        self.lbl_estado.setText(
            "Configuración del barrido cargada — pulsa «▶ Ejecutar backtest» "
            "para validar en IS+OOS")

    def agregar_setups(self, setups):
        """Añade varias configuraciones (elegidas en la pestaña Optimizador)
        como setups ADICIONALES del sistema actual, respetando MAX_SETUPS."""
        if not setups:
            return
        self._guardar_setup_actual()
        hueco = MAX_SETUPS - len(self._setups)
        entran = setups[:max(hueco, 0)]
        for s in entran:
            self._setups.append(dict(s, params=dict(s['params'])))
        if entran:
            self._refresh_lista(seleccionar=len(self._setups) - 1)
        if len(entran) < len(setups):
            self.lbl_estado.setText(
                f"Solo caben {len(entran)} de {len(setups)} setups "
                f"(máximo {MAX_SETUPS} por sistema)")
        else:
            self.lbl_estado.setText(
                f"{len(entran)} setups añadidos desde el Optimizador — pulsa "
                f"«▶ Ejecutar backtest» para probar el sistema conjunto")

    def _refresh_lista(self, seleccionar=None):
        # reconstrucción completa: los widgets se repoblarán, cualquier
        # estado pendiente ya se guardó en el mutador que nos llamó
        self._fila_editada = None
        fila_previa = self.lista_setups.currentRow() if seleccionar is None else seleccionar
        self.lista_setups.blockSignals(True)
        self.lista_setups.clear()
        for k, s in enumerate(self._setups):
            self.lista_setups.addItem(f"S{k} · {describir_setup(s)}")
        self.lista_setups.blockSignals(False)
        if self._setups:
            fila = min(max(fila_previa, 0), len(self._setups) - 1)
            self.lista_setups.setCurrentRow(fila)
            self._on_setup_selected(fila)
        self._refresh_codigo()
        self._actualizar_boton_ejecutable()

    def _refresh_item_actual(self):
        fila = self._fila_editada
        if fila is not None and 0 <= fila < len(self._setups) \
                and fila < self.lista_setups.count():
            self.lista_setups.item(fila).setText(
                f"S{fila} · {describir_setup(self._setups[fila])}")

    @_no_crash
    def _add_setup(self):
        if len(self._setups) >= MAX_SETUPS:
            self.lbl_estado.setText(f"Máximo {MAX_SETUPS} setups por sistema")
            return
        self._guardar_setup_actual()
        s = _setup_por_defecto()
        s['nombre'] = f"Setup {len(self._setups) + 1}"
        self._setups.append(s)
        self._refresh_lista(seleccionar=len(self._setups) - 1)

    @_no_crash
    def _del_setup(self):
        if len(self._setups) <= 1:
            self.lbl_estado.setText("El sistema necesita al menos un setup")
            return
        fila = self.lista_setups.currentRow()
        if 0 <= fila < len(self._setups):
            self._setups.pop(fila)
            self._refresh_lista(seleccionar=max(fila - 1, 0))

    @_no_crash
    def _dup_setup(self):
        self._guardar_setup_actual()
        s = self._setup_actual()
        if s is None or len(self._setups) >= MAX_SETUPS:
            return
        copia = dict(s, params=dict(s['params']), nombre=s['nombre'] + " (copia)")
        self._setups.insert(self.lista_setups.currentRow() + 1, copia)
        self._refresh_lista(seleccionar=self.lista_setups.currentRow() + 1)

    def _mover(self, delta):
        self._guardar_setup_actual()
        fila = self.lista_setups.currentRow()
        nueva = fila + delta
        if 0 <= fila < len(self._setups) and 0 <= nueva < len(self._setups):
            self._setups[fila], self._setups[nueva] = \
                self._setups[nueva], self._setups[fila]
            self._refresh_lista(seleccionar=nueva)

    def _subir_setup(self):
        self._mover(-1)

    def _bajar_setup(self):
        self._mover(+1)

    # ── editor del setup seleccionado ──
    @_no_crash
    def _on_setup_selected(self, fila):
        if not (0 <= fila < len(self._setups)):
            return
        # guardar las reglas custom del setup que se abandona (sus combos
        # internos no emiten señal al editarse)
        if self._fila_editada is not None and self._fila_editada != fila \
                and self._fila_editada < len(self._setups):
            self._guardar_setup_actual()
        self._fila_editada = fila
        s = self._setups[fila]
        self._cargando = True
        try:
            self.txt_nombre.setText(s['nombre'])
            self._plantilla_actual = s['plantilla']
            self._selector_plantilla.set_seleccion(s['plantilla'])
            self._rebuild_params(s['plantilla'], s['params'])
            self.sp_riesgo.setValue(s['riesgo_pct'] * 100.0)
            self.sp_stop.setValue(s['stop_atr'])
            self.cmb_stop_modo.setCurrentText(
                _MAPA_STOP_MODO_INV.get(s.get('stop_atr_modo', 'fijo'), 'Fijo'))
            self.sp_tp.setValue(s['tp_r'])
            self.sp_tiempo.setValue(s['salida_n_velas'])
            self.sp_be.setValue(s.get('be_atr', 0.0))
            self.cmb_be_unidad.setCurrentText(
                _MAPA_BE_UNIDAD_INV.get(s.get('be_unidad', 'atr'), '× ATR'))
            self.sp_trailing.setValue(s.get('trailing_atr', 0.0))
            entrada = s.get('entrada') or _entrada_por_defecto()
            self.cmb_tipo_entrada.setCurrentText(
                _MAPA_TIPO_ENTRADA_INV.get(entrada.get('tipo', 'mercado'), 'A mercado'))
            self.cmb_nivel_fib.setCurrentText(f"{entrada.get('nivel_fib', 0.618):g}")
            self.sp_zz_desviacion.setValue(entrada.get('zigzag_desviacion', 5.0))
            self.sp_zz_piernas.setValue(entrada.get('zigzag_piernas', 10))
            self.sp_vigencia_limite.setValue(int(s.get('limite_vigencia_velas', 0)))
            self.sp_cancelar_avance_r.setValue(
                float(s.get('limite_cancelar_avance_r', 0.0)))
            # migrar viejas condiciones_salida a parciales
            parciales = s.get('parciales', [])
            if not parciales:
                old_conds = (s.get('filtros') or {}).get('condiciones_salida')
                if old_conds and len(old_conds) > 0:
                    parciales = [dict(etapa_salida_por_defecto(), trigger='cond',
                                      condiciones=list(old_conds))]
                else:
                    # sistema guardado antes de que la salida por señal fuera
                    # una etapa explícita: mostrar la que ya tenía de hecho
                    parciales = [etapa_salida_por_defecto()]
                s['parciales'] = parciales
            self._cargar_parciales_tabla(parciales)
            # sistemas guardados antes de la entrada escalonada: un solo
            # tramo al 100% a la señal (idéntico a lo que ya hacían)
            tramos = s.get('tramos', [])
            if not tramos:
                tramos = [tramo_entrada_por_defecto()]
                s['tramos'] = tramos
            self._cargar_tramos_tabla(tramos)
            # setups guardados antes del cierre parcial por mecanismo: los
            # cuatro arrancan al 100%, que es lo que hacían de siempre
            for clave_mec in MECANISMOS_SALIDA:
                self._mecanismo_setup(s, clave_mec)
            edge = bool(s.get('edge', False))
            self.btn_edge.setChecked(edge)
            self._bloquear_stop_tp(edge)
            if s['plantilla'] == 'Custom (reglas)':
                self.editor_reglas.cargar_reglas(s['params'].get('reglas'))
            filtros = s.get('filtros') or _filtros_por_defecto()
            self._actualizar_dias_checkboxes()
            # Normalizar el identificador del régimen: setups guardados por
            # versiones antiguas pueden traer 'hurst_reversión' con tilde.
            metodo_regimen = _normalizar_metodo_regimen(
                (filtros.get('regimen') or {}).get('metodo', 'ninguno'))
            self.cmb_regimen.setCurrentText(
                _MAPA_REGIMEN_INV.get(metodo_regimen, 'Ninguno'))
            _reg_guardado = filtros.get('regimen', {}) or {}
            self.sp_regimen_periodo.setValue(
                _reg_guardado.get('periodo')
                or ventana_regimen_defecto(metodo_regimen))
            volatilidad = filtros.get('volatilidad') or {}
            self.cmb_volatilidad.setCurrentText(
                _MAPA_VOLATILIDAD_INV.get(volatilidad.get('metodo', 'ninguno'), 'Ninguna'))
            self.sp_volatilidad_periodo.setValue(volatilidad.get('periodo', 100))
            self.sp_volatilidad_percentil.setValue(volatilidad.get('percentil', 50.0))
            self.cmb_sesion.setCurrentText(
                _MAPA_SESION_INV.get(filtros.get('sesion', {}).get('tipo', 'ninguna'), 'Ninguna'))
            self.sp_hora_ini.setValue(filtros.get('sesion', {}).get('hora_inicio', 0))
            self.sp_hora_fin.setValue(filtros.get('sesion', {}).get('hora_fin', 0))
            self._recargar_filas_plantilla(s, filtros.get('condiciones_entrada'))
            tfs = filtros.get('tf_superior') or {}
            self.cmb_tf_sup_indicador.setCurrentText(
                _MAPA_TF_SUPERIOR_INV.get(tfs.get('indicador', 'ninguno'), 'Ninguno'))
            self.cmb_tf_sup_tf.setCurrentText(str(tfs.get('tf', '1h')))
            self.sp_tf_sup_periodo.setValue(int(tfs.get('periodo') or 200))
            self.cmb_tf_sup_relacion.setCurrentText(
                _MAPA_RELACION_TF_INV.get(tfs.get('relacion', 'ambos'), 'Ambos'))
            # Noticias: en revisión («próximamente») — nunca se activan, ni
            # siquiera para setups guardados con el filtro puesto.
            noticias = filtros.get('noticias') or {}
            self.chk_noticias.setChecked(False)
            self.sp_noticias_antes.setValue(noticias.get('minutos_antes', 30))
            self.sp_noticias_después.setValue(noticias.get('minutos_después', 30))
            self.cmb_noticias_impacto.setCurrentText(
                _MAPA_IMPACTO_NOTICIAS_INV.get(noticias.get('impacto_minimo', 'alto'), 'Alto'))
            self.chk_noticias_cerrar.setChecked(noticias.get('cerrar_posiciones', False))
        finally:
            self._cargando = False
        QTimer.singleShot(0, lambda: (
            self._ajustar_alto_tabla(self.tabla_tramos),
            self._ajustar_alto_tabla(self.tabla_parciales),
        ))
        self._actualizar_visibilidad_entrada()
        self._actualizar_visibilidad_filtros()
        self._refresh_definicion()
        self._actualizar_boton_deshacer()

    # ── señal de la plantilla dentro de la tabla de entrada ──
    def _recargar_filas_plantilla(self, s, condiciones=None):
        """Repinta por completo la tabla de entrada (filas de la señal de la
        plantilla + condiciones del usuario). Se usa al abrir un setup o
        cuando cambia la estructura de la señal (p. ej. Ambas → solo Long)."""
        if condiciones is None:
            condiciones = (s.get('filtros') or {}).get('condiciones_entrada')
        self.editor_cond_entrada.cargar(
            condiciones, filas_plantilla(s['plantilla'], s['params']),
            plantilla=s['plantilla'], on_param=self._on_param_desde_fila)
        self._refresh_resumen_entrada()

    def _sincronizar_filas_plantilla(self, s):
        """Refresca los valores de las filas de la señal tras un cambio de
        parámetros, sin recrear los widgets (si la estructura cambió, recarga)."""
        filas = filas_plantilla(s['plantilla'], s['params'])
        if not self.editor_cond_entrada.sincronizar_filas_plantilla(filas):
            self._recargar_filas_plantilla(s)
        self._refresh_resumen_entrada()

    def _refresh_resumen_entrada(self):
        """Qué tamaño abre la señal y con qué riesgo — lo que la tabla de
        condiciones por sí sola no dice."""
        riesgo = self.sp_riesgo.value()
        stop = self.sp_stop.value()
        dist = f"stop {stop:g}×ATR" if stop else "sin stop (dimensiona con 2×ATR de referencia)"
        self.lbl_resumen_entrada.setText(
            f"→ abre el 100% de la posición al open de la vela siguiente · "
            f"riesgo {riesgo:g}% del equity · {dist}")

    @_no_crash
    def _on_param_desde_fila(self, clave, valor):
        """Una celda editable de las filas de la señal ha cambiado: escribe el
        parámetro en el setup y propaga el valor al formulario «Parámetros» y
        al resto de filas, sin recrear el widget que se está editando."""
        s = self._setup_actual()
        if s is None or self._cargando:
            return
        if clave == 'direccion' and s['params'].get('direccion') != valor:
            # cambio de estructura (nº de filas long/short): a diferencia de
            # otros params, este SÍ se apila para deshacer — ver
            # EditorCondiciones._intentar_quitar_lado, que llama aquí para
            # borrar un lado desde la tabla de Entrada.
            pila = s.setdefault('_deshacer', [])
            pila.append(('direccion_param', s['params'].get('direccion')))
            del pila[:-20]
            self._actualizar_boton_deshacer()
        s['params'][clave] = valor
        tipo_w = self._param_widgets.get(clave)
        if tipo_w is not None:
            t, w = tipo_w
            w.blockSignals(True)
            try:
                if t == 'choice':
                    w.setCurrentText(str(valor))
                elif t == 'int':
                    w.setValue(int(valor))
                elif t == 'float':
                    w.setValue(float(valor))
            finally:
                w.blockSignals(False)
        self._agendar_refresco()

    @_no_crash
    def _on_plantilla_changed(self, plantilla):
        if self._cargando:
            return
        s = self._setup_actual()
        if s is None:
            return
        t0 = time.perf_counter()
        self._plantilla_actual = plantilla
        s['plantilla'] = plantilla
        s['params'] = params_por_defecto(plantilla)
        s.update(defaults_setup(plantilla))   # p.ej. cruce → sin stop ATR
        self._cargando = True
        try:
            self._rebuild_params(plantilla, s['params'])
            self.sp_stop.setValue(s['stop_atr'])
            self.sp_tp.setValue(s['tp_r'])
            self.sp_tiempo.setValue(s['salida_n_velas'])
            s['be_atr'] = 0.0
            s['trailing_atr'] = 0.0
            self.sp_be.setValue(0.0)
            self.sp_trailing.setValue(0.0)
            s['parciales'] = [etapa_salida_por_defecto()]
            self._cargar_parciales_tabla(s['parciales'])
            s['tramos'] = [tramo_entrada_por_defecto()]
            self._cargar_tramos_tabla(s['tramos'])
            # el modo edge sobrevive al cambio de plantilla: stop/TP siguen a 0
            if self.btn_edge.isChecked():
                self.sp_stop.setValue(0.0)
                self.sp_tp.setValue(0.0)
            if plantilla == 'Custom (reglas)':
                self.editor_reglas.cargar_reglas(s['params'].get('reglas'))
            self._recargar_filas_plantilla(s)
        finally:
            self._cargando = False
        _log_slow('cambiar_plantilla', (time.perf_counter() - t0) * 1000)
        self._guardar_setup_actual()

    def _rebuild_params(self, plantilla, valores):
        while self.form_params.rowCount():
            self.form_params.removeRow(0)
        self._param_widgets = {}
        for i, spec in enumerate(ESTRATEGIAS[plantilla]['params']):
            if i and i % 10 == 0:
                bombear_eventos()
            t = spec['tipo']
            v = valores.get(spec['clave'], spec['defecto'])
            if t == 'int':
                w = QSpinBox()
                w.setRange(spec.get('min', 1), spec.get('max', 100000))
                w.setValue(int(v))
                w.valueChanged.connect(self._guardar_setup_actual)
            elif t == 'float':
                w = QDoubleSpinBox()
                w.setRange(spec.get('min', -1e9), spec.get('max', 1e9))
                w.setDecimals(2)
                w.setValue(float(v))
                w.valueChanged.connect(self._guardar_setup_actual)
            elif t == 'choice':
                w = QComboBox()
                w.addItems(spec['opciones'])
                w.setCurrentText(str(v))
                w.currentTextChanged.connect(self._guardar_setup_actual)
            elif t == 'patrones':
                w = QListWidget()
                w.setMaximumHeight(150)
                w.setSelectionMode(QListWidget.SelectionMode.NoSelection)
                elegidos = set(v)
                for nom in spec['opciones']:
                    it = QListWidgetItem(nom)
                    it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                    it.setCheckState(Qt.CheckState.Checked if nom in elegidos
                                     else Qt.CheckState.Unchecked)
                    w.addItem(it)
                w.itemChanged.connect(lambda *_: self._guardar_setup_actual())
            elif t == 'reglas':
                continue   # lo gestiona el EditorReglas
            else:
                continue
            self._param_widgets[spec['clave']] = (t, w)
            self.form_params.addRow(spec['etiqueta'] + ':', w)
        self.editor_reglas.setVisible(plantilla == 'Custom (reglas)')

    def _leer_params_widgets(self, plantilla):
        p = params_por_defecto(plantilla)
        for clave, (t, w) in self._param_widgets.items():
            if t in ('int', 'float'):
                p[clave] = w.value()
            elif t == 'choice':
                p[clave] = w.currentText()
            elif t == 'patrones':
                p[clave] = [w.item(i).text() for i in range(w.count())
                            if w.item(i).checkState() == Qt.CheckState.Checked]
        if plantilla == 'Custom (reglas)':
            p['reglas'] = self.editor_reglas.reglas()
        return p

    def _actualizar_modo_stop_habilitado(self, *_):
        """El modo solo tiene efecto cuando existe un stop ATR real."""
        if hasattr(self, 'cmb_stop_modo'):
            self.cmb_stop_modo.setEnabled(
                self.sp_stop.value() > 0.0 and not self.btn_edge.isChecked())

    def _bloquear_stop_tp(self, bloquear):
        """Oscurece (deshabilita) los campos de stop/TP/BE/trailing y sus
        etiquetas — el estilo :disabled de STYLE_BACKTEST los pinta apagados."""
        for w in (self.sp_stop, self.sp_tp, self.sp_be, self.sp_trailing,
                  self._lbl_stop, self._lbl_tp):
            if w is not None:
                w.setEnabled(not bloquear)
        self._actualizar_modo_stop_habilitado()

    @_no_crash
    def _on_edge_toggled(self, on):
        """Modo edge: prueba la señal desnuda — bloquea stop/TP/BE/trailing a
        0 (el motor los interpreta como desactivados) y al salir restaura los
        valores que tenía el setup."""
        s = None if self._cargando else self._setup_actual()
        if on:
            if s is not None:
                s['edge_prev_stop'] = self.sp_stop.value()
                s['edge_prev_tp'] = self.sp_tp.value()
                s['edge_prev_be'] = self.sp_be.value()
                s['edge_prev_trail'] = self.sp_trailing.value()
            self.sp_stop.setValue(0.0)
            self.sp_tp.setValue(0.0)
            self.sp_be.setValue(0.0)
            self.sp_trailing.setValue(0.0)
        elif s is not None:
            self.sp_stop.setValue(s.get('edge_prev_stop', s.get('stop_atr', 0.0)))
            self.sp_tp.setValue(s.get('edge_prev_tp', s.get('tp_r', 0.0)))
            self.sp_be.setValue(s.get('edge_prev_be', s.get('be_atr', 0.0)))
            self.sp_trailing.setValue(s.get('edge_prev_trail', s.get('trailing_atr', 0.0)))
        self._bloquear_stop_tp(on)
        self._guardar_setup_actual()

    # ── filtros de entrada del setup ──
    def _detectar_dias_disponibles(self, path):
        """Días de la semana (0=Lun..6=Dom) con AL MENOS una vela en el CSV
        — para deshabilitar/destildar los checkboxes de días sin datos
        (forex/acciones sin sábado/domingo) en vez de dejar seleccionable
        un filtro que nunca haría nada."""
        try:
            try:
                df_ts = pd.read_csv(path, usecols=['timestamp'], engine='pyarrow')
            except (ImportError, ValueError):
                df_ts = pd.read_csv(path, usecols=lambda c: c == 'timestamp')
            dow = pd.to_datetime(df_ts['timestamp'], errors='coerce').dt.dayofweek.dropna()
            dias = set(int(d) for d in dow.unique())
            return dias or set(range(7))
        except Exception:
            return set(range(7))   # si algo falla, no restringir nada

    def _actualizar_dias_checkboxes(self):
        s = self._setup_actual()
        dias = (s.get('filtros', {}) or {}).get('dias_semana') if s else None
        for i, chk in enumerate(self._chk_dias):
            chk.blockSignals(True)
            disponible = i in self._dias_disponibles
            chk.setEnabled(disponible)
            chk.setChecked(disponible and (dias is None or i in dias))
            chk.blockSignals(False)

    def _actualizar_visibilidad_filtros(self):
        self.sp_regimen_periodo.setEnabled(self.cmb_regimen.currentText() != 'Ninguno')
        self._fila_volatilidad_widget.setEnabled(
            self.cmb_volatilidad.currentText() != 'Ninguna')
        horas_visible = self.cmb_sesion.currentText() == 'Personalizada'
        # En TF diario o mayor filtrar por horas no tiene sentido (todas las
        # velas caen a la misma hora del día → 0 señales): se deshabilita el
        # filtro de sesión. El motor lo ignora igualmente con aviso, pero
        # aquí se deja claro que no está disponible, atenuándolo con un fade
        # (mismos tiempos que el fade del arranque de la app).
        tf_efectivo = self._tf_actual or self._tf_nativo
        nat_min = tf_to_minutes(tf_efectivo) if tf_efectivo else None
        es_diario = nat_min is not None and nat_min >= 1440
        if not hasattr(self, '_efecto_sesion'):
            self._efecto_sesion = QGraphicsOpacityEffect(self.cmb_sesion)
            self.cmb_sesion.setGraphicsEffect(self._efecto_sesion)
            self._anim_sesion = QPropertyAnimation(
                self._efecto_sesion, b"opacity", self)
        self._anim_sesion.stop()
        objetivo = 0.45 if es_diario else 1.0
        self._anim_sesion.setDuration(_FADE_OUT_MS if es_diario else _FADE_IN_MS)
        self._anim_sesion.setStartValue(self._efecto_sesion.opacity())
        self._anim_sesion.setEndValue(objetivo)
        self._anim_sesion.setEasingCurve(
            QEasingCurve.Type.InCubic if es_diario else QEasingCurve.Type.OutCubic)
        self._anim_sesion.start()
        self.cmb_sesion.setEnabled(not es_diario)
        self.cmb_sesion.setToolTip(
            "Filtro no aplicable en TF diario o mayor: todas las velas caen "
            "a la misma hora del día" if es_diario else
            "Solo se abren posiciones NUEVAS mientras la vela cae dentro de "
            "la sesión elegida — fuera de ese horario no se abre nada (no es "
            "un filtro para EVITAR esa sesión). 'Ninguna' desactiva el "
            "filtro y permite operar a cualquier hora.")
        if es_diario:
            horas_visible = False
        self._fila_horas_widget.setVisible(horas_visible)
        if self._lbl_horas is not None:
            self._lbl_horas.setVisible(horas_visible)
        activo = self.chk_noticias.isChecked()
        self._fila_noticias_widget.setEnabled(activo)
        self.chk_noticias_cerrar.setEnabled(activo)
        tf_sup_activo = self.cmb_tf_sup_indicador.currentText() != 'Ninguno'
        self._fila_tf_sup_widget.setEnabled(tf_sup_activo)

    @_no_crash
    def _on_tf_sup_changed(self, _texto):
        self._actualizar_visibilidad_filtros()
        self._guardar_setup_actual()

    @_no_crash
    def _on_regimen_changed(self, texto):
        self._proponer_ventana_regimen(texto)
        self._actualizar_visibilidad_filtros()
        self._guardar_setup_actual()

    def _proponer_ventana_regimen(self, texto):
        """Al cambiar de método, llevar la ventana al valor de fábrica de ESE
        método: ER y Hurst no comparten escala (10 vs 400) y dejar la del otro
        deja el filtro inservible sin que se note.

        Solo se propone si la ventana actual sigue siendo un valor de fábrica —
        un número escrito a mano no se pisa nunca."""
        metodo = _MAPA_REGIMEN.get(texto, 'ninguno')
        if metodo == 'ninguno':
            return
        actual = self.sp_regimen_periodo.value()
        if actual not in (VENTANA_ER_DEFECTO, VENTANA_HURST_DEFECTO):
            return
        nueva = ventana_regimen_defecto(metodo)
        if nueva != actual:
            self.sp_regimen_periodo.setValue(nueva)

    def _bloquear_entrada_limite(self):
        """Deja la entrada por orden límite a la vista pero en gris y sin poder
        elegirse.

        Se deshabilita el item en el modelo en vez de quitarlo del combo: los
        setups ya guardados con ese tipo siguen mostrando el suyo real —
        setCurrentIndex sí puede posarse en un item deshabilitado— en vez de
        aparecer como "A mercado" y silenciosamente cambiar de estrategia."""
        item = self.cmb_tipo_entrada.model().item(
            list(_MAPA_TIPO_ENTRADA).index(_ETIQUETA_LIMITE_FIB))
        if item is None:
            return
        item.setEnabled(False)
        item.setData(QColor('#3a5a7a'), Qt.ItemDataRole.ForegroundRole)

    @_no_crash
    def _on_tipo_entrada_changed(self, _texto):
        self._actualizar_visibilidad_entrada()
        self._guardar_setup_actual()

    def _actualizar_visibilidad_entrada(self):
        """Los ajustes del retroceso y de la vida de la orden solo tienen
        sentido con entrada límite."""
        es_limite = self.cmb_tipo_entrada.currentText() != 'A mercado'
        self._fila_fib_widget.setVisible(es_limite)
        self._fila_vigencia_widget.setVisible(es_limite)
        for widget in (self._fila_fib_widget, self._fila_vigencia_widget):
            etiqueta = self._form_setup.labelForField(widget)
            if etiqueta is not None:
                etiqueta.setVisible(es_limite)

    @_no_crash
    def _on_volatilidad_changed(self, _texto):
        self._actualizar_visibilidad_filtros()
        self._guardar_setup_actual()

    @_no_crash
    def _on_sesion_changed(self, _texto):
        self._actualizar_visibilidad_filtros()
        self._guardar_setup_actual()

    @_no_crash
    def _on_noticias_clicked(self, checked):
        # try/finally: garantiza que el guardado SIEMPRE se ejecute, incluso
        # si _abrir_guia_calendario lanza (sin eso, @​_no_crash se tragaría la
        # excepción y _guardar_setup_actual nunca correría, dejando el setup
        # con 'activo' obsoleto -> el backtest seguiría abortando aunque el
        # checkbox se viera desmarcado).
        try:
            if checked and not get_te_api_key():
                # Sin API key: el checkbox nunca llega a marcarse y se abre la
                # guía paso a paso. Si al cerrarla ya hay key, se marca solo.
                self.chk_noticias.setChecked(False)
                if _abrir_guia_calendario(self):
                    self.chk_noticias.setChecked(True)
                return
        finally:
            self._actualizar_visibilidad_filtros()
            self._guardar_setup_actual()

    @_no_crash
    def _on_noticias_impacto_changed(self, texto):
        self.cmb_noticias_impacto.setToolTip(
            economic_calendar.DESCRIPCION_IMPACTO.get(
                _MAPA_IMPACTO_NOTICIAS.get(texto, 'alto'), ''))
        self._guardar_setup_actual()

    # ── salidas parciales ──
    def _empujar_deshacer(self, s, clave):
        """Guarda una copia del valor ACTUAL de s[clave] (o de
        s['filtros']['condiciones_entrada'] para clave='condiciones_entrada')
        en la pila de deshacer del setup — llamar SIEMPRE antes de mutar la
        lista en sitio (.append/.pop/asignación a un elemento), nunca
        después: el diffing de _guardar_setup_actual no detecta estas
        mutaciones porque ocurren por referencia sobre el mismo objeto que
        él compararía como "antes"."""
        if clave == 'condiciones_entrada':
            valor = (s.get('filtros') or {}).get('condiciones_entrada', [])
        else:
            valor = s.get(clave, [])
        pila = s.setdefault('_deshacer', [])
        pila.append((clave, copy.deepcopy(valor)))
        del pila[:-20]
        self._actualizar_boton_deshacer()

    def _add_etapa_parcial(self):
        s = self._setup_actual()
        if s is None:
            return
        parciales = s.setdefault('parciales', [])
        self._empujar_deshacer(s, 'parciales')
        parciales.append({'pct': 50.0, 'r': 2.0, 'trigger': 'r',
                          'condiciones': [], 'gestion': {'tipo': 0, 'val': 0.0}})
        self._cargar_parciales_tabla(parciales)
        self._guardar_setup_actual()

    def _del_etapa_parcial(self):
        s = self._setup_actual()
        if s is None:
            return
        fila = self.tabla_parciales.currentRow()
        item = self.tabla_parciales.item(fila, 0)
        if item is not None and item.data(Qt.ItemDataRole.UserRole) == _ROL_MECANISMO:
            clave = item.data(_ROL_CLAVE_MECANISMO)
            attr = _MECANISMOS_SALIDA_UI[clave][1]
            getattr(self, attr).setValue(0.0)
            return
        parciales = s.get('parciales', [])
        if len(parciales) <= 1:
            # sin etapas el setup solo cerraría por stop/TP/tiempo: la última
            # etapa ES la salida del sistema, así que no se puede borrar
            self.lbl_estado.setText(
                "La configuración necesita al menos una salida (pon 100% "
                "para cerrar toda la posición de una vez)")
            return
        self._empujar_deshacer(s, 'parciales')
        if 0 <= fila < len(parciales):
            parciales.pop(fila)
        else:
            parciales.pop()
        self._cargar_parciales_tabla(parciales)
        self._guardar_setup_actual()

    # ── entrada escalonada (tramos) ──
    def _add_tramo(self):
        s = self._setup_actual()
        if s is None:
            return
        tramos = s.setdefault('tramos', [])
        self._empujar_deshacer(s, 'tramos')
        tramos.append({'pct': 50.0, 'trigger': 'retroceso', 'val': 1.0,
                       'condiciones': [], 'gestion': {'tipo': 0, 'val': 0.0}})
        self._cargar_tramos_tabla(tramos)
        self._guardar_setup_actual()

    def _del_tramo(self):
        s = self._setup_actual()
        if s is None:
            return
        tramos = s.get('tramos', [])
        if len(tramos) <= 1:
            # el 1er tramo ES la entrada del sistema: sin él no habría forma
            # de abrir la posición
            self.lbl_estado.setText(
                "La entrada necesita al menos un tramo (pon 100% para "
                "construir toda la posición de una vez)")
            return
        self._empujar_deshacer(s, 'tramos')
        fila = self.tabla_tramos.currentRow()
        if 0 <= fila < len(tramos):
            tramos.pop(fila)
        else:
            tramos.pop()
        self._cargar_tramos_tabla(tramos)
        self._guardar_setup_actual()

    def _abrir_dialogo_condiciones_generico(self, fila, tabla, clave_lista, default_factory,
                                            cargar_fn, titulo):
        """Diálogo de condiciones (todas AND) de una etapa/tramo — idéntico
        para salidas parciales y entrada escalonada, solo cambia a qué lista
        del setup escribe y el título del diálogo."""
        s = self._setup_actual()
        if s is None:
            return
        lista = s.setdefault(clave_lista, [])
        self._empujar_deshacer(s, clave_lista)
        while len(lista) <= fila:
            lista.append(default_factory())
        self._editar_condiciones_de(lista[fila], titulo, tabla, fila,
                                    lambda: cargar_fn(lista))

    def _editar_condiciones_de(self, destino, titulo, tabla, fila, tras_aceptar):
        """Diálogo de condiciones sobre un dict cualquiera con clave
        'condiciones' — sirve tanto para una etapa de una lista
        (parciales/tramos) como para uno de los cuatro mecanismos globales de
        salida, que son dicts sueltos del setup."""
        conds = list(destino.get('condiciones', []))

        dlg = QDialog(self, Qt.WindowType.Popup)
        dlg.setMinimumSize(650, 420)
        dlg.setStyleSheet(
            "QDialog { background-color: #0d1424; border: 2px solid #253a60; border-radius: 6px; }")
        lay = QVBoxLayout(dlg)
        lay.setContentsMargins(10, 10, 10, 10)
        editor = EditorCondiciones(titulo)
        editor.cargar_condiciones(conds)
        lay.addWidget(editor)
        fila_btn = QHBoxLayout()
        fila_btn.addStretch()
        btn_ok = QPushButton('Aceptar')
        btn_ok.clicked.connect(dlg.accept)
        btn_cancel = QPushButton('Cancelar')
        btn_cancel.clicked.connect(dlg.reject)
        fila_btn.addWidget(btn_cancel)
        fila_btn.addWidget(btn_ok)
        lay.addLayout(fila_btn)

        QShortcut(Qt.Key.Key_Return, dlg, dlg.accept)
        QShortcut(Qt.Key.Key_Enter, dlg, dlg.accept)

        btn = tabla.cellWidget(fila, 2)
        if btn:
            pos = btn.mapToGlobal(btn.rect().bottomLeft())
            dlg.move(pos)

        if dlg.exec():
            destino['condiciones'] = editor.condiciones()
            tras_aceptar()
            self._guardar_setup_actual()

    def _abrir_dialogo_condiciones(self, fila):
        self._abrir_dialogo_condiciones_generico(
            fila, self.tabla_parciales, 'parciales', etapa_salida_por_defecto,
            self._cargar_parciales_tabla, f'Disparador de la Salida {fila + 1} (todas AND)')

    def _abrir_dialogo_condiciones_tramo(self, fila):
        self._abrir_dialogo_condiciones_generico(
            fila, self.tabla_tramos, 'tramos', tramo_entrada_por_defecto,
            self._cargar_tramos_tabla, f'Disparador del Promedio {fila + 1} (todas AND)')

    def _widget_disparador_generico(self, mapa_trigger, mapa_trigger_inv, trigger_actual,
                                    valor_actual, sufijos, tooltips_por_trigger,
                                    campo2=None, valor2_actual=0.0):
        """Celda «Disparador» genérica: combo + spin de valor numérico, oculto
        para los disparadores que no necesitan un número (señal/condiciones).
        Compartida entre las etapas de salida (R:R) y los tramos de entrada
        escalonada (velas/retroceso/avance): solo cambian las opciones del
        combo y la unidad (sufijo) del spin.

        'tooltips_por_trigger': dict {trigger: texto_breve} — cada opción del
        desplegable muestra SOLO su propia explicación (Qt.ItemDataRole.
        ToolTipRole), y el combo cerrado sigue la de la opción actualmente
        elegida, en vez de un único bloque con todas las opciones juntas.

        'campo2' (opcional): dict {trigger: (sufijo, valor_defecto)} para el
        único caso que necesita un SEGUNDO número (hoy, 'estancamiento': N
        velas + R mínimo) — añade un segundo spin, visible solo para esos
        disparadores."""
        cont = QWidget()
        lay = QHBoxLayout(cont)
        lay.setContentsMargins(2, 3, 2, 3)
        lay.setSpacing(4)
        etiqueta_actual = mapa_trigger_inv.get(trigger_actual, next(iter(mapa_trigger)))
        cmb = _combo_regla(list(mapa_trigger), etiqueta_actual)
        for i, texto_opcion in enumerate(mapa_trigger):
            cmb.setItemData(i, tooltips_por_trigger.get(mapa_trigger[texto_opcion], ''),
                           Qt.ItemDataRole.ToolTipRole)
        cmb.setToolTip(tooltips_por_trigger.get(trigger_actual, ''))
        sp = _spin_regla(valor_actual or 1.0, dec=1)
        sp.setRange(0.1, 1000.0)

        sp2 = None
        if campo2:
            sp2 = _spin_regla(valor2_actual or 1.0, dec=1)
            sp2.setRange(0.1, 1000.0)
            sp2.valueChanged.connect(self._guardar_setup_actual)

        def _actualizar_spin(texto):
            trigger = mapa_trigger[texto]
            sufijo = sufijos.get(trigger)
            sp.setVisible(sufijo is not None)
            if sufijo is not None:
                sp.setSuffix(sufijo)
            if sp2 is not None:
                info2 = campo2.get(trigger)
                sp2.setVisible(info2 is not None)
                if info2 is not None:
                    sp2.setSuffix(info2[0])
            cmb.setToolTip(tooltips_por_trigger.get(trigger, ''))
        cmb.currentTextChanged.connect(
            lambda texto: (_actualizar_spin(texto), self._guardar_setup_actual()))
        # ojo: _actualizar_spin llama a sp.setVisible(...) — hay que añadir sp
        # (y sp2) al layout ANTES de esa llamada. Si no, en ese instante son
        # QDoubleSpinBox sin padre (top-level) y setVisible(True) los muestra
        # como una ventana de Windows suelta (con su icono) durante un
        # instante, hasta que el addWidget de más abajo los reparenta.
        lay.addWidget(cmb, 1)
        lay.addWidget(sp)
        if sp2 is not None:
            lay.addWidget(sp2)
        _actualizar_spin(cmb.currentText())
        sp.valueChanged.connect(self._guardar_setup_actual)
        cont.cmb, cont.sp, cont.sp2 = cmb, sp, sp2
        return cont

    def _widget_disparador_etapa(self, ep):
        trigger = trigger_etapa(ep)
        if trigger == 'estancamiento':
            valor1 = ep.get('velas_max', 0.0) or 10.0
            valor2 = ep.get('r_min', 0.0) or 1.0
        else:
            valor1 = ep.get('r', 0.0) or 2.0
            valor2 = 0.0
        return self._widget_disparador_generico(
            _MAPA_TRIGGER, _MAPA_TRIGGER_INV, trigger, valor1,
            _SUFIJOS_TRIGGER_ETAPA, _TOOLTIPS_TRIGGER_ETAPA,
            campo2=_CAMPO2_TRIGGER_ETAPA, valor2_actual=valor2)

    def _widget_disparador_tramo(self, tr):
        return self._widget_disparador_generico(
            _MAPA_TRIGGER_ENTRADA, _MAPA_TRIGGER_ENTRADA_INV, trigger_tramo(tr),
            tr.get('val', 0.0) or 1.0, _SUFIJOS_TRIGGER_ENTRADA, _TOOLTIPS_TRIGGER_ENTRADA)

    def _ajustar_alto_tabla(self, tabla, filas_min=1):
        """Fija la altura de la tabla a la suma real de cabecera + todas sus
        filas, en vez de un límite fijo: así se ven todas sin scroll interno
        propio. El scroll, si hace falta, lo da la página (ya vive dentro de
        un QScrollArea con setWidgetResizable(True))."""
        alto = tabla.horizontalHeader().height() + 2 * tabla.frameWidth()
        for r in range(tabla.rowCount()):
            alto += tabla.rowHeight(r)
        if tabla.rowCount() < filas_min:
            alto += (filas_min - tabla.rowCount()) * tabla.verticalHeader().defaultSectionSize()
        tabla.setFixedHeight(alto)

    def _cargar_tabla_etapas(self, tabla, etapas, widget_disparador_fn, abrir_cond_fn,
                             abrir_gest_fn, label_ref_anterior):
        """Repuebla una tabla de 4 columnas (Cerrar/Entrar % · Disparador ·
        Condiciones · Gestión) — misma estructura para «Salida del setup»
        (parciales) y «Entrada escalonada» (tramos), solo cambia el
        disparador y a qué lista del setup apuntan los botones."""
        tabla.blockSignals(True)
        tabla.setRowCount(0)
        for e, etapa in enumerate(etapas or []):
            if e and e % 5 == 0:
                bombear_eventos()
            tabla.insertRow(e)
            tabla.setItem(e, 0, QTableWidgetItem(f'{etapa.get("pct", 100):.0f}%'))
            disp = widget_disparador_fn(etapa)
            _seleccionar_fila_al_clic(disp, tabla, e)
            _seleccionar_fila_al_clic(disp.cmb, tabla, e)
            _seleccionar_fila_al_clic(disp.sp, tabla, e)
            if disp.sp2 is not None:
                _seleccionar_fila_al_clic(disp.sp2, tabla, e)
            tabla.setCellWidget(e, 1, disp)
            # botón de condiciones
            conds = etapa.get('condiciones', [])
            n_cond = len(conds)
            btn_cond = QPushButton(_describir_condiciones(conds))
            btn_cond.setToolTip(_tooltip_condiciones(conds))
            btn_cond.setStyleSheet(
                'QPushButton { font-size: 9px; padding: 2px 6px; color: #4fc3f7; }'
                if n_cond else
                'QPushButton { font-size: 9px; padding: 2px 6px; color: #5a7a9a; }')
            btn_cond.clicked.connect(lambda _, fila=e: abrir_cond_fn(fila))
            _seleccionar_fila_al_clic(btn_cond, tabla, e)
            tabla.setCellWidget(e, 2, btn_cond)
            # botón de gestión
            g = etapa.get('gestion', {})
            tipo = g.get('tipo', 0)
            val = g.get('val', 0.0)
            if tipo == 1:
                label = f'BE {val:.1f}'
            elif tipo == 2:
                label = f'Trail {val:.1f}'
            elif tipo == 3:
                label = label_ref_anterior
            else:
                label = '+ Gestion'
            btn_gest = QPushButton(label)
            btn_gest.setStyleSheet(
                'QPushButton { font-size: 9px; padding: 2px 6px; color: #4fc3f7; }'
                if tipo != 0 else
                'QPushButton { font-size: 9px; padding: 2px 6px; color: #5a7a9a; }')
            btn_gest.clicked.connect(lambda _, fila=e: abrir_gest_fn(fila))
            _seleccionar_fila_al_clic(btn_gest, tabla, e)
            tabla.setCellWidget(e, 3, btn_gest)
        tabla.blockSignals(False)

    def _cargar_parciales_tabla(self, parciales):
        self._cargar_tabla_etapas(
            self.tabla_parciales, parciales, self._widget_disparador_etapa,
            self._abrir_dialogo_condiciones, self._abrir_dialogo_gestion, '-> Parcial')
        self._pintar_filas_mecanismo()
        self._validar_pct_parciales()
        self._ajustar_alto_tabla(self.tabla_parciales)
        _repintar_seleccion_fila(self.tabla_parciales)

    # ── mecanismos globales de salida (stop / TP / BE / trailing) ──
    def _mecanismo_setup(self, s, clave):
        """El dict del mecanismo dentro del setup, creándolo con los valores
        de siempre (cerrar el 100%) si el setup viene de una versión previa."""
        mec = s.get(clave)
        if not isinstance(mec, dict):
            mec = salida_mecanismo_por_defecto()
            s[clave] = mec
        return mec

    def _pintar_filas_mecanismo(self):
        """Añade al final de «Salida del setup» una fila por cada mecanismo
        global ACTIVO (stop/TP/BE/trailing con su spin > 0), coloreada y
        editable. No son etapas secuenciales: van marcadas con _ROL_MECANISMO
        para que _leer_tabla_etapas y _validar_pct_parciales las ignoren."""
        s = self._setup_actual()
        if s is None:
            return
        tabla = self.tabla_parciales
        tabla.blockSignals(True)
        try:
            for clave, (etiqueta, attr_spin, sufijo, color) in _MECANISMOS_SALIDA_UI.items():
                spin = getattr(self, attr_spin, None)
                if spin is None or spin.value() <= 0.0:
                    continue
                mec = self._mecanismo_setup(s, clave)
                fila = tabla.rowCount()
                tabla.insertRow(fila)

                item_pct = QTableWidgetItem(f'{float(mec.get("pct", 100.0)):.0f}%')
                item_pct.setData(Qt.ItemDataRole.UserRole, _ROL_MECANISMO)
                item_pct.setData(_ROL_CLAVE_MECANISMO, clave)
                item_pct.setForeground(QColor(color))
                item_pct.setToolTip(
                    f"% de lo que quede ABIERTO que cierra el {etiqueta.lower()} "
                    "al dispararse.\nCon 100% cierra toda la posición (lo de "
                    "siempre). Con menos, cierra esa parte UNA SOLA VEZ por "
                    "operación y el resto sigue vivo.")
                tabla.setItem(fila, 0, item_pct)

                valor_mec = self._widget_valor_mecanismo(clave)
                _seleccionar_fila_al_clic(valor_mec, tabla, fila)
                _seleccionar_fila_al_clic(valor_mec.sp, tabla, fila)
                if valor_mec.cmb is not None:
                    _seleccionar_fila_al_clic(valor_mec.cmb, tabla, fila)
                tabla.setCellWidget(fila, 1, valor_mec)

                conds = mec.get('condiciones', [])
                btn_cond = QPushButton(_describir_condiciones(conds))
                btn_cond.setToolTip(_tooltip_condiciones(conds))
                btn_cond.setStyleSheet(
                    'QPushButton { font-size: 9px; padding: 2px 6px; color: '
                    + (color if conds else '#5a7a9a') + '; }')
                btn_cond.clicked.connect(
                    lambda _, k=clave, t=etiqueta: self._abrir_dialogo_condiciones_mecanismo(k, t))
                _seleccionar_fila_al_clic(btn_cond, tabla, fila)
                tabla.setCellWidget(fila, 2, btn_cond)

                g = mec.get('gestion', {})
                tipo, val = g.get('tipo', 0), g.get('val', 0.0)
                if tipo == 1:
                    label = f'BE {val:.1f}'
                elif tipo == 2:
                    label = f'Trail {val:.1f}'
                elif tipo == 3:
                    label = '-> Cierre ant.'
                else:
                    label = '+ Gestion'
                btn_gest = QPushButton(label)
                btn_gest.setStyleSheet(
                    'QPushButton { font-size: 9px; padding: 2px 6px; color: '
                    + (color if tipo else '#5a7a9a') + '; }')
                btn_gest.clicked.connect(
                    lambda _, k=clave, t=etiqueta: self._abrir_dialogo_gestion_mecanismo(k, t))
                _seleccionar_fila_al_clic(btn_gest, tabla, fila)
                tabla.setCellWidget(fila, 3, btn_gest)
        finally:
            tabla.blockSignals(False)

    def _sufijo_mecanismo(self, clave):
        """Unidad que muestra el spin de la fila. Vacía en el break-even: su
        unidad no es fija y la lleva el combo de la propia fila, así que
        repetirla en el spin la mostraría dos veces."""
        return _MECANISMOS_SALIDA_UI[clave][2] or ''

    def _widget_valor_mecanismo(self, clave):
        """Celda «Disparador» de un mecanismo: su valor, editable, con la misma
        unidad que el campo de arriba. Editarlo escribe en ese campo (y, en el
        break-even, también su combo de unidad), de modo que los dos sitios
        quedan sincronizados en ambos sentidos — mismo patrón que
        _on_param_desde_fila usa para las filas de señal de la tabla de Entrada.

        El mínimo nunca es 0: para apagar el mecanismo se usa su campo de
        arriba (o «− Quitar» con esta fila seleccionada, ver
        _del_etapa_parcial), así la fila no se borra sola mientras la estás
        editando."""
        etiqueta, attr_spin, _sufijo, color = _MECANISMOS_SALIDA_UI[clave]
        origen = getattr(self, attr_spin)
        cont = QWidget()
        lay = QHBoxLayout(cont)
        lay.setContentsMargins(2, 3, 2, 3)
        lay.setSpacing(4)

        lbl = QLabel(etiqueta)
        lbl.setStyleSheet(f'color: {color}; font-size: 10px;')
        lay.addWidget(lbl)

        # QDoubleSpinBox no hereda de QSpinBox: ambos cuelgan de QAbstractSpinBox
        entero = isinstance(origen, QSpinBox)
        minimo = _MIN_VALOR_MECANISMO.get(clave, 0.1)
        if entero:
            sp = QSpinBox()
            sp.setRange(int(minimo), origen.maximum())
            sp.setValue(int(origen.value()))
        else:
            sp = QDoubleSpinBox()
            sp.setDecimals(origen.decimals())
            sp.setRange(minimo, origen.maximum())
            sp.setValue(float(origen.value()))
        sp.setSuffix(self._sufijo_mecanismo(clave))
        sp.setToolTip(
            f"Mismo valor que «{etiqueta}» arriba: editarlo aquí lo cambia allí "
            f"y al revés.\nPara desactivar el mecanismo, ponlo a 0 en el campo "
            f"de arriba (aquí el mínimo es {minimo:g}), o selecciona esta fila "
            f"y pulsa «− Quitar».")
        sp.valueChanged.connect(lambda v, k=clave: self._on_valor_mecanismo(k, v))
        lay.addWidget(sp)

        cmb = None
        if clave == 'salida_be':
            cmb = _combo_regla(list(_MAPA_BE_UNIDAD), self.cmb_be_unidad.currentText())
            cmb.setToolTip(self.cmb_be_unidad.toolTip())
            cmb.currentTextChanged.connect(self._on_unidad_be_desde_fila)
            lay.addWidget(cmb)
        lay.addStretch()
        cont.sp, cont.cmb = sp, cmb
        return cont

    @_no_crash
    def _on_valor_mecanismo(self, clave, valor):
        """Vuelca a su campo de arriba el valor editado en la fila."""
        if self._cargando:
            return
        origen = getattr(self, _MECANISMOS_SALIDA_UI[clave][1])
        origen.blockSignals(True)
        try:
            origen.setValue(valor)
        finally:
            origen.blockSignals(False)
        self._guardar_setup_actual()

    @_no_crash
    def _on_unidad_be_desde_fila(self, texto):
        """La unidad del break-even editada desde su fila: la propaga al combo
        de arriba y refresca el sufijo del spin de la fila."""
        if self._cargando:
            return
        self.cmb_be_unidad.blockSignals(True)
        try:
            self.cmb_be_unidad.setCurrentText(texto)
        finally:
            self.cmb_be_unidad.blockSignals(False)
        self._guardar_setup_actual()

    def _sincronizar_filas_mecanismo(self):
        """Refresca las filas de mecanismo tras cambiar un campo de arriba: si
        cambió el CONJUNTO de mecanismos activos (se encendió/apagó uno)
        repinta la tabla; si solo cambió un valor, lo vuelca al spin de la fila
        sin recrear el widget, para no destruir el que se esté editando."""
        s = self._setup_actual()
        if s is None or self._cargando:
            return
        activos = [c for c, (_e, attr, _s, _c) in _MECANISMOS_SALIDA_UI.items()
                   if getattr(self, attr, None) is not None
                   and getattr(self, attr).value() > 0.0]
        pintados = [c for c in _MECANISMOS_SALIDA_UI if self._fila_de_mecanismo(c) >= 0]
        if activos != pintados:
            self._cargar_parciales_tabla(s.get('parciales'))
            return
        for clave in activos:
            cont = self.tabla_parciales.cellWidget(self._fila_de_mecanismo(clave), 1)
            if cont is None:
                continue
            origen = getattr(self, _MECANISMOS_SALIDA_UI[clave][1])
            cont.sp.blockSignals(True)
            try:
                cont.sp.setSuffix(self._sufijo_mecanismo(clave))
                if origen.value() >= cont.sp.minimum():
                    cont.sp.setValue(origen.value())
            finally:
                cont.sp.blockSignals(False)
            if cont.cmb is not None:
                cont.cmb.blockSignals(True)
                try:
                    cont.cmb.setCurrentText(self.cmb_be_unidad.currentText())
                finally:
                    cont.cmb.blockSignals(False)

    def _fila_de_mecanismo(self, clave):
        """Índice de fila que ocupa un mecanismo en tabla_parciales (-1 si no
        está pintado). Se localiza por la clave guardada en la celda 0, no
        comparando textos: la celda del valor es un widget, no un item."""
        for fila in range(self.tabla_parciales.rowCount()):
            item = self.tabla_parciales.item(fila, 0)
            if item is not None and item.data(_ROL_CLAVE_MECANISMO) == clave:
                return fila
        return -1

    @_no_crash
    def _abrir_dialogo_condiciones_mecanismo(self, clave, etiqueta):
        s = self._setup_actual()
        if s is None:
            return
        self._editar_condiciones_de(
            self._mecanismo_setup(s, clave),
            f'Condiciones del cierre por {etiqueta} (todas AND)',
            self.tabla_parciales, self._fila_de_mecanismo(clave),
            lambda: self._cargar_parciales_tabla(s.get('parciales')))

    @_no_crash
    def _abrir_dialogo_gestion_mecanismo(self, clave, etiqueta):
        s = self._setup_actual()
        if s is None:
            return
        self._editar_gestion_de(
            self._mecanismo_setup(s, clave), 'Mover al precio del cierre anterior',
            self.tabla_parciales, self._fila_de_mecanismo(clave),
            lambda: self._cargar_parciales_tabla(s.get('parciales')))

    def _cargar_tramos_tabla(self, tramos):
        self._cargar_tabla_etapas(
            self.tabla_tramos, tramos, self._widget_disparador_tramo,
            self._abrir_dialogo_condiciones_tramo, self._abrir_dialogo_gestion_tramo,
            '-> Promedio ant.')
        self._validar_pct_tramos()
        self._ajustar_alto_tabla(self.tabla_tramos)
        _repintar_seleccion_fila(self.tabla_tramos)

    def _validar_pct_parciales(self):
        """Avisa si alguna etapa pide cerrar más de lo que queda de la
        posición en ese momento (el motor ya recorta al máximo disponible
        internamente, pero un valor así indica un error de captura). Cada
        «Cerrar %» es sobre el tamaño ORIGINAL de la posición (acumulativo,
        no compuesto): Etapa 1: 10% → queda 90% · Etapa 2: 40% → queda 50%.
        El aviso se muestra como etiqueta junto a la tabla en vez de
        colorear la celda: el QSS global de QTableWidget (línea ~191) hace
        que Qt ignore el setBackground() por celda, así que un fondo rojo
        ahí no se vería."""
        self.tabla_parciales.blockSignals(True)
        pcts = []
        resumen = []
        queda = 100.0
        for e in range(self.tabla_parciales.rowCount()):
            item = self.tabla_parciales.item(e, 0)
            if item is None or item.data(Qt.ItemDataRole.UserRole) == _ROL_MECANISMO:
                continue   # las filas de stop/TP/BE/trailing no son secuenciales
            try:
                val = float(item.text().replace('%', '').strip())
            except ValueError:
                val = 0.0
            pcts.append({'pct': val})
            item.setToolTip('' if val <= queda + 1e-9 else AVISO_EXCESO_PARCIALES)
            # cada etapa cierra `val`% del tamaño ORIGINAL, no de lo que quede
            queda -= min(max(val, 0.0), queda)
            resumen.append(f"Salida {len(pcts)}: cierra {val:g}% → queda {queda:.0f}%")
        self.tabla_parciales.blockSignals(False)
        if resumen:
            self.lbl_resumen_parciales.setText(" · ".join(resumen))
            self.lbl_resumen_parciales.setVisible(True)
        else:
            self.lbl_resumen_parciales.setVisible(False)
        avisos = validar_parciales(pcts)
        self.lbl_aviso_parciales.setText(f"⚠ {avisos[0]}" if avisos else "")
        self.lbl_aviso_parciales.setVisible(bool(avisos))
        self._actualizar_boton_ejecutable()

    def _validar_pct_tramos(self):
        """Muestra cuánto riesgo acumulan los tramos configurados: cada
        «Entrar %» es una porción del RIESGO TOTAL del setup (no del tamaño,
        a diferencia de las salidas parciales), así que deberían sumar 100%
        entre todos — si suman más, el sistema arriesga más del 1% (o el %
        que sea) pretendido; si suman menos, el resto simplemente no se usa."""
        self.tabla_tramos.blockSignals(True)
        resumen = []
        pcts = []
        total = 0.0
        for e in range(self.tabla_tramos.rowCount()):
            item = self.tabla_tramos.item(e, 0)
            if item is None:
                continue
            try:
                val = float(item.text().replace('%', '').strip())
            except ValueError:
                val = 0.0
            total += val
            pcts.append({'pct': val})
            item.setToolTip('' if total <= 100.0 + 1e-9 else
                            "Los promedios ya suman más del 100% del riesgo total.")
            resumen.append(f"Promedio {e + 1}: {val:g}% del riesgo")
        self.tabla_tramos.blockSignals(False)
        riesgo_total_pct = self.sp_riesgo.value() * total / 100.0
        if resumen:
            self.lbl_resumen_tramos.setText(
                " · ".join(resumen) + f" · total {total:g}% del riesgo "
                f"({riesgo_total_pct:g}% del equity)")
            self.lbl_resumen_tramos.setVisible(True)
        else:
            self.lbl_resumen_tramos.setVisible(False)
        avisos = validar_tramos(pcts)
        self.lbl_aviso_tramos.setText(f"⚠ {avisos[0]}" if avisos else "")
        self.lbl_aviso_tramos.setVisible(bool(avisos))
        self._actualizar_boton_ejecutable()

    def _leer_tabla_etapas(self, tabla, clave_lista, mapa_trigger, sufijos, valor_clave):
        """Lee de vuelta una tabla de etapas/tramos al formato que consume el
        motor — contraparte de _cargar_tabla_etapas."""
        s = self._setup_actual()
        if s is None:
            return []
        stored = s.get(clave_lista, [])
        etapas = []
        for e in range(tabla.rowCount()):
            item_pct = tabla.item(e, 0)
            if item_pct is not None \
                    and item_pct.data(Qt.ItemDataRole.UserRole) == _ROL_MECANISMO:
                # fila de stop/TP/BE/trailing: su % vive en s['salida_*'], no
                # en esta lista de etapas secuenciales
                self._leer_pct_mecanismo(s, e, item_pct)
                continue
            try:
                pct_str = item_pct.text().replace('%', '').strip()
                pct = float(pct_str) if pct_str else 100.0
                disp = tabla.cellWidget(e, 1)
                trigger = mapa_trigger[disp.cmb.currentText()]
                idx = len(etapas)
                if idx < len(stored):
                    g = dict(stored[idx].get('gestion', {'tipo': 0, 'val': 0.0}))
                    conds = list(stored[idx].get('condiciones', []))
                else:
                    g = {'tipo': 0, 'val': 0.0}
                    conds = []
                if trigger == 'estancamiento':
                    # único disparador con 2 números: no usa 'valor_clave'
                    # (que para parciales es 'r' — dejarlo a 0 evita que se
                    # cuele por la rama de R:R normal en el motor)
                    etapas.append({'pct': pct, 'trigger': trigger,
                                   'velas_max': int(disp.sp.value()),
                                   'r_min': disp.sp2.value(),
                                   'condiciones': conds, 'gestion': g})
                else:
                    valor = disp.sp.value() if sufijos.get(trigger) is not None else 0.0
                    etapas.append({'pct': pct, valor_clave: valor, 'trigger': trigger,
                                   'condiciones': conds, 'gestion': g})
            except (ValueError, AttributeError, IndexError, KeyError):
                pass
        return etapas

    def _leer_pct_mecanismo(self, s, fila, item_pct):
        """Vuelca el «Cerrar %» editado en una fila de mecanismo al dict
        correspondiente del setup, acotado a [0, 100]."""
        clave = item_pct.data(_ROL_CLAVE_MECANISMO)
        if clave not in _MECANISMOS_SALIDA_UI:
            return
        try:
            pct = float(item_pct.text().replace('%', '').strip())
        except ValueError:
            return
        self._mecanismo_setup(s, clave)['pct'] = min(max(pct, 0.0), 100.0)

    def _leer_parciales_tabla(self):
        return self._leer_tabla_etapas(
            self.tabla_parciales, 'parciales', _MAPA_TRIGGER, _SUFIJOS_TRIGGER_ETAPA, 'r')

    def _leer_tramos_tabla(self):
        return self._leer_tabla_etapas(
            self.tabla_tramos, 'tramos', _MAPA_TRIGGER_ENTRADA, _SUFIJOS_TRIGGER_ENTRADA, 'val')

    def _abrir_dialogo_gestion_generico(self, fila, tabla, clave_lista, default_factory,
                                        cargar_fn, label_ref_anterior):
        """Diálogo de gestión (BE/trailing/mover al precio de referencia
        anterior) de una etapa/tramo — idéntico para salidas parciales y
        entrada escalonada, solo cambia a qué lista del setup escribe."""
        s = self._setup_actual()
        if s is None:
            return
        lista = s.setdefault(clave_lista, [])
        self._empujar_deshacer(s, clave_lista)
        while len(lista) <= fila:
            lista.append(default_factory())
        self._editar_gestion_de(lista[fila], label_ref_anterior, tabla, fila,
                                lambda: cargar_fn(lista))

    def _editar_gestion_de(self, destino, label_ref_anterior, tabla, fila, tras_aceptar):
        """Diálogo de gestión sobre un dict cualquiera con clave 'gestion' —
        sirve tanto para una etapa de una lista (parciales/tramos) como para
        uno de los cuatro mecanismos globales de salida."""
        g_actual = destino.get('gestion', {'tipo': 0, 'val': 0.0})

        dlg = QDialog(self, Qt.WindowType.Popup)
        dlg.setStyleSheet(
            "QDialog { background-color: #0d1424; border: 2px solid #253a60; border-radius: 6px; }")
        lay = QVBoxLayout(dlg)
        lay.setContentsMargins(12, 12, 12, 12)
        grupo = QButtonGroup(dlg)

        rb_none = QRadioButton('Sin gestión')
        rb_be = QRadioButton('Break Even a')
        rb_trail = QRadioButton('Trailing Stop a')
        rb_prev = QRadioButton(label_ref_anterior)

        sp_val = QDoubleSpinBox()
        sp_val.setRange(0.0, 10.0)
        sp_val.setDecimals(1)
        sp_val.setValue(0.5)
        sp_val.setSuffix(' ×ATR')

        for rb in (rb_none, rb_be, rb_trail, rb_prev):
            grupo.addButton(rb)
        g_tipo = g_actual.get('tipo', 0)
        if g_tipo == 1:
            rb_be.setChecked(True)
            sp_val.setValue(g_actual.get('val', 0.5))
        elif g_tipo == 2:
            rb_trail.setChecked(True)
            sp_val.setValue(g_actual.get('val', 0.5))
        elif g_tipo == 3:
            rb_prev.setChecked(True)
        else:
            rb_none.setChecked(True)

        def _toggle_val():
            sp_val.setEnabled(rb_be.isChecked() or rb_trail.isChecked())
        rb_be.toggled.connect(lambda _: _toggle_val())
        rb_trail.toggled.connect(lambda _: _toggle_val())
        _toggle_val()

        lay.addWidget(rb_none)
        fila_be = QHBoxLayout()
        fila_be.addWidget(rb_be)
        fila_be.addWidget(sp_val)
        fila_be.addStretch()
        lay.addLayout(fila_be)
        fila_tr = QHBoxLayout()
        fila_tr.addWidget(rb_trail)
        fila_tr.addWidget(sp_val)
        fila_tr.addStretch()
        lay.addLayout(fila_tr)
        lay.addWidget(rb_prev)

        fila_btn = QHBoxLayout()
        fila_btn.addStretch()
        btn_ok = QPushButton('Aceptar')
        btn_ok.setDefault(True)
        btn_ok.clicked.connect(dlg.accept)
        btn_cancel = QPushButton('Cancelar')
        btn_cancel.clicked.connect(dlg.reject)
        fila_btn.addWidget(btn_cancel)
        fila_btn.addWidget(btn_ok)
        lay.addLayout(fila_btn)

        btn = tabla.cellWidget(fila, 3)
        if btn:
            pos = btn.mapToGlobal(btn.rect().bottomLeft())
            dlg.move(pos)

        if not dlg.exec():
            return

        if rb_be.isChecked():
            destino['gestion'] = {'tipo': 1, 'val': sp_val.value()}
        elif rb_trail.isChecked():
            destino['gestion'] = {'tipo': 2, 'val': sp_val.value()}
        elif rb_prev.isChecked():
            destino['gestion'] = {'tipo': 3, 'val': 0.0}
        else:
            destino['gestion'] = {'tipo': 0, 'val': 0.0}

        tras_aceptar()
        self._guardar_setup_actual()

    def _abrir_dialogo_gestion(self, fila):
        self._abrir_dialogo_gestion_generico(
            fila, self.tabla_parciales, 'parciales', etapa_salida_por_defecto,
            self._cargar_parciales_tabla, 'Mover al precio de la parcial anterior')

    def _abrir_dialogo_gestion_tramo(self, fila):
        self._abrir_dialogo_gestion_generico(
            fila, self.tabla_tramos, 'tramos', tramo_entrada_por_defecto,
            self._cargar_tramos_tabla, 'Mover al precio del tramo anterior')

    def _agendar_refresco(self):
        """Aplaza el refresco completo del editor (tablas, definición, código,
        botón ejecutable) ~150 ms para que una ráfaga de cambios colapse en un
        único repintado. Los widgets de datos ya se actualizaron en el acto; lo
        que se difiere es solo la parte visual derivada."""
        if self._refresco_pendiente:
            return
        self._refresco_pendiente = True
        self._timer_ui.start()

    @_no_crash
    def _flush_refrescos_ui(self):
        self._refresco_pendiente = False
        if self._cargando:
            return
        t0 = time.perf_counter()
        # En equipos lentos el refresco derivado (sincronizar tablas, definir,
        # regenerar código, validar) puede tardar: cursor de espera para que se
        # vea que está trabajando en vez de parecer colgado.
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            s = self._setup_actual()
            if s is not None:
                self._sincronizar_filas_plantilla(s)
                self._sincronizar_filas_mecanismo()
                self._refresh_item_actual()
                self._refresh_definicion()
            self._refresh_codigo()
            self._actualizar_boton_ejecutable()
        finally:
            QApplication.restoreOverrideCursor()
        _log_slow('flush_refrescos_ui', (time.perf_counter() - t0) * 1000)

    @_no_crash
    def _guardar_setup_actual(self, *_):
        if self._cargando:
            return
        s = self._setup_actual()
        if s is None:
            return
        t0 = time.perf_counter()
        # snapshot para deshacer: se compara contra el estado ya volcado al
        # final de este método, y solo se apila si algo de esto cambió
        antes_deshacer = {
            'parciales': copy.deepcopy(s.get('parciales', [])),
            'tramos': copy.deepcopy(s.get('tramos', [])),
            'condiciones_entrada': copy.deepcopy(
                (s.get('filtros') or {}).get('condiciones_entrada', [])),
            'stop_atr': s.get('stop_atr'),
            'stop_atr_modo': s.get('stop_atr_modo'),
            'tp_r': s.get('tp_r'),
            'salida_n_velas': s.get('salida_n_velas'),
            'be_atr': s.get('be_atr'),
            'be_unidad': s.get('be_unidad'),
            'trailing_atr': s.get('trailing_atr'),
            'direccion_param': (s.get('params') or {}).get('direccion'),
        }
        s['nombre'] = self.txt_nombre.text().strip() or s['nombre']
        s['plantilla'] = self._plantilla_actual
        s['params'] = self._leer_params_widgets(s['plantilla'])
        s['riesgo_pct'] = self.sp_riesgo.value() / 100.0
        s['stop_atr'] = self.sp_stop.value()
        s['stop_atr_modo'] = _MAPA_STOP_MODO[self.cmb_stop_modo.currentText()]
        s['tp_r'] = self.sp_tp.value()
        s['salida_n_velas'] = self.sp_tiempo.value()
        s['be_atr'] = self.sp_be.value()
        s['be_unidad'] = _MAPA_BE_UNIDAD[self.cmb_be_unidad.currentText()]
        s['trailing_atr'] = self.sp_trailing.value()
        s['entrada'] = {
            'tipo': _MAPA_TIPO_ENTRADA[self.cmb_tipo_entrada.currentText()],
            'nivel_fib': float(self.cmb_nivel_fib.currentText()),
            'zigzag_desviacion': self.sp_zz_desviacion.value(),
            'zigzag_piernas': self.sp_zz_piernas.value(),
        }
        s['limite_vigencia_velas'] = self.sp_vigencia_limite.value()
        s['limite_cancelar_avance_r'] = self.sp_cancelar_avance_r.value()
        s['parciales'] = self._leer_parciales_tabla()
        self._validar_pct_parciales()
        s['tramos'] = self._leer_tramos_tabla()
        self._validar_pct_tramos()
        s['edge'] = self.btn_edge.isChecked()
        elegidos = [i for i, chk in enumerate(self._chk_dias) if chk.isChecked()]
        dias_disponibles = self._dias_disponibles or set(range(7))
        s['filtros'] = {
            'dias_semana': None if set(elegidos) >= dias_disponibles else elegidos,
            'regimen': {
                'metodo': _MAPA_REGIMEN[self.cmb_regimen.currentText()],
                'periodo': self.sp_regimen_periodo.value(),
            },
            'volatilidad': {
                'metodo': _MAPA_VOLATILIDAD[self.cmb_volatilidad.currentText()],
                'periodo': self.sp_volatilidad_periodo.value(),
                'percentil': self.sp_volatilidad_percentil.value(),
            },
            'sesion': {
                'tipo': _MAPA_SESION[self.cmb_sesion.currentText()],
                'hora_inicio': self.sp_hora_ini.value(),
                'hora_fin': self.sp_hora_fin.value(),
            },
            'condiciones_entrada': self.editor_cond_entrada.condiciones(),
            'condiciones_salida': [],  # ahora en parciales[etapa]
            'tf_superior': {
                'indicador': _MAPA_TF_SUPERIOR[self.cmb_tf_sup_indicador.currentText()],
                'tf': self.cmb_tf_sup_tf.currentText(),
                'periodo': self.sp_tf_sup_periodo.value(),
                'relacion': _MAPA_RELACION_TF[self.cmb_tf_sup_relacion.currentText()],
            },
            'noticias': {
                # Noticias en revisión («próximamente»): nunca activas.
                'activo': False,
                'minutos_antes': self.sp_noticias_antes.value(),
                'minutos_después': self.sp_noticias_después.value(),
                'impacto_minimo': _MAPA_IMPACTO_NOTICIAS[self.cmb_noticias_impacto.currentText()],
                'monedas': None,
                'cerrar_posiciones': False,
            },
        }
        # deshacer: si alguna de las 3 listas cambió respecto al snapshot de
        # arriba, apilar el valor ANTERIOR (no el nuevo) para poder volver
        después_deshacer = {
            'parciales': s['parciales'], 'tramos': s['tramos'],
            'condiciones_entrada': s['filtros']['condiciones_entrada'],
            'stop_atr': s.get('stop_atr'),
            'stop_atr_modo': s.get('stop_atr_modo'),
            'tp_r': s.get('tp_r'),
            'salida_n_velas': s.get('salida_n_velas'),
            'be_atr': s.get('be_atr'),
            'be_unidad': s.get('be_unidad'),
            'trailing_atr': s.get('trailing_atr'),
            'direccion_param': (s.get('params') or {}).get('direccion'),
        }
        for clave, valor_antes in antes_deshacer.items():
            if valor_antes != después_deshacer[clave]:
                pila = s.setdefault('_deshacer', [])
                pila.append((clave, valor_antes))
                del pila[:-20]   # tope: últimas 20 acciones
        self._actualizar_boton_deshacer()
        # la señal mostrada en la tabla de entrada sigue a los parámetros, y
        # las filas de stop/TP/BE/trailing a sus spins
        _log_slow('guardar_setup_actual', (time.perf_counter() - t0) * 1000)
        self._agendar_refresco()

    # ── deshacer (Ctrl+Z): pila de snapshots por setup, compartida entre
    # "Entrada del setup", "Entrada escalonada" y "Salida del setup" ──
    def _crear_boton_deshacer(self):
        btn = QPushButton("↺")
        btn.setToolTip(
            "Deshacer (Ctrl+Z): revierte el último cambio en cualquiera de "
            "las tablas de Entrada/Salida — borrar una fila, editar su %, "
            "su disparador, sus condiciones o su gestión.")
        btn.setEnabled(False)
        btn.clicked.connect(self._deshacer)
        return btn

    def _deshacer(self):
        s = self._setup_actual()
        if s is None or not s.get('_deshacer'):
            return
        clave, valor = s['_deshacer'].pop()
        self._cargando = True
        try:
            if clave == 'parciales':
                s['parciales'] = valor
                self._cargar_parciales_tabla(valor)
            elif clave == 'tramos':
                s['tramos'] = valor
                self._cargar_tramos_tabla(valor)
            elif clave == 'condiciones_entrada':
                s.setdefault('filtros', {})['condiciones_entrada'] = valor
                self.editor_cond_entrada.cargar_condiciones(valor)
            elif clave == 'be_unidad':
                s['be_unidad'] = valor
                self.cmb_be_unidad.setCurrentText(_MAPA_BE_UNIDAD_INV.get(valor, '× ATR'))
            elif clave == 'stop_atr_modo':
                s['stop_atr_modo'] = valor
                self.cmb_stop_modo.setCurrentText(
                    _MAPA_STOP_MODO_INV.get(valor, 'Fijo'))
            elif clave == 'direccion_param':
                s.setdefault('params', {})['direccion'] = valor
                tipo_w = self._param_widgets.get('direccion')
                if tipo_w is not None:
                    tipo_w[1].setCurrentText(str(valor))
                self._sincronizar_filas_plantilla(s)
            elif clave in _ATTR_ESCALAR_MECANISMO:
                s[clave] = valor
                getattr(self, _ATTR_ESCALAR_MECANISMO[clave]).setValue(valor)
        finally:
            self._cargando = False
        self._sincronizar_filas_mecanismo()
        self._actualizar_boton_deshacer()
        self._refresh_item_actual()
        self._refresh_definicion()
        self._refresh_codigo()
        self._actualizar_boton_ejecutable()

    def _actualizar_boton_deshacer(self):
        s = self._setup_actual()
        habilitado = bool(s and s.get('_deshacer'))
        for btn in (getattr(self, 'btn_deshacer_entrada', None),
                    getattr(self, 'btn_deshacer_tramos', None),
                    getattr(self, 'btn_deshacer_parciales', None)):
            if btn is not None:
                btn.setEnabled(habilitado)

    def _refresh_definicion(self):
        s = self._setup_actual()
        if s is None:
            self.lbl_resumen.setText("")
            return
        try:
            texto = describir(s['plantilla'], s['params'])
            # tooltip en la tarjeta seleccionada del selector
            self._selector_plantilla.set_tooltip_actual(texto.replace('\n', '<br>'))
            # resumen compacto: primera línea significativa
            for linea in texto.split('\n'):
                if linea.strip():
                    self.lbl_resumen.setText(linea.strip()[:120])
                    return
            self.lbl_resumen.setText("")
        except Exception:
            self.lbl_resumen.setText("")

    def codigo_actual(self):
        """Código estructurado del sistema completo (con la cuenta)."""
        cfg = dict(self.config_global())
        cfg['pct_oos'] = self.slider_oos.value() / 100.0
        return codigo_sistema(self._setups, cfg)

    def _refresh_codigo(self):
        if not hasattr(self, 'txt_codigo'):
            return   # aún construyendo la UI
        try:
            self.txt_codigo.setPlainText(self.codigo_actual())
        except Exception as e:
            self.txt_codigo.setPlainText(f"(código no disponible: {e})")

    # ── config de cuenta para el hilo ──
    def config_global(self):
        return {
            'capital_inicial': self.sp_capital.value(),
            'comision_pct': self.sp_comision.value() / 100.0,
            'slippage_pct': self.sp_slippage.value() / 100.0,
        }

    # ── selector de sistema: predeterminados / guardados / nuevo ──
    def _leer_guardadas(self):
        """Sistemas guardados por el usuario, uno por carpeta bajo Sistemas/
        (cada una con su propio sistema.json)."""
        _migrar_estrategias_legacy()
        guardadas = {}
        if not os.path.isdir(SISTEMAS_DIR):
            return guardadas
        for entry in os.scandir(SISTEMAS_DIR):
            if not entry.is_dir():
                continue
            ruta = os.path.join(entry.path, 'sistema.json')
            try:
                with open(ruta, encoding='utf-8') as f:
                    datos = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            nombre = datos.get('nombre', entry.name)
            guardadas[nombre] = datos
        return guardadas

    def _recargar_guardadas(self):
        """Reconstruye el selector de sistemas guardados (combo y tarjetas)."""
        guardadas = self._leer_guardadas()
        # combo
        self.cmb_guardadas.blockSignals(True)
        self.cmb_guardadas.clear()
        self.cmb_guardadas.addItem("— seleccionar —", None)
        for nombre in sorted(guardadas):
            self.cmb_guardadas.addItem(nombre, ('guardado', nombre))
        self.cmb_guardadas.blockSignals(False)
        # tarjetas (hasta 6; si hay más se usa el combo)
        while self._guardadas_lay.count():
            item = self._guardadas_lay.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._guardadas_cards.clear()
        if guardadas and len(guardadas) <= 6:
            for nombre, datos in sorted(guardadas.items()):
                n_setups = len(datos.get('setups', [datos] if 'estrategia' in datos else []))
                desc = f"{n_setups} setup{'s' if n_setups != 1 else ''}"
                card = TemplateCard(nombre, desc, '#5a7a9a')
                card.clicked.connect(lambda n=nombre: self._cargar_guardado_nombre(n))
                card.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
                card.customContextMenuRequested.connect(
                    lambda pos, n=nombre: self._menu_guardada(pos, n))
                self._guardadas_cards[nombre] = card
                self._guardadas_lay.addWidget(card)
            if self._guardado_cargado is not None:
                for n, card in self._guardadas_cards.items():
                    card.setSelected(n == self._guardado_cargado)
            self._guardadas_container.setVisible(True)
            self.cmb_guardadas.setVisible(False)
        else:
            self._guardadas_container.setVisible(False)
            self.cmb_guardadas.setVisible(True)

    @_no_crash
    def _on_sistema_card_clicked(self, nombre):
        """Añade un setup nuevo (plantilla predeterminada elegida) al sistema
        actual — no reemplaza los setups existentes."""
        if len(self._setups) >= MAX_SETUPS:
            self.lbl_estado.setText(f"Máximo {MAX_SETUPS} setups por sistema")
            return
        self._guardar_setup_actual()
        s = _setup_por_defecto(nombre)
        s['nombre'] = nombre
        self._setups.append(s)
        self._guardado_cargado = None
        self._favorito_cargado = None
        for card in self._guardadas_cards.values():
            card.setSelected(False)
        for card in self._favoritos_cards.values():
            card.setSelected(False)
        self.lbl_estado.setText(f"Setup «{nombre}» añadido al sistema")
        self._refresh_lista(seleccionar=len(self._setups) - 1)

    @_no_crash
    def _cargar_guardado(self, _idx=None):
        """Carga un sistema guardado desde el combo."""
        dato = self.cmb_guardadas.currentData()
        if not dato:
            return
        _, valor = dato
        self._cargar_guardado_nombre(valor)

    def _cargar_guardado_nombre(self, valor):
        """Carga un sistema guardado por nombre."""
        datos = self._leer_guardadas().get(valor)
        if not datos:
            return
        if 'setups' in datos:
            self._setups = [dict(_setup_por_defecto(), **s)
                            for s in datos['setups']]
        elif 'estrategia' in datos:
            s = _setup_por_defecto(datos['estrategia'])
            s['nombre'] = valor
            s['params'] = dict(params_por_defecto(datos['estrategia']),
                               **datos.get('params', {}))
            self._setups = [s]
        self._guardado_cargado = valor
        for n, card in self._guardadas_cards.items():
            card.setSelected(n == valor)
        # Deseleccionar favoritos y predeterminados: solo uno activo a la vez
        self._favorito_cargado = None
        for n, card in self._favoritos_cards.items():
            card.setSelected(False)
        self._selector_sistema.limpiar_seleccion()
        self.lbl_estado.setText(f"Sistema «{valor}» cargado")
        self._refresh_lista(seleccionar=0)

    # ── favoritos: activo + temporalidad + setup(s), guardados desde
    # la pestaña Resultados (ver ResultadosWidget._guardar_favorito) ──
    def _leer_favoritos(self):
        """Favoritos guardados por el usuario, uno por carpeta bajo
        Favoritos/ (cada una con su propio favorito.json)."""
        favoritos = {}
        if not os.path.isdir(FAVORITOS_DIR):
            return favoritos
        for entry in os.scandir(FAVORITOS_DIR):
            if not entry.is_dir():
                continue
            ruta = os.path.join(entry.path, 'favorito.json')
            try:
                with open(ruta, encoding='utf-8') as f:
                    datos = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            nombre = datos.get('nombre', entry.name)
            favoritos[nombre] = datos
        return favoritos

    def _recargar_favoritos(self):
        """Reconstruye el selector de favoritos (combo y tarjetas)."""
        favoritos = self._leer_favoritos()
        self.cmb_favoritos.blockSignals(True)
        self.cmb_favoritos.clear()
        self.cmb_favoritos.addItem("— seleccionar —", None)
        for nombre in sorted(favoritos):
            self.cmb_favoritos.addItem(nombre, nombre)
        self.cmb_favoritos.blockSignals(False)
        while self._favoritos_lay.count():
            item = self._favoritos_lay.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._favoritos_cards.clear()
        if favoritos and len(favoritos) <= 6:
            for nombre, datos in sorted(favoritos.items()):
                tf = datos.get('tf') or '?'
                desc = f"{tf} · {os.path.basename(datos.get('csv', ''))}"
                card = TemplateCard(nombre, desc, AMBAR)
                card.clicked.connect(lambda n=nombre: self._cargar_favorito_nombre(n))
                card.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
                card.customContextMenuRequested.connect(
                    lambda pos, n=nombre: self._menu_favorito(pos, n))
                self._favoritos_cards[nombre] = card
                self._favoritos_lay.addWidget(card)
            if self._favorito_cargado is not None:
                for n, card in self._favoritos_cards.items():
                    card.setSelected(n == self._favorito_cargado)
            self._favoritos_container.setVisible(True)
            self.cmb_favoritos.setVisible(False)
        else:
            self._favoritos_container.setVisible(False)
            self.cmb_favoritos.setVisible(bool(favoritos))

    @_no_crash
    def _cargar_favorito(self):
        """Carga el favorito seleccionado en el combo."""
        valor = self.cmb_favoritos.currentData()
        if not valor:
            return
        self._cargar_favorito_nombre(valor)

    @_no_crash
    def _cargar_favorito_nombre(self, valor):
        """Carga un favorito por nombre: activo, temporalidad, setup(s) y,
        si es posible, la configuración de cuenta con la que se guardó."""
        datos = self._leer_favoritos().get(valor)
        if not datos:
            return
        csv_path = datos.get('csv')
        if not csv_path or not os.path.exists(csv_path):
            self.lbl_estado.setText(
                f"El activo del favorito «{valor}» ya no existe en esa ruta: {csv_path}")
            return
        self._on_file(csv_path)
        tf = datos.get('tf')
        if tf and tf in self._tf_buttons and self._tf_buttons[tf].isEnabled():
            self._restaurar_boton_tf(tf)
            self._seleccionar_tf(tf)
        if 'setups' in datos:
            self._setups = [dict(_setup_por_defecto(), **s)
                            for s in datos['setups']]
        cfg = datos.get('config') or {}
        if 'capital_inicial' in cfg:
            self.sp_capital.setValue(cfg['capital_inicial'])
        if 'comision_pct' in cfg:
            self.sp_comision.setValue(cfg['comision_pct'] * 100.0)
        if 'slippage_pct' in cfg:
            self.sp_slippage.setValue(cfg['slippage_pct'] * 100.0)
        self._favorito_cargado = valor
        for n, card in self._favoritos_cards.items():
            card.setSelected(n == valor)
        # Deseleccionar guardados y predeterminados: solo uno activo a la vez
        self._guardado_cargado = None
        for n, card in self._guardadas_cards.items():
            card.setSelected(False)
        self._selector_sistema.limpiar_seleccion()
        self.lbl_estado.setText(f"Favorito «{valor}» cargado")
        self._refresh_lista(seleccionar=0)

    @_no_crash
    def _nuevo_sistema_custom(self):
        """Crea un nuevo sistema custom desde cero."""
        s = _setup_por_defecto('Custom (reglas)')
        s['nombre'] = 'Mi setup'
        self._setups = [s]
        self._guardado_cargado = None
        self._favorito_cargado = None
        for card in self._guardadas_cards.values():
            card.setSelected(False)
        for card in self._favoritos_cards.values():
            card.setSelected(False)
        self._selector_sistema.limpiar_seleccion()
        self.lbl_estado.setText(
            "Sistema nuevo: define las reglas del setup y añade más si quieres")
        self._refresh_lista(seleccionar=0)

    @_no_crash
    def _guardar_sistema(self):
        nombre, ok = pedir_texto(self, "Guardar sistema",
                                       "Nombre del sistema:")
        if not ok or not nombre.strip():
            return
        nombre = nombre.strip()
        carpeta = os.path.join(SISTEMAS_DIR, _slug_sistema(nombre))
        os.makedirs(carpeta, exist_ok=True)
        datos = {'nombre': nombre, 'setups': self.setups()}
        with open(os.path.join(carpeta, 'sistema.json'), 'w', encoding='utf-8') as f:
            json.dump(datos, f, ensure_ascii=False, indent=2)
        self._recargar_guardadas()
        self.lbl_estado.setText(f"Sistema «{nombre}» guardado")

    # ── eliminar sistemas guardados ──
    def _menu_guardada(self, pos, nombre):
        menu = QMenu(self)
        accion = menu.addAction("🗑 Eliminar")
        if menu.exec(self.sender().mapToGlobal(pos)) == accion:
            self._eliminar_guardada_nombre(nombre)

    @_no_crash
    def _eliminar_guardada_seleccion(self):
        dato = self.cmb_guardadas.currentData()
        nombre = (dato[1] if dato else None) or self._guardado_cargado
        if not nombre:
            return
        self._eliminar_guardada_nombre(nombre)
        self._guardado_cargado = None

    def _eliminar_guardada_nombre(self, nombre):
        carpeta = os.path.join(SISTEMAS_DIR, _slug_sistema(nombre))
        if not os.path.isdir(carpeta):
            return
        # El código exportado se escribe también bajo Sistemas/<slug>/, así que
        # esta carpeta puede contener bastante más que el sistema.json. Borrarla
        # entera sin decirlo se llevaría por delante los .mq5 y .pine generados.
        extras = sorted(e.name for e in os.scandir(carpeta)
                        if e.name != 'sistema.json')
        aviso = ""
        if extras:
            aviso = ("\n\nSe borrará también el código exportado que hay en esa "
                     "carpeta:\n  " + "\n  ".join(extras))
        if not confirmar(self, "Eliminar sistema",
                         f"¿Eliminar definitivamente el sistema «{nombre}»?{aviso}"):
            return
        shutil.rmtree(carpeta)
        if self._guardado_cargado == nombre:
            self._guardado_cargado = None
        self._recargar_guardadas()
        self.lbl_estado.setText(f"Sistema «{nombre}» eliminado")

    # ── eliminar favoritos ──
    def _menu_favorito(self, pos, nombre):
        menu = QMenu(self)
        accion = menu.addAction("🗑 Eliminar")
        if menu.exec(self.sender().mapToGlobal(pos)) == accion:
            self._eliminar_favorito_nombre(nombre)

    @_no_crash
    def _eliminar_favorito_seleccion(self):
        dato = self.cmb_favoritos.currentData()
        nombre = dato or self._favorito_cargado
        if not nombre:
            return
        self._eliminar_favorito_nombre(nombre)
        self._favorito_cargado = None

    def _eliminar_favorito_nombre(self, nombre):
        carpeta = os.path.join(FAVORITOS_DIR, _slug_sistema(nombre))
        if not os.path.isdir(carpeta):
            return
        if not confirmar(self, "Eliminar favorito",
                         f"¿Eliminar definitivamente el favorito «{nombre}»?"):
            return
        shutil.rmtree(carpeta)
        if self._favorito_cargado == nombre:
            self._favorito_cargado = None
        self._recargar_favoritos()
        self.lbl_estado.setText(f"Favorito «{nombre}» eliminado")


# ══════════════ diálogo: configurar el barrido de parámetros ══════════════
class DialogoOptimizacion(QDialog):
    """Define, para el setup actualmente seleccionado en Constructor, qué
    parámetros (de la estrategia y/o del riesgo propio del setup) se barren
    y en qué rango, más la métrica de ranking. El barrido corre siempre
    SOLO sobre el tramo IS (core.optimizer.optimizar_setup) — los campos no
    marcados se quedan fijos en su valor actual."""

    METRICAS = [
        ('sharpe', 'Sharpe acumulado (IS)'),
        ('sharpe_smart', 'Sharpe smart (IS)'),
        ('sortino', 'Sortino (IS)'),
        ('calmar', 'Calmar (IS)'),
        ('omega', 'Omega (IS)'),
        ('serenity_index', 'Serenity Index (IS)'),
        ('retorno_pct', 'Retorno % (IS)'),
        ('profit_factor', 'Profit factor (IS)'),
        ('sqn', 'SQN (IS)'),
    ]
    # (clave, etiqueta, tipo, escala widget→valor guardado, min, max)
    CAMPOS_RIESGO = [
        ('riesgo_pct', 'Riesgo del setup (%)', 'float', 100.0, 0.01, 100.0),
        ('stop_atr', 'Stop Loss (× ATR)', 'float', 1.0, 0.0, 20.0),
        ('tp_r', 'Take Profit (R)', 'float', 1.0, 0.0, 20.0),
        ('salida_n_velas', 'Salida por tiempo (velas)', 'int', 1.0, 0, 10000),
        ('be_atr', 'Break Even (× ATR)', 'float', 1.0, 0.0, 10.0),
        ('trailing_atr', 'Trailing Stop (× ATR)', 'float', 1.0, 0.0, 10.0),
    ]
    # rango de barrido prellenado para campos concretos (en unidades del
    # widget): min, max, paso — el riesgo se explora de 0.25% a 2% en saltos
    # de 0.15 en vez de partir del valor actual
    RANGOS_SWEEP_DEFECTO = {
        'riesgo_pct': (0.25, 2.0, 0.15),
    }

    def __init__(self, setup, limite_combos=LIMITE_COMBOS_DEFECTO, parent=None):
        super().__init__(parent)
        self.setWindowTitle(
            f"Prueba de parametrización «{setup.get('nombre') or setup['plantilla']}» (Solo IS)")
        self.setMinimumWidth(520)
        self._limite = limite_combos
        self._filas = []   # (clave, chk, sp_min, sp_max, sp_step, tipo, escala, es_riesgo)

        lay = QVBoxLayout(self)
        info = QLabel(
            "Aquí defines QUÉ parámetros barrer y en qué rango. Casilla sin "
            "marcar = ese parámetro queda fijo en su valor actual; marcada = "
            "se prueban todos los valores de min a max en saltos de «paso», y "
            "se comparan todas las combinaciones resultantes en la pestaña "
            "Optimizador. Cada combinación se simula ÚNICAMENTE sobre el "
            "tramo In-Sample (el % que fija el slider IS/OOS del constructor) "
            "— el tramo OOS queda reservado para validar la configuración "
            "elegida con «▶ Ejecutar backtest».")
        info.setWordWrap(True)
        info.setObjectName("campo")
        lay.addWidget(info)

        form = QFormLayout()
        specs_estrategia = [s for s in ESTRATEGIAS[setup['plantilla']]['params']
                            if s['tipo'] in ('int', 'float')]
        for spec in specs_estrategia:
            valor_actual = setup['params'].get(spec['clave'], spec['defecto'])
            self._agregar_fila(form, spec['clave'], spec['etiqueta'], spec['tipo'],
                               1.0, spec.get('min', 0), spec.get('max', 100),
                               valor_actual)
        edge = bool(setup.get('edge'))
        for clave, etiqueta, tipo, escala, vmin, vmax in self.CAMPOS_RIESGO:
            if edge and clave in ('stop_atr', 'tp_r'):
                # modo edge: stop y TP no existen en esta prueba (señal
                # desnuda) — ni se muestran ni se pueden barrer
                continue
            valor_actual = setup.get(clave, 0.0)
            if escala != 1.0:
                valor_actual = valor_actual * escala
            self._agregar_fila(form, clave, etiqueta, tipo, escala, vmin, vmax,
                               valor_actual, es_riesgo=True)
        if edge:
            nota_edge = QLabel(
                "⚡ Prueba de Ventaja activa: stop y take-profit desactivados "
                "— se prueba la señal desnuda")
            nota_edge.setWordWrap(True)
            nota_edge.setStyleSheet("color: #f1c40f; font-size: 11px;")
            form.addRow(nota_edge)
        lay.addLayout(form)

        fila_metrica = QHBoxLayout()
        fila_metrica.addWidget(QLabel("Rankear combinaciones por:"))
        self.cmb_metrica = QComboBox()
        for clave, etiqueta in self.METRICAS:
            self.cmb_metrica.addItem(etiqueta, clave)
        fila_metrica.addWidget(self.cmb_metrica)
        fila_metrica.addStretch()
        lay.addLayout(fila_metrica)

        self.lbl_combos = QLabel("")
        lay.addWidget(self.lbl_combos)

        fila_btn = QHBoxLayout()
        fila_btn.addStretch()
        btn_cancelar = QPushButton("Cancelar")
        btn_cancelar.clicked.connect(self.reject)
        self.btn_lanzar = QPushButton("Lanzar optimización")
        self.btn_lanzar.setObjectName("run")
        self.btn_lanzar.clicked.connect(self.accept)
        fila_btn.addWidget(btn_cancelar)
        fila_btn.addWidget(self.btn_lanzar)
        lay.addLayout(fila_btn)

        self._actualizar_combos()

    def _agregar_fila(self, form, clave, etiqueta, tipo, escala, vmin, vmax,
                      valor_actual, es_riesgo=False):
        chk = QCheckBox("Barrer")
        # paso por defecto: ~1.5% del valor actual del parámetro (criterio de
        # Kaufman: los valores consecutivos probados no deben diferir mucho
        # entre sí — una malla fina alrededor de la zona explorada, no saltos
        # groseros proporcionales al rango total del parámetro). Algunos
        # campos tienen un rango prellenado propio (RANGOS_SWEEP_DEFECTO).
        PASO_PCT = 0.015
        rango_defecto = self.RANGOS_SWEEP_DEFECTO.get(clave)
        if tipo == 'int':
            vmin_i, vmax_i = int(vmin), max(int(vmax), int(vmin) + 1)
            sp_min, sp_max, sp_step = QSpinBox(), QSpinBox(), QSpinBox()
            for w in (sp_min, sp_max):
                w.setRange(vmin_i, vmax_i)
            sp_step.setRange(1, max(vmax_i - vmin_i, 1))
            v = int(round(valor_actual))
            if rango_defecto:
                sp_min.setValue(int(rango_defecto[0]))
                sp_max.setValue(int(rango_defecto[1]))
                sp_step.setValue(max(1, int(rango_defecto[2])))
            else:
                sp_min.setValue(v)
                sp_max.setValue(v)
                sp_step.setValue(max(1, round(abs(v) * PASO_PCT)))
        else:
            sp_min, sp_max, sp_step = QDoubleSpinBox(), QDoubleSpinBox(), QDoubleSpinBox()
            for w in (sp_min, sp_max):
                w.setRange(float(vmin), float(vmax))
                w.setDecimals(2)
            sp_step.setRange(0.01, max(float(vmax) - float(vmin), 0.01))
            sp_step.setDecimals(2)
            v = float(valor_actual)
            if rango_defecto:
                sp_min.setValue(float(rango_defecto[0]))
                sp_max.setValue(float(rango_defecto[1]))
                sp_step.setValue(float(rango_defecto[2]))
            else:
                sp_min.setValue(v)
                sp_max.setValue(v)
                sp_step.setValue(max(0.01, round(abs(v) * PASO_PCT, 2)))
        sp_step.setToolTip(
            "Salto entre valores consecutivos del barrido (min, min+paso, "
            "min+2·paso… hasta max). Por defecto ≈1.5% del valor actual del "
            "parámetro (Kaufman): valores vecinos que difieren poco entre sí, "
            "para ver si el rendimiento forma una meseta estable o un pico "
            "aislado (probable sobreajuste)")
        for w in (sp_min, sp_max, sp_step):
            w.setEnabled(False)
            w.valueChanged.connect(self._actualizar_combos)
        chk.toggled.connect(lambda on, ws=(sp_min, sp_max, sp_step): [w.setEnabled(on) for w in ws])
        chk.toggled.connect(self._actualizar_combos)

        fila = QHBoxLayout()
        fila.addWidget(chk)
        fila.addWidget(QLabel("min")); fila.addWidget(sp_min)
        fila.addWidget(QLabel("max")); fila.addWidget(sp_max)
        fila.addWidget(QLabel("paso")); fila.addWidget(sp_step)
        cont = QWidget()
        cont.setLayout(fila)
        form.addRow(etiqueta + ":", cont)

        self._filas.append((clave, chk, sp_min, sp_max, sp_step, tipo, escala, es_riesgo))

    def _sweeps(self):
        sweep_params, sweep_riesgo = {}, {}
        for clave, chk, sp_min, sp_max, sp_step, tipo, escala, es_riesgo in self._filas:
            if not chk.isChecked():
                continue
            vmin, vmax, vstep = sp_min.value(), sp_max.value(), sp_step.value()
            if escala != 1.0:
                vmin, vmax, vstep = vmin / escala, vmax / escala, vstep / escala
            spec = {'min': vmin, 'max': vmax, 'step': vstep, 'tipo': tipo}
            (sweep_riesgo if es_riesgo else sweep_params)[clave] = spec
        return sweep_params, sweep_riesgo

    @_no_crash
    def _actualizar_combos(self, *_):
        sweep_params, sweep_riesgo = self._sweeps()
        sweep = dict(sweep_params, **sweep_riesgo)
        if not sweep:
            self.lbl_combos.setText(
                "Ningún parámetro marcado — se ejecutará 1 combinación (la actual)")
            self.lbl_combos.setStyleSheet("color: #5a7a9a;")
            self.btn_lanzar.setEnabled(True)
            return
        total = n_combinaciones(sweep)
        if total > self._limite:
            self.lbl_combos.setText(
                f"≈ {total} combinaciones — supera el límite de {self._limite}, "
                f"reduce el rango o el paso")
            self.lbl_combos.setStyleSheet("color: #e74c3c;")
            self.btn_lanzar.setEnabled(False)
        else:
            self.lbl_combos.setText(f"≈ {total} combinaciones")
            self.lbl_combos.setStyleSheet("color: #5a7a9a;")
            self.btn_lanzar.setEnabled(True)

    def resultado(self):
        """(sweep_params, sweep_riesgo, metrica) listos para _OptimizerThread."""
        sweep_params, sweep_riesgo = self._sweeps()
        return sweep_params, sweep_riesgo, self.cmb_metrica.currentData()


# ══════════════ sub-pestaña Resultados ══════════════
# clave=None -> fila separadora (solo etiqueta, sin datos IS/OOS/Total)
_FILAS_METRICAS = [
    ('n_trades', 'Nº trades', 0, ''),
    ('win_rate', 'Win rate', None, ''),        # formato especial (%)
    ('profit_factor', 'Profit factor', 2, ''),
    ('retorno_pct', 'Retorno', 2, ' %'),
    ('retorno_anual_pct', 'Retorno anualizado', 2, ' %'),
    ('max_dd_pct', 'Max drawdown', 2, ' %'),
    ('sharpe', 'Sharpe', 2, ''),
    ('sharpe_smart', 'Sharpe smart (corr. autocorrelación)', 2, ''),
    ('sortino', 'Sortino', 2, ''),
    ('calmar', 'Calmar (CAGR/MaxDD)', 2, ''),
    ('omega', 'Omega', 2, ''),
    ('expectancy_pct', 'Expectancy por trade', 3, ' R'),
    ('racha_perdedora', 'Racha perdedora máx.', 0, ''),
    ('racha_ganadora', 'Racha ganadora máx.', 0, ''),
    ('duracion_media', 'Duración media (velas)', 1, ''),
    (None, '— Curva de capital —', None, None),
    ('r2_equity', 'R² de la equity curve', 3, ''),
    ('dd_promedio_pct', 'Drawdown promedio', 2, ' %'),
    ('ulcer_index', 'Ulcer Index', 2, ' %'),
    ('cvar_dd_pct', 'CVaR drawdown (peor 5%)', 2, ' %'),
    ('tiempo_recuperacion_medio', 'Tiempo recuperación medio', 1, ''),
    ('tiempo_recuperacion_max', 'Tiempo recuperación máx.', 0, ''),
    (None, '— Riesgo de cola —', None, None),
    ('cvar_ret_pct', 'CVaR retorno (peor 5% de velas)', 2, ' %'),
    ('serenity_index', 'Serenity Index', 2, ''),
    (None, '— Robustez y costes —', None, None),
    ('sqn', 'SQN (>2 bueno, >3 excelente)', 2, ''),
    ('payoff_ratio', 'Payoff ratio (ganancia/pérdida media)', 2, ''),
    ('kelly_pct', 'Kelly fraccional (sizing óptimo)', 1, ' %'),
    ('pct_mejor_trade', 'Beneficio Total Atribuible al mejor trade', 1, ' %'),
    ('slippage_minimo_pct', 'Slippage mínimo viable', 3, ' %'),
    ('impacto_comisiones_pct', 'Impacto de comisiones', 2, ' %'),
    (None, '— Eficiencia de ejecución —', None, None),
    ('etd_r_medio', 'ETD medio (beneficio dejado en la mesa)', 2, ' R'),
    ('eficiencia_entrada_media', 'Eficiencia de entrada', 1, ' %'),
    ('eficiencia_salida_media', 'Eficiencia de salida', 1, ' %'),
    (None, '— Estructura de trades —', None, None),
    ('win_rate_long', 'Win rate largos', 1, ' %'),
    ('win_rate_short', 'Win rate cortos', 1, ' %'),
    ('mayor_ganancia', 'Mayor ganancia', 2, ' €'),
    ('mayor_perdida', 'Mayor pérdida', 2, ' €'),
    ('ganancia_bruta', 'Ganancia bruta', 2, ' €'),
    ('perdida_bruta', 'Pérdida bruta', 2, ' €'),
    ('avg_trades_por_dia', 'Trades por día', 2, ''),
    (None, '— Exposición —', None, None),
    ('exposicion_pct', 'Exposición al mercado (% del tiempo)', 1, ' %'),
    ('exposicion_capital_pct', 'Capital medio comprometido', 1, ' %'),
    ('retorno_ajustado_exposicion_pct', 'Retorno ajustado por exposición', 2, ' %'),
]

_TOOLTIPS_METRICAS = {
    'r2_equity': "R² de ajustar una recta a la curva de capital: cerca de 1 "
                "= crecimiento consistente; cerca de 0 = errático",
    'dd_promedio_pct': "Media de la profundidad de cada episodio de "
                       "drawdown (no punto a punto de toda la serie)",
    'tiempo_recuperacion_medio': "Velas desde el pico hasta recuperar el "
                                 "máximo previo, media de los episodios ya recuperados",
    'tiempo_recuperacion_max': "El episodio de drawdown que más tardó en recuperarse",
    'sqn': "System Quality Number (Van Tharp) = √n · media(R) / desv(R), "
          "con R = PnL en múltiplos del riesgo arriesgado. >2.0 bueno, >3.0 excelente",
    'payoff_ratio': "Ganancia media de los trades ganadores / pérdida media "
                    "de los perdedores — bajo con win rate alto puede seguir siendo rentable",
    'kelly_pct': "Fracción óptima de Kelly f = p − (1−p)/b con p = win rate y "
                 "b = payoff ratio. Apunta el % del capital que maximiza el "
                 "crecimiento logarítmico; la práctica habitual es apostar "
                 "una fracción (medio Kelly = la mitad). Vacío si el sistema "
                 "no tiene edge (f ≤ 0).",
    'pct_mejor_trade': "% del PnL total que aporta un único trade — alto "
                       "(>20-30%) sugiere depender de un outlier, no de un edge consistente",
    'slippage_minimo_pct': "Slippage adicional (aplicado igual que la "
                          "comisión) que dejaría la expectancy media en cero. "
                          "Negativo = el sistema ya es negativo sin slippage extra",
    'impacto_comisiones_pct': "Comisión pagada como % de la ganancia bruta "
                             "— en scalping puede devorar el edge entero",
    'ulcer_index': "Penaliza la profundidad Y la duración de los drawdowns "
                   "(a diferencia de Max Drawdown, que solo mira el peor "
                   "punto) — dos sistemas con el mismo Max DD pueden tener "
                   "Ulcer Index muy distinto si uno se recupera rápido y el "
                   "otro no",
    'etd_r_medio': "Media de (MFE − resultado final) por trade, en R. Alto "
                  "= el sistema suele dejar beneficio no realizado sobre la "
                  "mesa antes de cerrar",
    'eficiencia_entrada_media': "Qué tan cerca del mínimo (long) o máximo "
                                "(short) del rango que llegó a tocar el "
                                "trade se hizo la entrada, en promedio",
    'eficiencia_salida_media': "Qué tan cerca de la cima del recorrido se "
                               "cerró el trade, en promedio: 100% = se salió "
                               "justo en el máximo del rango que llegó a tocar "
                               "(0% = en el mínimo). Cuánto beneficio se dejó "
                               "en la mesa en términos absolutos lo dice el ETD",
    'exposicion_pct': "% de velas del tramo con una posición abierta, desde "
                      "que se empieza a construir hasta que se ha cerrado el "
                      "100%. El tiempo que NO estás dentro es tiempo sin "
                      "riesgo de mercado y con el capital libre",
    'exposicion_capital_pct': "Fracción media del capital realmente "
                              "comprometida, contando las velas en plano como "
                              "0. Complementa a la de arriba: estar dentro con "
                              "el 20% del tamaño vivo (tras una parcial, o a "
                              "medio construir) no es estar dentro al 100%. "
                              "Por encima de 100% = apalancamiento",
    'retorno_ajustado_exposicion_pct': "Retorno anualizado dividido entre la "
                                       "exposición: cuánto rinde el sistema "
                                       "por cada unidad de tiempo dentro del "
                                       "mercado. Un 11% anual estando solo el "
                                       "10% del tiempo puntúa 110%; ese mismo "
                                       "11% estando el 90% del tiempo puntúa "
                                       "12.2%",
    'sharpe_smart': "Sharpe ratio corregido por autocorrelación de los "
                    "retornos: si los retornos están correlacionados en serie, "
                    "la volatilidad real es mayor que la muestral y el Sharpe "
                    "clásico lo sobreestima. Este lo penaliza.",
    'sortino': "Retorno medio entre la desviación SOLO de las pérdidas "
               "(downside deviation): premia sistemas que caen poco y "
               "controlado, sin castigar las subidas.",
    'calmar': "Retorno anualizado entre |Max drawdown|: cuánto retorno da el "
              "sistema por cada punto de drawdown que hay que soportar.",
    'omega': "Ganancia total sobre el umbral (0) entre pérdida total bajo él: "
             ">1 significa que las ganancias superan a las pérdidas en "
             "términos agregados.",
    'cvar_ret_pct': "CVaR (Expected Shortfall) del retorno por vela: la "
                    "pérdida media del peor 5% de las velas del tramo. Es "
                    "siempre mayor (en magnitud) que el VaR 95%, porque promedia "
                    "la cola en vez de quedarse en su umbral.",
    'cvar_dd_pct': "CVaR (Expected Shortfall) de la serie de drawdown: la "
                   "profundidad media del peor 5% de los drawdowns del tramo. "
                   "Complementa al Ulcer Index castigando la cola profunda.",
    'serenity_index': "Serenity Index: retorno total del tramo entre la "
                      "profundidad de la cola de drawdowns (Ulcer Index × "
                      "CVaR del drawdown). Premia sistemas con retornos altos "
                      "y drawdowns poco profundos y cortos. Más es mejor.",
    'win_rate_long': "Win rate solo de los trades largos (dir=1). Al filtrar "
                     "«Solo Largos» coincide con el Win rate global; con "
                     "«Solo Cortos» queda vacío.",
    'win_rate_short': "Win rate solo de los trades cortos (dir=-1). Al filtrar "
                      "«Solo Cortos» coincide con el Win rate global; con "
                      "«Solo Largos» queda vacío.",
    'racha_ganadora': "Máxima racha de trades ganadores consecutivos del tramo.",
    'mayor_ganancia': "El trade ganador más grande del tramo, en unidades "
                      "monetarias.",
    'mayor_perdida': "El trade perdedor más grande del tramo, en unidades "
                     "monetarias (valor negativo).",
    'ganancia_bruta': "Suma de todos los trades ganadores, en unidades "
                      "monetarias.",
    'perdida_bruta': "Suma de todos los trades perdedores, en unidades "
                     "monetarias (valor negativo).",
    'avg_trades_por_dia': "Promedio de trades completados por día de trading "
                          "de la clase de activo (STOCK 390 min/día, "
                          "CRYPTO/FUTURO/FOREX 1440). Requiere conocer la "
                          "sesión del activo; si no, queda vacío.",
}


# ── renderizadores del bloque de resultados ──
# Funciones a nivel de módulo en vez de métodos: reciben la tabla o el `dst`
# (destino) sobre el que pintar, de modo que el mismo código sirve para el
# resultado completo y para uno filtrado por dirección.

# métricas que se leen de la curva de capital y no de los trades: con un filtro
# de dirección activo se miden sobre una curva reconstruida, no sobre la equity
# marcada a mercado del motor, y conviene avisarlo en su tooltip
_METRICAS_DE_CURVA = frozenset((
    'retorno_pct', 'retorno_anual_pct', 'max_dd_pct', 'sharpe',
    'sharpe_smart', 'sortino', 'calmar', 'omega',
    'cvar_ret_pct', 'cvar_dd_pct', 'serenity_index',
    'r2_equity', 'dd_promedio_pct', 'ulcer_index',
    'tiempo_recuperacion_medio', 'tiempo_recuperacion_max',
    'retorno_ajustado_exposicion_pct',
))


def _wfa_filtrado(wfa, resultado, velas_por_anio=None, velas_por_dia=None):
    """Rehace el Walk-Forward sobre un resultado filtrado por dirección.

    No hace falta re-simular: las ventanas del WFA son tramos [idx_ini, idx_fin)
    de UNA sola simulación, así que basta con volver a pasar calcular_metricas
    por los mismos bordes usando los trades del lado elegido."""
    if not wfa:
        return None
    ventanas = []
    for v in wfa:
        m = calcular_metricas(resultado, v['idx_ini'], v['idx_fin'],
                              velas_por_anio, velas_por_dia)
        m['idx_ini'], m['idx_fin'] = v['idx_ini'], v['idx_fin']
        ventanas.append(m)
    return ventanas


def _fijar_altura_tabla(tabla):
    """Fija la altura de una tabla para que muestre TODAS sus filas a la vez,
    sin barra de scroll vertical (mismo patrón que la tabla de WFA)."""
    tabla.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    alto_filas = sum(tabla.rowHeight(r) for r in range(tabla.rowCount()))
    tabla.setFixedHeight(
        tabla.horizontalHeader().height() + alto_filas
        + 2 * tabla.frameWidth() + 2)


def render_tabla_metricas(tabla, metricas, claves, tf=None, nota_curva=None):
    """Rellena una tabla de métricas de 4 columnas: nombre + 3 tramos.

    claves: las 3 entradas de `metricas` a volcar, en orden — ('IS', 'OOS',
    'Total') en la vista normal, ('Largos', 'Cortos', 'Total') al comparar
    lados. Las cabeceras se ajustan solas.

    nota_curva: si no es None, se añade al tooltip de las métricas derivadas de
    la curva de capital para advertir de que están medidas sobre una curva
    reconstruida."""
    tabla.setHorizontalHeaderLabels(['Métrica', *claves])
    for fila, (clave, nombre, dec, sufijo) in enumerate(_FILAS_METRICAS):
        item_nombre = QTableWidgetItem(nombre)
        tooltip = _TOOLTIPS_METRICAS.get(clave)
        if nota_curva and clave in _METRICAS_DE_CURVA:
            tooltip = f"{tooltip}\n\n{nota_curva}" if tooltip else nota_curva
        if tooltip:
            item_nombre.setToolTip(tooltip)
        if clave is None:
            # fila separadora: solo etiqueta, en negrita, sin datos
            font = item_nombre.font()
            font.setBold(True)
            item_nombre.setFont(font)
            item_nombre.setForeground(QColor(AZUL))
            tabla.setItem(fila, 0, item_nombre)
            for col in (1, 2, 3):
                tabla.setItem(fila, col, QTableWidgetItem(''))
            continue
        tabla.setItem(fila, 0, item_nombre)
        for col, tramo in enumerate(claves, start=1):
            v = metricas[tramo][clave]
            if clave in ('tiempo_recuperacion_medio', 'tiempo_recuperacion_max'):
                tiempo_str = velas_a_tiempo_legible(v, tf)
                velas_txt = str(int(v)) if dec == 0 and v is not None else _fmt(v, dec, '')
                texto = f"{tiempo_str}  —  {velas_txt} velas" if v is not None else '—'
            elif clave in ('win_rate', 'win_rate_long', 'win_rate_short'):
                texto = _fmt(v * 100 if v is not None else None, 1, ' %')
            elif dec == 0:
                texto = str(int(v)) if v is not None else '—'
            else:
                texto = _fmt(v, dec, sufijo)
            it = QTableWidgetItem(texto)
            it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if clave in ('retorno_pct', 'retorno_anual_pct', 'expectancy_pct',
                         'sqn', 'r2_equity', 'payoff_ratio',
                         'retorno_ajustado_exposicion_pct',
                         'sharpe_smart', 'serenity_index',
                         'sortino', 'calmar', 'omega',
                         'mayor_ganancia', 'ganancia_bruta') and v is not None:
                it.setForeground(QColor(VERDE if v > 0 else ROJO))
            elif clave in ('mayor_perdida', 'perdida_bruta') and v is not None:
                it.setForeground(QColor(ROJO))
            elif clave in ('cvar_ret_pct', 'cvar_dd_pct') and v is not None:
                it.setForeground(QColor(ROJO))
            tabla.setItem(fila, col, it)


def clave_orden_trades(payload, col):
    """Array de claves numericas para ordenar la columna `col`."""
    tr = payload['resultado']['trades']
    if col == 0:
        return tr['idx_entrada'].astype('int64')
    if col == 1:
        return tr['idx_salida'].astype('int64')
    if col == 2:
        return tr['dir'].astype('int64')
    if col == 3:
        return tr['setup'].astype('int64')
    if col == 4:
        return tr['precio_entrada']
    if col == 5:
        return tr['precio_salida']
    if col == 6:
        return tr['pnl']
    if col == 7:
        return tr['motivo'].astype('int64')
    if col == 8:
        return tr['mfe_r']
    if col == 9:
        return tr['mae_r']
    if col == 10:
        return tr['etd_r']
    if col == 11:
        return tr['eficiencia_entrada']
    if col == 12:
        return tr['eficiencia_salida']
    return tr['pnl']


def render_tabla_trades(tabla, payload, orden=None):
    ts = pd.DatetimeIndex(payload['timestamps'])
    nombres_setup = payload.get('nombres_setup') or []
    tr = payload['resultado']['trades']
    n_tr = len(tr['pnl'])
    if orden is None:
        col = getattr(tabla, '_sort_col', -1)
        if col >= 0:
            keys = clave_orden_trades(payload, col)
            asc = getattr(tabla, '_sort_order',
                          Qt.SortOrder.AscendingOrder)
            orden = (np.argsort(keys, kind='stable')
                     if asc == Qt.SortOrder.AscendingOrder
                     else np.argsort(keys, kind='stable')[::-1])
        else:
            orden = np.arange(n_tr)
    tabla.setSortingEnabled(False)
    tabla.setRowCount(n_tr)
    # fila mostrada -> índice en el array de trades del payload; lo consulta
    # quien necesite volver del clic al trade concreto
    tabla._orden_filas = np.asarray(orden, dtype=np.int64)
    for disp_r, r in enumerate(orden):
        if disp_r and disp_r % 400 == 0:
            bombear_eventos()
        r = int(r)
        i_in, i_out = tr['idx_entrada'][r], tr['idx_salida'][r]
        sid = int(tr['setup'][r])
        nombre_setup = (f"S{sid} · {nombres_setup[sid]}"
                        if sid < len(nombres_setup) else str(sid))
        vals = [
            str(ts[i_in].strftime('%Y-%m-%d %H:%M')),
            str(ts[i_out].strftime('%Y-%m-%d %H:%M')),
            'Long' if tr['dir'][r] > 0 else 'Short',
            nombre_setup,
            f"{tr['precio_entrada'][r]:.4f}",
            f"{tr['precio_salida'][r]:.4f}",
            f"{tr['pnl'][r]:+.2f}",
            MOTIVOS_SALIDA.get(int(tr['motivo'][r]), '?'),
            f"{tr['mfe_r'][r]:+.2f}",
            f"{tr['mae_r'][r]:+.2f}",
            f"{tr['etd_r'][r]:+.2f}",
            (f"{tr['eficiencia_entrada'][r]:.0f}"
             if not np.isnan(tr['eficiencia_entrada'][r]) else "—"),
            (f"{tr['eficiencia_salida'][r]:.0f}"
             if not np.isnan(tr['eficiencia_salida'][r]) else "—"),
        ]
        for c_i, v in enumerate(vals):
            it = QTableWidgetItem(v)
            it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if c_i == 6:
                it.setForeground(QColor(VERDE if tr['pnl'][r] > 0 else ROJO))
            tabla.setItem(disp_r, c_i, it)


def render_tabla_setups(tabla, grupo, payload):
    """Métricas por setup recalculadas sobre los trades del payload — así la
    tabla sigue al filtro de dirección en vez de quedarse con el agregado que
    calculó el hilo de backtest."""
    base = payload.get('metricas_setup') or []
    if not base:
        grupo.setVisible(False)
        return
    grupo.setVisible(True)
    tr = payload['resultado']['trades']
    tabla.setRowCount(len(base))
    for r, s in enumerate(base):
        m = tr['setup'] == r
        pnl = tr['pnl'][m]
        r_setup = tr['r_multiple'][m]
        n_s = int(m.sum())
        win = float((pnl > 0).mean()) if n_s else None
        pnl_total = float(pnl.sum())
        expectancy = float(r_setup.mean()) if n_s else None
        vals = [f"S{r} · {s['nombre']}", f"{s['riesgo_pct'] * 100:g} %",
                str(n_s), _fmt(win * 100 if win is not None else None, 1, ' %'),
                f"{pnl_total:+.2f}", _fmt(expectancy, 3, ' R')]
        for c_i, v in enumerate(vals):
            it = QTableWidgetItem(v)
            it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if c_i == 4:
                it.setForeground(QColor(VERDE if pnl_total > 0 else ROJO))
            tabla.setItem(r, c_i, it)


def render_equity(dst, payload):
    if payload is None:
        return
    eq = payload['resultado']['equity']
    ts = pd.DatetimeIndex(payload['timestamps'])
    corte = payload['corte']
    cap0 = eq[0] if eq[0] > 0 else 1.0
    n = len(eq)

    pct_is = corte / n * 100 if n > 0 else 0
    pct_oos = 100 - pct_is

    modo = dst.combo_eq_modo.currentIndex()
    if modo == 1:
        y = eq
        ylabel = 'Capital ($)'
        fmt_val = lambda v: f'{v:,.0f}'
    elif modo == 2:
        y = np.log(eq / cap0) * 100.0
        ylabel = 'Log-retorno %'
        fmt_val = lambda v: f'{v:+.2f}%'
    elif modo == 3:
        eq_max = np.maximum.accumulate(eq)
        y = (eq / eq_max - 1.0) * 100.0
        ylabel = 'Drawdown %'
        fmt_val = lambda v: f'{v:+.2f}%'
    else:
        y = (eq / cap0 - 1.0) * 100.0
        ylabel = 'Retorno %'
        fmt_val = lambda v: f'{v:+.2f}%'

    c = payload['close']
    c0 = c[0] if c[0] else 1.0
    bh_eq = cap0 * c / c0
    if modo == 1:
        y_bh = bh_eq
    elif modo == 2:
        y_bh = np.log(bh_eq / cap0) * 100.0
    elif modo == 3:
        bh_max = np.maximum.accumulate(bh_eq)
        y_bh = (bh_eq / bh_max - 1.0) * 100.0
    else:
        y_bh = (bh_eq / cap0 - 1.0) * 100.0

    dst.grp_equity.setVisible(True)
    dst.fig_equity.clear()
    ax = dst.fig_equity.add_subplot(111)
    _style_ax(ax)

    ts_is = ts[:corte + 1]
    ts_oos = ts[corte:]
    y_is = y[:corte + 1]

    # OOS aislado: rebasado al valor de equity justo en el punto de
    # corte, ignorando lo acumulado en IS — así se ve el rendimiento que
    # generó el tramo OOS por sí solo, no arrastrando el resultado de IS
    # (que es lo que muestra la curva "Total").
    eq_oos = eq[corte:]
    base_oos = eq_oos[0] if len(eq_oos) else cap0
    if modo == 1:
        y_oos_solo = eq_oos / base_oos * cap0
    elif modo == 2:
        y_oos_solo = np.log(eq_oos / base_oos) * 100.0
    elif modo == 3:
        max_oos = np.maximum.accumulate(eq_oos)
        y_oos_solo = (eq_oos / max_oos - 1.0) * 100.0
    else:
        y_oos_solo = (eq_oos / base_oos - 1.0) * 100.0

    if modo == 3:
        ax.fill_between(ts_is, 0, y_is, color='#e74c3c', alpha=0.40, label='IS')
        if corte < n:
            ax.fill_between(ts_oos, 0, y_oos_solo, color='#c0392b', alpha=0.35,
                            label='OOS (aislado)')
            # Total: drawdown continuo sobre todo el periodo (IS+OOS),
            # calculado sobre el máximo acumulado de la equity completa
            # — permite ver caídas que arrancan en IS y siguen en OOS,
            # cosa que "OOS (aislado)" no muestra al rebasar su propio
            # máximo al inicio del tramo OOS.
            ax.plot(ts_oos, y[corte:], color=GRIS, linewidth=1.2, label='Total')
        ax.axhline(0, color=GRIS, linewidth=0.5, linestyle='--')
    else:
        ax.plot(ts_is, y_is, color=AZUL, linewidth=1.2, label='IS')
        if corte < n:
            ax.plot(ts_oos, y[corte:], color=GRIS, linewidth=1.2,
                    label='Total')
            ax.plot(ts_oos, y_oos_solo, color='#ff9900', linewidth=1.2,
                    label='OOS (aislado)')
        if modo in (0, 2):
            ax.axhline(0, color=GRIS, linewidth=0.5, linestyle='--')

    if modo in (0, 1, 2) and getattr(dst, 'chk_bh', None) is not None \
            and dst.chk_bh.isChecked():
        ax.plot(ts, y_bh, color='#9b59b6', linewidth=1.1, linestyle='--',
                alpha=0.85, label='Buy & Hold')
        ax.text(ts[-1], y_bh[-1], f'  B&H {fmt_val(y_bh[-1])}',
                color='#9b59b6', fontsize=7, ha='left', va='center')

    if 0 < corte < n:
        ax.axvline(ts[corte], color=GRIS, linewidth=0.8, linestyle='--', alpha=0.7)
        ylim = ax.get_ylim()
        ax.text(ts[corte], ylim[1] - (ylim[1] - ylim[0]) * 0.03,
                f'  IS {pct_is:.0f}% / OOS {pct_oos:.0f}%',
                fontsize=7, color=GRIS, ha='left', va='top',
                bbox=dict(facecolor=FIG_BG, alpha=0.7, edgecolor='none', pad=2))

    if corte > 0:
        ax.text(ts[corte], y_is[-1], f'  {fmt_val(y_is[-1])}',
                color=AZUL, fontsize=7, ha='left', va='center')
    if corte < n:
        va_tot, va_oos = ('bottom', 'top') if y[-1] >= y_oos_solo[-1] else ('top', 'bottom')
        ax.text(ts[-1], y_oos_solo[-1], f'  OOS {fmt_val(y_oos_solo[-1])}',
                color='#ff9900', fontsize=7, ha='left', va=va_oos)
        ax.text(ts[-1], y[-1], f'  Total {fmt_val(y[-1])}',
                color=GRIS, fontsize=7.5, fontweight='bold', ha='left', va=va_tot)

    ax.set_ylabel(ylabel, fontsize=8, color=AX_FG)
    ax.legend(fontsize=7, framealpha=0.2, loc='best')
    try:
        dst.fig_equity.tight_layout(pad=0.6)
    except Exception:
        pass
    dst.canvas_equity.draw_idle()


def render_montecarlo(dst, mc, capital, max_dd_base=None, retorno_base=None):
    if not mc or mc['n_sims'] == 0:
        dst.grp_mc.setVisible(False)
        return
    dst.grp_mc.setVisible(True)
    dst.lbl_mc.setText(
        f"Prob. de terminar por debajo del capital inicial (pérdida): {mc['prob_negativo'] * 100:.1f}%   ·   "
        f"Prob. de ruina (equity < 50% del inicial): {mc['prob_ruina'] * 100:.1f}%   ·   "
        f"Retorno mediano: {(np.median(mc['finales']) / capital - 1) * 100:+.1f}%   ·   "
        f"Max DD mediano: {np.median(mc['max_dds']):.1f}%")

    dst.fig_mc.clear()
    ax1 = dst.fig_mc.add_subplot(131)
    ax2 = dst.fig_mc.add_subplot(132)
    ax3 = dst.fig_mc.add_subplot(133)
    for ax in (ax1, ax2, ax3):
        _style_ax(ax)

    cur = mc['curvas_pct']
    x = np.arange(len(cur['p50']))
    ax1.fill_between(x, cur['p5'], cur['p95'], color=AZUL, alpha=0.2)
    ax1.plot(x, cur['p95'], color=AZUL, linewidth=0.7, linestyle='--', alpha=0.6)
    ax1.plot(x, cur['p5'], color=AZUL, linewidth=0.7, linestyle='--', alpha=0.6)
    ax1.plot(x, cur['p50'], color=AZUL, linewidth=1.0)
    ax1.axhline(capital, color=GRIS, linewidth=0.7, linestyle='--')
    ax1.set_title('Equity p5-p50-p95', fontsize=8, color=AX_FG)
    ax1.set_xlabel('Trade nº', fontsize=7, color=AX_FG)

    ret_fin = (mc['finales'] / capital - 1) * 100
    ax2.hist(ret_fin, bins=40, color=VERDE, alpha=0.8)
    ax2.axvline(0, color=GRIS, linewidth=0.7, linestyle='--')
    if retorno_base is not None:
        ax2.axvline(retorno_base, color=VERDE, linewidth=0.9, linestyle='--')
    ax2.set_title('Retorno final %', fontsize=8, color=AX_FG)

    ax3.hist(mc['max_dds'], bins=40, color=ROJO, alpha=0.8)
    if max_dd_base is not None:
        ax3.axvline(max_dd_base, color=ROJO, linewidth=0.9, linestyle='--')
    ax3.set_title('Max drawdown %', fontsize=8, color=AX_FG)
    try:
        dst.fig_mc.tight_layout(pad=0.6)
    except Exception:
        pass
    dst.canvas_mc.draw_idle()


def render_mfe_mae(dst, payload):
    if payload is None:
        dst.grp_mfe_mae.setVisible(False)
        return
    tr = payload['resultado']['trades']
    if len(tr['pnl']) == 0:
        dst.grp_mfe_mae.setVisible(False)
        return
    dst.grp_mfe_mae.setVisible(True)

    filtro = dst.combo_mfe_filtro.currentIndex()  # 0 todas, 1 ganadoras, 2 perdedoras
    if filtro == 1:
        mask = tr['pnl'] > 0
    elif filtro == 2:
        mask = tr['pnl'] <= 0
    else:
        mask = np.ones(len(tr['pnl']), dtype=bool)

    mfe = tr['mfe_r'][mask]
    mae = tr['mae_r'][mask]
    r_realizado = tr['r_multiple'][mask]
    pnl = tr['pnl'][mask]
    percentil = dst.spin_percentil.value()

    dst.fig_mfe_mae.clear()
    if len(mfe) == 0:
        ax = dst.fig_mfe_mae.add_subplot(111)
        _style_ax(ax)
        ax.text(0.5, 0.5, 'Sin operaciones para este filtro', ha='center',
                va='center', color=AX_FG, fontsize=9, transform=ax.transAxes)
        dst.canvas_mfe_mae.draw_idle()
        return

    colores = np.where(pnl > 0, VERDE, ROJO)
    modo = dst.combo_mfe_modo.currentIndex()
    ax1 = dst.fig_mfe_mae.add_subplot(121)
    ax2 = dst.fig_mfe_mae.add_subplot(122)
    for ax in (ax1, ax2):
        _style_ax(ax)

    if modo == 0:
        p_mfe = np.percentile(mfe, percentil)
        p_mae = np.percentile(mae, percentil)

        ax1.scatter(mfe, r_realizado, c=colores, s=30, alpha=0.85,
                   edgecolors=GRID_C, linewidths=0.4)
        ax1.axvline(p_mfe, color=GRIS, linewidth=0.9, linestyle='--')
        ax1.text(p_mfe, ax1.get_ylim()[1], f' P{percentil}={p_mfe:.2f}R',
                 color=GRIS, fontsize=7, ha='left', va='top')
        ax1.axhline(0, color=GRIS, linewidth=0.5, linestyle=':')
        ax1.set_xlabel('MFE alcanzado (R)', fontsize=8, color=AX_FG)
        ax1.set_ylabel('R realizado', fontsize=8, color=AX_FG)
        ax1.set_title('Eficiencia MFE', fontsize=8, color=AX_FG)
        # diagonal y=x: la distancia vertical de cada punto a esta línea
        # es el ETD (MFE - R realizado) de ese trade
        lim = ax1.get_xlim()
        ax1.plot([0, lim[1]], [0, lim[1]], color=GRIS, linewidth=0.7,
                 linestyle=':', alpha=0.5)
        ax1.set_xlim(lim)

        ax2.scatter(-mae, r_realizado, c=colores, s=30, alpha=0.85,
                   edgecolors=GRID_C, linewidths=0.4)
        ax2.set_xlim(right=0)
        ax2.axvline(-p_mae, color=GRIS, linewidth=0.9, linestyle='--')
        ax2.text(-p_mae, ax2.get_ylim()[1], f'P{percentil}={-p_mae:.2f}R ',
                 color=GRIS, fontsize=7, ha='right', va='top')
        ax2.axhline(0, color=GRIS, linewidth=0.5, linestyle=':')
        ax2.set_xlabel('MAE alcanzado (R)', fontsize=8, color=AX_FG)
        ax2.set_ylabel('R realizado', fontsize=8, color=AX_FG)
        ax2.set_title('Eficiencia MAE', fontsize=8, color=AX_FG)
    elif modo == 1:
        ax1.hist(mfe, bins=30, color=VERDE, alpha=0.8)
        p_mfe = np.percentile(mfe, percentil)
        ax1.axvline(p_mfe, color=GRIS, linewidth=0.9, linestyle='--')
        ax1.text(p_mfe, ax1.get_ylim()[1], f' P{percentil}={p_mfe:.2f}R', color=GRIS,
                 fontsize=7, ha='left', va='top')
        ax1.set_title('Máxima Excursión Favorable (R)', fontsize=8, color=AX_FG)
        ax1.set_xlabel('MFE (R)', fontsize=8, color=AX_FG)
        ax1.set_ylabel('Nº Trades', fontsize=8, color=AX_FG)

        subset_mfe = mfe[mfe <= p_mfe]
        if len(subset_mfe) > 0:
            mu = subset_mfe.mean()
            sigma = subset_mfe.std()
            ax1.axvline(mu, color=AZUL, linewidth=1.1, linestyle='-', alpha=0.9)
            ax1.axvline(mu + sigma, color=AZUL, linewidth=0.8, linestyle='-', alpha=0.35)
            ax1.axvline(mu - sigma, color=AZUL, linewidth=0.8, linestyle='-', alpha=0.35)
            ax1.text(0.98, 0.02,
                     f'μ={mu:.2f}R\n+σ={mu + sigma:.2f}R\n−σ={mu - sigma:.2f}R',
                     transform=ax1.transAxes, color=AZUL, fontsize=7,
                     ha='right', va='bottom',
                     bbox=dict(facecolor=FIG_BG, alpha=0.75, edgecolor='none', pad=3))

        ax2.hist(-mae, bins=30, color=ROJO, alpha=0.8)
        ax2.set_xlim(right=0)
        p_mae = np.percentile(mae, percentil)
        ax2.axvline(-p_mae, color=GRIS, linewidth=0.9, linestyle='--')
        ax2.text(-p_mae, ax2.get_ylim()[1], f'P{percentil}={-p_mae:.2f}R ', color=GRIS,
                 fontsize=7, ha='right', va='top')
        ax2.set_title('Máxima Excursión Adversa (R)', fontsize=8, color=AX_FG)
        ax2.set_xlabel('MAE (R)', fontsize=8, color=AX_FG)
        ax2.set_ylabel('Nº Trades', fontsize=8, color=AX_FG)

        subset_mae = mae[mae <= p_mae]
        if len(subset_mae) > 0:
            mu = subset_mae.mean()
            sigma = subset_mae.std()
            ax2.axvline(-mu, color=AZUL, linewidth=1.1, linestyle='-', alpha=0.9)
            ax2.axvline(-(mu - sigma), color=AZUL, linewidth=0.8, linestyle='-', alpha=0.35)
            ax2.axvline(-(mu + sigma), color=AZUL, linewidth=0.8, linestyle='-', alpha=0.35)
            ax2.text(0.98, 0.02,
                     f'μ={-mu:.2f}R\n+σ={-(mu - sigma):.2f}R\n−σ={-(mu + sigma):.2f}R',
                     transform=ax2.transAxes, color=AZUL, fontsize=7,
                     ha='right', va='bottom',
                     bbox=dict(facecolor=FIG_BG, alpha=0.75, edgecolor='none', pad=3))
    else:
        ent = tr['eficiencia_entrada'][mask]
        sal = tr['eficiencia_salida'][mask]
        ent = ent[~np.isnan(ent)]
        sal = sal[~np.isnan(sal)]

        if len(ent) == 0:
            ax1.text(0.5, 0.5, 'Sin datos', ha='center', va='center',
                     color=AX_FG, fontsize=9, transform=ax1.transAxes)
        else:
            ax1.hist(ent, bins=20, range=(0, 100), color=AZUL, alpha=0.8)
            p_ent = np.percentile(ent, percentil)
            ax1.axvline(p_ent, color=GRIS, linewidth=0.9, linestyle='--')
            ax1.text(p_ent, ax1.get_ylim()[1], f' P{percentil}={p_ent:.0f}%',
                     color=GRIS, fontsize=7, ha='left', va='top')
            mu = ent.mean()
            ax1.axvline(mu, color=VERDE, linewidth=1.1, linestyle='-', alpha=0.9)
            ax1.text(0.98, 0.02, f'μ={mu:.0f}%', transform=ax1.transAxes,
                     color=VERDE, fontsize=7, ha='right', va='bottom',
                     bbox=dict(facecolor=FIG_BG, alpha=0.75, edgecolor='none', pad=3))
        ax1.set_xlim(0, 100)
        ax1.set_title('Eficiencia de Entrada', fontsize=8, color=AX_FG)
        ax1.set_xlabel('% del rango tocado por el trade', fontsize=8, color=AX_FG)
        ax1.set_ylabel('Nº Trades', fontsize=8, color=AX_FG)

        if len(sal) == 0:
            ax2.text(0.5, 0.5, 'Sin datos', ha='center', va='center',
                     color=AX_FG, fontsize=9, transform=ax2.transAxes)
        else:
            ax2.hist(sal, bins=20, range=(0, 100), color=AMBAR, alpha=0.8)
            p_sal = np.percentile(sal, percentil)
            ax2.axvline(p_sal, color=GRIS, linewidth=0.9, linestyle='--')
            ax2.text(p_sal, ax2.get_ylim()[1], f' P{percentil}={p_sal:.0f}%',
                     color=GRIS, fontsize=7, ha='left', va='top')
            mu = sal.mean()
            ax2.axvline(mu, color=VERDE, linewidth=1.1, linestyle='-', alpha=0.9)
            ax2.text(0.98, 0.02, f'μ={mu:.0f}%', transform=ax2.transAxes,
                     color=VERDE, fontsize=7, ha='right', va='bottom',
                     bbox=dict(facecolor=FIG_BG, alpha=0.75, edgecolor='none', pad=3))
        ax2.set_xlim(0, 100)
        ax2.set_title('Eficiencia de Salida', fontsize=8, color=AX_FG)
        ax2.set_xlabel('% del rango tocado por el trade', fontsize=8, color=AX_FG)
        ax2.set_ylabel('Nº Trades', fontsize=8, color=AX_FG)

    try:
        dst.fig_mfe_mae.tight_layout(pad=0.6)
    except Exception:
        pass
    dst.canvas_mfe_mae.draw_idle()


# Capas que nacen APAGADAS. El ZigZag es una polilínea que recorre todo el
# histórico por encima del precio: al abrir el gráfico tapa lo que se va a
# mirar, y solo hace falta cuando se están programando entradas por orden
# límite. Sigue en el panel, listo para encenderse a mano.
PREFIJOS_CAPA_OCULTA = ('zigzag:',)


def capa_visible_por_defecto(clave):
    """Estado del ojo de una capa que el usuario todavía no ha tocado."""
    return not clave.startswith(PREFIJOS_CAPA_OCULTA)


class _FilaCapa(QFrame):
    """Una línea de la leyenda: cuadradito de color, nombre y ojo.

    El ojo solo aparece al pasar el ratón por encima de la fila (como en la
    leyenda de TradingView), salvo cuando la capa está apagada: entonces se
    queda visible y tachado, o no habría forma de encontrarla para volver a
    encenderla."""

    def __init__(self, capa, visible, al_conmutar, parent=None):
        super().__init__(parent)
        self.clave = capa['clave']
        self._al_conmutar = al_conmutar
        self.setObjectName('filaCapa')
        self.setToolTip(capa.get('ayuda', ''))
        lay = QHBoxLayout(self)
        lay.setContentsMargins(5, 1, 4, 1)
        lay.setSpacing(5)

        muestra = QLabel(self)
        muestra.setFixedSize(8, 8)
        muestra.setStyleSheet(
            f"background:{capa['color']}; border-radius:2px;")
        lay.addWidget(muestra)

        etiqueta = QLabel(capa['etiqueta'], self)
        etiqueta.setStyleSheet(f"color:{AX_FG}; font-size:10px;")
        lay.addWidget(etiqueta)

        self.ojo = QToolButton(self)
        self.ojo.setCheckable(True)
        self.ojo.setChecked(visible)
        self.ojo.setCursor(Qt.CursorShape.PointingHandCursor)
        self.ojo.setStyleSheet(
            "QToolButton{border:none; background:transparent; font-size:10px;}")
        self.ojo.toggled.connect(self._conmutar)
        lay.addWidget(self.ojo)
        self._pintar_ojo()

    def _conmutar(self, activo):
        self._pintar_ojo()
        self._al_conmutar(self.clave, activo)

    def _pintar_ojo(self):
        activo = self.ojo.isChecked()
        self.ojo.setText('👁' if activo else '🚫')
        self.ojo.setToolTip("Ocultar del gráfico" if activo
                            else "Mostrar en el gráfico")
        # apagada -> siempre visible, o quedaría irrecuperable
        self.ojo.setVisible(not activo or self.underMouse())

    def enterEvent(self, evento):
        self.setProperty('hover', True)
        self._repolir()
        self.ojo.setVisible(True)
        super().enterEvent(evento)

    def leaveEvent(self, evento):
        self.setProperty('hover', False)
        self._repolir()
        self.ojo.setVisible(not self.ojo.isChecked())
        super().leaveEvent(evento)

    def _repolir(self):
        self.style().unpolish(self)
        self.style().polish(self)


class PanelCapas(QWidget):
    """Leyenda interactiva flotando sobre el gráfico, al estilo TradingView:
    una fila por indicador activo, arriba a la izquierda, tenue mientras el
    ratón está fuera y legible al entrar.

    Va como hijo del canvas en vez de en la barra de herramientas para no robar
    sitio permanente a la interfaz por unas capas que dependen del sistema que
    se haya simulado. La contrapartida es que intercepta el ratón donde ocupa
    (arrastrar el gráfico justo debajo del panel no funciona), y de ahí que se
    pueda plegar."""

    OPACIDAD_REPOSO = 0.35

    def __init__(self, canvas, al_conmutar):
        super().__init__(canvas)
        self._al_conmutar = al_conmutar
        self._filas = []
        self._plegado = False
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setStyleSheet(
            "#filaCapa{background:transparent; border-radius:3px;}"
            "#filaCapa[hover=\"true\"]{background:rgba(20,32,56,190);}")
        self._efecto = QGraphicsOpacityEffect(self)
        self._efecto.setOpacity(self.OPACIDAD_REPOSO)
        self.setGraphicsEffect(self._efecto)

        self._lay = QVBoxLayout(self)
        self._lay.setContentsMargins(0, 0, 0, 0)
        self._lay.setSpacing(0)
        self._btn_plegar = QToolButton(self)
        self._btn_plegar.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_plegar.setStyleSheet(
            f"QToolButton{{border:none; background:transparent; color:{AX_FG};"
            " font-size:10px; padding:1px 5px;}}")
        self._btn_plegar.clicked.connect(self._alternar_plegado)
        self._lay.addWidget(self._btn_plegar,
                            alignment=Qt.AlignmentFlag.AlignLeft)

    def poblar(self, capas, estado):
        """Rehace las filas. Se llama en cada redibujado: el sistema simulado
        puede haber cambiado y con él la lista de indicadores."""
        for fila in self._filas:
            self._lay.removeWidget(fila)
            fila.deleteLater()
        self._filas = []
        for capa in capas:
            fila = _FilaCapa(capa,
                             estado.get(capa['clave'],
                                        capa_visible_por_defecto(capa['clave'])),
                             self._al_conmutar, self)
            self._lay.insertWidget(len(self._filas), fila)
            self._filas.append(fila)
        self.setVisible(bool(capas))
        self._aplicar_plegado()

    def _alternar_plegado(self):
        self._plegado = not self._plegado
        self._aplicar_plegado()

    def _aplicar_plegado(self):
        n = len(self._filas)
        for fila in self._filas:
            fila.setVisible(not self._plegado)
        self._btn_plegar.setText(f"▸ {n} indicadores" if self._plegado else "▾")
        self._btn_plegar.setToolTip("Desplegar la lista de capas" if self._plegado
                                    else "Plegar la lista de capas")
        self.adjustSize()

    def enterEvent(self, evento):
        self._efecto.setOpacity(1.0)
        super().enterEvent(evento)

    def leaveEvent(self, evento):
        self._efecto.setOpacity(self.OPACIDAD_REPOSO)
        super().leaveEvent(evento)

    def reposicionar(self):
        """Arriba a la izquierda del canvas. El eje de precio va a la derecha
        (yaxis.tick_right), así que ese lado está ocupado por las etiquetas."""
        self.adjustSize()
        self.move(8, 6)
        self.raise_()


class ResultadosWidget(QWidget):
    favorito_guardado = pyqtSignal()
    # se emite al terminar de pintar todos los gráficos de un resultado; lo usa
    # TabBacktest para ocultar el overlay de carga cuando el render (que puede
    # tardar segundos) ya ha acabado — antes se ocultaba antes y la ruedita
    # desaparecía justo en el momento del render pesado.
    graficos_listos = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._payload = None            # el que se está mostrando (puede ir filtrado)
        self._payload_base = None       # el del backtest, sin filtrar
        self._cache_dir = {}            # dirección -> payload derivado
        self._wfa_cache = None
        self._wfa_ts = None
        self._wfa_equity = None
        self._modo_grafico = 'velas'
        self._art_datos = []
        self._actualizando_xlim = False
        self._y_manual = False

        # Repintar los paneles de matplotlib cuesta ~250 ms el gráfico de precio
        # y hasta unos segundos el conjunto con series largas, mientras que
        # recalcular las métricas cuesta ~2 ms. Al cambiar de dirección las
        # tablas se rellenan en el acto y los gráficos se aplazan un ciclo del
        # bucle de eventos, de modo que varios cambios seguidos colapsan en un
        # solo repintado.
        self._graficos_sucios = False
        self._timer_graficos = QTimer(self)
        self._timer_graficos.setSingleShot(True)
        self._timer_graficos.setInterval(0)
        self._timer_graficos.timeout.connect(self._pintar_graficos)

        # eje X del gráfico principal: índice de vela (_x_full) + su traducción
        # a fecha (_ts_full), ambos poblados en _dibujar_principal
        self._x_full = None
        self._ts_full = None

        # estado de trades (poblado en _dibujar_principal) para recortar
        # compra/venta/trayecto/cajas de salida al rango visible en cada frame
        self._tr = None
        self._compra_idx_full = None
        self._venta_idx_full = None
        self._trayecto_segmentos_full = None
        self._salida_segmentos_full = None
        self._salida_cuadros_full = None
        self._salida_colores_full = None
        self._scatter_compra = None
        self._scatter_venta = None
        # tramos de entrada escalonada (aperturas NO son un cierre: viven en
        # resultado['entradas'], no en 'trades' — ver core/backtest). Solo
        # los tramos 2+ se marcan aquí (el 1º ya lo pinta compra/venta).
        self._entr = None
        self._tramo_compra_idx_full = None
        self._tramo_venta_idx_full = None
        self._scatter_tramo_compra = None
        self._scatter_tramo_venta = None
        # marcas del precio REAL de ejecución (tick) y su conector con la
        # flecha, que va desplazada fuera de la vela
        self._offset_flecha = 0.0
        self._orden_por_barra = {}
        self._compra_precio_full = None
        self._venta_precio_full = None
        self._tramo_compra_precio_full = None
        self._tramo_venta_precio_full = None
        self._scatter_nivel_compra = None
        self._scatter_nivel_venta = None
        self._scatter_nivel_tramo_compra = None
        self._scatter_nivel_tramo_venta = None
        self._art_trayecto = None
        self._art_salida_cuadros = None
        self._art_salida_segmentos = None
        self._art_stop_track = None
        self._art_entrada_track = None
        self._art_zona_riesgo = None
        # último precio (línea + badge en la escala derecha), como LWC/TradingView
        self._art_ultimo_precio = None
        self._lbl_ultimo_precio = None
        self._art_fijos_dinamicos = []
        self._art_overlays_extra = []
        self._art_osciladores = []
        # Line2D densos de overlays/osciladores etiquetados con su serie
        # completa (_x_all/_y_all) para decimarlos a la ventana visible en
        # cada frame (ver _etiquetar_decimables / _redibujar_datos)
        self._art_decimables = []

        # paneles apilados (precio + osciladores activos) y sus proporciones
        # de altura — persisten entre redibujados (pan/zoom/centrar trade)
        # para que un redimensionado manual del usuario no se pierda; solo
        # se usan valores por defecto la primera vez que aparece cada tipo
        self._paneles = []
        self._pesos_paneles = {}
        self._pesos_paneles_prev = {}
        # escala vertical que el usuario haya estirado a mano en cada panel,
        # {kind: (lo, hi)}. Indexado por TIPO igual que _pesos_paneles: así
        # sobrevive a apagar y reencender el panel y a los redibujados
        # completos, en vez de perderse con la posición.
        self._ylim_paneles = {}

        # capas del gráfico (ver _catalogo_capas): {clave: visible}. Persiste
        # entre redibujados y entre backtests — una clave lleva sus parámetros
        # dentro, así que apagar la EMA(20) no afecta a la EMA(50) ni se
        # arrastra a un sistema que use otras medias.
        self._capas_estado = {}
        self._capas_catalogo = []
        self._colores_capas = {}

        # estado de blitting (pan/zoom fluido) — ver _iniciar_sesion_blit
        self._blit_bg = None
        # sesión de blit activa (arrastre o ráfaga de scroll) y velas/overlays
        # pendientes de redibujar: el redibujado se difiere a
        # _pintar_frame_pendiente para que una ráfaga de eventos haga UN solo
        # redibujado por ciclo de eventos en vez de uno por evento de ratón
        # (ver _on_scroll / _on_xlim_changed)
        self._sesion_activa = False
        self._datos_sucios = False
        self._scroll_timer = QTimer(self)
        self._scroll_timer.setSingleShot(True)
        self._scroll_timer.timeout.connect(self._finalizar_blit)
        # coalescencia de frames de arrastre: el ratón manda eventos más
        # deprisa de lo que se rasteriza un frame, así que se pinta uno solo
        # por vuelta del bucle de eventos (ver _solicitar_frame_blit). El
        # intervalo de 20 ms (≈50 fps máx) evita que una ráfaga de scroll muy
        # rápida acumule frames pendientes: los eventos intermedios solo
        # mueven el xlim (~0,15 ms) y el gráfico pinta siempre el estado más
        # reciente, sin quedarse atrás del cursor.
        self._ax_frame_pendiente = None
        self._timer_frame = QTimer(self)
        self._timer_frame.setSingleShot(True)
        self._timer_frame.setInterval(20)
        self._timer_frame.timeout.connect(self._pintar_frame_pendiente)
        # zoom con rueda animado (easing hacia el objetivo, como LWC):
        # _zoom_target es el xlim destino de la ráfaga y _timer_zoom lo
        # interpola tick a tick (ver _on_scroll / _avanzar_zoom)
        self._zoom_target = None
        self._zoom_pasos = 0
        self._timer_zoom = QTimer(self)
        self._timer_zoom.setInterval(16)
        self._timer_zoom.timeout.connect(self._avanzar_zoom)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll = scroll
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)

        # Título y selector de dirección van FUERA del scroll: la columna de
        # bloques es larga y ambos deben seguir a la vista al bajar por ella —
        # sobre todo el selector, que cambia el significado de todo lo de abajo.
        fila_titulo = QHBoxLayout()
        self.lbl_titulo = QLabel("Ejecuta un backtest desde el Optimizador")
        self.lbl_titulo.setObjectName("titulo")
        self.lbl_titulo.setWordWrap(True)
        fila_titulo.addWidget(self.lbl_titulo, 1)
        self.btn_favorito = QPushButton("⭐ Guardar como favorito")
        self.btn_favorito.setToolTip(
            "Guarda el activo, la temporalidad y el/los setup(s) de este "
            "resultado para volver a cargarlos desde el Constructor")
        self.btn_favorito.setEnabled(False)
        self.btn_favorito.clicked.connect(self._guardar_favorito)
        fila_titulo.addWidget(self.btn_favorito)
        self.btn_exportar = QPushButton("📤 Exportar código")
        self.btn_exportar.setToolTip(
            "Traduce este sistema a código de MetaTrader 5 o TradingView, "
            "con la estructura de carpetas que espera cada una")
        self.btn_exportar.setEnabled(False)
        self.btn_exportar.clicked.connect(self._exportar_codigo)
        fila_titulo.addWidget(self.btn_exportar)
        fila_titulo.addWidget(_icono_ayuda(
            "Genera el código ejecutable de este sistema para otra "
            "plataforma.\n\n"
            "Cada setup sale a un archivo independiente, con los parámetros "
            "del backtest como valores por defecto editables.\n\n"
            "Antes de escribir nada se enseña qué se pierde en la traducción: "
            "hay cosas que una plataforma no puede hacer (Pine Script no tiene "
            "calendario económico) y otras que aún no emite el generador. En "
            "los dos casos se avisa y se explica la consecuencia."))
        root.addLayout(fila_titulo)
        # avisos del motor (filtros pedidos que no se pudieron aplicar). Va
        # fuera del scroll y pegado al título: si se pierde de vista, el
        # backtest se lee como si el filtro hubiera actuado.
        self.lbl_avisos = QLabel("")
        self.lbl_avisos.setWordWrap(True)
        self.lbl_avisos.setVisible(False)
        self.lbl_avisos.setStyleSheet(
            "background-color: #2a2010; color: #ffb74d; border: 1px solid #5a4520;"
            " border-radius: 4px; padding: 6px; font-size: 11px;")
        root.addWidget(self.lbl_avisos)
        root.addLayout(self._construir_fila_direccion())
        root.addWidget(scroll)

        cont = QWidget()
        lay = QVBoxLayout(cont)
        lay.setSpacing(10)
        scroll.setWidget(cont)

        # métricas IS/OOS/Total
        self.tabla_metricas = QTableWidget(len(_FILAS_METRICAS), 4)
        self.tabla_metricas.setHorizontalHeaderLabels(['Métrica', 'IS', 'OOS', 'Total'])
        self.tabla_metricas.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.tabla_metricas.verticalHeader().setVisible(False)
        self.tabla_metricas.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        # sin barra de scroll: la altura se fija tras rellenar (ver
        # _fijar_altura_tabla en _pintar_graficos), todo visible de golpe
        self.tabla_metricas.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        lay.addWidget(self.tabla_metricas)

        # gráfico principal: log-return + flechas + IS/OOS
        self.fig = Figure(figsize=(9, 4.8), facecolor=FIG_BG)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setMinimumHeight(480)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.toolbar.setStyleSheet("background-color: #141e30;")
        bombear_eventos()
        fila_zoom = QHBoxLayout()
        fila_zoom.addWidget(self.toolbar)
        fila_zoom.addWidget(QLabel("Rango:"))
        self.fecha_ini = QDateEdit(calendarPopup=True)
        self.fecha_fin = QDateEdit(calendarPopup=True)
        btn_rango = QPushButton("Aplicar")
        btn_rango.clicked.connect(self._aplicar_rango)
        btn_reset = QPushButton("Todo")
        btn_reset.clicked.connect(self._reset_rango)
        fila_zoom.addWidget(self.fecha_ini)
        fila_zoom.addWidget(self.fecha_fin)
        fila_zoom.addWidget(btn_rango)
        fila_zoom.addWidget(btn_reset)
        self.btn_modo_grafico = QPushButton("🕯 Velas")
        self.btn_modo_grafico.setCheckable(True)
        self.btn_modo_grafico.setChecked(True)
        self.btn_modo_grafico.clicked.connect(self._toggle_modo_grafico)
        fila_zoom.addWidget(self.btn_modo_grafico)
        self.chk_stop = QCheckBox("Mostrar operación")
        self.chk_stop.setToolTip(
            "Dibuja el nivel de stop-loss (rojo) de cada operación, y en "
            "verde la zona entre el precio de entrada y el de salida real "
            "cuando la operación cerró en ganancia (por take-profit, señal "
            "contraria u otro motivo).")
        self.chk_stop.toggled.connect(self._toggle_stop_loss)
        fila_zoom.addWidget(self.chk_stop)
        self.chk_trayecto = QCheckBox("Mostrar trayecto")
        self.chk_trayecto.setToolTip(
            "Dibuja una línea gris entre el precio de entrada y el de salida "
            "de cada operación (recorrido del trade).")
        self.chk_trayecto.toggled.connect(self._toggle_trayecto)
        fila_zoom.addWidget(self.chk_trayecto)
        self.chk_noticias = QCheckBox("Mostrar noticias")
        self.chk_noticias.setToolTip(
            "Próximamente: el calendario económico está en revisión (sin "
            "proveedor gratuito fiable).")
        self.chk_noticias.setEnabled(False)
        fila_zoom.addWidget(self.chk_noticias)
        # conmutador de vista: matplotlib (Clásica) <-> Lightweight Charts (Moderna)
        self.btn_vista = QPushButton("🖥 Vista: Clásica")
        self.btn_vista.setCheckable(True)
        self.btn_vista.setToolTip(
            "Alterna entre la gráfica clásica (matplotlib) y una vista moderna "
            "estilo TradingView (Lightweight Charts).")
        self.btn_vista.clicked.connect(self._toggle_vista)
        if not WEBENGINE_OK:
            self.btn_vista.setEnabled(False)
            self.btn_vista.setToolTip("Vista moderna no disponible: falta "
                                      "instalar PyQt6-WebEngine.")
        fila_zoom.addWidget(self.btn_vista)
        fila_zoom.addStretch()
        lay.addLayout(fila_zoom)
        # el canvas de matplotlib y la vista LWC comparten hueco en un stack
        self.lwc = LwcChart()
        self.stack_grafico = QStackedWidget()
        self.stack_grafico.addWidget(self.canvas)   # índice 0 = clásica
        self.stack_grafico.addWidget(self.lwc)       # índice 1 = moderna
        lay.addWidget(self.stack_grafico, 1)

        # arrastrar los márgenes de los ejes (estilo TradingView/MT4): eje Y
        # (precio) para estirar/comprimir la escala, eje X (fecha) para zoom
        # temporal, arrastrando directamente sobre la franja de etiquetas
        self._drag_modo = None
        self._drag_inicio = None
        self._drag_lim0 = None
        # cruceta de hover: artistas (recreados en cada _dibujar_principal) y
        # su bitmap de fondo, con el flag que evita capturarlo a media captura
        # del fondo de arrastre (ver _capturar_fondo_hover)
        self._art_cruceta = []
        self._cruceta_v = []
        self._cruceta_h = []
        self._cruceta_lbl_y = []
        self._cruceta_lbl_x = None
        self._bg_hover = None
        self._hover_pintado = False
        self._capturando_fondo_drag = False
        self._patron_fecha = '%Y-%m-%d'
        self.canvas.mpl_connect('button_press_event', self._on_press_ejes)
        self.canvas.mpl_connect('motion_notify_event', self._on_motion_ejes)
        self.canvas.mpl_connect('button_release_event', self._on_release_ejes)
        self.canvas.mpl_connect('scroll_event', self._on_scroll)
        self.canvas.mpl_connect('resize_event', self._on_resize_canvas)
        self.canvas.mpl_connect('draw_event', self._capturar_fondo_hover)
        self.canvas.mpl_connect('figure_leave_event',
                                lambda _e: self._ocultar_cruceta())

        # leyenda interactiva flotando sobre el gráfico (ver PanelCapas): hija
        # del canvas, así que se reposiciona con el eventFilter de abajo
        self.panel_capas = PanelCapas(self.canvas, self._conmutar_capa)
        self.panel_capas.hide()
        self.canvas.installEventFilter(self)

        # curva de equity (IS vs OOS)
        self.grp_equity = QGroupBox()
        header_eq = QHBoxLayout()
        # el texto cambia con el selector de dirección: al filtrar, la curva ya
        # no es la del motor sino la reconstruida desde cierres realizados
        self._lbl_equity = QLabel("Curva de Equity (IS vs OOS)")
        self._lbl_equity.setObjectName("titulo")
        header_eq.addWidget(self._lbl_equity)
        header_eq.addWidget(_icono_ayuda_popup(
            "La curva de equity trazada es la que produjo la simulación del backtest "
            "vela a vela; el conmutador de modo permite verla como retorno %, capital "
            "en $, log-retorno % o como drawdown continuo desde el máximo acumulado. "
            "El tramo OOS también se recalcula de forma aislada, rebasando su equity "
            "al valor justo en el punto de corte.",
            "\"IS\" es el tramo con el que se ajustó o seleccionó el sistema; \"OOS\" "
            "es el tramo posterior no usado para ajustar nada; \"OOS (aislado)\" "
            "muestra el rendimiento de solo ese tramo, sin arrastrar lo acumulado en "
            "IS; \"Total\" encadena ambos sin reiniciar.",
            "Sirve para juzgar si el sistema mantuvo su comportamiento fuera de la "
            "muestra con la que se construyó, y si superó a la alternativa pasiva de "
            "Buy & Hold.",
            "Un IS con buena pendiente seguido de un OOS plano o negativo es la señal "
            "clásica de sobreajuste (el sistema memorizó el pasado, no encontró una "
            "ventaja real); un OOS que mantiene una pendiente similar al IS es la "
            "evidencia más fuerte de que el sistema generaliza."))
        header_eq.addStretch()
        self.chk_bh = QCheckBox("Mostrar Buy && Hold")
        self.chk_bh.setToolTip(
            "Dibuja la evolución de una posición 'comprar y mantener' del "
            "mismo activo durante todo el periodo, con el mismo capital "
            "inicial, para comparar si el sistema generó alfa frente a una "
            "estrategia pasiva.")
        self.chk_bh.toggled.connect(lambda _: self._dibujar_equity(self._payload))
        header_eq.addWidget(self.chk_bh)
        self.combo_eq_modo = QComboBox()
        self.combo_eq_modo.setMinimumWidth(130)
        for m in MODOS_EQUITY:
            self.combo_eq_modo.addItem(m)
        self.combo_eq_modo.currentIndexChanged.connect(
            lambda _i: self._dibujar_equity(self._payload))
        header_eq.addWidget(self.combo_eq_modo)
        lay_eq = QVBoxLayout(self.grp_equity)
        lay_eq.addLayout(header_eq)
        self.fig_equity = Figure(figsize=(9, 2), facecolor=FIG_BG)
        self.canvas_equity = FigureCanvasQTAgg(self.fig_equity)
        self.canvas_equity.setMinimumHeight(160)
        self.canvas_equity.installEventFilter(self)
        lay_eq.addWidget(self.canvas_equity)
        self.grp_equity.setVisible(False)
        lay.addWidget(self.grp_equity)
        bombear_eventos()

        # código del sistema ejecutado (constancia exacta de qué se backtesteó)
        from PyQt6.QtWidgets import QPlainTextEdit
        self.grp_codigo = QGroupBox("Código del sistema ejecutado (clic para desplegar)")
        self.grp_codigo.setCheckable(True)
        self.grp_codigo.setChecked(False)
        lay_cod = QVBoxLayout(self.grp_codigo)
        lay_cod.insertLayout(0, _fila_ayuda(
            "El mismo pseudocódigo del constructor, pero congelado tal y "
            "como se ejecutó en ESTE backtest — útil para confirmar después "
            "qué configuración exacta produjo estos resultados."))
        self.txt_codigo = QPlainTextEdit()
        self.txt_codigo.setReadOnly(True)
        self.txt_codigo.setStyleSheet(
            "QPlainTextEdit { background-color: #0d1424; color: #8fb3d9;"
            "border: 1px solid #253a60; border-radius: 4px;"
            "font-family: Consolas, monospace; font-size: 11px; }")
        self.txt_codigo.setMinimumHeight(150)
        self.txt_codigo.setVisible(False)
        self.grp_codigo.toggled.connect(self.txt_codigo.setVisible)
        lay_cod.addWidget(self.txt_codigo)
        self.grp_codigo.setVisible(False)
        lay.addWidget(self.grp_codigo)

        # métricas por setup (¿aporta cada forma de entrada del sistema?)
        self.grp_setups = QGroupBox("Métricas por setup")
        lay_ms = QVBoxLayout(self.grp_setups)
        lay_ms.insertLayout(0, _fila_ayuda(
            "Cuando el sistema tiene más de un setup, esta tabla desglosa "
            "las métricas (trades, win rate, expectancy...) por separado "
            "para cada uno, además del resultado conjunto de arriba."))
        self.tabla_setups = QTableWidget(0, 6)
        self.tabla_setups.setHorizontalHeaderLabels(
            ['Setup', 'Riesgo %', 'Trades', 'Win rate', 'PnL', 'Expectancy %'])
        self.tabla_setups.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.tabla_setups.verticalHeader().setVisible(False)
        self.tabla_setups.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        # sin barra de scroll: la altura se fija tras rellenar, todo visible
        self.tabla_setups.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        lay_ms.addWidget(self.tabla_setups)
        self.grp_setups.setVisible(False)
        lay.addWidget(self.grp_setups)

        # tabla de trades
        fila_tr = QHBoxLayout()
        lbl_tr = QLabel("Trades (clic en una fila para centrar el gráfico)")
        lbl_tr.setObjectName("titulo")
        fila_tr.addWidget(lbl_tr)
        fila_tr.addWidget(_icono_ayuda(
            "Lista de todas las operaciones del backtest, con su motivo de "
            "cierre (señal, stop, TP, tiempo, parcial...). Haz clic en una "
            "fila para centrar el gráfico principal en esa operación."))
        fila_tr.addStretch()
        btn_lista_completa = QPushButton("Lista completa")
        btn_lista_completa.clicked.connect(self._abrir_lista_completa)
        fila_tr.addWidget(btn_lista_completa)
        lay.addLayout(fila_tr)
        self._dlg_trades = None
        self.tabla_trades = QTableWidget(0, 13)
        self.tabla_trades.setHorizontalHeaderLabels(
            ['Entrada', 'Salida', 'Dir', 'Setup', 'P. entrada', 'P. salida',
             'PnL', 'Motivo', 'MFE (R)', 'MAE (R)', 'ETD (R)', 'Ent. Ef %',
             'Sal. Ef %'])
        self.tabla_trades.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.tabla_trades.verticalHeader().setVisible(False)
        self.tabla_trades.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.tabla_trades.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.tabla_trades.setMaximumHeight(220)
        self.tabla_trades.setSortingEnabled(False)
        self.tabla_trades._sort_col = -1
        self.tabla_trades._sort_order = Qt.SortOrder.AscendingOrder
        self.tabla_trades.horizontalHeader().setSortIndicatorShown(True)
        self.tabla_trades.horizontalHeader().sectionClicked.connect(
            lambda c: self._ordenar_tabla_trades(self.tabla_trades, c))
        self.tabla_trades.cellClicked.connect(self._centrar_trade)
        lay.addWidget(self.tabla_trades)

        # WFA
        self.grp_wfa = QGroupBox()
        header_wfa = QHBoxLayout()
        lbl_wfa_titulo = QLabel("Walk-Forward Analysis")
        lbl_wfa_titulo.setObjectName("titulo")
        header_wfa.addWidget(lbl_wfa_titulo)
        header_wfa.addWidget(_icono_ayuda_popup(
            "El histórico se divide en ventanas sucesivas y se mide el rendimiento de "
            "cada una por separado con las mismas métricas del backtest global, en vez "
            "de mezclarlas en un único resultado agregado.",
            "Cada barra/punto es una ventana temporal independiente; el selector de "
            "vista permite leerlas como retorno individual, retorno acumulado, retorno "
            "contra drawdown, número de trades, o como la curva de equity de encadenar "
            "solo los tramos OOS.",
            "Sirve para distinguir un sistema realmente consistente (todas las "
            "ventanas rinden de forma parecida) de uno que solo funcionó gracias a un "
            "tramo concreto y afortunado del histórico.",
            "Barras de retorno mayoritariamente verdes y de magnitud parecida entre "
            "ventanas indican consistencia; si una única ventana concentra casi todo "
            "el beneficio (una barra muy alta entre varias planas o rojas), el "
            "resultado global depende de suerte de tramo y es frágil aunque la métrica "
            "agregada del backtest se vea bien."))
        header_wfa.addStretch()
        self.combo_wfa_modo = QComboBox()
        self.combo_wfa_modo.setMinimumWidth(160)
        for m in MODOS_WFA:
            self.combo_wfa_modo.addItem(m)
        self.combo_wfa_modo.currentIndexChanged.connect(
            lambda _i: self._dibujar_wfa(self._wfa_cache, self._wfa_ts, self._wfa_equity))
        header_wfa.addWidget(self.combo_wfa_modo)
        lay_wfa = QVBoxLayout(self.grp_wfa)
        lay_wfa.addLayout(header_wfa)
        self.fig_wfa = Figure(figsize=(9, 2.2), facecolor=FIG_BG)
        self.canvas_wfa = FigureCanvasQTAgg(self.fig_wfa)
        self.canvas_wfa.setMinimumHeight(200)
        self.canvas_wfa.installEventFilter(self)
        lay_wfa.addWidget(self.canvas_wfa)
        bombear_eventos()
        self.tabla_wfa = QTableWidget(0, 6)
        self.tabla_wfa.setHorizontalHeaderLabels(
            ['Ventana', 'Periodo', 'Trades', 'Win rate', 'Retorno %', 'Max DD %'])
        self.tabla_wfa.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.tabla_wfa.verticalHeader().setVisible(False)
        self.tabla_wfa.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.tabla_wfa.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        lay_wfa.addWidget(self.tabla_wfa)
        self.grp_wfa.setVisible(False)
        lay.addWidget(self.grp_wfa)

        # Montecarlo
        self.grp_mc = QGroupBox("Montecarlo (1000 remuestreos con reemplazo de los trades)")
        lay_mc = QVBoxLayout(self.grp_mc)
        lay_mc.insertLayout(0, _fila_ayuda_popup(
            "Bootstrap con reemplazo sobre los trades del backtest: en cada "
            "simulación se extrae al azar una muestra del mismo tamaño —un "
            "trade puede repetirse y otro omitirse— y se recalcula la curva "
            "de equity.",
            "El panel izquierdo es el abanico de curvas (percentiles 5/50/95); "
            "el central, la distribución de retorno final; el derecho, la del "
            "máximo drawdown. Separa qué parte del resultado depende del azar "
            "de la muestra de lo atribuible a la estrategia.",
            "Para estimar el rango plausible de resultados futuros, no solo el "
            "número puntual del backtest, y leer la probabilidad de pérdida "
            "(quedar por debajo del capital inicial) o de ruina.",
            "Las cifras (prob. de pérdida, prob. de ruina, retorno y Max DD "
            "medianos) resumen el riesgo más allá del resultado único. Una "
            "banda p5-p95 muy ancha indica alta sensibilidad a la muestra; "
            "una estrecha, resultado robusto."))
        self.lbl_mc = QLabel("")
        self.lbl_mc.setObjectName("campo")
        lay_mc.addWidget(self.lbl_mc)
        self.fig_mc = Figure(figsize=(9, 2.6), facecolor=FIG_BG)
        self.canvas_mc = FigureCanvasQTAgg(self.fig_mc)
        self.canvas_mc.setMinimumHeight(230)
        self.canvas_mc.installEventFilter(self)
        lay_mc.addWidget(self.canvas_mc)
        self.grp_mc.setVisible(False)
        lay.addWidget(self.grp_mc)
        bombear_eventos()

        # Análisis de Eficiencia (MFE/MAE)
        self.grp_mfe_mae = QGroupBox()
        header_mfe = QHBoxLayout()
        lbl_mfe_titulo = QLabel("Análisis de Eficiencia (MFE/MAE)")
        lbl_mfe_titulo.setObjectName("titulo")
        header_mfe.addWidget(lbl_mfe_titulo)
        header_mfe.addWidget(_icono_ayuda_popup(
            "MFE y MAE miden, en múltiplos de R, cuánto llegó a moverse el precio a "
            "favor y en contra durante cada trade antes de cerrarse, más allá del "
            "resultado con el que se cerró finalmente. La distancia de un punto a la "
            "diagonal y=x en la vista de dispersión es el ETD (lo que ese trade dejó "
            "de capturar).",
            "Un MFE alto con un R realizado muy inferior significa que el trade llegó "
            "a estar muy en ganancia pero se cerró con mucho menos (beneficio no "
            "capturado); un MAE profundo con el trade aun así ganador indica que el "
            "precio estuvo cerca de tocar stop antes de girar a favor.",
            "Sirve para diagnosticar si el stop-loss y el take-profit están bien "
            "calibrados al comportamiento real del precio durante el trade.",
            "Si el percentil marcado del MAE está muy por debajo (en valor absoluto) "
            "del stop configurado, hay margen para ajustarlo más ceñido sin saltarse "
            "la mayoría de los trades; si el percentil del MFE supera claramente el "
            "take-profit configurado, el sistema probablemente esté cerrando demasiado "
            "pronto y dejando beneficio sobre la mesa de forma sistemática."))
        header_mfe.addStretch()
        self.combo_mfe_modo = QComboBox()
        self.combo_mfe_modo.setMinimumWidth(160)
        for m in MODOS_MFE_MAE:
            self.combo_mfe_modo.addItem(m)
        self.combo_mfe_modo.currentIndexChanged.connect(
            lambda _i: self._dibujar_mfe_mae(self._payload))
        header_mfe.addWidget(self.combo_mfe_modo)
        self.combo_mfe_filtro = QComboBox()
        self.combo_mfe_filtro.addItems(['Todas', 'Ganadoras', 'Perdedoras'])
        self.combo_mfe_filtro.currentIndexChanged.connect(
            lambda _i: self._dibujar_mfe_mae(self._payload))
        header_mfe.addWidget(self.combo_mfe_filtro)
        header_mfe.addWidget(QLabel("Percentil:"))
        self.slider_percentil = QSlider(Qt.Orientation.Horizontal)
        self.slider_percentil.setRange(50, 99)
        self.slider_percentil.setValue(80)
        self.slider_percentil.setMaximumWidth(100)
        self.spin_percentil = QSpinBox()
        self.spin_percentil.setRange(50, 99)
        self.spin_percentil.setValue(80)
        self.spin_percentil.setSuffix('%')
        self.slider_percentil.valueChanged.connect(self.spin_percentil.setValue)
        self.spin_percentil.valueChanged.connect(self.slider_percentil.setValue)
        self.spin_percentil.valueChanged.connect(
            lambda _v: self._dibujar_mfe_mae(self._payload))
        header_mfe.addWidget(self.slider_percentil)
        header_mfe.addWidget(self.spin_percentil)
        lay_mfe = QVBoxLayout(self.grp_mfe_mae)
        lay_mfe.addLayout(header_mfe)
        self.fig_mfe_mae = Figure(figsize=(9, 2.6), facecolor=FIG_BG)
        self.canvas_mfe_mae = FigureCanvasQTAgg(self.fig_mfe_mae)
        self.canvas_mfe_mae.setMinimumHeight(230)
        self.canvas_mfe_mae.installEventFilter(self)
        lay_mfe.addWidget(self.canvas_mfe_mae)
        self.grp_mfe_mae.setVisible(False)
        lay.addWidget(self.grp_mfe_mae)

        lay.addStretch()
        bombear_eventos()

    def eventFilter(self, obj, event):
        """Los paneles WFA/Montecarlo son estáticos (sin zoom propio), pero al
        ser FigureCanvasQTAgg absorben la rueda del ratón igualmente y no la
        dejan pasar al QScrollArea de la pestaña — se reenvía a mano."""
        if event.type() == QEvent.Type.Wheel and obj in (
                self.canvas_wfa, self.canvas_mc, self.canvas_equity,
                self.canvas_mfe_mae):
            sb = self._scroll.verticalScrollBar()
            sb.setValue(sb.value() - event.angleDelta().y())
            return True
        # el panel de capas flota sobre el canvas del gráfico principal, así
        # que hay que recolocarlo cuando ese canvas cambia de tamaño
        if event.type() == QEvent.Type.Resize and obj is self.canvas:
            self.panel_capas.reposicionar()
        return super().eventFilter(obj, event)

    # ── selector de dirección ──
    def _construir_fila_direccion(self):
        """Fila `Mostrar: Todos | Solo Largos | Solo Cortos | Comparar lados`.

        Los cuatro modos son excluyentes. Los tres primeros filtran TODA la
        vista (métricas, gráfico de precio, curva de capital, setups, trades,
        WFA, Montecarlo y MFE/MAE); el cuarto deja la vista sin filtrar y solo
        cambia las columnas de la tabla de métricas a Largos / Cortos / Total."""
        fila = QVBoxLayout()
        fila.setSpacing(2)
        sel = QHBoxLayout()
        lbl = QLabel("Mostrar:")
        lbl.setObjectName("titulo")
        sel.addWidget(lbl)
        self._grupo_dir = QButtonGroup(self)
        for i, texto in enumerate(('Todos', 'Solo Largos', 'Solo Cortos',
                                   'Comparar lados')):
            rb = QRadioButton(texto)
            rb.setAutoExclusive(True)
            rb.setEnabled(False)
            self._grupo_dir.addButton(rb, i)
            sel.addWidget(rb)
        self._grupo_dir.button(_MODO_TODOS).setChecked(True)
        self._grupo_dir.idClicked.connect(lambda _i: self._aplicar_direccion())
        sel.addWidget(_icono_ayuda_popup(
            "Se toman los trades que ya produjo el backtest y se descartan los "
            "del lado no seleccionado, sin re-simular. Como la curva de capital "
            "del motor está marcada a mercado y mezcla ambos lados, al filtrar "
            "se reconstruye componiendo únicamente los retornos de los trades "
            "que sobreviven: la curva de cierres realizados de operar solo ese "
            "lado. \"Todos\" no pasa por ese proceso, así que muestra "
            "exactamente el resultado del backtest.",
            "Las métricas de trade (nº, win rate, profit factor, expectancy, "
            "SQN, payoff, eficiencias) son exactas para el lado elegido. Las de "
            "curva (retorno, max drawdown, Sharpe, R², Ulcer, tiempos de "
            "recuperación) se miden sobre la curva reconstruida — su tooltip lo "
            "recuerda cuando hay un filtro activo.",
            "Sirve para detectar si el sistema tiene edge en un solo lado: un "
            "sesgo direccional oculto, un stop que solo funciona en un sentido, "
            "o una pata que solo añade ruido y comisiones. \"Comparar lados\" "
            "pone largos y cortos uno al lado del otro para verlo de un vistazo.",
            "Si un lado tiene expectancy negativa y el otro la sostiene, la "
            "lectura es que el sistema es en realidad unidireccional y la pata "
            "mala está restando; antes de amputarla conviene mirar su nº de "
            "trades, porque con muestras pequeñas la diferencia puede ser azar."))
        sel.addStretch()
        fila.addLayout(sel)
        self.lbl_direccion = QLabel("")
        self.lbl_direccion.setObjectName("campo")
        fila.addWidget(self.lbl_direccion)
        return fila

    def _modo_direccion(self):
        return self._grupo_dir.checkedId()

    def _actualizar_botones_direccion(self):
        """Habilita los modos que tienen sentido para este backtest y devuelve
        el resumen de reparto largos/cortos."""
        tr = self._payload_base['resultado']['trades']
        n_l = int((tr['dir'] > 0).sum())
        n_c = int((tr['dir'] < 0).sum())
        self._grupo_dir.button(_MODO_TODOS).setEnabled(True)
        self._grupo_dir.button(_MODO_LARGOS).setEnabled(n_l > 0)
        self._grupo_dir.button(_MODO_CORTOS).setEnabled(n_c > 0)
        # comparar lados solo dice algo si el sistema opera en ambos sentidos
        self._grupo_dir.button(_MODO_COMPARAR).setEnabled(n_l > 0 and n_c > 0)
        boton = self._grupo_dir.checkedButton()
        if boton is None or not boton.isEnabled():
            self._grupo_dir.button(_MODO_TODOS).setChecked(True)
        return n_l, n_c

    def _payload_de_direccion(self, direccion):
        """Payload equivalente al del backtest pero con solo los trades de una
        dirección. Al ser intercambiable con el original, todos los
        renderizadores lo filtran sin enterarse."""
        if direccion in self._cache_dir:
            return self._cache_dir[direccion]
        p = self._payload_base
        cap0 = float(p['config'].get('capital_inicial', 10000.0))
        res = resultado_filtrado(p['resultado'], direccion, cap0)
        n, corte, va = p['n_velas'], p['corte'], p.get('velas_anio')
        vd = p.get('velas_dia')
        pf = {**p, 'resultado': res,
              'metricas': {'IS': calcular_metricas(res, 0, corte, va, vd),
                           'OOS': calcular_metricas(res, corte, n, va, vd),
                           'Total': calcular_metricas(res, 0, n, va, vd)},
              'montecarlo': (montecarlo(res['trades'], cap0, n_sims=1000,
                                        semilla=1234)
                             if res['n_trades'] else None),
              'wfa': _wfa_filtrado(p.get('wfa'), res, va, vd)}
        self._cache_dir[direccion] = pf
        return pf

    @_no_crash
    def _aplicar_direccion(self):
        """Recalcula y repinta la vista para el modo seleccionado. Es el único
        punto de entrada del render: lo llaman tanto `mostrar()` como los
        botones del selector."""
        p = self._payload_base
        if p is None:
            return
        modo = self._modo_direccion()
        # "Todos" y "Comparar lados" muestran el resultado del backtest tal
        # cual: los números de la vista principal no pasan por ningún filtro
        if modo in (_MODO_TODOS, _MODO_COMPARAR):
            self._payload = p
        else:
            self._payload = self._payload_de_direccion(
                1 if modo == _MODO_LARGOS else -1)

        lado = {_MODO_LARGOS: 'largos', _MODO_CORTOS: 'cortos'}.get(modo)
        nota = (f"Con el filtro de {lado} activo esta métrica se mide sobre la "
                f"curva de capital reconstruida (solo cierres realizados de los "
                f"{lado}), no sobre la equity marcada a mercado del backtest "
                f"completo.") if lado else None

        if modo == _MODO_COMPARAR:
            claves = ('Largos', 'Cortos', 'Total')
            met_tabla = {'Largos': self._payload_de_direccion(1)['metricas']['Total'],
                         'Cortos': self._payload_de_direccion(-1)['metricas']['Total'],
                         'Total': p['metricas']['Total']}
            nota = ("En las columnas Largos y Cortos esta métrica se mide sobre "
                    "la curva de capital reconstruida de ese lado; la columna "
                    "Total la mide sobre la equity real del backtest.")
        else:
            claves = ('IS', 'OOS', 'Total')
            met_tabla = self._payload['metricas']

        # las tablas de métricas y de setups son inmediatas a cualquier escala
        # (<1 ms): son las que responden al clic del selector
        render_tabla_metricas(self.tabla_metricas, met_tabla, claves,
                              p.get('tf'), nota)
        render_tabla_setups(self.tabla_setups, self.grp_setups, self._payload)
        # altura a contenido completo: sin barra de scroll, todo visible
        _fijar_altura_tabla(self.tabla_metricas)
        _fijar_altura_tabla(self.tabla_setups)
        self._lbl_equity.setText(
            f"Curva de Equity — solo {lado} (cierres realizados)" if lado
            else "Curva de Equity (IS vs OOS)")

        # el resto se aplaza un ciclo del bucle de eventos. La tabla de trades
        # va aquí y no arriba porque con miles de operaciones son decenas de
        # miles de celdas (~320 ms con 6.000 trades), tanto como los gráficos.
        self._graficos_sucios = True
        self._timer_graficos.start()

    @_no_crash
    def _pintar_graficos(self):
        if self._payload is None:
            return
        try:
            self._graficos_sucios = False
            payload = self._payload
            ts = pd.DatetimeIndex(payload['timestamps'])
            self._llenar_tabla_trades(self.tabla_trades, payload)
            if self._dlg_trades is not None:
                self._llenar_tabla_trades(self._dlg_trades.tabla, payload)
                self._actualizar_contador_trades()
            bombear_eventos()
            self._dibujar_principal()
            bombear_eventos()
            self._dibujar_equity(payload)
            bombear_eventos()
            if getattr(self, 'btn_vista', None) is not None and self.btn_vista.isChecked():
                self._mostrar_lwc(payload)
                bombear_eventos()
            self._wfa_cache = payload.get('wfa')
            self._wfa_ts = ts
            self._wfa_equity = payload['resultado']['equity']
            self._dibujar_wfa(payload.get('wfa'), ts, self._wfa_equity)
            bombear_eventos()
            self._dibujar_mc(payload.get('montecarlo'),
                             payload['config'].get('capital_inicial', 10000.0),
                             payload['metricas']['Total'].get('max_dd_pct'),
                             payload['metricas']['Total'].get('retorno_pct'))
            bombear_eventos()
            self._dibujar_mfe_mae(payload)
        finally:
            self.graficos_listos.emit()

    def _mostrar_avisos(self, avisos):
        """Banda ámbar sobre el gráfico con los filtros que se pidieron y no se
        pudieron aplicar. Sin esto, un backtest que ha corrido SIN el filtro de
        régimen se lee exactamente igual que uno filtrado."""
        self.lbl_avisos.setVisible(bool(avisos))
        if avisos:
            self.lbl_avisos.setText('⚠  ' + '\n⚠  '.join(avisos))

    # ── render principal ──
    @_no_crash
    def mostrar(self, payload):
        self._payload_base = payload
        self._cache_dir = {}
        self._y_manual = False
        ts = pd.DatetimeIndex(payload['timestamps'])
        estrategia = html.escape(payload['estrategia'])
        badge = _titulo_activo_html(payload['csv'], payload.get('tf'))
        self.lbl_titulo.setTextFormat(Qt.TextFormat.RichText)
        max_estr = 80
        estr_visible = estrategia if len(estrategia) <= max_estr else estrategia[:max_estr - 1] + '…'
        # el periodo cubierto va antes de las cifras: da la escala con la que
        # hay que leerlas (214 trades en 5 años no son 214 trades en 5 meses)
        periodo = _texto_periodo(ts)
        self.lbl_titulo.setText(
            f"{badge} — {estr_visible} — "
            + (f"{periodo} · " if periodo else "")
            + f"{payload['n_velas']:,} velas · {payload['resultado']['n_trades']} trades · "
            f"capital final {payload['resultado']['capital_final']:,.0f}")
        # el tooltip repite las fechas CON hora: en intradía el día suelto del
        # título no dice dónde empieza ni acaba realmente la serie
        detalle_periodo = (
            f"\n{ts[0].strftime('%d/%m/%Y %H:%M')} → "
            f"{ts[-1].strftime('%d/%m/%Y %H:%M')}" if len(ts) else "")
        self.lbl_titulo.setToolTip(
            f"{_nombre_activo_limpio(payload['csv'])} · {payload.get('tf', '')} — "
            f"{estrategia}{detalle_periodo}")
        self.btn_favorito.setEnabled(True)
        self.btn_favorito.setText("⭐ Guardar como favorito")
        self.btn_exportar.setEnabled(True)
        self._mostrar_avisos(payload.get('avisos') or [])

        # selector de dirección: el modo elegido se conserva entre ejecuciónes
        # (quien está analizando los cortos de un sistema no quiere que se
        # reinicie en cada ▶), salvo que este backtest no opere ese lado
        n_l, n_c = self._actualizar_botones_direccion()
        self.lbl_direccion.setText(
            f"{n_l:,} largos · {n_c:,} cortos · {n_l + n_c:,} trades")

        # fechas de los QDateEdit
        self.fecha_ini.setDate(QDate(ts[0].year, ts[0].month, ts[0].day))
        self.fecha_fin.setDate(QDate(ts[-1].year, ts[-1].month, ts[-1].day))

        # código del sistema ejecutado
        codigo = payload.get('codigo') or ''
        self.grp_codigo.setVisible(bool(codigo))
        self.txt_codigo.setPlainText(codigo)

        self._aplicar_direccion()

    @_no_crash
    def _guardar_favorito(self):
        """Guarda el activo + temporalidad + setup(s) de ESTE resultado (a
        diferencia de "Guardar sistema" en el Constructor, que solo guarda
        la estrategia sin recordar con qué activo/temporalidad se corrió)."""
        if self._payload is None:
            return
        nombre, ok = pedir_texto(self, "Guardar como favorito",
                                       "Nombre del favorito:")
        if not ok or not nombre.strip():
            return
        nombre = nombre.strip()
        carpeta = os.path.join(FAVORITOS_DIR, _slug_sistema(nombre))
        os.makedirs(carpeta, exist_ok=True)
        datos = {'nombre': nombre, 'csv': self._payload['csv'],
                 'tf': self._payload.get('tf'), 'setups': self._payload['setups'],
                 'config': _config_serializable(self._payload['config'])}
        _guardar_json_atomico(os.path.join(carpeta, 'favorito.json'), datos)
        self.btn_favorito.setText(f"✓ Guardado como «{nombre}»")
        self.favorito_guardado.emit()

    @_no_crash
    def _exportar_codigo(self):
        """Traduce ESTE sistema a código de las plataformas que se elijan.

        Se lee de _payload_base, no de _payload: el segundo puede venir
        filtrado por dirección (ver _payload_de_direccion) y exportaría en
        silencio un sistema recortado, sin los cortos o sin los largos. De la
        config del motor solo interesan los escalares de cuenta; el resto son
        máscaras precomputadas."""
        payload = self._payload_base
        if payload is None:
            return
        from gui.dialogs.export_codigo_dialog import DialogoExportarCodigo
        meta = {'sistema': _nombre_activo_limpio(payload['csv']) or 'sistema',
                'activo': _nombre_activo_limpio(payload['csv']),
                'tf': payload.get('tf') or ''}
        config = payload.get('config') or {}
        dlg = DialogoExportarCodigo(
            payload['setups'],
            {clave: config.get(clave) for clave in
             ('capital_inicial', 'comision_pct', 'slippage_pct')
             if config.get(clave) is not None},
            meta, self)
        if dlg.exec() != QDialog.DialogCode.Accepted or not dlg.resultado:
            return
        carpeta = dlg.resultado['carpeta']
        self.btn_exportar.setText("✓ Código exportado")
        informacion(
            self, "Código exportado",
            f"{len(dlg.resultado['archivos'])} archivos escritos en:\n"
            f"{carpeta}\n\n"
            f"Lee NOTAS_DE_FIDELIDAD.md antes de operar con ellos.")

    @_no_crash
    def _toggle_modo_grafico(self):
        self._modo_grafico = 'velas' if self.btn_modo_grafico.isChecked() else 'linea'
        self.btn_modo_grafico.setText("🕯 Velas" if self._modo_grafico == 'velas' else "📈 Línea")
        self._redibujar_principal_conservando_zoom()

    def _toggle_stop_loss(self, _=False):
        self._redibujar_principal_conservando_zoom()

    def _toggle_vista(self, _=False):
        """Alterna entre la gráfica matplotlib (Clásica) y Lightweight Charts
        (Moderna). Al pasar a Moderna, repinta la vista LWC con el payload
        actual; matplotlib sigue siendo el modo por defecto."""
        moderna = self.btn_vista.isChecked()
        self.btn_vista.setText("📈 Vista: Moderna" if moderna else "🖥 Vista: Clásica")
        self.stack_grafico.setCurrentIndex(1 if moderna else 0)
        if moderna and getattr(self, '_payload', None) is not None:
            self._mostrar_lwc(self._payload)

    def _mostrar_lwc(self, payload):
        """Repinta la vista LWC reflejando los mismos checkboxes (trayecto/
        stop-loss/noticias), las mismas capas (ZigZag/Fibonacci/órdenes) y los
        mismos indicadores (medias/Bollinger/KAMA/patrones/osciladores) que la
        vista clásica, para que ambas se vean coherentes al conmutar entre
        ellas."""
        self.lwc.mostrar(
            payload,
            mostrar_trayecto=getattr(self, 'chk_trayecto', None) is not None
                             and self.chk_trayecto.isChecked(),
            mostrar_stop=getattr(self, 'chk_stop', None) is not None
                        and self.chk_stop.isChecked(),
            mostrar_noticias=getattr(self, 'chk_noticias', None) is not None
                             and self.chk_noticias.isChecked(),
            eventos_noticias=payload.get('eventos_noticias'),
            # la vista Moderna dibuja el ZigZag como una sola capa, mientras que
            # el panel tiene un ojo por juego de parámetros: basta con que uno
            # esté encendido para que allí se pinte
            capas={'zigzag': any(
                       self._capa_activa(c['clave'])
                       for c in self._capas_catalogo
                       if c['clave'].startswith('zigzag:')),
                   'fib': self._capa_activa('fib'),
                   'ordenes': self._capa_activa('ordenes')},
            indicadores=self._recolectar_indicadores(payload))

    def _catalogo_capas(self, ind, payload):
        """Lista ordenada de capas dibujables de este backtest, como dicts
        {clave, etiqueta, color, grupo, ayuda}.

        Es la ÚNICA fuente de la que salen tanto las filas del panel de capas
        como los colores con que se pintan los indicadores. Si el panel derivara
        su lista por su cuenta acabaría ofreciendo un ojo para algo que no se
        pinta (o al revés), y bastaría con tocar un color en un sitio para que
        el cuadradito del panel dejara de corresponderse con su línea.

        La clave identifica la capa entre redibujados: incluye los parámetros,
        así que dos medias distintas son dos capas distintas y apagar la EMA(20)
        no apaga la EMA(50). `ind` es lo que devuelve _recolectar_indicadores.
        """
        capas = []
        idx_paleta = 0
        for tipo, per in sorted(ind['mas'], key=lambda x: x[1]):
            color = COLOR_MEDIA_FIJO.get(per)
            if color is None:
                color = PALETA_MEDIA[idx_paleta % len(PALETA_MEDIA)]
                idx_paleta += 1
            capas.append({'clave': f'ma:{tipo}:{per}', 'etiqueta': f'{tipo}({per})',
                          'color': color, 'grupo': 'ind',
                          'ayuda': f'Media {"exponencial" if tipo == "EMA" else "simple"} '
                                   f'de {per} periodos.'})
        for per, desv in sorted(ind['bbs']):
            capas.append({'clave': f'bb:{per}:{desv:g}',
                          'etiqueta': f'BB({per}, {desv:g})',
                          'color': COLOR_BOLLINGER, 'grupo': 'ind',
                          'ayuda': 'Bandas de Bollinger: media y desviación típica.'})
        for per_er, rapido, lento in sorted(ind['kamas']):
            capas.append({'clave': f'kama:{per_er}:{rapido}:{lento}',
                          'etiqueta': f'KAMA({per_er},{rapido},{lento})',
                          'color': COLOR_KAMA, 'grupo': 'ind',
                          'ayuda': 'Media adaptativa de Kaufman.'})
        for per in sorted(ind['donchianes']):
            capas.append({'clave': f'donchian:{per}', 'etiqueta': f'Donchian({per})',
                          'color': COLOR_DONCHIAN, 'grupo': 'ind',
                          'ayuda': f'Canal de máximos y mínimos de {per} velas.'})
        for desviacion, piernas in sorted(ind['zigzags']):
            capas.append({'clave': f'zigzag:{desviacion:g}:{piernas}',
                          'etiqueta': f'ZigZag({desviacion:g}%, {piernas})',
                          'color': AZUL_ZIGZAG, 'grupo': 'ind',
                          'ayuda': 'Polilínea que une los pivotes de swing '
                                   'confirmados, con los parámetros del setup.'})
        for af_i, af_p, af_m in sorted(ind['sars']):
            capas.append({'clave': f'sar:{af_i:g}:{af_p:g}:{af_m:g}',
                          'etiqueta': f'SAR({af_i:g},{af_p:g},{af_m:g})',
                          'color': COLOR_SAR, 'grupo': 'ind',
                          'ayuda': 'Parabolic SAR: puntos de stop and reverse.'})
        for per, mult in sorted(ind['supertrends']):
            capas.append({'clave': f'st:{per}:{mult:g}',
                          'etiqueta': f'Supertrend({per},×{mult:g})',
                          'color': COLOR_SUPERTREND, 'grupo': 'ind',
                          'ayuda': f'Envolvente ATR de {per} periodos con '
                                   f'multiplicador {mult:g}: cambia de lado en '
                                   'los giros de tendencia.'})
        for tenkan, kijun, senkou in sorted(ind['ichimokus']):
            capas.append({'clave': f'ichi:{tenkan}:{kijun}:{senkou}',
                          'etiqueta': f'Ichimoku({tenkan},{kijun},{senkou})',
                          'color': COLOR_ICHIMOKU_TENKAN, 'grupo': 'ind',
                          'ayuda': 'Tenkan, Kijun y Senkou A/B alineadas (sin '
                                   'el desplazamiento +26 del gráfico, que '
                                   'sería lookahead).'})
        for per, mult in sorted(ind['keltners']):
            capas.append({'clave': f'kc:{per}:{mult:g}',
                          'etiqueta': f'Keltner({per},×{mult:g})',
                          'color': COLOR_KELTNER, 'grupo': 'ind',
                          'ayuda': f'Bandas de Keltner: EMA ± {mult:g}×ATR de '
                                   f'{per} periodos.'})
        for idx, (anclaje, k, modo) in enumerate(sorted(ind['vwaps'])):
            color_v = _PALETA_VWAP[idx % len(_PALETA_VWAP)]
            unidad = 'σ' if modo == 'sd' else '%'
            capas.append({'clave': f'vwap:{anclaje}:{k:g}:{modo}',
                          'etiqueta': f'VWAP({_MAPA_ANCLAJE_INV.get(anclaje, anclaje)}, {k:g}{unidad})',
                          'color': color_v, 'grupo': 'ind',
                          'ayuda': f'VWAP del anclaje «{anclaje}» con banda '
                                   f'{k:g}{unidad} (gris, relleno del canal). '
                                   'Se reinicia en cada borde del anclaje.'})
        if ind['patrones']:
            capas.append({'clave': 'patrones', 'etiqueta': 'Patrones de velas',
                          'color': COLOR_PATRONES, 'grupo': 'ind',
                          'ayuda': 'Marca cada patrón de velas detectado.'})

        # capas de la entrada por orden límite: solo aparecen si este backtest
        # llegó a colocar órdenes, o serían filas que no pintan nada
        ol_cat = payload.get('resultado', {}).get('ordenes_limite') or {}
        n_ordenes = len(ol_cat.get('idx_alta', ()))
        # los tramos solo se pintan sobre órdenes rellenadas, así que sin
        # ninguna el ojo sería una fila que no apaga ni enciende nada
        n_rellenadas = int(np.sum(np.asarray(ol_cat.get('resultado', ()),
                                             dtype=np.int64) == ORDEN_RELLENADA)) \
            if n_ordenes else 0
        if ind['fibs'] and n_rellenadas:
            capas.append({'clave': 'fib', 'etiqueta': 'Tramos Fibonacci',
                          'color': AMARILLO_FIB, 'grupo': 'ind',
                          'ayuda': 'El swing del que se midió el retroceso de cada '
                                   'orden ejecutada: sus dos extremos y la zona '
                                   'entre ellos. Las órdenes que se cancelaron sin '
                                   'llegar al nivel no dibujan tramo.'})
        if n_ordenes:
            capas.append({'clave': 'ordenes', 'etiqueta': 'Órdenes límite',
                          'color': VERDE, 'grupo': 'ind',
                          'ayuda': 'Cada orden como un segmento a su precio, desde '
                                   'que se coloca hasta que se resuelve: verde '
                                   'rellenada, gris cancelada, ámbar expirada.'})

        # osciladores: cada uno vive en su propio panel bajo el precio, así que
        # apagarlo no quita una línea sino el panel entero
        for kind, clave_ind in PANELES_OSCILADOR:
            if ind.get(clave_ind):
                ayuda = ('Se dibuja en su propio panel bajo el precio; al '
                         'apagarlo el panel desaparece.')
                if kind in ('er', 'hurst'):
                    ayuda = ('La serie que usa el filtro de régimen de este '
                             'setup, con sus umbrales y la banda que admite '
                             'sombreada: las entradas deben caer dentro.')
                capas.append({'clave': f'panel:{kind}',
                              'etiqueta': ETIQUETA_PANEL[kind],
                              'color': COLOR_PANEL_OSC[kind], 'grupo': 'panel',
                              'ayuda': ayuda})
        return capas

    def _capa_activa(self, clave):
        """True si el ojo de esa capa está encendido. Una clave que aún no está
        en el dict es una capa que el usuario no ha tocado nunca: casi todas
        nacen visibles, salvo las de PREFIJOS_CAPA_OCULTA."""
        return self._capas_estado.get(clave, capa_visible_por_defecto(clave))

    def _color_capa(self, clave, defecto=GRIS):
        """Color con que se pinta esa capa, según el catálogo del último
        dibujo. Lo consultan los bucles de dibujo para no repetir la asignación
        de la paleta (que depende del orden) por su cuenta."""
        return self._colores_capas.get(clave, defecto)

    def _conmutar_capa(self, clave, visible):
        """Un ojo del panel. Las capas apagadas no se ocultan: se dejan de
        crear, y por eso hay que redibujar. Ocultarlas con set_visible no
        valdría — _iniciar_sesion_blit devuelve a visibles todos los artistas
        dinámicos en cuanto el usuario arrastra el gráfico."""
        self._capas_estado[clave] = bool(visible)
        self._redibujar_principal_conservando_zoom()

    def _actualizar_panel_capas(self, ind, payload):
        """Recalcula el catálogo y repuebla el panel. Se llama en cada dibujo
        completo: al cambiar de sistema cambian los indicadores."""
        self._capas_catalogo = self._catalogo_capas(ind, payload)
        self._colores_capas = {c['clave']: c['color'] for c in self._capas_catalogo}
        # el catálogo y los colores se dejan puestos aunque el panel aún no
        # exista: _dibujar_principal puede correr durante la construcción del
        # widget, antes de que el canvas tenga su leyenda montada
        panel = getattr(self, 'panel_capas', None)
        if panel is not None:
            panel.poblar(self._capas_catalogo, self._capas_estado)
            panel.reposicionar()

    def _redibujar_principal_conservando_zoom(self):
        """Redibuja el gráfico principal preservando el encuadre actual —
        compartido por los ojos del panel de capas y por los toggles de modo
        velas/línea, stop-loss y trayecto. Si la vista moderna (LWC) está
        activa, también la refresca.

        La escala de precio solo se conserva si el usuario la había ajustado a
        mano: si está en automático se recalcula sola sobre la ventana visible
        (y debe hacerlo, porque el modo velas y el de línea no encuadran igual).
        """
        xlim = ylim = None
        if getattr(self, '_ax_principal', None) is not None:
            # el eje ya está en índice de vela: se reenvía tal cual, sin pasar
            # por fechas (ver _dibujar_principal)
            xlim = self._ax_principal.get_xlim()
            if self._y_manual:
                ylim = self._ax_principal.get_ylim()
        self._dibujar_principal(xlim=xlim, ylim=ylim)
        if getattr(self, 'btn_vista', None) is not None and self.btn_vista.isChecked() \
                and getattr(self, '_payload', None) is not None:
            self._mostrar_lwc(self._payload)

    def _toggle_trayecto(self, _=False):
        self._redibujar_principal_conservando_zoom()

    def _toggle_noticias(self, checked=False):
        # try/finally: garantiza el redibujado aunque la guía falle (ver
        # _on_noticias_clicked para el detalle de por qué @​_no_crash puede
        # tragarse el resto del handler).
        try:
            if checked and not get_te_api_key():
                # Sin API key: no se marca y se abre la guía. Si al cerrar ya hay
                # key, se marca solo y se redibuja.
                self.chk_noticias.setChecked(False)
                if _abrir_guia_calendario(self):
                    self.chk_noticias.setChecked(True)
                return
        finally:
            self._redibujar_principal_conservando_zoom()

    def _trades_visibles(self, x0, x1):
        """Máscara booleana (una entrada por trade) de los trades cuyo
        rango [entrada, salida] solapa la ventana visible [x0, x1] —
        recorta flechas/trayecto/stop-loss al frame actual igual que
        `_decimar_ohlc` recorta las velas, pero vectorizado (sin bucle)."""
        if not self._tr or len(self._tr.get('pnl', [])) == 0:
            return np.zeros(0, dtype=bool)
        xe = self._x_full[self._tr['idx_entrada']]
        xs = self._x_full[self._tr['idx_salida']]
        return (xe <= x1) & (xs >= x0)

    def _tramo_extra_visible(self, idx_full, x0, x1):
        """Máscara booleana de un array de índices de tramos 2+ (evento
        puntual, no un rango como los trades) dentro de la ventana visible
        [x0, x1]."""
        if idx_full is None or len(idx_full) == 0:
            return np.zeros(0, dtype=bool)
        xt = self._x_full[idx_full]
        return (xt >= x0) & (xt <= x1)

    def _max_velas_visibles(self):
        """Tope de velas a dibujar, atado al ancho REAL del lienzo.

        Dibujar más velas que píxeles no enseña nada: con menos de ~3 px por
        vela el cuerpo no llega ni a un píxel y el gráfico se ve como una
        mancha, pero cuesta el rasterizado completo igual. Medido en un canvas
        de 900 px: 1200 velas = 22 ms/frame, 400 velas = 8 ms/frame con el
        mismo resultado en pantalla. El suelo de 120 evita quedarse sin
        detalle en un canvas diminuto."""
        ancho_px = self.canvas.get_width_height()[0] or 900
        return max(120, min(2500, int(ancho_px / 3)))

    def _descartar_artistas_datos(self):
        for art in self._art_datos:
            try:
                art.remove()
            except Exception:
                pass
        self._art_datos = []

    def _crear_artistas_datos(self, ax, segs, verts, colores, xb, cb):
        """Crea los artistas de la serie. Solo la primera vez tras un dibujo
        completo: a partir de ahí cada pan/zoom los MUTA (ver
        _redibujar_datos), que es entre 5 y 6 veces más barato."""
        self._descartar_artistas_datos()
        if self._modo_grafico == 'velas':
            # mechas y cuerpos como DOS colecciones y no un artista por vela:
            # ax.bar crea un Rectangle individual y actualiza los límites de
            # datos en cada add_patch, el cuello de botella real con miles de
            # barras (~1,2 s en add_patch/_update_patch_limits para 2500).
            #
            # autolim=False: los dos ejes se fijan a mano aquí abajo y en
            # _dibujar_principal, así que recalcular dataLim en cada frame de
            # arrastre solo sería trabajo tirado.
            mechas = LineCollection(segs, colors=colores, linewidths=0.6)
            cuerpos = PolyCollection(verts, facecolors=colores,
                                     edgecolors=colores, linewidths=0)
            ax.add_collection(mechas, autolim=False)
            ax.add_collection(cuerpos, autolim=False)
            self._art_datos = [mechas, cuerpos]
        else:
            linea, = ax.plot(xb, cb, color=AX_FG, linewidth=0.8)
            self._art_datos = [linea]

    def _redibujar_datos(self, ax):
        """Redibuja solo las velas/línea, decimadas al rango visible de
        `ax` — se llama tanto en el dibujo inicial como en cada zoom/pan
        (callback 'xlim_changed'), sin tocar sombreado/flechas/leyenda/slider.

        Los artistas se reutilizan mutándolos in-place mientras siga en pie el
        mismo dibujo: crear y destruir un PolyCollection por frame costaba unos
        19 ms de los ~50 que tardaba un frame de arrastre."""
        x0, x1 = ax.get_xlim()
        xb, ob, hb, lb, cb = _decimar_ohlc(
            self._x_full, self._o_full, self._h_full, self._l_full, self._c_full,
            x0, x1, max_velas=self._max_velas_visibles())
        if len(xb) == 0:
            self._descartar_artistas_datos()
            return
        # rango Y del slice COMPLETO visible, no solo de la muestra dibujada:
        # con el submuestreo, un spike que cae entre velas muestreadas queda
        # fuera de la muestra y haría saltar la escala al acercarse a él
        j0, j1 = np.searchsorted(self._x_full, [x0, x1])
        j0 = max(j0 - 1, 0)
        j1 = min(j1 + 1, len(self._x_full))
        if j1 <= j0:
            j1 = min(j0 + 1, len(self._x_full))

        if self._modo_grafico == 'velas':
            colores = np.where(cb >= ob, VERDE, ROJO)
            # ancho proporcional al espaciado entre velas (como LWC/TradingView):
            # el cuerpo crece y se ensancha con el zoom, siempre ~70% de la
            # separación, así las velas se ven estables y sin huecos crecientes
            espaciado = float(np.median(np.diff(xb))) if len(xb) > 1 else 1.0
            ancho = espaciado * 0.7
            half = ancho / 2.0
            yb = np.minimum(ob, cb)
            yt = np.maximum(ob, cb)
            segs = np.stack([np.column_stack([xb, lb]),
                             np.column_stack([xb, hb])], axis=1)
            verts = np.stack([
                np.column_stack([xb - half, yb]),
                np.column_stack([xb + half, yb]),
                np.column_stack([xb + half, yt]),
                np.column_stack([xb - half, yt]),
            ], axis=1)
            if len(self._art_datos) == 2:
                mechas, cuerpos = self._art_datos
                mechas.set_segments(segs)
                mechas.set_color(colores)
                cuerpos.set_verts(verts)
                cuerpos.set_facecolor(colores)
                cuerpos.set_edgecolor(colores)
            else:
                self._crear_artistas_datos(ax, segs, verts, colores, xb, cb)
            y_lo = float(np.min(self._l_full[j0:j1]))
            y_hi = float(np.max(self._h_full[j0:j1]))
        else:
            if len(self._art_datos) == 1:
                self._art_datos[0].set_data(xb, cb)
            else:
                self._crear_artistas_datos(ax, None, None, None, xb, cb)
            y_lo = float(np.min(self._c_full[j0:j1]))
            y_hi = float(np.max(self._c_full[j0:j1]))

        if not self._y_manual:
            pad = (y_hi - y_lo) * 0.05 or 1.0
            ax.set_ylim(y_lo - pad, y_hi + pad)
        self._actualizar_ultimo_precio(ax)
        # overlays/osciladores densos: cortarlos a la ventana visible como a
        # las velas. Antes se redibujaban a resolución completa en CADA frame
        # de blit (una media con 100k puntos costaba ~16 ms por frame), que
        # con la animación de la rueda era el lag/parpadeo del scroll.
        for art in self._art_decimables:
            xs, ys = _decimar_linea(art._x_all, art._y_all, x0, x1,
                                    max_puntos=self._max_velas_visibles())
            art.set_data(xs, ys)

    def _etiquetar_decimables(self):
        """Tras crear todos los overlays/osciladores (final de
        _dibujar_principal), etiqueta los Line2D densos (medias, Bollinger,
        KAMA, Supertrend, Ichimoku, Keltner, Donchian y las líneas de los
        paneles de oscilador) con su serie completa y los apunta en
        _art_decimables para que _redibujar_datos los recorte en cada frame.

        Se excluyen los pasos (steps-post: squeeze TTM y tracks de
        stop/entrada, cuyo submuestreo deformaría el patrón) y los artistas
        ralos (zigzag, tramos de Fibonacci, órdenes, axhline, leyendas), que
        no son el cuello de botella del blit."""
        from matplotlib.lines import Line2D
        self._art_decimables = []
        for art in self._art_overlays_extra + self._art_osciladores:
            if not isinstance(art, Line2D):
                continue
            if art.get_drawstyle().startswith('steps'):
                continue
            xd = art.get_xdata()
            if len(xd) < 3000:
                continue
            art._x_all = np.asarray(xd, dtype=float)
            art._y_all = np.asarray(art.get_ydata(), dtype=float)
            self._art_decimables.append(art)

    def _actualizar_ultimo_precio(self, ax):
        """Reubica el badge del último precio dentro del rango visible del eje
        (pegado al borde si el precio queda fuera, como TradingView/LWC)."""
        lbl = getattr(self, '_lbl_ultimo_precio', None)
        if lbl is None or self._c_full is None or not len(self._c_full):
            return
        ultimo = float(self._c_full[-1])
        ylo, yhi = ax.get_ylim()
        lbl.xy = (1, float(np.clip(ultimo, min(ylo, yhi), max(ylo, yhi))))
        lbl.set_text(_fmt_precio(ultimo))

    def _on_xlim_changed(self, ax):
        if self._actualizando_xlim:
            return
        if self._sesion_activa:
            # durante un arrastre/ráfaga de scroll: solo marcar los datos como
            # sucios y pedir frame — el redibujado real ocurre UNA vez por
            # ciclo de eventos en _pintar_frame_pendiente, no por evento de
            # ratón (un trackpad manda decenas de eventos por gesto)
            self._datos_sucios = True
            self._solicitar_frame_blit(ax)
            return
        self._actualizando_xlim = True
        try:
            self._redibujar_datos(ax)
            if self._blit_bg is None:
                self.canvas.draw_idle()
        finally:
            self._actualizando_xlim = False

    # ── layout de paneles apilados (precio + osciladores), estilo TradingView:
    # redimensionable arrastrando el borde entre paneles, colapsable con
    # doble clic en ese borde. self._paneles es [(kind, ax), ...] con el
    # precio siempre primero; self._pesos_paneles guarda la proporción de
    # altura de cada `kind` (persiste entre redibujados de la misma sesión).
    def _peso_panel(self, kind):
        defecto = PESO_PRECIO_DEFECTO if kind == 'precio' else PESO_OSC_DEFECTO
        return self._pesos_paneles.get(kind, defecto)

    def _aplicar_pesos_paneles(self):
        """Fija la posición [left, bottom, width, height] de cada Axes
        apilado según self._pesos_paneles — no depende del GridSpec (que
        solo se usa al crear los Axes, para el wiring de sharex), así que
        se puede llamar en cada frame de un arrastre sin reconstruir nada."""
        paneles = self._paneles
        if not paneles:
            return
        n = len(paneles)
        pesos = [max(self._peso_panel(kind), PESO_PANEL_MIN) for kind, _ax in paneles]
        alto_disponible = (TOP_PANEL - BOTTOM_STACK) - GAP_PANEL * (n - 1)
        suma = sum(pesos)
        y_top = TOP_PANEL
        for (kind, ax), peso in zip(paneles, pesos):
            h = alto_disponible * (peso / suma)
            y_bottom = y_top - h
            ax.set_position([LEFT_PANEL, y_bottom, RIGHT_PANEL - LEFT_PANEL, h])
            y_top = y_bottom - GAP_PANEL
        for kind, ax in paneles[:-1]:
            ax.tick_params(axis='x', labelbottom=False)

    def _iniciar_arrastre_panel(self, idx, event):
        self._drag_modo = f'resize_panel:{idx}'
        self._drag_inicio = (event.x, event.y)
        kind_arriba = self._paneles[idx][0]
        kind_abajo = self._paneles[idx + 1][0]
        self._drag_pesos0 = (self._peso_panel(kind_arriba), self._peso_panel(kind_abajo))
        self.canvas.setCursor(Qt.CursorShape.SizeVerCursor)

    def _arrastrar_borde_panel(self, idx, dy_px):
        """dy_px > 0 = el ratón subió en pantalla (convención y-arriba de
        matplotlib) desde que empezó el arrastre: el panel de encima se
        encoge y el de debajo crece, igual que arrastrar el handle de un
        splitter. Redistribuye SOLO entre el par (idx, idx+1); el resto de
        paneles no se mueve."""
        kind_arriba = self._paneles[idx][0]
        kind_abajo = self._paneles[idx + 1][0]
        peso_arriba0, peso_abajo0 = self._drag_pesos0
        suma_total = sum(max(self._peso_panel(k), PESO_PANEL_MIN)
                          for k, _ax in self._paneles)
        alto_disponible = (TOP_PANEL - BOTTOM_STACK) - GAP_PANEL * (len(self._paneles) - 1)
        dy_frac = dy_px / max(self.fig.bbox.height, 1)
        d_peso = -dy_frac * suma_total / alto_disponible
        suma_par = peso_arriba0 + peso_abajo0
        nuevo_arriba = min(max(peso_arriba0 + d_peso, PESO_PANEL_MIN),
                            suma_par - PESO_PANEL_MIN)
        nuevo_abajo = suma_par - nuevo_arriba
        self._pesos_paneles[kind_arriba] = nuevo_arriba
        self._pesos_paneles[kind_abajo] = nuevo_abajo
        self._aplicar_pesos_paneles()
        # un cambio de posición de Axes invalida cualquier bitmap de blit
        # cacheado con las posiciones antiguas; requiere un draw() completo,
        # no hay atajo de blitting posible aquí (a diferencia de pan/zoom)
        self._blit_bg = None
        self.canvas.draw_idle()

    def _alternar_colapso_panel(self, idx):
        """Doble clic en el borde entre dos paneles: colapsa el panel de
        DEBAJO del borde (el oscilador, nunca el precio — el precio es
        siempre paneles[0] y ningún borde tiene nada por encima de él) a
        una franja mínima, o lo restaura al tamaño que tenía antes de
        colapsarlo si ya estaba colapsado."""
        kind = self._paneles[idx + 1][0]
        if kind in self._pesos_paneles_prev:
            self._pesos_paneles[kind] = self._pesos_paneles_prev.pop(kind)
        else:
            self._pesos_paneles_prev[kind] = self._peso_panel(kind)
            self._pesos_paneles[kind] = PESO_PANEL_MIN
        self._aplicar_pesos_paneles()
        self.canvas.draw_idle()

    def _zona_eje(self, event):
        """'pan' si el evento cae dentro del propio gráfico (arrastrar
        desplaza la vista), 'y' si cae en la franja de etiquetas del eje Y
        (a la derecha del gráfico), 'x' si cae en la franja de etiquetas del
        eje X (debajo del último panel), 'resize_panel:<i>' si cae en el
        borde entre el panel i y el i+1 (precio y osciladores apilados), o
        None."""
        ax = getattr(self, '_ax_principal', None)
        if ax is None or event.x is None or event.y is None:
            return None
        if self.toolbar.mode:
            return None  # no interferir con pan/zoom nativo del toolbar

        paneles = self._paneles or [('precio', ax)]
        if len(paneles) > 1:
            bbox0 = paneles[0][1].get_window_extent()
            if bbox0.x0 <= event.x <= bbox0.x1:
                margen = 5
                for i in range(len(paneles) - 1):
                    b_arriba = paneles[i][1].get_window_extent()
                    b_abajo = paneles[i + 1][1].get_window_extent()
                    if b_abajo.y1 - margen <= event.y <= b_arriba.y0 + margen:
                        return f'resize_panel:{i}'

        bbox = ax.get_window_extent()
        if bbox.x0 <= event.x <= bbox.x1 and bbox.y0 <= event.y <= bbox.y1:
            return 'pan'
        if bbox.y0 <= event.y <= bbox.y1 and event.x > bbox.x1:
            return 'y'
        # interior y franja de etiquetas del eje Y de un panel de oscilador:
        # mismo criterio que el 'pan'/'y' del precio, pero contra el bbox de
        # ese panel. Van después del precio y de resize_panel, que tienen
        # prioridad. Las dos zonas son excluyentes: 'y:' exige estar A LA
        # DERECHA del panel y 'pan:' exige estar DENTRO.
        for i, (_kind, ax_panel) in enumerate(paneles[1:], start=1):
            b = ax_panel.get_window_extent()
            if b.y0 <= event.y <= b.y1 and event.x > b.x1:
                return f'y:{i}'
            if b.x0 <= event.x <= b.x1 and b.y0 <= event.y <= b.y1:
                return f'pan:{i}'
        ultimo = paneles[-1][1]
        bbox_ult = ultimo.get_window_extent()
        if bbox.x0 <= event.x <= bbox.x1 and 0 < event.y < bbox_ult.y0:
            return 'x'
        return None

    @_no_crash
    def _on_press_ejes(self, event):
        zona = self._zona_eje(event)
        if zona is None:
            return
        # empieza un gesto: la cruceta se retira (como en TradingView) para no
        # dejar su posición congelada mientras el gráfico se mueve debajo
        self._ocultar_cruceta(repintar=False)
        # si quedaba un zoom con rueda a medias, se interrumpe: el arrastre
        # toma el control directo del xlim
        self._timer_zoom.stop()
        self._zoom_target = None
        if zona.startswith('resize_panel:'):
            idx = int(zona.split(':')[1])
            if event.dblclick:
                self._alternar_colapso_panel(idx)
            else:
                self._iniciar_arrastre_panel(idx, event)
            return
        ax = self._ax_principal
        if zona.startswith('y:'):
            # eje Y de un panel de oscilador: mismo gesto que el del precio,
            # pero sobre el Axes de ese panel
            kind, ax_panel = self._paneles[int(zona.split(':')[1])]
            if event.dblclick:
                # doble clic: se olvida el estirado y vuelve la escala natural
                # de ese oscilador (0-100 en RSI, autoescala en ATR...)
                self._ylim_paneles.pop(kind, None)
                self._redibujar_principal_conservando_zoom()
                return
            # el modo se fija ANTES de capturar el fondo: _iniciar_sesion_blit
            # lo consulta para saber que el eje Y de este panel es dinámico y
            # no debe quedarse pegado en el bitmap
            self._drag_modo = zona
            self._drag_inicio = (event.x, event.y)
            self._drag_lim0 = ax_panel.get_ylim()
            self._sesion_activa = True
            self._iniciar_sesion_blit()
            return
        if zona.startswith('pan:'):
            # interior de un panel de oscilador: arrastrar mueve el tiempo
            # (eje X, compartido con el precio) y la escala de ESE panel
            kind, ax_panel = self._paneles[int(zona.split(':')[1])]
            if event.dblclick:
                # mismo gesto que el doble clic sobre su franja de números:
                # olvida el estirado y vuelve la escala natural del panel
                self._ylim_paneles.pop(kind, None)
                self._redibujar_principal_conservando_zoom()
                return
            # el modo, antes del blit: ver el comentario de la rama 'y:'
            self._drag_modo = zona
            self._drag_inicio = (event.x, event.y)
            self._drag_lim0 = (ax.get_xlim(), ax_panel.get_ylim())
            self._sesion_activa = True
            self._iniciar_sesion_blit()
            self.canvas.setCursor(Qt.CursorShape.ClosedHandCursor)
            return
        if zona == 'y' and event.dblclick:
            # doble clic sobre el eje Y: volver a autoescala de precio
            self._y_manual = False
            self._redibujar_datos(ax)
            self.canvas.draw_idle()
            return
        self._sesion_activa = True
        self._iniciar_sesion_blit()
        self._drag_modo = zona
        self._drag_inicio = (event.x, event.y)
        if zona == 'pan':
            self._drag_lim0 = (ax.get_xlim(), ax.get_ylim())
            self.canvas.setCursor(Qt.CursorShape.ClosedHandCursor)
        elif zona == 'y':
            self._drag_lim0 = ax.get_ylim()
        else:
            self._drag_lim0 = ax.get_xlim()

    def _lineas_ejecucion(self, fila, es_salida, precio):
        """Líneas del tooltip que permiten auditar la ejecución: a qué precio
        ejecutó el motor y cuánto se separó de la referencia.

        En una entrada a mercado la referencia es el open de esa vela, así que
        la diferencia ES el slippage aplicado. En una entrada por orden límite
        la referencia es el precio pedido, y la diferencia debería ser cero
        salvo que un hueco de apertura la llenara mejor."""
        lineas = [f"Ejecución: {precio:,.5g}"]
        tr = self._tr
        if not tr or es_salida:
            return lineas
        barra = int(tr['idx_entrada'][fila])
        pedido = (self._orden_por_barra or {}).get(barra)
        if pedido is not None:
            lineas.append(f"Pedido (límite): {pedido:,.5g}")
            referencia, etiqueta = pedido, "Diferencia"
        elif self._o_full is not None and 0 <= barra < len(self._o_full):
            referencia, etiqueta = float(self._o_full[barra]), "Slippage"
            lineas.append(f"Open de la vela: {referencia:,.5g}")
        else:
            return lineas
        dif = precio - referencia
        pct = (dif / referencia * 100.0) if referencia else 0.0
        lineas.append(f"{etiqueta}: {dif:+,.5g} ({pct:+.3f} %)")
        return lineas

    def _actualizar_tooltip_trade(self, event, ax, zona):
        """Muestra/oculta self._annot_trade con el lotaje (y el RR, si es la
        flecha de SALIDA) del marcador de compra/venta más cercano al
        cursor, dentro de un radio en píxeles de pantalla. Llamado desde
        _on_motion_ejes solo cuando no hay arrastre activo (pan/zoom con
        blit usa su propio camino y no necesita este tooltip)."""
        annot = getattr(self, '_annot_trade', None)
        if annot is None:
            return
        UMBRAL_PX2 = 10.0 ** 2
        mejor = None   # (dist2, x_dato, y_dato, texto)
        if zona == 'pan' and event.x is not None and event.y is not None \
                and len(self._compra_idx_full):
            mask = self._trades_visibles(*ax.get_xlim())
            filas = np.where(mask)[0]
            for lado in ('compra', 'venta'):
                # mismas alturas que el dibujo: el ratón apunta a la FLECHA
                x, y_flecha, y_nivel = self._alturas_marcadores(lado, mask)
                if not len(x):
                    continue
                pts = ax.transData.transform(np.column_stack([x, y_flecha]))
                d2 = (pts[:, 0] - event.x) ** 2 + (pts[:, 1] - event.y) ** 2
                j = int(np.argmin(d2))
                if mejor is None or d2[j] < mejor[0]:
                    r = filas[j]
                    es_salida = (self._es_long_full[r] if lado == 'venta'
                                 else not self._es_long_full[r])
                    lineas = [f"Lotaje: {self._lotaje_full[r]:.2f}"]
                    if es_salida:
                        lineas.append(f"RR: {self._rr_full[r]:+.2f}R")
                    lineas += self._lineas_ejecucion(r, es_salida, y_nivel[j])
                    mejor = (d2[j], x[j], y_flecha[j], '\n'.join(lineas))
        if mejor is not None and mejor[0] <= UMBRAL_PX2:
            _, x_dato, y_dato, texto = mejor
            renderer = self.canvas.get_renderer()
            bb = ax.get_window_extent(renderer=renderer)
            x_disp, y_disp = ax.transData.transform((x_dato, y_dato))
            dx, ha = ((15, 'left') if (bb.x1 - x_disp) >= (x_disp - bb.x0)
                      else (-15, 'right'))
            dy, va = ((15, 'bottom') if (bb.y1 - y_disp) >= (y_disp - bb.y0)
                      else (-15, 'top'))
            annot.xy = (x_dato, y_dato)
            annot.set_position((dx, dy))
            annot.set_ha(ha)
            annot.set_va(va)
            annot.set_text(texto)
            annot.set_visible(True)
        else:
            annot.set_visible(False)
        # no se pinta aquí: el repintado lo hace _actualizar_cruceta en una
        # sola pasada de blit con la cruceta. Antes esto llamaba a draw_idle()
        # en CADA movimiento del ratón, y un draw completo del gráfico cuesta
        # cientos de ms con series largas — con la cruceta encima, además, los
        # dos repintados competirían y se vería parpadear.

    @_no_crash
    def _on_motion_ejes(self, event):
        ax = getattr(self, '_ax_principal', None)
        if ax is None:
            return
        if self._drag_modo is None:
            zona = self._zona_eje(event)
            if zona is not None and (zona.startswith('resize_panel')
                                     or zona.startswith('y:')):
                cursor = Qt.CursorShape.SizeVerCursor
            elif zona is not None and zona.startswith('pan:'):
                # interior de un panel de oscilador: se arrastra igual que el
                # precio, así que la mano abierta lo anuncia igual
                cursor = Qt.CursorShape.OpenHandCursor
            else:
                cursores = {'y': Qt.CursorShape.SizeVerCursor,
                            'x': Qt.CursorShape.SizeHorCursor,
                            'pan': Qt.CursorShape.OpenHandCursor}
                cursor = cursores.get(zona, Qt.CursorShape.ArrowCursor)
            self.canvas.setCursor(cursor)
            # ambos son capa de hover y comparten pasada de blit: el tooltip
            # solo decide su visibilidad, y la cruceta pinta los dos
            self._actualizar_tooltip_trade(event, ax, zona)
            self._actualizar_cruceta(event, zona)
            return
        if event.x is None or event.y is None:
            return
        if self._drag_modo.startswith('resize_panel:'):
            idx = int(self._drag_modo.split(':')[1])
            self._arrastrar_borde_panel(idx, event.y - self._drag_inicio[1])
            return
        x0, y0 = self._drag_inicio
        if self._drag_modo == 'pan':
            inv = ax.transData.inverted()
            x0d, y0d = inv.transform((x0, y0))
            x1d, y1d = inv.transform((event.x, event.y))
            dxd, dyd = x1d - x0d, y1d - y0d
            xlim0, ylim0 = self._drag_lim0
            self._y_manual = True
            ax.set_ylim(ylim0[0] - dyd, ylim0[1] - dyd)
            ax.set_xlim(xlim0[0] - dxd, xlim0[1] - dxd)  # dispara xlim_changed
            self._sync_dateedits(*ax.get_xlim())
            self._solicitar_frame_blit(ax)
            return
        if self._drag_modo.startswith('pan:'):
            # el eje X es el del precio (compartido: mueve todo el gráfico) y
            # el Y es el del panel arrastrado. Son escalas distintas, así que
            # cada desplazamiento se convierte con SU propia transData.
            kind, ax_panel = self._paneles[int(self._drag_modo.split(':')[1])]
            xlim0, ylim0 = self._drag_lim0
            inv = ax.transData.inverted()
            dxd = inv.transform((event.x, event.y))[0] - inv.transform((x0, y0))[0]
            inv_p = ax_panel.transData.inverted()
            dyd = inv_p.transform((event.x, event.y))[1] - inv_p.transform((x0, y0))[1]
            nuevo_y = (ylim0[0] - dyd, ylim0[1] - dyd)
            ax_panel.set_ylim(*nuevo_y)
            # se recuerda como escala manual de ese TIPO de panel, igual que el
            # estirado: si no, apagar y encender el panel lo devolvería al sitio
            self._ylim_paneles[kind] = nuevo_y
            ax.set_xlim(xlim0[0] - dxd, xlim0[1] - dxd)   # dispara xlim_changed
            self._sync_dateedits(*ax.get_xlim())
            self._solicitar_frame_blit(ax)
            return
        lo, hi = self._drag_lim0
        centro = (lo + hi) / 2.0
        if self._drag_modo.startswith('y:'):
            # mismo estirado que el del precio, sobre el panel arrastrado
            kind, ax_panel = self._paneles[int(self._drag_modo.split(':')[1])]
            factor = np.exp(-(event.y - y0) * 0.005)
            medio = (hi - lo) / 2.0 * factor
            nuevo = (centro - medio, centro + medio)
            ax_panel.set_ylim(*nuevo)
            self._ylim_paneles[kind] = nuevo
            self._solicitar_frame_blit(ax)
        elif self._drag_modo == 'y':
            factor = np.exp(-(event.y - y0) * 0.005)
            medio = (hi - lo) / 2.0 * factor
            self._y_manual = True
            ax.set_ylim(centro - medio, centro + medio)
            self._solicitar_frame_blit(ax)
        else:
            factor = np.exp(-(event.x - x0) * 0.005)
            medio = (hi - lo) / 2.0 * factor
            ax.set_xlim(centro - medio, centro + medio)  # dispara xlim_changed
            self._sync_dateedits(*ax.get_xlim())
            self._solicitar_frame_blit(ax)

    @_no_crash
    def _on_release_ejes(self, event):
        self._finalizar_blit()
        self._drag_modo = None
        self._drag_inicio = None
        self._drag_lim0 = None

    @_no_crash
    def _on_scroll(self, event):
        """Rueda del ratón: zoom temporal (eje X) anclado bajo el cursor y
        ANIMADO hacia el objetivo con easing (como LWC): los ticks de una
        ráfaga solo re-anclan el destino y _timer_zoom lo interpola tick a
        tick, pintando a través del pipeline de frames diferidos. La sesión
        de blit NO se abre aquí: su draw() síncrono de captura del fondo
        cuesta ~20 ms, así que se abre de forma diferida en
        _pintar_frame_pendiente cuando llega el primer frame. Se cierra ~350
        ms después del último tick vía QTimer."""
        ax = getattr(self, '_ax_principal', None)
        if ax is None:
            return
        self._sesion_activa = True
        lo, hi = ax.get_xlim()
        ancla = event.xdata if event.xdata is not None else (lo + hi) / 2.0
        subir = (event.step > 0) if getattr(event, 'step', 0) else (event.button == 'up')
        factor = 0.85 if subir else 1.0 / 0.85
        self._zoom_target = (ancla + (lo - ancla) * factor,
                             ancla + (hi - ancla) * factor)
        self._zoom_pasos = 0
        if not self._timer_zoom.isActive():
            self._timer_zoom.start()
        self._solicitar_frame_blit(ax)
        self._scroll_timer.start(350)

    def _avanzar_zoom(self):
        """Un paso de la animación de zoom: acerca el xlim actual al objetivo
        con easing exponencial; al quedar suficientemente cerca, o tras
        MAX_PASOS_ZOOM pasos, ajusta el objetivo exacto y se detiene.

        El tope de pasos es lo que hace ágil la rueda: sin él, cada notch
        animaba ~20 frames (~600 ms con los frames de blit) y la ráfaga
        saturaba el hilo de la GUI — el lag y el parpadeo al hacer scroll.
        Con el tope el easing se resuelve en ~6 frames y la rueda responde."""
        ax = getattr(self, '_ax_principal', None)
        if ax is None or self._zoom_target is None:
            self._timer_zoom.stop()
            return
        self._zoom_pasos += 1
        lo, hi = ax.get_xlim()
        t_lo, t_hi = self._zoom_target
        d_lo, d_hi = t_lo - lo, t_hi - hi
        if (t_hi - t_lo) <= 0:
            self._timer_zoom.stop()
            return
        if (self._zoom_pasos >= MAX_PASOS_ZOOM
                or max(abs(d_lo), abs(d_hi)) < (t_hi - t_lo) * 1e-4):
            # prácticamente en el objetivo (o pasos agotados): ajustar exacto
            # y terminar
            ax.set_xlim(t_lo, t_hi)
            self._timer_zoom.stop()
        else:
            ax.set_xlim(lo + d_lo * 0.35, hi + d_hi * 0.35)
        self._sync_dateedits(*ax.get_xlim())
        self._solicitar_frame_blit(ax)

    # ── cruceta de hover (estilo TradingView) ──
    #
    # Capa aparte de la de pan/zoom: aquella repinta velas y trades en cada
    # frame de arrastre, esta solo cuatro artistas mientras el ratón se pasea
    # con el gráfico quieto. Comparten canvas pero nunca están activas a la
    # vez (la cruceta se esconde en cuanto empieza un arrastre), así que cada
    # una cachea su propio fondo.
    #
    # Todos los artistas van con animated=True: canvas.draw() los OMITE, y así
    # el fondo cacheado nunca sale con una cruceta vieja horneada dentro, que
    # es el fallo clásico de este patrón (se ve como un rastro fantasma que no
    # se borra hasta el siguiente redibujado completo).
    def _crear_cruceta(self):
        """Crea los artistas de la cruceta sobre los paneles ya posicionados.
        Se llama desde _dibujar_principal en cada dibujo completo porque
        fig.clear() destruye los Axes que los contienen."""
        self._art_cruceta = []
        self._cruceta_v = []       # línea vertical: una por panel, se ven todas
        self._cruceta_h = []       # horizontal: solo la del panel bajo el ratón
        self._cruceta_lbl_y = []   # etiqueta de precio/valor, idem
        self._cruceta_lbl_x = None
        if not self._paneles:
            return
        for _kind, axx in self._paneles:
            # líneas sólidas y etiquetas como la cruceta de Lightweight Charts:
            # #758696 semitransparente y fondo #4c525e con texto blanco
            v = axx.axvline(0, color=CRUCETA_COLOR, linewidth=1.0, alpha=0.5,
                            zorder=95, visible=False, animated=True)
            h = axx.axhline(0, color=CRUCETA_COLOR, linewidth=1.0, alpha=0.5,
                            zorder=95, visible=False, animated=True)
            # la franja de números está a la derecha en todos los paneles
            # (yaxis.tick_right), así que la etiqueta sale por x=1 en fracción
            # de ejes y sigue al ratón en Y en coordenadas de datos
            lbl_y = axx.annotate(
                "", xy=(1, 0), xycoords=('axes fraction', 'data'),
                xytext=(5, 0), textcoords='offset points',
                ha='left', va='center', fontsize=7, color=CRUCETA_TX,
                bbox=dict(boxstyle='round,pad=0.3', fc=CRUCETA_BG,
                          ec=CRUCETA_BG),
                zorder=102, annotation_clip=False, visible=False,
                animated=True)
            self._cruceta_v.append(v)
            self._cruceta_h.append(h)
            self._cruceta_lbl_y.append(lbl_y)
            self._art_cruceta += [v, h, lbl_y]
        # la fecha va una sola vez, bajo el último panel: es el único con las
        # etiquetas del eje X visibles (ver _aplicar_pesos_paneles)
        ax_ultimo = self._paneles[-1][1]
        self._cruceta_lbl_x = ax_ultimo.annotate(
            "", xy=(0, 0), xycoords=('data', 'axes fraction'),
            xytext=(0, -5), textcoords='offset points',
            ha='center', va='top', fontsize=7, color=CRUCETA_TX,
            bbox=dict(boxstyle='round,pad=0.3', fc=CRUCETA_BG, ec=CRUCETA_BG),
            zorder=102, annotation_clip=False, visible=False, animated=True)
        self._art_cruceta.append(self._cruceta_lbl_x)

    def _artistas_hover(self):
        """Capa de hover completa: cruceta + tooltip de operación."""
        arts = list(getattr(self, '_art_cruceta', ()))
        annot = getattr(self, '_annot_trade', None)
        if annot is not None:
            arts.append(annot)
        return arts

    def _capturar_fondo_hover(self, _event=None):
        """Cachea el lienzo tras CUALQUIER draw() completo (primer pintado,
        resize, fin de un arrastre, cambio de capa...), que es justo cuando el
        fondo de la cruceta deja de ser válido.

        Se salta la captura durante _iniciar_sesion_blit: ese draw() deja el
        lienzo sin velas ni trades a propósito, y guardarlo aquí dejaría la
        cruceta pintando sobre un gráfico vacío al soltar el arrastre."""
        if self._capturando_fondo_drag:
            return
        # un draw() completo se lleva por delante lo que hubiera pintado el
        # último blit de hover, así que ya no hay nada que borrar
        self._hover_pintado = False
        try:
            self._bg_hover = self.canvas.copy_from_bbox(self.fig.bbox)
        except Exception:
            self._bg_hover = None

    def _pintar_hover_blit(self):
        """Restaura el fondo cacheado y repinta encima la capa de hover.

        Se salta el trabajo cuando no hay nada que pintar NI nada que borrar
        (_hover_pintado recuerda si el último frame dejó algo en pantalla): si
        no, pasear el ratón por los márgenes del lienzo dispararía un blit por
        cada evento de movimiento sin cambiar un solo píxel."""
        if self._bg_hover is None:
            return
        arts = [a for a in self._artistas_hover()
                if a.get_visible() and a.axes is not None]
        if not arts and not self._hover_pintado:
            return
        self.canvas.restore_region(self._bg_hover)
        for art in arts:
            art.axes.draw_artist(art)
        self.canvas.blit(self.fig.bbox)
        self._hover_pintado = bool(arts)

    def _ocultar_cruceta(self, repintar=True):
        for art in getattr(self, '_art_cruceta', ()):
            art.set_visible(False)
        if repintar:
            self._pintar_hover_blit()

    @staticmethod
    def _fmt_nivel(valor, rango):
        """Decimales atados a la escala visible, no al activo: en un panel de
        RSI sobran los decimales y en un par de divisas hacen falta cuatro."""
        if rango >= 10:
            dec = 2
        elif rango >= 0.1:
            dec = 4
        else:
            dec = 6
        return f"{valor:,.{dec}f}"

    def _actualizar_cruceta(self, event, zona):
        """Coloca la cruceta sobre la vela bajo el cursor y repinta la capa de
        hover (cruceta + tooltip de operación) en una sola pasada de blit.

        La vertical se engancha al centro de la vela —trivial ahora que el eje
        es el índice— para que la fecha de la etiqueta sea la de una vela real
        y no un instante interpolado entre dos. La horizontal va libre, que es
        lo que permite usarla para medir un nivel de precio a ojo."""
        dentro = (zona is not None and (zona == 'pan' or zona.startswith('pan:'))
                  and event is not None and event.inaxes is not None
                  and event.xdata is not None and event.ydata is not None)
        ts = getattr(self, '_ts_full', None)
        if not dentro or ts is None or len(ts) == 0 or not self._cruceta_v:
            self._ocultar_cruceta()
            return
        i = int(np.clip(round(event.xdata), 0, len(ts) - 1))
        for v in self._cruceta_v:
            v.set_xdata([i, i])
            v.set_visible(True)
        lo, hi = event.inaxes.get_ylim()
        rango = abs(hi - lo)
        for axx, h, lbl in zip((a for _k, a in self._paneles),
                               self._cruceta_h, self._cruceta_lbl_y):
            if axx is not event.inaxes:
                h.set_visible(False)
                lbl.set_visible(False)
                continue
            h.set_ydata([event.ydata, event.ydata])
            h.set_visible(True)
            lbl.xy = (1, event.ydata)
            lbl.set_text(self._fmt_nivel(event.ydata, rango))
            lbl.set_visible(True)
        self._cruceta_lbl_x.xy = (i, 0)
        self._cruceta_lbl_x.set_text(
            ts[i].strftime(getattr(self, '_patron_fecha', '%Y-%m-%d')))
        self._cruceta_lbl_x.set_visible(True)
        self._pintar_hover_blit()

    # ── blitting: pan/zoom fluido sin re-rasterizar la figura completa ──
    def _artistas_dinamicos(self):
        """Artistas que cambian de posición en pantalla con cada pan/zoom
        (velas, sombreado IS/OOS, compra/venta, trayecto, stop-loss,
        overlays de indicadores MA/Bollinger/patrones) — se excluyen del
        fondo cacheado de blitting y se repintan cada frame. Los overlays
        van aquí (y no ocultos toda la sesión) para que no parpadeen: con
        rueda de ratón "a saltos" (sin trackpad) cada notch suele abrir y
        cerrar su propia sesión de blit (ver _on_scroll), y ocultarlos solo
        al abrir/cerrar sesión se veía como que el gráfico "se recargaba"
        en cada scroll. Incluye las líneas de los paneles de oscilador
        (_art_osciladores): comparten eje X con el precio (sharex), así que
        con cada pan/zoom su ventana visible cambia igual que las velas —
        si no se redibujaran cada frame se quedarían "congeladas" en el
        bitmap de fondo hasta soltar el arrastre."""
        return list(self._art_datos) + self._art_fijos_dinamicos \
            + self._art_overlays_extra + self._art_osciladores + [
            a for a in (self._scatter_compra, self._scatter_venta,
                        self._scatter_tramo_compra, self._scatter_tramo_venta,
                        # marcas del precio de ejecución: se mueven con las
                        # flechas, así que van en la misma capa
                        self._scatter_nivel_compra, self._scatter_nivel_venta,
                        self._scatter_nivel_tramo_compra,
                        self._scatter_nivel_tramo_venta,
                         self._art_trayecto, self._art_salida_cuadros,
                         self._art_salida_segmentos, self._art_stop_track,
                         self._art_entrada_track, self._art_zona_riesgo,
                         self._art_ultimo_precio, self._lbl_ultimo_precio)
            if a is not None]

    def _yaxis_panel_arrastrado(self):
        """Eje Y del panel de oscilador que se está estirando o desplazando
        ahora mismo, o None. Solo ESE eje se repinta por frame: cada uno cuesta
        ~10 ms, así que meter todos los paneles siempre saldría caro para nada.

        Cubre los dos gestos que mueven la escala de un panel: 'y:<i>' (estirar
        desde su franja de números) y 'pan:<i>' (arrastrar dentro del panel).
        Si el eje no entra aquí se queda pegado en el bitmap de fondo y sus
        números no acompañan a la línea durante el arrastre."""
        modo = self._drag_modo
        if not modo or not (modo.startswith('y:') or modo.startswith('pan:')):
            return None
        idx = int(modo.split(':')[1])
        if idx >= len(self._paneles):
            return None
        return self._paneles[idx][1].yaxis

    def _iniciar_sesion_blit(self):
        """Al empezar un arrastre/scroll: cachea un bitmap de fondo SIN la
        capa dinámica (para poder repintarla encima en cada frame sin dejar
        "fantasmas" de su posición anterior). copy_from_bbox necesita un
        buffer ya renderizado, por eso el render de aquí es síncrono, no
        draw_idle().

        El render se hace directo al buffer del renderer (fig.draw + clear)
        en vez de canvas.draw(): este último además de rasterizar llama a
        update(), encolando un repintado de Qt del frame "despojado" (sin
        velas, trades ni overlays) que puede llegar a pantalla antes de que
        el blit de _pintar_frame_blit recomponga el frame completo — es el
        parpadeo que se veía al hacer scroll/clic sobre el gráfico. Al no
        pedirle a Qt que repinte, la pantalla pasa directamente del frame
        anterior al frame completo de blit, sin estados intermedios.

        El frame "despojado" se rasteriza en un RendererAgg APARTE (offscreen),
        nunca en el buffer compartido del canvas: así, cualquier repintado de
        Qt que caiga a mitad de captura (p. ej. un update() asíncrono de un
        frame anterior, o un repintado del sistema en ventana real) lee del
        buffer del canvas el ÚLTIMO frame completo, nunca el despojado. Ese
        destello —el gráfico "mal colocado o extendido" un microsegundo al
        hacer scroll rápido— era exactamente el frame sin velas llegando a
        pantalla."""
        ax = getattr(self, '_ax_principal', None)
        if ax is None:
            return
        dinamicos = self._artistas_dinamicos() + [ax.xaxis, ax.yaxis]
        ultimo_ax = self._paneles[-1][1] if self._paneles else ax
        if ultimo_ax is not ax:
            # el último panel apilado es el único con etiquetas de fecha
            # visibles (los demás las ocultan, ver _aplicar_pesos_paneles)
            dinamicos.append(ultimo_ax.xaxis)
        eje_panel = self._yaxis_panel_arrastrado()
        if eje_panel is not None:
            dinamicos.append(eje_panel)
        for art in dinamicos:
            art.set_visible(False)
        # render "despojado" en un renderer aparte: que _capturar_fondo_hover
        # no se lo quede como fondo de la cruceta y, sobre todo, que el buffer
        # compartido del canvas nunca llegue a contener el frame sin velas
        self._capturando_fondo_drag = True
        try:
            from matplotlib.backends.backend_agg import RendererAgg
            pw, ph = self.canvas.get_width_height(physical=True)
            r_off = RendererAgg(pw, ph, self.fig.dpi)
            self.fig.draw(r_off)
        finally:
            self._capturando_fondo_drag = False
        self._blit_bg = r_off.copy_from_bbox(self.fig.bbox)
        for art in dinamicos:
            art.set_visible(True)
        self._pintar_frame_blit(ax)

    def _solicitar_frame_blit(self, ax):
        """Pide un frame de arrastre, sin pintarlo en el acto.

        Un ratón a 125 Hz manda un evento cada 8 ms y rasterizar un frame
        cuesta bastante más, así que atender uno por evento deja el gráfico
        corriendo por detrás del cursor: la sensación de "va trabado" no viene
        solo de lo que tarda cada frame, sino de pintar frames ya caducados.
        Con el timer a 0 ms los eventos intermedios solo mueven el xlim y se
        pinta una vez por vuelta del bucle de Qt, siempre con la última
        posición. Mismo patrón que _timer_graficos."""
        self._ax_frame_pendiente = ax
        if not self._timer_frame.isActive():
            self._timer_frame.start()

    def _pintar_frame_pendiente(self):
        ax = self._ax_frame_pendiente
        self._ax_frame_pendiente = None
        if ax is None:
            return
        # el redibujado de velas/overlays se difiere a este punto (una vez por
        # ciclo de eventos, ver _on_xlim_changed) para que una ráfaga de
        # scroll no redibuje por cada evento de ratón; también se abre aquí,
        # de forma diferida, la sesión de blit del scroll (su draw() síncrono
        # de captura del fondo solo paga cuando llega el primer frame)
        if self._datos_sucios:
            self._datos_sucios = False
            self._redibujar_datos(ax)
        if self._blit_bg is None:
            if self._sesion_activa:
                # _iniciar_sesion_blit captura el fondo y pinta ya el primer
                # frame con los datos recién redibujados
                self._iniciar_sesion_blit()
            else:
                self.canvas.draw_idle()
        else:
            self._pintar_frame_blit(ax)

    def _finalizar_blit(self):
        """Al soltar el ratón / tras el debounce de scroll: descarta la sesión
        de blit dejando en pantalla el frame final SIN re-rasterizar toda la
        figura.

        El cierre solía terminar con un draw_idle() completo: cada debounce de
        scroll (y cada soltar el ratón) re-renderizaba TODO el gráfico (~70 ms
        de bloqueo por rasterizado) — era el parpadeo al hacer scroll. Como el
        último frame de blit ya es correcto, se cierra pintándolo por blitting
        (aplicando antes el redibujado pendiente, si lo hay) y solo se
        refresca el fondo de la cruceta para que el hover siga consistente."""
        # un frame pendiente pintaría por blitting con el fondo ya descartado
        self._timer_frame.stop()
        self._timer_zoom.stop()
        ax = self._ax_frame_pendiente
        self._ax_frame_pendiente = None
        self._sesion_activa = False
        # si el zoom animado quedó a medias, ajustar el objetivo exacto antes
        # del frame final (set_xlim dispara _on_xlim_changed, que ya redibuja)
        if self._zoom_target is not None:
            if ax is None:
                ax = getattr(self, '_ax_principal', None)
            if ax is not None:
                ax.set_xlim(*self._zoom_target)
            self._zoom_target = None
        # si el último redibujado quedó pendiente (p. ej. soltar el ratón
        # antes de que dispare el timer de frames), ejecutarlo ANTES de pintar:
        # sin esto el frame final llevaría las velas del xlim ANTERIOR (la
        # misma serie repetida en cualquier ventana de zoom)
        if self._datos_sucios:
            self._datos_sucios = False
            if ax is None:
                ax = getattr(self, '_ax_principal', None)
            if ax is not None:
                self._redibujar_datos(ax)
        # pintar el estado final: por blitting si queda fondo válido (la capa
        # dinámica ya está al día, el fondo estático no ha cambiado), si no con
        # un draw_idle — el único repintado completo de cierre
        if ax is not None:
            self._pintar_frame_blit(ax)
        else:
            self.canvas.draw_idle()
        self._blit_bg = None
        # refrescar el fondo de la cruceta desde el buffer recién pintado: sin
        # el draw_idle de cierre ya no hay draw_event que lo recapture
        self._bg_hover = self.canvas.copy_from_bbox(self.fig.bbox)
        self._hover_pintado = False

    def _on_resize_canvas(self, event):
        """El bitmap cacheado tiene el tamaño de figura de cuando se
        capturó; si la ventana cambia de tamaño a mitad de un arrastre,
        restore_region contra un bbox de tamaño distinto corrompe el
        repintado — se invalida y, si el arrastre sigue activo, se
        recaptura de inmediato. Los paneles usan posiciones en fracción de
        figura (0-1), así que no hace falta recalcularlas aquí — siguen
        siendo válidas al cambiar de tamaño."""
        self._blit_bg = None
        # el fondo de la cruceta tiene el mismo problema de tamaño; el
        # draw_event que sigue al resize lo vuelve a capturar
        self._bg_hover = None
        if self._drag_modo is not None and not self._drag_modo.startswith('resize_panel:'):
            self._iniciar_sesion_blit()

    def _alturas_marcadores(self, lado, mask):
        """(x, y_flecha, y_nivel) de los marcadores de un lado ('compra' o
        'venta'), recortados por `mask`.

        y_nivel es el precio REAL de ejecución; y_flecha va pegada al extremo
        de la vela para que se lea sin taparse con el cuerpo. Este es el ÚNICO
        sitio donde se decide dónde va cada marcador: lo consumen el dibujo, el
        repintado por blitting y la detección del ratón. Si cada uno lo
        calculara por su cuenta, bastaría con tocar dos para que las flechas
        saltaran al hacer scroll o el tooltip dejara de encontrarlas.

        Ojo con el indexado: los arrays de índice son por OPERACIÓN y apuntan a
        una barra, así que las series por barra se leen con `idx` y los precios
        de ejecución —que también son por operación— con la propia `mask`.
        """
        if lado == 'compra':
            idx = self._compra_idx_full[mask]
            y_nivel = self._compra_precio_full[mask]
            y_flecha = self._l_full[idx] - self._offset_flecha
        else:
            idx = self._venta_idx_full[mask]
            y_nivel = self._venta_precio_full[mask]
            y_flecha = self._h_full[idx] + self._offset_flecha
        return self._x_full[idx], y_flecha, y_nivel

    def _alturas_tramos(self, lado, x0, x1):
        """Equivalente de _alturas_marcadores para los tramos 2+ de entrada
        escalonada, que viven en resultado['entradas'] y no en 'trades'."""
        if lado == 'compra':
            idx_full, precio_full = (self._tramo_compra_idx_full,
                                     self._tramo_compra_precio_full)
            signo = -1.0
            serie = self._l_full
        else:
            idx_full, precio_full = (self._tramo_venta_idx_full,
                                     self._tramo_venta_precio_full)
            signo = 1.0
            serie = self._h_full
        visible = self._tramo_extra_visible(idx_full, x0, x1)
        idx = idx_full[visible]
        return (self._x_full[idx], serie[idx] + signo * self._offset_flecha,
                precio_full[visible])

    def _actualizar_trades_dinamicos(self, x0, x1):
        """Muta in-place (set_offsets/set_segments/set_verts) los artistas
        persistentes de compra/venta/trayecto/stop-loss, recortados a la
        ventana visible actual — sin crear objetos matplotlib nuevos en
        cada frame de arrastre."""
        if not self._tr:
            return
        mask = self._trades_visibles(x0, x1)
        # mismas alturas que al crear los artistas (_alturas_marcadores es el
        # único sitio que las decide), o los marcadores saltarían al arrastrar
        for lado in ('compra', 'venta'):
            flecha = getattr(self, f'_scatter_{lado}')
            if flecha is None:
                continue
            x, y_flecha, y_nivel = self._alturas_marcadores(lado, mask)
            flecha.set_offsets(np.column_stack([x, y_flecha]))
            getattr(self, f'_scatter_nivel_{lado}').set_offsets(
                np.column_stack([x, y_nivel]))
        for lado in ('compra', 'venta'):
            circulo = getattr(self, f'_scatter_tramo_{lado}')
            if circulo is None:
                continue
            x, y_flecha, y_nivel = self._alturas_tramos(lado, x0, x1)
            circulo.set_offsets(np.column_stack([x, y_flecha]))
            getattr(self, f'_scatter_nivel_tramo_{lado}').set_offsets(
                np.column_stack([x, y_nivel]))
        if self._art_trayecto is not None:
            self._art_trayecto.set_segments(self._trayecto_segmentos_full[mask])
        if self._art_salida_cuadros is not None:
            # a diferencia del trayecto/stop, aquí "mask" ya cubre TODOS los
            # trades (la caja de Salida se dibuja en todas las operaciones,
            # no en un subconjunto): sin máscara base intermedia.
            self._art_salida_cuadros.set_verts(self._salida_cuadros_full[mask])
            self._art_salida_cuadros.set_facecolor(self._salida_colores_full[mask])
            self._art_salida_segmentos.set_segments(self._salida_segmentos_full[mask])
            self._art_salida_segmentos.set_color(self._salida_colores_full[mask])
        # _art_stop_track / _art_entrada_track / _art_zona_riesgo NO se
        # recortan aquí: son series por VELA (longitud n, no una por trade),
        # matplotlib ya las recorta a la ventana visible vía el xlim del eje,
        # igual que hace con las velas — solo hace falta que estén en
        # _artistas_dinamicos() para que se redibujen en cada frame.

    def _pintar_frame_blit(self, ax):
        """Un frame de arrastre/zoom: restaura el fondo cacheado y repinta
        SOLO la capa dinámica (velas ya actualizadas por _redibujar_datos
        vía el callback xlim_changed, ejes, sombreado, trades recortados al
        rango visible actual) sobre él."""
        if self._blit_bg is None:
            self.canvas.draw_idle()
            return
        self.canvas.restore_region(self._blit_bg)
        x0, x1 = ax.get_xlim()
        self._actualizar_trades_dinamicos(x0, x1)
        dinamicos = self._artistas_dinamicos()
        dinamicos.sort(key=lambda a: a.get_zorder())
        ax.draw_artist(ax.xaxis)
        ax.draw_artist(ax.yaxis)
        ultimo_ax = self._paneles[-1][1] if self._paneles else ax
        if ultimo_ax is not ax:
            ultimo_ax.draw_artist(ultimo_ax.xaxis)
        eje_panel = self._yaxis_panel_arrastrado()
        if eje_panel is not None:
            eje_panel.axes.draw_artist(eje_panel)
        for art in dinamicos:
            # los overlays/líneas de oscilador pertenecen a su propio Axes,
            # no siempre al de precio — draw_artist solo necesita el
            # renderer (compartido a nivel Figure), así que basta con
            # invocarlo desde el Axes dueño del artista
            (art.axes or ax).draw_artist(art)
        # Presentar el frame de forma ASÍNCRONA (update() en vez de blit():
        # blit() llama a repaint(), un repintado síncrono que en ráfagas de
        # scroll/arrastre se encadenaba decenas de veces por segundo con su
        # eraseRect + drawImage fuera del ciclo de composición — el parpadeo
        # al hacer scroll rápido. update() lo pinta en el ciclo normal de Qt,
        # coalesciendo los frames que lleguen antes de repintar (siempre el
        # último, que es el que se quiere ver). El buffer ya tiene el frame
        # completo (restore_region + draw_artist), así que el repintado
        # asíncrono lo muestra tal cual.
        self.canvas.update()

    def _dibujar_principal(self, xlim=None, ylim=None):
        p = self._payload
        if p is None:
            return
        bombear_eventos()
        ts = pd.DatetimeIndex(p['timestamps'])
        y = p['close']   # PRECIO real del activo, no log-return en tanto por uno
        corte = p['corte']
        tr = p['resultado']['trades']
        self._tr = tr
        self._entr = p['resultado'].get('entradas')

        # eje X = ÍNDICE de vela, no la fecha (ver eje_fechas_ordinal en
        # plot_common): con date2num, los tramos sin mercado —fin de semana,
        # festivos, cierres de sesión— ocupaban su hueco en el eje y la serie
        # salía a peine. Además el hueco aparecía y desaparecía según el zoom,
        # porque al alejar _decimar_ohlc submuestrea (o antes, agregaba en
        # bloques) y el ancho (mediana de np.diff) crecía hasta taparlo. Con
        # el índice, la separación entre velas es 1 en todo momento y a
        # cualquier zoom.
        # ts se conserva aparte: es lo que traduce índice -> fecha en las
        # etiquetas del eje, la cruceta y los QDateEdit de rango.
        self._ts_full = ts
        self._x_full = np.arange(len(ts), dtype=float)
        # nombre propio: `x` ya se usa más abajo para las X de los marcadores
        # (_alturas_marcadores), y pisarlo dejaba los overlays con 2 puntos
        x_idx = self._x_full
        # patrón de la etiqueta de fecha de la cruceta: con velas de un día o
        # más, la hora siempre sería 00:00 y solo estorba
        paso = (ts[1] - ts[0]) if len(ts) > 1 else pd.Timedelta(0)
        self._patron_fecha = ('%Y-%m-%d %H:%M' if paso < pd.Timedelta(days=1)
                              else '%Y-%m-%d')
        self._o_full = p.get('open', p['close'])
        self._h_full = p.get('high', p['close'])
        self._l_full = p.get('low', p['close'])
        self._c_full = p['close']
        self._art_datos = []
        self._art_overlays_extra = []
        self._art_osciladores = []

        ind = self._recolectar_indicadores(p)
        # el panel de capas se repuebla ANTES de dibujar: de él salen tanto los
        # colores de cada overlay como qué capas están encendidas
        self._actualizar_panel_capas(ind, p)
        paneles_spec = []
        n_osc_disponibles = 0
        for kind, clave in PANELES_OSCILADOR:
            datos = ind[clave]
            if not datos:
                continue
            n_osc_disponibles += 1
            # apagar un oscilador quita su panel entero, no solo sus líneas: el
            # precio se queda ese espacio. _pesos_paneles va indexado por kind,
            # así que la altura que el usuario le hubiera dado a mano se
            # conserva y vuelve intacta al reencenderlo.
            if self._capa_activa(f'panel:{kind}'):
                paneles_spec.append((kind, datos))

        self.fig.clear()
        # la figura se reconstruye entera: cualquier fondo de blit cacheado
        # pertenece a la figura anterior y restaurarlo pintaría "fantasmas"
        # (velas duplicadas) sobre la nueva si una sesión sigue abierta
        self._blit_bg = None
        # el repintado completo fija el xlim que le pasen: cualquier zoom con
        # rueda a medias queda cancelado
        self._timer_zoom.stop()
        self._zoom_target = None
        n_osc = len(paneles_spec)
        gs = self.fig.add_gridspec(1 + n_osc, 1)
        ax = self.fig.add_subplot(gs[0, 0])
        _style_ax(ax)
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position('right')
        self._paneles = [('precio', ax)]
        # tooltip de hover (lotaje/RR) sobre las flechas de compra/venta —
        # recreado en cada redibujo completo porque fig.clear() destruye el
        # anterior; la lógica de mostrar/ocultar vive en _on_motion_ejes
        # (mismo callback permanente que ya gestiona el cursor de pan/zoom).
        self._annot_trade = ax.annotate(
            "", xy=(0, 0), xytext=(15, 15), textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.4", fc=FIG_BG, ec=GRID_C, alpha=0.95),
            color=AX_FG, fontsize=7, zorder=99, annotation_clip=False)
        self._annot_trade.set_visible(False)
        # animated: el tooltip lo pinta la capa de hover por blitting junto a la
        # cruceta (ver _pintar_hover_blit). Marcado así, canvas.draw() lo omite
        # y nunca queda horneado en el bitmap de fondo.
        self._annot_trade.set_animated(True)
        for i, (kind, datos) in enumerate(paneles_spec, start=1):
            ax_osc = self.fig.add_subplot(gs[i, 0], sharex=ax)
            self._paneles.append((kind, ax_osc))
            self._art_osciladores += self._dibujar_panel_oscilador(
                ax_osc, kind, datos, x_idx, y, self._h_full, self._l_full)
        self._aplicar_pesos_paneles()
        self._crear_cruceta()
        # la altura se reserva por los osciladores que el sistema TIENE, no por
        # los que estén encendidos: si encogiera al apagar uno, el canvas se
        # haría más bajo y toda la pestaña se recolocaría bajo el ratón. Así el
        # hueco liberado se lo queda el panel de precio y nada más se mueve.
        self.canvas.setMinimumHeight(int(480 + 90 * n_osc_disponibles))

        # sombreado IS / OOS — artistas "dinámicos baratos": se repintan en
        # cada frame de blit junto con las velas (ver _pintar_frame_blit)
        p1 = ax.axvspan(0.0, corte - 1 if corte > 0 else 0.0,
                        color=AZUL, alpha=0.05)
        self._art_fijos_dinamicos = [p1]
        if corte < len(ts):
            p2 = ax.axvspan(corte, len(ts) - 1, color=AMBAR, alpha=0.06)
            p3 = ax.axvline(corte, color=AMBAR, linewidth=0.9, linestyle='--', alpha=0.8)
            self._art_fijos_dinamicos += [p2, p3]

        # Ticks contados y etiquetas cortas. Cada etiqueta se rasteriza de nuevo
        # en CADA frame de arrastre (matplotlib no cachea el texto dibujado) y
        # sale a ~1,4 ms la unidad: las 9 fechas largas del formateador por
        # defecto ("07-06 12") costaban 17 ms por frame, más que las propias
        # velas. eje_fechas_ordinal escribe "mar" y promociona el año solo en su
        # frontera, así que además se lee mejor.
        #
        # Con sharex, los Axes apilados COMPARTEN el objeto Ticker del eje X, así
        # que instalarlo aquí basta para que el panel de abajo —el único con las
        # etiquetas visibles, ver _aplicar_pesos_paneles— las herede.
        eje_fechas_ordinal(ax, ts, max_ticks=6)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

        if xlim is not None:
            ax.set_xlim(*xlim)
        else:
            ax.set_xlim(self._x_full[0], self._x_full[-1])
        self._redibujar_datos(ax)
        if ylim is not None:
            # escala de precio que el usuario ajustó a mano. _redibujar_datos
            # no la recalcula en ese caso (self._y_manual), y este Axes se acaba
            # de crear: sin fijarla aquí, matplotlib la autoescalaría a TODA la
            # serie y el gráfico daría un salto al encender o apagar una capa.
            ax.set_ylim(*ylim)

        # último precio, como LWC/TradingView: línea discontinua en el último
        # close + badge en la escala derecha. Creados aquí (fig.clear() los
        # destruye); el badge se reposiciona en cada _redibujar_datos y ambos
        # van en _artistas_dinamicos() para repintarse en cada frame de blit.
        ultimo = float(self._c_full[-1])
        self._art_ultimo_precio = ax.axhline(
            ultimo, color=GRIS, linewidth=1.0, linestyle='--', alpha=0.55,
            zorder=2.0)
        self._lbl_ultimo_precio = ax.annotate(
            "", xy=(1, ultimo), xycoords=('axes fraction', 'data'),
            xytext=(-2, 0), textcoords='offset points',
            ha='right', va='center', fontsize=7, color=CRUCETA_TX,
            bbox=dict(boxstyle='round,pad=0.3', fc=CRUCETA_BG, ec=CRUCETA_BG),
            zorder=102, annotation_clip=False)
        self._actualizar_ultimo_precio(ax)

        n_tr = len(tr['pnl'])
        mask_vis = self._trades_visibles(*ax.get_xlim())

        # flechas: verde = compra (abre long / cierra short), roja = venta —
        # en tonos más saturados que las velas para que no se camuflen.
        # Vectorizado (sin bucle Python) y recortado al rango visible; los
        # arrays _full se guardan para poder volver a recortar en cada
        # frame de pan/zoom sin reconstruir nada (_actualizar_trades_dinamicos).
        #
        # La flecha se pega al extremo de su vela (sin taparse con el cuerpo) y
        # el precio real de ejecución se marca aparte con un tick horizontal,
        # que permite comprobar a qué precio ejecutó de verdad el motor: el
        # nivel pactado en una orden límite, o el open más el slippage en una
        # entrada a mercado.
        #
        # La separación se mide contra la vela TÍPICA, no contra el rango total
        # del gráfico: con el rango total, en un activo que ha recorrido mucho
        # una vela es una fracción ínfima de ese rango y la flecha acababa
        # flotando a varias velas de distancia de la suya. El fallback cubre
        # series planas (mediana de rango = 0).
        rango_vela = float(np.nanmedian(self._h_full - self._l_full))
        self._offset_flecha = (
            rango_vela * 0.25 if rango_vela > 0
            else max(float(np.nanmax(self._h_full)
                           - np.nanmin(self._l_full)) * 0.001, 1e-9))
        # {barra de relleno: precio pedido} de las órdenes límite que llegaron a
        # ejecutarse. Una orden rellenada se resuelve (idx_fin) en la misma vela
        # en que el motor abre la posición, así que basta con esa barra para
        # saber si una entrada vino de un límite y a qué nivel se pidió. Se
        # calcula aquí y no en cada evento de ratón.
        ol_tt = p['resultado'].get('ordenes_limite') or {}
        self._orden_por_barra = {
            int(ol_tt['idx_fin'][k]): float(ol_tt['precio'][k])
            for k in range(len(ol_tt.get('idx_alta', ())))
            if int(ol_tt['resultado'][k]) == 0}
        self._scatter_compra = None
        self._scatter_venta = None
        self._scatter_nivel_compra = None
        self._scatter_nivel_venta = None
        if n_tr:
            es_long = tr['dir'] > 0
            self._compra_idx_full = np.where(es_long, tr['idx_entrada'], tr['idx_salida'])
            self._venta_idx_full = np.where(es_long, tr['idx_salida'], tr['idx_entrada'])
            # precios de EJECUCIÓN, por operación (no por vela): la compra es
            # la entrada de un largo o la salida de un corto, y al revés
            self._compra_precio_full = np.where(
                es_long, tr['precio_entrada'], tr['precio_salida'])
            self._venta_precio_full = np.where(
                es_long, tr['precio_salida'], tr['precio_entrada'])
            # para el hover de lotaje/RR (_on_motion_ejes): compra_idx_full[r]/
            # venta_idx_full[r] siguen siendo la fila r de `tr` sin reordenar,
            # así que unidades/r_multiple/es_long se leen con el mismo índice.
            self._es_long_full = es_long
            self._lotaje_full = tr['unidades']
            self._rr_full = tr['r_multiple']
            for lado, marca, color, etiqueta in (
                ('compra', '^', VERDE_FLECHA, 'Compra'),
                ('venta', 'v', ROJO_FLECHA, 'Venta'),
            ):
                x, y_flecha, y_nivel = self._alturas_marcadores(lado, mask_vis)
                flecha = ax.scatter(x, y_flecha, marker=marca, s=28,
                                    color=color, zorder=3, label=etiqueta)
                nivel = ax.scatter(x, y_nivel, marker='_', s=42, color=color,
                                   linewidths=1.2, alpha=0.9, zorder=3)
                setattr(self, f'_scatter_{lado}', flecha)
                setattr(self, f'_scatter_nivel_{lado}', nivel)
        else:
            self._compra_idx_full = np.array([], dtype=np.int64)
            self._venta_idx_full = np.array([], dtype=np.int64)
            self._compra_precio_full = np.array([], dtype=float)
            self._venta_precio_full = np.array([], dtype=float)
            self._es_long_full = np.array([], dtype=bool)
            self._lotaje_full = np.array([], dtype=float)
            self._rr_full = np.array([], dtype=float)

        # tramos 2+ de entrada escalonada (promediar/piramidar): círculo
        # hueco para no confundirlos con la apertura/cierre del trade — el
        # tramo 0 (la apertura) ya lo pinta compra/venta arriba. Viven en
        # resultado['entradas'], no en 'trades' (no son un cierre).
        self._scatter_tramo_compra = None
        self._scatter_tramo_venta = None
        self._scatter_nivel_tramo_compra = None
        self._scatter_nivel_tramo_venta = None
        self._tramo_compra_idx_full = np.array([], dtype=np.int64)
        self._tramo_venta_idx_full = np.array([], dtype=np.int64)
        self._tramo_compra_precio_full = np.array([], dtype=float)
        self._tramo_venta_precio_full = np.array([], dtype=float)
        entr = self._entr
        if entr is not None and len(entr.get('idx', [])):
            extra = entr['tramo'] > 0
            if extra.any():
                es_compra = extra & (entr['dir'] > 0)
                es_venta = extra & (entr['dir'] < 0)
                self._tramo_compra_idx_full = entr['idx'][es_compra]
                self._tramo_venta_idx_full = entr['idx'][es_venta]
                # entradas['precio'] ya es el precio de fill de ESE tramo
                self._tramo_compra_precio_full = entr['precio'][es_compra]
                self._tramo_venta_precio_full = entr['precio'][es_venta]
                for lado, color, etiqueta in (
                    ('compra', VERDE_FLECHA, 'Promedio'),
                    ('venta', ROJO_FLECHA, None),
                ):
                    x, y_flecha, y_nivel = self._alturas_tramos(lado, *ax.get_xlim())
                    circulo = ax.scatter(
                        x, y_flecha, marker='o', s=24, facecolors='none',
                        edgecolors=color, linewidths=1.3, zorder=3, label=etiqueta)
                    nivel = ax.scatter(x, y_nivel, marker='_', s=36, color=color,
                                       linewidths=1.0, alpha=0.9, zorder=3)
                    setattr(self, f'_scatter_tramo_{lado}', circulo)
                    setattr(self, f'_scatter_nivel_tramo_{lado}', nivel)

        # trayecto de cada operación (entrada→salida, precio real de fill),
        # opcional vía checkbox: la pendiente ya muestra de un vistazo si el
        # trade fue ganador o perdedor, y los marcadores de compra/venta ya
        # señalan la dirección en cada extremo — por eso un único
        # LineCollection (sin punta de flecha) en vez de un FancyArrowPatch
        # por trade, que con miles de trades era el verdadero cuello de
        # botella de cada redibujado (mismo antipatrón que ax.bar para las
        # velas, ver comentario en _redibujar_datos).
        self._art_trayecto = None
        self._trayecto_segmentos_full = None
        if n_tr and getattr(self, 'chk_trayecto', None) is not None \
                and self.chk_trayecto.isChecked():
            x_ent = self._x_full[tr['idx_entrada']]
            x_sal = self._x_full[tr['idx_salida']]
            self._trayecto_segmentos_full = np.stack([
                np.column_stack([x_ent, tr['precio_entrada']]),
                np.column_stack([x_sal, tr['precio_salida']]),
            ], axis=1)
            self._art_trayecto = LineCollection(
                self._trayecto_segmentos_full[mask_vis], colors=GRIS,
                linewidths=0.9, alpha=0.4, zorder=2.5)
            ax.add_collection(self._art_trayecto)
            ax.plot([], [], color=GRIS, linewidth=1.2, label='Trayecto')

        # «Mostrar operación»: trayectoria REAL del stop/precio medio (series
        # por vela emitidas por el motor, no reconstruidas desde los cierres)
        # + una caja de "Salida" por CADA operación, sin importar el motivo
        # ni si ganó o perdió — opcional, vía checkbox.
        self._art_stop_track = None
        self._art_entrada_track = None
        self._art_zona_riesgo = None
        self._art_salida_cuadros = None
        self._art_salida_segmentos = None
        self._salida_segmentos_full = None
        self._salida_cuadros_full = None
        self._salida_colores_full = None
        if getattr(self, 'chk_stop', None) is not None and self.chk_stop.isChecked():
            resultado = p['resultado']
            stop_track = resultado.get('stop_track')
            entrada_track = resultado.get('entrada_track')
            if stop_track is not None and entrada_track is not None:
                # zona de riesgo: se abre entre el precio de entrada vigente y
                # el stop vigente — se estrecha o desaparece sola cuando el
                # break-even anula el riesgo o cuando el setup no tiene stop
                # (ambas series NaN en ese tramo cortan el relleno)
                self._art_zona_riesgo = ax.fill_between(
                    self._x_full, entrada_track, stop_track, color=ROJO,
                    alpha=0.08, edgecolor='none', zorder=1.5)
                # línea del stop, escalonada: el salto vertical en la vela en
                # que el break-even/trailing mueve el nivel es justo lo que
                # se quería ver — 'steps-post' porque el valor de la vela i
                # rige hasta la i+1, no una rampa entre ambas
                (self._art_stop_track,) = ax.plot(
                    self._x_full, stop_track, drawstyle='steps-post',
                    color=ROJO, linestyle='--', linewidth=0.9, alpha=0.8,
                    zorder=2.5)
                # precio de entrada vigente (medio ponderado tras promediar):
                # punteado para no confundirlo con el trayecto (sólido, mismo
                # gris) si ambos checkboxes están activos a la vez
                (self._art_entrada_track,) = ax.plot(
                    self._x_full, entrada_track, drawstyle='steps-post',
                    color=GRIS, linestyle=':', linewidth=1.0, alpha=0.6,
                    zorder=2.4)
                if np.isfinite(stop_track).any():
                    ax.plot([], [], color=ROJO, linestyle='--', linewidth=1.0,
                            label='Stop Loss')
                if np.isfinite(entrada_track).any():
                    ax.plot([], [], color=GRIS, linestyle=':', linewidth=1.0,
                            label='Precio de entrada (medio)')

            if n_tr:
                idx_e = tr['idx_entrada']
                idx_s = tr['idx_salida']
                ent_v = tr['precio_entrada']
                sal_v = tr['precio_salida']
                x_e, x_s = self._x_full[idx_e], self._x_full[idx_s]
                self._salida_segmentos_full = np.stack([
                    np.column_stack([x_e, sal_v]),
                    np.column_stack([x_s, sal_v])], axis=1)
                self._salida_cuadros_full = np.stack([
                    np.column_stack([x_e, ent_v]), np.column_stack([x_s, ent_v]),
                    np.column_stack([x_s, sal_v]), np.column_stack([x_e, sal_v])],
                    axis=1)
                self._salida_colores_full = np.where(tr['pnl'] > 0, VERDE, ROJO)
                self._art_salida_cuadros = PolyCollection(
                    self._salida_cuadros_full[mask_vis],
                    facecolors=self._salida_colores_full[mask_vis], alpha=0.08,
                    edgecolors='none', zorder=1.5)
                self._art_salida_segmentos = LineCollection(
                    self._salida_segmentos_full[mask_vis], linewidths=0.8,
                    colors=self._salida_colores_full[mask_vis],
                    linestyles='--', alpha=0.7, zorder=2.5)
                ax.add_collection(self._art_salida_cuadros)
                ax.add_collection(self._art_salida_segmentos)
                # una sola entrada de leyenda: la caja significa siempre lo
                # mismo (dónde se salió), sea por TP, señal, tiempo o stop —
                # el color ya distingue ganadora (verde) de perdedora (rojo)
                ax.plot([], [], color=VERDE, linestyle='--', linewidth=1.0,
                        label='Salida')

        # etiquetas IS / OOS sobre el eje
        ax.text(0.01, 0.97, 'IS', transform=ax.transAxes, color=AZUL,
                fontsize=9, fontweight='bold', va='top')
        ax.text(0.99, 0.97, 'OOS', transform=ax.transAxes, color=AMBAR,
                fontsize=9, fontweight='bold', va='top', ha='right')
        ax.set_ylabel('Precio', fontsize=8, color=AX_FG)

        # ── indicadores overlays (medias, Bollinger, KAMA, patrones) ──
        # (ind ya se calculó arriba, antes de construir los paneles)
        #
        # Cada familia se salta las capas que el usuario tenga apagadas en el
        # panel, y saca su color de _color_capa en vez de asignarlo aquí: el
        # cuadradito del panel y la línea del gráfico salen así del mismo sitio.
        # Apagar NO es ocultar el artista — _iniciar_sesion_blit devuelve a
        # visibles todos los dinámicos en cuanto se arrastra el gráfico, así que
        # la única forma de que una capa se quede apagada es no crearla.
        mas, bbs, patrones_set, kamas, sars, donchianes, zigzags = (
            ind['mas'], ind['bbs'], ind['patrones'], ind['kamas'], ind['sars'],
            ind['donchianes'], ind['zigzags'])
        for tipo, per in sorted(mas, key=lambda x: x[1]):
            clave = f'ma:{tipo}:{per}'
            if not self._capa_activa(clave):
                continue
            f = ema if tipo == 'EMA' else sma
            line, = ax.plot(x_idx, f(y, per), color=self._color_capa(clave),
                            linewidth=1.0, alpha=0.75)
            self._art_overlays_extra.append(line)
        for per, desv in sorted(bbs):
            if not self._capa_activa(f'bb:{per}:{desv:g}'):
                continue
            media, sup, inf = bollinger(y, per, desv)
            sup_line, = ax.plot(x_idx, sup, color=COLOR_BOLLINGER, linewidth=0.5, alpha=0.4)
            inf_line, = ax.plot(x_idx, inf, color=COLOR_BOLLINGER, linewidth=0.5, alpha=0.4)
            fill = ax.fill_between(x_idx, inf, sup, color=COLOR_BOLLINGER, alpha=0.05)
            self._art_overlays_extra += [sup_line, inf_line, fill]
        for per_er, rapido, lento in sorted(kamas):
            if not self._capa_activa(f'kama:{per_er}:{rapido}:{lento}'):
                continue
            val = _kama_serie(y, per_er, rapido, lento)
            line, = ax.plot(x_idx, val, color=COLOR_KAMA, linewidth=1.1, alpha=0.8)
            self._art_overlays_extra.append(line)
        for per in sorted(donchianes):
            if not self._capa_activa(f'donchian:{per}'):
                continue
            sup, inf = donchian(p.get('high', y), p.get('low', y), per)
            sup_line, = ax.plot(x_idx, sup, color=COLOR_DONCHIAN, linewidth=0.6, alpha=0.5)
            inf_line, = ax.plot(x_idx, inf, color=COLOR_DONCHIAN, linewidth=0.6, alpha=0.5)
            fill = ax.fill_between(x_idx, inf, sup, color=COLOR_DONCHIAN, alpha=0.04)
            self._art_overlays_extra += [sup_line, inf_line, fill]
        if zigzags:
            df_zz = pd.DataFrame({'high': p.get('high', y), 'low': p.get('low', y)})
            for desviacion, piernas in sorted(zigzags):
                if not self._capa_activa(f'zigzag:{desviacion:g}:{piernas}'):
                    continue
                pivotes = _zigzag_pivotes(df_zz, desviacion, piernas)
                # se dibuja en la vela en que OCURRIÓ el pivote, no en la de
                # confirmación: es donde el usuario lo ve en el gráfico, aunque
                # la estrategia solo pudiera reaccionar unas velas después
                xs = [i for i, _conf, _pr, _tp in pivotes if i < len(ts)]
                ys = [pr for i, _conf, pr, _tp in pivotes if i < len(ts)]
                if not xs:
                    continue
                line, = ax.plot(xs, ys, color=AZUL_ZIGZAG, linewidth=1.0, alpha=0.9,
                                marker='o', markersize=3)
                self._art_overlays_extra.append(line)
        for af_i, af_p, af_m in sorted(sars):
            if not self._capa_activa(f'sar:{af_i:g}:{af_p:g}:{af_m:g}'):
                continue
            sar_val, _tend = _sar_serie(p.get('high', y), p.get('low', y),
                                        af_i, af_p, af_m)
            sc = ax.scatter(x_idx, sar_val, s=3, color=COLOR_SAR, alpha=0.75)
            self._art_overlays_extra.append(sc)
        for per, mult in sorted(ind['supertrends']):
            if not self._capa_activa(f'st:{per}:{mult:g}'):
                continue
            st_val, _tend_st = _supertrend_serie(
                p.get('high', y), p.get('low', y), y, per, mult)
            line, = ax.plot(x_idx, st_val, color=COLOR_SUPERTREND,
                            linewidth=1.1, alpha=0.85)
            self._art_overlays_extra.append(line)
        for tenkan, kijun, senkou in sorted(ind['ichimokus']):
            if not self._capa_activa(f'ichi:{tenkan}:{kijun}:{senkou}'):
                continue
            t_v, k_v, sa, sb, _ch = _ichimoku_series(
                p.get('high', y), p.get('low', y), y, tenkan, kijun, senkou)
            for serie_v, color in ((t_v, COLOR_ICHIMOKU_TENKAN),
                                   (k_v, COLOR_ICHIMOKU_KIJUN),
                                   (sa, COLOR_ICHIMOKU_SENKOU),
                                   (sb, COLOR_ICHIMOKU_SENKOU)):
                line, = ax.plot(x_idx, serie_v, color=color, linewidth=0.8,
                                alpha=0.8)
                self._art_overlays_extra.append(line)
        for per, mult in sorted(ind['keltners']):
            if not self._capa_activa(f'kc:{per}:{mult:g}'):
                continue
            media_kc, sup_kc, inf_kc = _keltner_series(
                y, p.get('high', y), p.get('low', y), per, mult)
            for serie_v in (media_kc, sup_kc, inf_kc):
                line, = ax.plot(x_idx, serie_v, color=COLOR_KELTNER,
                                linewidth=0.8, alpha=0.75)
                self._art_overlays_extra.append(line)
        vol_v = p.get('volume')
        if vol_v is not None:
            df_v = pd.DataFrame({'timestamp': ts, 'high': p.get('high', y),
                                 'low': p.get('low', y), 'close': y,
                                 'volume': vol_v})
            for anclaje, k, modo in sorted(ind['vwaps']):
                clave = f'vwap:{anclaje}:{k:g}:{modo}'
                if not self._capa_activa(clave):
                    continue
                color_v = self._color_capa(clave)
                r = _vwap_series(df_v, anclaje, k, modo)
                l_media, = ax.plot(x_idx, r['media'], color=color_v,
                                   linewidth=1.2, alpha=0.9)
                l_sup, = ax.plot(x_idx, r['sup'], color=color_v,
                                 linewidth=0.5, alpha=0.45)
                l_inf, = ax.plot(x_idx, r['inf'], color=color_v,
                                 linewidth=0.5, alpha=0.45)
                fill = ax.fill_between(x_idx, r['inf'], r['sup'],
                                       color=color_v, alpha=0.12)
                self._art_overlays_extra += [l_media, l_sup, l_inf, fill]
        if patrones_set and self._capa_activa('patrones'):
            o_all, h_all, l_all = (p.get('open', y), p.get('high', y),
                                   p.get('low', y))
            detectados = detectar_patrones(o_all, h_all, l_all, y)
            offset_pat = (np.nanmax(h_all) - np.nanmin(l_all)) * 0.008
            for nombre in patrones_set:
                occ = detectados.get(nombre)
                if occ is None:
                    continue
                idx, dirs = occ['idx'], occ['dir']
                filtro = (idx >= 0) & (idx < len(ts))
                idx = idx[filtro]
                dirs = dirs[filtro]
                if len(idx) == 0:
                    continue
                color_pat = COLOR_PATRONES if np.mean(dirs) > 0 else ROJO
                sc = ax.scatter(idx, l_all[idx] - offset_pat,
                                marker='^' if np.mean(dirs) > 0 else 'v',
                                color=color_pat, s=8, alpha=0.7, zorder=5)
                self._art_overlays_extra.append(sc)

        # ── tramos de Fibonacci: por cada orden límite EJECUTADA, el swing del
        # que se midió su retroceso. El precio del nivel NO se redibuja aquí
        # porque es exactamente el segmento de la orden (abajo); lo que aporta
        # esta capa es de dónde sale ese precio.
        #
        # Solo las rellenadas: una orden se coloca en cuanto confirma la señal
        # primaria, y la mayoría se cancelan sin que el precio llegue a
        # retroceder al nivel. Dibujar también esas llenaba el gráfico del
        # swing de retrocesos que nunca ocurrieron. Las canceladas y expiradas
        # siguen viéndose como segmento en la capa de órdenes, que es donde
        # tienen sentido — ahí lo que se cuenta son las oportunidades perdidas.
        ol_fib = p['resultado'].get('ordenes_limite') or {}
        if (ind['fibs'] and len(ol_fib.get('idx_alta', ()))
                and self._capa_activa('fib')):
            df_fib = pd.DataFrame({'high': p.get('high', y), 'low': p.get('low', y)})
            cache_tramos = {}
            for k in range(len(ol_fib['idx_alta'])):
                if int(ol_fib['resultado'][k]) != ORDEN_RELLENADA:
                    continue
                sid = int(ol_fib['setup'][k])
                if sid not in ind['fibs']:
                    continue
                desv, piernas, nivel = ind['fibs'][sid]
                if (desv, piernas) not in cache_tramos:
                    cache_tramos[(desv, piernas)] = tramos_zigzag_vigentes(
                        df_fib, desv, piernas)
                tramos = cache_tramos[(desv, piernas)]
                i0, i1 = int(ol_fib['idx_alta'][k]), int(ol_fib['idx_fin'][k])
                if i0 >= len(ts) or i1 >= len(ts):
                    continue
                origen, extremo = tramos['anterior'][i0], tramos['ultimo'][i0]
                if not np.isfinite(origen) or not np.isfinite(extremo):
                    continue
                x0, x1 = i0, i1
                banda = ax.fill_between([x0, x1], origen, extremo,
                                        color=AMARILLO_FIB, alpha=0.05, zorder=1)
                self._art_overlays_extra.append(banda)
                for precio_ancla in (origen, extremo):
                    borde, = ax.plot([x0, x1], [precio_ancla] * 2,
                                     color=AMARILLO_FIB, linewidth=0.7, alpha=0.55,
                                     zorder=4)
                    self._art_overlays_extra.append(borde)

        # ── órdenes límite: un segmento horizontal de la vela en que se
        # colocan a la vela en que se resuelven, al precio pedido. Deja ver de
        # un vistazo cuántas oportunidades se escaparon sin rellenarse.
        ordenes = (p['resultado'].get('ordenes_limite') or {}).get('idx_alta')
        if ordenes is not None and len(ordenes) and self._capa_activa('ordenes'):
            ol = p['resultado']['ordenes_limite']
            colores_orden = {0: VERDE, 1: GRIS, 2: AMBAR}
            vistos = set()
            for k in range(len(ol['idx_alta'])):
                i0, i1 = int(ol['idx_alta'][k]), int(ol['idx_fin'][k])
                if i0 >= len(ts) or i1 >= len(ts):
                    continue
                res = int(ol['resultado'][k])
                color_o = colores_orden.get(res, GRIS)
                linea, = ax.plot([i0, i1],
                                 [ol['precio'][k], ol['precio'][k]],
                                 color=color_o, linewidth=1.0, linestyle=':',
                                 alpha=0.8, zorder=4,
                                 label=(f'Orden {RESULTADOS_ORDEN[res].lower()}'
                                        if res not in vistos else None))
                vistos.add(res)
                self._art_overlays_extra.append(linea)

        # leyenda de OPERACIONES (compra/venta/stop/salida/órdenes). Los
        # indicadores ya no ponen `label=`: los lista el panel de capas, que
        # además deja apagarlos. Aquí se quedan los elementos que el panel no
        # cubre — los tres colores según cómo se resolvió cada orden límite, o
        # los marcadores de operación, cuyas casillas siguen en la barra.
        ax.legend(loc='lower right', fontsize=7, facecolor=FIG_BG,
                  edgecolor=GRID_C, labelcolor=AX_FG, framealpha=0.6)

        self._etiquetar_decimables()
        ax.callbacks.connect('xlim_changed', self._on_xlim_changed)
        self._ax_principal = ax
        self.canvas.draw_idle()

    def _fecha_de_indice(self, i):
        """Índice de vela (float, posiblemente fuera de rango tras un pan) ->
        Timestamp. Los extremos se recortan al primer/último dato: el eje deja
        desplazarse más allá de la serie, pero ahí no hay fecha que enseñar."""
        ts = getattr(self, '_ts_full', None)
        if ts is None or len(ts) == 0:
            return None
        return ts[int(np.clip(round(i), 0, len(ts) - 1))]

    def _indice_de_fecha(self, t):
        """Timestamp -> índice de la primera vela que lo alcanza. Inversa de
        _fecha_de_indice para lo que entra por la UI en fechas (los QDateEdit
        de rango, las celdas de la tabla de trades)."""
        ts = getattr(self, '_ts_full', None)
        if ts is None or len(ts) == 0:
            return 0.0
        # searchsorted del propio DatetimeIndex, no de sus enteros crudos:
        # `asi8` va en la unidad del índice (ns, us... según la versión de
        # pandas) y compararlo con Timestamp.value caía siempre fuera de rango
        return float(np.clip(ts.searchsorted(pd.Timestamp(t)), 0, len(ts) - 1))

    def _sync_dateedits(self, i_ini, i_fin):
        d0 = self._fecha_de_indice(i_ini)
        d1 = self._fecha_de_indice(i_fin)
        if d0 is None or d1 is None:
            return
        self.fecha_ini.setDate(QDate(d0.year, d0.month, d0.day))
        self.fecha_fin.setDate(QDate(d1.year, d1.month, d1.day))

    @_no_crash
    def _aplicar_rango(self):
        if self._payload is None:
            return
        d0 = self.fecha_ini.date().toPyDate()
        d1 = self.fecha_fin.date().toPyDate()
        self._dibujar_principal(xlim=(self._indice_de_fecha(d0),
                                      self._indice_de_fecha(d1)))

    @_no_crash
    def _reset_rango(self):
        self._dibujar_principal()

    @_no_crash
    def _centrar_trade(self, fila, _col):
        if self._payload is None:
            return
        # la tabla puede estar ordenada: leer las celdas de la fila; puede
        # venir tanto de la tabla incrustada como de la del diálogo "Lista
        # completa" (self.sender() es quien emitió cellClicked)
        tabla = self.sender()
        if tabla is None:
            tabla = self.tabla_trades
        it_in, it_out = tabla.item(fila, 0), tabla.item(fila, 1)
        if it_in is None or it_out is None:
            return
        it_p_in, it_p_out = tabla.item(fila, 4), tabla.item(fila, 5)
        try:
            precios = ((float(it_p_in.text()), float(it_p_out.text()))
                       if it_p_in is not None and it_p_out is not None else None)
        except ValueError:
            precios = None
        self._centrar_en(pd.Timestamp(it_in.text()), pd.Timestamp(it_out.text()),
                         precios)

    def _centrar_en(self, entrada, salida, precios=None):
        """Reencuadra el gráfico principal sobre la ventana [entrada, salida],
        con un margen proporcional a la duración del trade, y ajusta el eje Y
        al recorrido de precio si se pasan (precio_entrada, precio_salida).

        `entrada`/`salida` llegan como Timestamps (salen de la tabla de trades);
        el margen se calcula ya en VELAS, que es la unidad del eje: así una
        operación abierta sobre un fin de semana no se lleva un margen inflado
        por unas horas que en el gráfico no ocupan nada."""
        if getattr(self, '_ts_full', None) is None:
            self._ts_full = pd.DatetimeIndex(self._payload['timestamps'])
        i_ent = self._indice_de_fecha(entrada)
        i_sal = self._indice_de_fecha(salida)
        margen_min = len(self._ts_full) / 40.0
        margen = max((i_sal - i_ent) * 0.4, margen_min)
        self._y_manual = False  # re-autoescalar el precio a esta ventana concreta
        self._dibujar_principal(xlim=(i_ent - margen, i_sal + margen))

        if precios is not None and self._ax_principal is not None:
            p_in, p_out = precios
            lo, hi = min(p_in, p_out), max(p_in, p_out)
            pad = (hi - lo) * 0.25 or (abs(p_in) * 0.02 or 1.0)
            self._ax_principal.set_ylim(lo - pad, hi + pad)
            self._y_manual = True
            self.canvas.draw_idle()

    def _llenar_tabla_trades(self, tabla, payload, orden=None):
        return render_tabla_trades(tabla, payload, orden)

    def _clave_orden_trades(self, payload, col):
        return clave_orden_trades(payload, col)

    @_no_crash
    def _ordenar_tabla_trades(self, tabla, col):
        if self._payload is None:
            return
        if getattr(tabla, '_sort_col', -1) == col:
            tabla._sort_order = (Qt.SortOrder.DescendingOrder
                                 if tabla._sort_order == Qt.SortOrder.AscendingOrder
                                 else Qt.SortOrder.AscendingOrder)
        else:
            tabla._sort_col = col
            tabla._sort_order = Qt.SortOrder.AscendingOrder
        keys = self._clave_orden_trades(self._payload, col)
        if tabla._sort_order == Qt.SortOrder.AscendingOrder:
            orden = np.argsort(keys, kind='stable')
        else:
            orden = np.argsort(keys, kind='stable')[::-1]
        tabla.horizontalHeader().setSortIndicator(col, tabla._sort_order)
        self._llenar_tabla_trades(tabla, self._payload, orden=orden)

    @_no_crash
    def _abrir_lista_completa(self):
        if self._dlg_trades is None:
            dlg, dlg_lay, lbl_sub, _halo = _panel_flotante_dialog(
                'Trades — lista completa', 900, alto=600,
                subtitulo='', parent=self, boton_cerrar=True)
            tabla = QTableWidget(0, 13)
            tabla.setHorizontalHeaderLabels(
                ['Entrada', 'Salida', 'Dir', 'Setup', 'P. entrada', 'P. salida',
                 'PnL', 'Motivo', 'MFE (R)', 'MAE (R)', 'ETD (R)', 'Ent. Ef %',
                 'Sal. Ef %'])
            tabla.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            tabla.verticalHeader().setVisible(False)
            tabla.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
            tabla.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
            tabla.setSortingEnabled(False)
            tabla._sort_col = -1
            tabla._sort_order = Qt.SortOrder.AscendingOrder
            tabla.horizontalHeader().setSortIndicatorShown(True)
            tabla.horizontalHeader().sectionClicked.connect(
                lambda c, t=tabla: self._ordenar_tabla_trades(t, c))
            tabla.cellClicked.connect(self._centrar_trade)
            dlg_lay.addWidget(tabla)
            self._dlg_trades = dlg
            self._dlg_trades.tabla = tabla
            self._dlg_trades.lbl_sub = lbl_sub
            self._llenar_tabla_trades(tabla, self._payload)
            self._actualizar_contador_trades()
        self._dlg_trades.show()
        self._dlg_trades.raise_()
        self._dlg_trades.activateWindow()

    def _actualizar_contador_trades(self):
        """Subtítulo de la ventana de lista completa: nº de operaciones."""
        dlg = self._dlg_trades
        if dlg is None or dlg.lbl_sub is None:
            return
        n = dlg.tabla.rowCount()
        dlg.lbl_sub.setText(f"{n:,} operaciones".replace(',', '.'))

    def _dibujar_equity(self, payload):
        bombear_eventos()
        return render_equity(self, payload)

    def _recolectar_indicadores(self, payload):
        """Devuelve un dict de conjuntos únicos (mas, bbs, rsis, atrs,
        patrones, stochs, williams, ccis, kamas) extraídos de todos los
        setups del sistema. Sin repeticiones."""
        mas = set()
        bbs = set()
        rsis = set()
        atrs = set()
        patrones_set = set()
        stochs = set()
        williams = set()
        ccis = set()
        kamas = set()
        sars = set()
        donchianes = set()
        zigzags = set()
        macds = set()
        adxs = set()
        supertrends = set()
        aroones = set()
        cmos = set()
        trixes = set()
        stochrsis = set()
        ichimokus = set()
        keltners = set()
        ttms = set()
        vwaps = set()
        # (periodo, metodo) del filtro de régimen de cada setup
        ers = set()
        hursts = set()
        # {indice_setup: (desviacion, piernas, nivel_fib)} — indexado POR SETUP
        # porque ordenes_limite['setup'] dice quién colocó cada orden y cada
        # setup puede medir su retroceso con un nivel distinto
        fibs = {}
        setups = payload.get('setups', []) if payload else []
        for indice_setup, setup in enumerate(setups):
            plantilla = setup.get('plantilla', '')
            p = params_por_defecto(plantilla) if plantilla in ESTRATEGIAS else {}
            p.update(setup.get('params', {}))

            if plantilla == 'Cruce de medias':
                t = 'EMA' if p.get('tipo') == 'EMA' else 'SMA'
                mas.add((t, p.get('rapida', 20)))
                mas.add((t, p.get('lenta', 50)))
            elif plantilla == 'Bollinger + ATR':
                bbs.add((p.get('periodo', 20), p.get('desv', 2.0)))
            elif plantilla == 'RSI':
                rsis.add(p.get('periodo', 14))
            elif plantilla == 'Stochastic (%K/%D)':
                stochs.add((p.get('periodo_k', 14), p.get('suavizado_k', 3),
                            p.get('periodo_d', 3), p.get('sobreventa', 20.0),
                            p.get('sobrecompra', 80.0)))
            elif plantilla == 'Williams %R':
                williams.add((p.get('periodo', 14), p.get('sobreventa', -80.0),
                              p.get('sobrecompra', -20.0)))
            elif plantilla == 'CCI':
                ccis.add((p.get('periodo', 20), p.get('sobreventa', -100.0),
                          p.get('sobrecompra', 100.0)))
            elif plantilla == 'KAMA':
                kamas.add((p.get('periodo_er', 10), p.get('rapido', 2), p.get('lento', 30)))
            elif plantilla == 'Parabolic SAR':
                sars.add((p.get('af_inicial', 0.02), p.get('af_paso', 0.02),
                          p.get('af_max', 0.2)))
            elif plantilla == 'Breakout de canal (Donchian)':
                donchianes.add(p.get('periodo', 20))
                per_salida = int(p.get('periodo_salida') or 0)
                if per_salida > 0:
                    donchianes.add(per_salida)
            elif plantilla == 'Patrones de velas':
                patrones_set.update(p.get('patrones', []))
            elif plantilla == 'Supertrend':
                supertrends.add((int(p.get('periodo', 10)),
                                 float(p.get('multiplicador', 3.0))))
            elif plantilla == 'MACD':
                macds.add((int(p.get('rapido', 12)), int(p.get('lento', 26)),
                           int(p.get('senal', 9))))
            elif plantilla == 'ADX (fuerza de tendencia)':
                adxs.add((int(p.get('periodo', 14)),
                          float(p.get('umbral_fuerza', 20.0))))
            elif plantilla == 'Aroon':
                aroones.add(int(p.get('periodo', 25)))
            elif plantilla == 'CMO':
                cmos.add(int(p.get('periodo', 14)))
            elif plantilla == 'TRIX':
                trixes.add(int(p.get('periodo', 15)))
            elif plantilla == 'StochRSI':
                stochrsis.add(int(p.get('periodo', 14)))
            elif plantilla == 'Ichimoku':
                ichimokus.add((int(p.get('tenkan', 9)), int(p.get('kijun', 26)),
                               int(p.get('senkou', 52))))
            elif plantilla == 'Keltner':
                keltners.add((int(p.get('periodo', 20)),
                              float(p.get('mult', 2.0))))
            elif plantilla == 'TTM Squeeze':
                ttms.add((int(p.get('periodo', 20)),
                          float(p.get('mult_bb', 2.0)),
                          float(p.get('mult_kc', 1.5))))
            elif plantilla == 'VWAP':
                vwaps.add((p.get('anclaje', 'W'), float(p.get('k', 2.0)),
                           p.get('modo', 'sd')))
            elif plantilla == 'Custom (reglas)':
                for clave in ('entradas_long', 'entradas_short',
                              'salidas_long', 'salidas_short'):
                    for grupo in p.get('reglas', {}).get(clave, []):
                        for cond in grupo.get('condiciones', []):
                            for lado in (cond.get('izq', {}), cond.get('der', {})):
                                _acumular_indicador_spec(
                                    lado, mas, rsis, atrs, bbs,
                                    donchianes, zigzags, macds, adxs,
                                    supertrends, aroones, cmos, trixes,
                                    stochrsis, ichimokus, keltners, ttms,
                                    vwaps)

            # filtros extra del setup (condiciones_entrada/condiciones_salida) —
            # aplicables a CUALQUIER plantilla, no solo Custom (reglas); por
            # eso van fuera del if/elif de arriba.
            # el ZigZag de un setup con entrada límite en Fibonacci: se dibuja
            # aunque su plantilla no tenga nada que ver con el ZigZag
            entrada = setup.get('entrada') or {}
            if entrada.get('tipo') == 'limite_fib':
                desv = entrada.get('zigzag_desviacion', 5.0)
                piernas = entrada.get('zigzag_piernas', 10)
                zigzags.add((desv, piernas))
                fibs[indice_setup] = (desv, piernas,
                                      entrada.get('nivel_fib', 0.618))

            filtros = setup.get('filtros') or {}
            for clave in ('condiciones_entrada', 'condiciones_salida'):
                for cond in filtros.get(clave, []):
                    for lado in (cond.get('izq', {}), cond.get('der', {})):
                        _acumular_indicador_spec(
                            lado, mas, rsis, atrs, bbs,
                            donchianes, zigzags, macds, adxs,
                            supertrends, aroones, cmos, trixes, stochrsis,
                            ichimokus, keltners, ttms, vwaps)

            # el régimen NO viene de la plantilla sino del filtro del setup: se
            # dibuja para poder comprobar que las entradas caen donde el filtro
            # dice que caen, que es la única forma de auditarlo desde la GUI
            reg = filtros.get('regimen') or {}
            metodo_reg = reg.get('metodo', 'ninguno')
            if metodo_reg in METODOS_REGIMEN_ER + METODOS_REGIMEN_HURST:
                periodo_reg = int(reg.get('periodo')
                                  or ventana_regimen_defecto(metodo_reg))
                destino = (ers if metodo_reg in METODOS_REGIMEN_ER else hursts)
                destino.add((periodo_reg, metodo_reg))
        return {
            'mas': mas, 'bbs': bbs, 'rsis': rsis, 'atrs': atrs,
            'patrones': patrones_set, 'stochs': stochs, 'williams': williams,
            'ccis': ccis, 'kamas': kamas, 'sars': sars,
            'donchianes': donchianes, 'zigzags': zigzags, 'fibs': fibs,
            'ers': ers, 'hursts': hursts,
            'macds': macds, 'adxs': adxs, 'supertrends': supertrends,
            'aroones': aroones, 'cmos': cmos, 'trixes': trixes,
            'stochrsis': stochrsis, 'ichimokus': ichimokus,
            'keltners': keltners, 'ttms': ttms, 'vwaps': vwaps,
        }

    def _dibujar_panel_oscilador(self, ax, kind, datos, x, c, h, l):
        """Dibuja el contenido de un panel de oscilador (RSI/ATR/Estocástico/
        Williams %R/CCI) en el Axes ya posicionado `ax`. `datos` es el
        conjunto de tuplas de parámetros recolectado por
        _recolectar_indicadores para ese tipo. Devuelve la lista de artistas
        de datos (líneas/umbrales/relleno) creados, para que el llamador los
        registre como "dinámicos" de cara al blitting (ver
        _artistas_dinamicos): comparten eje X con el precio, así que su
        ventana visible cambia con cada pan/zoom igual que las velas.

        `x` es el índice de vela, no la fecha: el eje es el mismo (sharex) que
        el del precio, que dejó de ser temporal para no dejar huecos de fin de
        semana (ver _dibujar_principal)."""
        _style_ax(ax)
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position('right')
        artes = []
        # (valor, color) de cada línea de referencia del panel: se escriben en
        # el eje Y al final (ver _marcar_niveles_eje)
        niveles = []
        if kind == 'rsi':
            pal = ['#ffffff', '#e67e22', '#fd79a8']
            for i, per in enumerate(sorted(datos)):
                line, = ax.plot(x, rsi(c, per), color=pal[i % len(pal)],
                                linewidth=1.0, alpha=0.85, label=f'RSI({per})')
                artes.append(line)
            artes.append(ax.axhline(70, color=ROJO, linewidth=0.5, linestyle='--', alpha=0.4))
            artes.append(ax.axhline(50, color=GRIS, linewidth=0.5, linestyle='--', alpha=0.3))
            artes.append(ax.axhline(30, color=VERDE, linewidth=0.5, linestyle='--', alpha=0.4))
            niveles += [(70, ROJO), (50, GRIS), (30, VERDE)]
            ax.set_ylim(0, 100)
        elif kind == 'atr':
            pal = ['#2ecc71', '#1abc9c']
            for i, per in enumerate(sorted(datos)):
                val = atr(h, l, c, per)
                color = pal[i % len(pal)]
                line, = ax.plot(x, val, color=color, linewidth=0.9, alpha=0.85,
                                label=f'ATR({per})')
                fill = ax.fill_between(x, 0, val, color=color, alpha=0.08)
                artes += [line, fill]
        elif kind == 'stoch':
            pal_k = ['#26c6da', '#4fc3f7', '#80deea']
            pal_d = ['#f06292', '#ec407a', '#f8bbd0']
            for i, (per_k, suav_k, per_d, sobreventa, sobrecompra) in enumerate(sorted(datos)):
                k, d = stochastic(h, l, c, per_k, suav_k, per_d)
                lk, = ax.plot(x, k, color=pal_k[i % len(pal_k)], linewidth=1.0,
                              alpha=0.9, label=f'%K({per_k},{suav_k})')
                ld, = ax.plot(x, d, color=pal_d[i % len(pal_d)], linewidth=0.9,
                              alpha=0.75, label=f'%D({per_d})')
                h_sc = ax.axhline(sobrecompra, color=ROJO, linewidth=0.5, linestyle='--', alpha=0.4)
                h_sv = ax.axhline(sobreventa, color=VERDE, linewidth=0.5, linestyle='--', alpha=0.4)
                artes += [lk, ld, h_sc, h_sv]
                niveles += [(sobrecompra, ROJO), (sobreventa, VERDE)]
            ax.set_ylim(0, 100)
        elif kind == 'williams':
            pal = ['#ec407a', '#f06292', '#f8bbd0']
            for i, (per, sobreventa, sobrecompra) in enumerate(sorted(datos)):
                val = williams_r(h, l, c, per)
                line, = ax.plot(x, val, color=pal[i % len(pal)], linewidth=1.0,
                                alpha=0.85, label=f'%R({per})')
                h_sc = ax.axhline(sobrecompra, color=ROJO, linewidth=0.5, linestyle='--', alpha=0.4)
                h_sv = ax.axhline(sobreventa, color=VERDE, linewidth=0.5, linestyle='--', alpha=0.4)
                artes += [line, h_sc, h_sv]
                niveles += [(sobrecompra, ROJO), (sobreventa, VERDE)]
            ax.set_ylim(-100, 0)
        elif kind == 'cci':
            pal = ['#5c6bc0', '#7986cb', '#9fa8da']
            for i, (per, sobreventa, sobrecompra) in enumerate(sorted(datos)):
                val = cci(h, l, c, per)
                line, = ax.plot(x, val, color=pal[i % len(pal)], linewidth=1.0,
                                alpha=0.85, label=f'CCI({per})')
                h_sc = ax.axhline(sobrecompra, color=ROJO, linewidth=0.5, linestyle='--', alpha=0.4)
                h_0 = ax.axhline(0, color=GRIS, linewidth=0.5, linestyle='--', alpha=0.3)
                h_sv = ax.axhline(sobreventa, color=VERDE, linewidth=0.5, linestyle='--', alpha=0.4)
                artes += [line, h_sc, h_0, h_sv]
                niveles += [(sobrecompra, ROJO), (0, GRIS), (sobreventa, VERDE)]
        elif kind == 'macd':
            for i, (rapido, lento, senal) in enumerate(sorted(datos)):
                linea, senal_l, hist = _macd_series(c, rapido, lento, senal)
                l_linea, = ax.plot(x, linea, color='#4fc3f7', linewidth=1.0,
                                   alpha=0.9, label=f'MACD({rapido},{lento},{senal})')
                l_senal, = ax.plot(x, senal_l, color='#f1c40f', linewidth=0.9,
                                   alpha=0.85, label='Señal')
                colores = np.where(np.isfinite(hist) & (hist >= 0), VERDE, ROJO)
                barras = ax.bar(x, hist, width=0.8, color=colores, alpha=0.4)
                h_0 = ax.axhline(0, color=GRIS, linewidth=0.5, linestyle='--', alpha=0.3)
                artes += [l_linea, l_senal, barras, h_0]
                niveles += [(0, GRIS)]
        elif kind == 'adx':
            for i, (per, umbral) in enumerate(sorted(datos)):
                adx, pdi, mdi = _adx_series(h, l, c, per)
                l_adx, = ax.plot(x, adx, color='#7986cb', linewidth=1.0, alpha=0.9,
                                 label=f'ADX({per})')
                l_pdi, = ax.plot(x, pdi, color='#2ecc71', linewidth=0.8, alpha=0.8,
                                 label='+DI')
                l_mdi, = ax.plot(x, mdi, color='#e74c3c', linewidth=0.8, alpha=0.8,
                                 label='−DI')
                h_umb = ax.axhline(umbral, color=AMBAR, linewidth=0.5,
                                   linestyle='--', alpha=0.5)
                artes += [l_adx, l_pdi, l_mdi, h_umb]
                niveles += [(umbral, AMBAR)]
            ax.set_ylim(0, 100)
        elif kind == 'aroon':
            pal = ['#1abc9c', '#16a085']
            for i, per in enumerate(sorted(datos)):
                up, dn = _aroon_series(h, l, per)
                l_up, = ax.plot(x, up, color=pal[0], linewidth=1.0, alpha=0.9,
                                label=f'Aroon Up({per})')
                l_dn, = ax.plot(x, dn, color=pal[1], linewidth=0.9, alpha=0.8,
                                label=f'Aroon Down({per})')
                h_70 = ax.axhline(70, color=ROJO, linewidth=0.5, linestyle='--', alpha=0.4)
                h_30 = ax.axhline(30, color=VERDE, linewidth=0.5, linestyle='--', alpha=0.4)
                artes += [l_up, l_dn, h_70, h_30]
                niveles += [(70, ROJO), (30, VERDE)]
            ax.set_ylim(0, 100)
        elif kind == 'cmo':
            pal = ['#ffa726', '#ffb74d']
            for i, per in enumerate(sorted(datos)):
                val = _cmo_serie(c, per)
                line, = ax.plot(x, val, color=pal[i % len(pal)], linewidth=1.0,
                                alpha=0.85, label=f'CMO({per})')
                h_0 = ax.axhline(0, color=GRIS, linewidth=0.5, linestyle='--', alpha=0.3)
                artes += [line, h_0]
                niveles += [(0, GRIS)]
            ax.set_ylim(-100, 100)
        elif kind == 'trix':
            pal = ['#ff7043', '#ff8a65']
            for i, per in enumerate(sorted(datos)):
                val = _trix_serie(c, per)
                line, = ax.plot(x, val, color=pal[i % len(pal)], linewidth=1.0,
                                alpha=0.85, label=f'TRIX({per})')
                h_0 = ax.axhline(0, color=GRIS, linewidth=0.5, linestyle='--', alpha=0.3)
                artes += [line, h_0]
                niveles += [(0, GRIS)]
        elif kind == 'stochrsi':
            pal = ['#fd79a8', '#f8bbd0']
            for i, per in enumerate(sorted(datos)):
                k, d = _stochrsi_series(c, per)
                l_k, = ax.plot(x, k, color=pal[0], linewidth=1.0, alpha=0.9,
                               label=f'StochRSI %K({per})')
                l_d, = ax.plot(x, d, color=pal[1], linewidth=0.9, alpha=0.8,
                               label='%D')
                h_80 = ax.axhline(0.8, color=ROJO, linewidth=0.5, linestyle='--', alpha=0.4)
                h_20 = ax.axhline(0.2, color=VERDE, linewidth=0.5, linestyle='--', alpha=0.4)
                artes += [l_k, l_d, h_80, h_20]
                niveles += [(0.8, ROJO), (0.2, VERDE)]
            ax.set_ylim(0, 1)
        elif kind == 'ttm':
            for i, (per, mult_bb, mult_kc) in enumerate(sorted(datos)):
                sq, mom = _ttm_squeeze_series(c, h, l, per, mult_bb, mult_kc)
                sq_f = np.where(np.isfinite(mom), np.asarray(sq, dtype=float), np.nan)
                # squeeze como pasos de fondo (1 = comprimido)
                step = ax.step(x, sq_f, where='post', color=GRIS, linewidth=1.0,
                               alpha=0.5, label=f'Squeeze({per})')
                colores = np.where(np.isfinite(mom) & (mom >= 0), VERDE, ROJO)
                barras = ax.bar(x, mom, width=0.8, color=colores, alpha=0.5)
                h_0 = ax.axhline(0, color=GRIS, linewidth=0.5, linestyle='--', alpha=0.3)
                artes += [step, barras, h_0]
                niveles += [(0, GRIS)]
        elif kind == 'vwap':
            # distancia del cierre al VWAP en desviaciones: (close−media)/σ
            payload = getattr(self, '_payload', None)
            vol_v = (payload or {}).get('volume')
            ts_v = (pd.DatetimeIndex(payload['timestamps'])
                    if payload and payload.get('timestamps') is not None
                    else None)
            if ts_v is None or vol_v is None:
                return artes
            df_v = pd.DataFrame({'timestamp': ts_v, 'high': h, 'low': l,
                                 'close': c, 'volume': vol_v})
            for i, (anclaje, k, modo) in enumerate(sorted(datos)):
                r = _vwap_series(df_v, anclaje, k, modo)
                sd_pos = np.where(r['sd'] > 0, r['sd'], np.nan)
                dist = (c - r['media']) / sd_pos
                etiqueta = (f"VWAP({_MAPA_ANCLAJE_INV.get(anclaje, anclaje)}, "
                            f"{k:g}{'σ' if modo == 'sd' else '%'})")
                color_v = _PALETA_VWAP[i % len(_PALETA_VWAP)]
                line, = ax.plot(x, dist, color=color_v, linewidth=1.0,
                                alpha=0.9, label=etiqueta)
                artes.append(line)
            for ref in (1, 2, 3):
                h_pos = ax.axhline(ref, color=ROJO, linewidth=0.5,
                                   linestyle='--', alpha=0.3)
                h_neg = ax.axhline(-ref, color=VERDE, linewidth=0.5,
                                   linestyle='--', alpha=0.3)
                artes += [h_pos, h_neg]
                niveles += [(ref, ROJO), (-ref, VERDE)]
            h_0 = ax.axhline(0, color=GRIS, linewidth=0.5, linestyle='--', alpha=0.4)
            artes.append(h_0)
            niveles += [(0, GRIS)]
            ax.set_ylim(-4, 4)
        elif kind in ('er', 'hurst'):
            artes_reg, niveles_reg = self._dibujar_panel_regimen(ax, kind, datos, x, c)
            artes += artes_reg
            niveles += niveles_reg
        ax.set_ylabel(ETIQUETA_PANEL.get(kind, kind), fontsize=8, color=AX_FG)
        ax.legend(fontsize=6, framealpha=0.15, labelcolor=AX_FG, loc='upper left')
        # escala que el usuario haya estirado a mano en este tipo de panel
        # (ver _ylim_paneles): va DESPUÉS del set_ylim de cada rama, que es su
        # valor por defecto
        ylim_manual = self._ylim_paneles.get(kind)
        if ylim_manual:
            ax.set_ylim(*ylim_manual)
            # con la escala estirada a mano los niveles pueden caer fuera de la
            # vista, y el eje se quedaría SIN una sola etiqueta: ahí manda el
            # localizador automático
            niveles = []
        _marcar_niveles_eje(ax, niveles)
        return artes

    def _dibujar_panel_regimen(self, ax, kind, datos, x, c):
        """Panel del filtro de régimen: la serie de ER o de Hurst con sus
        umbrales y la banda que el filtro admite sombreada.

        Usa `_er_serie` / `_hurst_serie` de core.strategies, las MISMAS
        funciones que aplica el filtro — no una reimplementación. Así, si las
        entradas no caen dentro de la banda sombreada, el problema está en el
        filtro y no en el dibujo.

        Devuelve (artistas, niveles) — los niveles son los umbrales, que el
        llamador escribe en el eje Y (ver _marcar_niveles_eje)."""
        artes = []
        es_er = kind == 'er'
        pal = ['#ff9800', '#ffb74d'] if es_er else ['#ab47bc', '#ce93d8']
        for i, (periodo, metodo) in enumerate(sorted(datos)):
            if es_er:
                val = _er_serie(c, periodo).values
            else:
                val = _hurst_serie(c, periodo)
                if val is None:
                    # ventana mayor que el histórico: el filtro tampoco se
                    # aplica (ver _mascara_filtros_setup), y decirlo aquí es
                    # más útil que un panel en blanco
                    artes.append(ax.text(
                        0.5, 0.5, f'Hurst({periodo}) no calculable: la ventana '
                        f'no cabe en {len(c):,} velas',
                        transform=ax.transAxes, ha='center', va='center',
                        color=AMBAR, fontsize=7))
                    continue
            etiqueta = f"{'ER' if es_er else 'Hurst'}({periodo})"
            banda = BANDA_REGIMEN.get(metodo)
            if banda is not None:
                lo, hi = banda
                admitidas = ((val > lo) if hi >= 1.0 else (val < hi))
                etiqueta += f' — admite {admitidas.mean() * 100:.1f}% de las velas'
                artes.append(ax.axhspan(lo, hi, color=VERDE, alpha=0.07, zorder=0))
            line, = ax.plot(x, val, color=pal[i % len(pal)], linewidth=1.0,
                            alpha=0.9, label=etiqueta)
            artes.append(line)
        umbrales = ((UMBRAL_ER_TENDENCIA, UMBRAL_ER_RUIDO) if es_er
                    else (UMBRAL_HURST_TENDENCIA, UMBRAL_HURST_REVERSION))
        artes.append(ax.axhline(umbrales[0], color=VERDE, linewidth=0.5,
                                linestyle='--', alpha=0.5))
        artes.append(ax.axhline(umbrales[1], color=ROJO, linewidth=0.5,
                                linestyle='--', alpha=0.5))
        if es_er:
            ax.set_ylim(0, 1)
        else:
            ax.set_ylim(0.3, 0.9)
        return artes, [(umbrales[0], VERDE), (umbrales[1], ROJO)]


    def _dibujar_wfa(self, wfa, ts, equity=None):
        if not wfa:
            self.grp_wfa.setVisible(False)
            return
        bombear_eventos()
        self.grp_wfa.setVisible(True)
        self.fig_wfa.clear()
        ax = self.fig_wfa.add_subplot(111)
        _style_ax(ax)

        rets = [w['retorno_pct'] or 0.0 for w in wfa]
        dds = [w.get('max_dd_pct', 0.0) or 0.0 for w in wfa]
        wrs = [w['win_rate'] * 100 if w['win_rate'] is not None else 0.0 for w in wfa]
        trades = [w['n_trades'] or 0 for w in wfa]

        # posiciones/anchos reales (en días, vía date2num) de cada ventana —
        # así las barras ocupan el tiempo real que duró su tramo OOS en vez
        # de un ancho categórico fijo, y el eje X puede mostrar fechas.
        n_ts = len(ts)
        t0s = np.array([date2num(ts[w['idx_ini']]) for w in wfa])
        t1s = np.array([date2num(ts[min(w['idx_fin'], n_ts - 1)]) for w in wfa])
        widths = np.maximum(t1s - t0s, 1e-3)
        xpos = t0s + widths / 2
        x = xpos

        modo = getattr(self, 'combo_wfa_modo', None)
        idx = modo.currentIndex() if modo is not None else 0

        bars = None
        if idx == 0:  # Retorno %
            colores = [VERDE if r > 0 else ROJO for r in rets]
            bars = ax.bar(xpos, rets, width=widths, color=colores, alpha=0.85,
                          edgecolor=FIG_BG, linewidth=1.3)
            ax.axhline(0, color=GRIS, linewidth=0.7, linestyle='--')
            ax.set_ylabel('Retorno %', fontsize=8, color=AX_FG)
            _data_vals = rets
        elif idx == 1:  # Retorno acumulado
            acum = np.cumsum(rets)
            ax.plot(xpos, acum, color=AZUL, linewidth=1.5, marker='o',
                    markersize=5, markerfacecolor=AZUL, markeredgecolor='#111')
            ax.axhline(0, color=GRIS, linewidth=0.7, linestyle='--')
            ax.fill_between(xpos, 0, acum, where=np.array(acum) >= 0,
                            color=VERDE, alpha=0.15)
            ax.fill_between(xpos, 0, acum, where=np.array(acum) < 0,
                            color=ROJO, alpha=0.15)
            ax.set_ylabel('Retorno acumulado %', fontsize=8, color=AX_FG)
            _data_vals = acum
        elif idx == 2:  # Max DD %
            dds_abs = [abs(d) for d in dds]
            colores = [ROJO if dds_abs[i] > np.median(dds_abs) else AMBAR for i in range(len(dds_abs))]
            bars = ax.bar(xpos, dds_abs, width=widths, color=colores, alpha=0.85,
                          edgecolor=FIG_BG, linewidth=1.3)
            ax.set_ylabel('Max DD % (abs)', fontsize=8, color=AX_FG)
            _data_vals = dds_abs
        elif idx == 3:  # Win rate %
            colores = [VERDE if w > 50 else ROJO for w in wrs]
            bars = ax.bar(xpos, wrs, width=widths, color=colores, alpha=0.85,
                          edgecolor=FIG_BG, linewidth=1.3)
            ax.axhline(50, color=GRIS, linewidth=0.7, linestyle='--')
            ax.set_ylim(0, 100)
            ax.set_ylabel('Win rate %', fontsize=8, color=AX_FG)
            _data_vals = wrs
        elif idx == 4:  # Retorno vs Max DD
            bars_ret = ax.bar(xpos - widths * 0.2, rets, widths * 0.35,
                              color=AZUL, alpha=0.7, label='Retorno %',
                              edgecolor=FIG_BG, linewidth=1.3)
            dds_abs = [abs(d) for d in dds]
            ax2 = ax.twinx()
            bars_dd = ax2.bar(xpos + widths * 0.2, dds_abs, widths * 0.35 * 0.8,
                              color=ROJO, alpha=0.5, label='Max DD % (abs)',
                              edgecolor=FIG_BG, linewidth=1.3)
            ax.set_ylabel('Retorno %', fontsize=8, color=AZUL)
            ax2.set_ylabel('Max DD % (abs)', fontsize=8, color=ROJO)
            ax2.tick_params(colors=ROJO, labelsize=7)
            for spine in ax2.spines.values():
                spine.set_edgecolor(GRID_C)
            ax.tick_params(colors=AZUL, labelsize=7)
            _data_vals = [max(r, d) for r, d in zip(rets, dds_abs)]
            bars = None
        elif idx == 5:  # Trades
            colores = [VERDE if t > np.median(trades) else AMBAR for t in trades]
            bars = ax.bar(xpos, trades, width=widths, color=colores, alpha=0.85,
                          edgecolor=FIG_BG, linewidth=1.3)
            ax.set_ylabel('Nº Trades', fontsize=8, color=AX_FG)
            _data_vals = trades
        elif idx == 6:  # Curva de Equidad Combinada (OOS)
            _data_vals = None
            eq = np.asarray(equity) if equity is not None else np.array([])
            if eq.size:
                y = (eq / eq[0] - 1.0) * 100.0
                ax.plot(ts, y, color=AZUL, linewidth=1.1)
                ax.axhline(0, color=GRIS, linewidth=0.7, linestyle='--')
                for w in wfa[:-1]:
                    ax.axvline(ts[min(w['idx_fin'], n_ts - 1)], color=GRIS,
                               linewidth=0.8, linestyle=':', alpha=0.6)
                ylim0 = ax.get_ylim()
                for i, w in enumerate(wfa):
                    t_mid = ts[w['idx_ini']] + (
                        ts[min(w['idx_fin'], n_ts - 1)] - ts[w['idx_ini']]) / 2
                    ax.text(t_mid, ylim0[1], f'V{i + 1}', color=GRIS,
                            fontsize=6.5, ha='center', va='top')
            ax.set_ylabel('Retorno acumulado %', fontsize=8, color=AX_FG)

        # ----- etiquetas de valor sobre/bajo cada barra -----
        if bars is not None and _data_vals is not None:
            max_abs = max(abs(v) for v in _data_vals) if _data_vals else 1
            offset = max_abs * 0.03 if max_abs > 0 else 1
            for i, (bar, v) in enumerate(zip(bars, _data_vals)):
                y_bar = bar.get_height()
                if idx == 2:
                    y_lbl = y_bar
                    va = 'bottom'
                    color_tag = ROJO
                    val_str = f'{v:.2f}%'
                elif idx == 3:
                    y_lbl = y_bar + offset
                    va = 'bottom'
                    color_tag = VERDE if wrs[i] > 50 else ROJO
                    val_str = f'{v:.1f}%'
                elif idx == 5:
                    y_lbl = y_bar + offset
                    va = 'bottom'
                    color_tag = VERDE if trades[i] > np.median(trades) else AMBAR
                    val_str = str(v)
                else:
                    va = 'bottom' if v >= 0 else 'top'
                    y_lbl = y_bar + offset if v >= 0 else y_bar - offset
                    color_tag = VERDE if v > 0 else ROJO
                    val_str = f'{v:+.2f}%'
                ax.text(bar.get_x() + bar.get_width() / 2, y_lbl,
                        val_str, ha='center', va=va,
                        fontsize=6.5, color=color_tag, fontweight='bold')

        # ----- lineas de max/min ----- (solo Retorno % y Max DD %; en Max DD
        # % únicamente el máximo, que es el peor caso — no tiene sentido
        # destacar un "mínimo" en verde ahí. Win rate % y Trades sin marcar.
        if idx in (0, 2) and _data_vals:
            mx = max(_data_vals)
            mi = min(_data_vals)
            if mx != mi:
                fmt = {0: lambda v: f'{v:+.2f}%', 2: lambda v: f'{v:.2f}%'}[idx]
                # etiqueta como "tag" de color pegado al propio eje Y (blend:
                # x en fracción de ejes, y en coordenadas de datos), en vez de
                # texto flotando dentro del panel
                trans = ax.get_yaxis_transform()
                lbl_kw = dict(fontsize=6, va='center', ha='right',
                              transform=trans, color=FIG_BG, fontweight='bold')
                if idx == 2:   # Max DD %: solo el máximo (peor caso), en rojo
                    ax.axhline(mx, color=ROJO, linewidth=0.6, linestyle=':',
                               alpha=0.5)
                    ax.text(-0.01, mx, fmt(mx),
                            bbox=dict(facecolor=ROJO, edgecolor='none',
                                      boxstyle='round,pad=0.25'), **lbl_kw)
                else:          # Retorno %: máximo en verde, mínimo en rojo si es negativo
                    ax.axhline(mx, color=VERDE, linewidth=0.6, linestyle=':',
                               alpha=0.5)
                    ax.text(-0.01, mx, fmt(mx),
                            bbox=dict(facecolor=VERDE, edgecolor='none',
                                      boxstyle='round,pad=0.25'), **lbl_kw)
                    if mi < 0:
                        ax.axhline(mi, color=ROJO, linewidth=0.6, linestyle=':',
                                   alpha=0.5)
                        ax.text(-0.01, mi, fmt(mi),
                                bbox=dict(facecolor=ROJO, edgecolor='none',
                                          boxstyle='round,pad=0.25'), **lbl_kw)

        ax.set_xlabel('Periodo', fontsize=8, color=AX_FG)
        ax.xaxis_date()
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=8))
        ax.xaxis.set_major_formatter(
            mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
        ax.tick_params(axis='x', labelsize=6.5)

        # ----- hover tooltip -----
        annot = ax.annotate("", xy=(0, 0), xytext=(15, 15),
                            textcoords="offset points",
                            bbox=dict(boxstyle="round,pad=0.4", fc=FIG_BG, ec=GRID_C,
                                      alpha=0.95),
                            color=AX_FG, fontsize=7, zorder=99,
                            annotation_clip=False)
        annot.set_visible(False)

        def _hover_wfa(event):
            if event.inaxes != ax:
                if annot.get_visible():
                    annot.set_visible(False)
                    self.canvas_wfa.draw_idle()
                return
            # En el borde del panel matplotlib puede devolver xdata/ydata None
            # aunque el cursor siga dentro del eje (p. ej. justo sobre la línea
            # y=0 cuando la barra de una ventana es invisible): se recuperan
            # las coordenadas de datos por transformación inversa.
            if event.xdata is None or event.ydata is None:
                try:
                    xd, yd = ax.transData.inverted().transform((event.x, event.y))
                except Exception:
                    xd = yd = None
            else:
                xd, yd = event.xdata, event.ydata
            if xd is None or not (ax.get_xlim()[0] <= xd <= ax.get_xlim()[1]):
                if annot.get_visible():
                    annot.set_visible(False)
                    self.canvas_wfa.draw_idle()
                return
            hit = False
            for i in range(len(x)):
                if bars is not None:
                    bar_x = bars[i].get_x() + bars[i].get_width() / 2
                else:
                    bar_x = x[i]
                # El hover se activa por la extensión temporal de la ventana,
                # no por el rectángulo de la barra: si el retorno es ~0 la
                # barra no se ve y antes no se podía abrir el tooltip.
                contains = t0s[i] <= xd <= t1s[i]
                if contains:
                    periodo = (f"{ts[wfa[i]['idx_ini']].strftime('%d/%m/%y')} → "
                               f"{ts[min(wfa[i]['idx_fin'], len(ts)) - 1].strftime('%d/%m/%y')}")
                    lines = [
                        f"Ventana {i + 1}",
                        periodo,
                        f"Retorno: {rets[i]:+.2f}%",
                        f"Max DD: {dds[i]:.2f}%",
                        f"Win rate: {wrs[i]:.0f}%",
                        f"Trades: {trades[i]}",
                    ]
                    # espacio real disponible en píxeles dentro del panel —
                    # más fiable que un umbral sobre el % del rango de datos
                    # en un panel tan bajo (fig_wfa mide 2.2")
                    renderer = self.canvas_wfa.get_renderer()
                    bb = ax.get_window_extent(renderer=renderer)
                    x_disp, y_disp = ax.transData.transform((xd, yd))
                    dx, ha = ((15, 'left') if (bb.x1 - x_disp) >= (x_disp - bb.x0)
                              else (-15, 'right'))
                    dy, va = ((15, 'bottom') if (bb.y1 - y_disp) >= (y_disp - bb.y0)
                              else (-15, 'top'))
                    annot.xy = (xd, yd)
                    annot.set_position((dx, dy))
                    annot.set_ha(ha)
                    annot.set_va(va)
                    annot.set_text('\n'.join(lines))
                    annot.set_visible(True)
                    hit = True
                    break
            if not hit and annot.get_visible():
                annot.set_visible(False)
                self.canvas_wfa.draw_idle()
            elif hit:
                self.canvas_wfa.draw_idle()

        try:
            self.canvas_wfa.mpl_disconnect(self.__dict__.get('_wfa_hover_cid', ''))
        except Exception:
            pass
        self._wfa_hover_cid = self.canvas_wfa.mpl_connect("motion_notify_event",
                                                           _hover_wfa)

        try:
            self.fig_wfa.tight_layout(pad=0.6)
        except Exception:
            pass
        self.canvas_wfa.draw_idle()

        self.tabla_wfa.setRowCount(len(wfa))
        for r, w in enumerate(wfa):
            periodo = (f"{ts[w['idx_ini']].strftime('%Y-%m-%d')} → "
                       f"{ts[min(w['idx_fin'], len(ts)) - 1].strftime('%Y-%m-%d')}")
            wr = _fmt(w['win_rate'] * 100 if w['win_rate'] is not None else None, 1, ' %')
            vals = [str(r + 1), periodo, str(w['n_trades']), wr,
                    _fmt(w['retorno_pct'], 2, ' %'), _fmt(w['max_dd_pct'], 2, ' %')]
            for c_i, v in enumerate(vals):
                it = QTableWidgetItem(v)
                it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if c_i == 4 and w['retorno_pct'] is not None:
                    it.setForeground(QColor(VERDE if w['retorno_pct'] > 0 else ROJO))
                self.tabla_wfa.setItem(r, c_i, it)

        # la tabla crece para mostrar todas las ventanas de una vez — el
        # scroll de la pestaña (self._scroll) es el que baja hasta ellas
        alto_filas = sum(self.tabla_wfa.rowHeight(r) for r in range(self.tabla_wfa.rowCount()))
        alto_total = (self.tabla_wfa.horizontalHeader().height() + alto_filas
                      + 2 * self.tabla_wfa.frameWidth())
        self.tabla_wfa.setFixedHeight(alto_total)

    def _dibujar_mc(self, mc, capital, max_dd_base=None, retorno_base=None):
        bombear_eventos()
        return render_montecarlo(self, mc, capital, max_dd_base, retorno_base)

    def _dibujar_mfe_mae(self, payload):
        bombear_eventos()
        return render_mfe_mae(self, payload)


# ══════════════ sub-pestaña Optimizador (comparativa de combinaciones) ══════════════
_ROL_CURVA = Qt.ItemDataRole.UserRole
_ROL_INDICE = Qt.ItemDataRole.UserRole + 1
_ROL_VALOR = Qt.ItemDataRole.UserRole + 2

_COLS_METRICAS_COMBO = [
    ('n_trades', 'Trades', 0, ''),
    ('retorno_pct', 'Retorno', 2, ' %'),
    ('sharpe', 'Sharpe', 2, ''),
    ('sharpe_smart', 'Sharpe smart', 2, ''),
    ('sortino', 'Sortino', 2, ''),
    ('calmar', 'Calmar', 2, ''),
    ('serenity_index', 'Serenity', 2, ''),
    ('profit_factor', 'Profit factor', 2, ''),
    ('max_dd_pct', 'Max DD', 2, ' %'),
    ('win_rate', 'Win rate', 1, ' %'),
    ('sqn', 'SQN', 2, ''),
]

# presentación de los campos de riesgo barridos: la clave interna lleva el
# prefijo anticolisión de core/optimizer (_riesgo.) — aquí se muestra con la
# MISMA etiqueta que el campo del editor de setup del Constructor, y el riesgo
# escalado a % (internamente viaja como fracción)
_ETIQUETAS_RIESGO = {
    PREFIJO_RIESGO + 'riesgo_pct': ('Riesgo del setup (%)', 100.0, ' %'),
    PREFIJO_RIESGO + 'stop_atr': ('Stop Loss (× ATR)', 1.0, ''),
    PREFIJO_RIESGO + 'tp_r': ('Take Profit (R)', 1.0, ''),
    PREFIJO_RIESGO + 'salida_n_velas': ('Salida por tiempo (velas)', 1.0, ''),
}


def _etiqueta_param(clave):
    """Nombre legible de una clave barrida (los params de estrategia ya son
    legibles; los de riesgo llevan prefijo interno)."""
    return _ETIQUETAS_RIESGO.get(clave, (clave,))[0]


def _texto_param(clave, v):
    if v is None:
        return '—'
    _, escala, sufijo = _ETIQUETAS_RIESGO.get(clave, (clave, 1.0, ''))
    return f"{v * escala:g}{sufijo}"


class _ItemNumerico(QTableWidgetItem):
    """QTableWidgetItem que ordena por el valor numérico guardado en
    _ROL_VALOR en vez de comparar el texto mostrado (evita que el orden por
    columna sea alfabético con números: '10' < '9')."""

    def __lt__(self, other):
        a, b = self.data(_ROL_VALOR), other.data(_ROL_VALOR)
        try:
            return float(a) < float(b)
        except (TypeError, ValueError):
            return super().__lt__(other)


class _SparklineDelegate(QStyledItemDelegate):
    """Pinta la mini-curva de equity IS guardada en _ROL_CURVA directamente
    con QPainter, en vez de crear una figura matplotlib por fila — así
    aguanta miles de combinaciones porque Qt solo invoca paint() en las
    celdas visibles."""

    def paint(self, painter, option, index):
        curva = index.data(_ROL_CURVA)
        if curva is None or len(curva) < 2:
            super().paint(painter, option, index)
            return
        painter.save()
        if option.state & QStyle.StateFlag.State_Selected:
            painter.fillRect(option.rect, option.palette.highlight())
        rect = option.rect.adjusted(4, 4, -4, -4)
        vmin, vmax = float(np.min(curva)), float(np.max(curva))
        rango = (vmax - vmin) or 1.0
        n = len(curva)
        xs = [rect.left() + i * rect.width() / (n - 1) for i in range(n)]
        ys = [rect.bottom() - (v - vmin) / rango * rect.height() for v in curva]
        color = QColor(VERDE) if curva[-1] >= curva[0] else QColor(ROJO)
        pen = painter.pen()
        pen.setColor(color)
        pen.setWidthF(1.3)
        painter.setPen(pen)
        painter.setRenderHint(painter.RenderHint.Antialiasing)
        for i in range(n - 1):
            painter.drawLine(int(xs[i]), int(ys[i]), int(xs[i + 1]), int(ys[i + 1]))
        painter.restore()

    def sizeHint(self, option, index):
        return QSize(110, 30)


def _texto_metrica(v, clave, dec, sufijo):
    if v is None:
        return '—'
    if clave == 'win_rate':
        return _fmt(v * 100, dec, sufijo)
    if dec == 0:
        return str(int(v))
    return _fmt(v, dec, sufijo)


class ComparativaWidget(QWidget):
    """Resultado del barrido de parámetros (core.optimizer.optimizar_setup):
    scatter grande (vista principal, retorno vs riesgo de cada combinación)
    + tabla compacta con sparkline de la equity IS, enlazados por selección.
    Puramente una vista — no relanza nada por sí misma; TabBacktest la puebla
    tras cada _OptimizerThread."""
    usar_configuracion = pyqtSignal(dict)   # setup elegido -> OptimizadorWidget
    agregar_setups = pyqtSignal(list)       # varios setups -> sistema del Constructor

    _EJES = [('Retorno %', 'retorno_pct'), ('Max DD %', 'max_dd_pct'),
             ('Sharpe', 'sharpe'), ('Profit factor', 'profit_factor')]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._resultados = []
        self._metrica = 'sharpe'
        self._fila_seleccionada = None      # índice principal en self._resultados
        self._filas_seleccionadas = []      # todos los índices seleccionados
        self._actualizando_seleccion = False
        self._scatter = None

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        self.lbl_activo = QLabel("Ningún activo seleccionado")
        self.lbl_activo.setObjectName("titulo")
        root.addWidget(self.lbl_activo)

        self.lbl_resumen = QLabel(
            "Lanza «🔍 Prueba de parametrización (Solo IS)» desde la pestaña Constructor")
        self.lbl_resumen.setObjectName("titulo")
        root.addWidget(self.lbl_resumen)

        self.progreso = ProgressBarAnimada()
        self.progreso.setRange(0, 100)
        self.progreso.setTextVisible(True)
        self.progreso.setFixedHeight(10)
        self.progreso.setVisible(False)
        root.addWidget(self.progreso)

        # ── scatter grande: vista principal ──
        grp_scatter = QGroupBox("Retorno vs riesgo de cada combinación (tramo IS)")
        lay_sc = QVBoxLayout(grp_scatter)
        fila_ejes = QHBoxLayout()
        fila_ejes.addWidget(QLabel("Vista:"))
        self.cmb_vista = QComboBox()
        self.cmb_vista.addItem("Scatter (foto final)")
        self.cmb_vista.addItem("Evolución temporal (Sharpe acumulado)")
        self.cmb_vista.setToolTip(
            "Scatter: cada punto es una combinación con sus métricas finales "
            "del tramo IS.\nEvolución temporal: curva del Sharpe ACUMULADO a "
            "lo largo del IS de las combinaciones seleccionadas (o el top 10) "
            "— distingue un Sharpe que creció estable de uno fabricado por un "
            "único tramo afortunado")
        self.cmb_vista.currentIndexChanged.connect(self._on_vista_changed)
        fila_ejes.addWidget(self.cmb_vista)
        fila_ejes.addSpacing(12)
        fila_ejes.addWidget(QLabel("Eje X:"))
        self.cmb_eje_x = QComboBox()
        fila_ejes.addWidget(self.cmb_eje_x)
        fila_ejes.addWidget(QLabel("Eje Y:"))
        self.cmb_eje_y = QComboBox()
        fila_ejes.addWidget(self.cmb_eje_y)
        for etiqueta, clave in self._EJES:
            self.cmb_eje_x.addItem(etiqueta, clave)
            self.cmb_eje_y.addItem(etiqueta, clave)
        self.cmb_eje_x.setCurrentIndex(0)   # Retorno %
        self.cmb_eje_y.setCurrentIndex(1)   # Max DD %
        self.cmb_eje_x.currentIndexChanged.connect(self._redibujar)
        self.cmb_eje_y.currentIndexChanged.connect(self._redibujar)
        fila_ejes.addStretch()
        lay_sc.addLayout(fila_ejes)

        self.fig_scatter = Figure(figsize=(6, 4), facecolor=FIG_BG)
        self.canvas_scatter = FigureCanvasQTAgg(self.fig_scatter)
        self.canvas_scatter.setMinimumHeight(340)
        self.canvas_scatter.mpl_connect('pick_event', self._on_pick_scatter)
        lay_sc.addWidget(self.canvas_scatter, 1)
        root.addWidget(grp_scatter, 3)
        bombear_eventos()

        # ── tabla compacta con sparkline ──
        grp_tabla = QGroupBox("Todas las combinaciones probadas")
        lay_t = QVBoxLayout(grp_tabla)
        self.tabla = QTableWidget()
        self.tabla.setAlternatingRowColors(True)
        self.tabla.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.tabla.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
        self.tabla.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.tabla.itemSelectionChanged.connect(self._on_seleccion_tabla)
        lay_t.addWidget(self.tabla)
        root.addWidget(grp_tabla, 2)

        fila_btn = QHBoxLayout()
        self.lbl_seleccion = QLabel("")
        self.lbl_seleccion.setObjectName("estado")
        fila_btn.addWidget(self.lbl_seleccion, 1)
        self.btn_stats = QPushButton("📊 Estadísticas del conjunto")
        self.btn_stats.setEnabled(False)
        self.btn_stats.setToolTip(
            "Evalúa la salud del BARRIDO completo, no de cada combinación: "
            "% de configs rentables, qué parámetro decide que ganen o "
            "pierdan, si la mejor es una meseta robusta o un pico aislado "
            "(sobreoptimización) y cuánta confianza estadística hay. Si la "
            "mayoría pierde, conviene replantear activo, indicador o "
            "temporalidad en vez de quedarse con la ganadora")
        self.btn_stats.clicked.connect(self._abrir_estadisticas)
        fila_btn.addWidget(self.btn_stats)
        self.btn_agregar = QPushButton("Añadir seleccionadas como setups")
        self.btn_agregar.setEnabled(False)
        self.btn_agregar.setToolTip(
            "Añade las combinaciones seleccionadas (Ctrl/Shift + clic para "
            "elegir varias) al sistema del Constructor como setups que operan "
            "JUNTOS en cartera — si dos disparan en la misma vela, tiene "
            "prioridad el primero de la lista. No son alternativas a "
            "comparar: forman un único sistema conjunto")
        self.btn_agregar.clicked.connect(self._agregar_seleccionadas)
        fila_btn.addWidget(self.btn_agregar)
        self.btn_usar = QPushButton("Usar esta configuración")
        self.btn_usar.setEnabled(False)
        self.btn_usar.setToolTip(
            "Sustituye el setup seleccionado del Constructor por esta única "
            "combinación (requiere una sola fila seleccionada)")
        self.btn_usar.clicked.connect(self._usar_configuracion)
        fila_btn.addWidget(self.btn_usar)
        root.addLayout(fila_btn)
        bombear_eventos()

    # ── progreso durante el barrido ──
    def actualizar_progreso(self, i, total):
        self.progreso.setVisible(True)
        self.progreso.setMaximum(max(total, 1))
        self.progreso.setValue(i)

    # ── resultado del hilo ──
    @_no_crash
    def mostrar(self, payload):
        self.progreso.setVisible(False)
        self._resultados = payload['resultados']
        self._metrica = payload['metrica']
        self._fila_seleccionada = None
        self._filas_seleccionadas = []
        badge = _titulo_activo_html(payload.get('csv', ''), payload.get('tf'))
        self.lbl_activo.setTextFormat(Qt.TextFormat.RichText)
        self.lbl_activo.setText(badge)
        etiqueta_metrica = dict(DialogoOptimizacion.METRICAS).get(
            self._metrica, self._metrica)
        self.lbl_resumen.setText(
            f"{len(self._resultados)} combinaciones probadas sobre IS — "
            f"rankeadas por {etiqueta_metrica} (la primera fila es la mejor)")
        self.btn_stats.setEnabled(bool(self._resultados))
        self._poblar_tabla()
        bombear_eventos()
        self._redibujar()
        bombear_eventos()
        if self._resultados:
            self._seleccionar_resultado(0)

    def _claves_barridas(self):
        claves = set()
        for r in self._resultados:
            claves.update(r['params_barridos'])
        return sorted(claves)

    def _poblar_tabla(self):
        self.tabla.setSortingEnabled(False)
        claves = self._claves_barridas()
        n_cols = len(claves) + len(_COLS_METRICAS_COMBO) + 1
        self.tabla.setColumnCount(n_cols)
        etiquetas = ([_etiqueta_param(c) for c in claves]
                     + [e for _, e, _, _ in _COLS_METRICAS_COMBO] + ['Equity IS'])
        self.tabla.setHorizontalHeaderLabels(etiquetas)
        self.tabla.setItemDelegateForColumn(n_cols - 1, _SparklineDelegate(self.tabla))
        self.tabla.setRowCount(len(self._resultados))
        for fila, r in enumerate(self._resultados):
            if fila and fila % 400 == 0:
                bombear_eventos()
            col = 0
            for clave in claves:
                v = r['params_barridos'].get(clave)
                it = _ItemNumerico(_texto_param(clave, v))
                it.setData(_ROL_VALOR, v if v is not None else float('-inf'))
                it.setData(_ROL_INDICE, fila)
                it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.tabla.setItem(fila, col, it)
                col += 1
            for clave, _etq, dec, sufijo in _COLS_METRICAS_COMBO:
                v = r['metricas'].get(clave)
                it = _ItemNumerico(_texto_metrica(v, clave, dec, sufijo))
                it.setData(_ROL_VALOR, v if v is not None else float('-inf'))
                it.setData(_ROL_INDICE, fila)
                it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if clave in ('retorno_pct', 'sharpe', 'sqn',
                             'sharpe_smart', 'serenity_index',
                             'sortino', 'calmar', 'omega') and v is not None:
                    it.setForeground(QColor(VERDE if v > 0 else ROJO))
                self.tabla.setItem(fila, col, it)
                col += 1
            it_spark = _ItemNumerico('')
            it_spark.setData(_ROL_CURVA, r['equity_sparkline'])
            it_spark.setData(_ROL_VALOR, r['equity_sparkline'][-1] if len(r['equity_sparkline']) else 0.0)
            it_spark.setData(_ROL_INDICE, fila)
            self.tabla.setItem(fila, col, it_spark)
        self.tabla.resizeColumnsToContents()
        self.tabla.setSortingEnabled(True)

    # ── gráfico principal: scatter o evolución temporal ──
    @_no_crash
    def _on_vista_changed(self, *_):
        evolucion = self.cmb_vista.currentIndex() == 1
        self.cmb_eje_x.setEnabled(not evolucion)
        self.cmb_eje_y.setEnabled(not evolucion)
        self._redibujar()

    def _redibujar(self, *_):
        if self.cmb_vista.currentIndex() == 1:
            self._dibujar_evolucion()
        else:
            self._dibujar_scatter()

    def _dibujar_evolucion(self):
        """Curvas del Sharpe acumulado (ventana expansiva) a lo largo del
        tramo IS: las combinaciones seleccionadas, o el top 10 del ranking si
        no hay selección. Un Sharpe que crece estable es más fiable que el
        mismo valor final alcanzado por un único tramo afortunado."""
        self.fig_scatter.clear()
        ax = self.fig_scatter.add_subplot(111)
        _style_ax(ax)
        if not self._resultados:
            self.canvas_scatter.draw_idle()
            return
        indices = self._filas_seleccionadas or list(
            range(min(10, len(self._resultados))))
        indices = [i for i in indices
                   if len(self._resultados[i].get('sharpe_sparkline', [])) > 1]
        if not indices:
            ax.text(0.5, 0.5, "Relanza la prueba de parametrización para "
                    "ver la evolución del Sharpe",
                    transform=ax.transAxes, ha='center', va='center',
                    color=GRIS, fontsize=9)
            self.canvas_scatter.draw_idle()
            return

        import matplotlib.cm as cm
        mejor = min(indices)   # menor índice = mejor ranking (lista ordenada)
        colores = cm.viridis(np.linspace(0.15, 0.9, len(indices)))
        for orden, idx in enumerate(indices):
            r = self._resultados[idx]
            curva = np.asarray(r['sharpe_sparkline'], dtype=np.float64)
            x = np.linspace(0, 100, len(curva))
            es_mejor = idx == mejor
            etiqueta = self._nombre_combo(r)
            if len(etiqueta) > 40:
                etiqueta = etiqueta[:37] + '…'
            ax.plot(x, curva,
                    color=AMBAR if es_mejor else colores[orden],
                    linewidth=2.0 if es_mejor else 1.1,
                    alpha=1.0 if es_mejor else 0.85,
                    zorder=5 if es_mejor else 2,
                    label=etiqueta)
        ax.axhline(0, color=GRIS, linewidth=0.7, linestyle='--')
        ax.set_xlabel('Progreso del tramo IS (%)', color=AX_FG, fontsize=8)
        ax.set_ylabel('Sharpe acumulado', color=AX_FG, fontsize=8)
        if len(indices) <= 10:
            leg = ax.legend(fontsize=7, loc='best', framealpha=0.3,
                            facecolor=FIG_BG, edgecolor=GRID_C)
            for txt in leg.get_texts():
                txt.set_color(AX_FG)
        try:
            self.fig_scatter.tight_layout(pad=0.6)
        except Exception:
            pass
        self.canvas_scatter.draw_idle()

    def _dibujar_scatter(self, *_):
        self.fig_scatter.clear()
        ax = self.fig_scatter.add_subplot(111)
        _style_ax(ax)
        if not self._resultados:
            self.canvas_scatter.draw_idle()
            return
        clave_x = self.cmb_eje_x.currentData()
        clave_y = self.cmb_eje_y.currentData()
        xs = np.array([r['metricas'].get(clave_x) for r in self._resultados], dtype=np.float64)
        ys = np.array([r['metricas'].get(clave_y) for r in self._resultados], dtype=np.float64)
        colores = np.array([r['metricas'].get(self._metrica) or 0.0 for r in self._resultados])
        self._scatter = ax.scatter(xs, ys, c=colores, cmap='viridis', s=45,
                                   alpha=0.85, edgecolors=GRID_C, linewidths=0.4,
                                   picker=True, pickradius=6)
        ax.set_xlabel(self.cmb_eje_x.currentText(), color=AX_FG, fontsize=8)
        ax.set_ylabel(self.cmb_eje_y.currentText(), color=AX_FG, fontsize=8)
        for i in self._filas_seleccionadas:
            if 0 <= i < len(xs) and np.isfinite(xs[i]) and np.isfinite(ys[i]):
                ax.scatter([xs[i]], [ys[i]], s=160, facecolors='none',
                          edgecolors=AMBAR, linewidths=1.8, zorder=5)
        try:
            cb = self.fig_scatter.colorbar(self._scatter, ax=ax)
            cb.ax.tick_params(colors=AX_FG, labelsize=7)
            cb.set_label(dict(DialogoOptimizacion.METRICAS).get(self._metrica, self._metrica),
                        color=AX_FG, fontsize=7)
        except Exception:
            pass
        try:
            self.fig_scatter.tight_layout(pad=0.6)
        except Exception:
            pass
        self.canvas_scatter.draw_idle()

    def _on_pick_scatter(self, event):
        if not len(event.ind):
            return
        self._seleccionar_resultado(int(event.ind[0]))

    # ── selección (scatter <-> tabla) ──
    def _seleccionar_resultado(self, idx):
        if not (0 <= idx < len(self._resultados)):
            return
        self._actualizando_seleccion = True
        try:
            for fila in range(self.tabla.rowCount()):
                it = self.tabla.item(fila, 0)
                if it is not None and it.data(_ROL_INDICE) == idx:
                    self.tabla.selectRow(fila)
                    break
        finally:
            self._actualizando_seleccion = False
        self._aplicar_seleccion([idx])

    def _aplicar_seleccion(self, indices):
        """Estado común tras cambiar la selección (desde tabla o scatter)."""
        self._filas_seleccionadas = indices
        self._fila_seleccionada = indices[0] if indices else None
        n = len(indices)
        self.btn_usar.setEnabled(n == 1)
        self.btn_agregar.setEnabled(n >= 1)
        if n == 0:
            self.lbl_seleccion.setText("")
        elif n == 1:
            r = self._resultados[indices[0]]
            params_txt = ', '.join(f"{_etiqueta_param(k)}={_texto_param(k, v)}"
                                   for k, v in r['params_barridos'].items())
            self.lbl_seleccion.setText(
                f"Seleccionado: {params_txt or '(config. actual)'}")
        else:
            self.lbl_seleccion.setText(f"{n} combinaciones seleccionadas")
        self._redibujar()

    @_no_crash
    def _on_seleccion_tabla(self):
        if self._actualizando_seleccion:
            return
        indices = []
        for mi in self.tabla.selectionModel().selectedRows():
            it = self.tabla.item(mi.row(), 0)
            if it is not None and it.data(_ROL_INDICE) is not None:
                indices.append(int(it.data(_ROL_INDICE)))
        self._aplicar_seleccion(indices)

    def _nombre_combo(self, r):
        """Nombre descriptivo de una combinación para usarla como setup."""
        base = r['setup'].get('plantilla', 'Setup')
        params_txt = ', '.join(f"{_etiqueta_param(k)}={_texto_param(k, v)}"
                               for k, v in r['params_barridos'].items())
        return f"{base} · {params_txt}" if params_txt else base

    @_no_crash
    def _usar_configuracion(self):
        if self._fila_seleccionada is None:
            return
        setup = self._resultados[self._fila_seleccionada]['setup']
        self.usar_configuracion.emit(dict(setup, params=dict(setup['params'])))

    @_no_crash
    def _agregar_seleccionadas(self):
        if not self._filas_seleccionadas:
            return
        setups = []
        for idx in self._filas_seleccionadas:
            r = self._resultados[idx]
            setups.append(dict(r['setup'], params=dict(r['setup']['params']),
                               nombre=self._nombre_combo(r)))
        self.agregar_setups.emit(setups)

    # ── estadísticas del conjunto ──
    @staticmethod
    def _tabla_stats(filas):
        """QTableWidget métrica/valor: filas = [(nombre, texto, color|None,
        tooltip), ...]."""
        t = QTableWidget(len(filas), 2)
        t.setHorizontalHeaderLabels(['Métrica', 'Valor'])
        t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        t.verticalHeader().setVisible(False)
        t.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        t.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        t.setAlternatingRowColors(True)
        for f, (nombre, texto, color, tooltip) in enumerate(filas):
            it_n = QTableWidgetItem(nombre)
            if tooltip:
                it_n.setToolTip(tooltip)
            t.setItem(f, 0, it_n)
            it_v = QTableWidgetItem(texto)
            it_v.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if tooltip:
                it_v.setToolTip(tooltip)
            if color:
                it_v.setForeground(QColor(color))
            t.setItem(f, 1, it_v)
        t.setFixedHeight(t.verticalHeader().defaultSectionSize() * len(filas)
                        + t.horizontalHeader().height() + 6)
        return t

    @staticmethod
    def _color_pct(v, umbral=50.0):
        if v is None:
            return None
        return VERDE if v >= umbral else ROJO

    @_no_crash
    def _abrir_estadisticas(self):
        if not self._resultados:
            return
        est = estadisticas_conjunto(self._resultados)
        por_param = analisis_por_parametro(self._resultados)
        vec = analisis_vecindad(self._resultados)
        fia = fiabilidad_estadistica(self._resultados)

        dlg = QDialog(self)
        dlg.setWindowTitle("Estadísticas del conjunto de combinaciones")
        dlg.resize(640, 720)
        lay_dlg = QVBoxLayout(dlg)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        cont = QWidget()
        lay = QVBoxLayout(cont)
        lay.setSpacing(10)

        # ── salud global ──
        grp1 = QGroupBox("Salud global del conjunto")
        l1 = QVBoxLayout(grp1)
        n_txt = str(est['n_combos'])
        if est['n_sin_trades']:
            n_txt += f" ({est['n_sin_trades']} sin trades, excluidas de los %)"
        filas1 = [
            ('Combinaciones probadas', n_txt, None,
             "Las combinaciones que nunca operaron no cuentan como "
             "perdedoras: se excluyen de los porcentajes"),
            ('% rentables', _fmt(est['pct_rentables'], 1, ' %'),
             self._color_pct(est['pct_rentables']),
             "Combinaciones con retorno positivo en IS. Orientativo: >60% "
             "prometedor; <40% considera descartar este activo/indicador/"
             "temporalidad"),
            ('Retorno medio', _fmt(est['retorno_medio'], 2, ' %'),
             VERDE if (est['retorno_medio'] or 0) > 0 else ROJO,
             "Si la media es positiva pero la mediana negativa, el conjunto "
             "está volcado: unas pocas configs afortunadas arrastran la media"),
            ('Retorno mediana', _fmt(est['retorno_mediana'], 2, ' %'),
             VERDE if (est['retorno_mediana'] or 0) > 0 else ROJO,
             "La mitad de las combinaciones rinde menos que esto — más "
             "robusta que la media frente a outliers"),
            ('% profit factor > 1', _fmt(est['pct_pf_mayor_1'], 1, ' %'),
             self._color_pct(est['pct_pf_mayor_1']),
             "Confirmación cruzada: combos que ganan más de lo que pierden"),
            ('% Sharpe > 0', _fmt(est['pct_sharpe_pos'], 1, ' %'),
             self._color_pct(est['pct_sharpe_pos']),
             "Confirmación cruzada ajustada a volatilidad"),
            ('Sesgo de la distribución', _fmt(est['sesgo'], 2), None,
             "Positivo: cola de ganadoras (pocas configs muy buenas). "
             "Negativo: cola de perdedoras (riesgo de desplome escondido)"),
        ]
        l1.addWidget(self._tabla_stats(filas1))
        lay.addWidget(grp1)

        # ── culpable por parámetro ──
        grp2 = QGroupBox("¿Qué parámetro marca la diferencia?")
        l2 = QVBoxLayout(grp2)
        if not por_param:
            aviso = QLabel("(barre al menos un parámetro con varios valores "
                           "para ver este análisis)")
            aviso.setObjectName("estado")
            l2.addWidget(aviso)
        else:
            orden = sorted(por_param.items(), key=lambda kv: kv[1]['impacto'],
                           reverse=True)
            for clave, datos in orden:
                lbl = QLabel(f"«{_etiqueta_param(clave)}» — impacto: "
                             f"{datos['impacto']:.0f} puntos")
                lbl.setObjectName("campo")
                lbl.setToolTip(
                    "Impacto = diferencia de % rentables entre el mejor y el "
                    "peor valor de este parámetro. El de mayor impacto es el "
                    "principal causante de que las combinaciones ganen o "
                    "pierdan — si es un parámetro de riesgo, la señal puede "
                    "ser buena y la gestión el problema (o al revés)")
                l2.addWidget(lbl)
                t = QTableWidget(len(datos['valores']), 4)
                t.setHorizontalHeaderLabels(
                    ['Valor', 'Combos', '% rentables', 'Retorno mediana'])
                t.horizontalHeader().setSectionResizeMode(
                    QHeaderView.ResizeMode.Stretch)
                t.verticalHeader().setVisible(False)
                t.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
                t.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
                t.setAlternatingRowColors(True)
                for f, (valor, d) in enumerate(datos['valores']):
                    vals = [_texto_param(clave, valor), str(d['n']),
                            _fmt(d['pct_rentables'], 1, ' %'),
                            _fmt(d['retorno_mediana'], 2, ' %')]
                    for c, txt in enumerate(vals):
                        it = QTableWidgetItem(txt)
                        it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                        if c == 2:
                            color = self._color_pct(d['pct_rentables'])
                            if color:
                                it.setForeground(QColor(color))
                        t.setItem(f, c, it)
                t.setFixedHeight(
                    t.verticalHeader().defaultSectionSize() * len(datos['valores'])
                    + t.horizontalHeader().height() + 6)
                l2.addWidget(t)
        lay.addWidget(grp2)

        # ── vecindad y meseta ──
        grp3 = QGroupBox("Vecindad y meseta (anti-sobreoptimización)")
        l3 = QVBoxLayout(grp3)
        filas3 = [
            ('Rugosidad de la superficie', _fmt(vec['rugosidad'], 2, ' %'),
             None,
             "Cambio medio de retorno entre configs VECINAS (±1 paso en un "
             "parámetro). Bajo = meseta robusta; alto = el resultado depende "
             "del ajuste fino (ruido)"),
            ('Vecinas rentables del top 10', _fmt(vec['plateau_top'], 1, ' %'),
             self._color_pct(vec['plateau_top']),
             "¿Las mejores configs están rodeadas de vecinas también "
             "rentables? Si la élite son picos aislados entre perdedoras, es "
             "sobreoptimización, no edge"),
            ('Mayor zona ganadora contigua', _fmt(vec['pct_mayor_cluster'], 1, ' %'),
             self._color_pct(vec['pct_mayor_cluster'], 30.0),
             "Un edge real forma una REGIÓN conectada de configs rentables "
             "en el espacio de parámetros, no islas dispersas"),
        ]
        l3.addWidget(self._tabla_stats(filas3))
        lay.addWidget(grp3)

        # ── fiabilidad estadística ──
        grp4 = QGroupBox("Fiabilidad estadística")
        l4 = QVBoxLayout(grp4)
        filas4 = [
            (f"% con SQN ≥ {fia['min_sqn']:g}",
             _fmt(fia['pct_sqn_suficiente'], 1, ' %'),
             self._color_pct(fia['pct_sqn_suficiente']),
             "SQN = √n · media(r_multiple) / desv(r_multiple) — en vez de "
             "exigir un nº fijo de trades (injusto entre estilos: un "
             "position trading puede tardar años en juntar 30, un scalper "
             "los hace en días), el SQN compensa pocos trades si son muy "
             "consistentes y exige más si son ruidosos. Es, matemáticamente, "
             "un t-estadístico de la consistencia de la muestra"),
            ('Mediana de SQN', _fmt(fia['mediana_sqn'], 2), None,
             "SQN de la combinación mediana del conjunto (>2 bueno, >3 "
             "excelente, criterio de Van Tharp)"),
            ('Mediana de trades por combo', _fmt(fia['mediana_trades'], 0),
             None,
             "Dato informativo (no es el criterio de fiabilidad): cuántos "
             "trades genera típicamente una combinación en el tramo IS"),
            ('Ratio mejor/mediana', _fmt(fia['ratio_mejor_mediana'], 1, '×'),
             None,
             "Cuántas veces gana la mejor config respecto a la mediana. Muy "
             "alto = sospecha de pico afortunado. (Solo se calcula con "
             "mediana positiva)"),
            ('Concentración del top 5%', _fmt(fia['concentracion_top5'], 1, ' %'),
             None,
             "% del beneficio agregado que aportan las 5% mejores configs — "
             "muy alto = el conjunto vive de unas pocas combinaciones"),
        ]
        l4.addWidget(self._tabla_stats(filas4))
        lay.addWidget(grp4)

        lay.addStretch()
        scroll.setWidget(cont)
        lay_dlg.addWidget(scroll)
        btn_cerrar = QPushButton("Cerrar")
        btn_cerrar.clicked.connect(dlg.accept)
        lay_dlg.addWidget(btn_cerrar)
        dlg.exec()


# ══════════════ pestaña contenedora ══════════════
class TabBacktest(QWidget):
    """Tres sub-pestañas con responsabilidades separadas:
    Constructor (configurar) -> 🔍 Prueba de parametrización (Solo IS) -> Optimizador
    (comparar combinaciones, elegir una) -> ▶ Ejecutar backtest ->
    Resultados (detalle completo IS+OOS+WFA+MC de esa única configuración).
    Ver core/optimizer.py y el plan en docs internos para el razonamiento."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(STYLE_BACKTEST.replace('__CHEVRON__', _CHEVRON_SVG))
        self._threads = []
        # Comparativa (Optimizador) y Resultados crean figuras de matplotlib:
        # construirlas al abrir la app alargaba el arranque/entrada y la primera
        # vez inicializa backend+fuentes (varios segundos en equipos lentos).
        # Se construyen perezosamente al mostrar su sub-pestaña por primera vez.
        self._comparativa_widget = None
        self._resultados_widget = None

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.constructor = OptimizadorWidget()
        self._ph_comparativa = QWidget()
        self._ph_resultados = QWidget()
        self.tabs.addTab(self.constructor, "  Constructor  ")
        self.tabs.addTab(self._ph_comparativa, "  Optimizador  ")
        self.tabs.addTab(self._ph_resultados, "  Resultados  ")
        root.addWidget(self.tabs)

        self.constructor.ejecutar.connect(self._run_backtest)
        self.constructor.optimizar.connect(self._abrir_dialogo_optimizacion)
        self.tabs.currentChanged.connect(self._on_subtab_changed)

    @property
    def comparativa(self):
        if self._comparativa_widget is None:
            self._construir_comparativa()
        return self._comparativa_widget

    @property
    def resultados(self):
        if self._resultados_widget is None:
            self._construir_resultados()
        return self._resultados_widget

    def _construir_comparativa(self):
        w = ComparativaWidget()
        self._comparativa_widget = w
        idx = self.tabs.indexOf(self._ph_comparativa)
        if idx >= 0:
            self.tabs.removeTab(idx)
            self.tabs.insertTab(idx, w, "  Optimizador  ")
            self._ph_comparativa.deleteLater()
            # al quitar/insertar, la pestaña actual puede desplazarse: volver a
            # seleccionar la que se acaba de construir
            if self.tabs.currentIndex() != idx:
                self.tabs.setCurrentIndex(idx)
        w.usar_configuracion.connect(self._usar_configuracion)
        w.agregar_setups.connect(self._agregar_setups)

    def _construir_resultados(self):
        w = ResultadosWidget()
        self._resultados_widget = w
        idx = self.tabs.indexOf(self._ph_resultados)
        if idx >= 0:
            self.tabs.removeTab(idx)
            self.tabs.insertTab(idx, w, "  Resultados  ")
            self._ph_resultados.deleteLater()
            if self.tabs.currentIndex() != idx:
                self.tabs.setCurrentIndex(idx)
        w.favorito_guardado.connect(self.constructor._recargar_favoritos)

    def _on_subtab_changed(self, idx):
        # La primera visita a Optimizador/Resultados construye su widget
        # (matplotlib): cursor de espera + bombeo para que se vea antes del
        # trabajo síncrono, en vez de un cuadro "colgado".
        if idx == 1 and self._comparativa_widget is None:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                QApplication.processEvents()
                self._construir_comparativa()
            finally:
                QApplication.restoreOverrideCursor()
            QApplication.processEvents()
        elif idx == 2 and self._resultados_widget is None:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                QApplication.processEvents()
                self._construir_resultados()
            finally:
                QApplication.restoreOverrideCursor()
            QApplication.processEvents()

    def refresh_available(self):
        self.constructor.explorer.refresh()

    def _set_overlay(self, overlay):
        """Recibe el LoadingOverlay de la ventana principal y lo propaga al
        constructor (carga de archivos, backtest, optimización)."""
        self.constructor._set_overlay(overlay)

    # ── ▶ Ejecutar backtest: una configuración fija, serie completa ──
    @_no_crash
    def _run_backtest(self):
        opt = self.constructor
        if not opt.csv_path or not os.path.exists(opt.csv_path):
            opt.lbl_estado.setText("Selecciona primero un CSV limpiado")
            return
        opt.btn_run.setEnabled(False)
        opt.btn_optimizar.setEnabled(False)
        opt.lbl_estado.setText("Ejecutando backtest…")
        opt.progreso.setVisible(True)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        opt._overlay_begin("Ejecutando backtest…", con_barra=True)
        tf_label, regla_resample = opt.tf_resample()
        th = _BacktestThread(
            opt.csv_path,
            opt.setups(),
            opt.config_global(),
            opt.slider_oos.value() / 100.0,
            opt.chk_wfa.isChecked(),
            opt.sp_wfa.value(),
            codigo=opt.codigo_actual(),
            tf_label=tf_label,
            regla_resample=regla_resample,
            excluir_interpoladas=opt.chk_excluir_interpoladas.isChecked(),
            parent=self,
        )
        th.computed.connect(self._on_done)
        th.estado.connect(opt.lbl_estado.setText)
        th.progreso.connect(opt._overlay_set_progreso)
        th.finished.connect(lambda t=th: self._on_thread_finished(t))
        self._threads.append(th)
        th.start()

    @_no_crash
    def _on_thread_finished(self, th):
        if th in self._threads:
            self._threads.remove(th)
        th.deleteLater()
        self.constructor.progreso.setVisible(False)
        # El overlay NO se oculta aquí: en el backtest/optimización con éxito lo
        # oculta la señal graficos_listos (cuando el render ha terminado), y en
        # el error lo oculta _on_done/_on_optimizacion_terminada.
        # re-evalúa CSV + validez del sistema, en vez de reactivar a ciegas
        self.constructor._actualizar_boton_ejecutable()
        QApplication.restoreOverrideCursor()

    @_no_crash
    def _on_done(self, payload):
        opt = self.constructor
        if 'error' in payload:
            opt.lbl_estado.setText(f"Error: {payload['error']}")
            opt._overlay_end()
            return
        n_tr = payload['resultado']['n_trades']
        opt.lbl_estado.setText(f"Backtest completado: {n_tr} trades")
        try:
            self.resultados.graficos_listos.disconnect(self._cerrar_overlay_backtest)
        except TypeError:
            pass
        self.resultados.graficos_listos.connect(self._cerrar_overlay_backtest)
        self.resultados.mostrar(payload)
        self.tabs.setCurrentWidget(self.resultados)

    def _cerrar_overlay_backtest(self):
        """Oculta el overlay cuando el render de Resultados ha terminado."""
        try:
            self.resultados.graficos_listos.disconnect(self._cerrar_overlay_backtest)
        except TypeError:
            pass
        self.constructor._overlay_end()

    # ── 🔍 Prueba de parametrización (Solo IS): barrido del setup actual ──
    @_no_crash
    def _abrir_dialogo_optimizacion(self):
        opt = self.constructor
        if not opt.csv_path or not os.path.exists(opt.csv_path):
            opt.lbl_estado.setText("Selecciona primero un CSV limpiado")
            return
        setup = opt.setup_seleccionado()
        if setup is None:
            opt.lbl_estado.setText("Selecciona primero un setup en la lista")
            return
        dlg = DialogoOptimizacion(setup, parent=self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        sweep_params, sweep_riesgo, metrica = dlg.resultado()

        opt.btn_run.setEnabled(False)
        opt.btn_optimizar.setEnabled(False)
        opt.lbl_estado.setText("Optimizando IS… cargando datos")
        # Barra del Constructor (visible en la pestaña actual) + la de
        # Optimizador. Antes solo se actualizaba Comparativa (oculta) y el
        # cursor de espera: parecía colgado aunque el hilo avanzara.
        opt.progreso.setRange(0, 1)
        opt.progreso.setValue(0)
        opt.progreso.setTextVisible(True)
        opt.progreso.setFormat("%v / %m")
        opt.progreso.setFixedHeight(14)
        opt.progreso.setVisible(True)
        self.comparativa.progreso.setVisible(True)
        self.comparativa.progreso.setValue(0)
        self.comparativa.lbl_resumen.setText("Optimizando IS… cargando datos")
        self.tabs.setCurrentWidget(self.comparativa)
        opt._overlay_begin("Optimizando IS…", con_barra=True)
        tf_label, regla_resample = opt.tf_resample()
        th = _OptimizerThread(
            opt.csv_path, setup, sweep_params, sweep_riesgo,
            opt.config_global(), opt.slider_oos.value() / 100.0, metrica,
            tf_label=tf_label, regla_resample=regla_resample,
            excluir_interpoladas=opt.chk_excluir_interpoladas.isChecked(),
            parent=self,
        )
        th.progreso.connect(self._on_optimizacion_progreso)
        th.progreso.connect(opt._overlay_set_progreso)
        th.estado.connect(self.constructor.lbl_estado.setText)
        th.terminado.connect(self._on_optimizacion_terminada)
        th.finished.connect(lambda t=th: self._on_thread_finished(t))
        self._threads.append(th)
        th.start()

    @_no_crash
    def _on_optimizacion_progreso(self, i, total):
        total = max(int(total), 1)
        i = max(0, min(int(i), total))
        opt = self.constructor
        opt.progreso.setMaximum(total)
        opt.progreso.setValue(i)
        self.comparativa.actualizar_progreso(i, total)
        if i <= 0:
            msg = f"Optimizando IS… 0/{total} (preparando 1ª combinación)"
        else:
            msg = f"Optimizando IS… {i}/{total}"
        opt.lbl_estado.setText(msg)
        self.comparativa.lbl_resumen.setText(msg)

    @_no_crash
    def _on_optimizacion_terminada(self, payload):
        opt = self.constructor
        self.comparativa.progreso.setVisible(False)
        opt.progreso.setVisible(False)
        opt.progreso.setTextVisible(False)
        opt.progreso.setFixedHeight(6)
        opt.progreso.setRange(0, 0)
        if 'error' in payload:
            opt.lbl_estado.setText(f"Error: {payload['error']}")
            self.comparativa.lbl_resumen.setText(
                f"Error en la optimización: {payload['error']}")
            opt._overlay_end()
            return
        opt.lbl_estado.setText(
            f"Optimización completada: {len(payload['resultados'])} combinaciones probadas sobre IS")
        if payload.get('avisos'):
            opt.lbl_estado.setText(f"⚠ {payload['avisos'][0]}")
        self.comparativa.mostrar(payload)
        self.tabs.setCurrentWidget(self.comparativa)
        opt._overlay_end()

    @_no_crash
    def _usar_configuracion(self, setup):
        self.constructor.cargar_setup_en_constructor(setup)
        self.tabs.setCurrentWidget(self.constructor)

    @_no_crash
    def _agregar_setups(self, setups):
        self.constructor.agregar_setups(setups)
        self.tabs.setCurrentWidget(self.constructor)


