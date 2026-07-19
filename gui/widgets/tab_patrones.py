"""
Sub-pestaña "Patrones" del Analizador: detección y validación estadística de
patrones de velas japonesas sobre el CSV limpiado del activo analizado.

Por patrón: icono representativo dibujado con matplotlib, nº de ocurrencias,
hit rate a 1/3/5/10 velas con su evolución temporal, significancia (binomial
vs 50% y vs sesgo base del activo) y edge (retorno forward vs base). Filtros
por régimen de mercado (ER y Hurst) y exclusión de velas interpoladas/anómalas.

El escaneo O(n_velas) corre en un QThread (lectura CSV + detección + contexto);
el cambio de filtros solo recalcula estadística O(n_ocurrencias) en el hilo GUI.
"""
import os
import re

import numpy as np
import pandas as pd
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QCheckBox,
    QScrollArea, QFrame, QTableWidget, QTableWidgetItem, QHeaderView,
    QSizePolicy, QPushButton, QButtonGroup, QInputDialog,
)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.patches import Rectangle

from core.candle_patterns import (
    LAGS, MIN_OCURRENCIAS, MIN_OCURRENCIAS_BARRA, PATRONES_INFO,
    PATRONES_ORIGINALES,
    detectar_patrones, preparar_contexto, preparar_base_filtro,
    calcular_stats_patron, agregar_por_periodo,
)
from core.metrics import calcular_er_series, calcular_hurst_array
from core.config import tf_to_minutes

# Temporalidades ofrecidas para recalcular patrones (solo se puede subir de
# granularidad respecto a la nativa del archivo importado, nunca bajar).
TF_LABELS_PATRONES = ['1m', '3m', '5m', '15m', '30m', '1h', '4h', '1d', '1w']
REGLAS_RESAMPLE = {'1m': '1min', '3m': '3min', '5m': '5min', '15m': '15min',
                   '30m': '30min', '1h': '1h', '4h': '4h', '1d': '1D', '1w': '1W'}

# Familias de TF para el precálculo en segundo plano: al cargar un activo (o
# pulsar una TF de otra familia) se van precalculando las TFs hermanas para
# que el cambio dentro de la familia sea instantáneo (caché _payloads_por_tf).
TF_FAMILIAS = {
    'minutos':  ['1m', '3m', '5m', '15m', '30m'],
    'horario':  ['1h', '4h'],
    'superior': ['1d', '1w'],
}


def _familia_de_tf(tf_label):
    for tfs in TF_FAMILIAS.values():
        if tf_label in tfs:
            return tfs
    return []   # TFs custom: sin hermanas que precalcular


def _parsear_tf_custom(texto):
    """'20m'/'2 h'/'3d'/'2w' → etiqueta normalizada ('20m') o None si no es
    una temporalidad válida."""
    m = re.match(r'^\s*(\d+)\s*(m|h|d|w)\s*$', str(texto), re.IGNORECASE)
    if not m or int(m.group(1)) <= 0:
        return None
    return f"{int(m.group(1))}{m.group(2).lower()}"


def _regla_de_tf(tf_label):
    """Regla de resample de pandas para una TF (preset o custom)."""
    if tf_label in REGLAS_RESAMPLE:
        return REGLAS_RESAMPLE[tf_label]
    m = re.match(r'^(\d+)(m|h|d|w)$', tf_label)
    if not m:
        return None
    unidad = {'m': 'min', 'h': 'h', 'd': 'D', 'w': 'W'}[m.group(2)]
    return f"{m.group(1)}{unidad}"

# Bloques de calendario ofrecidos para agregar_por_periodo (etiqueta, regla
# de resample de pandas, minutos aprox. — para deshabilitar los que sean más
# finos que la TF nativa del archivo importado, igual que TF_LABELS_PATRONES).
PERIODOS_BARRAS = [
    ('Día', '1D', 1440), ('Semana', '1W', 10080), ('Mes', '1ME', 43200),
    ('Trimestre', '1QE', 129600), ('Año', '1YE', 525600),
]

FIG_BG = '#0d1424'
AX_FG = '#c8d6e5'
GRID_C = '#253a60'
VERDE = '#2ecc71'
ROJO = '#e74c3c'
GRIS = '#5a7a9a'
AMBAR = '#f1c40f'

STYLE_PATRONES = """
QComboBox {
    min-width: 105px;
}
QFrame#card {
    background-color: #141e30;
    border: 1px solid #253a60;
    border-radius: 6px;
}
QLabel#cardTitle { color: #4fc3f7; font-size: 13px; font-weight: bold; }
QLabel#cardSub   { color: #5a7a9a; font-size: 10px; }
QLabel#cardStat  { color: #c8d6e5; font-size: 11px; }
QLabel#estado    { color: #5a7a9a; font-size: 11px; }
QLabel#filtroLbl { color: #c8d6e5; font-size: 11px; }
QCheckBox { color: #c8d6e5; font-size: 11px; }
QTableWidget {
    background-color: #0d1424;
    alternate-background-color: #101a2e;
    color: #c8d6e5;
    gridline-color: #253a60;
    border: 1px solid #253a60;
    font-size: 11px;
}
QHeaderView::section {
    background-color: #141e30;
    color: #4fc3f7;
    border: 1px solid #253a60;
    padding: 3px;
    font-size: 11px;
}
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
QPushButton#vista {
    background-color: #1a2a45; color: #5a7a9a; padding: 4px 12px;
    font-size: 11px;
}
QPushButton#vista:checked {
    background-color: #2a4a6a; color: #4fc3f7; font-weight: bold;
}
"""


def _style_ax(ax):
    ax.set_facecolor(FIG_BG)
    ax.tick_params(colors=AX_FG, labelsize=6)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_C)
    ax.grid(True, alpha=0.25, color=GRID_C, linewidth=0.5)


def _fmt_pvalue(p):
    """Con muestras enormes (activos de 1m con cientos de miles de
    ocurrencias) el p-valor real puede ser del orden de 1e-50 — mostrarlo
    con 3 decimales fijos lo redondea a "0.000", pareciendo un cálculo roto
    en vez de un resultado matemáticamente correcto. Convención estándar
    en vez de notación científica: "< 0.001" (el valor exacto no cambia
    ninguna decisión por debajo de ese umbral)."""
    if p is None:
        return "—"
    if p < 0.001:
        return "< 0.001"
    return f"{p:.3f}"


def _fmt_pvalue_exacto(p):
    """Valor exacto para tooltip cuando _fmt_pvalue lo trunca a '< 0.001'."""
    if p is None:
        return "—"
    return f"p exacto = {p:.2e}"


def _no_crash(fn):
    """En PyQt6 una excepcion no capturada dentro de un slot ABORTA el proceso
    entero (qFatal): todo slot conectado a señales debe capturarlas."""
    def wrapper(self, *a, **kw):
        try:
            return fn(self, *a, **kw)
        except Exception as e:
            print(f"[Patrones] Error en {fn.__name__}: {e}")
    wrapper.__name__ = fn.__name__
    return wrapper


# ══════════════ iconos de patrón (velas hardcodeadas) ══════════════
# Tuplas (open, high, low, close, es_contexto). Las de contexto se pintan
# apagadas y transmiten la tendencia previa que da sentido al patrón.
EJEMPLOS_PATRON = {
    'Doji': [
        (97, 99.2, 96.8, 99, True), (99, 101.2, 98.8, 101, True),
        (101, 102.5, 99.5, 101.1, False)],
    'Martillo': [
        (103, 103.2, 100.8, 101, True), (101, 101.2, 98.8, 99, True),
        (99, 99.6, 95.5, 99.4, False)],
    'Hombre Colgado': [
        (97, 99.2, 96.8, 99, True), (99, 101.2, 98.8, 101, True),
        (101, 101.5, 97.5, 101.3, False)],
    'Martillo Invertido': [
        (103, 103.2, 100.8, 101, True), (101, 101.2, 98.8, 99, True),
        (99, 102.5, 98.9, 99.3, False)],
    'Estrella Fugaz': [
        (97, 99.2, 96.8, 99, True), (99, 101.2, 98.8, 101, True),
        (101, 104.5, 100.9, 101.3, False)],
    'Envolvente Alcista': [
        (102, 102.4, 100.6, 101, True),
        (101, 101.3, 99.7, 100, False), (99.7, 102.3, 99.5, 102, False)],
    'Envolvente Bajista': [
        (98, 99.4, 97.6, 99, True),
        (99, 100.3, 98.7, 100, False), (100.3, 100.5, 97.7, 98, False)],
    'Harami Alcista': [
        (103, 103.4, 101.6, 102, True),
        (102, 102.2, 98.8, 99, False), (100, 100.9, 99.7, 100.8, False)],
    'Harami Bajista': [
        (97, 98.4, 96.6, 98, True),
        (98, 101.2, 97.8, 101, False), (100.3, 100.4, 99.2, 99.4, False)],
    'Morning Star': [
        (103, 103.2, 99.8, 100, False), (99.7, 100.2, 99.2, 99.5, False),
        (99.8, 102.7, 99.7, 102.5, False)],
    'Evening Star': [
        (97, 100.2, 96.8, 100, False), (100.3, 100.8, 99.9, 100.6, False),
        (100.2, 100.3, 97.3, 97.5, False)],
    'Tres Soldados Blancos': [
        (97, 99.3, 96.9, 99, False), (98.5, 101, 98.4, 100.8, False),
        (100, 102.6, 99.9, 102.4, False)],
    'Tres Cuervos Negros': [
        (103, 103.1, 100.7, 101, False), (101.5, 101.6, 99, 99.2, False),
        (100, 100.1, 97.4, 97.6, False)],
    'Marubozu Alcista': [
        (97, 99.2, 96.8, 99, True), (99, 101.2, 98.8, 101, True),
        (99, 103, 99, 103, False)],
    'Marubozu Bajista': [
        (103, 105.2, 102.8, 105, True), (105, 106.2, 103.8, 104, True),
        (104, 104, 100, 100, False)],
    'Spinning Top': [
        (97, 99.2, 96.8, 99, True), (99, 101.2, 98.8, 101, True),
        (100.8, 102.5, 98.5, 101.2, False)],
    'Doji Libélula': [
        (103, 103.2, 100.8, 101, True), (101, 101.2, 98.8, 99, True),
        (99, 99.1, 95.5, 99, False)],
    'Doji Lápida': [
        (97, 99.2, 96.8, 99, True), (99, 101.2, 98.8, 101, True),
        (101, 104.5, 100.9, 101, False)],
    'Piercing Line': [
        (102, 102.2, 99.8, 100, True),
        (100, 100.2, 96.8, 97, False), (96.5, 99.2, 96.3, 99, False)],
    'Dark Cloud Cover': [
        (98, 98.2, 95.8, 97, True),
        (97, 100.2, 96.8, 100, False), (100.5, 100.7, 97.8, 98, False)],
    'Tweezer Top': [
        (99, 101.2, 98.8, 101, True),
        (101, 103.4, 100.6, 103, False), (103.2, 103.5, 101, 101.5, False)],
    'Tweezer Bottom': [
        (101, 101.2, 98.8, 99, True),
        (99, 99.4, 96.6, 97, False), (96.8, 99.5, 96.5, 99, False)],
    'Kicker Alcista': [
        (103, 103.2, 100.8, 101, True),
        (101, 101.2, 98.8, 99, False), (101.5, 104, 101.3, 103.8, False)],
    'Kicker Bajista': [
        (97, 99.2, 96.8, 99, True),
        (99, 101.2, 98.8, 101, False), (98.5, 98.7, 96, 96.2, False)],
    'Three Inside Up': [
        (103, 103.2, 99.8, 100, False), (100.2, 100.6, 99.4, 100.4, False),
        (100.3, 103.8, 100.1, 103.6, False)],
    'Three Inside Down': [
        (97, 100.2, 96.8, 100, False), (99.6, 100.6, 99.4, 99.8, False),
        (99.7, 99.9, 96.2, 96.4, False)],
    'Three Outside Up': [
        (102, 102.2, 99.8, 100, False), (99.7, 102.4, 99.5, 102.2, False),
        (102.2, 104.5, 102, 104.2, False)],
    'Three Outside Down': [
        (98, 100.2, 97.8, 100, False), (100.3, 100.5, 97.6, 97.8, False),
        (97.8, 98, 95.5, 95.7, False)],
    'Abandoned Baby Alcista': [
        (103, 103.2, 99.8, 100, False), (98.5, 98.7, 98.3, 98.53, False),
        (99.5, 102.7, 99.3, 102.5, False)],
    'Abandoned Baby Bajista': [
        (97, 100.2, 96.8, 100, False), (101.5, 101.7, 101.3, 101.53, False),
        (100.5, 100.7, 97.3, 97.5, False)],
    'Rising Three Methods': [
        (97, 103.5, 96.8, 103, False), (102, 102.3, 100.5, 100.8, False),
        (100.5, 101, 99.3, 99.6, False), (99.4, 100.5, 98.8, 100.2, False),
        (100, 105, 99.8, 104.8, False)],
    'Falling Three Methods': [
        (103, 103.2, 96.5, 97, False), (98, 99.8, 97.7, 98.1, False),
        (98.2, 99.5, 97.5, 98.6, False), (98.5, 100, 97.8, 99, False),
        (99, 99.2, 94, 94.2, False)],
}


def dibujar_icono_patron(ax, velas):
    """Pinta 2-3 mini-velas (Rectangle cuerpo + vlines mechas) sin ejes."""
    ax.set_facecolor(FIG_BG)
    for x, (o, h, l, c, ctx) in enumerate(velas):
        if ctx:
            color = GRIS
            alpha = 0.45
        else:
            color = VERDE if c >= o else ROJO
            alpha = 1.0
        ax.vlines(x, l, h, color=color, linewidth=1.2, alpha=alpha)
        y0, alto = min(o, c), abs(c - o)
        alto = max(alto, 0.15)  # un doji necesita cuerpo visible
        ax.add_patch(Rectangle((x - 0.3, y0), 0.6, alto,
                               facecolor=color, edgecolor=color, alpha=alpha))
    ax.set_xlim(-0.7, len(velas) - 0.3)
    lows = [v[2] for v in velas]
    highs = [v[1] for v in velas]
    pad = (max(highs) - min(lows)) * 0.08
    ax.set_ylim(min(lows) - pad, max(highs) + pad)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_C)


# ══════════════ thread de escaneo ══════════════
class _PatternScanThread(QThread):
    """Lee el CSV limpiado y precomputa detección + contexto forward.
    IMPORTANTE: parentar siempre al widget — destruir un QThread corriendo
    aborta el proceso entero (misma nota que en tab_comparador)."""
    computed = pyqtSignal(str, object)   # csv_path, payload

    _COLS = ['timestamp', 'open', 'high', 'low', 'close',
             'interpolado', 'anomalia', 'ER', 'hurst']

    def __init__(self, csv_path, parent=None):
        super().__init__(parent)
        self._path = csv_path

    def run(self):
        try:
            try:
                df = pd.read_csv(self._path, usecols=self._COLS, engine='pyarrow')
            except (ImportError, ValueError):
                # columnas opcionales ausentes (CSV antiguo) o sin pyarrow:
                # releer pidiendo solo las presentes
                presentes = pd.read_csv(self._path, nrows=0).columns
                cols = [c for c in self._COLS if c in presentes]
                df = pd.read_csv(self._path, usecols=cols)
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df = df.dropna(subset=['timestamp', 'close']).sort_values('timestamp')

            def col(nombre):
                return df[nombre].values if nombre in df.columns else None

            patrones = detectar_patrones(df['open'].values, df['high'].values,
                                         df['low'].values, df['close'].values)
            ctx = preparar_contexto(df['close'].values, col('interpolado'),
                                    col('anomalia'), col('ER'), col('hurst'),
                                    timestamps=df['timestamp'].values)
            self.computed.emit(self._path, {
                'patrones': patrones,
                'ctx': ctx,
                'timestamps': df['timestamp'].values,
                'n_velas': len(df),
                'tiene_er': ctx['regimen_er'] is not None,
                'tiene_hurst': ctx['regimen_hurst'] is not None,
                # arrays crudos para poder resamplear a otras TF sin releer
                # el CSV (una lectura de un 1m grande cuesta varios segundos)
                'raw': {
                    'timestamps': df['timestamp'].values,
                    'open': df['open'].values, 'high': df['high'].values,
                    'low': df['low'].values, 'close': df['close'].values,
                    'interpolado': col('interpolado'), 'anomalia': col('anomalia'),
                },
            })
        except Exception as e:
            self.computed.emit(self._path, {'error': str(e)})


# ══════════════ thread de resample a otra temporalidad ══════════════
class _ResampleThread(QThread):
    """Reconstruye velas OHLC en una temporalidad más gruesa a partir de los
    arrays crudos ya en memoria (sin tocar disco), recalcula ER y Hurst sobre
    esas velas y vuelve a detectar los patrones. Mismos parámetros de Hurst
    que usa la limpieza (library/scripts_utiles/limpieza_datos_er.py) según
    la TF resultante."""
    computed = pyqtSignal(str, str, object)   # csv_path, tf_label, payload

    # Parámetros de Hurst por escala temporal (mismos valores que la limpieza,
    # library/scripts_utiles/limpieza_datos_er.py): se eligen por MINUTOS de la
    # TF resultante y no por etiqueta exacta, para que una TF custom ('3d',
    # '2w'…) caiga en el tramo correcto en vez de en el intradía por defecto.
    _PARAMS_HURST_INTRADIA = (1024, 10, [16, 32, 64, 128, 256])   # < 1 día
    _PARAMS_HURST_DIARIO = (504, 5, [16, 32, 64, 128])            # 1d a < 1w
    _PARAMS_HURST_SEMANAL = (256, 2, [8, 16, 32, 64, 128])        # >= 1w
    _PERIODO_ER = 10

    def __init__(self, csv_path, tf_label, regla, raw, parent=None):
        super().__init__(parent)
        self._path = csv_path
        self._tf_label = tf_label
        self._regla = regla
        self._raw = raw

    def run(self):
        try:
            raw = self._raw
            cols = {'open': raw['open'], 'high': raw['high'],
                    'low': raw['low'], 'close': raw['close']}
            if raw.get('interpolado') is not None:
                cols['interpolado'] = raw['interpolado']
            if raw.get('anomalia') is not None:
                cols['anomalia'] = raw['anomalia']
            df = pd.DataFrame(cols, index=pd.DatetimeIndex(raw['timestamps']))

            agg = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}
            # 'min' y no 'max': una vela resampleada solo se marca sucia si
            # TODOS sus ticks originales lo eran (vela íntegramente
            # sintética) — con 'max' bastaría un solo tick interpolado en
            # todo el día para descartar la vela entera bajo el filtro de
            # "excluir interpoladas/anómalas", vaciando casi toda la serie.
            if 'interpolado' in df.columns:
                agg['interpolado'] = 'min'
            if 'anomalia' in df.columns:
                agg['anomalia'] = 'min'
            r = df.resample(self._regla).agg(agg).dropna(subset=['close'])

            retorno_log = np.log(r['close'] / r['close'].shift(1))
            er = calcular_er_series(retorno_log, periodo=self._PERIODO_ER)

            minutos = tf_to_minutes(self._tf_label)
            if minutos is not None and minutos >= 10080:
                ventana, paso, lags = self._PARAMS_HURST_SEMANAL
            elif minutos is not None and minutos >= 1440:
                ventana, paso, lags = self._PARAMS_HURST_DIARIO
            else:
                ventana, paso, lags = self._PARAMS_HURST_INTRADIA
            hurst_vals = calcular_hurst_array(
                retorno_log.fillna(0.0).values.astype(np.float64),
                ventana, paso, np.array(lags, dtype=np.int64))
            hurst = pd.Series(hurst_vals, index=r.index).interpolate() \
                .bfill().ffill().fillna(0.5).values

            interpolado = r['interpolado'].values if 'interpolado' in r.columns else None
            anomalia = r['anomalia'].values if 'anomalia' in r.columns else None

            patrones = detectar_patrones(r['open'].values, r['high'].values,
                                         r['low'].values, r['close'].values)
            ctx = preparar_contexto(r['close'].values, interpolado, anomalia,
                                    er.values, hurst, timestamps=r.index.values)
            self.computed.emit(self._path, self._tf_label, {
                'patrones': patrones,
                'ctx': ctx,
                'timestamps': r.index.values,
                'n_velas': len(r),
                'tiene_er': ctx['regimen_er'] is not None,
                'tiene_hurst': ctx['regimen_hurst'] is not None,
            })
        except Exception as e:
            self.computed.emit(self._path, self._tf_label, {'error': str(e)})


# ══════════════ tarjeta por patrón ══════════════
class PatternCard(QFrame):
    def __init__(self, nombre, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        self._nombre = nombre

        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setSpacing(12)

        # icono estático (se dibuja una sola vez)
        fig_icono = Figure(figsize=(1.5, 1.15), facecolor=FIG_BG)
        canvas_icono = FigureCanvasQTAgg(fig_icono)
        canvas_icono.setFixedSize(130, 100)
        ax = fig_icono.add_subplot(111)
        dibujar_icono_patron(ax, EJEMPLOS_PATRON[nombre])
        fig_icono.subplots_adjust(left=0.04, right=0.96, top=0.96, bottom=0.04)
        lay.addWidget(canvas_icono)

        # columna de texto
        col = QVBoxLayout()
        col.setSpacing(2)
        sesgo = PATRONES_INFO[nombre]['dir']
        flecha = {1: ('▲ alcista', VERDE), -1: ('▼ bajista', ROJO),
                  0: ('◆ giro contra-tendencia', AMBAR)}[sesgo]
        titulo = QLabel(nombre)
        titulo.setObjectName("cardTitle")
        col.addWidget(titulo)
        sub = QLabel(flecha[0])
        sub.setObjectName("cardSub")
        sub.setStyleSheet(f"color: {flecha[1]}; font-size: 10px;")
        col.addWidget(sub)
        self.lbl_n = QLabel("—")
        self.lbl_n.setObjectName("cardStat")
        col.addWidget(self.lbl_n)
        self.lbl_hr = QLabel("—")
        self.lbl_hr.setObjectName("cardStat")
        self.lbl_hr.setTextFormat(Qt.TextFormat.RichText)
        self.lbl_hr.setWordWrap(True)
        col.addWidget(self.lbl_hr)
        self.lbl_edge = QLabel("—")
        self.lbl_edge.setObjectName("cardStat")
        self.lbl_edge.setTextFormat(Qt.TextFormat.RichText)
        self.lbl_edge.setWordWrap(True)
        col.addWidget(self.lbl_edge)
        self.lbl_sig = QLabel("")
        self.lbl_sig.setObjectName("cardStat")
        self.lbl_sig.setTextFormat(Qt.TextFormat.RichText)
        self.lbl_sig.setWordWrap(True)
        col.addWidget(self.lbl_sig)
        col.addStretch()
        wrap = QWidget()
        wrap.setLayout(col)
        # ancho fijo + word-wrap: el badge de significancia (que puede ser
        # una frase larga, p.ej. "significativo pero EN CONTRA...") salta de
        # línea en vez de cortarse, y deja sitio a la derecha para el gráfico
        wrap.setFixedWidth(340)
        lay.addWidget(wrap)

        # evolución del hit rate/retorno en ventana móvil
        self.fig = Figure(figsize=(4, 1.4), facecolor=FIG_BG)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setStyleSheet("background-color: transparent;")
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                  QSizePolicy.Policy.Expanding)
        self.canvas.setMinimumHeight(130)
        lay.addWidget(self.canvas, 1)
        self.setMinimumHeight(150)

    def clear_stats(self, texto="Sin datos"):
        self.lbl_n.setText("—")
        self.lbl_hr.setText("—")
        self.lbl_edge.setText("—")
        self.lbl_sig.setText("")
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor(FIG_BG)
        ax.text(0.5, 0.5, texto, ha='center', va='center', color=GRIS,
                fontsize=8, transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_C)
        self.canvas.draw_idle()

    def update_stats(self, stats, timestamps, vista='hitrate', lag_sel=LAGS[0],
                     barras=None, ancho_dias=1.0):
        n_total = stats['n_total']
        self.lbl_n.setText(f"Ocurrencias: {n_total}")

        def color_hr(v):
            if v is None:
                return GRIS
            return VERDE if v > 0.55 else AMBAR if v >= 0.5 else ROJO

        trozos = []
        for lag in LAGS:
            s = stats['por_lag'][lag]
            hr = s['hit_rate']
            txt = f"{hr * 100:.0f}%" if hr is not None else "—"
            trozos.append(f"+{lag}: <b style='color:{color_hr(hr)}'>{txt}</b>")
        self.lbl_hr.setText("Hit rate &nbsp; " + " &nbsp; ".join(trozos))

        trozos = []
        for lag in LAGS:
            s = stats['por_lag'][lag]
            if s['edge'] is None:
                trozos.append("—")
            else:
                # puntos básicos (1 pb = 0.01%): con retornos por vela tan
                # pequeños, mostrar en % con 2 decimales redondea casi
                # cualquier edge real a "0.00%" y el color (por el signo
                # exacto, no el redondeado) parece parpadear sin sentido.
                bp = s['edge'] * 10000
                c = VERDE if bp > 0 else ROJO
                trozos.append(f"<b style='color:{c}'>{bp:+.1f}pb</b>")
        self.lbl_edge.setText("Edge vs base &nbsp; " + " &nbsp; ".join(trozos))

        # badge de significancia (en el mejor lag con p definido). Se exigen
        # AMBOS p-valores <0.05: distinto del 50% (no es azar) Y distinto del
        # sesgo base del propio activo bajo el mismo filtro (no es solo que
        # el activo ya tienda a subir/bajar en general).
        # OJO: el test es de dos colas — detecta desviación del 50% en
        # CUALQUIER dirección, así que "significativo" NO implica que el
        # patrón acierte más: puede ser significativamente peor que el azar
        # (hit rate < 50% de forma fiable, es decir, falla más de lo
        # esperado). Hay que mirar también el hit rate para saber el signo.
        mejor = None
        for lag in LAGS:
            s = stats['por_lag'][lag]
            if s['p_vs_50'] is not None and (mejor is None or s['p_vs_50'] < mejor[1]):
                mejor = (lag, s['p_vs_50'], s['p_vs_base'], s['significativo'], s['hit_rate'])
        if mejor is None:
            self.lbl_sig.setText(
                f"<span style='color:{GRIS}'>n insuficiente para "
                f"significancia (mín. {MIN_OCURRENCIAS})</span>")
            self.lbl_sig.setToolTip("")
        else:
            if mejor[3] and mejor[4] is not None and mejor[4] > 0.5:
                self.lbl_sig.setText(
                    f"<span style='color:{VERDE}'>✔ significativo y positivo a +{mejor[0]} "
                    f"(p={_fmt_pvalue(mejor[1])} vs 50%, p={_fmt_pvalue(mejor[2])} vs base)</span>")
            elif mejor[3]:
                self.lbl_sig.setText(
                    f"<span style='color:{ROJO}'>⚠ significativo pero EN CONTRA a +{mejor[0]} "
                    f"— falla más de lo esperado (p={_fmt_pvalue(mejor[1])} vs 50%, "
                    f"p={_fmt_pvalue(mejor[2])} vs base)</span>")
            else:
                self.lbl_sig.setText(
                    f"<span style='color:{GRIS}'>sin edge significativo a +{mejor[0]} "
                    f"(p={_fmt_pvalue(mejor[1])} vs 50%, p={_fmt_pvalue(mejor[2])} vs base)</span>")
            # tooltip con el valor exacto cuando alguno de los dos se truncó
            # a "< 0.001" (relevante con muestras enormes, p puede ser 1e-50)
            partes = []
            if mejor[1] is not None and mejor[1] < 0.001:
                partes.append(f"vs 50%: {_fmt_pvalue_exacto(mejor[1])}")
            if mejor[2] is not None and mejor[2] < 0.001:
                partes.append(f"vs base: {_fmt_pvalue_exacto(mejor[2])}")
            self.lbl_sig.setToolTip(" · ".join(partes))

        # barras agregadas por bloque de calendario (Día/Semana/Mes/...,
        # elegido arriba en "Periodo"), para el lag elegido en "Lag
        # ranking" — a diferencia de una ventana móvil por ocurrencias, cada
        # barra responde directamente a "¿en este periodo el patrón acertó
        # más o menos de lo esperado?" sin que el eje temporal quede
        # irregularmente espaciado ni el ruido de una ventana corta tape la
        # respuesta.
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        _style_ax(ax)
        hay_datos = bool(barras) and len(barras['fechas']) >= 1
        max_abs = 15.0  # mínimo de escala para que un gráfico plano no se vea amplificado
        if hay_datos:
            y = (barras['hit_rate'] - 0.5) * 100 if vista == 'hitrate' else barras['edge_pb']
            max_abs = max(max_abs, float(np.abs(y).max()))
            colores = [VERDE if v > 0 else ROJO for v in y]
            fechas = barras['fechas']
            # ancho fijo según el periodo elegido (no según el espaciado real
            # entre los bloques que sobrevivieron al mínimo de ocurrencias:
            # si se descarta un bloque intermedio, ese hueco se agranda y un
            # ancho "medio" quedaría sobredimensionado, solapando las barras
            # que sí están espaciadas con normalidad)
            ax.bar(fechas, y, width=ancho_dias, color=colores, alpha=0.85)
            ax.axhline(0, color=GRIS, linewidth=0.7, linestyle='--', alpha=0.6)
            ax.set_ylim(-max_abs * 1.15, max_abs * 1.15)
            ylabel = 'Desv. HR vs 50% (pp)' if vista == 'hitrate' else 'Edge por bloque (pb)'
            ax.set_ylabel(ylabel, fontsize=6, color=AX_FG)
        else:
            if n_total == 0:
                msg = "Sin ocurrencias con este filtro"
            else:
                msg = f"Ningún bloque con {MIN_OCURRENCIAS_BARRA}+ ocurrencias (hay {n_total} en total)"
            ax.text(0.5, 0.5, msg, ha='center', va='center', color=GRIS,
                    fontsize=8, transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
        try:
            self.fig.tight_layout(pad=0.4)
        except Exception:
            pass
        self.canvas.draw_idle()


# ══════════════ pestaña ══════════════
class TabPatrones(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(STYLE_PATRONES)
        self._csv_path = None
        self._cache_key = None
        self._payload = None
        self._needs_scan = False
        self._threads = []
        # estado de temporalidades: payloads por TF ya calculados, arrays
        # crudos del activo actual (para resamplear sin releer el CSV),
        # TF nativa del archivo y TF actualmente seleccionada/pendiente
        self._payloads_por_tf = {}
        self._raw = None
        self._tf_nativo = None
        self._tf_actual = None
        self._tf_pendiente = None
        self._mostrar_extra = False
        # cola de precálculo de TFs en segundo plano (un hilo en vuelo como
        # máximo, para no saturar CPU): tf_labels pendientes de resamplear
        self._cola_precalculo = []
        self._tf_en_calculo = None
        self._tf_custom = None   # última TF personalizada usada (para el diálogo)

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        def _lbl(txt):
            l = QLabel(txt)
            l.setObjectName("filtroLbl")
            return l

        # ── fila de temporalidad ──
        fila_tf = QHBoxLayout()
        fila_tf.setSpacing(6)
        fila_tf.addWidget(_lbl("Temporalidad:"))
        self._tf_group = QButtonGroup(self)
        self._tf_group.setExclusive(True)
        self._tf_buttons = {}
        for i, tf_label in enumerate(TF_LABELS_PATRONES):
            btn = QPushButton(tf_label)
            btn.setObjectName("tf")
            btn.setCheckable(True)
            btn.setEnabled(False)
            self._tf_group.addButton(btn, i)
            self._tf_buttons[tf_label] = btn
            fila_tf.addWidget(btn)
        # botón de TF personalizada: abre un diálogo para escribir cualquier
        # temporalidad (20m, 2h, 3d, 5d…) no prefijada en la fila
        self._btn_tf_custom = QPushButton("Custom…")
        self._btn_tf_custom.setObjectName("tf")
        self._btn_tf_custom.setCheckable(True)
        self._btn_tf_custom.setEnabled(False)
        self._btn_tf_custom.setToolTip(
            "Analizar los patrones en una temporalidad personalizada "
            "(p.ej. 20m, 2h, 3d, 5d). Debe ser igual o más gruesa que la "
            "nativa del archivo; idealmente múltiplo de ella")
        self._tf_group.addButton(self._btn_tf_custom, len(TF_LABELS_PATRONES))
        fila_tf.addWidget(self._btn_tf_custom)
        self._tf_group.idClicked.connect(self._on_tf_clicked)
        fila_tf.addStretch()
        root.addLayout(fila_tf)

        # ── cabecera: filtros (repartidos en dos filas para no forzar el
        # ancho mínimo de la ventana — con los 5 combos + toggle en una sola
        # fila sin scroll, la app se veía obligada a agrandarse) ──
        self.lbl_estado = QLabel("Analiza un activo para ver sus patrones")
        self.lbl_estado.setObjectName("estado")
        root.addWidget(self.lbl_estado)

        fila1 = QHBoxLayout()
        fila1.setSpacing(8)
        fila1.addWidget(_lbl("Régimen ER:"))
        self.cmb_er = QComboBox()
        for texto, val in [("Todos", None), ("Tendencia (ER alto)", 2),
                           ("Neutro", 1), ("Ruido (ER bajo)", 0)]:
            self.cmb_er.addItem(texto, val)
        fila1.addWidget(self.cmb_er)

        fila1.addWidget(_lbl("Régimen Hurst:"))
        self.cmb_hurst = QComboBox()
        for texto, val in [("Todos", None), ("Tendencia (H>0.58)", 2),
                           ("Paseo aleatorio", 1), ("Mean reversion (H<0.52)", 0)]:
            self.cmb_hurst.addItem(texto, val)
        fila1.addWidget(self.cmb_hurst)

        fila1.addWidget(_lbl("Sesión:"))
        self.cmb_sesion = QComboBox()
        self.cmb_sesion.addItem("Globex (día completo)", None)
        # Todas las franjas se muestran en HORA ESPAÑOLA peninsular (misma
        # referencia para todas las sesiones), aunque el cálculo interno sea
        # distinto por sesión: Londres/NY usan el reloj de la propia plaza
        # (8:00-17:00 allí, DST-aware) y Overnight un rango UTC fijo (1-9).
        # Equivalencias con Madrid: Londres siempre +1h (Reino Unido y la UE
        # cambian de hora el mismo día) → 9-18h exactas todo el año; NY +6h
        # casi todo el año → 14-23h salvo las 2-3 semanas en que EEUU ya
        # cambió y Europa aún no (13-22h); Overnight 1-9 UTC → 2-10h en
        # invierno y 3-11h en verano. El "≈" marca las que varían.
        _FRANJA_ES = {'overnight': '≈02-10h España',
                      'londres': '09-18h España',
                      'ny': '≈14-23h España'}
        for clave, nombre in [('overnight', 'Overnight'), ('londres', 'Londres'), ('ny', 'Nueva York')]:
            self.cmb_sesion.addItem(f"{nombre} ({_FRANJA_ES[clave]})", clave)
        self.cmb_sesion.setToolTip(
            "Filtra las velas por franja horaria antes de detectar y "
            "calcular los patrones. Franjas expresadas en hora española "
            "peninsular. Londres/Nueva York cubren las 8:00-17:00 del reloj "
            "de la propia plaza, ajustadas automáticamente al horario de "
            "verano/invierno; Overnight es un rango UTC fijo (01-09), que "
            "en hora española son las 2-10h en invierno y las 3-11h en "
            "verano. El ≈ marca las franjas que se desplazan con el cambio "
            "de hora")
        fila1.addWidget(self.cmb_sesion)

        self.chk_limpias = QCheckBox("Excluir interpoladas/anómalas")
        self.chk_limpias.setChecked(True)
        fila1.addWidget(self.chk_limpias)
        fila1.addStretch()

        self.btn_mas_patrones = QPushButton("Más patrones ▾")
        self.btn_mas_patrones.setCheckable(True)
        self.btn_mas_patrones.setToolTip(
            "Muestra/oculta 19 patrones adicionales (menos consolidados que "
            "los 13 clásicos): piercing, tweezer, kicker, marubozu, spinning "
            "top, doji direccionales, three inside/outside, abandoned baby y "
            "los métodos de continuación de 5 velas")
        self.btn_mas_patrones.toggled.connect(self._on_toggle_extra)
        fila1.addWidget(self.btn_mas_patrones)
        root.addLayout(fila1)

        fila2 = QHBoxLayout()
        fila2.setSpacing(8)
        fila2.addWidget(_lbl("Lag ranking:"))
        self.cmb_lag = QComboBox()
        for lag in LAGS:
            self.cmb_lag.addItem(f"+{lag}", lag)
        self.cmb_lag.setCurrentIndex(2)  # +5 por defecto
        fila2.addWidget(self.cmb_lag)

        fila2.addWidget(_lbl("Periodo:"))
        self.cmb_periodo = QComboBox()
        for etiqueta, regla, _minutos in PERIODOS_BARRAS:
            self.cmb_periodo.addItem(etiqueta, regla)
        self.cmb_periodo.setToolTip(
            "Agrupa las ocurrencias del patrón en bloques de calendario para "
            "el gráfico de la tarjeta (¿en qué épocas acertó más/menos?). "
            "Solo se ofrecen bloques igual de finos o más gruesos que la "
            "temporalidad nativa del archivo importado")
        fila2.addWidget(self.cmb_periodo)

        fila2.addWidget(_lbl("Gráfica:"))
        self._vista_group = QButtonGroup(self)
        self._vista_group.setExclusive(True)
        self._vista_buttons = {}
        opciones_vista = [('hitrate', 'Hit Rate'), ('retorno', 'Retorno')]
        for i, (clave, texto) in enumerate(opciones_vista):
            if i > 0:
                sep = QLabel("|")
                sep.setObjectName("filtroLbl")
                fila2.addWidget(sep)
            btn = QPushButton(texto)
            btn.setObjectName("vista")
            btn.setCheckable(True)
            self._vista_group.addButton(btn, i)
            self._vista_buttons[clave] = btn
            fila2.addWidget(btn)
        self._vista_buttons['hitrate'].setChecked(True)
        self._vista_group.idClicked.connect(
            lambda i: self._refresh_stats(f"vista {self._vista_group.button(i).text()}"))
        self._vista_buttons['hitrate'].setToolTip(
            "Gráfico y columnas de la tabla: hit rate (¿acierta la dirección "
            "más de lo esperado?)")
        self._vista_buttons['retorno'].setToolTip(
            "Gráfico y columnas de la tabla: edge en pb (¿el acierto viene "
            "acompañado de un movimiento de precio relevante?)")
        fila2.addStretch()
        root.addLayout(fila2)

        # lambdas sin arg de señal: evita el problema de aridad con @_no_crash.
        # Cada una pasa un `motivo` legible (con el valor recién elegido, leído
        # en el momento del disparo) para que lbl_estado muestre "Recalculando
        # patrones — <motivo>…" mientras dura el refresco síncrono.
        self.cmb_er.currentIndexChanged.connect(
            lambda _i: self._refresh_stats(f"Régimen ER: {self.cmb_er.currentText()}"))
        self.cmb_hurst.currentIndexChanged.connect(
            lambda _i: self._refresh_stats(f"Régimen Hurst: {self.cmb_hurst.currentText()}"))
        self.cmb_sesion.currentIndexChanged.connect(
            lambda _i: self._refresh_stats(f"Sesión: {self.cmb_sesion.currentText()}"))
        self.chk_limpias.toggled.connect(
            lambda _v: self._refresh_stats("filtro de velas limpias"))
        self.cmb_lag.currentIndexChanged.connect(
            lambda _i: self._refresh_stats(f"Lag {self.cmb_lag.currentText()}"))
        self.cmb_periodo.currentIndexChanged.connect(
            lambda _i: self._refresh_stats(f"Periodo: {self.cmb_periodo.currentText()}"))

        # ── scroll: ranking + tarjetas ──
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        cont = QWidget()
        self._cont_lay = QVBoxLayout(cont)
        self._cont_lay.setContentsMargins(0, 0, 6, 0)
        self._cont_lay.setSpacing(8)

        self.tabla = QTableWidget(0, 9)
        self.tabla.setHorizontalHeaderLabels(
            ['Patrón', 'N', 'HR +1', 'HR +3', 'HR +5', 'HR +10',
             'Edge (pb)', 'p vs 50%', 'Signif.'])
        self.tabla.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        self.tabla.verticalHeader().setVisible(False)
        self.tabla.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.tabla.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.tabla.setAlternatingRowColors(True)
        self.tabla.setSortingEnabled(True)
        self._cont_lay.addWidget(self.tabla)

        self._cards = {}
        for nombre in PATRONES_INFO:
            card = PatternCard(nombre)
            card.clear_stats()
            card.setVisible(nombre in PATRONES_ORIGINALES)
            self._cards[nombre] = card
            self._cont_lay.addWidget(card)
        self._cont_lay.addStretch()

        scroll.setWidget(cont)
        root.addWidget(scroll, 1)
        self._ajustar_altura_tabla(len(PATRONES_ORIGINALES))

    def _ajustar_altura_tabla(self, filas):
        alto = self.tabla.horizontalHeader().height() \
            + self.tabla.verticalHeader().defaultSectionSize() * filas + 4
        self.tabla.setFixedHeight(alto)

    @_no_crash
    def _on_toggle_extra(self, checked):
        self._mostrar_extra = checked
        self.btn_mas_patrones.setText("Menos patrones ▲" if checked else "Más patrones ▾")
        for nombre, card in self._cards.items():
            if nombre not in PATRONES_ORIGINALES:
                card.setVisible(checked)
        self._refresh_stats("más patrones" if checked else "menos patrones")

    # ══════════════ carga ══════════════
    def set_source(self, csv_path, ticker=None, tf=None):
        """Llamado por TabAnalisis.load_results. No escanea todavía: el
        escaneo arranca la primera vez que la pestaña se hace visible."""
        self._csv_path = csv_path
        self._tf_nativo = tf or None
        if not csv_path or not os.path.exists(csv_path):
            self._payload = None
            self._cache_key = None
            self._tf_actual = None
            self._payloads_por_tf = {}
            self._raw = None
            self._tf_pendiente = None
            self._cola_precalculo = []
            self._tf_en_calculo = None
            self.lbl_estado.setText("Sin CSV asociado al análisis")
            for card in self._cards.values():
                card.clear_stats()
            for btn in self._tf_buttons.values():
                btn.setEnabled(False)
            self._btn_tf_custom.setEnabled(False)
            return

        try:
            key = (csv_path, os.path.getmtime(csv_path))
        except OSError:
            key = (csv_path, 0)
        if key == self._cache_key and self._payload is not None:
            return  # mismo archivo sin cambios: se conserva todo (caché de TF incluida)

        self._cache_key = key
        self._payload = None
        self._payloads_por_tf = {}
        self._raw = None
        self._tf_pendiente = None
        # cola de precálculo del activo anterior: se vacía; los hilos aún en
        # vuelo se descartan solos por el guard de csv_path en _on_resample_done
        self._cola_precalculo = []
        self._tf_en_calculo = None
        self._configurar_botones_tf()
        self._needs_scan = True
        self.lbl_estado.setText("Pendiente de escaneo…")
        for card in self._cards.values():
            card.clear_stats("Pendiente…")
        self.tabla.setRowCount(0)
        if self.isVisible():
            self._start_scan()

    def showEvent(self, ev):
        super().showEvent(ev)
        if self._needs_scan and self._csv_path:
            self._start_scan()

    def _actualizar_habilitacion_botones(self):
        """Habilita solo las TF >= a la nativa (no se puede bajar de
        granularidad); no toca cuál está seleccionada."""
        nat_min = tf_to_minutes(self._tf_nativo) if self._tf_nativo else None
        for lbl, btn in self._tf_buttons.items():
            m = tf_to_minutes(lbl)
            btn.setEnabled(nat_min is None or (m is not None and m >= nat_min))
        # el Custom valida su TF contra la nativa en el propio diálogo
        self._btn_tf_custom.setEnabled(True)

    def _configurar_botones_tf(self):
        """Habilita las TF válidas y preselecciona el botón de la TF nativa
        (llamado solo al cargar un activo nuevo)."""
        self._actualizar_habilitacion_botones()
        nat_min = tf_to_minutes(self._tf_nativo) if self._tf_nativo else None
        tf_sel = self._tf_nativo if self._tf_nativo in self._tf_buttons else None
        if tf_sel is None and nat_min is not None:
            for lbl in TF_LABELS_PATRONES:
                if tf_to_minutes(lbl) == nat_min:
                    tf_sel = lbl
                    break
        if tf_sel is None:
            tf_sel = next((l for l in TF_LABELS_PATRONES
                          if self._tf_buttons[l].isEnabled()), TF_LABELS_PATRONES[0])
        self._tf_group.blockSignals(True)
        self._tf_buttons[tf_sel].setChecked(True)
        self._tf_group.blockSignals(False)
        self._tf_actual = tf_sel
        self._configurar_combo_periodo()

    def _configurar_combo_periodo(self):
        """Deshabilita en cmb_periodo los bloques más finos que la TF nativa
        del archivo importado (no la TF activa en los botones de arriba —
        igual criterio que _actualizar_habilitacion_botones, fijado una sola
        vez al cargar el activo). Si el bloque seleccionado deja de ser
        válido, cae al primero habilitado."""
        nat_min = tf_to_minutes(self._tf_nativo) if self._tf_nativo else None
        modelo = self.cmb_periodo.model()
        primero_habilitado = None
        for i, (_etiqueta, _regla, minutos) in enumerate(PERIODOS_BARRAS):
            habilitado = nat_min is None or minutos >= nat_min
            modelo.item(i).setEnabled(habilitado)
            if habilitado and primero_habilitado is None:
                primero_habilitado = i
        if primero_habilitado is not None \
                and not modelo.item(self.cmb_periodo.currentIndex()).isEnabled():
            self.cmb_periodo.setCurrentIndex(primero_habilitado)

    def _start_scan(self):
        self._needs_scan = False
        self.lbl_estado.setText("Analizando patrones…")
        for btn in self._tf_buttons.values():
            btn.setEnabled(False)
        self._btn_tf_custom.setEnabled(False)
        th = _PatternScanThread(self._csv_path, parent=self)
        th.computed.connect(self._on_scan_done)
        th.finished.connect(lambda t=th: self._on_thread_finished(t))
        self._threads.append(th)
        th.start()

    @_no_crash
    def _on_thread_finished(self, th):
        if th in self._threads:
            self._threads.remove(th)
        th.deleteLater()

    def _aplicar_payload_combos(self, payload):
        self.cmb_er.setEnabled(payload['tiene_er'])
        self.cmb_er.setToolTip("" if payload['tiene_er']
                               else "Esta temporalidad no tiene columna ER")
        if not payload['tiene_er']:
            self.cmb_er.setCurrentIndex(0)
        # los umbrales de ER son dinámicos (media±1σ del ER de ESTE activo/TF,
        # ver core/metrics.calcular_umbrales_er) — a diferencia de Hurst, que
        # usa umbrales fijos (0.52/0.58) ya visibles en las etiquetas del
        # combo, aquí hace falta mostrar el valor real calculado.
        umbrales = payload['ctx'].get('umbrales_er')
        if umbrales:
            u_ruido, u_tendencia = umbrales
            self.cmb_er.setItemText(1, f"Tendencia (ER > {u_tendencia:.2f})")
            self.cmb_er.setItemText(2, "Neutro")
            self.cmb_er.setItemText(3, f"Ruido (ER < {u_ruido:.2f})")
        else:
            self.cmb_er.setItemText(1, "Tendencia (ER alto)")
            self.cmb_er.setItemText(2, "Neutro")
            self.cmb_er.setItemText(3, "Ruido (ER bajo)")
        self.cmb_hurst.setEnabled(payload['tiene_hurst'])
        self.cmb_hurst.setToolTip("" if payload['tiene_hurst']
                                  else "Esta temporalidad no tiene columna hurst")
        if not payload['tiene_hurst']:
            self.cmb_hurst.setCurrentIndex(0)

        # una vela de 1d/1w no tiene una "hora" única representativa de
        # sesión intradía: el filtro solo tiene sentido en TF menores a 1 día
        minutos_tf = tf_to_minutes(self._tf_actual) if self._tf_actual else None
        intradia = minutos_tf is not None and minutos_tf < 1440
        self.cmb_sesion.setEnabled(intradia)
        self.cmb_sesion.setToolTip("" if intradia
                                   else "No aplica a temporalidades de 1 día o más")
        if not intradia:
            self.cmb_sesion.setCurrentIndex(0)  # Globex

    @_no_crash
    def _on_scan_done(self, path, payload):
        if path != self._csv_path:
            return  # resultado obsoleto: el activo cambió durante el escaneo
        self._actualizar_habilitacion_botones()
        if 'error' in payload:
            self.lbl_estado.setText(f"Error leyendo el CSV: {payload['error']}")
            for card in self._cards.values():
                card.clear_stats("Error")
            return
        self._raw = payload.pop('raw', None)
        self._payload = payload
        # el escaneo inicial siempre corresponde a self._tf_actual (fijado
        # en _configurar_botones_tf antes de arrancar el hilo)
        self._payloads_por_tf[self._tf_actual] = payload

        self._aplicar_payload_combos(payload)
        self._refresh_stats()
        # precalcular en segundo plano las TFs hermanas de la nativa, para
        # que el cambio dentro de la familia sea instantáneo
        self._encolar_familia(self._tf_actual)
        self._lanzar_siguiente()

    # ══════════════ cambio de temporalidad + cola de precálculo ══════════════
    def _encolar_familia(self, tf_label):
        """Añade a la cola las TFs hermanas de tf_label (misma familia) que
        estén habilitadas, no cacheadas, no en cola y no en vuelo."""
        for tf in _familia_de_tf(tf_label):
            btn = self._tf_buttons.get(tf)
            if (btn is not None and btn.isEnabled()
                    and tf not in self._payloads_por_tf
                    and tf not in self._cola_precalculo
                    and tf != self._tf_en_calculo):
                self._cola_precalculo.append(tf)

    def _lanzar_siguiente(self):
        """Lanza el siguiente resample de la cola si no hay ninguno en vuelo.
        Un solo hilo a la vez: el precálculo es oportunista, no debe competir
        por CPU con lo que el usuario esté mirando."""
        if self._tf_en_calculo is not None or not self._cola_precalculo \
                or self._raw is None:
            return
        tf_label = self._cola_precalculo.pop(0)
        if tf_label in self._payloads_por_tf:
            self._lanzar_siguiente()
            return
        self._tf_en_calculo = tf_label
        th = _ResampleThread(self._csv_path, tf_label,
                             _regla_de_tf(tf_label), self._raw, parent=self)
        th.computed.connect(self._on_resample_done)
        th.finished.connect(lambda t=th: self._on_thread_finished(t))
        self._threads.append(th)
        th.start()

    @_no_crash
    def _on_tf_clicked(self, btn_id):
        if btn_id == len(TF_LABELS_PATRONES):
            self._on_tf_custom_clicked()
            return
        self._seleccionar_tf(TF_LABELS_PATRONES[btn_id])

    def _restaurar_boton_tf(self, tf_label):
        """Re-marca el botón correspondiente a tf_label (al cancelar o fallar
        el diálogo de TF custom, que ya dejó marcado el botón Custom)."""
        btn = self._tf_buttons.get(tf_label)
        if btn is None and tf_label and tf_label == self._tf_custom:
            btn = self._btn_tf_custom
        if btn is not None:
            self._tf_group.blockSignals(True)
            btn.setChecked(True)
            self._tf_group.blockSignals(False)

    @_no_crash
    def _on_tf_custom_clicked(self):
        previo = self._tf_actual
        texto, ok = QInputDialog.getText(
            self, "Temporalidad personalizada",
            "Temporalidad (p.ej. 20m, 90m, 2h, 3d, 5d, 2w):",
            text=self._tf_custom or "")
        if not ok:
            self._restaurar_boton_tf(previo)
            return
        tf_label = _parsear_tf_custom(texto)
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
            # coincide con una preset: seleccionarla como si se hubiera pulsado
            self._restaurar_boton_tf(tf_label)
            self._seleccionar_tf(tf_label)
            return
        self._tf_custom = tf_label
        self._btn_tf_custom.setText(f"Custom: {tf_label}")
        self._seleccionar_tf(tf_label)

    def _seleccionar_tf(self, tf_label):
        self._tf_actual = tf_label
        self._encolar_familia(tf_label)   # precalcula las hermanas después
        if tf_label in self._payloads_por_tf:
            self._payload = self._payloads_por_tf[tf_label]
            self._aplicar_payload_combos(self._payload)
            self._refresh_stats()
            self._lanzar_siguiente()
            return
        if self._raw is None:
            return  # el escaneo inicial aún no ha terminado
        self._tf_pendiente = tf_label
        self.lbl_estado.setText(f"Recalculando patrones en {tf_label}…")
        # prioridad: la TF pedida por el usuario salta al frente de la cola
        # (si justo es la que está en vuelo, basta con esperar su resultado)
        if tf_label != self._tf_en_calculo:
            if tf_label in self._cola_precalculo:
                self._cola_precalculo.remove(tf_label)
            self._cola_precalculo.insert(0, tf_label)
        self._lanzar_siguiente()

    @_no_crash
    def _on_resample_done(self, path, tf_label, payload):
        if path != self._csv_path:
            return  # obsoleto: el activo cambió durante el cálculo
        self._tf_en_calculo = None
        if 'error' in payload:
            if tf_label == self._tf_pendiente:
                self._tf_pendiente = None
                self.lbl_estado.setText(
                    f"Error recalculando en {tf_label}: {payload['error']}")
            self._lanzar_siguiente()
            return
        # el resultado se cachea SIEMPRE (venga de un click o del precálculo)
        self._payloads_por_tf[tf_label] = payload
        if tf_label == self._tf_pendiente:
            self._tf_pendiente = None
            self._payload = payload
            self._aplicar_payload_combos(payload)
            self._refresh_stats()
        self._lanzar_siguiente()

    # ══════════════ estadística + refresco ══════════════
    @_no_crash
    def _refresh_stats(self, motivo=None):
        if self._payload is None:
            return
        p = self._payload
        if motivo:
            # el refresco es síncrono en el hilo de la GUI: sin el repaint()
            # inmediato el texto no llegaría a pintarse hasta el final y el
            # mensaje intermedio jamás se vería
            self.lbl_estado.setText(f"Recalculando patrones — {motivo}…")
            self.lbl_estado.repaint()
        filtro_er = self.cmb_er.currentData()
        filtro_hurst = self.cmb_hurst.currentData()
        filtro_sesion = self.cmb_sesion.currentData()
        solo_limpias = self.chk_limpias.isChecked()
        lag_rank = self.cmb_lag.currentData() or LAGS[0]
        vista = 'retorno' if self._vista_buttons['retorno'].isChecked() else 'hitrate'

        # lo único O(n_velas) del refresco: compartido por los 13 patrones
        base = preparar_base_filtro(p['ctx'], filtro_er=filtro_er,
                                    filtro_hurst=filtro_hurst,
                                    solo_limpias=solo_limpias,
                                    filtro_sesion=filtro_sesion)
        nombres_visibles = list(PATRONES_INFO) if self._mostrar_extra \
            else [n for n in PATRONES_INFO if n in PATRONES_ORIGINALES]

        regla_periodo = self.cmb_periodo.currentData()
        # ancho de barra fijo según el periodo elegido (no según el hueco
        # real entre bloques que sobrevivieron al filtro de mínimo de
        # ocurrencias en agregar_por_periodo — si un bloque intermedio se
        # descarta por tener pocas ocurrencias, ese hueco se agranda y un
        # ancho basado en "espaciado medio" queda sobredimensionado, solapando
        # las barras que sí están espaciadas con normalidad)
        _minutos_periodo = PERIODOS_BARRAS[self.cmb_periodo.currentIndex()][2]
        ancho_dias = (_minutos_periodo / 1440.0) * 0.8

        resultados = {}
        total_occ = 0
        for nombre in nombres_visibles:
            occ = p['patrones'][nombre]
            stats = calcular_stats_patron(
                occ, p['ctx'], filtro_er=filtro_er, filtro_hurst=filtro_hurst,
                solo_limpias=solo_limpias,
                velas_formacion=PATRONES_INFO[nombre]['velas'], base=base,
                filtro_sesion=filtro_sesion)
            resultados[nombre] = stats
            total_occ += stats['n_total']
            s_lag = stats['por_lag'][lag_rank]
            barras = agregar_por_periodo(
                s_lag['idx'], s_lag['dir'], s_lag['aciertos'], s_lag['signed_ret'],
                p['timestamps'], regla_periodo, s_lag['ret_base'] or 0.0)
            self._cards[nombre].update_stats(stats, p['timestamps'], vista, lag_rank,
                                             barras, ancho_dias)

        self.lbl_estado.setText(
            f"{p['n_velas']:,} velas · {total_occ:,} ocurrencias con el filtro actual")

        # ── ranking ──
        # las 4 columnas por lag pivotan con el toggle Hit Rate/Retorno:
        # lo que ves en la tabla es la misma métrica que eliges arriba
        etiquetas_lag = [f"HR +{lag}" for lag in LAGS] if vista == 'hitrate' \
            else [f"Retorno +{lag} (pb)" for lag in LAGS]
        self.tabla.setHorizontalHeaderLabels(
            ['Patrón', 'N', *etiquetas_lag, 'Edge (pb)', 'p vs 50%', 'Signif.'])

        self.tabla.setSortingEnabled(False)
        self.tabla.setRowCount(len(resultados))
        for row, (nombre, stats) in enumerate(resultados.items()):
            item = QTableWidgetItem(nombre)
            self.tabla.setItem(row, 0, item)

            def num_item(valor, texto, color=AX_FG):
                it = QTableWidgetItem()
                it.setData(Qt.ItemDataRole.DisplayRole,
                           valor if valor is not None else -1.0)
                it.setText(texto)
                it.setForeground(QColor(color))
                it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                return it

            self.tabla.setItem(row, 1, num_item(
                float(stats['n_total']), str(stats['n_total'])))
            for j, lag in enumerate(LAGS):
                s = stats['por_lag'][lag]
                if vista == 'hitrate':
                    hr = s['hit_rate']
                    if hr is None:
                        self.tabla.setItem(row, 2 + j, num_item(None, "—", GRIS))
                    else:
                        c = VERDE if hr > 0.55 else AMBAR if hr >= 0.5 else ROJO
                        self.tabla.setItem(row, 2 + j,
                                           num_item(hr, f"{hr * 100:.1f}%", c))
                else:
                    if s['edge'] is None:
                        self.tabla.setItem(row, 2 + j, num_item(None, "—", GRIS))
                    else:
                        bp = s['edge'] * 10000
                        self.tabla.setItem(row, 2 + j, num_item(
                            bp, f"{bp:+.1f}", VERDE if bp > 0 else ROJO))
            s = stats['por_lag'][lag_rank]
            if s['edge'] is None:
                self.tabla.setItem(row, 6, num_item(None, "—", GRIS))
            else:
                bp = s['edge'] * 10000
                self.tabla.setItem(row, 6, num_item(
                    bp, f"{bp:+.1f}", VERDE if bp > 0 else ROJO))
            if s['p_vs_50'] is None:
                self.tabla.setItem(row, 7, num_item(None, "—", GRIS))
            else:
                item_p = num_item(
                    1.0 - s['p_vs_50'], _fmt_pvalue(s['p_vs_50']),
                    VERDE if s['p_vs_50'] < 0.05 else AX_FG)
                if s['p_vs_50'] < 0.001:
                    item_p.setToolTip(_fmt_pvalue_exacto(s['p_vs_50']))
                self.tabla.setItem(row, 7, item_p)
            # el test es de dos colas: "significativo" puede ser un hit rate
            # fiablemente POR DEBAJO del 50% (el patrón falla más de lo
            # esperado), no solo por encima — hay que distinguirlo o un
            # checkmark verde parecería indicar que el patrón funciona bien
            positivo = s['hit_rate'] is not None and s['hit_rate'] > 0.5
            if s['significativo'] and positivo:
                self.tabla.setItem(row, 8, num_item(2.0, "✔", VERDE))
            elif s['significativo']:
                self.tabla.setItem(row, 8, num_item(1.0, "⚠ en contra", ROJO))
            else:
                self.tabla.setItem(row, 8, num_item(0.0, "—", GRIS))
        self.tabla.setSortingEnabled(True)
        self._ajustar_altura_tabla(len(resultados))
