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
import inspect
import json
import os

import numpy as np
import pandas as pd
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QDate, QSize
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QCheckBox,
    QScrollArea, QFrame, QTableWidget, QTableWidgetItem, QHeaderView,
    QPushButton, QSplitter, QLineEdit, QSpinBox, QDoubleSpinBox, QSlider,
    QListWidget, QListWidgetItem, QTabWidget, QFormLayout, QGroupBox,
    QInputDialog, QDateEdit, QSizePolicy, QApplication, QDialog, QProgressBar,
    QButtonGroup, QStyledItemDelegate, QStyle, QStackedWidget,
)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from matplotlib.widgets import RangeSlider
from matplotlib.dates import date2num, num2date
from matplotlib.collections import PolyCollection, LineCollection
from matplotlib.patches import FancyArrowPatch

from core.config import (
    LIMPIADOS_DIR, TF_PATTERN, tf_to_minutes, tipo_activo_de_csv,
    PRESETS_FRICCION, TIPO_MAP,
)
from core.backtest import (
    simular, dividir_is_oos, calcular_metricas, montecarlo, MOTIVOS_SALIDA,
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
    sma, ema, rsi, atr, bollinger,
)
from core.candle_patterns import detectar_patrones
from gui.widgets.file_explorer import FileExplorer
from gui.widgets.tf_common import TF_LABELS, parsear_tf_custom, regla_de_tf
from gui.widgets.lwc_chart import LwcChart, WEBENGINE_OK

FIG_BG = '#0d1424'
AX_FG = '#c8d6e5'
GRID_C = '#253a60'
VERDE = '#2ecc71'
ROJO = '#e74c3c'
GRIS = '#5a7a9a'
AMBAR = '#f1c40f'
AZUL = '#4fc3f7'

# flechas de operaciones: tonos más saturados que las velas (VERDE/ROJO de
# arriba) para que los marcadores no se camuflen sobre cuerpos del mismo color
VERDE_FLECHA = '#00e676'   # compra (abre largo / cierra corto)
ROJO_FLECHA = '#ff1744'    # venta (abre corto / cierra largo)

# color fijo por periodo para las medias (SMA/EMA) dibujadas en el gráfico de
# Resultados — así una media de 200 siempre se identifica por su color sin
# importar el orden en que aparezcan los demás setups/filtros.
COLOR_MEDIA_FIJO = {20: '#2B7FFF', 50: '#FF8904', 200: '#800000'}

MODOS_WFA = [
    'Retorno %',
    'Retorno acumulado',
    'Max DD %',
    'Win rate %',
    'Retorno vs Max DD',
    'Trades',
]

MODOS_EQUITY = [
    'Retorno %',
    'Capital',
    'Log-retorno',
    'Drawdown',
]

RUTA_ESTRATEGIAS_GUARDADAS = os.path.join(LIMPIADOS_DIR, 'backtest_estrategias.json')
_TIPO_MAP_INV = {v: k for k, v in TIPO_MAP.items()}   # 'CRYPTO' -> 'Crypto'

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
QLabel:disabled { color: #33465e; }
QToolTip {
    background-color: #101a2e; color: #c8d6e5;
    border: 1px solid #253a60; border-radius: 4px;
    padding: 6px; font-size: 11px;
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


def _style_ax(ax):
    ax.set_facecolor(FIG_BG)
    ax.tick_params(colors=AX_FG, labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_C)
    ax.grid(True, alpha=0.25, color=GRID_C, linewidth=0.5)


def _fmt(v, dec=2, sufijo=''):
    if v is None:
        return '—'
    if v == float('inf'):
        return '∞'
    return f"{v:.{dec}f}{sufijo}"


def _decimar_ohlc(x, o, h, l, c, x0, x1, max_velas=2500):
    """Recorta a la ventana visible [x0, x1] y, si hay más de max_velas
    puntos, los agrega en bloques (open=primero, high=max, low=min,
    close=último) para que el nº de artistas dibujados quede acotado sin
    importar la resolución total de la serie ni el nivel de zoom."""
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
    n_bins = int(np.ceil(n / paso))
    pad = n_bins * paso - n
    if pad:
        xs = np.pad(xs, (0, pad), mode='edge')
        os_ = np.pad(os_, (0, pad), mode='edge')
        hs = np.pad(hs, (0, pad), mode='edge')
        ls = np.pad(ls, (0, pad), mode='edge')
        cs = np.pad(cs, (0, pad), mode='edge')
    idx_starts = np.arange(0, n_bins * paso, paso)
    idx_ends = idx_starts + paso - 1
    x_bin = xs[idx_starts]
    o_bin = os_[idx_starts]
    h_bin = np.maximum.reduceat(hs, idx_starts)
    l_bin = np.minimum.reduceat(ls, idx_starts)
    c_bin = cs[idx_ends]
    return x_bin, o_bin, h_bin, l_bin, c_bin


def _cargar_ohlc(csv_path, regla_resample=None):
    """Lee timestamp/open/high/low/close de un CSV limpiado y, si se pide,
    lo reagrega a una temporalidad más gruesa (misma agregación que
    tab_patrones._ResampleThread). Compartido por _BacktestThread y
    _OptimizerThread."""
    cols = ['timestamp', 'open', 'high', 'low', 'close']
    try:
        df = pd.read_csv(csv_path, usecols=cols, engine='pyarrow')
    except (ImportError, ValueError):
        df = pd.read_csv(csv_path, usecols=lambda c: c in cols)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp', 'close']).sort_values('timestamp')
    df = df.reset_index(drop=True)
    if regla_resample:
        agg = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}
        df = (df.set_index('timestamp')[['open', 'high', 'low', 'close']]
                .resample(regla_resample).agg(agg)
                .dropna(subset=['close'])
                .reset_index())
    return df


def _velas_por_anio(df):
    """Mediana del paso temporal -> nº de velas/año, para anualizar retorno
    y Sharpe."""
    difs = df['timestamp'].diff().dropna().dt.total_seconds() / 60.0
    minutos_vela = float(difs.median()) if len(difs) else 60.0
    return 525600.0 / max(minutos_vela, 1e-9)


# ══════════════ hilo de backtest ══════════════
class _BacktestThread(QThread):
    """Lee el CSV, genera las señales del SISTEMA (multi-setup) y corre
    motor + métricas + WFA + MC. IMPORTANTE: parentar al widget (destruir un
    QThread corriendo aborta el proceso, misma nota que tab_patrones)."""
    computed = pyqtSignal(object)

    def __init__(self, csv_path, setups, config_global, pct_oos,
                 wfa_activo, wfa_ventanas, codigo='', tf_label=None,
                 regla_resample=None, parent=None):
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

    def run(self):
        try:
            df = _cargar_ohlc(self._csv, self._regla_resample)
            n = len(df)
            if n < 50:
                self.computed.emit({'error': f'Serie demasiado corta ({n} velas)'})
                return
            if not self._setups:
                self.computed.emit({'error': 'El sistema no tiene ningún setup'})
                return

            senales = generar_senales_sistema(df, self._setups)

            # config del motor: cuenta global + riesgo/stop/TP/tiempo por setup
            config = dict(self._cfg_global)
            config_por_setup = {}
            for k, setup in enumerate(self._setups):
                salida_n = int(setup.get('salida_n_velas', 0))
                if setup['plantilla'] == 'Patrones de velas' and salida_n == 0:
                    salida_n = int(setup.get('params', {}).get('lag_salida', 5))
                config_por_setup[k] = {
                    'riesgo_pct': float(setup.get('riesgo_pct', 0.01)),
                    'stop_atr': float(setup.get('stop_atr', 0.0)),
                    'tp_r': float(setup.get('tp_r', 0.0)),
                    'salida_n_velas': salida_n,
                }
            config['config_por_setup'] = config_por_setup

            resultado = simular(df['open'].values, df['high'].values,
                                df['low'].values, df['close'].values,
                                senales, config)

            velas_anio = _velas_por_anio(df)

            corte = dividir_is_oos(n, self._pct_oos)
            metricas = {
                'IS': calcular_metricas(resultado, 0, corte, velas_anio),
                'OOS': calcular_metricas(resultado, corte, n, velas_anio),
                'Total': calcular_metricas(resultado, 0, n, velas_anio),
            }

            # métricas por setup (valida si cada forma de entrada aporta)
            tr = resultado['trades']
            metricas_setup = []
            for k, setup in enumerate(self._setups):
                m = tr['setup'] == k
                pnl = tr['pnl'][m]
                ret = tr['ret_pct'][m]
                metricas_setup.append({
                    'nombre': setup.get('nombre') or setup['plantilla'],
                    'riesgo_pct': float(setup.get('riesgo_pct', 0.01)),
                    'n_trades': int(m.sum()),
                    'win_rate': float((pnl > 0).mean()) if len(pnl) else None,
                    'pnl_total': float(pnl.sum()),
                    'expectancy_pct': float(ret.mean()) * 100.0 if len(ret) else None,
                })

            wfa = None
            if self._wfa and self._wfa_n >= 2:
                bordes = np.linspace(0, n, self._wfa_n + 1).astype(int)
                ventanas = []
                for k in range(self._wfa_n):
                    met = calcular_metricas(resultado, bordes[k], bordes[k + 1],
                                            velas_anio)
                    met['idx_ini'] = int(bordes[k])
                    met['idx_fin'] = int(bordes[k + 1])
                    ventanas.append(met)
                wfa = ventanas

            mc = montecarlo(resultado['trades'],
                            config.get('capital_inicial', 10000.0),
                            n_sims=1000, semilla=1234)

            c = df['close'].values
            self.computed.emit({
                'timestamps': df['timestamp'].values,
                'open': df['open'].values,
                'high': df['high'].values,
                'low': df['low'].values,
                'close': c,
                'codigo': self._codigo,
                'log_ret_acum': np.log(c / c[0]),
                'resultado': resultado,
                'metricas': metricas,
                'metricas_setup': metricas_setup,
                'nombres_setup': [s.get('nombre') or s['plantilla']
                                  for s in self._setups],
                'corte': corte,
                'n_velas': n,
                'wfa': wfa,
                'montecarlo': mc,
                'estrategia': ' + '.join(s.get('nombre') or s['plantilla']
                                         for s in self._setups),
                'csv': self._csv,
                'config': config,
                'setups': self._setups,
                'tf': self._tf_label,
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
    terminado = pyqtSignal(object)

    def __init__(self, csv_path, setup_base, sweep_params, sweep_riesgo,
                 config_global, pct_oos, metrica, tf_label=None,
                 regla_resample=None, parent=None):
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

    def run(self):
        try:
            df = _cargar_ohlc(self._csv, self._regla_resample)
            n = len(df)
            if n < 50:
                self.terminado.emit({'error': f'Serie demasiado corta ({n} velas)'})
                return
            velas_anio = _velas_por_anio(df)
            resultados = optimizar_setup(
                df, self._setup_base, self._sweep_params, self._sweep_riesgo,
                self._cfg_global, self._pct_oos, metrica=self._metrica,
                velas_por_anio=velas_anio,
                progreso_cb=lambda i, total: self.progreso.emit(i, total))
            self.terminado.emit({
                'resultados': resultados,
                'metrica': self._metrica,
                'setup_base': self._setup_base,
                'csv': self._csv,
                'tf': self._tf_label,
            })
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.terminado.emit({'error': str(e)})


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


def _spec_regla(tipo, num):
    if tipo == 'Valor':
        return {'tipo': 'valor', 'valor': num}
    if tipo in ('close', 'open', 'high', 'low'):
        return {'tipo': tipo}
    return {'tipo': tipo, 'periodo': int(num)}


def _acumular_indicador_spec(spec, mas, rsis, atrs, bbs):
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


class EditorReglas(QGroupBox):
    """Tabla de condiciones: cada fila es una condición; las filas con la
    misma (regla, setup) se combinan con AND; setups distintos con OR."""
    _REGLAS = ['Entrada Long', 'Salida Long', 'Entrada Short', 'Salida Short']
    _CLAVES = {'Entrada Long': 'entradas_long', 'Salida Long': 'salidas_long',
               'Entrada Short': 'entradas_short', 'Salida Short': 'salidas_short'}

    def __init__(self, parent=None):
        super().__init__("Reglas custom (mismo setup = AND, setups distintos = OR)", parent)
        lay = QVBoxLayout(self)
        self.tabla = QTableWidget(0, 7)
        self.tabla.setHorizontalHeaderLabels(
            ['Regla', 'Setup', 'Indicador', 'Periodo', 'Operador',
             'Comparar con', 'Valor/Periodo'])
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
        self.tabla.setCellWidget(r, 2, _combo_regla(_INDICADORES_REGLA, d.get('izq', 'close')))
        self.tabla.setCellWidget(r, 3, _spin_regla(d.get('izq_periodo', 14), 1, 5000))
        self.tabla.setCellWidget(r, 4, _combo_regla(_OPERADORES_REGLA, d.get('op', '>')))
        self.tabla.setCellWidget(r, 5, _combo_regla(['Valor'] + _INDICADORES_REGLA,
                                                    d.get('der', 'Valor')))
        self.tabla.setCellWidget(r, 6, _spin_regla(d.get('der_valor', 0.0), dec=4))

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
            cond = {'izq': _spec_regla(izq, izq_per), 'op': op,
                    'der': _spec_regla(der, der_val)}
            grupos.setdefault((self._CLAVES[regla], setup), []).append(cond)
        out = {'entradas_long': [], 'salidas_long': [],
               'entradas_short': [], 'salidas_short': []}
        for (clave, setup), condiciones in grupos.items():
            out[clave].append({'setup_id': setup, 'condiciones': condiciones})
        return out

    def cargar_reglas(self, reglas):
        self.tabla.setRowCount(0)
        inversa = {v: k for k, v in self._CLAVES.items()}
        for clave, setups in (reglas or {}).items():
            for setup in setups:
                for cond in setup.get('condiciones', []):
                    izq, der = cond['izq'], cond['der']
                    self._add_fila(datos={
                        'regla': inversa.get(clave, 'Entrada Long'),
                        'setup': setup.get('setup_id', 0),
                        'izq': izq['tipo'], 'izq_periodo': izq.get('periodo', 14),
                        'op': cond['op'],
                        'der': 'Valor' if der['tipo'] == 'valor' else der['tipo'],
                        'der_valor': der.get('valor', der.get('periodo', 0.0)),
                    })


class EditorCondiciones(QGroupBox):
    """Tabla de condiciones planas (todas AND entre sí) — versión simplificada
    de EditorReglas sin agrupación por regla/setup: se usa como filtro extra
    de un setup (entrada o salida), aplicable sobre CUALQUIER plantilla, no
    solo 'Custom (reglas)'."""

    def __init__(self, titulo, parent=None):
        super().__init__(titulo, parent)
        lay = QVBoxLayout(self)
        self.tabla = QTableWidget(0, 6)
        self.tabla.setHorizontalHeaderLabels(
            ['Indicador', 'Periodo', 'Operador', 'Comparar con', 'Valor/Periodo',
             'Dirección'])
        self.tabla.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.tabla.verticalHeader().setVisible(False)
        self.tabla.setMinimumHeight(90)
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
        self.tabla.setCellWidget(r, 0, _combo_regla(_INDICADORES_REGLA, d.get('izq', 'close')))
        self.tabla.setCellWidget(r, 1, _spin_regla(d.get('izq_periodo', 14), 1, 5000))
        self.tabla.setCellWidget(r, 2, _combo_regla(_OPERADORES_REGLA, d.get('op', '>')))
        self.tabla.setCellWidget(r, 3, _combo_regla(['Valor'] + _INDICADORES_REGLA,
                                                    d.get('der', 'Valor')))
        self.tabla.setCellWidget(r, 4, _spin_regla(d.get('der_valor', 0.0), dec=4))
        cmb_dir = _combo_regla(list(_MAPA_DIRECCION), d.get('direccion', 'Ambas'))
        cmb_dir.setToolTip(
            "A qué lado se aplica esta condición: 'Ambas' la exige tanto para "
            "entradas/salidas long como short (comportamiento clásico); "
            "'Long'/'Short' la restringe solo a ese lado, sin afectar al otro "
            "— p.ej. close > SMA(200) en Long y close < SMA(200) en Short "
            "dentro del mismo setup.")
        self.tabla.setCellWidget(r, 5, cmb_dir)

    def _del_fila(self):
        r = self.tabla.currentRow()
        if r >= 0:
            self.tabla.removeRow(r)
        elif self.tabla.rowCount():
            self.tabla.removeRow(self.tabla.rowCount() - 1)

    def condiciones(self):
        """Lista plana de condiciones (AND entre sí dentro de su misma
        dirección; ver 'direccion' de cada una)."""
        out = []
        for r in range(self.tabla.rowCount()):
            izq = self.tabla.cellWidget(r, 0).currentText()
            izq_per = self.tabla.cellWidget(r, 1).value()
            op = self.tabla.cellWidget(r, 2).currentText()
            der = self.tabla.cellWidget(r, 3).currentText()
            der_val = self.tabla.cellWidget(r, 4).value()
            direccion = _MAPA_DIRECCION[self.tabla.cellWidget(r, 5).currentText()]
            out.append({'izq': _spec_regla(izq, izq_per), 'op': op,
                        'der': _spec_regla(der, der_val), 'direccion': direccion})
        return out

    def cargar_condiciones(self, condiciones):
        self.tabla.setRowCount(0)
        for cond in (condiciones or []):
            izq, der = cond['izq'], cond['der']
            self._add_fila(datos={
                'izq': izq['tipo'], 'izq_periodo': izq.get('periodo', 14),
                'op': cond['op'],
                'der': 'Valor' if der['tipo'] == 'valor' else der['tipo'],
                'der_valor': der.get('valor', der.get('periodo', 0.0)),
                'direccion': _MAPA_DIRECCION_INV.get(cond.get('direccion', 'ambas'), 'Ambas'),
            })


# ══════════════ sub-pestaña Optimizador ══════════════
def _setup_por_defecto(plantilla='Cruce de medias'):
    s = {'nombre': 'Setup 1', 'plantilla': plantilla,
         'params': params_por_defecto(plantilla),
         'riesgo_pct': 0.01, 'stop_atr': 2.0, 'tp_r': 0.0,
         'salida_n_velas': 0, 'edge': False,
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
_MAPA_SESION = {
    'Ninguna': 'ninguna', 'Overnight': 'overnight', 'Londres': 'londres',
    'NY': 'ny', 'Personalizada': 'personalizada',
}
_MAPA_REGIMEN_INV = {v: k for k, v in _MAPA_REGIMEN.items()}
_MAPA_SESION_INV = {v: k for k, v in _MAPA_SESION.items()}

# dirección de una condición de filtro (EditorCondiciones): a qué lado(s) se
# aplica. 'Ambas' es el valor por defecto y restringe long y short por igual
# (comportamiento previo a esta columna, 100% compatible con setups guardados
# sin el campo 'direccion').
_MAPA_DIRECCION = {'Ambas': 'ambas', 'Long': 'long', 'Short': 'short'}
_MAPA_DIRECCION_INV = {v: k for k, v in _MAPA_DIRECCION.items()}


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

        # ── selector de sistema (predeterminados / guardados / nuevo) ──
        grp_sel = QGroupBox("Sistema")
        lay_sel = QVBoxLayout(grp_sel)
        fila_sel = QHBoxLayout()
        self.cmb_sistema = QComboBox()
        # activated (no currentIndexChanged): permite re-seleccionar el
        # mismo sistema para recargar sus valores por defecto
        self.cmb_sistema.activated.connect(self._on_sistema_elegido)
        btn_guardar = QPushButton("Guardar sistema…")
        btn_guardar.clicked.connect(self._guardar_sistema)
        fila_sel.addWidget(self.cmb_sistema, 1)
        fila_sel.addWidget(btn_guardar)
        lay_sel.addLayout(fila_sel)
        lay.addWidget(grp_sel)

        # ── sistema: lista de setups ──
        grp_sis = QGroupBox("Setups del sistema (cada uno con su propio riesgo)")
        lay_sis = QVBoxLayout(grp_sis)
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

        # ── editor del setup seleccionado ──
        self.grp_setup = QGroupBox("Setup seleccionado")
        f_set = QFormLayout(self.grp_setup)
        self.txt_nombre = QLineEdit()
        self.txt_nombre.textEdited.connect(self._guardar_setup_actual)
        f_set.addRow("Nombre:", self.txt_nombre)
        self.cmb_plantilla = QComboBox()
        self.cmb_plantilla.addItems(list(ESTRATEGIAS))
        self.cmb_plantilla.currentTextChanged.connect(self._on_plantilla_changed)
        f_set.addRow("Plantilla:", self.cmb_plantilla)

        # parámetros autogenerados de la plantilla
        self.form_params_cont = QWidget()
        self.form_params = QFormLayout(self.form_params_cont)
        self.form_params.setContentsMargins(0, 0, 0, 0)
        f_set.addRow(self.form_params_cont)

        self.editor_reglas = EditorReglas()
        self.editor_reglas.setVisible(False)
        self.editor_reglas.tabla.cellChanged.connect(
            lambda *_: self._guardar_setup_actual())
        f_set.addRow(self.editor_reglas)

        # definición visible de la plantilla (reglas interpoladas)
        self.lbl_definicion = QLabel("")
        self.lbl_definicion.setObjectName("campo")
        self.lbl_definicion.setWordWrap(True)
        self.lbl_definicion.setStyleSheet(
            "color: #8fb3d9; font-size: 11px; background-color: #101a2e;"
            "border: 1px solid #253a60; border-radius: 4px; padding: 6px;")
        f_set.addRow("Definición:", self.lbl_definicion)

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
        self.sp_riesgo.setToolTip(
            "Riesgo de ESTE setup: % del equity que se pierde si salta su "
            "stop (o contra 2×ATR de referencia si no tiene stop)")
        self.sp_riesgo.valueChanged.connect(self._guardar_setup_actual)
        f_set.addRow("Riesgo del setup:", self.sp_riesgo)
        self.sp_stop = QDoubleSpinBox()
        self.sp_stop.setRange(0, 20)
        self.sp_stop.setDecimals(1)
        self.sp_stop.setValue(2.0)
        self.sp_stop.setSpecialValueText("Sin stop")
        self.sp_stop.valueChanged.connect(self._guardar_setup_actual)
        f_set.addRow("Stop (× ATR):", self.sp_stop)
        self.sp_tp = QDoubleSpinBox()
        self.sp_tp.setRange(0, 20)
        self.sp_tp.setDecimals(1)
        self.sp_tp.setValue(0.0)
        self.sp_tp.setSpecialValueText("Sin TP")
        self.sp_tp.valueChanged.connect(self._guardar_setup_actual)
        f_set.addRow("Take-profit (R):", self.sp_tp)
        self.sp_tiempo = QSpinBox()
        self.sp_tiempo.setRange(0, 10000)
        self.sp_tiempo.setValue(0)
        self.sp_tiempo.setSpecialValueText("Sin límite")
        self.sp_tiempo.setToolTip(
            "Cierra la posición N velas después de entrar (0 = sin límite; "
            "en Patrones de velas, 0 usa el lag de salida del patrón)")
        self.sp_tiempo.valueChanged.connect(self._guardar_setup_actual)
        f_set.addRow("Salida por tiempo (velas):", self.sp_tiempo)
        # etiquetas de stop/TP: se oscurecen junto al campo en modo edge
        self._lbl_stop = f_set.labelForField(self.sp_stop)
        self._lbl_tp = f_set.labelForField(self.sp_tp)

        # ── filtros de entrada del setup (no afectan a las salidas) ──
        grp_filtros = QGroupBox("Filtros del setup")
        grp_filtros.setToolTip(
            "Día/régimen/sesión restringen cuándo puede abrirse una posición "
            "NUEVA (nunca cierran una ya abierta). Las condiciones extra de "
            "entrada/salida, más abajo, sí pueden aplicarse también a la "
            "salida si así se configuran.")
        f_filtros = QFormLayout(grp_filtros)

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
        self.sp_regimen_periodo.setRange(10, 5000)
        self.sp_regimen_periodo.setValue(100)
        self.sp_regimen_periodo.setToolTip(
            "Ventana rodante del ER/Hurst usada para clasificar el régimen "
            "de la vela (mismos umbrales fijos que la pestaña Patrones)")
        self.sp_regimen_periodo.valueChanged.connect(self._guardar_setup_actual)
        f_filtros.addRow("Periodo (ventana):", self.sp_regimen_periodo)

        self.cmb_sesion = QComboBox()
        self.cmb_sesion.addItems(list(_MAPA_SESION))
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

        self.editor_cond_entrada = EditorCondiciones("Condiciones extra de ENTRADA (AND)")
        self.editor_cond_entrada.tabla.cellChanged.connect(
            lambda *_: self._guardar_setup_actual())
        f_filtros.addRow(self.editor_cond_entrada)
        self.editor_cond_salida = EditorCondiciones("Condiciones extra de SALIDA (AND)")
        self.editor_cond_salida.tabla.cellChanged.connect(
            lambda *_: self._guardar_setup_actual())
        f_filtros.addRow(self.editor_cond_salida)

        f_set.addRow(grp_filtros)
        lay.addWidget(self.grp_setup)

        # ── código del sistema (siempre visible, se regenera en vivo) ──
        grp_cod = QGroupBox("Código del sistema (variables, entrada y salida de cada setup)")
        lay_cod = QVBoxLayout(grp_cod)
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

        self.progreso = QProgressBar()
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

        self._param_widgets = {}
        self._refresh_lista(seleccionar=0)
        self._recargar_selector()
        # el código incluye las variables de cuenta: refrescarlo si cambian
        for w in (self.sp_capital, self.sp_comision, self.sp_slippage):
            w.valueChanged.connect(lambda *_: self._refresh_codigo())
        self.slider_oos.valueChanged.connect(lambda *_: self._refresh_codigo())
        self._refresh_codigo()

    # ── activo ──
    @_no_crash
    def _on_file(self, path):
        if not path.lower().endswith('.csv'):
            return
        self.csv_path = path
        self.lbl_activo.setText(os.path.basename(path))
        m = TF_PATTERN.search(os.path.basename(path))
        self._tf_nativo = (m.group(1) or m.group(2)) if m else None
        self._dias_disponibles = self._detectar_dias_disponibles(path)
        self._actualizar_dias_checkboxes()
        self._configurar_botones_tf()
        self._aplicar_preset_friccion(path)
        self.btn_run.setEnabled(True)
        self.btn_optimizar.setEnabled(True)

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
        texto, ok = QInputDialog.getText(
            self, "Temporalidad personalizada",
            "Temporalidad (p.ej. 20m, 90m, 2h, 3d, 5d, 2w):",
            text=self._tf_custom or "")
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
        return [dict(s, params=dict(s['params'])) for s in self._setups]

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
            self.cmb_plantilla.setCurrentText(s['plantilla'])
            self._rebuild_params(s['plantilla'], s['params'])
            self.sp_riesgo.setValue(s['riesgo_pct'] * 100.0)
            self.sp_stop.setValue(s['stop_atr'])
            self.sp_tp.setValue(s['tp_r'])
            self.sp_tiempo.setValue(s['salida_n_velas'])
            edge = bool(s.get('edge', False))
            self.btn_edge.setChecked(edge)
            self._bloquear_stop_tp(edge)
            if s['plantilla'] == 'Custom (reglas)':
                self.editor_reglas.cargar_reglas(s['params'].get('reglas'))
            filtros = s.get('filtros') or _filtros_por_defecto()
            self._actualizar_dias_checkboxes()
            self.cmb_regimen.setCurrentText(
                _MAPA_REGIMEN_INV.get(filtros.get('regimen', {}).get('metodo', 'ninguno'), 'Ninguno'))
            self.sp_regimen_periodo.setValue(filtros.get('regimen', {}).get('periodo', 100))
            self.cmb_sesion.setCurrentText(
                _MAPA_SESION_INV.get(filtros.get('sesion', {}).get('tipo', 'ninguna'), 'Ninguna'))
            self.sp_hora_ini.setValue(filtros.get('sesion', {}).get('hora_inicio', 0))
            self.sp_hora_fin.setValue(filtros.get('sesion', {}).get('hora_fin', 0))
            self.editor_cond_entrada.cargar_condiciones(filtros.get('condiciones_entrada'))
            self.editor_cond_salida.cargar_condiciones(filtros.get('condiciones_salida'))
        finally:
            self._cargando = False
        self._actualizar_visibilidad_filtros()
        self._refresh_definicion()

    @_no_crash
    def _on_plantilla_changed(self, plantilla):
        if self._cargando:
            return
        s = self._setup_actual()
        if s is None:
            return
        s['plantilla'] = plantilla
        s['params'] = params_por_defecto(plantilla)
        s.update(defaults_setup(plantilla))   # p.ej. cruce → sin stop ATR
        self._cargando = True
        try:
            self._rebuild_params(plantilla, s['params'])
            self.sp_stop.setValue(s['stop_atr'])
            self.sp_tp.setValue(s['tp_r'])
            self.sp_tiempo.setValue(s['salida_n_velas'])
            # el modo edge sobrevive al cambio de plantilla: stop/TP siguen a 0
            if self.btn_edge.isChecked():
                self.sp_stop.setValue(0.0)
                self.sp_tp.setValue(0.0)
            if plantilla == 'Custom (reglas)':
                self.editor_reglas.cargar_reglas(s['params'].get('reglas'))
        finally:
            self._cargando = False
        self._guardar_setup_actual()

    def _rebuild_params(self, plantilla, valores):
        while self.form_params.rowCount():
            self.form_params.removeRow(0)
        self._param_widgets = {}
        for spec in ESTRATEGIAS[plantilla]['params']:
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

    def _bloquear_stop_tp(self, bloquear):
        """Oscurece (deshabilita) los campos de stop y TP y sus etiquetas —
        el estilo :disabled de STYLE_BACKTEST los pinta apagados."""
        for w in (self.sp_stop, self.sp_tp, self._lbl_stop, self._lbl_tp):
            if w is not None:
                w.setEnabled(not bloquear)

    @_no_crash
    def _on_edge_toggled(self, on):
        """Modo edge: prueba la señal desnuda — bloquea stop y TP a 0 (el
        motor los interpreta como desactivados) y al salir restaura los
        valores que tenía el setup."""
        s = None if self._cargando else self._setup_actual()
        if on:
            if s is not None:
                s['edge_prev_stop'] = self.sp_stop.value()
                s['edge_prev_tp'] = self.sp_tp.value()
            self.sp_stop.setValue(0.0)
            self.sp_tp.setValue(0.0)
        elif s is not None:
            self.sp_stop.setValue(s.get('edge_prev_stop', s.get('stop_atr', 0.0)))
            self.sp_tp.setValue(s.get('edge_prev_tp', s.get('tp_r', 0.0)))
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
        self._fila_horas_widget.setVisible(self.cmb_sesion.currentText() == 'Personalizada')

    @_no_crash
    def _on_regimen_changed(self, _texto):
        self._actualizar_visibilidad_filtros()
        self._guardar_setup_actual()

    @_no_crash
    def _on_sesion_changed(self, _texto):
        self._actualizar_visibilidad_filtros()
        self._guardar_setup_actual()

    @_no_crash
    def _guardar_setup_actual(self, *_):
        if self._cargando:
            return
        s = self._setup_actual()
        if s is None:
            return
        s['nombre'] = self.txt_nombre.text().strip() or s['nombre']
        s['plantilla'] = self.cmb_plantilla.currentText()
        s['params'] = self._leer_params_widgets(s['plantilla'])
        s['riesgo_pct'] = self.sp_riesgo.value() / 100.0
        s['stop_atr'] = self.sp_stop.value()
        s['tp_r'] = self.sp_tp.value()
        s['salida_n_velas'] = self.sp_tiempo.value()
        s['edge'] = self.btn_edge.isChecked()
        elegidos = [i for i, chk in enumerate(self._chk_dias) if chk.isChecked()]
        dias_disponibles = self._dias_disponibles or set(range(7))
        s['filtros'] = {
            'dias_semana': None if set(elegidos) >= dias_disponibles else elegidos,
            'regimen': {
                'metodo': _MAPA_REGIMEN[self.cmb_regimen.currentText()],
                'periodo': self.sp_regimen_periodo.value(),
            },
            'sesion': {
                'tipo': _MAPA_SESION[self.cmb_sesion.currentText()],
                'hora_inicio': self.sp_hora_ini.value(),
                'hora_fin': self.sp_hora_fin.value(),
            },
            'condiciones_entrada': self.editor_cond_entrada.condiciones(),
            'condiciones_salida': self.editor_cond_salida.condiciones(),
        }
        self._refresh_item_actual()
        self._refresh_definicion()
        self._refresh_codigo()

    def _refresh_definicion(self):
        s = self._setup_actual()
        if s is None:
            self.lbl_definicion.setText("")
            return
        try:
            self.lbl_definicion.setText(describir(s['plantilla'], s['params']))
        except Exception:
            self.lbl_definicion.setText("(definición no disponible)")

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
        try:
            with open(RUTA_ESTRATEGIAS_GUARDADAS, encoding='utf-8') as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return {}

    def _recargar_selector(self):
        """Reconstruye el combo de sistemas: plantillas predeterminadas,
        sistemas guardados en JSON y la opción de crear uno nuevo."""
        self.cmb_sistema.blockSignals(True)
        self.cmb_sistema.clear()
        self.cmb_sistema.addItem("— elegir sistema —", None)
        for plantilla in ESTRATEGIAS:
            self.cmb_sistema.addItem(f"Predeterminado: {plantilla}",
                                     ('plantilla', plantilla))
        for nombre in sorted(self._leer_guardadas()):
            self.cmb_sistema.addItem(f"Guardado: {nombre}", ('guardado', nombre))
        self.cmb_sistema.addItem("Nuevo sistema custom…", ('nuevo', None))
        self.cmb_sistema.blockSignals(False)

    @_no_crash
    def _on_sistema_elegido(self, _idx):
        dato = self.cmb_sistema.currentData()
        if not dato:
            return
        tipo, valor = dato
        if tipo == 'plantilla':
            s = _setup_por_defecto(valor)
            s['nombre'] = valor
            self._setups = [s]
            self.lbl_estado.setText(f"Sistema predeterminado «{valor}» cargado")
        elif tipo == 'guardado':
            datos = self._leer_guardadas().get(valor)
            if not datos:
                return
            if 'setups' in datos:
                self._setups = [dict(_setup_por_defecto(), **s)
                                for s in datos['setups']]
            elif 'estrategia' in datos:
                # migración v1: una estrategia suelta → sistema de 1 setup
                s = _setup_por_defecto(datos['estrategia'])
                s['nombre'] = valor
                s['params'] = dict(params_por_defecto(datos['estrategia']),
                                   **datos.get('params', {}))
                self._setups = [s]
            self.lbl_estado.setText(f"Sistema «{valor}» cargado")
        elif tipo == 'nuevo':
            s = _setup_por_defecto('Custom (reglas)')
            s['nombre'] = 'Mi setup'
            self._setups = [s]
            self.lbl_estado.setText(
                "Sistema nuevo: define las reglas del setup y añade más si quieres")
        self._refresh_lista(seleccionar=0)

    @_no_crash
    def _guardar_sistema(self):
        nombre, ok = QInputDialog.getText(self, "Guardar sistema",
                                          "Nombre del sistema:")
        if not ok or not nombre.strip():
            return
        datos = self._leer_guardadas()
        datos[nombre.strip()] = {'setups': self.setups()}
        with open(RUTA_ESTRATEGIAS_GUARDADAS, 'w', encoding='utf-8') as f:
            json.dump(datos, f, ensure_ascii=False, indent=2)
        self._recargar_selector()
        self.lbl_estado.setText(f"Sistema «{nombre.strip()}» guardado")


# ══════════════ diálogo: configurar el barrido de parámetros ══════════════
class DialogoOptimizacion(QDialog):
    """Define, para el setup actualmente seleccionado en Constructor, qué
    parámetros (de la estrategia y/o del riesgo propio del setup) se barren
    y en qué rango, más la métrica de ranking. El barrido corre siempre
    SOLO sobre el tramo IS (core.optimizer.optimizar_setup) — los campos no
    marcados se quedan fijos en su valor actual."""

    METRICAS = [
        ('sharpe', 'Sharpe acumulado (IS)'),
        ('retorno_pct', 'Retorno % (IS)'),
        ('profit_factor', 'Profit factor (IS)'),
        ('sqn', 'SQN (IS)'),
    ]
    # (clave, etiqueta, tipo, escala widget→valor guardado, min, max)
    CAMPOS_RIESGO = [
        ('riesgo_pct', 'Riesgo del setup (%)', 'float', 100.0, 0.01, 100.0),
        ('stop_atr', 'Stop (× ATR)', 'float', 1.0, 0.0, 20.0),
        ('tp_r', 'Take-profit (R)', 'float', 1.0, 0.0, 20.0),
        ('salida_n_velas', 'Salida por tiempo (velas)', 'int', 1.0, 0, 10000),
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
    ('expectancy_pct', 'Expectancy por trade', 3, ' %'),
    ('racha_perdedora', 'Racha perdedora máx.', 0, ''),
    ('duracion_media', 'Duración media (velas)', 1, ''),
    (None, '— Curva de capital —', None, None),
    ('r2_equity', 'R² de la equity curve', 3, ''),
    ('dd_promedio_pct', 'Drawdown promedio', 2, ' %'),
    ('tiempo_recuperacion_medio', 'Tiempo recuperación medio (velas)', 1, ''),
    ('tiempo_recuperacion_max', 'Tiempo recuperación máx. (velas)', 0, ''),
    (None, '— Robustez y costes —', None, None),
    ('sqn', 'SQN (>2 bueno, >3 excelente)', 2, ''),
    ('payoff_ratio', 'Payoff ratio (ganancia/pérdida media)', 2, ''),
    ('pct_mejor_trade', '% PnL del mejor trade', 1, ' %'),
    ('slippage_minimo_pct', 'Slippage mínimo viable', 3, ' %'),
    ('impacto_comisiones_pct', 'Impacto de comisiones', 2, ' %'),
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
    'pct_mejor_trade': "% del PnL total que aporta un único trade — alto "
                       "(>20-30%) sugiere depender de un outlier, no de un edge consistente",
    'slippage_minimo_pct': "Slippage adicional (aplicado igual que la "
                          "comisión) que dejaría la expectancy media en cero. "
                          "Negativo = el sistema ya es negativo sin slippage extra",
    'impacto_comisiones_pct': "Comisión pagada como % de la ganancia bruta "
                             "— en scalping puede devorar el edge entero",
}


class ResultadosWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._payload = None
        self._wfa_cache = None
        self._wfa_ts = None
        self._modo_grafico = 'velas'
        self._art_datos = []
        self._actualizando_xlim = False
        self._y_manual = False

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.addWidget(scroll)
        cont = QWidget()
        lay = QVBoxLayout(cont)
        lay.setSpacing(10)
        scroll.setWidget(cont)

        self.lbl_titulo = QLabel("Ejecuta un backtest desde el Optimizador")
        self.lbl_titulo.setObjectName("titulo")
        lay.addWidget(self.lbl_titulo)

        # métricas IS/OOS/Total
        self.tabla_metricas = QTableWidget(len(_FILAS_METRICAS), 4)
        self.tabla_metricas.setHorizontalHeaderLabels(['Métrica', 'IS', 'OOS', 'Total'])
        self.tabla_metricas.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.tabla_metricas.verticalHeader().setVisible(False)
        self.tabla_metricas.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.tabla_metricas.setFixedHeight(24 * len(_FILAS_METRICAS) + 30)
        lay.addWidget(self.tabla_metricas)

        # gráfico principal: log-return + flechas + IS/OOS
        self.fig = Figure(figsize=(9, 3.2), facecolor=FIG_BG)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setMinimumHeight(300)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.toolbar.setStyleSheet("background-color: #141e30;")
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
        self.chk_stop = QCheckBox("Mostrar stop loss")
        self.chk_stop.setToolTip(
            "Dibuja el nivel de stop-loss (×ATR) de cada operación y la zona "
            "de riesgo entre el precio de entrada y el stop.")
        self.chk_stop.toggled.connect(self._toggle_stop_loss)
        fila_zoom.addWidget(self.chk_stop)
        self.chk_trayecto = QCheckBox("Mostrar trayecto")
        self.chk_trayecto.setToolTip(
            "Dibuja una línea gris entre el precio de entrada y el de salida "
            "de cada operación (recorrido del trade).")
        self.chk_trayecto.toggled.connect(self._toggle_trayecto)
        fila_zoom.addWidget(self.chk_trayecto)
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
        lay.addWidget(self.stack_grafico)

        # arrastrar los márgenes de los ejes (estilo TradingView/MT4): eje Y
        # (precio) para estirar/comprimir la escala, eje X (fecha) para zoom
        # temporal, arrastrando directamente sobre la franja de etiquetas
        self._drag_modo = None
        self._drag_inicio = None
        self._drag_lim0 = None
        self.canvas.mpl_connect('button_press_event', self._on_press_ejes)
        self.canvas.mpl_connect('motion_notify_event', self._on_motion_ejes)
        self.canvas.mpl_connect('button_release_event', self._on_release_ejes)
        self.canvas.mpl_connect('scroll_event', self._on_scroll)

        # curva de equity (IS vs OOS)
        self.grp_equity = QGroupBox()
        header_eq = QHBoxLayout()
        lbl_eq = QLabel("Curva de Equity (IS vs OOS)")
        lbl_eq.setObjectName("titulo")
        header_eq.addWidget(lbl_eq)
        header_eq.addStretch()
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
        lay_eq.addWidget(self.canvas_equity)
        self.grp_equity.setVisible(False)
        lay.addWidget(self.grp_equity)

        # indicadores usados por el sistema
        self.grp_indicadores = QGroupBox()
        header_ind = QHBoxLayout()
        lbl_ind = QLabel("Indicadores del sistema")
        lbl_ind.setObjectName("titulo")
        header_ind.addWidget(lbl_ind)
        header_ind.addStretch()
        lay_ind = QVBoxLayout(self.grp_indicadores)
        lay_ind.addLayout(header_ind)
        self.fig_ind = Figure(figsize=(9, 3), facecolor=FIG_BG)
        self.canvas_ind = FigureCanvasQTAgg(self.fig_ind)
        self.canvas_ind.setMinimumHeight(280)
        lay_ind.addWidget(self.canvas_ind)
        self.grp_indicadores.setVisible(False)
        lay.addWidget(self.grp_indicadores)

        # código del sistema ejecutado (constancia exacta de qué se backtesteó)
        from PyQt6.QtWidgets import QPlainTextEdit
        self.grp_codigo = QGroupBox("Código del sistema ejecutado (clic para desplegar)")
        self.grp_codigo.setCheckable(True)
        self.grp_codigo.setChecked(False)
        lay_cod = QVBoxLayout(self.grp_codigo)
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
        self.tabla_setups = QTableWidget(0, 6)
        self.tabla_setups.setHorizontalHeaderLabels(
            ['Setup', 'Riesgo %', 'Trades', 'Win rate', 'PnL', 'Expectancy %'])
        self.tabla_setups.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.tabla_setups.verticalHeader().setVisible(False)
        self.tabla_setups.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.tabla_setups.setMaximumHeight(160)
        lay_ms.addWidget(self.tabla_setups)
        self.grp_setups.setVisible(False)
        lay.addWidget(self.grp_setups)

        # tabla de trades
        fila_tr = QHBoxLayout()
        lbl_tr = QLabel("Trades (clic en una fila para centrar el gráfico)")
        lbl_tr.setObjectName("titulo")
        fila_tr.addWidget(lbl_tr)
        fila_tr.addStretch()
        btn_lista_completa = QPushButton("Lista completa")
        btn_lista_completa.clicked.connect(self._abrir_lista_completa)
        fila_tr.addWidget(btn_lista_completa)
        lay.addLayout(fila_tr)
        self._dlg_trades = None
        self.tabla_trades = QTableWidget(0, 8)
        self.tabla_trades.setHorizontalHeaderLabels(
            ['Entrada', 'Salida', 'Dir', 'Setup', 'P. entrada', 'P. salida',
             'PnL', 'Motivo'])
        self.tabla_trades.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.tabla_trades.verticalHeader().setVisible(False)
        self.tabla_trades.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.tabla_trades.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.tabla_trades.setMaximumHeight(220)
        self.tabla_trades.setSortingEnabled(True)
        self.tabla_trades.cellClicked.connect(self._centrar_trade)
        lay.addWidget(self.tabla_trades)

        # WFA
        self.grp_wfa = QGroupBox()
        header_wfa = QHBoxLayout()
        lbl_wfa_titulo = QLabel("Walk-Forward Analysis")
        lbl_wfa_titulo.setObjectName("titulo")
        header_wfa.addWidget(lbl_wfa_titulo)
        header_wfa.addStretch()
        self.combo_wfa_modo = QComboBox()
        self.combo_wfa_modo.setMinimumWidth(160)
        for m in MODOS_WFA:
            self.combo_wfa_modo.addItem(m)
        self.combo_wfa_modo.currentIndexChanged.connect(
            lambda _i: self._dibujar_wfa(self._wfa_cache, self._wfa_ts))
        header_wfa.addWidget(self.combo_wfa_modo)
        lay_wfa = QVBoxLayout(self.grp_wfa)
        lay_wfa.addLayout(header_wfa)
        self.fig_wfa = Figure(figsize=(9, 2.2), facecolor=FIG_BG)
        self.canvas_wfa = FigureCanvasQTAgg(self.fig_wfa)
        self.canvas_wfa.setMinimumHeight(200)
        lay_wfa.addWidget(self.canvas_wfa)
        self.tabla_wfa = QTableWidget(0, 6)
        self.tabla_wfa.setHorizontalHeaderLabels(
            ['Ventana', 'Periodo', 'Trades', 'Win rate', 'Retorno %', 'Max DD %'])
        self.tabla_wfa.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.tabla_wfa.verticalHeader().setVisible(False)
        self.tabla_wfa.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.tabla_wfa.setMaximumHeight(180)
        lay_wfa.addWidget(self.tabla_wfa)
        self.grp_wfa.setVisible(False)
        lay.addWidget(self.grp_wfa)

        # Montecarlo
        self.grp_mc = QGroupBox("Montecarlo (1000 remuestreos del orden de los trades)")
        lay_mc = QVBoxLayout(self.grp_mc)
        self.lbl_mc = QLabel("")
        self.lbl_mc.setObjectName("campo")
        lay_mc.addWidget(self.lbl_mc)
        self.fig_mc = Figure(figsize=(9, 2.6), facecolor=FIG_BG)
        self.canvas_mc = FigureCanvasQTAgg(self.fig_mc)
        self.canvas_mc.setMinimumHeight(230)
        lay_mc.addWidget(self.canvas_mc)
        self.grp_mc.setVisible(False)
        lay.addWidget(self.grp_mc)

        lay.addStretch()

    # ── render principal ──
    @_no_crash
    def mostrar(self, payload):
        self._payload = payload
        self._y_manual = False
        ts = pd.DatetimeIndex(payload['timestamps'])
        met = payload['metricas']
        tf_txt = f" [{payload['tf']}]" if payload.get('tf') else ''
        self.lbl_titulo.setText(
            f"{os.path.basename(payload['csv'])}{tf_txt} — {payload['estrategia']} — "
            f"{payload['n_velas']:,} velas · {payload['resultado']['n_trades']} trades · "
            f"capital final {payload['resultado']['capital_final']:,.0f}")

        # métricas
        for fila, (clave, nombre, dec, sufijo) in enumerate(_FILAS_METRICAS):
            item_nombre = QTableWidgetItem(nombre)
            tooltip = _TOOLTIPS_METRICAS.get(clave)
            if tooltip:
                item_nombre.setToolTip(tooltip)
            if clave is None:
                # fila separadora: solo etiqueta, en negrita, sin datos
                font = item_nombre.font()
                font.setBold(True)
                item_nombre.setFont(font)
                item_nombre.setForeground(QColor(AZUL))
                self.tabla_metricas.setItem(fila, 0, item_nombre)
                for col in (1, 2, 3):
                    self.tabla_metricas.setItem(fila, col, QTableWidgetItem(''))
                continue
            self.tabla_metricas.setItem(fila, 0, item_nombre)
            for col, tramo in enumerate(('IS', 'OOS', 'Total'), start=1):
                v = met[tramo][clave]
                if clave == 'win_rate':
                    texto = _fmt(v * 100 if v is not None else None, 1, ' %')
                elif dec == 0:
                    texto = str(int(v)) if v is not None else '—'
                else:
                    texto = _fmt(v, dec, sufijo)
                it = QTableWidgetItem(texto)
                it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if clave in ('retorno_pct', 'retorno_anual_pct', 'expectancy_pct',
                            'sqn', 'r2_equity', 'payoff_ratio') and v is not None:
                    it.setForeground(QColor(VERDE if v > 0 else ROJO))
                self.tabla_metricas.setItem(fila, col, it)

        # gráfico principal
        self._dibujar_principal()
        self._dibujar_equity(payload)
        self._dibujar_indicadores(payload)
        # si la vista moderna (LWC) está activa, refrescarla con el nuevo backtest
        if getattr(self, 'btn_vista', None) is not None and self.btn_vista.isChecked():
            self._mostrar_lwc(payload)

        # fechas de los QDateEdit
        self.fecha_ini.setDate(QDate(ts[0].year, ts[0].month, ts[0].day))
        self.fecha_fin.setDate(QDate(ts[-1].year, ts[-1].month, ts[-1].day))

        # código del sistema ejecutado
        codigo = payload.get('codigo') or ''
        self.grp_codigo.setVisible(bool(codigo))
        self.txt_codigo.setPlainText(codigo)

        # métricas por setup
        ms = payload.get('metricas_setup') or []
        self.grp_setups.setVisible(bool(ms))
        self.tabla_setups.setRowCount(len(ms))
        for r, s in enumerate(ms):
            wr = _fmt(s['win_rate'] * 100 if s['win_rate'] is not None else None, 1, ' %')
            vals = [f"S{r} · {s['nombre']}", f"{s['riesgo_pct'] * 100:g} %",
                    str(s['n_trades']), wr, f"{s['pnl_total']:+.2f}",
                    _fmt(s['expectancy_pct'], 3, ' %')]
            for c_i, v in enumerate(vals):
                it = QTableWidgetItem(v)
                it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if c_i == 4:
                    it.setForeground(QColor(VERDE if s['pnl_total'] > 0 else ROJO))
                self.tabla_setups.setItem(r, c_i, it)

        # trades
        self._llenar_tabla_trades(self.tabla_trades, payload)
        if self._dlg_trades is not None:
            self._llenar_tabla_trades(self._dlg_trades.tabla, payload)

        # WFA
        self._wfa_cache = payload.get('wfa')
        self._wfa_ts = ts
        self._dibujar_wfa(payload.get('wfa'), ts)
        # Montecarlo
        self._dibujar_mc(payload.get('montecarlo'),
                         payload['config'].get('capital_inicial', 10000.0))

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
        stop-loss) que la vista clásica, para que ambas se vean coherentes al
        conmutar entre ellas."""
        self.lwc.mostrar(
            payload,
            mostrar_trayecto=getattr(self, 'chk_trayecto', None) is not None
                             and self.chk_trayecto.isChecked(),
            mostrar_stop=getattr(self, 'chk_stop', None) is not None
                        and self.chk_stop.isChecked())

    def _redibujar_principal_conservando_zoom(self):
        """Redibuja el gráfico principal preservando el rango temporal (zoom)
        actual — compartido por los toggles de modo velas/línea, stop-loss y
        trayecto. Si la vista moderna (LWC) está activa, también la refresca."""
        xlim = None
        if getattr(self, '_ax_principal', None) is not None:
            a, b = self._ax_principal.get_xlim()
            xlim = (pd.Timestamp(num2date(a)), pd.Timestamp(num2date(b)))
        self._dibujar_principal(xlim=xlim)
        if getattr(self, 'btn_vista', None) is not None and self.btn_vista.isChecked() \
                and getattr(self, '_payload', None) is not None:
            self._mostrar_lwc(self._payload)

    def _toggle_trayecto(self, _=False):
        self._redibujar_principal_conservando_zoom()

    def _redibujar_datos(self, ax):
        """Redibuja solo las velas/línea, decimadas al rango visible de
        `ax` — se llama tanto en el dibujo inicial como en cada zoom/pan
        (callback 'xlim_changed'), sin tocar sombreado/flechas/leyenda/slider."""
        for art in self._art_datos:
            try:
                art.remove()
            except Exception:
                pass
        self._art_datos = []

        x0, x1 = ax.get_xlim()
        xb, ob, hb, lb, cb = _decimar_ohlc(
            self._x_full, self._o_full, self._h_full, self._l_full, self._c_full,
            x0, x1)
        if len(xb) == 0:
            return

        if self._modo_grafico == 'velas':
            colores = np.where(cb >= ob, VERDE, ROJO)
            ancho = np.median(np.diff(xb)) * 0.7 if len(xb) > 1 else 0.7
            col_vlines = ax.vlines(xb, lb, hb, color=colores, linewidth=0.6)
            # cuerpos como UN solo PolyCollection (no ax.bar): ax.bar crea un
            # Rectangle individual por vela y actualiza los límites de datos
            # en cada add_patch, lo que con miles de velas es el cuello de
            # botella real del redibujado (medido con cProfile: ~1.2s en
            # add_patch/_update_patch_limits para 2500 barras)
            half = ancho / 2.0
            yb = np.minimum(ob, cb)
            yt = np.maximum(ob, cb)
            verts = np.stack([
                np.column_stack([xb - half, yb]),
                np.column_stack([xb + half, yb]),
                np.column_stack([xb + half, yt]),
                np.column_stack([xb - half, yt]),
            ], axis=1)
            cuerpos = PolyCollection(verts, facecolors=colores, edgecolors=colores,
                                     linewidths=0)
            ax.add_collection(cuerpos)
            self._art_datos = [col_vlines, cuerpos]
            y_lo, y_hi = float(np.min(lb)), float(np.max(hb))
        else:
            line, = ax.plot(xb, cb, color=AX_FG, linewidth=0.8)
            self._art_datos = [line]
            y_lo, y_hi = float(np.min(cb)), float(np.max(cb))

        if not self._y_manual:
            pad = (y_hi - y_lo) * 0.05 or 1.0
            ax.set_ylim(y_lo - pad, y_hi + pad)

    def _on_xlim_changed(self, ax):
        if self._actualizando_xlim:
            return
        self._actualizando_xlim = True
        try:
            self._redibujar_datos(ax)
            self.canvas.draw_idle()
        finally:
            self._actualizando_xlim = False

    def _zona_eje(self, event):
        """'pan' si el evento cae dentro del propio gráfico (arrastrar
        desplaza la vista), 'y' si cae en la franja de etiquetas del eje Y
        (a la derecha del gráfico), 'x' si cae en la franja de etiquetas del
        eje X (debajo del gráfico, por encima del RangeSlider), o None."""
        ax = getattr(self, '_ax_principal', None)
        if ax is None or event.x is None or event.y is None:
            return None
        if self.toolbar.mode:
            return None  # no interferir con pan/zoom nativo del toolbar
        slider_ax = getattr(self._range_slider, 'ax', None) \
            if getattr(self, '_range_slider', None) is not None else None
        if slider_ax is not None and event.inaxes is slider_ax:
            return None
        bbox = ax.get_window_extent()
        if bbox.x0 <= event.x <= bbox.x1 and bbox.y0 <= event.y <= bbox.y1:
            return 'pan'
        if bbox.y0 <= event.y <= bbox.y1 and event.x > bbox.x1:
            return 'y'
        limite_inferior = slider_ax.get_window_extent().y1 if slider_ax is not None else 0
        if bbox.x0 <= event.x <= bbox.x1 and limite_inferior < event.y < bbox.y0:
            return 'x'
        return None

    @_no_crash
    def _on_press_ejes(self, event):
        zona = self._zona_eje(event)
        if zona is None:
            return
        ax = self._ax_principal
        if zona == 'y' and event.dblclick:
            # doble clic sobre el eje Y: volver a autoescala de precio
            self._y_manual = False
            self._redibujar_datos(ax)
            self.canvas.draw_idle()
            return
        self._drag_modo = zona
        self._drag_inicio = (event.x, event.y)
        if zona == 'pan':
            self._drag_lim0 = (ax.get_xlim(), ax.get_ylim())
            self.canvas.setCursor(Qt.CursorShape.ClosedHandCursor)
        elif zona == 'y':
            self._drag_lim0 = ax.get_ylim()
        else:
            self._drag_lim0 = ax.get_xlim()

    @_no_crash
    def _on_motion_ejes(self, event):
        ax = getattr(self, '_ax_principal', None)
        if ax is None:
            return
        if self._drag_modo is None:
            zona = self._zona_eje(event)
            cursores = {'y': Qt.CursorShape.SizeVerCursor,
                        'x': Qt.CursorShape.SizeHorCursor,
                        'pan': Qt.CursorShape.OpenHandCursor}
            self.canvas.setCursor(cursores.get(zona, Qt.CursorShape.ArrowCursor))
            return
        if event.x is None or event.y is None:
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
            if getattr(self, '_range_slider', None) is not None:
                try:
                    self._range_slider.set_val(ax.get_xlim())
                except Exception:
                    pass
            self._sync_dateedits(*ax.get_xlim())
            return
        lo, hi = self._drag_lim0
        centro = (lo + hi) / 2.0
        if self._drag_modo == 'y':
            factor = np.exp(-(event.y - y0) * 0.005)
            medio = (hi - lo) / 2.0 * factor
            self._y_manual = True
            ax.set_ylim(centro - medio, centro + medio)
            self.canvas.draw_idle()
        else:
            factor = np.exp(-(event.x - x0) * 0.005)
            medio = (hi - lo) / 2.0 * factor
            ax.set_xlim(centro - medio, centro + medio)  # dispara xlim_changed
            if getattr(self, '_range_slider', None) is not None:
                try:
                    self._range_slider.set_val(ax.get_xlim())
                except Exception:
                    pass
            self._sync_dateedits(*ax.get_xlim())

    @_no_crash
    def _on_release_ejes(self, event):
        self._drag_modo = None
        self._drag_inicio = None
        self._drag_lim0 = None

    @_no_crash
    def _on_scroll(self, event):
        """Rueda del ratón: zoom temporal (eje X) anclado bajo el cursor."""
        ax = getattr(self, '_ax_principal', None)
        if ax is None:
            return
        lo, hi = ax.get_xlim()
        ancla = event.xdata if event.xdata is not None else (lo + hi) / 2.0
        subir = (event.step > 0) if getattr(event, 'step', 0) else (event.button == 'up')
        factor = 0.85 if subir else 1.0 / 0.85
        ax.set_xlim(ancla + (lo - ancla) * factor, ancla + (hi - ancla) * factor)
        if getattr(self, '_range_slider', None) is not None:
            try:
                self._range_slider.set_val(ax.get_xlim())
            except Exception:
                pass
        self._sync_dateedits(*ax.get_xlim())

    def _dibujar_principal(self, xlim=None):
        p = self._payload
        if p is None:
            return
        ts = pd.DatetimeIndex(p['timestamps'])
        y = p['close']   # PRECIO real del activo, no log-return en tanto por uno
        corte = p['corte']
        tr = p['resultado']['trades']

        self._x_full = date2num(ts)
        self._o_full = p.get('open', p['close'])
        self._h_full = p.get('high', p['close'])
        self._l_full = p.get('low', p['close'])
        self._c_full = p['close']
        self._art_datos = []

        self.fig.clear()
        ax = self.fig.add_subplot(111)
        _style_ax(ax)
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position('right')
        # sombreado IS / OOS
        ax.axvspan(ts[0], ts[corte - 1] if corte > 0 else ts[0],
                   color=AZUL, alpha=0.05)
        if corte < len(ts):
            ax.axvspan(ts[corte], ts[-1], color=AMBAR, alpha=0.06)
            ax.axvline(ts[corte], color=AMBAR, linewidth=0.9, linestyle='--', alpha=0.8)

        if xlim is not None:
            ax.set_xlim(*xlim)
        else:
            ax.set_xlim(self._x_full[0], self._x_full[-1])
        self._redibujar_datos(ax)

        # trayecto de cada operación (entrada→salida, precio real de fill) —
        # flecha gris tenue con punta en la salida, opcional vía checkbox: la
        # pendiente muestra de un vistazo si el trade fue ganador o perdedor.
        # alpha bajo y zorder entre velas (1) y marcadores (3) para no tapar
        # nada a su paso.
        if getattr(self, 'chk_trayecto', None) is not None and self.chk_trayecto.isChecked():
            hay_trayecto = False
            for r in range(len(tr['pnl'])):
                x0 = self._x_full[tr['idx_entrada'][r]]
                x1 = self._x_full[tr['idx_salida'][r]]
                ax.add_patch(FancyArrowPatch(
                    (x0, tr['precio_entrada'][r]), (x1, tr['precio_salida'][r]),
                    arrowstyle='-|>', mutation_scale=8, color=GRIS,
                    linewidth=0.9, alpha=0.4, zorder=2.5, shrinkA=0, shrinkB=0))
                hay_trayecto = True
            if hay_trayecto:
                ax.plot([], [], color=GRIS, linewidth=1.2, label='Trayecto')

        # flechas: verde = compra (abre long / cierra short), roja = venta —
        # en tonos más saturados que las velas para que no se camuflen
        idx_compra, idx_venta = [], []
        for r in range(len(tr['pnl'])):
            if tr['dir'][r] > 0:
                idx_compra.append(tr['idx_entrada'][r])
                idx_venta.append(tr['idx_salida'][r])
            else:
                idx_venta.append(tr['idx_entrada'][r])
                idx_compra.append(tr['idx_salida'][r])
        if idx_compra:
            ax.scatter(ts[idx_compra], y[idx_compra], marker='^', s=28,
                       color=VERDE_FLECHA, zorder=3, label='Compra')
        if idx_venta:
            ax.scatter(ts[idx_venta], y[idx_venta], marker='v', s=28,
                       color=ROJO_FLECHA, zorder=3, label='Venta')

        # stop-loss por operación (nivel ×ATR fijado al entrar) + zona de
        # riesgo entre precio de entrada y stop — opcional, vía checkbox
        if getattr(self, 'chk_stop', None) is not None and self.chk_stop.isChecked() \
                and 'precio_stop' in tr:
            segmentos, cuadros = [], []
            for r in range(len(tr['pnl'])):
                stop = tr['precio_stop'][r]
                if stop <= 0:   # sin stop real (stop_atr=0) → nada que dibujar
                    continue
                x0 = self._x_full[tr['idx_entrada'][r]]
                x1 = self._x_full[tr['idx_salida'][r]]
                ent = tr['precio_entrada'][r]
                segmentos.append([(x0, stop), (x1, stop)])
                cuadros.append([(x0, ent), (x1, ent), (x1, stop), (x0, stop)])
            if segmentos:
                ax.add_collection(PolyCollection(
                    cuadros, facecolors=ROJO, alpha=0.08, edgecolors='none',
                    zorder=1.5))
                ax.add_collection(LineCollection(
                    segmentos, colors=ROJO, linewidths=0.8, linestyles='--',
                    alpha=0.7, zorder=2.5))
                ax.plot([], [], color=ROJO, linestyle='--', linewidth=1.0,
                        label='Stop loss')

        # etiquetas IS / OOS sobre el eje
        ax.text(0.01, 0.97, 'IS', transform=ax.transAxes, color=AZUL,
                fontsize=9, fontweight='bold', va='top')
        ax.text(0.99, 0.97, 'OOS', transform=ax.transAxes, color=AMBAR,
                fontsize=9, fontweight='bold', va='top', ha='right')
        ax.set_ylabel('Precio', fontsize=8, color=AX_FG)

        # ── indicadores overlays (medias, Bollinger, patrones) ──
        mas, bbs, _, _, patrones_set = self._recolectar_indicadores(p)
        paletas_ma = [AZUL, AMBAR, '#2ecc71', '#9b59b6', '#e67e22']
        idx_paleta = 0
        for tipo, per in sorted(mas, key=lambda x: x[1]):
            color = COLOR_MEDIA_FIJO.get(per)
            if color is None:
                color = paletas_ma[idx_paleta % len(paletas_ma)]
                idx_paleta += 1
            f = ema if tipo == 'EMA' else sma
            val = f(y, per)
            ax.plot(ts, val, color=color, linewidth=1.0, alpha=0.75,
                    label=f'{tipo}({per})')
        for per, desv in bbs:
            media, sup, inf = bollinger(y, per, desv)
            bb_col = '#9b59b6'
            ax.plot(ts, sup, color=bb_col, linewidth=0.5, alpha=0.4)
            ax.plot(ts, inf, color=bb_col, linewidth=0.5, alpha=0.4)
            ax.fill_between(ts, inf, sup, color=bb_col, alpha=0.05)
            ax.plot([], [], color=bb_col, linewidth=1.2,
                    label=f'BB({per}, {desv:g})')
        if patrones_set:
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
                color_pat = '#2ecc71' if np.mean(dirs) > 0 else '#e74c3c'
                ax.scatter(ts[idx], l_all[idx] - offset_pat,
                           marker='^' if np.mean(dirs) > 0 else 'v',
                           color=color_pat, s=8, alpha=0.7, zorder=5)

        ax.legend(loc='lower right', fontsize=7, facecolor=FIG_BG,
                  edgecolor=GRID_C, labelcolor=AX_FG, framealpha=0.6)

        # RangeSlider embebido bajo el gráfico (mismo patrón que
        # rango_dialog.py: referencias fuertes o el GC lo desconecta)
        self.fig.subplots_adjust(left=0.02, right=0.92, top=0.94, bottom=0.24)
        pos = ax.get_position()
        ax_sl = self.fig.add_axes([pos.x0, 0.06, pos.width, 0.05])
        ax_sl.set_facecolor('#1a2a45')
        x0n, x1n = self._x_full[0], self._x_full[-1]
        v0, v1 = ax.get_xlim()
        v0, v1 = max(v0, x0n), min(v1, x1n)
        self._range_slider = RangeSlider(ax_sl, '', x0n, x1n,
                                         valinit=(v0, v1), valfmt='')

        def on_slider(_val):
            a, b = self._range_slider.val
            ax.set_xlim(num2date(a), num2date(b))
            self._sync_dateedits(a, b)
            self.canvas.draw_idle()

        self._range_slider.on_changed(on_slider)
        ax.callbacks.connect('xlim_changed', self._on_xlim_changed)
        self._ax_principal = ax
        self.canvas.draw_idle()

    def _sync_dateedits(self, num_ini, num_fin):
        d0 = num2date(num_ini)
        d1 = num2date(num_fin)
        self.fecha_ini.setDate(QDate(d0.year, d0.month, d0.day))
        self.fecha_fin.setDate(QDate(d1.year, d1.month, d1.day))

    @_no_crash
    def _aplicar_rango(self):
        if self._payload is None:
            return
        d0 = self.fecha_ini.date().toPyDate()
        d1 = self.fecha_fin.date().toPyDate()
        self._dibujar_principal(xlim=(pd.Timestamp(d0), pd.Timestamp(d1)))

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
        entrada = pd.Timestamp(it_in.text())
        salida = pd.Timestamp(it_out.text())
        ts = pd.DatetimeIndex(self._payload['timestamps'])
        margen_min = (ts[-1] - ts[0]) / 40
        duracion = salida - entrada
        margen = max(duracion * 0.4, margen_min)
        self._y_manual = False  # re-autoescalar el precio a esta ventana concreta
        self._dibujar_principal(xlim=(entrada - margen, salida + margen))

        it_p_in, it_p_out = tabla.item(fila, 4), tabla.item(fila, 5)
        if it_p_in is not None and it_p_out is not None and self._ax_principal is not None:
            try:
                p_in, p_out = float(it_p_in.text()), float(it_p_out.text())
            except ValueError:
                return
            lo, hi = min(p_in, p_out), max(p_in, p_out)
            pad = (hi - lo) * 0.25 or (abs(p_in) * 0.02 or 1.0)
            self._ax_principal.set_ylim(lo - pad, hi + pad)
            self._y_manual = True
            self.canvas.draw_idle()

    def _llenar_tabla_trades(self, tabla, payload):
        ts = pd.DatetimeIndex(payload['timestamps'])
        nombres_setup = payload.get('nombres_setup') or []
        tr = payload['resultado']['trades']
        n_tr = len(tr['pnl'])
        tabla.setSortingEnabled(False)
        tabla.setRowCount(n_tr)
        for r in range(n_tr):
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
            ]
            for c_i, v in enumerate(vals):
                it = QTableWidgetItem(v)
                it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if c_i == 6:
                    it.setForeground(QColor(VERDE if tr['pnl'][r] > 0 else ROJO))
                tabla.setItem(r, c_i, it)
        tabla.setSortingEnabled(True)

    @_no_crash
    def _abrir_lista_completa(self):
        if self._dlg_trades is None:
            dlg = QDialog(self)
            dlg.setWindowTitle("Trades — lista completa")
            dlg.resize(900, 600)
            dlg_lay = QVBoxLayout(dlg)
            tabla = QTableWidget(0, 8)
            tabla.setHorizontalHeaderLabels(
                ['Entrada', 'Salida', 'Dir', 'Setup', 'P. entrada', 'P. salida',
                 'PnL', 'Motivo'])
            tabla.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            tabla.verticalHeader().setVisible(False)
            tabla.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
            tabla.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
            tabla.setSortingEnabled(True)
            tabla.cellClicked.connect(self._centrar_trade)
            dlg_lay.addWidget(tabla)
            dlg.tabla = tabla
            self._dlg_trades = dlg
        if self._payload is not None:
            self._llenar_tabla_trades(self._dlg_trades.tabla, self._payload)
        self._dlg_trades.show()
        self._dlg_trades.raise_()
        self._dlg_trades.activateWindow()

    def _dibujar_equity(self, payload):
        if payload is None:
            return
        eq = payload['resultado']['equity']
        ts = pd.DatetimeIndex(payload['timestamps'])
        corte = payload['corte']
        cap0 = eq[0] if eq[0] > 0 else 1.0
        n = len(eq)

        pct_is = corte / n * 100 if n > 0 else 0
        pct_oos = 100 - pct_is

        modo = self.combo_eq_modo.currentIndex()
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

        self.grp_equity.setVisible(True)
        self.fig_equity.clear()
        ax = self.fig_equity.add_subplot(111)
        _style_ax(ax)

        ts_is = ts[:corte + 1]
        ts_oos = ts[corte:]
        y_is = y[:corte + 1]
        y_oos = y[corte:]

        if modo == 3:
            ax.fill_between(ts_is, 0, y_is, color='#e74c3c', alpha=0.40, label='IS')
            ax.fill_between(ts_oos, 0, y_oos, color='#c0392b', alpha=0.35, label='OOS')
            ax.axhline(0, color=GRIS, linewidth=0.5, linestyle='--')
        else:
            ax.plot(ts_is, y_is, color=AZUL, linewidth=1.2, label='IS')
            ax.plot(ts_oos, y_oos, color='#ff9900', linewidth=1.2, label='OOS')
            if modo in (0, 2):
                ax.axhline(0, color=GRIS, linewidth=0.5, linestyle='--')

        if 0 < corte < n:
            ax.axvline(ts[corte], color=GRIS, linewidth=0.8, linestyle='--', alpha=0.7)
            ylim = ax.get_ylim()
            ax.text(ts[corte], ylim[1] - (ylim[1] - ylim[0]) * 0.03,
                    f'  IS {pct_is:.0f}% / OOS {pct_oos:.0f}%',
                    fontsize=7, color=GRIS, ha='left', va='top',
                    bbox=dict(facecolor=FIG_BG, alpha=0.7, edgecolor='none', pad=2))

        if corte > 0 and modo in (0, 1, 2):
            ax.text(ts[corte], y_is[-1], f'  {fmt_val(y_is[-1])}',
                    color=AZUL, fontsize=7, ha='left', va='center')
        if corte < n and modo in (0, 1, 2):
            ax.text(ts[-1], y_oos[-1], f'  {fmt_val(y_oos[-1])}',
                    color='#ff9900', fontsize=7, ha='left', va='center')

        ax.set_ylabel(ylabel, fontsize=8, color=AX_FG)
        ax.legend(fontsize=7, framealpha=0.2, loc='best')
        try:
            self.fig_equity.tight_layout(pad=0.6)
        except Exception:
            pass
        self.canvas_equity.draw_idle()

    def _recolectar_indicadores(self, payload):
        """Devuelve conjuntos únicos de (mas, bbs, rsis, atrs, patrones_set)
        extraídos de todos los setups del sistema. Sin repeticiones."""
        mas = set()
        bbs = set()
        rsis = set()
        atrs = set()
        patrones_set = set()
        setups = payload.get('setups', []) if payload else []
        for setup in setups:
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
            elif plantilla == 'Patrones de velas':
                patrones_set.update(p.get('patrones', []))
            elif plantilla == 'Custom (reglas)':
                for clave in ('entradas_long', 'entradas_short',
                              'salidas_long', 'salidas_short'):
                    for grupo in p.get('reglas', {}).get(clave, []):
                        for cond in grupo.get('condiciones', []):
                            for lado in (cond.get('izq', {}), cond.get('der', {})):
                                _acumular_indicador_spec(lado, mas, rsis, atrs, bbs)

            # filtros extra del setup (condiciones_entrada/condiciones_salida) —
            # aplicables a CUALQUIER plantilla, no solo Custom (reglas); por
            # eso van fuera del if/elif de arriba.
            filtros = setup.get('filtros') or {}
            for clave in ('condiciones_entrada', 'condiciones_salida'):
                for cond in filtros.get(clave, []):
                    for lado in (cond.get('izq', {}), cond.get('der', {})):
                        _acumular_indicador_spec(lado, mas, rsis, atrs, bbs)
        return mas, bbs, rsis, atrs, patrones_set

    def _dibujar_indicadores(self, payload):
        if payload is None:
            return
        ts = pd.DatetimeIndex(payload['timestamps'])
        c = payload['close']
        h = payload['high']
        l = payload['low']

        _, _, rsis, atrs, _ = self._recolectar_indicadores(payload)
        if not rsis and not atrs:
            self.grp_indicadores.setVisible(False)
            return

        self.grp_indicadores.setVisible(True)
        self.fig_ind.clear()

        n_paneles = (1 if rsis else 0) + (1 if atrs else 0)
        ratios = []
        if rsis:
            ratios.append(1)
        if atrs:
            ratios.append(1)
        gs = self.fig_ind.add_gridspec(n_paneles, 1, height_ratios=ratios, hspace=0.12)
        fila = 0

        if rsis:
            ax_rsi = self.fig_ind.add_subplot(gs[fila, 0])
            _style_ax(ax_rsi)
            pal_rsi = ['#f1c40f', '#e67e22', '#fd79a8']
            for i, per in enumerate(sorted(rsis)):
                val = rsi(c, per)
                color = pal_rsi[i % len(pal_rsi)]
                ax_rsi.plot(ts, val, color=color, linewidth=1.0, alpha=0.85,
                            label=f'RSI({per})')
            ax_rsi.axhline(70, color=ROJO, linewidth=0.5, linestyle='--', alpha=0.4)
            ax_rsi.axhline(50, color=GRIS, linewidth=0.5, linestyle='--', alpha=0.3)
            ax_rsi.axhline(30, color=VERDE, linewidth=0.5, linestyle='--', alpha=0.4)
            ax_rsi.set_ylim(0, 100)
            ax_rsi.set_ylabel('RSI', fontsize=8, color=AX_FG)
            ax_rsi.legend(fontsize=6, framealpha=0.15, labelcolor=AX_FG, loc='upper left')
            fila += 1

        if atrs:
            ax_atr = self.fig_ind.add_subplot(gs[fila, 0])
            _style_ax(ax_atr)
            pal_atr = ['#2ecc71', '#1abc9c']
            for i, per in enumerate(sorted(atrs)):
                val = atr(h, l, c, per)
                color = pal_atr[i % len(pal_atr)]
                ax_atr.plot(ts, val, color=color, linewidth=0.9, alpha=0.85,
                            label=f'ATR({per})')
                ax_atr.fill_between(ts, 0, val, color=color, alpha=0.08)
            ax_atr.set_ylabel('ATR', fontsize=8, color=AX_FG)
            ax_atr.legend(fontsize=6, framealpha=0.15, labelcolor=AX_FG, loc='upper left')

        try:
            self.fig_ind.tight_layout(pad=0.6)
        except Exception:
            pass
        self.canvas_ind.draw_idle()

    def _dibujar_wfa(self, wfa, ts):
        if not wfa:
            self.grp_wfa.setVisible(False)
            return
        self.grp_wfa.setVisible(True)
        self.fig_wfa.clear()
        ax = self.fig_wfa.add_subplot(111)
        _style_ax(ax)

        rets = [w['retorno_pct'] or 0.0 for w in wfa]
        dds = [w.get('max_dd_pct', 0.0) or 0.0 for w in wfa]
        wrs = [w['win_rate'] * 100 if w['win_rate'] is not None else 0.0 for w in wfa]
        trades = [w['n_trades'] or 0 for w in wfa]
        x = range(1, len(rets) + 1)

        modo = getattr(self, 'combo_wfa_modo', None)
        idx = modo.currentIndex() if modo is not None else 0

        bars = None
        if idx == 0:  # Retorno %
            colores = [VERDE if r > 0 else ROJO for r in rets]
            bars = ax.bar(x, rets, color=colores, alpha=0.85)
            ax.axhline(0, color=GRIS, linewidth=0.7, linestyle='--')
            ax.set_ylabel('Retorno %', fontsize=8, color=AX_FG)
            _data_vals = rets
        elif idx == 1:  # Retorno acumulado
            acum = np.cumsum(rets)
            ax.plot(x, acum, color=AZUL, linewidth=1.5, marker='o',
                    markersize=5, markerfacecolor=AZUL, markeredgecolor='#111')
            ax.axhline(0, color=GRIS, linewidth=0.7, linestyle='--')
            ax.fill_between(x, 0, acum, where=np.array(acum) >= 0,
                            color=VERDE, alpha=0.15)
            ax.fill_between(x, 0, acum, where=np.array(acum) < 0,
                            color=ROJO, alpha=0.15)
            ax.set_ylabel('Retorno acumulado %', fontsize=8, color=AX_FG)
            _data_vals = acum
        elif idx == 2:  # Max DD %
            dds_abs = [abs(d) for d in dds]
            colores = [ROJO if dds_abs[i] > np.median(dds_abs) else AMBAR for i in range(len(dds_abs))]
            bars = ax.bar(x, dds_abs, color=colores, alpha=0.85)
            ax.set_ylabel('Max DD % (abs)', fontsize=8, color=AX_FG)
            _data_vals = dds_abs
        elif idx == 3:  # Win rate %
            colores = [VERDE if w > 50 else ROJO for w in wrs]
            bars = ax.bar(x, wrs, color=colores, alpha=0.85)
            ax.axhline(50, color=GRIS, linewidth=0.7, linestyle='--')
            ax.set_ylim(0, 100)
            ax.set_ylabel('Win rate %', fontsize=8, color=AX_FG)
            _data_vals = wrs
        elif idx == 4:  # Retorno vs Max DD
            ancho = 0.35
            bars_ret = ax.bar([i - ancho / 2 for i in x], rets, ancho,
                              color=AZUL, alpha=0.7, label='Retorno %')
            dds_abs = [abs(d) for d in dds]
            ax2 = ax.twinx()
            bars_dd = ax2.bar([i + ancho / 2 for i in x], dds_abs, ancho * 0.8,
                              color=ROJO, alpha=0.5, label='Max DD % (abs)')
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
            bars = ax.bar(x, trades, color=colores, alpha=0.85)
            ax.set_ylabel('Nº Trades', fontsize=8, color=AX_FG)
            _data_vals = trades

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

        # ----- lineas de max/min -----
        if idx in (0, 2, 3, 5) and _data_vals:
            mx = max(_data_vals)
            mi = min(_data_vals)
            if mx != mi:
                if idx in (0, 3):
                    ax.axhline(mx, color=VERDE, linewidth=0.6, linestyle=':',
                               alpha=0.5)
                    ax.text(len(rets) + 0.5, mx, f'max', color=VERDE,
                            fontsize=6, va='center')
                    if mi < 0 if idx == 0 else mi < mx:
                        ax.axhline(mi, color=ROJO, linewidth=0.6, linestyle=':',
                                   alpha=0.5)
                        ax.text(len(rets) + 0.5, mi, f'min', color=ROJO,
                                fontsize=6, va='center')

        ax.set_xlabel('Ventana', fontsize=8, color=AX_FG)
        ax.set_xticks(list(x))
        ax.set_xticklabels([str(i) for i in x], fontsize=6.5)

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
            hit = False
            for i in range(len(x)):
                bar_x = x[i]
                bar_h = _data_vals[i] if idx in (0, 2, 3, 5) else 0
                if bars is not None:
                    b = bars[i]
                    bar_x = b.get_x() + b.get_width() / 2
                    bar_h = b.get_height()
                    contains, _ = b.contains(event)
                else:
                    contains = abs(event.xdata - bar_x) < 0.4
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
                    annot.xy = (bar_x, bar_h)
                    annot.set_text('\n'.join(lines))
                    annot.set_visible(True)
                    hit = True
                    break
            if not hit and annot.get_visible():
                annot.set_visible(False)
            if hit:
                self.canvas_wfa.draw_idle()
            elif not hit and annot.get_visible():
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

    def _dibujar_mc(self, mc, capital):
        if not mc or mc['n_sims'] == 0:
            self.grp_mc.setVisible(False)
            return
        self.grp_mc.setVisible(True)
        self.lbl_mc.setText(
            f"Prob. de acabar en negativo: {mc['prob_negativo'] * 100:.1f}%   ·   "
            f"Prob. de ruina (equity < 50% del inicial): {mc['prob_ruina'] * 100:.1f}%   ·   "
            f"Retorno mediano: {(np.median(mc['finales']) / capital - 1) * 100:+.1f}%   ·   "
            f"Max DD mediano: {np.median(mc['max_dds']):.1f}%")

        self.fig_mc.clear()
        ax1 = self.fig_mc.add_subplot(131)
        ax2 = self.fig_mc.add_subplot(132)
        ax3 = self.fig_mc.add_subplot(133)
        for ax in (ax1, ax2, ax3):
            _style_ax(ax)

        cur = mc['curvas_pct']
        x = np.arange(len(cur['p50']))
        ax1.fill_between(x, cur['p5'], cur['p95'], color=AZUL, alpha=0.2)
        ax1.plot(x, cur['p50'], color=AZUL, linewidth=1.0)
        ax1.axhline(capital, color=GRIS, linewidth=0.7, linestyle='--')
        ax1.set_title('Equity p5-p50-p95', fontsize=8, color=AX_FG)
        ax1.set_xlabel('Trade nº', fontsize=7, color=AX_FG)

        ret_fin = (mc['finales'] / capital - 1) * 100
        ax2.hist(ret_fin, bins=40, color=VERDE, alpha=0.8)
        ax2.axvline(0, color=GRIS, linewidth=0.7, linestyle='--')
        ax2.set_title('Retorno final %', fontsize=8, color=AX_FG)

        ax3.hist(mc['max_dds'], bins=40, color=ROJO, alpha=0.8)
        ax3.set_title('Max drawdown %', fontsize=8, color=AX_FG)
        try:
            self.fig_mc.tight_layout(pad=0.6)
        except Exception:
            pass
        self.canvas_mc.draw_idle()


# ══════════════ sub-pestaña Optimizador (comparativa de combinaciones) ══════════════
_ROL_CURVA = Qt.ItemDataRole.UserRole
_ROL_INDICE = Qt.ItemDataRole.UserRole + 1
_ROL_VALOR = Qt.ItemDataRole.UserRole + 2

_COLS_METRICAS_COMBO = [
    ('n_trades', 'Trades', 0, ''),
    ('retorno_pct', 'Retorno', 2, ' %'),
    ('sharpe', 'Sharpe', 2, ''),
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
    PREFIJO_RIESGO + 'stop_atr': ('Stop (× ATR)', 1.0, ''),
    PREFIJO_RIESGO + 'tp_r': ('Take-profit (R)', 1.0, ''),
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

        self.lbl_resumen = QLabel(
            "Lanza «🔍 Prueba de parametrización (Solo IS)» desde la pestaña Constructor")
        self.lbl_resumen.setObjectName("titulo")
        root.addWidget(self.lbl_resumen)

        self.progreso = QProgressBar()
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
        etiqueta_metrica = dict(DialogoOptimizacion.METRICAS).get(
            self._metrica, self._metrica)
        self.lbl_resumen.setText(
            f"{len(self._resultados)} combinaciones probadas sobre IS — "
            f"rankeadas por {etiqueta_metrica} (la primera fila es la mejor)")
        self.btn_stats.setEnabled(bool(self._resultados))
        self._poblar_tabla()
        self._redibujar()
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
                if clave in ('retorno_pct', 'sharpe', 'sqn') and v is not None:
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
        self.setStyleSheet(STYLE_BACKTEST)
        self._threads = []

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.constructor = OptimizadorWidget()
        self.comparativa = ComparativaWidget()
        self.resultados = ResultadosWidget()
        self.tabs.addTab(self.constructor, "  Constructor  ")
        self.tabs.addTab(self.comparativa, "  Optimizador  ")
        self.tabs.addTab(self.resultados, "  Resultados  ")
        root.addWidget(self.tabs)

        self.constructor.ejecutar.connect(self._run_backtest)
        self.constructor.optimizar.connect(self._abrir_dialogo_optimizacion)
        self.comparativa.usar_configuracion.connect(self._usar_configuracion)
        self.comparativa.agregar_setups.connect(self._agregar_setups)

    def refresh_available(self):
        self.constructor.explorer.refresh()

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
            parent=self,
        )
        th.computed.connect(self._on_done)
        th.finished.connect(lambda t=th: self._on_thread_finished(t))
        self._threads.append(th)
        th.start()

    @_no_crash
    def _on_thread_finished(self, th):
        if th in self._threads:
            self._threads.remove(th)
        th.deleteLater()
        self.constructor.progreso.setVisible(False)
        self.constructor.btn_run.setEnabled(True)
        self.constructor.btn_optimizar.setEnabled(bool(self.constructor.csv_path))
        QApplication.restoreOverrideCursor()

    @_no_crash
    def _on_done(self, payload):
        opt = self.constructor
        if 'error' in payload:
            opt.lbl_estado.setText(f"Error: {payload['error']}")
            return
        n_tr = payload['resultado']['n_trades']
        opt.lbl_estado.setText(f"Backtest completado: {n_tr} trades")
        self.resultados.mostrar(payload)
        self.tabs.setCurrentWidget(self.resultados)

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
        opt.lbl_estado.setText("Optimizando parámetros sobre IS…")
        self.comparativa.progreso.setVisible(True)
        self.comparativa.progreso.setValue(0)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        tf_label, regla_resample = opt.tf_resample()
        th = _OptimizerThread(
            opt.csv_path, setup, sweep_params, sweep_riesgo,
            opt.config_global(), opt.slider_oos.value() / 100.0, metrica,
            tf_label=tf_label, regla_resample=regla_resample, parent=self,
        )
        th.progreso.connect(self.comparativa.actualizar_progreso)
        th.terminado.connect(self._on_optimizacion_terminada)
        th.finished.connect(lambda t=th: self._on_thread_finished(t))
        self._threads.append(th)
        th.start()

    @_no_crash
    def _on_optimizacion_terminada(self, payload):
        opt = self.constructor
        self.comparativa.progreso.setVisible(False)
        if 'error' in payload:
            opt.lbl_estado.setText(f"Error: {payload['error']}")
            return
        opt.lbl_estado.setText(
            f"Optimización completada: {len(payload['resultados'])} combinaciones probadas sobre IS")
        self.comparativa.mostrar(payload)
        self.tabs.setCurrentWidget(self.comparativa)

    @_no_crash
    def _usar_configuracion(self, setup):
        self.constructor.cargar_setup_en_constructor(setup)
        self.tabs.setCurrentWidget(self.constructor)

    @_no_crash
    def _agregar_setups(self, setups):
        self.constructor.agregar_setups(setups)
        self.tabs.setCurrentWidget(self.constructor)
