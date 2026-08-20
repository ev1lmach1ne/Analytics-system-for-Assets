"""
gui/widgets/lwc_chart.py
Prueba aislada (POC) de una visualización estilo TradingView con la librería
open-source Lightweight Charts (Apache 2.0), como alternativa a la gráfica de
matplotlib de la pestaña Resultados. NO sustituye a matplotlib: se muestra
detrás de un botón de conmutación (ver ResultadosWidget en tab_backtest.py).

El JS se sirve embebido offline desde gui/assets/js/ — sin CDN ni internet. Los
datos del backtest (velas + marcadores de operaciones) se inyectan como JSON en
un HTML temporal que carga un QWebEngineView. El motor de backtest sigue siendo
Python; esto solo pinta.

Import tolerante a fallo: si PyQt6-WebEngine no está instalado, WEBENGINE_OK
queda False y el llamador deshabilita el botón en vez de romper la pestaña.
"""
import json
import logging
import os
import tempfile

import numpy as np
import pandas as pd
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import QUrl

from core.backtest import ORDEN_RELLENADA
from core.strategies import (
    sma, ema, rsi, atr, bollinger, stochastic, williams_r, cci, _kama_serie,
    _zigzag_pivotes, tramos_zigzag_vigentes, _er_serie, _hurst_serie,
    _supertrend_serie, _macd_series, _adx_series, _aroon_series, _cmo_serie,
    _trix_serie, _stochrsi_series, _ichimoku_series, _keltner_series,
    _ttm_squeeze_series, _vwap_series,
    UMBRAL_ER_TENDENCIA, UMBRAL_ER_RUIDO,
    UMBRAL_HURST_TENDENCIA, UMBRAL_HURST_REVERSION,
)
from core.candle_patterns import detectar_patrones

try:
    from PyQt6.QtWebEngineWidgets import QWebEngineView
    from PyQt6.QtWebEngineCore import QWebEnginePage
    WEBENGINE_OK = True
except ImportError:      # PyQt6-WebEngine no instalado
    QWebEngineView = None
    QWebEnginePage = None
    WEBENGINE_OK = False

_log = logging.getLogger(__name__)

if WEBENGINE_OK:
    class _PaginaConLog(QWebEnginePage):
        """QWebEnginePage que reenvía los errores de la consola JS al logger
        de Python — sin esto, un fallo en el HTML/JS embebido deja la vista
        moderna en blanco sin ninguna pista de qué salió mal."""

        def javaScriptConsoleMessage(self, level, message, line, source_id):
            if level >= QWebEnginePage.JavaScriptConsoleMessageLevel.WarningMessageLevel:
                _log.warning("lwc_chart JS [%s:%s] %s", source_id, line, message)

# el bundle vive en gui/assets/js/ (un nivel por encima de gui/widgets/)
_DIR_JS = os.path.join(os.path.dirname(__file__), '..', 'assets', 'js')
_JS_LWC = os.path.join(_DIR_JS, 'lightweight-charts.standalone.production.js')

# colores alineados con la gráfica matplotlib (tab_backtest.py). Las flechas
# de compra/venta usan tonos más saturados que las velas para no camuflarse
# sobre cuerpos del mismo color.
_BG = '#0d1424'
_TXT = '#c8d6e5'
_GRID = '#253a60'
_VERDE = '#2ecc71'
_ROJO = '#e74c3c'
_GRIS = '#5a7a9a'
_GRIS_NOTICIA = '#7a8a9a'  # distinto del gris del trayecto para no confundirlos
_VERDE_FLECHA = '#1b8a3a'   # compra (abre largo / cierra corto) — más oscuro
                            # que _VERDE para no confundirse con la vela
_ROJO_FLECHA = '#ff1744'    # venta (abre corto / cierra largo)
_AZUL = '#4fc3f7'
_AMBAR = '#f1c40f'
_AZUL_ZIGZAG = '#2962FF'    # azul del ZigZag de TradingView; distinto de _AZUL,
                            # que abre la paleta de medias
_AMARILLO_FIB = '#c9a227'   # tramos de Fibonacci, apagado para quedar de fondo

# paletas de indicadores/osciladores duplicadas de tab_backtest.py
# (_dibujar_principal / _dibujar_panel_oscilador) para que la vista moderna
# se vea igual que la clásica sin crear un import circular entre ambos
# módulos (tab_backtest.py ya importa LwcChart de aquí).
_COLOR_MEDIA_FIJO = {20: '#2B7FFF', 50: '#FF8904', 200: '#800000'}
_PALETA_MA = [_AZUL, _AMBAR, _VERDE, '#9b59b6', '#e67e22']
_BB_COLOR = '#9b59b6'
_KAMA_COLOR = '#ab47bc'
# mismos colores que COLOR_PANEL_OSC de tab_backtest.py para los paneles de
# régimen, que las dos vistas enseñan el mismo dato
_NARANJA_ER = '#ff9800'
_MORADO_HURST = '#ab47bc'
_PAL_RSI = ['#ffffff', '#e67e22', '#fd79a8']
_PAL_ATR = ['#2ecc71', '#1abc9c']
_PAL_STOCH_K = ['#26c6da', '#4fc3f7', '#80deea']
_PAL_STOCH_D = ['#f06292', '#ec407a', '#f8bbd0']
_PAL_WILLIAMS = ['#ec407a', '#f06292', '#f8bbd0']
_PAL_CCI = ['#5c6bc0', '#7986cb', '#9fa8da']
_COLOR_SUPERTREND = '#d4a5f5'
_PAL_MACD = ['#4fc3f7', '#f1c40f']
_PAL_ADX = ['#7986cb', '#2ecc71', '#e74c3c']
_PAL_AROON = ['#1abc9c', '#16a085']
_PAL_CMO = ['#ffa726', '#ffb74d']
_PAL_TRIX = ['#ff7043', '#ff8a65']
_PAL_STOCHRSI = ['#fd79a8', '#f8bbd0']
_COLOR_ICH_TENKAN = '#00bcd4'
_COLOR_ICH_KIJUN = '#ff7043'
_COLOR_ICH_SENKOU = '#78909c'
_COLOR_KELTNER = '#26a69a'
_PAL_VWAP = ['#9aa7b8', '#7a8ba3', '#b8c4d4', '#5d6f8a']

_HTML = """<!doctype html>
<html><head><meta charset="utf-8">
<style>html,body,#c{height:100%;width:100%;margin:0;background:__BG__;}</style>
<script>__JS__</script>
</head><body>
<div id="c"></div>
<script>
const candles = __CANDLES__;
const markers = __MARKERS__;
const trayectos = __TRAYECTOS__;   // [[{time,value}, {time,value}], ...] por trade
// trayectoria REAL vela a vela (no un segmento fijo por trade): un único
// array [{time,value?}] con huecos "whitespace" (sin 'value') donde no hay
// posición o el setup no tiene stop — mismas series que emite el motor
// (core/backtest.simular) y dibuja la vista clásica con 'steps-post'.
const stopTrack = __STOP_TRACK__;
const entradaTrack = __ENTRADA_TRACK__;
const eventos = __EVENTOS__;       // idem, una línea vertical por evento de noticia
const overlays = __OVERLAYS__;     // [{color, data}] medias/KAMA sobre el precio
const bandas = __BANDAS__;         // [{color, sup, inf}] Bollinger (sin relleno)
const osciladores = __OSCILADORES__;  // [{height, series:[{color,data}], lines:[{value,color}]}]
const chart = LightweightCharts.createChart(document.getElementById('c'), {
  layout: { background: { color: '__BG__' }, textColor: '__TXT__' },
  grid: { vertLines: { color: '__GRID__' }, horzLines: { color: '__GRID__' } },
  timeScale: { timeVisible: true, secondsVisible: false, borderColor: '__GRID__' },
  rightPriceScale: { borderColor: '__GRID__' },
  crosshair: { mode: 0 },
  autoSize: true,
});
const series = chart.addSeries(LightweightCharts.CandlestickSeries, {
  upColor: '__VERDE__', downColor: '__ROJO__',
  wickUpColor: '__VERDE__', wickDownColor: '__ROJO__',
  borderVisible: false,
});
series.setData(candles);
if (markers.length) LightweightCharts.createSeriesMarkers(series, markers);
overlays.forEach(function (o) {
  chart.addSeries(LightweightCharts.LineSeries, {
    color: o.color, lineWidth: 1, lastValueVisible: false,
    priceLineVisible: false, crosshairMarkerVisible: false,
  }).setData(o.data);
});
bandas.forEach(function (b) {
  const opts = { color: b.color, lineWidth: 1, lastValueVisible: false,
                priceLineVisible: false, crosshairMarkerVisible: false };
  chart.addSeries(LightweightCharts.LineSeries, opts).setData(b.sup);
  chart.addSeries(LightweightCharts.LineSeries, opts).setData(b.inf);
});
osciladores.forEach(function (osc, i) {
  const paneIndex = i + 1;
  let primera = null;
  osc.series.forEach(function (s) {
    const esHist = s.kind === 'histogram';
    const linea = chart.addSeries(esHist
        ? LightweightCharts.HistogramSeries
        : LightweightCharts.LineSeries, {
      color: s.color, lineWidth: 1, lastValueVisible: false,
      priceLineVisible: false, crosshairMarkerVisible: false,
    }, paneIndex);
    linea.setData(s.data);
    if (!primera) primera = linea;
  });
  if (primera) {
    osc.lines.forEach(function (l) {
      primera.createPriceLine({ price: l.value, color: l.color, lineWidth: 1,
                                lineStyle: 2, axisLabelVisible: false });
    });
  }
  chart.panes()[paneIndex].setHeight(osc.height);
});
// una serie de línea por operación (LWC no permite segmentos discontinuos
// dentro de una sola serie sin trucos de "whitespace"; para una POC, más
// series simples es más simple y correcto que ese truco)
trayectos.forEach(function (pts) {
  chart.addSeries(LightweightCharts.LineSeries, {
    color: '__GRIS__', lineWidth: 1, lastValueVisible: false,
    priceLineVisible: false, crosshairMarkerVisible: false,
  }).setData(pts);
});
if (stopTrack.length) {
  chart.addSeries(LightweightCharts.LineSeries, {
    color: '__ROJO__', lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dashed,
    lineType: LightweightCharts.LineType.WithSteps,
    lastValueVisible: false, priceLineVisible: false,
    crosshairMarkerVisible: false,
  }).setData(stopTrack);
}
if (entradaTrack.length) {
  chart.addSeries(LightweightCharts.LineSeries, {
    color: '__GRIS__', lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dotted,
    lineType: LightweightCharts.LineType.WithSteps,
    lastValueVisible: false, priceLineVisible: false,
    crosshairMarkerVisible: false,
  }).setData(entradaTrack);
}
eventos.forEach(function (pts) {
  chart.addSeries(LightweightCharts.LineSeries, {
    color: '__GRIS_NOTICIA__', lineWidth: 1, lineStyle: 2,
    lastValueVisible: false, priceLineVisible: false,
    crosshairMarkerVisible: false,
  }).setData(pts);
});
chart.timeScale().fitContent();
</script>
</body></html>"""


class LwcChart(QWidget):
    """Vista Lightweight Charts. Usar mostrar(payload) para pintar un backtest."""

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self._html_path = os.path.join(
            tempfile.gettempdir(), f'lwc_backtest_{id(self)}.html')
        self._js_cache = None
        self.view = None
        self._view_ok = False

    def _ensure_view(self):
        """Crea el QWebEngineView bajo demanda (no en __init__) para no
        disparar procesos Chromium durante el arranque de la app."""
        if self._view_ok or self.view is not None:
            return
        if WEBENGINE_OK:
            self.view = QWebEngineView(self)
            self.view.setPage(_PaginaConLog(self.view))
            self.view.setMinimumHeight(480)
            self.layout().addWidget(self.view)
            self._view_ok = True
        else:
            self._view_ok = True
            aviso = QLabel("PyQt6-WebEngine no está instalado — vista moderna "
                           "no disponible.")
            aviso.setStyleSheet("color: #8fb3d9; padding: 20px;")
            self.layout().addWidget(aviso)

    def _leer_js(self):
        if self._js_cache is None:
            with open(_JS_LWC, encoding='utf-8') as f:
                self._js_cache = f.read()
        return self._js_cache

    def mostrar(self, payload, mostrar_trayecto=True, mostrar_stop=False,
                mostrar_noticias=False, eventos_noticias=None, indicadores=None,
                capas=None):
        """Pinta velas + marcadores de operaciones del payload del backtest
        (mismo dict que consume ResultadosWidget). mostrar_trayecto/mostrar_stop
        reflejan los checkboxes homónimos de la vista clásica, para que ambas
        vistas se comporten igual al conmutar entre ellas. eventos_noticias:
        DataFrame crudo del calendario económico (columna timestamp UTC),
        dibujado como líneas verticales grises discontinuas si
        mostrar_noticias está activo. indicadores: dict devuelto por
        _recolectar_indicadores (tab_backtest.py) — si se pasa, se dibujan
        los mismos overlays (medias/Bollinger/KAMA/patrones) y paneles de
        oscilador (RSI/ATR/Stochastic/Williams %R/CCI) que en la vista
        clásica. capas: {'zigzag','fib','ordenes'} -> bool, los tres ojos de la
        vista clásica; ausente = todas encendidas."""
        self._ensure_view()
        if self.view is None or payload is None:
            return
        (candles, markers, trayectos, stop_track_pts, entrada_track_pts,
         eventos, overlays, bandas, osciladores) = self._construir_datos(
            payload, mostrar_trayecto, mostrar_stop, mostrar_noticias,
            eventos_noticias, indicadores, capas)
        # mismo criterio de alto que el canvas clásico (ver
        # ResultadosWidget._dibujar_principal en tab_backtest.py), para que
        # ambas vistas reserven el mismo espacio al tener paneles de oscilador
        self.view.setMinimumHeight(int(480 + 90 * len(osciladores)))
        html = (_HTML
                .replace('__JS__', self._leer_js())
                .replace('__CANDLES__', json.dumps(candles))
                .replace('__MARKERS__', json.dumps(markers))
                .replace('__TRAYECTOS__', json.dumps(trayectos))
                .replace('__STOP_TRACK__', json.dumps(stop_track_pts))
                .replace('__ENTRADA_TRACK__', json.dumps(entrada_track_pts))
                .replace('__EVENTOS__', json.dumps(eventos))
                .replace('__OVERLAYS__', json.dumps(overlays))
                .replace('__BANDAS__', json.dumps(bandas))
                .replace('__OSCILADORES__', json.dumps(osciladores))
                .replace('__BG__', _BG).replace('__TXT__', _TXT)
                .replace('__GRID__', _GRID).replace('__GRIS__', _GRIS)
                .replace('__GRIS_NOTICIA__', _GRIS_NOTICIA)
                .replace('__VERDE__', _VERDE).replace('__ROJO__', _ROJO))
        with open(self._html_path, 'w', encoding='utf-8') as f:
            f.write(html)
        self.view.load(QUrl.fromLocalFile(self._html_path))

    @staticmethod
    def _construir_datos(payload, mostrar_trayecto=True, mostrar_stop=False,
                          mostrar_noticias=False, eventos_noticias=None,
                          indicadores=None, capas=None):
        ts = pd.DatetimeIndex(payload['timestamps'])
        unix = (ts.asi8 // 1_000_000_000).astype(np.int64)   # segundos UTC
        o = np.asarray(payload['open'], dtype=float)
        h = np.asarray(payload['high'], dtype=float)
        l = np.asarray(payload['low'], dtype=float)
        c = np.asarray(payload['close'], dtype=float)
        candles = [{'time': int(unix[i]), 'open': o[i], 'high': h[i],
                    'low': l[i], 'close': c[i]} for i in range(len(unix))]

        markers, trayectos = [], []
        resultado = payload.get('resultado') or {}
        tr = resultado.get('trades') or {}
        n_tr = len(tr.get('pnl', []))
        n = len(unix)

        def _marcador(t, arriba, color, texto, precio):
            # anclado al PRECIO de ejecución, no al borde de la vela: es lo que
            # permite comprobar a ojo que una orden límite se llenó en su nivel
            # y que una entrada a mercado incorpora el slippage simulado.
            # Contrapartida: con 'atPrice*' la librería deja de apilar los
            # marcadores para evitar solapes, así que dos operaciones a precios
            # parecidos en la misma vela se pisan.
            markers.append({'time': int(t), 'text': texto, 'color': color,
                            'position': 'atPriceMiddle', 'price': float(precio),
                            'shape': 'arrowDown' if arriba else 'arrowUp'})

        for r in range(n_tr):
            ie, ix = int(tr['idx_entrada'][r]), int(tr['idx_salida'][r])
            if not (0 <= ie < n and 0 <= ix < n):
                continue
            ent, sal = float(tr['precio_entrada'][r]), float(tr['precio_salida'][r])
            if int(tr['dir'][r]) > 0:      # largo: compra al entrar, venta al salir
                _marcador(unix[ie], False, _VERDE_FLECHA, 'C', ent)
                _marcador(unix[ix], True, _ROJO_FLECHA, 'V', sal)
            else:                          # corto: venta al entrar, compra al salir
                _marcador(unix[ie], True, _ROJO_FLECHA, 'V', ent)
                _marcador(unix[ix], False, _VERDE_FLECHA, 'C', sal)
            if mostrar_trayecto:
                trayectos.append([{'time': int(unix[ie]), 'value': ent},
                                  {'time': int(unix[ix]), 'value': sal}])

        # stop-loss / precio de entrada vigentes, vela a vela — trayectoria
        # REAL (no un segmento fijo por trade, como antes): huecos
        # ("whitespace", sin 'value') donde no hay posición o el setup no
        # tiene stop, mismas series que emite el motor y dibuja con
        # 'steps-post' la vista clásica (ver ResultadosWidget._dibujar_principal).
        stop_track_pts, entrada_track_pts = [], []
        if mostrar_stop:
            st = resultado.get('stop_track')
            et = resultado.get('entrada_track')
            if st is not None:
                for i in range(min(n, len(st))):
                    v = float(st[i])
                    stop_track_pts.append(
                        {'time': int(unix[i])} if np.isnan(v)
                        else {'time': int(unix[i]), 'value': v})
            if et is not None:
                for i in range(min(n, len(et))):
                    v = float(et[i])
                    entrada_track_pts.append(
                        {'time': int(unix[i])} if np.isnan(v)
                        else {'time': int(unix[i]), 'value': v})

        # tramos 2+ de entrada escalonada (promediar/piramidar): círculo, no
        # flecha, para no confundirlos con la apertura/cierre del trade — el
        # tramo 0 (apertura) ya lo pinta el marcador de arriba. Viven en
        # resultado['entradas'], no en 'trades' (no son un cierre).
        entr = (payload.get('resultado') or {}).get('entradas') or {}
        for k in range(len(entr.get('idx', []))):
            if int(entr['tramo'][k]) == 0:
                continue
            ik = int(entr['idx'][k])
            if not (0 <= ik < n):
                continue
            largo = int(entr['dir'][k]) > 0
            markers.append({
                'time': int(unix[ik]), 'text': 'T',
                'color': _VERDE_FLECHA if largo else _ROJO_FLECHA,
                'position': 'atPriceMiddle', 'price': float(entr['precio'][k]),
                'shape': 'circle',
            })

        overlays, bandas, osciladores, patron_markers = (
            LwcChart._construir_indicadores(unix, c, h, l, o, indicadores,
                                            payload.get('volume')))
        markers += patron_markers
        overlays += LwcChart._construir_capas_limite(
            unix, h, l, resultado, indicadores, capas)

        # dedup: cuando un trade se cierra y otro se abre en la misma vela
        # (ej. reversión) se generan dos marcadores idénticos (mismo time/
        # position/shape/color) que LWC dibuja superpuestos — se ven como un
        # icono "doble".
        vistos = set()
        markers_unicos = []
        for m in markers:
            # el precio entra en la clave: ahora todos los marcadores comparten
            # position='atPriceMiddle', así que sin él dos ejecuciones a precios
            # distintos en la misma vela se descartarían como duplicadas
            clave = (m['time'], m.get('price'), m['shape'], m['color'])
            if clave in vistos:
                continue
            vistos.add(clave)
            markers_unicos.append(m)
        markers = markers_unicos
        markers.sort(key=lambda m: m['time'])   # LWC exige orden temporal

        eventos = []
        if mostrar_noticias and eventos_noticias is not None and len(eventos_noticias) and n:
            lo, hi = float(l.min()), float(h.max())
            ts_ini, ts_fin = ts[0], ts[-1]
            # payload['timestamps'] llega como numpy datetime64 naive (instante
            # UTC sin etiqueta de tz, efecto de .values sobre una columna
            # pandas tz-aware); se alinea quitando la tz del lado de eventos
            # para poder comparar ambos índices.
            ev_ts = pd.DatetimeIndex(eventos_noticias['timestamp']).tz_convert(None)
            visibles = ev_ts[(ev_ts >= ts_ini) & (ev_ts <= ts_fin)]
            ev_unix = (visibles.asi8 // 1_000_000_000).astype(np.int64)
            for t_ev in ev_unix:
                eventos.append([{'time': int(t_ev), 'value': lo},
                                 {'time': int(t_ev), 'value': hi}])
        return (candles, markers, trayectos, stop_track_pts, entrada_track_pts,
                eventos, overlays, bandas, osciladores)

    @staticmethod
    def _construir_capas_limite(unix, h, l, resultado, indicadores, capas):
        """Series de las tres capas de la entrada por orden límite: polilínea
        del ZigZag, tramos de Fibonacci de cada orden ejecutada y las propias
        órdenes (estas sí, todas, con su color según el desenlace).

        Devuelve overlays con el mismo formato que _construir_indicadores, para
        que la vista moderna enseñe lo mismo que la clásica (ver
        ResultadosWidget._dibujar_principal en tab_backtest.py).

        Los segmentos que no cubren todo el histórico se emiten como series de
        longitud n con huecos ("whitespace", punto sin 'value'), igual que
        stop_track más arriba: es como Lightweight Charts corta una línea.
        """
        capas = capas or {}
        overlays = []
        if not indicadores:
            return overlays
        n = len(unix)
        df_zz = pd.DataFrame({'high': h, 'low': l})

        def _serie_con_huecos(valores):
            return [{'time': int(unix[i])} if not np.isfinite(valores[i])
                    else {'time': int(unix[i]), 'value': float(valores[i])}
                    for i in range(n)]

        if capas.get('zigzag', True):
            for desviacion, piernas in sorted(indicadores.get('zigzags', ())):
                # puntos ralos: LineSeries une los pivotes consecutivos, así que
                # no hace falta rellenar las velas intermedias
                datos = [{'time': int(unix[i]), 'value': float(precio)}
                         for i, _conf, precio, _tipo
                         in _zigzag_pivotes(df_zz, desviacion, piernas)
                         if i < n]
                if datos:
                    overlays.append({'color': _AZUL_ZIGZAG, 'data': datos})

        ol = (resultado or {}).get('ordenes_limite') or {}
        n_ordenes = len(ol.get('idx_alta', ()))
        if not n_ordenes:
            return overlays

        # solo las órdenes rellenadas dibujan su swing, igual que en la vista
        # clásica (ver _dibujar_principal en tab_backtest.py): colocar la orden
        # no significa que el retroceso llegara a producirse
        fibs = indicadores.get('fibs') or {}
        if capas.get('fib', True) and fibs:
            origen = np.full(n, np.nan)
            extremo = np.full(n, np.nan)
            cache = {}
            for k in range(n_ordenes):
                if int(ol['resultado'][k]) != ORDEN_RELLENADA:
                    continue
                sid = int(ol['setup'][k])
                if sid not in fibs:
                    continue
                desv, piernas, _nivel = fibs[sid]
                if (desv, piernas) not in cache:
                    cache[(desv, piernas)] = tramos_zigzag_vigentes(
                        df_zz, desv, piernas)
                tramos = cache[(desv, piernas)]
                i0, i1 = int(ol['idx_alta'][k]), int(ol['idx_fin'][k])
                if i0 >= n or i1 >= n:
                    continue
                origen[i0:i1 + 1] = tramos['anterior'][i0]
                extremo[i0:i1 + 1] = tramos['ultimo'][i0]
            if np.isfinite(origen).any():
                overlays.append({'color': _AMARILLO_FIB,
                                 'data': _serie_con_huecos(origen)})
                overlays.append({'color': _AMARILLO_FIB,
                                 'data': _serie_con_huecos(extremo)})

        if capas.get('ordenes', True):
            # una serie por desenlace, para que cada una lleve su color
            for codigo, color in ((0, _VERDE), (1, _GRIS), (2, _AMBAR)):
                precios = np.full(n, np.nan)
                for k in range(n_ordenes):
                    if int(ol['resultado'][k]) != codigo:
                        continue
                    i0, i1 = int(ol['idx_alta'][k]), int(ol['idx_fin'][k])
                    if i0 >= n or i1 >= n:
                        continue
                    precios[i0:i1 + 1] = float(ol['precio'][k])
                if np.isfinite(precios).any():
                    overlays.append({'color': color,
                                     'data': _serie_con_huecos(precios)})
        return overlays

    @staticmethod
    def _construir_indicadores(unix, y, h, l, o, indicadores, volume=None):
        """Calcula, a partir del dict que devuelve _recolectar_indicadores
        (tab_backtest.py), las series de overlay (medias/KAMA), bandas
        (Bollinger) y paneles de oscilador para la vista moderna, más
        marcadores extra de patrones de velas. Duplica el cálculo/paleta ya
        usado en la vista clásica (_dibujar_principal /
        _dibujar_panel_oscilador de tab_backtest.py) — sin relleno sombreado
        para Bollinger, que no tiene equivalente directo en Lightweight
        Charts sin un plugin extra."""
        overlays, bandas, patron_markers = [], [], []
        if not indicadores:
            return overlays, bandas, [], patron_markers

        def _serie(val):
            return [{'time': int(unix[i]), 'value': float(val[i])}
                    for i in range(len(unix)) if np.isfinite(val[i])]

        idx_paleta = 0
        for tipo, per in sorted(indicadores.get('mas', ()), key=lambda x: x[1]):
            color = _COLOR_MEDIA_FIJO.get(per)
            if color is None:
                color = _PALETA_MA[idx_paleta % len(_PALETA_MA)]
                idx_paleta += 1
            f = ema if tipo == 'EMA' else sma
            overlays.append({'color': color, 'data': _serie(f(y, per))})

        for per, desv in indicadores.get('bbs', ()):
            _media, sup, inf = bollinger(y, per, desv)
            bandas.append({'color': _BB_COLOR, 'sup': _serie(sup), 'inf': _serie(inf)})

        for per_er, rapido, lento in indicadores.get('kamas', ()):
            overlays.append({'color': _KAMA_COLOR,
                              'data': _serie(_kama_serie(y, per_er, rapido, lento))})

        for per, mult in sorted(indicadores.get('supertrends', ()), key=lambda x: (x[0], x[1])):
            st, _tend = _supertrend_serie(h, l, y, per, mult)
            overlays.append({'color': _COLOR_SUPERTREND, 'data': _serie(st)})

        for tenkan, kijun, senkou in sorted(indicadores.get('ichimokus', ())):
            t_v, k_v, sa, sb, _ch = _ichimoku_series(h, l, y, tenkan, kijun, senkou)
            for color, serie_v in ((_COLOR_ICH_TENKAN, t_v),
                                   (_COLOR_ICH_KIJUN, k_v),
                                   (_COLOR_ICH_SENKOU, sa),
                                   (_COLOR_ICH_SENKOU, sb)):
                overlays.append({'color': color, 'data': _serie(serie_v)})

        for per, mult in sorted(indicadores.get('keltners', ())):
            media_kc, sup_kc, inf_kc = _keltner_series(y, h, l, per, mult)
            for serie_v in (media_kc, sup_kc, inf_kc):
                overlays.append({'color': _COLOR_KELTNER, 'data': _serie(serie_v)})

        vwaps = indicadores.get('vwaps', ())
        if vwaps and volume is not None:
            ts_idx = pd.DatetimeIndex(
                pd.to_datetime(unix, unit='s', utc=True))
            df_v = pd.DataFrame({'timestamp': ts_idx, 'high': h, 'low': l,
                                 'close': y, 'volume': volume})
            for i, (anclaje, k, modo) in enumerate(sorted(vwaps)):
                r = _vwap_series(df_v, anclaje, k, modo)
                color_v = _PAL_VWAP[i % len(_PAL_VWAP)]
                overlays.append({'color': color_v, 'data': _serie(r['media'])})
                overlays.append({'color': color_v, 'data': _serie(r['sup'])})
                overlays.append({'color': color_v, 'data': _serie(r['inf'])})

        patrones_set = indicadores.get('patrones', ())
        if patrones_set:
            detectados = detectar_patrones(o, h, l, y)
            for nombre in patrones_set:
                occ = detectados.get(nombre)
                if occ is None:
                    continue
                idx, dirs = occ['idx'], occ['dir']
                filtro = (idx >= 0) & (idx < len(unix))
                for i, d in zip(idx[filtro], dirs[filtro]):
                    patron_markers.append({
                        'time': int(unix[i]), 'text': '', 'shape': 'circle',
                        'position': 'belowBar' if d > 0 else 'aboveBar',
                        'color': _VERDE if d > 0 else _ROJO})

        osciladores = []

        rsis = indicadores.get('rsis', ())
        if rsis:
            panel = {'height': 120, 'series': [], 'lines': []}
            for i, per in enumerate(sorted(rsis)):
                panel['series'].append({'color': _PAL_RSI[i % len(_PAL_RSI)],
                                        'data': _serie(rsi(y, per))})
            panel['lines'] = [{'value': 70, 'color': _ROJO},
                              {'value': 50, 'color': _GRIS},
                              {'value': 30, 'color': _VERDE}]
            osciladores.append(panel)

        atrs = indicadores.get('atrs', ())
        if atrs:
            panel = {'height': 120, 'series': [], 'lines': []}
            for i, per in enumerate(sorted(atrs)):
                panel['series'].append({'color': _PAL_ATR[i % len(_PAL_ATR)],
                                        'data': _serie(atr(h, l, y, per))})
            osciladores.append(panel)

        stochs = indicadores.get('stochs', ())
        if stochs:
            panel = {'height': 120, 'series': [], 'lines': []}
            for i, (per_k, suav_k, per_d, sobreventa, sobrecompra) in enumerate(sorted(stochs)):
                k, d = stochastic(h, l, y, per_k, suav_k, per_d)
                panel['series'].append({'color': _PAL_STOCH_K[i % len(_PAL_STOCH_K)],
                                        'data': _serie(k)})
                panel['series'].append({'color': _PAL_STOCH_D[i % len(_PAL_STOCH_D)],
                                        'data': _serie(d)})
                panel['lines'] += [{'value': sobrecompra, 'color': _ROJO},
                                   {'value': sobreventa, 'color': _VERDE}]
            osciladores.append(panel)

        williams = indicadores.get('williams', ())
        if williams:
            panel = {'height': 120, 'series': [], 'lines': []}
            for i, (per, sobreventa, sobrecompra) in enumerate(sorted(williams)):
                panel['series'].append({'color': _PAL_WILLIAMS[i % len(_PAL_WILLIAMS)],
                                        'data': _serie(williams_r(h, l, y, per))})
                panel['lines'] += [{'value': sobrecompra, 'color': _ROJO},
                                   {'value': sobreventa, 'color': _VERDE}]
            osciladores.append(panel)

        ccis = indicadores.get('ccis', ())
        if ccis:
            panel = {'height': 120, 'series': [], 'lines': []}
            for i, (per, sobreventa, sobrecompra) in enumerate(sorted(ccis)):
                panel['series'].append({'color': _PAL_CCI[i % len(_PAL_CCI)],
                                        'data': _serie(cci(h, l, y, per))})
                panel['lines'] += [{'value': sobrecompra, 'color': _ROJO},
                                   {'value': 0, 'color': _GRIS},
                                   {'value': sobreventa, 'color': _VERDE}]
            osciladores.append(panel)

        macds = indicadores.get('macds', ())
        if macds:
            panel = {'height': 120, 'series': [], 'lines': []}
            for i, (rapido, lento, senal) in enumerate(sorted(macds)):
                linea, senal_l, hist = _macd_series(y, rapido, lento, senal)
                # histograma con color por barra (verde/rojo según el signo)
                datos_hist = [
                    {'time': int(unix[i]), 'value': float(hist[i]),
                     'color': _VERDE if hist[i] >= 0 else _ROJO}
                    for i in range(len(unix)) if np.isfinite(hist[i])]
                panel['series'].append({'kind': 'histogram',
                                        'color': _VERDE, 'data': datos_hist})
                panel['series'].append({'color': _PAL_MACD[0],
                                        'data': _serie(linea)})
                panel['series'].append({'color': _PAL_MACD[1],
                                        'data': _serie(senal_l)})
                panel['lines'] += [{'value': 0, 'color': _GRIS}]
            osciladores.append(panel)

        adxs = indicadores.get('adxs', ())
        if adxs:
            panel = {'height': 120, 'series': [], 'lines': []}
            for i, (per, umbral) in enumerate(sorted(adxs)):
                adx, pdi, mdi = _adx_series(h, l, y, per)
                panel['series'].append({'color': _PAL_ADX[0], 'data': _serie(adx)})
                panel['series'].append({'color': _PAL_ADX[1], 'data': _serie(pdi)})
                panel['series'].append({'color': _PAL_ADX[2], 'data': _serie(mdi)})
                panel['lines'] += [{'value': umbral, 'color': _AMBAR}]
            osciladores.append(panel)

        aroones = indicadores.get('aroones', ())
        if aroones:
            panel = {'height': 120, 'series': [], 'lines': []}
            for i, per in enumerate(sorted(aroones)):
                up, dn = _aroon_series(h, l, per)
                panel['series'].append({'color': _PAL_AROON[0], 'data': _serie(up)})
                panel['series'].append({'color': _PAL_AROON[1], 'data': _serie(dn)})
                panel['lines'] += [{'value': 70, 'color': _ROJO},
                                   {'value': 30, 'color': _VERDE}]
            osciladores.append(panel)

        cmos = indicadores.get('cmos', ())
        if cmos:
            panel = {'height': 120, 'series': [], 'lines': []}
            for i, per in enumerate(sorted(cmos)):
                panel['series'].append({'color': _PAL_CMO[i % len(_PAL_CMO)],
                                        'data': _serie(_cmo_serie(y, per))})
                panel['lines'] += [{'value': 0, 'color': _GRIS}]
            osciladores.append(panel)

        trixes = indicadores.get('trixes', ())
        if trixes:
            panel = {'height': 120, 'series': [], 'lines': []}
            for i, per in enumerate(sorted(trixes)):
                panel['series'].append({'color': _PAL_TRIX[i % len(_PAL_TRIX)],
                                        'data': _serie(_trix_serie(y, per))})
                panel['lines'] += [{'value': 0, 'color': _GRIS}]
            osciladores.append(panel)

        stochrsis = indicadores.get('stochrsis', ())
        if stochrsis:
            panel = {'height': 120, 'series': [], 'lines': []}
            for i, per in enumerate(sorted(stochrsis)):
                k, d = _stochrsi_series(y, per)
                panel['series'].append({'color': _PAL_STOCHRSI[0], 'data': _serie(k)})
                panel['series'].append({'color': _PAL_STOCHRSI[1], 'data': _serie(d)})
                panel['lines'] += [{'value': 0.8, 'color': _ROJO},
                                   {'value': 0.2, 'color': _VERDE}]
            osciladores.append(panel)

        ttms = indicadores.get('ttms', ())
        if ttms:
            panel = {'height': 120, 'series': [], 'lines': []}
            for i, (per, mult_bb, mult_kc) in enumerate(sorted(ttms)):
                sq, mom = _ttm_squeeze_series(y, h, l, per, mult_bb, mult_kc)
                sq_f = np.where(np.isfinite(mom), np.asarray(sq, dtype=float), np.nan)
                panel['series'].append({'color': _GRIS, 'data': _serie(sq_f)})
                datos_hist = [
                    {'time': int(unix[i]), 'value': float(mom[i]),
                     'color': _VERDE if mom[i] >= 0 else _ROJO}
                    for i in range(len(unix)) if np.isfinite(mom[i])]
                panel['series'].append({'kind': 'histogram', 'color': _VERDE,
                                        'data': datos_hist})
                panel['lines'] += [{'value': 0, 'color': _GRIS}]
            osciladores.append(panel)

        vwaps = indicadores.get('vwaps', ())
        if vwaps and volume is not None:
            ts_idx = pd.DatetimeIndex(pd.to_datetime(unix, unit='s', utc=True))
            df_v = pd.DataFrame({'timestamp': ts_idx, 'high': h, 'low': l,
                                 'close': y, 'volume': volume})
            panel = {'height': 120, 'series': [], 'lines': []}
            for i, (anclaje, k, modo) in enumerate(sorted(vwaps)):
                r = _vwap_series(df_v, anclaje, k, modo)
                sd_pos = np.where(r['sd'] > 0, r['sd'], np.nan)
                dist = (y - r['media']) / sd_pos
                color_v = _PAL_VWAP[i % len(_PAL_VWAP)]
                panel['series'].append({'color': color_v, 'data': _serie(dist)})
            for ref in (1, 2, 3):
                panel['lines'] += [{'value': ref, 'color': _ROJO},
                                   {'value': -ref, 'color': _VERDE}]
            panel['lines'] += [{'value': 0, 'color': _GRIS}]
            osciladores.append(panel)

        # régimen ER / Hurst: mismas series y umbrales que la vista clásica
        # (ver _dibujar_panel_regimen en tab_backtest.py), o conmutar de vista
        # enseñaría un régimen distinto del que aplica el filtro
        for clave, color, umbrales in (
                ('ers', _NARANJA_ER, (UMBRAL_ER_TENDENCIA, UMBRAL_ER_RUIDO)),
                ('hursts', _MORADO_HURST,
                 (UMBRAL_HURST_TENDENCIA, UMBRAL_HURST_REVERSION))):
            datos = indicadores.get(clave, ())
            if not datos:
                continue
            panel = {'height': 120, 'series': [], 'lines': []}
            for periodo, _metodo in sorted(datos):
                val = (_er_serie(y, periodo).values if clave == 'ers'
                       else _hurst_serie(y, periodo))
                if val is None:
                    continue
                panel['series'].append({'color': color, 'data': _serie(val)})
            if panel['series']:
                panel['lines'] = [{'value': umbrales[0], 'color': _VERDE},
                                  {'value': umbrales[1], 'color': _ROJO}]
                osciladores.append(panel)

        return overlays, bandas, osciladores, patron_markers
