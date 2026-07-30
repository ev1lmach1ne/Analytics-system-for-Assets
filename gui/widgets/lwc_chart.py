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

from core.strategies import (
    sma, ema, rsi, atr, bollinger, stochastic, williams_r, cci, _kama_serie,
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

# paletas de indicadores/osciladores duplicadas de tab_backtest.py
# (_dibujar_principal / _dibujar_panel_oscilador) para que la vista moderna
# se vea igual que la clásica sin crear un import circular entre ambos
# módulos (tab_backtest.py ya importa LwcChart de aquí).
_COLOR_MEDIA_FIJO = {20: '#2B7FFF', 50: '#FF8904', 200: '#800000'}
_PALETA_MA = [_AZUL, _AMBAR, _VERDE, '#9b59b6', '#e67e22']
_BB_COLOR = '#9b59b6'
_KAMA_COLOR = '#ab47bc'
_PAL_RSI = ['#f1c40f', '#e67e22', '#fd79a8']
_PAL_ATR = ['#2ecc71', '#1abc9c']
_PAL_STOCH_K = ['#26c6da', '#4fc3f7', '#80deea']
_PAL_STOCH_D = ['#f06292', '#ec407a', '#f8bbd0']
_PAL_WILLIAMS = ['#ec407a', '#f06292', '#f8bbd0']
_PAL_CCI = ['#5c6bc0', '#7986cb', '#9fa8da']

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
    const linea = chart.addSeries(LightweightCharts.LineSeries, {
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
                mostrar_noticias=False, eventos_noticias=None, indicadores=None):
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
        clásica."""
        self._ensure_view()
        if self.view is None or payload is None:
            return
        (candles, markers, trayectos, stop_track_pts, entrada_track_pts,
         eventos, overlays, bandas, osciladores) = self._construir_datos(
            payload, mostrar_trayecto, mostrar_stop, mostrar_noticias,
            eventos_noticias, indicadores)
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
                          indicadores=None):
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

        def _marcador(t, arriba, color, texto):
            markers.append({'time': int(t), 'text': texto, 'color': color,
                            'position': 'aboveBar' if arriba else 'belowBar',
                            'shape': 'arrowDown' if arriba else 'arrowUp'})

        for r in range(n_tr):
            ie, ix = int(tr['idx_entrada'][r]), int(tr['idx_salida'][r])
            if not (0 <= ie < n and 0 <= ix < n):
                continue
            ent, sal = float(tr['precio_entrada'][r]), float(tr['precio_salida'][r])
            if int(tr['dir'][r]) > 0:      # largo: compra al entrar, venta al salir
                _marcador(unix[ie], False, _VERDE_FLECHA, 'C')
                _marcador(unix[ix], True, _ROJO_FLECHA, 'V')
            else:                          # corto: venta al entrar, compra al salir
                _marcador(unix[ie], True, _ROJO_FLECHA, 'V')
                _marcador(unix[ix], False, _VERDE_FLECHA, 'C')
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
                'position': 'belowBar' if largo else 'aboveBar',
                'shape': 'circle',
            })

        overlays, bandas, osciladores, patron_markers = (
            LwcChart._construir_indicadores(unix, c, h, l, o, indicadores))
        markers += patron_markers

        # dedup: cuando un trade se cierra y otro se abre en la misma vela
        # (ej. reversión) se generan dos marcadores idénticos (mismo time/
        # position/shape/color) que LWC dibuja superpuestos — se ven como un
        # icono "doble".
        vistos = set()
        markers_unicos = []
        for m in markers:
            clave = (m['time'], m['position'], m['shape'], m['color'])
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
    def _construir_indicadores(unix, y, h, l, o, indicadores):
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

        return overlays, bandas, osciladores, patron_markers
