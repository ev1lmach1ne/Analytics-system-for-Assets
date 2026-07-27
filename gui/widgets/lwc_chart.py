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
import os
import tempfile

import numpy as np
import pandas as pd
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import QUrl

try:
    from PyQt6.QtWebEngineWidgets import QWebEngineView
    WEBENGINE_OK = True
except ImportError:      # PyQt6-WebEngine no instalado
    QWebEngineView = None
    WEBENGINE_OK = False

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
_VERDE_FLECHA = '#00e676'   # compra (abre largo / cierra corto)
_ROJO_FLECHA = '#ff1744'    # venta (abre corto / cierra largo)

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
const stops = __STOPS__;           // idem, nivel de stop-loss por trade
const eventos = __EVENTOS__;       // idem, una línea vertical por evento de noticia
const chart = LightweightCharts.createChart(document.getElementById('c'), {
  layout: { background: { color: '__BG__' }, textColor: '__TXT__' },
  grid: { vertLines: { color: '__GRID__' }, horzLines: { color: '__GRID__' } },
  timeScale: { timeVisible: true, secondsVisible: false, borderColor: '__GRID__' },
  rightPriceScale: { borderColor: '__GRID__' },
  crosshair: { mode: 0 },
  autoSize: true,
});
const series = chart.addCandlestickSeries({
  upColor: '__VERDE__', downColor: '__ROJO__',
  wickUpColor: '__VERDE__', wickDownColor: '__ROJO__',
  borderVisible: false,
});
series.setData(candles);
if (markers.length) series.setMarkers(markers);
// una serie de línea por operación (LWC no permite segmentos discontinuos
// dentro de una sola serie sin trucos de "whitespace"; para una POC, más
// series simples es más simple y correcto que ese truco)
trayectos.forEach(function (pts) {
  chart.addLineSeries({ color: '__GRIS__', lineWidth: 1, lastValueVisible: false,
                        priceLineVisible: false, crosshairMarkerVisible: false })
       .setData(pts);
});
stops.forEach(function (pts) {
  chart.addLineSeries({ color: '__ROJO__', lineWidth: 1, lineStyle: 2,
                        lastValueVisible: false, priceLineVisible: false,
                        crosshairMarkerVisible: false })
       .setData(pts);
});
eventos.forEach(function (pts) {
  chart.addLineSeries({ color: '__GRIS_NOTICIA__', lineWidth: 1, lineStyle: 2,
                        lastValueVisible: false, priceLineVisible: false,
                        crosshairMarkerVisible: false })
       .setData(pts);
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
            self.view.setMinimumHeight(300)
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
                mostrar_noticias=False, eventos_noticias=None):
        """Pinta velas + marcadores de operaciones del payload del backtest
        (mismo dict que consume ResultadosWidget). mostrar_trayecto/mostrar_stop
        reflejan los checkboxes homónimos de la vista clásica, para que ambas
        vistas se comporten igual al conmutar entre ellas. eventos_noticias:
        DataFrame crudo del calendario económico (columna timestamp UTC),
        dibujado como líneas verticales grises discontinuas si
        mostrar_noticias está activo."""
        self._ensure_view()
        if self.view is None or payload is None:
            return
        candles, markers, trayectos, stops, eventos = self._construir_datos(
            payload, mostrar_trayecto, mostrar_stop, mostrar_noticias, eventos_noticias)
        html = (_HTML
                .replace('__JS__', self._leer_js())
                .replace('__CANDLES__', json.dumps(candles))
                .replace('__MARKERS__', json.dumps(markers))
                .replace('__TRAYECTOS__', json.dumps(trayectos))
                .replace('__STOPS__', json.dumps(stops))
                .replace('__EVENTOS__', json.dumps(eventos))
                .replace('__BG__', _BG).replace('__TXT__', _TXT)
                .replace('__GRID__', _GRID).replace('__GRIS__', _GRIS)
                .replace('__GRIS_NOTICIA__', _GRIS_NOTICIA)
                .replace('__VERDE__', _VERDE).replace('__ROJO__', _ROJO))
        with open(self._html_path, 'w', encoding='utf-8') as f:
            f.write(html)
        self.view.load(QUrl.fromLocalFile(self._html_path))

    @staticmethod
    def _construir_datos(payload, mostrar_trayecto=True, mostrar_stop=False,
                          mostrar_noticias=False, eventos_noticias=None):
        ts = pd.DatetimeIndex(payload['timestamps'])
        unix = (ts.asi8 // 1_000_000_000).astype(np.int64)   # segundos UTC
        o = np.asarray(payload['open'], dtype=float)
        h = np.asarray(payload['high'], dtype=float)
        l = np.asarray(payload['low'], dtype=float)
        c = np.asarray(payload['close'], dtype=float)
        candles = [{'time': int(unix[i]), 'open': o[i], 'high': h[i],
                    'low': l[i], 'close': c[i]} for i in range(len(unix))]

        markers, trayectos, stops = [], [], []
        tr = (payload.get('resultado') or {}).get('trades') or {}
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
            if mostrar_stop and 'precio_stop' in tr:
                stop = float(tr['precio_stop'][r])
                if stop > 0:
                    stops.append([{'time': int(unix[ie]), 'value': stop},
                                  {'time': int(unix[ix]), 'value': stop}])
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
        return candles, markers, trayectos, stops, eventos
