"""Los indicadores nuevos del Backtester (Supertrend, MACD, ADX y los de
solo-editor: Aroon, CMO, TRIX, StochRSI) se representan en el gráfico de
operaciones: Supertrend como overlay sobre el precio, MACD/ADX/Aroon/CMO/
TRIX/StochRSI como paneles inferiores — tanto en la vista clásica
(matplotlib) como en la moderna (Lightweight Charts)."""
import os

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

pytest.importorskip('PyQt6.QtWidgets')
from PyQt6.QtWidgets import QApplication  # noqa: E402

from matplotlib.figure import Figure  # noqa: E402

from gui.widgets.tab_backtest import ResultadosWidget  # noqa: E402
from gui.widgets.lwc_chart import LwcChart  # noqa: E402


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


def _payload():
    rng = np.random.default_rng(21)
    n = 400
    ts = pd.date_range('2024-01-01', periods=n, freq='1h', tz='UTC')
    c = 100 + np.cumsum(rng.normal(0.01, 0.4, n))
    setups = [
        {'nombre': 'ST', 'plantilla': 'Supertrend',
         'params': {'periodo': 10, 'multiplicador': 3.0,
                    'direccion': 'Ambas'}},
        {'nombre': 'MACD', 'plantilla': 'MACD',
         'params': {'rapido': 12, 'lento': 26, 'senal': 9,
                    'direccion': 'Ambas'}},
        {'nombre': 'ADX', 'plantilla': 'ADX (fuerza de tendencia)',
         'params': {'periodo': 14, 'umbral_fuerza': 20.0,
                    'direccion': 'Ambas'}},
        {'nombre': 'custom', 'plantilla': 'Custom (reglas)',
         'params': {'reglas': {'entradas_long': [{'condiciones': [
             {'izq': {'tipo': 'AROON_UP', 'periodo': 25}, 'op': '>',
              'der': {'tipo': 'valor', 'valor': 70.0}},
             {'izq': {'tipo': 'CMO', 'periodo': 14}, 'op': '>',
              'der': {'tipo': 'valor', 'valor': 0.0}},
             {'izq': {'tipo': 'TRIX', 'periodo': 15}, 'op': '>',
              'der': {'tipo': 'valor', 'valor': 0.0}},
             {'izq': {'tipo': 'STOCHRSI', 'periodo': 14}, 'op': '>',
              'der': {'tipo': 'valor', 'valor': 0.5}},
             {'izq': {'tipo': 'MACD_LINEA', 'periodo': 12}, 'op': '>',
              'der': {'tipo': 'valor', 'valor': 0.0}},
             {'izq': {'tipo': 'ADX', 'periodo': 14}, 'op': '>',
              'der': {'tipo': 'valor', 'valor': 20.0}},
         ]}]}}, 'riesgo_pct': 0.01},
    ]
    return {
        'timestamps': ts.values, 'open': c, 'high': c + 0.5,
        'low': c - 0.5, 'close': c, 'corte': 280, 'setups': setups,
        'resultado': {'trades': {k: np.array([]) for k in (
            'idx_entrada', 'idx_salida', 'dir', 'pnl', 'ret_pct',
            'r_multiple', 'setup', 'notional_redondo', 'costo_comision',
            'mfe_r', 'mae_r', 'etd_r', 'eficiencia_entrada',
            'eficiencia_salida')},
            'entradas': None, 'equity': np.full(n, 10000.0),
            'drawdown': np.zeros(n), 'capital_final': 10000.0,
            'n_trades': 0},
    }


def _widget(app):
    w = ResultadosWidget()
    w._payload = _payload()
    return w


def test_recolector_captura_los_nuevos_indicadores(app):
    w = _widget(app)
    ind = w._recolectar_indicadores(w._payload)
    assert (10, 3.0) in ind['supertrends']
    assert (12, 26, 9) in ind['macds']
    assert (14, 20.0) in ind['adxs']
    assert 25 in ind['aroones']
    assert 14 in ind['cmos']
    assert 15 in ind['trixes']
    assert 14 in ind['stochrsis']


def test_paneles_osciladores_nuevos_se_dibujan(app):
    w = _widget(app)
    ind = w._recolectar_indicadores(w._payload)
    fig = Figure()
    ax = fig.add_subplot(111)
    n = len(w._payload['close'])
    x = np.arange(n, dtype=float)
    y = w._payload['close']
    h = w._payload['high']
    l = w._payload['low']
    for kind, clave in (('macd', 'macds'), ('adx', 'adxs'),
                        ('aroon', 'aroones'), ('cmo', 'cmos'),
                        ('trix', 'trixes'), ('stochrsi', 'stochrsis')):
        fig2 = Figure()
        ax2 = fig2.add_subplot(111)
        artes = w._dibujar_panel_oscilador(ax2, kind, ind[clave], x, y, h, l)
        assert artes, kind


def test_lwc_construye_overlay_y_paneles(app):
    p = _payload()
    ind = {
        'mas': set(), 'bbs': set(), 'rsis': set(), 'atrs': set(),
        'patrones': set(), 'stochs': set(), 'williams': set(),
        'ccis': set(), 'kamas': set(), 'ers': set(), 'hursts': set(),
        'supertrends': {(10, 3.0)},
        'macds': {(12, 26, 9)},
        'adxs': {(14, 20.0)},
        'aroones': {25}, 'cmos': {14}, 'trixes': {15}, 'stochrsis': {14},
    }
    unix = (pd.DatetimeIndex(p['timestamps']).asi8 // 10 ** 9).astype(int)
    overlays, bandas, osciladores, marcadores = LwcChart._construir_indicadores(
        unix, p['close'], p['high'], p['low'], p['open'], ind)
    assert len(overlays) == 1                     # el Supertrend
    assert sum(1 for o in osciladores if any(
        s.get('kind') == 'histogram' for s in o['series'])) == 1  # MACD
    assert len(osciladores) >= 6                  # macd/adx/aroon/cmo/trix/stochrsi
    # todos los puntos de las series son finitos (sin NaN en el JSON)
    for osc in osciladores:
        for s in osc['series']:
            assert all(np.isfinite(d['value']) for d in s['data'])
