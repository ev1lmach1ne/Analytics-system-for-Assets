import matplotlib
matplotlib.use('Agg')

import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backend_bases import MouseEvent
from matplotlib.dates import date2num
from matplotlib.figure import Figure

from gui.widgets.analisis_graficos import (
    _dib_precio_equity, _dib_estacionalidad, _dib_riesgo_dia_anual,
    _dib_qqplot, _dib_riesgo_intradia, _dib_perfil_horario, _dib_corr24,
    _dib_heatmaps, _dib_regimen_er, _dib_dependencia, _dib_natr_corr,
    _dib_dashboard_natr,
)


def _figura():
    fig = Figure(figsize=(8, 6))
    canvas = FigureCanvasAgg(fig)
    return fig, canvas


def _hover(canvas, ax, xdata, ydata=None):
    ev = MouseEvent('motion_notify_event', canvas, 300, 200)
    ev.xdata = xdata
    ev.ydata = ydata
    ev.inaxes = ax
    canvas.callbacks.process('motion_notify_event', ev)


def _verticales(eje):
    return [l for l in eje.lines
            if len(l.get_xdata()) == 2 and l.get_xdata()[0] == l.get_xdata()[1]]


def _textos_visibles(eje):
    return [t.get_text() for t in eje.texts if t.get_visible()]


def _eje_por_titulo(fig, fragmento):
    """plot_pacf de statsmodels crea un eje vacío extra en la figura, así que
    los ejes no se localizan por índice sino por su título."""
    return next(a for a in fig.axes if fragmento in a.get_title())


def _fechas(n=40):
    return np.arange(np.datetime64('2024-01-01'), n, dtype='datetime64[D]')


# ── Resumen ────────────────────────────────────────────────────────
def test_precio_equity_vertical_discontinua():
    fig, canvas = _figura()
    n = 40
    b = {'precio_equity': {
        'x': _fechas(n), 'close': np.linspace(100, 130, n),
        'sma': np.linspace(101, 129, n), 'equity': np.linspace(1.0, 1.3, n),
        'dd_pct': np.linspace(0, -10, n)}}
    _dib_precio_equity(fig, b, None)
    fig.canvas.draw()
    ax = fig.axes[0]
    assert _verticales(ax)[0].get_linestyle() == '--'


def test_estacionalidad_crosshair():
    fig, canvas = _figura()
    b = {'estacionalidad': {'disponible': True,
                            'meses': [0.5, -1.2, 2.3, 0.0, 0.8, -0.4],
                            'meses_labels': ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun'],
                            'dias': [0.2, -0.3, 0.1, 0.0, -0.1],
                            'dias_labels': ['Lun', 'Mar', 'Mié', 'Jue', 'Vie']}}
    _dib_estacionalidad(fig, b, None)
    fig.canvas.draw()
    ax = fig.axes[0]
    _hover(canvas, ax, 1.0, 1.0)
    v = _verticales(ax)
    assert v and v[0].get_visible()
    assert any('Feb → -1.20%' in t for t in _textos_visibles(ax))


# ── Riesgo ─────────────────────────────────────────────────────────
def test_campanas_y_qq_crosshair():
    fig, canvas = _figura()
    b = {'riesgo_dia_anual': {
        'diario': {'media': 0.001, 'std': 0.01, 'sigmas': {}},
        'anual': {'media': 0.1, 'std': 0.2, 'sigmas': {}},
        'qq': {'osm': np.linspace(-3, 3, 50),
               'osr': np.linspace(-0.05, 0.05, 50),
               'slope': 0.01, 'intercept': 0.0, 'r_sq': 0.95}}}
    _dib_riesgo_dia_anual(fig, b, None)
    fig.canvas.draw()
    ax = fig.axes[0]
    _hover(canvas, ax, 0.001, 0.5)     # sobre el pico de la campana diaria
    v = _verticales(ax)
    assert v and v[0].get_visible()
    assert any('Retorno:' in t for t in _textos_visibles(ax))

    fig2, canvas2 = _figura()
    _dib_qqplot(fig2, b, None)
    fig2.canvas.draw()
    _hover(canvas2, fig2.axes[0], 0.0, 0.0)
    assert any('Q. teórico' in t for t in _textos_visibles(fig2.axes[0]))


def test_riesgo_intradia_campanas_y_rolling_var():
    fig, canvas = _figura()
    x = _fechas(60)
    b = {'riesgo_intradia': {
        'media': 0.0002, 'std': 0.002, 'sigmas': {}, 'es_normal': True,
        'var': {}, 'p05_pct': -0.3, 'p01_pct': -0.5,
        'rolling': {'x': x, 'var95_pct': np.linspace(-0.2, -0.3, 60),
                    'var99_pct': np.linspace(-0.4, -0.5, 60),
                    'etiqueta_ventana': '20d'}}}
    _dib_riesgo_intradia(fig, b, None)
    fig.canvas.draw()
    ax = fig.axes[2]                    # panel Rolling VaR
    _hover(canvas, ax, date2num(x[30]), -0.2)
    v = _verticales(ax)
    assert v and v[0].get_visible()
    assert any('VaR 95%:' in t for t in _textos_visibles(ax))


# ── Intradía ───────────────────────────────────────────────────────
def _bxp_stats(n=24):
    stats = []
    for _ in range(n):
        stats.append({'med': 0.1, 'q1': 0.05, 'q3': 0.2,
                      'whislo': 0.01, 'whishi': 0.4, 'mean': 0.12,
                      'fliers': np.array([]), 'n': 100})
    return stats


def test_perfil_horario_crosshair():
    fig, canvas = _figura()
    b = {'perfil_horario': {'disponible': True, 'bxp': _bxp_stats(),
                            'medias': np.zeros(24), 'n_por_hora': np.ones(24, dtype=int),
                            'vol_media_total': 0.1,
                            'ret_por_hora': np.array([0.0] * 6 + [0.5] + [0.0] * 17)}}
    _dib_perfil_horario(fig, b, None)
    fig.canvas.draw()
    ax = fig.axes[0]
    _hover(canvas, ax, 6.0, 0.3)
    v = _verticales(ax)
    assert v and v[0].get_visible()
    assert any('Hora 06:00' in t for t in _textos_visibles(ax))
    assert any('Retorno acum.: +0.50%' in t for t in _textos_visibles(fig.axes[1]))


def test_corr24_y_heatmaps_hover_celdas():
    rng = np.random.default_rng(1)
    m = rng.normal(0, 0.5, (24, 24))
    m = np.tril(m, -1) + m.T
    np.fill_diagonal(m, 1.0)
    b = {'corr_24': {'m': m, 'labels': [str(h) for h in range(24)]}}
    fig, canvas = _figura()
    _dib_corr24(fig, b, None)
    fig.canvas.draw()
    _hover(canvas, fig.axes[0], 3.0, 14.0)   # celda fila 14, col 3
    textos = _textos_visibles(fig.axes[0])
    assert any('14 × 3 → ' in t for t in textos)

    b2 = {'heatmaps': {
        'week': {'m': np.random.default_rng(2).normal(0.1, 0.05, (5, 24)),
                 'filas': ['Lun', 'Mar', 'Mié', 'Jue', 'Vie'],
                 'cols': [str(h) for h in range(24)], 'vmax': 0.2},
        'month': {'m': np.random.default_rng(3).normal(0.1, 0.05, (3, 24)),
                  'filas': ['Ene', 'Feb', 'Mar'],
                  'cols': [str(h) for h in range(24)], 'vmax': 0.2}}}
    fig2, canvas2 = _figura()
    _dib_heatmaps(fig2, b2, None)
    fig2.canvas.draw()
    _hover(canvas2, fig2.axes[0], 14.0, 2.0)   # Mié × 14
    assert any('Mié × 14 → ' in t for t in _textos_visibles(fig2.axes[0]))


# ── Régimen ER: crosshair + sync X ─────────────────────────────────
def test_regimen_er_crosshair_y_sync_x():
    fig, canvas = _figura()
    x = _fechas(50)
    close = np.linspace(100, 130, 50)
    er = np.linspace(0.1, 0.9, 50)
    b = {'regimen_er': {'General': {
        'x': x, 'close': close, 'er': er, 'kama': None,
        'x_er': x, 'er_hist': er, 'er_sma': np.linspace(0.2, 0.7, 50),
        'periodo': 14, 'umbrales': {'er_medio': 0.5, 'er_std': 0.2,
                                    'umbral_tendencia': 0.7, 'umbral_ruido': 0.3},
        'hurst_ventana': 64, 'hurst_medio': 0.5}}}
    _dib_regimen_er(fig, b, 'General')
    fig.canvas.draw()

    ax_p, ax_e = fig.axes[0], fig.axes[1]
    # sync X: zoom del panel superior → el inferior le sigue (y viceversa)
    ax_p.set_xlim(date2num(x[10]), date2num(x[20]))
    assert ax_e.get_xlim() == pytest.approx(ax_p.get_xlim())
    ax_e.set_xlim(date2num(x[30]), date2num(x[40]))
    assert ax_p.get_xlim() == pytest.approx(ax_e.get_xlim())

    _hover(canvas, ax_p, date2num(x[15]), 0.5)
    v = _verticales(ax_p)
    assert v and v[0].get_visible()
    assert v[0].get_linestyle() == '--'
    textos = _textos_visibles(ax_p)
    assert any('Precio:' in t for t in textos)
    assert any('ER:' in t for t in _textos_visibles(ax_e))


# ── Dependencia ────────────────────────────────────────────────────
def test_dependencia_crosshairs():
    fig, canvas = _figura()
    rng = np.random.default_rng(4)
    serie = rng.normal(0, 1, 300)
    acf = np.array([1.0] + list(np.linspace(0.4, -0.2, 15)))
    pacf = np.array([1.0] + list(np.linspace(0.2, -0.1, 15)))
    b = {'dependencia': {
        'escalas': {'labels': ['1m', '1h', '1d'], 'valores': np.array([0.1, -0.2, 0.05])},
        'serie_dia': serie, 'acf': acf, 'pacf': pacf,
        'precio_real': np.linspace(100, 120, 50),
        'random_walks': [np.linspace(100, 90, 50), np.linspace(100, 110, 50)],
        'acf_sq': np.linspace(0.3, 0.05, 20), 'ci': 0.11,
        'lb_p': 0.001, 'clustering': True}}
    _dib_dependencia(fig, b, None)
    fig.canvas.draw()
    ax_acf = _eje_por_titulo(fig, 'ACF diario')
    _hover(canvas, ax_acf, 7.0, 0.2)
    assert any('lag 7 → ' in t for t in _textos_visibles(ax_acf))

    ax_walk = _eje_por_titulo(fig, 'Estructura')
    _hover(canvas, ax_walk, 25.0, 100.0)
    assert any('Precio real:' in t for t in _textos_visibles(ax_walk))

    ax_acf2 = _eje_por_titulo(fig, 'ACF retornos²')
    _hover(canvas, ax_acf2, 5.0, 0.2)
    assert any('CI ±0.11' in t for t in _textos_visibles(ax_acf2))

    # hover de celdas en el mini-heatmap de escalas
    ax_esc = _eje_por_titulo(fig, 'Dependencia por escala')
    _hover(canvas, ax_esc, 0.0, 1.0)
    assert any('× PACF lag 1' in t for t in _textos_visibles(ax_esc))


# ── NATR ───────────────────────────────────────────────────────────
def test_natr_corr_hover_celdas():
    fig, canvas = _figura()
    rng = np.random.default_rng(5)
    corr = np.tril(rng.normal(0, 0.4, (4, 4)), -1) + np.triu(rng.normal(0, 0.4, (4, 4)), 1)
    np.fill_diagonal(corr, 1.0)
    b = {'natr_multitf': {'corr': corr, 'labels': ['1m', '1h', '4h', '1d']}}
    _dib_natr_corr(fig, b, None)
    fig.canvas.draw()
    _hover(canvas, fig.axes[0], 2.0, 1.0)   # 1h × 4h
    assert any('1h × 4h → ' in t for t in _textos_visibles(fig.axes[0]))


def test_dashboard_natr_crosshairs():
    fig, canvas = _figura()
    x = _fechas(40)
    b = {'dashboard_natr': {'General': {
        'window_z_days': 252,
        'term': {'tfs': ['1m', '5m', '15m'], 'mins': [1.0, 5.0, 15.0],
                 'actual': [0.12, 0.25, 0.42], 'teorico': [0.12, 0.27, 0.46],
                 'estructura': 'CONTANGO'},
        'z': {'tfs': ['1m', '5m'], 'vals': [0.5, -2.3]},
        'serie_z': {'par': '1m/5m', 'series': [
            {'tf': '1m', 'x': x, 'y': np.linspace(0, 1.5, 40)},
            {'tf': '5m', 'x': x, 'y': np.linspace(0, -1.0, 40)}]},
        'ratio': [{'x': x, 'y': np.linspace(0.8, 1.2, 40), 'label': '1m/5m',
                   'bb': {'x': x, 'lo': np.linspace(0.7, 1.1, 40),
                          'up': np.linspace(0.9, 1.3, 40),
                          'mu': np.linspace(0.8, 1.2, 40)}}]}}}
    _dib_dashboard_natr(fig, b, 'General')
    fig.canvas.draw()
    ax_a, ax_b, ax_c, ax_d = fig.axes

    _hover(canvas, ax_a, 5.0, 0.3)        # TF 5m
    assert any('5m → NATR 0.250%' in t for t in _textos_visibles(ax_a))

    _hover(canvas, ax_b, 0.5, 0.0)        # barra 1m
    assert any('1m → Z +0.50' in t for t in _textos_visibles(ax_b))

    _hover(canvas, ax_c, date2num(x[20]), 0.5)
    textos_c = _textos_visibles(ax_c)
    assert any('1m:' in t and '5m:' in t for t in textos_c)

    _hover(canvas, ax_d, date2num(x[20]), 1.0)
    assert any('Banda μ:' in t for t in _textos_visibles(ax_d))
