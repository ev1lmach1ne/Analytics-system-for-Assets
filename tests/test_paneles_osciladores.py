"""Paneles de indicadores del gráfico de Resultados: qué se lee en su eje Y y
cómo se manejan con el ratón.

Los niveles (70/30 del RSI, umbrales del ER) ya se dibujaban como líneas de
puntos, pero el eje los repartía por su cuenta y salían 0-20-40-60-80-100: justo
los números que definen el indicador eran los únicos que no se leían.

El arrastre se prueba por sus EFECTOS sobre los límites de los ejes y no
simulando píxeles de ratón a mano, porque lo que puede romperse es el reparto de
ejes: que la X del panel mueva el precio (comparten eje) pero su Y no lo toque.
"""
import os

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

pytest.importorskip('PyQt6.QtWidgets')
from PyQt6.QtWidgets import QApplication            # noqa: E402

from core.strategies import (                       # noqa: E402
    UMBRAL_ER_TENDENCIA, UMBRAL_ER_RUIDO, VENTANA_ER_DEFECTO,
)
from gui.widgets.tab_backtest import (                 # noqa: E402
    ResultadosWidget, _marcar_niveles_eje,
)

COLS_TRADES = ('pnl', 'idx_entrada', 'idx_salida', 'dir', 'precio_entrada',
               'precio_salida', 'motivo', 'unidades', 'r_multiple', 'mae', 'mfe')


class _Evento:
    """Lo mínimo que _zona_eje / _on_press_ejes / _on_motion_ejes miran de un
    evento de matplotlib."""

    def __init__(self, x, y, dblclick=False):
        self.x, self.y, self.dblclick = x, y, dblclick
        self.button = 1
        self.xdata = self.ydata = None


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def widget(app):
    """Un setup de RSI y un filtro de régimen ER: así se crean los dos paneles
    que el usuario nombró, con sus dos formas de fijar niveles (constantes del
    indicador y umbrales del filtro)."""
    n = 3000
    close = 100 + np.cumsum(np.random.default_rng(3).normal(0, .3, n))
    ts = pd.date_range('2022-01-01', periods=n, freq='1h', tz='UTC')
    payload = {
        'timestamps': ts.values, 'open': close, 'high': close + .5,
        'low': close - .5, 'close': close,
        'resultado': {'trades': {c: np.array([]) for c in COLS_TRADES},
                      'entradas': None, 'ordenes_limite': None},
        'setups': [{'nombre': 'r', 'plantilla': 'RSI',
                    'params': {'periodo': 14},
                    'filtros': {'regimen': {'metodo': 'er_tendencia',
                                            'periodo': VENTANA_ER_DEFECTO}}}],
        'corte': int(n * .7), 'config': {},
    }
    w = ResultadosWidget()
    w._payload = payload
    w._payload_base = payload
    w.resize(1200, 800)
    w._dibujar_principal()
    return w


def _panel(w, kind):
    for i, (k, ax) in enumerate(w._paneles):
        if k == kind:
            return i, ax
    raise AssertionError(f"no se ha creado el panel {kind}: {[k for k, _ in w._paneles]}")


def _centro(ax):
    b = ax.get_window_extent()
    return (b.x0 + b.x1) / 2.0, (b.y0 + b.y1) / 2.0


# ── niveles escritos en el eje ──

def test_el_eje_del_rsi_escribe_sobrecompra_y_sobreventa(widget):
    _i, ax = _panel(widget, 'rsi')
    assert list(ax.get_yticks()) == [30, 50, 70]


def test_el_eje_del_regimen_escribe_sus_umbrales(widget):
    _i, ax = _panel(widget, 'er')
    assert list(ax.get_yticks()) == sorted([UMBRAL_ER_RUIDO, UMBRAL_ER_TENDENCIA])


def test_las_etiquetas_van_del_color_de_su_linea(widget):
    """Sobrecompra en rojo y sobreventa en verde: el número tiene que poder
    leerse sin cruzarlo con la línea de puntos que le corresponde."""
    _i, ax = _panel(widget, 'rsi')
    colores = [t.get_color() for t in ax.get_yticklabels()]
    assert len(set(colores)) == 3, f"las 3 marcas deberían distinguirse: {colores}"


def test_con_la_escala_estirada_a_mano_el_eje_no_se_queda_en_blanco(widget):
    """Los niveles fijos pueden caer fuera de la vista al estirar; ahí tiene que
    volver el reparto automático o el panel se queda sin ninguna referencia."""
    widget._ylim_paneles['rsi'] = (90.0, 99.0)
    widget._dibujar_principal()
    _i, ax = _panel(widget, 'rsi')
    lo, hi = ax.get_ylim()
    assert (lo, hi) == (90.0, 99.0)
    visibles = [t for t in ax.get_yticks() if lo <= t <= hi]
    assert visibles, "el eje se ha quedado sin una sola marca visible"


def test_sin_niveles_el_eje_se_deja_en_paz(app):
    """Un panel sin niveles de referencia (el ATR no tiene sobrecompra ni
    sobreventa) debe conservar el reparto automático, no una lista fija."""
    from matplotlib.figure import Figure
    ax = Figure().add_subplot(111)
    ticks_antes = list(ax.get_yticks())
    assert _marcar_niveles_eje(ax, []) is False
    assert list(ax.get_yticks()) == ticks_antes


def test_demasiados_niveles_no_llenan_el_panel_de_numeros(app):
    """Varios setups con umbrales distintos pueden aportar muchos valores, y en
    un panel de ~90 px no caben: ahí también manda el reparto automático."""
    from matplotlib.figure import Figure
    ax = Figure().add_subplot(111)
    muchos = [(v, '#ffffff') for v in range(0, 100, 10)]
    assert _marcar_niveles_eje(ax, muchos) is False


def test_los_niveles_repetidos_se_escriben_una_vez(app):
    """Dos setups con el mismo umbral no deben pintar la etiqueta dos veces."""
    from matplotlib.figure import Figure
    ax = Figure().add_subplot(111)
    assert _marcar_niveles_eje(ax, [(70, '#f00'), (30, '#0f0'), (70, '#f00')])
    assert list(ax.get_yticks()) == [30, 70]


# ── ratón dentro del panel ──

def test_dentro_del_panel_se_arrastra_y_a_su_derecha_se_estira(widget):
    i, ax = _panel(widget, 'rsi')
    x, y = _centro(ax)
    assert widget._zona_eje(_Evento(x, y)) == f'pan:{i}'
    b = ax.get_window_extent()
    assert widget._zona_eje(_Evento(b.x1 + 12, y)) == f'y:{i}'


def test_arrastrar_dentro_del_panel_mueve_el_tiempo_sin_tocar_el_precio(widget):
    """El eje X es compartido (mueve todo el gráfico), el Y es solo del panel:
    ese reparto es justo lo que puede romperse al tocar los transData.

    Que la escala del PRECIO cambie es correcto y no se comprueba aquí: mientras
    no esté fijada a mano se reajusta sola a la ventana visible en cada cambio de
    tiempo, igual que con la rueda. Lo que no puede pasar es que arrastrar en un
    panel de oscilador la deje fijada (_y_manual), porque entonces el precio
    dejaría de reencuadrarse para el resto de la sesión."""
    i, ax_rsi = _panel(widget, 'rsi')
    ax_precio = widget._ax_principal
    x, y = _centro(ax_rsi)

    widget._y_manual = False
    xlim0 = ax_precio.get_xlim()
    ylim_rsi0 = ax_rsi.get_ylim()

    widget._on_press_ejes(_Evento(x, y))
    assert widget._drag_modo == f'pan:{i}', "el press no ha abierto el arrastre"
    widget._on_motion_ejes(_Evento(x - 120, y - 25))
    widget._on_release_ejes(_Evento(x - 120, y - 25))

    assert ax_precio.get_xlim() != xlim0, "el arrastre no ha movido el tiempo"
    assert ax_rsi.get_ylim() != ylim_rsi0, "no ha movido la escala del panel"
    assert widget._y_manual is False, \
        "arrastrar en un panel ha fijado a mano la escala del precio"


def test_la_escala_movida_a_mano_sobrevive_al_redibujado(widget):
    """Se guarda por TIPO de panel, igual que el estirado: si no, apagar y
    encender el panel devolvería la escala a su sitio."""
    i, ax_rsi = _panel(widget, 'rsi')
    x, y = _centro(ax_rsi)
    widget._on_press_ejes(_Evento(x, y))
    widget._on_motion_ejes(_Evento(x, y - 30))
    widget._on_release_ejes(_Evento(x, y - 30))
    movida = widget._ylim_paneles.get('rsi')
    assert movida is not None

    widget._dibujar_principal()
    _i2, ax_rsi2 = _panel(widget, 'rsi')
    assert ax_rsi2.get_ylim() == pytest.approx(movida)
