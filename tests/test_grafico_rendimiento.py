"""Lo que mantiene fluido el pan/zoom del gráfico de Resultados.

Son tres invariantes de rendimiento, no de aspecto, y por eso son fáciles de
romper sin que se note hasta que el gráfico va a tirones con un histórico
grande. Medido sobre 120.000 velas en un canvas de 900 px, un frame de arrastre
pasó de ~106 ms a ~30 ms al imponerlos:

1. no se dibujan más velas que píxeles hay para enseñarlas,
2. los artistas de vela se mutan, no se recrean en cada frame,
3. varios eventos de ratón seguidos colapsan en un solo frame.
"""
import os

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

pytest.importorskip('PyQt6.QtWidgets')
from PyQt6.QtWidgets import QApplication          # noqa: E402

from gui.widgets.tab_backtest import ResultadosWidget   # noqa: E402

COLS_TRADES = ('pnl', 'idx_entrada', 'idx_salida', 'dir', 'precio_entrada',
               'precio_salida', 'motivo', 'unidades', 'r_multiple', 'mae', 'mfe')


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def widget(app):
    """Serie larga sin operaciones: aquí lo que se mide es el coste de pintar
    la serie, no el de los marcadores."""
    n = 20_000
    close = 100 + np.cumsum(np.random.default_rng(1).normal(0, .05, n))
    ts = pd.date_range('2020-01-01', periods=n, freq='5min', tz='UTC')
    payload = {
        'timestamps': ts.values, 'open': close, 'high': close + .1,
        'low': close - .1, 'close': close,
        'resultado': {'trades': {c: np.array([]) for c in COLS_TRADES},
                      'entradas': None, 'ordenes_limite': None},
        'setups': [{'nombre': 'x', 'plantilla': 'Cruce de medias',
                    'params': {'rapida': 20, 'lenta': 50}}],
        'corte': int(n * .7), 'config': {},
    }
    w = ResultadosWidget()
    w._payload = payload
    w._payload_base = payload
    w.resize(1200, 700)
    w._dibujar_principal()
    return w


def _n_velas(w):
    return len(w._art_datos[1].get_paths())


def test_no_se_dibujan_mas_velas_que_pixeles(widget):
    """Por debajo de ~3 px por vela el cuerpo no llega a un píxel: solo cuesta
    rasterizado. El tope sale del ancho REAL del lienzo, no de una constante."""
    ancho = widget.canvas.get_width_height()[0]
    tope = widget._max_velas_visibles()
    assert tope <= ancho / 3 + 1
    assert tope <= 2500

    ax = widget._ax_principal
    x = widget._x_full
    for lo, hi in ((x[0], x[-1]), (x[0], x[len(x) // 2]), (x[0], x[500])):
        ax.set_xlim(lo, hi)
        assert _n_velas(widget) <= tope + 1


def test_el_zoom_fino_no_agrega_velas(widget):
    """El tope recorta, no falsea: con menos velas visibles que el tope se
    dibujan todas, una por una."""
    ax = widget._ax_principal
    x = widget._x_full
    ax.set_xlim(x[0], x[100])
    assert _n_velas(widget) <= 102


def test_los_artistas_de_vela_se_reutilizan(widget):
    """Crear y destruir un PolyCollection por frame costaba ~19 ms de los ~50
    que tardaba un frame. Los mismos objetos deben sobrevivir al arrastre."""
    ax = widget._ax_principal
    x = widget._x_full
    mechas, cuerpos = widget._art_datos
    for i in range(5):
        ax.set_xlim(x[i * 100], x[5000 + i * 100])
        assert widget._art_datos[0] is mechas
        assert widget._art_datos[1] is cuerpos
    assert len(ax.collections) < 20, "artistas huérfanos acumulándose"


def test_varios_eventos_seguidos_pintan_un_solo_frame(widget, app):
    """El ratón manda eventos más deprisa de lo que se rasteriza un frame:
    pintarlos todos deja el gráfico por detrás del cursor."""
    ax = widget._ax_principal
    x = widget._x_full
    ax.set_xlim(x[0], x[5000])
    widget._iniciar_sesion_blit()

    pintados = []
    original = widget._pintar_frame_blit
    widget._pintar_frame_blit = lambda a: (pintados.append(a), original(a))[1]
    try:
        for i in range(5):
            ax.set_xlim(x[i * 10], x[5000 + i * 10])
            widget._solicitar_frame_blit(ax)
        assert pintados == [], "no debe pintarse dentro del propio evento"
        app.processEvents()
        assert len(pintados) == 1
        assert ax.get_xlim()[0] == pytest.approx(x[40]), "pinta la última posición"
    finally:
        widget._pintar_frame_blit = original
        widget._finalizar_blit()


def test_soltar_el_raton_cancela_el_frame_pendiente(widget, app):
    """Un frame pendiente que se ejecutara después de _finalizar_blit pintaría
    por blitting contra un fondo ya descartado."""
    ax = widget._ax_principal
    widget._iniciar_sesion_blit()
    widget._solicitar_frame_blit(ax)
    widget._finalizar_blit()
    assert not widget._timer_frame.isActive()
    assert widget._ax_frame_pendiente is None
    app.processEvents()


def test_el_eje_no_se_llena_de_etiquetas(widget):
    """Cada etiqueta se rasteriza otra vez en cada frame (~1,4 ms): 9 fechas
    largas costaban más que las propias velas."""
    ax = widget._ax_principal
    assert len(ax.get_xticklabels()) <= 7
    assert len(ax.get_yticks()) <= 9
