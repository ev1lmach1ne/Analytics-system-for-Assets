import matplotlib
matplotlib.use('Agg')

import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backend_bases import MouseEvent
from matplotlib.figure import Figure

from gui.widgets.analisis_graficos import _dib_cambio_acumulado
from gui.widgets.plot_common import agregar_crosshair


def _artistas_de(eje):
    """Línea vertical (xdata con 2 valores iguales) y horizontal (ydata con
    2 valores iguales) creadas por el crosshair."""
    verticales = [l for l in eje.lines
                  if len(l.get_xdata()) == 2 and l.get_xdata()[0] == l.get_xdata()[1]]
    horizontales = [l for l in eje.lines
                    if len(l.get_ydata()) == 2 and l.get_ydata()[0] == l.get_ydata()[1]
                    and len(l.get_xdata()) == 2 and l.get_xdata()[0] != l.get_xdata()[1]]
    return verticales, horizontales


def _hover(canvas, ax, x_pix, y_pix, xdata):
    ev = MouseEvent('motion_notify_event', canvas, x_pix, y_pix)
    ev.xdata = xdata
    ev.inaxes = ax
    canvas.callbacks.process('motion_notify_event', ev)


# ── Crosshair con línea horizontal + estilo discontinuo ────────────
def test_crosshair_muestra_vertical_horizontal_y_etiqueta():
    x = np.arange(24, dtype=float)
    y = np.linspace(0.0, 5.0, 24)
    labels = [f'{h:02d}:00' for h in range(24)]

    fig = Figure(figsize=(6, 4))
    canvas = FigureCanvasAgg(fig)
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    ax1.plot(x, y)
    ax2.plot(x, y)
    canvas.draw()

    def texto(idx):
        return [f'{labels[idx]} → {y[idx]:+.3f}%',
                f'{labels[idx]} → {y[idx]:+.3f}%']

    agregar_crosshair(fig, [ax1, ax2], x, [y, y], texto,
                      nombre='test_hover', colores=['#58a6ff', '#BA7517'],
                      linestyle='--', horizontal=True,
                      x_texto_fn=lambda i: labels[i])
    canvas.draw()   # draw_event → fondo cacheado

    _hover(canvas, ax1, 300, 200, xdata=6.2)   # snap al paso 6 → 06:00

    v1, h1 = _artistas_de(ax1)
    v2, h2 = _artistas_de(ax2)
    assert len(v1) == 1 and len(v2) == 1
    assert len(h1) == 1 and len(h2) == 1
    assert v1[0].get_linestyle() == '--'        # vertical discontinua
    assert v1[0].get_visible()
    assert v1[0].get_xdata()[0] == pytest.approx(6.0)   # fijada en 6:00
    assert h1[0].get_ydata()[0] == pytest.approx(y[6])  # horizontal al valor del punto
    assert h2[0].get_ydata()[0] == pytest.approx(y[6])

    textos = [t.get_text() for t in ax1.texts]
    assert f'06:00 → {y[6]:+.3f}%' in textos          # tooltip con 3 decimales
    assert '06:00' in textos                          # etiqueta inferior del eje X


def test_crosshair_default_no_anade_horizontal():
    x = np.arange(10, dtype=float)
    y = np.linspace(0.0, 1.0, 10)
    fig = Figure(figsize=(6, 4))
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    canvas.draw()

    agregar_crosshair(fig, [ax], x, [y], lambda i: ['x'],
                      nombre='test_default')
    canvas.draw()
    _hover(canvas, ax, 300, 200, xdata=4.2)

    _, horizontales = _artistas_de(ax)
    assert horizontales == []          # sin la línea horizontal pedida
    v, _ = _artistas_de(ax)
    assert v[0].get_linestyle() in ('solid', '-')   # estilo por defecto intacto


# ── Día: el eje X incluye las etiquetas 00:00 y 24:00 ──────────────
def test_dia_incluye_etiquetas_00h_y_24h():
    b = {'cambio_acumulado': {'dia': {
        'y': np.linspace(0.0, 4.6, 25),
        'pasos': np.arange(25, dtype=int),
        'labels': [f'{h:02d}:00' for h in range(24)] + ['24:00'],
        'paso_medio': np.zeros(25),
        'n': 180, 'total': 4.6}}}
    fig = Figure(figsize=(6, 4))
    _dib_cambio_acumulado(fig, b, None, 'Día')
    etiquetas = [t.get_text() for t in fig.axes[0].get_xticklabels()]
    assert '24:00' in etiquetas     # el salto nocturno 23:00→00:00 visible
    assert '22:00' in etiquetas     # ticks pares de 2h con 25 pasos
    assert etiquetas[0] == '00:00'
    assert etiquetas[-1] == '24:00'
