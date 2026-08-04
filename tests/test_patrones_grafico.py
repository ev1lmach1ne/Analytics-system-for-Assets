"""Mini-histograma de las tarjetas de "Patrones de velas".

Lo que se fija aquí es que las 32 tarjetas comparten UN marco: mismo eje X (el
histórico completo del activo) y misma escala Y por vista. Antes cada tarjeta
autoescalaba a sus propios datos, así que un patrón raro con un único bloque
superviviente lo pintaba ocupando el gráfico entero mientras el de al lado
mostraba barras de menos de un píxel — y la altura de una barra significaba
algo distinto en cada tarjeta.
"""
import os

import numpy as np
import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

pytest.importorskip('PyQt6.QtWidgets')
from PyQt6.QtWidgets import QApplication            # noqa: E402
from matplotlib.dates import date2num               # noqa: E402

from core.candle_patterns import LAGS               # noqa: E402
from gui.widgets.tab_patrones import (              # noqa: E402
    PatternCard, escala_y_compartida, ancho_barras, _alpha_por_n,
    PISO_Y_HITRATE, PISO_Y_EDGE,
)


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


def _barras(fechas_ini, dur_dias, hit_rate=None, edge_pb=None, n=None):
    """Dict con la forma que devuelve core.candle_patterns.agregar_por_periodo."""
    ini = np.array(fechas_ini, dtype='datetime64[ns]')
    fin = ini + np.timedelta64(dur_dias, 'D')
    k = len(ini)
    return {
        'fechas': fin,
        'fecha_ini': ini,
        'fecha_fin': fin,
        'n': np.full(k, 20, dtype=np.int64) if n is None else np.asarray(n),
        'hit_rate': np.full(k, 0.6) if hit_rate is None else np.asarray(hit_rate),
        'edge_pb': np.full(k, 12.0) if edge_pb is None else np.asarray(edge_pb),
    }


def _stats(n_total=500):
    """Lo mínimo de calcular_stats_patron que consume update_stats."""
    por_lag = {lag: {'hit_rate': 0.6, 'edge': 0.0012, 'p_vs_50': 0.01,
                     'p_vs_base': 0.02, 'significativo': True} for lag in LAGS}
    return {'n_total': n_total, 'por_lag': por_lag}


# ══════════════ escala Y compartida ══════════════
def test_escala_y_ignora_un_outlier_aislado():
    """Un bloque extremo (típicamente de pocas ocurrencias) no puede aplastar
    contra el cero a todos los demás: el tope sale de un percentil, no del máximo."""
    normales = np.full(200, 8.0)
    con_outlier = np.concatenate([normales, [900.0]])

    tope = escala_y_compartida([con_outlier], 'retorno')

    assert tope < 100
    assert tope >= 8.0


def test_escala_y_cubre_todos_los_patrones_visibles():
    """El tope es común: se calcula sobre los valores de todas las tarjetas."""
    flojo = np.full(50, 3.0)
    fuerte = np.full(50, 40.0)

    tope = escala_y_compartida([flojo, fuerte], 'retorno')

    assert tope == escala_y_compartida([fuerte, flojo], 'retorno')
    assert tope > 10


def test_escala_y_usa_el_piso_de_cada_unidad():
    """pp (hit rate) y pb (retorno) son unidades incomparables: compartir un
    único piso aplastaba los edges de las TFs finas, que son de pocos pb."""
    planos = np.zeros(20)

    assert escala_y_compartida([planos], 'hitrate') == PISO_Y_HITRATE
    assert escala_y_compartida([planos], 'retorno') == PISO_Y_EDGE
    # con edges pequeños pero reales, el piso de pb no los amplifica a pantalla
    # completa pero tampoco los deja en un eje degenerado
    assert escala_y_compartida([np.full(20, 0.4)], 'retorno') == PISO_Y_EDGE


def test_escala_y_sin_datos_ni_no_finitos():
    assert escala_y_compartida([], 'hitrate') == PISO_Y_HITRATE
    assert escala_y_compartida([np.array([])], 'retorno') == PISO_Y_EDGE
    assert escala_y_compartida([np.array([np.nan, np.inf])], 'retorno') == PISO_Y_EDGE


# ══════════════ ancho de barra ══════════════
def test_ancho_barras_usa_la_duracion_real_del_bloque():
    """Con bloques densos el ancho es el del bin, así que teselan el eje."""
    ini = np.arange(0.0, 100.0)
    fin = ini + 1.0

    w = ancho_barras(ini, fin, (0.0, 100.0), ancho_px=400)

    assert np.allclose(w, 1.0)


def test_ancho_barras_respeta_un_minimo_en_pixeles():
    """Un bloque diario en un eje de 10 años mide ~0.03% del ancho: sin suelo
    no llega ni a un píxel y la barra es invisible."""
    ini = np.array([1000.0])
    fin = np.array([1001.0])
    span = 3650.0

    w = ancho_barras(ini, fin, (0.0, span), ancho_px=400, min_px=1.5)

    assert w[0] > 1.0                      # el suelo se ha aplicado
    assert w[0] == pytest.approx(span / 400 * 1.5)


def test_ancho_barras_no_infla_un_bloque_suelto_hasta_llenar_el_eje():
    """La regresión directa de la queja: un único bloque anual NO puede ocupar
    una fracción apreciable de un eje de 10 años."""
    ini = np.array([0.0])
    fin = np.array([365.0])
    span = 3650.0

    w = ancho_barras(ini, fin, (0.0, span), ancho_px=400)

    assert w[0] == 365.0                   # su duración real, ni más ni menos
    assert w[0] / span < 0.15


def test_ancho_barras_eje_degenerado():
    ini, fin = np.array([0.0]), np.array([1.0])
    assert np.all(ancho_barras(ini, fin, (5.0, 5.0), ancho_px=400) > 0)
    assert np.all(ancho_barras(ini, fin, (0.0, 10.0), ancho_px=0) > 0)


# ══════════════ opacidad por ocurrencias ══════════════
def test_alpha_crece_con_las_ocurrencias_del_bloque():
    """Un bloque de 5 ocurrencias es ruido y no debe verse tan sólido como uno
    de 500 — de ahí salen la mayoría de las barras extremas."""
    a = _alpha_por_n(np.array([5, 50, 500]), n_max=500)

    assert a[0] < a[1] < a[2]
    assert a.min() >= 0.3 and a.max() <= 1.0


def test_alpha_es_opaco_si_no_hay_rango():
    a = _alpha_por_n(np.array([5, 5]), n_max=5)
    assert np.allclose(a, a[0])
    assert a[0] > 0.9


# ══════════════ el marco compartido llega al eje ══════════════
def test_tarjetas_distintas_comparten_eje_x_y_escala_y(app):
    """Dos tarjetas dibujadas en el mismo refresco: una con un único bloque y
    otra con 200. Antes la primera pintaba su barra ocupando todo el gráfico."""
    xlim = (date2num(np.datetime64('2015-01-01')),
            date2num(np.datetime64('2025-01-01')))
    ylim_abs = 40.0

    raro = PatternCard('Doji')
    raro.update_stats(_stats(7), None, 'hitrate', LAGS[0],
                      _barras(['2017-06-01'], 30), xlim, ylim_abs, 500)

    frecuente = PatternCard('Martillo')
    frecuente.update_stats(
        _stats(9000), None, 'hitrate', LAGS[0],
        _barras([f'2015-{m:02d}-01' for m in range(1, 13)], 30), xlim,
        ylim_abs, 500)

    ax_raro = raro.fig.axes[0]
    ax_frec = frecuente.fig.axes[0]

    assert ax_raro.get_xlim() == pytest.approx(xlim)
    assert ax_raro.get_xlim() == pytest.approx(ax_frec.get_xlim())
    assert ax_raro.get_ylim() == pytest.approx(ax_frec.get_ylim())
    assert ax_raro.get_ylim() == pytest.approx((-ylim_abs * 1.15, ylim_abs * 1.15))

    # y la barra del patrón raro sigue siendo una barra, no un bloque
    (barra,) = ax_raro.patches
    assert barra.get_width() / (xlim[1] - xlim[0]) < 0.05


def test_barra_ocupa_su_bloque_de_calendario(app):
    """align='edge' desde fecha_ini: la barra empieza donde empieza el bloque.
    Con la etiqueta del bin ('fechas', su extremo derecho en las reglas de fin
    de periodo) centrada, quedaba a caballo entre dos bloques."""
    xlim = (date2num(np.datetime64('2024-01-01')),
            date2num(np.datetime64('2024-12-31')))
    card = PatternCard('Doji')

    card.update_stats(_stats(), None, 'hitrate', LAGS[0],
                      _barras(['2024-03-01'], 31), xlim, 40.0, 100)

    (barra,) = card.fig.axes[0].patches
    assert barra.get_x() == pytest.approx(date2num(np.datetime64('2024-03-01')))
    assert barra.get_width() == pytest.approx(31.0)


def test_bloques_fuera_de_escala_se_recortan_y_se_marcan(app):
    """Un bloque que supera el tope compartido se recorta al borde y se señala,
    en vez de reventar el eje o mentir sobre su altura."""
    xlim = (date2num(np.datetime64('2024-01-01')),
            date2num(np.datetime64('2024-12-31')))
    card = PatternCard('Doji')
    tope = 20.0

    card.update_stats(
        _stats(), None, 'retorno', LAGS[0],
        _barras(['2024-01-01', '2024-02-01', '2024-03-01'], 28,
                edge_pb=[5.0, 900.0, -900.0]),
        xlim, tope, 100)

    ax = card.fig.axes[0]
    alturas = sorted(abs(b.get_height()) for b in ax.patches)
    assert alturas == pytest.approx([5.0, tope, tope])
    assert ax.get_ylim() == pytest.approx((-tope * 1.15, tope * 1.15))
    # una marca por cada bloque recortado (uno arriba, otro abajo)
    assert sum(len(c.get_offsets()) for c in ax.collections) == 2


def test_bloque_plano_se_marca_para_no_parecer_vacio(app):
    """Un bloque con hit rate exactamente 50% dibuja una barra de altura cero:
    invisible, e indistinguible de "aquí no hubo bloque" — lectura equivocada
    justo en el patrón raro de un solo bloque, donde la tarjeta parecía vacía."""
    xlim = (date2num(np.datetime64('2024-01-01')),
            date2num(np.datetime64('2024-12-31')))
    card = PatternCard('Doji')

    card.update_stats(_stats(), None, 'hitrate', LAGS[0],
                      _barras(['2024-06-01'], 30, hit_rate=[0.5]),
                      xlim, 40.0, 100)

    alturas = [b.get_height() for b in card.fig.axes[0].patches]
    assert 0.0 in alturas                      # la barra real, plana
    assert max(alturas) > 0                    # y su marca visible
    # la marca se queda pegada al eje: no simula una desviación que no existe
    assert max(alturas) < 40.0 * 0.1


def test_sin_bloques_no_revienta(app):
    """Patrón sin ningún bloque que supere el mínimo de ocurrencias."""
    card = PatternCard('Doji')

    card.update_stats(_stats(3), None, 'hitrate', LAGS[0],
                      _barras([], 30), (0.0, 100.0), 40.0, 100)

    assert len(card.fig.axes[0].patches) == 0
