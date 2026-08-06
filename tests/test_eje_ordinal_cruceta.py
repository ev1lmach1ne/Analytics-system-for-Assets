"""Eje sin huecos de mercado y cruceta del gráfico de operaciones (vista clásica).

Dos cosas que solo se ven mirando el gráfico y por eso es fácil romperlas sin
enterarse:

1. El eje X va en ÍNDICE de vela, no en fecha. Con `date2num`, sábado y domingo
   ocupaban su hueco y la serie diaria salía a peine — y encima el hueco
   aparecía y desaparecía según el zoom, porque al alejar `_decimar_ohlc` agrega
   velas en bloques y el ancho (mediana de `np.diff`) crecía hasta taparlo.
2. La cruceta se engancha a una vela real y sus dos etiquetas dicen la fecha de
   ESA vela y el nivel bajo el cursor.
"""
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

pytest.importorskip('PyQt6.QtWidgets')
from PyQt6.QtWidgets import QApplication          # noqa: E402

from gui.widgets.plot_common import eje_fechas_ordinal   # noqa: E402
from gui.widgets.tab_backtest import ResultadosWidget    # noqa: E402

COLS_TRADES = ('pnl', 'idx_entrada', 'idx_salida', 'dir', 'precio_entrada',
               'precio_salida', 'motivo', 'unidades', 'r_multiple', 'mae', 'mfe')


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


def _payload(ts):
    n = len(ts)
    close = 100 + np.cumsum(np.random.default_rng(3).normal(0, .5, n))
    return {
        'timestamps': ts.values, 'open': close, 'high': close + .3,
        'low': close - .3, 'close': close,
        'resultado': {'trades': {c: np.array([]) for c in COLS_TRADES},
                      'entradas': None, 'ordenes_limite': None},
        'setups': [{'nombre': 'x', 'plantilla': 'Cruce de medias',
                    'params': {'rapida': 5, 'lenta': 20}}],
        'corte': int(n * .7), 'config': {},
    }


def _widget(app, ts):
    w = ResultadosWidget()
    p = _payload(ts)
    w._payload = p
    w._payload_base = p
    w.resize(1200, 700)
    w._dibujar_principal()
    return w


# fechas de cierre DIARIO: bdate_range no trae sábados ni domingos, que es
# exactamente el caso que reportó el usuario
TS_DIARIO = pd.bdate_range('2021-01-04', '2024-12-31')
TS_INTRADIA = pd.date_range('2024-01-01', periods=3000, freq='1h')


# ─────────────────────── eje ordinal (plot_common) ───────────────────────

def _eje(ts):
    from matplotlib.figure import Figure
    ax = Figure().add_subplot()
    loc, fmt = eje_fechas_ordinal(ax, ts)
    return ax, loc, fmt


def test_las_marcas_son_indices_de_vela_dentro_del_rango():
    _ax, loc, _fmt = _eje(TS_DIARIO)
    marcas = loc.tick_values(0, len(TS_DIARIO) - 1)
    assert marcas, "el eje se queda sin etiquetas"
    assert all(0 <= m <= len(TS_DIARIO) - 1 for m in marcas)
    assert all(float(m).is_integer() for m in marcas), \
        "una marca a medio camino entre dos velas no corresponde a ninguna fecha"


def test_las_marcas_caen_en_fronteras_de_calendario():
    """Lo que distingue esto de un MaxNLocator sobre el índice: repartir las
    marcas cada N velas daría fechas arbitrarias que además bailan al hacer
    scroll. Aquí cada marca es la primera vela de su periodo."""
    _ax, loc, fmt = _eje(TS_DIARIO)
    marcas = loc.tick_values(0, len(TS_DIARIO) - 1)
    assert loc.unidad == 'anio'
    assert [fmt(m) for m in marcas] == ['2021', '2022', '2023', '2024']
    # y la vela de cada marca es la primera del año que TIENE dato, no una de
    # mitad de enero (el día exacto depende de en qué caiga el primer hábil)
    for m in marcas:
        t = TS_DIARIO[int(m)]
        assert t.month == 1 and t.day <= 5


def test_el_escalon_se_afina_al_acercar_la_vista():
    _ax, loc, fmt = _eje(TS_DIARIO)
    unidades = []
    for lo, hi in [(0, len(TS_DIARIO) - 1), (0, 300), (0, 40), (0, 8)]:
        loc.tick_values(lo, hi)
        unidades.append(loc.unidad)
    orden = ['minuto', 'hora', 'dia', 'semana', 'mes', 'anio']
    posiciones = [orden.index(u) for u in unidades]
    assert posiciones == sorted(posiciones, reverse=True), \
        f"el escalón debería ir afinándose al acercar, salió {unidades}"


def test_ningun_escalon_por_debajo_del_espaciado_de_las_velas():
    """Con 3 velas diarias en pantalla, marcas cada 12 h señalarían instantes
    en los que no hay ninguna vela."""
    _ax, loc, _fmt = _eje(TS_DIARIO)
    loc.tick_values(100, 103)
    assert loc.unidad not in ('minuto', 'hora')


def test_la_etiqueta_lleva_hora_solo_en_intradia():
    _ax, loc_d, fmt_d = _eje(TS_DIARIO)
    loc_d.tick_values(100, 103)
    assert ':' not in fmt_d(101)

    _ax2, loc_h, fmt_h = _eje(TS_INTRADIA)
    loc_h.tick_values(0, 8)
    assert any(':' in fmt_h(i) for i in range(1, 7))


def test_fuera_de_rango_no_se_inventa_fecha():
    """El eje deja arrastrar más allá del primer y último dato; ahí no hay
    fecha que enseñar."""
    _ax, _loc, fmt = _eje(TS_DIARIO)
    assert fmt(-5) == ''
    assert fmt(len(TS_DIARIO) + 5) == ''
    assert fmt(0) != ''


# ─────────────── el gráfico no deja huecos de fin de semana ───────────────

def test_el_eje_del_grafico_va_en_indice_de_vela(app):
    w = _widget(app, TS_DIARIO)
    assert np.array_equal(w._x_full, np.arange(len(TS_DIARIO), dtype=float))
    # el viernes y el lunes siguiente quedan a una vela de distancia, no a tres
    viernes = int(np.where(w._ts_full.dayofweek == 4)[0][0])
    assert (w._ts_full[viernes + 1] - w._ts_full[viernes]).days == 3
    assert w._x_full[viernes + 1] - w._x_full[viernes] == 1.0


@pytest.mark.parametrize('ancho_ventana', [len(TS_DIARIO) - 1, 400, 120, 30])
def test_las_velas_tienen_el_mismo_ancho_a_cualquier_zoom(app, ancho_ventana):
    """El síntoma que reportó el usuario: al alejar, la agregación por bloques
    ensanchaba las velas y tapaba el hueco; al acercar, el hueco reaparecía."""
    w = _widget(app, TS_DIARIO)
    ax = w._ax_principal
    ax.set_xlim(0, ancho_ventana)
    w._redibujar_datos(ax)
    cuerpos = w._art_datos[1].get_paths()
    anchos = [p.get_extents().width for p in cuerpos]
    assert len(set(round(a, 9) for a in anchos)) == 1, \
        "las velas no salen todas del mismo ancho"


# ────────────────────────────── cruceta ──────────────────────────────

def _mover(w, ax, x_dato, y_dato):
    """Evento de movimiento de ratón sintético sobre `ax`, con las mismas
    claves que consulta _on_motion_ejes."""
    px, py = ax.transData.transform((x_dato, y_dato))
    return SimpleNamespace(inaxes=ax, xdata=x_dato, ydata=y_dato,
                           x=px, y=py, name='motion_notify_event')


def test_la_cruceta_se_engancha_a_una_vela_y_dice_su_fecha(app):
    w = _widget(app, TS_DIARIO)
    ax = w._ax_principal
    # a medio camino entre la vela 40 y la 41: debe engancharse a la 41
    w._actualizar_cruceta(_mover(w, ax, 40.6, 100.0), 'pan')
    assert all(v.get_visible() for v in w._cruceta_v)
    assert w._cruceta_v[0].get_xdata()[0] == 41
    assert w._cruceta_lbl_x.get_text() == TS_DIARIO[41].strftime('%Y-%m-%d')


def test_la_horizontal_es_libre_y_etiqueta_el_nivel(app):
    """La vertical se engancha a la vela (una fecha real), pero la horizontal
    va donde esté el ratón: es lo que permite medir un nivel a ojo."""
    w = _widget(app, TS_DIARIO)
    ax = w._ax_principal
    w._actualizar_cruceta(_mover(w, ax, 40.6, 123.456), 'pan')
    assert w._cruceta_h[0].get_ydata()[0] == pytest.approx(123.456)
    assert '123.4' in w._cruceta_lbl_y[0].get_text().replace(',', '')


def test_fuera_del_area_de_datos_la_cruceta_desaparece(app):
    w = _widget(app, TS_DIARIO)
    ax = w._ax_principal
    w._actualizar_cruceta(_mover(w, ax, 40.0, 100.0), 'pan')
    assert w._cruceta_v[0].get_visible()
    # 'x' = la franja de fechas, no el interior del gráfico
    w._actualizar_cruceta(_mover(w, ax, 40.0, 100.0), 'x')
    assert not any(a.get_visible() for a in w._art_cruceta)


def test_la_cruceta_no_se_hornea_en_el_fondo_cacheado(app):
    """animated=True es lo que impide que canvas.draw() la pinte: si se colara
    en el bitmap de fondo quedaría un rastro fantasma que solo se borra con un
    redibujado completo."""
    w = _widget(app, TS_DIARIO)
    assert w._art_cruceta, "sin artistas no hay cruceta que pintar"
    assert all(a.get_animated() for a in w._art_cruceta)
    assert w._annot_trade.get_animated()


def test_la_cruceta_cruza_todos_los_paneles(app):
    """Con un oscilador abierto, la vertical marca la MISMA vela arriba y
    abajo; si no, el panel no sirve para leer el indicador de esa vela."""
    w = ResultadosWidget()
    p = _payload(TS_DIARIO)
    # una plantilla de RSI abre su propio panel bajo el precio
    p['setups'] = [{'nombre': 'r', 'plantilla': 'RSI', 'params': {'periodo': 14}}]
    w._payload = w._payload_base = p
    w.resize(1200, 800)
    w._dibujar_principal()
    assert len(w._paneles) >= 2, "el fixture debe abrir el panel de RSI"
    w._actualizar_cruceta(_mover(w, w._ax_principal, 40.0, 100.0), 'pan')
    assert len(w._cruceta_v) == len(w._paneles)
    assert {v.get_xdata()[0] for v in w._cruceta_v} == {40}


# ──────────── la UI de rango sigue hablando en fechas ────────────

def test_los_dateedits_traducen_indice_a_fecha(app):
    w = _widget(app, TS_DIARIO)
    w._sync_dateedits(10, 200)
    assert w.fecha_ini.date().toPyDate() == TS_DIARIO[10].date()
    assert w.fecha_fin.date().toPyDate() == TS_DIARIO[200].date()


def test_aplicar_un_rango_de_fechas_encuadra_esas_velas(app):
    w = _widget(app, TS_DIARIO)
    from PyQt6.QtCore import QDate
    d0, d1 = TS_DIARIO[50], TS_DIARIO[300]
    w.fecha_ini.setDate(QDate(d0.year, d0.month, d0.day))
    w.fecha_fin.setDate(QDate(d1.year, d1.month, d1.day))
    w._aplicar_rango()
    assert w._ax_principal.get_xlim() == pytest.approx((50.0, 300.0))


def test_centrar_un_trade_lo_deja_dentro_del_encuadre(app):
    w = _widget(app, TS_DIARIO)
    w._centrar_en(TS_DIARIO[100], TS_DIARIO[140])
    lo, hi = w._ax_principal.get_xlim()
    assert lo < 100 and hi > 140, "la operación debe quedar con margen a ambos lados"
