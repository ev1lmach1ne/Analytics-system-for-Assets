"""Panel de capas del gráfico de Resultados y marcadores de operación.

Lo que se fija aquí es la promesa del diseño: los ojos NO ocultan artistas, hacen
que no se creen. Por eso el test cuenta artistas en `_art_overlays_extra` en vez
de mirar `get_visible()` — ocultarlos no funcionaría, porque
`_iniciar_sesion_blit` los devuelve a visibles en cuanto se arrastra el gráfico.
"""
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

pytest.importorskip('PyQt6.QtWidgets')
from PyQt6.QtWidgets import QApplication   # noqa: E402

from core.backtest import simular, ORDEN_RELLENADA      # noqa: E402
from core.strategies import (                           # noqa: E402
    _entrada_por_defecto, generar_senales_sistema,
)


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope='module')
def payload_con_ordenes():
    """Backtest real (no un mock) de un setup con entrada límite Fibonacci,
    con el mismo juego de claves que emite _BacktestThread."""
    rng = np.random.default_rng(7)
    n = 1200
    close = 100.0 + np.cumsum(rng.normal(0, 1.0, n))
    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=n, freq='1h', tz='UTC'),
        'open': close, 'close': close, 'high': close + 0.6, 'low': close - 0.6,
    })
    setup = {
        'nombre': 'Fib', 'plantilla': 'Cruce de medias',
        'params': {'rapida': 10, 'lenta': 30, 'direccion': 'Ambas'},
        'entrada': dict(_entrada_por_defecto(), tipo='limite_fib', nivel_fib=0.5),
        'limite_vigencia_velas': 30,
    }
    senales = generar_senales_sistema(df, [setup])
    config = {'capital_inicial': 10000.0, 'riesgo_pct': 0.01,
              'comision_pct': 0.0, 'slippage_pct': 0.0,
              'stop_atr': 2.0, 'tp_r': 3.0, 'salida_n_velas': 0,
              'config_por_setup': {0: {'riesgo_pct': 0.01, 'stop_atr': 2.0,
                                       'tp_r': 3.0,
                                       'limite_vigencia_velas': 30}}}
    resultado = simular(df['open'].values, df['high'].values,
                        df['low'].values, df['close'].values, senales, config)
    assert len(resultado['ordenes_limite']['idx_alta']) > 0, \
        "el fixture debe producir órdenes; si no, el test no prueba nada"
    return {
        'timestamps': df['timestamp'].values,
        'open': df['open'].values, 'high': df['high'].values,
        'low': df['low'].values, 'close': df['close'].values,
        'resultado': resultado, 'setups': [setup], 'corte': int(n * 0.7),
        'config': config,
    }


def _widget(app, payload):
    from gui.widgets.tab_backtest import ResultadosWidget
    w = ResultadosWidget()
    w._payload = payload
    w._payload_base = payload
    return w


def _n_artistas(w):
    w._dibujar_principal()
    return len(w._art_overlays_extra)


def _claves(w):
    return [c['clave'] for c in w._capas_catalogo]


def _clave_zigzag(w):
    return next(c for c in _claves(w) if c.startswith('zigzag:'))


def test_las_tres_capas_de_limite_estan_en_el_catalogo(app, payload_con_ordenes):
    w = _widget(app, payload_con_ordenes)
    w._dibujar_principal()
    claves = _claves(w)
    assert 'fib' in claves
    assert 'ordenes' in claves
    assert any(c.startswith('zigzag:') for c in claves)
    # las capas nacen visibles salvo el ZigZag, que se enciende a mano
    assert all(w._capa_activa(c) for c in claves
               if not c.startswith('zigzag:'))


def test_el_zigzag_nace_apagado(app, payload_con_ordenes):
    """Su polilínea recorre todo el histórico por encima del precio: abrumaba
    al abrir el gráfico y solo hace falta al programar entradas por orden
    límite. Sigue en el panel para poder encenderla."""
    w = _widget(app, payload_con_ordenes)
    w._dibujar_principal()          # el catálogo se puebla al dibujar
    clave = _clave_zigzag(w)
    assert not w._capa_activa(clave)
    assert clave in _claves(w), "apagada no es lo mismo que ausente del panel"
    fila = next(f for f in w.panel_capas._filas if f.clave == clave)
    assert not fila.ojo.isChecked(), "el ojo del panel debe nacer tachado"

    sin_zigzag = _n_artistas(w)
    w._capas_estado[clave] = True
    assert _n_artistas(w) > sin_zigzag


def test_la_vista_moderna_tambien_nace_sin_zigzag(app, payload_con_ordenes):
    """El dict de capas que se le pasa a la vista Moderna sale del mismo estado,
    para que conmutar de vista no encienda de golpe lo que la clásica esconde."""
    w = _widget(app, payload_con_ordenes)
    w._dibujar_principal()
    assert not any(w._capa_activa(c) for c in _claves(w)
                   if c.startswith('zigzag:'))


def test_apagar_una_capa_deja_de_crear_sus_artistas(app, payload_con_ordenes):
    w = _widget(app, payload_con_ordenes)
    w._dibujar_principal()
    w._capas_estado[_clave_zigzag(w)] = True   # nace apagado, aquí se prueba apagarlo
    con_todo = _n_artistas(w)
    for clave in ('fib', 'ordenes', _clave_zigzag(w)):
        w._capas_estado[clave] = False
        sin_capa = _n_artistas(w)
        assert sin_capa < con_todo, f"apagar {clave} no quitó ningún artista"
        w._capas_estado[clave] = True
        assert _n_artistas(w) == con_todo   # y reencenderla los devuelve


def test_apagar_las_tres_las_quita_todas(app, payload_con_ordenes):
    w = _widget(app, payload_con_ordenes)
    con_todo = _n_artistas(w)
    w._dibujar_principal()
    for clave in ('fib', 'ordenes', _clave_zigzag(w)):
        w._capas_estado[clave] = False
    assert _n_artistas(w) < con_todo


def test_sin_ordenes_esas_capas_no_se_listan(app, payload_con_ordenes):
    """Un backtest sin entrada límite no tiene nada que enseñar en estas capas:
    no aparecen en el panel, en vez de dejar un ojo que no hace nada."""
    payload = dict(payload_con_ordenes)
    payload['setups'] = [{'nombre': 'Mercado', 'plantilla': 'Cruce de medias',
                          'params': {'rapida': 10, 'lenta': 30}}]
    payload['resultado'] = dict(payload['resultado'])
    payload['resultado']['ordenes_limite'] = {
        k: np.array([], dtype=v.dtype)
        for k, v in payload_con_ordenes['resultado']['ordenes_limite'].items()}
    w = _widget(app, payload)
    w._dibujar_principal()
    claves = _claves(w)
    assert 'fib' not in claves
    assert 'ordenes' not in claves
    assert not any(c.startswith('zigzag:') for c in claves)
    # pero las medias del setup sí, que es lo que este sistema usa
    assert 'ma:SMA:10' in claves and 'ma:SMA:30' in claves


def _rellenadas(payload):
    ol = payload['resultado']['ordenes_limite']
    return np.flatnonzero(ol['resultado'].astype(int) == ORDEN_RELLENADA)


def _payload_sin_rellenar(payload_con_ordenes):
    """Mismo backtest con todas las órdenes marcadas como canceladas. Copia los
    arrays: el fixture es de módulo y lo comparten los demás tests."""
    payload = dict(payload_con_ordenes)
    payload['resultado'] = dict(payload['resultado'])
    ol = {k: v.copy()
          for k, v in payload_con_ordenes['resultado']['ordenes_limite'].items()}
    ol['resultado'][:] = 1        # _O_CANCELADA
    payload['resultado']['ordenes_limite'] = ol
    return payload


def test_el_tramo_fib_solo_se_pinta_sobre_ordenes_ejecutadas(app, payload_con_ordenes):
    """Colocar la orden no es entrar: la señal primaria confirma y la orden
    queda esperando un retroceso que casi nunca llega. Cada tramo dibujado —
    banda + los dos bordes — tiene que corresponder a una orden RELLENADA, no a
    una simplemente colocada."""
    w = _widget(app, payload_con_ordenes)
    ol = payload_con_ordenes['resultado']['ordenes_limite']
    n_rellenadas = len(_rellenadas(payload_con_ordenes))
    assert 0 < n_rellenadas < len(ol['idx_alta']), \
        "el fixture debe mezclar rellenadas y canceladas, o el test no prueba nada"

    con_fib = _n_artistas(w)
    w._capas_estado['fib'] = False
    sin_fib = _n_artistas(w)
    w._capas_estado['fib'] = True

    assert con_fib - sin_fib == 3 * n_rellenadas


def test_sin_ninguna_orden_ejecutada_no_hay_capa_fib(app, payload_con_ordenes):
    """Todas canceladas: el ojo desaparece del panel en vez de quedarse como una
    fila que no enciende nada. El de órdenes sí sigue — ahí es justo donde se
    ven las oportunidades perdidas."""
    w = _widget(app, _payload_sin_rellenar(payload_con_ordenes))
    w._dibujar_principal()
    claves = _claves(w)
    assert 'fib' not in claves
    assert 'ordenes' in claves


def test_la_vista_moderna_recorta_los_tramos_igual(app, payload_con_ordenes):
    """Las dos vistas tienen que enseñar lo mismo: las series de Fibonacci solo
    llevan valor en las velas que van de la colocación al relleno de una orden
    ejecutada."""
    from gui.widgets.lwc_chart import LwcChart, _AMARILLO_FIB

    unix = (pd.DatetimeIndex(payload_con_ordenes['timestamps']).asi8
            // 1_000_000_000).astype(np.int64)
    indicadores = _widget(app, payload_con_ordenes)._recolectar_indicadores(
        payload_con_ordenes)
    capas = LwcChart._construir_capas_limite(
        unix, payload_con_ordenes['high'], payload_con_ordenes['low'],
        payload_con_ordenes['resultado'], indicadores,
        {'zigzag': False, 'fib': True, 'ordenes': False})
    assert len(capas) == 2 and all(c['color'] == _AMARILLO_FIB for c in capas)

    ol = payload_con_ordenes['resultado']['ordenes_limite']
    esperado = set()
    for k in _rellenadas(payload_con_ordenes):
        esperado.update(range(int(ol['idx_alta'][k]), int(ol['idx_fin'][k]) + 1))
    for serie in capas:
        pintadas = {i for i, punto in enumerate(serie['data']) if 'value' in punto}
        assert pintadas == esperado

    # y con todas canceladas no queda ninguna serie que emitir
    assert LwcChart._construir_capas_limite(
        unix, payload_con_ordenes['high'], payload_con_ordenes['low'],
        _payload_sin_rellenar(payload_con_ordenes)['resultado'], indicadores,
        {'zigzag': False, 'fib': True, 'ordenes': False}) == []


def test_la_marca_de_nivel_cae_en_el_precio_de_ejecucion(app, payload_con_ordenes):
    """El tick marca el precio REAL al que ejecutó el motor, no el cierre de la
    barra: es lo que permite auditar el slippage y el nivel de un límite."""
    w = _widget(app, payload_con_ordenes)
    w._dibujar_principal()
    tr = payload_con_ordenes['resultado']['trades']
    es_long = tr['dir'] > 0
    esperado = {
        'compra': np.where(es_long, tr['precio_entrada'], tr['precio_salida']),
        'venta': np.where(es_long, tr['precio_salida'], tr['precio_entrada']),
    }
    for lado in ('compra', 'venta'):
        y = getattr(w, f'_scatter_nivel_{lado}').get_offsets()[:, 1]
        assert len(y) > 0
        assert np.allclose(np.sort(y), np.sort(esperado[lado]))


def test_la_flecha_va_fuera_de_la_vela_y_no_en_el_precio(app, payload_con_ordenes):
    """La flecha se separa del rango de la vela para leerse; la marca de nivel
    es la que señala la ejecución. Si coincidieran, el conector sobraría."""
    w = _widget(app, payload_con_ordenes)
    w._dibujar_principal()
    assert w._offset_flecha > 0
    for lado, serie, signo in (('compra', payload_con_ordenes['low'], -1),
                               ('venta', payload_con_ordenes['high'], +1)):
        y_flecha = getattr(w, f'_scatter_{lado}').get_offsets()[:, 1]
        y_nivel = getattr(w, f'_scatter_nivel_{lado}').get_offsets()[:, 1]
        assert not np.allclose(y_flecha, y_nivel)
        # la flecha queda estrictamente fuera del rango de su vela
        idx = getattr(w, f'_{lado}_idx_full')[
            w._trades_visibles(*w._ax_principal.get_xlim())]
        assert np.allclose(y_flecha, serie[idx] + signo * w._offset_flecha)


def test_el_repintado_por_scroll_no_mueve_los_marcadores(app, payload_con_ordenes):
    """Las alturas se calculan en tres sitios (dibujo, blitting y hover). Este
    test fuerza el camino de blitting sin cambiar el zoom: si alguno se
    calculara por su cuenta, los marcadores saltarían al arrastrar."""
    w = _widget(app, payload_con_ordenes)
    w._dibujar_principal()
    antes = {}
    for attr in ('_scatter_compra', '_scatter_venta',
                 '_scatter_nivel_compra', '_scatter_nivel_venta'):
        antes[attr] = getattr(w, attr).get_offsets().copy()

    w._actualizar_trades_dinamicos(*w._ax_principal.get_xlim())

    for attr, valor in antes.items():
        assert np.allclose(getattr(w, attr).get_offsets(), valor), attr


def test_la_flecha_queda_pegada_a_su_vela(app, payload_con_ordenes):
    """El hueco entre la flecha y la mecha se mide contra la vela típica, no
    contra el rango del gráfico entero: con el rango total, en un activo que ha
    recorrido mucho la flecha acababa a varias velas de la suya."""
    w = _widget(app, payload_con_ordenes)
    w._dibujar_principal()
    vela = np.nanmedian(payload_con_ordenes['high'] - payload_con_ordenes['low'])
    rango_total = (np.nanmax(payload_con_ordenes['high'])
                   - np.nanmin(payload_con_ordenes['low']))
    assert 0 < w._offset_flecha < vela
    assert w._offset_flecha < rango_total * 0.008   # el criterio anterior


def test_el_tooltip_de_una_entrada_por_limite_muestra_el_precio_pedido(
        app, payload_con_ordenes):
    w = _widget(app, payload_con_ordenes)
    w._dibujar_principal()
    assert w._orden_por_barra, "el fixture debe tener alguna orden rellenada"
    tr = payload_con_ordenes['resultado']['trades']
    fila = next(r for r in range(len(tr['pnl']))
                if int(tr['idx_entrada'][r]) in w._orden_por_barra)
    lineas = w._lineas_ejecucion(fila, es_salida=False,
                                 precio=tr['precio_entrada'][fila])
    texto = '\n'.join(lineas)
    assert 'Ejecución' in texto
    assert 'Pedido (límite)' in texto
    assert 'Diferencia' in texto
    assert 'Slippage' not in texto     # esa entrada no fue a mercado


def test_el_tooltip_de_una_salida_no_inventa_slippage_de_entrada(
        app, payload_con_ordenes):
    """La referencia de entrada (open o precio pedido) no aplica a un cierre:
    el tooltip de salida solo muestra el precio ejecutado."""
    w = _widget(app, payload_con_ordenes)
    w._dibujar_principal()
    lineas = w._lineas_ejecucion(0, es_salida=True, precio=123.45)
    assert lineas == ['Ejecución: 123.45']


def test_la_vista_moderna_construye_las_mismas_capas(app, payload_con_ordenes):
    """La vista Moderna ignoraba el ZigZag por completo; ahora debe emitir las
    mismas capas que la clásica para que conmutar de vista no cambie lo que se
    ve."""
    from gui.widgets.lwc_chart import LwcChart

    unix = (pd.DatetimeIndex(payload_con_ordenes['timestamps']).asi8
            // 1_000_000_000).astype(np.int64)
    indicadores = _widget(app, payload_con_ordenes)._recolectar_indicadores(
        payload_con_ordenes)
    assert indicadores['zigzags'] and indicadores['fibs']

    todas = LwcChart._construir_capas_limite(
        unix, payload_con_ordenes['high'], payload_con_ordenes['low'],
        payload_con_ordenes['resultado'], indicadores,
        {'zigzag': True, 'fib': True, 'ordenes': True})
    ninguna = LwcChart._construir_capas_limite(
        unix, payload_con_ordenes['high'], payload_con_ordenes['low'],
        payload_con_ordenes['resultado'], indicadores,
        {'zigzag': False, 'fib': False, 'ordenes': False})
    assert len(todas) > 0
    assert ninguna == []


def test_la_polilinea_moderna_usa_los_pivotes_del_zigzag(app, payload_con_ordenes):
    from gui.widgets.lwc_chart import LwcChart, _AZUL_ZIGZAG
    from core.strategies import _zigzag_pivotes

    unix = (pd.DatetimeIndex(payload_con_ordenes['timestamps']).asi8
            // 1_000_000_000).astype(np.int64)
    indicadores = _widget(app, payload_con_ordenes)._recolectar_indicadores(
        payload_con_ordenes)
    capas = LwcChart._construir_capas_limite(
        unix, payload_con_ordenes['high'], payload_con_ordenes['low'],
        payload_con_ordenes['resultado'], indicadores,
        {'zigzag': True, 'fib': False, 'ordenes': False})
    assert len(capas) == 1
    serie = capas[0]
    assert serie['color'] == _AZUL_ZIGZAG

    desviacion, piernas = next(iter(indicadores['zigzags']))
    df_zz = pd.DataFrame({'high': payload_con_ordenes['high'],
                          'low': payload_con_ordenes['low']})
    pivotes = _zigzag_pivotes(df_zz, desviacion, piernas)
    assert len(serie['data']) == len(pivotes)
    assert serie['data'][0]['value'] == pivotes[0][2]


# ── panel de capas ──

@pytest.fixture(scope='module')
def payload_variado(payload_con_ordenes):
    """Mismo backtest, pero declarando setups que usan familias distintas de
    indicador: es lo que hace interesante al catálogo."""
    payload = dict(payload_con_ordenes)
    payload['setups'] = list(payload_con_ordenes['setups']) + [
        {'nombre': 'RSI', 'plantilla': 'RSI', 'params': {'periodo': 14}},
        {'nombre': 'BB', 'plantilla': 'Bollinger + ATR',
         'params': {'periodo': 20, 'desv': 2.0}},
        {'nombre': 'SAR', 'plantilla': 'Parabolic SAR', 'params': {}},
        {'nombre': 'DC', 'plantilla': 'Breakout de canal (Donchian)',
         'params': {'periodo': 20}},
    ]
    return payload


def test_el_panel_lista_una_fila_por_capa_del_catalogo(app, payload_variado):
    w = _widget(app, payload_variado)
    w._dibujar_principal()
    claves = _claves(w)
    assert [f.clave for f in w.panel_capas._filas] == claves
    # las cuatro familias declaradas están, cada una con su clave paramétrica
    assert 'bb:20:2' in claves
    assert 'donchian:20' in claves
    assert any(c.startswith('sar:') for c in claves)
    assert 'panel:rsi' in claves


def test_el_color_del_panel_es_el_de_la_linea(app, payload_variado):
    """El cuadradito de la fila y el artista salen del mismo sitio. Si el dibujo
    volviera a asignar su propia paleta, el panel diría un color y el gráfico
    pintaría otro."""
    from gui.widgets.tab_backtest import COLOR_MEDIA_FIJO, PALETA_MEDIA

    w = _widget(app, payload_variado)
    w._dibujar_principal()
    for capa in w._capas_catalogo:
        assert w._color_capa(capa['clave']) == capa['color']
    # y la asignación sigue el criterio de siempre: color fijo por periodo
    # cuando lo hay, paleta rotatoria en orden de periodo cuando no
    for capa in w._capas_catalogo:
        if capa['clave'].startswith('ma:'):
            per = int(capa['clave'].split(':')[2])
            if per in COLOR_MEDIA_FIJO:
                assert capa['color'] == COLOR_MEDIA_FIJO[per]
            else:
                assert capa['color'] in PALETA_MEDIA


def test_apagar_un_oscilador_quita_su_panel(app, payload_variado):
    """Un oscilador no es una línea sobre el precio sino un panel propio: al
    apagarlo tiene que desaparecer el panel entero."""
    w = _widget(app, payload_variado)
    w._dibujar_principal()
    con_rsi = len(w._paneles)
    assert ('rsi', ) in [(k, ) for k, _ax in w._paneles]

    w._capas_estado['panel:rsi'] = False
    w._dibujar_principal()
    assert len(w._paneles) == con_rsi - 1
    assert 'rsi' not in [k for k, _ax in w._paneles]

    w._capas_estado['panel:rsi'] = True
    w._dibujar_principal()
    assert len(w._paneles) == con_rsi


def test_reencender_un_oscilador_conserva_su_altura(app, payload_variado):
    """_pesos_paneles va indexado por tipo de panel, no por posición: la altura
    que el usuario le haya dado a mano sobrevive a apagarlo y encenderlo."""
    w = _widget(app, payload_variado)
    w._dibujar_principal()
    w._pesos_paneles['rsi'] = 2.5
    w._capas_estado['panel:rsi'] = False
    w._dibujar_principal()
    w._capas_estado['panel:rsi'] = True
    w._dibujar_principal()
    assert w._pesos_paneles['rsi'] == 2.5


# ── eje Y de los paneles de oscilador ──

def _evento(x, y, dblclick=False):
    """Lo mínimo que leen _zona_eje / _on_press_ejes / _on_motion_ejes de un
    evento de matplotlib."""
    return SimpleNamespace(x=x, y=y, dblclick=dblclick, inaxes=None,
                           xdata=None, ydata=None, button=1)


def _punto_franja_y(ax):
    b = ax.get_window_extent()
    return (b.x1 + 8, (b.y0 + b.y1) / 2)


def test_la_franja_y_de_cada_panel_tiene_su_propia_zona(app, payload_variado):
    """Antes, un clic sobre las etiquetas de un oscilador caía fuera de las
    tres zonas y no hacía nada. Las zonas ya existentes no pueden perder
    prioridad: el borde entre paneles sigue siendo para redimensionar."""
    w = _widget(app, payload_variado)
    w._dibujar_principal()
    ax_precio, ax_osc = w._paneles[0][1], w._paneles[1][1]

    assert w._zona_eje(_evento(*_punto_franja_y(ax_osc))) == 'y:1'
    assert w._zona_eje(_evento(*_punto_franja_y(ax_precio))) == 'y'
    b_precio, b_osc = ax_precio.get_window_extent(), ax_osc.get_window_extent()
    borde = _evento((b_precio.x0 + b_precio.x1) / 2, (b_osc.y1 + b_precio.y0) / 2)
    assert w._zona_eje(borde) == 'resize_panel:0'


def test_arrastrar_el_eje_y_de_un_panel_solo_le_afecta_a_el(app, payload_variado):
    w = _widget(app, payload_variado)
    w._dibujar_principal()
    kind, ax_osc = w._paneles[1]
    ylim_osc0, ylim_precio0 = ax_osc.get_ylim(), w._ax_principal.get_ylim()

    x, y = _punto_franja_y(ax_osc)
    w._on_press_ejes(_evento(x, y))
    assert w._drag_modo == 'y:1'
    w._on_motion_ejes(_evento(x, y + 60))
    w._on_release_ejes(_evento(x, y + 60))

    assert ax_osc.get_ylim() != ylim_osc0
    assert w._ax_principal.get_ylim() == ylim_precio0, "el precio no se toca"
    assert w._ylim_paneles[kind] == pytest.approx(ax_osc.get_ylim())


def test_la_escala_del_panel_sobrevive_a_apagarlo_y_reencenderlo(app, payload_variado):
    """_ylim_paneles va indexado por TIPO, igual que _pesos_paneles: si fuera
    por posición se perdería en cuanto cambie el número de paneles."""
    w = _widget(app, payload_variado)
    w._dibujar_principal()
    w._ylim_paneles['rsi'] = (20.0, 80.0)

    w._dibujar_principal()
    ax_rsi = dict(w._paneles)['rsi']
    assert ax_rsi.get_ylim() == pytest.approx((20.0, 80.0))

    w._capas_estado['panel:rsi'] = False
    w._dibujar_principal()
    w._capas_estado['panel:rsi'] = True
    w._dibujar_principal()
    assert dict(w._paneles)['rsi'].get_ylim() == pytest.approx((20.0, 80.0))


def test_doble_clic_en_la_franja_y_devuelve_la_escala_natural(app, payload_variado):
    w = _widget(app, payload_variado)
    w._dibujar_principal()
    w._ylim_paneles['rsi'] = (20.0, 80.0)
    w._dibujar_principal()

    ax_rsi = dict(w._paneles)['rsi']
    idx = [k for k, _ax in w._paneles].index('rsi')
    x, y = _punto_franja_y(ax_rsi)
    assert w._zona_eje(_evento(x, y)) == f'y:{idx}'
    w._on_press_ejes(_evento(x, y, dblclick=True))

    assert 'rsi' not in w._ylim_paneles
    assert dict(w._paneles)['rsi'].get_ylim() == pytest.approx((0.0, 100.0))


def test_conmutar_una_capa_no_mueve_el_encuadre(app, payload_variado):
    """Apagar un indicador redibuja el gráfico entero, y el Axes nuevo se
    autoescalaría a TODA la serie: sin conservar los límites, el gráfico pegaba
    un salto justo cuando el usuario está mirando una zona concreta."""
    w = _widget(app, payload_variado)
    w._dibujar_principal()
    ax = w._ax_principal
    ax.set_xlim(w._x_full[100], w._x_full[300])
    w._redibujar_datos(ax)

    # escala de precio en automático
    antes = (ax.get_xlim(), ax.get_ylim())
    w._conmutar_capa('donchian:20', False)
    ax = w._ax_principal
    assert np.allclose(ax.get_xlim(), antes[0])
    assert np.allclose(ax.get_ylim(), antes[1])

    # y ajustada a mano, que es donde estaba el fallo
    ax.set_ylim(60, 95)
    w._y_manual = True
    antes = (ax.get_xlim(), ax.get_ylim())
    w._conmutar_capa('donchian:20', True)
    ax = w._ax_principal
    assert np.allclose(ax.get_xlim(), antes[0])
    assert np.allclose(ax.get_ylim(), antes[1])


def test_apagar_un_oscilador_no_encoge_el_canvas(app, payload_variado):
    """Si el canvas menguara al ocultar un panel, la pestaña entera se
    recolocaría bajo el ratón. El hueco se lo queda el panel de precio."""
    w = _widget(app, payload_variado)
    w._dibujar_principal()
    alto = w.canvas.minimumHeight()
    w._capas_estado['panel:rsi'] = False
    w._dibujar_principal()
    assert w.canvas.minimumHeight() == alto
    assert 'rsi' not in [k for k, _ax in w._paneles]


def test_el_ojo_del_panel_redibuja_y_apaga_la_capa(app, payload_variado):
    """Recorre el camino real (el widget), no solo el dict de estado."""
    w = _widget(app, payload_variado)
    w._dibujar_principal()
    fila = next(f for f in w.panel_capas._filas if f.clave == 'donchian:20')
    antes = len(w._art_overlays_extra)
    fila.ojo.setChecked(False)
    assert w._capas_estado['donchian:20'] is False
    assert len(w._art_overlays_extra) < antes
