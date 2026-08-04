"""Primitiva de orden límite pendiente del motor (core/backtest.py): relleno,
cancelación, caducidad y su convivencia con las entradas a mercado."""
import numpy as np

from core.backtest import simular

CONFIG_BASE = {'capital_inicial': 10000.0, 'riesgo_pct': 0.01,
               'comision_pct': 0.0, 'slippage_pct': 0.0,
               'stop_atr': 0.0, 'tp_r': 0.0, 'salida_n_velas': 0}

RELLENADA, CANCELADA, EXPIRADA = 0, 1, 2


def _ohlc_plano(n, precio=100.0):
    o = np.full(n, precio)
    h = np.full(n, precio + 1.0)
    l = np.full(n, precio - 1.0)
    c = np.full(n, precio)
    return o, h, l, c


def _senales(n):
    return {'entradas_long': np.zeros(n, dtype=bool),
            'entradas_short': np.zeros(n, dtype=bool),
            'salidas_long': np.zeros(n, dtype=bool),
            'salidas_short': np.zeros(n, dtype=bool),
            'setup_id': np.zeros(n, dtype=np.int64),
            'atr': np.full(n, 2.0),
            'orden_long_precio': np.full(n, np.nan),
            'orden_short_precio': np.full(n, np.nan),
            'cancelar_orden_long': np.zeros(n, dtype=np.int64),
            'cancelar_orden_short': np.zeros(n, dtype=np.int64)}


def _config(**extra):
    cfg = dict(CONFIG_BASE)
    if extra:
        cfg['config_por_setup'] = {0: extra}
    return cfg


def test_sin_claves_de_orden_el_motor_se_comporta_igual_que_siempre():
    """Las señales sin las claves nuevas deben dar exactamente el mismo
    resultado: la primitiva no puede cambiar nada de lo existente."""
    n = 10
    o, h, l, c = _ohlc_plano(n)
    o[6] = 110.0
    sin_claves = {'entradas_long': np.zeros(n, dtype=bool),
                  'entradas_short': np.zeros(n, dtype=bool),
                  'salidas_long': np.zeros(n, dtype=bool),
                  'salidas_short': np.zeros(n, dtype=bool),
                  'setup_id': np.zeros(n, dtype=np.int64),
                  'atr': np.full(n, 2.0)}
    sin_claves['entradas_long'][2] = True
    sin_claves['salidas_long'][5] = True
    r = simular(o, h, l, c, sin_claves, CONFIG_BASE)
    assert r['n_trades'] == 1
    assert r['trades']['precio_entrada'][0] == 100.0
    assert len(r['ordenes_limite']['idx_alta']) == 0


def test_orden_nunca_tocada_expira_por_vigencia():
    n = 20
    o, h, l, c = _ohlc_plano(n)
    s = _senales(n)
    s['orden_long_precio'][2] = 50.0   # muy por debajo: nunca se toca
    r = simular(o, h, l, c, s, _config(limite_vigencia_velas=5))
    ol = r['ordenes_limite']
    assert r['n_trades'] == 0
    assert len(ol['idx_alta']) == 1
    assert ol['resultado'][0] == EXPIRADA
    assert ol['idx_alta'][0] == 2
    assert ol['idx_fin'][0] == 7   # 5 velas después de colocarla


def test_sin_vigencia_la_orden_sigue_viva_hasta_el_final():
    n = 20
    o, h, l, c = _ohlc_plano(n)
    s = _senales(n)
    s['orden_long_precio'][2] = 50.0
    r = simular(o, h, l, c, s, _config())   # vigencia 0 = sin caducidad
    assert len(r['ordenes_limite']['idx_alta']) == 0   # nunca se resolvió
    assert r['n_trades'] == 0


def test_se_rellena_al_precio_exacto_de_la_orden():
    n = 20
    o, h, l, c = _ohlc_plano(n)   # low de fondo = 99, así que 98 no se toca
    l[6] = 97.0            # esta vela sí toca el límite...
    o[6] = 100.0           # ...pero abre por encima, así que no hay mejora
    s = _senales(n)
    s['orden_long_precio'][2] = 98.0
    s['salidas_long'][10] = True
    r = simular(o, h, l, c, s, _config())
    ol = r['ordenes_limite']
    assert ol['resultado'][0] == RELLENADA
    assert ol['idx_fin'][0] == 6
    assert r['n_trades'] == 1
    assert r['trades']['precio_entrada'][0] == 98.0
    assert r['trades']['idx_entrada'][0] == 6   # abre en la vela del relleno


def test_a_mercado_el_precio_de_entrada_es_el_open_mas_el_slippage():
    """Lo que el gráfico enseñará como «Slippage» en el tooltip: la distancia
    entre el open de la vela y el precio realmente ejecutado."""
    n = 20
    o, h, l, c = _ohlc_plano(n)
    o[6] = 100.0
    slippage = 0.001
    cfg = dict(CONFIG_BASE, slippage_pct=slippage)
    s = _senales(n)
    s['entradas_long'][5] = True
    s['salidas_long'][10] = True
    r = simular(o, h, l, c, s, cfg)
    assert r['trades']['precio_entrada'][0] == 100.0 * (1.0 + slippage)

    s_corto = _senales(n)
    s_corto['entradas_short'][5] = True
    s_corto['salidas_short'][10] = True
    r_corto = simular(o, h, l, c, s_corto, cfg)
    # el slippage siempre juega en contra: el corto vende más barato
    assert r_corto['trades']['precio_entrada'][0] == 100.0 * (1.0 - slippage)


def test_por_limite_el_precio_de_entrada_es_el_nivel_pedido():
    """Sin slippage, una orden límite rellenada por mecha entra exactamente en
    su nivel: es la comprobación visual que se hace en el gráfico contrastando
    la marca de entrada con el segmento de la orden."""
    n = 20
    o, h, l, c = _ohlc_plano(n)
    l[10] = 97.0
    s = _senales(n)
    s['orden_long_precio'][2] = 98.0
    s['salidas_long'][15] = True
    r = simular(o, h, l, c, s, _config())
    assert r['trades']['precio_entrada'][0] == 98.0
    assert r['ordenes_limite']['precio'][0] == 98.0


def test_el_precio_registrado_es_el_pedido_no_el_de_ejecucion():
    """La columna 'precio' de ordenes_limite es el nivel en el que la orden
    estuvo esperando, también cuando se rellena mejor. Si guardara el precio de
    ejecución, el gráfico dibujaría la orden fuera de su nivel de Fibonacci."""
    n = 20
    o, h, l, c = _ohlc_plano(n)
    o[6] = 95.0        # abre mucho mejor que el límite
    l[6] = 94.0
    s = _senales(n)
    s['orden_long_precio'][2] = 98.0
    s['salidas_long'][10] = True
    r = simular(o, h, l, c, s, _config())
    assert r['ordenes_limite']['precio'][0] == 98.0     # lo pedido
    assert r['trades']['precio_entrada'][0] == 95.0     # lo pagado


def test_un_hueco_mejor_que_el_limite_rellena_al_open():
    """Una orden límite no se ejecuta peor que su precio, pero sí mejor: si el
    open ya abrió más abajo, ese es el precio real."""
    n = 20
    o, h, l, c = _ohlc_plano(n)
    o[6] = 95.0
    l[6] = 94.0
    s = _senales(n)
    s['orden_long_precio'][2] = 98.0
    s['salidas_long'][10] = True
    r = simular(o, h, l, c, s, _config())
    assert r['trades']['precio_entrada'][0] == 95.0


def test_short_se_rellena_al_tocar_por_arriba():
    n = 20
    o, h, l, c = _ohlc_plano(n)
    h[6] = 102.0
    s = _senales(n)
    s['orden_short_precio'][2] = 101.0
    s['salidas_short'][10] = True
    r = simular(o, h, l, c, s, _config())
    assert r['ordenes_limite']['resultado'][0] == RELLENADA
    assert r['trades']['precio_entrada'][0] == 101.0
    assert r['trades']['dir'][0] == -1


def test_el_bit_de_cancelacion_del_setup_propietario_la_mata():
    n = 20
    o, h, l, c = _ohlc_plano(n)
    s = _senales(n)
    s['orden_long_precio'][2] = 50.0
    s['cancelar_orden_long'][5] = np.int64(1)   # bit del setup 0
    r = simular(o, h, l, c, s, _config())
    ol = r['ordenes_limite']
    assert ol['resultado'][0] == CANCELADA
    assert ol['idx_fin'][0] == 5


def test_el_bit_de_otro_setup_no_toca_la_orden():
    """Aislamiento del bitmask: la cancelación de un setup no puede matar la
    orden de otro, igual que su señal de salida no cierra su posición."""
    n = 20
    o, h, l, c = _ohlc_plano(n)
    s = _senales(n)
    s['orden_long_precio'][2] = 50.0
    s['cancelar_orden_long'][5] = np.int64(1) << np.int64(3)   # setup 3
    r = simular(o, h, l, c, s, _config())
    assert len(r['ordenes_limite']['idx_alta']) == 0   # sigue viva


def test_cancelacion_por_avance_a_favor():
    """El precio se fue a favor sin tocar el límite: el retroceso ya no va a
    llegar a ese nivel y la orden se cancela."""
    n = 20
    o, h, l, c = _ohlc_plano(n)
    # stop_atr=1 y ATR=2 -> R = 2; avance de 3R = 6 sobre el precio de la orden
    h[8] = 105.0
    s = _senales(n)
    s['orden_long_precio'][2] = 98.0
    r = simular(o, h, l, c, s,
                _config(stop_atr=1.0, limite_cancelar_avance_r=3.0))
    ol = r['ordenes_limite']
    assert ol['resultado'][0] == CANCELADA
    assert ol['idx_fin'][0] == 8
    assert r['n_trades'] == 0


def test_el_avance_a_favor_desactivado_no_cancela():
    n = 20
    o, h, l, c = _ohlc_plano(n)
    h[8] = 105.0
    s = _senales(n)
    s['orden_long_precio'][2] = 98.0
    r = simular(o, h, l, c, s, _config(stop_atr=1.0))
    assert len(r['ordenes_limite']['idx_alta']) == 0


def test_una_orden_viva_no_se_pisa_con_una_senal_nueva():
    n = 20
    o, h, l, c = _ohlc_plano(n)
    l[10] = 97.0
    s = _senales(n)
    s['orden_long_precio'][2] = 98.0
    s['orden_long_precio'][4] = 80.0   # ignorada: ya hay una orden pendiente
    s['salidas_long'][15] = True
    r = simular(o, h, l, c, s, _config())
    ol = r['ordenes_limite']
    assert len(ol['idx_alta']) == 1
    assert ol['idx_alta'][0] == 2
    assert ol['idx_fin'][0] == 10
    assert r['trades']['precio_entrada'][0] == 98.0


def test_una_orden_viva_tiene_prioridad_sobre_una_entrada_a_mercado():
    n = 20
    o, h, l, c = _ohlc_plano(n)
    l[10] = 97.0
    s = _senales(n)
    s['orden_long_precio'][2] = 98.0
    s['entradas_long'][4] = True   # no debe abrir a mercado en la vela 5
    s['salidas_long'][15] = True
    r = simular(o, h, l, c, s, _config())
    assert r['trades']['idx_entrada'][0] == 10
    assert r['trades']['precio_entrada'][0] == 98.0


def test_la_salida_de_un_trade_por_limite_es_identica_a_una_a_mercado():
    """Test de paridad: nada de lo que ocurre DESPUÉS de abrir depende de cómo
    se abrió. Mismo precio de entrada por las dos vías -> mismo trade."""
    n = 20
    o, h, l, c = _ohlc_plano(n)
    l[:] = 99.5        # suelo por encima del límite: no se toca antes de tiempo
    o[6] = 99.0        # a mercado entraría aquí al open
    l[6] = 99.0        # y por límite se rellena en la misma vela y al mismo precio
    o[12] = 105.0

    s_mercado = _senales(n)
    s_mercado['entradas_long'][5] = True
    s_mercado['salidas_long'][11] = True
    r_mercado = simular(o, h, l, c, s_mercado, _config(stop_atr=2.0, tp_r=3.0))

    s_limite = _senales(n)
    s_limite['orden_long_precio'][2] = 99.0
    s_limite['salidas_long'][11] = True
    r_limite = simular(o, h, l, c, s_limite, _config(stop_atr=2.0, tp_r=3.0))

    for clave in ('idx_entrada', 'idx_salida', 'precio_entrada', 'precio_salida',
                  'pnl', 'motivo', 'parcial', 'unidades', 'precio_stop'):
        assert np.array_equal(r_mercado['trades'][clave], r_limite['trades'][clave]), clave


def test_tras_resolverse_se_admite_una_orden_nueva():
    n = 30
    o, h, l, c = _ohlc_plano(n)
    s = _senales(n)
    s['orden_long_precio'][2] = 50.0
    s['orden_long_precio'][12] = 40.0
    r = simular(o, h, l, c, s, _config(limite_vigencia_velas=4))
    ol = r['ordenes_limite']
    assert ol['idx_alta'].tolist() == [2, 12]
    assert ol['resultado'].tolist() == [EXPIRADA, EXPIRADA]
