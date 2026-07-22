import numpy as np
import pandas as pd
import pytest

from core.backtest import (
    simular, dividir_is_oos, calcular_metricas, walk_forward, montecarlo,
    _analizar_drawdowns,
)
from core.strategies import (
    generar_senales, generar_senales_sistema, describir, ESTRATEGIAS,
    params_por_defecto, sma, codigo_setup, codigo_sistema, defaults_setup,
)


def _ohlc_plano(n, precio=100.0):
    o = np.full(n, precio)
    h = np.full(n, precio + 1.0)
    l = np.full(n, precio - 1.0)
    c = np.full(n, precio)
    return o, h, l, c


def _senales_vacias(n):
    return {'entradas_long': np.zeros(n, dtype=bool),
            'entradas_short': np.zeros(n, dtype=bool),
            'salidas_long': np.zeros(n, dtype=bool),
            'salidas_short': np.zeros(n, dtype=bool),
            'setup_id': np.zeros(n, dtype=np.int64),
            'atr': np.full(n, 2.0)}


CONFIG_BASE = {'capital_inicial': 10000.0, 'riesgo_pct': 0.01,
               'comision_pct': 0.0, 'slippage_pct': 0.0,
               'stop_atr': 0.0, 'tp_r': 0.0, 'salida_n_velas': 0}


def test_trade_long_pnl_exacto():
    # señal en t=2 -> entra al open de t=3 (100); señal de salida en t=5 ->
    # sale al open de t=6 (110). ATR=2, sin stop -> dist referencia 4.
    # unidades = 10000*0.01/4 = 25 -> pnl = 25*(110-100) = 250.
    n = 10
    o, h, l, c = _ohlc_plano(n)
    o[6] = 110.0
    s = _senales_vacias(n)
    s['entradas_long'][2] = True
    s['salidas_long'][5] = True
    r = simular(o, h, l, c, s, CONFIG_BASE)
    assert r['n_trades'] == 1
    t = r['trades']
    assert t['idx_entrada'][0] == 3
    assert t['idx_salida'][0] == 6
    assert t['dir'][0] == 1
    assert t['pnl'][0] == pytest.approx(250.0)
    assert r['capital_final'] == pytest.approx(10250.0)


def test_comision_reduce_pnl():
    n = 10
    o, h, l, c = _ohlc_plano(n)
    o[6] = 110.0
    s = _senales_vacias(n)
    s['entradas_long'][2] = True
    s['salidas_long'][5] = True
    cfg = dict(CONFIG_BASE, comision_pct=0.001)
    r = simular(o, h, l, c, s, cfg)
    # comision = (100+110)*25*0.001 = 5.25
    assert r['trades']['pnl'][0] == pytest.approx(250.0 - 5.25)


def test_stop_se_ejecuta_en_la_vela_correcta():
    # entrada al open de t=3 a 100 con stop 1.5*ATR=3 -> stop en 97.
    # low de t=5 toca 96 -> sale a 97 en t=5 con motivo stop (1).
    n = 10
    o, h, l, c = _ohlc_plano(n)
    l[5] = 96.0
    s = _senales_vacias(n)
    s['entradas_long'][2] = True
    cfg = dict(CONFIG_BASE, stop_atr=1.5)
    r = simular(o, h, l, c, s, cfg)
    t = r['trades']
    assert r['n_trades'] == 1
    assert t['idx_salida'][0] == 5
    assert t['motivo'][0] == 1
    assert t['precio_salida'][0] == pytest.approx(97.0)
    # riesgo: unidades = 10000*0.01/3 -> perdida = unidades*3 = 100 = 1%
    assert t['pnl'][0] == pytest.approx(-100.0)
    # precio_stop persistido = precio de entrada (100) - 1.5*ATR(2) = 97
    assert t['precio_stop'][0] == pytest.approx(97.0)


def test_precio_stop_persistido_long_y_short():
    """El motor calcula stop_precio al entrar (core/backtest.py:107) pero se
    descartaba; ahora se guarda por trade en 'precio_stop' para poder
    dibujarlo en la gráfica de operaciones."""
    n = 10
    o, h, l, c = _ohlc_plano(n)   # ATR=2 constante (ver _base_senales/atr)
    s = _senales_vacias(n)
    s['entradas_long'][2] = True
    s['salidas_long'][5] = True
    cfg = dict(CONFIG_BASE, stop_atr=1.5)
    r = simular(o, h, l, c, s, cfg)
    t = r['trades']
    assert r['n_trades'] == 1
    # long: stop por debajo de la entrada -> 100 - 1.5*2 = 97
    assert t['precio_stop'][0] == pytest.approx(97.0)

    s2 = _senales_vacias(n)
    s2['entradas_short'][2] = True
    s2['salidas_short'][5] = True
    r2 = simular(o, h, l, c, s2, cfg)
    t2 = r2['trades']
    assert r2['n_trades'] == 1
    # short: stop por encima de la entrada -> 100 + 1.5*2 = 103
    assert t2['precio_stop'][0] == pytest.approx(103.0)


def test_precio_stop_es_cero_sin_stop_configurado():
    n = 10
    o, h, l, c = _ohlc_plano(n)
    s = _senales_vacias(n)
    s['entradas_long'][2] = True
    s['salidas_long'][5] = True
    r = simular(o, h, l, c, s, CONFIG_BASE)   # stop_atr=0.0 por defecto
    assert r['n_trades'] == 1
    assert r['trades']['precio_stop'][0] == pytest.approx(0.0)


def test_riesgo_por_setup_distinto():
    n = 14
    o, h, l, c = _ohlc_plano(n)
    l[4] = 90.0    # stop del primer trade
    l[10] = 90.0   # stop del segundo
    s = _senales_vacias(n)
    s['entradas_long'][2] = True
    s['setup_id'][2] = 0
    s['entradas_long'][8] = True
    s['setup_id'][8] = 3
    cfg = dict(CONFIG_BASE, stop_atr=1.0,
               riesgo_por_setup={3: 0.02})
    r = simular(o, h, l, c, s, cfg)
    t = r['trades']
    assert r['n_trades'] == 2
    # trade 1 (setup 0): pierde 1% del equity al entrar
    assert t['pnl'][0] == pytest.approx(-0.01 * t['equity_entrada'][0])
    # trade 2 (setup 3): pierde 2% del equity al entrar
    assert t['pnl'][1] == pytest.approx(-0.02 * t['equity_entrada'][1])


def test_salida_por_tiempo():
    n = 12
    o, h, l, c = _ohlc_plano(n)
    s = _senales_vacias(n)
    s['entradas_long'][2] = True
    cfg = dict(CONFIG_BASE, salida_n_velas=4)
    r = simular(o, h, l, c, s, cfg)
    t = r['trades']
    assert t['idx_entrada'][0] == 3
    assert t['idx_salida'][0] == 7   # 4 velas despues de entrar
    assert t['motivo'][0] == 3


def test_dividir_is_oos():
    assert dividir_is_oos(1000, 0.30) == 700
    assert dividir_is_oos(1000, 0.0) == 1000


def test_metricas_por_tramo():
    n = 20
    o, h, l, c = _ohlc_plano(n)
    o[6] = 110.0   # trade IS ganador
    l[15] = 90.0   # trade OOS perdedor (stop)
    s = _senales_vacias(n)
    s['entradas_long'][2] = True
    s['salidas_long'][5] = True
    s['entradas_long'][12] = True
    cfg = dict(CONFIG_BASE, stop_atr=1.0)
    r = simular(o, h, l, c, s, cfg)
    corte = 10
    m_is = calcular_metricas(r, 0, corte)
    m_oos = calcular_metricas(r, corte, n)
    m_tot = calcular_metricas(r)
    assert m_is['n_trades'] == 1 and m_is['win_rate'] == 1.0
    assert m_oos['n_trades'] == 1 and m_oos['win_rate'] == 0.0
    assert m_tot['n_trades'] == 2
    assert m_tot['pnl_total'] == pytest.approx(m_is['pnl_total'] + m_oos['pnl_total'])


def test_walk_forward_ventanas():
    n = 100
    o, h, l, c = _ohlc_plano(n)
    s = _senales_vacias(n)
    wf = walk_forward(o, h, l, c, s, CONFIG_BASE, n_ventanas=4)
    assert len(wf['ventanas']) == 4
    assert wf['ventanas'][0]['idx_ini'] == 0
    assert wf['ventanas'][-1]['idx_fin'] == n


def test_montecarlo_determinista_con_semilla():
    trades = {'ret_pct': np.array([0.02, -0.01, 0.03, -0.02, 0.01, 0.015])}
    mc1 = montecarlo(trades, 10000.0, n_sims=200, semilla=42)
    mc2 = montecarlo(trades, 10000.0, n_sims=200, semilla=42)
    assert mc1['n_sims'] == 200
    assert np.array_equal(mc1['finales'], mc2['finales'])
    assert 0.0 <= mc1['prob_negativo'] <= 1.0
    assert (mc1['max_dds'] <= 0).all()
    assert len(mc1['curvas_pct']['p50']) == len(trades['ret_pct']) + 1


def test_montecarlo_sin_trades():
    mc = montecarlo({'ret_pct': np.array([])}, 10000.0)
    assert mc['n_sims'] == 0


# ══════════════ estrategias ══════════════

def _df_sintetico(closes):
    closes = np.asarray(closes, dtype=np.float64)
    return pd.DataFrame({
        'open': closes, 'high': closes + 0.5,
        'low': closes - 0.5, 'close': closes,
    })


def test_cruce_medias_senales():
    # serie que baja y luego sube fuerte: la SMA rapida cruza a la lenta
    closes = [100] * 10 + [90] * 10 + [120] * 15
    df = _df_sintetico(closes)
    s = generar_senales('Cruce de medias', df,
                        {'tipo': 'SMA', 'rapida': 3, 'lenta': 8, 'direccion': 'Ambas'})
    assert s['entradas_long'].sum() >= 1
    assert s['entradas_short'].sum() >= 1
    # el cruce alcista debe ocurrir en el tramo de subida
    assert np.flatnonzero(s['entradas_long'])[-1] >= 20


def test_bollinger_senales():
    rng = np.random.default_rng(7)
    closes = 100 + np.cumsum(rng.normal(0, 0.3, 300))
    closes[150] -= 8   # pinchazo por debajo de la banda inferior
    df = _df_sintetico(closes)
    s = generar_senales('Bollinger + ATR', df,
                        {'periodo': 20, 'desv': 2.0, 'direccion': 'Long'})
    assert s['entradas_long'][150]
    assert s['entradas_short'].sum() == 0   # direccion Long


def test_patrones_senales_y_setup():
    # martillo tras tendencia bajista (mismo caso que test_candle_patterns)
    from tests.test_candle_patterns import _baseline, _arrays
    o, h, l, c = _arrays(_baseline(-1), [(99, 99.6, 95.5, 99.4)])
    df = pd.DataFrame({'open': o, 'high': h, 'low': l, 'close': c})
    s = generar_senales('Patrones de velas', df,
                        {'patrones': ['Martillo'], 'lag_salida': 5})
    assert s['entradas_long'][25]
    assert s['setup_id'][25] == 0


def test_custom_regla_close_mayor_sma():
    closes = [100] * 30 + [130] * 10
    df = _df_sintetico(closes)
    reglas = {'entradas_long': [{'setup_id': 2, 'condiciones': [
                  {'izq': {'tipo': 'close'}, 'op': 'cruza arriba',
                   'der': {'tipo': 'SMA', 'periodo': 20}}]}],
              'salidas_long': [], 'entradas_short': [], 'salidas_short': []}
    s = generar_senales('Custom (reglas)', df, {'reglas': reglas})
    idx = np.flatnonzero(s['entradas_long'])
    assert len(idx) == 1
    assert idx[0] == 30    # primera vela a 130, cruza sobre la SMA20
    assert s['setup_id'][30] == 2


def test_salida_de_otro_setup_no_cierra_la_posicion():
    # setup 0 entra en t=2; en t=5 SOLO el setup 1 pide salir (bit 1) -> la
    # posicion sigue abierta; en t=8 pide salir el setup 0 (bit 0) -> cierra
    # al open de t=9.
    n = 14
    o, h, l, c = _ohlc_plano(n)
    s = _senales_vacias(n)
    s['entradas_long'][2] = True
    s['setup_id'][2] = 0
    sal = np.zeros(n, dtype=np.int64)
    sal[5] = 1 << 1     # solo setup 1
    sal[8] = 1 << 0     # setup 0
    s['salidas_long'] = sal
    r = simular(o, h, l, c, s, CONFIG_BASE)
    assert r['n_trades'] == 1
    assert r['trades']['idx_salida'][0] == 9
    assert r['trades']['motivo'][0] == 0


def test_entrada_contraria_de_otro_setup_no_revierte():
    # setup 0 long abierto; setup 1 da entrada short en t=5 -> NO cierra ni
    # revierte (la posicion es del setup 0); sin mas senales, cierra por fin
    # de datos.
    n = 10
    o, h, l, c = _ohlc_plano(n)
    s = _senales_vacias(n)
    s['entradas_long'][2] = True
    s['setup_id'][2] = 0
    s['entradas_short'][5] = True
    s['setup_id'][5] = 1
    s['salidas_long'] = np.zeros(n, dtype=np.int64)   # bitmask sin salidas
    r = simular(o, h, l, c, s, CONFIG_BASE)
    assert r['n_trades'] == 1
    assert r['trades']['dir'][0] == 1
    assert r['trades']['motivo'][0] == 4   # fin de datos, nunca revirtio


def test_config_stop_y_tiempo_por_setup():
    # setup 0: stop 1xATR (salta en t=4); setup 5: sin stop y salida a 3
    # velas. Mismo sistema, comportamientos independientes.
    n = 16
    o, h, l, c = _ohlc_plano(n)
    l[4] = 90.0
    s = _senales_vacias(n)
    s['entradas_long'][2] = True
    s['setup_id'][2] = 0
    s['entradas_long'][8] = True
    s['setup_id'][8] = 5
    cfg = dict(CONFIG_BASE, config_por_setup={
        0: {'stop_atr': 1.0},
        5: {'stop_atr': 0.0, 'salida_n_velas': 3, 'riesgo_pct': 0.02},
    })
    r = simular(o, h, l, c, s, cfg)
    t = r['trades']
    assert r['n_trades'] == 2
    assert t['motivo'][0] == 1            # setup 0: stop
    assert t['motivo'][1] == 3            # setup 5: tiempo
    assert t['idx_salida'][1] == 12       # entro en 9, +3 velas
    assert t['setup'][1] == 5


def test_sistema_dos_setups_fusion_y_prioridad():
    closes = [100] * 30 + [130] * 10
    df = _df_sintetico(closes)
    setups = [
        {'plantilla': 'Cruce de medias',
         'params': {'tipo': 'SMA', 'rapida': 3, 'lenta': 10, 'direccion': 'Long'}},
        {'plantilla': 'RSI',
         'params': {'periodo': 5, 'sobreventa': 30.0, 'sobrecompra': 70.0,
                    'direccion': 'Ambas'}},
    ]
    s = generar_senales_sistema(df, setups)
    assert s['salidas_long'].dtype == np.int64    # bitmask
    # el RSI(5) con el salto a 130 dispara sobrecompra (entrada short, setup 1)
    assert (s['setup_id'][s['entradas_short']] == 1).all()
    # toda entrada tiene setup 0 o 1
    entradas = s['entradas_long'] | s['entradas_short']
    assert set(np.unique(s['setup_id'][entradas])) <= {0, 1}


def test_descripciones_no_vacias_e_interpoladas():
    for nombre in ESTRATEGIAS:
        texto = describir(nombre)
        assert isinstance(texto, str) and len(texto) > 0
    d = describir('Cruce de medias', {'tipo': 'EMA', 'rapida': 9, 'lenta': 21})
    assert 'EMA(9)' in d and 'EMA(21)' in d
    d = describir('RSI', {'periodo': 7, 'sobreventa': 25.0})
    assert 'RSI(7)' in d and '25' in d


def test_codigo_setup_estructura_e_interpolacion():
    setup = {'nombre': 'Cruce lento', 'plantilla': 'Cruce de medias',
             'params': {'tipo': 'EMA', 'rapida': 9, 'lenta': 21, 'direccion': 'Long'},
             'riesgo_pct': 0.015, 'stop_atr': 0.0, 'tp_r': 0.0, 'salida_n_velas': 0}
    cod = codigo_setup(setup, 0)
    assert 'VARIABLES' in cod and 'ENTRADA' in cod and 'SALIDA' in cod
    assert 'EMA(9)' in cod and 'EMA(21)' in cod
    assert 'riesgo = 1.5%' in cod
    assert 'stop = ninguno' in cod


def test_codigo_setup_todas_las_plantillas():
    for plantilla in ESTRATEGIAS:
        setup = {'nombre': 'x', 'plantilla': plantilla,
                 'params': params_por_defecto(plantilla),
                 'riesgo_pct': 0.01, 'stop_atr': 1.0, 'tp_r': 2.0,
                 'salida_n_velas': 0}
        cod = codigo_setup(setup, 0)
        assert 'VARIABLES' in cod and 'ENTRADA' in cod and 'SALIDA' in cod


def test_codigo_sistema_incluye_cuenta():
    setups = [{'nombre': 'a', 'plantilla': 'RSI',
               'params': params_por_defecto('RSI'),
               'riesgo_pct': 0.01, 'stop_atr': 2.0, 'tp_r': 0.0,
               'salida_n_velas': 0}]
    cod = codigo_sistema(setups, {'capital_inicial': 10000.0,
                                  'comision_pct': 0.0005,
                                  'slippage_pct': 0.0002, 'pct_oos': 0.3})
    assert 'CUENTA' in cod and 'IS 70' in cod and 'SETUP S0' in cod


def test_defaults_setup_cruce_sin_stop():
    assert defaults_setup('Cruce de medias') == {'stop_atr': 0.0}
    assert defaults_setup('RSI') == {}


def test_analizar_drawdowns_episodio_recuperado_y_pendiente():
    # episodio 1: pico 110 (idx1) -> fondo 90 (idx2) -> recupera en idx5 (120)
    # episodio 2: pico 120 (idx5) -> fondo 80 (idx6) -> nunca recupera
    equity = np.array([100., 110., 90., 95., 105., 120., 80.])
    episodios = _analizar_drawdowns(equity)
    assert len(episodios) == 2
    prof1, dur1 = episodios[0]
    assert prof1 == pytest.approx(-18.1818, abs=0.01)
    assert dur1 == 4   # velas desde el pico (idx1) hasta la recuperacion (idx5)
    prof2, dur2 = episodios[1]
    assert prof2 == pytest.approx(-33.3333, abs=0.01)
    assert dur2 is None   # no se recupero dentro de la serie


def test_metricas_robustez_con_trades_sinteticos():
    # equity plana: aisla las metricas basadas en trades (r2/dd/sharpe dan
    # None con std=0, asi el test se centra en SQN/payoff/%mejor/costes)
    equity = np.full(20, 10000.0)
    trades = {
        'idx_entrada': np.array([2, 5, 8, 11]),
        'idx_salida': np.array([3, 6, 9, 12]),
        'pnl': np.array([100.0, -50.0, 200.0, -50.0]),
        'ret_pct': np.array([0.01, -0.005, 0.02, -0.005]),
        'r_multiple': np.array([1.0, -1.0, 2.0, -1.0]),
        'notional_redondo': np.array([20000.0] * 4),
        'costo_comision': np.array([10.0] * 4),
    }
    resultado = {'equity': equity, 'trades': trades,
                'drawdown': np.zeros(20), 'capital_final': 10000.0, 'n_trades': 4}
    m = calcular_metricas(resultado)

    r_mult = np.array([1.0, -1.0, 2.0, -1.0])
    sqn_esperado = np.sqrt(4) * r_mult.mean() / r_mult.std()
    assert m['sqn'] == pytest.approx(sqn_esperado)
    assert m['payoff_ratio'] == pytest.approx(150.0 / 50.0)   # ganancia media 150 / perdida media 50
    assert m['pct_mejor_trade'] == pytest.approx(100.0)        # 200 / pnl_total(200) * 100
    assert m['slippage_minimo_pct'] == pytest.approx(50.0 / 20000.0 * 100)
    assert m['impacto_comisiones_pct'] == pytest.approx(40.0 / 240.0 * 100)


def test_impacto_comisiones_none_si_el_tramo_esta_en_perdidas():
    # pnl_total negativo (-40) pero pnl_total + comision (40) da positivo:
    # sin el guard de pnl_total>0 el ratio se dispara a un valor absurdo
    # (40/0... aqui 100%) en vez de reconocer que no hay "ganancia bruta"
    # real de la que hablar
    equity = np.full(10, 10000.0)
    trades = {
        'idx_entrada': np.array([2, 5]),
        'idx_salida': np.array([3, 6]),
        'pnl': np.array([-100.0, 60.0]),
        'ret_pct': np.array([-0.01, 0.006]),
        'r_multiple': np.array([-1.0, 0.6]),
        'notional_redondo': np.array([20000.0, 20000.0]),
        'costo_comision': np.array([20.0, 20.0]),
    }
    resultado = {'equity': equity, 'trades': trades,
                'drawdown': np.zeros(10), 'capital_final': 9960.0, 'n_trades': 2}
    m = calcular_metricas(resultado)
    assert m['pnl_total'] == pytest.approx(-40.0)
    assert m['impacto_comisiones_pct'] is None


def test_slippage_e_impacto_comisiones_consistentes_con_el_motor():
    # sin stop: el conjunto de trades no cambia al variar el slippage (con
    # stop, un pequeño desplazamiento de precio puede hacer que un trade
    # cruce el umbral del stop en otra vela, cambiando radicalmente ESE
    # trade — rompe la aproximación lineal que se quiere validar aquí)
    closes = np.concatenate([np.linspace(100, 90, 40), np.linspace(90, 140, 80)])
    df = _df_sintetico(closes)
    s = generar_senales('Cruce de medias', df,
                        {'tipo': 'SMA', 'rapida': 5, 'lenta': 15, 'direccion': 'Long'})
    cfg = dict(CONFIG_BASE, stop_atr=0.0, comision_pct=0.0005, slippage_pct=0.0002)
    r0 = simular(df['open'].values, df['high'].values, df['low'].values,
                df['close'].values, s, cfg)
    m0 = calcular_metricas(r0)
    assert m0['n_trades'] > 0 and m0['slippage_minimo_pct'] is not None

    # aplicar el slippage minimo calculado encima del ya configurado: la
    # expectancy resultante debe acercarse a 0 (aproximacion lineal, con
    # tolerancia) y nunca alejarse mas del punto de partida
    delta = max(m0['slippage_minimo_pct'], 0.0) / 100.0
    cfg2 = dict(cfg, slippage_pct=cfg['slippage_pct'] + delta)
    r1 = simular(df['open'].values, df['high'].values, df['low'].values,
                df['close'].values, s, cfg2)
    m1 = calcular_metricas(r1)
    assert abs(m1['expectancy_pct']) <= abs(m0['expectancy_pct'])
    assert abs(m1['expectancy_pct']) < 0.15   # cerca de cero, tolerancia de la aprox. lineal

    # impacto de comisiones: monotono con la comision configurada
    cfg_baja = dict(cfg, comision_pct=0.0001)
    cfg_alta = dict(cfg, comision_pct=0.005)
    m_baja = calcular_metricas(simular(df['open'].values, df['high'].values,
                                       df['low'].values, df['close'].values, s, cfg_baja))
    m_alta = calcular_metricas(simular(df['open'].values, df['high'].values,
                                       df['low'].values, df['close'].values, s, cfg_alta))
    assert m_baja['impacto_comisiones_pct'] is not None
    assert m_alta['impacto_comisiones_pct'] is not None
    assert m_alta['impacto_comisiones_pct'] > m_baja['impacto_comisiones_pct']


def test_backtest_completo_con_estrategia():
    # integración: cruce de medias sobre serie con tendencia limpia
    closes = np.concatenate([np.linspace(100, 90, 40), np.linspace(90, 140, 80)])
    df = _df_sintetico(closes)
    s = generar_senales('Cruce de medias', df,
                        {'tipo': 'SMA', 'rapida': 5, 'lenta': 15, 'direccion': 'Long'})
    r = simular(df['open'].values, df['high'].values, df['low'].values,
                df['close'].values, s, CONFIG_BASE)
    assert r['n_trades'] >= 1
    m = calcular_metricas(r)
    assert m['pnl_total'] > 0   # tendencia alcista clara: el cruce long gana
    assert np.isfinite(r['equity']).all()
    assert (r['drawdown'] <= 0).all()
