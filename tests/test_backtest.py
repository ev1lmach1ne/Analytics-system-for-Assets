import numpy as np
import pandas as pd
import pytest

from core.backtest import (
    simular, dividir_is_oos, calcular_metricas, walk_forward, montecarlo,
    resultado_filtrado, _analizar_drawdowns,
)
from core.strategies import (
    generar_senales, generar_senales_sistema, describir, ESTRATEGIAS,
    params_por_defecto, sma, codigo_setup, codigo_sistema, defaults_setup,
    etapa_salida_por_defecto, filas_plantilla, trigger_etapa,
    tramo_entrada_por_defecto, trigger_tramo,
    salida_mecanismo_por_defecto, validar_parciales, validar_tramos,
    validar_setup, AVISO_EXCESO_PARCIALES,
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


def test_trade_short_pnl_exacto():
    # señal en t=2 -> entra corto al open de t=3 (100); señal de salida en
    # t=5 -> sale al open de t=6 (90, precio bajó -> ganancia en corto).
    # unidades = 10000*0.01/4 = 25 -> pnl = 25*(100-90) = 250.
    n = 10
    o, h, l, c = _ohlc_plano(n)
    o[6] = 90.0
    s = _senales_vacias(n)
    s['entradas_short'][2] = True
    s['salidas_short'][5] = True
    r = simular(o, h, l, c, s, CONFIG_BASE)
    assert r['n_trades'] == 1
    t = r['trades']
    assert t['dir'][0] == -1
    assert t['pnl'][0] == pytest.approx(250.0)
    assert r['capital_final'] == pytest.approx(10250.0)


def test_mfe_mae_long():
    # entra t=3 a 100, sale t=6 al open (104), unidades=25, riesgo_absoluto=100,
    # con una excursión intermedia en t=4: sube hasta 108 (favorable) y baja
    # hasta 95 (adverso) antes de que la señal de salida se ejecute en t=6.
    # mfe = 108-100 = 8 -> mfe_r = 8*25/100 = 2.0
    # mae = 100-95 = 5  -> mae_r = 5*25/100 = 1.25
    # La vela 6 se define coherente (open dentro de su [low, high]): el trade
    # sale en su open, así que su rango ya no cuenta para el MFE y solo entra
    # el propio precio de salida (104), por debajo del máximo intermedio.
    n = 10
    o, h, l, c = _ohlc_plano(n)
    o[6], h[6], l[6], c[6] = 104.0, 104.5, 103.5, 104.0
    h[4] = 108.0
    l[4] = 95.0
    s = _senales_vacias(n)
    s['entradas_long'][2] = True
    s['salidas_long'][5] = True
    r = simular(o, h, l, c, s, CONFIG_BASE)
    t = r['trades']
    assert t['mfe_r'][0] == pytest.approx(2.0)
    assert t['mae_r'][0] == pytest.approx(1.25)


def test_mfe_mae_short():
    # entra corto t=3 a 100, sale t=6 al open (96), con excursión intermedia
    # en t=4: sube hasta 106 (adverso para un corto) y baja hasta 93 (favorable).
    # mfe = 100-93 = 7 -> mfe_r = 7*25/100 = 1.75
    # mae = 106-100 = 6 -> mae_r = 6*25/100 = 1.5
    # Vela 6 coherente y con el open (96) por encima del mínimo intermedio (93),
    # para que el MFE lo siga marcando la excursión de t=4 y no la salida.
    n = 10
    o, h, l, c = _ohlc_plano(n)
    o[6], h[6], l[6], c[6] = 96.0, 96.5, 95.5, 96.0
    h[4] = 106.0
    l[4] = 93.0
    s = _senales_vacias(n)
    s['entradas_short'][2] = True
    s['salidas_short'][5] = True
    r = simular(o, h, l, c, s, CONFIG_BASE)
    t = r['trades']
    assert t['mfe_r'][0] == pytest.approx(1.75)
    assert t['mae_r'][0] == pytest.approx(1.5)


def test_mfe_ignora_el_rango_posterior_a_una_salida_al_open():
    # El trade sale al open de t=6 (100): en ese instante deja de existir, así
    # que el rally posterior de esa misma vela hasta 130 NO es beneficio que
    # "dejara en la mesa". Contar la vela de salida entera inflaba el ETD con
    # una ganancia que nunca estuvo disponible.
    n = 10
    o, h, l, c = _ohlc_plano(n)
    o[6], h[6], l[6], c[6] = 100.0, 130.0, 99.0, 129.0
    s = _senales_vacias(n)
    s['entradas_long'][2] = True
    s['salidas_long'][5] = True
    t = simular(o, h, l, c, s, CONFIG_BASE)['trades']
    assert t['idx_salida'][0] == 6
    assert t['motivo'][0] == 0
    # solo las velas 3-5 (planas: máximo 101) más el precio de salida (100)
    assert t['mfe_r'][0] == pytest.approx(0.25)     # (101-100)*25/100
    assert t['etd_r'][0] == pytest.approx(0.25)     # r_multiple = 0


def test_eficiencia_salida_100_cuando_el_tp_se_clava():
    # Salir en el TP exacto es la ejecución perfecta: el trade capturó todo lo
    # que su propia salida permitía. Que la vela desborde el objetivo (hasta
    # 120) es irrelevante — la posición ya estaba cerrada a 108.
    n = 12
    o, h, l, c = _ohlc_plano(n)
    h[5], c[5] = 120.0, 119.0
    s = _senales_vacias(n)
    s['entradas_long'][2] = True
    r = simular(o, h, l, c, s, dict(CONFIG_BASE, stop_atr=2.0, tp_r=2.0))
    t = r['trades']
    assert t['motivo'][0] == 2                      # TP
    assert t['precio_salida'][0] == pytest.approx(108.0)
    assert t['mfe_r'][0] == pytest.approx(2.0)
    assert t['etd_r'][0] == pytest.approx(0.0)
    assert t['eficiencia_salida'][0] == pytest.approx(100.0)


def test_etd_nunca_negativo_en_un_backtest_real():
    # Invariante: el MFE es por definición >= el resultado final, así que el
    # "beneficio dejado en la mesa" no puede ser negativo. Se rompía cuando la
    # ventana del MFE no cubría el precio al que el trade salió de verdad.
    rng = np.random.default_rng(11)
    closes = 100 + np.cumsum(rng.normal(0, 1.0, 400))
    df = _df_sintetico(closes)
    s = generar_senales_sistema(df, [{'plantilla': 'Cruce de medias',
                                      'params': {'tipo': 'SMA', 'rapida': 5,
                                                 'lenta': 20, 'direccion': 'Ambas'}}])
    o, h, l, c = (df['open'].values, df['high'].values,
                  df['low'].values, df['close'].values)
    t = simular(o, h, l, c, s, dict(CONFIG_BASE, config_por_setup={
        0: {'stop_atr': 1.0, 'tp_r': 3.0}}))['trades']
    assert len(t['etd_r']) > 5
    assert (t['etd_r'] >= -1e-9).all()
    assert (t['mfe_r'] >= -1e-9).all() and (t['mae_r'] >= -1e-9).all()
    for clave in ('eficiencia_entrada', 'eficiencia_salida'):
        v = t[clave][~np.isnan(t[clave])]
        assert len(v) > 5
        assert ((v >= 0.0) & (v <= 100.0)).all(), clave


def test_eficiencia_salida_acotada_en_un_perdedor_con_mfe_minusculo():
    # El caso que rompía la métrica: el precio avanza 0.2 a favor (0.1 R) y
    # luego se para en el stop. Normalizar por el MFE daba r/mfe = -1/0.1 =
    # -1000%, y ese denominador diminuto arrastraba la media de la tarjeta a
    # cientos de puntos negativos. Normalizando por el rango completo el trade
    # queda en 0%: se cerró en el peor precio de su recorrido, que es la
    # lectura correcta y comparable con el resto.
    n = 10
    o = np.full(n, 100.0)
    h = np.full(n, 100.1)
    l = np.full(n, 99.9)
    c = np.full(n, 100.0)
    h[4] = 100.2          # excursión favorable mínima
    l[5] = 97.0           # toca el stop (98)
    s = _senales_vacias(n)
    s['entradas_long'][2] = True
    t = simular(o, h, l, c, s, dict(CONFIG_BASE, stop_atr=1.0))['trades']
    assert t['motivo'][0] == 1                          # stop
    assert t['precio_salida'][0] == pytest.approx(98.0)
    assert t['mfe_r'][0] == pytest.approx(0.1)          # el MFE minúsculo
    assert t['r_multiple'][0] == pytest.approx(-1.0)    # r/mfe habría dado -1000%
    assert t['eficiencia_salida'][0] == pytest.approx(0.0)
    assert t['eficiencia_entrada'][0] == pytest.approx(0.2 / 2.2 * 100.0)


def test_eficiencias_cumplen_la_identidad_de_sweeney():
    # entrada% + salida% - 100 = eficiencia total (el % del rango que el trade
    # se llevó). Que la identidad se cumpla confirma que ambas comparten el
    # mismo denominador —el rango recorrido— y por eso son comparables entre sí.
    rng = np.random.default_rng(3)
    closes = 100 + np.cumsum(rng.normal(0, 1.0, 400))
    df = _df_sintetico(closes)
    s = generar_senales_sistema(df, [{'plantilla': 'Cruce de medias',
                                      'params': {'tipo': 'SMA', 'rapida': 5,
                                                 'lenta': 20, 'direccion': 'Ambas'}}])
    o, h, l, c = (df['open'].values, df['high'].values,
                  df['low'].values, df['close'].values)
    t = simular(o, h, l, c, s, dict(CONFIG_BASE, config_por_setup={
        0: {'stop_atr': 1.0, 'tp_r': 3.0}}))['trades']
    bruto = (t['precio_salida'] - t['precio_entrada']) * t['dir']
    rango = (t['mfe_r'] + t['mae_r'])
    escala = np.where(rango > 0, (t['mfe_r'] + t['mae_r']), np.nan)
    total = bruto * t['unidades'] / (t['pnl'] / t['r_multiple']) / escala * 100.0
    assert np.nanmax(np.abs(
        t['eficiencia_entrada'] + t['eficiencia_salida'] - 100.0 - total)) < 1e-9


def test_calcular_metricas_admite_trades_sin_columnas_derivadas():
    # calcular_metricas se usa también con dicts de trades armados a mano; las
    # columnas derivadas (etd_r, eficiencias) son opcionales y su ausencia no
    # debe tumbar el resto de métricas.
    equity = np.full(20, 10000.0)
    trades = {
        'idx_entrada': np.array([2, 5]), 'idx_salida': np.array([3, 6]),
        'pnl': np.array([100.0, -50.0]), 'ret_pct': np.array([0.01, -0.005]),
        'r_multiple': np.array([1.0, -1.0]),
        'notional_redondo': np.array([20000.0] * 2),
        'costo_comision': np.array([10.0] * 2),
    }
    m = calcular_metricas({'equity': equity, 'trades': trades,
                           'drawdown': np.zeros(20), 'capital_final': 10000.0,
                           'n_trades': 2})
    assert m['win_rate'] == pytest.approx(0.5)
    assert m['etd_r_medio'] is None
    assert m['eficiencia_entrada_media'] is None
    assert m['eficiencia_salida_media'] is None


def test_ulcer_index_castiga_la_duracion_del_drawdown():
    # Peter Martin: RMS del drawdown punto a punto. A igual profundidad (-10%),
    # el drawdown que tarda más en recuperarse puntúa más alto — eso es lo que
    # lo distingue del max drawdown, que daría lo mismo en ambos casos.
    def _ui(eq):
        return calcular_metricas({'equity': np.array(eq, dtype=float),
                                  'trades': {k: np.array([]) for k in
                                             ('idx_entrada', 'idx_salida', 'pnl',
                                              'ret_pct', 'r_multiple')},
                                  'drawdown': np.zeros(len(eq)),
                                  'capital_final': float(eq[-1]), 'n_trades': 0})

    breve = _ui([100.0, 90.0, 100.0, 100.0])
    largo = _ui([100.0, 90.0, 90.0, 100.0])
    # dd = [0, -0.1, 0, 0] -> sqrt(0.01/4) = 0.05 -> 5%
    assert breve['ulcer_index'] == pytest.approx(5.0)
    # dd = [0, -0.1, -0.1, 0] -> sqrt(0.02/4) = 0.0707 -> 7.07%
    assert largo['ulcer_index'] == pytest.approx(7.0710678, abs=1e-5)
    assert breve['max_dd_pct'] == pytest.approx(largo['max_dd_pct'])   # mismo -10%


def test_comision_en_corto_cobrada_una_vez_por_ambos_lados():
    n = 10
    o, h, l, c = _ohlc_plano(n)
    o[6] = 90.0
    s = _senales_vacias(n)
    s['entradas_short'][2] = True
    s['salidas_short'][5] = True
    cfg = dict(CONFIG_BASE, comision_pct=0.001)
    r = simular(o, h, l, c, s, cfg)
    # comision = (100+90)*25*0.001 = 4.75 (entrada+salida, una sola vez)
    assert r['trades']['pnl'][0] == pytest.approx(250.0 - 4.75)


def test_slippage_precios_de_llenado_exactos_largo_y_corto():
    # Verifica la DIRECCIÓN del slippage explícitamente sobre el precio de
    # llenado (no solo sobre el PnL agregado): en largo, la entrada debe
    # llenar más CARO y la salida más BARATO (peor en ambos lados); en
    # corto, al revés — entrada más BARATA, salida más CARA.
    n = 10
    o, h, l, c = _ohlc_plano(n)
    o[6] = 110.0
    s = _senales_vacias(n)
    s['entradas_long'][2] = True
    s['salidas_long'][5] = True
    cfg = dict(CONFIG_BASE, slippage_pct=0.01)
    r = simular(o, h, l, c, s, cfg)
    t = r['trades']
    assert t['precio_entrada'][0] == pytest.approx(101.0)   # 100*(1+0.01)
    assert t['precio_salida'][0] == pytest.approx(108.9)    # 110*(1-0.01)

    o2, h2, l2, c2 = _ohlc_plano(n)
    o2[6] = 90.0
    s2 = _senales_vacias(n)
    s2['entradas_short'][2] = True
    s2['salidas_short'][5] = True
    r2 = simular(o2, h2, l2, c2, s2, cfg)
    t2 = r2['trades']
    assert t2['precio_entrada'][0] == pytest.approx(99.0)   # 100*(1-0.01)
    assert t2['precio_salida'][0] == pytest.approx(90.9)    # 90*(1+0.01)


def test_riesgo_realizado_supera_el_riesgo_nominal_por_las_friccion():
    # El tamaño de posición se calcula como cap*riesgo_pct/dist (sin
    # descontar comisión ni slippage), así que la pérdida REAL de un stop
    # con fricciones activas es *mayor* que el riesgo_pct nominal
    # configurado — matiz de la auditoría, no un bug: el riesgo nominal es
    # solo el punto de partida del sizing, no un techo garantizado.
    n = 10
    o, h, l, c = _ohlc_plano(n)
    l[5] = 97.0   # dispara el stop (ver cálculo de stop_precio abajo)
    s = _senales_vacias(n)
    s['entradas_long'][2] = True
    cfg = dict(CONFIG_BASE, stop_atr=1.5, comision_pct=0.001, slippage_pct=0.005)
    r = simular(o, h, l, c, s, cfg)
    t = r['trades']
    assert r['n_trades'] == 1
    assert t['motivo'][0] == 1   # stop
    # entrada con slippage: 100*(1+0.005) = 100.5; dist = 1.5*ATR(2) = 3
    # stop_precio = 100.5 - 3 = 97.5 (l[5]=97 <= 97.5 -> dispara)
    # salida con slippage sobre el stop: 97.5*(1-0.005) = 97.0125
    # unidades = 10000*0.01/3 = 33.3333...
    # pnl = (97.0125-100.5)*33.3333 - (100.5+97.0125)*33.3333*0.001
    riesgo_nominal = 10000 * 0.01
    assert t['pnl'][0] == pytest.approx(-122.83375)
    assert abs(t['pnl'][0]) > riesgo_nominal


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


def test_exposicion_cuenta_las_velas_con_posicion_abierta():
    # entra al open de t=3, sale al open de t=6 -> velas 3,4,5,6 con posición
    # abierta en algún momento = 4 de 20 = 20%. Inclusive en ambos extremos:
    # la vela de salida tuvo posición hasta que se ejecutó el cierre.
    n = 20
    o, h, l, c = _ohlc_plano(n)
    o[6] = 110.0
    s = _senales_vacias(n)
    s['entradas_long'][2] = True
    s['salidas_long'][5] = True
    r = simular(o, h, l, c, s, CONFIG_BASE)
    assert r['trades']['idx_entrada'][0] == 3 and r['trades']['idx_salida'][0] == 6
    assert calcular_metricas(r)['exposicion_pct'] == pytest.approx(4 / 20 * 100)


def test_exposicion_no_cuenta_dos_veces_las_salidas_parciales():
    # Una sola entrada cerrada en dos parciales genera DOS filas en trades que
    # comparten idx_entrada (3) y salen en t=6 y t=9. Sumar duraciones daría
    # (6-3) + (9-3) = 9 velas sobre 12; lo correcto es la unión 3..9 = 7.
    n = 12
    o, h, l, c = _ohlc_plano(n)
    o[6] = 110.0
    o[9] = 120.0
    c[9] = 120.0
    s = _senales_vacias(n)
    s['entradas_long'][2] = True
    s['salidas_long'][5] = True
    r = simular(o, h, l, c, s, _cfg_parciales(
        [{'pct': 50.0, 'r': 0.0, 'trigger': 'senal', 'condiciones': []},
         {'pct': 50.0, 'r': 0.0, 'trigger': 'senal', 'condiciones': []}],
        salida_n_velas=6))
    t = r['trades']
    assert r['n_trades'] == 2
    assert t['idx_entrada'].tolist() == [3, 3]
    assert t['idx_salida'].tolist() == [6, 9]
    assert calcular_metricas(r)['exposicion_pct'] == pytest.approx(7 / 12 * 100)


def test_exposicion_coincide_con_la_instrumentacion_del_motor():
    # Comprobación cruzada: la unión de intervalos de trades tiene que
    # reproducir exactamente las velas que el motor marca como "en posición"
    # para dibujar la trayectoria del stop.
    rng = np.random.default_rng(11)
    closes = 100 + np.cumsum(rng.normal(0, 1.0, 400))
    df = _df_sintetico(closes)
    s = generar_senales_sistema(df, [{'plantilla': 'Cruce de medias',
                                      'params': {'tipo': 'SMA', 'rapida': 5,
                                                 'lenta': 20, 'direccion': 'Ambas'}}])
    o, h, l, c = (df['open'].values, df['high'].values,
                  df['low'].values, df['close'].values)
    r = simular(o, h, l, c, s, dict(CONFIG_BASE, config_por_setup={
        0: {'stop_atr': 1.0, 'tp_r': 3.0}}))
    m = calcular_metricas(r)
    esperado = float(np.isfinite(r['entrada_track']).mean()) * 100.0
    assert m['exposicion_pct'] == pytest.approx(esperado)
    assert 0.0 < m['exposicion_pct'] <= 100.0


def test_exposicion_por_tramo_recorta_el_trade_a_caballo():
    # Un trade que entra en IS y sale en OOS ocupa tiempo real en los dos
    # tramos, así que su intervalo se recorta a la ventana en vez de asignarse
    # entero al tramo de su vela de entrada (que es como se cuentan los trades).
    n = 20
    o, h, l, c = _ohlc_plano(n)
    o[14] = 110.0
    s = _senales_vacias(n)
    s['entradas_long'][7] = True     # entra en t=8 (IS)
    s['salidas_long'][13] = True     # sale en t=14 (OOS)
    r = simular(o, h, l, c, s, CONFIG_BASE)
    corte = 10
    assert r['trades']['idx_entrada'][0] == 8 and r['trades']['idx_salida'][0] == 14
    m_is = calcular_metricas(r, 0, corte)
    m_oos = calcular_metricas(r, corte, n)
    assert m_is['n_trades'] == 1 and m_oos['n_trades'] == 0   # se cuenta en IS
    assert m_is['exposicion_pct'] == pytest.approx(2 / 10 * 100)    # velas 8,9
    assert m_oos['exposicion_pct'] == pytest.approx(5 / 10 * 100)   # velas 10..14


def test_retorno_ajustado_por_exposicion():
    n = 20
    o, h, l, c = _ohlc_plano(n)
    o[6] = 110.0
    s = _senales_vacias(n)
    s['entradas_long'][2] = True
    s['salidas_long'][5] = True
    r = simular(o, h, l, c, s, CONFIG_BASE)
    m = calcular_metricas(r, velas_por_anio=252)
    assert m['retorno_ajustado_exposicion_pct'] == pytest.approx(
        m['retorno_anual_pct'] / m['exposicion_pct'] * 100.0)
    # sin anualizar no hay retorno anual que ajustar
    assert calcular_metricas(r)['retorno_ajustado_exposicion_pct'] is None
    # sin trades no hay exposición: ni ratio ni división por cero
    vacio = simular(o, h, l, c, _senales_vacias(n), CONFIG_BASE)
    m_vacio = calcular_metricas(vacio, velas_por_anio=252)
    assert m_vacio['exposicion_pct'] == 0.0
    assert m_vacio['retorno_ajustado_exposicion_pct'] is None


def test_capital_comprometido_cae_a_la_mitad_tras_una_parcial():
    # Lo que la exposición en tiempo no puede distinguir: tras cerrar el 50%
    # sigues "dentro" las mismas velas, pero con la mitad del capital en juego.
    n = 12
    o, h, l, c = _ohlc_plano(n)
    s = _senales_vacias(n)
    s['entradas_long'][2] = True
    s['salidas_long'][5] = True
    r = simular(o, h, l, c, s, _cfg_parciales(
        [{'pct': 50.0, 'r': 0.0, 'trigger': 'senal', 'condiciones': []},
         {'pct': 50.0, 'r': 0.0, 'trigger': 'senal', 'condiciones': []}],
        salida_n_velas=6))
    et = r['exposicion_track']
    assert et[1] == 0.0                                  # antes de entrar
    assert et[5] > 0.0
    assert et[6] == pytest.approx(et[5] / 2, rel=1e-3)   # parcial del 50% en t=6
    assert et[10] == 0.0                                 # ya cerrada del todo
    m = calcular_metricas(r)
    assert 0.0 < m['exposicion_capital_pct'] < m['exposicion_pct']


def test_metricas_de_exposicion_con_trades_armados_a_mano():
    # Sin 'exposicion_track' (dicts construidos fuera de simular) la métrica de
    # capital queda en None, pero la de tiempo se calcula igual: solo necesita
    # los índices de entrada y salida.
    equity = np.full(20, 10000.0)
    trades = {
        'idx_entrada': np.array([2, 10]), 'idx_salida': np.array([5, 12]),
        'pnl': np.array([100.0, -50.0]), 'ret_pct': np.array([0.01, -0.005]),
        'r_multiple': np.array([1.0, -1.0]),
        'notional_redondo': np.array([20000.0] * 2),
        'costo_comision': np.array([10.0] * 2),
    }
    m = calcular_metricas({'equity': equity, 'trades': trades,
                           'drawdown': np.zeros(20), 'capital_final': 10000.0,
                           'n_trades': 2})
    assert m['exposicion_capital_pct'] is None
    # velas 2..5 (4) + 10..12 (3) = 7 de 20
    assert m['exposicion_pct'] == pytest.approx(7 / 20 * 100)


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


# ── salidas parciales ──────────────────────────────────────────────────────

def _cfg_parciales(etapas, **extra):
    return dict(CONFIG_BASE, config_por_setup={0: dict({'parciales': etapas}, **extra)})


def test_etapa_senal_al_100_equivale_a_no_tener_etapas():
    # La etapa por defecto del constructor (100% a la señal) debe dar
    # exactamente el mismo resultado que un setup sin etapas: es la garantía
    # de no-regresión para todos los sistemas ya guardados.
    n = 10
    o, h, l, c = _ohlc_plano(n)
    o[6] = 110.0
    s = _senales_vacias(n)
    s['entradas_long'][2] = True
    s['salidas_long'][5] = True

    base = simular(o, h, l, c, s, CONFIG_BASE)
    etapa = simular(o, h, l, c, s, _cfg_parciales(
        [{'pct': 100.0, 'r': 0.0, 'trigger': 'senal', 'condiciones': []}]))

    assert etapa['n_trades'] == base['n_trades'] == 1
    for clave in ('idx_entrada', 'idx_salida', 'pnl', 'motivo', 'parcial',
                  'unidades'):
        assert etapa['trades'][clave].tolist() == base['trades'][clave].tolist()
    assert etapa['capital_final'] == pytest.approx(base['capital_final'])
    assert etapa['equity'].tolist() == base['equity'].tolist()
    assert etapa['trades']['motivo'][0] == 0     # 'Señal', no 'Parcial'


def test_etapa_senal_parcial_deja_viva_la_posicion():
    # 50% a la señal + 50% al cierre por tiempo. unidades = 10000*0.01/4 = 25.
    # Señal de salida en t=5 -> parcial al open de t=6 (110): cierra 12.5 uds
    # -> pnl 12.5*10 = 125. El resto sigue abierto y sale por tiempo.
    n = 12
    o, h, l, c = _ohlc_plano(n)
    o[6] = 110.0
    o[9] = 120.0
    c[9] = 120.0
    s = _senales_vacias(n)
    s['entradas_long'][2] = True
    s['salidas_long'][5] = True
    r = simular(o, h, l, c, s, _cfg_parciales(
        [{'pct': 50.0, 'r': 0.0, 'trigger': 'senal', 'condiciones': []},
         {'pct': 50.0, 'r': 0.0, 'trigger': 'senal', 'condiciones': []}],
        salida_n_velas=6))
    t = r['trades']
    assert r['n_trades'] == 2
    assert t['motivo'][0] == 5 and t['parcial'][0] == 1
    assert t['idx_salida'][0] == 6
    assert t['unidades'][0] == pytest.approx(12.5)
    assert t['pnl'][0] == pytest.approx(125.0)
    # el resto cierra por tiempo (entró en t=3, +6 velas -> t=9) al close 120
    assert t['motivo'][1] == 3 and t['parcial'][1] == 0
    assert t['unidades'][1] == pytest.approx(12.5)
    assert t['pnl'][1] == pytest.approx(250.0)


def test_etapa_senal_parcial_mas_etapa_por_rr():
    # El caso del constructor: 50% por señal contraria, 50% a 2R.
    # stop 1xATR (ATR=2) -> dist = 2, 2R = +4 sobre 100 = 104.
    n = 14
    o, h, l, c = _ohlc_plano(n)
    o[6] = 101.0
    h[8] = 105.0          # toca 2R en t=8
    s = _senales_vacias(n)
    s['entradas_long'][2] = True
    s['salidas_long'][5] = True
    r = simular(o, h, l, c, s, _cfg_parciales(
        [{'pct': 50.0, 'r': 0.0, 'trigger': 'senal', 'condiciones': []},
         {'pct': 50.0, 'r': 2.0, 'trigger': 'r', 'condiciones': []}],
        stop_atr=1.0))
    t = r['trades']
    assert r['n_trades'] == 2
    assert t['motivo'].tolist() == [5, 5]
    assert t['parcial'].tolist() == [1, 2]
    assert t['idx_salida'][0] == 6            # parcial por señal, al open
    assert t['precio_salida'][0] == pytest.approx(101.0)
    assert t['idx_salida'][1] == 8            # parcial por R:R, intra-vela
    assert t['precio_salida'][1] == pytest.approx(104.0)


def test_etapa_senal_sin_trigger_explicito_dispara_igual():
    # Formato antiguo: r=0 y sin condiciones era una etapa muerta que nunca
    # se ejecutaba, pese a que el pseudocódigo la rotulaba "a la señal".
    n = 12
    o, h, l, c = _ohlc_plano(n)
    o[6] = 110.0
    s = _senales_vacias(n)
    s['entradas_long'][2] = True
    s['salidas_long'][5] = True
    r = simular(o, h, l, c, s, _cfg_parciales(
        [{'pct': 40.0, 'r': 0.0}, {'pct': 60.0, 'r': 0.0}]))
    assert r['trades']['motivo'][0] == 5
    assert r['trades']['unidades'][0] == pytest.approx(10.0)   # 40% de 25


def test_etapa_senal_no_impide_revertir_cuando_cierra_todo():
    # Señal contraria del mismo setup: si la etapa cierra el 100%, la
    # posición debe revertirse igual que sin etapas.
    n = 14
    o, h, l, c = _ohlc_plano(n)
    s = _senales_vacias(n)
    s['entradas_long'][2] = True
    s['entradas_short'][5] = True
    r = simular(o, h, l, c, s, _cfg_parciales(
        [{'pct': 100.0, 'r': 0.0, 'trigger': 'senal', 'condiciones': []}]))
    t = r['trades']
    assert r['n_trades'] >= 2
    assert t['dir'][0] == 1 and t['motivo'][0] == 0
    assert t['dir'][1] == -1              # revirtió a corto


def test_etapa_por_defecto_no_cambia_un_backtest_real():
    # Un sistema guardado antes de esta versión (sin 'parciales') y el mismo
    # sistema abierto ahora en el constructor (con la etapa 100% a la señal
    # que se le inyecta) deben dar exactamente el mismo backtest.
    rng = np.random.default_rng(7)
    closes = 100 + np.cumsum(rng.normal(0, 1.0, 400))
    df = _df_sintetico(closes)
    setup = {'plantilla': 'Cruce de medias',
             'params': {'tipo': 'SMA', 'rapida': 5, 'lenta': 20,
                        'direccion': 'Ambas'}}
    s = generar_senales_sistema(df, [setup])
    o, h, l, c = (df['open'].values, df['high'].values,
                  df['low'].values, df['close'].values)

    antiguo = simular(o, h, l, c, s, dict(
        CONFIG_BASE, config_por_setup={0: {'stop_atr': 1.0}}))
    nuevo = simular(o, h, l, c, s, dict(CONFIG_BASE, config_por_setup={
        0: {'stop_atr': 1.0, 'parciales': [etapa_salida_por_defecto()]}}))

    assert antiguo['n_trades'] > 5            # el escenario ejercita el motor
    assert nuevo['n_trades'] == antiguo['n_trades']
    assert nuevo['capital_final'] == pytest.approx(antiguo['capital_final'])
    for clave, arr in antiguo['trades'].items():
        assert nuevo['trades'][clave].tolist() == arr.tolist(), clave


def test_filas_plantilla_editables_solo_donde_la_tabla_puede():
    filas = filas_plantilla('Cruce de medias',
                            {'tipo': 'SMA', 'rapida': 20, 'lenta': 50,
                             'direccion': 'Ambas'})
    assert [f['direccion'] for f in filas] == ['long', 'short']
    assert filas[0]['texto'] == 'SMA(20) cruza arriba SMA(50)'
    assert filas[0]['mapeo']['izq.periodo'] == 'rapida'
    assert filas[0]['mapeo']['der.periodo'] == 'lenta'
    # la salida es el cruce contrario
    sal = filas_plantilla('Cruce de medias',
                          {'tipo': 'SMA', 'rapida': 20, 'lenta': 50,
                           'direccion': 'Ambas'}, salida=True)
    assert sal[0]['texto'] == 'SMA(20) cruza abajo SMA(50)'

    # una sola dirección -> una sola fila
    solo_long = filas_plantilla('Cruce de medias', {'direccion': 'Long'})
    assert len(solo_long) == 1 and solo_long[0]['direccion'] == 'long'

    # CCI no es representable como fila de indicador: solo texto
    cci = filas_plantilla('CCI')
    assert all(f['mapeo'] == {} and f['izq'] is None for f in cci)
    assert 'CCI(20)' in cci[0]['texto']


def test_filas_plantilla_cubren_todas_las_estrategias():
    for plantilla in ESTRATEGIAS:
        for salida in (False, True):
            for f in filas_plantilla(plantilla, salida=salida):
                assert f['texto']
                assert (f['izq'] is None) == (not f['mapeo'])


def test_trigger_etapa_deriva_el_formato_antiguo():
    assert trigger_etapa({'pct': 100, 'r': 0.0}) == 'senal'
    assert trigger_etapa({'pct': 50, 'r': 2.0}) == 'r'
    assert trigger_etapa({'pct': 50, 'r': 0.0, 'condiciones': [{'x': 1}]}) == 'cond'
    assert trigger_etapa({'pct': 50, 'r': 2.0, 'trigger': 'senal'}) == 'senal'


def test_codigo_setup_describe_cada_disparador():
    setup = {'nombre': 'x', 'plantilla': 'Cruce de medias',
             'params': params_por_defecto('Cruce de medias'),
             'riesgo_pct': 0.01, 'stop_atr': 1.0, 'tp_r': 0.0,
             'salida_n_velas': 0,
             'parciales': [dict(etapa_salida_por_defecto(), pct=50.0),
                           {'pct': 50.0, 'r': 2.0, 'trigger': 'r'}]}
    cod = codigo_setup(setup, 0)
    assert 'a la señal de salida de la plantilla' in cod
    assert 'cuando el precio ≥ +2 R' in cod


# ── entrada escalonada (tramos) ─────────────────────────────────────────────

def _cfg_tramos(tramos, parciales=None, **extra):
    cfg_s = dict({'tramos': tramos}, **extra)
    if parciales is not None:
        cfg_s['parciales'] = parciales
    return dict(CONFIG_BASE, config_por_setup={0: cfg_s})


def test_tramo_por_defecto_100_senal_equivale_a_no_tener_tramos():
    n = 10
    o, h, l, c = _ohlc_plano(n)
    o[6] = 110.0
    s = _senales_vacias(n)
    s['entradas_long'][2] = True
    s['salidas_long'][5] = True

    base = simular(o, h, l, c, s, CONFIG_BASE)
    con_tramo = simular(o, h, l, c, s, _cfg_tramos(
        [{'pct': 100.0, 'trigger': 'senal', 'val': 0.0, 'condiciones': [],
          'gestion': {'tipo': 0, 'val': 0.0}}]))

    assert con_tramo['n_trades'] == base['n_trades'] == 1
    for clave, arr in base['trades'].items():
        assert con_tramo['trades'][clave].tolist() == arr.tolist(), clave
    assert con_tramo['capital_final'] == pytest.approx(base['capital_final'])
    # una sola entrada registrada (tramo 0, la apertura)
    assert con_tramo['entradas']['tramo'].tolist() == [0]


def test_tramo_velas_promedia_precio_y_suma_unidades():
    # 50% a la señal + 50% a +2 velas de la 1ª entrada: mismo total que una
    # sola entrada al 100% (misma distancia de riesgo en ambos tramos), pero
    # el precio de entrada queda como la media ponderada de los dos rellenos.
    n = 12
    o, h, l, c = _ohlc_plano(n)
    o[6] = 110.0
    s = _senales_vacias(n)
    s['entradas_long'][2] = True
    s['salidas_long'][8] = True
    r = simular(o, h, l, c, s, _cfg_tramos(
        [{'pct': 50.0, 'trigger': 'senal'},
         {'pct': 50.0, 'trigger': 'velas', 'val': 2.0}]))
    ent = r['entradas']
    assert ent['tramo'].tolist() == [0, 1]
    assert ent['unidades'].tolist() == pytest.approx([12.5, 12.5])
    assert ent['precio'].tolist() == pytest.approx([100.0, 110.0])

    t = r['trades']
    assert r['n_trades'] == 1
    assert t['unidades'][0] == pytest.approx(25.0)             # 12.5 + 12.5
    assert t['precio_entrada'][0] == pytest.approx(105.0)      # media ponderada


def test_tramo_retroceso_dimensiona_cada_tramo_por_su_propia_distancia():
    # Promediar a la baja: el 2º tramo entra más cerca del stop que el 1º, así
    # que necesita MÁS unidades para arriesgar el mismo % — pero el riesgo
    # TOTAL (unidades × distancia al stop de cada tramo) debe sumar
    # exactamente el riesgo pretendido del setup (1% de 10000 = 100).
    n = 10
    o = np.full(n, 100.0)
    h = np.full(n, 101.0)
    l = np.full(n, 99.5)     # nunca toca el stop (98) ni el umbral (99) salvo donde se fuerza
    c = np.full(n, 100.0)
    l[5] = 99.0               # retroceso de 0.5×ATR desde 100 -> dispara el tramo 2
    o[6] = 99.0               # el tramo 2 entra a mejor precio (promedia a la baja)
    s = _senales_vacias(n)
    s['entradas_long'][2] = True
    s['salidas_long'][8] = True
    r = simular(o, h, l, c, s, _cfg_tramos(
        [{'pct': 50.0, 'trigger': 'senal'},
         {'pct': 50.0, 'trigger': 'retroceso', 'val': 0.5}],
        stop_atr=1.0))
    ent = r['entradas']
    assert ent['unidades'].tolist() == pytest.approx([25.0, 50.0])
    assert ent['precio'].tolist() == pytest.approx([100.0, 99.0])

    dist_ref, dist_tramo2 = 2.0, 1.0     # |100-98| y |99-98|
    riesgo_total = ent['unidades'][0] * dist_ref + ent['unidades'][1] * dist_tramo2
    assert riesgo_total == pytest.approx(100.0)     # exactamente 1% de 10000

    t = r['trades']
    assert t['unidades'][0] == pytest.approx(75.0)
    assert t['precio_entrada'][0] == pytest.approx((100.0 * 25.0 + 99.0 * 50.0) / 75.0)


def test_tramo_avance_piramide_con_be_autoajusta_el_stop():
    # Pirámide: el 2º tramo entra a favor (+2R) y su gestión mueve el stop al
    # nuevo precio medio (break-even del conjunto). El riesgo total sigue
    # siendo el 1% pretendido, y si el precio retrocede después, el stop-out
    # ocurre en el precio medio NUEVO, no en el stop original de la 1ª entrada
    # — la prueba de que el riesgo se autoajusta al piramidar.
    n = 12
    o = np.full(n, 100.0)
    h = np.full(n, 101.0)
    l = np.full(n, 100.6)    # por encima del stop original (98) Y del nuevo (100.5)
    c = np.full(n, 100.0)
    h[4] = 105.0             # +2R (100 + 2*2) -> dispara el tramo 2 (pirámide)
    o[5] = 104.0             # precio de relleno del tramo 2
    l[7] = 100.0             # rompe el stop YA MOVIDO a 100.5 (no el original 98)
    s = _senales_vacias(n)
    s['entradas_long'][1] = True
    r = simular(o, h, l, c, s, _cfg_tramos(
        [{'pct': 70.0, 'trigger': 'senal'},
         {'pct': 30.0, 'trigger': 'avance', 'val': 2.0,
          'gestion': {'tipo': 1, 'val': 0.01}}],
        stop_atr=1.0))
    ent = r['entradas']
    assert ent['unidades'].tolist() == pytest.approx([35.0, 5.0])
    assert ent['precio'].tolist() == pytest.approx([100.0, 104.0])

    riesgo_total = ent['unidades'][0] * 2.0 + ent['unidades'][1] * 6.0   # |104-98|
    assert riesgo_total == pytest.approx(100.0)     # 1% de 10000, ni un poco más

    t = r['trades']
    assert r['n_trades'] == 1
    precio_medio = (100.0 * 35.0 + 104.0 * 5.0) / 40.0
    assert t['precio_entrada'][0] == pytest.approx(precio_medio)
    assert t['motivo'][0] == 1                       # stop, no señal ni tiempo
    assert t['precio_stop'][0] == pytest.approx(precio_medio)   # BE al precio medio
    assert t['pnl'][0] == pytest.approx(0.0)          # break-even exacto


def test_tramo_clamp_evita_unidades_desbocadas_cerca_del_stop():
    # Si el tramo dispara pegado al stop, la distancia real (~0.05×ATR) se
    # acota a un mínimo del 25% de la distancia de la 1ª entrada — si no, las
    # unidades del tramo se dispararían (20× más en este caso).
    n = 10
    o = np.full(n, 100.0)
    h = np.full(n, 101.0)
    l = np.full(n, 99.5)
    c = np.full(n, 100.0)
    l[5] = 99.0
    o[6] = 98.05             # a solo 0.05 del stop (98)
    s = _senales_vacias(n)
    s['entradas_long'][2] = True
    r = simular(o, h, l, c, s, _cfg_tramos(
        [{'pct': 50.0, 'trigger': 'senal'},
         {'pct': 50.0, 'trigger': 'retroceso', 'val': 0.5}],
        stop_atr=1.0))
    ent = r['entradas']
    # sin clamp: (10000*0.01*0.5)/0.05 = 1000.0 — el clamp lo deja en 100.0
    assert ent['unidades'][1] == pytest.approx(100.0)
    assert ent['unidades'][1] < 500.0


def test_tramo_pendiente_que_nunca_dispara_no_construye_mas_posicion():
    n = 10
    o, h, l, c = _ohlc_plano(n)
    o[6] = 110.0
    s = _senales_vacias(n)
    s['entradas_long'][2] = True
    s['salidas_long'][5] = True
    r = simular(o, h, l, c, s, _cfg_tramos(
        [{'pct': 50.0, 'trigger': 'senal'},
         {'pct': 50.0, 'trigger': 'velas', 'val': 1000.0}]))
    ent = r['entradas']
    assert ent['tramo'].tolist() == [0]      # el 2º tramo nunca llegó a disparar
    t = r['trades']
    assert t['unidades'][0] == pytest.approx(12.5)     # solo lo que construyó el 1er tramo


def test_tramos_y_salidas_parciales_el_cerrar_pct_es_del_total_construido():
    # «Cerrar 50%» en una salida parcial debe cerrar la mitad de TODO lo
    # construido (los dos tramos), no la mitad del primer tramo.
    n = 10
    o, h, l, c = _ohlc_plano(n)
    h[7] = 105.0    # +1R (100 + 1*4) -> dispara la salida parcial
    s = _senales_vacias(n)
    s['entradas_long'][2] = True
    r = simular(o, h, l, c, s, _cfg_tramos(
        [{'pct': 50.0, 'trigger': 'senal'},
         {'pct': 50.0, 'trigger': 'velas', 'val': 1.0}],
        parciales=[{'pct': 50.0, 'r': 1.0, 'trigger': 'r', 'condiciones': [],
                    'gestion': {'tipo': 0, 'val': 0.0}}]))
    ent = r['entradas']
    total_construido = ent['unidades'].sum()
    assert total_construido == pytest.approx(25.0)   # 12.5 + 12.5

    t = r['trades']
    assert t['motivo'][0] == 5                       # parcial por R:R
    assert t['unidades'][0] == pytest.approx(total_construido / 2)   # 12.5, no 6.25


def test_tramo_por_defecto_no_cambia_un_backtest_real():
    # Mismo criterio que test_etapa_por_defecto_no_cambia_un_backtest_real,
    # pero para el lado de la entrada: un sistema real con un solo tramo
    # (100% a la señal) debe dar exactamente el mismo backtest que sin
    # 'tramos' en absoluto.
    rng = np.random.default_rng(11)
    closes = 100 + np.cumsum(rng.normal(0, 1.0, 400))
    df = _df_sintetico(closes)
    setup = {'plantilla': 'Cruce de medias',
             'params': {'tipo': 'SMA', 'rapida': 5, 'lenta': 20,
                        'direccion': 'Ambas'}}
    s = generar_senales_sistema(df, [setup])
    o, h, l, c = (df['open'].values, df['high'].values,
                  df['low'].values, df['close'].values)

    antiguo = simular(o, h, l, c, s, dict(
        CONFIG_BASE, config_por_setup={0: {'stop_atr': 1.0}}))
    nuevo = simular(o, h, l, c, s, dict(CONFIG_BASE, config_por_setup={
        0: {'stop_atr': 1.0, 'tramos': [tramo_entrada_por_defecto()]}}))

    assert antiguo['n_trades'] > 5
    assert nuevo['n_trades'] == antiguo['n_trades']
    assert nuevo['capital_final'] == pytest.approx(antiguo['capital_final'])
    for clave, arr in antiguo['trades'].items():
        assert nuevo['trades'][clave].tolist() == arr.tolist(), clave


def test_trigger_tramo_sin_formato_antiguo_que_migrar():
    assert trigger_tramo({'pct': 100, 'trigger': 'senal'}) == 'senal'
    assert trigger_tramo({'pct': 50, 'trigger': 'retroceso', 'val': 1.0}) == 'retroceso'
    assert trigger_tramo({'pct': 50, 'trigger': 'avance', 'val': 2.0}) == 'avance'
    assert trigger_tramo({'pct': 50}) == 'senal'          # sin 'trigger' -> por defecto
    assert trigger_tramo({'pct': 50, 'trigger': 'inventado'}) == 'senal'


def test_codigo_setup_describe_la_entrada_escalonada():
    setup = {'nombre': 'x', 'plantilla': 'Cruce de medias',
             'params': params_por_defecto('Cruce de medias'),
             'riesgo_pct': 0.01, 'stop_atr': 1.0, 'tp_r': 0.0,
             'salida_n_velas': 0,
             'tramos': [dict(tramo_entrada_por_defecto(), pct=70.0),
                        {'pct': 30.0, 'trigger': 'avance', 'val': 2.0,
                         'gestion': {'tipo': 1, 'val': 0.0}}]}
    cod = codigo_setup(setup, 0)
    assert 'ENTRADA ESCALONADA' in cod
    assert 'con la señal de entrada de la plantilla' in cod
    assert 'si avanza +2 R a favor (pirámide)' in cod
    assert '0.3% del equity' in cod or '0.30%' in cod  # 30% de 1% de riesgo
    # con un solo tramo (el caso normal) no se muestra el bloque
    setup['tramos'] = [tramo_entrada_por_defecto()]
    assert 'ENTRADA ESCALONADA' not in codigo_setup(setup, 0)


# ── cierre parcial por mecanismo (stop / TP / break-even / trailing) ──────

def _cfg_mec(**mecanismos):
    """CONFIG_BASE con los mecanismos globales del setup 0 configurados."""
    cfg_setup = {}
    for clave, valor in mecanismos.items():
        if clave in ('salida_stop', 'salida_tp', 'salida_be', 'salida_trailing'):
            cfg_setup[clave] = valor
        else:
            cfg_setup[clave] = valor
    return dict(CONFIG_BASE, config_por_setup={0: cfg_setup})


def test_mecanismos_al_100_no_cambian_un_backtest_real():
    # Los 4 mecanismos explícitos al 100% deben dar exactamente lo mismo que
    # no configurarlos: es la garantía de no-regresión para lo ya guardado.
    rng = np.random.default_rng(11)
    closes = 100 + np.cumsum(rng.normal(0, 1.0, 400))
    df = _df_sintetico(closes)
    setup = {'plantilla': 'Cruce de medias',
             'params': {'tipo': 'SMA', 'rapida': 5, 'lenta': 20,
                        'direccion': 'Ambas'}}
    s = generar_senales_sistema(df, [setup])
    o, h, l, c = (df['open'].values, df['high'].values,
                  df['low'].values, df['close'].values)
    base_setup = {'stop_atr': 1.0, 'tp_r': 3.0, 'be_atr': 1.0, 'trailing_atr': 0.5}

    antiguo = simular(o, h, l, c, s, dict(CONFIG_BASE,
                                          config_por_setup={0: dict(base_setup)}))
    nuevo = simular(o, h, l, c, s, dict(CONFIG_BASE, config_por_setup={0: dict(
        base_setup,
        **{k: salida_mecanismo_por_defecto()
           for k in ('salida_stop', 'salida_tp', 'salida_be', 'salida_trailing')})}))

    assert antiguo['n_trades'] > 5          # el escenario ejercita el motor
    assert nuevo['n_trades'] == antiguo['n_trades']
    assert nuevo['capital_final'] == pytest.approx(antiguo['capital_final'])
    for clave, arr in antiguo['trades'].items():
        assert nuevo['trades'][clave].tolist() == arr.tolist(), clave


def test_stop_parcial_dispara_una_sola_vez_por_posicion():
    # Stop al 50%: el primer toque cierra la mitad y la posición sigue viva.
    # En el SEGUNDO toque el mecanismo ya gastó su turno, así que cierra todo
    # el resto (parcial 0). Sin esta regla, condiciones que siguen siendo
    # ciertas vela tras vela drenarían la posición poco a poco.
    # unidades = 10000*0.01/(1*2) = 50 (ATR=2, stop 1xATR -> dist 2, stop=98).
    n = 20
    o, h, l, c = _ohlc_plano(n)
    l[5] = 97.0      # toca el stop (98)
    l[9] = 97.0      # vuelve a tocarlo
    s = _senales_vacias(n)
    s['entradas_long'][2] = True
    r = simular(o, h, l, c, s, _cfg_mec(
        stop_atr=1.0, salida_stop={'pct': 50.0, 'condiciones': [],
                                   'gestion': {'tipo': 0, 'val': 0.0}}))
    t = r['trades']
    assert t['motivo'][0] == 1 and t['parcial'][0] == -1
    assert t['unidades'][0] == pytest.approx(25.0)          # 50% de 50
    assert t['precio_salida'][0] == pytest.approx(98.0)
    # 2º toque: el stop ya no puede cerrar parcial -> se lleva todo el resto
    assert t['motivo'][1] == 1 and t['parcial'][1] == 0
    assert t['unidades'][1] == pytest.approx(25.0)
    assert r['n_trades'] == 2


def test_salida_por_tiempo_parcial_no_drena_la_posicion():
    # La condición de tiempo es cierta en TODAS las velas a partir de la N,
    # así que sin disparo único cerraría un 30% en cada una hasta agotarla.
    # Debe cerrar el 30% una sola vez y dejar correr el resto.
    n = 24
    o, h, l, c = _ohlc_plano(n)
    s = _senales_vacias(n)
    s['entradas_long'][2] = True
    r = simular(o, h, l, c, s, _cfg_mec(
        salida_n_velas=5,
        salida_tiempo={'pct': 30.0, 'condiciones': [],
                       'gestion': {'tipo': 0, 'val': 0.0}}))
    t = r['trades']
    # sin stop: dist de referencia 2xATR = 4 -> unidades = 100/4 = 25
    assert t['motivo'][0] == 3 and t['parcial'][0] == -5
    assert t['idx_salida'][0] == 8                          # entró en 3, +5 velas
    assert t['unidades'][0] == pytest.approx(7.5)           # 30% de 25
    # el resto NO se desangra vela a vela ni lo cierra el tiempo una vela
    # después: la salida por tiempo queda agotada y el remanente llega al final
    assert r['n_trades'] == 2
    assert t['motivo'][1] == 4 and t['parcial'][1] == 0
    assert t['idx_salida'][1] == n - 1
    assert t['unidades'][1] == pytest.approx(17.5)


def test_cada_mecanismo_conserva_su_propio_turno():
    # El stop original gasta su parcial; después el break-even mueve el nivel
    # y, al tocarse, dispara SU parcial: son mecanismos distintos.
    n = 24
    o, h, l, c = _ohlc_explicito(n, {
        3: (100.0, 100.2, 100.0, 100.1),   # entra (stop original en 98)
        4: (100.1, 100.2, 97.5, 99.0),     # toca el stop -> parcial -1
        5: (99.0, 103.0, 100.5, 102.5),    # +1xATR a favor -> BE mueve el stop a 100
        6: (102.5, 102.5, 99.5, 99.8),     # toca el stop en BE -> parcial -3
    })
    s = _senales_vacias(n)
    s['entradas_long'][2] = True
    r = simular(o, h, l, c, s, _cfg_mec(
        stop_atr=1.0, be_atr=0.5,
        salida_stop={'pct': 50.0, 'condiciones': [], 'gestion': {'tipo': 0, 'val': 0.0}},
        salida_be={'pct': 50.0, 'condiciones': [], 'gestion': {'tipo': 0, 'val': 0.0}}))
    t = r['trades']
    assert t['parcial'][0] == -1 and t['idx_salida'][0] == 4
    assert t['parcial'][1] == -3 and t['idx_salida'][1] == 6
    assert t['precio_salida'][1] == pytest.approx(100.0)


def test_tp_parcial_deja_correr_el_resto():
    # TP 2R al 30%: cierra el 30% al tocar el objetivo y el resto sigue.
    n = 16
    o, h, l, c = _ohlc_plano(n)
    h[5] = 105.0     # 2R sobre 100 con dist 2 -> 104
    s = _senales_vacias(n)
    s['entradas_long'][2] = True
    r = simular(o, h, l, c, s, _cfg_mec(
        stop_atr=1.0, tp_r=2.0,
        salida_tp={'pct': 30.0, 'condiciones': [], 'gestion': {'tipo': 0, 'val': 0.0}}))
    t = r['trades']
    assert t['motivo'][0] == 2 and t['parcial'][0] == -2
    assert t['unidades'][0] == pytest.approx(15.0)           # 30% de 50
    assert t['precio_salida'][0] == pytest.approx(104.0)
    assert r['n_trades'] >= 2                                # el resto cierra aparte


def _ohlc_explicito(n, velas):
    """OHLC plano en 100 con velas concretas sobrescritas por índice:
    {idx: (open, high, low, close)}. Necesario para los tests de break-even y
    trailing: _ohlc_plano deja low=99 en TODAS las velas, así que en cuanto el
    BE sube el stop a 100 el precio ya lo estaría tocando en esa misma vela."""
    o, h, l, c = _ohlc_plano(n)
    for idx, (vo, vh, vl, vc) in velas.items():
        o[idx], h[idx], l[idx], c[idx] = vo, vh, vl, vc
    return o, h, l, c


def test_origen_stop_distingue_break_even_de_stop_original():
    # BE a 0.5xATR mueve el stop a la entrada (100). Al tocarlo, debe
    # aplicarse el pct del BE (25%), no el del stop original (100%).
    # unidades = 10000*0.01/(1*2) = 50; stop original = 98.
    n = 20
    o, h, l, c = _ohlc_explicito(n, {
        3: (100.0, 100.2, 100.0, 100.1),   # entra; sin avance -> BE no salta
        4: (100.1, 102.0, 101.0, 101.5),   # +1xATR a favor -> BE mueve el stop a 100
        5: (101.0, 101.0, 99.0, 99.5),     # vuelve y toca el stop ya en BE
    })
    s = _senales_vacias(n)
    s['entradas_long'][2] = True
    r = simular(o, h, l, c, s, _cfg_mec(
        stop_atr=1.0, be_atr=0.5,
        salida_stop={'pct': 100.0, 'condiciones': [], 'gestion': {'tipo': 0, 'val': 0.0}},
        salida_be={'pct': 25.0, 'condiciones': [], 'gestion': {'tipo': 0, 'val': 0.0}}))
    t = r['trades']
    assert t['idx_salida'][0] == 5
    assert t['motivo'][0] == 1
    assert t['parcial'][0] == -3                             # break-even, no stop
    assert t['unidades'][0] == pytest.approx(12.5)           # 25% de 50
    assert t['precio_salida'][0] == pytest.approx(100.0)     # el stop movido a BE


def test_trailing_gana_el_origen_cuando_mueve_el_stop_despues_del_be():
    # El BE mueve el stop a 100 y acto seguido el trailing lo sube por encima:
    # el toque posterior debe clasificarse como trailing (-4), no como BE.
    n = 20
    o, h, l, c = _ohlc_explicito(n, {
        3: (100.0, 100.2, 100.0, 100.1),   # entra
        4: (100.1, 102.0, 101.5, 101.8),   # BE -> 100, luego trailing -> 101
        5: (101.8, 106.0, 105.5, 105.8),   # trailing -> 106 - 0.5*2 = 105
        6: (105.8, 105.8, 104.0, 104.5),   # toca el stop del trailing
    })
    s = _senales_vacias(n)
    s['entradas_long'][2] = True
    r = simular(o, h, l, c, s, _cfg_mec(
        stop_atr=1.0, be_atr=0.5, trailing_atr=0.5,
        salida_be={'pct': 25.0, 'condiciones': [], 'gestion': {'tipo': 0, 'val': 0.0}},
        salida_trailing={'pct': 40.0, 'condiciones': [], 'gestion': {'tipo': 0, 'val': 0.0}}))
    t = r['trades']
    assert t['idx_salida'][0] == 6
    assert t['parcial'][0] == -4                             # trailing
    assert t['unidades'][0] == pytest.approx(20.0)           # 40% de 50
    assert t['precio_salida'][0] == pytest.approx(105.0)


# ── stop_track / entrada_track: instrumentación para el gráfico ──────────

def test_stop_track_sigue_al_break_even():
    # Mismo escenario de OHLC que test_origen_stop_distingue_break_even_de_stop_original,
    # pero SIN los % parciales de ese test (aquí ambos al 100%, el defecto):
    # con un % parcial el stop se dispara dos veces (parcial en 5, total en 6)
    # y este test quiere un único cierre limpio para verificar el escalón.
    # Entra en 3 (stop 98), el BE salta en la vela 4 (stop -> 100), cierra en 5.
    n = 20
    o, h, l, c = _ohlc_explicito(n, {
        3: (100.0, 100.2, 100.0, 100.1),
        4: (100.1, 102.0, 101.0, 101.5),
        5: (101.0, 101.0, 99.0, 99.5),
    })
    s = _senales_vacias(n)
    s['entradas_long'][2] = True
    r = simular(o, h, l, c, s, _cfg_mec(stop_atr=1.0, be_atr=0.5))
    st = r['stop_track']
    assert np.isnan(st[:3]).all()               # antes de entrar
    assert st[3] == pytest.approx(98.0)          # stop original
    assert st[4] == pytest.approx(100.0)         # el BE salta esta misma vela
    assert st[5] == pytest.approx(100.0)         # la vela de cierre conserva el nivel
    assert np.isnan(st[6:]).all()                # tras cerrar


def test_stop_track_sigue_al_trailing():
    # Mismo escenario de OHLC que
    # test_trailing_gana_el_origen_cuando_mueve_el_stop_despues_del_be, pero
    # con los mecanismos al 100% (el defecto) para un único cierre limpio.
    n = 20
    o, h, l, c = _ohlc_explicito(n, {
        3: (100.0, 100.2, 100.0, 100.1),
        4: (100.1, 102.0, 101.5, 101.8),   # BE -> 100, trailing -> 101
        5: (101.8, 106.0, 105.5, 105.8),   # trailing -> 105
        6: (105.8, 105.8, 104.0, 104.5),   # toca el stop del trailing
    })
    s = _senales_vacias(n)
    s['entradas_long'][2] = True
    r = simular(o, h, l, c, s, _cfg_mec(
        stop_atr=1.0, be_atr=0.5, trailing_atr=0.5))
    st = r['stop_track']
    tramo_valido = st[3:7]
    assert not np.isnan(tramo_valido).any()
    # monótono creciente mientras el trailing arrastra (largo)
    assert (np.diff(tramo_valido) >= -1e-9).all()
    assert st[4] == pytest.approx(101.0)
    assert st[5] == pytest.approx(105.0)
    assert st[6] == pytest.approx(105.0)         # vela de cierre


def test_entrada_track_refleja_el_precio_medio_ponderado():
    # Dos tramos al 50%: la serie vale el precio del 1er fill hasta el 2º
    # tramo, y el medio ponderado a partir de ahí. ATR=2 (por defecto en
    # _senales_vacias), retroceso=1xATR -> dispara si low <= 100-2 = 98.
    n = 20
    o, h, l, c = _ohlc_plano(n)
    l[5] = 97.0    # retroceso detectado en la vela 5
    o[6] = 98.0    # 2º tramo ejecuta al open de la vela siguiente
    s = _senales_vacias(n)
    s['entradas_long'][2] = True
    cfg = dict(CONFIG_BASE, config_por_setup={0: {
        'tramos': [{'pct': 50.0, 'trigger': 'senal', 'val': 0.0, 'condiciones': [],
                   'gestion': {'tipo': 0, 'val': 0.0}},
                  {'pct': 50.0, 'trigger': 'retroceso', 'val': 1.0, 'condiciones': [],
                   'gestion': {'tipo': 0, 'val': 0.0}}]}})
    r = simular(o, h, l, c, s, cfg)
    et = r['entrada_track']
    assert et[3] == pytest.approx(100.0)          # solo el 1er tramo
    assert et[5] == pytest.approx(100.0)          # todavía sin el 2º tramo
    assert et[6] == pytest.approx(99.0)           # medio ponderado 50/50 (100+98)/2
    assert np.isnan(et[:3]).all()


def test_tracks_nan_fuera_de_posicion_y_longitud_n():
    # señal de salida en la vela 5 -> cierra al OPEN de la vela 6 (convención
    # del motor: señal en t, ejecución en t+1), así que la vela de cierre es
    # la 6, no la 5 -- esa vela conserva el último valor (no es NaN).
    n = 12
    o, h, l, c = _ohlc_plano(n)
    o[6] = 110.0
    s = _senales_vacias(n)
    s['entradas_long'][2] = True
    s['salidas_long'][5] = True
    r = simular(o, h, l, c, s, CONFIG_BASE)
    st, et = r['stop_track'], r['entrada_track']
    assert len(st) == n and len(et) == n
    assert np.isnan(st).all()                     # CONFIG_BASE no tiene stop
    assert np.isnan(et[:3]).all() and np.isnan(et[7:]).all()
    assert not np.isnan(et[3:7]).any()


def test_condiciones_del_mecanismo_no_desactivan_la_red_de_seguridad():
    # Si las condiciones del mecanismo NO se cumplen, el cierre sigue siendo
    # total: el stop nunca deja de proteger, solo deja de ser parcial.
    n = 16
    o, h, l, c = _ohlc_plano(n)
    l[5] = 97.0
    s = _senales_vacias(n)
    s['entradas_long'][2] = True
    nunca = np.zeros(n, dtype=bool)
    cfg = _cfg_mec(stop_atr=1.0,
                   salida_stop={'pct': 50.0, 'condiciones': [{'x': 1}],
                                'gestion': {'tipo': 0, 'val': 0.0}})
    cfg['mecanismos_masks_long'] = [[nunca, nunca, nunca, nunca]]
    cfg['mecanismos_masks_short'] = [[nunca, nunca, nunca, nunca]]
    r = simular(o, h, l, c, s, cfg)
    t = r['trades']
    assert r['n_trades'] == 1
    assert t['motivo'][0] == 1 and t['parcial'][0] == 0      # cierre completo
    assert t['unidades'][0] == pytest.approx(50.0)


def test_be_unidad_r_usa_la_distancia_de_riesgo_real():
    # Stop a 2xATR (ATR=2 -> dist de riesgo = 4, stop en 96). Con be=1.0:
    #   · en ×ATR  -> activa al avanzar 1×2 = 2  (o sea 0.5R)
    #   · en R     -> activa al avanzar 1×4 = 4  (1R de verdad)
    # La vela 4 avanza +3: suficiente para ×ATR, insuficiente para R.
    n = 20
    velas = {
        3: (100.0, 100.2, 100.0, 100.1),   # entra
        4: (100.1, 103.0, 100.5, 102.0),   # +3 a favor
        5: (102.0, 102.0, 99.5, 99.8),     # vuelve por debajo de la entrada
    }
    o, h, l, c = _ohlc_explicito(n, velas)
    s = _senales_vacias(n)
    s['entradas_long'][2] = True

    # en ×ATR el BE ya saltó en la vela 4 -> la 5 cierra en la entrada (100)
    r_atr = simular(o, h, l, c, s, _cfg_mec(stop_atr=2.0, be_atr=1.0,
                                            be_unidad='atr'))
    assert r_atr['trades']['motivo'][0] == 1
    assert r_atr['trades']['precio_salida'][0] == pytest.approx(100.0)

    # en R el BE NO llegó a activarse: el stop sigue en 96 y la vela 5 (low
    # 99.5) no lo toca, así que la posición sigue abierta
    r_r = simular(o, h, l, c, s, _cfg_mec(stop_atr=2.0, be_atr=1.0,
                                          be_unidad='r'))
    assert r_r['trades']['idx_salida'][0] > 5


def test_be_unidad_por_defecto_es_atr():
    # Sin 'be_unidad', el motor debe comportarse como siempre (×ATR).
    n = 20
    velas = {
        3: (100.0, 100.2, 100.0, 100.1),
        4: (100.1, 103.0, 100.5, 102.0),
        5: (102.0, 102.0, 99.5, 99.8),
    }
    o, h, l, c = _ohlc_explicito(n, velas)
    s = _senales_vacias(n)
    s['entradas_long'][2] = True
    sin_clave = simular(o, h, l, c, s, _cfg_mec(stop_atr=2.0, be_atr=1.0))
    con_atr = simular(o, h, l, c, s, _cfg_mec(stop_atr=2.0, be_atr=1.0,
                                              be_unidad='atr'))
    for clave, arr in con_atr['trades'].items():
        assert sin_clave['trades'][clave].tolist() == arr.tolist(), clave


def test_validar_parciales_y_tramos():
    assert validar_parciales([{'pct': 50}, {'pct': 50}]) == []
    assert validar_parciales([{'pct': 100}]) == []
    avisos = validar_parciales([{'pct': 100}, {'pct': 100}])
    assert len(avisos) == 1
    assert avisos[0] == f"Etapa 2: {AVISO_EXCESO_PARCIALES}"
    assert AVISO_EXCESO_PARCIALES == "Se excede el 100% del tamaño de la posición."

    assert validar_tramos([{'pct': 50}, {'pct': 50}]) == []
    assert validar_tramos([{'pct': 100}, {'pct': 50}])    # suma 150% -> avisa
    assert validar_setup({'parciales': [{'pct': 100}, {'pct': 100}],
                          'tramos': [{'pct': 100}]})


# ── disparador de estancamiento (N velas sin alcanzar un R mínimo) ─────────

def test_estancamiento_no_afecta_setups_que_no_lo_usan():
    # no-regresión: una etapa normal ('r') da el mismo resultado exacto que
    # antes de existir el disparador de estancamiento (sus arrays quedan a
    # su valor por defecto — False/0/0 — y la rama nueva nunca se activa).
    n = 14
    o, h, l, c = _ohlc_plano(n)
    h[6] = 105.0   # toca 2R (stop 1xATR -> dist=2) en la vela 6
    s = _senales_vacias(n)
    s['entradas_long'][2] = True
    r = simular(o, h, l, c, s, _cfg_parciales(
        [{'pct': 100.0, 'r': 2.0, 'trigger': 'r', 'condiciones': []}],
        stop_atr=1.0))
    t = r['trades']
    assert r['n_trades'] == 1
    assert t['motivo'][0] == 5 and t['parcial'][0] == 1
    assert t['idx_salida'][0] == 6
    assert t['precio_salida'][0] == pytest.approx(104.0)   # entrada 100 + 2R(2)


def test_estancamiento_dispara_por_falta_de_avance():
    # stop 1xATR (ATR=2) -> dist_pos=2, unidades=10000*0.01/2=50. OHLC plano:
    # high=101 siempre -> max_fav tope en 101 -> avance_r=(101-100)/2=0.5,
    # por debajo de r_min=1.0: la operación nunca demuestra que llega a 1R.
    # Entra en vela 2 -> abre en vela 3 (idx_in=3); velas_max=5 -> dispara en
    # la vela 3+5=8, cerrando a mercado (close=100).
    n = 15
    o, h, l, c = _ohlc_plano(n)
    s = _senales_vacias(n)
    s['entradas_long'][2] = True
    r = simular(o, h, l, c, s, _cfg_parciales(
        [{'pct': 60.0, 'trigger': 'estancamiento', 'velas_max': 5, 'r_min': 1.0,
          'condiciones': [], 'gestion': {'tipo': 0, 'val': 0.0}}],
        stop_atr=1.0))
    t = r['trades']
    assert t['idx_entrada'][0] == 3
    assert t['idx_salida'][0] == 8
    assert t['motivo'][0] == 5                    # cierre parcial
    assert t['parcial'][0] == 1                    # etapa secuencial nº1
    assert t['precio_salida'][0] == pytest.approx(100.0)   # cierre a mercado
    assert t['unidades'][0] == pytest.approx(30.0)         # 60% de 50
    assert r['n_trades'] >= 2                       # el resto sigue y cierra aparte


def test_estancamiento_no_dispara_si_ya_alcanzo_el_r():
    # el precio toca 1R pronto (vela 4) y luego se aplana: avance_r, basado
    # en max_fav (que no retrocede), se queda por encima de r_min para
    # siempre -> la etapa NUNCA dispara, aunque pasen de sobra las velas_max.
    n = 20
    o, h, l, c = _ohlc_explicito(n, {
        4: (100.0, 103.0, 100.0, 102.5),   # toca 1R (dist=2 -> 102) y se aplana
    })
    s = _senales_vacias(n)
    s['entradas_long'][2] = True
    r = simular(o, h, l, c, s, _cfg_parciales(
        [{'pct': 50.0, 'trigger': 'estancamiento', 'velas_max': 5, 'r_min': 1.0,
          'condiciones': [], 'gestion': {'tipo': 0, 'val': 0.0}}],
        stop_atr=1.0))
    t = r['trades']
    # ninguna fila debería ser un cierre parcial de la etapa de estancamiento
    assert not (t['motivo'] == 5).any()


def test_estancamiento_aplica_gestion_tras_disparar():
    # gestión tipo 1 (break-even, activación 0.5xATR) tras el parcial: activa
    # el break-even DEL SETUP con esa distancia (mismo mecanismo que usa
    # 'sp_be' — ver _aplicar_gestion_parcial). Con la vela plana ya a 1xATR
    # de avance (max_fav=101, entrada=100, ATR=2), el 0.5xATR (=1.0) ya se
    # cumple: el stop salta a la entrada (100) en la vela siguiente, y como
    # el low plano es 99, lo toca de inmediato.
    n = 15
    o, h, l, c = _ohlc_plano(n)
    s = _senales_vacias(n)
    s['entradas_long'][2] = True
    r = simular(o, h, l, c, s, _cfg_parciales(
        [{'pct': 50.0, 'trigger': 'estancamiento', 'velas_max': 5, 'r_min': 1.0,
          'condiciones': [], 'gestion': {'tipo': 1, 'val': 0.5}}],
        stop_atr=1.0))
    t = r['trades']
    assert t['motivo'][0] == 5 and t['parcial'][0] == 1
    # el resto cierra por el stop ya movido a break-even (100), no al
    # original (98): la gestión de la etapa sí se aplicó
    assert t['motivo'][1] == 1
    assert t['precio_stop'][1] == pytest.approx(100.0)
    assert t['precio_salida'][1] == pytest.approx(100.0)


def test_estancamiento_respeta_condiciones_extra():
    # condición adicional que nunca se cumple: aunque pasen las velas y el
    # avance sea insuficiente, la etapa no dispara.
    n = 15
    o, h, l, c = _ohlc_plano(n)
    s = _senales_vacias(n)
    s['entradas_long'][2] = True
    nunca = np.zeros(n, dtype=bool)
    cfg = _cfg_parciales(
        [{'pct': 100.0, 'trigger': 'estancamiento', 'velas_max': 5, 'r_min': 1.0,
          'condiciones': [{'x': 1}], 'gestion': {'tipo': 0, 'val': 0.0}}],
        stop_atr=1.0)
    cfg['parciales_masks_long'] = [[nunca]]
    cfg['parciales_masks_short'] = [[nunca]]
    r = simular(o, h, l, c, s, cfg)
    assert not (r['trades']['motivo'] == 5).any()


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


# ── resultado_filtrado: segmentación por dirección ──

def _resultado_dos_lados():
    """Simulación con largos y cortos alternados, para los tests de filtrado."""
    n = 60
    c = 100.0 + (np.arange(n) % 10).astype(float)
    o = c.copy()
    h = c + 1.0
    l = c - 1.0
    s = _senales_vacias(n)
    for k in range(5):
        s['entradas_long'][2 + k * 10] = True
        s['salidas_long'][6 + k * 10] = True
        s['entradas_short'][7 + k * 10] = True
        s['salidas_short'][9 + k * 10] = True
    r = simular(o, h, l, c, s, CONFIG_BASE)
    tr = r['trades']
    assert (tr['dir'] > 0).any() and (tr['dir'] < 0).any()   # el fixture sirve
    return r


def test_resultado_filtrado_sin_direccion_conserva_los_trades():
    r = _resultado_dos_lados()
    f = resultado_filtrado(r, 0)
    assert f['n_trades'] == r['n_trades']
    assert f['trades']['pnl'].sum() == pytest.approx(r['trades']['pnl'].sum())
    assert (f['trades']['idx_original'] == np.arange(r['n_trades'])).all()


def test_resultado_filtrado_parte_los_dos_lados_sin_perder_trades():
    r = _resultado_dos_lados()
    largos = resultado_filtrado(r, 1)
    cortos = resultado_filtrado(r, -1)
    assert largos['n_trades'] + cortos['n_trades'] == r['n_trades']
    assert (largos['trades']['dir'] > 0).all()
    assert (cortos['trades']['dir'] < 0).all()
    # idx_original apunta de verdad al trade del resultado sin filtrar
    for f in (largos, cortos):
        assert (r['trades']['pnl'][f['trades']['idx_original']]
                == pytest.approx(f['trades']['pnl']))


def test_resultado_filtrado_reconstruye_la_equity_componiendo_retornos():
    r = _resultado_dos_lados()
    cap0 = CONFIG_BASE['capital_inicial']
    f = resultado_filtrado(r, 1, cap0)
    tr = f['trades']
    orden = np.argsort(tr['idx_salida'], kind='stable')
    esperado = cap0 * np.prod(1.0 + tr['ret_pct'][orden])
    assert f['equity'][0] == pytest.approx(cap0)          # arranca en el capital
    assert f['capital_final'] == pytest.approx(esperado)
    assert f['equity'][-1] == pytest.approx(esperado)
    # escalonada: solo cambia de valor en velas de salida de un trade filtrado
    saltos = np.nonzero(np.diff(f['equity']))[0] + 1
    assert set(saltos.tolist()) <= set(tr['idx_salida'].tolist())
    assert (f['drawdown'] <= 0).all()


def test_resultado_filtrado_no_solapa_la_exposicion_de_cada_lado():
    # el motor mantiene una sola posición a la vez, así que las velas en
    # mercado de un lado y las del otro son conjuntos disjuntos
    r = _resultado_dos_lados()
    n = len(r['equity'])
    expo = lambda d: calcular_metricas(resultado_filtrado(r, d), 0, n)['exposicion_pct']
    assert expo(1) + expo(-1) == pytest.approx(expo(0))


def test_resultado_filtrado_lado_sin_trades_no_rompe_las_metricas():
    n = 20
    o, h, l, c = _ohlc_plano(n)
    s = _senales_vacias(n)
    s['entradas_long'][2] = True
    s['salidas_long'][5] = True
    r = simular(o, h, l, c, s, CONFIG_BASE)
    f = resultado_filtrado(r, -1, CONFIG_BASE['capital_inicial'])
    assert f['n_trades'] == 0
    assert (f['equity'] == CONFIG_BASE['capital_inicial']).all()   # curva plana
    m = calcular_metricas(f, 0, n)
    assert m['n_trades'] == 0
    assert m['retorno_pct'] == pytest.approx(0.0)
    assert m['win_rate'] is None
    assert m['exposicion_pct'] == pytest.approx(0.0)


def test_resultado_filtrado_respeta_la_ventana_is_oos():
    r = _resultado_dos_lados()
    n = len(r['equity'])
    corte = dividir_is_oos(n)
    f = resultado_filtrado(r, 1)
    tr = f['trades']
    is_ = calcular_metricas(f, 0, corte)
    oos = calcular_metricas(f, corte, n)
    assert is_['n_trades'] == int((tr['idx_entrada'] < corte).sum())
    assert oos['n_trades'] == int((tr['idx_entrada'] >= corte).sum())
    assert is_['n_trades'] + oos['n_trades'] == f['n_trades']
