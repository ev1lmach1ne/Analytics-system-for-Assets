import numpy as np
import pandas as pd
import pytest

from core.backtest import dividir_is_oos, calcular_metricas
from core.optimizer import (
    generar_grid, n_combinaciones, optimizar_setup,
    estadisticas_conjunto, analisis_por_parametro, analisis_vecindad,
    fiabilidad_estadistica, _sharpe_expansivo, N_PUNTOS_SPARKLINE,
    MIN_BARRAS_SHARPE,
)
from core.strategies import preparar_eventos_noticias


CONFIG_GLOBAL = {'capital_inicial': 10000.0, 'comision_pct': 0.0,
                  'slippage_pct': 0.0}


def _setup_cruce(rapida=5, lenta=20):
    return {'nombre': 'Setup 1', 'plantilla': 'Cruce de medias',
            'params': {'tipo': 'SMA', 'rapida': rapida, 'lenta': lenta,
                      'direccion': 'Ambas'},
            'riesgo_pct': 0.01, 'stop_atr': 0.0, 'tp_r': 0.0,
            'salida_n_velas': 0}


def _df_tendencia(n=250, semilla=3):
    rng = np.random.default_rng(semilla)
    closes = 100 + np.cumsum(rng.normal(0.05, 1.0, n))
    return pd.DataFrame({'open': closes, 'high': closes + 0.5,
                         'low': closes - 0.5, 'close': closes})


# ══════════════ generar_grid / n_combinaciones ══════════════

def test_generar_grid_un_param_int():
    combos = generar_grid({'rapida': {'min': 10, 'max': 20, 'step': 5, 'tipo': 'int'}})
    assert [c['rapida'] for c in combos] == [10, 15, 20]


def test_generar_grid_dos_params_producto_cartesiano():
    sweep = {'rapida': {'min': 5, 'max': 10, 'step': 5, 'tipo': 'int'},
             'lenta': {'min': 20, 'max': 30, 'step': 10, 'tipo': 'int'}}
    combos = generar_grid(sweep)
    assert len(combos) == 2 * 2
    pares = {(c['rapida'], c['lenta']) for c in combos}
    assert pares == {(5, 20), (5, 30), (10, 20), (10, 30)}


def test_generar_grid_float_redondea_paso():
    combos = generar_grid({'desv': {'min': 1.0, 'max': 2.0, 'step': 0.5, 'tipo': 'float'}})
    assert [c['desv'] for c in combos] == [1.0, 1.5, 2.0]


def test_generar_grid_min_igual_max_una_combinacion():
    combos = generar_grid({'rapida': {'min': 10, 'max': 10, 'step': 5, 'tipo': 'int'}})
    assert len(combos) == 1


def test_generar_grid_supera_limite():
    sweep = {'a': {'min': 1, 'max': 100, 'step': 1, 'tipo': 'int'},
             'b': {'min': 1, 'max': 100, 'step': 1, 'tipo': 'int'}}
    with pytest.raises(ValueError):
        generar_grid(sweep, limite=500)


def test_n_combinaciones_coincide_con_generar_grid():
    sweep = {'rapida': {'min': 5, 'max': 10, 'step': 5, 'tipo': 'int'},
             'lenta': {'min': 20, 'max': 30, 'step': 10, 'tipo': 'int'}}
    assert n_combinaciones(sweep) == len(generar_grid(sweep))


# ══════════════ optimizar_setup ══════════════

def test_optimizar_setup_no_ve_datos_de_oos():
    """El ranking en IS no debe cambiar si se altera el tramo OOS: prueba de
    que optimizar_setup nunca simula más allá del corte IS/OOS."""
    df_a = _df_tendencia()
    pct_oos = 0.30
    corte = dividir_is_oos(len(df_a), pct_oos)
    df_b = df_a.copy()
    # tramo OOS completamente distinto (caída brusca) — no debería tocar el
    # resultado si de verdad solo se simula [0, corte)
    df_b.loc[corte:, ['open', 'high', 'low', 'close']] -= 500.0

    kwargs = dict(setup_base=_setup_cruce(), sweep_params={}, sweep_riesgo={},
                 config_global=CONFIG_GLOBAL, pct_oos=pct_oos)
    res_a = optimizar_setup(df_a, **kwargs)
    res_b = optimizar_setup(df_b, **kwargs)

    assert res_a[0]['metricas'] == res_b[0]['metricas']
    assert np.array_equal(res_a[0]['equity_sparkline'], res_b[0]['equity_sparkline'])


def test_optimizar_setup_numero_de_resultados():
    df = _df_tendencia()
    sweep_params = {'rapida': {'min': 3, 'max': 9, 'step': 3, 'tipo': 'int'}}
    res = optimizar_setup(df, _setup_cruce(), sweep_params, {}, CONFIG_GLOBAL,
                          pct_oos=0.30)
    assert len(res) == 3
    assert {r['params_barridos']['rapida'] for r in res} == {3, 6, 9}


def test_optimizar_setup_ordenado_por_metrica_desc():
    df = _df_tendencia()
    sweep_params = {'rapida': {'min': 3, 'max': 15, 'step': 3, 'tipo': 'int'}}
    res = optimizar_setup(df, _setup_cruce(), sweep_params, {}, CONFIG_GLOBAL,
                          pct_oos=0.30, metrica='retorno_pct')
    valores = [r['metricas']['retorno_pct'] for r in res]
    valores_no_none = [v for v in valores if v is not None]
    assert valores_no_none == sorted(valores_no_none, reverse=True)


def test_optimizar_setup_barre_riesgo():
    df = _df_tendencia()
    sweep_riesgo = {'riesgo_pct': {'min': 0.01, 'max': 0.02, 'step': 0.01, 'tipo': 'float'}}
    res = optimizar_setup(df, _setup_cruce(), {}, sweep_riesgo, CONFIG_GLOBAL,
                          pct_oos=0.30)
    assert len(res) == 2
    riesgos = sorted(r['setup']['riesgo_pct'] for r in res)
    assert riesgos == pytest.approx([0.01, 0.02])


def _setup_bollinger(periodo=20, desv=2.0, stop_atr=1.5, riesgo=0.01):
    return {
        'nombre': 'BB', 'plantilla': 'Bollinger + ATR',
        'params': {'periodo': periodo, 'desv': desv, 'direccion': 'Ambas'},
        'riesgo_pct': riesgo, 'stop_atr': stop_atr, 'tp_r': 2.0,
        'salida_n_velas': 0,
    }


def test_optimizar_setup_bollinger_periodo_desv_riesgo_stop():
    """Caso GUI reportado: Bollinger + ATR barriendo periodo BB, desviaciones,
    riesgo del setup y stop ×ATR debe terminar y devolver todas las combos."""
    df = _df_tendencia(n=400)
    sweep_params = {
        'periodo': {'min': 10, 'max': 20, 'step': 5, 'tipo': 'int'},
        'desv': {'min': 1.5, 'max': 2.5, 'step': 0.5, 'tipo': 'float'},
    }
    sweep_riesgo = {
        'riesgo_pct': {'min': 0.005, 'max': 0.015, 'step': 0.005, 'tipo': 'float'},
        'stop_atr': {'min': 1.0, 'max': 2.0, 'step': 0.5, 'tipo': 'float'},
    }
    # 3 periodos × 3 desv × 3 riesgos × 3 stops = 81 combos
    avances = []
    res = optimizar_setup(
        df, _setup_bollinger(), sweep_params, sweep_riesgo, CONFIG_GLOBAL,
        pct_oos=0.30, metrica='sharpe',
        progreso_cb=lambda i, total: avances.append((i, total)))
    assert len(res) == 81
    assert avances[0] == (0, 81)
    assert avances[-1] == (81, 81)
    assert all('periodo' in r['params_barridos'] for r in res)
    assert all('_riesgo.stop_atr' in r['params_barridos'] for r in res)


def test_optimizar_setup_respeta_limite_combos():
    df = _df_tendencia()
    sweep_params = {'rapida': {'min': 1, 'max': 100, 'step': 1, 'tipo': 'int'},
                    'lenta': {'min': 1, 'max': 100, 'step': 1, 'tipo': 'int'}}
    with pytest.raises(ValueError):
        optimizar_setup(df, _setup_cruce(), sweep_params, {}, CONFIG_GLOBAL,
                        pct_oos=0.30, limite_combos=500)


def test_optimizar_setup_respeta_filtro_de_noticias():
    """eventos_noticias debe llegar hasta generar_senales_sistema y bloquear
    entradas: una ventana de noticia que cubre TODO el tramo IS debe dejar el
    barrido sin operaciones, frente al mismo setup sin el filtro."""
    n = 250
    df = _df_tendencia(n=n)
    df['timestamp'] = pd.date_range('2024-01-01', periods=n, freq='1h', tz='UTC')

    setup = _setup_cruce()
    setup['filtros'] = {'noticias': {
        'activo': True, 'minutos_antes': 999999, 'minutos_despues': 999999,
        'impacto_minimo': 'alto', 'monedas': None, 'cerrar_posiciones': False,
    }}
    eventos_df = pd.DataFrame({
        'timestamp': [df['timestamp'].iloc[n // 2]],
        'pais': ['US'], 'evento': ['Evento'], 'impacto': ['alto'],
    })
    eventos = preparar_eventos_noticias(eventos_df)

    res_sin_filtro = optimizar_setup(df, setup, {}, {}, CONFIG_GLOBAL, pct_oos=0.30,
                                     eventos_noticias=None)
    res_con_filtro = optimizar_setup(df, setup, {}, {}, CONFIG_GLOBAL, pct_oos=0.30,
                                     eventos_noticias=eventos)

    assert res_sin_filtro[0]['metricas']['n_trades'] > 0
    assert res_con_filtro[0]['metricas']['n_trades'] == 0


def test_optimizar_setup_solo_simula_tramo_is():
    """La equity curve devuelta debe tener longitud igual al tamaño del
    tramo IS, no de la serie completa."""
    df = _df_tendencia(n=300)
    corte = dividir_is_oos(len(df), 0.30)
    res = optimizar_setup(df, _setup_cruce(), {}, {}, CONFIG_GLOBAL, pct_oos=0.30)
    assert res[0]['metricas']['n_trades'] >= 0
    # el sparkline es una versión downsampleada del tramo IS: nunca más
    # puntos que velas de IS
    assert len(res[0]['equity_sparkline']) <= corte


# ══════════════ estadísticas del conjunto ══════════════

def _combo(params=None, retorno=0.0, n_trades=50, pf=None, sharpe=None,
           pnl=None, sqn=None):
    return {'params_barridos': params or {},
            'setup': {},
            'metricas': {'retorno_pct': retorno, 'n_trades': n_trades,
                        'profit_factor': pf, 'sharpe': sharpe, 'sqn': sqn,
                        'pnl_total': pnl if pnl is not None else retorno}}


def test_estadisticas_mitad_rentables():
    res = [_combo(retorno=10.0), _combo(retorno=5.0),
           _combo(retorno=-3.0), _combo(retorno=-8.0)]
    est = estadisticas_conjunto(res)
    assert est['pct_rentables'] == 50.0
    assert est['retorno_medio'] == pytest.approx(1.0)
    assert est['retorno_mediana'] == pytest.approx(1.0)


def test_estadisticas_sin_trades_excluidos():
    res = [_combo(retorno=10.0), _combo(retorno=0.0, n_trades=0),
           _combo(retorno=0.0, n_trades=0)]
    est = estadisticas_conjunto(res)
    assert est['n_combos'] == 3
    assert est['n_sin_trades'] == 2
    assert est['pct_rentables'] == 100.0   # solo cuenta la que operó


def test_estadisticas_pf_inf_cuenta_como_mayor_1():
    res = [_combo(retorno=5.0, pf=float('inf')),
           _combo(retorno=-5.0, pf=0.5),
           _combo(retorno=1.0, pf=None)]   # None se excluye
    est = estadisticas_conjunto(res)
    assert est['pct_pf_mayor_1'] == 50.0


def test_estadisticas_lista_vacia():
    est = estadisticas_conjunto([])
    assert est['n_combos'] == 0
    assert est['pct_rentables'] is None


def test_estadisticas_sesgo_asimetrico():
    # muchas pérdidas pequeñas + un ganador enorme -> cola derecha (sesgo > 0)
    res = [_combo(retorno=-1.0)] * 10 + [_combo(retorno=100.0)]
    assert estadisticas_conjunto(res)['sesgo'] > 0


def test_analisis_por_parametro_culpable():
    res = []
    for riesgo in (0.01, 0.05):
        for rapida in (10, 20):
            gana = riesgo == 0.01
            res.append(_combo({'_riesgo.riesgo_pct': riesgo, 'rapida': rapida},
                              retorno=10.0 if gana else -10.0))
    a = analisis_por_parametro(res)
    assert a['_riesgo.riesgo_pct']['impacto'] == 100.0
    assert a['rapida']['impacto'] == 0.0
    por_valor = dict(a['_riesgo.riesgo_pct']['valores'])
    assert por_valor[0.01]['pct_rentables'] == 100.0
    assert por_valor[0.05]['pct_rentables'] == 0.0


def test_analisis_por_parametro_valor_unico_omitido():
    res = [_combo({'rapida': 10, 'lenta': 50}, retorno=1.0),
           _combo({'rapida': 20, 'lenta': 50}, retorno=2.0)]
    a = analisis_por_parametro(res)
    assert 'rapida' in a
    assert 'lenta' not in a   # un solo valor -> sin información


def test_analisis_por_parametro_vacio():
    assert analisis_por_parametro([]) == {}


def test_vecindad_superficie_suave():
    res = [_combo({'rapida': v}, retorno=float(10 + i))
           for i, v in enumerate((5, 10, 15, 20))]
    v = analisis_vecindad(res)
    assert v['rugosidad'] == pytest.approx(1.0)
    assert v['plateau_top'] == 100.0
    assert v['pct_mayor_cluster'] == 100.0


def test_vecindad_pico_aislado():
    retornos = [-1.0, -1.0, 100.0, -1.0, -1.0]
    res = [_combo({'rapida': v}, retorno=r)
           for v, r in zip((5, 10, 15, 20, 25), retornos)]
    v = analisis_vecindad(res)
    # las 2 vecinas del pico pierden; el resto del top también está rodeado
    # de perdedoras -> plateau bajo
    assert v['plateau_top'] < 50.0
    assert v['pct_mayor_cluster'] == pytest.approx(20.0)   # cluster de 1 de 5


def test_vecindad_dos_islas():
    # rentables en posiciones 0-1 (isla de 2) y 4 (isla de 1)
    retornos = [5.0, 5.0, -1.0, -1.0, 5.0]
    res = [_combo({'rapida': v}, retorno=r)
           for v, r in zip((5, 10, 15, 20, 25), retornos)]
    v = analisis_vecindad(res)
    assert v['pct_mayor_cluster'] == pytest.approx(40.0)   # 2 de 5


def test_vecindad_un_solo_combo():
    v = analisis_vecindad([_combo({}, retorno=1.0)])
    assert v['rugosidad'] is None
    assert analisis_vecindad([])['rugosidad'] is None


def test_fiabilidad_sqn_suficiente():
    # SQN en vez de un nº fijo de trades: escala solo con la consistencia de
    # la muestra, no con el estilo de trading (scalping vs swing/position)
    res = [_combo(retorno=1.0, n_trades=10, sqn=0.5),
           _combo(retorno=2.0, n_trades=50, sqn=2.0)]
    f = fiabilidad_estadistica(res, min_sqn=1.0)
    assert f['pct_sqn_suficiente'] == 50.0
    assert f['mediana_sqn'] == pytest.approx(1.25)
    # el nº de trades se conserva como dato informativo, no como criterio
    assert f['mediana_trades'] == 30.0


def test_fiabilidad_sqn_pocos_trades_pero_consistentes_cuenta():
    # un sistema con pocos trades muy consistentes (posible position trading)
    # puede tener SQN alto — no debe penalizarse solo por el nº de trades
    res = [_combo(retorno=5.0, n_trades=8, sqn=2.5)]
    f = fiabilidad_estadistica(res, min_sqn=1.0)
    assert f['pct_sqn_suficiente'] == 100.0


def test_fiabilidad_sin_sqn_valido():
    res = [_combo(retorno=1.0, sqn=None), _combo(retorno=2.0, sqn=None)]
    f = fiabilidad_estadistica(res)
    assert f['pct_sqn_suficiente'] is None
    assert f['mediana_sqn'] is None
    assert f['mediana_trades'] == 50.0   # sigue siendo informativo


def test_fiabilidad_concentracion_dominante():
    res = [_combo(retorno=100.0, pnl=1000.0)] + \
          [_combo(retorno=0.1, pnl=1.0) for _ in range(19)]
    f = fiabilidad_estadistica(res)
    assert f['concentracion_top5'] > 95.0


def test_fiabilidad_ratio_none_con_mediana_negativa():
    res = [_combo(retorno=-5.0), _combo(retorno=-1.0), _combo(retorno=10.0)]
    f = fiabilidad_estadistica(res)
    assert f['ratio_mejor_mediana'] is None


# ══════════════ Sharpe expansivo (evolución temporal) ══════════════

def test_optimizar_setup_incluye_sharpe_sparkline():
    df = _df_tendencia()
    res = optimizar_setup(df, _setup_cruce(), {}, {}, CONFIG_GLOBAL, pct_oos=0.30)
    curva = res[0]['sharpe_sparkline']
    assert 0 < len(curva) <= N_PUNTOS_SPARKLINE


def test_sharpe_expansivo_final_coincide_con_metricas():
    # equity sintética con ruido: el último punto de la serie expansiva debe
    # ser el Sharpe (no anualizado) que calcular_metricas da al tramo entero
    rng = np.random.default_rng(11)
    equity = 10000 * np.exp(np.cumsum(rng.normal(0.001, 0.01, 300)))
    serie = _sharpe_expansivo(equity)
    resultado = {'equity': equity,
                 'trades': {'idx_entrada': np.array([], dtype=np.int64),
                            'idx_salida': np.array([], dtype=np.int64),
                            'pnl': np.array([]), 'ret_pct': np.array([]),
                            'r_multiple': np.array([]),
                            'notional_redondo': np.array([]),
                            'costo_comision': np.array([])}}
    esperado = calcular_metricas(resultado)['sharpe']
    assert serie[-1] == pytest.approx(esperado, abs=1e-9)


def test_sharpe_expansivo_primeras_barras_nan():
    equity = np.linspace(10000, 11000, 100)
    serie = _sharpe_expansivo(equity)
    assert np.isnan(serie[:MIN_BARRAS_SHARPE]).all()


def test_sharpe_expansivo_equity_plana_todo_nan():
    serie = _sharpe_expansivo(np.full(50, 10000.0))
    assert np.isnan(serie).all()


def test_sharpe_expansivo_corta_no_crashea():
    assert len(_sharpe_expansivo(np.array([10000.0]))) == 0
    assert len(_sharpe_expansivo(np.array([]))) == 0
