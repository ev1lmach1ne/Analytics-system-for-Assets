"""Canal de Donchian y plantilla 'Breakout de canal (Donchian)'."""
import numpy as np
import pandas as pd

from core.strategies import (
    ESTRATEGIAS, _serie_indicador, donchian, generar_senales, params_por_defecto,
)


def _df(highs, lows, closes=None):
    n = len(highs)
    closes = closes if closes is not None else [(h + l) / 2 for h, l in zip(highs, lows)]
    return pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=n, freq='1h', tz='UTC'),
        'open': closes, 'close': closes,
        'high': np.asarray(highs, dtype=np.float64),
        'low': np.asarray(lows, dtype=np.float64),
    })


def _senales(df, **params):
    p = params_por_defecto('Breakout de canal (Donchian)')
    p.update(params)
    return generar_senales('Breakout de canal (Donchian)', df, p)


def test_el_canal_excluye_la_vela_en_curso():
    """La propiedad que hace posible la ruptura: si el canal se incluyera a sí
    mismo, high nunca podría superar el máximo."""
    high = np.array([10.0, 11.0, 12.0, 20.0, 13.0])
    low = np.array([9.0, 10.0, 11.0, 12.0, 8.0])
    sup, inf = donchian(high, low, 3)
    assert np.isnan(sup[:3]).all()          # warm-up: 3 velas + el desplazamiento
    assert sup[3] == 12.0                   # máximo de las velas 0-2, no de la 3
    assert sup[4] == 20.0                   # ahora sí entra el 20 de la vela 3
    assert inf[3] == 9.0
    assert (high > sup)[3]                  # la vela 3 rompe de verdad


def test_incluir_actual_desactiva_el_desplazamiento():
    high = np.array([10.0, 11.0, 12.0, 20.0])
    low = np.array([9.0, 10.0, 11.0, 12.0])
    sup, _ = donchian(high, low, 3, incluir_actual=True)
    assert sup[3] == 20.0
    assert not (high > sup)[3]   # tautología: nunca rompe


def test_entrada_long_en_la_vela_que_rompe_el_maximo():
    high = np.array([10.0, 10.0, 10.0, 10.0, 15.0, 11.0])
    low = np.array([9.0] * 6)
    s = _senales(_df(high, low), periodo=3, direccion='Long')
    assert np.where(s['entradas_long'])[0].tolist() == [4]


def test_fuente_close_ignora_las_mechas():
    """Con fuente=close el canal se forma con cierres y lo rompe el cierre: una
    mecha que sobresale no dispara nada."""
    high = np.array([10.0, 10.0, 10.0, 10.0, 15.0, 10.5])
    low = np.array([9.0] * 6)
    close = np.array([9.5, 9.5, 9.5, 9.5, 9.5, 10.4])
    df = _df(high, low, close)
    con_high = _senales(df, periodo=3, fuente='high/low', direccion='Long')
    con_close = _senales(df, periodo=3, fuente='close', direccion='Long')
    assert con_high['entradas_long'][4]        # la mecha rompe el canal de máximos
    assert not con_close['entradas_long'][4]   # pero el cierre no se movió
    assert con_close['entradas_long'][5]       # aquí sí cierra por encima


def test_sin_canal_de_salida_propio_la_salida_es_la_ruptura_contraria():
    high = np.array([10.0, 10.0, 10.0, 10.0, 15.0, 11.0])
    low = np.array([9.0, 9.0, 9.0, 9.0, 9.0, 5.0])
    s = _senales(_df(high, low), periodo=3, periodo_salida=0, direccion='Ambas')
    assert np.array_equal(s['salidas_long'], s['entradas_short'])


def test_canal_de_salida_propio_es_mas_reactivo():
    """Un retroceso puede perder el canal corto sin llegar a perder el largo:
    es justo el esquema de las Tortugas, entrar lento y salir rápido."""
    high = np.concatenate([10.0 + np.arange(20), [21.0], 30.0 + np.arange(9)])
    low = np.concatenate([9.0 + np.arange(20), [20.0], 29.0 + np.arange(9)])
    df = _df(high, low)
    largo = _senales(df, periodo=10, periodo_salida=0, direccion='Long')
    corto = _senales(df, periodo=10, periodo_salida=3, direccion='Long')
    assert corto['salidas_long'][20]        # pierde el mínimo de 3 velas
    assert not largo['salidas_long'][20]    # pero no el de 10


def test_donchian_disponible_como_indicador_de_regla():
    high = np.array([10.0, 11.0, 12.0, 20.0, 13.0])
    low = np.array([9.0, 10.0, 11.0, 12.0, 8.0])
    df = _df(high, low)
    sup_esperado, inf_esperado = donchian(high, low, 3)
    sup = _serie_indicador(df, {'tipo': 'DONCHIAN_SUP', 'periodo': 3})
    inf = _serie_indicador(df, {'tipo': 'DONCHIAN_INF', 'periodo': 3})
    assert np.array_equal(sup[3:], sup_esperado[3:])
    assert np.array_equal(inf[3:], inf_esperado[3:])


def test_esta_registrada_con_los_parametros_esperados():
    est = ESTRATEGIAS['Breakout de canal (Donchian)']
    assert {p['clave'] for p in est['params']} == {
        'fuente', 'periodo', 'periodo_salida', 'direccion'}
