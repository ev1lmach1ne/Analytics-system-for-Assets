"""Plantilla 'Parabolic SAR' del constructor: señales de giro y su integración
con el registro de estrategias."""
import numpy as np
import pandas as pd

from core.strategies import (
    ESTRATEGIAS, _serie_indicador, _sar_serie, generar_senales,
    params_por_defecto,
)


def _df_v(n_subida=40, n_bajada=40):
    """Precio en V: sube y luego baja, para forzar un giro del SAR."""
    subida = np.arange(n_subida, dtype=np.float64)
    bajada = np.arange(n_bajada - 1, -1, -1, dtype=np.float64)
    close = 100.0 + np.concatenate([subida, bajada])
    n = len(close)
    return pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=n, freq='1h', tz='UTC'),
        'open': close, 'close': close, 'high': close + 0.5, 'low': close - 0.5,
    })


def _senales(df, **params):
    p = params_por_defecto('Parabolic SAR')
    p.update(params)
    return generar_senales('Parabolic SAR', df, p)


def test_esta_registrada_con_la_forma_esperada():
    est = ESTRATEGIAS['Parabolic SAR']
    assert est['defaults_setup']['stop_atr'] == 0.0   # el giro ya es la salida
    assert {p['clave'] for p in est['params']} == {
        'af_inicial', 'af_paso', 'af_max', 'direccion'}


def test_la_entrada_long_coincide_con_el_giro_alcista():
    df = _df_v()
    s = _senales(df, direccion='Ambas')
    _, tend = _sar_serie(df['high'].values, df['low'].values, 0.02, 0.02, 0.2)
    giros_alcistas = np.where((tend == 1) & (np.roll(tend, 1) == -1))[0]
    giros_alcistas = giros_alcistas[giros_alcistas > 0]
    assert np.array_equal(np.where(s['entradas_long'])[0], giros_alcistas)


def test_la_salida_long_es_el_giro_contrario():
    df = _df_v()
    s = _senales(df, direccion='Ambas')
    # simetría exacta: lo que abre corto cierra largo y viceversa
    assert np.array_equal(s['salidas_long'], s['entradas_short'])
    assert np.array_equal(s['salidas_short'], s['entradas_long'])


def test_direccion_long_no_genera_señales_short():
    df = _df_v()
    s = _senales(df, direccion='Long')
    assert not s['entradas_short'].any()
    assert not s['salidas_short'].any()
    assert s['salidas_long'].any()   # sigue pudiendo cerrar


def test_la_v_produce_al_menos_un_giro_bajista():
    df = _df_v()
    s = _senales(df, direccion='Ambas')
    assert s['entradas_short'].any()


def test_sar_disponible_como_indicador_de_regla():
    df = _df_v()
    serie = _serie_indicador(df, {'tipo': 'SAR'})
    sar_directo, _ = _sar_serie(df['high'].values, df['low'].values, 0.02, 0.02, 0.2)
    assert np.array_equal(serie, sar_directo)
