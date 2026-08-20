"""Indicadores nuevos del catálogo (Supertrend, MACD, ADX, Aroon, CMO, OBV,
TRIX, StochRSI, VWAP) y las plantillas Supertrend/MACD/ADX: valores sanos,
rango y que generan señales direccionales coherentes."""
import numpy as np
import pandas as pd
import pytest

from core.strategies import (
    _supertrend_serie, _macd_series, _adx_series, _aroon_series, _cmo_serie,
    _obv_serie, _trix_serie, _stochrsi_series, _vwap_serie,
    ESTRATEGIAS, generar_senales, params_por_defecto, _serie_indicador,
    _INDICADORES_REGLA, filas_plantilla,
)


@pytest.fixture(scope='module')
def df():
    rng = np.random.default_rng(11)
    n = 600
    ts = pd.date_range('2024-01-01', periods=n, freq='1h', tz='UTC')
    c = 100 + np.cumsum(rng.normal(0.01, 0.4, n))
    h = c + 0.5
    l = c - 0.5
    return pd.DataFrame({'timestamp': ts, 'open': c, 'close': c,
                         'high': h, 'low': l, 'volume': 1000.0})


def test_editor_incluye_los_nueve_indicadores():
    for tipo in ('SUPERTREND', 'MACD_LINEA', 'MACD_SENAL', 'MACD_HIST',
                 'ADX', 'DI_PLUS', 'DI_MINUS', 'AROON_UP', 'AROON_DN',
                 'CMO', 'OBV', 'TRIX', 'STOCHRSI', 'VWAP'):
        assert tipo in _INDICADORES_REGLA


def test_serie_indicador_calcula_sin_romper(df):
    for tipo in ('SUPERTREND', 'MACD_LINEA', 'MACD_SENAL', 'MACD_HIST',
                 'ADX', 'DI_PLUS', 'DI_MINUS', 'AROON_UP', 'AROON_DN',
                 'CMO', 'OBV', 'TRIX', 'STOCHRSI', 'VWAP'):
        v = _serie_indicador(df, {'tipo': tipo, 'periodo': 14})
        assert np.isfinite(v).any(), tipo


def test_supertrend_nivel_y_tendencia(df):
    st, tend = _supertrend_serie(df['high'].values, df['low'].values,
                                 df['close'].values, 10, 3.0)
    assert np.isfinite(st).any()
    assert set(np.unique(tend[np.isfinite(st)])) <= {1, -1}
    # el nivel alcista queda por debajo del precio y viceversa
    c = df['close'].values
    alcista = tend > 0
    if alcista.any():
        assert (c[alcista] >= st[alcista]).mean() > 0.5


def test_macd_relaciones(df):
    linea, senal, hist = _macd_series(df['close'].values, 12, 26, 9)
    assert np.isfinite(linea).any()
    np.testing.assert_allclose(hist, linea - senal, atol=1e-12)


def test_adx_en_rango_y_di(df):
    adx, pdi, mdi = _adx_series(df['high'].values, df['low'].values,
                                df['close'].values, 14)
    validos = np.isfinite(adx)
    assert validos.any()
    assert ((adx[validos] >= 0) & (adx[validos] <= 100)).all()
    assert ((pdi[validos] >= 0) & (mdi[validos] >= 0)).all()


def test_aroon_en_rango(df):
    up, dn = _aroon_series(df['high'].values, df['low'].values, 25)
    v = np.isfinite(up)
    assert ((up[v] >= 0) & (up[v] <= 100)).all()
    assert ((dn[v] >= 0) & (dn[v] <= 100)).all()


def test_cmo_en_rango(df):
    cmo = _cmo_serie(df['close'].values, 14)
    v = np.isfinite(cmo)
    assert ((cmo[v] >= -100) & (cmo[v] <= 100)).all()


def test_obv_acumula_con_signo(df):
    obv = _obv_serie(df['close'].values, df['volume'].values)
    c = df['close'].values
    # en la primera subida, el OBV suma volumen
    assert obv[1] == pytest.approx(1000.0)
    assert obv[-1] != 0.0 or not np.any(c[1:] != c[:-1])


def test_trix_es_un_porcentaje(df):
    trix = _trix_serie(df['close'].values, 15)
    assert np.isfinite(trix).any()
    assert np.abs(trix[np.isfinite(trix)]).max() < 100.0


def test_stochrsi_en_rango(df):
    k, d = _stochrsi_series(df['close'].values, 14, 3, 3)
    vk = np.isfinite(k)
    assert ((k[vk] >= 0) & (k[vk] <= 1)).all()
    vd = np.isfinite(d)
    assert vd.any()
    assert ((d[vd] >= 0) & (d[vd] <= 1)).all()


def test_vwap_sigue_al_precio(df):
    vwap = _vwap_serie(df, 14)
    v = np.isfinite(vwap)
    assert v.any()
    c = df['close'].values
    # el VWAP de sesión queda cerca del precio (todos los precios ~ iguales)
    assert np.abs(vwap[v] - c[v]).mean() < 1.0


def test_plantillas_nuevas_generan_senales(df):
    for nombre in ('Supertrend', 'MACD', 'ADX (fuerza de tendencia)'):
        p = params_por_defecto(nombre)
        s = generar_senales(nombre, df, p)
        assert s['entradas_long'].sum() + s['entradas_short'].sum() > 0, nombre


def test_plantillas_aroon_cmo_trix_stochrsi_generan_senales(df):
    for nombre in ('Aroon', 'CMO', 'TRIX', 'StochRSI'):
        p = params_por_defecto(nombre)
        s = generar_senales(nombre, df, p)
        assert s['entradas_long'].sum() + s['entradas_short'].sum() > 0, nombre
        # filas de señal legibles para la tabla/pseudocódigo
        assert len(filas_plantilla(nombre, p)) >= 1, nombre


def test_plantillas_ichimoku_keltner_ttm_generan_senales():
    rng = np.random.default_rng(17)
    n = 600
    ts = pd.date_range('2024-01-01', periods=n, freq='1h', tz='UTC')
    # la reversión a la media (Keltner) necesita un mercado oscilante; las
    # de cruce (Ichimoku/TTM) funcionan también en tendencia
    oscilante = 100 + 4.0 * np.sin(np.arange(n) / 18.0) + rng.normal(0, 0.3, n)
    tendencia = 100 + np.cumsum(rng.normal(0.01, 0.4, n))
    casos = {
        'Ichimoku': tendencia,
        'TTM Squeeze': tendencia,
        'Keltner': oscilante,
    }
    for nombre, c in casos.items():
        df = pd.DataFrame({'timestamp': ts, 'open': c, 'close': c,
                           'high': c + 0.5, 'low': c - 0.5, 'volume': 1000.0})
        p = params_por_defecto(nombre)
        s = generar_senales(nombre, df, p)
        assert s['entradas_long'].sum() + s['entradas_short'].sum() > 0, nombre
        assert len(filas_plantilla(nombre, p)) >= 1, nombre


def test_plantillas_direccion_restringe_lado(df):
    for nombre in ('Supertrend', 'MACD', 'ADX (fuerza de tendencia)',
                   'Aroon', 'CMO', 'TRIX', 'StochRSI',
                   'Ichimoku', 'Keltner', 'TTM Squeeze'):
        p = params_por_defecto(nombre)
        p['direccion'] = 'Long'
        s = generar_senales(nombre, df, p)
        assert not s['entradas_short'].any(), nombre
        assert not s['salidas_short'].any(), nombre


def test_registro_tiene_las_diez_plantillas_nuevas():
    # crossover/tendencia: el giro es la salida -> sin stop ATR por defecto
    sin_stop = ('Supertrend', 'MACD', 'ADX (fuerza de tendencia)',
                'Aroon', 'TRIX', 'Ichimoku', 'TTM Squeeze')
    # reversión a la media: salida por umbral -> stop ATR normal
    con_stop = ('CMO', 'StochRSI', 'Keltner')
    for nombre in sin_stop + con_stop:
        assert nombre in ESTRATEGIAS
        assert 'generar' in ESTRATEGIAS[nombre]
    for nombre in sin_stop:
        assert ESTRATEGIAS[nombre]['defaults_setup'] == {'stop_atr': 0.0}, nombre
