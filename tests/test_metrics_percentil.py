"""Percentil rodante (calcular_percentil_rodante_numba), base del filtro de
volatilidad del constructor."""
import numpy as np

from core.metrics import calcular_percentil_rodante_numba


def test_warm_up_en_nan_hasta_tener_ventana_completa():
    serie = np.arange(10, dtype=np.float64)
    out = calcular_percentil_rodante_numba(serie, 5)
    assert np.isnan(out[:4]).all()
    assert not np.isnan(out[4:]).any()


def test_serie_creciente_el_actual_siempre_es_el_maximo():
    serie = np.arange(10, dtype=np.float64)
    out = calcular_percentil_rodante_numba(serie, 5)
    # 4 menores + medio empate consigo mismo, de 5
    assert np.allclose(out[4:], 90.0)


def test_serie_decreciente_el_actual_siempre_es_el_minimo():
    serie = np.arange(10, 0, -1).astype(np.float64)
    out = calcular_percentil_rodante_numba(serie, 5)
    assert np.allclose(out[4:], 10.0)


def test_serie_constante_es_percentil_medio():
    """El empate a mitad de rango es lo que evita que un tramo plano se lea
    como volatilidad extrema."""
    serie = np.full(8, 7.0)
    out = calcular_percentil_rodante_numba(serie, 4)
    assert np.allclose(out[3:], 50.0)


def test_valor_intermedio_da_percentil_exacto():
    serie = np.array([10.0, 20.0, 30.0, 40.0, 25.0])
    out = calcular_percentil_rodante_numba(serie, 5)
    # 25 supera a 10 y 20; medio empate consigo mismo -> (2 + 0.5) de 5
    assert out[4] == 50.0


def test_los_nan_dentro_de_la_ventana_se_ignoran():
    serie = np.array([1.0, np.nan, 3.0, 2.0])
    out = calcular_percentil_rodante_numba(serie, 4)
    # ventana efectiva {1, 3, 2}: 2 supera a 1, medio empate consigo mismo
    assert out[3] == 100.0 * 1.5 / 3


def test_nan_en_la_vela_actual_no_produce_percentil():
    serie = np.array([1.0, 2.0, 3.0, np.nan])
    out = calcular_percentil_rodante_numba(serie, 4)
    assert np.isnan(out[3])
