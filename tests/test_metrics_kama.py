import numpy as np
import pytest

from core.metrics import calcular_kama_numba


def test_kama_precio_constante():
    close = np.full(100, 50.0)
    er = np.full(100, 0.5)
    kama = calcular_kama_numba(close, er, 2.0, 30.0)
    assert kama == pytest.approx(close)


def test_kama_er_uno_converge_rapido():
    # Con ER=1, SC = (2/(fast+1))^2; tras un salto de precio el KAMA debe
    # acercarse al nuevo nivel mucho más rápido que con ER=0.
    close = np.concatenate([np.full(50, 100.0), np.full(50, 110.0)])
    er_rapido = np.ones(100)
    er_lento = np.zeros(100)
    kama_r = calcular_kama_numba(close, er_rapido, 2.0, 30.0)
    kama_l = calcular_kama_numba(close, er_lento, 2.0, 30.0)
    assert abs(kama_r[-1] - 110.0) < abs(kama_l[-1] - 110.0)
    assert kama_r[-1] == pytest.approx(110.0, abs=0.1)


def test_kama_nan_en_er_no_propaga():
    close = np.linspace(100, 110, 60)
    er = np.full(60, np.nan)
    kama = calcular_kama_numba(close, er, 2.0, 30.0)
    assert not np.isnan(kama).any()


def test_kama_nan_iniciales_en_close_no_propagan():
    # Las velas de calentamiento de la limpieza llegan con close NaN al inicio:
    # el KAMA debe arrancar en el primer precio válido y no quedarse en NaN.
    close = np.concatenate([np.full(52, np.nan), np.full(48, 100.0)])
    er = np.full(100, 0.5)
    kama = calcular_kama_numba(close, er, 2.0, 30.0)
    assert np.isnan(kama[:52]).all()
    assert not np.isnan(kama[52:]).any()
    assert kama[-1] == pytest.approx(100.0)


def test_kama_arranca_en_primer_precio():
    close = np.array([42.0, 43.0, 44.0])
    er = np.array([0.5, 0.5, 0.5])
    kama = calcular_kama_numba(close, er, 2.0, 30.0)
    assert kama[0] == pytest.approx(42.0)


def test_kama_entre_min_max_del_precio():
    rng = np.random.default_rng(3)
    close = 100 + rng.normal(0, 1, 500).cumsum()
    er = rng.uniform(0, 1, 500)
    kama = calcular_kama_numba(close, er, 2.0, 30.0)
    assert kama.min() >= close.min() - 1e-9
    assert kama.max() <= close.max() + 1e-9
