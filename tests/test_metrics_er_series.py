import numpy as np
import pandas as pd
import pytest

from core.metrics import calcular_er_series


def test_er_uno_en_serie_monotona():
    # Retornos todos positivos → movimiento neto == movimiento total → ER = 1
    retornos = pd.Series([0.01] * 50)
    er = calcular_er_series(retornos, 10)
    assert er.iloc[-1] == pytest.approx(1.0)
    assert er.iloc[10:].to_numpy() == pytest.approx(np.ones(40))


def test_er_cero_en_zigzag_perfecto():
    # +r, -r alternos → movimiento neto 0 en ventana par → ER = 0
    retornos = pd.Series([0.01, -0.01] * 25)
    er = calcular_er_series(retornos, 10)
    assert er.iloc[-1] == pytest.approx(0.0, abs=1e-9)


def test_er_longitud_y_arranque():
    retornos = pd.Series(np.random.default_rng(7).normal(0, 0.01, 100))
    er = calcular_er_series(retornos, 14)
    assert len(er) == 100
    # Antes de completar la primera ventana el rolling es NaN → fillna(0)
    assert (er.iloc[:13] == 0).all()
    assert not er.isna().any()


def test_er_acotado_0_1():
    retornos = pd.Series(np.random.default_rng(11).normal(0, 0.01, 500))
    er = calcular_er_series(retornos, 20)
    assert (er >= 0).all() and (er <= 1).all()
