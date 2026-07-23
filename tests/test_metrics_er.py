import pandas as pd
import pytest

from core.metrics import calcular_umbrales_er


def test_umbrales_er_serie_constante():
    serie = pd.Series([0.5] * 100)
    r = calcular_umbrales_er(serie)
    assert r['er_std'] == pytest.approx(0.0, abs=1e-9)
    assert r['umbral_tendencia'] == pytest.approx(0.5, abs=1e-9)
    assert r['umbral_ruido'] == pytest.approx(0.5, abs=1e-9)


def test_umbrales_er_acotado_superior():
    serie = pd.Series([0.99] * 50 + [0.98] * 50)
    r = calcular_umbrales_er(serie)
    assert r['umbral_tendencia'] <= 0.95


def test_umbrales_er_acotado_inferior():
    serie = pd.Series([0.02] * 50 + [0.03] * 50)
    r = calcular_umbrales_er(serie)
    assert r['umbral_ruido'] >= 0.05
