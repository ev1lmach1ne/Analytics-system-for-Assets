import numpy as np
import pandas as pd
import pytest

from core.metrics import calcular_half_life_ou


def _serie_ar1_determinista(phi, n=500, p0=1.0):
    valores = p0 * (phi ** np.arange(n))
    idx = pd.date_range('2020-01-01', periods=n, freq='h')
    return pd.Series(valores, index=idx)


def test_half_life_ar1_determinista():
    phi = 0.9
    serie = _serie_ar1_determinista(phi)
    r = calcular_half_life_ou(serie)

    beta_esperado = phi - 1  # -0.1
    half_life_esperado = -np.log(2) / np.log(phi)

    assert r['beta_hl'] == pytest.approx(beta_esperado, abs=1e-6)
    assert r['half_life_velas'] == pytest.approx(half_life_esperado, rel=1e-6)
    assert r['half_life_dias'] is None  # velas_por_dia no se pasó


def test_half_life_ar1_con_ruido():
    rng = np.random.default_rng(seed=7)
    phi = 0.9
    n = 3000
    valores = np.empty(n)
    valores[0] = 1.0
    ruido = rng.normal(0, 0.01, n)
    for i in range(1, n):
        valores[i] = phi * valores[i - 1] + ruido[i]
    idx = pd.date_range('2020-01-01', periods=n, freq='h')
    serie = pd.Series(valores, index=idx)

    r = calcular_half_life_ou(serie)
    beta_esperado = phi - 1
    assert r['beta_hl'] == pytest.approx(beta_esperado, rel=0.2)
    assert r['half_life_velas'] is not None


def test_half_life_no_reversivo():
    # Tendencia pura (drift positivo fuerte, sin reversión) -> beta_hl >= 0
    idx = pd.date_range('2020-01-01', periods=500, freq='h')
    serie = pd.Series(np.arange(500) * 0.01 + 1.0, index=idx)
    r = calcular_half_life_ou(serie)
    assert r['half_life_velas'] is None
    assert r['half_life_dias'] is None


def test_half_life_conversion_a_dias():
    phi = 0.9
    serie = _serie_ar1_determinista(phi)
    r = calcular_half_life_ou(serie, velas_por_dia=24)
    assert r['half_life_dias'] == pytest.approx(r['half_life_velas'] / 24, rel=1e-9)


def test_half_life_velas_por_dia_none():
    phi = 0.9
    serie = _serie_ar1_determinista(phi)
    r = calcular_half_life_ou(serie, velas_por_dia=None)
    assert r['half_life_dias'] is None
