"""Utilidades de trading portadas del estudio de jesse: Kelly criterion,
cointegración de pares (Engle-Granger) y regresión alpha/beta."""
import numpy as np
import pytest

from core.metrics import (
    kelly_criterion, cointegracion_pares, calculate_alpha_beta,
)


# ── Kelly ─────────────────────────────────────────────────────────
def test_kelly_formula_clasica():
    # f = p - (1-p)/b ; con 50% de aciertos y payoff 2:1 -> 0.25
    assert kelly_criterion(0.5, 2.0) == pytest.approx(0.25)
    assert kelly_criterion(0.6, 1.0) == pytest.approx(0.2)


def test_kelly_sin_edge_cero_o_negativo():
    assert kelly_criterion(0.5, 1.0) == pytest.approx(0.0)
    assert kelly_criterion(0.3, 1.0) < 0.0


def test_kelly_datos_invalidos_none():
    assert kelly_criterion(None, 2.0) is None
    assert kelly_criterion(0.5, 0.0) is None
    assert kelly_criterion(0.5, -1.0) is None
    assert kelly_criterion(1.0, 2.0) is None     # win_rate extremo


# ── Alpha / Beta ──────────────────────────────────────────────────
def test_alpha_beta_recupera_los_parametros():
    rng = np.random.default_rng(5)
    bench = rng.normal(0.001, 0.01, 2000)
    act = 0.0002 + 1.5 * bench + rng.normal(0, 0.001, 2000)
    res = calculate_alpha_beta(act, bench, rf=0.0, periodos_anio=252)
    assert res['beta'] == pytest.approx(1.5, abs=0.1)
    assert res['alpha'] == pytest.approx(0.0002 * 252, abs=0.05)
    assert res['r2'] == pytest.approx(0.99, abs=0.01)


def test_alpha_beta_con_rf():
    rng = np.random.default_rng(6)
    bench = rng.normal(0.001, 0.01, 2000)
    act = bench + rng.normal(0, 0.001, 2000)   # beta 1, alpha 0
    res = calculate_alpha_beta(act, bench, rf=0.05 / 252, periodos_anio=252)
    # con rf_periodo aplicada, el alpha vuelve a ~0 (el activo no supera rf)
    assert res['alpha'] == pytest.approx(0.0, abs=0.01)


def test_alpha_beta_serie_corta_none():
    res = calculate_alpha_beta([1.0], [1.0])
    assert res['beta'] is None and res['alpha'] is None


# ── Cointegración ─────────────────────────────────────────────────
def test_par_cointegrado_detectado():
    rng = np.random.default_rng(8)
    n = 800
    x = np.cumsum(rng.normal(0, 1, n)) + 0.01 * np.arange(n)
    y = 2.0 * x + rng.normal(0, 0.5, n)        # residuo estacionario
    r = cointegracion_pares(x, y)
    assert r['cointegrados'] is True
    assert r['p_value'] is not None and r['p_value'] < 0.05


def test_paseos_independientes_menos_cointegracion_que_el_par_real():
    """Dos paseos aleatorios independientes PUEDEN dar falsos positivos a
    veces (es la naturaleza del test), pero el par verdaderamente cointegrado
    siempre tiene un p-value muchísimo menor que el par independiente."""
    rng = np.random.default_rng(9)
    n = 800
    x = np.cumsum(rng.normal(0, 1, n))
    z = np.cumsum(rng.normal(0, 1, n))
    y = 2.0 * x + rng.normal(0, 0.5, n)
    r_ind = cointegracion_pares(x, z)
    r_par = cointegracion_pares(x, y)
    assert r_par['p_value'] is not None and r_ind['p_value'] is not None
    assert r_par['p_value'] < r_ind['p_value']


def test_cointegracion_datos_insuficientes():
    r = cointegracion_pares([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert r['cointegrados'] is None
