import numpy as np
import pytest

from core.metrics import (
    conditional_value_at_risk, autocorr_penalty, sharpe_smart, serenity_index,
    sortino_ratio, omega_ratio, calmar_ratio,
)


# ── CVaR (Expected Shortfall) ──────────────────────────────────────
def test_cvar_es_media_de_la_cola():
    serie = np.array([-10.0, -8.0, -6.0, -4.0, 1.0, 2.0, 3.0, 4.0,
                      5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                      13.0, 14.0, 15.0, 16.0])   # 20 elem -> peor 5% = 1
    assert conditional_value_at_risk(serie, 0.95) == pytest.approx(-10.0)
    # con 20 elem, el peor 1% no llega a un elemento: cae al mínimo
    assert conditional_value_at_risk(serie, 0.99) == pytest.approx(-10.0)


def test_cvar_cola_pequena_devuelve_el_minimo():
    assert conditional_value_at_risk([-7.0, -3.0, 1.0, 2.0], 0.95) \
        == pytest.approx(-7.0)


def test_cvar_vacio_nan():
    assert np.isnan(conditional_value_at_risk([], 0.95))


def test_cvar_magnitud_no_menor_que_su_umbral():
    rng = np.random.default_rng(11)
    ret = rng.normal(0, 1, 5000)
    umbral = np.percentile(ret, 5)
    cvar = conditional_value_at_risk(ret, 0.95)
    assert cvar <= umbral        # la media de la cola es peor que su borde


# ── Penalización por autocorrelación ───────────────────────────────
def test_autocorr_penalty_ruido_blanco_aprox_uno():
    rng = np.random.default_rng(7)
    wn = rng.normal(0, 1, 5000)
    assert autocorr_penalty(wn) == pytest.approx(1.0, abs=0.1)


def test_autocorr_penalty_serie_correlada_mayor_uno():
    rng = np.random.default_rng(7)
    x = np.sin(np.arange(2000) * 0.15) + 0.3 * rng.normal(0, 1, 2000)
    assert autocorr_penalty(x) > 1.2


def test_autocorr_penalty_serie_corta_uno():
    assert autocorr_penalty([1.0, 2.0]) == 1.0


# ── Sharpe smart ───────────────────────────────────────────────────
def test_sharpe_smart_anualiza_y_penaliza():
    rng = np.random.default_rng(1)
    wn = rng.normal(0.0005, 0.01, 2000)
    base = sharpe_smart(wn, 252)
    assert base is not None and np.isfinite(base)
    # serie con fuerte autocorrelación positiva -> penalización grande
    ar = 0.9 * np.roll(wn, 1) + 0.1 * wn
    ar[0] = wn[0]
    smart_ar = sharpe_smart(ar, 252)
    assert smart_ar is not None
    assert smart_ar < base


# ── Serenity Index ─────────────────────────────────────────────────
def test_serenity_plano_none():
    assert serenity_index([0.001] * 100) is None


def test_serenity_tendencia_finito():
    rng = np.random.default_rng(3)
    ret = rng.normal(0.0004, 0.002, 5000)
    s = serenity_index(ret)
    assert s is not None and np.isfinite(s)


def test_serenity_serie_perdedora_negativo():
    rng = np.random.default_rng(4)
    ret = rng.normal(-0.0004, 0.002, 5000)
    s = serenity_index(ret)
    assert s is not None and s < 0


# ── Sortino / Calmar / Omega ────────────────────────────────────────
def test_sortino_solo_castiga_las_perdidas():
    rng = np.random.default_rng(12)
    ret = rng.normal(0.0003, 0.005, 3000)
    s = sortino_ratio(ret, velas_por_anio=252)
    assert s is not None and s > 0


def test_sortino_sin_perdidas_none():
    assert sortino_ratio(np.full(50, 0.001), 252) is None


def test_sortino_anualiza():
    rng = np.random.default_rng(13)
    ret = rng.normal(0.0003, 0.005, 3000)
    por_periodo = sortino_ratio(ret)
    anualizado = sortino_ratio(ret, velas_por_anio=252)
    assert anualizado is not None and por_periodo is not None
    assert anualizado == pytest.approx(por_periodo * np.sqrt(252), rel=1e-6)


def test_omega_ganancias_sobre_perdidas():
    # serie con más ganancia que pérdida agregada -> omega > 1
    ret = np.array([0.01, -0.005, 0.02, -0.003, 0.015, -0.004])
    assert omega_ratio(ret) == pytest.approx(
        (0.01 + 0.02 + 0.015) / (0.005 + 0.003 + 0.004))


def test_omega_sin_perdidas_none():
    assert omega_ratio(np.full(30, 0.001)) is None


def test_calmar_retorno_entre_drawdown():
    assert calmar_ratio(15.0, -10.0) == pytest.approx(1.5)
    assert calmar_ratio(None, -10.0) is None
    assert calmar_ratio(15.0, 0.0) is None
