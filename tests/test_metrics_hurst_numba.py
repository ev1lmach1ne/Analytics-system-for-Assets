import numpy as np

from core.metrics import hurst_rs_numba, calcular_hurst_array

LAGS = np.array([16, 32, 64, 128, 256], dtype=np.int64)


def _hurst_medio(retornos, ventana=1024, paso=10, lags=LAGS):
    vals = calcular_hurst_array(retornos, ventana, paso, lags)
    vals = vals[~np.isnan(vals)]
    assert len(vals) > 0
    return vals.mean()


def test_hurst_random_walk_cerca_de_0_5():
    rng = np.random.default_rng(seed=42)
    retornos = rng.normal(0, 1, 5000)
    media = _hurst_medio(retornos)
    assert 0.40 <= media <= 0.60


def _retornos_ar1(phi, n, rng):
    # El Hurst mide autocorrelación, no la media: un simple drift no la genera
    # porque el algoritmo resta la media local de cada bloque antes de acumular.
    # Para simular persistencia real (momentum) hace falta autocorrelación genuina.
    eps = rng.normal(0, 1, n)
    r = np.empty(n)
    r[0] = eps[0]
    for i in range(1, n):
        r[i] = phi * r[i - 1] + eps[i]
    return r


def test_hurst_tendencia_mayor_que_random_walk():
    rng = np.random.default_rng(seed=42)
    retornos_ruido = rng.normal(0, 1, 5000)
    retornos_tendencia = _retornos_ar1(phi=0.3, n=5000, rng=rng)

    media_ruido = _hurst_medio(retornos_ruido)
    media_tendencia = _hurst_medio(retornos_tendencia)

    assert media_tendencia > media_ruido


def test_hurst_rs_numba_rango_valido():
    # Serie normal
    rng = np.random.default_rng(seed=1)
    h = hurst_rs_numba(rng.normal(0, 1, 512), LAGS)
    assert 0.0 <= h <= 1.0


def test_hurst_rs_numba_fallback_den_cero():
    # Lags todos iguales -> log_lags constante -> den == 0.0 -> fallback 0.5
    lags_iguales = np.array([16, 16, 16], dtype=np.int64)
    rng = np.random.default_rng(seed=1)
    h = hurst_rs_numba(rng.normal(0, 1, 512), lags_iguales)
    assert h == 0.5
