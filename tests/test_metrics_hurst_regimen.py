import pandas as pd

from core.metrics import contar_regimen_hurst


def test_contar_regimen_hurst_conteos_exactos():
    serie = pd.Series([0.6, 0.6, 0.55, 0.55, 0.3, 0.3])
    r = contar_regimen_hurst(serie)
    assert r == {'total_tendencia': 2, 'total_aleatorio': 2, 'total_reversion': 2}


def test_contar_regimen_hurst_casos_borde():
    # 0.58 y 0.52 son los límites: > / >= / <= / < tal como en el código original
    serie = pd.Series([0.58, 0.52, 0.581, 0.519])
    r = contar_regimen_hurst(serie)
    # 0.58 -> aleatorio (no es > 0.58); 0.52 -> aleatorio (es >= 0.52);
    # 0.581 -> tendencia; 0.519 -> reversion
    assert r['total_tendencia'] == 1
    assert r['total_aleatorio'] == 2
    assert r['total_reversion'] == 1
