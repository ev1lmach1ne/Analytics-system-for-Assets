import pytest

from core.config import velas_por_anio


def test_crypto_vs_stock_mismo_tf_intradia():
    # 15min: CRYPTO opera 24h/365d, STOCK solo la sesión (390 min) y 252
    # días de trading -> factores muy distintos para el mismo tamaño de vela.
    # Coincide con la tabla FACTORES ya existente en config.py (15min: 35040
    # crypto, 6552 stock), buena validación cruzada independiente.
    crypto = velas_por_anio('CRYPTO', 15)
    stock = velas_por_anio('STOCK', 15)
    assert crypto == pytest.approx(35040.0)
    assert stock == pytest.approx(6552.0)
    assert crypto > stock


def test_velas_4h_coincide_con_tabla_factores():
    # FACTORES['STOCK']['4h']['anual'] = 409 (390/240 * 252 = 409.5)
    assert velas_por_anio('STOCK', 240) == pytest.approx(409.5)
    # FACTORES['CRYPTO']['4h']['anual'] = 2190 (1440/240 * 365)
    assert velas_por_anio('CRYPTO', 240) == pytest.approx(2190.0)


def test_velas_diarias_una_por_dia_de_trading():
    # Con vela de 1 día (1440 min de calendario entre cierres), el nº de
    # velas/año debe ser simplemente los días de trading al año de cada
    # clase, NO la razón sesión/vela (que daría un valor absurdamente bajo
    # para STOCK: 390/1440*252 ≈ 68, muy lejos de las 252 velas reales).
    assert velas_por_anio('CRYPTO', 1440) == pytest.approx(365.0)
    assert velas_por_anio('STOCK', 1440) == pytest.approx(252.0)
    assert velas_por_anio('FUTURO', 1440) == pytest.approx(252.0)


def test_fallback_tipo_activo_desconocido_usa_24_7():
    # tipo_activo=None (no se pudo determinar la clase del CSV) -> mismo
    # supuesto 24/7/365 que el código tenía antes de esta corrección.
    esperado_24_7 = 525600.0 / 15.0
    assert velas_por_anio(None, 15) == pytest.approx(esperado_24_7)
    assert velas_por_anio('ALGO_DESCONOCIDO', 15) == pytest.approx(esperado_24_7)
