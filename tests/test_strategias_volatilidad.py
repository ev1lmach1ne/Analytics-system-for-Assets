"""Filtro 'volatilidad' del constructor: percentil rodante de ATR / desviación
estándar aplicado a las entradas de un setup.

Ojo con la semántica: el filtro mide volatilidad RELATIVA al pasado reciente,
no absoluta. Un tramo de volatilidad alta pero estable acaba en el percentil
medio en cuanto la ventana de comparación se llena de ese mismo tramo — es el
comportamiento buscado, y por eso los fixtures de abajo usan picos, no
escalones permanentes.
"""
import numpy as np
import pandas as pd

from core.strategies import _filtros_por_defecto, _mascara_filtros_setup

VENTANA = 100
PICO_INI, PICO_FIN = 200, 220


def _df_con_pico(n=400):
    """Volatilidad plana con un pico corto en [PICO_INI, PICO_FIN)."""
    ts = pd.date_range('2024-01-01', periods=n, freq='1h', tz='UTC')
    close = np.full(n, 100.0)
    rango = np.full(n, 0.1)
    rango[PICO_INI:PICO_FIN] = 5.0
    return pd.DataFrame({
        'timestamp': ts, 'open': close, 'close': close,
        'high': close + rango, 'low': close - rango,
    })


def _mascara(df, metodo, ventana=VENTANA, percentil=50.0):
    filtros = _filtros_por_defecto()
    filtros['volatilidad'].update(
        {'metodo': metodo, 'periodo': ventana, 'percentil': percentil})
    m_long, m_short, _ = _mascara_filtros_setup(df, filtros)
    assert (m_long == m_short).all()   # el filtro no distingue dirección
    return m_long


def test_desactivado_no_bloquea_nada():
    assert _mascara(_df_con_pico(), 'ninguno').all()


def test_atr_alto_admite_el_pico_y_bloquea_la_calma():
    m = _mascara(_df_con_pico(), 'atr_percentil_alto', percentil=90.0)
    assert m[PICO_INI + 5:PICO_FIN].all()
    assert not m[VENTANA:PICO_INI].any()


def test_atr_bajo_admite_la_calma_posterior_al_pico():
    """Tras el pico el ATR vuelve al suelo mientras la ventana aún recuerda la
    volatilidad alta: eso es percentil bajo. El corte no puede ser muy
    estricto porque el suelo empata consigo mismo en la mayoría de la ventana,
    y los empates cuentan a mitad de rango. El ATR usa el suavizado de Wilder
    (recursivo), que decae despacio tras el pico: la calma se lee como
    percentil bajo algo más tarde que con una media simple (desde ~260 en
    esta serie en vez de ~240)."""
    m = _mascara(_df_con_pico(), 'atr_percentil_bajo', percentil=40.0)
    assert m[260:PICO_INI + VENTANA].all()
    assert not m[PICO_INI + 5:PICO_FIN].any()


def test_volatilidad_estable_cae_en_el_percentil_medio():
    """Ni 'alto ≥ 60' ni 'bajo ≤ 40' admiten un tramo plano: sin variación no
    hay nada que destacar (los empates cuentan a mitad de rango)."""
    df = _df_con_pico()
    tramo = slice(VENTANA + 20, PICO_INI)
    assert not _mascara(df, 'atr_percentil_alto', percentil=60.0)[tramo].any()
    assert not _mascara(df, 'atr_percentil_bajo', percentil=40.0)[tramo].any()


def test_warm_up_nunca_admite_entrada():
    """Sin ventana completa no hay percentil fiable: no debe pasar ni el corte
    alto ni el bajo (el NaN se mapea a -1, no a 0)."""
    df = _df_con_pico()
    for metodo in ('atr_percentil_alto', 'atr_percentil_bajo'):
        assert not _mascara(df, metodo, percentil=50.0)[:VENTANA - 1].any()


def test_stdev_detecta_el_pico_de_los_cierres():
    """La desviación estándar de retornos reacciona al CIERRE, no al rango de
    la vela: el mismo pico, pero movido a los cierres."""
    n = 400
    ts = pd.date_range('2024-01-01', periods=n, freq='1h', tz='UTC')
    paso = np.full(n, 0.01)
    paso[PICO_INI:PICO_FIN] = 1.0
    close = 100.0 + np.cumsum(paso * np.tile([1.0, -1.0], n // 2))
    df = pd.DataFrame({
        'timestamp': ts, 'open': close, 'close': close,
        'high': close + 0.01, 'low': close - 0.01,
    })
    m = _mascara(df, 'stdev_percentil_alto', percentil=90.0)
    assert m[PICO_INI + 5:PICO_FIN].all()
    assert not m[VENTANA:PICO_INI].any()
