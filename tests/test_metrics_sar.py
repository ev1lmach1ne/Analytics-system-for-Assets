"""Parabolic SAR de Wilder (calcular_sar_numba)."""
import numpy as np

from core.metrics import calcular_sar_numba

AF_I, AF_P, AF_M = 0.02, 0.02, 0.2


def _sar(high, low, af_i=AF_I, af_p=AF_P, af_m=AF_M):
    return calcular_sar_numba(np.asarray(high, dtype=np.float64),
                              np.asarray(low, dtype=np.float64), af_i, af_p, af_m)


def test_serie_vacia_no_revienta():
    sar, tend = _sar([], [])
    assert len(sar) == 0 and len(tend) == 0


def test_tendencia_alcista_sostenida_no_gira():
    n = 60
    high = 100.0 + np.arange(n) * 1.0
    low = high - 0.5
    sar, tend = _sar(high, low)
    assert (tend == 1).all()
    assert (sar[1:] < low[1:]).all()   # en tendencia alcista el SAR va por debajo


def test_tendencia_bajista_sostenida_mantiene_el_sar_por_encima():
    n = 60
    low = 100.0 - np.arange(n) * 1.0
    high = low + 0.5
    # arranca asumiendo alcista: gira en la vela 1 y ya no vuelve
    sar, tend = _sar(high, low)
    assert (tend[1:] == -1).all()
    assert (sar[2:] > high[2:]).all()


def test_el_af_topa_en_af_max():
    """Con nuevos máximos en cada vela el AF sube AF_P por vela hasta topar.
    Una vez topado, el SAR se acerca al precio cerrando una fracción fija
    af_max de la distancia restante: la distancia decae geométricamente con
    razón (1 - af_max), y esa razón ES el tope."""
    n = 80
    high = 100.0 + np.arange(n) * 1.0
    low = high - 0.5
    for af_max in (0.1, 0.2, 0.4):
        sar, _ = _sar(high, low, af_m=af_max)
        # ventana ya topada (el AF llega al tope antes de la vela 20) pero
        # todavía lejos de que la distancia se pierda en el epsilon del float
        saltos = np.diff(high - sar)[22:32]
        razones = saltos[1:] / saltos[:-1]
        assert np.allclose(razones, 1.0 - af_max, rtol=1e-5)


def test_v_de_precio_produce_exactamente_un_giro():
    subida = np.arange(30, dtype=np.float64)
    bajada = np.arange(28, -1, -1, dtype=np.float64)
    high = 100.0 + np.concatenate([subida, bajada])
    low = high - 0.5
    _, tend = _sar(high, low)
    giros = np.where(np.diff(tend) != 0)[0]
    assert len(giros) == 1
    assert giros[0] > 30   # el giro llega después del techo, no antes


def test_al_girar_el_sar_salta_al_extremo_de_la_tendencia_previa():
    subida = np.arange(30, dtype=np.float64)
    bajada = np.arange(28, -1, -1, dtype=np.float64)
    high = 100.0 + np.concatenate([subida, bajada])
    low = high - 0.5
    sar, tend = _sar(high, low)
    i = int(np.where(np.diff(tend) != 0)[0][0]) + 1
    assert sar[i] == high[:i].max()


def test_el_sar_alcista_nunca_entra_en_el_rango_de_las_dos_velas_previas():
    """Con un hueco a la baja, el clamp impide que el SAR se sitúe por encima
    del mínimo de las dos velas anteriores (giraría por un movimiento pasado)."""
    high = np.array([100.0, 105.0, 110.0, 120.0, 121.0, 121.5, 122.0])
    low = np.array([99.0, 104.0, 109.0, 119.0, 110.0, 120.5, 121.0])
    sar, tend = _sar(high, low)
    for i in range(2, len(high)):
        if tend[i] > 0 and tend[i - 1] > 0:
            assert sar[i] <= min(low[i - 1], low[i - 2]) + 1e-12


def test_af_max_bajo_hace_el_sar_mas_lento():
    n = 60
    high = 100.0 + np.arange(n) * 1.0
    low = high - 0.5
    sar_lento, _ = _sar(high, low, af_m=0.05)
    sar_rapido, _ = _sar(high, low, af_m=0.5)
    # un AF mayor acerca el SAR al precio más deprisa
    assert (sar_rapido[-1] - sar_lento[-1]) > 0
