"""ZigZag de pivotes confirmados: _pivotes_candidatos, _zigzag_pivotes y
_zigzag_series."""
import numpy as np
import pandas as pd

from core.strategies import (
    _pivotes_candidatos, _serie_indicador, _zigzag_eventos, _zigzag_pivotes,
    _zigzag_series,
)


def _df(close):
    close = np.asarray(close, dtype=np.float64)
    n = len(close)
    return pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=n, freq='1h', tz='UTC'),
        'open': close, 'close': close, 'high': close, 'low': close,
    })


def _sierra(n_tramos=6, largo=20, base=100.0, amplitud=20.0):
    """Dientes de sierra limpios: tramos alternos de subida y bajada."""
    tramos = []
    nivel = base
    for k in range(n_tramos):
        destino = base + (amplitud if k % 2 == 0 else 0.0)
        tramos.append(np.linspace(nivel, destino, largo, endpoint=False))
        nivel = destino
    return np.concatenate(tramos)


def test_un_candidato_se_confirma_der_velas_despues_de_ocurrir():
    close = np.array([1.0, 2.0, 5.0, 2.0, 1.0])
    candidatos, der = _pivotes_candidatos(close, close, piernas=4)
    assert der == 2
    altos = [c for c in candidatos if c[3] == 1]
    idx_piv, idx_conf, precio, _ = altos[0]
    assert (idx_piv, precio) == (2, 5.0)
    assert idx_conf == idx_piv + der   # nunca en la vela del propio pivote


def test_piernas_minimas_usan_una_vela_a_cada_lado():
    close = np.array([1.0, 3.0, 1.0])
    candidatos, der = _pivotes_candidatos(close, close, piernas=2)
    assert der == 1
    assert (1, 2, 3.0, 1) in candidatos


def test_la_sierra_produce_pivotes_alternados():
    pivotes = _zigzag_pivotes(_df(_sierra()), desviacion_pct=5.0, piernas=10)
    tipos = [p[3] for p in pivotes]
    assert len(tipos) >= 4
    assert all(a != b for a, b in zip(tipos, tipos[1:]))


def test_un_movimiento_por_debajo_de_la_desviacion_no_crea_pivote():
    """Misma serie, dos umbrales: con la desviación por encima de la amplitud
    de la sierra no puede confirmarse ningún tramo nuevo."""
    df = _df(_sierra(amplitud=20.0))   # ~20% de recorrido sobre base 100
    sueltos = _zigzag_pivotes(df, desviacion_pct=5.0, piernas=10)
    exigente = _zigzag_pivotes(df, desviacion_pct=50.0, piernas=10)
    assert len(sueltos) > len(exigente)
    assert len(exigente) == 1   # solo sobrevive el primer pivote, sin tramos


def test_dos_candidatos_del_mismo_tipo_dejan_solo_el_mas_extremo():
    # dos máximos consecutivos sin mínimo confirmado entre medias
    close = np.array([1.0, 2.0, 8.0, 6.0, 7.0, 12.0, 6.0, 5.0, 4.0, 3.0, 2.0])
    pivotes = _zigzag_pivotes(_df(close), desviacion_pct=1.0, piernas=4)
    altos = [p for p in pivotes if p[3] == 1]
    assert 8.0 not in [p[2] for p in altos]   # el 8 quedó absorbido por el 12
    assert 12.0 in [p[2] for p in altos]


def test_el_registro_de_eventos_solo_crece_al_alargar_el_historico():
    """La propiedad que hace el indicador utilizable en backtest: al añadir
    velas por la derecha, el registro de lo ya ocurrido no se reescribe — solo
    se le añaden eventos nuevos al final. Nada del pasado cambia ni desaparece.
    """
    close = _sierra(n_tramos=8, largo=15)
    df_total = _df(close)
    previos = []
    for k in range(20, len(close) + 1):
        eventos = _zigzag_eventos(df_total.iloc[:k], desviacion_pct=5.0, piernas=10)
        assert eventos[:len(previos)] == previos
        previos = eventos
    assert len(previos) >= 4


def test_la_serie_conserva_el_pivote_que_luego_es_sustituido():
    """Un pivote sustituido por otro más extremo estuvo realmente vigente hasta
    la sustitución: la serie causal debe recordarlo, aunque la polilínea final
    que se dibuja ya no lo incluya."""
    close = np.array([1.0, 2.0, 8.0, 6.0, 7.0, 12.0, 6.0, 5.0, 4.0, 3.0, 2.0])
    df = _df(close)
    precio, _tipo, _ev = _zigzag_series(df, desviacion_pct=1.0, piernas=4)
    eventos = _zigzag_eventos(df, desviacion_pct=1.0, piernas=4)
    sustituido = next(e for e in eventos if e[2] == 8.0)
    sustituto = next(e for e in eventos if e[2] == 12.0)
    assert sustituto[4] is True                       # efectivamente sustituye
    assert precio[sustituido[1]] == 8.0               # vigente en su momento
    assert precio[sustituto[1]] == 12.0               # y sustituido después
    # en la polilínea final, en cambio, el 8 ya no está
    assert 8.0 not in [p[2] for p in _zigzag_pivotes(df, 1.0, 4)]


def test_la_serie_propaga_el_ultimo_pivote_hasta_el_siguiente():
    df = _df(_sierra())
    precio, tipo, eventos = _zigzag_series(df, desviacion_pct=5.0, piernas=10)
    primero, segundo = eventos[0], eventos[1]
    assert np.isnan(precio[:primero[1]]).all()   # nada antes de la 1ª confirmación
    # entre confirmaciones el valor se mantiene constante
    assert np.all(precio[primero[1]:segundo[1]] == primero[2])
    assert np.all(tipo[primero[1]:segundo[1]] == primero[3])


def test_zigzag_disponible_como_indicador_de_regla():
    df = _df(_sierra())
    serie = _serie_indicador(df, {'tipo': 'ZIGZAG', 'periodo': 10})
    esperado, _tp, _pv = _zigzag_series(df, 5.0, 10)
    assert np.array_equal(np.nan_to_num(serie), np.nan_to_num(esperado))
