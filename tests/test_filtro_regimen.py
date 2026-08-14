"""El filtro de régimen ER/Hurst aplicado de verdad.

Los tests de core/metrics cubren las primitivas (ER=1 en serie monótona,
Hurst≈0.5 en paseo aleatorio); lo que aquí se fija es el cableado, que es donde
estaban los fallos y donde no había ninguna red:

- los umbrales son ABSOLUTOS (0.5/0.3), no la media±σ de la propia serie: con
  los adaptativos, "tendencia" significaba un número distinto en cada activo
  (0.16 en una prueba) y además dependía de datos futuros;
- cuando el filtro no se puede aplicar hay que AVISAR: antes se saltaba en
  silencio y el backtest salía idéntico a uno sin filtro.
"""
import numpy as np
import pandas as pd
import pytest

from core.strategies import (
    generar_senales_sistema, _er_serie, _hurst_serie,
    UMBRAL_ER_TENDENCIA, UMBRAL_ER_RUIDO, UMBRAL_HURST_TENDENCIA,
    VENTANA_ER_DEFECTO, VENTANA_HURST_DEFECTO, VENTANA_HURST_MINIMA,
    ventana_regimen_defecto, _filtros_por_defecto,
    _normalizar_metodo_regimen,
)

N = 3000


@pytest.fixture(scope='module')
def df():
    close = 100 + np.cumsum(np.random.default_rng(11).normal(0, .5, N))
    return pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=N, freq='1h', tz='UTC'),
        'open': close, 'close': close, 'high': close + .5, 'low': close - .5})


def _correr(df, regimen=None, nombre='S'):
    setup = {'nombre': nombre, 'plantilla': 'Cruce de medias',
             'params': {'rapida': 10, 'lenta': 30}}
    if regimen is not None:
        setup['filtros'] = {'regimen': regimen}
    r = generar_senales_sistema(df, [setup])
    entradas = np.flatnonzero(r['entradas_long'] | r['entradas_short'])
    return entradas, r['avisos']


def test_el_filtro_er_recorta_y_todas_las_entradas_cumplen_el_umbral(df):
    sin_filtro, _ = _correr(df)
    con_filtro, avisos = _correr(
        df, {'metodo': 'er_tendencia', 'periodo': VENTANA_ER_DEFECTO})
    assert 0 < len(con_filtro) < len(sin_filtro)
    assert avisos == []

    er = _er_serie(df['close'].values, VENTANA_ER_DEFECTO).values
    assert (er[con_filtro] > UMBRAL_ER_TENDENCIA).all(), \
        "hay entradas en velas que el filtro decía no admitir"


def test_el_filtro_er_rango_usa_el_otro_extremo(df):
    con_filtro, _ = _correr(
        df, {'metodo': 'er_rango', 'periodo': VENTANA_ER_DEFECTO})
    er = _er_serie(df['close'].values, VENTANA_ER_DEFECTO).values
    assert len(con_filtro) > 0
    assert (er[con_filtro] < UMBRAL_ER_RUIDO).all()


def test_el_filtro_hurst_recorta_y_cumple_su_umbral(df):
    sin_filtro, _ = _correr(df)
    con_filtro, avisos = _correr(
        df, {'metodo': 'hurst_tendencia', 'periodo': VENTANA_HURST_DEFECTO})
    assert 0 < len(con_filtro) < len(sin_filtro)
    assert avisos == []

    h = _hurst_serie(df['close'].values, VENTANA_HURST_DEFECTO)
    assert (h[con_filtro] > UMBRAL_HURST_TENDENCIA).all()


@pytest.mark.parametrize('metodo', ['er_tendencia', 'hurst_tendencia'])
def test_una_ventana_que_no_cabe_avisa_en_vez_de_ignorarse(df, metodo):
    """El fallo que motivó todo esto: con la ventana mayor que el histórico el
    resultado era idéntico a no filtrar, sin nada que lo delatara."""
    sin_filtro, _ = _correr(df)
    con_filtro, avisos = _correr(df, {'metodo': metodo, 'periodo': N * 2})
    assert len(con_filtro) == len(sin_filtro), \
        "el filtro no puede aplicarse; lo que se exige es que lo DIGA"
    assert len(avisos) == 1
    assert 'NO aplicado' in avisos[0]
    assert 'S:' in avisos[0], "el aviso debe decir de qué setup habla"


def test_hurst_con_ventana_corta_avisa_de_que_no_es_fiable(df):
    """Por debajo de VENTANA_HURST_MINIMA el estimador supera 0.58 incluso
    sobre ruido puro: el filtro se aplica, pero lo que marca no significa lo
    que dice."""
    _entradas, avisos = _correr(
        df, {'metodo': 'hurst_tendencia', 'periodo': VENTANA_HURST_MINIMA // 4})
    assert len(avisos) == 1
    assert 'no es fiable' in avisos[0]


def test_los_umbrales_no_dependen_de_cuantos_datos_futuros_haya(df):
    """Sin look-ahead: el régimen de una vela no puede cambiar porque el
    histórico siga más allá. Blinda contra volver a umbrales adaptativos."""
    mitad = N // 2
    ent_completo, _ = _correr(df, {'metodo': 'er_tendencia',
                                   'periodo': VENTANA_ER_DEFECTO})
    ent_mitad, _ = _correr(df.iloc[:mitad].copy(),
                           {'metodo': 'er_tendencia',
                            'periodo': VENTANA_ER_DEFECTO})
    # las señales de la primera mitad tienen que ser las mismas vistas desde
    # el histórico corto y desde el largo (salvo el borde del último cruce,
    # que necesita velas posteriores para confirmarse)
    assert set(ent_mitad) <= set(ent_completo)


def test_sin_filtro_no_hay_avisos(df):
    _entradas, avisos = _correr(df, None)
    assert avisos == []


def test_ventana_por_defecto_segun_metodo():
    """ER y Hurst no comparten escala: 10 deja a Hurst inservible y 400 deja
    al ER plano."""
    assert ventana_regimen_defecto('er_tendencia') == VENTANA_ER_DEFECTO
    assert ventana_regimen_defecto('er_rango') == VENTANA_ER_DEFECTO
    assert ventana_regimen_defecto('hurst_tendencia') == VENTANA_HURST_DEFECTO
    assert ventana_regimen_defecto('hurst_reversion') == VENTANA_HURST_DEFECTO
    assert VENTANA_ER_DEFECTO == 10, "el valor de Kaufman para el ER del AMA"
    assert _filtros_por_defecto()['regimen']['periodo'] == VENTANA_ER_DEFECTO


def test_el_identificador_antiguo_con_tilde_se_normaliza():
    """La GUI guardaba 'hurst_reversión' (con tilde); el core solo conoce
    'hurst_reversion'. El alias debe normalizarse en el motor, el
    pseudocódigo y la carga de setups."""
    assert _normalizar_metodo_regimen('hurst_reversión') == 'hurst_reversion'
    assert _normalizar_metodo_regimen('hurst_reversion') == 'hurst_reversion'
    assert _normalizar_metodo_regimen('er_tendencia') == 'er_tendencia'
    assert _normalizar_metodo_regimen(None) is None


def test_el_motor_aplica_el_filtro_hurst_con_el_identificador_antiguo(df):
    """Un setup guardado por una versión antigua con 'hurst_reversión' debe
    seguir filtrando (y no caer en silencio a «sin filtro»)."""
    con_canonico, avisos_c = _correr(
        df, {'metodo': 'hurst_reversion', 'periodo': VENTANA_HURST_DEFECTO})
    con_antiguo, avisos_a = _correr(
        df, {'metodo': 'hurst_reversión', 'periodo': VENTANA_HURST_DEFECTO})
    assert avisos_c == []
    assert avisos_a == []
    assert len(con_canonico) == len(con_antiguo)
    assert len(con_antiguo) < len(_correr(df)[0])


def test_patrones_conserva_sus_umbrales_adaptativos():
    """preparar_contexto sin `umbrales_er` tiene que seguir comportándose como
    siempre: la pestaña Patrones enseña ESE umbral en su desplegable y sus
    estadísticas ya publicadas no deben moverse."""
    from core.candle_patterns import preparar_contexto
    from core.metrics import calcular_umbrales_er

    close = 100 + np.cumsum(np.random.default_rng(3).normal(0, .5, 500))
    er = _er_serie(close, 10).values
    ctx = preparar_contexto(close, er=er)
    esperado = calcular_umbrales_er(pd.Series(er))
    assert ctx['umbrales_er'] == (esperado['umbral_ruido'],
                                  esperado['umbral_tendencia'])

    ctx_fijo = preparar_contexto(close, er=er,
                                 umbrales_er=(UMBRAL_ER_RUIDO, UMBRAL_ER_TENDENCIA))
    assert ctx_fijo['umbrales_er'] == (UMBRAL_ER_RUIDO, UMBRAL_ER_TENDENCIA)
    assert (ctx_fijo['regimen_er'] == 2).sum() == int((er > UMBRAL_ER_TENDENCIA).sum())
