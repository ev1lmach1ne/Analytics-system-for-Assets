"""Filtro «tendencia de TF superior» (multi-timeframe) del constructor de
setups: la serie se resamplea al TF elegido, se calcula la tendencia (SMA/
EMA/Supertrend) y solo se dejan entrar posiciones alineadas con ella.

El invariante central es el SIN LOOKAHEAD: la vela del TF superior que
contiene a la vela base todavía no ha cerrado, así que el valor usado solo
puede venir de la vela HTF anterior (desplazamiento de un bin + ffill).
"""
import numpy as np
import pandas as pd

from core.strategies import (
    _filtros_por_defecto, _mascara_filtros_setup, _tendencia_tf_superior,
    _desc_filtros, generar_senales_sistema, generar_senales,
)


def _df(n=864, tendencia_1=True):
    """5-min velas, 3 días: subida clara la 1ª mitad, caída la 2ª."""
    ts = pd.date_range('2024-01-01', periods=n, freq='5min', tz='UTC')
    precio = np.concatenate([np.linspace(100, 110, n // 2),
                             np.linspace(110, 100, n // 2)])
    return pd.DataFrame({
        'timestamp': ts, 'open': precio, 'close': precio,
        'high': precio + 0.1, 'low': precio - 0.1, 'volume': 1.0,
    })


def _cfg(indicador='SMA', tf='1h', periodo=2, relacion='ambos'):
    return {'indicador': indicador, 'tf': tf, 'periodo': periodo,
            'relacion': relacion}


def test_sin_filtro_no_restriccion():
    filtros = _filtros_por_defecto()
    m_long, m_short, m_forzar = _mascara_filtros_setup(_df(), filtros)
    assert m_long.all() and m_short.all() and m_forzar is None


def test_sma_alinea_con_la_tendencia():
    df = _df()
    m_up, m_dn = _tendencia_tf_superior(df, _cfg())
    n = len(df)
    # en la subida domina el lado «precio >= tendencia», en la caída el otro
    assert m_up[:n // 2].mean() > 0.9
    assert m_dn[:n // 2].mean() < 0.1
    assert m_up[n // 2:].mean() < 0.1
    assert m_dn[n // 2:].mean() > 0.9


def test_sin_lookahead_primera_vela_valida():
    """El invariante duro: ningún valor puede ser válido antes de que cierre
    la PRIMERA vela HTF (01:00 = vela base 12 para TF 1h sobre 5m)."""
    m_up, m_dn = _tendencia_tf_superior(_df(), _cfg(periodo=2))
    validos = m_up | m_dn
    assert not validos[:12].any()


def test_sma_necesita_dos_velas_htf():
    """Con SMA(2) de 1h, además del cierre del primer bin hace falta el
    segundo: primer valor válido en la vela base 24 (02:00)."""
    m_up, m_dn = _tendencia_tf_superior(_df(), _cfg(periodo=2))
    validos = m_up | m_dn
    assert not validos[:23].any()
    assert validos[24]


def test_filtro_aplicado_al_setup_solo_condiciona_entradas():
    df = _df()
    filtros = _filtros_por_defecto()
    filtros['tf_superior'] = _cfg()
    m_long, m_short, m_forzar = _mascara_filtros_setup(df, filtros)
    n = len(df)
    assert m_forzar is None
    # en la subida solo hay entradas long alineadas; en la caída solo short
    assert m_long[:n // 2].mean() > 0.9 and m_long[n // 2:].mean() < 0.1
    assert m_short[n // 2:].mean() > 0.9


def test_relacion_restringe_un_solo_lado():
    df = _df()
    filtros = _filtros_por_defecto()
    filtros['tf_superior'] = _cfg(relacion='long')
    m_long, m_short, _ = _mascara_filtros_setup(df, filtros)
    # los cortos quedan sin restricción por este filtro
    assert m_short.all()
    # los largos sí están filtrados por la alineación
    n = len(df)
    assert m_long[:n // 2].mean() > 0.9 and m_long[n // 2:].mean() < 0.1


def test_tf_no_mayor_se_ignora_con_aviso():
    avisos = []
    filtros = _filtros_por_defecto()
    filtros['tf_superior'] = _cfg(tf='1m')   # menor que las velas de 5m
    m_long, m_short, _ = _mascara_filtros_setup(_df(), filtros, avisos=avisos)
    assert m_long.all() and m_short.all()
    assert any('TF superior' in a and 'no es mayor' in a for a in avisos)


def test_indicador_desconocido_avisa_y_no_restriccion():
    avisos = []
    filtros = _filtros_por_defecto()
    filtros['tf_superior'] = _cfg(indicador='ZZZ')
    m_long, m_short, _ = _mascara_filtros_setup(_df(), filtros, avisos=avisos)
    assert m_long.all() and m_short.all()
    assert any('ZZZ' in a for a in avisos)


def test_supertrend_alinea_sin_lookahead():
    df = _df()
    m_up, m_dn = _tendencia_tf_superior(df, _cfg(indicador='SUPERTREND'))
    n = len(df)
    # nada válido antes del cierre del primer bin HTF (vela base 12)
    assert not (m_up[:12] | m_dn[:12]).any()
    # un lado domina en cada tramo (el giro del Supertrend va con retraso)
    assert m_up[:n // 2].mean() > 0.5
    assert m_dn[n // 2:].mean() > 0.5


def test_descripcion_incluye_el_filtro():
    filtros = _filtros_por_defecto()
    filtros['tf_superior'] = _cfg()
    lineas = _desc_filtros(filtros)
    assert any('TF superior' in l for l in lineas)


def test_generar_senales_sistema_con_filtro_tf_superior():
    """El pipeline completo: toda entrada del sistema cae dentro de la
    máscara de alineación del filtro (entradas ⊆ máscara)."""
    df = _df()
    filtros = _filtros_por_defecto()
    filtros['tf_superior'] = _cfg()
    setup = {'plantilla': 'RSI', 'params': {'periodo': 14, 'sobreventa': 30,
                                            'sobrecompra': 70,
                                            'direccion': 'Ambas'},
             'filtros': filtros, 'riesgo_pct': 0.01}
    ml_raw, ms_raw, _ = _mascara_filtros_setup(df, filtros)
    out = generar_senales_sistema(df, [setup])
    assert not (out['entradas_long'] & ~ml_raw).any()
    assert not (out['entradas_short'] & ~ms_raw).any()
    assert not out['avisos']
