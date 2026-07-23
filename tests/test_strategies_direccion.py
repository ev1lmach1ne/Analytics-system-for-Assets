"""Columna 'direccion' (ambas/long/short) de las condiciones de filtro de un
setup: dentro de un mismo setup, una condicion puede restringir solo las
entradas/salidas long, solo las short, o ambas (comportamiento por defecto,
compatible con setups guardados sin el campo)."""
import numpy as np
import pandas as pd

from core.strategies import (
    generar_senales_sistema, _mascaras_condiciones_dir, _mascara_condiciones,
)


def _df_close(valores):
    valores = np.asarray(valores, dtype=np.float64)
    return pd.DataFrame({
        'open': valores, 'high': valores + 1.0, 'low': valores - 1.0,
        'close': valores,
    })


def _cond(op, direccion=None):
    c = {'izq': {'tipo': 'close'}, 'op': op, 'der': {'tipo': 'valor', 'valor': 150.0}}
    if direccion is not None:
        c['direccion'] = direccion
    return c


def test_mascaras_condiciones_dir_separa_por_direccion():
    df = _df_close([100.0, 200.0, 100.0, 200.0])
    condiciones = [_cond('>', 'long'), _cond('<', 'short')]
    m_long, m_short = _mascaras_condiciones_dir(df, condiciones)
    # 'long' solo exige close > 150 -> True en las velas altas (idx 1,3)
    assert m_long.tolist() == [False, True, False, True]
    # 'short' solo exige close < 150 -> True en las velas bajas (idx 0,2)
    assert m_short.tolist() == [True, False, True, False]


def test_direccion_ambas_restringe_los_dos_lados_igual_que_antes():
    """Sin 'direccion' (o con 'ambas') debe reproducir exactamente
    _mascara_condiciones en ambos lados -> compatibilidad con setups
    guardados antes de esta funcionalidad."""
    df = _df_close([100.0, 200.0, 100.0, 200.0])
    condiciones = [_cond('>')]   # sin 'direccion' -> por defecto 'ambas'
    m_long, m_short = _mascaras_condiciones_dir(df, condiciones)
    esperado = _mascara_condiciones(df, condiciones)
    assert m_long.tolist() == esperado.tolist()
    assert m_short.tolist() == esperado.tolist()

    condiciones_explicitas = [_cond('>', 'ambas')]
    m_long2, m_short2 = _mascaras_condiciones_dir(df, condiciones_explicitas)
    assert m_long2.tolist() == esperado.tolist()
    assert m_short2.tolist() == esperado.tolist()


def _setup_custom_siempre_dispara(filtros):
    """Setup 'Custom (reglas)' cuya entrada/salida siempre disparan (antes
    del filtro), para poder aislar el efecto del filtro direccional."""
    siempre = [{'izq': {'tipo': 'close'}, 'op': '>', 'der': {'tipo': 'valor', 'valor': -1e9}}]
    return {
        'plantilla': 'Custom (reglas)',
        'params': {'reglas': {
            'entradas_long': [{'setup_id': 0, 'condiciones': siempre}],
            'entradas_short': [{'setup_id': 0, 'condiciones': siempre}],
            'salidas_long': [{'setup_id': 0, 'condiciones': siempre}],
            'salidas_short': [{'setup_id': 0, 'condiciones': siempre}],
        }},
        'filtros': filtros,
    }


def test_generar_senales_sistema_entrada_direccional_tipo_media_200():
    """Caso motivador: un mismo setup con una condicion tipo 'close > media'
    solo para long y 'close < media' solo para short -- ambas activas a la
    vez sin que una entrada bloquee a la otra."""
    df = _df_close([100.0, 200.0, 100.0, 200.0])
    setup = _setup_custom_siempre_dispara({
        'condiciones_entrada': [_cond('>', 'long'), _cond('<', 'short')],
    })
    out = generar_senales_sistema(df, [setup])
    assert out['entradas_long'].tolist() == [False, True, False, True]
    assert out['entradas_short'].tolist() == [True, False, True, False]


def test_generar_senales_sistema_salida_direccional():
    df = _df_close([100.0, 200.0, 100.0, 200.0])
    setup = _setup_custom_siempre_dispara({
        'condiciones_salida': [_cond('>', 'long'), _cond('<', 'short')],
    })
    out = generar_senales_sistema(df, [setup])
    bit0 = np.int64(1)
    assert (out['salidas_long'] == np.array([0, bit0, 0, bit0])).all()
    assert (out['salidas_short'] == np.array([bit0, 0, bit0, 0])).all()


def test_generar_senales_sistema_sin_filtros_no_rompe():
    """Regresión: un setup sin 'filtros' (o con filtros vacíos) sigue
    funcionando igual que antes de este cambio."""
    df = _df_close([100.0, 200.0, 100.0, 200.0])
    setup = {
        'plantilla': 'Custom (reglas)',
        'params': {'reglas': {
            'entradas_long': [{'setup_id': 0, 'condiciones': [
                {'izq': {'tipo': 'close'}, 'op': '>', 'der': {'tipo': 'valor', 'valor': -1e9}}]}],
            'entradas_short': [], 'salidas_long': [], 'salidas_short': [],
        }},
    }
    out = generar_senales_sistema(df, [setup])
    assert out['entradas_long'].tolist() == [True, True, True, True]
