"""_acumular_indicador_spec (gui/widgets/tab_backtest.py): clasifica un spec
de indicador de una condicion ({'tipo':'EMA','periodo':200}, etc.) en los
sets que consume ResultadosWidget para pintar overlays en la grafica de
Resultados. No requiere QApplication: es una funcion module-level sin
estado de widget."""
from gui.widgets.tab_backtest import _acumular_indicador_spec


def _clasificar(spec):
    mas, rsis, atrs, bbs = set(), set(), set(), set()
    _acumular_indicador_spec(spec, mas, rsis, atrs, bbs)
    return mas, rsis, atrs, bbs


def test_ema_y_sma_van_a_mas():
    mas, rsis, atrs, bbs = _clasificar({'tipo': 'EMA', 'periodo': 200})
    assert mas == {('EMA', 200)}
    assert not rsis and not atrs and not bbs

    mas, rsis, atrs, bbs = _clasificar({'tipo': 'SMA', 'periodo': 50})
    assert mas == {('SMA', 50)}


def test_rsi_va_a_rsis():
    mas, rsis, atrs, bbs = _clasificar({'tipo': 'RSI', 'periodo': 14})
    assert rsis == {14}
    assert not mas and not atrs and not bbs


def test_atr_va_a_atrs():
    mas, rsis, atrs, bbs = _clasificar({'tipo': 'ATR', 'periodo': 21})
    assert atrs == {21}


def test_bollinger_va_a_bbs_con_periodo_y_desviacion():
    mas, rsis, atrs, bbs = _clasificar({'tipo': 'BB_sup', 'periodo': 20, 'desv': 2.5})
    assert bbs == {(20, 2.5)}


def test_close_open_high_low_y_valor_se_ignoran():
    for spec in ({'tipo': 'close'}, {'tipo': 'open'}, {'tipo': 'high'},
                 {'tipo': 'low'}, {'tipo': 'valor', 'valor': 30.0}):
        mas, rsis, atrs, bbs = _clasificar(spec)
        assert not mas and not rsis and not atrs and not bbs


def test_filtro_de_setup_se_recolecta_para_cualquier_plantilla():
    """Regresión del bug reportado: un filtro EMA(200) en 'condiciones_entrada'
    de un setup de 'Patrones de velas' (una plantilla sin parámetros de media
    propios) debe aparecer igualmente en los sets recolectados."""
    import gui.widgets.tab_backtest as tb

    class _Stub(tb.ResultadosWidget):
        def __init__(self):
            pass   # evita construir widgets Qt; solo probamos la lógica pura

    payload = {'setups': [{
        'plantilla': 'Patrones de velas',
        'params': {'patrones': ['Martillo Invertido'], 'lag_salida': 5},
        'filtros': {
            'condiciones_entrada': [
                {'izq': {'tipo': 'close'}, 'op': '>',
                 'der': {'tipo': 'EMA', 'periodo': 200}, 'direccion': 'long'},
            ],
        },
    }]}
    mas, bbs, rsis, atrs, patrones_set = _Stub()._recolectar_indicadores(payload)
    assert ('EMA', 200) in mas
    assert patrones_set == {'Martillo Invertido'}
