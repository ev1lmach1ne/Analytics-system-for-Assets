"""
tests/test_stop_atr_modo.py
Modo del Stop Loss ×ATR ('stop_atr_modo'): 'fijo' (defecto) vs
'dinamico_promedio', y la invariante de que el riesgo nominal total de un
setup (distancia de cada entrada al stop × su volumen, sumando todos los
tramos) nunca supera equity_al_abrir × riesgo_pct.
"""
import numpy as np
import pytest

from core.backtest import simular
from core.strategies import validar_setup


def _ohlc_plano(n, precio=100.0):
    o = np.full(n, precio)
    h = np.full(n, precio + 1.0)
    l = np.full(n, precio - 1.0)
    c = np.full(n, precio)
    return o, h, l, c


def _senales(n, atr=2.0):
    return {'entradas_long': np.zeros(n, dtype=bool),
            'entradas_short': np.zeros(n, dtype=bool),
            'salidas_long': np.zeros(n, dtype=bool),
            'salidas_short': np.zeros(n, dtype=bool),
            'setup_id': np.zeros(n, dtype=np.int64),
            'atr': np.full(n, atr)}


CONFIG_BASE = {'capital_inicial': 10000.0, 'riesgo_pct': 0.01,
               'comision_pct': 0.0, 'slippage_pct': 0.0,
               'stop_atr': 0.0, 'tp_r': 0.0, 'salida_n_velas': 0}


def _cfg_por_setup(**extra):
    return dict(CONFIG_BASE, config_por_setup={0: extra})


def _tramos(*pcts):
    """Tramos por 'senal' (1º) y por 'velas' (+3) el resto, al 50% cada uno."""
    tramos = []
    for k, pct in enumerate(pcts):
        tr = {'pct': pct, 'val': 0.0, 'condiciones': [],
              'gestion': {'tipo': 0, 'val': 0.0}}
        if k == 0:
            tr['trigger'] = 'senal'
        else:
            tr['trigger'] = 'velas'
            tr['val'] = 3.0
        tramos.append(tr)
    return tramos


def _riesgo_nominal(r):
    """Σ u_j × max(0, d × (P_j − S)) con el stop que tenía la posición."""
    entradas = r['entradas']
    stop = r['trades']['precio_stop'][0]
    d = entradas['dir'][0]
    return sum(u * max(0.0, d * (p - stop))
               for p, u in zip(entradas['precio'], entradas['unidades']))


# ── causalidad: el ATR de la entrada es el de la última vela CERRADA ──

def test_el_atr_de_entrada_es_el_de_la_ultima_vela_cerrada():
    n = 10
    o, h, l, c = _ohlc_plano(n)
    s = _senales(n)
    s['entradas_long'][2] = True        # entra al open de la vela 3
    s['salidas_long'][8] = True
    # la vela 2 (ya cerrada) tiene ATR 10; la vela 3 (de entrada) ATR 20
    s['atr'][2] = 10.0
    s['atr'][3] = 20.0
    r = simular(o, h, l, c, s, _cfg_por_setup(stop_atr=1.0))
    # stop = 100 − 1×10 = 90 (vela cerrada), no 100 − 20 = 80
    assert r['trades']['precio_stop'][0] == pytest.approx(90.0)
    # tamaño con esa misma distancia: 10000×0.01/10 = 10 unidades
    assert r['trades']['unidades'][0] == pytest.approx(10.0)


# ── modo por defecto ──

def test_modo_ausente_equivale_a_fijo():
    n = 10
    o, h, l, c = _ohlc_plano(n)
    s = _senales(n)
    s['entradas_long'][2] = True
    s['salidas_long'][8] = True
    r_ausente = simular(o, h, l, c, s, _cfg_por_setup(stop_atr=1.0))
    r_fijo = simular(o, h, l, c, s,
                     _cfg_por_setup(stop_atr=1.0, stop_atr_modo='fijo'))
    for clave in r_ausente['trades']:
        assert np.allclose(r_ausente['trades'][clave],
                           r_fijo['trades'][clave]), clave
    assert np.allclose(r_ausente['equity'], r_fijo['equity'])


# ── dinámico: con una sola entrada es idéntico a fijo ──

def test_dinamico_con_una_entrada_es_igual_que_fijo():
    n = 10
    o, h, l, c = _ohlc_plano(n)
    s = _senales(n)
    s['entradas_long'][2] = True
    s['salidas_long'][8] = True
    r_fijo = simular(o, h, l, c, s,
                     _cfg_por_setup(stop_atr=1.0, stop_atr_modo='fijo'))
    r_din = simular(o, h, l, c, s,
                    _cfg_por_setup(stop_atr=1.0,
                                   stop_atr_modo='dinamico_promedio'))
    assert r_din['trades']['precio_stop'][0] == pytest.approx(98.0)
    for clave in r_fijo['trades']:
        assert np.allclose(r_fijo['trades'][clave],
                           r_din['trades'][clave]), clave


# ── dinámico con tramos: se reancla al precio medio ──

def test_dinamico_reancla_el_stop_al_precio_medio():
    n = 12
    o, h, l, c = _ohlc_plano(n)
    o[7] = 101.0                        # 2º tramo pirámide (por encima)
    s = _senales(n)
    s['entradas_long'][2] = True        # 1er tramo al open de 3 (100)
    s['salidas_long'][9] = True         # cierre por señal al open de 10
    cfg = _cfg_por_setup(stop_atr=1.0, stop_atr_modo='dinamico_promedio',
                         tramos=_tramos(50.0, 50.0))
    r = simular(o, h, l, c, s, cfg)
    entradas = r['entradas']
    assert len(entradas['precio']) == 2
    p_medio = ((entradas['precio'] * entradas['unidades']).sum()
               / entradas['unidades'].sum())
    assert p_medio == pytest.approx(100.4)
    # stop dinámico = precio medio − 1×ATR(2) = 98.4
    assert r['trades']['precio_stop'][0] == pytest.approx(98.4)
    assert _riesgo_nominal(r) <= 100.0 * (1 + 1e-9)


def test_dinamico_short_reancla_con_su_geometria():
    n = 12
    o, h, l, c = _ohlc_plano(n)
    o[7] = 99.0                         # 2º tramo a favor (precio más bajo)
    s = _senales(n)
    s['entradas_short'][2] = True
    s['salidas_short'][9] = True
    cfg = _cfg_por_setup(stop_atr=1.0, stop_atr_modo='dinamico_promedio',
                         tramos=_tramos(50.0, 50.0))
    r = simular(o, h, l, c, s, cfg)
    entradas = r['entradas']
    p_medio = ((entradas['precio'] * entradas['unidades']).sum()
               / entradas['unidades'].sum())
    assert p_medio == pytest.approx(99.6)
    # stop corto = precio medio + 1×ATR(2) = 101.6
    assert r['trades']['precio_stop'][0] == pytest.approx(101.6)
    assert _riesgo_nominal(r) <= 100.0 * (1 + 1e-9)


# ── invariante: el presupuesto de riesgo es el límite absoluto ──

def test_tramos_no_superan_el_presupuesto_aunque_sumen_mas_de_100():
    n = 12
    o, h, l, c = _ohlc_plano(n)
    s = _senales(n)
    s['entradas_long'][2] = True
    s['salidas_long'][9] = True
    # 50% + 70% = 120% del presupuesto: el 2º tramo se recorta al 50% real
    cfg = _cfg_por_setup(stop_atr=1.0, stop_atr_modo='fijo',
                         tramos=_tramos(50.0, 70.0))
    r = simular(o, h, l, c, s, cfg)
    assert list(r['entradas']['unidades']) == pytest.approx([25.0, 25.0])
    assert _riesgo_nominal(r) == pytest.approx(100.0)


def test_el_riesgo_nominal_nunca_supera_el_presupuesto():
    n = 12
    o, h, l, c = _ohlc_plano(n)
    s = _senales(n)
    s['entradas_long'][2] = True
    s['salidas_long'][9] = True
    for modo in ('fijo', 'dinamico_promedio'):
        cfg = _cfg_por_setup(stop_atr=1.0, stop_atr_modo=modo,
                             tramos=_tramos(25.0, 65.0, 10.0))
        r = simular(o, h, l, c, s, cfg)
        assert _riesgo_nominal(r) <= 100.0 * (1 + 1e-9), modo


def test_stop_dinamico_queda_limitado_por_el_presupuesto():
    n = 12
    o, h, l, c = _ohlc_plano(n)
    s = _senales(n)
    s['entradas_long'][2] = True
    s['salidas_long'][9] = True
    # el ATR de la vela cerrada antes del 2º tramo se dispara a 10
    s['atr'][6] = 10.0
    cfg = _cfg_por_setup(stop_atr=1.0, stop_atr_modo='dinamico_promedio',
                         tramos=_tramos(50.0, 50.0))
    r = simular(o, h, l, c, s, cfg)
    # sin tope el stop dinámico sería 100 − 1×10 = 90; el presupuesto exige
    # riesgo(98) = 100, así que el stop no puede bajar de 98
    assert r['trades']['precio_stop'][0] == pytest.approx(98.0)
    assert _riesgo_nominal(r) <= 100.0 * (1 + 1e-9)


# ── gaps: el open que cruza el stop llena al open y no ejecuta el tramo ──

def test_gap_de_apertura_se_llena_al_open_y_no_anade_el_tramo():
    n = 12
    o, h, l, c = _ohlc_plano(n)
    o[7] = 90.0                         # gap por debajo del stop (98)
    s = _senales(n)
    s['entradas_long'][2] = True
    cfg = _cfg_por_setup(stop_atr=1.0, stop_atr_modo='fijo',
                         tramos=_tramos(50.0, 50.0))
    r = simular(o, h, l, c, s, cfg)
    assert len(r['entradas']['precio']) == 1          # el tramo no se ejecutó
    t = r['trades']
    assert t['motivo'][0] == 1
    assert t['precio_salida'][0] == pytest.approx(90.0)
    assert t['unidades'][0] == pytest.approx(25.0)


def test_gap_short_se_llena_al_open_y_no_anade_el_tramo():
    n = 12
    o, h, l, c = _ohlc_plano(n)
    o[7] = 110.0                        # gap por encima del stop (102)
    s = _senales(n)
    s['entradas_short'][2] = True
    cfg = _cfg_por_setup(stop_atr=1.0, stop_atr_modo='fijo',
                         tramos=_tramos(50.0, 50.0))
    r = simular(o, h, l, c, s, cfg)
    assert len(r['entradas']['precio']) == 1
    t = r['trades']
    assert t['motivo'][0] == 1
    assert t['precio_salida'][0] == pytest.approx(110.0)
    assert t['unidades'][0] == pytest.approx(25.0)


# ── BE/trailing: el stop dinámico no rebaja un stop ya mejorado ──

def test_stop_dinamico_no_rebaja_un_stop_mejorado_por_be():
    n = 14
    # tendencia alcista suave: el precio sube tras la entrada y ya no vuelve
    o = np.array([100, 100, 100, 100, 100.5, 101, 101.5, 101, 101, 101,
                  101, 101, 101, 101], dtype=float)
    h = o + 0.5
    l = o - 0.2
    c = o + 0.1
    s = _senales(n)
    s['entradas_long'][2] = True        # entra al open de 3 (100)
    s['salidas_long'][10] = True        # cierra al open de 11
    s['atr'][6] = 5.0                   # ATR grande para el tramo dinámico
    cfg = _cfg_por_setup(
        stop_atr=1.0, stop_atr_modo='dinamico_promedio',
        be_atr=0.5, be_unidad='atr',
        tramos=_tramos(50.0, 50.0))
    r = simular(o, h, l, c, s, cfg)
    assert len(r['entradas']['precio']) == 2       # el tramo sí se ejecutó
    # el BE movió el stop a 100 (origen BE): el reanclaje dinámico no lo rebaja
    assert r['trades']['precio_stop'][0] == pytest.approx(100.0)


# ── sin stop real ──

def test_dinamico_sin_stop_no_inventa_ningun_stop():
    n = 10
    o, h, l, c = _ohlc_plano(n)
    s = _senales(n)
    s['entradas_long'][2] = True
    s['salidas_long'][8] = True
    r = simular(o, h, l, c, s,
                _cfg_por_setup(stop_atr=0.0, stop_atr_modo='dinamico_promedio'))
    assert r['n_trades'] == 1
    assert r['trades']['precio_stop'][0] == 0.0


# ── validación y propagación del modo ──

def test_validar_setup_rechaza_modo_desconocido():
    assert validar_setup({'stop_atr': 1.0, 'stop_atr_modo': 'fijo'}) == []
    assert validar_setup({'stop_atr': 1.0,
                          'stop_atr_modo': 'dinamico_promedio'}) == []
    avisos = validar_setup({'stop_atr': 1.0, 'stop_atr_modo': 'raro'})
    assert any('stop' in a.lower() for a in avisos)


def test_plan_gestion_y_fidelidad_conocen_el_modo():
    from core.codegen import fidelidad, ir
    setup = {'nombre': 'x', 'plantilla': 'RSI', 'stop_atr': 2.0,
             'stop_atr_modo': 'dinamico_promedio'}
    bloque = ir.ir_setup(setup)
    assert bloque['gestion']['stop_atr_modo'] == 'dinamico_promedio'
    sistema = ir.ir_sistema([setup], {'capital_inicial': 10000.0})
    avisos = fidelidad.analizar(sistema, 'tradingview')
    assert any(a['clave'] == 'stop_atr_dinamico' for a in avisos)


def test_config_motor_del_optimizador_conserva_el_modo():
    from core.optimizer import _config_motor
    cfg = _config_motor(
        {'riesgo_pct': 0.01, 'stop_atr': 2.0, 'stop_atr_modo': 'dinamico_promedio'},
        {})
    assert cfg['config_por_setup'][0]['stop_atr_modo'] == 'dinamico_promedio'
    cfg2 = _config_motor({'riesgo_pct': 0.01, 'stop_atr': 2.0}, {})
    assert cfg2['config_por_setup'][0]['stop_atr_modo'] == 'fijo'
