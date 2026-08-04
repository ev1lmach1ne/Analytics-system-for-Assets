"""Entrada por orden límite en el nivel de Fibonacci del último tramo del
ZigZag (_ordenes_limite_fib) y su integración en generar_senales_sistema."""
import numpy as np
import pandas as pd

from core.backtest import simular
from core.strategies import (
    _entrada_por_defecto, _ordenes_limite_fib, _zigzag_eventos,
    generar_senales_sistema, precio_nivel_fib, tramos_zigzag_vigentes,
)


def _df(close):
    close = np.asarray(close, dtype=np.float64)
    n = len(close)
    return pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=n, freq='1h', tz='UTC'),
        'open': close, 'close': close, 'high': close, 'low': close,
    })


def _sierra(n_tramos=6, largo=20, base=100.0, amplitud=20.0):
    tramos = []
    nivel = base
    for k in range(n_tramos):
        destino = base + (amplitud if k % 2 == 0 else 0.0)
        tramos.append(np.linspace(nivel, destino, largo, endpoint=False))
        nivel = destino
    return np.concatenate(tramos)


def _entrada(**extra):
    e = _entrada_por_defecto()
    e.update({'tipo': 'limite_fib'})
    e.update(extra)
    return e


def _primer_tramo(df, desviacion=5.0, piernas=10):
    """(vela de confirmación, precio anterior, precio último, tipo) del primer
    tramo completo (dos pivotes de tipo distinto)."""
    eventos = _zigzag_eventos(df, desviacion, piernas)
    for k in range(1, len(eventos)):
        if eventos[k][3] != eventos[k - 1][3]:
            return eventos[k][1], eventos[k - 1][2], eventos[k][2], eventos[k][3]
    raise AssertionError("la serie de prueba no produce ningún tramo")


def test_el_precio_de_la_orden_es_el_retroceso_exacto():
    df = _df(_sierra())
    conf, prev, ultimo, tipo = _primer_tramo(df)
    n = len(df)
    ent_l = np.zeros(n, dtype=bool)
    ent_s = np.zeros(n, dtype=bool)
    señal = conf + 1
    if tipo == 1:
        ent_l[señal] = True
    else:
        ent_s[señal] = True

    orden_l, orden_s, _cl, _cs = _ordenes_limite_fib(df, _entrada(nivel_fib=0.618),
                                                     ent_l, ent_s)
    recorrido = abs(ultimo - prev)
    if tipo == 1:
        assert orden_l[señal] == ultimo - 0.618 * recorrido
        assert np.isnan(orden_s[señal])
    else:
        assert orden_s[señal] == ultimo + 0.618 * recorrido
        assert np.isnan(orden_l[señal])


def test_un_nivel_mas_profundo_pide_un_precio_mejor():
    df = _df(_sierra())
    conf, _prev, _ult, tipo = _primer_tramo(df)
    n = len(df)
    ent_l = np.zeros(n, dtype=bool)
    ent_s = np.zeros(n, dtype=bool)
    señal = conf + 1
    (ent_l if tipo == 1 else ent_s)[señal] = True
    poco, _s1, _c1, _c2 = _ordenes_limite_fib(df, _entrada(nivel_fib=0.382), ent_l, ent_s)
    mucho, _s2, _c3, _c4 = _ordenes_limite_fib(df, _entrada(nivel_fib=0.786), ent_l, ent_s)
    if tipo == 1:
        assert mucho[señal] < poco[señal]   # comprar más abajo
    else:
        _l1, poco_s, _, _ = _ordenes_limite_fib(df, _entrada(nivel_fib=0.382), ent_l, ent_s)
        _l2, mucho_s, _, _ = _ordenes_limite_fib(df, _entrada(nivel_fib=0.786), ent_l, ent_s)
        assert mucho_s[señal] > poco_s[señal]   # vender más arriba


def test_sin_tramo_confirmado_no_se_coloca_orden():
    """Antes de la primera confirmación no hay swing del que medir nada."""
    df = _df(_sierra())
    n = len(df)
    ent_l = np.zeros(n, dtype=bool)
    ent_l[0] = True
    orden_l, orden_s, _cl, _cs = _ordenes_limite_fib(df, _entrada(), ent_l,
                                                     np.zeros(n, dtype=bool))
    assert np.isnan(orden_l).all()
    assert np.isnan(orden_s).all()


def test_una_senal_contra_la_direccion_del_tramo_no_coloca_orden():
    """Con un tramo alcista vigente, una señal SHORT no tiene retroceso que
    vender: el nivel de Fibonacci mide el retroceso del propio tramo."""
    df = _df(_sierra())
    conf, _prev, _ult, tipo = _primer_tramo(df)
    n = len(df)
    contraria = np.zeros(n, dtype=bool)
    contraria[conf + 1] = True
    if tipo == 1:
        _l, orden_s, _cl, _cs = _ordenes_limite_fib(
            df, _entrada(), np.zeros(n, dtype=bool), contraria)
        assert np.isnan(orden_s).all()
    else:
        orden_l, _s, _cl, _cs = _ordenes_limite_fib(
            df, _entrada(), contraria, np.zeros(n, dtype=bool))
        assert np.isnan(orden_l).all()


def test_un_tramo_nuevo_cancela_las_ordenes_del_anterior():
    df = _df(_sierra())
    n = len(df)
    vacio = np.zeros(n, dtype=bool)
    _l, _s, cancel_l, cancel_s = _ordenes_limite_fib(df, _entrada(), vacio, vacio)
    eventos = _zigzag_eventos(df, 5.0, 10)
    confirmaciones = [e[1] for k, e in enumerate(eventos)
                      if k > 0 and e[3] != eventos[k - 1][3] and e[1] < n]
    assert confirmaciones
    for conf in confirmaciones:
        assert cancel_l[conf] and cancel_s[conf]


def test_el_sistema_traduce_las_senales_del_setup_a_ordenes():
    df = _df(_sierra())
    setups = [{'plantilla': 'Cruce de medias',
               'params': {'rapida': 3, 'lenta': 8, 'direccion': 'Ambas'},
               'entrada': _entrada()}]
    s = generar_senales_sistema(df, setups)
    # el setup no aporta ni una sola entrada a mercado
    assert not s['entradas_long'].any()
    assert not s['entradas_short'].any()
    assert (~np.isnan(s['orden_long_precio'])).any() or \
           (~np.isnan(s['orden_short_precio'])).any()
    assert (s['cancelar_orden_long'] != 0).any()


def test_un_setup_a_mercado_no_toca_las_claves_de_orden():
    """Compatibilidad: sin 'entrada' el sistema produce exactamente lo de
    siempre y ninguna orden límite."""
    df = _df(_sierra())
    setups = [{'plantilla': 'Cruce de medias',
               'params': {'rapida': 3, 'lenta': 8, 'direccion': 'Ambas'}}]
    s = generar_senales_sistema(df, setups)
    assert s['entradas_long'].any() or s['entradas_short'].any()
    assert np.isnan(s['orden_long_precio']).all()
    assert np.isnan(s['orden_short_precio']).all()
    assert not (s['cancelar_orden_long'] != 0).any()


def test_los_setups_de_limite_compiten_por_la_vela_como_los_demas():
    """Prioridad por orden de la lista: si el primer setup ya reclamó la vela,
    el segundo no coloca orden en ella."""
    df = _df(_sierra())
    params = {'rapida': 3, 'lenta': 8, 'direccion': 'Ambas'}
    solo_uno = generar_senales_sistema(
        df, [{'plantilla': 'Cruce de medias', 'params': params,
              'entrada': _entrada()}])
    con_rival = generar_senales_sistema(
        df, [{'plantilla': 'Cruce de medias', 'params': params},
             {'plantilla': 'Cruce de medias', 'params': params,
              'entrada': _entrada()}])
    ordenes_solo = np.count_nonzero(~np.isnan(solo_uno['orden_long_precio']))
    ordenes_rival = np.count_nonzero(~np.isnan(con_rival['orden_long_precio']))
    assert ordenes_rival < ordenes_solo
    assert (con_rival['setup_id'] == 0).any()   # el primero se lleva las velas


def test_la_geometria_publica_reproduce_los_precios_de_orden():
    """El gráfico dibuja el tramo con tramos_zigzag_vigentes/precio_nivel_fib,
    y el motor coloca las órdenes con _ordenes_limite_fib. Este test es el pin
    de que ambos caminos no pueden divergir."""
    df = _df(_sierra())
    n = len(df)
    entrada = _entrada(nivel_fib=0.618)
    # una señal en cada vela, para cubrir todos los tramos de una pasada
    todas = np.ones(n, dtype=bool)
    orden_l, orden_s, _cl, _cs = _ordenes_limite_fib(df, entrada, todas, todas)
    tramos = tramos_zigzag_vigentes(df, entrada['zigzag_desviacion'],
                                    entrada['zigzag_piernas'])

    comprobadas = 0
    for i in range(n):
        precio = orden_l[i] if np.isfinite(orden_l[i]) else orden_s[i]
        if not np.isfinite(precio):
            continue
        esperado = precio_nivel_fib(tramos['anterior'][i], tramos['ultimo'][i],
                                    tramos['tipo'][i], entrada['nivel_fib'])
        assert precio == esperado
        comprobadas += 1
    assert comprobadas > 0


def test_la_geometria_ancla_los_pivotes_reales():
    """idx_anterior/idx_ultimo deben apuntar a las velas donde ocurrieron los
    pivotes, y sus precios coincidir con los del registro del ZigZag."""
    df = _df(_sierra())
    tramos = tramos_zigzag_vigentes(df, 5.0, 10)
    eventos = _zigzag_eventos(df, 5.0, 10)
    por_indice = {e[0]: e[2] for e in eventos}

    vigentes = np.where(tramos['idx_ultimo'] >= 0)[0]
    assert len(vigentes) > 0
    for i in vigentes:
        ia, iu = int(tramos['idx_anterior'][i]), int(tramos['idx_ultimo'][i])
        assert por_indice[ia] == tramos['anterior'][i]
        assert por_indice[iu] == tramos['ultimo'][i]
        assert ia < iu                      # el tramo avanza en el tiempo
        assert iu <= i                      # y nunca usa un pivote del futuro


def test_el_nivel_de_la_orden_cae_dentro_del_tramo():
    """Un retroceso está, por definición, entre los dos extremos del swing. Si
    saliera fuera, la fórmula estaría invertida."""
    df = _df(_sierra())
    n = len(df)
    entrada = _entrada(nivel_fib=0.618)
    todas = np.ones(n, dtype=bool)
    orden_l, orden_s, _cl, _cs = _ordenes_limite_fib(df, entrada, todas, todas)
    tramos = tramos_zigzag_vigentes(df, entrada['zigzag_desviacion'],
                                    entrada['zigzag_piernas'])
    for i in range(n):
        precio = orden_l[i] if np.isfinite(orden_l[i]) else orden_s[i]
        if not np.isfinite(precio):
            continue
        lo = min(tramos['anterior'][i], tramos['ultimo'][i])
        hi = max(tramos['anterior'][i], tramos['ultimo'][i])
        assert lo <= precio <= hi


def test_nuevo_marca_exactamente_las_velas_de_cambio_de_tramo():
    df = _df(_sierra())
    tramos = tramos_zigzag_vigentes(df, 5.0, 10)
    _l, _s, cancelar_l, _cs = _ordenes_limite_fib(
        df, _entrada(), np.zeros(len(df), dtype=bool), np.zeros(len(df), dtype=bool))
    # sin señales, la única fuente de cancelación es el cambio de tramo
    assert np.array_equal(tramos['nuevo'], cancelar_l)


def test_cadena_completa_hasta_el_motor():
    """Integración: señal -> orden límite -> motor. Con entrada límite deben
    operarse MENOS señales que a mercado (solo las que retroceden), y toda
    orden no rellenada debe aparecer registrada."""
    rng = np.random.default_rng(7)
    n = 1500
    close = 100.0 + np.cumsum(rng.normal(0, 1.0, n))
    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=n, freq='1h', tz='UTC'),
        'open': close, 'close': close, 'high': close + 0.6, 'low': close - 0.6,
    })
    params = {'rapida': 10, 'lenta': 30, 'direccion': 'Ambas'}
    base = {'capital_inicial': 10000.0, 'riesgo_pct': 0.01, 'comision_pct': 0.0,
            'slippage_pct': 0.0, 'stop_atr': 2.0, 'tp_r': 3.0, 'salida_n_velas': 0}

    def _correr(setup):
        s = generar_senales_sistema(df, [setup])
        cfg = dict(base)
        cfg['config_por_setup'] = {0: {
            'riesgo_pct': 0.01, 'stop_atr': 2.0, 'tp_r': 3.0,
            'limite_vigencia_velas': setup.get('limite_vigencia_velas', 0),
            'limite_cancelar_avance_r': setup.get('limite_cancelar_avance_r', 0.0),
        }}
        return simular(df['open'].values, df['high'].values, df['low'].values,
                       df['close'].values, s, cfg)

    r_mercado = _correr({'plantilla': 'Cruce de medias', 'params': params})
    r_limite = _correr({'plantilla': 'Cruce de medias', 'params': params,
                        'entrada': _entrada(nivel_fib=0.5),
                        'limite_vigencia_velas': 30,
                        'limite_cancelar_avance_r': 2.0})

    assert r_mercado['n_trades'] > 0
    assert len(r_mercado['ordenes_limite']['idx_alta']) == 0
    assert 0 < r_limite['n_trades'] < r_mercado['n_trades']

    ol = r_limite['ordenes_limite']
    rellenadas = int((ol['resultado'] == 0).sum())
    assert rellenadas == r_limite['n_trades']       # cada relleno abrió un trade
    assert (ol['resultado'] != 0).any()             # y hubo órdenes que no llegaron
    assert (ol['idx_fin'] >= ol['idx_alta']).all()
