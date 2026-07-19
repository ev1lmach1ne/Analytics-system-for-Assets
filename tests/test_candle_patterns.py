import numpy as np

from core.candle_patterns import (
    detectar_patrones, agregar_por_periodo, preparar_contexto, preparar_base_filtro,
)

BASE_N = 25


def _baseline(direction):
    """25 velas de cuerpo pequeño (~0.1-0.15) con tendencia neta en
    `direction` (+1 alcista / -1 bajista), suficientes para que
    cuerpo_medio (ventana=20) y tendencia (ventana=5) estén definidos al
    final de la serie."""
    o_list, h_list, l_list, c_list = [], [], [], []
    price = 100.0
    for i in range(BASE_N):
        o = price
        body = (0.15 if i % 2 == 0 else 0.10) * direction
        c = o + body
        h = max(o, c) + 0.05
        l = min(o, c) - 0.05
        o_list.append(o); h_list.append(h); l_list.append(l); c_list.append(c)
        price = c
    return o_list, h_list, l_list, c_list


def _arrays(base, extra):
    o, h, l, c = base
    o = list(o); h = list(h); l = list(l); c = list(c)
    for (eo, eh, el, ec) in extra:
        o.append(eo); h.append(eh); l.append(el); c.append(ec)
    return (np.array(o), np.array(h), np.array(l), np.array(c))


def _last_idx(resultados, nombre):
    return resultados[nombre]['idx']


def test_marubozu_alcista():
    o, h, l, c = _arrays(_baseline(-1), [(100, 105, 100, 105)])
    res = detectar_patrones(o, h, l, c)
    assert BASE_N in _last_idx(res, 'Marubozu Alcista')
    assert BASE_N not in _last_idx(res, 'Marubozu Bajista')


def test_marubozu_bajista():
    o, h, l, c = _arrays(_baseline(1), [(105, 105, 100, 100)])
    res = detectar_patrones(o, h, l, c)
    assert BASE_N in _last_idx(res, 'Marubozu Bajista')


def test_martillo_no_es_marubozu():
    # vela con mecha inferior dominante: Martillo sí, Marubozu no (mechas
    # grandes descartan el marubozu aunque el cuerpo sea razonable).
    o, h, l, c = _arrays(_baseline(-1), [(99, 99.6, 95.5, 99.4)])
    res = detectar_patrones(o, h, l, c)
    assert BASE_N in _last_idx(res, 'Martillo')
    assert BASE_N not in _last_idx(res, 'Marubozu Alcista')


def test_spinning_top():
    o, h, l, c = _arrays(_baseline(1), [(100, 101.3, 99.1, 100.4)])
    res = detectar_patrones(o, h, l, c)
    assert BASE_N in _last_idx(res, 'Spinning Top')


def test_doji_libelula():
    o, h, l, c = _arrays(_baseline(1), [(100, 100.3, 98.7, 100.1)])
    res = detectar_patrones(o, h, l, c)
    assert BASE_N in _last_idx(res, 'Doji Libélula')
    assert BASE_N not in _last_idx(res, 'Doji Lápida')


def test_doji_lapida():
    o, h, l, c = _arrays(_baseline(1), [(100, 101.4, 99.8, 100.1)])
    res = detectar_patrones(o, h, l, c)
    assert BASE_N in _last_idx(res, 'Doji Lápida')
    assert BASE_N not in _last_idx(res, 'Doji Libélula')


def test_piercing_line():
    extra = [(103, 103.2, 99.8, 100), (99.5, 102.2, 99.3, 102)]
    o, h, l, c = _arrays(_baseline(-1), extra)
    res = detectar_patrones(o, h, l, c)
    assert BASE_N + 1 in _last_idx(res, 'Piercing Line')


def test_dark_cloud_cover():
    extra = [(97, 100.2, 96.8, 100), (100.5, 100.7, 97.8, 98)]
    o, h, l, c = _arrays(_baseline(1), extra)
    res = detectar_patrones(o, h, l, c)
    assert BASE_N + 1 in _last_idx(res, 'Dark Cloud Cover')


def test_tweezer_top():
    extra = [(101, 103.4, 100.6, 103), (103.2, 103.5, 101, 101.5)]
    o, h, l, c = _arrays(_baseline(1), extra)
    res = detectar_patrones(o, h, l, c)
    assert BASE_N + 1 in _last_idx(res, 'Tweezer Top')


def test_tweezer_bottom():
    extra = [(99, 99.4, 96.6, 97), (96.8, 99.5, 96.5, 99)]
    o, h, l, c = _arrays(_baseline(-1), extra)
    res = detectar_patrones(o, h, l, c)
    assert BASE_N + 1 in _last_idx(res, 'Tweezer Bottom')


def test_kicker_alcista():
    extra = [(101, 101.2, 98.8, 99), (101.5, 104, 101.3, 103.8)]
    o, h, l, c = _arrays(_baseline(-1), extra)
    res = detectar_patrones(o, h, l, c)
    assert BASE_N + 1 in _last_idx(res, 'Kicker Alcista')


def test_kicker_bajista():
    extra = [(99, 101.2, 98.8, 101), (98.5, 98.7, 96, 96.2)]
    o, h, l, c = _arrays(_baseline(1), extra)
    res = detectar_patrones(o, h, l, c)
    assert BASE_N + 1 in _last_idx(res, 'Kicker Bajista')


def test_three_inside_up():
    extra = [
        (103, 103.2, 99.8, 100),
        (100.2, 100.6, 99.4, 100.4),
        (100.3, 103.8, 100.1, 103.6),
    ]
    o, h, l, c = _arrays(_baseline(-1), extra)
    res = detectar_patrones(o, h, l, c)
    assert BASE_N + 2 in _last_idx(res, 'Three Inside Up')


def test_three_inside_down():
    extra = [
        (97, 100.2, 96.8, 100),
        (99.6, 100.6, 99.4, 99.8),
        (99.7, 99.9, 96.2, 96.4),
    ]
    o, h, l, c = _arrays(_baseline(1), extra)
    res = detectar_patrones(o, h, l, c)
    assert BASE_N + 2 in _last_idx(res, 'Three Inside Down')


def test_three_outside_up():
    extra = [
        (102, 102.2, 99.8, 100),
        (99.7, 102.4, 99.5, 102.2),
        (102.2, 104.5, 102, 104.2),
    ]
    o, h, l, c = _arrays(_baseline(-1), extra)
    res = detectar_patrones(o, h, l, c)
    assert BASE_N + 2 in _last_idx(res, 'Three Outside Up')


def test_three_outside_down():
    extra = [
        (98, 100.2, 97.8, 100),
        (100.3, 100.5, 97.6, 97.8),
        (97.8, 98, 95.5, 95.7),
    ]
    o, h, l, c = _arrays(_baseline(1), extra)
    res = detectar_patrones(o, h, l, c)
    assert BASE_N + 2 in _last_idx(res, 'Three Outside Down')


def test_abandoned_baby_alcista():
    extra = [
        (103, 103.2, 99.8, 100),
        (98.5, 98.7, 98.3, 98.53),
        (99.5, 102.7, 99.3, 102.5),
    ]
    o, h, l, c = _arrays(_baseline(-1), extra)
    res = detectar_patrones(o, h, l, c)
    assert BASE_N + 2 in _last_idx(res, 'Abandoned Baby Alcista')


def test_abandoned_baby_bajista():
    extra = [
        (97, 100.2, 96.8, 100),
        (101.5, 101.7, 101.3, 101.53),
        (100.5, 100.7, 97.3, 97.5),
    ]
    o, h, l, c = _arrays(_baseline(1), extra)
    res = detectar_patrones(o, h, l, c)
    assert BASE_N + 2 in _last_idx(res, 'Abandoned Baby Bajista')


def test_rising_three_methods():
    extra = [
        (97, 103.5, 96.8, 103),
        (102, 102.3, 100.5, 100.8),
        (100.5, 101, 99.3, 99.6),
        (99.4, 100.5, 98.8, 100.2),
        (100, 105, 99.8, 104.8),
    ]
    o, h, l, c = _arrays(_baseline(-1), extra)
    res = detectar_patrones(o, h, l, c)
    assert BASE_N + 4 in _last_idx(res, 'Rising Three Methods')


def test_falling_three_methods():
    extra = [
        (103, 103.2, 96.5, 97),
        (99, 99.8, 97.7, 98),
        (98, 99, 97.6, 98.6),
        (98.6, 99, 97.5, 97.9),
        (98, 98.2, 93.8, 94),
    ]
    o, h, l, c = _arrays(_baseline(1), extra)
    res = detectar_patrones(o, h, l, c)
    assert BASE_N + 4 in _last_idx(res, 'Falling Three Methods')


def test_agregar_por_periodo_agrupa_por_mes_y_filtra_bloques_pequenos():
    timestamps = np.array([
        '2024-01-01', '2024-01-05', '2024-01-10', '2024-01-15', '2024-01-20',
        '2024-01-22', '2024-01-25', '2024-01-27', '2024-01-28', '2024-01-29',
        '2024-02-02', '2024-02-05', '2024-02-10',  # solo 3, por debajo del minimo
    ], dtype='datetime64[ns]')
    idx = np.arange(len(timestamps))
    aciertos = np.array([True] * 10 + [False, True, False])
    dir_arr = np.ones(len(timestamps))
    signed_ret = np.where(aciertos, 0.001, -0.001)

    res = agregar_por_periodo(idx, dir_arr, aciertos, signed_ret, timestamps,
                              '1ME', drift_base=0.0)

    assert len(res['fechas']) == 1  # solo enero pasa el minimo de 5 ocurrencias
    assert res['n'][0] == 10
    assert res['hit_rate'][0] == 1.0
    assert res['edge_pb'][0] > 0


def test_agregar_por_periodo_sin_ocurrencias():
    res = agregar_por_periodo(
        np.array([], dtype=np.int64), np.array([]), np.array([]), np.array([]),
        np.array([], dtype='datetime64[ns]'), '1ME', drift_base=0.0)
    assert len(res['fechas']) == 0


def test_agregar_por_periodo_todos_bloques_por_debajo_del_minimo():
    timestamps = np.array(
        ['2024-01-01', '2024-01-05', '2024-02-02', '2024-02-05'],
        dtype='datetime64[ns]')
    idx = np.arange(len(timestamps))
    aciertos = np.array([True, False, True, False])
    dir_arr = np.ones(len(timestamps))
    signed_ret = np.zeros(len(timestamps))

    res = agregar_por_periodo(idx, dir_arr, aciertos, signed_ret, timestamps,
                              '1ME', drift_base=0.0, min_ocurrencias=5)

    assert len(res['fechas']) == 0


def test_sesion_ny_ajusta_por_horario_verano_invierno():
    # misma hora UTC (12:00), un día de invierno y uno de verano
    timestamps = np.array(['2024-01-15T12:00:00', '2024-07-15T12:00:00'],
                          dtype='datetime64[ns]')
    close = np.array([100.0, 101.0])
    ctx = preparar_contexto(close, timestamps=timestamps)
    base = preparar_base_filtro(ctx, solo_limpias=False, filtro_sesion='ny')
    m = base['mascara']
    assert not m[0]  # invierno (EST, UTC-5): 12:00 UTC = 07:00 NY, fuera de 8-17
    assert m[1]      # verano (EDT, UTC-4): 12:00 UTC = 08:00 NY, dentro de 8-17


def test_sesion_londres_ajusta_por_horario_verano_invierno():
    timestamps = np.array(['2024-01-15T07:30:00', '2024-07-15T07:30:00'],
                          dtype='datetime64[ns]')
    close = np.array([100.0, 101.0])
    ctx = preparar_contexto(close, timestamps=timestamps)
    base = preparar_base_filtro(ctx, solo_limpias=False, filtro_sesion='londres')
    m = base['mascara']
    assert not m[0]  # invierno (GMT, UTC+0): 07:30 UTC = 07:30 Londres, fuera de 8-17
    assert m[1]      # verano (BST, UTC+1): 07:30 UTC = 08:30 Londres, dentro de 8-17


def test_sesion_overnight_no_depende_de_epoca_del_anio():
    # 'overnight' no tiene huso horario propio: rango UTC fijo (1,9) todo el año
    timestamps = np.array(['2024-01-15T05:00:00', '2024-07-15T05:00:00'],
                          dtype='datetime64[ns]')
    close = np.array([100.0, 101.0])
    ctx = preparar_contexto(close, timestamps=timestamps)
    base = preparar_base_filtro(ctx, solo_limpias=False, filtro_sesion='overnight')
    m = base['mascara']
    assert m[0] and m[1]  # 05:00 UTC cae dentro de (1,9) en ambos casos
