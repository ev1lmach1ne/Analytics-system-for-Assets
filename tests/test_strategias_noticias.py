"""Filtro 'noticias' del constructor (evitar/cerrar operaciones alrededor de
eventos económicos históricos): _mascara_evitar_ventanas, _mascara_filtros_setup
y su threading a través de generar_senales_sistema."""
import numpy as np
import pandas as pd

from core.strategies import (
    _mascara_filtros_setup, _mascara_evitar_ventanas, preparar_eventos_noticias,
    generar_senales_sistema, _filtros_por_defecto,
)


def _df_minutal(n=120, inicio='2024-01-01 00:00:00'):
    ts = pd.date_range(inicio, periods=n, freq='1min', tz='UTC')
    close = 100.0 + np.arange(n, dtype=np.float64) * 0.1
    return pd.DataFrame({
        'timestamp': ts, 'open': close, 'high': close + 0.5,
        'low': close - 0.5, 'close': close,
    })


def _eventos(momentos):
    """momentos: lista de (timestamp, pais, impacto)."""
    df = pd.DataFrame({
        'timestamp': pd.to_datetime([m[0] for m in momentos], utc=True),
        'pais': [m[1] for m in momentos],
        'evento': ['Evento'] * len(momentos),
        'impacto': [m[2] for m in momentos],
    })
    return preparar_eventos_noticias(df)


def test_mascara_evitar_ventanas_bloquea_solo_el_rango_esperado():
    ts_velas = np.arange(0, 100, dtype=np.int64)
    ts_eventos = np.array([50], dtype=np.int64)
    m = _mascara_evitar_ventanas(ts_velas, ts_eventos, antes_s=10, despues_s=5)
    bloqueadas = np.where(~m)[0]
    assert bloqueadas.tolist() == list(range(40, 56))


def test_mascara_filtros_setup_bloquea_entradas_en_ventana_noticia():
    df = _df_minutal(n=120)
    eventos = _eventos([(df['timestamp'].iloc[60], 'US', 'alto')])
    filtros = _filtros_por_defecto()
    filtros['noticias'].update({'activo': True, 'minutos_antes': 5, 'minutos_despues': 5,
                                 'impacto_minimo': 'alto'})
    m_long, m_short, m_forzar = _mascara_filtros_setup(df, filtros, eventos)
    assert not m_long[55:66].any()
    assert m_long[:55].all()
    assert m_long[66:].all()
    assert (m_long == m_short).all()   # noticias no distingue dirección
    assert m_forzar is None            # cerrar_posiciones desactivado por defecto


def test_mascara_filtros_setup_ignora_evento_de_impacto_insuficiente():
    df = _df_minutal(n=120)
    eventos = _eventos([(df['timestamp'].iloc[60], 'US', 'bajo')])
    filtros = _filtros_por_defecto()
    filtros['noticias'].update({'activo': True, 'impacto_minimo': 'alto'})
    m_long, m_short, _ = _mascara_filtros_setup(df, filtros, eventos)
    assert m_long.all() and m_short.all()


def test_mascara_filtros_setup_filtra_por_moneda_del_setup():
    df = _df_minutal(n=120)
    eventos = _eventos([(df['timestamp'].iloc[60], 'EU', 'alto')])
    filtros = _filtros_por_defecto()
    filtros['noticias'].update({'activo': True, 'impacto_minimo': 'alto', 'monedas': ['US']})
    m_long, _, _ = _mascara_filtros_setup(df, filtros, eventos)
    assert m_long.all()   # evento EU, el setup solo mira US -> no bloquea

    filtros['noticias']['monedas'] = ['EU']
    m_long2, _, _ = _mascara_filtros_setup(df, filtros, eventos)
    assert not m_long2.all()   # ahora sí coincide y bloquea


def test_mascara_filtros_setup_cerrar_posiciones_fuerza_salida():
    df = _df_minutal(n=120)
    eventos = _eventos([(df['timestamp'].iloc[60], 'US', 'alto')])
    filtros = _filtros_por_defecto()
    filtros['noticias'].update({'activo': True, 'minutos_antes': 5, 'minutos_despues': 5,
                                 'impacto_minimo': 'alto', 'cerrar_posiciones': True})
    _, _, m_forzar = _mascara_filtros_setup(df, filtros, eventos)
    assert m_forzar is not None
    assert m_forzar[55:66].all()
    assert not m_forzar[:55].any()
    assert not m_forzar[66:].any()


def test_mascara_filtros_setup_inactivo_no_bloquea_aunque_haya_eventos():
    df = _df_minutal(n=120)
    eventos = _eventos([(df['timestamp'].iloc[60], 'US', 'alto')])
    filtros = _filtros_por_defecto()   # 'noticias.activo' = False por defecto
    m_long, m_short, m_forzar = _mascara_filtros_setup(df, filtros, eventos)
    assert m_long.all() and m_short.all() and m_forzar is None


def _setup_siempre_entra(filtros=None):
    siempre = [{'izq': {'tipo': 'close'}, 'op': '>', 'der': {'tipo': 'valor', 'valor': -1e9}}]
    nunca = [{'izq': {'tipo': 'close'}, 'op': '>', 'der': {'tipo': 'valor', 'valor': 1e9}}]
    setup = {
        'plantilla': 'Custom (reglas)',
        'params': {'reglas': {
            'entradas_long': [{'setup_id': 0, 'condiciones': siempre}],
            'entradas_short': [], 'salidas_long': [{'setup_id': 0, 'condiciones': nunca}],
            'salidas_short': [],
        }},
    }
    if filtros is not None:
        setup['filtros'] = filtros
    return setup


def test_generar_senales_sistema_eventos_none_no_cambia_comportamiento_previo():
    df = _df_minutal(n=20)
    setup = _setup_siempre_entra()
    out_sin_arg = generar_senales_sistema(df, [setup])
    out_con_none = generar_senales_sistema(df, [setup], eventos_noticias=None)
    assert out_sin_arg['entradas_long'].tolist() == out_con_none['entradas_long'].tolist()
    assert out_sin_arg['salidas_long'].tolist() == out_con_none['salidas_long'].tolist()


def test_generar_senales_sistema_bloquea_entrada_por_noticia():
    df = _df_minutal(n=20)
    eventos = _eventos([(df['timestamp'].iloc[10], 'US', 'alto')])
    filtros = dict(_filtros_por_defecto(), noticias={
        'activo': True, 'minutos_antes': 3, 'minutos_despues': 3,
        'impacto_minimo': 'alto', 'monedas': None, 'cerrar_posiciones': False,
    })
    setup = _setup_siempre_entra(filtros)
    out = generar_senales_sistema(df, [setup], eventos)
    esperado = np.ones(20, dtype=bool)
    esperado[7:14] = False
    assert out['entradas_long'].tolist() == esperado.tolist()


def test_generar_senales_sistema_cierra_posicion_forzada_por_noticia():
    df = _df_minutal(n=20)
    eventos = _eventos([(df['timestamp'].iloc[10], 'US', 'alto')])
    filtros = dict(_filtros_por_defecto(), noticias={
        'activo': True, 'minutos_antes': 3, 'minutos_despues': 3,
        'impacto_minimo': 'alto', 'monedas': None, 'cerrar_posiciones': True,
    })
    setup = _setup_siempre_entra(filtros)
    out = generar_senales_sistema(df, [setup], eventos)
    bit0 = np.int64(1)
    esperado = np.zeros(20, dtype=np.int64)
    esperado[7:14] = bit0
    # la estrategia nunca señala salida propia ('nunca' jamás se cumple):
    # cualquier bit de salida solo puede venir del cierre forzado por noticia
    assert out['salidas_long'].tolist() == esperado.tolist()


# ══════════════ filtro de sesión en TF diario ══════════════
# En TF diario o mayor el filtro de sesión por horas no tiene sentido: todas
# las velas caen a la misma hora del día, así que un rango horario (o una
# sesión overnight/Londres/NY) dejaría 0 velas dentro → 0 señales. El motor
# debe ignorar el filtro (con aviso) en vez de anular el backtest.


def _df_diario(n=30, inicio='2024-01-01'):
    ts = pd.date_range(inicio, periods=n, freq='1D', tz='UTC')
    close = 100.0 + np.arange(n, dtype=np.float64) * 0.1
    return pd.DataFrame({
        'timestamp': ts, 'open': close, 'high': close + 0.5,
        'low': close - 0.5, 'close': close,
    })


def test_sesion_ignorada_en_tf_diario():
    df = _df_diario()
    filtros = dict(_filtros_por_defecto(), sesion={
        'tipo': 'personalizada', 'hora_inicio': 8, 'hora_fin': 16})
    avisos = []
    m_l, m_s, m_forzar = _mascara_filtros_setup(df, filtros, avisos=avisos)
    assert m_l.all() and m_s.all()
    assert m_forzar is None
    assert any('diario' in a for a in avisos)


def test_sesion_si_filtra_en_intradia():
    df = _df_minutal(n=120)
    df['timestamp'] = pd.date_range('2024-01-01 06:00', periods=120,
                                    freq='1h', tz='UTC')
    filtros = dict(_filtros_por_defecto(), sesion={
        'tipo': 'personalizada', 'hora_inicio': 8, 'hora_fin': 16})
    m_l, m_s, _ = _mascara_filtros_setup(df, filtros)
    n_dentro = int(m_l.sum())
    # 120 velas horarias → 5 días × 8 horas dentro del rango 8-16h
    assert n_dentro == 40
    assert not m_l.all()


def test_noticias_ventana_menor_que_tf_bloquea_la_vela_del_evento():
    """Ventana ±3 min en TF 15 min: sin ampliación la ventana cae entre
    velas y no bloquea nada; con la ampliación al TF se bloquea al menos la
    vela completa que contiene el evento (con aviso)."""
    df = _df_minutal(n=120)
    df['timestamp'] = pd.date_range('2024-01-01 00:00', periods=120,
                                    freq='15min', tz='UTC')
    # evento a las 00:07: dentro de la vela 00:00 (con ventana 3 min la
    # ventana [00:04,00:10] no toca ningún timestamp de vela)
    eventos = _eventos([(pd.Timestamp('2024-01-01 00:07', tz='UTC'),
                         'US', 'alto')])
    filtros = _filtros_por_defecto()
    filtros['noticias'].update({'activo': True, 'minutos_antes': 3,
                                'minutos_despues': 3, 'impacto_minimo': 'alto'})
    avisos = []
    m_long, m_short, _ = _mascara_filtros_setup(df, filtros, eventos,
                                                avisos=avisos)
    # la vela 00:00 (la que contiene el evento) queda bloqueada, y la
    # expansión al TF también alcanza la vela 00:15; desde 00:30 en adelante
    # todo admisible
    assert not m_long[0]
    assert m_long[2:].all()
    assert any('menor que el timeframe' in a for a in avisos)


def test_noticias_ventana_mayor_que_tf_sigue_igual():
    """Ventana ±30 min en TF 15 min: sin ampliación (ya cubre el TF) el
    comportamiento no cambia y no avisa."""
    df = _df_minutal(n=120)
    df['timestamp'] = pd.date_range('2024-01-01 00:00', periods=120,
                                    freq='15min', tz='UTC')
    eventos = _eventos([(pd.Timestamp('2024-01-01 00:30', tz='UTC'),
                         'US', 'alto')])
    filtros = _filtros_por_defecto()
    filtros['noticias'].update({'activo': True, 'minutos_antes': 30,
                                'minutos_despues': 30, 'impacto_minimo': 'alto'})
    avisos = []
    m_long, m_short, _ = _mascara_filtros_setup(df, filtros, eventos,
                                                avisos=avisos)
    assert not m_long[0] and not m_long[1] and not m_long[2]
    assert not any('menor que el timeframe' in a for a in avisos)
