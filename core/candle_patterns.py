"""
core/candle_patterns.py
Detección vectorizada de patrones de velas japonesas y estadística de su
rendimiento forward (hit rate por lag, significancia, edge vs base).

Funciones puras al estilo de core/metrics.py: sin Qt, sin I/O de config,
sin prints — pensadas para correr dentro de un QThread de la GUI o en tests.

Convenciones:
- df de entrada: columnas open/high/low/close (float) indexadas posicionalmente;
  opcionales interpolado, anomalia, ER, hurst.
- dir de un patrón: +1 sesgo alcista, -1 bajista. Los neutros (Doji) se
  interpretan como señal de giro contra la tendencia previa.
- Acierto a lag L: dir=+1 -> close[t+L] > close[t]; dir=-1 -> close[t+L] <
  close[t]. El empate cuenta como fallo.
- Las ocurrencias pueden solaparse (formaciones consecutivas comparten velas):
  el test binomial es aproximado bajo solapamiento, como es estándar en esta
  clase de estudios.
"""
import numpy as np
from scipy import stats as sps

from core.metrics import calcular_umbrales_er

LAGS = (1, 3, 5, 10)
MIN_OCURRENCIAS = 20   # por debajo no se calcula significancia
MIN_OCURRENCIAS_BARRA = 5   # por bloque de calendario en agregar_por_periodo

# Sesiones horarias. hora_fin exclusiva; si hora_inicio > hora_fin la sesión
# cruza medianoche. 'Globex' (día completo) no está aquí: se representa con
# filtro_sesion=None.
#
# 'londres'/'ny' llevan huso horario IANA ('tz') y hora LOCAL de esa plaza
# (8:00-17:00): preparar_contexto convierte cada timestamp UTC a esa zona,
# así que el rango efectivo en UTC se ajusta solo con el cambio de horario
# de verano/invierno (BST/GMT, EDT/EST) — no hace falta mantener dos
# constantes fijas ni recordar en qué mes estamos.
# 'overnight' no representa una plaza con huso horario propio (es solo la
# franja de baja liquidez fuera de Londres/NY): se queda en UTC fijo.
SESIONES = {
    'overnight': {'utc': (1, 9)},
    'londres':   {'tz': 'Europe/London', 'local': (8, 17)},
    'ny':        {'tz': 'America/New_York', 'local': (8, 17)},
}

PARAMS_DEFECTO = {
    'doji_cuerpo_max':      0.10,  # |cuerpo| <= 10% del rango H-L
    'mecha_dominante':      2.0,   # mecha principal >= 2x |cuerpo|
    'mecha_opuesta_max':    0.15,  # mecha contraria <= 15% del rango
    'cuerpo_grande_min':    1.0,   # |cuerpo| >= 1.0x cuerpo medio rodante
    'cuerpo_pequeno_max':   0.30,  # estrella central <= 30% del cuerpo grande
    'ventana_cuerpo_medio': 20,    # rolling mean de |cuerpo|
    'ventana_tendencia':    5,     # velas del contexto de tendencia previa
    'soldados_mecha_max':   0.30,  # mecha <= 30% del rango en soldados/cuervos
    'piercing_penetracion_min': 0.5,   # cierre penetra >= 50% del cuerpo previo
    'tweezer_tol':          0.10,  # |h[t]-h[t-1]| <= 10% del rango para "igual"
    'marubozu_mecha_max':   0.05,  # mechas <= 5% del rango
    'spinning_mecha_min':   1.5,   # ambas mechas >= 1.5x |cuerpo|
    'spinning_asimetria_max': 0.40,  # |mecha_sup-mecha_inf| <= 40% de la mecha mayor
    'tres_metodos_mecha_max': 0.30,  # mecha <= 30% del rango en las velas de contracción
    'doji_direccional_mecha_min': 0.6,  # mecha dominante >= 60% del rango (dragonfly/gravestone)
}

# Metadatos de los 32 patrones: (nº velas de la formación, sesgo teórico)
# dir 0 = depende del contexto (giro contra-tendencia).
PATRONES_INFO = {
    'Doji':                  {'velas': 1, 'dir': 0},
    'Martillo':              {'velas': 1, 'dir': +1},
    'Hombre Colgado':        {'velas': 1, 'dir': -1},
    'Martillo Invertido':    {'velas': 1, 'dir': +1},
    'Estrella Fugaz':        {'velas': 1, 'dir': -1},
    'Envolvente Alcista':    {'velas': 2, 'dir': +1},
    'Envolvente Bajista':    {'velas': 2, 'dir': -1},
    'Harami Alcista':        {'velas': 2, 'dir': +1},
    'Harami Bajista':        {'velas': 2, 'dir': -1},
    'Morning Star':          {'velas': 3, 'dir': +1},
    'Evening Star':          {'velas': 3, 'dir': -1},
    'Tres Soldados Blancos': {'velas': 3, 'dir': +1},
    'Tres Cuervos Negros':   {'velas': 3, 'dir': -1},
    'Doji Libélula':         {'velas': 1, 'dir': +1},
    'Doji Lápida':           {'velas': 1, 'dir': -1},
    'Marubozu Alcista':      {'velas': 1, 'dir': +1},
    'Marubozu Bajista':      {'velas': 1, 'dir': -1},
    'Spinning Top':          {'velas': 1, 'dir': 0},
    'Piercing Line':         {'velas': 2, 'dir': +1},
    'Dark Cloud Cover':      {'velas': 2, 'dir': -1},
    'Tweezer Bottom':        {'velas': 2, 'dir': +1},
    'Tweezer Top':           {'velas': 2, 'dir': -1},
    'Kicker Alcista':        {'velas': 2, 'dir': +1},
    'Kicker Bajista':        {'velas': 2, 'dir': -1},
    'Three Inside Up':       {'velas': 3, 'dir': +1},
    'Three Inside Down':     {'velas': 3, 'dir': -1},
    'Three Outside Up':      {'velas': 3, 'dir': +1},
    'Three Outside Down':    {'velas': 3, 'dir': -1},
    'Abandoned Baby Alcista': {'velas': 3, 'dir': +1},
    'Abandoned Baby Bajista': {'velas': 3, 'dir': -1},
    'Rising Three Methods':  {'velas': 5, 'dir': +1},
    'Falling Three Methods': {'velas': 5, 'dir': -1},
}

# Los 13 patrones clásicos, visibles por defecto en la GUI — el resto
# (patrones extendidos) queda oculto tras el botón "Más patrones".
PATRONES_ORIGINALES = frozenset({
    'Doji', 'Martillo', 'Hombre Colgado', 'Martillo Invertido', 'Estrella Fugaz',
    'Envolvente Alcista', 'Envolvente Bajista', 'Harami Alcista', 'Harami Bajista',
    'Morning Star', 'Evening Star', 'Tres Soldados Blancos', 'Tres Cuervos Negros',
})


def _shift(a, k):
    """a desplazada k posiciones (a[t-k] en la posición t). Relleno inicial:
    False para bool, NaN para float — ambos hacen fallar cualquier
    comparación, que es lo que se quiere en los bordes."""
    if k == 0:
        return a
    out = np.empty(len(a), dtype=a.dtype)
    out[:k] = False if a.dtype == bool else np.nan
    out[k:] = a[:-k]
    return out


def preparar_anatomia(o, h, l, c, params=None):
    """Descompone cada vela en cuerpo/mechas/rango y añade el contexto
    (cuerpo medio rodante y tendencia previa). Todo arrays numpy float64,
    con guard rango>0 (una vela plana no participa en ningún patrón)."""
    p = {**PARAMS_DEFECTO, **(params or {})}
    o = np.asarray(o, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    l = np.asarray(l, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)

    cuerpo = c - o
    abs_cuerpo = np.abs(cuerpo)
    rango = h - l
    valida = rango > 0
    mecha_sup = h - np.maximum(o, c)
    mecha_inf = np.minimum(o, c) - l
    alcista = cuerpo > 0
    bajista = cuerpo < 0

    # cuerpo medio rodante (escala tipo ATR del cuerpo) sin pandas:
    # media de |cuerpo| de las `ventana` velas ANTERIORES (excluye la actual
    # para no contaminar la comparación "cuerpo grande").
    ventana = p['ventana_cuerpo_medio']
    csum = np.concatenate(([0.0], np.cumsum(abs_cuerpo)))
    cuerpo_medio = np.full(len(c), np.nan)
    if len(c) > ventana:
        cuerpo_medio[ventana:] = (csum[ventana:-1] - csum[:-ventana - 1]) / ventana

    # tendencia previa: signo del retorno de las `vt` velas ANTERIORES a la
    # vela t (c[t-1] vs c[t-1-vt]); 0 si sin datos o plana.
    vt = p['ventana_tendencia']
    tendencia = np.zeros(len(c), dtype=np.int8)
    if len(c) > vt + 1:
        delta = c[vt:-1] - c[:-vt - 1]      # posición t-1 vs t-1-vt
        tendencia[vt + 1:] = np.sign(np.nan_to_num(delta)).astype(np.int8)

    return {
        'o': o, 'h': h, 'l': l, 'c': c,
        'cuerpo': cuerpo, 'abs_cuerpo': abs_cuerpo, 'rango': rango,
        'valida': valida, 'mecha_sup': mecha_sup, 'mecha_inf': mecha_inf,
        'alcista': alcista, 'bajista': bajista,
        'cuerpo_medio': cuerpo_medio, 'tendencia': tendencia,
        'params': p,
    }


def detectar_patrones(o, h, l, c, params=None):
    """Detecta los 13 patrones clásicos sobre arrays OHLC.

    Devuelve {nombre: {'idx': int64[], 'dir': int8[]}} donde idx es la
    posición de la ÚLTIMA vela de la formación y dir el sesgo esperado de
    cada ocurrencia (+1 alcista / -1 bajista).
    """
    a = preparar_anatomia(o, h, l, c, params)
    p = a['params']
    cu, r = a['abs_cuerpo'], a['rango']
    ms, mi = a['mecha_sup'], a['mecha_inf']
    alc, baj, val = a['alcista'], a['bajista'], a['valida']
    T = a['tendencia']
    cm = a['cuerpo_medio']
    o_, c_ = a['o'], a['c']
    h_, l_ = a['h'], a['l']

    def s(arr, k):
        return _shift(arr, k)

    resultados = {}

    def add(nombre, mask, direccion):
        idx = np.flatnonzero(mask)
        if np.isscalar(direccion):
            dirs = np.full(len(idx), direccion, dtype=np.int8)
        else:
            dirs = direccion[idx].astype(np.int8)
        resultados[nombre] = {'idx': idx.astype(np.int64), 'dir': dirs}

    # ── 1 vela ──
    doji = val & (cu <= p['doji_cuerpo_max'] * r) & (T != 0)
    add('Doji', doji, -T)

    # martillo: cuerpo real con mecha inferior dominante y casi sin mecha sup.
    geo_martillo = val & (cu > 0) & (mi >= p['mecha_dominante'] * cu) \
        & (ms <= p['mecha_opuesta_max'] * r)
    add('Martillo', geo_martillo & (T < 0), +1)
    add('Hombre Colgado', geo_martillo & (T > 0), -1)

    geo_invertido = val & (cu > 0) & (ms >= p['mecha_dominante'] * cu) \
        & (mi <= p['mecha_opuesta_max'] * r)
    add('Martillo Invertido', geo_invertido & (T < 0), +1)
    add('Estrella Fugaz', geo_invertido & (T > 0), -1)

    # doji direccional: mecha dominante casi pura (a diferencia de martillo/
    # estrella fugaz, aquí el cuerpo es ínfimo como en el Doji genérico, no
    # solo pequeño). Sesgo fijo por geometría, sin depender de la tendencia.
    doji_geo = val & (cu <= p['doji_cuerpo_max'] * r)
    add('Doji Libélula', doji_geo & (mi >= p['doji_direccional_mecha_min'] * r)
        & (ms <= p['mecha_opuesta_max'] * r), +1)
    add('Doji Lápida', doji_geo & (ms >= p['doji_direccional_mecha_min'] * r)
        & (mi <= p['mecha_opuesta_max'] * r), -1)

    # marubozu: cuerpo grande sin apenas mechas — fuerza direccional pura.
    geo_marubozu = val & (cu >= p['cuerpo_grande_min'] * cm) \
        & (ms <= p['marubozu_mecha_max'] * r) & (mi <= p['marubozu_mecha_max'] * r)
    add('Marubozu Alcista', geo_marubozu & alc, +1)
    add('Marubozu Bajista', geo_marubozu & baj, -1)

    # spinning top: cuerpo pequeño (no ínfimo, para no solapar con Doji) con
    # ambas mechas dominantes y parecidas entre sí (indecisión simétrica).
    spinning = val & (cu > p['doji_cuerpo_max'] * r) & (cu <= p['cuerpo_pequeno_max'] * r) \
        & (ms >= p['spinning_mecha_min'] * cu) & (mi >= p['spinning_mecha_min'] * cu) \
        & (np.abs(ms - mi) <= p['spinning_asimetria_max'] * np.maximum(ms, mi)) \
        & (T != 0)
    add('Spinning Top', spinning, -T)

    # ── 2 velas ──
    val1 = s(val, 1)
    env_alc = val & val1 & s(baj, 1) & alc \
        & (o_ <= s(c_, 1)) & (c_ >= s(o_, 1)) & (cu > s(cu, 1))
    add('Envolvente Alcista', env_alc, +1)
    env_baj = val & val1 & s(alc, 1) & baj \
        & (o_ >= s(c_, 1)) & (c_ <= s(o_, 1)) & (cu > s(cu, 1))
    add('Envolvente Bajista', env_baj, -1)

    # three outside up/down: envolvente + 3ª vela que continúa cerrando más
    # allá del cierre de la vela envolvente.
    add('Three Outside Up', val & s(env_alc, 1) & alc & (c_ > s(c_, 1)), +1)
    add('Three Outside Down', val & s(env_baj, 1) & baj & (c_ < s(c_, 1)), -1)

    # harami: vela previa de cuerpo grande que contiene el cuerpo actual
    grande1 = s(cu, 1) >= p['cuerpo_grande_min'] * s(cm, 1)
    contenido_baj1 = (np.maximum(o_, c_) <= s(o_, 1)) & (np.minimum(o_, c_) >= s(c_, 1))
    contenido_alc1 = (np.maximum(o_, c_) <= s(c_, 1)) & (np.minimum(o_, c_) >= s(o_, 1))
    harami_alc = val & val1 & s(baj, 1) & grande1 & contenido_baj1
    harami_baj = val & val1 & s(alc, 1) & grande1 & contenido_alc1
    add('Harami Alcista', harami_alc, +1)
    add('Harami Bajista', harami_baj, -1)

    # three inside up/down: harami + 3ª vela que confirma cerrando más allá
    # del open de la vela grande inicial (dos posiciones atrás).
    add('Three Inside Up', val & s(harami_alc, 1) & (c_ > s(o_, 2)), +1)
    add('Three Inside Down', val & s(harami_baj, 1) & (c_ < s(o_, 2)), -1)

    # piercing line / dark cloud cover: gap contra la vela previa grande y
    # cierre que penetra más allá de su punto medio, sin llegar a envolverla.
    mid1 = (s(o_, 1) + s(c_, 1)) / 2.0
    piercing = val & val1 & s(baj, 1) & grande1 & alc \
        & (o_ < s(c_, 1)) & (c_ > mid1) & (c_ < s(o_, 1))
    add('Piercing Line', piercing, +1)
    dark_cloud = val & val1 & s(alc, 1) & grande1 & baj \
        & (o_ > s(c_, 1)) & (c_ < mid1) & (c_ > s(o_, 1))
    add('Dark Cloud Cover', dark_cloud, -1)

    # tweezer: máximos (top) o mínimos (bottom) casi idénticos en dos velas
    # consecutivas, en la tendencia previa contraria al giro esperado.
    tweezer_top = val & val1 & (np.abs(h_ - s(h_, 1)) <= p['tweezer_tol'] * r) & (T > 0)
    tweezer_bottom = val & val1 & (np.abs(l_ - s(l_, 1)) <= p['tweezer_tol'] * r) & (T < 0)
    add('Tweezer Top', tweezer_top, -1)
    add('Tweezer Bottom', tweezer_bottom, +1)

    # kicker: cambio de color con gap real entre cuerpos (sin solape de
    # rango cuerpo a cuerpo). Con datos intradía sin gaps reales puede dar
    # pocas o ninguna ocurrencia salvo en TFs altas (1d/1w) — no es un bug.
    cuerpo_grande_actual = cu >= p['cuerpo_grande_min'] * cm
    kicker_alc = val & val1 & s(baj, 1) & grande1 & alc & cuerpo_grande_actual \
        & (np.minimum(o_, c_) > np.maximum(s(o_, 1), s(c_, 1)))
    kicker_baj = val & val1 & s(alc, 1) & grande1 & baj & cuerpo_grande_actual \
        & (np.maximum(o_, c_) < np.minimum(s(o_, 1), s(c_, 1)))
    add('Kicker Alcista', kicker_alc, +1)
    add('Kicker Bajista', kicker_baj, -1)

    # ── 3 velas ──
    val2 = s(val, 2)
    grande2 = s(cu, 2) >= p['cuerpo_grande_min'] * s(cm, 2)
    estrella1 = s(cu, 1) <= p['cuerpo_pequeno_max'] * s(cu, 2)
    mid2 = (s(o_, 2) + s(c_, 2)) / 2.0
    morning = val & val1 & val2 & s(baj, 2) & grande2 & estrella1 \
        & alc & (c_ > mid2)
    add('Morning Star', morning, +1)
    evening = val & val1 & val2 & s(alc, 2) & grande2 & estrella1 \
        & baj & (c_ < mid2)
    add('Evening Star', evening, -1)

    # abandoned baby: versión con gap real (en vez de solo cuerpo pequeño)
    # del morning/evening star — la vela central queda aislada a ambos lados.
    estrella_doji1 = s(cu, 1) <= p['doji_cuerpo_max'] * s(r, 1)
    gap_bajada_a_estrella = np.maximum(s(o_, 1), s(c_, 1)) < np.minimum(s(o_, 2), s(c_, 2))
    gap_estrella_a_subida = np.minimum(o_, c_) > np.maximum(s(o_, 1), s(c_, 1))
    abandoned_alc = val & val1 & val2 & s(baj, 2) & grande2 & estrella_doji1 \
        & alc & gap_bajada_a_estrella & gap_estrella_a_subida
    add('Abandoned Baby Alcista', abandoned_alc, +1)

    gap_subida_a_estrella = np.minimum(s(o_, 1), s(c_, 1)) > np.maximum(s(o_, 2), s(c_, 2))
    gap_estrella_a_bajada = np.maximum(o_, c_) < np.minimum(s(o_, 1), s(c_, 1))
    abandoned_baj = val & val1 & val2 & s(alc, 2) & grande2 & estrella_doji1 \
        & baj & gap_subida_a_estrella & gap_estrella_a_bajada
    add('Abandoned Baby Bajista', abandoned_baj, -1)

    mech_ok_sup = ms <= p['soldados_mecha_max'] * r
    mech_ok_inf = mi <= p['soldados_mecha_max'] * r
    soldados = val & val1 & val2 & alc & s(alc, 1) & s(alc, 2) \
        & (c_ > s(c_, 1)) & (s(c_, 1) > s(c_, 2)) \
        & (o_ >= s(o_, 1)) & (o_ <= s(c_, 1)) \
        & (s(o_, 1) >= s(o_, 2)) & (s(o_, 1) <= s(c_, 2)) \
        & mech_ok_sup & s(mech_ok_sup, 1) & s(mech_ok_sup, 2)
    add('Tres Soldados Blancos', soldados, +1)
    cuervos = val & val1 & val2 & baj & s(baj, 1) & s(baj, 2) \
        & (c_ < s(c_, 1)) & (s(c_, 1) < s(c_, 2)) \
        & (o_ <= s(o_, 1)) & (o_ >= s(c_, 1)) \
        & (s(o_, 1) <= s(o_, 2)) & (s(o_, 1) >= s(c_, 2)) \
        & mech_ok_inf & s(mech_ok_inf, 1) & s(mech_ok_inf, 2)
    add('Tres Cuervos Negros', cuervos, -1)

    # ── 5 velas (continuación) ──
    # rising/falling three methods: vela grande + 3 velas pequeñas de
    # contracción contenidas en su rango + vela grande final que continúa
    # en la misma dirección que la primera.
    val3, val4 = s(val, 3), s(val, 4)
    rango_hi0, rango_lo0 = s(h_, 4), s(l_, 4)
    grande0 = s(cu, 4) >= p['cuerpo_grande_min'] * s(cm, 4)
    contraccion = (
        (s(h_, 3) <= rango_hi0) & (s(l_, 3) >= rango_lo0) & (s(cu, 3) < s(cu, 4))
        & (s(h_, 2) <= rango_hi0) & (s(l_, 2) >= rango_lo0) & (s(cu, 2) < s(cu, 4))
        & (s(h_, 1) <= rango_hi0) & (s(l_, 1) >= rango_lo0) & (s(cu, 1) < s(cu, 4))
    )
    cuerpo_grande_final = cu >= p['cuerpo_grande_min'] * cm
    rising = val & val1 & val2 & val3 & val4 & s(alc, 4) & grande0 \
        & contraccion & alc & cuerpo_grande_final & (c_ > s(c_, 4))
    add('Rising Three Methods', rising, +1)
    falling = val & val1 & val2 & val3 & val4 & s(baj, 4) & grande0 \
        & contraccion & baj & cuerpo_grande_final & (c_ < s(c_, 4))
    add('Falling Three Methods', falling, -1)

    return resultados


def preparar_contexto(close, interpolado=None, anomalia=None,
                      er=None, hurst=None, lags=LAGS, timestamps=None,
                      umbrales_er=None):
    """Precomputa por vela todo lo que la estadística necesita, para que el
    cambio de filtros en la GUI sea O(n_ocurrencias) y no relea nada.

    Devuelve dict con, por lag: fwd_up/fwd_dn (bool), fwd_ret (float32 log),
    fwd_ok (bool: existe t+lag y esa vela es limpia); y por vela:
    vela_valida (bool), regimen_er / regimen_hurst (int8: 0 contra-régimen,
    1 neutro, 2 tendencia; None si falta la columna), hora_utc (int8 0-23,
    None si no se pasa `timestamps` — misma semántica que la columna
    hora_utc de library/scripts_utiles/limpieza_datos_er.py, calculada al
    vuelo para que funcione igual tras un resample a otra temporalidad),
    umbrales_er: (ruido, tendencia) absolutos para clasificar el régimen ER.
    Sin él se usan los adaptativos de calcular_umbrales_er (media ± 1σ de la
    propia serie), que es lo que espera la pestaña Patrones.
    hora_local (dict {clave_sesion: int8[0-23]} para las sesiones de
    SESIONES con huso horario propio — 'londres'/'ny' — convertido con la
    zona IANA correspondiente, así que ya incorpora el cambio de horario de
    verano/invierno vela a vela; dict vacío si no se pasa `timestamps`).
    """
    c = np.asarray(close, dtype=np.float64)
    n = len(c)

    ctx_hora_utc = None
    ctx_hora_local = {}
    if timestamps is not None:
        import pandas as pd
        dt_idx = pd.DatetimeIndex(timestamps)
        ctx_hora_utc = dt_idx.hour.values.astype(np.int8)
        dt_utc = dt_idx.tz_localize('UTC') if dt_idx.tz is None else dt_idx.tz_convert('UTC')
        for clave, cfg in SESIONES.items():
            if 'tz' in cfg:
                ctx_hora_local[clave] = dt_utc.tz_convert(cfg['tz']).hour.values.astype(np.int8)

    limpia = np.ones(n, dtype=bool)
    if interpolado is not None:
        limpia &= ~(np.nan_to_num(np.asarray(interpolado, dtype=np.float64)) > 0)
    if anomalia is not None:
        limpia &= ~(np.nan_to_num(np.asarray(anomalia, dtype=np.float64)) > 0)

    ctx = {'n': n, 'vela_valida': limpia, 'hora_utc': ctx_hora_utc,
           'hora_local': ctx_hora_local,
           'fwd_up': {}, 'fwd_dn': {}, 'fwd_ret': {}, 'fwd_ok': {}}
    with np.errstate(divide='ignore', invalid='ignore'):
        for lag in lags:
            up = np.zeros(n, dtype=bool)
            dn = np.zeros(n, dtype=bool)
            ret = np.zeros(n, dtype=np.float32)
            ok = np.zeros(n, dtype=bool)
            if n > lag:
                up[:-lag] = c[lag:] > c[:-lag]
                dn[:-lag] = c[lag:] < c[:-lag]
                ret[:-lag] = np.log(c[lag:] / c[:-lag]).astype(np.float32)
                ok[:-lag] = limpia[lag:]
            ctx['fwd_up'][lag] = up
            ctx['fwd_dn'][lag] = dn
            ctx['fwd_ret'][lag] = ret
            ctx['fwd_ok'][lag] = ok

    ctx['regimen_er'] = None
    if er is not None:
        er_arr = np.nan_to_num(np.asarray(er, dtype=np.float64), nan=0.0)
        if np.nanstd(er_arr) > 0:
            if umbrales_er is not None:
                # umbrales ABSOLUTOS impuestos por el llamador (los usa el
                # filtro de régimen del Backtester): "tendencia" significa
                # siempre lo mismo, sin depender de la distribución de ER de
                # este activo ni, por tanto, de datos futuros.
                u_ruido, u_tendencia = (float(x) for x in umbrales_er)
            else:
                # por defecto, media ± 1σ de la propia serie — es lo que
                # espera la pestaña Patrones, que enseña el umbral resultante
                # en la etiqueta de su desplegable
                import pandas as pd
                u = calcular_umbrales_er(pd.Series(er_arr))
                u_ruido, u_tendencia = u['umbral_ruido'], u['umbral_tendencia']
            reg = np.ones(n, dtype=np.int8)   # 1 = neutro
            reg[er_arr > u_tendencia] = 2     # tendencia
            reg[er_arr < u_ruido] = 0         # ruido
            ctx['regimen_er'] = reg
            ctx['umbrales_er'] = (u_ruido, u_tendencia)

    ctx['regimen_hurst'] = None
    if hurst is not None:
        h_arr = np.nan_to_num(np.asarray(hurst, dtype=np.float64), nan=0.5)
        # mismos umbrales fijos que contar_regimen_hurst (core/metrics.py)
        reg = np.ones(n, dtype=np.int8)   # 1 = paseo aleatorio
        reg[h_arr > 0.58] = 2             # tendencia
        reg[h_arr < 0.52] = 0             # mean reversion
        ctx['regimen_hurst'] = reg

    return ctx


def _mascara_sesion(ctx, sesion):
    """Máscara horaria para una sesión de SESIONES (None = sin filtro,
    'Globex'/día completo). Soporta sesiones que cruzan medianoche.
    Para sesiones con huso horario propio ('londres'/'ny') usa la hora YA
    convertida a esa zona (ctx['hora_local'], calculada en
    preparar_contexto) — el rango efectivo en UTC se ajusta solo con el
    horario de verano/invierno de esa plaza."""
    if sesion is None or sesion not in SESIONES:
        return None
    cfg = SESIONES[sesion]
    if 'tz' in cfg:
        horas = ctx.get('hora_local', {}).get(sesion)
        h_ini, h_fin = cfg['local']
    else:
        horas = ctx.get('hora_utc')
        h_ini, h_fin = cfg['utc']
    if horas is None:
        return None
    if h_ini <= h_fin:
        return (horas >= h_ini) & (horas < h_fin)
    return (horas >= h_ini) | (horas < h_fin)   # cruza medianoche


def _mascara_filtro(ctx, filtro_er, filtro_hurst, solo_limpias, filtro_sesion=None):
    """Máscara por vela con los filtros de régimen activos (None = todos)."""
    m = np.ones(ctx['n'], dtype=bool)
    if solo_limpias:
        m &= ctx['vela_valida']
    if filtro_er is not None and ctx['regimen_er'] is not None:
        m &= ctx['regimen_er'] == filtro_er
    if filtro_hurst is not None and ctx['regimen_hurst'] is not None:
        m &= ctx['regimen_hurst'] == filtro_hurst
    if filtro_sesion is not None:
        m_sesion = _mascara_sesion(ctx, filtro_sesion)
        if m_sesion is not None:
            m &= m_sesion
    return m


def preparar_base_filtro(ctx, filtro_er=None, filtro_hurst=None,
                         solo_limpias=True, lags=LAGS, filtro_sesion=None):
    """Estadística del universo comparable bajo un filtro (drift y p_up por
    lag). Se calcula UNA vez por combinación de filtros y se comparte entre
    los 13 patrones: es lo único O(n_velas) del refresco de la GUI."""
    m = _mascara_filtro(ctx, filtro_er, filtro_hurst, solo_limpias, filtro_sesion)
    base = {'mascara': m, 'por_lag': {}}
    for lag in lags:
        univ = m & ctx['fwd_ok'][lag]
        n = int(univ.sum())
        if n > 0:
            base['por_lag'][lag] = (float(ctx['fwd_ret'][lag][univ].mean()),
                                    float(ctx['fwd_up'][lag][univ].mean()))
        else:
            base['por_lag'][lag] = (0.0, 0.5)
    return base


def calcular_stats_patron(occ, ctx, filtro_er=None, filtro_hurst=None,
                          solo_limpias=True, lags=LAGS,
                          velas_formacion=1, base=None, filtro_sesion=None):
    """Estadística de un patrón bajo los filtros dados.

    occ: {'idx','dir'} de detectar_patrones. filtro_er/filtro_hurst: valor
    int8 del régimen (0/1/2) o None = sin filtro. Devuelve:
    {'n_total': ocurrencias que pasan el filtro,
     'por_lag': {lag: {'n','hits','hit_rate','p_vs_50','p_vs_base',
                       'edge','ret_fwd_medio','ret_base','significativo',
                       'idx','dir','aciertos','signed_ret'}}}

    'idx'/'dir'/'aciertos'/'signed_ret': arrays crudos por ocurrencia (una
    posición por ocurrencia con t+lag disponible), pensados para agregarlos
    por bloques de calendario con agregar_por_periodo — no para acumularlos
    directamente sobre toda la muestra: un promedio expandido converge por
    ley de los grandes números a la media de largo plazo según crece n, y
    acaba aplanado aunque el patrón haya dejado de funcionar en alguna
    época; agregar_por_periodo agrupa por bloques fijos de tiempo en vez de
    acumular desde el origen para evitar justo ese efecto.

    filtro_sesion: clave de SESIONES ('overnight'/'londres'/'ny') o None
    (Globex, día completo, sin filtro).

    base: resultado de preparar_base_filtro con los MISMOS filtros (incluido
    filtro_sesion); si es None se calcula aquí (cómodo para tests, evitar en
    bucles de GUI).
    """
    if base is None:
        base = preparar_base_filtro(ctx, filtro_er, filtro_hurst,
                                    solo_limpias, lags, filtro_sesion)
    base_m = base['mascara']

    idx, dirs = occ['idx'], occ['dir']
    # la ocurrencia debe pasar el filtro en su vela final y, si se excluyen
    # velas sucias, en TODAS las velas de la formación
    keep = base_m[idx]
    if solo_limpias and velas_formacion > 1:
        for k in range(1, velas_formacion):
            prev = idx - k
            keep &= (prev >= 0) & ctx['vela_valida'][np.maximum(prev, 0)]
    idx, dirs = idx[keep], dirs[keep]

    out = {'n_total': int(len(idx)), 'por_lag': {}}
    for lag in lags:
        ok = ctx['fwd_ok'][lag][idx]
        i_l, d_l = idx[ok], dirs[ok]
        n = int(len(i_l))
        stats_lag = {'n': n, 'hits': 0, 'hit_rate': None, 'p_vs_50': None,
                     'p_vs_base': None, 'edge': None, 'ret_fwd_medio': None,
                     'ret_base': None, 'significativo': False,
                     'idx': i_l, 'dir': d_l,
                     'aciertos': np.array([], dtype=bool),
                     'signed_ret': np.array([], dtype=np.float32)}
        if n == 0:
            out['por_lag'][lag] = stats_lag
            continue

        up, dn = ctx['fwd_up'][lag][i_l], ctx['fwd_dn'][lag][i_l]
        aciertos = np.where(d_l > 0, up, dn)
        hits = int(aciertos.sum())
        stats_lag['hits'] = hits
        stats_lag['hit_rate'] = hits / n
        stats_lag['aciertos'] = aciertos

        # retorno forward firmado por la dirección esperada, y drift base del
        # universo comparable (velas que pasan el mismo filtro y tienen t+lag)
        signed_ret = d_l * ctx['fwd_ret'][lag][i_l]
        stats_lag['signed_ret'] = signed_ret
        ret_fwd = float(np.mean(signed_ret))
        drift_base, p_up_base = base['por_lag'][lag]
        stats_lag['ret_fwd_medio'] = ret_fwd
        stats_lag['ret_base'] = drift_base
        # edge: mejora del retorno en la dirección apostada frente a haber
        # apostado esa misma dirección sin señal (drift firmado por dir media)
        stats_lag['edge'] = ret_fwd - float(np.mean(d_l)) * drift_base

        if n >= MIN_OCURRENCIAS:
            stats_lag['p_vs_50'] = float(
                sps.binomtest(hits, n, 0.5, alternative='two-sided').pvalue)
            # null direccional: P(acierto_i) = p_up si dir=+1, 1-p_up si dir=-1
            p_i = np.where(d_l > 0, p_up_base, 1.0 - p_up_base)
            mu, var = float(p_i.sum()), float((p_i * (1 - p_i)).sum())
            if var > 0:
                z = (hits - mu) / np.sqrt(var)
                stats_lag['p_vs_base'] = float(2.0 * sps.norm.sf(abs(z)))
            stats_lag['significativo'] = (
                stats_lag['p_vs_50'] < 0.05
                and stats_lag['p_vs_base'] is not None
                and stats_lag['p_vs_base'] < 0.05)
        out['por_lag'][lag] = stats_lag

    return out


def agregar_por_periodo(idx, dir_arr, aciertos, signed_ret, timestamps, regla,
                        drift_base, min_ocurrencias=MIN_OCURRENCIAS_BARRA):
    """Agrega las ocurrencias de un patrón (arrays 'idx'/'dir'/'aciertos'/
    'signed_ret' de un lag de calcular_stats_patron) en bloques FIJOS de
    calendario en vez de una ventana móvil por nº de ocurrencias — evita que
    épocas con el patrón muy frecuente aplasten a épocas con pocas
    ocurrencias, y que el eje temporal quede irregularmente espaciado.

    regla: código de resample de pandas ('1D'/'1W'/'1ME'/'1QE'/'1YE'...).
    drift_base: mismo 'ret_base' que devuelve calcular_stats_patron para
    este lag — el edge de cada bloque se mide contra ese mismo drift global,
    no contra un drift recalculado por bloque (con pocas ocurrencias por
    bloque el drift local sería demasiado ruidoso).

    Bloques con menos de min_ocurrencias se omiten: con muy pocas
    ocurrencias el hit rate de ese bloque es ruido, no señal.

    Devuelve {'fechas','fecha_ini','fecha_fin','n','hit_rate','edge_pb'} —
    arrays paralelos, uno por bloque que pasa el mínimo. 'fecha_ini'/'fecha_fin'
    son los bordes REALES del bin que usó pandas: quien dibuje puede pintar una
    barra que ocupe exactamente el bloque de calendario que representa, en vez
    de estimar un ancho a partir de la regla (los meses/trimestres/años no duran
    todos lo mismo, y 'fechas' es la etiqueta del bin, no su inicio)."""
    vacio = {'fechas': np.array([], dtype='datetime64[ns]'),
             'fecha_ini': np.array([], dtype='datetime64[ns]'),
             'fecha_fin': np.array([], dtype='datetime64[ns]'),
             'n': np.array([], dtype=np.int64),
             'hit_rate': np.array([], dtype=np.float64),
             'edge_pb': np.array([], dtype=np.float64)}
    if len(idx) == 0:
        return vacio

    import pandas as pd
    ts = pd.DatetimeIndex(np.asarray(timestamps)[idx])
    df = pd.DataFrame({
        'acierto': aciertos.astype(np.float64),
        'dir': dir_arr.astype(np.float64),
        'signed_ret': signed_ret.astype(np.float64),
    }, index=ts)

    g = df.resample(regla)
    n = g.size()
    mask = n >= min_ocurrencias
    if not mask.any():
        return vacio

    hit_rate = g['acierto'].mean()
    ret_fwd_medio = g['signed_ret'].mean()
    dir_medio = g['dir'].mean()
    edge = ret_fwd_medio - dir_medio * drift_base

    # bordes reales de cada bin. 'label' no cambia el binning, solo qué extremo
    # se usa como etiqueta, así que este índice va bin a bin en paralelo con el
    # de arriba; y como los bins son contiguos y completos (pandas rellena los
    # vacíos), el fin de uno es el inicio del siguiente — basta desplazar el
    # índice un periodo en vez de resamplear otra vez sobre todas las filas.
    ini_all = df.resample(regla, label='left').size().index
    fin_all = ini_all.shift(1, freq=regla)

    return {
        'fechas': n.index.values[mask.values],
        'fecha_ini': ini_all.values[mask.values],
        'fecha_fin': fin_all.values[mask.values],
        'n': n.values[mask.values].astype(np.int64),
        'hit_rate': hit_rate.values[mask.values],
        'edge_pb': edge.values[mask.values] * 10000,
    }
