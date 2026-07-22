"""
core/strategies.py
Catálogo de indicadores y registro declarativo de estrategias para el motor
de backtest (core/backtest.py). Sin Qt, sin I/O — la GUI genera los
formularios a partir de la spec de parámetros de cada estrategia.

Cada estrategia del registro ESTRATEGIAS:
- 'params': lista de specs {clave, etiqueta, tipo ('int'|'float'|'choice'|
  'patrones'), defecto, min/max u opciones} → formulario autogenerado.
- 'generar': fn(df, params) → dict con entradas_long/entradas_short/
  salidas_long/salidas_short (bool[n]), setup_id (int64[n]), atr (float64[n]).
- 'descripcion': fn(params) → texto legible de las reglas de entrada/salida
  con los parámetros interpolados (se muestra en la GUI para que la
  definición de cada plantilla no sea una caja negra).

Convención de señales: la señal en la vela t la ejecuta el motor al open de
t+1. setup_id etiqueta cada señal de ENTRADA para poder asignar un riesgo
distinto por forma de entrar.

SISTEMAS (multi-setup): generar_senales_sistema(df, setups) fusiona varios
setups (cada uno una plantilla con sus parámetros y su propio riesgo/stop/
TP/salida) en un único juego de señales para el motor: entradas etiquetadas
con el índice del setup, y salidas como BITMASK int64 (bit k = el setup k
pide salir) para que la salida de un setup no cierre la posición de otro.
Máximo MAX_SETUPS setups por sistema (límite del bitmask de 64 bits).
"""
import numpy as np
import pandas as pd

from core.candle_patterns import (
    PATRONES_INFO, detectar_patrones, preparar_contexto, _mascara_sesion,
)
from core.metrics import calcular_er_series, calcular_kama_numba, calcular_hurst_array

PERIODO_ATR_DEFECTO = 14
MAX_SETUPS = 64   # límite del bitmask int64 de salidas por setup


# ══════════════ indicadores (catálogo con defectos) ══════════════

def sma(c, periodo):
    return pd.Series(c).rolling(int(periodo)).mean().values


def ema(c, periodo):
    return pd.Series(c).ewm(span=int(periodo), adjust=False).mean().values


def rsi(c, periodo=14):
    s = pd.Series(c)
    delta = s.diff()
    ganancia = delta.clip(lower=0).ewm(alpha=1.0 / periodo, adjust=False).mean()
    perdida = (-delta.clip(upper=0)).ewm(alpha=1.0 / periodo, adjust=False).mean()
    rs = ganancia / perdida.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0).values


def atr(h, l, c, periodo=PERIODO_ATR_DEFECTO):
    hs, ls, cs = pd.Series(h), pd.Series(l), pd.Series(c)
    tr = pd.concat([hs - ls, (hs - cs.shift(1)).abs(),
                    (ls - cs.shift(1)).abs()], axis=1).max(axis=1)
    return tr.rolling(int(periodo)).mean().bfill().values


def bollinger(c, periodo=20, desv=2.0):
    s = pd.Series(c)
    media = s.rolling(int(periodo)).mean()
    std = s.rolling(int(periodo)).std()
    return media.values, (media + desv * std).values, (media - desv * std).values


def _retorno_log(close):
    """Retornos log cierre-a-cierre — misma convención que
    library/scripts_utiles/limpieza_datos_er.py y tab_patrones.py."""
    c = pd.Series(close, dtype=np.float64)
    return np.log(c / c.shift(1))


def _er_serie(close, periodo):
    """Efficiency Ratio (Kaufman) de `close`, ventana `periodo`."""
    return calcular_er_series(_retorno_log(close), int(periodo))


def _kama_serie(close, periodo_er, rapido, lento):
    """KAMA de `close`: ER interno con ventana periodo_er, SC entre
    2/(rapido+1) y 2/(lento+1)."""
    c = np.asarray(close, dtype=np.float64)
    er = _er_serie(c, periodo_er).values.astype(np.float64)
    return calcular_kama_numba(c, er, float(rapido), float(lento))


def _lags_hurst_defecto(periodo):
    """Deriva lags/paso de Hurst a partir de una única ventana `periodo`
    expuesta en la UI, con proporciones similares a las que usan
    library/scripts_utiles/limpieza_datos_er.py y tab_patrones.py."""
    max_lag = max(8, 2 ** int(np.floor(np.log2(max(periodo // 4, 8)))))
    lags = sorted({l for l in (max_lag // 8, max_lag // 4, max_lag // 2, max_lag) if l >= 4})
    if len(lags) < 2:
        lags = [4, 8]
    paso = max(1, periodo // 100)
    return np.array(lags, dtype=np.int64), paso


def _hurst_serie(close, periodo):
    """Hurst rodante (ventana=periodo) para el filtro de régimen. Devuelve
    None si el histórico es más corto que la ventana pedida — evita que el
    relleno de NaN→0.5 clasifique silenciosamente TODAS las velas como
    régimen neutro y bloquee el filtro de tendencia/reversión sin avisar."""
    n = len(close)
    if n <= periodo:
        return None
    retornos = _retorno_log(close).fillna(0.0).values.astype(np.float64)
    lags, paso = _lags_hurst_defecto(int(periodo))
    hurst_vals = calcular_hurst_array(retornos, int(periodo), paso, lags)
    if np.isnan(hurst_vals).all():
        return None
    return pd.Series(hurst_vals).interpolate(limit_direction='both').bfill().ffill().values


def _cruza_arriba(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    prev = np.roll(a <= b, 1)
    prev[0] = False
    return prev & (a > b)


def _cruza_abajo(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    prev = np.roll(a >= b, 1)
    prev[0] = False
    return prev & (a < b)


def _base_senales(n, h=None, l=None, c=None):
    s = {'entradas_long': np.zeros(n, dtype=bool),
         'entradas_short': np.zeros(n, dtype=bool),
         'salidas_long': np.zeros(n, dtype=bool),
         'salidas_short': np.zeros(n, dtype=bool),
         'setup_id': np.zeros(n, dtype=np.int64)}
    if c is not None:
        s['atr'] = atr(h, l, c)
    else:
        s['atr'] = np.zeros(n)
    return s


def _limpiar_nan(mask, *arrays):
    """Anula señales donde algún indicador aún es NaN (warm-up)."""
    for a in arrays:
        mask &= ~np.isnan(np.asarray(a, dtype=np.float64))
    return mask


# ══════════════ estrategias predeterminadas ══════════════

def _gen_cruce_medias(df, p):
    c = df['close'].values
    n = len(c)
    f = ema if p['tipo'] == 'EMA' else sma
    rapida = f(c, p['rapida'])
    lenta = f(c, p['lenta'])
    s = _base_senales(n, df['high'].values, df['low'].values, c)
    arriba = _cruza_arriba(rapida, lenta)
    abajo = _cruza_abajo(rapida, lenta)
    arriba = _limpiar_nan(arriba, rapida, lenta)
    abajo = _limpiar_nan(abajo, rapida, lenta)
    if p['direccion'] in ('Long', 'Ambas'):
        s['entradas_long'] = arriba.copy()
        s['salidas_long'] = abajo.copy()
    if p['direccion'] in ('Short', 'Ambas'):
        s['entradas_short'] = abajo.copy()
        s['salidas_short'] = arriba.copy()
    return s


def _gen_bollinger(df, p):
    c = df['close'].values
    n = len(c)
    media, sup, inf = bollinger(c, p['periodo'], p['desv'])
    s = _base_senales(n, df['high'].values, df['low'].values, c)
    ent_l = _limpiar_nan(c < inf, inf)
    sal_l = _limpiar_nan(c > media, media)
    ent_s = _limpiar_nan(c > sup, sup)
    sal_s = _limpiar_nan(c < media, media)
    if p['direccion'] in ('Long', 'Ambas'):
        s['entradas_long'], s['salidas_long'] = ent_l, sal_l
    if p['direccion'] in ('Short', 'Ambas'):
        s['entradas_short'], s['salidas_short'] = ent_s, sal_s
    return s


def _gen_rsi(df, p):
    c = df['close'].values
    n = len(c)
    r = rsi(c, p['periodo'])
    s = _base_senales(n, df['high'].values, df['low'].values, c)
    if p['direccion'] in ('Long', 'Ambas'):
        s['entradas_long'] = r < p['sobreventa']
        s['salidas_long'] = r > 50.0
    if p['direccion'] in ('Short', 'Ambas'):
        s['entradas_short'] = r > p['sobrecompra']
        s['salidas_short'] = r < 50.0
    return s


def _gen_kama(df, p):
    """KAMA (Kaufman Adaptive Moving Average): entrada en el cruce de
    close contra la línea KAMA, igual patrón que _gen_cruce_medias pero con
    una media que acelera en tendencia (ER alto) y se aplana en rango (ER
    bajo) en vez de un periodo fijo."""
    c = df['close'].values.astype(np.float64)
    n = len(c)
    kama = _kama_serie(c, p['periodo_er'], p['rapido'], p['lento'])
    s = _base_senales(n, df['high'].values, df['low'].values, c)
    arriba = _cruza_arriba(c, kama)
    abajo = _cruza_abajo(c, kama)
    arriba = _limpiar_nan(arriba, kama)
    abajo = _limpiar_nan(abajo, kama)
    if p['direccion'] in ('Long', 'Ambas'):
        s['entradas_long'] = arriba.copy()
        s['salidas_long'] = abajo.copy()
    if p['direccion'] in ('Short', 'Ambas'):
        s['entradas_short'] = abajo.copy()
        s['salidas_short'] = arriba.copy()
    return s


def _gen_patrones(df, p):
    """Entradas en cada ocurrencia de los patrones elegidos (dir del patrón
    decide long/short); salida por tiempo la gestiona el motor con
    salida_n_velas (el formulario del Optimizador la fija con el lag).
    setup_id = índice del patrón en la lista elegida → permite riesgo
    distinto por patrón."""
    o, h, l, c = (df['open'].values, df['high'].values,
                  df['low'].values, df['close'].values)
    n = len(c)
    s = _base_senales(n, h, l, c)
    detectados = detectar_patrones(o, h, l, c)
    for k, nombre in enumerate(p['patrones']):
        occ = detectados.get(nombre)
        if occ is None:
            continue
        idx, dirs = occ['idx'], occ['dir']
        s['entradas_long'][idx[dirs > 0]] = True
        s['entradas_short'][idx[dirs < 0]] = True
        s['setup_id'][idx] = k
    return s


# ══════════════ estrategia custom (constructor de reglas) ══════════════

_INDICADORES_REGLA = ['close', 'open', 'high', 'low', 'SMA', 'EMA', 'RSI',
                      'ATR', 'BB_sup', 'BB_inf', 'BB_media', 'KAMA', 'ER']
_OPERADORES_REGLA = ['>', '<', 'cruza arriba', 'cruza abajo']

# defaults fijos de KAMA cuando se usa desde el constructor de reglas: la
# tabla del editor solo expone un "Periodo" por indicador, así que el SC
# rápido/lento de KAMA quedan fijos aquí (el «Periodo» de la fila mapea a
# periodo_er). Para variar rápido/lento hace falta la plantilla KAMA dedicada.
_KAMA_RAPIDO_REGLA = 2
_KAMA_LENTO_REGLA = 30


def _serie_indicador(df, spec):
    """spec: {'tipo': 'SMA', 'periodo': 20} o {'tipo': 'close'} o
    {'tipo': 'valor', 'valor': 30.0}."""
    tipo = spec['tipo']
    c = df['close'].values
    if tipo == 'valor':
        return np.full(len(c), float(spec['valor']))
    if tipo in ('close', 'open', 'high', 'low'):
        return df[tipo].values.astype(np.float64)
    periodo = int(spec.get('periodo', 14))
    if tipo == 'SMA':
        return sma(c, periodo)
    if tipo == 'EMA':
        return ema(c, periodo)
    if tipo == 'RSI':
        return rsi(c, periodo)
    if tipo == 'ATR':
        return atr(df['high'].values, df['low'].values, c, periodo)
    if tipo in ('BB_sup', 'BB_inf', 'BB_media'):
        media, sup, inf = bollinger(c, periodo, float(spec.get('desv', 2.0)))
        return {'BB_sup': sup, 'BB_inf': inf, 'BB_media': media}[tipo]
    if tipo == 'KAMA':
        return _kama_serie(c, periodo, _KAMA_RAPIDO_REGLA, _KAMA_LENTO_REGLA)
    if tipo == 'ER':
        return _er_serie(c, periodo).values.astype(np.float64)
    raise ValueError(f"Indicador desconocido: {tipo}")


def _evaluar_condicion(df, cond):
    a = _serie_indicador(df, cond['izq'])
    b = _serie_indicador(df, cond['der'])
    op = cond['op']
    if op == '>':
        m = a > b
    elif op == '<':
        m = a < b
    elif op == 'cruza arriba':
        m = _cruza_arriba(a, b)
    elif op == 'cruza abajo':
        m = _cruza_abajo(a, b)
    else:
        raise ValueError(f"Operador desconocido: {op}")
    return _limpiar_nan(m, a, b)


def _evaluar_reglas(df, condiciones):
    """AND de todas las condiciones de una regla; sin condiciones → nunca."""
    if not condiciones:
        return np.zeros(len(df), dtype=bool)
    m = np.ones(len(df), dtype=bool)
    for cond in condiciones:
        m &= _evaluar_condicion(df, cond)
    return m


def _mascara_condiciones(df, condiciones):
    """AND de condiciones opcionales usadas como FILTRO de un setup — al
    contrario que _evaluar_reglas (que define una señal y sin condiciones
    no dispara nunca), aquí sin condiciones significa SIN RESTRICCIÓN."""
    if not condiciones:
        return np.ones(len(df), dtype=bool)
    m = np.ones(len(df), dtype=bool)
    for cond in condiciones:
        m &= _evaluar_condicion(df, cond)
    return m


def _mascaras_condiciones_dir(df, condiciones):
    """Como _mascara_condiciones pero direccional: cada condición lleva
    'direccion' ('ambas'|'long'|'short'; por defecto 'ambas') y solo entra en
    el AND del lado(s) al que se aplica. Devuelve (m_long, m_short). Sin
    condiciones → (True, True) = sin restricción en ningún lado."""
    n = len(df)
    m_long = np.ones(n, dtype=bool)
    m_short = np.ones(n, dtype=bool)
    for cond in (condiciones or []):
        mask = _evaluar_condicion(df, cond)
        d = cond.get('direccion', 'ambas')
        if d in ('ambas', 'long'):
            m_long &= mask
        if d in ('ambas', 'short'):
            m_short &= mask
    return m_long, m_short


def _gen_custom(df, p):
    """p['reglas']: {'entradas_long': [{'setup_id': 0, 'condiciones': [...]},
    ...], 'salidas_long': [...], 'entradas_short': [...], 'salidas_short':
    [...]} — cada entrada es una lista de setups; cada setup, un AND de
    condiciones y su setup_id (para el riesgo por setup)."""
    n = len(df)
    s = _base_senales(n, df['high'].values, df['low'].values, df['close'].values)
    reglas = p.get('reglas') or {}
    for clave in ('entradas_long', 'entradas_short'):
        for setup in reglas.get(clave, []):
            m = _evaluar_reglas(df, setup.get('condiciones', []))
            s[clave] |= m
            s['setup_id'][m] = int(setup.get('setup_id', 0))
    for clave in ('salidas_long', 'salidas_short'):
        for setup in reglas.get(clave, []):
            s[clave] |= _evaluar_reglas(df, setup.get('condiciones', []))
    return s


# ══════════════ descripciones (definición legible de cada plantilla) ══════════════

def _desc_spec(spec):
    tipo = spec['tipo']
    if tipo == 'valor':
        return f"{spec['valor']:g}"
    if tipo in ('close', 'open', 'high', 'low'):
        return tipo
    if tipo in ('BB_sup', 'BB_inf', 'BB_media'):
        return f"{tipo}({spec.get('periodo', 20)},{spec.get('desv', 2.0):g})"
    return f"{tipo}({spec.get('periodo', 14)})"


def _desc_condicion_dir(cond):
    """Descripción legible de una condición de filtro (condiciones_entrada/
    condiciones_salida), con la etiqueta de dirección al final solo cuando
    restringe un único lado (evita ruido cuando es 'ambas', el caso normal)."""
    texto = f"{_desc_spec(cond['izq'])} {cond['op']} {_desc_spec(cond['der'])}"
    d = cond.get('direccion', 'ambas')
    if d == 'long':
        return f"{texto} [Long]"
    if d == 'short':
        return f"{texto} [Short]"
    return texto


def _desc_cruce(p):
    t, r, l = p['tipo'], p['rapida'], p['lenta']
    partes = []
    if p['direccion'] in ('Long', 'Ambas'):
        partes.append(f"Entrada Long: {t}({r}) cruza arriba {t}({l}) · "
                      f"Salida Long: {t}({r}) cruza abajo {t}({l})")
    if p['direccion'] in ('Short', 'Ambas'):
        partes.append(f"Entrada Short: {t}({r}) cruza abajo {t}({l}) · "
                      f"Salida Short: cruce contrario")
    partes.append("(La salida es el cruce contrario de las medias; "
                  "sin stop ATR por defecto)")
    return "\n".join(partes)


def _desc_bollinger(p):
    per, d = p['periodo'], p['desv']
    partes = []
    if p['direccion'] in ('Long', 'Ambas'):
        partes.append(f"Entrada Long: close < banda inferior BB({per},{d:g}) · "
                      f"Salida Long: close > media BB({per})")
    if p['direccion'] in ('Short', 'Ambas'):
        partes.append(f"Entrada Short: close > banda superior BB({per},{d:g}) · "
                      f"Salida Short: close < media BB({per})")
    return "\n".join(partes) + "\n(Reversión a la media; stop/TP los pone el setup)"


def _desc_rsi(p):
    per = p['periodo']
    partes = []
    if p['direccion'] in ('Long', 'Ambas'):
        partes.append(f"Entrada Long: RSI({per}) < {p['sobreventa']:g} · "
                      f"Salida Long: RSI({per}) > 50")
    if p['direccion'] in ('Short', 'Ambas'):
        partes.append(f"Entrada Short: RSI({per}) > {p['sobrecompra']:g} · "
                      f"Salida Short: RSI({per}) < 50")
    return "\n".join(partes)


def _desc_kama(p):
    per_er, r, l = p['periodo_er'], p['rapido'], p['lento']
    partes = []
    if p['direccion'] in ('Long', 'Ambas'):
        partes.append(f"Entrada Long: close cruza arriba KAMA(ER={per_er}, rápido={r}, lento={l}) · "
                      f"Salida Long: close cruza abajo KAMA")
    if p['direccion'] in ('Short', 'Ambas'):
        partes.append(f"Entrada Short: close cruza abajo KAMA(ER={per_er}, rápido={r}, lento={l}) · "
                      f"Salida Short: close cruza arriba KAMA")
    partes.append("(KAMA se adapta: sigue de cerca en tendencia -ER alto- y "
                  "se aplana en rango -ER bajo-; sin stop ATR por defecto, "
                  "igual criterio que Cruce de medias)")
    return "\n".join(partes)


def _desc_patrones(p):
    pats = ", ".join(p['patrones']) if p['patrones'] else "(ninguno)"
    return (f"Entrada en cada ocurrencia de: {pats} — la dirección la da el "
            f"sesgo del patrón (+1 long / -1 bajista short).\n"
            f"Salida: a +{p['lag_salida']} velas de la entrada (por tiempo).")


def _desc_custom(p):
    reglas = p.get('reglas') or {}
    etiquetas = {'entradas_long': 'Entrada Long', 'salidas_long': 'Salida Long',
                 'entradas_short': 'Entrada Short', 'salidas_short': 'Salida Short'}
    lineas = []
    for clave, etiqueta in etiquetas.items():
        for setup in reglas.get(clave, []):
            conds = " Y ".join(
                f"{_desc_spec(c['izq'])} {c['op']} {_desc_spec(c['der'])}"
                for c in setup.get('condiciones', []))
            if conds:
                lineas.append(f"{etiqueta}: {conds}")
    return "\n".join(lineas) if lineas else "(sin reglas definidas)"


# ══════════════ registro ══════════════

_OPCIONES_DIRECCION = ['Ambas', 'Long', 'Short']

ESTRATEGIAS = {
    'Cruce de medias': {
        'generar': _gen_cruce_medias,
        'descripcion': _desc_cruce,
        # la salida ES el cruce contrario: sin stop ATR por defecto (un stop
        # cortaria los trades antes de que las medias vuelvan a cruzarse)
        'defaults_setup': {'stop_atr': 0.0},
        'params': [
            {'clave': 'tipo', 'etiqueta': 'Tipo de media', 'tipo': 'choice',
             'opciones': ['SMA', 'EMA'], 'defecto': 'SMA'},
            {'clave': 'rapida', 'etiqueta': 'Media rápida', 'tipo': 'int',
             'defecto': 20, 'min': 2, 'max': 500},
            {'clave': 'lenta', 'etiqueta': 'Media lenta', 'tipo': 'int',
             'defecto': 50, 'min': 3, 'max': 1000},
            {'clave': 'direccion', 'etiqueta': 'Dirección', 'tipo': 'choice',
             'opciones': _OPCIONES_DIRECCION, 'defecto': 'Ambas'},
        ],
    },
    'Bollinger + ATR': {
        'generar': _gen_bollinger,
        'descripcion': _desc_bollinger,
        'params': [
            {'clave': 'periodo', 'etiqueta': 'Periodo BB', 'tipo': 'int',
             'defecto': 20, 'min': 5, 'max': 200},
            {'clave': 'desv', 'etiqueta': 'Desviaciones', 'tipo': 'float',
             'defecto': 2.0, 'min': 0.5, 'max': 4.0},
            {'clave': 'direccion', 'etiqueta': 'Dirección', 'tipo': 'choice',
             'opciones': _OPCIONES_DIRECCION, 'defecto': 'Ambas'},
        ],
    },
    'RSI': {
        'generar': _gen_rsi,
        'descripcion': _desc_rsi,
        'params': [
            {'clave': 'periodo', 'etiqueta': 'Periodo RSI', 'tipo': 'int',
             'defecto': 14, 'min': 2, 'max': 100},
            {'clave': 'sobreventa', 'etiqueta': 'Umbral sobreventa', 'tipo': 'float',
             'defecto': 30.0, 'min': 5.0, 'max': 50.0},
            {'clave': 'sobrecompra', 'etiqueta': 'Umbral sobrecompra', 'tipo': 'float',
             'defecto': 70.0, 'min': 50.0, 'max': 95.0},
            {'clave': 'direccion', 'etiqueta': 'Dirección', 'tipo': 'choice',
             'opciones': _OPCIONES_DIRECCION, 'defecto': 'Ambas'},
        ],
    },
    'KAMA': {
        'generar': _gen_kama,
        'descripcion': _desc_kama,
        # la salida es el cruce contrario, igual razón que Cruce de medias:
        # un stop cortaría el trade antes de que KAMA vuelva a cruzarse
        'defaults_setup': {'stop_atr': 0.0},
        'params': [
            {'clave': 'periodo_er', 'etiqueta': 'Periodo ER', 'tipo': 'int',
             'defecto': 10, 'min': 2, 'max': 200},
            {'clave': 'rapido', 'etiqueta': 'SC rápido (velas)', 'tipo': 'int',
             'defecto': 2, 'min': 1, 'max': 50},
            {'clave': 'lento', 'etiqueta': 'SC lento (velas)', 'tipo': 'int',
             'defecto': 30, 'min': 5, 'max': 500},
            {'clave': 'direccion', 'etiqueta': 'Dirección', 'tipo': 'choice',
             'opciones': _OPCIONES_DIRECCION, 'defecto': 'Ambas'},
        ],
    },
    'Patrones de velas': {
        'generar': _gen_patrones,
        'descripcion': _desc_patrones,
        'params': [
            {'clave': 'patrones', 'etiqueta': 'Patrones', 'tipo': 'patrones',
             'defecto': ['Martillo'], 'opciones': list(PATRONES_INFO)},
            # el lag de salida se mapea a config['salida_n_velas'] del motor
            {'clave': 'lag_salida', 'etiqueta': 'Salida a +N velas', 'tipo': 'int',
             'defecto': 5, 'min': 1, 'max': 100},
        ],
    },
    'Custom (reglas)': {
        'generar': _gen_custom,
        'descripcion': _desc_custom,
        'params': [
            # el editor de reglas de la GUI rellena esta estructura
            {'clave': 'reglas', 'etiqueta': 'Reglas', 'tipo': 'reglas',
             'defecto': {'entradas_long': [], 'salidas_long': [],
                         'entradas_short': [], 'salidas_short': []}},
        ],
    },
}


def params_por_defecto(nombre_estrategia):
    return {p['clave']: p['defecto']
            for p in ESTRATEGIAS[nombre_estrategia]['params']}


def generar_senales(nombre_estrategia, df, params=None):
    est = ESTRATEGIAS[nombre_estrategia]
    p = params_por_defecto(nombre_estrategia)
    p.update(params or {})
    return est['generar'](df, p)


def describir(nombre_estrategia, params=None):
    """Definición legible de la plantilla con los parámetros interpolados."""
    est = ESTRATEGIAS[nombre_estrategia]
    p = params_por_defecto(nombre_estrategia)
    p.update(params or {})
    return est['descripcion'](p)


# ══════════════ filtros de entrada por setup ══════════════

def _filtros_por_defecto():
    """Filtros de ENTRADA de un setup — None/'ninguno' en cualquier eje =
    sin restricción en ese eje. Solo se aplican a NUEVAS entradas (ver
    generar_senales_sistema): las salidas nunca se filtran, para no dejar
    una posición abierta sin forma de cerrarse si el régimen/sesión/día
    cambia a mitad de una operación."""
    return {
        'dias_semana': None,      # None = todos; si no, lista de ints 0=Lun..6=Dom
        'regimen': {'metodo': 'ninguno', 'periodo': 100},
        # metodo: 'ninguno'|'er_tendencia'|'er_rango'|'hurst_tendencia'|'hurst_reversion'
        'sesion': {'tipo': 'ninguna', 'hora_inicio': 0, 'hora_fin': 0},
        # tipo: 'ninguna'|'overnight'|'londres'|'ny'|'personalizada' (horas UTC en 'personalizada')
        'condiciones_entrada': [],   # lista de {'izq':spec,'op':...,'der':spec}; AND; [] = sin restricción
        'condiciones_salida': [],    # idem, pero se aplica sobre la señal de SALIDA del setup
    }


def _mascara_filtros_setup(df, filtros):
    """(m_long, m_short): máscaras AND de los filtros activos de un setup
    (True = vela admitida para NUEVAS entradas). Día/régimen/sesión son
    agnósticos a la dirección y se aplican a ambos lados por igual;
    condiciones_entrada puede restringir un solo lado según su 'direccion'
    (ver _mascaras_condiciones_dir). Reutiliza los umbrales/sesiones de
    core.candle_patterns.preparar_contexto para no duplicar convenciones."""
    n = len(df)
    m = np.ones(n, dtype=bool)
    if not filtros:
        return m, m.copy()

    dias = filtros.get('dias_semana')
    if dias:
        dow = df['timestamp'].dt.dayofweek.values
        m &= np.isin(dow, list(dias))

    reg = filtros.get('regimen') or {}
    metodo = reg.get('metodo', 'ninguno')
    ses = filtros.get('sesion') or {}
    tipo_sesion = ses.get('tipo', 'ninguna')

    necesita_er = metodo in ('er_tendencia', 'er_rango')
    necesita_hurst = metodo in ('hurst_tendencia', 'hurst_reversion')
    necesita_ctx = necesita_er or necesita_hurst or tipo_sesion in ('overnight', 'londres', 'ny')

    if necesita_ctx:
        close = df['close'].values.astype(np.float64)
        periodo = int(reg.get('periodo', 100))
        er_vals = _er_serie(close, periodo).values if necesita_er else None
        hurst_vals = _hurst_serie(close, periodo) if necesita_hurst else None
        ctx = preparar_contexto(
            close, er=er_vals, hurst=hurst_vals,
            timestamps=df['timestamp'].values if tipo_sesion != 'ninguna' else None)

        if necesita_er and ctx['regimen_er'] is not None:
            m &= (ctx['regimen_er'] == (2 if metodo == 'er_tendencia' else 0))
        if necesita_hurst and ctx['regimen_hurst'] is not None:
            m &= (ctx['regimen_hurst'] == (2 if metodo == 'hurst_tendencia' else 0))
        if tipo_sesion in ('overnight', 'londres', 'ny'):
            m_sesion = _mascara_sesion(ctx, tipo_sesion)
            if m_sesion is not None:
                m &= m_sesion

    if tipo_sesion == 'personalizada':
        horas = pd.DatetimeIndex(df['timestamp']).hour.values
        h_ini, h_fin = int(ses.get('hora_inicio', 0)), int(ses.get('hora_fin', 0))
        if h_ini <= h_fin:
            m &= (horas >= h_ini) & (horas < h_fin)
        else:
            m &= (horas >= h_ini) | (horas < h_fin)   # cruza medianoche

    mc_long, mc_short = _mascaras_condiciones_dir(df, filtros.get('condiciones_entrada'))
    return m & mc_long, m & mc_short


# ══════════════ sistemas multi-setup ══════════════

def generar_senales_sistema(df, setups):
    """Fusiona las señales de una lista de setups en un único juego para el
    motor. setups: [{'plantilla': nombre de ESTRATEGIAS, 'params': {...}},
    ...] (máx. MAX_SETUPS).

    - entradas_long/short: bool[n]; si dos setups disparan entrada en la
      misma vela, gana el PRIMERO de la lista (prioridad por orden) y el
      resto se descarta en esa vela.
    - setup_id[n]: índice (posición en la lista) del setup que dispara la
      entrada de esa vela.
    - salidas_long/short: BITMASK int64[n] — bit k activo = el setup k pide
      salir. El motor solo cierra por señal si el bit del setup de la
      posición abierta está activo (la salida de un setup no toca la
      posición de otro).
    """
    if len(setups) > MAX_SETUPS:
        raise ValueError(f"Máximo {MAX_SETUPS} setups por sistema")
    n = len(df)
    out = {'entradas_long': np.zeros(n, dtype=bool),
           'entradas_short': np.zeros(n, dtype=bool),
           'salidas_long': np.zeros(n, dtype=np.int64),
           'salidas_short': np.zeros(n, dtype=np.int64),
           'setup_id': np.zeros(n, dtype=np.int64),
           'atr': None}
    reclamada = np.zeros(n, dtype=bool)   # vela ya reclamada por un setup previo

    for k, setup in enumerate(setups):
        s = generar_senales(setup['plantilla'], df, setup.get('params'))
        if out['atr'] is None:
            out['atr'] = s['atr']
        ent_l = np.asarray(s['entradas_long'], dtype=bool)
        ent_s = np.asarray(s['entradas_short'], dtype=bool)
        sal_l = np.asarray(s['salidas_long'], dtype=bool)
        sal_s = np.asarray(s['salidas_short'], dtype=bool)
        filtros = setup.get('filtros')
        if filtros:
            m_ent_long, m_ent_short = _mascara_filtros_setup(df, filtros)
            ent_l = ent_l & m_ent_long
            ent_s = ent_s & m_ent_short
            # condiciones_salida SÍ restringe la salida (a diferencia de
            # día/régimen/sesión): pedido explícito del usuario. El stop/TP/
            # tiempo del setup siguen siendo la red de seguridad si la
            # condición de salida no llega a cumplirse. Cada condición puede
            # ir dirigida solo a long, solo a short, o a ambas (ver
            # 'direccion' en _mascaras_condiciones_dir).
            m_sal_long, m_sal_short = _mascaras_condiciones_dir(
                df, filtros.get('condiciones_salida'))
            sal_l = sal_l & m_sal_long
            sal_s = sal_s & m_sal_short
        nuevas = (ent_l | ent_s) & ~reclamada
        out['entradas_long'] |= ent_l & nuevas
        out['entradas_short'] |= ent_s & ~ent_l & nuevas
        out['setup_id'][nuevas] = k
        reclamada |= nuevas
        bit = np.int64(1) << np.int64(k)
        out['salidas_long'][sal_l] |= bit
        out['salidas_short'][sal_s] |= bit

    if out['atr'] is None:
        out['atr'] = np.zeros(n)
    return out


def _codigo_reglas_plantilla(plantilla, p):
    """Líneas de ENTRADA y SALIDA (por señal) de una plantilla, con los
    parámetros interpolados. No incluye stop/TP/tiempo (van aparte, son del
    setup, no de la plantilla)."""
    ent, sal = [], []
    if plantilla == 'Cruce de medias':
        t, r, l = p['tipo'], p['rapida'], p['lenta']
        if p['direccion'] in ('Long', 'Ambas'):
            ent.append(f"LONG:  SI {t}({r}) cruza arriba {t}({l}) → comprar al open siguiente")
            sal.append(f"LONG:  SI {t}({r}) cruza abajo {t}({l}) → vender al open siguiente")
        if p['direccion'] in ('Short', 'Ambas'):
            ent.append(f"SHORT: SI {t}({r}) cruza abajo {t}({l}) → vender al open siguiente")
            sal.append(f"SHORT: SI {t}({r}) cruza arriba {t}({l}) → recomprar al open siguiente")
    elif plantilla == 'Bollinger + ATR':
        per, d = p['periodo'], p['desv']
        if p['direccion'] in ('Long', 'Ambas'):
            ent.append(f"LONG:  SI close < banda inferior BB({per},{d:g}) → comprar al open siguiente")
            sal.append(f"LONG:  SI close > media BB({per}) → vender al open siguiente")
        if p['direccion'] in ('Short', 'Ambas'):
            ent.append(f"SHORT: SI close > banda superior BB({per},{d:g}) → vender al open siguiente")
            sal.append(f"SHORT: SI close < media BB({per}) → recomprar al open siguiente")
    elif plantilla == 'RSI':
        per = p['periodo']
        if p['direccion'] in ('Long', 'Ambas'):
            ent.append(f"LONG:  SI RSI({per}) < {p['sobreventa']:g} → comprar al open siguiente")
            sal.append(f"LONG:  SI RSI({per}) > 50 → vender al open siguiente")
        if p['direccion'] in ('Short', 'Ambas'):
            ent.append(f"SHORT: SI RSI({per}) > {p['sobrecompra']:g} → vender al open siguiente")
            sal.append(f"SHORT: SI RSI({per}) < 50 → recomprar al open siguiente")
    elif plantilla == 'KAMA':
        per_er, r, l = p['periodo_er'], p['rapido'], p['lento']
        if p['direccion'] in ('Long', 'Ambas'):
            ent.append(f"LONG:  SI close cruza arriba KAMA(ER={per_er},rápido={r},lento={l}) → comprar al open siguiente")
            sal.append(f"LONG:  SI close cruza abajo KAMA(ER={per_er},rápido={r},lento={l}) → vender al open siguiente")
        if p['direccion'] in ('Short', 'Ambas'):
            ent.append(f"SHORT: SI close cruza abajo KAMA(ER={per_er},rápido={r},lento={l}) → vender al open siguiente")
            sal.append(f"SHORT: SI close cruza arriba KAMA(ER={per_er},rápido={r},lento={l}) → recomprar al open siguiente")
    elif plantilla == 'Patrones de velas':
        pats = p['patrones'] or ['(ninguno)']
        for nombre in pats:
            ent.append(f"SI se forma «{nombre}» → entrar en la dirección del "
                       f"sesgo del patrón al open siguiente")
        sal.append(f"Cierre a +{p['lag_salida']} velas de la entrada")
    elif plantilla == 'Custom (reglas)':
        etiquetas = {'entradas_long': ('ent', 'LONG'), 'entradas_short': ('ent', 'SHORT'),
                     'salidas_long': ('sal', 'LONG'), 'salidas_short': ('sal', 'SHORT')}
        reglas = p.get('reglas') or {}
        for clave, (destino, direccion) in etiquetas.items():
            for setup_r in reglas.get(clave, []):
                conds = " Y ".join(
                    f"{_desc_spec(c['izq'])} {c['op']} {_desc_spec(c['der'])}"
                    for c in setup_r.get('condiciones', []))
                if not conds:
                    continue
                linea = f"{direccion}: SI {conds}"
                (ent if destino == 'ent' else sal).append(linea)
        if not ent:
            ent.append("(sin reglas de entrada definidas)")
    return ent, sal


_NOMBRES_DIA_ES = ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom']


def _desc_filtros(filtros):
    """Líneas legibles de los filtros de entrada activos de un setup (vacío
    = sin filtros) — mismo principio de "no caja negra" que
    _codigo_reglas_plantilla."""
    if not filtros:
        return []
    lineas = []
    dias = filtros.get('dias_semana')
    if dias:
        lineas.append(f"Día de la semana: solo {', '.join(_NOMBRES_DIA_ES[d] for d in sorted(dias))}")
    reg = filtros.get('regimen') or {}
    metodo = reg.get('metodo', 'ninguno')
    if metodo != 'ninguno':
        per = reg.get('periodo', 100)
        etiquetas = {
            'er_tendencia': f"ER({per}) por encima del umbral de tendencia",
            'er_rango': f"ER({per}) por debajo del umbral de ruido",
            'hurst_tendencia': f"Hurst({per}) > 0.58 (tendencia)",
            'hurst_reversion': f"Hurst({per}) < 0.52 (reversión a la media)",
        }
        lineas.append(f"Régimen: solo entra si {etiquetas[metodo]}")
    ses = filtros.get('sesion') or {}
    tipo_sesion = ses.get('tipo', 'ninguna')
    if tipo_sesion != 'ninguna':
        if tipo_sesion == 'personalizada':
            lineas.append(f"Sesión: solo entre {ses.get('hora_inicio', 0):02d}:00 y "
                          f"{ses.get('hora_fin', 0):02d}:00 (UTC)")
        else:
            lineas.append(f"Sesión: solo en «{tipo_sesion}»")
    for clave, etiqueta in (('condiciones_entrada', 'Condición extra de entrada'),
                            ('condiciones_salida', 'Condición extra de salida')):
        conds = filtros.get(clave) or []
        if conds:
            texto = " Y ".join(_desc_condicion_dir(c) for c in conds)
            lineas.append(f"{etiqueta}: {texto}")
    return lineas


def codigo_setup(setup, indice=0):
    """Pseudocódigo estructurado del setup: VARIABLES (todos los parámetros
    de la plantilla + configuración de riesgo del setup), ENTRADA y SALIDA.
    Se genera igual para plantillas predefinidas y custom."""
    plantilla = setup['plantilla']
    p = params_por_defecto(plantilla)
    p.update(setup.get('params') or {})
    nombre = setup.get('nombre') or plantilla

    lineas = [f"SETUP S{indice} «{nombre}» — {plantilla}"]

    # VARIABLES: parámetros de la plantilla
    lineas.append("  VARIABLES:")
    vars_plantilla = []
    for spec in ESTRATEGIAS[plantilla]['params']:
        v = p.get(spec['clave'])
        if spec['tipo'] == 'patrones':
            v = ", ".join(v) if v else "(ninguno)"
        elif spec['tipo'] == 'reglas':
            continue   # las reglas custom se listan en ENTRADA/SALIDA
        vars_plantilla.append(f"{spec['clave']} = {v}")
    if vars_plantilla:
        lineas.append("    " + " · ".join(vars_plantilla))
    riesgo = setup.get('riesgo_pct', 0.01) * 100.0
    stop = setup.get('stop_atr', 0.0)
    tp = setup.get('tp_r', 0.0)
    tiempo = setup.get('salida_n_velas', 0)
    lineas.append(
        f"    riesgo = {riesgo:g}% del equity"
        f" · stop = {f'{stop:g}×ATR' if stop else 'ninguno'}"
        f" · take-profit = {f'{tp:g}R' if tp else 'ninguno'}"
        f" · salida por tiempo = {f'+{tiempo} velas' if tiempo else 'sin límite'}")

    filtros_lineas = _desc_filtros(setup.get('filtros'))
    if filtros_lineas:
        lineas.append("  FILTROS (día/régimen/sesión y condiciones de entrada solo "
                      "condicionan nuevas entradas; las condiciones de salida sí "
                      "pueden restringir cuándo se cierra la posición):")
        for fl in filtros_lineas:
            lineas.append(f"    {fl}")

    ent, sal = _codigo_reglas_plantilla(plantilla, p)
    lineas.append("  ENTRADA:")
    for e in ent:
        lineas.append(f"    {e}")
    lineas.append("  SALIDA:")
    for s_l in sal:
        lineas.append(f"    {s_l}")
    if stop:
        lineas.append(f"    ADEMÁS: stop-loss a {stop:g}×ATR de la entrada "
                      f"(contra low/high de cada vela)")
    if tp:
        lineas.append(f"    ADEMÁS: take-profit a {tp:g}R del riesgo")
    if tiempo and plantilla != 'Patrones de velas':
        lineas.append(f"    ADEMÁS: cierre forzoso a +{tiempo} velas de la entrada")
    return "\n".join(lineas)


def codigo_sistema(setups, config_global=None):
    """Código estructurado del sistema completo: variables de cuenta
    globales + un bloque por setup (VARIABLES/ENTRADA/SALIDA)."""
    lineas = []
    if config_global:
        cap = config_global.get('capital_inicial')
        com = config_global.get('comision_pct')
        slip = config_global.get('slippage_pct')
        partes = []
        if cap is not None:
            partes.append(f"capital inicial = {cap:,.0f}")
        if com is not None:
            partes.append(f"comisión = {com * 100:g}% por lado")
        if slip is not None:
            partes.append(f"slippage = {slip * 100:g}%")
        if 'pct_oos' in config_global:
            oos = config_global['pct_oos'] * 100
            partes.append(f"muestra = IS {100 - oos:g}% / OOS {oos:g}%")
        if partes:
            lineas.append("CUENTA: " + " · ".join(partes))
            lineas.append("")
    for k, setup in enumerate(setups):
        lineas.append(codigo_setup(setup, k))
        lineas.append("")
    return "\n".join(lineas).rstrip()


def defaults_setup(plantilla):
    """Valores de setup (stop/tp/...) que la plantilla recomienda por
    defecto — p.ej. el cruce de medias sale por cruce contrario, sin stop."""
    return dict(ESTRATEGIAS[plantilla].get('defaults_setup', {}))


def describir_setup(setup):
    """Resumen de una línea para la lista de setups de la GUI."""
    nombre = setup.get('nombre') or setup['plantilla']
    riesgo = setup.get('riesgo_pct')
    stop = setup.get('stop_atr')
    partes = [nombre, setup['plantilla']]
    if riesgo is not None:
        partes.append(f"riesgo {riesgo * 100:g}%")
    if stop:
        partes.append(f"stop {stop:g}×ATR")
    if setup.get('tp_r'):
        partes.append(f"TP {setup['tp_r']:g}R")
    if setup.get('salida_n_velas'):
        partes.append(f"salida +{setup['salida_n_velas']} velas")
    filtros = setup.get('filtros') or {}
    activos = []
    if filtros.get('dias_semana'):
        activos.append('día')
    if (filtros.get('regimen') or {}).get('metodo', 'ninguno') != 'ninguno':
        activos.append('régimen')
    if (filtros.get('sesion') or {}).get('tipo', 'ninguna') != 'ninguna':
        activos.append('sesión')
    if filtros.get('condiciones_entrada') or filtros.get('condiciones_salida'):
        activos.append('condición')
    if activos:
        partes.append(f"filtros: {'+'.join(activos)}")
    return " · ".join(partes)
