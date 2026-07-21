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

from core.candle_patterns import PATRONES_INFO, detectar_patrones

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
                      'ATR', 'BB_sup', 'BB_inf', 'BB_media']
_OPERADORES_REGLA = ['>', '<', 'cruza arriba', 'cruza abajo']


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
        nuevas = (ent_l | ent_s) & ~reclamada
        out['entradas_long'] |= ent_l & nuevas
        out['entradas_short'] |= ent_s & ~ent_l & nuevas
        out['setup_id'][nuevas] = k
        reclamada |= nuevas
        bit = np.int64(1) << np.int64(k)
        out['salidas_long'][np.asarray(s['salidas_long'], dtype=bool)] |= bit
        out['salidas_short'][np.asarray(s['salidas_short'], dtype=bool)] |= bit

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
    return " · ".join(partes)
