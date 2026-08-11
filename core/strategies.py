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
from core.metrics import (
    calcular_er_series, calcular_kama_numba, calcular_hurst_array,
    calcular_percentil_rodante_numba, calcular_sar_numba,
)
from core.backtest import TRIGGERS_TRAMO_ENTRADA

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


def donchian(h, l, periodo, incluir_actual=False):
    """Canal de Donchian: máximo y mínimo rodantes de `periodo` velas.

    incluir_actual=False (por defecto) desplaza el canal una vela, de modo
    que NO contiene la vela que se compara contra él. Es la diferencia entre
    una ruptura real y una tautología: si la vela actual formara parte del
    canal, su propio máximo sería el máximo del canal y jamás podría
    superarlo."""
    hs, ls = pd.Series(h), pd.Series(l)
    sup = hs.rolling(int(periodo)).max()
    inf = ls.rolling(int(periodo)).min()
    if not incluir_actual:
        sup, inf = sup.shift(1), inf.shift(1)
    return sup.values, inf.values


def stochastic(h, l, c, periodo_k=14, suavizado_k=3, periodo_d=3):
    """Oscilador estocástico (%K / %D). %K = posición del cierre dentro del
    rango [mínimo, máximo] de las últimas `periodo_k` velas, suavizado con
    `suavizado_k` (slow stochastic); %D = media móvil de %K. Ambos en 0-100."""
    hs, ls, cs = pd.Series(h), pd.Series(l), pd.Series(c)
    ll = ls.rolling(int(periodo_k)).min()
    hh = hs.rolling(int(periodo_k)).max()
    raw_k = 100.0 * (cs - ll) / (hh - ll).replace(0, np.nan)
    k = raw_k.rolling(int(suavizado_k)).mean()          # %K (slow)
    d = k.rolling(int(periodo_d)).mean()                # %D
    return k.values, d.values


def williams_r(h, l, c, periodo=14):
    """Williams %R: mismo rango que el estocástico pero invertido en signo,
    escala -100..0 (-100 = mínimo del rango, 0 = máximo)."""
    hs, ls, cs = pd.Series(h), pd.Series(l), pd.Series(c)
    hh = hs.rolling(int(periodo)).max()
    ll = ls.rolling(int(periodo)).min()
    return (-100.0 * (hh - cs) / (hh - ll).replace(0, np.nan)).values


def cci(h, l, c, periodo=20):
    """Commodity Channel Index: desviación del precio típico respecto a su
    media, normalizada por la desviación absoluta media (factor 0.015).
    Típicamente oscila entre -100 y +100 pero no está acotado."""
    tp = (pd.Series(h) + pd.Series(l) + pd.Series(c)) / 3.0
    sma_tp = tp.rolling(int(periodo)).mean()
    md = tp.rolling(int(periodo)).apply(
        lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return ((tp - sma_tp) / (0.015 * md.replace(0, np.nan))).values


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


def _sar_serie(h, l, af_inicial=0.02, af_paso=0.02, af_max=0.2):
    """Parabolic SAR de `h`/`l` → (sar, tendencia ±1). Ver calcular_sar_numba."""
    return calcular_sar_numba(
        np.asarray(h, dtype=np.float64), np.asarray(l, dtype=np.float64),
        float(af_inicial), float(af_paso), float(af_max))


def _pivotes_candidatos(h, l, piernas):
    """Candidatos a pivote alto/bajo al estilo ta.pivothigh/ta.pivotlow de Pine.

    La vela i es candidata a pivote alto si su máximo es el mayor de la
    ventana [i-izq, i+der], con izq = der = piernas // 2 (el contrato del
    indicador ZigZag de TradingView: «piernas» es el total de velas que
    confirman un pivote, la mitad a cada lado).

    Devuelve (candidatos, der) donde cada candidato es
    (idx_pivote, idx_confirmacion, precio, tipo ±1) e idx_confirmacion =
    idx_pivote + der. Esa distinción es lo que hace el indicador NO
    repintante: un pivote no puede conocerse hasta que han cerrado las `der`
    velas siguientes, así que una estrategia solo puede reaccionar a él a
    partir de esa vela, nunca en la del propio pivote.
    """
    der = max(1, int(piernas) // 2)
    izq = der
    n = len(h)
    hs, ls = pd.Series(h), pd.Series(l)
    ventana = izq + der + 1
    # el rolling mira hacia atrás: en la vela j = i+der, la ventana cubre
    # exactamente [i-izq, i+der], que es la ventana centrada en i
    max_ventana = hs.rolling(ventana).max().values
    min_ventana = ls.rolling(ventana).min().values

    candidatos = []
    for i in range(izq, n - der):
        j = i + der
        if h[i] >= max_ventana[j]:
            candidatos.append((i, j, float(h[i]), 1))
        if l[i] <= min_ventana[j]:
            candidatos.append((i, j, float(l[i]), -1))
    # orden de CONFIRMACIÓN, que es el orden en que el motor puede enterarse.
    # Si un alto y un bajo confirman en la misma vela, se procesa primero el
    # que ocurrió antes: desempate arbitrario pero determinista (la librería
    # Pine de referencia no expone el suyo).
    candidatos.sort(key=lambda t: (t[1], t[0]))
    return candidatos, der


def _zigzag_eventos(df, desviacion_pct, piernas):
    """Registro cronológico de cambios del pivote vigente del ZigZag:
    [(idx_pivote, idx_confirmacion, precio, tipo ±1, reemplaza), ...] en orden
    de confirmación, donde `reemplaza` indica que este pivote sustituye al
    anterior en vez de abrir un tramo nuevo.

    Es el registro, y no la lista final de pivotes, la fuente de verdad para
    cualquier uso en backtest: cuando un pivote es sustituido por otro más
    extremo del mismo tipo, el sustituido SÍ estuvo vigente entre su
    confirmación y la del sustituto, y una estrategia que corriera en vivo lo
    habría visto. Reconstruir el pasado solo con la lista final sería
    exactamente el repintado que este indicador evita.

    Reimplementación fiel del algoritmo que implica el contrato de entradas
    del indicador ZigZag de TradingView (desviación % + piernas), no un port
    del código de su librería Pine, que no es público.
    """
    candidatos, _der = _pivotes_candidatos(
        df['high'].values, df['low'].values, piernas)
    umbral = float(desviacion_pct)
    eventos = []
    ult_precio = None
    ult_tipo = 0
    for idx_piv, idx_conf, precio, tipo in candidatos:
        if ult_precio is None:
            eventos.append((idx_piv, idx_conf, precio, tipo, False))
            ult_precio, ult_tipo = precio, tipo
            continue
        if tipo == ult_tipo:
            # dos pivotes del mismo tipo seguidos: el nuevo solo cuenta si
            # supera al anterior, y entonces lo sustituye
            mas_extremo = precio > ult_precio if tipo == 1 else precio < ult_precio
            if mas_extremo:
                eventos.append((idx_piv, idx_conf, precio, tipo, True))
                ult_precio = precio
            continue
        if ult_precio == 0.0:
            continue
        variacion = abs(precio - ult_precio) / abs(ult_precio) * 100.0
        if variacion < umbral:
            continue   # retroceso insuficiente: no abre tramo, sigue vigente el previo
        eventos.append((idx_piv, idx_conf, precio, tipo, False))
        ult_precio, ult_tipo = precio, tipo
    return eventos


def _zigzag_pivotes(df, desviacion_pct, piernas):
    """Polilínea final del ZigZag: [(idx_pivote, idx_confirmacion, precio,
    tipo ±1), ...] tras aplicar las sustituciones. Es lo que se dibuja en el
    gráfico; para lógica de backtest usar _zigzag_eventos/_zigzag_series, que
    conservan lo que se sabía en cada momento."""
    pivotes = []
    for idx_piv, idx_conf, precio, tipo, reemplaza in _zigzag_eventos(
            df, desviacion_pct, piernas):
        if reemplaza and pivotes:
            pivotes[-1] = (idx_piv, idx_conf, precio, tipo)
        else:
            pivotes.append((idx_piv, idx_conf, precio, tipo))
    return pivotes


def _zigzag_series(df, desviacion_pct, piernas):
    """(precio, tipo, eventos): precio y tipo del pivote vigente en cada vela
    según lo que se sabía en esa vela (NaN / 0 antes de la primera
    confirmación). Pensado para condiciones del editor de reglas y como swing
    de referencia de los niveles de Fibonacci."""
    n = len(df)
    precio = np.full(n, np.nan)
    tipo = np.zeros(n, dtype=np.int8)
    eventos = _zigzag_eventos(df, desviacion_pct, piernas)
    for _idx_piv, idx_conf, prec, tp, _reemplaza in eventos:
        if idx_conf < n:
            precio[idx_conf:] = prec
            tipo[idx_conf:] = tp
    return precio, tipo, eventos


def _volatilidad_serie(df, metodo, ventana):
    """Percentil rodante de la volatilidad para el filtro homónimo: ATR si
    `metodo` empieza por 'atr', desviación estándar de retornos log si no.

    El indicador se calcula siempre con PERIODO_ATR_DEFECTO velas y `ventana`
    es solo el histórico contra el que se compara. Son dos cosas distintas a
    propósito: si coincidieran, un tramo de volatilidad estable saldría
    siempre en el percentil medio por muy alta o baja que fuera, porque solo
    se estaría comparando consigo mismo. Al comparar un ATR corto contra una
    ventana larga, el filtro mide volatilidad RELATIVA al pasado reciente."""
    if metodo.startswith('atr'):
        base = atr(df['high'].values, df['low'].values, df['close'].values,
                   PERIODO_ATR_DEFECTO)
    else:
        base = _retorno_log(df['close'].values).rolling(PERIODO_ATR_DEFECTO).std().values
    return calcular_percentil_rodante_numba(
        np.asarray(base, dtype=np.float64), int(ventana))


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


def _gen_stochastic(df, p):
    """Entrada híbrida: cruce de %K sobre %D CONFIRMADO dentro de la zona
    extrema (Long: %K cruza arriba de %D y además %K < sobreventa). Salida
    al recuperar la línea media (50)."""
    h, l, c = df['high'].values, df['low'].values, df['close'].values
    n = len(c)
    k, d = stochastic(h, l, c, p['periodo_k'], p['suavizado_k'], p['periodo_d'])
    s = _base_senales(n, h, l, c)
    cruce_up = _cruza_arriba(k, d)
    cruce_down = _cruza_abajo(k, d)
    if p['direccion'] in ('Long', 'Ambas'):
        s['entradas_long'] = _limpiar_nan(cruce_up & (k < p['sobreventa']), k, d)
        s['salidas_long'] = _limpiar_nan(k > 50.0, k)
    if p['direccion'] in ('Short', 'Ambas'):
        s['entradas_short'] = _limpiar_nan(cruce_down & (k > p['sobrecompra']), k, d)
        s['salidas_short'] = _limpiar_nan(k < 50.0, k)
    return s


def _gen_williams(df, p):
    h, l, c = df['high'].values, df['low'].values, df['close'].values
    n = len(c)
    wr = williams_r(h, l, c, p['periodo'])          # rango -100..0
    s = _base_senales(n, h, l, c)
    if p['direccion'] in ('Long', 'Ambas'):
        s['entradas_long'] = _limpiar_nan(wr < p['sobreventa'], wr)
        s['salidas_long'] = _limpiar_nan(wr > -50.0, wr)
    if p['direccion'] in ('Short', 'Ambas'):
        s['entradas_short'] = _limpiar_nan(wr > p['sobrecompra'], wr)
        s['salidas_short'] = _limpiar_nan(wr < -50.0, wr)
    return s


def _gen_cci(df, p):
    h, l, c = df['high'].values, df['low'].values, df['close'].values
    n = len(c)
    v = cci(h, l, c, p['periodo'])
    s = _base_senales(n, h, l, c)
    if p['direccion'] in ('Long', 'Ambas'):
        s['entradas_long'] = _limpiar_nan(v < p['sobreventa'], v)
        s['salidas_long'] = _limpiar_nan(v > 0.0, v)
    if p['direccion'] in ('Short', 'Ambas'):
        s['entradas_short'] = _limpiar_nan(v > p['sobrecompra'], v)
        s['salidas_short'] = _limpiar_nan(v < 0.0, v)
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


def _gen_breakout(df, p):
    """Ruptura del canal de Donchian: entra cuando el precio supera el máximo
    (o pierde el mínimo) de las `periodo` velas anteriores.

    `periodo_salida` = 0 significa "el mismo canal": la salida es entonces la
    ruptura contraria. Con un valor propio y más corto se obtiene el esquema
    clásico de las Tortugas — entrar con un canal largo y salir con uno más
    reactivo."""
    h, l, c = df['high'].values, df['low'].values, df['close'].values
    n = len(c)
    # la fuente define tanto qué forma el canal como qué lo rompe: con 'close'
    # es una ruptura de CIERRES (ignora las mechas por los dos lados), no un
    # cierre medido contra un canal de máximos
    usa_extremos = p['fuente'] == 'high/low'
    arriba_src = h if usa_extremos else c
    abajo_src = l if usa_extremos else c
    s = _base_senales(n, h, l, c)

    sup, inf = donchian(arriba_src, abajo_src, p['periodo'])
    rompe_arriba = _limpiar_nan(arriba_src > sup, sup)
    rompe_abajo = _limpiar_nan(abajo_src < inf, inf)

    per_salida = int(p.get('periodo_salida') or 0)
    if per_salida > 0 and per_salida != int(p['periodo']):
        sup_s, inf_s = donchian(arriba_src, abajo_src, per_salida)
        salida_long = _limpiar_nan(abajo_src < inf_s, inf_s)
        salida_short = _limpiar_nan(arriba_src > sup_s, sup_s)
    else:
        salida_long, salida_short = rompe_abajo, rompe_arriba

    if p['direccion'] in ('Long', 'Ambas'):
        s['entradas_long'] = rompe_arriba.copy()
        s['salidas_long'] = salida_long.copy()
    if p['direccion'] in ('Short', 'Ambas'):
        s['entradas_short'] = rompe_abajo.copy()
        s['salidas_short'] = salida_short.copy()
    return s


def _gen_sar(df, p):
    """Parabolic SAR: entrada en el GIRO de la tendencia del SAR, y salida en
    el giro contrario — mismo esquema que _gen_cruce_medias/_gen_kama, pero el
    nivel de giro se acelera conforme la tendencia avanza en vez de depender
    de un periodo fijo."""
    h, l, c = df['high'].values, df['low'].values, df['close'].values
    n = len(c)
    sar, tendencia = _sar_serie(h, l, p['af_inicial'], p['af_paso'], p['af_max'])
    s = _base_senales(n, h, l, c)
    previa = np.roll(tendencia, 1)
    giro_alcista = (tendencia == 1) & (previa == -1)
    giro_bajista = (tendencia == -1) & (previa == 1)
    giro_alcista[0] = giro_bajista[0] = False
    giro_alcista = _limpiar_nan(giro_alcista, sar)
    giro_bajista = _limpiar_nan(giro_bajista, sar)
    if p['direccion'] in ('Long', 'Ambas'):
        s['entradas_long'] = giro_alcista.copy()
        s['salidas_long'] = giro_bajista.copy()
    if p['direccion'] in ('Short', 'Ambas'):
        s['entradas_short'] = giro_bajista.copy()
        s['salidas_short'] = giro_alcista.copy()
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
                      'ATR', 'BB_sup', 'BB_inf', 'BB_media', 'KAMA', 'ER', 'SAR',
                      'DONCHIAN_SUP', 'DONCHIAN_INF', 'ZIGZAG']
_OPERADORES_REGLA = ['>', '<', 'cruza arriba', 'cruza abajo']

# defaults fijos de KAMA cuando se usa desde el constructor de reglas: la
# tabla del editor solo expone un "Periodo" por indicador, así que el SC
# rápido/lento de KAMA quedan fijos aquí (el «Periodo» de la fila mapea a
# periodo_er). Para variar rápido/lento hace falta la plantilla KAMA dedicada.
_KAMA_RAPIDO_REGLA = 2
_KAMA_LENTO_REGLA = 30

# misma limitación para el SAR, que necesita tres números y la fila solo
# ofrece uno: los tres quedan fijos en los valores clásicos de Wilder. Para
# tocarlos hace falta la plantilla Parabolic SAR dedicada. El «Periodo» de la
# fila se ignora para este indicador.
_SAR_AF_INICIAL_REGLA = 0.02
_SAR_AF_PASO_REGLA = 0.02
_SAR_AF_MAX_REGLA = 0.2

# ZIGZAG necesita dos números y la fila solo ofrece uno: el «Periodo» de la
# fila se interpreta como las PIERNAS del pivote (el parámetro más ligado a la
# escala temporal) y la desviación mínima queda fija. La plantilla de entrada
# límite en Fibonacci sí expone ambos.
_ZIGZAG_DESVIACION_REGLA = 5.0
_ZIGZAG_PIERNAS_REGLA = 10


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
    if tipo == 'ZIGZAG':
        precio_piv, _tp, _pivs = _zigzag_series(
            df, _ZIGZAG_DESVIACION_REGLA, spec.get('periodo', _ZIGZAG_PIERNAS_REGLA))
        return precio_piv
    if tipo in ('DONCHIAN_SUP', 'DONCHIAN_INF'):
        sup, inf = donchian(df['high'].values, df['low'].values, periodo)
        return {'DONCHIAN_SUP': sup, 'DONCHIAN_INF': inf}[tipo]
    if tipo == 'SAR':
        sar, _tendencia = _sar_serie(
            df['high'].values, df['low'].values,
            _SAR_AF_INICIAL_REGLA, _SAR_AF_PASO_REGLA, _SAR_AF_MAX_REGLA)
        return sar
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
    if tipo == 'SAR':
        return 'SAR'   # sus tres parámetros son fijos aquí, no hay periodo que mostrar
    if tipo == 'ZIGZAG':
        return f"ZIGZAG(piernas={spec.get('periodo', _ZIGZAG_PIERNAS_REGLA)})"
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


def _desc_stochastic(p):
    pk, sk, pd_ = p['periodo_k'], p['suavizado_k'], p['periodo_d']
    est = f"Stoch(%K={pk},suav={sk},%D={pd_})"
    partes = []
    if p['direccion'] in ('Long', 'Ambas'):
        partes.append(f"Entrada Long: %K cruza arriba %D con %K < {p['sobreventa']:g} · "
                      f"Salida Long: %K > 50   [{est}]")
    if p['direccion'] in ('Short', 'Ambas'):
        partes.append(f"Entrada Short: %K cruza abajo %D con %K > {p['sobrecompra']:g} · "
                      f"Salida Short: %K < 50   [{est}]")
    return "\n".join(partes) + "\n(Reversión a la media; stop/TP los pone el setup)"


def _desc_williams(p):
    per = p['periodo']
    partes = []
    if p['direccion'] in ('Long', 'Ambas'):
        partes.append(f"Entrada Long: %R({per}) < {p['sobreventa']:g} · "
                      f"Salida Long: %R({per}) > -50")
    if p['direccion'] in ('Short', 'Ambas'):
        partes.append(f"Entrada Short: %R({per}) > {p['sobrecompra']:g} · "
                      f"Salida Short: %R({per}) < -50")
    return "\n".join(partes) + "\n(Reversión a la media; stop/TP los pone el setup)"


def _desc_cci(p):
    per = p['periodo']
    partes = []
    if p['direccion'] in ('Long', 'Ambas'):
        partes.append(f"Entrada Long: CCI({per}) < {p['sobreventa']:g} · "
                      f"Salida Long: CCI({per}) > 0")
    if p['direccion'] in ('Short', 'Ambas'):
        partes.append(f"Entrada Short: CCI({per}) > {p['sobrecompra']:g} · "
                      f"Salida Short: CCI({per}) < 0")
    return "\n".join(partes) + "\n(Reversión a la media; stop/TP los pone el setup)"


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


def _desc_breakout(p):
    per = p['periodo']
    src_alto = 'high' if p['fuente'] == 'high/low' else 'close'
    src_bajo = 'low' if p['fuente'] == 'high/low' else 'close'
    per_salida = int(p.get('periodo_salida') or 0)
    usa_canal_propio = per_salida > 0 and per_salida != int(per)
    partes = []
    if p['direccion'] in ('Long', 'Ambas'):
        salida = (f"{src_bajo} pierde el mínimo de {per_salida} velas"
                  if usa_canal_propio else f"{src_bajo} pierde el mínimo de {per} velas")
        partes.append(f"Entrada Long: {src_alto} supera el máximo de las {per} velas "
                      f"anteriores · Salida Long: {salida}")
    if p['direccion'] in ('Short', 'Ambas'):
        salida = (f"{src_alto} supera el máximo de {per_salida} velas"
                  if usa_canal_propio else f"{src_alto} supera el máximo de {per} velas")
        partes.append(f"Entrada Short: {src_bajo} pierde el mínimo de las {per} velas "
                      f"anteriores · Salida Short: {salida}")
    partes.append("(el canal excluye la vela en curso: si se incluyera a sí "
                  "misma nunca podría superarlo)")
    return "\n".join(partes)


def _desc_sar(p):
    af_i, af_p, af_m = p['af_inicial'], p['af_paso'], p['af_max']
    firma = f"SAR(AF {af_i:g} +{af_p:g} hasta {af_m:g})"
    partes = []
    if p['direccion'] in ('Long', 'Ambas'):
        partes.append(f"Entrada Long: {firma} gira a alcista (el precio deja el SAR abajo) · "
                      f"Salida Long: el SAR gira a bajista")
    if p['direccion'] in ('Short', 'Ambas'):
        partes.append(f"Entrada Short: {firma} gira a bajista (el precio deja el SAR arriba) · "
                      f"Salida Short: el SAR gira a alcista")
    partes.append("(el nivel de giro acelera cuanto más avanza la tendencia; "
                  "sin stop ATR por defecto, igual criterio que Cruce de medias: "
                  "el propio giro ya es la salida)")
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
        'color': '#4fc3f7',
        'categoria': 'Direccional',
        'desc_corta': 'Cruces de medias móviles SMA/EMA',
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
        'color': '#66bb6a',
        'categoria': 'Mean Reversion',
        'desc_corta': 'Reversión a la media con Bandas Bollinger',
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
        'color': '#ffa726',
        'categoria': 'Mean Reversion',
        'desc_corta': 'Sobrecompra / sobreventa con RSI',
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
    'Stochastic (%K/%D)': {
        'color': '#26c6da',
        'categoria': 'Mean Reversion',
        'desc_corta': 'Cruce %K/%D en zona de sobreventa/sobrecompra',
        'generar': _gen_stochastic,
        'descripcion': _desc_stochastic,
        'params': [
            {'clave': 'periodo_k', 'etiqueta': 'Periodo %K', 'tipo': 'int',
             'defecto': 14, 'min': 2, 'max': 100},
            {'clave': 'suavizado_k', 'etiqueta': 'Suavizado %K', 'tipo': 'int',
             'defecto': 3, 'min': 1, 'max': 20},
            {'clave': 'periodo_d', 'etiqueta': 'Periodo %D', 'tipo': 'int',
             'defecto': 3, 'min': 1, 'max': 20},
            {'clave': 'sobreventa', 'etiqueta': 'Umbral sobreventa', 'tipo': 'float',
             'defecto': 20.0, 'min': 5.0, 'max': 45.0},
            {'clave': 'sobrecompra', 'etiqueta': 'Umbral sobrecompra', 'tipo': 'float',
             'defecto': 80.0, 'min': 55.0, 'max': 95.0},
            {'clave': 'direccion', 'etiqueta': 'Dirección', 'tipo': 'choice',
             'opciones': _OPCIONES_DIRECCION, 'defecto': 'Ambas'},
        ],
    },
    'Williams %R': {
        'color': '#ec407a',
        'categoria': 'Mean Reversion',
        'desc_corta': 'Sobrecompra / sobreventa con Williams %R',
        'generar': _gen_williams,
        'descripcion': _desc_williams,
        'params': [
            {'clave': 'periodo', 'etiqueta': 'Periodo %R', 'tipo': 'int',
             'defecto': 14, 'min': 2, 'max': 100},
            {'clave': 'sobreventa', 'etiqueta': 'Umbral sobreventa', 'tipo': 'float',
             'defecto': -80.0, 'min': -95.0, 'max': -55.0},
            {'clave': 'sobrecompra', 'etiqueta': 'Umbral sobrecompra', 'tipo': 'float',
             'defecto': -20.0, 'min': -45.0, 'max': -5.0},
            {'clave': 'direccion', 'etiqueta': 'Dirección', 'tipo': 'choice',
             'opciones': _OPCIONES_DIRECCION, 'defecto': 'Ambas'},
        ],
    },
    'CCI': {
        'color': '#5c6bc0',
        'categoria': 'Mean Reversion',
        'desc_corta': 'Reversión con Commodity Channel Index',
        'generar': _gen_cci,
        'descripcion': _desc_cci,
        'params': [
            {'clave': 'periodo', 'etiqueta': 'Periodo CCI', 'tipo': 'int',
             'defecto': 20, 'min': 5, 'max': 200},
            {'clave': 'sobreventa', 'etiqueta': 'Umbral sobreventa', 'tipo': 'float',
             'defecto': -100.0, 'min': -300.0, 'max': -50.0},
            {'clave': 'sobrecompra', 'etiqueta': 'Umbral sobrecompra', 'tipo': 'float',
             'defecto': 100.0, 'min': 50.0, 'max': 300.0},
            {'clave': 'direccion', 'etiqueta': 'Dirección', 'tipo': 'choice',
             'opciones': _OPCIONES_DIRECCION, 'defecto': 'Ambas'},
        ],
    },
    'KAMA': {
        'color': '#ab47bc',
        'categoria': 'Direccional',
        'desc_corta': 'Tendencia adaptativa (Kaufman)',
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
    'Breakout de canal (Donchian)': {
        'color': '#26a69a',
        'categoria': 'Direccional',
        'desc_corta': 'Ruptura de máximos / mínimos de N velas',
        'generar': _gen_breakout,
        'descripcion': _desc_breakout,
        'params': [
            {'clave': 'fuente', 'etiqueta': 'Precio que rompe', 'tipo': 'choice',
             'opciones': ['high/low', 'close'], 'defecto': 'high/low'},
            {'clave': 'periodo', 'etiqueta': 'Velas del canal (entrada)', 'tipo': 'int',
             'defecto': 20, 'min': 2, 'max': 500},
            {'clave': 'periodo_salida', 'etiqueta': 'Velas del canal de salida (0 = igual)',
             'tipo': 'int', 'defecto': 0, 'min': 0, 'max': 500},
            {'clave': 'direccion', 'etiqueta': 'Dirección', 'tipo': 'choice',
             'opciones': _OPCIONES_DIRECCION, 'defecto': 'Ambas'},
        ],
    },
    'Parabolic SAR': {
        'color': '#8d6e63',
        'categoria': 'Direccional',
        'desc_corta': 'Tendencia con parada y giro (Wilder)',
        'generar': _gen_sar,
        'descripcion': _desc_sar,
        # el giro del SAR ES la salida, igual razón que Cruce de medias/KAMA
        'defaults_setup': {'stop_atr': 0.0},
        'params': [
            {'clave': 'af_inicial', 'etiqueta': 'AF inicial', 'tipo': 'float',
             'defecto': 0.02, 'min': 0.001, 'max': 0.2},
            {'clave': 'af_paso', 'etiqueta': 'AF incremento', 'tipo': 'float',
             'defecto': 0.02, 'min': 0.001, 'max': 0.2},
            {'clave': 'af_max', 'etiqueta': 'AF máximo', 'tipo': 'float',
             'defecto': 0.2, 'min': 0.02, 'max': 1.0},
            {'clave': 'direccion', 'etiqueta': 'Dirección', 'tipo': 'choice',
             'opciones': _OPCIONES_DIRECCION, 'defecto': 'Ambas'},
        ],
    },
    'Patrones de velas': {
        'color': '#ef5350',
        'categoria': 'Rev. / Direcc.',
        'desc_corta': 'Patrones de velas japonesas',
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
        'color': '#78909c',
        'categoria': None,
        'desc_corta': 'Reglas personalizadas editables',
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

# Umbrales del filtro de régimen. ABSOLUTOS a propósito: "entro solo si el
# mercado es direccional" tiene que significar lo mismo en todos los activos y
# no depender de la distribución del propio histórico (que además metía datos
# futuros en la decisión de las primeras velas).
UMBRAL_ER_TENDENCIA = 0.5
UMBRAL_ER_RUIDO = 0.3
# Hurst: los mismos que contar_regimen_hurst (core/metrics.py) y que la tabla
# del Analizador, que ya eran coherentes entre sí.
UMBRAL_HURST_TENDENCIA = 0.58
UMBRAL_HURST_REVERSION = 0.52

# Ventanas por defecto — ER y Hurst NO comparten escala:
#  · ER 10 es el valor de Kaufman para el AMA (mismo que el periodo_er de la
#    plantilla KAMA). Con 100, ER>0.5 no lo cruza ni el 0,1% de las velas y el
#    filtro de tendencia bloquea todo.
#  · Hurst necesita ventana larga: el estimador R/S con pocos lags tiene sesgo
#    positivo y sobre ruido puro devuelve ~0,58-0,60, justo encima del umbral.
#    Por debajo de VENTANA_HURST_MINIMA el filtro deja de significar lo que dice.
VENTANA_ER_DEFECTO = 10
VENTANA_HURST_DEFECTO = 400
VENTANA_HURST_MINIMA = 200

METODOS_REGIMEN_ER = ('er_tendencia', 'er_rango')
METODOS_REGIMEN_HURST = ('hurst_tendencia', 'hurst_reversion')


def ventana_regimen_defecto(metodo):
    """Ventana de fábrica del filtro de régimen según el método elegido."""
    return (VENTANA_HURST_DEFECTO if metodo in METODOS_REGIMEN_HURST
            else VENTANA_ER_DEFECTO)


def _filtros_por_defecto():
    """Filtros de ENTRADA de un setup — None/'ninguno' en cualquier eje =
    sin restricción en ese eje. Solo se aplican a NUEVAS entradas (ver
    generar_senales_sistema): las salidas nunca se filtran, para no dejar
    una posición abierta sin forma de cerrarse si el régimen/sesión/día
    cambia a mitad de una operación."""
    return {
        'dias_semana': None,      # None = todos; si no, lista de ints 0=Lun..6=Dom
        'regimen': {'metodo': 'ninguno', 'periodo': VENTANA_ER_DEFECTO},
        # metodo: 'ninguno'|'er_tendencia'|'er_rango'|'hurst_tendencia'|'hurst_reversion'
        'volatilidad': {'metodo': 'ninguno', 'periodo': 100, 'percentil': 50.0},
        # metodo: 'ninguno'|'atr_percentil_alto'|'atr_percentil_bajo'|
        #         'stdev_percentil_alto'|'stdev_percentil_bajo'
        'sesion': {'tipo': 'ninguna', 'hora_inicio': 0, 'hora_fin': 0},
        # tipo: 'ninguna'|'overnight'|'londres'|'ny'|'personalizada' (horas UTC en 'personalizada')
        'condiciones_entrada': [],   # lista de {'izq':spec,'op':...,'der':spec}; AND; [] = sin restricción
        'condiciones_salida': [],    # idem, pero se aplica sobre la señal de SALIDA del setup
        'noticias': {
            'activo': False,
            'minutos_antes': 30,
            'minutos_despues': 30,
            'impacto_minimo': 'alto',   # 'bajo'|'medio'|'alto'
            'monedas': None,            # None = inferir del instrumento en tiempo de ejecución
            'cerrar_posiciones': False,  # además de bloquear entradas, forzar cierre si hay posición abierta
        },
    }


# rango (bajo=0, medio=1, alto=2); ver core.data_providers.economic_calendar.IMPACTO_RANK
_IMPACTO_RANK_NOTICIAS = {'bajo': 0, 'medio': 1, 'alto': 2}


def preparar_eventos_noticias(df_eventos):
    """DataFrame crudo del proveedor (timestamp/pais/evento/impacto) ->
    tupla de arrays (ts_epoch_s int64[], pais str[], impacto_rank int8[])
    lista para pasar a generar_senales_sistema/_mascara_filtros_setup. Se
    hace una sola vez por corrida para no repetir overhead de pandas si
    luego se llama en bucle (p.ej. desde el optimizador)."""
    if df_eventos is None or len(df_eventos) == 0:
        return (np.array([], dtype=np.int64), np.array([], dtype=object),
                np.array([], dtype=np.int8))
    ts = df_eventos['timestamp'].values.astype('datetime64[s]').astype(np.int64)
    pais = df_eventos['pais'].astype(str).values
    impacto = df_eventos['impacto'].map(_IMPACTO_RANK_NOTICIAS).fillna(0).astype(np.int8).values
    orden = np.argsort(ts)
    return ts[orden], pais[orden], impacto[orden]


def _mascara_evitar_ventanas(ts_velas, ts_eventos, antes_s, despues_s):
    """True = vela admitida para nuevas entradas (fuera de toda ventana de
    noticia). ts_velas/ts_eventos: int64[] epoch-segundos, ts_eventos ya
    ordenado. Vectorizado con searchsorted (una operación por evento, no un
    bucle n_velas × n_eventos)."""
    admitida = np.ones(len(ts_velas), dtype=bool)
    for t_ev in ts_eventos:
        ini = np.searchsorted(ts_velas, t_ev - antes_s, side='left')
        fin = np.searchsorted(ts_velas, t_ev + despues_s, side='right')
        if fin > ini:
            admitida[ini:fin] = False
    return admitida


def _mascara_filtros_setup(df, filtros, eventos_noticias=None, avisos=None):
    """(m_long, m_short, m_forzar_salida): máscaras AND de los filtros
    activos de un setup (True = vela admitida para NUEVAS entradas).
    Día/régimen/sesión/noticias son agnósticos a la dirección y se aplican a
    ambos lados por igual; condiciones_entrada puede restringir un solo lado
    según su 'direccion' (ver _mascaras_condiciones_dir). Reutiliza los
    umbrales/sesiones de core.candle_patterns.preparar_contexto para no
    duplicar convenciones.

    eventos_noticias: tupla (ts_epoch_s, pais, impacto_rank) ya extraída del
    calendario económico (ver preparar_eventos_noticias), cubriendo el rango
    completo de `df`. El filtrado por impacto/moneda del propio setup ocurre
    aquí, no en el proveedor, para que dos setups del mismo sistema puedan
    pedir umbrales distintos. m_forzar_salida es None si el setup no pide
    cerrar posiciones abiertas antes de la noticia (o si no hay filtro de
    noticias activo).

    avisos: lista donde se anotan los filtros que se han pedido pero NO se han
    podido aplicar. Antes eso pasaba en silencio y el backtest salía como si no
    hubiera filtro — indistinguible de uno sin filtrar."""
    n = len(df)
    m = np.ones(n, dtype=bool)
    m_forzar_salida = None
    if not filtros:
        return m, m.copy(), m_forzar_salida

    dias = filtros.get('dias_semana')
    if dias:
        dow = df['timestamp'].dt.dayofweek.values
        m &= np.isin(dow, list(dias))

    reg = filtros.get('regimen') or {}
    metodo = reg.get('metodo', 'ninguno')
    ses = filtros.get('sesion') or {}
    tipo_sesion = ses.get('tipo', 'ninguna')

    # En TF diario o mayor el filtro de sesión por horas es inaplicable:
    # todas las velas caen a la misma hora del día, así que un rango horario
    # (o una sesión overnight/Londres/NY) deja 0 velas dentro → 0 señales.
    # Se ignora el filtro (con aviso) en vez de anular el backtest.
    _tf_diario = False
    if tipo_sesion != 'ninguna':
        try:
            _ts_int = df['timestamp'].values.astype('datetime64[s]').astype(np.int64)
            _deltas = np.diff(_ts_int)
            _tf_diario = bool(len(_deltas)) and float(np.median(_deltas)) >= 86400.0
        except Exception:
            _tf_diario = False
        if _tf_diario:
            tipo_sesion = 'ninguna'

    necesita_er = metodo in METODOS_REGIMEN_ER
    necesita_hurst = metodo in METODOS_REGIMEN_HURST
    necesita_ctx = necesita_er or necesita_hurst or tipo_sesion in ('overnight', 'londres', 'ny')

    def _avisar(texto):
        if avisos is not None and texto not in avisos:
            avisos.append(texto)

    if _tf_diario:
        _avisar("Filtro de sesión ignorado: el timeframe es diario o mayor, "
                "no se puede filtrar por horas")

    if necesita_ctx:
        close = df['close'].values.astype(np.float64)
        periodo = int(reg.get('periodo') or ventana_regimen_defecto(metodo))
        er_vals = _er_serie(close, periodo).values if necesita_er else None
        hurst_vals = _hurst_serie(close, periodo) if necesita_hurst else None
        ctx = preparar_contexto(
            close, er=er_vals, hurst=hurst_vals,
            timestamps=df['timestamp'].values if tipo_sesion != 'ninguna' else None,
            umbrales_er=(UMBRAL_ER_RUIDO, UMBRAL_ER_TENDENCIA))

        # Cuando el régimen no se ha podido calcular hay que DECIRLO: seguir
        # sin aplicar la máscara deja un backtest idéntico a uno sin filtro,
        # y eso antes no se distinguía de que el filtro sí hubiera actuado.
        if necesita_er:
            if ctx['regimen_er'] is not None:
                m &= (ctx['regimen_er'] == (2 if metodo == 'er_tendencia' else 0))
            else:
                _avisar(f"Filtro de régimen ER NO aplicado: con ventana {periodo} "
                        f"sobre {len(close):,} velas no se puede calcular el ER. "
                        f"El backtest ha corrido SIN ese filtro.")
        if necesita_hurst:
            if ctx['regimen_hurst'] is not None:
                m &= (ctx['regimen_hurst'] == (2 if metodo == 'hurst_tendencia' else 0))
                if periodo < VENTANA_HURST_MINIMA:
                    _avisar(f"Filtro de régimen Hurst con ventana {periodo}: por "
                            f"debajo de {VENTANA_HURST_MINIMA} el estimador se va "
                            f"por encima de {UMBRAL_HURST_TENDENCIA} incluso sobre "
                            f"ruido puro, así que el régimen que marca no es "
                            f"fiable.")
            else:
                _avisar(f"Filtro de régimen Hurst NO aplicado: la ventana "
                        f"{periodo} no cabe en {len(close):,} velas. El backtest "
                        f"ha corrido SIN ese filtro.")
        if tipo_sesion in ('overnight', 'londres', 'ny'):
            m_sesion = _mascara_sesion(ctx, tipo_sesion)
            if m_sesion is not None:
                m &= m_sesion

    vol = filtros.get('volatilidad') or {}
    metodo_vol = vol.get('metodo', 'ninguno')
    if metodo_vol != 'ninguno':
        pct = _volatilidad_serie(df, metodo_vol, vol.get('periodo', 100))
        # el warm-up (NaN) se mapea a -1 para que no pase ni el corte alto ni
        # el bajo: sin ventana completa no hay percentil en el que confiar
        pct = np.nan_to_num(pct, nan=-1.0)
        umbral_vol = float(vol.get('percentil', 50.0))
        if metodo_vol.endswith('alto'):
            m &= (pct >= umbral_vol)
        else:
            m &= (pct >= 0.0) & (pct <= umbral_vol)

    if tipo_sesion == 'personalizada':
        horas = pd.DatetimeIndex(df['timestamp']).hour.values
        h_ini, h_fin = int(ses.get('hora_inicio', 0)), int(ses.get('hora_fin', 0))
        if h_ini <= h_fin:
            m &= (horas >= h_ini) & (horas < h_fin)
        else:
            m &= (horas >= h_ini) | (horas < h_fin)   # cruza medianoche

    noticias = filtros.get('noticias') or {}
    if noticias.get('activo') and eventos_noticias is not None and len(eventos_noticias[0]):
        ts_ev, pais_ev, impacto_ev = eventos_noticias
        umbral = _IMPACTO_RANK_NOTICIAS.get(noticias.get('impacto_minimo', 'alto'), 2)
        sel = impacto_ev >= umbral
        monedas = noticias.get('monedas')
        if monedas:
            sel &= np.isin(pais_ev, list(monedas))
        ts_ev_f = ts_ev[sel]
        if len(ts_ev_f):
            ts_velas = df['timestamp'].values.astype('datetime64[s]').astype(np.int64)
            antes_s = int(noticias.get('minutos_antes', 30)) * 60
            despues_s = int(noticias.get('minutos_despues', 30)) * 60
            # La ventana solo puede bloquear VELAS COMPLETAS (timestamps
            # discretos). Si es menor que la duración del TF puede caer entre
            # velas y no bloquear nada — ni siquiera la vela que CONTIENE el
            # evento. Se expande al TF: bloquear al menos la vela completa que
            # contiene el evento, y se avisa de la ampliación.
            _deltas_ts = np.diff(ts_velas)
            tf_s = int(np.median(_deltas_ts)) if len(_deltas_ts) else 0
            antes_ef = max(antes_s, tf_s)
            despues_ef = max(despues_s, tf_s)
            if tf_s > 0 and (antes_ef > antes_s or despues_ef > despues_s):
                _avisar(
                    f"La ventana de noticias ({antes_s // 60}-{despues_s // 60} "
                    f"min) es menor que el timeframe ({tf_s // 60} min): se "
                    f"bloqueará al menos la vela completa que contiene el evento")
            m_noticias = _mascara_evitar_ventanas(
                ts_velas, ts_ev_f, antes_ef, despues_ef)
            m &= m_noticias
            if noticias.get('cerrar_posiciones'):
                m_forzar_salida = ~m_noticias

    mc_long, mc_short = _mascaras_condiciones_dir(df, filtros.get('condiciones_entrada'))
    return m & mc_long, m & mc_short, m_forzar_salida


# ══════════════ tipo de entrada del setup (mercado / límite Fib) ══════════════

NIVELES_FIB = [0.236, 0.382, 0.5, 0.618, 0.786]


def _entrada_por_defecto():
    """Cómo ejecuta un setup sus señales de entrada. 'mercado' (o la clave
    ausente) es el comportamiento de siempre: entrar al open de la vela
    siguiente. 'limite_fib' coloca en su lugar una orden límite en el
    retroceso de Fibonacci del último tramo del ZigZag."""
    return {
        'tipo': 'mercado',            # 'mercado' | 'limite_fib'
        'nivel_fib': 0.618,
        'zigzag_desviacion': 5.0,
        'zigzag_piernas': 10,
    }


def tramos_zigzag_vigentes(df, desviacion_pct, piernas):
    """Tramo del ZigZag vigente en cada vela, según lo que se sabía en esa
    vela. Devuelve un dict de arrays de longitud n:

    - 'anterior' / 'ultimo': precio del pivote de inicio y de fin del tramo
    - 'idx_anterior' / 'idx_ultimo': vela en que ocurrió cada pivote (-1 si
      aún no hay tramo), para poder anclar el dibujo en el gráfico
    - 'tipo': +1 tramo alcista (bajo → alto), -1 bajista, 0 sin tramo
    - 'nuevo': True en la vela en que se confirma un tramo distinto

    NaN / -1 / 0 mientras no haya dos pivotes de tipo distinto confirmados.
    Es la geometría que usan tanto las órdenes límite de Fibonacci como su
    representación en el gráfico, para que no puedan divergir."""
    n = len(df)
    tramos = {
        'anterior': np.full(n, np.nan),
        'ultimo': np.full(n, np.nan),
        'idx_anterior': np.full(n, -1, dtype=np.int64),
        'idx_ultimo': np.full(n, -1, dtype=np.int64),
        'tipo': np.zeros(n, dtype=np.int8),
        'nuevo': np.zeros(n, dtype=bool),
    }
    eventos = _zigzag_eventos(df, desviacion_pct, piernas)
    for k in range(1, len(eventos)):
        idx_piv, conf, precio, tipo, _reemplaza = eventos[k]
        idx_prev, _conf_prev, precio_prev, tipo_prev, _r = eventos[k - 1]
        if tipo_prev == tipo:
            continue   # sustitución: no hay tramo entre dos pivotes del mismo tipo
        if conf >= n:
            continue
        tramos['anterior'][conf:] = precio_prev
        tramos['ultimo'][conf:] = precio
        tramos['idx_anterior'][conf:] = idx_prev
        tramos['idx_ultimo'][conf:] = idx_piv
        tramos['tipo'][conf:] = tipo
        tramos['nuevo'][conf] = True
    return tramos


def precio_nivel_fib(anterior, ultimo, tipo, nivel):
    """Precio del retroceso `nivel` de un tramo que va de `anterior` a
    `ultimo`. Misma fórmula con la que _ordenes_limite_fib coloca las órdenes:
    se mide desde el extremo alcanzado (`ultimo` = 0 %) hacia el origen del
    tramo (`anterior` = 100 %)."""
    recorrido = abs(ultimo - anterior)
    return ultimo - nivel * recorrido if tipo > 0 else ultimo + nivel * recorrido


def _ordenes_limite_fib(df, entrada, ent_long, ent_short):
    """Convierte las señales booleanas de un setup en órdenes límite al nivel
    de Fibonacci del último tramo confirmado del ZigZag.

    Devuelve (orden_long, orden_short, cancelar_long, cancelar_short): precios
    (NaN = sin orden) y máscaras booleanas de cancelación.

    El tramo de referencia son los dos últimos pivotes VIGENTES en la vela de
    la señal, tomados del registro cronológico del ZigZag — nunca de la
    polilínea final, que ya conoce el futuro. Una señal long solo coloca orden
    si el tramo vigente es alcista (el retroceso es a favor de la tendencia
    del tramo), y viceversa; si no lo es, no hay nivel que comprar y la señal
    se descarta.
    """
    n = len(df)
    orden_long = np.full(n, np.nan)
    orden_short = np.full(n, np.nan)
    cancelar_long = np.zeros(n, dtype=bool)
    cancelar_short = np.zeros(n, dtype=bool)

    nivel = float(entrada.get('nivel_fib', 0.618))
    tramos = tramos_zigzag_vigentes(df, entrada.get('zigzag_desviacion', 5.0),
                                    entrada.get('zigzag_piernas', 10))
    anterior, ultimo, tipo_ultimo = (tramos['anterior'], tramos['ultimo'],
                                     tramos['tipo'])
    # cada tramo nuevo invalida cualquier orden que siguiera viva del anterior
    cancelar_long |= tramos['nuevo']
    cancelar_short |= tramos['nuevo']

    hay_tramo = ~np.isnan(ultimo)
    # tramo alcista (bajo -> alto): el retroceso se compra
    tramo_alcista = hay_tramo & (tipo_ultimo == 1)
    tramo_bajista = hay_tramo & (tipo_ultimo == -1)
    recorrido = np.abs(ultimo - anterior)

    coloca_long = ent_long & tramo_alcista
    orden_long[coloca_long] = (ultimo - nivel * recorrido)[coloca_long]
    coloca_short = ent_short & tramo_bajista
    orden_short[coloca_short] = (ultimo + nivel * recorrido)[coloca_short]

    # la señal contraria del propio setup también cancela
    cancelar_long |= ent_short
    cancelar_short |= ent_long
    return orden_long, orden_short, cancelar_long, cancelar_short


# ══════════════ sistemas multi-setup ══════════════

def generar_senales_sistema(df, setups, eventos_noticias=None):
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
    - Un setup con setup['entrada']['tipo'] == 'limite_fib' no aporta
      entradas a mercado: sus señales se traducen a órdenes límite en
      orden_long_precio/orden_short_precio (ver _ordenes_limite_fib). Compite
      por la vela con la misma prioridad que los demás.

    eventos_noticias: tupla (ts_epoch_s, pais, impacto_rank) de
    preparar_eventos_noticias(), o None si ningún setup usa el filtro de
    noticias. Se reenvía tal cual a _mascara_filtros_setup por cada setup;
    cada uno aplica su propio umbral de impacto/monedas/ventana.
    """
    if len(setups) > MAX_SETUPS:
        raise ValueError(f"Máximo {MAX_SETUPS} setups por sistema")
    n = len(df)
    out = {'entradas_long': np.zeros(n, dtype=bool),
           'entradas_short': np.zeros(n, dtype=bool),
           'salidas_long': np.zeros(n, dtype=np.int64),
           'salidas_short': np.zeros(n, dtype=np.int64),
           'setup_id': np.zeros(n, dtype=np.int64),
           # órdenes límite: NaN = ninguna. Un sistema sin setups de entrada
           # límite las deja intactas y el motor se comporta como siempre.
           'orden_long_precio': np.full(n, np.nan),
           'orden_short_precio': np.full(n, np.nan),
           'cancelar_orden_long': np.zeros(n, dtype=np.int64),
           'cancelar_orden_short': np.zeros(n, dtype=np.int64),
           # filtros pedidos que no se han podido aplicar, o aplicados con
           # parámetros en los que no son fiables: la GUI los enseña para que
           # un backtest sin filtrar no pase por uno filtrado
           'avisos': [],
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
            avisos_setup = []
            m_ent_long, m_ent_short, m_forzar_salida = _mascara_filtros_setup(
                df, filtros, eventos_noticias, avisos=avisos_setup)
            nombre = setup.get('nombre') or f'Setup {k + 1}'
            out['avisos'] += [f'{nombre}: {a}' for a in avisos_setup]
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
            # noticias.cerrar_posiciones SÍ fuerza la salida (a diferencia
            # del resto de filtros de entrada): si hay una posición abierta
            # de este setup, se cierra al entrar en la ventana previa al
            # evento, igual que si la propia estrategia hubiera señalado
            # salida — el motor no distingue el origen del bit.
            if m_forzar_salida is not None:
                sal_l = sal_l | m_forzar_salida
                sal_s = sal_s | m_forzar_salida
        bit = np.int64(1) << np.int64(k)
        entrada = setup.get('entrada') or {}
        if entrada.get('tipo') == 'limite_fib':
            # las señales de este setup no entran a mercado: dejan una orden
            # límite en el retroceso de Fibonacci del tramo vigente
            orden_l, orden_s, cancel_l, cancel_s = _ordenes_limite_fib(
                df, entrada, ent_l, ent_s)
            nuevas = (~np.isnan(orden_l) | ~np.isnan(orden_s)) & ~reclamada
            pone_l = nuevas & ~np.isnan(orden_l)
            pone_s = nuevas & ~pone_l & ~np.isnan(orden_s)
            out['orden_long_precio'][pone_l] = orden_l[pone_l]
            out['orden_short_precio'][pone_s] = orden_s[pone_s]
            out['cancelar_orden_long'][cancel_l] |= bit
            out['cancelar_orden_short'][cancel_s] |= bit
        else:
            nuevas = (ent_l | ent_s) & ~reclamada
            out['entradas_long'] |= ent_l & nuevas
            out['entradas_short'] |= ent_s & ~ent_l & nuevas
        out['setup_id'][nuevas] = k
        reclamada |= nuevas
        out['salidas_long'][sal_l] |= bit
        out['salidas_short'][sal_s] |= bit

    if out['atr'] is None:
        out['atr'] = np.zeros(n)
    return out


def etapa_salida_por_defecto():
    """Etapa de salida implícita de cualquier plantilla: cerrar el 100% de la
    posición con la propia señal de salida de la plantilla (el cruce contrario
    en Cruce de medias/KAMA, la vuelta a la media en Bollinger/RSI/...).

    Materializarla como etapa explícita es lo que permite editarla desde la
    tabla «Salida del setup» (p. ej. bajarla al 50% y añadir una segunda etapa
    a 2R). Con pct=100 el motor la trata exactamente igual que no tener
    etapas — ver core/backtest.py."""
    return {'pct': 100.0, 'r': 0.0, 'trigger': 'senal', 'condiciones': [],
            'gestion': {'tipo': 0, 'val': 0.0}}


def _fila(izq, op, der, direccion, mapeo, nota=''):
    """Descriptor de una fila de condición derivada de una plantilla, en el
    mismo formato que consume la tabla de condiciones de la GUI.

    'mapeo' dice qué celda de la fila escribe en qué parámetro de la plantilla
    ({'izq.periodo': 'rapida'}), para que editar la fila cambie la plantilla en
    vez de añadir un filtro suelto. Una celda sin entrada en el mapeo va
    bloqueada. 'nota' avisa de lo que la fila no puede representar con
    fidelidad (ver KAMA)."""
    texto = f"{_desc_spec(izq)} {op} {_desc_spec(der)}"
    return {'izq': izq, 'op': op, 'der': der, 'direccion': direccion,
            'mapeo': mapeo, 'texto': texto, 'nota': nota}


def _fila_texto(texto, direccion='ambas'):
    """Fila que la tabla no sabe representar como indicador (Stochastic,
    Williams %R, CCI, patrones): solo texto, sin celdas editables."""
    return {'izq': None, 'op': None, 'der': None, 'direccion': direccion,
            'mapeo': {}, 'texto': texto, 'nota': ''}


def _lados(p):
    """(hace_long, hace_short) según el parámetro 'direccion' de la plantilla."""
    d = p.get('direccion', 'Ambas')
    return d in ('Long', 'Ambas'), d in ('Short', 'Ambas')


# KAMA en la tabla de condiciones solo expone un «Periodo» (el del ER): sus
# constantes de suavizado son fijas (_KAMA_RAPIDO_REGLA/_KAMA_LENTO_REGLA), así
# que la fila solo es fiel si la plantilla usa esos mismos valores.
_NOTA_KAMA = ("la fila usa el KAMA del editor de reglas (rápido="
              f"{_KAMA_RAPIDO_REGLA}, lento={_KAMA_LENTO_REGLA}); si cambias "
              "«SC rápido»/«SC lento» la plantilla seguirá usando los tuyos, "
              "pero esta fila no puede reflejarlo")

# el DONCHIAN del editor de reglas siempre se forma con máximos/mínimos; con
# fuente='close' la plantilla usa un canal de cierres, que la fila no sabe
# representar.
_NOTA_DONCHIAN = ("con «Precio que rompe» = close la plantilla forma el canal "
                  "con cierres, pero el DONCHIAN de esta fila siempre usa "
                  "máximos/mínimos")


def filas_plantilla(plantilla, params=None, salida=False):
    """Descriptores de fila de la señal de ENTRADA (o de SALIDA si
    salida=True) de una plantilla, uno por lado activo.

    Es la única fuente de verdad de "qué hace esta plantilla": la usan tanto
    la tabla de Entrada de la GUI como el pseudocódigo de _codigo_reglas_plantilla.
    Devuelve filas con 'mapeo' no vacío cuando la tabla puede representarlas,
    y filas de solo texto cuando no (Stochastic/Williams %R/CCI/patrones)."""
    p = params_por_defecto(plantilla)
    p.update(params or {})
    hace_long, hace_short = _lados(p)
    filas = []

    if plantilla == 'Cruce de medias':
        t, r, l = p['tipo'], p['rapida'], p['lenta']
        mapeo = {'izq.tipo': 'tipo', 'izq.periodo': 'rapida',
                 'der.tipo': 'tipo', 'der.periodo': 'lenta'}
        arriba, abajo = 'cruza arriba', 'cruza abajo'
        if hace_long:
            filas.append(_fila({'tipo': t, 'periodo': r},
                               abajo if salida else arriba,
                               {'tipo': t, 'periodo': l}, 'long', mapeo))
        if hace_short:
            filas.append(_fila({'tipo': t, 'periodo': r},
                               arriba if salida else abajo,
                               {'tipo': t, 'periodo': l}, 'short', mapeo))

    elif plantilla == 'Bollinger + ATR':
        per, d = p['periodo'], p['desv']
        mapeo = {'der.periodo': 'periodo'}
        media = {'tipo': 'BB_media', 'periodo': per, 'desv': d}
        if hace_long:
            filas.append(_fila({'tipo': 'close'}, '>' if salida else '<',
                               media if salida else
                               {'tipo': 'BB_inf', 'periodo': per, 'desv': d},
                               'long', mapeo))
        if hace_short:
            filas.append(_fila({'tipo': 'close'}, '<' if salida else '>',
                               media if salida else
                               {'tipo': 'BB_sup', 'periodo': per, 'desv': d},
                               'short', mapeo))

    elif plantilla == 'RSI':
        per = p['periodo']
        izq = {'tipo': 'RSI', 'periodo': per}
        if hace_long:
            filas.append(_fila(
                dict(izq), '>' if salida else '<',
                {'tipo': 'valor', 'valor': 50.0 if salida else p['sobreventa']},
                'long',
                {'izq.periodo': 'periodo'} if salida else
                {'izq.periodo': 'periodo', 'der.valor': 'sobreventa'}))
        if hace_short:
            filas.append(_fila(
                dict(izq), '<' if salida else '>',
                {'tipo': 'valor', 'valor': 50.0 if salida else p['sobrecompra']},
                'short',
                {'izq.periodo': 'periodo'} if salida else
                {'izq.periodo': 'periodo', 'der.valor': 'sobrecompra'}))

    elif plantilla == 'KAMA':
        per_er, rap, len_ = p['periodo_er'], p['rapido'], p['lento']
        mapeo = {'der.periodo': 'periodo_er'}
        nota = _NOTA_KAMA if (rap, len_) != (_KAMA_RAPIDO_REGLA, _KAMA_LENTO_REGLA) else ''
        kama = {'tipo': 'KAMA', 'periodo': per_er}
        if hace_long:
            filas.append(_fila({'tipo': 'close'},
                               'cruza abajo' if salida else 'cruza arriba',
                               dict(kama), 'long', mapeo, nota))
        if hace_short:
            filas.append(_fila({'tipo': 'close'},
                               'cruza arriba' if salida else 'cruza abajo',
                               dict(kama), 'short', mapeo, nota))

    elif plantilla == 'Stochastic (%K/%D)':
        est = f"Stoch(%K={p['periodo_k']},suav={p['suavizado_k']},%D={p['periodo_d']})"
        if hace_long:
            filas.append(_fila_texto(
                f"%K > 50  [{est}]" if salida else
                f"%K cruza arriba %D Y %K < {p['sobreventa']:g}  [{est}]", 'long'))
        if hace_short:
            filas.append(_fila_texto(
                f"%K < 50  [{est}]" if salida else
                f"%K cruza abajo %D Y %K > {p['sobrecompra']:g}  [{est}]", 'short'))

    elif plantilla == 'Williams %R':
        per = p['periodo']
        if hace_long:
            filas.append(_fila_texto(
                f"%R({per}) > -50" if salida else
                f"%R({per}) < {p['sobreventa']:g}", 'long'))
        if hace_short:
            filas.append(_fila_texto(
                f"%R({per}) < -50" if salida else
                f"%R({per}) > {p['sobrecompra']:g}", 'short'))

    elif plantilla == 'CCI':
        per = p['periodo']
        if hace_long:
            filas.append(_fila_texto(
                f"CCI({per}) > 0" if salida else
                f"CCI({per}) < {p['sobreventa']:g}", 'long'))
        if hace_short:
            filas.append(_fila_texto(
                f"CCI({per}) < 0" if salida else
                f"CCI({per}) > {p['sobrecompra']:g}", 'short'))

    elif plantilla == 'Breakout de canal (Donchian)':
        per = p['periodo']
        per_salida = int(p.get('periodo_salida') or 0)
        usa_canal_propio = per_salida > 0 and per_salida != int(per)
        per_sal = per_salida if usa_canal_propio else per
        alto = 'high' if p['fuente'] == 'high/low' else 'close'
        bajo = 'low' if p['fuente'] == 'high/low' else 'close'
        # la celda izquierda queda bloqueada: sus valores ('high'/'low') no
        # coinciden con los del parámetro 'fuente' ('high/low'/'close'), así
        # que no se puede escribir de vuelta desde la fila
        mapeo_ent = {'der.periodo': 'periodo'}
        mapeo_sal = {'der.periodo': 'periodo_salida' if usa_canal_propio else 'periodo'}
        nota = _NOTA_DONCHIAN if p['fuente'] == 'close' else ''
        if hace_long:
            filas.append(_fila(
                {'tipo': bajo if salida else alto}, '<' if salida else '>',
                {'tipo': 'DONCHIAN_INF' if salida else 'DONCHIAN_SUP',
                 'periodo': per_sal if salida else per},
                'long', mapeo_sal if salida else mapeo_ent, nota))
        if hace_short:
            filas.append(_fila(
                {'tipo': alto if salida else bajo}, '>' if salida else '<',
                {'tipo': 'DONCHIAN_SUP' if salida else 'DONCHIAN_INF',
                 'periodo': per_sal if salida else per},
                'short', mapeo_sal if salida else mapeo_ent, nota))

    elif plantilla == 'Parabolic SAR':
        # el giro del SAR no es una comparación entre dos series (depende del
        # estado acumulado del AF y del extremo de la tendencia viva), así que
        # la tabla no puede ofrecer celdas editables: solo texto
        firma = f"SAR(AF {p['af_inicial']:g} +{p['af_paso']:g} hasta {p['af_max']:g})"
        if hace_long:
            filas.append(_fila_texto(
                f"{firma} gira a bajista" if salida else
                f"{firma} gira a alcista", 'long'))
        if hace_short:
            filas.append(_fila_texto(
                f"{firma} gira a alcista" if salida else
                f"{firma} gira a bajista", 'short'))

    elif plantilla == 'Patrones de velas':
        if salida:
            filas.append(_fila_texto(f"a +{p['lag_salida']} velas de la entrada"))
        else:
            for nombre in (p['patrones'] or ['(ninguno)']):
                filas.append(_fila_texto(
                    f"se forma «{nombre}» (dirección = sesgo del patrón)"))

    elif plantilla == 'Custom (reglas)':
        etiquetas = {'salidas_long': 'long', 'salidas_short': 'short'} if salida \
            else {'entradas_long': 'long', 'entradas_short': 'short'}
        for clave, direccion in etiquetas.items():
            for setup_r in (p.get('reglas') or {}).get(clave, []):
                conds = " Y ".join(
                    f"{_desc_spec(cond['izq'])} {cond['op']} {_desc_spec(cond['der'])}"
                    for cond in setup_r.get('condiciones', []))
                if conds:
                    filas.append(_fila_texto(conds, direccion))

    return filas


# nombres históricos, más explícitos en el punto de uso
def condiciones_senal_plantilla(plantilla, params=None):
    return filas_plantilla(plantilla, params, salida=False)


def condiciones_salida_plantilla(plantilla, params=None):
    return filas_plantilla(plantilla, params, salida=True)


_ACCION_ENTRADA = {'long': ('LONG: ', 'comprar'), 'short': ('SHORT:', 'vender')}
_ACCION_SALIDA = {'long': ('LONG: ', 'vender'), 'short': ('SHORT:', 'recomprar')}


def _codigo_reglas_plantilla(plantilla, p):
    """Líneas de ENTRADA y SALIDA (por señal) de una plantilla, con los
    parámetros interpolados. No incluye stop/TP/tiempo (van aparte, son del
    setup, no de la plantilla). Se construye sobre filas_plantilla para que el
    pseudocódigo y las tablas de la GUI no puedan divergir."""
    if plantilla == 'Patrones de velas':
        ent = [f"SI {f['texto'].split(' (dirección')[0]} → entrar en la dirección "
               f"del sesgo del patrón al open siguiente"
               for f in filas_plantilla(plantilla, p)]
        sal = [f"Cierre {f['texto']}" for f in filas_plantilla(plantilla, p, salida=True)]
        return ent, sal

    def _lineas(filas, acciones):
        out = []
        for f in filas:
            etiqueta, verbo = acciones.get(f['direccion'], ('', 'operar'))
            out.append(f"{etiqueta} SI {f['texto']} → {verbo} al open siguiente")
        return out

    ent = _lineas(filas_plantilla(plantilla, p), _ACCION_ENTRADA)
    sal = _lineas(filas_plantilla(plantilla, p, salida=True), _ACCION_SALIDA)
    if plantilla == 'Custom (reglas)' and not ent:
        ent.append("(sin reglas de entrada definidas)")
    return ent, sal


def etapa_estancamiento_por_defecto():
    """Etapa de salida por estancamiento: cierra si tras 'velas_max' velas
    desde la entrada el avance a favor no llega a 'r_min' — ver
    core/backtest.py (disparador 'estancamiento'). Sin retrocompatibilidad
    que derivar: es un disparador nuevo, siempre se escribe explícito."""
    return {'pct': 100.0, 'trigger': 'estancamiento', 'velas_max': 10,
            'r_min': 1.0, 'condiciones': [], 'gestion': {'tipo': 0, 'val': 0.0}}


def trigger_etapa(etapa):
    """Disparador de una etapa de salida parcial: 'senal' (la propia señal de
    salida de la plantilla), 'r' (al alcanzar +N R), 'cond' (cuando se
    cumplen sus condiciones) o 'estancamiento' (N velas sin alcanzar un R
    mínimo). Deriva el valor para etapas del formato antiguo sin 'trigger',
    con el mismo criterio que core/backtest.simular."""
    trigger = etapa.get('trigger')
    if trigger in ('senal', 'r', 'cond', 'estancamiento'):
        return trigger
    if float(etapa.get('r', 0.0)) > 0.0:
        return 'r'
    return 'cond' if etapa.get('condiciones') else 'senal'


def _desc_disparador_etapa(etapa):
    trigger = trigger_etapa(etapa)
    if trigger == 'r':
        return f"cuando el precio ≥ +{float(etapa.get('r', 0.0)):g} R"
    if trigger == 'cond':
        conds = " Y ".join(_desc_condicion_dir(c) for c in etapa.get('condiciones', []))
        return f"cuando se cumpla: {conds}" if conds else "cuando se cumplan sus condiciones"
    if trigger == 'estancamiento':
        velas = int(etapa.get('velas_max', 0))
        r_min = float(etapa.get('r_min', 0.0))
        return (f"por estancamiento: si a +{velas} velas de la entrada no ha "
                f"alcanzado {r_min:g}R de avance")
    return "a la señal de salida de la plantilla"


def tramo_entrada_por_defecto():
    """Tramo de entrada implícito de cualquier plantilla: construir el 100%
    de la posición con la propia señal de entrada de la plantilla (el cruce,
    la ruptura de banda, el nivel de RSI...). Es el equivalente en la
    ENTRADA de etapa_salida_por_defecto(): un solo tramo al 100% a la señal
    es exactamente lo que hace el sistema sin entrada escalonada — ver
    core/backtest.simular, que trata la ausencia de 'tramos' igual que un
    único tramo con estos mismos valores."""
    return {'pct': 100.0, 'trigger': 'senal', 'val': 0.0, 'condiciones': [],
            'gestion': {'tipo': 0, 'val': 0.0}}


def trigger_tramo(tramo):
    """Disparador de un tramo de entrada escalonada: 'senal' (se repite la
    señal de entrada de la plantilla) · 'velas' (a +N velas de la 1ª
    entrada) · 'retroceso' (N×ATR en contra, promediar) · 'avance' (+N R a
    favor, pirámide) · 'cond' (solo sus condiciones) — ver
    core/backtest.TRIGGERS_TRAMO_ENTRADA. A diferencia de trigger_etapa, no
    hay formato antiguo que migrar: 'tramos' es un campo nuevo y el
    constructor siempre escribe 'trigger' explícito; 'senal' es solo el
    valor por defecto si faltara."""
    trigger = tramo.get('trigger')
    return trigger if trigger in TRIGGERS_TRAMO_ENTRADA else 'senal'


def _desc_disparador_tramo(tramo):
    trigger = trigger_tramo(tramo)
    val = float(tramo.get('val', 0.0))
    if trigger == 'velas':
        return f"a +{val:g} velas de la 1ª entrada"
    if trigger == 'retroceso':
        return f"si retrocede {val:g}×ATR en contra (promediar)"
    if trigger == 'avance':
        return f"si avanza +{val:g} R a favor (pirámide)"
    if trigger == 'cond':
        conds = " Y ".join(_desc_condicion_dir(c) for c in tramo.get('condiciones', []))
        return f"cuando se cumpla: {conds}" if conds else "cuando se cumplan sus condiciones"
    return "con la señal de entrada de la plantilla"


def salida_mecanismo_por_defecto():
    """Configuración por defecto de uno de los cuatro mecanismos globales de
    salida (stop-loss, take-profit, break-even, trailing): cerrar el 100% de
    lo que quede abierto, sin condiciones ni gestión extra. Es exactamente lo
    que hacían siempre, así que un setup sin estos campos se comporta igual —
    ver core/backtest.simular, que asume estos mismos valores si faltan."""
    return {'pct': 100.0, 'condiciones': [], 'gestion': {'tipo': 0, 'val': 0.0}}


# etiqueta legible de cada mecanismo, en el mismo orden que
# core.backtest.MECANISMOS_SALIDA (el orden fija el código de 'parcial')
# con artículo incluido: se interpolan en frases del pseudocódigo
NOMBRES_MECANISMO_SALIDA = {
    'salida_stop': 'el stop-loss original',
    'salida_tp': 'el take-profit',
    'salida_be': 'el stop movido a break-even',
    'salida_trailing': 'el trailing stop',
    'salida_tiempo': 'la salida por tiempo',
}


def _desc_mecanismos_salida(setup):
    """Líneas para el pseudocódigo describiendo los mecanismos globales que
    NO cierran el 100% de la posición (los que sí lo hacen ya se describen en
    las líneas de stop/TP/BE/trailing de codigo_setup)."""
    gestion_nombres = {1: 'break-even', 2: 'trailing',
                       3: 'stop al precio del cierre anterior'}
    lineas = []
    for clave, etiqueta in NOMBRES_MECANISMO_SALIDA.items():
        mec = setup.get(clave)
        if not mec:
            continue
        pct = float(mec.get('pct', 100.0))
        if pct >= 100.0:
            continue
        # la salida por tiempo queda AGOTADA tras su parcial; el resto son
        # redes de seguridad y siguen cortando (ver core/backtest)
        if clave == 'salida_tiempo':
            cola = ("una sola vez: el resto queda a cargo del stop/TP/señal, "
                    "el tiempo ya no vuelve a cerrarlo")
        else:
            cola = "una sola vez por operación (si vuelve a dispararse, cierra el resto)"
        texto = (f"ADEMÁS: al dispararse {etiqueta} cierra solo el {pct:g}% de "
                 f"lo que quede abierto, {cola}")
        conds = " Y ".join(_desc_condicion_dir(c) for c in mec.get('condiciones', []))
        if conds:
            texto += f", y solo si {conds}"
        g_tipo = (mec.get('gestion') or {}).get('tipo', 0)
        if g_tipo in gestion_nombres:
            texto += f" → sobre el resto activa {gestion_nombres[g_tipo]}"
        lineas.append(texto)
    return lineas


AVISO_EXCESO_PARCIALES = "Se excede el 100% del tamaño de la posición."


def validar_parciales(parciales):
    """Avisos de una lista de etapas de salida parcial (vacío = válida).

    Cada «Cerrar %» es sobre el tamaño ORIGINAL de la posición y las etapas
    son secuenciales, así que la suma acumulada no puede pasar del 100%: el
    motor recorta al máximo disponible, pero una configuración así siempre
    indica un error de captura (la etapa sobrante nunca llegaría a ejecutarse).
    Función pura para poder testearla y para que la GUI decida si deja correr
    el backtest — ver OptimizadorWidget._validar_sistema."""
    invalidas = []
    queda = 100.0
    for i, etapa in enumerate(parciales or []):
        try:
            pct = float((etapa or {}).get('pct', 0.0))
        except (TypeError, ValueError):
            pct = 0.0
        if pct > queda + 1e-9:
            invalidas.append(i + 1)
        queda -= min(max(pct, 0.0), queda)
    if not invalidas:
        return []
    palabra = "Etapa" if len(invalidas) == 1 else "Etapas"
    nums = ", ".join(str(x) for x in invalidas)
    return [f"{palabra} {nums}: {AVISO_EXCESO_PARCIALES}"]


def validar_tramos(tramos):
    """Avisos de una lista de tramos de entrada escalonada (vacío = válida).

    Cada «Entrar %» es una porcion del RIESGO total del setup, asi que la suma
    no debería pasar del 100%: por encima, el sistema arriesgaría más de lo
    pretendido (el motor no lo recorta, ver core/backtest)."""
    total = 0.0
    for tramo in (tramos or []):
        try:
            total += float((tramo or {}).get('pct', 0.0))
        except (TypeError, ValueError):
            pass
    if total > 100.0 + 1e-9:
        return [f"Los promedios suman {total:g}% del riesgo: el sistema "
                "arriesgaría más de lo pretendido."]
    return []


def validar_setup(setup):
    """Todos los avisos de configuración de un setup (vacío = válido)."""
    return (validar_parciales(setup.get('parciales'))
            + validar_tramos(setup.get('tramos')))


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
        per = reg.get('periodo') or ventana_regimen_defecto(metodo)
        etiquetas = {
            'er_tendencia': f"ER({per}) > {UMBRAL_ER_TENDENCIA} (tendencia)",
            'er_rango': f"ER({per}) < {UMBRAL_ER_RUIDO} (ruido/rango)",
            'hurst_tendencia': f"Hurst({per}) > {UMBRAL_HURST_TENDENCIA} (tendencia)",
            'hurst_reversion': f"Hurst({per}) < {UMBRAL_HURST_REVERSION} (reversión a la media)",
        }
        lineas.append(f"Régimen: solo entra si {etiquetas[metodo]}")
    vol = filtros.get('volatilidad') or {}
    metodo_vol = vol.get('metodo', 'ninguno')
    if metodo_vol != 'ninguno':
        per_vol = vol.get('periodo', 100)
        pct_vol = vol.get('percentil', 50.0)
        indicador = 'ATR' if metodo_vol.startswith('atr') else 'Desv. estándar'
        comp = '≥' if metodo_vol.endswith('alto') else '≤'
        lineas.append(
            f"Volatilidad: solo entra si {indicador}({PERIODO_ATR_DEFECTO}) está en el "
            f"percentil {comp} {pct_vol:g} de las últimas {per_vol} velas")
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
    noticias = filtros.get('noticias') or {}
    if noticias.get('activo'):
        monedas = noticias.get('monedas')
        alcance = "no abre ni deja posiciones abiertas" if noticias.get('cerrar_posiciones') else "no abre"
        lineas.append(
            f"Noticias: {alcance} de {noticias.get('minutos_antes', 30)} min antes a "
            f"{noticias.get('minutos_despues', 30)} min después de un evento de impacto "
            f"{noticias.get('impacto_minimo', 'alto')}"
            + (f" ({'/'.join(monedas)})" if monedas else ""))
    return lineas


def _desc_entrada_limite(setup):
    """Líneas que matizan la ENTRADA cuando el setup no entra a mercado. Vacío
    con entrada a mercado, que es lo que ya describe el pseudocódigo base
    («→ comprar al open siguiente»)."""
    entrada = setup.get('entrada') or {}
    if entrada.get('tipo') != 'limite_fib':
        return []
    nivel = float(entrada.get('nivel_fib', 0.618))
    desv = float(entrada.get('zigzag_desviacion', 5.0))
    piernas = int(entrada.get('zigzag_piernas', 10))
    lineas = [
        f"OJO: no entra al open. La señal COLOCA una orden límite en el "
        f"retroceso {nivel:g} del último tramo del ZigZag({desv:g}%, {piernas} "
        f"piernas), y solo se opera si el precio vuelve a ese nivel.",
        "La orden se cancela si el ZigZag confirma un tramo nuevo o si el "
        "propio setup señala en dirección contraria.",
    ]
    vigencia = int(setup.get('limite_vigencia_velas', 0))
    if vigencia:
        lineas.append(f"También caduca a las {vigencia} velas sin rellenarse.")
    avance = float(setup.get('limite_cancelar_avance_r', 0.0))
    if avance:
        lineas.append(f"También se cancela si el precio avanza {avance:g}R a favor "
                      f"sin haberla tocado (el movimiento se fue sin nosotros).")
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
    be = setup.get('be_atr', 0.0)
    be_uni = 'R' if setup.get('be_unidad') == 'r' else '×ATR'
    trail = setup.get('trailing_atr', 0.0)
    lineas.append(
        f"    riesgo = {riesgo:g}% del equity"
        f" · stop = {f'{stop:g}×ATR' if stop else 'ninguno'}"
        f" · take-profit = {f'{tp:g}R' if tp else 'ninguno'}"
        f" · salida por tiempo = {f'+{tiempo} velas' if tiempo else 'sin límite'}"
        f" · {'BE a ' + f'{be:g}{be_uni}' if be else 'sin BE'}"
        f" · {'trailing a ' + f'{trail:g}×ATR' if trail else 'sin trailing'}")
    gestion_nombres = {0: 'sin cambios', 1: 'BE', 2: 'trailing', 3: '→ precio de referencia anterior'}

    tramos = setup.get('tramos') or []
    if len(tramos) > 1:
        lineas.append("")
        lineas.append("  ENTRADA ESCALONADA:")
        for e, tr in enumerate(tramos, 1):
            pct = tr.get('pct', 100)
            g = tr.get('gestion', {})
            g_label = gestion_nombres.get(g.get('tipo', 0), '—')
            if g.get('tipo') in (1, 2) and g.get('val', 0) > 0:
                g_label += f' {g["val"]:g}×ATR'
            riesgo_tramo = riesgo * pct / 100.0
            lineas.append(
                f"    Tramo {e} ({pct:g}% del riesgo total → {riesgo_tramo:g}% del "
                f"equity): {_desc_disparador_tramo(tr)} → gestión: {g_label}")

    parciales = setup.get('parciales') or []
    if parciales:
        lineas.append("")
        lineas.append("  SALIDAS PARCIALES:")
        for e, ep in enumerate(parciales, 1):
            pct = ep.get('pct', 100)
            r = ep.get('r', 0)
            g = ep.get('gestion', {})
            g_label = gestion_nombres.get(g.get('tipo', 0), '—')
            if g.get('tipo') in (1, 2) and g.get('val', 0) > 0:
                g_label += f' {g["val"]:g}×ATR'
            lineas.append(
                f"    Salida {e} ({pct:g}% del total): {_desc_disparador_etapa(ep)}"
                f" → gestión: {g_label}")

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
    for linea_orden in _desc_entrada_limite(setup):
        lineas.append(f"    {linea_orden}")
    lineas.append("  SALIDA:")
    for s_l in sal:
        lineas.append(f"    {s_l}")
    if stop:
        lineas.append(f"    ADEMÁS: stop-loss a {stop:g}×ATR de la entrada "
                      f"(contra low/high de cada vela)")
    if tp:
        lineas.append(f"    ADEMÁS: take-profit a {tp:g}R del riesgo")
    if be:
        lineas.append(f"    ADEMÁS: break-even: mueve el stop al precio de entrada "
                      f"cuando el precio avanza {be:g}{be_uni} a favor")
    if trail:
        lineas.append(f"    ADEMÁS: trailing stop: el stop sigue al precio a "
                      f"{trail:g}×ATR del máximo alcanzado")
    if tiempo and plantilla != 'Patrones de velas':
        # si la salida por tiempo cierra solo una parte, la línea de abajo
        # (_desc_mecanismos_salida) es la que lo describe: no anunciar aquí un
        # cierre forzoso total que no va a ocurrir
        pct_tiempo = float((setup.get('salida_tiempo') or {}).get('pct', 100.0))
        if pct_tiempo >= 100.0:
            lineas.append(f"    ADEMÁS: cierre forzoso a +{tiempo} velas de la entrada")
        else:
            lineas.append(f"    ADEMÁS: a +{tiempo} velas de la entrada se cierra "
                          f"parte de la posición:")
    for linea_mec in _desc_mecanismos_salida(setup):
        lineas.append(f"    {linea_mec}")
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
    plantilla_txt = setup['plantilla']
    if plantilla_txt == 'Patrones de velas':
        pats = (setup.get('params') or {}).get('patrones') or []
        if pats:
            plantilla_txt = f"{plantilla_txt}: {' + '.join(pats)}"
    partes = [nombre, plantilla_txt]
    if (setup.get('entrada') or {}).get('tipo') == 'limite_fib':
        partes.append(f"entrada límite Fib {setup['entrada'].get('nivel_fib', 0.618):g}")
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
    if (filtros.get('volatilidad') or {}).get('metodo', 'ninguno') != 'ninguno':
        activos.append('volatilidad')
    if (filtros.get('sesion') or {}).get('tipo', 'ninguna') != 'ninguna':
        activos.append('sesión')
    if filtros.get('condiciones_entrada') or filtros.get('condiciones_salida'):
        activos.append('condición')
    if (filtros.get('noticias') or {}).get('activo'):
        activos.append('noticias')
    if activos:
        partes.append(f"filtros: {'+'.join(activos)}")
    return " · ".join(partes)
