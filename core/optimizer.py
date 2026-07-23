"""
core/optimizer.py
Búsqueda de parámetros para UN setup del backtest, restringida al tramo
IN-SAMPLE (ver core.backtest.dividir_is_oos): cada combinación se simula
únicamente sobre las velas [0, corte) del df, así el ranking de "mejor
configuración" nunca ve el tramo OOS.

Una vez elegida una configuración a partir de este ranking, la validación
completa (IS+OOS+WFA+Monte Carlo) se hace con el pipeline normal de
core.backtest (simular + calcular_metricas) sobre la serie entera — este
módulo no sustituye a ese pipeline, solo alimenta la selección previa.

Sin Qt, sin I/O — funciones puras pensadas para correr dentro de un QThread
de la GUI o en tests.
"""
import itertools

import numpy as np

from core.backtest import dividir_is_oos, simular, calcular_metricas
from core.strategies import generar_senales_sistema

LIMITE_COMBOS_DEFECTO = 2000
PREFIJO_RIESGO = '_riesgo.'   # evita colisión con claves de params de estrategia
N_PUNTOS_SPARKLINE = 80


def generar_grid(sweep, limite=LIMITE_COMBOS_DEFECTO):
    """sweep: {clave: {'min': ..., 'max': ..., 'step': ..., 'tipo': 'int'|'float'}}
    (solo las claves marcadas para barrer). Devuelve el producto cartesiano
    como lista de dicts {clave: valor}. Lanza ValueError si el nº de
    combinaciones supera `limite` (protección de rendimiento)."""
    claves = list(sweep)
    valores = [_valores_param(sweep[clave]) for clave in claves]

    total = 1
    for v in valores:
        total *= len(v)
    if total > limite:
        raise ValueError(
            f"{total} combinaciones supera el límite de {limite}")

    return [dict(zip(claves, tupla)) for tupla in itertools.product(*valores)]


def n_combinaciones(sweep):
    """Tamaño del producto cartesiano sin construirlo (para previsualizar en
    la UI antes de lanzar)."""
    total = 1
    for spec in sweep.values():
        total *= len(_valores_param(spec))
    return total


def _valores_param(spec):
    vmin, vmax = spec['min'], spec['max']
    paso = spec.get('step') or 1
    if spec.get('tipo') == 'int':
        paso = max(int(paso), 1)
        return list(range(int(vmin), int(vmax) + 1, paso))
    paso = float(paso) if paso else 1.0
    if paso <= 0:
        return [float(vmin)]
    n = int(round((float(vmax) - float(vmin)) / paso)) + 1
    vals = [round(float(vmin) + i * paso, 6) for i in range(max(n, 1))]
    return [v for v in vals if v <= float(vmax) + 1e-9] or [float(vmin)]


def _downsample(arr, n_puntos=N_PUNTOS_SPARKLINE):
    arr = np.asarray(arr, dtype=np.float64)
    if len(arr) <= n_puntos:
        return arr
    idx = np.linspace(0, len(arr) - 1, n_puntos).astype(int)
    return arr[idx]


MIN_BARRAS_SHARPE = 20   # con menos barras el Sharpe expansivo es ruido puro


def _sharpe_expansivo(equity, velas_por_anio=None):
    """Serie del Sharpe ACUMULADO con ventana expansiva: sharpe[t] = Sharpe
    de los log-retornos de la equity desde el inicio hasta t (misma
    convención que calcular_metricas, anualizado si se pasa velas_por_anio).
    Muestra si el Sharpe de una config creció de forma estable durante todo
    el tramo o si el valor final lo fabricó un único periodo afortunado.

    Los primeros MIN_BARRAS_SHARPE valores (y los tramos con varianza nula)
    van a NaN. Devuelve un array de longitud len(equity) - 1 (uno por
    retorno); vacío si no hay retornos suficientes."""
    eq = np.asarray(equity, dtype=np.float64)
    if len(eq) < 2:
        return np.array([])
    r = np.diff(np.log(np.maximum(eq, 1e-12)))
    n = np.arange(1, len(r) + 1, dtype=np.float64)
    media = np.cumsum(r) / n
    var = np.maximum(np.cumsum(r ** 2) / n - media ** 2, 0.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        sharpe = np.where(var > 0, media / np.sqrt(var), np.nan)
    if velas_por_anio:
        sharpe = sharpe * np.sqrt(velas_por_anio)
    sharpe[:min(MIN_BARRAS_SHARPE, len(sharpe))] = np.nan
    return sharpe


def optimizar_setup(df, setup_base, sweep_params, sweep_riesgo, config_global,
                     pct_oos, metrica='sharpe', velas_por_anio=None,
                     limite_combos=LIMITE_COMBOS_DEFECTO, progreso_cb=None):
    """Recorre la grid de combinaciones para UN setup, simulando SOLO el
    tramo IS (df.iloc[:corte], corte = dividir_is_oos(len(df), pct_oos)).

    sweep_params: barrido de claves de setup_base['params'] (numéricas).
    sweep_riesgo: barrido de 'riesgo_pct'/'stop_atr'/'tp_r'/'salida_n_velas'
    del propio setup (mismo formato que sweep_params).
    progreso_cb(i, total): opcional, llamado tras cada combinación simulada.

    Devuelve la lista de resultados — cada uno
    {'params_barridos', 'setup', 'metricas', 'equity_sparkline'} — ordenada
    descendentemente por `metrica` (una de las claves de calcular_metricas,
    p.ej. 'sharpe', 'retorno_pct', 'profit_factor', 'sqn').
    """
    n = len(df)
    corte = dividir_is_oos(n, pct_oos)
    df_is = df.iloc[:corte]
    o = df_is['open'].values
    h = df_is['high'].values
    l = df_is['low'].values
    c = df_is['close'].values
    n_is = len(df_is)

    sweep = dict(sweep_params)
    sweep.update({PREFIJO_RIESGO + k: v for k, v in sweep_riesgo.items()})
    combos = generar_grid(sweep, limite_combos) if sweep else [{}]

    resultados = []
    total = len(combos)
    for i, combo in enumerate(combos):
        params_combo = {k: v for k, v in combo.items()
                        if not k.startswith(PREFIJO_RIESGO)}
        riesgo_combo = {k[len(PREFIJO_RIESGO):]: v for k, v in combo.items()
                        if k.startswith(PREFIJO_RIESGO)}

        setup = dict(setup_base, params=dict(setup_base['params'], **params_combo))
        setup.update(riesgo_combo)

        senales = generar_senales_sistema(df_is, [setup])
        config = dict(config_global)
        config['config_por_setup'] = {0: {
            'riesgo_pct': float(setup.get('riesgo_pct', 0.01)),
            'stop_atr': float(setup.get('stop_atr', 0.0)),
            'tp_r': float(setup.get('tp_r', 0.0)),
            'salida_n_velas': int(setup.get('salida_n_velas', 0)),
        }}
        resultado = simular(o, h, l, c, senales, config)
        met = calcular_metricas(resultado, 0, n_is, velas_por_anio)

        equity = resultado['equity']
        eq0 = equity[0] if len(equity) and equity[0] > 0 else 1.0
        curva_pct = (equity / eq0 - 1.0) * 100.0

        resultados.append({
            'params_barridos': combo,
            'setup': setup,
            'metricas': met,
            'equity_sparkline': _downsample(curva_pct),
            'sharpe_sparkline': _downsample(
                _sharpe_expansivo(equity, velas_por_anio)),
        })
        if progreso_cb:
            progreso_cb(i + 1, total)

    def _clave_orden(item):
        v = item['metricas'].get(metrica)
        if v is None:
            return float('-inf')
        return v

    resultados.sort(key=_clave_orden, reverse=True)
    return resultados


# ══════════════ estadísticas del CONJUNTO de un barrido ══════════════
# Analizan la población completa de combinaciones probadas (no cada una):
# si la mayoría pierde, la "ganadora" probablemente es un outlier de
# configuración (sobreoptimización) y conviene replantear activo/indicador/
# temporalidad en vez de quedarse con el pico.

def _con_trades(resultados):
    return [r for r in resultados if r['metricas'].get('n_trades', 0) > 0]


def _pct(cumplen, total):
    return 100.0 * cumplen / total if total else None


def estadisticas_conjunto(resultados):
    """Salud global del barrido: % de combinaciones rentables, forma de la
    distribución de retornos y confirmaciones cruzadas (PF, Sharpe).
    Las combos con 0 trades no son "perdedoras" sino nulas: se excluyen de
    los porcentajes y se reportan aparte en n_sin_trades."""
    out = {'n_combos': len(resultados), 'n_sin_trades': 0,
           'pct_rentables': None, 'retorno_medio': None,
           'retorno_mediana': None, 'pct_pf_mayor_1': None,
           'pct_sharpe_pos': None, 'sesgo': None}
    operaron = _con_trades(resultados)
    out['n_sin_trades'] = len(resultados) - len(operaron)
    if not operaron:
        return out

    rets = np.array([r['metricas']['retorno_pct'] for r in operaron
                     if r['metricas'].get('retorno_pct') is not None])
    if len(rets):
        out['pct_rentables'] = _pct(int((rets > 0).sum()), len(rets))
        out['retorno_medio'] = float(rets.mean())
        out['retorno_mediana'] = float(np.median(rets))
        if len(rets) >= 3 and rets.std() > 0:
            z = (rets - rets.mean()) / rets.std()
            out['sesgo'] = float(np.mean(z ** 3))

    pfs = [r['metricas'].get('profit_factor') for r in operaron]
    pfs = [v for v in pfs if v is not None]   # inf cuenta como > 1
    out['pct_pf_mayor_1'] = _pct(sum(1 for v in pfs if v > 1), len(pfs))
    sharpes = [r['metricas'].get('sharpe') for r in operaron]
    sharpes = [v for v in sharpes if v is not None]
    out['pct_sharpe_pos'] = _pct(sum(1 for v in sharpes if v > 0), len(sharpes))
    return out


def analisis_por_parametro(resultados):
    """¿Qué parámetro decide que las combinaciones ganen o pierdan? Agrupa
    las combos por el valor de cada parámetro barrido y mide por valor el %
    rentable y el retorno mediano. 'impacto' = diferencia (puntos de %) entre
    el mejor y el peor valor de ese parámetro — el de mayor impacto es el
    principal causante. Parámetros con un único valor se omiten."""
    operaron = _con_trades(resultados)
    claves = set()
    for r in resultados:
        claves.update(r['params_barridos'])

    out = {}
    for clave in sorted(claves):
        grupos = {}
        for r in operaron:
            if clave not in r['params_barridos']:
                continue
            v = r['params_barridos'][clave]
            ret = r['metricas'].get('retorno_pct')
            if ret is not None:
                grupos.setdefault(v, []).append(ret)
        if len(grupos) < 2:
            continue
        valores = []
        pcts = []
        for v in sorted(grupos):
            rets = grupos[v]
            pct = _pct(sum(1 for x in rets if x > 0), len(rets))
            pcts.append(pct)
            valores.append((v, {'n': len(rets), 'pct_rentables': pct,
                                'retorno_mediana': float(np.median(rets))}))
        out[clave] = {'valores': valores,
                      'impacto': float(max(pcts) - min(pcts))}
    return out


def _indice_vecindad(resultados):
    """(posiciones -> resultado) para localizar combos vecinas: cada combo se
    representa por la POSICIÓN de su valor en la lista ordenada de valores
    probados de cada parámetro (posiciones consecutivas = valores adyacentes
    del barrido, sin depender del tamaño del paso)."""
    claves = sorted({k for r in resultados for k in r['params_barridos']})
    if not claves:
        return {}, [], {}
    posiciones = {}
    for k in claves:
        vals = sorted({r['params_barridos'][k] for r in resultados
                       if k in r['params_barridos']})
        posiciones[k] = {v: i for i, v in enumerate(vals)}
    indice = {}
    for r in resultados:
        try:
            pos = tuple(posiciones[k][r['params_barridos'][k]] for k in claves)
        except KeyError:
            continue
        indice[pos] = r
    return indice, claves, posiciones


def _pares_vecinos(indice, n_dims):
    """Pares (pos_a, pos_b) adyacentes: difieren en +1 en exactamente una
    dimensión (solo dirección positiva para no duplicar pares)."""
    pares = []
    for pos in indice:
        for d in range(n_dims):
            vecino = pos[:d] + (pos[d] + 1,) + pos[d + 1:]
            if vecino in indice:
                pares.append((pos, vecino))
    return pares


def analisis_vecindad(resultados, metrica='retorno_pct', top_n=10):
    """Meseta vs pico aislado. Un edge real es estable ante pequeños cambios
    de parámetros: superficie suave (rugosidad baja), élite rodeada de
    vecinas también rentables (plateau_top alto) y una zona ganadora conexa
    grande (pct_mayor_cluster) en vez de islas dispersas."""
    out = {'rugosidad': None, 'plateau_top': None, 'pct_mayor_cluster': None}
    if not resultados:
        return out
    indice, claves, _ = _indice_vecindad(resultados)
    if not indice:
        return out
    pares = _pares_vecinos(indice, len(claves))
    if not pares:
        return out

    difs = []
    for a, b in pares:
        va = indice[a]['metricas'].get(metrica)
        vb = indice[b]['metricas'].get(metrica)
        if va is not None and vb is not None:
            difs.append(abs(va - vb))
    if difs:
        out['rugosidad'] = float(np.mean(difs))

    # vecinas de cada posición (grafo no dirigido)
    vecinas = {pos: [] for pos in indice}
    for a, b in pares:
        vecinas[a].append(b)
        vecinas[b].append(a)

    # plateau del top: % de vecinas rentables de las top_n mejores combos
    con_metrica = [(pos, r) for pos, r in indice.items()
                   if r['metricas'].get(metrica) is not None]
    con_metrica.sort(key=lambda pr: pr[1]['metricas'][metrica], reverse=True)
    total_vec = rentables_vec = 0
    for pos, _r in con_metrica[:top_n]:
        for v in vecinas[pos]:
            ret = indice[v]['metricas'].get('retorno_pct')
            if ret is None:
                continue
            total_vec += 1
            rentables_vec += 1 if ret > 0 else 0
    out['plateau_top'] = _pct(rentables_vec, total_vec)

    # mayor cluster conexo de combos rentables
    rentables = {pos for pos, r in indice.items()
                 if (r['metricas'].get('retorno_pct') or 0) > 0}
    visitadas = set()
    mayor = 0
    for inicio in rentables:
        if inicio in visitadas:
            continue
        cola = [inicio]
        visitadas.add(inicio)
        tam = 0
        while cola:
            pos = cola.pop()
            tam += 1
            for v in vecinas[pos]:
                if v in rentables and v not in visitadas:
                    visitadas.add(v)
                    cola.append(v)
        mayor = max(mayor, tam)
    out['pct_mayor_cluster'] = _pct(mayor, len(indice))
    return out


def fiabilidad_estadistica(resultados, min_sqn=1.0):
    """¿Cuánto fiarse del conjunto? Un nº de trades fijo (p.ej. "≥30") no es
    un criterio válido entre estilos: un position trading en velas diarias
    puede tardar años de IS en juntar 30 trades, mientras que para un scalper
    30 es una muestra minúscula frente a los miles que genera en el mismo
    periodo. En su lugar se usa el SQN de cada combo (System Quality Number =
    sqrt(n) · media(r_multiple) / std(r_multiple), ya calculado por
    calcular_metricas en core/backtest.py) — matemáticamente es un
    t-estadístico de la consistencia de los trades: compensa solo pocos
    trades si son muy consistentes y exige más si son ruidosos, sin necesitar
    saber si el sistema es scalping o swing. El nº de trades se conserva como
    dato informativo (mediana), no como criterio de corte.

    Un beneficio desproporcionado en la mejor config frente a la mediana, o
    concentrado en un puñado de combos, siguen sugiriendo suerte y no edge."""
    out = {'pct_sqn_suficiente': None, 'mediana_sqn': None,
           'mediana_trades': None, 'ratio_mejor_mediana': None,
           'concentracion_top5': None, 'min_sqn': min_sqn}
    if not resultados:
        return out
    ns = [r['metricas'].get('n_trades', 0) for r in resultados]
    out['mediana_trades'] = float(np.median(ns))

    sqns = [r['metricas'].get('sqn') for r in resultados]
    sqns_validos = [v for v in sqns if v is not None]
    if sqns_validos:
        out['pct_sqn_suficiente'] = _pct(
            sum(1 for v in sqns_validos if v >= min_sqn), len(sqns_validos))
        out['mediana_sqn'] = float(np.median(sqns_validos))

    rets = [r['metricas']['retorno_pct'] for r in _con_trades(resultados)
            if r['metricas'].get('retorno_pct') is not None]
    if rets:
        mediana = float(np.median(rets))
        if mediana > 0:
            out['ratio_mejor_mediana'] = float(max(rets)) / mediana

    pnls = sorted((r['metricas'].get('pnl_total') or 0.0 for r in resultados),
                  reverse=True)
    ganadores = [p for p in pnls if p > 0]
    if ganadores:
        k = max(1, int(round(0.05 * len(resultados))))
        out['concentracion_top5'] = _pct(sum(ganadores[:k]), sum(ganadores))
    return out
