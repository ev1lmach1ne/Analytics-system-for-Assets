"""
core/backtest.py
Motor de backtest vectorizado + métricas + split IS/OOS + Walk-Forward +
Montecarlo. Funciones puras al estilo de core/candle_patterns.py: sin Qt,
sin I/O — pensadas para correr dentro de un QThread de la GUI o en tests.

Convenciones de ejecución (realistas, tomadas del backtest 1h de
library/Backtests, que es el más correcto de los scripts históricos):
- Señal en la vela t → entrada/salida al OPEN de t+1 (nunca se opera la
  misma vela que genera la señal).
- Stop-loss (si está activo) se comprueba contra el low/high de CADA vela
  mientras la posición está abierta y se ejecuta al precio del stop.
- Slippage: encarece la entrada y abarata la salida (en % del precio).
- Comisión: % del nocional, cobrada en entrada y en salida.
- Una sola posición a la vez (sin piramidar), long o short.

Dimensionamiento por riesgo:
- Cada trade arriesga un % del equity actual (riesgo_pct, o el específico
  del setup_id de la señal si se definió riesgo_por_setup).
- La distancia de riesgo es stop_atr * ATR si hay stop; si no hay stop se
  usa 2*ATR como distancia de REFERENCIA solo para dimensionar (documentado:
  sin stop real, una vela adversa grande puede perder más del riesgo teórico).

Motivos de salida (columna 'motivo' de cada trade):
0 = señal contraria/salida, 1 = stop-loss, 2 = take-profit,
3 = salida por tiempo (n velas), 4 = fin de datos.
"""
import numpy as np
from numba import njit

MOTIVOS_SALIDA = {0: 'Señal', 1: 'Stop', 2: 'Take-profit', 3: 'Tiempo', 4: 'Fin datos'}

# columnas del array de trades que devuelve _simular_numba
(_T_IDX_IN, _T_IDX_OUT, _T_DIR, _T_SETUP, _T_PIN, _T_POUT, _T_PNL, _T_MOTIVO,
 _T_EQ_IN, _T_UNIDADES) = range(10)


@njit(nogil=True)
def _simular_numba(o, h, l, c, ent_long, ent_short, sal_long, sal_short,
                   setup_id, atr, riesgos_setup, stops_setup, tps_setup,
                   tiempos_setup, capital_inicial, comision_pct, slippage_pct):
    """Bucle del motor. Devuelve (trades[n,10], n_trades, equity[n]).

    riesgos_setup/stops_setup/tps_setup/tiempos_setup: arrays[64] indexados
    por setup_id — riesgo %, stop en ×ATR, take-profit en R y salida por
    tiempo (velas) de cada setup (0 = desactivado en stop/tp/tiempo).
    sal_long/sal_short: BITMASK int64 por vela — bit k activo = el setup k
    pide salir; solo cierra la posición si el bit de SU setup está activo.
    Una señal de entrada contraria solo revierte la posición si viene del
    MISMO setup (setup_id de esa vela == setup de la posición).
    """
    n = len(c)
    max_trades = n // 2 + 1
    trades = np.zeros((max_trades, 10))
    n_trades = 0
    equity = np.full(n, capital_inicial)

    cap = capital_inicial
    en_pos = False
    dir_pos = 0            # +1 long, -1 short
    precio_in = 0.0
    unidades = 0.0
    stop_precio = 0.0
    tp_precio = 0.0
    idx_in = 0
    setup_in = 0
    pendiente_entrada = 0   # 0 no, +1 long, -1 short (señal de la vela previa)
    pendiente_salida = False
    pendiente_setup = 0

    for i in range(n):
        # ── ejecutar al open lo pendiente de la vela anterior ──
        if en_pos and pendiente_salida:
            precio_out = o[i] * (1.0 - slippage_pct * dir_pos)
            pnl = (precio_out - precio_in) * unidades * dir_pos
            pnl -= (precio_in + precio_out) * unidades * comision_pct
            cap += pnl
            trades[n_trades, _T_IDX_IN] = idx_in
            trades[n_trades, _T_IDX_OUT] = i
            trades[n_trades, _T_DIR] = dir_pos
            trades[n_trades, _T_SETUP] = setup_in
            trades[n_trades, _T_PIN] = precio_in
            trades[n_trades, _T_POUT] = precio_out
            trades[n_trades, _T_PNL] = pnl
            trades[n_trades, _T_MOTIVO] = 0
            trades[n_trades, _T_EQ_IN] = cap - pnl
            trades[n_trades, _T_UNIDADES] = unidades
            n_trades += 1
            en_pos = False
            pendiente_salida = False

        if (not en_pos) and pendiente_entrada != 0:
            d = pendiente_entrada
            precio = o[i] * (1.0 + slippage_pct * d)
            ref_atr = atr[i] if atr[i] > 0 else precio * 0.01
            stop_atr_s = stops_setup[pendiente_setup]
            tp_r_s = tps_setup[pendiente_setup]
            dist = stop_atr_s * ref_atr if stop_atr_s > 0 else 2.0 * ref_atr
            riesgo = riesgos_setup[pendiente_setup]
            if dist > 0 and riesgo > 0 and cap > 0:
                unidades = (cap * riesgo) / dist
                en_pos = True
                dir_pos = d
                precio_in = precio
                idx_in = i
                setup_in = pendiente_setup
                stop_precio = precio - dist * d if stop_atr_s > 0 else 0.0
                tp_precio = precio + tp_r_s * dist * d if tp_r_s > 0 else 0.0
            pendiente_entrada = 0

        # ── gestión intra-vela de la posición abierta ──
        if en_pos:
            salida_precio = 0.0
            motivo = -1
            # stop-loss contra low/high de la vela
            if stop_precio > 0.0:
                if dir_pos > 0 and l[i] <= stop_precio:
                    salida_precio = stop_precio
                    motivo = 1
                elif dir_pos < 0 and h[i] >= stop_precio:
                    salida_precio = stop_precio
                    motivo = 1
            # take-profit (si el stop no saltó antes en esta vela)
            if motivo < 0 and tp_precio > 0.0:
                if dir_pos > 0 and h[i] >= tp_precio:
                    salida_precio = tp_precio
                    motivo = 2
                elif dir_pos < 0 and l[i] <= tp_precio:
                    salida_precio = tp_precio
                    motivo = 2
            # salida por tiempo: al cierre de la vela n desde la entrada
            n_velas_s = tiempos_setup[setup_in]
            if motivo < 0 and n_velas_s > 0 and (i - idx_in) >= n_velas_s:
                salida_precio = c[i]
                motivo = 3
            if motivo < 0 and i == n - 1:
                salida_precio = c[i]
                motivo = 4

            if motivo >= 0:
                precio_out = salida_precio * (1.0 - slippage_pct * dir_pos)
                pnl = (precio_out - precio_in) * unidades * dir_pos
                pnl -= (precio_in + precio_out) * unidades * comision_pct
                cap += pnl
                trades[n_trades, _T_IDX_IN] = idx_in
                trades[n_trades, _T_IDX_OUT] = i
                trades[n_trades, _T_DIR] = dir_pos
                trades[n_trades, _T_SETUP] = setup_in
                trades[n_trades, _T_PIN] = precio_in
                trades[n_trades, _T_POUT] = precio_out
                trades[n_trades, _T_PNL] = pnl
                trades[n_trades, _T_MOTIVO] = motivo
                trades[n_trades, _T_EQ_IN] = cap - pnl
                trades[n_trades, _T_UNIDADES] = unidades
                n_trades += 1
                en_pos = False
                pendiente_salida = False

        # ── leer señales de la vela i (se ejecutarán al open de i+1) ──
        if en_pos:
            bit = np.int64(1) << np.int64(setup_in)
            if (dir_pos > 0 and (sal_long[i] & bit) != 0) \
                    or (dir_pos < 0 and (sal_short[i] & bit) != 0):
                pendiente_salida = True
            # señal contraria del MISMO setup también cierra (revierte);
            # la de otro setup no toca esta posición
            if setup_id[i] == setup_in and (
                    (dir_pos > 0 and ent_short[i]) or (dir_pos < 0 and ent_long[i])):
                pendiente_salida = True
        if not en_pos or pendiente_salida:
            if ent_long[i]:
                pendiente_entrada = 1
                pendiente_setup = setup_id[i]
            elif ent_short[i]:
                pendiente_entrada = -1
                pendiente_setup = setup_id[i]

        # ── equity mark-to-market al cierre ──
        if en_pos:
            equity[i] = cap + (c[i] - precio_in) * unidades * dir_pos
        else:
            equity[i] = cap

    return trades[:n_trades], n_trades, equity


def simular(o, h, l, c, senales, config):
    """Corre el motor y devuelve un dict con trades (dict de arrays),
    equity, drawdown y capital final.

    senales: dict de core/strategies.generar_senales o
    generar_senales_sistema — claves entradas_long/entradas_short (bool[n]),
    salidas_long/salidas_short (bool[n] o bitmask int64[n]: bool True
    equivale a "todos los setups pueden salir"), setup_id (int64[n], 0 por
    defecto, máx. 63) y atr (float64[n]).
    config: dict — capital_inicial, riesgo_pct, comision_pct, slippage_pct,
    stop_atr, tp_r, salida_n_velas (defectos globales) y opcionalmente
    config_por_setup: {id: {'riesgo_pct','stop_atr','tp_r','salida_n_velas'}}
    para que cada setup del sistema tenga su propio riesgo/stop/TP/tiempo
    (riesgo_por_setup: {id: pct} se acepta como forma corta, compat v1).
    """
    o = np.ascontiguousarray(o, dtype=np.float64)
    h = np.ascontiguousarray(h, dtype=np.float64)
    l = np.ascontiguousarray(l, dtype=np.float64)
    c = np.ascontiguousarray(c, dtype=np.float64)
    n = len(c)

    setup = senales.get('setup_id')
    setup = np.zeros(n, dtype=np.int64) if setup is None \
        else np.clip(np.ascontiguousarray(setup, dtype=np.int64), 0, 63)
    atr = senales.get('atr')
    atr = np.zeros(n, dtype=np.float64) if atr is None \
        else np.nan_to_num(np.ascontiguousarray(atr, dtype=np.float64))

    def _mascara_salida(a):
        a = np.asarray(a)
        if a.dtype == np.bool_:
            # bool True = cualquier setup puede salir (todos los bits a 1)
            return np.where(a, np.int64(-1), np.int64(0))
        return np.ascontiguousarray(a, dtype=np.int64)

    # config por setup: los escalares globales actúan de defecto
    riesgos = np.full(64, float(config.get('riesgo_pct', 0.01)))
    stops = np.full(64, float(config.get('stop_atr', 0.0)))
    tps = np.full(64, float(config.get('tp_r', 0.0)))
    tiempos = np.full(64, int(config.get('salida_n_velas', 0)), dtype=np.int64)
    for sid, pct in (config.get('riesgo_por_setup') or {}).items():
        sid = int(sid)
        if 0 <= sid < 64:
            riesgos[sid] = float(pct)
    for sid, cfg_s in (config.get('config_por_setup') or {}).items():
        sid = int(sid)
        if not 0 <= sid < 64:
            continue
        if 'riesgo_pct' in cfg_s:
            riesgos[sid] = float(cfg_s['riesgo_pct'])
        if 'stop_atr' in cfg_s:
            stops[sid] = float(cfg_s['stop_atr'])
        if 'tp_r' in cfg_s:
            tps[sid] = float(cfg_s['tp_r'])
        if 'salida_n_velas' in cfg_s:
            tiempos[sid] = int(cfg_s['salida_n_velas'])

    trades_arr, n_trades, equity = _simular_numba(
        o, h, l, c,
        np.ascontiguousarray(senales['entradas_long'], dtype=np.bool_),
        np.ascontiguousarray(senales['entradas_short'], dtype=np.bool_),
        _mascara_salida(senales['salidas_long']),
        _mascara_salida(senales['salidas_short']),
        setup, atr, riesgos, stops, tps, tiempos,
        float(config.get('capital_inicial', 10000.0)),
        float(config.get('comision_pct', 0.0005)),
        float(config.get('slippage_pct', 0.0002)),
    )

    t = trades_arr
    trades = {
        'idx_entrada': t[:, _T_IDX_IN].astype(np.int64),
        'idx_salida': t[:, _T_IDX_OUT].astype(np.int64),
        'dir': t[:, _T_DIR].astype(np.int8),
        'setup': t[:, _T_SETUP].astype(np.int64),
        'precio_entrada': t[:, _T_PIN],
        'precio_salida': t[:, _T_POUT],
        'pnl': t[:, _T_PNL],
        'motivo': t[:, _T_MOTIVO].astype(np.int64),
        'equity_entrada': t[:, _T_EQ_IN],
        'unidades': t[:, _T_UNIDADES],
    }
    # retorno del trade como fracción del equity al entrar (para Montecarlo
    # y expectancy comparables entre trades con capital distinto)
    with np.errstate(divide='ignore', invalid='ignore'):
        trades['ret_pct'] = np.where(trades['equity_entrada'] > 0,
                                     trades['pnl'] / trades['equity_entrada'], 0.0)

    # r_multiple: PnL en múltiplos del riesgo arriesgado en ESE trade (según
    # el riesgo % del setup que lo abrió) — base del SQN, más correcto que
    # ret_pct a secas porque normaliza por el riesgo pretendido, no por el
    # equity total.
    riesgo_trade = riesgos[np.clip(trades['setup'], 0, 63)]
    riesgo_absoluto = riesgo_trade * trades['equity_entrada']
    with np.errstate(divide='ignore', invalid='ignore'):
        trades['r_multiple'] = np.where(riesgo_absoluto > 0,
                                        trades['pnl'] / riesgo_absoluto, 0.0)

    # costo de comisión y notional ida+vuelta de cada trade — mismo cálculo
    # que hace el motor internamente, aquí solo se reexpone por trade para
    # las métricas de sensibilidad a comisión/slippage (ver calcular_metricas)
    comision_pct_v = float(config.get('comision_pct', 0.0005))
    trades['notional_redondo'] = (trades['precio_entrada'] + trades['precio_salida']) \
        * trades['unidades']
    trades['costo_comision'] = trades['notional_redondo'] * comision_pct_v

    eq_max = np.maximum.accumulate(equity)
    with np.errstate(divide='ignore', invalid='ignore'):
        drawdown = np.where(eq_max > 0, equity / eq_max - 1.0, 0.0)

    return {'trades': trades, 'equity': equity, 'drawdown': drawdown,
            'capital_final': float(equity[-1]) if n else 0.0,
            'n_trades': int(n_trades)}


def dividir_is_oos(n, pct_oos=0.30):
    """Índice de corte cronológico: [0, corte) = IS, [corte, n) = OOS."""
    pct_oos = min(max(float(pct_oos), 0.0), 0.95)
    return int(round(n * (1.0 - pct_oos)))


def _analizar_drawdowns(equity):
    """Descompone la equity curve en episodios de drawdown (tramos por
    debajo del máximo previo). Devuelve lista de (profundidad_pct,
    duracion_en_velas) — duracion es None si el episodio seguía abierto
    (equity no recuperada) al final de la serie.

    Un episodio empieza en el primer punto por debajo de un nuevo máximo y
    termina cuando la equity vuelve a igualar/superar ese máximo (duración =
    velas desde el PICO hasta la recuperación, no desde el fondo)."""
    n = len(equity)
    if n < 2:
        return []
    eq_max = np.maximum.accumulate(equity)
    bajo_maximo = equity < eq_max
    episodios = []
    i = 0
    while i < n:
        if not bajo_maximo[i]:
            i += 1
            continue
        # nuevo episodio: el pico es el máximo vigente justo antes de entrar
        pico_valor = eq_max[i]
        idx_pico = i - 1 if i > 0 else 0
        j = i
        fondo = equity[i]
        while j < n and bajo_maximo[j]:
            fondo = min(fondo, equity[j])
            j += 1
        # j es el primer índice (o n) donde ya no está bajo el máximo, es
        # decir, donde se recuperó — o fin de serie si nunca recuperó
        recuperado = j < n
        profundidad_pct = (fondo / pico_valor - 1.0) * 100.0 if pico_valor > 0 else 0.0
        duracion = (j - idx_pico) if recuperado else None
        episodios.append((profundidad_pct, duracion))
        i = j
    return episodios


def _r2_equity(eq):
    """R² de la regresión lineal de la equity curve contra el índice
    temporal — cuanto más cerca de 1, más consistente/recta es la curva."""
    n = len(eq)
    if n < 3:
        return None
    x = np.arange(n, dtype=np.float64)
    y = np.asarray(eq, dtype=np.float64)
    if np.std(y) == 0:
        return None
    pendiente, intercepto = np.polyfit(x, y, 1)
    pred = pendiente * x + intercepto
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot == 0:
        return None
    return 1.0 - ss_res / ss_tot


def calcular_metricas(resultado, idx_ini=0, idx_fin=None, velas_por_anio=None):
    """Métricas de un tramo [idx_ini, idx_fin) del resultado de simular().
    Un trade pertenece al tramo si su vela de ENTRADA cae dentro.
    velas_por_anio: para anualizar retorno y Sharpe (None = no anualiza).

    Incluye, además de rentabilidad, un bloque de ROBUSTEZ:
    - r2_equity: R² de la curva de capital contra el tiempo (cuanto más
      cerca de 1, más recta/consistente el crecimiento).
    - dd_promedio_pct / tiempo_recuperacion_medio / tiempo_recuperacion_max:
      por EPISODIO de drawdown (no punto a punto).
    - sqn: System Quality Number = sqrt(n) * media(r_multiple) / std(r_multiple)
      (Van Tharp; r_multiple = PnL en múltiplos del riesgo arriesgado en ese
      trade). >2.0 bueno, >3.0 excelente.
    - payoff_ratio: ganancia media de los trades ganadores / |pérdida media
      de los perdedores| — si es alto pero el win rate es bajo, cuidado con
      depender de pocos trades grandes (ver pct_mejor_trade).
    - pct_mejor_trade: % del PnL total que aporta el mejor trade aislado —
      alto (>20-30%) sugiere que el sistema depende de outliers, no de un
      edge consistente.
    - slippage_minimo_pct: slippage adicional (%, mismo cálculo lineal que
      la comisión) que agotaría la expectancy media del sistema hasta cero.
      Negativo = el tramo ya es negativo sin slippage adicional.
    - impacto_comisiones_pct: comisión pagada como % de la ganancia bruta
      (antes de comisión) — en scalping puede devorar el edge entero. Solo
      se calcula con el tramo en ganancia neta (pnl_total > 0): en pérdidas
      la "ganancia bruta" puede quedar accidentalmente positiva y pequeña,
      disparando el ratio a valores sin sentido.
    """
    equity = resultado['equity']
    n = len(equity)
    idx_fin = n if idx_fin is None else min(idx_fin, n)
    tr = resultado['trades']
    m = (tr['idx_entrada'] >= idx_ini) & (tr['idx_entrada'] < idx_fin)
    pnl = tr['pnl'][m]
    ret_pct = tr['ret_pct'][m]
    dur = (tr['idx_salida'][m] - tr['idx_entrada'][m])

    out = {'n_trades': int(m.sum()), 'win_rate': None, 'profit_factor': None,
           'retorno_pct': None, 'retorno_anual_pct': None, 'max_dd_pct': None,
           'sharpe': None, 'expectancy_pct': None, 'racha_perdedora': 0,
           'duracion_media': None, 'pnl_total': float(pnl.sum()),
           'r2_equity': None, 'dd_promedio_pct': None,
           'tiempo_recuperacion_medio': None, 'tiempo_recuperacion_max': None,
           'sqn': None, 'payoff_ratio': None, 'pct_mejor_trade': None,
           'slippage_minimo_pct': None, 'impacto_comisiones_pct': None}

    if idx_fin <= idx_ini:
        return out

    eq = equity[idx_ini:idx_fin]
    eq0 = eq[0] if eq[0] > 0 else 1.0
    out['retorno_pct'] = float(eq[-1] / eq0 - 1.0) * 100.0
    eq_max = np.maximum.accumulate(eq)
    with np.errstate(divide='ignore', invalid='ignore'):
        dd = np.where(eq_max > 0, eq / eq_max - 1.0, 0.0)
    out['max_dd_pct'] = float(dd.min()) * 100.0
    out['r2_equity'] = _r2_equity(eq)

    episodios = _analizar_drawdowns(eq)
    if episodios:
        out['dd_promedio_pct'] = float(np.mean([p for p, _ in episodios]))
        recuperados = [d for _, d in episodios if d is not None]
        if recuperados:
            out['tiempo_recuperacion_medio'] = float(np.mean(recuperados))
            out['tiempo_recuperacion_max'] = float(np.max(recuperados))

    with np.errstate(divide='ignore', invalid='ignore'):
        ret_barras = np.diff(np.log(np.maximum(eq, 1e-12)))
    if len(ret_barras) > 1 and np.std(ret_barras) > 0:
        sharpe = float(np.mean(ret_barras) / np.std(ret_barras))
        if velas_por_anio:
            sharpe *= np.sqrt(velas_por_anio)
        out['sharpe'] = sharpe
    if velas_por_anio and len(eq) > 1:
        anios = len(eq) / velas_por_anio
        if anios > 0 and eq[-1] > 0:
            out['retorno_anual_pct'] = float((eq[-1] / eq0) ** (1.0 / anios) - 1.0) * 100.0

    if len(pnl):
        ganadores = pnl > 0
        out['win_rate'] = float(ganadores.mean())
        ganancia = float(pnl[ganadores].sum())
        perdida = float(-pnl[~ganadores].sum())
        out['profit_factor'] = (ganancia / perdida) if perdida > 0 else float('inf')
        out['expectancy_pct'] = float(ret_pct.mean()) * 100.0
        out['duracion_media'] = float(dur.mean())
        # racha perdedora máxima
        racha = peor = 0
        for g in ganadores:
            racha = 0 if g else racha + 1
            peor = max(peor, racha)
        out['racha_perdedora'] = int(peor)

        r_mult = tr['r_multiple'][m]
        if len(r_mult) >= 2 and np.std(r_mult) > 0:
            out['sqn'] = float(np.sqrt(len(r_mult)) * np.mean(r_mult) / np.std(r_mult))
        if ganadores.any() and (~ganadores).any():
            perdida_media = float(-pnl[~ganadores].mean())
            if perdida_media > 0:
                out['payoff_ratio'] = float(pnl[ganadores].mean()) / perdida_media
        if out['pnl_total'] > 0:
            out['pct_mejor_trade'] = float(pnl.max()) / out['pnl_total'] * 100.0

        notional = tr['notional_redondo'][m]
        if notional.mean() > 0:
            out['slippage_minimo_pct'] = float(pnl.mean()) / float(notional.mean()) * 100.0
        costo_comision = float(tr['costo_comision'][m].sum())
        bruto = out['pnl_total'] + costo_comision
        # exigir pnl_total > 0 y no solo bruto > 0: con el sistema en
        # pérdidas netas, "bruto" puede quedar accidentalmente positivo y
        # pequeño (la comisión compensa parte de la pérdida), disparando el
        # ratio a valores sin sentido (p.ej. 700%) — el ratio solo tiene
        # lectura útil cuando de verdad hay ganancia de la que "comerse"
        if out['pnl_total'] > 0 and bruto > 0:
            out['impacto_comisiones_pct'] = costo_comision / bruto * 100.0
    return out


def walk_forward(o, h, l, c, senales, config, n_ventanas=5,
                 velas_por_anio=None):
    """Walk-forward v1 (parámetros fijos): divide la serie en n_ventanas
    tramos consecutivos y calcula las métricas de cada tramo por separado
    sobre UNA sola simulación completa — mide la estabilidad temporal del
    sistema (¿funciona en todas las épocas o solo en una?). Cuando se añada
    la búsqueda de parámetros, cada ventana reoptimizará en su IS.
    """
    n = len(c)
    resultado = simular(o, h, l, c, senales, config)
    bordes = np.linspace(0, n, n_ventanas + 1).astype(int)
    ventanas = []
    for k in range(n_ventanas):
        met = calcular_metricas(resultado, bordes[k], bordes[k + 1], velas_por_anio)
        met['idx_ini'] = int(bordes[k])
        met['idx_fin'] = int(bordes[k + 1])
        ventanas.append(met)
    return {'ventanas': ventanas, 'resultado': resultado}


def montecarlo(trades, capital_inicial, n_sims=1000, semilla=None,
               umbral_ruina=0.5):
    """Remuestreo del ORDEN de los trades (permutación con reemplazo /
    bootstrap de ret_pct): distribución de equity final y de max drawdown si
    los mismos trades hubieran llegado en otro orden u otra muestra.

    Devuelve percentiles 5/50/95 de la curva, histogramas de retorno final
    y max DD, probabilidad de acabar en negativo y de 'ruina' (equity por
    debajo de umbral_ruina * capital inicial en algún momento).
    """
    ret = np.asarray(trades['ret_pct'], dtype=np.float64)
    n_tr = len(ret)
    vacio = {'curvas_pct': None, 'finales': np.array([]), 'max_dds': np.array([]),
             'prob_negativo': None, 'prob_ruina': None, 'n_sims': 0}
    if n_tr < 2:
        return vacio

    rng = np.random.default_rng(semilla)
    idx = rng.integers(0, n_tr, size=(n_sims, n_tr))
    factores = 1.0 + ret[idx]                      # [n_sims, n_tr]
    curvas = capital_inicial * np.cumprod(factores, axis=1)
    curvas = np.concatenate(
        [np.full((n_sims, 1), capital_inicial), curvas], axis=1)

    maximos = np.maximum.accumulate(curvas, axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        dds = curvas / maximos - 1.0
    max_dds = dds.min(axis=1) * 100.0
    finales = curvas[:, -1]

    p5, p50, p95 = np.percentile(curvas, [5, 50, 95], axis=0)
    return {
        'curvas_pct': {'p5': p5, 'p50': p50, 'p95': p95},
        'finales': finales,
        'max_dds': max_dds,
        'prob_negativo': float((finales < capital_inicial).mean()),
        'prob_ruina': float((curvas.min(axis=1) < capital_inicial * umbral_ruina).mean()),
        'n_sims': int(n_sims),
    }
