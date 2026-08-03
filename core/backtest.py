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
- Una sola posición a la vez por setup (long o short, sin invertir directo);
  la posición SÍ puede construirse en varios tramos (ver «Entrada escalonada»).

simular() devuelve, además de 'trades'/'entradas'/'equity', dos series de
longitud n alineadas con las velas — instrumentación para el gráfico, no
afectan al resultado: 'stop_track' (nivel de stop vigente en cada vela; NaN
si no hay posición o el setup no tiene stop) y 'entrada_track' (precio de
entrada vigente, el medio ponderado tras promediar; NaN si no hay posición).
La vela en la que una posición cierra del todo conserva su último valor
válido (no NaN), para que una línea dibujada sobre estas series llegue hasta
el cierre en vez de cortarse una vela antes.

Dimensionamiento por riesgo:
- Cada trade arriesga un % del equity actual (riesgo_pct, o el específico
  del setup_id de la señal si se definió riesgo_por_setup).
- La distancia de riesgo es stop_atr * ATR si hay stop; si no hay stop se
  usa 2*ATR como distancia de REFERENCIA solo para dimensionar (documentado:
  sin stop real, una vela adversa grande puede perder más del riesgo teórico).

Motivos de salida (columna 'motivo' de cada trade):
0 = señal contraria/salida, 1 = stop-loss, 2 = take-profit,
3 = salida por tiempo (n velas), 4 = fin de datos, 5 = cierre parcial.

Columna 'parcial' de cada trade: 0 = cierre completo por el motivo de arriba ·
N>0 = etapa secuencial nº N de 'parciales' · N<0 = cierre PARCIAL de uno de los
cinco mecanismos globales de salida (-1 stop original, -2 take-profit,
-3 break-even, -4 trailing, -5 salida por tiempo). Ver abajo.

Cierre parcial por mecanismo (config_por_setup[sid]['salida_stop'],
['salida_tp'], ['salida_be'], ['salida_trailing'], ['salida_tiempo']): cada uno
es un dict con 'pct' (% de lo que quede ABIERTO en el momento del disparo, no
del tamaño original: estos mecanismos no son secuenciales y pueden dispararse en
cualquier orden), más 'condiciones' y 'gestion' opcionales con el mismo formato
que las etapas de 'parciales'. Con pct=100 (el defecto, y lo que se asume si el
dict no existe) el comportamiento es exactamente el de siempre: cierran toda la
posición.

DISPARO ÚNICO: cada mecanismo cierra su % como mucho UNA VEZ por posición; a
partir de ahí cualquier disparo posterior suyo cierra todo lo que quede. Sin
esta regla, condiciones que siguen siendo ciertas vela tras vela (la del TP
mientras el precio siga sobre el objetivo) drenarían la posición poco a poco.
Stop-loss, break-even y trailing comparten un único nivel de stop, así que el
motor recuerda cuál de los tres lo movió por última vez (`origen_stop`) para
saber qué 'pct' aplicar cuando el precio lo toca — y como son mecanismos
distintos, cada uno conserva su propio turno.

EXCEPCIÓN, la salida por tiempo: tras cerrar su parte queda AGOTADA para esa
posición, en vez de pasar a cerrar el resto. Su condición ((i - idx_in) >= N) es
cierta en TODAS las velas siguientes, así que si volviera a dispararse cerraría
el remanente una vela después y el parcial no serviría de nada. El resto vive
hasta que lo cierre el stop, el TP, la señal o el fin de datos. El stop y el TP
sí siguen cortando: son redes de seguridad, no reglas de tiempo.

Salidas parciales (config_por_setup[sid]['parciales']): lista de etapas, cada
una con 'pct' (% del tamaño ORIGINAL de la posición) y un disparador
('trigger'): 'r' = al alcanzar +N R (intra-vela), 'cond' = cuando se cumplen
sus condiciones (intra-vela), 'senal' = con la propia señal de salida de la
plantilla (al open de la vela siguiente, como cualquier salida por señal),
'estancamiento' = intra-vela, si tras 'velas_max' velas desde la entrada el
avance a favor (medido con max_fav, el mejor precio alcanzado hasta ahora, en
múltiplos de la R vigente en esa vela) no llega a 'r_min' — un time-stop
condicionado al rendimiento, para cortar operaciones que no arrancan sin
esperar al stop real. Como max_fav no retrocede, el avance en R es monótono:
una vez alcanzado 'r_min' la etapa ya no puede disparar por estancamiento.
Una sola etapa al 100% por señal equivale exactamente a no tener etapas.

Entrada escalonada (config_por_setup[sid]['tramos']): lista de tramos que
construyen la MISMA posición en varias entradas (promediar a la baja en
reversión a la media, piramidar a favor en tendencia, o repartir la ejecución
en varias velas). Cada tramo tiene 'pct' (% del RIESGO total del setup, no del
tamaño) y un disparador ('trigger'): 'senal' = se repite la señal de entrada
de la plantilla (tramo 1 por defecto) · 'velas' = a +N velas de la 1ª entrada
· 'retroceso' = N×ATR en contra (promediar) · 'avance' = +N R a favor
(pirámide) · 'cond' = solo sus condiciones. Cada tramo se dimensiona por SU
PROPIA distancia al stop vigente en ese momento (no el de la 1ª entrada), así
que si un tramo anterior movió el stop (BE/trailing) el riesgo total del
sistema sigue siendo el pretendido. Tras cada tramo se recalcula el precio de
entrada como la media ponderada de todos los tramos ejecutados, y desde ahí se
remiden el take-profit, el break-even y el trailing. Un solo tramo al 100% a
la señal equivale exactamente a no tener tramos (la entrada de siempre). Las
ejecuciones de tramos NO son filas de 'trades' (no son cierres): se devuelven
aparte en 'entradas', para no contaminar win rate/expectancy/Montecarlo.
"""
import numpy as np
from numba import njit

MOTIVOS_SALIDA = {0: 'Señal', 1: 'Stop', 2: 'Take-profit', 3: 'Tiempo', 4: 'Fin datos', 5: 'Parcial'}

# disparador de un tramo de entrada escalonada (ver docstring del módulo)
TRIGGERS_TRAMO_ENTRADA = {'senal': 0, 'velas': 1, 'retroceso': 2, 'avance': 3, 'cond': 4}

# mecanismos globales de salida con 'pct' propio (ver docstring del módulo).
# El orden fija el índice en los arrays mec_* y el código de 'parcial' (-1..-5).
MECANISMOS_SALIDA = ('salida_stop', 'salida_tp', 'salida_be', 'salida_trailing',
                     'salida_tiempo')
_M_STOP, _M_TP, _M_BE, _M_TRAILING, _M_TIEMPO = range(5)
# origen del nivel de stop vigente -> índice del mecanismo cuyo 'pct' se aplica
_ORIGEN_A_MECANISMO = (_M_STOP, _M_BE, _M_TRAILING)

# columnas del array de trades que devuelve _simular_numba
(_T_IDX_IN, _T_IDX_OUT, _T_DIR, _T_SETUP, _T_PIN, _T_POUT, _T_PNL, _T_MOTIVO,
 _T_EQ_IN, _T_UNIDADES, _T_STOP, _T_PARCIAL) = range(12)
_N_COLS_TRADE = 12

# columnas del array de entradas (aperturas y tramos) que devuelve _simular_numba
(_E_IDX, _E_DIR, _E_SETUP, _E_PRECIO, _E_UNIDADES, _E_TRAMO) = range(6)
_N_COLS_ENTRADA = 6


@njit(nogil=True)
def _aplicar_gestion_parcial(g_tipo, g_val, dir_pos, stop_precio, precio_ref,
                             setup_in, bes_setup, trails_setup, be_aplicado):
    """Gestión que deja activa un tramo/etapa ya ejecutado sobre lo que queda
    (o lo que se acaba de añadir a) la posición: break-even, trailing stop o
    mover el stop al precio de referencia (la etapa/tramo anterior). Devuelve
    (stop_precio, be_aplicado). Comparte lógica entre salidas parciales y
    entrada escalonada: mismo formato de 'gestion' en ambas.

    Ojo: los tipos 1 y 2 escriben en bes_setup/trails_setup, es decir, dejan
    el BE/trailing activo para el setup entero (comportamiento histórico del
    motor, no solo para el trade en curso)."""
    if g_tipo == 1:      # break-even
        bes_setup[setup_in] = g_val if g_val > 0.0 else 0.0
        be_aplicado = False
    elif g_tipo == 2:    # trailing ×ATR
        trails_setup[setup_in] = g_val
    elif g_tipo == 3:    # mover al precio de referencia (etapa/tramo anterior)
        if precio_ref > 0.0:
            if dir_pos > 0:
                stop_precio = max(stop_precio, precio_ref)
            else:
                stop_precio = min(stop_precio, precio_ref)
    return stop_precio, be_aplicado


@njit(nogil=True)
def _simular_numba(o, h, l, c, ent_long, ent_short, sal_long, sal_short,
                   setup_id, atr, riesgos_setup, stops_setup, tps_setup,
                   tiempos_setup, bes_setup, be_uni_setup, trails_setup,
                   parc_pct_setup, parc_r_setup,
                   parc_ml_setup, parc_ms_setup,
                   parc_gt_setup, parc_gv_setup,
                   parc_has_conds, parc_senal_setup,
                   parc_estanc_setup, parc_velas_setup, parc_rmin_setup,
                   n_parc_setup,
                   tramo_pct_setup, tramo_trig_setup, tramo_val_setup,
                   tramo_ml_setup, tramo_ms_setup,
                   tramo_gt_setup, tramo_gv_setup,
                   tramo_has_conds, n_tramos_setup,
                   mec_pct_setup, mec_ml_setup, mec_ms_setup,
                   mec_gt_setup, mec_gv_setup, mec_has_conds,
                   max_trades, max_entradas,
                   capital_inicial, comision_pct, slippage_pct):
    """Bucle del motor. Devuelve (trades[n,12], n_trades, equity[n],
    entradas[n,6], n_entradas, stop_track[n], entrada_track[n],
    unidades_track[n])."""
    n = len(c)
    trades = np.zeros((max_trades, _N_COLS_TRADE))
    n_trades = 0
    entradas = np.zeros((max_entradas, _N_COLS_ENTRADA))
    n_entradas = 0
    equity = np.full(n, capital_inicial)
    # nivel de stop vigente y precio de entrada (medio ponderado) vela a vela,
    # NaN fuera de posición — instrumentación para dibujar la trayectoria real
    # del stop (ver «Mostrar operación» en la GUI), no afecta al resultado.
    stop_track = np.full(n, np.nan)
    entrada_track = np.full(n, np.nan)
    # unidades abiertas al cierre de cada vela (0 fuera de posición). A
    # diferencia de los dos anteriores esto SÍ alimenta una métrica —el capital
    # medio comprometido—, así que lleva 0 y no NaN: una vela sin posición no es
    # un hueco, es exposición cero.
    unidades_track = np.zeros(n)

    cap = capital_inicial
    en_pos = False
    dir_pos = 0
    precio_in = 0.0
    unidades = 0.0
    unidades_iniciales = 0.0
    stop_precio = 0.0
    tp_precio = 0.0
    idx_in = 0
    setup_in = 0
    max_fav = 0.0
    be_aplicado = False
    # quién movió por última vez el nivel de stop vigente: 0 = el stop original
    # de la entrada, 1 = break-even, 2 = trailing. Decide qué 'pct' se aplica
    # cuando el precio lo toca (ver _ORIGEN_A_MECANISMO).
    origen_stop = 0
    # máscara de bits de los mecanismos que ya gastaron su cierre parcial en la
    # posición en curso (ver DISPARO ÚNICO en el docstring del módulo)
    mec_usado = 0
    etapa_actual = 0
    precio_parcial = 0.0
    tramo_actual = 0
    precio_tramo_ant = 0.0
    dist_ref = 0.0
    pendiente_entrada = 0
    pendiente_salida = False
    pendiente_parcial = False
    pendiente_tramo = False
    pendiente_setup = 0

    for i in range(n):
        # se marca True en cualquier cierre COMPLETO ocurrido en esta vela
        # (a diferencia de un parcial, que deja en_pos=True): así, aunque
        # en_pos ya sea False al llegar al final de la iteración, la vela de
        # cierre todavía registra en stop_track/entrada_track el nivel que
        # tenía la posición justo antes de cerrarse — si no, la línea del
        # gráfico se quedaría una vela corta del marcador de cierre.
        cerro_total_en_i = False

        # ── ejecutar al open lo pendiente ──
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
            trades[n_trades, _T_STOP] = stop_precio
            trades[n_trades, _T_PARCIAL] = 0
            n_trades += 1
            en_pos = False
            cerro_total_en_i = True
            pendiente_salida = False
            pendiente_parcial = False
            pendiente_tramo = False

        # ── etapa parcial disparada por la SEÑAL de la plantilla: cierra solo
        # su % al open de esta vela y la posición sigue viva con el resto ──
        if en_pos and pendiente_parcial:
            pct = parc_pct_setup[setup_in, etapa_actual] / 100.0
            u_cerrar = unidades_iniciales * pct
            cierra_todo = u_cerrar >= unidades
            if cierra_todo:
                u_cerrar = unidades
            precio_out = o[i] * (1.0 - slippage_pct * dir_pos)
            pnl_parcial = (precio_out - precio_in) * u_cerrar * dir_pos
            pnl_parcial -= (precio_in + precio_out) * u_cerrar * comision_pct
            cap += pnl_parcial
            trades[n_trades, _T_IDX_IN] = idx_in
            trades[n_trades, _T_IDX_OUT] = i
            trades[n_trades, _T_DIR] = dir_pos
            trades[n_trades, _T_SETUP] = setup_in
            trades[n_trades, _T_PIN] = precio_in
            trades[n_trades, _T_POUT] = precio_out
            trades[n_trades, _T_PNL] = pnl_parcial
            trades[n_trades, _T_MOTIVO] = 5
            trades[n_trades, _T_EQ_IN] = cap - pnl_parcial
            trades[n_trades, _T_UNIDADES] = u_cerrar
            trades[n_trades, _T_STOP] = stop_precio
            trades[n_trades, _T_PARCIAL] = etapa_actual + 1
            n_trades += 1
            if cierra_todo:
                en_pos = False
                cerro_total_en_i = True
            else:
                unidades -= u_cerrar
                stop_precio, be_aplicado = _aplicar_gestion_parcial(
                    parc_gt_setup[setup_in, etapa_actual],
                    parc_gv_setup[setup_in, etapa_actual],
                    dir_pos, stop_precio, precio_parcial,
                    setup_in, bes_setup, trails_setup, be_aplicado)
                precio_parcial = o[i]
            etapa_actual += 1
            pendiente_parcial = False

        # ── tramo de entrada escalonada: añade unidades a la posición ya
        # abierta al open de esta vela, recalculando el precio de entrada
        # como media ponderada de todo lo construido hasta ahora. A
        # diferencia de las salidas por R:R (que simulan un objetivo tocado
        # intra-vela), toda entrada de capital nuevo se ejecuta al open,
        # igual que la apertura inicial de la posición — nunca a un precio
        # sintético intra-vela. ──
        if en_pos and pendiente_tramo:
            precio_t = o[i] * (1.0 + slippage_pct * dir_pos)
            ref_atr_t = atr[i] if atr[i] > 0 else o[i] * 0.01
            if stop_precio > 0.0:
                dist_t = abs(precio_t - stop_precio)
            else:
                dist_t = dist_ref
            dist_t = max(dist_t, 0.25 * dist_ref)   # nunca por debajo del 25% de la 1ª distancia
            pct_t = tramo_pct_setup[setup_in, tramo_actual] / 100.0
            riesgo_t = riesgos_setup[setup_in]
            u_tramo = (cap * riesgo_t * pct_t) / dist_t if dist_t > 0.0 else 0.0
            if u_tramo > 0.0 and cap > 0.0:
                precio_in = (precio_in * unidades + precio_t * u_tramo) / (unidades + u_tramo)
                unidades += u_tramo
                unidades_iniciales += u_tramo
                # recalcular el take-profit desde el nuevo precio medio, con
                # la misma distancia de referencia (stop_atr o 2×ATR) que usa
                # el resto del motor para expresar "R"
                tp_r_s = tps_setup[setup_in]
                if tp_r_s > 0.0:
                    dist_tp = stops_setup[setup_in] * ref_atr_t \
                        if stops_setup[setup_in] > 0.0 else 2.0 * ref_atr_t
                    tp_precio = precio_in + tp_r_s * dist_tp * dir_pos
                stop_precio, be_aplicado = _aplicar_gestion_parcial(
                    tramo_gt_setup[setup_in, tramo_actual],
                    tramo_gv_setup[setup_in, tramo_actual],
                    dir_pos, stop_precio, precio_tramo_ant,
                    setup_in, bes_setup, trails_setup, be_aplicado)
                precio_tramo_ant = precio_t
                entradas[n_entradas, _E_IDX] = i
                entradas[n_entradas, _E_DIR] = dir_pos
                entradas[n_entradas, _E_SETUP] = setup_in
                entradas[n_entradas, _E_PRECIO] = precio_t
                entradas[n_entradas, _E_UNIDADES] = u_tramo
                entradas[n_entradas, _E_TRAMO] = tramo_actual
                n_entradas += 1
            tramo_actual += 1
            pendiente_tramo = False

        if (not en_pos) and pendiente_entrada != 0:
            d = pendiente_entrada
            precio = o[i] * (1.0 + slippage_pct * d)
            ref_atr = atr[i] if atr[i] > 0 else precio * 0.01
            stop_atr_s = stops_setup[pendiente_setup]
            tp_r_s = tps_setup[pendiente_setup]
            dist = stop_atr_s * ref_atr if stop_atr_s > 0 else 2.0 * ref_atr
            riesgo = riesgos_setup[pendiente_setup]
            n_tram_0 = n_tramos_setup[pendiente_setup]
            # el 1er tramo se lleva solo SU % del riesgo total (100% si no
            # hay entrada escalonada configurada, ver simular())
            pct0 = tramo_pct_setup[pendiente_setup, 0] / 100.0 if n_tram_0 > 0 else 1.0
            if dist > 0 and riesgo > 0 and cap > 0 and pct0 > 0.0:
                unidades = (cap * riesgo * pct0) / dist
                unidades_iniciales = unidades
                en_pos = True
                dir_pos = d
                precio_in = precio
                idx_in = i
                setup_in = pendiente_setup
                stop_precio = precio - dist * d if stop_atr_s > 0 else 0.0
                tp_precio = precio + tp_r_s * dist * d if tp_r_s > 0 else 0.0
                max_fav = precio
                be_aplicado = False
                origen_stop = 0
                mec_usado = 0
                etapa_actual = 0
                precio_parcial = 0.0
                pendiente_parcial = False
                tramo_actual = 1
                pendiente_tramo = False
                precio_tramo_ant = precio
                dist_ref = dist
                entradas[n_entradas, _E_IDX] = i
                entradas[n_entradas, _E_DIR] = dir_pos
                entradas[n_entradas, _E_SETUP] = setup_in
                entradas[n_entradas, _E_PRECIO] = precio
                entradas[n_entradas, _E_UNIDADES] = unidades
                entradas[n_entradas, _E_TRAMO] = 0
                n_entradas += 1
            pendiente_entrada = 0

        # ── gestión intra-vela de la posición abierta ──
        if en_pos:
            ref_atr_pos = atr[i] if atr[i] > 0 else o[i] * 0.01
            dist_pos = 2.0 * ref_atr_pos  # fallback si stop=0
            if stops_setup[setup_in] > 0.0:
                dist_pos = stops_setup[setup_in] * ref_atr_pos

            # ── break-even + trailing stop ──
            be_atr_s = bes_setup[setup_in]
            trail_atr_s = trails_setup[setup_in]
            # unidad de la distancia de activación del break-even: 0 = ×ATR
            # (histórico), 1 = R (múltiplos de la distancia de riesgo real del
            # setup). Solo coinciden si el stop está a 1×ATR — con stop a 2×ATR,
            # «1×ATR de avance» es 0.5R, no 1R.
            ref_be = dist_pos if be_uni_setup[setup_in] == 1 else ref_atr_pos
            # origen_stop se actualiza solo cuando el mecanismo MUEVE de hecho
            # el nivel (se compara el valor antes/después, sin tocar la
            # aritmética original): así refleja quién puso el stop donde está
            # ahora, que es lo que decide el 'pct' aplicable si el precio lo
            # toca. Ojo: con stop_precio == 0 (sin stop) los max/min de abajo
            # dejan el nivel en 0 para cortos, igual que siempre — el origen
            # tampoco cambia, porque nada se movió.
            if dir_pos > 0:
                if h[i] > max_fav:
                    max_fav = h[i]
                if not be_aplicado and be_atr_s > 0.0 and max_fav - precio_in >= be_atr_s * ref_be:
                    stop_previo = stop_precio
                    stop_precio = max(stop_precio, precio_in)
                    if stop_precio != stop_previo:
                        origen_stop = 1
                    be_aplicado = True
                if trail_atr_s > 0.0:
                    stop_previo = stop_precio
                    stop_precio = max(stop_precio, max_fav - trail_atr_s * ref_atr_pos)
                    if stop_precio != stop_previo:
                        origen_stop = 2
            else:
                if l[i] < max_fav:
                    max_fav = l[i]
                if not be_aplicado and be_atr_s > 0.0 and precio_in - max_fav >= be_atr_s * ref_be:
                    stop_previo = stop_precio
                    stop_precio = min(stop_precio, precio_in)
                    if stop_precio != stop_previo:
                        origen_stop = 1
                    be_aplicado = True
                if trail_atr_s > 0.0:
                    stop_previo = stop_precio
                    stop_precio = min(stop_precio, max_fav + trail_atr_s * ref_atr_pos)
                    if stop_precio != stop_previo:
                        origen_stop = 2

            # ── salidas parciales ──
            n_parc = n_parc_setup[setup_in]
            if etapa_actual < n_parc:
                dispara = False
                r_trig = parc_r_setup[setup_in, etapa_actual]
                if parc_estanc_setup[setup_in, etapa_actual]:
                    # Estancamiento: N velas desde la entrada sin alcanzar un
                    # R mínimo a favor (comprobado ANTES que r_trig>0 porque
                    # esta etapa deja parc_r_setup a 0 a propósito — no debe
                    # colarse por la rama de R:R normal).
                    velas_transcurridas = i - idx_in
                    if velas_transcurridas >= parc_velas_setup[setup_in, etapa_actual]:
                        avance_r = ((max_fav - precio_in) if dir_pos > 0
                                   else (precio_in - max_fav)) / dist_pos
                        if avance_r < parc_rmin_setup[setup_in, etapa_actual]:
                            dispara = True
                elif r_trig > 0.0:
                    if dir_pos > 0:
                        if h[i] >= precio_in + r_trig * dist_pos:
                            dispara = True
                    else:
                        if l[i] <= precio_in - r_trig * dist_pos:
                            dispara = True
                elif parc_has_conds[setup_in, etapa_actual] \
                        and not parc_senal_setup[setup_in, etapa_actual]:
                    # Sin R:R — dispara solo si se cumplen las condiciones
                    dispara = True
                # comprobar máscara de condiciones
                if dispara:
                    if dir_pos > 0:
                        if not parc_ml_setup[setup_in, etapa_actual, i]:
                            dispara = False
                    else:
                        if not parc_ms_setup[setup_in, etapa_actual, i]:
                            dispara = False

                if dispara:
                    pct = parc_pct_setup[setup_in, etapa_actual] / 100.0
                    u_cerrar = unidades_iniciales * pct
                    cierra_todo = u_cerrar >= unidades
                    if cierra_todo:
                        u_cerrar = unidades   # no queda más que esto: cierra el resto
                    precio_temp = max(precio_in, precio_in + r_trig * dist_pos * dir_pos) \
                        if dir_pos > 0 else min(precio_in, precio_in + r_trig * dist_pos * dir_pos)
                    if r_trig <= 0.0:
                        precio_temp = c[i]
                    precio_out = precio_temp * (1.0 - slippage_pct * dir_pos)
                    pnl_parcial = (precio_out - precio_in) * u_cerrar * dir_pos
                    pnl_parcial -= (precio_in + precio_out) * u_cerrar * comision_pct
                    cap += pnl_parcial
                    trades[n_trades, _T_IDX_IN] = idx_in
                    trades[n_trades, _T_IDX_OUT] = i
                    trades[n_trades, _T_DIR] = dir_pos
                    trades[n_trades, _T_SETUP] = setup_in
                    trades[n_trades, _T_PIN] = precio_in
                    trades[n_trades, _T_POUT] = precio_out
                    trades[n_trades, _T_PNL] = pnl_parcial
                    trades[n_trades, _T_MOTIVO] = 5
                    trades[n_trades, _T_EQ_IN] = cap - pnl_parcial
                    trades[n_trades, _T_UNIDADES] = u_cerrar
                    trades[n_trades, _T_STOP] = stop_precio
                    trades[n_trades, _T_PARCIAL] = etapa_actual + 1
                    n_trades += 1

                    if cierra_todo:
                        en_pos = False
                        cerro_total_en_i = True
                        pendiente_salida = False
                    else:
                        unidades -= u_cerrar
                        stop_precio, be_aplicado = _aplicar_gestion_parcial(
                            parc_gt_setup[setup_in, etapa_actual],
                            parc_gv_setup[setup_in, etapa_actual],
                            dir_pos, stop_precio, precio_parcial,
                            setup_in, bes_setup, trails_setup, be_aplicado)
                        precio_parcial = precio_temp
                    etapa_actual += 1

            # ── entrada escalonada: ¿toca el siguiente tramo? (triggers
            # velas/retroceso/avance/cond; 'senal' se resuelve más abajo,
            # junto a la lectura de señales) ──
            if en_pos:
                n_tram = n_tramos_setup[setup_in]
                if tramo_actual < n_tram:
                    trig_t = tramo_trig_setup[setup_in, tramo_actual]
                    val_t = tramo_val_setup[setup_in, tramo_actual]
                    dispara_t = False
                    if trig_t == 1:      # a +N velas de la 1ª entrada
                        if (i - idx_in) >= int(val_t):
                            dispara_t = True
                    elif trig_t == 2:    # retroceso adverso de N×ATR (promediar)
                        if dir_pos > 0:
                            if l[i] <= precio_in - val_t * ref_atr_pos:
                                dispara_t = True
                        else:
                            if h[i] >= precio_in + val_t * ref_atr_pos:
                                dispara_t = True
                    elif trig_t == 3:    # avance a favor de +N R (pirámide)
                        if dir_pos > 0:
                            if h[i] >= precio_in + val_t * dist_pos:
                                dispara_t = True
                        else:
                            if l[i] <= precio_in - val_t * dist_pos:
                                dispara_t = True
                    elif trig_t == 4:    # solo condiciones
                        dispara_t = tramo_has_conds[setup_in, tramo_actual]
                    # trig_t == 0 (señal de la plantilla) se resuelve al leer
                    # las señales de esta misma vela, más abajo
                    if dispara_t and trig_t != 0:
                        if dir_pos > 0:
                            if not tramo_ml_setup[setup_in, tramo_actual, i]:
                                dispara_t = False
                        else:
                            if not tramo_ms_setup[setup_in, tramo_actual, i]:
                                dispara_t = False
                        if dispara_t:
                            pendiente_tramo = True

            # ── stop / TP / tiempo / fin de datos ──
            salida_precio = 0.0
            motivo = -1
            if stop_precio > 0.0:
                if dir_pos > 0 and l[i] <= stop_precio:
                    salida_precio = stop_precio
                    motivo = 1
                elif dir_pos < 0 and h[i] >= stop_precio:
                    salida_precio = stop_precio
                    motivo = 1
            if motivo < 0 and tp_precio > 0.0:
                if dir_pos > 0 and h[i] >= tp_precio:
                    salida_precio = tp_precio
                    motivo = 2
                elif dir_pos < 0 and l[i] <= tp_precio:
                    salida_precio = tp_precio
                    motivo = 2
            n_velas_s = tiempos_setup[setup_in]
            if motivo < 0 and n_velas_s > 0 and (i - idx_in) >= n_velas_s \
                    and (mec_usado & (1 << _M_TIEMPO)) == 0:
                # Si la salida por tiempo ya cerró su parte, queda AGOTADA para
                # esta posición: a diferencia del stop/TP —redes de seguridad
                # que deben seguir cortando— su condición es cierta en todas
                # las velas siguientes, así que volver a dispararla cerraría el
                # resto una vela después y el parcial no serviría de nada.
                salida_precio = c[i]
                motivo = 3
            if motivo < 0 and i == n - 1:
                salida_precio = c[i]
                motivo = 4

            # ¿este cierre lo gestiona un mecanismo con 'pct' propio? El stop
            # (según quién movió el nivel: original/BE/trailing), el TP y la
            # salida por tiempo. Fin de datos cierra siempre el 100%.
            mec = -1
            if motivo == 1:
                mec = _ORIGEN_A_MECANISMO[origen_stop]
            elif motivo == 2:
                mec = _M_TP
            elif motivo == 3:
                mec = _M_TIEMPO
            u_mec = unidades
            parcial_mec = 0
            # disparo único: si este mecanismo ya gastó su parcial en esta
            # posición, vuelve a cerrar el 100% (ver docstring del módulo)
            if mec >= 0 and (mec_usado & (1 << mec)) != 0:
                mec = -1
            if mec >= 0:
                # las condiciones del mecanismo, si las tiene, acotan cuándo
                # puede cerrar SOLO SU PARTE: si no se cumplen, el cierre sigue
                # siendo total (la red de seguridad nunca se desactiva)
                aplica_pct = True
                if mec_has_conds[setup_in, mec]:
                    if dir_pos > 0:
                        aplica_pct = mec_ml_setup[setup_in, mec, i]
                    else:
                        aplica_pct = mec_ms_setup[setup_in, mec, i]
                if aplica_pct:
                    pct_mec = mec_pct_setup[setup_in, mec] / 100.0
                    if pct_mec < 1.0:
                        u_mec = unidades * pct_mec
                        if u_mec < unidades - 1e-12:
                            parcial_mec = -(mec + 1)
                        else:
                            u_mec = unidades

            if motivo >= 0:
                precio_out = salida_precio * (1.0 - slippage_pct * dir_pos)
                pnl = (precio_out - precio_in) * u_mec * dir_pos
                pnl -= (precio_in + precio_out) * u_mec * comision_pct
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
                trades[n_trades, _T_UNIDADES] = u_mec
                trades[n_trades, _T_STOP] = stop_precio
                trades[n_trades, _T_PARCIAL] = parcial_mec
                n_trades += 1
                if parcial_mec < 0:
                    # cierre parcial: la posición sigue viva con el resto. Este
                    # mecanismo queda gastado para lo que reste de la posición:
                    # si vuelve a dispararse, cerrará todo (disparo único).
                    mec_usado |= (1 << mec)
                    unidades -= u_mec
                    stop_precio, be_aplicado = _aplicar_gestion_parcial(
                        mec_gt_setup[setup_in, mec], mec_gv_setup[setup_in, mec],
                        dir_pos, stop_precio, precio_parcial,
                        setup_in, bes_setup, trails_setup, be_aplicado)
                    precio_parcial = salida_precio
                else:
                    en_pos = False
                    cerro_total_en_i = True
                    pendiente_salida = False
                    pendiente_parcial = False
                    pendiente_tramo = False

        # ── leer señales de la vela i (se ejecutarán al open de i+1) ──
        if en_pos:
            bit = np.int64(1) << np.int64(setup_in)
            senal_salida = (dir_pos > 0 and (sal_long[i] & bit) != 0) \
                or (dir_pos < 0 and (sal_short[i] & bit) != 0)
            # señal contraria del MISMO setup también cierra (revierte);
            # la de otro setup no toca esta posición
            if setup_id[i] == setup_in and (
                    (dir_pos > 0 and ent_short[i]) or (dir_pos < 0 and ent_long[i])):
                senal_salida = True
            if senal_salida:
                # Si la etapa parcial en curso se dispara por señal Y deja
                # posición viva, esta señal cierra solo su % (parcial). Si
                # cerraría el 100% —el caso por defecto: una sola etapa al
                # 100%— se trata como cierre completo, idéntico al motor
                # previo a las etapas por señal: así el trade se registra con
                # motivo 0 y la señal contraria puede revertir la posición.
                usa_parcial = False
                if etapa_actual < n_parc_setup[setup_in] \
                        and parc_senal_setup[setup_in, etapa_actual] \
                        and unidades_iniciales * (parc_pct_setup[setup_in, etapa_actual]
                                                  / 100.0) < unidades - 1e-12:
                    # las condiciones extra de la etapa, si las hay, acotan
                    # cuándo puede ejecutarse el parcial por señal
                    if not parc_has_conds[setup_in, etapa_actual]:
                        usa_parcial = True
                    elif dir_pos > 0:
                        usa_parcial = parc_ml_setup[setup_in, etapa_actual, i]
                    else:
                        usa_parcial = parc_ms_setup[setup_in, etapa_actual, i]
                if usa_parcial:
                    pendiente_parcial = True
                else:
                    pendiente_salida = True

            # tramo por SEÑAL: la propia señal de entrada de la plantilla
            # (mismo lado, mismo setup) se repite mientras la posición sigue
            # abierta — p.ej. RSI vuelve a marcar sobreventa, o el patrón
            # vuelve a formarse.
            n_tram = n_tramos_setup[setup_in]
            if tramo_actual < n_tram and tramo_trig_setup[setup_in, tramo_actual] == 0 \
                    and setup_id[i] == setup_in \
                    and ((dir_pos > 0 and ent_long[i]) or (dir_pos < 0 and ent_short[i])):
                dispara_t = True
                if tramo_has_conds[setup_in, tramo_actual]:
                    dispara_t = tramo_ml_setup[setup_in, tramo_actual, i] if dir_pos > 0 \
                        else tramo_ms_setup[setup_in, tramo_actual, i]
                if dispara_t:
                    pendiente_tramo = True

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
            # 'unidades' ya refleja los tramos añadidos y las parciales cerradas
            # en esta vela, así que es el tamaño realmente vivo al cierre
            unidades_track[i] = unidades
        else:
            equity[i] = cap

        # ── trayectoria del stop / precio medio, para el gráfico ──
        if en_pos or cerro_total_en_i:
            entrada_track[i] = precio_in
            if stop_precio > 0.0:
                stop_track[i] = stop_precio

    return (trades[:n_trades], n_trades, equity, entradas[:n_entradas],
            n_entradas, stop_track, entrada_track, unidades_track)


def simular(o, h, l, c, senales, config):
    """Corre el motor y devuelve un dict con trades (dict de arrays),
    entradas (dict de arrays: aperturas y tramos de entrada escalonada),
    equity, drawdown y capital final.

    senales: dict de core/strategies.generar_senales o
    generar_senales_sistema — claves entradas_long/entradas_short (bool[n]),
    salidas_long/salidas_short (bool[n] o bitmask int64[n]: bool True
    equivale a "todos los setups pueden salir"), setup_id (int64[n], 0 por
    defecto, máx. 63) y atr (float64[n]).
    config: dict — capital_inicial, riesgo_pct, comision_pct, slippage_pct,
    stop_atr, tp_r, salida_n_velas (defectos globales) y opcionalmente
    config_por_setup: {id: {'riesgo_pct','stop_atr','tp_r','salida_n_velas',
    'be_atr','be_unidad','trailing_atr','parciales','tramos'}}
    para que cada setup del sistema tenga su propio riesgo/stop/TP/tiempo,
    gestión de posición (break-even y trailing stop, 0 = off; el trailing
    siempre en ×ATR y el break-even en la unidad que diga 'be_unidad':
    'atr' (defecto) o 'r' = múltiplos de la distancia de riesgo real del
    setup, que solo coinciden si el stop está a 1×ATR),
    salidas parciales y entrada escalonada (ver docstring del módulo)
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
    bes = np.full(64, 0.0)
    # unidad de la distancia de activación del break-even: 0 = ×ATR (defecto,
    # comportamiento histórico), 1 = R
    be_unis = np.zeros(64, dtype=np.int64)
    trails = np.full(64, 0.0)
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
        if 'be_atr' in cfg_s:
            bes[sid] = float(cfg_s['be_atr'])
        if cfg_s.get('be_unidad') == 'r':
            be_unis[sid] = 1
        if 'trailing_atr' in cfg_s:
            trails[sid] = float(cfg_s['trailing_atr'])

    # ── salidas parciales ──
    max_parc = 8
    parc_pct_all = np.zeros((64, max_parc), dtype=np.float64)
    parc_r_all = np.zeros((64, max_parc), dtype=np.float64)
    parc_gt_all = np.zeros((64, max_parc), dtype=np.int64)
    parc_gv_all = np.zeros((64, max_parc), dtype=np.float64)
    parc_has_conds = np.zeros((64, max_parc), dtype=np.bool_)
    parc_senal_all = np.zeros((64, max_parc), dtype=np.bool_)
    # estancamiento: N velas desde la entrada sin alcanzar un R mínimo a favor
    parc_estanc_all = np.zeros((64, max_parc), dtype=np.bool_)
    parc_velas_all = np.zeros((64, max_parc), dtype=np.int64)
    parc_rmin_all = np.zeros((64, max_parc), dtype=np.float64)
    n_parc_all = np.zeros(64, dtype=np.int64)

    for sid, cfg_s in (config.get('config_por_setup') or {}).items():
        sid = int(sid)
        if not 0 <= sid < 64:
            continue
        parciales = cfg_s.get('parciales')
        if parciales and len(parciales) > 0:
            n_parc = min(len(parciales), max_parc)
            n_parc_all[sid] = n_parc
            for e in range(n_parc):
                ep = parciales[e]
                parc_pct_all[sid, e] = float(ep.get('pct', 100.0))
                r = float(ep.get('r', 0.0))
                parc_r_all[sid, e] = r
                g = ep.get('gestion', {})
                parc_gt_all[sid, e] = int(g.get('tipo', 0))
                parc_gv_all[sid, e] = float(g.get('val', 0.0))
                conds = ep.get('condiciones')
                tiene_conds = bool(conds and len(conds) > 0)
                if tiene_conds:
                    parc_has_conds[sid, e] = True
                # 'trigger' ausente = formato antiguo: sin R:R y sin
                # condiciones la etapa no tenía forma de dispararse (era una
                # etapa muerta), así que se interpreta como lo que su rótulo
                # ya prometía, "a la señal de la plantilla".
                trigger = ep.get('trigger')
                if trigger is None:
                    trigger = 'senal' if (r <= 0.0 and not tiene_conds) else \
                              ('r' if r > 0.0 else 'cond')
                if trigger == 'senal':
                    parc_senal_all[sid, e] = True
                    parc_r_all[sid, e] = 0.0   # una etapa por señal no usa R:R
                elif trigger == 'estancamiento':
                    parc_estanc_all[sid, e] = True
                    parc_velas_all[sid, e] = int(ep.get('velas_max', 0))
                    parc_rmin_all[sid, e] = float(ep.get('r_min', 0.0))

    # condition masks: 3D (64 setups × 8 stages × n bars). Padded with True.
    parc_ml = np.ones((64, max_parc, n), dtype=np.bool_)
    parc_ms = np.ones((64, max_parc, n), dtype=np.bool_)
    masks_in = config.get('parciales_masks_long')
    if masks_in and len(masks_in) > 0:
        for sid in range(min(len(masks_in), 64)):
            m = masks_in[sid]
            if m is not None and len(m) > 0:
                m = np.array(m, dtype=np.bool_)
                if m.ndim == 1:
                    m = m.reshape(1, -1)
                stages = min(m.shape[0], max_parc)
                parc_ml[sid, :stages, :] = m[:stages, :]
    masks_in = config.get('parciales_masks_short')
    if masks_in and len(masks_in) > 0:
        for sid in range(min(len(masks_in), 64)):
            m = masks_in[sid]
            if m is not None and len(m) > 0:
                m = np.array(m, dtype=np.bool_)
                if m.ndim == 1:
                    m = m.reshape(1, -1)
                stages = min(m.shape[0], max_parc)
                parc_ms[sid, :stages, :] = m[:stages, :]

    # ── entrada escalonada (tramos) ──
    # por defecto un solo tramo al 100% a la señal: idéntico a no tener
    # entrada escalonada (ver docstring del módulo)
    max_tram = 8
    tramo_pct_all = np.full((64, max_tram), 100.0, dtype=np.float64)
    tramo_trig_all = np.zeros((64, max_tram), dtype=np.int64)   # 0 = senal
    tramo_val_all = np.zeros((64, max_tram), dtype=np.float64)
    tramo_gt_all = np.zeros((64, max_tram), dtype=np.int64)
    tramo_gv_all = np.zeros((64, max_tram), dtype=np.float64)
    tramo_has_conds = np.zeros((64, max_tram), dtype=np.bool_)
    n_tramos_all = np.ones(64, dtype=np.int64)

    for sid, cfg_s in (config.get('config_por_setup') or {}).items():
        sid = int(sid)
        if not 0 <= sid < 64:
            continue
        tramos = cfg_s.get('tramos')
        if tramos and len(tramos) > 0:
            n_t = min(len(tramos), max_tram)
            n_tramos_all[sid] = n_t
            for e in range(n_t):
                tr = tramos[e]
                tramo_pct_all[sid, e] = float(tr.get('pct', 100.0))
                tramo_trig_all[sid, e] = TRIGGERS_TRAMO_ENTRADA.get(
                    tr.get('trigger', 'senal'), 0)
                tramo_val_all[sid, e] = float(tr.get('val', 0.0))
                g = tr.get('gestion', {})
                tramo_gt_all[sid, e] = int(g.get('tipo', 0))
                tramo_gv_all[sid, e] = float(g.get('val', 0.0))
                conds = tr.get('condiciones')
                tramo_has_conds[sid, e] = bool(conds and len(conds) > 0)

    tramo_ml = np.ones((64, max_tram, n), dtype=np.bool_)
    tramo_ms = np.ones((64, max_tram, n), dtype=np.bool_)
    masks_in = config.get('tramos_masks_long')
    if masks_in and len(masks_in) > 0:
        for sid in range(min(len(masks_in), 64)):
            m = masks_in[sid]
            if m is not None and len(m) > 0:
                m = np.array(m, dtype=np.bool_)
                if m.ndim == 1:
                    m = m.reshape(1, -1)
                stages = min(m.shape[0], max_tram)
                tramo_ml[sid, :stages, :] = m[:stages, :]
    masks_in = config.get('tramos_masks_short')
    if masks_in and len(masks_in) > 0:
        for sid in range(min(len(masks_in), 64)):
            m = masks_in[sid]
            if m is not None and len(m) > 0:
                m = np.array(m, dtype=np.bool_)
                if m.ndim == 1:
                    m = m.reshape(1, -1)
                stages = min(m.shape[0], max_tram)
                tramo_ms[sid, :stages, :] = m[:stages, :]

    # ── cierre parcial por mecanismo (stop / TP / break-even / trailing) ──
    # por defecto pct=100: cierran toda la posición, idéntico a no configurarlos
    n_mec = len(MECANISMOS_SALIDA)
    mec_pct_all = np.full((64, n_mec), 100.0, dtype=np.float64)
    mec_gt_all = np.zeros((64, n_mec), dtype=np.int64)
    mec_gv_all = np.zeros((64, n_mec), dtype=np.float64)
    mec_has_conds = np.zeros((64, n_mec), dtype=np.bool_)

    for sid, cfg_s in (config.get('config_por_setup') or {}).items():
        sid = int(sid)
        if not 0 <= sid < 64:
            continue
        for k, clave in enumerate(MECANISMOS_SALIDA):
            m_cfg = cfg_s.get(clave)
            if not m_cfg:
                continue
            mec_pct_all[sid, k] = float(m_cfg.get('pct', 100.0))
            g = m_cfg.get('gestion', {})
            mec_gt_all[sid, k] = int(g.get('tipo', 0))
            mec_gv_all[sid, k] = float(g.get('val', 0.0))
            conds = m_cfg.get('condiciones')
            mec_has_conds[sid, k] = bool(conds and len(conds) > 0)

    mec_ml = np.ones((64, n_mec, n), dtype=np.bool_)
    mec_ms = np.ones((64, n_mec, n), dtype=np.bool_)
    for clave_cfg, destino in (('mecanismos_masks_long', mec_ml),
                               ('mecanismos_masks_short', mec_ms)):
        masks_in = config.get(clave_cfg)
        if masks_in and len(masks_in) > 0:
            for sid in range(min(len(masks_in), 64)):
                m = masks_in[sid]
                if m is not None and len(m) > 0:
                    m = np.array(m, dtype=np.bool_)
                    if m.ndim == 1:
                        m = m.reshape(1, -1)
                    stages = min(m.shape[0], n_mec)
                    destino[sid, :stages, :] = m[:stages, :]

    # Buffer de trades: una fila por cierre. Cierres completos ≤ una posición
    # cada 2 velas, más una fila por etapa parcial ejecutada. El techo duro es
    # 3 filas por vela (parcial al open + parcial intra-vela + cierre por
    # stop/TP/tiempo), así que se recorta ahí para no reservar de más cuando
    # hay muchas etapas configuradas.
    # Con disparo único, cada mecanismo global aporta como mucho una fila
    # parcial por posición: esa es la cota real del término extra.
    mec_parcial = bool((mec_pct_all < 100.0).any())
    max_por_posicion = (1 + int(n_parc_all.max())
                        + (len(MECANISMOS_SALIDA) if mec_parcial else 0))
    max_trades = min((n // 2 + 1) * max_por_posicion + 1, 3 * n + 8)
    # Buffer de entradas: la apertura inicial + hasta (n_tramos-1) tramos por
    # posición, mismo techo que max_trades.
    max_por_posicion_e = int(n_tramos_all.max())
    max_entradas = min((n // 2 + 1) * max_por_posicion_e + 1, 3 * n + 8)

    (trades_arr, n_trades, equity, entradas_arr, n_entradas,
     stop_track, entrada_track, unidades_track) = _simular_numba(
        o, h, l, c,
        np.ascontiguousarray(senales['entradas_long'], dtype=np.bool_),
        np.ascontiguousarray(senales['entradas_short'], dtype=np.bool_),
        _mascara_salida(senales['salidas_long']),
        _mascara_salida(senales['salidas_short']),
        setup, atr, riesgos, stops, tps, tiempos, bes, be_unis, trails,
        parc_pct_all, parc_r_all, parc_ml, parc_ms, parc_gt_all, parc_gv_all,
        parc_has_conds, parc_senal_all,
        parc_estanc_all, parc_velas_all, parc_rmin_all, n_parc_all,
        tramo_pct_all, tramo_trig_all, tramo_val_all, tramo_ml, tramo_ms,
        tramo_gt_all, tramo_gv_all, tramo_has_conds, n_tramos_all,
        mec_pct_all, mec_ml, mec_ms, mec_gt_all, mec_gv_all, mec_has_conds,
        max_trades, max_entradas,
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
        'precio_stop': t[:, _T_STOP],   # nivel de stop-loss del trade (0 = sin stop)
        'parcial': t[:, _T_PARCIAL].astype(np.int64),   # 0=completa, 1+=etapa parcial
    }

    # entradas: apertura inicial (tramo=0) + cada tramo de entrada escalonada
    # ejecutado (tramo=1,2,...). NO son cierres: no tocar win rate/expectancy.
    ea = entradas_arr
    entradas = {
        'idx': ea[:, _E_IDX].astype(np.int64),
        'dir': ea[:, _E_DIR].astype(np.int8),
        'setup': ea[:, _E_SETUP].astype(np.int64),
        'precio': ea[:, _E_PRECIO],
        'unidades': ea[:, _E_UNIDADES],
        'tramo': ea[:, _E_TRAMO].astype(np.int64),
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

    # MFE/MAE: excursión favorable/adversa máxima durante la vida del trade,
    # en múltiplos de R (mismo denominador que r_multiple).
    #
    # La vela de ENTRADA cuenta entera: se entra al open, así que todo su rango
    # se recorrió con la posición abierta. La de SALIDA solo cuenta entera si el
    # trade siguió vivo hasta su cierre (motivo 3 = salida por tiempo, 4 = fin
    # de datos). En cualquier otra salida —al open por señal, o intra-vela por
    # stop/TP/parcial— la posición ya estaba cerrada durante parte de esa vela,
    # y contar su rango completo atribuiría al trade excursiones que nunca
    # fueron capturables: inflaría el ETD con beneficio que no llegó a existir y
    # hundiría la eficiencia de salida (un TP clavado en una vela que lo
    # desborda marcaría muy por debajo del 100% que merece).
    #
    # Los precios de entrada y salida se añaden siempre como candidatos: ambos
    # se tocaron con la posición abierta, por definición. Incluir el de entrada
    # además garantiza mfe/mae >= 0 aunque el slippage deje precio_entrada fuera
    # del rango high/low de su vela.
    mfe_precio = np.zeros(n_trades)
    mae_precio = np.zeros(n_trades)
    salida_precio_rel = np.zeros(n_trades)   # distancia del extremo adverso a la salida
    for i in range(n_trades):
        i0, i1 = trades['idx_entrada'][i], trades['idx_salida'][i]
        pin = trades['precio_entrada'][i]
        pout = trades['precio_salida'][i]
        motivo_i = trades['motivo'][i]
        cierra_al_close = (motivo_i == 3 or motivo_i == 4)
        # max(i0 + 1, ...): la vela de entrada entra SIEMPRE, aunque el trade
        # se cierre en ella misma (i0 == i1, típico de un stop tocado el mismo
        # día que se abre). Descontarla dejaría el trade sin rango alguno y con
        # una eficiencia de entrada degenerada al 0%.
        fin = max(i0 + 1, i1 + 1 if cierra_al_close else i1)
        seg_h = h[i0:fin]
        seg_l = l[i0:fin]
        alto = max(seg_h.max(), pin, pout)
        bajo = min(seg_l.min(), pin, pout)
        if trades['dir'][i] > 0:
            mfe_precio[i] = alto - pin
            mae_precio[i] = pin - bajo
            salida_precio_rel[i] = pout - bajo
        else:
            mfe_precio[i] = pin - bajo
            mae_precio[i] = alto - pin
            salida_precio_rel[i] = alto - pout

    with np.errstate(divide='ignore', invalid='ignore'):
        trades['mfe_r'] = np.where(riesgo_absoluto > 0,
            mfe_precio * trades['unidades'] / riesgo_absoluto, 0.0)
        trades['mae_r'] = np.where(riesgo_absoluto > 0,
            mae_precio * trades['unidades'] / riesgo_absoluto, 0.0)

    # ETD (End Trade Drawdown): cuánto beneficio no realizado se devolvió antes
    # de cerrar, en R. Diferencia absoluta, no cociente: es la forma estable de
    # expresar "lo que dejé en la mesa" (ver más abajo por qué el cociente no lo
    # es). Nunca negativo, porque el precio de salida entra en el cálculo del MFE.
    trades['etd_r'] = trades['mfe_r'] - trades['r_multiple']

    # Eficiencias de entrada y salida (Sweeney): posición relativa del precio de
    # entrada y del de salida dentro del rango [mín, máx] que el trade llegó a
    # recorrer. Ambas comparten denominador —el rango completo— y por eso caen
    # siempre en [0, 100]: entrada y salida son, por definición, precios que se
    # tocaron con la posición abierta.
    #
    #   entrada 100% = se compró justo en el mínimo (o se vendió en el máximo)
    #   salida  100% = se cerró justo en la cima del recorrido
    #
    # La salida NO se normaliza por el MFE (r_multiple / mfe_r). Ese cociente
    # parece medir lo mismo, pero su denominador tiende a cero justo en los
    # trades que peor van: un perdedor que avanzó 0.01 R a favor y luego se paró
    # en el stop daba −8600 %. En un sistema con stop y TP eso afectaba a ~3 de
    # cada 4 trades y hundía la media a valores sin sentido, mientras el
    # histograma —acotado a 0-100— solo dibujaba el cuarto restante. El "% del
    # MFE capturado" sigue disponible y estable a través de etd_r.
    rango_precio = mfe_precio + mae_precio
    denom_r = trades['mfe_r'] + trades['mae_r']
    with np.errstate(divide='ignore', invalid='ignore'):
        trades['eficiencia_entrada'] = np.where(
            denom_r > 0,
            np.clip(trades['mfe_r'] / denom_r * 100.0, 0.0, 100.0), np.nan)
        trades['eficiencia_salida'] = np.where(
            rango_precio > 0,
            np.clip(salida_precio_rel / rango_precio * 100.0, 0.0, 100.0), np.nan)

    eq_max = np.maximum.accumulate(equity)
    with np.errstate(divide='ignore', invalid='ignore'):
        drawdown = np.where(eq_max > 0, equity / eq_max - 1.0, 0.0)

    # Fracción del capital comprometida en el mercado al cierre de cada vela
    # (0 fuera de posición). Puede pasar de 1.0: el sizing es por riesgo, no por
    # capital, así que un stop estrecho compra más notional del que hay en caja
    # — eso es apalancamiento real y se reporta como tal, sin recortar.
    with np.errstate(divide='ignore', invalid='ignore'):
        exposicion_track = np.where(equity > 0, unidades_track * c / equity, 0.0)

    return {'trades': trades, 'entradas': entradas, 'equity': equity,
            'drawdown': drawdown,
            'stop_track': stop_track, 'entrada_track': entrada_track,
            'exposicion_track': exposicion_track,
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


def resultado_filtrado(resultado, direccion=0, capital_inicial=None):
    """Recorta un resultado de simular() a una sola dirección de trade.

    direccion: +1 solo largos, -1 solo cortos, 0 (por defecto) todos.

    Devuelve un dict con la misma forma que simular() — apto para pasarlo tal
    cual a calcular_metricas() o a montecarlo() — pero con los arrays de
    'trades' y 'entradas' filtrados por 'dir' y con la equity RECONSTRUIDA.

    Por qué hay que reconstruirla: resultado['equity'] se marca a mercado vela
    a vela (incluye el PnL no realizado de la posición abierta), así que es una
    única curva que mezcla ambos lados y no se puede trocear. Aquí se compone
    capital_inicial * cumprod(1 + ret_pct) sobre los trades que sobreviven al
    filtro, escalonando el salto en la vela de SALIDA de cada uno: la curva de
    cierres realizados de "qué habría pasado operando solo ese lado".

    Consecuencia a tener presente: con direccion=0 los recuentos y las métricas
    de trade (n_trades, win rate, profit factor, expectancy...) coinciden
    exactamente con las del resultado original, pero las métricas de curva
    (retorno, max drawdown, Sharpe, R², Ulcer, tiempos de recuperación) pueden
    diferir ligeramente, porque el original las mide sobre la equity marcada a
    mercado y aquí se miden sobre la de cierres realizados. Es el precio de que
    los tres modos (todos / largos / cortos) sean comparables entre sí.

    exposicion_track se conserva pero puesto a 0 en las velas que no pertenecen
    a ningún trade del filtro. Es correcto porque el motor mantiene una sola
    posición a la vez, de modo que los intervalos [entrada, salida] de trades
    distintos nunca se solapan.
    """
    tr = resultado['trades']
    equity_orig = np.asarray(resultado['equity'], dtype=np.float64)
    n = len(equity_orig)
    direccion = int(direccion)

    n_tr = len(tr['dir'])
    m = (np.ones(n_tr, dtype=bool) if direccion == 0
         else (tr['dir'] == direccion))
    tr_f = {k: v[m] for k, v in tr.items()}
    # posición de cada trade superviviente en el array sin filtrar: permite a la
    # GUI volver del trade mostrado al original (centrar el gráfico, etc.)
    tr_f['idx_original'] = np.nonzero(m)[0].astype(np.int64)

    entr = resultado.get('entradas')
    if entr is not None and len(entr.get('dir', ())):
        me = (np.ones(len(entr['dir']), dtype=bool) if direccion == 0
              else (entr['dir'] == direccion))
        entr_f = {k: v[me] for k, v in entr.items()}
    else:
        entr_f = entr

    cap0 = float(capital_inicial if capital_inicial is not None
                 else (equity_orig[0] if n else 0.0))
    if cap0 <= 0:
        cap0 = 1.0

    equity = np.full(n, cap0, dtype=np.float64)
    if n and len(tr_f['idx_salida']):
        orden = np.argsort(tr_f['idx_salida'], kind='stable')
        idx_sal = tr_f['idx_salida'][orden].astype(np.int64)
        acum = cap0 * np.cumprod(1.0 + tr_f['ret_pct'][orden].astype(np.float64))
        # nº de trades ya cerrados en cada vela -> valor de la curva ahí
        cerrados = np.searchsorted(idx_sal, np.arange(n), side='right')
        equity = np.where(cerrados > 0, acum[np.maximum(cerrados - 1, 0)], cap0)

    eq_max = np.maximum.accumulate(equity) if n else equity
    with np.errstate(divide='ignore', invalid='ignore'):
        drawdown = np.where(eq_max > 0, equity / eq_max - 1.0, 0.0)

    expo = resultado.get('exposicion_track')
    if expo is not None and n:
        dentro = np.zeros(n, dtype=bool)
        for a, b in zip(tr_f['idx_entrada'], tr_f['idx_salida']):
            dentro[int(a):int(b) + 1] = True
        expo = np.where(dentro, np.asarray(expo, dtype=np.float64), 0.0)

    return {'trades': tr_f, 'entradas': entr_f, 'equity': equity,
            'drawdown': drawdown,
            'stop_track': resultado.get('stop_track'),
            'entrada_track': resultado.get('entrada_track'),
            'exposicion_track': expo,
            'capital_final': float(equity[-1]) if n else 0.0,
            'n_trades': int(m.sum())}


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

    Y un bloque de EXPOSICIÓN, que responde a cuánto tiempo y cuánto capital
    hace falta inmovilizar para conseguir ese retorno:
    - exposicion_pct: % de las velas del tramo con posición abierta. A
      diferencia del resto de métricas, un trade NO se asigna al tramo por su
      vela de entrada: su intervalo se recorta a la ventana, porque un trade
      que entra en IS y sale en OOS ocupa tiempo real en ambos.
    - exposicion_capital_pct: fracción media del capital comprometida, contando
      las velas en plano como 0. Distingue lo que exposicion_pct no puede: estar
      dentro con el 20% del tamaño vivo no es estar dentro con el 100%. Puede
      superar el 100% (apalancamiento). Requiere 'exposicion_track' en el
      resultado; con dicts de trades armados a mano queda None.
    - retorno_ajustado_exposicion_pct: retorno anualizado / exposición. Un 11%
      anual con el 10% de exposición da 110%; el mismo 11% con el 90% da 12.2%.
      None si no se anualiza (velas_por_anio) o si no hubo exposición.
    """
    equity = resultado['equity']
    n = len(equity)
    idx_fin = n if idx_fin is None else min(idx_fin, n)
    tr = resultado['trades']
    m = (tr['idx_entrada'] >= idx_ini) & (tr['idx_entrada'] < idx_fin)
    pnl = tr['pnl'][m]
    ret_pct = tr['ret_pct'][m]
    r_m = tr['r_multiple'][m]
    dur = (tr['idx_salida'][m] - tr['idx_entrada'][m])

    out = {'n_trades': int(m.sum()), 'win_rate': None, 'profit_factor': None,
           'retorno_pct': None, 'retorno_anual_pct': None, 'max_dd_pct': None,
           'sharpe': None, 'expectancy_pct': None, 'racha_perdedora': 0,
           'duracion_media': None, 'pnl_total': float(pnl.sum()),
           'r2_equity': None, 'dd_promedio_pct': None,
           'tiempo_recuperacion_medio': None, 'tiempo_recuperacion_max': None,
           'sqn': None, 'payoff_ratio': None, 'pct_mejor_trade': None,
           'slippage_minimo_pct': None, 'impacto_comisiones_pct': None,
           'ulcer_index': None, 'etd_r_medio': None,
           'eficiencia_entrada_media': None, 'eficiencia_salida_media': None,
           'exposicion_pct': 0.0, 'exposicion_capital_pct': None,
           'retorno_ajustado_exposicion_pct': None}

    if idx_fin <= idx_ini:
        return out

    # ── exposición: unión de los intervalos [entrada, salida] ──
    # Sumar (idx_salida - idx_entrada) contaría velas repetidas, porque las
    # salidas parciales generan varias filas que comparten idx_entrada y solo
    # cambian idx_salida. La unión las colapsa y llega hasta la última parcial,
    # que es cuando la posición está cerrada del todo. El motor mantiene una
    # sola posición a la vez, así que intervalos de posiciones distintas nunca
    # se solapan. Inclusive en el extremo: un trade que entra y sale en la misma
    # vela (stop tocado el día de la entrada) ocupa 1 vela, no 0.
    en_mercado = np.zeros(n, dtype=bool)
    for a, b in zip(tr['idx_entrada'], tr['idx_salida']):
        en_mercado[int(a):int(b) + 1] = True
    out['exposicion_pct'] = float(en_mercado[idx_ini:idx_fin].mean()) * 100.0

    expo_track = resultado.get('exposicion_track')
    if expo_track is not None:
        out['exposicion_capital_pct'] = float(
            np.mean(expo_track[idx_ini:idx_fin])) * 100.0

    eq = equity[idx_ini:idx_fin]
    eq0 = eq[0] if eq[0] > 0 else 1.0
    out['retorno_pct'] = float(eq[-1] / eq0 - 1.0) * 100.0
    eq_max = np.maximum.accumulate(eq)
    with np.errstate(divide='ignore', invalid='ignore'):
        dd = np.where(eq_max > 0, eq / eq_max - 1.0, 0.0)
    out['max_dd_pct'] = float(dd.min()) * 100.0
    out['ulcer_index'] = float(np.sqrt(np.mean(dd ** 2))) * 100.0
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

    # retorno por unidad de tiempo expuesto: lo que permite comparar un sistema
    # que renta poco pero casi nunca está dentro con otro que renta lo mismo
    # sin salir nunca. Conserva el signo: en pérdidas el ratio es negativo.
    if out['retorno_anual_pct'] is not None and out['exposicion_pct'] > 0:
        out['retorno_ajustado_exposicion_pct'] = \
            out['retorno_anual_pct'] / out['exposicion_pct'] * 100.0

    if len(pnl):
        ganadores = pnl > 0
        out['win_rate'] = float(ganadores.mean())
        ganancia = float(pnl[ganadores].sum())
        perdida = float(-pnl[~ganadores].sum())
        out['profit_factor'] = (ganancia / perdida) if perdida > 0 else float('inf')
        out['expectancy_pct'] = float(r_m.mean())
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
        # columnas derivadas opcionales: calcular_metricas también admite dicts
        # de trades construidos a mano (tests, herramientas externas) que solo
        # traen lo básico, así que se ausentan sin romper el resto de métricas
        def _col(nombre):
            return tr[nombre][m] if nombre in tr else np.full(int(m.sum()), np.nan)

        etd_v = _col('etd_r')
        ent_v = _col('eficiencia_entrada')
        sal_v = _col('eficiencia_salida')
        if np.any(~np.isnan(etd_v)):
            out['etd_r_medio'] = float(np.nanmean(etd_v))
        if np.any(~np.isnan(ent_v)):
            out['eficiencia_entrada_media'] = float(np.nanmean(ent_v))
        if np.any(~np.isnan(sal_v)):
            out['eficiencia_salida_media'] = float(np.nanmean(sal_v))
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
