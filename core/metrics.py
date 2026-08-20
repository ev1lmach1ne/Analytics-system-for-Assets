"""
core/metrics.py
Funciones puras de métricas cuantitativas compartidas entre
library/scripts_utiles/limpieza_datos_er.py y
library/scripts_utiles/analisis_descriptivo.py.

Sin efectos secundarios (no leen config, no imprimen, no tocan DB) —
pensadas para ser testeadas de forma aislada.
"""
import numpy as np
from numba import njit


# ── Efficiency Ratio (ER) ──────────────────────────────────────────
def calcular_umbrales_er(er_series):
    """
    Umbral dinámico de tendencia/ruido para el Efficiency Ratio (ER):
    media ± 1 desviación estándar, acotado a [0.05, 0.95].
    """
    er_medio = er_series.mean()
    er_std   = er_series.std()
    umbral_tendencia = min(0.95, er_medio + er_std)
    umbral_ruido     = max(0.05, er_medio - er_std)
    return {
        'er_medio': er_medio,
        'er_std': er_std,
        'umbral_tendencia': umbral_tendencia,
        'umbral_ruido': umbral_ruido,
    }


def calcular_er_series(retorno_log, periodo):
    """
    Efficiency Ratio (ER) rodante sobre retornos log:
    |movimiento neto| / movimiento total en `periodo` velas.
    Misma fórmula que la limpieza (PERIODO_ER), parametrizada por ventana.
    """
    movimiento_neto  = retorno_log.rolling(periodo).sum().abs()
    movimiento_total = retorno_log.abs().rolling(periodo).sum()
    er = (movimiento_neto / movimiento_total).round(6)
    return er.fillna(0)


# nogil=True en todos los @njit: numba por defecto RETIENE el GIL durante la
# ejecución compilada, así que un cálculo largo (Hurst sobre millones de velas
# de 1m) congelaría la GUI entera aunque corra dentro de un QThread. Con nogil
# el hilo de fondo trabaja de verdad en paralelo al event loop de Qt.
@njit(nogil=True)
def calcular_kama_numba(close, er, fast, slow):
    """
    Kaufman Adaptive Moving Average: suaviza `close` con una constante
    SC = (ER*(2/(fast+1) - 2/(slow+1)) + 2/(slow+1))^2 que se adapta al ER.
    ER en NaN se trata como 0 (suavizado lento).
    """
    n = len(close)
    kama = np.full(n, np.nan)
    if n == 0:
        return kama

    sc_fast = 2.0 / (fast + 1.0)
    sc_slow = 2.0 / (slow + 1.0)

    # Arrancar en el primer precio válido; los NaN de close (p.ej. velas de
    # calentamiento de la limpieza) mantienen el último KAMA en vez de propagarse.
    inicio = -1
    for i in range(n):
        if not np.isnan(close[i]):
            inicio = i
            break
    if inicio == -1:
        return kama

    kama[inicio] = close[inicio]
    for i in range(inicio + 1, n):
        if np.isnan(close[i]):
            kama[i] = kama[i - 1]
            continue
        e = er[i]
        if np.isnan(e):
            e = 0.0
        sc = (e * (sc_fast - sc_slow) + sc_slow) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    return kama


# ── Parabolic SAR (Wilder) ──────────────────────────────────────────
@njit(nogil=True)
def calcular_sar_numba(high, low, af_inicial, af_paso, af_max):
    """
    Parabolic SAR de Wilder. Devuelve (sar[n], tendencia[n] int8: +1 alcista
    / -1 bajista). tendencia[i] es la tendencia vigente al cierre de la vela
    i y solo usa datos hasta i, así que no repinta.

    El factor de aceleración sube `af_paso` cada vez que la tendencia hace un
    nuevo extremo, con tope en `af_max`, y se reinicia en cada giro.

    La tendencia inicial se asume alcista: es una suposición arbitraria e
    inevitable (no hay historia previa a la vela 0) que solo condiciona las
    primeras velas, hasta el primer giro.
    """
    n = len(high)
    sar = np.full(n, np.nan)
    tendencia = np.zeros(n, dtype=np.int8)
    if n == 0:
        return sar, tendencia

    tendencia[0] = 1
    sar[0] = low[0]
    af = af_inicial
    ep = high[0]          # extreme point: el máximo/mínimo de la tendencia viva

    for i in range(1, n):
        nuevo_sar = sar[i - 1] + af * (ep - sar[i - 1])
        if tendencia[i - 1] > 0:
            # el SAR de una tendencia alcista nunca puede meterse dentro del
            # rango de las dos velas previas: si lo hiciera, el giro saltaría
            # por un movimiento que ya había ocurrido
            limite = low[i - 1]
            if i >= 2 and low[i - 2] < limite:
                limite = low[i - 2]
            if nuevo_sar > limite:
                nuevo_sar = limite
            if low[i] < nuevo_sar:
                tendencia[i] = -1
                sar[i] = ep       # al girar, el SAR salta al extremo alcanzado
                ep = low[i]
                af = af_inicial
            else:
                tendencia[i] = 1
                sar[i] = nuevo_sar
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + af_paso, af_max)
        else:
            limite = high[i - 1]
            if i >= 2 and high[i - 2] > limite:
                limite = high[i - 2]
            if nuevo_sar < limite:
                nuevo_sar = limite
            if high[i] > nuevo_sar:
                tendencia[i] = 1
                sar[i] = ep
                ep = high[i]
                af = af_inicial
            else:
                tendencia[i] = -1
                sar[i] = nuevo_sar
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + af_paso, af_max)
    return sar, tendencia


# ── Percentil rodante ───────────────────────────────────────────────
@njit(nogil=True)
def calcular_percentil_rodante_numba(serie, ventana):
    """
    Percentil (0-100) del valor actual de `serie` dentro de su propia
    ventana rodante de `ventana` velas, incluyéndose. NaN mientras no haya
    ventana completa; los NaN dentro de la ventana se ignoran.

    Percentil relativo a la propia historia del activo, no a un umbral
    absoluto: así el mismo ajuste ("solo el 30% de mayor volatilidad") vale
    para cualquier instrumento y timeframe sin recalibrar.

    Los empates cuentan a mitad de rango (equivalente a
    scipy.stats.percentileofscore con kind='mean'). Sin eso una serie
    constante daría percentil 100 en todas sus velas —cada valor "supera" a
    todos sus iguales— y un tramo de volatilidad plana se clasificaría como
    volatilidad extrema. Con el rango medio, una serie constante da 50.
    """
    n = len(serie)
    out = np.full(n, np.nan)
    if ventana < 1:
        return out
    for i in range(ventana - 1, n):
        actual = serie[i]
        if np.isnan(actual):
            continue
        menores = 0
        iguales = 0
        total = 0
        for k in range(i - ventana + 1, i + 1):
            v = serie[k]
            if np.isnan(v):
                continue
            total += 1
            if v < actual:
                menores += 1
            elif v == actual:
                iguales += 1
        if total > 0:
            out[i] = 100.0 * (menores + 0.5 * iguales) / total
    return out


# ── Régimen de Hurst ────────────────────────────────────────────────
def contar_regimen_hurst(hurst_series):
    """
    Clasifica una serie de Exponente de Hurst en tres regímenes con
    umbrales empíricos fijos: tendencia (H>0.58), aleatorio
    (0.52<=H<=0.58), reversión a la media (H<0.52).
    """
    total_tendencia = int((hurst_series > 0.58).sum())
    total_aleatorio = int(((hurst_series >= 0.52) & (hurst_series <= 0.58)).sum())
    total_reversion = int((hurst_series < 0.52).sum())
    return {
        'total_tendencia': total_tendencia,
        'total_aleatorio': total_aleatorio,
        'total_reversion': total_reversion,
    }


# ── Exponente de Hurst (R/S, Anis-Lloyd calibrado) ─────────────────
@njit(nogil=True)
def hurst_rs_numba(series, lags):
    """
    Calcula el Exponente de Hurst aplicando la corrección de Anis-Lloyd
    calibrada para la curtosis y ruido real de activos financieros.
    """
    n = len(series)
    n_lags = len(lags)

    log_lags = np.empty(n_lags)
    log_rs   = np.empty(n_lags)

    # Factor de ajuste cuantitativo para corregir colas pesadas en mercados reales
    CALIBRACION_RUIDO = 0.915

    for k in range(n_lags):
        lag = lags[k]
        n_chunks = n // lag

        rs_sum = 0.0
        count  = 0

        for c in range(n_chunks):
            chunk = series[c * lag : (c + 1) * lag]

            m = 0.0
            for v in chunk:
                m += v
            m /= lag

            cumsum = 0.0
            c_min  = 0.0
            c_max  = 0.0
            s_sq   = 0.0

            for v in chunk:
                cumsum += (v - m)
                if cumsum < c_min: c_min = cumsum
                if cumsum > c_max: c_max = cumsum
                s_sq += (v - m) ** 2

            if lag > 1:
                s = (s_sq / (lag - 1)) ** 0.5
            else:
                s = 0.0

            if s > 0.0:
                rs_sum += (c_max - c_min) / s
                count += 1

        rs_observado = rs_sum / count if count > 0 else 0.0

        # Ecuación base de Anis-Lloyd
        rs_teorico = ((lag - 0.5) / lag) * (1.0 / (2.0 * np.pi * lag))**(-0.5)
        for i in range(1, lag):
            rs_teorico += ((lag - i) / (lag * i))**0.5

        # Aplicamos el multiplicador de calibración al modelo teórico
        rs_teorico_ajustado = rs_teorico * CALIBRACION_RUIDO

        log_lags[k] = np.log(lag)
        if rs_observado > 0.0 and rs_teorico_ajustado > 0.0:
            log_rs[k] = np.log(rs_observado) - np.log(rs_teorico_ajustado) + np.log(lag) * 0.5
        else:
            log_rs[k] = np.log(lag) * 0.5

    # Regresión lineal manual (OLS)
    mean_x = 0.0
    mean_y = 0.0
    for i in range(n_lags):
        mean_x += log_lags[i]
        mean_y += log_rs[i]
    mean_x /= n_lags
    mean_y /= n_lags

    num = 0.0
    den = 0.0
    for i in range(n_lags):
        num += (log_lags[i] - mean_x) * (log_rs[i] - mean_y)
        den += (log_lags[i] - mean_x) ** 2

    if den == 0.0:
        return 0.5

    h = num / den

    if h < 0.0: h = 0.0
    if h > 1.0: h = 1.0
    return h


@njit(nogil=True)
def calcular_hurst_array(retornos, ventana, paso, lags):
    """Aplica hurst_rs_numba en ventana deslizante sobre `retornos`."""
    n         = len(retornos)
    resultado = np.full(n, np.nan)
    for i in range(ventana, n, paso):
        resultado[i] = hurst_rs_numba(retornos[i - ventana:i], lags)
    return resultado


# ── Half-Life OU (regresión AR(1) manual) ──────────────────────────
def calcular_half_life_ou(precio_log, velas_por_dia=None):
    """
    Vida media de reversión (Half-Life) de un proceso Ornstein-Uhlenbeck
    ajustado por regresión AR(1) de método de momentos sobre precio_log.
    """
    delta_p = precio_log.diff().dropna()
    p_lag   = precio_log.shift(1).dropna()

    idx_comun  = delta_p.index.intersection(p_lag.index)
    delta_p_hl = delta_p.loc[idx_comun].values
    p_lag_hl   = p_lag.loc[idx_comun].values

    try:
        p_lag_mean = p_lag_hl.mean()
        cov_hl     = np.mean((p_lag_hl - p_lag_mean) * (delta_p_hl - delta_p_hl.mean()))
        var_hl     = np.var(p_lag_hl)
        beta_hl    = cov_hl / var_hl if var_hl != 0 else 0.0

        if -1 < beta_hl < 0:
            half_life_velas = -np.log(2) / np.log(1 + beta_hl)
        else:
            half_life_velas = None

        if half_life_velas is not None and velas_por_dia is not None and velas_por_dia > 0:
            half_life_dias = half_life_velas / velas_por_dia
        else:
            half_life_dias = None
    except Exception:
        beta_hl = None
        half_life_velas = None
        half_life_dias = None

    return {'beta_hl': beta_hl, 'half_life_velas': half_life_velas, 'half_life_dias': half_life_dias}


# ── Riesgo de cola: CVaR / Expected Shortfall y derivados ──────────
def conditional_value_at_risk(serie, conf=0.95):
    """CVaR (Expected Shortfall) de una serie: la media del peor (1-conf)%
    de los valores. Misma definición que jesse (services/metrics.py):
    se ordena la serie de peor a mejor y se promedia la cola izquierda.

    conf=0.95 -> media del peor 5%; conf=0.99 -> media del peor 1%.
    Devuelve el peor valor (mínimo) si la cola queda por debajo de un
    elemento, igual que jesse (services/metrics.py)."""
    serie = np.asarray(serie, dtype=float)
    serie = serie[np.isfinite(serie)]
    n = len(serie)
    if n == 0:
        return np.nan
    index = int((1.0 - conf) * n)
    if index < 1:
        return float(np.min(serie))
    return float(np.sort(serie)[:index].mean())


def autocorr_penalty(retornos):
    """Penalización por autocorrelación de los retornos (jesse,
    services/metrics.py): si los retornos están correlacionados en serie, la
    varianza real del proceso es mayor que la muestral y el Sharpe queda
    sobreestimado. Devuelve 1.0 (sin penalización) cuando no hay autocorr.

    factor = sqrt(1 + 2 * Σ_{x=1..n-1} ((n-x)/n) * |coef|^x)
    donde coef es la correlación entre retornos[i] y retornos[i+1]."""
    r = np.asarray(retornos, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 3 or np.std(r) == 0:
        return 1.0
    coef = np.abs(np.corrcoef(r[:-1], r[1:])[0, 1])
    if not np.isfinite(coef):
        return 1.0
    corr = np.array([((n - x) / n) * coef ** x for x in range(1, n)])
    return float(np.sqrt(1.0 + 2.0 * np.sum(corr)))


def sharpe_smart(retornos, velas_por_anio=None, rf=0.0):
    """Sharpe ratio corregido por autocorrelación (jesse, smart=True):
    sharpe / autocorr_penalty, anualizado si se pasa velas_por_anio."""
    r = np.asarray(retornos, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 2:
        return None
    if rf != 0.0:
        r = r - (rf / velas_por_anio if velas_por_anio else rf / 252.0)
    desv = np.std(r)
    if desv == 0:
        return None
    pen = autocorr_penalty(r)
    if pen == 0:
        return None
    res = np.mean(r) / (desv * pen)
    if velas_por_anio:
        res = res * np.sqrt(velas_por_anio)
    return float(res)


# ── Cambio acumulado medio por periodo (día/semana/mes/año) ────────
def curvas_cambio_acumulado(retornos, fechas, dias_semana):
    """Curvas del cambio acumulado MEDIO (en %) de un activo, una por
    horizonte de calendario:

      - 'dia':    trayectoria intra-día, pasos = hora UTC 0..24. El paso 24:00
                  cierra el hueco nocturno: es el retorno de la vela 00:00 del
                  día siguiente (el movimiento 23:00 → 00:00 del día), así la
                  curva es un ciclo continuo 00:00 → 24:00 (siguiente 00:00).
      - 'semana': trayectoria de la semana, pasos = 0..dias_semana-1
                  (5 = Lun-Vie para STOCK/FUTURO/FOREX; 7 = Lun-Dom para CRYPTO)
      - 'mes':    trayectoria del mes, pasos = día del mes 1..31
      - 'anio':   trayectoria del año, pasos = mes 1..12

    Para cada periodo del histórico se calcula la trayectoria del cambio
    acumulado (retornos log sumados; un paso sin velas —mercado cerrado—
    cuenta como 0, la curva queda plana) y se promedian todas las
    trayectorias. El periodo que contiene la última vela está incompleto y
    se excluye del promedio. El último punto de cada curva es el cambio
    promedio total de ese periodo (equivalente a sumar 'paso_medio').

    Parámetros:
      retornos:    array 1D de retornos log (fraccionales) alineado con fechas.
      fechas:      array 1D de timestamps (datetime64, cualquier resolución).
      dias_semana: 5 o 7 — días de la semana que cotiza el activo.

    Devuelve un dict con las curvas disponibles ('dia'/'semana'/'mes'/'anio'),
    cada una con {'y', 'pasos', 'labels', 'paso_medio', 'n', 'total'}
    (valores en % salvo pasos/labels/n), más 'dias_semana'."""
    retornos = np.asarray(retornos, dtype=float)
    fechas = np.asarray(fechas)
    n_velas = min(len(retornos), len(fechas))
    if n_velas == 0:
        return {}
    retornos = retornos[:n_velas]
    fechas = fechas[:n_velas]
    retornos = np.where(np.isfinite(retornos), retornos, 0.0)

    dias = fechas.astype('datetime64[D]').astype('int64')
    hora = (fechas.astype('datetime64[h]').astype('int64') - dias * 24).astype('int64')
    mes_idx = fechas.astype('datetime64[M]').astype('int64')
    anio = mes_idx // 12 + 1970
    mes = mes_idx % 12 + 1
    dias_inicio_mes = fechas.astype('datetime64[M]').astype('datetime64[D]').astype('int64')
    dia_mes = (dias - dias_inicio_mes).astype('int64') + 1
    dow = (dias + 3) % 7          # 1970-01-01 era jueves (=3): Lun=0..Dom=6
    semana = (dias + 3) // 7      # semanas que arrancan en lunes

    def _curva(id_periodo, paso, max_paso, labels, paso_final=None):
        uids, idx = np.unique(id_periodo, return_inverse=True)
        tabla = np.zeros((len(uids), max_paso))
        np.add.at(tabla, (idx, paso), retornos)
        if paso_final is not None:
            # Retorno que cierra el hueco hacia el siguiente periodo: va a la
            # ÚLTIMA columna del periodo DESTINO. En el día, el retorno de la
            # vela 00:00 del día siguiente cierra el paso 24:00 (movimiento
            # 23:00 -> 00:00) y el ciclo queda continuo 00:00 -> 24:00.
            valido = (paso_final >= uids[0]) & (paso_final <= uids[-1])
            if valido.any():
                dest = np.clip(np.searchsorted(uids, paso_final[valido]),
                               0, len(uids) - 1)
                coincide = uids[dest] == paso_final[valido]
                if coincide.any():
                    np.add.at(tabla, (dest[coincide], max_paso - 1),
                              retornos[valido][coincide])
        fila_final = np.searchsorted(uids, id_periodo[-1])
        completos = np.ones(len(uids), dtype=bool)
        completos[fila_final] = False
        if not completos.any():
            return None
        filas = tabla[completos]
        y = np.cumsum(filas, axis=1).mean(axis=0) * 100.0
        return {
            'y': y,
            'pasos': np.arange(max_paso, dtype=int),
            'labels': labels,
            'paso_medio': filas.mean(axis=0) * 100.0,
            'n': int(filas.shape[0]),
            'total': float(y[-1]),
        }

    dias_semana = int(dias_semana)
    nombres_dia = ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom'][:dias_semana]
    meses = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun',
             'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']

    curvas = {}
    paso_final_dia = np.where(hora == 0, dias - 1, -1)
    curvas['dia'] = _curva(dias, np.clip(hora, 0, 23), 25,
                           [f'{h:02d}:00' for h in range(24)] + ['24:00'],
                           paso_final=paso_final_dia)
    curvas['semana'] = _curva(semana, np.clip(dow, 0, dias_semana - 1),
                              dias_semana, nombres_dia)
    curvas['mes'] = _curva(anio * 100 + mes, np.clip(dia_mes - 1, 0, 30), 31,
                           [str(d) for d in range(1, 32)])
    curvas['anio'] = _curva(anio, mes - 1, 12, meses)
    curvas['dias_semana'] = dias_semana
    return {k: v for k, v in curvas.items() if v is not None}


def _drawdown_series(retornos):
    """Serie de drawdown (<=0) a partir de retornos, convención de jesse:
    curva = (1+r).cumprod(); dd = curva/running_max - 1."""
    r = np.asarray(retornos, dtype=float)
    precios = np.cumprod(1.0 + r)
    running_max = np.maximum.accumulate(precios)
    dd = np.divide(precios, running_max, out=np.zeros_like(precios),
                   where=running_max > 0) - 1.0
    return np.nan_to_num(dd, nan=0.0, posinf=0.0, neginf=0.0)


def serenity_index(retornos, rf=0.0):
    """Serenity Index (jesse, services/metrics.py): retorno total entre la
    profundidad de la cola de drawdowns. Combina el Ulcer Index (penaliza
    profundidad Y duración del drawdown) con el CVaR del propio drawdown:
    premia retornos altos con drawdowns poco profundos y cortos.

    resultado = (Σ ret − rf) / (ulcer × pitfall)
    donde pitfall = −CVaR(drawdown) / std(retornos)"""
    r = np.asarray(retornos, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 2 or np.std(r) == 0:
        return None
    dd = _drawdown_series(r)
    cvar_dd = conditional_value_at_risk(dd, 0.95)
    ulcer = float(np.sqrt(np.divide((dd ** 2).sum(), n - 1)))
    if ulcer == 0 or cvar_dd == 0:
        return None
    pitfall = -cvar_dd / np.std(r)
    if pitfall == 0:
        return None
    return float((np.sum(r) - rf) / (ulcer * pitfall))


def sortino_ratio(retornos, velas_por_anio=None, rf=0.0):
    """Sortino ratio (jesse, services/metrics.py): retorno medio entre la
    desviación SOLO de las pérdidas (downside deviation), anualizado si se
    pasa velas_por_anio. rf se interpreta POR PERIODO. None si no hay
    downside (serie sin pérdidas)."""
    r = np.asarray(retornos, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 2:
        return None
    if rf != 0.0:
        r = r - (rf / velas_por_anio if velas_por_anio else rf / 252.0)
    downside = np.sqrt(np.mean(np.where(r < 0, r, 0.0) ** 2))
    if downside == 0:
        return None
    res = float(np.mean(r) / downside)
    if velas_por_anio:
        res = res * np.sqrt(velas_por_anio)
    return res


def omega_ratio(retornos, rf=0.0, velas_por_anio=None):
    """Omega ratio (jesse, services/metrics.py): ganancias sobre el umbral
    entre pérdidas bajo él. threshold = (1+rf)^(1/periodos)−1 (con
    periodos = velas_por_anio; sin anualización el umbral es rf directo).
    None si no hay pérdidas (denominador cero)."""
    r = np.asarray(retornos, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 2:
        return None
    if velas_por_anio and velas_por_anio > 0:
        threshold = (1.0 + rf) ** (1.0 / velas_por_anio) - 1.0
    else:
        threshold = rf
    r_less = r - threshold
    numer = float(r_less[r_less > 0.0].sum())
    denom = float(-r_less[r_less < 0.0].sum())
    if denom <= 0:
        return None
    return numer / denom


def calmar_ratio(retorno_anual_pct, max_dd_pct):
    """Calmar ratio: retorno anualizado entre |MaxDD| (ambos en %).
    None si el drawdown es cero."""
    if retorno_anual_pct is None or max_dd_pct is None or max_dd_pct == 0:
        return None
    return float(retorno_anual_pct) / abs(float(max_dd_pct))


# ── Utilidades de trading (kelly, cointegración, alpha/beta) ──────
def kelly_criterion(win_rate, ratio_ganancia_perdida):
    """Fracción óptima de Kelly para el sizing: f = p − (1−p)/b, con p =
    win_rate y b = ganancia media / pérdida media. Devuelve la fracción
    (0..1) o None si no se puede calcular (sin datos, o ratio <= 0). El
    Kelly completo apuesta agresivo; la práctica habitual es una fracción
    del Kelly (medio Kelly = 0.5·f)."""
    if win_rate is None or ratio_ganancia_perdida is None:
        return None
    p = float(win_rate)
    b = float(ratio_ganancia_perdida)
    if not (0.0 < p < 1.0) or b <= 0.0:
        return None
    return p - (1.0 - p) / b


def cointegracion_pares(a, b, p_valor=0.05):
    """Test de cointegración de Engle-Granger entre dos series de precios
    (statsmodels.tsa.stattools.coint: regresión OLS + ADF sobre residuos,
    con la tabla de MacKinnon). `a` y `b` deben estar alineadas por el
    mismo índice temporal y ser precios (no retornos).

    Devuelve {'cointegrados': bool, 'p_value': float, 'tau': float}."""
    try:
        from statsmodels.tsa.stattools import coint
    except ImportError:
        return {'cointegrados': None, 'p_value': None, 'tau': None}
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if len(a) < 30 or np.std(a) == 0 or np.std(b) == 0:
        return {'cointegrados': None, 'p_value': None, 'tau': None}
    tau, p_value, _crit = coint(a, b)
    if not np.isfinite(tau) or not np.isfinite(p_value):
        return {'cointegrados': None, 'p_value': None, 'tau': None}
    return {'cointegrados': bool(p_value < p_valor),
            'p_value': float(p_value), 'tau': float(tau)}


def calculate_alpha_beta(ret_activo, ret_benchmark, rf=0.0,
                         periodos_anio=None):
    """Alpha y Beta de un activo frente a un benchmark por regresión OLS
    sobre retornos alineados: beta = cov/var, alpha ANUALIZADO =
    (media_activo − rf − beta·(media_bench − rf)) · periodos_anio (si no se
    pasa periodos_anio, el alpha queda por periodo). rf se interpreta POR
    PERIODO (si rf es anual, dividirla antes por periodos_anio).

    Devuelve {'alpha': float, 'beta': float, 'r2': float, 'n': int}."""
    a = np.asarray(ret_activo, dtype=float)
    b = np.asarray(ret_benchmark, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    n = len(a)
    if n < 2 or np.var(b) == 0:
        return {'alpha': None, 'beta': None, 'r2': None, 'n': n}
    beta = float(np.cov(a, b)[0, 1] / np.var(b))
    corr = np.corrcoef(a, b)[0, 1]
    r2 = float(corr ** 2) if np.isfinite(corr) else None
    alpha_periodo = float(np.mean(a) - rf - beta * (np.mean(b) - rf))
    alpha = (alpha_periodo * periodos_anio
             if periodos_anio and periodos_anio > 0 else alpha_periodo)
    return {'alpha': alpha, 'beta': beta, 'r2': r2, 'n': n}
