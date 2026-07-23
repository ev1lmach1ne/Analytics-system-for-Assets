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
