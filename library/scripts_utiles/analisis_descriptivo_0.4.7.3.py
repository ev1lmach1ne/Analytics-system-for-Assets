import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy import stats
import os
import subprocess
import re
import seaborn as sns

# region ── 1. CONFIGURACIÓN — solo cambiar estos valores ────────────────────────────
CONFIG = {
    'activo':     'CRYPTO',
    'nombre':     'BNB/USDT',
    'tf':         '1min',
    'input_path': r"D:\DATOS\Activos\Crypto\Limpiados\BNBUSDT-1m 08-2017_to_03-2026 - copia_preparado.csv"
}

FACTORES = {
    'CRYPTO': {
        # ... CRYPTO (24/7) 
        '1min':  {'anual': 525600, 'trimestral': 129600, 'mensual': 43200, 'semanal': 10080, 'dia': 1440},
        '5min':  {'anual': 105120, 'trimestral': 25920,  'mensual': 8640,  'semanal': 2016,  'dia': 288},
        '15min': {'anual': 35040,  'trimestral': 8640,   'mensual': 2880,  'semanal': 672,   'dia': 96},
        '30min': {'anual': 17520,  'trimestral': 4320,   'mensual': 1440,  'semanal': 336,   'dia': 48},
        '1h':    {'anual': 8760,   'trimestral': 2160,   'mensual': 720,   'semanal': 168,   'dia': 24},
        '4h':    {'anual': 2190,   'trimestral': 540,    'mensual': 180,   'semanal': 42,    'dia': 6},
        '1d':    {'anual': 365,    'trimestral': 90,     'mensual': 30,    'semanal': 7,     'dia': 1}
    },
    'FUTURO': {
        # 1 día = 1440 min
        '1min':  {'anual': 362880, 'trimestral': 90720, 'mensual': 30240, 'semanal': 7056, 'dia': 1440, 'horaria': 60},
        '5min':  {'anual': 72576,  'trimestral': 18144, 'mensual': 6048,  'semanal': 1411, 'dia': 288,  'horaria': 12},
        '15min': {'anual': 24192,  'trimestral': 6048,  'mensual': 2016,  'semanal': 470,  'dia': 96,   'horaria': 4},
        '30min': {'anual': 12096,  'trimestral': 3024,  'mensual': 1008,  'semanal': 235,  'dia': 48,   'horaria': 2},
        '1h':    {'anual': 6048,   'trimestral': 1512,  'mensual': 504,   'semanal': 118,  'dia': 24,   'horaria': 1},
        '4h':    {'anual': 1512,   'trimestral': 378,   'mensual': 126,   'semanal': 29,   'dia': 6,    'horaria': 0.25},
    },
    'STOCK': {
        # 1 día = 390 min (Sesión 6.5h)
        '1min':  {'anual': 98280,  'trimestral': 24570, 'mensual': 8190,  'semanal': 1950, 'dia': 390,  'horaria': 60},
        '5min':  {'anual': 19656,  'trimestral': 4914,  'mensual': 1638,  'semanal': 390,  'dia': 78,   'horaria': 12},
        '15min': {'anual': 6552,   'trimestral': 1638,  'mensual': 546,   'semanal': 130,  'dia': 26,   'horaria': 4},
        '30min': {'anual': 3276,   'trimestral': 819,   'mensual': 273,   'semanal': 65,   'dia': 13,   'horaria': 2},
        '1h':    {'anual': 1638,   'trimestral': 409,   'mensual': 136,   'semanal': 32.5, 'dia': 6.5,  'horaria': 1},
        '4h':    {'anual': 409,    'trimestral': 102,   'mensual': 34,    'semanal': 8,    'dia': 1.6,  'horaria': 0.25},
    }
}

# endregion
# region ── 2. VALIDACIÓN ─────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"ANÁLISIS DESCRIPTIVO — {CONFIG['nombre']} ({CONFIG['tf']})")
print(f"{'='*60}")
print("[1/5] Validando configuración...")
OUTPUT_PDF = CONFIG['input_path'].replace('.csv', f"_informe_{CONFIG['activo']}_{CONFIG['tf']}.pdf")
# endregion
# region ── 3. CARGA ──────────────────────────────────────────────────────────────────
print("[2/5] Cargando archivo CSV...")
df = pd.read_csv(CONFIG['input_path'], parse_dates=['timestamp']).set_index('timestamp')

# --- PARCHE DE SEGURIDAD PARA LAS FECHAS ---
# Nos aseguramos de que el índice sea Datetime y no un objeto o texto
df.index = pd.to_datetime(df.index, errors='coerce')
df = df[df.index.notna()] 
es_datetime_valido = isinstance(df.index, pd.DatetimeIndex)


# ── DETECCIÓN Y COMPROBACIÓN TEMPORAL UNIFICADA ─────────────────────────────────────────────
# [AÑADIDO] Sustituye la dependencia rígida de FACTORES (manual) por una
# detección dinámica de la temporalidad real a partir de los timestamps del
# propio archivo. Si el archivo no tiene timestamps fiables (p. ej. viene en
# TICKS, volumen o rango en vez de tiempo fijo), las métricas que dependan del
# tiempo se omiten mostrando un aviso explícito en vez de hacer fallar el script.
velas_por_dia  = None
velas_por_anio = None
tipo_muestreo  = 'sin_tiempo'

es_datetime_valido = isinstance(df.index, pd.DatetimeIndex) and df.index.is_monotonic_increasing

if es_datetime_valido and len(df) > 2:
    diffs_seg = df.index.to_series().diff().dropna().dt.total_seconds()
    diffs_seg = diffs_seg[diffs_seg > 0]  # descarta timestamps duplicados

    if len(diffs_seg) > 0:
        mediana_seg = diffs_seg.median()
        cv = diffs_seg.std() / mediana_seg if mediana_seg > 0 else np.inf
        # CV bajo  -> duración entre velas casi constante -> barras de tiempo fijo
        # CV alto  -> duración muy irregular -> barras de evento (TICKS/volumen/rango)
        tipo_muestreo = 'tiempo_fijo' if cv < 0.15 else 'evento'

        rango_total_seg  = (df.index[-1] - df.index[0]).total_seconds()
        rango_total_dias = rango_total_seg / 86400

        if rango_total_dias > 0:
            velas_por_dia  = len(df) / rango_total_dias
            velas_por_anio = velas_por_dia * 365

# endregion
# region ── 4. INTEGRIDAD y DETECCIÓN DE DATOS y COLUMNAS ─────────────────────────────────────────────────────────────
print("[3/5] Verificando integridad de datos...")
if 'interpolado' not in df.columns: df['interpolado'] = 0
if 'anomalia'    not in df.columns: df['anomalia']    = 0
if 'ER'          not in df.columns: df['ER']          = 0.0

# endregion
# region ── 5. CÁLCULO DE RETORNOS ────────────────────────────────────────────────────
print("[4/5] Calculando métricas estadísticas...")
df['retorno'] = np.log(df['close'] / df['close'].shift(1))
df = df.dropna(subset=['retorno'])
r  = df['retorno']

if es_datetime_valido:
    mes_stats = df.groupby(df.index.month)['retorno'].sum() * 100
    dia_stats = df.groupby(df.index.dayofweek)['retorno'].sum() * 100
else:
    mes_stats = pd.Series(dtype=float)
    dia_stats = pd.Series(dtype=float)

# endregion
# region ── 6. AUDITORÍA DE CALIDAD Y CONTROL DE INTERPOLACIÓN ──────────────────────────
# ==============================================================================
# [CONFIGURACIÓN] Cambia el nombre si tu columna se llama diferente (ej. 'is_interpolated')
col_flag = 'interpolado' 

if col_flag in df.columns:
    total_filas = len(df)
    total_interpoladas = int(df[col_flag].sum())
    total_originales = total_filas - total_interpoladas
    
    # Calculamos la distancia real por calendario basada en tus fechas exactas
    fecha_inicio = pd.to_datetime('2017-12-31 23:00:00+00:00')
    fecha_fin = pd.to_datetime('2026-06-07 18:00:00+00:00')
    horas_teoricas = int((fecha_fin - fecha_inicio).total_seconds() / 3600) + 1
    
    # Cálculos de control
    filas_fantasma = horas_teoricas - total_filas
    nulos_restantes = df['retorno_log'].isna().sum()

    print(f"\n📊 [AUDITORÍA] DIAGNÓSTICO DE CONTROL DE CALIDAD:")
    print(f"======================================================================")
    print(f" • Horas teóricas por calendario:        {horas_teoricas} velas.")
    print(f" • Filas físicas en tu archivo actual:   {total_filas} velas.")
    print(f"   └── ⚠️ Huecos temporales ('Fantasma'): {filas_fantasma} horas que NO existen en el CSV.")
    print(f"----------------------------------------------------------------------")
    print(f" • Desglose de los datos de tu archivo:")
    print(f"   ├── 🟢 Datos reales (Exchange):       {total_originales} ({total_originales/total_filas:.2%})")
    print(f"   └── 🔵 Datos rellenados (Script):   {total_interpoladas} ({total_interpoladas/total_filas:.2%})")
    print(f"----------------------------------------------------------------------")
    print(f" • Celdas vacías (NaN) sin solucionar en 'retorno_log': {nulos_restantes}")
    print(f"======================================================================\n")
else:
    print(f"⚠️ No se encontró la columna '{col_flag}'. Verifica el nombre exacto en tu CSV.")

# endregion
# region ── 7. CONFIGURACIÓN DE BLOQUES MULTI-TIMEFRAME ──────────────────────────────
# region ── 7. CONFIGURACIÓN DE BLOQUES MULTI-TIMEFRAME ──────────────────────────────
activo_cfg = CONFIG['activo']
tf_cfg = CONFIG['tf']

if activo_cfg in FACTORES and tf_cfg in FACTORES[activo_cfg]:
    bloques = FACTORES[activo_cfg][tf_cfg]
else:
    bloques = {'dia': 1, 'semanal': 7, 'mensual': 30, 'trimestral': 90, 'anual': 365}

def normalizar_serie(serie, bloques_agrupacion):
    s = serie.groupby(np.arange(len(serie)) // int(bloques_agrupacion)).sum()
    return s[np.isfinite(s)]

def calc_max_dd_interno(grupo):
    cum = np.exp(np.cumsum(grupo.values))
    cum_max = np.maximum.accumulate(cum)
    dd = np.divide(cum - cum_max, cum_max, out=np.zeros_like(cum), where=cum_max!=0)
    return dd.min()

series_temporales = {}
stats_temporales = {}

for periodo, n_bloques in bloques.items():
    series_temporales[periodo] = normalizar_serie(df['retorno'], n_bloques)
    s_actual = series_temporales[periodo]
    
    # Max DD Macro
    cum_ret_p = np.exp(s_actual.cumsum())
    mdd_p = ((cum_ret_p - cum_ret_p.cummax()) / cum_ret_p.cummax()).min()
    
    # Max DD Intra-periodo
    bloques_id = np.arange(len(df)) // int(n_bloques)
    dds_por_bloque = df['retorno'].groupby(bloques_id).apply(calc_max_dd_interno)
    
    peor_dd_intrabloque = dds_por_bloque.min()
    idx_peor_bloque = dds_por_bloque.idxmin()
    
    if es_datetime_valido and len(df) > 0:
        fechas_bloque = df.index[bloques_id == idx_peor_bloque]
        fecha_peor = fechas_bloque[0].strftime('%Y-%m-%d')
    else:
        fecha_peor = "N/A"
    
    if len(s_actual) > 2:
        stat_jb_p, p_jb_p = stats.jarque_bera(s_actual)
    else:
        stat_jb_p, p_jb_p = 0.0, 1.0
    
    # 1. Crear el diccionario base
    stats_temporales[periodo] = {
        'media': s_actual.mean(),
        'std':   s_actual.std(),
        'max_dd': mdd_p,
        'max_dd_interno': peor_dd_intrabloque,
        'fecha_peor': fecha_peor,
        'p_value_normalidad': p_jb_p
    }
    
    # 2. Análisis de Dependencia Estadística (ACF/PACF)
    n_lags = min(40, len(s_actual) // 4) 
    if len(s_actual) > n_lags + 5:
        acf_vals = acf(s_actual, nlags=n_lags)
        pacf_vals = pacf(s_actual, nlags=n_lags)
        umbral_dependencia = 1.96 / np.sqrt(len(s_actual))
    else:
        acf_vals, pacf_vals, umbral_dependencia = [], [], 0.0

    # 3. Actualizar el diccionario existente
    stats_temporales[periodo].update({
        'acf_vals': acf_vals,
        'pacf_vals': pacf_vals,
        'umbral': umbral_dependencia
    })
    
    #Determinaciones para análisis de dependencia
def estado_adn(val, umbral):
    """Determina el estado táctico según la dependencia estadística."""
    if val > umbral: 
        return "INERCIA"
    elif val < -umbral: 
        return "REVERSIÓN"
    else: 
        return "RUIDO"
    

# Variables globales para acceso rápido
r_diario_real = series_temporales['dia']
ret_diario, vol_diaria = stats_temporales['dia']['media'], stats_temporales['dia']['std']
ret_semanal, vol_semanal = stats_temporales['semanal']['media'], stats_temporales['semanal']['std']
ret_mensual, vol_mensual = stats_temporales['mensual']['media'], stats_temporales['mensual']['std']
ret_trimestral, vol_trimestral = stats_temporales['trimestral']['media'], stats_temporales['trimestral']['std']
ret_anual, vol_anual = stats_temporales['anual']['media'], stats_temporales['anual']['std']

# Función de validación temporal
def validar_output_terminal(stats_temporales):
    print("\n" + "="*50)
    print("VALIDACIÓN: ANÁLISIS DE ARQUITECTURA TEMPORAL")
    print("="*50)
    print(f"{'Escala':<12} | {'Lag 1':<10} | {'Tipo':<15} | {'Umbral'}")
    print("-" * 50)
    orden = ['dia', 'semanal', 'mensual', 'trimestral', 'anual']
    for periodo in orden:
        if periodo in stats_temporales:
            data = stats_temporales[periodo]
            val = data['pacf_vals'][1] 
            umbral = data['umbral']
            if val > umbral: estado = "INERCIA"
            elif val < -umbral: estado = "REVERSIÓN"
            else: estado = "RUIDO"
            print(f"{periodo.capitalize():<12} | {val:>8.4f} | {estado:<15} | {umbral:.4f}")
    print("="*50 + "\n")

validar_output_terminal(stats_temporales)
# endregion
# region ── 8. MÉTRICAS ──────────────────────────────────────────────────────────────

# ==========================================================================
# region PASO 1: Cálculos de Volatilidad y Retornos Anualizados/Diarios Compuestos
# Calcula la dispersión del precio y el rendimiento compuesto esperado en base al régimen operativo anual del activo.
if velas_por_anio is not None:
    vol_anual  = r.std() * np.sqrt(velas_por_anio)
    vol_diaria = r.std() * np.sqrt(velas_por_dia)
    
    media_log_diaria_real = r_diario_real.mean()
    ret_diario = np.exp(media_log_diaria_real) - 1
    
    dias_ano_regimen = 365 if CONFIG['activo'] == 'CRYPTO' else 252
    ret_anual  = np.exp(media_log_diaria_real * dias_ano_regimen) - 1
    
    sharpe     = ret_anual / vol_anual if (vol_anual is not None and vol_anual != 0) else 0
    calmar_disponible = True
else:
    # AQUÍ ESTÁ EL CAMBIO: Asignamos 0.0 en lugar de None
    vol_anual = vol_diaria = ret_anual = ret_diario = sharpe = 0.0
    calmar_disponible = False
    
# endregion
# ==========================================================================
# region PASO 2: Cálculo del Máximo Drawdown (Max DD) Histórico
# Mide la peor pérdida acumulada de pico a valle sufrida por el precio a lo largo del histórico.
cum_returns = np.exp(r.cumsum())
peak        = cum_returns.cummax()
mdd         = ((cum_returns - peak) / peak).min()
# endregion
# ==========================================================================
# region PASO 3: Cálculo de Tiempos de Recuperación en Drawdown
# Mide la cantidad de velas y el tiempo exacto transcurrido entre el pico máximo y su break-even o recuperación completa.
drawdown_series = (cum_returns - peak) / peak
en_drawdown      = drawdown_series < 0

bloques          = (en_drawdown != en_drawdown.shift()).cumsum()
duraciones_velas = en_drawdown.groupby(bloques).sum()
duraciones_velas = duraciones_velas[duraciones_velas > 0]

if len(duraciones_velas) > 0:
    bloque_max_id      = duraciones_velas.idxmax()
    recovery_velas_max = int(duraciones_velas.max())

    if es_datetime_valido:
        indices_bloque = en_drawdown[bloques == bloque_max_id].index

        # Fecha del PICO (breakeven inicial): la última vela en máximo, justo antes de empezar a caer.
        pos_primera_en_dd = df.index.get_loc(indices_bloque[0])
        ts_pico = df.index[pos_primera_en_dd - 1] if pos_primera_en_dd > 0 else indices_bloque[0]

        # Fecha de RECUPERACIÓN COMPLETA: la primera vela donde el precio vuelve a superar (o igualar) el pico anterior.
        pos_ultima_en_dd = df.index.get_loc(indices_bloque[-1])
        if pos_ultima_en_dd + 1 < len(df.index):
            ts_recuperado = df.index[pos_ultima_en_dd + 1]
        else:
            ts_recuperado = indices_bloque[-1]  # aún no recuperado al final de los datos

        recovery_timedelta = ts_recuperado - ts_pico
        dias_rec    = recovery_timedelta.days
        horas_rec   = recovery_timedelta.seconds // 3600
        minutos_rec = (recovery_timedelta.seconds % 3600) // 60

        # Si la recuperación alcanza o supera 1 año (365 días), se descompone en años + el resto en d/h/m.
        if dias_rec >= 365:
            anios_rec    = dias_rec // 365
            dias_resto   = dias_rec % 365
            etiqueta_anio = "año" if anios_rec == 1 else "años"
            duracion_str = f"{anios_rec} {etiqueta_anio} {dias_resto}d {horas_rec}h {minutos_rec}m"
        else:
            duracion_str = f"{dias_rec}d {horas_rec}h {minutos_rec}m"

        # Formato de fecha: con hora solo si el timeframe es intradía
        fmt_fecha = '%Y-%m-%d %H:%M' if recovery_timedelta < pd.Timedelta(days=3) else '%Y-%m-%d'
        rango_fechas_str = f"{ts_pico.strftime(fmt_fecha)} → {ts_recuperado.strftime(fmt_fecha)}"

        recovery_str = f"{duracion_str}   ({rango_fechas_str})"
    else:
        recovery_str = "N/A (TICKS — sin temporalidad)"
else:
    recovery_velas_max = 0
    recovery_str = "0d 0h 0m" if es_datetime_valido else "N/A (TICKS — sin temporalidad)"

# endregion
# ==========================================================================
# region PASO 4: Cálculo de la Profundidad del Drawdown Medio
# Obtiene el promedio del punto más bajo registrado de manera aislada en cada episodio consecutivo de caída.
if en_drawdown.any():
    profundidad_por_episodio = drawdown_series[en_drawdown].groupby(bloques[en_drawdown]).min()
    drawdown_medio = profundidad_por_episodio.mean()
    num_episodios_dd = len(profundidad_por_episodio)
else:
    drawdown_medio = None
    num_episodios_dd = 0

# endregion
# ==========================================================================
# region PASO 5: Cálculo del Calmar Ratio
# Relaciona el retorno compuesto anualizado frente al máximo drawdown histórico para medir la recompensa por unidad de riesgo.
if calmar_disponible and mdd != 0:
    calmar_ratio = ret_anual / abs(mdd)
else:
    calmar_ratio = None

# endregion
# ==========================================================================
# region PASO 6: Inicialización de Parámetros Auxiliares de Riesgo y Formatos
# Establece las funciones de formateo numérico de strings y las variables de temporalidad del activo.
tf_actual = CONFIG['tf']
var_95_hist = np.percentile(r, 5)
var_99_hist = np.percentile(r, 1)
z_95 = stats.norm.ppf(0.05)
z_99 = stats.norm.ppf(0.01)
var_95_param = r.mean() + z_95 * r.std()
var_99_param = r.mean() + z_99 * r.std()

def fmt_pct(valor, decimales=2):
    return f"{valor*100:.{decimales}f}%" if valor is not None else "N/A (TICKS)"

def fmt_num(valor, decimales=4):
    return f"{valor:.{decimales}f}" if valor is not None else "N/A (TICKS)"

tf_actual = CONFIG['tf']

# endregion
# ==========================================================================
# region PASO 7: Test de Normalidad de Jarque-Bera y Lógica Adaptativa de Módulo VaR
# Evalúa la asimetría y curtosis de la serie para decidir dinámicamente si aplicar un cálculo de VaR Histórico o Paramétrico.
r_clean = r[np.isfinite(r)].dropna()

if len(r_clean) > 0:
    stat_jb, p_jb = stats.jarque_bera(r_clean)
else:
    stat_jb, p_jb = 0.0, 1.0 # Valores seguros si la serie está vacía

if p_jb < 0.05:
    var_95 = np.percentile(r_clean, 5)
    var_99 = np.percentile(r_clean, 1)
    lbl_var_95, lbl_var_99 = f'VaR 95% Histórico ({tf_actual})', f'VaR 99% Histórico ({tf_actual})'
else:
    z_95, z_99 = stats.norm.ppf(0.05), stats.norm.ppf(0.01)
    var_95 = r_clean.mean() + z_95 * r_clean.std()
    var_99 = r_clean.mean() + z_99 * r_clean.std()
    lbl_var_95, lbl_var_99 = f'VaR 95% Paramétrico ({tf_actual})', f'VaR 99% Paramétrico ({tf_actual})'

val_var_95, val_var_99 = f"{-var_95*100:.4f}%", f"{-var_99*100:.4f}%"

# endregion
# ==========================================================================
# region PASO 8: Cálculo de Persistencia de Rangos vía Ratio de Eficiencia (ER)
# Cuantifica el ruido direccional de las series analizando cuántas velas rompen los umbrales de tendencia o ruido locales.
er_medio         = df['ER'].mean()
er_std           = df['ER'].std()

umbral_tendencia = min(0.95, er_medio + er_std)
umbral_ruido     = max(0.05, er_medio - er_std)

total_tendencia  = (df['ER'] > umbral_tendencia).sum()
total_ruido      = (df['ER'] < umbral_ruido).sum()

# Agrupa y contabiliza las ocurrencias del coeficiente Hurst en zonas de tendencia, paseo aleatorio o reversión a la media.

total_tendencia_h = ((df['hurst'] > 0.58)).sum()
total_aleatorio_h  = ((df['hurst'] >= 0.52) & (df['hurst'] <= 0.58)).sum()
total_reversion_h  = ((df['hurst'] < 0.52)).sum()

# endregion
# ==========================================================================
# region PASO 9: Análisis de Periodos Tendenciales (H > 0.6) y Mean Reversion (H < 0.52)
# ==========================================================================
mask_h_tend = df['hurst'] > 0.6
mask_h_rev  = df['hurst'] < 0.52

# --- Rachas consecutivas de tendencia ---
en_tend = mask_h_tend
rachas_tend = (en_tend != en_tend.shift()).cumsum()
rachas_tend_len = en_tend.groupby(rachas_tend).sum()
rachas_tend_len = rachas_tend_len[rachas_tend_len > 0]
duracion_media_racha_tend = rachas_tend_len.mean() if len(rachas_tend_len) > 0 else 0.0
duracion_max_racha_tend   = rachas_tend_len.max() if len(rachas_tend_len) > 0 else 0

pct_tiempo_tend = mask_h_tend.sum() / len(df)
pct_tiempo_rev  = mask_h_rev.sum() / len(df)

if velas_por_dia is not None and velas_por_dia > 0:
    ret_dia_tend  = r[mask_h_tend].mean() * velas_por_dia * 100
    vol_dia_tend  = r[mask_h_tend].std() * np.sqrt(velas_por_dia) * 100
    ret_dia_rev   = r[mask_h_rev].mean() * velas_por_dia * 100
    vol_dia_rev   = r[mask_h_rev].std() * np.sqrt(velas_por_dia) * 100
    sharpe_tend   = ret_dia_tend / vol_dia_tend if vol_dia_tend != 0 else 0.0
    sharpe_rev    = ret_dia_rev / vol_dia_rev if vol_dia_rev != 0 else 0.0
else:
    ret_dia_tend = vol_dia_tend = ret_dia_rev = vol_dia_rev = sharpe_tend = sharpe_rev = None

# endregion
# ==========================================================================
# region PASO 10: Cálculos Multitemporales de Volatilidad Histórica Rodante (Rolling Vol)
# Extrae ventanas rodantes estandarizadas (7d, 30d, 90d, 365d) parametrizando las velas operativas según el tipo de producto.
tipo_activo = CONFIG['activo']  
tf_actual = CONFIG['tf']

velas_dia = FACTORES[tipo_activo][tf_actual]['dia']
velas_anual = FACTORES[tipo_activo][tf_actual]['anual']

dias_ano = 365 if tipo_activo == 'CRYPTO' else 252
dias_trimestre = 90 if tipo_activo == 'CRYPTO' else 63

vol_historica_total = r.std() * np.sqrt(velas_anual)

hv_7d  = r.rolling(window=int(7 * velas_dia)).std() * np.sqrt(velas_anual)
hv_30d = r.rolling(window=int(30 * velas_dia)).std() * np.sqrt(velas_anual)
hv_90d = r.rolling(window=int(dias_trimestre * velas_dia)).std() * np.sqrt(velas_anual)
hv_365d = r.rolling(window=int(dias_ano * velas_dia)).std() * np.sqrt(velas_anual)

val_hv_7d   = hv_7d.iloc[-1] if len(hv_7d) >= (7 * velas_dia) else None
val_hv_30d  = hv_30d.iloc[-1] if len(hv_30d) >= (30 * velas_dia) else None
val_hv_90d  = hv_90d.iloc[-1] if len(hv_90d) >= (dias_trimestre * velas_dia) else None
val_hv_365d = hv_365d.iloc[-1] if len(hv_365d) >= (dias_ano * velas_dia) else None

# --- Comparativa de Regímenes Geométricos de Tendencia Estructural ---
ventana_sma = int(200 * velas_dia)

if len(df) > ventana_sma:
    df['SMA_Regimen'] = df['close'].rolling(window=ventana_sma).mean()
    es_bull = df['close'] > df['SMA_Regimen']
    es_bear = df['close'] <= df['SMA_Regimen']
    
    ret_bull = r[es_bull].mean() * velas_anual
    vol_bull = r[es_bull].std() * np.sqrt(velas_anual)
    
    ret_bear = r[es_bear].mean() * velas_anual
    vol_bear = r[es_bear].std() * np.sqrt(velas_anual)
    
    str_bull = f"Ret: {ret_bull*100:.2f}% | Vol: {vol_bull*100:.2f}%"
    str_bear = f"Ret: {ret_bear*100:.2f}% | Vol: {vol_bear*100:.2f}%"
else:
    str_bull = "N/A"
    str_bear = "N/A"

# endregion
# ==========================================================================
# region PASO 11: Análisis Estadístico de Correlaciones Temporales (Leverage Effect)
# ==========================================================================
ventana_vol = 0
ventana_corr = 0
corr_media_historica = 0.0
corr_max = 0.0
corr_min = 0.0
tiempo_negativa = 0.0
tiempo_positiva = 0.0
hora_mas_volatil = 0
vol_maxima = 0.0
vol_por_hora = pd.Series(dtype=float)

try:
    if 'hora_utc' not in df.columns:
        df['hora_utc'] = df.index.hour

    velas_por_dia = FACTORES[CONFIG['activo']][CONFIG['tf']]['dia']
    ventana_vol = int(velas_por_dia)
    ventana_corr = int(velas_por_dia * 7)
    
    if CONFIG['tf'] == '1d':
        ventana_vol = 7
        ventana_corr = 30

    df['vol_rodante_tf'] = df['retorno'].rolling(window=ventana_vol).std()
    df['corr_leverage'] = df['retorno'].rolling(window=ventana_corr).corr(df['vol_rodante_tf'])
    
    df_clean_corr = df.dropna(subset=['corr_leverage'])
    corr_media_historica = df_clean_corr['corr_leverage'].mean()
    corr_max = df_clean_corr['corr_leverage'].max()
    corr_min = df_clean_corr['corr_leverage'].min()
    tiempo_negativa = (df_clean_corr['corr_leverage'] < 0).mean() * 100
    tiempo_positiva = (df_clean_corr['corr_leverage'] > 0).mean() * 100
    
    vol_por_hora = df.groupby('hora_utc')['retorno'].std() * 100
    hora_mas_volatil = vol_por_hora.idxmax()
    vol_maxima = vol_por_hora.max()

except Exception as e:
    print(f"AVISO: Cálculo de correlaciones saltado: {e}")

# endregion
# ==========================================================================
# region PASO 12: Desviación de la Ley de Escalado Fractal (Raíz de T)
# ==========================================================================
vol_diaria_real = stats_temporales['dia']['std']
vol_mensual_real = stats_temporales['mensual']['std']
# Buscamos la anomalía matemática
vol_mensual_teorica = vol_diaria_real * np.sqrt(30) 
desviacion_escalado = vol_mensual_real - vol_mensual_teorica

# endregion
# ==========================================================================
# region PASO 13: Score de Confluencia Macro (Trend Confluence)
# ==========================================================================
score_tendencia = 0
if series_temporales['mensual'].iloc[-1] > 0: score_tendencia += 1
if series_temporales['semanal'].iloc[-1] > 0: score_tendencia += 1
if series_temporales['dia'].iloc[-1] > 0:     score_tendencia += 1
# endregion
# ==========================================================================
# region PASO 14: Análisis de dependencia — Autocorrelación (ACF) y Parcial (PACF)
def analizar_dependencia_temporal(df, lags=50):
    # Asegurar que tenemos retornos logarítmicos
    if 'retorno_log' not in df.columns:
        df['retorno_log'] = np.log(df['close'] / df['close'].shift(1))
    
    df_clean = df.dropna(subset=['retorno_log'])
    n = len(df_clean)
    
    # Calcular ACF y PACF
    acf_vals = acf(df_clean['retorno_log'], nlags=lags)
    pacf_vals = pacf(df_clean['retorno_log'], nlags=lags)
    
    # Calcular umbral de significancia (95% confianza)
    umbral = 1.96 / np.sqrt(n)
    
    return acf_vals, pacf_vals, umbral
# endregion
# ==========================================================================
# endregion
# region ── 9. 📊 ESTRUCTURACIÓN Y PRESENTACIÓN DE MÉTRICAS FINANCIERAS EN TERMINAL
# ── CONFIGURACIÓN DE ETIQUETAS DINÁMICAS (Para Ticks o Tiempo Fijo) ──────────
if tipo_muestreo == 'tiempo_fijo':
    lbl_c, lbl_m, lbl_l = 'Diario', 'Semanal', 'Mensual'
else:
    lbl_c, lbl_m, lbl_l = 'Bloque Corto', 'Bloque Medio', 'Bloque Largo'
metricas = {
    '1. Información General y tipo de muestreo': {
        'Periodo': f"{df.index.min()} → {df.index.max()}" if es_datetime_valido else f"{len(df):,} ticks (archivo sin temporalidad)",
        'Tipo de muestreo': tipo_muestreo,
        'Total velas': f"{len(df):,}"
    },
    '2. Rendimiento y Retornos': {
        'Retorno anualizado (CAGR)': fmt_pct(ret_anual),
        f'Media retorno ({tf_actual})': f"{r.mean()*100:.6f}%",
        f'Mediana retorno ({tf_actual})': f"{r.median()*100:.6f}%",
        'Retorno diario promedio': f"{(ret_diario or 0)*100:.4f}%",
        'Retornos positivos': f"{(r > 0).sum() / r.count() * 100:.2f}%",
        'Retornos negativos': f"{(r < 0).sum() / r.count() * 100:.2f}%",
        'Alineación de marcos temporales': f"{score_tendencia}/3"
    },
    '3. Avanzada & Volatilidad Histórica': {
        'Volatilidad Histórica Total': f"{vol_historica_total*100:.2f}%",
        'HV 7d': f"{val_hv_7d*100:.2f}%" if val_hv_7d is not None else "N/A",
        'HV 30d': f"{val_hv_30d*100:.2f}%" if val_hv_30d is not None else "N/A",
        f'HV {dias_trimestre}d': f"{val_hv_90d*100:.2f}%" if val_hv_90d is not None else "N/A",
        f'HV {dias_ano}d': f"{val_hv_365d*100:.2f}%" if val_hv_365d is not None else "N/A",
        'Desviación Escalado Fractal': f"{desviacion_escalado:.4%}",
        ' ': '',
    },
    '4. Riesgo y Volatilidad': {
        'Volatilidad anualizada': fmt_pct(vol_anual),
        'Volatilidad diaria': fmt_pct(vol_diaria),
        'Desv. estandar': f"{r.std()*100:.6f}%",
        'Ratio Sharpe (Rf=0)': fmt_num(sharpe),
        'Calmar Ratio': fmt_num(calmar_ratio)
    },
    '5. Drawdown Analysis': {
        'Max Drawdown Histórico': f"{mdd*100:.2f}%",
        'Max Drawdown Interno (Semanal)': f"{stats_temporales['semanal']['max_dd_interno']*100:.2f}%",
        'Max Drawdown Interno (Diario)': f"{stats_temporales['dia']['max_dd_interno']*100:.2f}%  ({stats_temporales['dia']['fecha_peor']})",
        'Drawdown medio': f"{drawdown_medio*100:.2f}%" if drawdown_medio is not None else "N/A",
        'Episodios de drawdown': f"{num_episodios_dd:,}",
        'Tiempo recuperación (velas)': f"{recovery_velas_max:,}",
        'Tiempo recuperación (real)': recovery_str
    },
    '6. VaR y Riesgo del Activo': {
        lbl_var_95: val_var_95,
        lbl_var_99: val_var_99,
        f'Peor caída en {tf_actual} (Mínimo)': f"{r.min()*100:.4f}%",
        f'Mayor subida en {tf_actual} (Máximo)': f"{r.max()*100:.4f}%",
        'Skewness': f"{r.skew():.4f}",
        'Kurtosis': f"{r.kurtosis():.4f}",
        'Jarque-Bera stat': f"{stat_jb:.2f}",
        'Jarque-Bera p-value': f"{p_jb:.6f}",
        f'Jarque-Bera p-value ({lbl_c})': f"{stats_temporales['dia']['p_value_normalidad']:.6f}",
        'Distribucion normal': f"{'NO (fat tails)' if p_jb < 0.05 else 'SI'}"
    },
    '7. Ratio Eficiencia (ER) y Exponente de Hurst': {
        'ER medio': f"{df['ER'].mean():.4f}",
        'ER maximo': f"{df['ER'].max():.4f}",
        'ER minimo': f"{df['ER'].min():.4f}",
        'Periodos tendencia (ER>0.5)': f"{(df['ER'] > 0.5).sum():,}",
        'Paseo aleatorio (ER 0.3-0.5)': f"{((df['ER'] >= 0.3) & (df['ER'] <= 0.5)).sum():,} (Random Walk)",
        'Periodos ruido (ER<0.3)': f"{(df['ER'] < 0.3).sum():,}",
        
        ' ': '',
        
        'Hurst medio': f"{df['hurst'].mean():.4f}",
        'Hurst maximo': f"{df['hurst'].max():.4f}",
        'Hurst minimo': f"{df['hurst'].min():.4f}",
        'Periodos tendencia (H>0.58)': f"{total_tendencia_h:,}",
        'Paseo aleatorio (H 0.52-0.58)': f"{total_aleatorio_h:,}",
        'Periodos mean reversion (H<0.52)': f"{total_reversion_h:,}",
        ' ': '',
        '--- Tendencia Pura (H > 0.6) ---': '',
        '% Tiempo en tendencia pura': f"{pct_tiempo_tend:.2%}",
        'Retorno diario promedio en tendencia': f"{ret_dia_tend:.4f}%" if ret_dia_tend is not None else "N/A",
        'Volatilidad diaria en tendencia': f"{vol_dia_tend:.4f}%" if vol_dia_tend is not None else "N/A",
        'Sharpe en tendencia': fmt_num(sharpe_tend),
        'Duración media racha tendencial': f"{duracion_media_racha_tend:.1f} velas",
        'Duración máxima racha tendencial': f"{duracion_max_racha_tend:.0f} velas",
        ' ': '',
        '--- Mean Reversion (H < 0.52) ---': '',
        '% Tiempo en mean reversion': f"{pct_tiempo_rev:.2%}",
        'Retorno diario promedio en mean rev.': f"{ret_dia_rev:.4f}%" if ret_dia_rev is not None else "N/A",
        'Volatilidad diaria en mean reversion': f"{vol_dia_rev:.4f}%" if vol_dia_rev is not None else "N/A",
        'Sharpe en mean reversion': fmt_num(sharpe_rev),
        'Mejora Sharpe (Tend vs Rev)': f"{(sharpe_tend - sharpe_rev):.4f}" if (sharpe_tend is not None and sharpe_rev is not None) else "N/A"
    },
    '8. Análisis de Correlación y Estacionalidad': {
        'Contexto': f"Timeframe: {CONFIG['tf']} | Análisis Generalizado",
        'Ventana Volatilidad (1 dia)': f"{int(ventana_vol)} velas",
        'Ventana Correlación (1 semana)': f"{int(ventana_corr)} velas",
        'Correlación media Retorno-Vol': f"{corr_media_historica:.4f}",
        'Maxima correlación (FOMO)': f"{corr_max:.4f}",
        'Minima correlación (Panic)': f"{corr_min:.4f}",
        'Tiempo corr. negativa (%)': f"{tiempo_negativa:.2f}%",
        'Hora más volátil (UTC)': f"{int(hora_mas_volatil):02d}:00",
        'Vol. hora pico (%)': f"{vol_maxima:.4f}%",
        'Vol. Londres (08:00)': f"{vol_por_hora.get(8, 0):.4f}%",
        'Vol. NY (14:00)': f"{vol_por_hora.get(14, 0):.4f}%"
    },
    '9. Análisis de dependencia — Autocorrelación (ACF) y Parcial (PACF)': {
        'Contexto': f"Análisis basado en PACF Lag 1",
        'Significancia (Umbral)': f"{stats_temporales['dia']['umbral']:.4f}",
        'Dependencia Diaria': f"{stats_temporales['dia']['pacf_vals'][1]:.4f} → {estado_adn(stats_temporales['dia']['pacf_vals'][1], stats_temporales['dia']['umbral'])}",
        'Dependencia Semanal': f"{stats_temporales['semanal']['pacf_vals'][1]:.4f} → {estado_adn(stats_temporales['semanal']['pacf_vals'][1], stats_temporales['semanal']['umbral'])}",
        'Dependencia Mensual': f"{stats_temporales['mensual']['pacf_vals'][1]:.4f} → {estado_adn(stats_temporales['mensual']['pacf_vals'][1], stats_temporales['mensual']['umbral'])}",
        'Dependencia Trimestral': f"{stats_temporales['trimestral']['pacf_vals'][1]:.4f} → {estado_adn(stats_temporales['trimestral']['pacf_vals'][1], stats_temporales['trimestral']['umbral'])}",
        'Memoria Estructural': "Fuerte (D)" if abs(stats_temporales['dia']['pacf_vals'][1]) > stats_temporales['dia']['umbral'] else "Débil/Ruido"
    }
}
# Output formateado y elegante en la terminal
print(f"\n{'═'*70}")
print(f"{'SISTEMA DE ANALÍTICA DE ACTIVOS - REPORTE DESCRIPTIVO':^70}")
print(f"{'═'*70}")

for categoria, items in metricas.items():
    print(f"\n ▶ {categoria}")
    print(f"  {'─' * (len(categoria) + 2)}")
    for metrica, valor in items.items():
        print(f"    {metrica:<35} : {valor}")

print(f"\n{'═'*70}\n")

# endregion
# region ── 10. PREPARACIÓN PREVIA PDF ──────────────────────────────────────────
# Preparación máscaras de regimen
mask_tend  = df['ER'] > 0.45
mask_trans = (df['ER'] >= 0.30) & (df['ER'] <= 0.45)
mask_ruido = df['ER'] < 0.30

def color_regimen(er_val):
    if er_val > 0.45:   return '#1D9E75'
    elif er_val > 0.30: return '#888888'
    else:               return '#E24B4A'

print("Generando PDF...")

# ── CÁLCULO DE RENDIMIENTOS ESTACIONALES (Capitalización Compuesta) ──
df['factor_crecimiento'] = np.exp(df['retorno'])

orden_meses = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio',
               'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
orden_dias  = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']

if es_datetime_valido:
    # Mapeo por número (independiente del locale del sistema)
    df['mes_nombre'] = df.index.month.map({
        1:'Enero', 2:'Febrero', 3:'Marzo', 4:'Abril', 5:'Mayo', 6:'Junio',
        7:'Julio', 8:'Agosto', 9:'Septiembre', 10:'Octubre', 11:'Noviembre', 12:'Diciembre'
    })
    df['dia_nombre'] = df.index.dayofweek.map({
        0:'Lunes', 1:'Martes', 2:'Miércoles', 3:'Jueves',
        4:'Viernes', 5:'Sábado', 6:'Domingo'
    })

    mes_stats = (df.groupby('mes_nombre')['factor_crecimiento'].prod() - 1) * 100
    dia_stats = (df.groupby('dia_nombre')['factor_crecimiento'].prod() - 1) * 100
    mes_stats = mes_stats.reindex(orden_meses)
    dia_stats = dia_stats.reindex(orden_dias)
else:
    mes_stats = pd.Series(dtype=float)
    dia_stats = pd.Series(dtype=float)
# endregion
# region ── 11. REPRESENTACIONES VISUALES Y GRÁFICAS DE DATOS, GENERACIÓN DE PDF  ──

with PdfPages(OUTPUT_PDF) as pdf:

# region ── PÁGINA 1 Y 2 — Métricas Estructuradas (MOTOR DINÁMICO MULTIPÁGINA) ──
    categorias_reales = list(metricas.keys())
    
    # Configuración de diseño: Máximo de categorías por página (2 por columna)
    CATS_POR_PAGINA = 5
    dy = 0.020  
    
    # Iniciamos el contador de páginas en 2 para el output de la terminal
    for pag_num, i in enumerate(range(0, len(categorias_reales), CATS_POR_PAGINA), start=2):
        real_idx = (pag_num - 2) * CATS_POR_PAGINA
        chunk_pagina = categorias_reales[real_idx : real_idx + CATS_POR_PAGINA]
        
        # Dividimos el lote de esta página equitativamente entre las dos columnas
        mitad = (len(chunk_pagina) + 1) // 2
        col_izq = chunk_pagina[:mitad]
        col_der = chunk_pagina[mitad:]
        
        # Inicializar una nueva hoja en el PDF
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        ax.axis('off')
        fig.patch.set_facecolor('#0f0f0f')

        # Encabezados principales
        texto_titulo = f"{CONFIG['nombre']} ({CONFIG['tf']}) — Analisis Descriptivo"
        if len(categorias_reales) > CATS_POR_PAGINA:
            texto_titulo += f" (Parte {pag_num - 1})"
            
        fig.text(0.5, 0.94, texto_titulo, ha='center', va='top', fontsize=16, fontweight='bold', color='white')
        fig.text(0.5, 0.89, f"Activo: {CONFIG['activo']} | Archivo: {os.path.basename(CONFIG['input_path'])}",
                 ha='center', va='top', fontsize=9, color='#888780')

        # --- RENDEREAR COLUMNA 1 (IZQUIERDA) ---
        y_current = 0.83
        for categoria in col_izq:
            fig.text(0.04, y_current, categoria, fontsize=10, color='#ff9900', fontweight='bold', va='center')
            y_current -= 0.022
            
            for idx, (k, v) in enumerate(metricas[categoria].items()):
                nombre_metrica = str(k).strip()
                # Saltar separadores vacíos para no desperdiciar espacio
                if nombre_metrica == '':
                    y_current -= dy * 0.5
                    continue
                    
                bg = '#161616' if idx % 2 == 0 else '#0d0d0d'
                fig.patches.append(plt.Rectangle((0.02, y_current - 0.008), 0.46, dy,
                                                   transform=fig.transFigure, facecolor=bg, zorder=0))
                key_fsize = 7.0 if len(nombre_metrica) > 33 else 8.5
                fig.text(0.04, y_current, nombre_metrica, fontsize=key_fsize, color='#a0a0a0', va='center')
                
                # Valores por defecto para la columna izquierda
                valor_str = str(v)
                x_val = 0.33
                f_size = 7.0 if len(valor_str) > 30 else 9
                
                # CORRECCIÓN: Match exacto para 'Periodo' y específico para la versión '(real)'
                if nombre_metrica == 'Periodo':
                    valor_str = re.sub(r'\+\d{2}:\d{2}', '', valor_str)
                    x_val = 0.20
                    f_size = min(f_size, 7.0)
                elif 'Tiempo recuperación (real)' in nombre_metrica:
                    x_val = 0.26
                    f_size = min(f_size, 6.5)
                elif any(w in nombre_metrica for w in ['Periodo', 'Régimen', 'Fractal', 'Paseo', 'reversion']):
                    f_size = min(f_size, 8)
                    
                fig.text(x_val, y_current, valor_str, fontsize=f_size, color='white', va='center', fontweight='bold')
                y_current -= dy
            y_current -= 0.020  

        # --- RENDEREAR COLUMNA 2 (DERECHA) ---
        y_current = 0.83
        for categoria in col_der:
            fig.text(0.53, y_current, categoria, fontsize=10, color='#ff9900', fontweight='bold', va='center')
            y_current -= 0.022
            
            for idx, (k, v) in enumerate(metricas[categoria].items()):
                nombre_metrica = str(k).strip()
                # Saltar separadores vacíos para no desperdiciar espacio
                if nombre_metrica == '':
                    y_current -= dy * 0.5
                    continue
                    
                bg = '#161616' if idx % 2 == 0 else '#0d0d0d'
                fig.patches.append(plt.Rectangle((0.52, y_current - 0.008), 0.46, dy,
                                                   transform=fig.transFigure, facecolor=bg, zorder=0))
                key_fsize = 7.0 if len(nombre_metrica) > 33 else 8.5
                fig.text(0.54, y_current, nombre_metrica, fontsize=key_fsize, color='#a0a0a0', va='center')
                
                # Valores por defecto para la columna derecha
                valor_str = str(v)
                x_val = 0.74
                f_size = 7.0 if len(valor_str) > 30 else 9
                
                # CORRECCIÓN: Match exacto para 'Periodo' y específico para la versión '(real)'
                if nombre_metrica == 'Periodo':
                    valor_str = re.sub(r'\+\d{2}:\d{2}', '', valor_str)
                    x_val = 0.61
                    f_size = min(f_size, 7.0)
                elif 'Tiempo recuperación (real)' in nombre_metrica:
                    x_val = 0.70
                    f_size = min(f_size, 6.5)
                elif any(w in nombre_metrica for w in ['real', 'Normalidad', 'p-value', 'Paseo', 'reversion']):
                    f_size = min(f_size, 7.5)
                    
                fig.text(x_val, y_current, valor_str, fontsize=f_size, color='white', va='center', fontweight='bold')
                y_current -= dy
            y_current -= 0.020  

        # Guardar la página actual
        pdf.savefig(fig, facecolor=fig.get_facecolor())
        plt.close()
        
        print(f"Generado página {pag_num}/7 — Métricas")
        
    # endregion
# region ── PÁGINA 2 — Precio y Retornos ─────────────────────
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor('#0f0f0f')
    gs  = gridspec.GridSpec(2, 1, hspace=0.35)
    
    paso = 200
    df_plot = df.iloc[::paso]
    r_plot = r[::paso]
    eje_x_plot = df_plot.index if es_datetime_valido else np.arange(len(df_plot))

    ax1 = fig.add_subplot(gs[0])
    ax1.plot(eje_x_plot, df_plot['close'], color='#1D9E75', linewidth=0.5, rasterized=True)
    ax1.set_facecolor('#111111')
    ax1.set_title(f"{CONFIG['nombre']} — Precio de Cierre ({CONFIG['tf']})",
                  color='white', fontsize=11)
    ax1.set_ylabel('Precio', color='#888780')
    ax1.set_xlabel('Tiempo' if es_datetime_valido else 'Nº de vela (TICKS)', color='#888780')
    ax1.tick_params(colors='#888780')
    ax1.grid(True, alpha=0.2, color='#444')
    for spine in ax1.spines.values(): spine.set_edgecolor('#333')

    ax2 = fig.add_subplot(gs[1])
    ax2.plot(eje_x_plot, r_plot, color='#185FA5', linewidth=0.3, alpha=0.8, rasterized=True)
    ax2.axhline(0, color='#E24B4A', linewidth=0.8, linestyle='--')
    ax2.set_facecolor('#111111')
    ax2.set_title(f"{CONFIG['nombre']} — Retornos Logaritmicos ({CONFIG['tf']})",
                  color='white', fontsize=11)
    ax2.set_ylabel('Retorno log', color='#888780')
    ax2.set_xlabel('Tiempo' if es_datetime_valido else 'Nº de vela (TICKS)', color='#888780')
    ax2.tick_params(colors='#888780')
    ax2.grid(True, alpha=0.2, color='#444')
    for spine in ax2.spines.values(): spine.set_edgecolor('#333')

    pdf.savefig(fig, facecolor=fig.get_facecolor(), dpi=150)
    plt.close()
    print("Generado página 2/7 — Precio y Retornos...")

# endregion
# region ── PÁGINA 3 — Análisis de Estacionalidad ────────
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor('#0f0f0f')

    # ── TÍTULO PRINCIPAL ──
    fig.text(0.5, 0.97, f"{CONFIG['nombre']} ({CONFIG['tf']}) — Análisis de Estacionalidad",
             ha='center', va='top', fontsize=16, fontweight='bold', color='white')
    fig.text(0.5, 0.93, f"Activo: {CONFIG['activo']} | Archivo: {os.path.basename(CONFIG['input_path'])}",
             ha='center', va='top', fontsize=9, color='#888780')

    if es_datetime_valido and mes_stats.notna().any():
        gs = gridspec.GridSpec(2, 1, hspace=0.4)

        ax1 = fig.add_subplot(gs[0])
        mes_plot = mes_stats.fillna(0)
        cols_m = ['#1D9E75' if x >= 0 else '#E24B4A' for x in mes_plot]
        bars1 = ax1.bar(mes_plot.index, mes_plot, color=cols_m, alpha=0.8)
        ax1.set_title('Retorno Acumulado por Mes (%)', color='white', fontsize=12)
        ax1.tick_params(colors='#888780')
        ax1.grid(True, axis='y', alpha=0.2)
        ax1.set_facecolor('#111111')
        for spine in ax1.spines.values(): spine.set_edgecolor('#333')
        for bar in bars1:
            h = bar.get_height()
            if h != 0:
                ax1.text(bar.get_x() + bar.get_width()/2, h,
                         f'{h:.1f}%', ha='center',
                         va='bottom' if h > 0 else 'top',
                         color='white', fontsize=8)

        ax2 = fig.add_subplot(gs[1])
        dia_plot = dia_stats.fillna(0)
        cols_d = ['#1D9E75' if x >= 0 else '#E24B4A' for x in dia_plot]
        bars2 = ax2.bar(dia_plot.index, dia_plot, color=cols_d, alpha=0.8)
        ax2.set_title('Retorno Acumulado por Día de la Semana (%)', color='white', fontsize=12)
        ax2.tick_params(colors='#888780')
        ax2.grid(True, axis='y', alpha=0.2)
        ax2.set_facecolor('#111111')
        for spine in ax2.spines.values(): spine.set_edgecolor('#333')
        for bar in bars2:
            h = bar.get_height()
            if h != 0:
                ax2.text(bar.get_x() + bar.get_width()/2, h,
                         f'{h:.1f}%', ha='center',
                         va='bottom' if h > 0 else 'top',
                         color='white', fontsize=8)
    else:
        ax = fig.add_subplot(111)
        ax.axis('off')
        ax.set_facecolor('#111111')
        ax.text(0.5, 0.5,
                "Análisis de estacionalidad no disponible.\n\n"
                "El archivo está basado en TICKS (sin temporalidad de calendario fija),\n"
                "por lo que no es posible agrupar los retornos por mes o día de la semana.",
                ha='center', va='center', fontsize=12, color='#888780',
                transform=ax.transAxes)

    pdf.savefig(fig, facecolor=fig.get_facecolor())
    plt.close()
    print("Generado página 3/7 — Análisis de Estacionalidad...")
    
# endregion
# region ── PÁGINA 4 — Precio coloreado por régimen ER ────────
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor('#0f0f0f')
    gs  = gridspec.GridSpec(2, 1, hspace=0.4, height_ratios=[3, 1])

    # --- MUESTREO PARA VISUALIZACIÓN ---
    step = 500
    df_sub = df.iloc[::step]
    
    ax4 = fig.add_subplot(gs[0])
    ax4.set_facecolor('#111111')
    ax4.set_title(f"{CONFIG['activo']} — Precio coloreado por Regimen ER ({CONFIG['tf']})", color='white', fontsize=11)
    ax4.set_ylabel('Precio', color='#888780')
    ax4.set_xlabel('Tiempo' if es_datetime_valido else 'Nº de vela (TICKS)', color='#888780')
    ax4.tick_params(colors='#888780')
    ax4.grid(True, alpha=0.2, color='#444')
    for spine in ax4.spines.values(): spine.set_edgecolor('#333')

    # Aplicamos el muestreo a las coordenadas y colores
    eje_x_num = df_sub.index.astype(np.int64) if es_datetime_valido else np.arange(len(df_sub))
    points = np.array([eje_x_num, df_sub['close'].values]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    colors = [color_regimen(er) for er in df_sub['ER'].values]
    
    lc = LineCollection(segments, colors=colors, linewidth=0.6, alpha=0.8, rasterized=True)
    ax4.add_collection(lc)
    ax4.autoscale()

    legend_elements = [
        Line2D([0], [0], color='#1D9E75', linewidth=2, label='Tendencia (ER>0.45)'),
        Line2D([0], [0], color='#888888', linewidth=2, label='Transicion (0.30-0.45)'),
        Line2D([0], [0], color='#E24B4A', linewidth=2, label='Ruido (ER<0.30)'),
    ]
    ax4.legend(handles=legend_elements, facecolor='#222', labelcolor='white', fontsize=8, loc='upper right')

    ax5 = fig.add_subplot(gs[1])
    ax5.set_title("Histórico ER Acumulado", color='white', fontsize=10, pad=10)
    
    x_plot = df.index[::step] if es_datetime_valido else np.arange(len(df))[::step]
    er_plot = df['ER'].values[::step]
    
    ax5.fill_between(x_plot, er_plot, color='#BA7517', alpha=0.2, linewidth=0, rasterized=True)
    ax5.plot(x_plot, er_plot, color='#BA7517', linewidth=0.3, alpha=0.5, rasterized=True)

    er_suavizado = df['ER'].rolling(200).mean()
    ax5.plot(x_plot, er_suavizado[::step], color='white', linewidth=1.0, label='Tendencia (SMA 200)', rasterized=True)
    
    media_er = df['ER'].mean()
    ax5.axhline(media_er, color='#185FA5', linewidth=1.2, linestyle='-', label=f'Media: {media_er:.2f}')
    ax5.axhline(0.45, color='#1D9E75', linewidth=0.8, linestyle='--', label='Ruido Bajo (ER > 0.45)')
    ax5.axhline(0.30, color='#E24B4A', linewidth=0.8, linestyle='--', label='Ruido Alto (ER < 0.30)')

    ax5.set_facecolor('#111111')
    ax5.set_ylabel('ER', color='#888780')
    ax5.set_ylim(0, 1)
    ax5.tick_params(colors='#888780')
    ax5.grid(True, alpha=0.2, color='#444')
    ax5.legend(loc='upper right', facecolor='#222', labelcolor='white', fontsize=7, ncol=2)

    for spine in ax5.spines.values(): spine.set_edgecolor('#333')

    pdf.savefig(fig, facecolor=fig.get_facecolor(), dpi=150)
    plt.close()
    print("Generado página 4/7 — Precio por régimen ER...")

# endregion
# region ── PÁGINA 5 — Análisis de Riesgo: Diario y Anual ────────
    try:
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.patch.set_facecolor('#0f0f0f')
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.50)

        sigmas = [1, 2, 3]
        colores_sigmas = ['#1D9E75', '#BA7517', '#E24B4A']

        def dibujar_campana_p5(ax, media, std, titulo, es_anual=False):
            ax.set_facecolor('#111111')
            x = np.linspace(media - 4*std, media + 4*std, 200)
            y = stats.norm.pdf(x, media, std)
            ax.plot(x, y, color='#185FA5', linewidth=2, label='Distribución Proyectada', rasterized=True)

            # Etiqueta de la Media (μ)
            ax.axvline(media, color='white', linestyle='--', linewidth=1.2, alpha=0.6)
            ax.text(media, -max(y) * 0.06, f"μ: {media:.2%}", color='white', fontsize=7,
                    ha='center', fontweight='bold', bbox=dict(facecolor='black', alpha=0.8, edgecolor='none', pad=1))

            # Lógica de niveles sigma
            for s in sigmas:
                ax.fill_between(x, y, where=(x >= media-s*std) & (x <= media+s*std),
                                color=colores_sigmas[s-1], alpha=0.15, rasterized=True)
                for lado in [-1, 1]:
                    val = media + (lado * s * std)
                    ax.axvline(val, color=colores_sigmas[s-1], linestyle=':', alpha=0.5)
                    
                    if not es_anual:
                        prob = np.mean(r_diario_real <= val) if lado == -1 else np.mean(r_diario_real >= val)
                        txt = f"1 c/{1/prob:.1f}d" if prob > 0 else "No reg."
                    else:
                        factor = 365 if CONFIG['activo'] == 'CRYPTO' else 252
                        umbral = val / np.sqrt(factor)
                        prob = np.mean(r_diario_real <= umbral) if lado == -1 else np.mean(r_diario_real >= umbral)
                        txt = f"1 c/{(1/prob)/factor:.1f}añ" if prob > 0 else "No reg."
                    
                    etiqueta = f"{lado*s}σ: {val:.2%}\n({txt})"
                    ax.text(val, -max(y) * (0.05 if s % 2 != 0 else 0.14), etiqueta,
                            color=colores_sigmas[s-1], fontsize=6, ha='center', fontweight='bold',
                            bbox=dict(facecolor='black', alpha=0.8, edgecolor='none', pad=1))

            ax.set_ylim(-max(y)*0.24, max(y)*1.15)
            ax.set_title(titulo, color='white', fontsize=11, pad=8)
            ax.tick_params(colors='#888780', labelsize=7)
            ax.grid(True, alpha=0.03)
            ax.legend(facecolor='#222', labelcolor='white', fontsize=7, loc='upper right')
            for spine in ax.spines.values(): spine.set_edgecolor('#333')

        # Subplots 1 y 2
        ax1 = fig.add_subplot(gs[0])
        dibujar_campana_p5(ax1, ret_diario, vol_diaria, "Retorno Diario (Proyección)", es_anual=False)

        ax2 = fig.add_subplot(gs[1])
        dibujar_campana_p5(ax2, ret_anual, vol_anual, "Retorno Anual (Proyección)", es_anual=True)

        pdf.savefig(fig, facecolor=fig.get_facecolor(), dpi=150)
        plt.close()
        print("Generado página 5/7 — Riesgo Diario/Anual con Media...")
    except Exception as e:
        print(f"ERROR EN PÁGINA 5: {e}")
        plt.close()

# endregion
# region ── PÁGINA 6 — Análisis de Riesgo Intradiario: Temporalidad Pura ────────
    try:
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.patch.set_facecolor('#0f0f0f')
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.50)

        sigmas = [1, 2, 3]
        colores_sigmas = ['#1D9E75', '#BA7517', '#E24B4A']
        jb_stat_p6, jb_p_val_p6 = stats.jarque_bera(r)
        es_normal_p6 = jb_p_val_p6 > 0.05

        def dibujar_campana_micro(ax, media, std, titulo, tipo_grafico='tf_puro'):
            ax.set_facecolor('#111111')
            x = np.linspace(media - 4*std, media + 4*std, 200)
            y = stats.norm.pdf(x, media, std)
            ax.plot(x, y, color='#185FA5', linewidth=2, label='Distribución Proyectada', rasterized=True)

            if tipo_grafico == 'tf_puro':
                for s in sigmas:
                    ax.fill_between(x, y, where=(x >= media-s*std) & (x <= media+s*std),
                                    color=colores_sigmas[s-1], alpha=0.15, rasterized=True)
                    for lado in [-1, 1]:
                        val = media + (lado * s * std)
                        ax.axvline(val, color=colores_sigmas[s-1], linestyle=':', alpha=0.5)
                        prob_evento = np.mean(r <= val) if lado == -1 else np.mean(r >= val)
                        texto_frecuencia = f"1 c/{1/prob_evento:.1f}vel" if prob_evento > 0 else "No reg."
                        etiqueta = f"{lado*s}σ: {val:.2%}\n({texto_frecuencia})"
                        ax.text(val, -max(y) * (0.05 if s % 2 != 0 else 0.14), etiqueta,
                                color=colores_sigmas[s-1], fontsize=6, ha='center', fontweight='bold',
                                bbox=dict(facecolor='black', alpha=0.8, edgecolor='none', pad=1))
            else:
                niveles_var  = [0.95, 0.99]
                colores_var  = ['#BA7517', '#E24B4A']
                z_scores_var = [1.645, 2.326]
                for idx, (conf, z) in enumerate(zip(niveles_var, z_scores_var)):
                    val_var = media - (z * std) if es_normal_p6 else np.percentile(r, (1-conf)*100)
                    ax.fill_between(x, y, where=(x <= val_var), color=colores_var[idx], alpha=0.2, rasterized=True)
                    ax.axvline(val_var, color=colores_var[idx], linestyle='-', linewidth=1.5, alpha=0.7)
                    prob_evento = np.mean(r <= val_var)
                    texto_frecuencia = f"1 c/{1/prob_evento:.1f}vel" if prob_evento > 0 else "No reg."
                    etiqueta = f"VaR ({int(conf*100)}%): {val_var:.2%}\n({texto_frecuencia})"
                    ax.text(val_var, -max(y) * (0.06 if idx == 0 else 0.16), etiqueta,
                            color=colores_var[idx], fontsize=6, ha='center', fontweight='bold',
                            bbox=dict(facecolor='black', alpha=0.8, edgecolor='none', pad=1))

            ax.axvline(media, color='white', linestyle='--', linewidth=1.2, alpha=0.5)
            ax.set_ylim(-max(y)*0.24, max(y)*1.15)
            ax.set_title(titulo, color='white', fontsize=11, pad=8)
            ax.tick_params(colors='#888780', labelsize=7)
            ax.grid(True, alpha=0.03)
            ax.legend(facecolor='#222', labelcolor='white', fontsize=7, loc='upper right')
            for spine in ax.spines.values(): spine.set_edgecolor('#333')

        ax1 = fig.add_subplot(gs[0])
        dibujar_campana_micro(ax1, r.mean(), r.std(),
                              f"Retorno por Vela {CONFIG['tf']}", tipo_grafico='tf_puro')

        ax2 = fig.add_subplot(gs[1])
        dibujar_campana_micro(ax2, r.mean(), r.std(),
                              f"VaR {'Paramétrico' if es_normal_p6 else 'Histórico'} ({CONFIG['tf']})",
                              tipo_grafico='var')

        pdf.savefig(fig, facecolor=fig.get_facecolor(), dpi=150)
        plt.close()
        print("Generado página 6/7 — Análisis de Riesgos VaR...")

    except Exception as e:
        print(f"ERROR EN PÁGINA 6: {e}")
        plt.close()

# endregion
# region ── PÁGINA 7 — Volatilidad por Hora (Estacionalidad Intradía) ────────
    if 'vol_por_hora' not in locals():
        vol_por_hora = pd.Series(dtype=float)
    tiene_datos_hora = len(vol_por_hora) > 0

    try:
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.patch.set_facecolor('#0f0f0f')

        if tiene_datos_hora:
            gs = gridspec.GridSpec(2, 1, hspace=0.45, top=0.88, bottom=0.08, left=0.07, right=0.95)

            # ── GRÁFICO 1: Volatilidad por hora ──
            ax1 = fig.add_subplot(gs[0])
            ax1.set_facecolor('#111111')
            cols_h = ['#E24B4A' if i == hora_mas_volatil else '#185FA5'
                      for i in vol_por_hora.index]
            ax1.bar(vol_por_hora.index, vol_por_hora.values, color=cols_h, alpha=0.8)
            ax1.set_title(f"{CONFIG['nombre']} — Volatilidad por Hora UTC ({CONFIG['tf']})",
                          color='white', fontsize=11)
            ax1.set_xlabel('Hora UTC', color='#888780')
            ax1.set_ylabel('Volatilidad (%)', color='#888780')
            ax1.tick_params(colors='#888780')
            ax1.set_xticks(range(0, 24))
            ax1.grid(True, alpha=0.2, color='#444')
            for spine in ax1.spines.values(): spine.set_edgecolor('#333')

            # Etiqueta hora más volátil
            ax1.axvline(hora_mas_volatil, color='#E24B4A', linewidth=1.2,
                        linestyle='--', label=f'Hora pico: {int(hora_mas_volatil):02d}:00')
            ax1.legend(facecolor='#222', labelcolor='white', fontsize=8)

            # Líneas sesiones
            for hora, nombre, color in [(0, 'Asia', '#888888'), (8, 'Londres', '#BA7517'), (14, 'NY', '#1D9E75')]:
                ax1.axvline(hora, color=color, linewidth=1, linestyle=':', alpha=0.7, label=nombre)

            # ── GRÁFICO 2: Correlación Retorno-Volatilidad ──
            ax2 = fig.add_subplot(gs[1])
            ax2.set_facecolor('#111111')
            for spine in ax2.spines.values(): spine.set_edgecolor('#333')

            if 'corr_leverage' in df.columns and df['corr_leverage'].notna().any():
                paso = 500
                eje_x    = (df.index[::paso] if es_datetime_valido else np.arange(len(df))[::paso])
                corr_vals = df['corr_leverage'].values[::paso]

                ax2.fill_between(eje_x, corr_vals, where=(corr_vals >= 0),
                                 color='#1D9E75', alpha=0.4, label='Corr. positiva', rasterized=True)
                ax2.fill_between(eje_x, corr_vals, where=(corr_vals < 0),
                                 color='#E24B4A', alpha=0.4, label='Corr. negativa', rasterized=True)
                ax2.plot(eje_x, corr_vals, color='#888888', linewidth=0.4, alpha=0.6, rasterized=True)
                ax2.axhline(0, color='white', linewidth=0.8, linestyle='--', alpha=0.5)
                ax2.axhline(corr_media_historica, color='#185FA5', linewidth=1.2,
                            linestyle='-', label=f'Media: {corr_media_historica:.4f}')
                ax2.set_title('Correlación Retorno-Volatilidad (Leverage Effect)',
                              color='white', fontsize=11)
                ax2.set_ylabel('Correlación', color='#888780')
                ax2.tick_params(colors='#888780')
                ax2.grid(True, alpha=0.2, color='#444')
                ax2.legend(facecolor='#222', labelcolor='white', fontsize=8)
            else:
                ax2.axis('off')
                ax2.text(0.5, 0.5, "Datos de correlación no disponibles.",
                         ha='center', va='center', color='#888780',
                         fontsize=11, transform=ax2.transAxes)
        else:
            ax = fig.add_subplot(111)
            ax.axis('off')
            ax.set_facecolor('#111111')
            ax.text(0.5, 0.5,
                    "Análisis intradía no disponible.\n\n"
                    "El archivo no contiene temporalidad horaria.",
                    ha='center', va='center', fontsize=12,
                    color='#888780', transform=ax.transAxes)

        pdf.savefig(fig, facecolor=fig.get_facecolor(), dpi=150)
        plt.close()
        print("Generado página 7/8 — Volatilidad Intradía...")

    except Exception as e:
        print(f"ERROR EN PÁGINA 7: {e}")
        plt.close()

# endregion
# region ── PÁGINA 8 — Correlograma (ACF/PACF) ────────
# Creamos una figura de tamaño A4 (8.27 x 11.69 pulgadas)
    fig = plt.figure(figsize=(8.27, 11.69))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1])

    # 1. Heatmap (Arriba, ocupando todo el ancho)
    ax1 = fig.add_subplot(gs[0, :])
    data = {p: stats['pacf_vals'][1] for p, stats in stats_temporales.items() if 'pacf_vals' in stats}
    df_heat = pd.DataFrame(data, index=['PACF Lag 1']).T
    sns.heatmap(df_heat, annot=True, cmap='coolwarm', center=0, vmin=-0.5, vmax=0.5, ax=ax1)
    ax1.set_title("Mapa de Calor: Dependencia por Escala")

    # 2. Correlograma (Centro, izquierda y derecha)
    ax2 = fig.add_subplot(gs[1, 0])
    plot_acf(series_temporales['dia'], ax=ax2, lags=15, title='ACF Diario')
    ax3 = fig.add_subplot(gs[1, 1])
    plot_pacf(series_temporales['dia'], ax=ax3, lags=15, title='PACF Diario')

    # 3. Comparativa Aleatoria (Abajo, ocupando todo el ancho)
    ax4 = fig.add_subplot(gs[2, :])
    serie = series_temporales['dia']
    rw = np.random.normal(loc=serie.mean(), scale=serie.std(), size=len(serie))
    ax4.plot(np.cumsum(serie), label='Activo Real', alpha=0.7)
    ax4.plot(np.cumsum(rw), label='Random Walk (Ruido)', alpha=0.5, linestyle='--')
    ax4.legend()
    ax4.set_title("Estructura vs. Ruido Aleatorio")

    plt.tight_layout(pad=3.0)
    pdf.savefig(fig)
    plt.close(fig)
print("Generado página 8/8 Análisis de Dependencia(ACF/PACF)")
# endregion
# endregion
# ── FINALIZACIÓN ──────────────────────────────────────────────────────────────
print(f"{'='*60}")
print(f"✅ Analisis completado con exito.")
print(f"📁 Guardado en: {OUTPUT_PDF}")
print(f"{'='*60}")

carpeta_contenedora = os.path.dirname(OUTPUT_PDF)
subprocess.Popen(f'explorer "{carpeta_contenedora}"')