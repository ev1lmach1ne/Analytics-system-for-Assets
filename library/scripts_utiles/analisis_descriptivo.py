import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from scipy import stats
import os
import subprocess

# ── 1. CONFIGURACIÓN — solo cambiar estos valores ────────────────────────────
CONFIG = {
    'activo':     'CRYPTO',
    'nombre':     'BTC/USDT',
    'tf':         '1h',
    'input_path': r"D:\DATOS\Activos\Crypto\Limpiados\btc_1h_limpio.csv"
}

FACTORES = {
    'CRYPTO': {
        '1min':  {'anual': 525600, 'dia': 1440}, #24/7 abierto
        '5min':  {'anual': 105120, 'dia': 288},
        '15min': {'anual': 35040,  'dia': 96},
        '30min': {'anual': 17520,  'dia': 48},
        '1h':    {'anual': 8760,   'dia': 24},
        '4h':    {'anual': 2190,   'dia': 6},
        '1d':    {'anual': 365,    'dia': 1}
    },
    'FUTURO': {
        '1min':  {'anual': 362880, 'dia': 1440}, # Ajusta según mercado
        '1h':    {'anual': 6048,   'dia': 24},
        '1d':    {'anual': 365,    'dia': 1}
    },
    'STOCK': {
        '1min':  {'anual': 98280,  'dia': 390}, # 6.5 horas trading
        '1h':    {'anual': 1638,   'dia': 6.5},
        '1d':    {'anual': 252,    'dia': 1}
    }
}



# ── 2. VALIDACIÓN ─────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"ANÁLISIS DESCRIPTIVO — {CONFIG['nombre']} ({CONFIG['tf']})")
print(f"{'='*60}")
print("[1/5] Validando configuración...")
OUTPUT_PDF = CONFIG['input_path'].replace('.csv', f"_informe_{CONFIG['activo']}_{CONFIG['tf']}.pdf")

# ── 3. CARGA ──────────────────────────────────────────────────────────────────
print("[2/5] Cargando archivo CSV...")
df = pd.read_csv(CONFIG['input_path'], parse_dates=['timestamp']).set_index('timestamp')

# --- PARCHE DE SEGURIDAD PARA LAS FECHAS ---
# Nos aseguramos de que el índice sea Datetime y no un objeto o texto
df.index = pd.to_datetime(df.index, errors='coerce')
df = df[df.index.notna()] 
es_datetime_valido = isinstance(df.index, pd.DatetimeIndex)

# Verificación de diagnóstico
print(f"DEBUG: ¿Fecha válida detectada?: {es_datetime_valido}")
print(f"DEBUG: Tipo de índice: {df.index.dtype}")

# ── DETECCIÓN TEMPORAL UNIFICADA ─────────────────────────────────────────────
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

print(f"  Tipo de muestreo detectado: {tipo_muestreo}")
if velas_por_dia is not None:
    print(f"  Velas/día (real): {velas_por_dia:.2f} | Velas/año (real): {velas_por_anio:.1f}")
else:
    print("  ⚠️ Archivo basado en TICKS (sin temporalidad fiable): las métricas "
          "anualizadas y de recuperación temporal mostrarán 'N/A (TICKS)'.")

# ── 4. INTEGRIDAD ─────────────────────────────────────────────────────────────
print("[3/5] Verificando integridad de datos...")
if 'interpolado' not in df.columns: df['interpolado'] = 0
if 'anomalia'    not in df.columns: df['anomalia']    = 0
if 'ER'          not in df.columns: df['ER']          = 0.0

# ── 5. CÁLCULO DE RETORNOS ────────────────────────────────────────────────────
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

# ── 6. CONFIGURACIÓN DE BLOQUES DIARIOS MULTI-TIMEFRAME ────────────────
# ==============================================================================
# Nos aseguramos de tener una serie limpia de retornos (quitando infinitos y NaNs)
r_limp = r[np.isfinite(r)].copy()

# Determinamos el tamaño del bloque para un día completo usando tus factores detectados.
# Si tus variables de detección dinámica fallan o dan None, usamos por defecto 24.
bloques_diarios = int(velas_por_dia) if (velas_por_dia is not None and velas_por_dia > 0) else 24

# Agrupamos los retornos en bloques de "un día real" y los sumamos.
# Al ser retornos logarítmicos, la suma matemática consolida perfectamente el periodo diario.
r_diario_real = r_limp.groupby(np.arange(len(r_limp)) // bloques_diarios).sum()
total_dias_muestra = len(r_diario_real)

print(f"   [CONFIG] Mapeo Multi-Timeframe: Agrupando en bloques de {bloques_diarios} velas por día.")
print(f"   [CONFIG] Total de días reales detectados en el histórico: {total_dias_muestra} días.")

# ── 7. MÉTRICAS ───────────────────────────────────────────────────────────────

# ==========================================================================
# PASO 1: Cálculos de Volatilidad y Retornos Anualizados/Diarios Compuestos
# ==========================================================================
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


# ==========================================================================
# PASO 2: Cálculo del Máximo Drawdown (Max DD) Histórico
# ==========================================================================
# Mide la peor pérdida acumulada de pico a valle sufrida por el precio a lo largo del histórico.
cum_returns = np.exp(r.cumsum())
peak        = cum_returns.cummax()
mdd         = ((cum_returns - peak) / peak).min()


# ==========================================================================
# PASO 3: Cálculo de Tiempos de Recuperación en Drawdown
# ==========================================================================
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


# ==========================================================================
# PASO 4: Cálculo de la Profundidad del Drawdown Medio
# ==========================================================================
# Obtiene el promedio del punto más bajo registrado de manera aislada en cada episodio consecutivo de caída.
if en_drawdown.any():
    profundidad_por_episodio = drawdown_series[en_drawdown].groupby(bloques[en_drawdown]).min()
    drawdown_medio = profundidad_por_episodio.mean()
    num_episodios_dd = len(profundidad_por_episodio)
else:
    drawdown_medio = None
    num_episodios_dd = 0


# ==========================================================================
# PASO 5: Cálculo del Calmar Ratio
# ==========================================================================
# Relaciona el retorno compuesto anualizado frente al máximo drawdown histórico para medir la recompensa por unidad de riesgo.
if calmar_disponible and mdd != 0:
    calmar_ratio = ret_anual / abs(mdd)
else:
    calmar_ratio = None


# ==========================================================================
# PASO 6: Inicialización de Parámetros Auxiliares de Riesgo y Formatos
# ==========================================================================
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


# ==========================================================================
# PASO 7: Test de Normalidad de Jarque-Bera y Lógica Adaptativa de Módulo VaR
# ==========================================================================
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


# ==========================================================================
# PASO 8: Cálculo de Persistencia de Rangos vía Ratio de Eficiencia (ER)
# ==========================================================================
# Cuantifica el ruido direccional de las series analizando cuántas velas rompen los umbrales de tendencia o ruido locales.
er_medio         = df['ER'].mean()
er_std           = df['ER'].std()

umbral_tendencia = min(0.95, er_medio + er_std)
umbral_ruido     = max(0.05, er_medio - er_std)

total_tendencia  = (df['ER'] > umbral_tendencia).sum()
total_ruido      = (df['ER'] < umbral_ruido).sum()


# ==========================================================================
# PASO 9: Conteo de Regímenes de Memoria a Largo Plazo vía Exponente de Hurst
# ==========================================================================
# Agrupa y contabiliza las ocurrencias del coeficiente Hurst en zonas de tendencia, paseo aleatorio o reversión a la media.
total_tendencia_h = ((df['hurst'] > 0.58)).sum()
total_aleatorio_h  = ((df['hurst'] >= 0.52) & (df['hurst'] <= 0.58)).sum()
total_reversion_h  = ((df['hurst'] < 0.52)).sum()


# ==========================================================================
# PASO 10: Cálculos Multitemporales de Volatilidad Histórica Rodante (Rolling Vol)
# ==========================================================================
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

# ==========================================================================
# PASO 11: Análisis Estadístico de Correlaciones Temporales (Leverage Effect)
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

    
# 📊 ESTRUCTURACIÓN Y PRESENTACIÓN DE MÉTRICAS FINANCIERAS

metricas = {
    '1. Información General y tipo de muestreo': {
        'Periodo': f"{df.index.min()} → {df.index.max()}" if es_datetime_valido else f"{len(df):,} ticks (archivo sin temporalidad)",
        'Tipo de muestreo': tipo_muestreo,
        'Total velas': f"{len(df):,}"
    },
    '2. Rendimiento y Retornos': {
    'Retorno anualizado (CAGR)': fmt_pct(ret_anual),
    f'Media retorno ({tf_actual})': f"{r.mean()*100:.6f}%",     # Dinámico: 'Media retorno (1h)'
    f'Mediana retorno ({tf_actual})': f"{r.median()*100:.6f}%", # Dinámico: 'Mediana retorno (1h)'
    'Retorno diario promedio': f"{(ret_diario or 0)*100:.4f}%",         # ¡Súper útil añadirlo para comparar!
    'Retornos positivos': f"{(r > 0).sum() / r.count() * 100:.2f}%",
    'Retornos negativos': f"{(r < 0).sum() / r.count() * 100:.2f}%",
    },
    '3b. Avanzada & Volatilidad Histórica': {
        'Volatilidad Histórica Total': f"{vol_historica_total*100:.2f}%",
        'HV 7d': f"{val_hv_7d*100:.2f}%" if val_hv_7d is not None else "N/A",
        'HV 30d': f"{val_hv_30d*100:.2f}%" if val_hv_30d is not None else "N/A",
        f'HV {dias_trimestre}d': f"{val_hv_90d*100:.2f}%" if val_hv_90d is not None else "N/A",
        f'HV {dias_ano}d': f"{val_hv_365d*100:.2f}%" if val_hv_365d is not None else "N/A",
        ' ': '',
    },
    '3. Riesgo y Volatilidad': {
        'Volatilidad anualizada': fmt_pct(vol_anual),
        'Volatilidad diaria': fmt_pct(vol_diaria),
        'Desv. estandar': f"{r.std()*100:.6f}%",
        'Ratio Sharpe (Rf=0)': fmt_num(sharpe),
        'Calmar Ratio': fmt_num(calmar_ratio)
    },
    '4. Drawdown Analysis': {
        'Max Drawdown': f"{mdd*100:.2f}%",
        'Drawdown medio': f"{drawdown_medio*100:.2f}%" if drawdown_medio is not None else "N/A",
        'Episodios de drawdown': f"{num_episodios_dd:,}",
        'Tiempo recuperación (velas)': f"{recovery_velas_max:,}",
        'Tiempo recuperación (real)': recovery_str
    },
    '5. VaR y Riesgo del Activo': {
        lbl_var_95: val_var_95,
        lbl_var_99: val_var_99,
        f'Peor caída en {tf_actual} (Mínimo)': f"{r.min()*100:.4f}%",
        f'Mayor subida en {tf_actual} (Máximo)': f"{r.max()*100:.4f}%",
        'Skewness': f"{r.skew():.4f}",
        'Kurtosis': f"{r.kurtosis():.4f}",
        'Jarque-Bera stat': f"{stat_jb:.2f}",
        'Jarque-Bera p-value': f"{p_jb:.6f}",
        'Distribucion normal': f"{'NO (fat tails)' if p_jb < 0.05 else 'SI'}"
    },
    '6. Ratio Eficiencia (ER) y Exponente de Hurst': {
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
        'Periodos mean reversion (H<0.52)': f"{total_reversion_h:,}"
    },
    '7. Análisis de Correlación y Estacionalidad': {
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

# ── 7. PREPARAR MÁSCARAS DE RÉGIMEN ──────────────────────────────────────────
mask_tend  = df['ER'] > 0.45
mask_trans = (df['ER'] >= 0.30) & (df['ER'] <= 0.45)
mask_ruido = df['ER'] < 0.30

def color_regimen(er_val):
    if er_val > 0.45:   return '#1D9E75'
    elif er_val > 0.30: return '#888888'
    else:               return '#E24B4A'

# ── 8. GENERACIÓN PDF ─────────────────────────────────────────────────────────
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

with PdfPages(OUTPUT_PDF) as pdf:

# ── PÁGINA 1 — Métricas Estructuradas  ──
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.axis('off')
    fig.patch.set_facecolor('#0f0f0f')

    # Encabezados principales
    fig.text(0.5, 0.94,
             f"{CONFIG['nombre']} ({CONFIG['tf']}) — Informe de Analisis Descriptivo",
             ha='center', va='top', fontsize=16, fontweight='bold', color='white')
    fig.text(0.5, 0.89,
             f"Activo: {CONFIG['activo']} | Archivo: {os.path.basename(CONFIG['input_path'])}",
             ha='center', va='top', fontsize=9, color='#888780')

    # Vinculación exacta con tus títulos de categorías
    bloque_col1 = [
        '1. Información General y tipo de muestreo',
        '2. Rendimiento y Retornos',
        '3. Riesgo y Volatilidad'
    ]
    bloque_col2 = [
        '4. Drawdown Analysis',
        '5. VaR, Campana and Risks',
        '6. Efficiency Ratio (ER)'
    ]

    dy = 0.031  # Altura de cada línea de métrica

    # Definición de las categorías asignadas a la columna izquierda
    bloque_col1 = [
        '1. Información General y tipo de muestreo',
        '2. Rendimiento y Retornos',
        '3. Riesgo y Volatilidad',
        '3b. Avanzada & Volatilidad Histórica'
    ]

    dy = 0.025  # Factor de espaciado optimizado para que quepan todos los bloques

    # --- COLUMNA 1 (IZQUIERDA) ---
    y_current = 0.84
    for categoria in bloque_col1:
        if categoria in metricas:
            # Imprimir título de la categoría
            fig.text(0.04, y_current, categoria, fontsize=10, color='#ff9900', fontweight='bold', va='center')
            y_current -= 0.022
            
            # Imprimir sus métricas
            for idx, (k, v) in enumerate(metricas[categoria].items()):
                bg = '#161616' if idx % 2 == 0 else '#0d0d0d'
                fig.patches.append(plt.Rectangle((0.03, y_current - 0.010), 0.44, dy,
                                           transform=fig.transFigure, facecolor=bg, zorder=0))
                
                # Validación: Si la clave no está vacía, pintamos el texto y su valor
                if k.strip() != '':
                    fig.text(0.05, y_current, k, fontsize=9, color='#a0a0a0', va='center')
                    
                    # Ajuste dinámico de tamaño de fuente según longitud de cadena
                    f_size = 8 if k == 'Periodo' or 'Régimen' in k else 9
                    fig.text(0.21, y_current, v, fontsize=f_size, color='white', va='center', fontweight='bold')
                
                y_current -= dy
            y_current -= 0.018  # Espacio de separación entre bloques

    # --- COLUMNA 2 (DERECHA) ---
    y_current = 0.83
    for categoria in bloque_col2:
        if categoria in metricas:
            # Imprimir título de la categoría
            fig.text(0.53, y_current, categoria, fontsize=10, color='#ff9900', fontweight='bold', va='center')
            y_current -= 0.025
            
            # Imprimir sus métricas
            for idx, (k, v) in enumerate(metricas[categoria].items()):
                bg = '#161616' if idx % 2 == 0 else '#0d0d0d'
                fig.patches.append(plt.Rectangle((0.52, y_current - 0.011), 0.44, dy,
                                           transform=fig.transFigure, facecolor=bg, zorder=0))
                fig.text(0.54, y_current, k, fontsize=9, color='#a0a0a0', va='center')
                
                # AJUSTE: Si es el tiempo de recuperación real, bajamos la fuente a 7.5 debido a la fecha final
                f_size = 7.5 if k == 'Tiempo recuperación (real)' else 9
                # AJUSTE: Movido el valor a X=0.70 (antes 0.77) para expandirse sin salir del recuadro
                fig.text(0.70, y_current, v, fontsize=f_size, color='white', va='center', fontweight='bold')
                y_current -= dy
            y_current -= 0.025  # Espacio de separación entre bloques

    # Guardar y cerrar la página del PDF
    pdf.savefig(fig, facecolor=fig.get_facecolor())
    plt.close()
    print("Generando página 1/7 — Métricas...")

    # ── PÁGINA 2 — Precio y Retornos ─────────────────────
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor('#0f0f0f')
    gs  = gridspec.GridSpec(2, 1, hspace=0.35)

    eje_x = df.index if es_datetime_valido else np.arange(len(df))

    ax1 = fig.add_subplot(gs[0])
    ax1.plot(eje_x, df['close'], color='#1D9E75', linewidth=0.5)
    ax1.set_facecolor('#111111')
    ax1.set_title(f"{CONFIG['nombre']} — Precio de Cierre ({CONFIG['tf']})",
                  color='white', fontsize=11)
    ax1.set_ylabel('Precio', color='#888780')
    ax1.set_xlabel('Tiempo' if es_datetime_valido else 'Nº de vela (TICKS)', color='#888780')
    ax1.tick_params(colors='#888780')
    ax1.grid(True, alpha=0.2, color='#444')
    for spine in ax1.spines.values(): spine.set_edgecolor('#333')

    ax2 = fig.add_subplot(gs[1])
    ax2.plot(eje_x, r, color='#185FA5', linewidth=0.3, alpha=0.8)
    ax2.axhline(0, color='#E24B4A', linewidth=0.8, linestyle='--')
    ax2.set_facecolor('#111111')
    ax2.set_title(f"{CONFIG['nombre']} — Retornos Logaritmicos ({CONFIG['tf']})",
                  color='white', fontsize=11)
    ax2.set_ylabel('Retorno log', color='#888780')
    ax2.set_xlabel('Tiempo' if es_datetime_valido else 'Nº de vela (TICKS)', color='#888780')
    ax2.tick_params(colors='#888780')
    ax2.grid(True, alpha=0.2, color='#444')
    for spine in ax2.spines.values(): spine.set_edgecolor('#333')

    pdf.savefig(fig, facecolor=fig.get_facecolor())
    plt.close()
    print("Generando página 2/7 — Precio y Retornos... ")

   # ── PÁGINA 3 — Análisis de Estacionalidad ────────
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
    print("Generando página 3/7 — Análisis de Estacionalidad...")

    # ── PÁGINA 4 — Precio coloreado por régimen ER ────────
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor('#0f0f0f')
    gs  = gridspec.GridSpec(2, 1, hspace=0.4, height_ratios=[3, 1])

    ax4 = fig.add_subplot(gs[0])
    ax4.set_facecolor('#111111')
    ax4.set_title(f"{CONFIG['activo']} — Precio coloreado por Regimen ER ({CONFIG['tf']})", color='white', fontsize=11)
    ax4.set_ylabel('Precio', color='#888780')
    ax4.set_xlabel('Tiempo' if es_datetime_valido else 'Nº de vela (TICKS)', color='#888780')
    ax4.tick_params(colors='#888780')
    ax4.grid(True, alpha=0.2, color='#444')
    for spine in ax4.spines.values(): spine.set_edgecolor('#333')

    # [MODIFICADO] El eje X usa timestamp real si existe; si no, índice secuencial
    eje_x_num = df.index.astype(np.int64) if es_datetime_valido else np.arange(len(df))
    points = np.array([eje_x_num, df['close'].values]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    colors = [color_regimen(er) for er in df['ER'].values]
    lc = LineCollection(segments, colors=colors, linewidth=0.6, alpha=0.8)
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
    ax5.fill_between(eje_x, df['ER'], color='#BA7517', alpha=0.2, linewidth=0)
    ax5.plot(eje_x, df['ER'], color='#BA7517', linewidth=0.3, alpha=0.5)

    er_suavizado = df['ER'].rolling(200).mean()
    ax5.plot(eje_x, er_suavizado, color='white', linewidth=1.0, label='Tendencia (SMA 200)')
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

    pdf.savefig(fig, facecolor=fig.get_facecolor())
    plt.close()
    print("Generando página 4/7 — Precio por régimen ER...")

# ── PÁGINA 5 — Análisis de Riesgo: Diario, Anual y VaR Inteligente ────────
    try:
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.patch.set_facecolor('#0f0f0f')
        
        # 3 filas para alojar Diario, Anual y VaR
        gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1], hspace=0.45)

        sigmas = [1, 2, 3]
        colores_sigmas = ['#1D9E75', '#BA7517', '#E24B4A'] # Verde, Naranja, Rojo
        
        # Régimen de días según el activo para la conversión analítica
        factor_regimen_dias = 365 if CONFIG['activo'] == 'CRYPTO' else 252

        # Calculamos estadísticas empíricas reales de la serie diaria agrupada para la evaluación
        mu_diaria_emp = r_diario_real.mean()
        sigma_diaria_emp = r_diario_real.std()

        # Ejecutamos el test de Jarque-Bera sobre la serie diaria real para decidir el tipo de VaR
        jb_stat, jb_p_value = stats.jarque_bera(r_diario_real)
        es_normal = jb_p_value > 0.05

        def dibujar_campana_extendida(ax, media, std, titulo, tipo_grafico='diario'):
            """
            tipo_grafico puede ser: 'diario', 'anual' o 'var'
            """
            ax.set_facecolor('#111111')
            
            # Generamos la curva de distribución teórica proyectada suave (se mantiene como referencia visual de fondo)
            x = np.linspace(media - 4*std, media + 4*std, 200)
            y = stats.norm.pdf(x, media, std)
            ax.plot(x, y, color='#185FA5', linewidth=2, label='Distribución Proyectada')
            
            if tipo_grafico != 'var':
                # ── LÓGICA ESTÁNDAR PARA SIGMAS (DIARIO Y ANUAL) ──
                for s in sigmas:
                    ax.fill_between(x, y, where=(x >= media-s*std) & (x <= media+s*std), 
                                    color=colores_sigmas[s-1], alpha=0.15)
                    
                    for lado in [-1, 1]:
                        val = media + (lado * s * std)
                        ax.axvline(val, color=colores_sigmas[s-1], linestyle=':', alpha=0.5)
                        
                        if tipo_grafico == 'diario':
                            if lado == -1:
                                prob_evento = np.mean(r_diario_real <= val)
                            else:
                                prob_evento = np.mean(r_diario_real >= val)
                                
                            if prob_evento > 0:
                                inverso = 1 / prob_evento
                                texto_frecuencia = f"1 c/{inverso:.1f}d" if inverso >= 1 else f"{prob_evento*100:.1f}%/d"
                            else:
                                texto_frecuencia = "No reg."
                        else: # 'anual'
                            prob_teorica_anual = stats.norm.sf(s)
                            if prob_teorica_anual > 0:
                                unidades_anos = 1 / prob_teorica_anual
                                texto_frecuencia = f"1 c/{unidades_anos:.1f}añ" if unidades_anos >= 1 else f"{1/unidades_anos:.1f}x/añ"
                            else:
                                texto_frecuencia = "No reg."

                        val_mostrado = max(val, -0.999) if lado == -1 else val
                        nivel_escalon = 0.05 if s % 2 != 0 else 0.14
                        etiqueta = f"{lado*s}σ: {val_mostrado:.1%}\n({texto_frecuencia})" 
                        
                        ax.text(val, -max(y) * nivel_escalon, etiqueta, color=colores_sigmas[s-1], fontsize=6, 
                                ha='center', fontweight='bold', bbox=dict(facecolor='black', alpha=0.8, edgecolor='none', pad=1))
            else:
                # ── LÓGICA INTELIGENTE PARA VALUE AT RISK (VaR) SEGÚN JARQUE-BERA ──
                niveles_var = [0.95, 0.99]
                colores_var = ['#BA7517', '#E24B4A'] # Naranja para 95%, Rojo para 99%
                z_scores_var = [1.645, 2.326]
                
                for idx, (conf, z) in enumerate(zip(niveles_var, z_scores_var)):
                    
                    if es_normal:
                        # 1. VaR Paramétrico (Distribución Normal)
                        val_var = media - (z * std)
                    else:
                        # 2. VaR Histórico (Percentiles empíricos reales de la muestra diaria)
                        percentil = (1 - conf)
                        val_var = np.percentile(r_diario_real, percentil * 100)
                    
                    # Sombreado de la zona de pérdida extrema en el lienzo
                    ax.fill_between(x, y, where=(x <= val_var), color=colores_var[idx], alpha=0.2)
                    ax.axvline(val_var, color=colores_var[idx], linestyle='-', linewidth=1.5, alpha=0.7)
                    
                    # Buscamos la frecuencia real observada en los datos para validar el corte
                    prob_evento = np.mean(r_diario_real <= val_var)
                    if prob_evento > 0:
                        inverso = 1 / prob_evento
                        texto_frecuencia = f"1 c/{inverso:.1f}d" if inverso >= 1 else f"{prob_evento*100:.1f}%/d"
                    else:
                        texto_frecuencia = "No reg."
                        
                    val_mostrado = max(val_var, -0.999)
                    etiqueta = f"VaR ({int(conf*100)}%): {val_mostrado:.1%}\n({texto_frecuencia})"
                    
                    nivel_escalon = 0.06 if idx == 0 else 0.16
                    ax.text(val_var, -max(y) * nivel_escalon, etiqueta, color=colores_var[idx], fontsize=6, 
                            ha='center', fontweight='bold', bbox=dict(facecolor='black', alpha=0.8, edgecolor='none', pad=1))

            # Línea central de la media (μ)
            ax.axvline(media, color='white', linestyle='--', linewidth=1.2, alpha=0.5)
            ax.text(media, -max(y)*0.05, f"μ: {media:.2%}", color='white', fontsize=7,
                    ha='center', fontweight='bold', bbox=dict(facecolor='black', alpha=0.8, edgecolor='none', pad=1))

            ax.set_ylim(-max(y)*0.26, max(y)*1.15)
            ax.set_title(titulo, color='white', fontsize=10, pad=8)
            ax.tick_params(colors='#888780', labelsize=7)
            ax.grid(True, alpha=0.03)
            
            # Mostramos asimetría, curtosis y p-valor del Test de Jarque-Bera
            skew_local = stats.skew(r_diario_real)
            kurt_local = stats.kurtosis(r_diario_real, fisher=True)
            texto_stats = f"Skewness: {skew_local:.2f} | Kurtosis: {kurt_local:.2f} | Jarque-Bera p-val: {jb_p_value:.4f} ({'Normal' if es_normal else 'No Normal'})"
            ax.text(0.015, 0.94, texto_stats, transform=ax.transAxes, color='#888780', fontsize=7, verticalalignment='top')
            ax.legend(facecolor='#222', labelcolor='white', fontsize=7, loc='upper right')

        # 1. Gráfico Diario
        ax1 = fig.add_subplot(gs[0])
        titulo_diario = f"Retorno Diario basado en cierres de {CONFIG['tf']} (Vol Proyectada: {vol_diaria:.2%})"
        dibujar_campana_extendida(ax1, ret_diario, vol_diaria, titulo_diario, tipo_grafico='diario')

        # 2. Gráfico Anual
        ax2 = fig.add_subplot(gs[1])
        titulo_anual = f"Retorno Anual basado en cierres de {CONFIG['tf']} (Vol Proyectada: {vol_anual:.2%})"
        dibujar_campana_extendida(ax2, ret_anual, vol_anual, titulo_anual, tipo_grafico='anual')

        # 3. Gráfico de Value at Risk (VaR) Adaptativo
        ax3 = fig.add_subplot(gs[2])
        tipo_var_label = "Paramétrico" if es_normal else "Histórico"
        titulo_var = f"Value at Risk (VaR {tipo_var_label} Diario) basado en cierres de {CONFIG['tf']}"
        dibujar_campana_extendida(ax3, ret_diario, vol_diaria, titulo_var, tipo_grafico='var')

        pdf.savefig(fig, facecolor=fig.get_facecolor())
        plt.close()
        print(f"Generando página 5/7 — Análisis de Riesgos ({CONFIG['tf']}) ({tipo_var_label})")
    except Exception as e:
        print(f"ERROR EN PÁGINA 5: {e}")
        
# ── PÁGINA 6 — Análisis de Riesgo Intradiario: Temporalidad Pura ────────
    try:
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.patch.set_facecolor('#0f0f0f')
        
        # 2 filas para alojar el Retorno del TF y su VaR respectivo de forma equilibrada
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.50)

        sigmas = [1, 2, 3]
        colores_sigmas = ['#1D9E75', '#BA7517', '#E24B4A'] # Verde, Naranja, Rojo

        # Ejecutamos Jarque-Bera sobre la serie original (tf) para esta página micro
        jb_stat_p6, jb_p_val_p6 = stats.jarque_bera(r)
        es_normal_p6 = jb_p_val_p6 > 0.05

        def dibujar_campana_micro(ax, media, std, titulo, tipo_grafico='tf_puro'):
            ax.set_facecolor('#111111')
            
            # Generamos la curva de distribución teórica proyectada suave como fondo de referencia
            x = np.linspace(media - 4*std, media + 4*std, 200)
            y = stats.norm.pdf(x, media, std)
            ax.plot(x, y, color='#185FA5', linewidth=2, label='Distribución Proyectada')
            
            if tipo_grafico == 'tf_puro':
                # ── LÓGICA DE SIGMAS PARA LA TEMPORALIDAD PURA ──
                for s in sigmas:
                    ax.fill_between(x, y, where=(x >= media-s*std) & (x <= media+s*std), 
                                    color=colores_sigmas[s-1], alpha=0.15)
                    
                    for lado in [-1, 1]:
                        val = media + (lado * s * std)
                        ax.axvline(val, color=colores_sigmas[s-1], linestyle=':', alpha=0.5)
                        
                        # Evaluamos la probabilidad empírica en la serie de tiempo nativa (r)
                        if lado == -1:
                            prob_evento = np.mean(r <= val)
                        else:
                            prob_evento = np.mean(r >= val)
                            
                        if prob_evento > 0:
                            inverso = 1 / prob_evento
                            texto_frecuencia = f"1 c/{inverso:.1f}vel" if inverso >= 1 else f"{prob_evento*100:.1f}%/vel"
                        else:
                            texto_frecuencia = "No reg."

                        val_mostrado = max(val, -0.999) if lado == -1 else val
                        nivel_escalon = 0.05 if s % 2 != 0 else 0.14
                        etiqueta = f"{lado*s}σ: {val_mostrado:.2%}\n({texto_frecuencia})" 
                        
                        ax.text(val, -max(y) * nivel_escalon, etiqueta, color=colores_sigmas[s-1], fontsize=6, 
                                ha='center', fontweight='bold', bbox=dict(facecolor='black', alpha=0.8, edgecolor='none', pad=1))
            
            else:
                # ── LÓGICA INTELIGENTE PARA VALUE AT RISK (VaR) DEL TIMEFRAME ──
                niveles_var = [0.95, 0.99]
                colores_var = ['#BA7517', '#E24B4A'] 
                z_scores_var = [1.645, 2.326]
                
                for idx, (conf, z) in enumerate(zip(niveles_var, z_scores_var)):
                    if es_normal_p6:
                        # VaR Paramétrico basado en datos intradiarios nativos
                        val_var = media - (z * std)
                    else:
                        # VaR Histórico Puro sobre los percentiles de la serie original (r)
                        percentil = (1 - conf)
                        val_var = np.percentile(r, percentil * 100)
                    
                    # Sombreado de colas de pérdida del VaR
                    ax.fill_between(x, y, where=(x <= val_var), color=colores_var[idx], alpha=0.2)
                    ax.axvline(val_var, color=colores_var[idx], linestyle='-', linewidth=1.5, alpha=0.7)
                    
                    # Frecuencia real observada en base a velas/barras
                    prob_evento = np.mean(r <= val_var)
                    if prob_evento > 0:
                        inverso = 1 / prob_evento
                        texto_frecuencia = f"1 c/{inverso:.1f}vel" if inverso >= 1 else f"{prob_evento*100:.1f}%/vel"
                    else:
                        texto_frecuencia = "No reg."
                        
                    val_mostrado = max(val_var, -0.999)
                    etiqueta = f"VaR ({int(conf*100)}%): {val_mostrado:.2%}\n({texto_frecuencia})"
                    
                    nivel_escalon = 0.06 if idx == 0 else 0.16
                    ax.text(val_var, -max(y) * nivel_escalon, etiqueta, color=colores_var[idx], fontsize=6, 
                            ha='center', fontweight='bold', bbox=dict(facecolor='black', alpha=0.8, edgecolor='none', pad=1))

            # Línea central de la media (μ)
            ax.axvline(media, color='white', linestyle='--', linewidth=1.2, alpha=0.5)
            ax.text(media, -max(y)*0.05, f"μ: {media:.3%}", color='white', fontsize=7,
                    ha='center', fontweight='bold', bbox=dict(facecolor='black', alpha=0.8, edgecolor='none', pad=1))

            ax.set_ylim(-max(y)*0.24, max(y)*1.15)
            ax.set_title(titulo, color='white', fontsize=11, pad=8)
            ax.tick_params(colors='#888780', labelsize=7)
            ax.grid(True, alpha=0.03)
            
            # Estadísticas de la serie nativa (r) impresas en la esquina superior izquierda
            skew_local = stats.skew(r)
            kurt_local = stats.kurtosis(r, fisher=True)
            texto_stats = f"Skewness (Muestra {CONFIG['tf']}): {skew_local:.2f} | Kurtosis (Muestra {CONFIG['tf']}): {kurt_local:.2f} | Jarque-Bera p-val: {jb_p_val_p6:.4f} ({'Normal' if es_normal_p6 else 'No Normal'})"
            ax.text(0.015, 0.94, texto_stats, transform=ax.transAxes, color='#888780', fontsize=7, verticalalignment='top')
            ax.legend(facecolor='#222', labelcolor='white', fontsize=7, loc='upper right')

        # 1. Gráfico de Retorno Puro de la Temporalidad Configurada
        ax1 = fig.add_subplot(gs[0])
        titulo_tf = f"Retorno por Vela basado en cierres de {CONFIG['tf']} (Desviación Típica: {r.std():.3%})"
        dibujar_campana_micro(ax1, r.mean(), r.std(), titulo_tf, tipo_grafico='tf_puro')

        # 2. Gráfico de Value at Risk (VaR) Adaptativo para la Temporalidad Configurada
        ax2 = fig.add_subplot(gs[1])
        tipo_var_label = "Paramétrico" if es_normal_p6 else "Histórico"
        titulo_var = f"Value at Risk (VaR {tipo_var_label} {CONFIG['tf']}) basado en cierres de {CONFIG['tf']}"
        dibujar_campana_micro(ax2, r.mean(), r.std(), titulo_var, tipo_grafico='var')

        pdf.savefig(fig, facecolor=fig.get_facecolor())
        plt.close()
        print(f"Generando página 6/7 — Análisis de Riesgos VaR ({CONFIG['tf']})")
    except Exception as e:
        print(f"ERROR EN PÁGINA 6: {e}")

# ── PÁGINA 7 — Volatilidad por Hora (Estacionalidad Intradía) ────────
    try:
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.patch.set_facecolor('#0f0f0f')

        # ── TÍTULO PRINCIPAL ──
        fig.text(0.5, 0.97, f"{CONFIG['nombre']} ({CONFIG['tf']}) — Volatilidad Intradía por Hora UTC",
                 ha='center', va='top', fontsize=16, fontweight='bold', color='white')
        fig.text(0.5, 0.93, f"Activo: {CONFIG['activo']} | Archivo: {os.path.basename(CONFIG['input_path'])}",
                 ha='center', va='top', fontsize=9, color='#888780')

        # ── LÓGICA DE ADAPTACIÓN POR TIMEFRAME ──
        # Timeframes intradía (tienen hora_utc): 1min, 5min, 15min, 30min, 1h, 4h
        # Timeframes diarios (1d): no tiene sentido agrupar por hora → aviso
        tf_con_hora = CONFIG['tf'] not in ('1d',)
        tiene_datos_hora = (
            tf_con_hora
            and es_datetime_valido
            and len(vol_por_hora) > 0
            and vol_por_hora.notna().any()
        )

        if tiene_datos_hora:
            gs = gridspec.GridSpec(2, 1, hspace=0.45,
                                   top=0.88, bottom=0.08,
                                   left=0.07, right=0.95)

            # ── GRÁFICO 1: Histograma de Volatilidad por Hora ──
            ax1 = fig.add_subplot(gs[0])
            ax1.set_facecolor('#111111')
            for spine in ax1.spines.values(): spine.set_edgecolor('#333')

            horas     = vol_por_hora.index.astype(int)
            vols      = vol_por_hora.values
            vol_media = vols.mean()

            # Color: rojo si está por encima de la media, verde si por debajo
            cols_hora = ['#E24B4A' if v >= vol_media else '#1D9E75' for v in vols]

            bars = ax1.bar(horas, vols, color=cols_hora, alpha=0.85, width=0.7, zorder=3)

            # Línea de media horizontal
            ax1.axhline(vol_media, color='#185FA5', linewidth=1.5,
                        linestyle='--', label=f'Media: {vol_media:.4f}%', zorder=4)

            # Marcadores de sesiones de mercado relevantes
            sesiones = {}
            if CONFIG['activo'] == 'CRYPTO':
                sesiones = {'Asia\n(00h)': 0, 'Londres\n(08h)': 8, 'NY\n(14h)': 14}
            elif CONFIG['activo'] in ('FUTURO', 'STOCK'):
                sesiones = {'Premarket\n(13h)': 13, 'NY Open\n(14h)': 14, 'NY Close\n(21h)': 21}

            for label, hora_ses in sesiones.items():
                if hora_ses in horas:
                    ax1.axvline(hora_ses, color='#BA7517', linewidth=1.2,
                                linestyle=':', alpha=0.8, zorder=5)
                    ax1.text(hora_ses, vol_media * 1.02, label,
                             color='#BA7517', fontsize=7, ha='center', va='bottom')

            # Etiqueta en la barra más volátil
            idx_max = vols.argmax()
            ax1.text(horas[idx_max], vols[idx_max],
                     f'  {vols[idx_max]:.4f}%\n  ← Hora pico',
                     color='white', fontsize=7, va='bottom', fontweight='bold')

            # Etiquetas de valor encima de cada barra
            for bar, v in zip(bars, vols):
                ax1.text(bar.get_x() + bar.get_width()/2, v,
                         f'{v:.3f}%', ha='center', va='bottom',
                         color='white', fontsize=6, rotation=90)

            ax1.set_title(f'Volatilidad media por Hora UTC — Timeframe {CONFIG["tf"]}',
                          color='white', fontsize=11, pad=8)
            ax1.set_xlabel('Hora UTC', color='#888780', fontsize=9)
            ax1.set_ylabel('Volatilidad (%)', color='#888780', fontsize=9)
            ax1.set_xticks(range(0, 24))
            ax1.tick_params(colors='#888780', labelsize=8)
            ax1.grid(True, axis='y', alpha=0.15, color='#444', zorder=0)
            ax1.legend(facecolor='#222', labelcolor='white', fontsize=8, loc='upper right')

            # ── GRÁFICO 2: Correlación Retorno-Volatilidad (Leverage Effect) ──
            ax2 = fig.add_subplot(gs[1])
            ax2.set_facecolor('#111111')
            for spine in ax2.spines.values(): spine.set_edgecolor('#333')

            if 'corr_leverage' in df.columns and df['corr_leverage'].notna().any():
                eje_x = df.index if es_datetime_valido else np.arange(len(df))

                corr_vals = df['corr_leverage']
                ax2.fill_between(eje_x, corr_vals,
                                 where=(corr_vals >= 0), color='#1D9E75',
                                 alpha=0.4, label='Corr. positiva (FOMO)')
                ax2.fill_between(eje_x, corr_vals,
                                 where=(corr_vals < 0), color='#E24B4A',
                                 alpha=0.4, label='Corr. negativa (Pánico)')
                ax2.plot(eje_x, corr_vals, color='#888888',
                         linewidth=0.4, alpha=0.6)
                ax2.axhline(0, color='white', linewidth=0.8,
                            linestyle='--', alpha=0.5)
                ax2.axhline(corr_media_historica, color='#185FA5',
                            linewidth=1.2, linestyle='-',
                            label=f'Media: {corr_media_historica:.4f}')

                # Bloque de estadísticas en esquina
                texto_corr = (
                    f"Media: {corr_media_historica:.4f}  |  "
                    f"Máx: {corr_max:.4f}  |  "
                    f"Mín: {corr_min:.4f}  |  "
                    f"Tiempo negativa: {tiempo_negativa:.1f}%  |  "
                    f"Tiempo positiva: {tiempo_positiva:.1f}%"
                )
                ax2.text(0.01, 0.95, texto_corr,
                         transform=ax2.transAxes, color='#888780',
                         fontsize=7, va='top')

                ax2.set_title(
                    f'Leverage Effect — Correlación Retorno vs Volatilidad '
                    f'(ventana {int(ventana_corr)} velas)',
                    color='white', fontsize=11, pad=8)
                ax2.set_ylabel('Correlación', color='#888780', fontsize=9)
                ax2.set_xlabel('Tiempo' if es_datetime_valido else 'Nº vela',
                               color='#888780', fontsize=9)
                ax2.tick_params(colors='#888780', labelsize=8)
                ax2.grid(True, alpha=0.15, color='#444')
                ax2.legend(facecolor='#222', labelcolor='white',
                           fontsize=8, loc='upper right')
            else:
                ax2.axis('off')
                ax2.text(0.5, 0.5,
                         "Datos de correlación no disponibles.",
                         ha='center', va='center',
                         color='#888780', fontsize=11,
                         transform=ax2.transAxes)

        else:
            # Timeframe diario o sin datos horarios → aviso
            ax = fig.add_subplot(111)
            ax.axis('off')
            motivo = (
                f"El timeframe '{CONFIG['tf']}' no tiene resolución intradía.\n\n"
                "El análisis de volatilidad por hora UTC requiere timeframes\n"
                "iguales o menores a 4h (1min, 5min, 15min, 30min, 1h, 4h)."
                if not tf_con_hora else
                "No se pudieron calcular datos de volatilidad por hora.\n\n"
                "Verifica que el archivo tenga timestamps válidos."
            )
            ax.text(0.5, 0.5, motivo,
                    ha='center', va='center',
                    fontsize=12, color='#888780',
                    transform=ax.transAxes)

        pdf.savefig(fig, facecolor=fig.get_facecolor())
        plt.close()
        print("Generando página 7/7 — Volatilidad Intradía por Hora UTC...")

    except Exception as e:
        print(f"ERROR EN PÁGINA 7: {e}")
        plt.close()
        
# ── FINALIZACIÓN ──────────────────────────────────────────────────────────────
print(f"{'='*60}")
print(f"✅ Analisis completado con exito.")
print(f"📁 Guardado en: {OUTPUT_PDF}")
print(f"{'='*60}")

carpeta_contenedora = os.path.dirname(OUTPUT_PDF)
subprocess.Popen(f'explorer "{carpeta_contenedora}"')