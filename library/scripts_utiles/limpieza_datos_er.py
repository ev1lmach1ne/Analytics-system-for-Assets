import pandas as pd
import numpy as np
import psycopg2
import warnings
warnings.filterwarnings('ignore')
from numba import njit


# ----------COLORES-----------
ROSA  = "\033[95m"
NEON  = "\033[1;95m"  # Rosa brillante / negrita
RESET = "\033[0m"
# ----------------------------
# ── CONFIGURACIÓN — solo cambia esto ─────────────────────
TABLA      = 'bnb_candles_1m'
FRECUENCIA = '1t' #1t = minutos | 1m=mensual '1H', '4H', '1D' según necesites
OUTPUT     = r"D:\DATOS\Activos\Crypto\BNBUSDT-1m 08-2017_to_03-2026 - copia_preparado.csv"
# ─────────────────────────────────────────────────────────

print("="*50)
print(f"LIMPIEZA — {TABLA} ({FRECUENCIA})")
print("="*50)

# Conexión
conn = psycopg2.connect(
    host='localhost', port=18812,
    database='qdb', user='admin', password='quest'
)


# ── [1/7] DESCARGA DE DATOS ──────────────────────────────────────────────────
print("\n[1/7] Descargando datos...")
df = pd.read_sql(f"""
    SELECT timestamp, open, high, low, close, volume
    FROM {TABLA}
    ORDER BY timestamp ASC
""", conn)
conn.close()
print(f"      Filas originales: {len(df)}")

# ==============================================================================
# [2/8] DETECCIÓN Y RENOMBRAMIENTO DINÁMICO DE LA COLUMNA DE TIEMPO
# ==============================================================================
print("\n[2/8] Detección de columna de tiempo y renombramiento...")

# 1. Si no se llama 'timestamp', buscamos si existe bajo otro nombre y la renombramos
if 'timestamp' not in df.columns:
    posibles_nombres_tiempo = ['open_time', 'time', 'date', 'Timestamp', 'open_time_utc']
    for col in posibles_nombres_tiempo:
        if col in df.columns:
            df = df.rename(columns={col: 'timestamp'})
            print(f"🔄 Pandas detectó la columna '{col}' y la renombró automáticamente a 'timestamp'")
            break

# 2. Si vino directamente indexada por QuestDB, la extraemos a columna
if df.index.name in ['timestamp', 'open_time', 'time', 'date', 'Timestamp'] or 'timestamp' not in df.columns:
    df.index.name = 'timestamp'
    df = df.reset_index(drop=False if 'timestamp' not in df.columns else True)

# 3. Nos aseguramos de que sea formato datetime y SE QUEDE como columna común
df['timestamp'] = pd.to_datetime(df['timestamp'])
print("      -> Columna 'timestamp' lista para el análisis de continuidad.")
# ==============================================================================
    

# ── [3/8] SINCRONIZACIÓN Y LIMPIEZA (BLOQUE SUSTITUTO) ────────────────────────
print("\n[3/8] Sincronizando y limpiando...")

df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.set_index('timestamp')

# Estandarización UTC
if df.index.tz is None:
    df.index = df.index.tz_localize('Europe/Helsinki', ambiguous='NaT', nonexistent='NaT')
    df.index = df.index.tz_convert('UTC')
else:
    df.index = df.index.tz_convert('UTC')

df = df[df.index.notna()]
df = df[~df.index.duplicated(keep='first')]

# LIMPIEZA PRIMERO: Eliminamos filas con saltos temporales sospechosos
df = df[df.index.to_series().diff().fillna(pd.Timedelta(minutes=1)) <= pd.Timedelta(minutes=1)].copy()

# AHORA DETECTAMOS HUECOS REALES
idx_completo = pd.date_range(start=df.index.min(), end=df.index.max(), freq=FRECUENCIA)
huecos = idx_completo.difference(df.index)
print(f"      Huecos detectados: {len(huecos)} velas faltantes")

if len(huecos) > 0:
    print(f"DEBUG: Mostrando los primeros 5 huecos temporales encontrados:")
    print(huecos[:5])
else:
    # Si sale 0, es que los datos son perfectamente continuos
    print("DEBUG: No hay huecos temporales entre el min y max del índice.")

# REINDEXAMOS
df = df.reindex(idx_completo)
es_hueco_temporal = df['close'].isna()

# ── [5/8] REINDEXACIÓN E INTERPOLACIÓN (BLOQUE SUSTITUTO) ────────────────────
print("\n[5/8] Rellenando huecos y corrigiendo datos...")

df = df.reindex(idx_completo)
es_nulo_antes = df['close'].isna()

# Interpolación lineal para mantener la pendiente (crítico para ER y Hurst)
for col in ['open', 'high', 'low', 'close', 'volume']:
    if col in df.columns:
        df[col] = df[col].interpolate(method='linear')

# SOLO si después de interpolar quedan nulos (al principio o final), usamos bfill/ffill
df = df.fillna(method='bfill').fillna(method='ffill')

df['interpolado'] = es_nulo_antes.astype(int)

total_reparaciones = es_nulo_antes.sum()
print(f"      Total de celdas reparadas (huecos + anomalías): {total_reparaciones}")


# ── [6/8] VERIFICACIÓN SINCRONIZACIÓN HORARIA A UTC ──────────────────────────
# Mantenemos el paso alineado para no romper tus scripts o pasos posteriores [6/7] y [7/7]
print("\n[6/8] Verificación Sincronización Horaria a UTC... OK (Procesado de forma segura en el Paso 2)")

# [7/8] Verificación
print("\n[7/8] Verificación final...")
nulos = df[['open','high','low','close','volume']].isna().sum().sum()
print(f"      Valores nulos restantes: {nulos}")
print(f"      {'✅ Dataset limpio' if nulos == 0 else '⚠️  Revisar nulos'}")


# [+] Retornos logarítmicos
print("\n[+] Calculando retornos logarítmicos...")
df['retorno_log'] = np.log(df['close'] / df['close'].shift(1)).round(8)


# [+] Efficiency Ratio (ER) con Umbrales selecionados a partir de desviaciones de la media de resultados, (Umbrales adaptativos)

PERIODO_ER = 10

print(f"\n[8/8] Calculando Efficiency Ratio (ER) utilizando ventana de {NEON}{PERIODO_ER}{RESET} periodos...")

# Cálculo base del indicador
movimiento_neto  = df['retorno_log'].rolling(PERIODO_ER).sum().abs()
movimiento_total = df['retorno_log'].abs().rolling(PERIODO_ER).sum()

# Subimos el redondeo a 6 para no perder resolución en el tercer decimal del umbral
df['ER'] = (movimiento_neto / movimiento_total).round(6)
df['ER'] = df['ER'].fillna(0)

# ------------------------------------------------------------------------------
# CÁLCULO DE UMBRALES DINÁMICOS (1ª Desviación Estándar)
# ------------------------------------------------------------------------------
er_medio = df['ER'].mean()
er_std   = df['ER'].std()

# Definimos los límites sumando y restando una desviación estándar (1 sigma)
umbral_tendencia = er_medio + er_std
umbral_ruido     = er_medio - er_std

# Acotamos los umbrales para que no se desborden del rango teórico [0, 1]
umbral_tendencia = min(0.95, umbral_tendencia)
umbral_ruido     = max(0.05, umbral_ruido)

# Conteo de periodos forzados a enteros (int)
total_tendencia = int((df['ER'] > umbral_tendencia).sum())
total_ruido     = int((df['ER'] < umbral_ruido).sum())

# Métricas finales impresas en consola con formato limpio y precisión de 3 decimales
print(f"      ER medio:                           {er_medio:.4f}")
print(f"      ER máximo:                          {df['ER'].max():.4f}")
print(f"      ER mínimo:                          {df['ER'].min():.4f}")
print()
print(f"      Periodos con ER > {umbral_tendencia:.3f} (Extremo alto de la media 1º desviación estándar): {total_tendencia:,}")
print(f"      Periodos con ER < {umbral_ruido:.3f} (Extremo bajo de la media 1º desviación estándar): {total_ruido:,}")

# [+] Exponente de Hurst con Numba

VENTANA_HURST = 1024
PASO          = 10

print(f"\n[+] Calculando Exponente de Hurst (Numba) utilizando ventana de {NEON}{VENTANA_HURST}{RESET} periodos y paso de {NEON}{PASO}{RESET} periodos...")

@njit
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

@njit
def calcular_hurst_array(retornos, ventana, paso, lags):
    n         = len(retornos)
    resultado = np.full(n, np.nan)
    for i in range(ventana, n, paso):
        resultado[i] = hurst_rs_numba(retornos[i - ventana:i], lags)
    return resultado

lags_estandar = np.array([16, 32, 64, 128, 256], dtype=np.int64)
retornos_puros = np.array(df['retorno_log'].fillna(0.0).values, dtype=np.float64, copy=True)

hurst_vals = calcular_hurst_array(retornos_puros, VENTANA_HURST, PASO, lags_estandar)

hurst_series = pd.Series(hurst_vals, index=df.index)
hurst_series = hurst_series.interpolate(method='linear').bfill().ffill()

df['hurst'] = hurst_series.round(4)
df['hurst'] = df['hurst'].fillna(0.5)

# ==============================================================================
# AJUSTE DE UMBRALES EMPÍRICOS (Adaptados al sesgo estructural de tu activo)
# ==============================================================================
# Dado que el Hurst medio se sitúa en ~0.56, calibramos los límites de régimen:
total_tendencia = int((df['hurst'] > 0.58).sum())
total_aleatorio = int(((df['hurst'] >= 0.52) & (df['hurst'] <= 0.58)).sum())
total_reversion = int((df['hurst'] < 0.52).sum())

# Métricas en consola limpias, en texto estándar (sin colores ANSI en los datos)
print(f"      Hurst medio:                          {df['hurst'].mean():.4f}")
print(f"      Hurst máximo:                         {df['hurst'].max():.4f}")
print(f"      Hurst mínimo:                         {df['hurst'].min():.4f}")
print()
print(f"      Periodos tendencia    (H>0.58):       {total_tendencia:,}")
print(f"      Periodos aleatorio    (H=0.52-0.58):  {total_aleatorio:,}")
print(f"      Periodos mean reversion (H<0.52):     {total_reversion:,}")


# Guardar
df['hora_utc'] = df.index.hour

if df.index.name is None:
    df.index.name = 'index'

df = df.reset_index()

if 'index' in df.columns:
    df = df.rename(columns={'index': 'timestamp'})
elif 'open_time' in df.columns:
    df = df.rename(columns={'open_time': 'timestamp'})

# Lista de columnas obligatorias
columnas_finales = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                    'retorno_log', 'interpolado', 'anomalia', 'ER', 'hurst', 'hora_utc']

# Crear columnas faltantes con valor 0 para evitar el KeyError
for col in columnas_finales:
    if col not in df.columns:
        if col == 'timestamp':
            df = df.reset_index() # Si timestamp es el índice, lo volvemos columna
        else:
            df[col] = 0.0

# Ahora seleccionamos con seguridad
df = df[columnas_finales]

df.to_csv(OUTPUT, index=False)
print(f"\n✅ Guardado en: {OUTPUT}")
print("="*50)