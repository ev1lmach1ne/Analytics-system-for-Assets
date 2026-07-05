# region ── 1. IMPORTS Y CONFIGURACIÓN ──────────────────────────
import pandas as pd
import numpy as np
import psycopg2
import json
import os
import warnings
warnings.filterwarnings('ignore')
from numba import njit
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from core.config import CONFIG_PATH, LIMPIADOS_DIR, DB_CONFIG


# ----------COLORES-----------
ROSA  = "\033[95m"
NEON  = "\033[1;95m"  # Rosa brillante / negrita
RESET = "\033[0m"
# ----------------------------
# ── CONFIGURACIÓN — solo cambia esto ─────────────────────
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH) as f:
        sesion = json.load(f)
    TABLA      = f"{sesion['nombre']}_candles_{sesion['tf']}"
    FRECUENCIA = sesion['tf']
    ACTIVO     = sesion['activo'].lower()
    print(f"↳ Config desde sesión: {TABLA}")
else:
    TABLA      = 'xauusd_candles_1h'
    FRECUENCIA = '1h'
    ACTIVO     = 'futuro'
    print("↳ Usando config manual (sin sesion_config.json)")

# Mapear timeframe de usuario ('1m','5m','1h','1d') a frecuencia pandas ('1min','5min','1H','1D')
_tf = FRECUENCIA.lower().strip()
import re as _re
_m = _re.match(r'(\d+)([a-z]+)', _tf)
if _m:
    _num, _unit = _m.groups()
    _mapa = {'s':'S','sec':'S','m':'min','min':'min','h':'H','d':'D','w':'W'}
    if _unit in _mapa:
        FRECUENCIA_PD = f"{_num}{_mapa[_unit]}"
    else:
        FRECUENCIA_PD = _tf
else:
    FRECUENCIA_PD = _tf
print(f"      Frecuencia pandas: {FRECUENCIA_PD}")

OUTPUT_DIR  = LIMPIADOS_DIR
nombre_activo = TABLA.split('_')[0]
OUTPUT = os.path.join(OUTPUT_DIR, nombre_activo, f"{nombre_activo}_{FRECUENCIA}_limpiado.csv")
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
# ─────────────────────────────────────────────────────────
# endregion

# region ── 2. CONEXIÓN Y DESCARGA ──────────────────────────────
print("="*50)
print(f"LIMPIEZA — {TABLA} ({FRECUENCIA})")
print("="*50)

# Conexión
conn = psycopg2.connect(**DB_CONFIG)


# ── [1/7] DESCARGA DE DATOS ──────────────────────────────────────────────────
print("\n[1/7] Descargando datos...")
try:
    df = pd.read_sql(f"""
        SELECT timestamp, open, high, low, close, volume, spread
        FROM {TABLA}
        ORDER BY timestamp ASC
    """, conn)
    print(f"      Columna 'spread' detectada en la tabla.")
except Exception:
    conn.rollback()
    df = pd.read_sql(f"""
        SELECT timestamp, open, high, low, close, volume
        FROM {TABLA}
        ORDER BY timestamp ASC
    """, conn)
    df['spread'] = 0.0
    print(f"      Tabla sin columna 'spread' — se rellena con 0.")
conn.close()
print(f"      Filas originales: {len(df)}")
# endregion

# region ── 3. DETECCIÓN DE COLUMNA DE TIEMPO ───────────────────
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
# endregion

# region ── 4. SINCRONIZACIÓN Y BAD TICKS ───────────────────────
# ── [3/8] SINCRONIZACIÓN Y LIMPIEZA (BLOQUE SUSTITUTO CON FILTRO ANTI-BAD TICKS) ──
print("\n[3/8] Sincronizando y limpiando...")

df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.set_index('timestamp')

# Estandarización UTC
if df.index.tz is None:
    df.index = df.index.tz_localize('UTC')
    df.index = df.index.tz_convert('UTC')
else:
    df.index = df.index.tz_convert('UTC')

df = df[df.index.notna()]
df = df[~df.index.duplicated(keep='first')]

# ==============================================================================
# DETECCIÓN DE BAD TICKS (Rolling Z-Score adaptativo)
# ==============================================================================
retorno_rolling = df['close'].pct_change()
vol_local = retorno_rolling.rolling(20).std()
media_local = retorno_rolling.rolling(20).mean()
es_bad_tick = (retorno_rolling - media_local).abs() > vol_local * 3
es_bad_tick = es_bad_tick.fillna(False)

anomalias_detectadas = es_bad_tick.sum()
if anomalias_detectadas > 0:
    print(f"⚠️ Detectadas {anomalias_detectadas} anomalías (rolling z-score > 3σ)")
    df.loc[es_bad_tick, ['open', 'high', 'low', 'close', 'volume', 'spread']] = np.nan
    df['anomalia'] = es_bad_tick.astype(int)
else:
    print("      Sin anomalías detectadas.")
# ==============================================================================
# endregion

# region ── 5. REINDEX Y FORWARD FILL ───────────────────────────
# ── [4/8] REINDEX A RANGO COMPLETO ────────────────────────────────────────────
print("\n[4/8] Reindexando a rango temporal completo...")
ts_min, ts_max = df.index.min(), df.index.max()
dias_totales = (ts_max - ts_min).days
print(f"      Rango timestamps: {ts_min} → {ts_max} ({dias_totales} días)")
_es_diario = bool(_re.match(r'\d+[dD]', FRECUENCIA_PD))
if _es_diario:
    idx_completo = pd.bdate_range(start=ts_min, end=ts_max)
else:
    # Intradiario: detectar patrón real de días y horas activas
    df_orig = df.copy()
    df_orig['_dow'] = df_orig.index.dayofweek
    df_orig['_time'] = df_orig.index.time

    # Días de la semana activos (cualquier día con al menos 3 observaciones)
    obs_por_dow = df_orig.groupby('_dow').size()
    min_obs = max(3, len(df_orig) * 0.001)
    dow_activos = sorted(obs_por_dow[obs_por_dow >= min_obs].index)

    # Horas activas (>80% de los días con datos)
    horas_por_dia = df_orig.groupby(df_orig.index.date)['_time'].apply(set)
    if len(horas_por_dia) >= 3:
        todas_las_horas = set().union(*horas_por_dia)
        recuento = {h: sum(1 for s in horas_por_dia if h in s) for h in todas_las_horas}
        umbral = len(horas_por_dia) * 0.8
        horas_activas = sorted([h for h, c in recuento.items() if c >= umbral])
    else:
        horas_activas = sorted(set().union(*horas_por_dia)) if horas_por_dia else []

    if horas_activas and dow_activos:
        # Todos los días del rango, filtrados por días activos
        todos_dias = pd.date_range(start=ts_min, end=ts_max, freq='D')
        dias_filtro = todos_dias[todos_dias.dayofweek.isin(dow_activos)]
        idx_naive = pd.DatetimeIndex([
            pd.Timestamp.combine(d.date(), h) for d in dias_filtro for h in horas_activas
        ]).sort_values()
        if df.index.tz is not None:
            idx_completo = idx_naive.tz_localize(df.index.tz)
        else:
            idx_completo = idx_naive
    else:
        idx_completo = pd.date_range(start=ts_min, end=ts_max, freq=FRECUENCIA_PD)
huecos = len(idx_completo) - len(df)
print(f"      Huecos temporales detectados: {huecos} velas faltantes")
df = df.reindex(idx_completo)

# ── [5/8] FORWARD FILL (sin interpolación lineal) ────────────────────────────
print("\n[5/8] Rellenando huecos con Forward Fill...")

# Precio 0 no es válido — tratarlo como NaN para que ffill propague el último real
for col in ['open', 'high', 'low', 'close']:
    if col in df.columns:
        df.loc[df[col] == 0, col] = np.nan

era_nulo = df['close'].isna()
total_reparaciones = era_nulo.sum()

for col in ['open', 'high', 'low', 'close', 'volume']:
    if col in df.columns:
        df[col] = df[col].ffill()

# Spread no se hace ffill (los huecos son reales), solo 0 si queda NaN
if 'spread' in df.columns:
    df['spread'] = df['spread'].fillna(0)

df['interpolado'] = era_nulo.astype(int)
total_reparaciones = era_nulo.sum()
total_filas = len(df)
total_reales = total_filas - total_reparaciones

# Desglose
etiqueta_origen = {'crypto': 'Exchange', 'futuro': 'Futuro', 'stock': 'Broker'}.get(ACTIVO.lower(), 'Origen')
print(f"      Total de filas rellenadas (huecos + anomalías): {total_reparaciones}")
print()
print(f" 📊 DESGLOSE DE LOS DATOS DEL ARCHIVO:")
print(f" ======================================================================")
print(f"    Total filas en el archivo:                    {total_filas:>10,}")
print(f"    ├── 🟢 Datos reales ({etiqueta_origen}):      {total_reales:>10,}  ({total_reales/total_filas:.2%})")
print(f"    └── 🔵 Datos rellenados (Script):             {total_reparaciones:>10,}  ({total_reparaciones/total_filas:.2%})")
if 'anomalia' in df.columns:
    total_anomalias = int(df['anomalia'].sum())
    total_huecos_puros = total_reparaciones - total_anomalias
    print(f"        ├── 🟡 Bad Ticks (anomalías):            {total_anomalias:>10,}  ({total_anomalias/total_filas:.2%})")
    print(f"        └── ⚪ Huecos temporales:                 {total_huecos_puros:>10,}  ({total_huecos_puros/total_filas:.2%})")
print(f" ======================================================================\n")
# endregion

# region ── 6. VERIFICACIÓN Y RETORNOS ──────────────────────────
# ── [6/8] VERIFICACIÓN FINAL ────────────────────────────────────────────────
print("\n[6/8] Verificación final...")
nulos = df[['open','high','low','close','volume']].isna().sum().sum()
print(f"      Valores nulos restantes: {nulos}")
print(f"      {'✅ Dataset limpio' if nulos == 0 else '⚠️  Revisar nulos'}")

# [+] Retornos logarítmicos
print("\n[+] Calculando retornos logarítmicos...")
df['retorno_log'] = np.log(df['close'] / df['close'].shift(1)).round(8)
# endregion

# region ── 7. EFFICIENCY RATIO (ER) ────────────────────────────
# [+] Efficiency Ratio (ER)
PERIODO_ER = 10
print(f"\n[7/8] Calculando Efficiency Ratio (ER) utilizando ventana de {NEON}{PERIODO_ER}{RESET} periodos...")

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
# endregion

# region ── 8. EXPONENTE DE HURST (Numba) ───────────────────────
# [+] Exponente de Hurst con Numba

VENTANA_HURST = 1024
PASO          = 10
lags_estandar = np.array([16, 32, 64, 128, 256], dtype=np.int64)

if FRECUENCIA.endswith('d'):
    VENTANA_HURST = 504
    PASO = 5
    lags_estandar = np.array([16, 32, 64, 128], dtype=np.int64)
elif FRECUENCIA.endswith('w'):
    VENTANA_HURST = 256
    PASO = 2
    lags_estandar = np.array([8, 16, 32, 64, 128], dtype=np.int64)

print(f"\n[8/8] Calculando Exponente de Hurst (Numba) utilizando ventana de {NEON}{VENTANA_HURST}{RESET} periodos y paso de {NEON}{PASO}{RESET} periodos...")

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
# endregion

# region ── 9. GUARDADO FINAL ───────────────────────────────────
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
columnas_finales = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'spread',
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
# endregion