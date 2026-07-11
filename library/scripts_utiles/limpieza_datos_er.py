# region ── 1. IMPORTS Y CONFIGURACIÓN ──────────────────────────
import pandas as pd
import numpy as np
import psycopg2
import json
import os
import warnings
warnings.filterwarnings('ignore')
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from core.config import CONFIG_PATH, LIMPIADOS_DIR, DB_CONFIG
from core.metrics import calcular_umbrales_er, contar_regimen_hurst, hurst_rs_numba, calcular_hurst_array


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

    # Horas activas (>80% de los días con datos) — vectorizado con unstack
    _fecha_idx = pd.Index(df_orig.index.date)
    _horas_matriz = df_orig.groupby([_fecha_idx, '_time']).size().unstack(fill_value=0)
    _presencia = _horas_matriz > 0
    if len(_presencia) >= 3:
        _recuento = _presencia.sum()
        _umbral = len(_presencia) * 0.8
        horas_activas = sorted([h for h in _presencia.columns if _recuento[h] >= _umbral])
    else:
        horas_activas = sorted(_presencia.columns.tolist()) if len(_presencia) > 0 else []

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
# CÁLCULO DE UMBRALES DINÁMICOS (1ª Desviación Estándar) — core.metrics
# ------------------------------------------------------------------------------
_umbrales_er = calcular_umbrales_er(df['ER'])
er_medio         = _umbrales_er['er_medio']
er_std           = _umbrales_er['er_std']
umbral_tendencia = _umbrales_er['umbral_tendencia']
umbral_ruido     = _umbrales_er['umbral_ruido']

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

# hurst_rs_numba y calcular_hurst_array viven en core.metrics (importadas arriba)
retornos_puros = np.array(df['retorno_log'].fillna(0.0).values, dtype=np.float64, copy=True)

hurst_vals = calcular_hurst_array(retornos_puros, VENTANA_HURST, PASO, lags_estandar)

hurst_series = pd.Series(hurst_vals, index=df.index)
hurst_series = hurst_series.interpolate(method='linear').bfill().ffill()

df['hurst'] = hurst_series.round(4)
df['hurst'] = df['hurst'].fillna(0.5)

# ==============================================================================
# AJUSTE DE UMBRALES EMPÍRICOS (Adaptados al sesgo estructural de tu activo) — core.metrics
# ==============================================================================
# Dado que el Hurst medio se sitúa en ~0.56, calibramos los límites de régimen:
_regimen_hurst  = contar_regimen_hurst(df['hurst'])
total_tendencia = _regimen_hurst['total_tendencia']
total_aleatorio = _regimen_hurst['total_aleatorio']
total_reversion = _regimen_hurst['total_reversion']

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