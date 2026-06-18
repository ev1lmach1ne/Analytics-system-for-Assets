import pandas as pd
import numpy as np
import psycopg2

# ── CONFIGURACIÓN — solo cambia esto ─────────────────────
TABLA      = 'btc_candles_1h'
FRECUENCIA = '1h'
OUTPUT     = r'D:\DATOS\Activos\Crypto\Limpiados\btc_1h_limpio.csv'
# ─────────────────────────────────────────────────────────

print("="*50)
print(f"LIMPIEZA — {TABLA} ({FRECUENCIA})")
print("="*50)

# Conexión
conn = psycopg2.connect(
    host='localhost', port=18812,
    database='qdb', user='admin', password='quest'
)


# [1/5] Descarga
print("\n[1/5] Descargando datos...")
df = pd.read_sql(f"""
    SELECT timestamp, open, high, low, close, volume
    FROM {TABLA}
    ORDER BY timestamp ASC
""", conn)
conn.close()
print(f"      Filas originales: {len(df)}")


# [2/5] Detección de huecos
print("\n[2/5] Detectando huecos...")
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.set_index('timestamp')

duplicados = df.index.duplicated().sum()
print(f"      Duplicados detectados: {duplicados}")
df = df[~df.index.duplicated(keep='first')]

idx_completo = pd.date_range(
    start=df.index.min(),
    end=df.index.max(),
    freq=FRECUENCIA
)
huecos = idx_completo.difference(df.index)
print(f"      Huecos detectados: {len(huecos)} velas faltantes")


# [3/5] Detección de anomalías (Universal y Estándar)
print("\n[3/5] Detectando cambios erráticos en la recodiga de datos...")

diff_anterior = (df['close'] - df['close'].shift(1)).abs()
diff_siguiente = (df['close'] - df['close'].shift(-1)).abs()
rango_normal = (df['high'] - df['low']).rolling(window=24).mean()

f_salto = (diff_anterior > (rango_normal * 10)) & (diff_siguiente > (rango_normal * 10))

# Auditoría visual antes de reparar
anomalias_indices = df.index[f_salto]
if len(anomalias_indices) > 0:
    print(f"      >>> AUDITORÍA: Detectadas {len(anomalias_indices)} anomalías")
    cols_aud = ['close', 'volume'] if 'volume' in df.columns else ['close']
    for idx in anomalias_indices:
        # Contexto de -1 a +2 (incluye la fila anterior, la errónea y la posterior)
        start = max(0, df.index.get_loc(idx) - 1)
        end = min(len(df), df.index.get_loc(idx) + 2)
        print(f"\n      Contexto detectado para {idx}:")
        print(df.iloc[start:end][cols_aud])
else:
    print("      No se detectaron anomalías.")

# Registro y Reparación
df['anomalia'] = np.where(f_salto, df['close'], 0)
df.loc[f_salto, ['open', 'high', 'low', 'close']] = np.nan

# [4/5] Interpolación y consolidación de reparaciones
print("\n[4/5] Rellenando huecos y corrigiendo datos...")

# 1. Identificamos qué es nulo ANTES de rellenar (esto marca los huecos originales)
es_nulo_antes = df['close'].isna()

# 2. Interpolamos
for col in ['open', 'high', 'low', 'close', 'volume']:
    if col in df.columns:
        df[col] = df[col].interpolate(method='linear')

# 3. Rellenamos extremos que no se pudieron interpolar
df = df.bfill().ffill()

# 4. AQUI GUARDAMOS EL REGISTRO:
# Si el dato era nulo antes, le ponemos 1, si no, 0.
df['interpolado'] = es_nulo_antes.astype(int)

# 5. Auditoría
total_reparaciones = es_nulo_antes.sum()
print(f"      Total de celdas reparadas (huecos + anomalías): {total_reparaciones}")

# [5/5] Verificación
print("\n[5/5] Verificación final...")
nulos = df[['open','high','low','close','volume']].isna().sum().sum()
print(f"      Valores nulos restantes: {nulos}")
print(f"      {'✅ Dataset limpio' if nulos == 0 else '⚠️  Revisar nulos'}")

# [+] Efficiency Ratio
print("\n[+] Calculando Efficiency Ratio (ER)...")
PERIODO_ER = 10

movimiento_neto  = df['close'].diff(PERIODO_ER).abs()
movimiento_total = df['close'].diff().abs().rolling(PERIODO_ER).sum()

df['ER'] = (movimiento_neto / movimiento_total).round(4)
df['ER'] = df['ER'].fillna(0)

print(f"      ER medio:                           {df['ER'].mean():.4f}")
print(f"      ER máximo:                          {df['ER'].max():.4f}")
print(f"      ER mínimo:                          {df['ER'].min():.4f}")
print(f"      Periodos con ER > 0.5 (tendencia): {(df['ER'] > 0.5).sum()}")
print(f"      Periodos con ER < 0.3 (ruido):     {(df['ER'] < 0.3).sum()}")

# Guardar
print("\n[5/5] Guardando CSV...")
print(f"      Columnas antes de reset: {list(df.columns)}")
print(f"      Filas antes de reset: {len(df)}")
df = df.reset_index().rename(columns={'index': 'timestamp'})
print(f"      Columnas tras reset: {list(df.columns)}")
print(f"      Filas tras reset: {len(df)}")
df.to_csv(OUTPUT, index=False)
print(f"\n✅ Guardado en: {OUTPUT}")
print("="*50)