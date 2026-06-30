import pandas as pd
import numpy as np
import psycopg2
from pathlib import Path
import os
import csv
import sys
sys.stdout.reconfigure(encoding='utf-8')

# ── CONFIGURACIÓN ────────────────────────────────────────
CSV_INPUT     = r"D:\DATOS\Activos\XAUUSD_M15_202103220700_202506122215.csv"
NOMBRE_ACTIVO = 'XAUUSD'      # nombre corto del activo (ej: btc, eth, bnb, sol)
TIMEFRAME     = '15m'       # 1m, 5m, 15m, 30m, 1h, 4h, 1d
QUESTDB_HOST  = 'localhost'
QUESTDB_PORT  = 18812

TABLA_DESTINO = f"{NOMBRE_ACTIVO}_candles_{TIMEFRAME}"

_p = Path(CSV_INPUT)
CSV_OUTPUT = str(_p.parent / f"{_p.stem}_preparado{_p.suffix}")

# Definición explícita de ALIAS_COLUMNAS
ALIAS_COLUMNAS = {
    'timestamp': ['time', 'date', 'datetime', 'ts', 'fecha', 'open_time', 'Date', 'Time', 'Datetime', 'Timestamp', 'index', 'Open_time'],
    'open':      ['Open', 'OPEN', 'o', 'precio_apertura', 'open_price'],
    'high':      ['High', 'HIGH', 'h', 'max', 'maximo', 'precio_maximo'],
    'low':       ['Low',  'LOW',  'l', 'min', 'minimo', 'precio_minimo'],
    'close':     ['Close','CLOSE','c', 'price', 'precio', 'last', 'ultimo', 'settle', 'settlement'],
    'volume':    ['Volume','VOLUME','vol','Vol','VOL','qty','quantity', 'amount', 'volumen', 'tickvol', 'TICKVOL'],
    'spread':    ['Spread', 'SPREAD', 'spread'],
}

print("="*50)
print("PREPARACIÓN DE DATOS")
print("="*50)

# Detectar separador automáticamente
with open(CSV_INPUT, 'r', encoding='utf-8') as f:
    primera_linea = f.readline()
    # Probar separadores comunes y elegir el que más columnas genere
    SEPARADOR = max(['\t', ',', ';', '|', ' '], key=lambda s: len(primera_linea.split(s)))
print(f"Separador detectado: {repr(SEPARADOR)}")

# [1/4] Lectura
df = pd.read_csv(CSV_INPUT, sep=SEPARADOR, low_memory=False)
print(f"Filas iniciales: {len(df)}")

# [2/4] Normalización — limpiar nombres de columna
df.columns = df.columns.str.strip().str.replace('\ufeff', '').str.replace('<', '').str.replace('>', '')
print(f"      Columnas detectadas: {list(df.columns)}")

# Combinar DATE + TIME si vienen separadas (formato MT4) antes del renombrado
cols_upper = {c.upper(): c for c in df.columns}
if 'DATE' in cols_upper and 'TIME' in cols_upper:
    col_date = cols_upper['DATE']
    col_time = cols_upper['TIME']
    df['timestamp'] = pd.to_datetime(df[col_date].astype(str) + ' ' + df[col_time].astype(str), errors='coerce')
    df = df.drop(columns=[col_date, col_time])
    print(f"      Formato MT4 detectado — columnas DATE+TIME combinadas en 'timestamp'")

# Renombrar alias canónicos
mapa_inverso = {alias.lower(): canonico for canonico, aliases in ALIAS_COLUMNAS.items() for alias in aliases}
renombrado = {col: mapa_inverso[col.lower()] for col in df.columns if col.lower() in mapa_inverso}
df = df.rename(columns=renombrado)

# Consolidar columnas duplicadas (ej: TICKVOL + VOL → volume): sumarlas
df = df.loc[:, ~df.columns.duplicated(keep='first')]

# [3/4] Conversión
if 'timestamp' not in df.columns:
    # Buscar cualquier columna que parezca una fecha
    posibles_ts = [c for c in df.columns if any(k in c.lower() for k in ['date', 'time', 'fecha', 'datetime'])]
    if posibles_ts:
        df['timestamp'] = pd.to_datetime(df[posibles_ts[0]], errors='coerce')
        df = df.drop(columns=[c for c in posibles_ts if c != 'timestamp' and c in df.columns])
        print(f"      Columna '{posibles_ts[0]}' usada como timestamp")
    else:
        # Usar la primera columna como timestamp e intentar parsear
        primera_col = df.columns[0]
        df['timestamp'] = pd.to_datetime(df[primera_col], errors='coerce')
        print(f"      Intentando usar columna '{primera_col}' como timestamp")
        if df['timestamp'].isna().all():
            print(f"      ⚠️ No se pudo parsear '{primera_col}' como fecha. Usando índice numérico.")
            df['timestamp'] = pd.Timestamp.now().normalize() + pd.to_timedelta(range(len(df)), unit='h')

df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
df = df.dropna(subset=['timestamp'])
df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%S')

if 'volume' not in df.columns: df['volume'] = 0
if 'spread' not in df.columns: df['spread'] = 0
cols_finales = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'spread']
df = df[[c for c in cols_finales if c in df.columns]]

# [4/4] Guardar y abrir
df.to_csv(CSV_OUTPUT, index=False)
print(f"✅ Archivo listo: {CSV_OUTPUT}")

# [5/5] Subir a QuestDB
print(f"\n[5/5] Subiendo {len(df):,} filas a QuestDB — Tabla: {TABLA_DESTINO} ...")
try:
    conn = psycopg2.connect(
        host=QUESTDB_HOST, port=QUESTDB_PORT,
        database='qdb', user='admin', password='quest'
    )
    cur = conn.cursor()

    cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
    tablas_existentes = [r[0].lower() for r in cur.fetchall()]
    existe = TABLA_DESTINO.lower() in tablas_existentes

    if existe:
        print(f"      ↳ La tabla '{TABLA_DESTINO}' ya existe — se omite la subida.")
        print(f"      ↳ Si quieres actualizarla, bórrala antes desde la consola de QuestDB.")
    else:
        tiene_spread = 'spread' in df.columns
        cur.execute(f"""
            CREATE TABLE {TABLA_DESTINO} (
                timestamp TIMESTAMP,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume DOUBLE
                {', spread DOUBLE' if tiene_spread else ''}
            )
        """)
        conn.commit()

        BATCH = max(1000, min(50000, len(df) // 20))
        total = len(df)
        cols_insert = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        if tiene_spread:
            cols_insert.append('spread')
        placeholders = ','.join(['%s'] * len(cols_insert))
        sql = f"INSERT INTO {TABLA_DESTINO} VALUES ({placeholders})"
        n_batches = (total + BATCH - 1) // BATCH
        for batch_idx, i in enumerate(range(0, total, BATCH), 1):
            batch = df.iloc[i:i+BATCH]
            records = []
            for _, row in batch.iterrows():
                rec = []
                for c in cols_insert:
                    v = row[c]
                    if isinstance(v, (np.integer,)):
                        rec.append(int(v))
                    elif isinstance(v, (np.floating,)):
                        rec.append(float(v))
                    else:
                        rec.append(v)
                records.append(tuple(rec))
            cur.executemany(sql, records)
            conn.commit()
            avanzado = min(i + BATCH, total)
            pct = batch_idx / n_batches
            llenos = int(round(pct * 20))
            barra_p = '█' * llenos + '░' * (20 - llenos)
            print(f"\r      Progreso: |{barra_p}| {avanzado:>6,} / {total:,}", end='')
        print(f"\n✅ Subida completada — {total:,} filas en {TABLA_DESTINO}")

    cur.close()
    conn.close()
except Exception as e:
    print(f"⚠️  Error al subir a QuestDB: {e}")
    print("   El archivo preparado se ha guardado igualmente.")