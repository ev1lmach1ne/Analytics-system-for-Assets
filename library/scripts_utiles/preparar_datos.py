import pandas as pd
from pathlib import Path
import subprocess
import os

# ── CONFIGURACIÓN ────────────────────────────────────────
CSV_INPUT  = r"D:\DATOS\Activos\Crypto\BNBUSDT-1m 08-2017_to_03-2026 - copia.csv"
SEPARADOR  = ','    

_p = Path(CSV_INPUT)
CSV_OUTPUT = str(_p.parent / f"{_p.stem}_preparado{_p.suffix}")

# Definición explícita de ALIAS_COLUMNAS
ALIAS_COLUMNAS = {
    'timestamp': ['time', 'date', 'datetime', 'ts', 'fecha', 'open_time', 'Date', 'Time', 'Datetime', 'Timestamp', 'index', 'Open_time'],
    'open':      ['Open', 'OPEN', 'o', 'precio_apertura', 'open_price'],
    'high':      ['High', 'HIGH', 'h', 'max', 'maximo', 'precio_maximo'],
    'low':       ['Low',  'LOW',  'l', 'min', 'minimo', 'precio_minimo'],
    'close':     ['Close','CLOSE','c', 'price', 'precio', 'last', 'ultimo', 'settle', 'settlement'],
    'volume':    ['Volume','VOLUME','vol','Vol','VOL','qty','quantity', 'amount', 'volumen'],
}

print("="*50)
print("PREPARACIÓN DE DATOS")
print("="*50)

# [1/4] Lectura
df = pd.read_csv(CSV_INPUT, sep=SEPARADOR, low_memory=False)
print(f"Filas iniciales: {len(df)}")

# [2/4] Normalización
df.columns = df.columns.str.strip().str.replace('\ufeff', '')
mapa_inverso = {alias.lower(): canonico for canonico, aliases in ALIAS_COLUMNAS.items() for alias in aliases}
renombrado = {col: mapa_inverso[col.lower()] for col in df.columns if col.lower() in mapa_inverso}
df = df.rename(columns=renombrado)

# [3/4] Conversión
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
df = df.dropna(subset=['timestamp'])
df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%S')

if 'volume' not in df.columns: df['volume'] = 0
df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

# [4/4] Guardar y abrir
df.to_csv(CSV_OUTPUT, index=False)
print(f"✅ Archivo listo: {CSV_OUTPUT}")