import pandas as pd
print("Iniciando conversión...")

# Leer CSV sin cabecera, forzar tipos y ignorar errores
df = pd.read_csv(
    r'D:\DATOS\Activos\Crypto\BTCUSDT 5m.csv',
    header=None,
    names=['timestamp','open','high','low','close','volume',
           'close_timestamp','quote_asset_volume','number_of_trades',
           'taker_buy_base_asset_volume','taker_buy_quote_asset_volume'],
    dtype={'timestamp': str, 'close_timestamp': str},
    low_memory=False
)
print(f"CSV leído — {len(df)} filas")

# Eliminar filas donde timestamp no es numérico (cabeceras repetidas)
df = df[pd.to_numeric(df['timestamp'], errors='coerce').notna()]
print(f"Filas limpias — {len(df)} filas")

# Convertir milisegundos a datetime
df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='ms')
df['close_timestamp'] = pd.to_datetime(df['close_timestamp'].astype(float), unit='ms')
print("Fechas convertidas")

# Formatear sin milisegundos
df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
df['close_timestamp'] = df['close_timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
print("Formato aplicado")

# Guardar
df.to_csv(r'D:\DATOS\Activos\Crypto\BTCUSDT_5m_clean.csv', index=False)
print(f"Listo — {len(df)} filas convertidas")