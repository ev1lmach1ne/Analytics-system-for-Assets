import sys
import io
import psycopg2
import pandas as pd

# Forzar codificación del sistema
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')

# Conexión forzada evitando configuraciones locales
try:
    conn = psycopg2.connect(
        "host=localhost dbname=crypto_trading user=postgres password=1368 port=5432"
    )
    print("Conexión establecida con éxito.")
except Exception as e:
    print(f"Error crítico de conexión: {e}")
    sys.exit(1)

# Consulta de huecos
query = """
SELECT fecha_actual, siguiente_fecha, (siguiente_fecha - fecha_actual) as hueco
FROM (
    SELECT timestamp as fecha_actual, 
           LEAD(timestamp) OVER (ORDER BY timestamp) as siguiente_fecha
    FROM btc_candles_15m
) sub
WHERE (siguiente_fecha - fecha_actual) > interval '15 minutes';
"""

# Ejecutar y mostrar
print("Buscando huecos en la base de datos...")
df_huecos = pd.read_sql(query, conn)

if df_huecos.empty:
    print("¡Felicidades! No se encontraron huecos. Los datos son continuos.")
else:
    print(f"Se han encontrado {len(df_huecos)} huecos:")
    print(df_huecos)

conn.close()