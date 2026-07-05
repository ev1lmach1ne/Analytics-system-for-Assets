import psycopg2
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.config import DB_CONFIG

# ── FUNCIÓN DE DESCARGA ───────────────────────────────────
def get_data(tabla, fecha_inicio, fecha_fin):
    """
    Descarga datos OHLCV de QuestDB.
    
    Parámetros:
        tabla        : nombre de la tabla en QuestDB
                       ej. 'btc_candles_1h', 'btc_candles_5m'
        fecha_inicio : string con fecha de inicio 'YYYY-MM-DD'
        fecha_fin    : string con fecha de fin    'YYYY-MM-DD'
    
    Devuelve:
        DataFrame con columnas: timestamp, open, high, low, close, volume
    """
    conn = psycopg2.connect(**DB_CONFIG)
    
    query = f"""
        SELECT timestamp, open, high, low, close, volume
        FROM {tabla}
        WHERE timestamp >= '{fecha_inicio}'
          AND timestamp <= '{fecha_fin}'
        ORDER BY timestamp ASC
    """
    
    df = pd.read_sql(query, conn)
    conn.close()
    
    print(f"Tabla: {tabla} | Filas: {len(df)} | {fecha_inicio} → {fecha_fin}")
    return df