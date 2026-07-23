import pandas as pd
import numpy as np
from questdb.ingress import Sender
import psycopg2
print("Conectando a QuestDB...")

# Conexión a QuestDB via PostgreSQL wire protocol
conn = psycopg2.connect(
    host='localhost',
    port=18812,
    database='qdb',
    user='admin',
    password='quest'
)

# Cargar datos
print("Cargando datos de btc_candles_1h...")
df = pd.read_sql("""
    SELECT timestamp, open, high, low, close, volume
    FROM btc_candles_1h
    ORDER BY timestamp ASC
""", conn)
conn.close()
print(f"Filas cargadas: {len(df)}")

# ── INDICADORES ──────────────────────────────────────────
# Bollinger Bands (20, 2)
df['sma20']   = df['close'].rolling(20).mean()
df['std20']   = df['close'].rolling(20).std()
df['bb_upper'] = df['sma20'] + 2 * df['std20']
df['bb_lower'] = df['sma20'] - 2 * df['std20']

# ATR (14)
df['tr'] = np.maximum(
    df['high'] - df['low'],
    np.maximum(
        abs(df['high'] - df['close'].shift(1)),
        abs(df['low']  - df['close'].shift(1))
    )
)
df['atr14'] = df['tr'].rolling(14).mean()

# ── BACKTEST ─────────────────────────────────────────────
capital      = 10000  # USD inicial
position     = 0      # 0 = sin posición, 1 = long
entry_price  = 0
stop_loss    = 0
trades       = []

for i in range(20, len(df)):
    row  = df.iloc[i]
    prev = df.iloc[i-1]

    # ENTRADA — precio toca banda inferior
    if position == 0 and prev['close'] <= prev['bb_lower']:
        position    = 1
        entry_price = row['open']
        stop_loss   = entry_price - 2 * row['atr14']
        entry_time  = row['timestamp']

    # SALIDA — precio toca banda superior
    elif position == 1 and prev['close'] >= prev['bb_upper']:
        exit_price = row['open']
        pnl        = (exit_price - entry_price) / entry_price * 100
        capital   *= (1 + pnl / 100)
        trades.append({
            'entry_time':  entry_time,
            'exit_time':   row['timestamp'],
            'entry_price': entry_price,
            'exit_price':  exit_price,
            'stop_loss':   stop_loss,
            'pnl_pct':     round(pnl, 4),
            'capital':     round(capital, 2),
            'resultado':   'WIN' if pnl > 0 else 'LOSS'
        })
        position = 0

    # STOP LOSS
    elif position == 1 and row['low'] <= stop_loss:
        exit_price = stop_loss
        pnl        = (exit_price - entry_price) / entry_price * 100
        capital   *= (1 + pnl / 100)
        trades.append({
            'entry_time':  entry_time,
            'exit_time':   row['timestamp'],
            'entry_price': entry_price,
            'exit_price':  exit_price,
            'stop_loss':   stop_loss,
            'pnl_pct':     round(pnl, 4),
            'capital':     round(capital, 2),
            'resultado':   'STOP LOSS'
        })
        position = 0

# ── RESULTADOS ───────────────────────────────────────────
results = pd.DataFrame(trades)
print(f"\n{'='*50}")
print(f"RESULTADOS DEL BACKTEST")
print(f"{'='*50}")
print(f"Total trades:      {len(results)}")
print(f"Wins:              {len(results[results['resultado']=='WIN'])}")
print(f"Losses:            {len(results[results['resultado']=='LOSS'])}")
print(f"Stop losses:       {len(results[results['resultado']=='STOP LOSS'])}")
print(f"Win rate:          {len(results[results['resultado']=='WIN'])/len(results)*100:.1f}%")
print(f"Capital inicial:   $10,000")
print(f"Capital final:     ${capital:,.2f}")
print(f"Retorno total:     {(capital-10000)/10000*100:.2f}%")
print(f"PnL medio/trade:   {results['pnl_pct'].mean():.4f}%")
print(f"Mejor trade:       {results['pnl_pct'].max():.2f}%")
print(f"Peor trade:        {results['pnl_pct'].min():.2f}%")
print(f"{'='*50}")

# Guardar trades a CSV
results.to_csv(r'D:\Proyectos\Analytics-system-for-Assets\backtest_resultados_1h.csv', index=False)
print(f"\nResultados guardados en backtest_resultados_1h.csv")