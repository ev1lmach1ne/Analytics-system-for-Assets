import pandas as pd
from sqlalchemy import create_engine
import vectorbt as vbt

# ---------------------------------------------------------
# 1. Conexión a QuestDB
# ---------------------------------------------------------
engine = create_engine("postgresql://localhost:19000/qdb")

# ---------------------------------------------------------
# 2. Cargar datos
# ---------------------------------------------------------
query = """
SELECT timestamp, open, high, low, close, volume
FROM btc_candles_15m
ORDER BY timestamp
"""

df = pd.read_sql(query, engine)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.set_index('timestamp')

close = df['close']
high = df['high']
low = df['low']

# ---------------------------------------------------------
# 3. Indicadores: Bollinger Bands + ATR
# ---------------------------------------------------------
bb = vbt.BBANDS.run(close, window=20, std=2)
atr = vbt.ATR.run(high, low, close, window=14)

lower = bb.lower
middle = bb.middle
upper = bb.upper

# ---------------------------------------------------------
# 4. Señales de entrada (long + short)
# ---------------------------------------------------------
long_entries = close < lower
short_entries = close > upper

# ---------------------------------------------------------
# 5. Stop-loss ATR * 1.8
# ---------------------------------------------------------
long_stop = close - atr.atr * 1.8
short_stop = close + atr.atr * 1.8

# ---------------------------------------------------------
# 6. Take Profit 1:1.5 RR
# ---------------------------------------------------------
# Distancia de riesgo
long_R = close - long_stop
short_R = short_stop - close

# Objetivo 1.5R
long_tp = close + 1.5 * long_R
short_tp = close - 1.5 * short_R

# ---------------------------------------------------------
# 7. Señales de salida
# ---------------------------------------------------------
# Salida por volver a la media
long_exit_mid = close > middle
short_exit_mid = close < middle

# Salida por TP 1.5R
long_exit_tp = close >= long_tp
short_exit_tp = close <= short_tp

# Salida final: cualquiera que ocurra primero
long_exits = long_exit_mid | long_exit_tp
short_exits = short_exit_mid | short_exit_tp

# ---------------------------------------------------------
# 8. Gestión monetaria (0,5% del capital por trade)
# ---------------------------------------------------------
initial_capital = 25000
usd_per_trade = initial_capital * 0.005  # 0.5% = 125 USD

long_size =  usd_per_trade / close
short_size = -usd_per_trade / close

# ---------------------------------------------------------
# 9. Backtest
# ---------------------------------------------------------
portfolio = vbt.Portfolio.from_signals(
    close,
    entries=long_entries,
    exits=long_exits,
    short_entries=short_entries,
    short_exits=short_exits,
    size=long_size,
    short_size=short_size,
    init_cash=initial_capital,
    fees=0.0005,
    slippage=0.0002,
    stop_price=long_stop,
    short_stop_price=short_stop
)

# ---------------------------------------------------------
# 10. Resultados
# ---------------------------------------------------------
print(portfolio.stats())

# ---------------------------------------------------------
# 11. Gráfica
# ---------------------------------------------------------
portfolio.plot().show()
