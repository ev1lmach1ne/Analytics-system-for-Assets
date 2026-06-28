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
# 4. Señales de entrada/salida (long + short)
# ---------------------------------------------------------
# Largos: reversión desde banda inferior hacia la media
long_entries = close < lower
long_exits   = close > middle

# Cortos: reversión desde banda superior hacia la media
short_entries = close > upper
short_exits   = close < middle

# ---------------------------------------------------------
# 5. Stop-loss dinámico ATR * 1.8
# ---------------------------------------------------------
# Para largos: stop por debajo del precio
long_stop = close - atr.atr * 1.8

# Para cortos: stop por encima del precio
short_stop = close + atr.atr * 1.8

# ---------------------------------------------------------
# 6. Gestión monetaria (0,5% del capital por trade)
# ---------------------------------------------------------
initial_capital = 25000
usd_per_trade = initial_capital * 0.005  # 0.5% = 125 USD

# tamaño en unidades de BTC (positivo para largos, negativo para cortos)
long_size =  usd_per_trade / close
short_size = -usd_per_trade / close

# ---------------------------------------------------------
# 7. Backtest long + short
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
# 8. Resultados
# ---------------------------------------------------------
print(portfolio.stats())

# ---------------------------------------------------------
# 9. Gráfica
# ---------------------------------------------------------
portfolio.plot().show()
