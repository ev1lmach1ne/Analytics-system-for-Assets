import pandas as pd
import numpy as np
import psycopg2
import plotly.graph_objects as go
print("Conectando a QuestDB...")

# Conexión a QuestDB
conn = psycopg2.connect(
    host='localhost',
    port=18812,
    database='qdb',
    user='admin',
    password='quest'
)

# Cargar datos — limitamos a 500 velas para que el gráfico sea legible
print("Cargando datos...")
df = pd.read_sql("""
    SELECT timestamp, open, high, low, close, volume
    FROM btc_candles_1h
    ORDER BY timestamp ASC
    LIMIT 500
""", conn)
conn.close()

# ── INDICADORES ──────────────────────────────────────────
df['sma20']    = df['close'].rolling(20).mean()
df['std20']    = df['close'].rolling(20).std()
df['bb_upper'] = df['sma20'] + 2 * df['std20']
df['bb_lower'] = df['sma20'] - 2 * df['std20']

df['tr'] = np.maximum(
    df['high'] - df['low'],
    np.maximum(
        abs(df['high'] - df['close'].shift(1)),
        abs(df['low']  - df['close'].shift(1))
    )
)
df['atr14'] = df['tr'].rolling(14).mean()

# ── SEÑALES ───────────────────────────────────────────────
entries    = []
exits      = []
stop_hits  = []
position   = 0
entry_price = 0
stop_loss   = 0

for i in range(20, len(df)):
    row  = df.iloc[i]
    prev = df.iloc[i-1]

    if position == 0 and prev['close'] <= prev['bb_lower']:
        position    = 1
        entry_price = row['open']
        stop_loss   = entry_price - 2 * row['atr14']
        entries.append({'timestamp': row['timestamp'], 'price': entry_price})

    elif position == 1 and prev['close'] >= prev['bb_upper']:
        exits.append({'timestamp': row['timestamp'], 'price': row['open']})
        position = 0

    elif position == 1 and row['low'] <= stop_loss:
        stop_hits.append({'timestamp': row['timestamp'], 'price': stop_loss})
        position = 0

entries_df   = pd.DataFrame(entries)
exits_df     = pd.DataFrame(exits)
stop_hits_df = pd.DataFrame(stop_hits)

# ── GRÁFICO ───────────────────────────────────────────────
fig = go.Figure()

# Velas OHLCV
fig.add_trace(go.Candlestick(
    x=df['timestamp'],
    open=df['open'],
    high=df['high'],
    low=df['low'],
    close=df['close'],
    name='BTC 1h',
    increasing_line_color='#1D9E75',
    decreasing_line_color='#E24B4A'
))

# Bollinger Bands
fig.add_trace(go.Scatter(
    x=df['timestamp'], y=df['bb_upper'],
    name='BB Superior', line=dict(color='rgba(24,95,165,0.6)', width=1, dash='dash')
))
fig.add_trace(go.Scatter(
    x=df['timestamp'], y=df['sma20'],
    name='SMA 20', line=dict(color='rgba(24,95,165,0.4)', width=1)
))
fig.add_trace(go.Scatter(
    x=df['timestamp'], y=df['bb_lower'],
    name='BB Inferior', line=dict(color='rgba(24,95,165,0.6)', width=1, dash='dash'),
    fill='tonexty', fillcolor='rgba(24,95,165,0.05)'
))

# Entradas (flechas verdes hacia arriba)
if not entries_df.empty:
    fig.add_trace(go.Scatter(
        x=entries_df['timestamp'],
        y=entries_df['price'] * 0.995,
        mode='markers',
        name='Entrada',
        marker=dict(
            symbol='triangle-up',
            size=12,
            color='#1D9E75',
            line=dict(color='white', width=1)
        )
    ))

# Salidas (flechas rojas hacia abajo)
if not exits_df.empty:
    fig.add_trace(go.Scatter(
        x=exits_df['timestamp'],
        y=exits_df['price'] * 1.005,
        mode='markers',
        name='Salida',
        marker=dict(
            symbol='triangle-down',
            size=12,
            color='#185FA5',
            line=dict(color='white', width=1)
        )
    ))

# Stop losses (X roja)
if not stop_hits_df.empty:
    fig.add_trace(go.Scatter(
        x=stop_hits_df['timestamp'],
        y=stop_hits_df['price'],
        mode='markers',
        name='Stop Loss',
        marker=dict(
            symbol='x',
            size=12,
            color='#E24B4A',
            line=dict(color='white', width=1)
        )
    ))

# Layout
fig.update_layout(
    title='BTC 1h — Backtest Bollinger + ATR (primeras 500 velas)',
    xaxis_title='Fecha',
    yaxis_title='Precio (USDT)',
    xaxis_rangeslider_visible=False,
    template='plotly_dark',
    height=700,
    legend=dict(orientation='h', y=1.02)
)

fig.show()
print("Gráfico abierto en el navegador.")