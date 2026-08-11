"""
Proveedor de datos de exchanges de criptomonedas usando ccxt.

Usa Binance como exchange por defecto (datos publicos, sin API key).
ccxt soporta 1m, 5m, 15m, 1h, 4h, 1d nativamente.
"""

import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import calendar
import io
import json
import os
import tempfile
import threading
import time
import urllib.request
import zipfile

from .base_provider import BaseProvider, AssetInfo

CCXT_TF_MAP = {
    '1m': '1m', '5m': '5m', '15m': '15m',
    '1h': '1h', '4h': '4h', '1d': '1d',
}

_TF_MINUTES = {'1m': 1, '5m': 5, '15m': 15, '1h': 60, '4h': 240, '1d': 1440}
_PARALLEL_WORKERS = 4

# Binance limita el histórico accesible vía API según el timeframe: pide '1d'
# con since=0 para descubrir el first candle, pero para TFs intradiarios muy
# finos (1m, 5m) la API no conserva todo el histórico desde el listing del
# activo — solo varios meses a ~1-2 años atrás.  Sin este límite, calcular
# el rango con la primera vela '1d' (ej. 2018 para TRX) hace que los workers
# paralelos pidan 1m desde 2018, reciban vacío, y solo el último worker traiga
# datos recientes → huecos masivos al reindexar en la limpieza.
#
# Valores en número de velas. None = sin límite (histórico completo disponible).
_BINANCE_MAX_LOOKBACK = {
    '1m':  525600,   # ~1 año
    '5m':  1051200,  # ~2 años
    '15m': 1576800,  # ~3 años
    '1h':  None,      # histórico completo
    '4h':  None,
    '1d':  None,
}

# TFs que usan data.binance.vision (ZIPs mensuales/diarios) en lugar de la
# API ccxt, porque la API no sirve histórico completo para estos TFs.
_ARCHIVE_TF = {'1m', '5m', '15m'}
_ARCHIVE_BASE = 'https://data.binance.vision'

_CACHE_CATALOG = None
_CACHE_TIMESTAMP = None
_CACHE_TTL = 3600  # 1 hora
_RANGE_CACHE = {}  # symbol -> (first_dt, end_dt)

_CACHE_DIR = os.path.join(tempfile.gettempdir(), 'analytics_cache')
_CACHE_FILE = 'ccxt_catalog.json'

def _catalog_cache_path():
    return os.path.join(_CACHE_DIR, _CACHE_FILE)

def _save_catalog_to_disk(catalog):
    os.makedirs(_CACHE_DIR, exist_ok=True)
    data = {
        '_timestamp': time.time(),
        '_ttl': _CACHE_TTL,
        'assets': [
            {'symbol': a.symbol, 'name': a.name,
             'category': a.category, 'max_history_start': a.max_history_start}
            for a in catalog
        ],
    }
    with open(_catalog_cache_path(), 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)

def _load_catalog_from_disk():
    path = _catalog_cache_path()
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        ts = data.get('_timestamp', 0)
        ttl = data.get('_ttl', _CACHE_TTL)
        if time.time() - ts > ttl:
            return None
        return [AssetInfo(**a) for a in data['assets']]
    except Exception:
        return None

_SEPARATOR_SYMBOL = '__SEPARATOR__'

_TOP_PAIRS = [
    AssetInfo('BTC/USDT', 'Bitcoin', 'Spot Crypto', '2017-08-01'),
    AssetInfo('ETH/USDT', 'Ethereum', 'Spot Crypto', '2017-08-01'),
    AssetInfo('BNB/USDT', 'BNB', 'Spot Crypto', '2018-07-01'),
    AssetInfo('SOL/USDT', 'Solana', 'Spot Crypto', '2020-08-01'),
    AssetInfo('XRP/USDT', 'Ripple', 'Spot Crypto', '2018-05-01'),
    AssetInfo('ADA/USDT', 'Cardano', 'Spot Crypto', '2018-04-01'),
    AssetInfo('DOGE/USDT', 'Dogecoin', 'Spot Crypto', '2019-07-01'),
    AssetInfo('AVAX/USDT', 'Avalanche', 'Spot Crypto', '2020-09-01'),
    AssetInfo('DOT/USDT', 'Polkadot', 'Spot Crypto', '2020-08-01'),
    AssetInfo('MATIC/USDT', 'Polygon', 'Spot Crypto', '2019-04-01'),
    AssetInfo('LINK/USDT', 'Chainlink', 'Spot Crypto', '2019-01-01'),
    AssetInfo('UNI/USDT', 'Uniswap', 'Spot Crypto', '2020-09-01'),
    AssetInfo('ATOM/USDT', 'Cosmos', 'Spot Crypto', '2020-01-01'),
    AssetInfo('LTC/USDT', 'Litecoin', 'Spot Crypto', '2018-01-01'),
    AssetInfo('ETH/BTC', 'Ethereum / Bitcoin', 'Spot Crypto', '2017-08-01'),
]
_TOP_SYMBOLS = {a.symbol for a in _TOP_PAIRS}


# Mercados compartidos entre workers: load_markets() es una petición pesada y
# repetirla por worker es inútil (y en descargas largas cada fallo puntual de
# red/rate-limit mataría todo). Ver el mismo patrón en hyperliquid_provider.
_MARKETS_CACHE = None
_MARKETS_LOCK = threading.Lock()

_MAX_REINTENTOS = 5


def _obtener_markets():
    global _MARKETS_CACHE
    with _MARKETS_LOCK:
        if _MARKETS_CACHE is None:
            import ccxt
            exchange = ccxt.binance({'enableRateLimit': True})
            for intento in range(_MAX_REINTENTOS):
                try:
                    exchange.load_markets()
                    break
                except (ccxt.RateLimitExceeded, ccxt.NetworkError):
                    if intento == _MAX_REINTENTOS - 1:
                        raise
                    time.sleep(2 ** (intento + 1))
            _MARKETS_CACHE = exchange.markets
        return _MARKETS_CACHE


def _crear_exchange():
    import ccxt
    exchange = ccxt.binance({'enableRateLimit': True})
    exchange.set_markets(_obtener_markets())
    return exchange


def _fetch_ohlcv_con_reintentos(exchange, symbol, ccxt_tf, since_ms, limit=1000):
    """fetch_ohlcv con backoff exponencial ante 429 o error de red puntual."""
    import ccxt
    for intento in range(_MAX_REINTENTOS):
        try:
            return exchange.fetch_ohlcv(symbol, ccxt_tf, since=since_ms, limit=limit)
        except (ccxt.RateLimitExceeded, ccxt.NetworkError):
            if intento == _MAX_REINTENTOS - 1:
                raise
            time.sleep(2 ** (intento + 1))


def _fetch_first_candle(symbol: str, tf: str = '1d') -> datetime:
    """Obtiene la fecha de la primera vela disponible en Binance.

    Usa '1d' para el probe porque con TFs intradiarios y since=0 la API de
    Binance devuelve velas recientes en lugar de las más antiguas.
    """
    try:
        exchange = _crear_exchange()
        first = _fetch_ohlcv_con_reintentos(exchange, symbol, '1d', 0, limit=1)
        if first:
            return datetime.fromtimestamp(first[0][0] / 1000, tz=timezone.utc)
    except Exception:
        pass
    return datetime(2017, 8, 1, tzinfo=timezone.utc)

def _fetch_catalog() -> list:
    """Obtiene todos los pares USDT listados en Binance vía ccxt."""
    import ccxt
    try:
        exchange = ccxt.binance({'enableRateLimit': True})
        markets = exchange.load_markets()
        usdt_pairs = []
        for symbol, market in markets.items():
            if not market.get('spot'):
                continue
            if symbol.endswith('/USDT') and market.get('active'):
                name = market.get('info', {}).get('baseAsset', symbol.split('/')[0])
                usdt_pairs.append(AssetInfo(
                    symbol, f'{name} / Tether', 'Spot Crypto', None
                ))
        # Ordenar alfabéticamente
        usdt_pairs.sort(key=lambda a: a.symbol)
        return usdt_pairs
    except Exception:
        return []  # fallback: lista vacía, se reintentará


def _fetch_range(symbol: str, ccxt_tf: str, range_start_ms: int,
                 range_end_ms: int, on_batch=None) -> list:
    """Descarga un rango continuo de velas desde Binance.

    `on_batch(n)` se invoca tras cada lote de velas recibido, para que el
    llamante pueda reportar progreso en vivo desde varios workers.
    """
    exchange = _crear_exchange()

    all_data = []
    since_ms = range_start_ms

    while since_ms < range_end_ms:
        try:
            data = _fetch_ohlcv_con_reintentos(exchange, symbol, ccxt_tf, since_ms)
        except Exception as e:
            raise RuntimeError(f"Error fetch_ohlcv: {e}")

        if not data:
            break

        all_data.extend(data)
        since_ms = data[-1][0] + 1
        if on_batch:
            on_batch(len(data))

    return all_data


# ── data.binance.vision (archivos ZIP mensuales/diarios) ──────────────────────
# Binance publica todo el histórico de klines en archivos ZIP abiertos.
# Cada ZIP contiene un CSV con columnas:
#   open_time, open, high, low, close, volume, close_time,
#   quote_volume, count, taker_buy_base, taker_buy_quote, ignore

_ARCHIVE_COLS = [
    'open_time', 'open', 'high', 'low', 'close', 'volume',
    'close_time', 'quote_volume', 'count',
    'taker_buy_base', 'taker_buy_quote', 'ignore',
]


def _archive_symbol(symbol: str) -> str:
    """SOL/USDT → SOLUSDT (formato que usa data.binance.vision)."""
    return symbol.replace('/', '').replace(':', '').upper()


def _download_one_zip(url: str) -> Optional[pd.DataFrame]:
    """Descarga un ZIP desde data.binance.vision y devuelve el DataFrame interno.

    Devuelve None si el archivo no existe (404) o hay un error de red.
    """
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Analytics/1.0'})
        with urllib.request.urlopen(req, timeout=60) as resp:
            if resp.status != 200:
                return None
            raw = resp.read()
    except Exception:
        return None
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            csv_name = zf.namelist()[0]
            with zf.open(csv_name) as f:
                df = pd.read_csv(f, header=None, names=_ARCHIVE_COLS)
        return df
    except Exception:
        return None


def _gen_monthly_urls(base_symbol: str, ccxt_tf: str, start: datetime, end: datetime) -> list:
    """Genera URLs de ZIPs mensuales desde start hasta el mes anterior a end."""
    urls = []
    year = start.year
    month = start.month
    end_year = end.year
    end_month = end.month
    while (year, month) <= (end_year, end_month):
        fname = f"{base_symbol}-{ccxt_tf}-{year}-{month:02d}.zip"
        urls.append(f"{_ARCHIVE_BASE}/data/spot/monthly/klines/{base_symbol}/{ccxt_tf}/{fname}")
        month += 1
        if month > 12:
            month = 1
            year += 1
    return urls


def _gen_daily_urls(base_symbol: str, ccxt_tf: str, year: int, month: int, day_from: int, day_to: int) -> list:
    """Genera URLs de ZIPs diarios para un mes concreto."""
    urls = []
    for d in range(day_from, day_to + 1):
        fname = f"{base_symbol}-{ccxt_tf}-{year}-{month:02d}-{d:02d}.zip"
        urls.append(f"{_ARCHIVE_BASE}/data/spot/daily/klines/{base_symbol}/{ccxt_tf}/{fname}")
    return urls


def _fetch_archive_range(symbol: str, ccxt_tf: str, start: datetime, end: datetime, progress_callback=None) -> pd.DataFrame:
    """Descarga histórico completo desde data.binance.vision.

    1. ZIPs mensuales desde start hasta el mes anterior al actual.
    2. ZIPs diarios del mes en curso (hasta ayer).
    3. Velas de hoy vía API ccxt (1-2 requests).
    """
    base_sym = _archive_symbol(symbol)
    now = datetime.now(timezone.utc)

    monthly_urls = _gen_monthly_urls(base_sym, ccxt_tf, start, now)
    monthly_urls = monthly_urls[:-1]  # descartar el mes en curso (no tiene ZIP)

    current_year = now.year
    current_month = now.month
    today_day = now.day
    daily_urls = _gen_daily_urls(base_sym, ccxt_tf, current_year, current_month, 1, today_day - 1)

    all_urls = monthly_urls + daily_urls
    if progress_callback:
        progress_callback(f"  Archive: {len(monthly_urls)} ZIPs mensuales + {len(daily_urls)} diarios")

    all_dfs = []
    _prog_lock = threading.Lock()
    _completed = [0]
    total = len(all_urls)

    def _on_done(df):
        if df is not None and len(df) > 0:
            all_dfs.append(df)
        if progress_callback and total > 0:
            with _prog_lock:
                _completed[0] += 1
                pct = min(99, int(_completed[0] / total * 100))
                progress_callback(f"  [{pct}/100] {_completed[0]}/{total} ZIPs procesados")

    workers = min(8, max(2, len(all_urls) // 4)) if total > 4 else 1
    if workers == 1:
        for url in all_urls:
            _on_done(_download_one_zip(url))
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_download_one_zip, u): u for u in all_urls}
            for f in as_completed(futures):
                _on_done(f.result())

    # Velas de hoy vía API ccxt (desde medianoche UTC hasta end)
    today_start = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
    today_end = min(end, now)
    today_ms_start = int(today_start.timestamp() * 1000)
    today_ms_end = int(today_end.timestamp() * 1000)
    if today_ms_end > today_ms_start:
        try:
            exchange = _crear_exchange()
            today_data = _fetch_ohlcv_con_reintentos(exchange, symbol, ccxt_tf, today_ms_start, limit=1000)
            while today_data and today_data[-1][0] < today_ms_end:
                batch = _fetch_ohlcv_con_reintentos(exchange, symbol, ccxt_tf, today_data[-1][0] + 1, limit=1000)
                if not batch:
                    break
                today_data.extend(batch)
            if today_data:
                today_df = pd.DataFrame(today_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                today_df['timestamp'] = pd.to_datetime(today_df['timestamp'], unit='ms', utc=True)
                all_dfs.append(today_df)
        except Exception:
            pass

    if progress_callback:
        progress_callback(f"  [100/100] Archive completo")

    if not all_dfs:
        return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'spread'])

    # Concatenar y normalizar
    parts = []
    for df in all_dfs:
        if 'open_time' in df.columns:
            df = df[['open_time', 'open', 'high', 'low', 'close', 'volume']].rename(columns={'open_time': 'timestamp'})
            # Binance archive cambia de ms (13 digitos, meses antiguos) a
            # us (16 digitos, meses recientes).  Se detecta por longitud.
            sample_ts = int(df['timestamp'].iloc[0])
            unit = 'us' if sample_ts > 10**15 else 'ms'
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit=unit, utc=True)
        parts.append(df)

    combined = pd.concat(parts, ignore_index=True)
    combined = combined.drop_duplicates(subset='timestamp').set_index('timestamp').sort_index()

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    combined = combined[(combined.index >= start_ts) & (combined.index <= end_ts)]

    combined['spread'] = 0
    combined = combined.reset_index()
    combined = combined[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'spread']]

    return combined


class CCXTProvider(BaseProvider):
    name = 'ccxt (Binance)'

    @staticmethod
    def refresh_catalog() -> List[AssetInfo]:
        global _CACHE_CATALOG, _CACHE_TIMESTAMP
        _CACHE_CATALOG = None
        _CACHE_TIMESTAMP = None
        path = _catalog_cache_path()
        if os.path.exists(path):
            os.remove(path)
        return CCXTProvider.get_catalog()

    @staticmethod
    def get_catalog() -> List[AssetInfo]:
        global _CACHE_CATALOG, _CACHE_TIMESTAMP
        now = datetime.now(timezone.utc)
        if _CACHE_CATALOG is not None and _CACHE_TIMESTAMP is not None:
            if (now - _CACHE_TIMESTAMP).total_seconds() < _CACHE_TTL:
                return _CACHE_CATALOG.copy()
        disk = _load_catalog_from_disk()
        if disk is not None:
            _CACHE_CATALOG = disk
            _CACHE_TIMESTAMP = now
            return disk.copy()
        rest = _fetch_catalog()
        if not rest:
            if _CACHE_CATALOG is not None:
                return _CACHE_CATALOG.copy()
            return _TOP_PAIRS.copy()
        rest = [a for a in rest if a.symbol not in _TOP_SYMBOLS]
        separator = AssetInfo(_SEPARATOR_SYMBOL, '--- Mas activos ---', '', None)
        catalog = _TOP_PAIRS + [separator] + rest
        _CACHE_CATALOG = catalog
        _CACHE_TIMESTAMP = now
        _save_catalog_to_disk(catalog)
        return catalog.copy()

    @staticmethod
    def get_available_range(symbol: str, progress_callback=None) -> Optional[tuple]:
        global _RANGE_CACHE
        if symbol not in _RANGE_CACHE:
            first_dt = _fetch_first_candle(symbol)
            _RANGE_CACHE[symbol] = first_dt
        first_dt = _RANGE_CACHE[symbol]
        end_dt = datetime.now(timezone.utc)
        if progress_callback:
            progress_callback(
                f"  Rango estimado: {first_dt.date()} -> {end_dt.date()}"
            )
        return (first_dt, end_dt)

    @staticmethod
    def download_ohlc(symbol: str, tf: str,
                      start: Optional[datetime] = None,
                      end: Optional[datetime] = None,
                      progress_callback=None) -> pd.DataFrame:
        if tf not in CCXT_TF_MAP:
            raise ValueError(f"TF '{tf}' no soportado por ccxt")

        ccxt_tf = CCXT_TF_MAP[tf]
        minutes_per_candle = _TF_MINUTES.get(tf, 60)

        if start is None or end is None:
            first_dt = _fetch_first_candle(symbol, tf)
            if start is None:
                start = first_dt
            if end is None:
                end = datetime.now(timezone.utc)

        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)

        start_ms = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)
        total_span = end_ms - start_ms

        if total_span <= 0:
            raise RuntimeError("Rango de fechas invalido.")

        est_candles = int(total_span / (minutes_per_candle * 60000))

        # ── Branch: TFs intradiarios → data.binance.vision (ZIPs mensuales) ──
        # La API ccxt no sirve histórico completo para 1m/5m/15m, pero
        # data.binance.vision tiene archivos ZIP con todo el histórico desde
        # el listing del activo.
        if tf in _ARCHIVE_TF:
            if progress_callback:
                progress_callback(f"Descargando {symbol}: {start.date()} -> {end.date()}")
                progress_callback(f"Timeframe: {tf}  |  ~{est_candles:,} velas estimadas (archivo histórico)")

            df = _fetch_archive_range(symbol, ccxt_tf, start, end, progress_callback)

            if progress_callback:
                progress_callback(f"Velas {tf} generadas: {len(df):,}")
                if len(df) > 0:
                    progress_callback(f"  Primera vela: {df['timestamp'].iloc[0]}")
                    progress_callback(f"  Ultima vela:  {df['timestamp'].iloc[-1]}")
            return df

        # ── Branch: TFs altos (1h+) → ccxt API (flujo original) ──────────────
        if progress_callback:
            progress_callback(f"Conectando a Binance...")
        _obtener_markets()

        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        max_lookback_candles = _BINANCE_MAX_LOOKBACK.get(tf)
        if max_lookback_candles is not None:
            earliest_ms = now_ms - max_lookback_candles * minutes_per_candle * 60000
            if start_ms < earliest_ms:
                start_ms = earliest_ms
                start = datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc)
                total_span = end_ms - start_ms
                est_candles = int(total_span / (minutes_per_candle * 60000))
                if progress_callback:
                    progress_callback(
                        f"  ⚠ Binance limita el histórico de {tf}: "
                        f"datos disponibles desde {start.date()}"
                    )
        if progress_callback:
            progress_callback(f"Descargando {symbol}: {start.date()} -> {end.date()}")
            progress_callback(f"Timeframe: {tf}  |  ~{est_candles:,} velas estimadas")
            if _PARALLEL_WORKERS > 1:
                progress_callback(f"Workers: {_PARALLEL_WORKERS}")

        # Progreso en vivo por lote (contador compartido entre workers). Se
        # emite como "[pct/100]" para que la barra de la GUI también avance.
        _prog_lock = threading.Lock()
        _prog = {'velas': 0, 'ultimo_pct': -1}

        def _on_batch(n):
            if not progress_callback:
                return
            with _prog_lock:
                _prog['velas'] += n
                pct = min(99, int(_prog['velas'] / max(1, est_candles) * 100))
                if pct > _prog['ultimo_pct']:
                    _prog['ultimo_pct'] = pct
                    progress_callback(f"  [{pct}/100] {_prog['velas']:,} velas")

        chunk_span = total_span // _PARALLEL_WORKERS
        chunks = []
        for w in range(_PARALLEL_WORKERS):
            c_start = start_ms + w * chunk_span
            c_end = min(end_ms, start_ms + (w + 1) * chunk_span) if w < _PARALLEL_WORKERS - 1 else end_ms
            if c_end > c_start:
                chunks.append((c_start, c_end))

        if _PARALLEL_WORKERS == 1:
            all_data = _fetch_range(symbol, ccxt_tf, start_ms, end_ms, on_batch=_on_batch)
        else:
            all_data = []
            with ThreadPoolExecutor(max_workers=_PARALLEL_WORKERS) as executor:
                futures = {}
                for c_start, c_end in chunks:
                    f = executor.submit(
                        _fetch_range, symbol, ccxt_tf, c_start, c_end, _on_batch,
                    )
                    futures[f] = (c_start, c_end)

                for f in as_completed(futures):
                    all_data.extend(f.result())

        if progress_callback:
            progress_callback(f"  [100/100] {len(all_data):,} velas")

        if not all_data:
            raise RuntimeError(f"No se obtuvieron datos para {symbol}")

        if progress_callback:
            progress_callback(f"Procesando y ordenando {len(all_data):,} velas...")

        cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        df = pd.DataFrame(all_data, columns=cols)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df = df.drop_duplicates(subset='timestamp').set_index('timestamp').sort_index()

        end_ts = pd.Timestamp(end)
        df = df[df.index <= end_ts]

        df['spread'] = 0
        df = df.reset_index()
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'spread']]

        if progress_callback:
            progress_callback(f"Velas {tf} generadas: {len(df):,}")
            if len(df) > 0:
                progress_callback(f"  Primera vela: {df['timestamp'].iloc[0]}")
                progress_callback(f"  Ultima vela:  {df['timestamp'].iloc[-1]}")

        return df
