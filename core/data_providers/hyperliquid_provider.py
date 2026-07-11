"""
Proveedor de datos de Hyperliquid via ccxt.

Hyperliquid ofrece:
  - HIP-3 markets: acciones tokenizadas, indices, materias primas (XYZ-*)
  - Swap markets: perps de criptomonedas

Todos los mercados son perpetual swaps (no spot).
Los datos historicos dependen del activo (muchos HIP-3 listados en 2025-2026).
"""

import pandas as pd
from datetime import datetime, timezone
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import tempfile
import time

from .base_provider import BaseProvider, AssetInfo

HL_TF_MAP = {
    '1m': '1m', '5m': '5m', '15m': '15m',
    '1h': '1h', '4h': '4h', '1d': '1d',
}

_TF_MINUTES = {'1m': 1, '5m': 5, '15m': 15, '1h': 60, '4h': 240, '1d': 1440}
_PARALLEL_WORKERS = 4

_SEPARATOR_SYMBOL = '__SEPARATOR__'
_BUILDER_PREFIXES = ('CASH-', 'FLX-', 'HYNA-', 'KM-', 'MKTS-', 'PARA-', 'VNTL-', 'ABCD-')

_COMMODITIES = {
    'ALUMINIUM', 'BRENTOIL', 'COPPER', 'CORN', 'GOLD', 'NATGAS',
    'PALLADIUM', 'PLATINUM', 'SILVER', 'TTF', 'URANIUM', 'WHEAT',
}
_INDICES = {
    'DXY', 'EWJ', 'EWT', 'EWY', 'EWZ', 'IBOV', 'JP225', 'KR200',
    'NIFTY', 'SP500', 'SPCX', 'USAR', 'VIX', 'XYZ100',
}
_FX = {'EUR', 'GBP', 'JPY', 'KRW'}

_TOP_SWAPS = [
    'BTC/USDC:USDC', 'ETH/USDC:USDC', 'SOL/USDC:USDC', 'XRP/USDC:USDC',
    'DOGE/USDC:USDC', 'ADA/USDC:USDC', 'AVAX/USDC:USDC', 'DOT/USDC:USDC',
    'LINK/USDC:USDC', 'UNI/USDC:USDC', 'ATOM/USDC:USDC', 'LTC/USDC:USDC',
    'BNB/USDC:USDC', 'SUI/USDC:USDC', 'NEAR/USDC:USDC', 'OP/USDC:USDC',
    'ARB/USDC:USDC', 'APT/USDC:USDC', 'INJ/USDC:USDC', 'ONDO/USDC:USDC',
    'HYPE/USDC:USDC', 'AAVE/USDC:USDC', 'PENDLE/USDC:USDC', 'TIA/USDC:USDC',
    'SEI/USDC:USDC', 'STRK/USDC:USDC', 'TRX/USDC:USDC',
]
_TOP_SWAP_SYMBOLS = set(_TOP_SWAPS)

_CACHE_CATALOG = None
_CACHE_TIMESTAMP = None
_CACHE_TTL = 3600
_RANGE_CACHE = {}

_CACHE_DIR = os.path.join(tempfile.gettempdir(), 'analytics_cache')
_CACHE_FILE = 'hyperliquid_catalog.json'

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


def _classify_hip3(name: str) -> str:
    if name in _COMMODITIES or name in _FX:
        return name
    if name in _INDICES:
        return name
    return name


def _fetch_catalog():
    """Obtiene HIP-3 + crypto perps de Hyperliquid."""
    import ccxt
    try:
        exchange = ccxt.hyperliquid({'enableRateLimit': True})
        exchange.load_markets()

        hip3_markets = exchange.fetch_hip3_markets()
        hip3_xyz = {}
        for m in hip3_markets:
            sym = m['symbol']
            if not sym.startswith('XYZ-'):
                continue
            name = sym.split('/')[0].replace('XYZ-', '')
            if name.startswith('CASH-') or name.startswith('FLX-') or name.startswith('HYNA-'):
                continue
            hip3_xyz[sym] = name

        commodities = []
        indices = []
        stocks = []
        for sym, name in sorted(hip3_xyz.items(), key=lambda x: x[1]):
            if name in _COMMODITIES:
                commodities.append(AssetInfo(sym, name, 'Commodities', None))
            elif name in _INDICES:
                indices.append(AssetInfo(sym, name, 'Indices', None))
            elif name in _FX:
                indices.append(AssetInfo(sym, name, 'FX', None))
            else:
                stocks.append(AssetInfo(sym, name, 'Acciones', None))

        hip3_symbols = set(hip3_xyz.keys())
        all_swaps = [s for s in exchange.markets.values()
                     if s.get('swap') and s['symbol'] not in hip3_symbols]

        swap_clean = []
        for s in all_swaps:
            sym = s['symbol']
            base = sym.split('/')[0]
            if base.startswith(_BUILDER_PREFIXES):
                continue
            if base in ('USDC', 'USDT', 'DAI', 'FDUSD'):
                continue
            swap_clean.append(AssetInfo(sym, base, 'Perps Crypto', None))

        swap_clean.sort(key=lambda a: a.symbol)

        separator = AssetInfo(_SEPARATOR_SYMBOL, '', '', None)
        catalog = []

        if stocks:
            catalog.append(AssetInfo(_SEPARATOR_SYMBOL, '--- Acciones ---', '', None))
            catalog.extend(stocks)
        if indices:
            catalog.append(AssetInfo(_SEPARATOR_SYMBOL, '--- Indices / FX ---', '', None))
            catalog.extend(indices)
        if commodities:
            catalog.append(AssetInfo(_SEPARATOR_SYMBOL, '--- Materias Primas ---', '', None))
            catalog.extend(commodities)
        if swap_clean:
            catalog.append(AssetInfo(_SEPARATOR_SYMBOL, '--- Perps Crypto ---', '', None))
            top = [a for a in swap_clean if a.symbol in _TOP_SWAP_SYMBOLS]
            rest = [a for a in swap_clean if a.symbol not in _TOP_SWAP_SYMBOLS]
            catalog.extend(top)
            if rest:
                catalog.append(AssetInfo(_SEPARATOR_SYMBOL, '--- Mas Perps ---', '', None))
                catalog.extend(rest)

        return catalog
    except Exception:
        return []


def _fetch_first_candle(symbol: str) -> datetime:
    import ccxt
    try:
        exchange = ccxt.hyperliquid({'enableRateLimit': True})
        first = exchange.fetch_ohlcv(symbol, '1d', since=0, limit=1)
        if first:
            return datetime.fromtimestamp(first[0][0] / 1000, tz=timezone.utc)
    except Exception:
        pass
    return datetime(2025, 1, 1, tzinfo=timezone.utc)


def _fetch_range(symbol: str, ccxt_tf: str, range_start_ms: int,
                 range_end_ms: int) -> list:
    import ccxt
    exchange = ccxt.hyperliquid({'enableRateLimit': True})
    exchange.load_markets()

    all_data = []
    since_ms = range_start_ms

    while since_ms < range_end_ms:
        try:
            data = exchange.fetch_ohlcv(symbol, ccxt_tf, since=since_ms, limit=1000)
        except Exception as e:
            raise RuntimeError(f"Error fetch_ohlcv: {e}")

        if not data:
            break

        all_data.extend(data)
        since_ms = data[-1][0] + 1

    return all_data


class HyperliquidProvider(BaseProvider):
    name = 'Hyperliquid'

    @staticmethod
    def refresh_catalog() -> List[AssetInfo]:
        global _CACHE_CATALOG, _CACHE_TIMESTAMP
        _CACHE_CATALOG = None
        _CACHE_TIMESTAMP = None
        path = _catalog_cache_path()
        if os.path.exists(path):
            os.remove(path)
        return HyperliquidProvider.get_catalog()

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
        catalog = _fetch_catalog()
        if not catalog:
            if _CACHE_CATALOG is not None:
                return _CACHE_CATALOG.copy()
            return []
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
        if tf not in HL_TF_MAP:
            raise ValueError(f"TF '{tf}' no soportado por Hyperliquid")

        if start is None or end is None:
            rango = HyperliquidProvider.get_available_range(symbol, progress_callback)
            if rango is None:
                raise RuntimeError(f"No se pudo determinar el rango para {symbol}")
            if start is None:
                start = rango[0]
            if end is None:
                end = rango[1]

        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)

        ccxt_tf = HL_TF_MAP[tf]
        start_ms = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)
        total_span = end_ms - start_ms

        if total_span <= 0:
            raise RuntimeError("Rango de fechas invalido.")

        minutes_per_candle = _TF_MINUTES.get(tf, 60)
        est_candles = int(total_span / (minutes_per_candle * 60000))

        if progress_callback:
            progress_callback(f"Conectando a Hyperliquid...")
            progress_callback(f"Descargando {symbol}: {start.date()} -> {end.date()}")
            progress_callback(f"Timeframe: {tf}  |  ~{est_candles:,} velas estimadas")
            if _PARALLEL_WORKERS > 1:
                progress_callback(f"Workers: {_PARALLEL_WORKERS}")

        chunk_span = total_span // _PARALLEL_WORKERS
        chunks = []
        for w in range(_PARALLEL_WORKERS):
            c_start = start_ms + w * chunk_span
            c_end = min(end_ms, start_ms + (w + 1) * chunk_span) if w < _PARALLEL_WORKERS - 1 else end_ms
            if c_end > c_start:
                chunks.append((c_start, c_end))

        if _PARALLEL_WORKERS == 1:
            all_data = _fetch_range(symbol, ccxt_tf, start_ms, end_ms)
            if progress_callback:
                progress_callback(f"  [100/100] {len(all_data):,} velas")
        else:
            all_data = []
            with ThreadPoolExecutor(max_workers=_PARALLEL_WORKERS) as executor:
                futures = {}
                for c_start, c_end in chunks:
                    f = executor.submit(
                        _fetch_range, symbol, ccxt_tf, c_start, c_end,
                    )
                    futures[f] = (c_start, c_end)

                completed = 0
                total_chunks = len(chunks)
                for f in as_completed(futures):
                    data = f.result()
                    all_data.extend(data)
                    completed += 1
                    pct = int(completed / total_chunks * 100)
                    if progress_callback:
                        progress_callback(f"  [{pct}/100] {len(all_data):,} velas")

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
