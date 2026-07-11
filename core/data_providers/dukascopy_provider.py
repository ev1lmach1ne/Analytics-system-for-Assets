"""
Proveedor de datos Dukascopy usando requests + decoder .bi5 propio.

Dukascopy publica tick data comprimida en formato .bi5 (LZMA) accesible
gratis a traves de https://datafeed.dukascopy.com/datafeed/.  Este modulo
descarga los archivos horarios, los descomprime y resamplea a velas OHLC.
"""

import lzma
import time
import struct
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta,date
from typing import List, Optional

from .base_provider import BaseProvider, AssetInfo, TF_TO_PANDAS

# --- Configuracion ------------------------------------------------------------

BASE_URL = "https://datafeed.dukascopy.com/datafeed/"
HEADERS = {'User-Agent': 'Mozilla/5.0'}
REQUEST_TIMEOUT = 60
RETRY_ATTEMPTS = 4
RETRY_BASE_DELAY = 2.0
REQUEST_POLITE_DELAY = 0.05   # small delay between requests to avoid rate-limit
BATCH_SIZE = 500              # flush ticks to OHLC every N chunks (streaming)

PIPET_SIZE_REGISTRY = {
    'EURUSD': 1e-5, 'GBPUSD': 1e-5, 'AUDUSD': 1e-5,
    'NZDUSD': 1e-5, 'USDCAD': 1e-5, 'USDCHF': 1e-5,
    'USDJPY': 1e-3, 'XAUUSD': 1e-3, 'XAGUSD': 1e-3,
    'BTCUSD': 0.1, 'ETHUSD': 0.1,
}

# --- Catalogo pre-cargado -----------------------------------------------------

_CATALOG = [
    # --- Forex majors ---
    AssetInfo("EURUSD", "Euro / US Dollar", "Forex", "2003-01-01"),
    AssetInfo("GBPUSD", "British Pound / US Dollar", "Forex", "2003-01-01"),
    AssetInfo("USDJPY", "US Dollar / Japanese Yen", "Forex", "2003-01-01"),
    AssetInfo("AUDUSD", "Australian Dollar / US Dollar", "Forex", "2003-01-01"),
    AssetInfo("USDCAD", "US Dollar / Canadian Dollar", "Forex", "2003-01-01"),
    AssetInfo("USDCHF", "US Dollar / Swiss Franc", "Forex", "2003-01-01"),
    AssetInfo("NZDUSD", "New Zealand Dollar / US Dollar", "Forex", "2003-01-01"),
    # --- Forex crosses ---
    AssetInfo("EURGBP", "Euro / British Pound", "Forex", "2008-01-01"),
    AssetInfo("EURJPY", "Euro / Japanese Yen", "Forex", "2008-01-01"),
    AssetInfo("GBPJPY", "British Pound / Japanese Yen", "Forex", "2008-01-01"),
    AssetInfo("AUDJPY", "Australian Dollar / Japanese Yen", "Forex", "2008-01-01"),
    # --- Metales ---
    AssetInfo("XAUUSD", "Oro / US Dollar", "Metal", "2006-01-01"),
    AssetInfo("XAGUSD", "Plata / US Dollar", "Metal", "2006-01-01"),
    # --- Criptomonedas ---
    AssetInfo("BTCUSD", "Bitcoin / US Dollar", "Cripto", "2017-01-01"),
    AssetInfo("ETHUSD", "Ethereum / US Dollar", "Cripto", "2018-01-01"),
]

# --- Helpers ------------------------------------------------------------------

def _build_url(symbol: str, dt: datetime) -> str:
    """Construye la URL de Dukascopy para una hora especifica."""
    return (f"{BASE_URL}{symbol}/{dt.year}/"
            f"{dt.month - 1:02d}/{dt.day:02d}/{dt.hour:02d}h_ticks.bi5")


def _decode_bi5(content: bytes, hour_start: datetime, pipet_scale: float) -> np.ndarray:
    """
    Descomprime y decodifica un chunk .bi5 a un array estructurado de ticks.

    Formato Dukascopy: LZMA + registros big-endian:
    - time: uint32 (ms desde el inicio de la hora)
    - ask: uint32 (precio * inverso de pipet_scale)
    - bid: uint32 (precio * inverso de pipet_scale)
    - ask_volume: float32 (escala 1e-6)
    - bid_volume: float32 (escala 1e-6)
    """
    buf = lzma.decompress(content)
    dtype = np.dtype([
        ("time", ">u4"), ("ask", ">u4"), ("bid", ">u4"),
        ("ask_volume", ">f4"), ("bid_volume", ">f4"),
    ])
    raw = np.frombuffer(buf, dtype=dtype)
    n = len(raw)
    out = np.empty(n, dtype=[
        ("time", "datetime64[ms]"),
        ("ask", "f8"), ("bid", "f8"),
        ("ask_volume", "i8"), ("bid_volume", "i8"),
    ])
    start = np.datetime64(hour_start.replace(tzinfo=None), "ms")
    out["time"] = start + raw["time"].astype(np.int64).astype("timedelta64[ms]")
    out["ask"] = raw["ask"].astype(np.float64) * pipet_scale
    out["bid"] = raw["bid"].astype(np.float64) * pipet_scale
    out["ask_volume"] = np.round(raw["ask_volume"].astype(np.float64) * 1e6).astype(np.int64)
    out["bid_volume"] = np.round(raw["bid_volume"].astype(np.float64) * 1e6).astype(np.int64)
    return out


def _fetch_chunk(url: str, progress_callback=None) -> Optional[bytes]:
    """Descarga un chunk .bi5 con reintentos."""
    for attempt in range(RETRY_ATTEMPTS + 1):
        try:
            r = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            if r.status_code == 200 and r.content:
                return r.content
            elif r.status_code == 404:
                return None  # No hay datos en esa hora
            elif r.status_code in (429, 503):
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                if progress_callback:
                    progress_callback(f"   Rate-limited ({r.status_code}), esperando {delay:.0f}s...")
                time.sleep(delay)
                continue
            else:
                if progress_callback:
                    progress_callback(f"   HTTP {r.status_code} en {url}")
                return None
        except (requests.Timeout, requests.ConnectionError) as e:
            if attempt < RETRY_ATTEMPTS:
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                if progress_callback:
                    progress_callback(f"   Timeout, reintentando en {delay:.0f}s (intento {attempt + 1}/{RETRY_ATTEMPTS})...")
                time.sleep(delay)
                continue
            if progress_callback:
                progress_callback(f"   Error final tras {RETRY_ATTEMPTS} intentos: {e}")
            return None
    return None


def _generate_hourly_datetimes(start: datetime, end: datetime) -> List[datetime]:
    """Genera lista de datetimes por hora entre start y end (UTC)."""
    start = start.replace(tzinfo=timezone.utc, minute=0, second=0, microsecond=0)
    end = end.replace(tzinfo=timezone.utc, minute=0, second=0, microsecond=0)
    out = []
    current = start
    while current < end:
        out.append(current)
        current += timedelta(hours=1)
    return out


def _resample_to_ohlc(ticks_df: pd.DataFrame, tf: str) -> pd.DataFrame:
    """
    Resamplea un DataFrame de ticks a velas OHLC.

    timestamps del DataFrame deben ser naive (sin tz), como indice.
    La columna 'bid' se usa como precio (mid seria mas preciso pero
    'bid' es el estandar de Dukascopy).
    """
    rule = TF_TO_PANDAS.get(tf, '1h')

    ohlc = ticks_df['bid'].resample(rule).agg(['first', 'max', 'min', 'last'])
    ohlc.columns = ['open', 'high', 'low', 'close']

    vol = ticks_df[['ask_volume', 'bid_volume']].resample(rule).sum().sum(axis=1)
    vol.name = 'volume'

    spread = (ticks_df['ask'] - ticks_df['bid']).resample(rule).mean()
    spread.name = 'spread'

    result = ohlc.join([vol, spread]).dropna()
    return result


# --- Clase provider -----------------------------------------------------------

class DukascopyProvider(BaseProvider):
    name = "Dukascopy"

    @staticmethod
    def get_catalog() -> List[AssetInfo]:
        """Devuelve el catalogo pre-cargado de Dukascopy."""
        return _CATALOG.copy()

    @staticmethod
    def get_available_range(symbol: str, progress_callback=None) -> Optional[tuple]:
        """
        Determina el rango de datos disponibles para un simbolo.

        Hace un request al catalogo JSON de InstrumentInfo de Dukascopy
        para obtener el primer tick disponible. Si falla, usa el default
        del catalogo pre-cargado.
        """
        if progress_callback:
            progress_callback(f"Consultando rango disponible para {symbol}...")
        try:
            url = (f"https://freeserv.dukascopy.com/svc/datainstrument?&"
                   f"instrument={symbol}&true")
            r = requests.get(url, headers=HEADERS, timeout=15)
            if r.status_code == 200:
                data = r.json()
                start_str = data.get('fromTime', '').split(' ')[0]
                end_str = data.get('tillTime', '').split(' ')[0]
                if start_str and end_str:
                    start_dt = datetime.strptime(start_str, '%Y-%m-%d').replace(tzinfo=timezone.utc)
                    end_dt = datetime.strptime(end_str, '%Y-%m-%d').replace(tzinfo=timezone.utc)
                    if progress_callback:
                        progress_callback(f"  Rango: {start_dt.date()} -> {end_dt.date()}")
                    return (start_dt, end_dt)
        except Exception as e:
            if progress_callback:
                progress_callback(f"  API catalogo fallo ({e}); usando default.")

        # Fallback: default del catalogo
        for asset in _CATALOG:
            if asset.symbol == symbol and asset.max_history_start:
                start_dt = datetime.strptime(asset.max_history_start, '%Y-%m-%d').replace(tzinfo=timezone.utc)
                end_dt = datetime.now(timezone.utc)
                if progress_callback:
                    progress_callback(f"  Default: {start_dt.date()} -> {end_dt.date()}")
                return (start_dt, end_dt)
        return None

    @staticmethod
    def download_ohlc(symbol: str, tf: str,
                      start: Optional[datetime] = None,
                      end: Optional[datetime] = None,
                      progress_callback=None) -> pd.DataFrame:
        """
        Descarga tick data y resamplea a OHLC.

        TFs soportados: '1m', '5m', '15m', '1h', '4h', '1d'.
        start/end None => usa toda la historia disponible.
        """
        if symbol not in PIPET_SIZE_REGISTRY:
            raise ValueError(
                f"Simbolo '{symbol}' no tiene pipet_scale registrado. "
                f"Simbolos disponibles: {list(PIPET_SIZE_REGISTRY.keys())}"
            )
        pipet = PIPET_SIZE_REGISTRY[symbol]

        if start is None or end is None:
            rango = DukascopyProvider.get_available_range(symbol, progress_callback)
            if rango is None:
                raise RuntimeError(f"No se pudo determinar el rango para {symbol}")
            if start is None:
                start = rango[0]
            if end is None:
                end = rango[1]

        hours = _generate_hourly_datetimes(start, end)
        total = len(hours)
        if total == 0:
            raise RuntimeError("El rango de fechas es invalido o vacio.")

        if progress_callback:
            progress_callback(f"Descargando {symbol}: {start.date()} -> {end.date()}")
            progress_callback(f"Total chunks horarios: {total}")
            progress_callback(f"Timeframe objetivo: {tf}")

        batch_ticks = []
        ohlc_chunks = []
        downloaded = 0
        no_data = 0
        errors = 0
        tick_count = 0

        for i, hour_dt in enumerate(hours):
            if i % 25 == 0 or i == total - 1:
                if progress_callback:
                    pct = (i + 1) / total * 100
                    progress_callback(f"  [{i+1}/{total}] {pct:.2f}% - "
                                       f"OK:{downloaded} NoData:{no_data} Err:{errors}")

            if i > 0:
                time.sleep(REQUEST_POLITE_DELAY)

            url = _build_url(symbol, hour_dt)
            content = _fetch_chunk(url, progress_callback)
            if content is None:
                no_data += 1
                continue

            try:
                arr = _decode_bi5(content, hour_dt, pipet)
                if len(arr) > 0:
                    batch_ticks.append(arr)
                    downloaded += 1
            except Exception as e:
                errors += 1
                if progress_callback:
                    progress_callback(f"   Error decode {hour_dt}: {e}")

            should_flush = len(batch_ticks) >= BATCH_SIZE or (i == total - 1 and batch_ticks)
            if should_flush:
                combined = np.concatenate(batch_ticks)
                batch_df = pd.DataFrame({
                    'time': combined['time'],
                    'ask': combined['ask'],
                    'bid': combined['bid'],
                    'ask_volume': combined['ask_volume'],
                    'bid_volume': combined['bid_volume'],
                })
                batch_df['time'] = batch_df['time'].astype('datetime64[ns]')
                batch_df = batch_df.drop_duplicates(subset='time').set_index('time').sort_index()
                tick_count += len(batch_df)

                ohlc_batch = _resample_to_ohlc(batch_df, tf)
                if len(ohlc_batch) > 0:
                    ohlc_chunks.append(ohlc_batch)
                batch_ticks = []

        if not ohlc_chunks:
            raise RuntimeError(f"No se descargaron datos para {symbol}")

        if progress_callback:
            progress_callback(f"Descarga completa: {downloaded} horas con datos "
                              f"({no_data} sin datos, {errors} errores)")
            progress_callback(f"Total ticks: {tick_count:,}")
            progress_callback(f"Resampleando a {tf}...")

        ohlc = pd.concat(ohlc_chunks)
        ohlc = ohlc[~ohlc.index.duplicated(keep='first')].sort_index()

        ohlc = ohlc.reset_index()
        ohlc.rename(columns={'time': 'timestamp'}, inplace=True)
        ohlc['timestamp'] = ohlc['timestamp'].dt.tz_localize('UTC')

        if progress_callback:
            progress_callback(f"Velas {tf} generadas: {len(ohlc):,}")
            if len(ohlc) > 0:
                progress_callback(f"  Primera vela: {ohlc['timestamp'].iloc[0]}")
                progress_callback(f"  Ultima vela:  {ohlc['timestamp'].iloc[-1]}")

        return ohlc