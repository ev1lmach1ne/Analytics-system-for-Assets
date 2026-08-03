"""
Proveedor de datos Dukascopy usando requests + decoder .bi5 propio.

Dukascopy publica tick data comprimida en formato .bi5 (LZMA) accesible
gratis a traves de https://datafeed.dukascopy.com/datafeed/.  Este modulo
descarga los archivos horarios, los descomprime y resamplea a velas OHLC.
"""

import glob
import lzma
import os
import tempfile
import time
import struct
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta, date
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from .base_provider import BaseProvider, AssetInfo, TF_TO_PANDAS

# --- Configuracion ------------------------------------------------------------

BASE_URL = "https://datafeed.dukascopy.com/datafeed/"
HEADERS = {'User-Agent': 'Mozilla/5.0'}
REQUEST_TIMEOUT = 30
RETRY_BASE_DELAY = 2.0
RETRY_MAX_DELAY = 60.0        # techo de backoff exponencial por intento
RETRY_MAX_ATTEMPTS = 8        # tope de reintentos ante fallos transitorios
                              # (429/503/timeouts/conexion cortada); al
                              # superarlo esa hora se cuenta como error y
                              # la descarga sigue con las demas, en vez de
                              # bloquear el worker para siempre
_PARALLEL_WORKERS = 8         # conexiones simultáneas (Dukascopy es restrictivo).
                              # El pacing ya no es un sleep fijo entre requests
                              # (no escala para 2 décadas de datos): se logra
                              # acotando la concurrencia a este valor y
                              # dejando que _fetch_chunk reintente con backoff
                              # exponencial ante 429/503.
_BATCH_HOURS = 500            # horas por lote (control de memoria + progreso).
                              # Cada lote se descarga en paralelo pero se
                              # resamplea completo y en orden antes de pasar
                              # al siguiente, para que cada vela OHLC se
                              # calcule siempre sobre un rango horario
                              # contiguo (nunca sobre ticks mezclados fuera
                              # de orden entre lotes distintos).

# --- Checkpoint / resume -------------------------------------------------------
# Descargas de decadas en 1m pueden tardar dias reales (Dukascopy limita la
# concurrencia). Cada lote ya resampleado se guarda en disco segun se
# completa; si el proceso se corta a mitad, relanzar la misma descarga
# (mismo simbolo/tf/inicio) reanuda desde el ultimo lote guardado en vez de
# volver a empezar.
#
# La clave NO incluye la fecha fin a proposito: en modo "toda la historia"
# el fin se recalcula en cada corrida (es "ahora"/la ultima fecha publicada
# por Dukascopy) y cambia de una ejecucion a otra, incluso el mismo dia. Si
# la clave dependiera del fin, cada reintento generaria una clave distinta
# y jamas encontraria los checkpoints de la corrida anterior (exactamente
# el sintoma de "empieza desde 0%"). Cada lote es siempre el mismo rango de
# horas fijo (start + N*_BATCH_HOURS) sin importar donde termine el rango
# total, asi que reusarlos por indice es seguro; para detectar el caso
# borde en que el ultimo lote de una corrida anterior quedo mas corto de lo
# que ahora deberia ser (el fin crecio), cada checkpoint guarda tambien
# cuantas horas cubria, y solo se reutiliza si coincide con las horas que
# le corresponden en la corrida actual.
_CHECKPOINT_DIR = os.path.join(tempfile.gettempdir(), 'analytics_cache', 'dukascopy_checkpoints')
_CHECKPOINT_MAX_AGE_DAYS = 7  # limpieza de checkpoints huerfanos muy viejos (tambien acota
                              # cuanto tiempo vive el cache incremental de "toda la historia")


def _checkpoint_key(symbol: str, tf: str, start: datetime) -> str:
    safe_symbol = symbol.replace('/', '_').replace(':', '_')
    return f"{safe_symbol}_{tf}_{start.strftime('%Y%m%dT%H')}"


def _checkpoint_batch_path(key: str, batch_idx: int) -> str:
    return os.path.join(_CHECKPOINT_DIR, f"{key}__batch{batch_idx:05d}.pkl")


def _cleanup_old_checkpoints():
    """Borra checkpoints huerfanos de descargas abandonadas hace mas de
    _CHECKPOINT_MAX_AGE_DAYS. No afecta a la descarga en curso (distinto key)."""
    try:
        cutoff = time.time() - _CHECKPOINT_MAX_AGE_DAYS * 86400
        for path in glob.glob(os.path.join(_CHECKPOINT_DIR, '*.pkl')):
            if os.path.getmtime(path) < cutoff:
                os.remove(path)
    except Exception:
        pass


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


class _MaxRetriesExceeded(Exception):
    """Se agotaron los reintentos para un chunk sin obtener una respuesta valida."""


def _fetch_chunk(url: str, progress_callback=None) -> Optional[bytes]:
    """Descarga un chunk .bi5.

    404, o 200 con cuerpo vacio, confirman que no hay datos en esa hora
    (fin de semana, festivo, par poco liquido...): respuesta legitima, no
    se reintenta. Dukascopy no siempre usa 404 para esto; a menudo responde
    200 con 0 bytes.
    Cualquier otro resultado (429/503, HTTP inesperado, timeout, conexion
    cortada) se considera transitorio y se reintenta con backoff
    exponencial acotado a RETRY_MAX_DELAY, hasta un maximo de
    RETRY_MAX_ATTEMPTS intentos; superado ese limite se lanza
    _MaxRetriesExceeded para que el llamador cuente la hora como error en
    vez de bloquear el worker para siempre.
    """
    attempt = 0
    while True:
        try:
            r = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            if r.status_code == 200 and r.content:
                return r.content
            elif r.status_code == 404 or r.status_code == 200:
                return None  # No hay datos en esa hora (legitimo)
            if attempt >= RETRY_MAX_ATTEMPTS:
                raise _MaxRetriesExceeded(
                    f"HTTP {r.status_code} tras {attempt} intentos: {url}"
                )
            delay = min(RETRY_BASE_DELAY * (2 ** attempt), RETRY_MAX_DELAY)
            if progress_callback:
                progress_callback(f"   HTTP {r.status_code}, reintentando en {delay:.0f}s (intento {attempt + 1})...")
            time.sleep(delay)
        except requests.exceptions.RequestException as e:
            if attempt >= RETRY_MAX_ATTEMPTS:
                raise _MaxRetriesExceeded(
                    f"{e.__class__.__name__} tras {attempt} intentos: {url}"
                ) from e
            delay = min(RETRY_BASE_DELAY * (2 ** attempt), RETRY_MAX_DELAY)
            if progress_callback:
                progress_callback(f"   Error de red ({e.__class__.__name__}), reintentando en {delay:.0f}s (intento {attempt + 1})...")
            time.sleep(delay)
        attempt += 1


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

        ohlc_chunks = []
        downloaded = 0
        no_data = 0
        errors = 0
        tick_count = 0
        completed = 0

        _SENTINEL_ERROR = object()
        _SENTINEL_NODATA = object()

        def _process_hour(hour_dt):
            url = _build_url(symbol, hour_dt)
            try:
                content = _fetch_chunk(url, progress_callback)
            except Exception as e:
                if progress_callback:
                    progress_callback(f"   Error descarga {hour_dt}: {e}")
                return _SENTINEL_ERROR
            if content is None:
                return _SENTINEL_NODATA
            try:
                arr = _decode_bi5(content, hour_dt, pipet)
                return arr if len(arr) > 0 else _SENTINEL_NODATA
            except Exception as e:
                if progress_callback:
                    progress_callback(f"   Error decode {hour_dt}: {e}")
                return _SENTINEL_ERROR

        def _flush_batch(batch_ticks):
            nonlocal tick_count
            if not batch_ticks:
                return None
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
            return ohlc_batch if len(ohlc_batch) > 0 else None

        # Se descarga por lotes cronológicos de _BATCH_HOURS horas: dentro de
        # cada lote las horas se piden en paralelo (rápido), pero el batch
        # completo se resamplea a OHLC de una sola vez y en orden antes de
        # pasar al siguiente lote. Así cada vela se calcula siempre sobre un
        # rango horario contiguo, nunca sobre ticks de horas dispersas que
        # llegaron mezcladas por el orden de respuesta de la red.
        batches = [hours[i:i + _BATCH_HOURS] for i in range(0, total, _BATCH_HOURS)]

        _cleanup_old_checkpoints()
        checkpoint_key = _checkpoint_key(symbol, tf, start)
        os.makedirs(_CHECKPOINT_DIR, exist_ok=True)
        resumed_batches = 0

        for batch_idx, batch in enumerate(batches):
            cp_path = _checkpoint_batch_path(checkpoint_key, batch_idx)
            if os.path.exists(cp_path):
                try:
                    saved_n_hours, saved_chunk = pd.read_pickle(cp_path)
                    if saved_n_hours != len(batch):
                        raise ValueError(
                            f"checkpoint cubre {saved_n_hours}h, este lote ahora tiene {len(batch)}h"
                        )
                    if saved_chunk is not None and len(saved_chunk) > 0:
                        ohlc_chunks.append(saved_chunk)
                    resumed_batches += 1
                    completed += len(batch)
                    if progress_callback:
                        progress_callback(
                            f"  [{completed}/{total}] {completed/total*100:.2f}% - "
                            f"(lote {batch_idx + 1}/{len(batches)} recuperado de checkpoint)"
                        )
                    continue
                except Exception as e:
                    if progress_callback:
                        progress_callback(
                            f"  ⚠ Checkpoint del lote {batch_idx + 1}/{len(batches)} invalido "
                            f"({e.__class__.__name__}): se volvera a descargar"
                        )

            batch_ticks = []
            with ThreadPoolExecutor(max_workers=_PARALLEL_WORKERS) as executor:
                futures = [executor.submit(_process_hour, h) for h in batch]
                for f in as_completed(futures):
                    arr = f.result()
                    completed += 1
                    if arr is _SENTINEL_ERROR:
                        errors += 1
                    elif arr is _SENTINEL_NODATA:
                        no_data += 1
                    else:
                        batch_ticks.append(arr)
                        downloaded += 1
                    if progress_callback and (completed % 10 == 0 or completed == total):
                        progress_callback(
                            f"  [{completed}/{total}] {completed/total*100:.2f}% - "
                            f"OK:{downloaded} NoData:{no_data} Err:{errors}"
                        )
            batch_chunk = _flush_batch(batch_ticks)
            if batch_chunk is not None:
                ohlc_chunks.append(batch_chunk)
            try:
                pd.to_pickle(
                    (len(batch), batch_chunk if batch_chunk is not None else pd.DataFrame()),
                    cp_path,
                )
            except Exception as e:
                # No bloquear la descarga si falla el guardado del checkpoint
                # (ej. disco lleno o sin permisos), pero avisar: si se
                # interrumpe la descarga a partir de aqui, este lote no se
                # podra recuperar y habra que re-descargarlo.
                if progress_callback:
                    progress_callback(
                        f"  ⚠ No se pudo guardar el checkpoint del lote {batch_idx + 1}/{len(batches)} "
                        f"({e.__class__.__name__}): si la descarga se corta a partir de aqui, "
                        f"este lote habra que re-descargarlo"
                    )

        if resumed_batches and progress_callback:
            progress_callback(f"  Checkpoint: {resumed_batches}/{len(batches)} lotes recuperados de una descarga anterior")

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

        # Los checkpoints NO se borran al completar: como la clave no depende
        # del fin, sirven de cache incremental para la proxima descarga de
        # "toda la historia" de este simbolo/tf (solo habra que bajar las
        # horas nuevas). Los huerfanos se limpian solos via
        # _cleanup_old_checkpoints tras _CHECKPOINT_MAX_AGE_DAYS de inactividad.

        if progress_callback:
            progress_callback(f"Velas {tf} generadas: {len(ohlc):,}")
            if len(ohlc) > 0:
                progress_callback(f"  Primera vela: {ohlc['timestamp'].iloc[0]}")
                progress_callback(f"  Ultima vela:  {ohlc['timestamp'].iloc[-1]}")

        return ohlc