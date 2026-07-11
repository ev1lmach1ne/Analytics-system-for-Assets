"""
Script CLI para descargar datos de mercado.

Se lanza via QProcess desde la GUI (tab_descargar.py) o directamente
desde linea de comandos. Lee la configuracion de variables de entorno
y guarda el CSV resultante en la carpeta del activo.

Variables de entorno:
  DOWNLOAD_PROVIDER  - "dukascopy"  (futuro: "binance", "yfinance")
  DOWNLOAD_SYMBOL    - Simbolo del activo (ej: "EURUSD", "XAUUSD")
  DOWNLOAD_TF        - Timeframe: "1m", "5m", "15m", "1h", "4h", "1d"
  DOWNLOAD_START     - (opcional) Fecha inicio YYYY-MM-DD. Default: max history
  DOWNLOAD_END       - (opcional) Fecha fin YYYY-MM-DD. Default: hoy
  DOWNLOAD_OUTPUT    - (opcional) Ruta de salida. Default: auto

Salida CSV:
  D:\\DATOS\\Activos\\<symbol>\\<symbol>_<tf>_<MM-YYYY>_to_<MM-YYYY>.csv
  Columnas: timestamp,open,high,low,close,volume,spread

Uso desde CLI:
  python descargar_datos.py
  (con env vars configuradas)
"""
import os
import sys
from datetime import datetime, timezone

# Path setup para poder importar core
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from core.config import BASE_DATA
from core.data_providers.dukascopy_provider import DukascopyProvider
from core.data_providers.yfinance_provider import YFinanceProvider
from core.data_providers.ccxt_provider import CCXTProvider
from core.data_providers.base_provider import BaseProvider
from core.data_providers.hyperliquid_provider import HyperliquidProvider
from core.connectors.oanda_provider import OANDAProvider

PROVIDERS = {
    'dukascopy': DukascopyProvider,
    'yfinance': YFinanceProvider,
    'ccxt': CCXTProvider,
    'hyperliquid': HyperliquidProvider,
    'oanda': OANDAProvider,
}


def _log(msg: str):
    """Log con flush para que la GUI vea el output en tiempo real."""
    print(msg, flush=True)


def _determine_output_path(provider_name: str, symbol: str, tf: str, df) -> str:
    """
    Construye la ruta de salida:
    D:\\DATOS\\Activos\\<fuente>\\<simbolo>\\<simbolo>_<tf>_<YYYY-MM-DD>_to_<YYYY-MM-DD>.csv
    """
    symbol_slug = symbol.lower().replace('/', '_')
    asset_dir = os.path.join(BASE_DATA, provider_name, symbol_slug)
    os.makedirs(asset_dir, exist_ok=True)

    first_ts = df['timestamp'].iloc[0]
    last_ts = df['timestamp'].iloc[-1]

    start_str = first_ts.strftime('%Y-%m-%d')
    end_str = last_ts.strftime('%Y-%m-%d')
    filename = f"{symbol_slug}_{tf}_{start_str}_to_{end_str}.csv"
    return os.path.join(asset_dir, filename)


def main():
    # Leer env vars
    provider_name = os.environ.get('DOWNLOAD_PROVIDER', 'dukascopy').lower()
    symbol = os.environ.get('DOWNLOAD_SYMBOL', '')
    tf = os.environ.get('DOWNLOAD_TF', '1h')
    start_str = os.environ.get('DOWNLOAD_START', '')
    end_str = os.environ.get('DOWNLOAD_END', '')
    custom_output = os.environ.get('DOWNLOAD_OUTPUT', '')

    _log("=" * 60)
    _log(f"DESCARGA DE DATOS")
    _log(f"  Provider: {provider_name}")
    _log(f"  Simbolo:  {symbol}")
    _log(f"  TF:       {tf}")
    if start_str:
        _log(f"  Desde:    {start_str}")
    if end_str:
        _log(f"  Hasta:    {end_str}")
    _log("=" * 60)

    # Validar
    if not symbol:
        _log("ERROR: DOWNLOAD_SYMBOL no especificado.")
        sys.exit(1)

    if provider_name not in PROVIDERS:
        _log(f"ERROR: Provider '{provider_name}' no soportado.")
        _log(f"  Disponibles: {list(PROVIDERS.keys())}")
        sys.exit(1)

    # Parsear fechas opcionales
    start = None
    end = None
    if start_str:
        try:
            start = datetime.strptime(start_str, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        except ValueError:
            _log(f"ERROR: Formato de fecha inicio invalido: {start_str}")
            _log("  Formato esperado: YYYY-MM-DD")
            sys.exit(1)
    if end_str:
        try:
            end = datetime.strptime(end_str, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        except ValueError:
            _log(f"ERROR: Formato de fecha fin invalido: {end_str}")
            _log("  Formato esperado: YYYY-MM-DD")
            sys.exit(1)

    provider = PROVIDERS[provider_name]

    # Descargar
    _log("")
    _log("Iniciando descarga...")
    try:
        df = provider.download_ohlc(
            symbol=symbol, tf=tf, start=start, end=end,
            progress_callback=_log,
        )
    except Exception as e:
        _log(f"ERROR durante la descarga: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(2)

    if df is None or len(df) == 0:
        _log("ERROR: No se obtuvieron datos.")
        sys.exit(3)

    _log("")
    _log(f"Datos descargados: {len(df):,} velas {tf}")
    _log(f"  Primera vela: {df['timestamp'].iloc[0]}")
    _log(f"  Ultima vela:  {df['timestamp'].iloc[-1]}")

    # Formatear timestamp para CSV (ISO con Z, como preparar_datos.py espera)
    df_to_save = df.copy()
    # Asegurar que timestamp sea string ISO UTC
    if df_to_save['timestamp'].dt.tz is not None:
        df_to_save['timestamp'] = df_to_save['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%S.000000Z')
    else:
        df_to_save['timestamp'] = df_to_save['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%S.000000Z')

    # Rellenar volumen y spread si faltan
    if 'volume' not in df_to_save.columns:
        df_to_save['volume'] = 0
    if 'spread' not in df_to_save.columns:
        df_to_save['spread'] = 0

    # Reordenar columnas
    df_to_save = df_to_save[['timestamp', 'open', 'high', 'low', 'close',
                             'volume', 'spread']]

    # Determinar ruta de salida
    if custom_output:
        output_path = custom_output
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    else:
        output_path = _determine_output_path(provider_name, symbol, tf, df)

    # Guardar
    df_to_save.to_csv(output_path, index=False)
    _log("")
    _log(f"CSV guardado en:")
    _log(f"  {output_path}")
    _log(f"  Tamano: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
    _log("")
    _log("=" * 60)
    _log(f"DESCARGA COMPLETADA")
    _log(f"  Para usarlo: ve a la pestana Importar y selecciona el CSV.")
    _log("=" * 60)


if __name__ == '__main__':
    main()