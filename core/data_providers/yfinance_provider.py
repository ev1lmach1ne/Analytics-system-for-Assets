"""
Proveedor de datos Yahoo Finance usando la libreria yfinance.

yfinance es gratis y cubre forex, metales, cripto, acciones e indices.
Intraday limitado: 1m→7 dias, 5m/15m→60 dias, 1h→730 dias.
4h no esta soportado nativamente; se descarga en 1h y se resamplea.
"""

import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import List, Optional

from .base_provider import BaseProvider, AssetInfo, TF_TO_PANDAS

YF_TF_MAP = {
    '1m': '1m', '5m': '5m', '15m': '15m',
    '1h': '1h', '4h': '1h', '1d': '1d',
}

_CATALOG = [
    AssetInfo('EURUSD=X', 'Euro / US Dollar', 'Forex', '2005-01-01'),
    AssetInfo('GBPUSD=X', 'British Pound / US Dollar', 'Forex', '2005-01-01'),
    AssetInfo('USDJPY=X', 'US Dollar / Japanese Yen', 'Forex', '2005-01-01'),
    AssetInfo('AUDUSD=X', 'Australian Dollar / US Dollar', 'Forex', '2005-01-01'),
    AssetInfo('USDCAD=X', 'US Dollar / Canadian Dollar', 'Forex', '2005-01-01'),
    AssetInfo('USDCHF=X', 'US Dollar / Swiss Franc', 'Forex', '2005-01-01'),
    AssetInfo('NZDUSD=X', 'New Zealand Dollar / US Dollar', 'Forex', '2005-01-01'),
    AssetInfo('EURGBP=X', 'Euro / British Pound', 'Forex', '2008-01-01'),
    AssetInfo('EURJPY=X', 'Euro / Japanese Yen', 'Forex', '2008-01-01'),
    AssetInfo('GBPJPY=X', 'British Pound / Japanese Yen', 'Forex', '2008-01-01'),
    AssetInfo('GC=F', 'Gold Futures / US Dollar', 'Metal', '2000-01-01'),
    AssetInfo('SI=F', 'Silver Futures / US Dollar', 'Metal', '2000-01-01'),
    AssetInfo('GLD', 'SPDR Gold Trust ETF', 'Metal-ETF', '2004-11-01'),
    AssetInfo('SLV', 'iShares Silver Trust ETF', 'Metal-ETF', '2006-04-01'),
    AssetInfo('BTC-USD', 'Bitcoin / US Dollar', 'Cripto', '2015-01-01'),
    AssetInfo('ETH-USD', 'Ethereum / US Dollar', 'Cripto', '2017-07-01'),
    AssetInfo('^GSPC', 'S&P 500', 'Indice', '1990-01-01'),
    AssetInfo('^DJI', 'Dow Jones Industrial', 'Indice', '1990-01-01'),
    AssetInfo('^IXIC', 'NASDAQ Composite', 'Indice', '1990-01-01'),
    AssetInfo('^VIX', 'CBOE Volatility Index', 'Indice', '2005-01-01'),
    AssetInfo('AAPL', 'Apple Inc.', 'Accion', '1990-01-01'),
    AssetInfo('MSFT', 'Microsoft Corp.', 'Accion', '1990-01-01'),
    AssetInfo('GOOGL', 'Alphabet Inc.', 'Accion', '2005-01-01'),
    AssetInfo('AMZN', 'Amazon.com Inc.', 'Accion', '1997-01-01'),
    AssetInfo('TSLA', 'Tesla Inc.', 'Accion', '2010-07-01'),
    AssetInfo('NVDA', 'NVIDIA Corp.', 'Accion', '1999-01-01'),
]


class YFinanceProvider(BaseProvider):
    name = 'yfinance'

    @staticmethod
    def get_catalog() -> List[AssetInfo]:
        return _CATALOG.copy()

    @staticmethod
    def search_symbols(query: str, max_results: int = 15) -> List[AssetInfo]:
        """Busca simbolos en el autocompletado real de Yahoo Finance.

        A diferencia de get_catalog() (lista fija precargada), esto golpea
        la API de Yahoo en vivo y cubre cualquier accion/indice/forex/
        commodity que Yahoo conozca, sin tener que mantener un catalogo
        manual.
        """
        import yfinance as yf
        query = query.strip()
        if not query:
            return []
        try:
            s = yf.Search(query, max_results=max_results, news_count=0, lists_count=0)
        except Exception:
            return []
        return [
            AssetInfo(
                q['symbol'],
                q.get('shortname') or q.get('longname') or q['symbol'],
                q.get('quoteType', ''),
                None,
            )
            for q in s.quotes if q.get('symbol')
        ]

    @staticmethod
    def get_available_range(symbol: str, progress_callback=None) -> Optional[tuple]:
        for asset in _CATALOG:
            if asset.symbol == symbol and asset.max_history_start:
                start_dt = datetime.strptime(asset.max_history_start, '%Y-%m-%d').replace(tzinfo=timezone.utc)
                end_dt = datetime.now(timezone.utc)
                if progress_callback:
                    progress_callback(f"  Rango estimado: {start_dt.date()} -> {end_dt.date()}")
                return (start_dt, end_dt)
        # Simbolo no catalogado (ticker personalizado): Yahoo no falla con
        # fechas anteriores al inicio real de cotizacion, simplemente
        # devuelve datos desde que existan. Se usa un inicio generico amplio
        # para que 'Historial Completo' funcione tambien con tickers libres.
        start_dt = datetime(1970, 1, 1, tzinfo=timezone.utc)
        end_dt = datetime.now(timezone.utc)
        if progress_callback:
            progress_callback(f"  Simbolo no catalogado: se pedira historial desde {start_dt.date()} (Yahoo recortara al inicio real)")
        return (start_dt, end_dt)

    @staticmethod
    def download_ohlc(symbol: str, tf: str,
                      start: Optional[datetime] = None,
                      end: Optional[datetime] = None,
                      progress_callback=None) -> pd.DataFrame:
        import yfinance as yf

        if tf not in ('1m', '5m', '15m', '1h', '4h', '1d'):
            raise ValueError(f"TF '{tf}' no soportado por yfinance")

        if start is None or end is None:
            rango = YFinanceProvider.get_available_range(symbol, progress_callback)
            if rango is None:
                raise RuntimeError(f"No se pudo determinar el rango para {symbol}")
            if start is None:
                start = rango[0]
            if end is None:
                end = rango[1]

        if start.tzinfo is not None:
            start = start.replace(tzinfo=None)
        if end.tzinfo is not None:
            end = end.replace(tzinfo=None)

        fetch_tf = YF_TF_MAP.get(tf, '1h')

        # El limite de Yahoo depende del intervalo real solicitado a su API
        # (fetch_tf), no del tf logico pedido por el usuario: '4h' se pide
        # como '1h' y luego se resamplea, asi que el limite de 730 dias de
        # '1h' aplica igual.
        _MAX_INTRADAY_DAYS = {'1m': 7, '5m': 60, '15m': 60, '1h': 730}
        max_days = _MAX_INTRADAY_DAYS.get(fetch_tf)
        if max_days and end is not None:
            earliest = end - timedelta(days=max_days)
            if start is None or start < earliest:
                if progress_callback:
                    old_start = start.date() if start else 'inicio'
                    progress_callback(
                        f"  Yahoo limita datos {fetch_tf} a {max_days} dias. "
                        f"Ajustando inicio: {old_start} -> {earliest.date()}"
                    )
                start = earliest

        if progress_callback:
            progress_callback(f"Descargando {symbol}: {start.date()} -> {end.date()}")
            progress_callback(f"Intervalo: {fetch_tf}" + (f" (resampleando a {tf})" if tf == '4h' else ''))

        try:
            data = yf.download(
                symbol, start=start, end=end,
                interval=fetch_tf, progress=False, auto_adjust=True,
            )
        except Exception as e:
            raise RuntimeError(f"Error descargando de Yahoo Finance: {e}")

        if data is None or len(data) == 0:
            raise RuntimeError(f"Yahoo Finance devolvio datos vacios para {symbol}")

        if isinstance(data.columns, pd.MultiIndex):
            data = data.xs(symbol, axis=1, level=1)
            data.index.name = None

        required = {'Open', 'High', 'Low', 'Close', 'Volume'}
        if not required.issubset(data.columns):
            missing = required - set(data.columns)
            raise RuntimeError(f"Columnas faltantes en respuesta de Yahoo: {missing}")

        if progress_callback:
            progress_callback(f"  Datos crudos: {len(data):,} velas {fetch_tf}")

        if tf == '4h':
            rule = '4h'
            ohlc = pd.DataFrame({
                'open': data['Open'].resample(rule).first(),
                'high': data['High'].resample(rule).max(),
                'low': data['Low'].resample(rule).min(),
                'close': data['Close'].resample(rule).last(),
                'volume': data['Volume'].resample(rule).sum(),
            }).dropna()
            data = ohlc

            if progress_callback:
                progress_callback(f"  Resampleado a {tf}: {len(data):,} velas")
        else:
            data = data.rename(columns={
                'Open': 'open', 'High': 'high', 'Low': 'low',
                'Close': 'close', 'Volume': 'volume',
            })
            data = data[['open', 'high', 'low', 'close', 'volume']].dropna()

        data = data.reset_index()

        ts_col = 'Date' if 'Date' in data.columns else ('Datetime' if 'Datetime' in data.columns else data.columns[0])
        if ts_col not in ('open',):
            data.rename(columns={ts_col: 'timestamp'}, inplace=True)

        if data['timestamp'].dt.tz is None:
            data['timestamp'] = data['timestamp'].dt.tz_localize('UTC')
        else:
            data['timestamp'] = data['timestamp'].dt.tz_convert('UTC')

        data['spread'] = 0
        data = data[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'spread']]

        if progress_callback:
            progress_callback(f"Velas {tf} generadas: {len(data):,}")
            if len(data) > 0:
                progress_callback(f"  Primera vela: {data['timestamp'].iloc[0]}")
                progress_callback(f"  Ultima vela:  {data['timestamp'].iloc[-1]}")

        return data
