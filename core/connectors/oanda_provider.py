"""Proveedor de datos OANDA (Forex, Indices, Materias Primas)."""

import pandas as pd
import os
from datetime import datetime, timezone
from typing import List, Optional, Dict
from core.data_providers.base_provider import BaseProvider, AssetInfo

_OANDA_TF = {
    '1m': 'M1', '5m': 'M5', '15m': 'M15',
    '1h': 'H1', '4h': 'H4', '1d': 'D',
}

_OANDA_GRANULARITY_MS = {
    'M1': 60000, 'M5': 300000, 'M15': 900000,
    'H1': 3600000, 'H4': 14400000, 'D': 86400000,
}

_OANDA_ASSETS = [
    AssetInfo('EUR_USD', 'Euro / US Dollar', 'Forex', '2003-05-04'),
    AssetInfo('GBP_USD', 'British Pound / US Dollar', 'Forex', '2003-05-04'),
    AssetInfo('USD_JPY', 'US Dollar / Japanese Yen', 'Forex', '2003-05-04'),
    AssetInfo('USD_CHF', 'US Dollar / Swiss Franc', 'Forex', '2003-05-04'),
    AssetInfo('AUD_USD', 'Australian Dollar / US Dollar', 'Forex', '2003-05-04'),
    AssetInfo('USD_CAD', 'US Dollar / Canadian Dollar', 'Forex', '2003-05-04'),
    AssetInfo('NZD_USD', 'New Zealand Dollar / US Dollar', 'Forex', '2003-05-04'),
    AssetInfo('EUR_GBP', 'Euro / British Pound', 'Forex', '2003-05-04'),
    AssetInfo('EUR_JPY', 'Euro / Japanese Yen', 'Forex', '2003-05-04'),
    AssetInfo('GBP_JPY', 'British Pound / Japanese Yen', 'Forex', '2003-05-04'),
    AssetInfo('EUR_CHF', 'Euro / Swiss Franc', 'Forex', '2003-05-04'),
    AssetInfo('EUR_AUD', 'Euro / Australian Dollar', 'Forex', '2003-05-04'),
    AssetInfo('GBP_AUD', 'British Pound / Australian Dollar', 'Forex', '2003-05-04'),
    AssetInfo('GBP_CAD', 'British Pound / Canadian Dollar', 'Forex', '2003-05-04'),
    AssetInfo('GBP_CHF', 'British Pound / Swiss Franc', 'Forex', '2003-05-04'),
    AssetInfo('GBP_NZD', 'British Pound / New Zealand Dollar', 'Forex', '2003-05-04'),
    AssetInfo('AUD_CAD', 'Australian Dollar / Canadian Dollar', 'Forex', '2003-05-04'),
    AssetInfo('AUD_CHF', 'Australian Dollar / Swiss Franc', 'Forex', '2003-05-04'),
    AssetInfo('AUD_JPY', 'Australian Dollar / Japanese Yen', 'Forex', '2003-05-04'),
    AssetInfo('AUD_NZD', 'Australian Dollar / New Zealand Dollar', 'Forex', '2003-05-04'),
    AssetInfo('CAD_CHF', 'Canadian Dollar / Swiss Franc', 'Forex', '2003-05-04'),
    AssetInfo('CAD_JPY', 'Canadian Dollar / Japanese Yen', 'Forex', '2003-05-04'),
    AssetInfo('CHF_JPY', 'Swiss Franc / Japanese Yen', 'Forex', '2003-05-04'),
    AssetInfo('NZD_CAD', 'New Zealand Dollar / Canadian Dollar', 'Forex', '2003-05-04'),
    AssetInfo('NZD_CHF', 'New Zealand Dollar / Swiss Franc', 'Forex', '2003-05-04'),
    AssetInfo('NZD_JPY', 'New Zealand Dollar / Japanese Yen', 'Forex', '2003-05-04'),
    AssetInfo('EUR_NZD', 'Euro / New Zealand Dollar', 'Forex', '2003-05-04'),
    AssetInfo('EUR_CAD', 'Euro / Canadian Dollar', 'Forex', '2003-05-04'),
    AssetInfo('XAU_USD', 'Gold / US Dollar', 'Commodities', '2003-05-04'),
    AssetInfo('XAG_USD', 'Silver / US Dollar', 'Commodities', '2003-05-04'),
    AssetInfo('XCU_USD', 'Copper / US Dollar', 'Commodities', '2003-05-04'),
    AssetInfo('WTICO_USD', 'WTI Crude Oil / US Dollar', 'Commodities', '2003-05-04'),
    AssetInfo('NATGAS_USD', 'Natural Gas / US Dollar', 'Commodities', '2003-05-04'),
    AssetInfo('JP225_USD', 'Japan 225 / US Dollar', 'Indices', '2003-05-04'),
    AssetInfo('DE30_EUR', 'Germany 30 / Euro', 'Indices', '2003-05-04'),
    AssetInfo('UK100_GBP', 'UK 100 / British Pound', 'Indices', '2003-05-04'),
    AssetInfo('FR40_EUR', 'France 40 / Euro', 'Indices', '2003-05-04'),
    AssetInfo('EU50_EUR', 'Europe 50 / Euro', 'Indices', '2003-05-04'),
    AssetInfo('US30_USD', 'US Wall Street 30 / US Dollar', 'Indices', '2003-05-04'),
    AssetInfo('SPX500_USD', 'US S&P 500 / US Dollar', 'Indices', '2003-05-04'),
    AssetInfo('NAS100_USD', 'US Nasdaq 100 / US Dollar', 'Indices', '2003-05-04'),
    AssetInfo('HK33_HKD', 'Hong Kong 33 / HKD', 'Indices', '2003-05-04'),
    AssetInfo('CHINA50_USD', 'China A50 / US Dollar', 'Indices', '2003-05-04'),
]


class OANDAProvider(BaseProvider):
    name = 'OANDA'

    @staticmethod
    def get_catalog() -> List[AssetInfo]:
        return _OANDA_ASSETS.copy()

    @staticmethod
    def get_available_range(symbol: str,
                            progress_callback=None) -> Optional[tuple]:
        start_dt = datetime(2003, 5, 4, tzinfo=timezone.utc)
        end_dt = datetime.now(timezone.utc)
        if progress_callback:
            progress_callback(f"  Rango: {start_dt.date()} -> {end_dt.date()}")
        return (start_dt, end_dt)

    @staticmethod
    def download_ohlc(symbol: str, tf: str,
                      start: Optional[datetime] = None,
                      end: Optional[datetime] = None,
                      progress_callback=None) -> pd.DataFrame:
        if tf not in _OANDA_TF:
            raise ValueError(f"TF '{tf}' no soportado por OANDA")

        api_key = os.environ.get('OANDA_API_KEY', '')
        account_id = os.environ.get('OANDA_ACCOUNT_ID', '')
        env = os.environ.get('OANDA_ENVIRONMENT', 'practice')

        if not api_key:
            raise RuntimeError("OANDA_API_KEY no configurada. Configurala en el conector [+]")

        granularity = _OANDA_TF[tf]
        step_ms = _OANDA_GRANULARITY_MS[granularity]

        if start is None:
            start = datetime(2003, 5, 4, tzinfo=timezone.utc)
        if end is None:
            end = datetime.now(timezone.utc)

        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)

        est = int((end.timestamp() - start.timestamp()) * 1000 / step_ms)
        if progress_callback:
            progress_callback(f"Conectando a OANDA...")
            progress_callback(f"Descargando {symbol}: {start.date()} -> {end.date()}")
            progress_callback(f"Timeframe: {tf}  |  ~{est:,} velas estimadas")

        import oandapyV20
        import oandapyV20.endpoints.instruments as instruments

        base = 'https://api-fxpractice.oanda.com'
        if env == 'live':
            base = 'https://api-fxtrade.oanda.com'

        client = oandapyV20.API(access_token=api_key, environment=env)
        client._base_url = base

        all_data = []
        current_start = start

        while current_start < end:
            params = {
                'granularity': granularity,
                'from': current_start.strftime('%Y-%m-%dT%H:%M:%SZ'),
                'to': end.strftime('%Y-%m-%dT%H:%M:%SZ'),
                'price': 'MBA',
            }
            try:
                r = instruments.InstrumentsCandles(
                    instrument=symbol, params=params
                )
                client.request(r)
            except Exception as e:
                raise RuntimeError(f"Error OANDA API: {e}")

            candles = r.response.get('candles', [])

            if not candles:
                break

            for c in candles:
                if c['complete']:
                    ts = c['time']
                    mid = c.get('mid', {})
                    o = float(mid.get('o', 0))
                    h = float(mid.get('h', 0))
                    l = float(mid.get('l', 0))
                    c_close = float(mid.get('c', 0))
                    v = float(c.get('volume', 0))
                    all_data.append((ts, o, h, l, c_close, v))

            last_ts = datetime.fromisoformat(candles[-1]['time'].replace('Z', '+00:00'))
            current_start = last_ts

            if progress_callback and len(all_data) % 10000 < 5000:
                progress_callback(f"  {len(all_data):,} velas descargadas...")

            if len(candles) < 5000:
                break

        if not all_data:
            raise RuntimeError(f"No se obtuvieron datos para {symbol}")

        df = pd.DataFrame(all_data,
                          columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df = df.drop_duplicates(subset='timestamp').set_index('timestamp').sort_index()

        df = df[df.index <= pd.Timestamp(end)]
        df['spread'] = 0
        df = df.reset_index()
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'spread']]

        if progress_callback:
            progress_callback(f"Velas {tf} generadas: {len(df):,}")
            if len(df) > 0:
                progress_callback(f"  Primera vela: {df['timestamp'].iloc[0]}")
                progress_callback(f"  Ultima vela:  {df['timestamp'].iloc[-1]}")

        return df
