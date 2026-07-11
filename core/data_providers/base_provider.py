"""
Clase abstracta base para todos los proveedores de datos de mercado.

Cada provider (Dukascopy, Binance, yfinance...) implementa esta interfaz
para que la GUI y los scripts de descarga puedan usarlos de forma uniforme.
"""
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Dict, Optional
import pandas as pd


class AssetInfo:
    """Información de un activo disponible en un provider."""

    def __init__(self, symbol: str, name: str, category: str = "",
                 max_history_start: Optional[str] = None):
        self.symbol = symbol
        self.name = name
        self.category = category
        self.max_history_start = max_history_start

    def __repr__(self):
        return f"AssetInfo({self.symbol!r}, {self.name!r}, {self.category!r})"


class BaseProvider(ABC):
    """Interfaz común para todos los proveedores de datos."""

    #: Nombre legible del provider (ej: "Dukascopy")
    name: str = "Base"

    @staticmethod
    @abstractmethod
    def get_catalog() -> List[AssetInfo]:
        """Devuelve la lista de activos disponibles en este provider."""
        ...

    @staticmethod
    @abstractmethod
    def download_ohlc(symbol: str, tf: str,
                     start: Optional[datetime] = None,
                     end: Optional[datetime] = None,
                     progress_callback=None) -> pd.DataFrame:
        """
        Descarga datos OHLC para un símbolo y timeframe.

        Args:
            symbol: Símbolo del activo (ej: 'EURUSD', 'XAUUSD')
            tf: Timeframe ('1m', '5m', '15m', '1h', '4h', '1d')
            start: Fecha de inicio (None = máxima historia disponible)
            end: Fecha de fin (None = hoy)
            progress_callback: Función opcional (msg: str) -> None
                para reportar progreso en tiempo real.

        Returns:
            pd.DataFrame con columnas: timestamp, open, high, low, close,
            volume, spread. Indexado por timestamp en UTC.
        """
        ...

    @staticmethod
    @abstractmethod
    def get_available_range(symbol: str,
                            progress_callback=None) -> Optional[tuple]:
        """
        Devuelve el rango de fechas disponibles para un símbolo.

        Returns:
            Tupla (start: datetime, end: datetime) o None si no se
            puede determinar.
        """
        ...


TF_TO_PANDAS = {
    '1m': '1min', '5m': '5min', '15m': '15min',
    '1h': '1h', '4h': '4h', '1d': '1D',
}