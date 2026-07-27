"""
Script CLI para reorganizar los activos ya descargados de:
  BASE_DATA/<proveedor>/<activo>/
a la nueva estructura con carpeta de categoria:
  BASE_DATA/<proveedor>/<categoria>/<activo>/

Se lanza en segundo plano desde la GUI (tab_descargar.py). Idempotente: si
se ejecuta varias veces no duplica ni rompe nada -- las carpetas de
categoria ya creadas se detectan y se saltan (no se tratan como carpetas
de activo), y si el destino de una carpeta ya existe se fusiona archivo
por archivo sin sobrescribir.

Uso desde CLI:
  python reorganizar_categorias.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.config import BASE_DATA, normalizar_categoria_descarga
from core.data_providers.dukascopy_provider import DukascopyProvider
from core.data_providers.yfinance_provider import YFinanceProvider
from core.data_providers.ccxt_provider import CCXTProvider
from core.data_providers.hyperliquid_provider import HyperliquidProvider
from core.connectors import load_connectors, provider_class_for_type
from categorias_comun import (
    CATEGORIAS_CONOCIDAS, log, mover_carpeta, resolver_categoria_con_busqueda,
)

PROVIDERS = {
    'dukascopy': DukascopyProvider,
    'yfinance': YFinanceProvider,
    'ccxt': CCXTProvider,
    'hyperliquid': HyperliquidProvider,
}


def _categoria_de_symbol(provider_key: str, provider_cls, symbol_slug: str) -> str:
    """ccxt es siempre Cripto, no hace falta pedir el catalogo. Para el
    resto, se prueba el catalogo del proveedor y, si hace falta, busqueda
    en vivo."""
    if provider_key == 'ccxt':
        return normalizar_categoria_descarga('Cripto')
    categoria = resolver_categoria_con_busqueda(provider_cls, symbol_slug)
    return categoria if categoria is not None else normalizar_categoria_descarga('')


def _reintentar_otros(provider_key: str, provider_cls, provider_dir: str):
    """OTROS es un cajon de 'no se pudo clasificar', no una categoria
    definitiva: en cada ejecucion se reintenta por si ahora se puede
    resolver (p.ej. porque se añadio busqueda en vivo como fallback)."""
    otros_dir = os.path.join(provider_dir, 'OTROS')
    if not os.path.isdir(otros_dir):
        return 0, 0
    movidas, omitidas = 0, 0
    for nombre in sorted(os.listdir(otros_dir)):
        ruta = os.path.join(otros_dir, nombre)
        if not os.path.isdir(ruta):
            continue
        categoria = _categoria_de_symbol(provider_key, provider_cls, nombre)
        if categoria == 'OTROS':
            omitidas += 1
            continue
        destino = os.path.join(provider_dir, categoria, nombre)
        log(f"  {provider_key}/OTROS/{nombre}  ->  {provider_key}/{categoria}/{nombre}")
        try:
            mover_carpeta(ruta, destino)
            movidas += 1
        except OSError as e:
            log(f"  ERROR moviendo {ruta}: {e}")
    return movidas, omitidas


def _reorganizar_proveedor(provider_key: str, provider_cls):
    provider_dir = os.path.join(BASE_DATA, provider_key)
    if not os.path.isdir(provider_dir):
        return 0, 0
    movidas, omitidas = _reintentar_otros(provider_key, provider_cls, provider_dir)
    for nombre in sorted(os.listdir(provider_dir)):
        ruta = os.path.join(provider_dir, nombre)
        if not os.path.isdir(ruta):
            continue
        if nombre.upper() in CATEGORIAS_CONOCIDAS:
            omitidas += 1
            continue
        categoria = _categoria_de_symbol(provider_key, provider_cls, nombre)
        destino = os.path.join(provider_dir, categoria, nombre)
        log(f"  {provider_key}/{nombre}  ->  {provider_key}/{categoria}/{nombre}")
        try:
            mover_carpeta(ruta, destino)
            movidas += 1
        except OSError as e:
            log(f"  ERROR moviendo {ruta}: {e}")
    return movidas, omitidas


def main():
    log("=" * 60)
    log("REORGANIZACION DE CARPETAS POR CATEGORIA (descargas)")
    log("=" * 60)

    registro = dict(PROVIDERS)
    for conector in load_connectors():
        tipo = conector.get('type')
        cls = provider_class_for_type(tipo) if tipo else None
        if cls is not None:
            registro[tipo] = cls

    total_movidas = 0
    total_omitidas = 0
    for provider_key, provider_cls in registro.items():
        log("")
        log(f"Proveedor: {provider_key}")
        movidas, omitidas = _reorganizar_proveedor(provider_key, provider_cls)
        total_movidas += movidas
        total_omitidas += omitidas
        if movidas == 0 and omitidas == 0:
            log("  (sin carpeta de descargas, o ya reorganizado)")

    log("")
    log("=" * 60)
    log(f"COMPLETADO: {total_movidas} carpetas movidas, {total_omitidas} ya organizadas")
    log("=" * 60)


if __name__ == '__main__':
    main()
