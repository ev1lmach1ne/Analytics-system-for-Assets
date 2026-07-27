"""
Utilidades compartidas por los scripts de reorganizacion de carpetas por
categoria (reorganizar_categorias.py para el arbol de descargas en bruto,
reorganizar_limpiados.py para el arbol de datos limpiados).

No es un script ejecutable por si mismo.
"""
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from core.config import CATEGORIAS_DESCARGA_CONOCIDAS, normalizar_categoria_descarga

# Carpetas de categoria ya canonicas: si una subcarpeta tiene uno de estos
# nombres es que viene de una reorganizacion previa, no es una carpeta de
# activo.
CATEGORIAS_CONOCIDAS = CATEGORIAS_DESCARGA_CONOCIDAS


def log(msg: str):
    print(msg, flush=True)


def slug(symbol: str) -> str:
    return symbol.lower().translate(str.maketrans({k: '_' for k in '/:<>"|?*\\'}))


def resolver_categoria_en_catalogo(catalogo, symbol_slug: str):
    """Busca symbol_slug en una lista de AssetInfo y devuelve su categoria
    normalizada, o None si no aparece."""
    for asset in catalogo:
        if slug(asset.symbol) == symbol_slug:
            return normalizar_categoria_descarga(asset.category)
    return None


def resolver_categoria_con_busqueda(provider_cls, symbol_slug: str):
    """Prueba get_catalog() y, si el proveedor soporta busqueda en vivo
    (search_symbols), cae a ella cuando el catalogo estatico no conoce el
    simbolo (activos descargados via buscador o ticker escrito a mano,
    p.ej. META, SHY, TLT, ^FCHI). Devuelve la categoria normalizada, o None
    si no se pudo resolver por ninguna via."""
    try:
        catalogo = provider_cls.get_catalog()
    except Exception as e:
        log(f"  AVISO: no se pudo obtener catalogo de {provider_cls}: {e}")
        catalogo = []
    categoria = resolver_categoria_en_catalogo(catalogo, symbol_slug)
    if categoria is not None:
        return categoria
    if hasattr(provider_cls, 'search_symbols'):
        try:
            resultados = provider_cls.search_symbols(symbol_slug.upper())
        except Exception as e:
            log(f"  AVISO: busqueda en vivo fallo para '{symbol_slug}': {e}")
            resultados = []
        categoria = resolver_categoria_en_catalogo(resultados, symbol_slug)
        if categoria is not None:
            return categoria
    return None


def mover_carpeta(origen: str, destino: str):
    """Mueve la carpeta origen a destino. Si destino ya existe (reejecucion
    parcial anterior), fusiona archivo por archivo sin sobrescribir."""
    if not os.path.exists(destino):
        os.makedirs(os.path.dirname(destino), exist_ok=True)
        shutil.move(origen, destino)
        return
    for nombre in os.listdir(origen):
        src = os.path.join(origen, nombre)
        dst = os.path.join(destino, nombre)
        if os.path.exists(dst):
            log(f"  AVISO: ya existe {dst}, se omite {src}")
            continue
        shutil.move(src, dst)
    if not os.listdir(origen):
        os.rmdir(origen)
