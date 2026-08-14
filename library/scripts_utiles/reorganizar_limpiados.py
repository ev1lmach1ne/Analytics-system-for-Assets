"""
Script CLI para reorganizar los activos ya limpiados de:
  LIMPIADOS_DIR/<activo>/
a la nueva estructura con carpeta de categoria:
  LIMPIADOS_DIR/<categoria>/<activo>/

Ademas repara las rutas absolutas guardadas en Favoritos/*/favorito.json y
en Limpiados/comparador_config.json para que sigan apuntando al CSV
correcto tras moverse.

Se lanza en segundo plano desde la GUI (tab_descargar.py), encadenado tras
reorganizar_categorias.py. Idempotente: si se ejecuta varias veces no
duplica ni rompe nada.

Uso desde CLI:
  python reorganizar_limpiados.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.config import LIMPIADOS_DIR, FAVORITOS_DIR
from categorias_comun import CATEGORIAS_CONOCIDAS, log, mover_carpeta

# Entradas de LIMPIADOS_DIR que no son carpetas de activo.
_ENTRADAS_NO_ACTIVO = {'Informes'}

# El 'nombre' de un activo limpiado es texto libre que escribe el usuario
# (no un simbolo de un proveedor), asi que NO se busca por nombre contra
# catalogos/buscadores en vivo -- se probo y da falsos positivos peligrosos
# (p.ej. el activo "dax", el indice aleman, encontro por busqueda de texto
# un ticker real pero no relacionado "DAX" que es un ETF, y se catalogaria
# mal con total confianza). La unica fuente fiable es el .meta.json que ya
# escribio la pestana Importar con el 'Tipo' que el usuario eligio a mano.
# Ese 'activo' (FUTURO/FOREX/STOCK/CRYPTO) es mas grueso que las categorias
# de descarga, y ademas 'STOCK' ahi mezcla acciones, ETFs e indices sin
# distincion -- asi que solo se resuelven con confianza FOREX/CRYPTO/FUTURO
# (mapeo directo, sin ambiguedad) y todo lo demas (incluido 'STOCK', o sin
# meta.json) se deja en OTROS para que el usuario lo revise a mano.
_ACTIVO_A_CATEGORIA = {
    'FOREX': 'FOREX',
    'CRYPTO': 'CRYPTO',
    'FUTURO': 'FUTURO',
}


def _meta_de_carpeta(asset_dir):
    """Lee el primer .meta.json que encuentre dentro de la carpeta del
    activo, o None si no hay ninguno legible."""
    try:
        nombres = os.listdir(asset_dir)
    except OSError:
        return None
    for nombre in nombres:
        if not nombre.endswith('.meta.json'):
            continue
        try:
            with open(os.path.join(asset_dir, nombre), encoding='utf-8') as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
    return None


def _categoria_de_meta(meta):
    """Categoria segun un .meta.json: primero 'categoria' explicita, que la
    pestana Importar deduce de la ruta de descarga de origen (p.ej.
    ccxt/CRYPTO/link_usdt -> CRYPTO): es un dato fino y fiable. Solo si
    falta o no es una categoria conocida se cae a 'activo', que sale de un
    desplegable que arranca en Futuro/Cfd y que el usuario puede no haber
    tocado (ver comentario de _ACTIVO_A_CATEGORIA)."""
    if not meta:
        return None
    categoria = (meta.get('categoria') or '').upper()
    if categoria in CATEGORIAS_CONOCIDAS and categoria != 'OTROS':
        return categoria
    activo = (meta.get('activo') or '').upper()
    return _ACTIVO_A_CATEGORIA.get(activo)


def _categoria_de_carpeta(asset_dir: str) -> str:
    """Resuelve la categoria de una carpeta de activo de forma determinista."""
    return _categoria_de_meta(_meta_de_carpeta(asset_dir)) or 'OTROS'


def _categoria_de_hermana(nombre):
    """Categoria de una carpeta hermana ya clasificada (LIMPIADOS_DIR/<cat>/
    <nombre>) cuyo .meta.json declare una 'categoria' EXPLICITA (no el
    'activo' grueso).

    Cura el estado que dejaba el bug de sesion_config.json: el CSV limpio
    terminaba en OTROS (sin meta propio) mientras el .meta.json se escribia
    en la carpeta de categoria correcta — p.ej. FOREX/eurgbp/...meta.json
    con OTROS/eurgbp/...csv huerfano. La hermana con categoria explicita es
    la fuente fiable y sirve para reunir el CSV con su meta."""
    for cat in sorted(CATEGORIAS_CONOCIDAS):
        if cat == 'OTROS':
            continue
        dir_hermana = os.path.join(LIMPIADOS_DIR, cat, nombre)
        if not os.path.isdir(dir_hermana):
            continue
        for meta_nombre in os.listdir(dir_hermana):
            if not meta_nombre.endswith('.meta.json'):
                continue
            try:
                with open(os.path.join(dir_hermana, meta_nombre),
                          encoding='utf-8') as f:
                    meta = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            categoria = (meta.get('categoria') or '').upper()
            if categoria in CATEGORIAS_CONOCIDAS and categoria != 'OTROS':
                return categoria
    return None


def _resolver_categoria(asset_dir, nombre):
    """Categoria de una carpeta de activo sin clasificar, por orden de
    confianza: 'categoria' explicita de su .meta.json, 'categoria' explicita
    de una carpeta hermana, y por ultimo el 'activo' del .meta.json local
    (desplegable de la pestana Importar: grueso y a menudo sin tocar)."""
    meta = _meta_de_carpeta(asset_dir)
    if meta:
        categoria = (meta.get('categoria') or '').upper()
        if categoria in CATEGORIAS_CONOCIDAS and categoria != 'OTROS':
            return categoria
    categoria = _categoria_de_hermana(nombre)
    if categoria is not None:
        return categoria
    if meta:
        return _ACTIVO_A_CATEGORIA.get((meta.get('activo') or '').upper(), 'OTROS')
    return 'OTROS'


def _reintentar_otros():
    """OTROS es un cajon de 'no se pudo clasificar con confianza', no una
    categoria definitiva: en cada ejecucion se reintenta por si ahora hay
    un .meta.json con un 'activo' resoluble que antes no estaba."""
    otros_dir = os.path.join(LIMPIADOS_DIR, 'OTROS')
    if not os.path.isdir(otros_dir):
        return 0, 0, {}
    movidas, omitidas = 0, 0
    mapeo = {}
    for nombre in sorted(os.listdir(otros_dir)):
        ruta = os.path.join(otros_dir, nombre)
        if not os.path.isdir(ruta):
            continue
        categoria = _resolver_categoria(ruta, nombre)
        if categoria == 'OTROS':
            omitidas += 1
            continue
        destino = os.path.join(LIMPIADOS_DIR, categoria, nombre)
        log(f"  Limpiados/OTROS/{nombre}  ->  Limpiados/{categoria}/{nombre}")
        try:
            mover_carpeta(ruta, destino)
            movidas += 1
            mapeo[os.path.normpath(ruta)] = os.path.normpath(destino)
        except OSError as e:
            log(f"  ERROR moviendo {ruta}: {e}")
    return movidas, omitidas, mapeo


def _reorganizar():
    if not os.path.isdir(LIMPIADOS_DIR):
        return 0, 0, {}
    movidas, omitidas, mapeo = _reintentar_otros()
    for nombre in sorted(os.listdir(LIMPIADOS_DIR)):
        ruta = os.path.join(LIMPIADOS_DIR, nombre)
        if not os.path.isdir(ruta):
            continue
        if nombre in _ENTRADAS_NO_ACTIVO or nombre.upper() in CATEGORIAS_CONOCIDAS:
            omitidas += 1
            continue
        categoria = _resolver_categoria(ruta, nombre)
        destino = os.path.join(LIMPIADOS_DIR, categoria, nombre)
        log(f"  Limpiados/{nombre}  ->  Limpiados/{categoria}/{nombre}")
        try:
            mover_carpeta(ruta, destino)
            movidas += 1
            mapeo[os.path.normpath(ruta)] = os.path.normpath(destino)
        except OSError as e:
            log(f"  ERROR moviendo {ruta}: {e}")
    return movidas, omitidas, mapeo


def _reescribir_ruta(path, mapeo):
    """Si path cuelga de alguna carpeta que se movio, devuelve la ruta
    equivalente bajo la carpeta nueva. Si no, devuelve path tal cual."""
    if not path:
        return path
    norm = os.path.normpath(path)
    for vieja, nueva in mapeo.items():
        if norm == vieja or norm.startswith(vieja + os.sep):
            return nueva + norm[len(vieja):]
    return path


def _reparar_favoritos(mapeo):
    if not mapeo or not os.path.isdir(FAVORITOS_DIR):
        return 0
    reparados = 0
    for entry in os.scandir(FAVORITOS_DIR):
        if not entry.is_dir():
            continue
        ruta_json = os.path.join(entry.path, 'favorito.json')
        try:
            with open(ruta_json, encoding='utf-8') as f:
                datos = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        csv_actual = datos.get('csv')
        csv_nuevo = _reescribir_ruta(csv_actual, mapeo)
        if csv_nuevo != csv_actual and os.path.exists(csv_nuevo):
            datos['csv'] = csv_nuevo
            try:
                with open(ruta_json, 'w', encoding='utf-8') as f:
                    json.dump(datos, f, indent=2, ensure_ascii=False)
                log(f"  Favorito «{datos.get('nombre', entry.name)}»: ruta actualizada")
                reparados += 1
            except OSError as e:
                log(f"  ERROR reescribiendo {ruta_json}: {e}")
    return reparados


def _reparar_comparador(mapeo):
    if not mapeo:
        return False
    config_path = os.path.join(LIMPIADOS_DIR, 'comparador_config.json')
    if not os.path.exists(config_path):
        return False
    try:
        with open(config_path, encoding='utf-8') as f:
            cfg = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False

    cambiado = False

    def _reparar_lista(activos):
        nonlocal cambiado
        for a in activos:
            nuevo = _reescribir_ruta(a.get('path'), mapeo)
            if nuevo != a.get('path'):
                a['path'] = nuevo
                cambiado = True

    _reparar_lista(cfg.get('activos', []))
    for lista in cfg.get('listas', []):
        _reparar_lista(lista.get('activos', []))
    bench = cfg.get('benchmark')
    nuevo_bench = _reescribir_ruta(bench, mapeo)
    if nuevo_bench != bench:
        cfg['benchmark'] = nuevo_bench
        cambiado = True

    if not cambiado:
        return False
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        log("  comparador_config.json: rutas actualizadas")
    except OSError as e:
        log(f"  ERROR reescribiendo {config_path}: {e}")
        return False
    return True


def main():
    log("=" * 60)
    log("REORGANIZACION DE CARPETAS POR CATEGORIA (limpiados)")
    log("=" * 60)

    movidas, omitidas, mapeo = _reorganizar()

    reparados_fav = _reparar_favoritos(mapeo) if mapeo else 0
    reparado_comp = _reparar_comparador(mapeo) if mapeo else False

    log("")
    log("=" * 60)
    log(f"COMPLETADO: {movidas} carpetas movidas, {omitidas} ya organizadas, "
        f"{reparados_fav} favoritos reparados, "
        f"comparador {'reparado' if reparado_comp else 'sin cambios'}")
    log("=" * 60)


if __name__ == '__main__':
    main()
