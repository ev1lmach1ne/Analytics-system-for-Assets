"""Clasificación de errores y consultas auxiliares contra QuestDB.

El wire protocol PostgreSQL de QuestDB es un subconjunto y no garantiza
SQLSTATE estándar (p.ej. una tabla inexistente no siempre llega como
'42P01'), así que la clasificación combina código y texto del mensaje, y
las comprobaciones de existencia se hacen por la vía HTTP /exec, que es la
nativa de QuestDB y la misma que usa la subida de datos.
"""
import requests


def pgcode(error):
    """Devuelve el código PostgreSQL aunque pandas lo envuelva."""
    actual = error
    for _ in range(3):
        code = getattr(actual, 'pgcode', None)
        if code:
            return code
        actual = getattr(actual, 'orig', None)
        if actual is None:
            break
    return None


def _mensaje(error):
    """Primer argumento de texto de la excepción, siguiendo la cadena orig."""
    actual = error
    for _ in range(3):
        args = getattr(actual, 'args', None)
        if args and isinstance(args[0], str) and args[0]:
            return args[0]
        actual = getattr(actual, 'orig', None)
        if actual is None:
            break
    return ''


def es_tabla_inexistente(error):
    if pgcode(error) == '42P01':
        return True
    texto = _mensaje(error).lower()
    return ('does not exist' in texto or 'no existe' in texto
            or 'no such table' in texto or 'table not found' in texto)


def es_columna_inexistente(error):
    if pgcode(error) == '42703':
        return True
    texto = _mensaje(error).lower()
    return ('column' in texto and ('does not exist' in texto
                                   or 'no existe' in texto
                                   or 'not found' in texto
                                   or 'invalid column' in texto))


def filas_en_tabla(host, port, tabla, timeout=15):
    """Nº de filas de `tabla` en QuestDB vía HTTP /exec.

    Devuelve 0 si la tabla no existe (respuesta de error con «table does
    not exist»). Cualquier fallo de conexión o respuesta inesperada se lanza
    como RuntimeError: nunca se interpreta como «tabla vacía», porque eso
    escondería a QuestDB caída o credenciales inválidas.
    """
    url = f"http://{host}:{port}/exec"
    try:
        resp = requests.get(
            url, params={'query': f'SELECT count() FROM {tabla}'},
            timeout=timeout)
    except requests.exceptions.RequestException as e:
        raise RuntimeError(
            f"No se pudo consultar QuestDB ({host}:{port}): {e}") from e
    if resp.status_code == 200:
        try:
            datos = resp.json()
            return int(datos['dataset'][0][0])
        except (ValueError, KeyError, IndexError, TypeError) as e:
            raise RuntimeError(
                f"Respuesta inesperada de QuestDB: {resp.text[:300]}") from e
    texto = (resp.text or '').strip()
    if 'does not exist' in texto.lower() or 'no existe' in texto.lower():
        return 0
    raise RuntimeError(
        f"QuestDB respondió HTTP {resp.status_code}: {texto[:300]}")
