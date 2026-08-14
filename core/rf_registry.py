"""Registro ligero de la tasa libre de riesgo por archivo limpio.

El .meta.json de cada CSV limpio ya guarda el Rf, pero los archivos
importados antes de esa característica no lo tienen, y una importación que
no termina tampoco deja meta nuevo. Este registro es el plan B: Importar
anota aquí el Rf de cada archivo (también si la limpieza falla) y el
Limpiador lo consulta cuando el meta no declara el Rf.
"""
import json
import os


def _ruta_registro(limpiados_dir):
    return os.path.join(limpiados_dir, '.rf_registry.json')


def _clave(limpiados_dir, csv_path):
    try:
        return os.path.relpath(csv_path, limpiados_dir).replace('\\', '/')
    except ValueError:
        return csv_path


def leer_rf(limpiados_dir, csv_path):
    """Rf guardado para `csv_path`, o None si no hay ninguno."""
    if not limpiados_dir or not csv_path:
        return None
    try:
        with open(_ruta_registro(limpiados_dir), encoding='utf-8') as f:
            datos = json.load(f) or {}
        return datos.get(_clave(limpiados_dir, csv_path))
    except Exception:
        return None


def guardar_rf(limpiados_dir, csv_path, rf):
    """Anota el Rf de `csv_path`. Fallos de escritura se ignoran: es un
    registro auxiliar, nunca debe romper la importación."""
    if rf is None or not limpiados_dir or not csv_path:
        return
    try:
        ruta = _ruta_registro(limpiados_dir)
        datos = {}
        if os.path.exists(ruta):
            with open(ruta, encoding='utf-8') as f:
                datos = json.load(f) or {}
        datos[_clave(limpiados_dir, csv_path)] = rf
        os.makedirs(os.path.dirname(ruta), exist_ok=True)
        with open(ruta, 'w', encoding='utf-8') as f:
            json.dump(datos, f, indent=1, ensure_ascii=False)
    except Exception:
        pass
