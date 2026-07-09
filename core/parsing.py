"""
core/parsing.py
Parseo tolerante de valores numéricos "sucios" (formato europeo/americano,
sufijos K/M/B, espacios) usado por library/scripts_utiles/preparar_datos.py.
"""
import pandas as pd

SUFIJOS = {'K': 1_000, 'M': 1_000_000, 'B': 1_000_000_000}


def parse_numero_flexible(v):
    """
    Parsea un valor "sucio" a float, tolerando entradas vacías o corruptas.

    Devuelve (valor, motivo):
        valor  : número parseado, o 0.0 si no se pudo convertir.
        motivo : 'ok'    — se parseó correctamente
                 'vacio' — entrada NaN/vacía/'null'/'none'
                 'fallo' — había contenido pero no se pudo convertir a float
    """
    if pd.isna(v):
        return 0.0, 'vacio'
    s = str(v).strip().replace(' ', '')
    if not s or s.lower() in ('nan', 'null', 'none', ''):
        return 0.0, 'vacio'
    # Detectar formato europeo (punto= miles, coma=decimal)
    if ',' in s and '.' in s:
        if s.rfind(',') > s.rfind('.'):
            s = s.replace('.', '')
            s = s.replace(',', '.')
        else:
            s = s.replace(',', '')
    elif ',' in s:
        parts = s.split(',')
        if len(parts) == 2:
            last_clean = parts[1].rstrip('KkMmBb')
            if last_clean.isdigit() and len(last_clean) <= 2:
                s = s.replace(',', '.')
    suf = s[-1].upper()
    if suf in SUFIJOS:
        try:
            return float(s[:-1]) * SUFIJOS[suf], 'ok'
        except ValueError:
            return 0.0, 'fallo'
    try:
        return float(s), 'ok'
    except ValueError:
        return 0.0, 'fallo'
