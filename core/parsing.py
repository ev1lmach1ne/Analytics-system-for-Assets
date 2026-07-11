"""
core/parsing.py
Parseo tolerante de valores numéricos "sucios" (formato europeo/americano,
sufijos K/M/B, espacios) usado por library/scripts_utiles/preparar_datos.py.
"""
import re
import numpy as np
import pandas as pd

SUFIJOS = {'K': 1_000, 'M': 1_000_000, 'B': 1_000_000_000}

_NULL_TOKENS = re.compile(r'^\s*(nan|null|none)?\s*$', re.IGNORECASE)
_SUFIX_RE = re.compile(r'^([+-]?[\d.]+)\s*([KkMmBb])$')


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


def parse_columna_flexible(col: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    Versión vectorizada de parse_numero_flexible para una columna entera.

    Devuelve (valores, motivos) como dos pd.Series alineadas al índice de col.
    Equivalente a col.apply(parse_numero_flexible) pero ~100x más rápido
    porque evita el bucle Python fila por fila.
    """
    n = len(col)
    valores = pd.Series(0.0, index=col.index, dtype=float)
    motivos = pd.Series('vacio', index=col.index, dtype=object)

    s = col.astype(str).str.strip().str.replace(' ', '', regex=False)
    null_mask = col.isna() | s.str.match(_NULL_TOKENS) | (s == '')
    valores[null_mask] = 0.0
    motivos[null_mask] = 'vacio'

    active = ~null_mask
    if not active.any():
        return valores, motivos

    sa = s[active]

    # Formato europeo: punto=miles, coma=decimal → quitar puntos, coma→punto
    eu_mask = sa.str.contains(',') & sa.str.contains(r'\.')
    eu_comma_last = eu_mask & (sa.str.rfind(',') > sa.str.rfind('.'))
    sa_eu = sa[eu_comma_last].str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
    sa_us = sa[eu_mask & ~eu_comma_last].str.replace(',', '', regex=False)
    sa[eu_comma_last] = sa_eu
    sa[eu_mask & ~eu_comma_last] = sa_us

    # Coma decimal simple: "1,23" o "1,23K" (≤2 dígitos después de coma)
    only_comma = ~eu_mask & sa.str.contains(',')
    for idx in sa[only_comma].index:
        parts = sa[idx].split(',')
        if len(parts) == 2:
            last_clean = parts[1].rstrip('KkMmBb')
            if last_clean.isdigit() and len(last_clean) <= 2:
                sa[idx] = sa[idx].replace(',', '.')

    # Intentar conversión directa
    parsed = pd.to_numeric(sa, errors='coerce')
    ok_mask = parsed.notna()
    valores.loc[sa.index[ok_mask]] = parsed[ok_mask].values
    motivos.loc[sa.index[ok_mask]] = 'ok'

    # Sufijos K/M/B para los que fallaron
    fail_mask = ~ok_mask
    if fail_mask.any():
        sf = sa[fail_mask]
        m = sf.str.extract(_SUFIX_RE)
        has_sufix = m[1].notna()
        for idx in sf.index[has_sufix]:
            try:
                mult = SUFIJOS[m.loc[idx, 1].upper()]
                valores[idx] = float(m.loc[idx, 0]) * mult
                motivos[idx] = 'ok'
            except (ValueError, KeyError):
                motivos[idx] = 'fallo'
        still_fail = ~has_sufix
        motivos.loc[sf.index[still_fail]] = 'fallo'

    return valores, motivos
